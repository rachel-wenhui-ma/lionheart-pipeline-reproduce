# run_pipeline_all_chroms.py
"""
Run feature extraction pipeline for all 22 autosomes (chr1-chr22).
Accumulates statistics across all chromosomes (matching LIONHEART's approach).
"""
import sys
import os
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extract_insert_size import extract_fragment_lengths_per_bin
from src.compute_coverage import compute_coverage, normalize_megabins_simple
from src.compute_correlation import compute_all_features
from src.assemble_features import write_feature_vector
from src.utils import calculate_gc_content_per_bin, load_lionheart_gc_content
from src.outlier_detection import find_clipping_threshold, clip_outliers
from src.load_masks import load_cell_type_masks, load_cell_type_list
from src.running_stats import RunningPearsonR
try:
    from lionheart.features.running_pearson_r import (
        RunningPearsonR as OfficialRunningPearsonR,
    )
except ImportError:
    OfficialRunningPearsonR = None
from lionheart.features.running_pearson_r import RunningPearsonR as OfficialRunningPearsonR
from src.blacklist import load_exclude_bins, apply_exclude_to_coverage
from src.correction_gc import (
    calculate_correction_factors,
)
from src.correction_helpers import (
    correct_bias,
)
from src.correction_insert_size import (
    calculate_insert_size_correction_factors,
)
from pathlib import Path
import pysam
import numpy as np
from scipy import special
import scipy.sparse
import json
from typing import Dict
from collections import OrderedDict, defaultdict
from multiprocessing import cpu_count


def load_sparse_vector(path: Path, indices=None, dtype=np.float64, decimals=7):
    s = scipy.sparse.load_npz(path).tocsr()
    if indices is not None:
        s = s[indices, :]
    arr = s.astype(dtype, copy=False).toarray().ravel()
    if decimals is not None and decimals >= 0:
        arr = np.round(arr, decimals=decimals)
    return arr


# -----------------------------
# Config
# -----------------------------
# All 22 autosomes (chr1-chr22)
all_chroms = [f"chr{i}" for i in range(1, 23)]
run_chroms_env = os.environ.get("RUN_CHROMS")
if run_chroms_env:
    requested_chroms = [
        chrom.strip()
        for chrom in run_chroms_env.split(",")
        if chrom.strip()
    ]
    if requested_chroms:
        all_chroms = requested_chroms
        print(f"RUN_CHROMS override active: processing {len(all_chroms)} chromosome(s): {', '.join(all_chroms)}")
bin_size = 10  # LIONHEART works at 10bp level (not 100bp aggregated)

# Configuration for real LIONHEART masks
from src.paths import get_resources_dir, get_reference_dir, get_bam_path

# Get paths using unified path configuration
resources_dir = get_resources_dir()
reference_dir = get_reference_dir()
bam_path = get_bam_path()
gc_correction_bin_edges = np.load(resources_dir / "gc_contents_bin_edges.npy")
insert_size_correction_bin_edges = np.load(resources_dir / "insert_size_bin_edges.npy")

insert_size_sparse_dir = os.environ.get("INSERT_SIZE_SPARSE_DIR")
if insert_size_sparse_dir:
    insert_size_sparse_dir = Path(insert_size_sparse_dir)
elif (resources_dir / "insert_sizes_sparse").exists():
    insert_size_sparse_dir = resources_dir / "insert_sizes_sparse"
else:
    insert_size_sparse_dir = None

coverage_sparse_dir = os.environ.get("COVERAGE_SPARSE_DIR")
if coverage_sparse_dir:
    coverage_sparse_dir = Path(coverage_sparse_dir)
elif (resources_dir / "sparse_coverage_by_chromosome").exists():
    coverage_sparse_dir = resources_dir / "sparse_coverage_by_chromosome"
else:
    coverage_sparse_dir = None
using_sparse_coverage = coverage_sparse_dir is not None

use_real_masks = True
resources_dir = get_resources_dir()
reference_dir = get_reference_dir()
bam_path = get_bam_path()

# Optional debug dumping: set DEBUG_DUMP_CHROM & DEBUG_DUMP_CELL_TYPE
debug_dump_chrom = os.environ.get("DEBUG_DUMP_CHROM")
debug_dump_cell_type = os.environ.get("DEBUG_DUMP_CELL_TYPE")
debug_dump_dir = Path(os.environ.get("DEBUG_DUMP_DIR", "debug_dumps"))
debug_dump_enabled = bool(debug_dump_chrom and debug_dump_cell_type)
if debug_dump_enabled:
    print(
        f"[DEBUG] Will dump arrays for cell_type={debug_dump_cell_type} "
        f"on {debug_dump_chrom} into {debug_dump_dir}"
    )

def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean-like environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


trace_consensus_env = os.environ.get("TRACE_CONSENSUS_REMOVAL", "")
trace_consensus_mask_types = {
    entry.strip()
    for entry in trace_consensus_env.split(",")
    if entry.strip()
}
trace_consensus_all = "*" in trace_consensus_mask_types
keep_consensus_bins = _env_bool("KEEP_CONSENSUS_BINS", default=False)
consensus_removal_totals = defaultdict(
    lambda: {"x_sum": 0.0, "x_squared_sum": 0.0, "bins": 0}
)

def _env_int(name: str, default: int) -> int:
    """Read integer from environment with fallback."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


mask_type_configs: Dict[str, Dict] = OrderedDict()
mask_types_to_process = []
total_mask_cell_types = 0
mask_type_worker_overrides: Dict[str, int] = {}
compare_official_running = _env_bool("COMPARE_OFFICIAL_RUNNING_STATS", default=False)
if compare_official_running and OfficialRunningPearsonR is None:
    print(
        "Warning: lionheart RunningPearsonR import failed; "
        "disabling COMPARE_OFFICIAL_RUNNING_STATS"
    )
    compare_official_running = False


mask_types_env = [
    mt.strip()
    for mt in os.environ.get("MASK_TYPES", "").split(",")
    if mt.strip()
]
mask_types_filter = {mt for mt in mask_types_env} if mask_types_env else None

if use_real_masks and resources_dir.exists():
    print(f"Using real LIONHEART masks from: {resources_dir}")
    possible_mask_types = ["DNase", "ATAC"]
    generic_cell_type_filter = os.environ.get("CELL_TYPE_FILTER")
    for mask_type in possible_mask_types:
        if mask_types_filter and mask_type not in mask_types_filter:
            print(f"  Skipping {mask_type} due to MASK_TYPES filter")
            continue

        mask_base_dir = (
            resources_dir
            / "chromatin_masks"
            / mask_type
            / "sparse_overlaps_by_chromosome"
        )
        if not mask_base_dir.exists():
            print(f"  Warning: Mask directory not found for {mask_type}: {mask_base_dir}")
            continue

        cell_type_df = load_cell_type_list(resources_dir, mask_type)
        original_cell_types = cell_type_df["cell_type"].tolist()
        if "idx" in cell_type_df.columns:
            cell_type_to_idx = dict(zip(cell_type_df["cell_type"], cell_type_df["idx"]))
        else:
            cell_type_to_idx = {ct: i for i, ct in enumerate(original_cell_types)}
        cell_types = original_cell_types.copy()

        env_filter_name = f"{mask_type.upper()}_CELL_TYPE_FILTER"
        cell_type_filter = os.environ.get(env_filter_name)
        if mask_type == "DNase" and not cell_type_filter:
            cell_type_filter = generic_cell_type_filter

        if cell_type_filter:
            requested = [ct.strip() for ct in cell_type_filter.split(",") if ct.strip()]
            filtered = [ct for ct in cell_types if ct in requested]
            missing = [ct for ct in requested if ct not in filtered]
            if filtered:
                cell_types = filtered
                mask_indices = [
                    int(cell_type_df.loc[cell_type_df["cell_type"] == ct, "idx"].iloc[0])
                    for ct in filtered
                ]
                print(
                    f"  {mask_type} CELL_TYPE_FILTER active: {len(cell_types)} cell type(s) will be processed"
                )
                if missing:
                    print(
                        f"    Warning: {len(missing)} requested {mask_type} cell type(s) not found: {', '.join(missing)}"
                    )
            else:
                print(
                    f"  Warning: {mask_type} CELL_TYPE_FILTER specified but none matched. Processing zero {mask_type} cell types."
                )
                cell_types = []
                mask_indices = []

        consensus_present = "consensus" in cell_types
        cell_types = [ct for ct in cell_types if ct != "consensus"]
        mask_indices = [cell_type_to_idx[ct] for ct in cell_types if ct in cell_type_to_idx]
        output_order = []
        allowed = set(cell_types)
        for ct in original_cell_types:
            if ct == "consensus" and consensus_present:
                output_order.append(ct)
            elif ct in allowed:
                output_order.append(ct)

        if not cell_types and not consensus_present:
            print(f"  Warning: No cell types available for {mask_type}, skipping.")
            continue

        mask_type_configs[mask_type] = {
            "cell_types": cell_types,
            "cell_type_df": cell_type_df,
            "indices": mask_indices,
            "mask_dir": mask_base_dir,
            "consensus_dirs": [
                mask_base_dir / "consensus",
                resources_dir.parent
                / "inference_resources_v003"
                / "chromatin_masks"
                / mask_type
                / "sparse_overlaps_by_chromosome"
                / "consensus",
            ],
            "output_order": output_order,
            "has_consensus": consensus_present,
        }
        total_mask_cell_types += len(cell_types)
        mask_types_to_process.append(mask_type)
        mask_type_worker_overrides[mask_type] = _env_int(
            f"{mask_type.upper()}_MASK_WORKERS",
            1 if mask_type == "ATAC" else 4,
        )
        print(f"  Loaded {len(cell_types)} {mask_type} cell types")
else:
    print("Warning: Real masks not available, using test masks")

if not mask_types_to_process:
    print("ERROR: No mask types available to process. Exiting.")
    sys.exit(1)
else:
    print(
        f"Will process mask types in order: {', '.join(mask_types_to_process)} "
        f"({total_mask_cell_types} total cell types, {total_mask_cell_types * 10} features)"
    )

# Blacklist/exclude paths (LIONHEART uses outlier and zero coverage indices)
exclude_paths = []
# LIONHEART exclude files are in resources/outliers/
outliers_dir = resources_dir / "outliers"

# LIONHEART uses two exclude files:
# 1. outlier_indices.npz - bins with extreme outlier values
# 2. zero_coverage_indices.npz - bins with zero coverage
exclude_files = [
    outliers_dir / "outlier_indices.npz",
    outliers_dir / "zero_coverage_indices.npz",
]

for path in exclude_files:
    if path.exists():
        exclude_paths.append(path)
        print(f"Found exclude file: {path.name}")
    else:
        print(f"Warning: Exclude file not found: {path}")

# Load exclude bins for all chromosomes
exclude_bins_by_chrom = load_exclude_bins(exclude_paths, all_chroms)
total_excluded = sum(len(excl) for excl in exclude_bins_by_chrom.values())
if total_excluded > 0:
    print(f"Loaded exclude bins: {total_excluded:,} total bins to exclude across all chromosomes")
else:
    print("No exclude bins loaded (blacklist removal will be skipped)")

output_path = "features_all_22chroms_10bp.csv"  # Results at 10bp level (matching LIONHEART)

# -----------------------------
# Initialize RunningPearsonR for each cell type
# -----------------------------
print("\n" + "=" * 70)
print("Initializing feature accumulators for all mask types...")
print("=" * 70)

mask_type_calculators: Dict[str, OrderedDict] = OrderedDict()
mask_type_official_calculators: Dict[str, OrderedDict] = OrderedDict()
consensus_calculators: Dict[str, RunningPearsonR] = {}
consensus_official_calculators: Dict[str, OfficialRunningPearsonR] = {}
total_calculators = 0
for mask_type in mask_types_to_process:
    calculators = OrderedDict()
    official_calculators = OrderedDict()
    for cell_type in mask_type_configs[mask_type]["cell_types"]:
        calculators[cell_type] = RunningPearsonR(
            ignore_nans=True, name=f"{mask_type}::{cell_type}"
        )
        if compare_official_running:
            official_calculators[cell_type] = OfficialRunningPearsonR(ignore_nans=True)
        total_calculators += 1
    mask_type_calculators[mask_type] = calculators
    if compare_official_running:
        mask_type_official_calculators[mask_type] = official_calculators
    print(f"  {mask_type}: {len(calculators)} RunningPearsonR calculators initialized")

print(f"Total calculators: {total_calculators}")

# -----------------------------
# Process each chromosome
# -----------------------------
print("\n" + "=" * 70)
print(f"Processing all {len(all_chroms)} autosomes...")
print("=" * 70)

for chrom_idx, chrom in enumerate(all_chroms, 1):
    print(f"\n{'='*70}")
    print(f"Processing {chrom} ({chrom_idx}/{len(all_chroms)})")
    print(f"{'='*70}")
    
    # Check if reference FASTA exists for this chromosome
    reference_fasta = reference_dir / f"{chrom}.fa"
    if not reference_fasta.exists():
        print(f"  Warning: Reference FASTA not found: {reference_fasta}")
        print(f"  Skipping {chrom}. Run: python download_reference.py all")
        continue
    
    # Open BAM file
    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        if chrom not in bam.references:
            print(f"  Warning: {chrom} not found in BAM file, skipping")
            continue
        
        chrom_len = bam.get_reference_length(chrom)
        print(f"  Chromosome length: {chrom_len:,} bp")
        print(f"  Number of bins: {chrom_len // bin_size + 1:,}")
    finally:
        bam.close()
    
    # -----------------------------
    # 1. Extract Fragment Lengths
    # -----------------------------
    print(f"\n  Step 1: Extracting fragment lengths for {chrom}...")
    mean_frag_len, median_frag_len = extract_fragment_lengths_per_bin(
        bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True
    )
    
    # -----------------------------
    # 2. Compute Coverage
    # -----------------------------
    if not using_sparse_coverage:
        print(f"  Step 2: Computing coverage for {chrom}...")
        coverage = compute_coverage(bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True)
    else:
        coverage = None
    
    # -----------------------------
    # 3. Load GC Content (using LIONHEART's GC values from parquet file)
    # -----------------------------
    print(f"  Step 3: Loading GC content for {chrom} (from LIONHEART parquet file)...")
    exclude_indices_10bp = exclude_bins_by_chrom.get(chrom, np.array([], dtype=np.int64))

    # Load LIONHEART's GC values (matching LIONHEART's approach)
    # At 10bp level, use LIONHEART's _load_bins_and_exclude directly
    if resources_dir.exists():
        try:
            from lionheart.features.create_dataset_inference import _load_bins_and_exclude
            bins_path = resources_dir / "bin_indices_by_chromosome" / f"{chrom}.parquet"
            if bins_path.exists():
                include_indices_10bp, gc_content = _load_bins_and_exclude(
                    bins_path=bins_path,
                    exclude=exclude_indices_10bp if len(exclude_indices_10bp) > 0 else None,
                )
                print(f"    Loaded GC from LIONHEART parquet (10bp): {len(gc_content):,} bins")
                print(f"    GC range: {gc_content.min():.4f} to {gc_content.max():.4f}, mean: {gc_content.mean():.4f}")
                use_lionheart_gc = True
            else:
                raise FileNotFoundError(f"Bins file not found: {bins_path}")
        except Exception as e:
            print(f"    Warning: Failed to load LIONHEART GC ({e}), falling back to calculated GC")
            gc_content = calculate_gc_content_per_bin(str(reference_fasta), chrom, bin_size=bin_size)
            include_indices_10bp = None
            use_lionheart_gc = False
    else:
        print(f"    Warning: Resources directory not found, calculating GC from reference")
        gc_content = calculate_gc_content_per_bin(str(reference_fasta), chrom, bin_size=bin_size)
        include_indices_10bp = None
        use_lionheart_gc = False

    # -----------------------------
    # 4. Filter coverage to match GC's include_indices (if using LIONHEART GC)
    # -----------------------------
    if use_lionheart_gc and include_indices_10bp is not None:
            # Filter coverage and fragment lengths to match GC's include_indices (10bp level)
            # This ensures all arrays have the same length for subsequent corrections
            print(f"  Step 4a: Filtering coverage to match GC's include_indices for {chrom}...")
            if coverage is not None:
                print(f"    Original coverage bins: {len(coverage):,}")
                if len(include_indices_10bp) <= len(coverage):
                    coverage = coverage[include_indices_10bp]
                    mean_frag_len = mean_frag_len[include_indices_10bp]
                else:
                    print(f"    Warning: include_indices_10bp ({len(include_indices_10bp)}) > coverage ({len(coverage)})")
                    # Fallback: clip include_indices to valid range
                    valid_indices = include_indices_10bp[include_indices_10bp < len(coverage)]
                    coverage = coverage[valid_indices]
                    mean_frag_len = mean_frag_len[valid_indices]
                    gc_content = gc_content[:len(valid_indices)]  # Also clip GC to match
                    include_indices_10bp = valid_indices
            else:
                print("    Loading coverage from sparse arrays")
                print(f"    GC bins (after exclude, 10bp): {len(gc_content):,}")
                if using_sparse_coverage:
                    sparse_cov_file = coverage_sparse_dir / f"{chrom}.npz"
                    if sparse_cov_file.exists():
                        coverage = load_sparse_vector(
                            sparse_cov_file,
                            indices=include_indices_10bp,
                            dtype=np.float64,
                            decimals=2,
                        )
                    else:
                        print(f"    Warning: sparse coverage file not found: {sparse_cov_file}. Falling back to BAM coverage.")
                        coverage = compute_coverage(bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True)
                        coverage = coverage[include_indices_10bp]
                else:
                    coverage = compute_coverage(bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True)
                    coverage = coverage[include_indices_10bp]
            print(f"    Filtered coverage bins: {len(coverage):,}")
            if insert_size_sparse_dir is not None:
                sparse_file = insert_size_sparse_dir / f"{chrom}.npz"
                if sparse_file.exists():
                    print("    Loading insert sizes from sparse array")
                    mean_frag_len = load_sparse_vector(
                        sparse_file,
                        indices=include_indices_10bp,
                        dtype=np.float64,
                        decimals=1,
                    )
                    positive_mask = coverage > 0
                    mean_frag_len[positive_mask] /= coverage[positive_mask]
                    mean_frag_len = np.round(mean_frag_len, decimals=7)
                    if debug_dump_enabled and chrom == debug_dump_chrom:
                        np.save(debug_dump_dir / f"{chrom}_insert_sizes_mean.npy", mean_frag_len.astype(np.float64))
                else:
                    print(f"    Warning: Insert size sparse file not found: {sparse_file}. Using extracted fragment lengths.")
    else:
        # Not using LIONHEART GC, ensure coverage is loaded
        if coverage is None:
            if using_sparse_coverage:
                sparse_cov_file = coverage_sparse_dir / f"{chrom}.npz"
                if sparse_cov_file.exists():
                    coverage = load_sparse_vector(
                        sparse_cov_file,
                        indices=None,
                        dtype=np.float64,
                        decimals=2,
                    )
                else:
                    print(f"    Warning: sparse coverage file not found: {sparse_cov_file}. Computing from BAM.")
                    coverage = compute_coverage(bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True)
            else:
                coverage = compute_coverage(bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True)
    
    coverage_raw_counts = coverage.copy()
    if debug_dump_enabled and chrom == debug_dump_chrom:
        debug_dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(debug_dump_dir / f"{chrom}_raw_sample_cov.npy", coverage.astype(np.float64))

    # -----------------------------
    # 5. Outlier Detection (ZIPoisson)
    # -----------------------------
    print(f"  Step 5: Detecting outliers with ZIPoisson for {chrom}...")
    clipping_val = find_clipping_threshold(coverage)
    coverage_clipped = clip_outliers(coverage, clipping_val)
    coverage_raw_counts[coverage_raw_counts > clipping_val] = np.nan
    if debug_dump_enabled and chrom == debug_dump_chrom:
        np.save(debug_dump_dir / f"{chrom}_clip_sample_cov.npy", coverage_clipped.astype(np.float64))

    # -----------------------------
    # 6. GC Correction
    # -----------------------------
    print(f"  Step 6: Applying GC correction for {chrom}...")
    gc_midpoints, gc_factors = calculate_correction_factors(
        bias_scores=gc_content,
        coverages=coverage_clipped,
        bin_edges=gc_correction_bin_edges,
    )
    coverage_gc_corrected = correct_bias(
        coverages=coverage_clipped,
        correct_factors=gc_factors,
        bias_scores=gc_content,
        bin_edges=gc_correction_bin_edges,
    )
    if debug_dump_enabled and chrom == debug_dump_chrom:
        np.save(debug_dump_dir / f"{chrom}_gc_corrected_sample_cov.npy", coverage_gc_corrected.astype(np.float64))

    # -----------------------------
    # 7. Insert Size Correction
    # -----------------------------
    print(f"  Step 7: Applying insert size correction for {chrom}...")

    insert_size_factors = calculate_insert_size_correction_factors(
        coverages=coverage_raw_counts,
        insert_sizes=mean_frag_len,
        bin_edges=insert_size_correction_bin_edges,
        base_sigma=8.026649608460776,
        df=5,
        nan_extremes=True,
    )

    # Apply three corrections sequentially
    coverage_after_noise = coverage_gc_corrected.copy()
    valid_mask = mean_frag_len > 0
    if np.any(valid_mask):
        # 1. Noise correction
        coverage_after_noise[valid_mask] = correct_bias(
            coverages=coverage_gc_corrected[valid_mask],
            correct_factors=insert_size_factors["noise_correction_factor"],
            bias_scores=mean_frag_len[valid_mask],
            bin_edges=insert_size_correction_bin_edges,
        )
        if debug_dump_enabled and chrom == debug_dump_chrom:
            np.save(debug_dump_dir / f"{chrom}_insert_noise_corrected_cov.npy", coverage_after_noise.astype(np.float64))
    
        # 2. Skewness correction
        coverage_after_skew = coverage_after_noise.copy()
        coverage_after_skew[valid_mask] = correct_bias(
            coverages=coverage_after_noise[valid_mask],
            correct_factors=insert_size_factors["skewness_correction_factor"],
            bias_scores=mean_frag_len[valid_mask],
            bin_edges=insert_size_correction_bin_edges,
        )
        if debug_dump_enabled and chrom == debug_dump_chrom:
            np.save(debug_dump_dir / f"{chrom}_insert_skew_corrected_cov.npy", coverage_after_skew.astype(np.float64))
    
        # 3. Mean shift correction
        coverage_insert_corrected = coverage_after_skew.copy()
        coverage_insert_corrected[valid_mask] = correct_bias(
            coverages=coverage_after_skew[valid_mask],
            correct_factors=insert_size_factors["mean_correction_factor"],
            bias_scores=mean_frag_len[valid_mask],
            bin_edges=insert_size_correction_bin_edges,
        )
        if debug_dump_enabled and chrom == debug_dump_chrom:
            np.save(debug_dump_dir / f"{chrom}_insert_mean_corrected_cov.npy", coverage_insert_corrected.astype(np.float64))
    else:
        coverage_insert_corrected = coverage_gc_corrected.copy()

    # -----------------------------
    # 8. Megabin Normalization
    # -----------------------------
    print(f"  Step 8: Applying megabin normalization for {chrom}...")
    # Calculate start coordinates for each bin (LIONHEART uses actual genomic positions)
    # LIONHEART uses: include_indices * 10 (10bp-based indices converted to genomic coordinates)
    if include_indices_10bp is not None:
        start_coordinates = include_indices_10bp * bin_size  # bin_size=10, so include_indices_10bp * 10
    else:
        # Fallback: use sequential indices if include_indices not available
        start_coordinates = np.arange(len(coverage_insert_corrected)) * bin_size
    coverage_megabin_norm = normalize_megabins_simple(
        coverage_insert_corrected,
        bin_size=bin_size,
        mbin_size=5_000_000,  # 5MB megabins (LIONHEART default)
        stride=500_000,       # 500KB stride (LIONHEART default)
        center=None,          # No centering (LIONHEART uses center=None)
        scale="mean",         # Scale by mean (LIONHEART default)
        start_coordinates=start_coordinates,
        clip_above_quantile=None,  # LIONHEART default: no clipping
    )

    # LIONHEART does NOT apply final z-score normalization after megabin normalization
    # Use megabin-normalized coverage directly for correlation calculation
    coverage_norm = coverage_megabin_norm
    if debug_dump_enabled and chrom == debug_dump_chrom:
        np.save(debug_dump_dir / f"{chrom}_megabin_normalized_cov.npy", coverage_norm.astype(np.float64))

    # -----------------------------
    # 9. Apply Blacklist/Exclude (if not already applied via LIONHEART GC)
    # -----------------------------
    # Note: If using LIONHEART GC, exclude is already applied via include_indices_10bp
    # So coverage_norm is already filtered. We just need to ensure valid_mask is set.
    exclude_indices = exclude_bins_by_chrom.get(chrom, np.array([], dtype=np.int64))

    if use_lionheart_gc and include_indices_10bp is not None:
        # Already filtered in Step 4a, just set valid_mask
        valid_mask = np.ones(len(coverage_norm), dtype=bool)
        print(f"  Step 9: Blacklist already applied via LIONHEART GC filtering ({len(coverage_norm):,} bins)")
    else:
        # Apply exclude if not using LIONHEART GC
        if len(exclude_indices) > 0:
            print(f"  Step 9: Applying blacklist removal for {chrom}...")
            print(f"    Excluding {len(exclude_indices):,} bins (10bp resolution)")
            # At 10bp level, exclude indices are already at 10bp
            exclude_indices_10bp = exclude_indices[exclude_indices < len(coverage_norm)]
        
            valid_mask = np.ones(len(coverage_norm), dtype=bool)
            valid_mask[exclude_indices_10bp] = False
        
            coverage_norm = coverage_norm[valid_mask]
            gc_content = gc_content[valid_mask]
            mean_frag_len = mean_frag_len[valid_mask]
        
            print(f"    Coverage after exclusion: {len(coverage_norm):,} bins (removed {len(exclude_indices_10bp):,} bins)")
        else:
            valid_mask = np.ones(len(coverage_norm), dtype=bool)
    
    # Deferred debug save for include indices (shared across mask types)
    if debug_dump_enabled and chrom == debug_dump_chrom:
        debug_dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            debug_dump_dir / f"{chrom}_include_indices.npy",
            include_indices_10bp if (use_lionheart_gc and include_indices_10bp is not None) else np.array([], dtype=np.int64),
        )

    # Preserve base arrays for per-mask processing
    base_coverage_norm = coverage_norm.copy()
    base_gc_content = gc_content.copy()
    base_mean_frag_len = mean_frag_len.copy()

    # -----------------------------
    # 11. Load Masks and Accumulate Statistics (per mask type)
    # -----------------------------
    print(f"  Step 10: Loading masks and accumulating statistics for {chrom}...")

    if use_real_masks and resources_dir.exists():
        from joblib import Parallel, delayed
        from multiprocessing import cpu_count

        include_indices_10bp_for_mask = (
            include_indices_10bp if use_lionheart_gc and include_indices_10bp is not None else None
        )

        for mask_type in mask_types_to_process:
            cell_types = mask_type_configs[mask_type]["cell_types"]
            if not cell_types:
                print(f"    [{mask_type}] Skipping: no cell types configured")
                continue

            print(f"    [{mask_type}] Processing {len(cell_types)} masks...")

            coverage_norm_mask = base_coverage_norm.copy()
            gc_content_mask = base_gc_content.copy()
            mean_frag_len_mask = base_mean_frag_len.copy()

            # Load and apply consensus mask specific to this mask type
            consensus_indices = None
            consensus_overlap = None
            for consensus_dir in mask_type_configs[mask_type]["consensus_dirs"]:
                candidate = consensus_dir / f"{chrom}.npz"
                if candidate.exists():
                    try:
                        consensus_sparse = scipy.sparse.load_npz(candidate).tocsr()
                        if include_indices_10bp_for_mask is not None:
                            if not (consensus_sparse.shape[1] == 1 and consensus_sparse.shape[0] > 1):
                                raise ValueError(f"Consensus mask had unexpected shape: {consensus_sparse.shape}")
                            consensus_sparse = consensus_sparse[include_indices_10bp_for_mask, :]
                        consensus_overlap = consensus_sparse.astype(np.float32, copy=False).toarray().ravel()
                        consensus_overlap = np.round(consensus_overlap, decimals=2)
                        if len(consensus_overlap) != len(coverage_norm_mask):
                            min_len = min(len(consensus_overlap), len(coverage_norm_mask))
                            consensus_overlap = consensus_overlap[:min_len]
                        consensus_indices = np.nonzero(consensus_overlap)[0]
                        break
                    except Exception as e:
                        print(f"      [{mask_type}] Warning: Failed to load consensus mask ({e}) from {candidate}")
                        consensus_indices = None
                # continue loop if not found

            if (
                consensus_overlap is not None
                and mask_type_configs[mask_type].get("has_consensus")
            ):
                calc = consensus_calculators.setdefault(
                    mask_type,
                    RunningPearsonR(
                        ignore_nans=True, name=f"{mask_type}::consensus"
                    ),
                )
                calc.add_data(
                    coverage_norm_mask.astype(np.float64),
                    consensus_overlap.astype(np.float64),
                )
                if compare_official_running:
                    official_calc = consensus_official_calculators.setdefault(
                        mask_type, OfficialRunningPearsonR(ignore_nans=True)
                    )
                    official_calc.add_data(
                        coverage_norm_mask.astype(np.float64),
                        consensus_overlap.astype(np.float64),
                    )
                if debug_dump_enabled and chrom == debug_dump_chrom:
                    np.save(
                        debug_dump_dir / f"{mask_type}_{chrom}_consensus_overlap.npy",
                        consensus_overlap.astype(np.float64),
                    )

            if consensus_indices is not None and len(consensus_indices) > 0:
                removed_slice = coverage_norm_mask[consensus_indices].astype(
                    np.float64, copy=False
                )
                removed_x = float(np.nansum(removed_slice))
                removed_x_sq = float(np.nansum(removed_slice * removed_slice))
                stats = consensus_removal_totals[mask_type]
                stats["x_sum"] += removed_x
                stats["x_squared_sum"] += removed_x_sq
                stats["bins"] += len(consensus_indices)

                should_trace = trace_consensus_all or (
                    trace_consensus_mask_types
                    and mask_type in trace_consensus_mask_types
                )
                if should_trace:
                    print(
                        f"      [{mask_type}] Consensus removal trace ({chrom}): "
                        f"bins={len(consensus_indices):,} "
                        f"x_sum={removed_x:.6f} "
                        f"x_squared_sum={removed_x_sq:.6f}"
                    )

                if keep_consensus_bins:
                    print(
                        f"      [{mask_type}] KEEP_CONSENSUS_BINS=1 -> skipping consensus bin deletion "
                        f"(measured bins={len(consensus_indices):,}, x_sum={removed_x:.6f})"
                    )
                else:
                    coverage_norm_mask = np.delete(
                        coverage_norm_mask, consensus_indices
                    )
                    gc_content_mask = np.delete(gc_content_mask, consensus_indices)
                    mean_frag_len_mask = np.delete(
                        mean_frag_len_mask, consensus_indices
                    )
                    print(
                        f"      [{mask_type}] Removed {len(consensus_indices):,} consensus bins "
                        f"(x_sum={removed_x:.6f}, x_squared_sum={removed_x_sq:.6f})"
                    )
            else:
                print(f"      [{mask_type}] No consensus bins removed post-correction")

            if debug_dump_enabled and chrom == debug_dump_chrom:
                suffix = f"{chrom}" if mask_type == "DNase" else f"{mask_type}_{chrom}"
                debug_dump_dir.mkdir(parents=True, exist_ok=True)
                np.save(debug_dump_dir / f"{suffix}_coverage_norm.npy", coverage_norm_mask.astype(np.float64))
                np.save(debug_dump_dir / f"{suffix}_gc_content.npy", gc_content_mask.astype(np.float64))
                np.save(debug_dump_dir / f"{suffix}_mean_frag_len.npy", mean_frag_len_mask.astype(np.float64))
                np.save(debug_dump_dir / f"{suffix}_post_consensus_sample_cov.npy", coverage_norm_mask.astype(np.float64))
                np.save(debug_dump_dir / f"{suffix}_sample_cov.npy", coverage_norm_mask.astype(np.float64))
                if consensus_indices is not None:
                    np.save(debug_dump_dir / f"{suffix}_consensus_indices.npy", consensus_indices.astype(np.int64))

            # Closure for parallel processing
            masks_base_dir = mask_type_configs[mask_type]["mask_dir"]
            consensus_indices_for_mask = None if keep_consensus_bins else consensus_indices

            def _load_and_add_mask(cell_type):
                from pathlib import Path
                import scipy.sparse
                import numpy as np

                mask_path = masks_base_dir / cell_type / f"{chrom}.npz"
                if not mask_path.exists():
                    return None
                try:
                    s = scipy.sparse.load_npz(mask_path).tocsr()
                    if include_indices_10bp_for_mask is not None:
                        if not (s.shape[1] == 1 and s.shape[0] > 1):
                            return None
                        s = s[include_indices_10bp_for_mask, :]
                    mask = s.astype(np.float32, copy=False).toarray().ravel()
                    mask = np.round(mask, decimals=2)

                    if bin_size != 10:
                        from src.optimized import aggregate_10bp_to_100bp_numba, NUMBA_AVAILABLE
                        if NUMBA_AVAILABLE:
                            mask = aggregate_10bp_to_100bp_numba(mask)
                        else:
                            n_10bp = len(mask)
                            n_100bp = (n_10bp + 9) // 10
                            mask_100bp = np.zeros(n_100bp, dtype=np.float32)
                            for i in range(n_100bp):
                                start = i * 10
                                end = min(start + 10, n_10bp)
                                mask_100bp[i] = np.sum(mask[start:end])
                            mask = mask_100bp

                    if consensus_indices_for_mask is not None and len(consensus_indices_for_mask) > 0:
                        if np.max(consensus_indices_for_mask) < len(mask):
                            mask = np.delete(mask, consensus_indices_for_mask)
                        else:
                            print(
                                f"      [{mask_type}] Warning: Consensus indices exceed mask length for {cell_type}; skipping removal"
                            )

                    if len(coverage_norm_mask) != len(mask):
                        min_len = min(len(coverage_norm_mask), len(mask))
                        if min_len == 0:
                            return None
                        x = coverage_norm_mask[:min_len].astype(np.float64)
                        y = mask[:min_len].astype(np.float64)
                    else:
                        x = coverage_norm_mask.astype(np.float64)
                        y = mask.astype(np.float64)

                    return (cell_type, x, y)
                except Exception as e:
                    print(f"      [{mask_type}] Warning: Failed to load mask for {cell_type}: {e}")
                    return None

            requested_workers = mask_type_worker_overrides.get(mask_type, 1)
            num_workers = min(cpu_count(), max(1, requested_workers))
            print(
                f"      [{mask_type}] Starting mask processing with {num_workers} worker(s) "
                f"(env {mask_type.upper()}_MASK_WORKERS={requested_workers})..."
            )
            import sys
            sys.stdout.flush()

            calculators = mask_type_calculators[mask_type]
            official_calculators = (
                mask_type_official_calculators.get(mask_type)
                if compare_official_running
                else None
            )
            successful = 0

            try:
                if num_workers == 1:
                    # Stream masks sequentially to keep memory usage low
                    for idx, cell_type in enumerate(cell_types, 1):
                        r = _load_and_add_mask(cell_type)
                        if r is not None:
                            cell_type_key, x, y = r
                            if cell_type_key in calculators:
                                calculators[cell_type_key].add_data(x, y)
                                if (
                                    compare_official_running
                                    and official_calculators
                                    and cell_type_key in official_calculators
                                ):
                                    official_calculators[cell_type_key].add_data(x, y)
                                if (
                                    debug_dump_enabled
                                    and chrom == debug_dump_chrom
                                    and cell_type_key == debug_dump_cell_type
                                ):
                                    suffix = f"{mask_type}_{cell_type_key}_{chrom}"
                                    np.save(debug_dump_dir / f"{suffix}_x.npy", x)
                                    np.save(debug_dump_dir / f"{suffix}_y.npy", y)
                                successful += 1
                        if idx % 25 == 0:
                            print(
                                f"        [{mask_type}] Loaded {idx}/{len(cell_types)} masks sequentially..."
                            )
                            sys.stdout.flush()
                else:
                    results = Parallel(
                        n_jobs=num_workers,
                        verbose=0,
                        batch_size=5,
                        timeout=None,
                    )(
                        delayed(_load_and_add_mask)(cell_type)
                        for cell_type in cell_types
                    )

                    for idx, r in enumerate(results):
                        if r is not None:
                            cell_type, x, y = r
                            if cell_type in calculators:
                                calculators[cell_type].add_data(x, y)
                                if (
                                    compare_official_running
                                    and official_calculators
                                    and cell_type in official_calculators
                                ):
                                    official_calculators[cell_type].add_data(x, y)
                                if (
                                    debug_dump_enabled
                                    and chrom == debug_dump_chrom
                                    and cell_type == debug_dump_cell_type
                                ):
                                    suffix = f"{mask_type}_{cell_type}_{chrom}"
                                    np.save(debug_dump_dir / f"{suffix}_x.npy", x)
                                    np.save(debug_dump_dir / f"{suffix}_y.npy", y)
                                successful += 1
                        if (idx + 1) % 50 == 0:
                            print(
                                f"        [{mask_type}] Added {idx + 1}/{len(results)} results to calculators..."
                            )
                            sys.stdout.flush()
            except Exception as e:
                print(f"      [{mask_type}] ERROR in mask processing: {e}")
                import traceback
                traceback.print_exc()
                raise

            print(
                f"      [{mask_type}] ✓ Processed {successful}/{len(cell_types)} masks successfully"
            )
            sys.stdout.flush()
    else:
        print(f"    Warning: Real masks not available for {chrom}")

    print(f"  Completed {chrom}")

if (trace_consensus_mask_types or keep_consensus_bins) and consensus_removal_totals:
    print("\nConsensus removal aggregates:")
    for mask_type, stats in consensus_removal_totals.items():
        if stats["bins"] == 0:
            continue
        print(
            f"  [{mask_type}] total_bins={stats['bins']:,} "
            f"x_sum={stats['x_sum']:.6f} "
            f"x_squared_sum={stats['x_squared_sum']:.6f}"
        )

if compare_official_running:
    print("\nComparing RunningPearsonR stats against official implementation...")
    for mask_type, calculators in mask_type_calculators.items():
        official_map = mask_type_official_calculators.get(mask_type, {})
        for cell_type, calc in calculators.items():
            stats = calc.get_stats()
            official_calc = official_map.get(cell_type)
            if official_calc is None:
                continue
            official_stats = official_calc.state
            diff_x = stats["x_sum"] - official_stats["x_sum"]
            diff_x2 = stats["x_squared_sum"] - official_stats["x_squared_sum"]
            diff_n = stats["n"] - official_stats["n"]
            if any(abs(val) > 1e-6 for val in (diff_x, diff_x2, diff_n)):
                print(
                    f"  [{mask_type}::{cell_type}] Δx_sum={diff_x:.6f} "
                    f"Δx_squared_sum={diff_x2:.6f} Δn={diff_n:.0f}"
                )
    for mask_type, calc in consensus_calculators.items():
        official_calc = consensus_official_calculators.get(mask_type)
        if not official_calc:
            continue
        stats = calc.get_stats()
        official_stats = official_calc.state
        diff_x = stats["x_sum"] - official_stats["x_sum"]
        diff_x2 = stats["x_squared_sum"] - official_stats["x_squared_sum"]
        diff_n = stats["n"] - official_stats["n"]
        if any(abs(val) > 1e-6 for val in (diff_x, diff_x2, diff_n)):
            print(
                f"  [{mask_type}::consensus] Δx_sum={diff_x:.6f} "
                f"Δx_squared_sum={diff_x2:.6f} Δn={diff_n:.0f}"
            )

# -----------------------------
# 10. Compute Final Features from Accumulated Statistics
# -----------------------------
print("\n" + "=" * 70)
print("Step 10: Computing final features from accumulated statistics...")
print("=" * 70)

corr_stats = {}
for mask_type in mask_types_to_process:
    calculators = mask_type_calculators.get(mask_type, {})
    output_order = mask_type_configs[mask_type]["output_order"]
    for cell_type in output_order:
        if cell_type == "consensus":
            r_calc = consensus_calculators.get(mask_type)
            if r_calc is None:
                continue
        else:
            r_calc = calculators.get(cell_type)
            if r_calc is None:
                continue

        stats = r_calc.get_stats()
        n = stats['n']
        
        key_prefix = f"{mask_type}::{cell_type}"

        if n < 2:
            # Not enough data
            corr_stats[f"pearson_r_{key_prefix}"] = 0.0
            corr_stats[f"p_value_{key_prefix}"] = 1.0
            corr_stats[f"fraction_within_{key_prefix}"] = 0.0
            corr_stats[f"cosine_similarity_{key_prefix}"] = 0.0
            corr_stats[f"x_sum_{key_prefix}"] = 0.0
            corr_stats[f"y_sum_{key_prefix}"] = 0.0
            corr_stats[f"x_squared_sum_{key_prefix}"] = 0.0
            corr_stats[f"y_squared_sum_{key_prefix}"] = 0.0
            corr_stats[f"xy_sum_{key_prefix}"] = 0.0
            corr_stats[f"n_{key_prefix}"] = 0.0
            continue
        
        # Compute Pearson R
        r = r_calc.compute_pearson_r()
        
        # Compute p-value
        if abs(r) == 1.0:
            p = 0.0
        else:
            df = n - 2
            if df <= 0:
                p = 1.0
            else:
                ab = df / 2.0
                p = 2 * special.btdtr(ab, ab, 0.5 * (1.0 - abs(r)))
        
        # Compute other features
        fraction_within = stats['xy_sum'] / n if n > 0 else 0.0
        
        # Cosine similarity
        x_norm = np.sqrt(stats['x_squared_sum'])
        y_norm = np.sqrt(stats['y_squared_sum'])
        if x_norm > 0 and y_norm > 0:
            cosine_sim = stats['xy_sum'] / (x_norm * y_norm)
            cosine_sim = max(min(cosine_sim, 1.0), -1.0)
        else:
            cosine_sim = 0.0
        
        # Store all 10 features (Pearson R will be standardized later)
        corr_stats[f"pearson_r_{key_prefix}"] = float(r)
        corr_stats[f"p_value_{key_prefix}"] = float(p)
        corr_stats[f"fraction_within_{key_prefix}"] = float(fraction_within)
        corr_stats[f"cosine_similarity_{key_prefix}"] = float(cosine_sim)
        corr_stats[f"x_sum_{key_prefix}"] = stats['x_sum']
        corr_stats[f"y_sum_{key_prefix}"] = stats['y_sum']
        corr_stats[f"x_squared_sum_{key_prefix}"] = stats['x_squared_sum']
        corr_stats[f"y_squared_sum_{key_prefix}"] = stats['y_squared_sum']
        corr_stats[f"xy_sum_{key_prefix}"] = stats['xy_sum']
        corr_stats[f"n_{key_prefix}"] = float(n)

# -----------------------------
# 10.5. Standardize Pearson R (matching LIONHEART)
# -----------------------------
print("\nStep 10.5: Standardizing Pearson R values...")
# Extract all Pearson R values
pearson_r_keys = [k for k in corr_stats.keys() if k.startswith("pearson_r_")]
pearson_r_values = np.array([corr_stats[k] for k in pearson_r_keys])

# Standardize: z = (x - mean) / std
pearson_r_mean = np.mean(pearson_r_values)
pearson_r_std = np.std(pearson_r_values)
if pearson_r_std > 0:
    pearson_r_standardized = (pearson_r_values - pearson_r_mean) / pearson_r_std
else:
    pearson_r_standardized = pearson_r_values - pearson_r_mean  # Only center if std=0

# Update corr_stats with standardized values
for i, key in enumerate(pearson_r_keys):
    corr_stats[key] = float(pearson_r_standardized[i])

print(f"  Standardized {len(pearson_r_keys)} Pearson R values")
print(f"  Mean (before): {pearson_r_mean:.6f}, Std (before): {pearson_r_std:.6f}")
print(f"  Mean (after): {np.mean(pearson_r_standardized):.6f}, Std (after): {np.std(pearson_r_standardized):.6f}")

# Save standardization parameters (for reference)
std_params_path = Path(output_path).with_suffix(".standardization_params.json")
with open(std_params_path, "w") as f:
    json.dump({"mean": float(pearson_r_mean), "std": float(pearson_r_std)}, f)
print(f"  Saved standardization parameters to {std_params_path}")

# -----------------------------
# 11. Write Features
# -----------------------------
print("\nStep 11: Writing features...")
write_feature_vector(output_path, {}, {}, corr_stats)
if os.environ.get("SAVE_CORR_STATS"):
    np.save(
        Path(output_path).with_suffix(".corr_stats.npy"),
        corr_stats,
        allow_pickle=True,
    )

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
print(f"\nProcessed {len(all_chroms)} chromosome(s): {', '.join(all_chroms)}")
print(f"Output: {output_path}")
print(f"Total features: {len(corr_stats)}")
print(f"Features per cell type: 10")
print(f"Cell types: {total_calculators}")
print(f"\nNote: To process all 22 autosomes, change 'all_chroms' to:")
print(f"      all_chroms = [f'chr{{i}}' for i in range(1, 23)]")

