# run_pipeline_all_chroms.py
"""
Run feature extraction pipeline for all 22 autosomes (chr1-chr22).
Accumulates statistics across all chromosomes (matching LIONHEART's approach).
"""
import sys
import os
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extract_insert_size import extract_fragment_lengths, compute_fragment_features, extract_fragment_lengths_per_bin
from src.compute_coverage import compute_coverage, normalize_coverage, calculate_gc_correction_factors, correct_gc_bias, normalize_megabins_simple
from src.compute_correlation import compute_all_features
from src.assemble_features import write_feature_vector
from src.utils import calculate_gc_content_per_bin, load_lionheart_gc_content
from src.outlier_detection import find_clipping_threshold, clip_outliers
from src.insert_size_correction import calculate_insert_size_correction_factors, correct_bias
from src.load_masks import load_cell_type_masks, load_cell_type_list
from src.running_stats import RunningPearsonR
from src.blacklist import load_exclude_bins, apply_exclude_to_coverage
from pathlib import Path
import pysam
import numpy as np
from scipy import special
from typing import Dict

# -----------------------------
# Config
# -----------------------------
# All 22 autosomes (chr1-chr22)
all_chroms = [f"chr{i}" for i in range(1, 23)]
bin_size = 10  # LIONHEART works at 10bp level (not 100bp aggregated)

# Configuration for real LIONHEART masks
from src.paths import get_resources_dir, get_reference_dir, get_bam_path

# Get paths using unified path configuration
resources_dir = get_resources_dir()
reference_dir = get_reference_dir()
bam_path = get_bam_path()

use_real_masks = True
resources_dir = get_resources_dir()
reference_dir = get_reference_dir()
bam_path = get_bam_path()

if use_real_masks and resources_dir.exists():
    print(f"Using real LIONHEART masks from: {resources_dir}")
    cell_type_df = load_cell_type_list(resources_dir, "DNase")
    all_cell_types = cell_type_df["cell_type"].tolist()
    print(f"  Loading ALL {len(all_cell_types)} DNase cell types")
    print(f"  This will generate {len(all_cell_types) * 10} features (10 feature types × {len(all_cell_types)} cell types)")
else:
    all_cell_types = None
    print("Warning: Real masks not available, using test masks")

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
print("Initializing feature accumulators for all cell types...")
print("=" * 70)

r_calculators: Dict[str, RunningPearsonR] = {}
if all_cell_types:
    for cell_type in all_cell_types:
        r_calculators[cell_type] = RunningPearsonR(ignore_nans=True)
    print(f"Initialized {len(r_calculators)} RunningPearsonR calculators")

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
    print(f"  Step 2: Computing coverage for {chrom}...")
    coverage = compute_coverage(bam_path, chrom, bin_size=bin_size, mapq_threshold=20, use_overlap=True)
    
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
        print(f"    Original coverage bins: {len(coverage):,}")
        print(f"    GC bins (after exclude, 10bp): {len(gc_content):,}")
        if len(include_indices_10bp) <= len(coverage):
            coverage = coverage[include_indices_10bp]
            mean_frag_len = mean_frag_len[include_indices_10bp]
            print(f"    Filtered coverage bins: {len(coverage):,}")
        else:
            print(f"    Warning: include_indices_10bp ({len(include_indices_10bp)}) > coverage ({len(coverage)})")
            # Fallback: clip include_indices to valid range
            valid_indices = include_indices_10bp[include_indices_10bp < len(coverage)]
            coverage = coverage[valid_indices]
            mean_frag_len = mean_frag_len[valid_indices]
            gc_content = gc_content[:len(valid_indices)]  # Also clip GC to match
            include_indices_10bp = valid_indices
            print(f"    Filtered coverage bins (clipped): {len(coverage):,}")
    
    # -----------------------------
    # 5. Outlier Detection (ZIPoisson)
    # -----------------------------
    print(f"  Step 5: Detecting outliers with ZIPoisson for {chrom}...")
    clipping_val = find_clipping_threshold(coverage)
    coverage_clipped = clip_outliers(coverage, clipping_val)
    
    # -----------------------------
    # 6. GC Correction
    # -----------------------------
    print(f"  Step 6: Applying GC correction for {chrom}...")
    bin_edges, gc_factors = calculate_gc_correction_factors(coverage_clipped, gc_content)
    coverage_gc_corrected = correct_gc_bias(coverage_clipped, gc_content, bin_edges, gc_factors)
    
    # -----------------------------
    # 7. Insert Size Correction
    # -----------------------------
    print(f"  Step 7: Applying insert size correction for {chrom}...")
    # Create insert size bin edges (100-220bp, typical range)
    insert_size_bin_edges = np.linspace(100, 220, 25)  # 24 bins, 5bp per bin
    
    # Calculate correction factors
    insert_size_factors = calculate_insert_size_correction_factors(
        coverages=coverage_gc_corrected,
        insert_sizes=mean_frag_len,
        bin_edges=insert_size_bin_edges,
        final_mean_insert_size=166.0,
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
            bin_edges=insert_size_bin_edges,
        )
        
        # 2. Skewness correction
        coverage_after_skew = coverage_after_noise.copy()
        coverage_after_skew[valid_mask] = correct_bias(
            coverages=coverage_after_noise[valid_mask],
            correct_factors=insert_size_factors["skewness_correction_factor"],
            bias_scores=mean_frag_len[valid_mask],
            bin_edges=insert_size_bin_edges,
        )
        
        # 3. Mean shift correction
        coverage_insert_corrected = coverage_after_skew.copy()
        coverage_insert_corrected[valid_mask] = correct_bias(
            coverages=coverage_after_skew[valid_mask],
            correct_factors=insert_size_factors["mean_correction_factor"],
            bias_scores=mean_frag_len[valid_mask],
            bin_edges=insert_size_bin_edges,
        )
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
    
    # -----------------------------
    # 10. Load Masks and Accumulate Statistics
    # -----------------------------
    print(f"  Step 10: Loading masks and accumulating statistics for {chrom}...")
    
    if use_real_masks and resources_dir.exists():
        # LIONHEART's approach: Load and process masks one at a time to save memory
        # Instead of loading all 354 masks into memory (31GB!), we load each mask,
        # immediately add it to the calculator, and let it be garbage collected.
        
        from joblib import Parallel, delayed
        from multiprocessing import cpu_count
        
        # At 10bp level, use the same include_indices_10bp as coverage
        include_indices_10bp_for_mask = include_indices_10bp if use_lionheart_gc and include_indices_10bp is not None else None
        
        def _load_and_add_mask(cell_type):
            """Load a single mask and add it to the calculator (memory-efficient)."""
            # Use closure to access coverage_norm (simpler, may have serialization overhead but should work)
            from pathlib import Path
            import scipy.sparse
            import numpy as np
            
            # Get mask path
            masks_base_dir = resources_dir / "chromatin_masks" / "DNase" / "sparse_overlaps_by_chromosome"
            mask_path = masks_base_dir / cell_type / f"{chrom}.npz"
            
            if not mask_path.exists():
                return None
            
            try:
                # Load mask (same as LIONHEART's _load_from_sparse_array)
                s = scipy.sparse.load_npz(mask_path).tocsr()
                if include_indices is not None:
                    if not (s.shape[1] == 1 and s.shape[0] > 1):
                        return None
                    s = s[include_indices, :]
                mask = s.astype(np.float32, copy=False).toarray().ravel()
                mask = np.round(mask, decimals=2)
                
                # No aggregation needed at 10bp level
                if bin_size != 10:
                    # Aggregate if needed (shouldn't happen at 10bp)
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
                
                # Ensure same length
                if len(coverage_norm) != len(mask):
                    min_len = min(len(coverage_norm), len(mask))
                    if min_len == 0:
                        return None
                    x = coverage_norm[:min_len].astype(np.float64)
                    y = mask[:min_len].astype(np.float64)
                else:
                    x = coverage_norm.astype(np.float64)
                    y = mask.astype(np.float64)
                
                # Return (cell_type, x, y) to add in main process
                return (cell_type, x, y)
            except Exception as e:
                print(f"    Warning: Failed to load mask for {cell_type}: {e}")
                return None
        
        # Process masks in parallel (but each worker only holds one mask at a time)
        num_workers = min(cpu_count(), 8)
        print(f"    Processing {len(all_cell_types)} masks in parallel ({num_workers} workers, memory-efficient)...")
        
        print(f"    Starting parallel processing...")
        import sys
        sys.stdout.flush()  # Force flush before parallel processing
        
        # Use closure to access coverage_norm (simpler approach, may serialize but should work)
        # This is the approach that worked before we added shared_memory
        try:
            results = Parallel(n_jobs=num_workers, verbose=1, batch_size=10, timeout=300)(
                delayed(_load_and_add_mask)(cell_type) for cell_type in all_cell_types
            )
            print(f"    Parallel processing returned {len(results)} results")
            sys.stdout.flush()
        except Exception as e:
            print(f"    ERROR in parallel processing: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"    Parallel processing completed, processing results...")
        sys.stdout.flush()
        
        # Add data to r_calculators in the main process
        # (worker processes return the data, we add it here)
        successful = 0
        for idx, r in enumerate(results):
            if r is not None:
                cell_type, x, y = r
                if cell_type in r_calculators:
                    r_calculators[cell_type].add_data(x, y)
                    successful += 1
            if (idx + 1) % 50 == 0:
                print(f"      Added {idx + 1}/{len(results)} results to calculators...")
                sys.stdout.flush()
        
        print(f"    ✓ Processed {successful}/{len(all_cell_types)} masks successfully")
        sys.stdout.flush()
    else:
        print(f"    Warning: Real masks not available for {chrom}")
    
    print(f"  Completed {chrom}")

# -----------------------------
# 10. Compute Final Features from Accumulated Statistics
# -----------------------------
print("\n" + "=" * 70)
print("Step 10: Computing final features from accumulated statistics...")
print("=" * 70)

corr_stats = {}
for cell_type, r_calc in r_calculators.items():
    stats = r_calc.get_stats()
    n = stats['n']
    
    if n < 2:
        # Not enough data
        corr_stats[f"pearson_r_{cell_type}"] = 0.0
        corr_stats[f"p_value_{cell_type}"] = 1.0
        corr_stats[f"fraction_within_{cell_type}"] = 0.0
        corr_stats[f"cosine_similarity_{cell_type}"] = 0.0
        corr_stats[f"x_sum_{cell_type}"] = 0.0
        corr_stats[f"y_sum_{cell_type}"] = 0.0
        corr_stats[f"x_squared_sum_{cell_type}"] = 0.0
        corr_stats[f"y_squared_sum_{cell_type}"] = 0.0
        corr_stats[f"xy_sum_{cell_type}"] = 0.0
        corr_stats[f"n_{cell_type}"] = 0.0
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
    
    # Store all 10 features
    corr_stats[f"pearson_r_{cell_type}"] = float(r)
    corr_stats[f"p_value_{cell_type}"] = float(p)
    corr_stats[f"fraction_within_{cell_type}"] = float(fraction_within)
    corr_stats[f"cosine_similarity_{cell_type}"] = float(cosine_sim)
    corr_stats[f"x_sum_{cell_type}"] = stats['x_sum']
    corr_stats[f"y_sum_{cell_type}"] = stats['y_sum']
    corr_stats[f"x_squared_sum_{cell_type}"] = stats['x_squared_sum']
    corr_stats[f"y_squared_sum_{cell_type}"] = stats['y_squared_sum']
    corr_stats[f"xy_sum_{cell_type}"] = stats['xy_sum']
    corr_stats[f"n_{cell_type}"] = float(n)

# -----------------------------
# 11. Write Features
# -----------------------------
print("\nStep 11: Writing features...")
write_feature_vector(output_path, {}, {}, corr_stats)

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
print(f"\nProcessed {len(all_chroms)} chromosome(s): {', '.join(all_chroms)}")
print(f"Output: {output_path}")
print(f"Total features: {len(corr_stats)}")
print(f"Features per cell type: 10")
print(f"Cell types: {len(r_calculators)}")
print(f"\nNote: To process all 22 autosomes, change 'all_chroms' to:")
print(f"      all_chroms = [f'chr{{i}}' for i in range(1, 23)]")

