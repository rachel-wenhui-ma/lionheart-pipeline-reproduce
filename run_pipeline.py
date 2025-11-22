# run_pipeline.py
import sys
import os
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extract_insert_size import extract_fragment_lengths, compute_fragment_features, extract_fragment_lengths_per_bin
from src.compute_coverage import compute_coverage, normalize_coverage, calculate_gc_correction_factors, correct_gc_bias
from src.compute_correlation import compute_correlations
from src.assemble_features import write_feature_vector
from src.utils import calculate_gc_content_per_bin
import pysam
import numpy as np

# -----------------------------
# Config
# -----------------------------
# Check if running in WSL
if os.path.exists("/mnt/d"):
    base_path = "/mnt/d/MADS/25 Fall/CSC 527/Final project"
    bam_path = f"{base_path}/example_bam_hg38/IC38.hg38.downsampled.aligned.sorted.markdup.bam"
    reference_fasta = f"{base_path}/reproduce/data/chr21.fa"
else:
    bam_path = "data/demo.bam"
    reference_fasta = "data/chr21.fa"

chrom = "chr21"
bin_size = 100

mask_paths = {
    "Tcell": "data/masks/Tcell.bed",
    "Monocyte": "data/masks/Monocyte.bed",
    "Liver": "data/masks/Liver.bed"
}

output_path = "features_sample.csv"

# -----------------------------
# 1. Fragment length
# -----------------------------
print("Step 1: Extracting fragment lengths...")
lengths = extract_fragment_lengths(bam_path, chrom=chrom)
if len(lengths) == 0:
    raise ValueError(f"No valid fragments found in {bam_path} for {chrom}")
frag_stats = compute_fragment_features(lengths)
print(f"  Global frag mean: {frag_stats['frag_mean']:.2f} bp")
print(f"  Total fragments: {len(lengths)}")

# Per-bin fragment length (for future insert size correction, not output as feature)
print("\nStep 1b: Extracting per-bin fragment lengths...")
mean_frag_per_bin, median_frag_per_bin = extract_fragment_lengths_per_bin(bam_path, chrom, bin_size)
n_bins_with_frags = np.sum(~np.isnan(mean_frag_per_bin))
print(f"  Per-bin fragment length: {len(mean_frag_per_bin)} bins")
print(f"  Bins with fragment data: {n_bins_with_frags:,}")
if n_bins_with_frags > 0:
    valid_mean = mean_frag_per_bin[~np.isnan(mean_frag_per_bin)]
    print(f"  Mean frag length (per-bin): {np.mean(valid_mean):.2f} bp")
    print(f"  Note: Per-bin fragment length used for correction, not output as feature")

# -----------------------------
# 2. Coverage
# -----------------------------
print("\nStep 2: Computing coverage...")
chrom_len = pysam.AlignmentFile(bam_path).get_reference_length(chrom)
coverage = compute_coverage(bam_path, chrom, bin_size)
print(f"  Coverage computed: {len(coverage)} bins")
print(f"  Mean coverage: {np.mean(coverage):.2f}")

# -----------------------------
# 3. GC Content Calculation
# -----------------------------
print("\nStep 3: Calculating GC content...")
if not os.path.exists(reference_fasta):
    raise FileNotFoundError(f"Reference genome not found: {reference_fasta}")
gc_content = calculate_gc_content_per_bin(reference_fasta, chrom, bin_size)
print(f"  GC content computed: {len(gc_content)} bins")
print(f"  Mean GC content: {np.mean(gc_content):.4f} ({np.mean(gc_content)*100:.2f}%)")

# -----------------------------
# 4. GC Correction
# -----------------------------
print("\nStep 4: Applying GC correction...")
gc_bin_edges, gc_correction_factors = calculate_gc_correction_factors(
    coverage, gc_content, num_bins=20
)
coverage_gc_corrected = correct_gc_bias(
    coverage, gc_content, gc_bin_edges, gc_correction_factors
)
print(f"  GC correction factors: {len(gc_correction_factors)} bins")
print(f"  Mean correction factor: {np.nanmean(gc_correction_factors):.4f}")
print(f"  Coverage after GC correction: mean={np.mean(coverage_gc_corrected):.4f}")

# Normalize coverage (after GC correction)
coverage_norm = normalize_coverage(coverage_gc_corrected)

# -----------------------------
# 5. Correlation
# -----------------------------
print("\nStep 5: Computing correlations...")
corr_stats = compute_correlations(coverage_norm, mask_paths, chrom, chrom_len, bin_size)

# -----------------------------
# 6. Assemble
# -----------------------------
print("\nStep 6: Assembling features...")
# Only output correlation features (matching LIONHEART format)
# frag_stats, cov_stats, and per-bin fragment length are computed but not included in output
write_feature_vector(output_path, {}, {}, corr_stats)

print("\nDone!")
print(f"\nNote: Fragment length, coverage, and GC statistics are computed but not included in output.")
print(f"      Output contains only correlation features (matching LIONHEART format).")
print(f"      Total features: {len(corr_stats)}")

