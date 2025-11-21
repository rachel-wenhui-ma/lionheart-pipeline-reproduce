# run_pipeline.py
import sys
import os
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extract_insert_size import extract_fragment_lengths, compute_fragment_features
from src.compute_coverage import compute_coverage, normalize_coverage
from src.compute_correlation import compute_correlations
from src.assemble_features import write_feature_vector
import pysam
import numpy as np

# -----------------------------
# Config
# -----------------------------
bam_path = "data/demo.bam"
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
lengths = extract_fragment_lengths(bam_path)
frag_stats = compute_fragment_features(lengths)
print(f"  Global frag mean: {frag_stats['frag_mean']:.2f} bp")

# -----------------------------
# 2. Coverage
# -----------------------------
print("\nStep 2: Computing coverage...")
chrom_len = pysam.AlignmentFile(bam_path).get_reference_length(chrom)
coverage = compute_coverage(bam_path, chrom, bin_size)
print(f"  Coverage computed: {len(coverage)} bins")
print(f"  Mean coverage: {np.mean(coverage):.2f}")

# Normalize coverage
coverage_norm = normalize_coverage(coverage)
cov_stats = {
    "cov_mean": float(np.mean(coverage_norm)),
    "cov_std": float(np.std(coverage_norm)),
    "cov_median": float(np.median(coverage_norm)),
}

# -----------------------------
# 3. Correlation
# -----------------------------
print("\nStep 3: Computing correlations...")
corr_stats = compute_correlations(coverage_norm, mask_paths, chrom, chrom_len, bin_size)

# -----------------------------
# 4. Assemble
# -----------------------------
print("\nStep 4: Assembling features...")
write_feature_vector(output_path, frag_stats, cov_stats, corr_stats)

print("\nDone!")

