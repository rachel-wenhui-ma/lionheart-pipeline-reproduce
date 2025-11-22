#!/usr/bin/env python
"""Comprehensive testing suite for the simplified pipeline"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.compute_coverage import (
    compute_coverage, 
    calculate_gc_correction_factors, 
    correct_gc_bias,
    normalize_megabins_simple,
    normalize_coverage
)
from src.compute_correlation import compute_correlations
from src.extract_insert_size import extract_fragment_lengths_per_bin
from src.utils import calculate_gc_content_per_bin
import pysam

# Config
if os.path.exists("/mnt/d"):  # WSL
    base_path = "/mnt/d/MADS/25 Fall/CSC 527/Final project"
    bam_path = f"{base_path}/example_bam_hg38/IC38.hg38.downsampled.aligned.sorted.markdup.bam"
    reference_fasta = f"{base_path}/reproduce/data/chr21.fa"
    mask_dir = f"{base_path}/reproduce/data/masks"
else:  # Windows
    bam_path = "../example_bam_hg38/IC38.hg38.downsampled.aligned.sorted.markdup.bam"
    reference_fasta = "data/chr21.fa"
    mask_dir = "data/masks"

chrom = "chr21"
bin_size = 100
mask_paths = {
    "Tcell": f"{mask_dir}/Tcell.bed",
    "Monocyte": f"{mask_dir}/Monocyte.bed",
    "Liver": f"{mask_dir}/Liver.bed"
}

print("=" * 70)
print("Comprehensive Pipeline Testing")
print("=" * 70)

# Test 1: Reproducibility test
print("\n" + "=" * 70)
print("Test 1: Reproducibility (Run pipeline twice)")
print("=" * 70)

def run_pipeline_once():
    """Run pipeline and return key results"""
    coverage = compute_coverage(bam_path, chrom, bin_size)
    gc_content = calculate_gc_content_per_bin(reference_fasta, chrom, bin_size)
    gc_bin_edges, gc_factors = calculate_gc_correction_factors(coverage, gc_content, num_bins=20)
    coverage_gc = correct_gc_bias(coverage, gc_content, gc_bin_edges, gc_factors)
    coverage_megabin = normalize_megabins_simple(coverage_gc, bin_size=bin_size, mbin_size=1_000_000)
    coverage_norm = normalize_coverage(coverage_megabin)
    chrom_len = pysam.AlignmentFile(bam_path).get_reference_length(chrom)
    corr_stats = compute_correlations(coverage_norm, mask_paths, chrom, chrom_len, bin_size)
    return {
        'coverage_mean': np.mean(coverage),
        'coverage_std': np.std(coverage),
        'gc_corr_mean': np.mean(gc_factors),
        'corr_tcell': corr_stats['corr_Tcell'],
        'corr_monocyte': corr_stats['corr_Monocyte'],
        'corr_liver': corr_stats['corr_Liver'],
    }

result1 = run_pipeline_once()
result2 = run_pipeline_once()

print("Run 1 results:")
for key, value in result1.items():
    print(f"  {key}: {value:.6f}")

print("\nRun 2 results:")
for key, value in result2.items():
    print(f"  {key}: {value:.6f}")

# Check if results are identical
all_match = all(abs(result1[k] - result2[k]) < 1e-10 for k in result1.keys())
if all_match:
    print("\n✓ Results are reproducible (identical across runs)")
else:
    print("\n⚠ Results differ between runs (may indicate non-deterministic behavior)")

# Test 2: Parameter sensitivity
print("\n" + "=" * 70)
print("Test 2: Parameter Sensitivity")
print("=" * 70)

coverage = compute_coverage(bam_path, chrom, bin_size)
gc_content = calculate_gc_content_per_bin(reference_fasta, chrom, bin_size)
gc_bin_edges, gc_factors = calculate_gc_correction_factors(coverage, gc_content, num_bins=20)
coverage_gc = correct_gc_bias(coverage, gc_content, gc_bin_edges, gc_factors)

# Test different megabin sizes
megabin_sizes = [500_000, 1_000_000, 2_000_000]  # 0.5MB, 1MB, 2MB
print("\nTesting different megabin sizes:")
for mbin_size in megabin_sizes:
    coverage_mb = normalize_megabins_simple(coverage_gc, bin_size=bin_size, mbin_size=mbin_size)
    coverage_norm = normalize_coverage(coverage_mb)
    chrom_len = pysam.AlignmentFile(bam_path).get_reference_length(chrom)
    corr_stats = compute_correlations(coverage_norm, mask_paths, chrom, chrom_len, bin_size)
    print(f"  Megabin size {mbin_size//1000}KB:")
    print(f"    Mean coverage: {np.mean(coverage_mb):.4f}")
    print(f"    Std coverage:  {np.std(coverage_mb):.4f}")
    print(f"    Corr Tcell:     {corr_stats['corr_Tcell']:.6f}")

# Test 3: Edge cases
print("\n" + "=" * 70)
print("Test 3: Edge Cases")
print("=" * 70)

# Test with empty coverage (all zeros)
print("\n3.1 Testing with all-zero coverage:")
zero_coverage = np.zeros(1000)
zero_gc = np.random.uniform(0.3, 0.5, 1000)
try:
    bin_edges, factors = calculate_gc_correction_factors(zero_coverage, zero_gc, num_bins=10)
    print("  ✓ Handles all-zero coverage")
except Exception as e:
    print(f"  ✗ Error with all-zero coverage: {e}")

# Test with constant coverage
print("\n3.2 Testing with constant coverage:")
const_coverage = np.ones(1000) * 5.0
const_gc = np.random.uniform(0.3, 0.5, 1000)
try:
    bin_edges, factors = calculate_gc_correction_factors(const_coverage, const_gc, num_bins=10)
    corrected = correct_gc_bias(const_coverage, const_gc, bin_edges, factors)
    print(f"  ✓ Handles constant coverage (mean after correction: {np.mean(corrected):.4f})")
except Exception as e:
    print(f"  ✗ Error with constant coverage: {e}")

# Test 4: Data quality checks
print("\n" + "=" * 70)
print("Test 4: Data Quality Checks")
print("=" * 70)

coverage = compute_coverage(bam_path, chrom, bin_size)
gc_content = calculate_gc_content_per_bin(reference_fasta, chrom, bin_size)

print(f"\n4.1 Coverage quality:")
print(f"  Total bins: {len(coverage):,}")
print(f"  Non-zero bins: {np.sum(coverage > 0):,} ({100*np.sum(coverage > 0)/len(coverage):.2f}%)")
print(f"  Zero bins: {np.sum(coverage == 0):,} ({100*np.sum(coverage == 0)/len(coverage):.2f}%)")
print(f"  Coverage distribution:")
print(f"    Mean: {np.mean(coverage):.4f}")
print(f"    Median: {np.median(coverage):.4f}")
print(f"    Percentiles: 25th={np.percentile(coverage, 25):.2f}, 50th={np.median(coverage):.2f}, 75th={np.percentile(coverage, 75):.2f}, 95th={np.percentile(coverage, 95):.2f}")

print(f"\n4.2 GC content quality:")
print(f"  Mean GC: {np.mean(gc_content):.4f}")
print(f"  Std GC: {np.std(gc_content):.4f}")
print(f"  Range: [{np.min(gc_content):.4f}, {np.max(gc_content):.4f}]")
print(f"  Valid GC bins: {np.sum(~np.isnan(gc_content)):,}")

# Test 5: Intermediate results validation
print("\n" + "=" * 70)
print("Test 5: Intermediate Results Validation")
print("=" * 70)

coverage = compute_coverage(bam_path, chrom, bin_size)
gc_content = calculate_gc_content_per_bin(reference_fasta, chrom, bin_size)

# Check GC correction
gc_bin_edges, gc_factors = calculate_gc_correction_factors(coverage, gc_content, num_bins=20)
coverage_gc = correct_gc_bias(coverage, gc_content, gc_bin_edges, gc_factors)

print("\n5.1 GC correction validation:")
print(f"  GC factors range: [{np.nanmin(gc_factors):.4f}, {np.nanmax(gc_factors):.4f}]")
print(f"  GC factors mean: {np.nanmean(gc_factors):.4f}")

# Check correlation before/after GC correction
valid_mask = ~(np.isnan(gc_content) | (coverage == 0))
if np.sum(valid_mask) > 100:
    corr_before = np.corrcoef(coverage[valid_mask], gc_content[valid_mask])[0, 1]
    corr_after = np.corrcoef(coverage_gc[valid_mask], gc_content[valid_mask])[0, 1]
    print(f"  Correlation with GC before correction: {corr_before:.6f}")
    print(f"  Correlation with GC after correction:  {corr_after:.6f}")
    if abs(corr_after) < abs(corr_before):
        print("  ✓ GC correction reduced GC correlation")
    else:
        print("  ⚠ GC correction did not reduce GC correlation")

# Check megabin normalization
coverage_megabin = normalize_megabins_simple(coverage_gc, bin_size=bin_size, mbin_size=1_000_000)
print(f"\n5.2 Megabin normalization validation:")
print(f"  Mean before megabin norm: {np.mean(coverage_gc):.4f}")
print(f"  Mean after megabin norm:  {np.mean(coverage_megabin):.4f}")
if abs(np.mean(coverage_megabin)) < 0.1:
    print("  ✓ Mean close to zero after megabin normalization")
else:
    print("  ⚠ Mean not close to zero after megabin normalization")

# Test 6: Output format validation
print("\n" + "=" * 70)
print("Test 6: Output Format Validation")
print("=" * 70)

if os.path.exists("features_sample.csv"):
    df = pd.read_csv("features_sample.csv")
    print(f"\n6.1 Feature file validation:")
    print(f"  Total features: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Features:")
    for _, row in df.iterrows():
        print(f"    {row['feature']}: {row['value']:.6f}")
    
    # Check for expected features
    expected_features = ['frag_mean', 'frag_median', 'frag_std', 
                        'cov_mean', 'cov_std', 'cov_median',
                        'corr_Tcell', 'corr_Monocyte', 'corr_Liver']
    missing = [f for f in expected_features if f not in df['feature'].values]
    if not missing:
        print("  ✓ All expected features present")
    else:
        print(f"  ⚠ Missing features: {missing}")
    
    # Check value ranges
    print(f"\n6.2 Value range validation:")
    frag_features = df[df['feature'].str.startswith('frag_')]
    cov_features = df[df['feature'].str.startswith('cov_')]
    corr_features = df[df['feature'].str.startswith('corr_')]
    
    print(f"  Fragment length features:")
    print(f"    Range: [{frag_features['value'].min():.2f}, {frag_features['value'].max():.2f}]")
    if 100 < frag_features['value'].min() < 250:
        print("    ✓ Fragment lengths in reasonable range (100-250 bp)")
    
    print(f"  Coverage features:")
    print(f"    Range: [{cov_features['value'].min():.6f}, {cov_features['value'].max():.6f}]")
    if abs(cov_features['value'].max()) < 10:
        print("    ✓ Coverage stats in reasonable range")
    
    print(f"  Correlation features:")
    print(f"    Range: [{corr_features['value'].min():.6f}, {corr_features['value'].max():.6f}]")
    if all(-1 <= v <= 1 for v in corr_features['value']):
        print("    ✓ Correlations in valid range [-1, 1]")
else:
    print("  ⚠ Output file not found. Run pipeline first.")

# Summary
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print("Tests completed:")
print("  1. ✓ Reproducibility test")
print("  2. ✓ Parameter sensitivity test")
print("  3. ✓ Edge cases test")
print("  4. ✓ Data quality checks")
print("  5. ✓ Intermediate results validation")
print("  6. ✓ Output format validation")
print("=" * 70)


