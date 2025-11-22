#!/usr/bin/env python
"""Compare our simplified pipeline output with LIONHEART output"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json

# Paths
lionheart_output_dir = "/home/mara/lionheart_work/output/dataset"
our_output_file = "features_sample.csv"

print("=" * 70)
print("Comparing Simplified Pipeline vs LIONHEART Output")
print("=" * 70)

# Load LIONHEART output
print("\n" + "-" * 70)
print("LIONHEART Output:")
print("-" * 70)

lionheart_features = np.load(f"{lionheart_output_dir}/feature_dataset.npy")
print(f"Feature dataset shape: {lionheart_features.shape}")
print(f"  - 10 feature types (Pearson R, p-value, cosine similarity, etc.)")
print(f"  - 898 cell types (354 DNase + 544 ATAC)")

# Feature type 0 is the main LIONHEART score (Pearson R)
pearson_r_scores = lionheart_features[0, :]
print(f"\nPearson R scores (feature type 0):")
print(f"  Mean: {np.mean(pearson_r_scores):.6f}")
print(f"  Std:  {np.std(pearson_r_scores):.6f}")
print(f"  Min:  {np.min(pearson_r_scores):.6f}")
print(f"  Max:  {np.max(pearson_r_scores):.6f}")

# Load standardization params
if os.path.exists(f"{lionheart_output_dir}/standardization_params.json"):
    with open(f"{lionheart_output_dir}/standardization_params.json") as f:
        std_params = json.load(f)
    print(f"\nStandardization parameters:")
    print(f"  Mean: {std_params['mean']:.6f}")
    print(f"  Std:  {std_params['std']:.6f}")

# Load GC correction factors
if os.path.exists(f"{lionheart_output_dir}/gc_correction_factor.npy"):
    lionheart_gc_factors = np.load(f"{lionheart_output_dir}/gc_correction_factor.npy")
    print(f"\nGC correction factors:")
    print(f"  Shape: {lionheart_gc_factors.shape}")
    print(f"  Mean: {np.nanmean(lionheart_gc_factors):.6f}")
    print(f"  Min:  {np.nanmin(lionheart_gc_factors):.6f}")
    print(f"  Max:  {np.nanmax(lionheart_gc_factors):.6f}")

# Load coverage stats
if os.path.exists(f"{lionheart_output_dir}/coverage_stats.json"):
    with open(f"{lionheart_output_dir}/coverage_stats.json") as f:
        coverage_stats = json.load(f)
    print(f"\nCoverage statistics:")
    for key, value in coverage_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

# Load our output
print("\n" + "-" * 70)
print("Our Simplified Pipeline Output:")
print("-" * 70)

if os.path.exists(our_output_file):
    our_features = pd.read_csv(our_output_file)
    print(f"Number of features: {len(our_features)}")
    print(f"\nFeatures:")
    for _, row in our_features.iterrows():
        print(f"  {row['feature']}: {row['value']:.6f}")
    
    # Extract correlation features
    corr_features = our_features[our_features['feature'].str.startswith('corr_')]
    if len(corr_features) > 0:
        print(f"\nCorrelation features (our version):")
        for _, row in corr_features.iterrows():
            print(f"  {row['feature']}: {row['value']:.6f}")
        
        # Compare with LIONHEART (note: different cell types, so direct comparison not possible)
        print(f"\n  Note: Our version has {len(corr_features)} cell types")
        print(f"        LIONHEART has 898 cell types")
        print(f"        Direct comparison not possible due to different cell type sets")
        
        # Compare statistics
        our_corr_mean = corr_features['value'].mean()
        our_corr_std = corr_features['value'].std()
        print(f"\n  Our correlation statistics:")
        print(f"    Mean: {our_corr_mean:.6f}")
        print(f"    Std:  {our_corr_std:.6f}")
        print(f"    Min:  {corr_features['value'].min():.6f}")
        print(f"    Max:  {corr_features['value'].max():.6f}")
        
        print(f"\n  LIONHEART Pearson R statistics:")
        print(f"    Mean: {np.mean(pearson_r_scores):.6f}")
        print(f"    Std:  {np.std(pearson_r_scores):.6f}")
        print(f"    Min:  {np.min(pearson_r_scores):.6f}")
        print(f"    Max:  {np.max(pearson_r_scores):.6f}")
        
        # Compare ranges
        print(f"\n  Comparison:")
        print(f"    Our correlations are in range: [{corr_features['value'].min():.4f}, {corr_features['value'].max():.4f}]")
        print(f"    LIONHEART scores are in range: [{np.min(pearson_r_scores):.4f}, {np.max(pearson_r_scores):.4f}]")
        
        if abs(our_corr_mean) < 0.1 and abs(np.mean(pearson_r_scores)) < 0.1:
            print(f"    ✓ Both have small mean values (close to 0), which is expected")
        if abs(our_corr_std) < 1.0 and abs(np.std(pearson_r_scores)) < 1.0:
            print(f"    ✓ Both have reasonable standard deviations")
else:
    print(f"ERROR: Our output file not found: {our_output_file}")
    print("Please run: python run_pipeline.py")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("1. LIONHEART output:")
print("   - Shape: (10, 898) - 10 feature types, 898 cell types")
print("   - Full pipeline with all corrections")
print("   - All autosomes (chr1-chr22)")
print()
print("2. Our simplified pipeline output:")
print("   - 9 features total")
print("   - 3 cell types (Tcell, Monocyte, Liver)")
print("   - Single chromosome (chr21)")
print("   - GC correction implemented")
print("   - No megabin normalization yet")
print()
print("3. Key differences:")
print("   - Cell type count: 3 vs 898")
print("   - Feature types: 1 (Pearson R) vs 10")
print("   - Chromosomes: 1 (chr21) vs 22 (chr1-chr22)")
print("   - Megabin normalization: Not implemented vs Implemented")
print("=" * 70)


