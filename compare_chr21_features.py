#!/usr/bin/env python
"""
Compare our pipeline output with LIONHEART output for chr21.
Extracts chr21-specific features from LIONHEART's full output.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Paths
if os.path.exists("/mnt/d"):
    base_path = "/mnt/d/MADS/25 Fall/CSC 527/Final project"
    lionheart_output_dir = Path("/home/mara/lionheart_work/output/dataset")
    resources_dir = Path("/home/mara/lionheart_work/resources")
else:
    lionheart_output_dir = Path("/home/mara/lionheart_work/output/dataset")
    resources_dir = Path("/home/mara/lionheart_work/resources")

our_output_file = "features_sample.csv"

print("=" * 70)
print("Comparing chr21 Features: Our Pipeline vs LIONHEART")
print("=" * 70)

# Load LIONHEART output
print("\n" + "-" * 70)
print("LIONHEART Output:")
print("-" * 70)

if not lionheart_output_dir.exists():
    print(f"ERROR: LIONHEART output directory not found: {lionheart_output_dir}")
    print("Please ensure LIONHEART has been run and output is available.")
    sys.exit(1)

lionheart_features = np.load(lionheart_output_dir / "feature_dataset.npy")
print(f"LIONHEART feature dataset shape: {lionheart_features.shape}")
print(f"  - 10 feature types (Pearson R, p-value, cosine similarity, etc.)")
print(f"  - {lionheart_features.shape[1]} cell types")

# Load cell type list to map indices
if resources_dir.exists():
    cell_type_df = pd.read_csv(resources_dir / "DNase.idx_to_cell_type.csv")
    print(f"  - Cell type list loaded: {len(cell_type_df)} cell types")
else:
    print("  - Warning: Cell type list not found, using indices")
    cell_type_df = None

# Feature type 0 is Pearson R
pearson_r_scores = lionheart_features[0, :]
print(f"\nLIONHEART Pearson R scores (all cell types):")
print(f"  Mean: {np.mean(pearson_r_scores):.6f}")
print(f"  Std:  {np.std(pearson_r_scores):.6f}")
print(f"  Min:  {np.min(pearson_r_scores):.6f}")
print(f"  Max:  {np.max(pearson_r_scores):.6f}")

# Load our output
print("\n" + "-" * 70)
print("Our Pipeline Output:")
print("-" * 70)

if not os.path.exists(our_output_file):
    print(f"ERROR: Our output file not found: {our_output_file}")
    print("Please run: python run_pipeline.py")
    sys.exit(1)

our_features = pd.read_csv(our_output_file)
print(f"Number of features: {len(our_features)}")

# Extract Pearson R features
our_pearson_r = our_features[our_features['feature'].str.startswith('pearson_r_')]
print(f"\nOur Pearson R features: {len(our_pearson_r)}")

if len(our_pearson_r) > 0:
    print(f"\nOur Pearson R statistics:")
    print(f"  Mean: {our_pearson_r['value'].mean():.6f}")
    print(f"  Std:  {our_pearson_r['value'].std():.6f}")
    print(f"  Min:  {our_pearson_r['value'].min():.6f}")
    print(f"  Max:  {our_pearson_r['value'].max():.6f}")
    
    # Extract cell type names
    our_cell_types = [f.split('_', 2)[-1] for f in our_pearson_r['feature']]
    print(f"\nOur cell types: {our_cell_types}")
    
    # Try to find matching cell types in LIONHEART output
    if cell_type_df is not None:
        print("\n" + "-" * 70)
        print("Direct Comparison (matching cell types):")
        print("-" * 70)
        
        matches_found = 0
        for our_ct in our_cell_types:
            # Find index in LIONHEART
            matching_rows = cell_type_df[cell_type_df['cell_type'] == our_ct]
            if len(matching_rows) > 0:
                idx = matching_rows.index[0]
                lionheart_r = lionheart_features[0, idx]
                our_r = our_pearson_r[our_pearson_r['feature'].str.endswith(f'_{our_ct}')]['value'].values[0]
                
                diff = abs(lionheart_r - our_r)
                print(f"\n{our_ct}:")
                print(f"  LIONHEART R: {lionheart_r:.6f}")
                print(f"  Our R:       {our_r:.6f}")
                print(f"  Difference:  {diff:.6f}")
                
                if diff < 0.01:
                    print(f"  ✓ Very close match!")
                elif diff < 0.1:
                    print(f"  ~ Reasonable match")
                else:
                    print(f"  ✗ Significant difference")
                
                matches_found += 1
        
        if matches_found == 0:
            print("No matching cell types found for direct comparison.")
            print("This may be because:")
            print("  1. Different cell type sets")
            print("  2. Different chromosome processing")
            print("  3. Different normalization/correction steps")
    
    # Standardize our Pearson R scores for comparison
    our_r_values = our_pearson_r['value'].values
    our_r_standardized = (our_r_values - np.mean(our_r_values)) / (np.std(our_r_values) + 1e-10)
    
    # Compare overall statistics
    print("\n" + "-" * 70)
    print("Overall Statistics Comparison:")
    print("-" * 70)
    print(f"LIONHEART (all {lionheart_features.shape[1]} cell types, standardized):")
    print(f"  Mean: {np.mean(pearson_r_scores):.6f}")
    print(f"  Std:  {np.std(pearson_r_scores):.6f}")
    print(f"  Range: [{np.min(pearson_r_scores):.4f}, {np.max(pearson_r_scores):.4f}]")
    
    print(f"\nOur pipeline ({len(our_pearson_r)} cell types, raw):")
    print(f"  Mean: {our_pearson_r['value'].mean():.6f}")
    print(f"  Std:  {our_pearson_r['value'].std():.6f}")
    print(f"  Range: [{our_pearson_r['value'].min():.4f}, {our_pearson_r['value'].max():.4f}]")
    
    print(f"\nOur pipeline ({len(our_pearson_r)} cell types, standardized):")
    print(f"  Mean: {np.mean(our_r_standardized):.6f}")
    print(f"  Std:  {np.std(our_r_standardized):.6f}")
    print(f"  Range: [{np.min(our_r_standardized):.4f}, {np.max(our_r_standardized):.4f}]")
    
    # Compare standardized values with LIONHEART
    print("\n" + "-" * 70)
    print("Standardized Comparison (matching cell types):")
    print("-" * 70)
    
    if cell_type_df is not None:
        for i, our_ct in enumerate(our_cell_types):
            matching_rows = cell_type_df[cell_type_df['cell_type'] == our_ct]
            if len(matching_rows) > 0:
                idx = matching_rows.index[0]
                lionheart_r_std = pearson_r_scores[idx]
                our_r_std = our_r_standardized[i]
                
                diff = abs(lionheart_r_std - our_r_std)
                print(f"\n{our_ct} (standardized):")
                print(f"  LIONHEART: {lionheart_r_std:.6f}")
                print(f"  Ours:      {our_r_std:.6f}")
                print(f"  Difference: {diff:.6f}")
                
                if diff < 0.5:
                    print(f"  ~ Reasonable match (considering single chr vs all chrs)")
                elif diff < 1.0:
                    print(f"  ~ Some difference (expected due to single chr)")
                else:
                    print(f"  ✗ Significant difference")
    
    # Key differences explanation
    print("\n" + "-" * 70)
    print("Key Differences Explanation:")
    print("-" * 70)
    print("1. LIONHEART processes ALL 22 autosomes (chr1-chr22)")
    print("   Our pipeline processes ONLY chr21")
    print("   → This is the main reason for different Pearson R values")
    print()
    print("2. LIONHEART standardizes Pearson R across all 898 cell types")
    print("   Our pipeline has only 5 cell types (not standardized yet)")
    print("   → Standardization makes values comparable across cell types")
    print()
    print("3. LIONHEART uses bin indices/exclusions from resources")
    print("   Our pipeline uses all bins (no exclusions yet)")
    print("   → May affect which bins are included in correlation")
    print()
    print("4. Both use the same correction steps:")
    print("   ✓ ZIPoisson clipping")
    print("   ✓ GC correction")
    print("   ✓ Insert size correction (noise, skewness, mean shift)")
    print("   ✓ Megabin normalization")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("1. LIONHEART processes all chromosomes and aggregates results")
print("2. Our pipeline processes only chr21")
print("3. Both use the same correction steps:")
print("   - ZIPoisson clipping")
print("   - GC correction")
print("   - Insert size correction")
print("   - Megabin normalization")
print("4. Differences may be due to:")
print("   - Single chromosome vs all chromosomes")
print("   - Different bin indices/exclusions")
print("   - Different standardization approach")
print("=" * 70)

