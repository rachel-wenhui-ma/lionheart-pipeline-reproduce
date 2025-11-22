#!/usr/bin/env python
"""Test loading real LIONHEART masks"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
from src.load_masks import load_cell_type_list, load_cell_type_masks

# Check resources directory
if os.path.exists("/mnt/d"):
    base_path = "/mnt/d/MADS/25 Fall/CSC 527/Final project"
    resources_dir = Path("/home/mara/lionheart_work/resources")
    if not resources_dir.exists():
        resources_dir = Path(f"{base_path}/lionheart_resources")
else:
    resources_dir = Path("../lionheart_resources")

print(f"Checking resources_dir: {resources_dir}")
print(f"Exists: {resources_dir.exists()}")

if resources_dir.exists():
    try:
        # Test loading cell type list
        print("\n1. Testing cell type list loading...")
        df = load_cell_type_list(resources_dir, "DNase")
        print(f"   Loaded {len(df)} DNase cell types")
        print(f"   First 5: {df.head(5)['cell_type'].tolist()}")
        
        # Test loading masks for chr21
        print("\n2. Testing mask loading for chr21...")
        test_cell_types = df.head(3)["cell_type"].tolist()
        print(f"   Loading masks for: {test_cell_types}")
        
        masks = load_cell_type_masks(
            resources_dir=resources_dir,
            mask_type="DNase",
            chrom="chr21",
            cell_types=test_cell_types,
            bin_size=100,
        )
        
        print(f"   Successfully loaded {len(masks)} masks")
        for cell_type, mask in masks.items():
            print(f"     {cell_type}: shape={mask.shape}, non-zero={np.sum(mask > 0)}, mean={np.mean(mask):.6f}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n✗ Resources directory not found: {resources_dir}")
    print("   Please check the path or set use_real_masks=False in run_pipeline.py")

