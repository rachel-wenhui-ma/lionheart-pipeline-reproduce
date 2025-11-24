# src/blacklist.py
"""
Blacklist/exclude functionality for removing problematic genomic regions.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def load_chrom_indices(path: Path) -> Dict[str, np.ndarray]:
    """
    Load chromosome indices from a file (.npz format used by LIONHEART).
    
    LIONHEART uses .npz format with keys like 'chr1', 'chr2', etc.
    Each key maps to an array of bin indices to exclude.
    
    Parameters:
        path: Path to exclude/blacklist file (.npz format)
    
    Returns:
        Dictionary mapping chromosome name -> array of bin indices to exclude
    """
    if not path.exists():
        raise FileNotFoundError(f"Exclude file not found: {path}")
    
    # LIONHEART uses .npz format
    try:
        chrom_to_indices_arr = np.load(path, allow_pickle=True)
        # Extract indices for each chromosome (chr1-chr22)
        result = {
            chrom: chrom_to_indices_arr[chrom].flatten().astype(np.int64)
            for chrom in [f"chr{i}" for i in range(1, 23)]
            if chrom in chrom_to_indices_arr.files
        }
        return result
    except Exception as e:
        raise ValueError(f"Could not parse exclude file {path}: {e}")


def load_exclude_bins(exclude_paths: Optional[List[Path]], chroms: List[str]) -> Dict[str, np.ndarray]:
    """
    Load exclude/blacklist bins for all chromosomes.
    
    Parameters:
        exclude_paths: List of paths to exclude files
        chroms: List of chromosome names
    
    Returns:
        Dictionary mapping chromosome -> array of bin indices to exclude
    """
    exclude_bins_by_chrom = {}
    
    if not exclude_paths or len(exclude_paths) == 0:
        # No exclude files, return empty dict
        for chrom in chroms:
            exclude_bins_by_chrom[chrom] = np.array([], dtype=np.int64)
        return exclude_bins_by_chrom
    
    # Load exclude indices from all files
    exclude_dicts = []
    for path in exclude_paths:
        try:
            exclude_dicts.append(load_chrom_indices(path))
        except Exception as e:
            print(f"Warning: Failed to load exclusion indices from {path}: {e}")
            continue
    
    # Combine exclude indices for each chromosome
    for chrom in chroms:
        excl_arrays = [
            excl_dict[chrom]
            for excl_dict in exclude_dicts
            if chrom in excl_dict.keys()
        ]
        
        if excl_arrays:
            # Combine and get unique indices
            exclude_bins_by_chrom[chrom] = np.unique(np.concatenate(excl_arrays))
        else:
            exclude_bins_by_chrom[chrom] = np.array([], dtype=np.int64)
    
    return exclude_bins_by_chrom


def apply_exclude_to_coverage(
    coverage: np.ndarray,
    exclude_indices: np.ndarray,
    bin_indices: Optional[np.ndarray] = None
) -> tuple:
    """
    Apply exclude/blacklist to coverage array.
    
    Parameters:
        coverage: Coverage array
        exclude_indices: Bin indices to exclude (in original bin space)
        bin_indices: Optional mapping from current indices to original bin indices
    
    Returns:
        (filtered_coverage, valid_mask)
    """
    if exclude_indices is None or len(exclude_indices) == 0:
        return coverage, np.ones(len(coverage), dtype=bool)
    
    if bin_indices is not None:
        # Map exclude_indices to current indices
        # exclude_indices are in original bin space
        # bin_indices maps current position -> original bin index
        valid_mask = ~np.isin(bin_indices, exclude_indices)
    else:
        # Direct exclusion (assume exclude_indices are in current space)
        valid_mask = ~np.isin(np.arange(len(coverage)), exclude_indices)
    
    filtered_coverage = coverage[valid_mask]
    
    return filtered_coverage, valid_mask

