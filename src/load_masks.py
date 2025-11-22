# src/load_masks.py
"""
Load real LIONHEART cell type masks from sparse arrays.
"""
import numpy as np
import scipy.sparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_sparse_mask(
    mask_path: Path,
    indices: Optional[np.ndarray] = None,
    decimals: int = 2,
    dtype=np.float32,
) -> np.ndarray:
    """
    Load a sparse array (.npz) and convert to dense array.
    
    Parameters:
        mask_path: Path to .npz file
        indices: Optional indices to subset (for bin filtering)
        decimals: Number of decimals for rounding
        dtype: Output data type
    
    Returns:
        Dense array of mask values
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load sparse array
    s = scipy.sparse.load_npz(mask_path).tocsr()
    
    # Subset to specified indices if provided
    if indices is not None:
        if not (s.shape[1] == 1 and s.shape[0] > 1):
            raise ValueError(
                f"Sparse array had unexpected shape: {s.shape}. Expected (>1, 1)."
            )
        s = s[indices, :]
    
    # Convert to dense array
    x = s.astype(dtype, copy=False).toarray().ravel()
    
    # Round to avoid rounding errors
    if decimals >= 0:
        x = np.round(x, decimals=decimals)
    
    return x


def aggregate_10bp_to_100bp(mask_10bp: np.ndarray) -> np.ndarray:
    """
    Aggregate 10bp mask to 100bp by summing values.
    
    Parameters:
        mask_10bp: Mask array at 10bp resolution
    
    Returns:
        Mask array at 100bp resolution
    """
    n_10bp = len(mask_10bp)
    n_100bp = (n_10bp + 9) // 10  # Round up
    
    mask_100bp = np.zeros(n_100bp, dtype=np.float32)
    
    for i in range(n_100bp):
        start_10bp = i * 10
        end_10bp = min(start_10bp + 10, n_10bp)
        mask_100bp[i] = np.sum(mask_10bp[start_10bp:end_10bp])
    
    return mask_100bp


def load_cell_type_list(resources_dir: Path, mask_type: str) -> pd.DataFrame:
    """
    Load cell type index mapping from CSV.
    
    Parameters:
        resources_dir: Path to LIONHEART resources directory
        mask_type: "DNase" or "ATAC"
    
    Returns:
        DataFrame with cell_type column and index
    """
    csv_path = resources_dir / f"{mask_type}.idx_to_cell_type.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cell type list not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def load_cell_type_masks(
    resources_dir: Path,
    mask_type: str,
    chrom: str,
    cell_types: Optional[List[str]] = None,
    bin_indices: Optional[np.ndarray] = None,
    bin_size: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Load masks for multiple cell types for a chromosome.
    
    Parameters:
        resources_dir: Path to LIONHEART resources directory
        mask_type: "DNase" or "ATAC"
        chrom: Chromosome name (e.g., "chr21")
        cell_types: List of cell type names (None = all)
        bin_indices: Optional indices to subset bins
        bin_size: Target bin size (100bp, will aggregate from 10bp)
    
    Returns:
        Dictionary mapping cell_type -> mask array
    """
    # Load cell type list
    cell_type_df = load_cell_type_list(resources_dir, mask_type)
    
    # Get cell types to load
    if cell_types is None:
        cell_types = cell_type_df["cell_type"].tolist()
    
    # Base path for masks (LIONHEART structure)
    # Try different possible paths
    possible_paths = [
        resources_dir / "chromatin_masks" / mask_type / "sparse_overlaps_by_chromosome",
        resources_dir / f"{mask_type}_masks",
        resources_dir / mask_type / "masks",
    ]
    
    masks_base_dir = None
    for path in possible_paths:
        if path.exists():
            masks_base_dir = path
            break
    
    if masks_base_dir is None:
        raise FileNotFoundError(
            f"Could not find masks directory. Tried: {possible_paths}"
        )
    
    masks = {}
    for cell_type in cell_types:
        # Path to chromosome-specific mask
        mask_path = masks_base_dir / cell_type / f"{chrom}.npz"
        
        if not mask_path.exists():
            print(f"Warning: Mask not found for {cell_type} {chrom}, skipping")
            continue
        
        try:
            # Load 10bp mask
            mask_10bp = load_sparse_mask(
                mask_path,
                indices=bin_indices,
                decimals=2,
                dtype=np.float32,
            )
            
            # Aggregate to 100bp if needed
            if bin_size == 100:
                mask_100bp = aggregate_10bp_to_100bp(mask_10bp)
            else:
                # For other bin sizes, use direct aggregation
                n_bins = (len(mask_10bp) + bin_size // 10 - 1) // (bin_size // 10)
                mask_100bp = np.zeros(n_bins, dtype=np.float32)
                for i in range(n_bins):
                    start = i * (bin_size // 10)
                    end = min(start + (bin_size // 10), len(mask_10bp))
                    mask_100bp[i] = np.sum(mask_10bp[start:end])
            
            masks[cell_type] = mask_100bp
            
        except Exception as e:
            print(f"Error loading mask for {cell_type} {chrom}: {e}")
            continue
    
    return masks

