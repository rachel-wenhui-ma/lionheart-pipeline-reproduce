# src/load_masks.py
"""
Load real LIONHEART cell type masks from sparse arrays.
Supports parallel loading for performance.
"""
import numpy as np
import scipy.sparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count


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
    Uses Numba-optimized version if available.
    
    Parameters:
        mask_10bp: Mask array at 10bp resolution
    
    Returns:
        Mask array at 100bp resolution
    """
    # Try to use optimized version
    try:
        from .optimized import aggregate_10bp_to_100bp_numba, NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            return aggregate_10bp_to_100bp_numba(mask_10bp)
    except ImportError:
        pass
    
    # Fallback to original implementation
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


def _load_single_mask_worker(args):
    """Worker function for parallel mask loading (must be at module level for pickle)."""
    cell_type, mask_path_str, bin_indices, bin_size = args
    from pathlib import Path
    mask_path = Path(mask_path_str)
    
    try:
        if not mask_path.exists():
            return (cell_type, None)
        
        # Import here to avoid circular imports
        import scipy.sparse
        
        # Load sparse array directly (avoiding function call for pickle)
        s = scipy.sparse.load_npz(mask_path).tocsr()
        if bin_indices is not None:
            if not (s.shape[1] == 1 and s.shape[0] > 1):
                return (cell_type, None)
            s = s[bin_indices, :]
        x = s.astype(np.float32, copy=False).toarray().ravel()
        if True:  # decimals >= 0
            x = np.round(x, decimals=2)
        mask_10bp = x
        
        # Return mask at requested bin size
        if bin_size == 10:
            # Direct use of 10bp mask (LIONHEART default)
            return (cell_type, mask_10bp)
        elif bin_size == 100:
            # Aggregate to 100bp if needed
            # CRITICAL: mask_10bp is already filtered by bin_indices (include_indices_10bp)
            # So we can directly aggregate: every 10 consecutive 10bp bins -> 1 100bp bin
            n_10bp = len(mask_10bp)
            n_100bp = (n_10bp + 9) // 10  # Round up
            
            try:
                from .optimized import aggregate_10bp_to_100bp_numba, NUMBA_AVAILABLE
                if NUMBA_AVAILABLE:
                    mask_100bp = aggregate_10bp_to_100bp_numba(mask_10bp)
                else:
                    # Fallback: manual aggregation
                    mask_100bp = np.zeros(n_100bp, dtype=np.float32)
                    for i in range(n_100bp):
                        start = i * 10
                        end = min(start + 10, n_10bp)
                        mask_100bp[i] = np.sum(mask_10bp[start:end])
            except ImportError:
                # Fallback: manual aggregation
                mask_100bp = np.zeros(n_100bp, dtype=np.float32)
                for i in range(n_100bp):
                    start = i * 10
                    end = min(start + 10, n_10bp)
                    mask_100bp[i] = np.sum(mask_10bp[start:end])
            
            return (cell_type, mask_100bp)
        else:
            # For other bin sizes, use direct aggregation
            n_bins = (len(mask_10bp) + bin_size // 10 - 1) // (bin_size // 10)
            mask_agg = np.zeros(n_bins, dtype=np.float32)
            for i in range(n_bins):
                start = i * (bin_size // 10)
                end = min(start + (bin_size // 10), len(mask_10bp))
                mask_agg[i] = np.sum(mask_10bp[start:end])
            return (cell_type, mask_agg)
    except Exception as e:
        return (cell_type, None)


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
    # Try different possible paths, including fallback to original LIONHEART resources
    possible_paths = [
        resources_dir / "chromatin_masks" / mask_type / "sparse_overlaps_by_chromosome",
        resources_dir / f"{mask_type}_masks",
        resources_dir / mask_type / "masks",
    ]
    
    # Fallback to original LIONHEART resources if local copy doesn't have masks
    import os
    if os.path.exists("/home/mara/lionheart_work/resources"):
        from pathlib import Path
        original_resources = Path("/home/mara/lionheart_work/resources")
        possible_paths.append(
            original_resources / "chromatin_masks" / mask_type / "sparse_overlaps_by_chromosome"
        )
    
    masks_base_dir = None
    for path in possible_paths:
        if path.exists():
            masks_base_dir = path
            break
    
    if masks_base_dir is None:
        raise FileNotFoundError(
            f"Could not find masks directory. Tried: {possible_paths}"
        )
    
    # Use parallel loading for large batches
    use_parallel = len(cell_types) > 20
    
    if use_parallel:
        # Try joblib first (same as LIONHEART), fallback to multiprocessing
        try:
            from joblib import Parallel, delayed
            num_workers = min(cpu_count(), 8)  # Limit to 8 workers
            print(f"    Loading {len(cell_types)} masks in parallel using joblib ({num_workers} workers)...")
            
            def _load_single_mask_joblib(cell_type):
                """Wrapper for joblib parallel loading."""
                mask_path = masks_base_dir / cell_type / f"{chrom}.npz"
                if not mask_path.exists():
                    return (cell_type, None)
                try:
                    import scipy.sparse
                    s = scipy.sparse.load_npz(mask_path).tocsr()
                    if bin_indices is not None:
                        if not (s.shape[1] == 1 and s.shape[0] > 1):
                            return (cell_type, None)
                        s = s[bin_indices, :]
                    x = s.astype(np.float32, copy=False).toarray().ravel()
                    x = np.round(x, decimals=2)
                    
                    if bin_size == 10:
                        return (cell_type, x)
                    elif bin_size == 100:
                        from .optimized import aggregate_10bp_to_100bp_numba, NUMBA_AVAILABLE
                        if NUMBA_AVAILABLE:
                            mask_100bp = aggregate_10bp_to_100bp_numba(x)
                        else:
                            n_10bp = len(x)
                            n_100bp = (n_10bp + 9) // 10
                            mask_100bp = np.zeros(n_100bp, dtype=np.float32)
                            for i in range(n_100bp):
                                start = i * 10
                                end = min(start + 10, n_10bp)
                                mask_100bp[i] = np.sum(x[start:end])
                        return (cell_type, mask_100bp)
                    else:
                        # For other bin sizes
                        n_bins = (len(x) + bin_size // 10 - 1) // (bin_size // 10)
                        mask_agg = np.zeros(n_bins, dtype=np.float32)
                        for i in range(n_bins):
                            start = i * (bin_size // 10)
                            end = min(start + (bin_size // 10), len(x))
                            mask_agg[i] = np.sum(x[start:end])
                        return (cell_type, mask_agg)
                except Exception as e:
                    return (cell_type, None)
            
            # Add progress tracking
            print(f"    Starting parallel mask loading...")
            results = Parallel(n_jobs=num_workers, verbose=1)(
                delayed(_load_single_mask_joblib)(cell_type) for cell_type in cell_types
            )
            masks = {cell_type: mask for cell_type, mask in results if mask is not None}
            print(f"    Loaded {len(masks)} masks successfully")
        except ImportError:
            # Fallback to multiprocessing
            num_workers = min(cpu_count(), 8)  # Limit to 8 workers
            print(f"    Loading {len(cell_types)} masks in parallel using multiprocessing ({num_workers} workers)...")
            
            # Prepare arguments (convert Path to string for pickle)
            args_list = [
                (cell_type, str(masks_base_dir / cell_type / f"{chrom}.npz"), bin_indices, bin_size)
                for cell_type in cell_types
            ]
            
            # Load in parallel
            # Use chunksize to avoid memory issues with large batches
            chunksize = max(1, len(args_list) // (num_workers * 4))
            with Pool(num_workers) as pool:
                results = pool.map(_load_single_mask_worker, args_list, chunksize=chunksize)
            
            # Collect results
            masks = {cell_type: mask for cell_type, mask in results if mask is not None}
        
    else:
        # Sequential loading (for small batches)
        masks = {}
        total = len(cell_types)
        for idx, cell_type in enumerate(cell_types):
            # Path to chromosome-specific mask
            mask_path = masks_base_dir / cell_type / f"{chrom}.npz"
            
            if not mask_path.exists():
                if total <= 10:
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
                
                # Return mask at requested bin size
                if bin_size == 10:
                    # Direct use of 10bp mask (LIONHEART default)
                    masks[cell_type] = mask_10bp
                elif bin_size == 100:
                    # Aggregate to 100bp
                    mask_100bp = aggregate_10bp_to_100bp(mask_10bp)
                    masks[cell_type] = mask_100bp
                else:
                    # For other bin sizes, use direct aggregation
                    n_bins = (len(mask_10bp) + bin_size // 10 - 1) // (bin_size // 10)
                    mask_agg = np.zeros(n_bins, dtype=np.float32)
                    for i in range(n_bins):
                        start = i * (bin_size // 10)
                        end = min(start + (bin_size // 10), len(mask_10bp))
                        mask_agg[i] = np.sum(mask_10bp[start:end])
                    masks[cell_type] = mask_agg
                
            except Exception as e:
                if total <= 10:
                    print(f"Error loading mask for {cell_type} {chrom}: {e}")
                continue
    
    return masks

