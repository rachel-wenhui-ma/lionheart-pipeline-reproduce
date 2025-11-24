# src/utils.py
import numpy as np
import pysam
import pandas as pd
from pathlib import Path
from typing import Optional

def summarize_coverage(coverage):
    return {
        "cov_mean": float(np.mean(coverage)),
        "cov_std": float(np.std(coverage)),
        "cov_median": float(np.median(coverage)),
    }


def calculate_gc_content(sequence):
    """
    Calculate GC content (percentage of G and C bases) for a DNA sequence.
    """
    seq_upper = sequence.upper()
    if len(seq_upper) == 0:
        return 0.0
    gc_count = seq_upper.count('G') + seq_upper.count('C')
    return gc_count / len(seq_upper)


def calculate_gc_content_per_bin(reference_fasta, chrom, bin_size=100):
    """
    Calculate GC content for each bin from reference genome.
    
    Returns:
        gc_content: array of GC content (0-1) for each bin
    """
    ref = pysam.FastaFile(reference_fasta)
    try:
        chrom_len = ref.get_reference_length(chrom)
        n_bins = chrom_len // bin_size + 1
        gc_content = np.zeros(n_bins, dtype=np.float64)
        
        for bin_id in range(n_bins):
            start = bin_id * bin_size
            end = min(start + bin_size, chrom_len)
            if start >= chrom_len:
                break
            
            seq = ref.fetch(chrom, start, end)
            gc_content[bin_id] = calculate_gc_content(seq)
        
        return gc_content
    finally:
        ref.close()


def load_lionheart_gc_content(
    resources_dir: Path,
    chrom: str,
    exclude_indices_10bp: Optional[np.ndarray] = None,
    bin_size: int = 100,
) -> tuple:
    """
    Load GC content from LIONHEART's parquet file (matching LIONHEART's approach).
    
    This function replicates LIONHEART's _load_bins_and_exclude behavior:
    - Loads GC from parquet file
    - Applies exclude indices
    - Rounds GC to 2 decimal places (as LIONHEART does)
    - Aggregates 10bp GC to 100bp if needed
    
    Parameters:
        resources_dir: Path to LIONHEART resources directory
        chrom: Chromosome name (e.g., "chr21")
        exclude_indices_10bp: Optional array of 10bp indices to exclude
        bin_size: Target bin size (100 for 100bp bins)
    
    Returns:
        include_indices_100bp: Array of 100bp bin indices (after exclude)
        gc_content_100bp: Array of GC content (0-1) for each included bin
    """
    # Load from parquet file (matching LIONHEART's _load_bins_and_exclude)
    bins_path = resources_dir / "bin_indices_by_chromosome" / f"{chrom}.parquet"
    if not bins_path.exists():
        raise FileNotFoundError(f"GC bins file not found: {bins_path}")
    
    df = pd.read_parquet(
        path=bins_path,
        engine="pyarrow",
        columns=["idx", "GC"],
    )
    
    # Apply exclude (matching LIONHEART)
    if exclude_indices_10bp is not None and exclude_indices_10bp.size > 0:
        df = df[~df["idx"].isin(exclude_indices_10bp)].reset_index(drop=True)
    
    include_indices_10bp = df["idx"].to_numpy().astype(np.int64)
    gc_content_10bp = np.round(
        df["GC"].to_numpy().astype(np.float64),
        decimals=2,  # NOTE: Rounding required (as per LIONHEART comment)
    )
    
    # Aggregate 10bp to 100bp
    if bin_size == 10:
        # Return 10bp directly
        return include_indices_10bp, gc_content_10bp
    elif bin_size == 100:
        # Aggregate to 100bp: group by 100bp bin and take mean
        include_indices_100bp = np.unique(include_indices_10bp // 10)
        
        # Create mapping from 10bp indices to 100bp bins
        gc_100bp_dict = {}
        for i, idx_10bp in enumerate(include_indices_10bp):
            idx_100bp = idx_10bp // 10
            if idx_100bp not in gc_100bp_dict:
                gc_100bp_dict[idx_100bp] = []
            gc_100bp_dict[idx_100bp].append(gc_content_10bp[i])
        
        # Calculate mean GC per 100bp bin
        gc_content_100bp = np.zeros(len(include_indices_100bp), dtype=np.float32)
        for j, idx_100bp in enumerate(include_indices_100bp):
            if idx_100bp in gc_100bp_dict:
                gc_content_100bp[j] = np.mean(gc_100bp_dict[idx_100bp])
        
        return include_indices_100bp, gc_content_100bp
    else:
        raise ValueError(f"Unsupported bin_size: {bin_size} (only 10 or 100 supported)")

