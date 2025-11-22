# src/compute_coverage.py
import pysam
import numpy as np
from scipy.stats import binned_statistic
from typing import Optional
from typing import Optional

def compute_coverage(bam_path, chrom="chr21", bin_size=100):
    """
    Compute simple coverage per bin. 
    (Real LIONHEART uses mosdepth; this is simplified)
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    chrom_len = bam.get_reference_length(chrom)

    n_bins = chrom_len // bin_size + 1
    coverage = np.zeros(n_bins, dtype=np.int32)

    for read in bam.fetch(chrom):
        start = read.reference_start
        if start < 0:
            continue
        
        bin_id = start // bin_size
        if 0 <= bin_id < n_bins:
            coverage[bin_id] += 1

    bam.close()
    return coverage


def normalize_coverage(coverage):
    """Optional simple normalization."""
    mean = np.mean(coverage)
    std = np.std(coverage)
    if std == 0:
        return coverage - mean
    return (coverage - mean) / std


def calculate_gc_correction_factors(coverage, gc_content, num_bins=20):
    """
    Calculate GC correction factors following LIONHEART algorithm.
    
    Parameters:
        coverage: array of coverage values
        gc_content: array of GC content (0-1) for each bin
        num_bins: number of GC bins for correction
    
    Returns:
        bin_edges: edges of GC bins
        correction_factors: correction factor for each GC bin
    """
    # Create GC bins (0.0 to 1.0)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    
    # Calculate mean coverage for each GC bin
    bin_means, bin_edges_result, bin_numbers = binned_statistic(
        gc_content, coverage, statistic='mean', bins=bin_edges
    )
    
    # Calculate correction factors: overall_mean / bin_mean
    overall_mean = np.nanmean(coverage)
    correction_factors = overall_mean / (bin_means + 1e-10)  # Avoid division by zero
    
    # Set NaN factors to 1.0 (no correction)
    correction_factors[np.isnan(correction_factors)] = 1.0
    
    # Normalize factors to center around 1.0
    correction_factors /= np.nanmean(correction_factors)
    
    return bin_edges_result, correction_factors


def correct_gc_bias(coverage, gc_content, bin_edges, correction_factors):
    """
    Apply GC correction to coverage following LIONHEART algorithm.
    
    Parameters:
        coverage: array of coverage values to correct
        gc_content: array of GC content (0-1) for each bin
        bin_edges: edges of GC bins
        correction_factors: correction factor for each GC bin
    
    Returns:
        corrected_coverage: GC-corrected coverage
    """
    # Find which GC bin each coverage value belongs to
    bin_indices = np.digitize(gc_content, bins=bin_edges, right=False) - 1
    
    # Clip bin indices to valid range
    bin_indices = np.clip(bin_indices, 0, len(correction_factors) - 1)
    
    # Get correction factor for each bin
    corrections = correction_factors[bin_indices]
    
    # Avoid zero-division
    corrections[corrections == 0] = 1.0
    
    # Apply correction: divide coverage by correction factor
    corrected_coverage = coverage.astype(np.float64) / corrections
    
    # Clip negative values to 0
    corrected_coverage[corrected_coverage < 0] = 0.0
    
    return corrected_coverage


def normalize_megabins_simple(
    coverage: np.ndarray,
    bin_size: int = 100,
    mbin_size: int = 5_000_000,
    stride: int = 500_000,
    center: Optional[str] = None,
    scale: Optional[str] = "mean",
) -> np.ndarray:
    """
    Normalize coverage using megabins with multiple strides (simplified LIONHEART version).
    
    Parameters:
        coverage: Coverage array
        bin_size: Size of each bin in bp
        mbin_size: Size of megabin in bp (default 5MB from LIONHEART)
        stride: Stride size in bp (default 500KB from LIONHEART)
        center: How to center ('mean' or 'median', or None)
        scale: How to scale ('mean', 'median', 'std', 'iqr', or None)
    
    Returns:
        normalized_coverage: Normalized coverage array
    """
    if center is None and scale is None:
        raise ValueError("At least one of {center, scale} must be specified.")
    
    n_bins = len(coverage)
    bins_per_mbin = mbin_size // bin_size
    bins_per_stride = stride // bin_size
    
    # Number of strides needed
    num_stridings = int(np.ceil(bins_per_mbin / bins_per_stride))
    
    # Calculate megabin averages for each stride
    mbin_averages = np.zeros((n_bins, num_stridings))
    
    for striding in range(num_stridings):
        # Offset for this stride
        start_offset = bins_per_mbin - bins_per_stride * striding if striding > 0 else 0
        
        # Calculate megabin indices with this stride
        mbin_indices = (np.arange(n_bins) + start_offset) // bins_per_mbin
        
        # Calculate mean for each megabin
        for mbin_idx in range(int(np.max(mbin_indices)) + 1):
            mbin_mask = (mbin_indices == mbin_idx)
            mbin_values = coverage[mbin_mask]
            
            if len(mbin_values) > 0:
                # Clip outliers (top 1%)
                if len(mbin_values) > 100:
                    q99 = np.nanquantile(mbin_values, 0.99)
                    mbin_values_clipped = np.clip(mbin_values, None, q99)
                else:
                    mbin_values_clipped = mbin_values
                
                # Calculate mean
                mbin_mean = np.nanmean(mbin_values_clipped)
                mbin_averages[mbin_mask, striding] = mbin_mean
    
    # Average across all strides for each bin
    mbin_overall_mean = np.nanmean(mbin_averages, axis=1)
    
    # Apply normalization
    normalized_coverage = coverage.copy().astype(np.float64)
    
    if scale == "mean":
        # Scale by mean (LIONHEART default)
        normalized_coverage = normalized_coverage / (mbin_overall_mean + 1e-10)
        # Normalize to keep original scale
        scale_factor = np.nanmean(coverage) / np.nanmean(normalized_coverage)
        normalized_coverage = normalized_coverage * scale_factor
    elif scale == "median":
        # Scale by median
        mbin_overall_median = np.nanmedian(mbin_averages, axis=1)
        normalized_coverage = normalized_coverage / (mbin_overall_median + 1e-10)
        scale_factor = np.nanmedian(coverage) / np.nanmedian(normalized_coverage)
        normalized_coverage = normalized_coverage * scale_factor
    elif center == "mean":
        # Center by mean
        normalized_coverage = normalized_coverage - mbin_overall_mean
    elif center == "median":
        # Center by median
        mbin_overall_median = np.nanmedian(mbin_averages, axis=1)
        normalized_coverage = normalized_coverage - mbin_overall_median
    
    # Clip negative values
    normalized_coverage[normalized_coverage < 0] = 0.0
    
    return normalized_coverage

