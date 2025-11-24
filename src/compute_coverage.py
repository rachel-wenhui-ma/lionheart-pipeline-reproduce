# src/compute_coverage.py
import pysam
import numpy as np
from scipy.stats import binned_statistic
from typing import Optional
from typing import Optional

def compute_coverage(bam_path, chrom="chr21", bin_size=100, mapq_threshold=20, use_overlap=True):
    """
    Compute coverage per bin using overlap fraction (matching mosdepth behavior).
    
    Parameters:
        bam_path: Path to BAM file
        chrom: Chromosome name
        bin_size: Bin size in bp (default 100)
        mapq_threshold: Minimum MAPQ score (default 20, matching LIONHEART)
        use_overlap: If True, use overlap fraction (mosdepth-like). If False, count reads.
    
    Returns:
        coverage: Coverage array (float if use_overlap, int otherwise)
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    chrom_len = bam.get_reference_length(chrom)

    n_bins = chrom_len // bin_size + 1
    
    if use_overlap:
        # Use overlap fraction (matching mosdepth behavior)
        coverage = np.zeros(n_bins, dtype=np.float32)
        
        for read in bam.fetch(chrom):
            # Apply MAPQ filter (matching LIONHEART's --mapq 20)
            if read.mapping_quality < mapq_threshold:
                continue
            
            read_start = read.reference_start
            read_end = read.reference_end
            
            if read_start < 0 or read_end is None:
                continue
            
            # Find all bins this read overlaps
            start_bin = read_start // bin_size
            end_bin = read_end // bin_size if read_end else start_bin
            
            # Calculate overlap fraction for each overlapping bin
            for bin_id in range(start_bin, min(end_bin + 1, n_bins)):
                bin_start = bin_id * bin_size
                bin_end = (bin_id + 1) * bin_size
                
                # Calculate overlap
                overlap_start = max(read_start, bin_start)
                overlap_end = min(read_end, bin_end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                # Add overlap fraction (overlap_length / bin_size)
                # This matches mosdepth's behavior: coverage = sum of overlaps / bin_size
                coverage[bin_id] += overlap_length / bin_size
    else:
        # Simple read counting (original method)
        coverage = np.zeros(n_bins, dtype=np.int32)
        
        for read in bam.fetch(chrom):
            if read.mapping_quality < mapq_threshold:
                continue
            
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
    
    LIONHEART uses: bin_means (average coverage per GC bin), then normalize.
    Correction factors are normalized bin_means.
    When applied: coverage / correction_factors
    
    Parameters:
        coverage: array of coverage values
        gc_content: array of GC content (0-1) for each bin
        num_bins: number of GC bins for correction
    
    Returns:
        bin_edges: edges of GC bins
        correction_factors: correction factor for each GC bin (normalized bin_means)
    """
    # Create GC bins (0.0 to 1.0)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    
    # Calculate mean coverage for each GC bin (this is what LIONHEART uses)
    bin_means, bin_edges_result, bin_numbers = binned_statistic(
        gc_content, coverage, statistic='mean', bins=bin_edges
    )
    
    # Set zero-means to NaN (matching LIONHEART's nan_zeroes=True)
    bin_means[bin_means == 0.0] = np.nan
    
    # Normalize factors to center around 1.0 (matching LIONHEART's div_by_mean=True)
    correction_factors = bin_means.copy()
    if np.nanmean(correction_factors) > 0:
        correction_factors /= np.nanmean(correction_factors)
    
    # Set NaN factors to 1.0 (no correction)
    correction_factors[np.isnan(correction_factors)] = 1.0
    
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
    start_coordinates: Optional[np.ndarray] = None,
    clip_above_quantile: Optional[float] = None,
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
        start_coordinates: Genomic start coordinates for each bin (bin_idx * bin_size if None)
        clip_above_quantile: Quantile to clip above (default None, matching LIONHEART's default)
    
    Returns:
        normalized_coverage: Normalized coverage array
    """
    if center is None and scale is None:
        raise ValueError("At least one of {center, scale} must be specified.")
    
    n_bins = len(coverage)
    
    # Calculate start coordinates if not provided
    if start_coordinates is None:
        start_coordinates = np.arange(n_bins) * bin_size
    
    min_start = int(np.min(start_coordinates))
    max_start = int(np.max(start_coordinates))
    
    # Number of strides needed
    num_stridings = int(np.ceil(mbin_size / stride))
    
    # Calculate megabin averages for each stride
    mbin_averages = np.zeros((n_bins, num_stridings))
    
    for striding in range(num_stridings):
        # Offset for this stride (LIONHEART starts before min_start for smoothing)
        start_offset = mbin_size - stride * striding if striding > 0 else 0
        first_start_pos = min_start - start_offset
        
        # Calculate megabin edges
        total_length_covered = max_start - first_start_pos
        num_mbins = int(np.ceil(total_length_covered / mbin_size))
        mbin_edges = [first_start_pos + mbin_size * bi for bi in range(num_mbins + 1)]
        
        # Find which megabin each bin belongs to
        # np.digitize returns 1-based indices (0 means before first edge, len(edges) means after last edge)
        # We need to convert to 0-based and handle edge cases
        digitize_result = np.digitize(start_coordinates, mbin_edges)
        # Convert to 0-based: digitize_result 1->0, 2->1, etc.
        # But digitize_result can be 0 (before first edge) or len(edges) (after last edge)
        mbin_indices = digitize_result - 1
        # Clip to valid range [0, num_mbins-1]
        # Note: num_mbins = len(mbin_edges) - 1, so max valid index is num_mbins - 1
        mbin_indices = np.clip(mbin_indices, 0, num_mbins - 1)
        
        # Calculate mean for each megabin
        for mbin_idx in range(num_mbins):
            mbin_mask = (mbin_indices == mbin_idx)
            mbin_values = coverage[mbin_mask]
            
            if len(mbin_values) > 0:
                # Clip outliers only if clip_above_quantile is specified
                # LIONHEART defaults to None (no clipping)
                if clip_above_quantile is not None:
                    uq = np.nanquantile(mbin_values, clip_above_quantile)
                    mbin_values_clipped = np.clip(mbin_values, None, uq)
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
        # Scale by mean (LIONHEART default: divide by megabin mean)
        # No additional scaling factor - just divide by megabin mean
        normalized_coverage = normalized_coverage / (mbin_overall_mean + 1e-10)
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

