# src/compute_coverage.py
import pysam
import numpy as np
from scipy.stats import binned_statistic

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

