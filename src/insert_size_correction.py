# src/insert_size_correction.py
"""
Insert size correction for coverage.
Simplified version of LIONHEART's insert_size correction.

Implements three types of corrections:
1. Noise correction
2. Skewness correction  
3. Mean shift correction
"""
import numpy as np
from scipy.stats import binned_statistic


def calculate_insert_size_correction_factors(
    coverages: np.ndarray,
    insert_sizes: np.ndarray,
    bin_edges: np.ndarray,
    final_mean_insert_size: float = 166.0,
    nan_extremes: bool = True,
) -> dict:
    """
    Calculate insert size correction factors (simplified version).
    
    Parameters:
        coverages: Array of coverage values
        insert_sizes: Array of mean insert size per bin
        bin_edges: Edges for binning insert sizes
        final_mean_insert_size: Target mean insert size
        nan_extremes: Whether to set extreme bins to NaN
    
    Returns:
        Dictionary with correction factors
    """
    coverages = coverages.copy()
    insert_sizes = insert_sizes.copy()
    
    # Clip negative coverages
    coverages[coverages < 0] = 0
    
    # Only process bins with valid insert sizes
    valid_mask = insert_sizes > 0
    if not np.any(valid_mask):
        # Return identity corrections if no valid data
        n_bins = len(bin_edges) - 1
        return {
            "noise_correction_factor": np.ones(n_bins),
            "skewness_correction_factor": np.ones(n_bins),
            "mean_correction_factor": np.ones(n_bins),
            "bin_midpoints": (bin_edges[:-1] + bin_edges[1:]) / 2,
        }
    
    # Calculate observed distribution (mean coverage per insert size bin)
    bin_means, bin_edges_result, _ = binned_statistic(
        insert_sizes[valid_mask],
        coverages[valid_mask],
        statistic='mean',
        bins=bin_edges
    )
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize observed distribution
    observed_dist = bin_means.copy()
    valid_obs = np.isfinite(observed_dist)
    if np.any(valid_obs) and np.nanmean(observed_dist[valid_obs]) > 0:
        observed_dist = observed_dist / np.nanmean(observed_dist[valid_obs])
    
    # 1. Noise correction: normalize by observed vs expected distribution
    # Simplified: use observed distribution as reference
    noise_correction = np.ones(len(bin_midpoints))
    valid_bins = ~np.isnan(observed_dist) & (observed_dist > 0)
    if np.any(valid_bins):
        # Normalize observed distribution
        observed_norm = observed_dist.copy()
        observed_norm[~valid_bins] = np.nanmean(observed_dist[valid_bins])
        observed_norm = observed_norm / np.nanmean(observed_norm[valid_bins])
        
        # Correction factor: inverse of normalized distribution
        noise_correction[valid_bins] = 1.0 / (observed_norm[valid_bins] + 1e-10)
        noise_correction = noise_correction / np.nanmean(noise_correction[valid_bins])
    
    # Apply noise correction first
    coverages_corrected = coverages.copy()
    coverages_corrected[valid_mask] = correct_bias(
        coverages=coverages[valid_mask],
        correct_factors=noise_correction,
        bias_scores=insert_sizes[valid_mask],
        bin_edges=bin_edges,
    )
    
    # 2. Skewness correction: simplified approach
    # Calculate skewness of insert size distribution
    valid_insert_sizes = insert_sizes[valid_mask]
    if len(valid_insert_sizes) > 0:
        mean_is = np.mean(valid_insert_sizes)
        std_is = np.std(valid_insert_sizes)
        if std_is > 0:
            # Simplified skewness: (mean - median) / std
            median_is = np.median(valid_insert_sizes)
            skewness = (mean_is - median_is) / std_is
        else:
            skewness = 0.0
    else:
        skewness = 0.0
    
    # Inverse skewness weights: reduce skewness effect
    # Simplified: linear correction based on distance from mean
    skewness_correction = np.ones(len(bin_midpoints))
    if abs(skewness) > 0.1 and len(valid_insert_sizes) > 0:
        mean_is = np.mean(valid_insert_sizes)
        # Weight bins by distance from mean (reduce extreme bins)
        distances = np.abs(bin_midpoints - mean_is)
        max_dist = np.max(distances)
        if max_dist > 0:
            # Reduce weight for bins far from mean
            weights = 1.0 - 0.3 * (distances / max_dist) * np.sign(skewness)
            skewness_correction = np.clip(weights, 0.5, 1.5)
            skewness_correction = skewness_correction / np.nanmean(skewness_correction)
    
    # Apply skewness correction
    coverages_corrected[valid_mask] = correct_bias(
        coverages=coverages_corrected[valid_mask],
        correct_factors=skewness_correction,
        bias_scores=insert_sizes[valid_mask],
        bin_edges=bin_edges,
    )
    
    # 3. Mean shift correction: adjust to target mean insert size
    # Recalculate distribution after noise and skewness correction
    bin_means_corrected, _, _ = binned_statistic(
        insert_sizes[valid_mask],
        coverages_corrected[valid_mask],
        statistic='mean',
        bins=bin_edges
    )
    
    corrected_dist = bin_means_corrected.copy()
    valid_corr = np.isfinite(corrected_dist)
    if np.any(valid_corr) and np.nanmean(corrected_dist[valid_corr]) > 0:
        corrected_dist = corrected_dist / np.nanmean(corrected_dist[valid_corr])
    
    # Target distribution: centered at final_mean_insert_size
    # Simplified: create target distribution with peak at final_mean_insert_size
    target_dist = np.ones(len(bin_midpoints))
    if len(valid_insert_sizes) > 0:
        # Find bin closest to target mean
        target_bin_idx = np.argmin(np.abs(bin_midpoints - final_mean_insert_size))
        
        # Create smooth target distribution (Gaussian-like)
        distances = np.abs(bin_midpoints - final_mean_insert_size)
        std_target = np.std(valid_insert_sizes) if len(valid_insert_sizes) > 0 else 20.0
        target_dist = np.exp(-0.5 * (distances / std_target) ** 2)
        target_dist = target_dist / np.nanmean(target_dist)
    
    # Mean correction factor
    mean_correction = np.ones(len(bin_midpoints))
    valid_bins = ~np.isnan(corrected_dist) & (corrected_dist > 0) & (target_dist > 0)
    if np.any(valid_bins):
        corrected_norm = corrected_dist[valid_bins] / np.nanmean(corrected_dist[valid_bins])
        target_norm = target_dist[valid_bins] / np.nanmean(target_dist[valid_bins])
        mean_correction[valid_bins] = corrected_norm / (target_norm + 1e-10)
        mean_correction = mean_correction / np.nanmean(mean_correction[valid_bins])
    
    # Set extreme bins to NaN if requested
    if nan_extremes:
        noise_correction[0] = np.nan
        noise_correction[-1] = np.nan
        skewness_correction[0] = np.nan
        skewness_correction[-1] = np.nan
        mean_correction[0] = np.nan
        mean_correction[-1] = np.nan
    
    return {
        "noise_correction_factor": noise_correction,
        "skewness_correction_factor": skewness_correction,
        "mean_correction_factor": mean_correction,
        "bin_midpoints": bin_midpoints,
    }


def correct_bias(
    coverages: np.ndarray,
    correct_factors: np.ndarray,
    bias_scores: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Apply bias correction factors to coverages.
    
    Parameters:
        coverages: Coverage values to correct
        correct_factors: Correction factors per bin
        bias_scores: Bias scores (e.g., insert sizes) for binning
        bin_edges: Edges for binning
    
    Returns:
        corrected_coverages: Corrected coverage values
    """
    # Find which bin each coverage belongs to
    bin_indices = np.digitize(bias_scores, bins=bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(correct_factors) - 1)
    
    # Get correction factors
    corrections = correct_factors[bin_indices]
    
    # Avoid zero-division
    corrections[corrections == 0] = np.nan
    
    # Apply correction: divide by correction factors (LIONHEART uses division)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = coverages.astype(np.float64) / corrections
    
    # Clip negative finite values
    neg_mask = np.isfinite(corrected) & (corrected < 0)
    corrected[neg_mask] = 0.0
    
    return corrected

