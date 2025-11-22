# src/compute_correlation.py
import os
import numpy as np
from scipy.stats import pearsonr
from scipy import special

def load_mask(bed_path, chrom, chrom_len, bin_size):
    """
    Loads a BED file and converts it into a mask (0/1 per bin).
    """
    n_bins = chrom_len // bin_size + 1
    mask = np.zeros(n_bins, dtype=np.int8)

    if not os.path.exists(bed_path):
        print(f"Warning: Mask file not found: {bed_path}, using empty mask")
        return mask

    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            c, start, end = parts[0], parts[1], parts[2]
            if c != chrom:
                continue
            try:
                start, end = int(start), int(end)
                start_bin = start // bin_size
                end_bin = end // bin_size
                if start_bin < n_bins and end_bin < n_bins:
                    mask[start_bin:end_bin+1] = 1
            except ValueError:
                continue

    return mask


def compute_all_features(coverage, mask_dict, chrom, chrom_len, bin_size):
    """
    Compute all 10 LIONHEART feature types between coverage and each mask.
    
    Parameters:
        coverage: Coverage array
        mask_dict: Dictionary mapping cell_type -> mask_array or mask_path
                  If value is array, use directly; if string, load from BED file
        chrom: Chromosome name
        chrom_len: Chromosome length
        bin_size: Bin size
    
    LIONHEART's 10 feature types:
    0) Pearson R
    1) p-value
    2) Normalized dot product (fraction_within = xy_sum / n)
    3) Cosine Similarity
    4) x_sum (sum of coverage values)
    5) y_sum (sum of mask overlap fractions)
    6) x_squared_sum
    7) y_squared_sum
    8) xy_sum
    9) n (number of included bins)
    
    This follows LIONHEART's RunningPearsonR algorithm.
    """
    all_features = {}
    
    for name, mask_input in mask_dict.items():
        # Check if mask_input is already an array or a path
        if isinstance(mask_input, (np.ndarray, list)):
            # Already loaded mask array
            mask = np.asarray(mask_input, dtype=np.float64)
        else:
            # Path to BED file, load it
            mask = load_mask(mask_input, chrom, chrom_len, bin_size)
        # Ensure same length
        min_len = min(len(coverage), len(mask))
        if min_len == 0:
            # Return zeros for all features if no data
            all_features[f"pearson_r_{name}"] = 0.0
            all_features[f"p_value_{name}"] = 1.0
            all_features[f"fraction_within_{name}"] = 0.0
            all_features[f"cosine_similarity_{name}"] = 0.0
            all_features[f"x_sum_{name}"] = 0.0
            all_features[f"y_sum_{name}"] = 0.0
            all_features[f"x_squared_sum_{name}"] = 0.0
            all_features[f"y_squared_sum_{name}"] = 0.0
            all_features[f"xy_sum_{name}"] = 0.0
            all_features[f"n_{name}"] = 0.0
            continue
        
        x = coverage[:min_len].astype(np.float64)
        y = mask[:min_len].astype(np.float64)
        
        # Remove NaNs (following LIONHEART's RunningPearsonR._check_data)
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]
        
        n = len(x)
        if n < 2:
            # Not enough data for correlation
            all_features[f"pearson_r_{name}"] = 0.0
            all_features[f"p_value_{name}"] = 1.0
            all_features[f"fraction_within_{name}"] = 0.0
            all_features[f"cosine_similarity_{name}"] = 0.0
            all_features[f"x_sum_{name}"] = float(np.sum(x)) if n > 0 else 0.0
            all_features[f"y_sum_{name}"] = float(np.sum(y)) if n > 0 else 0.0
            all_features[f"x_squared_sum_{name}"] = float(np.sum(x**2)) if n > 0 else 0.0
            all_features[f"y_squared_sum_{name}"] = float(np.sum(y**2)) if n > 0 else 0.0
            all_features[f"xy_sum_{name}"] = float(np.sum(x * y)) if n > 0 else 0.0
            all_features[f"n_{name}"] = float(n)
            continue
        
        # Calculate running statistics (following LIONHEART's RunningPearsonR)
        x_sum = np.sum(x)
        y_sum = np.sum(y)
        x_squared_sum = np.sum(x**2)
        y_squared_sum = np.sum(y**2)
        xy_sum = np.sum(x * y)
        
        # 0) Pearson R
        numerator = n * xy_sum - x_sum * y_sum
        denominator = np.sqrt(n * x_squared_sum - x_sum**2) * np.sqrt(n * y_squared_sum - y_sum**2)
        if denominator == 0:
            r = 0.0
            p = 1.0
        else:
            r = numerator / denominator
            r = max(min(r, 1.0), -1.0)  # Clip to [-1, 1]
            
            # 1) p-value (following scipy.stats.pearsonr)
            if abs(r) == 1.0:
                p = 0.0
            else:
                df = n - 2
                if df <= 0:
                    p = 1.0
                else:
                    # Using beta distribution CDF (same as scipy.stats.pearsonr)
                    ab = df / 2.0
                    p = 2 * special.btdtr(ab, ab, 0.5 * (1.0 - abs(r)))
        
        # 2) Normalized dot product (fraction_within)
        fraction_within = xy_sum / n if n > 0 else 0.0
        
        # 3) Cosine Similarity
        x_norm = np.sqrt(x_squared_sum)
        y_norm = np.sqrt(y_squared_sum)
        if x_norm > 0 and y_norm > 0:
            cosine_sim = xy_sum / (x_norm * y_norm)
            cosine_sim = max(min(cosine_sim, 1.0), -1.0)  # Clip to [-1, 1]
        else:
            cosine_sim = 0.0
        
        # Store all 10 features
        all_features[f"pearson_r_{name}"] = float(r)
        all_features[f"p_value_{name}"] = float(p)
        all_features[f"fraction_within_{name}"] = float(fraction_within)
        all_features[f"cosine_similarity_{name}"] = float(cosine_sim)
        all_features[f"x_sum_{name}"] = float(x_sum)
        all_features[f"y_sum_{name}"] = float(y_sum)
        all_features[f"x_squared_sum_{name}"] = float(x_squared_sum)
        all_features[f"y_squared_sum_{name}"] = float(y_squared_sum)
        all_features[f"xy_sum_{name}"] = float(xy_sum)
        all_features[f"n_{name}"] = float(n)
    
    return all_features


def compute_correlations(coverage, mask_paths, chrom, chrom_len, bin_size):
    """
    Compute correlation features (wrapper for backward compatibility).
    Now computes all 10 feature types.
    """
    return compute_all_features(coverage, mask_paths, chrom, chrom_len, bin_size)

