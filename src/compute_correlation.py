# src/compute_correlation.py
import os
import numpy as np
from scipy.stats import pearsonr

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


def compute_correlations(coverage, mask_paths, chrom, chrom_len, bin_size):
    """
    Compute Pearson correlation between coverage and each mask.
    """
    corr_stats = {}
    for name, path in mask_paths.items():
        mask = load_mask(path, chrom, chrom_len, bin_size)
        # Ensure same length
        min_len = min(len(coverage), len(mask))
        if min_len == 0:
            corr_stats[f"pearson_r_{name}"] = 0.0
            corr_stats[f"p_value_{name}"] = 1.0
            continue
        cov_subset = coverage[:min_len]
        mask_subset = mask[:min_len]
        try:
            r, p = pearsonr(cov_subset, mask_subset)
            corr_stats[f"pearson_r_{name}"] = float(r) if not np.isnan(r) else 0.0
            corr_stats[f"p_value_{name}"] = float(p) if not np.isnan(p) else 1.0
        except Exception as e:
            print(f"Warning: Error computing correlation for {name}: {e}")
            corr_stats[f"pearson_r_{name}"] = 0.0
            corr_stats[f"p_value_{name}"] = 1.0
    return corr_stats

