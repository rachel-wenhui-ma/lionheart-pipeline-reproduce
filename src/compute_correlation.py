# src/compute_correlation.py
import numpy as np
from scipy.stats import pearsonr

def load_mask(bed_path, chrom, chrom_len, bin_size):
    """
    Loads a BED file and converts it into a mask (0/1 per bin).
    """
    mask = np.zeros(chrom_len // bin_size + 1, dtype=np.int8)

    with open(bed_path) as f:
        for line in f:
            c, start, end = line.strip().split("\t")
            if c != chrom:
                continue
            start, end = int(start), int(end)
            start_bin = start // bin_size
            end_bin = end // bin_size
            mask[start_bin:end_bin+1] = 1

    return mask


def compute_correlations(coverage, mask_paths, chrom, chrom_len, bin_size):
    """
    Compute Pearson correlation between coverage and each mask.
    """
    corr_stats = {}
    for name, path in mask_paths.items():
        mask = load_mask(path, chrom, chrom_len, bin_size)
        r, p = pearsonr(coverage, mask)
        corr_stats[f"pearson_r_{name}"] = float(r)
        corr_stats[f"p_value_{name}"] = float(p)
    return corr_stats

