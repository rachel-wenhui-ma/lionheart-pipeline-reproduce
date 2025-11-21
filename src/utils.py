# src/utils.py
import numpy as np

def summarize_coverage(coverage):
    return {
        "cov_mean": float(np.mean(coverage)),
        "cov_std": float(np.std(coverage)),
        "cov_median": float(np.median(coverage)),
    }

