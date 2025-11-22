# src/outlier_detection.py
"""
ZIPoisson-based outlier detection for coverage clipping.
Simplified version of LIONHEART's ZIPoisson implementation.
"""
import numpy as np
from scipy.stats import poisson
from numbers import Number
from typing import Union, List, Optional


class ZIPoisson:
    """
    Zero-Inflated Poisson distribution for outlier detection.
    
    Simplified version of LIONHEART's ZIPoisson class.
    Used to find clipping threshold for coverage values.
    """
    
    def __init__(self, handle_negatives="warn_clip", max_num_negatives=50):
        """
        Initialize ZIPoisson model.
        
        Parameters:
            handle_negatives: How to handle negative numbers
            max_num_negatives: Max number of negatives allowed
        """
        self.handle_negatives = handle_negatives
        self.max_num_negatives = max_num_negatives
        self.n = 0
        self.mu = 0.0
        self.non_zeros = 0
        self._iter_pos = 0
    
    def reset(self):
        """Reset all fitted parameters."""
        self.n = 0
        self.mu = 0.0
        self.non_zeros = 0
        self._iter_pos = 0
        return self
    
    def partial_fit(self, x: np.ndarray):
        """
        Partially fit the distribution.
        
        Parameters:
            x: 1D array of non-negative integers (coverage values)
        """
        x = np.asarray(x, dtype=np.int64)
        
        # Handle negatives
        if np.any(x < 0):
            if self.handle_negatives == "raise":
                raise ValueError(f"Found {np.sum(x < 0)} negative values")
            elif self.handle_negatives == "warn_clip":
                n_neg = np.sum(x < 0)
                if n_neg > self.max_num_negatives:
                    raise ValueError(f"Too many negatives: {n_neg} > {self.max_num_negatives}")
                x[x < 0] = 0
        
        # Update statistics
        if len(x) > 0:
            old_n = self.n
            old_mu = self.mu
            
            new_n = len(x)
            new_mu = np.mean(x[x > 0]) if np.any(x > 0) else 0.0
            
            if old_n == 0:
                self.n = new_n
                self.mu = new_mu
                self.non_zeros = np.count_nonzero(x)
            else:
                # Weighted average
                total_n = old_n + new_n
                self.mu = (old_mu * old_n + new_mu * new_n) / total_n if total_n > 0 else 0.0
                self.n = total_n
                self.non_zeros += np.count_nonzero(x)
        
        return self
    
    def set_iter_pos(self, pos: int):
        """Set iterator position for finding threshold."""
        self._iter_pos = max(0, int(pos))
    
    def __iter__(self):
        """Make class iterable for finding threshold."""
        return self
    
    def __next__(self):
        """
        Get next value, probability, and cumulative probability.
        
        Returns:
            (val, pmf, cdf): value, PMF, CDF
        """
        if self.n == 0:
            raise RuntimeError("Model not fitted. Call partial_fit() first.")
        
        val = self._iter_pos
        self._iter_pos += 1
        
        # Calculate zero-inflated Poisson probabilities
        prob_non_zero = self.non_zeros / self.n if self.n > 0 else 0.0
        
        if val == 0:
            pmf = (1 - prob_non_zero) + prob_non_zero * poisson.pmf(0, self.mu)
            cdf = pmf
        else:
            pmf = prob_non_zero * poisson.pmf(val, self.mu)
            cdf = (1 - prob_non_zero) + prob_non_zero * poisson.cdf(val, self.mu)
        
        return val, pmf, cdf


def find_clipping_threshold(coverage: np.ndarray, threshold: float = 1.0 / 263_000_000) -> int:
    """
    Find clipping threshold using ZIPoisson distribution.
    
    Parameters:
        coverage: Array of coverage values
        threshold: Tail probability threshold (default from LIONHEART)
    
    Returns:
        clipping_val: Threshold value for clipping
    """
    # Round coverage to integers
    coverage_int = np.round(coverage).astype(np.int64)
    coverage_int = np.clip(coverage_int, 0, None)  # Remove negatives
    
    # Fit ZIPoisson model
    poiss = ZIPoisson(handle_negatives="warn_clip", max_num_negatives=50)
    poiss.reset().partial_fit(coverage_int)
    
    # Start iteration from mean
    start_pos = int(np.floor(np.nanmean(coverage_int)))
    poiss.set_iter_pos(start_pos)
    poiss = iter(poiss)
    
    # Find threshold where tail probability < threshold
    while True:
        val, _, cum_prob = next(poiss)
        tail_prob = 1.0 - cum_prob  # P(X > val)
        
        if tail_prob < threshold:
            return val


def clip_outliers(coverage: np.ndarray, clipping_val: int) -> np.ndarray:
    """
    Clip coverage values above threshold.
    
    Parameters:
        coverage: Array of coverage values
        clipping_val: Threshold value
    
    Returns:
        clipped_coverage: Coverage with outliers clipped
    """
    clipped = coverage.copy()
    clipped[clipped > clipping_val] = clipping_val
    return clipped

