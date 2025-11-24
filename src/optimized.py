# src/optimized.py
"""
Optimized functions using Numba JIT compilation for performance.
Based on LIONHEART's optimization approach.
"""
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def aggregate_10bp_to_100bp_numba(mask_10bp: np.ndarray) -> np.ndarray:
    """
    Optimized aggregation of 10bp mask to 100bp using Numba.
    
    Parameters:
        mask_10bp: Mask array at 10bp resolution
    
    Returns:
        Mask array at 100bp resolution
    """
    n_10bp = len(mask_10bp)
    n_100bp = (n_10bp + 9) // 10  # Round up
    
    mask_100bp = np.zeros(n_100bp, dtype=np.float32)
    
    for i in range(n_100bp):
        start = i * 10
        end = min(start + 10, n_10bp)
        s = 0.0
        for j in range(start, end):
            s += mask_10bp[j]
        mask_100bp[i] = s
    
    return mask_100bp


@njit(cache=True)
def compute_sums_numba(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute all required sums for Pearson R calculation in a single pass.
    Optimized version based on LIONHEART's implementation.
    
    Parameters:
        x: First array (e.g., sample coverage)
        y: Second array (e.g., cell type coverage)
    
    Returns:
        (n, x_sum, y_sum, x_squared_sum, y_squared_sum, xy_sum)
    """
    n = len(x)
    x_sum = 0.0
    y_sum = 0.0
    x_squared_sum = 0.0
    y_squared_sum = 0.0
    xy_sum = 0.0
    
    for i in range(n):
        xi = x[i]
        yi = y[i]
        if not (np.isnan(xi) or np.isnan(yi)):
            x_sum += xi
            y_sum += yi
            x_squared_sum += xi * xi
            y_squared_sum += yi * yi
            xy_sum += xi * yi
    
    return n, x_sum, y_sum, x_squared_sum, y_squared_sum, xy_sum


@njit(cache=True)
def filter_nans_numba(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Filter out NaN values from both arrays.
    
    Parameters:
        x: First array
        y: Second array
    
    Returns:
        (x_valid, y_valid, valid_count)
    """
    n = len(x)
    # Count valid values first
    valid_count = 0
    for i in range(n):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            valid_count += 1
    
    # Create arrays for valid values
    x_valid = np.zeros(valid_count, dtype=np.float64)
    y_valid = np.zeros(valid_count, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            x_valid[idx] = x[i]
            y_valid[idx] = y[i]
            idx += 1
    
    return x_valid, y_valid, valid_count


@njit(cache=True)
def compute_pearson_r_numba(
    n: float,
    x_sum: float,
    y_sum: float,
    x_squared_sum: float,
    y_squared_sum: float,
    xy_sum: float,
) -> float:
    """
    Compute Pearson R from pre-computed sums.
    
    Parameters:
        n: Number of data points
        x_sum, y_sum: Sums of x and y
        x_squared_sum, y_squared_sum: Sums of squares
        xy_sum: Sum of x*y
    
    Returns:
        Pearson correlation coefficient
    """
    if n < 2:
        return 0.0
    
    numerator = n * xy_sum - x_sum * y_sum
    denominator = np.sqrt(n * x_squared_sum - x_sum * x_sum) * np.sqrt(n * y_squared_sum - y_sum * y_sum)
    
    if denominator == 0:
        return 0.0
    
    r = numerator / denominator
    # Clip to [-1, 1]
    if r > 1.0:
        return 1.0
    elif r < -1.0:
        return -1.0
    return r

