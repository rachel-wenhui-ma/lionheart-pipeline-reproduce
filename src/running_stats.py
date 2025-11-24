# src/running_stats.py
"""
Running statistics accumulator for accumulating Pearson R across all chromosomes.
Simplified version of LIONHEART's RunningPearsonR.
"""
import numpy as np
from typing import Dict


class RunningPearsonR:
    """
    Accumulate statistics for Pearson R calculation across multiple chromosomes.
    This allows us to compute a single Pearson R value from data across all chromosomes.
    """
    
    def __init__(self, ignore_nans: bool = True):
        self.ignore_nans = ignore_nans
        self.n = 0
        self.x_sum = 0.0
        self.y_sum = 0.0
        self.x_squared_sum = 0.0
        self.y_squared_sum = 0.0
        self.xy_sum = 0.0
    
    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add data from a chromosome to the running statistics.
        
        Parameters:
            x: Coverage array (sample)
            y: Coverage array (cell type mask)
        """
        # Filter NaNs if needed
        if self.ignore_nans:
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            x = x[valid_mask]
            y = y[valid_mask]
        
        if len(x) == 0:
            return
        
        # Accumulate statistics
        self.n += len(x)
        self.x_sum += np.sum(x)
        self.y_sum += np.sum(y)
        self.x_squared_sum += np.sum(x ** 2)
        self.y_squared_sum += np.sum(y ** 2)
        self.xy_sum += np.sum(x * y)
    
    def compute_pearson_r(self) -> float:
        """
        Compute Pearson R from accumulated statistics.
        
        Returns:
            Pearson correlation coefficient
        """
        if self.n < 2:
            return 0.0
        
        numerator = self.n * self.xy_sum - self.x_sum * self.y_sum
        denominator = np.sqrt(self.n * self.x_squared_sum - self.x_sum ** 2) * np.sqrt(
            self.n * self.y_squared_sum - self.y_sum ** 2
        )
        
        if denominator == 0:
            return 0.0
        
        r = numerator / denominator
        return max(min(r, 1.0), -1.0)  # Clip to [-1, 1]
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get all accumulated statistics.
        
        Returns:
            Dictionary with n, x_sum, y_sum, x_squared_sum, y_squared_sum, xy_sum
        """
        return {
            'n': float(self.n),
            'x_sum': float(self.x_sum),
            'y_sum': float(self.y_sum),
            'x_squared_sum': float(self.x_squared_sum),
            'y_squared_sum': float(self.y_squared_sum),
            'xy_sum': float(self.xy_sum),
        }

