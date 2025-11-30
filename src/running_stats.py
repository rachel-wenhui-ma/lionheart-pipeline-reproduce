import os
import numpy as np
from typing import Dict, Optional

TRACE_ENV = os.environ.get("TRACE_RUNNING_STATS", "").strip()
if TRACE_ENV:
    _trace_entries = {entry.strip() for entry in TRACE_ENV.split(",") if entry.strip()}
else:
    _trace_entries = set()
TRACE_ALL = "*" in _trace_entries
TRACE_TARGETS = _trace_entries - {"*"}


class RunningPearsonR:
    """
    Accumulate statistics for Pearson R calculation across multiple chromosomes.
    This allows us to compute a single Pearson R value from data across all chromosomes.
    """
    
    def __init__(self, ignore_nans: bool = True, name: Optional[str] = None):
        self.ignore_nans = ignore_nans
        self.name = name
        self._trace_enabled = bool(
            TRACE_ALL or (self.name is not None and self.name in TRACE_TARGETS)
        )
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
        
        prev_state = (
            self.n,
            self.x_sum,
            self.x_squared_sum,
            self.y_sum,
            self.y_squared_sum,
            self.xy_sum,
        )

        delta_n = len(x)
        delta_x_sum = float(np.sum(x))
        delta_y_sum = float(np.sum(y))
        delta_x_squared = float(np.sum(x ** 2))
        delta_y_squared = float(np.sum(y ** 2))
        delta_xy = float(np.sum(x * y))

        # Accumulate statistics
        self.n += delta_n
        self.x_sum += delta_x_sum
        self.y_sum += delta_y_sum
        self.x_squared_sum += delta_x_squared
        self.y_squared_sum += delta_y_squared
        self.xy_sum += delta_xy

        if self._trace_enabled and delta_n > 0:
            self._emit_trace(
                prev_state,
                {
                    "n": delta_n,
                    "x_sum": delta_x_sum,
                    "y_sum": delta_y_sum,
                    "x_squared_sum": delta_x_squared,
                    "y_squared_sum": delta_y_squared,
                    "xy_sum": delta_xy,
                },
            )
    
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

    def _emit_trace(self, prev_state, deltas: Dict[str, float]) -> None:
        name = self.name or "unnamed"
        prev_n, prev_x_sum, prev_x_sq, prev_y_sum, prev_y_sq, prev_xy = prev_state
        msg = (
            f"[TRACE_RUNNING_STATS] {name} "
            f"Δn={deltas['n']}, Δx_sum={deltas['x_sum']:.6f}, Δx2_sum={deltas['x_squared_sum']:.6f}, "
            f"Δy_sum={deltas['y_sum']:.6f}, Δy2_sum={deltas['y_squared_sum']:.6f}, "
            f"Δxy_sum={deltas['xy_sum']:.6f} -> "
            f"n: {prev_n}→{self.n}, "
            f"x_sum: {prev_x_sum:.6f}→{self.x_sum:.6f}, "
            f"x2_sum: {prev_x_sq:.6f}→{self.x_squared_sum:.6f}, "
            f"y_sum: {prev_y_sum:.6f}→{self.y_sum:.6f}, "
            f"y2_sum: {prev_y_sq:.6f}→{self.y_squared_sum:.6f}, "
            f"xy_sum: {prev_xy:.6f}→{self.xy_sum:.6f}"
        )
        print(msg)


