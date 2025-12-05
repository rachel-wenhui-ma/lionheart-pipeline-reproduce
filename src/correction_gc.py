from typing import Tuple, Union
import numpy as np

# Import helper functions from correction_helpers (official LIONHEART code)
from src.correction_helpers import (
    average_bins,
    smoothe_signal,
    correct_bias,
)


def calculate_correction_factors(
    bias_scores: np.ndarray,
    coverages: np.ndarray,
    bin_edges: np.ndarray,
    nan_extremes: Union[str, bool] = True,
    div_by_mean: bool = True,
    smoothe_factors: bool = True,
    smoothing_settings: dict = {
        "kernel_size": 5,
        "kernel_type": "gaussian",
        "kernel_std": 1.0,
    },
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate correction factors for GC bias correction.
    This is a wrapper around the official LIONHEART helper functions.
    """
    midpoints, correct_factors = average_bins(
        x=bias_scores,
        y=coverages,
        bin_edges=bin_edges,
        nan_zeroes=True,
        nan_extremes=nan_extremes,
    )

    if smoothe_factors:
        correct_factors = smoothe_signal(x=correct_factors, **smoothing_settings)

    if div_by_mean:
        correct_factors /= np.nanmean(correct_factors)

    return midpoints, correct_factors






