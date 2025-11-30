from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import binned_statistic


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


def correct_bias(
    coverages: np.ndarray,
    correct_factors: np.ndarray,
    bias_scores: Optional[np.ndarray] = None,
    bin_indices: Optional[np.ndarray] = None,
    bin_edges: Optional[np.ndarray] = None,
    normalize_correct_factors: bool = True,
    clip_negative_coverages: bool = True,
) -> np.ndarray:
    assert isinstance(coverages, np.ndarray)
    assert isinstance(correct_factors, np.ndarray)

    assert sum([bin_indices is not None, bias_scores is not None]) == 1, (
        "Exactly one of `bin_indices` and `bias_scores` should be specified."
    )
    if bias_scores is not None:
        assert bin_edges is not None and isinstance(bin_edges, np.ndarray)
        assert isinstance(bias_scores, np.ndarray)
    else:
        assert isinstance(bin_indices, np.ndarray)
        assert bin_edges is None, (
            "`bin_edges` should only be specified along with `bias_scores`."
        )

    if bias_scores is not None:
        bin_indices = np.digitize(bias_scores, bins=bin_edges, right=False) - 1
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices > (len(bin_edges) - 2)] = len(bin_edges) - 2

    if normalize_correct_factors:
        correct_factors /= np.nanmean(correct_factors)

    corrections = np.zeros_like(bin_indices, dtype=np.float32)
    unique_bin_indices = list(np.unique(bin_indices))
    for binn in unique_bin_indices:
        corrections[bin_indices == binn] = correct_factors[binn]

    corrections[corrections == 0] = np.nan

    if clip_negative_coverages:
        coverages[coverages < 0] = 0.0

    if coverages.ndim == 3:
        for end_type in range(coverages.shape[1]):
            coverages[:, end_type, :] /= corrections
    else:
        coverages /= corrections

    return coverages


def bin_midpoints(edges: np.ndarray) -> np.ndarray:
    diffs = np.diff(edges) / 2
    return edges[:-1] + diffs


def average_bins(
    x: np.ndarray,
    y: np.ndarray,
    bin_edges: np.ndarray,
    nan_zeroes: bool = False,
    nan_extremes: Union[str, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(bin_edges, np.ndarray)
    assert x.ndim == 1
    assert y.ndim == 1
    assert bin_edges.ndim == 1
    assert isinstance(nan_extremes, (str, bool))
    if isinstance(nan_extremes, str) and nan_extremes not in ["min", "max"]:
        raise ValueError(
            "When `nan_extremes` is supplied as string, it must be either "
            f"'min' or 'max'. Got: '{nan_extremes}'."
        )

    bin_edges = np.unique(bin_edges)
    binned = binned_statistic(x=x, values=y, statistic=np.nanmean, bins=bin_edges)
    averages = binned.statistic
    edges = binned.bin_edges
    midpoints = bin_midpoints(edges)

    if nan_zeroes:
        averages[averages == 0.0] = np.nan

    averages = set_extremes_to_nan(averages, nan_extremes=nan_extremes)

    return midpoints, averages


def set_extremes_to_nan(x: np.ndarray, nan_extremes: Union[bool, str]) -> np.ndarray:
    if nan_extremes:
        if isinstance(nan_extremes, str):
            if nan_extremes == "min":
                x[0] = np.nan
            elif nan_extremes == "max":
                x[-1] = np.nan
        else:
            x[0] = np.nan
            x[-1] = np.nan
    return x


def smoothe_signal(
    x: np.ndarray,
    kernel_size: int = 5,
    kernel_type: str = "gaussian",
    kernel_std: float = 1.0,
) -> np.ndarray:
    if kernel_type == "uniform":
        kernel = np.ones(kernel_size) / kernel_size
    elif kernel_type == "gaussian":
        kernel = gaussian_kernel_1d(n=kernel_size, std=kernel_std)
    else:
        raise ValueError(f"Unknown `kernel_type`: {kernel_type}")

    smooth_x = np.convolve(x, kernel, mode="same")
    smooth_x[np.isnan(smooth_x)] = x[np.isnan(smooth_x)]

    return smooth_x


def gaussian_kernel_1d(n: int = 5, std: float = 1.0) -> np.ndarray:
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(std))
    return gauss / np.sum(gauss)




