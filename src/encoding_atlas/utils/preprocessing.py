"""Data preprocessing utilities."""

import numpy as np
from numpy.typing import ArrayLike


def scale_features(
    x: ArrayLike,
    range_min: float = 0.0,
    range_max: float = np.pi,
) -> np.ndarray:
    """Scale features to a specified range.

    Parameters
    ----------
    x : array-like
        Input features.
    range_min : float
        Minimum of target range.
    range_max : float
        Maximum of target range.

    Returns
    -------
    ndarray
        Scaled features.
    """
    x = np.asarray(x, dtype=np.float64)
    x_min, x_max = x.min(), x.max()

    if x_max - x_min < 1e-10:
        return np.full_like(x, (range_min + range_max) / 2)

    return (x - x_min) / (x_max - x_min) * (range_max - range_min) + range_min


def normalize_features(x: ArrayLike) -> np.ndarray:
    """Normalize features to unit norm.

    Parameters
    ----------
    x : array-like
        Input features.

    Returns
    -------
    ndarray
        Normalized features.
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)

    if norm < 1e-10:
        return x

    return x / norm
