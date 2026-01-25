"""Input validation utilities."""

import numpy as np
from numpy.typing import ArrayLike
from typing import Literal

from encoding_atlas.core.types import BackendType


def validate_features(
    x: ArrayLike,
    n_features: int,
) -> np.ndarray:
    """Validate feature input."""
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        if x.shape[0] != n_features:
            raise ValueError(f"Expected {n_features} features, got {x.shape[0]}")
    elif x.ndim == 2:
        if x.shape[1] != n_features:
            raise ValueError(f"Expected {n_features} features, got {x.shape[1]}")
    else:
        raise ValueError(f"Input must be 1D or 2D, got {x.ndim}D")

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input contains NaN or infinite values")

    return x


def validate_backend(backend: str) -> BackendType:
    """Validate backend name."""
    valid_backends = ("pennylane", "qiskit", "cirq")
    if backend not in valid_backends:
        raise ValueError(f"Backend must be one of {valid_backends}, got {backend}")
    return backend  # type: ignore
