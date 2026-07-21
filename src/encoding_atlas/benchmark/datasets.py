"""Standard datasets for benchmarking."""

from __future__ import annotations

from typing import cast

import numpy as np

_DATASETS = ["iris", "moons", "circles", "linear", "xor", "iris3", "blobs3"]

# Datasets with more than two classes (for multi-class benchmarking).
_MULTICLASS_DATASETS = frozenset({"iris3", "blobs3"})


def list_multiclass_datasets() -> list[str]:
    """List the available multi-class (more than two labels) datasets."""
    return sorted(_MULTICLASS_DATASETS)


def list_datasets() -> list[str]:
    """List available benchmark datasets."""
    return _DATASETS.copy()


def get_dataset(
    name: str,
    n_samples: int = 200,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get a benchmark dataset.

    Parameters
    ----------
    name : str
        Dataset name.
    n_samples : int
        Number of samples.
    seed : int or None
        Random seed.

    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Labels.
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {_DATASETS}")

    rng = np.random.default_rng(seed)

    if name == "iris":
        try:
            from sklearn.datasets import load_iris

            data = load_iris()
            X, y = data.data[:, :2], data.target
            # Binary classification
            mask = y < 2
            return X[mask], y[mask]
        except ImportError:
            raise ImportError("sklearn required for iris dataset")

    elif name == "moons":
        try:
            from sklearn.datasets import make_moons

            return cast(
                "tuple[np.ndarray, np.ndarray]",
                make_moons(n_samples=n_samples, noise=0.1, random_state=seed),
            )
        except ImportError:
            raise ImportError("sklearn required for moons dataset")

    elif name == "circles":
        try:
            from sklearn.datasets import make_circles

            return cast(
                "tuple[np.ndarray, np.ndarray]",
                make_circles(
                    n_samples=n_samples, noise=0.1, factor=0.5, random_state=seed
                ),
            )
        except ImportError:
            raise ImportError("sklearn required for circles dataset")

    elif name == "linear":
        X = rng.standard_normal((n_samples, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    elif name == "xor":
        X = rng.standard_normal((n_samples, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        return X, y

    elif name == "iris3":
        try:
            from sklearn.datasets import load_iris
        except ImportError:
            raise ImportError("sklearn required for iris3 dataset")
        data = load_iris()
        # All three classes, first two features.
        return data.data[:, :2], data.target

    elif name == "blobs3":
        try:
            from sklearn.datasets import make_blobs
        except ImportError:
            raise ImportError("sklearn required for blobs3 dataset")
        X, y = make_blobs(
            n_samples=n_samples,
            centers=3,
            n_features=2,
            cluster_std=1.0,
            random_state=seed,
        )
        return X, y.astype(int)

    raise ValueError(f"Unknown dataset: {name}")


# Regression datasets (continuous targets), all two-dimensional so that
# two-qubit encodings apply directly.
_REGRESSION_DATASETS = ["linear_reg", "sine_reg"]


def list_regression_datasets() -> list[str]:
    """List available regression datasets (continuous targets)."""
    return _REGRESSION_DATASETS.copy()


def get_regression_dataset(
    name: str,
    n_samples: int = 200,
    seed: int | None = None,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Get a benchmark regression dataset with continuous targets.

    Parameters
    ----------
    name : {"linear_reg", "sine_reg"}
        Dataset name. ``linear_reg`` is a noisy linear function of two
        features; ``sine_reg`` is a smooth non-linear (sinusoidal) function.
    n_samples : int, default=200
        Number of samples.
    seed : int or None, default=None
        Random seed.
    noise : float, default=0.1
        Standard deviation of additive Gaussian noise on the target.

    Returns
    -------
    X : ndarray, shape (n_samples, 2)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Continuous targets (float).
    """
    if name not in _REGRESSION_DATASETS:
        raise ValueError(
            f"Unknown regression dataset: {name}. Available: {_REGRESSION_DATASETS}"
        )

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))

    if name == "linear_reg":
        y = 2.0 * X[:, 0] - 3.0 * X[:, 1]
    else:  # "sine_reg"
        # One period across the input range: high-frequency variants are not
        # learnable at practical sample sizes (by quantum *or* classical
        # models), which would make the benchmark uninformative.
        y = np.sin(np.pi * X[:, 0]) + 0.5 * np.cos(np.pi * X[:, 1])

    y = y + rng.normal(0.0, noise, size=n_samples)
    return X, y.astype(np.float64)
