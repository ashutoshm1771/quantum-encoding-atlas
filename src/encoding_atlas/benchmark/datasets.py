"""Standard datasets for benchmarking."""

from __future__ import annotations

import numpy as np

_DATASETS = ["iris", "moons", "circles", "linear", "xor"]


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

            return make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
        except ImportError:
            raise ImportError("sklearn required for moons dataset")

    elif name == "circles":
        try:
            from sklearn.datasets import make_circles

            return make_circles(
                n_samples=n_samples, noise=0.1, factor=0.5, random_state=seed
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

    raise ValueError(f"Unknown dataset: {name}")
