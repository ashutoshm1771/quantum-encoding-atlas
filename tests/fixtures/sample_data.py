"""Sample data for testing."""

import numpy as np


def get_sample_classification_data(
    n_samples: int = 100,
    n_features: int = 4,
    seed: int = 42,
):
    """Generate sample classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def get_sample_regression_data(
    n_samples: int = 100,
    n_features: int = 4,
    seed: int = 42,
):
    """Generate sample regression data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n_samples) * 0.1
    return X, y
