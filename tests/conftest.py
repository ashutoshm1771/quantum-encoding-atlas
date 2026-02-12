"""Pytest configuration and fixtures."""

# Configure matplotlib to use non-interactive backend before any imports
# This prevents TclError when running tests in headless environments
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def sample_data_2d():
    """Sample 2D feature data."""
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


@pytest.fixture
def sample_data_4d():
    """Sample 4D feature data."""
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )


@pytest.fixture
def random_data():
    """Random feature data generator."""

    def _generate(n_samples: int = 10, n_features: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_samples, n_features))

    return _generate
