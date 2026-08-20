"""Pytest configuration and fixtures."""

# Configure matplotlib to use non-interactive backend before any imports
# This prevents TclError when running tests in headless environments
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tests._backends import (
    ALL_BACKENDS,
    INSTALL_HINT,
    REQUIRE_ALL_BACKENDS_ENV,
    missing_backends,
    require_all_backends,
)


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run a strict session with an advertised backend missing.

    Most modules guard backend-specific tests with ``pytest.importorskip``, so
    an environment lacking Qiskit or Cirq produces a *green* run that verified
    far less than it appears to. That is the failure mode which let a
    bit-reversed Qiskit state ship in ``SO2EquivariantFeatureMap``.

    CI sets ``ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1``, which makes that
    situation a usage error raised before collection, rather than a silent
    thinning of the suite. See :mod:`tests._backends` for the full policy.
    """
    if not require_all_backends():
        return
    missing = missing_backends(ALL_BACKENDS)
    if not missing:
        return
    raise pytest.UsageError(
        f"{REQUIRE_ALL_BACKENDS_ENV} is set, but these advertised backends are "
        f"not installed: {missing}. Their tests would skip and the run would "
        f"still pass, so the session is refused instead. "
        f"Install with: {INSTALL_HINT}  "
        f"(unset {REQUIRE_ALL_BACKENDS_ENV} to allow skipping.)"
    )


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
