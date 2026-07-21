"""Random-input sampling helpers for the analysis pipeline.

The three core analysis functions sample random inputs into a fixed
range ``[low, high]`` and feed each into the encoding. This module
exposes a single ``generate_sample_batch`` helper that picks between
two strategies:

* ``"uniform"`` — the existing pseudo-random uniform sampling via
  ``numpy.random.Generator.uniform``. Default and unchanged.
* ``"sobol"`` — a Sobol low-discrepancy sequence via
  ``scipy.stats.qmc.Sobol`` scaled to ``[low, high]``. Low-discrepancy
  sequences cover the hypercube more evenly than i.i.d. uniform draws,
  which reduces the sample count needed to estimate quantities like
  expressibility's KL divergence and entanglement capability's mean by
  roughly 30-50% in typical settings.

Both paths consume the caller-supplied ``rng`` so the existing
``seed`` parameter on the public analysis functions stays the single
source of randomness — including for Sobol scrambling. This keeps
reproducibility byte-identical for a given ``(seed, sampling)``
pair.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

SamplingMethod = Literal["uniform", "sobol"]


def validate_sampling(sampling: SamplingMethod) -> None:
    """Validate the sampling argument with a clear error message.

    Parameters
    ----------
    sampling : {'uniform', 'sobol'}
        Caller's choice of input sampling strategy.

    Raises
    ------
    ValueError
        If ``sampling`` is not one of the accepted values. The error
        names exactly which strings are valid, so users can self-
        correct quickly.
    """
    if sampling not in ("uniform", "sobol"):
        raise ValueError(f"sampling must be 'uniform' or 'sobol', got {sampling!r}")


def generate_sample_batch(
    n_samples: int,
    n_features: int,
    input_range: tuple[float, float],
    rng: np.random.Generator,
    sampling: SamplingMethod = "uniform",
) -> NDArray[np.floating[Any]]:
    """Generate an ``(n_samples, n_features)`` input batch.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Dimensionality of each sample (matches the encoding's
        ``n_features``).
    input_range : (float, float)
        ``(low, high)`` range; samples are uniform / quasi-uniform
        within this interval.
    rng : numpy.random.Generator
        Random source. For the Sobol path it seeds the scrambling so
        the same ``rng`` state always produces the same sequence.
    sampling : {'uniform', 'sobol'}, default='uniform'
        Sampling strategy. See module docstring.

    Returns
    -------
    NDArray[np.floating]
        Float64 array of shape ``(n_samples, n_features)`` with
        values in ``[input_range[0], input_range[1])``.

    Raises
    ------
    ValueError
        If ``sampling`` is not one of the accepted strings.
    ImportError
        If ``sampling='sobol'`` is requested but ``scipy.stats.qmc``
        is unavailable. SciPy is a hard dependency of
        ``encoding_atlas`` so this is unlikely in practice, but the
        error message names the missing import.

    Notes
    -----
    The Sobol implementation deliberately suppresses the
    ``scipy.stats.qmc`` warning that fires when ``n_samples`` is not
    a power of two. The warning describes a balance property of
    Sobol' nets; the resulting samples are still a valid low-
    discrepancy draw and produce better convergence than i.i.d.
    uniform for the analysis statistics here. Sobol' shines when
    ``n_samples`` is a power of two — users who need the absolute
    best statistical properties should round their batch size up to
    the nearest power of two.
    """
    validate_sampling(sampling)
    low, high = float(input_range[0]), float(input_range[1])

    if sampling == "uniform":
        return rng.uniform(low, high, size=(n_samples, n_features)).astype(
            np.float64, copy=False
        )

    # sampling == "sobol"
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - scipy is a hard dep
        raise ImportError(
            "scipy.stats.qmc is required for sampling='sobol'. " "Install scipy >= 1.7."
        ) from exc

    # Seed the Sobol scrambler from the caller's Generator. Passing the
    # ``rng`` object (not a raw int) lets ``np.random.Generator`` mediate
    # the seeding so the public ``seed`` parameter remains the single
    # source of randomness across both sampling paths.
    sampler = qmc.Sobol(d=n_features, scramble=True, seed=rng)
    with warnings.catch_warnings():
        # See the Notes block above — the power-of-two balance warning
        # is informative for QMC theorists but noise for our users.
        warnings.simplefilter("ignore", UserWarning)
        unit_samples = sampler.random(n_samples)
    return cast(
        "NDArray[np.floating[Any]]",
        (unit_samples * (high - low) + low).astype(np.float64, copy=False),
    )
