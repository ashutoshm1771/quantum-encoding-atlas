"""Trainability estimation for quantum encodings."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding


def estimate_trainability(
    encoding: "BaseEncoding",
    n_samples: int = 100,
    seed: int | None = None,
) -> float:
    """Estimate the trainability of an encoding.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.
    n_samples : int, default=100
        Number of gradient samples.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    float
        Trainability estimate (higher is better).
    """
    # TODO: Implement trainability estimation
    raise NotImplementedError("Trainability estimation not yet implemented")
