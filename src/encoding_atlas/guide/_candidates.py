"""Canonical candidate set for data-driven encoding screening.

The benchmark measured every encoding at a fixed configuration (see
``experiments/configs/stage6b_kernel.json``). Screening a user's data has to
use those same configurations, otherwise the measured alignment cannot be
compared against the atlas's recorded numbers. This module is the single
source of truth for that configuration; the concentration scan in
``experiments/`` imports it rather than keeping its own copy.

Not every encoding can be built at every width — SO(2) equivariance needs
exactly two features, the swap-equivariant map needs an even count, and the
symmetry-inspired map has its own constraint. :func:`build_candidates`
therefore reports what it could not build, with the reason, instead of
failing.

This module is private: end users go through
:func:`encoding_atlas.guide.screen_encodings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from encoding_atlas.core.registry import get_encoding

if TYPE_CHECKING:
    from collections.abc import Sequence

    from encoding_atlas.core.base import BaseEncoding

# Fixed (non-width) parameters per encoding, mirroring the benchmark's kernel
# stage so screened alignments are comparable with the bundled atlas column.
#
# ``trainable_encoding`` pins ``seed`` because its variational parameters are
# drawn at construction time; without it screening would not be reproducible.
# Holding them fixed also matches how the benchmark treats it — as a feature
# map, not a jointly-optimised model.
BENCHMARK_PARAMS: dict[str, dict[str, Any]] = {
    "angle": {"rotation": "Y"},
    "amplitude": {},
    "basis": {},
    "iqp": {"reps": 2},
    "zz_feature_map": {"reps": 2},
    "pauli_feature_map": {"reps": 2},
    "data_reuploading": {"n_layers": 2},
    "hardware_efficient": {"reps": 2},
    "higher_order_angle": {"order": 2},
    "qaoa_encoding": {"reps": 2},
    "hamiltonian_encoding": {"reps": 2},
    "symmetry_inspired": {"reps": 2},
    "trainable_encoding": {"n_layers": 2, "seed": 42},
    "so2_equivariant": {"max_angular_momentum": 2},
    "cyclic_equivariant": {"reps": 2},
    "swap_equivariant": {"reps": 2},
}


def default_candidate_names() -> list[str]:
    """Return the benchmarked encoding names, in a stable order."""
    return list(BENCHMARK_PARAMS)


def build_candidates(
    n_features: int,
    names: Sequence[str] | None = None,
) -> tuple[list[tuple[str, BaseEncoding]], dict[str, str]]:
    """Instantiate the candidate encodings that support ``n_features``.

    Parameters
    ----------
    n_features : int
        Feature count to build each encoding at.
    names : sequence of str, optional
        Restrict to these encodings. Defaults to all benchmarked ones.
        Duplicates are collapsed, and the requested order is preserved.

    Returns
    -------
    (list[tuple[str, BaseEncoding]], dict[str, str])
        The successfully built ``(name, encoding)`` pairs, and a mapping of
        the encodings that could not be built to the reason why.

    Raises
    ------
    ValueError
        If ``n_features`` is not a positive integer, or ``names`` contains an
        encoding outside the benchmarked set.
    """
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise ValueError(f"n_features must be a positive integer, got {n_features!r}")
    if n_features < 1:
        raise ValueError(f"n_features must be a positive integer, got {n_features!r}")

    if names is None:
        requested = default_candidate_names()
    else:
        seen: set[str] = set()
        requested = []
        for name in names:
            if name not in seen:
                seen.add(name)
                requested.append(name)
        unknown = [n for n in requested if n not in BENCHMARK_PARAMS]
        if unknown:
            raise ValueError(
                f"Unknown encoding(s) {unknown}. Choose from "
                f"{default_candidate_names()}."
            )

    built: list[tuple[str, BaseEncoding]] = []
    skipped: dict[str, str] = {}
    for name in requested:
        try:
            built.append(
                (
                    name,
                    get_encoding(name, n_features=n_features, **BENCHMARK_PARAMS[name]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - report and keep screening
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return built, skipped
