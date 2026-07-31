"""Queryable API over the bundled empirical Encoding Atlas.

This module turns the project's measured benchmark results into a small,
read-only, dependency-free API so that users of the installed package can query
*what was actually measured* for each encoding — not just compute circuits.

Examples
--------
>>> from encoding_atlas.atlas import get_encoding_profile, rank_encodings
>>> angle = get_encoding_profile("angle")
>>> angle.rank
1
>>> round(angle.metric("kernel_accuracy"), 3)
0.958
>>> [p.name for p in rank_encodings(by="kernel_accuracy", limit=3)]
['angle', 'cyclic_equivariant', 'qaoa']

All numbers come from ``master_summary.json``, the consolidated output of the
8-stage empirical pipeline in ``experiments/``. See :func:`atlas_metadata` for
provenance and :func:`hypothesis_verdicts` for the pre-registered hypothesis
outcomes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from encoding_atlas.atlas._data import ATLAS_SOURCE, canonical_name, load_raw

# Scalar (rankable) metric keys present in every profile's ``metrics`` mapping.
# Confidence-interval entries (``vqc_ci`` / ``kernel_ci``) are intentionally
# excluded here because they are pairs, not single sortable values.
#
# ``kernel_target_alignment`` is the benchmark's *validated* predictor of
# downstream kernel accuracy (Spearman rho = 0.91 across encodings and
# datasets), in contrast to ``expressibility``, which the study refutes
# (rho = -0.68). Both are rankable so the two can be compared directly.
_SCALAR_METRICS: tuple[str, ...] = (
    "depth",
    "expressibility",
    "entanglement_capability",
    "trainability_estimate",
    "noise_resilience",
    "vqc_accuracy",
    "kernel_accuracy",
    "kernel_target_alignment",
)

# Synthetic ranking keys that are not stored inside ``metrics``.
_SCORE_KEY = "score"
_RANK_KEY = "rank"

# Keys for which a *lower* value is better (so ascending order is the default).
_LOWER_IS_BETTER: frozenset[str] = frozenset({_RANK_KEY, "depth"})


@dataclass(frozen=True)
class EncodingProfile:
    """Measured empirical profile of a single encoding.

    Attributes
    ----------
    name : str
        Canonical (registry-primary) encoding name, e.g. ``"qaoa"``.
    display_name : str
        Human-friendly class name, e.g. ``"QAOAEncoding"``.
    family : str
        Taxonomic family the encoding belongs to.
    rank : int
        Overall benchmark rank (1 = best) by the composite trade-off score.
    score : float
        Composite trade-off score in ``[0, 1]`` (higher is better).
    is_pareto : bool
        Whether the encoding lies on the Pareto front across the benchmark
        objectives (accuracy, inverse depth, trainability, noise resilience).
    is_simulable : bool
        Whether the encoding is classically efficiently simulable.
    metrics : Mapping[str, Any]
        Read-only mapping of measured metrics. Scalar keys are listed by
        :func:`list_metrics`; ``vqc_ci`` and ``kernel_ci`` hold 95% confidence
        intervals as ``[low, high]`` pairs. Some entries may be ``None`` when a
        metric is not defined for that encoding (e.g. expressibility for
        basis/amplitude state-preparation circuits).

        ``kernel_target_alignment`` is the mean *centered* alignment over the
        benchmark's (configuration, dataset) pairs — the training-free quantity
        that predicts kernel accuracy. To measure it on your own data instead,
        see :func:`encoding_atlas.guide.screen_encodings`.
    """

    name: str
    display_name: str
    family: str
    rank: int
    score: float
    is_pareto: bool
    is_simulable: bool
    metrics: Mapping[str, Any]

    def metric(self, key: str, default: Any = None) -> Any:
        """Return a measured metric value, or ``default`` if absent/undefined.

        Parameters
        ----------
        key : str
            Metric name (see :func:`list_metrics`).
        default : Any
            Value to return if the metric is missing or recorded as ``None``.
        """
        value = self.metrics.get(key, default)
        return default if value is None else value


def _build_profile(raw_profile: Mapping[str, Any]) -> EncodingProfile:
    """Construct an :class:`EncodingProfile` from one raw dataset entry."""
    metrics: Mapping[str, Any] = MappingProxyType(dict(raw_profile["metrics"]))
    return EncodingProfile(
        name=canonical_name(str(raw_profile["encoding"])),
        display_name=str(raw_profile["display_name"]),
        family=str(raw_profile["family"]),
        rank=int(raw_profile["rank"]),
        score=float(raw_profile["score"]),
        is_pareto=bool(raw_profile["is_pareto"]),
        is_simulable=bool(raw_profile["is_simulable"]),
        metrics=metrics,
    )


@lru_cache(maxsize=1)
def _profiles_by_rank() -> tuple[EncodingProfile, ...]:
    """All profiles, ordered by benchmark rank (best first)."""
    raw = load_raw()
    profiles = [_build_profile(p) for p in raw["encoding_profiles"]]
    return tuple(sorted(profiles, key=lambda p: p.rank))


@lru_cache(maxsize=1)
def _profile_index() -> dict[str, EncodingProfile]:
    """Lookup index accepting canonical, dataset, and display-name keys."""
    by_canonical = {p.name: p for p in _profiles_by_rank()}
    index: dict[str, EncodingProfile] = {}
    for raw_profile in load_raw()["encoding_profiles"]:
        canonical = canonical_name(str(raw_profile["encoding"]))
        profile = by_canonical[canonical]
        for alias in (
            str(raw_profile["encoding"]),
            canonical,
            str(raw_profile["display_name"]),
        ):
            index[alias.lower()] = profile
    return index


def get_encoding_profile(name: str) -> EncodingProfile:
    """Return the measured profile for a single encoding.

    Parameters
    ----------
    name : str
        Encoding identifier. Canonical names (``"qaoa"``), dataset aliases
        (``"qaoa_encoding"``), and class display names (``"QAOAEncoding"``) are
        all accepted, case-insensitively.

    Returns
    -------
    EncodingProfile
        The encoding's empirical profile.

    Raises
    ------
    KeyError
        If ``name`` does not correspond to any benchmarked encoding.
    """
    index = _profile_index()
    key = name.strip().lower()
    if key not in index:
        available = ", ".join(available_encodings())
        raise KeyError(f"Unknown encoding {name!r}. Available encodings: {available}")
    return index[key]


def list_profiles() -> list[EncodingProfile]:
    """Return all 16 encoding profiles, ordered by benchmark rank (best first)."""
    return list(_profiles_by_rank())


def available_encodings() -> list[str]:
    """Return the sorted canonical names of all benchmarked encodings."""
    return sorted(p.name for p in _profiles_by_rank())


def list_metrics() -> list[str]:
    """Return the scalar metric keys available for ranking and lookup."""
    return list(_SCALAR_METRICS)


def _sort_key(by: str) -> Callable[[EncodingProfile], Any]:
    """Return a callable extracting the sort value for ranking key ``by``."""
    if by == _SCORE_KEY:
        return lambda p: p.score
    if by == _RANK_KEY:
        return lambda p: p.rank
    return lambda p: p.metrics[by]


def rank_encodings(
    by: str = _SCORE_KEY,
    *,
    ascending: bool | None = None,
    limit: int | None = None,
) -> list[EncodingProfile]:
    """Rank encodings by a measured metric or by the composite score.

    Parameters
    ----------
    by : str
        Ranking key: ``"score"``, ``"rank"``, or any scalar metric from
        :func:`list_metrics` (e.g. ``"kernel_accuracy"``, ``"depth"``).
    ascending : bool or None
        Sort direction. If ``None`` (default), a sensible direction is chosen
        per key: ascending for ``"rank"`` and ``"depth"`` (lower is better),
        descending otherwise (higher is better).
    limit : int or None
        If given, return at most this many profiles.

    Returns
    -------
    list[EncodingProfile]
        Profiles sorted as requested. Encodings whose value for ``by`` is
        undefined (``None``) are omitted from metric rankings.

    Raises
    ------
    ValueError
        If ``by`` is not a recognised ranking key, or ``limit`` is negative.
    """
    valid_keys = {_SCORE_KEY, _RANK_KEY, *_SCALAR_METRICS}
    if by not in valid_keys:
        raise ValueError(f"Cannot rank by {by!r}. Valid keys: {sorted(valid_keys)}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    profiles = list(_profiles_by_rank())
    if by in _SCALAR_METRICS:
        profiles = [p for p in profiles if p.metrics.get(by) is not None]

    if ascending is None:
        ascending = by in _LOWER_IS_BETTER

    ranked = sorted(profiles, key=_sort_key(by), reverse=not ascending)
    if limit is not None:
        ranked = ranked[:limit]
    return ranked


def pareto_front() -> list[EncodingProfile]:
    """Return the Pareto-optimal encoding profiles, ordered by rank.

    These are the encodings not dominated on the benchmark objectives
    (accuracy, inverse depth, trainability, noise resilience).
    """
    return [p for p in _profiles_by_rank() if p.is_pareto]


def hypothesis_verdicts() -> dict[str, Any]:
    """Return the pre-registered hypothesis outcomes (``H1``–``H7``).

    Each verdict carries its ``verdict`` (``"supported"`` / ``"refuted"`` /
    ``"inconclusive"``), ``confidence``, a plain-language ``evidence`` summary,
    and the supporting ``test_statistic``. A deep copy is returned so callers
    cannot mutate the cached dataset.
    """
    return copy.deepcopy(load_raw()["hypothesis_verdicts"])


def atlas_metadata() -> dict[str, Any]:
    """Return provenance and summary metadata for the bundled atlas.

    Returns
    -------
    dict
        Schema version, encoding count, benchmark objective names, Pareto-front
        size, per-stage success counts, and a human-readable ``source`` string.
    """
    raw = load_raw()
    pareto = raw["pareto_front"]
    return {
        "schema_version": raw["schema_version"],
        "n_encodings": raw["n_encodings"],
        "objective_names": list(pareto["objective_names"]),
        "n_pareto_optimal": pareto["n_pareto_optimal"],
        "stage_counts": copy.deepcopy(raw["stage_counts"]),
        "source": ATLAS_SOURCE,
    }
