"""Queryable API over the bundled feature-scaling sensitivity scan.

Encodings turn feature values into rotation angles, so the range features are
scaled into decides how much of each qubit's Bloch circle the data sweeps. The
empirical pipeline scales everything into ``[0, 2*pi]``, a full rotation period
— the regime the concentration scan identifies as maximally Haar-like. This
module serves the companion measurement of what that choice costs, and of how
far the benchmark's own headline correlation moves with it.

Two things are recorded
-----------------------
Per encoding, per range: mean kernel-target alignment, mean kernel accuracy and
mean kernel concentration across the benchmark datasets, plus the range that
maximised alignment.

Per range: the Spearman correlation between measured expressibility and
measured accuracy across encodings — the quantity hypothesis H1 is about.
Reported both over every measurable encoding and over the *atlas subset*, the
encodings the bundled atlas records an expressibility for, which is the set the
published analysis used and therefore the only one comparable with it.

The headline
------------
The correlation reverses sign across the range sweep: strongly positive at
``[0, pi/2]`` and negative at the published ``[0, 2*pi]``, both significant on
the atlas subset. Expressibility-versus-accuracy is therefore not a
scaling-invariant property of an encoding, and the range belongs in any report
of it.

Examples
--------
>>> from encoding_atlas.atlas import expressibility_accuracy_correlation
>>> rows = expressibility_accuracy_correlation()
>>> published = [r for r in rows if r["is_published_range"]][0]
>>> published["spearman_rho_atlas_subset"] < 0
True

See :func:`scaling_metadata` for provenance and the exact protocol.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from encoding_atlas.atlas._data import (
    SCALING_SOURCE,
    canonical_name,
    load_scaling_raw,
)


@dataclass(frozen=True)
class ScalingPoint:
    """One encoding measured at one feature-scaling range.

    Attributes
    ----------
    low, high : float
        The range the features were scaled into.
    width : float
        ``high - low``; ``2*pi`` is one full rotation period.
    mean_alignment : float
        Mean centred kernel-target alignment across the benchmark datasets.
    mean_accuracy : float
        Mean cross-validated quantum-kernel accuracy across those datasets.
    mean_concentration_ratio : float
        Mean kernel variance relative to the Haar floor. Falling towards 1 as
        the range widens is the mechanism behind the accuracy loss.
    expressibility : float or None
        Expressibility re-measured with inputs drawn from this same range, so
        both axes of the correlation describe one regime.
    n_datasets : int
        Datasets contributing to these means.
    """

    low: float
    high: float
    width: float
    mean_alignment: float
    mean_accuracy: float
    mean_concentration_ratio: float
    expressibility: float | None
    n_datasets: int


@dataclass(frozen=True)
class ScalingProfile:
    """Measured feature-scaling sensitivity for a single encoding.

    Attributes
    ----------
    name : str
        Canonical (registry-primary) encoding name.
    display_name : str
        Human-friendly class name.
    params : Mapping[str, Any]
        Fixed circuit parameters the scan used.
    points : tuple[ScalingPoint, ...]
        Per-range measurements, ordered by increasing range width.
    best_range : tuple[float, float]
        The range that maximised alignment.
    accuracy_spread : float
        Highest minus lowest mean accuracy across ranges. For several
        encodings this exceeds the spread *between* encodings, which is the
        point: scaling is not a minor knob.
    """

    name: str
    display_name: str
    params: Mapping[str, Any]
    points: tuple[ScalingPoint, ...]
    best_range: tuple[float, float]
    accuracy_spread: float

    def at_range(self, low: float, high: float) -> ScalingPoint | None:
        """Return the measurement for an exact range, or ``None``."""
        for point in self.points:
            if point.low == low and point.high == high:
                return point
        return None

    @property
    def published(self) -> ScalingPoint | None:
        """The measurement at the pipeline's own ``[0, 2*pi]`` range."""
        low, high = scaling_metadata()["protocol"]["published_range"]
        return self.at_range(float(low), float(high))


def _build_point(raw: Mapping[str, Any]) -> ScalingPoint:
    """Construct a :class:`ScalingPoint` from one raw record."""
    low, high = float(raw["low"]), float(raw["high"])
    expressibility = raw.get("expressibility")
    return ScalingPoint(
        low=low,
        high=high,
        width=high - low,
        mean_alignment=float(raw["mean_alignment"]),
        mean_accuracy=float(raw["mean_accuracy"]),
        mean_concentration_ratio=float(raw["mean_concentration_ratio"]),
        expressibility=None if expressibility is None else float(expressibility),
        n_datasets=int(raw["n_datasets"]),
    )


def _build_profile(raw: Mapping[str, Any]) -> ScalingProfile:
    """Construct a :class:`ScalingProfile` from one raw dataset entry."""
    points = tuple(
        sorted((_build_point(p) for p in raw["points"]), key=lambda p: p.width)
    )
    best = raw["best_range"]
    return ScalingProfile(
        name=canonical_name(str(raw["encoding"])),
        display_name=str(raw["display_name"]),
        params=MappingProxyType(dict(raw["params"])),
        points=points,
        best_range=(float(best[0]), float(best[1])),
        accuracy_spread=float(raw["accuracy_spread"]),
    )


@lru_cache(maxsize=1)
def _scaling_profiles() -> tuple[ScalingProfile, ...]:
    """All scaling profiles, ordered by canonical name."""
    raw = load_scaling_raw()
    profiles = [_build_profile(entry) for entry in raw["encodings"].values()]
    return tuple(sorted(profiles, key=lambda p: p.name))


@lru_cache(maxsize=1)
def _scaling_index() -> dict[str, ScalingProfile]:
    """Lookup index accepting canonical, dataset, and display-name keys."""
    by_canonical = {p.name: p for p in _scaling_profiles()}
    index: dict[str, ScalingProfile] = {}
    for key, entry in load_scaling_raw()["encodings"].items():
        profile = by_canonical[canonical_name(str(entry["encoding"]))]
        for alias in (key, profile.name, profile.display_name):
            index[alias.lower()] = profile
    return index


def get_scaling_profile(name: str) -> ScalingProfile:
    """Return the measured scaling sensitivity for a single encoding.

    Parameters
    ----------
    name : str
        Encoding identifier. Canonical names, dataset aliases, and class
        display names are all accepted, case-insensitively.

    Returns
    -------
    ScalingProfile

    Raises
    ------
    KeyError
        If ``name`` does not correspond to any scanned encoding.

    Examples
    --------
    >>> profile = get_scaling_profile("iqp")
    >>> profile.accuracy_spread > 0.2
    True
    """
    index = _scaling_index()
    key = name.strip().lower()
    if key not in index:
        available = ", ".join(sorted(p.name for p in _scaling_profiles()))
        raise KeyError(f"Unknown encoding {name!r}. Scanned encodings: {available}")
    return index[key]


def list_scaling_profiles() -> list[ScalingProfile]:
    """Return all scaling profiles, ordered by canonical name."""
    return list(_scaling_profiles())


def scaling_sensitive_encodings(*, threshold: float = 0.15) -> list[ScalingProfile]:
    """Return encodings whose accuracy moves more than ``threshold`` with scaling.

    These are the encodings for which the feature range is not a detail: report
    it, and tune it before drawing conclusions.

    Parameters
    ----------
    threshold : float, default=0.15
        Minimum accuracy spread across the scanned ranges. Must be
        non-negative.

    Returns
    -------
    list[ScalingProfile]
        Matching profiles, most sensitive first.

    Raises
    ------
    ValueError
        If ``threshold`` is negative.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    matches = [p for p in _scaling_profiles() if p.accuracy_spread > threshold]
    return sorted(matches, key=lambda p: (-p.accuracy_spread, p.name))


def expressibility_accuracy_correlation() -> list[dict[str, Any]]:
    """Return the expressibility-accuracy correlation at each scaling range.

    This is the scan's scientific payload: hypothesis H1 evaluated as a
    function of the preprocessing choice it was originally measured under.

    Returns
    -------
    list[dict]
        One entry per range, ordered as scanned, each with ``low``, ``high``,
        ``spearman_rho`` / ``p_value`` / ``n_encodings`` over every measurable
        encoding, the same three over the ``_atlas_subset`` (the encodings the
        atlas reports an expressibility for — the set the published analysis
        used), and ``is_published_range``. A deep copy is returned, so callers
        cannot mutate the cached dataset.
    """
    return copy.deepcopy(
        list(load_scaling_raw()["expressibility_accuracy_correlation"])
    )


def scaling_metadata() -> dict[str, Any]:
    """Return provenance and the measurement protocol for the scan.

    Returns
    -------
    dict
        Schema version, encoding count, the full protocol (ranges, datasets,
        sample and fold counts, seed, backend, and the pipeline's own
        ``published_range``), and a human-readable ``source`` string.
    """
    raw = load_scaling_raw()
    return {
        "schema_version": raw["schema_version"],
        "n_encodings": raw["n_encodings"],
        "protocol": copy.deepcopy(raw["protocol"]),
        "generated_by": raw["generated_by"],
        "source": SCALING_SOURCE,
    }
