"""Queryable API over the bundled fidelity-kernel concentration scan.

The accuracy stages of the empirical pipeline ran at ``n_features`` in
``{2, 4}``, so the atlas's published ranking is measured entirely in the regime
*before* kernel concentration switches on. This module serves the companion
scan that sweeps 2-8 qubits and records, per encoding, how quickly the
fidelity kernel's off-diagonal variance falls to the Haar floor — the point
past which a quantum-kernel method has no geometry left to learn from.

Reading the numbers
-------------------
``concentration_ratio`` is the off-diagonal variance divided by the Haar-random
variance ``(d - 1) / (d^2 (d + 1))`` at ``d = 2^n_qubits``. Near 1 means the
kernel has collapsed onto the floor; larger means structure remains.
``horizon`` is the narrowest measured width from which the encoding stays at
the floor, or ``None`` if it never gets there over the measured range.

The result lines up with the benchmark ranking: the encodings with a finite
horizon are precisely the high-expressibility feature maps the benchmark ranks
worst, which is the mechanism behind its headline negative result.

Examples
--------
>>> from encoding_atlas.atlas import get_concentration_profile
>>> get_concentration_profile("iqp").horizon
2
>>> get_concentration_profile("angle").horizon is None
True

See :func:`concentration_metadata` for provenance and the exact protocol.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from encoding_atlas.atlas._data import (
    CONCENTRATION_SOURCE,
    canonical_name,
    load_concentration_raw,
)

# Default ``concentration_ratio`` cut, mirroring
# ``encoding_atlas.analysis.concentration.CONCENTRATION_THRESHOLD``. Duplicated
# rather than imported so this module stays free of the analysis package's
# NumPy-heavy import chain.
_DEFAULT_THRESHOLD: float = 2.0


@dataclass(frozen=True)
class ConcentrationPoint:
    """One measured ``(width, concentration)`` point for an encoding.

    Attributes
    ----------
    n_features : int
        Feature count the encoding was built with.
    n_qubits : int
        Resulting circuit width. Equal to ``n_features`` for most encodings;
        ``ceil(log2(n_features))`` for amplitude encoding.
    concentration_ratio : float
        Off-diagonal kernel variance divided by the Haar variance at this
        width. Near 1 means the kernel is at the floor.
    mean_ratio : float
        Off-diagonal kernel mean divided by the Haar mean. Reported for
        completeness only — it is 1 for any ensemble of independent random
        single-qubit states as well as for Haar, so it cannot discriminate.
    offdiagonal_mean, offdiagonal_variance : float
        The raw measured moments of the off-diagonal entries.
    shots_per_entry : float
        Shots per kernel entry needed for the compute-uncompute estimator's
        standard error to fall below half the measured spread. A noiseless
        lower bound on hardware cost.
    is_concentrated : bool
        Whether ``concentration_ratio`` was below the scan's threshold.
    """

    n_features: int
    n_qubits: int
    concentration_ratio: float
    mean_ratio: float
    offdiagonal_mean: float
    offdiagonal_variance: float
    shots_per_entry: float
    is_concentrated: bool


@dataclass(frozen=True)
class ConcentrationProfile:
    """Measured kernel-concentration scaling for a single encoding.

    Attributes
    ----------
    name : str
        Canonical (registry-primary) encoding name, e.g. ``"qaoa"``.
    display_name : str
        Human-friendly class name, e.g. ``"QAOAEncoding"``.
    params : Mapping[str, Any]
        Fixed circuit parameters the scan used (matching the benchmark's
        kernel stage), e.g. ``{"reps": 2}``.
    points : tuple[ConcentrationPoint, ...]
        Per-width measurements, ordered by circuit width.
    decay_rate : float or None
        Per-qubit factor by which the off-diagonal **variance** shrinks. The
        Haar floor itself falls at 4.0, so a rate near 4 means the encoding
        scrambles as fast as it can. ``None`` if unfittable (too few widths).
    mean_decay_rate : float or None
        The same for the off-diagonal mean; the Haar floor falls at 2.0.
    haar_normalized_slope : float or None
        Slope of ``log(concentration_ratio)`` against qubit count. Negative
        means the kernel is collapsing towards the floor as the circuit
        widens; positive means it is pulling away.
    r_squared : float or None
        Quality of the log-linear variance fit, in ``[0, 1]``.
    horizon : int or None
        Narrowest measured width from which the encoding stays at the Haar
        floor, or an extrapolation when the trend points there. ``None`` means
        the encoding shows no sign of reaching the floor over 2-8 qubits.
    skipped : Mapping[str, str]
        Feature counts the scan could not measure (an encoding requiring
        exactly two features, or an even count), mapped to the reason.
    """

    name: str
    display_name: str
    params: Mapping[str, Any]
    points: tuple[ConcentrationPoint, ...]
    decay_rate: float | None
    mean_decay_rate: float | None
    haar_normalized_slope: float | None
    r_squared: float | None
    horizon: int | None
    skipped: Mapping[str, str]

    def at_features(self, n_features: int) -> ConcentrationPoint:
        """Return the measured point closest to ``n_features``.

        The scan measured a fixed grid of feature counts, so this snaps to the
        nearest measured configuration rather than interpolating. Ties break
        towards the wider circuit, which is the conservative direction for a
        concentration question.

        Parameters
        ----------
        n_features : int
            Feature count of interest. Must be a positive integer.

        Returns
        -------
        ConcentrationPoint
            The nearest measured point.

        Raises
        ------
        ValueError
            If ``n_features`` is not a positive integer.
        RuntimeError
            If this profile has no measured points at all.
        """
        if isinstance(n_features, bool) or not isinstance(n_features, int):
            raise ValueError(
                f"n_features must be a positive integer, got {n_features!r}"
            )
        if n_features < 1:
            raise ValueError(
                f"n_features must be a positive integer, got {n_features!r}"
            )
        if not self.points:
            raise RuntimeError(
                f"encoding {self.name!r} has no measured concentration points"
            )
        return min(
            self.points,
            key=lambda p: (abs(p.n_features - n_features), -p.n_features),
        )

    def is_concentrated_at(
        self, n_features: int, *, threshold: float = _DEFAULT_THRESHOLD
    ) -> bool:
        """Whether the kernel is at the Haar floor at roughly ``n_features``.

        Uses the nearest measured point (see :meth:`at_features`).

        Parameters
        ----------
        n_features : int
            Feature count of interest.
        threshold : float, default=2.0
            ``concentration_ratio`` defining the floor. Must be positive.

        Returns
        -------
        bool
            ``True`` when the nearest measured ``concentration_ratio`` falls
            below ``threshold``.

        Raises
        ------
        ValueError
            If ``threshold`` is not positive, or ``n_features`` is invalid.
        """
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        return self.at_features(n_features).concentration_ratio < threshold


def _optional_float(value: Any) -> float | None:
    """Coerce a JSON value to ``float`` or ``None`` (``null`` means unfittable)."""
    return None if value is None else float(value)


def _build_point(raw: Mapping[str, Any]) -> ConcentrationPoint:
    """Construct a :class:`ConcentrationPoint` from one raw record."""
    return ConcentrationPoint(
        n_features=int(raw["n_features"]),
        n_qubits=int(raw["n_qubits"]),
        concentration_ratio=float(raw["concentration_ratio"]),
        mean_ratio=float(raw["mean_ratio"]),
        offdiagonal_mean=float(raw["offdiagonal_mean"]),
        offdiagonal_variance=float(raw["offdiagonal_variance"]),
        # A null shot cost means "no spread at all", i.e. unbounded.
        shots_per_entry=(
            float("inf")
            if raw["shots_per_entry"] is None
            else float(raw["shots_per_entry"])
        ),
        is_concentrated=bool(raw["is_concentrated"]),
    )


def _build_profile(raw: Mapping[str, Any]) -> ConcentrationProfile:
    """Construct a :class:`ConcentrationProfile` from one raw dataset entry."""
    points = tuple(
        sorted(
            (_build_point(p) for p in raw["points"]),
            key=lambda p: (p.n_qubits, p.n_features),
        )
    )
    horizon = raw["concentration_horizon"]
    return ConcentrationProfile(
        name=canonical_name(str(raw["encoding"])),
        display_name=str(raw["display_name"]),
        params=MappingProxyType(dict(raw["params"])),
        points=points,
        decay_rate=_optional_float(raw["decay_rate"]),
        mean_decay_rate=_optional_float(raw["mean_decay_rate"]),
        haar_normalized_slope=_optional_float(raw["haar_normalized_slope"]),
        r_squared=_optional_float(raw["r_squared"]),
        horizon=None if horizon is None else int(horizon),
        skipped=MappingProxyType(dict(raw["skipped"])),
    )


@lru_cache(maxsize=1)
def _concentration_profiles() -> tuple[ConcentrationProfile, ...]:
    """All concentration profiles, ordered by canonical name."""
    raw = load_concentration_raw()
    profiles = [_build_profile(entry) for entry in raw["encodings"].values()]
    return tuple(sorted(profiles, key=lambda p: p.name))


@lru_cache(maxsize=1)
def _concentration_index() -> dict[str, ConcentrationProfile]:
    """Lookup index accepting canonical, dataset, and display-name keys."""
    by_canonical = {p.name: p for p in _concentration_profiles()}
    index: dict[str, ConcentrationProfile] = {}
    for key, entry in load_concentration_raw()["encodings"].items():
        profile = by_canonical[canonical_name(str(entry["encoding"]))]
        for alias in (key, profile.name, profile.display_name):
            index[alias.lower()] = profile
    return index


def get_concentration_profile(name: str) -> ConcentrationProfile:
    """Return the measured concentration profile for a single encoding.

    Parameters
    ----------
    name : str
        Encoding identifier. Canonical names (``"qaoa"``), dataset aliases
        (``"qaoa_encoding"``), and class display names (``"QAOAEncoding"``)
        are all accepted, case-insensitively.

    Returns
    -------
    ConcentrationProfile
        The encoding's measured concentration scaling.

    Raises
    ------
    KeyError
        If ``name`` does not correspond to any scanned encoding.

    Examples
    --------
    >>> profile = get_concentration_profile("iqp")
    >>> profile.horizon
    2
    """
    index = _concentration_index()
    key = name.strip().lower()
    if key not in index:
        available = ", ".join(sorted(p.name for p in _concentration_profiles()))
        raise KeyError(f"Unknown encoding {name!r}. Scanned encodings: {available}")
    return index[key]


def list_concentration_profiles() -> list[ConcentrationProfile]:
    """Return all concentration profiles, ordered by canonical name."""
    return list(_concentration_profiles())


def concentrated_encodings(
    *, threshold: float = _DEFAULT_THRESHOLD
) -> list[ConcentrationProfile]:
    """Return the encodings whose kernel reaches the Haar floor.

    These are the encodings for which a fidelity-kernel method stops working
    as the circuit widens, regardless of shot budget or noise level.

    Parameters
    ----------
    threshold : float, default=2.0
        ``concentration_ratio`` defining the floor. An encoding qualifies when
        its widest measured point falls below this. Must be positive.

    Returns
    -------
    list[ConcentrationProfile]
        Matching profiles, ordered by canonical name.

    Raises
    ------
    ValueError
        If ``threshold`` is not positive.

    Examples
    --------
    >>> sorted(p.name for p in concentrated_encodings())
    ['hamiltonian', 'iqp', 'pauli_feature_map', 'zz_feature_map']
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    return [
        p
        for p in _concentration_profiles()
        if p.points and p.points[-1].concentration_ratio < threshold
    ]


def concentration_metadata() -> dict[str, Any]:
    """Return provenance and the measurement protocol for the scan.

    Returns
    -------
    dict
        Schema version, encoding count, the full measurement protocol
        (feature counts, sample count, input range, sampling strategy,
        threshold, seed, backend), and a human-readable ``source`` string.
    """
    raw = load_concentration_raw()
    return {
        "schema_version": raw["schema_version"],
        "n_encodings": raw["n_encodings"],
        "protocol": copy.deepcopy(raw["protocol"]),
        "generated_by": raw["generated_by"],
        "source": CONCENTRATION_SOURCE,
    }
