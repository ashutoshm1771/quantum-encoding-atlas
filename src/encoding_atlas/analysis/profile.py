"""Unified encoding profiler.

A single entry point that characterises an *arbitrary* encoding across every
analysis axis and returns a structured, atlas-comparable profile. Without it,
fully characterising an encoding means calling a dozen separate functions, and
the atlas/recommender only know the 16 curated encodings — a researcher's own
encoding cannot be profiled or placed in context.

:func:`profile_encoding` computes the data-free axes (resources, simulability,
expressibility, entanglement, trainability, noise resilience, kernel
concentration) always, and the data-dependent kernel-geometry axes
(kernel-target alignment, geometric difference, effective dimension) when a
dataset ``(X, y)`` is supplied. Each axis is computed defensively, so a failure
on one axis records ``None`` with a note rather than aborting the whole profile.

:func:`compare_to_atlas` ranks a profiled encoding against the 16 built-in
encodings on the axes whose definitions match the bundled atlas.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis.concentration import compute_kernel_concentration
from encoding_atlas.analysis.entanglement import compute_entanglement_capability
from encoding_atlas.analysis.expressibility import compute_expressibility
from encoding_atlas.analysis.generalization import (
    compute_effective_dimension,
    compute_geometric_difference,
    compute_kernel_target_alignment,
)
from encoding_atlas.analysis.noise import compute_noise_resilience
from encoding_atlas.analysis.resources import count_resources
from encoding_atlas.analysis.simulability import check_simulability
from encoding_atlas.analysis.trainability import estimate_trainability

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

# Axes whose profiler definition matches the bundled atlas column, with
# ``True`` = higher is better (for ranking). ``depth`` is lower-is-better.
_ATLAS_COMPARABLE: dict[str, bool] = {
    "depth": False,
    "expressibility": True,
    "entanglement_capability": True,
    "trainability_estimate": True,
}


@dataclass(frozen=True)
class EncodingCharacterization:
    """A computed, multi-axis profile of a single encoding.

    Attributes
    ----------
    encoding_name : str
        The encoding's class name.
    n_qubits, n_features : int
        Circuit width and input dimensionality.
    is_simulable : bool or None
        Whether the encoding is classically efficiently simulable.
    metrics : Mapping[str, float | None]
        Measured metrics. Always includes ``depth``, ``expressibility``,
        ``entanglement_capability``, ``trainability_estimate``,
        ``noise_retained_fidelity`` and the concentration axis
        (``kernel_concentration_ratio``, ``kernel_offdiagonal_mean``,
        ``kernel_shots_per_entry``, ``kernel_is_concentrated``); includes
        ``kernel_target_alignment``, ``geometric_difference`` and
        ``effective_dimension`` when a dataset was supplied. A value is
        ``None`` if that axis could not be computed (see ``notes``).
        ``depth``/``expressibility``/``entanglement_capability``/
        ``trainability_estimate`` share the atlas's definitions;
        ``noise_retained_fidelity`` is the fidelity-based noise metric (distinct
        from the atlas's ``noise_resilience`` trade-off column).
        ``kernel_is_concentrated`` is a ``bool``, not a float.
    notes : Mapping[str, str]
        Per-axis failure reasons for any metric that is ``None``.
    """

    encoding_name: str
    n_qubits: int
    n_features: int
    is_simulable: bool | None
    metrics: Mapping[str, Any]
    notes: Mapping[str, str]

    def metric(self, key: str, default: float | None = None) -> float | None:
        """Return a metric value, or ``default`` if absent or uncomputable."""
        value = self.metrics.get(key, default)
        return default if value is None else value


def _safe(
    metrics: dict[str, Any],
    notes: dict[str, str],
    key: str,
    fn: Callable[[], float],
) -> None:
    """Run one axis, storing its value or (``None`` + failure note)."""
    try:
        metrics[key] = float(fn())
    except Exception as exc:  # noqa: BLE001 - one bad axis must not abort the profile
        metrics[key] = None
        notes[key] = f"{type(exc).__name__}: {exc}"
        logger.debug("Profile axis %r failed: %s", key, exc)


def profile_encoding(
    encoding: BaseEncoding,
    *,
    X: NDArray[np.floating[Any]] | None = None,
    y: NDArray[np.integer[Any]] | None = None,
    expressibility_samples: int = 500,
    entanglement_samples: int = 500,
    trainability_samples: int = 100,
    noise_level: str = "medium",
    noise_samples: int = 20,
    include_noise: bool = True,
    concentration_samples: int = 30,
    seed: int | None = None,
) -> EncodingCharacterization:
    """Characterise an encoding across every analysis axis.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to profile (may be any custom encoding, not just a
        built-in one).
    X : ndarray or None, default=None
        Optional feature matrix (``n_features`` must match the encoding). When
        given, the kernel-geometry axes are computed.
    y : ndarray or None, default=None
        Optional binary labels; required for kernel-target alignment.
    expressibility_samples, entanglement_samples, trainability_samples : int
        Sample counts for the corresponding stochastic estimators.
    noise_level : {"low", "medium", "high"}, default="medium"
        Noise preset for the noise-resilience axis.
    noise_samples : int, default=20
        Inputs averaged for the noise-resilience axis.
    include_noise : bool, default=True
        Whether to compute the (density-matrix) noise axis.
    concentration_samples : int, default=30
        Inputs used to build the kernel for the concentration axis. When ``X``
        is supplied, ``X`` is used instead and this is ignored.
    seed : int or None, default=None
        Base random seed for all stochastic axes.

    Returns
    -------
    EncodingCharacterization
        The computed multi-axis profile. Axes that fail are recorded as ``None``
        with a note, so the call always returns a profile for a valid encoding.
    """
    metrics: dict[str, Any] = {}
    notes: dict[str, str] = {}

    # --- Resources (depth) and simulability ---------------------------------
    n_qubits = int(encoding.n_qubits)
    try:
        resources = count_resources(encoding)
        metrics["depth"] = float(resources["depth"])
        n_qubits = int(resources.get("n_qubits", n_qubits))
    except Exception as exc:  # noqa: BLE001
        metrics["depth"] = None
        notes["depth"] = f"{type(exc).__name__}: {exc}"

    is_simulable: bool | None
    try:
        is_simulable = bool(check_simulability(encoding)["is_simulable"])
    except Exception as exc:  # noqa: BLE001
        is_simulable = None
        notes["is_simulable"] = f"{type(exc).__name__}: {exc}"

    # --- Stochastic descriptive axes ----------------------------------------
    _safe(
        metrics,
        notes,
        "expressibility",
        lambda: compute_expressibility(
            encoding, n_samples=expressibility_samples, seed=seed
        ),
    )
    _safe(
        metrics,
        notes,
        "entanglement_capability",
        lambda: compute_entanglement_capability(
            encoding, n_samples=entanglement_samples, seed=seed
        ),
    )
    _safe(
        metrics,
        notes,
        "trainability_estimate",
        lambda: estimate_trainability(
            encoding, n_samples=trainability_samples, seed=seed
        ),
    )

    # --- Noise resilience (fidelity-based) ----------------------------------
    if include_noise:
        _safe(
            metrics,
            notes,
            "noise_retained_fidelity",
            lambda: compute_noise_resilience(
                encoding,
                noise_level=noise_level,
                n_samples=noise_samples,
                seed=seed,
            ).retained_fidelity,
        )

    # --- Kernel concentration (data-free unless X is supplied) --------------
    # Always computed: it is the axis that says whether this encoding's kernel
    # still carries usable geometry at this circuit width, and it needs no
    # labels. Uses X when available so it describes the user's actual data.
    try:
        concentration = compute_kernel_concentration(
            encoding,
            X,
            n_samples=concentration_samples,
            seed=seed,
        )
        metrics["kernel_concentration_ratio"] = concentration.concentration_ratio
        metrics["kernel_offdiagonal_mean"] = concentration.offdiagonal_mean
        metrics["kernel_shots_per_entry"] = concentration.shots_per_entry
        metrics["kernel_is_concentrated"] = concentration.is_concentrated
    except Exception as exc:  # noqa: BLE001 - one bad axis must not abort
        for key in (
            "kernel_concentration_ratio",
            "kernel_offdiagonal_mean",
            "kernel_shots_per_entry",
            "kernel_is_concentrated",
        ):
            metrics[key] = None
        notes["kernel_concentration_ratio"] = f"{type(exc).__name__}: {exc}"
        logger.debug("Profile axis 'kernel_concentration_ratio' failed: %s", exc)

    # --- Data-dependent kernel-geometry axes --------------------------------
    if X is not None:
        _safe(
            metrics,
            notes,
            "geometric_difference",
            lambda: compute_geometric_difference(encoding, X),
        )
        _safe(
            metrics,
            notes,
            "effective_dimension",
            lambda: compute_effective_dimension(encoding, X),
        )
        if y is not None:
            _safe(
                metrics,
                notes,
                "kernel_target_alignment",
                lambda: compute_kernel_target_alignment(encoding, X, y),
            )

    return EncodingCharacterization(
        encoding_name=type(encoding).__name__,
        n_qubits=n_qubits,
        n_features=int(encoding.n_features),
        is_simulable=is_simulable,
        metrics=MappingProxyType(metrics),
        notes=MappingProxyType(notes),
    )


def atlas_comparable_metrics() -> list[str]:
    """Return the metric keys that can be compared against the atlas."""
    return list(_ATLAS_COMPARABLE)


def compare_to_atlas(
    characterization: EncodingCharacterization,
    metric: str,
) -> dict[str, Any]:
    """Rank a profiled encoding against the 16 built-in encodings on ``metric``.

    Only axes that share the atlas's definition are supported (see
    :func:`atlas_comparable_metrics`): ``depth`` (lower is better),
    ``expressibility``, ``entanglement_capability`` and
    ``trainability_estimate`` (higher is better).

    Parameters
    ----------
    characterization : EncodingCharacterization
        A profile produced by :func:`profile_encoding`.
    metric : str
        The axis to compare on.

    Returns
    -------
    dict
        ``{"metric", "value", "rank", "n_atlas", "percentile", "beats",
        "atlas_min", "atlas_median", "atlas_max", "higher_is_better"}`` where
        ``rank`` is 1-based (1 = better than every atlas encoding) and
        ``percentile`` is the fraction of atlas encodings the value beats.

    Raises
    ------
    ValueError
        If ``metric`` is not atlas-comparable or was not computed.
    """
    from encoding_atlas.atlas import list_profiles

    if metric not in _ATLAS_COMPARABLE:
        raise ValueError(
            f"{metric!r} is not atlas-comparable; choose from "
            f"{atlas_comparable_metrics()}."
        )
    value = characterization.metric(metric)
    if value is None:
        raise ValueError(
            f"metric {metric!r} was not computed for this encoding "
            f"(note: {characterization.notes.get(metric, 'not requested')})."
        )

    higher_is_better = _ATLAS_COMPARABLE[metric]
    atlas_values = [
        float(p.metrics[metric])
        for p in list_profiles()
        if p.metrics.get(metric) is not None
    ]
    beats = sum(
        1 for v in atlas_values if (value > v if higher_is_better else value < v)
    )
    n_atlas = len(atlas_values)
    return {
        "metric": metric,
        "value": float(value),
        "rank": n_atlas - beats + 1,
        "n_atlas": n_atlas,
        "percentile": (100.0 * beats / n_atlas) if n_atlas else float("nan"),
        "beats": beats,
        "atlas_min": min(atlas_values) if atlas_values else float("nan"),
        "atlas_median": (
            float(np.median(atlas_values)) if atlas_values else float("nan")
        ),
        "atlas_max": max(atlas_values) if atlas_values else float("nan"),
        "higher_is_better": higher_is_better,
    }
