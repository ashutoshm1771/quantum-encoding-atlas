"""Internal loaders and name normalisation for the bundled empirical atlas.

Two datasets ship as package data under ``encoding_atlas/atlas/data/``:

``master_summary.json``
    The consolidated output of the project's 8-stage empirical pipeline (see
    ``experiments/``). Records, for all 16 encodings, the measured circuit
    resources, simulability, expressibility, entanglement capability,
    trainability, noise resilience, and downstream VQC / quantum-kernel
    accuracy that back every table and figure in the accompanying paper.

``concentration.json``
    The fidelity-kernel concentration scan (``experiments/concentration_scan.py``),
    measuring how each encoding's kernel approaches the Haar floor as the
    circuit widens. This is the axis that says whether the accuracy numbers —
    measured at 2-4 qubits — transfer to wider circuits.

``scaling_sensitivity.json``
    The feature-scaling scan (``experiments/scaling_scan.py``), measuring what
    the range features are scaled into costs in alignment, accuracy and
    concentration — and how far the study's expressibility-versus-accuracy
    correlation moves with that choice.

This module is private: end users go through :mod:`encoding_atlas.atlas`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

# Package and filenames of the bundled datasets (shipped as package data).
_DATA_PACKAGE = "encoding_atlas.atlas"
_DATA_SUBDIR = "data"
_DATA_FILENAME = "master_summary.json"
_CONCENTRATION_FILENAME = "concentration.json"
_SCALING_FILENAME = "scaling_sensitivity.json"

# Human-readable provenance, surfaced through ``atlas_metadata()``.
ATLAS_SOURCE = (
    "Consolidated from the 8-stage empirical pipeline in experiments/ "
    "(experiments/results/report/master_summary.json); the same data backs "
    "the tables and figures in the Quantum Encoding Atlas paper."
)

# Provenance for the concentration scan, surfaced through
# ``concentration_metadata()``.
CONCENTRATION_SOURCE = (
    "Measured by experiments/concentration_scan.py: fidelity-kernel "
    "off-diagonal variance relative to the Haar floor, swept over 2-8 "
    "qubits with the same circuit parameters the accuracy stages used "
    "(experiments/configs/stage6b_kernel.json). Regenerate with "
    "'python -m experiments.concentration_scan'."
)

# Provenance for the feature-scaling scan, surfaced through
# ``scaling_metadata()``.
SCALING_SOURCE = (
    "Measured by experiments/scaling_scan.py: kernel-target alignment, "
    "quantum-kernel accuracy and kernel concentration per encoding across "
    "feature-scaling ranges, with the expressibility-accuracy correlation "
    "recomputed at each range. The empirical pipeline scales into "
    "[0, 2*pi] (experiments/datasets.py), which this scan records as "
    "'published_range'. Regenerate with 'python -m experiments.scaling_scan'."
)

# Canonical-name normalisation.
#
# The empirical dataset labels three encodings with identifiers that differ
# from the *primary* registry names used by ``encoding_atlas.encodings`` and
# ``encoding_atlas.guide.rules``. We map them onto the canonical names so the
# atlas, the recommender, and ``get_encoding()`` share one vocabulary. Every
# alias on the left is also a valid (secondary) registry name, so round-tripping
# through ``get_encoding()`` continues to work either way.
_ATLAS_TO_CANONICAL: dict[str, str] = {
    "qaoa_encoding": "qaoa",
    "hamiltonian_encoding": "hamiltonian",
    "trainable_encoding": "trainable",
}


def _read_bundled(filename: str) -> dict[str, Any]:
    """Parse one bundled JSON dataset from the package's ``data`` directory."""
    text = (
        resources.files(_DATA_PACKAGE)
        .joinpath(_DATA_SUBDIR)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )
    data: dict[str, Any] = json.loads(text)
    return data


@lru_cache(maxsize=1)
def load_raw() -> dict[str, Any]:
    """Load and cache the raw atlas dataset as a plain dictionary.

    Returns
    -------
    dict
        Parsed contents of the bundled ``master_summary.json``. The returned
        object is cached and shared; callers must treat it as read-only.
    """
    return _read_bundled(_DATA_FILENAME)


@lru_cache(maxsize=1)
def load_scaling_raw() -> dict[str, Any]:
    """Load and cache the raw feature-scaling dataset as a plain dictionary.

    Returns
    -------
    dict
        Parsed contents of the bundled ``scaling_sensitivity.json``. The
        returned object is cached and shared; callers must treat it as
        read-only.
    """
    return _read_bundled(_SCALING_FILENAME)


@lru_cache(maxsize=1)
def load_concentration_raw() -> dict[str, Any]:
    """Load and cache the raw concentration dataset as a plain dictionary.

    Returns
    -------
    dict
        Parsed contents of the bundled ``concentration.json``. The returned
        object is cached and shared; callers must treat it as read-only.
    """
    return _read_bundled(_CONCENTRATION_FILENAME)


def canonical_name(name: str) -> str:
    """Normalise an encoding identifier to its canonical registry name.

    Parameters
    ----------
    name : str
        An encoding identifier as it appears in the dataset (e.g.
        ``"qaoa_encoding"``).

    Returns
    -------
    str
        The canonical registry-primary name (e.g. ``"qaoa"``). Identifiers
        that are already canonical are returned unchanged.
    """
    return _ATLAS_TO_CANONICAL.get(name, name)
