"""Internal loader and name normalisation for the bundled empirical atlas.

The dataset shipped under ``encoding_atlas/atlas/data/master_summary.json`` is
the consolidated output of the project's 8-stage empirical pipeline (see
``experiments/``). It records, for all 16 encodings, the measured circuit
resources, simulability, expressibility, entanglement capability, trainability,
noise resilience, and downstream VQC / quantum-kernel accuracy that back every
table and figure in the accompanying paper.

This module is private: end users go through :mod:`encoding_atlas.atlas`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

# Package and filename of the bundled dataset (shipped as package data).
_DATA_PACKAGE = "encoding_atlas.atlas"
_DATA_SUBDIR = "data"
_DATA_FILENAME = "master_summary.json"

# Human-readable provenance, surfaced through ``atlas_metadata()``.
ATLAS_SOURCE = (
    "Consolidated from the 8-stage empirical pipeline in experiments/ "
    "(experiments/results/report/master_summary.json); the same data backs "
    "the tables and figures in the Quantum Encoding Atlas paper."
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


@lru_cache(maxsize=1)
def load_raw() -> dict[str, Any]:
    """Load and cache the raw atlas dataset as a plain dictionary.

    Returns
    -------
    dict
        Parsed contents of the bundled ``master_summary.json``. The returned
        object is cached and shared; callers must treat it as read-only.
    """
    text = (
        resources.files(_DATA_PACKAGE)
        .joinpath(_DATA_SUBDIR, _DATA_FILENAME)
        .read_text(encoding="utf-8")
    )
    data: dict[str, Any] = json.loads(text)
    return data


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
