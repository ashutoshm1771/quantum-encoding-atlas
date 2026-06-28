"""Queryable empirical atlas of encoding properties.

This subpackage exposes the project's measured benchmark results — circuit
resources, simulability, expressibility, entanglement, trainability, noise
resilience, and downstream VQC / quantum-kernel accuracy for all 16 encodings —
as a small, read-only API bundled with the installed package.

>>> from encoding_atlas.atlas import get_encoding_profile, pareto_front
>>> get_encoding_profile("angle").rank
1
>>> sorted(p.name for p in pareto_front())
['angle', 'basis', 'higher_order_angle', 'swap_equivariant']

See :mod:`encoding_atlas.atlas.profiles` for the full API and
:func:`encoding_atlas.atlas.atlas_metadata` for data provenance.
"""

from encoding_atlas.atlas.profiles import (
    EncodingProfile,
    atlas_metadata,
    available_encodings,
    get_encoding_profile,
    hypothesis_verdicts,
    list_metrics,
    list_profiles,
    pareto_front,
    rank_encodings,
)

__all__ = [
    "EncodingProfile",
    "atlas_metadata",
    "available_encodings",
    "get_encoding_profile",
    "hypothesis_verdicts",
    "list_metrics",
    "list_profiles",
    "pareto_front",
    "rank_encodings",
]
