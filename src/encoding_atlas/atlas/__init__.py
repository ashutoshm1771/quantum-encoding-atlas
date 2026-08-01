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

A companion dataset records **fidelity-kernel concentration**: how quickly each
encoding's kernel collapses onto the Haar floor as the circuit widens. The
accuracy numbers above were measured at 2-4 qubits, so this is the axis that
says whether they transfer:

>>> from encoding_atlas.atlas import get_concentration_profile
>>> get_concentration_profile("iqp").horizon      # at the floor from 2 qubits
2
>>> get_concentration_profile("angle").horizon is None
True

A third records **feature-scaling sensitivity**: what the range features are
scaled into costs, and how far the study's own expressibility-accuracy
correlation moves with it:

>>> from encoding_atlas.atlas import expressibility_accuracy_correlation
>>> rows = expressibility_accuracy_correlation()
>>> [round(r["spearman_rho_atlas_subset"], 2) for r in rows]   # narrow -> wide
[0.58, 0.78, -0.29, -0.54]

See :mod:`encoding_atlas.atlas.profiles`,
:mod:`encoding_atlas.atlas.concentration` and
:mod:`encoding_atlas.atlas.scaling` for the full APIs, and
:func:`encoding_atlas.atlas.atlas_metadata` /
:func:`encoding_atlas.atlas.concentration_metadata` for data provenance.
"""

from encoding_atlas.atlas.concentration import (
    ConcentrationPoint,
    ConcentrationProfile,
    concentrated_encodings,
    concentration_metadata,
    get_concentration_profile,
    list_concentration_profiles,
)
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
from encoding_atlas.atlas.scaling import (
    ScalingPoint,
    ScalingProfile,
    expressibility_accuracy_correlation,
    get_scaling_profile,
    list_scaling_profiles,
    scaling_metadata,
    scaling_sensitive_encodings,
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
    # Kernel-concentration scan
    "ConcentrationPoint",
    "ConcentrationProfile",
    "concentrated_encodings",
    "concentration_metadata",
    "get_concentration_profile",
    "list_concentration_profiles",
    # Feature-scaling sensitivity scan
    "ScalingPoint",
    "ScalingProfile",
    "expressibility_accuracy_correlation",
    "get_scaling_profile",
    "list_scaling_profiles",
    "scaling_metadata",
    "scaling_sensitive_encodings",
]
