"""Resource counting for quantum encodings."""

from typing import TYPE_CHECKING
from encoding_atlas.core.properties import ResourceSummary

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding


def count_resources(encoding: "BaseEncoding") -> ResourceSummary:
    """Count computational resources for an encoding.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.

    Returns
    -------
    ResourceSummary
        Summary of computational resources.
    """
    props = encoding.properties
    return ResourceSummary(
        n_qubits=props.n_qubits,
        depth=props.depth,
        gate_count=props.gate_count,
        two_qubit_gates=props.two_qubit_gates,
    )
