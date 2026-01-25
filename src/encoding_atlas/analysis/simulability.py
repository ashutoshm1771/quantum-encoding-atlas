"""Classical simulability checking for quantum encodings."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding


def check_simulability(
    encoding: "BaseEncoding",
) -> Literal["simulable", "conditionally_simulable", "not_simulable"]:
    """Check the classical simulability of an encoding.

    Parameters
    ----------
    encoding : BaseEncoding
        The encoding to analyze.

    Returns
    -------
    str
        Simulability status.
    """
    return encoding.properties.simulability
