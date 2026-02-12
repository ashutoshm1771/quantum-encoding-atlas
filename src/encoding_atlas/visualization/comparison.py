"""Encoding comparison visualization tools.

This module provides functions to visually compare quantum encodings,
helping users understand trade-offs between different encoding strategies.

Example
-------
>>> from encoding_atlas.visualization import compare_encodings
>>> compare_encodings(['angle', 'iqp', 'amplitude'], n_features=4)

>>> # Or with matplotlib figure output
>>> fig = compare_encodings(['angle', 'iqp'], n_features=8, output='matplotlib')
>>> fig.savefig('comparison.png')
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class EncodingComparison:
    """Container for encoding comparison data.

    Attributes
    ----------
    name : str
        Name of the encoding.
    encoding : BaseEncoding
        The encoding instance.
    properties : EncodingProperties
        Computed properties of the encoding.
    config : dict
        Configuration used to create the encoding.
    """

    name: str
    encoding: Any  # BaseEncoding, but avoid circular import
    properties: Any  # EncodingProperties
    config: dict[str, Any]


def _get_encoding_data(
    encoding_names: Sequence[str],
    n_features: int,
    configs: dict[str, dict[str, Any]] | None = None,
) -> list[EncodingComparison]:
    """Gather encoding data for comparison.

    Parameters
    ----------
    encoding_names : Sequence[str]
        Names of encodings to compare.
    n_features : int
        Number of features for all encodings.
    configs : dict, optional
        Custom configurations per encoding. Keys are encoding names,
        values are dicts of parameters to override defaults.

    Returns
    -------
    list[EncodingComparison]
        List of comparison data for each encoding.
    """
    from encoding_atlas import get_encoding

    configs = configs or {}
    results = []

    for name in encoding_names:
        try:
            config = configs.get(name, {})
            encoding = get_encoding(name, n_features=n_features, **config)
            properties = encoding.properties

            results.append(
                EncodingComparison(
                    name=name,
                    encoding=encoding,
                    properties=properties,
                    config=config,
                )
            )
        except Exception as e:
            # Skip encodings that fail (e.g., amplitude needs power of 2)
            import warnings

            warnings.warn(
                f"Could not create {name} with n_features={n_features}: {e}",
                stacklevel=2,
            )

    return results


def _create_bar(value: float, max_value: float, width: int = 20) -> str:
    """Create an ASCII bar for visualization."""
    if max_value == 0:
        return ""
    filled = int((value / max_value) * width)
    return "█" * filled


def _format_text_comparison(
    data: list[EncodingComparison],
    n_features: int,
    show_notes: bool = False,
) -> str:
    """Format comparison as ASCII text table.

    Parameters
    ----------
    data : list[EncodingComparison]
        Encoding comparison data.
    n_features : int
        Number of features used.
    show_notes : bool
        Whether to include encoding notes.

    Returns
    -------
    str
        Formatted ASCII comparison table.
    """
    if not data:
        return "No encodings to compare."

    lines = []
    width = 78

    # Header
    lines.append("┌" + "─" * (width - 2) + "┐")
    title = f"ENCODING COMPARISON (n_features={n_features})"
    padding = (width - 2 - len(title)) // 2
    lines.append(
        "│" + " " * padding + title + " " * (width - 2 - padding - len(title)) + "│"
    )
    lines.append("├" + "─" * (width - 2) + "┤")

    # Extract max values for scaling bars
    max_qubits = max(d.properties.n_qubits for d in data)
    max_depth = max(d.properties.depth for d in data)
    max_gates = max(d.properties.gate_count for d in data)
    max_two_qubit = max(d.properties.two_qubit_gates for d in data) or 1

    # Find max name length for alignment
    max_name_len = max(len(d.name) for d in data)

    # Qubits section
    lines.append("│" + " " * (width - 2) + "│")
    lines.append("│  QUBITS" + " " * 28 + "CIRCUIT DEPTH" + " " * (width - 51) + "│")
    lines.append("│  " + "─" * 6 + " " * 29 + "─" * 13 + " " * (width - 52) + "│")

    for d in data:
        qubits_bar = _create_bar(d.properties.n_qubits, max_qubits, 15)
        depth_bar = _create_bar(d.properties.depth, max_depth, 15)
        name_padded = d.name.ljust(max_name_len)
        left = f"  {name_padded} {qubits_bar} {d.properties.n_qubits}"
        right = f"{name_padded} {depth_bar} {d.properties.depth}"
        # Combine left and right sections
        line = f"│  {name_padded} {qubits_bar:<15} {d.properties.n_qubits:<4}   {name_padded} {depth_bar:<15} {d.properties.depth:<4}"
        # Pad to width
        line = line[: width - 1].ljust(width - 1) + "│"
        lines.append(line)

    # Gate count section
    lines.append("│" + " " * (width - 2) + "│")
    lines.append(
        "│  GATE COUNT" + " " * 24 + "TWO-QUBIT GATES" + " " * (width - 53) + "│"
    )
    lines.append("│  " + "─" * 10 + " " * 25 + "─" * 15 + " " * (width - 54) + "│")

    for d in data:
        gates_bar = _create_bar(d.properties.gate_count, max_gates, 15)
        two_q_bar = _create_bar(d.properties.two_qubit_gates, max_two_qubit, 15)
        name_padded = d.name.ljust(max_name_len)
        line = f"│  {name_padded} {gates_bar:<15} {d.properties.gate_count:<4}   {name_padded} {two_q_bar:<15} {d.properties.two_qubit_gates:<4}"
        line = line[: width - 1].ljust(width - 1) + "│"
        lines.append(line)

    # Properties table
    lines.append("│" + " " * (width - 2) + "│")
    lines.append("│  PROPERTIES" + " " * (width - 14) + "│")
    lines.append("│  " + "─" * 10 + " " * (width - 14) + "│")

    # Header row
    header = f"│  {'Encoding':<{max_name_len}}  {'Entangling':<12}  {'Simulability':<20}  {'Trainability':<15}"
    header = header[: width - 1].ljust(width - 1) + "│"
    lines.append(header)

    sep = "│  " + "─" * (max_name_len + 2 + 12 + 2 + 20 + 2 + 15)
    sep = sep[: width - 1].ljust(width - 1) + "│"
    lines.append(sep)

    for d in data:
        entangling = "✓ Yes" if d.properties.is_entangling else "✗ No"
        simulability = d.properties.simulability.replace("_", " ").title()

        # Trainability bar
        trainability = d.properties.trainability_estimate or 0.5
        train_bar = _create_bar(trainability, 1.0, 8)
        train_str = f"{train_bar} {trainability:.1f}"

        row = f"│  {d.name:<{max_name_len}}  {entangling:<12}  {simulability:<20}  {train_str:<15}"
        row = row[: width - 1].ljust(width - 1) + "│"
        lines.append(row)

    # Notes section (optional)
    if show_notes:
        lines.append("│" + " " * (width - 2) + "│")
        lines.append("│  NOTES" + " " * (width - 9) + "│")
        lines.append("│  " + "─" * 5 + " " * (width - 9) + "│")

        for d in data:
            if d.properties.notes:
                note = (
                    d.properties.notes[:60] + "..."
                    if len(d.properties.notes) > 60
                    else d.properties.notes
                )
                row = f"│  {d.name}: {note}"
                row = row[: width - 1].ljust(width - 1) + "│"
                lines.append(row)

    lines.append("│" + " " * (width - 2) + "│")
    lines.append("└" + "─" * (width - 2) + "┘")

    return "\n".join(lines)


def _create_matplotlib_comparison(
    data: list[EncodingComparison],
    n_features: int,
    figsize: tuple[float, float] = (12, 10),
) -> Figure:
    """Create matplotlib figure for encoding comparison.

    Parameters
    ----------
    data : list[EncodingComparison]
        Encoding comparison data.
    n_features : int
        Number of features used.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    Figure
        Matplotlib figure with comparison plots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for graphical output. "
            "Install it with: pip install matplotlib"
        ) from e

    names = [d.name for d in data]
    n_encodings = len(names)

    # Extract metrics
    qubits = [d.properties.n_qubits for d in data]
    depths = [d.properties.depth for d in data]
    gate_counts = [d.properties.gate_count for d in data]
    single_qubit = [d.properties.single_qubit_gates for d in data]
    two_qubit = [d.properties.two_qubit_gates for d in data]
    trainability = [d.properties.trainability_estimate or 0.5 for d in data]

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_encodings))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f"Encoding Comparison (n_features={n_features})",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Qubits
    ax = axes[0, 0]
    bars = ax.barh(names, qubits, color=colors)
    ax.set_xlabel("Number of Qubits")
    ax.set_title("Qubit Count")
    ax.bar_label(bars, padding=3)

    # 2. Circuit Depth
    ax = axes[0, 1]
    bars = ax.barh(names, depths, color=colors)
    ax.set_xlabel("Depth")
    ax.set_title("Circuit Depth")
    ax.bar_label(bars, padding=3)

    # 3. Gate Count (stacked)
    ax = axes[0, 2]
    bars1 = ax.barh(names, single_qubit, label="Single-qubit", color=colors, alpha=0.7)
    bars2 = ax.barh(
        names, two_qubit, left=single_qubit, label="Two-qubit", color=colors, alpha=0.4
    )
    ax.set_xlabel("Gate Count")
    ax.set_title("Gate Composition")
    ax.legend(loc="lower right", fontsize=8)

    # 4. Trainability
    ax = axes[1, 0]
    bars = ax.barh(names, trainability, color=colors)
    ax.set_xlabel("Trainability Estimate")
    ax.set_title("Trainability")
    ax.set_xlim(0, 1)
    ax.bar_label(bars, fmt="%.2f", padding=3)

    # 5. Entanglement indicator
    ax = axes[1, 1]
    entangling = [1 if d.properties.is_entangling else 0 for d in data]
    bar_colors = ["#2ecc71" if e else "#e74c3c" for e in entangling]
    bars = ax.barh(names, [1] * n_encodings, color=bar_colors, alpha=0.7)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_title("Entanglement")

    # Add text labels
    for i, (_name, ent) in enumerate(zip(names, entangling)):
        label = "Entangling" if ent else "Product States"
        ax.text(0.5, i, label, ha="center", va="center", fontsize=10, fontweight="bold")

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")

    # Create summary text
    summary_lines = ["Simulability:\n"]
    for d in data:
        sim = d.properties.simulability.replace("_", " ").title()
        icon = "✓" if sim == "Simulable" else "⚠" if "Conditional" in sim else "✗"
        summary_lines.append(f"  {icon} {d.name}: {sim}\n")

    ax.text(
        0.1,
        0.9,
        "".join(summary_lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    return fig


def compare_encodings(
    encodings: Sequence[str],
    n_features: int,
    *,
    configs: dict[str, dict[str, Any]] | None = None,
    output: Literal["text", "matplotlib"] = "text",
    show_notes: bool = False,
    figsize: tuple[float, float] = (12, 10),
) -> str | Figure:
    """Compare multiple quantum encodings side by side.

    This function creates a visual comparison of encoding properties including
    qubit count, circuit depth, gate composition, trainability, and more.

    Parameters
    ----------
    encodings : Sequence[str]
        Names of encodings to compare. Available encodings can be listed
        with ``encoding_atlas.list_encodings()``.
    n_features : int
        Number of classical features. All encodings will be instantiated
        with this feature count.
    configs : dict, optional
        Custom configuration per encoding. Keys are encoding names, values
        are dicts of parameters. For example::

            configs={
                'angle': {'rotation': 'X'},
                'iqp': {'reps': 2, 'entanglement': 'circular'},
            }

    output : {'text', 'matplotlib'}, default='text'
        Output format:

        - 'text': Returns an ASCII art table (prints to console)
        - 'matplotlib': Returns a matplotlib Figure object

    show_notes : bool, default=False
        Whether to include encoding notes in text output.
    figsize : tuple, default=(12, 10)
        Figure size for matplotlib output.

    Returns
    -------
    str or Figure
        If output='text', returns the formatted comparison string.
        If output='matplotlib', returns a matplotlib Figure.

    Examples
    --------
    Basic comparison with text output:

    >>> from encoding_atlas.visualization import compare_encodings
    >>> print(compare_encodings(['angle', 'iqp', 'amplitude'], n_features=4))

    Comparison with matplotlib output:

    >>> fig = compare_encodings(
    ...     ['angle', 'iqp', 'zz_feature_map', 'hardware_efficient'],
    ...     n_features=8,
    ...     output='matplotlib'
    ... )
    >>> fig.savefig('encoding_comparison.png', dpi=150)

    Custom configurations:

    >>> compare_encodings(
    ...     ['angle', 'angle', 'iqp'],
    ...     n_features=4,
    ...     configs={
    ...         'angle': {'rotation': 'X', 'reps': 2},
    ...         'iqp': {'entanglement': 'full'},
    ...     }
    ... )

    See Also
    --------
    encoding_atlas.list_encodings : List all available encodings.
    encoding_atlas.get_encoding : Get a specific encoding by name.

    Notes
    -----
    Some encodings have constraints on n_features:

    - ``AmplitudeEncoding`` requires n_features to be a power of 2
    - Some entanglement patterns require n_features >= 2

    Encodings that fail to instantiate are silently skipped with a warning.
    """
    # Gather encoding data
    data = _get_encoding_data(encodings, n_features, configs)

    if not data:
        msg = "No encodings could be created with the given parameters."
        if output == "matplotlib":
            raise ValueError(msg)
        return msg

    # Generate output
    if output == "text":
        result = _format_text_comparison(data, n_features, show_notes)
        print(result)
        return result
    elif output == "matplotlib":
        return _create_matplotlib_comparison(data, n_features, figsize)
    else:
        raise ValueError(
            f"Unknown output format: {output!r}. Use 'text' or 'matplotlib'."
        )
