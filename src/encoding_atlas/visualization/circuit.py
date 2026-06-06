"""Circuit and entanglement-structure visualization helpers.

This module renders the actual gate-level structure of an encoding (via the
target backend's drawing facility) and the qubit connectivity implied by
its entangling pairs.

Quick Start
-----------
>>> import numpy as np
>>> from encoding_atlas import IQPEncoding
>>> from encoding_atlas.visualization import (
...     visualize_circuit, plot_entanglement_graph,
... )
>>>
>>> enc = IQPEncoding(n_features=4)
>>> x = np.array([0.1, 0.2, 0.3, 0.4])
>>>
>>> # Render the actual circuit (PennyLane → matplotlib by default)
>>> fig = visualize_circuit(enc, x)
>>> fig.savefig("iqp_circuit.png", bbox_inches="tight")
>>>
>>> # Plot the entanglement connectivity graph
>>> graph_fig = plot_entanglement_graph(enc)

Design Notes
------------
* The functions delegate as much as possible to the backends'
  battle-tested drawing routines (``qml.draw_mpl``, ``QuantumCircuit.draw``,
  ``cirq.contrib.svg.circuit_to_svg``) rather than reimplementing gate
  rendering — keeps the code small and stays accurate when those
  libraries improve.
* matplotlib is treated as an optional dependency (declared under the
  ``[visualization]`` extra). All entry points raise a clear
  ``ImportError`` with installation instructions if matplotlib is not
  available, before doing any work.
* The Qiskit matplotlib drawer requires ``pylatexenc``; if it's missing
  we catch the cryptic upstream error and re-raise with a clear remedy.
* The Cirq backend has no native matplotlib renderer, so for
  ``output='mpl'`` we paint its text diagram into a monospaced Figure.
  Users get a uniform return type across all three backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union

from encoding_atlas.core.protocols import is_entanglement_queryable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from encoding_atlas.core.base import BaseEncoding
    from encoding_atlas.core.types import BackendType


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_matplotlib() -> Any:
    """Import matplotlib.pyplot or raise a clear, actionable error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization functions. "
            "Install with: pip install matplotlib "
            "(or pip install 'encoding-atlas[visualization]')"
        ) from exc
    return plt


def _require_networkx() -> Any:
    """Import networkx or raise a clear, actionable error."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for entanglement-graph visualization. "
            "Install with: pip install networkx"
        ) from exc
    return nx


def _render_pennylane(circuit: Any, output: str) -> Any:
    """Render a PennyLane qfunc as text or matplotlib Figure."""
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required to visualize a PennyLane circuit. "
            "Install with: pip install pennylane"
        ) from exc

    if output == "text":
        # ``qml.draw`` returns a wrapper that must be called to actually
        # produce the text — the wrapped qfunc takes the same arguments
        # as the circuit (here: none).
        return qml.draw(circuit)()

    # output == "mpl"
    # ``qml.draw_mpl`` returns ``(fig, ax)``; we want just the Figure.
    fig, _ = qml.draw_mpl(circuit)()
    return fig


def _render_qiskit(circuit: Any, output: str) -> Any:
    """Render a Qiskit ``QuantumCircuit`` as text or matplotlib Figure."""
    if output == "text":
        # ``draw('text')`` returns a ``TextDrawing`` whose ``__str__`` is the
        # plain-text diagram. We return a Python string for ergonomics.
        return str(circuit.draw("text"))

    # output == "mpl"
    try:
        return circuit.draw("mpl")
    except ImportError as exc:
        # Qiskit's matplotlib drawer requires pylatexenc for nice symbol
        # rendering. The upstream error references "pylatexenc" but is
        # not always obvious; re-raise with an installation hint.
        raise ImportError(
            "Qiskit's matplotlib drawer requires the 'pylatexenc' package. "
            "Install with: pip install pylatexenc"
        ) from exc


def _render_cirq(circuit: Any, output: str) -> Any:
    """Render a Cirq ``Circuit`` as text or matplotlib Figure.

    Cirq has no native matplotlib renderer; for ``output='mpl'`` we paint
    the text diagram into a monospaced Figure so the public API returns
    a uniform type across backends.
    """
    text = str(circuit)
    if output == "text":
        return text

    # output == "mpl"
    plt = _require_matplotlib()
    # Size the figure to the text dimensions. Approximate character cell
    # is 0.08" wide × 0.18" tall at the chosen font size; clamp to a
    # sensible range.
    lines = text.splitlines() or [""]
    width = max(1, max(len(line) for line in lines))
    height = max(1, len(lines))
    fig_w = max(4.0, min(20.0, width * 0.08 + 1.0))
    fig_h = max(2.0, min(20.0, height * 0.25 + 1.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.text(
        0.01,
        0.99,
        text,
        family="monospace",
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_circuit(
    encoding: BaseEncoding,
    x: ArrayLike,
    *,
    backend: BackendType = "pennylane",
    output: Literal["mpl", "text"] = "mpl",
    save_path: Union[str, None] = None,
) -> Union[Figure, str]:
    """Render the gate-level circuit produced by ``encoding`` on input ``x``.

    Delegates to the target backend's native drawing facility:

    - **pennylane**: :func:`pennylane.draw_mpl` / :func:`pennylane.draw`
    - **qiskit**: ``QuantumCircuit.draw('mpl' | 'text')``
    - **cirq**: native text diagram; for ``output='mpl'`` the text is
      painted into a monospaced matplotlib Figure to return a uniform
      type across backends.

    Parameters
    ----------
    encoding : BaseEncoding
        Any instantiated encoding. Used only to produce a single
        circuit via :meth:`BaseEncoding.get_circuit`.
    x : array-like
        Input sample to encode (shape ``(n_features,)`` or
        ``(1, n_features)``). Passed through the encoding's normal
        validation pipeline.
    backend : {'pennylane', 'qiskit', 'cirq'}, default='pennylane'
        Target framework for circuit construction *and* rendering.
    output : {'mpl', 'text'}, default='mpl'
        Output format. ``'mpl'`` returns a :class:`matplotlib.figure.Figure`;
        ``'text'`` returns a plain ``str`` containing the ASCII diagram.
    save_path : str or None, default=None
        When ``output='mpl'`` and a path is given, the figure is written
        to that path with ``bbox_inches='tight'`` before being returned.
        Ignored for ``output='text'``.

    Returns
    -------
    matplotlib.figure.Figure or str
        Figure for ``output='mpl'``; ASCII string for ``output='text'``.

    Raises
    ------
    ValueError
        If ``output`` is not one of ``'mpl'`` or ``'text'``, or if
        ``backend`` is not supported.
    ImportError
        If the requested backend or matplotlib (when needed) is not
        installed. The message includes the installation command.

    Examples
    --------
    PennyLane (default), saved to a file:

    >>> import numpy as np
    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.visualization import visualize_circuit
    >>> enc = IQPEncoding(n_features=4)
    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> fig = visualize_circuit(enc, x, save_path='iqp.png')  # doctest: +SKIP

    Qiskit, text output:

    >>> diagram = visualize_circuit(enc, x, backend='qiskit', output='text')

    See Also
    --------
    plot_entanglement_graph : Show qubit connectivity as a graph.
    """
    if output not in ("mpl", "text"):
        raise ValueError(f"output must be 'mpl' or 'text', got {output!r}")

    # Ensure matplotlib is available before generating a circuit if we
    # plan to render to a Figure. Failing fast avoids confusing the user
    # with backend-specific errors when the real problem is matplotlib.
    if output == "mpl":
        _require_matplotlib()

    circuit = encoding.get_circuit(x, backend=backend)

    if backend == "pennylane":
        result = _render_pennylane(circuit, output)
    elif backend == "qiskit":
        result = _render_qiskit(circuit, output)
    elif backend == "cirq":
        result = _render_cirq(circuit, output)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
        )

    if output == "mpl" and save_path is not None:
        # ``result`` is a Figure here; tight bbox keeps the saved image
        # crisp without extra whitespace.
        result.savefig(save_path, bbox_inches="tight")

    return result


def plot_entanglement_graph(
    encoding: BaseEncoding,
    *,
    layout: Literal["spring", "circular", "shell", "kamada_kawai"] = "circular",
    ax: Union[Axes, None] = None,
    figsize: Union[tuple[float, float], None] = None,
    node_color: str = "#4C72B0",
    edge_color: str = "#555555",
    node_size: int = 800,
    with_labels: bool = True,
    title: Union[str, None] = None,
    save_path: Union[str, None] = None,
) -> Figure:
    """Visualize the qubit connectivity (entanglement structure) of ``encoding``.

    Reads the entangling pairs via the
    :class:`~encoding_atlas.core.protocols.EntanglementQueryable` protocol
    when available; falls back to a node-only graph (no edges) for
    non-entangling encodings such as ``AngleEncoding`` or ``BasisEncoding``.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding whose qubit connectivity to plot.
    layout : {'spring', 'circular', 'shell', 'kamada_kawai'}, default='circular'
        networkx graph layout algorithm. ``'circular'`` is the most
        readable for typical entanglement patterns (linear / circular /
        full).
    ax : matplotlib.axes.Axes or None
        Plot into an existing Axes if given; otherwise a new Figure is
        created.
    figsize : (float, float) or None
        Figure size in inches. Ignored when ``ax`` is provided. Defaults
        to a sensible size based on ``n_qubits``.
    node_color, edge_color : str
        matplotlib-compatible colors.
    node_size : int
        Per-node area in points² (matplotlib convention).
    with_labels : bool, default=True
        Whether to draw the qubit index inside each node.
    title : str or None
        Optional figure title. If ``None``, a default is generated from
        the encoding class name.
    save_path : str or None
        When given, the figure is written to that path with
        ``bbox_inches='tight'`` before being returned.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure containing the entanglement graph (the same Figure
        whose ``ax`` was used, when one was provided).

    Raises
    ------
    ValueError
        If ``layout`` is not one of the supported algorithms.
    ImportError
        If matplotlib or networkx is not installed.

    Examples
    --------
    Plot full entanglement of IQP:

    >>> from encoding_atlas import IQPEncoding
    >>> from encoding_atlas.visualization import plot_entanglement_graph
    >>> fig = plot_entanglement_graph(IQPEncoding(n_features=4))

    Plot a non-entangling encoding (only nodes, no edges):

    >>> from encoding_atlas import AngleEncoding
    >>> fig = plot_entanglement_graph(AngleEncoding(n_features=4))

    Use within an existing matplotlib figure:

    >>> import matplotlib.pyplot as plt   # doctest: +SKIP
    >>> fig, ax = plt.subplots()          # doctest: +SKIP
    >>> plot_entanglement_graph(enc, ax=ax)  # doctest: +SKIP

    See Also
    --------
    visualize_circuit : Render the actual circuit diagram.
    """
    plt = _require_matplotlib()
    nx = _require_networkx()

    if layout not in ("spring", "circular", "shell", "kamada_kawai"):
        raise ValueError(
            f"layout must be one of 'spring', 'circular', 'shell', "
            f"'kamada_kawai', got {layout!r}"
        )

    # Collect the entangling pairs. Non-entangling encodings produce an
    # empty list — we still plot the qubit nodes so the user sees them.
    if is_entanglement_queryable(encoding):
        # Cast normalizes (qi, qj) so we don't draw both (0,1) and (1,0).
        pairs = [
            (int(min(a, b)), int(max(a, b)))
            for a, b in encoding.get_entanglement_pairs()
        ]
        # De-duplicate while preserving the natural ordering.
        pairs = list(dict.fromkeys(pairs))
    else:
        pairs = []

    n_qubits = int(encoding.n_qubits)

    graph: Any = nx.Graph()
    graph.add_nodes_from(range(n_qubits))
    graph.add_edges_from(pairs)

    # Build / acquire the matplotlib surface.
    if ax is None:
        if figsize is None:
            # Scale gently with qubit count; clamp to a sensible window.
            side = max(4.0, min(12.0, 1.0 + n_qubits * 0.5))
            figsize = (side, side)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute positions.
    if layout == "spring":
        # Deterministic positions across runs of the test suite.
        positions = nx.spring_layout(graph, seed=0)
    elif layout == "circular":
        positions = nx.circular_layout(graph)
    elif layout == "shell":
        positions = nx.shell_layout(graph)
    else:  # kamada_kawai
        # KK requires at least one edge; fall back to circular for empty graphs.
        positions = (
            nx.kamada_kawai_layout(graph)
            if graph.number_of_edges() > 0
            else nx.circular_layout(graph)
        )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.0,
    )
    if graph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edge_color=edge_color,
            width=1.5,
        )
    if with_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=ax,
            font_color="white",
            font_size=11,
            font_weight="bold",
        )

    if title is None:
        cls_name = type(encoding).__name__
        n_edges = graph.number_of_edges()
        title = (
            f"{cls_name} entanglement graph "
            f"({n_qubits} qubits, {n_edges} edge{'s' if n_edges != 1 else ''})"
        )
    ax.set_title(title)
    ax.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
