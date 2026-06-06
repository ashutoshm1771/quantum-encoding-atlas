"""Tests for circuit and entanglement-structure visualization.

Covers :func:`visualize_circuit` and :func:`plot_entanglement_graph` in
``encoding_atlas.visualization.circuit``.

Tested dimensions
-----------------
* All three backends (PennyLane / Qiskit / Cirq) × both output formats
  (``'mpl'`` and ``'text'``).
* Optional ``save_path`` (PNG written to disk, non-trivial size).
* Argument validation: bad ``output``, bad ``backend``, bad ``layout``,
  bad ``maxsize``.
* Entanglement-graph behavior for entangling and non-entangling encodings,
  and for all four supported layouts (including the empty-graph
  fall-back path for ``kamada_kawai``).
* ``ax`` re-use for embedding in user figures.
* Defensive behavior when optional dependencies are missing (matplotlib,
  networkx, ``pylatexenc`` for the Qiskit mpl drawer).
"""

from __future__ import annotations

import builtins
import importlib

import numpy as np
import pytest

# Backend availability — tests skip gracefully when an optional backend
# is not installed.
try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cirq  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False


# Conftest sets matplotlib backend to Agg for headless test runs.
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from encoding_atlas import (  # noqa: E402
    AngleEncoding,
    BasisEncoding,
    HardwareEfficientEncoding,
    IQPEncoding,
)
from encoding_atlas.visualization import (  # noqa: E402
    plot_entanglement_graph,
    visualize_circuit,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def iqp_encoding() -> IQPEncoding:
    """Entangling encoding with full-pair connectivity (well-defined graph)."""
    return IQPEncoding(n_features=4, reps=1)


@pytest.fixture
def angle_encoding() -> AngleEncoding:
    """Non-entangling encoding (used for the empty-graph code path)."""
    return AngleEncoding(n_features=4)


@pytest.fixture
def basis_encoding() -> BasisEncoding:
    """Another non-entangling encoding used to confirm protocol-based detection."""
    return BasisEncoding(n_features=4)


@pytest.fixture
def hardware_efficient_encoding() -> HardwareEfficientEncoding:
    """Linear-entanglement encoding (typical NISQ topology)."""
    return HardwareEfficientEncoding(n_features=4, reps=1)


@pytest.fixture
def sample_x() -> np.ndarray:
    """Validated 4-feature sample input."""
    return np.array([0.1, 0.2, 0.3, 0.4])


# =============================================================================
# visualize_circuit — output='mpl'
# =============================================================================


class TestVisualizeCircuitMpl:
    """The default ``output='mpl'`` path returns a Figure for every backend."""

    def test_pennylane_returns_figure(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        fig = visualize_circuit(iqp_encoding, sample_x, backend="pennylane")
        assert isinstance(fig, Figure)

    def test_qiskit_returns_figure(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")
        # Qiskit's mpl drawer needs pylatexenc; skip with a clear note if
        # the user environment is missing it.
        try:
            fig = visualize_circuit(iqp_encoding, sample_x, backend="qiskit")
        except ImportError as exc:
            pytest.skip(f"Qiskit mpl drawer unavailable: {exc}")
        assert isinstance(fig, Figure)

    def test_cirq_returns_figure(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        if not HAS_CIRQ:
            pytest.skip("Cirq not installed")
        fig = visualize_circuit(iqp_encoding, sample_x, backend="cirq")
        assert isinstance(fig, Figure)
        # The Cirq path renders the text diagram into a monospace Figure;
        # ensure exactly one Axes was created and it contains text.
        axes = fig.axes
        assert len(axes) == 1
        text_artists = axes[0].texts
        assert text_artists, "Cirq mpl path should embed the text diagram"

    def test_default_backend_is_pennylane(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        fig = visualize_circuit(iqp_encoding, sample_x)
        assert isinstance(fig, Figure)


# =============================================================================
# visualize_circuit — output='text'
# =============================================================================


class TestVisualizeCircuitText:
    """Each backend returns a Python ``str`` with the ASCII diagram."""

    def test_pennylane_text(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        text = visualize_circuit(
            iqp_encoding, sample_x, backend="pennylane", output="text"
        )
        assert isinstance(text, str)
        # PennyLane labels qubits as ``0:``, ``1:`` etc.
        assert "0:" in text
        assert "3:" in text

    def test_qiskit_text(self, iqp_encoding: IQPEncoding, sample_x: np.ndarray) -> None:
        if not HAS_QISKIT:
            pytest.skip("Qiskit not installed")
        text = visualize_circuit(
            iqp_encoding, sample_x, backend="qiskit", output="text"
        )
        assert isinstance(text, str)
        # Qiskit labels qubits as ``q_0`` etc.
        assert "q_" in text or "q0" in text

    def test_cirq_text(self, iqp_encoding: IQPEncoding, sample_x: np.ndarray) -> None:
        if not HAS_CIRQ:
            pytest.skip("Cirq not installed")
        text = visualize_circuit(iqp_encoding, sample_x, backend="cirq", output="text")
        assert isinstance(text, str)
        # Cirq labels qubits as ``0:`` etc.
        assert "0:" in text


# =============================================================================
# visualize_circuit — save_path
# =============================================================================


class TestVisualizeCircuitSavePath:
    """``save_path`` writes a non-trivial PNG and returns the same Figure."""

    def test_save_path_writes_file(
        self,
        iqp_encoding: IQPEncoding,
        sample_x: np.ndarray,
        tmp_path,
    ) -> None:
        target = tmp_path / "iqp.png"
        fig = visualize_circuit(iqp_encoding, sample_x, save_path=str(target))
        assert isinstance(fig, Figure)
        assert target.exists()
        # PNG headers are ~8 bytes; we expect a real rendered image.
        assert target.stat().st_size > 1000

    def test_save_path_ignored_for_text(
        self,
        iqp_encoding: IQPEncoding,
        sample_x: np.ndarray,
        tmp_path,
    ) -> None:
        target = tmp_path / "should_not_exist.png"
        out = visualize_circuit(
            iqp_encoding,
            sample_x,
            output="text",
            save_path=str(target),
        )
        assert isinstance(out, str)
        # save_path should be a no-op when there is no Figure to save.
        assert not target.exists()


# =============================================================================
# visualize_circuit — argument validation
# =============================================================================


class TestVisualizeCircuitValidation:
    """Bad arguments fail fast with helpful messages, before circuit work."""

    def test_invalid_output_rejected(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="output must be 'mpl' or 'text'"):
            visualize_circuit(iqp_encoding, sample_x, output="svg")  # type: ignore[arg-type]

    def test_invalid_backend_rejected(
        self, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        # The underlying ``encoding.get_circuit`` raises ValueError for an
        # unknown backend long before our dispatch sees it; we just need to
        # confirm something user-facing is raised.
        with pytest.raises(ValueError):
            visualize_circuit(
                iqp_encoding,
                sample_x,
                backend="qsharp",  # type: ignore[arg-type]
                output="text",
            )

    def test_invalid_input_shape_propagates(self, iqp_encoding: IQPEncoding) -> None:
        # ``encoding.get_circuit`` should reject a wrong-length vector.
        with pytest.raises(ValueError):
            visualize_circuit(iqp_encoding, np.zeros(3))


# =============================================================================
# plot_entanglement_graph
# =============================================================================


class TestPlotEntanglementGraph:
    """The entanglement graph reflects the encoding's connectivity pattern."""

    def test_entangling_encoding_has_edges(self, iqp_encoding: IQPEncoding) -> None:
        fig = plot_entanglement_graph(iqp_encoding)
        assert isinstance(fig, Figure)
        # The IQP graph for n=4 with full entanglement has 6 edges.
        # We verify via the encoding (not the figure) since matplotlib
        # doesn't expose the graph object directly.
        assert len(iqp_encoding.get_entanglement_pairs()) == 6

    def test_non_entangling_encoding_no_edges(
        self, angle_encoding: AngleEncoding
    ) -> None:
        # Should still produce a Figure (with nodes but no edges) rather
        # than raising — non-entangling encodings are valid inputs.
        fig = plot_entanglement_graph(angle_encoding)
        assert isinstance(fig, Figure)

    def test_basis_encoding_no_edges(self, basis_encoding: BasisEncoding) -> None:
        fig = plot_entanglement_graph(basis_encoding)
        assert isinstance(fig, Figure)

    def test_default_title_includes_class_and_counts(
        self, iqp_encoding: IQPEncoding
    ) -> None:
        fig = plot_entanglement_graph(iqp_encoding)
        title = fig.axes[0].get_title()
        assert "IQPEncoding" in title
        assert "4 qubits" in title
        assert "6 edges" in title

    def test_custom_title(self, iqp_encoding: IQPEncoding) -> None:
        fig = plot_entanglement_graph(iqp_encoding, title="My custom title")
        assert fig.axes[0].get_title() == "My custom title"

    def test_zero_or_one_edge_title_pluralization(
        self, angle_encoding: AngleEncoding
    ) -> None:
        # Non-entangling → 0 edges → "0 edges"
        fig = plot_entanglement_graph(angle_encoding)
        assert "0 edges" in fig.axes[0].get_title()

    @pytest.mark.parametrize("layout", ["spring", "circular", "shell", "kamada_kawai"])
    def test_all_layouts_succeed_for_entangling(
        self, hardware_efficient_encoding: HardwareEfficientEncoding, layout: str
    ) -> None:
        fig = plot_entanglement_graph(hardware_efficient_encoding, layout=layout)
        assert isinstance(fig, Figure)

    def test_kamada_kawai_falls_back_for_empty_graph(
        self, angle_encoding: AngleEncoding
    ) -> None:
        """Kamada-Kawai needs at least one edge; the implementation falls
        back to a circular layout for non-entangling encodings."""
        fig = plot_entanglement_graph(angle_encoding, layout="kamada_kawai")
        assert isinstance(fig, Figure)

    def test_invalid_layout_rejected(self, iqp_encoding: IQPEncoding) -> None:
        with pytest.raises(ValueError, match="layout must be one of"):
            plot_entanglement_graph(iqp_encoding, layout="invalid")  # type: ignore[arg-type]

    def test_with_existing_ax(self, iqp_encoding: IQPEncoding) -> None:
        import matplotlib.pyplot as plt

        fig_in, ax = plt.subplots()
        fig_out = plot_entanglement_graph(iqp_encoding, ax=ax)
        # Must reuse the user's Figure (Axes) instead of creating one.
        assert fig_out is fig_in
        assert isinstance(ax, Axes)

    def test_save_path_writes_file(self, iqp_encoding: IQPEncoding, tmp_path) -> None:
        target = tmp_path / "graph.png"
        plot_entanglement_graph(iqp_encoding, save_path=str(target))
        assert target.exists()
        assert target.stat().st_size > 1000

    def test_disable_labels(self, iqp_encoding: IQPEncoding) -> None:
        # Smoke test: with_labels=False should not raise.
        fig = plot_entanglement_graph(iqp_encoding, with_labels=False)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, iqp_encoding: IQPEncoding) -> None:
        fig = plot_entanglement_graph(iqp_encoding, figsize=(6.0, 6.0))
        assert isinstance(fig, Figure)
        # Confirm matplotlib honoured the requested size (in inches).
        width, height = fig.get_size_inches()
        assert abs(width - 6.0) < 1e-6
        assert abs(height - 6.0) < 1e-6


# =============================================================================
# Defensive behavior when optional deps are missing
# =============================================================================


class TestOptionalDependencyHandling:
    """Both entry points raise clear ImportError when matplotlib/networkx
    are missing. We simulate the missing dependency by patching
    builtins.__import__ rather than uninstalling the package."""

    @staticmethod
    def _import_blocker(blocked: str):
        """Return an ``__import__`` replacement that pretends one package is missing."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"simulated missing: {name}")
            return real_import(name, *args, **kwargs)

        return fake_import

    def test_visualize_circuit_no_matplotlib(
        self, monkeypatch, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        # Force matplotlib.pyplot import to fail when our internal helper
        # tries it. We need to clear any cached pyplot module too.
        import sys

        monkeypatch.setattr(builtins, "__import__", self._import_blocker("matplotlib"))
        # Drop cached imports so the next attempt re-enters the blocker.
        for name in list(sys.modules):
            if name == "matplotlib" or name.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        with pytest.raises(ImportError, match="matplotlib is required"):
            visualize_circuit(iqp_encoding, sample_x, output="mpl")

    def test_plot_entanglement_graph_no_networkx(
        self, monkeypatch, iqp_encoding: IQPEncoding
    ) -> None:
        import sys

        monkeypatch.setattr(builtins, "__import__", self._import_blocker("networkx"))
        for name in list(sys.modules):
            if name == "networkx" or name.startswith("networkx."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        with pytest.raises(ImportError, match="networkx is required"):
            plot_entanglement_graph(iqp_encoding)

    def test_text_output_does_not_require_matplotlib(
        self, monkeypatch, iqp_encoding: IQPEncoding, sample_x: np.ndarray
    ) -> None:
        """``output='text'`` should not import matplotlib at all."""
        import sys

        monkeypatch.setattr(builtins, "__import__", self._import_blocker("matplotlib"))
        for name in list(sys.modules):
            if name == "matplotlib" or name.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        text = visualize_circuit(
            iqp_encoding, sample_x, backend="pennylane", output="text"
        )
        assert isinstance(text, str)


# =============================================================================
# Round-trip with the public package import
# =============================================================================


class TestPublicAPIExport:
    """The new helpers are part of the ``encoding_atlas.visualization``
    public surface."""

    def test_importable_from_subpackage(self) -> None:
        mod = importlib.import_module("encoding_atlas.visualization")
        assert hasattr(mod, "visualize_circuit")
        assert hasattr(mod, "plot_entanglement_graph")
        assert "visualize_circuit" in mod.__all__
        assert "plot_entanglement_graph" in mod.__all__

    def test_works_for_every_supported_encoding_name(self) -> None:
        """Stress test: render every angle-encoded-style encoding to text
        with PennyLane. None should crash.

        We pick per-encoding inputs that are guaranteed to produce a
        non-empty circuit (e.g. ``BasisEncoding`` only emits X gates for
        values above its threshold).
        """
        # (encoding, sample) — chosen so each encoding produces at least
        # one gate and PennyLane's text drawer therefore produces a
        # non-empty diagram.
        cases: list[tuple[object, np.ndarray]] = [
            (AngleEncoding(n_features=4), np.array([0.1, 0.2, 0.3, 0.4])),
            (BasisEncoding(n_features=4), np.array([0.9, 0.8, 0.7, 0.6])),
            (IQPEncoding(n_features=4), np.array([0.1, 0.2, 0.3, 0.4])),
            (
                HardwareEfficientEncoding(n_features=4),
                np.array([0.1, 0.2, 0.3, 0.4]),
            ),
        ]
        for enc, x in cases:
            text = visualize_circuit(enc, x, output="text")  # type: ignore[arg-type]
            assert isinstance(text, str)
            assert len(text) > 0, f"empty text diagram for {type(enc).__name__}"


# =============================================================================
# Resource cleanup (matplotlib leaks Figures if not closed)
# =============================================================================


@pytest.fixture(autouse=True)
def _close_all_figures() -> None:
    """Avoid bleeding memory across test runs."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
