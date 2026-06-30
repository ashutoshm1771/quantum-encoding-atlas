"""Tests for benchmark metrics and the standalone accuracy plot helper."""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas.benchmark.metrics import compute_metrics
from encoding_atlas.benchmark.visualization import plot_accuracy_comparison


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([0, 1, 0, 1])
        m = compute_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_known_confusion(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])  # 1 TP, 1 FP, 1 FN, 1 TN
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.5
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.5)

    def test_all_keys_present(self) -> None:
        y = np.array([0, 1])
        assert set(compute_metrics(y, y)) == {"accuracy", "precision", "recall", "f1"}


class TestPlotAccuracyComparison:
    def test_returns_figure(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_accuracy_comparison({"angle": [0.9, 0.92], "iqp": [0.6, 0.62]})
        assert fig is not None
        plt.close(fig)
