"""Tests for multi-class classification support in the benchmark.

Confirms that metrics, the VQC (one-vs-rest), the quantum kernel, baselines,
datasets, and the benchmark orchestrator all handle more than two classes, and
that the binary path is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from encoding_atlas import AngleEncoding
from encoding_atlas.benchmark import (
    EncodingBenchmark,
    QuantumKernelClassifier,
    VQCClassifier,
    compute_metrics,
    get_dataset,
    list_datasets,
    list_multiclass_datasets,
)
from encoding_atlas.benchmark.baselines import run_baseline_single_fold
from encoding_atlas.benchmark.kernel import run_kernel_single_fold
from encoding_atlas.benchmark.vqc import run_vqc_single_fold


def _three_class(n_per: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.25, (n_per, 2)) for c in (0.4, 1.6, 2.8)])
    y = np.array([0] * n_per + [1] * n_per + [2] * n_per, dtype=np.intp)
    return X, y


def _stratified_split(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=0.34, stratify=y, random_state=0)


# =====================================================================
# Metrics
# =====================================================================


class TestMetrics:
    def test_binary_unchanged(self) -> None:
        # Binary path must remain positive-class precision/recall/F1.
        m = compute_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        assert m == {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }

    def test_multiclass_macro(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        pred = np.array([0, 1, 2, 0, 2, 1])  # 4 / 6 correct
        m = compute_metrics(y, pred)
        assert m["accuracy"] == pytest.approx(4 / 6)
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= m[key] <= 1.0

    def test_perfect_multiclass(self) -> None:
        y = np.array([0, 1, 2, 2, 1, 0])
        m = compute_metrics(y, y)
        assert all(v == 1.0 for v in m.values())


# =====================================================================
# VQC one-vs-rest
# =====================================================================


class TestMultiClassVQC:
    def test_ovr_structure(self) -> None:
        X, y = _three_class()
        vqc = VQCClassifier(
            AngleEncoding(n_features=2), n_var_layers=1, epochs=6, seed=0
        )
        vqc.fit(X, y)
        assert list(vqc.classes_) == [0, 1, 2]
        assert vqc._ovr_models is not None
        assert len(vqc._ovr_models) == 3
        proba = vqc.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        preds = vqc.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 2})
        assert 0.0 <= vqc.score(X, y) <= 1.0

    def test_binary_path_unchanged(self) -> None:
        # Two-class fit must keep the single-VQC behavior (no OvR ensemble).
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0.5, 0.2, (8, 2)), rng.normal(2.5, 0.2, (8, 2))])
        y = np.array([0] * 8 + [1] * 8, dtype=np.intp)
        vqc = VQCClassifier(AngleEncoding(n_features=2), epochs=4, seed=0).fit(X, y)
        assert vqc._ovr_models is None
        assert vqc.params_ is not None
        assert vqc.predict_proba(X).shape == (16, 2)

    @pytest.mark.slow
    def test_learns_separable_three_class(self) -> None:
        X, y = _three_class(n_per=12)
        Xtr, Xte, ytr, yte = _stratified_split(X, y)
        vqc = VQCClassifier(
            AngleEncoding(n_features=2), n_var_layers=2, epochs=25, lr=0.1, seed=0
        ).fit(Xtr, ytr)
        assert vqc.score(Xte, yte) >= 0.7


# =====================================================================
# Kernel and baselines
# =====================================================================


class TestMultiClassKernelAndBaselines:
    def test_kernel_classifier_multiclass(self) -> None:
        X, y = _three_class()
        clf = QuantumKernelClassifier(AngleEncoding(n_features=2), seed=0).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_kernel_fold_kta_is_none_for_multiclass(self) -> None:
        X, y = _three_class()
        Xtr, Xte, ytr, yte = _stratified_split(X, y)
        result = run_kernel_single_fold(AngleEncoding(n_features=2), Xtr, Xte, ytr, yte)
        assert result["status"] == "success"
        assert result["kernel_target_alignment"] is None  # undefined for >2 classes
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_vqc_fold_multiclass(self) -> None:
        X, y = _three_class()
        Xtr, Xte, ytr, yte = _stratified_split(X, y)
        result = run_vqc_single_fold(
            AngleEncoding(n_features=2), Xtr, Xte, ytr, yte, epochs=6, seed=0
        )
        assert result["status"] in {"success", "diverged"}
        assert 0.0 <= result["f1"] <= 1.0

    def test_baseline_fold_multiclass(self) -> None:
        X, y = _three_class()
        Xtr, Xte, ytr, yte = _stratified_split(X, y)
        result = run_baseline_single_fold("svm_rbf", Xtr, Xte, ytr, yte, seed=0)
        assert result["status"] == "success"
        assert result["test_accuracy"] >= 0.6  # well-separated clusters


# =====================================================================
# Datasets
# =====================================================================


class TestMultiClassDatasets:
    def test_iris3_has_three_classes(self) -> None:
        X, y = get_dataset("iris3")
        assert len(np.unique(y)) == 3
        assert X.shape[1] == 2

    def test_blobs3_has_three_classes(self) -> None:
        X, y = get_dataset("blobs3", n_samples=60, seed=0)
        assert len(np.unique(y)) == 3
        assert X.shape == (60, 2)

    def test_list_multiclass_datasets(self) -> None:
        assert set(list_multiclass_datasets()) == {"iris3", "blobs3"}

    def test_binary_datasets_still_binary(self) -> None:
        # Existing binary datasets must be unchanged.
        _, y_iris = get_dataset("iris")
        assert len(np.unique(y_iris)) == 2
        assert {"iris3", "blobs3"} <= set(list_datasets())


# =====================================================================
# Benchmark orchestration
# =====================================================================


class TestMultiClassBenchmark:
    def test_benchmark_runs_multiclass(self) -> None:
        bench = EncodingBenchmark(
            [AngleEncoding(n_features=2)],
            ["blobs3"],
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=45,
        )
        cell = bench.run()["results"]["kernel"]["AngleEncoding"]["blobs3"]
        assert cell["status"] == "success"
        assert 0.0 <= cell["mean"] <= 1.0
        assert cell["n_scores"] == 3
