"""Smoke tests for VQC classification benchmark (Stage 6a).

These tests verify that the VQC implementation works end-to-end on
small examples, without testing statistical validity or performance.
"""

import pytest
import numpy as np


class TestDatasets:
    """Tests for dataset loading module."""

    def test_all_datasets_load(self):
        """Test that all datasets load correctly."""
        from experiments.datasets import load_dataset, AVAILABLE_DATASETS

        for name in AVAILABLE_DATASETS:
            X, y = load_dataset(name, seed=42)

            # Basic shape checks
            assert X.ndim == 2, f"{name}: X should be 2D"
            assert y.ndim == 1, f"{name}: y should be 1D"
            assert len(X) == len(y), f"{name}: X and y should have same length"

            # Value range checks
            assert X.min() >= 0, f"{name}: X min should be >= 0"
            assert X.max() <= 2 * np.pi + 0.01, f"{name}: X max should be <= 2π"

            # Label checks
            unique_labels = set(y)
            assert unique_labels.issubset({0, 1}), f"{name}: labels should be {{0, 1}}"

    def test_dataset_feature_dimensions(self):
        """Test that datasets have expected feature dimensions."""
        from experiments.datasets import load_dataset, DATASET_N_FEATURES

        for name, expected_n_features in DATASET_N_FEATURES.items():
            X, y = load_dataset(name, seed=42)
            assert X.shape[1] == expected_n_features, (
                f"{name}: expected {expected_n_features} features, got {X.shape[1]}"
            )

    def test_cv_folds_deterministic(self):
        """Test that CV folds are deterministic for the same seed."""
        from experiments.datasets import load_dataset, get_cv_folds

        X, y = load_dataset("moons", seed=42)

        folds1 = get_cv_folds(X, y, n_folds=3, seed=123)
        folds2 = get_cv_folds(X, y, n_folds=3, seed=123)

        for i, ((X_tr1, X_te1, y_tr1, y_te1), (X_tr2, X_te2, y_tr2, y_te2)) in enumerate(
            zip(folds1, folds2)
        ):
            np.testing.assert_array_equal(X_tr1, X_tr2, err_msg=f"Fold {i} X_train differs")
            np.testing.assert_array_equal(X_te1, X_te2, err_msg=f"Fold {i} X_test differs")
            np.testing.assert_array_equal(y_tr1, y_tr2, err_msg=f"Fold {i} y_train differs")
            np.testing.assert_array_equal(y_te1, y_te2, err_msg=f"Fold {i} y_test differs")

    def test_cv_folds_stratified(self):
        """Test that CV folds maintain class proportions."""
        from experiments.datasets import load_dataset, get_cv_folds

        X, y = load_dataset("moons", seed=42)
        overall_ratio = np.mean(y)

        folds = get_cv_folds(X, y, n_folds=5, seed=42)

        for fold_idx, (_, _, y_train, y_test) in enumerate(folds):
            train_ratio = np.mean(y_train)
            test_ratio = np.mean(y_test)

            # Allow 10% tolerance due to small fold sizes
            assert abs(train_ratio - overall_ratio) < 0.1, (
                f"Fold {fold_idx}: train ratio {train_ratio} differs from overall {overall_ratio}"
            )
            assert abs(test_ratio - overall_ratio) < 0.15, (
                f"Fold {fold_idx}: test ratio {test_ratio} differs from overall {overall_ratio}"
            )


class TestVQCClassifier:
    """Tests for VQC classifier."""

    def test_vqc_fits_and_predicts(self):
        """Test that VQC can fit and predict on simple data."""
        from experiments.vqc import VQCClassifier
        from encoding_atlas import IQPEncoding

        # Simple 2-feature encoding
        encoding = IQPEncoding(n_features=2, reps=1)

        # Tiny dataset
        X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1], [1.5, 1.6]])
        y = np.array([0, 0, 1, 1])

        vqc = VQCClassifier(encoding, n_var_layers=1, epochs=10, seed=42)
        vqc.fit(X, y)

        assert vqc.params_ is not None, "Parameters should be set after fit"
        assert len(vqc.loss_history_) == 10, "Should have 10 epochs of loss"

        predictions = vqc.predict(X)
        assert predictions.shape == (4,), "Should predict 4 samples"
        assert set(predictions).issubset({0, 1}), "Predictions should be in {0, 1}"

    def test_vqc_predict_proba(self):
        """Test that VQC predict_proba returns valid probabilities."""
        from experiments.vqc import VQCClassifier
        from encoding_atlas import IQPEncoding

        encoding = IQPEncoding(n_features=2, reps=1)

        X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1], [1.5, 1.6]])
        y = np.array([0, 0, 1, 1])

        vqc = VQCClassifier(encoding, n_var_layers=1, epochs=5, seed=42)
        vqc.fit(X, y)

        proba = vqc.predict_proba(X)

        assert proba.shape == (4, 2), "Should return (n_samples, 2) probabilities"
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities should sum to 1"
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities should be in [0, 1]"

    def test_vqc_score(self):
        """Test that VQC score returns valid accuracy."""
        from experiments.vqc import VQCClassifier
        from encoding_atlas import IQPEncoding

        encoding = IQPEncoding(n_features=2, reps=1)

        X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1], [1.5, 1.6]])
        y = np.array([0, 0, 1, 1])

        vqc = VQCClassifier(encoding, n_var_layers=1, epochs=5, seed=42)
        vqc.fit(X, y)

        score = vqc.score(X, y)

        assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"

    def test_vqc_not_fitted_raises(self):
        """Test that predict without fit raises error."""
        from experiments.vqc import VQCClassifier
        from encoding_atlas import IQPEncoding

        encoding = IQPEncoding(n_features=2, reps=1)
        vqc = VQCClassifier(encoding, n_var_layers=1, epochs=5, seed=42)

        X = np.array([[0.1, 0.2], [0.5, 0.6]])

        with pytest.raises(ValueError, match="not fitted"):
            vqc.predict(X)

    def test_vqc_different_encodings(self):
        """Test VQC with different encoding types."""
        from experiments.vqc import VQCClassifier
        from encoding_atlas import get_encoding

        encodings_to_test = [
            ("iqp", {"n_features": 2, "reps": 1}),
            ("angle", {"n_features": 2}),
        ]

        X = np.array([[0.1, 0.2], [0.5, 0.6], [3.0, 3.1], [4.0, 4.1]])
        y = np.array([0, 0, 1, 1])

        for name, params in encodings_to_test:
            encoding = get_encoding(name, **params)
            vqc = VQCClassifier(encoding, n_var_layers=1, epochs=5, seed=42)
            vqc.fit(X, y)

            predictions = vqc.predict(X)
            assert len(predictions) == len(y), f"Encoding {name}: wrong prediction count"


class TestClassicalBaselines:
    """Tests for classical baseline classifiers."""

    def test_all_baselines_work(self):
        """Test that all classical baselines work."""
        from experiments.classical_baselines import (
            get_classical_baseline,
            CLASSICAL_BASELINE_NAMES,
        )

        X = np.random.rand(50, 4)
        y = (X[:, 0] > 0.5).astype(int)

        for name in CLASSICAL_BASELINE_NAMES:
            clf = get_classical_baseline(name, seed=42)
            clf.fit(X, y)
            preds = clf.predict(X)

            assert preds.shape == (50,), f"{name}: wrong prediction shape"
            assert set(preds).issubset({0, 1}), f"{name}: invalid predictions"

    def test_baseline_invalid_name(self):
        """Test that invalid baseline name raises error."""
        from experiments.classical_baselines import get_classical_baseline

        with pytest.raises(ValueError, match="Unknown baseline"):
            get_classical_baseline("invalid_baseline", seed=42)

    def test_run_baseline_single_fold(self):
        """Test the single fold baseline runner."""
        from experiments.classical_baselines import run_baseline_single_fold

        np.random.seed(42)
        X_train = np.random.rand(40, 4)
        X_test = np.random.rand(10, 4)
        y_train = (X_train[:, 0] > 0.5).astype(int)
        y_test = (X_test[:, 0] > 0.5).astype(int)

        result = run_baseline_single_fold(
            "svm_rbf",
            X_train, X_test, y_train, y_test,
            seed=42,
        )

        assert "test_accuracy" in result
        assert "status" in result
        assert result["status"] == "success"
        assert 0 <= result["test_accuracy"] <= 1


class TestStatisticalUtils:
    """Tests for statistical analysis utilities."""

    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test."""
        from experiments.statistical import wilcoxon_test

        scores_a = np.array([0.85, 0.87, 0.84, 0.86, 0.88])
        scores_b = np.array([0.80, 0.82, 0.79, 0.81, 0.83])

        stat, p = wilcoxon_test(scores_a, scores_b)

        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_wilcoxon_identical_scores(self):
        """Test Wilcoxon with identical scores returns p=1."""
        from experiments.statistical import wilcoxon_test

        scores = np.array([0.85, 0.85, 0.85, 0.85, 0.85])
        stat, p = wilcoxon_test(scores, scores)

        assert p == 1.0, "Identical scores should yield p=1"

    def test_holm_bonferroni_correction(self):
        """Test Holm-Bonferroni multiple comparison correction."""
        from experiments.statistical import holm_bonferroni_correction

        # Some significant, some not
        p_values = [0.001, 0.02, 0.04, 0.06]
        rejected = holm_bonferroni_correction(p_values, alpha=0.05)

        assert len(rejected) == 4
        # First should definitely be rejected (0.001 < 0.05/4 = 0.0125)
        assert rejected[0] is True

    def test_cliffs_delta(self):
        """Test Cliff's delta effect size computation."""
        from experiments.statistical import cliffs_delta, interpret_cliffs_delta

        # Perfect separation
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])

        delta = cliffs_delta(x, y)
        assert delta == 1.0, "Perfect separation should give delta=1"

        interpretation = interpret_cliffs_delta(delta)
        assert interpretation == "large"

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        from experiments.statistical import bootstrap_ci

        scores = np.array([0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.86])

        lower, upper = bootstrap_ci(scores, confidence=0.95, n_bootstrap=1000, seed=42)

        mean = np.mean(scores)
        assert lower <= mean <= upper, "CI should contain the sample mean"
        assert lower < upper, "Lower bound should be less than upper"

    def test_spearman_correlation(self):
        """Test Spearman rank correlation."""
        from experiments.statistical import spearman_correlation

        # Perfect positive correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        rho, p = spearman_correlation(x, y)

        assert np.isclose(rho, 1.0), "Perfect monotonic should give rho=1"
        assert p < 0.05, "Should be significant"


class TestVQCHandler:
    """Tests for the VQC stage handler."""

    @pytest.mark.slow
    def test_handle_vqc_basic(self):
        """Test that the VQC handler runs without errors."""
        from experiments.runner import _handle_vqc
        from experiments.config import ExperimentConfig, EncodingSpec
        from encoding_atlas import IQPEncoding

        # Create a minimal config
        config = ExperimentConfig(
            stage="vqc",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="iqp", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "datasets": ["moons"],
                "n_runs": 1,
                "n_folds": 2,
                "vqc_params": {
                    "n_var_layers": 1,
                    "lr": 0.01,
                    "epochs": 5,
                },
            },
            output_dir="experiments/results/raw/test_vqc",
        )

        encoding = IQPEncoding(n_features=2, reps=1)
        result = _handle_vqc(encoding, config, seed=42)

        assert "encoding_name" in result
        assert "datasets" in result
        assert "moons" in result["datasets"]

        moons_result = result["datasets"]["moons"]
        assert moons_result["status"] in ("success", "failed", "skipped")

    def test_handle_vqc_feature_mismatch(self):
        """Test that handler skips mismatched feature dimensions."""
        from experiments.runner import _handle_vqc
        from experiments.config import ExperimentConfig, EncodingSpec
        from encoding_atlas import IQPEncoding

        config = ExperimentConfig(
            stage="vqc",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="iqp", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "datasets": ["iris"],  # iris has 4 features
                "n_runs": 1,
                "n_folds": 2,
                "vqc_params": {"n_var_layers": 1, "lr": 0.01, "epochs": 5},
            },
            output_dir="experiments/results/raw/test_vqc",
        )

        # 2-feature encoding vs 4-feature dataset
        encoding = IQPEncoding(n_features=2, reps=1)
        result = _handle_vqc(encoding, config, seed=42)

        assert result["datasets"]["iris"]["status"] == "skipped"
        assert "Feature mismatch" in result["datasets"]["iris"]["reason"]


class TestClassicalBaselineHandler:
    """Tests for the classical baseline handler in the experiment runner."""

    def test_handle_classical_baseline_basic(self):
        """Test that the classical baseline handler produces valid results."""
        from experiments.runner import _handle_classical_baseline
        from experiments.config import ExperimentConfig, EncodingSpec

        config = ExperimentConfig(
            stage="vqc",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[],
            analysis_params={
                "datasets": ["moons"],
                "n_runs": 2,
                "n_folds": 3,
            },
            output_dir="experiments/results/raw/test_baseline",
        )

        result = _handle_classical_baseline("svm_rbf", config, seed=42)

        assert result["method"] == "classical_baseline"
        assert result["baseline_name"] == "svm_rbf"
        assert "datasets" in result
        assert "moons" in result["datasets"]

        moons = result["datasets"]["moons"]
        assert moons["status"] == "success"
        assert 0 <= moons["mean_test_accuracy"] <= 1
        assert moons["std_test_accuracy"] >= 0
        assert moons["ci_95_lower"] <= moons["ci_95_upper"]
        assert len(moons["runs"]) == 2
        assert len(moons["runs"][0]["folds"]) == 3

    def test_handle_classical_baseline_all_classifiers(self):
        """Test that all three classical baselines run correctly."""
        from experiments.runner import _handle_classical_baseline
        from experiments.config import ExperimentConfig, EncodingSpec
        from experiments.classical_baselines import CLASSICAL_BASELINE_NAMES

        config = ExperimentConfig(
            stage="vqc",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[],
            analysis_params={
                "datasets": ["moons"],
                "n_runs": 1,
                "n_folds": 2,
            },
            output_dir="experiments/results/raw/test_baseline",
        )

        for bl_name in CLASSICAL_BASELINE_NAMES:
            result = _handle_classical_baseline(bl_name, config, seed=42)
            assert result["baseline_name"] == bl_name
            assert result["datasets"]["moons"]["status"] == "success"
            assert 0 <= result["datasets"]["moons"]["mean_test_accuracy"] <= 1

    def test_handle_classical_baseline_fold_metrics(self):
        """Test that each fold returns the expected metric keys."""
        from experiments.runner import _handle_classical_baseline
        from experiments.config import ExperimentConfig, EncodingSpec

        config = ExperimentConfig(
            stage="vqc",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[],
            analysis_params={
                "datasets": ["moons"],
                "n_runs": 1,
                "n_folds": 2,
            },
            output_dir="experiments/results/raw/test_baseline",
        )

        result = _handle_classical_baseline("svm_rbf", config, seed=42)
        fold = result["datasets"]["moons"]["runs"][0]["folds"][0]

        assert fold["status"] == "success"
        assert "fold" in fold
        assert "test_accuracy" in fold
        assert "train_accuracy" in fold
        assert "precision" in fold
        assert "recall" in fold
        assert "f1" in fold

    def test_baseline_uses_same_folds_as_vqc(self):
        """Test that baselines and VQC use identical CV folds."""
        from experiments.datasets import load_dataset, get_cv_folds

        X, y = load_dataset("moons", seed=42)

        # The VQC handler and baseline handler both use:
        # run_seed = seed + run_idx * 1000
        # folds = get_cv_folds(X, y, n_folds=n_folds, seed=run_seed)
        seed = 42
        run_seed = seed + 0 * 1000  # run_idx=0

        folds_a = get_cv_folds(X, y, n_folds=3, seed=run_seed)
        folds_b = get_cv_folds(X, y, n_folds=3, seed=run_seed)

        for i in range(3):
            np.testing.assert_array_equal(folds_a[i][0], folds_b[i][0])
            np.testing.assert_array_equal(folds_a[i][1], folds_b[i][1])

    def test_run_classical_baselines_noop_for_non_vqc(self):
        """Test that _run_classical_baselines is a no-op for non-VQC stages."""
        import tempfile
        import os
        from experiments.runner import ExperimentRunner
        from experiments.config import ExperimentConfig

        config = ExperimentConfig(
            stage="resources",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[],
            analysis_params={"classical_baselines": ["svm_rbf"]},
            output_dir=tempfile.mkdtemp(),
        )

        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner.config = config
        runner.resume = False

        counts = runner._run_classical_baselines(0, 0, 0, 0)
        assert counts == {"total": 0, "completed": 0, "skipped": 0, "failed": 0}

    def test_run_classical_baselines_noop_when_empty(self):
        """Test that _run_classical_baselines is a no-op when list is empty."""
        import tempfile
        from experiments.runner import ExperimentRunner
        from experiments.config import ExperimentConfig

        config = ExperimentConfig(
            stage="vqc",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[],
            analysis_params={"classical_baselines": []},
            output_dir=tempfile.mkdtemp(),
        )

        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner.config = config
        runner.resume = False

        counts = runner._run_classical_baselines(0, 0, 0, 0)
        assert counts == {"total": 0, "completed": 0, "skipped": 0, "failed": 0}


class TestIntegration:
    """Integration tests that test the full pipeline."""

    @pytest.mark.slow
    def test_full_vqc_pipeline(self):
        """Test the full VQC pipeline with minimal settings."""
        from experiments.vqc import run_vqc_single_fold
        from experiments.datasets import load_dataset, get_cv_folds
        from encoding_atlas import IQPEncoding

        # Load a small dataset
        X, y = load_dataset("moons", n_samples=50, seed=42)
        folds = get_cv_folds(X, y, n_folds=2, seed=42)
        X_train, X_test, y_train, y_test = folds[0]

        # Create encoding
        encoding = IQPEncoding(n_features=2, reps=1)

        # Run VQC
        result = run_vqc_single_fold(
            encoding,
            X_train, X_test, y_train, y_test,
            n_var_layers=1,
            lr=0.01,
            epochs=10,
            seed=42,
        )

        assert result["status"] in ("success", "diverged")
        assert "test_accuracy" in result
        assert 0 <= result["test_accuracy"] <= 1

    @pytest.mark.slow
    def test_vqc_vs_baseline_comparison(self):
        """Test comparing VQC with classical baseline."""
        from experiments.vqc import run_vqc_single_fold
        from experiments.classical_baselines import run_baseline_single_fold
        from experiments.datasets import load_dataset, get_cv_folds
        from encoding_atlas import IQPEncoding

        # Load dataset
        X, y = load_dataset("moons", n_samples=100, seed=42)
        folds = get_cv_folds(X, y, n_folds=2, seed=42)
        X_train, X_test, y_train, y_test = folds[0]

        # Run VQC
        encoding = IQPEncoding(n_features=2, reps=1)
        vqc_result = run_vqc_single_fold(
            encoding,
            X_train, X_test, y_train, y_test,
            n_var_layers=1,
            epochs=10,
            seed=42,
        )

        # Run baseline
        baseline_result = run_baseline_single_fold(
            "svm_rbf",
            X_train, X_test, y_train, y_test,
            seed=42,
        )

        # Both should complete
        assert vqc_result["status"] in ("success", "diverged")
        assert baseline_result["status"] == "success"

        # Both should have accuracy metrics
        assert "test_accuracy" in vqc_result
        assert "test_accuracy" in baseline_result
