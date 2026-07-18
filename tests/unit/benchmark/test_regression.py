"""Tests for regression support in the benchmarking framework.

Covers regression metrics, datasets, the VQC/quantum-kernel regressors, the
regression baselines and fold runners, and the ``task="regression"`` paths of
``evaluate_encoding``/``EncodingBenchmark`` — including the correctness details
that differ from classification: K-fold (not stratified) splits, unbounded R^2
confidence intervals, and continuous targets that must not be truncated.
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import AngleEncoding
from encoding_atlas.benchmark import (
    REGRESSION_BASELINE_NAMES,
    EncodingBenchmark,
    QuantumKernelRegressor,
    VQCRegressor,
    compute_regression_metrics,
    evaluate_encoding,
    get_regression_baseline,
    get_regression_dataset,
    list_regression_datasets,
    run_kernel_regression_fold,
    run_regression_baseline_single_fold,
    run_vqc_regression_fold,
)
from encoding_atlas.benchmark.runner import _scale_features


def _angle() -> AngleEncoding:
    return AngleEncoding(n_features=2, rotation="Y")


@pytest.fixture(scope="module")
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    """Scaled features and continuous targets from the sine regression set."""
    X, y = get_regression_dataset("sine_reg", n_samples=60, seed=0)
    return _scale_features(X, 0.0, 2 * np.pi), y


# =====================================================================
# Metrics
# =====================================================================


class TestRegressionMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mse"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)

    def test_mean_predictor_gives_zero_r2(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_regression_metrics(y, np.full(4, y.mean()))
        assert m["r2"] == pytest.approx(0.0)

    def test_r2_can_be_negative(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_regression_metrics(y, np.array([10.0, -5.0, 8.0, 0.0]))["r2"] < 0

    def test_known_errors(self) -> None:
        m = compute_regression_metrics(np.array([0.0, 0.0]), np.array([1.0, 3.0]))
        assert m["mse"] == pytest.approx(5.0)  # (1 + 9) / 2
        assert m["rmse"] == pytest.approx(np.sqrt(5.0))
        assert m["mae"] == pytest.approx(2.0)

    def test_r2_undefined_for_single_sample(self) -> None:
        m = compute_regression_metrics(np.array([1.0]), np.array([1.0]))
        assert np.isnan(m["r2"])


# =====================================================================
# Datasets
# =====================================================================


class TestRegressionDatasets:
    def test_listed(self) -> None:
        assert list_regression_datasets() == ["linear_reg", "sine_reg"]

    @pytest.mark.parametrize("name", ["linear_reg", "sine_reg"])
    def test_shape_and_continuity(self, name: str) -> None:
        X, y = get_regression_dataset(name, n_samples=50, seed=0)
        assert X.shape == (50, 2)
        assert y.shape == (50,)
        assert y.dtype == np.float64
        # Targets are continuous, not a handful of class labels.
        assert len(np.unique(y)) > 10

    def test_deterministic_for_seed(self) -> None:
        a = get_regression_dataset("sine_reg", n_samples=20, seed=3)[1]
        b = get_regression_dataset("sine_reg", n_samples=20, seed=3)[1]
        assert np.array_equal(a, b)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown regression dataset"):
            get_regression_dataset("not_a_dataset")


# =====================================================================
# Estimators
# =====================================================================


class TestQuantumKernelRegressor:
    def test_learns_smooth_target(
        self, reg_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = reg_data
        model = QuantumKernelRegressor(_angle(), alpha=0.1).fit(X[:45], y[:45])
        preds = model.predict(X[45:])
        assert preds.shape == (15,)
        assert model.score(X[45:], y[45:]) > 0.5  # smooth, learnable target

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            QuantumKernelRegressor(_angle()).predict(np.array([[0.1, 0.2]]))

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be positive"):
            QuantumKernelRegressor(_angle(), alpha=0.0)


class TestVQCRegressor:
    def test_fit_predict_continuous(
        self, reg_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = reg_data
        model = VQCRegressor(_angle(), epochs=4, lr=0.1, seed=0).fit(X[:30], y[:30])
        preds = model.predict(X[30:40])
        assert preds.shape == (10,)
        assert preds.dtype.kind == "f"
        # Predictions are continuous, not collapsed onto integers.
        assert len(np.unique(preds)) > 1
        assert model.status_ in {"success", "diverged"}

    def test_predictions_respect_target_scale(
        self, reg_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Targets are min-max mapped onto <Z> in [-1, 1] using training
        statistics, so predictions land within the training target range."""
        X, y = reg_data
        y_train = y[:30]
        model = VQCRegressor(_angle(), epochs=4, lr=0.1, seed=0).fit(X[:30], y_train)
        preds = model.predict(X[30:])
        assert preds.min() >= y_train.min() - 1e-9
        assert preds.max() <= y_train.max() + 1e-9

    def test_constant_target_is_safe(self) -> None:
        X = np.linspace(0.0, 2 * np.pi, 8).reshape(4, 2)
        model = VQCRegressor(_angle(), epochs=2, seed=0).fit(X, np.full(4, 3.0))
        assert np.all(np.isfinite(model.predict(X)))

    def test_determinism_same_seed(
        self, reg_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = reg_data
        a = VQCRegressor(_angle(), epochs=3, seed=5).fit(X[:20], y[:20]).predict(X[:5])
        b = VQCRegressor(_angle(), epochs=3, seed=5).fit(X[:20], y[:20]).predict(X[:5])
        assert np.allclose(a, b)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            VQCRegressor(_angle()).predict(np.array([[0.1, 0.2]]))

    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            ({"n_var_layers": 0}, "n_var_layers"),
            ({"lr": 0}, "lr"),
            ({"epochs": 0}, "epochs"),
        ],
    )
    def test_invalid_args(self, kwargs: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            VQCRegressor(_angle(), **kwargs)


# =====================================================================
# Baselines and fold runners
# =====================================================================


class TestRegressionBaselines:
    @pytest.mark.parametrize("name", REGRESSION_BASELINE_NAMES)
    def test_baseline_runs(
        self, name: str, reg_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = reg_data
        res = run_regression_baseline_single_fold(
            name, X[:45], X[45:], y[:45], y[45:], seed=0
        )
        assert res["status"] == "success"
        assert np.isfinite(res["test_r2"])

    def test_unknown_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown regression baseline"):
            get_regression_baseline("not_a_model", seed=0)


class TestRegressionFoldRunners:
    def test_kernel_fold(self, reg_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = reg_data
        res = run_kernel_regression_fold(_angle(), X[:45], X[45:], y[:45], y[45:])
        assert res["status"] == "success"
        for key in ("test_r2", "mse", "rmse", "mae"):
            assert np.isfinite(res[key])

    def test_vqc_fold(self, reg_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = reg_data
        res = run_vqc_regression_fold(
            _angle(), X[:30], X[30:40], y[:30], y[30:40], epochs=3, seed=0
        )
        assert res["status"] in {"success", "diverged"}
        assert np.isfinite(res["mse"])

    def test_failed_fold_reports_nan_not_zero(self) -> None:
        """A failed fold must not report R^2 = 0.0, which would falsely imply
        mean-level performance."""
        res = run_kernel_regression_fold(
            _angle(), np.zeros((4, 5)), np.zeros((2, 5)), np.zeros(4), np.zeros(2)
        )
        assert res["status"] == "failed"
        assert np.isnan(res["test_r2"])


# =====================================================================
# evaluate_encoding / EncodingBenchmark regression paths
# =====================================================================


class TestEvaluateEncodingRegression:
    def test_kernel_regression(self) -> None:
        X, y = get_regression_dataset("sine_reg", n_samples=60, seed=0)
        out = evaluate_encoding(
            _angle(),
            X,
            y,
            task="regression",
            method="kernel",
            n_runs=1,
            n_folds=3,
            seed=0,
        )
        assert out["task"] == "regression"
        assert out["score_metric"] == "r2"
        assert out["n_scores"] == 3
        assert out["mean"] > 0.5  # learnable target

    def test_vqc_regression(self) -> None:
        X, y = get_regression_dataset("linear_reg", n_samples=30, seed=0)
        out = evaluate_encoding(
            _angle(),
            X,
            y,
            task="regression",
            method="vqc",
            n_runs=1,
            n_folds=2,
            seed=0,
            vqc_epochs=3,
        )
        assert out["task"] == "regression"
        assert out["n_scores"] >= 1

    def test_invalid_task_raises(self) -> None:
        X, y = get_regression_dataset("linear_reg", n_samples=20, seed=0)
        with pytest.raises(ValueError, match="task must be one of"):
            evaluate_encoding(_angle(), X, y, task="clustering")

    def test_continuous_targets_not_truncated(self) -> None:
        """Regression targets must stay float; casting to int would collapse
        values such as 0.4/0.6 onto 0."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (24, 2))
        y = rng.uniform(0.2, 0.8, 24)  # all values inside (0, 1)
        out = evaluate_encoding(
            _angle(),
            X,
            y,
            task="regression",
            method="kernel",
            n_runs=1,
            n_folds=3,
            seed=0,
        )
        # If y were truncated to zeros, R^2 would be nan/degenerate.
        assert np.isfinite(out["mean"])

    def test_regression_ci_is_not_clipped_at_zero(self) -> None:
        """R^2 is unbounded below, so a regression summary must not clamp the
        interval to [0, 1] the way a bounded accuracy summary does.

        Asserted on the summariser directly: whether a given model happens to
        score below zero is stochastic, but the clipping rule is exact.
        """
        from encoding_atlas.benchmark.runner import _summarize

        negative = [-0.8, -0.5, -1.2, -0.3]
        unbounded = _summarize(negative, bounded=False)
        assert unbounded["mean"] < 0.0
        assert unbounded["ci_low"] < 0.0  # not clamped to 0.0

        # The bounded (accuracy) summary still clips into [0, 1].
        bounded = _summarize([0.98, 0.99, 1.0], bounded=True)
        assert bounded["ci_high"] <= 1.0
        assert bounded["ci_low"] >= 0.0

    def test_regression_summary_bounds_are_consistent(self) -> None:
        X, y = get_regression_dataset("sine_reg", n_samples=40, seed=0)
        out = evaluate_encoding(
            _angle(),
            X,
            y,
            task="regression",
            method="kernel",
            n_runs=1,
            n_folds=3,
            seed=0,
        )
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]


class TestEncodingBenchmarkRegression:
    def test_run_structure(self) -> None:
        bench = EncodingBenchmark(
            [_angle()],
            ["sine_reg"],
            task="regression",
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=60,
        )
        res = bench.run()
        assert res["config"]["task"] == "regression"
        assert res["config"]["score_metric"] == "r2"
        cell = res["results"]["kernel"]["AngleEncoding"]["sine_reg"]
        assert cell["status"] == "success"
        assert cell["mean"] > 0.5

    def test_regression_baselines(self) -> None:
        bench = EncodingBenchmark(
            [_angle()],
            ["sine_reg"],
            task="regression",
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=60,
            baselines=("svr_rbf",),
        )
        res = bench.run()
        assert np.isfinite(res["baselines"]["svr_rbf"]["sine_reg"]["mean"])

    def test_invalid_task_raises(self) -> None:
        with pytest.raises(ValueError, match="task must be one of"):
            EncodingBenchmark([_angle()], ["sine_reg"], task="ranking")

    def test_plot_regression(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bench = EncodingBenchmark(
            [_angle()],
            ["sine_reg"],
            task="regression",
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=40,
        )
        bench.run()
        fig = bench.plot_comparison()
        assert fig is not None
        plt.close(fig)


class TestClassificationUnaffected:
    """Guard: the default classification path must be unchanged."""

    def test_defaults_to_classification(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0.5, 0.2, (8, 2)), rng.normal(2.7, 0.2, (8, 2))])
        y = np.array([0] * 8 + [1] * 8, dtype=np.intp)
        out = evaluate_encoding(
            _angle(), X, y, method="kernel", n_runs=1, n_folds=3, seed=0
        )
        assert out["task"] == "classification"
        assert out["score_metric"] == "accuracy"
        # Accuracy is bounded, so its interval stays inside [0, 1].
        assert 0.0 <= out["ci_low"] <= 1.0
        assert 0.0 <= out["ci_high"] <= 1.0
