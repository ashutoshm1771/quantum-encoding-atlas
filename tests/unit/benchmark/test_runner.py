"""Tests for EncodingBenchmark and evaluate_encoding (fast tiny configs)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.benchmark import EncodingBenchmark, evaluate_encoding


def _angle(rot: str = "Y") -> AngleEncoding:
    return AngleEncoding(n_features=2, rotation=rot)


@pytest.fixture
def custom_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.3, 0.2, (12, 2)), rng.normal(2.7, 0.2, (12, 2))])
    y = np.array([0] * 12 + [1] * 12, dtype=np.intp)
    return X, y


# =====================================================================
# evaluate_encoding (custom data entry point)
# =====================================================================


class TestEvaluateEncoding:
    def test_kernel_custom_data(
        self, custom_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = custom_data
        out = evaluate_encoding(
            _angle(), X, y, method="kernel", n_runs=1, n_folds=3, seed=0
        )
        assert out["method"] == "kernel"
        assert out["n_scores"] == 3
        assert 0.0 <= out["mean"] <= 1.0
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]

    def test_vqc_custom_data(self, custom_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = custom_data
        out = evaluate_encoding(
            _angle(), X, y, method="vqc", n_runs=1, n_folds=2, seed=0, vqc_epochs=3
        )
        assert out["method"] == "vqc"
        assert out["n_scores"] >= 1

    def test_invalid_method(self, custom_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = custom_data
        with pytest.raises(ValueError, match="method must be one of"):
            evaluate_encoding(_angle(), X, y, method="randomforest")

    def test_feature_mismatch_raises(
        self, custom_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = custom_data
        with pytest.raises(ValueError, match="features"):
            evaluate_encoding(IQPEncoding(n_features=4), X, y, method="kernel")

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            evaluate_encoding(_angle(), np.array([1.0, 2.0]), np.array([0, 1]))


# =====================================================================
# EncodingBenchmark
# =====================================================================


class TestConstructorValidation:
    def test_empty_encodings(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EncodingBenchmark([], ["moons"])

    def test_no_datasets(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EncodingBenchmark([_angle()], [])

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="invalid methods"):
            EncodingBenchmark([_angle()], ["moons"], methods=("bogus",))

    def test_invalid_folds(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            EncodingBenchmark([_angle()], ["moons"], n_folds=1)

    def test_backward_compatible_positional(self) -> None:
        # Original documented signature: (encodings, datasets, n_runs, seed).
        bench = EncodingBenchmark([_angle()], ["moons"], 1, 0)
        assert bench.n_runs == 1 and bench.seed == 0


class TestRun:
    def _bench(self, **kw: object) -> EncodingBenchmark:
        defaults = dict(
            encodings=[_angle("Y"), _angle("X")],
            datasets=["moons"],
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=40,
        )
        defaults.update(kw)
        return EncodingBenchmark(**defaults)  # type: ignore[arg-type]

    def test_run_structure(self) -> None:
        res = self._bench().run()
        assert set(res) == {"config", "encodings", "datasets", "results", "baselines"}
        assert res["encodings"] == ["AngleEncoding", "AngleEncoding_2"]  # dedup labels
        cell = res["results"]["kernel"]["AngleEncoding"]["moons"]
        assert cell["status"] == "success"
        assert 0.0 <= cell["mean"] <= 1.0
        assert cell["n_scores"] == 3

    def test_run_with_baselines(self) -> None:
        res = self._bench(baselines=("svm_rbf",)).run()
        assert "svm_rbf" in res["baselines"]
        assert 0.0 <= res["baselines"]["svm_rbf"]["moons"]["mean"] <= 1.0

    def test_run_vqc_method(self) -> None:
        res = self._bench(methods=("vqc",), vqc_epochs=3).run()
        assert res["results"]["vqc"]["AngleEncoding"]["moons"]["status"] == "success"

    def test_mismatch_skipped(self) -> None:
        bench = EncodingBenchmark(
            [IQPEncoding(n_features=4)],
            ["moons"],
            methods=("kernel",),
            n_runs=1,
            n_folds=2,
            seed=0,
            n_samples=30,
        )
        cell = bench.run()["results"]["kernel"]["IQPEncoding"]["moons"]
        assert cell["status"] == "skipped"
        assert "n_features" in cell["reason"]

    def test_custom_dataset(self, custom_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = custom_data
        bench = EncodingBenchmark(
            [_angle()],
            [],
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            custom_datasets={"mine": (X, y)},
        )
        res = bench.run()
        assert "mine" in res["datasets"]
        assert res["results"]["kernel"]["AngleEncoding"]["mine"]["status"] == "success"


class TestStatisticalTests:
    def test_requires_run(self) -> None:
        with pytest.raises(RuntimeError, match="Call run"):
            EncodingBenchmark([_angle()], ["moons"]).statistical_tests()

    def test_pairwise_output(self) -> None:
        bench = EncodingBenchmark(
            [_angle("Y"), _angle("X")],
            ["moons"],
            methods=("kernel",),
            n_runs=2,
            n_folds=3,
            seed=0,
            n_samples=40,
        )
        bench.run()
        stats = bench.statistical_tests()
        assert "kernel/moons" in stats
        assert stats["kernel/moons"]["n_comparisons"] == 1


class TestSaveAndPlot:
    def test_save_requires_run(self, tmp_path: object) -> None:
        with pytest.raises(RuntimeError, match="Call run"):
            EncodingBenchmark([_angle()], ["moons"]).save_results(
                str(tmp_path) + "/x.json"
            )

    def test_save_results_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        bench = EncodingBenchmark(
            [_angle()],
            ["moons"],
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=40,
        )
        bench.run()
        path = tmp_path / "results.json"
        bench.save_results(str(path))
        loaded = json.loads(path.read_text())
        assert loaded["config"]["methods"] == ["kernel"]
        assert "results" in loaded

    def test_plot_requires_run(self) -> None:
        with pytest.raises(RuntimeError, match="Call run"):
            EncodingBenchmark([_angle()], ["moons"]).plot_comparison()

    def test_plot_returns_figure(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        bench = EncodingBenchmark(
            [_angle()],
            ["moons"],
            methods=("kernel",),
            n_runs=1,
            n_folds=3,
            seed=0,
            n_samples=40,
        )
        bench.run()
        fig = bench.plot_comparison()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# =========================================================================
# Feature scaling inside cross-validation
# =========================================================================


class TestFoldScaling:
    """Scaling must be fitted on the training split of each fold.

    Fitting on the whole dataset before splitting lets every test fold's own
    minimum and maximum set the scale it is later evaluated under — a mild but
    real leak in an otherwise carefully paired protocol.
    """

    def test_scaler_is_fitted_on_the_training_split_only(self) -> None:
        from encoding_atlas.benchmark.runner import _scale_fold

        X_train = np.array([[0.0], [1.0]])
        X_test = np.array([[-1.0], [2.0]])
        y = np.array([0, 1])
        scaled_train, scaled_test, _, _ = _scale_fold(
            (X_train, X_test, y, y), (0.0, np.pi)
        )
        # Train spans the target range exactly...
        assert scaled_train.min() == pytest.approx(0.0)
        assert scaled_train.max() == pytest.approx(np.pi)
        # ...and test escapes it, which is only possible without the leak.
        assert scaled_test.min() < 0.0
        assert scaled_test.max() > np.pi

    def test_labels_pass_through_untouched(self) -> None:
        from encoding_atlas.benchmark.runner import _scale_fold

        y_train, y_test = np.array([0, 1]), np.array([1, 0])
        _, _, out_train, out_test = _scale_fold(
            (np.zeros((2, 1)), np.ones((2, 1)), y_train, y_test), (0.0, 1.0)
        )
        assert out_train is y_train
        assert out_test is y_test

    def test_constant_feature_survives(self) -> None:
        from encoding_atlas.benchmark.runner import _scale_fold

        X = np.array([[5.0], [5.0]])
        scaled_train, scaled_test, _, _ = _scale_fold(
            (X, X, np.array([0, 1]), np.array([0, 1])), (0.0, np.pi)
        )
        assert np.all(np.isfinite(scaled_train))
        assert np.all(np.isfinite(scaled_test))

    def test_evaluate_encoding_still_scales(self) -> None:
        """Turning scaling off must change the result, proving it is applied.

        Uses IQP, whose kernel is strongly range-dependent; a shallow
        non-entangling map can coincidentally score the same either way.
        """
        from encoding_atlas import IQPEncoding
        from encoding_atlas.benchmark import evaluate_encoding, get_dataset

        X, y = get_dataset("moons", n_samples=60, seed=0)
        enc = IQPEncoding(n_features=2, reps=2)
        scaled = evaluate_encoding(enc, X, y, method="kernel", n_folds=3)
        unscaled = evaluate_encoding(enc, X, y, method="kernel", n_folds=3, scale=False)
        assert scaled["mean"] != unscaled["mean"]

    def test_scale_range_is_honoured(self) -> None:
        """A different target range must produce a different score."""
        from encoding_atlas import IQPEncoding
        from encoding_atlas.benchmark import evaluate_encoding, get_dataset

        X, y = get_dataset("moons", n_samples=60, seed=0)
        enc = IQPEncoding(n_features=2, reps=2)
        wide = evaluate_encoding(
            enc, X, y, method="kernel", n_folds=3, scale_range=(0.0, 2 * np.pi)
        )
        narrow = evaluate_encoding(
            enc, X, y, method="kernel", n_folds=3, scale_range=(0.0, np.pi / 2)
        )
        assert wide["mean"] != narrow["mean"]

    def test_shared_definition_with_the_analysis_module(self) -> None:
        from encoding_atlas.analysis.scaling import scale_to_range
        from encoding_atlas.benchmark.runner import _scale_features

        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 3))
        assert np.allclose(
            _scale_features(X, 0.0, np.pi), scale_to_range(X, 0.0, np.pi)
        )
