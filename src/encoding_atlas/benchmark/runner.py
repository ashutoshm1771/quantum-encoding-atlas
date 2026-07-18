"""Benchmark execution framework.

:class:`EncodingBenchmark` evaluates quantum encodings on **classification** or
**regression** tasks using variational quantum models and/or quantum-kernel
methods, with paired cross-validation and optional classical baselines. It turns
the research benchmarking protocol that produced the empirical atlas into a
self-contained, installable API so users can compare encodings on their own
data.

Classification reports accuracy (bounded in ``[0, 1]``) and uses stratified
folds; regression reports R^2 (unbounded below) and uses plain K-fold, since
stratification is undefined for continuous targets.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.benchmark.baselines import (
    run_baseline_single_fold,
    run_regression_baseline_single_fold,
)
from encoding_atlas.benchmark.datasets import get_dataset, get_regression_dataset
from encoding_atlas.benchmark.kernel import (
    run_kernel_regression_fold,
    run_kernel_single_fold,
)
from encoding_atlas.benchmark.statistical import compare_encodings_corrected
from encoding_atlas.benchmark.vqc import run_vqc_regression_fold, run_vqc_single_fold

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

# Supported quantum evaluation methods.
_VALID_METHODS = ("vqc", "kernel")

# Supported learning tasks.
_VALID_TASKS = ("classification", "regression")

# Primary score reported per task (accuracy is bounded in [0, 1]; R^2 is not).
_SCORE_KEY = {"classification": "test_accuracy", "regression": "test_r2"}
_SCORE_METRIC = {"classification": "accuracy", "regression": "r2"}

# Default feature scaling range (radians), matching the empirical pipeline.
_DEFAULT_SCALE = (0.0, 2.0 * math.pi)


def _scale_features(
    X: NDArray[np.floating[Any]], low: float, high: float
) -> NDArray[np.floating[Any]]:
    """Min-max scale each feature into ``[low, high]`` (constant features -> low)."""
    X = np.asarray(X, dtype=np.float64)
    mins = X.min(axis=0)
    span = X.max(axis=0) - mins
    span = np.where(span == 0.0, 1.0, span)
    return low + (X - mins) / span * (high - low)


def _stratified_folds(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.intp],
    n_folds: int,
    seed: int,
) -> list[tuple[Any, Any, Any, Any]]:
    """Return ``n_folds`` deterministic stratified ``(Xtr, Xte, ytr, yte)`` splits."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(X[tr], X[te], y[tr], y[te]) for tr, te in skf.split(X, y)]


def _make_folds(
    X: NDArray[np.floating[Any]],
    y: NDArray[Any],
    n_folds: int,
    seed: int,
    task: str = "classification",
) -> list[tuple[Any, Any, Any, Any]]:
    """Return deterministic CV splits appropriate for the task.

    Classification uses stratified folds; regression uses plain ``KFold``, since
    stratification is undefined for continuous targets.
    """
    if task == "regression":
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(X[tr], X[te], y[tr], y[te]) for tr, te in kf.split(X)]
    return _stratified_folds(X, y, n_folds, seed)


def _summarize(scores: list[float], *, bounded: bool = True) -> dict[str, Any]:
    """Summarise a list of scores (mean, std, 95% CI, count).

    Non-finite scores are dropped. ``bounded`` clips the interval to ``[0, 1]``
    (valid for accuracy); it must be ``False`` for unbounded scores such as
    R^2, which can be negative.
    """
    arr = np.asarray([s for s in scores if np.isfinite(s)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_scores": 0,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    ci_low, ci_high = mean - half, mean + half
    if bounded:
        ci_low, ci_high = max(0.0, ci_low), min(1.0, ci_high)
    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_scores": int(arr.size),
    }


def _label_encodings(encodings: list[Any]) -> list[str]:
    """Return readable, unique labels for a list of encoding instances."""
    labels: list[str] = []
    counts: dict[str, int] = {}
    for enc in encodings:
        base = type(enc).__name__
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return labels


def _run_method_fold(
    method: str,
    encoding: Any,
    fold: tuple[Any, Any, Any, Any],
    seed: int,
    params: dict[str, Any],
    task: str = "classification",
) -> dict[str, Any]:
    """Dispatch a single fold to the VQC or kernel evaluator for the task."""
    X_train, X_test, y_train, y_test = fold
    if task == "regression":
        if method == "vqc":
            return run_vqc_regression_fold(
                encoding,
                X_train,
                X_test,
                y_train,
                y_test,
                n_var_layers=params["vqc_layers"],
                lr=params["vqc_lr"],
                epochs=params["vqc_epochs"],
                seed=seed,
            )
        return run_kernel_regression_fold(
            encoding,
            X_train,
            X_test,
            y_train,
            y_test,
            alpha=params["kernel_alpha"],
            seed=seed,
        )

    if method == "vqc":
        return run_vqc_single_fold(
            encoding,
            X_train,
            X_test,
            y_train,
            y_test,
            n_var_layers=params["vqc_layers"],
            lr=params["vqc_lr"],
            epochs=params["vqc_epochs"],
            seed=seed,
        )
    return run_kernel_single_fold(
        encoding, X_train, X_test, y_train, y_test, C=params["kernel_C"], seed=seed
    )


def evaluate_encoding(
    encoding: Any,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.intp],
    *,
    method: str = "kernel",
    task: str = "classification",
    n_runs: int = 1,
    n_folds: int = 5,
    seed: int = 42,
    scale: bool = True,
    scale_range: tuple[float, float] = _DEFAULT_SCALE,
    vqc_layers: int = 2,
    vqc_epochs: int = 30,
    vqc_lr: float = 0.05,
    kernel_C: float = 1.0,
    kernel_alpha: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one encoding on a single dataset via cross-validation.

    This is the entry point for benchmarking on *custom* data: pass any feature
    matrix ``X`` with either class labels or continuous targets ``y``.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding whose ``n_features`` must equal ``X.shape[1]``.
    X, y : ndarray
        Feature matrix and targets — class labels for ``task="classification"``
        or continuous values for ``task="regression"``.
    method : {"vqc", "kernel"}, default="kernel"
        Quantum model family.
    task : {"classification", "regression"}, default="classification"
        Learning task. Classification reports accuracy and uses stratified
        folds; regression reports R^2 and uses plain K-fold.
    n_runs : int, default=1
        Independent repetitions, each with a different CV split seed.
    n_folds : int, default=5
        CV folds per run.
    seed : int, default=42
        Base random seed.
    scale : bool, default=True
        Min-max scale features into ``scale_range`` before encoding.
    scale_range : tuple, default=(0, 2*pi)
        Target range for feature scaling.
    vqc_layers, vqc_epochs, vqc_lr : int, int, float
        VQC ansatz depth, training epochs, and learning rate.
    kernel_C : float, default=1.0
        SVM regularisation (classification kernel method).
    kernel_alpha : float, default=1.0
        Ridge regularisation (regression kernel method).

    Returns
    -------
    dict
        ``{"method", "task", "score_metric", "scores", "n_failed", ...summary}``
        where the summary keys are
        ``mean``/``std``/``ci_low``/``ci_high``/``n_scores``. The score is
        accuracy for classification and R^2 for regression.

    Raises
    ------
    ValueError
        If ``method``/``task`` is invalid or the encoding's feature count does
        not match ``X``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got {task!r}")

    X = np.asarray(X, dtype=np.float64)
    # Continuous targets must not be truncated to integers.
    y = np.asarray(y, dtype=np.float64 if task == "regression" else np.intp)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if getattr(encoding, "n_features", X.shape[1]) != X.shape[1]:
        raise ValueError(
            f"Encoding expects {encoding.n_features} features but X has "
            f"{X.shape[1]}; construct the encoding with matching n_features."
        )

    if scale:
        X = _scale_features(X, scale_range[0], scale_range[1])

    params = {
        "vqc_layers": vqc_layers,
        "vqc_epochs": vqc_epochs,
        "vqc_lr": vqc_lr,
        "kernel_C": kernel_C,
        "kernel_alpha": kernel_alpha,
    }
    score_key = _SCORE_KEY[task]

    scores: list[float] = []
    n_failed = 0
    for run in range(n_runs):
        folds = _make_folds(X, y, n_folds, seed=seed + run, task=task)
        for fold_idx, fold in enumerate(folds):
            result = _run_method_fold(
                method,
                encoding,
                fold,
                seed=(seed + run) * 100 + fold_idx,
                params=params,
                task=task,
            )
            if result["status"] == "success":
                scores.append(result[score_key])
            else:
                n_failed += 1

    summary = _summarize(scores, bounded=task == "classification")
    return {
        "method": method,
        "task": task,
        "score_metric": _SCORE_METRIC[task],
        "scores": scores,
        "n_failed": n_failed,
        **summary,
    }


class EncodingBenchmark:
    """Framework for benchmarking encodings on classification datasets.

    Parameters
    ----------
    encodings : list[BaseEncoding]
        Encodings to benchmark. Each must accept the datasets' feature count.
    datasets : list[str]
        Built-in dataset names (see
        :func:`encoding_atlas.benchmark.list_datasets`).
    n_runs : int, default=10
        Independent repetitions per configuration (each a different CV split).
    seed : int or None, default=None
        Base random seed (``None`` -> 0).
    methods : tuple, default=("vqc", "kernel")
        Quantum methods to evaluate; subset of ``("vqc", "kernel")``.
    task : {"classification", "regression"}, default="classification"
        Learning task. Regression draws from the regression dataset registry,
        uses K-fold splits, and reports R^2.
    n_folds : int, default=5
        CV folds per run.
    baselines : tuple, default=()
        Classical baseline names for calibration — classifier names (e.g.
        ``("svm_rbf",)``) for classification, regressor names (e.g.
        ``("svr_rbf",)``) for regression.
    custom_datasets : dict or None, default=None
        Optional mapping ``name -> (X, y)`` of user-provided datasets, evaluated
        alongside the named ``datasets``.
    n_samples : int, default=200
        Sample count requested from the built-in dataset generators.
    vqc_layers, vqc_epochs, vqc_lr, kernel_C, scale_range
        Method hyper-parameters (see :func:`evaluate_encoding`).

    Notes
    -----
    Cost scales as ``len(encodings) x len(datasets) x len(methods) x n_runs x
    n_folds`` fold evaluations (VQC folds train a circuit and are the expensive
    part). Start small.
    """

    def __init__(
        self,
        encodings: list[BaseEncoding],
        datasets: list[str],
        n_runs: int = 10,
        seed: int | None = None,
        *,
        methods: tuple[str, ...] = ("vqc", "kernel"),
        task: str = "classification",
        n_folds: int = 5,
        baselines: tuple[str, ...] = (),
        custom_datasets: dict[str, tuple[Any, Any]] | None = None,
        n_samples: int = 200,
        vqc_layers: int = 2,
        vqc_epochs: int = 30,
        vqc_lr: float = 0.05,
        kernel_C: float = 1.0,
        kernel_alpha: float = 1.0,
        scale_range: tuple[float, float] = _DEFAULT_SCALE,
    ) -> None:
        if not encodings:
            raise ValueError("encodings must be a non-empty list")
        if not datasets and not custom_datasets:
            raise ValueError("provide at least one named or custom dataset")
        invalid = [m for m in methods if m not in _VALID_METHODS]
        if invalid:
            raise ValueError(f"invalid methods {invalid}; valid: {_VALID_METHODS}")
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS}, got {task!r}")
        if n_runs < 1 or n_folds < 2:
            raise ValueError("n_runs must be >= 1 and n_folds must be >= 2")

        self.encodings = encodings
        self.datasets = datasets
        self.n_runs = n_runs
        self.seed = 0 if seed is None else seed
        self.methods = tuple(methods)
        self.task = task
        self.n_folds = n_folds
        self.baselines = tuple(baselines)
        self.custom_datasets = custom_datasets or {}
        self.n_samples = n_samples
        self._params = {
            "vqc_layers": vqc_layers,
            "vqc_epochs": vqc_epochs,
            "vqc_lr": vqc_lr,
            "kernel_C": kernel_C,
            "kernel_alpha": kernel_alpha,
        }
        self.scale_range = scale_range

        self.labels = _label_encodings(encodings)
        self.results: dict[str, Any] = {}
        # Raw per-(method, dataset) scores keyed by encoding label, for stats.
        self._raw: dict[tuple[str, str], dict[str, list[float]]] = {}

    def _resolve_datasets(self) -> dict[str, tuple[Any, Any]]:
        """Load named datasets and merge with any custom datasets (scaled).

        Regression tasks draw from the regression dataset registry and keep
        targets as floats; classification casts labels to integers.
        """
        regression = self.task == "regression"
        target_dtype = np.float64 if regression else np.intp
        loader = get_regression_dataset if regression else get_dataset

        resolved: dict[str, tuple[Any, Any]] = {}
        for name in self.datasets:
            X, y = loader(name, n_samples=self.n_samples, seed=self.seed)
            resolved[name] = (
                _scale_features(np.asarray(X, float), *self.scale_range),
                np.asarray(y, target_dtype),
            )
        for name, (X, y) in self.custom_datasets.items():
            resolved[name] = (
                _scale_features(np.asarray(X, float), *self.scale_range),
                np.asarray(y, target_dtype),
            )
        return resolved

    def run(self) -> dict[str, Any]:
        """Run the full benchmark and return a structured results dictionary.

        Returns
        -------
        dict
            ``{"config", "encodings", "datasets", "results", "baselines"}``.
            ``results[method][label][dataset]`` holds the accuracy summary;
            mismatched (encoding, dataset) feature counts are recorded with
            ``status="skipped"``.
        """
        data = self._resolve_datasets()
        results: dict[str, Any] = {
            m: {label: {} for label in self.labels} for m in self.methods
        }
        baselines: dict[str, Any] = {b: {} for b in self.baselines}
        score_key = _SCORE_KEY[self.task]
        bounded = self.task == "classification"

        for dname, (X, y) in data.items():
            # Pre-compute the shared fold splits per run for paired comparison.
            run_folds = [
                _make_folds(X, y, self.n_folds, seed=self.seed + r, task=self.task)
                for r in range(self.n_runs)
            ]

            for method in self.methods:
                for enc, label in zip(self.encodings, self.labels):
                    if getattr(enc, "n_features", X.shape[1]) != X.shape[1]:
                        results[method][label][dname] = {
                            "status": "skipped",
                            "reason": (
                                f"encoding n_features={enc.n_features} != "
                                f"dataset n_features={X.shape[1]}"
                            ),
                        }
                        continue

                    scores: list[float] = []
                    for r, folds in enumerate(run_folds):
                        for fold_idx, fold in enumerate(folds):
                            res = _run_method_fold(
                                method,
                                enc,
                                fold,
                                seed=(self.seed + r) * 100 + fold_idx,
                                params=self._params,
                                task=self.task,
                            )
                            if res["status"] == "success":
                                scores.append(res[score_key])

                    results[method][label][dname] = {
                        "status": "success" if scores else "failed",
                        **_summarize(scores, bounded=bounded),
                    }
                    self._raw[(method, dname)] = self._raw.get((method, dname), {})
                    self._raw[(method, dname)][label] = scores

            # Classical baselines (encoding-independent) — one pass per dataset.
            baseline_runner = (
                run_regression_baseline_single_fold
                if self.task == "regression"
                else run_baseline_single_fold
            )
            for bname in self.baselines:
                bscores: list[float] = []
                for r, folds in enumerate(run_folds):
                    for fold_idx, (Xtr, Xte, ytr, yte) in enumerate(folds):
                        res = baseline_runner(
                            bname,
                            Xtr,
                            Xte,
                            ytr,
                            yte,
                            seed=(self.seed + r) * 100 + fold_idx,
                        )
                        if res["status"] == "success":
                            bscores.append(res[score_key])
                baselines[bname][dname] = {
                    "status": "success" if bscores else "failed",
                    **_summarize(bscores, bounded=bounded),
                }

        self.results = {
            "config": {
                "n_runs": self.n_runs,
                "n_folds": self.n_folds,
                "methods": list(self.methods),
                "task": self.task,
                "score_metric": _SCORE_METRIC[self.task],
                "baselines": list(self.baselines),
                "seed": self.seed,
                **self._params,
            },
            "encodings": list(self.labels),
            "datasets": list(data.keys()),
            "results": results,
            "baselines": baselines,
        }
        return self.results

    def statistical_tests(self, *, alpha: float = 0.05) -> dict[str, Any]:
        """Pairwise encoding comparison per (method, dataset).

        Runs :func:`compare_encodings_corrected` (Wilcoxon + Holm-Bonferroni +
        Cliff's delta) on the paired per-fold scores. Requires :meth:`run` to
        have been called first.

        Returns
        -------
        dict
            ``{f"{method}/{dataset}": {<corrected comparison>}}``.
        """
        if not self._raw:
            raise RuntimeError("Call run() before statistical_tests().")

        out: dict[str, Any] = {}
        for (method, dname), per_label in self._raw.items():
            usable = {
                label: scores for label, scores in per_label.items() if len(scores) >= 2
            }
            lengths = {len(s) for s in usable.values()}
            if len(usable) < 2 or len(lengths) != 1:
                # Need >= 2 encodings with equal-length paired scores.
                continue
            out[f"{method}/{dname}"] = compare_encodings_corrected(usable, alpha=alpha)
        return out

    def plot_comparison(self, *, method: str | None = None) -> Any:
        """Return a matplotlib bar chart of mean accuracy per encoding/dataset.

        Parameters
        ----------
        method : str or None
            Which method to plot. Defaults to the first configured method.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called.
        ImportError
            If matplotlib is not installed.
        """
        if not self.results:
            raise RuntimeError("Call run() before plot_comparison().")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib required for plotting") from exc

        method = method or self.methods[0]
        datasets = self.results["datasets"]
        labels = self.results["encodings"]
        method_results = self.results["results"][method]

        x = np.arange(len(datasets))
        width = 0.8 / max(len(labels), 1)
        fig, ax = plt.subplots(figsize=(max(8, 2 * len(datasets)), 6))
        for i, label in enumerate(labels):
            means = [
                method_results[label].get(d, {}).get("mean", float("nan"))
                for d in datasets
            ]
            ax.bar(x + i * width, means, width, label=label)
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        if self.task == "regression":
            # R^2 is unbounded below, so let matplotlib pick the limits.
            ax.set_ylabel("Mean test $R^2$")
            ax.axhline(0.0, color="grey", linewidth=0.8)
        else:
            ax.set_ylabel("Mean test accuracy")
            ax.set_ylim(0, 1)
        ax.set_title(f"Encoding comparison ({method}, {self.task})")
        ax.legend(fontsize="small")
        fig.tight_layout()
        return fig

    def save_results(self, path: str) -> None:
        """Write the benchmark results to ``path`` as JSON.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called.
        """
        if not self.results:
            raise RuntimeError("Call run() before save_results().")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
