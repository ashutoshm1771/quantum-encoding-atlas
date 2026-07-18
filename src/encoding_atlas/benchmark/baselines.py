"""Classical machine-learning baselines for benchmark calibration.

Provides standard scikit-learn classifiers (SVM-RBF, random forest, 2-layer
MLP) so quantum encoding results can be contextualised against classical
reference accuracy on the same train/test splits.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Names accepted by :func:`get_classical_baseline`.
CLASSICAL_BASELINE_NAMES: list[str] = ["svm_rbf", "random_forest", "mlp_2layer"]


def get_classical_baseline(name: str, seed: int) -> Any:
    """Return a fresh scikit-learn classifier for the named baseline.

    Parameters
    ----------
    name : {"svm_rbf", "random_forest", "mlp_2layer"}
        Baseline identifier.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If ``name`` is not a known baseline.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    builders = {
        "svm_rbf": lambda: SVC(kernel="rbf", random_state=seed, probability=True),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=seed
        ),
        "mlp_2layer": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=200, random_state=seed
        ),
    }
    if name not in builders:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {CLASSICAL_BASELINE_NAMES}"
        )
    return builders[name]()


def run_baseline_single_fold(
    name: str,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    y_train: NDArray[np.intp],
    y_test: NDArray[np.intp],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a classical baseline on one train/test split.

    Returns a dict with ``test_accuracy``, ``precision``, ``recall``, ``f1``,
    and ``status``. Metrics use macro averaging for multi-class tasks. Failures
    are reported as ``status="failed"``.
    """
    from encoding_atlas.benchmark.metrics import compute_metrics

    try:
        clf = get_classical_baseline(name, seed=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = compute_metrics(y_test, y_pred)
        return {
            "test_accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "status": "success",
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        logger.error("Baseline %s fold failed: %s", name, exc)
        return {
            "test_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "status": "failed",
            "error": str(exc),
        }


# Names accepted by :func:`get_regression_baseline`.
REGRESSION_BASELINE_NAMES: list[str] = [
    "svr_rbf",
    "random_forest_reg",
    "mlp_2layer_reg",
]


def get_regression_baseline(name: str, seed: int) -> Any:
    """Return a fresh scikit-learn regressor for the named baseline.

    Parameters
    ----------
    name : {"svr_rbf", "random_forest_reg", "mlp_2layer_reg"}
        Baseline identifier.
    seed : int
        Random seed (ignored by ``svr_rbf``, which is deterministic).

    Raises
    ------
    ValueError
        If ``name`` is not a known regression baseline.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    builders = {
        # SVR has no random_state parameter; it is deterministic.
        "svr_rbf": lambda: SVR(kernel="rbf"),
        "random_forest_reg": lambda: RandomForestRegressor(
            n_estimators=100, random_state=seed
        ),
        "mlp_2layer_reg": lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=200, random_state=seed
        ),
    }
    if name not in builders:
        raise ValueError(
            f"Unknown regression baseline: {name}. "
            f"Available: {REGRESSION_BASELINE_NAMES}"
        )
    return builders[name]()


def run_regression_baseline_single_fold(
    name: str,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    y_test: NDArray[np.floating[Any]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a classical regression baseline on one split.

    Returns a dict with ``test_r2``, ``mse``, ``rmse``, ``mae``, and ``status``.
    Failures are reported as ``status="failed"`` with ``nan`` scores.
    """
    from encoding_atlas.benchmark.metrics import compute_regression_metrics

    try:
        model = get_regression_baseline(name, seed=seed)
        model.fit(X_train, y_train)
        scores = compute_regression_metrics(y_test, model.predict(X_test))
        return {
            "test_r2": scores["r2"],
            "mse": scores["mse"],
            "rmse": scores["rmse"],
            "mae": scores["mae"],
            "status": "success",
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        logger.error("Regression baseline %s fold failed: %s", name, exc)
        return {
            "test_r2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "status": "failed",
            "error": str(exc),
        }
