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
    and ``status``. Failures are reported as ``status="failed"``.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    try:
        clf = get_classical_baseline(name, seed=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return {
            "test_accuracy": float(np.mean(y_pred == y_test)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
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
