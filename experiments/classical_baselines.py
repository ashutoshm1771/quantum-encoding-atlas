"""Classical ML baselines for comparison with VQC.

This module provides classical machine learning classifiers as baselines
for the VQC benchmark (Stage 6a). These baselines contextualize quantum
results by providing reference accuracy levels.

Baselines
---------
- SVM-RBF: Support Vector Machine with RBF kernel
- Random Forest: Ensemble of decision trees
- MLP (2-layer): Multi-layer Perceptron neural network

All classifiers are scikit-learn compatible and support:
- fit(X, y): Train the model
- predict(X): Predict class labels
- predict_proba(X): Predict class probabilities
- score(X, y): Compute accuracy
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Available baseline names
CLASSICAL_BASELINE_NAMES: list[str] = [
    "svm_rbf",
    "random_forest",
    "mlp_2layer",
]


def get_classical_baseline(name: str, seed: int) -> Any:
    """Get a classical classifier by name.

    Parameters
    ----------
    name : str
        Classifier name. One of:
        - "svm_rbf": Support Vector Machine with RBF kernel
        - "random_forest": Random Forest with 100 trees
        - "mlp_2layer": Multi-layer Perceptron with (32, 16) hidden layers
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    classifier
        A scikit-learn compatible classifier instance.

    Raises
    ------
    ValueError
        If the classifier name is not recognized.

    Examples
    --------
    >>> clf = get_classical_baseline("svm_rbf", seed=42)
    >>> clf.fit(X_train, y_train)
    >>> accuracy = clf.score(X_test, y_test)
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    baselines = {
        "svm_rbf": lambda: SVC(
            kernel="rbf",
            random_state=seed,
            probability=True,  # Enable predict_proba
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
        ),
        "mlp_2layer": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=200,
            random_state=seed,
        ),
    }

    if name not in baselines:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {list(baselines.keys())}"
        )

    clf = baselines[name]()
    logger.debug("Created classical baseline: %s with seed=%d", name, seed)
    return clf


def run_baseline_single_fold(
    name: str,
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_train: NDArray[np.intp],
    y_test: NDArray[np.intp],
    seed: int = 42,
) -> dict[str, Any]:
    """Run a classical baseline for a single fold.

    Convenience function that encapsulates the full train-evaluate cycle
    and returns all relevant metrics in a dictionary.

    Parameters
    ----------
    name : str
        Baseline classifier name.
    X_train, X_test : np.ndarray
        Training and test features.
    y_train, y_test : np.ndarray
        Training and test labels.
    seed : int, default=42
        Random seed.

    Returns
    -------
    dict
        Results dictionary with keys:
        - 'train_accuracy': float
        - 'test_accuracy': float
        - 'precision': float
        - 'recall': float
        - 'f1': float
        - 'status': str ('success' or 'failed')
        - 'error': str (only if status='failed')
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    try:
        clf = get_classical_baseline(name, seed)
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        result = {
            "train_accuracy": float(np.mean(y_pred_train == y_train)),
            "test_accuracy": float(np.mean(y_pred_test == y_test)),
            "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
            "status": "success",
        }

    except Exception as e:
        logger.error("Classical baseline '%s' failed: %s", name, str(e))
        result = {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "status": "failed",
            "error": str(e),
        }

    return result


def run_all_baselines_cv(
    X: NDArray[np.floating],
    y: NDArray[np.intp],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run all classical baselines with cross-validation.

    Runs each baseline classifier with stratified k-fold cross-validation
    and returns aggregated results.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Features.
    y : np.ndarray, shape (n_samples,)
        Labels.
    n_folds : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping baseline name to results dict containing:
        - 'mean_test_accuracy': float
        - 'std_test_accuracy': float
        - 'mean_train_accuracy': float
        - 'per_fold_results': list of fold result dicts
    """
    from experiments.datasets import get_cv_folds

    results = {}

    folds = get_cv_folds(X, y, n_folds=n_folds, seed=seed)

    for baseline_name in CLASSICAL_BASELINE_NAMES:
        fold_results = []
        test_accs = []
        train_accs = []

        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
            fold_seed = seed + fold_idx
            fold_result = run_baseline_single_fold(
                name=baseline_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                seed=fold_seed,
            )
            fold_results.append(fold_result)

            if fold_result["status"] == "success":
                test_accs.append(fold_result["test_accuracy"])
                train_accs.append(fold_result["train_accuracy"])

        results[baseline_name] = {
            "mean_test_accuracy": float(np.mean(test_accs)) if test_accs else 0.0,
            "std_test_accuracy": float(np.std(test_accs, ddof=1)) if test_accs else 0.0,
            "mean_train_accuracy": float(np.mean(train_accs)) if train_accs else 0.0,
            "per_fold_results": fold_results,
        }

        logger.debug(
            "Baseline %s: test_acc=%.3f±%.3f",
            baseline_name,
            results[baseline_name]["mean_test_accuracy"],
            results[baseline_name]["std_test_accuracy"],
        )

    return results
