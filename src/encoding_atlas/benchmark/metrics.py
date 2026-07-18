"""Evaluation metrics for benchmarking (classification and regression)."""

from __future__ import annotations

import numpy as np


def _average_for(y_true: np.ndarray) -> str:
    """Return the sklearn averaging mode: ``"binary"`` for two classes, else
    ``"macro"`` (so every class contributes equally)."""
    return "binary" if len(np.unique(y_true)) == 2 else "macro"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1.

    Binary tasks use standard positive-class precision/recall/F1; multi-class
    tasks (more than two labels) use macro averaging. Values are ``0.0`` where
    a class has no predictions (``zero_division=0``).

    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    dict
        ``{"accuracy", "precision", "recall", "f1"}``.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average = _average_for(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression error metrics.

    Parameters
    ----------
    y_true : ndarray
        True continuous targets.
    y_pred : ndarray
        Predicted targets.

    Returns
    -------
    dict
        ``{"mse", "rmse", "mae", "r2"}``. ``r2`` is ``nan`` for fewer than two
        samples, where the coefficient of determination is undefined.

    Notes
    -----
    ``r2`` (the coefficient of determination) is unbounded below: a model worse
    than predicting the mean yields a negative value.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
    }
