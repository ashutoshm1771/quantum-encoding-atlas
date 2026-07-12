"""Evaluation metrics for benchmarking (binary and multi-class)."""

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
