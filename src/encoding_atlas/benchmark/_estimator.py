"""Shared scikit-learn plumbing for the encoding-backed estimators.

The four estimators in this package (:class:`~encoding_atlas.benchmark.\
VQCClassifier`, :class:`~encoding_atlas.benchmark.VQCRegressor`,
:class:`~encoding_atlas.benchmark.QuantumKernelClassifier`,
:class:`~encoding_atlas.benchmark.QuantumKernelRegressor`) follow scikit-learn's
estimator contract so they compose with the ecosystem — ``cross_val_score``,
``GridSearchCV``, ``Pipeline``, ``VotingClassifier`` and anything else that
clones or introspects an estimator.

The contract, and what it requires here
---------------------------------------
``__init__`` stores its arguments verbatim and does nothing else: no
validation, no derived state, no attributes ending in an underscore. That is
what makes ``clone`` and ``set_params`` safe, because both reconstruct an
estimator from ``get_params()`` output. Hyper-parameter validation therefore
happens at ``fit`` time, via :func:`require_at_least` and friends, exactly as
scikit-learn's own estimators do — ``SVC(C=-1)`` also constructs happily and
fails when fitted.

Reading ``encoding.n_qubits`` in ``__init__`` would break the same contract in
a subtler way: ``set_params(encoding=...)`` would leave the cached width stale.
The estimators read the encoding at ``fit`` time instead.

Notes
-----
Validation is written by hand rather than through scikit-learn's
``_validate_data`` / ``validate_data`` helper, whose name and location changed
across the versions this package supports (``scikit-learn >= 1.0``). Only
``BaseEstimator``, the mixins, and ``check_is_fitted`` — all long-stable — are
imported from scikit-learn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

__all__ = [
    "check_feature_matrix",
    "check_targets",
    "require_at_least",
    "require_positive",
]


def require_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is strictly positive.

    Called from ``fit``, not ``__init__`` — see the module docstring.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_at_least(name: str, value: int, minimum: int) -> None:
    """Raise ``ValueError`` unless ``value >= minimum``."""
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def check_feature_matrix(
    estimator: BaseEstimator,
    X: Any,
    *,
    reset: bool,
) -> NDArray[np.float64]:
    """Validate ``X`` and synchronise ``estimator.n_features_in_``.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator being fitted or used for prediction. Must expose an
        ``encoding`` attribute.
    X : array-like of shape (n_samples, n_features)
        Feature matrix to validate.
    reset : bool
        ``True`` when called from ``fit``: sets ``n_features_in_`` from ``X``.
        ``False`` when called from ``predict``: checks ``X`` against the
        feature count seen during ``fit``, as scikit-learn requires.

    Returns
    -------
    ndarray of shape (n_samples, n_features), dtype float64

    Raises
    ------
    ValueError
        If ``X`` is not a 2D numeric array, is empty, contains non-finite
        values, disagrees with the width seen at ``fit`` time, or does not
        match the encoding's ``n_features``.
    """
    X_array = np.asarray(X, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_array.shape}")
    if X_array.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X contains NaN or infinite values")

    n_features = X_array.shape[1]
    if reset:
        estimator.n_features_in_ = n_features
    else:
        expected = getattr(estimator, "n_features_in_", None)
        if expected is not None and n_features != expected:
            raise ValueError(
                f"X has {n_features} features, but this "
                f"{type(estimator).__name__} was fitted with {expected}"
            )

    # The encoding fixes its own width, so a mismatch is a construction error
    # rather than a data error; say so explicitly.
    encoding_features = getattr(
        getattr(estimator, "encoding", None), "n_features", None
    )
    if encoding_features is not None and n_features != encoding_features:
        raise ValueError(
            f"X has {n_features} features but the encoding expects "
            f"{encoding_features}; construct the encoding with "
            f"n_features={n_features}"
        )
    return X_array


def check_targets(y: Any, n_samples: int, *, dtype: Any) -> NDArray[Any]:
    """Validate a target vector against a feature matrix's sample count.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Targets to validate.
    n_samples : int
        Number of rows in the corresponding feature matrix.
    dtype : numpy dtype
        Dtype to cast to — integer for classification, float for regression.

    Returns
    -------
    ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If ``y`` is not 1D or its length disagrees with ``n_samples``.
    """
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y_array.shape}")
    if y_array.shape[0] != n_samples:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: "
            f"{n_samples} and {y_array.shape[0]}"
        )
    return y_array.astype(dtype)
