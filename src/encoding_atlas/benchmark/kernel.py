"""Quantum-kernel classification for the benchmarking framework.

Implements the fidelity (overlap) kernel

    K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2 = |<0|U(x_i)^dagger U(x_j)|0>|^2

and a precomputed-kernel SVM classifier built on top of it, plus the
kernel-target alignment diagnostics. The statevectors are obtained through the
package's analysis utilities, so the kernel is consistent with the rest of the
library.

References
----------
Havlicek et al. (2019), *Nature* 567:209; Schuld & Killoran (2019), *PRL*
122:040504; Cristianini et al. (2002); Cortes et al. (2012).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from encoding_atlas.analysis.generalization import (
    centered_kernel_target_alignment as _centered_kernel_target_alignment,
)
from encoding_atlas.analysis.generalization import (
    kernel_target_alignment as _kernel_target_alignment,
)
from encoding_atlas.benchmark._estimator import (
    check_feature_matrix,
    check_targets,
    require_positive,
)

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)


def simulate_encoding_state(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    backend: str = "pennylane",
) -> NDArray[np.complexfloating[Any, Any]]:
    """Return the statevector produced by the encoding for input ``x``."""
    from encoding_atlas.analysis._utils import simulate_encoding_statevector

    return simulate_encoding_statevector(encoding, x, backend=backend)  # type: ignore[arg-type]


def compute_kernel_entry(
    state_i: NDArray[np.complexfloating[Any, Any]],
    state_j: NDArray[np.complexfloating[Any, Any]],
) -> float:
    """Return the fidelity ``|<state_i|state_j>|^2`` clamped to ``[0, 1]``."""
    overlap = np.vdot(state_i, state_j)
    return float(np.clip(float(np.abs(overlap) ** 2), 0.0, 1.0))


def compute_kernel_matrix(
    encoding: BaseEncoding,
    X: NDArray[np.floating[Any]],
    *,
    backend: str = "pennylane",
    return_states: bool = False,
) -> Any:
    """Compute the symmetric ``n x n`` fidelity kernel matrix for ``X``.

    Statevectors are simulated once per sample and reused for all pairwise
    fidelities. When ``return_states=True`` a ``(K, states)`` tuple is returned
    so the states can be reused for the cross kernel.
    """
    n = len(X)
    K: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
    states = [simulate_encoding_state(encoding, xi, backend) for xi in X]

    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            K[i, j] = compute_kernel_entry(states[i], states[j])
            K[j, i] = K[i, j]

    if return_states:
        return K, states
    return K


def compute_kernel_matrix_cross(
    encoding: BaseEncoding,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    *,
    train_states: list[NDArray[np.complexfloating[Any, Any]]] | None = None,
    backend: str = "pennylane",
) -> NDArray[np.floating[Any]]:
    """Compute the ``(n_test, n_train)`` kernel between test and train sets."""
    if train_states is None:
        train_states = [simulate_encoding_state(encoding, x, backend) for x in X_train]

    K: NDArray[np.float64] = np.zeros((len(X_test), len(X_train)), dtype=np.float64)
    for i, xi in enumerate(X_test):
        state_i = simulate_encoding_state(encoding, xi, backend)
        for j, state_j in enumerate(train_states):
            K[i, j] = compute_kernel_entry(state_i, state_j)
    return K


def ensure_psd(
    K: NDArray[np.floating[Any]], epsilon: float = 1e-8
) -> tuple[NDArray[np.floating[Any]], bool]:
    """Project ``K`` onto the PSD cone by clipping negative eigenvalues.

    Returns the (symmetrised) PSD matrix and a flag indicating whether any
    negative eigenvalues had to be clipped.
    """
    K_sym = (K + K.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(K_sym)
    if float(np.min(eigenvalues)) >= -epsilon:
        return K_sym, False
    eigenvalues_clipped = np.maximum(eigenvalues, epsilon)
    K_psd = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
    return K_psd, True


# Kernel-target alignment is defined once, in the analysis package, and
# re-exported here so the benchmark keeps its historical public surface. Two
# independent copies had drifted apart in label handling; a single definition
# means the metric the benchmark records and the metric a user screens with can
# never disagree.
kernel_target_alignment = _kernel_target_alignment
centered_kernel_target_alignment = _centered_kernel_target_alignment


class QuantumKernelClassifier(ClassifierMixin, BaseEstimator):
    """Fidelity-kernel SVM classifier for a fixed encoding.

    Computes the quantum kernel for the training data, enforces PSD, and fits a
    precomputed-kernel :class:`sklearn.svm.SVC`. Prediction uses the test/train
    cross kernel.

    Follows scikit-learn's estimator contract, so it composes with
    ``cross_val_score``, ``GridSearchCV``, ``Pipeline`` and other
    meta-estimators. As there, hyper-parameters are validated at ``fit`` time
    rather than in the constructor; see
    :mod:`encoding_atlas.benchmark._estimator`.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding used for state preparation. Fixes the accepted feature count.
    C : float, default=1.0
        SVM regularisation strength. Must be positive.
    seed : int or None, default=None
        Seed forwarded to the SVM for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Class labels seen during ``fit``.
    n_features_in_ : int
        Feature count seen during ``fit``.
    kernel_regularized_ : bool
        Whether the training kernel needed PSD projection.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from encoding_atlas import AngleEncoding
    >>> from encoding_atlas.benchmark import get_dataset
    >>> X, y = get_dataset("moons", n_samples=30, seed=0)
    >>> clf = QuantumKernelClassifier(AngleEncoding(n_features=2))
    >>> scores = cross_val_score(clf, X, y, cv=3)
    >>> len(scores)
    3
    """

    def __init__(
        self, encoding: Any = None, *, C: float = 1.0, seed: int | None = None
    ) -> None:
        # Store verbatim only: no validation, no derived state. See
        # encoding_atlas.benchmark._estimator for why.
        self.encoding = encoding
        self.C = C
        self.seed = seed

    def fit(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]
    ) -> QuantumKernelClassifier:
        """Fit the precomputed-kernel SVM on ``(X, y)``.

        Raises
        ------
        ValueError
            If ``C`` is not positive, ``encoding`` is unset, or ``X``/``y`` are
            malformed or mismatched.
        """
        from sklearn.svm import SVC

        require_positive("C", self.C)
        if self.encoding is None:
            raise ValueError("encoding must be set before fitting")
        X = check_feature_matrix(self, X, reset=True)
        y = check_targets(y, X.shape[0], dtype=np.intp)
        self.classes_ = np.unique(y)

        K_train, states = compute_kernel_matrix(self.encoding, X, return_states=True)
        K_train_psd, self.kernel_regularized_ = ensure_psd(K_train)
        self.svm_ = SVC(kernel="precomputed", C=self.C, random_state=self.seed)
        self.svm_.fit(K_train_psd, y)
        self.X_train_ = X
        self.train_states_ = states
        return self

    def _cross_kernel(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Validate ``X`` and return the test/train cross kernel."""
        check_is_fitted(self)
        X = check_feature_matrix(self, X, reset=False)
        return compute_kernel_matrix_cross(
            self.encoding, self.X_train_, X, train_states=self.train_states_
        )

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        """Predict labels for ``X`` using the test/train cross kernel."""
        # Build the kernel first: attribute access on ``svm_`` would otherwise
        # be evaluated before the fitted check inside ``_cross_kernel``.
        K_test = self._cross_kernel(X)
        return cast("NDArray[np.intp]", self.svm_.predict(K_test).astype(np.intp))

    def decision_function(
        self, X: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Signed distance to the separating hyperplane in feature space.

        Exposed so the classifier works with threshold-based metrics such as
        ROC-AUC, and with probability calibration.
        """
        K_test = self._cross_kernel(X)
        return np.asarray(self.svm_.decision_function(K_test), dtype=np.float64)


class QuantumKernelRegressor(RegressorMixin, BaseEstimator):
    """Fidelity-kernel ridge regressor for a fixed encoding.

    Computes the quantum kernel for the training data, enforces PSD, and fits a
    precomputed-kernel :class:`sklearn.kernel_ridge.KernelRidge`. Prediction
    uses the test/train cross kernel.

    Follows scikit-learn's estimator contract; ``score`` returns R^2 through
    :class:`sklearn.base.RegressorMixin`.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding used for state preparation.
    alpha : float, default=1.0
        Ridge regularisation strength. Must be positive.

    Attributes
    ----------
    n_features_in_ : int
        Feature count seen during ``fit``.
    kernel_regularized_ : bool
        Whether the training kernel needed PSD projection.
    """

    def __init__(self, encoding: Any = None, *, alpha: float = 1.0) -> None:
        self.encoding = encoding
        self.alpha = alpha

    def fit(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> QuantumKernelRegressor:
        """Fit the precomputed-kernel ridge regressor on ``(X, y)``.

        Raises
        ------
        ValueError
            If ``alpha`` is not positive, ``encoding`` is unset, or ``X``/``y``
            are malformed or mismatched.
        """
        from sklearn.kernel_ridge import KernelRidge

        require_positive("alpha", self.alpha)
        if self.encoding is None:
            raise ValueError("encoding must be set before fitting")
        X = check_feature_matrix(self, X, reset=True)
        y = check_targets(y, X.shape[0], dtype=np.float64)

        K_train, states = compute_kernel_matrix(self.encoding, X, return_states=True)
        K_train_psd, self.kernel_regularized_ = ensure_psd(K_train)
        self.model_ = KernelRidge(kernel="precomputed", alpha=self.alpha)
        self.model_.fit(K_train_psd, y)
        self.X_train_ = X
        self.train_states_ = states
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Predict continuous targets using the test/train cross kernel."""
        check_is_fitted(self)
        X = check_feature_matrix(self, X, reset=False)
        K_test = compute_kernel_matrix_cross(
            self.encoding, self.X_train_, X, train_states=self.train_states_
        )
        return np.asarray(self.model_.predict(K_test), dtype=np.float64)


def run_kernel_single_fold(
    encoding: Any,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    y_train: NDArray[np.intp],
    y_test: NDArray[np.intp],
    *,
    C: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a quantum-kernel SVM on one train/test split.

    Returns a dict with ``test_accuracy``, ``precision``, ``recall``, ``f1``,
    ``kernel_target_alignment`` (centred, on the training kernel; ``None`` for
    multi-class tasks, where it is undefined), ``kernel_regularized``, and
    ``status``. Metrics use macro averaging for multi-class tasks. Failures are
    reported as ``status="failed"`` so a sweep can continue.
    """
    from encoding_atlas.benchmark.metrics import compute_metrics

    try:
        K_train, train_states = compute_kernel_matrix(
            encoding, X_train, return_states=True
        )
        K_train_psd, regularized = ensure_psd(K_train)
        # Kernel-target alignment is a binary label-alignment; undefined for >2.
        alignment = (
            centered_kernel_target_alignment(K_train, y_train)
            if len(np.unique(y_train)) == 2
            else None
        )

        from sklearn.svm import SVC

        svm = SVC(kernel="precomputed", C=C, random_state=seed)
        svm.fit(K_train_psd, y_train)

        K_test = compute_kernel_matrix_cross(
            encoding, X_train, X_test, train_states=train_states
        )
        y_pred = svm.predict(K_test)
        scores = compute_metrics(y_test, y_pred)
        return {
            "test_accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "kernel_target_alignment": alignment,
            "kernel_regularized": regularized,
            "status": "success",
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        logger.error("Kernel fold failed: %s", exc)
        return {
            "test_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "kernel_target_alignment": 0.0,
            "kernel_regularized": False,
            "status": "failed",
            "error": str(exc),
        }


def run_kernel_regression_fold(
    encoding: Any,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    y_test: NDArray[np.floating[Any]],
    *,
    alpha: float = 1.0,
    seed: int = 42,  # noqa: ARG001 - kept for a uniform fold-runner signature
) -> dict[str, Any]:
    """Train and evaluate a quantum-kernel ridge regressor on one split.

    Returns a dict with ``test_r2``, ``mse``, ``rmse``, ``mae``,
    ``kernel_regularized``, and ``status``. Failures are reported as
    ``status="failed"`` with ``nan`` scores so a sweep can continue.
    """
    from encoding_atlas.benchmark.metrics import compute_regression_metrics

    try:
        model = QuantumKernelRegressor(encoding, alpha=alpha)
        model.fit(X_train, y_train)
        scores = compute_regression_metrics(y_test, model.predict(X_test))
        return {
            "test_r2": scores["r2"],
            "mse": scores["mse"],
            "rmse": scores["rmse"],
            "mae": scores["mae"],
            "kernel_regularized": model.kernel_regularized_,
            "status": "success",
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        logger.error("Kernel regression fold failed: %s", exc)
        return {
            "test_r2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "kernel_regularized": False,
            "status": "failed",
            "error": str(exc),
        }
