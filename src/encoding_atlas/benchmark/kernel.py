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
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

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
    K = np.zeros((n, n), dtype=np.float64)
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

    K = np.zeros((len(X_test), len(X_train)), dtype=np.float64)
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


def kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Uncentred kernel-target alignment in ``[-1, 1]`` (Cristianini, 2002)."""
    y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    y_outer = np.outer(y_signed, y_signed)
    norm_K = float(np.linalg.norm(K, "fro"))
    norm_y = float(np.linalg.norm(y_outer, "fro"))
    if norm_K < 1e-10 or norm_y < 1e-10:
        return 0.0
    return float(np.sum(K * y_outer) / (norm_K * norm_y))


def centered_kernel_target_alignment(
    K: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
) -> float:
    """Centred kernel-target alignment in ``[-1, 1]`` (Cortes et al., 2012).

    The centring removes the inflation the uncentred score suffers for fidelity
    kernels (whose diagonal is always 1), giving a more faithful measure of an
    encoding's task alignment.
    """
    n = K.shape[0]
    if n < 2:
        return 0.0
    y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    y_outer = np.outer(y_signed, y_signed)
    K_c = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    y_c = (
        y_outer
        - y_outer.mean(axis=0, keepdims=True)
        - y_outer.mean(axis=1, keepdims=True)
        + y_outer.mean()
    )
    norm_K = float(np.linalg.norm(K_c, "fro"))
    norm_y = float(np.linalg.norm(y_c, "fro"))
    if norm_K < 1e-10 or norm_y < 1e-10:
        return 0.0
    return float(np.sum(K_c * y_c) / (norm_K * norm_y))


class QuantumKernelClassifier:
    """Fidelity-kernel SVM classifier for a fixed encoding.

    Computes the quantum kernel for the training data, enforces PSD, and fits a
    precomputed-kernel :class:`sklearn.svm.SVC`. Prediction uses the test/train
    cross kernel.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding used for state preparation.
    C : float, default=1.0
        SVM regularisation strength.
    seed : int or None, default=None
        Seed forwarded to the SVM for reproducibility.
    """

    def __init__(
        self, encoding: Any, *, C: float = 1.0, seed: int | None = None
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        self.encoding = encoding
        self.C = C
        self.seed = seed
        self._svm: Any | None = None
        self._X_train: NDArray[np.floating[Any]] | None = None
        self._train_states: list[NDArray[np.complexfloating[Any, Any]]] | None = None
        self.kernel_regularized_: bool = False

    def fit(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]
    ) -> QuantumKernelClassifier:
        """Fit the precomputed-kernel SVM on ``(X, y)``."""
        from sklearn.svm import SVC

        K_train, states = compute_kernel_matrix(self.encoding, X, return_states=True)
        K_train_psd, self.kernel_regularized_ = ensure_psd(K_train)
        self._svm = SVC(kernel="precomputed", C=self.C, random_state=self.seed)
        self._svm.fit(K_train_psd, y)
        self._X_train = X
        self._train_states = states
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        """Predict labels for ``X`` using the test/train cross kernel."""
        if self._svm is None or self._X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
        K_test = compute_kernel_matrix_cross(
            self.encoding, self._X_train, X, train_states=self._train_states
        )
        return self._svm.predict(K_test).astype(np.intp)

    def score(self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]) -> float:
        """Return classification accuracy on ``(X, y)``."""
        return float(np.mean(self.predict(X) == y))


class QuantumKernelRegressor:
    """Fidelity-kernel ridge regressor for a fixed encoding.

    Computes the quantum kernel for the training data, enforces PSD, and fits a
    precomputed-kernel :class:`sklearn.kernel_ridge.KernelRidge`. Prediction
    uses the test/train cross kernel.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding used for state preparation.
    alpha : float, default=1.0
        Ridge regularisation strength (must be positive).
    """

    def __init__(self, encoding: Any, *, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.encoding = encoding
        self.alpha = alpha
        self._model: Any | None = None
        self._X_train: NDArray[np.floating[Any]] | None = None
        self._train_states: list[NDArray[np.complexfloating[Any, Any]]] | None = None
        self.kernel_regularized_: bool = False

    def fit(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> QuantumKernelRegressor:
        """Fit the precomputed-kernel ridge regressor on ``(X, y)``."""
        from sklearn.kernel_ridge import KernelRidge

        K_train, states = compute_kernel_matrix(self.encoding, X, return_states=True)
        K_train_psd, self.kernel_regularized_ = ensure_psd(K_train)
        self._model = KernelRidge(kernel="precomputed", alpha=self.alpha)
        self._model.fit(K_train_psd, np.asarray(y, dtype=np.float64))
        self._X_train = X
        self._train_states = states
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Predict continuous targets using the test/train cross kernel."""
        if self._model is None or self._X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
        K_test = compute_kernel_matrix_cross(
            self.encoding, self._X_train, X, train_states=self._train_states
        )
        return np.asarray(self._model.predict(K_test), dtype=np.float64)

    def score(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> float:
        """Return the coefficient of determination (R^2) on ``(X, y)``."""
        from encoding_atlas.benchmark.metrics import compute_regression_metrics

        return compute_regression_metrics(y, self.predict(X))["r2"]


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
