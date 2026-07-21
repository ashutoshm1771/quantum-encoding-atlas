"""Variational quantum classifier for the benchmarking framework.

A lightweight VQC used to evaluate how well a quantum encoding supports a
downstream binary classification task:

    Encoding(x) -> Variational(theta) -> <Z_0> -> class probability

The variational ansatz is ``n_var_layers`` layers of RY rotations followed by a
linear CNOT chain. Training uses binary cross-entropy with the Adam optimizer
(online SGD with per-epoch shuffling for reproducible dynamics).

This mirrors the protocol used to produce the empirical atlas, lifted into the
installed package so users can benchmark encodings on their own data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Maximum average epoch loss before training is considered diverged.
_MAX_LOSS = 10.0


class VQCClassifier:
    """Variational quantum classifier with a configurable encoding.

    Architecture::

        |0> -- Encoding(x) -- [RY(theta) + CNOT chain] x L -- <Z_0> -- class

    Parameters
    ----------
    encoding : BaseEncoding
        The quantum encoding to benchmark. Must expose ``n_qubits`` and a
        ``get_circuit(x, backend="pennylane")`` method.
    n_var_layers : int, default=2
        Number of variational layers after the encoding.
    lr : float, default=0.05
        Adam learning rate.
    epochs : int, default=30
        Number of training epochs.
    seed : int or None, default=None
        Random seed for parameter initialisation and shuffling.

    Attributes
    ----------
    params_ : ndarray or None
        Trained variational parameters of shape ``(n_var_layers, n_qubits)``.
    loss_history_ : list[float]
        Mean training loss per epoch.
    status_ : str
        One of ``"not_fitted"``, ``"success"``, ``"diverged"``.
    """

    def __init__(
        self,
        encoding: Any,
        n_var_layers: int = 2,
        lr: float = 0.05,
        epochs: int = 30,
        seed: int | None = None,
    ) -> None:
        if n_var_layers < 1:
            raise ValueError(f"n_var_layers must be at least 1, got {n_var_layers}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")

        self.encoding = encoding
        self.n_var_layers = n_var_layers
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

        self.params_: NDArray[np.floating[Any]] | None = None
        self.loss_history_: list[float] = []
        self.status_: str = "not_fitted"
        self.classes_: NDArray[np.intp] | None = None

        self._qnode: Any | None = None
        self._device: Any | None = None
        self._n_qubits: int = encoding.n_qubits
        # For >2 classes, a one-vs-rest ensemble of binary VQCs is used.
        self._ovr_models: list[VQCClassifier] | None = None

    def _build_circuit(self) -> None:
        """Construct the PennyLane QNode (encoding + variational + ``<Z_0>``)."""
        import pennylane as qml

        n_qubits = self._n_qubits
        n_var_layers = self.n_var_layers
        self._device = qml.device("lightning.qubit", wires=n_qubits)
        encoding = self.encoding

        @qml.qnode(self._device, interface="autograd", diff_method="adjoint")  # type: ignore[untyped-decorator]
        def circuit(
            x: NDArray[np.floating[Any]], params: NDArray[np.floating[Any]]
        ) -> Any:
            encoding.get_circuit(x, backend="pennylane")()
            for layer in range(n_var_layers):
                for i in range(n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]) -> VQCClassifier:
        """Train on ``X`` and labels ``y``.

        Two-class problems (labels in ``{0, 1}``) train a single VQC that reads
        out ``<Z_0>``. Problems with more than two classes train a one-vs-rest
        ensemble of binary VQCs.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            return self._fit_multiclass(X, y)

        # --- Binary path (labels in {0, 1}) -------------------------------
        self._ovr_models = None
        import pennylane as qml
        from pennylane import numpy as pnp

        if self._qnode is None:
            self._build_circuit()
        assert self._qnode is not None  # set by _build_circuit
        qnode = self._qnode

        rng = np.random.default_rng(self.seed)
        self.params_ = pnp.array(
            rng.uniform(-np.pi, np.pi, size=(self.n_var_layers, self._n_qubits)),
            requires_grad=True,
        )

        opt = qml.AdamOptimizer(stepsize=self.lr)
        self.loss_history_ = []
        self.status_ = "success"
        n_samples = len(X)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            indices = rng.permutation(n_samples)
            X_epoch, y_epoch = X[indices], y[indices]

            for xi, yi in zip(X_epoch, y_epoch):

                def cost(params: NDArray[np.floating[Any]]) -> Any:
                    pred = qnode(xi, params)
                    p = (pred + 1) / 2
                    p = pnp.clip(p, 1e-7, 1 - 1e-7)
                    return -yi * pnp.log(p) - (1 - yi) * pnp.log(1 - p)

                self.params_, loss = opt.step_and_cost(cost, self.params_)
                epoch_loss += float(loss)

            avg_loss = epoch_loss / n_samples
            self.loss_history_.append(avg_loss)

            if np.isnan(avg_loss) or avg_loss > _MAX_LOSS:
                logger.warning("VQC training diverged at epoch %d", epoch)
                self.status_ = "diverged"
                break

        return self

    def _fit_multiclass(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]
    ) -> VQCClassifier:
        """Train a one-vs-rest ensemble, one binary VQC per class."""
        assert self.classes_ is not None
        self._ovr_models = []
        statuses = []
        for i, cls in enumerate(self.classes_):
            sub = VQCClassifier(
                self.encoding,
                n_var_layers=self.n_var_layers,
                lr=self.lr,
                epochs=self.epochs,
                seed=None if self.seed is None else self.seed + i,
            )
            sub.fit(X, (y == cls).astype(np.intp))
            self._ovr_models.append(sub)
            statuses.append(sub.status_)
        self.status_ = "diverged" if "diverged" in statuses else "success"
        self.loss_history_ = []
        return self

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Return class probabilities of shape ``(n_samples, n_classes)``."""
        if self._ovr_models is not None:
            columns = [model.predict_proba(X)[:, 1] for model in self._ovr_models]
            probs = np.column_stack(columns)
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            probs_normalized: NDArray[np.floating[Any]] = probs / row_sums
            return probs_normalized

        if self.params_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self._qnode is None:
            self._build_circuit()
        assert self._qnode is not None  # set by _build_circuit

        predictions = []
        for xi in X:
            exp_val = float(self._qnode(xi, self.params_))
            p1 = float(np.clip((1 + exp_val) / 2, 0.0, 1.0))
            predictions.append([1 - p1, p1])
        return np.array(predictions)

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        """Return predicted labels (in ``{0, 1}`` for binary tasks)."""
        if self._ovr_models is not None:
            assert self.classes_ is not None
            indices = np.argmax(self.predict_proba(X), axis=1)
            labels: NDArray[np.intp] = self.classes_[indices].astype(np.intp)
            return labels
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.intp)

    def score(self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]) -> float:
        """Return classification accuracy on ``(X, y)``."""
        return float(np.mean(self.predict(X) == y))

    def get_final_loss(self) -> float | None:
        """Return the final training loss, or ``None`` if not fitted.

        For a one-vs-rest ensemble this is the mean of the sub-models' final
        losses.
        """
        if self._ovr_models is not None:
            losses = [
                loss
                for model in self._ovr_models
                if (loss := model.get_final_loss()) is not None
            ]
            return float(np.mean(losses)) if losses else None
        return self.loss_history_[-1] if self.loss_history_ else None


class VQCRegressor:
    """Variational quantum regressor with a configurable encoding.

    Uses the same circuit as :class:`VQCClassifier` but reads ``<Z_0>`` as a
    continuous prediction and trains with mean-squared error. Targets are
    min-max scaled into ``[-1, 1]`` (the range of ``<Z_0>``) using **training**
    statistics only, and predictions are mapped back to the original scale.

    Parameters
    ----------
    encoding : BaseEncoding
        The quantum encoding to benchmark.
    n_var_layers : int, default=2
        Number of variational layers after the encoding.
    lr : float, default=0.05
        Adam learning rate.
    epochs : int, default=30
        Number of training epochs.
    seed : int or None, default=None
        Random seed for parameter initialisation and shuffling.
    """

    def __init__(
        self,
        encoding: Any,
        n_var_layers: int = 2,
        lr: float = 0.05,
        epochs: int = 30,
        seed: int | None = None,
    ) -> None:
        if n_var_layers < 1:
            raise ValueError(f"n_var_layers must be at least 1, got {n_var_layers}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")

        self.encoding = encoding
        self.n_var_layers = n_var_layers
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

        self.params_: NDArray[np.floating[Any]] | None = None
        self.loss_history_: list[float] = []
        self.status_: str = "not_fitted"

        self._qnode: Any | None = None
        self._device: Any | None = None
        self._n_qubits: int = encoding.n_qubits
        self._y_min: float = 0.0
        self._y_span: float = 1.0

    def _build_circuit(self) -> None:
        """Construct the PennyLane QNode (encoding + variational + ``<Z_0>``)."""
        import pennylane as qml

        n_qubits = self._n_qubits
        n_var_layers = self.n_var_layers
        self._device = qml.device("lightning.qubit", wires=n_qubits)
        encoding = self.encoding

        @qml.qnode(self._device, interface="autograd", diff_method="adjoint")  # type: ignore[untyped-decorator]
        def circuit(
            x: NDArray[np.floating[Any]], params: NDArray[np.floating[Any]]
        ) -> Any:
            encoding.get_circuit(x, backend="pennylane")()
            for layer in range(n_var_layers):
                for i in range(n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit

    def fit(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> VQCRegressor:
        """Train on ``X`` and continuous targets ``y`` using MSE loss."""
        import pennylane as qml
        from pennylane import numpy as pnp

        if self._qnode is None:
            self._build_circuit()
        assert self._qnode is not None  # set by _build_circuit
        qnode = self._qnode

        y = np.asarray(y, dtype=np.float64)
        self._y_min = float(y.min())
        span = float(y.max()) - self._y_min
        self._y_span = span if span > 0.0 else 1.0
        # Map targets onto [-1, 1], the range of <Z_0>.
        y_scaled = 2.0 * (y - self._y_min) / self._y_span - 1.0

        rng = np.random.default_rng(self.seed)
        self.params_ = pnp.array(
            rng.uniform(-np.pi, np.pi, size=(self.n_var_layers, self._n_qubits)),
            requires_grad=True,
        )

        opt = qml.AdamOptimizer(stepsize=self.lr)
        self.loss_history_ = []
        self.status_ = "success"
        n_samples = len(X)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            indices = rng.permutation(n_samples)
            X_epoch, y_epoch = X[indices], y_scaled[indices]

            for xi, ti in zip(X_epoch, y_epoch):

                def cost(params: NDArray[np.floating[Any]]) -> Any:
                    return (qnode(xi, params) - ti) ** 2

                self.params_, loss = opt.step_and_cost(cost, self.params_)
                epoch_loss += float(loss)

            avg_loss = epoch_loss / n_samples
            self.loss_history_.append(avg_loss)

            if np.isnan(avg_loss) or avg_loss > _MAX_LOSS:
                logger.warning("VQC regression diverged at epoch %d", epoch)
                self.status_ = "diverged"
                break

        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Predict continuous targets on the original target scale."""
        if self.params_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self._qnode is None:
            self._build_circuit()
        assert self._qnode is not None  # set by _build_circuit

        raw = np.array([float(self._qnode(xi, self.params_)) for xi in X])
        return (raw + 1.0) / 2.0 * self._y_span + self._y_min

    def score(
        self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> float:
        """Return the coefficient of determination (R^2) on ``(X, y)``."""
        from encoding_atlas.benchmark.metrics import compute_regression_metrics

        return compute_regression_metrics(y, self.predict(X))["r2"]

    def get_final_loss(self) -> float | None:
        """Return the final training loss, or ``None`` if not fitted."""
        return self.loss_history_[-1] if self.loss_history_ else None


def run_vqc_single_fold(
    encoding: Any,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    y_train: NDArray[np.intp],
    y_test: NDArray[np.intp],
    *,
    n_var_layers: int = 2,
    lr: float = 0.05,
    epochs: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a VQC on one train/test split.

    Returns a dict with ``test_accuracy``, ``train_accuracy``, ``precision``,
    ``recall``, ``f1``, ``final_loss``, and ``status``. Metrics use macro
    averaging for multi-class tasks. Failures are caught and reported as
    ``status="failed"`` so a sweep can continue.
    """
    from encoding_atlas.benchmark.metrics import compute_metrics

    try:
        vqc = VQCClassifier(
            encoding=encoding,
            n_var_layers=n_var_layers,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        vqc.fit(X_train, y_train)
        y_pred_test = vqc.predict(X_test)
        scores = compute_metrics(y_test, y_pred_test)
        return {
            "train_accuracy": float(np.mean(vqc.predict(X_train) == y_train)),
            "test_accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "final_loss": vqc.get_final_loss(),
            "status": vqc.status_,
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        logger.error("VQC fold failed: %s", exc)
        return {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "final_loss": None,
            "status": "failed",
            "error": str(exc),
        }


def run_vqc_regression_fold(
    encoding: Any,
    X_train: NDArray[np.floating[Any]],
    X_test: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    y_test: NDArray[np.floating[Any]],
    *,
    n_var_layers: int = 2,
    lr: float = 0.05,
    epochs: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a VQC regressor on one train/test split.

    Returns a dict with ``test_r2``, ``mse``, ``rmse``, ``mae``, ``final_loss``,
    and ``status``. Failures are reported as ``status="failed"`` so a sweep can
    continue; failed folds report ``test_r2`` as ``nan`` (0.0 would falsely
    imply mean-level performance).
    """
    from encoding_atlas.benchmark.metrics import compute_regression_metrics

    try:
        model = VQCRegressor(
            encoding=encoding,
            n_var_layers=n_var_layers,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        model.fit(X_train, y_train)
        scores = compute_regression_metrics(y_test, model.predict(X_test))
        return {
            "test_r2": scores["r2"],
            "mse": scores["mse"],
            "rmse": scores["rmse"],
            "mae": scores["mae"],
            "final_loss": model.get_final_loss(),
            "status": model.status_,
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        logger.error("VQC regression fold failed: %s", exc)
        return {
            "test_r2": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "final_loss": None,
            "status": "failed",
            "error": str(exc),
        }
