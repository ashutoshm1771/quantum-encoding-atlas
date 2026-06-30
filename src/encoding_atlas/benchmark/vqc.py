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

        self._qnode: Any | None = None
        self._device: Any | None = None
        self._n_qubits: int = encoding.n_qubits

    def _build_circuit(self) -> None:
        """Construct the PennyLane QNode (encoding + variational + ``<Z_0>``)."""
        import pennylane as qml

        n_qubits = self._n_qubits
        n_var_layers = self.n_var_layers
        self._device = qml.device("lightning.qubit", wires=n_qubits)
        encoding = self.encoding

        @qml.qnode(self._device, interface="autograd", diff_method="adjoint")
        def circuit(
            x: NDArray[np.floating[Any]], params: NDArray[np.floating[Any]]
        ) -> float:
            encoding.get_circuit(x, backend="pennylane")()
            for layer in range(n_var_layers):
                for i in range(n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]) -> VQCClassifier:
        """Train on ``X`` (features) and ``y`` (labels in ``{0, 1}``)."""
        import pennylane as qml
        from pennylane import numpy as pnp

        if self._qnode is None:
            self._build_circuit()

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

                def cost(params: NDArray[np.floating[Any]]) -> float:
                    pred = self._qnode(xi, params)
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

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Return class probabilities of shape ``(n_samples, 2)``."""
        if self.params_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self._qnode is None:
            self._build_circuit()

        predictions = []
        for xi in X:
            exp_val = float(self._qnode(xi, self.params_))
            p1 = float(np.clip((1 + exp_val) / 2, 0.0, 1.0))
            predictions.append([1 - p1, p1])
        return np.array(predictions)

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        """Return predicted labels in ``{0, 1}``."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.intp)

    def score(self, X: NDArray[np.floating[Any]], y: NDArray[np.intp]) -> float:
        """Return classification accuracy on ``(X, y)``."""
        return float(np.mean(self.predict(X) == y))

    def get_final_loss(self) -> float | None:
        """Return the final epoch's mean loss, or ``None`` if not fitted."""
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
    ``recall``, ``f1``, ``final_loss``, and ``status``. Failures are caught and
    reported as ``status="failed"`` so a sweep can continue.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

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
        return {
            "train_accuracy": float(np.mean(vqc.predict(X_train) == y_train)),
            "test_accuracy": float(np.mean(y_pred_test == y_test)),
            "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
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
