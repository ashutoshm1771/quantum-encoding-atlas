"""Variational Quantum Classifier for Stage 6a experiments.

This module implements a lightweight VQC for benchmarking quantum encodings
on classification tasks. The architecture follows the standard pattern:

    Encoding(x) → Variational(θ) → Measurement → Classical output

The variational ansatz consists of L layers, each containing:
1. RY rotations on each qubit (parameterized by θ)
2. A linear CNOT chain for entanglement

The measurement is the expectation value of Z on qubit 0, which is mapped
to class probabilities via (1 + ⟨Z⟩) / 2.

Training uses binary cross-entropy loss with the Adam optimizer (online SGD
with per-epoch shuffling for order-independent training dynamics).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Maximum loss value before considering training diverged
_MAX_LOSS = 10.0


class VQCClassifier:
    """Variational Quantum Classifier with configurable encoding.

    Architecture
    ------------
    |0⟩ ─── Encoding(x) ─── Variational(θ) ─── ⟨Z₀⟩ ─── threshold ─── class

    The variational ansatz is L layers of:
        [RY(θ) on each qubit] → [linear CNOT chain]

    Parameters
    ----------
    encoding : BaseEncoding
        The quantum encoding to use. Must have a get_circuit() method
        that returns a PennyLane-compatible circuit function.
    n_var_layers : int, default=2
        Number of variational layers after encoding.
    lr : float, default=0.01
        Learning rate for Adam optimizer.
    epochs : int, default=100
        Number of training epochs.
    seed : int, optional
        Random seed for parameter initialization and reproducibility.

    Attributes
    ----------
    params_ : np.ndarray or None
        Trained variational parameters, shape (n_var_layers, n_qubits).
        None before fit() is called.
    loss_history_ : list[float]
        Training loss per epoch. Empty before fit() is called.
    status_ : str
        Training status: 'not_fitted', 'success', 'diverged'.

    Examples
    --------
    >>> from encoding_atlas import IQPEncoding
    >>> encoding = IQPEncoding(n_features=2, reps=1)
    >>> vqc = VQCClassifier(encoding, n_var_layers=2, epochs=50, seed=42)
    >>> X_train = np.array([[0.1, 0.2], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>> vqc.fit(X_train, y_train)
    >>> predictions = vqc.predict(X_train)
    """

    def __init__(
        self,
        encoding: Any,
        n_var_layers: int = 2,
        lr: float = 0.01,
        epochs: int = 100,
        seed: Optional[int] = None,
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

        # Attributes set during training
        self.params_: Optional[NDArray[np.floating]] = None
        self.loss_history_: list[float] = []
        self.status_: str = "not_fitted"

        # Internal caching
        self._qnode: Optional[Any] = None
        self._device: Optional[Any] = None
        self._n_qubits: int = encoding.n_qubits

    def _build_circuit(self) -> None:
        """Build the PennyLane QNode for the VQC.

        The circuit consists of:
        1. Encoding layer: Apply the quantum encoding to input x
        2. Variational layers: RY rotations + CNOT chain
        3. Measurement: Expectation of Z on qubit 0
        """
        import pennylane as qml

        n_qubits = self._n_qubits
        n_var_layers = self.n_var_layers

        # Create device
        self._device = qml.device("default.qubit", wires=n_qubits)

        # Store encoding reference for closure
        encoding = self.encoding

        @qml.qnode(self._device, interface="autograd")
        def circuit(x: NDArray[np.floating], params: NDArray[np.floating]) -> float:
            """VQC circuit: encoding + variational + measurement.

            Parameters
            ----------
            x : array, shape (n_features,)
                Input features.
            params : array, shape (n_var_layers, n_qubits)
                Variational parameters.

            Returns
            -------
            float
                Expectation value ⟨Z₀⟩ ∈ [-1, +1].
            """
            # Apply encoding
            # get_circuit returns a callable that applies gates when invoked
            encoding_circuit = encoding.get_circuit(x, backend="pennylane")
            encoding_circuit()  # Apply the encoding gates

            # Variational ansatz
            for layer in range(n_var_layers):
                # RY rotations on each qubit
                for i in range(n_qubits):
                    qml.RY(params[layer, i], wires=i)
                # Linear CNOT chain for entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Measurement: expectation of Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit
        logger.debug(
            "Built VQC circuit: n_qubits=%d, n_var_layers=%d",
            n_qubits, n_var_layers
        )

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.intp],
    ) -> "VQCClassifier":
        """Train the classifier on the given data.

        Uses binary cross-entropy loss with Adam optimizer (online SGD).
        Each epoch processes all samples individually in shuffled order,
        updating parameters after each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Training labels, values in {0, 1}.

        Returns
        -------
        self : VQCClassifier
            The fitted classifier.

        Notes
        -----
        Training data is shuffled at the start of each epoch using the
        seeded RNG to prevent order-dependent bias in the optimizer's
        internal state. The same seed always produces the same shuffle
        sequence, ensuring full reproducibility.

        Training may stop early if:
        - Loss becomes NaN
        - Loss exceeds _MAX_LOSS (indicates divergence)

        The status_ attribute indicates training outcome:
        - 'success': Training completed normally
        - 'diverged': Training stopped due to divergence
        """
        import pennylane as qml
        from pennylane import numpy as pnp

        if self._qnode is None:
            self._build_circuit()

        # Initialize parameters
        rng = np.random.default_rng(self.seed)
        self.params_ = pnp.array(
            rng.uniform(-np.pi, np.pi, size=(self.n_var_layers, self._n_qubits)),
            requires_grad=True,
        )

        # Setup optimizer
        opt = qml.AdamOptimizer(stepsize=self.lr)

        self.loss_history_ = []
        self.status_ = "success"

        n_samples = len(X)

        logger.debug(
            "Starting VQC training: epochs=%d, n_samples=%d, lr=%f",
            self.epochs, n_samples, self.lr
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            # Shuffle training data each epoch to eliminate order-dependent
            # bias in Adam's momentum/velocity estimates.
            indices = rng.permutation(n_samples)
            X_epoch = X[indices]
            y_epoch = y[indices]

            for xi, yi in zip(X_epoch, y_epoch):
                # Define cost function for this sample
                def cost(params: NDArray[np.floating]) -> float:
                    """Binary cross-entropy loss for a single sample."""
                    # Get prediction ⟨Z⟩ ∈ [-1, +1]
                    pred = self._qnode(xi, params)
                    # Map to probability p ∈ [0, 1]
                    p = (pred + 1) / 2
                    # Clip for numerical stability
                    p = pnp.clip(p, 1e-7, 1 - 1e-7)
                    # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
                    loss = -yi * pnp.log(p) - (1 - yi) * pnp.log(1 - p)
                    return loss

                # Optimization step
                self.params_, loss = opt.step_and_cost(cost, self.params_)
                epoch_loss += float(loss)

            # Average loss for this epoch
            avg_loss = epoch_loss / n_samples
            self.loss_history_.append(avg_loss)

            # Check for divergence
            if np.isnan(avg_loss) or avg_loss > _MAX_LOSS:
                logger.warning(
                    "Training diverged at epoch %d with loss=%f",
                    epoch, avg_loss
                )
                self.status_ = "diverged"
                break

            if (epoch + 1) % 20 == 0:
                logger.debug("Epoch %d/%d, loss=%.4f", epoch + 1, self.epochs, avg_loss)

        logger.debug(
            "Training completed: status=%s, final_loss=%.4f",
            self.status_,
            self.loss_history_[-1] if self.loss_history_ else float('nan')
        )

        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, 2)
            Class probabilities. Column 0 is P(class=0), column 1 is P(class=1).

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.params_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self._qnode is None:
            self._build_circuit()

        predictions = []
        for xi in X:
            # Get ⟨Z⟩ ∈ [-1, +1]
            exp_val = float(self._qnode(xi, self.params_))
            # Map to P(class=1) = (1 + ⟨Z⟩) / 2
            p1 = (1 + exp_val) / 2
            # Ensure valid probabilities
            p1 = np.clip(p1, 0.0, 1.0)
            predictions.append([1 - p1, p1])

        return np.array(predictions)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.intp]:
        """Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels in {0, 1}.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.intp)

    def score(self, X: NDArray[np.floating], y: NDArray[np.intp]) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.
        y : np.ndarray, shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
            Classification accuracy (proportion of correct predictions).

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def get_final_loss(self) -> Optional[float]:
        """Get the final training loss.

        Returns
        -------
        float or None
            Final training loss, or None if not fitted.
        """
        if not self.loss_history_:
            return None
        return self.loss_history_[-1]


def run_vqc_single_fold(
    encoding: Any,
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_train: NDArray[np.intp],
    y_test: NDArray[np.intp],
    n_var_layers: int = 2,
    lr: float = 0.01,
    epochs: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Run VQC training and evaluation for a single fold.

    Convenience function that encapsulates the full train-evaluate cycle
    and returns all relevant metrics in a dictionary.

    Parameters
    ----------
    encoding : BaseEncoding
        The quantum encoding to use.
    X_train, X_test : np.ndarray
        Training and test features.
    y_train, y_test : np.ndarray
        Training and test labels.
    n_var_layers : int, default=2
        Number of variational layers.
    lr : float, default=0.01
        Learning rate.
    epochs : int, default=100
        Number of training epochs.
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
        - 'final_loss': float or None
        - 'status': str ('success', 'diverged', or 'failed')
        - 'error': str (only if status='failed')
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    try:
        vqc = VQCClassifier(
            encoding=encoding,
            n_var_layers=n_var_layers,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        vqc.fit(X_train, y_train)

        y_pred_train = vqc.predict(X_train)
        y_pred_test = vqc.predict(X_test)

        result = {
            "train_accuracy": float(np.mean(y_pred_train == y_train)),
            "test_accuracy": float(np.mean(y_pred_test == y_test)),
            "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
            "final_loss": vqc.get_final_loss(),
            "status": vqc.status_,
            "n_epochs_trained": len(vqc.loss_history_),
        }

    except Exception as e:
        logger.error("VQC training failed: %s", str(e))
        result = {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "final_loss": None,
            "status": "failed",
            "error": str(e),
        }

    return result
