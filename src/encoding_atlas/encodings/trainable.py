"""Trainable encoding module.

This module implements TrainableEncoding, a parameterized quantum encoding that
combines classical data encoding with learnable (trainable) parameters. Unlike
fixed encodings that only depend on input data, trainable encodings have
additional variational parameters that can be optimized through training to
better adapt the encoding to specific tasks.

Trainable encoding is a cornerstone of variational quantum machine learning,
offering several key advantages:

1. **Task-Specific Adaptation**: Trainable parameters allow the encoding to be
   optimized for specific datasets and tasks, potentially learning better
   feature representations than fixed encodings.

2. **Enhanced Expressivity**: The interleaving of data-dependent and trainable
   layers increases the circuit's expressive power while maintaining favorable
   optimization properties.

3. **Noise Absorption**: During training on noisy quantum hardware, trainable
   parameters can partially absorb systematic errors, improving effective
   fidelity of the encoding.

4. **Flexibility**: Supports various initialization strategies, rotation gates,
   and entanglement patterns, allowing customization for different hardware
   and problem domains.

The encoding creates quantum states using:

    |ψ(x, θ)⟩ = [U_ent · U_trainable(θ) · U_data(x)]^L |0⟩^⊗n

where U_data(x) encodes the classical features, U_trainable(θ) applies
learnable rotations, and U_ent provides entanglement between qubits.

Mathematical Background
-----------------------
Trainable encoding bridges the gap between pure data encoding and variational
ansätze. The key insight is that by making some parameters learnable, the
encoding can:

1. **Amplify relevant features**: Learn to emphasize features important for
   the downstream task
2. **Create task-specific correlations**: The trainable layers can create
   entanglement patterns tailored to the data structure
3. **Implement implicit preprocessing**: Rather than manually scaling features,
   the trainable parameters can learn appropriate transformations

The circuit structure per layer consists of:
- **Data encoding sublayer**: R_α(x_i) for each qubit, encoding input features
- **Trainable sublayer**: R_β(θ_i) for each qubit, with learnable parameters
- **Entanglement sublayer**: CNOT gates following the connectivity pattern

The expressivity of this ansatz is characterized by the Fourier frequencies
that can be represented. With L layers and n qubits, the model can represent
functions with up to L·n Fourier frequencies in each dimension.

Note on Data Preprocessing
--------------------------
For trainable encoding, the interplay between data preprocessing and trainable
parameters is important:

- Raw data can often be used directly, as trainable parameters learn to scale
- For faster convergence, standardizing features (zero mean, unit variance) and
  then scaling to [0, π] or [-π/2, π/2] is recommended
- The trainable layers can learn to compensate for suboptimal preprocessing,
  but this may slow convergence

Use Cases
---------
TrainableEncoding is particularly suited for:

- **Quantum neural networks**: As the input encoding layer where adaptation
  to the specific dataset improves model performance
- **Variational quantum classifiers**: When fixed encodings underperform,
  trainable encoding can learn better feature mappings
- **Transfer learning**: Pre-trained encoding parameters can be transferred
  to related tasks, reducing training time
- **Noisy hardware**: Trainable parameters can absorb systematic errors,
  making the encoding more robust on NISQ devices
- **Feature selection**: Training reveals which features (and their
  combinations) are most relevant through the learned parameter magnitudes

Limitations
-----------
- **Training overhead**: Requires optimization of additional parameters,
  increasing computational cost and potential for local minima
- **Barren plateaus**: Very deep trainable circuits may exhibit vanishing
  gradients, though the interleaved structure helps mitigate this
- **Overfitting risk**: With too many trainable parameters relative to
  training data, the encoding may overfit to specific samples
- **Parameter initialization**: Careful initialization is important for
  avoiding poor local minima and ensuring trainability

Resource Analysis
-----------------
TrainableEncoding provides methods to analyze circuit resources:

- ``gate_count_breakdown()``: Get detailed gate counts by type
- ``resource_summary()``: Comprehensive resource analysis including depth,
  gate counts, trainability metrics, and parameter statistics
- ``get_trainable_parameters()``: Access current trainable parameter values
- ``set_trainable_parameters()``: Update trainable parameters (e.g., after
  optimization step)

These methods help understand resource requirements and facilitate integration
with classical optimization routines:

    >>> enc = TrainableEncoding(n_features=4, n_layers=3)
    >>> enc.gate_count_breakdown()
    {'data_rx': 0, 'data_ry': 12, 'data_rz': 0, 'trainable_rx': 0, ...}
    >>> summary = enc.resource_summary()
    >>> summary['n_trainable_parameters']
    12

Debugging
---------
This module supports Python's standard logging for debugging circuit
generation and parameter management. To enable debug output:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger('encoding_atlas.encodings.trainable').setLevel(logging.DEBUG)

Debug logs include:

- Initialization parameters and trainable parameter shapes
- Parameter initialization statistics (mean, std)
- Parameter updates via set_trainable_parameters()
- Circuit generation progress for each backend
- Input validation warnings for out-of-range values

References
----------
.. [1] Schuld, M., et al. (2021). "Effect of data encoding on the expressive
       power of variational quantum machine learning models." Physical Review A.
.. [2] Benedetti, M., et al. (2019). "Parameterized quantum circuits as machine
       learning models." Quantum Science and Technology.
.. [3] Pérez-Salinas, A., et al. (2020). "Data re-uploading for a universal
       quantum classifier." Quantum.
.. [4] McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
       network training landscapes." Nature Communications.
.. [5] Grant, E., et al. (2019). "Initialization strategy for addressing barren
       plateaus in parameterized quantum circuits." Quantum.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.properties import EncodingProperties
from encoding_atlas.core.types import BackendType, CircuitType

# =============================================================================
# Module-Level Logger
# =============================================================================
# Configure a logger for this module to enable optional debug output.
# By default, Python's logging is disabled (level WARNING), so these
# debug statements have no performance impact in production.
#
# To enable debug logging for this module:
#   import logging
#   logging.getLogger('encoding_atlas.encodings.trainable').setLevel(logging.DEBUG)
#   logging.basicConfig(level=logging.DEBUG)
#
# Or configure via your application's logging setup.
_logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ["TrainableEncoding"]

# =============================================================================
# Module-Level Constants
# =============================================================================

# Default number of trainable layers.
#
# The choice of 2 layers provides a good balance between:
#   - Expressivity: Multiple layers increase representation capacity
#   - Trainability: Fewer layers reduce barren plateau risk
#   - Gate count: Each layer adds O(n) gates
#
# Research suggests 2-4 layers often suffice for many QML tasks.
_DEFAULT_N_LAYERS: int = 2

# Threshold for warning about deep circuits.
#
# Deep trainable circuits may face trainability challenges due to barren
# plateaus (exponentially vanishing gradients). While the interleaved
# structure provides better gradient flow than purely random circuits,
# very deep configurations can still be problematic.
#
# At 8 layers with trainable parameters, the circuit becomes deep enough
# to warrant a warning. Users should consider:
#   - Whether the task requires such expressivity
#   - Using gradient-free optimization methods
#   - Layer-wise training strategies
#   - Smaller initialization scales
_DEEP_CIRCUIT_WARNING_THRESHOLD: int = 8

# Threshold for DEBUG logging about input values outside optimal range.
#
# Trainable encoding uses rotation gates which are 2π-periodic. While any
# input value produces valid results, the encoding works best with inputs
# in [0, 2π] or [-π, π] to ensure meaningful phase differences.
#
# Values beyond this threshold trigger a debug log to help users identify
# potential preprocessing issues.
_INPUT_RANGE_DEBUG_THRESHOLD: float = 4.0 * np.pi

# Maximum number of trainable parameters before issuing a warning.
#
# With very high parameter counts, optimization becomes challenging and
# overfitting risk increases. This threshold is a heuristic based on
# typical QML workloads.
_LARGE_PARAMETER_COUNT_WARNING_THRESHOLD: int = 100


# =============================================================================
# Type Definitions
# =============================================================================


class GateCountBreakdown(TypedDict):
    """Type definition for gate count breakdown dictionary.

    This TypedDict provides type safety and IDE autocompletion for the
    dictionary returned by ``gate_count_breakdown()``.

    Attributes
    ----------
    data_rx : int
        Number of RX gates for data encoding.
    data_ry : int
        Number of RY gates for data encoding.
    data_rz : int
        Number of RZ gates for data encoding.
    trainable_rx : int
        Number of RX gates for trainable parameters.
    trainable_ry : int
        Number of RY gates for trainable parameters.
    trainable_rz : int
        Number of RZ gates for trainable parameters.
    cnot_gates : int
        Number of CNOT entangling gates.
    total_single_qubit : int
        Total single-qubit gates.
    total_two_qubit : int
        Total two-qubit gates.
    total : int
        Total gate count.
    gates_per_layer : int
        Total gates per layer.
    """

    data_rx: int
    data_ry: int
    data_rz: int
    trainable_rx: int
    trainable_ry: int
    trainable_rz: int
    cnot_gates: int
    total_single_qubit: int
    total_two_qubit: int
    total: int
    gates_per_layer: int


class TrainableEncoding(BaseEncoding):
    """Trainable parameterized quantum encoding with learnable parameters.

    TrainableEncoding implements a quantum data encoding where classical input
    features are combined with trainable (learnable) parameters. This allows
    the encoding to be optimized for specific tasks through variational training,
    potentially learning better feature representations than fixed encodings.

    The circuit structure repeats L times:

        |0⟩ ─ R_d(x₀) ─ R_t(θ₀) ─╭────╮─ R_d(x₀) ─ R_t(θ₄) ─╭────╮─ ...
        |0⟩ ─ R_d(x₁) ─ R_t(θ₁) ─│CNOT│─ R_d(x₁) ─ R_t(θ₅) ─│CNOT│─ ...
        |0⟩ ─ R_d(x₂) ─ R_t(θ₂) ─│    │─ R_d(x₂) ─ R_t(θ₆) ─│    │─ ...
        |0⟩ ─ R_d(x₃) ─ R_t(θ₃) ─╰────╯─ R_d(x₃) ─ R_t(θ₇) ─╰────╯─ ...

    where R_d are data-encoding rotations, R_t are trainable rotations with
    learnable parameters θ, and CNOT gates provide entanglement.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode. Must be a positive integer.
        Each feature requires one qubit.
    n_layers : int, default=2
        Number of encoding-trainable-entanglement layer repetitions. Higher
        values increase expressivity but also circuit depth and training
        difficulty. Must be at least 1.
    data_rotation : {"X", "Y", "Z"}, default="Y"
        Axis of rotation for data-encoding gates. RY is commonly used as it
        creates real-valued amplitudes from the |0⟩ state.
    trainable_rotation : {"X", "Y", "Z"}, default="Y"
        Axis of rotation for trainable gates. Can be the same as or different
        from data_rotation. Using different axes increases expressivity.
    entanglement : {"linear", "circular", "full", "none"}, default="linear"
        Topology of CNOT entangling gates:

        - "linear": Nearest-neighbor connectivity (n-1 CNOT gates per layer)
        - "circular": Ring topology (n CNOT gates per layer)
        - "full": All-to-all connectivity (n(n-1)/2 CNOT gates per layer)
        - "none": No entanglement (useful for separable feature encoding)

    initialization : {"xavier", "he", "zeros", "random", "small_random"}, default="xavier"
        Strategy for initializing trainable parameters:

        - "xavier": Xavier/Glorot initialization, scale=√(2/(n_in+n_out))
        - "he": He initialization, scale=√(2/n_in)
        - "zeros": All parameters initialized to zero
        - "random": Uniform random in [-π, π]
        - "small_random": Uniform random in [-0.1, 0.1]

    seed : int or None, default=None
        Random seed for reproducible parameter initialization. If None, uses
        system entropy for random initialization.

    Attributes
    ----------
    n_layers : int
        Number of encoding layers.
    data_rotation : str
        Rotation axis for data encoding ("X", "Y", or "Z").
    trainable_rotation : str
        Rotation axis for trainable parameters ("X", "Y", or "Z").
    entanglement : str
        Entanglement topology.
    initialization : str
        Parameter initialization strategy used.
    n_trainable_parameters : int
        Total number of trainable parameters in the circuit.

    Examples
    --------
    Create a basic trainable encoding:

    >>> from encoding_atlas import TrainableEncoding
    >>> import numpy as np
    >>> enc = TrainableEncoding(n_features=4)
    >>> enc.n_qubits
    4
    >>> enc.n_layers
    2
    >>> enc.n_trainable_parameters
    8

    Generate a PennyLane circuit with current parameters:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4])
    >>> circuit = enc.get_circuit(x, backend='pennylane')
    >>> callable(circuit)
    True

    Access and modify trainable parameters:

    >>> params = enc.get_trainable_parameters()
    >>> params.shape
    (2, 4)
    >>> new_params = np.random.randn(2, 4) * 0.1
    >>> enc.set_trainable_parameters(new_params)

    Use different rotation gates for data and trainable layers:

    >>> enc_mixed = TrainableEncoding(
    ...     n_features=4,
    ...     data_rotation='Y',
    ...     trainable_rotation='Z'
    ... )

    Reproducible initialization with seed:

    >>> enc1 = TrainableEncoding(n_features=4, seed=42)
    >>> enc2 = TrainableEncoding(n_features=4, seed=42)
    >>> np.allclose(enc1.get_trainable_parameters(), enc2.get_trainable_parameters())
    True

    Full entanglement for maximum expressivity:

    >>> enc_full = TrainableEncoding(n_features=4, entanglement='full')
    >>> enc_full.properties.two_qubit_gates
    12

    References
    ----------
    .. [1] Schuld, M., et al. (2021). "Effect of data encoding on the
           expressive power of variational quantum machine learning models."
    .. [2] Benedetti, M., et al. (2019). "Parameterized quantum circuits as
           machine learning models." Quantum Science and Technology.

    See Also
    --------
    DataReuploading : Fixed data re-uploading without trainable parameters.
    HardwareEfficientEncoding : Hardware-optimized encoding without trainable
        parameters.
    AngleEncoding : Simple angle encoding without entanglement or trainability.

    Notes
    -----
    **Trainable vs Data Parameters**: This encoding distinguishes between
    data parameters (determined by input x) and trainable parameters (θ,
    optimized during training). The data parameters encode the input features,
    while trainable parameters allow the circuit to learn task-specific
    transformations.

    **Initialization Importance**: The initialization strategy significantly
    affects training dynamics. Xavier initialization is recommended as a
    starting point, but for specific problems, experimenting with different
    strategies may improve results.

    **Gradient Computation**: Trainable parameters support gradient computation
    via the parameter-shift rule, enabling integration with gradient-based
    optimizers in frameworks like PennyLane, TensorFlow Quantum, or manual
    implementations.

    **Thread Safety**: The encoding object stores trainable parameters as
    instance state. While circuit generation is thread-safe for reading,
    concurrent calls to ``set_trainable_parameters()`` should be externally
    synchronized.
    """

    # Valid options for rotation gates, entanglement, and initialization
    _VALID_ROTATIONS: frozenset[str] = frozenset({"X", "Y", "Z"})
    _VALID_ENTANGLEMENTS: frozenset[str] = frozenset(
        {"linear", "circular", "full", "none"}
    )
    _VALID_INITIALIZATIONS: frozenset[str] = frozenset(
        {"xavier", "he", "zeros", "random", "small_random"}
    )

    __slots__ = (
        "n_layers",
        "data_rotation",
        "trainable_rotation",
        "entanglement",
        "initialization",
        "_trainable_params",
        "_entanglement_pairs",
        "_seed",
        "_rng",
    )

    def __init__(
        self,
        n_features: int,
        n_layers: int = _DEFAULT_N_LAYERS,
        data_rotation: Literal["X", "Y", "Z"] = "Y",
        trainable_rotation: Literal["X", "Y", "Z"] = "Y",
        entanglement: Literal["linear", "circular", "full", "none"] = "linear",
        initialization: Literal[
            "xavier", "he", "zeros", "random", "small_random"
        ] = "xavier",
        seed: int | None = None,
    ) -> None:
        """Initialize the trainable encoding.

        Parameters
        ----------
        n_features : int
            Number of classical features to encode.
        n_layers : int, default=2
            Number of encoding layer repetitions.
        data_rotation : {"X", "Y", "Z"}, default="Y"
            Rotation axis for data-encoding gates.
        trainable_rotation : {"X", "Y", "Z"}, default="Y"
            Rotation axis for trainable gates.
        entanglement : {"linear", "circular", "full", "none"}, default="linear"
            Topology for entangling CNOT gates.
        initialization : {"xavier", "he", "zeros", "random", "small_random"},
            default="xavier"
            Strategy for initializing trainable parameters.
        seed : int or None, default=None
            Random seed for reproducible initialization.

        Raises
        ------
        ValueError
            If n_layers is less than 1.
        ValueError
            If data_rotation or trainable_rotation is not one of "X", "Y", "Z".
        ValueError
            If entanglement is not one of "linear", "circular", "full", "none".
        ValueError
            If initialization is not a valid strategy name.
        ValueError
            If n_features is less than 1 (raised by parent class).

        Warns
        -----
        UserWarning
            If n_layers exceeds the deep circuit threshold (default 8),
            as very deep circuits may face trainability challenges.
        UserWarning
            If total trainable parameters exceeds threshold, indicating
            potential overfitting risk.
        """
        # =====================================================================
        # PARAMETER VALIDATION
        # =====================================================================
        # Validate n_layers before calling parent constructor to fail fast.
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError(f"n_layers must be a positive integer, got {n_layers!r}")

        # Validate rotation axes
        if data_rotation not in self._VALID_ROTATIONS:
            raise ValueError(
                f"data_rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got {data_rotation!r}"
            )
        if trainable_rotation not in self._VALID_ROTATIONS:
            raise ValueError(
                f"trainable_rotation must be one of {sorted(self._VALID_ROTATIONS)}, "
                f"got {trainable_rotation!r}"
            )

        # Validate entanglement
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENTS)}, "
                f"got {entanglement!r}"
            )

        # Validate initialization
        if initialization not in self._VALID_INITIALIZATIONS:
            raise ValueError(
                f"initialization must be one of {sorted(self._VALID_INITIALIZATIONS)}, "
                f"got {initialization!r}"
            )

        # Validate seed if provided
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        super().__init__(
            n_features,
            n_layers=n_layers,
            data_rotation=data_rotation,
            trainable_rotation=trainable_rotation,
            entanglement=entanglement,
            initialization=initialization,
            seed=seed,
        )

        self.n_layers: int = n_layers
        self.data_rotation: Literal["X", "Y", "Z"] = data_rotation
        self.trainable_rotation: Literal["X", "Y", "Z"] = trainable_rotation
        self.entanglement: Literal["linear", "circular", "full", "none"] = entanglement
        self.initialization: str = initialization
        self._seed: int | None = seed

        # Initialize RNG for reproducibility
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Cache entanglement pairs at initialization for performance
        self._entanglement_pairs: list[tuple[int, int]] = (
            self._compute_entanglement_pairs()
        )

        # Initialize trainable parameters
        self._trainable_params: NDArray[np.floating[Any]] = (
            self._initialize_parameters()
        )

        # Log initialization
        _logger.debug(
            "TrainableEncoding initialized: n_features=%d, n_layers=%d, "
            "data_rotation=%r, trainable_rotation=%r, entanglement=%r, "
            "initialization=%r, n_trainable_params=%d, seed=%s",
            self.n_features,
            self.n_layers,
            self.data_rotation,
            self.trainable_rotation,
            self.entanglement,
            self.initialization,
            self.n_trainable_parameters,
            self._seed,
        )

        # =====================================================================
        # WARNINGS FOR POTENTIAL ISSUES
        # =====================================================================
        # Warn about deep circuits
        if n_layers > _DEEP_CIRCUIT_WARNING_THRESHOLD:
            total_params = self.n_trainable_parameters
            warnings.warn(
                f"TrainableEncoding with {n_layers} layers creates a deep circuit "
                f"with {total_params} trainable parameters. Very deep trainable "
                f"circuits may face trainability challenges due to barren plateaus. "
                f"Consider: (1) reducing n_layers if task permits, "
                f"(2) using layer-wise training, or "
                f"(3) specialized initialization strategies.",
                UserWarning,
                stacklevel=2,
            )
            _logger.warning(
                "Deep trainable circuit: n_layers=%d exceeds threshold=%d, "
                "n_trainable_params=%d",
                n_layers,
                _DEEP_CIRCUIT_WARNING_THRESHOLD,
                total_params,
            )

        # Warn about large parameter counts
        total_params = self.n_trainable_parameters
        if total_params > _LARGE_PARAMETER_COUNT_WARNING_THRESHOLD:
            warnings.warn(
                f"TrainableEncoding has {total_params} trainable parameters, "
                f"which may lead to overfitting with small training sets. "
                f"Consider reducing n_layers or n_features if experiencing "
                f"poor generalization.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit.

        For trainable encoding, each feature is encoded on a dedicated qubit,
        so n_qubits equals n_features.

        Returns
        -------
        int
            Number of qubits, equal to n_features.
        """
        return self.n_features

    @property
    def depth(self) -> int:
        """Circuit depth of the encoding.

        Each layer consists of:
        - Data encoding sublayer (depth 1)
        - Trainable sublayer (depth 1)
        - Entanglement sublayer (depth depends on pattern)

        Returns
        -------
        int
            Exact circuit depth based on structure.
        """
        # Each layer: data encoding (1) + trainable (1) + entanglement (n_qubits-1 for linear)
        if self.entanglement == "none":
            entangling_depth = 0
        elif self.entanglement == "linear":
            entangling_depth = max(0, self.n_qubits - 1)
        elif self.entanglement == "circular":
            entangling_depth = (
                self.n_qubits if self.n_qubits > 2 else max(0, self.n_qubits - 1)
            )
        else:  # full
            # Full entanglement: all pairs, can be parallelized to some extent
            # Depth is approximately n_qubits - 1 for most efficient layout
            entangling_depth = max(0, self.n_qubits - 1)

        return self.n_layers * (2 + entangling_depth)

    @property
    def n_trainable_parameters(self) -> int:
        """Total number of trainable parameters in the circuit.

        Returns
        -------
        int
            Number of learnable parameters (n_layers × n_features).
        """
        return self.n_layers * self.n_features

    def _compute_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Compute qubit pairs for CNOT entanglement based on topology.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for CNOT gates.
        """
        n = self.n_features

        if self.entanglement == "none":
            pairs = []
        elif self.entanglement == "linear":
            pairs = [(i, i + 1) for i in range(n - 1)]
        elif self.entanglement == "circular":
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
        else:  # full
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        _logger.debug(
            "Computed entanglement pairs for %s topology: %d pairs",
            self.entanglement,
            len(pairs),
        )
        return pairs

    def _initialize_parameters(self) -> NDArray[np.floating[Any]]:
        """Initialize trainable parameters according to the specified strategy.

        Returns
        -------
        NDArray
            Trainable parameters of shape (n_layers, n_features).
        """
        shape = (self.n_layers, self.n_features)

        if self.initialization == "xavier":
            # Xavier/Glorot initialization
            # scale = sqrt(2 / (fan_in + fan_out))
            # For rotation parameters, we treat fan_in = fan_out = n_features
            scale = np.sqrt(2.0 / (2 * self.n_features))
            params = self._rng.normal(0, scale, shape)

        elif self.initialization == "he":
            # He initialization
            # scale = sqrt(2 / fan_in)
            scale = np.sqrt(2.0 / self.n_features)
            params = self._rng.normal(0, scale, shape)

        elif self.initialization == "zeros":
            # Zero initialization
            params = np.zeros(shape, dtype=np.float64)

        elif self.initialization == "random":
            # Uniform random in [-π, π]
            params = self._rng.uniform(-np.pi, np.pi, shape)

        else:  # small_random
            # Small random values in [-0.1, 0.1]
            params = self._rng.uniform(-0.1, 0.1, shape)

        _logger.debug(
            "Initialized parameters: shape=%s, strategy=%s, mean=%.4f, std=%.4f",
            shape,
            self.initialization,
            float(np.mean(params)),
            float(np.std(params)),
        )

        return params.astype(np.float64)

    def get_trainable_parameters(self) -> NDArray[np.floating[Any]]:
        """Get a copy of the current trainable parameters.

        Returns
        -------
        NDArray[np.floating]
            Copy of trainable parameters with shape (n_layers, n_features).

        Notes
        -----
        Returns a copy to prevent external modification of internal state.
        Use ``set_trainable_parameters()`` to update parameters.

        Examples
        --------
        >>> enc = TrainableEncoding(n_features=4, n_layers=2)
        >>> params = enc.get_trainable_parameters()
        >>> params.shape
        (2, 4)
        """
        return self._trainable_params.copy()

    def set_trainable_parameters(
        self,
        params: ArrayLike,
        *,
        validate: bool = True,
    ) -> None:
        """Set the trainable parameters to new values.

        This method is typically called after an optimization step to update
        the learned parameters.

        Parameters
        ----------
        params : ArrayLike
            New trainable parameters. Must have shape (n_layers, n_features)
            or be flattenable to that shape.
        validate : bool, default=True
            If True, validate that params has correct shape and no NaN/Inf
            values. Set to False for performance in hot optimization loops
            where params are guaranteed valid.

        Raises
        ------
        ValueError
            If params has wrong shape (when validate=True).
        ValueError
            If params contains NaN or Inf values (when validate=True).

        Notes
        -----
        This method invalidates any cached properties that depend on parameter
        values. After calling this method, ``_compute_properties()`` will be
        re-evaluated on next access if it depends on parameters.

        Thread Safety: This method modifies instance state. Concurrent calls
        should be externally synchronized.

        Examples
        --------
        >>> enc = TrainableEncoding(n_features=4, n_layers=2)
        >>> new_params = np.random.randn(2, 4) * 0.1
        >>> enc.set_trainable_parameters(new_params)
        """
        params_array = np.asarray(params, dtype=np.float64)

        expected_shape = (self.n_layers, self.n_features)

        # Attempt to reshape flat arrays
        if params_array.ndim == 1:
            if params_array.size == self.n_layers * self.n_features:
                params_array = params_array.reshape(expected_shape)
            elif validate:
                raise ValueError(
                    f"Cannot reshape flat array of size {params_array.size} to "
                    f"expected shape {expected_shape}"
                )

        if validate:
            if params_array.shape != expected_shape:
                raise ValueError(
                    f"Expected parameters with shape {expected_shape}, "
                    f"got {params_array.shape}"
                )
            if np.any(np.isnan(params_array)) or np.any(np.isinf(params_array)):
                raise ValueError("Parameters contain NaN or infinite values")

        self._trainable_params = params_array.copy()

        # Invalidate cached properties
        with self._properties_lock:
            self._properties = None

        _logger.debug(
            "Updated trainable parameters: mean=%.4f, std=%.4f",
            float(np.mean(params_array)),
            float(np.std(params_array)),
        )

    def reset_parameters(self, seed: int | None = None) -> None:
        """Reset trainable parameters to fresh initialization.

        This is useful for restarting training from scratch or trying
        different random initializations.

        Parameters
        ----------
        seed : int or None, default=None
            New random seed to use. If None, uses the original seed specified
            during construction (or system entropy if none was specified).

        Examples
        --------
        >>> enc = TrainableEncoding(n_features=4, seed=42)
        >>> orig_params = enc.get_trainable_parameters().copy()
        >>> enc.set_trainable_parameters(np.zeros((2, 4)))  # Modify
        >>> enc.reset_parameters()  # Reset to original
        >>> np.allclose(orig_params, enc.get_trainable_parameters())
        True
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        else:
            # Reset to original seed or new random state
            self._rng = np.random.default_rng(self._seed)

        self._trainable_params = self._initialize_parameters()

        # Invalidate cached properties
        with self._properties_lock:
            self._properties = None

        _logger.debug("Reset trainable parameters with seed=%s", seed or self._seed)

    def get_entanglement_pairs(self) -> list[tuple[int, int]]:
        """Get the qubit pairs used for CNOT entanglement.

        Returns
        -------
        list[tuple[int, int]]
            List of (control, target) qubit pairs for CNOT gates.

        Examples
        --------
        >>> enc = TrainableEncoding(n_features=4, entanglement='linear')
        >>> enc.get_entanglement_pairs()
        [(0, 1), (1, 2), (2, 3)]

        >>> enc_full = TrainableEncoding(n_features=4, entanglement='full')
        >>> enc_full.get_entanglement_pairs()
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        """
        return list(self._entanglement_pairs)

    # =========================================================================
    # Resource Analysis Methods
    # =========================================================================

    def gate_count_breakdown(self) -> GateCountBreakdown:
        """Get a detailed breakdown of gate counts by type.

        Returns
        -------
        GateCountBreakdown
            Dictionary with gate counts by type.

        Examples
        --------
        >>> enc = TrainableEncoding(n_features=4, n_layers=2)
        >>> breakdown = enc.gate_count_breakdown()
        >>> breakdown['total']
        22
        """
        n = self.n_features
        n_pairs = len(self._entanglement_pairs)

        # Per-layer counts
        data_gates_per_layer = n
        trainable_gates_per_layer = n
        cnot_per_layer = n_pairs

        # Total counts
        data_gates = self.n_layers * data_gates_per_layer
        trainable_gates = self.n_layers * trainable_gates_per_layer
        cnot_gates = self.n_layers * cnot_per_layer

        # Break down by rotation type
        data_rx = data_gates if self.data_rotation == "X" else 0
        data_ry = data_gates if self.data_rotation == "Y" else 0
        data_rz = data_gates if self.data_rotation == "Z" else 0

        trainable_rx = trainable_gates if self.trainable_rotation == "X" else 0
        trainable_ry = trainable_gates if self.trainable_rotation == "Y" else 0
        trainable_rz = trainable_gates if self.trainable_rotation == "Z" else 0

        total_single_qubit = data_gates + trainable_gates
        total = total_single_qubit + cnot_gates

        _logger.debug(
            "Gate breakdown: data=%d, trainable=%d, CNOT=%d, total=%d",
            data_gates,
            trainable_gates,
            cnot_gates,
            total,
        )

        return GateCountBreakdown(
            data_rx=data_rx,
            data_ry=data_ry,
            data_rz=data_rz,
            trainable_rx=trainable_rx,
            trainable_ry=trainable_ry,
            trainable_rz=trainable_rz,
            cnot_gates=cnot_gates,
            total_single_qubit=total_single_qubit,
            total_two_qubit=cnot_gates,
            total=total,
            gates_per_layer=data_gates_per_layer
            + trainable_gates_per_layer
            + cnot_per_layer,
        )

    def resource_summary(self) -> dict[str, Any]:
        """Generate a comprehensive resource summary for this encoding.

        Returns
        -------
        dict[str, Any]
            Dictionary containing circuit structure, gate counts, encoding
            characteristics, hardware requirements, and training-related
            statistics.

        Examples
        --------
        >>> enc = TrainableEncoding(n_features=4, n_layers=2)
        >>> summary = enc.resource_summary()
        >>> summary['n_trainable_parameters']
        8
        """
        gate_counts = self.gate_count_breakdown()
        props = self.properties
        params = self.get_trainable_parameters()

        # Build recommendations
        recommendations: list[str] = []

        if self.n_layers > _DEEP_CIRCUIT_WARNING_THRESHOLD:
            recommendations.append(
                f"Deep circuit ({self.n_layers} layers): Consider layer-wise "
                f"training or gradient-free optimization."
            )

        if self.entanglement == "none":
            recommendations.append(
                "No entanglement: Circuit produces separable states only. "
                "Consider adding entanglement for richer representations."
            )

        if self.entanglement == "full" and self.n_features > 10:
            recommendations.append(
                f"Full entanglement with {self.n_features} qubits: Gate count "
                f"scales as O(n²). Consider linear or circular for efficiency."
            )

        if self.n_trainable_parameters > _LARGE_PARAMETER_COUNT_WARNING_THRESHOLD:
            recommendations.append(
                f"High parameter count ({self.n_trainable_parameters}): "
                f"Monitor for overfitting on small datasets."
            )

        if not recommendations:
            recommendations.append("Configuration looks good for typical QML tasks.")

        return {
            # Circuit structure
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "n_layers": self.n_layers,
            "depth": self.depth,
            # Gate counts
            "gate_counts": gate_counts,
            # Encoding configuration
            "data_rotation": self.data_rotation,
            "trainable_rotation": self.trainable_rotation,
            "entanglement": self.entanglement,
            "initialization": self.initialization,
            # Trainable parameters
            "n_trainable_parameters": self.n_trainable_parameters,
            "parameter_statistics": {
                "mean": float(np.mean(params)),
                "std": float(np.std(params)),
                "min": float(np.min(params)),
                "max": float(np.max(params)),
            },
            # Encoding characteristics
            "is_entangling": self.entanglement != "none" and self.n_qubits > 1,
            "simulability": props.simulability,
            "trainability_estimate": props.trainability_estimate,
            # Hardware requirements
            "hardware_requirements": {
                "connectivity": (
                    self.entanglement if self.entanglement != "none" else "any"
                ),
                "native_gates": [
                    f"R{self.data_rotation}",
                    f"R{self.trainable_rotation}",
                    "CNOT",
                ],
                "min_qubit_count": self.n_qubits,
            },
            # Entanglement details
            "n_entanglement_pairs": len(self._entanglement_pairs),
            "entanglement_pairs": self.get_entanglement_pairs(),
            # Recommendations
            "recommendations": recommendations,
        }

    # =========================================================================
    # Circuit Generation
    # =========================================================================

    def _get_circuit_from_validated(
        self,
        x: NDArray[np.floating[Any]],
        backend: BackendType,
    ) -> CircuitType:
        """Generate circuit from pre-validated input.

        Encoding-specific seam called by the inherited ``get_circuit``/
        ``get_circuits`` template methods. Emits a debug log when input
        values fall outside the typical [-2π, 2π] range and dispatches to
        the backend-specific implementation.

        Accepts a 1D array (the normal case) or a 2D single-sample
        ``(1, n_features)`` array for robustness when called directly.
        """
        # Defensive 2D-to-1D handling for direct callers; the inherited
        # template methods already pass 1D input.
        if x.ndim == 2:
            x = x[0]

        if _logger.isEnabledFor(logging.DEBUG):
            x_min, x_max = float(x.min()), float(x.max())
            if (
                abs(x_min) > _INPUT_RANGE_DEBUG_THRESHOLD
                or abs(x_max) > _INPUT_RANGE_DEBUG_THRESHOLD
            ):
                _logger.debug(
                    "Input values [%.3g, %.3g] are outside typical range "
                    "[-2π, 2π]. Consider scaling features.",
                    x_min,
                    x_max,
                )

        if backend == "pennylane":
            return self._to_pennylane(x)
        elif backend == "qiskit":
            return self._to_qiskit(x)
        elif backend == "cirq":
            return self._to_cirq(x)
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Supported backends: 'pennylane', 'qiskit', 'cirq'"
            )

    def _to_pennylane(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate PennyLane circuit function.

        Parameters
        ----------
        x : NDArray
            Validated input features of shape (n_features,).

        Returns
        -------
        callable
            Function that applies the encoding gates when called.
        """
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required for the 'pennylane' backend. "
                "Install it with: pip install pennylane"
            ) from e

        n_qubits = self.n_qubits
        pairs = self._entanglement_pairs
        params = self._trainable_params

        # Select rotation gate factories
        data_rot_gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[self.data_rotation]
        trainable_rot_gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[
            self.trainable_rotation
        ]

        def circuit() -> None:
            """Apply the trainable encoding gates."""
            for layer in range(self.n_layers):
                # Data encoding sublayer
                for i in range(n_qubits):
                    data_rot_gate(x[i], wires=i)

                # Trainable sublayer
                for i in range(n_qubits):
                    trainable_rot_gate(params[layer, i], wires=i)

                # Entanglement sublayer
                for i, j in pairs:
                    qml.CNOT(wires=[i, j])

        return circuit

    def _to_qiskit(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Qiskit QuantumCircuit.

        Parameters
        ----------
        x : NDArray
            Validated input features of shape (n_features,).

        Returns
        -------
        QuantumCircuit
            Qiskit quantum circuit implementing the encoding.
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError(
                "Qiskit is required for the 'qiskit' backend. "
                "Install it with: pip install qiskit"
            ) from e

        qc = QuantumCircuit(self.n_qubits, name="TrainableEncoding")
        pairs = self._entanglement_pairs
        params = self._trainable_params

        # Map rotation types to Qiskit methods
        data_rot_method = {"X": "rx", "Y": "ry", "Z": "rz"}[self.data_rotation]
        trainable_rot_method = {"X": "rx", "Y": "ry", "Z": "rz"}[
            self.trainable_rotation
        ]

        for layer in range(self.n_layers):
            # Data encoding sublayer
            for i in range(self.n_qubits):
                getattr(qc, data_rot_method)(float(x[i]), i)

            # Trainable sublayer
            for i in range(self.n_qubits):
                getattr(qc, trainable_rot_method)(float(params[layer, i]), i)

            # Entanglement sublayer
            for i, j in pairs:
                qc.cx(i, j)

        return qc

    def _to_cirq(self, x: NDArray[np.floating[Any]]) -> Any:
        """Generate Cirq Circuit.

        Parameters
        ----------
        x : NDArray
            Validated input features of shape (n_features,).

        Returns
        -------
        cirq.Circuit
            Cirq circuit implementing the trainable encoding.
        """
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required for the 'cirq' backend. "
                "Install it with: pip install cirq-core"
            ) from e

        qubits = cirq.LineQubit.range(self.n_qubits)
        pairs = self._entanglement_pairs
        params = self._trainable_params
        circuit = cirq.Circuit()

        # Map rotation types to Cirq gates
        rotation_gates = {"X": cirq.rx, "Y": cirq.ry, "Z": cirq.rz}
        data_rot = rotation_gates[self.data_rotation]
        trainable_rot = rotation_gates[self.trainable_rotation]

        for layer in range(self.n_layers):
            # Data encoding sublayer
            circuit.append(
                [data_rot(float(x[i]))(qubits[i]) for i in range(self.n_qubits)]
            )

            # Trainable sublayer
            circuit.append(
                [
                    trainable_rot(float(params[layer, i]))(qubits[i])
                    for i in range(self.n_qubits)
                ]
            )

            # Entanglement sublayer
            if pairs:
                circuit.append([cirq.CNOT(qubits[i], qubits[j]) for i, j in pairs])

        return circuit

    def _compute_properties(self) -> EncodingProperties:
        """Compute theoretical properties of this encoding.

        Returns
        -------
        EncodingProperties
            Computed properties including qubit count, depth, gate counts,
            and encoding characteristics.
        """
        n = self.n_qubits
        n_pairs = len(self._entanglement_pairs)

        # Gates per layer
        data_gates = self.n_layers * n
        trainable_gates = self.n_layers * n
        cnot_gates = self.n_layers * n_pairs

        total_single_qubit = data_gates + trainable_gates
        total_gates = total_single_qubit + cnot_gates

        # Trainability estimate
        # Deeper circuits have lower trainability due to barren plateaus
        base_trainability = 0.85
        depth_penalty = 0.03 * self.n_layers
        trainability = max(0.4, base_trainability - depth_penalty)

        is_entangling = self.entanglement != "none" and n > 1

        return EncodingProperties(
            n_qubits=n,
            depth=self.depth,
            gate_count=total_gates,
            single_qubit_gates=total_single_qubit,
            two_qubit_gates=cnot_gates,
            parameter_count=data_gates + self.n_trainable_parameters,
            is_entangling=is_entangling,
            simulability="not_simulable" if is_entangling else "simulable",
            trainability_estimate=trainability,
            notes=(
                f"Trainable encoding with {self.n_layers} layers, "
                f"{self.n_trainable_parameters} trainable parameters, "
                f"{self.data_rotation} data rotation, "
                f"{self.trainable_rotation} trainable rotation, "
                f"and {self.entanglement} entanglement."
            ),
        )

    # =========================================================================
    # Serialization Support
    # =========================================================================

    def __getstate__(self) -> dict[str, Any]:
        """Prepare instance state for pickling.

        Returns
        -------
        dict[str, Any]
            Serializable state dictionary.
        """
        state = super().__getstate__()

        # Add trainable-specific state
        state["slots"]["_trainable_params"] = self._trainable_params.copy()
        state["slots"]["_entanglement_pairs"] = self._entanglement_pairs.copy()
        state["slots"]["_seed"] = self._seed
        # Note: _rng is not picklable, will be recreated

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state after unpickling.

        Parameters
        ----------
        state : dict[str, Any]
            State dictionary from __getstate__.
        """
        super().__setstate__(state)

        # Recreate RNG from seed
        self._rng = np.random.default_rng(self._seed)

    def __repr__(self) -> str:
        """Return detailed string representation.

        Returns
        -------
        str
            String representation showing all configuration parameters.
        """
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"n_layers={self.n_layers}, "
            f"data_rotation={self.data_rotation!r}, "
            f"trainable_rotation={self.trainable_rotation!r}, "
            f"entanglement={self.entanglement!r}, "
            f"initialization={self.initialization!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another encoding.

        Two TrainableEncodings are equal if they have the same configuration
        AND the same trainable parameters.
        """
        if not isinstance(other, TrainableEncoding):
            return NotImplemented
        return (
            self.n_features == other.n_features
            and self.n_layers == other.n_layers
            and self.data_rotation == other.data_rotation
            and self.trainable_rotation == other.trainable_rotation
            and self.entanglement == other.entanglement
            and self.initialization == other.initialization
            and np.allclose(self._trainable_params, other._trainable_params)
        )

    def __hash__(self) -> int:
        """Compute hash for the encoding.

        Note: Hash does not include trainable parameters as they can change.
        """
        return hash(
            (
                self.__class__.__name__,
                self.n_features,
                self.n_layers,
                self.data_rotation,
                self.trainable_rotation,
                self.entanglement,
                self.initialization,
            )
        )

    def copy(self) -> TrainableEncoding:
        """Create a deep copy of this encoding.

        Returns
        -------
        TrainableEncoding
            New encoding with copied parameters.
        """
        new_enc = TrainableEncoding(
            n_features=self.n_features,
            n_layers=self.n_layers,
            data_rotation=self.data_rotation,
            trainable_rotation=self.trainable_rotation,
            entanglement=self.entanglement,
            initialization=self.initialization,
            seed=None,  # Don't use same seed, copy params instead
        )
        new_enc.set_trainable_parameters(self._trainable_params.copy())
        return new_enc
