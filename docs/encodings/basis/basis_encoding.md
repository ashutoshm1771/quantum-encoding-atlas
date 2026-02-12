# Basis Encoding

The simplest quantum data encoding — a direct one-to-one map from
**classical bits** to **computational basis states**.

```
 ╔══════════════════════════════════════════════════════════════════════╗
 ║                                                                      ║
 ║   |ψ(x)⟩  =  |x₀ x₁ ... xₙ₋₁⟩       where  xᵢ ∈ {0, 1}          ║
 ║                                                                      ║
 ║   "n binary features  →  n qubits  →  one basis state"              ║
 ║                                                                      ║
 ╚══════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

Each classical bit is stored in a **dedicated qubit**. A `0` leaves the
qubit in |0⟩; a `1` flips it to |1⟩ with a Pauli-X gate. The result is
always a single computational basis state — no superposition, no
entanglement.

```
  Classical Bits  (4 features)              Quantum State  (4 qubits)
  ┌──────────────────────────┐              ┌──────────────────────────┐
  │                          │              │                          │
  │  x₀ = 1                 │              │   |1⟩  ← X applied      │
  │  x₁ = 0                 │   Encode     │   |0⟩  ← untouched      │
  │  x₂ = 1   ──────────────────────────►  │   |1⟩  ← X applied      │
  │  x₃ = 0                 │              │   |0⟩  ← untouched      │
  │                          │              │                          │
  └──────────────────────────┘              └──────────────────────────┘
       bit string [1, 0, 1, 0]                  state  |1010⟩
```

> Think of it as writing binary data onto a quantum register — the
> quantum version of loading bits into a classical register.

---

## 2. The Mathematics

### State Definition

Given a binary vector **x** = (x₀, x₁, ..., xₙ₋₁)  with  xᵢ ∈ {0, 1}:

```
  |ψ(x)⟩  =  X^{x₀} ⊗ X^{x₁} ⊗ ... ⊗ X^{xₙ₋₁}  |0⟩^⊗n
```

where `X^{xᵢ}` means:

```
  xᵢ = 0  →  X⁰ = I  (identity, qubit stays |0⟩)
  xᵢ = 1  →  X¹ = X  (Pauli-X flip, qubit becomes |1⟩)
```

### The Pauli-X Gate

```
       ┌     ┐
  X =  │ 0 1 │     X|0⟩ = |1⟩
       │ 1 0 │     X|1⟩ = |0⟩
       └     ┘

  It is the quantum analogue of a classical NOT gate.
```

### Resulting Quantum State

The output is always a **single** computational basis state with
coefficient 1. No superposition is created:

```
  Input: x = [1, 0, 1, 1]

  |ψ⟩ = |1011⟩  =  |1⟩ ⊗ |0⟩ ⊗ |1⟩ ⊗ |1⟩

  Statevector:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                                                   ↑
                                            index 11 (binary 1011)

  Measurement: |1011⟩ with probability 1.000  (deterministic)
```

---

## 3. Circuit Structure

The circuit is the simplest possible — parallel X gates on selected
qubits, all in a **single layer** (depth = 1):

```
  x = [1, 0, 1, 0]

  q₀: |0⟩ ───[X]─── |1⟩       ← x₀ = 1, flip
  q₁: |0⟩ ─────────── |0⟩       ← x₁ = 0, no gate
  q₂: |0⟩ ───[X]─── |1⟩       ← x₂ = 1, flip
  q₃: |0⟩ ─────────── |0⟩       ← x₃ = 0, no gate

  Result: |1010⟩
  Depth:  1
  Gates:  2 (only the X gates that fire)
```

For the all-ones case (worst case):

```
  x = [1, 1, 1, 1]

  q₀: |0⟩ ───[X]─── |1⟩
  q₁: |0⟩ ───[X]─── |1⟩
  q₂: |0⟩ ───[X]─── |1⟩
  q₃: |0⟩ ───[X]─── |1⟩

  Result: |1111⟩
  Depth:  1
  Gates:  4 (maximum for n=4)
```

For the all-zeros case (best case):

```
  x = [0, 0, 0, 0]

  q₀: |0⟩ ─────────── |0⟩
  q₁: |0⟩ ─────────── |0⟩
  q₂: |0⟩ ─────────── |0⟩
  q₃: |0⟩ ─────────── |0⟩

  Result: |0000⟩
  Depth:  1 (the `depth` property always returns 1)
  Gates:  0 (no gates needed!)
```

> The gate count is **data-dependent**: it equals the number of 1s in
> the input. For sparse binary data, the circuit is almost empty.

---

## 4. Binarization of Continuous Data

Real-world data is often continuous. Basis encoding handles this via a
**threshold** rule (default threshold = 0.5):

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    BINARIZATION RULE                              │
  │                                                                  │
  │   x > threshold   →   1   (X gate applied)                      │
  │   x ≤ threshold   →   0   (no gate)                             │
  │                                                                  │
  │   Note: values exactly AT the threshold map to 0                 │
  └──────────────────────────────────────────────────────────────────┘
```

### Example

```
  Continuous input:     [0.8,  0.2,  0.6,  0.4,  0.5,  0.51]
  Threshold:            0.5
                         ↓     ↓     ↓     ↓     ↓     ↓
  Binarized:            [ 1,    0,    1,    0,    0,    1  ]
                        >0.5  ≤0.5  >0.5  ≤0.5  =0.5  >0.5
```

### Custom Thresholds

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Threshold │ Use Case              │ Behavior               │
  ├────────────┼───────────────────────┼────────────────────────┤
  │    0.5     │ Data in [0, 1]        │ Standard split         │
  │    0.0     │ Signed data           │ Negative→0, Positive→1 │
  │    0.7     │ Conservative "on"     │ Only strong signals→1  │
  │   custom   │ Domain-specific       │ Match decision boundary│
  └─────────────────────────────────────────────────────────────┘

  Example with threshold = 0.0:

    Input:    [-0.5, 0.5, -0.1, 0.1, 0.0]
    Binary:   [  0,   1,    0,   1,   0 ]
              neg   pos   neg   pos   =0
```

---

## 5. Example Walkthrough

Encode the binary string **[1, 0, 1]** using 3 qubits:

```
  Step 1: Input
  ─────────────
  x = [1, 0, 1]     (already binary, no binarization needed)

  Step 2: Apply gates
  ────────────────────
  q₀: |0⟩ ─ X ─ |1⟩     (x₀ = 1 → apply X)
  q₁: |0⟩ ───── |0⟩     (x₁ = 0 → identity)
  q₂: |0⟩ ─ X ─ |1⟩     (x₂ = 1 → apply X)

  Step 3: Output state
  ─────────────────────
  |ψ⟩ = |101⟩ = |1⟩ ⊗ |0⟩ ⊗ |1⟩

  Statevector (8 entries for 3 qubits):

  |000⟩  ░░░░░░░░░░░░░░░░░░░░  0.0
  |001⟩  ░░░░░░░░░░░░░░░░░░░░  0.0
  |010⟩  ░░░░░░░░░░░░░░░░░░░░  0.0
  |011⟩  ░░░░░░░░░░░░░░░░░░░░  0.0
  |100⟩  ░░░░░░░░░░░░░░░░░░░░  0.0
  |101⟩  ████████████████████  1.0  ← the encoded state
  |110⟩  ░░░░░░░░░░░░░░░░░░░░  0.0
  |111⟩  ░░░░░░░░░░░░░░░░░░░░  0.0

  Measurement: |101⟩ with probability 1 (deterministic)
```

---

## 6. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      BASIS ENCODING PROPERTIES                      │
  ├──────────────────────┬──────────────────────────────────────────────┤
  │  Qubits required     │  n  (one per feature, linear scaling)       │
  │  Circuit depth       │  1  (constant, all gates parallel)          │
  │  Gate count          │  n  (worst-case; actual is 0 to n,          │
  │                      │      data-dependent — see actual_gate_count) │
  │  Two-qubit gates     │  0  (never, no entanglement)                │
  │  Parameter count     │  0  (no trainable parameters)               │
  │  Entangling?         │  No  (always produces product states)       │
  │  Simulability        │  Trivially classically simulable            │
  │  Trainability        │  N/A (nothing to train)                     │
  │  Expressibility      │  Minimal  (only 2ⁿ basis states reachable) │
  │  Normalization       │  Not needed  (data is binary)               │
  │  Data-dependent      │  Gate count only  (which X gates fire)      │
  │  Deterministic?      │  Yes  (measurement always returns input)    │
  └──────────────────────┴──────────────────────────────────────────────┘
```

### Why Trivially Simulable?

```
  A classical computer can predict the output of basis encoding
  in O(n) time — just read the input bits. There is:

  • No superposition  (exactly one basis state has amplitude 1)
  • No entanglement   (state is a tensor product of single qubits)
  • No interference   (no amplitudes to combine)

  The output is always:  |x₀ x₁ ... xₙ₋₁⟩  with probability 1.
```

### The 2ⁿ Reachable States

```
  For n = 3, basis encoding can produce exactly 2³ = 8 states:

  [0,0,0] → |000⟩     [1,0,0] → |100⟩
  [0,0,1] → |001⟩     [1,0,1] → |101⟩
  [0,1,0] → |010⟩     [1,1,0] → |110⟩
  [0,1,1] → |011⟩     [1,1,1] → |111⟩

  These 8 states form an orthonormal basis for the 3-qubit Hilbert
  space, but basis encoding can ONLY reach these 8 points — it cannot
  produce any superposition state like (|000⟩ + |111⟩)/√2.
```

---

## 7. Data-Dependent Gate Counts

Unlike most encodings where the circuit structure is fixed, basis
encoding's gate count depends on the input data:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    GATE COUNT ANALYSIS                            │
  │                                                                  │
  │  Theoretical maximum (worst case):  n  (all features = 1)       │
  │  Theoretical minimum (best case):   0  (all features = 0)       │
  │  Actual for specific data:          Σ xᵢ  (count of 1s)        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  n_features = 8

  Input                 Binarized       X Gates   Depth   Sparsity
  ────────────────────  ──────────────  ───────   ─────   ────────
  [1,1,1,1,1,1,1,1]    [1,1,1,1,1,1,1,1]   8       1      0%
  [1,0,1,0,1,0,1,0]    [1,0,1,0,1,0,1,0]   4       1     50%
  [1,0,0,0,0,0,0,0]    [1,0,0,0,0,0,0,0]   1       1     88%
  [0,0,0,0,0,0,0,0]    [0,0,0,0,0,0,0,0]   0       1    100%

  For sparse binary data (common in practice), the actual hardware
  cost is much lower than the worst-case bound.
```

---

## 8. Comparison with Other Encodings

```
  ┌───────────────────┬──────────┬─────────┬────────────┬──────────────┐
  │    Encoding       │  Qubits  │  Depth  │ Entangling │ Data Type    │
  ├───────────────────┼──────────┼─────────┼────────────┼──────────────┤
  │  Basis        ★   │    n     │   1     │     No     │  Binary      │
  │  Angle            │    n     │   1     │     No     │  Continuous  │
  │  Amplitude        │  log₂n  │  O(2ⁿ) │    Yes     │  Continuous  │
  │  IQP              │    n     │  O(n)   │    Yes     │  Continuous  │
  │  ZZ Feature Map   │    n     │  O(n²) │    Yes     │  Continuous  │
  └───────────────────┴──────────┴─────────┴────────────┴──────────────┘
```

```
  Visual comparison (n = 4 features):

  BASIS ENCODING               ANGLE ENCODING               IQP ENCODING
  4 qubits, depth 1            4 qubits, depth 1            4 qubits, depth ~4
  No entanglement              No entanglement              Entangling

  |0⟩─[X]──                   |0⟩─[RY(θ₀)]──               |0⟩─[H]─[RZ]─●───────
  |0⟩──────                   |0⟩─[RY(θ₁)]──               |0⟩─[H]─[RZ]─●──●────
  |0⟩─[X]──                   |0⟩─[RY(θ₂)]──               |0⟩─[H]─[RZ]────●──●─
  |0⟩──────                   |0⟩─[RY(θ₃)]──               |0⟩─[H]─[RZ]───────●─

  Simplest possible           Same width, rotations         Entangling ZZ layers
  Binary data only            Continuous features           Rich feature mixing
```

---

## 9. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌───────────────────────────┐       ┌───────────────────────────────┐
  │                           │       │                               │
  │  ✓ Simplest encoding     │       │  ✗ Binary data only           │
  │    One gate type (X)      │       │    Continuous info is lost     │
  │    Zero entanglement      │       │    via thresholding            │
  │                           │       │                               │
  │  ✓ Minimal circuit depth │       │  ✗ Linear qubit scaling       │
  │    Always depth 1         │       │    n features need n qubits    │
  │    Ideal for NISQ         │       │    (no compression)            │
  │                           │       │                               │
  │  ✓ Deterministic output  │       │  ✗ Minimal expressibility     │
  │    Measurement always     │       │    Only 2ⁿ states reachable   │
  │    returns the input      │       │    out of continuous Hilbert   │
  │                           │       │    space                       │
  │  ✓ Hardware efficient    │       │                               │
  │    No two-qubit gates     │       │  ✗ No quantum advantage       │
  │    No calibration issues  │       │    Trivially simulable —      │
  │                           │       │    a classical computer does   │
  │  ✓ Zero error from       │       │    this just as well           │
  │    encoding itself         │       │                               │
  │    (only hardware noise)  │       │  ✗ No entanglement            │
  │                           │       │    Cannot leverage quantum     │
  │                           │       │    correlations                │
  └───────────────────────────┘       └───────────────────────────────┘
```

---

## 10. Use Cases

```
                          Best suited for
                    ┌──────────────────────────────┐
                    │                              │
  ┌─────────────────┤  Combinatorial Optimization  │  QAOA, VQE encode
  │                 │  (QAOA / VQE)                │  problem solutions
  │                 ├──────────────────────────────┤  as bit strings
  │                 │                              │
  ├─────────────────┤  Database Search             │  Grover's algorithm
  │                 │  (Grover's Algorithm)        │  marks basis states
  │                 ├──────────────────────────────┤
  │                 │                              │
  ├─────────────────┤  Discrete / Categorical      │  One-hot vectors,
  │                 │  Data Classification         │  binary flags,
  │                 ├──────────────────────────────┤  feature presence
  │                 │                              │
  ├─────────────────┤  Quantum Memory (QRAM)       │  Address encoding
  │                 │                              │  for quantum RAM
  │                 ├──────────────────────────────┤
  │                 │                              │
  └─────────────────┤  Encoding Baseline           │  Simplest reference
                    │  for Benchmarking            │  to compare against
                    └──────────────────────────────┘
```

---

## 11. Resource Scaling

```
  n_features │ n_qubits │ Max X Gates │ Depth │ Two-Qubit │ Simulable
  ───────────┼──────────┼─────────────┼───────┼───────────┼──────────
       1     │    1     │      1      │   1   │     0     │   Yes
       2     │    2     │      2      │   1   │     0     │   Yes
       4     │    4     │      4      │   1   │     0     │   Yes
       8     │    8     │      8      │   1   │     0     │   Yes
      16     │   16     │     16      │   1   │     0     │   Yes
      64     │   64     │     64      │   1   │     0     │   Yes
     256     │  256     │    256      │   1   │     0     │   Yes
    1024     │  1024    │   1024      │   1   │     0     │   Yes

  Depth is ALWAYS 1. The only scaling dimension is qubit count,
  which grows linearly with features.
```

---

## 12. Backend Implementations

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    BACKEND COMPARISON                            │
  ├──────────────┬──────────────────────────────────────────────────┤
  │              │                                                  │
  │  PennyLane   │  Manual X gates via qml.PauliX                  │
  │              │  • Returns callable closure                      │
  │              │  • Big-endian (wire 0 = MSB)                     │
  │              │  • Feature x[i] → wire i  (direct mapping)      │
  │              │                                                  │
  ├──────────────┼──────────────────────────────────────────────────┤
  │              │                                                  │
  │  Qiskit      │  QuantumCircuit with qc.x(i)                   │
  │              │  • Little-endian (qubit 0 = LSB)                │
  │              │  • Feature x[i] → qubit i  (direct mapping)     │
  │              │  • LSB→MSB conversion handled by analysis module │
  │              │                                                  │
  ├──────────────┼──────────────────────────────────────────────────┤
  │              │                                                  │
  │  Cirq        │  cirq.X on LineQubits                           │
  │              │  • All X gates in a single Moment (parallel)     │
  │              │  • Big-endian (LineQubit(0) = MSB)               │
  │              │  • Feature x[i] → qubit i  (direct mapping)     │
  │              │  • ⚠ all_qubits() may return fewer qubits       │
  │              │    for sparse inputs (only active qubits)        │
  │              │                                                  │
  └──────────────┴──────────────────────────────────────────────────┘

  Qubit ordering:  All backends use direct mapping (feature x[i] →
  qubit i).  PennyLane and Cirq are both big-endian (qubit 0 = MSB),
  so no conversion is needed.  Qiskit is little-endian (qubit 0 =
  LSB), but the analysis module's _simulate_qiskit automatically
  converts statevectors to big-endian for cross-backend consistency.
```

---

## 13. When To Use (and When Not To)

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  USE BASIS ENCODING WHEN:                                        │
  │                                                                  │
  │  • Data is naturally binary or categorical                       │
  │  • You need the shallowest possible circuit (NISQ hardware)     │
  │  • Your algorithm works with marked basis states (Grover, QAOA)  │
  │  • You want a simple baseline encoding for comparisons           │
  │  • Hardware error budget is tight (fewer gates = less error)     │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  DO NOT USE BASIS ENCODING WHEN:                                 │
  │                                                                  │
  │  • Data is continuous and pattern-rich (use Angle or IQP)        │
  │  • You need superposition or interference effects                │
  │  • You need entanglement from the encoding layer                 │
  │  • You need amplitude-level information processing               │
  │  • You want sub-linear qubit scaling (use Amplitude encoding)    │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 14. Orthogonality of Encoded States

A unique property of basis encoding: **every pair of distinct inputs
maps to orthogonal quantum states**.

```
  ⟨ψ(x) | ψ(y)⟩  =  δ_{x,y}    (Kronecker delta)

  Inputs           States         Inner Product
  ──────           ──────         ─────────────
  [0,0] , [0,1]   |00⟩ , |01⟩   ⟨00|01⟩ = 0   (orthogonal)
  [0,0] , [1,0]   |00⟩ , |10⟩   ⟨00|10⟩ = 0   (orthogonal)
  [0,1] , [1,0]   |01⟩ , |10⟩   ⟨01|10⟩ = 0   (orthogonal)
  [1,0] , [1,0]   |10⟩ , |10⟩   ⟨10|10⟩ = 1   (identical)

  This means:
  • Every distinct bit string is perfectly distinguishable
  • No information overlap between different encodings
  • Kernel matrix = Identity (for distinct inputs)
```

> This perfect distinguishability is both a strength (no confusion
> between inputs) and a limitation (no notion of "similarity" between
> nearby bit strings — [0,0,0,1] and [0,0,1,0] are equally "far"
> from each other as [0,0,0,0] and [1,1,1,1]).

---

## References

1. Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and
   Quantum Information." Cambridge University Press. Chapter 1.

2. Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
   Quantum Computers." Springer. Chapter 4: Quantum Feature Maps.

3. Weigold, M., et al. (2020). "Data encoding patterns for quantum
   computing." IEEE International Conference on Quantum Computing
   and Engineering (QCE).

4. LaRose, R., & Coyle, B. (2020). "Robust data encodings for quantum
   classifiers." Physical Review A, 102(3), 032420.
