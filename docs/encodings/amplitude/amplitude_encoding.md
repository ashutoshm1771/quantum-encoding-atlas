# Amplitude Encoding

**Author:** Ashutosh Mishra

A quantum data encoding that maps classical features directly to the
**amplitudes** of a quantum state, achieving exponential compression.

```
 ╔══════════════════════════════════════════════════════════════════════╗
 ║                                                                      ║
 ║   |ψ(x)⟩  =  Σᵢ  xᵢ |i⟩       where  Σ|xᵢ|² = 1                  ║
 ║                                                                      ║
 ║   "n features  →  log₂(n) qubits  →  exponential compression"      ║
 ║                                                                      ║
 ╚══════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

Instead of dedicating one qubit per feature (like angle encoding), amplitude
encoding stores **every feature value as a probability amplitude** of a
multi-qubit quantum state. A 2ⁿ-dimensional classical vector fits into
just **n qubits**.

```
  Classical Data  (8 features)              Quantum State  (3 qubits)
  ┌──────────────────────────┐              ┌──────────────────────────┐
  │ x₀ = 0.25               │              │                          │
  │ x₁ = 0.40               │              │  |ψ⟩ = x₀|000⟩          │
  │ x₂ = 0.10   ──────────────────────►    │       + x₁|001⟩          │
  │ x₃ = 0.35               │   Encode     │       + x₂|010⟩          │
  │ x₄ = 0.50               │              │       + x₃|011⟩          │
  │ x₅ = 0.20               │              │       + x₄|100⟩          │
  │ x₆ = 0.45               │              │       + x₅|101⟩          │
  │ x₇ = 0.30               │              │       + x₆|110⟩          │
  └──────────────────────────┘              │       + x₇|111⟩          │
       8 numbers                            └──────────────────────────┘
       (normalized to ‖x‖=1)                    3 qubits  =  log₂(8)
```

> The **catch**: preparing this arbitrary state requires a circuit with
> O(2ⁿ) gates — the compression in qubits is paid for in circuit depth.

---

## 2. Circuit Structure

Unlike angle encoding's parallel single-qubit gates, amplitude encoding
needs a **global state preparation unitary** that entangles all qubits.

```
                    ┌─────────────────────────────────────────┐
  |0⟩ ──────────────┤                                         ├──  ╮
                    │                                         │    │
  |0⟩ ──────────────┤     State Preparation Unitary  U(x)    ├──  ├─ |ψ(x)⟩
                    │                                         │    │
  |0⟩ ──────────────┤                                         ├──  ╯
                    └─────────────────────────────────────────┘

                    U|000⟩  =  |ψ(x)⟩  =  Σᵢ xᵢ|i⟩
```

### Decomposed View (Mottonen et al.)

The unitary decomposes into layers of **uniformly controlled rotations**
(RY, RZ) interleaved with CNOT gates:

```
  |0⟩ ──RY──●──────RY──●──────RY──●──────RY──●──────  ...
            │          │          │          │
  |0⟩ ──RY─⊕──RY──●──RY─⊕──RY─●──                    ...
                   │          │
  |0⟩ ──RY────────⊕──RY─────⊕──                       ...

  Legend:   RY = single-qubit rotation
             ● = control qubit
             ⊕ = CNOT target
```

> The decomposition is **data-dependent**: rotation angles are computed
> from the input amplitudes. Different inputs produce different gate angles.

---

## 3. The Mathematics

### State Definition

Given a classical vector **x** = (x₀, x₁, ..., x_{n-1}):

```
                        1
  |ψ(x)⟩  =  ─────  Σᵢ  xᵢ |i⟩
               ‖x‖

  where:
    ‖x‖ = √(Σᵢ |xᵢ|²)    ← L2 norm  (ensures valid quantum state)
    |i⟩                     ← computational basis state (binary encoding of i)
    Σ|xᵢ/‖x‖|² = 1        ← Born rule constraint
```

### Qubit Count

```
  n_qubits  =  ⌈ log₂(n_features) ⌉

  If n_features is not a power of 2, zero-pad to the next power of 2:

  n_features │ n_qubits │ state_dim │ padding
  ───────────┼──────────┼───────────┼─────────
       1     │    1     │     2     │    1
       2     │    1     │     2     │    0
       3     │    2     │     4     │    1
       4     │    2     │     4     │    0
       5     │    3     │     8     │    3
       8     │    3     │     8     │    0
      10     │    4     │    16     │    6
      16     │    4     │    16     │    0
     100     │    7     │   128     │   28
    1024     │   10     │  1024     │    0
```

### Normalization

```
  Raw input:        x = [3, 4, 0, 0]
  L2 norm:          ‖x‖ = √(9 + 16 + 0 + 0) = 5
  Normalized:       x' = [0.6, 0.8, 0, 0]

  Quantum state:    |ψ⟩ = 0.6|00⟩ + 0.8|01⟩ + 0|10⟩ + 0|11⟩

  ⚠ Normalization discards the original magnitude (‖x‖ = 5 is lost).
    Only the relative ratios between features are preserved.
```

---

## 4. Example Walkthrough

Encode **x** = [1, 2, 3, 4] into 2 qubits:

```
  Step 1: Normalize
  ─────────────────
  ‖x‖ = √(1 + 4 + 9 + 16) = √30 ≈ 5.477

  x' = [1/√30, 2/√30, 3/√30, 4/√30]
     ≈ [0.183, 0.365, 0.548, 0.730]

  Step 2: Map to amplitudes
  ─────────────────────────
  |ψ⟩ = 0.183|00⟩ + 0.365|01⟩ + 0.548|10⟩ + 0.730|11⟩
          x₀           x₁           x₂           x₃

  Step 3: Verify normalization
  ────────────────────────────
  |0.183|² + |0.365|² + |0.548|² + |0.730|² = 1.000  ✓
```

```
  Measurement probabilities:

  |00⟩  █████░░░░░░░░░░░░░░░░░  0.033   (x₀² / ‖x‖²  =  1/30)
  |01⟩  ████████░░░░░░░░░░░░░░  0.133   (x₁² / ‖x‖²  =  4/30)
  |10⟩  █████████████░░░░░░░░░  0.300   (x₂² / ‖x‖²  =  9/30)
  |11⟩  ██████████████████░░░░  0.533   (x₃² / ‖x‖²  = 16/30)
                                 ─────
                                 1.000

  The largest feature (x₃ = 4) dominates the measurement distribution.
```

---

## 5. Compression vs. Depth Tradeoff

This is the **fundamental tension** of amplitude encoding:

```
                Exponential Compression
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  n features  ───►  log₂(n) qubits           │
  │                                              │
  │  4 features    →  2 qubits     (2x)         │
  │  16 features   →  4 qubits     (4x)         │
  │  256 features  →  8 qubits     (32x)        │
  │  1024 features →  10 qubits    (102x)       │
  │  1M features   →  20 qubits    (50000x!)    │
  │                                              │
  └──────────────────────────────────────────────┘
                        BUT
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  Circuit depth  =  O(2^n_qubits)             │
  │                                              │
  │  2 qubits   →  depth ~4       gates ~6       │
  │  4 qubits   →  depth ~16     gates ~30       │
  │  8 qubits   →  depth ~256    gates ~510      │
  │  10 qubits  →  depth ~1024   gates ~2046     │
  │  20 qubits  →  depth ~1M     gates ~2M       │
  │                                              │
  └──────────────────────────────────────────────┘

  The qubit savings are exactly cancelled by circuit depth growth.
  This is not a loophole — it is a fundamental result in quantum computing.
```

```
  Visual intuition:

  ANGLE ENCODING (n=8)              AMPLITUDE ENCODING (n=8)
  8 qubits, depth 1                 3 qubits, depth ~8

  |0⟩─[R]─                         |0⟩─[RY]─●──[RY]─●──[RY]─●──[RY]─●──
  |0⟩─[R]─                               │       │       │       │
  |0⟩─[R]─                         |0⟩─[RY]─⊕──[RY]─●──[RY]─⊕──[RY]─●──
  |0⟩─[R]─                                          │              │
  |0⟩─[R]─                         |0⟩──────────────⊕──[RY]───────⊕──
  |0⟩─[R]─
  |0⟩─[R]─                         Wide but shallow    Narrow but deep
  |0⟩─[R]─                         No entanglement     Fully entangled
```

---

## 6. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   AMPLITUDE ENCODING PROPERTIES                     │
  ├──────────────────────┬──────────────────────────────────────────────┤
  │  Qubits required     │  ⌈log₂(n)⌉  (exponential compression)      │
  │  Circuit depth       │  O(2^n_qubits)  (exponential in qubits)     │
  │  Total gates         │  ~2^(n_qubits+1) - 2  (estimate)           │
  │  Single-qubit gates  │  ~2^n_qubits  (RY/RZ rotations)            │
  │  Two-qubit gates     │  ~2^n_qubits - 2  (CNOTs)                  │
  │  Entangling?         │  Yes  (creates fully entangled states)      │
  │  Simulability        │  Not classically simulable                  │
  │  Trainability        │  Moderate (~0.5)  deep circuits → plateaus  │
  │  Expressibility      │  Maximal  (any state in Hilbert space)      │
  │  Normalization       │  Required  (‖ψ‖ = 1, magnitudes lost)      │
  │  Data-dependent      │  Yes  (gate angles depend on input values)  │
  └──────────────────────┴──────────────────────────────────────────────┘
```

### Property Comparison with Angle Encoding

```
  Trainability     Angle  ████████████████████░░  ~0.9
                   Ampl.  ████████████░░░░░░░░░░  ~0.5

  Expressibility   Angle  ████████░░░░░░░░░░░░░░  Low  (product states)
                   Ampl.  ████████████████████░░  Maximal (full Hilbert)

  Hardware Cost    Angle  ██░░░░░░░░░░░░░░░░░░░░  O(n) depth
                   Ampl.  ██████████████████████  O(2ⁿ) depth

  Qubit Efficiency Angle  ██░░░░░░░░░░░░░░░░░░░░  n qubits for n features
                   Ampl.  ████████████████████░░  log₂n qubits for n feat.

  Noise Resilience Angle  ████████████████░░░░░░  High (shallow)
                   Ampl.  ████░░░░░░░░░░░░░░░░░░  Low  (deep circuits)
```

---

## 7. Normalization: What Is Preserved, What Is Lost

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    NORMALIZATION EFFECTS                          │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  PRESERVED:                                                      │
  │    • Relative ratios between features  (xᵢ / xⱼ)                │
  │    • Angular relationships             (direction on unit sphere)│
  │    • Feature ranking / ordering                                  │
  │                                                                  │
  │  LOST:                                                           │
  │    • Absolute magnitudes               (‖x‖ is discarded)       │
  │    • Scale information                 ([1,2] ≡ [100,200])      │
  │    • Distance from origin                                        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  Example:

    x = [1, 2, 3]    ──normalize──►  [0.267, 0.535, 0.802]  ─┐
                                                               ├─ Same state!
    x = [10, 20, 30]  ──normalize──►  [0.267, 0.535, 0.802]  ─┘

  These two very different vectors produce the IDENTICAL quantum state.
```

---

## 8. Zero-Padding for Non-Power-of-2 Features

```
  n_features = 5  →  n_qubits = 3  →  state_dim = 8

  Input:    x  = [x₀, x₁, x₂, x₃, x₄]
  Padded:   x' = [x₀, x₁, x₂, x₃, x₄,  0,  0,  0]
                                          ↑───────↑
                                          zero padding

  Quantum state:
  |ψ⟩ = x₀'|000⟩ + x₁'|001⟩ + x₂'|010⟩ + x₃'|011⟩ + x₄'|100⟩
         + 0|101⟩  +  0|110⟩  +  0|111⟩

  The padded amplitudes are always zero, wasting part of the Hilbert space.
  3 out of 8 basis states carry no information.
```

---

## 9. Backend Implementations

Each quantum framework implements state preparation differently:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    BACKEND COMPARISON                            │
  ├──────────────┬──────────────────────────────────────────────────┤
  │              │                                                  │
  │  PennyLane   │  qml.AmplitudeEmbedding                         │
  │              │  • Optimized decomposition                       │
  │              │  • Supports autodiff                             │
  │              │  • MSB qubit ordering (native)                   │
  │              │                                                  │
  ├──────────────┼──────────────────────────────────────────────────┤
  │              │                                                  │
  │  Qiskit      │  QuantumCircuit.initialize()                    │
  │              │  • Automatic synthesis & optimization            │
  │              │  • LSB qubit ordering (bit-reversal applied)     │
  │              │  • Good transpiler integration                   │
  │              │                                                  │
  ├──────────────┼──────────────────────────────────────────────────┤
  │              │                                                  │
  │  Cirq        │  Custom unitary gate via QR decomposition        │
  │              │  • Constructs full 2ⁿ x 2ⁿ unitary matrix       │
  │              │  • O(4ⁿ) memory — expensive for large n         │
  │              │  • MSB qubit ordering (native)                   │
  │              │                                                  │
  └──────────────┴──────────────────────────────────────────────────┘
```

### Qubit Ordering Convention

```
  This library uses MSB (Most Significant Bit) ordering throughout:

  Index    Binary    Qubit 0    Qubit 1    Convention
  ─────    ──────    ───────    ───────    ──────────
    0       00         0          0        MSB (q0 = leftmost)
    1       01         0          1
    2       10         1          0
    3       11         1          1

  PennyLane & Cirq:  MSB natively  →  no conversion needed
  Qiskit:            LSB natively  →  bit-reversal applied internally
```

---

## 10. Resource Scaling

```
  n_features │ n_qubits │ Compression │ Est. Gates │ Est. Depth │ Cirq Memory
  ───────────┼──────────┼─────────────┼────────────┼────────────┼────────────
        4    │    2     │    2.0x     │         6  │         4  │   256 B
        8    │    3     │    2.7x     │        14  │         8  │   4 KB
       16    │    4     │    4.0x     │        30  │        16  │   64 KB
       32    │    5     │    6.4x     │        62  │        32  │   1 MB
       64    │    6     │   10.7x     │       126  │        64  │   16 MB
      256    │    8     │   32.0x     │       510  │       256  │   4 GB
     1024    │   10     │  102.4x     │      2046  │      1024  │   16 MB *
     4096    │   12     │  341.3x     │      8190  │      4096  │   256 MB *

  * Cirq memory = 4^n_qubits × 16 bytes (full unitary matrix)
    ⚠ Warning issued at n_qubits ≥ 12 (~256 MB)
```

---

## 11. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌───────────────────────────┐       ┌───────────────────────────────┐
  │                           │       │                               │
  │  ✓ Exponential            │       │  ✗ O(2ⁿ) gate complexity     │
  │    compression             │       │    Exponentially deep circuits│
  │    n → log₂(n) qubits     │       │                               │
  │                           │       │                               │
  │  ✓ Full Hilbert space    │       │  ✗ Normalization loss         │
  │    Any quantum state       │       │    Original magnitudes lost   │
  │    is reachable            │       │                               │
  │                           │       │                               │
  │  ✓ Natural for quantum   │       │  ✗ NISQ-unfriendly           │
  │    algorithms (HHL, QSVM)  │       │    Deep circuits accumulate   │
  │                           │       │    noise and decoherence       │
  │                           │       │                               │
  │  ✓ Preserves geometric   │       │  ✗ Barren plateaus            │
  │    structure               │       │    Deep entangling circuits   │
  │    (direction on sphere)   │       │    hinder gradient training   │
  │                           │       │                               │
  │  ✓ Not classically       │       │  ✗ Data-dependent circuits    │
  │    simulable               │       │    Each input needs a new     │
  │    (potential advantage)   │       │    circuit compilation         │
  │                           │       │                               │
  └───────────────────────────┘       └───────────────────────────────┘
```

---

## 12. Use Cases

```
                          Best suited for
                    ┌──────────────────────────┐
                    │                          │
  ┌─────────────────┤  Quantum Linear Solvers  │  HHL algorithm requires
  │                 │  (HHL)                   │  amplitude-encoded input
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Quantum Kernel          │  Inner products computed
  │                 │  Methods (QSVM)          │  in exponential space
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Quantum Neural          │  Input layer for VQCs
  │                 │  Networks                │  with full state access
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Quantum PCA             │  Exponential speedup for
  │                 │  (QPCA)                  │  low-rank matrices
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Quantum Sampling        │  Represent probability
  │                 │  & Monte Carlo           │  distributions as states
  │                 ├──────────────────────────┤
  │                 │                          │
  └─────────────────┤  Fault-Tolerant          │  Deep circuits viable
                    │  Quantum Computing       │  with error correction
                    └──────────────────────────┘
```

---

## 13. Data Preprocessing

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    PREPROCESSING PIPELINE                             │
  │                                                                      │
  │  Raw Data ──► Feature Scaling ──► (Optional) ──► Normalization ──►  │
  │                                                    (auto or manual)  │
  │                                                                      │
  │  Recommendations:                                                    │
  │                                                                      │
  │  1. Scale features to similar ranges BEFORE encoding                 │
  │     (prevents one feature from dominating the quantum state)         │
  │                                                                      │
  │  2. Use normalize=True (default) for automatic L2 normalization      │
  │     or pre-normalize with: x = x / np.linalg.norm(x)                │
  │                                                                      │
  │  3. Non-power-of-2 features are auto-padded with zeros              │
  │     Consider if zero-padding biases downstream algorithms            │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 14. Comparison with Other Encodings

```
  ┌───────────────────┬──────────┬──────────┬────────────┬──────────────┐
  │    Encoding       │  Qubits  │  Depth   │ Entangling │ Simulability │
  ├───────────────────┼──────────┼──────────┼────────────┼──────────────┤
  │  Amplitude ★      │  log₂n  │  O(2ⁿ)  │    Yes     │  Not sim.    │
  │  Angle            │    n     │  O(1)    │     No     │  Simulable   │
  │  Basis            │    n     │  O(1)    │     No     │  Simulable   │
  │  IQP              │    n     │  O(n²)  │    Yes     │  Not sim.    │
  │  ZZ Feature Map   │    n     │  O(n²)  │    Yes     │  Not sim.    │
  └───────────────────┴──────────┴──────────┴────────────┴──────────────┘

  Key insight:
  ┌────────────────────────────────────────────────────────────────┐
  │  Amplitude encoding is the only encoding that achieves        │
  │  SUB-LINEAR qubit scaling, but at the cost of exponential     │
  │  circuit depth. Every other encoding uses at least n qubits.  │
  └────────────────────────────────────────────────────────────────┘
```

---

## 15. The State Preparation Problem

Why is amplitude encoding so expensive? It reduces to a fundamental
result in quantum information theory:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  THEOREM (Knill, 1995; Plesch & Buzek, 2011):                   │
  │                                                                  │
  │  Preparing an arbitrary n-qubit quantum state requires           │
  │  Ω(2ⁿ) elementary gates in the worst case.                      │
  │                                                                  │
  │  No quantum circuit can do better for general state vectors.     │
  │                                                                  │
  │  This is because the set of n-qubit states is a 2^(n+1) - 2    │
  │  dimensional manifold (real parameters), and each gate adds      │
  │  at most O(1) parameters.                                        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  The Mottonen et al. decomposition used in this implementation achieves
  this theoretical lower bound (up to constant factors), making it
  asymptotically optimal for general state preparation.
```

---

## References

1. Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
   Quantum Computers." Springer. Chapter 4: Quantum Feature Maps.

2. Mottonen, M., et al. (2004). "Transformation of quantum states using
   uniformly controlled rotations." Quantum Information & Computation.

3. Shende, V., Bullock, S., & Markov, I. (2006). "Synthesis of quantum
   logic circuits." IEEE Transactions on CAD.

4. Araujo, I. F., et al. (2021). "A divide-and-conquer algorithm for
   quantum state preparation." Scientific Reports, 11, 6329.
