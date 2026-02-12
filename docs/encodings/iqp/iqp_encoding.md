# IQP Encoding

A quantum data encoding based on **Instantaneous Quantum Polynomial** circuits —
diagonal phase gates sandwiched between Hadamard layers — producing highly
entangled states with **provable classical hardness** guarantees.

```
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║                                                                       ║
 ║   |ψ(x)⟩  =  [ H⊗ⁿ · U_Z(x) · U_ZZ(x) ]^reps  |0⟩⊗ⁿ              ║
 ║                                                                       ║
 ║   "Hadamards create superposition,                                    ║
 ║    diagonal phases encode data,                                       ║
 ║    ZZ gates entangle features"                                        ║
 ║                                                                       ║
 ╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

IQP encoding builds quantum states in three stages per layer:
**superpose**, **phase-encode**, **entangle**. The result is a state
whose output distribution is **provably hard to sample classically**.

```
                                  ┌──────────────┐
  Classical Features ─────────────┤              │
                                  │  1. Hadamard │  Create uniform
  x = [x₀, x₁, ..., x_{n-1}]    │     layer    │  superposition
                                  │              │
                                  ├──────────────┤
                                  │              │
                                  │  2. RZ(2xᵢ)  │  Encode individual
                                  │     layer    │  features as phases
                                  │              │
                                  ├──────────────┤
                                  │              │
                                  │  3. ZZ(xᵢxⱼ) │  Encode pairwise
                                  │     layer    │  feature products
                                  │              │
                                  └──────┬───────┘
                                         │
                                    Repeat × reps
                                         │
                                         ▼
                                  Entangled |ψ(x)⟩
```

> Unlike angle encoding (product states) or amplitude encoding (single
> global unitary), IQP interleaves **local phases** with **pairwise
> entanglement**, capturing both individual features and their interactions.

---

## 2. Circuit Structure

### Single Layer (reps = 1, full entanglement, 4 qubits)

```
        ┌───┐ ┌─────────┐
  |0⟩ ──┤ H ├─┤ RZ(2x₀) ├──●─────────●─────────●─────────────────
        └───┘ └─────────┘  │         │         │
        ┌───┐ ┌─────────┐  │         │         │
  |0⟩ ──┤ H ├─┤ RZ(2x₁) ├──⊕──RZ────●─────────│────●─────────────
        └───┘ └─────────┘  (2x₀x₁)  │         │    │
        ┌───┐ ┌─────────┐           │         │    │
  |0⟩ ──┤ H ├─┤ RZ(2x₂) ├───────────⊕──RZ────│────⊕──RZ──●──────
        └───┘ └─────────┘           (2x₀x₂)  │   (2x₁x₂) │
        ┌───┐ ┌─────────┐                     │           │
  |0⟩ ──┤ H ├─┤ RZ(2x₃) ├────────────────────⊕──RZ──────⊕──RZ──
        └───┘ └─────────┘                    (2x₀x₃)    (2x₁x₃)

                                              ... + ZZ(x₂x₃) pair
```

### The ZZ Gate — Up Close

Each ZZ interaction decomposes into a CNOT–RZ–CNOT sandwich:

```
  ZZ(xᵢxⱼ)  =

  qubit i ──●───────────●──
            │           │
  qubit j ──⊕──RZ(2θ)──⊕──     where θ = xᵢ · xⱼ

  This applies the unitary:  exp(-i · xᵢxⱼ · Zᵢ ⊗ Zⱼ)
```

> The CNOT pair creates a temporary entanglement, the RZ encodes the
> **product** of two features, and the second CNOT uncomputes the
> computational basis correlation — leaving only the phase imprint.

### Multiple Layers (reps = 2)

```
  ┌─────────────────────────────┐ ┌─────────────────────────────┐
  │   H ─ RZ(2xᵢ) ─ ZZ(xᵢxⱼ) │ │   H ─ RZ(2xᵢ) ─ ZZ(xᵢxⱼ) │
  │        Layer 1              │ │        Layer 2              │
  └─────────────────────────────┘ └─────────────────────────────┘
                 ↓                              ↓
       Creates entangled             Re-encodes data with
       state with feature            additional phase structure
       phases & interactions          (increases expressibility)
```

> Additional layers **do not** simply scale angles (unlike angle encoding).
> Each repetition applies the Hadamard-phase-entangle cycle again,
> creating progressively more complex interference patterns.

---

## 3. Entanglement Topologies

The `entanglement` parameter controls which qubit pairs get ZZ gates.
This is a critical design choice affecting expressivity, gate count,
and hardware compatibility.

### Full Entanglement

Every pair (i, j) with i < j has a ZZ interaction.

```
  4 qubits, full:  6 pairs         Connectivity graph:

  Pairs: (0,1) (0,2) (0,3)         0 ──── 1
         (1,2) (1,3)                │ ╲  ╱ │
         (2,3)                      │  ╳   │
                                    │ ╱  ╲ │
  n(n-1)/2 = 6 pairs               3 ──── 2
```

### Linear Entanglement

Only nearest-neighbor pairs (i, i+1).

```
  4 qubits, linear:  3 pairs       Connectivity graph:

  Pairs: (0,1) (1,2) (2,3)         0 ──── 1 ──── 2 ──── 3

  n-1 = 3 pairs
```

### Circular Entanglement

Nearest-neighbor plus a wrap-around edge.

```
  4 qubits, circular:  4 pairs     Connectivity graph:

  Pairs: (0,1) (1,2) (2,3) (3,0)   0 ──── 1
                                    │      │
  n = 4 pairs                       3 ──── 2
```

### Topology Comparison

```
  ┌──────────┬───────────┬────────────────┬──────────────────┬───────────────┐
  │ Topology │ ZZ Pairs  │ CNOTs/layer    │ Connectivity     │ Best For      │
  │          │ per layer │                │ Required         │               │
  ├──────────┼───────────┼────────────────┼──────────────────┼───────────────┤
  │          │           │                │                  │               │
  │ full     │ n(n-1)/2  │ n(n-1)         │ All-to-all       │ Max           │
  │          │           │                │                  │ expressivity  │
  │          │           │                │                  │               │
  │ linear   │ n - 1     │ 2(n-1)         │ Nearest-neighbor │ NISQ hardware │
  │          │           │                │ (chain)          │ linear chips  │
  │          │           │                │                  │               │
  │ circular │ n         │ 2n             │ Ring             │ Periodic      │
  │          │           │                │                  │ problems      │
  │          │           │                │                  │               │
  └──────────┴───────────┴────────────────┴──────────────────┴───────────────┘

  Gate count scaling (per layer, for n features):

  n     │  full CNOTs  │  linear CNOTs  │  circular CNOTs
  ──────┼──────────────┼────────────────┼─────────────────
    4   │      12      │       6        │        8
    8   │      56      │      14        │       16
   12   │     132      │      22        │       24
   16   │     240      │      30        │       32
   20   │     380      │      38        │       40

  Full entanglement grows as O(n²), while linear/circular grow as O(n).
```

---

## 4. The Mathematics

### Per-Layer Unitary

Each IQP layer applies three unitary operations in sequence:

```
  U_layer(x) = U_ZZ(x) · U_Z(x) · H⊗ⁿ

  where:

  H⊗ⁿ        =  Hadamard on every qubit      (superposition)
  U_Z(x)     =  ⊗ᵢ RZ(2xᵢ)                   (individual phases)
  U_ZZ(x)    =  ∏_{(i,j)} ZZ(xᵢ · xⱼ)        (pairwise interactions)
```

### Gate Definitions

```
           ┌         ┐            1   ┌       ┐
  H   =   │ 1    1  │  ·  ───── │ 1  1│
           │ 1   -1  │     √2    │ 1 -1│
           └         ┘            └       ┘

                    ┌              ┐
  RZ(θ)   =        │ e^{-iθ/2}  0 │
                    │ 0    e^{iθ/2}│
                    └              ┘

  ZZ(θ)   =   exp(-iθ Z⊗Z)

           =   CNOT · (I ⊗ RZ(2θ)) · CNOT

  Applied with θ = xᵢ · xⱼ  (product of features)
```

### Why the Factor of 2?

```
  The RZ gates use  RZ(2xᵢ)  rather than  RZ(xᵢ).

  Reason:  RZ(θ) produces phases  e^{±iθ/2}.
           With θ = 2x, the effective phase difference is:
              e^{-ix} - e^{ix} = -2i·sin(x)

  This ensures full [0, 2π] phase coverage when features span [0, π],
  matching the convention in Havlíček et al. (2019).
```

### What the State Encodes

```
  After one layer on |0⟩⊗ⁿ, the state is (schematically):

  |ψ(x)⟩ = (1/√2ⁿ) Σ_{z∈{0,1}ⁿ}  exp(i · φ(x, z)) |z⟩

  where the phase function φ encodes:

    φ(x, z) = Σᵢ  2xᵢ · zᵢ           ← individual feature phases
             + Σ_{(i,j)}  2xᵢxⱼ · zᵢzⱼ  ← pairwise interaction phases

  Every computational basis state |z⟩ acquires a phase that is a
  POLYNOMIAL FUNCTION of the input features — hence "Instantaneous
  Quantum Polynomial."
```

---

## 5. Classical Hardness — Why IQP Matters

This is the property that makes IQP encoding theoretically significant.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  THEOREM (Bremner, Montanaro & Shepherd, 2016):                      │
  │                                                                      │
  │  If the output distribution of IQP circuits could be sampled         │
  │  classically (even approximately), then the polynomial hierarchy     │
  │  would collapse to the third level.                                  │
  │                                                                      │
  │  This is considered extremely unlikely in complexity theory,          │
  │  providing strong evidence that:                                     │
  │                                                                      │
  │    Classical computers CANNOT efficiently simulate IQP circuits.      │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

  What this means for machine learning:

  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  The quantum kernel  k(x, x') = |⟨ψ(x)|ψ(x')⟩|²             │
  │  computed from IQP-encoded states CANNOT be efficiently        │
  │  computed classically (under standard assumptions).            │
  │                                                                │
  │  This provides a theoretical foundation for potential          │
  │  quantum advantage in kernel-based classification.             │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

```
  Comparison of classical simulation complexity:

  Encoding         │ Classical Simulation │ Why?
  ─────────────────┼──────────────────────┼──────────────────────────
  Angle            │ Easy   O(n)          │ Product states, no entangl.
  Basis            │ Easy   O(n)          │ Computational basis, trivial
  IQP ★            │ HARD   (believed)    │ Polynomial hierarchy collapse
  Amplitude        │ HARD   O(2ⁿ)        │ Arbitrary entangled states
  ZZ Feature Map   │ HARD   (believed)    │ Same IQP family
```

---

## 6. Feature Interactions — The Quantum Kernel Perspective

IQP encoding naturally performs a **quantum polynomial feature expansion**.

```
  Classical polynomial kernel:

    k(x, x') = (⟨x, x'⟩ + c)^d

    This computes ALL monomials up to degree d, but the number of
    terms grows combinatorially: O(n^d).

  IQP quantum kernel:

    k(x, x') = |⟨ψ(x)|ψ(x')⟩|²

    The ZZ gates encode specific pairwise products xᵢ·xⱼ as PHASES.
    The Hadamard layers create INTERFERENCE between these phases.

    The resulting kernel captures feature interactions that are
    embedded in an exponentially large Hilbert space.
```

```
  Which interactions are captured?

  ┌─────────────────┬──────────────────────────────────────────┐
  │ Gate            │ Interaction Type                          │
  ├─────────────────┼──────────────────────────────────────────┤
  │ RZ(2xᵢ)        │ Linear:     xᵢ                           │
  │ ZZ(xᵢxⱼ)       │ Quadratic:  xᵢ · xⱼ                     │
  │ Multiple reps   │ Higher-order interference from repeated  │
  │                 │ Hadamard-phase-entangle cycles            │
  └─────────────────┴──────────────────────────────────────────┘

  With full entanglement:  ALL n(n-1)/2 pairwise interactions.
  With linear:             Only n-1 nearest-neighbor interactions.
```

---

## 7. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      IQP ENCODING PROPERTIES                        │
  ├──────────────────────┬──────────────────────────────────────────────┤
  │  Qubits required     │  n  (one per feature)                       │
  │  Depth per rep       │  3 logical layers  (H + RZ + ZZ)            │
  │  Total gates (full)  │  reps × (2n + 3·n(n-1)/2)                  │
  │  Hadamard gates      │  n × reps                                   │
  │  RZ gates (single)   │  n × reps                                   │
  │  RZ gates (ZZ)       │  n_pairs × reps                             │
  │  CNOT gates          │  2 × n_pairs × reps                         │
  │  Entangling?         │  Yes  (ZZ creates pairwise entanglement)    │
  │  Simulability        │  NOT classically simulable (provably hard)  │
  │  Trainability        │  max(0.3, 0.9 - 0.1×reps)                  │
  │  Expressibility      │  High  (entangled, data-dependent phases)   │
  │  Data-dependent      │  Yes  (both RZ and ZZ angles from input)    │
  │  Classical hardness   │  Polynomial hierarchy collapse assumption  │
  └──────────────────────┴──────────────────────────────────────────────┘
```

### How It Compares

```
  Trainability     Angle  ████████████████████░░  ~0.9
                   IQP-1  ████████████████░░░░░░  ~0.8  (1 rep)
                   IQP-2  ██████████████░░░░░░░░  ~0.7  (2 reps, default)
                   IQP-5  ████████░░░░░░░░░░░░░░  ~0.4  (5 reps)

  Expressibility   Angle  ████████░░░░░░░░░░░░░░  Low   (product states)
                   IQP    ████████████████████░░  High  (entangled + phases)

  Gate Cost        Angle  ██░░░░░░░░░░░░░░░░░░░░  O(n)
  (full ent.)      IQP    ██████████████████░░░░  O(n²) per layer

  Noise Resilience Angle  ████████████████░░░░░░  High  (shallow)
                   IQP    ████████████░░░░░░░░░░  Moderate (deeper)
```

---

## 8. Example Walkthrough

Encode **x** = [0.5, 1.0, 1.5] with `reps=1`, `entanglement='full'`:

```
  Step 1: Hadamard layer
  ──────────────────────
  |000⟩  ──H⊗³──►  |+++⟩  =  (1/√8) Σ_{z∈{0,1}³} |z⟩

  All 8 basis states have equal amplitude 1/√8.

  Step 2: Single-qubit RZ rotations
  ──────────────────────────────────
  RZ(2·0.5) = RZ(1.0) on qubit 0    →  phase: e^{±i·0.5}
  RZ(2·1.0) = RZ(2.0) on qubit 1    →  phase: e^{±i·1.0}
  RZ(2·1.5) = RZ(3.0) on qubit 2    →  phase: e^{±i·1.5}

  Each basis state |z₀z₁z₂⟩ picks up phase:
    exp(-i·(0.5·z₀ + 1.0·z₁ + 1.5·z₂))     (z with ±1 mapping)

  Step 3: ZZ interactions (full = 3 pairs)
  ─────────────────────────────────────────
  Pair (0,1):  ZZ(0.5 × 1.0) = ZZ(0.5)
  Pair (0,2):  ZZ(0.5 × 1.5) = ZZ(0.75)
  Pair (1,2):  ZZ(1.0 × 1.5) = ZZ(1.5)

  Each pair adds product-dependent phases to basis states where
  BOTH qubits in the pair are in the |1⟩ state.

  Result: A fully entangled 3-qubit state where each of the 8
  basis states has a unique phase determined by both individual
  features AND their pairwise products.
```

```
  Phase structure (schematic):

  |000⟩  →  φ = 0                               (no features active)
  |001⟩  →  φ = f(x₂)                           (only feature 2)
  |010⟩  →  φ = f(x₁)                           (only feature 1)
  |011⟩  →  φ = f(x₁) + f(x₂) + g(x₁·x₂)      (features 1,2 + interaction)
  |100⟩  →  φ = f(x₀)                           (only feature 0)
  |101⟩  →  φ = f(x₀) + f(x₂) + g(x₀·x₂)      (features 0,2 + interaction)
  |110⟩  →  φ = f(x₀) + f(x₁) + g(x₀·x₁)      (features 0,1 + interaction)
  |111⟩  →  φ = f(x₀)+f(x₁)+f(x₂) + ALL pairs  (everything interacts)

  All amplitudes have EQUAL magnitude (1/√8) — only the PHASES differ.
  This is the hallmark of IQP circuits: pure phase encoding.
```

---

## 9. The Barren Plateau Tradeoff

More entanglement and depth increase expressivity but risk **barren plateaus**
— exponentially vanishing gradients that make training impossible.

```
                  Expressivity
                       ▲
                       │          ╱ Barren plateau zone
                       │        ╱  (gradients → 0 exponentially)
                       │      ╱
                       │    ╱    ★ Sweet spot
                       │  ╱      (reps=1-2, default=2)
                       │╱
                       └──────────────────────► Depth / Entanglement


  Trainability estimate (this implementation):

    trainability = max(0.3,  0.9 - 0.1 × reps)

    reps = 1  →  0.8    Good
    reps = 2  →  0.7    Good  (default)
    reps = 3  →  0.6    Moderate
    reps = 5  →  0.4    Challenging
    reps = 7+ →  0.3    Floor (barren plateaus likely)
```

### Mitigation Strategies

```
  ┌──────────────────────────────┬─────────────────────────────────┐
  │ Strategy                     │ How It Helps                    │
  ├──────────────────────────────┼─────────────────────────────────┤
  │ Fewer repetitions (reps≤2)   │ Shallower circuits, larger      │
  │                              │ gradients                       │
  ├──────────────────────────────┼─────────────────────────────────┤
  │ Linear/circular entanglement │ Fewer CNOT gates, less          │
  │                              │ entanglement entropy             │
  ├──────────────────────────────┼─────────────────────────────────┤
  │ Layer-wise training          │ Train one layer at a time,      │
  │                              │ freezing others                  │
  ├──────────────────────────────┼─────────────────────────────────┤
  │ Careful initialization       │ Start near identity to preserve │
  │                              │ gradient signal                  │
  └──────────────────────────────┴─────────────────────────────────┘
```

---

## 10. Resource Scaling

### Full Entanglement (default)

```
  n_features │ Qubits │ ZZ Pairs │ CNOTs/layer │ Total Gates (reps=2)
  ───────────┼────────┼──────────┼─────────────┼─────────────────────
       3     │   3    │     3    │       6     │       30
       4     │   4    │     6    │      12     │       52
       6     │   6    │    15    │      30     │      102
       8     │   8    │    28    │      56     │      200
      10     │  10    │    45    │      90     │      310
      12 ⚠   │  12    │    66    │     132     │      444
      16 ⚠   │  16    │   120    │     240     │      784
      20 ⚠   │  20    │   190    │     380     │     1220

  ⚠  Warning issued at n > 12 (circuit may exceed NISQ capabilities)
```

### Linear Entanglement

```
  n_features │ Qubits │ ZZ Pairs │ CNOTs/layer │ Total Gates (reps=2)
  ───────────┼────────┼──────────┼─────────────┼─────────────────────
       4     │   4    │     3    │       6     │       34
       8     │   8    │     7    │      14     │       74
      16     │  16    │    15    │      30     │      154
      32     │  32    │    31    │      62     │      314
      64     │  64    │    63    │     126     │      634

  Linear scaling: practical even for large feature counts.
```

---

## 11. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌───────────────────────────┐       ┌───────────────────────────────┐
  │                           │       │                               │
  │  ✓ Provably hard to      │       │  ✗ O(n²) gates (full ent.)   │
  │    simulate classically    │       │    Quadratic CNOT scaling     │
  │                           │       │                               │
  │  ✓ Captures pairwise     │       │  ✗ All-to-all connectivity   │
  │    feature interactions    │       │    needed for full ent.       │
  │    via ZZ gates            │       │                               │
  │                           │       │                               │
  │  ✓ High expressibility   │       │  ✗ Barren plateaus risk       │
  │    Entangled states with   │       │    with deep circuits         │
  │    complex interference    │       │    (high reps)                │
  │                           │       │                               │
  │  ✓ Tunable entanglement  │       │  ✗ Linear qubit scaling       │
  │    full / linear / circ.   │       │    n features → n qubits     │
  │                           │       │                               │
  │  ✓ Natural quantum kernel│       │  ✗ Phase-only encoding        │
  │    for QSVM / QML          │       │    Equal amplitudes — info    │
  │                           │       │    only in phases              │
  │                           │       │                               │
  └───────────────────────────┘       └───────────────────────────────┘
```

---

## 12. Use Cases

```
                          Best suited for
                    ┌──────────────────────────┐
                    │                          │
  ┌─────────────────┤  Quantum Kernel          │  Provably hard-to-compute
  │                 │  Methods                 │  kernels for classification
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Quantum Advantage       │  Candidate for demonstrating
  │                 │  Benchmarking            │  quantum vs classical gaps
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Feature Interaction     │  Problems where xᵢ·xⱼ terms
  │                 │  Modeling                │  matter (physics, chemistry)
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Variational             │  Fixed or trainable feature
  │                 │  Classifiers             │  map in hybrid models
  │                 ├──────────────────────────┤
  │                 │                          │
  └─────────────────┤  Quantum Neural          │  Entangling input layer
                    │  Networks                │  for QNN architectures
                    └──────────────────────────┘
```

---

## 13. Data Preprocessing

```
  IQP phases are controlled by two types of terms:

    Individual:    2xᵢ           (from RZ gates)
    Pairwise:      2xᵢ · xⱼ     (from ZZ gates)

  The pairwise term is a PRODUCT, so feature scaling affects
  interaction strengths quadratically.

  ┌──────────────────────────────────────────────────────────────────┐
  │                    SCALING RECOMMENDATIONS                       │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  1. Scale features to [0, 2π] or [-π, π]                         │
  │     → RZ(2x) gives full [0, 4π] phase range                     │
  │                                                                  │
  │  2. Standardize features to similar scales BEFORE encoding       │
  │     → Prevents one feature pair from dominating ZZ phases        │
  │                                                                  │
  │  3. Watch for large products: if x₀=10, x₁=10, then             │
  │     ZZ angle = 2·100 = 200 rad → many redundant 2π wraps        │
  │     → Information wasted in periodic rotations                   │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 14. Comparison with Related Encodings

```
  ┌───────────────────┬──────────┬──────────┬────────────┬───────────────┐
  │    Encoding       │  Qubits  │  Depth   │ Entangling │ Hardness      │
  ├───────────────────┼──────────┼──────────┼────────────┼───────────────┤
  │  IQP ★            │    n     │  O(n²)  │    Yes     │ Provably hard │
  │  ZZ Feature Map   │    n     │  O(n²)  │    Yes     │ Same family   │
  │  Pauli Feature Map│    n     │  O(n²)  │    Yes     │ Depends       │
  │  Angle            │    n     │  O(1)    │     No     │ Easy          │
  │  Amplitude        │  log₂n  │  O(2ⁿ)  │    Yes     │ Hard (generic)│
  └───────────────────┴──────────┴──────────┴────────────┴───────────────┘

  IQP vs ZZ Feature Map:
  ┌────────────────────────────────────────────────────────────────┐
  │  Both use H-RZ-ZZ structure. IQP follows the Havlíček et al.  │
  │  convention with 2x scaling in RZ gates. ZZFeatureMap may use  │
  │  different phase conventions. Structurally very similar —      │
  │  the distinction is primarily in parameterization details.     │
  └────────────────────────────────────────────────────────────────┘
```

---

## References

1. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
   feature spaces." Nature, 567(7747), 209-212.

2. Bremner, M. J., Montanaro, A., & Shepherd, D. J. (2016). "Average-case
   complexity versus approximate simulation of commuting quantum
   computations." Physical Review Letters, 117(8), 080501.

3. Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature
   Hilbert spaces." Physical Review Letters, 122(4), 040504.

4. McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
   network training landscapes." Nature Communications, 9, 4812.