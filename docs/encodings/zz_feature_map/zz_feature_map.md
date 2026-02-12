# ZZ Feature Map

A second-order Pauli-Z feature map that encodes classical data into
**entangled quantum states** using pairwise ZZ interactions with the
Qiskit **(pi - x)** phase convention.

```
 ╔══════════════════════════════════════════════════════════════════════╗
 ║                                                                      ║
 ║   |ψ(x)⟩  =  [ U_ZZ(x) · H^⊗n ]^reps  |0⟩^⊗n                     ║
 ║                                                                      ║
 ║   U_ZZ  =  ∏ₖ P(2xₖ)  ·  ∏_{i<j} CNOT · P(2(π-xᵢ)(π-xⱼ)) · CNOT ║
 ║                                                                      ║
 ║   "n features  →  n qubits  →  entangled state  →  quantum kernel"  ║
 ║                                                                      ║
 ╚══════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

The ZZ Feature Map encodes classical data by interleaving three
operations: **superposition** (Hadamard), **data-dependent phases**
(single-qubit P gates), and **pairwise entanglement** (ZZ interactions).
This is repeated `reps` times to build up an expressive quantum state.

```
  Classical Data  (4 features)              Quantum State  (4 qubits)
  ┌──────────────────────────┐              ┌──────────────────────────┐
  │                          │              │                          │
  │  x₀ = 0.5               │              │  Superposition of all    │
  │  x₁ = 1.2               │   Encode     │  2⁴ = 16 basis states    │
  │  x₂ = 0.8   ──────────────────────►    │  with data-dependent     │
  │  x₃ = 2.1               │              │  amplitudes & phases     │
  │                          │              │  ENTANGLED across        │
  └──────────────────────────┘              │  all qubits              │
       4 real numbers                       └──────────────────────────┘
                                                 4 qubits
```

> The key difference from simpler encodings: ZZ Feature Map creates
> **entanglement** between qubits, enabling quantum kernels that can
> separate data classes impossible to distinguish classically.

---

## 2. Circuit Structure — Layer by Layer

Each repetition applies three layers. The full circuit repeats this
structure `reps` times (default: 2).

### Single Repetition (n = 4, full entanglement)

```
  Layer 1          Layer 2              Layer 3: ZZ Interactions
  Hadamard         Phase P(2xᵢ)        CNOT-P-CNOT for each pair
  ─────────        ─────────────        ──────────────────────────────

  q₀: ─[H]──────[P(2x₀)]────●─────────────────●──●─────────────────●──●─────────────────●──
                              │                 │  │                 │  │                 │
  q₁: ─[H]──────[P(2x₁)]──[⊕]─[P(φ₀₁)]─────[⊕]─│─────────────────│──│─────────────────│──
                                                   │                 │  │                 │
  q₂: ─[H]──────[P(2x₂)]─────────────────────────[⊕]─[P(φ₀₂)]───[⊕]─│─────────────────│──
                                                                       │                 │
  q₃: ─[H]──────[P(2x₃)]─────────────────────────────────────────────[⊕]─[P(φ₀₃)]────[⊕]──

  (... continued for pairs (1,2), (1,3), (2,3))

  Where:
    φᵢⱼ  =  2(π - xᵢ)(π - xⱼ)     ← the (π - x) phase convention
```

### Zoomed In: One ZZ Interaction Block

```
  The ZZ(φ) gate between qubits i and j is decomposed as:

      qᵢ: ───●───────────●───
              │           │
      qⱼ: ──[⊕]─[P(φ)]─[⊕]──

  This is equivalent to:   e^{-i φ/2 · ZᵢZⱼ}

  Step-by-step:
  1. CNOT(i→j):  Entangle qubits i and j
  2. P(φ) on j:  Apply data-dependent phase  φ = 2(π-xᵢ)(π-xⱼ)
  3. CNOT(i→j):  Disentangle (partially), leaving correlated phase
```

### Two Full Repetitions (reps = 2)

```
  |0⟩ ─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─ ┐
  |0⟩ ─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─ ├─ |ψ(x)⟩
  |0⟩ ─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─ │
  |0⟩ ─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─ ┘

       ╰── rep 1 ──╯ ╰── rep 2 ──╯

  More repetitions = deeper encoding = richer feature space
  but also more gates, more noise, harder to train.
```

---

## 3. The Mathematics

### The (pi - x) Phase Convention

This is what distinguishes ZZ Feature Map from other IQP-style encodings:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Single-qubit phase:    P(2xᵢ)                                  │
  │                                                                  │
  │  Two-qubit ZZ phase:    φᵢⱼ = 2(π - xᵢ)(π - xⱼ)               │
  │                                                                  │
  │  NOT:  2·xᵢ·xⱼ   (that would be standard IQP)                  │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

### What the (pi - x) Convention Means Physically

```
  Interaction strength as a function of feature values:

  Feature value xᵢ     │   (π - xᵢ)    │  Interaction strength
  ─────────────────────┼────────────────┼──────────────────────
  xᵢ = 0              │   π            │  Maximum  (π²)
  xᵢ = π/2            │   π/2          │  Medium   (π²/4)
  xᵢ = π              │   0            │  ZERO     (vanishes!)
  xᵢ = 3π/2           │   -π/2         │  Medium   (π²/4)
  xᵢ = 2π             │   -π           │  Maximum  (π²)

  The interaction VANISHES when either feature equals π.
  This creates a unique kernel geometry centered around x = π.

  Visualized for two features x₀, x₁:

  Interaction strength  φ₀₁ = 2(π-x₀)(π-x₁)

           x₁
        0  │ strong   strong   zero    strong   strong
       π/2 │ strong   medium   zero    medium   strong
        π  │ zero     zero    ►ZERO◄   zero     zero
      3π/2 │ strong   medium   zero    medium   strong
       2π  │ strong   strong   zero    strong   strong
           └─────────────────────────────────────── x₀
             0       π/2       π      3π/2      2π

  The cross-shaped "dead zone" at x = π is characteristic
  of this encoding and affects which data patterns are
  distinguishable in the quantum feature space.
```

### The Phase Gate P

```
           ┌          ┐
  P(θ) =   │  1    0  │       Applies phase e^{iθ} to the |1⟩ component
           │  0  e^iθ │       while leaving |0⟩ unchanged.
           └          ┘

  P(θ)|0⟩ = |0⟩
  P(θ)|1⟩ = e^{iθ}|1⟩
```

### Full State Expression

```
  For a single repetition:

  |ψ(x)⟩ = U_ZZ(x) · H^⊗n |0⟩^⊗n

  Step 1:  H^⊗n |0⟩^⊗n  =  (1/√2ⁿ) Σ_{z∈{0,1}ⁿ} |z⟩
           Creates equal superposition over all 2ⁿ basis states.

  Step 2:  P(2xᵢ) on each qubit i
           Each basis state |z⟩ picks up phase  exp(2i · Σᵢ xᵢzᵢ)

  Step 3:  ZZ(φᵢⱼ) for each pair (i,j)
           Each basis state |z⟩ picks up phase  exp(i · Σ_{i<j} φᵢⱼ · zᵢ · zⱼ)
           where  φᵢⱼ = 2(π - xᵢ)(π - xⱼ)

  Combined, the state has amplitudes:

                    1
  ⟨z|ψ(x)⟩  =  ────── exp[ i·( Σᵢ 2xᵢzᵢ  +  Σ_{i<j} φᵢⱼ·zᵢzⱼ ) ]
                  √2ⁿ

  Every basis state has EQUAL probability (1/2ⁿ), but DIFFERENT phases.
  The information is stored entirely in the relative phase structure.
```

### The Quantum Kernel

```
  When used in QSVM, the kernel function is:

  K(x, x')  =  |⟨ψ(x') | ψ(x)⟩|²

  This kernel implicitly computes a similarity measure in the
  exponentially large quantum feature space. Two data points
  are "similar" if their encoded states have high overlap.

  The (π - x) convention creates a kernel geometry that differs
  from standard IQP:

  ┌─────────────────────────────────────────────────────────┐
  │  IQP kernel:         depends on  xᵢxⱼ - x'ᵢx'ⱼ        │
  │  ZZ Feature Map:     depends on  (π-xᵢ)(π-xⱼ)          │
  │                                 -(π-x'ᵢ)(π-x'ⱼ)        │
  │                                                         │
  │  The shifted convention can better separate data that    │
  │  clusters near the center (x ≈ π) of the feature range. │
  └─────────────────────────────────────────────────────────┘
```

---

## 4. Entanglement Topologies

The `entanglement` parameter controls which qubit pairs get ZZ
interactions. This is a critical choice for both expressivity and
hardware compatibility.

### Full Entanglement (default)

```
  Every pair of qubits interacts.  n(n-1)/2 pairs.

  n = 4:  6 ZZ pairs

  q₀ ──── q₁            q₀ ● ─ ─ ● q₁
   │ ╲  ╱ │                │ ╲   ╱ │
   │  ╳   │        →       │   ╳   │
   │ ╱  ╲ │                │ ╱   ╲ │
  q₃ ──── q₂            q₃ ● ─ ─ ● q₂

  Pairs: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)

  Pro:  Maximum expressivity, captures ALL pairwise correlations
  Con:  O(n²) gates, requires all-to-all hardware connectivity
```

### Linear Entanglement

```
  Only nearest neighbors interact.  n-1 pairs.

  n = 4:  3 ZZ pairs

  q₀ ─── q₁ ─── q₂ ─── q₃

  Pairs: (0,1) (1,2) (2,3)

  Pro:  O(n) gates, native on most superconducting chips
  Con:  No direct interaction between distant qubits
        (q₀ and q₃ never directly interact)
```

### Circular Entanglement

```
  Nearest neighbors + wrap-around.  n pairs (for n > 2).

  n = 4:  4 ZZ pairs

  q₀ ─── q₁
  │       │
  q₃ ─── q₂

  Pairs: (0,1) (1,2) (2,3) (3,0)

  Pro:  Symmetric — every qubit has exactly 2 neighbors
  Con:  Wrap-around may require SWAP gates on linear hardware
```

### Topology Comparison at a Glance

```
  ┌────────────┬────────────┬──────────────┬─────────────────────────┐
  │ Topology   │ ZZ Pairs   │ CNOTs/rep    │ Hardware Compatibility   │
  ├────────────┼────────────┼──────────────┼─────────────────────────┤
  │ full       │ n(n-1)/2   │ n(n-1)       │ Needs all-to-all or     │
  │            │ (quadratic)│              │ many SWAP gates          │
  ├────────────┼────────────┼──────────────┼─────────────────────────┤
  │ linear     │ n - 1      │ 2(n-1)       │ Native on chain/ladder  │
  │            │ (linear)   │              │ superconducting chips    │
  ├────────────┼────────────┼──────────────┼─────────────────────────┤
  │ circular   │ n          │ 2n           │ Native on ring topology  │
  │            │ (linear)   │              │ hardware                 │
  └────────────┴────────────┴──────────────┴─────────────────────────┘

  For n = 10:
    full:      45 pairs →  90 CNOTs per rep
    linear:     9 pairs →  18 CNOTs per rep   (5x fewer!)
    circular:  10 pairs →  20 CNOTs per rep
```

---

## 5. Example Walkthrough

Encode **x** = [0.5, 1.0] using 2 qubits, reps=1, full entanglement:

```
  Step 1: Initialize
  ──────────────────
  |ψ₀⟩ = |00⟩

  Step 2: Hadamard layer
  ──────────────────────
  H⊗H |00⟩ = ½(|00⟩ + |01⟩ + |10⟩ + |11⟩)

  Step 3: Single-qubit phases  P(2xᵢ)
  ─────────────────────────────────────
  P(2·0.5) on q₀  =  P(1.0)    →  phase e^{i·1.0} on |1_⟩ states
  P(2·1.0) on q₁  =  P(2.0)    →  phase e^{i·2.0} on |_1⟩ states

  |ψ₁⟩ = ½( |00⟩ + e^{i2.0}|01⟩ + e^{i1.0}|10⟩ + e^{i3.0}|11⟩ )

  Step 4: ZZ interaction for pair (0,1)
  ──────────────────────────────────────
  φ₀₁ = 2(π - 0.5)(π - 1.0)
       = 2 × 2.642 × 2.142
       = 2 × 5.658
       ≈ 11.316

  CNOT(0→1), P(φ₀₁) on q₁, CNOT(0→1)
  Only |11⟩ picks up the additional phase e^{iφ₀₁}

  |ψ(x)⟩ = ½( |00⟩ + e^{i2.0}|01⟩ + e^{i1.0}|10⟩ + e^{i(3.0+11.316)}|11⟩ )

  All amplitudes have magnitude ½ — equal probability ¼ for each state.
  The information is encoded entirely in the PHASES.

  Measurement probabilities:
  |00⟩  █████████████████████  0.250
  |01⟩  █████████████████████  0.250
  |10⟩  █████████████████████  0.250
  |11⟩  █████████████████████  0.250
                                ─────
                                1.000

  ⚠ Individual measurements look uniformly random!
    The useful information is in the INTERFERENCE pattern
    when comparing two different encoded states (→ quantum kernel).
```

---

## 6. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    ZZ FEATURE MAP PROPERTIES                        │
  ├──────────────────────┬──────────────────────────────────────────────┤
  │  Qubits required     │  n  (one per feature, linear scaling)       │
  │  Circuit depth       │  reps × (2 + 3 × parallelized_pairs)       │
  │  Repetitions         │  configurable  (default: 2)                 │
  │  Single-qubit gates  │  reps × (n_H + n_P + n_ZZ_P)               │
  │  Two-qubit gates     │  reps × 2 × n_pairs  (CNOTs)               │
  │  Parameter count     │  reps × (n + n_pairs)  (data-dependent)     │
  │  Entangling?         │  Yes  (ZZ creates genuine entanglement)     │
  │  Simulability        │  NOT classically simulable                  │
  │  Trainability        │  ~0.65 (2 reps), decreases with depth      │
  │  Expressibility      │  High  (entanglement + phase encoding)      │
  │  Data type           │  Continuous  (real-valued features)          │
  │  Phase convention    │  2(π - xᵢ)(π - xⱼ)  for ZZ interactions    │
  │  Qiskit compatible   │  Yes  (matches qiskit ZZFeatureMap)         │
  └──────────────────────┴──────────────────────────────────────────────┘
```

---

## 7. Resource Scaling

### Gate Count Formulas

```
  Per repetition:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Hadamard gates:      n                                          │
  │  Phase (single):      n                                          │
  │  Phase (ZZ):          n_pairs                                    │
  │  CNOT gates:          2 × n_pairs                                │
  │                       ──────────                                 │
  │  Total per rep:       2n + 3 × n_pairs                           │
  │  Total:               reps × (2n + 3 × n_pairs)                 │
  └──────────────────────────────────────────────────────────────────┘
```

### Concrete Numbers (reps = 2)

```
  n_feat │ Topology │ Pairs │  H  │  P₁  │ P_ZZ │ CNOT │ Total │ Depth
  ───────┼──────────┼───────┼─────┼──────┼──────┼──────┼───────┼──────
     2   │ full     │   1   │   4 │    4 │    2 │    4 │    14 │   10
     4   │ full     │   6   │   8 │    8 │   12 │   24 │    52 │   22
     4   │ linear   │   3   │   8 │    8 │    6 │   12 │    34 │   22
     4   │ circular │   4   │   8 │    8 │    8 │   16 │    40 │   28
     8   │ full     │  28   │  16 │   16 │   56 │  112 │   200 │   46
     8   │ linear   │   7   │  16 │   16 │   14 │   28 │    74 │   46
    10   │ full     │  45   │  20 │   20 │   90 │  180 │   310 │   58
    10   │ linear   │   9   │  20 │   20 │   18 │   36 │    94 │   58
    16   │ full     │ 120   │  32 │   32 │  240 │  480 │   784 │   94
    16   │ linear   │  15   │  32 │   32 │   30 │   60 │   154 │   94
```

```
  Scaling visualization (full entanglement, reps=2):

  Total gates vs. number of features:

  800 │                                              ╱
      │                                            ╱
  600 │                                         ╱
      │                                       ╱
  400 │                                   ╱
      │                               ╱
  200 │                          ╱
      │                    ╱─ ─
  100 │              ╱─ ─
      │        ╱─ ─
   14 │  ●─ ─
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬── n_features
         2  4  6  8  10 12 14 16

  Full entanglement: O(n²) growth    ← quadratic!
  Linear entanglement: O(n) growth   ← much gentler
```

---

## 8. Depth Analysis

```
  Circuit depth per repetition:

  ┌──────────────────────────────────────────────────────────────────┐
  │  Per rep = single_qubit_depth + zz_block_depth                   │
  │         = 2 + zz_block_depth                                     │
  │                                                                  │
  │  ZZ block depth depends on parallelization of pairs:             │
  │                                                                  │
  │  Linear:    3 × (n-1)       pairs share qubits, sequential      │
  │  Circular:  3 × n           (for n > 2)                          │
  │  Full:      3 × χ           χ = chromatic index of Kₙ           │
  │                              = n-1 (even n), n (odd n)           │
  │                                                                  │
  │  Total depth = reps × (2 + zz_block_depth)                      │
  └──────────────────────────────────────────────────────────────────┘

  Edge coloring of full entanglement (why χ = n-1 for even n):

  K₄ (4 qubits, even):  6 pairs can be colored with 3 colors
  ┌───────────────────────────────────────────────┐
  │  Color 1 (parallel): (0,1) and (2,3)          │
  │  Color 2 (parallel): (0,2) and (1,3)          │
  │  Color 3 (parallel): (0,3) and (1,2)          │
  │  → 3 time steps × 3 depth each = depth 9      │
  └───────────────────────────────────────────────┘

  Non-overlapping pairs can run simultaneously on hardware,
  reducing effective depth from 3×6=18 (sequential) to 9 (parallel).
```

---

## 9. Comparison: ZZ Feature Map vs IQP Encoding

Both are IQP-style circuits, but they use different phase conventions:

```
  ┌───────────────────┬──────────────────────┬──────────────────────┐
  │                   │    ZZ Feature Map     │    IQP Encoding      │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Single-qubit     │  P(2xᵢ)             │  RZ(2xᵢ)            │
  │  phase gate       │                      │                      │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Two-qubit        │  2(π-xᵢ)(π-xⱼ)      │  2·xᵢ·xⱼ            │
  │  ZZ phase         │  (pi - x convention) │  (direct product)    │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Interaction      │  Vanishes at x = π   │  Vanishes at x = 0   │
  │  zero point       │  Max at x = 0, 2π    │  Max at large |x|    │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Kernel center    │  Around x = π        │  Around x = 0        │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Best data range  │  [0, 2π]             │  Arbitrary            │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Origin           │  Qiskit / Havlicek   │  Shepherd & Bremner  │
  │                   │  et al. (2019)       │  (2009)              │
  ├───────────────────┼──────────────────────┼──────────────────────┤
  │  Primary use      │  QSVM kernels        │  Hardness proofs     │
  └───────────────────┴──────────────────────┴──────────────────────┘

  Both produce states that are NOT classically simulable —
  this is the basis for potential quantum advantage.
```

---

## 10. The Repetition Parameter (reps)

```
  reps controls how many times the H + P + ZZ block is applied.

  reps = 1:
  |0⟩─[H]─[P]─╌ZZ╌─
                     → |ψ(x)⟩

  reps = 2 (default):
  |0⟩─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─
                                    → |ψ(x)⟩

  reps = 3:
  |0⟩─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─[H]─[P]─╌ZZ╌─
                                                   → |ψ(x)⟩
```

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                   EFFECT OF REPETITIONS                          │
  ├──────┬──────────────┬──────────────────────┬─────────────────────┤
  │ reps │ Expressivity │ Circuit depth        │ Trainability        │
  ├──────┼──────────────┼──────────────────────┼─────────────────────┤
  │  1   │ Low          │ Shallow              │ High (~0.75)        │
  │  2   │ Good         │ Moderate (default)   │ Good (~0.65)        │
  │  3   │ High         │ Deep                 │ Moderate (~0.55)    │
  │  5+  │ Very high    │ Very deep            │ Low (plateaus)      │
  └──────┴──────────────┴──────────────────────┴─────────────────────┘

  The default of 2 reps is a balance between expressivity and
  trainability, matching the Qiskit convention.

  ⚠ Warning: More reps ≠ always better.
    Deep circuits suffer from:
    1. Barren plateaus (vanishing gradients)
    2. Noise accumulation on real hardware
    3. Longer execution times
```

---

## 11. Data Preprocessing

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    PREPROCESSING GUIDELINES                      │
  │                                                                  │
  │  The (π - x) phase convention has specific geometric meaning.    │
  │  Input scaling matters for kernel quality.                       │
  │                                                                  │
  │  RECOMMENDED RANGES:                                             │
  │                                                                  │
  │    [0, 2π]   Full phase coverage, symmetric interaction          │
  │    [0, π]    Half-range, interactions decrease toward π          │
  │    [-π, π]   Centered at zero, symmetric                         │
  │                                                                  │
  │  PREPROCESSING PIPELINE:                                         │
  │                                                                  │
  │    Raw Data ──► StandardScaler ──► MinMaxScaler([0, 2π]) ──►    │
  │                  (zero mean,        (scale to target range)      │
  │                   unit variance)                                 │
  │                                                                  │
  │  ⚠ Features near x = π have ZERO interaction strength.          │
  │    If your data clusters around π after scaling, the encoding    │
  │    will lose pairwise correlation information.                   │
  │                                                                  │
  │  NOTE: Rotations are 2π-periodic, so values outside [0, 2π]     │
  │  still work but may produce unexpected kernel geometry.          │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 12. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌───────────────────────────┐       ┌───────────────────────────────┐
  │                           │       │                               │
  │  ✓ Entangling circuit    │       │  ✗ O(n²) gates for full      │
  │    Captures pairwise      │       │    entanglement               │
  │    feature correlations   │       │    (quadratic scaling)         │
  │                           │       │                               │
  │  ✓ Not classically       │       │  ✗ Hardware connectivity      │
  │    simulable              │       │    Full topology needs         │
  │    (potential advantage)  │       │    all-to-all or SWAP gates   │
  │                           │       │                               │
  │  ✓ Qiskit compatible     │       │  ✗ Phase convention makes     │
  │    Drop-in replacement    │       │    features near π invisible  │
  │    for qiskit ZZFeatureMap│       │    to pairwise interactions   │
  │                           │       │                               │
  │  ✓ Configurable topology │       │  ✗ Noise-sensitive            │
  │    Full / linear /        │       │    Many two-qubit gates       │
  │    circular options       │       │    accumulate errors on NISQ  │
  │                           │       │                               │
  │  ✓ Proven in literature  │       │  ✗ Barren plateaus at         │
  │    Havlicek et al. (2019) │       │    high reps / large n        │
  │    demonstrated quantum   │       │    Gradients vanish           │
  │    advantage on real HW   │       │    exponentially              │
  │                           │       │                               │
  │  ✓ Rich quantum kernel   │       │  ✗ Equal-probability states   │
  │    Separates non-linear   │       │    Info is only in phases,    │
  │    data distributions     │       │    not extractable from       │
  │                           │       │    single-shot measurements   │
  └───────────────────────────┘       └───────────────────────────────┘
```

---

## 13. Use Cases

```
                          Best suited for
                    ┌──────────────────────────────┐
                    │                              │
  ┌─────────────────┤  Quantum Kernel Methods      │  QSVM classification
  │                 │  (QSVM)                      │  with quantum advantage
  │                 ├──────────────────────────────┤
  │                 │                              │
  ├─────────────────┤  Quantum Neural Networks     │  Feature embedding
  │                 │  (QNN / VQC)                 │  before variational
  │                 ├──────────────────────────────┤  layers
  │                 │                              │
  ├─────────────────┤  Quantum Feature Space       │  Kernel-based
  │                 │  Exploration                 │  similarity in
  │                 ├──────────────────────────────┤  exponential space
  │                 │                              │
  ├─────────────────┤  Cross-Platform              │  Qiskit-compatible
  │                 │  Benchmarking                │  standard circuit
  │                 ├──────────────────────────────┤
  │                 │                              │
  └─────────────────┤  Quantum Advantage           │  Proven separation
                    │  Research                    │  on ad hoc datasets
                    └──────────────────────────────┘
```

---

## 14. Comparison with Other Encodings

```
  ┌───────────────────┬──────────┬──────────┬────────────┬───────────────┐
  │    Encoding       │  Qubits  │  Depth   │ Entangling │ Simulability  │
  ├───────────────────┼──────────┼──────────┼────────────┼───────────────┤
  │  ZZ Feature Map ★ │    n     │  O(n²)  │    Yes     │  Not sim.     │
  │  IQP              │    n     │  O(n²)  │    Yes     │  Not sim.     │
  │  Pauli Feature Map│    n     │  O(n²)  │    Yes     │  Not sim.     │
  │  Angle            │    n     │  O(1)    │     No     │  Simulable    │
  │  Basis            │    n     │  O(1)    │     No     │  Simulable    │
  │  Amplitude        │  log₂n  │  O(2ⁿ)  │    Yes     │  Not sim.     │
  └───────────────────┴──────────┴──────────┴────────────┴───────────────┘

  Key insight:
  ┌────────────────────────────────────────────────────────────────────┐
  │  ZZ Feature Map sits at the intersection of:                      │
  │  • Practical (n qubits, not log₂n) — works on real hardware       │
  │  • Expressive (entangling) — captures pairwise correlations       │
  │  • Standard (Qiskit-compatible) — reproducible across platforms   │
  │  • Theoretically grounded (Havlicek et al.) — proven advantage    │
  └────────────────────────────────────────────────────────────────────┘
```

---

## 15. Hardware Considerations

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                   TOPOLOGY → HARDWARE MAPPING                    │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  FULL entanglement:                                              │
  │    Requires all-to-all qubit connectivity.                       │
  │    Most superconducting chips don't have this natively.          │
  │    Transpiler inserts SWAP gates → actual depth >> theoretical.  │
  │                                                                  │
  │    Native on:  Trapped-ion (IonQ, Quantinuum) — full connect.   │
  │    Needs SWAPs: IBM (heavy-hex), Google (Sycamore grid)         │
  │                                                                  │
  │  LINEAR entanglement:                                            │
  │    Nearest-neighbor chain — native on almost all hardware.       │
  │    No SWAP overhead.  Best choice for NISQ experiments.          │
  │                                                                  │
  │    Native on:  IBM, Google, Rigetti, IonQ                       │
  │                                                                  │
  │  CIRCULAR entanglement:                                          │
  │    Ring topology — native on ring-connected architectures.       │
  │    The single wrap-around edge may need 1 SWAP on linear chips. │
  │                                                                  │
  │    Native on:  Ring topologies, some IBM layouts                │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  Recommendation:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Simulation / Theory → use 'full'  (maximum expressivity)       │
  │  Real hardware (NISQ) → use 'linear'  (minimal error)          │
  │  Ring-topology chips  → use 'circular'  (balanced)              │
  └──────────────────────────────────────────────────────────────────┘
```

---

## References

1. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
   feature spaces." Nature, 567(7747), 209-212.

2. Qiskit Development Team. "Qiskit Machine Learning: ZZFeatureMap."
   https://qiskit.org/documentation/machine-learning/

3. Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature
   Hilbert spaces." Physical Review Letters, 122(4), 040504.