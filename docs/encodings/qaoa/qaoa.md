# QAOA-Style Encoding

> A quantum data encoding that adapts the structure of the Quantum Approximate
> Optimization Algorithm (QAOA) for classical data embedding -- alternating
> data-dependent "cost" rotations with "mixer" operations and entanglement.

---

## Overview

QAOA encoding borrows the layered cost-mixer architecture from the original QAOA
algorithm and repurposes it as a feature map. Instead of encoding a combinatorial
problem Hamiltonian, the **cost layer encodes classical data** through
parameterized rotations, while the **mixer layer** creates inter-qubit
correlations via fixed rotations and entangling gates.

```
  Standard QAOA                          QAOA-Style Encoding
  ─────────────                          ────────────────────
  e^{-i gamma H_C}  (cost)       --->   R_data(gamma * x_i)   (data encoding)
  e^{-i beta  H_M}  (mixer)      --->   R_mixer(beta) + ENT   (mixing + entanglement)
  applied p times                 --->   applied reps times
```

### Mathematical Definition

```
  |psi(x)> =  prod_{p=1}^{reps}  U_M(beta) . U_C(gamma, x)  |+>^{tensor n}
                                                                  ^
                                                         (optional Hadamard init)
  where:
    U_C(gamma, x) =  tensor_i  R_data( phi(x_i) )       data-dependent rotations
    U_M(beta)     =  U_ent  .  tensor_i  R_mixer(beta)   mixer rotations + entanglement

    phi(x_i) = gamma * x_i          (linear feature map)
             = gamma * x_i^2        (quadratic feature map)
```

---

## Circuit Structure (4 qubits, defaults: RZ data, RX mixer, CZ, linear, reps=2)

```
              INIT       ┌──────── Rep 1 ────────┐┌──────── Rep 2 ────────┐
              │          │ COST    MIXER    ENT   ││ COST    MIXER    ENT   │
              │          │                        ││                        │

q0: |0> ──H──RZ(gx0)──RX(b)──*─────────RZ(gx0)──RX(b)──*─────────
                              |                          |
q1: |0> ──H──RZ(gx1)──RX(b)──*──*──────RZ(gx1)──RX(b)──*──*──────
                                 |                          |
q2: |0> ──H──RZ(gx2)──RX(b)─────*──*───RZ(gx2)──RX(b)─────*──*───
                                    |                          |
q3: |0> ──H──RZ(gx3)──RX(b)────────*───RZ(gx3)──RX(b)────────*───
```

**Reading the diagram:**
- `H` = Hadamard gate (creates initial `|+>` superposition)
- `RZ(gx_i)` = data rotation: `RZ(gamma * x_i)` encoding feature `x_i`
- `RX(b)` = mixer rotation: `RX(beta)` with fixed angle `beta`
- `*──*` = CZ (controlled-Z) gate between adjacent qubits

---

## Anatomy of One Repetition

Each of the `reps` repetitions contains **three sub-layers** applied in sequence:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                     ONE REPETITION (p)                          │
  │                                                                 │
  │  1. DATA ENCODING          2. MIXER ROTATION     3. ENTANGLING │
  │  (Cost Layer)              (Mixer Layer)          (Mixer Layer) │
  │                                                                 │
  │  q0: R_data(phi(x0))      q0: R_mixer(beta)      q0─*         │
  │  q1: R_data(phi(x1))      q1: R_mixer(beta)      q1─*─*       │
  │  q2: R_data(phi(x2))      q2: R_mixer(beta)      q2───*─*     │
  │  q3: R_data(phi(x3))      q3: R_mixer(beta)      q3─────*     │
  │                                                                 │
  │  Data-dependent            Fixed angle             Entangling   │
  │  Parallel on all qubits    Parallel on all qubits  gates        │
  └─────────────────────────────────────────────────────────────────┘
```

Note the asymmetry: the **data layer uses input features** (different angle per
qubit), while the **mixer layer uses a fixed angle** `beta` (same for all qubits).

---

## Parameters: gamma and beta

The encoding has two key scaling hyperparameters inherited from QAOA:

```
  gamma (default=1.0)                  beta (default=1.0)
  ──────────────────                   ─────────────────
  Scales data rotations:               Scales mixer rotations:
    angle = gamma * x_i                  angle = beta  (constant)
    (or gamma * x_i^2 if quadratic)

  Effect of gamma:                     Effect of beta:
  ┌───────────────────────────┐       ┌───────────────────────────┐
  │ gamma << 1: small angles, │       │ beta << 1: weak mixing,   │
  │   features clustered near │       │   states close to data    │
  │   |+> state               │       │   encoding subspace       │
  │                           │       │                           │
  │ gamma ~ 1:  balanced      │       │ beta ~ 1:  balanced       │
  │   encoding (recommended)  │       │   mixing  (recommended)   │
  │                           │       │                           │
  │ gamma >> 1: large angles, │       │ beta >> 1: strong mixing, │
  │   aliasing due to 2pi     │       │   mixer dominates the     │
  │   periodicity             │       │   encoding structure      │
  └───────────────────────────┘       └───────────────────────────┘
```

---

## Feature Maps

Two mappings from classical features to rotation angles:

```
  "linear" (default)                 "quadratic"
  ──────────────────                 ───────────────
  phi(x_i) = gamma * x_i            phi(x_i) = gamma * x_i^2

  Linear in feature space.           Nonlinear -- amplifies large
  Preserves relative spacing         feature values, compresses
  between data points.               small ones.

  x:     -1   0   1   2             x:     -1   0   1   2
  phi:   -g   0   g   2g            phi:    g   0   g   4g
          |   |   |   |                     |   |   |   |
  ────────┼───┼───┼───┼─────        ────────┼───┼───┼───────┼──
         -g   0   g   2g                    0   g       4g
```

---

## Rotation Axis Choices

### Data Rotation (default: Z)

```
  Axis  |  Gate   |  Effect on |+> state              |  Typical Use
  ──────+─────────+──────────────────────────────────+──────────────────
  "Z"   |  RZ(x)  |  Phase rotation (diagonal H_C)    |  Standard QAOA
  "Y"   |  RY(x)  |  Amplitude + phase rotation       |  Amplitude-sensitive
  "X"   |  RX(x)  |  Bit-flip rotation                |  Alternative basis
```

### Mixer Rotation (default: X)

```
  Axis  |  Gate   |  Effect                            |  Typical Use
  ──────+─────────+────────────────────────────────────+──────────────────
  "X"   |  RX(b)  |  Standard QAOA mixer (bit-flip)    |  Default choice
  "Y"   |  RY(b)  |  Amplitude mixing                  |  Alternative mixer
  "Z"   |  RZ(b)  |  Phase-only mixing                 |  Phase manipulation
```

The **standard QAOA pairing** is `data_rotation="Z"` + `mixer_rotation="X"`,
mirroring the diagonal cost Hamiltonian and transverse-field mixer of the
original algorithm.

---

## Entangling Gate Options

Three two-qubit gates are available for the mixer entanglement layer:

```
  "cz" (default)          "cx" (CNOT)              "rzz" (parameterized)
  ──────────────          ───────────              ────────────────────
  Controlled-Z            Controlled-NOT           RZZ(beta) = exp(-i*b/2 Z*Z)

  q0 ──*──                q0 ──@──                 q0 ──ZZ(b)──
       |                       |                        |
  q1 ──*──                q1 ──X──                 q1 ──ZZ(b)──

  - Symmetric (no          - Asymmetric (control/   - Parameterized by beta
    control/target            target distinction)    - Data-dependent strength
    distinction)           - Widely available        - Richer feature space
  - Native on many HW     - Non-commuting: gates    - Commuting: gates can
  - Commuting: gates        must be sequential        be parallelized
    can be parallelized
```

**Impact on circuit depth:** CZ and RZZ gates commute and can be parallelized
into fewer layers. CX (CNOT) gates do not commute and are applied sequentially,
resulting in deeper circuits.

---

## Entanglement Topologies

Four connectivity patterns, including a "none" option unique to this encoding:

### None (product state)

```
  q0 ──(no entanglement)──
  q1 ──(no entanglement)──
  q2 ──(no entanglement)──
  q3 ──(no entanglement)──

  Pairs: (none)
  Count: 0
  Use: Baseline / ablation studies
```

### Linear

```
  q0 ──*
       |
  q1 ──*──*
          |
  q2 ─────*──*
             |
  q3 ────────*

  Pairs: (0,1), (1,2), (2,3)
  Count: n - 1
  Depth: min(2, n-1) with parallel scheduling
```

### Circular

```
  q0 ──*──────────*
       |          |
  q1 ──*──*       |
          |       |
  q2 ─────*──*    |
             |    |
  q3 ────────*──*─┘

  Pairs: (0,1), (1,2), (2,3), (3,0)
  Count: n  (n>2), else n-1
  Depth: 2 (n even), 3 (n odd)  -- by edge coloring of cycle graph C_n
```

### Full

```
  q0 ──*──*──*
       |  |  |
  q1 ──*  |  |──*──*
          |  |  |  |
  q2 ─────*  |──*  |──*
             |     |  |
  q3 ────────*─────*──*

  Pairs: all (i,j) where i < j
  Count: n(n-1)/2
  Depth: n-1 (n even), n (n odd)  -- by Vizing's theorem on K_n
```

### Topology Comparison

```
  Topology  | Gates/layer | Parallel Depth      | Hardware Match
  ──────────+─────────────+─────────────────────+──────────────────────
  none      |   0         | 0                   | Any (no connectivity needed)
  linear    |   n - 1     | min(2, n-1)         | Superconducting chains
  circular  |   n         | 2 or 3              | Ring architectures
  full      |  n(n-1)/2   | n-1 or n            | Ion traps (all-to-all)
```

---

## Initial State: Hadamard Layer

```
  include_initial_h=True (default)       include_initial_h=False
  ────────────────────────────           ──────────────────────────
  |0> ──H── ... (encoding) ...          |0> ── ... (encoding) ...

  Starts from |+>^n                      Starts from |0>^n
  (uniform superposition)                (computational basis)

  - Standard in QAOA                     - Useful when composing
  - Provides symmetric starting            with other circuits
    point for all features               - Saves n gates of depth
```

---

## Depth Calculation

The depth formula accounts for optimal parallel gate scheduling:

```
  depth = H_layer + reps * (data_layer + mixer_layer + entangle_layer)

  where:
    H_layer       = 1 if include_initial_h, else 0
    data_layer    = 1  (all R_data gates in parallel)
    mixer_layer   = 1  (all R_mixer gates in parallel)
    entangle_layer:
      none     -> 0
      linear   -> min(2, n-1)
      circular -> 2 (n even) or 3 (n odd)
      full     -> n-1 (n even) or n (n odd)
```

**Example:** `n=4, reps=2, linear, include_initial_h=True`

```
  depth = 1 + 2 * (1 + 1 + min(2, 3))
        = 1 + 2 * (1 + 1 + 2)
        = 1 + 2 * 4
        = 9
```

---

## Resource Scaling

For `n` qubits, `r` repetitions, and `p` entanglement pairs per layer:

```
  Resource              |  Formula                         | Example (n=4, r=2, linear)
  ──────────────────────+──────────────────────────────────+──────────────────────────
  Qubits                |  n                               |  4
  Hadamard gates        |  n  (if include_initial_h)       |  4
  Data rotation gates   |  n * r                           |  8
  Mixer rotation gates  |  n * r                           |  8
  Entangling gates      |  p * r                           |  6
  Total single-qubit    |  n*(1 + 2r) or n*2r              |  20 or 16
  Total two-qubit       |  p * r                           |  6
  Total gates           |  single + two-qubit              |  26
  Data parameters       |  n * r  (angles from input data) |  8
  Trainable parameters  |  0  (gamma, beta are fixed)      |  0

  Entanglement pairs (p) per layer:
    none:     0
    linear:   n - 1                                          3
    circular: n (if n>2)                                     4
    full:     n(n-1)/2                                       6
```

---

## Connection to Original QAOA

The encoding mirrors the QAOA circuit but replaces problem-specific elements
with data-dependent operations:

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │   QAOA for Optimization            QAOA-Style Encoding              │
  │   ─────────────────────            ────────────────────             │
  │                                                                      │
  │   |+>^n                            |+>^n  (or |0>^n)                │
  │     |                                |                               │
  │   e^{-i gamma_p H_C}              R_data(gamma * x_i)              │
  │   (diagonal cost unitary)          (data-dependent rotations)       │
  │     |                                |                               │
  │   e^{-i beta_p H_M}               R_mixer(beta) + Entangling       │
  │   (transverse field mixer)         (fixed mixer + entanglement)     │
  │     |                                |                               │
  │   Repeat p times                   Repeat reps times                │
  │     |                                |                               │
  │   Measure in Z basis               Use as feature map / kernel      │
  │                                                                      │
  │   gamma_p, beta_p: LEARNED         gamma, beta: HYPERPARAMETERS     │
  │   H_C: problem-specific            x_i: classical input data        │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

Key difference: in standard QAOA, `gamma_p` and `beta_p` are **variational
parameters optimized during training**. In QAOA-style encoding, `gamma` and
`beta` are **fixed hyperparameters** -- the data `x` provides the variability.

---

## Key Properties

```
  Property                |  Value / Behavior
  ────────────────────────+───────────────────────────────────────────
  Entangling?             |  Yes (unless entanglement='none')
  Simulability            |  Not classically simulable (with entanglement)
  Trainable parameters    |  0 (gamma, beta are hyperparameters)
  Data re-uploading       |  Yes (same features re-encoded each rep)
  Initial state           |  |+>^n (default) or |0>^n
  Feature-to-qubit ratio  |  1:1 (one qubit per feature)
  Backends supported      |  PennyLane, Qiskit, Cirq
  Serializable            |  Yes (to_dict / from_dict, pickle)
```

---

## Practical Considerations

### Data Preprocessing

```
  Rotation gates are 2pi-periodic:  R(x) = R(x + 2pi)

  Recommended:
    - Scale features to [0, 2pi] or [-pi, pi]
    - Standardize if feature magnitudes differ
    - gamma * x values beyond +/- 2pi trigger warnings
    - For quadratic map: beware x^2 amplification of large values
```

### Choosing gamma and beta

```
  Start with defaults (gamma=1.0, beta=1.0), then:

  1. If features are pre-normalized to [0, 1]:
     -> gamma = pi or 2*pi to span the full rotation range

  2. If underfitting: increase gamma (spreads states further apart)

  3. If overfitting or noisy: decrease beta (weaker mixing)

  4. For quadratic feature map: use smaller gamma since x^2 amplifies
```

### Choosing reps

```
  reps=1     Minimal expressibility, minimal noise
  reps=2     Default -- good balance for NISQ          <-- start here
  reps=3-5   Higher expressibility, moderate depth
  reps>10    Risk of barren plateaus (warning issued)
```

### Hardware Recommendations

```
  Hardware Type          |  Recommended Config
  ───────────────────────+───────────────────────────────────────────
  IBM / Google / Rigetti |  entanglement="linear", entangling_gate="cz"
  (superconducting)      |  CZ is native; linear matches chip topology
                         |
  IonQ / Quantinuum      |  entanglement="full", entangling_gate="rzz"
  (ion trap)             |  All-to-all native; RZZ adds expressivity
                         |
  Simulators             |  entanglement="full", any gate
                         |  Maximize expressibility (no HW constraints)
                         |
  Noisy environments     |  entanglement="none" or "linear", low reps
                         |  Minimize gate count and depth
```

---

## Expressibility Hierarchy

The expressibility of the encoding increases along multiple axes:

```
  Less Expressive                               More Expressive
  ─────────────────────────────────────────────────────────────

  Entanglement:   none  <  linear  <  circular  <  full

  Reps:           1     <    2     <    3       <   ...

  Gate type:      cx/cz    <    rzz  (parameterized adds richness)

  Feature map:    linear   <    quadratic  (nonlinear kernel)

  ┌───────────────────────────────────────────────────────────┐
  │  WARNING: Higher expressibility does NOT always mean      │
  │  better performance. On NISQ hardware, noise and barren   │
  │  plateaus can negate the benefits of deeper circuits.     │
  └───────────────────────────────────────────────────────────┘
```

---

## Strengths and Limitations

### Strengths

- **Rich parameter space** -- 9 configurable hyperparameters allow fine-tuning
  for specific tasks and hardware
- **QAOA heritage** -- well-studied theoretical foundation from combinatorial
  optimization
- **Flexible entanglement** -- four topologies including "none" for ablation
- **Feature map choice** -- linear or quadratic nonlinearity
- **Three entangling gates** -- CX, CZ (symmetric), RZZ (parameterized)
- **Serializable** -- `to_dict` / `from_dict` for experiment reproducibility

### Limitations

- **Many hyperparameters** -- gamma, beta, reps, rotations, entanglement,
  gate type, feature map all interact; tuning can be complex
- **Fixed gamma/beta** -- unlike trainable QAOA, these are not learned
- **O(n^2) scaling** -- full entanglement becomes impractical beyond ~12 qubits
- **Barren plateaus** -- deep circuits (reps > 10) risk vanishing gradients
- **Not universal** -- for universal approximation, consider Data Reuploading

---

## Comparison with Related Encodings

```
  Encoding             |  Cost Layer    |  Mixer Layer    |  Entangling Gate
  ─────────────────────+────────────────+─────────────────+──────────────────
  QAOA-Style           |  R_data(g*x)   |  R_mixer(b)+ENT |  CX / CZ / RZZ
  Hardware-Efficient   |  R_alpha(x)    |  (none)         |  CNOT only
  ZZ Feature Map       |  RZ(x)         |  (none)         |  RZZ(x_i*x_j)
  Pauli Feature Map    |  R_pauli(x)    |  (none)         |  Pauli rotations
  Data Reuploading     |  R(x) + R(w)   |  trainable      |  Various

  Unique to QAOA-Style:
    - Separate cost & mixer layers (inspired by QAOA algorithm structure)
    - Fixed-angle mixer (beta) independent of data
    - Optional initial Hadamard for |+> start state
    - "none" entanglement option for product-state baseline
    - Quadratic feature map option
```

---

## References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate
   Optimization Algorithm." arXiv:1411.4028.

2. Hadfield, S., et al. (2019). "From the Quantum Approximate Optimization
   Algorithm to a Quantum Alternating Operator Ansatz." Algorithms, 12(2), 34.

3. Lloyd, S., et al. (2020). "Quantum embeddings for machine learning."
   arXiv:2001.03622.

4. Schuld, M., & Petruccione, F. (2021). "Machine Learning with Quantum
   Computers." Springer.