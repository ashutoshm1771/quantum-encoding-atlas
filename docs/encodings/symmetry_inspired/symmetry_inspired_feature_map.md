# Symmetry-Inspired Feature Map

A **heuristic** quantum data encoding that incorporates symmetry-aware gate
structures inspired by geometric quantum machine learning — providing
**inductive bias** for problems with underlying symmetries without the
complexity of mathematically rigorous equivariant constructions.

```
 ╔════════════════════════════════════════════════════════════════════════╗
 ║                                                                        ║
 ║   |psi(x)> = [ H^n . RY(f(x)) . RZ(phi(x)) . U_ent(x) ]^reps |0>^n  ║
 ║                                                                        ║
 ║   "Hadamards superpose, RY encodes features,                           ║
 ║    RZ applies symmetry-invariant rotations,                            ║
 ║    entangling gates preserve symmetry structure"                       ║
 ║                                                                        ║
 ╚════════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

Most quantum feature maps encode data without regard for the structure of
the problem. If your data has a **known symmetry** — rotational patterns,
cyclic periodicity, mirror symmetry, or permutation invariance — a generic
encoding wastes capacity learning what symmetry should already provide for free.

SymmetryInspiredFeatureMap addresses this by computing gate angles from
**symmetry-invariant functions** of the input, so that the encoding itself
carries inductive bias aligned with the problem's geometry.

```
  Classical data          Symmetry-invariant         Quantum state with
  with structure    --->  angle computation     --->  symmetry-aware
  x = [x0, ..., xn]      (radius, mean, mirror)      encoding |psi(x)>

                ┌─────────────────────────────────┐
                │                                 │
                │    1. Hadamard layer            │  Superposition
                │       H on all qubits           │
                │                                 │
                ├─────────────────────────────────┤
                │                                 │
                │    2. Encoding layer            │  Features -> angles
                │       RY(f(xi)) per qubit       │  via feature map
                │                                 │
                ├─────────────────────────────────┤
                │                                 │
                │    3. Equivariant layer          │  Symmetry-invariant
                │       RZ(phi(x, i)) per qubit   │  rotations
                │                                 │
                ├─────────────────────────────────┤
                │                                 │
                │    4. Entanglement layer         │  Symmetry-specific
                │       2-qubit gates on pairs    │  entangling gates
                │                                 │
                └────────────┬────────────────────┘
                             │
                        Repeat x reps
                             │
                             v
                    Encoded |psi(x)>
```

> **Key distinction from true equivariant encodings:** This encoding does
> NOT satisfy U(g)|psi(x)> = |psi(g.x)> for group elements g. It uses
> symmetry-motivated heuristics that provide useful inductive bias at much
> lower implementation complexity.

---

## 2. Symmetry Types

The encoding supports four symmetry-aware configurations, each computing
gate angles differently to respect a specific symmetry structure.

### Overview

```
  ┌──────────────┬────────────┬───────────────────────────────────────────┐
  │  Symmetry    │  Group     │  What It Respects                        │
  ├──────────────┼────────────┼───────────────────────────────────────────┤
  │  rotation    │  SO(2)     │  Pairs (xi, xi+1) treated as 2D coords  │
  │              │            │  Angle: radius = sqrt(xi^2 + xi+1^2)    │
  ├──────────────┼────────────┼───────────────────────────────────────────┤
  │  cyclic      │  Z_n       │  Cyclic inductive bias via topology     │
  │              │            │  and (π-x) interaction terms             │
  │              │            │  Angle: feature value directly (NOT      │
  │              │            │  invariant to cyclic permutations)       │
  ├──────────────┼────────────┼───────────────────────────────────────────┤
  │  reflection  │  Z_2       │  Mirror symmetry (reversal of features) │
  │              │            │  Angle: (xi + x_{n-1-i}) / 2            │
  ├──────────────┼────────────┼───────────────────────────────────────────┤
  │  full        │  S_2       │  Permutation symmetry on pairs          │
  │              │            │  Angle: xi + mean(x)                    │
  └──────────────┴────────────┴───────────────────────────────────────────┘
```

### Rotation Symmetry (SO(2)-inspired)

Treats consecutive pairs of features as 2D coordinates (x, y). The
equivariant angle uses the **radius** — an SO(2)-invariant quantity.

```
  Features:     x = [x0, x1, x2, x3]
  Pairs:           (x0,x1)   (x2,x3)

  Equivariant angle for qubit i:

       phi_i = sqrt( x_{2k}^2 + x_{2k+1}^2 )      k = i // 2
                └─── radius (rotation-invariant) ───┘

  Interaction angle for pair (i, j):

       theta_{ij} = sqrt( f(xi)^2 + f(xj)^2 )     (hypot for stability)


  Visual — what radius invariance means:

       y                  Rotating (x,y) by any angle alpha
       ^                  does NOT change the radius r.
       |   . (x,y)
       |  /               (x', y') = R(alpha) . (x, y)
       | / r              r' = sqrt(x'^2 + y'^2) = r
       |/
       +---------> x      The gate angle phi = r is the same
                          regardless of rotation.

  Constraint: n_features must be EVEN (for coordinate pairs).
  Entanglement: ALWAYS uses coordinate pairs (0,1), (2,3), ...
                regardless of the entanglement parameter.
```

### Cyclic Symmetry (Z_n-inspired)

Designed for data with circular/periodic structure where feature i and
feature (i+1) mod n should be treated similarly. The cyclic bias comes
from the entanglement topology and (pi - x) interaction terms, **not**
from cyclic-invariant angle computation. The entanglement topology is
controlled by the `entanglement` parameter (not forced to circular).

```
  Features:    x = [x0, x1, x2, x3]

  Equivariant angle for qubit i:

       phi_i = x_i                (direct feature value — NOT cyclic-invariant)

  Interaction angle for pair (i, j):

       theta_{ij} = (pi - f(xi)) * (pi - f(xj))

                    └── symmetric product centered at pi ──┘


  Visual — cyclic structure:

          x0
        /    \          Features are arranged in a ring.
      x3      x1        Shifting all indices by 1 maps
        \    /          x0->x1->x2->x3->x0.
          x2

  The interaction uses (pi - x) factors so that the product
  is symmetric around the midpoint of the angle range.
```

### Reflection Symmetry (Z_2-inspired)

Pairs features from opposite ends: feature i with feature (n-1-i).
The equivariant angle is the average of mirror partners.

```
  Features:    x = [x0, x1, x2, x3]
  Mirror pairs:   (x0, x3)  (x1, x2)

  Equivariant angle for qubit i:

       phi_i = (x_i + x_{n-1-i}) / 2

               └── average of mirror partners ──┘


  Visual — mirror pairing:

       x0    x1  |  x2    x3
        \    /   |   \    /
         \/      |    \/
         /\   mirror  /\
        /    \   |   /    \
       x3    x2  |  x1    x0

       phi_0 = (x0 + x3) / 2     phi_2 = (x2 + x1) / 2
       phi_1 = (x1 + x2) / 2     phi_3 = (x3 + x0) / 2

  Notice: phi_i = phi_{n-1-i}  (mirror partners get identical angles)

  Interaction angle:  theta_{ij} = f(xi) * f(xj)
```

### Full Symmetry (S_2-inspired)

Permutation symmetry using symmetric polynomials. Each qubit's angle
depends on both its own feature AND the global mean.

```
  Features:    x = [x0, x1, x2, x3]

  Equivariant angle for qubit i:

       phi_i = x_i + mean(x)

               └── individual + global ──┘


  The mean(x) term is a symmetric polynomial (invariant under
  ANY permutation of features), providing a shared "context"
  that every qubit receives.

  Interaction angle:  theta_{ij} = f(xi) * f(xj)
```

---

## 3. Circuit Structure

### Single Repetition (4 qubits, rotation symmetry, linear entanglement)

```
         Layer 1          Layer 2         Layer 3         Layer 4
        (Hadamard)       (Encoding)     (Equivariant)   (Entanglement)

  |0> ──── H ──────── RY(f(x0)) ──── RZ(phi_0) ────●──────────────────
                                                     │ CRZ(theta_01)
  |0> ──── H ──────── RY(f(x1)) ──── RZ(phi_1) ────⊕──────────────────
                                                        (pair 0,1)
  |0> ──── H ──────── RY(f(x2)) ──── RZ(phi_2) ────●──────────────────
                                                     │ CRZ(theta_23)
  |0> ──── H ──────── RY(f(x3)) ──── RZ(phi_3) ────⊕──────────────────
                                                        (pair 2,3)

  where:
    f(xi)     = feature mapping (angle, fourier, or polynomial)
    phi_i     = symmetry-specific equivariant angle (here: radius)
    theta_ij  = interaction angle from symmetry-aware combination
```

### Entanglement Layer by Symmetry Type

Each symmetry type uses a different entangling gate structure:

```
  ROTATION — CRZ gates on coordinate pairs:
  ──────────────────────────────────────────

    qi ──●─────────         CRZ preserves the rotation
         │ CRZ(theta)       symmetry of the coordinate pair.
    qj ──⊕─────────         Only pairs (0,1), (2,3), ... are coupled.


  CYCLIC — CNOT-RZ-CNOT sandwich:
  ────────────────────────────────

    qi ──●────────────●──   The CNOT pair creates temporary
         │            │     entanglement, RZ encodes the
    qj ──⊕── RZ(theta) ──⊕──   symmetric product interaction.


  REFLECTION — CZ + symmetric RZ:
  ────────────────────────────────

    qi ──●── RZ(theta) ──   CZ creates a phase flip conditioned
         │                  on both qubits being |1>.
    qj ──●── RZ(theta) ──   Both qubits get the SAME RZ angle
                            (preserving mirror symmetry).

  Note: both qi and qj receive identical RZ(theta) — this is
  deliberate, ensuring the gate pattern is symmetric under
  reflection (swapping qi <-> qj produces the same result).


  FULL — Permutation-inspired CNOT-RY structure:
  ───────────────────────────────────────────────

    qi ──●──────────────────⊕── RY(-theta/2) ──●──
         │                  │                   │
    qj ──⊕── RY(theta) ──●──────────────────⊕──

         CNOT      RY      CNOT    RY         CNOT

    A SWAP-like structure using 3 CNOTs + 2 RY gates
    that respects permutation symmetry of qubit pairs.
```

### Multiple Repetitions (reps = 2)

```
  ┌──────────────────────────────────┐ ┌──────────────────────────────────┐
  │  H ─ RY(f(x)) ─ RZ(phi) ─ Ent  │ │  H ─ RY(f(x)) ─ RZ(phi) ─ Ent  │
  │            Rep 1                 │ │            Rep 2                 │
  └──────────────────────────────────┘ └──────────────────────────────────┘
                  │                                    │
        Creates initial                      Re-applies the same
        symmetry-aware                       structure, building
        encoded state                        deeper entanglement
                                             and expressibility
```

> Each repetition applies the **same angle values** (they depend only on
> the input x, not on trainable parameters). Additional reps increase
> circuit depth and expressibility at the cost of trainability.

---

## 4. Feature Mapping Functions

The `feature_map` parameter controls how raw feature values become
rotation angles before entering the circuit.

```
  ┌──────────────┬─────────────────┬──────────────────┬──────────────────┐
  │  Mapping     │  Formula        │  Recommended     │  Use Case        │
  │              │                 │  Input Range     │                  │
  ├──────────────┼─────────────────┼──────────────────┼──────────────────┤
  │  angle       │  theta = x      │  [0, pi]         │  General purpose │
  │  (default)   │                 │                  │  direct mapping  │
  ├──────────────┼─────────────────┼──────────────────┼──────────────────┤
  │  fourier     │  theta = 2*pi*x │  [0, 1]          │  Periodic data,  │
  │              │                 │                  │  phase values    │
  ├──────────────┼─────────────────┼──────────────────┼──────────────────┤
  │  polynomial  │  theta = x + x² │  [-1, 1]         │  Non-linear      │
  │              │                 │                  │  feature mixing  │
  └──────────────┴─────────────────┴──────────────────┴──────────────────┘

  Visual comparison for x in [0, 1]:

  angle:       |====                  |    Linear: theta = x
  fourier:     |========================|    Full circle: theta = 2*pi*x
  polynomial:  |=====                 |    Curved: theta = x + x^2

  The polynomial mapping introduces a non-linearity BEFORE encoding,
  which can help separate data that is linearly inseparable in angle space.
```

---

## 5. Entanglement Topologies

The `entanglement` parameter controls which qubit pairs receive two-qubit
gates, independent of the symmetry type (except rotation, which overrides).

### Full Entanglement

```
  4 qubits:  6 pairs                Connectivity graph:

  Pairs: (0,1) (0,2) (0,3)         0 ──── 1
         (1,2) (1,3)                │ \  / │
         (2,3)                      │  \/  │
                                    │  /\  │
  n(n-1)/2 pairs                    │ /  \ │
                                    3 ──── 2
```

### Linear Entanglement

```
  4 qubits:  3 pairs                Connectivity graph:

  Pairs: (0,1) (1,2) (2,3)         0 ──── 1 ──── 2 ──── 3

  n-1 pairs
```

### Circular Entanglement

```
  4 qubits:  4 pairs                Connectivity graph:

  Pairs: (0,1) (1,2) (2,3) (3,0)   0 ──── 1
                                    │      │
  n pairs                           3 ──── 2
```

### No Entanglement

```
  entanglement='none'               0      1      2      3

  0 pairs                           (product state, classically simulable)
```

### Topology Comparison

```
  ┌────────────┬───────────┬──────────────────┬──────────────────────────┐
  │  Topology  │  Pairs    │  Connectivity    │  Best For                │
  ├────────────┼───────────┼──────────────────┼──────────────────────────┤
  │  full      │  n(n-1)/2 │  All-to-all      │  Max expressivity        │
  │  linear    │  n - 1    │  Nearest-neighbor │  NISQ hardware, default  │
  │  circular  │  n        │  Ring            │  Periodic/cyclic data    │
  │  none      │  0        │  None            │  Baseline, debugging     │
  └────────────┴───────────┴──────────────────┴──────────────────────────┘

  Important: rotation symmetry ALWAYS uses coordinate pairs (0,1), (2,3),...
  regardless of the entanglement parameter setting.
```

---

## 6. Angle Computation — Complete Picture

For a concrete example with x = [0.1, 0.2, 0.3, 0.4], rotation symmetry,
angle feature map:

```
  Step 1: Feature Mapping (angle)
  ────────────────────────────────
  f(x0) = 0.1     f(x1) = 0.2     f(x2) = 0.3     f(x3) = 0.4

  Step 2: Equivariant Angles (rotation -> radius)
  ─────────────────────────────────────────────────
  Pair 0: (x0, x1) = (0.1, 0.2)   ->  phi_0 = phi_1 = sqrt(0.01 + 0.04)
                                                      = sqrt(0.05)
                                                      = 0.2236

  Pair 1: (x2, x3) = (0.3, 0.4)   ->  phi_2 = phi_3 = sqrt(0.09 + 0.16)
                                                      = sqrt(0.25)
                                                      = 0.5

  Step 3: Interaction Angles
  ──────────────────────────
  Pair (0,1): theta = sqrt(f(x0)^2 + f(x1)^2) = sqrt(0.01 + 0.04) = 0.2236
  Pair (2,3): theta = sqrt(f(x2)^2 + f(x3)^2) = sqrt(0.09 + 0.16) = 0.5


  Complete Circuit for 1 rep:

  |0> ─ H ─ RY(0.1) ─ RZ(0.2236) ─ ●──────────────
                                     │ CRZ(0.2236)
  |0> ─ H ─ RY(0.2) ─ RZ(0.2236) ─ ⊕──────────────

  |0> ─ H ─ RY(0.3) ─ RZ(0.5) ──── ●──────────────
                                     │ CRZ(0.5)
  |0> ─ H ─ RY(0.4) ─ RZ(0.5) ──── ⊕──────────────

  Notice: qubits within a coordinate pair share the SAME RZ angle
  (both get the radius). This is the symmetry-aware inductive bias.
```

---

## 7. Gate Count Breakdown

### Per-Symmetry Gate Composition (1 rep, n qubits, p entanglement pairs)

```
  ┌────────────┬─────┬──────────┬──────────┬──────────┬──────────┬───────┐
  │  Symmetry  │  H  │  RY_enc  │  RZ_equiv│  2Q Gate │  Extra   │ Total │
  │            │     │          │          │  Type    │  1Q      │ 2Q    │
  ├────────────┼─────┼──────────┼──────────┼──────────┼──────────┼───────┤
  │  rotation  │  n  │    n     │    n     │  CRZ     │    -     │ n/2   │
  │            │     │          │          │  (n/2)   │          │       │
  ├────────────┼─────┼──────────┼──────────┼──────────┼──────────┼───────┤
  │  cyclic    │  n  │    n     │    n     │  CNOT    │  RZ      │ 2*p   │
  │            │     │          │          │  (2*p)   │  (p)     │       │
  ├────────────┼─────┼──────────┼──────────┼──────────┼──────────┼───────┤
  │  reflection│  n  │    n     │    n     │  CZ      │  RZ      │ p     │
  │            │     │          │          │  (p)     │  (2*p)   │       │
  ├────────────┼─────┼──────────┼──────────┼──────────┼──────────┼───────┤
  │  full      │  n  │    n     │    n     │  CNOT    │  RY      │ 3*p   │
  │            │     │          │          │  (3*p)   │  (2*p)   │       │
  └────────────┴─────┴──────────┴──────────┴──────────┴──────────┴───────┘

  All counts are multiplied by `reps` for the complete circuit.
```

### Concrete Example (n=4, linear entanglement, reps=2)

```
  n = 4,  p (linear pairs) = 3

  Symmetry   │  H  │ RY_enc │ RZ_eq │ 2Q gates     │ Extra 1Q   │ Total
  ───────────┼─────┼────────┼───────┼──────────────┼────────────┼──────
  rotation   │  8  │   8    │   8   │  4 CRZ       │    -       │  28
  cyclic     │  8  │   8    │   8   │ 12 CNOT      │  6 RZ      │  42
  reflection │  8  │   8    │   8   │  6 CZ        │ 12 RZ      │  42
  full       │  8  │   8    │   8   │ 18 CNOT      │ 12 RY      │  54

  Note: CRZ decomposes to 2 CNOTs + 2 RZ on real hardware, so
  rotation's 4 CRZ gates become 8 CNOTs + 8 RZ at the hardware level.
```

---

## 8. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │              SYMMETRY-INSPIRED FEATURE MAP PROPERTIES                    │
  ├──────────────────────┬──────────────────────────────────────────────────┤
  │  Qubits required     │  n  (one per feature)                           │
  │  Trainable params    │  0  (all angles are data-dependent)             │
  │  Depth per rep       │  3 layers (no ent.) or 4 layers (with ent.)    │
  │  Feature maps        │  angle | fourier | polynomial                   │
  │  Symmetry types      │  rotation | cyclic | reflection | full          │
  │  Entanglement        │  full | linear | circular | none                │
  │  Entangling?         │  Yes (except entanglement='none')               │
  │  Simulability        │  Not simulable (with entanglement)              │
  │                      │  Classically simulable (without entanglement)   │
  │  Parameter count     │  0 (pure data encoding, no variational params)  │
  │  Backends            │  PennyLane, Qiskit, Cirq                        │
  │  Batch processing    │  Sequential and parallel (ThreadPoolExecutor)   │
  └──────────────────────┴──────────────────────────────────────────────────┘
```

### How It Compares

```
  Expressibility   Angle    ████████░░░░░░░░░░░░  Low   (product states)
                   Sym-Insp ██████████████████░░  High  (entangled + symmetry)
                   IQP      ████████████████████  High  (provably hard)

  Symmetry Bias    Angle    ░░░░░░░░░░░░░░░░░░░░  None
                   Sym-Insp ████████████████████  Strong (by design)
                   IQP      ░░░░░░░░░░░░░░░░░░░░  None

  Complexity       Angle    ██░░░░░░░░░░░░░░░░░░  O(n) gates
                   Sym-Insp ██████████░░░░░░░░░░  O(n) to O(n^2)
                   Equivar. ████████████████████  Full group-theoretic

  Impl. Effort     Angle    ██░░░░░░░░░░░░░░░░░░  Trivial
                   Sym-Insp ████████░░░░░░░░░░░░  Moderate
                   Equivar. ████████████████████  High (group theory)
```

---

## 9. The Trainability-Expressibility Tradeoff

More repetitions and entanglement increase expressivity but risk
**barren plateaus** — exponentially vanishing gradients.

```
                Expressivity
                     ^
                     |              / Barren plateau zone
                     |            /   (gradients -> 0)
                     |          /
                     |   full /   * Sweet spot
                     |  sym  /      (reps=1-2, linear ent.)
                     |      /
                     | lin /
                     |    /
                     |   / none (product states)
                     |  /
                     └───────────────────────────> Depth / Entanglement


  Trainability estimate (heuristic):

    trainability = max(0.1, 0.85 - 0.03*depth - 0.02*n_pairs)

    Config                            │  Estimate
    ──────────────────────────────────┼──────────
    4 feat, rotation, 1 rep, linear   │   0.73
    4 feat, rotation, 2 rep, linear   │   0.61
    4 feat, cyclic, 2 rep, full       │   0.49
    8 feat, full sym, 3 rep, full     │   0.10 (floor)
```

### Empirical Trainability Measurement

Unlike most encodings, SymmetryInspiredFeatureMap provides a
`measure_trainability()` method that **empirically measures gradient
variance** by sampling random parameters appended to the encoding circuit.

```
  >>> result = enc.measure_trainability(x, n_samples=100, seed=42)
  >>> result['gradient_variance']     # Higher = more trainable
  >>> result['empirical_trainability'] # Scaled to [0, 1]
  >>> result['heuristic_trainability'] # Compare with heuristic

  Supports both 'local' (single-qubit Z) and 'global' (all-qubit Z)
  cost observables. Local cost functions typically show better
  trainability (Cerezo et al., 2021).
```

---

## 10. Design Philosophy — Where It Sits

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │   Simple Feature Maps        Symmetry-Inspired        Equivariant  │
  │   (Angle, Basis)             Feature Map (this)       QNNs         │
  │                                                                     │
  │   - No structure            - Symmetry-AWARE          - Symmetry-  │
  │   - Product states          - Heuristic angles          EXACT      │
  │   - Easy to implement       - Practical middle-       - U(g)|psi(x)│
  │   - No inductive bias         ground                    = |psi(gx)>│
  │                              - Good generalization    - Complex    │
  │                                on symmetric data        group theory│
  │                                                                     │
  │   ◄──────────── Increasing Symmetry Guarantees ──────────────────►  │
  │   ◄──────────── Increasing Implementation Complexity ────────────►  │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  The key insight: for many practical problems, EXACT equivariance is
  overkill. Symmetry-inspired inductive bias often suffices, while being
  far simpler to implement and debug.
```

---

## 11. Use Cases

```
                        Best suited for
                  ┌───────────────────────────┐
                  │                           │
  ┌───────────────┤  Rotational patterns      │  Image features, molecular
  │               │  (symmetry='rotation')    │  coordinates, 2D point data
  │               ├───────────────────────────┤
  │               │                           │
  ├───────────────┤  Time-series / Periodic   │  Seasonal data, signals,
  │               │  (symmetry='cyclic')      │  circular features
  │               ├───────────────────────────┤
  │               │                           │
  ├───────────────┤  Mirror-symmetric data    │  Palindromic sequences,
  │               │  (symmetry='reflection')  │  spectral data, physics
  │               ├───────────────────────────┤
  │               │                           │
  ├───────────────┤  Permutation-invariant    │  Set-like data, unordered
  │               │  (symmetry='full')        │  feature collections
  │               ├───────────────────────────┤
  │               │                           │
  └───────────────┤  Quantum kernels / QML    │  When you want symmetry-aware
                  │  (any symmetry)           │  feature spaces for QSVM
                  └───────────────────────────┘
```

---

## 12. Resource Scaling

### Linear Entanglement (recommended for NISQ)

```
  n_features │ Qubits │  Pairs  │ Depth (reps=2) │ Total Gates (rotation)
  ───────────┼────────┼─────────┼────────────────┼───────────────────────
       4     │   4    │    3    │       8        │       28
       6     │   6    │    5    │       8        │       42
       8     │   8    │    7    │       8        │       56
      12     │  12    │   11    │       8        │       84
      16     │  16    │   15    │       8        │      112
      20     │  20    │   19    │       8        │      140

  Linear scaling: O(n) gates per rep. Practical for large feature counts.
```

### Full Entanglement

```
  n_features │ Qubits │  Pairs  │ Total Gates (rotation, reps=2)
  ───────────┼────────┼─────────┼───────────────────────────────
       4     │   4    │    6    │       36
       6     │   6    │   15    │       66
       8     │   8    │   28    │      104
      12     │  12    │   66    │      204
      16     │  16    │  120    │      344
      20  !  │  20    │  190    │      524

  !  Warning issued at n > 20 with full entanglement.
     O(n^2) scaling — consider linear/circular instead.
```

---

## 13. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌───────────────────────────────┐    ┌───────────────────────────────┐
  │                               │    │                               │
  │  + Symmetry-aware inductive   │    │  - NOT mathematically         │
  │    bias for structured data   │    │    rigorous equivariance      │
  │                               │    │    (heuristic only)           │
  │  + Simpler than true          │    │                               │
  │    equivariant constructions  │    │  - Rotation symmetry needs    │
  │                               │    │    even n_features            │
  │  + Four symmetry types cover  │    │                               │
  │    common problem structures  │    │  - Rotation overrides the     │
  │                               │    │    entanglement parameter     │
  │  + Built-in trainability      │    │                               │
  │    measurement (unique)       │    │  - O(n^2) gates with full     │
  │                               │    │    entanglement               │
  │  + Three feature mapping      │    │                               │
  │    functions (angle/fourier/  │    │  - No trainable parameters    │
  │    polynomial)                │    │    (pure encoding, not a      │
  │                               │    │     variational ansatz)       │
  │  + Multi-backend support      │    │                               │
  │    (PennyLane, Qiskit, Cirq)  │    │  - Heuristic trainability     │
  │                               │    │    estimate (approximate)     │
  │  + Parallel batch processing  │    │                               │
  │    for large datasets         │    │                               │
  │                               │    │                               │
  └───────────────────────────────┘    └───────────────────────────────┘
```

---

## 14. Data Preprocessing

```
  Each feature mapping has different input range requirements.
  Proper normalization is critical for meaningful encoding.

  ┌──────────────────────────────────────────────────────────────────────┐
  │                    PREPROCESSING RECOMMENDATIONS                     │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  angle mapping:      Normalize features to [0, pi]                   │
  │                      Direct mapping, full RY rotation range          │
  │                                                                      │
  │  fourier mapping:    Normalize features to [0, 1]                    │
  │                      Maps to [0, 2*pi], full rotation range          │
  │                                                                      │
  │  polynomial mapping: Normalize features to [-1, 1]                   │
  │                      x + x^2 grows unboundedly outside this range    │
  │                                                                      │
  │  General tips:                                                       │
  │  - Standardize features to similar scales before encoding            │
  │  - The encoding does NOT normalize internally                        │
  │  - NaN and Inf values will raise ValueError                          │
  │  - Out-of-range values trigger UserWarning (not errors)              │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 15. Comparison with Related Encodings

```
  ┌────────────────────────┬────────┬──────────┬────────────┬──────────────┐
  │    Encoding            │ Qubits │  Depth   │ Symmetry   │ Equivariance │
  │                        │        │          │ Bias       │ Guarantee    │
  ├────────────────────────┼────────┼──────────┼────────────┼──────────────┤
  │  Symmetry-Inspired *   │   n    │ O(n)     │ Strong     │ Heuristic    │
  │  Equivariant FM        │   n    │ O(n)     │ Strong     │ Provable     │
  │  Angle                 │   n    │ O(1)     │ None       │ None         │
  │  IQP                   │   n    │ O(n^2)   │ None       │ None         │
  │  Pauli Feature Map     │   n    │ O(n^2)   │ None       │ None         │
  │  Amplitude             │ log(n) │ O(2^n)   │ None       │ None         │
  └────────────────────────┴────────┴──────────┴────────────┴──────────────┘

  Symmetry-Inspired vs Equivariant Feature Map:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  The Equivariant FM provides PROVABLE guarantees:                      │
  │    U(g)|psi(x)> = |psi(g.x)>  for group elements g                    │
  │                                                                        │
  │  Symmetry-Inspired provides HEURISTIC bias:                            │
  │    Gate angles computed from invariant functions of x                   │
  │    Cheaper to implement, often sufficient in practice                   │
  │                                                                        │
  │  Choose Equivariant when: mathematical rigor is required               │
  │  Choose Symmetry-Inspired when: practical bias + simplicity preferred  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. Larocca, M., et al. (2022). "Group-Invariant Quantum Machine Learning."
   PRX Quantum 3, 030341. https://doi.org/10.1103/PRXQuantum.3.030341

2. Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational Quantum
   Machine Learning." PRX Quantum 4, 010328.
   https://doi.org/10.1103/PRXQuantum.4.010328

3. Schatzki, L., et al. (2022). "Theoretical Guarantees for Permutation-
   Equivariant Quantum Neural Networks." npj Quantum Information 8, 74.
   https://doi.org/10.1038/s41534-022-00585-1

4. Skolik, A., et al. (2023). "Equivariant quantum circuits for learning
   on weighted graphs." npj Quantum Information 9, 47.
   https://doi.org/10.1038/s41534-023-00710-y

5. Nguyen, Q., et al. (2022). "Theory for Equivariant Quantum Neural
   Networks." arXiv:2210.08566.