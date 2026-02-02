# Pauli Feature Map

**Author:** Ashutosh Mishra

A highly configurable quantum encoding that uses **arbitrary Pauli rotation
gates** to map classical data into quantum states — generalizing ZZFeatureMap
and enabling custom problem-specific encoding structures.

```
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║                                                                       ║
 ║   |ψ(x)⟩ = [ U_Pauli(x) · H^⊗n ]^reps |0⟩^⊗n                       ║
 ║                                                                       ║
 ║   "Choose your Paulis → shape your quantum kernel → solve your task" ║
 ║                                                                       ║
 ╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

PauliFeatureMap gives you a **menu of Pauli rotation operators** to build
a custom data encoding. You pick which single-qubit (X, Y, Z) and
two-qubit (XX, YY, ZZ, XY, XZ, YZ) terms to include, how they connect
(entanglement topology), and how many times to repeat (reps).

```
  Configuration knobs:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  paulis = ["Z", "ZZ"]         Which Pauli rotations to apply       │
  │  entanglement = "full"        How qubits connect (topology)        │
  │  reps = 2                     How many layers to repeat            │
  │                                                                     │
  │  These 3 choices fully determine the circuit structure.            │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  When paulis=["Z","ZZ"] and entanglement="full":
    → Equivalent to the standard ZZFeatureMap from Havlicek et al. (2019)

  When paulis=["Y","XX"]:
    → A completely different encoding, tailored to your problem
```

> The power of PauliFeatureMap lies in its **flexibility**: by choosing
> different Pauli terms and topologies, you are effectively designing the
> geometry of your quantum feature space.

---

## 2. Circuit Structure

Each of the `reps` layers consists of three sublayers:
1. **Hadamard layer** — creates uniform superposition
2. **Single-qubit Pauli rotations** — encode individual features
3. **Two-qubit Pauli rotations** — encode pairwise feature interactions

### Full Circuit for 4 Qubits, paulis=["Z", "ZZ"], 2 Reps

```
         ┌──── Rep 1 ──────────────────────────┐┌──── Rep 2 ─────── ...
         │                                      ││
  |0⟩ ── H ── RZ(2x₀) ──╌╌ ZZ pairs ╌╌──────── H ── RZ(2x₀) ── ...
  |0⟩ ── H ── RZ(2x₁) ──╌╌         ╌╌──────── H ── RZ(2x₁) ── ...
  |0⟩ ── H ── RZ(2x₂) ──╌╌         ╌╌──────── H ── RZ(2x₂) ── ...
  |0⟩ ── H ── RZ(2x₃) ──╌╌         ╌╌──────── H ── RZ(2x₃) ── ...
         │                                      ││
         └──────────────────────────────────────┘└────────────── ...
```

### Anatomy of One Repetition

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      SINGLE REPETITION                              │
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
  │  │  Hadamard     │  │  Single-Qubit    │  │  Two-Qubit Pauli     │  │
  │  │  Layer        │  │  Paulis          │  │  Interactions         │  │
  │  │              │  │                  │  │                      │  │
  │  │  q₀ ── H    │  │  q₀ ── RZ(2x₀) │  │  For each pair (i,j) │  │
  │  │  q₁ ── H    │  │  q₁ ── RZ(2x₁) │  │  in entanglement:    │  │
  │  │  q₂ ── H    │  │  q₂ ── RZ(2x₂) │  │                      │  │
  │  │  q₃ ── H    │  │  q₃ ── RZ(2x₃) │  │  exp(-iφ P_a⊗P_b)   │  │
  │  │              │  │                  │  │                      │  │
  │  │  Creates     │  │  Encodes each   │  │  Encodes pairwise    │  │
  │  │  |+⟩^⊗n     │  │  feature as a   │  │  feature products    │  │
  │  │              │  │  phase           │  │  as entanglement     │  │
  │  └──────────────┘  └──────────────────┘  └──────────────────────┘  │
  │                                                                     │
  │  Depth per rep:  1 (H) + |single_paulis| + 3×|two_paulis|         │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Mapping Functions

The rotation angles are not raw feature values — they pass through
specific mapping functions that shape the quantum kernel.

### Single-Qubit Features

```
  φ(xᵢ) = 2·xᵢ

  Each feature is doubled before rotation:

  xᵢ ──[×2]──► R_P(2xᵢ)

  Why the factor of 2?
  It ensures the full Bloch sphere is utilized when x ∈ [0, π].
```

### Two-Qubit Feature Interactions

```
  φ(xᵢ, xⱼ) = 2·(π - xᵢ)·(π - xⱼ)

  This is the key design choice from Havlicek et al. (2019):

      x₀ ──┐
            ├──► 2(π - x₀)(π - x₁) ──► exp(-iφ/2  P_a⊗P_b)
      x₁ ──┘

  Properties of this mapping:
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  • φ = 0  when either xᵢ = π or xⱼ = π  (interaction off)   │
  │  • φ is maximized when both features are near 0              │
  │  • Product structure captures pairwise correlations          │
  │  • The (π - x) centering ensures non-trivial interactions    │
  │    across the typical input range [0, 2π]                    │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

---

## 4. The Pauli Menu — All Available Terms

### Single-Qubit Pauli Terms

```
  ┌────────┬──────────────────────────┬────────────────────────────────┐
  │  Term  │  Gate Applied             │  Effect on |+⟩ State           │
  ├────────┼──────────────────────────┼────────────────────────────────┤
  │        │                          │                                │
  │   Z    │  RZ(2xᵢ)                │  Phase rotation around Z       │
  │        │  = diag(e^{-ixᵢ},       │  (computational basis)         │
  │        │         e^{+ixᵢ})       │  Native on most hardware       │
  │        │                          │                                │
  ├────────┼──────────────────────────┼────────────────────────────────┤
  │        │                          │                                │
  │   X    │  RX(2xᵢ)                │  Rotation in Y-Z plane         │
  │        │  = cos(xᵢ)I             │  Changes populations           │
  │        │    - i·sin(xᵢ)X         │  Needs basis change gate       │
  │        │                          │                                │
  ├────────┼──────────────────────────┼────────────────────────────────┤
  │        │                          │                                │
  │   Y    │  RY(2xᵢ)                │  Rotation in X-Z plane         │
  │        │  = cos(xᵢ)I             │  Real amplitude changes        │
  │        │    - i·sin(xᵢ)Y         │  Highest basis change cost     │
  │        │                          │                                │
  └────────┴──────────────────────────┴────────────────────────────────┘
```

### Two-Qubit Pauli Terms

Each two-qubit term creates entanglement between qubit pairs. The
rotation `exp(-iφ/2 Pa⊗Pb)` is decomposed into native gates:

```
  ┌────────┬─────────────────────────────────────────────────────────┐
  │  Term  │  Gate Decomposition                                     │
  ├────────┼─────────────────────────────────────────────────────────┤
  │        │                                                         │
  │  ZZ    │  ── CNOT ── RZ(φ) ── CNOT ──                           │
  │        │  No basis change needed. Standard IQP/ZZ encoding.      │
  │        │                                                         │
  ├────────┼─────────────────────────────────────────────────────────┤
  │        │                                                         │
  │  XX    │  ── H ── CNOT ── RZ(φ) ── CNOT ── H ──                │
  │        │  Hadamards rotate to X-basis on both qubits.            │
  │        │                                                         │
  ├────────┼─────────────────────────────────────────────────────────┤
  │        │                                                         │
  │  YY    │  ── Sdg ── H ── CNOT ── RZ(φ) ── CNOT ── H ── S ──   │
  │        │  Full Y-basis change on both qubits. Highest overhead.  │
  │        │                                                         │
  ├────────┼─────────────────────────────────────────────────────────┤
  │        │                                                         │
  │  XZ    │  ── H(q_a) ── CNOT ── RZ(φ) ── CNOT ── H(q_a) ──     │
  │        │  Mixed: X-basis on first qubit, Z-basis on second.      │
  │        │                                                         │
  ├────────┼─────────────────────────────────────────────────────────┤
  │        │  (Similarly for XY, YZ, YX, ZX, ZY — each with         │
  │  ...   │   appropriate basis changes on the respective qubits)   │
  │        │                                                         │
  └────────┴─────────────────────────────────────────────────────────┘
```

### Gate Decomposition of ZZ Interaction (Detailed)

```
  exp(-iφ/2 Z⊗Z) decomposition:

  q_i ───────■────────────────■───────
             │                │
  q_j ──── CNOT ── RZ(φ) ── CNOT ────

  How it works:

  Step 1: CNOT maps |ab⟩ → |a, a⊕b⟩
  Step 2: RZ(φ) on target applies phase e^{-iφ/2} or e^{+iφ/2}
          depending on the target qubit state
  Step 3: CNOT uncomputes, restoring the computational basis
  Net effect: Applies phase e^{-iφ/2 (-1)^{a⊕b}} = e^{±iφ/2}
              which equals exp(-iφ/2 Z⊗Z)
```

### Gate Overhead Comparison

```
  ┌────────┬──────────────────────────────────────────────────────────┐
  │  Term  │  Gates per pair     │  Total overhead   │  Basis cost   │
  ├────────┼─────────────────────┼───────────────────┼───────────────┤
  │  ZZ    │  2 CNOT + 1 RZ      │  3 gates          │  None ★       │
  │  XX    │  2 CNOT + 1 RZ + 4H │  7 gates          │  Low          │
  │  XZ    │  2 CNOT + 1 RZ + 2H │  5 gates          │  Low          │
  │  XY    │  2 CNOT + 1 RZ      │  7 gates          │  Medium       │
  │        │  + 2H + 2S/Sdg      │                   │               │
  │  YY    │  2 CNOT + 1 RZ      │  11 gates         │  High         │
  │        │  + 4H + 4S/Sdg      │                   │               │
  └────────┴─────────────────────┴───────────────────┴───────────────┘

  ★ ZZ is the cheapest two-qubit term — this is why ZZFeatureMap
    is the most commonly used variant in the literature.
```

---

## 5. Entanglement Topologies

The entanglement parameter controls **which qubit pairs** receive
two-qubit Pauli gates. This is one of the most impactful design choices.

### Full Entanglement

```
  All pairs (i, j) where i < j.         Pairs for n=4:
  Maximum expressivity.                   (0,1) (0,2) (0,3)
  Requires all-to-all connectivity.             (1,2) (1,3)
                                                      (2,3)
  q₀ ──────╳──────╳──────╳
           │      │      │              n(n-1)/2 = 6 pairs
  q₁ ──────╳      │      │
           ╳──────╳      │
  q₂ ─────────────╳      │
           ╳─────────────╳
  q₃ ────────────────────╳

  Scaling: O(n²) pairs  →  O(n²) CNOT gates per layer per Pauli term
```

### Linear Entanglement

```
  Consecutive pairs only.                Pairs for n=4:
  Hardware-friendly.                      (0,1) (1,2) (2,3)
  Nearest-neighbor connectivity.

  q₀ ──■──                               n-1 = 3 pairs
       │
  q₁ ──╳──■──
           │
  q₂ ─────╳──■──
              │
  q₃ ────────╳──

  Scaling: O(n) pairs  →  O(n) CNOT gates per layer per Pauli term
```

### Circular Entanglement

```
  Linear + wrap-around connection.       Pairs for n=4:
  Ring topology.                          (0,1) (1,2) (2,3) (3,0)
  Good for problems with periodicity.

  q₀ ──■─────────────────╳               n pairs (for n>2)
       │                  │
  q₁ ──╳──■──            │
           │              │
  q₂ ─────╳──■──         │
              │           │
  q₃ ────────╳───────────■

  Scaling: O(n) pairs  →  O(n) CNOT gates per layer per Pauli term
```

### Topology Comparison at a Glance

```
  ┌─────────────┬────────────┬────────────────┬──────────────────────────┐
  │  Topology   │  # Pairs   │  Connectivity  │  When to Use             │
  ├─────────────┼────────────┼────────────────┼──────────────────────────┤
  │             │            │                │                          │
  │  full       │  n(n-1)/2  │  All-to-all    │  Max expressivity,       │
  │             │  O(n²)     │                │  simulators, small n     │
  │             │            │                │                          │
  ├─────────────┼────────────┼────────────────┼──────────────────────────┤
  │             │            │                │                          │
  │  linear     │  n-1       │  Nearest-      │  NISQ hardware,          │
  │             │  O(n)      │  neighbor       │  superconducting chips   │
  │             │            │                │                          │
  ├─────────────┼────────────┼────────────────┼──────────────────────────┤
  │             │            │                │                          │
  │  circular   │  n         │  Ring          │  Periodic/cyclic data,   │
  │             │  O(n)      │                │  ion trap architectures  │
  │             │            │                │                          │
  └─────────────┴────────────┴────────────────┴──────────────────────────┘

  Number of CNOT pairs (full entanglement) grows quickly:

  n_features │  Pairs (full)  │  CNOTs per layer per Pauli
  ───────────┼────────────────┼───────────────────────────
       2     │       1        │        2
       4     │       6        │       12
       8     │      28        │       56
      12     │      66        │      132   ← warning threshold
      16     │     120        │      240
```

---

## 6. Why the Hadamard Layer Matters

```
  Without H gates:                   With H gates:
  ─────────────────                  ────────────────

  |0⟩ ── RZ(2x) ──                  |0⟩ ── H ── RZ(2x) ──

  RZ(θ)|0⟩ = e^{-iθ/2}|0⟩          RZ(θ)|+⟩ = e^{-iθ/2}|0⟩ + e^{iθ/2}|1⟩
                                                ─────────────────────────────
  Just a global phase!               Actual relative phase encoding!
  No observable effect.               Measurable interference.

  ┌──────────────────────────────────────────────────────────────┐
  │  The Hadamard creates superposition FIRST, then the Pauli   │
  │  rotation imprints data as a RELATIVE PHASE between |0⟩     │
  │  and |1⟩. Without H, Z-rotations on |0⟩ do nothing useful. │
  └──────────────────────────────────────────────────────────────┘
```

---

## 7. Quantum Kernel Connection

PauliFeatureMap was introduced specifically for **quantum kernel methods**.
The quantum kernel between two data points is:

```
  κ(x, x') = |⟨ψ(x)|ψ(x')⟩|²

  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  Classical       Quantum Feature       Kernel                  │
  │  Data            Map                   Evaluation              │
  │                                                                │
  │  x ───────► |ψ(x)⟩ ─────────────┐                            │
  │                                   ├──► κ(x,x') = |⟨ψ|ψ'⟩|²  │
  │  x' ──────► |ψ(x')⟩ ────────────┘                            │
  │                                                                │
  │  The choice of Pauli terms and entanglement determines         │
  │  the geometry of the feature space and hence the kernel.       │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘

  Different Paulis → Different kernel geometries → Different problems

  ┌──────────────────┬────────────────────────────────────────────┐
  │  Configuration   │  Feature Space Geometry                    │
  ├──────────────────┼────────────────────────────────────────────┤
  │  ["Z"] only      │  Diagonal phases, no correlations         │
  │  ["Z","ZZ"]      │  Standard: phases + pairwise products     │
  │  ["X","XX"]      │  Amplitude-based, population changes      │
  │  ["Y","YY"]      │  Complex rotations, richest structure     │
  │  ["Z","XX","YY"] │  Multiple interaction types combined      │
  └──────────────────┴────────────────────────────────────────────┘
```

---

## 8. Key Properties

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                   PAULI FEATURE MAP PROPERTIES                           │
  ├──────────────────────┬───────────────────────────────────────────────────┤
  │  Qubits required     │  n  (one per feature)                            │
  │  Circuit depth       │  reps × [1 + |single| + 3×|two-qubit|]          │
  │  Hadamard gates      │  n × reps                                        │
  │  Single-qubit rots   │  n × reps × |single_paulis|                     │
  │  CNOT gates          │  2 × reps × pairs × |two_paulis|                │
  │  Entangling?         │  Yes (when two-qubit Paulis present)             │
  │  Simulability        │  Not classically simulable (entangling case)     │
  │  Trainability        │  Depends on depth: max(0.1, 0.9 - 0.05·depth)   │
  │  Connectivity        │  Depends on topology (all-to-all / linear / ring)│
  │  Configurable Paulis │  X, Y, Z, XX, YY, ZZ, XY, XZ, YZ               │
  └──────────────────────┴───────────────────────────────────────────────────┘
```

### Property Trade-off Bars (default: paulis=["Z","ZZ"], full, reps=2)

```
  Expressibility  ████████████████░░░░  ~0.8   ← High: entangling + products
  Trainability    ██████████████░░░░░░  ~0.55  ← Moderate: depth-dependent
  Hardware Cost   ██████████████░░░░░░  Medium ← O(n²) CNOTs with full entanglement
  Configurability ████████████████████  Max    ← Most configurable encoding
```

---

## 9. Comparison with Related Encodings

```
  ┌──────────────────┬──────────┬──────────────┬───────────────┬────────────────┐
  │    Encoding      │  Paulis  │  Entangle.   │  Gate Scaling │  Configurabil. │
  ├──────────────────┼──────────┼──────────────┼───────────────┼────────────────┤
  │  Angle           │  1 axis  │     No       │    O(n)       │  Low           │
  │  ZZFeatureMap    │  Z, ZZ   │  full/lin    │  O(n²)/O(n)  │  Medium        │
  │  IQP             │  ZZ      │    Yes       │    O(n²)      │  Low           │
  │  PauliFeatMap ★  │  Any!    │  full/lin/   │  O(n²)/O(n)  │  High ★        │
  │                  │          │  circular    │               │                │
  │  DataReuploading │  RY      │  linear      │    O(L·n)     │  Medium        │
  └──────────────────┴──────────┴──────────────┴───────────────┴────────────────┘

  PauliFeatureMap with paulis=["Z","ZZ"] is EXACTLY equivalent to ZZFeatureMap.
  It is a strict generalization — every ZZFeatureMap is a PauliFeatureMap.
```

---

## 10. Choosing Your Pauli Terms — A Decision Guide

```
  START
    │
    ├── Do you need pairwise feature interactions?
    │     │
    │     ├── No  → paulis=["Z"]  (simplest, no entanglement)
    │     │
    │     └── Yes → Which basis captures your problem structure?
    │               │
    │               ├── Standard / Unknown → paulis=["Z", "ZZ"] ★
    │               │   (proven in quantum kernel literature)
    │               │
    │               ├── Need amplitude changes → paulis=["X", "XX"]
    │               │   (populations change, not just phases)
    │               │
    │               ├── Need rich rotations → paulis=["Y", "YY"]
    │               │   (most expressive but highest gate overhead)
    │               │
    │               └── Exploring → paulis=["Z", "ZZ", "XX"]
    │                   (combine multiple interaction types)
    │
    └── How many qubits / features?
          │
          ├── n ≤ 8   → entanglement="full"  (affordable)
          ├── n ≤ 12  → entanglement="full" or "circular"
          └── n > 12  → entanglement="linear" or "circular"
                        (full creates too many CNOTs)
```

---

## 11. Example Walkthrough

Encode `x = [0.5, 1.0, 1.5]` with `paulis=["Z", "ZZ"]`, `entanglement="full"`, `reps=1`:

```
  Configuration: n_features=3, paulis=["Z","ZZ"], full, reps=1
  Entanglement pairs: (0,1), (0,2), (1,2)

  ══════════════════════════════════════════════════════════════
  STEP 1 — Hadamard Layer
  ══════════════════════════════════════════════════════════════

  q₀: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
  q₁: H|0⟩ = |+⟩
  q₂: H|0⟩ = |+⟩

  State: |+++⟩ = (1/√8) Σ|ijk⟩  (uniform superposition)

  ══════════════════════════════════════════════════════════════
  STEP 2 — Single-Qubit Z Rotations:  RZ(2xᵢ)
  ══════════════════════════════════════════════════════════════

  q₀: RZ(2·0.5) = RZ(1.0)    phase encoding of x₀
  q₁: RZ(2·1.0) = RZ(2.0)    phase encoding of x₁
  q₂: RZ(2·1.5) = RZ(3.0)    phase encoding of x₂

  Each qubit acquires a data-dependent phase:
    |k⟩ → e^{-i·xᵢ·(-1)^k} |k⟩

  ══════════════════════════════════════════════════════════════
  STEP 3 — Two-Qubit ZZ Interactions
  ══════════════════════════════════════════════════════════════

  For each pair, compute φ = 2(π - xᵢ)(π - xⱼ):

  Pair (0,1):  φ = 2(π - 0.5)(π - 1.0) = 2·2.642·2.142 = 11.32
  Pair (0,2):  φ = 2(π - 0.5)(π - 1.5) = 2·2.642·1.642 =  8.68
  Pair (1,2):  φ = 2(π - 1.0)(π - 1.5) = 2·2.142·1.642 =  7.03

  Each pair gets the CNOT-RZ-CNOT decomposition:

       q_i ───■────────────■───
              │            │
       q_j ──CNOT─ RZ(φ) ─CNOT──

  This creates entanglement encoding the product (π-xᵢ)(π-xⱼ).

  ══════════════════════════════════════════════════════════════
  RESULT
  ══════════════════════════════════════════════════════════════

  |ψ(x)⟩: An entangled 3-qubit state where:
    • Individual features are encoded as Z-phases
    • Pairwise products are encoded as ZZ-entangling phases
    • The quantum kernel κ(x,x') = |⟨ψ(x)|ψ(x')⟩|²
      captures both individual and interaction features
```

---

## 12. Resource Scaling

### Default Configuration: paulis=["Z", "ZZ"], full entanglement

```
  n_feat │ reps │  H Gates │  RZ (1q) │  RZ (2q) │  CNOTs │  Total │  Depth
  ───────┼──────┼──────────┼──────────┼──────────┼────────┼────────┼───────
     2   │   2  │     4    │     4    │     2    │     4  │    14  │    8
     4   │   2  │     8    │     8    │    12    │    24  │    52  │    8
     4   │   3  │    12    │    12    │    18    │    36  │    78  │   12
     8   │   2  │    16    │    16    │    56    │   112  │   200  │    8
     8   │   2  │    16    │    16    │    56    │   112  │   200  │    8
    12   │   2  │    24    │    24    │   132    │   264  │   444  │    8

  With linear entanglement (same paulis):

  n_feat │ reps │  H Gates │  RZ (1q) │  RZ (2q) │  CNOTs │  Total │  Depth
  ───────┼──────┼──────────┼──────────┼──────────┼────────┼────────┼───────
     4   │   2  │     8    │     8    │     6    │    12  │    34  │    8
     8   │   2  │    16    │    16    │    14    │    28  │    74  │    8
    16   │   2  │    32    │    32    │    30    │    60  │   154  │    8

  Gate count formulas (per Pauli term type, per rep):
    Hadamards:         n
    Single-qubit rot:  n  per single Pauli
    CNOTs:             2 × n_pairs  per two-qubit Pauli
    RZ (two-qubit):    n_pairs  per two-qubit Pauli
    Basis change:      depends on Pauli type (see Section 4)
```

---

## 13. Trainability Considerations

```
  Trainability vs. Circuit Depth

  Pauli feature maps can get deep fast, especially with multiple
  two-qubit Pauli terms and full entanglement.

  Depth per rep = 1 + |single_paulis| + 3 × |two_paulis|

  ┌──────────────────────────┬────────────────┬─────────────────────┐
  │  Configuration           │  Depth/rep     │  Trainability (r=2) │
  ├──────────────────────────┼────────────────┼─────────────────────┤
  │  ["Z"]                   │     2          │  ~0.70              │
  │  ["Z", "ZZ"]  ★         │     4          │  ~0.50              │
  │  ["Z", "ZZ", "XX"]      │     7          │  ~0.20              │
  │  ["X", "Y", "Z", "ZZ"]  │     7          │  ~0.20              │
  │  ["Z", "XX", "YY", "ZZ"]│    10          │  ~0.10 (floor)      │
  └──────────────────────────┴────────────────┴─────────────────────┘

  Mitigation strategies for deep circuits:
    1. Use fewer Pauli terms (["Z","ZZ"] is usually sufficient)
    2. Use fewer reps (reps=1 or 2)
    3. Switch to linear/circular entanglement
    4. Use gradient-free optimizers
    5. Apply layer-wise training
```

---

## 14. Data Preprocessing

```
  The feature mapping functions determine the optimal input range:

  Single-qubit:   φ = 2xᵢ         → x ∈ [0, π] gives φ ∈ [0, 2π]
  Two-qubit:      φ = 2(π-xᵢ)(π-xⱼ) → x ∈ [0, 2π] is typical

  Recommended scaling:

  Method              Formula                          Output Range
  ──────────────────────────────────────────────────────────────────
  Min-max [0, 2π]     x' = 2π·(x - min)/(max - min)    [0, 2π]
  Min-max [0, π]      x' = π·(x - min)/(max - min)     [0, π]
  Arctan              x' = π/2 + arctan(x)              (0, π)

  Important: The (π - xᵢ) term in the two-qubit mapping means
  features near π produce weak interactions. Consider centering
  your data away from π if strong pairwise correlations are desired.
```

---

## 15. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌────────────────────────────────┐  ┌────────────────────────────────┐
  │                                │  │                                │
  │  + Maximum configurability    │  │  - Complexity of choices       │
  │    Choose any Pauli terms,     │  │    Many knobs to tune: paulis, │
  │    topology, and reps          │  │    topology, reps              │
  │                                │  │                                │
  │  + Proven kernel methods      │  │  - O(n²) scaling (full)       │
  │    Theoretical foundations     │  │    Full entanglement becomes   │
  │    from Havlicek et al.        │  │    expensive for n > 12        │
  │                                │  │                                │
  │  + Topology flexibility       │  │  - Connectivity demands        │
  │    full / linear / circular    │  │    Full entanglement needs     │
  │    for different hardware      │  │    all-to-all qubit coupling   │
  │                                │  │                                │
  │  + Generalizes ZZFeatureMap   │  │  - Trainability degrades       │
  │    Strict superset: more       │  │    Many Pauli terms create     │
  │    options, same baseline      │  │    deep circuits with barren   │
  │                                │  │    plateaus                    │
  │  + Pairwise feature products  │  │                                │
  │    (π-xᵢ)(π-xⱼ) captures     │  │  - Y-basis overhead            │
  │    interactions naturally      │  │    Y-Pauli terms need extra    │
  │                                │  │    basis change gates (S, Sdg) │
  └────────────────────────────────┘  └────────────────────────────────┘
```

---

## 16. Use Cases

```
                          Best suited for
                    ┌────────────────────────────────────┐
                    │                                    │
  ┌─────────────────┤  Quantum Kernel Methods            │  Designed specifically
  │                 │  (SVM with quantum kernel)         │  for this by Havlicek
  │                 ├────────────────────────────────────┤
  │                 │                                    │
  ├─────────────────┤  Research & Experimentation        │  Explore which Pauli
  │                 │  (encoding architecture search)    │  terms fit your data
  │                 ├────────────────────────────────────┤
  │                 │                                    │
  ├─────────────────┤  Hardware-Aware Design             │  Match topology to
  │                 │  (NISQ device constraints)         │  chip connectivity
  │                 ├────────────────────────────────────┤
  │                 │                                    │
  ├─────────────────┤  Variational Classifiers           │  Configurable feature
  │                 │  (hybrid quantum-classical)        │  map for VQC circuits
  │                 ├────────────────────────────────────┤
  │                 │                                    │
  └─────────────────┤  Generalized QNN Architectures     │  Building blocks beyond
                    │  (custom QML models)               │  standard ZZ structure
                    └────────────────────────────────────┘
```

---

## 17. Quick Reference: Common Configurations

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  NAME                 │  PAULIS          │  ENTANGLE.  │  NOTES      │
  ├───────────────────────┼──────────────────┼─────────────┼─────────────┤
  │  ZZFeatureMap         │  ["Z", "ZZ"]     │  full       │  Standard   │
  │  (default)            │                  │             │  baseline   │
  ├───────────────────────┼──────────────────┼─────────────┼─────────────┤
  │  Z-only (no entangl.) │  ["Z"]           │  any        │  Simplest,  │
  │                       │                  │             │  simulable  │
  ├───────────────────────┼──────────────────┼─────────────┼─────────────┤
  │  NISQ-friendly ZZ     │  ["Z", "ZZ"]     │  linear     │  Low gate   │
  │                       │                  │             │  overhead   │
  ├───────────────────────┼──────────────────┼─────────────┼─────────────┤
  │  Rich interactions    │  ["Z","ZZ","XX"] │  full       │  Multiple   │
  │                       │                  │             │  bases      │
  ├───────────────────────┼──────────────────┼─────────────┼─────────────┤
  │  Periodic topology    │  ["Z", "ZZ"]     │  circular   │  Ring data  │
  │                       │                  │             │  structure  │
  └───────────────────────┴──────────────────┴─────────────┴─────────────┘
```

---

## References

1. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
   feature spaces." Nature, 567(7747), 209-212.

2. Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature
   Hilbert Spaces." Physical Review Letters, 122(4), 040504.

3. McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
   network training landscapes." Nature Communications, 9, 4812.