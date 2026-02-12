# Angle Encoding

A fundamental quantum data encoding that maps classical features directly
to rotation angles of single-qubit gates, producing **product states**.

```
 ╔═══════════════════════════════════════════════════════════════════╗
 ║                                                                   ║
 ║   |ψ(x)⟩  =  ⊗ᵢ  Rₐ(xᵢ) |0⟩                                   ║
 ║                                                                   ║
 ║   "One feature  →  one qubit  →  one rotation"                   ║
 ║                                                                   ║
 ╚═══════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

Each classical feature value `xᵢ` is used **directly as a rotation angle**
on a dedicated qubit. No entanglement is created — each qubit is independent.

```
  Classical Data                          Quantum State
  ┌──────────┐                            ┌──────────────────────┐
  │ x₀ = 1.2 │──── Rₐ(1.2) on qubit 0 ──│                      │
  │ x₁ = 0.5 │──── Rₐ(0.5) on qubit 1 ──│  |ψ⟩ = Rₐ(1.2)|0⟩   │
  │ x₂ = 3.1 │──── Rₐ(3.1) on qubit 2 ──│      ⊗ Rₐ(0.5)|0⟩   │
  │ x₃ = 0.8 │──── Rₐ(0.8) on qubit 3 ──│      ⊗ Rₐ(3.1)|0⟩   │
  └──────────┘                            │      ⊗ Rₐ(0.8)|0⟩   │
                                          └──────────────────────┘
       n features                           n qubits (product state)
```

---

## 2. Circuit Structure

All rotation gates execute **in parallel** within each layer.
The circuit depth equals `reps`, independent of qubit count.

### Single Layer (`reps = 1`)

```
        ┌─────────┐
  |0⟩ ──┤ Rₐ(x₀) ├──
        └─────────┘
        ┌─────────┐
  |0⟩ ──┤ Rₐ(x₁) ├──
        └─────────┘
        ┌─────────┐
  |0⟩ ──┤ Rₐ(x₂) ├──
        └─────────┘
            ...
        ┌─────────────┐
  |0⟩ ──┤ Rₐ(x_{n-1}) ├──
        └─────────────┘
```

### Multiple Layers (`reps = 3`)

```
        ┌─────────┐ ┌─────────┐ ┌─────────┐
  |0⟩ ──┤ Rₐ(x₀) ├─┤ Rₐ(x₀) ├─┤ Rₐ(x₀) ├──   ≡  Rₐ(3·x₀)|0⟩
        └─────────┘ └─────────┘ └─────────┘
        ┌─────────┐ ┌─────────┐ ┌─────────┐
  |0⟩ ──┤ Rₐ(x₁) ├─┤ Rₐ(x₁) ├─┤ Rₐ(x₁) ├──   ≡  Rₐ(3·x₁)|0⟩
        └─────────┘ └─────────┘ └─────────┘
        ┌─────────┐ ┌─────────┐ ┌─────────┐
  |0⟩ ──┤ Rₐ(x₂) ├─┤ Rₐ(x₂) ├─┤ Rₐ(x₂) ├──   ≡  Rₐ(3·x₂)|0⟩
        └─────────┘ └─────────┘ └─────────┘

  Effective rotation:  Rₐ(xᵢ)^reps  =  Rₐ(reps · xᵢ)
```

> Repeating the encoding layer `reps` times simply **scales** the rotation
> angle, since consecutive same-axis rotations compose additively.

---

## 3. Rotation Gates — The Mathematics

The axis parameter `a ∈ {X, Y, Z}` determines which Pauli rotation is applied.

### Gate Definitions

```
                 ┌                            ┐
  RX(θ)    =    │  cos(θ/2)      -i·sin(θ/2) │
                 │ -i·sin(θ/2)    cos(θ/2)    │
                 └                            ┘

                 ┌                            ┐
  RY(θ)    =    │  cos(θ/2)      -sin(θ/2)   │
                 │  sin(θ/2)       cos(θ/2)   │
                 └                            ┘

                 ┌                            ┐
  RZ(θ)    =    │  e^{-iθ/2}        0        │
                 │      0        e^{iθ/2}     │
                 └                            ┘
```

### What Each Axis Does to |0⟩

```
                  Bloch Sphere View
                       |z⟩ = |0⟩
                        *
                       /|\
                      / | \
                     /  |  \
                    /   |   \
               ────/────+────\────  |y⟩
                  /     |     \
                 /      |      \
                        *
                      |-z⟩ = |1⟩

  RY(θ)|0⟩ :  Rotates in the X-Z plane     (real amplitudes)
  RX(θ)|0⟩ :  Rotates in the Y-Z plane     (complex amplitudes)
  RZ(θ)|0⟩ :  Rotates around Z-axis        (phase only, stays at |0⟩!)
```

### Applied to |0⟩

```
  RX(θ)|0⟩  =  cos(θ/2)|0⟩  -  i·sin(θ/2)|1⟩     ← complex amplitudes
  RY(θ)|0⟩  =  cos(θ/2)|0⟩  +    sin(θ/2)|1⟩     ← real amplitudes ★
  RZ(θ)|0⟩  =  e^{-iθ/2} |0⟩                      ← global phase only!
```

> **Why RY is the default**: Starting from |0⟩, only RY produces a
> superposition with purely real amplitudes and non-trivial population
> transfer. RZ adds only a global phase (unobservable), while RX
> introduces imaginary components.

---

## 4. Axis Comparison at a Glance

```
  ┌──────────┬───────────────────┬────────────────────┬───────────────────┐
  │   Axis   │  Amplitudes from  │   Bloch Sphere     │   Best For        │
  │          │       |0⟩         │   Trajectory       │                   │
  ├──────────┼───────────────────┼────────────────────┼───────────────────┤
  │          │                   │                    │                   │
  │  RY ★    │  Real             │  X-Z great circle  │  General purpose  │
  │          │  cos(θ/2)|0⟩      │  (standard choice) │  Real classifiers │
  │          │ +sin(θ/2)|1⟩      │                    │                   │
  │          │                   │                    │                   │
  ├──────────┼───────────────────┼────────────────────┼───────────────────┤
  │          │                   │                    │                   │
  │  RX      │  Complex          │  Y-Z great circle  │  Complex-valued   │
  │          │  cos(θ/2)|0⟩      │                    │  representations  │
  │          │ -i·sin(θ/2)|1⟩    │                    │                   │
  │          │                   │                    │                   │
  ├──────────┼───────────────────┼────────────────────┼───────────────────┤
  │          │                   │                    │                   │
  │  RZ      │  Phase only       │  Equatorial circle │  Phase-based      │
  │          │  e^{-iθ/2}|0⟩     │  (no pop. change)  │  interference     │
  │          │  (global phase)   │                    │  (needs pre-rot.) │
  │          │                   │                    │                   │
  └──────────┴───────────────────┴────────────────────┴───────────────────┘
```

---

## 5. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     ANGLE ENCODING PROPERTIES                       │
  ├──────────────────────┬──────────────────────────────────────────────┤
  │  Qubits required     │  n  (one per feature)                       │
  │  Circuit depth       │  reps  (parallel rotations per layer)       │
  │  Total gates         │  n × reps  (all single-qubit)               │
  │  Two-qubit gates     │  0  (none — no entanglement)                │
  │  Entangling?         │  No  (product states only)                  │
  │  Simulability        │  Classically simulable  O(n)                │
  │  Trainability        │  High (~0.9)  no barren plateaus            │
  │  Expressibility      │  Limited  (no entangled correlations)       │
  │  Periodicity         │  2π  (RX, RY)  /  4π  (RZ full Bloch)      │
  │  Connectivity        │  None required  (no qubit coupling)         │
  └──────────────────────┴──────────────────────────────────────────────┘
```

### Why These Properties Matter

```
  Trainability ████████████████████░░  ~0.9   ← Excellent: no barren plateaus
  Expressibility ████████░░░░░░░░░░░░  ~0.4   ← Limited: product states only
  Hardware Cost ██░░░░░░░░░░░░░░░░░░░  Low    ← Minimal: single-qubit gates
  Noise Resilience ████████████████░░░░  High   ← Shallow depth, no CNOT errors
```

---

## 6. Comparison with Other Encodings

```
  ┌───────────────────┬──────────┬─────────┬────────────┬──────────────┐
  │    Encoding       │  Qubits  │  Depth  │ Entangling │ Simulability │
  ├───────────────────┼──────────┼─────────┼────────────┼──────────────┤
  │  Angle ★          │    n     │  reps   │     No     │  Simulable   │
  │  Basis            │    n     │    1    │     No     │  Simulable   │
  │  Amplitude        │  log₂n  │  O(2ⁿ) │    Yes     │  Not sim.    │
  │  IQP              │    n     │  O(n²) │    Yes     │  Not sim.    │
  │  ZZ Feature Map   │    n     │  O(n²) │    Yes     │  Not sim.    │
  └───────────────────┴──────────┴─────────┴────────────┴──────────────┘

  Angle encoding trades expressibility for trainability and efficiency.
  It is the natural baseline against which complex encodings are compared.
```

---

## 7. Data Preprocessing

Rotation gates are **periodic** — features should be scaled to use the
full rotation range without redundancy.

```
  Rotation angle
  ──────────────────────────────────────────────────►
  0              π              2π             3π
  |──────────────|──────────────|──────────────|
  |   ← useful range (0 to 2π) →   | redundant |
  |              |              |              |
  |0⟩          |+⟩           |0⟩           |+⟩    (RY example)
  full weight   equal         back to        equal
  on |0⟩        superposition |0⟩ again      again
```

### Recommended Scaling Methods

```
  Method                Formula                        Output Range
  ─────────────────────────────────────────────────────────────────
  Min-max              x' = 2π·(x - min)/(max - min)  [0, 2π]
  Arctan               x' = arctan(x)                  (-π/2, π/2)
  Double arctan        x' = 2·arctan(x)                (-π, π)
  Standard + arctan    x' = arctan((x - μ) / σ)        (-π/2, π/2)
```

---

## 8. Example Walkthrough

Encode `x = [π/3, π/2, π]` with `RY` rotation:

```
  Step 1: Initialize qubits
  ─────────────────────────
  |q₀⟩ = |0⟩     |q₁⟩ = |0⟩     |q₂⟩ = |0⟩

  Step 2: Apply RY rotations
  ──────────────────────────
  |q₀⟩ = RY(π/3)|0⟩  =  cos(π/6)|0⟩ + sin(π/6)|1⟩  =  0.866|0⟩ + 0.500|1⟩
  |q₁⟩ = RY(π/2)|0⟩  =  cos(π/4)|0⟩ + sin(π/4)|1⟩  =  0.707|0⟩ + 0.707|1⟩
  |q₂⟩ = RY(π)  |0⟩  =  cos(π/2)|0⟩ + sin(π/2)|1⟩  =  0.000|0⟩ + 1.000|1⟩

  Step 3: Full state (tensor product)
  ────────────────────────────────────
  |ψ⟩ = |q₀⟩ ⊗ |q₁⟩ ⊗ |q₂⟩

       = (0.866|0⟩ + 0.500|1⟩) ⊗ (0.707|0⟩ + 0.707|1⟩) ⊗ (0.000|0⟩ + 1.000|1⟩)

  Expanding in the computational basis {|000⟩, |001⟩, ..., |111⟩}:

  |ψ⟩ = 0.000|000⟩ + 0.612|001⟩ + 0.000|010⟩ + 0.612|011⟩
       + 0.000|100⟩ + 0.354|101⟩ + 0.000|110⟩ + 0.354|111⟩
```

```
  Measurement probabilities:

  |000⟩  ░░░░░░░░░░░░░░░░░░░░  0.000
  |001⟩  █████████████████░░░░  0.375
  |010⟩  ░░░░░░░░░░░░░░░░░░░░  0.000
  |011⟩  █████████████████░░░░  0.375
  |100⟩  ░░░░░░░░░░░░░░░░░░░░  0.000
  |101⟩  █████████░░░░░░░░░░░░  0.125
  |110⟩  ░░░░░░░░░░░░░░░░░░░░  0.000
  |111⟩  █████████░░░░░░░░░░░░  0.125
                                ─────
                                1.000
```

> Notice: qubit 2 was rotated by `π`, placing it entirely in |1⟩.
> Hence all amplitudes with qubit 2 = |0⟩ are zero.

---

## 9. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌───────────────────────────┐       ┌───────────────────────────────┐
  │                           │       │                               │
  │  ✓ Simplest encoding     │       │  ✗ Product states only        │
  │    Direct x → θ mapping   │       │    No entangled correlations  │
  │                           │       │                               │
  │  ✓ O(1) depth per layer  │       │  ✗ Linear qubit scaling       │
  │    All gates in parallel   │       │    n features → n qubits     │
  │                           │       │                               │
  │  ✓ High trainability     │       │  ✗ Limited expressibility     │
  │    No barren plateaus      │       │    Simple decision boundaries │
  │                           │       │                               │
  │  ✓ NISQ-friendly         │       │  ✗ Classically simulable      │
  │    Minimal gate errors     │       │    No quantum advantage       │
  │                           │       │                               │
  │  ✓ No connectivity needed│       │  ✗ Periodic redundancy        │
  │    Works on any topology   │       │    Features must be scaled    │
  │                           │       │                               │
  └───────────────────────────┘       └───────────────────────────────┘
```

---

## 10. Use Cases

```
                          Best suited for
                    ┌──────────────────────────┐
                    │                          │
  ┌─────────────────┤  NISQ Hardware           │  Minimal depth reduces
  │                 │  Experiments              │  decoherence effects
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Baseline                │  Entanglement-free reference
  │                 │  Benchmarking             │  for comparing encodings
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Variational             │  Guaranteed gradient flow
  │                 │  Classifiers              │  in trainable input layers
  │                 ├──────────────────────────┤
  │                 │                          │
  ├─────────────────┤  Hybrid                  │  Pairs well with classical
  │                 │  Architectures            │  neural network layers
  │                 ├──────────────────────────┤
  │                 │                          │
  └─────────────────┤  Classical               │  Fast prototyping via
                    │  Simulation               │  efficient O(n) simulation
                    └──────────────────────────┘
```

---

## 11. Resource Scaling

```
  n_features │  Qubits │  Gates (reps=1) │  Gates (reps=3) │  Depth
  ───────────┼─────────┼─────────────────┼─────────────────┼────────
       2     │    2    │        2        │        6        │  reps
       4     │    4    │        4        │       12        │  reps
       8     │    8    │        8        │       24        │  reps
      16     │   16    │       16        │       48        │  reps
      64     │   64    │       64        │      192        │  reps
     256     │  256    │      256        │      768        │  reps

  Scaling: Linear in features, constant depth per layer.
  All gates are single-qubit — no two-qubit gate overhead.
```

---

## References

1. Schuld, M., & Petruccione, F. (2018). "Supervised Learning with
   Quantum Computers." Springer.

2. LaRose, R., & Coyle, B. (2020). "Robust data encodings for quantum
   classifiers." Physical Review A, 102(3), 032420.

3. Weigold, M., et al. (2020). "Data encoding patterns for quantum
   computing." IEEE International Conference on Quantum Computing
   and Engineering (QCE).
