# Hardware-Efficient Encoding

> A quantum data encoding designed to match native gate sets and connectivity
> constraints of near-term (NISQ) quantum hardware, minimizing gate decomposition
> overhead and reducing circuit depth.

---

## Overview

Hardware-efficient encoding maps classical data into quantum states using only
gates that are **native to the target device** and CNOT entanglement that
**respects physical qubit connectivity**. This avoids costly transpilation,
keeps circuits shallow, and preserves coherence on noisy hardware.

```
                                   reps
                            ┌────────┴────────┐
  |psi(x)> = [ U_ent . U_rot(x) ]^reps  |0>^n
                  |         |
                  |         +-- Data-dependent single-qubit rotations
                  +------------ Connectivity-respecting CNOT layer
```

Each repetition ("rep") consists of exactly **two layers**:

1. **Rotation layer** -- one R_alpha(x_i) gate per qubit
2. **Entanglement layer** -- CNOT gates following the chosen topology

---

## Circuit Structure (4 qubits, RY, linear, reps=2)

```
          ┌──── Rep 1 ─────┐┌──── Rep 2 ─────┐
          │  ROT     ENT   ││  ROT     ENT   │
          │                ││                │
q0: |0> ──RY(x0)──@────────RY(x0)──@────────
                   |                |
q1: |0> ──RY(x1)──X──@─────RY(x1)──X──@─────
                      |                |
q2: |0> ──RY(x2)─────X──@──RY(x2)─────X──@──
                         |                |
q3: |0> ──RY(x3)────────X──RY(x3)────────X──
```

**Reading the diagram:**
- `RY(xi)` = single-qubit rotation by angle `xi` around the Y-axis
- `@` = CNOT control qubit
- `X` = CNOT target qubit
- Features `x0..x3` are re-applied in every repetition

---

## Rotation Gate Options

The rotation axis determines which Bloch-sphere direction the data encodes into:

```
            Z                         Rotation  |  Gate   |  Bloch Axis
            |                         ----------+---------+-------------
            |    . theta              "X"       |  RX(x)  |  Around X
            |  .                      "Y"       |  RY(x)  |  Around Y  (default)
            | .                       "Z"       |  RZ(x)  |  Around Z
     -------+---------- Y
           /|
          / |                         RY creates real-valued amplitudes.
         /                            RZ is often a "virtual" gate on
        X                             superconducting hardware (zero error).
```

**Matrix definitions:**

```
         ┌                  ┐         ┌                  ┐         ┌             ┐
         │  cos(x/2)  -i sin(x/2) │         │  cos(x/2)  -sin(x/2) │         │ e^(-ix/2)    0    │
RX(x) =  │                  │  RY(x) = │                  │  RZ(x) = │             │
         │ -i sin(x/2)  cos(x/2) │         │  sin(x/2)   cos(x/2) │         │    0    e^(ix/2) │
         └                  ┘         └                  ┘         └             ┘
```

---

## Entanglement Topologies

Three connectivity patterns are available, matching different hardware architectures:

### Linear (default)

Nearest-neighbor chain. Matches linear qubit arrays on **superconducting chips**
(IBM, Google, Rigetti).

```
  q0 ──@
       |
  q1 ──X──@
          |
  q2 ─────X──@
             |
  q3 ────────X

  Pairs: (0,1), (1,2), (2,3)
  Count: n - 1
```

### Circular

Linear chain with wrap-around. Matches **ring topologies**.

```
  q0 ──@──────────X
       |          |
  q1 ──X──@       |
          |       |
  q2 ─────X──@    |
             |    |
  q3 ────────X──@─┘

  Pairs: (0,1), (1,2), (2,3), (3,0)
  Count: n
```

### Full

All-to-all connectivity. Matches **ion trap** devices (IonQ, Quantinuum).

```
  q0 ──@──@──@
       |  |  |
  q1 ──X  |  |──@──@
          |  |  |  |
  q2 ─────X  |──X  |──@
             |     |  |
  q3 ────────X─────X──X

  Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
  Count: n(n-1)/2   [O(n^2) scaling]
```

### Topology Comparison

```
  Topology  | CNOT/layer | Connectivity Required  | Best For
  ----------+------------+------------------------+----------------------------
  linear    |   n - 1    | Nearest-neighbor       | Superconducting (IBM, Google)
  circular  |   n        | Ring / wrap-around     | Ring topologies
  full      | n(n-1)/2   | All-to-all             | Ion traps (IonQ, Quantinuum)
```

---

## Resource Scaling

For `n` qubits and `r` repetitions:

```
  Resource              |  Formula              | Example (n=4, r=2, linear)
  ----------------------+-----------------------+---------------------------
  Qubits                |  n                    |  4
  Circuit depth         |  2 * r                |  4
  Single-qubit gates    |  n * r                |  8
  Two-qubit gates (lin) | (n-1) * r             |  6
  Two-qubit gates (cir) |  n * r                |  8
  Two-qubit gates (ful) |  n(n-1)/2 * r         |  12
  Total gates (linear)  |  n*r + (n-1)*r        |  14
  Parameters            |  n * r  (= rotations) |  8
```

---

## Mathematical Formulation

### State Preparation

Starting from the all-zero state, the encoding produces:

```
  |psi(x)> = ( U_ent . U_rot(x) )^r  |0>^{tensor n}
```

Where:

```
  U_rot(x) = R_alpha(x_0) (x) R_alpha(x_1) (x) ... (x) R_alpha(x_{n-1})
                                                    ^
                                                 tensor product

  U_ent    =   product     CNOT_{i,j}
             (i,j) in pairs
```

### Single Repetition Unitary (4 qubits, linear)

```
  U_rep(x) = CNOT_{2,3} . CNOT_{1,2} . CNOT_{0,1} . [RY(x3) (x) RY(x2) (x) RY(x1) (x) RY(x0)]
```

For `r` repetitions, the same unitary is applied `r` times with the **same data**
(features are re-uploaded each rep).

---

## Key Properties

```
  Property                |  Value / Behavior
  ------------------------+-----------------------------------------------
  Entangling?             |  Yes (when n > 1)
  Simulability            |  Not classically simulable (entangled states)
  Trainability estimate   |  0.8 (high -- simple structure avoids barren
                          |  plateaus better than random circuits)
  Data re-uploading       |  Yes (same features applied every rep)
  Native gate compatible  |  Yes (RX/RY/RZ + CNOT are universal native set)
  Feature-to-qubit ratio  |  1:1 (one qubit per feature)
```

---

## Practical Considerations

### Data Preprocessing

```
  Rotation gates are 2pi-periodic:  R(x) = R(x + 2pi)

  Recommended scaling:
    - Features to [0, 2pi] or [-pi, pi]
    - Standardize features if magnitudes differ significantly
    - Values beyond +/- 4pi trigger debug warnings
```

### Depth vs. Expressivity Trade-off

```
  reps=1     Low depth,  low expressivity    -->  Less noise, less power
  reps=2     Moderate    (default)            -->  Good balance for NISQ
  reps=3+    High depth, high expressivity    -->  More expressive, more noise

  Rule of thumb:  Start with reps=2, increase only if underfitting
```

### Hardware Choice Guide

```
  Hardware Type          |  Recommended Config
  -----------------------+--------------------------------------------
  IBM / Google / Rigetti |  rotation="Z" or "Y", entanglement="linear"
  (superconducting)      |  RZ is virtual (zero error) on many chips
                         |
  IonQ / Quantinuum      |  entanglement="full" (native all-to-all)
  (ion trap)             |  Any rotation axis
                         |
  Simulators             |  Any config (no hardware constraints)
```

---

## Strengths and Limitations

### Strengths

- **Minimal transpilation** -- circuits use native gates, no decomposition needed
- **Shallow depth** -- only `2 * reps` layers, preserving coherence
- **Good trainability** -- simple structure resists barren plateaus
- **Flexible topology** -- matches linear, ring, or all-to-all hardware
- **Multi-backend** -- works with PennyLane, Qiskit, and Cirq

### Limitations

- **Limited expressivity** -- shallow circuits may miss complex correlations
- **Topology constraints** -- circular/full may not match all hardware
- **O(n^2) scaling** -- full entanglement becomes expensive for large `n`
- **Barren plateaus** -- still possible at very high depth (`reps >> 1`)
- **Feature scaling sensitivity** -- performance depends on proper preprocessing

---

## Comparison with Other Encodings

```
  Encoding              |  Expressivity  |  Depth      |  NISQ-friendly
  ----------------------+----------------+-------------+----------------
  Hardware-Efficient    |  Moderate      |  2*reps     |  +++ (designed for it)
  Angle Encoding        |  Low           |  1          |  +++ (no entanglement)
  IQP Encoding          |  High          |  O(n^2)     |  +   (deep circuits)
  ZZ Feature Map        |  High          |  O(n^2)     |  +   (deep circuits)
  Data Reuploading      |  Universal*    |  Variable   |  ++  (tunable depth)

  * Universal approximation with sufficient parameters
```

---

## References

1. Kandala, A., et al. (2017). "Hardware-efficient variational quantum
   eigensolver for small molecules and quantum magnets." Nature,
   549(7671), 242-246.

2. Cerezo, M., et al. (2021). "Variational quantum algorithms."
   Nature Reviews Physics, 3(9), 625-644.

3. Bharti, K., et al. (2022). "Noisy intermediate-scale quantum algorithms."
   Reviews of Modern Physics, 94(1), 015004.