# Cyclic Equivariant Feature Map

> A quantum data encoding with a mathematically provable guarantee: cyclically
> shifting the input features is exactly equivalent to cyclically permuting the
> qubits in the output quantum state.

---

## Core Idea

Most quantum encodings make no promises about how the output state relates to
symmetry transformations of the input. The Cyclic Equivariant Feature Map is
different -- it satisfies a **rigorous algebraic constraint**:

```
  SWAP_sigma |psi(x)>  =  |psi(sigma . x)>       for ALL sigma in Z_n

      ^                        ^
      |                        |
  Permute the QUBITS      Permute the INPUT DATA
  cyclically                cyclically
```

This means the quantum circuit "knows" about cyclic symmetry. If your data has
a natural ring or periodic structure, this encoding bakes that structure into
the quantum state by construction -- it doesn't need to learn it.

---

## The Z_n Cyclic Group

The symmetry group is **Z_n** -- the cyclic group of order n:

```
  Group elements:  {0, 1, 2, ..., n-1}
  Operation:       (k1 + k2) mod n       (addition modulo n)
  Identity:        0                      (no shift)
  Generator:       1                      (shift by one position)

  Example: Z_4 acting on input vector (a, b, c, d)

    k=0:  (a, b, c, d)   identity
    k=1:  (b, c, d, a)   shift left by 1
    k=2:  (c, d, a, b)   shift left by 2
    k=3:  (d, a, b, c)   shift left by 3
    k=4:  (a, b, c, d)   = k=0 (wraps around)
```

The encoding guarantees that for **every** one of these shifts, the
corresponding qubit permutation produces the same quantum state as encoding
the shifted data directly.

---

## Circuit Structure (4 qubits, reps=2)

Each repetition consists of exactly **three layers**:

```
         ┌──────────────── Rep 1 ────────────────┐┌──── Rep 2 ────...
         │  ENCODE       ENTANGLE       MIX      ││
         │                                        ││

q0: ──RY(x0)──ZZ(theta)──────────ZZ(theta)──RX(pi/6)──RY(x0)── ...
                  |                   |
q1: ──RY(x1)──ZZ(theta)──────────    |   ──RX(pi/6)──RY(x1)── ...
                  |                   |
q2: ──RY(x2)──ZZ(theta)──────────    |   ──RX(pi/6)──RY(x2)── ...
                  |                   |
q3: ──RY(x3)──ZZ(theta)──────────────┘   ──RX(pi/6)──RY(x3)── ...
               ring: (0,1)(1,2)(2,3)(3,0)
```

Spelled out, the three layers per repetition are:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       ONE REPETITION                                │
  │                                                                     │
  │  1. FEATURE ENCODING       2. RING ENTANGLEMENT     3. UNIFORM MIX │
  │                                                                     │
  │  q0: RY(x0)                q0──ZZ──q1               q0: RX(pi/6)   │
  │  q1: RY(x1)                q1──ZZ──q2               q1: RX(pi/6)   │
  │  q2: RY(x2)                q2──ZZ──q3               q2: RX(pi/6)   │
  │  q3: RY(x3)                q3──ZZ──q0  <-- wrap     q3: RX(pi/6)   │
  │                                                                     │
  │  Each qubit gets           Ring with periodic        SAME angle on  │
  │  its OWN feature           boundary (key for         ALL qubits     │
  │                             cyclic symmetry)         (translational │
  │                                                       invariance)   │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Why This Structure Guarantees Equivariance

The equivariance arises because **every layer** is translationally invariant
on the ring:

### Layer 1: Feature Encoding -- RY(x_i) on qubit i

```
  If we cyclically shift the input:   (x0, x1, x2, x3) --> (x1, x2, x3, x0)

  Then qubit i gets angle x_{(i+1) mod n} instead of x_i.

  This is EXACTLY what happens if we cyclically permute the qubits
  AFTER encoding the original data.  (RY is applied identically to each qubit,
  only the data value differs.)
```

### Layer 2: Ring Entanglement -- RZZ(theta) on (i, i+1 mod n)

```
  The entanglement pairs form a RING:

        q0 ── q1
        |       |        Pairs: (0,1), (1,2), (2,3), (3,0)
        q3 ── q2

  A cyclic shift of the qubits maps:
    (0,1) -> (1,2),  (1,2) -> (2,3),  (2,3) -> (3,0),  (3,0) -> (0,1)

  The ring maps to itself!  And since ALL RZZ gates use the SAME coupling
  strength theta, the entanglement layer is invariant under cyclic permutation.
```

### Layer 3: Uniform Mixing -- RX(pi/6) on all qubits

```
  Every qubit gets the EXACT SAME rotation angle (pi/6).
  Permuting qubits changes nothing -- this layer is trivially invariant.
```

Since each layer commutes with cyclic permutation, their composition does too.
Repeating `reps` times preserves the property. **QED.**

---

## The Coupling Strength Parameter

The `coupling_strength` (default: pi/4) controls the RZZ entangling gates:

```
  RZZ(theta) = exp(-i * theta/2 * Z (x) Z)

  theta = 0:      No entanglement (product state)
  theta = pi/4:   Default -- moderate entanglement
  theta = pi/2:   Maximum ZZ entanglement (CZ-like)
  theta = pi:     Wraps back (periodic)

  ┌──────────────────────────────────────────────────┐
  │  Choosing coupling_strength:                      │
  │                                                    │
  │  - Start with default pi/4                        │
  │  - Increase for stronger inter-feature            │
  │    correlations in the quantum state              │
  │  - Decrease for weaker coupling / less noise      │
  │  - Treat as a hyperparameter to tune              │
  └──────────────────────────────────────────────────┘
```

---

## Ring Topology: The Essential Ingredient

The ring (circular) entanglement with periodic boundary is not just a choice --
it is **required** for cyclic equivariance:

```
  Linear entanglement:              Ring entanglement:
  q0 ── q1 ── q2 ── q3             q0 ── q1
                                    |       |
  BREAKS cyclic symmetry!           q3 ── q2
  Shifting maps edge (2,3) to
  (3, ???) -- no connection         Shifting maps the ring to itself.
  at the boundary.                  Every edge maps to another edge.
                                    Cyclic symmetry is PRESERVED.
```

The entanglement pairs are always:

```
  (0, 1),  (1, 2),  (2, 3),  ...,  (n-2, n-1),  (n-1, 0)
                                                     ^
                                               wrap-around pair
  Total: n pairs (one per qubit)
```

---

## Equivariance Verification

The encoding provides built-in verification that the equivariance property
actually holds for any input:

```
  Verification Methods:
  ─────────────────────────────────────────────────────────────────────
  Method                    |  How It Works              |  Scalability
  ──────────────────────────+────────────────────────────+─────────────
  verify_equivariance(x, k) |  Computes state vectors,   |  <= 12 qubits
                             |  checks |<Ug.psi|psi_gx>| |  (exact)
                             |  = 1 up to global phase   |
  ──────────────────────────+────────────────────────────+─────────────
  verify_statistical(x, k)  |  Measures both circuits,   |  Any size
                             |  compares distributions    |  (sampling)
                             |  via chi-squared test      |
  ──────────────────────────+────────────────────────────+─────────────
  verify_auto(x, k)         |  Picks exact or            |  Any size
                             |  statistical based on      |  (automatic)
                             |  system size               |

  Testing on the GENERATOR (k=1) is sufficient to prove equivariance
  for ALL group elements, because Z_n is generated by the single shift.
```

### What the Verification Checks

```
  For input x and shift k:

  1. Encode x           -->  |psi(x)>
  2. Apply qubit perm   -->  SWAP_k |psi(x)>           (left side)
  3. Shift input        -->  sigma^k . x = roll(x, -k)
  4. Encode shifted     -->  |psi(sigma^k . x)>        (right side)
  5. Compare            -->  |<left|right>| = 1?        (up to phase)

  If YES for generator k=1  -->  equivariant for ALL k in {0,...,n-1}
```

---

## Resource Scaling

For `n` qubits (= features) and `r` repetitions:

```
  Resource              |  Formula    | Example (n=4, r=2)
  ──────────────────────+─────────────+────────────────────
  Qubits                |  n          |  4
  Circuit depth         |  3 * r      |  6
  RY gates (encoding)   |  n * r      |  8
  RZZ gates (ring)      |  n * r      |  8
  RX gates (mixing)     |  n * r      |  8
  Total single-qubit    |  2n * r     |  16
  Total two-qubit       |  n * r      |  8
  Total gates           |  3n * r     |  24
  Trainable parameters  |  0          |  0
```

Note: the two-qubit gate count scales **linearly** with `n` (not quadratically),
because the ring topology has exactly `n` edges.

---

## Group-Theoretic Properties

```
  Property                |  Value
  ────────────────────────+──────────────────────────────────────────
  Symmetry group          |  Z_n (cyclic group of order n)
  Group order             |  n
  Generator               |  sigma = shift by 1 position
  Group action on input   |  (x0,...,x_{n-1}) -> (x_k,...,x_{k+n-1 mod n})
  Unitary representation  |  Qubit permutation matrix (SWAP cascade)
  Equivariance type       |  Exact (provable, not approximate)
  Verification sufficient |  Test on generator k=1 only
  Invariance?             |  No -- equivariant, not invariant
                          |  (output TRANSFORMS with symmetry,
                          |   not stays fixed)
```

### Equivariance vs. Invariance

```
  EQUIVARIANCE (what this encoding does):
    Shift input  -->  Output state transforms predictably
    |psi(shift . x)>  =  SWAP_shift |psi(x)>

  INVARIANCE (a special case -- NOT what this encoding does):
    Shift input  -->  Output state unchanged
    |psi(shift . x)>  =  |psi(x)>

  To get INVARIANCE from an equivariant encoding, compose with an
  invariant measurement (e.g., measure total magnetization, not
  individual qubit states).
```

---

## Practical Considerations

### Data Preprocessing

```
  RY gates are 2pi-periodic:  RY(x) = RY(x + 2pi)

  Recommended:
    - Scale ALL features uniformly to [0, 2pi] or [-pi, pi]
    - IMPORTANT: Use the SAME scaling for all features!
      (Per-feature scaling breaks the cyclic symmetry.)
    - Preserve relative feature ordering (don't shuffle indices)
```

### When to Use This Encoding

```
  GOOD FIT:                              POOR FIT:
  ─────────                              ────────
  - Periodic time series                 - Data with no cyclic structure
  - Ring-structured molecules            - Features with different units
  - Cyclic signal processing             - Sparse / high-dimensional data
  - Spatial data on a circle             - Data where order doesn't matter
  - Rotational sensor arrays               (use permutation-equivariant
  - Phase-based measurements               instead)
```

### Hardware Considerations

```
  Hardware Type          |  Suitability  |  Notes
  ───────────────────────+───────────────+──────────────────────────────
  Ring-topology chips    |  Ideal        |  Native ring connectivity
  IBM heavy-hex          |  Good         |  Rings are subgraphs of hex
  Linear chains          |  Poor         |  Missing wrap-around edge
  Ion traps (all-to-all) |  Good         |  Ring is a subgraph
  Simulators             |  Perfect      |  No constraints
```

### Choosing reps

```
  reps=1     Minimal depth (3 layers), weak inter-feature mixing
  reps=2     Default -- good balance between depth and expressivity
  reps=3+    Stronger correlations, deeper circuit, more noise
```

---

## Comparison with Related Encodings

```
  Encoding                |  Symmetry   |  Entanglement   |  Guarantee
  ────────────────────────+─────────────+─────────────────+────────────
  Cyclic Equivariant      |  Z_n        |  Ring (RZZ)     |  Provable
  SO(2) Equivariant       |  SO(2)      |  State prep     |  Provable
  Swap Equivariant        |  S_2^k      |  Pair CZ        |  Provable
  Hardware-Efficient      |  None       |  Linear/Full    |  None
  QAOA-Style              |  None       |  Configurable   |  None
  Angle Encoding          |  None       |  None           |  None

  Equivariant encodings trade flexibility for mathematical guarantees.
  Use them when your data has known symmetry; use general encodings
  when symmetry is unknown or absent.
```

### Why Not Just Use a General Encoding?

```
  Without equivariance guarantee:
    - Model must LEARN cyclic symmetry from data
    - Requires more training samples
    - May learn approximate symmetry with errors
    - No theoretical guarantee on generalization

  With equivariance guarantee:
    - Cyclic symmetry is BUILT IN by construction
    - Reduced hypothesis space -> better sample efficiency
    - Exact symmetry (not approximate)
    - Provable generalization bounds [Meyer et al., 2023]
```

---

## Strengths and Limitations

### Strengths

- **Provable equivariance** -- mathematically guaranteed, not learned
- **Linear gate scaling** -- O(n) two-qubit gates per layer (ring topology)
- **Built-in verification** -- can check equivariance on any input
- **Sample efficiency** -- symmetry constraint reduces hypothesis space
- **Simple structure** -- three-layer design is easy to understand and debug
- **Single hyperparameter** -- only `coupling_strength` to tune (beyond `reps`)

### Limitations

- **Requires cyclic structure** -- data must have meaningful cyclic symmetry
- **Ring connectivity** -- needs wrap-around connection (not all hardware)
- **Fixed topology** -- cannot switch to linear or full entanglement
- **Not invariant** -- equivariant only; invariance requires additional
  post-processing
- **Uniform scaling required** -- per-feature normalization breaks symmetry

---

## References

1. Larocca, M., et al. (2022). "Group-Invariant Quantum Machine Learning."
   PRX Quantum 3, 030341. https://doi.org/10.1103/PRXQuantum.3.030341

2. Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational Quantum
   Machine Learning." PRX Quantum 4, 010328.

3. Schatzki, L., et al. (2022). "Theoretical Guarantees for Permutation-
   Equivariant Quantum Neural Networks." npj Quantum Information 8, 74.

4. Skolik, A., et al. (2023). "Equivariant quantum circuits for learning
   on weighted graphs." npj Quantum Information 9, 47.