# Swap (S_2) Equivariant Feature Map

**Author:** Ashutosh Mishra

A quantum data encoding with **provable pair-swap equivariance** -- swapping
two features within any pair in the input is exactly equivalent to applying a
SWAP gate on the corresponding qubit pair, guaranteeing the encoding respects
the discrete S_2 x S_2 x ... symmetry of paired data.

```
 +=====================================================================+
 |                                                                     |
 |   SWAP_{01} (x) SWAP_{23} (x) ...  |psi(x)>  =  |psi(swap . x)>  |
 |                                                                     |
 |   "Swap the qubits  <===>  swap the features.                      |
 |    The circuit doesn't care which came first."                      |
 |                                                                     |
 +=====================================================================+
```

---

## 1. The Core Idea

Many real-world datasets have **paired features** where the two elements in
a pair are interchangeable: particle/antiparticle measurements, buy/sell
signals, left/right sensor readings. For such data, the function we learn
should not depend on which element came first within each pair.

This encoding *enforces* that symmetry at the circuit level:

```
  Input:  x = [ x0, x1,  x2, x3,  x4, x5, ... ]
                ~~~~~     ~~~~~     ~~~~~
                pair 0    pair 1    pair 2

  Swap pair 0:  [ x1, x0,  x2, x3,  x4, x5 ]
                  ~~~~~
                  swapped

  Equivariance guarantee:

  +---------------------------------------------------------------------+
  |                                                                     |
  |  SWAP_{q0,q1}  |psi(x0, x1, x2, x3)>  =  |psi(x1, x0, x2, x3)>  |
  |                                                                     |
  |  Swapping qubits 0,1         Encoding the data with                 |
  |  in the quantum state   =    pair 0 features swapped                |
  |                                                                     |
  +---------------------------------------------------------------------+
```

The symmetry group is **S_2^k** (the direct product of k copies of S_2),
where k = n_features / 2. Each pair can be independently swapped or not,
giving 2^k distinct group elements.

---

## 2. Why CZ and Not CNOT?

This is the central design decision of the encoding, and the reason it
works. The entire equivariance proof hinges on one property:

```
  +---------------------------------------------------------------------+
  |  CZ is SYMMETRIC under qubit exchange:                              |
  |                                                                     |
  |    SWAP . CZ(q0, q1) . SWAP  =  CZ(q1, q0)  =  CZ(q0, q1)       |
  |                                                                     |
  |  CNOT is NOT symmetric:                                             |
  |                                                                     |
  |    SWAP . CNOT(q0, q1) . SWAP  =  CNOT(q1, q0)  =/=  CNOT(q0, q1)|
  |                                                                     |
  +---------------------------------------------------------------------+
```

In matrix form:

```
  CZ = |00><00| + |01><01| + |10><10| - |11><11|

       [ 1  0  0  0 ]
       [ 0  1  0  0 ]       Diagonal matrix --
       [ 0  0  1  0 ]       trivially symmetric
       [ 0  0  0 -1 ]


  CNOT = |00><00| + |01><01| + |10><11| + |11><10|

         [ 1  0  0  0 ]
         [ 0  1  0  0 ]     NOT symmetric --
         [ 0  0  0  1 ]     control/target roles differ
         [ 0  0  1  0 ]
```

The CZ gate applies a phase flip (-1) if and only if *both* qubits are |1>.
It does not distinguish between control and target. This symmetry is what
makes pair-swap equivariance possible.

---

## 3. Circuit Structure

### Single Layer (1 rep, 4 features = 2 pairs)

```
                   Layer 1:         Layer 2:        Layer 3:
                   Encode           Mix             Entangle
                   features         bases           pairs

  |0> -- q0 ---[ RY(x0) ]------[ H ]------*---------
                                           |
  |0> -- q1 ---[ RY(x1) ]------[ H ]------*---------
                                          CZ
                                     (symmetric!)
  |0> -- q2 ---[ RY(x2) ]------[ H ]------*---------
                                           |
  |0> -- q3 ---[ RY(x3) ]------[ H ]------*---------
                                          CZ

               Pair 0: q0, q1           Pair 1: q2, q3
```

### Multiple Layers (reps = 2)

```
  |0>--[ RY(x0) ]-[ H ]-*---[ RY(x0) ]-[ H ]-*---
  |0>--[ RY(x1) ]-[ H ]-*---[ RY(x1) ]-[ H ]-*---
                        CZ                    CZ
  |0>--[ RY(x2) ]-[ H ]-*---[ RY(x2) ]-[ H ]-*---
  |0>--[ RY(x3) ]-[ H ]-*---[ RY(x3) ]-[ H ]-*---
                        CZ                    CZ

       <------- rep 1 ------> <------- rep 2 ------>
```

Each repetition re-encodes the *same* data x, building deeper interference
patterns. The data is not rescaled between layers.

### The Three Sub-layers

```
  +-------------------+------------------------------------------+-------------+
  | Sub-layer         | Purpose                                  | Gate        |
  +-------------------+------------------------------------------+-------------+
  |                   |                                          |             |
  | 1. RY(x_i)       | Encode feature x_i directly on qubit i.  | RY          |
  |    on each qubit  | This is why swapping data = swapping     |             |
  |                   | qubits: the angle goes where the qubit   |             |
  |                   | goes.                                    |             |
  |                   |                                          |             |
  | 2. Hadamard       | Mix computational basis states.          | H           |
  |    on each qubit  | H (x) H commutes with SWAP because      |             |
  |                   | it's a tensor product of identical       |             |
  |                   | single-qubit gates.                      |             |
  |                   |                                          |             |
  | 3. CZ             | Entangle qubits within each pair.       | CZ          |
  |    within pairs   | CZ is symmetric under qubit exchange,    |             |
  |                   | preserving equivariance.                 |             |
  |                   |                                          |             |
  +-------------------+------------------------------------------+-------------+
```

---

## 4. Equivariance Proof

The proof proceeds by pushing the SWAP gate through each layer of the circuit.
Consider one pair (qubits 0 and 1) with one repetition:

```
  Circuit:  |psi(x0,x1)> = CZ . (H (x) H) . (RY(x0) (x) RY(x1)) . |00>

  Goal: show  SWAP |psi(x0,x1)> = |psi(x1,x0)>
```

**Step 1 -- SWAP commutes with CZ:**

```
  SWAP . CZ = CZ . SWAP          (CZ is symmetric)
```

**Step 2 -- SWAP commutes with H (x) H:**

```
  SWAP . (H (x) H) = (H (x) H) . SWAP      (tensor product of identical ops)
```

**Step 3 -- SWAP exchanges single-qubit gates:**

```
  SWAP . (RY(x0) (x) RY(x1)) = (RY(x1) (x) RY(x0)) . SWAP
```

**Step 4 -- SWAP acts trivially on |00>:**

```
  SWAP |00> = |00>
```

**Putting it together:**

```
  SWAP |psi(x0, x1)>
    = SWAP . CZ . (H(x)H) . (RY(x0)(x)RY(x1)) . |00>

  Push SWAP through CZ (Step 1):
    = CZ . SWAP . (H(x)H) . (RY(x0)(x)RY(x1)) . |00>

  Push SWAP through H(x)H (Step 2):
    = CZ . (H(x)H) . SWAP . (RY(x0)(x)RY(x1)) . |00>

  SWAP exchanges RY gates (Step 3):
    = CZ . (H(x)H) . (RY(x1)(x)RY(x0)) . SWAP . |00>

  SWAP|00> = |00> (Step 4):
    = CZ . (H(x)H) . (RY(x1)(x)RY(x0)) . |00>

    = |psi(x1, x0)>                                       Q.E.D.
```

For multiple pairs, the proof applies independently to each pair because
pairs occupy disjoint qubit pairs and the CZ gates do not cross pair
boundaries.

For multiple reps, the argument repeats layer by layer -- SWAP commutes
through every layer for the same reasons.

---

## 5. The Symmetry Group: S_2^k

Unlike SO(2) (a continuous group with infinite elements), the swap group
is **finite** and **discrete**:

```
  k pairs  -->  group S_2^k  -->  2^k elements

  +-------------------+-------+-------------------------------------------+
  | n_features (2k)   | Group | Elements                                  |
  +-------------------+-------+-------------------------------------------+
  | 2 (1 pair)        | S_2   | {id, swap_0}                       = 2   |
  | 4 (2 pairs)       | S_2^2 | {id, swap_0, swap_1, swap_both}    = 4   |
  | 6 (3 pairs)       | S_2^3 | all 3-bit combinations             = 8   |
  | 2k (k pairs)      | S_2^k | all k-bit boolean vectors          = 2^k |
  +-------------------+-------+-------------------------------------------+

  Each group element is a list of booleans:
    [True, False, True]  =  swap pair 0, keep pair 1, swap pair 2
    [False, False, False] = identity (no swaps)
    [True, True, True]    = swap all pairs
```

### Key group properties

```
  Involution:    swap . swap = identity     (swap^2 = id for each pair)
  Abelian:       order of swaps doesn't matter (pairs are independent)
  Generators:    k generators, one per pair   [True, False, ..., False]
                                              [False, True, ..., False]
                                              ...
                                              [False, False, ..., True]

  Testing equivariance on k generators is sufficient to prove it
  for all 2^k group elements (by the group homomorphism property).
```

---

## 6. Pair Topology and Entanglement

Entanglement is strictly **within** pairs. No CZ gate ever crosses a pair
boundary:

```
  6 features (3 pairs):

      q0 ---*--- q1       q2 ---*--- q3       q4 ---*--- q5
           CZ                  CZ                  CZ
        pair 0              pair 1              pair 2

  Entanglement pairs: [(0,1), (2,3), (4,5)]

  Connectivity required: pairs only (nearest-neighbor within pairs)

  This is EXTREMELY hardware-friendly:
    - No long-range connections needed
    - Each pair is an independent 2-qubit unit
    - Trivially maps to any hardware with nearest-neighbor connectivity
```

Compare with other entanglement topologies:

```
  +-------------------+---------------------+------------------------------+
  | Encoding          | Entanglement        | Connectivity Required        |
  +-------------------+---------------------+------------------------------+
  | Swap Equivariant  | Within pairs only   | Pairs: (0,1),(2,3),...       |
  |  (this encoding)  | [(0,1),(2,3),...]   | Minimal -- 2-qubit islands   |
  +-------------------+---------------------+------------------------------+
  | Cyclic Equivariant| Ring topology       | Ring: each qubit to next     |
  |                   | [(0,1),(1,2),...,    | + wrap-around (n-1,0)        |
  |                   |  (n-1,0)]           |                              |
  +-------------------+---------------------+------------------------------+
  | IQP Encoding      | All pairs (full)    | All-to-all                   |
  |                   | or linear/circular  | (most demanding)             |
  +-------------------+---------------------+------------------------------+
```

---

## 7. Gate Counts and Resources

```
  For n features (n/2 pairs), reps repetitions:

  +-------------------+----------------------------+-----------------------+
  | Gate Type         | Count per rep              | Total (reps = r)      |
  +-------------------+----------------------------+-----------------------+
  | RY (data encode)  | n                          | n * r                 |
  | H  (basis mix)    | n                          | n * r                 |
  | CZ (entangle)     | n/2                        | (n/2) * r             |
  +-------------------+----------------------------+-----------------------+
  | Single-qubit      | 2n                         | 2n * r                |
  | Two-qubit         | n/2                        | (n/2) * r             |
  | TOTAL             | 2n + n/2 = 5n/2            | (5n/2) * r            |
  +-------------------+----------------------------+-----------------------+

  Circuit depth per rep: 3  (RY layer + H layer + CZ layer)
  Total depth: 3 * reps

  Concrete examples:

  +----------+------+-------+--------+------+------+-------+-------+
  | features | pairs| qubits| reps   | RY   | H    | CZ    | total |
  +----------+------+-------+--------+------+------+-------+-------+
  |    4     |  2   |   4   |   1    |   4  |   4  |   2   |   10  |
  |    4     |  2   |   4   |   2    |   8  |   8  |   4   |   20  |
  |    6     |  3   |   6   |   2    |  12  |  12  |   6   |   30  |
  |    8     |  4   |   8   |   3    |  24  |  24  |  12   |   60  |
  |   20     |  10  |  20   |   2    |  40  |  40  |  20   |  100  |
  +----------+------+-------+--------+------+------+-------+-------+

  Scaling: O(n * reps) -- LINEAR in both features and reps.
  This is among the most efficient encodings in the atlas.
```

---

## 8. The Direct Encoding Principle

The equivariance of this encoding relies on a subtle but critical property:
**each feature is encoded directly on exactly one qubit** via RY(x_i).

```
  Direct encoding:   feature x_i  --->  RY(x_i) on qubit i

  This means:

    Swapping features (x0, x1) --> (x1, x0)

    is IDENTICAL to

    Swapping qubits q0 <--> q1 (i.e., applying SWAP gate)

  Because the ONLY place x_i appears in the circuit is on qubit i.
```

This would **break** if features were encoded differently, e.g.:

```
  WOULD NOT WORK:

    sum_val = (x0 + x1) / 2       <-- symmetric combination
    diff_val = (x0 - x1) / 2      <-- antisymmetric combination
    RY(sum_val) on q0
    RY(diff_val) on q1

  Here, swapping (x0, x1) changes diff_val to -diff_val,
  but SWAP(q0,q1) would put sum_val on q1 and diff_val on q0.
  These are NOT the same transformation.
```

The direct encoding principle is what makes the equivariance proof so
clean -- it reduces to "move the data, move the qubit."

---

## 9. Data Preprocessing

```
  +---------------------------------------------------------------------+
  | REQUIREMENTS                                                        |
  +---------------------------------------------------------------------+
  |                                                                     |
  | 1. Even number of features (n_features % 2 == 0)                   |
  |                                                                     |
  | 2. Features organized into meaningful pairs:                        |
  |    indices (2i, 2i+1) form pair i                                   |
  |                                                                     |
  |    Good:  [left_sensor, right_sensor, buy_price, sell_price]        |
  |           pair 0: (left, right)   pair 1: (buy, sell)               |
  |                                                                     |
  |    Bad:   [temperature, humidity, pressure, wind_speed]             |
  |           no natural pair structure --> swap symmetry is meaningless |
  |                                                                     |
  | 3. Scale features to rotation-friendly range for RY gates:          |
  |                                                                     |
  |    x_scaled = (x - mean) / std * (pi / 3)                          |
  |    (maps ~3 sigma to +/- pi, good coverage of Bloch sphere)        |
  |                                                                     |
  +---------------------------------------------------------------------+
```

---

## 10. When to Use This Encoding

### Ideal Use Cases

```
  [x] Data with natural pair structure
      - Particle/antiparticle features in HEP
      - Buy/sell or bid/ask in financial data
      - Left/right sensor readings in robotics
      - Symmetric interaction features (A->B and B->A)

  [x] Problems where pair order is arbitrary
      - Edge features in undirected graphs (node_i, node_j)
      - Pairwise distance features (d(a,b) = d(b,a))

  [x] Hardware-constrained deployments
      - Only needs nearest-neighbor pairs (no long-range coupling)
      - Linear gate count scaling
      - Shallow circuits (depth = 3 * reps)
```

### When NOT to Use

```
  [ ] Odd number of features
      - Strict requirement: n_features must be even

  [ ] No natural pair structure
      - Encoding forces swap symmetry on pairs (2i, 2i+1)
      - If pairs are arbitrary, the symmetry is meaningless bias

  [ ] Need continuous symmetry (use SO2 Equivariant instead)
      - Swap is discrete: only {id, swap} per pair
      - No smooth interpolation between swapped/unswapped

  [ ] Need global permutation symmetry
      - This encoding only handles swaps WITHIN pairs
      - Does not handle permutations ACROSS pairs
      - For full permutation symmetry, consider other approaches
```

---

## 11. Comparison with Related Encodings

```
  +-------------------------+----------+----------+----------+-----------+
  | Encoding                | Symmetry | Qubits   | CX/CZ    | Depth     |
  |                         | Group    |          | per rep  | per rep   |
  +-------------------------+----------+----------+----------+-----------+
  | Swap Equivariant        | S_2^k    | n        | n/2 (CZ) | 3         |
  |   (this encoding)       | discrete | = 2k     | within   |           |
  |                         | finite   |          | pairs    |           |
  +-------------------------+----------+----------+----------+-----------+
  | SO2 Equivariant         | SO(2)    | log(2m+1)| 0 (state | O(2^n)    |
  |                         | contin.  |          | prep)    |           |
  +-------------------------+----------+----------+----------+-----------+
  | Cyclic Equivariant      | Z_n      | n        | n (RZZ)  | ~3        |
  |                         | discrete |          | ring     |           |
  +-------------------------+----------+----------+----------+-----------+
  | Angle Encoding          | none     | n        | 0        | 1         |
  |                         |          |          |          |           |
  +-------------------------+----------+----------+----------+-----------+
  | Hardware Efficient      | none     | n        | varies   | varies    |
  |                         |          |          |          |           |
  +-------------------------+----------+----------+----------+-----------+

  Key distinction: Swap Equivariant is the only encoding in the atlas
  with PAIR-WISE discrete exchange symmetry. It has the lightest
  connectivity requirements of all entangling equivariant encodings.
```

---

## 12. Properties at a Glance

```
  +---------------------------+-----------------------------------------------+
  | Property                  | Value                                         |
  +---------------------------+-----------------------------------------------+
  | Symmetry Group            | S_2^k (direct product of k swap groups)       |
  | Group Element             | list[bool] of length k (which pairs to swap)  |
  | Group Order               | 2^k (k = n_features / 2)                     |
  | Input Space               | R^n where n is even                           |
  | Group Action              | swap features within selected pairs           |
  | Unitary Representation    | tensor product of SWAP and I_4 gates          |
  | Generators                | k single-pair swaps                           |
  | Features per Qubit        | 1 (direct encoding: x_i on qubit i)          |
  | Trainable Parameters      | 0                                             |
  | Is Entangling             | Yes (CZ gates within pairs)                   |
  | Simulability              | Not classically simulable                     |
  | Connectivity Required     | Pairs only (minimal)                          |
  | Native Gates              | RY, H, CZ                                    |
  +---------------------------+-----------------------------------------------+
```

---

## 13. Quick Start

```python
from encoding_atlas import SwapEquivariantFeatureMap
import numpy as np

# Create encoder for 4 features (2 pairs), 2 repetitions
enc = SwapEquivariantFeatureMap(n_features=4, reps=2)
print(enc.n_qubits)   # 4
print(enc.n_pairs)     # 2

# Encode data
x = np.array([0.5, -0.2, 0.8, 0.3])
circuit = enc.get_circuit(x, backend='pennylane')

# Verify swap equivariance -- swap first pair only
swaps = [True, False]
enc.verify_equivariance(x, swaps)    # True

# Verify all possible swap patterns
from itertools import product
for swap_pattern in product([True, False], repeat=enc.n_pairs):
    assert enc.verify_equivariance(x, list(swap_pattern))

# Inspect resources
summary = enc.resource_summary()
# summary['symmetry_group']          -> 'S_2^2'
# summary['gate_counts']['cz']       -> 4
# summary['entanglement_pairs']      -> [(0, 1), (2, 3)]
# summary['hardware_requirements']   -> {'connectivity': 'pairs',
#                                        'native_gates': ['RY', 'H', 'CZ']}
```

---

## 14. References

1. Schatzki, L., et al. (2022). "Theoretical Guarantees for
   Permutation-Equivariant Quantum Neural Networks." *npj Quantum
   Information*, 8, 74.

2. Larocca, M., et al. (2022). "Group-Invariant Quantum Machine Learning."
   *PRX Quantum*, 3, 030341.

3. Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational Quantum
   Machine Learning." *PRX Quantum*, 4, 010328.

4. Nguyen, Q., et al. (2022). "Theory for Equivariant Quantum Neural
   Networks." arXiv:2210.08566.


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
