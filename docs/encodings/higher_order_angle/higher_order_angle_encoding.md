# Higher-Order Angle Encoding

**Author:** Ashutosh Mishra

A generalization of standard angle encoding that captures **non-linear feature
interactions** by using polynomial combinations of features as rotation angles,
while still producing **product states** (no entanglement).

```
 +===================================================================+
 ||                                                                   ||
 ||   |psi(x)>  =  (x)_q  R_a(theta_q) |0>                          ||
 ||                                                                   ||
 ||   where  theta_q  =  s * SUM_S  PROD_{i in S}  x_i               ||
 ||                                                                   ||
 ||   "Polynomial features  ->  rotation angles  ->  product state"   ||
 ||                                                                   ||
 +===================================================================+
```

---

## 1. The Core Idea

Standard angle encoding maps each feature **linearly** to a rotation angle:
`theta_q = x_q`. Higher-order angle encoding goes further --- it computes
**polynomial interaction terms** (products like `x_i * x_j`) and folds them
into the rotation angles via summation.

```
  Standard Angle Encoding          Higher-Order Angle Encoding (order=2)
  ========================          =====================================

  theta_0 = x_0                    theta_0 = x_0 + x_0*x_1
  theta_1 = x_1                    theta_1 = x_1 + x_0*x_2
  theta_2 = x_2                    theta_2 = x_2 + x_1*x_2

  Only linear features.            Linear + pairwise interactions
  No feature correlations.         baked into rotation angles.
```

The key insight from Schuld & Killoran (2019): encoding polynomial feature
interactions into rotation angles creates a **richer feature map** in
Hilbert space, capturing non-linear relationships between features ---
all **without any entangling gates**.

```
                    +--------- Feature Interactions ----------+
                    |                                          |
      STANDARD:    x_0        x_1        x_2                  |
                    |          |          |                    |
    HIGHER-ORDER:  x_0        x_1        x_2                  |
                   x_0*x_1    x_0*x_2    x_1*x_2    <-- NEW  |
                   x_0*x_1*x_2                       <-- NEW  |
                    |                                          |
                    +------------------------------------------+
                    These terms are SUMMED into qubit angles.
```

---

## 2. Term Generation

Terms are generated as all combinations of feature indices, ordered by
polynomial degree and then lexicographically within each degree.

### Example: `n_features=4, order=2`

```
  Order 1 (first-order):     Order 2 (second-order):
  C(4,1) = 4 terms           C(4,2) = 6 terms
  +-------------------+      +-------------------+
  | Index | Term      |      | Index | Term      |
  |-------+-----------|      |-------+-----------|
  |   0   | (0,)  x_0 |      |   4   | (0,1) x_0*x_1 |
  |   1   | (1,)  x_1 |      |   5   | (0,2) x_0*x_2 |
  |   2   | (2,)  x_2 |      |   6   | (0,3) x_0*x_3 |
  |   3   | (3,)  x_3 |      |   7   | (1,2) x_1*x_2 |
  +-------------------+      |   8   | (1,3) x_1*x_3 |
                              |   9   | (2,3) x_2*x_3 |
                              +-------------------+
                              Total: 4 + 6 = 10 terms
```

### Example: `n_features=3, order=3`

```
  Order 1         Order 2          Order 3
  C(3,1)=3        C(3,2)=3         C(3,3)=1
  +-----------+   +-------------+  +-----------------+
  | (0,) x_0  |   | (0,1) x_0*x_1 |  | (0,1,2) x_0*x_1*x_2 |
  | (1,) x_1  |   | (0,2) x_0*x_2 |  +-----------------+
  | (2,) x_2  |   | (1,2) x_1*x_2 |
  +-----------+   +-------------+  Total: 3 + 3 + 1 = 7 terms
```

### Term Count Formula

```
                           order
  Total terms  =   SUM     C(n, k)      where C(n,k) = n! / (k!(n-k)!)
                  k=start

  start = 1  if include_first_order = True
  start = 2  if include_first_order = False
```

### Scaling Table

```
  n_features | order=1 | order=2 | order=3 | order=4 |  order=n
  -----------+---------+---------+---------+---------+------------
       3     |    3    |    6    |    7    |    -    |      7
       4     |    4    |   10    |   14    |   15   |     15
       8     |    8    |   36    |   92    |  162   |    255
      10     |   10    |   55    |  175    |  385   |  1,023
      16     |   16    |  136    |  696    | 2,516  | 65,535

  WARNING: When order approaches n_features, term count grows
  toward 2^n - 1 (exponential). A warning is issued at >100 terms.
```

---

## 3. Term-to-Qubit Assignment (Round-Robin)

Terms are distributed across qubits using **round-robin by term index**:

```
  qubit_index  =  term_index  mod  n_qubits
```

This distributes terms **evenly**, regardless of which features appear
in each term.

### Worked Example: `n_features=3, order=2` (6 terms, 3 qubits)

```
  Terms generated:

  Index 0:  (0,)    x_0       --+
  Index 1:  (1,)    x_1       --|--+
  Index 2:  (2,)    x_2       --|--|--+
  Index 3:  (0,1)   x_0*x_1   --+  |  |
  Index 4:  (0,2)   x_0*x_2   -----+  |
  Index 5:  (1,2)   x_1*x_2   --------+

  Round-robin assignment:

  +-----------------------------------------------------------+
  |  Qubit 0:  indices 0, 3  -->  terms (0,), (0,1)           |
  |  Qubit 1:  indices 1, 4  -->  terms (1,), (0,2)           |
  |  Qubit 2:  indices 2, 5  -->  terms (2,), (1,2)           |
  +-----------------------------------------------------------+

  Note: term x_0*x_2 (index 4) goes to qubit 1, even though
  it contains features 0 and 2. Assignment is by INDEX, not
  by feature content.
```

---

## 4. Angle Computation

Each qubit's rotation angle is the **scaled sum** of all term values
assigned to it.

### Product Combination (default)

```
  theta_q  =  scaling  *  SUM  [ PROD x_i  for i in term ]
                          terms
                        on qubit q
```

### Sum Combination

```
  theta_q  =  scaling  *  SUM  [ SUM x_i  for i in term ]
                          terms
                        on qubit q
```

### Worked Example

`x = [0.5, 0.3, 0.8]`, `n_features=3`, `order=2`, `scaling=1.0`, `combination=product`:

```
  Qubit 0 terms:  (0,)  and  (0,1)
    term (0,)   = x_0            = 0.5
    term (0,1)  = x_0 * x_1     = 0.5 * 0.3 = 0.15
    theta_0 = 1.0 * (0.5 + 0.15) = 0.65

  Qubit 1 terms:  (1,)  and  (0,2)
    term (1,)   = x_1            = 0.3
    term (0,2)  = x_0 * x_2     = 0.5 * 0.8 = 0.40
    theta_1 = 1.0 * (0.3 + 0.40) = 0.70

  Qubit 2 terms:  (2,)  and  (1,2)
    term (2,)   = x_2            = 0.8
    term (1,2)  = x_1 * x_2     = 0.3 * 0.8 = 0.24
    theta_2 = 1.0 * (0.8 + 0.24) = 1.04
```

---

## 5. Circuit Structure

All rotation gates execute **in parallel** --- no two-qubit gates exist.
Circuit depth equals `reps`, independent of qubit count or term count.

### Single Layer (`reps=1`)

```
        +-------------------+
  |0> --| R_a(theta_0)      |--   theta_0 = s*(x_0 + x_0*x_1)
        +-------------------+
        +-------------------+
  |0> --| R_a(theta_1)      |--   theta_1 = s*(x_1 + x_0*x_2)
        +-------------------+
        +-------------------+
  |0> --| R_a(theta_2)      |--   theta_2 = s*(x_2 + x_1*x_2)
        +-------------------+

  depth = 1                       All gates are independent.
  No CNOT / CZ / entangling gates anywhere.
```

### Multiple Layers (`reps=3`)

```
        +------------+ +------------+ +------------+
  |0> --| R_a(theta_0) |-| R_a(theta_0) |-| R_a(theta_0) |--  = R_a(3*theta_0)|0>
        +------------+ +------------+ +------------+
        +------------+ +------------+ +------------+
  |0> --| R_a(theta_1) |-| R_a(theta_1) |-| R_a(theta_1) |--  = R_a(3*theta_1)|0>
        +------------+ +------------+ +------------+
        +------------+ +------------+ +------------+
  |0> --| R_a(theta_2) |-| R_a(theta_2) |-| R_a(theta_2) |--  = R_a(3*theta_2)|0>
        +------------+ +------------+ +------------+

  Effective rotation:  R_a(theta)^reps = R_a(reps * theta)
```

> Repeating the encoding layer `reps` times simply **scales** the rotation
> angle, since consecutive same-axis rotations compose additively.

---

## 6. Comparison: Where Interactions Live

The fundamental distinction between Higher-Order Angle Encoding and
entanglement-based encodings (IQP, ZZ Feature Map):

```
  +=====================================================================+
  |              WHERE DO FEATURE INTERACTIONS HAPPEN?                   |
  +=====================================================================+
  |                                                                     |
  |  HIGHER-ORDER ANGLE:                                                |
  |                                                                     |
  |    Interactions are in the ANGLES (classical preprocessing).        |
  |    The quantum state is a PRODUCT state.                            |
  |                                                                     |
  |    x_0*x_1 --> theta  --> R(theta)|0>  (x)  R(theta')|0>            |
  |                                        ^^                           |
  |                              tensor product (no entanglement)       |
  |                                                                     |
  +---------------------------------------------------------------------+
  |                                                                     |
  |  IQP / ZZ FEATURE MAP:                                              |
  |                                                                     |
  |    Interactions are via ENTANGLING GATES (quantum correlations).    |
  |    The quantum state is an ENTANGLED state.                         |
  |                                                                     |
  |    x_0*x_1 --> ZZ(theta) --> |psi>  (entangled, non-separable)     |
  |                  ^^                                                 |
  |       two-qubit gate creates quantum correlations                   |
  |                                                                     |
  +=====================================================================+

  Result:    Higher-Order Angle  -->  Classically simulable (product state)
             IQP / ZZ Feature Map --> Hard to simulate (entangled state)
```

This is the core tradeoff: Higher-Order Angle Encoding gets richer feature
maps **cheaply** (no entangling gates, no connectivity requirements, no
two-qubit errors) but remains classically simulable. The entanglement-based
encodings can achieve quantum advantage but at the cost of hardware
requirements and noise sensitivity.

---

## 7. Key Properties

```
  +=========================================================================+
  |              HIGHER-ORDER ANGLE ENCODING PROPERTIES                     |
  +===========================+=============================================+
  |  Qubits required          |  n  (one per feature, same as angle)        |
  |  Circuit depth            |  reps  (all rotations in parallel)          |
  |  Total gates              |  n * reps  (all single-qubit)               |
  |  Two-qubit gates          |  0  (none --- no entanglement)              |
  |  Polynomial terms         |  SUM_{k=1}^{order} C(n, k)                 |
  |  Entangling?              |  No  (product states only)                  |
  |  Simulability             |  Classically simulable  O(n)               |
  |  Trainability             |  High (~0.93-0.95)  no barren plateaus     |
  |  Expressibility           |  Higher than standard angle (polynomial)    |
  |  Connectivity required    |  None  (no qubit coupling needed)           |
  |  Trainable parameters     |  0  (all data-dependent)                    |
  +===========================+=============================================+
```

### Property Intuition

```
  Trainability   ||||||||||||||||||||||__  ~0.93   High: no entangling gates
  Expressibility ||||||||||||__________  ~0.55   Better than angle, below IQP
  Hardware Cost  ||||__________________  Low     Single-qubit gates only
  Noise Resil.   ||||||||||||||||||____  High    Shallow, no CNOT errors
  Feature Corr.  ||||||||||||__________  Moderate  Polynomial, not quantum
```

---

## 8. Parameters Guide

```
  +===========================================================================+
  |  Parameter           | Default  |  Effect                                 |
  +======================+==========+=========================================+
  |                      |          |                                         |
  |  n_features          | required |  Number of classical input features.    |
  |                      |          |  Determines qubit count (n_qubits = n). |
  |                      |          |                                         |
  |  order               |    2     |  Max polynomial degree of interactions. |
  |                      |          |  1 = standard angle encoding            |
  |                      |          |  2 = + pairwise products                |
  |                      |          |  3 = + triple products                  |
  |                      |          |  n = all 2^n - 1 terms (exponential!)   |
  |                      |          |                                         |
  |  rotation            |   'Y'    |  Rotation axis: 'X', 'Y', or 'Z'.      |
  |                      |          |  Y = real amplitudes (recommended).     |
  |                      |          |                                         |
  |  combination         |'product' |  How features combine in a term:        |
  |                      |          |  'product': x_i * x_j (multiplicative) |
  |                      |          |  'sum':     x_i + x_j (additive)       |
  |                      |          |                                         |
  |  include_first_order |  True    |  Include single-feature terms (x_i).    |
  |                      |          |  False = only interaction terms.         |
  |                      |          |                                         |
  |  scaling             |   1.0    |  Multiplicative factor on all angles.   |
  |                      |          |  Controls rotation range.               |
  |                      |          |                                         |
  |  reps                |    1     |  Number of encoding repetitions.        |
  |                      |          |  Equivalent to scaling angles by reps.  |
  |                      |          |                                         |
  +===========================================================================+
```

---

## 9. Reduction to Standard Angle Encoding

When `order=1`, Higher-Order Angle Encoding is **exactly** standard Angle
Encoding. This is by construction --- only first-order terms `(i,)` are
generated, and each qubit gets exactly one term:

```
  order=1, n_features=3:

  Terms: (0,)  (1,)  (2,)        <-- only first-order

  Qubit 0: theta_0 = scaling * x_0
  Qubit 1: theta_1 = scaling * x_1
  Qubit 2: theta_2 = scaling * x_2

  This is exactly:   |psi> = R(s*x_0)|0>  (x)  R(s*x_1)|0>  (x)  R(s*x_2)|0>

  = Standard Angle Encoding with a scaling factor.
```

---

## 10. Example Walkthrough

Encode `x = [0.5, 0.3, 0.8]` with `order=2, rotation='Y', scaling=1.0`:

```
  Step 1: Generate terms
  ----------------------
  First-order:   (0,)  (1,)  (2,)
  Second-order:  (0,1) (0,2) (1,2)

  Step 2: Assign terms to qubits (round-robin)
  ---------------------------------------------
  Qubit 0: (0,) and (0,1)    -->  x_0  and  x_0*x_1
  Qubit 1: (1,) and (0,2)    -->  x_1  and  x_0*x_2
  Qubit 2: (2,) and (1,2)    -->  x_2  and  x_1*x_2

  Step 3: Compute rotation angles
  --------------------------------
  theta_0 = 0.5 + (0.5)(0.3)    = 0.5 + 0.15 = 0.65
  theta_1 = 0.3 + (0.5)(0.8)    = 0.3 + 0.40 = 0.70
  theta_2 = 0.8 + (0.3)(0.8)    = 0.8 + 0.24 = 1.04

  Step 4: Apply RY rotations
  ---------------------------
  |q_0> = RY(0.65)|0> = cos(0.325)|0> + sin(0.325)|1> = 0.948|0> + 0.319|1>
  |q_1> = RY(0.70)|0> = cos(0.350)|0> + sin(0.350)|1> = 0.939|0> + 0.343|1>
  |q_2> = RY(1.04)|0> = cos(0.520)|0> + sin(0.520)|1> = 0.868|0> + 0.497|1>

  Step 5: Full state (tensor product)
  ------------------------------------
  |psi> = |q_0> (x) |q_1> (x) |q_2>

  = (0.948|0> + 0.319|1>) (x) (0.939|0> + 0.343|1>) (x) (0.868|0> + 0.497|1>)
```

```
  Measurement probabilities:

  |000>  ||||||||||||||||||||||||||||  0.674
  |001>  ||||||||||||||||             0.221
  |010>  ||||                        0.090
  |011>  ||                          0.029
  |100>  ||||                        0.076
  |101>  ||                          0.025
  |110>  |                           0.010
  |111>  |                           0.003
                                     -----
                                     ~1.00

  Dominated by |000> because all angles are < pi/2,
  keeping each qubit closer to |0> than to |1>.
```

---

## 11. Comparison with Other Encodings

```
  +========================+=========+=========+============+==============+
  |      Encoding          | Qubits  |  Depth  | Entangling | Simulability |
  +========================+=========+=========+============+==============+
  |  Angle                 |    n    |  reps   |     No     |  Simulable   |
  |  Higher-Order Angle *  |    n    |  reps   |     No     |  Simulable   |
  |  Basis                 |    n    |    1    |     No     |  Simulable   |
  |  Amplitude             | log_2 n |  O(2^n) |    Yes     |  Not sim.    |
  |  IQP                   |    n    |  O(n^2) |    Yes     |  Not sim.    |
  |  ZZ Feature Map        |    n    |  O(n^2) |    Yes     |  Not sim.    |
  |  Pauli Feature Map     |    n    |  O(n^2) |    Yes     |  Not sim.    |
  +========================+=========+=========+============+==============+

  * Higher-Order Angle has the SAME circuit cost as standard Angle,
    but with richer feature interactions encoded in the angles.
```

### The Expressibility-Efficiency Spectrum

```
  Expressibility
       ^
       |
  High |              Amplitude
       |                 *
       |        IQP  ZZ-FeatureMap
       |         *       *
       |      Pauli-FM
       |         *
       |
       |   Higher-Order Angle    <-- better than Angle,
       |         *                   same hardware cost
       |
       |   Angle
       |     *
       |
  Low  |   Basis
       |     *
       +------------------------------------->
       Low                              High
                 Hardware Cost
```

---

## 12. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  +-----------------------------+     +-------------------------------+
  |                             |     |                               |
  |  + Richer than angle        |     |  - Still a product state      |
  |    Polynomial feature       |     |    No quantum correlations    |
  |    interactions in angles   |     |    between qubits             |
  |                             |     |                               |
  |  + Same hardware cost       |     |  - Classically simulable      |
  |    as standard angle        |     |    Cannot provide quantum     |
  |    No entangling gates      |     |    computational advantage    |
  |                             |     |                               |
  |  + O(1) depth per layer     |     |  - Exponential term growth    |
  |    All gates in parallel    |     |    C(n,k) terms at order k    |
  |                             |     |    2^n - 1 when order = n     |
  |                             |     |                               |
  |  + High trainability        |     |  - Feature scale sensitivity  |
  |    No barren plateaus       |     |    Products amplify/diminish  |
  |                             |     |    feature magnitudes         |
  |                             |     |                               |
  |  + No connectivity needed   |     |  - Round-robin assignment     |
  |    Works on any topology    |     |    is heuristic, not          |
  |    and any quantum hardware |     |    information-theoretically  |
  |                             |     |    optimal                    |
  +-----------------------------+     +-------------------------------+
```

---

## 13. When to Use Higher-Order Angle Encoding

```
                        Best suited for
                  +----------------------------+
                  |                            |
  +---------------+  Feature Interactions      |  When you need pairwise
  |               |  Without Entanglement      |  (or higher) correlations
  |               +----------------------------+  but can't afford 2-qubit gates
  |               |                            |
  +---------------+  NISQ Hardware             |  Minimal depth, no CNOT
  |               |  with Limited Connectivity |  errors, any qubit topology
  |               +----------------------------+
  |               |                            |
  +---------------+  Enhanced Baselines        |  Strictly richer than angle
  |               |  for Benchmarking          |  encoding at no extra
  |               +----------------------------+  hardware cost
  |               |                            |
  +---------------+  Pre-entanglement Layer     |  Encode polynomial features
  |               |  in Variational Circuits   |  THEN add trainable
  |               +----------------------------+  entangling layers
  |               |                            |
  +---------------+  Classical Simulation      |  Product states allow
                  |  and Prototyping           |  efficient O(n) simulation
                  +----------------------------+
```

---

## 14. Resource Scaling

```
  n_features | order | Terms  | Qubits | Gates (r=1) | Gates (r=3) | Depth
  -----------+-------+--------+--------+-------------+-------------+------
       3     |   2   |    6   |    3   |      3      |      9      |  reps
       4     |   2   |   10   |    4   |      4      |     12      |  reps
       4     |   3   |   14   |    4   |      4      |     12      |  reps
       8     |   2   |   36   |    8   |      8      |     24      |  reps
       8     |   3   |   92   |    8   |      8      |     24      |  reps
      10     |   2   |   55   |   10   |     10      |     30      |  reps
      10     |   3   |  175   |   10   |     10      |     30      |  reps
      16     |   2   |  136   |   16   |     16      |     48      |  reps

  Key insight: gates and depth scale with n_features and reps ONLY.
  More terms don't add more gates --- they're SUMMED into existing angles.
  The cost is in CLASSICAL preprocessing (computing angle sums), not
  in quantum circuit complexity.
```

---

## 15. Implementation Configuration Quick Reference

```
  +---------------------------------------------------------------------------+
  |  USE CASE                           |  RECOMMENDED CONFIG                 |
  +-------------------------------------+-------------------------------------+
  |  Standard angle encoding equivalent |  order=1                            |
  |  Pairwise feature interactions      |  order=2 (default)                  |
  |  Rich feature interactions          |  order=3                            |
  |  Only interaction terms (no linear) |  include_first_order=False, order>=2|
  |  Real-valued amplitudes             |  rotation='Y' (default)             |
  |  Phase encoding                     |  rotation='Z'                       |
  |  Widen rotation range               |  scaling=2.0  or  reps=2            |
  |  Additive features (not products)   |  combination='sum'                  |
  +---------------------------------------------------------------------------+
```

---

## References

1. Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in
   Feature Hilbert Spaces." Physical Review Letters, 122(4), 040504.

2. Havlíček, V., et al. (2019). "Supervised learning with quantum-
   enhanced feature spaces." Nature, 567(7747), 209-212.

3. Stoudenmire, E., & Schwab, D. J. (2016). "Supervised learning with
   tensor networks." Advances in Neural Information Processing Systems.