# Hamiltonian Encoding

**Author:** Ashutosh Mishra

A physics-motivated quantum data encoding that maps classical features into
quantum states through **time evolution under a data-dependent Hamiltonian**.
Creates rich **entanglement structures** via configurable two-qubit interactions.

```
 +=====================================================================+
 ||                                                                     ||
 ||   |psi(x)>  =  exp(-i H(x) t)  |+>^{(x)n}                        ||
 ||                                                                     ||
 ||   "Data shapes the Hamiltonian  -->  physics does the encoding"    ||
 ||                                                                     ||
 +=====================================================================+
```

---

## 1. The Core Idea

Instead of mapping features directly to rotation angles, Hamiltonian encoding
treats the input data as **couplings in a physical Hamiltonian** and then
**simulates the time evolution** of that system. The quantum state that
emerges after evolution is the encoded representation.

```
  Classical Data               Hamiltonian                   Quantum State
  +------------+        +---------------------+        +-------------------+
  | x_0 = 0.3  |--+     |                     |        |                   |
  | x_1 = 0.7  |--+--->-| H(x) = SUM c(x) P  |--exp-->| |psi> = e^{-iHt} |
  | x_2 = 0.1  |--+--->-|       data-dependent |        |          |+++>   |
  | x_3 = 0.9  |--+     |       Pauli strings  |        |  (entangled!)    |
  +------------+        +---------------------+        +-------------------+
                                 |                             |
                         Features become                 Entanglement
                         coupling strengths              encodes correlations
```

The initial state is `|+>^n = H^n |0...0>`, an **equal superposition** over
all computational basis states. This starting point gives the Hamiltonian
evolution maximal "room" to create complex quantum states.

```
  Step 1:  |0...0>  --[H (x) H (x) ... (x) H]-->  |+...+>  (uniform superposition)

  Step 2:  |+...+>  --[exp(-i H(x) t)]-------->  |psi(x)>  (entangled data state)
```

---

## 2. The Hamiltonian Framework

### Mathematical Definition

The data-dependent Hamiltonian `H(x)` is a sum of Pauli operators weighted
by functions of the input features:

```
  H(x)  =  SUM_j  c_j(x)  P_j

  where:
    c_j(x)  = coefficient derived from input features x
    P_j     = Pauli string (tensor product of I, X, Y, Z)
```

The encoding unitary is the **matrix exponential**:

```
  U(x)  =  exp(-i H(x) t)

  where:
    t = evolution_time parameter (controls encoding strength)
```

### Trotterization

Since terms in H(x) generally don't commute, the exponential is approximated
via **first-order Trotter decomposition**:

```
  exp(-i H(x) t)  ~=  [ PROD_j exp(-i c_j(x) (t/r) P_j) ]^r

  where r = reps (number of Trotter steps)

  Accuracy: error ~ O(t^2 / r)  -->  more reps = better approximation
```

```
  Exact Evolution        First-Order Trotter (r=1)       Higher-Order (r=3)
  =================      =========================       ===================

  e^{-i(A+B+C)t}   ~=   e^{-iAt} e^{-iBt} e^{-iCt}    [e^{-iAt/3} e^{-iBt/3} e^{-iCt/3}]^3

  (intractable for       (product of simple gates,        (more accurate,
   large systems)         easy to implement)               deeper circuit)
```

---

## 3. Four Hamiltonian Types

Each type creates a different interaction structure. The choice of Hamiltonian
determines what correlations are captured and the resulting entanglement.

### 3.1 IQP (Instantaneous Quantum Polynomial)

Diagonal Hamiltonian with Z and ZZ terms. Creates correlations in the
computational basis without basis changes.

```
  H_iqp(x)  =  SUM_i  x_i Z_i   +   SUM_{i<j}  f(x_i, x_j) Z_i Z_j

  Single-qubit:    x_i * Z_i           (local field)
  Two-qubit:       (pi - x_i)(pi - x_j) * Z_i Z_j  (pairwise coupling)
```

```
  Circuit for one Trotter step (4 qubits, full entanglement):

         +---+  +----------+
  |+> ---| H |--| RZ(2a_0) |--*-----------*-----------*-----------
         +---+  +----------+  |           |           |
         +---+  +----------+  |  +------+ |           |
  |+> ---| H |--| RZ(2a_1) |--+--| ZZ_01| *-----------*-----------
         +---+  +----------+     +------+ |  +------+ |
         +---+  +----------+              |  |      | |  +------+
  |+> ---| H |--| RZ(2a_2) |--------------+--| ZZ_02| *--| ZZ_12|--
         +---+  +----------+                 +------+ |  +------+
         +---+  +----------+                          |  +------+
  |+> ---| H |--| RZ(2a_3) |--------------------------+--| ZZ_03|--
         +---+  +----------+                             +------+
                                                     (+ ZZ_13, ZZ_23)

  a_i = time_step * x_i                (single-qubit angles)
  ZZ_ij uses angle = time_step * (pi - x_i)(pi - x_j)
```

### 3.2 XY Model

Non-diagonal Hamiltonian with XX and YY interactions. Creates coherent
superpositions through off-diagonal coupling --- enables "hopping" of
excitations between qubits.

```
  H_xy(x)  =  [SUM_i x_i Z_i]  +  SUM_{i<j}  f(x_i, x_j) (X_i X_j + Y_i Y_j)
                optional                        XX + YY interactions
```

### 3.3 Heisenberg Model

The most expressive type. Combines XX, YY, and ZZ interactions --- the
full isotropic spin-1/2 Heisenberg Hamiltonian.

```
  H_heis(x)  =  [SUM_i x_i Z_i]  +  SUM_{i<j}  f(x_i, x_j) (X_i X_j + Y_i Y_j + Z_i Z_j)
                  optional                         XX + YY + ZZ interactions
```

### 3.4 Pauli-Z

Simplest entangling type. Only Z and ZZ terms (identical to IQP in structure
but distinguished for API clarity).

```
  H_pz(x)  =  SUM_i  x_i Z_i   +   SUM_{i<j}  f(x_i, x_j) Z_i Z_j
```

### Comparison at a Glance

```
  +==============+============+============+===================+==============+
  |    Type      | Single-q   | Two-qubit  | Interaction       | Expressivity |
  |              | Terms      | Terms      | Character         |              |
  +==============+============+============+===================+==============+
  |  iqp         |  Z_i       |  ZZ        | Diagonal (phase)  | High         |
  |  pauli_z     |  Z_i       |  ZZ        | Diagonal (phase)  | High         |
  |  xy          |  Z_i (opt) |  XX + YY   | Off-diagonal      | Higher       |
  |  heisenberg  |  Z_i (opt) |  XX+YY+ZZ  | Full isotropic    | Highest      |
  +==============+============+============+===================+==============+
```

---

## 4. Two-Qubit Gate Decompositions

Each Pauli interaction `exp(-i * angle * PP)` is decomposed into native
gates (CNOT, RZ, H, S/Sdg). This is how the abstract Hamiltonian becomes
a real circuit.

### ZZ Interaction: `exp(-i * angle * Z_i Z_j)`

```
  q_i ----*-------------------*----
          |                   |
  q_j ----+---[ RZ(2*angle) ]--+----
         CNOT               CNOT

  Depth: 3 layers       Gates: 2 CNOT + 1 RZ
```

### XX Interaction: `exp(-i * angle * X_i X_j)`

```
  q_i --[H]---*-------------------*---[H]--
              |                   |
  q_j --[H]---+---[ RZ(2*angle) ]--+---[H]--
             CNOT               CNOT

  Depth: 5 layers       Gates: 4 H + 2 CNOT + 1 RZ

  Idea: H rotates X-basis -> Z-basis, apply ZZ, rotate back.
```

### YY Interaction: `exp(-i * angle * Y_i Y_j)`

```
  q_i --[Sdg]--[H]---*-------------------*---[H]--[S]--
                      |                   |
  q_j --[Sdg]--[H]---+---[ RZ(2*angle) ]--+---[H]--[S]--
                     CNOT               CNOT

  Depth: 7 layers       Gates: 2 Sdg + 4 H + 2 CNOT + 1 RZ + 2 S

  Idea: Sdg rotates Y-basis -> X-basis, H -> Z-basis, apply ZZ, reverse.
```

---

## 5. Rotation Angle Formulas

### Single-Qubit Terms

```
  angle_single  =  time_step  *  x_i

  where:  time_step = evolution_time / reps
```

Linear in feature value. Larger features produce larger rotations.

### Two-Qubit Terms (Quantum Kernel Feature Map)

```
  angle_two_qubit  =  time_step  *  (pi - x_i)(pi - x_j)
```

This formula comes directly from the quantum kernel literature (Havlicek
et al., 2019). It has a **critical point at x = pi** where interactions
vanish:

```
  Interaction Strength vs Feature Value
  (for x_i = x_j = x, time_step = 1)

  Strength
  |
  |**
  |  **
  |    **
  |      ***
  |         ****
  |             *****
  |                  ********
  |                          **********
  +---|---|---|---|---|---|---|---|---|--->  x
  0  0.5  1  1.5  2  2.5  3  pi  3.5

  Maximum at x=0:  angle = pi^2 ~ 9.87
  Zero at x=pi:    angle = 0  (CRITICAL POINT)

  WARNING: Features near pi (~3.14) produce near-zero interactions!
  Recommended input ranges: [0, 1] or [-1, 1]
```

---

## 6. Entanglement Topologies

The `entanglement` parameter controls which qubit pairs interact.
This determines both circuit depth and hardware connectivity requirements.

### Full Entanglement

Every pair of qubits interacts. Maximum expressivity, but O(n^2) gates.

```
  n=5 qubits, full entanglement:  C(5,2) = 10 pairs

  q0 ----*----*----*----*----
         |    |    |    |
  q1 ----+----|----|----|---------*----*----*----
              |    |    |         |    |    |
  q2 ---------+----|----|----     +----|----|----*----*----
                   |    |              |    |    |    |
  q3 --------------+----|---------     +----|----|----+----
                        |                   |    |
  q4 -------------------+-------------------+----+--------

  10 pairs: (0,1)(0,2)(0,3)(0,4)(1,2)(1,3)(1,4)(2,3)(2,4)(3,4)
  Requires: all-to-all qubit connectivity
```

### Linear Entanglement

Only nearest neighbors interact. Hardware-friendly, O(n) gates.

```
  n=5 qubits, linear entanglement:  4 pairs

  q0 ----*---------------------------------------------
         |
  q1 ----+----*----------------------------------------
              |
  q2 ---------+----*-----------------------------------
                   |
  q3 --------------+----*------------------------------
                        |
  q4 -------------------+------------------------------

  4 pairs: (0,1)(1,2)(2,3)(3,4)
  Requires: nearest-neighbor connectivity only
```

### Circular Entanglement

Nearest neighbors plus wrap-around. Captures periodic boundary conditions.

```
  n=5 qubits, circular entanglement:  5 pairs

  q0 ----*---------------------------------------------*----
         |                                              |
  q1 ----+----*-----------------------------------------|----
              |                                         |
  q2 ---------+----*-----------------------------------|----
                   |                                    |
  q3 --------------+----*------------------------------|----
                        |                               |
  q4 -------------------+------------------------------+----
                                                    wrap-around

  5 pairs: (0,1)(1,2)(2,3)(3,4)(4,0)
  Requires: ring topology connectivity
```

### Topology Scaling

```
  +===========+====================+==================+==================+
  | Topology  | Pairs              | CNOT per         | Connectivity     |
  |           |                    | Trotter step     | Required         |
  +===========+====================+==================+==================+
  | full      | n(n-1)/2           | O(n^2)           | All-to-all       |
  | linear    | n - 1              | O(n)             | Nearest-neighbor |
  | circular  | n                  | O(n)             | Ring             |
  +===========+====================+==================+==================+

  n_features |  full  |  linear  |  circular
  -----------+--------+----------+-----------
       4     |    6   |     3    |     4
       8     |   28   |     7    |     8
      16     |  120   |    15    |    16
      32     |  496   |    31    |    32
      64     | 2016   |    63    |    64
```

---

## 7. Complete Circuit Structure

The full circuit for one Trotter step with IQP type, 4 qubits, full
entanglement:

```
  Layer 1:         Layer 2:          Layer 3:
  Initial          Single-qubit      Two-qubit interactions
  superposition    Z rotations       (ZZ for each pair)

  |0> --[H]--------[RZ(2a_0)]--------*---[ZZ_01]---*---[ZZ_02]---*---[ZZ_03]--
                                      |             |             |
  |0> --[H]--------[RZ(2a_1)]--------+-------------*---[ZZ_12]---*---[ZZ_13]--
                                                    |             |
  |0> --[H]--------[RZ(2a_2)]------------------------+-------------*---[ZZ_23]--
                                                                  |
  |0> --[H]--------[RZ(2a_3)]------------------------------------------+------


  With reps=2 Trotter steps:

  |0> --[H]--[RZ]--[ZZ pairs]--||--[RZ]--[ZZ pairs]--
  |0> --[H]--[RZ]--[ZZ pairs]--||--[RZ]--[ZZ pairs]--   || = barrier
  |0> --[H]--[RZ]--[ZZ pairs]--||--[RZ]--[ZZ pairs]--
  |0> --[H]--[RZ]--[ZZ pairs]--||--[RZ]--[ZZ pairs]--
              ^                         ^
           Trotter step 1          Trotter step 2
```

For **Heisenberg** type, each pair gets XX + YY + ZZ:

```
  Each pair (i,j) in one Trotter step:

  q_i --[RZ]--[H]*[CNOT]*[RZ]*[CNOT]*[H]--[Sdg][H]*[CNOT]*[RZ]*[CNOT]*[H][S]--*[CNOT]*[RZ]*[CNOT]*--
  q_j --------[H]+------+[RZ]+------+[H]--[Sdg][H]+------+[RZ]+------+[H][S]--+------+[RZ]+------+--
               ^--- XX interaction ---^     ^---------- YY interaction ----------^  ^-- ZZ interaction --^
```

---

## 8. Key Properties

```
  +==========================================================================+
  |                 HAMILTONIAN ENCODING PROPERTIES                          |
  +============================+=============================================+
  |  Qubits required           |  n  (one per feature)                       |
  |  Circuit depth             |  1 + reps * (interaction depth per type)    |
  |  Initial state             |  |+>^n  (Hadamard layer)                   |
  |  Entangling?               |  YES  (two-qubit gates create entanglement)|
  |  Simulability              |  Not classically simulable                  |
  |  Trainable parameters      |  0  (all data-dependent)                    |
  |  Trainability              |  Depends on depth and entanglement          |
  |  Connectivity required     |  Depends on topology (none to all-to-all)  |
  +============================+=============================================+
```

### Gate Count by Hamiltonian Type (per Trotter step, p = number of pairs)

```
  +==============+=========+=======+=====+========+=======+
  |    Type      |    H    |  RZ   |  S  |  CNOT  | Total |
  +==============+=========+=======+=====+========+=======+
  |  iqp/pauli_z | n (init)| n + p |  0  |  2p    |       |
  |  xy          | n + 8p  | [n]+2p| 4p  |  4p    |       |
  |  heisenberg  | n + 8p  | [n]+3p| 4p  |  6p    |       |
  +==============+=========+=======+=====+========+=======+
                                                  [n] = if include_single_qubit_terms

  Where:  n = n_qubits,  p = n_entanglement_pairs

  For n=4 qubits, full entanglement (p=6), reps=1:

  +--------------+-----+-----+----+------+-------+
  |    Type      |  H  |  RZ |  S | CNOT | Total |
  +--------------+-----+-----+----+------+-------+
  |  iqp         |  4  |  10 |  0 |  12  |   26  |
  |  xy          | 52  |  16 | 24 |  24  |  116  |
  |  heisenberg  | 52  |  22 | 24 |  36  |  134  |
  +--------------+-----+-----+----+------+-------+
```

---

## 9. Depth Analysis and Parallelization

Two-qubit gates on **disjoint** qubit pairs can execute in parallel.
The parallelization depends on the entanglement topology and maps to
the **edge coloring** problem from graph theory.

```
  Topology      Graph           Chromatic Index       Parallel Layers
  =========     ===========     =================     ===============
  linear        Path P_n        2                     2
  circular      Cycle C_n       2 (n even) / 3 (n odd)   2 or 3
  full          Complete K_n    n-1 (n even) / n (n odd)  n-1 or n
```

### Example: Linear, n=6

```
  Pairs: (0,1) (1,2) (2,3) (3,4) (4,5)

  Group A (disjoint): (0,1)  (2,3)  (4,5)   <-- can run in parallel
  Group B (disjoint): (1,2)  (3,4)           <-- can run in parallel

  => 2 parallel layers instead of 5 sequential
```

### Depth per Trotter Step

```
  +============+============================+================================+
  |  Type      |  Depth per parallel group  |  Total depth per Trotter step  |
  +============+============================+================================+
  | iqp/pz     |  1(RZ) + 3(ZZ) * groups   |  1 + 3*G                       |
  | xy         |  [1(RZ)] + 5(XX)*G + 7(YY)*G  |  [1] + 12*G               |
  | heisenberg |  [1(RZ)] + 5(XX)*G + 7(YY)*G + 3(ZZ)*G  |  [1] + 15*G    |
  +============+============================+================================+
                                            G = number of parallel groups
  Total depth = 1 (Hadamard) + reps * (depth per Trotter step)
```

---

## 10. Parameters Guide

```
  +=============================+==========+==========================================+
  |  Parameter                  | Default  |  Effect                                  |
  +=============================+==========+==========================================+
  |                             |          |                                          |
  |  n_features                 | required |  Number of features = number of qubits.  |
  |                             |          |                                          |
  |  hamiltonian_type           |  'iqp'   |  Interaction structure:                  |
  |                             |          |  'iqp'        - ZZ (diagonal)            |
  |                             |          |  'pauli_z'    - ZZ (same as iqp)         |
  |                             |          |  'xy'         - XX + YY                  |
  |                             |          |  'heisenberg' - XX + YY + ZZ             |
  |                             |          |                                          |
  |  evolution_time             |   1.0    |  Total time t in exp(-iHt).              |
  |                             |          |  Larger = stronger encoding.             |
  |                             |          |  Too large = Trotter error.              |
  |                             |          |                                          |
  |  reps                       |    2     |  Number of Trotter steps.                |
  |                             |          |  More = better approximation,            |
  |                             |          |  deeper circuit.                         |
  |                             |          |                                          |
  |  entanglement               |  'full'  |  Qubit pair topology:                    |
  |                             |          |  'full'     - all pairs, O(n^2)          |
  |                             |          |  'linear'   - nearest-neighbor, O(n)     |
  |                             |          |  'circular' - ring, O(n)                 |
  |                             |          |                                          |
  |  include_single_qubit_terms |  True    |  Add Z_i single-qubit rotations.         |
  |                             |          |  Always True for iqp/pauli_z.            |
  |                             |          |  Optional for xy/heisenberg.             |
  |                             |          |                                          |
  |  max_pairs                  |  None    |  Limit number of qubit pairs.            |
  |                             |          |  Truncates full entanglement for         |
  |                             |          |  large n to control depth.               |
  |                             |          |                                          |
  |  insert_barriers            |  True    |  Barriers between Trotter steps          |
  |                             |          |  (visualization, prevent optimization).  |
  |                             |          |                                          |
  +=============================+==========+==========================================+
```

---

## 11. Example Walkthrough

Encode `x = [0.3, 0.7, 0.1]` with IQP Hamiltonian, `evolution_time=1.0`,
`reps=1`, `entanglement='full'`:

```
  Step 1: Compute time step
  -------------------------
  time_step = evolution_time / reps = 1.0 / 1 = 1.0

  Step 2: Compute single-qubit angles
  ------------------------------------
  a_0 = time_step * x_0 = 1.0 * 0.3 = 0.30
  a_1 = time_step * x_1 = 1.0 * 0.7 = 0.70
  a_2 = time_step * x_2 = 1.0 * 0.1 = 0.10

  Step 3: Compute two-qubit angles  (formula: t * (pi - x_i)(pi - x_j))
  ---------------------------------------------------------------------
  a_01 = 1.0 * (pi - 0.3)(pi - 0.7) = 2.841 * 2.441 = 6.935
  a_02 = 1.0 * (pi - 0.3)(pi - 0.1) = 2.841 * 3.041 = 8.640
  a_12 = 1.0 * (pi - 0.7)(pi - 0.1) = 2.441 * 3.041 = 7.424

  Step 4: Build circuit
  ---------------------
  |0> --[H]--[RZ(0.60)]--*--[CNOT]--[RZ(13.87)]--[CNOT]--*-----------..
                          |                                |
  |0> --[H]--[RZ(1.40)]--+--------------------------------+--*--------..
                                                              |
  |0> --[H]--[RZ(0.20)]--------------------------------------+--------..

  (continues with ZZ_02 and ZZ_12 interactions)
  Note: RZ angles are 2*a for the gate decomposition.

  Step 5: Result
  --------------
  The output state |psi(x)> is an ENTANGLED 3-qubit state.
  Unlike product-state encodings, individual qubits are correlated.
  Measurement outcomes exhibit quantum correlations from ZZ coupling.
```

---

## 12. Trotterization: Accuracy vs Depth

The Trotter approximation introduces error that decreases with more steps.
This is the fundamental tradeoff of Hamiltonian encoding.

```
  Trotter Error  ~  O( t^2 / r )

  where:  t = evolution_time,  r = reps

  For evolution_time=1.0:

  reps |  Error bound  |  Circuit depth multiplier
  -----+--------------+---------------------------
    1  |  O(1)         |  1x  (minimum depth)
    2  |  O(0.25)      |  2x
    4  |  O(0.0625)    |  4x
    8  |  O(0.0156)    |  8x
   16  |  O(0.0039)    |  16x

  In practice, reps=2 is sufficient for most ML tasks.
  Exact Trotter accuracy is less critical than for physics simulation
  because the ML model learns to use the encoded representation
  regardless of small approximation errors.
```

```
  Fidelity with exact evolution:

  reps=1   ||||||||||||________________  ~0.6     (rough approximation)
  reps=2   ||||||||||||||||||||________  ~0.85    (good for ML)
  reps=4   ||||||||||||||||||||||||||__  ~0.95    (high fidelity)
  reps=8   ||||||||||||||||||||||||||||  ~0.99    (near-exact)
```

---

## 13. Comparison with Related Encodings

```
  +===================+==========+=========+============+===============+
  |    Encoding       |  Qubits  |  Depth  | Entangling |  Approach     |
  +===================+==========+=========+============+===============+
  |  Hamiltonian *    |    n     | O(rn^2) |    YES     | Time evolution|
  |  IQP Encoding     |    n     | O(rn^2) |    YES     | Direct ZZ     |
  |  ZZ Feature Map   |    n     | O(rn^2) |    YES     | Direct ZZ     |
  |  Pauli Feature Map|    n     | O(rn^2) |    YES     | Pauli rotations|
  |  Angle            |    n     |  O(r)   |    NO      | Direct angles |
  |  Higher-Order     |    n     |  O(r)   |    NO      | Poly angles   |
  |  Basis            |    n     |  O(1)   |    NO      | Bit strings   |
  |  Amplitude        | log_2(n) | O(2^n)  |    YES     | State prep    |
  +===================+==========+=========+============+===============+

  * Hamiltonian encoding is distinguished by its PHYSICS-MOTIVATED design.
    It doesn't just apply gates --- it simulates a physical system.
```

### What Makes Hamiltonian Encoding Special?

```
  +-------------------------------------------------------------------+
  |  Feature              | IQP/ZZ/Pauli Maps  | Hamiltonian Encoding |
  +-----------------------+--------------------+----------------------+
  |  Gate application     | Direct, ad-hoc     | From Trotterized     |
  |                       |                    | time evolution        |
  |                       |                    |                       |
  |  evolution_time       | No analogue        | Explicit control      |
  |  parameter            |                    | over encoding strength|
  |                       |                    |                       |
  |  Multiple interaction | Choose one         | Choose from IQP, XY, |
  |  types                | (ZZ only)          | Heisenberg, Pauli-Z  |
  |                       |                    |                       |
  |  Physical motivation  | Circuit-centric    | Hamiltonian dynamics  |
  |                       |                    | (quantum simulation)  |
  |                       |                    |                       |
  |  Trotter framework    | Fixed structure    | Systematic accuracy   |
  |                       |                    | improvement via reps  |
  +-------------------------------------------------------------------+
```

---

## 14. Data Preprocessing

The two-qubit angle formula `(pi - x_i)(pi - x_j)` makes input scaling
especially important for Hamiltonian encoding.

```
  +==================+=================================+=======================+
  |  Input Range     | Two-qubit angle range           | Recommendation        |
  +==================+=================================+=======================+
  |  [0, 1]          | [4.60, 9.87] * time_step        | SAFE (recommended)    |
  |  [-1, 1]         | [4.60, 17.2] * time_step        | SAFE (wide range)     |
  |  [0, pi]         | [0, 9.87] * time_step           | CAUTION (includes 0)  |
  |  [0, 2*pi]       | includes critical point at pi   | AVOID                 |
  +==================+=================================+=======================+

  Critical point:  When x_i = pi, the factor (pi - x_i) = 0.
  This makes the two-qubit interaction VANISH.

  Recommended: Normalize features to [0, 1] or [-1, 1] before encoding.
```

---

## 15. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  +-----------------------------+     +-------------------------------+
  |                             |     |                               |
  |  + Physics-grounded design  |     |  - O(n^2) gates for full     |
  |    Hamiltonian dynamics     |     |    entanglement (quadratic)   |
  |    provide natural inductive|     |                               |
  |    bias                     |     |  - Critical point at x = pi   |
  |                             |     |    requires careful input     |
  |  + Rich entanglement        |     |    scaling                    |
  |    Creates highly entangled |     |                               |
  |    states via two-qubit     |     |  - Trotter error with few    |
  |    interactions             |     |    reps (accuracy vs depth)   |
  |                             |     |                               |
  |  + Flexible Hamiltonian     |     |  - Deep circuits may exhibit |
  |    Four types to match      |     |    barren plateaus in        |
  |    different problems       |     |    gradient-based training    |
  |                             |     |                               |
  |  + Explicit evolution_time  |     |  - Full entanglement needs   |
  |    Fine-grained control     |     |    all-to-all connectivity   |
  |    over encoding strength   |     |    (not always available)     |
  |                             |     |                               |
  |  + Not classically simulable|     |  - Heisenberg type has       |
  |    Potential for quantum    |     |    highest gate overhead      |
  |    advantage in kernels     |     |    (6 CNOT per pair per rep) |
  |                             |     |                               |
  +-----------------------------+     +-------------------------------+
```

---

## 16. When to Use Hamiltonian Encoding

```
                        Best suited for
                  +----------------------------+
                  |                            |
  +---------------+  Quantum Kernel Methods    |  Physics-motivated kernels
  |               |  and Classification        |  with configurable structure
  |               +----------------------------+
  |               |                            |
  +---------------+  Physics-Inspired ML       |  Molecular properties,
  |               |  Problems                  |  materials science,
  |               +----------------------------+  quantum chemistry
  |               |                            |
  +---------------+  Exploring Hamiltonian     |  Systematic comparison of
  |               |  Structures for ML         |  IQP vs XY vs Heisenberg
  |               +----------------------------+  for your specific task
  |               |                            |
  +---------------+  Hardware with Good        |  Leverage entanglement
  |               |  Two-Qubit Fidelity        |  when hardware supports it
  |               +----------------------------+
  |               |                            |
  +---------------+  Time-Series Data          |  Natural temporal structure
                  |  Encoding                  |  via evolution_time
                  +----------------------------+
```

---

## 17. Configuration Quick Reference

```
  +============================================+===================================+
  |  USE CASE                                  |  RECOMMENDED CONFIG               |
  +============================================+===================================+
  |  Standard quantum kernel (Havlicek 2019)   |  hamiltonian_type='iqp',          |
  |                                            |  entanglement='full'              |
  |                                            |                                   |
  |  NISQ-friendly (limited connectivity)      |  entanglement='linear', reps=2    |
  |                                            |                                   |
  |  Maximum expressivity                      |  hamiltonian_type='heisenberg',   |
  |                                            |  entanglement='full'              |
  |                                            |                                   |
  |  Coherent excitation transfer              |  hamiltonian_type='xy'            |
  |                                            |                                   |
  |  Large feature count (n > 20)              |  entanglement='linear' or         |
  |                                            |  max_pairs=100                    |
  |                                            |                                   |
  |  Periodic boundary conditions              |  entanglement='circular'          |
  |                                            |                                   |
  |  High Trotter fidelity                     |  reps=4+, evolution_time=1.0      |
  |                                            |                                   |
  |  Strong encoding (wider rotations)         |  evolution_time=2.0 or higher     |
  |                                            |  (increase reps proportionally)   |
  +============================================+===================================+
```

---

## 18. Resource Scaling

```
  IQP type, full entanglement, reps=2:

  n_features | Pairs | H gates | RZ gates | CNOT gates | Total | Depth
  -----------+-------+---------+----------+------------+-------+------
       2     |   1   |    2    |     6    |      4     |   12  |   7
       4     |   6   |    4    |    20    |     24     |   48  |  25
       6     |  15   |    6    |    42    |     60     |  108  |  31
       8     |  28   |    8    |    72    |    112     |  192  |  43
      10     |  45   |   10    |   110    |    180     |  300  |  55
      16     |  120  |   16    |   272    |    480     |  768  |  91

  Linear entanglement, reps=2 (same n values):

  n_features | Pairs | CNOT gates | Total | Depth
  -----------+-------+------------+-------+------
       4     |   3   |     12     |   30  |  13
       8     |   7   |     28     |   66  |  13
      16     |  15   |     60     |  136  |  13
      32     |  31   |    124     |  272  |  13
      64     |  63   |    252     |  544  |  13

  Key: linear entanglement has CONSTANT depth (independent of n)
  because all nearest-neighbor pairs fit in 2 parallel groups!
```

---

## References

1. Schuld, M., Sweke, R., & Meyer, J. J. (2021). "Effect of data encoding on
   the expressive power of variational quantum-machine-learning models."
   Physical Review A, 103(3), 032430.

2. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced
   feature spaces." Nature, 567(7747), 209-212.