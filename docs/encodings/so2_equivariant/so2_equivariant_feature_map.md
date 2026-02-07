# SO(2) Equivariant Feature Map

**Author:** Ashutosh Mishra

A quantum data encoding with **provable rotation equivariance** -- the quantum
state transforms predictably under 2D rotations of the input, guaranteeing
that the encoding respects the continuous symmetry of the SO(2) group.

```
 +=====================================================================+
 |                                                                     |
 |   |psi(r, theta)> = SUM_m  c_m(r) * e^{im*theta} |m>              |
 |                                                                     |
 |   "Radial amplitudes set the weights,                               |
 |    angular phases encode direction,                                 |
 |    rotation in input = rotation in Hilbert space"                   |
 |                                                                     |
 +=====================================================================+
```

---

## 1. The Core Idea

Given a 2D point (x, y), this encoding converts it to polar coordinates
(r, theta) and maps it into a quantum state where:

- The **angle** theta is encoded as **phases** on angular momentum eigenstates
- The **radius** r is encoded as **amplitudes** across those eigenstates
- A **rotation** of the input point by phi produces the *exact same* effect as
  applying the rotation operator U(phi) to the quantum state

```
                        Cartesian             Polar
                        (x, y)       --->     (r, theta)
                          |                      |
                          |                      |
                          v                      v
                    +-----------+          +-------------+
                    |  SO(2)    |          |  Amplitude  |
                    |  Encoding |          |  + Phase    |
                    +-----------+          +-------------+
                          |                      |
                          v                      v
                    |psi(r, theta)>  =  SUM_m  c_m(r) * e^{im*theta} |m>
```

This is not just a design choice -- it is a **mathematical guarantee**:

```
  +-------------------------------------------------------------------+
  |  EQUIVARIANCE PROPERTY                                            |
  |                                                                   |
  |  For ANY rotation angle phi:                                      |
  |                                                                   |
  |     U(phi) |psi(x, y)>  =  |psi( rotate(x, y, phi) )>           |
  |     ~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       |
  |     Apply rotation          Encode the                            |
  |     operator on the         rotated input                         |
  |     quantum state           directly                              |
  |                                                                   |
  |  Both paths produce the SAME quantum state (up to global phase).  |
  +-------------------------------------------------------------------+
```

> This means any downstream quantum model inherits rotation symmetry
> automatically -- the hypothesis space is constrained to functions
> that respect SO(2), improving sample efficiency and generalization.

---

## 2. Mathematical Construction

### Step 1: Polar Decomposition

The 2D input is decomposed into rotation-invariant and rotation-covariant parts:

```
  (x, y)  ---->  r = sqrt(x^2 + y^2)       (invariant under rotation)
                 theta = arctan2(y, x)      (shifts by phi under rotation)
```

### Step 2: Angular Momentum Basis

The encoding uses angular momentum eigenstates |m>, where
m in {-max_m, ..., -1, 0, +1, ..., +max_m}:

```
  max_angular_momentum = 1  (3 states, 2 qubits)
  ~~~~~~~~~~~~~~~~~~~~~~~~

    m = +1  <--->  |00>                    Spin-1 triplet
    m =  0  <--->  (|01> + |10>) / sqrt(2)   (symmetric subspace
    m = -1  <--->  |11>                        of 2 qubits)

  max_angular_momentum = 2  (5 states, 3 qubits)
  ~~~~~~~~~~~~~~~~~~~~~~~~

    m = -2  <--->  |000>         Standard computational
    m = -1  <--->  |001>         basis ordering for
    m =  0  <--->  |010>         general case
    m = +1  <--->  |011>
    m = +2  <--->  |100>
```

### Step 3: State Construction

The quantum state is built by combining radial amplitudes with angular phases:

```
                    c_{-m}(r)                       c_{+m}(r)
                       |          c_0(r)                |
                       |            |                   |
              +--------v---+  +----v------+  +---------v--+
  |psi>  =   | e^{-im*th} |  | e^{i*0*th}|  | e^{+im*th} |
              |    |m=-m>  |  |    |m=0>   |  |    |m=+m>  |
              +------------+  +-----------+  +------------+
                   |               |                |
                   +-------+-------+-------+--------+
                           |                                 th = theta
                           v
                    Normalized quantum state
                    |psi(r, theta)>
```

Written explicitly:

```
  |psi(r, theta)>  =  SUM_{m = -max_m}^{+max_m}  c_m(r) * e^{i*m*theta} * |m>

  where:
    c_m(r) = radial amplitude coefficient (real, non-negative, normalized)
    e^{i*m*theta} = angular phase factor
    |m> = angular momentum basis state in computational basis
```

### Step 4: The Rotation Operator

The unitary representation of a rotation by angle phi acts as:

```
  U(phi) |m>  =  e^{+i*m*phi} |m>

  Therefore:

  U(phi) |psi(r, theta)>
    = SUM_m  c_m(r) * e^{i*m*theta} * U(phi)|m>
    = SUM_m  c_m(r) * e^{i*m*theta} * e^{+i*m*phi} |m>
    = SUM_m  c_m(r) * e^{i*m*(theta + phi)} |m>          <--- theta shifted!
    = |psi(r, theta + phi)>
    = |psi( rotate(x, y, phi) )>                           Q.E.D.
```

This is why equivariance holds exactly: the rotation operator simply
shifts the angular phases, which is equivalent to rotating the input.

---

## 3. Radial Functions

The coefficients c_m(r) control how the radius r distributes amplitude
across angular momentum states. Two options are available:

### Gaussian (default)

```
  c_m(r)  ~  exp( -(r - |m|)^2 / (2 * sigma^2) )     then normalized

  Effect: amplitude is concentrated on |m| closest to r

  Example (sigma=1.0):

  r = 0.0:  c_{-2} = 0.02  c_{-1} = 0.24  c_0 = 0.93  c_1 = 0.24  c_2 = 0.02
                                              ^
                                         peaked at m=0

  r = 1.5:  c_{-2} = 0.11  c_{-1} = 0.46  c_0 = 0.46  c_1 = 0.46  c_2 = 0.11
                                    ^               ^
                              shared between |m|=1 and |m|=2

  r = 3.0:  c_{-2} = 0.24  c_{-1} = 0.02  c_0 = 0.00  c_1 = 0.02  c_2 = 0.24
                  ^                                                      ^
            peaked at |m|=2 (near r=3)
```

### Uniform

```
  c_m(r) = 1 / sqrt(2*max_m + 1)        for all m

  Effect: radius is ignored, only direction matters

  Example (max_m = 2):

    c_m = 1/sqrt(5) ~ 0.447   for every m

  Use this when your data has meaningful direction
  but distance from origin is irrelevant.
```

---

## 4. Circuit Implementation

The encoding prepares an arbitrary quantum state using state preparation:

```
  |0>^{n}  ----[ StatePrep(|psi(r,theta)>) ]---->  |psi(r,theta)>

  Where |psi(r,theta)> is the target state vector computed classically:
    1. Convert (x,y) to (r, theta)
    2. Compute c_m(r) for all m
    3. Compute phases e^{i*m*theta} for all m
    4. Assemble target state
    5. Use StatePrep (PennyLane) or initialize (Qiskit) to prepare it
```

For the **spin-1 case** (max_m = 1, 2 qubits), the circuit prepares:

```
  Qubit 0: __|___________________________
              |
  Qubit 1: __|___________________________
              |
         [ StatePrep ]
              |
              v

  |psi> = c_1(r)*e^{i*theta} |00>
        + c_0(r)              (|01> + |10>)/sqrt(2)
        + c_{-1}(r)*e^{-i*theta} |11>
```

---

## 5. Qubit Requirements

```
  +---------------------+----------+------------+---------+
  | max_angular_momentum | # States | # Qubits   | Hilbert |
  |  (max_m)            | 2m+1     | ceil(log2)  | dim 2^n |
  +---------------------+----------+------------+---------+
  |        0            |    1     |     1      |    2    |
  |        1            |    3     |     2      |    4    |
  |        2            |    5     |     3      |    8    |
  |        3            |    7     |     3      |    8    |
  |        4            |    9     |     4      |   16    |
  |        7            |   15     |     4      |   16    |
  +---------------------+----------+------------+---------+

  n_qubits = max(1, ceil(log2(2 * max_m + 1)))

  NOTE: State preparation has O(2^n) gate complexity.
  For max_m > 3, a warning is emitted about scalability.
```

---

## 6. Equivariance Verification

The encoding provides three verification methods with different trade-offs:

```
  +----------------------------+-----------+-----------+-------------+
  | Method                     | Memory    | Time      | Scalability |
  +----------------------------+-----------+-----------+-------------+
  | verify_equivariance()      | O(2^n)    | O(2^n)    | <= 12 qubits|
  |   exact state comparison   |           |           |             |
  +----------------------------+-----------+-----------+-------------+
  | verify_statistical()       | O(shots)  | O(shots   | any size    |
  |   chi-squared on samples   |           |  * depth) |             |
  +----------------------------+-----------+-----------+-------------+
  | verify_auto()              | auto      | auto      | any size    |
  |   picks best method        |           |           |             |
  +----------------------------+-----------+-----------+-------------+
```

### How Exact Verification Works

```
  Path A:  |psi(x)>  ---[ U(phi) ]--->  U(phi)|psi(x)>

  Path B:  rotate(x, phi)  ---[ encode ]--->  |psi(rotate(x, phi))>

  Check:  |<Path A | Path B>|  ~=  1.0   (up to global phase)
```

### How Statistical Verification Works

```
  Circuit 1:  |0> --[ encode(x) ]--[ U(phi) ]-- measure --> samples_A
  Circuit 2:  |0> --[ encode(rotate(x,phi)) ]-- measure --> samples_B

  Chi-squared test: H0 = "distributions are identical"
  p_value > significance  -->  equivariance holds (statistically)
```

---

## 7. Symmetry Properties at a Glance

```
  +---------------------------+---------------------------------------------+
  | Property                  | Value                                       |
  +---------------------------+---------------------------------------------+
  | Symmetry Group            | SO(2) -- continuous 2D rotations            |
  | Group Element             | phi in [0, 2*pi) -- rotation angle          |
  | Input Space               | R^2 (2D Cartesian coordinates)              |
  | Group Action              | (x,y) -> (x*cos(phi)-y*sin(phi),           |
  |                           |           x*sin(phi)+y*cos(phi))            |
  | Unitary Representation    | U(phi) = exp(+i*phi*L_z)                   |
  | Generator (L_z)           | L_z |m> = m |m>                            |
  | Features Encoded          | 2 (always -- x and y coordinates)           |
  | Trainable Parameters      | 0                                           |
  | Is Entangling             | Yes (via state preparation)                 |
  | Simulability              | Not classically simulable                   |
  +---------------------------+---------------------------------------------+
```

---

## 8. When to Use This Encoding

### Ideal Use Cases

```
  [x] Data with rotational symmetry in 2D
      - Molecular orientations on a surface
      - Radar/sonar signal processing
      - Image patches with rotation invariance
      - Robotic sensor readings in polar coordinates

  [x] Problems where sample efficiency matters
      - Constraining to SO(2)-symmetric functions
        reduces the hypothesis space exponentially

  [x] Physics simulations
      - 2D quantum systems with angular momentum
      - Particle scattering in 2D
```

### When NOT to Use

```
  [ ] High-dimensional data (> 2 features)
      - SO(2) equivariance is defined for 2D only
      - Consider CyclicEquivariant or SwapEquivariant instead

  [ ] Data without rotational symmetry
      - The encoding *forces* rotation equivariance
      - If your problem is not rotation-symmetric, this adds bias

  [ ] Large-scale hardware deployment
      - State preparation is O(2^n) gates
      - Impractical beyond ~10 qubits on NISQ devices
```

---

## 9. Data Preprocessing Guide

```
  Raw 2D Data                    Preprocessing                   Encoding
  ============                   ==============                  ========

  (x, y) with arbitrary    -->   Center around mean:        -->  SO2Equivariant
  scale and offset               x_c = x - mean(x)              FeatureMap
                                 y_c = y - mean(y)
                                                                 Rotation is now
                                 Optional: scale to              meaningful around
                                 unit variance                   the centroid

  IMPORTANT:
    - No normalization required (encoding handles arbitrary magnitudes)
    - Centering is recommended so rotational symmetry is around a
      meaningful point
    - Choose radial_function="gaussian" if distance matters
    - Choose radial_function="uniform" if only direction matters
```

---

## 10. Comparison with Related Encodings

```
  +-------------------------+-------------+------------+----------+---------+
  | Encoding                | Symmetry    | Input Dim  | Qubits   | Gates   |
  +-------------------------+-------------+------------+----------+---------+
  | SO2 Equivariant         | SO(2)       | 2          | log(2m+1)| O(2^n)  |
  |   (this encoding)       | continuous  |            |          | state   |
  |                         | rotations   |            |          | prep    |
  +-------------------------+-------------+------------+----------+---------+
  | Cyclic Equivariant      | Z_n         | n          | n        | O(n)    |
  |                         | discrete    |            |          | per rep |
  |                         | shifts      |            |          |         |
  +-------------------------+-------------+------------+----------+---------+
  | Swap Equivariant        | S_2^k       | 2k         | 2k       | O(k)    |
  |                         | pair swaps  |            |          | per rep |
  +-------------------------+-------------+------------+----------+---------+
  | Angle Encoding          | none        | n          | n        | O(n)    |
  |                         |             |            |          |         |
  +-------------------------+-------------+------------+----------+---------+
  | Amplitude Encoding      | none        | 2^n        | n        | O(2^n)  |
  |                         |             |            |          |         |
  +-------------------------+-------------+------------+----------+---------+

  Key distinction: SO(2) Equivariant is the ONLY encoding in the atlas that
  guarantees continuous rotation symmetry. Other equivariant encodings handle
  discrete symmetries (cyclic shifts, pair swaps).
```

---

## 11. Quick Start

```python
from encoding_atlas import SO2EquivariantFeatureMap
import numpy as np

# Create encoder (max_angular_momentum=1 -> 2 qubits)
enc = SO2EquivariantFeatureMap(max_angular_momentum=1)

# Encode a 2D point
x = np.array([0.5, 0.3])
circuit = enc.get_circuit(x, backend='pennylane')

# Verify rotation equivariance
phi = np.pi / 4  # 45-degree rotation
assert enc.verify_equivariance(x, phi)  # True!

# Test multiple rotation angles
angles = [np.pi/6, np.pi/4, np.pi/2, np.pi, 2*np.pi]
assert all(enc.verify_equivariance(x, a) for a in angles)

# Inspect resources
summary = enc.resource_summary()
# summary['symmetry_group']   -> 'SO(2)'
# summary['n_qubits']         -> 2
# summary['angular_momenta']  -> [-1, 0, 1]
```

---

## 12. References

1. Larocca, M., et al. (2022). "Group-Invariant Quantum Machine Learning."
   *PRX Quantum*, 3, 030341.

2. Meyer, J.J., et al. (2023). "Exploiting Symmetry in Variational Quantum
   Machine Learning." *PRX Quantum*, 4, 010328.

3. Nguyen, Q., et al. (2022). "Theory for Equivariant Quantum Neural
   Networks." arXiv:2210.08566.

4. Schatzki, L., et al. (2022). "Theoretical Guarantees for
   Permutation-Equivariant Quantum Neural Networks." *npj Quantum Information*, 8, 74.

5. Skolik, A., et al. (2023). "Equivariant quantum circuits for learning on
   weighted graphs." *npj Quantum Information*, 9, 47.


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
