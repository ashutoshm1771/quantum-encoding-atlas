# Trainable Encoding

> A parameterized quantum encoding that **interleaves data-dependent rotations
> with learnable (trainable) parameters**, allowing the encoding itself to be
> optimized for a specific downstream task via variational training.

---

## Overview

Unlike fixed encodings where the circuit is entirely determined by input data,
trainable encoding introduces **variational parameters** that are updated through
classical optimization. This bridges the gap between pure data encoding and
variational ansatze, enabling the circuit to learn task-specific feature
representations.

```
                              L repetitions
                     ┌────────────┴────────────┐
  |psi(x,theta)> = [ U_ent . U_train(theta) . U_data(x) ]^L  |0>^n
                       |          |                |
                       |          |                +-- Data rotations  R_d(x_i)
                       |          +------------------- Trainable rotations R_t(theta_i)
                       +------------------------------ Entangling CNOT layer
```

The key insight: by making some rotation angles **learnable**, the encoding can
amplify relevant features, suppress irrelevant ones, and create task-specific
correlations -- all without manual feature engineering.

---

## Circuit Structure (4 qubits, RY data, RY trainable, linear, L=2)

```
         ┌──────────── Layer 1 ──────────────┐┌──────────── Layer 2 ──────────────┐
         │  DATA     TRAIN      ENTANGLE     ││  DATA     TRAIN      ENTANGLE     │

q0: |0> ─RY(x0)───RY(theta_0)──@─────────────RY(x0)───RY(theta_4)──@─────────────
                                |                                    |
q1: |0> ─RY(x1)───RY(theta_1)──X──@──────────RY(x1)───RY(theta_5)──X──@──────────
                                   |                                    |
q2: |0> ─RY(x2)───RY(theta_2)─────X──@───────RY(x2)───RY(theta_6)─────X──@───────
                                      |                                    |
q3: |0> ─RY(x3)───RY(theta_3)────────X───────RY(x3)───RY(theta_7)────────X───────
```

**Reading the diagram:**
- `RY(xi)` = data-encoding rotation (fixed by input)
- `RY(theta_i)` = trainable rotation (learned during optimization)
- `@` = CNOT control qubit
- `X` = CNOT target qubit
- Data features `x0..x3` are **re-uploaded** every layer
- Trainable parameters `theta_0..theta_7` are **unique per layer**

---

## Three Sublayers per Repetition

Each of the `L` layers consists of three distinct sublayers applied in sequence:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                          ONE LAYER                                  │
  │                                                                     │
  │   SUBLAYER 1          SUBLAYER 2            SUBLAYER 3              │
  │   Data Encoding       Trainable Rotation    Entanglement            │
  │                                                                     │
  │   ┌──────────┐        ┌──────────┐          ┌──────────┐           │
  │   │ R_d(x_0) │        │ R_t(t_0) │          │   @      │           │
  │   │ R_d(x_1) │  -->   │ R_t(t_1) │   -->    │   X  @   │           │
  │   │ R_d(x_2) │        │ R_t(t_2) │          │      X @ │           │
  │   │ R_d(x_3) │        │ R_t(t_3) │          │        X │           │
  │   └──────────┘        └──────────┘          └──────────┘           │
  │                                                                     │
  │   Input-dependent      Learnable             Creates quantum        │
  │   (frozen angles)      (optimized)           correlations           │
  └─────────────────────────────────────────────────────────────────────┘
```

**Why this order matters:**
1. **Data first** -- encodes classical features into qubit rotations
2. **Trainable second** -- shifts/scales the encoded information
3. **Entanglement third** -- creates multi-qubit correlations from the transformed data

---

## Trainable Parameters

### Parameter Shape

```
  Parameters are stored as a matrix:  theta[layer, qubit]

                   qubit_0   qubit_1   qubit_2   qubit_3
                 ┌─────────┬─────────┬─────────┬─────────┐
  Layer 0        │ theta_0 │ theta_1 │ theta_2 │ theta_3 │
                 ├─────────┼─────────┼─────────┼─────────┤
  Layer 1        │ theta_4 │ theta_5 │ theta_6 │ theta_7 │
                 └─────────┴─────────┴─────────┴─────────┘

  Total trainable parameters = n_layers x n_features
```

### Initialization Strategies

The initialization of trainable parameters significantly affects training
dynamics and convergence:

```
  Strategy      |  Distribution / Range         |  Use Case
  --------------+-------------------------------+-------------------------------
  xavier        |  N(0, sqrt(2/(n_in+n_out)))   |  General purpose (default)
  he            |  N(0, sqrt(2/n_in))           |  Deeper circuits
  zeros         |  All parameters = 0           |  Start as identity transform
  random        |  Uniform[-pi, pi]             |  Maximum initial exploration
  small_random  |  Uniform[-0.1, 0.1]           |  Near-identity, gentle start
```

```
  Initialization Landscape:

  random:             xavier:             zeros:
  .-*-..*.-.          ...-*-...           ......*......
  *..-..*-..          ..*...*..           .............
  .-.*..-*-.          ...*.*...           ......*......
  (scattered)         (moderate)          (all at origin)

  * = initial parameter positions on the loss landscape
```

**Recommendation:** Start with `xavier` (default). Use `small_random` when
you want the trainable layer to begin as a near-identity operation, letting
the data encoding dominate initially.

---

## Entanglement Topologies

Four connectivity patterns control how qubits become correlated:

### Linear (default)

```
  q0 ──@                       Pairs: (0,1), (1,2), (2,3)
       |                       Count: n - 1
  q1 ──X──@
          |
  q2 ─────X──@
             |
  q3 ────────X
```

### Circular

```
  q0 ──@──────────X            Pairs: (0,1), (1,2), (2,3), (3,0)
       |          |            Count: n
  q1 ──X──@       |
          |       |
  q2 ─────X──@    |
             |    |
  q3 ────────X──@─┘
```

### Full

```
  q0 ──@──@──@                 Pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
       |  |  |                 Count: n(n-1)/2
  q1 ──X  |  |──@──@
          |  |  |  |
  q2 ─────X  |──X  |──@
             |     |  |
  q3 ────────X─────X──X
```

### None (Separable)

```
  q0 ──                        Pairs: (none)
                               Count: 0
  q1 ──
                               No correlations between qubits.
  q2 ──                        Useful as a baseline or when
                               entanglement is not desired.
  q3 ──
```

### Topology Comparison

```
  Topology  | CNOT/layer | Connectivity     | Best For
  ----------+------------+------------------+----------------------------------
  linear    |   n - 1    | Nearest-neighbor | Superconducting (IBM, Google)
  circular  |   n        | Ring / wrap      | Ring topologies
  full      | n(n-1)/2   | All-to-all       | Ion traps (IonQ, Quantinuum)
  none      |   0        | None required    | Separable encodings / baselines
```

---

## Data + Trainable Interaction

The interplay between data rotations and trainable rotations on a single qubit
creates a combined transformation:

```
  |0> ── R_d(x) ── R_t(theta) ──

  Effective rotation on Bloch sphere:

              Z
              |       . R_t(theta) shifts the
              |     .    encoded point
              |   .
              | .    * final state
       -------+---------- Y
             /
            / . R_d(x) encodes the data
           /
          X

  When d = t = Y:   RY(theta) . RY(x)  =  RY(x + theta)

  The trainable parameter acts as a LEARNED BIAS, shifting
  each feature's encoding angle by an optimized amount.
```

When the data and trainable rotations use **different axes**, the combined
effect is richer than a simple additive shift:

```
  When d = Y, t = Z:   RZ(theta) . RY(x)

  This creates a rotation on the Bloch sphere that cannot
  be expressed as a single rotation -- the trainable layer
  adds a genuinely new degree of freedom.
```

---

## Fourier Perspective

The expressivity of the encoding is characterized by the Fourier frequencies
it can represent. With `L` layers and `n` qubits:

```
  Accessible Fourier spectrum:

  L=1  __|__|__|__|__      up to n frequencies per dimension
       -n         n

  L=2  _|_|_|_|_|_|_|_    up to 2n frequencies per dimension
       -2n        2n

  L=3  ||||||||||||||||    up to 3n frequencies per dimension
       -3n        3n

  More layers --> richer frequency spectrum --> more expressive functions
```

The trainable parameters control the **amplitudes** (coefficients) of these
Fourier components, while the data re-uploading determines the available
**frequencies**. This is the core insight from Schuld et al. (2021).

---

## Resource Scaling

For `n` qubits/features and `L` layers:

```
  Resource                |  Formula                 | Example (n=4, L=2, linear)
  ------------------------+--------------------------+---------------------------
  Qubits                  |  n                       |  4
  Trainable parameters    |  L * n                   |  8
  Data parameters         |  L * n                   |  8
  Single-qubit gates      |  2 * L * n               |  16
  Two-qubit gates (lin)   |  (n-1) * L               |  6
  Two-qubit gates (cir)   |  n * L                   |  8
  Two-qubit gates (full)  |  n(n-1)/2 * L            |  12
  Total gates (linear)    |  2*L*n + (n-1)*L         |  22
  Circuit depth           |  L * (2 + ent_depth)     |  8
```

---

## Key Properties

```
  Property                 |  Value / Behavior
  -------------------------+--------------------------------------------------
  Entangling?              |  Yes (when n > 1 and entanglement != "none")
  Simulability             |  Not classically simulable (with entanglement)
  Trainability estimate    |  ~0.85 - 0.03*L  (decreases with depth)
  Data re-uploading        |  Yes (features re-applied every layer)
  Trainable parameters     |  Yes (L * n learnable angles)
  Gradient support         |  Parameter-shift rule compatible
  Feature-to-qubit ratio   |  1:1 (one qubit per feature)
```

---

## What Makes This Different from Fixed Encodings

```
  FIXED ENCODING (e.g., Angle, Hardware-Efficient):

  |0> ── R(x_0) ────── R(x_0) ──────          Data controls everything.
  |0> ── R(x_1) ────── R(x_1) ──────          No adaptability.

  TRAINABLE ENCODING:

  |0> ── R(x_0) ── R(theta_0) ──── R(x_0) ── R(theta_2) ────
  |0> ── R(x_1) ── R(theta_1) ──── R(x_1) ── R(theta_3) ────
                      ^                          ^
                      |                          |
                    LEARNED                    LEARNED
                    parameters                 parameters

  The trainable parameters allow the encoding to:
    1. Amplify important features      (large |theta_i|)
    2. Suppress irrelevant features    (theta_i near 0)
    3. Create task-specific biases     (shift encoding angles)
    4. Absorb systematic noise         (compensate hardware errors)
```

---

## Training Loop Integration

Trainable encoding is designed to fit into a variational optimization loop:

```
  ┌─────────────┐     ┌───────────────┐     ┌─────────────┐
  │  Initialize  │     │   Forward      │     │  Measure     │
  │  theta       │────>│   Pass         │────>│  Expectation │
  │  (xavier)    │     │  |psi(x,theta)>│     │  <O>         │
  └─────────────┘     └───────────────┘     └──────┬──────┘
                                                    │
  ┌─────────────┐     ┌───────────────┐            │
  │  Update      │     │  Compute       │            │
  │  theta       │<────│  Gradients     │<───────────┘
  │  (optimizer) │     │  (param-shift) │
  └─────────────┘     └───────────────┘

  API:
    params = enc.get_trainable_parameters()     # shape (L, n)
    # ... run optimization step ...
    enc.set_trainable_parameters(new_params)    # update
    enc.reset_parameters(seed=42)               # restart training
```

---

## Practical Considerations

### Data Preprocessing

```
  Rotation gates are 2*pi-periodic:   R(x) = R(x + 2*pi)

  Recommended pipeline:
    raw features  -->  standardize (mean=0, std=1)  -->  scale to [0, pi]

  With trainable encoding, preprocessing is less critical because
  the trainable parameters can learn to compensate. However, proper
  scaling still helps convergence speed.
```

### Depth vs. Trainability Trade-off

```
  L=1     Shallow, highly trainable      -->  Limited expressivity
  L=2     Good balance (default)         -->  Recommended starting point
  L=3-4   More expressive               -->  Still trainable for most tasks
  L=5-7   High expressivity              -->  Monitor for vanishing gradients
  L=8+    Very deep                      -->  Barren plateau warning issued

  Trainability:   0.85 ─────────────────\
                                          \
                  0.70 ─────────────────────\
                                              \
                  0.40 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\── (floor)
                        |   |   |   |   |   |   |
                        1   3   5   7   9  11  13   layers
```

### Overfitting Risk

```
  Total trainable params = L * n

  If  (L * n)  >>  (training samples):
      High risk of overfitting.
      Mitigations:
        - Reduce L
        - Use regularization
        - Add more training data
        - Monitor validation loss

  Warning issued when total params > 100.
```

---

## Comparison with Related Encodings

```
  Encoding              |  Trainable?  |  Expressivity  |  Depth       |  Overhead
  ----------------------+--------------+----------------+--------------+-----------
  Trainable Encoding    |  YES         |  High          |  3*L layers  |  Training
  Hardware-Efficient    |  No          |  Moderate      |  2*reps      |  None
  Data Re-uploading     |  No          |  Universal*    |  Variable    |  None
  Angle Encoding        |  No          |  Low           |  1 layer     |  None
  IQP Encoding          |  No          |  High          |  O(n^2)      |  None

  * with sufficient layers

  Trainable encoding occupies a unique niche: it is more expressive
  than simple fixed encodings, yet avoids the full complexity of a
  general variational ansatz by keeping the structure constrained.
```

---

## Strengths and Limitations

### Strengths

- **Task adaptability** -- learns to emphasize features important for the problem
- **Noise absorption** -- trainable parameters partially compensate systematic errors
- **Transfer learning** -- pre-trained parameters can be reused across related tasks
- **Flexible structure** -- configurable rotation axes, entanglement, initialization
- **Gradient-friendly** -- supports parameter-shift rule for exact gradients
- **Multi-backend** -- works with PennyLane, Qiskit, and Cirq

### Limitations

- **Training overhead** -- requires classical optimization loop (more compute)
- **Barren plateaus** -- deep circuits (L > 8) may face vanishing gradients
- **Overfitting risk** -- too many parameters relative to data causes poor generalization
- **Initialization sensitivity** -- poor initial values can trap optimization in local minima
- **Not universal** -- less expressive than a full variational ansatz for the same depth

---

## References

1. Schuld, M., et al. (2021). "Effect of data encoding on the expressive
   power of variational quantum machine learning models." Physical Review A.

2. Benedetti, M., et al. (2019). "Parameterized quantum circuits as machine
   learning models." Quantum Science and Technology.

3. Pérez-Salinas, A., et al. (2020). "Data re-uploading for a universal
   quantum classifier." Quantum.

4. McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
   network training landscapes." Nature Communications.

5. Grant, E., et al. (2019). "Initialization strategy for addressing barren
   plateaus in parameterized quantum circuits." Quantum.