# What is Quantum Data Encoding?

Quantum computers operate on **quantum states** — objects that live in a Hilbert space and obey the laws of quantum mechanics. Classical data (numbers, images, sensor readings) does not. Before any quantum algorithm can process classical data, that data must be **encoded** into a quantum state.

This page explains what that means, why it matters, and how different encoding strategies produce fundamentally different quantum representations.

---

## The Problem

```
  Classical world                     Quantum world
  ─────────────────                   ─────────────────

  x = [0.3, 0.7, 0.1, 0.5]          |ψ⟩ ∈ ℂ^{2ⁿ}

  A vector of real numbers.           A unit vector in exponentially
  Lives in ℝⁿ.                        large Hilbert space.
  Processed by CPUs / GPUs.           Processed by quantum gates.

                         ?
                   ──────────────►
                   How do we get
                   from here to there?
```

A quantum computer cannot directly "read" a floating-point number. It can only manipulate quantum bits (qubits) through **unitary operations** (gates). The encoding defines a mapping:

\[
  x \in \mathbb{R}^n \;\longmapsto\; |\psi(x)\rangle = U(x)\,|0\rangle^{\otimes m}
\]

where \( U(x) \) is a **data-dependent quantum circuit** — a sequence of gates whose angles, structure, or both depend on the input \( x \).

---

## Why It Matters

The encoding determines the **geometry of the feature space** that the quantum model works in. Two different encodings map the same classical data into completely different quantum states, leading to different:

- **Decision boundaries** in classification
- **Kernel functions** in quantum kernel methods
- **Gradient landscapes** in variational training
- **Expressibility** (which functions the model can represent)
- **Trainability** (how easy it is to optimise)

!!! tip "The encoding is not just plumbing"
    In classical ML, the "input layer" is often trivial — you feed numbers into a neural network. In quantum ML, the encoding **is** the feature map. It is the single most important design choice.

---

## Encoding Strategies

There are several fundamentally different approaches to encoding classical data:

### Basis Encoding

Map discrete values to computational basis states.

```
  x = [1, 0, 1, 1]  ──►  |1011⟩
```

Simple and efficient, but only works for binary or discrete data.

### Angle Encoding

Use feature values as rotation angles for single-qubit gates.

```
  x = [θ₁, θ₂, θ₃]  ──►  RY(θ₁)|0⟩ ⊗ RY(θ₂)|0⟩ ⊗ RY(θ₃)|0⟩
```

One qubit per feature, no entanglement, classically simulable — but a natural starting point.

### Amplitude Encoding

Store features as the amplitudes of a quantum state.

```
  x = [x₁, ..., xₙ]  ──►  |ψ⟩ = Σᵢ xᵢ|i⟩    (after normalisation)
```

Exponential compression (log n qubits for n features), but exponential circuit depth.

### Entangling Encodings

Interleave data-dependent gates with entangling operations.

```
  |0⟩⊗ⁿ  ──►  H⊗ⁿ  ──►  RZ(xᵢ)  ──►  ZZ(xᵢxⱼ)  ──►  |ψ(x)⟩
```

Creates non-separable quantum states whose properties are provably hard to compute classically (IQP, ZZ Feature Map).

### Equivariant Encodings

Build symmetry constraints into the circuit so the quantum state transforms predictably under group actions.

```
  If the data has rotational symmetry:
    Rotate input  ──►  Quantum state rotates correspondingly
```

Reduces the effective hypothesis space and improves generalisation on symmetric problems.

---

## The Encoding Landscape

```
         Expressibility ──►
    Low                          High
     │                            │
     │  Basis         Angle       │   IQP      Amplitude
     │  ───────       ─────       │   ───      ─────────
     │  Discrete      Product     │   Entangled Maximal
     │  states        states      │   phases    Hilbert
     │                            │             space
     │                            │
     │  Easy to simulate          │   Hard to simulate
     │  Easy to train             │   Risk of barren plateaus
     │  Limited power             │   Potential quantum advantage
     │                            │
```

Moving right increases what the encoding can represent, but also increases circuit complexity, noise sensitivity, and the risk of training difficulties. The art of quantum ML is finding the right point on this spectrum for your problem.

---

## What's Next

- [Encoding Properties](encoding-properties.md) — how to quantify and compare encodings
- [Quantum Advantage](quantum-advantage.md) — when quantum encodings provably outperform classical methods
- [Encodings Reference](../encodings/index.md) — detailed documentation for all 16 encodings
