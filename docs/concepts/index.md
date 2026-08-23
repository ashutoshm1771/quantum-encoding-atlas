# Concepts

Understanding quantum data encoding requires bridging classical machine learning with quantum information theory. This section provides the conceptual foundations you need to use the Quantum Encoding Atlas effectively.

---

## What You'll Learn

<div class="grid cards" markdown>

-   **[What is Encoding?](what-is-encoding.md)**

    ---

    Why classical data must be transformed before a quantum computer can process it, and the different strategies for doing so.

-   **[Encoding Properties](encoding-properties.md)**

    ---

    The metrics that characterise an encoding: expressibility, entanglement capability, trainability, circuit depth, and simulability.

-   **[Statistical Comparison](statistical-comparison.md)**

    ---

    Why all-pairs testing has zero power across a handful of datasets, and the Friedman/Nemenyi procedure that replaces it — including what a missing result must *not* be imputed as.

-   **[Quantum Advantage](quantum-advantage.md)**

    ---

    When and why quantum encodings outperform classical feature maps, and what theoretical guarantees exist.

</div>

---

## The Big Picture

```
  Classical Data            Quantum Encoding            Quantum Processing
  ┌──────────────┐         ┌──────────────────┐        ┌──────────────────┐
  │              │         │                  │        │                  │
  │  x = [x₁,   │  ────►  │  |ψ(x)⟩ = U(x)  │  ────► │  Measurement,    │
  │       x₂,   │ Encode  │       |0⟩⊗ⁿ      │        │  Kernel, or      │
  │       ...]   │         │                  │        │  Variational     │
  │              │         │                  │        │  Optimization    │
  └──────────────┘         └──────────────────┘        └──────────────────┘

  The encoding step is the bridge. Its design determines what the
  quantum computer can learn from your data.
```

Choosing an encoding is one of the most consequential decisions in a quantum machine learning pipeline. A poor choice can make quantum processing no better than classical — or even worse. A good choice can unlock computational structures that no classical kernel can efficiently replicate.

The pages in this section give you the vocabulary and intuition to make that choice deliberately.
