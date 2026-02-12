# Benchmarking

This tutorial shows how to use the experiment framework to systematically benchmark quantum encodings on classification tasks, comparing them against each other and against classical baselines.

!!! abstract "What you'll learn"
    - Setting up a benchmarking experiment
    - Running VQC and quantum kernel classification
    - Comparing against classical baselines (SVM, Random Forest, etc.)
    - Interpreting benchmark results

---

## Overview

The Quantum Encoding Atlas includes a full experiment framework for benchmarking encodings:

```
  Encoding × Dataset × Metric  →  Structured Results
```

The framework handles cross-validation, seeding, checkpointing, and statistical analysis automatically.

---

## Quick Benchmark

```python
from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.analysis import count_resources

# Compare resource costs
for Enc in [AngleEncoding, IQPEncoding]:
    enc = Enc(n_features=4)
    res = count_resources(enc)
    print(f"{enc.__class__.__name__:<20s}  "
          f"qubits={res['n_qubits']}  depth={res['depth']}  "
          f"gates={res['gate_count']}")
```

---

## Full Experiment Framework

!!! info "Under Development"
    Detailed documentation for the full experiment framework (VQC classification, quantum kernel SVM, classical baselines, cross-validation, and statistical comparison) is being prepared alongside the Stage 6a/6b experiment results.

    **Planned content:**

    - Configuring experiments via JSON config files
    - Running the experiment runner with checkpointing
    - VQC classification benchmark (encoding + variational ansatz + optimiser)
    - Quantum kernel SVM benchmark (fidelity kernel + classical SVM)
    - Classical baselines for fair comparison
    - Statistical analysis of results (confidence intervals, hypothesis tests)
    - Generating comparison tables and plots

---

## Next Steps

- [Hardware Considerations](hardware-considerations.md) — adapting benchmarks for real hardware
- [Comparing Encodings](comparing-encodings.md) — property-level comparison (faster than full benchmarks)
