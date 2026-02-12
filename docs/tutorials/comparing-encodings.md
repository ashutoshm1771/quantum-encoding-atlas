# Comparing Encodings

This tutorial shows how to use the analysis module to quantitatively compare encodings across expressibility, entanglement capability, trainability, and resource cost.

!!! abstract "What you'll learn"
    - Running resource analysis across multiple encodings
    - Computing expressibility and entanglement metrics
    - Interpreting the results to make informed encoding choices

---

## Setup

```python
from encoding_atlas import (
    AngleEncoding,
    IQPEncoding,
    AmplitudeEncoding,
    DataReuploading,
    ZZFeatureMap,
)
from encoding_atlas.analysis import count_resources

encodings = {
    'Angle':           AngleEncoding(n_features=4),
    'IQP':             IQPEncoding(n_features=4, reps=2),
    'Amplitude':       AmplitudeEncoding(n_features=4),
    'Data Reuploading': DataReuploading(n_features=4, n_layers=2),
    'ZZ Feature Map':  ZZFeatureMap(n_features=4, reps=2),
}
```

---

## Resource Comparison

```python
print(f"{'Encoding':<20s} {'Qubits':>6s} {'Depth':>6s} {'Gates':>6s} {'CNOTs':>6s}")
print("-" * 50)

for name, enc in encodings.items():
    resources = count_resources(enc)
    print(f"{name:<20s} {resources['n_qubits']:>6d} {resources['depth']:>6d} "
          f"{resources['gate_count']:>6d} {resources['cnot_count']:>6d}")
```

---

## Expressibility Analysis

Expressibility measures how well an encoding covers the Hilbert space. Lower KL divergence means higher expressibility.

```python
from encoding_atlas.analysis import compute_expressibility

for name, enc in encodings.items():
    expr = compute_expressibility(enc, n_samples=1000)
    print(f"{name:<20s}  expressibility = {expr:.4f}")
```

!!! note
    Expressibility computation requires sampling random inputs and simulating circuits. This can take a few minutes for large encodings. Use `n_samples` to control the trade-off between accuracy and speed.

---

## Entanglement Capability

```python
from encoding_atlas.analysis import compute_entanglement_capability

for name, enc in encodings.items():
    ent = compute_entanglement_capability(enc, n_samples=500)
    print(f"{name:<20s}  entanglement = {ent:.4f}")
```

---

## Interpreting the Results

When comparing encodings, consider the full picture:

| If you need... | Prioritise... |
|:----------------|:-------------|
| Maximum accuracy on hard problems | High expressibility + high entanglement |
| Reliable training | High trainability (low depth, limited entanglement) |
| NISQ hardware compatibility | Low depth + low CNOT count |
| Provable quantum advantage | Non-simulable encodings (IQP, ZZ) |

No single encoding dominates on all axes. The [Decision Guide](../guide/index.md) helps you navigate these tradeoffs systematically.

---

## Next Steps

- [Custom Encodings](custom-encoding.md) — build your own encoding
- [Benchmarking](benchmarking.md) — run full classification benchmarks
