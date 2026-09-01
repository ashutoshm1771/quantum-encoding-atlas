# Missing Metrics

Ranking encodings means combining metrics that are not available for every encoding. What you substitute for an absent one is not a technicality — on this project's own data, one substitution created a headline result that does not otherwise exist.

---

## Two absences that look identical and are not

Both arrive as `None`. They demand opposite treatment.

**Structurally zero.** The quantity is defined, known, and equal to zero, so nothing was measured. `AngleEncoding` applies only single-qubit rotations; its entanglement capability is exactly 0. The pipeline never ran it through the entanglement stage because there was nothing to run: `angle`, `higher_order_angle` and `basis` appear in **zero** of that stage's 122 records, and all three are the atlas's own `Non-Entangling` family.

**Not measured.** No number is defensible. `BasisEncoding` prepares computational basis states, so its fidelity distribution is degenerate and expressibility was never attempted at all.

Filling both with a column statistic is wrong for both, and wrong in different directions.

---

## What the imputation cost

The atlas's Monte Carlo weight sweep filled the missing entanglement capability with the **median of the encodings that do entangle** — 0.605, against a true value of 0.

| | published | corrected |
|:---|---:|---:|
| `angle` mean rank | 1.71 | **3.26** |
| `angle` in top 3 | 97.8% | **80.8%** |
| `higher_order_angle` in top 3 | 75.0% | **50.9%** |
| encodings above the 90% bar | `['angle']` | **`[]`** |

The published claim was that angle encoding is *robustly* top-3 across 1000 random weightings. Corrected, nothing clears the bar — the ranking does not survive reweighting at all. That conclusion existed only because a non-entangling circuit was credited with the median entangling capability of circuits that do entangle.

---

## Saying which you mean

```python
from encoding_atlas.benchmark import (
    MetricSpec, measured, structurally_zero, not_measured, score_encodings,
)

readings = {
    "angle": {
        "accuracy": measured(0.85),
        "entanglement": structurally_zero("no entangling gates"),
    },
    "basis": {
        "accuracy": measured(0.61),
        "entanglement": structurally_zero("no entangling gates"),
        "expressibility": not_measured("degenerate fidelity distribution"),
    },
}
```

A structural zero resolves to `0.0` and normalises alongside every measured value — it is *data*, and it widens the range its rivals are scaled against. A `not_measured` reading contributes nothing and is reported.

!!! warning "A true zero is not a missing value"
    Dropping the structural zero is as wrong as imputing it. With `angle` at 0 the entanglement axis spans `[0, 1]`; without it the floor rises and every other encoding's normalised entanglement shifts.

---

## When a metric genuinely cannot be had

The encoding is scored on the weight that remains, renormalised — which is defensible only if it is *visible*:

```python
result = score_encodings(readings, specs)
result.effective_weight   # {'angle': 1.0, 'basis': 0.7, ...}
print(result.summary())
```

```
16 encodings scored on 6 metrics
  best: angle (0.733)
  scored on a reduced objective: amplitude (85% of the weight), basis (70% of the weight)
  known to be zero, not missing: angle: ['entanglement']; basis: ['entanglement']; higher_order_angle: ['entanglement']
```

An encoding at 0.70 was ranked against a **different objective** than one at 1.0. That is sometimes acceptable and never acceptable silently.

The failure mode is sharp: an encoding measured only on the axis it happens to win is scored 1.0 and ranked first, on half the objective. Pass `on_unusable="exclude"` to drop such encodings instead of ranking them against a shorter yardstick.

---

## Reproducibility of the weight sweep

[`weight_sensitivity`][ws] draws random weight vectors with `numpy.random.RandomState`, not `numpy.random.default_rng`. NumPy guarantees the legacy stream across releases and explicitly does not guarantee `Generator`.

This is not hypothetical here. Re-running the atlas's sweep from unchanged code, byte-identical inputs and the same `seed=42` reproduced the deterministic equal-weight ranking exactly while returning **97.2%** where **97.8%** was published — the RNG stream had moved underneath a published number.

```python
from encoding_atlas.benchmark import weight_sensitivity

result = weight_sensitivity(readings, specs, n_samples=1000, seed=42)
print(result.summary())
```

```
16 encodings over 1000 random weight vectors (seed 42)
  angle: mean rank 3.26, top-3 in 80.8% of draws
  swap_equivariant: mean rank 5.68, top-3 in 39.9% of draws
  qaoa_encoding: mean rank 5.96, top-3 in 23.9% of draws
  no encoding reaches the top 3 in 90% of draws: the ranking does not survive reweighting
  scored on a reduced objective: ['amplitude', 'basis']
```

An empty `robust` is a result, not a failure. It says the ordering depends on the weights you chose.

---

## API

| Name | Purpose |
|:---|:---|
| `measured` / `structurally_zero` / `not_measured` | Build a reading that states why it is absent |
| `readings_from_values` | Convert a plain value table plus a structural-zero register |
| [`score_encodings`][se] | Weighted, min-max normalised ranking |
| [`weight_sensitivity`][ws] | Reproducible Monte Carlo sweep over weightings |

---

## References

The distinction between a structural zero and an unobserved value is standard in missing-data theory; imputing the latter's estimator into the former is a misspecification, not a robustness measure. See Rubin, D. B. (1976), *Inference and Missing Data*, Biometrika 63(3), 581–592, for the classification this rests on.

[se]: ../api/index.md
[ws]: ../api/index.md
