# Screening Encodings on Your Data

[`recommend_encoding`](which-encoding.md) answers from *metadata* — how many features you have, which task, which hardware. It never looks at your data.

But the benchmark's central positive result is about a quantity computed **on your data**: the centered kernel-target alignment tracks downstream kernel accuracy closely (Spearman ρ = 0.91, p ≈ 3×10⁻⁴⁸, positive and significant on all eight datasets). `screen_encodings` turns that result into the workflow it implies — score every candidate on your own `(X, y)`, train only the ones that score well.

---

## The 30-second version

```python
from encoding_atlas.benchmark import get_dataset
from encoding_atlas.guide import screen_encodings

X, y = get_dataset("moons", n_samples=200, seed=0)

result = screen_encodings(X, y, seed=0)

for candidate in result.top(3):
    print(f"{candidate.rank}. {candidate.name:22s} alignment={candidate.alignment:+.3f}")
```

```
1. trainable_encoding     alignment=+0.524
2. angle                  alignment=+0.442
3. data_reuploading       alignment=+0.430
```

No training happens. Each candidate needs one statevector per sample, so a full 16-encoding screen on 100 samples runs in about a second.

The built encodings come back ready to use — no need to rebuild them:

```python
from encoding_atlas.benchmark import evaluate_encoding

for candidate in result.top(3):
    report = evaluate_encoding(candidate.encoding, X, y, method="kernel")
    print(candidate.name, report["mean"])
```

---

## Read it as a shortlist, not an oracle

This is the most important thing to know about screening, and it is worth being blunt about.

On the benchmark's eight datasets:

| Strategy | Mean accuracy | Regret vs. oracle |
|:---|---:|---:|
| Oracle (best encoding per dataset) | 0.974 | — |
| **Alignment top-3 shortlist** | **0.973** | **0.001** |
| Alignment top-1 | 0.960 | 0.014 |
| Always use `angle` | 0.958 | 0.016 |
| Random pick | 0.815 | 0.159 |

**The single highest-aligned encoding is not meaningfully better than just defaulting to `angle`** (0.960 vs. 0.958). What screening buys you is the *shortlist*: taking the top three lands within 0.1 percentage points of the best achievable, while training three encodings instead of sixteen.

That is also exactly what the study prescribes — score, keep the ones that do well, train only those. `result.top()` defaults to 3 for this reason.

!!! warning "Where it can miss"
    On `moons`, the alignment shortlist tops out at 0.890 against an oracle of 0.945. Alignment is a strong predictor, not a proof. Screening narrows the search; it does not replace measuring.

The other reason to screen is **adaptivity**. The always-`angle` baseline looks strong partly because the benchmark's eight datasets are low-dimensional and mostly easy. When your data does not resemble them, the ranking moves — and that is precisely when a metadata-only recommendation is least reliable.

---

## What comes back

```python
result = screen_encodings(X, y, seed=0)

result.candidates      # every scored encoding, best alignment first
result.top(3)          # the shortlist
result.names(3)        # just the names
result.best()          # highest-aligned candidate
result.get("iqp")      # look one up by name
result.skipped         # what could not be built here, and why
```

Each `ScreenedEncoding` carries:

| Field | Meaning |
|:---|:---|
| `alignment` | Kernel-target alignment on **your** data, in [-1, 1]. The ranking key. |
| `encoding` | The built instance, ready to train with. |
| `rank`, `n_qubits`, `params` | Position, circuit width, and the benchmark-matched parameters used. |
| `atlas_rank`, `atlas_alignment` | What the benchmark measured, for reference. |
| `concentration_ratio`, `is_concentrated` | Kernel concentration on your data (opt-in, see below). |

A large gap between `alignment` and `atlas_alignment` is informative: it means your data behaves differently from the benchmark's, which is the situation screening exists for.

---

## Encodings that don't fit your data

Not every encoding builds at every width. SO(2) equivariance needs exactly two features; the swap-equivariant map needs an even count. Screening reports these rather than crashing:

```python
X_odd = np.random.default_rng(0).uniform(0, np.pi, (50, 3))
result = screen_encodings(X_odd, y, seed=0)

len(result.candidates)   # 13
list(result.skipped)     # ['symmetry_inspired', 'so2_equivariant', 'swap_equivariant']
result.skipped["so2_equivariant"]
# 'ValueError: SO2EquivariantFeatureMap requires n_features=2 ...'
```

---

## Options worth knowing

**Labels.** Alignment is a two-class quantity. Any two distinct values work — `{0, 1}`, `{1, 2}` and `{-1, +1}` all describe the same split and score identically. Multi-class and continuous targets are rejected with a clear error rather than silently scored; benchmark those directly with `evaluate_encoding`.

**Sub-sampling.** Alignment is O(n²) in kernel entries, so inputs above `max_samples` (default 200) are sub-sampled — stratified, so class balance is preserved, and seeded, so the screen is reproducible.

```python
result = screen_encodings(X, y, max_samples=100, seed=0)
result.n_samples_supplied, result.n_samples_used   # (5000, 100)
```

**Finite shots.** Screen on the kernel a real device would return:

```python
result = screen_encodings(X, y, shots=1000, seed=0)
```

Alignment aggregates over all n(n−1)/2 kernel entries, so independent shot noise largely averages out — which is why this screening rule stays usable under a realistic shot budget even though the kernel matrix itself does not. See [Kernel Concentration](../concepts/kernel-concentration.md#finite-shots).

**Restricting candidates.** Screen a subset, for instance after narrowing by hardware constraints:

```python
result = screen_encodings(X, y, candidates=["angle", "cyclic_equivariant", "qaoa"])
```

**Concentration annotation.** Opt in to see whether each kernel still carries usable geometry at this width:

```python
result = screen_encodings(X, y, include_concentration=True, seed=0)
```

It is computed from the same kernel, so it costs no extra simulation — and it deliberately **does not affect the ranking**. At a fixed circuit width, alignment already ranks the Haar-floor encodings last on its own, so a concentration veto would be redundant. Its value is telling you whether the ranking will survive at *wider* circuits.

---

## Screening vs. the atlas

The benchmark's own measured alignment ships as a queryable column, so you can compare your data against it:

```python
from encoding_atlas.atlas import rank_encodings

for p in rank_encodings(by="kernel_target_alignment", limit=5):
    print(p.name, round(p.metric("kernel_target_alignment"), 3))
```

| Encoding | Alignment | Kernel acc. | Expressibility | Benchmark rank |
|:---|---:|---:|---:|---:|
| `amplitude` | 0.521 | 0.910 | — | 5 |
| `angle` | 0.486 | 0.958 | 0.930 | 1 |
| `so2_equivariant` | 0.447 | 0.895 | 0.650 | 11 |
| `cyclic_equivariant` | 0.374 | 0.954 | 0.989 | 4 |
| `qaoa` | 0.344 | 0.917 | 0.981 | 6 |
| `swap_equivariant` | 0.317 | 0.835 | 0.860 | 2 |
| `hardware_efficient` | 0.290 | 0.863 | 0.946 | 9 |
| `data_reuploading` | 0.290 | 0.863 | 0.946 | 10 |
| `trainable` | 0.279 | 0.862 | 0.941 | 7 |
| `symmetry_inspired` | 0.265 | 0.849 | 0.914 | 8 |
| `higher_order_angle` | 0.186 | 0.767 | 0.934 | 3 |
| `basis` | 0.129 | 0.628 | — | 14 |
| `hamiltonian` | 0.097 | 0.730 | **0.999** | 15 |
| `pauli_feature_map` | 0.076 | 0.693 | **0.999** | 12 |
| `zz_feature_map` | 0.076 | 0.693 | **0.998** | 13 |
| `iqp` | 0.058 | 0.639 | **0.999** | 16 |

Read the alignment and expressibility columns side by side. Across these encodings alignment tracks kernel accuracy at ρ = +0.91, while expressibility runs the *other* way (ρ = −0.49 here, −0.68 in the study's per-dataset analysis). The four circuits scoring ≈ 0.999 on expressibility sit at the bottom on both alignment and accuracy — the mechanism is [kernel concentration](../concepts/kernel-concentration.md).

**Do not select an encoding by expressibility.** Screen by alignment instead.

---

## Which one should I use?

| Situation | Use |
|:---|:---|
| No data yet; picking a starting point | [`recommend_encoding`](which-encoding.md) |
| Data in hand, binary classification, kernel method | `screen_encodings`, then train the top 3 |
| Multi-class or regression | `evaluate_encoding` directly — alignment is undefined |
| Deciding whether results transfer to wider circuits | [Kernel concentration](../concepts/kernel-concentration.md) |

The two are complementary: recommend to start, screen once you have data.

---

## See Also

- [Which Encoding?](which-encoding.md) — the metadata-based recommendation
- [Kernel Concentration](../concepts/kernel-concentration.md) — why high expressibility hurts
- [API Reference](../api/index.md) — full signatures
