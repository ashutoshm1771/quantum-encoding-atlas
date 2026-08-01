# Feature Scaling

Encodings turn numbers into rotation angles. The numeric range you scale your features into therefore decides how much of each qubit's Bloch circle your data sweeps — and that changes the geometry of the kernel your model actually sees.

This is not a tuning detail. For several encodings the scaling range moves accuracy **further than the choice of encoding does**, and it is the one axis of the benchmark that was held fixed rather than measured.

---

## The mechanism

Rotation gates have period $2\pi$. Scaling into $[0, 2\pi]$ drives every feature over a *full* period, which is exactly the regime [kernel concentration](kernel-concentration.md) identifies as maximally Haar-like: pairwise fidelities collapse towards the $1/2^n$ floor and the kernel approaches the identity.

Angle encoding makes this exact. Under inputs uniform on $[0, 2\pi]$, each qubit contributes an overlap $\cos^2((x - x')/2)$ with mean $1/2$, so across $n$ qubits

$$
\mathbb{E}[K] = 2^{-n} = \frac{1}{d},
$$

precisely the Haar mean. Narrower ranges keep the encoded states in a cone, preserving the spread a kernel method needs.

```python
from encoding_atlas import IQPEncoding
from encoding_atlas.analysis import scan_feature_ranges
from encoding_atlas.benchmark import get_dataset

X, y = get_dataset("moons", n_samples=120, seed=0)
scan = scan_feature_ranges(IQPEncoding(n_features=2, reps=2), X, y, seed=0)

for r in scan.results:
    print(f"[0, {r.high:.3f}]  align={r.alignment:+.3f}  mean={r.offdiagonal_mean:.3f}")
```

```
[0, 0.785]  align=+0.517  mean=0.746
[0, 1.571]  align=+0.352  mean=0.423
[0, 3.142]  align=+0.167  mean=0.290
[0, 6.283]  align=+0.036  mean=0.315     <- the pipeline's default
```

Alignment falls monotonically as the range widens, and the mean overlap descends towards the Haar value (0.25 at two qubits).

!!! warning "Read the mean alongside `concentration_ratio`"
    `concentration_ratio` is a *variance* relative to Haar, and it is **not** monotone in range width. A very narrow range leaves every kernel entry close to 1 — degenerate in its own way, but with tiny variance and so a *low* ratio. Only a ratio near 1 **together with** a mean near $1/2^n$ means the Haar floor.

---

## What it costs

The bundled scan measures mean quantum-kernel accuracy for every encoding across four ranges, on the library's benchmark datasets:

| Encoding | [0, π/4] | [0, π/2] | [0, π] | [0, 2π] | Spread | Best |
|:---|---:|---:|---:|---:|---:|:---|
| `iqp` | 0.875 | 0.903 | 0.792 | **0.560** | **0.343** | [0, π/4] |
| `hamiltonian` | 0.881 | 0.935 | 0.867 | **0.667** | **0.268** | [0, π/2] |
| `pauli_feature_map` | 0.839 | 0.886 | 0.823 | 0.672 | 0.214 | [0, π/2] |
| `zz_feature_map` | 0.839 | 0.886 | 0.823 | 0.672 | 0.214 | [0, π/2] |
| `basis` | 0.678 | 0.755 | 0.658 | 0.578 | 0.177 | [0, π/2] |
| `symmetry_inspired` | 0.734 | 0.877 | 0.860 | 0.809 | 0.144 | [0, π/2] |
| `cyclic_equivariant` | 0.813 | 0.885 | 0.946 | 0.889 | 0.133 | [0, π] |
| `angle` | 0.803 | 0.828 | 0.900 | **0.931** | 0.128 | [0, π] |
| `data_reuploading` | 0.826 | 0.895 | 0.920 | 0.796 | 0.124 | [0, π/2] |
| `hardware_efficient` | 0.826 | 0.895 | 0.920 | 0.796 | 0.124 | [0, π/2] |
| `higher_order_angle` | 0.789 | 0.846 | 0.867 | 0.751 | 0.116 | [0, π/4] |
| `trainable` | 0.800 | 0.876 | 0.915 | 0.804 | 0.115 | [0, π/2] |
| `swap_equivariant` | 0.791 | 0.745 | 0.683 | 0.781 | 0.108 | [0, π/4] |
| `qaoa` | 0.826 | 0.900 | 0.909 | 0.852 | 0.083 | [0, π/2] |
| `so2_equivariant` | 0.842 | 0.883 | 0.895 | 0.853 | 0.053 | [0, π] |
| `amplitude` | 0.750 | 0.750 | 0.750 | 0.750 | 0.000 | [0, π/4] |

IQP swings by **34 accuracy points** across the four ranges. Note the asymmetry: the entangling maps lose the most at $[0, 2\pi]$, while `angle` is the one encoding that *prefers* the wide range. `amplitude` is flat, because it normalises its input to a state vector and discards the scale entirely.

```python
from encoding_atlas.atlas import scaling_sensitive_encodings

[(p.name, round(p.accuracy_spread, 3)) for p in scaling_sensitive_encodings()][:4]
# [('iqp', 0.343), ('hamiltonian', 0.268), ('pauli_feature_map', 0.214), ('zz_feature_map', 0.214)]
```

---

## The consequence for hypothesis H1

The empirical pipeline scales every dataset into $[0, 2\pi]$ at load time, and the study's headline negative result — that expressibility is *anti*-correlated with accuracy — is measured under that choice. Since $[0, 2\pi]$ is the range that pushes circuits hardest onto the concentration floor, and the most expressible circuits are pushed furthest, the natural question is whether the correlation is a property of expressibility or of the preprocessing.

The scan answers it by recomputing the correlation at each range, with expressibility re-measured in the same regime so both axes describe one setting:

| Range | ρ (atlas subset, n=14) | p | ρ (all 16) | p |
|:---|---:|---:|---:|---:|
| [0, π/4] | **+0.580** | 0.030 | +0.339 | 0.200 |
| [0, π/2] | **+0.784** | **0.0009** | +0.637 | 0.008 |
| [0, π] | −0.289 | 0.316 | −0.094 | 0.728 |
| **[0, 2π]** ← pipeline default | **−0.541** | **0.046** | −0.286 | 0.283 |

**The sign reverses.** At the pipeline's own range the association is negative and significant, reproducing the direction the study reports. At $[0, \pi/2]$ it is positive and more significant.

```python
from encoding_atlas.atlas import expressibility_accuracy_correlation

for row in expressibility_accuracy_correlation():
    print(row["high"], row["spearman_rho_atlas_subset"], row["is_published_range"])
```

The "atlas subset" column restricts to the 14 encodings the atlas records an expressibility for — `basis` and `amplitude` are state-preparation circuits whose expressibility the pipeline stores as null. That is the set the published analysis used, so it is the only column comparable with the published number.

!!! note "What this does and does not say"
    It does **not** say the published result is wrong. $[0, 2\pi]$ is a defensible convention — it is the full rotation period — and the study's claim is scoped to its own protocol.

    It says the sign of the expressibility–accuracy association is **set by a preprocessing choice**, so it is not a scaling-invariant property of an encoding. The range belongs in any report of that correlation.

    Caveats on the scan itself: it uses the library's five binary datasets at two features with single-run 5-fold CV, whereas the study used eight datasets across {2, 4} features with 10 runs × 5 folds and multiple-comparison correction. It is a sensitivity measurement, not a re-run of the study.

---

## Choosing a range

```python
from encoding_atlas.analysis import recommend_feature_range, scale_to_range

low, high = recommend_feature_range(IQPEncoding(n_features=2, reps=2), X, y, seed=0)
X_scaled = scale_to_range(X, low, high)
```

`recommend_feature_range` picks the range with the highest kernel-target alignment — training-free, and alignment is the quantity that [predicts accuracy](../guide/screening.md). Ties break towards the narrower range, the more conservative side.

To search encodings and ranges together, pass `feature_ranges=` to the screener:

```python
from encoding_atlas.analysis import DEFAULT_FEATURE_RANGES
from encoding_atlas.guide import screen_encodings

result = screen_encodings(X, y, seed=0, feature_ranges=DEFAULT_FEATURE_RANGES)
for c in result.top(3):
    print(c.name, c.feature_range, round(c.alignment, 3))
```

Candidates are then ranked over (encoding, range) pairs, so the same encoding can appear more than once.

---

## Scaling inside cross-validation

`scale_to_range` accepts a `reference` array, so the scaler can be fitted on a training split and applied to a test split without the test fold's extremes setting its own scale:

```python
X_train_scaled = scale_to_range(X_train, 0, np.pi)
X_test_scaled = scale_to_range(X_test, 0, np.pi, reference=X_train)
```

Transformed test values may then fall outside the target range, which is correct and matches `sklearn.preprocessing.MinMaxScaler`. The library's own benchmark does this per fold; fitting on the full dataset before splitting is a mild but real leak.

---

## Practical rules of thumb

- **Report the range you used.** It is a result-affecting choice, not a formatting detail.
- **Do not default to $[0, 2\pi]$ for entangling feature maps.** It is the worst of the four measured ranges for every one of them.
- **`angle` is the exception** — it is the only encoding that prefers the full period, which is worth knowing given it tops the benchmark ranking.
- **Screen the range with your data**, the same way you screen the encoding. It costs no training.

---

## Reproducing the scan

```bash
python -m experiments.scaling_scan            # rewrite package data
python -m experiments.scaling_scan --check    # verify, do not write
```

Deterministic and seeded; `scaling_metadata()` returns the full protocol.

---

## See Also

- [Kernel Concentration](kernel-concentration.md) — the mechanism behind the loss
- [Screening on Your Data](../guide/screening.md) — search encodings and ranges together
- [API Reference](../api/index.md) — full signatures
