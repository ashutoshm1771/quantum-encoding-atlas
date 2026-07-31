# Kernel Concentration

Every other property in this library describes an encoding at a **fixed** circuit width. Kernel concentration describes what happens as that width grows — and it is the property that decides whether a result measured at 2–8 qubits says anything about the same encoding at 20.

It is also the mechanism behind the atlas's headline negative result. Expressibility does not predict accuracy; concentration explains why not.

---

## The problem

A quantum-kernel method learns from the fidelity kernel

$$
K(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2, \qquad |\phi(x)\rangle = U(x)|0\rangle .
$$

What the model actually uses is the **spread** of the off-diagonal entries. If every distinct pair of inputs looks equally similar, the matrix is the identity up to sampling noise, and there is nothing left to fit — the model can memorise its training set and will generalise at chance.

*Exponential concentration* is the statement that this spread vanishes exponentially in the qubit count:

$$
\operatorname{Var}_{x, x'}\!\left[K(x, x')\right] \in \mathcal{O}(b^{-n}), \quad b > 1 .
$$

Two things go wrong at once. The geometry disappears, and separating two kernel entries starts to require a shot budget growing like $1/\operatorname{Var}$ — so the kernel becomes unaffordable on hardware well before it becomes uninformative in exact arithmetic.

---

## The measure

The reference point is the Haar-random ensemble. For two independent Haar states in dimension $d = 2^n$, the overlap follows $\mathrm{Beta}(1, d-1)$, so

$$
\mathbb{E}_{\text{Haar}}[K] = \frac{1}{d}, \qquad
\operatorname{Var}_{\text{Haar}}[K] = \frac{d - 1}{d^2 (d + 1)} \sim d^{-2} .
$$

That gives a dimensionless order parameter, comparable across qubit counts:

$$
\texttt{concentration\_ratio} = \frac{\operatorname{Var}[K]}{\operatorname{Var}_{\text{Haar}}[K]} .
$$

| Value | Meaning |
|:------|:--------|
| $\approx 1$ | At the Haar floor. The kernel is the identity up to noise; a kernel method cannot generalise from it, at any shot budget. |
| $\gg 1$ | Structure remains. The circuit width has not yet destroyed the geometry. |

```python
from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.analysis import compute_kernel_concentration

angle = compute_kernel_concentration(AngleEncoding(n_features=6), seed=0)
iqp = compute_kernel_concentration(IQPEncoding(n_features=6, reps=2), seed=0)

print(angle.concentration_ratio, angle.is_concentrated)   # 9.0   False
print(iqp.concentration_ratio, iqp.is_concentrated)       # 1.0   True
```

---

## Why the variance, and not the mean

This is the subtlety that makes the measure non-obvious, so it is worth stating plainly: **the mean cannot distinguish the two regimes.**

Take an encoding that puts each qubit into an independent, uniformly random single-qubit state — angle encoding with inputs drawn from $[0, 2\pi)$ is exactly this. Each qubit contributes an overlap $\cos^2\!\big((x - x')/2\big)$ with $\mathbb{E} = 1/2$ and $\mathbb{E}[\cdot^2] = 3/8$, so across $n$ independent qubits

$$
\mathbb{E}[K] = 2^{-n}, \qquad
\operatorname{Var}[K] = \left(\tfrac{3}{8}\right)^{n} - \left(\tfrac{1}{4}\right)^{n} .
$$

The mean is $2^{-n} = 1/d$ — **identical to Haar**. A mean-based measure would declare angle encoding maximally concentrated, which is exactly backwards. The variance, meanwhile, is exponentially *larger* than the Haar $\sim 4^{-n}$, and their ratio grows by a factor of $1.5$ per qubit:

| Qubits | Angle `mean_ratio` | Angle `concentration_ratio` |
|-------:|-------------------:|----------------------------:|
| 2 | 1.00 | 2.1 |
| 4 | 1.00 | 4.7 |
| 6 | 1.00 | 10.9 |
| 8 | 1.00 | 28.8 |

`mean_ratio` is reported on every result for completeness, but it is degenerate. Do not use it.

!!! note "Concentration depends on your data, not just your circuit"
    It is a joint property of the encoding **and** the input distribution. The library's data-free default draws inputs from $[0, 2\pi)$, matching expressibility and entanglement capability. Narrowing that range — which is what feature scaling does — generally reduces concentration. Pass your own `X` to measure the axis on your actual data.

---

## Scaling across widths

A single width tells you where you are; the sweep tells you where you are heading.

```python
from encoding_atlas.analysis import estimate_concentration_scaling

scaling = estimate_concentration_scaling(
    lambda d: IQPEncoding(n_features=d, reps=2),
    feature_counts=(2, 4, 6, 8),
    seed=0,
)

scaling.decay_rate              # 3.88 — variance falls per qubit (Haar rate is 4.0)
scaling.haar_normalized_slope   # ~0 — tracking the floor rather than pulling away
scaling.concentration_horizon() # 2 — already at the floor at the narrowest width
scaling.shots_per_entry_at(20)  # extrapolated hardware cost per kernel entry
```

| Field | Reading |
|:------|:--------|
| `decay_rate` | Per-qubit factor by which the variance shrinks. The Haar floor itself falls at **4.0**, so a rate near 4 means the encoding scrambles as fast as it possibly can. |
| `haar_normalized_slope` | Slope of $\log(\texttt{concentration\_ratio})$ vs. qubits. **Negative** = collapsing towards the floor. **Positive** = pulling away. This is the honest discriminator: a large `decay_rate` alone proves nothing, because the floor is falling too. |
| `concentration_horizon()` | Narrowest width from which the encoding *stays* at the floor. `None` means it never gets there over the measured range. A transient dip at the narrowest width does not count. |
| `r_squared` | Quality of the log-linear fit. Below ~0.9, do not trust any extrapolation from it. |

The sweep fits against **qubit count**, not feature count — concentration is a statement about Hilbert-space dimension. The distinction matters for amplitude encoding, where $n_{\text{qubits}} = \lceil \log_2 n_{\text{features}} \rceil$.

---

## The measured atlas

Every built-in encoding has been scanned and the results ship with the package. Sorted by the benchmark's overall rank:

| Encoding | Rank | Expressibility | Kernel acc. | Ratio @8q | Variance decay/qubit | Horizon | Shots/entry @8q |
|:---|---:|---:|---:|---:|---:|:---:|---:|
| `angle` | 1 | 0.930 | 0.958 | 28.8 | 2.38 | — | 37 |
| `swap_equivariant` | 2 | 0.860 | 0.835 | 230.0 | 1.74 | — | 26 |
| `higher_order_angle` | 3 | 0.934 | 0.767 | 29.8 | 2.37 | — | 37 |
| `cyclic_equivariant` | 4 | 0.989 | 0.954 | 12.0 | 2.58 | — | 127 |
| `amplitude` | 5 | — | 0.910 | 2.4 | 1.46 | — | 34 |
| `qaoa` | 6 | 0.981 | 0.917 | 15.5 | 2.48 | — | 99 |
| `trainable` | 7 | 0.941 | 0.862 | 26.0 | 2.55 | — | 69 |
| `symmetry_inspired` | 8 | 0.914 | 0.849 | 117.1 | 1.82 | — | 51 |
| `hardware_efficient` | 9 | 0.946 | 0.863 | 26.3 | 2.55 | — | 70 |
| `data_reuploading` | 10 | 0.946 | 0.863 | 26.3 | 2.55 | — | 70 |
| `so2_equivariant` | 11 | 0.650 | 0.895 | 9.9 | — | — | 8 |
| **`pauli_feature_map`** | 12 | **0.999** | 0.693 | **1.8** | **3.53** | **2** | 770 |
| **`zz_feature_map`** | 13 | **0.998** | 0.693 | **1.8** | **3.53** | **2** | 770 |
| `basis` | 14 | — | 0.628 | 13525.3 | 0.97 | — | 4 |
| **`hamiltonian`** | 15 | **0.999** | 0.730 | **1.9** | **3.48** | **2** | 745 |
| **`iqp`** | 16 | **0.999** | 0.639 | **1.1** | **3.86** | **2** | 980 |

Read the bold rows. The four encodings whose kernels reach the Haar floor are:

- the four with **expressibility $\approx 0.999$**, and
- four of the five **worst-ranked** encodings in the benchmark.

That is the mechanism. High expressibility *means* Haar-likeness; Haar-likeness *implies* a kernel variance at the $\sim d^{-2}$ floor; a kernel at the floor cannot generalise. The encodings that score best on expressibility are the ones whose kernels are least usable — which is why the atlas's pre-registered hypothesis H1 ("expressibility predicts accuracy") came back *refuted* with $\rho = -0.68$.

`basis` is the instructive exception: it ranks 14th but is nowhere near the floor. Its kernel is a 0/1 matrix with enormous spread — it fails for a different reason (it discretises away information), not through concentration. Concentration explains four of the five bottom-ranked encodings, not all of them.

```python
from encoding_atlas.atlas import concentrated_encodings, get_concentration_profile

sorted(p.name for p in concentrated_encodings())
# ['hamiltonian', 'iqp', 'pauli_feature_map', 'zz_feature_map']

get_concentration_profile("iqp").horizon        # 2
get_concentration_profile("angle").horizon      # None
```

!!! warning "Scope of the accuracy numbers"
    The benchmark's accuracy stages ran at `n_features` in `{2, 4}`, so the published ranking is measured entirely in the regime *before* concentration switches on. That is precisely why this axis exists: it tells you which parts of the ranking are expected to survive as circuits widen.

---

## Finite shots

Concentration is what makes shot budgets matter, so the two are measured together.

On hardware the standard construction is the **compute–uncompute** circuit $U(x_j)^\dagger U(x_i)|0\rangle$, whose probability of returning the all-zeros bitstring is exactly $K(x_i, x_j)$. The count of all-zeros outcomes over $M$ shots is therefore exactly $\mathrm{Binomial}(M, K)$ — so sampling that binomial from the exact kernel is statistically identical to running the circuits, at a fraction of the cost.

```python
import numpy as np
from encoding_atlas import AngleEncoding
from encoding_atlas.analysis import compute_fidelity_kernel, compute_kernel_target_alignment

enc = AngleEncoding(n_features=2)
X = np.random.default_rng(0).uniform(0, np.pi, (20, 2))
y = np.array([0] * 10 + [1] * 10)

K_exact = compute_fidelity_kernel(enc, X)                       # infinite shots
K_shots = compute_fidelity_kernel(enc, X, shots=1000, seed=0)   # what a device returns

# The same switch threads through every kernel-geometry diagnostic:
kta = compute_kernel_target_alignment(enc, X, y, shots=1000, seed=0)
```

The estimate is unbiased entrywise, symmetric, and has an exact unit diagonal. It is **not** positive semidefinite — independent noise on each entry routinely pushes the smallest eigenvalues negative. Project it before handing it to a precomputed-kernel estimator:

```python
from encoding_atlas.benchmark.kernel import ensure_psd

K_psd, was_clipped = ensure_psd(K_shots)
```

Two consequences worth knowing:

- **Kernel-target alignment is shot-robust.** It aggregates over all $n(n-1)/2$ entries, so independent noise largely averages out — which is why the atlas's KTA-based screening rule remains usable under a realistic shot budget even though the kernel matrix itself does not.
- **Geometric difference is not.** Shot noise is a direction no classical kernel reproduces, so it *inflates* $g$. A large value under finite shots is not by itself evidence of an advantage.

`shots_per_entry` reports the budget needed for the estimator's standard error to fall below half the measured spread, $4\bar{K}(1 - \bar{K}) / \operatorname{Var}$. It is a **noiseless lower bound**: gate noise, readout error, and compilation overhead are not modelled.

```python
result = compute_kernel_concentration(IQPEncoding(n_features=8, reps=2), seed=0)
result.shots_per_entry          # ~1,050 per entry at 8 qubits
result.shots_for_dataset(1000)  # ~5.2e8 across the 499,500 pairs of a 1000-sample kernel
```

---

## Using it

**Profiling any encoding**, including one you wrote yourself. The axis is data-free, so `profile_encoding` always computes it — pass `X=` to measure it on your own data instead:

```python
from encoding_atlas.analysis import profile_encoding

profile = profile_encoding(AngleEncoding(n_features=6))
profile.metrics["kernel_concentration_ratio"]
profile.metrics["kernel_is_concentrated"]
profile.metrics["kernel_shots_per_entry"]
```

**Getting warned automatically.** The decision guide raises a flag when it recommends an encoding the scan measured at the floor:

```python
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(n_features=4, feature_interactions="custom_pauli")
print(rec.encoding_name)   # pauli_feature_map
print(rec.scale_warning)   # "Kernel concentration: ... sits at the Haar floor ..."
```

**Practical rules of thumb.**

- A finite `concentration_horizon` at or below your feature count means a fidelity-kernel method is the wrong tool. Switch encodings, or use a variational (VQC) model, which does not depend on pairwise overlaps in the same way.
- A `decay_rate` above ~3 means the encoding is scrambling near the maximal rate; expect the shot cost to roughly double or worse per added qubit.
- Do not select an encoding on expressibility. High expressibility is a *predictor of concentration*, and concentration is a predictor of failure.

---

## Reproducing the scan

The bundled dataset is regenerated by a committed, deterministic script:

```bash
python -m experiments.concentration_scan            # rewrite package data
python -m experiments.concentration_scan --check    # verify, do not write
```

It uses the same circuit parameters as the benchmark's kernel stage (`experiments/configs/stage6b_kernel.json`), 200 random inputs per point (19,900 off-diagonal pairs), and a fixed seed. `concentration_metadata()` returns the full protocol.

---

## References

- Thanasilp, Wang, Cerezo & Holmes (2024). *Exponential concentration in quantum kernel methods.* Nature Communications 15:5200.
- Huang et al. (2021). *Power of data in quantum machine learning.* Nature Communications 12:2631.
- Sim, Johnson & Aspuru-Guzik (2019). *Expressibility and entangling capability of parameterized quantum circuits.* Advanced Quantum Technologies 2:1900070.
- Havlíček et al. (2019). *Supervised learning with quantum-enhanced feature spaces.* Nature 567:209.

---

## See Also

- [Encoding Properties](encoding-properties.md) — the fixed-width axes
- [Quantum Advantage](quantum-advantage.md) — when a quantum kernel can help at all
- [API Reference](../api/index.md) — full signatures
