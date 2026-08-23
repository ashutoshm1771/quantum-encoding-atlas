# Comparing Encodings Statistically

Once you have accuracies for sixteen encodings across eight datasets, the question is no longer "which number is biggest" but "which of these differences would survive being asked again". That is a multiple-comparison problem, and the wrong procedure does not merely lose power — it can be arithmetically incapable of returning any answer at all.

---

## Why the obvious approach cannot work

The natural move is a paired test on every pair, corrected for multiplicity: `wilcoxon_test` plus [`holm_bonferroni`][holm]. At atlas scale this is guaranteed to find nothing.

A paired Wilcoxon signed-rank test over $N$ datasets is a test on $N$ signs. Its smallest attainable two-sided $p$-value is

$$
p_{\min} = \frac{2}{2^{N}},
$$

reached only when one method wins on *every* dataset. Comparing $k$ methods generates $m = k(k-1)/2$ comparisons, and Holm's strictest threshold is $\alpha/m$. For the atlas's $k = 15$, $N = 8$:

$$
p_{\min} = \frac{2}{2^{8}} = 0.0078
\qquad\text{vs}\qquad
\frac{\alpha}{m} = \frac{0.05}{105} = 0.00048 .
$$

Since $0.0078 > 0.00048$, **no pair can reach significance regardless of the data**. Run on the atlas's kernel results, it returns 0 significant pairs out of 105, with the smallest raw $p$ sitting exactly on the $0.0078$ floor. The test is not being conservative; it has no power at all.

---

## The procedure that does work

Demšar (2006) replaces per-pair testing with rank-based testing across datasets.

**1. Rank within each dataset.** Rank 1 goes to the best method on that dataset. Ranking is what makes an easy dataset and a hard one commensurable — a 3-point accuracy gap means something different on `iris` than on `digits_01`, but "came second" means the same thing on both.

**2. Omnibus.** The Friedman test asks whether the average ranks differ more than chance allows. Its $\chi^2$ form is conservative, so the Iman–Davenport $F$ is reported alongside and is what `rejected` uses.

**3. Post hoc — only if the omnibus rejects.** Two methods differ when their average ranks differ by at least the critical difference

$$
\mathrm{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6N}} .
$$

```python
from encoding_atlas.benchmark import compare_over_datasets

# {encoding: {dataset: accuracy}}
result = compare_over_datasets(scores)
print(result.summary())
```

```
15 methods over 8 datasets
  Friedman chi2(14) = 79.093, p = 4.17e-11
  Iman-Davenport F(14, 98) = 16.825, p = 3.26e-20
  omnibus rejects at alpha = 0.05; nemenyi CD = 7.583 over a rank range of 11.19
  best: angle (rank 2.81)
  separated from best: 5/14 -> ['basis', 'iqp', 'pauli_feature_map', 'zz_feature_map', 'hamiltonian_encoding']
  excluded (incomplete): so2_equivariant lacks ['breast_cancer', 'digits_01', 'iris', 'wine']
```

!!! warning "A post-hoc test without a rejecting omnibus is invalid"
    Nemenyi is a *post-hoc* test. Applying it when Friedman has not rejected inflates the family-wise error rate, and it is the most common misuse of this procedure. `compare_over_datasets` therefore returns `posthoc=None` rather than a table you might quote:

    ```python
    if result.posthoc is None:
        print("no difference may be claimed")
    ```

---

## Missing cells are not zeros

Friedman needs a **complete block design**: every method scored on every dataset. Real suites violate this. `SO2EquivariantFeatureMap` accepts only two features, so it has no result on the atlas's four 4-feature datasets.

Imputing the gap — with zero, or with the column mean — is not a neutral default. It assigns worst rank for being *inapplicable* rather than for performing badly:

| | rank if missing → 0.0 | rank over datasets it ran on |
|:---|---:|---:|
| SO(2), VQC | 10.50 | **5.00** |
| SO(2), kernel-SVM | 10.19 | **4.38** |

The encoding moves from the bottom half to the top third, and every *other* encoding's rank is unchanged — the distortion falls entirely on the method with the missing cells.

So nothing is imputed. Incomplete methods are excluded and **named**, so the exclusion is reported rather than silent:

```python
result.excluded
# {'so2_equivariant': ('breast_cancer', 'digits_01', 'iris', 'wine')}
```

Their descriptive ranks are still available, with a caveat that matters:

```python
from encoding_atlas.benchmark import average_ranks

average_ranks(scores, missing="available")["so2_equivariant"]  # 4.38
```

!!! danger "Descriptive ranks are not on a common scale"
    A rank earned against 15 competitors is not comparable to one earned against 11. Use `missing="available"` for reporting where an encoding stood on the problems it can actually address — never as input to a critical difference computed for the full $k$.

---

## Read the critical difference before the diagram

`CD` scales as $\sqrt{k(k+1)/6N}$: it grows with the number of methods and shrinks only with the square root of the number of *datasets*. Adding encodings costs resolution; the only way to buy it back is more datasets.

At the atlas's $k = 15$, $N = 8$, `CD = 7.58` against a total rank range of `11.19`. Two thirds of the scale is consumed by the critical difference, and only 5 of 14 encodings separate from the best. That is the honest headline, and it is a statement about the size of the suite as much as about the encodings.

```python
from encoding_atlas.benchmark import critical_difference

critical_difference(15, 8)    # 7.583
critical_difference(15, 32)   # 3.792  — four times the datasets, half the CD
```

---

## Comparing everything against one control

If the question is "does anything beat angle encoding?" rather than "which pairs differ?", Bonferroni–Dunn makes $k-1$ comparisons instead of $k(k-1)/2$, and is correspondingly more powerful:

```python
result = compare_over_datasets(scores, test="bonferroni-dunn", control="angle")
```

The control must be chosen **before** looking at the ranks. Picking the winner and then testing everything against it is a different, uncorrected procedure.

---

## Critical values

`q_alpha` is computed from the studentised range distribution rather than read from a lookup table, so any $k$ and any $\alpha$ work. It reproduces Demšar's Table 5 to within that table's own rounding — a maximum deviation of 0.0007 across $k = 2 \ldots 20$ — and the published table is used as a test oracle rather than as the implementation.

---

## API

| Function | Purpose |
|:---|:---|
| [`compare_over_datasets`][cmp] | The full procedure: rank, omnibus, post-hoc if licensed |
| `friedman_test` | Omnibus only, with both $\chi^2$ and Iman–Davenport forms |
| `iman_davenport` | The $\chi^2 \to F$ correction, on its own |
| `average_ranks` | Average ranks under a stated missing-data policy |
| `critical_difference` | `CD` for Nemenyi or Bonferroni–Dunn |
| `complete_cases` | Split a score table into its complete design and the leftovers |

---

## References

Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1–30.

Iman, R. L., & Davenport, J. M. (1980). Approximations of the critical region of the Friedman statistic. *Communications in Statistics*, 9(6), 571–595.

[holm]: ../api/index.md
[cmp]: ../api/index.md
