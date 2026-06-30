"""Tests for benchmark statistical utilities (deterministic, no quantum sim)."""

from __future__ import annotations

import pytest

from encoding_atlas.benchmark.statistical import (
    cliffs_delta,
    compare_encodings,
    compare_encodings_corrected,
    holm_bonferroni,
    wilcoxon_test,
)


class TestWilcoxon:
    def test_identical_scores_return_unity_pvalue(self) -> None:
        # All-zero differences must not raise; p-value should be 1.0.
        stat, p = wilcoxon_test([0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
        assert stat == 0.0
        assert p == 1.0

    def test_clear_difference_is_significant(self) -> None:
        # Six paired samples, all a > b -> minimum two-sided p = 2/2**6 < 0.05.
        _stat, p = wilcoxon_test(
            [0.90, 0.92, 0.88, 0.95, 0.91, 0.93],
            [0.50, 0.52, 0.48, 0.55, 0.51, 0.53],
        )
        assert p < 0.05

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            wilcoxon_test([0.1, 0.2], [0.1])


class TestCliffsDelta:
    def test_full_dominance_is_plus_one(self) -> None:
        delta, magnitude = cliffs_delta([0.9, 0.8], [0.5, 0.6])
        assert delta == 1.0
        assert magnitude == "large"

    def test_full_reverse_dominance_is_minus_one(self) -> None:
        delta, _ = cliffs_delta([0.5, 0.6], [0.9, 0.8])
        assert delta == -1.0

    def test_identical_is_negligible(self) -> None:
        delta, magnitude = cliffs_delta([0.7, 0.7], [0.7, 0.7])
        assert delta == 0.0
        assert magnitude == "negligible"

    def test_empty_inputs_safe(self) -> None:
        assert cliffs_delta([], [0.1]) == (0.0, "negligible")

    def test_delta_in_range(self) -> None:
        delta, _ = cliffs_delta([0.6, 0.7, 0.5], [0.55, 0.65, 0.45])
        assert -1.0 <= delta <= 1.0


class TestHolmBonferroni:
    def test_known_correction(self) -> None:
        # m=3: smallest *3, next *2, largest *1, with monotonic enforcement.
        corrected = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.5})
        assert corrected["a"] == pytest.approx(0.03)
        assert corrected["b"] == pytest.approx(0.08)
        assert corrected["c"] == pytest.approx(0.5)

    def test_monotone_and_clipped(self) -> None:
        corrected = holm_bonferroni({"a": 0.3, "b": 0.4, "c": 0.9})
        # Corrected values are clipped to 1 and non-decreasing by rank.
        assert all(0.0 <= v <= 1.0 for v in corrected.values())
        assert corrected["a"] <= corrected["b"] <= corrected["c"]

    def test_empty(self) -> None:
        assert holm_bonferroni({}) == {}


class TestCompareEncodings:
    def _results(self) -> dict[str, list[float]]:
        return {
            "good": [0.9, 0.92, 0.88, 0.91],
            "mid": [0.7, 0.72, 0.68, 0.71],
            "poor": [0.5, 0.52, 0.48, 0.51],
        }

    def test_backward_compatible_pairwise(self) -> None:
        comp = compare_encodings(self._results())
        assert "good" in comp and "mid" in comp["good"]
        stat, p = comp["good"]["poor"]
        assert isinstance(stat, float) and isinstance(p, float)

    def test_corrected_structure(self) -> None:
        out = compare_encodings_corrected(self._results())
        assert out["n_comparisons"] == 3  # 3 choose 2
        assert out["alpha"] == 0.05
        for pair in out["pairs"]:
            assert {
                "encoding_a",
                "encoding_b",
                "p_value",
                "p_value_corrected",
                "cliffs_delta",
                "effect_magnitude",
                "significant",
            } <= set(pair)
            assert 0.0 <= pair["p_value_corrected"] <= 1.0

    def test_corrected_sorted_by_corrected_pvalue(self) -> None:
        out = compare_encodings_corrected(self._results())
        ps = [p["p_value_corrected"] for p in out["pairs"]]
        assert ps == sorted(ps)
