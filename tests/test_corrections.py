"""Tests for multiple-comparison correction helpers."""

import numpy as np
import pandas as pd
import pytest

from pystars.corrections import adjust_pairwise, adjust_pvalues, adjust_results
from pystars.result import TestResult, to_dataframe


def make_results() -> list[TestResult]:
    return [
        TestResult("test a", 1.0, 0.01),
        TestResult("test b", 2.0, 0.03),
        TestResult("test c", 3.0, 0.20),
    ]


class TestAdjustPvalues:
    def test_holm_correction_returns_table(self):
        adjusted = adjust_pvalues([0.01, 0.03, 0.20], method="holm", alpha=0.05)

        assert adjusted["p_adjusted"].to_list() == pytest.approx([0.03, 0.06, 0.20])
        assert adjusted["reject"].to_list() == [True, False, False]
        assert (adjusted["p_adjust_method"] == "holm").all()
        assert (adjusted["p_adjust_alpha"] == 0.05).all()

    def test_fdr_bh_correction(self):
        adjusted = adjust_pvalues([0.50, 0.003, 0.32, 0.054, 0.0003], method="fdr_bh")

        assert adjusted["p_adjusted"].to_list() == pytest.approx([0.5, 0.0075, 0.4, 0.09, 0.0015])
        assert adjusted["reject"].to_list() == [False, True, False, False, True]

    def test_nan_is_preserved_and_not_rejected(self):
        adjusted = adjust_pvalues([0.01, np.nan, 0.20], method="holm")

        assert adjusted.loc[0, "p_adjusted"] == pytest.approx(0.02)
        assert np.isnan(adjusted.loc[1, "p_adjusted"])
        assert adjusted.loc[1, "reject"] is False

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            adjust_pvalues([0.01], alpha=1.0)

    def test_invalid_p_value_raises(self):
        with pytest.raises(ValueError, match="p-values"):
            adjust_pvalues([1.2])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            adjust_pvalues([0.01], method="unknown")


class TestAdjustResults:
    def test_copy_by_default(self):
        results = make_results()
        adjusted = adjust_results(results, method="holm")

        assert adjusted is not results
        assert adjusted[0] is not results[0]
        assert results[0].p_adjusted is None
        assert adjusted[0].p_adjusted == pytest.approx(0.03)
        assert adjusted[0].reject is True
        assert adjusted[0].p_adjust_method == "holm"

    def test_inplace_updates_original_results(self):
        results = make_results()
        adjusted = adjust_results(results, method="holm", inplace=True)

        assert adjusted is results
        assert results[0].p_adjusted == pytest.approx(0.03)

    def test_empty_results_returns_empty_list(self):
        assert adjust_results([]) == []


class TestAdjustPairwise:
    def test_adjusts_pairwise_table(self):
        pairwise = pd.DataFrame(
            {
                "A": ["a", "a", "b"],
                "B": ["b", "c", "c"],
                "p": [0.01, 0.03, 0.20],
            }
        )

        adjusted = adjust_pairwise(pairwise, method="holm")

        assert adjusted is not pairwise
        assert adjusted["p_adjusted"].to_list() == pytest.approx([0.03, 0.06, 0.20])
        assert adjusted["reject"].to_list() == [True, False, False]
        assert "p_adjust_method" in adjusted.columns

    def test_custom_p_column(self):
        pairwise = pd.DataFrame({"raw_p": [0.01, 0.03]})

        adjusted = adjust_pairwise(pairwise, p_col="raw_p", method="bonf")

        assert adjusted["p_adjusted"].to_list() == pytest.approx([0.02, 0.06])

    def test_missing_p_column_raises(self):
        with pytest.raises(ValueError, match="p_col"):
            adjust_pairwise(pd.DataFrame({"pval": [0.01]}), p_col="p")


class TestToDataframeCorrection:
    def test_module_level_to_dataframe_can_adjust_results(self):
        df = to_dataframe(make_results(), p_adjust="holm")

        assert df["p_adjusted"].to_list() == pytest.approx([0.03, 0.06, 0.20])
        assert df["reject"].to_list() == [True, False, False]
        assert df["p_adjust_method"].to_list() == ["holm", "holm", "holm"]
