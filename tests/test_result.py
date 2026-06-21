"""Tests for pystars.result: TestResult, to_dataframe, summary, rich output."""

import io

import pandas as pd
import pytest
from rich.console import Console

from pystars.result import TestResult, to_dataframe


def make_simple_result() -> TestResult:
    return TestResult(
        test_name="Welch's t-test",
        statistic=2.45,
        p_value=0.025,
        effect_size={"cohen_d": 0.8, "CI95%": [0.2, 1.4]},
    )


def make_result_with_assumptions() -> TestResult:
    return TestResult(
        test_name="Welch's t-test",
        statistic=2.45,
        p_value=0.025,
        effect_size={"cohen_d": 0.8, "CI95%": [0.2, 1.4]},
        assumptions={
            "normality": {"method": "shapiro", "statistic": 0.97, "p": 0.45},
            "equal_variance": {"method": "levene", "statistic": 2.0, "p": 0.12},
        },
    )


def make_result_with_posthoc() -> TestResult:
    posthoc = TestResult(
        test_name="Tukey HSD",
        statistic=3.1,
        p_value=0.01,
        effect_size={},
        pairwise=pd.DataFrame({"A": ["wt"], "B": ["mut"], "p": [0.01], "diff": [2.0]}),
    )
    return TestResult(
        test_name="One-way ANOVA",
        statistic=5.2,
        p_value=0.01,
        effect_size={"np2": 0.3},
        posthoc=posthoc,
    )


class TestToDataframe:
    def test_simple_result_core_columns(self):
        df = make_simple_result().to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.loc[0, "test"] == "Welch's t-test"
        assert df.loc[0, "statistic"] == pytest.approx(2.45)
        assert df.loc[0, "p_value"] == pytest.approx(0.025)

    def test_effect_size_flattened(self):
        df = make_simple_result().to_dataframe()
        assert "cohen_d" in df.columns
        assert df.loc[0, "cohen_d"] == pytest.approx(0.8)
        # CI95% list expanded into _0 / _1
        assert "CI95%_0" in df.columns
        assert "CI95%_1" in df.columns
        assert df.loc[0, "CI95%_0"] == pytest.approx(0.2)
        assert df.loc[0, "CI95%_1"] == pytest.approx(1.4)

    def test_assumptions_flattened(self):
        df = make_result_with_assumptions().to_dataframe()
        assert "normality_p" in df.columns
        assert "normality_statistic" in df.columns
        assert "equal_variance_p" in df.columns
        assert df.loc[0, "normality_p"] == pytest.approx(0.45)

    def test_posthoc_present_marked(self):
        df = make_result_with_posthoc().to_dataframe()
        assert "posthoc" in df.columns
        assert df.loc[0, "posthoc"] == "Tukey HSD"

    def test_posthoc_absent_is_none(self):
        df = make_simple_result().to_dataframe()
        assert "posthoc" in df.columns
        assert pd.isna(df.loc[0, "posthoc"])

    def test_module_level_to_dataframe_concatenates(self):
        r1 = make_simple_result()
        r2 = make_result_with_posthoc()
        df = to_dataframe([r1, r2])
        assert len(df) == 2
        assert df.loc[0, "test"] == "Welch's t-test"
        assert df.loc[1, "test"] == "One-way ANOVA"

    def test_module_level_to_dataframe_single_result(self):
        df = to_dataframe(make_simple_result())
        assert len(df) == 1


class TestSummary:
    def test_summary_contains_test_name(self):
        s = make_simple_result().summary()
        assert "Welch's t-test" in s

    def test_summary_contains_p_value(self):
        s = make_simple_result().summary()
        assert "0.025" in s

    def test_summary_contains_effect_size(self):
        s = make_simple_result().summary()
        assert "cohen_d" in s.lower() or "cohen" in s.lower()

    def test_summary_contains_assumptions(self):
        s = make_result_with_assumptions().summary()
        assert "normality" in s.lower()
        assert "equal variance" in s.lower()

    def test_summary_contains_posthoc(self):
        s = make_result_with_posthoc().summary()
        assert "tukey" in s.lower()


class TestRichOutput:
    def test_show_runs_without_error(self):
        buf = io.StringIO()
        console = Console(file=buf, width=80)
        make_result_with_assumptions().show(console=console)
        output = buf.getvalue()
        assert "Welch's t-test" in output

    def test_rich_returns_renderable(self):
        result = make_simple_result()
        renderable = result.__rich__()
        assert renderable is not None
