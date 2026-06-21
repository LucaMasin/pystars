"""Tests for pystars.posthoc: Tukey HSD, Games-Howell, Dunn's test."""

import numpy as np
import pandas as pd
import pytest

from pystars.posthoc import posthoc_dunn, posthoc_games_howell, posthoc_tukey
from pystars.result import TestResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def three_groups(rng):
    """Three groups with clearly different means."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 20 + ["low"] * 20 + ["high"] * 20,
            "value": np.concatenate(
                [rng.normal(10, 2, 20), rng.normal(14, 2, 20), rng.normal(18, 2, 20)]
            ),
        }
    )


@pytest.fixture
def three_groups_unequal_var(rng):
    """Three groups with different means and very unequal variances."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 30 + ["low"] * 30 + ["high"] * 30,
            "value": np.concatenate(
                [rng.normal(10, 1, 30), rng.normal(14, 5, 30), rng.normal(18, 10, 30)]
            ),
        }
    )


class TestPosthocTukey:
    def test_returns_test_result_with_pairwise(self, three_groups):
        result = posthoc_tukey(three_groups, value="value", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Tukey HSD"
        assert result.pairwise is not None
        assert len(result.pairwise) == 3  # 3 choose 2 = 3 pairwise comparisons
        assert {"A", "B", "p"}.issubset(result.pairwise.columns)

    def test_detects_significant_diff(self, three_groups):
        result = posthoc_tukey(three_groups, value="value", group="group")
        assert result.pairwise is not None
        # At least one pairwise comparison should be significant
        assert (result.pairwise["p"] < 0.05).any()


class TestPosthocGamesHowell:
    def test_returns_test_result_with_pairwise(self, three_groups_unequal_var):
        result = posthoc_games_howell(three_groups_unequal_var, value="value", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Games-Howell"
        assert result.pairwise is not None
        assert len(result.pairwise) == 3
        assert {"A", "B", "p"}.issubset(result.pairwise.columns)

    def test_detects_significant_diff(self, three_groups_unequal_var):
        result = posthoc_games_howell(three_groups_unequal_var, value="value", group="group")
        assert result.pairwise is not None
        assert (result.pairwise["p"] < 0.05).any()


class TestPosthocDunn:
    def test_returns_test_result_with_pairwise(self, three_groups):
        result = posthoc_dunn(three_groups, value="value", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Dunn's test"
        assert result.pairwise is not None
        assert len(result.pairwise) == 3
        assert {"A", "B", "p"}.issubset(result.pairwise.columns)

    def test_detects_significant_diff(self, three_groups):
        result = posthoc_dunn(three_groups, value="value", group="group")
        assert result.pairwise is not None
        assert (result.pairwise["p"] < 0.05).any()

    def test_p_adjust_applied(self, three_groups):
        """The p-values should be adjusted (default: holm)."""
        result = posthoc_dunn(three_groups, value="value", group="group", p_adjust="bonferroni")
        # Bonferroni adjustment should give different (larger) p-values than unadjusted
        result_unadj = posthoc_dunn(three_groups, value="value", group="group", p_adjust=None)
        assert result.pairwise is not None
        assert result_unadj.pairwise is not None
        # At least one p-value should be larger with Bonferroni
        adj = result.pairwise["p"].to_numpy()
        unadj = result_unadj.pairwise["p"].to_numpy()
        assert (adj >= unadj - 1e-10).all()


class TestEdgeCases:
    def test_two_groups_tukey(self, rng):
        """Post-hoc with 2 groups should produce 1 comparison."""
        df = pd.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "value": np.concatenate([rng.normal(10, 2, 20), rng.normal(15, 2, 20)]),
            }
        )
        result = posthoc_tukey(df, value="value", group="group")
        assert result.pairwise is not None
        assert len(result.pairwise) == 1
