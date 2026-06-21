"""Tests for pystars.data: long/wide normalization and validation."""

import pandas as pd
import pytest

from pystars.data import LongData, normalize_data


class TestLongFormat:
    def test_long_passthrough_single_factor(self):
        df = pd.DataFrame(
            {
                "group": ["wt", "wt", "mut", "mut"],
                "length": [1.0, 2.0, 3.0, 4.0],
                "animal": ["a1", "a2", "a3", "a4"],
            }
        )
        result = normalize_data(df, value="length", group="group", subject="animal", format="long")
        assert isinstance(result, LongData)
        assert result.group_cols == ["group"]
        assert result.subject_col == "subject"
        assert result.value_col == "value"
        assert len(result.df) == 4
        assert "value" in result.df.columns
        assert "group" in result.df.columns
        assert "subject" in result.df.columns

    def test_long_without_subject(self):
        df = pd.DataFrame({"genotype": ["wt", "wt", "mut", "mut"], "length": [1.0, 2.0, 3.0, 4.0]})
        result = normalize_data(df, value="length", group="genotype", format="long")
        assert result.subject_col is None
        assert "subject" not in result.df.columns

    def test_long_multi_factor_preserves_factor_columns(self):
        df = pd.DataFrame(
            {
                "genotype": ["wt", "wt", "mut", "mut"],
                "time": ["0", "1", "0", "1"],
                "length": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = normalize_data(df, value="length", group=["genotype", "time"], format="long")
        assert result.group_cols == ["genotype", "time"]
        assert "genotype" in result.df.columns
        assert "time" in result.df.columns
        assert "group" in result.df.columns
        # composite group combines factors
        assert result.df["group"].nunique() == 4

    def test_long_missing_value_column_raises(self):
        df = pd.DataFrame({"group": ["wt", "mut"]})
        with pytest.raises(ValueError, match="value"):
            normalize_data(df, value="length", group="group", format="long")

    def test_long_missing_group_column_raises(self):
        df = pd.DataFrame({"length": [1.0, 2.0]})
        with pytest.raises(ValueError, match="group"):
            normalize_data(df, value="length", group="genotype", format="long")


class TestWideFormat:
    def test_wide_two_groups_converts_to_long(self):
        df = pd.DataFrame({"wt": [1.0, 2.0], "mut": [3.0, 4.0]})
        result = normalize_data(df, format="wide", groups=["wt", "mut"])
        assert result.group_cols == ["group"]
        assert result.subject_col is not None  # generated from index
        assert len(result.df) == 4
        assert set(result.df["group"].unique()) == {"wt", "mut"}
        assert "value" in result.df.columns
        assert "subject" in result.df.columns

    def test_wide_with_subject_index(self):
        df = pd.DataFrame({"animal": ["a1", "a2"], "wt": [1.0, 2.0], "mut": [3.0, 4.0]})
        result = normalize_data(df, format="wide", groups=["wt", "mut"], subject_index="animal")
        assert result.subject_col == "subject"
        assert set(result.df["subject"].unique()) == {"a1", "a2"}

    def test_wide_missing_group_column_raises(self):
        df = pd.DataFrame({"wt": [1.0, 2.0]})
        with pytest.raises(ValueError, match="group"):
            normalize_data(df, format="wide", groups=["wt", "mut"])

    def test_wide_requires_groups(self):
        df = pd.DataFrame({"wt": [1.0, 2.0], "mut": [3.0, 4.0]})
        with pytest.raises(ValueError, match="groups"):
            normalize_data(df, format="wide")


class TestValidation:
    def test_invalid_format_raises(self):
        df = pd.DataFrame({"group": ["wt"], "length": [1.0]})
        with pytest.raises(ValueError, match="format"):
            normalize_data(df, value="length", group="group", format="square")  # type: ignore[reportArgumentType]

    def test_long_missing_subject_column_raises(self):
        df = pd.DataFrame({"group": ["wt", "mut"], "length": [1.0, 2.0]})
        with pytest.raises(ValueError, match="subject"):
            normalize_data(df, value="length", group="group", subject="animal", format="long")
