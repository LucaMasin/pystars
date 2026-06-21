"""Data normalization: convert long/wide inputs into a canonical long format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class LongData:
    """Canonical long-format data produced by :func:`normalize_data`.

    Attributes:
        df: Long-format dataframe with a ``value`` column, a ``group`` column
            (composite string for multi-factor designs), and optional ``subject``
            and individual factor columns.
        group_cols: Names of the factor column(s) in ``df``.
        subject_col: Name of the subject column in ``df`` (``"subject"``) or ``None``.
        value_col: Name of the value column in ``df`` (always ``"value"``).
    """

    df: pd.DataFrame
    group_cols: list[str]
    subject_col: str | None
    value_col: str


def normalize_data(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    subject: str | None = None,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> LongData:
    """Normalize an input dataframe into canonical long format.

    Parameters
    ----------
    df:
        Input dataframe in long or wide format.
    value:
        (Long format) Name of the outcome column.
    group:
        (Long format) Name of the grouping column, or list of factor columns
        for a factorial design.
    subject:
        (Long format) Name of the subject/id column (required for paired tests).
    format:
        ``"long"`` (default) or ``"wide"``.
    groups:
        (Wide format) Column names representing each group.
    subject_index:
        (Wide format) Optional subject-id column; if absent, the row index is used.

    Returns
    -------
    LongData
        Canonical long-format data.
    """
    if format == "long":
        return _normalize_long(df, value=value, group=group, subject=subject)
    if format == "wide":
        return _normalize_wide(df, groups=groups, subject_index=subject_index)
    raise ValueError(f"format must be 'long' or 'wide', got {format!r}")


def _normalize_long(
    df: pd.DataFrame,
    *,
    value: str | None,
    group: str | list[str] | None,
    subject: str | None,
) -> LongData:
    if value is None:
        raise ValueError("long format requires a `value` column name.")
    if value not in df.columns:
        raise ValueError(f"value column {value!r} not found in dataframe.")
    if group is None:
        raise ValueError("long format requires a `group` column name.")
    group_cols = [group] if isinstance(group, str) else list(group)
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group column(s) {missing!r} not found in dataframe.")

    subject_col: str | None = None
    if subject is not None:
        if subject not in df.columns:
            raise ValueError(f"subject column {subject!r} not found in dataframe.")
        subject_col = "subject"

    out = df.copy()
    out = out.rename(columns={value: "value", **({subject: "subject"} if subject else {})})
    # Build composite group label for multi-factor designs.
    if len(group_cols) > 1:
        out["group"] = out[group_cols].astype(str).agg("::".join, axis=1)
    else:
        out["group"] = out[group_cols[0]].astype(str)

    keep = ["value", "group"]
    for c in group_cols:
        if c not in keep:
            keep.append(c)
    if subject_col is not None and "subject" not in keep:
        keep.append("subject")
    out = out.loc[:, keep].reset_index(drop=True)
    return LongData(df=out, group_cols=group_cols, subject_col=subject_col, value_col="value")


def _normalize_wide(
    df: pd.DataFrame,
    *,
    groups: list[str] | None,
    subject_index: str | None,
) -> LongData:
    if not groups:
        raise ValueError("wide format requires `groups` (list of group column names).")
    missing = [c for c in groups if c not in df.columns]
    if missing:
        raise ValueError(f"group column(s) {missing!r} not found in dataframe.")

    subject_col = "subject"
    id_cols: list[str] = []
    if subject_index is not None:
        if subject_index not in df.columns:
            raise ValueError(f"subject column {subject_index!r} not found in dataframe.")
        id_cols = [subject_index]

    melted = df.melt(
        id_vars=id_cols,
        value_vars=groups,
        var_name="group",
        value_name="value",
    )
    if id_cols:
        melted = melted.rename(columns={subject_index: "subject"})
    else:
        melted["subject"] = melted.index.astype(str)
    melted["group"] = melted["group"].astype(str)
    melted = melted.loc[:, ["value", "group", "subject"]].reset_index(drop=True)
    return LongData(df=melted, group_cols=["group"], subject_col=subject_col, value_col="value")


def paired_differences(data: LongData) -> np.ndarray:
    """Compute per-subject paired differences for a 2-group long-format dataset.

    Parameters
    ----------
    data:
        Canonical long data with a ``subject`` column and exactly 2 groups.

    Returns
    -------
    np.ndarray
        Differences ``group2 - group1`` for each subject present in both groups.
    """
    if data.subject_col is None:
        raise ValueError("paired tests require a subject column.")
    groups = data.df["group"].unique().tolist()
    if len(groups) != 2:
        raise ValueError(f"paired tests require exactly 2 groups, got {len(groups)}.")
    pivot = data.df.pivot(index="subject", columns="group", values="value")
    pivot = pivot.dropna()
    g1, g2 = groups[0], groups[1]
    return (pivot[g2] - pivot[g1]).to_numpy()


def n_per_group(data: LongData) -> dict[str, int]:
    """Return a mapping of group label to sample size."""
    return data.df.groupby("group")["value"].size().to_dict()
