"""Continuous-data statistical tests: t-test, Mann-Whitney, Wilcoxon, ANOVA, Kruskal-Wallis."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pingouin as pg

from pystars.data import LongData, normalize_data
from pystars.result import TestResult


def _groups_as_arrays(data: LongData) -> list[tuple[str, np.ndarray]]:
    """Return [(label, values), ...] for each group in the data."""
    return [(str(label), grp["value"].to_numpy()) for label, grp in data.df.groupby("group")]


def _require_n_groups(data: LongData, n: int, test_name: str) -> list[tuple[str, np.ndarray]]:
    groups = _groups_as_arrays(data)
    if len(groups) != n:
        raise ValueError(f"{test_name} requires exactly {n} groups, got {len(groups)}.")
    return groups


def _require_min_groups(data: LongData, n: int, test_name: str) -> list[tuple[str, np.ndarray]]:
    groups = _groups_as_arrays(data)
    if len(groups) < n:
        raise ValueError(f"{test_name} requires at least {n} groups, got {len(groups)}.")
    return groups


def _paired_pivot(data: LongData) -> pd.DataFrame:
    """Pivot paired data to wide form (subjects × groups), dropping missing."""
    if data.subject_col is None:
        raise ValueError("paired tests require a subject column.")
    pivot = data.df.pivot(index="subject", columns="group", values="value").dropna()
    if pivot.shape[1] != 2:
        raise ValueError(f"paired tests require exactly 2 groups, got {pivot.shape[1]}.")
    return pivot


# --------------------------------------------------------------------- t-test


def ttest(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    subject: str | None = None,
    paired: bool = False,
    welch: bool = True,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Perform a t-test (Welch, Student's, or paired).

    By default uses Welch's t-test (``welch=True``), which is safer for
    biological data where equal variance is rarely guaranteed.
    """
    data = normalize_data(
        df,
        value=value,
        group=group,
        subject=subject,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )

    if paired:
        pivot = _paired_pivot(data)
        g1, g2 = pivot.columns[0], pivot.columns[1]
        pg_res = pg.ttest(pivot[g1], pivot[g2], paired=True)
        test_name = "Paired t-test"
    else:
        groups_arr = _require_n_groups(data, 2, "t-test")
        x, y = groups_arr[0][1], groups_arr[1][1]
        pg_res = pg.ttest(x, y, paired=False, correction=bool(welch))  # type: ignore[reportArgumentType]
        test_name = "Welch's t-test" if welch else "Student's t-test"

    row = pg_res.iloc[0]
    ci95 = row["CI95"]
    effect_size: dict[str, Any] = {
        "cohen_d": float(row["cohen_d"]),
        "CI95%": [float(ci95[0]), float(ci95[1])],
    }
    return TestResult(
        test_name=test_name,
        statistic=float(row["T"]),
        p_value=float(row["p_val"]),
        effect_size=effect_size,
    )


# ----------------------------------------------------- Mann-Whitney U test


def mannwhitney(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Perform the Mann-Whitney U test (non-parametric, 2 independent groups)."""
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    groups_arr = _require_n_groups(data, 2, "Mann-Whitney U test")
    x, y = groups_arr[0][1], groups_arr[1][1]
    pg_res = pg.mwu(x, y)
    row = pg_res.iloc[0]
    return TestResult(
        test_name="Mann-Whitney U test",
        statistic=float(row["U_val"]),
        p_value=float(row["p_val"]),
        effect_size={"CLES": float(row["CLES"]), "RBC": float(row["RBC"])},
    )


# ------------------------------------------------------- Wilcoxon signed-rank


def wilcoxon(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    subject: str | None = None,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Perform the Wilcoxon signed-rank test (non-parametric, paired)."""
    data = normalize_data(
        df,
        value=value,
        group=group,
        subject=subject,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    pivot = _paired_pivot(data)
    g1, g2 = pivot.columns[0], pivot.columns[1]
    pg_res = pg.wilcoxon(pivot[g1], pivot[g2])
    row = pg_res.iloc[0]
    return TestResult(
        test_name="Wilcoxon signed-rank test",
        statistic=float(row["W_val"]),
        p_value=float(row["p_val"]),
        effect_size={"CLES": float(row["CLES"]), "RBC": float(row["RBC"])},
    )


# --------------------------------------------------------------------- ANOVA


def anova(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    welch: bool = False,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Perform one-way ANOVA (or Welch's ANOVA for unequal variances)."""
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    _require_min_groups(data, 2, "ANOVA")

    if welch:
        pg_res = pg.welch_anova(data=data.df, dv="value", between="group")
        test_name = "Welch's ANOVA"
    else:
        pg_res = pg.anova(data=data.df, dv="value", between="group", detailed=True)
        test_name = "One-way ANOVA"

    row = pg_res.iloc[0]
    return TestResult(
        test_name=test_name,
        statistic=float(row["F"]),
        p_value=float(row["p_unc"]),
        effect_size={"np2": float(row["np2"])},
    )


# ------------------------------------------------------- Kruskal-Wallis test


def kruskal(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Perform the Kruskal-Wallis test (non-parametric, ≥2 independent groups)."""
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    _require_min_groups(data, 2, "Kruskal-Wallis test")

    pg_res = pg.kruskal(data=data.df, dv="value", between="group")
    row = pg_res.iloc[0]
    h = float(row["H"])
    n = len(data.df)
    return TestResult(
        test_name="Kruskal-Wallis test",
        statistic=h,
        p_value=float(row["p_unc"]),
        effect_size={"epsilon_squared": h / (n - 1)},
    )


# ------------------------------------------------------------- Two-way ANOVA


def anova_twoway(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    format: Literal["long", "wide"] = "long",
) -> TestResult:
    """Perform a two-way (factorial) ANOVA.

    The ``group`` parameter must be a list of at least 2 factor column names.
    The interaction term is reported as the main result; the full ANOVA table
    (all factors + interaction) is stored in ``result.details``.
    """
    data = normalize_data(df, value=value, group=group, format=format)
    if len(data.group_cols) < 2:
        raise ValueError(
            f"two-way ANOVA requires at least 2 factor columns, got {len(data.group_cols)}."
        )

    pg_res = pg.anova(data=data.df, dv="value", between=data.group_cols, detailed=True)

    # Report the interaction term as the main result, or the last non-residual source.
    interaction = pg_res[pg_res["Source"].str.contains(" \\* ", regex=False)]
    if not interaction.empty:
        row = interaction.iloc[0]
        source_name = str(row["Source"])
    else:
        non_residual = pg_res[pg_res["Source"] != "Residual"]
        row = non_residual.iloc[-1]
        source_name = str(row["Source"])

    return TestResult(
        test_name=f"Two-way ANOVA ({source_name})",
        statistic=float(row["F"]),
        p_value=float(row["p_unc"]),
        effect_size={"np2": float(row["np2"])},
        details=pg_res,
    )
