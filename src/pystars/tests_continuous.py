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

    Independent samples use Welch's t-test by default, which does not assume
    equal variances. Set ``welch=False`` to use Student's pooled-variance
    t-test. If ``paired=True``, the function compares the two measurements for
    each subject and ignores ``welch``.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column. Exactly two groups are required.
    subject:
        For long format paired data, the subject or matched-unit column.
        Required when ``paired=True``.
    paired:
        If ``True``, perform a paired t-test on two measurements per subject.
        If ``False`` (default), treat the two groups as independent.
    welch:
        For independent samples, use Welch's t-test when ``True`` (default) or
        Student's t-test when ``False``. This option is ignored for paired data.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the two value-column names representing the groups.
        Their order determines the order of the comparison.
    subject_index:
        For wide format, an optional subject-id column. If omitted, row
        indices are used as matched subject labels.

    Returns
    -------
    TestResult
        A result containing the selected test name, t statistic, p-value,
        Cohen's d, and the 95% confidence interval under ``effect_size``.

    Raises
    ------
    ValueError
        If the input schema is invalid, the independent data does not contain
        exactly two groups, or paired data lacks a subject column or has other
        than two groups.

    Examples
    --------
    >>> result = ttest(df, value="length", group="genotype", welch=True)
    >>> result.test_name
    "Welch's t-test"
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
    """Perform the Mann-Whitney U test for two independent groups.

    This is a non-parametric alternative to an independent-samples t-test. It
    tests whether the two groups have different distributions and is useful
    when normality is not a reasonable assumption. Exactly two groups are
    required.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column. Exactly two groups are required.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the two value-column names representing the groups.
    subject_index:
        For wide format, an optional subject-id column. It is not used because
        this test treats observations as independent.

    Returns
    -------
    TestResult
        A result containing the Mann-Whitney ``U`` statistic, p-value, and the
        common-language effect size (``CLES``) and rank-biserial correlation
        (``RBC``) under ``effect_size``.

    Raises
    ------
    ValueError
        If the input schema is invalid or the data does not contain exactly
        two groups.

    Examples
    --------
    >>> result = mannwhitney(df, value="length", group="genotype")
    >>> result.test_name
    'Mann-Whitney U test'
    """
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
    """Perform the Wilcoxon signed-rank test for paired measurements.

    The test compares two measurements from the same subjects without assuming
    normally distributed paired differences. Exactly two groups and a subject
    identifier are required. Subjects missing either measurement are excluded
    by the paired-data pivot.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column. Exactly two groups are required.
    subject:
        For long format, the subject or matched-unit column. Required for
        paired data.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the two value-column names representing the groups.
    subject_index:
        For wide format, an optional subject-id column. If omitted, row
        indices are used as subject labels.

    Returns
    -------
    TestResult
        A result containing the Wilcoxon ``W`` statistic, p-value, and the
        common-language effect size (``CLES``) and rank-biserial correlation
        (``RBC``) under ``effect_size``.

    Raises
    ------
    ValueError
        If the input schema is invalid, no subject column is available, or
        the data does not contain exactly two groups.

    Examples
    --------
    >>> result = wilcoxon(df, value="length", group="genotype", subject="animal")
    >>> result.test_name
    'Wilcoxon signed-rank test'
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
    """Perform a one-way ANOVA or Welch's ANOVA across independent groups.

    Use the classic one-way ANOVA when group variances can reasonably be
    treated as equal. Set ``welch=True`` for Welch's ANOVA when variances are
    unequal. A list passed to ``group`` is normalized into composite group
    labels and is therefore analyzed as one group factor; use
    :func:`anova_twoway` to report separate factorial effects.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column, or a list of factor columns
        whose combinations should be treated as groups. At least two groups
        are required.
    welch:
        If ``True``, use Welch's ANOVA for unequal variances. If ``False``
        (default), use classic one-way ANOVA.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the value-column names representing the groups.
    subject_index:
        For wide format, an optional subject-id column. It is not used by this
        independent-groups analysis.

    Returns
    -------
    TestResult
        A result containing the ANOVA ``F`` statistic, p-value, and partial
        eta-squared (``np2``) under ``effect_size``. The test name is
        ``"One-way ANOVA"`` or ``"Welch's ANOVA"``.

    Raises
    ------
    ValueError
        If the input schema is invalid or fewer than two groups are present.

    Examples
    --------
    >>> result = anova(df, value="length", group="treatment")
    >>> result.test_name
    'One-way ANOVA'
    """
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
    """Perform the Kruskal-Wallis test for two or more independent groups.

    This non-parametric omnibus test is an alternative to one-way ANOVA when
    normality is not a reasonable assumption. It does not identify which
    groups differ; use :func:`pystars.posthoc_dunn` for follow-up pairwise
    comparisons.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column, or a list of factor columns
        whose combinations should be treated as groups. At least two groups
        are required.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the value-column names representing the groups.
    subject_index:
        For wide format, an optional subject-id column. It is not used by this
        independent-groups analysis.

    Returns
    -------
    TestResult
        A result containing the Kruskal-Wallis ``H`` statistic, p-value, and
        epsilon-squared effect size (``epsilon_squared``), calculated as
        ``H / (n - 1)``.

    Raises
    ------
    ValueError
        If the input schema is invalid or fewer than two groups are present.

    Examples
    --------
    >>> result = kruskal(df, value="length", group="treatment")
    >>> result.test_name
    'Kruskal-Wallis test'
    """
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

    The ``group`` parameter must be a list of at least two factor column names.
    The interaction term is reported as the main result when one is present;
    if the backend does not produce an interaction row, the last non-residual
    source is reported instead. The full ANOVA table, including all main
    effects and interactions, is stored in ``result.details``.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe in long format.
    value:
        Name of the numeric outcome column.
    group:
        List of at least two factor column names. Factor combinations are also
        represented by a composite ``"factor1::factor2"`` group label in the
        normalized data.
    format:
        Input layout. Use ``"long"`` (default) for factorial data; wide input
        has only one generated group column and cannot describe multiple
        factors.

    Returns
    -------
    TestResult
        A result named for the reported ANOVA source, containing its ``F``
        statistic, p-value, partial eta-squared (``np2``), and the complete
        ANOVA table in ``details``.

    Raises
    ------
    ValueError
        If the input schema is invalid or fewer than two factor columns are
        supplied.

    Examples
    --------
    >>> result = anova_twoway(
    ...     df, value="length", group=["genotype", "time"]
    ... )
    >>> details = result.details
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
