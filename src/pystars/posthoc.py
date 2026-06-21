"""Post-hoc tests: Tukey HSD, Games-Howell, Dunn's test."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pingouin as pg
import scikit_posthocs as sp

from pystars.data import LongData, normalize_data
from pystars.result import TestResult


def _require_min_groups(data: LongData, n: int, test_name: str) -> None:
    n_groups = data.df["group"].nunique()
    if n_groups < n:
        raise ValueError(f"{test_name} requires at least {n} groups, got {n_groups}.")


def _matrix_to_pairwise(matrix: pd.DataFrame, *, p_adjust: str | None) -> pd.DataFrame:
    """Convert a symmetric p-value matrix to a tidy pairwise dataframe."""
    labels = matrix.index.tolist()
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append(
                {
                    "A": labels[i],
                    "B": labels[j],
                    "p": float(matrix.iloc[i, j]),
                    "p_adjust": p_adjust or "none",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- Tukey


def posthoc_tukey(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Tukey HSD post-hoc test (for one-way ANOVA with equal variance)."""
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    _require_min_groups(data, 2, "Tukey HSD")
    pg_res = pg.pairwise_tukey(data=data.df, dv="value", between="group")
    pairwise = pg_res.loc[:, ["A", "B", "diff", "p_tukey"]].rename(columns={"p_tukey": "p"})
    return TestResult(
        test_name="Tukey HSD",
        statistic=float("nan"),
        p_value=float("nan"),
        effect_size={},
        pairwise=pairwise.reset_index(drop=True),
    )


# ------------------------------------------------------------- Games-Howell


def posthoc_games_howell(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Games-Howell post-hoc test (for Welch's ANOVA with unequal variance)."""
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    _require_min_groups(data, 2, "Games-Howell")
    pg_res = pg.pairwise_gameshowell(data=data.df, dv="value", between="group")
    pairwise = pg_res.loc[:, ["A", "B", "diff", "pval"]].rename(columns={"pval": "p"})
    return TestResult(
        test_name="Games-Howell",
        statistic=float("nan"),
        p_value=float("nan"),
        effect_size={},
        pairwise=pairwise.reset_index(drop=True),
    )


# --------------------------------------------------------------------- Dunn


def posthoc_dunn(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    p_adjust: str | None = "holm",
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Dunn's post-hoc test (for Kruskal-Wallis, non-parametric).

    Parameters
    ----------
    p_adjust:
        Multiple-comparison correction method (default ``"holm"``).
        See :func:`scikit_posthocs.posthoc_dunn` for available methods.
        Pass ``None`` for unadjusted p-values.
    """
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    _require_min_groups(data, 2, "Dunn's test")
    matrix = sp.posthoc_dunn(data.df, val_col="value", group_col="group", p_adjust=p_adjust)
    pairwise = _matrix_to_pairwise(matrix, p_adjust=p_adjust)
    return TestResult(
        test_name="Dunn's test",
        statistic=float("nan"),
        p_value=float("nan"),
        effect_size={},
        pairwise=pairwise,
    )
