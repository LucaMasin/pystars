"""TestResult: unified container for test outputs, dataframe export, and rich display."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table


@dataclass
class TestResult:
    """Result of a statistical test.

    Attributes:
        test_name: Human-readable name of the test.
        statistic: Test statistic value. Post-hoc-only results use ``NaN``.
        p_value: Raw p-value of the test. Post-hoc-only results use ``NaN``.
        effect_size: Mapping of effect-size measures, such as
            ``{"cohen_d": 0.8, "CI95%": [0.2, 1.4]}``.
        assumptions: Mapping of assumption names to their test results, or
            ``None`` when no assumptions were attached.
        posthoc: A :class:`TestResult`, a list of results, or ``None`` for
            post-hoc comparisons.
        pairwise: Pairwise comparison table as a pandas dataframe, or ``None``
            when no pairwise results are available.
        details: Detailed dataframe, such as a full ANOVA table or per-group
            normality results, or ``None``.
        extra: Optional mapping of additional metadata, such as dispatcher
            decisions or assumption verdicts.
        p_adjusted: Corrected p-value, or ``None`` until a correction is
            applied.
        p_adjust_method: Name of the correction method, or ``None``.
        p_adjust_alpha: Significance threshold used for correction, or ``None``.
        reject: Whether ``p_adjusted`` is significant at ``p_adjust_alpha``,
            or ``None`` until a correction is applied.
    """

    test_name: str
    statistic: float
    p_value: float
    effect_size: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] | None = None
    posthoc: TestResult | list[TestResult] | None = None
    pairwise: pd.DataFrame | None = None
    details: pd.DataFrame | None = None
    extra: dict[str, Any] | None = None
    p_adjusted: float | None = None
    p_adjust_method: str | None = None
    p_adjust_alpha: float | None = None
    reject: bool | None = None

    # Prevent pytest from collecting this dataclass as a test class.
    __test__ = False

    # ------------------------------------------------------------------ export
    def to_dataframe(self) -> pd.DataFrame:
        """Convert this result to a single-row tidy dataframe.

        Parameters
        ----------
        None
            This method takes no parameters.

        Returns
        -------
        pandas.DataFrame
            A one-row dataframe containing the test name, statistic, p-value,
            optional correction fields, flattened effect sizes, flattened
            assumptions, and a post-hoc test-name column. Nested dictionaries
            use underscore-separated keys such as ``normality_p``; lists and
            tuples use indexed keys such as ``CI95%_0``. Detailed and pairwise
            dataframes are not expanded into this one-row representation.

        Examples
        --------
        >>> table = result.to_dataframe()
        >>> table.loc[0, "test"]
        "Welch's t-test"
        """
        row: dict[str, Any] = {
            "test": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }
        if self.p_adjusted is not None:
            row["p_adjusted"] = self.p_adjusted
        if self.reject is not None:
            row["reject"] = self.reject
        if self.p_adjust_method is not None:
            row["p_adjust_method"] = self.p_adjust_method
        if self.p_adjust_alpha is not None:
            row["p_adjust_alpha"] = self.p_adjust_alpha
        row.update(_flatten(self.effect_size))
        if self.assumptions:
            for prefix, sub in self.assumptions.items():
                row.update(_flatten(sub, prefix=prefix))
        if self.posthoc is not None:
            names = (
                self.posthoc.test_name
                if isinstance(self.posthoc, TestResult)
                else ", ".join(r.test_name for r in self.posthoc)
            )
            row["posthoc"] = names
        else:
            row["posthoc"] = pd.NA
        return pd.DataFrame([row])

    # ------------------------------------------------------------------ summary
    def summary(self) -> str:
        """Return a plain-text summary of this result.

        Parameters
        ----------
        None
            This method takes no parameters.

        Returns
        -------
        str
            Multiline text containing the test name, statistic, p-value,
            correction information, effect sizes, assumption verdicts, and
            post-hoc names when those fields are present. P-values below
            ``1e-4`` are displayed as ``<0.0001``. Assumption verdicts use
            ``0.05`` as their display threshold.

        Examples
        --------
        >>> summary_text = result.summary()
        """
        lines = [
            f"Test: {self.test_name}",
            f"Statistic: {self._fmt_statistic()}",
            f"p-value: {self._fmt_p(self.p_value)}",
        ]
        if self.p_adjusted is not None:
            lines.append(
                f"adjusted p-value ({self.p_adjust_method}): {self._fmt_p(self.p_adjusted)}"
            )
        if self.effect_size:
            es_parts = []
            for k, v in self.effect_size.items():
                if isinstance(v, list | tuple):
                    es_parts.append(f"{k}=[{', '.join(self._fmt_num(x) for x in v)}]")
                else:
                    es_parts.append(f"{k}={self._fmt_num(v)}")
            lines.append(f"Effect size: {', '.join(es_parts)}")
        if self.assumptions:
            lines.append("Assumptions:")
            for name, sub in self.assumptions.items():
                method = sub.get("method", "")
                p = sub.get("p")
                verdict = "not rejected" if (p is not None and p > 0.05) else "rejected"
                p_str = self._fmt_p(p) if p is not None else "N/A"
                lines.append(f"  {name.replace('_', ' ')} ({method}): p={p_str} ({verdict})")
        if self.posthoc is not None:
            names = (
                self.posthoc.test_name
                if isinstance(self.posthoc, TestResult)
                else ", ".join(r.test_name for r in self.posthoc)
            )
            lines.append(f"Post-hoc: {names}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ rich
    def __rich__(self) -> Any:
        """Build the Rich renderable used when this result is printed.

        Parameters
        ----------
        None
            This method takes no parameters.

        Returns
        -------
        rich.panel.Panel
            A Rich panel containing the test result, assumptions, detailed
            table, and pairwise table when available.
        """
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Test", self.test_name)
        table.add_row("Statistic", self._fmt_statistic())
        table.add_row("p-value", self._fmt_p(self.p_value))
        if self.p_adjusted is not None:
            table.add_row(
                f"adjusted p-value ({self.p_adjust_method})",
                self._fmt_p(self.p_adjusted),
            )
        if self.reject is not None:
            table.add_row("reject", str(self.reject))
        if self.effect_size:
            for k, v in self.effect_size.items():
                if isinstance(v, list | tuple):
                    table.add_row(k, f"[{', '.join(self._fmt_num(x) for x in v)}]")
                else:
                    table.add_row(k, self._fmt_num(v))

        renderables: list[Any] = [table]

        if self.assumptions:
            assump = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
            assump.add_column("Assumption", style="cyan")
            assump.add_column("Method")
            assump.add_column("p-value", justify="right")
            assump.add_column("Verdict")
            for name, sub in self.assumptions.items():
                p = sub.get("p")
                verdict = "not rejected" if (p is not None and p > 0.05) else "rejected"
                assump.add_row(
                    name.replace("_", " "),
                    str(sub.get("method", "")),
                    self._fmt_p(p) if p is not None else "N/A",
                    verdict,
                )
            renderables.append(assump)

        if self.details is not None and not self.details.empty:
            renderables.append(_dataframe_to_rich_table(self.details, title="Details"))

        if self.pairwise is not None and not self.pairwise.empty:
            renderables.append(_dataframe_to_rich_table(self.pairwise, title="Post-hoc"))

        return Panel(Group(*renderables), title=f"[bold]{self.test_name}[/bold]")

    def show(self, console: Console | None = None) -> None:
        """Render this result to a Rich console.

            Parameters
            ----------
            console:
                Rich console to render to. If ``None`` (default), create a console
                that writes to standard output.

            Returns
            -------
            None
                The result is rendered as a side effect; nothing is returned.

        Examples
        --------
        >>> result.show()  # doctest: +SKIP
        """
        (console if console is not None else Console()).print(self)

    # ------------------------------------------------------------------ helpers
    def _fmt_statistic(self) -> str:
        return self._fmt_num(self.statistic)

    @staticmethod
    def _fmt_num(x: Any) -> str:
        if isinstance(x, int):
            return str(x)
        return f"{x:.4g}"

    @staticmethod
    def _fmt_p(p: float | None) -> str:
        if p is None:
            return "N/A"
        if p < 0.0001:
            return "<0.0001"
        return f"{p:.4g}"


# --------------------------------------------------------------------- helpers


def _flatten(d: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten a dict, expanding list/tuple values into indexed keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        elif isinstance(v, list | tuple):
            for i, item in enumerate(v):
                out[f"{key}_{i}"] = item
        else:
            out[key] = v
    return out


def _dataframe_to_rich_table(df: pd.DataFrame, *, title: str = "") -> Table:
    table = Table(show_header=True, header_style="bold", title=title)
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[_fmt_cell(v) for v in row])
    return table


def _fmt_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def to_dataframe(
    results: TestResult | Iterable[TestResult],
    *,
    p_adjust: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Convert one or more :class:`TestResult` objects into a concatenated tidy dataframe.

    Columns are the union of all result columns; missing values are filled with NaN.
    Pass ``p_adjust`` to attach corrected p-values across the provided results.

    Parameters
    ----------
    results:
        One :class:`TestResult` or an iterable of results. An empty iterable
        returns an empty dataframe.
    p_adjust:
        Optional multiple-comparison correction method. When provided, the
        results are corrected on copies using :func:`adjust_results` before
        export. The default is ``None`` (no correction).
    alpha:
        Significance threshold used when ``p_adjust`` is provided. The default
        is ``0.05``. It is ignored when ``p_adjust`` is ``None``.

    Returns
    -------
    pandas.DataFrame
        Concatenated one-row result tables with the union of their columns;
        missing values are filled by pandas with ``NaN`` or ``NA`` as
        appropriate.

    Raises
    ------
    ValueError
        If ``p_adjust`` or ``alpha`` is invalid, or a result contains an
        invalid p-value for correction.

    Examples
    --------
    >>> table = to_dataframe([result_a, result_b], p_adjust="holm")
    """
    if isinstance(results, TestResult):
        results = [results]
    else:
        results = list(results)
    if p_adjust is not None:
        from pystars.corrections import adjust_results

        results = adjust_results(results, method=p_adjust, alpha=alpha)
    frames = [r.to_dataframe() for r in results]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
