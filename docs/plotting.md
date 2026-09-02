# Plot annotation

`pystars.annotate_significance(ax, result)` adds significance brackets
(`*`, `**`, `***`, `****`, or `ns`) to an *existing* Matplotlib axes without
re-plotting your data. It inspects only the artists the axes already shows
(bars, points, error bars, dodged cells) and draws `Line2D` brackets and
`Text` labels on top.

Matplotlib is a PyStars runtime dependency because this function operates on
Matplotlib artists. Seaborn is intentionally optional: this function consumes
the artists Seaborn has already drawn and never imports Seaborn itself.

## Two-group result

For a direct two-group result (`TestResult` without a `pairwise` table),
provide the two group names yourself — `TestResult` does not retain the
group labels that fed the test:

```python
import matplotlib.pyplot as plt
import pystars as ps

fig, ax = plt.subplots()
ax.bar(["control", "treated"], [10.0, 12.5])
result = ps.test(df, value="value", group="group")
ps.annotate_significance(ax, result, groups=("control", "treated"))
```

## Auto-dispatched result with post-hoc

When a primary result carries post-hoc comparisons, `annotate_significance`
walks `result.pairwise` and `result.posthoc` recursively and renders every
significant comparison by default:

```python
result = ps.test(df, value="value", group="treatment")
# result.posthoc is a TestResult whose .pairwise DataFrame has columns A, B, p
ps.annotate_significance(ax, result)  # draws all significant brackets
```

To draw every comparison (including non-significant ones), pass
`comparisons="all"`. To draw exactly one pair, pass a sequence of
two-tuples:

```python
ps.annotate_significance(ax, result, comparisons=[("a", "b")])
ps.annotate_significance(ax, result, comparisons="all")
```

## Dodged Seaborn hue plots

Multi-factor PyStars results use composite labels like `"control::drug"`.
Map each composite to a `(x_category, hue_category)` tuple so the resolver
targets a specific dodged cell on the plot:

```python
import seaborn as sns

sns.barplot(data=df, x="genotype", y="value", hue="treatment", ax=ax)
result = ps.test(df, value="value", group=["genotype", "treatment"])
ps.annotate_significance(
    ax,
    result,
    label_map={
        "wt::vehicle": ("wt", "vehicle"),
        "ko::vehicle": ("ko", "vehicle"),
    },
)
```

Scalar categories are anchored to their exact x-tick positions. Tuple mappings
are anchored to the nominal center of the selected dodged cell. On layered
raw-point and summary plots, summary markers and error bars determine the
horizontal position, while all rendered data artists contribute to vertical
clearance.

## Compact-letter display

With `mode="letters"`, draw a compact-letter display above each category
instead of brackets. This mode requires `comparisons="all"` so the full
pairwise graph is available:

```python
ps.annotate_significance(ax, result, mode="letters", comparisons="all")
```

## Format, style, and brackets

```python
ps.annotate_significance(
    ax,
    result,
    mode="pvalue",      # one of "stars" (default), "pvalue", "value", "letters"
    p_decimals=2,        # fixed decimal places for "pvalue" / "value" labels
    bracket="line",     # "line" (default) or "square" (with caps)
    color="black",      # default is "black"; pass any Matplotlib color
    y_offset=6,         # shift the whole annotation stack in display points;
                        # positive moves up, negative moves down
    text_kws={"fontsize": 12, "fontweight": "bold"},
    line_kws={"linewidth": 1.0},
    rc={"font.size": 12},  # scoped rc; does not change global rcParams
)
```

`y_offset` is a signed display-point value (1 point = 1/72 inch, scaled by the
figure DPI). It shifts the entire bracket stack uniformly so spacing between
overlapping brackets is preserved, and it also applies to compact-letter
labels in `mode="letters"`. The visual direction ("up" / "down") is
consistent on linear, log, and inverted axes because the offset is applied in
display space.

By default, p-values use four significant figures, with scientific notation
below `1e-4`. Set `p_decimals=N` for fixed decimal places in `pvalue` and
`value` modes; both render the number without a `p=` prefix. Values that would
round to zero use a `<...` threshold instead. This option is ignored in
`stars` and `letters` modes.

The function returns the same `ax` you passed in, so calls chain with other
Matplotlib operations.
