# Plot Significance Annotations: Implementation Plan

## Goal And Scope

Add a production-quality `pystars.annotate_significance` function that takes
an existing, already-rendered Matplotlib `Axes` and a PyStars `TestResult`, then
adds statistical annotations to that plot.

This feature is an annotation layer, not a plotting API. It must not call a
Matplotlib or Seaborn plotting function, create a figure, call `show()`, save a
file, inspect raw data, recompute a test, or infer a comparison that is absent
from the supplied statistical result. It may only:

- inspect the existing axes, its ticks, legend, and rendered artists;
- add bracket `Line2D` and label `Text` artists; and
- adjust the existing y limits enough to keep those annotation artists visible.

The public scope is vertical categorical bar, strip, and swarm plots drawn by
Matplotlib or by Seaborn. Seaborn support means operating on the Matplotlib
artists Seaborn has already created; `plotting.py` must not import Seaborn.

Horizontal categorical charts, `FacetGrid` input, polar and 3-D axes,
automatic palette selection, arbitrary custom artists, raw-data inspection,
and automated statistical testing are deliberately out of scope. Fail clearly
rather than placing an annotation at an uncertain position.

## Files To Change

| File | Change |
| --- | --- |
| `src/pystars/plotting.py` | New annotation API and private implementation helpers. |
| `src/pystars/__init__.py` | Re-export `annotate_significance` and add it to `__all__`. |
| `tests/test_plotting.py` | New focused, Agg-backed plotting tests. |
| `pyproject.toml` | Add direct runtime dependency `matplotlib` and development dependency `seaborn`. |
| `uv.lock` | Synchronize after dependency changes. |
| `README.md` | Add user-facing plotting documentation and runnable examples. |
| `AGENTS.md` | Record the module, tests, extraction contract, and resolver conventions. |

No change is required to `TestResult`, dispatcher output, statistics modules,
or data normalization. The existing `TestResult` contract already contains the
needed direct p-values, pairwise tables, and recursive post-hoc results.

## Public API

Implement and export this signature:

```python
def annotate_significance(
    ax: matplotlib.axes.Axes,
    result: TestResult,
    *,
    comparison_table: pandas.DataFrame | None = None,
    comparisons: Literal["significant", "all"] | Sequence[tuple[object, object]] = "significant",
    groups: tuple[object, object] | None = None,
    label_map: Mapping[object, object] | None = None,
    mode: Literal["stars", "pvalue", "value", "letters"] = "stars",
    bracket: Literal["line", "square"] = "square",
    alpha: float = 0.05,
    p_column: str = "auto",
    color: str | None = None,
    rc: Mapping[str, Any] | None = None,
    line_kws: Mapping[str, Any] | None = None,
    text_kws: Mapping[str, Any] | None = None,
) -> matplotlib.axes.Axes:
```

`plotting.py` will import Matplotlib directly because it is a public runtime
API. Its type and collection imports will use public Matplotlib classes and
APIs only. It will use `pandas` and the existing `TestResult`, plus standard
library `collections.abc`, `dataclasses`, `math`, and `typing` facilities as
needed.

The function always returns the exact same `ax` object. With no selected
comparisons, it returns the unchanged axes without adding artists or changing
limits. Letter mode is the exception to pairwise bracket rendering: it adds
only compact-letter text labels.

The initial public API will not expose a `positions` escape hatch. The declared
bar, strip, swarm, and Seaborn-dodge scope can be resolved from public artist
geometry. Adding an override for arbitrary custom artists would enlarge the API
without being necessary for this version.

## Module Structure

Keep the module intentionally narrow. Use a small immutable internal
comparison record, for example `_Comparison(left, right, p_value, source,
ordinal)`, and private helpers with these responsibilities:

| Helper | Responsibility |
| --- | --- |
| `_validate_axes_and_options` | Validate axes, modes, alpha, mappings, and style mappings. |
| `_extract_comparisons` | Collect an override table or recursive `TestResult` pairwise tables. |
| `_normalise_pairwise_table` | Validate schema and convert one table into comparison records. |
| `_select_comparisons` | Apply significant/all/requested-pair selection before layout work. |
| `_resolve_positions` | Resolve mapped statistical groups to rendered x centers and y extents. |
| `_collect_bar_geometry` | Read visible vertical `BarContainer` rectangles deliberately. |
| `_collect_point_geometry` | Read visible `PathCollection` offsets deliberately. |
| `_layout_brackets` | Assign compact non-overlapping levels in display space. |
| `_format_label` | Produce stars, `p=<value>`, or raw formatted p-values. |
| `_compact_letters` | Build deterministic valid compact-letter assignments. |
| `_expand_y_headroom` | Preserve axis orientation while making annotations visible. |

Avoid a plotting class, configuration object, Seaborn adapter, or any broader
plot abstraction. The helpers may share small private geometry records where
that makes bar and point handling explicit, but should otherwise remain in the
single module.

## Validation And Error Contract

Validate before adding any artists whenever possible. Errors must identify the
invalid input and explain the corrective input the caller can provide.

### Axes And Basic Options

- Require `ax` to be a Matplotlib `Axes` instance.
- Require a rectilinear 2-D Cartesian axes. Reject polar and 3-D axes directly.
- Reject a recognized horizontal `BarContainer` and explain that only vertical
  categorical plots are supported.
- Require usable x category ticks when resolving categorical labels. A failure
  to resolve labels through x ticks must state that the plot must be vertical
  categorical and that `label_map` can map statistical identifiers to displayed
  categories.
- Require `mode` to be exactly one of `"stars"`, `"pvalue"`, `"value"`, or
  `"letters"`.
- Require `bracket` to be exactly `"line"` or `"square"`.
- Require `alpha` to be a finite real number strictly between zero and one;
  reject booleans.
- Require `p_column` to be `"auto"` or a non-empty column-name string.
- Require `label_map`, `rc`, `line_kws`, and `text_kws`, when supplied, to be
  mappings. Copy them before augmenting defaults so caller-owned mappings are
  never mutated.

### Annotation Style Inputs

- Build default line and text styles inside `matplotlib.rc_context(rc=rc)`.
- If `color` is supplied, make it the default for both line and text artists.
  A `color` in `line_kws` or `text_kws` takes precedence for that artist type.
- Use compact defaults suitable for a Prism-like annotation, an annotation
  z-order above existing artists, and `clip_on=False`.
- Permit visual Matplotlib properties such as `linewidth`, `linestyle`,
  `fontsize`, `fontweight`, and `color`.
- Reject placement-changing or reserved properties such as `x`, `y`, `s`,
  `transform`, `data`, alignment values managed by the API, and internal
  annotation identifiers. This prevents a style mapping from moving labels or
  brackets away from their computed geometry.
- Let Matplotlib validate the remaining forwarded properties. Its error remains
  useful for unsupported line or text style names.
- Use `rc_context` only around annotation creation and measurement. Never
  assign to global `matplotlib.rcParams`; annotations created outside the call
  must observe the previous global settings.

## Comparison Extraction And P-Value Validation

### Sources And Precedence

The required `result` is a `TestResult`. A caller-supplied
`comparison_table` takes absolute precedence: when present, it is the only
comparison source and no result-level table or post-hoc result is considered.

Without an override, walk the result tree depth-first in this deterministic
order:

1. Process `result.pairwise` first when it is present.
2. Then process `result.posthoc`.
3. A post-hoc value may be one `TestResult` or a list of `TestResult` objects;
   process a list in list order and recurse into each result.
4. Track object identities during traversal and reject a cycle rather than
   recursing indefinitely.

`pairwise` must be a pandas `DataFrame` when present. A non-`TestResult`
post-hoc value, a malformed list item, or a non-DataFrame `pairwise` value is
an invalid result contract and should raise rather than be silently skipped.

An empty, valid pairwise table is a valid source with no comparisons. It does
not cause the function to synthesize a direct pair from `groups`; direct
two-group fallback applies only when no usable pairwise table is found at all.

### Pairwise Table Rules

For every selected source table:

- Require unique column names plus columns `A` and `B`.
- With `p_column="auto"`, use `p_adjusted` for that table when the column
  exists and contains at least one non-null value; otherwise use `p`.
- Do not choose p-values row by row. A table using adjusted values must use
  that column for every row so adjusted and unadjusted families are never mixed.
- With an explicit `p_column`, require that exact column in every collected
  table.
- Convert the selected values numerically with a raising conversion. Reject
  nonnumeric values, missing values, infinities, and values outside `[0, 1]`.
- Require non-null, hashable `A` and `B` labels and reject self-comparisons.
- Treat `(A, B)` and `(B, A)` as the same unordered pair. Reject duplicates
  within one table or across recursively discovered tables, even when their
  p-values happen to agree. The caller can pass `comparison_table` to select
  one post-hoc procedure explicitly.

The normalizer will retain an ordinal that reflects source and row order. It is
used only as a deterministic tie breaker; labels need not be sortable or be
strings.

### Direct Two-Group Fallback

If no pairwise table exists, `groups=(left, right)` is required. Validate that
it is exactly two distinct, hashable labels. Use `result.p_adjusted` when it is
numeric, finite, and in `[0, 1]`; otherwise use a valid `result.p_value`.
Raise if neither scalar p-value is usable.

The error must explain why this is required: direct two-group `TestResult`
objects retain a p-value but intentionally do not retain their source-group
names. The function cannot infer those names from the result or raw data.

### Comparison Selection

Apply selection after all p-values are validated and before label mapping,
artist inspection, or layout:

- `comparisons="significant"` selects records where `p <= alpha`.
- `comparisons="all"` selects every available record, including
  non-significant records.
- A sequence of two-item pairs selects exactly the named unordered pairs,
  independent of p-value. Every requested pair must appear exactly once in the
  available records; otherwise raise a helpful error identifying that pair.
- An empty result from normal selection is not an error. Return `ax` unchanged.
- Reject any other `comparisons` value. In particular, reject a bare string
  other than `"significant"` or `"all"` rather than treating its characters as
  requested pairs.

## Statistical Labels And Plot Labels

Statistics and displayed categorical labels can differ. Resolve every selected
statistical label through `label_map` first, with an unmapped label retaining
its identity. The resolved visual labels have two supported shapes:

| Visual label shape | Meaning |
| --- | --- |
| Scalar | Match `str(value)` to a unique displayed x tick label. |
| Two-item tuple `(x_category, hue_category)` | Resolve `x_category` against x ticks and `hue_category` against legend text for a dodged plot. |

All resolved labels for one call must be consistently scalar or consistently
two-item tuples. Reject malformed tuples, a mixture of scalar and tuple target
labels, duplicate visual mappings, or a comparison whose two statistics map to
one plot location.

For example, two-factor PyStars groups are normalized as composite strings such
as `"control::drug"`. A tuple-valued mapping makes the corresponding dodged
plot position explicit:

```python
label_map = {
    "control::vehicle": ("control", "vehicle"),
    "control::drug": ("control", "drug"),
}
```

When scalar resolution fails, name the missing statistical and visual label,
list displayed x tick labels, and suggest `label_map`. When tuple resolution
fails, say whether the category or hue is missing and show the required tuple
mapping shape.

## Rendered-Artist Position Resolution

### General Resolver Procedure

The resolver works only with the existing axes. It will not call a plotting
function or depend on Seaborn internals.

1. Call `ax.figure.canvas.draw()` before inspection. This is necessary because
   Seaborn swarm placement is finalized during drawing.
2. Read visible x ticks and visible tick-label text with public axes methods.
   Require unique displayed tick text so scalar category matching is unambiguous.
3. Ignore annotation artists created by this module, marked with a private
   stable `gid`, when collecting plot geometry. This prevents a later call from
   treating an earlier bracket as data.
4. Collect bar, point, and relevant error-bar geometry using public artist
   APIs. Store x center plus visual top in display coordinates.
5. Resolve each mapped target to exactly one x center. Resolve relevant visual
   tops as well, so a bracket begins above actual rendered data rather than a
   fixed data y value.

The resolver uses x tick locations as category anchors, not a guessed unit
width. It does not assume any Seaborn ordering convention or fixed hue dodge
width. It supports more than two hue levels provided the rendered legend colors
and artist geometry distinguish them.

### Bars

Inspect visible vertical `BarContainer` instances and each `Rectangle` patch:

- Reject horizontal containers.
- Compute the x center as `rectangle.get_x() + rectangle.get_width() / 2`.
- Assign a rectangle to the nearest category tick only when it lies within the
  local half-category spacing. Otherwise leave it out rather than guessing.
- Transform both y bounds, `y` and `y + height`, to display coordinates and
  retain the visually highest bound. This works for negative bars, log-scaled
  data that Matplotlib can render, and inverted y axes.
- For dodged plots, compare the visible bar face color, falling back to edge
  color for transparent faces, to the legend hue color.

### Strip And Swarm Points

Inspect visible non-empty `PathCollection` instances:

- Read positions using public `get_offsets()` after the canvas draw.
- Associate a collection with a category using its center or median x offset
  and the nearest category tick under the same spacing tolerance.
- Use the median x offset as the rendered center and the maximum transformed
  y offset as its visual top.
- For dodged plots, require one effective collection color and match it to one
  legend hue color. Deliberately reject multicolored collections because they
  cannot be attributed to one hue safely.

When a bar and an overlaid point collection represent the same group, prefer
the bar center for the group position but combine relevant geometry for the
top. If their candidate centers materially disagree, raise rather than guess.

### Error Bars And Other Relevant Heights

To meet the requirement that annotations begin above the highest relevant
plotted artist, include vertical error-bar extents associated with a resolved
group center. Read vertical `Line2D` and `LineCollection` error-bar segments
through their public geometry APIs only when their x geometry identifies them
as belonging to that group. Do not treat arbitrary fitted lines or unrelated
artists as categorical data.

For each resolved group, combine bar, point, and associated error-bar tops.
This keeps annotations above bars, raw points, and Seaborn confidence intervals
without modifying existing artists.

### Hue And Legend Resolution

Tuple labels are the explicit signal that hue resolution is required:

1. Resolve the tuple's first element against x tick text.
2. Require a legend and resolve the second element against legend text.
3. Obtain each legend handle's public visible color. Use patch face or edge
   color, then `Line2D` marker face/edge or line color as applicable.
4. Normalize candidate colors to RGBA and require a unique legend entry for a
   hue color. A repeated palette color is ambiguous and must raise.
5. Match the category and hue color to one rendered bar center or, when no bar
   exists, one point-collection center.

Raise an actionable error when a plotted cell has no artist, when a legend was
disabled, when a hue label is unknown, when reused colors are ambiguous, or
when multiple materially different candidates match. The error should mention
the unresolved target and show the `(x_category, hue_category)` `label_map`
shape needed to correct it.

## Label Formatting

Formatting must be deterministic and documented in the README.

- `mode="stars"` uses fixed conventional thresholds, independent of a custom
  selection `alpha`: `****` for `p <= 0.0001`, `***` for `p <= 0.001`, `**`
  for `p <= 0.01`, `*` for `p <= 0.05`, and `ns` otherwise.
- `mode="pvalue"` renders `p=<formatted value>`.
- `mode="value"` renders only `<formatted value>`.
- Use the same formatter in both p-value modes: scientific notation with two
  significant decimal places for values below `1e-4`, otherwise a compact
  four-significant-figure representation.
- `ns` appears for non-significant values when `comparisons="all"` or when a
  caller explicitly selects a non-significant pair. It is absent by default
  because default selection filters it out.

## Bracket Layout And Headroom

### Bracket Artists

Create one `Line2D` and one `Text` per selected non-letter comparison:

- A `line` bracket is one horizontal segment between the two resolved centers.
- A `square` bracket is one polyline with x positions
  `[left, left, right, right]`, vertical end caps, and a horizontal top.
- The label is horizontally centered over the span, positioned just above the
  bracket top, centered horizontally, and visually bottom-aligned.
- Mark artists with the module `gid`, use a high z-order, and keep clipping off.

### Level Assignment

Use display-space y geometry for all vertical offsets. This avoids fixed data
units and keeps annotations sensible over small or large ranges, log axes, and
inverted y axes.

1. For each comparison, form a closed horizontal interval from its two x
   centers and calculate the highest relevant visual top under that span.
2. Sort intervals by left x, right x, and original comparison ordinal.
3. Greedily assign the lowest level whose already assigned closed intervals do
   not overlap. Intervals touching at an endpoint count as overlapping.
4. Use a point/pixel base gap, square-cap height, label gap, and level step
   calculated from the renderer and effective text size. Disjoint spans can use
   the same level; intersecting spans cannot.
5. Convert every final display y coordinate back through
   `ax.transData.inverted()` before creating or updating the artists.

The common display-space machinery provides compact brackets and stable visual
spacing even when y values are small, large, logarithmic, negative, or shown
on an inverted axis.

### Y-Limit Expansion

Do not use a hard-coded data y coordinate, fixed data-unit margin, or a sorted
`ylim` assumption. Instead:

1. Draw and measure annotation text and bracket extents with the renderer.
2. Identify the y-limit endpoint that maps to the visual top of the axes.
3. Expand only that outward-facing endpoint enough to include the measured
   bracket/text extent plus a small display-space margin.
4. Preserve the original endpoint ordering, including an inverted y axis.
5. Redraw and, if scaling changed annotation geometry, recompute display-space
   positions and remeasure until the annotations fit. Keep this bounded and
   deterministic.

Existing artist data, styles, and positions remain untouched. The only allowed
axes mutation is the outward y-limit adjustment needed for annotation headroom.

## Compact-Letter Display

`mode="letters"` draws no brackets. It requires `comparisons="all"`; reject
`"significant"` and an explicit comparison sequence because a valid compact
letter display needs all relationships.

Use every pairwise relationship among the groups represented by the available
comparison table(s). Because the function cannot inspect raw data, these are
the plotted groups it can know. Validate a complete undirected pairwise graph:
for `n` groups, require `n * (n - 1) / 2` distinct pairs and name missing pairs
in the error.

Build a valid deterministic display with an insert-and-absorb procedure:

1. Order groups by resolved visual position, with source ordinal as a stable
   tie breaker.
2. Mark a pair significant when `p <= alpha`.
3. Start with one letter column containing all groups.
4. For each significant pair, split every current column containing both
   groups into two columns, each omitting one endpoint.
5. Remove duplicate columns and columns that are proper subsets of another
   retained column.
6. Sort retained columns deterministically and assign letter names `a` through
   `z`, then `aa`, `ab`, and onward.
7. Give each group the concatenation of all letters from columns containing it.

This represents non-transitive relationships correctly. If A and B are not
significantly different, B and C are not significantly different, and A and C
are significantly different, a valid result is A=`a`, B=`ab`, C=`b`. Every
non-significant pair shares a letter and no significant pair shares one.

Place each letter label just above that group's rendered top with the same
display-space spacing/headroom mechanism as brackets. Add only `Text` artists
in this mode.

## Expected Errors

Use precise `ValueError` messages for user-correctable input. At minimum cover
these cases:

| Situation | Error guidance |
| --- | --- |
| No pairwise result and no `groups` | Explain the direct two-group group-name requirement and offer `comparison_table`. |
| Invalid comparison table | Require a DataFrame with `A`, `B`, and the selected p-value column. |
| Explicit missing p column | Name the requested column and the failing source table. |
| Invalid p values | Name the selected column and require finite numeric values in `[0, 1]`. |
| Duplicate unordered pair | Name the pair and recommend `comparison_table` to select one procedure. |
| Unavailable requested pair | Name the requested pair and state that it is absent from available comparisons. |
| Invalid or duplicate visual mapping | Identify the mapping and require unique scalar labels or unique two-item tuples. |
| Unknown category or hue | List displayed tick or legend labels and suggest the correct `label_map` shape. |
| Missing or ambiguous dodged artist | Identify the target, require a rendered cell, legend, distinct palette colors, or tuple mapping. |
| Unsupported axes or horizontal plot | State that only vertical 2-D Cartesian categorical axes are supported. |
| Letter mode selection | State that `mode="letters"` requires `comparisons="all"`. |
| Incomplete letter-mode pairs | State the required complete relationship count and list missing pairs. |

## Test-First Plan

Create `tests/test_plotting.py` before adding `plotting.py`. It must call
`matplotlib.use("Agg")` before importing `matplotlib.pyplot`, never open a
window, and close created figures in fixtures or test cleanup. Use fixed data
and seeds. Test observable `Line2D` and `Text` geometry and content, never
pixel snapshots or irrelevant private Matplotlib internals.

### Core Tests

1. Construct a direct two-group `TestResult`, draw a pre-made Matplotlib bar
   or strip plot, call with `groups`, and assert that the returned object is the
   same axes, a square bracket uses the two category centers, and its text is
   the expected star label.
2. Construct a dispatcher-like primary result with a `posthoc` `TestResult`
   whose `pairwise` table contains multiple pairs. Assert recursive discovery
   and rendering of all significant pairs under default selection.
3. Give the result a conflicting pairwise table, supply a replacement
   `comparison_table`, and prove the replacement wins. Include `p` and
   `p_adjusted` values that lead to different selected pairs to prove automatic
   adjusted-p preference.
4. Use statistical group names different from x tick labels and assert that
   scalar `label_map` produces bracket x endpoints at the displayed categories.
5. Verify a direct two-group result without `groups` fails with an error that
   explains the missing pair identities.

### Artist-Resolver Tests

1. Draw a Matplotlib vertical bar plot and a categorical point plot, then
   assert scalar labels resolve to existing x tick positions.
2. Draw a Seaborn bar plot with `hue`, `dodge=True`, and at least three hue
   levels. Map composite statistical labels to category/hue tuples and assert
   bracket endpoints equal the centers of distinct rendered dodge artists.
3. Repeat tuple-resolution assertions for Seaborn strip and swarm plots with
   hue dodging. Use structural line/text coordinate checks and fixed seeds, not
   image comparisons.
4. Assert failures for a missing legend, an unknown x category, an unknown hue,
   an empty category-hue cell, duplicate visual mapping targets, and ambiguous
   palette colors when they prevent a unique position.
5. Assert that any annotations from an earlier call are ignored as data
   geometry on a subsequent call.

### Selection, Formatting, And Layout Tests

1. Parametrize star boundaries: just above `.05` is `ns`; `.05`, `.01`,
   `.001`, and `.0001` produce `*`, `**`, `***`, and `****` respectively.
2. Assert default significant-only filtering, `comparisons="all"` inclusion of
   `ns`, and explicit selection of a non-significant pair.
3. Assert the documented p-value and raw-value formatting, including a very
   small value rendered in scientific notation.
4. Assert `line` produces a horizontal two-endpoint connector while `square`
   produces a capped four-vertex polyline.
5. Assert `color` applies to both line and text unless a style mapping overrides
   one of them; also assert supported line/text style properties are forwarded.
6. Build overlapping and disjoint pairs across four categories. Assert
   overlapping spans have distinct top levels while disjoint spans share one.
7. Exercise a logarithmic y axis and an inverted y axis. Assert annotations are
   outward from rendered data, remain within expanded limits, and preserve the
   original y-limit direction.
8. Assert existing data artists retain their geometry and only y limits plus
   module annotation artists change.

### Compact-Letter And Validation Tests

1. Use a non-transitive three-group relation: A-B and B-C non-significant,
   A-C significant. Parse displayed letters and assert the two non-significant
   pairs share letters while the significant pair shares none.
2. Supply one missing pair in letter mode and assert the error identifies its
   absence.
3. Assert letter mode rejects both default `"significant"` selection and an
   explicit sequence.
4. Parametrize invalid table schemas, nonnumeric p-values, NaN, infinity,
   p-values outside `[0, 1]`, invalid p-column names, self pairs, and duplicate
   unordered pairs.
5. Assert invalid `mode`, `bracket`, `alpha`, `groups`, `label_map`, and
   unsupported axes raise useful errors.
6. Capture a global Matplotlib rcParam, call with `rc` changing font or line
   settings, then assert the global value is unchanged while created annotation
   artists use the scoped setting.

## Documentation Plan

Add a concise README section titled for plot annotations after the result or
post-hoc documentation. State that Matplotlib is a PyStars runtime dependency
and that Seaborn remains optional because the function consumes the Axes it has
already drawn.

Include runnable examples in this order:

1. A two-group Matplotlib bar or strip plot drawn before calling
   `annotate_significance`, using `groups=("control", "treated")` for a direct
   test result.
2. An auto-dispatched multi-group result with a post-hoc table, demonstrating
   that `result.posthoc.pairwise` is discovered without manually extracting it.
3. A Seaborn dodged hue bar or strip plot, using composite statistical labels
   mapped to `(x_category, hue_category)` tuples.
4. A complete pairwise result rendered with `mode="letters"` and
   `comparisons="all"`.

Document accepted comparison sources in precedence order, the direct two-group
`groups` requirement, significant/all/explicit selection behavior, p-column
selection, label-map semantics, all modes, square versus line brackets, p-value
formatting, compact-letter completeness, and unsupported plot types.

Update `AGENTS.md` rather than the README with contributor-facing details:
add `plotting.py` and `test_plotting.py` to the architecture listing, record
recursive comparison extraction priority, the direct result fallback rule, the
tuple label-map convention, canvas drawing before swarm inspection, and the
rule to fail on ambiguous rendered geometry rather than guess.

## Dependencies And Verification

Use the repository-required `uv` workflow:

```bash
uv add matplotlib
uv add --dev seaborn
```

Matplotlib must be a direct runtime dependency because the public function is
typed around and creates Matplotlib artists. Seaborn must only be a development
dependency for integration tests; it remains an optional user dependency and is
never imported by `pystars.plotting`. Commit the resulting `pyproject.toml` and
`uv.lock` changes with the implementation.

Follow TDD: write focused tests first, run the targeted plotting tests while
developing, then run the required full checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

After formatting, inspect `git status --short`, `git diff --check`, and the
full diff to ensure changes are limited to the planned module, export, tests,
dependencies, lockfile, and documentation. Do not commit unless explicitly
requested.
