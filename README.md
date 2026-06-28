# PyStars

PyStars is a Python library for automating significance testing for common biological and life sciences data. It provides a simple interface to perform statistical tests on pandas dataframes.

## Installation

```bash
uv add pystars
```

PyStars depends on `pandas`, `pingouin`, `scikit-posthocs`, and `rich`.

## Quick start

```python
import pystars
import pandas as pd

df = pd.DataFrame(
    {
        "group": ["wt"] * 30 + ["mut"] * 30,
        "length": [10.1, 9.8, ...] ,  # your measurements
    }
)

# Let PyStars walk the flowchart and pick the right test.
result = pystars.test(df, value="length", group="group")
result.show()                      # rich terminal summary
result.to_dataframe()              # tidy one-row dataframe
print(result.summary())            # plain-text summary
```

## Examples

- [`examples/tutorial.ipynb`](examples/tutorial.ipynb) — guided tour of the API with synthetic data.
- [`examples/iris_real_data.ipynb`](examples/iris_real_data.ipynb) — real-world morphology analysis using Fisher's Iris dataset from UCI.

## Features

### Auto-selecting dispatcher

`pystars.test(...)` walks the decision flowchart described in
[`Biological-Data-Test-Flowchart.md`](Biological-Data-Test-Flowchart.md). Based
on the number of groups, pairing, normality, and equal-variance assumptions it
selects and runs the appropriate test:

| Data shape | Assumptions | Selected test |
|---|---|---|
| 2 groups, independent | normal + equal var | Student's t-test |
| 2 groups, independent | normal + unequal var | Welch's t-test |
| 2 groups, independent | non-normal / small n | Mann-Whitney U |
| 2 groups, paired | normal differences | Paired t-test |
| 2 groups, paired | non-normal differences | Wilcoxon signed-rank |
| >2 groups, one factor | normal + equal var | One-way ANOVA + Tukey HSD |
| >2 groups, one factor | normal + unequal var | Welch's ANOVA + Games-Howell |
| >2 groups, one factor | non-normal | Kruskal-Wallis + Dunn's test |
| >=2 factors | — | Two-way ANOVA (interaction reported) |

Post-hoc tests are run automatically when an ANOVA is significant
(`auto_posthoc=True`, the default). Set `auto_posthoc=False` to suppress them.

```python
result = pystars.test(df, value="length", group="group", auto_posthoc=False)
```

### Direct test functions

Each test is also available directly with the same `TestResult` return type:

```python
pystars.ttest(df, value="length", group="group", welch=True)
pystars.ttest(df, value="length", group="group", subject="animal", paired=True)
pystars.mannwhitney(df, value="length", group="group")
pystars.wilcoxon(df, value="length", group="group", subject="animal")
pystars.anova(df, value="length", group="group")              # one-way
pystars.anova(df, value="length", group="group", welch=True)  # Welch
pystars.kruskal(df, value="length", group="group")
pystars.anova_twoway(df, value="length", group=["genotype", "time"])
```

### Assumption checks

Normality and equal-variance tests are exposed so you can check assumptions on
their own. The dispatcher uses the same defaults: Shapiro-Wilk for normality
(per group, or on paired differences) and Levene's test (median-centered) for
equal variance, both at `alpha=0.05`.

```python
pystars.check_normality(df, value="length", group="group")
pystars.check_normality(df, value="length", group="group", subject="animal", paired=True)
pystars.check_equal_variance(df, value="length", group="group")
```

### Post-hoc tests

```python
pystars.posthoc_tukey(df, value="length", group="group")
pystars.posthoc_games_howell(df, value="length", group="group")
pystars.posthoc_dunn(df, value="length", group="group", p_adjust="holm")
```

### Long and wide data formats

PyStars accepts both long (tidy) and wide formats. The format is declared
explicitly — PyStars never guesses the schema.

```python
# Long format (default): one row per observation
pystars.test(df, value="length", group="group", subject="animal", format="long")

# Wide format: one column per group
pystars.test(df, format="wide", groups=["wt", "mut"], subject_index="animal")
```

### Result export

Every test returns a `TestResult` object that can be converted to a tidy
one-row pandas dataframe (suitable both for inspection and as input for
programmatic pipelines):

```python
result.to_dataframe()
#   test            statistic  p_value  cohen_d  CI95%_0  CI95%_1  normality_p  equal_variance_p  posthoc
# 0 Welch's t-test      2.45     0.025     0.80     0.20     1.40       0.453             0.121      NaN
```

Combine several results into one table:

```python
pystars.to_dataframe([result1, result2, result3])
```

### Multiple-comparison corrections

When you run many primary tests, correct their p-values with Holm, Bonferroni,
Sidak, Benjamini-Hochberg FDR, or Benjamini-Yekutieli FDR:

```python
results = [pystars.test(df, value=feature, group="group") for feature in features]

# Attach corrected p-values to TestResult objects.
adjusted = pystars.adjust_results(results, method="fdr_bh")

# Or export directly to a dataframe with correction columns.
table = pystars.to_dataframe(results, p_adjust="fdr_bh")
```

Use `pystars.adjust_pairwise(...)` to apply the same correction helpers to an
arbitrary pairwise comparison table with a p-value column.

### Rich printing

`result.show()` renders a rich panel with the test summary, the assumptions
checked (with verdicts), and the post-hoc pairwise table when present.

```python
result.show()
```

## Development

### Features

Minimal features required:

- [x] Implement a general function that walks through a flowchart and performs the appropriate test based on the data type and distribution. See [this](Biological-Data-Test-Flowchart.md) document for a flowchart.
- [x] Offer the ability to perform specific tests directly, so they should have a user-friendly interface.
- [x] Expose normality and variance tests to the user, so they can check assumptions before performing a test.
- [x] Offer the ability to take in both long and wide data formats, and convert between them as needed.
- [x] Export the results of the tests as a pandas dataframe that is both friendly for the user and as an input for programmatic use.
- [x] Offer rich printing of the results, including a summary of the test performed, the assumptions checked, and the results of the test.
- [x] Offer multiple-comparison correction across batches of tests or arbitrary pairwise tables.

### Scope

Phase 1 covers continuous-data tests (t-test, Mann-Whitney, Wilcoxon, one-way /
Welch ANOVA, Kruskal-Wallis, two-way ANOVA) with post-hoc comparisons. Counts
(Poisson / negative binomial), proportions (Fisher / logistic / beta-binomial),
survival analysis and mixed-effects models are planned for later phases. See
[`AGENTS.md`](AGENTS.md) for the roadmap and contributor guidance.
