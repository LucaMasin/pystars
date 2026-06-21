"""PyStars: automated significance testing for biological and life sciences data."""

from pystars.assumptions import check_equal_variance, check_normality
from pystars.dispatcher import test
from pystars.posthoc import posthoc_dunn, posthoc_games_howell, posthoc_tukey
from pystars.result import TestResult, to_dataframe
from pystars.tests_continuous import (
    anova,
    anova_twoway,
    kruskal,
    mannwhitney,
    ttest,
    wilcoxon,
)

__all__ = [
    "TestResult",
    "to_dataframe",
    "test",
    "ttest",
    "mannwhitney",
    "wilcoxon",
    "anova",
    "kruskal",
    "anova_twoway",
    "check_normality",
    "check_equal_variance",
    "posthoc_tukey",
    "posthoc_games_howell",
    "posthoc_dunn",
]
