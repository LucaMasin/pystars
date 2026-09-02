# PyStars

Automated significance testing for biological and life-sciences data.

PyStars takes a pandas dataframe, checks the assumptions, picks the right
statistical test, and returns results you can print, annotate on plots, and
export — often in a single call.

## Installation

```bash
uv add pystars
```

## Quick start

```python
import numpy as np
import pandas as pd
import pystars

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "genotype": ["wt"] * 30 + ["mut"] * 30,
    "length": np.concatenate([rng.normal(10.0, 1.0, 30), rng.normal(11.2, 1.0, 30)]),
})

# PyStars checks assumptions and picks the appropriate test.
result = pystars.test(df, value="length", group="genotype")

result.show()          # rich terminal summary
result.to_dataframe()  # tidy one-row dataframe
```

For full control, call any test directly (`ttest`, `anova`, `kruskal`, ...)
with the same `TestResult` return type.

## Documentation

- [User guide](docs/user-guide.md) — dispatcher, direct tests, assumptions, post-hoc, data formats, export, corrections.
- [Plot annotation](docs/plotting.md) — significance brackets and compact letters on Matplotlib axes.
- [Test-selection flowchart](docs/flowchart.md) — the logic the dispatcher follows.
- [Examples](examples/) — guided tutorials, including one on real data.

In Jupyter, `pystars.test?` opens the help panel for any public function.

## Development

See [AGENTS.md](AGENTS.md) for contributor guidance and the
[Conda + uv environment guide](docs/conda-uv-environment.md) for environment
setup.
