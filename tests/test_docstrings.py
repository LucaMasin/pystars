"""Tests for the discoverability of the public PyStars API."""

import inspect

import pystars
from pystars.data import LongData, n_per_group, normalize_data, paired_differences

PUBLIC_FUNCTIONS = [
    getattr(pystars, name) for name in pystars.__all__ if inspect.isfunction(getattr(pystars, name))
]
PUBLIC_FUNCTIONS.extend([normalize_data, paired_differences, n_per_group])
PUBLIC_FUNCTIONS.extend(
    method
    for name, method in inspect.getmembers(LongData, inspect.isfunction)
    if not name.startswith("_")
)
PUBLIC_FUNCTIONS.extend(
    method
    for name, method in inspect.getmembers(pystars.TestResult, inspect.isfunction)
    if not name.startswith("_")
)


def test_public_callables_have_helpful_docstrings():
    """Public callables should expose useful help in notebooks and ``help()``."""
    for function in PUBLIC_FUNCTIONS:
        doc = inspect.getdoc(function)
        assert doc, f"{function.__qualname__} is missing a docstring"
        assert "Parameters" in doc, f"{function.__qualname__} needs a Parameters section"
        assert "Returns" in doc, f"{function.__qualname__} needs a Returns section"
