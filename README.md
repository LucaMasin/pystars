# PyStars

PyStars is a Python library for automating significance testing for common biological and life sciences data. It provides a simple interface to perform statistical tests on pandas dataframes.

## Development

### Features

Minimal features required:

- [ ] Implement a general function that walks through a flowchart and performs the appropriate test based on the data type and distribution. See [this](Biological-Data-Test-Flowchart.md) document for a flowchart.
- [ ] Offer the ability to perform specific tests directly, so they should have a user-friendly interface.
- [ ] Expose normality and variance tests to the user, so they can check assumptions before performing a test.
- [ ] Offer the ability to take in both long and wide data formats, and convert between them as needed.
- [ ] Export the results of the tests as a pandas dataframe that is both friendly for the user and as an input for programmatic use.
- [ ] Offer rich printing of the results, including a summary of the test performed, the assumptions checked, and the results of the test.
