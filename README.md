## pl_compare: Compare and find the differences between two Polars DataFrames. 

- Get statistical summaries and/or examples and/or a boolean to indicate:
  - Schema differences
  - Row differences
  - Value differences
- Easily works for Pandas dataframes and other tabular data formats with conversion using Apache arrow 
- View differences as a text report
- Get differences as a Polars LazyFram or DataFrame.
- Use LazyFrames for larger than memory comparisons
- Specify the equality calculation that is used to dermine value differences

### Installation

```zsh
pip install pl_compare
```

### Examples

<details>
<summary>Return booleans to check for schema, row and value differences</summary>

```python
import polars as pl
from pl_compare import compare

base_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1, 6, 3],
        "Example2": ["1", "2", "3"],
    }
)
compare_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1, 2, 3],
        "Example2": [1, 2, 3],
        "Example3": [1, 2, 3],
    },
)

compare_result = compare(["ID"], base_df, compare_df)
print("is_schema_unequal:", compare_result.is_schema_unequal())
print("is_rows_unequal:", compare_result.is_rows_unequal())
print("is_values_unequal:", compare_result.is_values_unequal())
```
output:
```bash
is_schema_unequal: True
is_rows_unequal: True
is_values_unequal: True
```
</details>

<details>
<summary>Schema differences summary</summary>

```python
import polars as pl
from pl_compare import compare

base_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1, 6, 3],
        "Example2": ["1", "2", "3"],
    }
)
compare_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1, 2, 3],
        "Example2": [1, 2, 3],
        "Example3": [1, 2, 3],
    },
)

compare_result = compare(["ID"], base_df, compare_df)
print(compare_result.schema_differences_summary())
```
output:
```bash
shape: (6, 2)
┌─────────────────────────────────┬───────┐
│ Statistic                       ┆ Count │
│ ---                             ┆ ---   │
│ str                             ┆ i64   │
╞═════════════════════════════════╪═══════╡
│ Columns in base                 ┆ 1     │
│ Columns in compare              ┆ 1     │
│ Columns in base and compare     ┆ 3     │
│ Columns only in base            ┆ 0     │
│ Columns only in compare         ┆ 1     │
│ Columns with schema differences ┆ 1     │
└─────────────────────────────────┴───────┘
```
</details>

<details>
<summary>Schema differences details</summary>

```python
import polars as pl
from pl_compare import compare

base_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1, 6, 3],
        "Example2": ["1", "2", "3"],
    }
)
compare_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1, 2, 3],
        "Example2": [1, 2, 3],
        "Example3": [1, 2, 3],
    },
)

compare_result = compare(["ID"], base_df, compare_df)
print(compare_result.schema_differences_summary())
```
output:
```bash
shape: (2, 3)
┌──────────┬─────────────┬────────────────┐
│ column   ┆ base_format ┆ compare_format │
│ ---      ┆ ---         ┆ ---            │
│ str      ┆ str         ┆ str            │
╞══════════╪═════════════╪════════════════╡
│ Example2 ┆ Utf8        ┆ Int64          │
│ Example3 ┆ null        ┆ Int64          │
└──────────┴─────────────┴────────────────┘
```
</details>

- pandas comparison example
- custom equality function
- use of column aliases

### To DO:
- [x] Linting (Ruff)
- [x] Make into python package
- [x] Add makefile for easy linting and tests
- [x] Statistics should indicate which statistics are referencing columns
- [x] Add all statistics frame to tests
- [x] Add schema differences to schema summary
- [x] Make row examples alternate between base only and compare only so that it is more readable.
- [x] Add limit value to the examples.
- [x] Updated value differences summary so that Statistic is something that makes sense.
- [x] Publish package to pypi
- [x] Add difference criterion.
- [x] Add license
- [x] Make package easy to use (i.e. so you only have to import pl_compare and then you can us pl_compare)
- [x] Add table name labels that can replace 'base' and 'compare'.
- [] Write up docstrings
- [] Write up readme (with code examples)
- [] Raise error and print examples if duplicates are present.
- [] Add a count of the number of rows that have any differences to the value differences summary.
- [] Add total number of value differences to the value differences summary.
- [] Add parameter to hide column differences with 0 differences.
- [] Update report so that non differences are (optionally) not displayed.
- [] Change id_columns to be named 'join_on' and add a test that checks that abritrary join conditions work.
- [] Update code to use a config dataclass that can be passed between the class and functions.
- [] Test for large amounts of data
- [] Benchmark for different sizes of data.
- [] strict MyPy type checking
- [] Github actions for testing
- [] Github actions for linting
- [] Github actions for publishing

