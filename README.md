## pl_compare: Compare and find the differences between two Polars DataFrames. 

- Get statistical summaries and/or examples and/or a boolean to indicate:
  - Schema differences
  - Row differences
  - Value differences
- Easily works for Pandas dataframes and other tabular data formats with conversion using Apache arrow 
- View differences as a text report
- Get differences as a Polars LazyFrame or DataFrame
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
```
is_schema_unequal: True
is_rows_unequal: True
is_values_unequal: True
```
</details>

<details>
<summary>Schema differences summary and details</summary>

```python
import olars as pl.
from pl_compare import ceompar

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
print("schema_differences_summary()")
print(compare_result.schema_differences_summary())
print("schema_differences_sample()")
print(compare_result.schema_differences_sample())
```
output:
```
schema_differences_summary()
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
schema_differences_sample()
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

<details>
<summary>Row differences summary and details</summary>

```python
import olars as pl.
from pl_compare import ceompar

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
print("row_differences_summary()")
print(compare_result.row_differences_summary())
print("row_differences_sample()")
print(compare_result.row_differences_sample())
```
output:
```
row_differences_summary()
shape: (5, 2)
┌──────────────────────────┬───────┐
│ Statistic                ┆ Count │
│ ---                      ┆ ---   │
│ str                      ┆ i64   │
╞══════════════════════════╪═══════╡
│ Rows in base             ┆ 3     │
│ Rows in compare          ┆ 3     │
│ Rows only in base        ┆ 1     │
│ Rows only in compare     ┆ 1     │
│ Rows in base and compare ┆ 2     │
└──────────────────────────┴───────┘
row_differences_sample()
shape: (2, 3)
┌────────────┬──────────┬─────────────────┐
│ ID         ┆ variable ┆ value           │
│ ---        ┆ ---      ┆ ---             │
│ str        ┆ str      ┆ str             │
╞════════════╪══════════╪═════════════════╡
│ 12345678   ┆ status   ┆ in base only    │
│ 1234567810 ┆ status   ┆ in compare only │
└────────────┴──────────┴─────────────────┘
```
</details>

<details>
<summary>Value differences summary and details</summary>

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
print("value_differences_summary()")
print(compare_result.value_differences_summary())
print("value_differences_sample()")
print(compare_result.value_differences_sample())
```
output:
```
value_differences_summary()
shape: (1, 2)
┌──────────────────────────────┬───────┐
│ Value Differences for Column ┆ Count │
│ ---                          ┆ ---   │
│ str                          ┆ i64   │
╞══════════════════════════════╪═══════╡
│ Example1                     ┆ 1     │
└──────────────────────────────┴───────┘
value_differences_sample()
shape: (1, 4)
┌─────────┬──────────┬──────┬─────────┐
│ ID      ┆ variable ┆ base ┆ compare │
│ ---     ┆ ---      ┆ ---  ┆ ---     │
│ str     ┆ str      ┆ i64  ┆ i64     │
╞═════════╪══════════╪══════╪═════════╡
│ 1234567 ┆ Example1 ┆ 6    ┆ 2       │
└─────────┴──────────┴──────┴─────────┘
```
</details>

<details>
<summary>Full report</summary>

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
compare_result.report()
```
output:
```
Schema summary:
shape: (6, 2)
┌─────────────────────────────────┬───────┐
│ Statistic                       ┆ Count │
│ ---                             ┆ ---   │
│ str                             ┆ i64   │
╞═════════════════════════════════╪═══════╡
│ Columns in base                 ┆ 3     │
│ Columns in compare              ┆ 4     │
│ Columns in base and compare     ┆ 3     │
│ Columns only in base            ┆ 0     │
│ Columns only in compare         ┆ 1     │
│ Columns with schema differences ┆ 1     │
└─────────────────────────────────┴───────┘
Schema differences: True
shape: (2, 3)
┌──────────┬─────────────┬────────────────┐
│ column   ┆ base_format ┆ compare_format │
│ ---      ┆ ---         ┆ ---            │
│ str      ┆ str         ┆ str            │
╞══════════╪═════════════╪════════════════╡
│ Example2 ┆ Utf8        ┆ Int64          │
│ Example3 ┆ null        ┆ Int64          │
└──────────┴─────────────┴────────────────┘
Row summary:
shape: (5, 2)
┌──────────────────────────┬───────┐
│ Statistic                ┆ Count │
│ ---                      ┆ ---   │
│ str                      ┆ i64   │
╞══════════════════════════╪═══════╡
│ Rows in base             ┆ 3     │
│ Rows in compare          ┆ 3     │
│ Rows only in base        ┆ 1     │
│ Rows only in compare     ┆ 1     │
│ Rows in base and compare ┆ 2     │
└──────────────────────────┴───────┘
Row differences: True
shape: (2, 3)
┌────────────┬──────────┬─────────────────┐
│ ID         ┆ variable ┆ value           │
│ ---        ┆ ---      ┆ ---             │
│ str        ┆ str      ┆ str             │
╞════════════╪══════════╪═════════════════╡
│ 12345678   ┆ status   ┆ in base only    │
│ 1234567810 ┆ status   ┆ in compare only │
└────────────┴──────────┴─────────────────┘
Value summary:
shape: (1, 2)
┌──────────────────────────────┬───────┐
│ Value Differences for Column ┆ Count │
│ ---                          ┆ ---   │
│ str                          ┆ i64   │
╞══════════════════════════════╪═══════╡
│ Example1                     ┆ 1     │
└──────────────────────────────┴───────┘
Value differences: True
shape: (1, 4)
┌─────────┬──────────┬──────┬─────────┐
│ ID      ┆ variable ┆ base ┆ compare │
│ ---     ┆ ---      ┆ ---  ┆ ---     │
│ str     ┆ str      ┆ i64  ┆ i64     │
╞═════════╪══════════╪══════╪═════════╡
│ 1234567 ┆ Example1 ┆ 6    ┆ 2       │
└─────────┴──────────┴──────┴─────────┘
All differences summary:
shape: (12, 2)
┌─────────────────────────────────┬───────┐
│ Statistic                       ┆ Count │
│ ---                             ┆ ---   │
│ str                             ┆ i64   │
╞═════════════════════════════════╪═══════╡
│ Columns in base                 ┆ 3     │
│ Columns in compare              ┆ 4     │
│ Columns in base and compare     ┆ 3     │
│ Columns only in base            ┆ 0     │
│ Columns only in compare         ┆ 1     │
│ Columns with schema differences ┆ 1     │
│ Rows in base                    ┆ 3     │
│ Rows in compare                 ┆ 3     │
│ Rows only in base               ┆ 1     │
│ Rows only in compare            ┆ 1     │
│ Rows in base and compare        ┆ 2     │
│ Value diffs Col:Example1        ┆ 1     │
└─────────────────────────────────┴───────┘
```
</details>

<details>
<summary>Compare two pandas dataframes</summary>

```python
import polars as pl
import pandas as pd
from pl_compare import compare

base_df = pd.DataFrame(data=
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1, 6, 3],
        "Example2": ["1", "2", "3"],
    }
)
compare_df = pd.DataFrame(data=
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1, 2, 3],
        "Example2": [1, 2, 3],
        "Example3": [1, 2, 3],
    },
)

compare_result = compare(["ID"], pl.from_pandas(base_df), pl.from_pandas(compare_df))
compare_result.report()
```
output:
```
Schema summary:
shape: (6, 2)
┌─────────────────────────────────┬───────┐
│ Statistic                       ┆ Count │
│ ---                             ┆ ---   │
│ str                             ┆ i64   │
╞═════════════════════════════════╪═══════╡
│ Columns in base                 ┆ 3     │
│ Columns in compare              ┆ 4     │
│ Columns in base and compare     ┆ 3     │
│ Columns only in base            ┆ 0     │
│ Columns only in compare         ┆ 1     │
│ Columns with schema differences ┆ 1     │
└─────────────────────────────────┴───────┘
Schema differences: True
shape: (2, 3)
┌──────────┬─────────────┬────────────────┐
│ column   ┆ base_format ┆ compare_format │
│ ---      ┆ ---         ┆ ---            │
│ str      ┆ str         ┆ str            │
╞══════════╪═════════════╪════════════════╡
│ Example2 ┆ Utf8        ┆ Int64          │
│ Example3 ┆ null        ┆ Int64          │
└──────────┴─────────────┴────────────────┘
Row summary:
shape: (5, 2)
┌──────────────────────────┬───────┐
│ Statistic                ┆ Count │
│ ---                      ┆ ---   │
│ str                      ┆ i64   │
╞══════════════════════════╪═══════╡
│ Rows in base             ┆ 3     │
│ Rows in compare          ┆ 3     │
│ Rows only in base        ┆ 1     │
│ Rows only in compare     ┆ 1     │
│ Rows in base and compare ┆ 2     │
└──────────────────────────┴───────┘
Row differences: True
shape: (2, 3)
┌────────────┬──────────┬─────────────────┐
│ ID         ┆ variable ┆ value           │
│ ---        ┆ ---      ┆ ---             │
│ str        ┆ str      ┆ str             │
╞════════════╪══════════╪═════════════════╡
│ 12345678   ┆ status   ┆ in base only    │
│ 1234567810 ┆ status   ┆ in compare only │
└────────────┴──────────┴─────────────────┘
Value summary:
shape: (1, 2)
┌──────────────────────────────┬───────┐
│ Value Differences for Column ┆ Count │
│ ---                          ┆ ---   │
│ str                          ┆ i64   │
╞══════════════════════════════╪═══════╡
│ Example1                     ┆ 1     │
└──────────────────────────────┴───────┘
Value differences: True
shape: (1, 4)
┌─────────┬──────────┬──────┬─────────┐
│ ID      ┆ variable ┆ base ┆ compare │
│ ---     ┆ ---      ┆ ---  ┆ ---     │
│ str     ┆ str      ┆ i64  ┆ i64     │
╞═════════╪══════════╪══════╪═════════╡
│ 1234567 ┆ Example1 ┆ 6    ┆ 2       │
└─────────┴──────────┴──────┴─────────┘
All differences summary:
shape: (12, 2)
┌─────────────────────────────────┬───────┐
│ Statistic                       ┆ Count │
│ ---                             ┆ ---   │
│ str                             ┆ i64   │
╞═════════════════════════════════╪═══════╡
│ Columns in base                 ┆ 3     │
│ Columns in compare              ┆ 4     │
│ Columns in base and compare     ┆ 3     │
│ Columns only in base            ┆ 0     │
│ Columns only in compare         ┆ 1     │
│ Columns with schema differences ┆ 1     │
│ Rows in base                    ┆ 3     │
│ Rows in compare                 ┆ 3     │
│ Rows only in base               ┆ 1     │
│ Rows only in compare            ┆ 1     │
│ Rows in base and compare        ┆ 2     │
│ Value diffs Col:Example1        ┆ 1     │
└─────────────────────────────────┴───────┘
```
</details>


<details>
<summary>Specify a threshold to control the frandularity of hte comparison for numeric columns.</summary>

```python
import polars as pl
from pl_compare import compare

base_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1.111, 6.11, 3.11],
    }
)

compare_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1.114, 6.14, 3.12],
    },
)

def custom_equality_check(col: str, format: pl.DataType) -> pl.Expr:
    return (
        (pl.col(f"{col}_base") != pl.col(f"{col}_compare"))
        | (pl.col(f"{col}_base").is_null() & ~pl.col(f"{col}_compare").is_null())
        | (~pl.col(f"{col}_base").is_null() & pl.col(f"{col}_compare").is_null())
    )
print("With threshold of 0.01")
compare_result = compare(["ID"], base_df, compare_df, threshold=0.01)
print(compare_result.value_differences_sample())
print("With no threshold")
compare_result = compare(["ID"], base_df, compare_df)
print(compare_result.value_differences_sample())
```
output:
```
With threshold of 0.01
shape: (1, 4)
┌─────────┬──────────┬──────┬─────────┐
│ ID      ┆ variable ┆ base ┆ compare │
│ ---     ┆ ---      ┆ ---  ┆ ---     │
│ str     ┆ str      ┆ f64  ┆ f64     │
╞═════════╪══════════╪══════╪═════════╡
│ 1234567 ┆ Example1 ┆ 6.11 ┆ 6.14    │
└─────────┴──────────┴──────┴─────────┘
With no threshold
shape: (2, 4)
┌─────────┬──────────┬───────┬─────────┐
│ ID      ┆ variable ┆ base  ┆ compare │
│ ---     ┆ ---      ┆ ---   ┆ ---     │
│ str     ┆ str      ┆ f64   ┆ f64     │
╞═════════╪══════════╪═══════╪═════════╡
│ 123456  ┆ Example1 ┆ 1.111 ┆ 1.114   │
│ 1234567 ┆ Example1 ┆ 6.11  ┆ 6.14    │
└─────────┴──────────┴───────┴─────────┘
```
</details>


<details>
<summary>Example using alias for base and compare dataframes.</summary>

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

compare_result = compare(["ID"], 
                         base_df, 
                         compare_df, 
                         base_alias="before_change", 
                         compare_alias="after_change")

print("value_differences_summary()")
print(compare_result.schema_differences_sample())
print("value_differences_sample()")
print(compare_result.value_differences_sample())
```
output:
```
value_differences_summary()
shape: (2, 3)
┌──────────┬──────────────────────┬─────────────────────┐
│ column   ┆ before_change_format ┆ after_change_format │
│ ---      ┆ ---                  ┆ ---                 │
│ str      ┆ str                  ┆ str                 │
╞══════════╪══════════════════════╪═════════════════════╡
│ Example2 ┆ Utf8                 ┆ Int64               │
│ Example3 ┆ null                 ┆ Int64               │
└──────────┴──────────────────────┴─────────────────────┘
value_differences_sample()
shape: (1, 4)
┌─────────┬──────────┬───────────────┬──────────────┐
│ ID      ┆ variable ┆ before_change ┆ after_change │
│ ---     ┆ ---      ┆ ---           ┆ ---          │
│ str     ┆ str      ┆ i64           ┆ i64          │
╞═════════╪══════════╪═══════════════╪══════════════╡
│ 1234567 ┆ Example1 ┆ 6             ┆ 2            │
└─────────┴──────────┴───────────────┴──────────────┘
```
</details>


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
- [x] Write up readme (with code examples)
- [x] Add parameter to hide column differences with 0 differences.
- [] Raise error and print examples if duplicates are present.
- [] Add a count of the number of rows that have any differences to the value differences summary.
- [] Add total number of value differences to the value differences summary.
- [] Update report so that non differences are (optionally) not displayed.
- [] Change id_columns to be named 'join_on' and add a test that checks that abritrary join conditions work.
- [] Update code to use a config dataclass that can be passed between the class and functions.
- [] Simplify custom equality checks and add example.
- [] Test for large amounts of data
- [] Benchmark for different sizes of data.
- [] strict MyPy type checking
- [] Github actions for testing
- [] Github actions for linting
- [x] Github actions for publishing

### not sure of:
- [] Seperate out dev dependencies from library dependnecies?
- [] Change 'threshold' to be 'granularity'/'numeric granularity?'
