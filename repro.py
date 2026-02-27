import polars as pl
import sys

print(f"Polars version: {pl.__version__}")

lf = pl.LazyFrame({
    "id": [1, 2],
    "example_col": ["existing_data", "existing_data"] 
})

# We are melting a column into a new column with the same name
result = lf.unpivot(
    index="id",
    on=["example_col"], 
    variable_name="example_col",
    value_name="value"
).collect()

