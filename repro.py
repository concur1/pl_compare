import polars as pl
import sys

print(f"Polars version: {pl.__version__}")

lf = pl.LazyFrame({
    "id": [1, 2],
    "variable": ["existing_data", "existing_data"] 
})

result = lf.unpivot(
    index="id",
    on=["variable"], # We are melting a column into a new column with the same name
    variable_name="exampe_variable",
    value_name="example_value"
).collect()

print(result)
