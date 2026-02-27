import polars as pl

print(f"Polars version: {pl.__version__}")

lf = pl.LazyFrame({
    "id": [1, 2],
    "example_variable": ["existing_data", "existing_data"] 
})

result = lf.unpivot(
    index="id",
    on=["example_variable"], # We are melting a column into a new column with the same name
    variable_name="variable",
    value_name="value"
).collect()

print(result)
