import polars as pl
import sys

print(f"Polars version: {pl.__version__}")

# 1. Start with a simple LazyFrame
lf = pl.LazyFrame({"a": [1], "b": [2]})

# 2. Replicate the behavior in compare.py:
# Mapping two internal columns to the same output name.
target_alias = "duplicated_name"

# We select the same alias twice. 
# Polars 1.37: Allows the Lazy plan to be built and collected.
# Polars 1.38: Fails during 'Sink' resolution.
lf_lazy = lf.select([
    pl.col("a").alias(target_alias),
    pl.col("b").alias(target_alias)
])

print("Attempting to .collect()...")
# No try-except block. We want a natural crash to fail the CI/CD.
result = lf_lazy.collect()

print("Success! (This only prints if the version allows duplicates)")
print(result)
