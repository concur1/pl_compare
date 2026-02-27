import polars as pl
import sys

# Display environment info for the log
print(f"Python version: {sys.version}")
print(f"Polars version: {pl.__version__}")

# 1. Create a LazyFrame
lf = pl.LazyFrame({
    "column_a": [1, 2, 3],
    "column_b": [4, 5, 6]
})

# 2. Rename both columns to the same name
# Polars allows this during the planning stage (LazyFrame)
lf = lf.rename({
    "column_a": "duplicate_name",
    "column_b": "duplicate_name"
})

print("Attempting to .collect() the LazyFrame...")

# 3. Trigger the crash
# This will raise a DuplicateError and exit the script
result = lf.collect()

# This line will never be reached if the bug/behavior is present
print("Successfully collected:")
print(result)
