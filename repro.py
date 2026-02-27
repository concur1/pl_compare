import polars as pl
import sys

print(f"Python version: {sys.version}")
print(f"Polars version: {pl.__version__}")

# 1. Setup a standard LazyFrame
lf = pl.LazyFrame({
    "id": [1, 2],
    "column_to_melt": [10, 20]
})

# 2. This simulates the logic in your library's 'get_column_value_differences'
# In your code, 'variable_alias' and 'value_alias' were accidentally 
# pointing to the same string.
conflicting_name = "clash_name"

print(f"Attempting unpivot with conflicting variable/value names: '{conflicting_name}'")

# In Polars, unpivot/melt creates two new columns. 
# If you name them the same thing, it creates a corrupt Lazy plan.
lf_melted = lf.unpivot(
    index="id",
    on=["column_to_melt"],
    variable_name=conflicting_name,
    value_name=conflicting_name
)

# 3. Trigger the collection
# This should produce the DuplicateError during the optimization/collect phase
try:
    result = lf_melted.collect()
    print("Columns in result:", result.columns)
except Exception as e:
    print("\n--- CAUGHT ERROR ---")
    print(e)
    # Re-raise to ensure the GitHub Action fails
    raise
