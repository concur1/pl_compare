import polars as pl
import sys

print(f"Polars version: {pl.__version__}")

# 1. Start with a LazyFrame where one column has the name we want to use later
lf = pl.LazyFrame({
    "id": [1, 2],
    "variable": ["existing_data", "existing_data"] 
})

# 2. Replicate the library's unpivot logic
# We unpivot a column, but we set the variable_name to 'variable'
# (which is the default, but also already exists in our schema above)
try:
    # 1.37: Succeeds. It just creates a second 'variable' column or shadows it.
    # 1.38: Fails with DuplicateError: column with name 'variable' has more than one occurrence
    result = lf.unpivot(
        index="id",
        on=["variable"], # We are melting a column into a new column with the same name
        variable_name="variable",
        value_name="value"
    ).collect()
    
    print("SUCCESS: Collected successfully.")
except pl.exceptions.DuplicateError as e:
    print(f"FAILURE: Caught DuplicateError: {e}")
    # Only exit 1 on 1.38 to prove it's a regression
    if "1.38" in pl.__version__:
        sys.exit(1)
