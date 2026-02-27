import polars as pl
import sys

print(f"Python version: {sys.version}")
print(f"Polars version: {pl.__version__}")

# 1. Start with a LazyFrame that has a column named 'clash_name'
lf = pl.LazyFrame({
    "id": [1],
    "clash_name": [10]  # This is the existing column
})

# 2. Perform an unpivot where the variable_name is ALSO 'clash_name'
# In 1.37.0, Polars often allows the Lazy plan to be built and only 
# errors if a downstream operation becomes truly ambiguous.
# In 1.38.1, the schema validation is stricter during the optimization.
try:
    # We unpivot 'id', meaning 'clash_name' stays in the index/schema
    # while we also try to create a new column named 'clash_name' via variable_name.
    poisoned_lf = lf.unpivot(
        on=["id"], 
        variable_name="clash_name", 
        value_name="some_value"
    )
    
    print("LazyFrame plan created. Resolving schema...")
    
    # This call triggers the DuplicateError in 1.38.1
    schema = poisoned_lf.collect_schema()
    print("Schema resolved successfully.")
    
    # This call would also trigger it
    # res = poisoned_lf.collect()
    
except pl.exceptions.DuplicateError as e:
    print(f"\nCaught expected error: {e}")
    sys.exit(1) # Exit with error for GitHub Action
except Exception as e:
    print(f"\nCaught unexpected error: {type(e).__name__}: {e}")
    sys.exit(1)

print("\nSuccess: No DuplicateError thrown.")
sys.exit(0)
