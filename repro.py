import polars as pl
import sys

print(f"Python version: {sys.version}")
print(f"Polars version: {pl.__version__}")

# 1. Create a LazyFrame with two distinct columns
lf = pl.LazyFrame({
    "col_1": [1, 2, 3],
    "col_2": [4, 5, 6]
})

# 2. Rename 'col_1' to 'col_2'. 
# Now we have TWO columns named 'col_2' in the Lazy plan.
# Polars 1.37 allows this plan to sit in memory.
lf_duplicate = lf.rename({"col_1": "col_2"})

print("Lazy plan with duplicate names created.")

# 3. Perform a 'harmless' operation.
# In your library, this might be a join or a select.
# We will just filter.
lf_final = lf_duplicate.filter(pl.lit(True))

print("Attempting to collect...")

try:
    # 1.37: Likely succeeds because it doesn't resolve the conflict unless forced.
    # 1.38: Fails with DuplicateError: column with name 'col_2' has more than one occurrence.
    result = lf_final.collect()
    print("SUCCESS: Collected successfully (Polars 1.37 behavior)")
    print(result)
except pl.exceptions.DuplicateError as e:
    print(f"FAILURE: Caught DuplicateError (Polars 1.38 behavior)")
    print(e)
    # We exit with 1 only if we are on 1.38+ to show the regression
    if pl.__version__ >= "1.38":
        sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
