import polars as pl

def reproduce_polars_issue():
    # 1. Create a basic LazyFrame
    # We use two columns that we will attempt to rename to the same alias
    lf = pl.LazyFrame({
        "col_a": [1, 2, 3],
        "col_b": ["x", "y", "z"]
    })

    # 2. Define a rename mapping where two different keys point to the same value
    # This simulates your internal mapping bug where:
    # "__pl_compare_variable" -> variable_alias
    # "__pl_compare_status"   -> variable_alias
    target_alias = "duplicated_name"
    mapping = {
        "col_a": target_alias,
        "col_b": target_alias
    }

    print(f"Attempting to rename columns to the same alias: {target_alias}")
    
    # 3. Apply the rename
    # In Lazy mode, Polars often accepts this and adds it to the query plan
    lf = lf.rename(mapping)

    # 4. Trigger the error
    # The DuplicateError typically surfaces here during materialization
    try:
        result = lf.collect()
        print("Resulting Columns:", result.columns)
    except Exception as e:
        print("\n--- CAUGHT ERROR ---")
        print(e)
        print("--------------------")

if __name__ == "__main__":
    reproduce_polars_issue()
