import polars as pl
from pl_compare import compare

base_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1, 6, 3],
        "Example2": ["1", "2", "3"],
    }
)

compare_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1, 2, 3],
        "Example2": [1, 2, 3],
        "Example3": [1, 2, 3],
    },
)


compare_result = compare(["ID"], base_df, compare_df)
print("-----------------------------------------")
print("-----------boolean indicator-------------")
print("-----------------------------------------")
print("is_schemas_equal:", compare_result.is_schemas_equal())
print("is_rows_equal:", compare_result.is_rows_equal())
print("is_values_equal:", compare_result.is_values_equal())
print("-----------------------------------------")
print("----------statistical summaries----------")
print("-----------------------------------------")
print("schema differences summary:")
print(compare_result.schemas_summary())
print("row differences summary:")
print(compare_result.rows_summary())
print("Value differences summary:")
print(compare_result.values_summary())
print("all differences statistics:")
print(compare_result.summary())
print("-----------------------------------------")
print("----------differences samples----------")
print("-----------------------------------------")
print("schema differences sample:")
print(compare_result.schemas_sample())
print("row differences sample:")
print(compare_result.rows_sample())
print("Value differences sample:")
print(compare_result.values_sample())
