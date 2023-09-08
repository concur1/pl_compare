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
print("-----------boolean indicatior------------")
print("-----------------------------------------")
print("is_schema_unequal:", compare_result.is_schema_unequal())
print("is_rows_unequal:", compare_result.is_rows_unequal())
print("is_values_unequal:", compare_result.is_values_unequal())
print("-----------------------------------------")
print("----------statistical summaries----------")
print("-----------------------------------------")
print("schema differences summary:")
print(compare_result.schema_differences_summary())
print("row differences summary:")
print(compare_result.row_differences_summary())
print("Value differences summary:")
print(compare_result.value_differences_summary())
print("all differences statistics:")
print(compare_result.all_differences_summary())
print("-----------------------------------------")
print("----------differences samples----------")
print("-----------------------------------------")
print("schema differences sample:")
print(compare_result.schema_differences_sample())
print("row differences sample:")
print(compare_result.row_differences_sample())
print("Value differences sample:")
print(compare_result.value_differences_sample())
