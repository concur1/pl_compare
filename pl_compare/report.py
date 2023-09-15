import polars as pl
from compare import compare

import time

start = time.time()
print("hello")

pl.Config.set_tbl_rows(100)
#chicago1 = pl.scan_parquet("pl_compare/output_data/chicago1.parquet").head(1000000)
#chicago2 = pl.scan_parquet("pl_compare/output_data/chicago2.parquet").head(1000000)
#compare(["ID", "Case Number"], chicago1, chicago2, base_alias="test").report()

end = time.time()
print(end - start)
# data1 = pl.read_parquet("output_data/used_cars1.parquet")
# data2 = pl.read_arquet("output_data/used_cars2.parquet")

# compare(["vin"], data1, data2, sample_limit=100).report()
import polars as pl

base_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "12345678"],
        "Example1": [1, 6, 3],
        "Example2": [1, 2, 3],
    }
)
compare_df = pl.DataFrame(
    {
        "ID": ["123456", "1234567", "1234567810"],
        "Example1": [1, 2, 3],
        "Example2": [1, 2, 3],
    },
)

compare_result = compare(["ID"], base_df, compare_df, hide_empty_stats=True)

expected_value_differences = pl.DataFrame(
    {
        "Statistic": [
            "Columns in base",
            "Columns in compare",
            "Columns in base and compare",
            "Rows in base",
            "Rows in compare",
            "Rows only in base",
            "Rows only in compare",
            "Rows in base and compare",
            "Value diffs Col:Example1",
        ],
        "Count": [3, 3, 3, 3, 3, 1, 1, 2, 1, ],
    }
)



print(compare_result.all_differences_summary() == expected_value_differences)