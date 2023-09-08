import polars as pl
from compare import compare

pl.Config.set_tbl_rows(100)
chicago1 = pl.scan_parquet("pl_compare/output_data/chicago1.parquet").head(1000000)
chicago2 = pl.scan_parquet("pl_compare/output_data/chicago2.parquet").head(1000000)
compare(["ID", "Case Number"], chicago1, chicago2).value_differences_sample()

# data1 = pl.read_parquet("output_data/used_cars1.parquet")
# data2 = pl.read_arquet("output_data/used_cars2.parquet")

# compare(["vin"], data1, data2, sample_limit=100).report()
