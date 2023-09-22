import polars as pl
from compare import compare
import logging

import time

start = time.time()
logging.basicConfig(level=logging.DEBUG)
logging.info("hello")

pl.Config.set_tbl_rows(100)
chicago1 = pl.scan_parquet("pl_compare/output_data/chicago1.parquet").head(1000000)
chicago2 = pl.scan_parquet("pl_compare/output_data/chicago2.parquet").head(1000000)
compare(["ID", "Case Number"], chicago1, chicago2).report(print=logging.info)

end = time.time()
