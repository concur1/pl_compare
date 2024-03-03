import polars as pl
from polars.testing import assert_frame_equal
from pl_compare.compare import compare


def base_df():
    return pl.DataFrame(
        {
            "ID": ["123456", "1234567", "1234567810"],
            "Example1": [1, 2, 3],
            "Example2": [1, 2, 3],
            "Example3": [1, 2, 3],
        },
    )

def compare_df():
    return pl.DataFrame(
        {
            "ID": ["123456", "1234567", "1234567810"],
            "Example1": [1, 2, 3],
            "Example2": [1, 2, 3],
            "Example3": [1, 2, 3],
        },
    )

print(compare(["ID", "Example1"],base_df(), compare_df()).report())
