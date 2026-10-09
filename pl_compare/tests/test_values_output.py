"""Regression tests for the exact output of values_summary() / values_sample().

These pin the behaviour of `main` so any future change to the value-differences
pipeline (column stringification across dtypes, null semantics, row order,
the no-join-keys and nested-dtype paths) is caught. The output of the values
path is compared row-by-row against `main` in the benchmark, so both the
string values and their order are load-bearing.
"""

import polars as pl
from polars.testing import assert_frame_equal
from pl_compare.compare import compare


def _sample(expected_rows, row_column="join_columns.ID", row_dtype=pl.Utf8):
    rows = list(zip(*expected_rows))  # (ids, variables, bases, compares)
    return pl.DataFrame(
        {
            row_column: pl.Series(row_column, rows[0], dtype=row_dtype),
            "variable": pl.Series("variable", rows[1], dtype=pl.Utf8),
            "base": pl.Series("base", rows[2], dtype=pl.Utf8),
            "compare": pl.Series("compare", rows[3], dtype=pl.Utf8),
        }
    )


def _check_values(base, compare_df, summary_rows, sample_rows, join_columns=["ID"], row_column="join_columns.ID", row_dtype=pl.Utf8):
    result = compare(join_columns, base, compare_df)
    expected_summary = pl.DataFrame(
        {"Value Differences": [r[0] for r in summary_rows],
         "Count": pl.Series([r[1] for r in summary_rows], dtype=pl.Int64),
         "Percentage": pl.Series([r[2] for r in summary_rows], dtype=pl.Float64)},
        schema={"Value Differences": pl.Utf8, "Count": pl.Int64, "Percentage": pl.Float64},
    )
    assert_frame_equal(result.values_summary(), expected_summary)
    assert_frame_equal(result.values_sample(), _sample(sample_rows, row_column, row_dtype))


def test_values_numeric_bool_stringification():
    """Int/bool/float column values stringify deterministically (incl. -0.0, 1e16, 1e-300)."""
    base = pl.DataFrame({
        "ID": ["a", "b", "c", "d", "e"],
        "i":  [1, -2, 3, 4, 5],
        "b":  [True, False, True, False, True],
        "f":  [0.0, -0.0, 1e16, 1e-300, None],
    })
    compare_df = pl.DataFrame({
        "ID": ["a", "b", "c", "d", "e"],
        "i":  [1, -2, 3, 9, 5],
        "b":  [True, False, False, False, True],
        "f":  [1.0, 0.0, 1e16, 2e-300, None],
    })
    _check_values(
        base, compare_df,
        [("Total Value Differences", 4, 26.666666666666668), ("b", 1, 20.0), ("f", 2, 40.0), ("i", 1, 20.0)],
        [
            ("a", "f", "0.0", "1.0"),
            ("c", "b", "true", "false"),
            ("d", "i", "4", "9"),
            ("d", "f", "1e-300", "2e-300"),
        ],
    )


def test_values_date_stringification():
    base = pl.DataFrame({"ID": ["a", "b", "c"], "d": [18823, 18824, None]}, schema={"ID": pl.Utf8, "d": pl.Date})
    compare_df = pl.DataFrame({"ID": ["a", "b", "c"], "d": [18823, 18825, None]}, schema={"ID": pl.Utf8, "d": pl.Date})
    _check_values(
        base, compare_df,
        [("Total Value Differences", 1, 33.333333333333336), ("d", 1, 33.333333333333336)],
        [("b", "d", "2021-07-16", "2021-07-17")],
    )


def test_values_datetime_stringification():
    """Datetime fractions drop trailing zeros; a zero fraction omits the fraction entirely."""
    base = pl.DataFrame(
        {"ID": ["a", "b", "c"],
         "dt": ["2021-01-02 00:00:00.123456", "2021-01-02 01:23:45.500000", "2021-01-02 03:00:00"]},
        schema={"ID": pl.Utf8, "dt": pl.Datetime("us")},
    )
    compare_df = pl.DataFrame(
        {"ID": ["a", "b", "c"],
         "dt": ["2021-01-02 00:00:00.123456", "2021-01-02 01:23:45.000000", "2021-01-02 03:00:00"]},
        schema={"ID": pl.Utf8, "dt": pl.Datetime("us")},
    )
    _check_values(
        base, compare_df,
        [("Total Value Differences", 1, 33.333333333333336), ("dt", 1, 33.333333333333336)],
        [("b", "dt", "2021-01-02 01:23:45.500", "2021-01-02 01:23:45")],
    )


def test_values_time_stringification():
    base = pl.DataFrame({"ID": ["a", "b", "c"], "t": ["12:34:56.5", "12:34:56", "23:59:59"]}, schema={"ID": pl.Utf8, "t": pl.Time})
    compare_df = pl.DataFrame({"ID": ["a", "b", "c"], "t": ["12:34:56.5", "12:34:57", "23:59:59"]}, schema={"ID": pl.Utf8, "t": pl.Time})
    _check_values(
        base, compare_df,
        [("Total Value Differences", 1, 33.333333333333336), ("t", 1, 33.333333333333336)],
        [("b", "t", "12:34:56", "12:34:57")],
    )


def test_values_duration_stringification():
    import datetime
    base = pl.DataFrame({
        "ID": ["a", "b", "c"],
        "dur": [datetime.timedelta(hours=1, minutes=2, seconds=3), datetime.timedelta(hours=2), datetime.timedelta(seconds=45)],
    })
    compare_df = pl.DataFrame({
        "ID": ["a", "b", "c"],
        "dur": [datetime.timedelta(hours=1, minutes=2, seconds=3), datetime.timedelta(hours=2, minutes=5), datetime.timedelta(seconds=45)],
    })
    _check_values(
        base, compare_df,
        [("Total Value Differences", 1, 33.333333333333336), ("dur", 1, 33.333333333333336)],
        [("b", "dur", "PT7200S", "PT7500S")],
    )


def test_values_decimal_and_categorical_stringification():
    base = pl.DataFrame(
        {"ID": ["a", "b", "c"], "dec": ["1.50", "2.50", None], "cat": ["x", "y", "z"]},
        schema={"ID": pl.Utf8, "dec": pl.Decimal(10, 2), "cat": pl.Categorical},
    )
    compare_df = pl.DataFrame(
        {"ID": ["a", "b", "c"], "dec": ["1.50", "2.51", None], "cat": ["x", "q", "z"]},
        schema={"ID": pl.Utf8, "dec": pl.Decimal(10, 2), "cat": pl.Categorical},
    )
    _check_values(
        base, compare_df,
        [("Total Value Differences", 2, 33.333333333333336), ("cat", 1, 33.333333333333336), ("dec", 1, 33.333333333333336)],
        [
            ("b", "dec", "2.50", "2.51"),
            ("b", "cat", "y", "q"),
        ],
    )


def test_values_null_handling():
    """null vs value is a diff both ways; null vs null is not."""
    base = pl.DataFrame({"ID": ["n1", "n2", "n3", "n4", "n5"], "v": [None, None, "x", None, "keep"]})
    compare_df = pl.DataFrame({"ID": ["n1", "n2", "n3", "n4", "n5"], "v": [None, "y", None, "z", "keep"]})
    _check_values(
        base, compare_df,
        [("Total Value Differences", 3, 60.0), ("v", 3, 60.0)],
        [
            ("n2", "v", None, "y"),
            ("n3", "v", "x", None),
            ("n4", "v", None, "z"),
        ],
    )


def test_values_sample_row_order_is_deterministic():
    """Rows keep (ID, column) order from the melted comparison output; the benchmark
    zips current vs main row-by-row, so order is part of the contract."""
    base = pl.DataFrame({"ID": ["r1", "r2", "r3", "r4"], "A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "C": [100, 200, 300, 400]})
    compare_df = pl.DataFrame({"ID": ["r1", "r2", "r3", "r4"], "A": [1, 9, 3, 8], "B": [10, 20, 7, 40], "C": [100, 200, 300, 6]})
    expected = _sample([
        ("r2", "A", "2", "9"),
        ("r3", "B", "30", "7"),
        ("r4", "A", "4", "8"),
        ("r4", "C", "400", "6"),
    ])
    for _ in range(10):
        assert_frame_equal(compare(["ID"], base, compare_df).values_sample(), expected)


def test_values_sample_without_join_columns():
    base = pl.DataFrame({"A": [1, 2, 3], "B": ["u", "v", "w"]})
    compare_df = pl.DataFrame({"A": [1, 9, 3], "B": ["u", "v", "x"]})
    _check_values(
        base, compare_df,
        [("Total Value Differences", 2, 33.333333333333336), ("A", 1, 33.333333333333336), ("B", 1, 33.333333333333336)],
        [
            (2, "A", "2", "9"),
            (3, "B", "w", "x"),
        ],
        join_columns=None,
        row_column="join_columns.row_number",
        row_dtype=pl.UInt32,
    )


def test_values_nested_list_columns():
    """List-typed columns (nested dtype) use the json fallback; exact repr: '[true]'."""
    base = pl.DataFrame({"ID": ["l1", "l2", "l3"], "n": [1, 2, 3], "L": [[True], [True], [True, False]]})
    compare_df = pl.DataFrame({"ID": ["l1", "l2", "l3"], "n": [1, 2, 9], "L": [[True], [False], [True, False]]})
    _check_values(
        base, compare_df,
        [("Total Value Differences", 2, 33.333333333333336), ("L", 1, 33.333333333333336), ("n", 1, 33.333333333333336)],
        [
            ("l2", "L", "[true]", "[false]"),
            ("l3", "n", "3", "9"),
        ],
    )
