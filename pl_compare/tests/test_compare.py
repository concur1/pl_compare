import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pl_compare.compare import compare


@pytest.fixture
def base_df():
    return pl.DataFrame(
        {
            "ID": ["123456", "1234567", "12345678"],
            "Example1": [1, 6, 3],
            "Example2": ["1", "2", "3"],
        }
    )


@pytest.fixture
def compare_df():
    return pl.DataFrame(
        {
            "ID": ["123456", "1234567", "1234567810"],
            "Example1": [1, 2, 3],
            "Example2": [1, 2, 3],
            "Example3": [1, 2, 3],
        },
    )


def test_expected_values_returned_for_bools(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    assert compare_result.is_schema_unequal() is True
    assert compare_result.is_rows_unequal() is True
    assert compare_result.is_values_unequal() is True


def test_expected_values_returned_for_schema_summary(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_schema_summary = pl.DataFrame(
        {
            "Statistic": [
                "Columns in base",
                "Columns in compare",
                "Columns in base and compare",
                "Columns only in base",
                "Columns only in compare",
                "Columns with schema differences",
            ],
            "Count": [3, 4, 3, 0, 1, 1],
        }
    )
    print(compare_result.schema_differences_summary())
    assert_frame_equal(compare_result.schema_differences_summary(), expected_schema_summary)


def test_expected_values_returned_for_schema_differences(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_schema_differnces = pl.DataFrame(
        {
            "column": ["Example2", "Example3"],
            "base_format": ["Utf8", None],
            "compare_format": ["Int64", "Int64"],
        }
    )
    assert_frame_equal(compare_result.schema_differences_sample(), expected_schema_differnces)


def test_expected_values_returned_for_row_summary(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_row_summary = pl.DataFrame(
        {
            "Statistic": [
                "Rows in base",
                "Rows in compare",
                "Rows only in base",
                "Rows only in compare",
                "Rows in base and compare",
            ],
            "Count": [3, 3, 1, 1, 2],
        }
    )
    assert_frame_equal(compare_result.row_differences_summary(), expected_row_summary)


def test_expected_values_returned_row_differences(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_row_differences = pl.DataFrame(
        {
            "ID": ["12345678", "1234567810"],
            "variable": ["status", "status"],
            "value": ["in base only", "in compare only"],
        }
    )
    assert_frame_equal(compare_result.row_differences_sample(), expected_row_differences)


def test_expected_values_returned_value_summary(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_value_summary = pl.DataFrame(
        {"Value Differences for Column": ["Total Value Differences", "Example1"], "Count": [1, 1]},
        schema={"Value Differences for Column": pl.Utf8, "Count": pl.Int64},
    )
    assert_frame_equal(compare_result.value_differences_summary(), expected_value_summary)


def test_expected_values_returned_value_differences(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_value_differences = pl.DataFrame(
        {"ID": ["1234567"], "variable": ["Example1"], "base": [6], "compare": [2]}
    )
    assert_frame_equal(compare_result.value_differences_sample(), expected_value_differences)


def test_expected_values_returned_all_differences_summary():
    base_df = pl.DataFrame(
            {
                "ID": ["123456", "1234567", "12345678"],
                "Example1": [1, 6, 3],
                "Example2": [100, 2, 3],
            }
        )
    compare_df = pl.DataFrame(
            {
                "ID": ["123456", "1234567", "1234567810"],
                "Example1": [1, 2, 3],
                "Example2": [1, 2, 3],
            },
        )
    compare_result = compare(["ID"], base_df, compare_df)
    expected_value_differences = pl.DataFrame(
        {
            "Statistic": [
                "Columns in base",
                "Columns in compare",
                "Columns in base and compare",
                "Columns only in base",
                "Columns only in compare",
                "Columns with schema differences",
                "Rows in base",
                "Rows in compare",
                "Rows only in base",
                "Rows only in compare",
                "Rows in base and compare",
                "Total Value Differences",
                "Value diffs Col:Example2",
                "Value diffs Col:Example1",
            ],
            "Count": [3, 3, 3, 0, 0, 0, 3, 3, 1, 1, 2, 2, 1, 1],
        }
    )
    print(compare_result.all_differences_summary())
    print(expected_value_differences)
    assert_frame_equal(compare_result.all_differences_summary(), expected_value_differences)

def test_streaming_input_without_streaming_flag_returns_non_lazy_dfs():
    base_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/base.csv")
    compare_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/compare.csv")
    compare_result = compare(["ID"], base_df, compare_df)
    assert isinstance(compare_result.schema_differences_summary(), pl.DataFrame)
    assert isinstance(compare_result.schema_differences_sample(), pl.DataFrame)
    assert isinstance(compare_result.row_differences_summary(), pl.DataFrame)
    assert isinstance(compare_result.row_differences_sample(), pl.DataFrame)
    assert isinstance(compare_result.value_differences_summary(), pl.DataFrame)
    assert isinstance(compare_result.value_differences_sample(), pl.DataFrame)


def test_streaming_input_with_streaming_flag_returns_lazy_dfs():
    """test"""
    base_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/base.csv")
    compare_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/compare.csv")
    compare_result = compare(["ID"], base_df, compare_df, streaming=True)
    assert isinstance(compare_result.schema_differences_summary(), pl.LazyFrame)
    assert isinstance(compare_result.schema_differences_sample(), pl.LazyFrame)
    assert isinstance(compare_result.row_differences_summary(), pl.LazyFrame)
    assert isinstance(compare_result.row_differences_sample(), pl.LazyFrame)
    assert isinstance(compare_result.value_differences_summary(), pl.LazyFrame)
    assert isinstance(compare_result.value_differences_sample(), pl.LazyFrame)


def test_sample_limit():
    base_df = pl.DataFrame(
        {
            "ID": ["1", "123456", "1234567", "12345678", "123456789"],
            "Example1": [5, 1, 6, 3, 1],
            "Example2": ["5", "1", "2", "3", "1"],
        }
    )

    compare_df = pl.DataFrame(
        {
            "ID": ["1", "123456", "1234567", "1234567810", "12345678910"],
            "Example1": [4, 1, 2, 3, 1],
            "Example2": [5, 1, 2, 3, 1],
            "Example3": [5, 1, 2, 3, 1],
        },
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=1)
        .value_differences_sample()
        .select(pl.count())
        .item()
        == 1
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=1)
        .row_differences_sample()
        .select(pl.count())
        .item()
        == 2
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=2)
        .value_differences_sample()
        .select(pl.count())
        .item()
        == 2
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=2)
        .row_differences_sample()
        .select(pl.count())
        .item()
        == 4
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=2, threshold=1)
        .value_differences_sample()
        .select(pl.count())
        .item()
        == 1
    )


def test_hide_empty_stats():
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
                "Total Value Differences",
                "Value diffs Col:Example1",
            ],
            "Count": [
                3,
                3,
                3,
                3,
                3,
                1,
                1,
                2,
                1,
                1,
            ],
        }
    )
    assert_frame_equal(compare_result.all_differences_summary(), expected_value_differences)
