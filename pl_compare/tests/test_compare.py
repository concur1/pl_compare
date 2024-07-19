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


def test_expected_values_returned_for_bools_for_equal_dfs_none_id_columns(base_df):
    compare_result = compare(None, base_df, base_df)
    assert compare_result.is_schemas_equal() is True
    assert compare_result.is_rows_equal() is True
    assert compare_result.is_values_equal() is True
    assert compare_result.is_equal() is True


def test_expected_values_returned_for_bools_for_unequal_dfs(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    assert compare_result.is_schemas_equal() is False
    assert compare_result.is_rows_equal() is False
    assert compare_result.is_values_equal() is False
    assert compare_result.is_equal() is False


def test_expected_values_returned_for_bools_for_equal_dfs_no_id_columns(base_df):
    compare_result = compare([], base_df, base_df)
    assert compare_result.is_schemas_equal() is True
    assert compare_result.is_rows_equal() is True
    assert compare_result.is_values_equal() is True
    assert compare_result.is_equal() is True


def test_expected_values_returned_for_bools_for_equal_dfs(base_df):
    compare_result = compare(["ID"], base_df, base_df)
    assert compare_result.is_schemas_equal() is True
    assert compare_result.is_rows_equal() is True
    assert compare_result.is_values_equal() is True
    assert compare_result.is_equal() is True


def test_expected_values_returned_for_schemas_summary(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_schemas_summary = pl.DataFrame(
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
    print(compare_result.schemas_summary())
    assert_frame_equal(compare_result.schemas_summary(), expected_schemas_summary)


def test_expected_values_returned_for_schema_differences(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_schema_differnces = pl.DataFrame(
        {
            "column": ["Example2", "Example3"],
            "base_format": ["String", None],
            "compare_format": ["Int64", "Int64"],
        }
    )
    assert_frame_equal(compare_result.schemas_sample(), expected_schema_differnces)


def test_expected_values_returned_for_rows_summary(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_rows_summary = pl.DataFrame(
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
    assert_frame_equal(compare_result.rows_summary(), expected_rows_summary)


def test_expected_values_returned_row_differences(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_row_differences = pl.DataFrame(
        {
            "ID": ["12345678", "1234567810"],
            "variable": ["status", "status"],
            "value": ["in base only", "in compare only"],
        }
    )
    assert_frame_equal(compare_result.rows_sample(), expected_row_differences)


def test_expected_values_returned_values_summary(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_values_summary = pl.DataFrame(
        {
            "Value Differences": ["Total Value Differences", "Example1"],
            "Count": [1, 1],
            "Percentage": [50.0, 50.0],
        },
        schema={"Value Differences": pl.Utf8, "Count": pl.Int64, "Percentage": pl.Float64},
    )
    print(compare_result.values_summary())
    print(expected_values_summary)
    assert_frame_equal(compare_result.values_summary(), expected_values_summary)


def test_expected_values_returned_value_differences(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df)
    expected_value_differences = pl.DataFrame(
        {"ID": ["1234567"], "variable": ["Example1"], "base": [6], "compare": [2]}
    )
    assert_frame_equal(compare_result.values_sample(), expected_value_differences)


def test_expected_values_returned_summary():
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
    pl.Config.set_tbl_rows(100)
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
                "Value diffs Col:Example1",
                "Value diffs Col:Example2",
            ],
            "Count": [3, 3, 3, 0, 0, 0, 3, 3, 1, 1, 2, 2, 1, 1],
        },
        schema={"Statistic": pl.Utf8, "Count": pl.Int64},
    )
    print(compare_result.summary())
    print(expected_value_differences)
    assert_frame_equal(compare_result.summary(), expected_value_differences)


def test_streaming_input_without_streaming_flag_returns_non_lazy_dfs():
    base_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/base.csv")
    compare_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/compare.csv")
    compare_result = compare(["ID"], base_df, compare_df)
    assert isinstance(compare_result.schemas_summary(), pl.DataFrame)
    assert isinstance(compare_result.schemas_sample(), pl.DataFrame)
    assert isinstance(compare_result.rows_summary(), pl.DataFrame)
    assert isinstance(compare_result.rows_sample(), pl.DataFrame)
    assert isinstance(compare_result.values_summary(), pl.DataFrame)
    assert isinstance(compare_result.values_sample(), pl.DataFrame)


def test_streaming_input_with_streaming_flag_returns_lazy_dfs():
    """test"""
    base_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/base.csv")
    compare_df = pl.scan_csv("pl_compare/tests/test_data/scenario_1/compare.csv")
    compare_result = compare(["ID"], base_df, compare_df, streaming=True)
    assert isinstance(compare_result.schemas_summary(), pl.LazyFrame)
    assert isinstance(compare_result.schemas_sample(), pl.LazyFrame)
    assert isinstance(compare_result.rows_summary(), pl.LazyFrame)
    assert isinstance(compare_result.rows_sample(), pl.LazyFrame)
    assert isinstance(compare_result.values_summary(), pl.LazyFrame)
    assert isinstance(compare_result.values_sample(), pl.LazyFrame)


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
        .values_sample()
        .select(pl.len().alias("Count"))
        .item()
        == 1
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=1)
        .rows_sample()
        .select(pl.len().alias("Count"))
        .item()
        == 2
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=2)
        .values_sample()
        .select(pl.len().alias("Count"))
        .item()
        == 2
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=2)
        .rows_sample()
        .select(pl.len().alias("Count"))
        .item()
        == 4
    )
    assert (
        compare(["ID"], base_df, compare_df, sample_limit=2, resolution=1)
        .values_sample()
        .select(pl.len().alias("Count"))
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
    assert_frame_equal(compare_result.summary(), expected_value_differences)


def test_error_raised_when_dupes_supplied_for_1_1_validation():
    base_df = pl.DataFrame(
        {
            "ID": ["123456", "123456", "1234567", "12345678"],
            "ID2": ["123456", "123457", "1234567", "12345678"],
            "Example1": [1, 1, 6, 3],
            "Example2": [1, 1, 2, 3],
        }
    )
    compare_df = pl.DataFrame(
        {
            "ID": ["123456", "1234567", "1234567810"],
            "ID2": ["123456", "1234567", "1234567810"],
            "Example1": [1, 2, 3],
            "Example2": [1, 2, 3],
        },
    )

    with pytest.raises(pl.exceptions.ComputeError):
        compare(["ID"], base_df, compare_df, validate="1:1").values_summary()
    with pytest.raises(pl.exceptions.ComputeError):
        compare(["ID"], base_df, compare_df, validate="1:1").rows_summary()

    with pytest.raises(pl.exceptions.ComputeError):
        compare(["ID"], base_df, compare_df, validate="1:m").values_summary()
    with pytest.raises(pl.exceptions.ComputeError):
        compare(["ID"], base_df, compare_df, validate="1:m").rows_summary()
    with pytest.raises(pl.exceptions.ComputeError):
        compare(["ID"], compare_df, base_df, validate="m:1").values_summary()
    with pytest.raises(pl.exceptions.ComputeError):
        compare(["ID"], compare_df, base_df, validate="m:1").rows_summary()

    compare(["ID", "ID2"], base_df, compare_df, "m:1").values_summary()
    compare(["ID", "ID2"], base_df, compare_df, "m:1").rows_summary()
    compare(["ID", "ID2"], compare_df, base_df, "1:m").values_summary()
    compare(["ID", "ID2"], compare_df, base_df, "1:m").rows_summary()
