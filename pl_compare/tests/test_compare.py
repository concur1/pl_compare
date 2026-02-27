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


def test_report_with_aliases(base_df, compare_df):
    compare_result = compare(["ID"], base_df, compare_df, base_alias="test", compare_alias="other")
    assert compare_result.is_schemas_equal() is False
    assert compare_result.is_rows_equal() is False
    assert compare_result.is_values_equal() is False
    assert compare_result.is_equal() is False
    assert """SCHEMA DIFFERENCES:
shape: (6, 2)
┌─────────────────────────────────┬───────┐
│ Statistic                       ┆ Count │
│ ---                             ┆ ---   │
│ str                             ┆ i64   │
╞═════════════════════════════════╪═══════╡
│ Columns in test                 ┆ 3     │
│ Columns in other                ┆ 4     │
│ Columns in test and other       ┆ 3     │
│ Columns only in test            ┆ 0     │
│ Columns only in other           ┆ 1     │
│ Columns with schema difference… ┆ 1     │
└─────────────────────────────────┴───────┘
shape: (2, 3)
┌──────────┬─────────────┬──────────────┐
│ column   ┆ test_format ┆ other_format │
│ ---      ┆ ---         ┆ ---          │
│ str      ┆ str         ┆ str          │
╞══════════╪═════════════╪══════════════╡
│ Example2 ┆ String      ┆ Int64        │
│ Example3 ┆ null        ┆ Int64        │
└──────────┴─────────────┴──────────────┘""" in str(
        compare_result.report()
    )

    assert """ROW DIFFERENCES:
shape: (5, 2)
┌────────────────────────┬───────┐
│ Statistic              ┆ Count │
│ ---                    ┆ ---   │
│ str                    ┆ i64   │
╞════════════════════════╪═══════╡
│ Rows in test           ┆ 3     │
│ Rows in other          ┆ 3     │
│ Rows only in test      ┆ 1     │
│ Rows only in other     ┆ 1     │
│ Rows in test and other ┆ 2     │
└────────────────────────┴───────┘
shape: (2, 3)
┌─────────────────┬──────────┬───────────────┐
│ join_columns.ID ┆ variable ┆ value         │
│ ---             ┆ ---      ┆ ---           │
│ str             ┆ str      ┆ str           │
╞═════════════════╪══════════╪═══════════════╡
│ 12345678        ┆ status   ┆ in test only  │
│ 1234567810      ┆ status   ┆ in other only │
└─────────────────┴──────────┴───────────────┘""" in str(
        compare_result.report()
    )
    assert """VALUE DIFFERENCES:
shape: (2, 3)
┌─────────────────────────┬───────┬────────────┐
│ Value Differences       ┆ Count ┆ Percentage │
│ ---                     ┆ ---   ┆ ---        │
│ str                     ┆ i64   ┆ f64        │
╞═════════════════════════╪═══════╪════════════╡
│ Total Value Differences ┆ 1     ┆ 50.0       │
│ Example1                ┆ 1     ┆ 50.0       │
└─────────────────────────┴───────┴────────────┘
shape: (1, 4)
┌─────────────────┬──────────┬──────┬───────┐
│ join_columns.ID ┆ variable ┆ test ┆ other │
│ ---             ┆ ---      ┆ ---  ┆ ---   │
│ str             ┆ str      ┆ str  ┆ str   │
╞═════════════════╪══════════╪══════╪═══════╡
│ 1234567         ┆ Example1 ┆ 6    ┆ 2     │
└─────────────────┴──────────┴──────┴───────┘""" in str(
        compare_result.report()
    )


@pytest.mark.parametrize(
    "column_name",
    [
        # "value",
        "variable",
        "in_base",
        "in_compare",
        "status",
        "join_columns",
        "base",
        "compare",
        "Count",
        "column",
        "Statistic",
        "ID",
        "Percentage",
        "format",
        "base_format",
        "compare_format",
        "Value Differences",
    ],
)
def test_special_column_names(column_name):
    base_df = pl.DataFrame({column_name: ["123", "456", "888"], "x": [1, 2, 3]})
    compare_df = pl.DataFrame({column_name: ["123", "456", "789"], "x": [1, 22, 3]})
    # With consistent join column prefixing, ALL join columns are always prefixed
    join_col = f"join_columns.{column_name}"

    expected_rows = pl.DataFrame(
        {
            join_col: ["888", "789"],
            "variable": ["status", "status"],
            "value": ["in base only", "in compare only"],
        }
    )
    expected_values = pl.DataFrame(
        {
            join_col: ["456"],
            "variable": ["x"],
            "base": ["2"],
            "compare": ["22"],
        }
    )

    # Without value_alias and variable_alias, the special column names would produce
    #    polars.exceptions.DuplicateError: projections contained duplicate output name 'variable'
    # because the same column names are used internally.
    compare_result = compare([column_name], base_df, compare_df)

    # Test multiple methods that use column aliases
    pl.testing.assert_frame_equal(compare_result.rows_sample(), expected_rows)
    pl.testing.assert_frame_equal(compare_result.values_sample(), expected_values)

    schemas_sample = compare_result.schemas_sample()
    assert "column" in schemas_sample.columns

    rows_summary = compare_result.rows_summary()
    assert rows_summary.height > 0
    assert "Statistic" in rows_summary.columns
    assert "Count" in rows_summary.columns

    values_summary = compare_result.values_summary()
    assert values_summary.height > 0
    assert "Value Differences" in values_summary.columns
    assert "Count" in values_summary.columns
    assert "Percentage" in values_summary.columns

    summary = compare_result.summary()
    assert summary.height > 0
    assert "Statistic" in summary.columns
    assert "Count" in summary.columns

    assert compare_result.is_rows_equal() is False  # Rows differ
    assert compare_result.is_values_equal() is False  # Values differ
    assert compare_result.is_equal() is False  # Overall not equal


def test_expected_values_returned_for_bools_for_equal_dfs_none_id_columns(base_df):
    compare_result = compare(None, base_df, base_df)
    assert compare_result.is_schemas_equal() is True
    assert compare_result.is_rows_equal() is True
    assert compare_result.is_values_equal() is True
    assert compare_result.is_equal() is True
    assert """┌────────────────────┬───────┐
│ Statistic          ┆ Count │
│ ---                ┆ ---   │
│ str                ┆ i64   │
╞════════════════════╪═══════╡
│ Columns in base    ┆ 4     │
│ Columns in compare ┆ 4     │
│ Rows in base       ┆ 3     │
│ Rows in compare    ┆ 3     │
└────────────────────┴───────┘""" in str(
        compare_result.report()
    )


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
            "join_columns.ID": ["12345678", "1234567810"],
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
        {"join_columns.ID": ["1234567"], "variable": ["Example1"], "base": ["6"], "compare": ["2"]}
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

    compare(["ID", "ID2"], base_df, compare_df, validate="m:1").values_summary()
    compare(["ID", "ID2"], base_df, compare_df, validate="m:1").rows_summary()
    compare(["ID", "ID2"], compare_df, base_df, validate="1:m").values_summary()
    compare(["ID", "ID2"], compare_df, base_df, validate="1:m").rows_summary()


def test_for_single_column_table_columns():
    base_df = pl.DataFrame(
        {
            "ID": [123456, 1234567, 12345678],
        }
    )
    compare_df = pl.DataFrame(
        {
            "ID": [123456, 1234567, 12345678],
        },
    )
    comp = compare(["ID"], base_df, compare_df)
    assert comp.is_equal()


def test_output_when_there_are_row_differences_but_no_columns_to_compare_exist():
    base_df = pl.DataFrame(
        {
            "ID": ["123456", "123456", "1234567", "12345678"],
            "ID2": ["123456", "123457", "1234567", "12345678"],
            "ID3": ["123456", "123457", "1234567", "12345678"],
            "ID4": ["123456", "123457", "1234567", "12345678"],
        }
    )
    compare_df = pl.DataFrame(
        {
            "ID": ["123456", "1234567", "1234567810"],
            "ID2": ["123456", "1234567", "1234567810"],
            "ID3": ["123456", "1234567", "1234567810"],
            "ID4": ["123456", "1234567", "1234567810"],
        },
    )

    comp = compare(["ID", "ID2", "ID3", "ID4"], base_df, compare_df, validate="1:1")
    assert not comp.is_equal()
    assert not comp.is_rows_equal()
    assert comp.is_values_equal()
    assert comp.is_schemas_equal()
    assert """--------------------------------------------------------------------------------
COMPARISON REPORT
--------------------------------------------------------------------------------
No Schema differences found.
--------------------------------------------------------------------------------

ROW DIFFERENCES:
shape: (5, 2)
┌──────────────────────────┬───────┐
│ Statistic                ┆ Count │
│ ---                      ┆ ---   │
│ str                      ┆ i64   │
╞══════════════════════════╪═══════╡
│ Rows in base             ┆ 4     │
│ Rows in compare          ┆ 3     │
│ Rows only in base        ┆ 2     │
│ Rows only in compare     ┆ 1     │
│ Rows in base and compare ┆ 2     │
└──────────────────────────┴───────┘
shape: (3, 6)
┌─────────────────┬─────────────────┬─────────────────┬────────────────┬──────────┬────────────────┐
│ join_columns.ID ┆ join_columns.ID ┆ join_columns.ID ┆ join_columns.I ┆ variable ┆ value          │
│ ---             ┆ 2               ┆ 3               ┆ D4             ┆ ---      ┆ ---            │
│ str             ┆ ---             ┆ ---             ┆ ---            ┆ str      ┆ str            │
│                 ┆ str             ┆ str             ┆ str            ┆          ┆                │
╞═════════════════╪═════════════════╪═════════════════╪════════════════╪══════════╪════════════════╡
│ 123456          ┆ 123457          ┆ 123457          ┆ 123457         ┆ status   ┆ in base only   │
│ 1234567810      ┆ 1234567810      ┆ 1234567810      ┆ 1234567810     ┆ status   ┆ in compare     │
│                 ┆                 ┆                 ┆                ┆          ┆ only           │
│ 12345678        ┆ 12345678        ┆ 12345678        ┆ 12345678       ┆ status   ┆ in base only   │
└─────────────────┴─────────────────┴─────────────────┴────────────────┴──────────┴────────────────┘
""" in str(
        comp.report()
    )


def test_comparing_list_raises_exception():
    """Polars has a bug/regression where an unpivot will not work if on columns of multiple types are used.

    It has been raised here: https://github.com/pola-rs/polars/issues/17501

    We will stick with an older version of polars until this bug is fixed.

    The error:
    E       polars.exceptions.InvalidOperationError: 'unpivot' not supported for dtype: struct[3]

    """

    base_df = pl.LazyFrame(
        {
            "ID": [
                "1",
                "2",
                "3",
            ],
            "col1": [
                "1",
                "2",
                "3",
            ],
            "col2": [[True], [True], [True, False]],
        }
    )
    compare_df = pl.LazyFrame(
        {
            "ID": [
                "1",
                "2",
                "3",
            ],
            "col1": [
                "1",
                "2",
                "3",
            ],
            "col2": [[True], [True], [True, False]],
        }
    )
    comp = compare(["ID"], base_df, compare_df)
    comp.is_equal()


def test_empty_tables_dont_cause_error():
    base_df = pl.LazyFrame(
        {
            "ID": [],
            "col1": [],
            "col2": [],
        }
    )
    compare_df = pl.LazyFrame(
        {
            "ID": [],
            "col1": [],
            "col2": [],
        }
    )
    comp = compare(["ID"], base_df, compare_df)

    assert comp.is_equal(), "Tables are not equal"
    comp.report()


def test_usage_of_status_column():
    base_df = pl.DataFrame(
        {
            "status": ["123456", "1234567", "12345678"],
            "Example1": [1, 6, 3],
        }
    )
    compare_df = pl.DataFrame(
        {
            "status": ["123456", "1234567", "1234567810"],
            "Example1": [1, 2, 3],
        }
    )

    # This should not cause an error
    compare(["status"], base_df, compare_df)


def test_column_aliases_in_metadata():
    """Test that value_alias and variable_alias are properly stored in column mapping."""
    base_df = pl.DataFrame(
        {
            "ID": [1, 2, 3],
            "value": [10, 20, 30],
            "variable": ["a", "b", "c"],
        }
    )
    compare_df = pl.DataFrame(
        {
            "ID": [1, 2, 4],
            "value": [10, 25, 40],
            "variable": ["a", "x", "y"],
        }
    )

    # Test with default aliases
    result_default = compare(["ID"], base_df, compare_df)
    meta_default = result_default._comparison_metadata

    # Verify default aliases are in column mapping
    assert meta_default.column_mapping.mapping["__pl_compare_value"] == "value"
    assert meta_default.column_mapping.mapping["__pl_compare_variable"] == "variable"

    # Test with custom aliases
    result_custom = compare(
        ["ID"], base_df, compare_df, value_alias="custom_value", variable_alias="custom_var"
    )
    meta_custom = result_custom._comparison_metadata

    # Verify custom aliases are in column mapping
    assert meta_custom.column_mapping.mapping["__pl_compare_value"] == "custom_value"
    assert meta_custom.column_mapping.mapping["__pl_compare_variable"] == "custom_var"

    # Verify that ComparisonMetadata no longer has value_alias and variable_alias attributes
    assert not hasattr(meta_default, "value_alias")
    assert not hasattr(meta_default, "variable_alias")
    assert not hasattr(meta_custom, "value_alias")
    assert not hasattr(meta_custom, "variable_alias")

    # Test that the custom aliases appear in the output
    row_diff = result_custom.rows_sample()
    assert "custom_var" in row_diff.columns
    assert "custom_value" in row_diff.columns

    value_diff = result_custom.values_sample()
    assert "custom_var" in value_diff.columns
    assert "custom_value" not in value_diff.columns  # value is base/compare columns
    assert "base" in value_diff.columns
    assert "compare" in value_diff.columns


def test_column_aliases_with_special_names():
    """Test that column aliases work correctly with special column names."""
    # Test with 'value' as a user column name
    base_df = pl.DataFrame(
        {
            "ID": [1, 2, 3],
            "value": [10, 20, 30],  # User column named 'value'
            "data": ["a", "b", "c"],
        }
    )
    compare_df = pl.DataFrame(
        {
            "ID": [1, 2, 4],
            "value": [10, 25, 40],  # User column named 'value'
            "data": ["a", "x", "y"],
        }
    )

    # Use custom aliases to avoid conflict
    result = compare(
        ["ID"], base_df, compare_df, value_alias="result_value", variable_alias="result_var"
    )

    # Verify the output uses custom aliases
    row_diff = result.rows_sample()
    assert "result_var" in row_diff.columns
    assert "result_value" in row_diff.columns
    assert "join_columns.ID" in row_diff.columns

    # Verify user's 'value' column is preserved with join_columns prefix
    assert "value" not in row_diff.columns  # Should be prefixed

    value_diff = result.values_sample()
    assert "result_var" in value_diff.columns
    assert "base" in value_diff.columns
    assert "compare" in value_diff.columns


def test_reserved_column_names_error():
    """Test that using reserved internal column names raises an appropriate error."""
    base_df = pl.DataFrame(
        {"__pl_compare_value": [1, 2, 3], "ID": [1, 2, 3], "data": ["a", "b", "c"]}
    )
    compare_df = pl.DataFrame(
        {"__pl_compare_status": ["x", "y", "z"], "ID": [1, 2, 3], "data": ["a", "b", "c"]}
    )

    # Test single reserved column
    with pytest.raises(ValueError, match="Column name.*are reserved for internal use"):
        compare(["ID"], base_df, compare_df)

    # Test multiple reserved columns
    base_df_multi = pl.DataFrame(
        {"__pl_compare_value": [1, 2, 3], "__pl_compare_base": [10, 20, 30], "ID": [1, 2, 3]}
    )

    with pytest.raises(ValueError, match="Column name.*are reserved for internal use"):
        compare(["ID"], base_df_multi, base_df_multi)

    # Test that the error message mentions the pattern
    with pytest.raises(ValueError, match="All columns starting with '__pl_compare_' are reserved"):
        compare(["ID"], base_df, compare_df)


def test_reserved_column_names_error_specific_columns():
    """Test error for each specific reserved column name."""
    reserved_columns = [
        "__pl_compare_in_base",
        "__pl_compare_in_compare",
        "__pl_compare_value",
        "__pl_compare_variable",
        "__pl_compare_base",
        "__pl_compare_compare",
        "__pl_compare_status",
    ]

    for reserved_col in reserved_columns:
        base_df = pl.DataFrame({reserved_col: [1, 2, 3], "ID": [1, 2, 3], "data": ["a", "b", "c"]})
        compare_df = pl.DataFrame({"ID": [1, 2, 3], "data": ["a", "b", "c"]})

        with pytest.raises(ValueError, match=f"Column name.*{reserved_col}.*are reserved"):
            compare(["ID"], base_df, compare_df)


def test_apply_column_renames_type_error():
    """Test that @apply_column_renames raises TypeError for non-DataFrame/LazyFrame results."""
    from pl_compare.compare import apply_column_renames, ComparisonMetadata, ColumnMapping

    # Create a mock function that returns an invalid type
    @apply_column_renames
    def mock_function_returns_string(meta: ComparisonMetadata):
        return "invalid_result"

    @apply_column_renames
    def mock_function_returns_int(meta: ComparisonMetadata):
        return 42

    @apply_column_renames
    def mock_function_returns_list(meta: ComparisonMetadata):
        return [1, 2, 3]

    # Create a mock metadata object
    mock_meta = ComparisonMetadata(
        join_columns=["ID"],
        base_df=pl.LazyFrame({"ID": [1, 2, 3]}),
        compare_df=pl.LazyFrame({"ID": [1, 2, 3]}),
        streaming=False,
        resolution=None,
        equality_check=None,
        sample_limit=None,
        base_alias="base",
        compare_alias="compare",
        schema_comparison=False,
        hide_empty_stats=False,
        validate="m:m",
        column_mapping=ColumnMapping(mapping={"__pl_compare_value": "value"}),
    )

    # Test string result
    with pytest.raises(TypeError, match="Expected result to be a polars DataFrame or LazyFrame"):
        mock_function_returns_string(mock_meta)

    # Test int result
    with pytest.raises(TypeError, match="Expected result to be a polars DataFrame or LazyFrame"):
        mock_function_returns_int(mock_meta)

    # Test list result
    with pytest.raises(TypeError, match="Expected result to be a polars DataFrame or LazyFrame"):
        mock_function_returns_list(mock_meta)
