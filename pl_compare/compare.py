import polars as pl
from polars.datatypes.classes import DataTypeClass
from typing import Literal, Callable, List, Union, Dict
from dataclasses import dataclass


@dataclass
class ComparisonMetadata:
    """Class for holding the (meta)data used to generate the comparison dataframes."""

    join_columns: List[str]
    base_df: pl.LazyFrame
    compare_df: pl.LazyFrame
    streaming: bool
    resolution: Union[float, None]
    equality_check: Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None]
    sample_limit: Union[int, None]
    base_alias: str
    compare_alias: str
    schema_comparison: bool
    hide_empty_stats: bool
    validate: Literal["m:m", "m:1", "1:m", "1:1"]


def convert_to_dataframe(df: Union[pl.LazyFrame, pl.DataFrame]) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    return df


def convert_to_lazyframe(df: Union[pl.LazyFrame, pl.DataFrame]) -> pl.LazyFrame:
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
    return df


def set_df_type(
    df: Union[pl.LazyFrame, pl.DataFrame], streaming: bool = False
) -> Union[pl.LazyFrame, pl.DataFrame]:
    if streaming:
        return convert_to_lazyframe(df)
    if not streaming:
        return convert_to_dataframe(df)


def get_uncertain_row_count(df: pl.LazyFrame) -> int:
    df_solid = convert_to_dataframe(df).select(pl.col("Count"))
    if df_solid.height > 0:
        row_count: int = df_solid.item()
        return row_count
    else:
        return 0


def get_row_comparison_summary(meta: ComparisonMetadata) -> pl.DataFrame:
    combined_table = meta.base_df.select(meta.join_columns + [pl.lit(True).alias("in_base")]).join(
        meta.compare_df.select(meta.join_columns + [pl.lit(True).alias("in_compare")]),
        on=meta.join_columns,
        how="full",
        coalesce=True,
        validate=meta.validate,
    )
    grouped_rows = (
        combined_table.select(meta.join_columns + ["in_base", "in_compare"])
        .group_by(["in_base", "in_compare"])
        .agg(pl.len().alias("Count"))
    )
    base_only_rows = get_uncertain_row_count(
        grouped_rows.filter(pl.col("in_base") & pl.col("in_compare").is_null())
    )
    compare_only_rows = get_uncertain_row_count(
        grouped_rows.filter(pl.col("in_base").is_null() & pl.col("in_compare"))
    )
    shared_rows = get_uncertain_row_count(
        grouped_rows.filter(
            pl.col("in_base")
            & pl.col("in_base").is_not_null()
            & pl.col("in_compare")
            & pl.col("in_compare").is_not_null()
        )
    )
    final_df = (
        pl.DataFrame(
            {
                f"Rows in {meta.base_alias}": [shared_rows + base_only_rows],
                f"Rows in {meta.compare_alias}": [shared_rows + compare_only_rows],
                f"Rows only in {meta.base_alias}": [base_only_rows],
                f"Rows only in {meta.compare_alias}": [compare_only_rows],
                f"Rows in {meta.base_alias} and {meta.compare_alias}": [shared_rows],
            }
        )
        .transpose(include_header=True, column_names=["Col Differences"])
        .rename({"column": "Statistic", "Col Differences": "Count"})
    )
    if meta.hide_empty_stats:
        final_df = final_df.filter(pl.col("Count") > 0)
    return final_df


def get_base_only_rows(meta: ComparisonMetadata) -> pl.LazyFrame:
    combined_table = meta.base_df.select(meta.join_columns).join(
        meta.compare_df.select(meta.join_columns),
        on=meta.join_columns,
        how="anti",
    )
    return combined_table.select(
        meta.join_columns + [pl.lit(f"in {meta.base_alias} only").alias("status")]
    ).unpivot(index=meta.join_columns, on=["status"])


def get_compare_only_rows(meta: ComparisonMetadata) -> pl.LazyFrame:
    combined_table = meta.compare_df.select(meta.join_columns).join(
        meta.base_df.select(meta.join_columns), on=meta.join_columns, how="anti"
    )
    return combined_table.select(
        meta.join_columns + [pl.lit(f"in {meta.compare_alias} only").alias("status")]
    ).unpivot(index=meta.join_columns, on=["status"])


def get_row_differences(meta: ComparisonMetadata) -> pl.LazyFrame:
    base_only_rows = get_base_only_rows(meta).with_row_index()
    compare_only_rows = get_compare_only_rows(meta).with_row_index()
    if meta.sample_limit is not None:
        base_only_rows = base_only_rows.limit(meta.sample_limit)
        compare_only_rows = compare_only_rows.limit(meta.sample_limit)
    return (
        pl.concat(
            [
                base_only_rows,
                compare_only_rows,
            ]
        )
        .sort("index")
        .drop("index")
    )


def get_equality_check(
    equality_check: Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None],
    resolution: Union[float, None],
    col: str,
    format: Union[pl.DataType, DataTypeClass],
) -> pl.Expr:
    def default_equality_check(col: str, format: Union[pl.DataType, DataTypeClass]) -> pl.Expr:
        return (
            (pl.col(f"{col}_base") != pl.col(f"{col}_compare"))
            | (pl.col(f"{col}_base").is_null() & ~pl.col(f"{col}_compare").is_null())
            | (~pl.col(f"{col}_base").is_null() & pl.col(f"{col}_compare").is_null())
        )

    def ignore_numeric_differences_equality_check(
        col: str, format: Union[pl.DataType, DataTypeClass]
    ) -> pl.Expr:
        return (
            ((pl.col(f"{col}_base") - pl.col(f"{col}_compare")).abs() > resolution)
            | (pl.col(f"{col}_base").is_null() & ~pl.col(f"{col}_compare").is_null())
            | (~pl.col(f"{col}_base").is_null() & pl.col(f"{col}_compare").is_null())
        )

    if resolution is not None and format in [
        pl.Float32,
        pl.Float64,
        pl.Decimal,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]:
        return ignore_numeric_differences_equality_check(col, format)
    if equality_check is not None:
        return equality_check(col, format)
    return default_equality_check(col, format)


def get_combined_tables(
    join_columns: List[str],
    base_df: pl.LazyFrame,
    compare_df: pl.LazyFrame,
    compare_columns: Dict[str, Union[DataTypeClass, pl.DataType]],
    equality_check: Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None],
    coalesce: bool,
    how_join: Literal["inner", "full"] = "inner",
    resolution: Union[float, None] = None,
    validate: Literal["m:m", "m:1", "1:m", "1:1"] = "1:1",
) -> pl.LazyFrame:
    base_df = base_df.rename({col: f"{col}_base" for col, format in compare_columns.items()})
    compare_df = compare_df.rename(
        {col: f"{col}_compare" for col, format in compare_columns.items()}
    )
    compare_df = base_df.with_columns([pl.lit(True).alias("in_base")]).join(
        compare_df.with_columns([pl.lit(True).alias("in_compare")]),
        on=join_columns,
        how=how_join,
        coalesce=coalesce,
        validate=validate,
    )
    return compare_df.with_columns(
        [
            get_equality_check(equality_check, resolution, col, format).alias(f"{col}_has_diff")
            for col, format in compare_columns.items()
        ]
    )


def summarise_value_difference(meta: ComparisonMetadata) -> pl.DataFrame:
    value_differences = get_column_value_differences(meta)
    final_df = (
        value_differences.group_by(["variable"])
        .agg(pl.sum("has_diff"))
        .sort("variable", descending=False)
        .rename({"variable": "Value Differences", "has_diff": "Count"})
    )
    total_value_comparisons = value_differences.select(
        pl.lit("Total Value Comparisons").alias("Value Differences"),
        pl.len().alias("Count"),
        pl.lit(100.0).alias("Percentage"),
    )
    value_comparisons = (
        total_value_comparisons.filter(pl.col("Value Differences") == "Total Value Comparisons")
        .select("Count")
        .item()
    )
    total_differences = final_df.select(
        pl.lit("Total Value Differences").alias("Value Differences"),
        pl.sum("Count").alias("Count"),
        (pl.sum("Count") / pl.lit(0.01 * value_comparisons)).alias("Percentage"),
    )
    columns_compared = final_df.select(pl.len().alias("Count")).item()
    value_comparisons_per_column = value_comparisons / columns_compared
    final_df_with_percentages = final_df.with_columns(
        (pl.col("Count") / pl.lit(0.01 * value_comparisons_per_column)).alias("Percentage")
    )
    final_df2 = pl.concat(
        [
            total_differences,
            final_df_with_percentages,
        ]
    )
    if meta.hide_empty_stats:
        final_df2 = final_df2.filter(pl.col("Count") > 0)
    return final_df2


def column_value_differences(
    join_columns: List[str], compare_column: str, combined_tables: pl.LazyFrame
) -> pl.LazyFrame:
    final = combined_tables.filter(f"{compare_column}_has_diff").select(
        [pl.lit(compare_column).alias("Compare Column")]
        + join_columns
        + [
            pl.col(f"{compare_column}_base").cast(pl.Utf8).alias("base"),
            pl.col(f"{compare_column}_compare").cast(pl.Utf8).alias("compare"),
        ]
    )
    return final


def get_columns_to_compare(
    meta: ComparisonMetadata,
) -> Dict[str, Union[pl.DataType, DataTypeClass]]:
    # if schema_comparison:
    #    return ["format"]
    columns_to_exclude: List[str] = []
    if not meta.schema_comparison:
        columns_to_exclude.extend(
            get_schema_comparison(meta).select(pl.col("column")).to_series(0).to_list()
        )

    return {
        col: format
        for col, format in meta.base_df.collect().schema.items()
        if col not in meta.join_columns
        and col not in columns_to_exclude
        and col in meta.compare_df.collect_schema().names()
    }


def get_column_value_differences(meta: ComparisonMetadata) -> pl.DataFrame:
    how_join: Literal["inner", "full"] = "inner"
    coalesce: bool = False
    if meta.schema_comparison:
        how_join = "full"
        coalesce = True
    compare_columns = get_columns_to_compare(meta)
    # if len(compare_columns) == 0:
    combined_tables = get_combined_tables(
        meta.join_columns,
        meta.base_df,
        meta.compare_df,
        compare_columns,
        meta.equality_check,
        resolution=meta.resolution,
        coalesce=coalesce,
        how_join=how_join,
        validate=meta.validate,
    )
    temp = combined_tables.with_columns(
        [
            pl.struct(
                base=f"{col}_base",
                compare=f"{col}_compare",
                has_diff=f"{col}_has_diff",
            )
            .struct.rename_fields([meta.base_alias, meta.compare_alias, "has_diff"])
            .alias(col)
            .struct.json_encode()
            for col, format in compare_columns.items()
        ]
    )

    melted_df = temp.unpivot(
        index=meta.join_columns,
        on=[col for col, format in compare_columns.items()],
    )
    # melted_df = melted_df.with_columns(pl.col("value").str.json_decode(dtype).alias("value"))

    if convert_to_dataframe(melted_df).height > 0 and len(compare_columns) > 0:
        melted_df = (
            melted_df.with_columns(
                pl.col("value").str.json_path_match(f"$.{meta.base_alias}").alias(meta.base_alias),
                pl.col("value")
                .str.json_path_match(f"$.{meta.compare_alias}")
                .alias(meta.compare_alias),
                pl.col("value").str.json_path_match("$.has_diff").alias("has_diff"),
            )
            .drop(["value"])
            .with_columns(pl.col("has_diff").replace_strict({"false": False, "true": True}))
        )
    else:
        melted_df = melted_df.with_columns(pl.lit(False).alias("has_diff"))

    return melted_df.collect()


def get_column_value_differences_filtered(meta: ComparisonMetadata) -> pl.DataFrame:
    df = get_column_value_differences(meta)
    filtered_df = df.filter(pl.col("has_diff")).drop("has_diff")
    if meta.sample_limit is not None:
        filtered_df = (
            filtered_df.with_columns(pl.lit(1).alias("ones"))
            .with_columns(pl.col("ones").cum_sum().over("variable").alias("rows_sample_number"))
            .filter(pl.col("rows_sample_number") <= pl.lit(meta.sample_limit))
            .drop("ones", "rows_sample_number")
        )
    return filtered_df


def get_schema_comparison(meta: ComparisonMetadata) -> pl.DataFrame:
    base_df_schema = pl.LazyFrame(
        {
            "column": meta.base_df.collect().schema.keys(),
            "format": [str(val) for val in meta.base_df.collect().schema.values()],
        }
    )
    compare_df_schema = pl.LazyFrame(
        {
            "column": meta.compare_df.collect().schema.keys(),
            "format": [str(val) for val in meta.compare_df.collect().schema.values()],
        }
    )
    return get_column_value_differences_filtered(
        ComparisonMetadata(
            join_columns=["column"],
            base_df=base_df_schema,
            compare_df=compare_df_schema,
            streaming=True,
            resolution=None,
            equality_check=None,
            sample_limit=None,
            base_alias=f"{meta.base_alias}_format",
            compare_alias=f"{meta.compare_alias}_format",
            schema_comparison=True,
            hide_empty_stats=False,
            validate="1:1",
        )
    ).drop("variable")


def summarise_column_differences(meta: ComparisonMetadata) -> pl.LazyFrame:
    schema_comparison = get_schema_comparison(meta)
    schema_differences = (
        schema_comparison.filter(
            pl.col(f"{meta.base_alias}_format").is_not_null()
            & pl.col(f"{meta.compare_alias}_format").is_not_null()
            & (pl.col(f"{meta.base_alias}_format") != pl.col(f"{meta.compare_alias}_format"))
        )
        .select(pl.len().alias("Count"))
        .item()
    )
    final_df = pl.LazyFrame(
        {
            "Statistic": [
                f"Columns in {meta.base_alias}",
                f"Columns in {meta.compare_alias}",
                f"Columns in {meta.base_alias} and {meta.compare_alias}",
                f"Columns only in {meta.base_alias}",
                f"Columns only in {meta.compare_alias}",
                "Columns with schema differences",
            ],
            "Count": [
                len(meta.base_df.collect().schema.keys()),
                len(meta.compare_df.collect().schema.keys()),
                len(
                    [col for col in meta.compare_df.collect().schema.keys() if col in meta.base_df]
                ),
                len(
                    [
                        col
                        for col in meta.base_df.collect().schema.keys()
                        if col not in meta.compare_df.collect().schema.keys()
                    ]
                ),
                len(
                    [
                        col
                        for col in meta.compare_df.collect().schema.keys()
                        if col not in meta.base_df.collect().schema.keys()
                    ]
                ),
                schema_differences,
            ],
        }
    )
    if meta.hide_empty_stats:
        final_df = final_df.filter(pl.col("Count") > 0)
    return final_df


class FuncAppend:
    """
    When initialised FuncAppend will take a function as an argument. This function will be called for a value whenevr a value is supplied to the append method.
    """

    def __init__(self, func: Union[Callable[[str], None], None] = None):
        self.special_list: List[str] = []
        self.func = func

    def __repr__(self) -> str:
        return str("\n".join(self.special_list))

    def append(self, value: str) -> None:
        if self.func is not None:
            self.func(value)
        self.special_list.append(value)


class compare:
    """
    Compare two dataframes.
    """

    def __init__(
        self,
        join_columns: Union[List[str], None],
        base_df: Union[pl.LazyFrame, pl.DataFrame],
        compare_df: Union[pl.LazyFrame, pl.DataFrame],
        streaming: bool = False,
        resolution: Union[float, None] = None,
        equality_check: Union[
            Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None
        ] = None,
        sample_limit: int = 5,
        base_alias: str = "base",
        compare_alias: str = "compare",
        hide_empty_stats: bool = False,
        validate: Literal["m:m", "m:1", "1:m", "1:1"] = "m:m",
    ):
        """
        Initialize a new instance of the compare class.

        Parameters:
            join_columns (Union[List[str], None]): Columns to be joined on for the comparison. If "None" is supplied then the row number for each dataframe will be used instead.
            base_df (Union[pl.LazyFrame, pl.DataFrame]): The base dataframe for comparison.
            compare_df (Union[pl.LazyFrame, pl.DataFrame]): The dataframe that will be compared with the base dataframe.
            streaming (bool): Whether the comparison will return LazyFrames (defaults to False).
            resolution (Union[float, None]): The resolution for comparison. Applies to numeric values only. If the difference between two values is greater than the resolution then the values are considered to be unequal.
            equality_check (Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None]): The function to check equality.
            sample_limit (int): The number of rows to sample from the comparison. This only applies to methods that return a sample.
            base_alias (str): The alias for the base dataframe. This will be displayed in the final result.
            compare_alias (str): The alias for the dataframe to be compared. This will be displayed in the final result.
            hide_empty_stats (bool): Whether to hide empty statistics. Comparison statistics where there are zero differences will be excluded from the result.
            validate (str): Checks if join is of specified type {‘m:m’, ‘m:1’, ‘1:m’, ‘1:1’ }.
        """
        base_lazy_df = convert_to_lazyframe(base_df)
        compare_lazy_df = convert_to_lazyframe(compare_df)
        if join_columns is None or join_columns == []:
            base_lazy_df = base_lazy_df.with_row_index(offset=1).rename({"index": "row_number"})
            compare_lazy_df = compare_lazy_df.with_row_index(offset=1).rename(
                {"index": "row_number"}
            )
            join_columns = ["row_number"]

        self._comparison_metadata = ComparisonMetadata(
            join_columns,
            base_lazy_df,
            compare_lazy_df,
            streaming,
            resolution,
            equality_check,
            sample_limit,
            base_alias,
            compare_alias,
            False,
            hide_empty_stats,
            validate,
        )
        self._created_frames: Dict[str, Union[pl.DataFrame, pl.LazyFrame]] = {}

    def _get_or_create(
        self,
        func: Callable[[ComparisonMetadata], Union[pl.LazyFrame, pl.DataFrame]],
        *args: ComparisonMetadata,
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get or create a dataframe based on the given function and arguments.

        Parameters:
            func (Callable[[ComparisonMetadata], Union[pl.LazyFrame, pl.DataFrame]]): The function to get or create the dataframe.
            *args (ComparisonMetadata): The arguments for the function.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The dataframe.
        """
        if func.__name__ not in self._created_frames:
            self._created_frames[func.__name__] = set_df_type(
                func(*args), streaming=self._comparison_metadata.streaming
            )
        return self._created_frames[func.__name__]

    def schemas_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of schema differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of schema differences.
        """
        return self._get_or_create(summarise_column_differences, self._comparison_metadata)

    def rows_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of row differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of row differences.
        """
        return self._get_or_create(get_row_comparison_summary, self._comparison_metadata)

    def rows_sample(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a sample of row differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The sample of row differences.
        """
        return self._get_or_create(get_row_differences, self._comparison_metadata)

    def values_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of value differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of value differences.
        """
        return self._get_or_create(summarise_value_difference, self._comparison_metadata).select(
            "Value Differences", pl.col("Count").cast(pl.Int64).alias("Count"), "Percentage"
        )

    def schemas_sample(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a sample of schema differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The sample of schema differences.
        """
        return self._get_or_create(get_schema_comparison, self._comparison_metadata)

    def values_sample(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a sample of value differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The sample of the value differences.
        """
        return self._get_or_create(get_column_value_differences_filtered, self._comparison_metadata)

    def is_equal(self) -> bool:
        """
        Check if the two dataframes are unequal based on schema, rows, and values.

        Returns:
            bool: True if the dataframes are unequal, False otherwise.
        """
        if not self.is_schemas_equal():
            return False
        if not self.is_rows_equal():
            return False
        if not self.is_values_equal():
            return False
        return True

    def is_schemas_equal(self) -> bool:
        """
        Check if the schemas of the two dataframes are unequal.

        Returns:
            bool: True if the schemas are unequal, False otherwise.
        """
        return convert_to_dataframe(self.schemas_sample()).height == 0

    def is_rows_equal(self) -> bool:
        """
        Check if the rows of the two dataframes are unequal.

        Returns:
            bool: True if the rows are unequal, False otherwise.
        """
        return convert_to_dataframe(self.rows_sample()).height == 0

    def is_values_equal(self) -> bool:
        """
        Check if the values of the two dataframes are unequal.

        Returns:
            bool: True if the values are unequal, False otherwise.
        """
        return convert_to_dataframe(set_df_type(self.values_sample(), streaming=False)).height == 0

    def summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of all differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of all differences.
        """
        return pl.concat(  # type: ignore
            [
                self.schemas_summary(),
                self.rows_summary(),
                self.values_summary()
                .select("Value Differences", "Count")
                .rename({"Value Differences": "Statistic"})
                .filter(pl.col("Statistic") == pl.lit("Total Value Differences")),
                self.values_summary()
                .rename({"Value Differences": "Statistic"})
                .filter(pl.col("Statistic") != pl.lit("Total Value Differences"))
                .select(
                    pl.concat_str(pl.lit("Value diffs Col:"), pl.col("Statistic")).alias(
                        "Statistic"
                    ),
                    "Count",
                ),
            ]
        )

    def equals_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of schema and row differences between two dataframes.
        This is used for a short summary in the report when two dataframes are equal.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of all differences.
        """
        return pl.concat(  # type: ignore
            [
                self.schemas_summary().filter(
                    (pl.col("Statistic") == "Columns in base")
                    | (pl.col("Statistic") == "Columns in compare")
                ),
                self.rows_summary().filter(
                    (pl.col("Statistic") == "Rows in base")
                    | (pl.col("Statistic") == "Rows in compare")
                ),
            ]
        )

    def _value_comparison_columns_exist(self) -> bool:
        """
        Checks if there are any columns to be used in the value comparison.

        Returns:
            bool: True if columns for value comparison exist. False if no columns exist for the value comparison.
        """
        return len(get_columns_to_compare(self._comparison_metadata)) > 0

    def report(self, print: Union[Callable[[str], None], None] = None) -> Union[FuncAppend, None]:
        """
        Generate a report of the comparison results.

        meters:
            print (Union[Callable[[str], None], None]): The function to print the report for example you could supply the standard 'print' or logger.info if you want the output to be logged instead of printed.

        Returns:
            Union[FuncAppend, None]: The report or None if the dataframes are equal.
        """
        combined = FuncAppend(print)
        combined.append(80 * "-")
        combined.append("COMPARISON REPORT")
        combined.append(80 * "-")
        if self.is_equal():
            combined.append("Tables are exactly equal.")
            combined.append(f"\nSUMMARY:\n{self.equals_summary()}")
            return combined
        if not self.is_schemas_equal():
            combined.append(
                f"\nSCHEMA DIFFERENCES:\n{self.schemas_summary()}\n{self.schemas_sample()}"
            )
        else:
            combined.append("No Schema differences found.")
        combined.append(80 * "-")
        if not self.is_rows_equal():
            combined.append(f"\nROW DIFFERENCES:\n{self.rows_summary()}\n{self.rows_sample()}")
        else:
            combined.append("No Row differences found (when joining by the supplied join_columns).")
        combined.append(80 * "-")
        if not self._value_comparison_columns_exist():
            combined.append("No columns to compare.")
        elif self.is_values_equal():
            combined.append("No Column Value differences found.")
        elif not self.is_values_equal():
            combined.append(
                f"\nVALUE DIFFERENCES:\n{self.values_summary()}\n{self.values_sample()}"
            )
        combined.append(80 * "-")
        combined.append("End of Report")
        combined.append(80 * "-")
        return combined
