from typing import Literal, Callable, List, Union, Dict
from dataclasses import dataclass
import narwhals as nw
from narwhals.typing import IntoFrameT


@dataclass
class ComparisonMetadata:
    """Class for holding the (meta)data used to generate the comparison dataframes."""

    join_columns: List[str]
    base_df: IntoFrameT
    compare_df: IntoFrameT
    streaming: bool
    resolution: Union[float, None]
    equality_check: Union[Callable[[str, Union[nw.typing.DTypes]], nw.Expr], None]
    sample_limit: Union[int, None]
    base_alias: str
    compare_alias: str
    schema_comparison: bool
    hide_empty_stats: bool
    validate: Literal["m:m", "m:1", "1:m", "1:1"]


def convert_to_dataframe(df: Union[IntoFrameT, IntoFrameT]) -> IntoFrameT:
    df = df.collect()
    return df


def convert_to_lazyframe(df: Union[IntoFrameT, IntoFrameT]) -> IntoFrameT:
    df = df.lazy()
    return df


def set_df_type(
    df: Union[IntoFrameT, IntoFrameT], streaming: bool = False
) -> Union[IntoFrameT, IntoFrameT]:
    if streaming:
        return convert_to_lazyframe(df)
    if not streaming:
        return convert_to_dataframe(df)


def get_uncertain_row_count(df: IntoFrameT) -> int:
    df_solid = convert_to_dataframe(df).select(nw.col("Count"))
    if df_solid.height > 0:
        row_count: int = df_solid.item()
        return row_count
    else:
        return 0


def get_row_comparison_summary(meta: ComparisonMetadata) -> IntoFrameT:
    combined_table = meta.base_df.select(meta.join_columns + [nw.lit(True).alias("in_base")]).join(
        meta.compare_df.select(meta.join_columns + [nw.lit(True).alias("in_compare")]),
        on=meta.join_columns,
        how="cross",
        coalesce=True,
        validate=meta.validate,
    )
    grouped_rows = (
        combined_table.select(meta.join_columns + ["in_base", "in_compare"])
        .group_by(["in_base", "in_compare"])
        .agg(nw.len().alias("Count"))
    )
    base_only_rows = get_uncertain_row_count(
        grouped_rows.filter(nw.col("in_base") & nw.col("in_compare").is_null())
    )
    compare_only_rows = get_uncertain_row_count(
        grouped_rows.filter(nw.col("in_base").is_null() & nw.col("in_compare"))
    )
    shared_rows = get_uncertain_row_count(
        grouped_rows.filter(
            nw.col("in_base")
            & nw.col("in_base").is_not_null()
            & nw.col("in_compare")
            & nw.col("in_compare").is_not_null()
        )
    )
    final_df = (
        nw.from_dict(
            {
                "Rows in base": [shared_rows + base_only_rows],
                "Rows in compare": [shared_rows + compare_only_rows],
                "Rows only in base": [base_only_rows],
                "Rows only in compare": [compare_only_rows],
                "Rows in base and compare": [shared_rows],
            },
            backend="polars",
        )
        .transpose(include_header=True, column_names=["Col Differences"])
        .rename({"column": "Statistic", "Col Differences": "Count"})
    )
    if meta.hide_empty_stats:
        final_df = final_df.filter(nw.col("Count") > 0)
    return final_df


def get_base_only_rows(
    join_columns: List[str],
    base_df: IntoFrameT,
    compare_df: IntoFrameT,
) -> IntoFrameT:
    combined_table = base_df.select(join_columns).join(
        compare_df.select(join_columns),
        on=join_columns,
        how="anti",
    )
    return combined_table.select(join_columns + [nw.lit("in base only").alias("status")]).melt(
        id_vars=join_columns, value_vars=["status"]
    )


def get_compare_only_rows(
    join_columns: List[str],
    base_df: IntoFrameT,
    compare_df: IntoFrameT,
) -> IntoFrameT:
    combined_table = compare_df.select(join_columns).join(
        base_df.select(join_columns), on=join_columns, how="anti"
    )
    return combined_table.select(join_columns + [nw.lit("in compare only").alias("status")]).melt(
        id_vars=join_columns, value_vars=["status"]
    )


def get_row_differences(meta: ComparisonMetadata) -> IntoFrameT:
    base_only_rows = get_base_only_rows(
        meta.join_columns, meta.base_df, meta.compare_df
    ).with_row_index()
    compare_only_rows = get_compare_only_rows(
        meta.join_columns, meta.base_df, meta.compare_df
    ).with_row_index()
    if meta.sample_limit is not None:
        base_only_rows = base_only_rows.limit(meta.sample_limit)
        compare_only_rows = compare_only_rows.limit(meta.sample_limit)
    return (
        nw.concat(
            [
                base_only_rows,
                compare_only_rows,
            ]
        )
        .sort("index")
        .drop("index")
    )


def get_equality_check(
    equality_check: Union[Callable[[str, Union[nw.typing.DTypes]], nw.Expr], None],
    resolution: Union[float, None],
    col: str,
    format: Union[nw.typing.DTypes],
) -> nw.Expr:
    def default_equality_check(col: str, format: Union[nw.typing.DTypes]) -> nw.Expr:
        return (
            (nw.col(f"{col}_base") != nw.col(f"{col}_compare"))
            | (nw.col(f"{col}_base").is_null() & ~nw.col(f"{col}_compare").is_null())
            | (~nw.col(f"{col}_base").is_null() & nw.col(f"{col}_compare").is_null())
        )

    def ignore_numeric_differences_equality_check(
        col: str, format: Union[nw.typing.DTypes]
    ) -> nw.Expr:
        return (
            ((nw.col(f"{col}_base") - nw.col(f"{col}_compare")).abs() > resolution)
            | (nw.col(f"{col}_base").is_null() & ~nw.col(f"{col}_compare").is_null())
            | (~nw.col(f"{col}_base").is_null() & nw.col(f"{col}_compare").is_null())
        )

    if resolution is not None and format in [
        nw.Float32,
        nw.Float64,
        nw.Decimal,
        nw.Int8,
        nw.Int16,
        nw.Int32,
        nw.Int64,
        nw.UInt8,
        nw.UInt16,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
    ]:
        return ignore_numeric_differences_equality_check(col, format)
    if equality_check is not None:
        return equality_check(col, format)
    return default_equality_check(col, format)


def get_combined_tables(
    join_columns: List[str],
    base_df: IntoFrameT,
    compare_df: IntoFrameT,
    compare_columns: Dict[str, nw.typing.DTypes],
    equality_check: Union[Callable[[str, nw.typing.DTypes], nw.Expr], None],
    coalesce: bool,
    how_join: Literal["inner", "cross"] = "inner",
    resolution: Union[float, None] = None,
    validate: Literal["m:m", "m:1", "1:m", "1:1"] = "1:1",
) -> IntoFrameT:
    base_df = base_df.rename({col: f"{col}_base" for col, format in compare_columns.items()})
    compare_df = compare_df.rename(
        {col: f"{col}_compare" for col, format in compare_columns.items()}
    )
    compare_df = (
        base_df.with_columns([nw.lit(True).alias("in_base")])
        .join(
            compare_df.with_columns([nw.lit(True).alias("in_compare")]),
            on=join_columns,
        )
        .filter(nw.col("in_base") & nw.col("in_compare"))
    )
    return compare_df.with_columns(
        [
            get_equality_check(equality_check, resolution, col, format).alias(f"{col}_has_diff")
            for col, format in compare_columns.items()
        ]
    )


def summarise_value_difference(meta: ComparisonMetadata) -> IntoFrameT:
    value_differences = get_column_value_differences(meta)
    final_df = (
        value_differences.group_by(["variable"])
        .agg(nw.sum("has_diff"))
        .sort("variable", descending=False)
        .rename({"variable": "Value Differences", "has_diff": "Count"})
    )
    total_value_comparisons = value_differences.select(
        nw.lit("Total Value Comparisons").alias("Value Differences"),
        nw.len().alias("Count"),
        nw.lit(100.0).alias("Percentage"),
    )
    value_comparisons = (
        total_value_comparisons.filter(nw.col("Value Differences") == "Total Value Comparisons")
        .select("Count")
        .item()
    )
    total_differences = final_df.select(
        nw.lit("Total Value Differences").alias("Value Differences"),
        nw.sum("Count").alias("Count"),
        (nw.sum("Count") / nw.lit(0.01 * value_comparisons)).alias("Percentage"),
    )
    columns_compared = final_df.select(nw.len().alias("Count")).item()
    value_comparisons_per_column = value_comparisons / columns_compared
    final_df_with_percentages = final_df.with_columns(
        (nw.col("Count") / nw.lit(0.01 * value_comparisons_per_column)).alias("Percentage")
    )
    final_df2 = nw.concat(
        [
            total_differences,
            final_df_with_percentages,
        ]
    )
    if meta.hide_empty_stats:
        final_df2 = final_df2.filter(nw.col("Count") > 0)
    return final_df2


def column_value_differences(
    join_columns: List[str], compare_column: str, combined_tables: IntoFrameT
) -> IntoFrameT:
    final = combined_tables.filter(f"{compare_column}_has_diff").select(
        [nw.lit(compare_column).alias("Compare Column")]
        + join_columns
        + [
            nw.col(f"{compare_column}_base").cast(nw.Utf8).alias("base"),
            nw.col(f"{compare_column}_compare").cast(nw.Utf8).alias("compare"),
        ]
    )
    return final


def get_columns_to_compare(
    meta: ComparisonMetadata,
) -> Dict[str, Union[nw.typing.DTypes]]:
    # if schema_comparison:
    #    return ["format"]
    columns_to_exclude: List[str] = []
    if not meta.schema_comparison:
        columns_to_exclude.extend(
            get_schema_comparison(meta).select(nw.col("column")).to_series(0).to_list()
        )

    return {
        col: format
        for col, format in meta.base_df.schema.items()
        if col not in meta.join_columns
        and col not in columns_to_exclude
        and col in meta.compare_df.columns
    }


def get_column_value_differences(meta: ComparisonMetadata) -> IntoFrameT:
    how_join: Literal["inner", "cross"] = "inner"
    coalesce: bool = False
    if meta.schema_comparison:
        how_join = "cross"
        coalesce = True
    compare_columns = get_columns_to_compare(meta)
    if len(compare_columns) == 0:
        raise Exception(
            "There are no columns to compare the value of. Please check the columns in the base and compare datasets as well as the join columns that have been supplied."
        )
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
            nw.Struct(
                {
                    meta.base_alias: f"{col}_base",
                    meta.compare_alias: f"{col}_compare",
                    "has_diff": f"{col}_has_diff",
                }
            )
            for col, format in compare_columns.items()
        ]
    )

    melted_df = temp.melt(
        id_vars=meta.join_columns,
        value_vars=[col for col, format in compare_columns.items()],
    ).with_columns(nw.col("value").str.json_decode().alias("value"))
    if convert_to_dataframe(melted_df).height > 0:
        melted_df = melted_df.unnest("value")
    else:
        melted_df = melted_df.with_columns(nw.lit(False).alias("has_diff"))

    return melted_df.collect()


def get_column_value_differences_filtered(meta: ComparisonMetadata) -> IntoFrameT:
    df = get_column_value_differences(meta)
    filtered_df = df.filter(nw.col("has_diff")).drop("has_diff")
    if meta.sample_limit is not None:
        filtered_df = (
            filtered_df.with_columns(nw.lit(1).alias("ones"))
            .with_columns(nw.col("ones").cum_sum().over("variable").alias("rows_sample_number"))
            .filter(nw.col("rows_sample_number") <= nw.lit(meta.sample_limit))
            .drop("ones", "rows_sample_number")
        )
    return filtered_df


def get_schema_comparison(meta: ComparisonMetadata) -> IntoFrameT:
    base_df_schema = nw.from_dict(
        {
            "column": meta.base_df.collect().schema.keys(),
            "format": [str(val) for val in meta.base_df.collect().schema.values()],
        },
        backend="polars",
    )
    compare_df_schema = nw.from_dict(
        {
            "column": meta.compare_df.collect().schema.keys(),
            "format": [str(val) for val in meta.compare_df.collect().schema.values()],
        },
        backend="polars",
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


def summarise_column_differences(meta: ComparisonMetadata) -> IntoFrameT:
    schema_comparison = get_schema_comparison(meta)
    schema_differences = (
        schema_comparison.filter(
            nw.col(f"{meta.base_alias}_format").is_not_null()
            & nw.col(f"{meta.compare_alias}_format").is_not_null()
            & (nw.col(f"{meta.base_alias}_format") != nw.col(f"{meta.compare_alias}_format"))
        )
        .select(nw.len().alias("Count"))
        .item()
    )
    final_df = nw.from_dict(
        {
            "Statistic": [
                "Columns in base",
                "Columns in compare",
                "Columns in base and compare",
                "Columns only in base",
                "Columns only in compare",
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
        final_df = final_df.filter(nw.col("Count") > 0)
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
        base_df: Union[IntoFrameT, IntoFrameT],
        compare_df: Union[IntoFrameT, IntoFrameT],
        streaming: bool = False,
        resolution: Union[float, None] = None,
        equality_check: Union[Callable[[str, Union[nw.typing.DTypes]], nw.Expr], None] = None,
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
            base_df (Union[IntoFrameT, IntoFrameT]): The base dataframe for comparison.
            compare_df (Union[IntoFrameT, IntoFrameT]): The dataframe that will be compared with the base dataframe.
            streaming (bool): Whether the comparison will return IntoFrameTs (defaults to False).
            resolution (Union[float, None]): The resolution for comparison. Applies to numeric values only. If the difference between two values is greater than the resolution then the values are considered to be unequal.
            equality_check (Union[Callable[[str, Union[nw.typing.DTypes]], nw.Expr], None]): The function to check equality.
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
        self._created_frames: Dict[str, Union[IntoFrameT, IntoFrameT]] = {}

    def _get_or_create(
        self,
        func: Callable[[ComparisonMetadata], Union[IntoFrameT, IntoFrameT]],
        *args: ComparisonMetadata,
    ) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get or create a dataframe based on the given function and arguments.

        Parameters:
            func (Callable[[ComparisonMetadata], Union[IntoFrameT, IntoFrameT]]): The function to get or create the dataframe.
            *args (ComparisonMetadata): The arguments for the function.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The dataframe.
        """
        if func.__name__ not in self._created_frames:
            self._created_frames[func.__name__] = set_df_type(
                func(*args), streaming=self._comparison_metadata.streaming
            )
        return self._created_frames[func.__name__]

    def schemas_summary(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a summary of schema differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The summary of schema differences.
        """
        return self._get_or_create(summarise_column_differences, self._comparison_metadata)

    def rows_summary(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a summary of row differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The summary of row differences.
        """
        return self._get_or_create(get_row_comparison_summary, self._comparison_metadata)

    def rows_sample(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a sample of row differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The sample of row differences.
        """
        return self._get_or_create(get_row_differences, self._comparison_metadata)

    def values_summary(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a summary of value differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The summary of value differences.
        """
        return self._get_or_create(summarise_value_difference, self._comparison_metadata).select(
            "Value Differences", nw.col("Count").cast(nw.Int64).alias("Count"), "Percentage"
        )

    def schemas_sample(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a sample of schema differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The sample of schema differences.
        """
        return self._get_or_create(get_schema_comparison, self._comparison_metadata)

    def values_sample(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a sample of value differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The sample of the value differences.
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

    def summary(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a summary of all differences between the two dataframes.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The summary of all differences.
        """
        return nw.concat(  # type: ignore
            [
                self.schemas_summary(),
                self.rows_summary(),
                self.values_summary()
                .select("Value Differences", "Count")
                .rename({"Value Differences": "Statistic"})
                .filter(nw.col("Statistic") == nw.lit("Total Value Differences")),
                self.values_summary()
                .rename({"Value Differences": "Statistic"})
                .filter(nw.col("Statistic") != nw.lit("Total Value Differences"))
                .select(
                    nw.concat_str(nw.lit("Value diffs Col:"), nw.col("Statistic")).alias(
                        "Statistic"
                    ),
                    "Count",
                ),
            ]
        )

    def equals_summary(self) -> Union[IntoFrameT, IntoFrameT]:
        """
        Get a summary of schema and row differences between two dataframes.
        This is used for a short summary in the report when two dataframes are equal.

        Returns:
            Union[IntoFrameT, IntoFrameT]: The summary of all differences.
        """
        return nw.concat(  # type: ignore
            [
                self.schemas_summary().filter(
                    (nw.col("Statistic") == "Columns in base")
                    | (nw.col("Statistic") == "Columns in compare")
                ),
                self.rows_summary().filter(
                    (nw.col("Statistic") == "Rows in base")
                    | (nw.col("Statistic") == "Rows in compare")
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
