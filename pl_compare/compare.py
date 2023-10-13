import polars as pl
from polars.datatypes.classes import DataTypeClass
from polars.type_aliases import PolarsType
import types
from typing import Literal, Callable, List, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class ComparisonMetadata:
    """Class for holding the (meta)data used to generate the comparison dataframes."""

    id_columns: List[str]
    base_df: pl.LazyFrame
    compare_df: pl.LazyFrame
    streaming: bool
    equality_resolution: Union[float, None]
    equality_check: Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None]
    sample_limit: Union[int, None]
    base_alias: str
    compare_alias: str
    schema_comparison: bool
    hide_empty_stats: bool


def get_duplicates(
    df: Union[pl.LazyFrame, pl.DataFrame], id_columns: List[str]
) -> Union[pl.LazyFrame, pl.DataFrame]:
    ctx = pl.SQLContext(input_table=df)
    query = f"""SELECT {', '.join(id_columns)}, count(*) AS row_count 
                FROM input_table GROUP BY {", ".join(id_columns)} 
                HAVING row_count>1"""
    return ctx.execute(query)


def duplicates_summary(df: Union[pl.LazyFrame, pl.DataFrame]) -> Union[pl.LazyFrame, pl.DataFrame]:
    ctx = pl.SQLContext(input_table=df)
    query = """SELECT  (sum(row_count)-count(*)) AS TOTAL_DUPLICATE_ROWS, 
                   (max(row_count)-1) AS MAXIMUM_DUPLICATES_FOR_AN_ID 
               FROM input_table"""
    return ctx.execute(query)


def duplicate_examples(df: Union[pl.LazyFrame, pl.DataFrame]) -> Union[pl.LazyFrame, pl.DataFrame]:
    ctx = pl.SQLContext(input_table=df)
    query = """SELECT * EXCLUDE(row_count), (row_count-1) AS DUPLICATES FROM input_table"""
    return ctx.execute(query)


def convert_to_dataframe(df: Union[pl.LazyFrame, pl.DataFrame]) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=True)
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
    df_solid = convert_to_dataframe(df.select("count"))
    if df_solid.height > 0:
        row_count: int = df_solid.item()
        return row_count
    else:
        return 0


def get_row_comparison_summary(meta: ComparisonMetadata) -> pl.DataFrame:
    combined_table = meta.base_df.select(meta.id_columns + [pl.lit(True).alias("in_base")]).join(
        meta.compare_df.select(meta.id_columns + [pl.lit(True).alias("in_compare")]),
        on=meta.id_columns,
        how="outer",
    )
    grouped_rows = (
        combined_table.select(meta.id_columns + ["in_base", "in_compare"])
        .group_by(["in_base", "in_compare"])
        .agg(pl.count())
    )

    base_only_rows = get_uncertain_row_count(
        grouped_rows.filter(pl.col("in_base") & pl.col("in_compare").is_null())
    )
    compare_only_rows = get_uncertain_row_count(
        grouped_rows.filter(pl.col("in_base").is_null() & pl.col("in_compare"))
    )
    shared_rows = get_uncertain_row_count(
        grouped_rows.filter(pl.col("in_base") & pl.col("in_compare"))
    )
    final_df = (
        pl.DataFrame(
            {
                "Rows in base": [shared_rows + base_only_rows],
                "Rows in compare": [shared_rows + compare_only_rows],
                "Rows only in base": [base_only_rows],
                "Rows only in compare": [compare_only_rows],
                "Rows in base and compare": [shared_rows],
            }
        )
        .transpose(include_header=True, column_names=["Col Differences"])
        .rename({"column": "Statistic", "Col Differences": "Count"})
    )
    if meta.hide_empty_stats:
        final_df = final_df.filter(pl.col("Count") > 0)
    return final_df


def get_base_only_rows(
    id_columns: List[str],
    base_df: pl.LazyFrame,
    compare_df: pl.LazyFrame,
) -> pl.LazyFrame:
    combined_table = base_df.select(id_columns).join(
        compare_df.select(id_columns), on=id_columns, how="anti"
    )
    return combined_table.select(id_columns + [pl.lit("in base only").alias("status")]).melt(
        id_vars=id_columns, value_vars=["status"]
    )


def get_compare_only_rows(
    id_columns: List[str],
    base_df: pl.LazyFrame,
    compare_df: pl.LazyFrame,
) -> pl.LazyFrame:
    combined_table = compare_df.select(id_columns).join(
        base_df.select(id_columns), on=id_columns, how="anti"
    )
    return combined_table.select(id_columns + [pl.lit("in compare only").alias("status")]).melt(
        id_vars=id_columns, value_vars=["status"]
    )


def get_row_differences(meta: ComparisonMetadata) -> pl.LazyFrame:
    base_only_rows = get_base_only_rows(
        meta.id_columns, meta.base_df, meta.compare_df
    ).with_row_count()
    compare_only_rows = get_compare_only_rows(
        meta.id_columns, meta.base_df, meta.compare_df
    ).with_row_count()
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
        .sort("row_nr")
        .drop("row_nr")
    )


def get_equality_check(
    equality_check: Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None],
    equality_resolution: Union[float, None],
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
            ((pl.col(f"{col}_base") - pl.col(f"{col}_compare")).abs() > equality_resolution)
            | (pl.col(f"{col}_base").is_null() & ~pl.col(f"{col}_compare").is_null())
            | (~pl.col(f"{col}_base").is_null() & pl.col(f"{col}_compare").is_null())
        )

    if equality_check is not None:
        return equality_check(col, format)
    if equality_resolution is not None and format in [
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
    return default_equality_check(col, format)


def get_combined_tables(
    id_columns: List[str],
    base_df: pl.LazyFrame,
    compare_df: pl.LazyFrame,
    compare_columns: Dict[str, Union[DataTypeClass, pl.DataType]],
    equality_check: Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None],
    how_join: Literal["inner", "outer"] = "inner",
    equality_resolution: Union[float, None] = None,
) -> pl.LazyFrame:
    base_df = base_df.rename({col: f"{col}_base" for col, format in compare_columns.items()})
    compare_df = compare_df.rename(
        {col: f"{col}_compare" for col, format in compare_columns.items()}
    )
    compare_df = base_df.with_columns([pl.lit(True).alias("in_base")]).join(
        compare_df.with_columns([pl.lit(True).alias("in_compare")]),
        on=id_columns,
        how=how_join,
    )
    return compare_df.with_columns(
        [
            get_equality_check(equality_check, equality_resolution, col, format).alias(
                f"{col}_has_diff"
            )
            for col, format in compare_columns.items()
        ]
    )


def summarise_value_difference(meta: ComparisonMetadata) -> pl.DataFrame:
    final_df = (
        get_column_value_differences(meta)
        .group_by(["variable"])
        .agg(pl.sum("has_diff"))
        .sort("variable", descending=False)
        .rename({"variable": "Value Differences for Column", "has_diff": "Count"})
    )
    total_differences = final_df.select(
        pl.lit("Total Value Differences").alias("Value Differences for Column"),
        pl.sum("Count").alias("Count"),
    )
    final_df2 = pl.concat(
        [total_differences.collect(streaming=True), final_df.collect(streaming=True)]
    )
    if meta.hide_empty_stats:
        final_df2 = final_df2.filter(pl.col("Count") > 0)
    return final_df2


def column_value_differences(
    id_columns: List[str], compare_column: str, combined_tables: pl.LazyFrame
) -> pl.LazyFrame:
    final = combined_tables.filter(f"{compare_column}_has_diff").select(
        [pl.lit(compare_column).alias("Compare Column")]
        + id_columns
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
            get_schema_comparison(meta)
            .select(pl.col("column"))
            .collect(streaming=True)
            .to_series(0)
            .to_list()
        )

    return {
        col: format
        for col, format in meta.base_df.schema.items()
        if col not in meta.id_columns
        and col not in columns_to_exclude
        and col in meta.compare_df.columns
    }


def get_column_value_differences(meta: ComparisonMetadata) -> pl.LazyFrame:
    how_join: Literal["inner", "outer"] = "inner"
    if meta.schema_comparison:
        how_join = "outer"
    compare_columns = get_columns_to_compare(meta)
    combined_tables = get_combined_tables(
        meta.id_columns,
        meta.base_df,
        meta.compare_df,
        compare_columns,
        meta.equality_check,
        equality_resolution=meta.equality_resolution,
        how_join=how_join,
    )
    melted_df = (
        combined_tables.with_columns(
            [
                pl.struct(
                    base=f"{col}_base",
                    compare=f"{col}_compare",
                    has_diff=f"{col}_has_diff",
                )
                .struct.rename_fields([meta.base_alias, meta.compare_alias, "has_diff"])
                .alias(col)
                for col, format in compare_columns.items()
            ]
        )
        .melt(
            id_vars=meta.id_columns,
            value_vars=[col for col, format in compare_columns.items()],
        )
        .unnest("value")
    )

    return melted_df


def get_column_value_differences_filtered(meta: ComparisonMetadata) -> pl.LazyFrame:
    df = get_column_value_differences(meta)
    filtered_df = df.filter(pl.col("has_diff")).drop("has_diff")
    if meta.sample_limit is not None:
        filtered_df = (
            filtered_df.with_columns(pl.lit(1).alias("ones"))
            .with_columns(pl.col("ones").cumsum().over("variable").alias("row_sample_number"))
            .filter(pl.col("row_sample_number") <= pl.lit(meta.sample_limit))
            .drop("ones", "row_sample_number")
        )
    return filtered_df


def get_schema_comparison(meta: ComparisonMetadata) -> pl.LazyFrame:
    base_df_schema = pl.LazyFrame(
        {
            "column": meta.base_df.schema.keys(),
            "format": [str(val) for val in meta.base_df.schema.values()],
        }
    )
    compare_df_schema = pl.LazyFrame(
        {
            "column": meta.compare_df.schema.keys(),
            "format": [str(val) for val in meta.compare_df.schema.values()],
        }
    )

    return get_column_value_differences_filtered(
        ComparisonMetadata(
            id_columns=["column"],
            base_df=base_df_schema,
            compare_df=compare_df_schema,
            streaming=True,
            equality_resolution=None,
            equality_check=None,
            sample_limit=None,
            base_alias=f"{meta.base_alias}_format",
            compare_alias=f"{meta.compare_alias}_format",
            schema_comparison=True,
            hide_empty_stats=False,
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
        .select(pl.count())
        .collect(streaming=True)
        .item()
    )
    final_df = pl.LazyFrame(
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
                len(meta.base_df.columns),
                len(meta.compare_df.columns),
                len([col for col in meta.compare_df.columns if col in meta.base_df]),
                len([col for col in meta.base_df.columns if col not in meta.compare_df.columns]),
                len([col for col in meta.compare_df.columns if col not in meta.base_df.columns]),
                schema_differences,
            ],
        }
    )
    if meta.hide_empty_stats:
        final_df = final_df.filter(pl.col("Count") > 0)
    return final_df


class FuncAppend:
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
        id_columns: Union[List[str], None],
        base_df: Union[pl.LazyFrame, pl.DataFrame],
        compare_df: Union[pl.LazyFrame, pl.DataFrame],
        streaming: bool = False,
        equality_resolution: Union[float, None] = None,
        equality_check: Union[
            Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None
        ] = None,
        sample_limit: int = 5,
        base_alias: str = "base",
        compare_alias: str = "compare",
        hide_empty_stats: bool = False,
    ):
        """
        Initialize a new instance of the compare class.

        Parameters:
            id_columns (Union[List[str], None]): Columns to be joined on for the comparison. If "None" is supplied then the row number for each dataframe will be used instead.
            base_df (Union[pl.LazyFrame, pl.DataFrame]): The base dataframe for comparison.
            compare_df (Union[pl.LazyFrame, pl.DataFrame]): The dataframe that will be compared with the base dataframe.
            streaming (bool): Whether the comparison will return LazyFrames (defaults to False).
            equality_resolution (Union[float, None]): The equality_resolution for comparison. Applies to numeric values only. If the difference between two values is greater than the equality_resolution then the values are considered to be unequal.
            equality_check (Union[Callable[[str, Union[pl.DataType, DataTypeClass]], pl.Expr], None]): The function to check equality.
            sample_limit (int): The number of rows to sample from the comparison. This only applies to methods that return a sample.
            base_alias (str): The alias for the base dataframe. This will be displayed in the final result.
            compare_alias (str): The alias for the dataframe to be compared. This will be displayed in the final result.
            hide_empty_stats (bool): Whether to hide empty statistics. Comparison statistics where there are zero differences will be excluded from the result.
        """
        base_lazy_df = convert_to_lazyframe(base_df)
        compare_lazy_df = convert_to_lazyframe(compare_df)
        if id_columns is None or id_columns == []:
            base_lazy_df = base_lazy_df.with_row_count(offset=1).rename({"row_nr": "row_number"})
            compare_lazy_df = compare_lazy_df.with_row_count(offset=1).rename(
                {"row_nr": "row_number"}
            )
            id_columns = ["row_number"]

        self._comparison_metadata = ComparisonMetadata(
            id_columns,
            base_lazy_df,
            compare_lazy_df,
            streaming,
            equality_resolution,
            equality_check,
            sample_limit,
            base_alias,
            compare_alias,
            False,
            hide_empty_stats,
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

    def schema_differences_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of schema differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of schema differences.
        """
        return self._get_or_create(summarise_column_differences, self._comparison_metadata)

    def row_differences_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of row differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of row differences.
        """
        return self._get_or_create(get_row_comparison_summary, self._comparison_metadata)

    def row_differences_sample(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a sample of row differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The sample of row differences.
        """
        return self._get_or_create(get_row_differences, self._comparison_metadata)

    def value_differences_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of value differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of value differences.
        """
        return self._get_or_create(summarise_value_difference, self._comparison_metadata).select(
            "Value Differences for Column", pl.col("Count").cast(pl.Int64).alias("Count")
        )

    def schema_differences_sample(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a sample of schema differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The sample of schema differences.
        """
        return self._get_or_create(get_schema_comparison, self._comparison_metadata)

    def value_differences_sample(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a sample of value differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The sample of the value differences.
        """
        return self._get_or_create(get_column_value_differences_filtered, self._comparison_metadata)

    def is_unequal(self) -> bool:
        """
        Check if the two dataframes are unequal based on schema, rows, and values.

        Returns:
            bool: True if the dataframes are unequal, False otherwise.
        """
        if self.is_schema_unequal():
            return True
        if self.is_rows_unequal():
            return True
        if self.is_values_unequal():
            return True
        return False

    def is_schema_unequal(self) -> bool:
        """
        Check if the schemas of the two dataframes are unequal.

        Returns:
            bool: True if the schemas are unequal, False otherwise.
        """
        return convert_to_dataframe(self.schema_differences_sample()).height != 0

    def is_rows_unequal(self) -> bool:
        """
        Check if the rows of the two dataframes are unequal.

        Returns:
            bool: True if the rows are unequal, False otherwise.
        """
        return convert_to_dataframe(self.row_differences_sample()).height != 0

    def is_values_unequal(self) -> bool:
        """
        Check if the values of the two dataframes are unequal.

        Returns:
            bool: True if the values are unequal, False otherwise.
        """
        return (
            convert_to_dataframe(
                set_df_type(self.value_differences_sample(), streaming=False)
            ).height
            != 0
        )

    def all_differences_summary(self) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Get a summary of all differences between the two dataframes.

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: The summary of all differences.
        """
        return pl.concat( # type: ignore 
            [
                self.schema_differences_summary(),
                self.row_differences_summary(),
                self.value_differences_summary()
                .rename({"Value Differences for Column": "Statistic"})
                .filter(pl.col("Statistic") == pl.lit("Total Value Differences")),
                self.value_differences_summary()
                .rename({"Value Differences for Column": "Statistic"})
                .filter(pl.col("Statistic") != pl.lit("Total Value Differences"))
                .select(
                    pl.concat_str(pl.lit("Value diffs Col:"), pl.col("Statistic")).alias(
                        "Statistic"
                    ),
                    "Count",
                ),
            ]
        )

    def report(self, print: Union[Callable[[str], None], None] = None) -> Union[FuncAppend, None]:
        """
        Generate a report of the comparison results.

        Parameters:
            print (Union[Callable[[str], None], None]): The function to print the report for example you could supply the standard 'print' or logger.info if you want the output to be logged instead of printed.

        Returns:
            Union[FuncAppend, None]: The report or None if the dataframes are equal.
        """
        combined = FuncAppend(print)
        combined.append(80 * "-")
        combined.append("COMPARISON REPORT")
        combined.append(80 * "-")
        if not self.is_unequal():
            combined.append("Tables are exactly equal.")
            return None
        if self.is_schema_unequal():
            combined.append(
                f"\nSCHEMA DIFFERENCES:\n{self.schema_differences_summary()}\n{self.schema_differences_sample()}"
            )
        else:
            combined.append("No Schema differences found.")
        combined.append(80 * "-")
        if self.is_rows_unequal():
            combined.append(
                f"\nROW DIFFERENCES:\n{self.row_differences_summary()}\n{self.row_differences_sample()}"
            )
        else:
            combined.append("No Row differences found (when joining by the supplied id_columns).")
        combined.append(80 * "-")
        if self.is_values_unequal():
            combined.append(
                f"\nVALUE DIFFERENCES:\n{self.value_differences_summary()}\n{self.value_differences_sample()}"
            )
        else:
            combined.append("No Column Value differences found.")
        combined.append(80 * "-")
        combined.append("End of Report")
        combined.append(80 * "-")
        return combined
