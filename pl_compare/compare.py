import polars as pl
import time
import types
from typing import Literal, Callable, List
from dataclasses import dataclass


@dataclass
class ComparisonMetadata:
    """Class for holding the (meta)data used to generate the comparison dataframes."""

    id_columns: list[str]
    base_df: pl.LazyFrame
    compare_df: pl.LazyFrame
    streaming: bool
    threshold: float | None
    equality_check: Callable[[str, pl.DataType], pl.Expr] | None
    sample_limit: int
    base_alias: str
    compare_alias: str
    schema_comparison: bool
    hide_empty_stats: bool


def get_duplicates(
    df: pl.LazyFrame | pl.DataFrame, id_columns: List[str]
) -> pl.LazyFrame | pl.DataFrame:
    ctx = pl.SQLContext(input_table=df)
    query = f"""SELECT {', '.join(id_columns)}, count(*) AS row_count 
                FROM input_table GROUP BY {", ".join(id_columns)} 
                HAVING row_count>1"""
    return ctx.execute(query)


def duplicates_summary(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    ctx = pl.SQLContext(input_table=df)
    query = """SELECT  (sum(row_count)-count(*)) AS TOTAL_DUPLICATE_ROWS, 
                   (max(row_count)-1) AS MAXIMUM_DUPLICATES_FOR_AN_ID 
               FROM input_table"""
    return ctx.execute(query)


def duplicate_examples(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    ctx = pl.SQLContext(input_table=df)
    query = """SELECT * EXCLUDE(row_count), (row_count-1) AS DUPLICATES FROM input_table"""
    return ctx.execute(query)


def collect_if_lazy(df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=True)
    return df


def lazy_if_dataframe(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
    return df


def set_df_type(
    df: pl.LazyFrame | pl.DataFrame, streaming: bool = False
) -> pl.LazyFrame | pl.DataFrame:
    if streaming:
        return lazy_if_dataframe(df)
    if not streaming:
        return collect_if_lazy(df)


def get_uncertain_row_count(df: pl.LazyFrame) -> int:
    df_solid = collect_if_lazy(df.select("count"))
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
        .groupby(["in_base", "in_compare"])
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
    id_columns: list[str],
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
    id_columns: list[str],
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
    equality_check: Callable[[str, pl.DataType], pl.Expr] | None,
    threshold: float | None,
    col: str,
    format: pl.DataType,
) -> pl.Expr:
    def default_equality_check(col: str, format: pl.DataType) -> pl.Expr:
        return (
            (pl.col(f"{col}_base") != pl.col(f"{col}_compare"))
            | (pl.col(f"{col}_base").is_null() & ~pl.col(f"{col}_compare").is_null())
            | (~pl.col(f"{col}_base").is_null() & pl.col(f"{col}_compare").is_null())
        )

    def ignore_numeric_differences_equality_check(col: str, format: pl.DataType) -> pl.Expr:
        return (
            ((pl.col(f"{col}_base") - pl.col(f"{col}_compare")).abs() > threshold)
            | (pl.col(f"{col}_base").is_null() & ~pl.col(f"{col}_compare").is_null())
            | (~pl.col(f"{col}_base").is_null() & pl.col(f"{col}_compare").is_null())
        )

    if equality_check is not None:
        return equality_check(col, format)
    if threshold is not None and format in [
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
    id_columns: list[str],
    base_df: pl.LazyFrame,
    compare_df: pl.LazyFrame,
    compare_columns: dict[str, pl.DataType],
    equality_check: Callable[[str, pl.DataType], pl.Expr] | None,
    how_join: Literal["inner", "outer"] = "inner",
    threshold: float | None = None,
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
            get_equality_check(equality_check, threshold, col, format).alias(f"{col}_has_diff")
            for col, format in compare_columns.items()
        ]
    )


def summarise_value_difference(meta: ComparisonMetadata) -> pl.LazyFrame:
    final_df = (
        get_column_value_differences(meta)
        .groupby(["variable"])
        .agg(pl.sum("has_diff"))
        .sort("has_diff", descending=True)
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
    id_columns: list[str], compare_column: str, combined_tables: pl.LazyFrame
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


def get_columns_to_compare(meta) -> dict[str, pl.DataType]:
    # if schema_comparison:
    #    return ["format"]
    columns_to_exclude: list[str] = []
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


def get_column_value_differences(meta) -> pl.LazyFrame:
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
        threshold=meta.threshold,
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


def get_column_value_differences_filtered(meta) -> pl.LazyFrame:
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


def get_schema_comparison(meta) -> pl.LazyFrame:
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
            threshold=None,
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


class compare:
    """
    Compare two dataframes.
    """

    def __init__(
        self,
        id_columns: list[str],
        base_df: pl.DataFrame | pl.LazyFrame,
        compare_df: pl.DataFrame | pl.LazyFrame,
        streaming: bool = False,
        threshold: float | None = None,
        equality_check: Callable[[str, pl.DataType], pl.Expr] | None = None,
        sample_limit: int = 5,
        base_alias: str = "base",
        compare_alias: str = "compare",
        hide_empty_stats: bool = False,
    ):
        self.comparison_metadata = ComparisonMetadata(
            id_columns,
            lazy_if_dataframe(base_df),
            lazy_if_dataframe(compare_df),
            streaming,
            threshold,
            equality_check,
            sample_limit,
            base_alias,
            compare_alias,
            False,
            hide_empty_stats,
        )
        self.created_frames: dict[str, pl.DataFrame | pl.LazyFrame] = {}

    def get_or_create(
        self, func: types.FunctionType, *args
    ) -> dict[str, pl.LazyFrame | pl.DataFrame]:
        if func.__name__ not in self.created_frames:
            self.created_frames[func.__name__] = set_df_type(
                func(*args), streaming=self.comparison_metadata.streaming
            )
        return self.created_frames.get(func.__name__)

    def schema_differences_summary(self):
        return self.get_or_create(summarise_column_differences, self.comparison_metadata)

    def row_differences_summary(self):
        return self.get_or_create(get_row_comparison_summary, self.comparison_metadata)

    def row_differences_sample(self):
        return self.get_or_create(get_row_differences, self.comparison_metadata)

    def value_differences_summary(self):
        return self.get_or_create(summarise_value_difference, self.comparison_metadata).select(
            "Value Differences for Column", pl.col("Count").cast(pl.Int64).alias("Count")
        )

    def schema_differences_sample(self):
        return self.get_or_create(get_schema_comparison, self.comparison_metadata)

    def value_differences_sample(self):
        return self.get_or_create(get_column_value_differences_filtered, self.comparison_metadata)

    def is_schema_unequal(self):
        return self.schema_differences_sample().height != 0

    def is_rows_unequal(self):
        return self.row_differences_sample().height != 0

    def is_values_unequal(self):
        return set_df_type(self.value_differences_sample(), streaming=False).height != 0

    def all_differences_summary(self):
        return pl.concat(
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

    def report(self):
        print("Schema summary:")
        print(self.schema_differences_summary())
        print("Schema differences:", self.is_schema_unequal())
        print(self.schema_differences_sample())
        print("Row summary:")
        print(self.row_differences_summary())
        print("Row differences:", self.is_rows_unequal())
        print(self.row_differences_sample())
        print("Value summary:")
        print(self.value_differences_summary())
        print("Value differences:", self.is_values_unequal())
        print(self.value_differences_sample())
        print("All differences summary:")
        print(self.all_differences_summary())
