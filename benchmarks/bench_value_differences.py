"""Benchmark for summarise_value_difference (a.k.a. values_summary).

Timing in CI is noisy, so instead of absolute numbers we report a ratio against
an eager reference implementation of the same result. That keeps the measurement
meaningful across machines, and guards against the catastrophic case where the
production path ever loses badly to the reference. (The production path becomes
lazy once the perf-value-differences-lazy branch lands; until then it is the
current implementation, and the benchmark still catches regressions.)

Exits 0 unless the production path is >= 3x slower than the eager reference.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from pl_compare.compare import (
    compare,
    convert_to_dataframe,
    get_column_value_differences,
    summarise_value_difference,
)


def make_data(rows: int, cols: int, mutate_frac: float, seed: int = 42):
    """Deterministic base/compare dataframes with a mix of row and value diffs."""
    rng = random.Random(seed)
    schema = {"id": pl.Int64} | {f"c{i}": pl.Utf8 for i in range(cols)}
    base = {
        "id": list(range(rows)),
        **{f"c{i}": [f"v{rng.randint(0, 99)}" for _ in range(rows)] for i in range(cols)},
    }
    cmp = {"id": list(range(rows)), **{f"c{i}": base[f"c{i}"][:] for i in range(cols)}}
    for r in range(rows):
        if rng.random() < mutate_frac:
            cmp[f"c{rng.randrange(cols)}"][r] = f"DIFF{rng.randint(1000, 9999)}"
    return pl.DataFrame(base, schema=schema), pl.DataFrame(cmp, schema=schema)


def summarise_value_difference_eager(meta):
    """Reference: the old implementation that materialised the melted frame first."""
    value_differences = convert_to_dataframe(get_column_value_differences(meta))
    variable_alias = meta.column_mapping.mapping[meta.column_mapping.variable]
    final_df = (
        value_differences.group_by([variable_alias])
        .agg(pl.sum("has_diff"))
        .sort(variable_alias, descending=False)
        .rename({variable_alias: "Value Differences", "has_diff": "Count"})
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
    final_df2 = pl.concat([total_differences, final_df_with_percentages])
    if meta.hide_empty_stats:
        final_df2 = final_df2.filter(pl.col("Count") > 0)
    return final_df2


def timeit(fn, meta, reps: int) -> float:
    fn(meta)  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(meta)
    return (time.perf_counter() - t0) / reps


def check_equal(cur, eager) -> None:
    assert cur.schema == eager.schema
    for ra, rb in zip(cur.to_dicts(), eager.to_dicts()):
        for k in ra:
            assert ra[k] == rb[k] or (isinstance(ra[k], float) and abs(ra[k] - rb[k]) < 1e-6), (
                ra[k],
                rb[k],
            )


def main() -> None:
    # CI runs land on tiny runners with tight job timeouts, so keep the data
    # proportionate: 25k/50k rows is plenty to measure the current-vs-eager ratio.
    cases = [(25_000, 3), (50_000, 5)]
    reps = 2
    print(f"polars {pl.__version__}, {reps} reps")
    header = f"{'rows':>8} {'cols':>4} {'current (ms)':>12} {'eager (ms)':>12} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for rows, cols in cases:
        base_df, compare_df = make_data(rows, cols, mutate_frac=0.2)
        meta = compare(["id"], base_df, compare_df)._comparison_metadata
        check_equal(summarise_value_difference(meta), summarise_value_difference_eager(meta))
        cur_ms = timeit(summarise_value_difference, meta, reps) * 1000
        eager_ms = timeit(summarise_value_difference_eager, meta, reps) * 1000
        ratio = cur_ms / eager_ms
        print(f"{rows:>8} {cols:>4} {cur_ms:>12.1f} {eager_ms:>12.1f} {ratio:>6.2f}x")
        assert cur_ms < 3 * eager_ms, (
            f"current {cur_ms:.0f}ms is >= 3x the eager reference {eager_ms:.0f}ms"
        )
    print("OK: production path is within 3x of the eager reference")


if __name__ == "__main__":
    main()
