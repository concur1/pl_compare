"""Benchmark for every data-producing method of the compare() class.

Times each method on the current branch against the same method from main's
product code (loaded from git so it can never drift from a hand-written copy).
CI timing is noisy, so we report a ratio to main rather than absolute numbers,
and use data large enough that the measured work sits well above the runner's
scheduling noise.

The composite methods (summary, equals_summary, is_*, report) only call the
six methods below, so they are covered transitively.

Regression gate: exits non-zero only if the current branch is slower than main
by more than BOTH 50% (1.5x) AND an absolute floor. The most expensive methods
are gated by the 1.5x ratio; the sub-30ms methods are dominated by scheduling
noise (locally the paired cur/main difference wanders up to ~10ms), so the
ratio is meaningless there and only the absolute floor applies.
"""

import random
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from pl_compare.compare import compare

# Gate floor (ms): ignore regressions smaller than this even if they exceed
# the 1.5x ratio. Chosen from the noise seen on a 500kx8 run: cheap methods'
# cur/main paired difference wanders up to ~10ms locally, and a shared tiny CI
# runner is a few times noisier still. 100ms sits well above that while staying
# far below any real regression on the methods that do real work.
NOISE_FLOOR_MS = 100.0

METHODS = [
    "schemas_summary",
    "rows_summary",
    "rows_sample",
    "values_summary",
    "schemas_sample",
    "values_sample",
]


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


def load_main_compare():
    """Load main's actual pl_compare/compare.py from git into its own module."""
    # Accept a couple of ref spellings: local checkouts use "main", CI clones
    # (which only fetch the pushed branch) expose it as "origin/main".
    source = None
    for ref in ("main", "origin/main"):
        result = subprocess.run(
            ["git", "show", f"{ref}:pl_compare/compare.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            source = result.stdout
            break
    if source is None:
        raise RuntimeError(
            "Could not resolve main:pl_compare/compare.py from git. Make sure the "
            "'main' branch is available locally (the bench target fetches it) so "
            "the benchmark can compare against main's real product code."
        )
    module = types.ModuleType("pl_compare_main")
    exec(compile(source, "pl_compare/compare.py@main", "exec"), module.__dict__)
    return module


# Reference: main's real product code, so the benchmark never drifts from a copy.
main_compare_cls = load_main_compare().compare


def time_methods(method, cur_cls, main_cls, base_df, compare_df, reps):
    """Time one method on fresh instances of both impls, interleaved.

    A fresh compare() instance per rep keeps its ``_created_frames`` cache from
    hiding the real computation, and the instance is built *before* the timer
    starts so construction cost isn't measured. Current and main alternate
    which one runs first each rep so drift can't bias one side.
    """
    warm = lambda cls: getattr(cls(["id"], base_df, compare_df), method)()
    warm(cur_cls)
    warm(main_cls)
    totals = [0.0, 0.0]
    for i in range(reps):
        for idx in ((0, 1) if i % 2 == 0 else (1, 0)):
            cls = (cur_cls, main_cls)[idx]
            inst = cls(["id"], base_df, compare_df)
            t0 = time.perf_counter()
            getattr(inst, method)()
            totals[idx] += time.perf_counter() - t0
    return totals[0] / reps, totals[1] / reps


def gate(method, cur_ms, main_ms) -> None:
    """Fail if current is slower than main by more than the allowed slack.

    Tolerance is the LARGER of 50% of main's time and the absolute floor: for a
    fast method the ratio is just noise, so it must also clear the floor before
    it counts; for a slow method 50% is the meaningful bound.
    """
    slack_ms = max(0.5 * main_ms, NOISE_FLOOR_MS)
    assert cur_ms - main_ms <= slack_ms, (
        f"{method}: current {cur_ms:.0f}ms is more than "
        f"max(50% of main, {NOISE_FLOOR_MS:.0f}ms) slower than main {main_ms:.0f}ms"
    )


def gate_selfcheck() -> None:
    """Cheap smoke test of the gate (fast methods survive noise, slow ones don't)."""
    gate("heavy", 2040, 2000)  # 2% noise on a 2s method: fine

    def fails(method, cur_ms, main_ms):
        try:
            gate(method, cur_ms, main_ms)
        except AssertionError:
            return
        raise AssertionError(f"{method} {cur_ms}ms vs {main_ms}ms should have failed")

    fails("heavy", 3200, 2000)   # 1.6x on a 2s method: real regression
    gate("fast", 3.3, 2.0)       # 1.65x on a 2ms method: just noise
    fails("fast", 110, 2.0)      # pathological, clears the floor


def check_equal(cur, main) -> None:
    assert cur.schema == main.schema
    assert len(cur) == len(main)
    for ra, rb in zip(cur.to_dicts(), main.to_dicts()):
        for k in ra:
            assert ra[k] == rb[k] or (
                isinstance(ra[k], float) and abs(ra[k] - rb[k]) < 1e-6
            ), (ra[k], rb[k])


def main() -> None:
    gate_selfcheck()
    # 500k rows put each values_* call at roughly 2s (locally), well above the
    # shared tiny runner's scheduling noise, while the rest of the methods stay
    # cheap. This runs once in a dedicated CI job (not the test matrix), so the
    # single larger run costs less than four matrix copies of the old bench.
    cases = [(200_000, 6), (500_000, 8)]
    reps = 3
    print(f"polars {pl.__version__}, {reps} reps")
    header = f"{'rows':>8} {'cols':>4} {'method':<20} {'current (ms)':>12} {'main (ms)':>12} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for rows, cols in cases:
        base_df, compare_df = make_data(rows, cols, mutate_frac=0.2)
        for method in METHODS:
            check_equal(
                getattr(compare(["id"], base_df, compare_df), method)(),
                getattr(main_compare_cls(["id"], base_df, compare_df), method)(),
            )
            cur_s, main_s = time_methods(
                method, compare, main_compare_cls, base_df, compare_df, reps
            )
            cur_ms, main_ms = cur_s * 1000, main_s * 1000
            ratio = cur_ms / main_ms
            print(
                f"{rows:>8} {cols:>4} {method:<20} {cur_ms:>12.1f} {main_ms:>12.1f} {ratio:>6.2f}x"
            )
            gate(method, cur_ms, main_ms)
    print(f"OK: within max(50%, {NOISE_FLOOR_MS:.0f}ms) of main on all methods")


if __name__ == "__main__":
    main()
