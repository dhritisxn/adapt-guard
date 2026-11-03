# OpenMP Parallelization

`src/openmp/main.cpp`. Reuses the exact same `IsolationForest` core as the
serial baseline (`src/common/isolation_forest.hpp`) — no algorithmic
changes, only parallel loop annotations. This header is deliberately kept
free of any `#include <omp.h>` / OpenMP pragmas so the serial binary has zero
OpenMP dependency; only `src/openmp/main.cpp` includes `<omp.h>` and wraps
the two hot loops.

## Parallelization strategy

1. **Tree construction**: `#pragma omp parallel for schedule(dynamic)` over
   the `n_trees` loop. Each iteration calls `forest.build_one_tree(train_data,
   seed + t)` and writes to `forest.trees[t]` — trees is pre-sized via
   `prepare_trees()` before the parallel region so there's no reallocation
   race, and each iteration touches a distinct vector element, which is
   safe under the C++ standard's rules for concurrent writes to disjoint
   `vector` elements. `dynamic` scheduling because tree build cost isn't
   perfectly uniform (depends on how many `max_depth` splits actually get
   made per subsample).
2. **Transaction scoring**: `#pragma omp parallel for schedule(dynamic, 64)`
   over the stream. `dynamic` scheduling matters much more here than for
   tree construction: with `--adaptive` on, per-transaction cost varies by
   up to 10x (some transactions early-exit after `min_trees`, others run
   the full ensemble), so a `static` split would leave some threads idle
   while others are still grinding through hard cases — this is exactly the
   load-imbalance failure mode the profiling in `docs/06_results` checks for.
3. Both loops write to pre-sized output arrays (`scores`, `pred_label`,
   `trees_used`, `contribs`) indexed by `i`, so there's no false sharing
   beyond normal cache-line effects, and no locks/atomics needed anywhere.

## Determinism under parallelism

Because every tree's RNG is seeded by `seed_base + tree_index` (not by
thread ID or by call order), `forest.trees[t]` is bit-identical no matter
which thread builds it or in what order threads finish. This is what makes
the correctness check below an exact diff instead of a statistical one.

## Correctness validation

`scripts/validate_openmp.sh`: runs serial once, then OpenMP at 1/2/4/8
threads, all with identical `--seed --trees --psi --score-n`, non-adaptive,
dumps per-transaction CSVs, diffs them against the serial output.

**Result: 0 diff lines at every thread count (1, 2, 4, 8).** Scores,
predicted labels, and trees-used are bit-for-bit identical to serial.

## Implementation notes / decisions

- `--threads N` calls `omp_set_num_threads(N)`; the actual thread count used
  (`omp_get_max_threads()`) is echoed back in the `RESULT` line so benchmark
  scripts can verify the requested count was actually honored (OpenMP will
  silently cap at the number of available cores if you ask for more, and the
  script's parsed `threads=` value should reflect that rather than the
  requested value).
- No `reduction()` clauses are used for the confusion-matrix / agreement
  metrics — those are computed in a fast serial pass *after* the parallel
  scoring loop, since they're O(n) over already-materialized label arrays
  and not worth parallelizing or risking a race on.
- Full scaling numbers (strong scaling, weak scaling, `perf` profiling) are
  in `docs/06_results/`.
