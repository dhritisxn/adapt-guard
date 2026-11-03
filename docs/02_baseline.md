# Serial Baseline: Isolation Forest

## Algorithm as implemented

Standard Isolation Forest (Liu, Ting & Zhou, 2008), implemented from scratch
in `src/common/isolation_forest.hpp` (header-only, shared by all three
binaries — see rationale in `docs/03_openmp.md`).

- **Forest construction**: `n_trees` (default 100) trees, each built from an
  independently-drawn subsample of `psi` (default 256) rows from the
  training pool (sampling without replacement, partial Fisher-Yates —
  `IsolationForest::build_one_tree`).
- **Tree construction**: at each node, pick a uniformly random feature
  (retrying up to 8 times if the first pick doesn't vary in the current
  subset), then a uniformly random split value between that feature's min
  and max in the current subset. Recurse until `max_depth = ceil(log2(psi))`
  or the node holds ≤1 point.
- **Path length**: `depth + c(size)` at the leaf, where `c(n)` is the average
  path length of an unsuccessful BST search, `c(n) = 2H(n-1) - 2(n-1)/n`,
  using the `H(n) ≈ ln(n) + γ` approximation from the original paper.
- **Anomaly score**: `s(x) = 2^(-E[h(x)] / c(psi))`, averaging path length
  `h(x)` across all trees.
- **Decision threshold (`theta`)**: NOT a fixed 0.5 cutoff (0.5 is not a
  universally meaningful cutoff for IF scores; see below). Calibrated as the
  `(1 - contamination)` quantile of full-ensemble scores over (a sample of)
  the training pool, `contamination` default 0.01 (1%) — see
  `src/common/calibrate.hpp`. This mirrors how scikit-learn's
  `IsolationForest` picks its `offset_` from a `contamination` parameter, and
  is standard practice since IF gives a *relative* ranking whose absolute
  scale, in general, has no fixed anomalous/normal boundary.

## Deviations from the Review-1 proposal

- The proposal left open a choice between Isolation Forest and LOF; **Isolation
  Forest was chosen** because its scoring is embarrassingly parallel per-tree
  and per-transaction, which is a cleaner fit for the OpenMP/MPI/adaptive
  work than LOF's k-NN-based local density computation.
- Trees are stored as a flat `vector<IFNode>` (index-based children) rather
  than pointer-linked nodes — chosen for cache locality and because it makes
  a tree trivially POD/serializable if MPI tree-broadcast is ever added
  (see `docs/07_known_limitations.md`).
- **Deterministic per-tree seeding** (`seed_base + tree_index`) was added
  specifically so serial, OpenMP, and MPI builds produce bit-identical
  forests regardless of thread/rank scheduling — this isn't in the original
  proposal but is what makes the correctness validation in
  `docs/03_openmp.md` / `docs/03_mpi.md` an exact diff rather than a
  statistical "close enough" comparison.

## Correctness validation method

The serial binary is the reference. For OpenMP and MPI:

1. Run all three binaries with identical `--seed`, `--trees`, `--psi`,
   `--score-n`, and non-adaptive (`--adaptive` off, so there's no early-exit
   randomness in trees-used to compare against) on the same data.
2. Dump per-transaction `(row_id, true_label, pred_label, score, trees_used)`
   to CSV via `--out`.
3. Diff the CSVs (MPI output is sorted by `row_id` first, since `Gatherv`
   reconstructs stream order already but sorting is a defensive check against
   any future change to the shard-partitioning scheme).

Result: **0 diff lines** between serial and OpenMP at 1/2/4/8 threads, and
between serial and MPI at 1/2/4/8 processes (`scripts/validate_openmp.sh`,
`scripts/validate_mpi.sh`). This is only possible because of the
deterministic seeding scheme above; without it, OpenMP/MPI builds would
produce different (but statistically similar) trees depending on execution
order, and validation would have to fall back to a score-distribution or
label-agreement comparison instead of an exact diff.

## Baseline timing (reference numbers)

Single-threaded, `n_trees=100`, `psi=256`, `train_frac=0.5` (~142,403 training
rows, ~142,404 streamed rows), on the dev machine (12 logical cores, WSL2/Ubuntu
24.04, g++ 13.3.0 `-O3`):

| Metric | Value |
|---|---|
| Build time (100 trees) | ~8.3 ms |
| Full-ensemble score time (142,404 tx) | ~1,182 ms |
| Throughput (full ensemble, 1 thread) | ~120,500 tx/s |
| Precision / Recall / F1 (contamination=0.01, full stream) | 0.092 / 0.395 / 0.149 |
| Precision / Recall / F1 (contamination=0.01, first 20k stream rows) | 0.400 / 0.713 / 0.512 |

**Note on the F1 gap between the 20k-row subset and the full stream**: this
is real, not a bug. Review 2 attributed it to a never-retrained model;
the rolling-retraining experiment later showed it's mostly a property of
the data (the first 20k streamed rows contain an unusual burst of easy
fraud). See `docs/10_drift_and_tuning.md`.

Full raw numbers for every configuration tested are in
`docs/06_results/data/` (see `docs/06_results/results_summary.md`).
