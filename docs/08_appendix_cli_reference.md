# Appendix: CLI Flags & Hyperparameter Reference

Consolidated reference for the report's Experimental Setup / Appendix
section. Extracted directly from the argument-parsing code in
`src/serial/main.cpp`, `src/openmp/main.cpp`, `src/mpi/main.cpp`, `src/cuda/main.cu` (grepped,
not transcribed from memory), so this is guaranteed to match what the
binaries actually accept as of this writing.

## Flag reference (all four binaries, except where noted)

| Flag | Type | Default | Binaries | Meaning |
|---|---|---|---|---|
| `--data` | path | `data/creditcard.csv` | all | Input CSV path. |
| `--trees` | int | `100` | all | Number of trees in the forest (`n_trees`). |
| `--psi` | int | `256` | all | Subsample size per tree. `max_depth = ceil(log2(psi))`. |
| `--seed` | int | `42` | all | Base RNG seed. Tree `t` is always seeded `seed + t` — this is what makes serial/OpenMP/MPI outputs bit-identical (see `docs/02_baseline.md`). |
| `--train-frac` | double | `0.5` | all | Fraction of rows (chronological prefix) used as the training pool tree subsamples are drawn from. Remainder is the "stream." |
| `--score-n` | int | `-1` (= all remaining stream rows) | all | Caps how many streamed transactions are scored — the main lever for controlling workload size in scaling experiments. |
| `--adaptive` | boolean flag | off | all | Enables the early-exit mechanism. Without it, every transaction runs the full ensemble (`min_trees` forced to `n_trees`). |
| `--theta` | double | *(none — see note)* | all | Decision threshold. **If omitted, auto-calibrated** from `--contamination` (see below); if passed explicitly, that exact value is used and calibration is skipped. |
| `--contamination` | double | `0.01` | all | Only used when `--theta` is *not* passed: `theta` is set to the `(1-contamination)` quantile of full-ensemble scores over (a sample of) the training pool. See `docs/02_baseline.md`. |
| `--z` | double | `2.0` | all | Confidence-band multiplier for the adaptive stopping rule (`docs/04_adaptive_early_exit.md`). Higher = more conservative = more trees used on average = closer to full-ensemble accuracy. |
| `--min-trees` | int | `10` | all | Minimum trees evaluated before early exit is allowed to trigger. |
| `--explain` | string: `none`\|`integrated`\|`posthoc` | `none` | serial, openmp (**parsed but has no effect in the mpi binary** — echoed in its `RESULT` line but never wired into scoring; see `docs/03_mpi.md`) | Explainability mode — `integrated` reuses the scoring traversal (near-zero-cost), `posthoc` runs a full separate traversal for contributions (the naive baseline it's compared against). |
| `--out` | path | `""` (no file written) | all | Writes per-transaction `(row_id, true_label, pred_label, score, trees_used)` to this CSV. serial's `--explain` modes additionally write `<out>.explain.csv` with each anomaly's top-3 contributing features. |
| `--compare-full` | boolean flag | off | all | Only meaningful with `--adaptive`: additionally scores the same stream with the full ensemble (same paradigm, same resource count) to compute `agreement_full` and enable a speedup comparison. Adds real compute cost — off by default so it doesn't inflate ordinary timing runs. |
| `--repeat` | int | `1` | all | Repeats just the scoring phase internally this many times, averaging the timing — reduces run-to-run noise without re-paying dataset load/build cost. All benchmark sweeps in `docs/06_results/` use `--repeat 3`. |
| `--threads` | int | `0` (= OpenMP runtime default, i.e. `OMP_NUM_THREADS` or all cores) | **openmp only** | Calls `omp_set_num_threads()`. The *actual* thread count used (`omp_get_max_threads()`) is echoed in the `RESULT` line, since OpenMP silently caps requests above the available core count. |
| `--retrain-every` | int | `0` (off) | **openmp only** | Rolling retraining: rebuild the forest and re-calibrate `theta` every N streamed transactions (`docs/10_drift_and_tuning.md`). Disables `--compare-full`. |
| `--window` | int | `50000` | **openmp only** | Number of most-recent already-arrived rows each retrain uses. |
| `--drop-time` | boolean flag | off | **openmp only** | Removes the `Time` feature (used in the hyperparameter sweep). |
| `--dist-build` | boolean flag | off | **mpi only** | Distributed forest construction: trees built round-robin across ranks, exchanged with `MPI_Allgatherv`. Bit-identical result (`docs/03_mpi.md`). |
| `--block` | int | `128` | **cuda only** | CUDA threads per block. |
| `--device` | int | `0` | **cuda only** | CUDA device index. |

**MPI process count is not a program flag** — it's set via `mpirun -np N`
(optionally `--oversubscribe` to request more processes than physical
cores, used throughout this project's single-node benchmarks; see
`docs/07_known_limitations.md`).

## `RESULT` output line fields

Every run prints one `RESULT key=value ...` line to stdout (parsed by
`scripts/run_benchmarks.py`); MPI additionally prints one `RANKINFO
key=value ...` line **per rank, to stderr** (shard bounds, local
build/score time — used for the `load_imbalance_ms` calculation).

| Field | Meaning |
|---|---|
| `binary` | `serial` \| `openmp` \| `mpi` \| `cuda` |
| `n_trees`, `psi`, `threads`, `ranks` | Echoed config (see flags above). |
| `n_train`, `n_stream` | Actual row counts after applying `--train-frac` / `--score-n`. |
| `adaptive`, `theta`, `contamination`, `z`, `min_trees`, `explain` | Echoed config. |
| `build_ms` | Forest construction time. MPI: max across ranks. |
| `score_ms` | Scoring-phase time only (averaged over `--repeat`). MPI: max across ranks — the wall-clock bound for the bulk-synchronous gather pattern. |
| `avg_score_ms` (mpi only) | Mean across ranks (vs. `score_ms`'s max) — the gap between the two is the load-imbalance signal. |
| `load_imbalance_ms` (mpi only) | `max_score_ms - min_score_ms` across ranks. |
| `comm_ms` (mpi only) | Time in the three `Gatherv` calls only, isolated from compute. |
| `throughput_tx_s` | `n_stream / (score_ms / 1000)`. |
| `avg_trees_used` | Mean trees evaluated per transaction (100 = full ensemble; lower under `--adaptive`). |
| `precision`, `recall`, `f1`, `tp`, `fp`, `fn`, `tn` | Standard detection metrics vs. the true `Class` label (evaluation only — never used in training). |
| `agreement_full`, `full_score_ms` | Only populated with `--compare-full`: agreement between adaptive and full-ensemble labels, and the full ensemble's own timing (same paradigm/resource count). `-1` when not requested. |
| `retrain_every`, `window`, `n_retrains`, `retrain_ms`, `drop_time` (openmp only) | Rolling-retraining config, number of retrains done, and their total time (rebuild + re-calibration, excluded from `score_ms`). |
| `dist_build`, `build_comm_ms` (mpi only) | Whether `--dist-build` was used, and time spent in its two `Allgatherv` calls. |
| `gpu`, `h2d_ms`, `d2h_ms` (cuda only) | GPU compute capability (`sm_75`), host→device and device→host copy times. For CUDA, `threads` is the block size and `score_ms` is kernel time (`cudaEvent`). |
| `warp_efficiency` (cuda only) | `sum(trees_used) / sum over 32-lane warps of 32 × max(trees_used)` — the fraction of issued tree evaluations that were useful under adaptive early exit (1.0 for full ensemble). |

## Hyperparameters actually used for the reported experiments

(`docs/06_results/`, unless a specific experiment overrides one — see
`scripts/run_benchmarks.py` for the exact per-experiment invocations.)

| Hyperparameter | Value(s) used |
|---|---|
| `n_trees` | 100 (all experiments) |
| `psi` | 256 (all experiments) |
| `seed` | 42 (all experiments — fixed, not swept) |
| `train_frac` | 0.5 |
| `contamination` (theta auto-calibration) | 0.01 |
| `min_trees` (adaptive) | 10 |
| `z` (adaptive confidence multiplier) — swept | 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 |
| Thread/rank counts swept (strong & weak scaling) | 1, 2, 3, 4, 6, 8, 12 |
| Strong-scaling fixed workload | 100,000 streamed transactions |
| Weak-scaling workload | 8,000 transactions × worker count |
| Accuracy-vs-speedup fixed resource count | 8 threads (OpenMP), 8 ranks (MPI); 50,000 streamed transactions |
| Explainability latency transaction counts | 500, 1,000, 2,000, 5,000, 10,000 |
| `--repeat` (internal timing averaging) | 3, for every benchmark sweep |

These are the standard Isolation Forest literature defaults
(`psi=256`, `n_trees=100`). They are kept for every performance
experiment so results stay comparable. The separate hyperparameter sweep
(`docs/10_drift_and_tuning.md`) covers psi ∈ {64..1024}, n_trees ∈ {50..400},
contamination ∈ {0.1%..1%} and with/without `Time`.

Completion experiments: CUDA vs CPU at 10k/50k/100k/142,404 tx; CUDA
z-sweep at 50,000 tx; CUDA block sizes 32-1024; MPI build at
(100, 256), (1000, 256), (1000, 2048) × 1/2/4/8 ranks; drift with
`--retrain-every` 0/20000/10000 and `--window 50000`.
