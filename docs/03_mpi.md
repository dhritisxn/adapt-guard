# MPI Parallelization

`src/mpi/main.cpp`, built with OpenMPI 4.1.6 (`mpic++`).

## Parallelization strategy

The spec calls for "partition the streaming dataset across processes to
simulate multi-node deployment" — this is a data-parallel design over the
**scoring/streaming phase**, not the forest-training phase. Design:

1. **Every rank independently builds the identical forest.** Because tree
   `t` is always seeded with `seed_base + t` regardless of who builds it
   (see `docs/02_baseline.md`), every rank ends up with a bit-identical
   `IsolationForest` without any communication. This is a deliberate
   design decision, not a missed opportunity for parallelism — see
   "Why not also parallelize tree construction over MPI?" below.
2. **The stream is partitioned into contiguous per-rank shards** via a
   simple block distribution (`compute_shard`): `n_stream / nprocs`
   transactions per rank, with the remainder spread across the first
   few ranks. Every rank computes its own shard bounds independently
   (deterministic function of `rank`, `nprocs`, `n_stream` — no
   communication needed for this either).
3. **Each rank scores its local shard** (full-ensemble or adaptive,
   matching the serial/OpenMP logic exactly), timed locally with
   `MPI_Wtime`.
4. **Real MPI communication**: results (`scores`, `pred_label`,
   `trees_used`) are collected back to rank 0 with three `MPI_Gatherv`
   calls (variable-sized shards, since `n_stream` doesn't always divide
   evenly by `nprocs`). Timing/load-balance stats are aggregated with
   `MPI_Reduce` (`MAX`, `MIN`, `SUM` over each rank's local score time).
5. `MPI_Barrier` calls bracket the build and score phases so the reported
   `MPI_Wtime` deltas reflect genuinely synchronized start points across
   ranks, not clock skew from staggered process startup.

## Why is tree construction replicated by default?

(A distributed alternative now exists, see `--dist-build` below.)
Distributing it was considered (split `n_trees` across ranks, then `Allgatherv` the
built trees — `IFNode` is a POD struct so this is technically
straightforward). It was **deliberately not made the default**, for two reasons:

1. Build time is ~8-13ms regardless of rank count (see
   `docs/06_results/`) — at `n_trees=100`, `psi=256`, it is not the
   bottleneck, so distributing it wouldn't move the needle on any
   scaling result in this project.
2. Redundant identical-forest replication across ranks is actually a
   *more realistic* multi-node deployment pattern for this use case than a
   live broadcast: independent fraud-detection nodes typically hold a
   local replica of a periodically-synced model and score their own
   traffic shard without a live model-serving dependency on a central
   node. Making the model a broadcast dependency would add a
   synchronization point that a real deployment would try to avoid.

For the default configuration, MPI speedup comes entirely from the
scoring phase, and build time stays flat (not decreasing) as rank count
grows. The benchmark data confirms this.

## Distributed tree construction (`--dist-build`)

Added to complete the proposal, for forests large enough that build
cost matters. With `--dist-build`:

1. Rank `r` builds trees `r, r+P, r+2P, ...` (round-robin, so cheap and
   expensive trees spread evenly) with the usual seed `seed + t`.
2. `MPI_Allgatherv` #1 exchanges the node count of every tree.
3. `MPI_Allgatherv` #2 exchanges all nodes as raw bytes (`IFNode` is POD,
   20 bytes), using byte counts/displacements derived from step 2.
4. Each rank unpacks the rank-major buffer back into `trees[t]`.

Every rank ends up with the bit-identical forest:
`scripts/validate_mpi_dist_build.sh` gives **0 diff lines vs serial at
1/2/3/4/8 ranks**. The exchange time is reported as `build_comm_ms`.

Results (`mpi_dist_build.csv`, `mpi_dist_build.png`; build time includes
the exchange, max over ranks):

| n_trees, ψ | Ranks | Replicated build | Distributed build | of which exchange |
|---|---|---|---|---|
| 100, 256 | 1 / 8 | 4.5 / 14.3 ms | 5.1 / 6.1 ms | 0.03 / 2.0 ms |
| 1000, 256 | 1 / 8 | 52 / 149 ms | 53 / 22 ms | 2.0 / 4.8 ms |
| 1000, 2048 | 1 / 8 | 282 / 862 ms | 270 / 129 ms | 2.2 / 18.5 ms |

Two findings:

- **Replicated build gets *slower* with more ranks** on one machine (282
  → 862 ms at ψ=2048), because 8 processes building the same forest
  compete for the same memory bandwidth and caches. On separate nodes
  this contention would disappear, and replicated build would stay flat.
- **Distributed build gives ~2.2x at ψ=2048** (282 → 129 ms; 6.7x faster than
  replicated at 8 ranks). It plateaus after 2 ranks for the same
  memory-bandwidth reason, plus the growing exchange. For the default
  100-tree forest the gain is a few milliseconds, which is why
  replicated stays the default.

## Correctness validation

`scripts/validate_mpi.sh`: serial once, then MPI at 1/2/4/8 processes via
`mpirun -np P --oversubscribe`, identical `--seed --trees --psi --score-n`,
non-adaptive. Rank 0's gathered output is sorted by `row_id` (defensive;
`Gatherv`'s displacement scheme already reconstructs stream order) and
diffed against the serial CSV.

**Result: 0 diff lines at every process count (1, 2, 4, 8).**

## Implementation notes / decisions

- Every rank loads `creditcard.csv` independently (redundant I/O). On a
  real cluster this assumes a shared/parallel filesystem visible to every
  node, which is the norm in HPC environments; on the single-machine WSL2
  setup used for this project's benchmarks, this means `P` processes read
  the same file concurrently from the same disk, which is a real (and
  reported) contributor to `mpirun`'s wall-clock at higher process counts
  on a single node — see `docs/06_results/results_summary.md`.
- `RANKINFO` lines (per-rank shard bounds, local build/score time) are
  printed to **stderr** by every rank, kept separate from the single
  `RESULT` line rank 0 prints to stdout, so benchmark scripts can parse
  `RESULT` without needing to filter out N interleaved per-rank lines.
- `load_imbalance_ms` in the `RESULT` line (`max_score_ms - min_score_ms`
  across ranks) is the direct measurement used to check for load imbalance
  in the profiling analysis — see `docs/06_results/`.
- `comm_ms` isolates just the three `Gatherv` calls (barrier-to-barrier),
  separate from compute time, for the "is the bottleneck compute, memory,
  or synchronization" profiling question.
- Explainability latency (`--explain integrated|posthoc`) is **not**
  wired into the MPI binary — that comparison is an inherently
  single-node, per-transaction algorithmic question (single traversal
  reusing scoring vs. a separate full pass), and doesn't gain anything
  from being run distributed. It's benchmarked once, on the serial/OpenMP
  binaries — see `docs/05_explainability.md`.
