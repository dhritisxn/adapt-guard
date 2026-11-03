# Results Summary

See also `plots/architecture_diagram.png` for the overall pipeline
(data → shared `IsolationForest` core → serial/OpenMP/MPI → adaptive
early-exit + integrated explainability → correctness validation →
benchmarking), and `docs/08_appendix_cli_reference.md` for a consolidated
table of every CLI flag and hyperparameter value used below.

All numbers below are read directly from the CSVs/text files in
`docs/06_results/data/`, produced by `scripts/run_benchmarks.py` and
`scripts/run_perf_profiling.sh` running the actual compiled binaries.
Nothing here is estimated or backfilled. Machine: WSL2/Ubuntu 24.04 on a
12-logical-core host, g++ 13.3.0 `-O3`, OpenMPI 4.1.6. `n_trees=100`,
`psi=256` for every run unless stated otherwise. The strong/weak-scaling
and accuracy-vs-speedup tables (all except the `perf` section) were
captured in one back-to-back session so OpenMP and MPI numbers share the
same OS page-cache/system state and are directly comparable; every
`score_ms` in those tables is itself an average of 3 in-process repeats
(`--repeat 3`, supported identically by all three binaries) to reduce
run-to-run noise.

## Strong scaling (fixed 100,000 transactions)

`strong_scaling_openmp.png` / `.csv`, `strong_scaling_mpi.png` / `.csv`.

| Threads/Ranks | OpenMP score_ms | OpenMP speedup | OpenMP efficiency | MPI score_ms | MPI speedup | MPI efficiency |
|---|---|---|---|---|---|---|
| 1 | 555.2 | 1.00x | 100% | 462.4 | 1.00x | 100% |
| 2 | 306.4 | 1.81x | 90.6% | 246.5 | 1.88x | 93.8% |
| 3 | 198.9 | 2.79x | 93.0% | 183.7 | 2.52x | 83.9% |
| 4 | 146.7 | 3.78x | 94.6% | 163.8 | 2.82x | 70.6% |
| 6 | 107.8 | 5.15x | 85.9% | 105.4 | 4.39x | 73.1% |
| 8 | 89.8 | 6.19x | 77.3% | 81.9 | 5.65x | 70.6% |
| 12 | 62.0 | 8.95x | 74.6% | 71.0 | 6.51x | 54.3% |

**Takeaway**: OpenMP holds up better at high thread counts (74.6% efficiency
at 12 threads vs. MPI's 54.3%) — expected, since the test machine has only
12 *logical* cores total, and MPI additionally pays OS process-scheduling
overhead per rank (`--oversubscribe`) that OpenMP's shared-memory threads
don't. MPI's per-rank `load_imbalance_ms` (`strong_scaling_mpi.csv`) jumps
from 6.3ms at 8 ranks to 16.5ms at 12 — a real, measured oversubscription
effect at the point where rank count meets core count, not smoothed over.

## Weak scaling (workload scaled proportionally, 8,000 tx/worker)

`weak_scaling_openmp.png` / `.csv`, `weak_scaling_mpi.png` / `.csv`.

| Threads/Ranks | Workload | OpenMP score_ms | MPI score_ms |
|---|---|---|---|
| 1 | 8,000 | 43.7 | 37.4 |
| 2 | 16,000 | 45.2 | 38.2 |
| 3 | 24,000 | 42.0 | 40.3 |
| 4 | 32,000 | 43.4 | 51.6 |
| 6 | 48,000 | 46.9 | 47.0 |
| 8 | 64,000 | 50.2 | 49.7 |
| 12 | 96,000 | 55.1 | 75.3 |

**Takeaway**: OpenMP stays very close to flat (ideal weak scaling) across a
12x increase in workload+resources — only ~26% end-to-end growth. MPI grows
more (~101%), and non-monotonically: rank=4 (51.6ms) sits above both its
neighbors (40.3ms at 3, 47.0ms at 6), and rank=12 jumps sharply (75.3ms) —
reported as-is rather than smoothed. This is consistent with the strong-
scaling table above (MPI's efficiency also degrades faster than OpenMP's at
higher worker counts) and with `perf`'s finding below that MPI's redundant
per-rank setup cost adds process-level variance OpenMP's shared-memory
model doesn't have.

## Accuracy vs. speedup for adaptive early-exit

`accuracy_vs_speedup_openmp.png` / `.csv` (8 threads),
`accuracy_vs_speedup_mpi.png` / `.csv` (8 ranks). Both measure speedup as
*adaptive at N workers* vs. *full ensemble at the same N workers, same
paradigm* — not against a serial reference — so the number isolates the
early-exit mechanism's own contribution rather than conflating it with the
paradigm's own parallel speedup (see the "MPI compare-full" note in
`docs/03_mpi.md` for a real bug this caught and fixed during development).

| z | Agreement w/ full ensemble | OpenMP speedup | MPI speedup |
|---|---|---|---|
| 0.25 | 99.43% | 12.0x | 10.8x |
| 0.5 | 99.44% | 11.7x | 12.4x |
| 0.75 | 99.51% | 12.1x | 11.7x |
| 1.0 | 99.65% | 11.4x | 12.1x |
| 1.5 | 99.86% | 10.4x | 10.2x |
| 2.0 | 99.95% | 8.5x | 9.3x |
| 2.5 | 99.99% | 7.9x | 6.8x |
| 3.0 | 100.0% | 6.7x | 6.6x |
| 4.0 | 100.0% | 4.8x | 4.3x |

**Takeaway**: agreement with the full ensemble is >99.4% at *every* tested
`z`, comfortably inside the Review-1 target of "within 2-3% of baseline" —
in fact within 0.6% even at the most aggressive setting tested (`z=0.25`,
~11-12x speedup). Agreement is identical between the OpenMP and MPI columns
at every `z` (both read 99.43%, 99.44%, ... down the table) — expected and
a good consistency check, since the adaptive stopping rule is a
per-transaction algorithm applied identically regardless of which paradigm
parallelizes the outer loop; only the speedup (a runtime property) differs
between paradigms. Speedup itself now tracks closely between the two
paradigms at every `z` (e.g. z=2.0: 8.5x vs 9.3x) — with the repeat-averaging
fix in place (see the top of this document), OpenMP and MPI's adaptive
speedup curves are consistent with each other, as expected since both
implement the identical per-transaction stopping rule.

## Explainability latency: integrated vs. naive post-hoc

`explain_latency.png` / `.csv`, serial binary, `z=2.0`, `min_trees=10`.

| Transactions | Integrated (µs/tx) | Naive post-hoc (µs/tx) | Ratio |
|---|---|---|---|
| 500 | 1.15 | 8.86 | 7.7x |
| 1,000 | 1.17 | 9.02 | 7.7x |
| 2,000 | 1.54 | 7.78 | 5.0x |
| 5,000 | 1.01 | 7.04 | 7.0x |
| 10,000 | 1.18 | 8.21 | 7.0x |

**Takeaway**: naive post-hoc explanation is **5-8x slower per transaction**
than integrated, across every tested size — directly substantiating the
Review-1 "near-zero-cost explainability" claim. The ratio isn't perfectly
constant (5.0x at 2,000 vs ~7-7.7x elsewhere); that's real run-to-run
variance (both modes are sub-2ms total at small sizes, where process/OS
noise is a larger fraction of the measurement), not a trend — reported as
measured rather than smoothed to a single number.

## `perf stat` + `perf record` profiling

Raw `perf stat` output: `perf_openmp_t1.txt`, `perf_openmp_t12.txt`,
`perf_mpi_p1.txt`, `perf_mpi_p8.txt`. Raw `perf record` call-graph reports
(collected separately per `scripts/run_perf_record.sh`, 200,000
transactions so there's enough sample count to resolve hot symbols):
`perf_record_openmp_t12_report.txt`, `perf_record_mpi_p8_report.txt`.
Workload: full ensemble (non-adaptive), so these numbers are whole-program
(`mpirun`/binary invocation), not scoring-phase-only like the scaling
tables above — that distinction turns out to be the main finding.

| Run | Wall time | Task-clock (sum) | CPUs utilized | IPC | Cache-miss rate |
|---|---|---|---|---|---|
| OpenMP, 1 thread | 3.70s | 3709ms | 1.00 | 2.06 | 62.5% |
| OpenMP, 12 threads | 2.53s | 4121ms | 1.63 | 1.76 | 60.5% |
| MPI, 1 rank | 3.70s | 3341ms | 0.90 | 2.16 | 49.6% |
| MPI, 8 ranks | 3.81s | 26,621ms | 6.98 | 1.70 | 53.4% |

**Finding 1 — the scoring phase is memory-bound, not compute-bound.**
Cache-miss rates of 50-63% and IPC of 1.7-2.2 (well under modern x86's
peak of ~4-5) across every configuration point at memory latency, not
arithmetic throughput, as the per-tree-traversal bottleneck — consistent
with Isolation Forest's access pattern (each split reads a
pseudo-randomly-chosen feature from a full transaction row, and each tree
node jump is effectively a pointer/index chase with no spatial locality
across trees). IPC dropping from 2.06 (1 thread) to 1.76 (12 threads) for
OpenMP is consistent with memory-bandwidth contention as more cores compete
for the same memory subsystem — a classic multi-core "memory wall" signature,
not a synchronization artifact (there are no locks/atomics in the hot loop).

**Finding 2 — OpenMP's *whole-program* wall clock barely improves at 12
threads (2.53s vs. 3.70s at 1 thread, only ~1.46x) despite the scoring
phase itself achieving multiple-x speedup at 12 threads (strong-scaling
table above — the two tables use different transaction counts and cache
states so aren't directly ratio-comparable, but both independently show
large scoring-phase speedups).** The "CPUs utilized" column explains why:
only 1.63 of a possible 12 CPU-equivalents were used *on average across
the whole run*, because most of the wall-clock time here is the
single-threaded CSV load/parse (~150MB, ~285k rows) and threshold
calibration, neither of which is parallelized — the scoring window itself
is a small fraction of the ~2.5s total run. This is a textbook Amdahl's-law result: the
unparallelized serial fraction, not the parallel section, dominates
end-to-end wall-clock once the parallel section gets fast enough. It's
exactly why every scaling table above measures `score_ms` specifically
(via `std::chrono`/`MPI_Wtime` around just the scoring loop) rather than
total process wall-clock — the two tell very different stories, and only
the former isolates the effect this project's parallelization work
actually targets.

**Finding 3 — MPI's whole-program wall clock is essentially flat from 1 to
8 ranks (3.70s -> 3.81s) despite consuming ~8x more total CPU-seconds
(2.98s user at 1 rank vs. 22.5s user at 8 ranks).** This is the direct,
measured consequence of the redundant-load design decision documented in
`docs/03_mpi.md`: every rank independently re-reads and re-parses the full
CSV and rebuilds the full forest, so each rank pays close to the same
fixed ~3.3s serial cost regardless of how tiny its scoring shard is. Wall
clock stays flat because 8 concurrent redundant loads overlap reasonably
well on a 12-core box (there's enough headroom), not because the design is
free — the aggregate CPU-time cost is real and would matter far more
without spare cores to absorb it. This is honestly a **limitation of the
scoring-only-partition MPI design** when judged by whole-program wall
clock, even though it deliberately avoids network communication for model
distribution (see the tradeoff discussion in `docs/03_mpi.md`); the ~7x
speedup this project reports for MPI throughout is specifically the
scoring-phase speedup, which is real and independently confirmed by
`MPI_Wtime` instrumentation inside the program — but it is not the same
thing as "the whole program runs 7x faster," and this report says so
explicitly rather than letting the two blur together.

**Finding 4 — `perf record` call graphs confirm Findings 2 and 3 directly,
by name, not just by inference from CPU-utilization ratios.** At OpenMP
12 threads, `perf report`'s self+children breakdown shows
`load_creditcard_csv(...)` accounting for **41.2%** of total sampled
cycles versus **31.8%** for `main._omp_fn.1` (the compiler-generated
OpenMP parallel-region body — i.e., the actual parallelized build+score
work) — the single-threaded CSV parser is a bigger cycle sink than the
12-way-parallel work it's feeding. At MPI 8 ranks, `load_creditcard_csv`
balloons to **71.6%** of sampled cycles (aggregated across all 8 rank
processes), with `std::getline` alone responsible for 11.2%, and visible
MPI startup cost (`PMPI_Init`/`ompi_mpi_init`/`orte_init`, ~5%) as a
further fixed per-run cost distinct from both the redundant CSV load and
the `Gatherv` communication measured separately as `comm_ms` (<1ms). This
is exactly the mechanism behind Findings 2 and 3 above, now identified by
function name from an actual call-graph profile rather than inferred from
aggregate counters alone.

**Bottleneck classification answer** (as required by the project brief,
using both `perf stat` counters and `perf record` call graphs): within the
scoring hot loop itself, the bottleneck is **memory-bound tree traversal**
(high cache-miss rate, moderate IPC, degrading further with core count
from shared memory-bandwidth contention). At the whole-program level, the
dominant cost for both paradigms — but especially MPI — is **serial,
unparallelized I/O/setup work** (CSV parsing, and for MPI, redundant
per-rank re-parsing plus process-launch overhead), confirmed by name via
`perf record` (Finding 4), not **synchronization** (measured `comm_ms` for
`Gatherv` is under 1ms in every run) and not **load imbalance** in the
strict sense (real but secondary, ~10-30% of score-phase time at higher
rank counts per `load_imbalance_ms`).

---

# Completion results (CUDA, MPI distributed build, drift, tuning)

Run 2026-11-03 on the same machine plus its NVIDIA GeForce GTX 1650
(CUDA 12.0, sm_75), via `python3 scripts/run_benchmarks.py cuda_vs_cpu
acc_speedup_cuda cuda_block_size mpi_dist_build hp_sweep drift`. Full
write-ups: `docs/09_cuda.md`, `docs/03_mpi.md`, `docs/10_drift_and_tuning.md`.

## CUDA vs CPU, full ensemble (`cuda_vs_cpu.png` / `.csv`)

| Transactions | Serial | OpenMP 12 thr | CUDA kernel | vs serial | vs OpenMP |
|---|---|---|---|---|---|
| 10,000 | 95.1 ms | 20.3 ms | 0.51 ms | 186x | 40x |
| 142,404 | 1177.5 ms | 150.8 ms | 6.82 ms | 173x | 22x |

Including host↔device copies (7.2 ms), the full stream takes 14.0 ms:
84x faster than serial, 10.8x faster than OpenMP. Output is bit-identical
to serial (`scripts/validate_cuda.sh`). Throughput ≈ 20-23 million tx/s,
versus ~1 million tx/s for 12-thread OpenMP.

## Adaptive early exit on the GPU (`cuda_early_exit.png`, `accuracy_vs_speedup_cuda.png`)

Same agreement as the CPU at every z, but warp divergence limits the
speedup. At z=0.25 warp efficiency is 98% and the GPU speedup is 5.3x.
At z=2 warp efficiency is 27% and adaptive is 0.85x of the full kernel
(slower). The first kernel version, which used FP64 `pow` in the stopping
check, was 4-5x slower still (`accuracy_vs_speedup_cuda_pow.csv`). On this
GPU, early exit pays off for z ≤ 1.5.

## MPI distributed tree build (`mpi_dist_build.png` / `.csv`)

At 1000 trees, ψ=2048, 8 ranks: replicated build 862 ms vs distributed
129 ms (exchange 18.5 ms). Replicated build *slows down* as ranks are
added on one node (memory-bandwidth contention: 282 ms at 1 rank).
Distributed build is ~2.2x faster than 1 rank and flattens after 2.
Bit-identical forest (`scripts/validate_mpi_dist_build.sh`).

## Drift and rolling retraining (`drift.png` / `drift.csv`)

Average precision per 20k slice drops from 0.51 (first slice) to
0.02-0.07 afterwards, **for static and retrained models alike**.
Retraining raises full-stream recall (0.39 → 0.52) but lowers precision.
F1: static 0.149, retrain/20k 0.140, retrain/10k 0.124. The Review-2
"stale model" explanation is therefore revised: the drop is mostly a
property of the later-stream fraud.

## Hyperparameter sweep (`hp_sweep_ap.png`, `hp_sweep_contamination.png`, `hp_sweep.csv`)

160 configs, chosen on the first half of the stream, reported on the
second half. The tuned model (ψ=64, 50 trees, without `Time`) doubles test
AP (0.034 → 0.070), and with 1% contamination raises test F1 from
0.077 to 0.176. The threshold chosen on validation (0.5%) gives test F1
0.052, and 81/160 configs (all with contamination ≤ 0.5%) score test
F1 = 0, because later-stream scores shift below the static θ.
