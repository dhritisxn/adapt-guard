# AdaptGuard — Presentation Notes

Everything you need to present this project, in one place. Every number
comes from a CSV in `docs/06_results/data/` (real measured runs). Deeper
detail lives in `docs/` (file names given in each section).

---

## 1. One-line pitch

**AdaptGuard detects fraudulent credit-card transactions in a stream,
in parallel on CPU (OpenMP), cluster (MPI) and GPU (CUDA). It spends
compute only where it's needed: easy transactions stop after ~10 trees
instead of 100, and every flagged transaction comes with a free
explanation of *why*.**

Full title: *Adaptive Compute Allocation for GPU-Accelerated, Explainable
Anomaly Detection in Streaming Transaction Data* (HPC Mini Project).

## 2. The problem

- Fraud detection must score huge volumes of transactions fast.
- Fraud is rare: **492 of 284,807 transactions (0.17%)**. Almost every
  transaction is obviously normal, yet a standard model spends the same
  compute on all of them. That's the waste we target.
- Analysts need to know *why* something was flagged, but explanation
  methods (SHAP etc.) usually need extra passes over the model.

## 3. Contributions (memorise these)

1. **One algorithm, four implementations**: serial C++, OpenMP, MPI and
   CUDA, all producing **bit-identical output** (checked with exact
   `diff`, 0 lines).
2. **Adaptive early exit (main novelty)**: a statistical stopping rule.
   ~9-12x fewer trees at >99.4% agreement with the full model.
3. **Near-zero-cost explainability**: attributions from the same tree
   walk, 5-8x cheaper than a post-hoc pass.
4. **GPU: 173x faster than serial, 22x faster than 12-core OpenMP**
   (scoring kernel), with a measured analysis of why early exit behaves
   differently on GPUs.
5. **An honest evaluation of detection quality**: drift, rolling
   retraining and a 160-configuration tuning sweep. This corrected our
   own earlier conclusion (Section 10).

## 4. Dataset

- Kaggle "Credit Card Fraud Detection" (`mlg-ulb/creditcardfraud`).
- 284,807 rows over ~2 days, 30 features: `Time`, `V1..V28`
  (PCA-anonymised by the bank), `Amount`. Label `Class` (1 = fraud).
- Preprocessing: only `Time` and `Amount` standardised. No resampling.
  The model is **unsupervised**: labels are used only to evaluate.
- **Chronological split**: first 50% = training pool (~142k), last 50% =
  the "live stream" (~142k). It simulates "train on history, score new
  traffic". A random shuffle would leak the future.

## 5. The algorithm: Isolation Forest (explain simply)

- Idea: anomalies are **few and different**, so random cuts isolate them
  quickly. Normal points need many cuts.
- 100 trees, each built from 256 random rows (`psi`). At each node, pick
  a random feature and a random split value. Depth limit 8.
- Path length `h(x)` = depth at which x gets isolated. Short = suspicious.
- Score `s(x) = 2^(-E[h(x)] / c(psi))`. Near 1 = anomaly.
- **Threshold θ**: the 99th percentile of training scores
  (`contamination = 1%`), the same idea as scikit-learn.
- **Why Isolation Forest?** Every tree and every transaction is
  independent → embarrassingly parallel. LOF needs nearest-neighbour
  search, which is much harder to parallelise.

Implementation points worth saying:
- C++17 from scratch, one shared header-only core
  (`src/common/isolation_forest.hpp`).
- Trees stored as a **flat array of nodes** (indices, not pointers):
  cache-friendly, and it can be copied straight into an MPI message or to
  the GPU.
- **Deterministic seeding**: tree `t` always uses seed `42 + t`, so every
  implementation builds the *same* forest whatever the thread/rank order.
  That's what makes exact-diff validation possible.

## 6. The four implementations

**OpenMP** (`docs/03_openmp.md`): parallel tree build
(`schedule(dynamic)`) and parallel scoring (`schedule(dynamic, 64)`).
Dynamic scheduling matters because with early exit one transaction may
need 10 trees and another 100, and a static split would leave threads
idle. No locks, no atomics.

**MPI** (`docs/03_mpi.md`): the stream is split into contiguous shards per
rank, and results are collected with `MPI_Gatherv`. Timing stats come
via `MPI_Reduce` (max/min/sum → load imbalance). By default every rank
builds the same forest (a model replica per node, like real
deployments). **New: `--dist-build`** spreads tree building across ranks
and exchanges trees with two `MPI_Allgatherv` calls: 862 → 129 ms for a
large forest (1000 trees, psi 2048) at 8 ranks.

**CUDA** (`docs/09_cuda.md`): forest built on the CPU (5-8 ms, not worth
porting), flattened into one array, copied to the GPU once. **One GPU
thread per transaction.** Stream stored feature-major so neighbouring
threads' reads coalesce.

**How the GPU matches the CPU bit for bit** (examiners love this):
- leaf corrections are precomputed on the host;
- doubles are summed in the same order;
- `--fmad=false` stops the compiler fusing multiply-adds (different
  rounding);
- the final `pow()` runs on the host.

**Correctness, in one sentence**: serial is the reference, and OpenMP
(1/2/4/8 threads), MPI (1/2/4/8 ranks), MPI distributed build (1-8 ranks)
and CUDA (full + adaptive, 3 block sizes, 20k and 142k transactions) all
give **0 diff lines**. There are also unit tests: **3,352 checks, 0
failures** (`make test`).

## 7. Novelty 1: Adaptive early exit

Evaluate trees one at a time. After each tree:
1. Update the running mean and variance of path length (Welford's
   algorithm, O(1) memory).
2. After at least 10 trees, compute `SE = sqrt(var / k)` and the band
   `mean ± z·SE`.
3. If the **whole band** is above θ → stop, fraud. Whole band below →
   stop, normal. Otherwise evaluate another tree.
4. Never confident → use all 100 trees (same as the full model).

Analogy: *a doctor who stops ordering tests once the diagnosis is certain
and runs the full panel only for borderline patients.* `z` is the
caution knob. Related ML idea: Hoeffding races / sequential testing.

**CPU results (50,000 tx, 8 workers):**

| z | Agreement with full model | OpenMP speedup | MPI speedup |
|---|---|---|---|
| 0.25 | 99.43% | 12.0x | 10.8x |
| 1.0 | 99.65% | 11.4x | 12.1x |
| 2.0 (default) | 99.95% | 8.5x | 9.3x |
| 3.0 | 100.0% | 6.7x | 6.6x |

Speedup = adaptive vs full model on the *same* number of workers, so it
isolates the early-exit effect. Review-1 target: within 2-3% of baseline
accuracy. We're within 0.6% even at the most aggressive setting.

**On the GPU it's different, and that's an interesting finding:**

| z | GPU speedup | Warp efficiency |
|---|---|---|
| 0.25 | **5.3x** | 98% |
| 1.0 | 2.1x | 57% |
| 1.5 | 1.2x | 35% |
| 2.0 | 0.85x (slower) | 27% |

Why: GPU threads run in **warps of 32 in lockstep**, so a warp finishes
only when its slowest transaction does. One borderline transaction makes
the other 31 wait. Warp efficiency = useful work ÷ issued work. Also, the
GTX 1650 does double-precision maths at 1/32 speed, so the per-tree
stopping check is expensive. Our first GPU version called `pow()` twice
per tree and was 4-5x *slower* than the full model. Rewriting the test as
`mean + z·SE < h_θ` (same maths, no `pow`) made it 4.4x faster with
identical decisions. **Conclusion: on this GPU, use z ≤ 1.5.**

## 8. Novelty 2: Explainability at near-zero cost

- While walking a tree, every split the transaction passes adds
  `1 / (depth + 1)` to that feature's score. Shallow splits (which isolate
  fastest) count most. Top-3 features are reported per flagged
  transaction. Example: row 142405 (real fraud, caught) → `V5, V7, V25`.
- **Integrated** (same walk as scoring, respects early exit) vs **naive
  post-hoc** (separate walk over all 100 trees):

| Transactions | Integrated µs/tx | Post-hoc µs/tx | Ratio |
|---|---|---|---|
| 500 | 1.15 | 8.86 | 7.7x |
| 2,000 | 1.54 | 7.78 | 5.0x |
| 10,000 | 1.18 | 8.21 | 7.0x |

## 9. Performance results

Machine: 12-thread laptop CPU, WSL2 Ubuntu 24.04, g++ 13.3 `-O3`, OpenMPI
4.1.6, CUDA 12.0, NVIDIA GTX 1650 (4 GB). 100 trees, psi 256.

**Headline: scoring all 142,404 streamed transactions (full model)**

| Implementation | Time | Throughput | vs serial |
|---|---|---|---|
| Serial | 1177 ms | 0.12 M tx/s | 1x |
| OpenMP, 12 threads | 151 ms | 0.94 M tx/s | 7.8x |
| CUDA kernel | 6.8 ms | 20.9 M tx/s | **173x** |
| CUDA incl. copies to/from GPU | 14.0 ms | — | 84x |

**Strong scaling (100,000 tx fixed):**

| Workers | OpenMP speedup (eff.) | MPI speedup (eff.) |
|---|---|---|
| 2 | 1.81x (91%) | 1.88x (94%) |
| 4 | 3.78x (95%) | 2.82x (71%) |
| 8 | 6.19x (77%) | 5.65x (71%) |
| 12 | 8.95x (75%) | 6.51x (54%) |

OpenMP scales better on one machine: shared memory, no per-process
overhead. MPI's load imbalance jumps at 12 ranks (6.3 → 16.5 ms).

**Weak scaling (8,000 tx per worker)**: OpenMP nearly flat (+26% time for
12x the work, close to ideal). MPI +101%, noisier.

**Profiling — what's the bottleneck?** (`perf stat` + `perf record`)
- Inside scoring: **memory-bound**. Cache-miss rate 50-63%, IPC 1.7-2.2
  (the CPU could manage ~4). Tree walking is pointer chasing with random
  feature reads. IPC falls with more threads → memory-bandwidth
  contention.
- Whole program: **serial CSV loading dominates** (Amdahl's law): 41% of
  cycles at 12 OpenMP threads, 72% at 8 MPI ranks.
- **Not** synchronisation: `Gatherv` < 1 ms.

## 10. Detection quality: drift, retraining, tuning (`docs/10_drift_and_tuning.md`)

This section corrects our own earlier claim, so present it as a
strength: *we tested our hypothesis and it was wrong.*

**The observation**: F1 = 0.51 on the first 20k streamed transactions but
only 0.15 on the whole stream.

**Review-2 hypothesis**: the model was trained once and goes stale →
fix it with rolling retraining.

**What we built**: `--retrain-every N --window W`. Every N transactions,
rebuild the forest on the last W rows that have already arrived and
re-calibrate θ. Still unsupervised. ~30-60 ms per retrain with 12 threads.

**What we found**: ranking quality (average precision, threshold-free)
per 20k slice drops from **0.51 to 0.02-0.07, and retrained models are no
better than the static one**. So the model isn't stale. The first slice
holds an unusual burst of easy fraud (**87 of 223 frauds in 14% of the
stream**), and later fraud looks normal to *any* unsupervised model on
these features. Retraining raised recall (0.39 → 0.52) but lowered
precision. F1: 0.149 static vs 0.140 retrained.

**Hyperparameter sweep**: 160 configs (psi × trees × contamination ×
with/without Time). Picked on the first half of the stream, reported on
the second half (never tuned on the reported data).
- **Tuning the model works**: test average precision doubles, 0.034 →
  0.070. With the standard 1% threshold, test F1 rises from 0.077 to
  **0.176**.
- **Tuning the threshold doesn't transfer**: later scores shift down, so
  a threshold fixed from history flags almost nothing. 81 of 160 configs
  scored test F1 = 0.
- **Lesson**: tune the model for ranking, and don't freeze the threshold.
  Re-calibrate it or use an alert budget ("review the top K per hour").

## 11. Likely questions and answers

- **Why Isolation Forest, not deep learning?** Unsupervised (fraud labels
  arrive late and are scarce), linear time, tiny memory, embarrassingly
  parallel. Ideal for an HPC project.
- **How do you know the parallel versions are correct?** Exact `diff`
  against serial, 0 lines, for every implementation and worker count,
  plus 3,352 unit-test checks.
- **How can a GPU give *bit-identical* results?** Same summation order in
  double precision, FMA contraction disabled, `pow` on the host (§6).
- **Does early exit hurt accuracy?** z=2: 99.95% of decisions match the
  full model; z ≥ 3: 100%. One parameter sets the trade-off.
- **Why doesn't early exit speed up the GPU at z=2?** Warp divergence (a
  warp waits for its slowest lane, 27% efficiency) plus slow FP64 on
  consumer GPUs. It does help at z ≤ 1.5 (up to 5.3x). Fixes: group
  similar-difficulty transactions into warps, check every few trees.
- **Why is MPI slower than OpenMP?** On one machine MPI pays process
  launch, per-rank data loading and oversubscription. Its advantage
  (many machines) can't show on one node.
- **Did you run MPI on multiple machines?** No, we only had one machine.
  The code is node-agnostic (just add a hostfile). Network latency would
  increase `comm_ms`, but replicated build would stop slowing down, since
  that slowdown comes from ranks sharing one machine's memory bandwidth.
- **Why is F1 low?** 0.17% fraud makes precision hard, and later-stream
  fraud is intrinsically hard for unsupervised detection (§10). We tested
  retraining and tuning rather than hiding it.
- **Why not parallelise CSV loading?** It's the true end-to-end
  bottleneck (Amdahl). We report scoring time separately, and
  whole-program numbers honestly. Listed as future work.
- **Is the explanation trustworthy?** It's path-based attribution, not
  SHAP-exact, but it's free and always consistent with the trees the
  score actually used.

## 12. Live demo (WSL Ubuntu, from `/root/adaptguard`)

```
make all cuda test
./build/serial --data data/creditcard.csv --score-n 20000
./build/openmp --data data/creditcard.csv --threads 12 --adaptive --compare-full
mpirun -np 8 --oversubscribe ./build/mpi --data data/creditcard.csv --adaptive --compare-full
./build/cuda --data data/creditcard.csv                      # whole stream in ~7 ms
./build/cuda --data data/creditcard.csv --adaptive --z 0.5 --compare-full
./build/serial --data data/creditcard.csv --score-n 2000 --adaptive --explain integrated --out /tmp/o.csv
head /tmp/o.csv.explain.csv
bash scripts/validate_cuda.sh                                # expect all 0
```

Each run prints one `RESULT key=value ...` line. Point at `score_ms`,
`throughput_tx_s`, `avg_trees_used`, `agreement_full`, `warp_efficiency`
(CUDA), `f1`. If running as root, MPI needs
`export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.

## 13. Suggested slide order (~15 min)

1. Title + one-line pitch
2. Problem: rare fraud, wasted compute, need explanations
3. Dataset + chronological split
4. Isolation Forest in one picture
5. Architecture (`docs/06_results/plots/architecture_diagram.png`)
6. Correctness: deterministic seeds → bit-identical across 4 paradigms
7. Adaptive early exit: algorithm + doctor analogy + `accuracy_vs_speedup_openmp.png`
8. Explainability: example + `explain_latency.png`
9. CPU scaling: `strong_scaling_openmp.png`, `strong_scaling_mpi.png`
10. GPU: `cuda_vs_cpu.png` (173x)
11. Early exit on GPU: `cuda_early_exit.png` (warp divergence story)
12. MPI distributed build: `mpi_dist_build.png`
13. Profiling: memory-bound + Amdahl
14. Detection quality: `drift.png` + `hp_sweep_contamination.png`
15. Limitations + future work, Q&A

## 14. Honest limitations (say these before you're asked)

- MPI ran on one machine; real network costs weren't measured.
- Later-stream fraud is hard for unsupervised detection; tuning helps
  ranking but absolute quality stays low.
- GPU early exit only pays off for z ≤ 1.5 on this GPU.
- Whole-program speedup is capped by serial CSV loading.
- Explainability is heuristic, not SHAP; not ported to the GPU.
- The sweep used a single random seed.

## 15. Glossary

- **psi**: rows sampled per tree (256). **n_trees**: forest size (100).
- **θ (theta)**: decision threshold. **contamination**: assumed fraud
  rate used to set θ (1%).
- **z / min_trees**: early-exit confidence multiplier / minimum trees.
- **Strong scaling**: same work, more workers. **Weak scaling**: work
  grows with workers. **Efficiency** = speedup ÷ workers.
- **Average precision (AP)**: area under the precision-recall curve.
  Measures ranking, independent of threshold. Random ≈ fraud rate.
- **IPC**: instructions per cycle. **Amdahl's law**: the serial part
  caps total speedup.
- **Warp / SIMT**: a GPU executes 32 threads together. If they branch
  differently, the warp runs as long as the slowest one (divergence).
- **FMA**: fused multiply-add, one rounding instead of two.
- **Allgatherv / Gatherv**: MPI collectives that combine variable-sized
  pieces from all ranks (to everyone / to one rank).
