# Adaptive Early-Exit Scoring (Novelty Component)

Implemented in `IsolationForest::score_adaptive` and
`score_adaptive_explain` (`src/common/isolation_forest.hpp`). This is the
project's centerpiece novelty piece — implemented as a real statistical
stopping rule, not a fixed "always use N trees" shortcut.

## Motivation

In a full-ensemble scan, every transaction pays for all `n_trees` tree
traversals even though, for the overwhelming majority of transactions
(>99.8% are normal in this dataset), the anomaly score converges to a
clearly-normal value after a handful of trees. The idea: stop evaluating
trees for a transaction as soon as we're statistically confident which
side of the decision threshold it's on, and only "pay full price" for the
genuinely ambiguous cases.

## Exact algorithm

For a transaction `x`, trees are evaluated one at a time, in the same
fixed order (`t = 0, 1, 2, ...`) they'd be evaluated in the full-ensemble
case. After each tree `t`'s path length `h_t = path_length(tree_t, x)`:

1. **Running mean/variance** of path length, updated online via Welford's
   algorithm (no need to store the individual `h_t` values):
   ```
   k += 1
   delta = h_t - mean
   mean += delta / k
   delta2 = h_t - mean
   M2 += delta * delta2
   ```
2. Once `k >= min_trees` (a minimum sample size before any stopping
   decision is trusted — default 10 — since variance/SE estimates from a
   handful of trees are unreliable), compute:
   ```
   variance = M2 / (k - 1)
   se = sqrt(variance / k)                         # standard error of the mean path length
   score_upper = score_to_anomaly(mean - z*se)     # optimistic bound (shorter mean path -> more anomalous)
   score_lower = score_to_anomaly(mean + z*se)      # pessimistic bound
   ```
   (`score_to_anomaly(m) = 2^(-m / c(psi))`, monotonically decreasing in
   `m`, hence the bound directions above.)
3. **Stopping rule**: if `score_lower > theta`, the confidence band lies
   entirely above the decision threshold — stop, label = anomaly (1). If
   `score_upper < theta`, the band lies entirely below — stop, label =
   normal (0). Otherwise, evaluate tree `t+1` and repeat.
4. If no stopping condition triggers before `k == n_trees`, finalize using
   the point estimate: `label = (score_to_anomaly(mean) > theta) ? 1 : 0`.

This is a sequential confidence-interval stopping rule (the mean path
length's confidence band shrinks as `~1/sqrt(k)`, standard for a running
mean of i.i.d.-ish samples across independently-seeded trees) — closely
related in spirit to Hoeffding-race-style early stopping used elsewhere in
ML for anytime/progressive algorithms. It is *not* a fixed "check the first
K trees and stop" heuristic: the number of trees actually used
(`trees_used` in the output) varies per transaction based on how quickly
its score separates from `theta`.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--adaptive` | off | Enables early-exit; without it, every transaction runs the full ensemble. |
| `--theta` | auto-calibrated | Decision threshold (see `docs/02_baseline.md`). |
| `--z` | 2.0 | Confidence multiplier on the standard error band. Higher `z` = more conservative = more trees used on average = closer to full-ensemble accuracy. |
| `--min-trees` | 10 | Minimum trees evaluated before any early-exit is allowed. |

## Wiring into OpenMP and MPI

Both `src/openmp/main.cpp` and `src/mpi/main.cpp` call the exact same
`score_adaptive` (or `score_adaptive_explain`) method per transaction —
the adaptive logic itself is not paradigm-specific. What each paradigm
adds is parallelizing *across* transactions (OpenMP: `parallel for` with
`schedule(dynamic, 64)`, chosen specifically because per-transaction cost
now varies enormously under adaptive scoring, so a static split would
badly load-imbalance; MPI: static block partition of the stream across
ranks, see `docs/03_mpi.md`).

The CUDA kernel (`src/cuda/main.cu`) runs the same rule per GPU thread,
with the band test rewritten in path-length space to avoid FP64 `pow()`
(identical decisions, 0 diff lines vs serial). On SIMT hardware the
speedup is limited by warp divergence; see `docs/09_cuda.md`.

## Validation: does it actually work?

Preliminary numbers (`--trees 100 --psi 256`, 20,000 stream transactions,
single-threaded, `--compare-full` against the same run's full-ensemble
labels):

| z | min_trees | avg trees used | agreement w/ full ensemble | score time vs. full ensemble |
|---|---|---|---|---|
| 2.0 | 10 | ~11.5 / 100 | 99.95% | ~10x faster |
| 1.0 | 10 | ~10.3 / 100 | 99.65% | ~12x faster |

Both configurations meet (comfortably beat) the Review-1 target of
"accuracy within 2-3% of baseline as throughput improves" — agreement loss
is under 0.4% even at the more aggressive `z=1.0` setting, for a ~10-12x
reduction in average trees evaluated. The full accuracy-vs-speedup sweep
across a range of `z`/`theta` settings, run against both the OpenMP and
MPI paths as required, is in `docs/06_results/` (see
`accuracy_vs_speedup_*.png` and `results_summary.md`) — this table is a
sanity-check preview, not the full reported result.
