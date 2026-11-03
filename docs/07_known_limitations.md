# Known Limitations & Future Work

What's simplified, what's missing, and what a production version would
still need. Maps onto the report's Conclusion / Future Work section.
(The Review-2 version listed CUDA, retraining, tuning, distributed build
and unit tests as missing. All five are now done; see `docs/00_status.md`.)

## Cannot be done with the available hardware

- **MPI runs are single-node.** `mpirun -np N --oversubscribe` on one
  12-core WSL2 machine, not across separate nodes with a real network.
  The measured `comm_ms` (Gatherv, < 1 ms) and `build_comm_ms`
  (Allgatherv, ≤ 28 ms) go through shared memory. A real cluster would add
  network latency and bandwidth limits. On the other hand, it would
  remove the memory-bandwidth contention that makes replicated build
  *slower* with more ranks here (`docs/03_mpi.md`). The code itself is
  node-agnostic: the same binary runs unchanged with a hostfile.

## Measured limitations

- **Detection quality on the later stream is low for every
  configuration tried** (test AP 0.026-0.070 across 160 configs, vs
  0.51 on the first 20k streamed transactions). Neither retraining nor
  tuning fixes this (`docs/10_drift_and_tuning.md`). It's a limit of
  unsupervised Isolation Forest on these PCA features. A
  supervised/semi-supervised model using delayed fraud labels would be
  the next step and is outside this project's unsupervised scope.
- **A static threshold calibrated on history doesn't transfer.**
  Later-stream scores shift downwards, so a θ fixed on training data
  flags too few transactions. Rolling re-calibration fixes the flag rate
  (but not ranking). Setting the threshold by alert budget (top-K) is the
  more robust option and is not implemented.
- **GPU early exit only pays off for z ≤ 1.5** on the GTX 1650 (warp
  divergence + 1/32-rate FP64). Untried improvements: sort/bin
  transactions by expected difficulty so warps agree, check the stopping
  rule every k trees, FP32 bookkeeping (the last two would give up the
  bit-exact match with the CPU). See `docs/09_cuda.md`.
- **Whole-program speedup is limited by serial CSV parsing** (Amdahl):
  41% of cycles at 12 OpenMP threads, 72% at 8 MPI ranks
  (`docs/06_results/results_summary.md`, Finding 4). A binary input
  format or parallel parser would fix it. Scaling tables therefore
  report the scoring phase.
- **MPI redundantly loads the CSV on every rank.** Rank 0 could read and
  `Scatterv` the stream shards instead. Not implemented.
- **CUDA host→device transfer is not overlapped with compute** (6.7 ms
  vs 6.8 ms kernel for the full stream). CUDA streams with double
  buffering would hide most of it for a real stream.

## Simplifications (reasonable for a course project)

- **Explainability is a heuristic attribution** (depth-weighted split
  usage along the isolation path), not SHAP. It is consistent with the
  trees the score actually used, and free, but has no axiomatic
  guarantees. Not ported to CUDA.
- **Rolling retraining is implemented in the OpenMP binary only.** It is
  a model-lifecycle feature, not a paradigm comparison.
- **Hyperparameter sweep uses one seed** (42). Small differences between
  neighbouring configurations (e.g. the strong ψ=64/50-tree corner) may
  be seed noise. Multi-seed averaging would tighten it.
- **Rate-limited streaming replay (`scripts/replay_demo.py`) is a demo,
  not a benchmark mode** (`docs/01_dataset.md`).
- **`perf` is a version-mismatched binary invoked directly** (WSL2's kernel
  isn't recognised by the distro wrapper). Counters were sanity-checked
  (IPC 1.7-2.2, consistent across runs), but this isn't the official
  build for this kernel.
- **No LOF implementation.** Isolation Forest was chosen
  (`docs/02_baseline.md`); the proposal left this as an either/or.
- **No model persistence** — every run trains from scratch (≈5-8 ms for
  the default forest).

## Future work (in priority order)

1. Run MPI on a real multi-node cluster.
2. Threshold by alert budget and/or semi-supervised use of delayed labels.
3. Divergence-aware GPU early exit (transaction binning, periodic checks).
4. Parallel/binary data loading to lift the Amdahl ceiling.
5. Overlap CUDA transfers with compute for a true streaming pipeline.
