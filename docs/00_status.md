# Project Status — Final

Last updated: 2026-11-03. Supersedes the Review-2 status (2026-09-16),
which put the project at ~75-80% of the proposal with CUDA deferred.

## Stage-by-stage status

| # | Stage | Status | Notes |
|---|---|---|---|
| 1 | Serial C++ baseline (Isolation Forest) | **Done** | `src/serial/`. Deterministic per-tree seeding, calibrated threshold. `docs/02_baseline.md`. |
| 2 | Dataset + streaming replay harness | **Done** | Kaggle creditcard.csv (284,807 tx), chronological train/stream split. `docs/01_dataset.md`. |
| 3 | OpenMP parallelization | **Done** | Build + scoring + threshold calibration parallelized; bit-identical to serial at 1/2/4/8 threads. `docs/03_openmp.md`. |
| 4 | MPI parallelization | **Done** | Stream partitioned via `Gatherv`; optional distributed tree build via `Allgatherv` (`--dist-build`). Both bit-identical to serial. `docs/03_mpi.md`. |
| 5 | **CUDA** | **Done (new)** | Full-ensemble and adaptive scoring kernels, one thread per transaction. Bit-identical to serial (full + adaptive). 173-205x faster than serial and 22x faster than 12-thread OpenMP on the full stream (kernel time). `docs/09_cuda.md`. |
| 6 | Adaptive early-exit (novelty) | **Done** | Welford + z·SE stopping rule in serial/OpenMP/MPI/CUDA. ~9-12x fewer trees at >99.4% agreement (CPU). On the GPU it pays off for z ≤ 1.5 (warp divergence + FP64 cost, measured). `docs/04_adaptive_early_exit.md`, `docs/09_cuda.md`. |
| 7 | Explainability layer | **Done** | Integrated attribution, 5-8x cheaper than post-hoc. `docs/05_explainability.md`. |
| 8 | **Rolling retraining / re-calibration** | **Done (new)** | `--retrain-every`, `--window` in the OpenMP binary. Evaluated honestly: it raises recall but not ranking quality or F1. The F1 drop turns out to be mostly a data property, not model staleness. `docs/10_drift_and_tuning.md`. |
| 9 | **Hyperparameter sweep** | **Done (new)** | 160 configs, chronological validation/test split. Tuned model doubles test AP; a tuned threshold doesn't transfer. `docs/10_drift_and_tuning.md`. |
| 10 | **Unit tests** | **Done (new)** | `tests/test_core.cpp`, `make test`: 3,352 checks, 0 failures. |
| — | Benchmarking & analysis | **Done** | Strong/weak scaling, accuracy-vs-speedup (OpenMP, MPI, CUDA), CUDA vs CPU, block size, MPI distributed build, drift, sweep, `perf`. `docs/06_results/`. |
| — | Multi-node MPI | **Not possible here** | Needs a second physical machine or a cluster. All MPI runs are single-node. `docs/07_known_limitations.md`. |

## Completion estimate

Every component of the Review-1 proposal is implemented, validated and
benchmarked. The one exception is running MPI across physically separate
machines, which needs hardware this project doesn't have (the MPI code
itself is node-agnostic).

## What "done" means here

- **Bit-identical outputs** across serial, OpenMP (1/2/4/8 threads), MPI
  (1/2/4/8 ranks), MPI `--dist-build` (1/2/3/4/8 ranks), and CUDA (full
  and adaptive, block sizes 64/128/256, 20k and 142k transactions). Each
  is checked by an exact diff script in `scripts/validate_*.sh`.
- **Unit tests** for the shared core (`make test`).
- Every number in `docs/` comes from a CSV produced by
  `scripts/run_benchmarks.py`. Negative or mixed results (GPU early exit
  at high z, retraining not improving F1, tuned threshold not
  transferring) are reported as measured.

## Corrections to earlier reviews

- Review 2 explained the F1 drop (0.51 → 0.15) as temporal drift from a
  never-retrained model. Rolling retraining shows otherwise: retrained
  models rank later fraud no better than the static one. The first 20k
  streamed transactions contain an unusual burst of easy fraud (39% of
  all streamed fraud). See `docs/10_drift_and_tuning.md`.
- `Timer` used `std::chrono::high_resolution_clock`, which in libstdc++ is
  the wall clock and can jump when WSL syncs time (one run reported a
  negative duration). It now uses `steady_clock`. The earlier scaling
  tables show no anomalous values (a jump would show up as a negative or
  wildly off timing), but they can't be proven jump-free. Runs from now
  on are protected.

## Environment

Windows 11 laptop, WSL2 Ubuntu 24.04, 12 logical cores, g++ 13.3 `-O3`,
OpenMPI 4.1.6, CUDA 12.0, NVIDIA GeForce GTX 1650 (4 GB, sm_75). Timed
runs use data and binaries on the native WSL ext4 filesystem
(`/root/adaptguard`); see `docs/01_dataset.md` for why. `perf` is a
directly-invoked, version-mismatched binary (WSL2 kernel); see
`docs/07_known_limitations.md`.
