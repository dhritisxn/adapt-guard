# AdaptGuard

Adaptive Compute Allocation for GPU-Accelerated, Explainable Anomaly
Detection in Streaming Transaction Data — HPC Mini Project.

An Isolation Forest fraud detector implemented four ways (serial C++,
OpenMP, MPI, CUDA), all producing **bit-identical** output, plus the
project's novelty components:
- **adaptive early-exit scoring**: stop evaluating trees once the decision
  is statistically certain; ~9-12x fewer trees at >99.4% agreement;
- **near-zero-cost explainability**: per-feature attributions from the
  same tree walk as the score; 5-8x cheaper than a separate post-hoc pass.

The completion stage adds the CUDA path, MPI distributed tree
construction, rolling retraining, a hyperparameter sweep, and unit tests.

## Start here

- `PRESENTATION_NOTES.md` — everything needed to present the project.
- `DEMO_GUIDE.md` — step-by-step live demo script with expected outputs.
- `docs/00_status.md` — what's done, and corrections to earlier reviews.
- `docs/06_results/results_summary.md` — all benchmark findings.
- `docs/06_results/plots/architecture_diagram.png` — pipeline diagram.
- `docs/08_appendix_cli_reference.md` — every CLI flag and hyperparameter.

## Layout

```
src/
  common/          shared Isolation Forest core (used by all 4 binaries)
  serial/          serial baseline (correctness reference)
  openmp/          OpenMP version (+ rolling retraining)
  mpi/             MPI version (+ optional distributed tree build)
  cuda/            CUDA version (full + adaptive scoring kernels)
tests/
  test_core.cpp    unit tests for src/common (make test)
scripts/
  run_benchmarks.py         drives every experiment, writes the CSVs
  plot_results.py           reads those CSVs, writes the PNGs
  plot_architecture_diagram.py
  run_perf_profiling.sh     perf stat profiling runs
  run_perf_record.sh        perf record + report (call-graph hot spots)
  validate_openmp.sh        exact diff: OpenMP vs serial
  validate_mpi.sh           exact diff: MPI vs serial
  validate_mpi_dist_build.sh exact diff: MPI --dist-build vs serial
  validate_cuda.sh          exact diff: CUDA (full + adaptive) vs serial
  replay_demo.py            small rate-limited streaming demo (not a benchmark)
docs/
  00_status.md  01_dataset.md  02_baseline.md  03_openmp.md  03_mpi.md
  04_adaptive_early_exit.md  05_explainability.md  06_results/
  07_known_limitations.md  08_appendix_cli_reference.md
  09_cuda.md  10_drift_and_tuning.md
data/                       creditcard.csv goes here (gitignored, download via Kaggle)
```

## Build & run (Linux/WSL — g++ with OpenMP, OpenMPI, CUDA toolkit)

```
make all                       # build/serial, build/openmp, build/mpi
make cuda                      # build/cuda (needs nvcc; -arch=sm_75 in Makefile, adjust for your GPU)
make test                      # unit tests

./build/serial  --data data/creditcard.csv --trees 100 --psi 256
./build/openmp  --data data/creditcard.csv --threads 8 --adaptive
mpirun -np 8 ./build/mpi --data data/creditcard.csv --adaptive --compare-full
mpirun -np 8 ./build/mpi --data data/creditcard.csv --dist-build
./build/cuda    --data data/creditcard.csv --adaptive --z 1.0 --compare-full
./build/openmp  --data data/creditcard.csv --threads 12 --retrain-every 20000
```

Key flags: `--adaptive` enables early exit, `--z`/`--min-trees` tune
it, `--explain integrated|posthoc` (serial/openmp) enables explanations,
`--compare-full` also scores with the full ensemble for an
agreement/speedup comparison, `--out path.csv` dumps per-transaction
results. Full list: `docs/08_appendix_cli_reference.md`.

## Reproducing the results

```
python3 scripts/run_benchmarks.py      # every experiment (~10 minutes)
bash scripts/run_perf_profiling.sh
bash scripts/run_perf_record.sh
python3 scripts/plot_results.py        # regenerates every PNG
python3 scripts/plot_architecture_diagram.py
```

These were run from a native Linux filesystem path (`/root/adaptguard/`
in this project's WSL2 setup), not the Windows-mounted project directory.
See `docs/01_dataset.md` for why (9p/DrvFs I/O overhead skews timing).
Adjust `ROOT`/paths in the scripts if your data/build live elsewhere.

## Correctness

Every optimization (OpenMP, MPI, MPI distributed build, CUDA, adaptive
early exit) is validated against the serial baseline with an exact diff,
not a "looks close enough" comparison. Run `scripts/validate_*.sh` and
`make test` to reproduce. `docs/02_baseline.md` describes the method, and
`docs/09_cuda.md` explains how the GPU matches bit for bit.
