# Dataset & Streaming Replay

## Source

Kaggle "Credit Card Fraud Detection" (`mlg-ulb/creditcardfraud`), downloaded via
the Kaggle API (`kaggle datasets download -d mlg-ulb/creditcardfraud`).

## Size & features

- 284,807 transactions, 31 columns: `Time`, `V1`..`V28`, `Amount`, `Class`.
- `V1`..`V28` are PCA-transformed, already zero-mean/unit-variance-ish (whitened);
  the paper/dataset authors withheld the original features for confidentiality.
- `Time`: seconds elapsed since the first transaction in the dataset (spans ~2 days).
- `Amount`: transaction amount.
- `Class`: 1 = fraud, 0 = normal. 492 fraud out of 284,807 rows (0.1727%) —
  severe class imbalance, as expected for real fraud data.
- The CSV's `Class` column is quoted (`"0"`/`"1"`) in the raw file; the loader
  (`src/common/dataset.hpp`) strips surrounding quotes before numeric parsing.

## Preprocessing

- `Time` and `Amount` are standardized (zero mean, unit variance) in
  `standardize_columns()`; `V1`..`V28` are used as-is since they're already
  PCA-whitened by the dataset's original authors.
- No rows are dropped and no synthetic resampling (SMOTE etc.) is applied —
  Isolation Forest is unsupervised and doesn't need class balance; labels are
  used only for evaluation, never for training.

## Train / stream split

The CSV is already ordered by ascending `Time`. `split_train_stream()`
(`src/common/split.hpp`) takes a `--train-frac` (default 0.5) and does a
**chronological prefix/suffix split**, not a random shuffle:

- First `train_frac` of rows (~142k) -> the "training pool" that tree
  subsamples are drawn from (unsupervised — labels are never read here).
- Remaining rows (~142k) -> the "stream" that gets replayed/scored.

This simulates the realistic deployment story from the proposal: a forest
trained on historical data, then scoring newly-arriving transactions.
See `docs/07_known_limitations.md` for an honest note on what this split
does *not* control for (temporal drift between the two halves).

## Streaming replay harness

Two replay modes are implemented, both driven by the same `--score-n` flag
that caps how many stream rows are processed:

1. **Benchmark / throughput mode** (used for all scaling and accuracy
   experiments in this project): the stream is processed back-to-back, as
   fast as the implementation allows, so wall-clock measurements reflect
   pure compute+communication cost rather than injected delay. This is what
   `--score-n N` controls in `serial`, `openmp`, and `mpi`.
2. **Rate-limited demo mode** (`scripts/replay_demo.py`): replays a small
   subset of the stream at a configurable transactions/sec rate using
   real-time pacing, to demonstrate end-to-end latency under a simulated
   arrival process rather than to produce benchmark numbers. Not used for
   any of the reported performance results — see script docstring.

Rationale: the raw `Time` column's inter-arrival gaps span from
sub-second to tens of thousands of seconds, so literally sleeping between
every transaction to match real inter-arrival times would take ~2 days of
wall-clock time to replay the full stream. That's fine for a short demo of
a handful of transactions, but useless for benchmarking, hence the split
into two explicit modes.

## Where the raw data actually lives

The repository's `data/creditcard.csv` is the canonical copy (gitignored,
too large to commit). All benchmark runs in this project were executed
from a **native WSL2 (ext4) filesystem copy** at `/root/adaptguard/data/`,
not from the Windows-mounted `/mnt/c/...` path — the 9p/DrvFs bridge WSL2
uses to reach Windows drives has high per-syscall overhead and measurably
slows down large sequential file reads, which would have contaminated the
build/score timing numbers. Source code is still edited/compiled from the
Windows-mounted project directory; only the data + build artifacts used for
timed runs are on native Linux storage.
