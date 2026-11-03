# Temporal Drift, Rolling Retraining, and Hyperparameter Tuning

Earlier reviews reported that F1 falls from 0.512 on the first 20k
streamed transactions to 0.149 on the full stream, and blamed a static
model that is never retrained. This document tests that explanation
with two additions:

- **Rolling retraining** in the OpenMP binary (`--retrain-every N
  --window W`).
- A **hyperparameter sweep** with a proper validation/test split
  (`--drop-time` added as a feature option).

**Short answer: the original explanation was mostly wrong.** Retraining
doesn't recover detection quality. Fraud in the later part of the stream
is much harder for *any* unsupervised Isolation Forest on these features
to separate. Tuning roughly doubles ranking quality on unseen data, but
a threshold tuned on early data doesn't carry over to later data.

All labels are used for **evaluation only**. Neither retraining nor
threshold calibration ever reads `Class`.

## Rolling retraining (implementation)

`src/openmp/main.cpp`. With `--retrain-every N`, the stream is scored in
chunks of `N` transactions. Before each chunk after the first:

1. Take the `W` most recent rows that have **already arrived** (window
   ending right before the chunk; `--window`, default 50,000).
2. Rebuild the forest on that window (OpenMP-parallel; seed
   `seed + retrain_index * n_trees`, so it's deterministic).
3. Re-calibrate `θ` as the `(1 - contamination)` quantile over the last
   ≤20,000 window rows (calibration is now OpenMP-parallel too).

Rebuild and re-calibration time is reported separately as `retrain_ms`,
so `score_ms` stays comparable with the other experiments. Cost on the
full stream: 14 retrains ≈ 0.4-0.8 s total with 12 threads (~30-60 ms
each). That's cheap compared with the hours of real time each 10k-row
chunk represents. `--retrain-every 0` (default) gives exactly the old
static behaviour, and the OpenMP-vs-serial validation still gives 0 diff
lines.

## Experiment 1: does retraining fix drift? (`drift.csv`, `drift.png`)

Default config (ψ=256, 100 trees, 1% contamination), full 142,404-row
stream, metrics per 20k-transaction slice. **AP** = average precision,
the area under the precision-recall curve. It measures how well the
score *ranks* fraud above normal, independent of any threshold.
Random ranking would give AP ≈ the fraud rate (≈0.002).

| Slice start | Frauds | AP static | AP retrain/20k | AP retrain/10k | Recall static | Recall retrain/20k |
|---|---|---|---|---|---|---|
| 0 | 87 | 0.509 | 0.509 | 0.513 | 0.71 | 0.71 |
| 20k | 9 | 0.017 | 0.025 | 0.026 | 0.33 | 0.33 |
| 40k | 22 | 0.017 | 0.018 | 0.018 | 0.00 | 0.09 |
| 60k | 23 | 0.029 | 0.044 | 0.027 | 0.35 | 0.65 |
| 80k | 30 | 0.034 | 0.055 | 0.040 | 0.20 | 0.47 |
| 100k | 34 | 0.067 | 0.060 | 0.048 | 0.21 | 0.38 |
| 120k | 18 | 0.028 | 0.026 | 0.042 | 0.11 | 0.44 |

Whole stream: static P/R/F1 = 0.092 / 0.395 / **0.149**; retrain every
20k = 0.081 / 0.525 / **0.140**; retrain every 10k = 0.071 / 0.475 /
**0.124**.

What this shows:

- **Ranking quality collapses after the first slice, and retraining
  doesn't bring it back.** AP drops from 0.51 to 0.02-0.07 for every
  model, including models retrained on the 50k transactions just before
  each slice. If staleness were the cause, retrained models would recover.
  They don't.
- **The first slice is unusual.** 87 of the stream's 223 frauds (39%)
  fall in its first 20k transactions (14% of the stream), and they're
  easy to isolate. The headline "F1 = 0.51 on the first 20k rows" from
  Review 2 comes from this burst, not from the model being fresh.
- **Retraining does raise recall** (0.39 → 0.52 over the stream), because
  re-calibration restores the target 1% flag rate whenever scores drift
  downwards. It flags more transactions, but since ranking doesn't
  improve, precision falls and F1 drops slightly.

**Conclusion**: the F1 drop is mainly a property of the data (later
fraud looks like normal traffic to an unsupervised model on these
features), not of the static model. Rolling retraining is implemented
and working, and it's the right design for a production system where
the *normal* traffic itself shifts. On this dataset, though, it doesn't
improve F1, and we report that rather than tune it until it does.

## Experiment 2: hyperparameter sweep (`hp_sweep.csv`)

160 configurations: ψ ∈ {64, 128, 256, 512, 1024} × n_trees ∈ {50, 100,
200, 400} × contamination ∈ {0.1%, 0.2%, 0.5%, 1%} × {all features,
without `Time`}. Seed 42 for all of them. To avoid tuning on the numbers
we report, the stream is split chronologically:

- **Validation**: first half of the stream (71,202 tx, 129 frauds), used to
  pick a configuration.
- **Test**: second half (71,202 tx, 94 frauds), touched only to report.

| Configuration | Val F1 | Val AP | **Test F1** | **Test AP** |
|---|---|---|---|---|
| Default (ψ=256, 100 trees, 1%, all features) | 0.205 | 0.172 | **0.077** | **0.034** |
| Best on validation F1 (ψ=64, 50 trees, 0.5%, without Time) | 0.422 | 0.303 | **0.052** | **0.070** |
| Same model, default 1% contamination | 0.317 | 0.303 | **0.176** | **0.070** |
| Best on validation AP (ψ=1024, 400 trees, all features) | — | 0.309 | — | **0.054** |

(`hp_sweep_ap.png`: validation AP across ψ × n_trees. `hp_sweep_contamination.png`:
validation vs test F1 across contamination for the default model.)

What this shows:

- **Tuning the model transfers.** Test AP (threshold-free ranking)
  doubles from 0.034 to 0.070 with the validation-selected model. The
  model ranks later fraud better, though it's still weak in absolute
  terms (test AP across all 160 configs: 0.026-0.070).
- **Tuning the threshold does not transfer.** Validation preferred 0.5%
  contamination, but on the test half that threshold flags almost
  nothing: later-stream scores sit lower, so a θ fixed from training
  data is crossed less often. **81 of the 160 configurations have test
  F1 = 0**, all at contamination ≤ 0.5%. The same tuned model with the
  default 1% contamination gets the best test F1 (0.176, vs 0.077 for
  the default model).
- **Dropping `Time` gives no consistent gain.** Mean validation AP is
  0.216 with `Time` and 0.197 without; mean test AP is 0.039 for both.
  `Time` is monotonic, so streamed values always fall outside the
  training range, but in practice this doesn't systematically hurt.
- Larger ψ with more trees is the most consistent improvement on
  validation (ψ=1024 × 400 trees: AP 0.309). The ψ=64 × 50 trees corner
  also scores well, which with a single seed is probably partly noise.

**Practical recommendation** (what to say if asked "so what would you
deploy?"): tune ψ/n_trees for ranking quality (AP), and don't fix the
threshold from history. Either keep re-calibrating it (rolling
retraining) or set it by alert budget ("review the top K transactions
per hour"), which ranking-based detectors support naturally. Supervised
or semi-supervised models using the (delayed) fraud labels would be the
next step beyond this project's unsupervised scope.

## Reproduce

```
python3 scripts/run_benchmarks.py hp_sweep drift   # ~5 minutes
python3 scripts/plot_results.py
```
