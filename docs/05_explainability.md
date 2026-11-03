# Low-Overhead Explainability

Implemented in `IsolationForest::path_length_contrib`,
`score_adaptive_explain`, and `explain_posthoc`
(`src/common/isolation_forest.hpp`).

## How per-feature contributions are accumulated

Isolation Forest already traverses a root-to-leaf path per tree per
transaction to compute path length — explainability reuses that exact
traversal instead of running a separate attribution method (e.g., a
perturbation- or SHAP-style approach that would need extra passes over the
model).

At every **internal** node visited on the path (i.e., every split actually
used to route this specific transaction), the split's feature gets a
contribution weight:

```
contrib[feature] += 1.0 / (depth + 1)
```

where `depth` is the 0-indexed depth of that split on the path. Rationale:
a feature that isolates the point in a *shallow* split contributes more to
why the point got a short path length (i.e., a high anomaly score) than a
feature used deep in the tree, where the point was already mostly isolated
by other features. Contributions are accumulated across every tree actually
evaluated for that transaction, then normalized by `trees_used` at the end
so contributions are comparable between transactions that used different
numbers of trees (relevant when combined with adaptive early-exit).

The output is a per-transaction vector of length `n_features` (30); the
CLI (`--out path.csv` with `--explain integrated|posthoc`) writes the
top-3 contributing features by weight for every transaction predicted
anomalous, to `<path>.explain.csv`.

## Example output

From an actual run (`row_id` refers to the original dataset row):

```
row_id,true_label,pred_label,top1_feat,top1_contrib,top2_feat,top2_contrib,top3_feat,top3_contrib
142405,1,1,V5,0.2625,V7,0.233333,V25,0.183333
142514,0,1,V10,0.217674,V9,0.169551,V4,0.153846
```

Row 142405 is a true fraud case correctly flagged, with `V5`, `V7`, `V25`
as the top drivers of its anomaly score — this is the kind of
per-transaction explanation a fraud analyst would see alongside the score,
generated at (near) zero additional cost beyond scoring itself.

## Two ways to get contributions, and why one is "near-zero-cost"

- **Integrated** (`--explain integrated` -> `score_adaptive_explain`):
  contribution accumulation happens *inside* the same tree-traversal loop
  used to compute the score. If adaptive early-exit is active, contributions
  only reflect the trees actually visited — exactly the trees the score
  itself is based on, so there's no logical inconsistency between "why" and
  "what."
- **Naive post-hoc** (`--explain posthoc` -> `explain_posthoc`): scores the
  transaction first (via `score_adaptive` or `score_full`, same as any
  non-explaining run), then runs a **second, completely separate traversal
  of every tree** purely to accumulate contributions, ignoring any early
  exit the scoring pass took. This is what a bolted-on post-hoc explanation
  layer looks like — it doesn't get to reuse the scoring pass's work at all.

## Measured latency comparison

Single-threaded, `n_trees=100`, `psi=256`, `--adaptive --z 2.0 --min-trees 10`,
2,000 stream transactions:

| Mode | Total scoring+explain time | Per-transaction avg |
|---|---|---|
| Integrated (`score_adaptive_explain`) | ~2.3 ms | ~1.2 µs |
| Naive post-hoc (`score_adaptive` + `explain_posthoc`) | ~19.3 ms | ~9.6 µs |

**Naive post-hoc is ~8x slower per transaction than integrated**, in this
run — because post-hoc always pays for a full `n_trees`-tree traversal for
attribution regardless of how few trees the adaptive score itself needed,
while integrated's attribution cost is bounded by the same early-exit that
already bounds its scoring cost. This is the concrete evidence behind the
Review-1 novelty claim that explainability can be produced as a "near-zero
cost by-product" of adaptive scoring rather than a bolted-on post-hoc step.
The full latency sweep (more transaction counts, more `z`/`min_trees`
settings) is in `docs/06_results/` (`explain_latency.png`,
`explain_latency.csv`).

## Scope note

This comparison is run on the serial (and optionally OpenMP) binary only —
it's a single-transaction algorithmic cost question, not a distributed-
scaling one, so MPI wasn't a relevant paradigm to add it to. See
`docs/03_mpi.md`.
