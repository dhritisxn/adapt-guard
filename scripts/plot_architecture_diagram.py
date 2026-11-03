#!/usr/bin/env python3
"""
Generates the AdaptGuard pipeline architecture diagram
(docs/06_results/plots/architecture_diagram.png) for the project report.
Pure matplotlib (boxes + arrows) -- no external diagramming dependency.
Palette matches scripts/plot_results.py (dataviz skill reference palette).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/root/adaptguard/plots/architecture_diagram.png"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
MUTED = "#898781"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(11, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14.5)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)


def box(cx, cy, w, h, text, edge=INK, fill=SURFACE, fontsize=10, fontweight="normal", alpha_fill=1.0, textcolor=INK):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.06,rounding_size=0.08",
                        linewidth=1.8, zorder=3)
    # Patch-level alpha (if set) overrides any per-channel alpha in face/edge
    # RGBA tuples, so leave it unset (None) and bake alpha into each color
    # independently -- this is what actually gives a light fill + solid edge.
    b.set_edgecolor((*matplotlib.colors.to_rgb(edge), 1.0))
    b.set_facecolor((*matplotlib.colors.to_rgb(fill), alpha_fill))
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor, zorder=4, linespacing=1.4)


def arrow(x0, y0, x1, y1, color=MUTED):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.6, color=color, zorder=2, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# --- Title ---
ax.text(5, 14.15, "AdaptGuard Pipeline Architecture", ha="center", va="center",
        fontsize=16, fontweight="bold", color=INK)

# --- Row 1: data source ---
box(5, 13.1, 7.6, 0.75, "Kaggle “Credit Card Fraud Detection”\ncreditcard.csv — 284,807 transactions, 30 features",
    edge=INK, fill=SURFACE, fontsize=9.5)
arrow(5, 12.72, 5, 12.28)

# --- Row 2: preprocessing + split ---
box(5, 11.9, 6.6, 0.65, "Preprocess: standardize Time & Amount  (src/common/dataset.hpp)",
    edge=INK, fill=SURFACE, fontsize=9.2)
arrow(5, 11.58, 5, 11.15)
box(5, 10.75, 5.2, 0.65, "Chronological split (src/common/split.hpp)", edge=INK, fill=SURFACE, fontsize=9.2)

arrow(3.6, 10.44, 2.5, 9.95)
arrow(6.4, 10.44, 7.5, 9.95)
box(2.5, 9.55, 3.0, 0.65, "Train pool (50%)\nunsupervised tree subsamples", edge=VIOLET, fill=VIOLET, alpha_fill=0.08, fontsize=8.6)
box(7.5, 9.55, 3.0, 0.65, "Stream (50%)\nreplayed / scored transactions", edge=VIOLET, fill=VIOLET, alpha_fill=0.08, fontsize=8.6)

arrow(2.5, 9.23, 5, 8.85)
arrow(7.5, 9.23, 5, 8.85)

# --- Shared core ---
box(5, 8.5, 7.4, 0.75, "Shared core: IsolationForest  (src/common/isolation_forest.hpp)\nDeterministic per-tree seed = seed_base + tree_index",
    edge=VIOLET, fill=VIOLET, alpha_fill=0.10, fontsize=9.3, fontweight="bold")
arrow(5, 8.12, 5, 7.75)

# --- Row: three paradigms ---
ax.text(5, 7.55, "Four implementations, identical core algorithm", ha="center",
        fontsize=9, color=MUTED, style="italic")

y_par = 6.7
xs = [1.25, 3.75, 6.25, 8.75]
for x in xs:
    arrow(5, 7.4, x, y_par + 0.5)

box(xs[0], y_par, 2.3, 1.05, "Serial\n(src/serial/)\ncorrectness reference\nsequential build + score",
    edge=MUTED, fill=MUTED, alpha_fill=0.10, fontsize=8.2)
box(xs[1], y_par, 2.3, 1.05, "OpenMP\n(src/openmp/)\nparallel for: build +\nscore + rolling retrain",
    edge=BLUE, fill=BLUE, alpha_fill=0.10, fontsize=8.2)
box(xs[2], y_par, 2.3, 1.05, "MPI\n(src/mpi/)\nstream shards + Gatherv\nopt. Allgatherv build",
    edge=ORANGE, fill=ORANGE, alpha_fill=0.10, fontsize=8.2)
box(xs[3], y_par, 2.3, 1.05, "CUDA\n(src/cuda/)\n1 GPU thread per tx\nflattened forest",
    edge=VIOLET, fill=VIOLET, alpha_fill=0.10, fontsize=8.2)

for x, x_end in zip(xs, [3.2, 4.4, 5.6, 6.8]):
    arrow(x, y_par - 0.53, x_end, 4.85)

# --- Per-transaction scoring layer (adaptive + explainability) ---
box(5, 4.5, 8.2, 1.15,
    "Per-transaction scoring (identical logic in every paradigm)\n"
    "Adaptive early-exit: Welford running mean/variance + z·SE confidence band vs. θ\n"
    "Integrated explainability: per-feature contribution accumulated in the same tree traversal",
    edge=AQUA, fill=AQUA, alpha_fill=0.10, fontsize=8.8, fontweight="bold")

arrow(5, 3.92, 5, 3.5)

# --- Validation ---
box(5, 3.15, 7.2, 0.7, "Correctness validation: exact diff vs. serial output (0 diff lines for OpenMP,\nMPI, MPI --dist-build, CUDA full + adaptive)  ·  unit tests (make test)",
    edge=INK, fill=SURFACE, fontsize=8.8)
arrow(5, 2.8, 5, 2.4)

# --- Benchmarking ---
box(5, 2.05, 8.6, 0.85,
    "Benchmarking: strong/weak scaling · accuracy-vs-speedup · CUDA vs CPU ·\n"
    "explainability latency · drift + rolling retraining · hyperparameter sweep · perf",
    edge=INK, fill=SURFACE, fontsize=8.6)
arrow(5, 1.62, 5, 1.2)

# --- Output ---
box(5, 0.85, 6.0, 0.6, "docs/06_results/  —  CSVs, PNGs, perf reports, results_summary.md",
    edge=INK, fill="#e1e0d9", alpha_fill=0.5, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {OUT}")
