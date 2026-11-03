#!/usr/bin/env python3
"""
Generates every required plot from the raw CSVs in docs/06_results/data/.
Run after scripts/run_benchmarks.py and scripts/run_perf_profiling.sh have
produced the CSVs. Every number plotted here is read from those files --
nothing is hand-entered.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/root/adaptguard/results"
PLOT_DIR = "/root/adaptguard/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Palette (see dataviz skill references/palette.md): fixed categorical order.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SEC_INK,
    "ytick.color": SEC_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    with open(path) as f:
        return list(csv.DictReader(f))


def savefig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def plot_strong_scaling(csv_name, out_name, label):
    rows = read_csv(csv_name)
    x = [int(r["threads"]) if "threads" in r and r["threads"] != "1" or "ranks" not in r else int(r["ranks"]) for r in rows]
    # use ranks for mpi, threads for openmp
    key = "ranks" if rows[0].get("binary") == "mpi" else "threads"
    x = [int(r[key]) for r in rows]
    score_ms = [float(r["score_ms"]) for r in rows]
    t1 = score_ms[0]
    speedup = [t1 / t for t in score_ms]
    efficiency = [s / n * 100 for s, n in zip(speedup, x)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.plot(x, speedup, marker="o", color=BLUE, linewidth=2, markersize=6, label=f"{label} (measured)")
    ax.plot(x, x, linestyle="--", color=MUTED, linewidth=1.5, label="ideal linear")
    ax.set_xlabel(f"{key}")
    ax.set_ylabel("speedup (vs 1 " + key[:-1] + ")")
    ax.set_title(f"Strong scaling: speedup")
    ax.legend(loc="upper left")

    ax2 = axes[1]
    ax2.plot(x, efficiency, marker="o", color=ORANGE, linewidth=2, markersize=6)
    ax2.axhline(100, linestyle="--", color=MUTED, linewidth=1.5)
    ax2.set_xlabel(f"{key}")
    ax2.set_ylabel("parallel efficiency (%)")
    ax2.set_title("Strong scaling: efficiency")
    ax2.set_ylim(0, 115)

    fig.suptitle(f"{label} strong scaling (fixed 100,000 transactions)", y=1.03)
    savefig(out_name)


def plot_weak_scaling(csv_name, out_name, label):
    rows = read_csv(csv_name)
    key = "ranks" if rows[0].get("binary") == "mpi" else "threads"
    x = [int(r[key]) for r in rows]
    score_ms = [float(r["score_ms"]) for r in rows]
    t1 = score_ms[0]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(x, score_ms, marker="o", color=BLUE, linewidth=2, markersize=6, label="measured")
    ax.axhline(t1, linestyle="--", color=MUTED, linewidth=1.5, label="ideal (flat)")
    ax.set_xlabel(f"{key} (workload scaled proportionally)")
    ax.set_ylabel("score time (ms)")
    ax.set_title(f"{label} weak scaling")
    ax.set_ylim(0, max(score_ms) * 1.3)
    ax.legend(loc="upper left")
    savefig(out_name)


def plot_accuracy_vs_speedup(csv_name, out_name, label):
    rows = read_csv(csv_name)
    rows = [r for r in rows if float(r.get("full_score_ms", -1)) > 0]
    speedup = [float(r["full_score_ms"]) / float(r["score_ms"]) for r in rows]
    accuracy = [float(r["agreement_full"]) * 100 for r in rows]
    z = [float(r["z"]) for r in rows]

    order = sorted(range(len(speedup)), key=lambda i: speedup[i])
    speedup = [speedup[i] for i in order]
    accuracy = [accuracy[i] for i in order]
    z = [z[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(speedup, accuracy, marker="o", color=BLUE, linewidth=2, markersize=6)
    for sx, sy, sz in zip(speedup, accuracy, z):
        ax.annotate(f"z={sz:g}", (sx, sy), textcoords="offset points", xytext=(6, -4),
                    fontsize=8, color=SEC_INK)
    ax.axhline(97, linestyle="--", color=MUTED, linewidth=1.3, label="2-3% degradation target floor")
    ax.set_xlabel("speedup vs full-ensemble baseline (same paradigm, same resource count)")
    ax.set_ylabel("agreement with full ensemble (%)")
    ax.set_title(f"Adaptive early-exit: accuracy vs speedup ({label})")
    ax.set_ylim(min(96.5, min(accuracy) - 1), 100.5)
    ax.legend(loc="lower left")
    savefig(out_name)


def plot_explain_latency():
    rows = read_csv("explain_latency.csv")
    sizes = sorted(set(int(r["n_stream"]) for r in rows))
    integrated = []
    posthoc = []
    for n in sizes:
        for r in rows:
            if int(r["n_stream"]) == n:
                per_tx_us = float(r["score_ms"]) / n * 1000.0
                if r["explain"] == "integrated":
                    integrated.append(per_tx_us)
                else:
                    posthoc.append(per_tx_us)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    import numpy as np
    xpos = np.arange(len(sizes))
    width = 0.35
    ax.bar(xpos - width / 2, integrated, width, color=BLUE, label="integrated (score_adaptive_explain)")
    ax.bar(xpos + width / 2, posthoc, width, color=ORANGE, label="naive post-hoc (score + separate full pass)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("transactions scored")
    ax.set_ylabel("per-transaction latency (µs)")
    ax.set_title("Explainability latency: integrated vs naive post-hoc")
    ax.legend(loc="upper left")
    savefig("explain_latency.png")


def plot_cuda_vs_cpu():
    import numpy as np
    rows = read_csv("cuda_vs_cpu.csv")
    sizes = sorted(set(int(r["n_stream"]) for r in rows))
    impls = [("serial", "serial (1 core)", MUTED), ("openmp_t12", "OpenMP (12 threads)", BLUE),
             ("cuda", "CUDA (GTX 1650)", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xpos = np.arange(len(sizes))
    width = 0.27
    for k, (impl, label, color) in enumerate(impls):
        tput = [float(next(r for r in rows if r["impl"] == impl and int(r["n_stream"]) == n)["throughput_tx_s"]) / 1e6
                for n in sizes]
        ax.bar(xpos + (k - 1) * width, tput, width, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.set_xlabel("transactions scored (full ensemble, 100 trees)")
    ax.set_ylabel("throughput (million tx/s, log scale)")
    ax.set_title("Scoring throughput: serial vs OpenMP vs CUDA")
    ax.legend(loc="upper left")
    savefig("cuda_vs_cpu.png")


def plot_cuda_early_exit():
    rows = read_csv("accuracy_vs_speedup_cuda.csv")
    z = [float(r["z"]) for r in rows]
    work_ratio = [100.0 / float(r["avg_trees_used"]) for r in rows]   # what a CPU core would save
    measured = [float(r["full_score_ms"]) / float(r["score_ms"]) for r in rows]
    warp_eff = [float(r["warp_efficiency"]) * 100 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(z, work_ratio, marker="o", color=MUTED, linewidth=2, label="tree evaluations saved (100 / avg trees)")
    ax.plot(z, measured, marker="o", color=ORANGE, linewidth=2, label="measured GPU speedup")
    ax.set_xlabel("z (confidence multiplier)")
    ax.set_ylabel("speedup vs full ensemble on GPU")
    ax.set_title("Early exit on SIMT hardware")
    ax.set_ylim(0, max(work_ratio) * 1.2)
    ax.legend(loc="upper right")

    ax2 = axes[1]
    ax2.plot(z, warp_eff, marker="o", color=BLUE, linewidth=2)
    ax2.set_xlabel("z (confidence multiplier)")
    ax2.set_ylabel("warp efficiency (%)")
    ax2.set_title("Useful / issued tree evaluations per warp")
    ax2.set_ylim(0, 105)
    savefig("cuda_early_exit.png")


def plot_mpi_dist_build():
    rows = read_csv("mpi_dist_build.csv")
    configs = []
    for r in rows:
        c = (int(r["n_trees"]), int(r["psi"]))
        if c not in configs:
            configs.append(c)
    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 4))
    for ax, (trees, psi) in zip(axes, configs):
        for dist, label, color in [("0", "replicated (every rank builds all)", MUTED),
                                   ("1", "distributed (--dist-build)", BLUE)]:
            sel = [r for r in rows if int(r["n_trees"]) == trees and int(r["psi"]) == psi and r["dist_build"] == dist]
            ax.plot([int(r["ranks"]) for r in sel], [float(r["build_ms"]) for r in sel],
                    marker="o", color=color, linewidth=2, label=label)
        ax.set_xlabel("ranks")
        ax.set_ylabel("build time incl. exchange (ms)")
        ax.set_title(f"n_trees={trees}, psi={psi}")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("MPI forest construction: replicated vs distributed", y=1.03)
    savefig("mpi_dist_build.png")


def plot_hp_sweep():
    import numpy as np
    rows = read_csv("hp_sweep.csv")
    psis = sorted(set(int(r["psi"]) for r in rows))
    trees = sorted(set(int(r["n_trees"]) for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, drop in zip(axes, ["0", "1"]):
        grid = np.zeros((len(trees), len(psis)))
        for i, t in enumerate(trees):
            for j, p in enumerate(psis):
                r = next(r for r in rows if r["drop_time"] == drop and int(r["psi"]) == p and int(r["n_trees"]) == t)
                grid[i, j] = float(r["val_ap"])
        im = ax.imshow(grid, cmap="Blues", origin="lower", aspect="auto")
        for i in range(len(trees)):
            for j in range(len(psis)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > grid.max() * 0.75 else INK)
        ax.set_xticks(range(len(psis)))
        ax.set_xticklabels(psis)
        ax.set_yticks(range(len(trees)))
        ax.set_yticklabels(trees)
        ax.set_xlabel("psi (subsample size)")
        ax.set_ylabel("n_trees")
        ax.set_title("all 30 features" if drop == "0" else "without Time")
        ax.grid(False)
    fig.suptitle("Validation average precision (threshold-free)", y=1.03)
    savefig("hp_sweep_ap.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    conts = sorted(set(float(r["contamination"]) for r in rows))
    sel = [next(r for r in rows if r["drop_time"] == "0" and int(r["psi"]) == 256 and int(r["n_trees"]) == 100
                and float(r["contamination"]) == c) for c in conts]
    best = max(rows, key=lambda r: float(r["val_f1"]))
    ax.plot([c * 100 for c in conts], [float(r["val_f1"]) for r in sel], marker="o", color=BLUE,
            linewidth=2, label="validation (first half of stream)")
    ax.plot([c * 100 for c in conts], [float(r["test_f1"]) for r in sel], marker="o", color=ORANGE,
            linewidth=2, label="test (second half of stream)")
    ax.scatter([float(best["contamination"]) * 100] * 2, [float(best["val_f1"]), float(best["test_f1"])],
               marker="*", s=160, color=[BLUE, ORANGE], zorder=5,
               label=f"best-on-validation config (psi={best['psi']}, {best['n_trees']} trees)")
    ax.set_xscale("log")
    ax.set_xlabel("contamination = % of transactions flagged (log scale)")
    ax.set_ylabel("F1")
    ax.set_title("Tuning on the early stream does not transfer")
    ax.set_ylim(0, 0.5)
    ax.legend(loc="upper left", fontsize=8.5)
    savefig("hp_sweep_contamination.png")


def plot_drift():
    rows = read_csv("drift.csv")
    rows = [r for r in rows if r["config"] == "default" and r["chunk_start"] != "all"]
    modes = [("static", "static model", MUTED), ("retrain_20k", "retrain every 20k", BLUE),
             ("retrain_10k", "retrain every 10k", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax, title in [("ap", axes[0], "average precision per slice (threshold-free)"),
                              ("recall", axes[1], "recall per slice")]:
        for mode, label, color in modes:
            sel = [r for r in rows if r["mode"] == mode and int(r["n_fraud"]) > 0]
            ax.plot([int(r["chunk_start"]) / 1000 for r in sel], [float(r[metric]) for r in sel],
                    marker="o", color=color, linewidth=2, markersize=5, label=label)
        ax.set_xlabel("slice start (thousand tx into stream)")
        ax.set_ylabel(metric if metric != "ap" else "average precision")
        ax.set_title(title)
        ax.set_ylim(0, 1.0 if metric == "recall" else 0.6)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Detection quality over the stream (default config: psi=256, 100 trees, 1% contamination)", y=1.03)
    savefig("drift.png")


if __name__ == "__main__":
    plot_strong_scaling("strong_scaling_openmp.csv", "strong_scaling_openmp.png", "OpenMP")
    plot_strong_scaling("strong_scaling_mpi.csv", "strong_scaling_mpi.png", "MPI")
    plot_weak_scaling("weak_scaling_openmp.csv", "weak_scaling_openmp.png", "OpenMP")
    plot_weak_scaling("weak_scaling_mpi.csv", "weak_scaling_mpi.png", "MPI")
    plot_accuracy_vs_speedup("accuracy_vs_speedup_openmp.csv", "accuracy_vs_speedup_openmp.png", "OpenMP, 8 threads")
    plot_accuracy_vs_speedup("accuracy_vs_speedup_mpi.csv", "accuracy_vs_speedup_mpi.png", "MPI, 8 ranks")
    plot_explain_latency()
    plot_cuda_vs_cpu()
    plot_accuracy_vs_speedup("accuracy_vs_speedup_cuda.csv", "accuracy_vs_speedup_cuda.png", "CUDA, GTX 1650")
    plot_cuda_early_exit()
    plot_mpi_dist_build()
    plot_hp_sweep()
    plot_drift()
