#!/usr/bin/env python3
"""
Runs every benchmark experiment for the AdaptGuard project and writes raw
CSV results under docs/06_results/data/. This is the single source of truth
for every number that ends up in a plot or in results_summary.md -- nothing
here is hand-typed after the fact.

Run from the project root INSIDE WSL (needs the compiled build/serial,
build/openmp, build/mpi, build/cuda binaries and data/creditcard.csv on a native Linux
filesystem -- see docs/01_dataset.md for why 9p/DrvFs paths are avoided):

    python3 scripts/run_benchmarks.py
"""
import csv
import os
import re
import subprocess
import sys
import time

ROOT = "/root/adaptguard"
DATA = os.path.join(ROOT, "data/creditcard.csv")
OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

RESULT_RE = re.compile(r"(\w+)=(\S+)")


def parse_result_line(stdout: str):
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            d = dict(RESULT_RE.findall(line[len("RESULT "):]))
            return d
    raise RuntimeError(f"No RESULT line found in output:\n{stdout}")


def run(cmd, env=None):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return parse_result_line(p.stdout)


def openmp_cmd(threads, n_stream, trees=100, psi=256, extra=None):
    cmd = [f"{ROOT}/build/openmp", "--data", DATA, "--trees", str(trees), "--psi", str(psi),
           "--seed", "42", "--score-n", str(n_stream), "--contamination", "0.01",
           "--threads", str(threads), "--repeat", "3"]
    if extra:
        cmd += extra
    return cmd


def mpi_env():
    env = os.environ.copy()
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return env


def mpi_cmd(nprocs, n_stream, trees=100, psi=256, extra=None):
    cmd = ["mpirun", "-np", str(nprocs), "--oversubscribe", f"{ROOT}/build/mpi",
           "--data", DATA, "--trees", str(trees), "--psi", str(psi),
           "--seed", "42", "--score-n", str(n_stream), "--contamination", "0.01",
           "--repeat", "3"]
    if extra:
        cmd += extra
    return cmd


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


PROC_COUNTS = [1, 2, 3, 4, 6, 8, 12]


def strong_scaling_openmp():
    n_stream = 100000
    rows = []
    for t in PROC_COUNTS:
        r = run(openmp_cmd(t, n_stream))
        rows.append(r)
        print("openmp strong", t, r["score_ms"], r["throughput_tx_s"])
    write_csv(os.path.join(OUT_DIR, "strong_scaling_openmp.csv"), rows, list(rows[0].keys()))


def strong_scaling_mpi():
    n_stream = 100000
    rows = []
    for p in PROC_COUNTS:
        r = run(mpi_cmd(p, n_stream), env=mpi_env())
        rows.append(r)
        print("mpi strong", p, r["score_ms"], r["throughput_tx_s"], r.get("load_imbalance_ms"))
    write_csv(os.path.join(OUT_DIR, "strong_scaling_mpi.csv"), rows, list(rows[0].keys()))


def weak_scaling_openmp():
    base_per_thread = 8000
    rows = []
    for t in PROC_COUNTS:
        n_stream = base_per_thread * t
        r = run(openmp_cmd(t, n_stream))
        rows.append(r)
        print("openmp weak", t, n_stream, r["score_ms"])
    write_csv(os.path.join(OUT_DIR, "weak_scaling_openmp.csv"), rows, list(rows[0].keys()))


def weak_scaling_mpi():
    base_per_rank = 8000
    rows = []
    for p in PROC_COUNTS:
        n_stream = base_per_rank * p
        r = run(mpi_cmd(p, n_stream), env=mpi_env())
        rows.append(r)
        print("mpi weak", p, n_stream, r["score_ms"])
    write_csv(os.path.join(OUT_DIR, "weak_scaling_mpi.csv"), rows, list(rows[0].keys()))


Z_SWEEP = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def accuracy_vs_speedup_openmp():
    n_stream = 50000
    threads = 8
    rows = []
    for z in Z_SWEEP:
        extra = ["--adaptive", "--compare-full", "--z", str(z), "--min-trees", "10"]
        r = run(openmp_cmd(threads, n_stream, extra=extra))
        rows.append(r)
        print("acc-speedup openmp", z, r["agreement_full"], r["score_ms"], r["full_score_ms"])
    write_csv(os.path.join(OUT_DIR, "accuracy_vs_speedup_openmp.csv"), rows, list(rows[0].keys()))


def accuracy_vs_speedup_mpi():
    n_stream = 50000
    nprocs = 8
    rows = []
    for z in Z_SWEEP:
        extra = ["--adaptive", "--compare-full", "--z", str(z), "--min-trees", "10"]
        r = run(mpi_cmd(nprocs, n_stream, extra=extra), env=mpi_env())
        rows.append(r)
        print("acc-speedup mpi", z, r["agreement_full"], r["score_ms"], r["full_score_ms"])
    write_csv(os.path.join(OUT_DIR, "accuracy_vs_speedup_mpi.csv"), rows, list(rows[0].keys()))


def explain_latency():
    sizes = [500, 1000, 2000, 5000, 10000]
    rows = []
    for n in sizes:
        for mode in ["integrated", "posthoc"]:
            cmd = [f"{ROOT}/build/serial", "--data", DATA, "--trees", "100", "--psi", "256",
                   "--seed", "42", "--score-n", str(n), "--contamination", "0.01",
                   "--adaptive", "--z", "2.0", "--min-trees", "10", "--explain", mode]
            r = run(cmd)
            rows.append(r)
            print("explain", n, mode, r["score_ms"])
    write_csv(os.path.join(OUT_DIR, "explain_latency.csv"), rows, list(rows[0].keys()))


# ---------------------------------------------------------------------------
# Completion experiments (docs/09_cuda.md, docs/10_drift_and_tuning.md).
# Run individually by name, e.g.  python3 scripts/run_benchmarks.py cuda_vs_cpu
# ---------------------------------------------------------------------------

def cuda_cmd(n_stream, trees=100, psi=256, extra=None):
    cmd = [f"{ROOT}/build/cuda", "--data", DATA, "--trees", str(trees), "--psi", str(psi),
           "--seed", "42", "--score-n", str(n_stream), "--contamination", "0.01", "--repeat", "3"]
    if extra:
        cmd += extra
    return cmd


def serial_cmd(n_stream, extra=None):
    cmd = [f"{ROOT}/build/serial", "--data", DATA, "--trees", "100", "--psi", "256",
           "--seed", "42", "--score-n", str(n_stream), "--contamination", "0.01", "--repeat", "3"]
    if extra:
        cmd += extra
    return cmd


def cuda_vs_cpu():
    """Full-ensemble scoring-phase time: serial vs OpenMP (12 threads) vs CUDA."""
    rows = []
    for n in [10000, 50000, 100000, -1]:
        for name, cmd in [("serial", serial_cmd(n)), ("openmp_t12", openmp_cmd(12, n)), ("cuda", cuda_cmd(n))]:
            r = run(cmd)
            rows.append({"impl": name, "n_stream": r["n_stream"], "score_ms": r["score_ms"],
                         "throughput_tx_s": r["throughput_tx_s"], "h2d_ms": r.get("h2d_ms", ""),
                         "d2h_ms": r.get("d2h_ms", ""), "f1": r["f1"]})
            print("cuda_vs_cpu", name, r["n_stream"], r["score_ms"])
    write_csv(os.path.join(OUT_DIR, "cuda_vs_cpu.csv"), rows, list(rows[0].keys()))


def accuracy_vs_speedup_cuda():
    n_stream = 50000
    rows = []
    for z in Z_SWEEP:
        extra = ["--adaptive", "--compare-full", "--z", str(z), "--min-trees", "10"]
        r = run(cuda_cmd(n_stream, extra=extra))
        rows.append(r)
        print("acc-speedup cuda", z, r["agreement_full"], r["score_ms"], r["full_score_ms"], r["warp_efficiency"])
    write_csv(os.path.join(OUT_DIR, "accuracy_vs_speedup_cuda.csv"), rows, list(rows[0].keys()))


def cuda_block_size():
    rows = []
    for b in [32, 64, 128, 256, 512, 1024]:
        for adaptive in [False, True]:
            extra = ["--block", str(b)] + (["--adaptive", "--z", "2.0", "--min-trees", "10"] if adaptive else [])
            r = run(cuda_cmd(-1, extra=extra))
            rows.append(r)
            print("cuda block", b, adaptive, r["score_ms"])
    write_csv(os.path.join(OUT_DIR, "cuda_block_size.csv"), rows, list(rows[0].keys()))


def mpi_dist_build():
    """Replicated vs distributed (--dist-build) forest construction."""
    rows = []
    for trees, psi in [(100, 256), (1000, 256), (1000, 2048)]:
        for p in [1, 2, 4, 8]:
            for dist in [False, True]:
                extra = ["--dist-build"] if dist else []
                cmd = mpi_cmd(p, 1000, trees=trees, psi=psi, extra=extra)
                r = run(cmd, env=mpi_env())
                rows.append(r)
                print("mpi build", trees, psi, p, dist, r["build_ms"], r["build_comm_ms"])
    write_csv(os.path.join(OUT_DIR, "mpi_dist_build.csv"), rows, list(rows[0].keys()))


# --- detection-quality helpers (labels used for EVALUATION only) ---

def read_out(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        return [(int(r["true_label"]), int(r["pred_label"]), float(r["score"])) for r in rd]


def prf(rows):
    tp = sum(1 for y, p, _ in rows if y == 1 and p == 1)
    fp = sum(1 for y, p, _ in rows if y == 0 and p == 1)
    fn = sum(1 for y, p, _ in rows if y == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def average_precision(rows):
    """Threshold-free ranking quality (area under the precision-recall curve)."""
    ranked = sorted(rows, key=lambda r: -r[2])
    n_pos = sum(r[0] for r in rows)
    if n_pos == 0:
        return 0.0
    tp, ap = 0, 0.0
    for k, (y, _, _) in enumerate(ranked, 1):
        if y == 1:
            tp += 1
            ap += tp / k
    return ap / n_pos


def hp_sweep():
    """Grid over psi x n_trees x contamination x drop_time. The stream is split
    in half chronologically: the first half is the validation set used to pick
    a configuration, the second half is the untouched test set it is reported on."""
    rows = []
    tmp = "/tmp/hp_out.csv"
    for drop_time in [False, True]:
        for psi in [64, 128, 256, 512, 1024]:
            for trees in [50, 100, 200, 400]:
                for cont in [0.001, 0.002, 0.005, 0.01]:
                    cmd = [f"{ROOT}/build/openmp", "--data", DATA, "--trees", str(trees), "--psi", str(psi),
                           "--seed", "42", "--contamination", str(cont), "--threads", "12", "--out", tmp]
                    if drop_time:
                        cmd.append("--drop-time")
                    r = run(cmd)
                    out = read_out(tmp)
                    half = len(out) // 2
                    val, test = out[:half], out[half:]
                    vp, vr, vf = prf(val)
                    tp_, tr, tf = prf(test)
                    rows.append({"drop_time": int(drop_time), "psi": psi, "n_trees": trees, "contamination": cont,
                                 "theta": r["theta"], "val_precision": round(vp, 4), "val_recall": round(vr, 4),
                                 "val_f1": round(vf, 4), "val_ap": round(average_precision(val), 4),
                                 "test_precision": round(tp_, 4), "test_recall": round(tr, 4),
                                 "test_f1": round(tf, 4), "test_ap": round(average_precision(test), 4),
                                 "full_f1": r["f1"], "score_ms": r["score_ms"]})
                    print("hp", drop_time, psi, trees, cont, round(vf, 4), round(tf, 4))
    write_csv(os.path.join(OUT_DIR, "hp_sweep.csv"), rows, list(rows[0].keys()))


DRIFT_CHUNK = 20000


def drift():
    """Static model vs rolling retraining, detection quality per 20k-tx slice of the stream,
    for the default config and for the configuration hp_sweep picked on validation data."""
    best = None
    hp_path = os.path.join(OUT_DIR, "hp_sweep.csv")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            best = max(csv.DictReader(f), key=lambda r: float(r["val_f1"]))
    configs = [("default", ["--trees", "100", "--psi", "256", "--contamination", "0.01"])]
    if best:
        tuned = ["--trees", best["n_trees"], "--psi", best["psi"], "--contamination", best["contamination"]]
        if best["drop_time"] == "1":
            tuned.append("--drop-time")
        configs.append(("tuned", tuned))
    rows = []
    tmp = "/tmp/drift_out.csv"
    for cfg_name, cfg in configs:
        for retrain in [0, 20000, 10000]:
            cmd = [f"{ROOT}/build/openmp", "--data", DATA, "--seed", "42", "--threads", "12",
                   "--retrain-every", str(retrain), "--window", "50000", "--out", tmp] + cfg
            r = run(cmd)
            out = read_out(tmp)
            mode = "static" if retrain == 0 else f"retrain_{retrain // 1000}k"
            for c0 in range(0, len(out), DRIFT_CHUNK):
                chunk = out[c0:c0 + DRIFT_CHUNK]
                p, rc, f1 = prf(chunk)
                rows.append({"config": cfg_name, "mode": mode, "chunk_start": c0, "n": len(chunk),
                             "n_fraud": sum(y for y, _, _ in chunk),
                             "flag_rate": round(sum(p_ for _, p_, _ in chunk) / len(chunk), 5),
                             "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4),
                             "ap": round(average_precision(chunk), 4)})
            p, rc, f1 = prf(out)
            rows.append({"config": cfg_name, "mode": mode, "chunk_start": "all", "n": len(out),
                         "n_fraud": sum(y for y, _, _ in out),
                         "flag_rate": round(sum(p_ for _, p_, _ in out) / len(out), 5),
                         "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4), "ap": ""})
            print("drift", cfg_name, mode, round(p, 4), round(rc, 4), round(f1, 4), "retrain_ms", r["retrain_ms"])
    write_csv(os.path.join(OUT_DIR, "drift.csv"), rows, list(rows[0].keys()))


def main():
    experiments = {
        "strong_openmp": strong_scaling_openmp,
        "strong_mpi": strong_scaling_mpi,
        "weak_openmp": weak_scaling_openmp,
        "weak_mpi": weak_scaling_mpi,
        "acc_speedup_openmp": accuracy_vs_speedup_openmp,
        "acc_speedup_mpi": accuracy_vs_speedup_mpi,
        "explain_latency": explain_latency,
        "cuda_vs_cpu": cuda_vs_cpu,
        "acc_speedup_cuda": accuracy_vs_speedup_cuda,
        "cuda_block_size": cuda_block_size,
        "mpi_dist_build": mpi_dist_build,
        "hp_sweep": hp_sweep,  # must run before drift, which reads its output
        "drift": drift,
    }
    only = sys.argv[1:] if len(sys.argv) > 1 else list(experiments.keys())
    for name in only:
        print(f"=== {name} ===")
        t0 = time.time()
        experiments[name]()
        print(f"=== {name} done in {time.time()-t0:.1f}s ===\n")


if __name__ == "__main__":
    main()
