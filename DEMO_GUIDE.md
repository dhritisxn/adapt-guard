# AdaptGuard — Live Demo Guide

A step-by-step script for demonstrating the project live: what to type,
what to point at, and what to say. Every "expected output" below comes
from an actual dry run on this laptop. For the theory and likely
questions, see `PRESENTATION_NOTES.md`.

**Full demo: ~10 minutes. Short version: ~3 minutes (Section 5).**

---

## 1. Before the demo (do this 15 minutes early)

- [ ] **Plug in the charger.** On battery the GPU caps its memory clock
      (measured: 810 of 6001 MHz), and the CUDA full-ensemble kernel gets
      ~3.7x slower (25 ms instead of ~7 ms). CPU numbers also drop.
- [ ] Windows power mode → **Best performance** (Settings → System → Power).
- [ ] Close GPU-heavy apps (games, Overwolf, browser tabs with video).
- [ ] Open PowerShell and start Ubuntu **as root** (the project lives in
      `/root`, and this avoids a sudo password prompt in front of the room):
      ```
      wsl -d Ubuntu-24.04 -u root
      ```
      Use `-d Ubuntu-24.04` explicitly. Plain `wsl` opens the
      `docker-desktop` distro, which is the default here and has no bash.
- [ ] Make the terminal font large (Ctrl + `+`) so the room can read it.
- [ ] Run the setup block below once, so that nothing compiles live.

### Setup block (inside Ubuntu)

```bash
cd /root/adaptguard
cp -r /mnt/c/Users/ayush/OneDrive/Desktop/adapt-guard/{src,tests,scripts,Makefile} .
sed -i 's/\r$//' scripts/*.sh scripts/*.py
make all cuda test                       # should end with: 3352 checks, 0 failures
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
./build/cuda --data data/creditcard.csv --repeat 20 > /dev/null   # wake the GPU up
clear
```

*If `/root/adaptguard` doesn't exist* (new machine): `git clone
https://github.com/dhritisxn/adapt-guard.git /root/adaptguard`, put
`creditcard.csv` from Kaggle into `/root/adaptguard/data/`, then run the
block above (skip the `cp` line).

### Have these open in another window (backup + visuals)

`docs/06_results/plots/` in File Explorer, with:
`architecture_diagram.png`, `accuracy_vs_speedup_openmp.png`,
`cuda_vs_cpu.png`, `cuda_early_exit.png`, `drift.png`.

---

## 2. How to read the output

Every program prints one line starting with `RESULT`, made of
`key=value` pairs. Only point at these fields:

| Field | Meaning |
|---|---|
| `score_ms` | Time to score the transactions (the part we parallelised) |
| `throughput_tx_s` | Transactions per second |
| `avg_trees_used` | Trees evaluated per transaction (100 = full model) |
| `agreement_full` | Fraction of decisions identical to the full 100-tree model |
| `full_score_ms` | Time the full model took on the same hardware (for comparison) |
| `precision`, `recall`, `f1` | Detection quality vs true fraud labels |
| `warp_efficiency` | CUDA only: useful ÷ issued GPU work |

**Tip:** to show only the important fields, pipe through `grep`, e.g.
`... | grep -oE 'score_ms=[^ ]+|avg_trees_used=[^ ]+|agreement_full=[^ ]+'`.
In practice it's often simpler to show the full line and point.

---

## 3. The demo, step by step

### Step 1 — The baseline (serial) · ~1 min

```bash
./build/serial --data data/creditcard.csv --score-n 20000
```

**Expected:** `score_ms≈160`, `throughput_tx_s≈123000`,
`precision=0.4 recall=0.71 f1=0.51`.

**Say:** "This is the plain single-core version and our correctness
reference. It builds 100 isolation trees in about 8 ms, then scores
20,000 streamed transactions. It catches 71% of the fraud, with no labels
used in training: Isolation Forest is unsupervised."

### Step 2 — OpenMP on all 12 cores · ~1 min

```bash
./build/openmp --data data/creditcard.csv --threads 12
```

**Expected:** whole stream (`n_stream=142404`), `score_ms≈115`,
`throughput_tx_s≈1.2 million`, `avg_trees_used=100`.

**Say:** "Same algorithm, now on all 12 CPU threads with OpenMP, over the
whole stream of 142,000 transactions: about 1.2 million transactions per
second."

### Step 3 — The novelty: adaptive early exit · ~1.5 min

```bash
./build/openmp --data data/creditcard.csv --threads 12 --adaptive --compare-full
```

**Expected:** `score_ms≈20` vs `full_score_ms≈120` (**~6x faster**),
`avg_trees_used≈11.4`, `agreement_full≈0.9994`.

**Say:** "With early exit on, each transaction stops evaluating trees as
soon as a confidence interval says the decision is settled. On average it
uses 11 trees instead of 100, and it's 6 times faster, with 99.94% of
decisions identical to the full model. That's our main contribution."
*(Optional: show `accuracy_vs_speedup_openmp.png`.)*

### Step 4 — MPI, 8 processes · ~1 min

```bash
mpirun -np 8 --oversubscribe ./build/mpi --data data/creditcard.csv --adaptive --compare-full 2>/dev/null
```

**Expected:** `ranks=8`, `score_ms≈19`, `comm_ms≈0.6`,
`load_imbalance_ms≈6`, `agreement_full≈0.9994` (same as OpenMP).

**Say:** "MPI splits the stream across 8 processes, as if on 8 machines,
and gathers the results with `MPI_Gatherv`. Communication takes under a
millisecond. Agreement is exactly the same as OpenMP, because every
version makes identical decisions."

(Leave out `2>/dev/null` if you want to show each rank's `RANKINFO`
line: its shard and timing.)

### Step 5 — CUDA on the GPU · ~1.5 min

```bash
./build/cuda --data data/creditcard.csv --repeat 10
```

**Expected (charger plugged in):** `score_ms≈7`,
`throughput_tx_s≈20 million`. *(On battery: ~25 ms, ~5.5 million.
Still the fastest, but say "it's on battery".)*

**Say:** "The same 142,000 transactions on the laptop's GTX 1650, with one
GPU thread per transaction: about 7 milliseconds. That's roughly 170
times faster than serial and 20 times faster than all 12 CPU cores. The
output is bit-for-bit identical to the serial version."

Then the GPU twist:

```bash
./build/cuda --data data/creditcard.csv --repeat 10 --adaptive --z 0.5 --compare-full
./build/cuda --data data/creditcard.csv --repeat 10 --adaptive --z 2 --compare-full
```

**Expected:** at `z=0.5`, `score_ms≈3.3`, `warp_efficiency≈0.91`,
clearly faster than `full_score_ms`. At `z=2`, `score_ms≈8.3`,
`warp_efficiency≈0.27`: plugged in, that's about the same as or slower
than the full kernel.

**Say:** "On a GPU, threads run in groups of 32 that move together, so a
group finishes only when its slowest transaction does. With a cautious
setting (z=2) only 27% of the GPU's work is useful, and the saving
disappears. With z=0.5 it's 91% useful and early exit pays off. Same
idea, different trade-off on different hardware."
*(Show `cuda_early_exit.png`.)*

### Step 6 — Explainability · ~1.5 min

```bash
./build/serial --data data/creditcard.csv --score-n 2000 --adaptive --explain integrated --out /tmp/demo.csv
head -5 /tmp/demo.csv.explain.csv
./build/serial --data data/creditcard.csv --score-n 2000 --adaptive --explain posthoc
```

**Expected:** integrated `score_ms≈2.2`, post-hoc `score_ms≈15.8`
(**~7x slower**). The explain file shows:
```
row_id,true_label,pred_label,top1_feat,top1_contrib,top2_feat,...
142405,1,1,V5,0.2625,V7,0.233333,V25,0.183333
142514,0,1,V10,0.217674,V9,0.169551,V4,0.153846
```

**Say:** "Row 142405 is real fraud, and we caught it. The top reasons are
features V5, V7 and V25. We collect these during the same tree walk that
computes the score, so the explanation is nearly free. Computing it in a
separate pass afterwards, the usual way, is 7 times slower."
(Row 142514 is a false alarm (true label 0). It's fine to point that out
honestly.)

### Step 7 — Correctness proof · ~1 min

```bash
bash scripts/validate_openmp.sh | grep diff
make test
```

**Expected:** four lines `threads=1/2/4/8 diff_lines=0` (~10 s), then
`3352 checks, 0 failures`.

**Say:** "Parallel code is easy to get subtly wrong, so we don't eyeball
it. Every version's output is diffed line by line against the serial
version: zero differences, at every thread count. The same scripts exist
for MPI and CUDA. That's possible because every tree has a fixed seed,
so all versions build the identical forest."

*(CUDA version, ~1 min longer: `bash scripts/validate_cuda.sh`, all 0.)*

### Optional extras (if asked or if time allows)

**Distributed MPI tree building** (~10 s):
```bash
mpirun -np 8 --oversubscribe ./build/mpi --data data/creditcard.csv --trees 1000 --psi 2048 --score-n 1000 2>/dev/null
mpirun -np 8 --oversubscribe ./build/mpi --data data/creditcard.csv --trees 1000 --psi 2048 --score-n 1000 --dist-build 2>/dev/null
```
Expected `build_ms`: ≈1000 → ≈160 (~6x). "For a big forest, each process
builds an eighth of the trees and they swap them with `MPI_Allgatherv`."

**Streaming replay** (cosmetic, shows "live" arrival):
```bash
python3 scripts/replay_demo.py --rate 20 --n 20
```

**Rolling retraining** (~3 s):
```bash
./build/openmp --data data/creditcard.csv --threads 12 --retrain-every 20000
```
Point at `n_retrains=7`, `retrain_ms`, `recall` (≈0.52 vs 0.39 static).
Then show `drift.png` and explain why F1 doesn't improve
(`PRESENTATION_NOTES.md` §10).

---

## 4. Wrap-up line

"So: one fraud detector, four parallel implementations that agree bit
for bit, up to 170x faster on a GPU, about 6x less work from early exit
at 99.9% agreement, and free explanations for every alert."

---

## 5. Short version (~3 minutes)

If time is tight, run only these three and show plots for the rest:

```bash
./build/openmp --data data/creditcard.csv --threads 12 --adaptive --compare-full
./build/cuda --data data/creditcard.csv --repeat 10
head -5 /tmp/demo.csv.explain.csv     # run Step 6's first command during setup
```

---

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| `wsl` opens a shell without bash / "execvpe(bash) failed" | Use `wsl -d Ubuntu-24.04 -u root` |
| `Permission denied` on `/root/adaptguard` | You started WSL as `ayush`. Exit and use `-u root` |
| `mpirun` refuses to run as root | `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` |
| `mpirun`: not enough slots | Keep `--oversubscribe` in the command |
| CUDA numbers ~4x slower than expected | Charger unplugged or power-saving mode. Plug in, and say "on battery the GPU memory is throttled" |
| `CUDA error ... no CUDA-capable device` | Restart WSL from PowerShell (`wsl --shutdown`, then reopen) |
| `Could not open dataset file` | You're not in `/root/adaptguard`, or `data/creditcard.csv` is missing |
| `bash\r: No such file` running a script | Run `sed -i 's/\r$//' scripts/*.sh` |
| Anything else, live | Switch to the plots in `docs/06_results/plots/`. Every number there is from a real run |

Timings vary by a few percent between runs. Speak in round numbers
("about 6x", "around 7 milliseconds"), not exact decimals.
