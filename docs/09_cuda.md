# CUDA Implementation

`src/cuda/main.cu`, built with `nvcc` 12.0 (`make cuda`, flags
`-O3 -arch=sm_75 --fmad=false`). GPU: NVIDIA GeForce GTX 1650 (Turing,
compute capability 7.5, 4 GB, 14 SMs), used from WSL2.

This replaces the earlier "CUDA out of scope" decision: the dev machine
turned out to have a usable GPU and a working CUDA toolchain inside WSL2.

## Design

1. **Forest built on the CPU, scored on the GPU.** Building 100 trees takes
   ~5-8 ms, so it isn't worth porting. The forest is built with the same
   deterministic code as the other three binaries, so the GPU scores the
   *same* forest.
2. **Flattened forest.** All trees are concatenated into one `GNode` array
   (`feature`, `split_value`, absolute `left`/`right` indices, 16 bytes per
   node) plus a per-tree root offset. Leaf corrections `c(size)` are
   precomputed on the host into a parallel `double` array. The whole
   forest is at most ~1.2 MB (100 trees × ≤511 nodes × 24 bytes),
   comparable to the GPU's 1 MB L2 cache, so most node reads hit cache.
3. **One GPU thread per transaction.** Each thread walks all trees for
   its transaction. There are no atomics or shared-memory reductions:
   each thread writes its own output slot.
4. **Column-major (feature-major) stream layout.** Thread `i` reads feature
   `f` at `X[f * n + i]`. Near the root of every tree, neighbouring threads
   test the same feature, so their loads coalesce into one memory
   transaction.
5. **Score conversion on the host.** The kernel returns the mean path
   length, and the host applies the same `score_to_anomaly()` as the CPU
   code.

## Bit-exact correctness on a GPU

GPU results usually only "match approximately". Here they match
**exactly**, for three reasons:

- The device sums `depth + c(size)` in `double`, in the same tree order as
  the CPU, with `c(size)` precomputed by the same host function.
- `--fmad=false` stops `nvcc` from fusing multiply-add into FMA
  instructions. FMA rounds once instead of twice, which would change the
  last bits of Welford's running mean/variance in adaptive mode.
- The final `pow()` (path length → score) runs on the host, not the
  device, because CUDA's `pow` may differ from glibc's by an ulp.

`scripts/validate_cuda.sh` diffs CUDA against serial for full-ensemble
**and** adaptive mode, at 20,000 and all 142,404 stream rows, with block
sizes 64/128/256. **Result: 0 diff lines in all 12 comparisons.**

## Results: full-ensemble scoring (`cuda_vs_cpu.csv`, `cuda_vs_cpu.png`)

Kernel time only (`cudaEvent` timing, the same "scoring phase only"
definition used for OpenMP/MPI), averaged over 3 repeats:

| Transactions | Serial | OpenMP (12 thr) | CUDA kernel | CUDA vs serial | CUDA vs OpenMP |
|---|---|---|---|---|---|
| 10,000 | 95.1 ms | 20.3 ms | 0.51 ms | 186x | 40x |
| 50,000 | 458.2 ms | 60.3 ms | 2.36 ms | 194x | 26x |
| 100,000 | 879.7 ms | 94.2 ms | 4.29 ms | 205x | 22x |
| 142,404 | 1177.5 ms | 150.8 ms | 6.82 ms | 173x | 22x |

**Including data transfer** (host→device 6.65 ms for the 17 MB stream +
forest, device→host 0.52 ms), the full stream takes 14.0 ms end to end:
**84x faster than serial and 10.8x faster than 12-thread OpenMP.** For
a real stream the host→device copy would overlap with compute (CUDA
streams, double buffering). That isn't implemented here, so the
transfer cost is reported separately rather than hidden.

CPU timings vary between sessions on this laptop (an earlier run in the
same day measured serial at 763 ms and OpenMP at 118 ms for the full
stream, giving 121x / 19x). The ratios above are from the run whose
CSV is checked in.

Block size barely matters for full-ensemble scoring (`cuda_block_size.csv`):
6.7-6.9 ms for 64-1024 threads per block, 8.8 ms at 32 (too few warps
per SM to hide memory latency).

## Results: adaptive early exit on SIMT hardware

`accuracy_vs_speedup_cuda.csv`, `cuda_early_exit.png`. 50,000
transactions, `min_trees=10`. Speedup = adaptive kernel vs full kernel on
the same GPU.

| z | Agreement | Avg trees | Trees saved | Warp efficiency | Measured GPU speedup |
|---|---|---|---|---|---|
| 0.25 | 99.43% | 10.0 | 10.0x | 98% | **5.3x** |
| 0.5 | 99.44% | 10.0 | 10.0x | 92% | 4.7x |
| 1.0 | 99.65% | 10.3 | 9.7x | 57% | 2.1x |
| 1.5 | 99.86% | 10.8 | 9.3x | 35% | 1.2x |
| 2.0 | 99.95% | 11.4 | 8.7x | 27% | 0.85x (slower) |
| 3.0 | 100% | 13.2 | 7.6x | 21% | 0.58x (slower) |

Agreement is identical to the OpenMP and MPI sweeps (same algorithm,
same decisions). The speedup is very different. Two effects explain it:

1. **Warp divergence.** The 32 threads of a warp execute in lockstep, so
   a warp keeps running until its *slowest* transaction stops. Warp
   efficiency = useful tree evaluations ÷ issued evaluations
   (`sum(trees_used) / sum over warps of 32 × max(trees_used)`). At z=2
   only 27% of issued work is useful: most warps contain at least one
   borderline transaction that needs many more trees. On a CPU each
   thread stops independently, so all of the 8.7x saving is realised.
2. **FP64 cost of the stopping check.** The GTX 1650 runs double
   precision at 1/32 of single-precision speed. The full kernel does one
   FP64 add per tree. The adaptive kernel also does Welford updates,
   divisions and a `sqrt` per tree. The first version also called `pow()`
   twice per tree, which made adaptive scoring **4-5x slower than the full
   ensemble** at z=2 (12.6 ms vs 2.2 ms, kept in
   `accuracy_vs_speedup_cuda_pow.csv`). Rewriting the check in path-length
   space (`mean + z·SE < h_θ` with `h_θ = -c(ψ)·log2 θ` precomputed once)
   removed both `pow` calls and made the kernel 4.4x faster at z=2,
   with the same decisions (still 0 diff lines vs serial).

**Takeaway**: the adaptive idea carries over to GPUs, but the
accuracy/speed trade-off moves. On this GPU it pays off for z ≤ 1.5
(1.2-5.3x faster at ≥ 99.4% agreement). At the CPU default z=2, the
divergence and FP64 overhead outweigh the saved trees. Possible
improvements (not implemented): sort or bin transactions by expected
difficulty so warps agree, check the stopping rule every few trees
instead of every tree, or do the bookkeeping in FP32. The last two would
give up the bit-exact match with the CPU reference.

## Not ported to CUDA

- **Explainability.** Integrated contributions would need a 30-wide
  per-thread accumulator (register pressure) or shared memory. The
  latency comparison it supports is a per-transaction algorithmic
  question and is answered on the CPU (`docs/05_explainability.md`).
- **Rolling retraining.** Built on the CPU in the OpenMP binary
  (`docs/10_drift_and_tuning.md`).
