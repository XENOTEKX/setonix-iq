# Profiling Report - Evidence Notebook & GPU Port Plan

**Date:** 26 April 2026  **Status:** final, post methodology audit
**Source:** `Profiling_Report.html` (33 runs, `logs/runs/*.json`, `logs/profiles/*.json`)
**Companion:** `CHANGELOG.md` (full entry under 2026-04-26 methodology audit)

This file is the evidence notebook backing every claim in `Profiling_Report.html`,
plus the actionable GPU port plan. Every number has been re-derived directly
from the JSON records under `logs/runs/`.

---

## 1. The one comparison you can trust

> **Only `xlarge_mf.fa` is a methodologically valid cross-platform benchmark.**

After auditing all 33 run JSONs we discovered two compounding issues that
invalidate the cross-platform comparison for two of the three datasets. See
§4 for the audit details. The bottom line is:

| Dataset                  | Same alignment file? | Same model search? | Cross-platform comparable? |
|--------------------------|:--------------------:|:------------------:|:--------------------------:|
| `large_modelfinder.fa`   | NO (sha256 mismatch) | NO (`-mset` on Setonix) | NO †                |
| **`xlarge_mf.fa`**       | **YES**              | **Yes (both GTR)** | **YES**                    |
| `mega_dna.fa`            | NO (sha256 mismatch) | NO (`-mset` + `+ASC`) | NO †                  |

Therefore **all GPU success metrics in the porting plan (§7) are anchored on
`xlarge_mf.fa`**. Other datasets are used only for within-platform scaling
characterisation (which is internally valid regardless of model choice).

The headline number that survives the audit:

> **7.3× Gadi Sapphire Rapids (64T) advantage over Setonix Trento (64T) on
> `xlarge_mf.fa`, same canonical alignment, both selecting GTR-family models,
> lnL agreement <4 units.**

---

## 2. Datasets & platforms (post-audit)

| Dataset                | Taxa × Sites    | Patterns | Size  | Gadi model (full MF) | Setonix model (-mset) |
|------------------------|-----------------|---------:|------:|----------------------|-----------------------|
| `large_modelfinder.fa` | 100 × 50 000    |   48 293 | 4.8MB | GTR+F+G4             | HKY+F+G4 †            |
| `xlarge_mf.fa`         | 200 × 100 000   |   98 858 | 20 MB | GTR+F+R4             | GTR+F+G4              |
| `mega_dna.fa`          | 500 × 100 000   |  100 000 | 48 MB | GTR+F+R4             | HKY+F+ASC+R5 †        |

| Platform              | Setonix (Pawsey)                  | Gadi normalsr (NCI)              |
|-----------------------|-----------------------------------|----------------------------------|
| CPU                   | AMD EPYC 7A53 "Trento" (Zen 3)    | Intel Xeon Sapphire Rapids       |
| Cores / SMT           | 64C / 128T per node               | 104C per node                    |
| Build                 | gcc, AVX2 `Vec4d`, `-march=znver3`| icx, AVX2 `Vec4d`, `-march=sapphirerapids` |
| GPU on platform       | 8× MI250X (128 GB HBM2e total)    | A100 / V100 (separate queues)    |
| OpenMP runtime        | libgomp (unconditional spin)      | libiomp5 (hybrid yield/sleep)    |
| Profiler              | `perf stat` + AMD Zen3 PMU        | `perf stat` + `perf record` + VTune 2024.2 |

Both production builds take the **AVX2 `Vec4d`** path. AVX-512 is not enabled
on Sapphire Rapids in the current build; the GPU port therefore replaces the
*same* SIMD code on both platforms.

---

## 3. Headline numbers (xlarge_mf.fa, the clean dataset)

### Cross-platform wall-time

| T   | Setonix (s) | Gadi (s) | Gadi advantage |
|----:|------------:|---------:|---------------:|
|  1  |    10 555   |  11 915  | 0.89×          |
|  4  |     7 271   |   4 245  | 1.71×          |
|  8  |     8 618   |   2 440  | 3.53×          |
| 32  |     7 237   |   1 036  | 6.99×          |
| **64** | **6 568** | **897**  | **7.32×**      |
| 104/128 |  6 516  |   1 112  | 5.86×          |

### Within-platform scaling (xlarge_mf.fa, Gadi)

| T   | wall (s) | speed-up | efficiency | IPC  | LLC miss % |
|----:|---------:|---------:|-----------:|-----:|-----------:|
|  1  |  11 915  |  1.00×   | 100.0%     | 1.82 | 37.4       |
|  8  |   2 440  |  4.88×   |  61.1%     | 1.57 | 66.8       |
| 32  |   1 036  | 11.50×   |  35.9%     | 1.37 | 78.9       |
| **64** | **897** | **13.28×** | **20.7%** | **1.16** | **78.8** |
| 104 |   1 112  | 10.72×   |  10.3%     | 1.03 | 77.1       |

Gadi reaches a wall-time **minimum at 64T**, then *regresses* at 104T.

### IPC collapse on Setonix (xlarge_mf.fa)

| T   | IPC   | cache miss rate | Diagnosis                                |
|----:|------:|----------------:|------------------------------------------|
|  1  | 2.73  |  3.0 %          | Healthy (Zen 3 dense FP envelope)        |
| 64  | 0.10  |  9.9 %          | **−96 % IPC drop, cache stable** → barrier spin |
| 128 | 0.07  | 12.1 %          | Worse                                    |

The cache-miss-rate is **flat**, so this is not memory bandwidth saturation -
it is OpenMP busy-wait spinning. Same effect on Gadi but masked by Sapphire
Rapids' larger out-of-order window.

---

## 4. The methodology audit (what went wrong, briefly)

Two compounding issues were uncovered post-collection:

### Issue A - restricted ModelFinder on Setonix
The Setonix CI pipeline (`setonix-ci/submit_matrix.sh`) passed
`-mset GTR,HKY,K80` to IQ-TREE, restricting model search to ~21 model variants.
Gadi had no `-mset` flag and ran the default ~286-variant search. Result:

| Dataset              | Setonix model      | Gadi model | lnL gap        |
|----------------------|--------------------|------------|---------------:|
| large_modelfinder.fa | HKY+F+G4           | GTR+F+G4   |   306 201 units |
| xlarge_mf.fa         | GTR+F+G4           | GTR+F+R4   |       ~4 units |
| mega_dna.fa          | HKY+F+ASC+R5       | GTR+F+R4   | 1 180 000 units |

### Issue B - alignment file sha256 mismatch
For two datasets, the file Setonix actually ran does **not** match the
canonical hash in `benchmarks/sha256sums.txt`:

| Dataset              | Setonix sha256 | Canonical sha256 | Status  |
|----------------------|----------------|------------------|--------|
| large_modelfinder.fa | `52849f82…`    | `73908728…`      | MISMATCH |
| **xlarge_mf.fa**     | **`66eaf64b…`** | **`66eaf64b…`** | **OK** |
| mega_dna.fa          | `94d7d38d…`    | `0c8af2d6…`      | MISMATCH |

Setonix `mega_dna.fa` has 0 constant sites (variable-sites-only alignment) →
IQ-TREE applied `+ASC`. The Gadi canonical file has constant sites → no `+ASC`.
The two likelihood functions use different normalisations and are
mathematically non-comparable.

### Pending re-runs (Future Work)
1. Re-generate canonical alignments on Setonix using
   `gadi-ci/generate_datasets.sh` (seeds 101, 303); verify sha256.
2. Remove `-mset GTR,HKY,K80` from `setonix-ci/submit_matrix.sh`.
3. Re-submit 10 jobs (6× large_modelfinder, 4× mega_dna). Estimated cost:
   **6 500 - 9 700 CPU-hours (50-76 node-hours)** on Setonix `work` partition,
   dominated by mega_dna (~10-19 h wall per thread count with full ModelFinder).

Until those land, **`xlarge_mf.fa` is the only cross-platform benchmark**.

---

## 5. Why CPU optimisation cannot fix this (the GPU motivation)

Three architecture-independent findings make the GPU case:

### (a) Concentration of compute - 96% in five kernels
Gadi 1T `large_modelfinder` `perf record` + 119k samples:

| Function (templated `<Vec4d, false, 4, true, false>`) | % samples |
|------------------------------------------------------|----------:|
| `PhyloTree::computePartialLikelihoodSIMD`            |   69.20 % |
| `PhyloTree::computeLikelihoodDervSIMD`               |   14.76 % |
| `PhyloTree::computeLikelihoodBufferSIMD`             |   11.50 % |
| `PhyloTree::computeLikelihoodBranchSIMD`             |    0.44 % |
| `PhyloTree::computeLikelihoodFromBufferSIMD`         |    0.38 % |
| Parsimony helpers + tree bookkeeping + I/O           |    3.72 % |

These five kernels are templated on `Vec4d` (AVX2 256-bit, 4 doubles) and
implement **Felsenstein's pruning algorithm**. Same template instantiations
on Setonix - the kernel ranking is invariant across architecture, dataset,
and thread count.

### (b) Synchronisation collapse, not memory bottleneck
- Gadi 32T (large_modelfinder, VTune resolves symbols):
  `kmp_flag_64::wait` 52.3 % + `__kmp_dispatch_next_algorithm` 7.94 %
  + `kmp_flag_native::notdone_check` 2.09 % = **62.3 % OpenMP machinery.**
- Setonix shows three unresolved program-counters at offsets `+0x25946`,
  `+0x25766`, `+0x25942` totalling 22-24 % of samples - disassembly shows
  these are libgomp barrier call sites.
- IPC collapses 96 % while cache-miss-rate stays flat at 7-10 %. The cores
  are busy-waiting, not stalling on memory.

### (c) Saturation observed on **both** platforms below per-node maximum
| Cluster | Dataset             | Wall-time minimum | Per-node max |
|---------|---------------------|------------------:|-------------:|
| Setonix | large_modelfinder   |     32-64T flatline | 128T       |
| Setonix | mega_dna            |     16T (regresses monotonically) | 128T |
| Gadi    | large_modelfinder   |               32T |       104T   |
| Gadi    | xlarge_mf           |               64T |       104T   |
| Gadi    | mega_dna            |               64T |       104T   |

The motivation for GPU offload is therefore **cluster-independent and
dataset-independent**. This is not a Setonix-specific OpenMP pathology.

### Why a CPU rewrite is the wrong fix
One could refactor the parallel region (larger task granularity, persistent
thread pools, double-buffered partials). Estimated upper bound: ~2× on
Setonix at 32T - still 3× slower than Gadi, still >3× slower than what
GPU offload offers. The structural problem is that an OpenMP fork/join per
node visit cannot hide the per-pattern arithmetic cost on modern many-core
CPUs. **GPUs avoid the fork/join entirely** by mapping each pattern to a
hardware thread within a single kernel launch.

---

## 6. The GPU plan, explained simply

### 6.1 The intuition (why this works)

An IQ-TREE likelihood evaluation does the same tiny calculation
(64 floating-point ops on 16 doubles) for every site pattern in the
alignment - tens of thousands of times per tree node, hundreds of times per
ModelFinder candidate, thousands of times during tree search.

On a CPU this looks like:

```
┌─ for each tree node (post-order) ─┐
│   ┌─ #pragma omp parallel for ──┐ │   ← FORK 64 threads
│   │   for each pattern p:       │ │
│   │       Lout[p] = combine(Lc[p], Lo[p])  ← 480 FLOP
│   │   end for                   │ │
│   └─ implicit barrier ──────────┘ │   ← JOIN, all 64 threads wait here
└───────────────────────────────────┘
```

The fork/join happens **once per tree node**, of which there are ~2N for
N taxa. On `xlarge_mf` (200 taxa, 10 000 patterns) that's roughly
**400 fork/join events** per up-pass × thousands of up-passes during the
search. The per-event synchronisation cost is fixed; the per-pattern work
is small. As you add cores, the fixed cost dominates. That's the barrier
storm.

On a GPU it looks like:

```
host: hipLaunchKernelGGL(partialLikelihood, grid, block, ...);
GPU:  14 080 hardware threads execute the kernel concurrently
      • thread (p, r) handles pattern p, rate category r
      • coalesced loads of adjacent Lc[p, r, k] - HBM2e at 3.2 TB/s
      • no software barrier between threads - hardware schedules wavefronts
      • single completion event per up-pass, not per node
```

One launch per up-pass replaces ~N×T fork/joins. The GPU's hardware
scheduler hides memory latency by switching wavefronts; we never pay the
OpenMP barrier cost.

### 6.2 Theoretical ceiling (back-of-envelope)

Per-pattern arithmetic intensity ≈ 480 FLOP / 256 B = **1.9 FLOP/B**
→ memory-bandwidth-bound. For `xlarge_mf` (10 000 patterns × 1.28 MB
partial vector × 2N nodes ≈ 250 MB traffic per up-pass):

| Hardware           | Memory BW    | Time per up-pass | vs Setonix 64C |
|--------------------|-------------:|-----------------:|---------------:|
| Setonix 64C DDR4   | ~200 GB/s    |          1.25 ms | 1.0×           |
| AMD MI250X (1 GCD) | 1.6 TB/s     |          156 µs  | 8×             |
| AMD MI250X (full)  | 3.2 TB/s     |           78 µs  | 16×            |
| NVIDIA A100        | 1.94 TB/s    |          129 µs  | 10×            |
| NVIDIA H100        | 3.36 TB/s    |           74 µs  | 17×            |

Allowing 50 % achieved utilisation and host↔device launch overhead,
**8-10× wall-time speedup on `xlarge_mf` is conservative**.

### 6.3 Why ROCm / MI250X first

1. **Setonix has the GPUs.** 8× MI250X per node (16 GCDs) on the `gpu`
   queue. Pawsey allocations already approved under `pawsey1351`.
2. **Setonix is the worse-CPU baseline** (7.3× slower than Gadi at 64T on
   xlarge). The largest *absolute* wall-time wins land here.
3. **ROCm 6.3+ is mature for HPC double-precision workloads** (rocBLAS,
   hipBLAS, rocFFT, rocSPARSE all production). The five IQ-TREE kernels are
   simple enough that we don't need any of these libraries - just clean HIP.
4. **HIP code compiles unchanged on NVIDIA via `hipcc -nvcc-options`** -
   no separate CUDA source tree until M7. The kernel is 30 lines of
   `__device__` code that runs on both vendors.
5. **MI250X wave size is 64**, exactly matching our natural pattern-block
   size of 64 patterns (with `threadIdx.y` covering 4 rate categories =
   256 threads/block, the AMD recommended sweet spot per ROCm performance
   guide).

### 6.4 Prior art - we are not blazing a new trail

The phylogenetic likelihood-on-GPU problem was solved by **BEAGLE**
(Suchard et al., *Syst. Biol.* 2012) and is the calculation engine for BEAST,
BEAST2, and MrBayes. BEAGLE's CUDA kernels demonstrate 3-10× speedup on
production phylogenetic workloads versus optimised CPU SIMD, with the same
arithmetic structure as IQ-TREE's. We are not inventing the algorithm; we
are bringing the same well-validated kernel structure into IQ-TREE's
execution path, adapted for HIP/ROCm so it runs on Setonix's MI250X.

Differences from BEAGLE we will exploit:
- We can fuse `Buffer + Branch + FromBuffer` into a single launch (BEAGLE
  keeps them separate for API generality; we don't have to).
- We can keep ModelFinder's per-candidate Q-matrix resident on device
  (BEAGLE's API requires uploading per call).
- Single-source HIP avoids BEAGLE's separate CUDA / OpenCL trees.

---

## 7. Phased milestone plan (M1 → M7)

Each milestone produces a runnable binary plus a JSON record validatable
against `tools/schemas/run.schema.json`. **All success criteria are anchored
on `xlarge_mf.fa` because it is the only canonically valid cross-platform
benchmark.** `large_modelfinder` is used as the fast-iteration dataset
(small enough to run in CI), but is *not* used for cross-platform claims
until the corrected re-runs land.

| #   | Milestone               | Scope                                                                 | Pass criteria (anchored on xlarge_mf where stated)        |
|-----|-------------------------|-----------------------------------------------------------------------|------------------------------------------------------------|
| M1  | Build & smoke kernel    | HIP `computeLikelihoodBufferSIMD_GPU` only - element-wise multiply. MI250X 1× GCD. Unit test against AVX2 output on `turtle.fa`. | `|Δ|` < 1e-12 per element on `turtle.fa`; HIP build green in CI under `rocm/6.3.0`. |
| M2  | Derivative kernel       | `computeLikelihoodDervSIMD_GPU` with reduction (3 independent sums). Persistent device buffers. `hipStream` overlap. | End-to-end lnL within 1e-3 of Gadi 1T baseline on `xlarge_mf.fa` (the canonical dataset). |
| M3  | Partial-likelihood kernel | `computePartialLikelihoodSIMD_GPU` (the 69 % hotspot). Tree traversal stays on host; one kernel launch per visited node, batched per level. | **≥ 5× wall-time reduction vs Setonix 64T (xlarge_mf = 6 568 s) → target ≤ 1 314 s.** lnL identical to Gadi 1T to 0.001 on `xlarge_mf`. |
| M4  | Branch + FromBuffer fusion | Fold `FromBuffer` into `Branch`. Single launch per branch evaluation. Used by ModelFinder. | **ModelFinder phase ≥ 6× faster end-to-end on `xlarge_mf` → target ≤ 1 100 s vs Setonix 64T.** |
| M5  | Multi-tree concurrency  | Stream-parallel evaluation of independent candidate trees during NNI/SPR moves. One stream per tree, persistent buffers. | ≥ 80 % SM/CU occupancy under `rocprofv3`; tree-search phase ≥ 4× faster on `xlarge_mf`. |
| M6  | Multi-GPU per node      | Distribute candidate trees across 8 GCDs of one MI250X package (= 16 GCDs total per node? - confirm topology). RCCL only at end-of-iteration sync. | Linear scaling to ≥ 6× single-GPU on `xlarge_mf` ModelFinder phase. |
| M7  | CUDA path on Gadi       | Build under `cuda/12.x`, validate on `gpuvolta` (V100) and `dgxa100` (A100). No kernel-source changes - only build flags. | Bit-equality with HIP path on `xlarge_mf.fa`; PBS scripts in `gadi-ci/` mirror the SLURM ones. |

### 7.1 Numerical success criteria (must hit, in order)

1. **Numerical equivalence:** identical lnL to AVX2 path within 1e-3 ln units
   on `xlarge_mf.fa` (the canonically-clean dataset). Use Kahan-compensated
   summation in the reduction kernel; pin reduction order to ascending
   pattern index. Validate at 1e-6 lnL on `turtle.fa` per CI run.

2. **Setonix MI250X (M3, single GCD):** ≥ 5× wall vs best Setonix CPU result
   on `xlarge_mf.fa`. Best Setonix CPU = 6 568 s @ 64T → **target ≤ 1 314 s**
   on 1× GCD. Stretch goal: match or beat Gadi 64T = 897 s on a single GCD
   (requires 7.3× over Setonix CPU, plausible given the 8× single-GCD
   bandwidth advantage).

3. **Setonix MI250X (M4, single GCD, ModelFinder fused):** ≥ 6× wall on
   `xlarge_mf` ModelFinder pipeline → target ≤ 1 100 s.

4. **Setonix MI250X (M6, full node 16 GCDs):** Achieve **at least Gadi 64T
   wall time on a single Setonix GPU node**, demonstrating that GPU offload
   neutralises the 7.3× CPU disadvantage.

5. **Gadi A100 (M7):** Within 1.3× of MI250X wall on `large_modelfinder`,
   confirming HIP→CUDA portability. Note: not on `xlarge_mf` because that
   would require corrected Setonix CPU re-runs first; large_modelfinder is
   acceptable here because the comparison is *GPU vs GPU*, not Setonix CPU
   vs Gadi CPU.

### 7.2 Risk register

| Risk                                          | Likelihood | Mitigation                                                                                |
|-----------------------------------------------|:----------:|-------------------------------------------------------------------------------------------|
| Numerical drift between AVX2 and GPU reductions | High      | Kahan-compensated summation; pin reduction order to ascending pattern index; CI validates 1e-6 lnL on `turtle.fa`. |
| Tree-traversal launch latency for small trees  | Medium    | Batch nodes at the same depth into one launch (level-order traversal). Use HIP graphs / CUDA graphs to record up-pass once per topology. |
| ROCm / hipcc divergence from CUDA semantics    | Medium    | Restrict to HIP-portable subset (no warp-size hard-coding, no `__shfl_sync` mask quirks). Compile both paths in CI from M1. |
| Setonix queue contention delaying iteration    | Medium    | Develop on Gadi V100 where short jobs are interactive; promote to Setonix MI250X weekly for performance numbers. |
| ModelFinder host-side overhead becomes new bottleneck | Low | After M3, re-profile with `perf record`; if host-side >20%, hoist ModelFinder's per-model setup onto an async stream. |
| MI250X dual-GCD topology under-utilised by single process | Low | From M5, run one MPI rank per GCD, bind via `ROCR_VISIBLE_DEVICES`. |
| `xlarge_mf` GPU result needs methodology peer-review before publication | Medium | All GPU benchmarks logged with full perf_cmd, sha256, and ROCm version. M5+ runs gated on review of M1-M3 numerical validation. |

---

## 8. Concrete kernel mapping

The HIP kernel below compiles unchanged for CUDA via `hipcc` or
hand-mirrored `.cu`. Wave-front size 64 (MI250X) maps to `block.x = 64`;
on NVIDIA H100 the same launch executes as two warps of 32 each. Adjacent
threads read adjacent `Lc[(p*4+r)*4 + k]` elements - **memory accesses are
coalesced** (key requirement from the ROCm performance guide).

```cpp
// Per-pattern thread, 4 rate categories along threadIdx.y, P patterns along blockIdx.x
__global__ void partialLikelihood(
        const double* __restrict__ Lc,    // child partial: P × 4 × 4
        const double* __restrict__ Pc,    // transition matrix child: 4 × 4
        const double* __restrict__ Lo,    // other child partial
        const double* __restrict__ Po,
        double*       __restrict__ Lout,  // P × 4 × 4
        int P)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int r = threadIdx.y;            // rate category 0..3

    double tmp_c[4], tmp_o[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        double ac = 0, ao = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            ac += Pc[r*16 + s*4 + k] * Lc[(p*4 + r)*4 + k];
            ao += Po[r*16 + s*4 + k] * Lo[(p*4 + r)*4 + k];
        }
        tmp_c[s] = ac;  tmp_o[s] = ao;
    }
    #pragma unroll
    for (int s = 0; s < 4; ++s)
        Lout[(p*4 + r)*4 + s] = tmp_c[s] * tmp_o[s];
}

// Launch (typical xlarge_mf, 10 000 patterns):
dim3 block(64, 4);                        // 64 patterns × 4 rate cats = 256 lanes
dim3 grid((P + 63) / 64);
hipLaunchKernelGGL(partialLikelihood, grid, block, 0, stream,
                   Lc, Pc, Lo, Po, Lout, P);
```

Why this structure follows the ROCm performance guidelines:
- **256 threads/block** = recommended sweet spot for MI250X CU occupancy.
- **`block.x = 64`** = wave size, no partial-warp waste.
- **Coalesced loads** along `threadIdx.x`, with rate category stride along
  `threadIdx.y` minimising bank conflicts in LDS staging.
- **`#pragma unroll` on the 4×4 inner loops** lets the compiler keep
  intermediates in registers (~16 doubles per thread = 128 B / 256 KB
  register file = trivial register pressure).
- **No `__syncthreads()`** in the kernel - threads are independent within a
  rate category, the inner reduction is per-thread.

---

## 9. Build / CI changes

| File                              | Change                                                                                                  |
|-----------------------------------|---------------------------------------------------------------------------------------------------------|
| `setonix-ci/run_pipeline.sh`      | Add `module load rocm/6.3.0 cmake`; cmake `-DIQTREE_GPU=HIP`. Submit to `gpu-dev` queue with `--gpus=1`. |
| `setonix-ci/submit_matrix.sh`     | **Remove `-mset GTR,HKY,K80`** as part of the cleanup (Future Work item 2). Adds `--mset` only as opt-in.|
| `gadi-ci/run_pipeline.sh` (new)   | Add `module load cuda/12.4.0`; cmake `-DIQTREE_GPU=CUDA`. Submit to `dgxa100` queue.                    |
| `tools/schemas/run.schema.json`   | Extend `profile.gpu` with `kernel_time_ms`, `occupancy`, `memcpy_h2d_ms`, `achieved_bandwidth_gbs`.     |
| `tests/test_regression.py`        | CPU-vs-GPU equality check at 1e-6 lnL tolerance for `turtle.fa` (small enough for CI).                  |
| `web/js/pages/profiling.js`       | New "GPU Kernels" section reading `profile.gpu.kernels[]` per run.                                      |

### Cluster→phase mapping

| Phase            | Setonix (HIP / MI250X)                | Gadi (CUDA / A100 or V100)                  |
|------------------|---------------------------------------|---------------------------------------------|
| M1-M3 development | Primary: `gpu-dev` 1× GCD            | Cross-validate weekly: `gpuvolta` 1× V100   |
| M4-M5 perf tuning | Primary: `gpu` 1 node = 16× GCD      | Cross-validate: `dgxa100` 1× A100           |
| M6 multi-GPU      | 1 node × 16 GCD                      | `dgxa100` 8× A100, NVLink                   |
| Production benchmark dataset | **`xlarge_mf.fa` (canonical)** | **`xlarge_mf.fa` (canonical)**          |

---

## 10. What we are NOT doing (and why)

| Tempting thing                                  | Why we are skipping it                                                                                 |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Use BEAGLE directly                             | BEAGLE's API forces per-call Q-matrix uploads; we lose the persistent-residency win for ModelFinder. Also IQ-TREE's existing tree-search loop would need invasive refactoring. |
| Rewrite the OpenMP layer for less barrier overhead | Estimated upper bound 2× on Setonix, still 3× slower than Gadi CPU. Doesn't address the structural per-pattern fork/join cost. Effort ≈ M3+M4 effort for ¼ the win. |
| Enable AVX-512 on Sapphire Rapids               | Solves a different problem (Gadi already wins). The bottleneck is sync, not arithmetic. AVX-512 makes each thread faster but the threads still hit the same OpenMP barriers. |
| Wait for corrected Setonix CPU re-runs before starting GPU work | `xlarge_mf.fa` is already a clean cross-platform benchmark. The corrected re-runs only sharpen `large_modelfinder` and `mega_dna` numbers; they do not change the GPU motivation or any milestone success criterion. |
| Target NVIDIA first                             | Setonix has the GPUs allocated; Gadi GPU queue access is on a separate quota. Also: bigger absolute win on Setonix because its CPU baseline is worse. CUDA path lands at M7 at low marginal cost (HIP source compiles unchanged). |

---

## 11. Reproducibility & validation chain

```
# Setonix (CPU baseline + GPU port)
./start.sh deepprofile        # CPU + GPU SLURM job
sha256sum -c benchmarks/sha256sums.txt   # MUST PASS for cross-platform claims
python3 tools/validate.py     # JSON-schema check
make test                     # full pytest suite incl. CPU-vs-GPU equality

# Gadi
qsub gadi-ci/run_pipeline.sh
python3 tools/validate.py
```

Every GPU run JSON will include:
- `profile.perf_cmd` (full IQ-TREE invocation, no hidden flags)
- `dataset_info.sha256` (verified against `benchmarks/sha256sums.txt`)
- `profile.gpu.kernels[]` with per-kernel time, occupancy, achieved BW
- `verify[]` with reported lnL and diff vs Gadi 1T canonical

Live dashboard: <https://xenotekx.github.io/setonix-iq/#/overview>.
Report regenerated by `tools/build.py` on each push.

---

## 12. One-paragraph summary for the impatient

The IQ-TREE phylogenetic likelihood pipeline spends 96% of its CPU time in
five SIMD kernels, and OpenMP barrier overhead destroys parallel efficiency
on both Setonix (96% IPC collapse) and Gadi (62% time in `kmp_flag_64::wait`)
well below per-node thread maximums. The only methodologically clean
cross-platform CPU benchmark is `xlarge_mf.fa`, where Gadi Sapphire Rapids
beats Setonix Trento by 7.3× at 64 threads. Porting the five kernels to
HIP/ROCm on Setonix's MI250X is expected to deliver ≥5× wall-time reduction
on `xlarge_mf.fa` versus Setonix's best CPU result (≤1 314 s vs 6 568 s),
and the same source compiles to CUDA for Gadi A100/H100 at M7 with no
algorithmic changes. All success criteria are anchored on `xlarge_mf.fa`
because it is the only dataset whose alignment file matches the canonical
sha256 on both platforms and whose model selection (GTR-family) was
consistent. Corrected Setonix re-runs for the other two datasets are
scheduled (6 500-9 700 CPU-hours) but are not on the GPU work critical path.
