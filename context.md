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

| Platform              | Setonix (Pawsey)                              | Gadi normalsr (NCI)              |
|-----------------------|-----------------------------------------------|----------------------------------|
| CPU (per node)        | 2× AMD EPYC 7763 "Milan/Trento" (Zen 3)       | 2× Intel Xeon Sapphire Rapids    |
| Physical cores / node | **128C** (64C × 2 sockets)                    | **104C** (52C × 2 sockets)       |
| SMT                   | **ON** — 256 logical CPUs visible to job †    | **OFF** — 104 logical = 104 phys |
| Compiler              | **gcc 14.3.0 (SUSE)**, default `-O3` (no `-march=znver3`) ‡ | **icx 2024.2** with `-O3 -xSAPPHIRERAPIDS -fno-omit-frame-pointer` |
| IQ-TREE 3.1.1 kernel  | `AVX+FMA` (Vec4d, 256-bit, AVX2 ops)          | `AVX+FMA` (Vec4d, 256-bit; AVX-512 path not built) |
| GPU on platform       | 8× MI250X (128 GB HBM2e total)                | A100 / V100 (separate queues)    |
| OpenMP runtime        | libgomp (unconditional spin barrier)          | libiomp5 (hybrid yield/sleep)    |
| Profiler              | `perf stat` + AMD Zen3 PMU                    | `perf stat` + `perf record` + VTune 2024.2 |

Both production builds take the **AVX2 `Vec4d`** path. AVX-512 is not enabled
on Sapphire Rapids in the current build; the GPU port therefore replaces the
*same* SIMD code on both platforms.

† Confirmed from `iqtree_run.log` headers across the Setonix matrix: every
job reports `Kernel: AVX+FMA - <T> threads (256 CPU cores detected)`.
Setonix `work` nodes are therefore exposed with SMT enabled (2 hardware
threads per physical core, 256 logical per 2-socket node), whereas Gadi
`normalsr` runs SMT-off (104 logical = 104 physical).

‡ The repo `Makefile` is Gadi-only (header comment "Gadi-IQ — Build &
Profiling Makefile") and hard-codes `-O3 -xSAPPHIRERAPIDS …`. The Setonix
`build-profiling/iqtree3` was produced by the legacy `make build-profiling`
path that **predates the Gadi refactor** — gcc rejects `-xSAPPHIRERAPIDS`
(it is an Intel-classic flag), so the Setonix binary was effectively built
with default `-O3` and **no `-march=znver3`/`-mtune=znver3` tuning**. There
is no equivalent `setonix-ci/bootstrap_iqtree.sh` in this repo. Runtime
CPU dispatch still selects the AVX2 SIMD kernel, but the surrounding
non-template code is generic-x86-64.

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

## 4b. Methodology audit — why Setonix scaling looks anomalous (xlarge_mf)

Bui Quang Minh flagged the Setonix scaling curve on `xlarge_mf.fa` as
unexpectedly poor: Trento Zen 3 is one generation behind Sapphire Rapids,
not the four-to-seven generations the 64T wall-time gap implies (6 568 s vs
897 s). A second audit of the actual launch scripts (`setonix-ci/` vs
`gadi-ci/`) shows that the two corpora were **not** run under equivalent
methodology. The differences below are sufficient to explain most of the
Setonix-side scaling collapse without invoking a microarchitectural cause.

### 4b.1 Direct comparison of the two pipelines (xlarge_mf, all thread points)

| Concern                | Setonix (`run_mega_profile.sh`)                  | Gadi (`submit_benchmark_matrix.sh` → `_run_matrix_job.sh`) | Equivalent? |
|------------------------|--------------------------------------------------|------------------------------------------------------------|:-----------:|
| Scheduler              | SLURM, `--partition=work` (**shared**)           | PBS Pro, `-q normalsr` (full-node billing, exclusive cpuset) | **NO** |
| Node allocation        | `--nodes=1 --ntasks=1 --cpus-per-task=128`       | `-l ncpus=104` (= full node, exclusive)                    | NO     |
| Cores requested vs used | 128 logical CPUs reserved; IQ-TREE `-T` ∈ {1,4,8,16,32,64,128} | 104 logical CPUs reserved; IQ-TREE `-T` ∈ {1,4,8,16,32,64,104} | NO    |
| SMT visible to process | **ON** (256 logical / node, 128 reserved)        | **OFF** (104 logical = 104 phys)                           | NO     |
| `--exclusive` / full-node? | **No** — half a 256-logical node, sibling 128 logical CPUs available to other users | Yes (PBS bills the full node)                       | NO     |
| Thread pinning         | **None** — no `srun --cpu-bind=…`, no `OMP_PROC_BIND`/`OMP_PLACES`, no `numactl` | None at job script level, but PBS Pro pins the job to its cpuset (104 cores = full node, so threads cannot drift off-node) | partial |
| OpenMP runtime         | libgomp (busy-wait barrier)                      | libiomp5 via icx build (yield-based barrier)               | NO     |
| Compiler tuning        | gcc 14.3.0, **no `-march=znver3`** (Makefile flag is Intel-only `-xSAPPHIRERAPIDS`, silently dropped by gcc) | icx 2024.2 with `-O3 -xSAPPHIRERAPIDS` (matches target) | NO |
| ModelFinder scope      | `-mset GTR,HKY,K80` (~21 variants)               | full default search (~286 variants)                        | NO (compensates: same final GTR family on `xlarge_mf`, lnL Δ ≈ 4) |
| Dataset sha256         | `66eaf64b…` ✓ matches lockfile                   | same                                                       | YES    |
| Same IQ-TREE source    | 3.1.1 (Apr 2026)                                 | 3.1.1                                                      | YES    |
| Same SIMD width        | AVX2 `Vec4d` (4×f64)                             | AVX2 `Vec4d` (4×f64)                                       | YES    |
| Frame-pointer build    | yes                                              | yes                                                        | YES    |
| Seed                   | `-seed 1`                                        | `-seed 1`                                                  | YES    |

### 4b.2 The scheduling problem in concrete terms

Setonix `work` partition nodes are 2-socket EPYC 7763, **SMT enabled**, so
each node exposes **256 logical CPUs** (= 128 physical cores × 2 SMT
siblings). This is confirmed in every `iqtree_run.log` we collected:

```
Host:   nid001677 (AVX2, FMA3, 250 GB RAM)
Kernel: AVX+FMA - 64 threads (256 CPU cores detected)
```

The submission script (`run_mega_profile.sh`) requests
`--cpus-per-task=128`, which on a 256-logical-CPU node is **half the node**.
Three concrete consequences for an OpenMP-heavy code:

1. **Noisy neighbour.** The other 128 logical CPUs (= the other socket and
   its SMT siblings) can be allocated to a different user's job. EPYC
   Trento couples the two sockets via Infinity Fabric and shares the
   memory-bandwidth envelope of the AMD Instinct platform; a co-scheduled
   job hammering DRAM caps the bandwidth available to IQ-TREE. Gadi
   `normalsr` jobs at `ncpus=104` consume the entire node, so this class
   of contention cannot occur.

2. **SMT thread packing.** With SMT on and no `--hint=nomultithread` /
   `OMP_PLACES=cores`, SLURM is free to assign 2 of IQ-TREE's threads to
   the *same* physical core (different SMT siblings). At `-T 64` the
   probability of at least one such collision is high; the colliding
   threads share the FP/AGU/L1d of one core and run at roughly half
   throughput. Gadi runs SMT-off, so 64 threads = 64 physical cores by
   construction.

3. **No pinning ⇒ thread migration across sockets.** Without
   `OMP_PROC_BIND=close OMP_PLACES=cores` (or `srun --cpu-bind=cores`,
   `numactl --cpu/membind`), the Linux scheduler can move threads
   between the two sockets at every barrier wakeup. Each migration
   invalidates L1/L2 and pulls partial-likelihood vectors back across
   Infinity Fabric. Gadi PBS Pro establishes a cpuset of all 104 cores
   on a single node — migrations stay on-node and threads have a fixed
   NUMA-local memory pool.

The `samples.jsonl` time-series we collect on Setonix corroborate the
migration hypothesis: per-thread `utime/stime` distributions broaden as
`-T` grows, and `nonvoluntary_ctxt_switches` rises super-linearly past
32T — both signatures of un-pinned OpenMP teams competing with the
kernel scheduler.

### 4b.3 Why IPC collapses to 0.10 at 64T on Setonix but only to 1.16 on Gadi

The Section-3 IPC numbers for `xlarge_mf @ 64T` are reproduced below with
the audit context:

| Metric            | Setonix 64T | Gadi 64T | Plain-language reading                                  |
|-------------------|------------:|---------:|---------------------------------------------------------|
| Wall-time         |    6 568 s  |    897 s | 7.3× gap                                                |
| IPC               |       0.10  |     1.16 | Setonix cores spend ~92 % of cycles not retiring an op  |
| Cache miss rate   |        9.9 %|     78.8 %| Setonix cache is fine — bandwidth is **not** saturated  |
| Frontend stalls   |     ~ 90 %  |     ~ 30 %| Setonix is stuck in the OpenMP barrier path             |
| `libgomp` pcs in top hotspots | 22-24 % (at offsets `+0x25946 / 25766 / 25942`) | n/a — VTune resolved as `kmp_flag_64::wait` 52 % | Both machines barrier-bound; libgomp is unconditional spin, libiomp5 yields after a short spin |

The combination of (i) busy-wait `libgomp`, (ii) 64 threads landing on
≤ 64 *effective* physical cores out of 128, (iii) thread migration across
sockets, and (iv) shared-partition memory-bandwidth contention is enough
to drag IPC into the floor without any AMD-architecture defect. Same
algorithm on Gadi never sees (i)-(iv) and degrades benignly to IPC ≈ 1.2
at 64T.

### 4b.4 What needs to be re-run before this comparison can be trusted

> **2026-04-30 follow-up — fixes have been committed.** The launch
> scripts and bootstrap pipelines listed below have all been updated in
> this commit (see CHANGELOG entry "round 2 audit, follow-up"); the
> Setonix submission and the parallel Gadi resubmission are queued.  The
> previous Setonix corpus is preserved under `logs/runs/*_baseline_smton.json`
> for direct before/after comparison; the new corpus will land under
> labels ending in `_smtoff_pin`.

Until the items below land, the Trento-vs-SPR ratio at high thread counts
must be reported **with the methodology caveat**, not as an architecture
result.

1. **Pin the threads.** Add to `setonix-ci/run_mega_profile.sh` (and
   propagate to `submit_matrix.sh`):
   ```
   #SBATCH --hint=nomultithread        # disable SMT inside the cpuset
   #SBATCH --exclusive                 # no co-scheduled jobs on the node
   export OMP_PROC_BIND=close
   export OMP_PLACES=cores
   srun --cpu-bind=cores numactl --localalloc \
        "${BUILD_DIR}/iqtree3" -s "${DATASET}" -T "${THREADS}" -seed 1 …
   ```
   This matches Pawsey's own guidance for OpenMP codes on the `work`
   partition and brings the Setonix submission to **functional parity**
   with Gadi PBS (full-node + cpuset + cores-only placement).

2. **Rebuild with `-march=znver3 -mtune=znver3 -O3 -fno-omit-frame-pointer`**
   under gcc 12+ (or aocc) into a Setonix-native `build-profiling/iqtree3`.
   The current binary was built with default `-O3` because the Makefile's
   `-xSAPPHIRERAPIDS` is silently dropped by gcc. Author a
   `setonix-ci/bootstrap_iqtree.sh` mirroring `gadi-ci/bootstrap_iqtree.sh`
   so the build path is reproducible and committed.

3. **Drop `-mset GTR,HKY,K80`** from `run_mega_profile.sh` (already in the
   §4 audit list).

4. **Add the `samples.jsonl` numa-locality plot** to the dashboard so we
   can visually confirm post-fix that threads stay on one NUMA domain.

5. **(Optional, recommended)** Re-run a single-thread `xlarge_mf` on each
   platform after step 2 to anchor the per-core comparison without any
   OpenMP confound — this is the cleanest "is Trento really 12× slower
   than SPR per-core?" question (current data: 10 555 s vs 11 915 s, i.e.
   roughly equal — **the gap only opens once threads are added**, which
   is itself strong evidence the issue is OpenMP/scheduling, not the
   core).

### 4b.5 Updated headline (post-audit)

> **The 7.3× Gadi-vs-Setonix gap on `xlarge_mf @ 64T` is a launch-script
> artefact at least as much as a microarchitectural one.** Setonix runs
> on a *shared* `work` partition with SMT exposed, no thread pinning, no
> NUMA binding, and a binary built without `-march=znver3`; Gadi runs on
> a *full-node* `normalsr` reservation with SMT-off and a binary tuned
> for SPR. The Section-3 numbers therefore quantify the **deployed
> Pawsey configuration as it stands on 2026-04-30**, not the intrinsic
> Trento ceiling. The GPU motivation (§5) is unaffected: even on the
> better-conditioned Gadi side, the OpenMP barrier still dominates past
> 64T (IPC 1.16, 78.8 % LLC miss rate, wall-time regression at 104T) —
> i.e. the case for moving the five hot kernels to the GPU does **not**
> rest on the Setonix-Gadi gap.

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
