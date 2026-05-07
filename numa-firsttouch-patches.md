# IQ-TREE NUMA First-Touch Patches — R1 + R2

**Source tree:** `/scratch/pawsey1351/asamuel/iqtree3-numa-firsttouch/src/iqtree3/`
**Build:** AOCC 5.1.0 / libomp, `build-profiling-aocc/iqtree3` (tagged on disk as `iqtree3.numa-firsttouch-r2`)
**Hardware:** AMD EPYC 7763 "Milan" (2 sockets × 8 CCDs × 8 cores; 128 logical threads)
**Validation:** log-likelihood `−10956936.6117` bit-identical across baseline / R1 / R2 at 32T/64T/128T
**Wall-time impact (xlarge_mf):** 32T −34.7%, 64T −56.4%, 128T **−68.9%** (2368 s → 736.6 s)

---

## What problem the patches solve

Linux uses **first-touch** placement: the *first* thread to write a virtual page decides which NUMA node holds the physical page for the rest of the process's lifetime. IQ-TREE was filling several large per-pattern arrays from a single thread (the master), pinning every page to one NUMA node. When the parallel kernel later spawned threads on the *other* socket, every read from those arrays was a remote-DRAM load with cross-socket coherence cost.

The fix is two pieces:

- **R1** — make the *one-time* init loops (`ptn_freq`, `ptn_invar`) write in parallel with `schedule(static)` so each page is first-touched by the worker that will own that pattern range.
- **R2** — extend the same idiom to (a) the *per-call* `_pattern_lh_cat` zero-fill and (b) the kernel packet loops in `phylokernelnew.h`, which were using `schedule(dynamic,1)` and randomly re-assigning packets to threads each call (defeating any locality R1 established).

Together R1+R2 give a consistent `(thread → static pattern range)` mapping end-to-end: pages are placed by the writing thread, and the same thread reads them back forever after.

---

## R1 — One-time init array first-touch

### R1a: `ptn_freq` parallel-static fill

**File:** `tree/phylotreesse.cpp:546`
**Function:** `PhyloTree::computePtnFreq`
**Hot path:** read every iteration of the likelihood Newton-Raphson loop (#1 hot func, 40% CPU)

**Before** (serial fill on master thread, all pages first-touched on master's NUMA node):
```cpp
void PhyloTree::computePtnFreq() {
    if (ptn_freq_computed) return;
    size_t nptn = aln->size();
    for (size_t ptn = 0; ptn < nptn; ptn++)
        ptn_freq[ptn] = (*aln)[ptn].frequency;
    ptn_freq_computed = true;
}
```

**After** (parallel-static fill — each worker first-touches the pages it will later read):
```cpp
void PhyloTree::computePtnFreq() {
    if (ptn_freq_computed) return;
    size_t nptn = aln->size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t ptn = 0; ptn < nptn; ptn++)
        ptn_freq[ptn] = (*aln)[ptn].frequency;
    ptn_freq_computed = true;
}
```

### R1b: `ptn_invar` parallel-static fill

**File:** `tree/phylotreesse.cpp:577`
**Function:** `PhyloTree::computePtnInvar`
**Hot path:** read every iteration of `computeLikelihoodDervSIMD` and `computeLikelihoodBranchSIMD`

**Before:**
```cpp
void PhyloTree::computePtnInvar() {
    size_t nptn = aln->size();
    memset(ptn_invar, 0, sizeof(double) * nptn);
    // ... per-state contribution accumulation (already mostly serial init) ...
    for (size_t ptn = 0; ptn < nptn; ptn++)
        ptn_invar[ptn] *= state_freq_factor;
}
```

**After:**
```cpp
void PhyloTree::computePtnInvar() {
    size_t nptn = aln->size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t ptn = 0; ptn < nptn; ptn++)
        ptn_invar[ptn] = 0.0;
    // ... per-state contribution accumulation ...
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t ptn = 0; ptn < nptn; ptn++)
        ptn_invar[ptn] *= state_freq_factor;
}
```

### Why R1 alone wasn't enough

R1 fixed the two arrays that are written *once at startup* and then read forever. That helped 32T/64T runs modestly (the 64T cross-CCD hop is what those reads were paying), but **didn't move 128T at all**. The reason: at 128T the dominant penalty isn't the one-time init pages — it's the per-call buffers the kernel zero-fills on every NNI iteration, plus the fact that the kernel's `schedule(dynamic,1)` was re-assigning packets to different threads each call. Even with R1 in place, dynamic stealing kept defeating page locality.

---

## R2 — Per-call buffer + kernel scheduler

### R2a: `_pattern_lh_cat` parallel-static zero-fill

**File:** `tree/phylotreesse.cpp:1294-1295`
**Function:** `PhyloTree::computeLikelihoodBranchEigen`
**Hot path:** called once per `computePatternLhCat` invocation, which happens on every NNI candidate evaluation (thousands of times per tree-search iteration)
**Buffer size:** `nptn * ncat_mix * sizeof(double)` ≈ 1.6 MB for xlarge_mf

**Before** (serial `memset` — all pages first-touched on master):
```cpp
double prob_const = 0.0;
memset(_pattern_lh_cat, 0, sizeof(double) * nptn * ncat_mix);
// ... downstream reduction loops at :1321 and :1365 use schedule(static) ...
```

**After** (parallel-static zero-fill — page placement matches the static reduction loops below):
```cpp
double prob_const = 0.0;
// NUMA first-touch: parallel-static zero-fill so _pattern_lh_cat pages are
// placed on the same NUMA node as the worker that later reads/writes them
// in the static-scheduled reduction loops below (lines 1321 and 1365).
{
    size_t lh_cat_n = nptn * ncat_mix;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < lh_cat_n; k++)
        _pattern_lh_cat[k] = 0.0;
}
```

### R2b: kernel `schedule(dynamic,1)` → `schedule(static)` (5 sites)

**File:** `tree/phylokernelnew.h`
**Pattern at every site:**

```cpp
// Before
#pragma omp parallel for schedule(dynamic,1) num_threads(num_threads)
for (int packet_id = 0; packet_id < num_packets; packet_id++) {
    // ... per-packet kernel work ...
}

// After
#pragma omp parallel for schedule(static) num_threads(num_threads)
for (int packet_id = 0; packet_id < num_packets; packet_id++) {
    // ... per-packet kernel work ...
}
```

Sites:

| Line | Enclosing function | Hot rank | Why this site matters |
|---|---|---|---|
| 1275 | `PhyloTree::computeTraversalInfo` (drives `computePartialLikelihoodSIMD`) | #2 (18.3% CPU) | Pre-search post-order traversal; `partial_lh` and `scale_num` pages first-touched on call N must be read locally on call N+1. Dynamic stealing was sending page reads cross-CCD/socket. |
| 2386 | `PhyloTree::computeLikelihoodDervGenericSIMD` | #1 (40.1% CPU) | The Newton-Raphson driver — single highest-impact NUMA site in the whole program. Any per-call cross-socket traffic here multiplies by thousands of branch-length optimizations per tree-search step. |
| 2838 | `PhyloTree::computeLikelihoodBranchGenericSIMD` (SAFE_NUMERIC path) | #4 | Reads `theta_all` and `ptn_invar` per packet; static schedule pairs each thread with the same pattern range across the entire ModelFinder sweep. |
| 3005 | `PhyloTree::computeLikelihoodBranchGenericSIMD` (non-SAFE_NUMERIC fast path) | #4 | Second parallel region in same function, reachable from a different code path; needs the same fix or the fast path re-introduces dynamic. |
| 3595 | `PhyloTree::computeLikelihoodDervMixlenGenericSIMD` | #1 (mixlen variant) | Mirror of 2386 for mixed-branch-length models; without it, mixlen workloads would still hit the cliff. |

The `num_packets` loop is sized by `computeBounds` to be roughly equal-cost per packet, so the load-balancing argument that originally motivated `schedule(dynamic,1)` is weak in practice. We measured the trade-off: any imbalance loss is dwarfed by the NUMA-locality win at every thread count.

---

## How R1 + R2 combine

The two patch sets enforce the same `(thread → static pattern range)` mapping on both sides of every page:

- **Write side** — R1 (`ptn_freq`, `ptn_invar`) and R2a (`_pattern_lh_cat`): pages are first-touched by the worker that owns that range under `schedule(static)`.
- **Read side** — R2b (5 kernel sites): the *same* worker reads the *same* range every call, so subsequent reads always hit local DRAM.

Result: the cross-socket cliff (32T → 64T → 128T was 1940 → 1905 → 2368 s) becomes super-linear scaling (1267 → 831 → 737 s).

---

## Hardware verification

`perf stat` counters from the same SLURM jobs that produced the wall-time numbers:

| Counter | Meaning | 128T baseline | 128T R2 | Δ |
|---|---|---|---|---|
| IPC | instructions / cycle | 0.556 | 0.733 | +32% |
| L3 miss % | last-level cache miss rate | 9.08% | 5.37% | −41% |
| `l2_pf_miss_l2_l3` | AMD: L2 prefetches that missed L2 *and* L3 (i.e. went to a remote node) | 636 B | 289 B | **−55%** |
| L1d miss % | L1 data cache miss rate | ~unchanged | ~unchanged | — |

The unchanged L1d combined with the dramatic drop in `l2_pf_miss_l2_l3` is the smoking gun: the patches didn't change *what* is computed (same instructions, same L1 access pattern), they changed *where the data lives* (no longer cross-socket). That is exactly the signature of a NUMA-locality fix.

---

## Provenance

| Item | Value |
|---|---|
| Source tree | `/scratch/pawsey1351/asamuel/iqtree3-numa-firsttouch/src/iqtree3/` |
| Build dir | `/scratch/pawsey1351/asamuel/iqtree3-numa-firsttouch/build-profiling-aocc/` |
| Binary tag | `iqtree3.numa-firsttouch-r2` |
| Compiler | AOCC 5.1.0 (`clang` from AOCC), OpenMP runtime: libomp |
| `KMP_BLOCKTIME` | 200 (matches Gadi libiomp5 default for fair comparison) |
| SLURM jobs (R2 sweep) | 32T = 42422004, 64T = 42422005, 128T = 42422006 |
| Result JSONs | `logs/runs/xlarge_mf_{32,64,128}t_clang_omp_pin_numa_ft_r2.json` |
| Dashboard label | `AOCC · NUMA patch r2` |
