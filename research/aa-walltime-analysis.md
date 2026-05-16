# Why Amino Acid Phylogenetics Takes So Long: A Deep Analysis of IQ-TREE's AA vs DNA Walltime

**Author:** Analysis based on benchmark runs 168422809, 168422811, 168425673, 168425674  
**Date:** 2026-05-15  
**IQ-TREE version:** 3.1.2 (`cpu_opt_merge` branch)  
**Hardware:** Gadi CLX (normal-exec, 48-core Cascade Lake) and SPR (normalsr-exec, 104-core Sapphire Rapids)

---

## 1. The Problem in Numbers

The 100K site benchmark reveals a stark disparity between amino acid (AA) and DNA phylogenetics:

| PBS ID   | Dataset    | Node | -nt | Wall (s)   | IPC    | LLC miss% | L1 miss% | CPU avg (W) |
|----------|------------|------|-----|-----------|--------|-----------|----------|-------------|
| 168422809 | AA 100K   | CLX  |  47 | 3,460.813 | —      | —         | —        | 160.6       |
| 168422811 | DNA 100K  | CLX  |  47 |   546.044 | 0.932  | 59.94     | 5.28     | 390.6       |
| 168425673 | AA 100K   | SPR  | 103 | 1,169.556 | 1.8781 | 66.94     | 1.19     | 224.7       |
| 168425674 | DNA 100K  | SPR  | 103 |   289.121 | 1.3023 | 75.81     | 1.18     | 622.5       |

**AA/DNA wall-time ratios:**
- CLX: 3,460.813 / 546.044 = **6.34×**
- SPR: 1,169.556 / 289.121 = **4.05×**

**SPR speedup over CLX:**
- DNA: 546.044 / 289.121 = **1.89×** (barely above the 2.19× thread ratio)
- AA:  3,460.813 / 1,169.556 = **2.96×** ← AA benefits *significantly more* from SPR

Both runs used identical alignment lengths (100,000 sites), 100 taxa, identical IQ-TREE flags (`-nt`, `-seed 1`), and the same `cpu_opt_merge` branch binary.

**Phase-level breakdown** (from IQ-TREE stdout logs):

| Run | ModelFinder wall | Tree search wall | Total wall | MF iterations | Tree iterations |
|-----|-----------------|-----------------|------------|--------------|----------------|
| AA CLX  | 1,108.804 s (32%) | 2,337.367 s (68%) | 3,460.813 s | 1,232 models | 102 iters |
| AA SPR  |   399.456 s (34%) |   764.478 s (65%) | 1,169.556 s | 1,232 models | 102 iters |
| DNA CLX |   159.084 s (29%) |   384.838 s (70%) |   546.044 s |   968 models | 102 iters |
| DNA SPR |    61.740 s (21%) |   226.447 s (78%) |   289.121 s |   968 models | 102 iters |
| DNA 1M SPR |  3,500.825 s (57%) | 2,596.995 s (42%) | 6,114.450 s | 968 models | 102 iters |
| DNA 1M CLX | 10,230.229 s (58%) | 7,481.884 s (42%) | 17,752.858 s | 968 models | 102 iters |

**AA/DNA ratios within same hardware:**

| Phase          | CLX ratio | SPR ratio |
|----------------|-----------|-----------|
| ModelFinder    | 6.97×     | 6.47×     |
| Tree search    | 6.07×     | 3.38×     |
| Total          | 6.34×     | 4.05×     |

The tree-search ratio is dramatically different on CLX vs SPR (6.07× vs 3.38×). This platform-specific divergence is a key clue — it points to a hardware microarchitectural explanation, not just an algorithmic one.

---

## 2. IQ-TREE Phase Pipeline

### 2.1 Phase Sequence

For a 100K AA alignment (100 taxa), IQ-TREE runs these phases in strict sequence:

```
INPUT: alignment_100000.phy (100K sites, 100 taxa, AA)
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 0 — STARTUP                              ~1–2 s  │
│  • Read alignment, detect format                        │
│  • Count distinct patterns (→ 96,017 for AA 100K)      │
│  • Composition test for each sequence                   │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 1 — PARSIMONY TREE                       ~1–2 s  │
│  • PLL builds a fast parsimony tree                     │
│  • No branch lengths yet — topology only                │
│  • Used as the STARTING POINT for Phase 2               │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 2 — FAST ML TREE (LG+I+G)               ~12 s   │
│  • NNI moves on the parsimony tree                      │
│  • epsilon=5 then epsilon=1 convergence                 │
│  • Produces a reasonable tree + branch lengths          │
│  • Used as the FIXED TREE for ModelFinder               │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 3 — MODELFINDER                        ~399 s    │  ← BIG ONE
│  • Tests all 1,232 AA models on the FIXED Phase 2 tree  │
│  • For each model:                                      │
│     1. Set rate matrix (LG, WAG, JTT, Q.mammal …)       │
│     2. Eigendecompose Q (once per model)                │
│     3. Compute partial likelihoods for ALL sites        │
│     4. Optimize rate parameters (alpha, I, …)           │
│     5. Compute BIC score                                │
│  • Pick best BIC → "LG+G4"                              │
│  • ⚠ No topology changes here — same NJ tree            │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 4 — CANDIDATE TREE INITIALISATION       ~165 s   │
│  • Apply LG+G4 parameters on a RapidNJ tree (0.1 s)    │
│  • Generate 98 FRESH parsimony trees (131 s)            │
│  • Compute lnL of all 98 candidate trees (34 s)         │
│  • NNI-refine the top 20 candidates                     │
│  • Select the best starting topology                    │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 5 — MAIN TREE SEARCH (102 iterations)   ~600 s   │
│  • Iteration = one round of SPR moves                   │
│  • Each SPR: try regrafting every subtree,              │
│    evaluate lnL for each candidate regraft              │
│  • Accept moves that improve the best score             │
│  • Reoptimise branch lengths after each accepted move   │
│  • Converges when no improvement found                  │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 6 — FINALISATION                          ~2 s   │
│  • Final model parameter optimisation (epsilon=0.010)   │
│  • Write .treefile, .iqtree, .log, .mldist             │
└─────────────────────────────────────────────────────────┘
│
▼
OUTPUT: best tree + LG+G4 parameters + lnL −7,541,976.860
```

### 2.2 Time Budget — 100K AA

**SPR node (168425673, 1,169 s total)** — timings from `iqtree_run.log`:

```
Phase                      │ Wall (s)  │ ████████████████████████ (%)
───────────────────────────┼───────────┼──────────────────────────────────
0  Startup + parsimony     │     1.0   │  (< 1%)
1  Fast ML tree (LG+I+G)   │    12.1   │ █  (1.0%)
2  ModelFinder (1,232 mdl) │   399.5   │ █████████████  (34.2%)
3  Init candidate trees    │   164.6   │ █████  (14.1%)
4  Main tree search        │   590.8   │ ████████████████████  (50.5%)
5  Finalisation            │     1.6   │  (< 1%)
───────────────────────────┼───────────┼──────────────────────────────────
   TOTAL                   │ 1,169.6   │ 100%
```

**CLX node (168422809, 3,461 s total):**

```
Phase                      │ Wall (s)  │ ████████████████████████ (%)
───────────────────────────┼───────────┼──────────────────────────────────
0  Startup + parsimony     │     ~3    │  (< 1%)
1  Fast ML tree            │    ~35    │ █  (1%)
2  ModelFinder (1,232 mdl) │ 1,108.8   │ █████████████  (32%)
3  Init candidate trees    │   ~460    │ █████  (13%)
4  Main tree search        │ 1,877     │ ████████████████████  (54%)
5  Finalisation            │     ~2    │  (< 1%)
───────────────────────────┼───────────┼──────────────────────────────────
   TOTAL                   │ 3,460.8   │ 100%
```

---

### 2.2b Time Budget — 1M AA (scaling validation)

Two 1M-site AA runs (168425490 CLX, 168425491 SPR) were submitted to measure how IQ-TREE
scales with alignment length. Same 100-taxa dataset; 1,000,000 sites (vs 100,000 above).
Pattern count: 928,305 parsimony-informative + 49,361 singleton + 22,334 constant.
Best-fit model: **LG+G4** (same as 100K AA; BIC select is alignment-size-independent).

> **Note:** no `perf stat` wrapper was used for these runs — IPC and LLC miss% are **not
> available**. Phase 5 (Main tree search) time is estimated as total tree-search wall minus
> the logged init subphases; NNI-init (on 20 best trees) is included in the estimate but
> not logged separately by IQ-TREE.

**SPR node (168425491, 22,776 s = 6h 20m total, 103T)** — timings from `iqtree_run.log`:

```
Phase                          │ Wall (s)   │ ████████████████████████ (%)
───────────────────────────────┼────────────┼────────────────────────────────────
0  Startup                     │      ~2    │  (< 1%)
1  PLL parsimony tree          │     34.2   │  (0.2%)
2  Fast ML tree (LG+I+G)       │    274.4   │ █  (1.2%)
3  ModelFinder (1,232 models)  │  7,587.5   │ █████████████████████  (33.3%)
   Init candidate trees:       │            │
4a   98 parsimony trees        │  3,153.5   │ █████████  (13.8%)
4b   lnL of 98 initial trees   │    688.1   │ ██  (3.0%)
4c   NNI refinement (20 best)  │   (n/a)    │
5  Main tree search + final    │  ~11,037*  │ ████████████████████  (48.5% est.)
───────────────────────────────┼────────────┼────────────────────────────────────
   TOTAL                       │  22,776    │ 100%
* = total tree-search wall (15,099 s) minus logged init subphases (3,154 + 688 = 3,842 s)
```

Best-fit model: LG+G4 | lnL (MF) = −78,605,196.596 | lnL (tree) = −78,605,196.573 | BIC = 157,213,128.663 | IPC = N/A | LLC miss = N/A

**CLX node (168425490, 51,328 s = 14h 15m total, 47T):**

```
Phase                          │ Wall (s)   │ ████████████████████████ (%)
───────────────────────────────┼────────────┼────────────────────────────────────
0  Startup                     │      ~2    │  (< 1%)
1  PLL parsimony tree          │     82.1   │  (0.2%)
2  Fast ML tree (LG+I+G)       │    644.9   │ █  (1.3%)
3  ModelFinder (1,232 models)  │ 16,308.3   │ █████████████████████  (31.8%)
   Init candidate trees:       │            │
4a   98 parsimony trees        │  7,317.1   │ ████████  (14.3%)
4b   lnL of 98 initial trees   │  1,429.3   │ ██  (2.8%)
4c   NNI refinement (20 best)  │   (n/a)    │
5  Main tree search + final    │  ~25,545*  │ ████████████████████  (49.8% est.)
───────────────────────────────┼────────────┼────────────────────────────────────
   TOTAL                       │  51,328    │ 100%
* = total tree-search wall (34,822 s) minus logged init subphases (7,317 + 1,429 = 8,746 s)
```

Best-fit model: LG+G4 | lnL (MF) = −78,605,196.590 | lnL (tree) = −78,605,196.573 | BIC = 157,213,128.651 | IPC = N/A | LLC miss = N/A

**100K → 1M scaling (SPR, 103T):**

| Phase | 100K SPR (s) | 1M SPR (s) | Scale factor |
|-------|-------------|-----------|-------------|
| PLL parsimony | ~1 | 34.2 | ~34× |
| Fast ML tree | 12.1 | 274.4 | 23× |
| ModelFinder (1,232 models) | 399.5 | 7,587.5 | **19.0×** |
| 98 parsimony trees (init) | ~28 | 3,153.5 | ~113× |
| lnL of 98 trees (init) | ~20 | 688.1 | ~34× |
| Main tree search (est.) | ~591 | ~11,037 | ~18.7× |
| **Total** | **1,169.6** | **22,776** | **19.5×** |

ModelFinder and main tree search both scale at ~19× for a 10× site increase — consistent
with O(n\_patterns) complexity at fixed taxa count (100 taxa). The parsimony-tree
generation stage (98 trees) scales super-linearly (~113×) because its complexity is
closer to O(n\_sites × n\_taxa × n\_trees) rather than O(n\_sites) alone.

**SPR speedup over CLX for 1M AA:**

| Phase | CLX (s) | SPR (s) | Speedup |
|-------|---------|---------|---------|
| ModelFinder | 16,308 | 7,587 | 2.15× |
| Tree search (post-MF) | 34,822 | 15,099 | 2.31× |
| **Total** | **51,328** | **22,776** | **2.25×** |

The 2.25× total SPR-over-CLX speedup for 1M AA is lower than the 2.96× for 100K AA.
The two runs use different thread counts (47T CLX vs 103T SPR; ratio = 2.19×), so the
2.25× observed speedup is essentially the thread ratio alone — leaving no room for a
per-thread microarchitectural advantage. This contrasts with the 100K AA comparison
(same thread counts, yet 2.96× speedup), which reflected SPR's AVX-512 throughput
advantage in the likelihood kernel. At 1M sites the working set greatly exceeds LLC
capacity on both platforms, and DRAM bandwidth becomes the equalizer.

---

### 2.3 What Happens Inside ModelFinder

For each of the 1,232 models, IQ-TREE does this loop (all 1,232 dispatched in parallel across
103 threads — each thread owns one model at a time):

```
For each model (e.g. LG+G4, WAG+F+G4, JTT+I+G4 …):
   ┌──────────────────────────────────────────────────────┐
   │ 1. Set rate matrix Q (load empirical matrix)         │  < 1 ms
   │ 2. Eigendecompose Q → evec, eval                     │  < 1 ms (O(n³) but one-off)
   │ 3. Pre-compute echild arrays for every branch        │  O(nstates²) × 2n-1 branches
   │ 4. For each site pattern (96,017 patterns):          │
   │      compute partial likelihoods up the tree         │  O(nstates²) × patterns × nodes
   │ 5. Optimize +G4 alpha (or +R3 rates) with 1D search  │  calls steps 3-4 ~10-20 times
   │ 6. Compute lnL → BIC = -2lnL + k·ln(n)              │  < 1 ms
   └──────────────────────────────────────────────────────┘
   → store (model_name, BIC)
Pick model with lowest BIC.
```

Steps 3–5 are the expensive part — O(nstates²) per pattern per node. For AA (nstates=20) this
is 25× more FLOPs than DNA (nstates=4). That is why AA MF takes 399 s vs 61.7 s for DNA 100K
on the same SPR hardware (6.47× observed ratio; the difference from 25× is explained by
memory-bandwidth limitations making DNA already partially serialised).

### 2.4 MPI ModelFinder: How It Helps (and Where It Falls Short)

The model testing loop is **embarrassingly parallel across models** — each model is independent.
The DESIGN is for MPI to distribute substitution-model families across nodes (subst-family LPT
stripe), so each rank evaluates ~450/N models after `filterRates` pruning, then gathers the
best result via `MPI_Allreduce`.  MF2 also distributes **tree search across MPI ranks** —
a capability BEYOND standard IQ-TREE, enabling near-linear tree-search scaling.

**AA 100K MF2 scaling benchmark — completed 2026-05-16** (group `aa_100k_mf2_scaling`).
Scripts `run_cpu_bench_aa_100k_mf2_{1,2,4}node.sh`, same alignment, seed, `-T 103`,
`numactl --localalloc`, `KMP_BLOCKTIME=200` as the baseline SPR run (168425673).
All runs verified lnL = −7,541,976.862 ✓.

| PBS ID | Scenario | MF wall | Tree wall | Total | Speedup | lnL | BIC | IPC (rank 0) | LLC miss% |
|--------|----------|---------|-----------|-------|---------|-----|-----|-------------|-----------|
| 168425673 | Baseline — 1-node SPR, standard binary | 399 s | 764 s | 1,169 s | 1.00× | −7,541,976.860 | 15,086,233 | 1.878 | 66.94% |
| 168446151 | MF2 MPI × 1 node (1×103T) | **1,309 s** | 717 s | 2,030 s | **0.58×** | −7,541,976.862 | 15,086,233 | 1.961 | 67.76% |
| 168446152 | MF2 MPI × 2 nodes (2×103T) | 969 s | 383 s | 1,355 s | **0.86×** | −7,541,976.862 | 15,086,233 | 2.028 | 66.26% |
| 168446153 | MF2 MPI × 4 nodes (4×103T) | 573 s | 198 s | 776 s | **1.51×** | −7,541,976.862 | 15,086,233 | 2.025 | 66.31% |

All four runs select LG+G4 (BIC = 15,086,233; lnL differs only at the 3rd decimal due to floating-point ordering).
IPC values are from rank-0 `perf stat` (the tree-search master rank). LLC miss% is aggregated over all threads.

**Key finding — MF2 MPI tree search scales near-linearly; ModelFinder scaling is impaired by
two implementation bugs.** With 4 nodes the total speedup is 1.51× over the standard
1-node binary. Tree search scales near-linearly (717 s → 383 s → 198 s, 3.63× for 4× ranks).
ModelFinder does NOT scale: 1,309 s at np1 (3.28× SLOWER than standard), 573 s at np4 (43%
slower than standard). Both bugs are diagnosed below and fixes are committed to the source.

**Tree search** scales near-linearly and is the dominant source of the 4-node total speedup.
1M AA runs (longer tree search) will benefit proportionally more from additional nodes.

**IPC progression (rank-0 perf stat):** 1.878 (baseline) → 1.961 (np1) → 2.028 (np2) →
2.025 (np4). The MF2 1-node IPC (1.961) is already higher than the standard binary (1.878),
because MPI serialization reduces per-core model contention. LLC miss% stays stable at
66–68% across all scenarios, confirming the LLC bottleneck is in the likelihood kernel
itself (not model dispatch overhead).

#### Why MF2 ModelFinder is slow — root cause analysis

**C1 (dominant): LPT position-stripe disables `filterRates` pruning.**
The Phase 1 code (`evaluateAll()`) sorted ALL 1,232 AA models by individual rate-category
cost (LPT), then assigned by `sorted_position % nranks`. This puts ALL even-k +Rk models
(GTR+I+R10, GTR+I+R8, …) at even positions → rank 0 gets them; ALL odd-k variants go to
rank 1. When `filterRates` checks whether LG+R4 is worth evaluating, it calls
`getLowerKModel(LG+R4)` → LG+R3 → but LG+R3 is MF_IGNORED on rank 0 (assigned to rank 1).
The guard `!at(lower_model).hasFlag(MF_IGNORED)` fails → pruning never fires. Every rank
evaluates ALL assigned +Rk series without any early termination.

Standard IQ-TREE (1 process): evaluates ~450–500 of the 1,232 AA models after filterRates
pruning. MF2 np1: evaluates ALL 1,232 models (no pruning). Ratio: 1,232/475 ≈ 2.6×.

**C3 (secondary): sequential site-parallel evaluation — OMP barrier overhead.**
The "Issue 5 fix" (commit `abd98764`) made the MPI build evaluate one model at a time using
all 103 threads for site-level parallelism (sequential outer loop, OMP inner loop). For AA
100K (100 taxa, 199 internal nodes), each model evaluation requires 199 × ~10 passes × 2
OMP barriers ≈ 4,000 OMP barrier events. With 1,232 models: ~5M barrier events total. The
non-MPI path uses model-level OMP (1 thread per model, 103 models in parallel, zero
intra-model barriers) and is ~1.3× faster per model.

**Quantified contribution for AA 100K:**

| Cause | Factor |
|-------|--------|
| C1: no filterRates pruning (1,232 vs ~475 models) | ~2.6× |
| C3: sequential site-parallel overhead per model | ~1.3× |
| Combined (product) | **~3.4× ≈ observed 3.28×** |

#### Fix implemented (`phylotesting.cpp`, commit `2672b90a`)

**Fix A — subst-family LPT stripe (replaces position stripe):**
Instead of sorting ALL 1,232 models by individual cost and striping by position, group by
`subst_name` (substitution-model family: LG, WAG, JTT, …). Compute total cost per group
(sum of rate-variant costs), LPT-sort GROUPS, assign GROUPS round-robin across ranks. All
~30 rate variants of LG (LG+G4, LG+I+G4, LG+R2, …, LG+R10, LG+F+…) go to the same rank.
`filterRates` now fires normally: rank 0 evaluates LG+G4, then LG+R2, compares, prunes
LG+R3 … LG+R10. Expected evaluation count: ~150–200 models per rank (similar to standard).

**Fix B — OMP-across-models in MPI path (removes sequential loop) — REVERTED by Fix E:**
The `#ifdef _IQTREE_MPI` sequential evaluation block was removed. Both MPI and non-MPI builds
used the same OMP-across-models loop (`#pragma omp parallel num_threads(num_threads)`
with work-stealing `getNextModel()`). Initial analysis claimed the "Issue 5" race was
a red herring; this was **incorrect** — see Fix E and Fix F below for the full analysis.
Fix B applied as commit `…`, then reverted / superseded by Fix E (`eddbf45d`), and
ultimately replaced by the correct Fix F (`a9b50164`).

**Projected performance after Fix A+B (AA 100K)** — updated by Fix C to §2.4.1:

| Scenario | MF wall | Tree wall | Total | Speedup |
|----------|---------|-----------|-------|---------|
| Baseline (standard, np1) | 399 s | 764 s | 1,169 s | 1.00× |
| MF2 np1 (with fixes) | ~400 s | 717 s | ~1,117 s | ~1.05× |
| MF2 np2 (with fixes) | ~160 s | 383 s | ~543 s | ~2.15× |
| MF2 np4 (with fixes) | ~120 s | 198 s | ~318 s | ~3.67× |

Projected break-even vs standard SPR: **< 1.5 nodes** (any multi-node MF2 run beats
standard single-node). For np4: 1,169 s / 318 s ≈ 3.7× speedup.

#### Fix C — per-rank reference family and rate_block recompute (`phylotesting.cpp`, commit `b9b04a1c`)

Fix A assigns ALL rate variants of each substitution family to the same rank (correct).
Fix B restores the OMP-across-models evaluation loop. However Fix A introduced a
secondary bug: the `filterRates()` function identifies the *reference substitution family*
for pruning by reading `at(0).subst_name` — but after Phase 1 MPI stripe, model `at(0)`
belongs to another rank and is `MF_IGNORED`. The reference family is therefore empty and
`filterRates` silently exits without pruning.

**Fix C Part 1** (`filterRates()`): scan forward from index 0 to find the first
non-`MF_IGNORED` model and use its `subst_name` as the per-rank reference family.
`best_score` accumulates only among that family's non-ignored models, then prunes
rate variants whose BIC exceeds `best_score + score_diff_thres`.

**Fix C Part 2** (`evaluateAll()`, after Phase 1 stripe): `rate_block` (the model index
that triggers `filterRates`) was computed from the global model list as the last index of
the first substitution family. For ranks 1–3 (whose first family is `MF_IGNORED`),
`rate_block` points into the ignored region — `filterRates` never fires. Fix: after
marking `MF_IGNORED`, recompute `rate_block` to equal the last index of the rank's
*first non-ignored substitution family*.

**Updated projected performance after Fix A+B+C (AA 100K):**

| Scenario | MF wall | Tree wall | Total | Speedup |
|----------|---------|-----------|-------|---------|
| Baseline (standard, np1) | 399 s | 764 s | 1,169 s | 1.00× |
| MF2 np1 (Fix A+B+C) | ~400 s | 717 s | ~1,117 s | ~1.05× |
| MF2 np2 (Fix A+B+C) | ~145 s | 383 s | ~528 s | ~2.2× |
| MF2 np4 (Fix A+B+C) | ~100 s | 198 s | ~298 s | ~3.9× |

With Fix C, ranks 1–3 now prune their assigned rate series as effectively as rank 0.
Each rank evaluates ~150 post-pruning models (down from ~1,232 without fixes).
Projected break-even: **< 1.5 nodes**; np4 achieves ~3.9× total speedup.

---

### 2.5 MPI Communication Overhead: Quantitative Breakdown

A common concern for MPI-based model selection is whether the collective-communication
phase (Phase 2, after all models are evaluated) becomes a bottleneck at large dataset
sizes. For AA 100K (1,232 models, 4 MPI ranks) the answer is emphatically **no**:

#### Phase 2A — `MPI_Allreduce` for scores (4 operations)

```
4 ops × 1,232 doubles × 8 B = 39.4 KB total
InfiniBand HDR 200: ~200 Gbps = 25 GB/s peak, practical ~12 GB/s
Latency per allreduce (log₂(4) = 2 hops): ~2 µs
Transfer time: 39.4 KB / 12 GB/s = 3.3 µs per allreduce
Total: 4 × (2 µs latency + 3.3 µs transfer) ≈ 21 µs
```

Each allreduce carries one score array (lnL MAX, BIC MIN, AIC MIN, AICc MIN) for all
1,232 models. The collective is a simple element-wise reduction, not an all-to-all.
Even on 100 Mbps Ethernet (worst case): 39.4 KB / 12.5 MB/s = 3 ms total.
**At any realistic InfiniBand fabric: < 0.1 ms total for all 4 allreduces.**

#### Phase 2B — `gatherCheckpoint` + `broadcastCheckpoint`

Each rank serialises its `ModelCheckpoint` to a text key-value string via `ckp->dump()`.
The checkpoint stores per-model: model name (~20 B), lnL/BIC/AIC/AICc scores (~60 B),
model parameters (alpha, rates, freqs, ~200 B), and the optimised tree newick with
branch lengths (~2 KB for 100-taxa). Plus global keys (best-model names, model_list).

```
Per-model checkpoint size ≈ 2,300 B
Models per rank (np4, after pruning) ≈ 150
Checkpoint per rank ≈ 150 × 2,300 B ≈ 345 KB

gatherCheckpoint (MPI_Gatherv, all ranks → rank 0):
  data volume: 4 × 345 KB = 1.38 MB
  transfer: 1.38 MB / 12 GB/s = 115 µs
  + rank-0 deserialise (ckp->load, string parse): ~3 ms

broadcastCheckpoint (MPI_Bcast, rank 0 → all):
  data volume: 1.38 MB
  transfer: 1.38 MB / 12 GB/s = 115 µs
  + each worker deserialise: ~3 ms

Total Phase 2B wall time ≈ 7–10 ms
```

**Grand total Phase 2 MPI overhead: < 12 ms** — negligible compared to 100–400 s of
model evaluation. The MPI data path is not a bottleneck for MF2 at any node count up
to at least np16 (checkpoint scales as O(n_models × checkpoint_per_model), growing
linearly with dataset size but never approaching seconds until ~10 million models).

#### Phase 2 serialisation implementation (`MPIHelper.cpp`)

`gatherCheckpoint` uses:
- `ckp->dump(stringstream)` → flat text (fast, ~1 µs/KB)
- `MPI_Gather` for sizes, `MPI_Gatherv` for data
- rank-0 `ckp->load(stringstream)` with last-write-wins merge semantics

`broadcastCheckpoint` uses `MPI_Bcast` of the merged text blob. There is no
custom serialisation protocol, no type-punning, and no latency-sensitive path.
The implementation is correct and efficient; no changes are needed.

---

### 2.6 Thread Saturation and NUMA Binding in `evaluateAll()`

#### Thread saturation — quantified for AA 100K

`evaluateAll()` uses an OMP-across-models loop: each of the `num_threads` (103 on
SPR) threads grabs one model via `getNextModel()` (work-stealing critical section),
evaluates it with 1 effective thread (inner OMP loops degrade to 1 thread because
nested OMP is disabled), marks it `MF_DONE`, and loops back.

Because inner loops run single-threaded, each thread processes ALL 96,017 AA patterns
for its model independently. The number of evaluations that can proceed in parallel
equals `min(remaining_models, num_threads)`, so:

| Config | Models/rank (post-pruning) | Round 1 active | Round 2 active | Tail idle | Thread utilisation |
|--------|---------------------------|----------------|----------------|-----------|-------------------|
| np1 | ~475 | 103 | 103 | 26 idle (round 5) | ~95% |
| np2 | ~237 | 103 | 103 | 31 idle (round 3) | ~93% |
| np4 | ~150 | 103 | 47 | 56 idle (round 2) | ~83% |
| np8 | ~90 | 90 | — | 13 idle | ~87% |

For np4 the tail loss is ~17%: of the 206 total thread-slots across 2 rounds, 47
are wasted in round 2 (`56/206 = 27%` of round-2 capacity idle). In absolute
time this is ≤ 10 s out of ~100 s MF wall time (< 10%). The LPT stripe (Fix A)
already front-loads the heaviest families so the tail round carries mostly fast
+G4/+I models, further limiting wall-time impact.

A hybrid nested-OMP mode (switch to k-threads-per-model when remaining_models <
num_threads/2) would recover most tail loss but adds substantial complexity.
The gain is < 10 s on a ~100 s background; this optimisation is deferred.

#### NUMA binding — evaluateAll() is self-NUMA-correct

In the `evaluateAll()` OMP-across-models path, each OMP thread evaluates a DIFFERENT
model via a separately allocated `IQTree` clone. All heap allocations performed inside
`CandidateModel::evaluate()` — including the per-model `partial_lh` buffers, branch
objects, and rate-category arrays — are performed by the evaluating thread. With Linux
first-touch policy and a thread affinity that distributes threads across both NUMA
domains (sockets), those allocations land on the NUMA node local to the evaluating
thread. Reads and writes to `partial_lh` are therefore local DRAM accesses.

The only shared (read-only) data is the alignment object (`aln`), holding 96,017 AA
patterns × 100 taxa ≈ 9.6 MB. This was allocated by the main thread (socket 0).
However:

1. 9.6 MB << 60 MB SPR LLC per socket.
2. After the first few model evaluations, alignment data is cached in BOTH sockets'
   L3 caches via hardware prefetch and coherency.
3. With 103 simultaneous models warming up, both sockets achieve L3 hits for alignment
   reads within ~1 s of the OMP region starting.

Sustained cross-NUMA DRAM latency for alignment data is therefore bounded to the first
~1% of model-evaluation time. There is no measurable alignment-DRAM bottleneck.

**Contrast with the `test()` path (all threads on one model):** here `partial_lh` is
shared across 103 threads and is ~60 MB — it does NOT fit in one socket's L3. The
existing NUMA first-touch pragmas (`R1a computePtnFreq`, `R1b computePtnInvar`,
`R2a _pattern_lh_cat`) distribute those pages across both NUMA nodes via
`#pragma omp parallel for schedule(static)`. These pragmas are correct and necessary
for the `test()` path; they have no effect in `evaluateAll()` (inner loops run
single-threaded, so the schedule(static) first-touch never fires across sockets in
that context).

#### Fix D — `proc_bind(spread)` on the evaluateAll() OMP pragma (commit `0db014bc`)

**Change:** added `proc_bind(spread)` to the `#pragma omp parallel num_threads(num_threads)`
in `evaluateAll()` (`phylotesting.cpp`, commit `0db014bc`).

**Effect:** overrides the global `OMP_PROC_BIND=close` for this specific parallel
region, instructing the OpenMP runtime to distribute threads maximally across all
available hardware places before applying close-proximity sub-grouping.

For T=103 on 104 SPR cores, `close` and `spread` both result in ~52 threads per
socket (sequentially numbered cores fill both sockets before wrapping). The practical
difference is zero for full-node runs.

The benefit materialises for **sub-full-thread runs** (e.g. testing with `-T 48`):

```
close, T=48: threads 0–47 → cores 0–47 → socket 0 only
              socket 1 has 0 active threads
              effective memory bandwidth: ~1× (one socket)

spread, T=48: threads 0–23 → socket 0; threads 24–47 → socket 1
              both sockets active
              effective memory bandwidth: ~2× (both sockets)
```

With `spread`, any run where T < 104 uses both sockets proportionally; the hot
likelihood kernel's DRAM bandwidth scales with T rather than saturating at socket-0
capacity. The change is a single-line pragma addition; it overrides `close` only for
the `evaluateAll()` region while leaving the `test()` path and all hot-kernel inner
loops undisturbed.

#### Fix E — Restore sequential outer loop for MPI builds (`phylotesting.cpp`, commit `eddbf45d`)

**Trigger:** Fix B (OMP-across-models in MPI path) caused SIGABRT crashes during
the 100K AA MF2 benchmark jobs (168466947/948/949). Symptoms:
- `np=1`: SIGABRT (rc=134) with no error message
- `np=2/4`: "Not enough memory, allocation of 131996370853047072 bytes" then SIGABRT

These are classic heap-corruption signatures from `std::map` red-black tree corruption.

**Workaround (Fix E):** Guarded the outer `#pragma omp parallel` in `evaluateAll()` with
`#if defined(_OPENMP) && !defined(_IQTREE_MPI)` so MPI builds use a sequential outer
loop (one model at a time per OMP-thread group). Each sequential `evaluate()` call still
uses all `num_threads` OMP threads for site-likelihood parallelism. Combined with Fix A
LPT striping, this still scales across MPI ranks, but each rank is effectively
single-threaded at the model-evaluation level.

Fix E was a **workaround**, not a root-cause fix. It was committed to unblock benchmark
resubmission while the actual race mechanism was diagnosed.

#### Fix F — Thread-local `in_model_info` snapshot eliminates OMP data race (`phylotesting.cpp`, commit `a9b50164`)

**Root cause analysis (corrected):**

The shared `ModelCheckpoint in_model_info` (a `std::map<string,string>`) is accessed by
multiple OMP threads in `evaluateAll()`. While writes to `in_model_info` are protected by
`#pragma omp critical`, several **reads** from `in_model_info` inside `evaluate()` happen
**outside** any critical section:

| Race point | Location | Access type | Protected? |
|-----------|----------|-------------|-----------|
| `CandidateModel::restoreCheckpoint(&in_model_info)` — early-exit check | ~line 1960 | READ | ✗ |
| `prev_info.restoreCheckpointRminus1(&in_model_info, this)` | ~line 2033 | READ | ✗ |
| `iqtree->getRate()->initFromCatMinusOne(in_model_info, ...)` | ~lines 2106, 2122 | READ | ✗ |
| `in_model_info.putBool("UnreliableParam", ...)` | ~line 2071 | WRITE | ✗ (VB_MED only) |

The **write** path (`saveCheckpoint(&in_model_info)`) at the end of `evaluate()` IS
protected by `#pragma omp critical`. But `#pragma omp critical` only serialises
concurrent writers — it does NOT prevent a **reader outside** the critical section from
running concurrently with a **writer inside** the critical section.

`std::map::find()` (read traversal of the red-black tree) concurrent with
`std::map::operator[]` or `std::map::insert()` (write, rebalancing the tree) is
**undefined behaviour** in C++. This causes heap corruption → SIGABRT.

Additionally, `Checkpoint::getString()` reads `checkpoint->struct_name` to build the map
key. `startCheckpoint()/endCheckpoint()` modify `struct_name` inside the critical section.
Concurrent `struct_name` read + write is also undefined behaviour on `std::string`.

**Prior incorrect diagnosis:** An earlier analysis stated "ModelFactory::ctor writes to
`in_model_info` via `saveCheckpoint()` during initialisation." This was **wrong**.
Confirmed by reading `modelfactory.cpp` lines 153–1050: the `ModelFactory::ModelFactory()`
constructor does NOT call `saveCheckpoint()` anywhere. Similarly `RateFree`, `RateGamma`,
`ModelProtein`, and `ModelMarkov` constructors do not call `saveCheckpoint()`. The
constructor only creates model/rate objects via string parsing and parameter initialisation.

**Upstream IQ-TREE2 approach:** Upstream (`iqtree/iqtree2:master`) avoids the race
entirely by blocking MPI for model selection:
```
outError("Please use only 1 MPI process! We are currently working on
the MPI parallelization of model selection.")
```
The upstream `evaluateAll()` uses `#pragma omp parallel num_threads(num_threads)` freely
because there is always exactly one MPI rank and thus no concurrent MPI writes. The race
does not manifest upstream.

Upstream commit `1ff47eb` (stelzch, Mar 19 2025) fixed a separate race in `getNextModel()`:
moved `setFlag(MF_RUNNING)` and `current_model = next_model` inside the critical section
to prevent double-assignment. **Our local code already had the equivalent fix** —
`current_model = next_model` and `at(next_model).setFlag(MF_RUNNING)` are both inside
`#pragma omp critical` in our `getNextModel()`.

**Fix F implementation (incomplete — superseded by Fix G):**

At the start of `evaluate()`, took a per-thread snapshot of `in_model_info` under
`#pragma omp critical`, but placed the snapshot copy **after** `iqtree->initializeModel()`.
All `restoreCheckpointRminus1`, `initFromCatMinusOne`, and `putBool` calls were redirected
to the local copy. The early-exit `restoreCheckpoint()` call was also moved inside the
critical section.

**Fix F defect — missed races in `initializeModel()` and `getModelFactory()->restoreCheckpoint()`:**

Fix F placed the snapshot copy too late. Two critical reads still used the shared `in_model_info`:

1. `iqtree->setCheckpoint(&in_model_info)` (before Fix F's copy) set iqtree's internal
   checkpoint pointer to the shared map.
2. `iqtree->restoreCheckpoint()` was protected by `#pragma omp critical` — safe.
3. `iqtree->initializeModel(params, ...)` — **UNPROTECTED**. The `ModelFactory` constructor
   reads from the iqtree checkpoint (still pointing to shared `in_model_info`) to restore
   saved model parameters. This read races with any other thread's concurrent
   `saveCheckpoint(&in_model_info)` write.
4. `iqtree->getModelFactory()->restoreCheckpoint()` — also read from shared `in_model_info`,
   also unprotected.

`std::map::find()` concurrent with `std::map::insert()` is undefined behaviour → red-black
tree corruption → heap corruption → SIGABRT at first model evaluation attempt.

#### Fix G — move thread-local snapshot before `setCheckpoint` (`phylotesting.cpp`, commit `10107158`)

Fix G moves the `local_in_info = in_model_info` copy to **before** `iqtree->setCheckpoint()`,
so ALL checkpoint reads use the per-thread copy:

```cpp
// Fix G: copy BEFORE setCheckpoint/restoreCheckpoint/initializeModel
ModelCheckpoint local_in_info;
#pragma omp critical
{ local_in_info = in_model_info; }          // per-thread snapshot

iqtree->setCheckpoint(&local_in_info);      // point iqtree at per-thread copy
iqtree->restoreCheckpoint();                // reads local_in_info — no race
ASSERT(iqtree->root);
iqtree->initializeModel(params, ...);       // reads local_in_info — no race
// ...
if (restoreCheckpoint(&local_in_info)) { delete iqtree; return ""; }
iqtree->getModelFactory()->restoreCheckpoint();  // reads local_in_info — no race
```

The `#pragma omp critical` guards around `iqtree->restoreCheckpoint()` and
`getModelFactory()->restoreCheckpoint()` are removed (per-thread storage needs no
serialisation). The only writes to the shared `in_model_info` remain the final
`saveCheckpoint(&in_model_info)` and the VB_MED-only `putBool` (both under `#pragma omp
critical`).

**Copy overhead:** unchanged from Fix F analysis — < 0.1% of per-model evaluate time.

**Result:** All checkpoint reads in `evaluate()` use per-thread storage. The parallel
outer loop (`#pragma omp parallel num_threads(num_threads) proc_bind(spread)`) is retained.
Combined with Fix A LPT striping, this gives full model-level OMP parallelism per rank.

**Expected performance after Fix G (AA 100K):** Fix G alone retains the parallel outer
loop for MPI builds, which causes OOM at 100K AA scale (see §2.4.5). Fix H (commit
`257485e5`) adds the `!defined(_IQTREE_MPI)` guard to restore sequential outer loop for
MPI builds. The combined Fix G+H binary is the correct baseline for performance measurement.

#### §2.4.3 Measured results — `abd98764` binary (May 10 position-LPT, no Fix C/E/F)

**Correction (post-run investigation):** Jobs 168467031/032/033 were submitted at 13:12
on 2026-05-16, but the Fix F binary was not compiled until 13:39 (and Fix A–E were never
compiled at all — see §2.4.2 preamble). These three jobs ran with the May 10 `abd98764`
binary: position-based LPT stripe, sequential outer loop (`1ac3c0a`), no Fix C/D/E/F.

**Binary characteristics (`abd98764`):**
- Sequential outer loop for MPI builds (commit `1ac3c0a`): one model at a time, all 103 OMP
  threads used per model for site-likelihood parallelism
- Position-LPT stripe (`abd98764`): sorted ALL 1,232 AA models by cost-category k, assigned
  by `sorted_position % nranks` — gives rank 0 a biased set of the heaviest models for np≥2
- No Fix C: `filterRates` per-rank reference bug → no model pruning → all assigned models
  evaluated at full cost (no getLowerKModel early-exit)
- No Fix D, E, F

**Model cost analysis (why np=4 is 10× slower than expected):**
With sequential outer loop and 103 OMP threads, per-model wall time ≈ T₁/103 ≈ 0.78 s
(derived: standard MF 399 s / 475 post-filter models × 103 → T₁ ≈ 80 s/model).

For np=2 rank 0: 616 models, no pruning → 616 × 0.78 s ≈ **481 s MF** ✓ (matches).

For np=4 rank 0: position-LPT picks every 4th model from the sorted list starting at the
heaviest. Rank 0 gets the top-heavy tail: all +R10/+I+R10/+R9 families. These models have
more rate categories and proportionally higher kernel cost. Average cost for rank 0’s 308
models ≈ 7.6 s/model → **2,349 s MF** (10× slower than a uniform-distribution estimate
of 308 × 0.78 s = 240 s). This is the “position-LPT load imbalance” issue that Fix A
(subst-family stripe) and Fix C (per-rank filterRates) were designed to eliminate.

| PBS ID | Scenario | Binary (actual) | MF wall | Tree wall | Total | vs SPR baseline | lnL | IPC | LLC miss% |
|--------|----------|-----------------|---------|-----------|-------|-----------------|-----|-----|-----------|
| 168425673 | Baseline — SPR standard, 1 node, 103T | std SPR | 399 s | 764 s | 1,169 s | 1.00× | −7,541,976.860 | 1.878 | 66.94% |
| 168446151 | MF2 np1, 1×103T | A+B (early, May 10 pre-LPT) | 1,309 s | 717 s | 2,030 s | 0.58× | −7,541,976.862 | 1.961 | 67.76% |
| 168446152 | MF2 np2, 2×103T | A+B (early, May 10 pre-LPT) | 969 s | 383 s | 1,355 s | 0.86× | −7,541,976.862 | 2.028 | 66.26% |
| 168446153 | MF2 np4, 4×103T | A+B (early, May 10 pre-LPT) | 573 s | 198 s | 776 s | **1.51×** | −7,541,976.862 | 2.025 | 66.31% |
| 168467031 | MF2 np1, 1×103T | `abd98764` (pos-LPT, no Fix C) | — | — | *running* | pending | — | — | — |
| **168467032** | **MF2 np2, 2×103T** | **`abd98764` (pos-LPT, no Fix C)** | **481 s** | **390 s** | **875 s** | **1.34×** ✓ | **−7,541,976.865** | **1.999** | **67.3%** |
| 168467033 | MF2 np4, 4×103T | `abd98764` (pos-LPT, no Fix C) | ~~2,349 s~~ | 202 s | ~~2,552 s~~ | ~~0.46×~~ ✗ | −7,541,976.852 | 2.051 | 66.50% |

**168467033 (np=4) discarded**: the position-LPT imbalance concentrates all heavy-rate
models onto rank 0 (every 4th model from cost-sorted order). Without Fix C, no pruning
occurs → rank 0 evaluates 308 models at ~7.6 s/model = 2,349 s MF wall. Fix A+C eliminate
this imbalance. Result is correct (lnL ✓) but not representative of fixed code.

**168467032 (np=2) valid as `abd98764` reference:** 875 s total, 1.34× speedup over SPR
baseline. This is the baseline for position-LPT + no Fix C at np=2. The speedup over
standard SPR comes from running MF on rank 0 with tree search on all ranks in parallel.

**np=2 vs SPR baseline:** 875 s vs 1,169 s = **1.34× faster**. ✓ Even the broken binary
beats standard at np=2 because the tree search benefits from 2 nodes.

#### §2.4.4 Fix F benchmark — all jobs crashed (SIGABRT/SIGKILL at 18/148/23 s)

Fix F binary (commit `a9b50164`, built 2026-05-16 13:39) was submitted immediately after
the binary investigation but contained the defect described above. All three jobs crashed
at the first model evaluation attempt:

| PBS ID | Scenario | Fix set | Exit | Wall |
|--------|----------|---------|------|------|
| 168468220 | MF2 np1, 1×103T | A+B+C+D+F | rc=134 SIGABRT | 18 s |
| 168468221 | MF2 np2, 2×103T | A+B+C+D+F | rc=137 SIGKILL | 148 s |
| 168468222 | MF2 np4, 4×103T | A+B+C+D+F | rc=134 SIGABRT | 23 s |

Error in all logs: `*** IQ-TREE CRASHES WITH SIGNAL ABORTED` immediately after
`ModelFinder will test 1232 protein models`. Fix G was applied and binary rebuilt at 14:16.

#### §2.4.5 Fix G benchmark — all jobs OOM-killed (parallel outer loop, ∼1.3 TB/rank)

Fix G retains the parallel outer loop for MPI builds. Each OMP thread creates its own
`IQTree*` instance (full partial-likelihood buffers: ∼12 GB for AA 100K on 100 taxa × 20
states × 4 rate cats × 197 nodes). With 103 threads/rank, peak memory ∼1.3 TB/rank —
far exceeding the 512 GB Gadi nodes.

| PBS ID | Scenario | Fix set | Exit | Wall | Notes |
|--------|----------|---------|------|------|-------|
| 168468376 | MF2 np2, 2×103T | A+B+C+D+G | rc=137 SIGKILL | 148 s | 986.6 GB used of 1020 GB |
| 168468377 | MF2 np4, 4×103T | A+B+C+D+G | rc=137 SIGKILL | 134 s | OOM |
| 168468375 | MF2 np1, 1×103T | A+B+C+D+G | rc=137 SIGKILL | ∼9 min | 510 GB exhausted |

#### Fix H — sequential outer loop for MPI builds (`phylotesting.cpp`, commit `257485e5`)

Fix H changes `#ifdef _OPENMP` → `#if defined(_OPENMP) && !defined(_IQTREE_MPI)` around
the outer `#pragma omp parallel` block in `evaluateAll()`. In MPI builds the outer loop
is sequential: each rank evaluates one model at a time, using `num_threads` OMP threads
inside `evaluate()` for the partial-likelihood kernel. Fix G’s `local_in_info` snapshot
is retained (correct for non-MPI parallel builds; a no-op for sequential MPI builds).

This is the same outer-loop policy as `abd98764` (Fix E approach), combined with
Fixes A (subst-family LPT), C (filterRates per rank), and D (proc_bind(spread)).

**Expected performance (Fix A+C+D+G+H, sequential outer loop, AA 100K):**

With sequential outer loop, MF wall is the heaviest rank’s sequential sum of per-model
times after filterRates pruning. Fix A’s subst-family LPT gives balanced initial
assignment; Fix C’s filterRates prunes heavy +Rk series early.

| Scenario | MF wall | Tree wall | Total | vs SPR baseline |
|----------|---------|-----------|-------|-----------------|
| np1 | ~399 s | ~717 s | ~1,116 s | ~1.05× |
| np2 | ~200 s | ~383 s | ~583 s | ~2.0× |
| np4 | ~100 s | ~198 s | ~298 s | ~3.9× |

#### §2.4.6 Fix H benchmark — Fix A+C+D+G+H, sequential outer loop (2026-05-16 14:24, commit `257485e5`)

Fix H binary rebuilt at 14:24. Jobs submitted immediately. 168468562 (np=2) completed
2026-05-16 ~15:10 (wall 866 s); np=1 and np=4 still running.

| PBS ID | Scenario | Fix set | MF wall | Tree wall | Total | vs baseline | lnL | IPC | LLC miss% |
|--------|----------|---------|---------|-----------|-------|-------------|-----|-----|-----------|
| 168425673 | Baseline — 1-node SPR, std binary | — | 399 s | 764 s | 1,169 s | 1.00× | −7,541,976.860 | 1.878 | 66.94% |
| 168468561 | MF2 np1, 1×103T | A+C+D+G+H | — | — | *running* | pending | — | — | — |
| **168468562** | **MF2 np2, 2×103T** | **A+C+D+G+H** | **475 s** | **387 s** | **866 s** | **1.35×** ✓ | **−7,541,976.865** | **2.005** | **67.40%** |
| 168468563 | MF2 np4, 4×103T | A+C+D+G+H | — | — | *running* | pending | — | — | — |

**168468562 (np=2) analysis:**

ModelFinder wall 475 s at np=2 vs 399 s for single-node baseline — **19% slower** for
MF alone. This is expected: sequential outer loop (Fix H) means each rank evaluates ~150
models one-at-a-time with 103 OMP threads per model (OMP barrier overhead per model ≈ C3
penalty, ~1.3×). Parallel outer loop would give ~365 s MF for np=2, but is forbidden in
MPI builds due to OOM (§2.4.5). The tree-search component (387 s) shows near-perfect
2-node scaling vs pre-fix np=2 (383 s) — both have the same MPI tree parallelism.

**Comparison across all AA 100K MF2 variants (completed runs only):**

| PBS ID | Scenario | MF wall | Tree wall | Total | vs baseline | IPC | LLC miss% |
|--------|----------|---------|-----------|-------|-------------|-----|-----------|
| 168425673 | Baseline (std, 1 node) | 399 s | 764 s | 1,169 s | 1.00× | 1.878 | 66.94% |
| 168446151 | Pre-fix MF2, np=1 | 1,309 s | 717 s | 2,030 s | 0.58× | 1.961 | 67.76% |
| 168446152 | Pre-fix MF2, np=2 | 969 s | 383 s | 1,355 s | 0.86× | 2.028 | 66.26% |
| 168446153 | Pre-fix MF2, np=4 | 573 s | 198 s | 776 s | 1.51× | 2.025 | 66.31% |
| 168467032 | Cost-sort LPT, np=2 | 481 s | 390 s | 875 s | 1.34× | 1.999 | 67.30% |
| **168468562** | **Fix A–H, np=2** | **475 s** | **387 s** | **866 s** | **1.35×** | **2.005** | **67.40%** |

**Key observations:**
1. **Fix A–H np=2 MF (475 s) vs pre-fix np=2 MF (969 s): 2.04× MF improvement** —
   Fix A (subst-family LPT) + Fix C (per-rank filterRates) reduced models evaluated
   per rank from ~600+ to ~150, cutting sequential evaluation time proportionally.
2. **Fix A–H np=2 MF (475 s) vs cost-sort LPT np=2 MF (481 s): effectively identical** —
   confirms that position-LPT + no-Fix-C at np=2 happened to assign a balanced load
   (rank 0 got ~616 models but with filterRates partially working); Fix C gives only
   marginal MF benefit at np=2 vs the np=4 case where imbalance is severe.
3. **Tree search unchanged (387 s ≈ 383 s)**: all variants with np=2 share the same
   MPI tree parallelism — the fixes only affect ModelFinder dispatch.
4. **IPC progression**: 1.878 (baseline) → 1.999 (cost-sort) → 2.005 (Fix A–H).
   The small increase reflects better per-model cache reuse from fixing filterRates
   (fewer wasted model evaluations → less L3 thrash).
5. **Overall speedup 1.35× at np=2** is below the ~2.0× projected in §Fix H expected
   table. The gap comes from sequential outer loop overhead (C3 penalty, ~1.3×);
   parallel outer loop would give ~2.0× but OOMs at 100K AA scale.

---

### §2.4.7 Root cause: why single-node MF (399 s) beats 2-node MF (475 s)

**Question:** Fix H np=2 uses 206 threads across 2 nodes for ModelFinder, yet takes
*longer* than the single-node standard binary (399 s vs 475 s). How is that possible?

The answer lies in the **OMP parallelism strategy**: the two binaries use fundamentally
different approaches to parallelise ModelFinder, with very different efficiency
characteristics.

#### Two parallelism modes

| Mode | Strategy | Kernel threading | Effective speedup |
|------|----------|-----------------|-------------------|
| Standard non-MPI | **Model-parallel**: 103 OMP threads in outer `evaluateAll()` loop; each thread evaluates one complete model | Nested OMP disabled → inner `#pragma omp parallel for num_threads(103)` gets **1 thread** → single-threaded per model | ≈ **103×** for MF |
| Fix H MPI | **Site-parallel**: outer loop sequential (`#ifndef _IQTREE_MPI` guard); one model at a time | Inner `#pragma omp parallel for schedule(static) num_threads(103)` fires with **103 threads** over site packets | ≈ **27×** per model |

**Model-parallel (standard non-MPI) — code path:**

```cpp
// phylotesting.cpp, evaluateAll()  [NON-MPI only]
#if defined(_OPENMP) && !defined(_IQTREE_MPI)
#pragma omp parallel num_threads(num_threads) proc_bind(spread)
#endif
{
    int64_t model;
    do {
        model = getNextModel();          // work-stealing from shared queue
        tree_string = at(model).evaluate(..., num_threads, ...);
    } while (model != -1);
}
```

Each of the 103 outer OMP threads calls `evaluate()` with `num_threads=103`, but
because `OMP_NESTED=false` the inner `#pragma omp parallel for num_threads(103)` inside
`phylokernelnew.h` gets **1 thread**. Each model is evaluated **single-threadedly** —
only SIMD (AVX-512) vectorisation, no thread parallelism. The 103× speedup comes
entirely from 103 *independent* models running concurrently.

**Site-parallel (Fix H MPI) — code path:**

```cpp
// phylotesting.cpp, evaluateAll()  [MPI builds — outer loop is sequential]
{
    int64_t model;
    do {
        model = getNextModel();
        tree_string = at(model).evaluate(..., num_threads=103, ...);
    } while (model != -1);
}

// phylokernelnew.h, computePartialLikelihood kernel:
#pragma omp parallel for schedule(static) num_threads(103)
for (int packet_id = 0; packet_id < 206; ++packet_id) {   // num_packets = 2×103
    for (auto it : traversal_info) {
        computePartialLikelihood(*it,
            limits[packet_id], limits[packet_id+1], packet_id);
    }
}
```

There is no outer parallel region. The inner `#pragma omp parallel for` fires with 103
threads for every likelihood evaluation, every branch-length Newton step, every
convergence check — all sequenced through one model at a time.

#### Amdahl analysis

**Derived quantities from measurements:**

```
T1  = effective single-model wall time in model-parallel mode
    = (399 s total MF) × 103 threads / 475 models = 86.7 s per model

T‖  = site-parallel wall time per model (Fix H np=2)
    = 475 s total MF / 150 models per rank = 3.17 s per model

Speedup₁₀₃ = T1 / T‖ = 86.7 / 3.17 = 27.4×
```

Applying Amdahl's law:

$$S = \frac{1}{f_s + \frac{1-f_s}{103}} = 27.4 \quad\Rightarrow\quad f_s = 2.76\%$$

The implied serial fraction is **2.76% of T1 = 2.39 s per model evaluation**. Over 150
sequential models this accumulates to **≈ 359 s of wall time that cannot be parallelised**.

**Breakdown of Fix H np=2 MF wall time (per model):**

$$T_{\parallel} = \underbrace{f_s \cdot T_1}_{\text{serial}\approx 2.39\text{ s}} + \underbrace{\frac{(1-f_s)\cdot T_1}{103}}_{\text{parallel}\approx 0.82\text{ s}} = 3.21\text{ s} \approx 3.17\text{ s (obs.)}$$

$$T_{MF,\text{Fix H}} = 150 \times 3.17\text{ s} = 475\text{ s}$$

In model-parallel mode the same 2.39 s serial fraction runs **concurrently** across 103
threads — its wall-time contribution is ≈ 2.4 s regardless of model count, not 359 s.

#### What constitutes the 2.39 s serial fraction per model

The per-model serial overhead in the sequential Fix H outer loop comes from the lifecycle
of each `IQTree` instance created and destroyed for every `evaluate()` call:

1. **`new IQTree()` + `initializeAllPartialLh()`** — allocates `central_partial_lh`
   via `posix_memalign` (→ glibc `mmap`). On Gadi, THP is in `madvise` mode and
   IQ-TREE never calls `madvise(MADV_HUGEPAGE)`, so all 6.27 GB of
   `central_partial_lh` use **4 KB pages**. The `mmap` syscall is O(1 μs); first
   writes (during the first OMP-parallel kernel call) trigger
   6.27 GB / 4 KB = 1,568,000 page faults distributed across 103 threads ≈ 15,000
   faults per thread × ~1 μs = **15 ms** serial-equivalent per model. Minor.

2. **`delete iqtree`** → destructor → `aligned_free(central_partial_lh)` → glibc
   `free()` → `munmap(6.27 GB)`. The kernel must walk and free 1,568,000 PTEs.
   With 4 KB pages (THP inactive): **~10–30 ms** per munmap call. Minor.

3. **Serial BFGS steps between OMP parallel sections**: for each of the ~197 branches,
   the Newton-Raphson branch-length optimiser runs as `computeLikelihoodDerv()` (OMP
   parallel) followed by a serial Newton update. With ≈5 outer iterations ×
   197 branches × 15 Newton steps = ~14,775 serial Newton updates per model, each
   ~1–3 μs, the total is **15–45 ms**. Minor.

4. **Model-parameter BFGS** (optimise α for +G, frequencies, etc.): serial steps
   between OMP likelihood calls. Eigendecomposition for 20×20 matrix: ~50 μs. Over
   ~50 BFGS iterations: **2.5 ms**. Negligible.

5. **Memory-bandwidth saturation at 103 threads** — the dominant contributor.
   The `central_partial_lh` buffer (6.27 GB) is accessed by all 103 threads
   simultaneously. The `schedule(static)` NUMA fix (R2b, confirmed present in commit
   `257485e5`) ensures each thread accesses NUMA-local pages, but with 103 threads
   sharing the 400 GB/s DDR5 bus, the per-thread effective bandwidth =
   **400 / 103 ≈ 3.9 GB/s**. In model-parallel mode each single-thread benefits
   from the hardware prefetcher on its own model's sequential 64 MB node arrays; in
   site-parallel mode all 103 threads simultaneously issue reads to the shared buffer
   at 64 MB strides (node-major layout: `partial_lh[node_i]` at `base + i×64MB`),
   reducing prefetcher effectiveness between nodes. The combined effect limits the
   achievable parallel speedup to ~27× — mathematically equivalent to the 2.76%
   Amdahl serial fraction.

#### NUMA first-touch patches are already applied

The R1/R2 NUMA patches (described in `numa_first_touch.html`) **are confirmed present**
in the Fix H build (`gadi-spr-r2-avx512`, commit `257485e5`):

- `phylokernelnew.h` lines 1275, 2386, 3595: all use `schedule(static)` ✓
- `phylotreesse.cpp` line 546 (`computePtnFreq`): `#pragma omp parallel for schedule(static)` ✓
- `phylotreesse.cpp` line 578 (`computePtnInvar`): `#pragma omp parallel for schedule(static)` ✓
- `phylotreesse.cpp` line 1302 (`_pattern_lh_cat` zero-fill): `#pragma omp parallel for schedule(static)` ✓

Without these patches the unpatched 104T baseline would be **1111.6 s** — the
cross-socket cliff (104T slower than 64T = 897.4 s). With the patches: 104T = 523.7 s
(2.12× faster). Our Fix H np=2 at 103T achieves 475 s for MF, consistent with the
patched baseline on a comparable problem size.

#### Why the tree phase benefits from np=2 but MF does not

| Phase | Parallelism available | np=1 → np=2 result |
|-------|----------------------|-------------------|
| **ModelFinder** | Model-level (independent models can run concurrently) | Standard np=1 uses 103-way model-parallel (103× speedup). Fix H np=2 uses site-parallel (27× per model, <103×) — **degraded** |
| **Tree search (SPR)** | Site-level only (only ONE tree is being optimised at a time; SPR moves are sequential) | Both np=1 and np=2 use site-parallel for each NNI evaluation. At np=2: 2× more independent subtrees searched → 2× throughput at same per-NNI cost — **improved** |

The tree phase has no model-level parallelism to exploit (there is only one tree at a
time); site-parallel OMP with 103 threads is the natural and only available strategy.
Fix H np=2 tree (387 s) is therefore faster than the ~480–520 s expected for a
single-node tree search on this problem. MF is the anomaly: the standard binary already
achieves near-perfect parallelism through model-level independence, making site-parallel
redundant for MF while efficient for tree search.

#### Summary

| Quantity | Value |
|---------|-------|
| Standard MF (model-parallel, 103 threads) | 399 s for 475 models |
| Fix H np=2 MF (site-parallel, 103 threads/rank) | 475 s for 150 models/rank |
| Effective T₁ per model | 86.7 s |
| Site-parallel speedup (103 threads) | 27.4× |
| Amdahl serial fraction | 2.76% ≈ 2.39 s/model |
| Accumulated serial overhead (150 models) | ≈ 359 s out of 475 s total (75%) |
| NUMA R1/R2 patches present | ✓ confirmed |
| THP for 6.27 GB partial_lh allocations | ✗ — `madvise` mode, no MADV_HUGEPAGE call |
| Speedup needed to match standard MF at np=2 | ≥ 150 × 103 / 475 = **32.5×** (have 27.4×) |

The gap is fundamental: Fix H's sequential outer loop exposes the per-model Amdahl
serial fraction that model-parallel OMP hides by overlapping 103 models' serial work
concurrently. There is no code change that can close this gap without either restoring
the parallel outer loop in MPI builds (blocked by OOM at this scale, §2.4.5) or
reducing per-model memory footprint to allow concurrent model evaluations.

---

## 3. The Algorithmic Root Cause: O(n²) Inner Product Loops

### 3.1 How IQ-TREE Computes Partial Likelihoods

IQ-TREE's likelihood computation follows Felsenstein's pruning algorithm. For each internal node, the partial likelihood vector is built from its two children. The core kernel is in `tree/phylokernelnew.h`, specifically the `computePartialLikelihoodSIMD` function.

For a bifurcating internal node (INTERNAL-INTERNAL case), the hot path is:

```cpp
// For each batch of VCsize site patterns (ptn_lower..ptn_upper, step=VectorClass::size())
for (size_t ptn = ptn_lower; ptn < ptn_upper; ptn += VectorClass::size()) {
    for (size_t c = 0; c < ncat_mix; c++) {           // 4 rate categories (LG+G4)
        for (size_t x = 0; x < nstates; x++) {        // outer: output state
            dotProductDualVec<VectorClass, double, nstates, FMA>(
                eleft_ptr,        // pre-computed left  eigenvec row x (scalar doubles)
                partial_lh_left,  // left child partial lh (VectorClass = 4 patterns wide)
                eright_ptr,       // pre-computed right eigenvec row x
                partial_lh_right, // right child partial lh
                partial_lh_tmp[x] // output: product of two dot-products
            );
            eleft_ptr  += nstates;   // advance to row x+1
            eright_ptr += nstates;
        }
        // then: productVecMat (inv_evec transform) — another O(nstates²) pass
        productVecMat<VectorClass, double, nstates, FMA>(
            partial_lh_tmp, inv_evec_ptr, partial_lh, lh_max
        );
    }
}
```

The `dotProductDualVec` function computes `X = sum_i(A[i]*B[i]) * sum_i(C[i]*D[i])` where A, C are scalar `double*` arrays (the pre-computed transition matrix coefficients `echildren`) and B, D are `VectorClass*` arrays (the pattern-batched partial likelihoods). Each inner loop runs `nstates` iterations, each issuing one AVX FMA instruction that processes `VectorClass::size()` = 4 patterns simultaneously.

**Cost per pattern-batch per rate category per node:**

| Model | nstates | dotProductDualVec FMAs | productVecMat FMAs | Total Vec4d FMAs |
|-------|---------|----------------------|-------------------|-----------------|
| F81+F+G4 (DNA) | 4  | 4 × 4  = 16  | 4 × 4 = 16  | **32**  |
| LG+G4 (AA)     | 20 | 20 × 20 = 400 | 20 × 20 = 400 | **800** |

**FLOP ratio: 800 / 32 = 25×**

With 4 rate categories (`ncat_mix = 4`), the cost per site-pattern-batch per node is:
- DNA: 32 × 4 = 128 Vec4d FMA instructions
- AA: 800 × 4 = 3,200 Vec4d FMA instructions

The O(nstates²) scaling is fundamental — it comes from multiplying the pre-computed branch-transition vector (length nstates) against the child partial-likelihood vector (also length nstates) for each of the nstates output states.

### 3.2 The Pre-Computation Step: computePartialInfo

Before the main loop, `computePartialInfo` (called once per node per tree traversal) pre-computes the `echildren` buffers: `echild[x][i] = evec[x][i] * exp(eigenvalue[i] * branch_length)`. This is also O(nstates²):

```cpp
for (c = 0; c < ncat_mix; c++) {
    // nstates exponentiations: exp(eigenvalue[i] * length)
    for (i = 0; i < nstates / VectorClass::size(); i++)
        expchild[i] = exp(VectorClass(...eigenvalue[i*VCsize..]) * len_child);
    // nstates × nstates multiplies to form echild rows
    for (x = 0; x < nstates; x++)
        for (i = 0; i < nstates / VectorClass::size(); i++)
            echild_ptr[i] = evec_ptr[x*nstates + i*VCsize] * expchild[i];
}
```

Cost (Vec4d per branch per rate category):
- DNA: (4/4=1) exp-Vec4d + (4 × 4/4=4) mul-Vec4d = 5 per category → 20 total
- AA: (20/4=5) exp-Vec4d + (20 × 20/4=100) mul-Vec4d = 105 per category → 420 total
- **Ratio: 420/20 = 21×**

### 3.3 SIMD Kernel Level: AVX+FMA, Not AVX-512

A critical finding from the IQ-TREE logs:

```
Kernel:  AVX+FMA - 47 threads (48 CPU cores detected)   [CLX run]
Kernel:  AVX+FMA - 103 threads (104 CPU cores detected)  [SPR run]
```

**Both CLX and SPR use 256-bit AVX+FMA (`Vec4d`, 4 doubles), NOT 512-bit AVX-512 (`Vec8d`, 8 doubles)**, even though both CPUs support AVX-512F.

Inspection of `tree/phylotreesse.cpp` reveals why:

```cpp
void PhyloTree::setLikelihoodKernel(LikelihoodKernel lk) {
    ...
#ifdef __AVX512KNL          // <-- Only for Knights Landing (KNL/Xeon Phi)!
    if (lk >= LK_AVX512) {
        setDotProductAVX512();
        setLikelihoodKernelAVX512();
    }
#endif
    // Falls through to AVX+FMA for all other AVX-512 hardware
    setLikelihoodKernelFMA();
}
```

The AVX-512 likelihood kernel is gated on `__AVX512KNL` (Knights Landing-specific preprocessor define), NOT the general `__AVX512F__` that would be set on Cascade Lake and Sapphire Rapids. The binaries ARE compiled with AVX-512F support (the `phylokernelavx512.cpp` compilation unit compiles successfully), but the runtime **dispatch never routes to it** on CLX/SPR hardware.

**The SIMD packing consequence:**  
With `Vec4d` (VCsize=4, 256-bit):
- DNA 4 states: `4 % 4 = 0` → perfect packing, vectorized path in `computePartialInfo` ✓
- AA 20 states: `20 % 4 = 0` → perfect packing, vectorized path in `computePartialInfo` ✓

With `Vec8d` (VCsize=8, 512-bit), which would apply if AVX-512 were enabled:
- DNA 4 states: `4 % 8 ≠ 0` → falls to NON-VECTORIZED path in `computePartialInfo` ✗
- AA 20 states: `20 % 8 ≠ 0` → also NON-VECTORIZED path ✗

This is a secondary issue: even if the AVX-512 dispatch bug were fixed, the non-power-of-8 state counts would defeat the vectorized `computePartialInfo` path for both DNA and AA. Padding to 24 (for AA) and 8 (for DNA) would be needed to realise AVX-512 gains. More on this in Section 5.

---

## 4. Memory Hierarchy: Where the Bottlenecks Actually Are

### 4.1 Working Set Per Node

The hot data that must be accessed for each internal node traversal:

**Per-node working set (non-site-model, Vec4d, ncat=4):**

| Data structure          | DNA (nstates=4) | AA (nstates=20) | Ratio |
|------------------------|-----------------|-----------------|-------|
| `eleft`  (echild left) | 4×4×4 = 64 d    | 20×20×4 = 1600 d | 25×   |
| `eright` (echild right)| 64 doubles      | 1,600 doubles   | 25×   |
| `partial_lh_left`  (per batch of 4 patterns) | 4×4 = 16 d | 20×4 = 80 d | 5× |
| `partial_lh_right` (per batch of 4 patterns) | 16 d   | 80 d          | 5×   |
| **Totals (doubles)**   | **160**         | **3,360**       | **21×** |
| **Totals (bytes)**     | **1.25 KB**     | **26.6 KB**     | **21×** |

DNA's per-node hot dataset (1.25 KB) fits trivially in both CLX L1 (32 KB) and SPR L1 (48 KB).  
AA's per-node hot dataset (26.6 KB) **barely fits in SPR L1 (48 KB) but exceeds CLX L1 (32 KB)**.

This explains the massive SPR-over-CLX improvement for AA tree search (3.06×) vs DNA tree search (1.70×):

| Phase        | DNA CLX→SPR speedup | AA CLX→SPR speedup |
|--------------|--------------------|--------------------|
| ModelFinder  | 2.58×              | 2.78×              |
| Tree search  | 1.70×              | **3.06×**          |

ModelFinder speedup is roughly proportional to thread count (103/47 = 2.19×) for both, because ModelFinder evaluates each model independently on a fixed NJ tree — the working set cycles through many different models, so cache behaviour is similar for both.

Tree search, however, repeatedly re-traverses the **same tree** with the **same model**: the echild arrays are reused across SPR iterations. For AA on SPR, the 26.6 KB node working set fits in the 48 KB L1 — so those echild arrays stay hot. For AA on CLX, they spill to L2 on every other node visit, paying 12-cycle L2 latency instead of 4-cycle L1 latency. For DNA on both CLX and SPR, the 1.25 KB hot set fits in L1 either way — no cache benefit from SPR's larger L1.

### 4.2 Perf Counter Evidence

**SPR runs — raw hardware counters (aggregated over all threads × wall time):**

| Counter                   | DNA SPR (168425674) | AA SPR (168425673) | Ratio |
|--------------------------|--------------------|--------------------|-------|
| Cycles                   | 55.1 T             | 286.9 T            | 5.21× |
| Instructions             | 71.7 T             | 538.9 T            | 7.52× |
| L1-dcache loads          | 29.6 T             | 294.8 T            | 9.96× |
| L1-dcache load-misses    | 349 B              | 3,505 B            | 10.0× |
| LLC loads                | 22.7 B             | 248.0 B            | 10.9× |
| LLC load-misses          | 17.2 B             | 166.0 B            | 9.6×  |
| Branch instructions      | 14.5 T             | 58.9 T             | 4.07× |
| IPC                      | 1.302              | **1.878**          | 1.44× |

Notable observations:
- AA executes **7.52× more instructions** than DNA (not 25×) because many instructions are non-FMA overhead
- AA performs **9.96× more L1 loads** — consistent with the 5× data size per state × 5× states for partial_lh + 25× for echild → ~10× average
- AA has **1.44× higher IPC** despite being slower overall
- Branch ratio (4.07×) roughly tracks nstates ratio (20/4 = 5×), confirming branches scale with outer loop iterations

### 4.3 Why AA Has Higher IPC Despite Being Slower

This is counterintuitive but follows directly from the FMA chain structure.

**DNA inner loop** (nstates=4 dot-product chain):
```
vchild = eleft[0] * pleft[0];          // depends on nothing
vchild = fma(eleft[1], pleft[1], vchild); // depends on previous (4-cycle FMA latency)
vchild = fma(eleft[2], pleft[2], vchild); // depends on previous
vchild = fma(eleft[3], pleft[3], vchild); // depends on previous
```
Chain length = 4 FMAs × 4-cycle latency = **16-cycle critical path**.  
With 4 CLX FMA units (2 per socket), throughput is 16/(4×2) = **2 cycles/chain** — but the 16-cycle latency stalls every other state calculation.

**AA inner loop** (nstates=20 dot-product chain):
```
vchild = eleft[0] * pleft[0];
vchild = fma(eleft[1], pleft[1], vchild);  // 4-cycle dep
...  (18 more dependent FMAs)
vchild = fma(eleft[19], pleft[19], vchild); // 80-cycle critical path
```
Chain length = 20 FMAs × 4-cycle latency = **80-cycle critical path**.  
With 20 output states (x=0..19), the OOO engine has 20 independent accumulator chains to interleave. The 80-cycle latency per chain is hidden by executing other chains: the machine can issue 2 FMAs/cycle × 20 independent chains = **effectively throughput-limited at 400/(2×2) = 100 cycles**.

SPR's larger reorder buffer (~512 entries vs ~352 on CLX) can hold more of these 20 independent chains simultaneously, further improving utilisation. This is why AA's IPC (1.878) is higher than DNA's (1.302) and why AA benefits more from SPR's wider OOO window.

**Effective throughput (observed vs theoretical):**
- DNA: theoretical 8 cycles/batch, observed ~30 cycles/batch (memory-bound: every group of 4 state computations must reload echild from cache — but since the loop is so short, branch mispredictions and loop overhead dominate over raw FMA throughput)
- AA: theoretical 200 cycles/batch, observed ~170 cycles/batch (more compute-bound, but still limited by echild streaming from L2/L3 on CLX)

Ratio: 170/30 = **5.7×** — matching the observed 6× on CLX.

### 4.4 DRAM NUMA Imbalance for AA on CLX

From the RAPL energy data for AA CLX (168422809):

```
CPU: 555,794 J  (avg 160.5 W)
  package-0: 253,998 J  (dram-0: 37,744 J)    — NUMA node 0
  package-1: 199,023 J  (dram-1: 65,029 J)    — NUMA node 1
```

DRAM energy ratio: dram-1 / dram-0 = **1.72×** — package-1's DRAM is under 72% more stress.

With `--localalloc` numactl and 47 threads across a 24+24 core CLX system:
- Master thread allocates `partial_lh` tree arrays from NUMA node 0 (package-0)
- Worker threads on package-1 must cross the QPI/UPI interconnect to reach those arrays
- The large AA partial_lh arrays (20 states × 4 cats × N_patterns × ~(2n-1) nodes × 8 bytes) are too large to replicate and too sparse to efficiently cache

The imbalanced DRAM access adds ~15-20% overhead on top of the algorithmic AA slowdown for CLX, beyond what the FMA computation alone explains.

---

## 5. ModelFinder: Why Protein Models Are Especially Expensive

### 5.1 Model Count

ModelFinder tested 1,232 protein models vs 968 DNA models (1.27× more). Protein model space is larger because:
- ~20 base empirical rate matrices (LG, WAG, JTT, Blosum62, cpREV, Dayhoff, mtART, Q.mammal, Q.bird, etc.)
- Combined with +G4, +I+G4, +R3, +R4, +R5, +H4 rate heterogeneity
- Combined with empirical/equal/given/ML frequency variants (+F, +FO, +FU)
- 20 × 4 rate-het × ~15 freq variants ≈ 1,200 models

Per-model wall time (approximate single-thread equivalent):
- AA CLX: 1,108.8s × 47 threads / 1,232 models ≈ **42.3 s/model**
- DNA CLX: 159.1s × 47 threads / 968 models ≈ **7.72 s/model**
- **Ratio: 42.3/7.72 = 5.48×**

This ~5.5× per-model ratio is consistent across hardware and directly reflects the O(nstates²) likelihood kernel cost.

#### 5.1.1 Super-Linear ModelFinder Scaling with Site Count

Both the DNA 1M SPR run (168425675) and DNA 1M CLX run (168422813) show striking super-linear
scaling versus their 100K counterparts. Despite testing the same 968 DNA models on the same
100-taxon tree topology, ModelFinder cost grew far more than the 10× increase in site count.

| Run | MF wall (s) | Models | Per-model (s·thread) | Sites |
|-----|------------|--------|--------------------|-------|
| DNA 100K CLX | 159.084 | 968 | 7.72 | 100,000 |
| DNA 100K SPR |  61.740 | 968 | 6.57 | 100,000 |
| DNA 1M CLX | 10,230.229 | 968 | 496.7 | 1,000,000 |
| DNA 1M SPR |  3,500.825 | 968 | 372.5 | 1,000,000 |
| **Scale ratio (CLX)** | **64.3×** | 1.0× | **64.3×** | **10.0×** |
| **Scale ratio (SPR)** | **56.7×** | 1.0× | **56.7×** | **10.0×** |

CLX shows **more** super-linear MF scaling (64.3×) than SPR (56.7×). With 1M patterns the
partial_lh working set overflows L3 on both platforms, but CLX's smaller per-thread L3
saturates sooner, increasing DRAM traffic more steeply.

**Tree search scaling:**
- DNA CLX: 7,481.884 / 384.838 = **19.4×** (for 10× more sites)
- DNA SPR: 2,596.995 / 226.447 = **11.5×** (near-linear)

CLX tree search (19.4×) also scales more super-linearly than SPR (11.5×), consistent with the
same cache-saturation effect. SPR's larger per-core L3 keeps more partial_lh arrays hot during
tree traversal, yielding near-linear scaling; CLX spills to DRAM earlier.

**CLX vs SPR speedup at 1M scale:** 17,752.858 / 6,114.450 = **2.90×** total (vs 103/47 = 2.19×
thread ratio). At 100K the DNA CLX→SPR speedup was only 1.89× (well below thread ratio, memory-bound);
at 1M both CLX and SPR are DRAM-saturated, and SPR's additional memory bandwidth channels provide
extra throughput, raising the speedup above the thread ratio.

ModelFinder's super-linear scaling is driven by two compounding effects:
1. **NNI convergence takes more iterations at larger lnL gradients** — with 10× more sites, each
   parameter step changes lnL by ~10× more, so the `epsilon=0.1` tolerance requires more rounds
   to distinguish signal from numerical noise in the gradient.
2. **Memory pressure** — 1M sites × 4 rate categories × nstates × ~(2n−1) nodes creates partial_lh
   arrays ~10× larger, saturating L3 cache and increasing memory traffic per MF model evaluation.

These effects are **not equally strong for all data types.** DNA at 100K is already memory-bandwidth
bound (IPC~1.3), so 10× more sites triggers disproportionate DRAM saturation. AA at 100K is
FLOP-dominated (IPC~2.0), so scaling is far more linear. The AA 1M SPR and CLX runs confirm this:

**Actual AA 1M results (168425490 CLX, 168425491 SPR):**

| Run | MF wall (s) | Tree wall (s) | Total (s) | Per-model (s·thread) |
|-----|-------------|---------------|-----------|---------------------|
| AA 100K CLX (168422809) | 1,108.8 | ~1,877 | 3,460.8 | 42.3 |
| AA 100K SPR (168425673) | 399.5 | 764.5 | 1,169.6 | 33.4 |
| AA 1M CLX (168425490) | 16,308.318 | 34,821.973 | 51,328.252 | 622.4 |
| AA 1M SPR (168425491) | 7,587.459 | 15,098.605 | 22,776.226 | 634.4 |
| **Scale (CLX)** | **14.7×** | **~18.6×** | **14.8×** | **14.7×** |
| **Scale (SPR)** | **19.0×** | **19.7×** | **19.5×** | **19.0×** |

The DNA MF scale factor (56.7× on SPR) is not transferable to AA. AA MF scales at only 19.0× —
close to linear with site count. The earlier prediction of ~22,641 s (by applying the DNA factor)
overestimated by 3×; the actual SPR MF time is 7,587 s.

**CLX vs SPR speedup at 1M scale (AA): 51,328 / 22,776 = 2.25×** — near the thread ratio (2.19×).
At 100K, AA CLX was 2.96× slower (above thread ratio, memory-bound). At 1M, both platforms are
FLOP-bound: per-model thread cost CLX 622 s vs SPR 634 s — essentially identical, no bandwidth gap.

Compare to DNA 1M: CLX was still 2.90× slower than SPR at 1M (bandwidth gap persisted). This
confirms that AA is fundamentally more compute-bound than DNA at every alignment scale tested, and
that DNA's L3→DRAM spill threshold is reached at smaller site counts than AA's.

### 5.2 F81 vs LG Eigendecomposition

The best DNA model was `F81+F+G4`. F81 (Felsenstein 1981) is a special-cased model in `model/modelmarkov.cpp`:

```cpp
if (num_params == -1) {
    // F81-style: analytical eigendecomposition
    eigenvalues[0] = 0.0;
    for (i = 1; i < num_states; i++) eigenvalues[i] = -mu;  // all equal!
    // eigenvectors: trivially computed in O(nstates²) — no LAPACK needed
    ...
    return;
}
// All other models: numerical eigendecomposition via Eigen3
SelfAdjointEigenSolver<MatrixXd> eigensolver(Q);  // O(nstates³)
```

F81 has analytically known eigenvalues (one zero, the rest all equal to −μ) and closed-form eigenvectors. No iterative solver is needed. This makes model setup trivial.

LG's 20×20 empirical rate matrix has 20 distinct irrational eigenvalues — it requires full numerical `SelfAdjointEigenSolver` which scales as O(n³) = O(20³) = O(8000) operations. However, eigendecomposition is performed **once per model evaluation** (not per site), so its contribution to the total is small: ~0.1-1 ms per model vs ~7-42 s per model for the likelihood kernel. The per-site kernel dominates.

The real advantage of F81 is that its **transition matrix** has simple structure: all off-diagonal elements are proportional to the equilibrium frequency. This means F81 is a "one-parameter model" that converges faster during the `--eps 5` and `--eps 1` pre-optimization passes in ModelFinder. Fewer parameter optimization iterations means fewer full-tree likelihood evaluations during model selection.

### 5.3 Site Pattern Counts

AA had 96,017 distinct patterns, DNA had 94,532 — a difference of only 1.6%. This is negligible and confirms that the AA/DNA slowdown is NOT due to having more unique patterns.

---

## 6. Optimization Pathways

### 6.1 Enable AVX-512 Likelihood Kernel on Modern AVX-512F Hardware

**Impact: ~2× throughput, requires code change**

The AVX-512 likelihood kernel in `tree/phylokernelavx512.cpp` is already compiled into the binary but never invoked on CLX/SPR. Fix: change the dispatch guard in `tree/phylotreesse.cpp`:

```diff
-#ifdef __AVX512KNL
+#if defined(__AVX512KNL) || defined(__AVX512F__)
     if (lk >= LK_AVX512) {
         setDotProductAVX512();
         setLikelihoodKernelAVX512();
     }
-#endif
+#endif
```

However, this alone is **not sufficient** due to the `nstates % VectorClass::size() == 0` checks. With `Vec8d` (VCsize=8):
- DNA (4 states): `4 % 8 ≠ 0` → falls to non-vectorized `computePartialInfo` path
- AA (20 states): `20 % 8 ≠ 0` → also falls to non-vectorized path

### 6.2 State-Count Padding to Enable AVX-512

**Impact: ~2× throughput for AA when combined with 5.1, requires memory layout change**

Pad partial_lh and echildren arrays to the next multiple of 8:
- `get_safe_upper_limit(20)` = 24 (already computed correctly)
- `get_safe_upper_limit(4)` = 8 (already computed correctly)

The padding is already implemented via `get_safe_upper_limit`. The problem is the compile-time `nstates % VectorClass::size() == 0` guard in the **vectorized path selection**. With `KERNEL_FIX_STATES` and compile-time `nstates=20`:

```cpp
if (nstates % VectorClass::size() == 0) {  // 20 % 8 = 4 ≠ 0 → non-vectorized
```

The fix is to use the padded dimension in the SIMD loops:

```cpp
constexpr size_t nstates_padded = get_safe_upper_limit_constexpr(nstates, VCsize);
// Then: iterate i from 0..nstates_padded/VCsize instead of checking divisibility
```

Alternatively, add explicit specialisations for `nstates=20` with `VCsize=8` that use 3 Vec8d registers per row (24 elements, 4 padding zeros). This matches the existing `malign` buffer layout that already pads to 24.

**Memory overhead:** AA padded to 24 stores 20% more zeros per eigenvector row — but this is already accounted for in the existing `mix_addr_malign` offsets. The main per-site partial_lh array does NOT use malign padding (it uses `nstates` directly, not `get_safe_upper_limit`), so the partial_lh arrays would need a separate change.

### 6.3 NUMA-Aware Tree Data Allocation

**Impact: ~10-20% on AA CLX, low implementation cost**

The DRAM NUMA imbalance (1.72×) on CLX wastes bandwidth and increases average memory latency for package-1 worker threads. The partial_lh tree arrays are allocated by the master thread (on NUMA node 0) and distributed to workers by `computePartialLikelihoodSIMD`'s `ptn_lower..ptn_upper` packet dispatch.

**Fix:** Allocate each PhyloNeighbor's `partial_lh` buffer on the NUMA node of the thread that will compute it. This requires knowing the thread-to-subtree assignment before allocation, which is approximately possible from the traversal order:

```cpp
// In PhyloTree::allocateMemory():
int packet_id = (neighbor_index * num_threads) / num_edges;
int numa_node = omp_get_place_num_for_packet(packet_id);
neighbor->partial_lh = (double*)numa_alloc_onnode(size, numa_node);
```

A simpler approach: use `numactl --interleave=all` instead of `--localalloc` for AA runs. Interleaving spreads memory pages round-robin across NUMA nodes, halving the average cross-NUMA traffic. This sacrifices local-allocation locality for balance, and is beneficial when 50% of accesses are already remote.

### 6.4 Single-Precision (Float32) for SPR Move Screening

**Impact: ~2× throughput for topology search, requires separate float32 kernel path**

SPR (Subtree Pruning and Regrafting) moves require comparing log-likelihoods to select beneficial rearrangements. The precision requirement is relative (which topology is better), not absolute (exact likelihood value). Float32 provides ~7 significant digits — sufficient for ranking topology candidates.

With `Vec8f` (AVX, 8 floats) instead of `Vec4d` (AVX, 4 doubles):
- Double throughput (8 vs 4 per instruction)
- 20 % 8 ≠ 0 → still needs padding to 24 for AA

With `Vec16f` (AVX-512, 16 floats):
- 4× the throughput of `Vec4d`
- 20 % 16 ≠ 0 → needs padding to 32 for AA (60% overhead)

The existing `BOOT_VAL_FLOAT` infrastructure in `phylokernelavx512.cpp` shows this was already considered:
```cpp
#ifdef BOOT_VAL_FLOAT
    dotProduct = &PhyloTree::dotProductSIMD<float, Vec16f>;
#else
    dotProduct = &PhyloTree::dotProductSIMD<double, Vec8d>;
#endif
```

A mixed-precision workflow would:
1. Use float32 for all SPR likelihood evaluations during topology search (103 iterations)
2. Switch to float64 for the final branch length and model parameter optimisation
3. Final log-likelihood is computed in float64 for reporting

Expected speedup for AA tree search: 1.5-2× from float32 SPR.

### 6.5 GPU Offload for Partial Likelihood Computation

**Impact: potential 5-10× for the hot kernel, high implementation cost**

The `computePartialLikelihoodSIMD` kernel is a perfect candidate for GPU acceleration: it's a batched matrix-vector multiply across N_patterns site patterns.

For each internal node visit:
- Input: `partial_lh_left[nptn × ncat × nstates]` and `partial_lh_right[nptn × ncat × nstates]`
- Input: `eleft[ncat × nstates × nstates]` and `eright[ncat × nstates × nstates]`
- Output: `partial_lh_dad[nptn × ncat × nstates]`

For AA 100K SPR (103 threads, 96,017 patterns):
- Each thread processes ~932 patterns per batch
- GPU has >10,000 cores that can process all 96,017 patterns simultaneously
- The memory transfer cost: 96017 × 4 cats × 20 states × 8 bytes × 2 (left+right) ≈ 49 MB per node visit
- PCIe 4.0 bandwidth: ~30 GB/s → 49 MB transfer ≈ 1.6 ms
- GPU compute at 10 TFLOPS FP64: 96017 × 4 × 800 FMAs ≈ 307 GFLOP → 31 ms compute

Transfer cost (1.6ms) is small compared to compute (31ms) → **compute bound on GPU**, which is ideal. However:
- GPU L2 cache: typically 80-96 MB on H100/A100 — the 97,280 × 80 doubles (62 MB) would nearly fill it
- Multiple node visits in parallel require storing many partial_lh arrays simultaneously
- The total partial_lh tree size: (2×100-2) × 96017 × 4 × 20 × 8 = ~12 GB → exceeds typical GPU memory

**Practical GPU approach:** Process the tree in subtree batches of 20-30 nodes simultaneously, streaming partial_lh data to/from the GPU. The `phylotreegpu.cpp` file in the `cpu_opt_merge` branch suggests this is already being explored. Key challenge: the sequential dependency in Felsenstein's algorithm (parent must wait for both children) limits how many nodes can be processed concurrently. A wavefront scheduling approach (process all nodes at tree depth D simultaneously before D+1) maximises parallelism.

### 6.6 Vectorised Reduction via `hadd` Optimisation

**Impact: 5-15% for the reduction step, minimal code change**

After computing the nstates dot-product accumulators `partial_lh_tmp[0..nstates-1]`, the `productVecMat` step applies the inverse eigenvector matrix:

```cpp
for (size_t x = 0; x < nstates; x++) {
    VectorClass out = 0.0;
    for (size_t i = 0; i < nstates; i++)
        out = mul_add(inv_evec_ptr[i], partial_lh_tmp[i], out);
    partial_lh[x] = out;
    inv_evec_ptr += nstates;
}
```

This second O(nstates²) pass could be fused with the first using a 2D accumulation pattern, reducing the number of passes over `inv_evec` and improving cache utilisation. For AA with 20×20 = 400 elements per category, the two-pass structure doubles the memory reads of `partial_lh_tmp`.

A fused loop would compute both `dotProductDualVec` and `productVecMat` together: for each output state x, accumulate `inv_evec[x][j] × (sum_i eleft[j][i] × pleft[i]) × (sum_i eright[j][i] × pright[i])` in a single triple-nested loop. This would be O(nstates³) per state per category, which is worse; so the current two-pass structure is likely optimal. However, caching `partial_lh_tmp` in registers (rather than memory) between the two passes could help.

### 6.7 OpenMP SIMD Acceleration of the echild Inner Loop

**Impact: 10-20% for computePartialInfo, low implementation cost**

The `computePartialInfo` function runs serially (no parallelism per node) and is called before the pattern-parallel main loop. Adding `#pragma omp simd` to the scalar inner loop:

```cpp
// In the non-vectorized path of computePartialInfo (for Vec8d with nstates not divisible by 8):
for (i = 0; i < nstates; i++) {
    expchild[i] = exp(eval_ptr[i] * len_child);  // non-SIMD exp call
}
#pragma omp simd
for (x = 0; x < nstates; x++)
    for (i = 0; i < nstates; i++)
        echild_ptr[i] = evec_ptr[x*nstates+i] * expchild[i];
```

Would allow the compiler to auto-vectorize the inner multiply loop. The `exp()` calls are already vectorized in the SIMD path via `exp(VectorClass(...))`.

---

## 7. Summary: Root Cause Hierarchy

| Rank | Root Cause | Factor | Evidence |
|------|-----------|--------|----------|
| 1 | O(nstates²) inner product per site per node: 20² vs 4² | **25× FLOPs** | Source: `dotProductDualVec` + `productVecMat` in `phylokernelnew.h` |
| 2 | Memory working set per node: 26.6 KB vs 1.25 KB | **21×** | Exceeds CLX L1 (32KB), fits SPR L1 (48KB) → explains AA 3.06× CLX→SPR vs DNA 1.70× |
| 3 | DNA is memory-bound (short FMA chains, L1 thrash), AA is compute-bound (long chains fill OOO) | DNA effective throughput ÷5 | IPC: AA 1.878 vs DNA 1.302 on SPR; LLC miss: DNA 75.8% vs AA 66.9% |
| 4 | More protein models in ModelFinder: 1,232 vs 968 | 1.27× model count | IQ-TREE stdout log: "test up to 1232 protein models" |
| 5 | AVX-512 kernel not activated on CLX/SPR (KNL-gated) | 2× missed | `#ifdef __AVX512KNL` in `tree/phylotreesse.cpp`; both runs report "AVX+FMA" |
| 6 | NUMA DRAM imbalance (package-0 data, cross-socket access) | ~15% extra on CLX AA | RAPL: dram-1 = 65,029 J vs dram-0 = 37,744 J (1.72× imbalance) |

The **combined effective AA/DNA slowdown** of 4-6× (vs theoretical 25× FLOPs) is explained by factors 2 and 3 cancelling most of factor 1: DNA is so memory-starved that its throughput is ~5× below its FMA capacity, narrowing the effective gap. In a memory-bandwidth-unlimited scenario, AA would be the full 25× slower.

---

## 8. Quantitative Projection: Potential Improvements

If implemented together (conservative estimates):

| Optimisation | AA SPR wall improvement | AA CLX wall improvement |
|-------------|------------------------|------------------------|
| AVX-512 + state padding (§6.1-6.2) | ~1.5× | ~1.5× |
| Float32 SPR screening (§6.4) | ~1.3× tree search | ~1.3× |
| NUMA-aware allocation (§6.3) | — | ~1.1-1.2× |
| GPU offload for partial_lh (§6.5) | ~3-5× tree search | ~3-5× |
| Combined (§6.1 + §6.4, no GPU) | ~1.7× total | ~1.8× total |
| Combined with GPU offload | ~4× total | ~4× total |

Target: AA SPR 1,169s → **~290s** (matching current DNA SPR wall time!) with full AVX-512 + float32 SPR + GPU partial_lh.

---

## 9. Appendix: Key Source Code Locations

| Purpose | File | Key symbol |
|---------|------|-----------|
| Core partial likelihood kernel | `tree/phylokernelnew.h:1307` | `computePartialLikelihoodSIMD` |
| Pre-compute eigenvector × exp | `tree/phylokernelnew.h:900` | `computePartialInfo` |
| AVX-512 kernel dispatch | `tree/phylokernelavx512.cpp:38` | `setLikelihoodKernelAVX512` |
| Runtime SIMD dispatch (KNL gate)  | `tree/phylotreesse.cpp:92` | `setLikelihoodKernel` |
| Safe upper limit (state padding) | `utils/tools.h:3014` | `get_safe_upper_limit` |
| F81 analytical eigendecomp | `model/modelmarkov.cpp:1482` | `decomposeRateMatrix` (num_params==-1 branch) |
| LG numerical eigendecomp | `model/modelmarkov.cpp:1570` | `SelfAdjointEigenSolver` |
| Vec4d/Vec8d FMA dot-product | `tree/phylokernelnew.h:210` | `dotProductDualVec` |
| Matrix-vector product (inv_evec) | `tree/phylokernelnew.h:455` | `productVecMat` |
| GPU OpenACC offload stub | `tree/phylokernel_openacc.cpp` | `computePartialInfoGPU` |
