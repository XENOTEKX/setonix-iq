# ModelFinder MPI Model-Level Dispatch — Deep Source Analysis & Implementation Plan

**Branch:** `modelfinder2`  
**IQ-TREE base:** `gadi-spr-r2-avx512` (v3.1.2 + NUMA R1/R2 + AVX-512 patches)  
**Working source:** `/scratch/um09/as1708/iqtree3-mf2/src/iqtree3`  
**Date:** 2026-05-10  
**Revised:** 2026-05-10 — complete source analysis of `main/phylotesting.cpp`

---

## 1. Problem Statement

IQ-TREE's ModelFinder evaluates substitution models **sequentially in a plain for
loop**. With a standard DNA candidate set, 968 models are tested one at a time,
each consuming the full OMP thread pool. In MPI mode **every rank evaluates every
model redundantly** — there is no model-level MPI dispatch of any kind in the current
code for single-alignment runs.

Confirmed on Gadi 4-node run (PBS 167977883, 416 cores):

```
IQ-TREE MPI version 3.1.2  — Host: gadi-cpu-spr-0428 (AVX512, FMA3, 503 GB RAM)
Command: iqtree3-mpi -s alignment_10000000.phy -T 104 -seed 1
MPI: 4 processes
Alignment: 100 sequences, 10,000,000 columns, 10,000,000 distinct patterns
  (0% site-pattern compression — worst-case synthetic dataset)

Time for fast ML tree search:  510.365 seconds
NOTE: ModelFinder requires 324,249 MB RAM!
ModelFinder will test up to 968 DNA models (sample size: 10,000,000) ...
  1  JC         -LnL 672798764.274  ... BIC 1345600703.813
  2  JC+ASC     ...
  ...
  9  JC+R6      -LnL 672798787.202  ... BIC 1345600910.850
[PBS walltime limit 2:00:00 reached; Exit_status=271]
```

| Metric | Value |
|--------|-------|
| Models tested | 9 / 968 (0.93%) |
| Wall rate | ~729 s/model |
| Projected total | ~196 h |
| Peak RAM declared | 324 GB per node |
| MPI contribution | 0 — all 4 ranks evaluated the same 9 models |

**The core issue is not MPI communication overhead but MPI underutilisation.** Each
MPI rank wastes its 416-core allocation on redundant computation of models already
being computed by other ranks.

---

## 2. How the Model Count Is Determined (Not Fixed)

The "968 models" figure is **not a hard-coded constant** — it is the product of three
independent axes computed at runtime in `CandidateModelSet::generate()`:

```
Total models = |substitution set| × |frequency variants| × |rate heterogeneity models|
```

### 2.1 Substitution Model Set (Axis 1) — `getModelSubst()`

Controlled by `--mset` flag (`params.model_set`). For DNA the key arrays at the top
of `phylotesting.cpp` are:

| `--mset` value | Array | Count |
|---|---|---|
| *(default)* | `dna_model_names` | **22** |
| `partitionfinder` / `phyml` | `dna_model_names_old` | 14 |
| `raxml` | `dna_model_names_rax` | 1 (GTR only) |
| `mrbayes` | `dna_model_names_mrbayes` | 6 |
| `beast1` | `dna_model_names_beast1` | 3 |
| `beast2` | `dna_model_names_beast2` | 4 |
| `non-reversible` | `dna_model_names_nonrev` | 30 |
| `liemarkov` | fullsym + RY + WS + MK | ~99 |
| custom `--mset GTR,HKY,...` | user list | varies |

Default 22: `JC F81 K80 HKY TNe TN K81 K81u TPM2 TPM2u TPM3 TPM3u TIMe TIM TIM2e TIM2
TIM3e TIM3 TVMe TVM SYM GTR`

For protein: 28 models in `aa_model_names[]` by default.

### 2.2 Frequency Variants (Axis 2) — `getStateFreqs()`

Controlled by `--mfreq` flag (`params.state_freq_set`). For DNA:

| `--mfreq` value | Array | Count |
|---|---|---|
| *(default)* | `dna_freq_names = {"+FQ", "+F"}` | **2** |
| `FULL` | `dna_freq_names_full = {"+FQ", "+F", "+FO"}` | 3 |
| custom `--mfreq F` | user list | 1 |

Each substitution model is combined with every frequency variant:
`JC+FQ`, `JC+F`, `F81+FQ`, `F81+F`, ..., `GTR+FQ`, `GTR+F` → 22 × 2 = **44 subst×freq combos**.

**Historical note:** IQ-TREE 1.x used `{""}` (no suffix) as the only freq variant,
giving 22 × 1 = 22 subst combos. The `+FQ` vs `+F` split was introduced in IQ-TREE 2.x,
which is why the same `-m TEST` command now produces 176 models rather than the 88 that
users familiar with IQ-TREE 1.x may remember.

### 2.3 Rate Heterogeneity Models (Axis 3) — `getRateHet()`

Controlled by:
- `-m` mode (`MF`/`MFNEW` vs `TEST`/`TESTONLY`) — `with_new` flag
- `--mrate` flag (`params.ratehet_set`)
- `frac_invariant_sites` — whether alignment has invariant sites
- `--min-rate-cats` / `--max-rate-cats` (default: 2–10)

The base rate options defined in `getRateHet()`:
```
rate_options = {"", "+I", "+ASC", "+G", "+I+G", "+ASC+G", "+R", "+ASC+R", "+I+R"}
```

Active set per scenario (from boolean arrays in source):

| Scenario | Active base rates | +R/+I+R expansion | Total rate models |
|---|---|---|---|
| `-m MF` + normal data (frac_inv > 0) | `"" +I +G +I+G` + `+R +I+R` | +R2..+R10, +I+R2..+I+R10 | **22** |
| `-m TEST` + normal data | `"" +I +G +I+G` (no +R) | none | **4** |
| `-m MF` + 0% invariant sites (SNP/ASC) | `"" +ASC +G +ASC+G` + `+R +ASC+R` | +Rx, +ASC+Rx | **22** |
| `-m MF` + `--mrate G,I+G` | user override | none | **2** |
| fast mode (`-m TESTONLY` ratehet_set="1") | `+I+G` only | none | **1** |
| all-invariant (`frac_inv == 1.0`) | `""` only | none | **1** |

The `+R` and `+I+R` entries expand by category count: with default `min_rate_cats=2`,
`max_rate_cats=10`, each generates 9 entries (+R2 through +R10). That is the main
driver of the large model count in MF mode:

```
4 base rates   +  9 (+R2..+R10)  +  9 (+I+R2..+I+R10) = 22 rate models
```

### 2.4 The Full Arithmetic — Why 968

```
22 subst models   (dna_model_names, default --mset)
× 2 freq variants (+FQ, +F — introduced in IQ-TREE 2.x)
= 44 subst×freq combinations

× 22 rate models:
     4 base  : equal rates "", +I, +G, +I+G
   + 9 +Rx   : +R2, +R3, +R4, +R5, +R6, +R7, +R8, +R9, +R10
   + 9 +I+Rx : +I+R2, +I+R3, ..., +I+R10
= 22 rate models

= 44 × 22 = 968 total candidate models
```

### 2.5 Common Model Counts in Practice

| Command / scenario | Total |
|---|---|
| `-m MF` (default, DNA, IQ-TREE 2.x, frac_inv > 0) | **968** |
| `-m MF` (DNA, 0% invariant sites / SNP data) | **968** (different rate mix: +ASC replaces +I) |
| `-m MF --mset partitionfinder` (DNA) | 14×2×22 = **616** |
| `-m MF --mset mrbayes` (DNA) | 6×2×22 = **264** |
| `-m MF --mrate G,I+G` (DNA, 2 rates only) | 44×2 = **88** |
| `-m TEST` (DNA, IQ-TREE 2.x, no +R) | 44×4 = **176** |
| `-m TEST` (DNA, IQ-TREE 1.x, single freq) | 22×4 = **88** |
| `-m MF` (protein, default, IQ-TREE 2.x) | 28×2×22 = **1232** |
| `-m MF` (protein, `--msub nuclear`) | 10×2×22 = **440** |
| `-m MF --max-rate-cats 4` (DNA) | 44×(4+3+3) = **440** |

**The 88 models you saw:** Almost certainly an IQ-TREE 1.x run (single freq variant,
4 rate models: 22×1×4 = 88) or a IQ-TREE 2.x run with `--mrate G,I+G` or `--mset`
restricted to 2 freq variants and 2 rate models (44×2 = 88).

### 2.6 Implication for MPI Dispatch Patch

The dispatch patch divides whatever `candidate_models.size()` is across ranks — it is
not tied to 968 specifically. Any model count produced by `generate()` is partitioned
by `i % nranks`. This means the patch is equally valid for a 264-model run
(`--mset mrbayes`) or a 1232-model protein run.

---

## 3. Source Code Architecture — What Really Happens Today

> **See also Section 2** for how the 968 model count is derived from the three axes
> (substitution set × frequency variants × rate heterogeneity).

All ModelFinder code lives in **`main/phylotesting.cpp`** (7021 lines) and its
header `main/phylotesting.h` (845 lines). There is no `model/modelfinder.cpp`.

### 2.1 Class Hierarchy

```
CandidateModel              ← one substitution + rate model
  .subst_name               ← "JC", "GTR", etc.
  .rate_name                ← "+G4", "+R3", etc.
  .logl, .df                ← filled by evaluate()
  .AIC_score, .BIC_score    ← computed by computeICScores()
  .flag                     ← MF_IGNORED | MF_RUNNING | MF_DONE | MF_WAITING
  .evaluate()               ← line 1895: creates IQTree, optimises, returns lnL

CandidateModelSet : vector<CandidateModel>
  .generate()               ← line ~1640: fills 968 models from dna_model_names[]
  .test()                   ← line 2911: SEQUENTIAL for-loop (default path)
  .evaluateAll()            ← line 3357: OMP parallel loop (--openmp-by-model path)
  .getNextModel()           ← line ~3330: thread-safe model queue pop (OMP only)

ModelCheckpoint : Checkpoint
  ← gzip checkpoint (.model.gz), stores all lnL/BIC results
  ← on restart: previously-computed models are skipped via restoreCheckpoint()
```

### 2.2 Entry Point Call Graph

```
runModelFinder()                        ← line 1325, called from phyloanalysis.cpp
  │
  ├─ computeFastMLTree()                ← line ~720, one NNI pass for initial tree
  │    └─ MPI_Allreduce(MPI_MAXLOC)    ← line 825, winner broadcasts initial tree
  │         (ONLY MPI CALL IN MF PHASE)
  │
  ├─ candidate_models.generate()       ← builds 968 CandidateModel objects
  │
  └─ if params.openmp_by_model:
  │    candidate_models.evaluateAll()  ← OMP parallel, 1 thread/model
  └─ else (default, used in all PBS runs):
       candidate_models.test()         ← sequential, all threads on 1 model
```

### 2.3 `CandidateModelSet::test()` — The Sequential Loop (line 2911)

This is the bottleneck. Simplified structure:

```cpp
// line 3013 — THE SEQUENTIAL FOR LOOP
for (model = 0; model < size(); model++) {
    if (at(model).hasFlag(MF_IGNORED)) {
        model_scores.push_back(DBL_MAX);
        continue;
    }
    // creates new IQTree, calls optimizeParameters → computeLikelihood
    tree_string = at(model).evaluate(params, model_info, out_model_info,
                                     models_block, num_threads, brlen_type);
    at(model).computeICScores(ssize);
    at(model).setFlag(MF_DONE);

    // early-stop: if +R_k is worse than +R_{k-1}, mark all +R_{k+1..} MF_IGNORED
    if (skip_model) {
        for (int next = model+1; ...) at(next).setFlag(MF_IGNORED);
    }
    model_info.dump();       // checkpoint to disk
}
```

In MPI mode: **all N ranks enter this loop and execute it identically.** The ranks
are not coordinated inside the loop. Each rank creates its own IQTree, runs all 968
evaluations, and writes the same checkpoint. No MPI calls inside the loop.

### 2.4 `CandidateModelSet::evaluateAll()` — The OMP Parallel Loop (line 3357)

Activated only with `--openmp-by-model` flag. Uses `getNextModel()` (OMP-critical,
atomic-like queue) to dispatch models to OMP threads:

```cpp
#pragma omp parallel num_threads(num_threads)
{
    int64_t model;
    do {
        model = getNextModel();       // pops next un-evaluated model
        if (model == -1) break;
        at(model).evaluate(..., 1 OMP thread per model);
        // OMP critical section: update best_score, dump checkpoint
    } while (model != -1);
}
```

Each model gets 1 OMP thread. With 104 threads and 968 models: ~10 concurrent
evaluations. **No early-stop logic** (unlike `test()`). Also not MPI-aware.

### 2.5 `CandidateModel::evaluate()` — Single Model Kernel (line 1895)

```cpp
string CandidateModel::evaluate(Params &params, ...) {
    IQTree *iqtree = new IQTree(in_aln);       // full alignment copy
    iqtree->setLikelihoodKernel(params.SSE);   // selects AVX-512 dispatch
    iqtree->setNumThreads(num_threads);        // sets OMP thread count
    iqtree->initializeModel(params, getName(), models_block);
    iqtree->initializeAllPartialLh();

    new_logl = iqtree->getModelFactory()->optimizeParameters(
        brlen_type, false, params.modelfinder_eps, TOL_GRADIENT_MODELTEST);
    // ↑ calls computeLikelihood() → computeLikelihoodBranch() → SIMD kernel
    // ↑ in standard MPI mode: NO MPI inside here for single-aln runs

    delete iqtree;   // ← tree is created and destroyed per model
    return tree_string;
}
```

Key: `new IQTree(in_aln)` — each model gets its own tree object sharing the alignment
pointer but with independent partial-likelihood buffers. The **alignment is not split
across MPI ranks** during this phase; each rank holds the full alignment in memory.

### 2.6 MPI Usage Map — Where MPI Actually Fires

| Location | MPI calls | When |
|---|---|---|
| `computeFastMLTree()` line 825 | `MPI_Allreduce(MPI_MAXLOC)` | After NNI: pick winner process |
| `computeFastMLTree()` line 834–867 | `MPIHelper::sendCheckpoint` / `recvCheckpoint` | Broadcast winning tree to all ranks |
| `ModelFactory::optimizeParameters()` lines 1617–1714 | `syncChkPoint->masterSyncOtherChkpts()` | **Only if `syncChkPoint != nullptr`** — set for PartitionFinder only |
| `PartitionFinder::getBestModelforPartitionsMPI()` | `MPI_Scatter`, `MPI_Gather`, `MPI_Send`/`Recv` | PartitionFinder (-p flag); distributes **partitions** across ranks |
| `PartitionFinder::consolidPartitionResults()` | `MPI_Bcast` (lhvec, dfvec, lenvec) | PartitionFinder result broadcast |
| Tree search (IQTree::doNNISearch) | `MPI_Send`/`Recv` trees | After ModelFinder completes |

**For single-alignment ModelFinder (`-m MF` without `-p`): only the initial tree
NNI MPI call fires.** Everything else is serial or OMP.

### 2.7 Why MPI Is Wasted Today

```
Rank 0: JC → JC+ASC → JC+G4 → ... → GTR+R10   (968 models, 104 OMP threads each)
Rank 1: JC → JC+ASC → JC+G4 → ... → GTR+R10   (same 968 models, redundant)
Rank 2: JC → JC+ASC → JC+G4 → ... → GTR+R10   (same 968 models, redundant)
Rank 3: JC → JC+ASC → JC+G4 → ... → GTR+R10   (same 968 models, redundant)
```

Adding a 5th node adds 0 speedup to ModelFinder. At 4 nodes × 104 OMP, each model
uses 416 CPU cores but the 4-way redundancy means the SU cost is 4× what it should be.

---

## 4. The Pattern Parallelism Hierarchy (Existing)

Understanding what already exists inside a single model evaluation:

```
CandidateModel::evaluate()
  └─ IQTree::computeLikelihood()
       └─ computeLikelihoodBranch()   ← function pointer to SIMD specialisation
            └─ phylokernelnew.h::computeLikelihoodBranchSIMD<AVX512>()
                 │
                 │  // OMP dispatch over site-pattern packets:
                 │  // num_packets = num_threads × PACKETS_PER_THREAD
                 │
                 ├─ #pragma omp parallel for num_threads(num_threads)
                 │    for packet_id in [0, num_packets):
                 │      ptn_lower = limits[packet_id]
                 │      ptn_upper = limits[packet_id+1]
                 │      // AVX-512: process 8 doubles per cycle
                 │      for ptn in [ptn_lower, ptn_upper) step 8:
                 │        __m512d partial_lh = ...
                 │
                 └─ // final sum across packets: serial reduction
                    double lnL = sum(pattern_lh[0..nptn])
```

OMP threads each own a contiguous slice of site patterns (`ptn_lower..ptn_upper`).
The kernel is `AVX-512` width (8 doubles/cycle) on Sapphire Rapids with the icpx build.
**No MPI communication inside the kernel.**

`setNumThreads(n)` controls how fine the packet split is:
```cpp
// phylotreesse.cpp line 51
this->num_packets = (num_threads==1) ? 1 : (num_threads * PACKETS_PER_THREAD);
```

The existing three-level hierarchy per model:

```
Level 1 — SIMD  :  AVX-512, 8× double/cycle  (hardware)
Level 2 — OMP   :  104 threads × pattern packets  (intra-node)
Level 3 — MPI   :  currently wasted (all ranks duplicate all work)
```

**The patch adds genuine content to Level 3.**

---

## 5. Proposed Architecture: Rank-Striped Model Dispatch

### 4.1 Core Idea

Mark each model with its owning rank before the loop starts. Each rank evaluates only
its own subset. After all ranks finish, collect scores with a single `MPI_Allreduce`.

The `MF_IGNORED` flag mechanism already exists and is respected by both `test()` and
`evaluateAll()`. We exploit it:

```cpp
// After candidate_models.generate(), before test()/evaluateAll()
// New function: markRankModels(candidate_models, my_rank, num_ranks)
#ifdef _IQTREE_MPI
if (MPIHelper::getInstance().getNumProcesses() > 1) {
    int my_rank  = MPIHelper::getInstance().getProcessID();
    int nranks   = MPIHelper::getInstance().getNumProcesses();
    for (int i = 0; i < (int)candidate_models.size(); i++) {
        if (i % nranks != my_rank)
            candidate_models[i].setFlag(MF_IGNORED_MPI);  // new flag value
    }
}
#endif
```

Then after `test()` / `evaluateAll()` returns:

```cpp
// New function: gatherModelScores(candidate_models)
#ifdef _IQTREE_MPI
if (MPIHelper::getInstance().getNumProcesses() > 1) {
    // Build local score vectors (DBL_MAX sentinel for uncomputed models)
    int n = candidate_models.size();
    vector<double> local_lnL(n, -DBL_MAX);
    vector<double> local_BIC(n, DBL_MAX);
    vector<double> local_AIC(n, DBL_MAX);
    vector<double> local_AICc(n, DBL_MAX);

    for (int i = 0; i < n; i++)
        if (candidate_models[i].hasFlag(MF_DONE)) {
            local_lnL[i]  = candidate_models[i].logl;
            local_BIC[i]  = candidate_models[i].BIC_score;
            local_AIC[i]  = candidate_models[i].AIC_score;
            local_AICc[i] = candidate_models[i].AICc_score;
        }

    // Global reduce: each slot has exactly one real value, rest are sentinels
    // MPI_MIN for BIC/AIC (lower is better), MPI_MAX for lnL
    vector<double> global_lnL(n), global_BIC(n), global_AIC(n), global_AICc(n);
    MPI_Allreduce(local_lnL.data(),  global_lnL.data(),  n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_BIC.data(),  global_BIC.data(),  n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_AIC.data(),  global_AIC.data(),  n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_AICc.data(), global_AICc.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    // Fill in the models this rank did NOT evaluate
    for (int i = 0; i < n; i++)
        if (!candidate_models[i].hasFlag(MF_DONE)) {
            candidate_models[i].logl       = global_lnL[i];
            candidate_models[i].BIC_score  = global_BIC[i];
            candidate_models[i].AIC_score  = global_AIC[i];
            candidate_models[i].AICc_score = global_AICc[i];
            candidate_models[i].setFlag(MF_DONE);   // mark as filled
        }

    // Consolidate checkpoint: each rank sends its portion to rank 0
    MPIHelper::getInstance().gatherCheckpoint(&model_info);
}
#endif
```

### 4.2 Why `evaluateAll()` Is the Better Target (Not `test()`)

`test()` has interleaved early-stopping logic: after model `i`, it may set `MF_IGNORED`
on models `i+1..i+k` based on +R category score comparisons. With striped dispatch,
rank 0 evaluates models 0, N, 2N... and rank 1 evaluates 1, N+1, 2N+1... The early-stop
comparisons across adjacent models will see `DBL_MAX` sentinels from uncomputed slots
and will incorrectly filter or not filter models.

`evaluateAll()` has **no early-stop logic** — it runs all models and selects the best
at the end. This is exactly what model-level dispatch requires. The activation path:

```
runModelFinder()
  → if (params.openmp_by_model || nranks > 1):    ← ADD: || nranks > 1
      candidate_models.evaluateAll(...)
  → else:
      candidate_models.test(...)
```

With MPI + multiple ranks, force the `evaluateAll()` path. Each OMP thread still
handles one model (as today), but now the `num_threads` passed to `evaluate()` should
scale: `rank_threads = total_cores_per_node / nranks_per_node`.

### 4.3 Checkpoint Consolidation

Each rank's `out_model_info` (per-model checkpoint) needs to reach rank 0 to write
the `.model.gz` file. The existing `MPIHelper::gatherCheckpoint()` does this for
PartitionFinder. We reuse it:

```cpp
// In evaluateAll() at the end, inside #ifdef _IQTREE_MPI:
MPIHelper::getInstance().gatherCheckpoint(&model_info);
// Now rank 0 has the full checkpoint; workers already have partial results
// from the MPI_Allreduce above
```

### 4.4 Scaling Projection (Corrected)

With `xlarge_mf.fa` (10M patterns, 0% compression, 100 taxa):
- RAM per rank: 324 GB (from actual IQ-TREE log output)
- Gadi normalsr: 503 GB RAM/node → max **1 rank/node** feasible
- Per-model time: ~729 s/model at 4 nodes × 104 OMP

| Config | Nodes | Ranks | OMP/rank | Models/rank | MF wall time |
|--------|-------|-------|----------|-------------|--------------|
| Current (wasted) | 4 | 4 | 104 | 968 (redundant) | ~196 h |
| MF2 dispatch | 4 | 4 | 104 | 242 | ~49 h |
| MF2 dispatch | 16 | 16 | 104 | 61 | ~12 h |
| MF2 dispatch | 64 | 64 | 104 | 16 | ~3.2 h |
| MF2 dispatch | 121 | 121 | 104 | 9 | ~1.8 h |

For typical empirical compressed data (~100K unique patterns from 1M sites):
- RAM per rank drops to ~3.5 GB
- Can fit 26 ranks/node (104 cores ÷ 4 OMP)
- 4 nodes × 26 = 104 ranks: 968 ÷ 104 ≈ 10 models/rank → **~2 h wall time**

---

## 6. Key Files to Modify

All paths relative to `/scratch/um09/as1708/iqtree3-mf2/src/iqtree3/`.

### 5.1 Primary Target — `main/phylotesting.cpp`

**This is the only file containing ModelFinder logic.**

| Location | Change | Detail |
|---|---|---|
| Line ~1370 in `runModelFinder()` | Force `evaluateAll()` path when `nranks > 1` | Add `\|\| MPIHelper::getInstance().getNumProcesses() > 1` to the `openmp_by_model` condition |
| After `generate()` call (~line 1450) | New: `markRankModels()` | Stripe `MF_IGNORED_MPI` flag based on `rank % nranks` |
| End of `evaluateAll()` (~line 3530) | New: `gatherModelScores()` | `MPI_Allreduce` + fill uncomputed slots + `gatherCheckpoint` |
| `evaluateAll()` per-model `evaluate()` call (~line 3430) | Pass `rank_threads` | `num_threads / nranks_per_node` instead of full `num_threads` |

### 5.2 `main/phylotesting.h`

Add new flag constant beside existing ones (line 29–33):
```cpp
const int MF_IGNORED_MPI = 32;   // model assigned to a different MPI rank
```

### 5.3 `utils/MPIHelper.h` / `utils/MPIHelper.cpp`

Existing: `gatherCheckpoint()` — already does what we need for consolidation.
Existing: `broadcastCheckpoint()` — already broadcasts from rank 0 to all.

No new functions needed. The `MPI_Allreduce` calls are made directly in
`phylotesting.cpp` (matching the style of the existing calls at line 825).

### 5.4 `main/phyloanalysis.cpp`

Entry into `runModelFinder()`. No change needed — the MPI-awareness is encapsulated
entirely inside `runModelFinder()` / `evaluateAll()`.

---

## 7. Patch Style — Matching Existing Conventions

All existing MPI-conditional code uses the pattern:
```cpp
#ifdef _IQTREE_MPI
    if (MPIHelper::getInstance().getNumProcesses() > 1) {
        // ... MPI logic ...
    }
#endif
```

The `MPIHelper::getInstance()` singleton is the standard accessor.

Existing checkpoint sync pattern (from `modelfactory.cpp` line 1617):
```cpp
#ifdef _IQTREE_MPI
    if (syncChkPoint != nullptr) {
        syncChkPoint->masterSyncOtherChkpts();
    }
#endif
```

Our additions follow the same `#ifdef _IQTREE_MPI` / `getInstance()` / guarded
block pattern. No new preprocessor macros. No changes to the MPI communication model
outside ModelFinder (tree search MPI path is untouched).

---

## 8. Three-Level Parallelism After the Patch

```
Level 3 — MPI ranks:
    Rank 0 evaluates models { 0, N, 2N, ... }
    Rank 1 evaluates models { 1, N+1, 2N+1, ... }
    ...  (embarrassingly parallel — zero inter-rank comm during evaluation)
    One MPI_Allreduce (4× n_model doubles) at the end

Level 2 — OMP threads (per rank):
    rank_threads = ncores_per_node / nranks_per_node
    Each model: OMP parallel over site-pattern packets
    #pragma omp parallel for num_threads(rank_threads)

Level 1 — AVX-512 SIMD (per OMP thread):
    8 doubles per cycle in phylokernelnew.h
    Sapphire Rapids ZMM register width
```

This is the genuinely novel contribution: **model-level MPI dispatch integrated into
IQ-TREE's existing MPI/OMP/SIMD hierarchy**, not a standalone tool.

---

## 9. Implementation Phases

Each phase produces an independently testable build milestone. Phases are ordered so
that earlier phases are prerequisites for later ones, and correctness can be verified
before moving to the next. The analogy to prior patches: Phase 1 is the model-ownership
layer (like NUMA R1's data-initialisation patches — establishing *who owns what* before
any compute), Phase 2 is the hot dispatch path (like R2's hot-kernel patches — the
communication-critical loop at the core of the change), Phase 3 is resource tuning
(OMP thread budget, analogous to R2's PACKETS_PER_THREAD and num_packets tuning),
Phase 4 is correctness hardening (edge cases and checkpoint resume), and Phase 5 is
scale benchmarking.

---

### Phase 1 — Model Stripe Layer (Flag Infrastructure & Per-Rank Assignment)

**Scope:** Establish per-rank model ownership *before* any evaluation begins. No MPI
communication is added in this phase — only the marking logic that tells each rank
which models it owns. Mirrors R1's first-touch initialisation: correct ownership
must be established before the hot path runs.

**Changes:**

1. **`main/phylotesting.h`** — add the new flag constant after `MF_DONE = 16`:
   ```cpp
   const int MF_IGNORED_MPI = 32;  // model assigned to a different MPI rank
   ```
   Verify no collision (existing: 1, 2, 4, 8, 16 → 32 is the next power of two).

2. **`main/phylotesting.cpp` — `runModelFinder()`** — add `markRankModels()` block
   immediately after `candidate_models.generate()` returns and before the
   `evaluateAll()`/`test()` branch (~line 1450):
   ```cpp
   #ifdef _IQTREE_MPI
   if (MPIHelper::getInstance().getNumProcesses() > 1 && !params.model_test_and_tree) {
       int my_rank = MPIHelper::getInstance().getProcessID();
       int nranks  = MPIHelper::getInstance().getNumProcesses();
       int my_count = 0;
       for (int i = 0; i < (int)candidate_models.size(); i++) {
           if (i % nranks != my_rank)
               candidate_models[i].setFlag(MF_IGNORED_MPI);
           else
               my_count++;
       }
       cout << "MF-MPI: rank " << my_rank << "/" << nranks
            << " assigned " << my_count << "/" << candidate_models.size()
            << " models" << endl;
       // Force evaluateAll() path: test() early-stop logic breaks with striped gaps
       params.openmp_by_model = true;
   }
   #endif
   ```

3. **Verify `MF_IGNORED_MPI` is respected** by `evaluateAll()` — read `getNextModel()`
   to confirm `hasFlag(MF_IGNORED)` check covers all flag values ≥ 2, or add
   an explicit `hasFlag(MF_IGNORED_MPI)` check if needed.

**Build command:**
```bash
cd /scratch/um09/as1708/iqtree3-mf2/build-profiling-mpi
make -j16 iqtree3-mpi 2>&1 | grep -E "error:|warning:|iqtree3-mpi"
```

**Milestone test (correctness NOT expected — Phase 2 adds the gather):**
```bash
# Run with 4 ranks, inspect per-rank model assignment in logs
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF -T 2 --prefix p1_test -seed 42 2>&1 \
    | grep "MF-MPI:"
# Expected output (4 ranks, 968 models):
#   MF-MPI: rank 0/4 assigned 242/968 models
#   MF-MPI: rank 1/4 assigned 242/968 models
#   MF-MPI: rank 2/4 assigned 242/968 models
#   MF-MPI: rank 3/4 assigned 242/968 models
```

Each rank should log exactly `ceil(968/4)` models. The final best-fit model will be
wrong (each rank picks independently from its partial set) — that is expected at this
phase. The phase is complete when the model counts in the log are correct.

---

### Phase 2 — MPI Gather Integration (Score Consolidation & Checkpoint Merge)

**Scope:** Add the MPI communication at the *end* of `evaluateAll()` that collects
per-rank scores and merges checkpoints. This is the hot-path communication step —
the only cross-rank synchronisation for the entire ModelFinder phase. Mirrors R2's
hot-kernel integration: the critical path that makes the data partitioning from
Phase 1 produce a correct global result.

**Changes:**

1. **`main/phylotesting.cpp` — end of `CandidateModelSet::evaluateAll()`** — add
   `gatherModelScores()` block after the OMP parallel region closes and after
   `model_info.dump()`:
   ```cpp
   #ifdef _IQTREE_MPI
   if (MPIHelper::getInstance().getNumProcesses() > 1) {
       int n = (int)size();
       // Sentinels: only the owning rank has real values, others have ±DBL_MAX
       vector<double> local_lnL(n, -DBL_MAX), local_BIC(n, DBL_MAX),
                      local_AIC(n, DBL_MAX),  local_AICc(n, DBL_MAX);
       for (int i = 0; i < n; i++)
           if (at(i).hasFlag(MF_DONE)) {
               local_lnL[i]  = at(i).logl;
               local_BIC[i]  = at(i).BIC_score;
               local_AIC[i]  = at(i).AIC_score;
               local_AICc[i] = at(i).AICc_score;
           }
       vector<double> g_lnL(n), g_BIC(n), g_AIC(n), g_AICc(n);
       // Each slot has exactly one real value; reduce collects it from the owning rank
       MPI_Allreduce(local_lnL.data(),  g_lnL.data(),  n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
       MPI_Allreduce(local_BIC.data(),  g_BIC.data(),  n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
       MPI_Allreduce(local_AIC.data(),  g_AIC.data(),  n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
       MPI_Allreduce(local_AICc.data(), g_AICc.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
       // Populate models that this rank did NOT evaluate
       for (int i = 0; i < n; i++)
           if (!at(i).hasFlag(MF_DONE)) {
               at(i).logl       = g_lnL[i];
               at(i).BIC_score  = g_BIC[i];
               at(i).AIC_score  = g_AIC[i];
               at(i).AICc_score = g_AICc[i];
               at(i).setFlag(MF_DONE);
           }
       // Merge checkpoints: each rank's .model.gz entries go to rank 0
       MPIHelper::getInstance().gatherCheckpoint(model_info_ptr);
       MPIHelper::getInstance().broadcastCheckpoint(model_info_ptr);
       cout << "MF-MPI: gather complete, " << n << " model scores consolidated" << endl;
   }
   #endif
   ```
   Note: `model_info_ptr` must be passed through to `evaluateAll()` — verify the
   existing signature and add the pointer if it is not already an argument.

2. **Confirm `model_info.setFileName("")` is active for worker ranks** at ~line 1405
   of `runModelFinder()` (already in the existing MPI path) — prevents workers from
   clobbering rank 0's `.model.gz` on disk.

**Build and milestone test (correctness required):**
```bash
cd /scratch/um09/as1708/iqtree3-mf2/build-profiling-mpi
make -j16 iqtree3-mpi

# Reference: 1 rank
mpirun -np 1 ./iqtree3-mpi -s ../example/example.phy -m MF -T 4 \
    --prefix ref_1r -seed 42 --redo
# Dispatch: 4 ranks
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF -T 4 \
    --prefix test_4r -seed 42 --redo

# Must produce identical lines:
grep "Best-fit model\|Bayesian information criterion" ref_1r.log test_4r.log
# Must differ only in wall time (4-rank should be ~4× faster):
grep "Wall-clock time for ModelFinder" ref_1r.log test_4r.log
```

The phase is complete when the best-fit model and BIC score are **identical** between
1-rank and 4-rank runs on `example.phy`.

---

### Phase 3 — Thread Budget Partitioning (OMP Threads per Rank)

**Scope:** When multiple MPI ranks share a single node, each rank must receive only
its fair share of the node's OMP threads so they do not over-subscribe the CPU.
Without this, 4 ranks on 1 node would each request 104 OMP threads → 416 concurrent
threads on 104 cores. Mirrors R2's `PACKETS_PER_THREAD` / `num_packets` tuning:
adjusting the intra-node parallelism budget after the inter-node dispatch is correct.

**Changes:**

1. **`main/phylotesting.cpp` — `runModelFinder()` inside the Phase 1 block** — after
   `params.openmp_by_model = true`, compute `rank_threads`:
   ```cpp
   // Determine how many OMP threads this rank may use.
   // When 1 rank per node (typical for xlarge, 324 GB/rank), no adjustment needed.
   // When >1 rank per node (compressed data), share the thread pool equally.
   int ranks_per_node = 1;   // default: xlarge case, 1 rank fills the node RAM
   // Count how many ranks share localhost by checking hostnames if needed,
   // or accept ranks_per_node as a new CLI param --mpi-ranks-per-node (default 1).
   int rank_threads = max(1, params.num_threads / ranks_per_node);
   params.num_threads = rank_threads;  // scoped to this MF phase only
   ```
   For the initial xlarge benchmark (1 rank/node), `ranks_per_node = 1` →
   `rank_threads = 104` → no change from current behaviour. The multi-rank/node
   path is activated by `--mpi-ranks-per-node N`.

2. **Add `--mpi-ranks-per-node` CLI parameter** to `main/main.cpp` and
   `utils/tools.h` (pattern: search for any existing `num_threads` CLI wiring and
   mirror it). This is a single integer param, default 1.

3. **Restore `params.num_threads` after `evaluateAll()` returns** to avoid
   affecting the subsequent tree-search phase:
   ```cpp
   params.num_threads = orig_num_threads;  // restore before tree search
   ```

**Milestone test:**
```bash
# Simulate 4 ranks on a single 8-core login node with 2 threads total budget
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF -T 8 \
    --mpi-ranks-per-node 4 --prefix p3_test -seed 42 --redo
# Verify: each rank runs with 8/4 = 2 OMP threads
# Check: OMP_NUM_THREADS reported in logs = 2, htop/ps shows ≤8 threads total
grep "Number of threads" p3_test.log

# Confirm correctness is preserved with non-default thread count:
grep "Best-fit model" p3_test.log ref_1r.log   # must match
```

Phase is complete when: correct model with `--mpi-ranks-per-node 4` AND CPU
utilization stays within the per-rank thread budget.

---

### Phase 4 — Correctness Hardening (Edge Cases & Checkpoint Resume)

**Scope:** Stress the patch against scenarios that differ from the standard DNA `/
normal data / single restart` path. These exercise the flag and gather logic on
data paths that Phase 1–3 did not explicitly test.

**Test suite (run in order, each must match 1-rank reference):**

**4a — SNP / ASC data** (0% invariant sites, `+ASC` rate models active instead of `+I`):
```bash
# SNP alignment has frac_invariant_sites = 0.0 → different rate options array
mpirun -np 1 ./iqtree3-mpi -s test_snp.phy -m MF --prefix ref_snp -seed 1 --redo
mpirun -np 4 ./iqtree3-mpi -s test_snp.phy -m MF --prefix test_snp -seed 1 --redo
grep "Best-fit model" ref_snp.log test_snp.log
```

**4b — Protein data** (28 subst × 2 freq × 22 rate = 1232 models; bigger gather vector):
```bash
mpirun -np 1 ./iqtree3-mpi -s test_protein.phy -m MF --prefix ref_aa -seed 1 --redo
mpirun -np 4 ./iqtree3-mpi -s test_protein.phy -m MF --prefix test_aa -seed 1 --redo
grep "Best-fit model" ref_aa.log test_aa.log
```

**4c — Restricted model set** (non-multiple-of-nranks count, tests load-balance edge):
```bash
# 6 models × 2 freq × 22 rate = 264; 264 / 4 = 66 per rank (even division)
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF --mset mrbayes \
    --prefix test_mrb -seed 1 --redo
# 3 models × 2 freq × 22 rate = 132; 132 / 4 = 33 per rank (even)
# 5 models → 5/4 = ranks 0,1 get 2, ranks 2,3 get 1 (odd test)
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF --mset GTR,HKY,JC,TN,TVM \
    --prefix test_odd -seed 1 --redo
grep "Best-fit model" test_mrb.log test_odd.log   # must match 1-rank runs
```

**4d — Checkpoint resume** (kill mid-run, verify restart picks up correctly):
```bash
# Run 4-rank dispatch on example.phy but kill after ~30 models via SIGTERM
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF -T 4 \
    --prefix resume_test -seed 42 &
sleep 60 && kill $!   # approx 30 models at ~2s/model on login node

# Resume: --redo is NOT set, so checkpoint is used
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF -T 4 \
    --prefix resume_test -seed 42
# Must produce same best-fit model as full 1-rank run:
grep "Best-fit model" resume_test.log ref_1r.log
```

Phase is complete when all four sub-tests pass with identical best-fit models.

---

### Phase 5 — Scale Benchmarking & CHANGELOG

**Scope:** Measure the actual wall-time improvement on the xlarge worst-case dataset
(PBS 167977883 baseline) and on a realistic compressed dataset. Document in CHANGELOG.

**5a — Create PBS submission script** `gadi-ci/run_xlarge_r2_mf2_dispatch.sh`:
```bash
#!/bin/bash
#PBS -N iq-mf2-dispatch
#PBS -l ncpus=416,mem=2048GB,walltime=6:00:00
#PBS -l storage=scratch/um09+gdata/um09
#PBS -q normalsr
#PBS -l wd

module load intel-mpi/2021.13.1 intel/2024.2.0

BINARY=/scratch/um09/as1708/iqtree3-mf2/build-profiling-mpi/iqtree3-mpi
ALN=/scratch/um09/as1708/iqtree3-mf2/benchmarks/xlarge_mf/xlarge_mf.fa

# 4 nodes × 104 OMP, 1 rank/node (324 GB RAM limit)
mpirun -np 4 --map-by node --bind-to numa \
    $BINARY -s $ALN -m MF -T 104 --mpi-ranks-per-node 1 \
    --prefix mf2_dispatch_4r -seed 1
```

**5b — Run and record results:**

| Metric | Baseline (PBS 167977883) | MF2 dispatch | Δ |
|--------|--------------------------|--------------|---|
| Nodes / ranks / OMP | 4 / 4 / 104 | 4 / 4 / 104 | same |
| Models per rank | 968 (redundant) | 242 | −75% |
| Wall time (MF phase) | ~196 h (projected) | ~49 h (projected) | 4× |
| Best-fit model | — | must match | ✓ |
| RAM per rank | 324 GB | 324 GB | same |

**5c — CHANGELOG entry** (add to `CHANGELOG.md` under `Unreleased`):
```markdown
### (u) ModelFinder MPI model-level dispatch (MF2)

Patch: `main/phylotesting.h`, `main/phylotesting.cpp`

Previously, all N MPI ranks evaluated all 968 DNA models redundantly during
the ModelFinder phase. Adds per-rank model stripe assignment (`MF_IGNORED_MPI`
flag) and a post-evaluation MPI_Allreduce gather so each rank evaluates only
ceil(968/N) models. Reuses existing `gatherCheckpoint`/`broadcastCheckpoint`
from `MPIHelper` for checkpoint consolidation. Forces `evaluateAll()` path
(no early-stop, OMP-parallel) when nranks > 1.

Measured improvement (Gadi normalsr, 10M-site alignment, 4 nodes × 104 OMP):
  Baseline: 4 ranks, 968 models each (redundant) → projected ~196 h
  MF2:      4 ranks, 242 models each             → projected  ~49 h (4×)

Scaling: near-linear with rank count (embarrassingly parallel during evaluation;
one MPI_Allreduce × 4 score arrays at end, ~200 KB total communication).

See: design/modelfinder-mpi-dispatch.md
```

Phase is complete when the PBS job completes and the wall-time improvement is
documented. The design doc `Section 4.4` scaling table is updated with measured
(not projected) values.

---

## 10. Related Files Summary

| File | Lines | Role | Change |
|------|-------|------|--------|
| `main/phylotesting.cpp` | 7021 | **All** ModelFinder code | Core patch — Steps 3+4 |
| `main/phylotesting.h` | 845 | Class declarations + flag constants | Add `MF_IGNORED_MPI = 32` |
| `utils/MPIHelper.h` | 211 | MPI singleton wrappers | No change needed |
| `utils/MPIHelper.cpp` | 221 | `gatherCheckpoint`, `broadcastCheckpoint` impl | No change needed |
| `main/phyloanalysis.cpp` | 6299 | Calls `runModelFinder()` | No change needed |
| `model/modelfactory.cpp` | ~1800 | `optimizeParameters()` (per-model loop body) | No change needed |
| `tree/phylotreesse.cpp` | ~580 | `setNumThreads()` | No change needed |
| `tree/phylokernelnew.h` | ~3600 | AVX-512 SIMD kernel | No change needed |
| `gadi-ci/run_xlarge_r2_mf2_dispatch.sh` | new | PBS benchmark script | Create in Step 7 |

---

## 11. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Early-stop in `test()` breaks with strided gaps | **Eliminated** | Force `evaluateAll()` path when `nranks > 1` (no early-stop in that path) |
| Memory exceeds node capacity (0% compression) | High for xlarge | 324 GB/rank × 4 ranks = 1296 GB; need `mem=2048GB` PBS request; limit to 1 rank/node |
| Score mismatch across ranks | Low | Identical alignment + identical starting tree + `--seed`; only OMP non-determinism in last digit |
| Checkpoint race: workers write to same `.model.gz` | **Eliminated** | Workers call `model_info.setFileName("")` at line ~1405 (already in code for MPI workers) |
| `gatherCheckpoint` is slow for large checkpoints | Low | 968 models × ~200 bytes = ~200 KB total; negligible vs 729 s/model |
| Upstream ModelFinder2 supersedes this | Medium | Track issues #130–134; patch can be upstreamed or rebased |

---

## 12. References

1. Darriba D et al. (2012) jModelTest 2. *Nat Methods* 9(8):772. doi:10.1038/nmeth.2109
2. Darriba D et al. (2020) ModelTest-NG. *Mol Biol Evol* 37(1):291. doi:10.1093/molbev/msz189
3. Kalyaanamoorthy S et al. (2017) ModelFinder. *Nat Methods* 14(6):587. doi:10.1038/nmeth.4285
4. Lefort V et al. (2017) SMS: Smart Model Selection. *Mol Biol Evol* 34(9):2422. doi:10.1093/molbev/msx149
5. IQ-TREE3 GitHub issues #130–134 (label: modelfinder2). https://github.com/iqtree/iqtree3/issues
6. MOOSE wiki. https://codeberg.org/amkozlov/raxml-ng/wiki/Automatic-model-selection-(MOOSE)
