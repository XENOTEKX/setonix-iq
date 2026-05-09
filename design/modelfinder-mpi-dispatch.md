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

## 2. Source Code Architecture — What Really Happens Today

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

## 3. The Pattern Parallelism Hierarchy (Existing)

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

## 4. Proposed Architecture: Rank-Striped Model Dispatch

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

## 5. Key Files to Modify

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

## 6. Patch Style — Matching Existing Conventions

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

## 7. Three-Level Parallelism After the Patch

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

## 8. Step-by-Step Implementation Plan

### Step 0 — Baseline measurement ✅
- 968 models, 729 s/model at 4 nodes × 104 OMP (PBS 167977883)
- 9 JC models completed, all MPI ranks ran identical computation
- 324 GB RAM required per node (10M patterns, 0% compression)

### Step 1 — Read and annotate `phylotesting.cpp`
```bash
cd /scratch/um09/as1708/iqtree3-mf2/src/iqtree3
# Trace the exact evaluateAll() code path
grep -n "evaluateAll\|openmp_by_model\|getNextModel\|MF_DONE\|MF_IGNORED" \
    main/phylotesting.cpp | head -40
# Confirm no MPI inside evaluate()
grep -n "MPI_\|_IQTREE_MPI" main/phylotesting.cpp | grep -v "PartitionFinder\|Partition"
```

### Step 2 — Add `MF_IGNORED_MPI = 32` flag to `phylotesting.h`
One line. Verify existing flag values don't collide (existing: 2, 4, 8, 16 → 32 is next).

### Step 3 — Add `markRankModels()` block in `runModelFinder()` (line ~1450)
After `candidate_models.generate()` returns, before the `evaluateAll()` / `test()` branch:
```cpp
#ifdef _IQTREE_MPI
if (MPIHelper::getInstance().getNumProcesses() > 1 && !params.model_test_and_tree) {
    int my_rank = MPIHelper::getInstance().getProcessID();
    int nranks  = MPIHelper::getInstance().getNumProcesses();
    for (int i = 0; i < (int)candidate_models.size(); i++)
        if (i % nranks != my_rank)
            candidate_models[i].setFlag(MF_IGNORED_MPI);
    // Force evaluateAll() path — test() has early-stop that breaks with gaps
    params.openmp_by_model = true;
    // Adjust thread count so total cores/node is preserved
    // (caller sets num_threads = total OMP; here we share across ranks on same node)
    // NOTE: thread count adjustment is handled in evaluateAll() via rank_threads
}
#endif
```

### Step 4 — Add `gatherModelScores()` block at end of `evaluateAll()`
After the OMP parallel block closes and the final `model_info.dump()` call:
```cpp
#ifdef _IQTREE_MPI
if (MPIHelper::getInstance().getNumProcesses() > 1) {
    // ... MPI_Allreduce for lnL, BIC, AIC, AICc as shown in Section 4.1 ...
    MPIHelper::getInstance().gatherCheckpoint(&model_info);
    MPIHelper::getInstance().broadcastCheckpoint(&model_info);
}
#endif
```

### Step 5 — Correctness test on example.phy
```bash
cd /scratch/um09/as1708/iqtree3-mf2/build-profiling-mpi
make -j8

# 1-rank reference
mpirun -np 1 ./iqtree3-mpi -s ../example/example.phy -m MF -T 4 --prefix ref_1rank -seed 42

# 4-rank dispatch
mpirun -np 4 ./iqtree3-mpi -s ../example/example.phy -m MF -T 4 --prefix test_4rank -seed 42

# Must match:
grep "Best-fit model\|BIC" ref_1rank.log test_4rank.log
```

### Step 6 — Memory measurement at scale
```bash
# RSS at 1 rank (full-alignment mode, no sharing)
/usr/bin/time -v mpirun -np 1 ./iqtree3-mpi \
    -s benchmarks/xlarge_mf/xlarge_mf.fa -m MF -T 104 --prefix mem_test 2>&1 \
    | grep "Maximum resident"
# Compare to 4-rank run: should be ~4× total (4× independent full-aln copies)
```

### Step 7 — PBS submission script `gadi-ci/run_xlarge_r2_mf2_dispatch.sh`
```bash
#!/bin/bash
#PBS -N iq-mf2-dispatch
#PBS -l ncpus=416,mem=2048GB,walltime=6:00:00
#PBS -l storage=scratch/um09+gdata/um09
#PBS -q normalsr
cd /scratch/um09/as1708/iqtree3-mf2/gadi-ci
module load intel-mpi/2021.13.1 intel/2024.2.0
# 4 ranks × 104 OMP = 416 cores; each rank holds full 324 GB alignment
mpirun -np 4 --hostfile hostfile.txt \
    /scratch/um09/as1708/iqtree3-mf2/build-profiling-mpi/iqtree3-mpi \
    -s /scratch/um09/as1708/iqtree3-mf2/benchmarks/xlarge_mf/xlarge_mf.fa \
    -m MF -T 104 --prefix mf2_dispatch_4r -seed 1
```

### Step 8 — Benchmark and CHANGELOG entry
Compare wall times, verify best-fit model is identical, record scaling vs model count.

---

## 9. Related Files Summary

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

## 10. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Early-stop in `test()` breaks with strided gaps | **Eliminated** | Force `evaluateAll()` path when `nranks > 1` (no early-stop in that path) |
| Memory exceeds node capacity (0% compression) | High for xlarge | 324 GB/rank × 4 ranks = 1296 GB; need `mem=2048GB` PBS request; limit to 1 rank/node |
| Score mismatch across ranks | Low | Identical alignment + identical starting tree + `--seed`; only OMP non-determinism in last digit |
| Checkpoint race: workers write to same `.model.gz` | **Eliminated** | Workers call `model_info.setFileName("")` at line ~1405 (already in code for MPI workers) |
| `gatherCheckpoint` is slow for large checkpoints | Low | 968 models × ~200 bytes = ~200 KB total; negligible vs 729 s/model |
| Upstream ModelFinder2 supersedes this | Medium | Track issues #130–134; patch can be upstreamed or rebased |

---

## 11. References

1. Darriba D et al. (2012) jModelTest 2. *Nat Methods* 9(8):772. doi:10.1038/nmeth.2109
2. Darriba D et al. (2020) ModelTest-NG. *Mol Biol Evol* 37(1):291. doi:10.1093/molbev/msz189
3. Kalyaanamoorthy S et al. (2017) ModelFinder. *Nat Methods* 14(6):587. doi:10.1038/nmeth.4285
4. Lefort V et al. (2017) SMS: Smart Model Selection. *Mol Biol Evol* 34(9):2422. doi:10.1093/molbev/msx149
5. IQ-TREE3 GitHub issues #130–134 (label: modelfinder2). https://github.com/iqtree/iqtree3/issues
6. MOOSE wiki. https://codeberg.org/amkozlov/raxml-ng/wiki/Automatic-model-selection-(MOOSE)
