# ModelFinder MPI Model-Level Dispatch — Architecture & Implementation Plan

**Branch:** `modelfinder2`  
**IQ-TREE base:** `gadi-spr-r2-avx512` (v3.1.2 + NUMA R1/R2 + AVX-512 patches)  
**Working source:** `/scratch/um09/as1708/iqtree3-mf2/src/iqtree3`  
**Date:** 2026-05-10  

---

## 1. Problem Statement

IQ-TREE's ModelFinder evaluates substitution models **sequentially**: all MPI ranks
cooperate on a single model at a time via `MPI_Allreduce` to sum partial-likelihood
contributions across pattern slices. Adding more nodes reduces time-per-model but
**does not reduce the number of models evaluated** — model count stays fixed at 968 for
a standard DNA search.

Observed on Gadi 4-node run (PBS 167977883, 416 cores):

| Metric | Value |
|--------|-------|
| Models tested | 9 / 968 (0.93%) |
| Rate | ~729 s/model |
| Projected full run | ~196 h |
| Nodes that help | Adding nodes reduces s/model but **not** model count |

At 10 M distinct patterns (0% compression on `xlarge_mf.fa`), each per-model tree
optimisation is bandwidth-limited at ~15.7 GB of partial-LH data per rank, giving a
hard floor even with perfect OMP scaling.

---

## 2. Prior Art & Scientific Basis

### 2.1 jModelTest 2 (Darriba et al. 2012, *Nat. Methods*)

The original proof-of-concept for **model-level MPI dispatch** in phylogenetics.
jModelTest 2 assigns disjoint subsets of candidate models to MPI ranks. Each rank
evaluates its own subset independently using its local alignment copy; results are
gathered to rank 0 for BIC selection. Demonstrated near-linear scaling with rank count.

> Darriba D, Taboada GL, Doallo R, Posada D (2012) jModelTest 2: more models, new
> heuristics and parallel computing. *Nat Methods* 9(8):772.
> doi:10.1038/nmeth.2109

### 2.2 ModelTest-NG (Darriba et al. 2020, *Mol. Biol. Evol.*)

Full reimplementation of jModelTest/ProtTest in C++ using libpll-2/pll-modules with
AVX2 SIMD kernels. Supports PThreads + MPI with model-level dispatch. Results:

- **DNA accuracy**: 81% true-model recovery vs ModelFinder's 70% on simulated data
- **Speed**: similar to ModelFinder per thread, but MPI-scales with rank count
- **Scalability**: near-linear from 1 to N ranks on model count axis

Build with MPI: `cmake -DENABLE_MPI=ON ..`  
GitHub: <https://github.com/ddarriba/modeltest>  
Paper: doi:10.1093/molbev/msz189

### 2.3 RAxML-NG v2.0 MOOSE

MOOSE (MOdel Optimization and SElection) was merged into RAxML-NG v2.0 from the
`modeltest2` development branch (2024). Uses the same coraxlib/libpll backend as
ModelTest-NG. The "Parallelization" section of the MOOSE wiki is marked TODO,
so the MPI dispatch path may not be fully exposed in the CLI yet.

Reference: <https://codeberg.org/amkozlov/raxml-ng/wiki/Automatic-model-selection-(MOOSE)>

### 2.4 IQ-TREE3 ModelFinder2 (upstream, active)

GitHub issues #130–134 on `iqtree/iqtree3` are all labelled `modelfinder2`, covering
new model types and documentation. The IQ-TREE team is aware of the scaling limitation
and developing ModelFinder2, but no timeline or parallel-architecture description is
published. This work proceeds independently and may be upstreamed.

---

## 3. Architecture of the Proposed Patch

### 3.1 Current Communication Pattern (Serial Model Loop)

```
for each model_i in candidate_models:               ← sequential, 968 iterations
    foreach MPI rank r:
        evaluate partial_lh for pattern_slice[r]    ← O(nptn/nranks * nstates * ncats)
    MPI_Allreduce(partial_lh_sum)                    ← barrier per model
    rank 0: update BIC table
```

Each model evaluation forces a global synchronisation. With 0% pattern compression,
this is ~729 s/model at 4 nodes.

### 3.2 Target Communication Pattern (Model-Level Dispatch)

```
Phase 1 — distribute models:
    rank r evaluates models: { i | i % nranks == r }  ← embarrassingly parallel
    no inter-rank communication during evaluation

Phase 2 — gather results:
    MPI_Gather( {model_name, lnL, df, BIC}, ... ) → rank 0
    rank 0 selects best BIC
    MPI_Bcast(best_model_index)

Phase 3 — tree search uses best_model (existing code, unchanged)
```

No `MPI_Allreduce` inside the model loop. Each rank runs its OMP threads over the
**full** alignment (not a pattern slice). One `MPI_Gather` + `MPI_Bcast` at the end.

### 3.3 Scaling Projection

With model-level dispatch and N ranks, each rank evaluates `ceil(968/N)` models.
Per-model time stays at ~729 s (same OMP threads, same data), but models run in
parallel across ranks.

| Config | Ranks | Models/rank | Projected MF time |
|--------|-------|-------------|-------------------|
| 4 nodes, 1 rank/node, 104 OMP | 4 | 242 | ~49 h |
| 4 nodes, 8 ranks/node, 13 OMP | 32 | 31 | ~6.2 h |
| 4 nodes, 16 ranks/node, 6 OMP | 64 | 16 | ~3.2 h |
| 4 nodes, 26 ranks/node, 4 OMP | 104 | 10 | ~2.0 h |
| 8 nodes, 26 ranks/node, 4 OMP | 208 | 5 | ~1.0 h |

Best operating point: **~26 ranks/node × 4 OMP threads** on Gadi normalsr (104 cores/node).  
Memory per rank: 200 taxa × 10 M patterns × 4 states × 8 bytes × ~3 buffers ≈ **~190 GB/rank**.

> **Memory is the binding constraint.** Gadi normalsr nodes have 256 GB RAM, so
> `ceil(190/256) = 1` rank per node. With pattern compression on a real dataset,
> the per-rank memory is typically 10–100× lower.

For the `xlarge_mf.fa` 0%-compression worst case: 1 rank/node (4 ranks total) →
242 models/rank → ~49 h. Not a breakthrough at 4 nodes for this pathological dataset,
but scales linearly to ~3 h at 64 nodes.

For a typical compressed dataset (e.g. 1–5% compression, ~10K–50K distinct patterns):
- Per-rank memory drops to ~1–5 GB
- Can run 8–26 ranks/node → 3 h at 4 nodes for standard analyses

---

## 4. Key Files to Modify

All paths relative to `/scratch/um09/as1708/iqtree3-mf2/src/iqtree3/`.

### 4.1 Primary target — `model/modelfinder.cpp`

Contains the main model evaluation loop. Search for:
```cpp
// The candidate model loop that calls optimizeModelParameters / computeLikelihood
// and records BIC/AIC scores in the model checkpoint.
```

The model loop is in `ModelFinder::findBestFitModel()` (or equivalent). The dispatch
change replaces the sequential loop with rank-striped iteration:

```cpp
// BEFORE (conceptual):
for (int i = 0; i < (int)model_names.size(); i++) {
    double score = evaluateModel(model_names[i]);
    recordScore(i, score);
}

// AFTER:
int my_rank, num_ranks;
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

for (int i = my_rank; i < (int)model_names.size(); i += num_ranks) {
    double score = evaluateModelLocal(model_names[i]);  // full alignment, local OMP only
    local_results[i] = score;
}

// Gather phase:
MPI_Allreduce(local_results, global_results, model_names.size(),
              MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
// MPI_MAX works because only one rank writes each slot (others leave at -∞ sentinel)
if (my_rank == 0) selectBestModel(global_results);
MPI_Bcast(&best_model_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

### 4.2 Secondary target — `tree/phylotree.cpp` / `phylotreesse.cpp`

**Current MPI mode**: `MPI_Allreduce` inside `computeLikelihoodBranchNaive()` sums
partial-LH over pattern slices held by different ranks. The alignment is split across
ranks (`MPI_Scatterv` at startup).

**Required change for model-level dispatch**: each rank must hold the **full alignment**
during the ModelFinder phase, then resume split-alignment mode for tree search.

Two implementation options:

1. **Mode flag** (preferred): add `bool modelfinder_mode` to `PhyloTree`. When true,
   disable pattern-splitting in `distributePatterns()` and disable the `MPI_Allreduce`
   inside `computeLikelihoodBranch`. Each rank computes lnL independently.

2. **Separate tree object**: create a lightweight `PhyloTree` clone per rank for the
   model-selection phase with `setNumThreads(omp_rank_threads)` and no MPI comms,
   then destroy it and re-enter standard MPI mode for tree search.

Option 2 is safer for a first implementation — it avoids touching the hot path of the
existing MPI likelihood kernel.

### 4.3 `main/iqtreemain.cpp` / `main/modelfinder_main.cpp`

Entry points that call `ModelFinder::findBestFitModel()`. May need to restructure the
MPI initialisation so all ranks enter the ModelFinder phase before splitting patterns
for tree search.

### 4.4 `utils/MPIHelper.h` / `utils/MPIHelper.cpp`

Helper classes wrapping `MPI_Bcast`, `MPI_Gather`, etc. New helpers needed:
- `MPIHelper::gatherDouble(local_arr, n)` — returns global array on rank 0
- `MPIHelper::broadcastInt(val)` — broadcast best model index

---

## 5. Step-by-Step Implementation Plan

### Step 0 — Baseline measurement (already done)
- ✅ Confirmed 968 models, 729 s/model at 4 nodes × 104 OMP
- ✅ Checkpoint shows 9 JC models completed before cancellation
- ✅ Identified sequential loop as bottleneck

### Step 1 — Understand current ModelFinder call graph
```bash
cd /scratch/um09/as1708/iqtree3-mf2/src/iqtree3
grep -rn "findBestFitModel\|ModelFinder\|MF_TESTNEW" model/ main/ --include="*.cpp" | head -40
grep -rn "MPI_Allreduce\|MPI_Scatter\|distributePattern" tree/ utils/ --include="*.cpp" | head -20
```
Trace from `main()` → ModelFinder entry → model loop → likelihood call → MPI barrier.

### Step 2 — Implement `evaluateModelLocal()` in `modelfinder.cpp`
A wrapper that sets `modelfinder_mode = true` on the tree before calling the standard
`evaluateModel()`, disabling the MPI pattern-scatter and using local full-alignment
likelihood. Unit test: single rank, result must match existing output.

### Step 3 — Add rank-striped model loop
Replace the sequential `for (int i = 0; i < n; i++)` with the striped version.
Add `local_score_arr[n]` initialised to `-1e300` (sentinel). After the loop,
`MPI_Allreduce` with `MPI_MAX` to reconstruct the complete score array on all ranks.

### Step 4 — Verify correctness on small dataset
```bash
mpirun -np 4 ./iqtree3 -s example/example.phy -m MF --prefix mf_test_4ranks
# Compare best-fit model to 1-rank reference:
mpirun -np 1 ./iqtree3 -s example/example.phy -m MF --prefix mf_test_1rank
diff mf_test_4ranks.log mf_test_1rank.log | grep "Best-fit model"
```

### Step 5 — Memory profiling at scale
Run with Valgrind massif or `/usr/bin/time -v` on `xlarge_mf.fa` with 1 rank to
measure peak RSS in full-alignment mode. Compare with existing 4-rank split mode.
Determine feasible ranks-per-node for Setonix (256 GB) and Gadi normalsr (256 GB).

### Step 6 — Integrate with checkpoint
The existing `.model.gz` checkpoint stores per-model lnL values. In model-level
dispatch mode, each rank writes only its own models' entries. On restart, all ranks
read the full checkpoint and skip already-evaluated models regardless of which rank
originally computed them. The checkpoint format is unchanged.

### Step 7 — PBS submission script
New script `gadi-ci/run_xlarge_r2_mf2_model_dispatch.sh`:
```bash
#PBS -l ncpus=416,mem=1000GB,walltime=4:00:00
#PBS -l storage=scratch/um09+gdata/um09
module load intel-mpi/2021.13.1 intel/2024.2.0
mpirun -np 208 --map-by slot:pe=2 \
  ./iqtree3-mf2 -s xlarge_mf.fa -m MF -T 2 --prefix mf2_dispatch
```
208 ranks × 2 OMP = 416 cores. Per-rank data = full alignment at ~2 OMP threads.

### Step 8 — Benchmark & CHANGELOG entry
Compare:
- Wall time for ModelFinder phase: dispatch vs sequential
- Model selection accuracy: same best-fit model?
- Memory overhead: N × full_alignment vs existing 1 × full_alignment

---

## 6. Related Files Summary

| File | Role | Change needed |
|------|------|---------------|
| `model/modelfinder.cpp` | Model loop | Rank-striped iteration + MPI_Allreduce gather |
| `model/modelfinder.h` | Header | Add `evaluateModelLocal()` declaration |
| `tree/phylotree.cpp` | Tree setup | `modelfinder_mode` flag, disable pattern-split |
| `tree/phylotree.h` | Header | `bool modelfinder_mode` member |
| `tree/phylotreesse.cpp` | Likelihood kernel | Respect `modelfinder_mode` in `setNumThreads` |
| `utils/MPIHelper.h` | MPI wrappers | `gatherDouble`, `broadcastInt` helpers |
| `main/iqtreemain.cpp` | Entry point | Ensure all ranks enter MF phase with full aln |
| `gadi-ci/run_xlarge_r2_mf2_model_dispatch.sh` | PBS script | New benchmark submission |

---

## 7. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Memory exceeds node capacity (0% compression) | High | Use `-mset GTR -mrate G,I+G` to reduce models; or split into model subsets across PBS jobs |
| Model score divergence between ranks | Low | Each rank uses identical alignment + starting tree; only OMP non-determinism is a risk — test with `--seed` |
| Checkpoint incompatibility | Low | Checkpoint format is unchanged; ranks write only their own models' entries, rank 0 holds master index |
| Upstream ModelFinder2 makes this redundant | Medium | Track IQ-TREE3 issues #130–134; this patch can be upstreamed or superseded |
| Build system conflict with existing MPI build | Low | All changes isolated to `modelfinder.cpp` + `phylotree.cpp`; existing `cmake -DIQTREE_FLAGS=mpi` path unchanged |

---

## 8. References

1. Darriba D et al. (2012) jModelTest 2. *Nat Methods* 9(8):772. doi:10.1038/nmeth.2109
2. Darriba D et al. (2020) ModelTest-NG. *Mol Biol Evol* 37(1):291. doi:10.1093/molbev/msz189
3. Kalyaanamoorthy S et al. (2017) ModelFinder. *Nat Methods* 14(6):587. doi:10.1038/nmeth.4285
4. Lefort V et al. (2017) SMS: Smart Model Selection. *Mol Biol Evol* 34(9):2422. doi:10.1093/molbev/msx149
5. IQ-TREE3 GitHub issues #130–134 (label: modelfinder2). https://github.com/iqtree/iqtree3/issues
6. MOOSE wiki. https://codeberg.org/amkozlov/raxml-ng/wiki/Automatic-model-selection-(MOOSE)
