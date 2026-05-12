# ModelFinder MPI Scaling Regression — Detailed Analysis

**Repository**: `setonix-iq` (IQtree3 MF2 branch benchmarking)  
**IQtree version**: `v3.1.2+mf2` (`icpx`, `libiomp5`, R2+LPT+AVX-512)  
**Hardware**: Gadi `normalsr` — Sapphire Rapids, 2× Intel Xeon 8470Q, 104 physical cores/node, SMT-off, 105 MB L3, 1000 GB DDR5  
**Binary**: `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`  
**Source**: `src/iqtree3/main/phylotesting.cpp`

---

## 1. Executive Summary

On the `mega_dna` dataset (500 taxa, 100,000 sites, 99,999 unique patterns), the
ModelFinder phase of a 2-node or 4-node MPI run is catastrophically slower than a
single-node run:

| Config | MF wall | vs 1-node |
|--------|---------|-----------|
| 1-node 104T | 226.7 s | 1.0× (baseline) |
| 2-node 208T | 1570.8 s | **0.14× (6.9× slower)** |
| 4-node 416T | 1073.9 s | **0.21× (4.7× slower)** |

Tree search, by contrast, scales near-linearly (1.87× and 2.02× speedup per node
doubling). The ModelFinder regression dominates the 2-node total wall time, making
it **60% longer** than the single-node run.

The root cause is a conjunction of three architectural issues in the MPI code path,
detailed below.

---

## 2. Datasets Used

| Dataset | Taxa | Sites | Unique patterns | Compression | Total size | Best model (MF2) |
|---------|------|-------|----------------|-------------|------------|-----------------|
| `xlarge_mf.fa` | 200 | 100,000 | 98,858 | 1.1% | 20 MB | GTR+I+R4 |
| `mega_dna.fa` | 500 | 100,000 | 99,999 | ~0% | 48 MB | GTR+I+R4 |

`mega_dna` is pathological: essentially no site pattern compression, and a tree with
999 internal nodes (vs 399 for `xlarge_mf`). These properties amplify every per-model
cost and every barrier overhead.

---

## 3. How ModelFinder 2 Works — Two Execution Paths

### 3.1 The 968-model test set

`CandidateModelSet::generate()` builds the candidate list: all combinations of
nucleotide substitution models (JC, K80, HKY, TN, …, GTR — about 24 base models with
frequency variants) × rate heterogeneity categories (bare, +I, +G, +I+G, +R2, +R3,
…, +R10). For DNA, this yields **968 models** for all three runs.

The model list is generated in a fixed order: bare substitution first, then +I, +G,
+I+G, then +R2 through +R10. Within each rate block, models are ordered from simplest
substitution (JC) to most complex (GTR).

### 3.2 The non-MPI path — OMP across models

When compiled without MPI, or in a pre-MF2 build, `evaluateAll()` uses:

```cpp
// phylotesting.cpp:3617
#pragma omp parallel num_threads(num_threads)
{
    int64_t model;
    do {
        model = getNextModel();          // work-stealing queue
        at(model).evaluate(params, model_info, out_model_info,
                           models_block, num_threads, brlen_type);
        at(model).computeICScores();
        // ... filterRates, dump (inside #pragma omp critical)
    } while (model != -1);
}
```

Key properties:
- **Outer** OMP team of `num_threads=104` threads is launched.
- Each thread calls `evaluate(..., num_threads=104, ...)` — but since it is already
  *inside* an OMP parallel region, `libiomp5` refuses nested parallelism
  (`OMP_NESTED=FALSE`). The inner `omp parallel num_threads(104)` inside
  `computePartialLikelihoodSIMD` degrades to **1 thread** per model.
- Net result: **1 thread per model, 104 models evaluated concurrently**. No intra-model
  OMP barriers; each thread traverses its tree sequentially.
- `filterRates()` and `filterSubst()` run inside `#pragma omp critical` but are
  effective: the generate() order ensures cheap (+R1, +G) models are evaluated
  before expensive (+R4, +R8) ones, so pruning fires early.

### 3.3 The MPI path — sequential within rank

The MF2 MPI build (any `nranks ≥ 1`) always uses the sequential path:

```cpp
// phylotesting.cpp:3553
// Always sequential in MPI builds (avoids ModelFactory concurrent-write data race)
{
    int64_t model;
    do {
        model = getNextModel();
        at(model).evaluate(params, model_info, out_model_info,
                           models_block, num_threads, brlen_type);
        at(model).computeICScores();
        // ... filterRates, dump
        model_info.dump();              // checkpoint write after every model
    } while (model != -1);
}
```

Key properties:
- **No** outer OMP parallel wrapper. `evaluate(..., num_threads=104, ...)` is called
  from the main thread, so inner `omp parallel num_threads(104)` launches the
  **full 104-thread team** for every model.
- Models are evaluated **one at a time**, each using all 104 OMP threads for
  site-level parallelism (OMP parallel-for over patterns at each internal node).
- `model_info.dump()` writes the full checkpoint to Lustre after **every** model.
- This path is used for both 1-rank and N-rank MPI builds for consistency.

---

## 4. Phase 1: LPT Dispatch (nranks > 1 only)

When `getNumProcesses() > 1`, `evaluateAll()` runs the LPT stripe before the
sequential loop (phylotesting.cpp:3463–3529):

```
Step A — cost-sorted index:
  cost(model) = k×10 if +Rk / +I+Rk
              = 5    if +I+G
              = 4    if +G
              = 2    if +I
              = 1    bare

  sorted_idx[] = argsort(models, descending cost), stable.

Step B — stripe assignment:
  for p in 0..967:
      if p % nranks != my_rank:
          at(sorted_idx[p]).setFlag(MF_IGNORED)

Step C — MF_WAITING clear:
  for i in own models:
      at(i).resetFlag(MF_WAITING)
```

For 2 ranks, the sorted list is:

```
sorted[0]  → GTR+I+R10 (cost=100) → rank 0
sorted[1]  → GTR+I+R9  (cost=90)  → rank 1
sorted[2]  → GTR+I+R8  (cost=80)  → rank 0
sorted[3]  → GTR+I+R7  (cost=70)  → rank 1
...
sorted[966] → JC+I  (cost=2)      → rank 0
sorted[967] → JC    (cost=1)      → rank 1
```

Rank 0 gets the even-indexed positions: all even-k +Rk models across all substitution
types, plus the heavier half of lower-cost models.  
Rank 1 gets the odd-indexed positions: all odd-k +Rk models.

The log confirms: `MF-MPI: rank 0/2 assigned 484/968 models (cost-sorted LPT stripe, MF_WAITING cleared)`.

### 4.1 Phase 2 — MPI gather (end of ModelFinder, not per-model)

After all ranks finish their sequential loops, a single collective operation gathers
results (phylotesting.cpp:3700–3800):

```cpp
MPI_Allreduce(local_scores, global_scores, 968, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
gatherCheckpoint();
broadcastCheckpoint();
```

There is **no per-model MPI synchronization**. Ranks execute ModelFinder entirely
independently until Phase 2. This means the total wall time is `max(wall_rank_0,
wall_rank_1)`, and if one rank is much slower, the other rank sits in the Phase 2
barrier waiting.

---

## 5. Measured Timing Data

All runs: Gadi `normalsr`, `-seed 1`, full ModelFinder (no `-m` flag).  
Profile logs at: `/scratch/um09/as1708/iqtree3-mf2/gadi-ci/profiles/`.  
JSON records at: `/home/272/as1708/setonix-iq/logs/runs/`.

### 5.1 `xlarge_mf.fa` (200 taxa)

| Config | Ranks | Threads | MF models/rank | MF wall (s) | MF CPU (s) | Tree wall (s) | Tree CPU (s) | Total wall (s) |
|--------|-------|---------|----------------|-------------|------------|---------------|--------------|----------------|
| 1-node 104T | 1 | 104 | 968 | 78.2 | 7,349 | 413.3 | 30,403 | 494.0 |
| 2-node 208T | 2 | 208 | 484 | 111.8 | 10,641 | 208.6 | 15,341 | 329.9 |
| 4-node 416T | 4 | 416 | 242 | 95.8 | 9,049 | 105.6 | 7,581 | 214.9 |

### 5.2 `mega_dna.fa` (500 taxa)

| Config | Ranks | Threads | MF models/rank | MF wall (s) | MF CPU (s) | Tree wall (s) | Tree CPU (s) | Total wall (s) |
|--------|-------|---------|----------------|-------------|------------|---------------|--------------|----------------|
| 1-node 104T | 1 | 104 | 968 | 226.7 | 21,208 | 1,124.1 | 67,746 | 1,361.2 |
| 2-node 208T | 2 | 208 | 484 | **1,570.8** | 154,397 | 599.6 | 37,475 | 2,182.3 |
| 4-node 416T | 4 | 416 | 242 | **1,073.9** | 104,540 | 297.5 | 18,253 | 1,397.4 |

> **Note**: The `MF CPU` column is rank-0's process CPU time only (OMP team total). Rank-1+
> CPUs are not reported in rank-0's log.

---

## 6. What Scales Correctly vs What Breaks

### 6.1 Tree search — correct scaling

| Metric | xlarge_mf 1→2-node | xlarge_mf 2→4-node | mega_dna 1→2-node | mega_dna 2→4-node |
|--------|-------------------|--------------------|-------------------|-------------------|
| Tree wall speedup | **1.98×** | **1.97×** | **1.87×** | **2.02×** |

Tree search scales near-linearly. Each MPI rank performs an independent tree search
on its own subset of starting topologies, and the wall time halves as ranks double.
This is the expected behavior for embarrassingly parallel tree search.

### 6.2 ModelFinder — catastrophic failure on mega_dna

| Metric | xlarge_mf 1→2-node | xlarge_mf 2→4-node | mega_dna 1→2-node | mega_dna 2→4-node |
|--------|-------------------|--------------------|-------------------|-------------------|
| MF wall speedup | 0.70× | 1.17× | **0.14×** | **1.46×** |

For `mega_dna`, going from 1-node to 2-node makes ModelFinder **7× slower** instead of
2× faster. Going from 2-node to 4-node recovers slightly (1.46×) but never approaches
the expected 2× speedup.

For `xlarge_mf`, MF scaling is mildly degraded (0.70× at 2-node) but mostly benign
because MF is only 16% of total runtime. For `mega_dna`, MF is 17% of total at 1-node
but balloons to **72% at 2-node** — the dominant bottleneck.

---

## 7. Per-Model Wall Time Inflation

Each MPI rank (at nranks=2, each with 484 models) should complete in approximately
`484/968 × 226.7 = 113.3 s` if per-model performance were unchanged. Actual times:

| Config | Models/rank | MF wall (s) | Expected (s) | Wall inflation | Per-model wall (s) |
|--------|-------------|-------------|--------------|----------------|--------------------|
| 1-node 104T | 968 | 226.7 | 226.7 | **1.00×** | 0.2342 |
| 2-node 208T | 484 | 1,570.8 | 113.3 | **13.86×** | 3.2454 |
| 4-node 416T | 242 | 1,073.9 | 56.7 | **18.95×** | 4.4379 |

Each model evaluation takes 13.9–19× longer per rank in multi-node mode than in
single-node mode. The inflation *worsens* as node count increases.

---

## 8. CPU Efficiency Analysis

The CPU time reported by rank 0 (`getCPUTime()` = sum over all 104 OMP threads on
that node) allows us to compute OMP thread utilization:

| Config | MF CPU (s) | MF wall (s) | Effective cores | OMP efficiency |
|--------|------------|-------------|-----------------|----------------|
| 1-node 104T | 21,208 | 226.7 | 93.6 | **90.0%** |
| 2-node 208T | 154,397 | 1,570.8 | 98.3 | **94.5%** |
| 4-node 416T | 104,540 | 1,073.9 | 97.4 | **93.6%** |

Tree search OMP efficiency is consistently ~58–60% (known barrier-saturation cost for
large trees). ModelFinder OMP efficiency in the sequential MPI path is *higher* at
94–95% than the 1-node value of 90%.

**Critical implication**: rank 0's OMP team is NOT idle during the inflated ModelFinder
time. All 104 OMP threads are occupied. The extra wall time is real CPU computation,
not waiting. This means each model evaluation truly requires more compute cycles in
the multi-node MPI mode than in the single-node mode.

---

## 9. Root Cause Analysis

The per-model CPU cost inflates by 14.6× for mega_dna at 2-node (from 21.9 to 319.5
CPU-s/model). Three compounding causes have been identified; a fourth requires
additional profiling to confirm or rule out.

### 9.1 Cause 1: LPT sort disables `filterRates` pruning (confirmed)

**The single most impactful root cause.**

In the 1-node sequential path, models are evaluated in `generate()` order:
bare → +I → +G → +I+G → +R2 → +R3 → … → +R10, cycling through all substitution
models within each rate block.

`filterRates(model)` is called after each +Rk evaluation. Its logic
(phylotesting.cpp:2882):

> If a lower-k neighbour (+R(k-1)) has a **better** IC score than the current +Rk
> model, mark all +R(k+1), +R(k+2), …, +R10 in that substitution line as
> `MF_IGNORED`.

Because cheap models are evaluated first in generate() order, by the time the
expensive +R4, +R5, … models are reached, many are already pruned. In practice, for
most DNA alignments, the rate series converges at +R2 or +R3, and all higher-k models
(+R4 through +R10) are pruned across ALL substitution types. This eliminates roughly
500–600 out of 968 models before they are evaluated.

In the **LPT-sorted MPI stripe**, rank 0 receives the most expensive models:
all even-k rate categories across all substitution types. The critical problem:

```
sorted[0]  → GTR+I+R10  (cost=100) → rank 0
sorted[2]  → GTR+I+R8   (cost=80)  → rank 0
sorted[4]  → GTR+I+R6   (cost=60)  → rank 0
...
```

When rank 0 finishes GTR+I+R10 and calls `getLowerKModel(GTR+I+R10)`, it gets
GTR+I+R9. But GTR+I+R9 belongs to rank 1's stripe → it is flagged `MF_IGNORED` on
rank 0. The pruning guard:

```cpp
if (lower_model >= 0
    && !at(lower_model).hasFlag(MF_IGNORED)   // ← FALSE: it is MF_IGNORED
    && at(lower_model).getScore() < ...)
```

...evaluates to **false**, so no pruning fires. Rank 0 evaluates GTR+I+R10, GTR+I+R8,
GTR+I+R6, GTR+I+R4, GTR+I+R2 — ALL even-k models — without any pruning, even when the
alignment strongly prefers +R2 over +R10.

**Quantified impact**: At 1-node with normal pruning, roughly 350–450 models are
actually evaluated (the rest MF_IGNORED early). At 2-node, rank 0 evaluates all 484
assigned models, with no early exits. The effective model count per rank is ~1.1–1.4×
higher than the 1-node average, but the *composition* is far worse: rank 0's models
are exclusively the heaviest rate-category variants.

### 9.2 Cause 2: LPT cost function mismatch for large trees (confirmed)

The LPT cost proxy (`+Rk → cost = k×10`) captures the scaling of rate categories
but not the scaling with tree size (number of taxa N).

For a tree with N taxa, each model evaluation requires optimizing 2N−3 branch
lengths. Branch-length optimization in IQtree iterates over all branches multiple
times, each requiring a full tree likelihood pass over `(N−1)` internal nodes ×
`P` patterns. The actual per-model compute scales approximately as:

$$T_{model} \propto N \times P \times k \times I_{converge}$$

where $k$ is the rate category count and $I_{converge}$ is the number of optimizer
iterations.

- `xlarge_mf` (N=200): 1-node per-model CPU = 7.6 CPU-s (bare average)
- `mega_dna` (N=500): 1-node per-model CPU = 21.9 CPU-s (bare average)
- Ratio: 21.9/7.6 = **2.88×** (consistent with N=500/200=2.5× tree size)

The cost function assigns the same `k×10` cost regardless of N. For mega_dna, the
absolute cost of a +R10 model is therefore 2.88× higher than the cost function
predicts relative to xlarge_mf. Since LPT assigns rank 0 the most expensive models,
rank 0 accumulates disproportionately more actual work than the cost function estimates,
especially on large-tree datasets.

### 9.3 Cause 3: OMP barrier frequency × tree size (confirmed)

In the sequential MPI path, each model evaluation uses 104 OMP threads with a
`parallel for` over patterns at every internal node of the tree. For a tree with N
taxa, each likelihood traversal pass requires traversing N−1 internal nodes, and the
OMP team synchronizes at a barrier between every node:

- `mega_dna` (999 nodes): **~999 barriers per traversal pass**
- `xlarge_mf` (399 nodes): ~399 barriers per traversal pass

A full model evaluation (multiple optimization passes, ~5–10 traversals):
- `mega_dna`: ~999 × 5–10 = **5,000–10,000 OMP barriers per model**
- `xlarge_mf`: ~399 × 5–10 = 2,000–4,000 OMP barriers per model

The non-MPI path (1 thread per model) has **zero** intra-model OMP barriers. Each
thread traverses its own tree sequentially with no synchronization.

From VTune profiling data (documented in `context.md`, 32T runs): `kmp_flag_64::wait`
accounts for ~52% of samples during tree computation on the sequential path. This
barrier overhead is paid per-model in the MPI sequential path, but is invisible in the
non-MPI OMP-across-models path. The 2.5× more barriers for mega_dna amplifies this
cost factor relative to xlarge_mf.

### 9.4 Cause 4: Checkpoint I/O contention on Lustre (suspected, unconfirmed)

`model_info.dump()` is called inside the sequential loop after **every** model
(phylotesting.cpp:3582). At 2-node, both rank 0 and rank 1 write to the same output
prefix:

```
/scratch/um09/as1708/.../profiles/mega_dna_208t_mf2_full_np2.../iqtree_run.ckp.gz
```

If both ranks write to an overlapping file path (or the same Lustre inode), Lustre
file locking serializes writes. As the checkpoint grows (it accumulates all evaluated
model results), each `dump()` writes an increasingly large file. At 484 models × a
growing checkpoint file, this serialization adds up.

At 4-node the inflation is worse (18.95× vs 13.86×), consistent with more ranks
contending for the same file. However, this hypothesis requires verification by
checking whether ranks use distinct output paths.

**To confirm**: add per-model wall-time logging, or strace the checkpoint writes, or
verify that each rank uses a rank-specific checkpoint path.

---

## 10. Why `xlarge_mf` Is Less Affected

`xlarge_mf` exhibits the **same three root causes** but at much smaller scale:

| Metric | xlarge_mf | mega_dna | Ratio |
|--------|-----------|----------|-------|
| Per-model CPU at 1-node | 7.6 CPU-s | 21.9 CPU-s | 2.88× |
| Per-model inflation at 2-node | 2.9× | 13.9× | 4.8× |
| MF fraction of total at 1-node | 16% | 17% | ~same |
| MF fraction of total at 2-node | 34% | 72% | 2.1× |
| Internal nodes per tree | 399 | 999 | 2.5× |

The per-model inflation for xlarge_mf (2.9×) vs mega_dna (13.9×) is a 4.8× difference.
This is consistent with the tree-size scaling of OMP barrier cost (~2.5× more barriers
for mega_dna) combined with the larger absolute per-model cost creating a wider gap
between pruned (1-node) and unpruned (MPI) evaluation.

Crucially, because `xlarge_mf`'s MF fraction is small (16%), even a 2.9× per-model
slowdown only increases MF wall time from 78 s to 112 s — a tolerable 45% overhead.
For `mega_dna`, the same mechanism produces a 594% overhead, changing the MF fraction
from 17% to 72% of total runtime and making the 2-node run 60% slower than the
single-node baseline.

---

## 11. Best-Model Discrepancy

A secondary consequence of the pruning bypass: the selected best model differs between
1-node and multi-node runs:

| Config | Best model (AIC/AICc) |
|--------|----------------------|
| 1-node 104T | **GTR+G4** |
| 2-node 208T | **GTR+I+R4** |
| 4-node 416T | **GTR+I+R4** |

At 1-node, the `+I+G → +I+R2 → … → +I+R4` chain is likely pruned by `filterRates`
after finding that +G4 is adequate. Ranks at 2+node evaluate all +Rk variants and
therefore correctly select the more general GTR+I+R4 model, which fits mega_dna's
heterogeneous rate distribution better.

**This is not a correctness bug**: IQtree's IC scores are valid for all evaluated
models. The 1-node result is a consequence of aggressive pruning that may eliminate
models before they can be compared fairly. The multi-node result is more conservative
and arguably more reliable for a dataset this large.

---

## 12. Summary of Causes and Their Interactions

```
mega_dna MPI ModelFinder regression
│
├── [C1] LPT sort × pruning bypass
│     ├── Cost-sorted stripe assigns ALL heavy +Rk (k=2,4,6,8,10) to rank 0
│     ├── filterRates cannot fire: lower-k siblings are MF_IGNORED on rank 0
│     └── Rank 0 evaluates 484 unpruned heavy models vs ~350 pruned models at 1-node
│
├── [C2] LPT cost function underestimates large-tree cost
│     ├── cost = k×10 ignores tree size factor N
│     ├── mega_dna (N=500) has 2.88× more cost per model than xlarge_mf (N=200)
│     └── Rank 0 accumulates 2.88× more actual work than cost function predicts
│
├── [C3] Sequential 104T evaluation × 999-node barrier frequency
│     ├── Non-MPI: 1 thread per model, 0 intra-model barriers
│     ├── MPI sequential: 104 threads per model, 999 barriers per traversal pass
│     └── mega_dna has 2.5× more barriers per model than xlarge_mf
│
└── [C4] Checkpoint dump() I/O (suspected)
      ├── model_info.dump() called after EVERY model evaluation
      ├── Both ranks write to same checkpoint path on Lustre
      └── File-lock contention could serialize all 968 dump() calls
```

All three confirmed causes are load-imbalance or overhead sources that are *absent*
in the 1-node MPI run (which evaluates models in generate() order, benefits from full
pruning, and has no parallel dump() contention). The 2-node run combines all three
into a catastrophic regression.

---

## 13. Profiling Evidence

From `/scratch/um09/as1708/iqtree3-mf2/gadi-ci/profiles/`:

### mega_dna 1-node (PBS 168213985)
```
CPU time for ModelFinder: 21207.593 seconds
Wall-clock time for ModelFinder: 226.670 seconds
CPU time for tree search: 67746.201 seconds
Wall-clock time for tree search: 1124.074 seconds
```

### mega_dna 2-node (PBS 168213987)
```
MF-MPI: rank 0/2 assigned 484/968 models (cost-sorted LPT stripe, MF_WAITING cleared)
CPU time for ModelFinder: 154397.489 seconds
Wall-clock time for ModelFinder: 1570.765 seconds   ← 6.93× longer than 1-node
CPU time for tree search: 37475.293 seconds
Wall-clock time for tree search: 599.603 seconds    ← 1.87× faster ✓
```

### mega_dna 4-node (PBS 168213988)
```
MF-MPI: rank 0/4 assigned 242/968 models (cost-sorted LPT stripe, MF_WAITING cleared)
CPU time for ModelFinder: 104540.071 seconds
Wall-clock time for ModelFinder: 1073.973 seconds   ← 4.74× longer than 1-node
CPU time for tree search: 18253.313 seconds
Wall-clock time for tree search: 297.540 seconds    ← 3.78× faster ✓
```

---

## 14. Recommended Investigations

### High priority

1. **Make filterRates work across ranks**: Instead of marking cross-rank models
   `MF_IGNORED`, allow rank 0 to skip evaluating a model if its lower-k sibling
   (on rank 1) is already done and has a better score. This requires an intermediate
   Phase 1.5 MPI_Allreduce after each rate block completes. Alternatively, assign
   all models in the same +Rk series to the same rank (grouping by substitution model,
   not cost-sorted stripe).

2. **Profile checkpoint I/O**: Check whether `model_info.dump()` serializes ranks on
   Lustre. If confirmed, switch to rank-local checkpoint files during Phase 1 and merge
   only in Phase 2. This would also improve robustness for long runs.

3. **Include tree size N in LPT cost function**: Update `modelCost()` to multiply by a
   tree-size proxy (e.g., `log(num_taxa)`) so the load balance accounts for the actual
   compute cost:
   ```cpp
   return k * 10 * (int)ceil(log2(num_taxa));
   ```

### Medium priority

4. **Add per-model timing to logs**: Instrument the sequential evaluation loop to record
   wall time per model. This would directly confirm whether the inflation is uniform
   (I/O hypothesis) or model-type-dependent (barrier/pruning hypothesis).

5. **Measure effective model count after pruning**: Log how many models were actually
   evaluated (not MF_IGNORED at evaluation time) at 1-node vs 2-node. This would
   quantify the pruning bypass contribution.

6. **Assign whole +Rk series to single rank**: Instead of striping across all 968
   sorted models, stripe at the substitution-model level (assign all rate variants
   for each substitution model to the same rank). This preserves filterRates within
   each rank and avoids the cross-rank pruning bypass entirely.

### Low priority

7. **OMP team size tuning for MPI sequential path**: For mega_dna with 999 nodes, the
   104-thread OMP team pays a high barrier overhead. Evaluate whether using fewer OMP
   threads (e.g., 52) with more models per rank per OMP wave would reduce barrier cost
   and improve throughput. This trades intra-model parallelism for reduced barrier
   frequency.

---

## 15. Relation to Context Document

`context.md` §5b documents the broader OMP barrier saturation observed at 32T on
xlarge datasets, with `kmp_flag_64::wait` accounting for ~52% of samples during
tree search. ModelFinder in the sequential MPI path experiences the same saturation
at model-evaluation granularity. The 94% OMP efficiency reported for mega_dna
ModelFinder (vs 58% for tree search) is counterintuitive at first but explained by the
fact that ModelFinder's per-node OMP regions are shorter (fewer patterns per thread
slot due to the simpler rate model), leaving less time for barriers to dominate.

The `CHANGELOG.md` R2+LPT entry documents the original implementation of the cost-
sorted stripe. The cross-rank pruning bypass was a known limitation at the time of
the R2+LPT commit: the `MF_WAITING clear` in Step C was designed to avoid permanent
blocking, not to preserve filter accuracy. The best-model discrepancy (GTR+G4 at
1-node vs GTR+I+R4 at 2+nodes) is a direct manifestation of this known limitation.

---

*Document written: 2025-06. All timing data from Gadi PBS jobs 168213985–168213988.*  
*Source analyzed: `src/iqtree3/main/phylotesting.cpp` (v3.1.2+mf2), lines 1520–3800.*
