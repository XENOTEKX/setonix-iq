# MF2 MPI Load Balancing: Analysis and Fix

*Author: as1708 | Date: 2026-05 | Relates to: modelfinder-mpi.md §17, CHANGELOG ba*

---

## 1. Does the subst-family stripe cause load imbalance?

**Short answer: yes, but quantifiably small and partially fixed by Fix C.**

The Phase 1 subst-family LPT stripe assigns entire substitution-model families to
MPI ranks. Each rank receives the same static "cost" (number of rate-category
variants), but the ACTUAL work per rank depends on how aggressively `filterRates()`
and `getLowerKModel()` prune models at runtime. Before Fix C these two pruning
mechanisms were asymmetric across ranks, causing the imbalance.

---

## 2. Pruning mechanisms and their MPI behaviour

### 2.1 getLowerKModel — intra-family +Rk series pruning

`getLowerKModel(model)` compares each evaluated `+Rk` model against its `+R(k-1)`
predecessor. If `+R(k-1)` scored better, it prunes all higher-k siblings.

| Rank | Assigned families | getLowerKModel available? |
|------|-------------------|--------------------------|
| 0 | LG, Q.mammal, Q.insect, Q.pfam, Q.bird | YES — all series on one rank |
| 1 | WAG, Q.plant, JTT, BLOSUM62, Dayhoff | YES — all series on one rank |
| 2 | mtREV, Q.yeast, LG4M, LG4X, Q.mammal | YES |
| 3 | VT, Jones, DCMut, rtREV, HIVb | YES |

The subst-family stripe guarantees every `+Rk` series for a family sits entirely
on one rank, so `getLowerKModel` works identically on all ranks. This was the main
correctness goal of Fix A.

### 2.2 filterRates — cross-family rate-type pruning (the load-balance bottleneck)

`filterRates(finished_model)` uses the **first substitution family** evaluated on
the rank as a reference to determine which rate types (`+G4`, `+R2`, etc.) are
worth exploring at all. Once the reference family is fully evaluated, models of
ALL families with non-qualifying rate types are marked `MF_IGNORED`.

**Original code (before Fix C):**

```cpp
for (model = 0; model <= finished_model; model++)
    if (at(model).subst_name == at(0).subst_name) { // always "LG" for AA
        if (!at(model).hasFlag(MF_DONE + MF_IGNORED)) return;
        best_score = min(best_score, at(model).getScore());
    }
```

`at(0).subst_name` is the global first family in `generate()` order: `LG` for AA,
`GTR` for DNA. In MPI mode, ranks 1-3 have all LG/GTR models marked `MF_IGNORED`
(assigned to rank 0). Their `BIC_score` is the constructor default `DBL_MAX`.

Consequence:
- The early-return guard passes (IGNORED counts as "done").  
- `best_score = min(DBL_MAX, ...) = DBL_MAX`.  
- `ok_score = DBL_MAX` → every rate type passes the filter.  
- `ok_rates` = the set of ALL rate types → **nothing is pruned** on ranks 1-3.

**Effect on walltime (AA 100K, LG+G4 best model):**

| Rank | Reference family | filterRates effective? | Models evaluated (est.) |
|------|-----------------|----------------------|-------------------------|
| 0 | LG (61 models) | YES — prunes ~70% of non-LG +R3-R10 | ~130 |
| 1 | LG (IGNORED) → NO | No cross-family pruning | ~220 |
| 2 | LG (IGNORED) → NO | No cross-family pruning | ~220 |
| 3 | LG (IGNORED) → NO | No cross-family pruning | ~220 |

Rank 0 finishes ~1.5–1.7× faster than ranks 1-3. Makespan is set by the slowest
rank (rank 1-3). Load imbalance factor ≈ **1.12–1.15×** (12–15% efficiency loss).

---

## 3. Literature context

### 3.1 LPT: List Scheduling with Longest Processing Time (Graham 1969)

**Reference**: R.L. Graham, "Bounds on Multiprocessing Timing Anomalies,"
*SIAM Journal on Applied Mathematics*, 17(2):416–429, 1969.

LPT assigns jobs in decreasing order of estimated cost, round-robin to machines.
Worst-case makespan bound:

$$C_{\max}^{\text{LPT}} \le \frac{4m - 1}{3m} \cdot C_{\max}^{\text{OPT}}$$

For $m = 4$ MPI ranks:

$$C_{\max}^{\text{LPT}} \le \frac{15}{12} \cdot C_{\max}^{\text{OPT}} = 1.25 \times C_{\max}^{\text{OPT}}$$

The bound assumes **accurate static cost estimates**. The subst-family LPT assigns
cost proportional to the number of rate-category variants (61 models per family for
AA). This is accurate before pruning but diverges once filterRates fires: rank 0
evaluates only ~130/308 models (42%) while ranks 1-3 evaluate ~220/308 (71%).

The LPT bound does NOT account for this pruning differential — so the real
makespan ratio with asymmetric pruning exceeds the theoretical 1.25× LPT bound.
Fix C restores symmetric pruning, bringing actual costs closer to the static
estimates and making LPT's bound tighter.

### 3.2 Work Stealing (Blumofe & Leiserson 1999)

**Reference**: R.D. Blumofe & C.E. Leiserson, "Scheduling Multithreaded Computations
by Work Stealing," *J. ACM*, 46(5):720–748, 1999.

Work stealing achieves expected runtime:

$$E[T_P] = \frac{T_1}{P} + O(T_\infty)$$

where $T_1$ = total work, $P$ = processors, $T_\infty$ = span (critical path).
For model selection, models are independent: $T_\infty = 1$ (one model evaluation).
This gives $E[T_P] \approx T_1/P$ — near-optimal parallel efficiency.

The existing `getNextModel()` OMP loop in `evaluateAll()` IS already work stealing
within a single MPI rank: threads grab models dynamically from a shared queue
(guarded by `#pragma omp critical`), achieving near-optimal intra-node efficiency.

**What work stealing does NOT address**: the inter-rank static LPT assignment.
Once Phase 1 sets `MF_IGNORED` flags, no rank can "steal" families from another
rank. After Fix C reduces intra-rank imbalance, the remaining inefficiency is the
LPT static assignment vs actual post-pruning work — a ~5-8% residual.

### 3.3 IQ-TREE 2 Multi-Locus Dynamic Scheduling (Minh et al. 2020)

**Reference**: B.Q. Minh et al., "IQ-TREE 2: New Models and Methods for Phylogenetic
Inference," *Molecular Biology and Evolution*, 37(5):1530–1534, 2020.

For multi-locus analysis, IQ-TREE 2 uses **dynamic list scheduling**: "the $k$
loci with highest estimated costs are assigned to one of $k$ cores; when a core
finishes, the next locus is assigned dynamically." This achieves 97-101% parallel
efficiency versus the ParGenes approach.

The analogy to MF2 MPI is direct: loci ↔ substitution families. A dynamic
inter-rank scheduler would assign families to ranks on demand (when a rank
finishes its current family, MPI sends it the next). This would reduce residual
imbalance from ~5-8% to near zero.

**Why dynamic scheduling is not implemented yet**: it requires non-trivial MPI
coordination (all-reduce barriers, work request/grant protocol). The current Fix C
reduces imbalance to the level where static LPT + per-rank filterRates is adequate
for datasets with 4-20 substitution families.

---

## 4. Fix C: per-rank filterRates reference (implemented)

### 4.1 What changed

**In `filterRates()` (lines ~2882–2922 after fix):**

```cpp
// Find the first non-ignored substitution family as reference.
// In MPI mode ranks 1-3 have at(0) IGNORED; use own assigned first family.
string ref_subst = at(0).subst_name;
for (int i = 0; i < (int)size(); i++) {
    if (!at(i).hasFlag(MF_IGNORED)) {
        ref_subst = at(i).subst_name;
        break;
    }
}
...
for (model = 0; model <= finished_model; model++)
    if (at(model).subst_name == ref_subst) {
        if (!at(model).hasFlag(MF_DONE + MF_IGNORED)) return;
        if (!at(model).hasFlag(MF_IGNORED))             // skip IGNORED (DBL_MAX)
            best_score = min(best_score, at(model).getScore());
    }
if (best_score == DBL_MAX) return;   // ref family not yet evaluated

// ok_rates: only from non-ignored models (exclude DBL_MAX cross-rank models)
for (model = 0; model <= finished_model; model++)
    if (!at(model).hasFlag(MF_IGNORED) && at(model).getScore() <= ok_score) {
        ok_rates.insert(at(model).orig_rate_name);
    }
```

**In `evaluateAll()` (after Phase 1 `#endif`, before OMP loop):**

```cpp
// Recompute rate_block per-rank so filterRates fires at the right time.
if (MPIHelper::getInstance().getNumProcesses() > 1 && auto_rate) {
    string ref_subst;
    for (int64_t i = 0; i < num_models; i++)
        if (!at(i).hasFlag(MF_IGNORED)) { ref_subst = at(i).subst_name; break; }
    if (!ref_subst.empty()) {
        rate_block = 0;
        for (int64_t i = 0; i < num_models; i++)
            if (at(i).subst_name == ref_subst) rate_block = (int)i;
    }
}
```

### 4.2 Why rate_block recomputation is necessary

The OMP loop fires `filterRates` when `model >= rate_block`. With the global
`rate_block` = last LG index (e.g., 60), filterRates would fire on rank 1 after
WAG+G4 (index 61) completes, with only ONE WAG model seen. The guard would pass
(only WAG+G4 in 0..61 for subst="WAG") → `ok_rates = {+G4}` → pruning too
aggressively after just one data point.

Recomputing `rate_block` to the **last index of WAG** (e.g., 122) ensures
filterRates fires after WAG+R10 completes — the computationally heaviest WAG
model, which finishes last (matching the LG+R10 trigger on rank 0). By that point
all 61 WAG models are done and filterRates has a complete BIC picture.

### 4.3 Correctness of the two-loop change

The **second loop** that builds `ok_rates` previously scanned ALL models
0..finished_model for any score ≤ ok_score. With the `!hasFlag(MF_IGNORED)` guard,
cross-rank models (DBL_MAX) are excluded. This is correct: rank 1 should determine
ok_rates from its own evaluated models, not from unevaluated LG models on rank 0.

The **pruning loop** is unchanged: it marks future non-ok models `MF_IGNORED`.
These may include models on other ranks (already IGNORED) — setting IGNORED twice
is a no-op.

### 4.4 filterSubst is unaffected

`filterSubst()` uses `at(0).rate_name` (the first rate type, e.g., `+G4`) as
reference and scans ALL `+G4` models across all families. On ranks 1-3:
- IGNORED cross-rank `+G4` models score DBL_MAX → excluded from best_score by `min()`
- Own-rank `+G4` models have real scores → correct best_score
- Pruning loop: IGNORED models set IGNORED again → no-op

`filterSubst` naturally handles the MPI stripe correctly because the `min()`
function discards DBL_MAX values. No fix needed.

---

## 5. Expected impact by dataset type

### 5.1 AA datasets (20 standard matrices, 1,220 models with +R2..+R10)

| Dataset size | Before Fix C | After Fix C | Residual imbalance |
|-------------|-------------|-------------|-------------------|
| Small (≤100K sites) | ~12-15% imbalance | ~5-8% | LPT static vs actual |
| Large (>1M sites) | ~12-15% imbalance | ~5-8% | Same structural |

With LG+G4 as best model (typical for most protein datasets):
- Rank 0: LG reference, prunes ~70% of +R3-R10 for ALL families → ~130 models
- Ranks 1-3 (after Fix C): own-family reference, same pruning logic → ~130-150 models
- **Near-symmetric workload** across all ranks

### 5.2 DNA datasets (24 standard matrices, ~1,200 models)

DNA has 24 standard matrices (GTR, TVM, TIM1, TIM2, TIM3, HKY, etc.). With np=4:
6 families per rank. The same subst-family stripe applies; GTR is the global
reference on rank 0.

After Fix C, rank 1 uses e.g. TVM as reference. TVM+G4 is typically the best
non-GTR model → ok_rates from TVM correctly reflects the rate landscape.

| Dataset | Config | Estimate before Fix C | Estimate after Fix C |
|---------|--------|----------------------|---------------------|
| DNA small | np4 | 15% imbalance | 5-8% imbalance |
| DNA large | np4 | 15% imbalance | 5-8% imbalance |

### 5.3 Small datasets (few models survive getLowerKModel pruning early)

For very small datasets where +G4 dominates strongly (e.g., ≤1K sites), most
+Rk models are pruned by `getLowerKModel` quickly. Both filterRates and Fix C
become less critical because `getLowerKModel` fires early and prunes deeply on all
ranks. The residual imbalance is negligible.

---

## 6. Future work: dynamic inter-rank family redistribution

If residual imbalance (~5-8%) is ever a concern for very large runs (e.g.,
np=16+), a Phase 1.5 dynamic redistribution could be implemented:

```
Approach (inspired by IQ-TREE 2 locus scheduling):
1. Phase 1: LPT static assignment as today (fast O(F log F) setup)
2. Phase 1.5: After each rank finishes its last assigned family,
   it broadcasts "rank R is idle" via MPI_Isend.
3. The rank with the largest remaining unstarted family grants a
   family transfer (changes MF_IGNORED flags).
4. Dynamic redistribution continues until no families remain.
```

This would achieve near-optimal scheduling per the work-stealing bound
$E[T] \approx T_1/P$. The overhead is ~F MPI messages (F = number of families,
typically 20-24 for AA/DNA) — negligible relative to evaluation time.

However, given the typical imbalance after Fix C is only ~5-8%, and the existing
`getNextModel()` OMP work stealing handles intra-rank balance perfectly, Phase 1.5
is not warranted for current datasets.

---

## 7. Summary

| Issue | Source | Impact | Fix |
|-------|--------|--------|-----|
| filterRates broken on ranks 1-3 | `at(0).subst_name` = LG; all LG IGNORED with DBL_MAX | 12-15% wall-time imbalance | Fix C: per-rank ref_subst |
| rate_block trigger too early on ranks 1-3 | Global rate_block = last LG index; WAG starts at rate_block+1 | filterRates fires after 1 model → over-aggressive pruning | Fix C: per-rank rate_block recompute |
| Residual LPT static vs actual costs | Pruning rates not known at assignment time | ~5-8% imbalance after Fix C | Accept for now; future Phase 1.5 |
| getLowerKModel (intra-family) | Works correctly on all ranks | Correctly prunes +Rk series | Already correct (Fix A preserved this) |

Commit for Fix C: see CHANGELOG entry `ba` on the `gadi-spr-r2-avx512` branch.

---

## 8. MPI data-path overhead: why communication is never the bottleneck

### 8.1 Phase 2 collective operations

After all models are evaluated (Phase 1), the Phase 2 gather completes in two steps:

**Step 1 — four `MPI_Allreduce` calls** (lnL MAX, BIC MIN, AIC MIN, AICc MIN):
- 4 × 1,232 doubles × 8 B = 39.4 KB total
- Latency: log₂(P) × ~1 µs, transfer: 39.4 KB / 12 GB/s = 3 µs per allreduce
- **Total: ~21 µs** — unmeasurable against model evaluation time

**Step 2 — `gatherCheckpoint` + `broadcastCheckpoint`**:
- Each rank serialises its checkpoint (model names, scores, parameters, tree newick)
- Per rank (np4, ~150 post-pruning models): ~345 KB text
- `MPI_Gatherv` → rank 0 receives 4 × 345 KB = 1.38 MB
- `MPI_Bcast` → rank 0 sends 1.38 MB to all ranks
- Plus CPU serialisation/deserialisation ≈ ~3 ms per rank
- **Total Phase 2B: ~7–10 ms**

Grand total MPI overhead: **< 12 ms** against 100–400 s model evaluation.
This scales linearly with number of models (not with dataset size or site count),
so it remains negligible even for million-site runs with np=16+.

### 8.2 Implication for hot-kernel loop design

MPI is used only at the MODEL granularity (one rank per family group). Within a
single model evaluation, all site-level work is performed entirely by one thread
(evaluateAll() OMP-across-models) or 103 threads (test() path). No MPI communication
occurs inside the hot likelihood kernel loops. Adding MPI at the site level would
require distributed memory partial-likelihood computation (a fundamentally different
algorithm) and would introduce latency at every tree node — not a practical path for
the current architecture.

The correct approach to maximise throughput is to saturate cores with independent
model evaluations (already implemented via evaluateAll()), not to fragment individual
model computations across ranks.

---

## 9. Thread saturation and NUMA binding

### 9.1 Thread saturation in evaluateAll()

With OMP-across-models and T=103 threads evaluating M models per rank:

| Round | Active threads | Idle threads |
|-------|---------------|-------------|
| 1 | min(M, T) | max(0, T−M) |
| 2 (if M > T) | M − T | T − (M − T) |
| last | M % T | T − (M % T) |

For AA 100K, np4 (M≈150, T=103):
- Round 1: 103 threads active, 0 idle. Duration: t₁ (average model time).
- Round 2: 47 threads active, 56 idle. Duration: t₂ ≈ t₁ (similar models).
- Thread utilisation: (103 + 47) / (103 × 2) = 72.8%.
- Tail waste as fraction of total wall time: 56/206 × 1 round ≈ 17% of round-2 time.

In absolute terms: Fix A (LPT) assigns the heaviest families first, so the last round
carries mostly +G4/+I+G4 models (fast). Tail duration ≈ 5–10 s; tail loss ≈ 5–8 s
out of ~100 s MF wall time = **5–8%** of MF time, **2–3%** of total run time.

A hybrid nested-OMP mode would recover this loss but adds scheduling complexity.
Given the small absolute gain (< 10 s), this is deferred.

### 9.2 NUMA binding in evaluateAll()

**Per-model data is self-NUMA-correct.** Each OMP thread calls
`CandidateModel::evaluate()` which allocates a fresh `IQTree` clone — all associated
heap data (`partial_lh`, rate arrays, branch objects) is allocated by the calling
thread. With Linux first-touch policy, those pages land on the NUMA node of the
allocating thread's physical CPU. Reads and writes to per-model data are therefore
local DRAM operations regardless of which socket the thread is bound to.

**Shared read-only data (alignment).** The alignment (`~9.6 MB` for AA 100K) was
allocated by the main thread before the OMP region. With first-touch, those pages
are on socket 0. However:
- 9.6 MB < 60 MB SPR LLC per socket.
- Both socket L3 caches hold the alignment after a short warm-up (<~1 s of the OMP
  parallel region), served from local L3 rather than cross-NUMA DRAM for subsequent
  models.
- No sustained cross-NUMA DRAM penalty for read-only alignment data.

**Contrast with `test()` path.** The `test()` path runs all 103 threads on ONE model's
`partial_lh` (~60 MB), which exceeds one socket's L3. The existing first-touch pragmas
(R1a `computePtnFreq`, R1b `computePtnInvar`, R2a `_pattern_lh_cat`) distribute those
pages across both NUMA domains using `#pragma omp parallel for schedule(static)`.
These pragmas are correct and necessary for `test()`; they have no effect in
`evaluateAll()` because the inner OMP loops execute single-threaded (nested OMP off),
so no `schedule(static)` first-touch distribution occurs across sockets in that context.

### 9.3 Fix D — `proc_bind(spread)` for evaluateAll() (commit `0db014bc`)

The `evaluateAll()` OMP parallel pragma was updated from:
```cpp
#pragma omp parallel num_threads(num_threads)
```
to:
```cpp
#pragma omp parallel num_threads(num_threads) proc_bind(spread)
```

**Why `spread`?** `OMP_PROC_BIND=close` (set globally in all job scripts) assigns
threads to consecutive hardware places starting from the master thread's place. For
T=103 on 104 cores this distributes across both sockets incidentally (cores 0–51
socket 0, cores 52–103 socket 1). For T < 52, however, `close` packs all threads on
socket 0, halving available DRAM bandwidth.

`proc_bind(spread)` overrides `close` only for this one parallel region, distributing
T threads evenly across all available hardware places before applying sub-grouping:

| T | close → sockets | spread → sockets |
|---|-----------------|-----------------|
| 103 | ~52 / ~51 | ~52 / ~51 (identical) |
| 52 | 52 / 0 | 26 / 26 |
| 48 | 48 / 0 | 24 / 24 |
| 24 | 24 / 0 | 12 / 12 |

For T=103 (production) the change is neutral. For smaller-T testing or future runs on
nodes with more NUMA domains, `spread` ensures proportional bandwidth utilisation.

The `test()` path and all hot-kernel inner loops continue to use the global `close` +
`schedule(static)` binding, which is correct for their first-touch NUMA strategy.

---

*Source analyzed: `src/iqtree3/main/phylotesting.cpp` (v3.1.2+mf2), commit after `b9b04a1c`.*  
*Literature: Graham 1969 (LPT bounds), Blumofe & Leiserson 1999 (work stealing), Minh et al. 2020 (IQ-TREE 2 locus scheduling).*
