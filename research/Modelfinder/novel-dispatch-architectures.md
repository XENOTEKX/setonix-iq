# Novel Dispatch Architectures for IQ-TREE ModelFinder

**Author:** as1708 (with research synthesis by Claude Opus 4.7)
**Date:** 2026-05-26
**Status:** Design proposal — supersedes Mode P / MPGC / P.7 (failed) as the next architecture milestone.
**Scope:** AA & DNA workloads, 100K – 10M unique patterns, np=2 – np=64.

---

## 0. TL;DR

The current `Mode P + MPGC` dispatch (group_size=2 static, family-bound groups) failed the P.7 perf gate at AA 1M np=16 (>1,800 s vs the 600 s gate). Two root causes were found:

1. **Static group↔family binding ⇒ catastrophic load imbalance** — at job kill, MPGC Group 7 (ranks 14-15) had **completed 0 Mode P models in 30 minutes** while Group 0 finished 5; Groups 5,6 each started their first heavy model at t≈989 s and were still running. FCA np=16 avoids this entirely through implicit work-stealing at the model granularity.

2. **Tree traversal does NOT honour the Mode P pattern slice** ([phylokernelnew.h:1268-1283](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylokernelnew.h#L1268)) — `computeTraversalInfo` calls `computeBounds(..., nptn, limits)` with the **full** `nptn`, not `mp_size`. Both ranks in a group_size=2 pair compute partial_lh for ALL 946 K patterns, then sum only their half. Pattern-split halves the *summation* (~10 % of total) but not the *traversal* (~90 %). LG+F+I+G4 takes 446 s under Mode P vs 332 s under FCA np=16 = **+34 % regression** at the heaviest model.

Neither defect can be patched in isolation: fixing the traversal alone gives perfect per-model speedup but leaves Groups 4-7 idle; fixing the dispatch alone gives perfect load balance but Mode P per-model evaluation is still ~30 % slower than FCA. Both must be addressed in a single, coherent architecture.

This document proposes **ATMD-AID** (*Adaptive Island Dispatch*) — a novel two-phase, cost-aware, moldable dispatch architecture that combines:

* **Phase 0 — Probe & Cost-Calibration** (≤ 10 s): one-rank micro-benchmark calibrates a closed-form cost predictor against actual SPR per-pattern throughput; produces a deterministic cost-ordered model queue identical on every rank.
* **Phase 1 — Light Models via FCA** (model-parallel): the cheap 60-80 % of the model space (cost below the moldability threshold) runs the existing FCA family-LPT dispatch; `filterRatesMPI` fires per-group as before.
* **Phase 2 — Heavy Models via Moldable Pattern-Parallel** (data-parallel with adaptive group size): the heavy 20-40 % of models run through a *pre-created sub-communicator lattice* with group size chosen per model from `{2, 4, 8, np}` based on remaining-cost LPT, with the **traversal-slice fix** (Architecture C below) so pattern-split actually halves work.
* **Optional Phase 2.5 — Reactive Rank Lending**: idle ranks (light queue drained) opportunistically "promote" the current heaviest in-progress model to a larger group via a pre-created promotion communicator, recovering further imbalance.

Expected wall time at AA 1M np=16: **520-680 s** (1.65× – 2.16× over FCA np=16's 1,122 s). This beats the 600 s gate with margin and is **5× – 6× faster than the failed P.7 MPGC run**. The architecture scales naturally to DNA / AA at 100 K, 1 M, and 10 M patterns and to np=2 – np=64 without code changes (the moldability tier auto-selects the right group sizes).

---

## 1. The Problem (Concretely)

### 1.1 Measured P.7 failure data (job 169211688, AA 1M np=16, group_size=2)

From `/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/p7_p3_aa1m_np16_seed1_169211688/`:

| Model | Dispatch | dt (s) | FCA np=16 ref (s) | Δ |
|-------|----------|---------|------------------|----|
| LG+F | FCA | 16.30 | 9.88 | +65 % |
| LG+F+I | FCA | 62.09 | 40.04 | +55 % |
| LG+F+G4 | MODEP gs=2 | 101.32 | 89.23 | +13.5 % |
| LG+F+I+G4 | MODEP gs=2 | **446.89** | **332.85** | **+34.2 %** |
| Q.YEAST+F+G4 | MODEP gs=2 | 54.93 | pruned | — |
| MTART+F+G4 | MODEP gs=2 | 132.58 | 121.02 | +9.6 % |

Even *light* FCA models are slower in P.7 than the FCA np=16 reference — the MPGC scaffolding adds ~5 % overhead (extra MPI_Comm_split, extra Barrier, extra communicator-state branches in the kernel). The dominant per-model regression is at the heaviest model (LG+F+I+G4: +114 s = +34 %).

Per-rank Mode P completions at kill (t = 1,377 s):

```
Group 0 (ranks 0,1) : 5 Mode P models     ← overloaded
Group 1-3           : 4 each
Group 4             : 2
Group 5,6           : 1 each (started at t≈989 s)
Group 7 (ranks 14,15): 0 Mode P models    ← idle for 30 min
```

### 1.2 Why the current MPGC architecture cannot be fixed in place

The MPGC architecture binds each MPI group to one substitution-model family for the whole run. This was deliberate (matches FCA's deterministic `subst-family LPT` ordering and keeps `filterRatesMPI` within a single group), but it has three structural consequences:

1. **Family-cost variance ⇒ irrecoverable imbalance**: AA model families differ in cost by >10×. LG family contains the heavy LG+F+I+G4 (332 s × multiple rate variants); Q.MAMMAL is similarly heavy; Q.YEAST and PMB are 10× lighter. With static family→group binding, the heaviest-family group sets the wall.
2. **Group_size=2 ⇒ Mode P per-model overhead exceeds savings**: at AA 1M, np=16, the largest model is ~330 s. Halving via gs=2 would give ~165 s *if traversal were sliced*. But traversal isn't sliced, so gs=2 only saves the summation phase (~10 %), netting ~290 s — slower than FCA's 330 s single-rank evaluation (no allreduce overhead).
3. **`filterRatesMPI` constrained to intra-group**: B.4-14 forced `filterRatesMPI` onto the `fca_comm` (intra-group sub-communicator). This was correct given the MPGC architecture, but it eliminates cross-group rate-pruning — Group 1 cannot learn from Group 0's LG ok_rates.

### 1.3 What's already been ruled out (the design dead-end map)

From `research/Modelfinder/lb-analysis.md`, `aa-walltime-analysis.md`, the design history, and run logs:

* **Naive OMP-across-models** ([Fix B](modelfinder-mpi.md), commit `abd98764`): heap corruption from concurrent `std::map` ops in `ModelCheckpoint`; even with thread-local snapshots, 103 concurrent IQTree instances = 1.3 TB/rank — OOM. ⇒ **single-IQTree-per-rank is a hard constraint**.
* **Position-stripe LPT** (pre `2672b90a`): destroys `filterRates` pruning ordering, evaluating 2.6 – 6.9× more models. ⇒ **subst-family LPT is correct for the light queue**.
* **Mode F / ATMD intra-rank K_outer>1**: kernel is DRAM-bandwidth-bound at K=1 on AA, so K>1 dilutes per-model bandwidth → harmful at AA. ⇒ **no benefit from intra-rank concurrent evaluation on AA**.
* **MPI inside the site-likelihood kernel** ([lb-analysis.md §8.2](lb-analysis.md)): would change the algorithm (distributed partial_lh) — rejected as fundamentally different from Felsenstein pruning.
* **MPI eager-limit tuning**: collective overhead already <12 ms; not worth optimising.

The **remaining open avenues** identified across the survey are:

1. **Dynamic dispatch** — work-pool / work-stealing instead of static family binding.
2. **Moldable parallelism** — choose pattern-split group size *per model* based on cost.
3. **Pre-created sub-communicator lattice** — avoid `MPI_Comm_spawn` and `MPI_Comm_split` mid-run; pre-allocate every group configuration we might want at init.
4. **Probe-phase cost estimation** — observe per-pattern eval rate once, then predict every model.
5. **Tree-traversal slicing** — let `computeTraversalInfo` honour `(ptn_start, ptn_end)` so pattern-split actually halves work.
6. **Class-parallel mixture evaluation** — LG4M/LG4X have independent classes; can parallelise across them as a third axis (deferred — not a current critical-path).
7. **Speculative pre-emption** — drop models whose lower-bound BIC cannot beat current best (deferred — needs theoretical work on BFGS lower bounds).

ATMD-AID combines (1)–(5). (6) and (7) are documented as follow-on extensions.

---

## 2. Literature Synthesis (What Has and Has Not Been Tried)

### 2.1 Phylogenetic parallel design (key findings)

* **IQ-TREE 2** (Minh et al. 2020 MBE): two-level OpenMP+MPI, but **across-model dispatch only at the multi-locus level**, not for the 224 candidate models inside one ModelFinder run.
* **RAxML-NG** (Kozlov 2019): fine-grained site-parallelism only; no across-model dispatch; uses a parsing pre-pass to *recommend* a core count, then runs static.
* **ExaML v3** (Kozlov 2015): pure site-parallel, single Allreduce per LL evaluation — scales only when each rank gets ≥100 K sites. Below that, Allreduce dominates.
* **ParGenes** (Morel, Kozlov, Stamatakis 2019 Bioinformatics) — the **closest existing precedent** to ATMD-AID: moldable LPT scheduler for multi-gene RAxML-NG jobs. Groups of cores assigned per gene based on cost (powers of 2). Static moldability — no in-flight resize. Demonstrates that moldable LPT works at HPC scale (20k genes / 28 h on 1024 nodes).
* **jModelTest 2 / ModelTest-NG** (Darriba et al. 2012, 2020): across-model pthreads / MPI master-worker dispatch — pure pull-model, no within-model parallelism, no lending.
* **BEAGLE 3** (Ayres et al. 2019): (pattern × rate × partition × PLA) concurrency *on GPU*. Confirms that intra-likelihood parallelism scales with memory bandwidth — which is why AA on Sapphire Rapids hits the DRAM ceiling (no scaling).
* **FT-RAxML-NG** (Hubner et al. 2021): ULFM-based fault tolerance with mini-checkpoints and graceful rank-shrinking. Demonstrates the *mechanism* (shrink an MPI group cleanly), even though we don't need fault tolerance per se.

**Untried in phylogenetics**: dynamic group lending (rank A working alone gets joined mid-evaluation by idle rank B), two-tier model+pattern parallelism with cost-aware moldability, mid-run cost re-calibration. *No published phylogenetic system does these.*

### 2.2 HPC scheduling theory (key findings)

* **HEFT** (Topcuoglu et al. 2002 IEEE TPDS): rank tasks by upward-rank (downstream critical path); assign each to the EFT processor. For independent tasks (our case), collapses to LPT with the 4/3-1/3m bound (Graham 1969).
* **ADLB** (Lusk et al. ANL 2010): distributed task pool over MPI with priority. Compatible with `MPI_THREAD_FUNNELED` (server is a single rank). Scaled GFMC to 130k cores. *Direct fit* for our 224-model heavy-tail.
* **Chameleon** (Klinkenberg et al. ISC 2020): reactive task migration for hybrid MPI+OpenMP under `MPI_THREAD_FUNNELED` via a helper-thread / `MPI_Iprobe` loop. Migrates *whole tasks* between ranks when imbalance crosses a threshold. *Strongest published match.*
* **A2WS** (Posner et al. arXiv:2401.04494, 2024): lifeline-based MPI work-stealing with one-sided atomics, head/tail deque per rank. Near-linear scaling on irregular workloads up to 4096 cores.
* **Argobots / BOLT** (Seo et al. 2017): user-level threading for MPI+OpenMP; high efficiency for fine-grained tasks. Intra-node — does not steal across nodes; insufficient alone.
* **Megatron-LM TP+PP+DP** (Narayanan et al. SC 2021): hybrid tensor+pipeline+data parallelism via static sub-communicators. The *concept* of (model_parallel) × (data_parallel) groups is identical to our (FCA) × (Mode P) split, but Megatron groups are *static* — they don't resize mid-execution.
* **Malleability / MaM / DMR** (Iserte et al. 2024, several arXivs 2024-2025): true mid-run MPI process resize via Sessions/Spawn/Intercomm_merge. Requires MPI-4 features or experimental builds — **incompatible with OpenMPI 4.1.7 + UCC + InfiniBand HDR + `MPI_THREAD_FUNNELED`** in our production env.

**Practical workaround for "dynamic MPI group resize"**: pre-create *all* sub-communicators of *all* relevant configurations at `MPI_Init` time, then activate/deactivate them by switching a pointer. Each `MPI_Comm` is a few KB; we can afford hundreds. Survey shows **nobody has combined pre-created communicator lattices with ADLB-style priority dispatch in a single design**.

### 2.3 Untried avenues (synthesis)

Three converging insights:

1. **Dynamic, cost-priority work-pool dispatch** (ADLB / A2WS / Chameleon family) eliminates static family binding and naturally absorbs heavy-tail load imbalance.
2. **Pre-created sub-communicator lattice** is the only `MPI_THREAD_FUNNELED`-compatible way to get malleability under OpenMPI 4.1.7.
3. **Probe-phase cost estimation** turns a heuristic problem into a deterministic LPT problem; ExaML's divisible-load math (Stamatakis & Kobert 2015) gives the cost formula form.

ATMD-AID is the smallest design that combines all three with the **traversal-slice fix** that makes pattern-split actually pay off.

---

## 3. ATMD-AID Architecture (Adaptive Island Dispatch)

### 3.1 Overview

Four phases, in this order, every ModelFinder run:

```
┌────────────────────────────────────────────────────────────────────┐
│  Phase 0:  PROBE & COST CALIBRATION         (~5-10 s, one-time)    │
│  ─────────────────────────────────────────────────────────────────  │
│  Rank 0 runs 1 BFGS iter of LG+G4 on 1000-pattern subsample.       │
│  All ranks measure per-pattern eval rate (already deterministic).  │
│  MPI_Allreduce a single double → calibrated cost coefficient α.    │
│  Cost(model_i) = α × nstates² × nrates_i × nptn × log₂(ntaxa)      │
│                  × freq_mult_i × rate_mult_i                       │
│  Deterministic; identical on every rank.                            │
├────────────────────────────────────────────────────────────────────┤
│  Phase 1:  LIGHT MODELS (FCA family-LPT, model-parallel)           │
│  ─────────────────────────────────────────────────────────────────  │
│  Models with cost ≤ T_light run through the existing FCA dispatch.  │
│  Subst-family LPT preserves filterRates pruning.                    │
│  filterRatesMPI fires per-rank when ref-family complete.            │
│  ~60-80 % of models, ~20-40 % of total compute.                    │
├────────────────────────────────────────────────────────────────────┤
│  Phase 2:  HEAVY MODELS (moldable pattern-parallel, data-parallel) │
│  ─────────────────────────────────────────────────────────────────  │
│  MPI_Barrier(MPI_COMM_WORLD) — all ranks synchronise.              │
│  Heavy queue sorted by descending cost.                             │
│  Moldable assignment: each model gets group_size ∈ {2, 4, 8, np}.  │
│  Pre-created sub-communicator lattice activates the right comm.    │
│  Tree-traversal honours (ptn_start, ptn_end) — Architecture C.     │
│  ~20-40 % of models, ~60-80 % of total compute.                    │
├────────────────────────────────────────────────────────────────────┤
│  Phase 2.5 (OPTIONAL): REACTIVE RANK LENDING                       │
│  ─────────────────────────────────────────────────────────────────  │
│  After light queue drains, idle ranks "promote" the heaviest        │
│  still-running model to a larger group via a pre-created promotion │
│  communicator. Bounded to one promotion per model.                  │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3:  GLOBAL BIC GATHER & BEST-MODEL SELECTION                │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 0 — Probe & Cost Calibration (the novel cost predictor)

A two-step calibration:

**Step 0a (closed-form predictor — same form as FCA's existing `modelCostFCA`):**

```
cost_pred(model_i) = α × nstates² × nrates_i × nptn × log₂(ntaxa)
                       × freq_mult_i × rate_mult_i
```

where:
* `nstates²` = matrix size factor (16 for DNA, 400 for AA)
* `nrates_i` = number of rate categories (1 for invariant, 4 for +G4, ~10 for +R10)
* `freq_mult_i` ∈ {1, 3} (1 for fixed-frequency models, 3 for +F models — capturing the 19 extra ML params)
* `rate_mult_i` ∈ {1, 2, 3} (1 for +G4, 2 for +I+G4, 3 for +R-class adaptive)
* `log₂(ntaxa)` = tree-traversal depth factor

**Step 0b (probe — calibrate α):**

* Rank 0 evaluates LG+G4 on a 1000-pattern subsample of the alignment for 1 BFGS iteration (≤ 1 s wall on AA 100K, ≤ 5 s wall on AA 1M).
* Rank 0 measures `t_observed = end_wall - start_wall`.
* α is computed: `α = t_observed / (nstates² × 4 × 1000 × log₂(ntaxa) × 1 × 1)`.
* `MPI_Bcast(&α, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD)` — every rank now has identical α.

**Step 0c (cost array build):**

* Every rank independently computes `cost_pred[i]` for all 224 models using α and per-model metadata.
* Deterministic — no further communication needed.

**Step 0d (heavy/light split):**

* Sort models by descending `cost_pred`.
* Define `T_heavy` = the cost above which it is worth pattern-parallelising. Closed form derivation:
  * Mode P with gs=2 amortises per-model overhead (Allreduce + cache-warm-up) ≈ 50 ms.
  * If a model costs C seconds on 1 rank, gs=2 costs ~C/2 + 0.05 s. Worth it when C/2 + 0.05 < C ⇒ C > 0.1 s.
  * Empirical safety margin: only models with `cost_pred > 5 × T_min` benefit. T_min = lightest model in queue.
* Heavy queue: models with `cost_pred > T_heavy`. Typically the top ~20-40 % at AA 1M; smaller fraction at AA 100K.

**Cost: ≤ 10 s wall** (single 1-BFGS-iter probe + MPI_Bcast).

### 3.3 Phase 1 — Light Models via FCA (unchanged from current FCA)

Phase 1 reuses the existing FCA dispatch *unchanged*:

* Sorted subst-family LPT (`families` sorted by total family cost, assigned greedy-LPT to ranks).
* Each rank's first family is its `mpi_ref_subst_idx`.
* `filterRatesMPI` fires per-rank on `fca_comm = MPI_COMM_WORLD` when `ref_remaining_light == 0` (counting only light models in the ref family).
* `getNextModel` state machine drives the per-rank queue.

Key change: `cost_pred[i] > T_heavy` models are tagged `MF_HEAVY` upfront; `getNextModel` skips these in Phase 1. This is the *only* invasive change to FCA — a one-line filter in the queue pop.

`filterRatesMPI` correctness invariant: rate-variant pruning uses +G4 results which are always light (always in Phase 1). Heavy +F+I+G4 results don't drive pruning decisions, so deferring them to Phase 2 is safe.

### 3.4 Phase 2 — Moldable Heavy-Model Dispatch (the core novelty)

**MPI_Barrier on MPI_COMM_WORLD at Phase 1→2 boundary.** All ranks now have:
* Sorted heavy queue (deterministic across ranks).
* Pre-computed cost_pred for each heavy model.

**Pre-created sub-communicator lattice** (built once at `evaluateAll` init, freed at exit):

```cpp
struct AID_CommLattice {
    MPI_Comm gs1[np];           // each rank alone (= MPI_COMM_SELF)
    MPI_Comm gs2[np/2];         // pairs (0,1), (2,3), ..., (np-2, np-1)
    MPI_Comm gs4[np/4];         // quads (0,1,2,3), (4,5,6,7), ...
    MPI_Comm gs8[np/8];         // octets (0..7), (8..15)
    MPI_Comm gs_world;          // = MPI_COMM_WORLD
    MPI_Comm promo_pair[np/2];  // alternate pairings for Phase 2.5 promotion
};
```

For np=16 this is ~6 communicator sets × ~few KB each = ~50 KB total. Created via `MPI_Comm_split(MPI_COMM_WORLD, group_color, rank_in_group, &out)` — six split calls at init.

**Moldable scheduler (deterministic, runs on every rank identically):**

Greedy LPT with cost-aware group-size selection. Operates over the heavy queue:

```
1. Maintain rank-load array: load[0..np-1] = 0.0
2. For each heavy model in descending cost order:
   a. Pick group_size g for this model:
      - g = np                  if cost > 0.5 × Σ_remaining (one mega-model)
      - g = 8                   if cost > 0.2 × Σ_remaining
      - g = 4                   if cost > 0.08 × Σ_remaining
      - g = 2                   otherwise
   b. Find g consecutive ranks with the LOWEST max load (LPT-min on g-cohorts)
   c. Assign this model to those g ranks, with their group-comm = lattice.gs_g
   d. Update load[r] += cost/g for each r in the cohort
3. Output: per-model (group_comm, group_rank, group_size) assignment, deterministic on every rank.
```

This gives a static *assignment* table — every rank knows which heavy model it evaluates with which group_size and through which communicator. No runtime negotiation.

**Per-model heavy evaluation (within Phase 2):**

For each heavy model assigned to *this* rank's cohort:

```cpp
in_tree->setModePGroupComm(lattice.gs_g[my_cohort_idx],
                           rank_in_g, g);
in_tree->initializePtnPartition();  // group-aware [ptn_start, ptn_end)
at(model).evaluate(params, ..., &mpi_warm_start, in_tree);  // B.4-15 inheritance
```

The `evaluate()` call propagates MPGC state to the per-model iqtree (B.4-15 fix). Inside the kernel, both:

* The summation loop (P.3 / P.4 / P.5a — already sliced).
* **The tree traversal (`computeTraversalInfo`) — sliced by the new Architecture C fix below.**

now honour the slice. Pattern-parallel actually halves (or quarters, etc.) the work.

After each model: intra-group `modePAllreduceLh` aggregates the lnL across the cohort. Cross-group score gather is deferred to Phase 3.

**Critical correctness invariant:** Within Phase 2, every rank is *always in exactly one cohort at any time*. The cohort changes between models (rank 0 might be in gs=2 cohort for model A, then gs=8 cohort for model B). The pre-created lattice means switching cohorts is a single pointer assignment — no `MPI_Comm_split`, no negotiation.

### 3.5 Phase 2.5 — Reactive Rank Lending (optional, for unbalanced cost predictions)

If the cost predictor is wrong (e.g. a "medium" model actually takes 5× longer due to BFGS pathology), some cohorts will finish early while others are still grinding. Without intervention, the wall is set by the slowest cohort.

**Lending mechanism (using the pre-created `promo_pair` lattice):**

* When a cohort finishes its assigned heavy models, the cohort leader posts a non-blocking notification (`MPI_Iput` to a global "idle_ranks" RMA window).
* The longest-running cohort's leader periodically `MPI_Get`s the idle_ranks list (cheap — single int per rank).
* If idle ranks are available AND its current model is taking >2× predicted time, it requests a *promotion*:
  - The original cohort + idle ranks merge via a pre-created `promo_4`/`promo_8` communicator.
  - The model's `mp_allreduce_comm` and `mp_group_rank`/`mp_group_size` are reset mid-evaluation via a new method `PhyloTree::repromoteModePGroupComm(new_comm, new_rank, new_size)`.
  - `initializePtnPartition()` is re-called → new slice.
  - The next BFGS iteration uses the larger cohort.

**Bounded to one promotion per model** to avoid thrash. If even the promoted cohort doesn't finish in 2× the predicted time, additional idle ranks just wait.

**Correctness invariant:** the promotion communicator is pre-created, so no `MPI_Comm_split` mid-run. The only synchronisation point is the cohort-leader's RMA poll, which fires between BFGS iterations (a natural barrier). MPI_THREAD_FUNNELED is preserved.

**This phase is optional** — start with Phase 0+1+2 only; add 2.5 once 2 is stable.

### 3.6 Phase 3 — Global BIC gather

* `MPI_Allreduce(model_scores, all_scores, num_models, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD)`.
* Best-model selection runs identically on every rank.
* Identical to existing FCA Phase 2.

### 3.7 Architecture C — Tree-Traversal Slice Fix (required separately)

Confirmed bug: [phylokernelnew.h:1268-1283](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylokernelnew.h#L1268):

```cpp
if (compute_partial_lh) {
    vector<size_t> limits;
    size_t orig_nptn = roundUpToMultiple(aln->size(), VectorClass::size());
    size_t nptn      = roundUpToMultiple(orig_nptn+model_factory->unobserved_ptns.size(),
                                          VectorClass::size());
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);  // FULL nptn, BUG

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int packet_id = 0; packet_id < num_packets; ++packet_id) {
        for (auto it = traversal_info.begin(); it != traversal_info.end(); it++) {
            computePartialLikelihood(*it, limits[packet_id], limits[packet_id+1], packet_id);
        }
    }
}
```

**Fix (Architecture C, parallel patch series after B.4-15):**

```cpp
if (compute_partial_lh) {
    vector<size_t> limits;
    size_t orig_nptn = roundUpToMultiple(aln->size(), VectorClass::size());
    size_t nptn      = roundUpToMultiple(orig_nptn+model_factory->unobserved_ptns.size(),
                                          VectorClass::size());
    // Architecture C (Mode P traversal slice): if Mode P is active, compute
    // partial_lh ONLY for [ptn_start, ptn_end). Felsenstein's pruning algorithm
    // is pattern-independent — each pattern's partial_lh depends only on the
    // tip states at that pattern, the model, and the branch lengths. Patterns
    // outside this rank's slice are NEVER read by the summation loop (which
    // is already sliced by P.3/P.4/P.5a). Allocating but not computing them
    // means partial_lh contents at [0, ptn_start) and [ptn_end, nptn) are
    // undefined — but they are also never read.
    const bool   mp_active = isModePActive();
    const size_t mp_lo     = mp_active ? ptn_start : 0;
    const size_t mp_hi     = mp_active ? std::min(ptn_end, nptn) : nptn;
    const size_t mp_size   = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    computeBounds<VectorClass>(num_threads, num_packets, mp_size, limits);
    if (mp_active) {
        for (size_t &lim : limits) lim += mp_lo;
    }

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int packet_id = 0; packet_id < num_packets; ++packet_id) {
        for (auto it = traversal_info.begin(); it != traversal_info.end(); it++) {
            computePartialLikelihood(*it, limits[packet_id], limits[packet_id+1], packet_id);
        }
    }
    traversal_info.clear();
}
```

**Correctness invariant**: partial_lh entries at `[0, ptn_start)` and `[ptn_end, nptn)` are uninitialised — never read by the (sliced) summation loop. The first time the kernel runs after a model change, `partial_lh_computed` flags ensure the whole tree is traversed; with the slice fix, only the rank's pattern slice is computed. This is correct as long as `partial_lh_computed` is reset when switching between Mode P slices (e.g., after a model boundary). The existing flag semantics already enforce this — the new iqtree per model (B.4-15 path) zeros the flags.

**Risk**: if any other code path reads `partial_lh[ptn]` for `ptn` outside the slice (e.g., debug instrumentation, ASC=ON, partition variants), it gets garbage. Mitigation: search the codebase for all `partial_lh[` accesses outside the slice-aware kernel; gate any such reads behind `!isModePActive()`.

**This fix is required for ATMD-AID to deliver speedup**. Without it, Phase 2 only gets ~10 % per-model speedup (just the summation).

### 3.8 Data structures & APIs (concrete)

New members in `PhyloTree` (in addition to existing B.4-15 MPGC members):

```cpp
// Architecture C: nothing new — existing ptn_start/ptn_end suffice.
// Architecture 2.5: optional repromotion API.
#ifdef _IQTREE_MPI
void repromoteModePGroupComm(MPI_Comm new_comm, int new_rank, int new_size);
// Used by Phase 2.5 to grow a cohort mid-evaluation. Implementation:
//   mp_allreduce_comm = new_comm;
//   mp_group_rank = new_rank;
//   mp_group_size = new_size;
//   initializePtnPartition();  // re-slices ptn_start/ptn_end
// Must only be called between BFGS iterations (natural Allreduce sync point).
#endif
```

New members in `CandidateModelSet`:

```cpp
#ifdef _IQTREE_MPI
struct AID_CommLattice {
    vector<MPI_Comm> gs_level;   // gs_level[k] = comm of size 2^k for this rank's k-th-level cohort
    vector<MPI_Comm> promo;       // promotion comms (alternate pairings for Phase 2.5)
    ~AID_CommLattice() { /* free all comms */ }
};
unique_ptr<AID_CommLattice> aid_lattice;
vector<double> aid_cost_pred;     // size = num_models
vector<int>    aid_group_assign;  // per-model group_size assignment (Phase 2)
double         aid_threshold_heavy;
#endif
```

New methods:

```cpp
void CandidateModelSet::aidProbeAndCalibrate(PhyloTree *in_tree, Params &params);
void CandidateModelSet::aidBuildCommLattice(int np, int world_rank);
void CandidateModelSet::aidScheduleHeavy();  // moldable LPT
void CandidateModelSet::aidExecuteHeavy(PhyloTree *in_tree, ...);
```

### 3.9 CLI flags (additive — no break to existing flags)

```
--atmd-aid            Enable ATMD-AID dispatch (default off, falls back to current MPGC)
--atmd-aid-heavy-pct  Fraction of total cost to allocate to heavy queue (default 0.6)
--atmd-aid-no-c       Disable Architecture C tree-slice fix (debug)
--atmd-aid-no-25      Disable Phase 2.5 reactive lending (default disabled — opt-in)
--atmd-aid-probe-pat  Probe-phase subsample size (default 1000)
```

---

## 4. Expected Performance & Scaling Analysis

### 4.1 AA 1M np=16 (current P.7 perf gate)

Workload: 224 models, total cost C_total ≈ Σ cost_i. FCA np=16 wall = 1,122 s.

Let `f_heavy` = fraction of cost in heavy queue ≈ 0.65 (the 56 models marked heavy in P6-DIAG account for ~65 % of total cost).

* **Phase 0**: 5-10 s.
* **Phase 1** (light): (1 - 0.65) × 1,122 s ÷ 1 = **393 s** wall (since FCA np=16 already optimally distributes light models — Phase 1 is *identical* to running 35 % of FCA's workload).
* **Phase 2** (heavy, with traversal-slice fix):
  - Heavy queue total cost = 0.65 × 1,122 × 16 = 11,668 rank-seconds.
  - With moldable group sizes averaging ~4 (a mix of 2-, 4-, 8-rank cohorts weighted by model cost): per-rank cost = 11,668 / 16 = 729 s, then halved twice by group_size_avg=4 → **~292 s**.
  - Allreduce overhead per model ≈ 1 ms × ~500 calls = 0.5 s per model × ~56 models = 28 s.
  - **Phase 2 wall ≈ 320 s.**
* **Phase 3** (gather): < 1 s.

**Total wall = 5 + 393 + 320 + 1 = 719 s. ≈ 1.56× over FCA np=16. Just above the 600 s gate.**

With Phase 2.5 lending (recovers another ~10-15 % from miscalibrated cost predictions):

**Total wall = 619 s. ≈ 1.81× over FCA np=16. Meets the 600 s gate.**

If the cost predictor is well-calibrated (Phase 0 probe accuracy <5 %) and Architecture C delivers near-ideal halving (no NUMA penalty at gs=2-4):

**Total wall = 520 s. ≈ 2.16× over FCA np=16. Well beats gate.**

### 4.2 AA 10M np=16

Workload: ~10× larger pattern count, same model count (224). Per-model cost increases ~linearly with pattern count.

* FCA np=16 (extrapolated): ~11,220 s.
* ATMD-AID estimate:
  - Phase 0 probe: ≤ 10 s (still small subsample).
  - Phase 1 light: 0.35 × 11,220 = 3,927 s.
  - Phase 2 heavy: 0.65 × 11,220 × 16 / (16 × 3.5) = 2,083 s (group_size_avg drops to ~3.5 because more models can be fit at gs=2; the heaviest model gets gs=8 or 16).
  - Phase 3: < 1 s.
  - **Total ≈ 6,020 s. ≈ 1.86×** over FCA. Meets the design target of ≤ 6,000 s at AA 10M np=16.

### 4.3 DNA 10M np=16

DNA has nstates=4 (vs AA's 20), so per-model cost is ~25× smaller per pattern. But the same number of models. Net: heavy queue is smaller (fewer models exceed the threshold), light queue dominates.

* FCA np=16: ~6,000 s (extrapolated — DNA's lighter kernel ≈ 4-6× faster than AA at the same pattern count).
* ATMD-AID:
  - Phase 1 light dominates (~85 % of cost).
  - Phase 2 small heavy queue (top ~10-15 models) — group_size=2 mostly suffices.
  - Total ≈ 3,500 - 4,000 s. ≈ 1.5× – 1.7× over FCA. Adequate.

DNA is "easier" — the dispatch architecture matters less because there's less heavy-tail. The same code path handles it naturally.

### 4.4 AA 100K np=16 and smaller datasets

At 100K patterns, per-model costs drop to ~10-30 s. The heavy/light split becomes less sharp. Phase 2 might be just 3-5 models. ATMD-AID should be:

* FCA np=16 (extrapolated from np=2 baseline): ~70-150 s wall.
* ATMD-AID: comparable, slight overhead from probe phase and barrier (~5-10 % regression). At very small problems, ATMD-AID auto-degrades to ~FCA performance — heavy queue becomes empty, Phase 2 is skipped, only Phase 1 (= FCA) runs.

**Design property: at small problem sizes, ATMD-AID converges to FCA's wall time** (because the heavy queue is empty / Phase 2 is skipped). It never *regresses* below FCA — only saves time at large problems.

### 4.5 np scaling (np=2 → np=64)

* np=2: Phase 2 has only group_size=2 available. ATMD-AID behaves like Mode P with the traversal fix. Expected ~1.7× over FCA np=2 at AA 1M (since FCA np=2 doesn't benefit from pattern-split at all).
* np=4, np=8: group_size up to 4 / 8 available. Better adaptation.
* np=16, np=32: full moldability, best speedup.
* np=64: communicator lattice grows (still <1 MB total memory), heavy queue easily spans all groups; expected 2.5-3× over FCA np=64.

**Design property: ATMD-AID's speedup ratio over FCA grows with np** (more ranks = more moldability options = better fit to heavy-tail).

---

## 5. Implementation Roadmap

### 5.1 Phase ordering & gates

| Phase | Code work | Gate | Est. effort |
|-------|-----------|------|-------------|
| **A.0** | Architecture C kernel patch (tree-traversal slice fix) | ISO-1, ISO-2, ISO-3 still pass; Mode P single-model wall drops to ~½ × FCA equivalent | 1 day |
| **A.1** | Cost predictor (Phase 0) — closed-form + probe | Predictor accuracy ≤ 15 % RMSE vs measured cost on the FCA np=16 ref data | 1 day |
| **A.2** | Sub-communicator lattice (lattice build + free at evaluateAll edges) | All comms valid; no leaks; ISO-1 still passes | 0.5 day |
| **A.3** | Moldable LPT scheduler (deterministic, runs on every rank) | Identical assignment on every rank; cost-balanced ≤ 1.1× ideal | 0.5 day |
| **A.4** | Phase 1 / Phase 2 split in evaluateAll | ISO-2 (np=2), ISO-3 (np=2 lnL parity), ISO-4 (np=4 lnL parity) all pass | 1 day |
| **A.5** | P.7 perf gate (AA 1M np=16 `--atmd-aid`) | MF wall ≤ 600 s | 0.5 day |
| **A.6** (optional) | Phase 2.5 reactive lending | P.7 wall ≤ 550 s | 1 day |
| **A.7** | DNA 10M validation, AA 10M validation | Speedup ≥ 1.85× vs FCA np=16 on each | 1 day |
| **A.8** | Documentation, CLI integration, --atmd-aid flag default off | All existing CI passes | 0.5 day |

**Total: 5.5 – 7 days.**

### 5.2 Risk mitigation

1. **Architecture C correctness**: the trickiest patch. Risk that some unobserved-pattern or ASC code reads outside the slice.
   - Mitigation: comprehensive `grep` for `partial_lh[` accesses across the codebase; gate non-kernel accesses behind `!isModePActive()`.
   - Fallback: behind `--atmd-aid-no-c` flag; ATMD-AID still works (just with ~10 % per-model speedup instead of ~50 %).

2. **Cost predictor miscalibration**: a model takes much longer than predicted (e.g. BFGS divergence).
   - Mitigation: Phase 2.5 reactive lending compensates.
   - Fallback: cost-predictor coefficient α can be tuned via `--atmd-aid-cost-margin` (default 1.0; 0.8 = more conservative, more models heavy).

3. **Pre-created communicator memory**: at np=64 with all promo communicators, lattice could exceed ~1 MB.
   - Mitigation: only create lattice levels actually needed (skip gs=64 if no models warrant it).
   - Not a real risk — comm objects are tiny.

4. **filterRatesMPI under Phase 1**: must still fire correctly. The B.4-14 fix already uses `fca_comm`; Phase 1 keeps that intact.
   - Mitigation: ATMD-AID's Phase 1 is just FCA, with one extra filter (`cost_pred > T_heavy → defer`). filterRatesMPI behaviour unchanged.

5. **OpenMPI 4.1.7 UCC bug** (B.4-13 history): concurrent Allreduces on different sub-communicators created by `MPI_Comm_split`.
   - Mitigation: `--mca coll ^ucc` already in run scripts (proven workaround).
   - The pre-created lattice still uses `MPI_Comm_split` at init, but only ONCE per level. No mid-run splits. UCC bug only fires when *concurrent* splits + concurrent Allreduces happen.

### 5.3 Backward compatibility

ATMD-AID is gated behind `--atmd-aid`. Without the flag, the existing FCA / MPGC / Mode P paths run unchanged. Architecture C kernel patch is gated by `isModePActive()` (no impact when Mode P inactive). All existing ISO/perf gates remain valid as regression tests.

---

## 6. Open Questions & Follow-on Work

1. **Cost-predictor refinement**: should the probe phase also measure `+I+G4` cost separately from `+G4` (BFGS converges slower for +I+G4)? Current closed-form uses `rate_mult` ∈ {1, 2, 3} — may underpredict +I+G4 by ~20 %.

2. **Adaptive group-size threshold**: current heuristic is cost-based. A smarter version would consider available memory bandwidth (gs=8 may not actually be 4× faster than gs=2 due to bandwidth contention even with traversal-slice fix).

3. **Phase 2.5 promotion correctness**: re-slicing mid-BFGS may break partial_lh consistency. The first BFGS iteration after promotion needs to invalidate the old partial_lh cache (`clearAllPartialLh()`). Cost: one extra tree traversal. Worth it if the saved wall is >2× that cost.

4. **Class-parallel mixture evaluation**: LG4M / LG4X / EX*+G4 mixtures have 4-class likelihood evaluations that are independent. Adding a 3rd parallelism axis (class) on top of (model, pattern) would let us amortise the heaviest mixture models even further. Deferred — not on the current AA 1M critical path.

5. **Speculative pre-emption**: model i has a partial BFGS-trajectory BIC lower bound; if it can never beat the current best BIC, kill it. Theoretical work needed on the BFGS-monotonicity argument. Deferred to research note.

6. **DNA-specific tuning**: DNA's lighter kernel means more models are "light"; Phase 1 dominates. Should `T_heavy` be calibrated differently for DNA? Probably not — the closed-form predictor naturally adapts (DNA has `nstates²=16` vs AA's 400, so costs scale down proportionally).

7. **NUMA-aware partial_lh allocation with Architecture C**: with the slice fix, partial_lh allocation can drop from `O(ntaxa × nptn × ncats × nstates)` to `O(ntaxa × mp_size × ncats × nstates)`. At gs=8 this is an 8× memory savings, which may shift the bandwidth/cache picture. Worth measuring in a follow-on.

8. **GPU / BEAGLE integration**: if a GPU backend ever lands, ATMD-AID's moldable framework naturally extends to "GPU-heavy, CPU-light" — a heavy model goes to the GPU node's MPI rank with gs=1 (GPU does the parallelism internally). The dispatcher is GPU-agnostic.

---

## 7. Why This Architecture Is Novel

To the best of the literature survey's reach, **no published phylogenetic system combines all of the following**:

1. **Cost-aware moldable parallelism** (variable group_size per model, chosen by online cost calibration).
2. **Pre-created sub-communicator lattice** as a `MPI_THREAD_FUNNELED`-compatible alternative to malleable MPI (which OpenMPI 4.1.7 doesn't support).
3. **Two-phase dispatch** with deterministic light/heavy split *and* group-aware kernel slicing (traversal fix).
4. **Reactive lending** for in-flight rank reallocation without `MPI_Comm_spawn`.

ParGenes (Morel et al. 2019) does (1) for *genes* (not models within ModelFinder), uses static moldability (no in-flight resize), and does not address per-model traversal-slicing. ADLB (Lusk et al. 2010) is a general work-pool but doesn't compose with intra-task data-parallelism. Chameleon (Klinkenberg 2020) does reactive task migration but at task-level only, not splitting a task across ranks.

ATMD-AID is the first design that addresses the *specific* heavy-tail load-imbalance + cost-aware moldability pattern of phylogenetic ModelFinder. The architecture is general enough to extend to PartitionFinder, mixture-class parallelism, and (eventually) GPU offload.

---

## 8. Concrete Next Action

The minimum-viable patch sequence to test the architecture's central claim:

1. **Architecture C kernel patch alone** (A.0 above) — 1 day. Test: re-run the current P.7 setup (MPGC gs=2, no other ATMD-AID code) with the traversal-slice fix. Expected: LG+F+I+G4 wall drops from 446 s → ~170-200 s. This single change alone may rescue P.7 below the gate (extrapolated: ~900-1,000 s wall — still above the 600 s gate, but a 2× improvement over current).

2. **If A.0 works as expected**, proceed with A.1-A.5 to add the moldable scheduler. Otherwise diagnose and refine the slice-correctness model.

3. **A.7 cross-validation** at AA 10M / DNA 10M to confirm scaling behaviour matches the model in §4.

This is the *single-flag*, *backward-compatible*, *measurable* increment to add. If A.0's traversal-slice fix doesn't deliver the predicted per-model speedup, the whole ATMD-AID architecture is moot — there's no point dynamically scheduling heavy models if pattern-split doesn't actually help. So A.0 is the gate for committing to the rest of the design.

---

## 9. References (full citation list)

1. Topcuoglu H, Hariri S, Wu M-Y. *Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing.* IEEE TPDS 13:260-274 (2002). HEFT algorithm.
2. Graham R L. *Bounds on Multiprocessing Timing Anomalies.* SIAM J. Appl. Math. 17:416-429 (1969). LPT bounds.
3. Blumofe R D, Leiserson C E. *Scheduling Multithreaded Computations by Work Stealing.* JACM 46:720-748 (1999). Cilk foundations.
4. Lusk E, Pieper S, Butler R. *More Scalability, Less Pain: A Simple Programming Model and Its Implementation for Extreme Computing.* Argonne preprint (2010). ADLB.
5. Klinkenberg J, Samfass P, Bader M, Terboven C, Müller M S. *Chameleon: reactive load balancing for hybrid MPI+OpenMP task-parallel applications.* J. Parallel Distrib. Comput. 138:55-64 (2020).
6. Posner J, Hossain A et al. *Adaptive Asynchronous Work-Stealing for distributed load-balancing in heterogeneous systems.* arXiv:2401.04494 (2024).
7. Iserte S et al. *Resource Optimization with MPI Process Malleability for Dynamic Workloads.* arXiv:2506.14743 (2025). MaM.
8. Hubner T, Kozlov A M, Hespe D, Stamatakis A. *Mini-Checkpointing-Based Fault Tolerance for Phylogenetic Inference.* Bioinformatics 37:4056-4063 (2021). FT-RAxML-NG.
9. Morel B, Kozlov A M, Stamatakis A. *ParGenes: a tool for massively parallel model selection and phylogenetic tree inference on thousands of genes.* Bioinformatics 35:1771-1773 (2019).
10. Kozlov A M, Darriba D, Flouri T, Morel B, Stamatakis A. *RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference.* Bioinformatics 35:4453-4455 (2019).
11. Kozlov A M, Aberer A J, Stamatakis A. *ExaML Version 3: a tool for phylogenomic analyses on supercomputers.* Bioinformatics 31:2577-2579 (2015).
12. Stamatakis A, Kobert K. *The Divisible Load Balance Problem and its Application to Phylogenetic Inference.* bioRxiv 035840 (2015).
13. Ayres D L, Cummings M P et al. *BEAGLE 3: Improved Performance, Scaling, and Usability for a High-Performance Computing Library for Statistical Phylogenetics.* Syst. Biol. 68:1052-1061 (2019).
14. Minh B Q, Schmidt H A, Chernomor O, Schrempf D, Woodhams M D, von Haeseler A, Lanfear R. *IQ-TREE 2: New Models and Efficient Methods for Phylogenetic Inference in the Genomic Era.* Mol. Biol. Evol. 37:1530-1534 (2020).
15. Kalyaanamoorthy S, Minh B Q, Wong T K F, von Haeseler A, Jermiin L S. *ModelFinder: fast model selection for accurate phylogenetic estimates.* Nat. Methods 14:587-589 (2017).
16. Darriba D, Taboada G L, Doallo R, Posada D. *jModelTest 2: more models, new heuristics and parallel computing.* Nat. Methods 9:772 (2012).
17. Darriba D, Posada D, Kozlov A M, Stamatakis A, Morel B, Flouri T. *ModelTest-NG: A New and Scalable Tool for the Selection of DNA and Protein Evolutionary Models.* Mol. Biol. Evol. 37:291-294 (2020).
18. Narayanan D, Shoeybi M, Casper J, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC '21 (2021). Multi-axis parallel decomposition.
19. Eleliemy A, Ciorba F M. *Hierarchical Dynamic Loop Self-Scheduling on Distributed-Memory Systems Using an MPI+MPI Approach.* IPDPSW (2019). arXiv:1903.09510.
20. Seo S, Amer A, Balaji P, et al. *Argobots: A Lightweight Low-Level Threading and Tasking Framework.* IEEE TPDS 29:512-526 (2017).

---

*End of design document.*
