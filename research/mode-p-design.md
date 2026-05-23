# Mode P (Pattern-Parallel Intra-Model MPI_Allreduce) — Design & Roadmap

**Status:** Design only. P.1/P.2 scaffolding is now present in the working IQ-TREE source tree; P.3+ kernel work must first go through the **Mode P kernel ISO sandbox** documented in `research/mode-p-implementation-status.md`. Estimated remaining effort: **~3 working days** including ISO build/test turnaround.

**Source of motivation:** `research/lbfgs-and-warmstart-implementation.md` §15.9.14 ATMD root-cause analysis showed Mode F provides no speedup on AA workloads on Gadi SPR. The §14.9 projected speedup model depends materially on Mode P for the largest contributions (~380 s of the 733 s total projected gain). Mode P was deferred from the B.3+B.4 patch set; this document is the roadmap to closing that gap.

**See also:** `CHANGELOG.md` entries `(cj)`, `(ck)`, `(cm)`, `(cn)`; `research/mode-p-implementation-status.md` (live phase tracker and ISO runbook); phylotesting.cpp lines 4100–4220 (Mode F dispatch); phylotree.cpp lines 842–1001 (likelihood kernel).

---

## 1. What Mode P does

When a single heavy model (e.g., LG+F+I+G4) is evaluated, instead of one MPI rank computing all 1 M patterns × 4 rates × 20 states on its own 103 OMP threads while 15 other ranks sit idle, **all 16 MPI ranks cooperatively evaluate that single model**. Each rank handles a slice of the patterns; after each likelihood iteration an `MPI_Allreduce` combines the per-rank partial log-likelihood sum so all ranks see the same total before the next L-BFGS step.

**Why it works:** the per-pattern likelihood `Σᵢ Pr(siteᵢ | tree, model)` is a sum over independent terms. Splitting the sum across ranks does not change the result; combining via `MPI_SUM Allreduce` is mathematically exact (modulo FP non-associativity, which is bounded and acceptable).

**Why it bypasses Mode F's bandwidth ceiling:** with 16 ranks each owning ~62.5 K patterns, the per-rank working set drops 16× (~64 GB → ~4 GB per rank) and fits comfortably in each node's LLC + DRAM bandwidth. Aggregate cross-node bandwidth = 16 × 500 GB/s = 8 TB/s for the parallel sum; the bottleneck shifts from intra-node DRAM to MPI network latency, which on Gadi InfiniBand HDR is ~1 µs/Allreduce — negligible against 100–500 ms per L-BFGS iteration.

## 2. Why Mode F alone cannot deliver Layer 2/3 of the §14.9 projection

§14.9's three-layer projection:

| Layer | Mechanism | Projected MF wall |
|---|---|---|
| FCA baseline | One model per rank, K_outer=1 | 1,139 s |
| Layer 1 | Mode F K=4 on post-prune light models | 787 s |
| Layer 2 | Mode P on LG+F+I+G4 (332 s → 32 s) | 486 s |
| Layer 3 | Mode P on LG+F+G4 (88 s → 8.5 s) | 406 s |

Layer 1 is what B.3+B.4 implemented. Layers 2 and 3 are Mode P. **Without Mode P, the projection caps at 787 s even in the best case** — and §15.9.14 shows even that is unreachable on AA because the AA per-pattern kernel is memory-bandwidth-bound at K=1.

The heavy models (LG+F+I+G4 ~332 s, LG+F+G4 ~88 s at np=1) sit on the longest critical path of MF; under FCA they are owned by rank 0 alone, idling 15 other ranks for ~420 s of wall time. Mode P reclaims that 15-rank-idle time directly.

## 3. Code map: where Mode P hooks in

### 3.1 The likelihood kernel ([phylotree.cpp / phylokernel.h:842–1001](/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/))

`PhyloTree::computeLikelihoodBranchEigenSIMD` (template instantiated for each state/rate combination). Inner structure:

```cpp
#pragma omp parallel for schedule(static) reduction(+:tree_lh)  // line 946
for (ptn = 0; ptn < orig_nptn; ptn+=VCSIZE) {                    // line 949
    // ... compute partial_lh[ptn*block] for VCSIZE patterns ...
    // ... contract over rate categories and states ...
    _pattern_lh[ptn] = lh;                                       // line 984
    tree_lh += lh * ptn_freq[ptn];                               // line 988
}
// final reduction via OMP across threads → tree_lh                 line 1001
```

**Mode P modification:** replace `0` and `orig_nptn` with `ptn_start` and `ptn_end` (per-rank slice bounds). After the OMP reduction completes, `MPI_Allreduce` `tree_lh` across ranks.

### 3.2 The branch-length optimisation loop ([phylotree.cpp:2766–2773](/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylotree.cpp#L2766))

`PhyloTree::optimizeAllBranches` iterates over branches:

```cpp
for (int j = 0; j < nodes.size(); j++) {
    optimizeOneBranch(nodes[j], nodes2[j]);   // 1D Newton-Raphson per branch
    double new_tree_lh = computeLikelihoodFromBuffer();
}
```

Each branch's optimiser calls `computeLikelihoodDerv` (for gradient) and `computeLikelihoodBranch` (for tree_lh). Both go through the per-pattern kernel above. Under Mode P, each call needs the Allreduce immediately after the kernel returns.

### 3.3 The model parameter optimisation loop ([modelfactory.cpp:1624–1700](/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/model/modelfactory.cpp#L1624))

`ModelFactory::optimizeParameters` outer loop:

```cpp
for (i = 2; i < num_param_iterations; i++) {
    new_lh = tree->optimizeAllBranches(...);     // branches step (calls kernel ×O(nbranches))
    new_lh = optimizeParametersOnly(...);        // rates step (calls kernel ×O(rates))
    if (new_lh > cur_lh + logl_epsilon) cur_lh = new_lh;
    else break;
}
```

Each call to `optimizeAllBranches` and `optimizeParametersOnly` invokes the kernel many times internally. The Allreduce burden lives inside the kernel layer (3.1), not here — this loop body needs no MPI changes.

### 3.4 The dispatch entry ([phylotesting.cpp:1925–2175](/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/main/phylotesting.cpp#L1925))

`CandidateModel::evaluate` creates a fresh `IQTree`, calls `initializeModel`, then `getModelFactory()->optimizeParameters`. This is where Mode P **sets the per-rank pattern slice** on the IQTree object before optimisation begins:

```cpp
// PROPOSED: after iqtree->initializeModel() (line ~1986)
if (params.mode_p_enabled && MPIHelper::getInstance().getNumProcesses() > 1) {
    size_t nptn  = iqtree->aln->size();
    int    nrank = MPIHelper::getInstance().getNumProcesses();
    int    rank  = MPIHelper::getInstance().getProcessID();
    iqtree->ptn_start = (nptn / nrank) * rank;
    iqtree->ptn_end   = (rank == nrank - 1) ? nptn : (nptn / nrank) * (rank + 1);
} else {
    iqtree->ptn_start = 0;
    iqtree->ptn_end   = iqtree->aln->size();
}
```

### 3.5 Mode F + Mode P interaction

Mode P **cannot coexist** with Mode F K_outer > 1 — if rank 0 is evaluating model A pattern-parallel with all 16 ranks, the other ranks can't simultaneously evaluate model B. The dispatcher must pick **one mode per model**:

- If a model is "heavy" (predicted cost > threshold τ): use Mode P (all ranks cooperate on this single model)
- Otherwise: use Mode F (each rank owns its share of light models, K_outer = K_mem)

This gating lives in phylotesting.cpp at the model dequeue point. Suggested threshold: `predicted_cost > avg_cost × 8` (mark the top ~3 models as "heavy" via FCA's existing per-model cost predictor at lines 3927–3958).

## 4. New data structures

### 4.1 `PhyloTree::ptn_start, ptn_end` (additions to phylotree.h)

```cpp
class PhyloTree : public MTree {
public:
    // ... existing fields ...

    // Mode P: per-rank pattern slice [ptn_start, ptn_end). When MPI is not active
    // or Mode P is disabled, ptn_start=0 and ptn_end=aln->size() (full coverage).
    size_t ptn_start = 0;
    size_t ptn_end   = 0;  // set by initializePtnPartition() after model init
};
```

### 4.2 `Params::mode_p_enabled, mode_p_min_cost` (additions to tools.h)

```cpp
/** Mode P (pattern-parallel intra-model MPI_Allreduce). 0 = disabled (default);
 *  1 = enabled for models above cost threshold; -1 = force enabled for all models. */
int mode_p_enabled = 0;

/** Mode P cost threshold: models with predicted_cost > avg_cost * this multiplier
 *  use Mode P. Default 8.0 catches the top ~3 heaviest models in AA TEST. */
double mode_p_min_cost_mult = 8.0;
```

### 4.3 CLI flags (additions to tools.cpp)

```cpp
if (strcmp(argv[cnt], "--mode-p") == 0) { params.mode_p_enabled = 1; continue; }
if (strcmp(argv[cnt], "--mode-p-all") == 0) { params.mode_p_enabled = -1; continue; }
if (strcmp(argv[cnt], "--no-mode-p") == 0) { params.mode_p_enabled = 0; continue; }
if (strcmp(argv[cnt], "--mode-p-min-cost-mult") == 0) {
    cnt++;
    params.mode_p_min_cost_mult = convert_double(argv[cnt]);
    continue;
}
```

## 5. Allreduce placement strategy

There are **two strategies** for where to place `MPI_Allreduce` inside the kernel; pick one and commit.

### Strategy A: Allreduce after each branch optimisation (low-cost, low-coverage)

Only Allreduce the `tree_lh` value emitted at the end of `optimizeOneBranch`. Each rank computes a partial sum over its pattern slice; Allreduce gives the full tree_lh. The per-branch gradient is still computed locally on each rank's slice — but since branch optimisation is 1D Newton-Raphson on log-likelihood, the per-rank slice gradient direction differs from the full-tree gradient direction. **This is INCORRECT** unless we also Allreduce the gradient.

### Strategy B: Allreduce after every kernel call (high-cost, exact)

Modify the kernel itself (`computeLikelihoodBranch`, `computeLikelihoodDerv`) to emit the per-rank partial sum and immediately Allreduce. Every gradient and likelihood evaluation sees the full-tree value. **Correct by construction**.

Cost: one Allreduce per kernel call. The kernel is called O(num_branches × num_iterations × num_param_iterations) ≈ 200 × 8 × 5 = 8000 times per model. At ~1 µs per MPI_Allreduce on Gadi InfiniBand HDR, that's 8 ms of MPI overhead per model — negligible against 300 s of model wall.

**Decision: Strategy B.** Exactness over micro-optimisation.

### Implementation skeleton (Strategy B)

```cpp
// computeLikelihoodBranchEigenSIMD — Mode P modification
template<...> void PhyloTree::computeLikelihoodBranchEigenSIMD(...) {
    double tree_lh_local = 0.0;
#pragma omp parallel for schedule(static) reduction(+:tree_lh_local)
    for (ptn = ptn_start; ptn < ptn_end; ptn+=VCSIZE) {
        // ... existing kernel body, accumulating into tree_lh_local ...
    }

#ifdef _IQTREE_MPI
    if (params && params->mode_p_enabled && MPIHelper::getInstance().getNumProcesses() > 1) {
        double tree_lh_global = 0.0;
        MPI_Allreduce(&tree_lh_local, &tree_lh_global, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        return tree_lh_global;
    }
#endif
    return tree_lh_local;
}
```

For `computeLikelihoodDerv` (which returns both `df` and `ddf` derivatives), the Allreduce takes a buffer of 3 doubles `[tree_lh, df, ddf]`.

## 6. Implementation phases (incremental delivery)

### Phase P.1 — Scaffolding (0.5 days)

- Add `Params::mode_p_enabled, mode_p_min_cost_mult` and CLI flags.
- Add `PhyloTree::ptn_start, ptn_end` member variables (default to 0, full coverage).
- Add a no-op `initializePtnPartition()` method on PhyloTree.
- Build with `-DIQTREE_MODE_P=ON` cmake flag; no behaviour change unless `--mode-p` is set.
- **Test:** existing FCA / ATMD runs unchanged (regression gate).

### Phase P.2 — Pattern partition wiring (0.5 days)

- Hook `initializePtnPartition()` after `iqtree->initializeModel()` in `evaluate()`.
- Set `ptn_start, ptn_end` based on MPIHelper rank/size.
- Verify with a debug `cout` line: each rank prints its `[ptn_start, ptn_end)` slice.
- **Test:** a single model run with `--mode-p` shows rank 0 owns `[0, 62500)`, rank 1 owns `[62500, 125000)`, ... rank 15 owns `[937500, 1000000)`.

### Phase P.ISO — Kernel ISO sandbox before P.3 (highest priority, 0.5–1 day)

- Create `/scratch/rc29/as1708/iqtree3-mode-p-iso/` as an isolated source/build/run tree, following the FCA `mf.iso` workflow.
- Build an inert P.1/P.2 baseline binary, then P.3/P.4/P.5 binaries in separate build dirs.
- Run single-model AA 100K np=1/np=2 gates before applying production kernel patches.
- Compare `.iqtree`, stdout, and rank logs for lnL, BIC, best model, `[Mode P]` partition coverage, and branch/NR traces.
- **Promotion rule:** no P.3+ patch is promoted to the main b4/Mode P tree until the ISO passes lnL/BIC parity.

### Phase P.3 — Allreduce in scalar kernel (1 day)

- Modify the **non-SIMD** `computeLikelihoodBranch` to use `ptn_start, ptn_end` and Allreduce `tree_lh`.
- Modify the **non-SIMD** `computeLikelihoodDerv` similarly (Allreduce `[tree_lh, df, ddf]`).
- Run on AA 100K with `--mode-p` and a slow non-SIMD build; compare lnL to FCA baseline. **Must match to 1e-6.**
- **Test:** a single heavy model (LG+F+I+G4) under Mode P; np=2 ranks; lnL identical to np=1.

### Phase P.4 — Allreduce in SIMD kernels (1 day)

- Same modification for `computeLikelihoodBranchEigenSIMD`, `computeLikelihoodDervEigenSIMD`, `computeLikelihoodBranchEigenMixSIMD` (mixture models), and any other variants instantiated by the template machinery.
- Each kernel variant requires the same change. Use a helper inline function `mode_p_allreduce(double&)` and `mode_p_allreduce3(double&, double&, double&)` to avoid copy-paste.
- **Test:** same correctness gate at AA 100K; ATMD-b3c lnL match.

### Phase P.5 — Dispatcher integration (0.5 days)

- In phylotesting.cpp `evaluateAll`, after FCA assignment but before the outer dispatch loop, compute the per-model `predicted_cost` (FCA cost predictor already exists).
- Mark models with `predicted_cost > avg_cost × mode_p_min_cost_mult` as `MF_MODE_P`.
- In the outer loop: if next model is `MF_MODE_P`, **suspend** Mode F (all ranks block), all ranks evaluate this one model cooperatively, then resume Mode F dispatch.
- This requires an MPI barrier and a coordination broadcast — see §7 below.

### Phase P.6 — Correctness validation (1 day)

- AA 100K np=2 with Mode P enabled: lnL within 1e-6 of FCA np=1.
- AA 100K np=4 with Mode P enabled on top 3 heaviest models: same.
- AA 100K np=16: same.
- AA 1M np=16: lnL within 1e-2 (looser tolerance for cumulative FP non-associativity over 8000 Allreduces × 224 models).
- Cross-check: with `--mode-p --no-fca`, just FCA disabled → all models go through Mode P → MF should be slower than FCA (Allreduce overhead dominates for light models) but lnL identical.

### Phase P.7 — Performance measurement (0.5 days)

- AA 1M np=16 with Mode P on top 3 heaviest models. Expected MF wall: ≤ 600 s (vs FCA 1,122 s).
- AA 1M np=16 with Mode P on top 8 heaviest. Expected MF wall: ≤ 500 s.

## 7. Dispatch coordination for mixed Mode F + Mode P

The tricky part: under Mode P, all ranks block on one model. Under Mode F, each rank dequeues independently. Switching modes mid-dispatch requires synchronisation.

**Proposed protocol** (in `evaluateAll`, replacing the do-while loop body around phylotesting.cpp:4226):

```cpp
do {
    model = getNextModelMode();   // returns (model_idx, mode), -1 if done
    if (model == -1) break;

    if (mode == MF_MODE_P) {
        // All ranks must reach this point together.
        MPI_Barrier(MPI_COMM_WORLD);
        // All ranks cooperatively evaluate this one model.
        tree_string = at(model).evaluate(...with mode_p=true...);
        // After evaluate(), all ranks have the same result.
    } else {
        // Mode F path: per-rank dispatch as today.
        tree_string = at(model).evaluate(...);
    }
} while (...);
```

`getNextModelMode()` is a new function that returns a `(model_idx, mode)` tuple, where `mode = MF_MODE_P` for heavy models (which must be popped in deterministic order on all ranks via FCA's existing ordered queue) and `mode = MF_MODE_F` for light models (which use the existing per-rank getNextModel state machine).

The key invariant: **all ranks dequeue Mode P models in the same order**. FCA's deterministic LPT ordering already provides this; we just need to label models as Mode P consistently across ranks (via the predicted_cost threshold which is deterministic).

## 8. What this does NOT touch

- **FCA Phase 0.5 / 0.6** (the existing rank-to-family assignment): unchanged. FCA assigns one family of models per rank; Mode F runs the light models in those families; Mode P pre-empts to handle the heavy outliers.
- **A.2 warm-start broadcast** (`mpi_warm_start`): unchanged. Warm-start runs after each model evaluation regardless of mode.
- **B.-1 nested OMP setup** (`omp_set_max_active_levels(2)`): retained for Mode F's K_outer > 1 path.
- **B.3 NUMA first-touch** (`initializeAllPartialLh` with `atmd_inner_threads`): retained for Mode F path; bypassed for Mode P since all-ranks-one-model has no nested team to pin.

## 9. Validation strategy summary

| Phase | Test | Pass criterion |
|---|---|---|
| P.1 | Existing FCA AA 100K np=4 | lnL & timing unchanged from baseline |
| P.ISO | AA 100K np=1/np=2 isolated kernel sandbox | lnL/BIC/model parity vs b4/FCA before P.3 promotion |
| P.3 | AA 100K np=2 `--mode-p-all` non-SIMD | lnL = FCA np=1 lnL ± 1e-6 |
| P.4 | AA 100K np=2 `--mode-p-all` SIMD | lnL = FCA np=1 lnL ± 1e-6 |
| P.5 | AA 100K np=4 mixed Mode F + Mode P (auto-selected) | lnL = FCA np=1 lnL ± 1e-6 |
| P.6 | AA 1M np=16 mixed | lnL = FCA np=16 lnL ± 1e-2 |
| P.7 | AA 1M np=16 mixed | MF wall ≤ 600 s (vs FCA 1,122 s) |

## 10. Effort summary

| Phase | Time | Cumulative | Risk |
|---|---|---|---|
| P.1 Scaffolding | 0.5 days | 0.5 | low |
| P.2 Pattern partition | 0.5 days | 1.0 | low |
| P.ISO Kernel sandbox | 0.5–1.0 days | 1.5–2.0 | **medium** (test harness correctness) |
| P.3 Non-SIMD/kernel Allreduce | 1.0 days | 2.5–3.0 | **medium** (correctness) |
| P.4 SIMD kernel Allreduce ×N variants | 1.0 days | 3.5–4.0 | medium (template bloat) |
| P.5 Mixed-mode dispatcher | 0.5 days | 4.0–4.5 | medium (deadlock if ordering breaks) |
| P.6 Correctness validation | 1.0 days | 5.0–5.5 | high (numerical reproducibility) |
| P.7 Performance measurement | 0.5 days | 5.5–6.0 | low |
| **Total** | **5.5–6.0 days** | | |

Dependencies on prior patches:
- A.2 mpi_warm_start broadcast (in place ✓)
- B.-1 MPI_Init_thread(FUNNELED) (in place ✓)
- B.3 nested OMP setup (in place ✓; will coexist with Mode P via the dispatcher gate)
- B.5 per_tree_bytes formula fix (in place ✓ but not strictly required — Mode P doesn't depend on K_mem)

## 11. Why this is worth doing despite the §15.9.14 conclusions

§15.9.14 showed Mode F cannot help AA workloads on Gadi SPR. But the §14.9 projection is **not** dead — it just relies on the Mode P layer that was never implemented. Specifically:

- AA 1M np=16 FCA baseline: MF wall = 1,122 s
- AA 1M np=16 with Mode P on LG+F+I+G4 alone: projected MF wall ≈ 822 s (332 s → 32 s = -300 s)
- AA 1M np=16 with Mode P on LG+F+I+G4 + LG+F+G4: projected MF wall ≈ 742 s (-380 s)
- AA 1M np=16 with Mode P on top 8 heavy + Mode F K=4 on the rest: projected ≈ 500–600 s

This is the **only remaining path** to beat the ~400 s MF wall target. The b3c K=1 dead-end and the b4 K=6 bandwidth-saturation test (pending) confirm Mode F alone hits a wall; Mode P is the orthogonal lever.

## 12. Open questions for implementation kick-off

1. **MPI_COMM_WORLD vs sub-communicator**: should Mode P Allreduce use the full communicator (all ranks block on heavy model), or a sub-communicator (some ranks continue light-model work in parallel)? The latter is more complex but better for utilisation if Mode F and Mode P run on disjoint rank subsets.
2. **FP non-associativity**: 8000 Allreduces per model × O(1e-15) rounding per Allreduce = O(1e-11) cumulative drift per model, well below the model selection lnL_epsilon = 0.1. Numerical safety is fine but worth measuring.
3. **Mixture model handling**: LG4M, LG4X have nmixtures=4 and a different kernel template (`computeLikelihoodBranchEigenMixSIMD`). Mode P partition is still valid (patterns are mixture-independent) but the kernel modification needs to be applied to the mixture variant too.
4. **Mode P + warm-start interaction**: when Mode P broadcasts the result of LG+F+I+G4 to all ranks, does the existing A.2 warm-start broadcast still fire? It should, but the ordering needs review (Mode P result is canonical on all ranks → A.2 broadcast is redundant but harmless).
5. **Determining "heavy" threshold**: §5 suggests `predicted_cost > avg × 8`. Should this be tunable? Empirically the top 3 models account for ~75% of MF wall under FCA; threshold 8 catches them but is conservative. A learned threshold (from a calibration run) might be more robust but adds complexity.

These are tractable engineering decisions, not blockers. The phase plan in §6 absorbs them into the implementation tasks.
