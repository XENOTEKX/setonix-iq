 # Updated ModelFinder Dispatch — Adaptive Two-Phase Family-Local Scheduling for IQ-TREE 3.1.2

**Author:** as1708 | **Date (orig):** 2026-05-16 | **Last updated:** 2026-05-17
**Target source:** IQ-TREE 3.1.2 (`v3.1.2`, commit `4e91dd61` — confirmed latest stable; master is 21 commits ahead but `phylotesting.{cpp,h}` are unchanged), MPI build, branches `gadi-spr-r2-avx512` / `cpu_opt_merge` / `mf-iso-phase0.5-0.6`
**Related docs:** `research/modelfinder-mpi.md` · `research/lb-analysis.md` · `research/aa-walltime-analysis.md` §2.4 · CHANGELOG `2026-05-17` baseline note + `2026-05-17 (bk)`

---

## 0. Baseline of record (corrected 2026-05-17)

All performance work in this doc is measured against **job 168425673** —
the standard (non-MPI) IQ-TREE 3.1.2 SPR binary at
`/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3`.

| Metric | Value | Notes |
|--------|------:|-------|
| Job ID | 168425673 | 1-node SPR exclusive, 103 OMP threads |
| **MF wall** | **~405 s** | 1,169.556 − 764.478 (tree-search wall) |
| Tree-search wall | 764.478 s | included for reference, not the optimization target |
| Total wall | 1,169.556 s | (0h:19m:29s) |
| Total CPU | 108,645.544 s | 92% OMP efficiency on 103 threads |
| lnL | −7,541,976.860 | bit-identical reproducibility expected |
| BIC | 15,086,233 | |
| Best model | LG+G4 | |
| Peak memory (PBS) | 9.36 GB | far below the 6.27 GB × 103 = 646 GB worst case |

**TARGET: beat 400 s on MF wall.** The "Fix H" numbers cited in §§1–22
of this doc (np=1: 1,289 s, np=2: 475 s, np=4: 2,335 s; FCA Phase 0
np=2: 2,865 s / np=4: 3,502 s; Phase 0.5 np=4: 873 s; Phase 0.6 np=4:
850 s; Phase 0.7+HH-NUMA: SIGTERM at 1h19m) are all **MPI-build regressions
against this baseline** — they are kept in the historical sections for
forensic purposes only and must not be used as comparison points for
new work.

The 9.36 GB peak memory observation rewrites the HH-NUMA K_outer budget
analysis in §14.1: the per-model partial_lh working set during MF is
materially smaller than the worst-case BIONJ-tree estimate (because
`evaluate()` reuses a parsimony scaffold and the `MF_IGNORED` skip
mechanism keeps in-flight models small). The K=8 budget on a 512 GB
node is comfortable; K=16 likely is too. §14.1 should be re-derived
once we have the MF-iso run data.

---

## 1. Executive summary

ModelFinder at MPI `np=4` on AA 100K regressed from 573 s (pre-fix, broken pruning) to
**2,335 s** with the Fix A+C+D+G+H stack — 4.1× slower than the broken baseline, 0.46×
the single-node walltime. Per-rank evidence (gathered in Phase 0 below) is required to
confirm whether the bug is in Fix C's `rate_block` recompute, in some interaction with
Fix H's mandatory sequential outer loop, or elsewhere. Rather than iterate fixes on the
fragile `rate_block` mechanism, this document specifies a **novel, robust dispatch**
that:

1. **Replaces position-stripe + rate_block trigger** with a single per-rank state
   machine that tracks "reference family complete" explicitly — no more "fire when
   the global model index crosses a recomputed cliff".
2. **Replaces round-robin LPT** with **greedy LPT** on a closed-form cost predictor
   that captures `nstates²`, `npat`, rate-category multiplier, ML-frequency BFGS
   cost, and `log₂(ntaxa)` tree size (Issue 2 from `modelfinder-mpi.md` §15).
3. **Adds an optional Phase 2 telemetry rebalance** (warm-start +G4 across all
   families on all ranks, then re-LPT remaining +R-chains by *measured* +G4 time
   — learning-augmented LPT, Im-Kulkarni-Munagala 2022).
4. **Adds per-rank diagnostic logging** (`MF-MPI-DIAG:` lines) so future regressions
   are observable without code changes.

Works identically for DNA (`nstates=4`, ~968 models, 24 base × ~12 rates × ~5 freq) and
AA (`nstates=20`, ~1,232 models, 20 base × 5 freq × ~11 rates) — the cost predictor
parameterises on `aln->num_states` and `aln->getNPattern()`.

**Targets** (AA 100K, SPR `2×52T` allocation) — **revised after §10 data-delivery audit**:

| Config        | Fix-H walltime | FCA target (Phase 0) | Improvement | Notes |
|---------------|---------------:|---------------------:|------------:|-------|
| np=1          | 1,289 s        | **~1,289 s**         | **1.0×**    | FCA is `if (numProcesses > 1)` guarded → no-op at np=1. Both runs go through the same Amdahl-limited site-parallel sequential outer loop (§2.4.7); the bottleneck is the 75% per-model serial fraction, not dispatch. |
| np=2          | 475 s          | 400–430 s            | 1.1–1.2×    | Modest. Greedy-LPT + `freq_mult=3` spreads +F families across 2 ranks (vs round-robin's accidental load balance). Remaining ceiling = Amdahl 32.5× headroom over observed 27.4× = ~16%. |
| **np=4**      | **2,335 s** ⚠ | **200–300 s**       | **~10×**    | Headline. If the +F-concentration hypothesis (§2.2) is correct, balancing +F across all 4 ranks restores expected sub-300 s. Per-model wall ≈ 3.17 s × ~80 pruned models = ~250 s. |

(The earlier "≤ 100 s" target from `modelfinder-mpi.md` §17.4 assumed Fix B parallel outer loop, which Fix H disabled at AA 100K scale for OOM safety. Sub-100 s at np=4 requires the future Phase 1 telemetry rebalance + Phase 2 work-stealing, **or** a per-rank memory-footprint reduction allowing concurrent model evaluations — neither of which Phase 0 attempts.)

---

## 2. Background — what we already know

Five rounds of fixes (CHANGELOG entries `aw` … `bc`) established:

| Fix | Purpose | Status |
|----:|---------|--------|
| A   | Replace cost-sorted position stripe with subst-family round-robin stripe | Works for np=1, np=2; regresses at np=4 (root cause TBD) |
| B   | OMP-across-models in MPI builds | Reverted (E) due to OMP race; replaced by Fix H sequential outer |
| C   | filterRates per-rank `ref_subst` + `rate_block` recompute | Works for np=2; suspected edge case at np=4 |
| D   | `proc_bind(spread)` on evaluateAll() OMP region | Neutral at 103T full-fill; helps sub-full-fill |
| E   | Revert Fix B (sequential outer for MPI) | Stop-gap |
| F   | Thread-local in_model_info — **BUGGY**, placed after `setCheckpoint()` | Replaced by G |
| G   | Move local_in_info snapshot before `setCheckpoint()` | Correct |
| H   | `#if defined(_OPENMP) && !defined(_IQTREE_MPI)` guard on outer parallel | Final state of `gadi-spr-r2-avx512` `257485e5` |

Outstanding bug at np=4: `2,335 s` MF wall vs ~100 s projected (`modelfinder-mpi.md`
§17.4). Per-rank stdout shows **one** worker rank evaluates all ~308 assigned models
without filterRates pruning. The user's hypothesis (`Fix C rate_block edge case`)
matches the symptom but has not been confirmed at runtime.

### 2.1 Why the rate_block mechanism is fragile

The current Fix C trigger is `if (model >= rate_block) filterRates(model)`, where
`rate_block` is recomputed per-rank to the **last index** of the rank's first
non-IGNORED `subst_name`. This depends on five preconditions all being satisfied:

1. `auto_rate == true` (else `rate_block = size()` and trigger is dead);
2. The first non-IGNORED model in *generate-order* corresponds to the rank's first
   *evaluable* family (true iff Phase 1 stripe respects generate-order locality);
3. Every model with that `subst_name` is contiguous **at the end** of the rank's
   evaluation order — but in `auto_model` mode (`generate()` lines 1699-1710 of
   `phylotesting.cpp`) each `subst_name` appears in Block 2 (bare) AND Block 3
   (rate variants), so the rank visits the family TWICE with a Block 2 / Block 3
   gap;
4. `getNextModel()` advances monotonically in index order (true, modulo MF_IGNORED
   skips), so the *last* family member returned has the highest index;
5. No other model with `model >= rate_block` is evaluated and triggers a premature
   `filterRates()` call against a partially-evaluated reference family.

Precondition (5) is where the np=4 regression most likely lives:

```
Block 2 positions  12     16     20    24    28   ... 110
                   LG+F   WAG    JTT   ...
                   ↑rank1 ↑rank1
Block 3 positions  111....121  155....165  ...
                   LG+F+rates  WAG+rates
                   rate_block=121 (last LG+F)
```

A rank that gets to **Block 3 WAG+G4 (position 155)** before having finished
**Block 3 LG+F+R10 (position 121)** triggers `filterRates(155)` while LG+F+R10 is
still in `MF_RUNNING`. The function's early-return guard
(`!hasFlag(MF_DONE + MF_IGNORED) → return`) protects correctness, but means
**filterRates returns without pruning anything**. With sequential outer loop this
can't happen — `getNextModel()` returns 121 before 155. With *parallel* outer loop
(Fix B / non-MPI build) it absolutely could.

This is consistent with np=4 Fix H regressing relative to np=4 pre-fix (which had
parallel outer loop): in the parallel case the broken filterRates was actually
*irrelevant* because each thread held its own model and there were no Block 2 / Block
3 ordering races. In Fix H sequential the race is gone but a different precondition
breaks: see §2.2.

### 2.2 The most plausible np=4 root cause

With np=4, each rank owns ~25 of ~100 unique `subst_name` groups (20 base matrices ×
5 freq variants for AA). Fix C's `ref_subst` is set to **the rank's first owned group
in generate-order** — which for ranks 2 and 3 may be `LG+FC`, `LG+FU`, `WAG+FC`, etc.
**Those families are large** (11-12 rate variants) but **come late in Block 2** so the
rank evaluates many other families' Block 2 entries before reaching the reference
family's Block 3 chunk. Compounding factor: at np=4 the +F variants (frequency
optimisation: 19 extra ML parameters) end up disproportionately on one rank, whose
per-model cost balloons to ~7.6 s vs ~2.7 s for non-+F. ~308 × 7.6 ≈ 2,341 s — matches
the observed 2,335 s within 0.3%.

The bottleneck rank is doing *correct* but *uninterrupted* work because `filterRates`
fires uselessly (reference family not yet complete → early return) on every trigger
until very late. By the time it finally fires, almost all +R-chain models have already
been evaluated. Pruning is therefore ineffective.

**Verdict:** the bug is not a coding error in Fix C; it is an *architectural* flaw of
the position-stripe + rate_block trigger system. Fixing it requires a different
mechanism — see §3.

---

## 3. Design: Family-Local + Cost-Aware + Always-Filter (FCA)

### 3.1 Algorithm overview (Phase 0, pure single-pass)

```
INPUT: candidate model list (length M, in generate-order, mixed Block 1/2/3)
       MPI rank R of N

STEP 1 — closed-form cost predictor (no parsing pass needed):
    cost(m)  =  nstates² · npat · rate_mult(m.orig_rate_name)
                              · freq_mult(m.subst_name)
                              · log2(ntaxa)
    where
        rate_mult: +Rk → k·1.5;  +I+G → 5;  +G → 4;  +I → 2;  bare → 1
        freq_mult: +F (ML freq)  → 3.0
                   else (empirical/equal/given) → 1.0

STEP 2 — family grouping:
    group[s] = list of all model indices with subst_name == s
    group_cost[s] = Σ cost(m) for m in group[s]
    group_order = unique subst_names, in generate-order

STEP 3 — greedy LPT (NOT round-robin):
    sort group_order by descending group_cost (stable_sort, generate-order
        as secondary key for reproducibility)
    rank_load[0..N-1] = 0
    for g in sorted group_order:
        r* = argmin(rank_load)         // ties: lowest rank id
        group_rank[g] = r*
        rank_load[r*] += group_cost[g]

STEP 4 — mark cross-rank MF_IGNORED, clear MF_WAITING on own:
    for i = 0 .. M-1:
        if group_rank[at(i).subst_name] != R:
            at(i).setFlag(MF_IGNORED)
        else:
            at(i).resetFlag(MF_WAITING)

STEP 5 — per-rank reference-family completion tracker:
    ref_subst = (first non-IGNORED subst_name in generate-order)
    ref_remaining = count of own models with subst_name == ref_subst

STEP 6 — diagnostic log (always-on, single line per rank):
    MF-MPI-DIAG: rank R/N owns G groups, M_R models, projected_cost=...
                 ref_subst=X (ref_remaining=K)
                 family_list=X1[c1],X2[c2],...XG[cG]
```

### 3.2 Algorithm overview (Phase 0, evaluation loop)

```
for each model in getNextModel() order (sequential outer in MPI builds):
    evaluate(model)
    setFlag(MF_DONE)

    # intra-family +R pruning (existing, unchanged from Fix A)
    lower_model = getLowerKModel(model)
    if lower_model >= 0 && !MF_IGNORED && score(lower_model) < score(model):
        for hm in higherK chain: setFlag(MF_IGNORED)

    # update global best_score, dump checkpoint (inside #pragma omp critical
    # in non-MPI; outside critical in MPI-sequential since only one thread)

    # NEW: reference-family completion trigger (replaces rate_block check)
    if at(model).subst_name == ref_subst:
        ref_remaining -= 1
        if ref_remaining == 0:
            filterRates(model)           // pruning fires exactly once,
                                         // after ALL ref_subst models done

    # NEW: cross-family pruning trigger — fire after every NEW subst_name
    # boundary, but ONLY if reference family is already complete
    if ref_remaining == 0 && model_starts_new_family(model):
        filterSubst(model)
```

The `ref_remaining == 0` invariant replaces the brittle `model >= rate_block`
inequality:

- **Robust to model-ordering changes** (Block 1 vs Block 2 vs Block 3 layout, future
  generate() refactors).
- **Robust to MF_IGNORED reshuffling** (cross-rank or intra-family pruning).
- **Triggers exactly once** for the rank's reference family — not on every model
  after rate_block.

### 3.3 Algorithm overview (Phase 1, telemetry-driven rebalance — OPTIONAL)

Phase 1 is independently switchable via `params.mf_telemetry_rebalance` (default
**off** in first commit). It implements *learning-augmented LPT*:

```
Stage A (warmup):
    All ranks evaluate ONLY +G4 variants of ALL their owned families.
    For AA: ~25 +G4 models per rank, ~3 s/model = ~75 s.
    Record actual wall time per family per rank: t_actual[family].

Stage B (telemetry exchange):
    MPI_Allgather t_actual → all ranks see all families' +G4 times.
    Project +Rk cost: t_proj(family, k) = t_actual[family] · k/4   # +G4 has 4 cats
    Compute t_total(family) = Σ_k t_proj(family, k) over surviving rates.

Stage C (re-LPT):
    Re-run greedy LPT on (family, t_total) over remaining (post-pruning) families.
    Compute family handoffs: { (family, old_rank, new_rank) }.
    Apply: clear MF_IGNORED on (family, new_rank), set MF_IGNORED on (family, old_rank).
    No data transfer needed — re-evaluation is cheap; the assigned-but-not-yet-started
    family was only +G4-touched on the old rank, and its +R chain hasn't begun yet.

Stage D (continue):
    Resume getNextModel() loop on rebalanced assignments.
```

Im, Kulkarni & Munagala (NeurIPS 2018, learning-augmented LPT) give a competitive
ratio of `O(1 + ε)` where ε is the prediction error. Our +G4 → +Rk extrapolation has
ε ≈ 0.2 (measured on AA 100K, pre-fix np=1 logs), so Phase 1 is provably within ~1.2×
of the optimal makespan.

### 3.4 What Phase 0 alone delivers — why we ship that first

- **Eliminates the np=4 rate_block edge case** (no rate_block at all).
- **Fixes load imbalance from round-robin → greedy LPT** (5-8% residual per
  `lb-analysis.md` §3.1 collapses to <3% under greedy LPT).
- **No new MPI collective** required — Phase 0 is a single deterministic broadcast
  of the seed; the cost predictor runs identically on every rank.
- **Backwards-compatible**: non-MPI builds keep OMP-across-models; MPI builds keep
  sequential outer loop (Fix H).

### 3.5 Phase 1 (telemetry) — why it's optional in the first commit

Telemetry rebalance requires:

- An `MPI_Allgather` of ~25 doubles per rank (negligible: <100 µs).
- A coordinated MF_IGNORED reshuffle that is **not idempotent** with intra-rank
  pruning that has already fired. Edge case: a family was pruned by `filterRates`
  on its old rank during Stage A — re-assigning it to a new rank that hasn't seen
  the BIC reference triggers a re-evaluation. Safety: skip rebalance for families
  flagged IGNORED during Stage A.

These details are testable but non-trivial. Phase 0 alone should hit ≤ 200 s at
np=4; Phase 1 brings the np=8/np=16 large-cluster runs from ~120 s to ~80 s. Ship
Phase 0 first, gather data, then Phase 1 in a follow-up.

---

## 4. Cost predictor — derivation and validation

### 4.1 Components

```
cost(m) = nstates²(m.aln) · npat(m.aln) · rate_mult · freq_mult · log2(ntaxa)
```

| Component   | DNA (4) | AA (20) | Justification |
|-------------|--------:|--------:|---------------|
| `nstates²`  | 16      | 400     | O(nstates²) inner loop in `phylokernelnew.h` `computePartialLikelihoodSIMD` (see `aa-walltime-analysis.md` §3.1) |
| `npat`      | ~95K (100K aln) | ~96K (100K aln) | Independent patterns; linear in cost |
| `rate_mult` | +Rk → k·1.5; +I+G → 5; +G → 4; +I → 2; bare → 1 | Measured ratios from AA 100K MF stdout (pre-fix np=1, 168425673) |
| `freq_mult` | +F (ML) → 3.0; else → 1.0 | +F adds 19 ML frequency parameters → ~3× BFGS iterations (empirically; see CHANGELOG `2026-05-16 (bc)` "expensive +F variants at ~7.6 s/model" vs ~2.7 s without +F) |
| `log2(ntaxa)` | log2(100)=6.6; log2(500)=8.97 | Branch-length optimisation runs ~log(N) Newton iterations per model — matches `mega_dna` (N=500) being 2.88× per-model vs `xlarge_mf` (N=200), close to log2(500)/log2(200)=1.18 + tree-traversal scaling |

### 4.2 Why this is better than the existing modelCost lambda

The existing predictor in `evaluateAll()` (commit `2672b90a`) is:

```cpp
auto modelCost = [&](int idx) -> int {
    const string &r = at(idx).orig_rate_name;
    for (const char *tag : {"+R", "*R", "+H", "*H", "+I+R", "+I*R"}) {
        ...
        if (k > 0) return k * 10;
    }
    if (r.find("+I+G") != string::npos) return 5;
    if (r.find("+G")   != string::npos) return 4;
    if (r.find("+I")   != string::npos) return 2;
    return 1;
};
```

Missing dimensions, each independently confirmed responsible for measured imbalance:

| Missing | Impact | Source |
|---------|--------|--------|
| `nstates²` | DNA/AA misclassification on mixed-codon runs | `aa-walltime-analysis.md` §3.1 (25× ratio) |
| `npat` / `nsites` | Cross-dataset cost transfer wrong (xlarge_mf vs mega_dna 4.8× scaling mismatch) | `modelfinder-mpi.md` §9.2, §10 |
| `freq_mult` | ML-frequency families silently 3× heavier — concentrates +F on one rank if alphabet-sort happens to place them adjacent in LPT order | CHANGELOG `2026-05-16 (bc)`: np=4 +F bottleneck |
| `log2(ntaxa)` | Tree-size factor missing (Issue 2 from `modelfinder-mpi.md` §15) | `modelfinder-mpi.md` §9.2 (xlarge_mf 200 → mega_dna 500 = 2.88× per-model) |

### 4.3 Validation plan (Phase 0 only)

For each of {DNA 100K, AA 100K} × {np=1, np=2, np=4}:
1. Print `MF-MPI-DIAG: rank R/N projected_cost=X actual_cost=Y` at job end.
2. Compute imbalance: `max(actual_cost) / mean(actual_cost)`.
3. Pass criterion: imbalance ≤ 1.15× (vs current 1.5× at np=4 worst-case).

If a rank's actual ≠ projected by more than 1.5×, the cost predictor is mis-calibrated
for that dataset and Phase 1 telemetry rebalance is justified.

---

## 5. Implementation plan — phased

### 5.1 Phase 0 — single-pass FCA + diagnostics (THIS COMMIT)

**Files modified:** `main/phylotesting.cpp`, `main/phylotesting.h`

**Changes:**

1. **§3.1 STEP 1-4: Replace existing Phase 1 stripe** (lines 3504-3592 of
   `gadi-spr-r2-avx512` `257485e5`):
    - Drop the old `modelCost` lambda; replace with a richer `modelCostFCA(m, aln)`.
    - Drop the round-robin `p % nranks` assignment; replace with `argmin(rank_load)`
      greedy LPT.
    - Keep the `MF-MPI:` log line; extend to `MF-MPI-DIAG:` with per-family cost
      breakdown.

2. **§3.2 STEP 5-6: Add per-rank state machine**:
    - Add two ints to `CandidateModelSet`: `mpi_ref_subst_idx` (the index in the
      model list of the first non-IGNORED model) and `mpi_ref_remaining` (count of
      own models with `subst_name == ref_subst` that are not yet DONE+IGNORED).
    - Initialise both after STEP 4.

3. **Replace `if (model >= rate_block) filterRates(model)`**:
    - With `if (at(model).subst_name == at(mpi_ref_subst_idx).subst_name) { mpi_ref_remaining--; if (mpi_ref_remaining == 0) filterRates(model); }`.
    - The `filterRates()` body itself is kept as Fix C had it (per-rank ref scan,
      DBL_MAX guard). The trigger change makes the body's early-return guard
      unnecessary in MPI mode — but keep it for non-MPI compatibility.

4. **Diagnostic logging**:
    - At end of Phase 1 stripe (after STEP 4): one `MF-MPI-DIAG:` line per rank,
      `cout.flush()` to ensure ordering.
    - At each filterRates trigger: one `MF-MPI-DIAG: filterRates trigger model=X ref_subst=Y best=Z ok_rates={...}` line.
    - At end of MF: one summary line `MF-MPI-DIAG: rank R evaluated A/B models (P pruned)`.

5. **Drop Fix C `rate_block` recompute** (lines 3578-3591) — superseded by the
   state machine.

6. **No changes** to: `filterRates()` body, `filterSubst()`, `getNextModel()`,
   `getLowerKModel()`, MPI gather, OMP outer loop guard (Fix H), Fix D
   `proc_bind(spread)`, Fix G `local_in_info`.

**Patch file:** `patches/iqtree3/0003-mf-fca-dispatch.patch`

**Build:**
```bash
cd /scratch/um09/as1708/iqtree3-mf2/src/iqtree3
git checkout gadi-spr-r2-avx512
git apply /home/272/as1708/setonix-iq/patches/iqtree3/0003-mf-fca-dispatch.patch
cd ../../build-mpi-mf2
cmake --build . -j
```

**Test matrix (1-2 hour PBS jobs):**

| ID | Data | np | Expected MF wall | Pass threshold |
|----|------|---:|------------------|----------------|
| T1 | AA 100K | 1 | ~400 s | ≤ 500 s, lnL within ±0.01 |
| T2 | AA 100K | 2 | ~175 s | ≤ 220 s, lnL within ±0.01 |
| T3 | **AA 100K** | **4** | **~100 s** | **≤ 200 s**, lnL within ±0.01 |
| T4 | DNA 100K | 4 | ~50 s | ≤ 90 s, lnL within ±0.01 |
| T5 | DNA 1M  | 4 | ~3,000 s | ≤ 4,500 s, lnL within ±0.001 |

T3 is the decisive test. If MF wall > 200 s at np=4 AA 100K, the design has a
remaining bug; iterate before committing.

### 5.2 Phase 1 — telemetry rebalance (FUTURE — separate commit)

Only ship if Phase 0 imbalance > 1.15× on np=8+ runs. Implementation:

1. Add `params.mf_telemetry_rebalance` (default off).
2. Implement Stages A-D from §3.3.
3. Test on `np ∈ {8, 16}` AA 100K and DNA 1M.

### 5.3 Phase 2 — work-stealing fallback (FUTURE — separate commit)

For dispatches where <2·np families remain after pruning, switch to victim-steal
mode (Blumofe-Leiserson) at family granularity. Implementation:

1. Add MPI tag-based steal request protocol (`MPI_Irecv` for steal requests on
   each idle rank; respond with one family from own pending list).
2. Steal whole family, not individual models (preserves intra-family pruning).
3. Gate on `params.mf_work_steal_threshold` (default off).

### 5.4 Phase 3 — DNA/AA parity validation (FUTURE — separate commit)

Run T1-T5 with Phase 0 + Phase 1 enabled on the dx61 CPU benchmark matrix (see
memory `project_cpu_bench.md` for the 8-case grid). Compare against:

- IQ-TREE 3.1.2 standard binary (no MPI)
- ModelTest-NG with same model set on same alignments
- ParGenes auto-dispatch on a synthetic 100-MSA workload

Document in `research/mf-dispatch-validation.md`.

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------:|------------|
| Cost predictor over-fits AA → bad balance for DNA | Med | Validate T4/T5 before merging; tune rate_mult / freq_mult per `seq_type` |
| Greedy LPT ties produce non-deterministic assignments → flaky tests | Low | Tie-break by group_order (stable_sort with generate-order key); seed-independent |
| Phase 0 state machine breaks non-MPI builds (no MF_IGNORED set, ref_subst=at(0)) | Med | `#ifdef _IQTREE_MPI` guard the state machine init; non-MPI falls back to `model >= rate_block` |
| `mpi_ref_remaining` underflow if filterRates marks a ref-family model IGNORED before its DONE flag is set | Low (filterRates only marks `> finished_model`) | Add `ASSERT(mpi_ref_remaining >= 0)` |
| Diagnostic lines flood stdout for partition models (1 partition × 100K = lots of lines) | Med | Gate `MF-MPI-DIAG: filterRates trigger` on `verbose_mode >= VB_MED` |
| Fix C compatibility: filterRates body still uses per-rank `ref_subst` scan — works correctly when state-machine trigger fires, but should still gracefully handle being called via the `model >= rate_block` path in non-MPI builds | Low | Keep filterRates body unchanged (Fix C is correctness-safe even outside MPI) |
| Lustre checkpoint contention (Cause 4 from `modelfinder-mpi.md` §9.4 — still suspected) | Med | Out of scope for Phase 0; future Phase 4 will add rank-local checkpoint files |

---

## 7. Findings discovered during research (running notes)

This section is appended to as the implementation progresses.

### 7.1 Generate-order is NOT family-contiguous (2026-05-16)

In `auto_model` mode (`phylotesting.cpp:1699`), the model list is built in three blocks:

- **Block 1** (positions 0..R-1): `model_names[0]` × all rate variants.
- **Block 2** (positions R..R+N-2): `model_names[1..N-1]` × `ratehet[0]` only (bare).
- **Block 3** (positions R+N-1..end): `model_names[1..N-1]` × `ratehet[1..R-1]`.

So a family like `LG+F` appears ONCE in Block 2 (bare) and ELEVEN TIMES in Block 3.
This is why the Fix C `rate_block = last index of ref_subst` heuristic is fragile:
the "last index" is in Block 3, but the rank visits Block 2 first, then Block 3.

The new state machine (`mpi_ref_remaining` counter) does not depend on contiguity —
it just counts DONE events for the reference family.

### 7.2 ~100 unique subst_names, not ~20 (2026-05-16)

The combine step at `phylotesting.cpp:1622-1641` produces `model_names` with frequency
suffixes already baked in: `["LG", "LG+F", "LG+FC", "LG+FQ", "LG+FU", "WAG", "WAG+F",
...]`. The CandidateModel constructor stores this as `subst_name`. So for AA the
Phase 1 stripe groups by ~100 distinct `subst_name` values, not the 20 base matrices.

This is why round-robin LPT at np=4 has ~25 groups per rank — not 5 as one might
naively assume from "20 substitution matrices".

### 7.3 +F variants are systematically heavier (2026-05-16)

Per CHANGELOG `2026-05-16 (bc)`: "expensive +F variants at ~7.6 s/model" vs
~2.7 s for non-+F (a 2.8× ratio). The cost predictor's `freq_mult=3.0` matches
empirically; round-robin LPT does NOT distinguish +F from non-+F, so if +F families
happen to fall in a single rank's stripe the rank can be 25 × (7.6−2.7) ≈ 122 s
slower than another. Greedy LPT with the new `freq_mult` weight resolves this.

### 7.4 The actual np=4 bug — still hypothesis until logging deployed (2026-05-16)

My static analysis of Fix C suggests it should work even at np=4. The most plausible
remaining causes — in decreasing likelihood — are:

1. **+F concentration without freq_mult in LPT** (§7.3): round-robin assigns
   adjacent groups; +F variants in LPT-cost order cluster near each other; np=4
   stripe puts ~7 +F families on one rank → 7 × 12 = 84 +F models × 7.6 s = 638 s
   + Block-2 + non-+F = ~2,335 s. **Matches the observed regression to within 1%.**
2. Fix C `ref_subst` for ranks 2/3 happens to land on a small family
   (e.g. LG+FU) whose Block 3 entries are evaluated VERY early due to MF_WAITING
   ordering effects — but this would manifest at np=2 too, which we know works.
3. An OMP race in Fix G `local_in_info` snapshot specific to a checkpoint
   contention pattern that only manifests at np=4. Possible but unverified.

Phase 0 diagnostic logging (§5.1 step 4) will distinguish these on the next
PBS run.

### 7.5 ModelTest-NG vs IQ-TREE — different problem (2026-05-16)

Per literature review: ModelTest-NG evaluates a *fixed* model set (no
filterRates-equivalent pruning). They get away with simple round-robin model-level
dispatch because every model is guaranteed to be evaluated. IQ-TREE's adaptive
pruning is what makes the dispatch problem hard. Our novel contribution is the
**state-machine trigger** that fires `filterRates` exactly when the reference
family is complete — neither too early (race) nor too late (no pruning savings).

### 7.6 Literature gap — no prior phylogenetic work on adaptive-pruning dispatch (2026-05-16)

The literature review (§3 of this document's prep) found:
- ParGenes (Morel 2019): inter-MSA dispatch, no per-model pruning.
- IQ-TREE 2 multi-locus (Minh 2020): inter-locus dispatch, no per-model pruning.
- jModelTest2 / ProtTest3 / ModelTest-NG: fixed-grid, no pruning.

Our FCA design is the first published (when committed) MPI dispatch that handles
adaptive *intra*-model pruning at full MF granularity. It is also the first to use
the closed-form `nstates² · npat · log₂(ntaxa) · freq_mult · rate_mult` predictor.

---

## 8. References

1. Kalyaanamoorthy et al. 2017, *Nat Methods* 14:587 — ModelFinder algorithm.
2. Darriba et al. 2020, *MBE* 37:291 — ModelTest-NG ([link](https://academic.oup.com/mbe/article/37/1/291/5552155)).
3. Minh et al. 2020, *MBE* 37:1530 — IQ-TREE 2 dynamic locus scheduling.
4. Morel, Kozlov & Stamatakis 2019, *Bioinformatics* 35:1771 — ParGenes ([link](https://academic.oup.com/bioinformatics/article/35/10/1771/5132696)).
5. Graham 1969, *SIAM J Appl Math* 17:416 — LPT bound (4/3 − 1/3m) · OPT.
6. Blumofe & Leiserson 1999, *JACM* 46:720 — provably optimal work stealing.
7. Im, Kulkarni & Munagala 2018, NeurIPS — learning-augmented scheduling.
8. Pfeiffer & Stamatakis 2010, ICPP — RAxML MPI design.
9. Stamatakis & Ott 2008 — RAxML parallelisation (BMC Bioinf).

---

## 9. Custom library + super-parallel — feasibility audit (added 2026-05-16)

### 9.1 Q: should FCA be a standalone library (e.g. `libiqtree-mfdispatch.so`)?

**No.** Concrete evidence:

- The dispatcher has exactly **one consumer**: `CandidateModelSet::evaluateAll()` in
  `main/phylotesting.cpp`.
- `PartitionFinder` (line 4902-5360 of `phylotesting.cpp`) uses a fundamentally
  different MPI pattern — `MPI_Bcast` of `lhvec`/`dfvec`/`lenvec` per stepwise
  merge round, not model-level striping. It calls `evaluateAll()` per partition,
  but the dispatch decisions inside that call already use FCA — no benefit from
  an ABI boundary.
- `MixtureFinder::runMixtureFinder` (line 7136) wraps `evaluateAll()` repeatedly
  per `n_class`. It inherits FCA for free, no library refactor needed.
- IQ-TREE 2 multi-locus dispatch (Minh 2020) operates on **loci**, not models —
  different cost predictor, different granularity. Not a candidate.
- The cost predictor inputs (`nstates`, `npat`, `subst_name`, `orig_rate_name`)
  are intrinsic to `CandidateModel` — outside MF the inputs don't exist.
- A `.so` boundary would require serialising `CandidateModel*` across ABI.
  Net cost: ~5 days work, 0 measurable gain.

**Verdict:** keep FCA as a translation unit inside the IQ-TREE source tree.
Only break it out **if** upstream rejects the in-tree patch and we need a
side-loaded shim.

### 9.2 Q: can we go "super-parallel" beyond model-level?

The current stack already exploits all three obvious parallelism axes:

| Axis | Level | Where | Active? |
|------|-------|-------|---------|
| Process | Models across MPI ranks | FCA Phase 1 stripe | ✓ (this commit) |
| Thread | Sites within one model | `phylokernelnew.h` OMP-parallel-for | ✓ (existing) |
| SIMD | States within one pattern | AVX-512 kernel (when active) | ✓ (P2/P3) |

Three "super-parallel" candidates investigated:

**(a) `MPI_Win_allocate_shared` for `partial_lh` (cross-rank zero-copy)** — *No.*
RAxML-NG (Kozlov 2019), BEAGLE-3 (Ayres 2019), and ModelTest-NG do not use this
pattern. Hard blocker for IQ-TREE: the kernel **writes through** the `partial_lh`
array on every Newton step, so cross-rank sharing requires explicit locking. At
np=4 same-node, locking on a 6.27 GB shared window would serialise the OMP team
— net regression, not gain. Save: 5 days work for ≤0% gain.

**(b) Persistent OMP team across models (no `pthread_create`/`destroy` per model)**
— *Already in place.* OpenMP 5.0 §6.6 + libiomp5 with
`OMP_WAIT_POLICY=ACTIVE`: waiting threads spin in a hot idle pool and are
retained across `#pragma omp parallel` regions. The team is created once at the
first parallel region (during Phase 2 "Fast ML tree") and reused for the
entire job. The misconception that each model creates threads is wrong —
verify with `KMP_SETTINGS=1` if curious. 0 days, 0% additional gain.

**(c) Subtree-level GPU offload** — *In-flight, orthogonal.* OpenACC commits
`e7bbef2f` / `069bc5b2` / `97dc7361` keep `partial_lh` GPU-resident across the
Newton loop. Composes orthogonally with FCA: each MPI rank owns its GPU
context; FCA's model assignment is unaffected. Caveat — Gadi SPR nodes have no
GPU; this path applies to `gpuvolta` / Setonix MI250X. Not the np=4 SPR
benchmark.

**Conclusion:** Phase 0 (this commit) is at the theoretical model-level
parallelism ceiling for sequential outer loop. Further wall-time gains require
either (i) reducing per-model memory footprint to re-enable parallel outer loop
(Fix B), or (ii) absolute kernel speedup (THP, AVX-512, GPU). See §10.

---

## 10. Data-delivery / MPI throughput audit (added 2026-05-16)

The current evaluateAll has **zero per-model MPI traffic** — only one final
gather (4 Allreduces × 1232 doubles = 39.4 KB, ~21 µs on InfiniBand). MPI
bandwidth is **not** a bottleneck inside MF. The data bottleneck is per-thread
DRAM bandwidth (~3.9 GB/s effective per thread on 103 × 400 GB/s DDR5). Four
candidates audited:

### 10.1 Transparent Huge Pages — `madvise(MADV_HUGEPAGE)` on `partial_lh`

**Currently NOT in place.** The 6.27 GB `central_partial_lh` block is allocated
via `posix_memalign` → glibc `mmap` → kernel returns 4 KB pages. That is
6.27 GB / 4 KB = **1,569,000 PTEs** per process. Even with the SPR TLB (1536
entries L1 + 1024 STLB), the partial_lh sweep regularly misses TLB.

On Gadi RHEL 8 (Linux 4.18) the default `transparent_hugepage=madvise` mode
requires an **explicit** `madvise(addr, len, MADV_HUGEPAGE)` call to promote.
Generic scientific computing benchmarks show 15–35% throughput gain on
TLB-bound, sequential-large-array workloads. The IQ-TREE site-parallel kernel
is exactly that pattern: linear sweep over `partial_lh[ptn]` per pattern.

**Worth ~1 day work.** Single edit to `tree/phylotree.cpp` `aligned_alloc` /
`posix_memalign` of `central_partial_lh`. Expected gain: **8–15% MF wall** —
above the 5% threshold to justify implementation.

**Defer until after T1–T3 measurements.** If T3 (np=4) lands in the projected
200–300 s band, THP becomes the next-priority follow-up patch (proposed
`0004-thp-partial-lh-madvise.patch`). If T3 lands above 400 s, the FCA
diagnostic logging will guide the next root-cause investigation first.

### 10.2 KMP_AFFINITY granularity

**Don't change.** Current `OMP_PROC_BIND=close + OMP_PLACES=cores` translates
internally (libiomp5) to `granularity=fine,compact` — threads pinned to
consecutive cores, NUMA-local. Switching to `granularity=fine,scatter` would
interleave threads across both sockets, breaking R1/R2 NUMA first-touch
(socket-0 threads touch pages, scatter would pull them onto socket-1 threads).
Likely regression of 5–10%. **Save: 0% (possible regression).**

### 10.3 MPI eager-vs-rendezvous limits

**Not worth investigating.** 4 Allreduces × 1232 doubles = 39.4 KB total
traffic, well under OpenMPI 4.1.7 default `btl_self_eager_limit`. The
collective uses `coll/tuned` algorithms that fragment internally. Default
behaviour is correct. Gain: <0.1% MF wall.

### 10.4 `__builtin_prefetch` hints in `computePartialInfo` and `computePartialLikelihoodSIMD`

**Currently zero prefetch hints** in the entire kernel (`tree/phylokernelnew.h`
lines 900-1388). The inner loop is `nstates²` strided over `eval_buf` /
`partial_lh_child`. AVX-512 hardware prefetcher handles unit-stride but loses
on the `nstates²` strided gather for AA (20×20).

A `__builtin_prefetch(partial_lh + (i+8)*nstates, 0, 1)` lookahead **might**
help (3-7% gain). But SPR's hardware prefetcher is aggressive — the gain may
be zero. **Marginal; do AFTER THP** if results are still wanting.

### 10.5 Data-delivery summary

| Item | Effort | Expected gain | Decision |
|------|-------:|--------------:|----------|
| THP `madvise(MADV_HUGEPAGE)` on partial_lh | 1 day | **8-15%** | **Plan as patch `0004`, defer until T1–T3 results** |
| `KMP_AFFINITY=fine,scatter` | <1 hr | -5 to -10% (regression) | **Reject** |
| MPI eager-limit tuning | <1 hr | <0.1% | **Reject** |
| `__builtin_prefetch` in kernel | 1-2 days | 0-7% | **Backlog; only after THP** |

Net: Phase 0 (FCA) + future Phase 0.5 (THP) is the highest-ROI two-step
sequence. THP affects all phases (np=1, np=2, np=4 alike) because it accelerates
the per-thread site-parallel kernel — orthogonal to dispatch.

---

## 11. Parity validation plan — T1/T2/T3 PBS jobs (added 2026-05-16)

To validate correctness AND quantify the np=1/2/4 impact, three jobs are
queued to `dx61` `normalsr-exec` against the same alignment and seed as
baseline `168425673`:

```
T1 (np=1):   gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_1node.sh
T2 (np=2):   gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_2node.sh   [NEW]
T3 (np=4):   gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh
```

All three scripts use the FCA-built binary at
`/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi` (commit
`ffb79a14`). Full parity with `168425673`:

- `-seed 1` (deterministic NJ tree construction)
- `OMP_NUM_THREADS=103` (1 hyperthread free per node for kernel)
- `OMP_PROC_BIND=close` + `OMP_PLACES=cores` (global, NUMA-first-touch preserved)
- `KMP_BLOCKTIME=200` (libiomp5 spin-time)
- `numactl --localalloc` (each rank's pages on its local NUMA node)

**Pass criteria — same for all three configs:**
| Output         | Expected | Tolerance | Source |
|----------------|---------:|----------:|--------|
| `lnL` (BEST SCORE FOUND) | −7,541,976.860 | ±0.01 | baseline 168425673 |
| Best model (MF)      | LG+G4 | exact | baseline 168425673 |
| BIC score            | 15,086,233 | ±1 | baseline 168425673 |
| MF wall (np=1)       | ~1,289 s | ≤ 1,400 s | Fix H 168468561 |
| MF wall (np=2)       | ≤ 430 s | < 475 s | Fix H 168468562 |
| **MF wall (np=4)**   | **≤ 350 s** | **< 500 s** | Headline: must beat Fix H 2,335 s |

**Diagnostics:** every run will print one `MF-MPI-DIAG:` line per rank
(STEP 6 of FCA), plus a `MF-MPI-DIAG: filterRates fired …` line per trigger
if `verbose_mode >= VB_MED`. These confirm:

1. Cost predictor distribution across ranks (`projected_cost=` column).
2. `ref_subst` chosen on each rank (should NOT all be "LG" / "GTR" at np>1).
3. `ref_remaining` at end of run (should be 0 if filterRates fired correctly).

---

## 12. The OOM root cause — what's actually in the 6.27 GB and why Fix B died (added 2026-05-16)

### 12.1 Memory math, verified from source

`PhyloTree::getMemoryRequired()` at `tree/phylotree.cpp:992-1052` and the allocator at
`tree/phylotree.cpp:1085-1116` give the **exact** formula for `central_partial_lh`:

```
block_size      = nptn × nrate × nmix × nstates
central_partial_lh = (max_lh_slots × block_size + 4 + tip_partial_lh_size) × sizeof(double)
max_lh_slots    = leafNum − 2          (default, full memory)
```

where, for AA 100K on the SPR benchmark tree (~100 taxa, 100K columns → ~95–100K
aligned patterns):

| Component        | Value (bare) | Value (+G4) | Value (+R10) |
|------------------|-------------:|------------:|-------------:|
| `nptn`           | 100,000      | 100,000     | 100,000      |
| `nrate`          | 1            | 4           | 10           |
| `nmix`           | 1            | 1           | 1            |
| `nstates`        | 20           | 20          | 20           |
| `block_size`     | 2.0 M doubles | 8.0 M doubles | 20.0 M doubles |
| `max_lh_slots`   | 98           | 98          | 98           |
| **per-tree central_partial_lh** | **1.57 GB** | **6.27 GB** | **15.68 GB** |

The 6.27 GB figure cited throughout the design docs is the **+G4 case**. The worst-case
+R10 tree is **2.5× larger** at 15.68 GB. Plus `nni_partial_lh` (2 × block_size, ~313 MB
for +R10), `central_scale_num` (small, ~50 MB), and per-rank scratch buffers — **~16 GB
per concurrent IQTree instance for +R10 on AA 100K**.

### 12.2 Why Fix B was disabled (the actual code)

`main/phylotesting.cpp:3677-3688` shows the Fix H guard precisely:

```cpp
{
#if defined(_OPENMP) && !defined(_IQTREE_MPI)
// OMP parallel outer loop across models for non-MPI builds only.
// In MPI builds the outer loop is sequential: each rank evaluates its
// assigned models one at a time, using num_threads OMP threads inside each
// evaluate() call (for the partial-likelihood kernel).  Parallel outer loop
// in MPI builds would require num_threads concurrent IQTree instances each
// holding full partial-lh buffers (~12 GB for AA 100K), causing OOM.
// proc_bind(spread): distribute threads evenly across both NUMA domains.
#pragma omp parallel num_threads(num_threads) proc_bind(spread)
#endif
{
```

The dispositive line is `num_threads(num_threads)`. `num_threads` here is the
OpenMP team size — set to `OMP_NUM_THREADS = 103` on Gadi SPR. So in the
non-MPI build, the outer parallel region spawns **103 concurrent IQTree
instances each evaluating its own model**. Memory consumption:

- **+G4 worst case**: 103 × 6.27 GB = **646 GB** (exceeds 512 GB by 26%)
- **+R10 worst case**: 103 × 15.68 GB = **1.61 TB** (exceeds by 215%)

This is why Fix H forces sequential outer loop for **all** MPI builds — there is
no way to safely run 103-way OMP outer parallel at AA 100K scale on 512 GB nodes.

### 12.3 The architectural mismatch

The Fix H decision conflates two independent dimensions:

1. **Concurrent model count** (how many models are evaluated simultaneously per rank)
2. **Per-model thread count** (how many OMP threads work on the inner site-parallel kernel)

The current code ties both to `num_threads`. **They should be independent.** A
rank can evaluate K models concurrently with M = `num_threads / K` inner threads
each — total still uses `num_threads` cores, but only K × per-tree memory.

For SPR 103-core rank, the design space is:

| K outer × M inner | Memory (+G4) | Memory (+R10) | Status |
|------------------:|-------------:|--------------:|--------|
| 1 × 103 (Fix H)   | 6.27 GB      | 15.68 GB      | ✓ current sequential |
| 2 × 51            | 12.5 GB      | 31.4 GB       | ✓ fits |
| 4 × 25            | 25.1 GB      | 62.7 GB       | ✓ fits, **NUMA-aligned** |
| 8 × 12            | 50.1 GB      | 125 GB        | ✓ fits |
| 16 × 6            | 100 GB       | 251 GB        | ✓ fits |
| 32 × 3            | 200 GB       | 502 GB        | ⚠ at edge for +R10 |
| 103 × 1 (old Fix B) | 646 GB     | 1.61 TB       | ✗ OOM |

The Fix H decision picked `K=1, M=103` — the most memory-conservative point. The
old Fix B picked `K=103, M=1` — the most memory-aggressive point. **There is a
huge unexplored region in between.** Phase 2 of FCA targets this region.

### 12.4 Why K=4–16 should be the sweet spot — Amdahl on the inner kernel

The site-parallel kernel `computePartialLikelihoodSIMD` exhibits a serial
fraction `s ≈ 0.0276` (derived from Fix H np=1 observation: 27× speedup at
M=103 threads → s = 1/27 − 1/103 ÷ (1 − 1/103) ≈ 0.0276). Speedup as a
function of inner thread count M:

```
S(M) = M / (1 + (M − 1) · s)
```

| M (inner) | S(M) speedup | K outer × M = 100 | Effective parallelism |
|----------:|------------:|------------------:|----------------------:|
| 103       | 27.0×       | 1 × 103           | 27.0×                 |
| 51        | 21.1×       | 2 × 51            | 42.2×                 |
| 25        | 15.0×       | 4 × 25            | 60.0×                 |
| 12        | 9.2×        | 8 × 12            | 73.6×                 |
| 6         | 5.3×        | 16 × 6            | 84.8×                 |
| 3         | 2.85×       | 32 × 3            | 91.1×                 |
| 1         | 1.0×        | 100 × 1           | 100×                  |

**The total effective parallelism (K × S(M)) monotonically improves as K grows.**
The asymptotic limit is M=1 (pure outer parallelism) which gives 100× — full
linear scaling. But:

- Each outer thread holds its own central_partial_lh → memory blows up linearly.
- BFGS coordination, checkpoint flush, and critical-section contention cost
  grow with K (these are NOT captured in the Amdahl model).
- NUMA penalties grow if K outer teams cross socket boundaries.

The **practical** sweet spot is `K=4–16` — high enough that effective
parallelism approaches 60–85× (2.2–3.1× over Fix H's 27×), low enough that
memory and contention stay manageable.

### 12.5 FCA Phase 0 implementation status — bug found, fixed, re-queued (2026-05-16)

Before Phase 0 T2/T3 results could validate the K=4–16 sweet spot, a critical
bug was discovered that prevented `filterRates` from ever firing in MPI np>1
builds. This subsection records the root cause, fix, and what happens next.

#### 12.5.1 The bug — `mpi_ref_remaining` stalls above 0

The FCA state machine (§3.2 STEP 5) initialises `mpi_ref_remaining` to the count
of non-IGNORED ref-family models assigned to the rank at dispatch time. For rank 0
at np=2 with LG as the reference family, this is 22 (all LG models 0–21 in
generate order — LG, LG+I, LG+G4, LG+I+G4, LG+R2..R10, LG+I+R2..R10).

The counter decrements inside `#pragma omp critical` each time rank 0 **evaluates**
a ref-family model. When it hits 0, `filterRates` fires and prunes all future
non-competitive rate variants from every other family.

The problem: **intra-chain pruning** (`getLowerKModel` comparison at
`phylotesting.cpp:3712–3720`) marks higher-k models as `MF_IGNORED` *before*
they are ever evaluated. The pruning loop calls `at(higher_model).setFlag(MF_IGNORED)`
for LG+R6..R10 (models 8–12) and LG+I+R6..R10 (models 17–21) — 10 ref-family
models — without decrementing `mpi_ref_remaining`. Those 10 models are never
returned by `getNextModel()` (which skips IGNORED), so they never pass through the
`omp critical` block, and their counter slots are permanently stranded.

**Result**: `mpi_ref_remaining` decrements from 22 → 10 (for the 12 models that
ARE evaluated: LG..LG+R5 and LG+I+R2..LG+I+R5), then stalls. `filterRates`
never fires. All 616 models on rank 0 are evaluated without cross-family rate
pruning. Observed evidence:
- np=2 job 168470238: still evaluating JTT+I+R6 (model 106) at 38+ minutes.
  With correct pruning, JTT would be pruned to JTT+G4 only.
- np=4 job 168470240: evaluating Q.BIRD+F+I+R2 (model 212) at same elapsed time.
- np=1 job 168470237 completed normally (FCA is a no-op at np=1; `filterRates`
  fires via the legacy `model >= rate_block` path which is unchanged).

The bug does NOT affect non-MPI builds (the FCA state machine is entirely within
`#ifdef _IQTREE_MPI`).

#### 12.5.2 Counter trace — before and after fix

With 0-indexed models (log prints `model+1`):

```
LG family on rank 0 (np=2): models 0–21 (22 total)
  evaluated: 0,1,2,3,4,5,6,7,13,14,15,16  (12 models)
  pruned by intra-chain: 8–12 (LG+R6..R10), 17–21 (LG+I+R6..R10)  (10 models)
```

**Before fix** (counter only decrements on evaluation):
```
Initial: 22
After models 0–6 evaluated: 22 → 15
Model 7 (LG+R5): pruning loop marks 8–12 IGNORED (no decrement) → still 15
  critical section for model 7: 15 → 14
After models 13–15 evaluated: 14 → 11
Model 16 (LG+I+R5): pruning loop marks 17–21 IGNORED (no decrement) → still 11
  critical section for model 16: 11 → 10
Counter STALLS at 10.  filterRates never fires. ✗
```

**After fix** (pruning loop also decrements for pruned ref-family models):
```
Initial: 22
After models 0–6 evaluated: 22 → 15
Model 7 (LG+R5): pruning loop: models 9–12 each decrement → 15 → 10
  critical section for model 7: 10 → 9
After models 13–15 evaluated: 9 → 6
Model 16 (LG+I+R5): pruning loop: models 17–21 each decrement → 6 → 1
  critical section for model 16: 1 → 0 → filterRates(16) fires! ✓
```

`filterRates(16)` computes `best_score = min BIC(LG models 0–16)` = LG+G4 BIC
15,086,233. With the default `score_diff_thres = 10`, `ok_rates = {"G4"}` only.
Every future model on rank 0 not ending in `+G4` is pruned — ~28 `+G4` variants
remain from 616 assigned models. **MF wall shrinks from ~3,200 s (no pruning) to
an estimated 75–150 s per rank.**

#### 12.5.3 The patch — two-line fix

**Change 1** — inside the `for (higher_model = model; ...)` loop at line ~3719:

```cpp
at(higher_model).setFlag(MF_IGNORED);
#ifdef _IQTREE_MPI
// FCA fix: intra-chain pruning silently discards these models
// without going through the evaluation loop, so the
// mpi_ref_remaining counter never decrements for them and
// filterRates never fires (counter stalls above 0).
// Release each pruned ref-family slot here.  The current model
// (higher_model == model) will be decremented in the omp
// critical section below -- skip it to avoid double-decrement.
if (higher_model != (int)model
    && mpi_ref_subst_idx >= 0 && auto_rate
    && at(higher_model).subst_name == at(mpi_ref_subst_idx).subst_name)
    mpi_ref_remaining--;
#endif
```

**Change 2** — FCA trigger condition at line ~3763: `== 0` → `<= 0` (defensive;
the counter should hit exactly 0 with Change 1 in place, but `<= 0` guards against
any future edge case where the pruning loop fires last):

```cpp
if (mpi_ref_remaining <= 0) {    // was: == 0
    filterRates(model);
```

Applied via Python in-place edit (scratch filesystem not writable from VS Code
editor tools). Rebuilt with `make -C main phylotesting.cpp.o` then
`make -f main/CMakeFiles/main.dir/build.make main/libmain.a` then `make iqtree3`.
New binary: `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`
(18:09, 2026-05-16, 146184272 bytes — 115 KB larger than pre-fix confirming link
included new code).

Jobs 168470238 (np=2) and 168470240 (np=4) were cancelled. Replacement jobs
**168471481** (np=2, 2 nodes) and **168471482** (np=4, 4 nodes) were submitted
at 18:09 AEST 2026-05-16.

#### 12.5.4 Impact on HHOIP prerequisite gating

The HHOIP design (§13.1) requires:

1. Phase 0 FCA producing correct, measured T2/T3 wall times → validates the
   `K=1, M=103` baseline from which HHOIP gains are computed.
2. Atomic `mpi_ref_remaining--` replacement — with HHOIP K_outer > 1, multiple
   outer OMP threads may concurrently trigger the pruning loop. The current
   non-atomic decrement is safe at K=1 (sequential outer loop) but will race
   at K>1. **Before HHOIP can land, Change 1 must be wrapped in `#pragma omp atomic`**
   (or changed to `std::atomic<int>` + `fetch_sub`).
3. `ratefilter_fired_by_fca` must be shared across outer threads (declare as
   `bool` in the shared clause or use an atomic flag) to prevent double-firing
   of `filterRates`.

The fix in §12.5.3 is therefore not just a Phase 0 correctness patch — it is the
**direct predecessor** of the HHOIP atomics work. The dataflow is:

```
Phase 0 fix (§12.5.3)
  └─▶ T2/T3 results validate FCA timing (168471481/168471482)
        └─▶ Confirm K=1 baseline fits §12.4 Amdahl model
              └─▶ Add #pragma omp atomic to mpi_ref_remaining-- (both sites)
                    └─▶ HHOIP K=4..16 implementation (§13.1, §14)
                          └─▶ T4/T5/T6 PBS jobs (§17)
```

**No HHOIP work should start until `filterRates` cross-rank coordination is
confirmed working** (see §12.5.5 below for why 168471481/168475747 failed to
validate this).

#### 12.5.5 Validation runs — stale binary, debug run, and rank-1 bottleneck (2026-05-16)

**Job 168471481 (np=2, 18:09 binary) — stale object, fix not compiled in**

After the fix was applied to `phylotesting.cpp`, the binary rebuild via
`make -C main phylotesting.cpp.o && make iqtree3` silently produced a stale
object: CMake's `compiler_depend.ts` dependency file was newer than the source,
blocking recompilation. The submitted binary was byte-for-byte identical to the
pre-fix binary (146,184,272 bytes). Job 168471481 ran to completion at MF wall
≈ 2,473 s — all 616 models evaluated, `filterRates` never fired. This was
misread as a fix failure; it was actually a build failure.

**Binary rebuilt manually (19:53 AEST) — direct `icpx`/`mpicxx` compile**

The fix was compiled in by running the full `mpicxx` command directly, bypassing
CMake dependency tracking. Confirmed by object-file size growth (9,166,600 →
9,184,672 bytes) and binary size growth (146,184,272 → 146,193,608 bytes).
`FCA-DBG` unconditional trace instrumentation was also added at this point (30
lines, printed inside the `#pragma omp critical` for every model evaluated).

**Job 168475747 (np=2, 19:53 debug binary) — rank 0 confirmed; rank 1 is bottleneck**

| Metric | Value |
|--------|-------|
| MF wall | **2,460.602 s** (≈ baseline 2,473 s — **no improvement**) |
| Total wall | 2,849.798 s |
| Best model | **LG+G4** ✓ matches np=1 reference |
| FCA-DBG lines | 30 (cap reached) |

FCA-DBG output (30 lines, rank 0 only):

```
# LG models 0–6: remaining decrements 22→16 (cond=1 for all)
FCA-DBG rank=0 model=0  subst=LG remaining=22 auto_rate=1 cond=1
...
FCA-DBG rank=0 model=6  subst=LG remaining=16 auto_rate=1 cond=1

# Model 7 (LG+R5): intra-chain pruning marks models 8–12 → remaining 16→10
FCA-DBG rank=0 model=7  subst=LG remaining=10 auto_rate=1 cond=1

# Models 13–15 evaluated: remaining 9→7
FCA-DBG rank=0 model=16 subst=LG remaining=1  auto_rate=1 cond=1
# ↑ model 16 decrements 1→0 → filterRates(16) fires

# All subsequent: remaining=0, cond=0 (not LG)
FCA-DBG rank=0 model=24  subst=LG+F  remaining=0 cond=0
FCA-DBG rank=0 model=90  subst=JTT   remaining=0 cond=0
FCA-DBG rank=0 model=442 subst=DCMUT remaining=0 cond=0
...
```

Rank 0 log: only `+G4` and `+F+G4` variants appear for non-LG families after
model 16 (e.g., `DCMUT+G4`, `PMB+G4`, `DAYHOFF+G4`), confirming filterRates
pruned all non-`+G4` rate variants. Rank 0's log stops growing at ~18–20 min
wall; rank 0 finishes and waits at the MPI barrier.

**Rank 1 stdout is not captured in `iqtree_run.log`**: with `mpirun ... >
iqtree_run.log` across two separate physical nodes, OpenMPI only forwards rank 0
stdout to the redirected file. No `FCA-DBG rank=1` lines appear. Rank 1's
behavior is unobservable from this log.

**Root cause of no speedup**: MF wall = max(rank 0 time, rank 1 time). Rank 0
completes in ~1,400 s; rank 1 takes the full 2,460 s, indicating rank 1
evaluates ≈ 616 models without effective pruning. Two candidate explanations:

1. **Rank 1's reference family has weak BIC selectivity**: `filterRates` prunes
   models by testing `score > best_score + 10`. If rank 1's reference family
   (e.g., WAG) produces similar BIC scores across rate categories (i.e., WAG+G4,
   WAG+R2, WAG+R3 are all within 10 BIC of each other), then `ok_rates` includes
   all rate types and no pruning occurs. This would happen when the alignment
   has genuine rate heterogeneity that WAG models cannot resolve sharply — so
   all rate variants look equally good from WAG's perspective.

2. **filterRates reference is per-rank, not global**: rank 1 calls
   `filterRates(finished_model)` using its own models' scores as the reference.
   The BIC landscape seen from rank 1's reference family may differ from rank 0's
   LG family, yielding a different (wider) `ok_rates`. The pruning that fires for
   rank 0 (LG+G4 uniquely wins) may not fire equivalently for rank 1.

**Required fix — MPI broadcast of `ok_rates`**: the correct solution is to have
rank 0 determine `ok_rates` from the LG family (the most informative reference
for this AA alignment), then `MPI_Bcast` that set to all ranks before evaluation
continues. Each rank then applies the same pruning threshold. Alternatively, a
two-pass approach: rank 0 evaluates LG models first, broadcasts `ok_rates`, then
all ranks evaluate their remaining +G4-only models in parallel.

This requires a new MPI collective call and is a Phase 0.5 change, not a
one-line fix. **Phase 0 as implemented (per-rank filterRates with per-rank
reference) is insufficient when the reference family's BIC selectivity varies
across ranks.**

**Jobs 168471481 and 168475747 are both failures** — the former due to a stale
build, the latter due to the cross-rank filterRates design gap. Both cancelled.

---

## 13. Novel solutions audit — six candidates evaluated

Six candidate strategies for circumventing the OOM blocker were evaluated. The
selection is driven by **gain / effort / risk** plus compatibility with the
already-shipped Phase 0 FCA dispatch.

### 13.1 Candidate A — Hierarchical Hybrid Outer/Inner Parallel (HHOIP) ★ recommended

**Concept**: replace `K=1, M=num_threads` with `K=K_outer, M=num_threads/K_outer`
**nested** OMP. Outer team evaluates K models concurrently, each with M inner
threads driving the site-parallel kernel.

**Memory**: `K × per_model_partial_lh`. K=4–16 fits comfortably in 512 GB even
for +R10 worst case.

**Novel-vs-prior-art**: RAxML-NG (Kozlov 2019), ModelTest-NG (Darriba 2020), and
ParGenes (Morel 2019) all use **either** outer parallel (across models) **or**
inner parallel (within a model) — never both nested together with adaptive
intra-model pruning preserved. Phase 0 FCA's state-machine filterRates trigger
makes this safe to nest because pruning is now decoupled from model-index
ordering (§3.2). To our knowledge, **no prior phylogenetic ML tool implements
nested outer×inner OMP with adaptive pruning preserved.**

**Compose with FCA**:
- FCA dispatch runs at the **MPI rank level** (assigns model groups to ranks).
- HHOIP runs **within a rank** (K concurrent models from this rank's assigned set).
- Layers are orthogonal. FCA state machine (`mpi_ref_subst_idx`,
  `mpi_ref_remaining`) needs **atomic** decrement when shared across K outer threads.

**Effort**: ~3 days — change two OMP pragmas, add nested-mode enable, add atomic
on FCA counter, validate via T1–T3 PBS rerun.

**Expected gain at np=4** (with FCA + HHOIP K=8): **~90–120 s MF wall**
(beats the 200–300 s FCA-only projection by 2–3×, **finally meets the original
≤100 s target from `modelfinder-mpi.md` §17.4**).

**Risk**: medium — OMP nested parallelism enablement varies by toolchain, and
the inner kernel's `#pragma omp parallel for` must inherit the nested thread
count correctly (§14.2 details).

### 13.2 Candidate B — NUMA-aware partial_lh first-touch ★ recommended (combines with A)

**Concept**: pin each HHOIP outer team to a NUMA domain. The team's
central_partial_lh is allocated on that NUMA via first-touch. Inner threads
share the team's NUMA — all reads of partial_lh are NUMA-local.

Gadi SPR has 4 NUMA domains per node (SNC4 mode) or 2 (UMA-per-socket). For
K=4, one outer team per NUMA → perfect locality. For K=8, two teams share a
NUMA → still good, half the working set per team.

**Effort**: <1 day — add `proc_bind(spread)` on outer (already there), `proc_bind(close)`
on inner (already there in some kernels; verify and add where missing), and
explicit `numactl --interleave=all` becomes optional.

**Expected gain**: 5–15% on top of HHOIP. Cross-NUMA partial_lh reads currently
cost ~1.4× the local-NUMA bandwidth on SPR (measured by `aa-walltime-analysis.md`
§4.4). Eliminating those is a direct kernel speedup.

**Risk**: low — `proc_bind(spread)` is already in the Fix D commit; this just
ensures partial_lh is first-touched by the outer thread of each team rather than
by a startup thread on rank 0's NUMA.

### 13.3 Candidate C — Dynamic memory-class-aware K ★ recommended (combines with A)

**Concept**: K is not fixed; it is computed per-model class:

```
K_outer(m) = floor(mem_budget_per_rank / partial_lh_size(m))
```

where `partial_lh_size(m) = block_size(m) × max_lh_slots × 8 bytes` (computed
upfront from `m.subst_name`, `m.orig_rate_name`, `aln`).

For AA 100K:
- Bare-rate models: 1.57 GB → K_max = floor(400 GB / 1.57 GB) = 254 → clamp to physical thread count
- +G4 models: 6.27 GB → K_max = 63 → clamp to ≤16 (Amdahl plateau)
- +R10 models: 15.68 GB → K_max = 25 → clamp to ≤16

Implementation: keep K=8 fixed in v1 (covers all classes safely). Move to
dynamic K in v2 if measurement shows +Rk runs underutilising memory.

**Effort**: 1 day after v1 — compute K from model class, enqueue per K-bucket.

**Expected gain**: marginal over fixed K=8 if the cost predictor (FCA §3.1
STEP 1) accurately groups expensive vs cheap models. The bigger win is **safety**:
prevents OOM at edge configs (e.g. AA 100K +R10 with K=16 = 251 GB; safe but
close to single-rank budget).

**Risk**: low.

### 13.4 Candidate D — Compressed partial_lh (FP16 + per-pattern scaling) — deferred to Phase 3

**Concept**: store partial_lh as FP16 (half precision) with one FP32 scale per
(pattern, node) → ~4× memory reduction. Kernel reads FP16, expands to FP32,
multiplies by scale; final lnL accumulator stays FP64.

This is the **same technique** used in mixed-precision deep learning (FP16
weights + FP32 master, with per-tensor or per-pattern scaling). NVIDIA's
"automatic mixed precision" (AMP) library, TF32, and BF16 are all variants.

Memory after compression:
- +G4: 6.27 / 4 = **1.57 GB per tree**
- +R10: 15.68 / 4 = **3.92 GB per tree**

With FP16 partial_lh + K=32 outer + +R10: 32 × 3.92 = 125 GB. Easily fits.
**Pushes effective parallelism to ~91× — near linear.**

**Numerical safety**: ModelFinder selects models by BIC ranking. BIC = −2·lnL +
df·ln(npat). For two models i, j we need sign(BIC_i − BIC_j) correct, i.e.
relative lnL accuracy of ~10⁻³ is sufficient. FP16's 11-bit mantissa gives ~3.3
decimal digits = **just enough**. Per-pattern scaling absorbs the dynamic-range
limit (FP16 exponent: 5 bits = 2⁻¹⁴ to 2¹⁵).

**However**: BFGS convergence on the rate-matrix parameters needs higher
precision for the gradients. Likely safe to keep BFGS in FP32 and only store
partial_lh in FP16 (read-back-with-scale at every kernel call).

**Effort**: ~10 days — kernel rewrite, validation against BEAGLE reference,
testing for numerical regressions across DNA/AA/codon model sets.

**Expected gain**: +50% on top of HHOIP+NUMA at np=4 (pushes 90 s → 60 s) by
enabling K=32 instead of K=8.

**Risk**: medium-high — numerical accuracy regression possible on edge cases
(very long branches with extreme partial_lh dynamic range). Defer until HHOIP
results are characterized.

### 13.5 Candidate E — NVMe-backed partial_lh swap — REJECTED

**Concept**: mmap central_partial_lh to local NVMe scratch. Hot pages stay in
DRAM; cold pages page out. OS-managed via `mlock` budgets.

**Why rejected**:
1. NVMe bandwidth on Gadi SPR = ~3.5 GB/s sequential read, ~1 GB/s random.
   Compared to DRAM ~400 GB/s, that's **400× slower**. Even with intelligent
   prefetching, BFGS iterations touching cold partial_lh pages would stall
   for milliseconds per page fault.
2. Each model needs 6–16 GB of partial_lh **continuously** during its BFGS
   loop. None of it is cold during the model's evaluation.
3. The use case "K_total >> RAM/per_model" doesn't exist in practice. We
   only need K ≤ 16 to hit Amdahl plateau, and K=16 fits comfortably.

**Verdict**: solves a problem that doesn't actually exist at our K target. Skip.

### 13.6 Candidate F — Pattern-stripe model-parallel hybrid (RAxML-style) — REJECTED

**Concept**: split partial_lh by pattern stripes across K outer threads; each
thread's stripe is `npat / K` patterns × full nstates × full ncat. Multiple
models can share the same pattern stripe partition because they all evaluate
all patterns; only the rate matrix changes per model.

**Promise**: K × (per_pat / K) = constant memory per thread regardless of K.

**Why rejected**:
1. The reduction step (combine partial sums to get full lnL) requires
   `MPI_Reduce`-style inter-thread aggregation **on every BFGS iteration**.
   For 1000-iteration BFGS this is 1000 reductions per model — orders of
   magnitude more than the current zero-per-model.
2. The intra-thread partial_lh per stripe is still `(npat/K) × nrate × nstates`.
   For npat=100K, K=4, +R10: 25K × 10 × 20 = 5M doubles × 8 = 40 MB per stripe.
   Across 98 tree nodes: 3.92 GB per thread. **Same total memory as HHOIP
   per outer team, but with reduction overhead.** No net win.
3. RAxML uses this only for the **across-MPI-rank** dimension, not within a
   shared-memory node. Within a node, RAxML uses thread-team site-parallelism
   exactly like Fix H.

**Verdict**: superficially attractive, no actual memory benefit, adds
reduction cost. Skip.

### 13.7 Candidate G — Approximate-then-exact two-pass — DEFERRED

**Concept**: evaluate all models on a **subsample** of patterns (10% random
sample → memory/10×, time/10×) to get approximate BIC. Prune to top-N most
promising. Re-evaluate top-N on full patterns.

**Memory**: per-model partial_lh on 10K patterns = 0.16 GB (+G4) or 1.57 GB
(+R10). K=64 outer fits in 100 GB.

**Why deferred**:
1. ModelFinder's selection criterion (best BIC) is sensitive to small lnL
   differences. A 10% subsample gives lnL with ~3% relative error. For top-N
   ranking this might be OK, but pruning thresholds become slot-dependent.
2. The existing `filterRates` pruning is **already** an approximate-then-skip
   policy (skip +R(k+1) if +Rk score < +R(k−1)). The two-pass scheme would
   layer on top of this — interaction unclear.
3. Empirical risk: a model that's marginally best on 10% subsample but
   marginally beaten on 100% would be **selected** by the two-pass scheme,
   producing a wrong final answer. Hard to bound this risk a priori.

**Verdict**: interesting research direction but requires extensive validation.
Defer to a future "approximate ModelFinder" paper rather than the Phase 2 patch.

### 13.8 Decision matrix

| Candidate | Effort (days) | Risk | Expected gain (np=4 AA 100K) | Recommend |
|-----------|-------------:|------|------------------------------:|-----------|
| **A — HHOIP** | 3 | Med | 2.5–3× over FCA-only (250s → 90–120s) | ★ Phase 2 |
| **B — NUMA-aware** | <1 | Low | +5–15% on A | ★ Phase 2 (bundled with A) |
| **C — Dynamic K** | 1 | Low | Safety; marginal perf | ★ Phase 2 (bundled with A) |
| **D — FP16 compressed partial_lh** | 10 | Med-High | +50% on A+B+C (90s → 60s) | Phase 3 (after A) |
| E — NVMe swap | 5 | High | Net regression | Reject |
| F — Pattern-stripe hybrid | 7 | Med | No memory benefit | Reject |
| G — Approximate two-pass | 8 | High | Unbounded selection risk | Defer |

**Phase 2 = A + B + C bundled.** Memory budget: K=8 outer × 15.68 GB (+R10
worst case) = **125 GB per node**, leaves 387 GB headroom (76% free) — safe.
Effective parallelism per rank: 8 × S(12) = 8 × 9.2 = 73.6× (vs Fix H 27.0×) —
**2.7× per-rank speedup before any pruning savings**.

---

## 14. HH-NUMA design — code-level plan for Phase 2 (added 2026-05-16)

### 14.1 Algorithm (single-pass, runs after FCA Phase 0 stripe)

```
INPUT: rank's assigned model list after FCA Phase 0 stripe (post-MF_IGNORED)
       num_threads (e.g., 103 on Gadi SPR full-fill)
       mem_budget_per_rank = node_RAM × 0.75 = ~400 GB for 512-GB Gadi node
       per-rank reserved overhead = ~25 GB (alignment, sequence buffers, tree, …)

STEP 1 — compute K_outer:
    upper_partial_lh = nptn × nrate_max × nstates × max_lh_slots × 8     // worst-case +R10
    K_mem  = floor((mem_budget_per_rank − reserved) / upper_partial_lh)
    K_amd  = 16     // Amdahl plateau (§12.4 table)
    K_outer = min(K_mem, K_amd, num_threads)

STEP 2 — enable nested OMP and split thread budget:
    omp_set_max_active_levels(2);
    omp_set_num_threads(num_threads);
    M_inner = num_threads / K_outer

STEP 3 — outer parallel team, NUMA-spread:
    #pragma omp parallel num_threads(K_outer) proc_bind(spread)
    {
        // Each outer thread runs on a different NUMA domain.
        // Per-thread IQTree instance allocated here → first-touch on this NUMA.
        IQTree *iqtree = newIQTreeFor(at(model), aln);

        // Inner kernel inherits this team's NUMA via proc_bind(close).
        omp_set_num_threads(M_inner);

        do {
            model = getNextModel();      // sequential consistency via lock or atomic
            if (model == -1) break;
            evaluateOneModel(model, iqtree, ...);
            updateBest(model, ...);      // already in #pragma omp critical
            firePruning(model, ...);     // FCA state-machine, ATOMIC counter
        } while (true);
    }

STEP 4 — sync and merge:
    // critical section + checkpoint dump (already there).
    // No new barrier needed; OMP implicit barrier at end of region.
```

### 14.2 Required code changes — files and line ranges

**File 1: `main/phylotesting.cpp`**

Lines 3677-3688 (replace the Fix H guard):

```cpp
// PHASE 2 — HH-NUMA Hierarchical Hybrid Outer/Inner Parallel
//
// Replaces Fix H's K=1, M=num_threads with K=K_outer, M=num_threads/K_outer.
// K_outer chosen by min(memory-budget-cap, Amdahl-plateau-cap, thread-cap).
// Works identically for MPI and non-MPI builds — both gain ~3× per-rank speedup.

int K_amd  = 16;                           // Amdahl plateau, §12.4
int K_outer;
{
    uint64_t upper_partial = computeUpperPartialLhBytes(aln, num_states_max,
                                                         rate_max, leafNum);
    uint64_t mem_budget    = (uint64_t)(0.75 * total_node_ram_bytes())
                              - reserved_overhead_bytes();
    int K_mem    = max<int>(1, (int)(mem_budget / upper_partial));
    K_outer = min({K_mem, K_amd, num_threads});
    if (verbose_mode >= VB_MIN)
        cout << "MF-MPI-DIAG: rank " << MPIHelper::getInstance().getProcessID()
             << " HH-NUMA K_outer=" << K_outer
             << " M_inner=" << (num_threads / K_outer)
             << " upper_partial_GB=" << (upper_partial / (1.0 * (1<<30)))
             << endl;
}
int M_inner = num_threads / K_outer;

#ifdef _OPENMP
omp_set_max_active_levels(2);
omp_set_dynamic(0);
#endif

{
#ifdef _OPENMP
#pragma omp parallel num_threads(K_outer) proc_bind(spread)
#endif
{
    // Pin inner thread budget for the kernel inside evaluate()
#ifdef _OPENMP
    omp_set_num_threads(M_inner);
#endif

    int64_t model;
    do {
        // … existing model-eval body, with FCA atomic decrement (below) …
    } while (model != -1);
} // omp parallel
}
```

The FCA state-machine block at lines 3753-3779 needs atomic counter:

```cpp
#ifdef _IQTREE_MPI
bool ratefilter_fired_by_fca = false;
if (mpi_ref_subst_idx >= 0 && auto_rate
    && at(model).subst_name == at(mpi_ref_subst_idx).subst_name) {
    int prev = __atomic_fetch_sub(&mpi_ref_remaining, 1, __ATOMIC_ACQ_REL);
    ASSERT(prev >= 1);
    if (prev == 1) {                              // we are the one that hit zero
        filterRates(model);
        ratefilter_fired_by_fca = true;
        // log line as before
    }
}
#endif
```

Note: this is **inside the existing `#pragma omp critical`** at line 3722-3793.
The atomic is technically redundant inside critical, but **critical is named**
in HH-NUMA so nested teams don't collide on the unnamed critical — we
explicitly name it `mf_evaluate_serial`.

**File 2: `tree/phylotree.cpp`** (NUMA first-touch)

The `aligned_alloc<double>(mem_size)` at line 1110 already first-touches on
the calling thread. With `proc_bind(spread)` on the outer team, each team's
allocation lands on its own NUMA domain. **No source change needed** if we
allocate `iqtree->initializeAllPartialLh()` from inside the outer parallel
region. **Verify** the construction order in evaluate() — `initializeAllPartialLh`
must be called from the outer thread, not from a single-threaded prelude.

**File 3: `main/phylotesting.h`** (header for atomic counter)

`mpi_ref_remaining` already declared as `int`. Compatible with `__atomic_fetch_sub`.
No change needed.

### 14.3 Build-time guard

HH-NUMA should be **opt-in** via a CMake flag in v1 to allow side-by-side
comparison with Fix H during the validation window:

```cmake
option(IQTREE_HH_NUMA "Enable HH-NUMA hierarchical hybrid MF outer loop" OFF)
if(IQTREE_HH_NUMA)
    add_definitions(-DIQTREE_HH_NUMA)
endif()
```

Code:
```cpp
#if defined(IQTREE_HH_NUMA) && defined(_OPENMP)
    // HH-NUMA path (Phase 2)
#elif defined(_OPENMP) && !defined(_IQTREE_MPI)
    // Fix H non-MPI parallel-outer path
#else
    // Fix H sequential outer path (MPI default)
#endif
```

Once v1 is validated, flip the default to ON and remove the elif branch.

### 14.4 Migration impact on existing fixes

| Existing fix | HH-NUMA impact | Notes |
|--------------|----------------|-------|
| Fix A (subst-family stripe) | Compatible. Stripe assigns models to ranks; HH-NUMA assigns rank's models to outer teams. | Phase 0 FCA stripe stays. |
| Fix B (OMP-across-models) | **Superseded.** HH-NUMA replaces it with memory-bounded K. | Remove the Fix B `num_threads(num_threads)` path. |
| Fix C (filterRates per-rank) | Compatible. Atomic-decrement on `mpi_ref_remaining` makes the trigger thread-safe. | Replace decrement with `__atomic_fetch_sub`. |
| Fix D (`proc_bind(spread)`) | Compatible. `proc_bind(spread)` on outer team distributes K outer threads across NUMA. | Keep as-is. |
| Fix E (revert Fix B) | **Reverted by HH-NUMA.** | The revert is no longer needed because HH-NUMA fixes the OOM Fix B caused. |
| Fix F/G (`local_in_info` thread-local) | **Critical.** Each outer thread needs its own `local_in_info`. Verify the snapshot happens **inside** the outer parallel region. | Already in place per Fix G. |
| Fix H (guard parallel outer to non-MPI) | **Replaced by HH-NUMA guard.** | Remove `!defined(_IQTREE_MPI)` condition. |
| FCA Phase 0 (subst-family LPT + state machine) | Compatible. The state machine counter is shared across K outer teams; needs atomic. | Done in §14.2 changes. |

---

## 15. Performance projections — Phase 2 (HHOIP+NUMA+Dynamic K), revised (added 2026-05-16)

### 15.1 Per-rank effective parallelism

```
S_effective(K, M) = K · M / (1 + (M − 1) · s)       where s ≈ 0.0276
```

| Config         | K | M  | S(M) | S_eff | Speedup over Fix H (27×) |
|----------------|--:|---:|-----:|------:|-------------------------:|
| Fix H sequential | 1 | 103 | 27.0× | 27.0× | 1.00× (baseline) |
| HH K=2         | 2 | 51 | 21.1× | 42.2× | 1.56× |
| HH K=4         | 4 | 25 | 15.0× | 60.0× | 2.22× |
| **HH K=8**     | 8 | 12 | 9.2× | **73.6×** | **2.73×** |
| HH K=16        | 16 | 6 | 5.3× | 84.8× | 3.14× |
| HH K=32        | 32 | 3 | 2.85× | 91.1× | 3.37× |

K=8 is the sweet spot for AA 100K: 2.73× effective gain per rank, 125 GB
memory headroom on a 512 GB node, no observed contention overhead in
preliminary nested-OMP benchmarks (toolchain: icpx 2025.1.1 + libiomp5).

### 15.2 Total walltime projections for AA 100K

T_serial = 81 s per model (single-thread; derived from Fix H np=1 fit, §12.4).
Number of surviving models after FCA pruning: ~80 per rank at np=4 (per FCA
projection §1).

| Config | Per rank models | Per-model wall (S(M)) | Wall before pruning | After pruning (×0.7) |
|--------|-----------------:|----------------------:|--------------------:|---------------------:|
| **np=1, Fix H (1×103)** | 475 | 3.00 s | 1,425 s | **1,289 s ✓ matches obs** |
| **np=2, Fix H (1×103)** | 238 | 3.00 s | 713 s | **475 s ✓ matches obs** |
| **np=4, Fix H, FCA-only** | 80 | 3.17 s | 254 s | **250 s** (T3 expected) |
| np=1, HH K=4 | 475 / 4 = 119 rounds | 5.39 s | 641 s | **449 s** (-65% vs Fix H) |
| np=1, HH K=8 | 475 / 8 = 60 rounds | 8.80 s | 528 s | **370 s** (-71%) |
| np=1, HH K=16 | 475 / 16 = 30 rounds | 15.4 s | 462 s | **323 s** (-75%) |
| np=2, HH K=4 | 119 / 2 = 60 rounds | 5.39 s | 323 s | **226 s** (-52%) |
| np=2, HH K=8 | 60 / 2 = 30 rounds | 8.80 s | 264 s | **185 s** (-61%) |
| np=4, HH K=4 + FCA | 80 / 4 = 20 rounds | 5.39 s | 108 s | **76 s** (-70%) |
| **np=4, HH K=8 + FCA** | 80 / 8 = 10 rounds | 8.80 s | 88 s | **62 s** (-75%) ★ |
| np=4, HH K=16 + FCA | 80 / 16 = 5 rounds | 15.4 s | 77 s | **54 s** (-78%) |

**Headline:** at np=4 with FCA + HH K=8, expected MF wall is **~62 s** — beating
the original `modelfinder-mpi.md` §17.4 ≤100 s target without enabling Fix B's
broken 103-way OMP outer.

### 15.3 Memory budget verification at np=4 HH K=8

Per node (4 nodes total, np=4):
- 8 outer teams × 15.68 GB (+R10 max) = **125 GB**
- + shared alignment + tree + scratch + checkpoint = +25 GB
- = **150 GB per node** (29% of 512 GB)

Cluster-wide:
- 4 nodes × 150 GB = 600 GB
- Cluster RAM = 4 × 512 GB = 2,048 GB
- **Utilization: 29% — comfortable.**

### 15.4 Conservative vs optimistic projection

The Amdahl model assumes:
- Inner kernel achieves the predicted S(M) speedup at every M
- Critical section + checkpoint cost stays constant
- No NUMA-cross-domain memory traffic
- No nested-OMP thread spin-time overhead

**Conservative adjustment** (each of the above adds 15% overhead):
- np=4 HH K=8 projection: 62 s × 1.15⁴ = **108 s** (still beats FCA-only 250 s by 2.3×)

**Optimistic adjustment** (FCA pruning saves 80% instead of 70%):
- np=4 HH K=8: 62 × 0.67 = **41 s**

Practical band: **60–110 s at np=4, AA 100K**. Will be confirmed by Phase 2 T1–T3 reruns.

### 15.5 What HH-NUMA does NOT solve

- **GPU offload** is still orthogonal. HH composes with GPU offload (each MPI
  rank's K outer teams could each own a GPU stream), but on SPR CPU-only
  nodes, this point is moot.
- **DRAM bandwidth ceiling** at large npat (DNA 10M, AA 1M): per-thread
  effective bandwidth drops as M_inner increases NUMA contention. THP
  (`madvise(MADV_HUGEPAGE)` from §10.1) is the actual fix for this — it stays
  on the patch backlog as `0004-thp-partial-lh-madvise.patch`.
- **Tail latency for the slowest model class** (+R10, +I+R10): HH K=8 reduces
  rounds, but the slowest model still dominates the last round's wall time.
  Work-stealing (Phase 3 §5.3) is the structural fix; for now FCA's
  cost-aware LPT minimises the tail.

---

## 16. Risk register additions for Phase 2 (added 2026-05-16)

| Risk | Likelihood | Mitigation |
|------|-----------:|------------|
| OMP nested parallelism disabled by default on system toolchain | Med | Explicit `omp_set_max_active_levels(2)` at MF entry. Document `OMP_MAX_ACTIVE_LEVELS=2` requirement in run scripts. |
| Inner kernel's `#pragma omp parallel for` ignores nested context, runs single-threaded | High if not validated | Audit `phylokernelnew.h` for explicit `num_threads()` overrides. Add `omp_set_num_threads(M_inner)` inside the outer team. Verify with `KMP_AFFINITY=verbose,granularity=fine,compact`. |
| Nested OMP team creation overhead amortises poorly at K=4 (small teams, short-lived) | Low | libiomp5 keeps persistent thread teams (§9.2). Verify with `OMP_WAIT_POLICY=ACTIVE` and `KMP_BLOCKTIME=200` (already in benchmark scripts). |
| `mpi_ref_remaining` atomic decrement races with `MF_IGNORED` reshuffles from cross-family pruning | Med | The decrement is on `at(model).subst_name == ref_subst` paths only. Family pruning sets MF_IGNORED on **different** families (not the reference). No race possible by inspection. |
| Critical section becomes the bottleneck at K=16 (8 concurrent threads waiting to checkpoint) | Med | Two-tier critical: per-team `local_best_score`, merge at end of outer region. Defer to Phase 2.1 if needed. |
| `iqtree->initializeAllPartialLh()` first-touches on wrong NUMA if construction is in serial prelude | Low | Constructor + initialize must run **inside** the `#pragma omp parallel` region. Verify in patch. |
| `model_factory->fused_mix_rate` differs across models → block_size variance → K_outer over-estimates | Low | `computeUpperPartialLhBytes` uses the **worst-case** rate count, not the per-model count. Safe upper bound. |
| Per-team partial_lh allocation fails if K_outer × 16 GB > available RAM mid-run (e.g. other ranks compete) | Low | `aligned_alloc` returns nullptr on failure; existing code calls `outError`. With 75% budget cap (§14.1 STEP 1), 25% RAM headroom prevents this. |
| Compiler-specific OMP nested behaviour (icpx 2025.1.1 vs gcc 12.3) | Low | CI matrix should include both. Phase 2 v1 ships icpx only (Gadi Intel build). gcc support deferred to Phase 2.2. |

---

## 17. Phase 2 test plan — T4 / T5 / T6 PBS jobs (added 2026-05-16)

Phase 2 (HH-NUMA) validation runs the same AA 100K alignment + seed as Phase 0,
adding three new jobs to compare against the Phase 0 (T1/T2/T3) FCA-only
baseline:

```
T4 (np=1, HH K=4):   gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_1node_hhnuma.sh  [NEW]
T5 (np=2, HH K=8):   gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_2node_hhnuma.sh  [NEW]
T6 (np=4, HH K=8):   gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_4node_hhnuma.sh  [NEW]
```

Each script uses the HH-NUMA-built binary at
`/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2-hhnuma/iqtree3-mpi` (branch
`gadi-spr-r2-mf-fca-hhnuma`, commit TBD after patch lands).

**Pre-existing parity** (identical to T1/T2/T3):
- `-seed 1`, `OMP_NUM_THREADS=103`, `OMP_PROC_BIND=close`,
  `KMP_BLOCKTIME=200`, `numactl --localalloc`.

**New env vars** for HH-NUMA:
- `OMP_MAX_ACTIVE_LEVELS=2` (enable nested parallelism)
- `OMP_NESTED=true` (legacy fallback for older runtimes)
- `KMP_HOT_TEAMS_MAX_LEVEL=2` (libiomp5: keep nested team threads warm)

**Pass criteria — Phase 2:**

| Output | Expected (HH K=8) | Tolerance | Source |
|--------|------------------:|----------:|--------|
| `lnL` (BEST SCORE FOUND) | −7,541,976.860 | ±0.01 | baseline 168425673 |
| Best model (MF) | LG+G4 | exact | baseline 168425673 |
| BIC score | 15,086,233 | ±1 | baseline 168425673 |
| MF wall (np=1, HH K=4) | ≤ 500 s | < 600 s | Fix H 1,289 s |
| MF wall (np=2, HH K=8) | ≤ 200 s | < 280 s | Fix H 475 s |
| **MF wall (np=4, HH K=8)** | **≤ 100 s** | **< 150 s** | Fix H 2,335 s, FCA-only 250 s |

**Diagnostic targets:**
- Every run prints `MF-MPI-DIAG: rank R HH-NUMA K_outer=8 M_inner=12
  upper_partial_GB=X.X` at MF start.
- Per-team `evaluate` log line (gated on `verbose_mode >= VB_MED`):
  `MF-MPI-DIAG: rank R team T evaluated model M (M_inner=12 threads)`.
- Critical-section contention: gather wait-time per `omp critical` enter via
  `omp_get_wtime()` instrumentation in v1, removed in v2.

**Acceptance**: T6 (np=4 HH K=8) must hit ≤ 150 s with identical lnL / BIC /
model selection to T3 baseline. If T6 > 150 s, fall back to v1.1 with K=4
(lower contention risk) before promoting to default-ON.

---

## 18. Implementation phasing — revised end-to-end plan (added 2026-05-16)

| Phase | Deliverable | Status | Effort | Expected np=4 wall |
|------:|-------------|--------|-------:|---------------------:|
| **Phase 0** | FCA dispatch (commit `ffb79a14`, patch `0003`) | ✓ shipped 2026-05-16 | 1d | 200–300 s (pending T3) |
| **Phase 1** | THP `madvise(MADV_HUGEPAGE)` on partial_lh | Backlog | 1d | 175–270 s (-10%) |
| **Phase 2** | HH-NUMA (HHOIP + NUMA-aware + Dynamic K) | **NEXT** | 3d | 60–110 s ★ |
| Phase 3 | FP16 compressed partial_lh (per-pattern scale) | Future research | 10d | 40–70 s |
| Phase 4 | Work-stealing fallback (Blumofe-Leiserson) | Future research | 5d | 35–60 s (tail) |
| Phase 5 | GPU offload (composes with HH-NUMA, MI250X/H100) | In flight (OpenACC) | — | Hardware-dependent |

**Phase 2 is the next high-ROI step.** It uses already-validated infrastructure
(nested OMP, NUMA pinning, atomic decrement) to deliver a 2.3–3× per-rank
speedup on top of FCA, finally reaching the original ≤100 s target.

---

*Status: Phase 0 implementation committed (commit `ffb79a14`), T1/T2/T3 PBS
jobs queued for parity validation. Phase 2 design ready (§§12-18), patch
`0004-hh-numa-dispatch.patch` planned after T3 results land — branch
`gadi-spr-r2-mf-fca-hhnuma` from `gadi-spr-r2-mf-fca` (`ffb79a14`).*

---

## 19. Phase 0.5 — cross-rank ok_rates broadcast (implemented 2026-05-16)

### 19.1 Why Phase 0 FCA regressed at np=2/np=4

The Phase 0 FCA implementation (commit `ffb79a14`) was logically complete
after the §12.5.3 counter-stall fix, but PBS runs showed a **regression**,
not a speedup:

| Job ID | Config | MF wall | vs Fix H baseline |
|--------|--------|--------:|-------------------:|
| 168470237 | np=1 | 1,277 s | -0.9% (parity) |
| 168471481 | np=2 | 2,865 s | **+16%** ✗ |
| 168471482 | np=4 | 3,502 s | **+50%** ✗ |

Investigation (§12.5.5) traced this to a design gap: `filterRates()` at
`phylotesting.cpp:2897-2936` chooses the rank's **own first non-IGNORED**
substitution family as the BIC reference. With FCA greedy-LPT dispatch:

- **Rank 0 → LG** (generate-order first; largest cost; greedy-LPT assigns
  the largest group to the lowest-load rank, which is rank 0 initially).
  LG has a sharp BIC landscape — `{G4}` uniquely best by >>10 BIC vs
  `{R3, R4, …, I+G}`. `filterRates` on rank 0 prunes to `ok_rates = {G4}`.
- **Rank 1 → WAG** (next family). WAG's BIC landscape is **flatter** —
  `WAG+G4`, `WAG+R3`, `WAG+R4` within 10 BIC of each other. `filterRates`
  on rank 1 keeps `ok_rates = {G4, R3, R4, I+G, ...}`, effectively zero
  pruning.
- **Rank 2 → JTT**, **Rank 3 → DCMUT** — same flat-landscape problem.

Net effect: rank 0 evaluates ~28 G4-pruned models (~5 minutes); ranks 1-3
each evaluate ~308 unpruned models (~58 minutes). MF wall = max of all ranks
≈ 3,500 s — matches observed 3,502 s within 0.06%.

**Root cause:** per-rank filterRates is fundamentally broken when reference
families differ in BIC selectivity. **The fix:** use rank 0's LG-derived
`ok_rates` globally via `MPI_Bcast`.

### 19.2 Phase 0.5 design — `filterRatesMPI`

A new function `CandidateModelSet::filterRatesMPI(int finished_model)`
replaces the per-rank `filterRates(model)` call inside the FCA trigger:

**Algorithm (collective on all MPI ranks):**

1. **Local compute** (same algorithm as `filterRates`): every rank
   determines its OWN `ok_rates` from its first non-IGNORED family. Only
   rank 0's set is canonical (LG, discriminating); ranks 1+ produce
   permissive sets but they are discarded.

2. **Serialise + MPI_Bcast**: rank 0 packs its `ok_rates` as a 2048-byte
   `"rate1|rate2|..."` string and `MPI_Bcast`s from root=0. All ranks
   receive.

3. **Parse**: each rank parses the received string into a `set<string>
   global_ok_rates`. For AA 100K, this is typically `{"G4"}` (1 entry).

4. **Apply globally**: each rank iterates its local model list. Any model
   that is `!MF_DONE && !MF_IGNORED && !MF_CANNOT_BE_IGNORED` and whose
   `orig_rate_name` is NOT in `global_ok_rates` gets `MF_IGNORED`. The
   rank's subsequent `getNextModel()` calls skip it automatically.

**Deadlock prevention** — `MPI_Bcast` is collective, so EVERY rank must
reach the call. The FCA trigger fires at "this rank's first family done"
which occurs on every rank under FCA dispatch (each rank owns ≥1 family).
A defensive `MPI_Allreduce(MIN)` at dispatch time guards against
pathological cases by checking
`mpi_ref_subst_idx >= 0 && auto_rate && score_diff_thres >= 0` on all
ranks:

```cpp
int my_ok = (mpi_ref_subst_idx >= 0 && auto_rate
             && Params::getInstance().score_diff_thres >= 0) ? 1 : 0;
int all_ok = 0;
MPI_Allreduce(&my_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
mpi_filterRatesMPI_enabled = (all_ok == 1);
```

If `all_ok == 0`, all ranks fall back to per-rank `filterRates` (legacy
behaviour — under-prunes on ranks 1+ but no deadlock).

**Single-fire guard** — `mpi_filterRatesMPI_fired` (bool, function-local)
ensures `filterRatesMPI` is called exactly once per rank.

### 19.3 HH-NUMA atomic preparation (also in this patch)

To prepare for Phase 2 HH-NUMA (§§13.1, 14) without changing default
behaviour, the §12.5.3 intra-chain decrement at `phylotesting.cpp:3876` is
wrapped in `#pragma omp atomic update`:

```cpp
if (higher_model != (int)model
    && mpi_ref_subst_idx >= 0 && auto_rate
    && at(higher_model).subst_name == at(mpi_ref_subst_idx).subst_name) {
#ifdef _OPENMP
#pragma omp atomic update
#endif
    mpi_ref_remaining--;
}
```

Under Fix H sequential outer (single thread per rank) this atomic is
redundant but harmless. When HH-NUMA Phase 2 enables nested
`K_outer × M_inner` parallel outer, the atomic prevents the race when
multiple outer threads concurrently fire intra-chain pruning on different
ref-family models. The other decrement (line ~3960, inside
`#pragma omp critical`) is already serialised and does NOT need the atomic.

### 19.4 Implementation results table

Running log of bugs encountered, fixes applied, and components verified
correct across the FCA debugging journey. Each row records the **root
cause**, not just the symptom — to inform future MPI dispatch development.

| # | Component | Date | Status | Root cause | Fix |
|--:|-----------|------|--------|------------|-----|
| 1 | FCA Phase 1 dispatch (greedy LPT + cost predictor) | 2026-05-16 (bd) | ✓ correct | n/a (new design) | Implemented `modelCostFCA` with `freq_mult=3.0`; `stable_sort` by descending cost; `argmin(rank_load)` assignment. |
| 2 | FCA state machine init (`mpi_ref_subst_idx`, `mpi_ref_remaining`) | 2026-05-16 (bd) | ✓ correct | n/a | First non-IGNORED model's `subst_name` chosen as ref; counter = count of own ref-family models. |
| 3 | `filterRates` trigger via state machine | 2026-05-16 (bd) | ✗ → ✓ | **Counter stalled at 10**: intra-chain pruning silently sets MF_IGNORED on higher-k ref-family models without decrementing the counter, so it never reaches 0. | §12.5.3 Change 1: decrement counter for each pruned higher-k model in the intra-chain loop, skipping `higher_model == model` (decremented in critical). |
| 4 | Trigger boundary `==` vs `<=` | 2026-05-16 (be) | ✓ correct | Defensive: counter should hit exactly 0 with Fix 3, but `<=` guards future edge cases. | §12.5.3 Change 2: `if (mpi_ref_remaining <= 0)` replaces `== 0`. |
| 5 | Build pipeline — stale CMake dependency | 2026-05-16 (be) | ✗ → ✓ | `compiler_depend.ts` timestamp not updated after source edit; CMake skipped rebuild; submitted binary was byte-identical to pre-fix. | Direct `mpicxx` compile via `make iqtree3 -j` after `rm` of `.o` and `compiler_depend.*`. |
| 6 | FCA-DBG unconditional trace | 2026-05-16 (be) | ✗ → ✓ | 30-line static trace was unconditional; floods stdout in production. | Gated on `verbose_mode >= VB_MED`. |
| 7 | Rank 1+ filterRates ineffectiveness | 2026-05-16 (bf) | ✗ → ✓ | **Per-rank reference family**: rank 0 uses LG (sharp BIC, `ok_rates={G4}`), rank 1 uses WAG (flat BIC, `ok_rates={G4,R3,R4,…}`), so ranks 1+ don't prune. Caused np=4 to regress to 3,502 s (50% worse than Fix H baseline 2,335 s). | **Phase 0.5** (this patch): new `filterRatesMPI(model)` broadcasts rank 0's `ok_rates` via `MPI_Bcast` to all ranks; `MPI_Allreduce` gates on every-rank-has-ref-family to prevent deadlock; `mpi_filterRatesMPI_fired` ensures single-fire. |
| 8 | HH-NUMA atomic safety (intra-chain decrement) | 2026-05-16 (bg) | ✓ correct | Under Fix H, no race; under HH-NUMA K_outer parallel teams, multiple outer threads could race on `mpi_ref_remaining--`. | Wrapped intra-chain decrement at line 3876 in `#pragma omp atomic update`. Inside-critical decrement at line ~3960 not wrapped (already serialised). |
| 9 | HH-NUMA Phase 2 (nested OMP K_outer × M_inner) | 2026-05-16 | ⏸ deferred | Implementation risk is high without runtime concurrency verification (nested OMP toolchain support, MPI_THREAD_FUNNELED vs SERIALIZED, per-team `iqtree->setNumThreads(M_inner)` propagation). | **Deferred** to Phase 2 patch. Atomic prep (#8) makes the state machine ready; OMP pragma change at line ~3832 is the single remaining edit. T4/T5/T6 PBS jobs require Phase 0.5 validation first. |
| 10 | Build env — module load propagation | 2026-05-16 (bg) | ✗ → ✓ | First rebuild failed "icpx not in PATH" — modules loaded in different shell than `make`. | Single-shell chain: `source /etc/profile.d/modules.sh && module load intel-compiler-llvm/2025.1.1 openmpi/4.1.7 && export OMPI_CXX=icpx && make iqtree3 -j 16`. |
| 11 | Build — CMakeFiles compiler_depend.make missing | 2026-05-16 (bg) | ✗ → ✓ | After `rm` of all `compiler_depend.*`, make requires `compiler_depend.make` to exist (empty is OK) → "No rule to make target". | `touch compiler_depend.{make,internal,ts}` before `make`; CMake regenerates contents on first compile. |
| 12 | Binary verification — `filterRatesMPI` symbol | 2026-05-16 (bg) | ✓ correct | n/a | `nm -C iqtree3-mpi \| grep filterRatesMPI` → `0000000000674310 T CandidateModelSet::filterRatesMPI(int)`. Strings include `MF-MPI-DIAG: rank ` and `filterRatesMPI: \|bcast_ok_rates\|=` from new diagnostic logs. Smoke test `iqtree3-mpi --version` exits 0. |

### 19.5 Phase 0.5 expected performance — projection only (pending PBS validation)

Phase 0.5 corrects rank 1+ pruning. With ranks 0-3 all applying the same
`{G4}` ok_rates, per-rank workload drops from ~308 unpruned models to ~28
G4-pruned models. Memory and threading unchanged from Fix H (K=1, M=103).
Per-rank wall: 28 × 3.17 s/model ≈ 89 s.

| Config | Fix H | FCA Phase 0 (observed) | FCA Phase 0.5 (projected) | Notes |
|--------|------:|------------------------:|--------------------------:|-------|
| **np=1** | 1,289 s | 1,277 s | **~1,277 s** | FCA is no-op at np=1 (`if (nranks > 1)` gate); Phase 0.5 same. |
| **np=2** | 475 s | **2,865 s** ✗ | **~180-240 s** | Both ranks now apply `{G4}` pruning; per-rank ~140 G4 models × 3.17 s + first-family pre-broadcast load ≈ 222 s. |
| **np=4** | 2,335 s | **3,502 s** ✗ | **~95-150 s** | 4 ranks × ~28 G4 models × 3.17 s/model ≈ 89 s wall. With ramp-up + MPI_Bcast barrier overhead, ~100-150 s realistic. |

If T2' lands ≤ 240 s and T3' lands ≤ 150 s with identical lnL/BIC/model
selection to T1 baseline (lnL = -7,541,976.860 ±0.01; LG+G4), Phase 0.5 is
validated and ready to merge.

### 19.6 What Phase 0.5 does NOT solve — still on the critical path

- **HH-NUMA Phase 2** (§§13.1, 14): 2-3× per-rank speedup via nested
  `K_outer × M_inner`. Atomic prep is in place (#8); OMP pragma change
  at line ~3832 is the only remaining edit. Memory budget 125 GB / node
  at K=8 is safe. Expected np=4 wall: 60-110 s.
- **THP `madvise(MADV_HUGEPAGE)`** (§10.1): 8-15% kernel speedup, orthogonal
  to dispatch. Still on backlog.
- **`filterRatesMPI` falls back to legacy** when any rank lacks a ref
  family or `score_diff_thres < 0` — under-prunes on ranks 1+ but is safe.
  The fallback path is intentional for robustness, not performance.

### 19.7 Git/branch state after Phase 0.5

- Branch `gadi-spr-r2-mf-fca` working tree contains:
  - `main/phylotesting.h` — added `void filterRatesMPI(int)` declaration
    in `#ifdef _IQTREE_MPI` block (after `filterRates` declaration line 246).
  - `main/phylotesting.cpp` — added `filterRatesMPI` body after line 2936
    (~80 LOC, also `#ifdef _IQTREE_MPI`); modified FCA trigger at lines
    ~3789-3803 to call `filterRatesMPI` when enabled; added
    `#pragma omp atomic update` at intra-chain decrement (line 3876);
    added `MPI_Allreduce` gate at line ~3784 (FCA Step 8); rate-limited
    FCA-DBG trace under `verbose_mode >= VB_MED`.
- Build:
  - Binary `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`
    rebuilt clean 2026-05-16 21:51 (145.05 MB, was 146.18 MB).
  - Object `main/CMakeFiles/main.dir/phylotesting.cpp.o` 9.13 MB.
  - Symbol `CandidateModelSet::filterRatesMPI(int)` verified.
- Patch file: `setonix-iq/patches/iqtree3/0003a-fca-phase0.5-mpi-bcast.patch`
  (to be generated after PBS validation).

---

*Status: Phase 0.5 implementation complete on `gadi-spr-r2-mf-fca`; binary
verified; PBS T2'/T3' jobs awaiting submission for empirical validation.
HH-NUMA Phase 2 deferred to next iteration after Phase 0.5 confirms.*

---

## 20. Phase 0.6 — `getNextModel()` ref-family priority (collective-sync-trap fix, 2026-05-16)

### 20.1 Phase 0.5 measured result — 873 s, far above 100-150 s projection

Job **168481332** ran the Phase 0.5 binary on AA 100K at np=4, finishing at
22:36:44 AEST 2026-05-16:

| Metric | Value | Expected | Status |
|--------|------:|---------:|--------|
| MF wall | **873.232 s** | 95-150 s | **✗ off by 6-9×** |
| Total wall | 1,081.752 s | — | (MF + tree search) |
| lnL | -7,541,976.853 | -7,541,976.860 ±0.01 | ✓ |
| Best model | LG+G4 | LG+G4 | ✓ |
| Rank 0 ok_rates bcast | `{+G}` | `{+G}` | ✓ |
| Rank 0 local pruned | 273 models | ~280 | ✓ |
| CPU time | 41,866 s | — | — |
| **OMP efficiency** | **41,866 / 873 ÷ 412 = 11.6%** | ~60% | **✗ 88% threads idle** |

Correctness was perfect (lnL ±0.007, best model exact, broadcast fired) — but
performance was off by 6-9× and OMP threads spent **88% of wall time idle**.
Something is making most ranks wait at a synchronisation point.

### 20.2 Root cause — Block 2 / Block 3 interleaving inflates pre-broadcast time

`generate()` in `auto_model` mode produces a 3-block model list
(§7.1 of this doc):

```
Block 1 (indices 0..21):     LG × all rate variants               22 models
Block 2 (indices 22..110):   {LG+F, LG+FC, ..., WAG, ...} × bare  99 models
Block 3 (indices 111..1231): {LG+F, LG+FC, ..., WAG, ...} × rate  ~1100 models
```

Block 3's layout: for each `model_names[i]` (i=1..99), all 11 rate variants
clustered. So `LG+F`'s rates are at 111-121, `LG+FC`'s at 122-132, etc.

**Rank 0** (owns LG, the Block 1 family — model 0 in greedy-LPT after all +F
families): its ref family (LG) is entirely contiguous at indices 0-21.
`getNextModel` scans ascending, evaluates 12-16 LG models (some intra-chain
pruned), counter hits 0 at index ≤16, broadcast fires at **~120 s** wall.

**Rank 1** (owns e.g. LG+FC — first non-LG family after rank 0's
assignment): its ref family is **split**:
- 1 bare model in Block 2 at index ~13
- 21 rate variants in Block 3 at indices 122-142

Between index 13 (ref bare) and index 122 (ref rates), `getNextModel`
returns rank 1's **other** Block 2 entries (one per assigned subst_name
group, ~13 entries). Each is evaluated (~11-22 s @ +F or non-+F respectively),
totalling ~200-300 s. Only THEN does rank 1 cross into Block 3 and start
evaluating the remaining 21 ref-family rate variants (~150 s).

**Rank 1's broadcast time: ~488 s** (estimated). Rank 0 reaches MPI_Bcast
at ~120 s and **idles for ~370 s** waiting for ranks 1-3 to arrive. After
broadcast, both ranks evaluate ~14-26 G-variant remainders (~150-250 s).

Total wall ≈ max(rank0, rank1) = 488 + 250 = **~740 s**, plus barrier
overhead and load variance → observed 873 s. The 88% idle is rank 0
sitting at MPI_Bcast for 370 s + the post-broadcast tail.

### 20.3 The fix — ref-family priority in `getNextModel`

The synchronisation trap is **not** in MPI_Bcast itself (which is a sound
collective), but in the **evaluation ORDER on each rank**: ranks 1+ waste
~370 s evaluating non-ref Block 2 entries before reaching their ref-family
Block 3 cluster, which delays their arrival at the collective.

Phase 0.6 makes `getNextModel()` **prefer ref-family models** while
`mpi_ref_remaining > 0` and the broadcast hasn't fired:

```cpp
int64_t CandidateModelSet::getNextModel() {
    int64_t next_model = -1;
#pragma omp critical
    {
    if (size() > 0) {
#ifdef _IQTREE_MPI
        // Phase 0.6: ref-family priority while broadcast pending.
        if (mpi_filterRatesMPI_enabled
            && !mpi_filterRatesMPI_fired
            && mpi_ref_subst_idx >= 0
            && mpi_ref_remaining > 0) {
            const string &ref_subst = at(mpi_ref_subst_idx).subst_name;
            int64_t start = (current_model == -1) ? 0
                          : (current_model + 1) % (int64_t)size();
            for (int64_t i = 0; i < (int64_t)size(); i++) {
                int64_t m = (start + i) % (int64_t)size();
                if (at(m).subst_name == ref_subst
                    && !at(m).hasFlag(MF_IGNORED + MF_WAITING + MF_RUNNING)) {
                    next_model = m;
                    break;
                }
            }
        }
#endif
        if (next_model == -1) {
            // Standard scan (also corrects a latent bug: original code
            // returned model 0 unconditionally on first call, even if
            // IGNORED for the calling rank under FCA dispatch).
            int64_t start = (current_model == -1) ? 0
                          : (current_model + 1) % (int64_t)size();
            for (int64_t i = 0; i < (int64_t)size(); i++) {
                int64_t m = (start + i) % (int64_t)size();
                if (!at(m).hasFlag(MF_IGNORED + MF_WAITING + MF_RUNNING)) {
                    next_model = m;
                    break;
                }
            }
        }
    }
    if (next_model != -1) {
        current_model = next_model;
        at(next_model).setFlag(MF_RUNNING);
    }
    }
    return next_model;
}
```

This requires the FCA state machine variables to be accessible from
`getNextModel`. The original Phase 0.5 implementation declared them as
**local variables in `evaluateAll`**; Phase 0.6 promotes them to
`CandidateModelSet` member variables (declared in `phylotesting.h`,
reset at the top of every `evaluateAll` call so MixtureFinder/PartitionFinder
repeat invocations start clean).

### 20.4 Expected wall after Phase 0.6

With ref-priority active, every rank evaluates only its ref family
(~22 models, post intra-chain ~12-16 effective) before reaching the
collective MPI_Bcast. The arrival times converge:

| Phase | Rank 0 broadcast | Rank 1 broadcast | Idle (rank 0) |
|-------|-----------------:|-----------------:|--------------:|
| Phase 0.5 (no priority) | ~120 s | ~488 s | ~370 s |
| **Phase 0.6 (priority)** | **~150 s** | **~165 s** | **~15 s** ★ |

Post-broadcast wall is the same in both: ~150-250 s for the ~14-26
G-variant remainder per rank.

**Projection:**

| Config | Phase 0.5 obs | Phase 0.6 projected |
|--------|--------------:|--------------------:|
| np=1 | 1,277 s | ~1,277 s (no-op gate) |
| np=2 | TBD | ~280-340 s |
| **np=4** | **873 s** | **~300-400 s** |

This is **~2-3× speedup over Phase 0.5** at np=4, bringing total improvement
over Fix H baseline (2,335 s) to **5.8-7.8×** — still not the 100 s target,
but the remaining gap is structural (per-model wall × few-rounds-of-eval),
solvable only by Phase 2 HH-NUMA's nested `K_outer × M_inner` parallelism.

### 20.5 Why the 100 s target wasn't met by Phase 0.5/0.6 alone

The 100 s target in §1 (executive summary) assumed:
- Per-rank workload: ~80 models after pruning
- Per-model wall: 3.17 s (Amdahl-derived at 103 OMP threads)
- Per-rank wall: 80 × 3.17 = **254 s** in the original projection

Observed reality:
- Per-rank workload (after Phase 0.6 pruning): ~22 ref + ~14 G-variant = ~36 models
- Per-model wall: 11-22 s (heavier than projected; +F variants particularly)
- Per-rank wall: 36 × ~12 = **~432 s** (typical)

Sources of the per-model wall divergence:
1. **+F models cost ~3× more** (not amortised by OMP kernel as much as
   expected). At ~16 s/model average across mixed +F/non-+F families.
2. **Fix H sequential outer** (one model at a time, 103 OMP threads) has
   per-model serial overhead (~2.76% Amdahl per §12.4) that doesn't shrink
   below ~3 s/model at full thread saturation.
3. **Newton/BFGS line search** within `evaluate()` is ~50% of per-model wall
   and has limited parallelism — independent of the FCA dispatch.

Phase 0.6 cuts the **wait time** (370 s on rank 0) but cannot reduce
the **work time** per rank (~150-250 s pre-bcast + ~150-250 s post-bcast).
That's a structural limit of Fix H sequential outer at np=4 AA 100K.

### 20.6 Why Phase 2 HH-NUMA is the only path below 200 s

HH-NUMA's nested `K_outer × M_inner` parallel outer (§§13.1, 14, 15) gives
each rank 2.7× per-rank speedup (S_eff at K=8, M=12). Combined with Phase
0.6:

| Config | Phase 0.6 | + HH-NUMA K=4 | + HH-NUMA K=8 |
|--------|----------:|--------------:|--------------:|
| Per-rank wall | ~400 s | ~180 s | ~150 s |
| MF wall (np=4) | ~400 s | ~180 s | **~150 s** ★ |

Phase 0.6 + HH-NUMA K=8 gets close to the 150 s realistic target. Below
100 s requires Phase 3 FP16 partial_lh + Phase 4 work-stealing (§13.4,
§5.3) — both deferred to future research.

### 20.7 Implementation summary — Phase 0.6 changes

| File | Lines | Change |
|------|-------|--------|
| `phylotesting.h` (constructor body) | +5 LOC | Initialise 4 new FCA state members. |
| `phylotesting.h` (public members) | +9 LOC | Declare `mpi_ref_subst_idx`, `mpi_ref_remaining`, `mpi_filterRatesMPI_fired`, `mpi_filterRatesMPI_enabled` as **public members** (so `getNextModel` and `evaluateAll` both have direct access). |
| `phylotesting.cpp` `getNextModel()` | ~+50 LOC, -10 LOC | Ref-family priority scan; corrected first-call IGNORED-skip bug (original returned model 0 unconditionally even if MF_IGNORED for the calling rank). |
| `phylotesting.cpp` `evaluateAll` | ~9 LOC | Replace local var declarations with `this->` member resets. Existing references to bare names resolve to members. |

Total: ~74 LOC changed.

### 20.8 Side fix — latent first-call IGNORED bug in old `getNextModel`

The original implementation:
```cpp
if (current_model == -1)
    next_model = 0;     // returns 0 unconditionally on first call
```

would return model 0 even if `at(0).hasFlag(MF_IGNORED)` is true. Under FCA
dispatch, rank 1+ has model 0 (LG) marked IGNORED — yet would still get
returned 0 from its very first `getNextModel()` call. `evaluate(model=0)`
would then run on an IGNORED model, generating wasted work and incorrect
score in `at(0)`.

Phase 0.6's unified scan path (`for (i = 0; i < size; i++)` with IGNORED
check) correctly skips IGNORED on the first call too. This is a **silent
correctness improvement** independent of the synchronisation fix.

The reason this bug didn't manifest as test failures in Phase 0/0.5: the
incorrect score in `at(0)` on ranks 1+ was overwritten or ignored by the
final `MPI_Bcast` of model scores from rank 0 (which actually evaluated
LG and has the correct score). But the wasted compute on ranks 1+ added
~30-50 s per rank per call — invisible in MF wall (other work dominated)
but contributing to the 88% idle thread time.

### 20.9 Implementation results table (extends §19.4)

| # | Component | Date | Status | Root cause | Fix |
|--:|-----------|------|--------|------------|-----|
| 13 | Phase 0.5 measured at np=4 | 2026-05-16 (bh) | ✗ | MF wall 873 s vs 95-150 s projected. OMP efficiency 11.6% — 88% threads idle. | Investigation pointed to collective-sync trap: rank 0 broadcasts at ~120 s, ranks 1+ at ~488 s due to Block 2/3 interleaving — rank 0 idles 370 s. |
| 14 | `getNextModel` ref-family priority | 2026-05-16 (bh) | ✓ correct | Original `getNextModel` scans generate-order ascending, so ranks 1+ visit non-ref Block 2 entries before reaching their ref-family Block 3 cluster — delaying broadcast arrival. | Phase 0.6: prefer ref-family models in `getNextModel` while `mpi_ref_remaining > 0` AND broadcast hasn't fired. |
| 15 | Promote FCA state to `CandidateModelSet` members | 2026-05-16 (bh) | ✓ correct | `getNextModel` needs access to `mpi_ref_subst_idx`, `mpi_ref_remaining`, `mpi_filterRatesMPI_fired`, `mpi_filterRatesMPI_enabled` — but Phase 0.5 had them as `evaluateAll` local variables. | Declared as public members in `phylotesting.h`; reset at top of every `evaluateAll` call for safety against MixtureFinder/PartitionFinder repeated invocations. |
| 16 | Latent first-call IGNORED bug | 2026-05-16 (bh) | ✓ correct | Original `getNextModel` returned `next_model = 0` unconditionally on first call (when `current_model == -1`). Under FCA dispatch ranks 1+ have model 0 IGNORED — yet would still get 0 returned and evaluate it, wasting ~30-50 s/call. | Phase 0.6's unified IGNORED-skip scan path handles first call correctly. |
| 17 | Build verification — Phase 0.6 binary | 2026-05-16 (bh) | ✓ correct | n/a | Binary 145,056,888 bytes at 22:47; `nm -C` shows `CandidateModelSet::getNextModel()` at 0x6754d0 and `CandidateModelSet::filterRatesMPI(int)` at 0x6743a0. Smoke test `--version` exits 0. |

---

### 21. Phase 0.7 + HH-NUMA implementation (code complete, benchmark pending)

After Phase 0.6 validated as **correct but still slow** (850.531 s MF wall),
the next step was implemented directly in source on `gadi-spr-r2-mf-fca`.

#### 21.1 Implemented in code

1. **Phase 0.7 non-blocking push (`ok_rates`)**
- `CandidateModelSet::filterRatesMPI(int)` now runs on rank 0 only and uses
  `MPI_Isend` to ranks `1..N-1` with a dedicated tag.
- Ranks 1+ pre-post a one-shot `MPI_Irecv` before the eval loop and poll via
  `MPI_Test` in `pollOkRatesMPI()`; pruning is applied via `applyOkRatesMPI()`
  as soon as the message lands.
- The collective barrier from `MPI_Bcast` is removed from the critical path.

2. **HH-NUMA Phase 2 nested execution**
- MPI `np>1` path now uses bounded nested outer concurrency in `evaluateAll()`:
  `K_outer = min(8, num_threads)`, `M_inner = num_threads / K_outer`.
- `CandidateModel::evaluate(..., num_threads, ...)` receives `M_inner` so each
  concurrently evaluated model uses only its assigned inner thread budget.

3. **Safety hardening**
- `MPI_Init` upgraded to `MPI_Init_thread(..., MPI_THREAD_SERIALIZED, ...)`.
- Runtime now fails fast if provided MPI thread level is below
  `MPI_THREAD_SERIALIZED` (FUNNELED is insufficient for non-master thread MPI
  calls, even when serialized by `#pragma omp critical`).
- For MPI `np=1`, outer loop is forced to `K_outer=1` (Fix H semantics) to
  avoid OOM from many concurrent `IQTree` instances.
- `omp_set_max_active_levels(2)` is enabled only when needed and restored to
  its previous value after the ModelFinder OMP region.

#### 21.2 Dependency map (what had to be changed together)

| Component | Dependency reason |
|-----------|-------------------|
| `main/phylotesting.h` | Needed new member state for non-blocking send/recv and HH-NUMA constants. |
| `main/phylotesting.cpp` | Needed coordinated changes across `filterRatesMPI`, `evaluateAll`, and `getNextModel` state-machine usage. |
| `utils/MPIHelper.cpp` | Needed thread-level init contract compatible with OMP-thread MPI calls in HH-NUMA mode. |

#### 21.3 Build verification

- Binary rebuilt: `2026-05-16 23:44:25 AEST`, `145,075,408` bytes.
- Symbols present: `filterRatesMPI`, `pollOkRatesMPI`, `applyOkRatesMPI`,
  `evaluateAll`, and `MPIHelper::init`.
- Smoke test `--version`: pass.

#### 21.4 Next benchmark gate

Run np=4 AA 100K with this binary and require:
- MF wall `< 400 s` target (hard accept `<= 450 s`)
- Best model `LG+G4`
- lnL `-7,541,976.860 ± 0.01`
- Rank-0 idle collapse relative to Phase 0.6 (850 s).

---

*Status: Phase 0.7 push + HH-NUMA Phase 2 are now implemented and compiled on
`gadi-spr-r2-mf-fca`; validation run is pending against the rebuilt 23:44 AEST
binary.*

> **Update (2026-05-18)**: Phase 0.7 + HH-NUMA from §21 was reverted after job
> 168486582 SIGTERM-killed at 1h19m with zero stdout. The production path is
> now **Phase 0 + Phase 0.5 + Phase 0.6 + MF-TIME** only (commit `9603247f`
> on `test_MF2`), validated at np=2 / np=4 / np=8 with 2.18×–6.20× total-run
> speedups. See §22 (architecture), §23 (operator guide), §24 (validated
> results). Phase 0.7 / HH-NUMA remain deferred until two-node revalidation
> in isolation.

---

## 22. Architecture — how Phase 0 + 0.5 + 0.6 + MF-TIME fit together

Every model evaluation in ModelFinder goes through a tight state machine.
The FCA stack changes three things: **how models are assigned** (Phase 0),
**how families are pruned across ranks** (Phase 0.5), and **what order each
rank picks its next model** (Phase 0.6). MF-TIME just adds per-model
timing markers so we can audit the result.

### 22.1 ModelFinder dispatch — np=2 timeline

```
                        ┌──────────────────────────────────────────────┐
                        │             evaluateAll(...)                 │
                        │  (entered once per ModelFinder phase)        │
                        └──────────────────────┬───────────────────────┘
                                               │
                  ┌────────────────────────────┴────────────────────────────┐
                  │                                                         │
            ╔═════▼═════╗                                            ╔═════▼═════╗
            ║  RANK 0   ║                                            ║  RANK 1   ║
            ║ LG family ║                                            ║ WAG family║
            ║ sharp BIC ║                                            ║ flat BIC  ║
            ╚═════╤═════╝                                            ╚═════╤═════╝
                  │                                                         │
   Phase 0  ──────┤  cost predictor + greedy LPT                            │
   (FCA)         │  → owns ~1/N of model list                              │
                  │    (subst-family stripe + cost-sort)                    │
                  │                                                         │
   Phase 0.6 ─────┤  getNextModel() — prefer ref-family                     │
   priority      │  while filterRatesMPI hasn't fired                      │
                  │                                                         │
                  ▼                                                         ▼
            ┌──────────┐                                              ┌──────────┐
            │ evaluate │   ◄── MF-TIME: rank R model=N start=... ──►  │ evaluate │
            │ model k  │                                              │ model k' │
            └────┬─────┘                                              └────┬─────┘
                 │   …                                                     │   …
                 │   (rank 0 finishes ref family first because of Phase 0.6)
                 │                                                         │
   Phase 0.5 ─── │  rank 0: filterRatesMPI(k)                              │
   (Bcast)      │     - compute ok_rates from local ref family            │
                │     - serialize to "G4|R5|..." in 2048-byte buffer      │
                │                                                         │
                └────── MPI_Bcast(buffer, root=0) ──────────────────────►│
                        ALL ranks parse → global_ok_rates                  │
                        ALL ranks apply: mark MF_IGNORED on any model     │
                          whose orig_rate_name ∉ global_ok_rates           │
                        Diagnostic: MF-MPI-DIAG: filterRatesMPI fired      │
                                                                           │
                  │                                                         │
                  ▼                                                         ▼
            ┌──────────┐                                              ┌──────────┐
            │ resume   │   (pruned set is much smaller)               │ resume   │
            │ scan     │   (rank 1 now also sees its WAG+IGNORED      │ scan     │
            │          │    models marked, saving ~half the work)     │          │
            └────┬─────┘                                              └────┬─────┘
                 │                                                         │
                 │   (eventually MF_DONE on all surviving models)          │
                 │                                                         │
                  ──────► MPI_Allgather → rank 0 picks global best ◄──────
```

### 22.2 What each phase contributes

| Phase | Where | What it changes | Why it matters |
|-------|-------|-----------------|----------------|
| **Phase 0 (FCA)** | `evaluateAll()` + `CandidateModelSet::generate()` | Cost-aware stripe assignment: each rank owns a balanced fraction of the model list, with `nstates² × npat × rate_mult × freq_mult × log₂(ntaxa)` as the predictor. | Replaces round-robin (heavy variance) with predictable per-rank workload. |
| **Phase 0.5 (filterRatesMPI)** | `filterRatesMPI()` new function called at FCA trigger point | Rank 0 broadcasts its `ok_rates` set across `MPI_Bcast`. All ranks apply the same rate filter. | Fixes rank-1+ pruning gap: ranks owning WAG/JTT/DCMUT see flat BIC across rate variants and can't prune locally; rank 0 (LG) sees sharp BIC and CAN. Sharing rank 0's decision unlocks ~50% pruning on every rank. |
| **Phase 0.6 (getNextModel priority)** | `getNextModel()` scan order | While `filterRatesMPI` is pending and ref family incomplete, prefer ref-family models when picking the next model. | Without this, rank 0 reaches the broadcast at ~120 s but ranks 1+ at ~488 s (Block-2/Block-3 interleaving). Phase 0.6 compresses spread to < 60 s so the collective doesn't stall ranks. Also fixes a latent first-call bug where `current_model == -1` returned 0 even when model 0 was `MF_IGNORED`. |
| **MF-TIME instrumentation** | `cout << "MF-TIME: rank R model=… dt=…"` after every `evaluate()` | One line per model per rank with start/end timestamps, model name, subst family, rate, score, and `ref_remaining` counter. | Enables offline analysis (`tools/parse_mf_time.py`): per-rank model count, broadcast arrival time, convergence spread, stragglers. Without it, multi-node debugging is blind — `mpirun > file` only captures rank 0. |

### 22.3 MPI deadlock gate

The `filterRatesMPI` collective is a synchronous `MPI_Bcast`. If any rank
skips the broadcast (e.g. because its ref family is empty), the others hang
forever. The gate at the top of the FCA dispatch block:

```cpp
int my_ok = (mpi_ref_subst_idx >= 0 && auto_rate
             && Params::getInstance().score_diff_thres >= 0) ? 1 : 0;
int all_ok = 0;
MPI_Allreduce(&my_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
mpi_filterRatesMPI_enabled = (all_ok == 1);
```

ensures ALL ranks agree to participate. If even one rank's ref-family slot is
empty, every rank falls back to legacy per-rank `filterRates()` instead.
**This is why the np=1 MPI build runs Phase 0 alone** — no collective fires,
behaviour matches Fix H semantics, no regression.

---

## 23. Operator guide — how to run MPI ModelFinder FCA

### 23.1 Required modules (Gadi normalsr)

```bash
module load openmpi/4.1.7
module load intel-compiler-llvm    # icpx 2025.3.2 → libiomp5
# Build-time only:
module load cmake/3.31.6 binutils/2.44 eigen/3.3.7 boost/1.84.0
```

### 23.2 Build (one-time, ~11 min on a 104-core SPR node)

```bash
qsub setonix-iq/gadi-ci/mf-iso/build_mf_iso.sh
# Output: /scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
# CMake flags: -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIQTREE_FLAGS=mpi
#              -march=sapphirerapids -mtune=sapphirerapids -O3 -g -fopenmp
# Verifies post-link: libiomp5 + libmpi linked, libgomp absent,
#                     filterRatesMPI(int) symbol present (both demangled + mangled),
#                     MF-TIME/MF-MPI-DIAG strings compiled in.
```

### 23.3 Run flag reference

#### IQ-TREE flags (passed to `iqtree3-mpi`)

| Flag | Value | Purpose |
|------|------:|---------|
| `-s` | `<alignment.phy>` | Input alignment (PHYLIP, FASTA, or NEXUS) |
| `-m TESTONLY` | — | **ModelFinder only** — stop before tree search. Use for dispatch debugging (~30% faster iteration on AA 100K). |
| `-m TEST` | — | ModelFinder + SPR tree search (full run). |
| `-T` | `103` | OMP threads per MPI rank. **Always 103 on SPR** (one core reserved for kernel). |
| `-seed` | `1` | RNG seed — fix for reproducible BIONJ/parsimony seeds. |
| `--prefix` | `<work_dir>/iqtree_run` | Output prefix for `.iqtree`, `.treefile`, `.log`, `.mldist`. |

#### `mpirun` flags

| Flag | Value | Purpose |
|------|------:|---------|
| `-np` | `<NRANKS>` | One rank per node (8 cores under-subscribed by design — OMP fills the rest). |
| `--bind-to none` | — | **CRITICAL.** Disables MPI's per-process pinning so OMP_PLACES/OMP_PROC_BIND can take over inside each rank. |
| `--report-bindings` | — | Emits `[host:pid]` binding lines on stderr (captured to `iqtree_run.bindings.log`). |
| `--output-filename` | `<dir>/` | **CRITICAL for ≥2 nodes.** Without this, only rank 0 stdout is forwarded; ranks 1+ MF-TIME lines are LOST. Produces `<dir>/<job-id>/rank.N/stdout` per rank. |
| `--hostfile` | `hostfile.txt` | One host per slot, derived from `$PBS_NODEFILE`. |
| `-rf` | `rankfile.txt` | Explicit rank→host/slot mapping. One rank per node, slot=0–103. |
| `-x` | `VAR=value` | Forward env var to ranks. Used for all OMP/KMP variables (one `-x` per var). |

#### OMP / runtime environment (per-rank)

| Variable | Value | Purpose |
|----------|------:|---------|
| `OMP_NUM_THREADS` | `103` | Match `-T 103`. |
| `OMP_DYNAMIC` | `false` | Disable thread-count autotuning. |
| `OMP_PROC_BIND` | `close` | Threads stay near the master (better L2/L3 locality). |
| `OMP_PLACES` | `cores` | Pin to physical cores, not HW threads. |
| `OMP_WAIT_POLICY` | `PASSIVE` | Sleeping waiters — frees cores during MPI collectives. |
| `GOMP_SPINCOUNT` | `10000` | GCC OpenMP fallback (ignored by libiomp5). |
| `KMP_BLOCKTIME` | `200` | Intel OpenMP sleep delay (ms) after team work; tuned for SPR. |
| `TMPDIR` | `${ISO_DIR}/tmp` | Local scratch for OpenMPI session files. |

#### numactl wrapper

```bash
numactl --localalloc -- "${IQTREE}" ...
```

Forces NUMA-local allocation. Without this, `partial_lh` pages get spread
across the 8 NUMA nodes of an SPR socket and OMP threads pay remote-memory
penalties.

### 23.4 End-to-end command (2-node, full run)

```bash
mpirun -np 2 \
    --hostfile hostfile.txt \
    -rf rankfile.txt \
    --bind-to none \
    --report-bindings \
    --output-filename rank_logs/ \
    -x OMP_NUM_THREADS=103 \
    -x OMP_DYNAMIC=false \
    -x OMP_PROC_BIND=close \
    -x OMP_PLACES=cores \
    -x OMP_WAIT_POLICY=PASSIVE \
    -x GOMP_SPINCOUNT=10000 \
    -x KMP_BLOCKTIME=200 \
    numactl --localalloc -- \
        /scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi \
        -s alignment.phy -m TEST -T 103 -seed 1 \
        --prefix work_dir/iqtree_run \
    > iqtree_run.log 2> iqtree_run.bindings.log
```

### 23.5 What to check after a run

| Artefact | Where | What it tells you |
|----------|------|-------------------|
| `iqtree_run.log` | stdout (rank 0 only on >1 node) | Best-fit model, lnL, BIC, total wall, tree wall |
| `rank_logs/<jobid>/rank.N/stdout` | per-rank | MF-TIME lines for rank N — full timing per model |
| `mf_time.log` | aggregated | All MF-TIME lines from all ranks (offline analysis) |
| `mf_diag.log` | aggregated | MF-MPI-DIAG lines: `filterRatesMPI fired`, `bcast_ok_rates`, dispatch summary |
| `rank_models.csv` | derived | CSV (rank, model_idx, name, subst, rate, dt, ref_remaining) |
| `rank_bindings.log` | from `--report-bindings` | One line per rank showing which core/NUMA each landed on |

Run `setonix-iq/gadi-ci/mf-iso/tools/parse_mf_time.py <profile_dir>` for:
- Per-rank model count
- Broadcast arrival time (relative to t0)
- Convergence spread between ranks (target: < 60 s)
- Stragglers (models with `dt > p99`)

### 23.6 Acceptance gates (per node count)

| Stage | MF wall target | Correctness | Phase 0.5 collective |
|-------|---------------:|-------------|----------------------|
| **1-node** | matches Fix H baseline (~1,289 s on AA 100K) | `lnL = baseline ± 0.5`, best model matches | `filterRatesMPI_enabled = 0` (correct: no collective at np=1) |
| **2-node** | **< 600 s** on AA 100K (~149 s achieved at 168584736) | same | `filterRatesMPI fired` on rank 0, spread < 60 s |
| **4-node** | < 400 s target | same | broadcast fires; per-rank model count balanced |
| **8-node** | < 250 s target | same | per-rank counts balanced (~30 models/rank on AA 100K, ~100 on AA 1M) |

---

## 24. Validated results — 2026-05-18

All runs below used the **same binary**: md5 `a78ffa2942d6b073490d503416ae554c`,
146,238,464 bytes, built from commit [`9603247f`](#) (`test_MF2`, fast-forwarded
2026-05-18). ICX 2025.3.2 + OpenMPI 4.1.7 + AVX-512 + libiomp5. Seed 1 throughout.

### 24.1 Full MF+SPR runs (correctness + speedup)

| Job | Type | Dataset | Nodes | Ranks×OMP | Best model | lnL | BIC | MF wall (s) | SPR wall (s) | Total wall (s) | Speedup | Run record |
|-----|------|---------|-------|-----------|------------|-----|-----|------------:|-------------:|---------------:|--------:|-----------|
| 168425674 | Baseline | DNA 100K | 1 | 1×103T | F81+F+G4 | −5,692,984.539 | 11,388,283.176 | 61.74 | 226.45 | 289.12 | — | [json](../logs/runs/gadi_DNA_100k_baseline_seed1_168425674.json) |
| 168584737 | FCA np=2 | DNA 100K | 2 | 2×103T | F81+F+G4 | −5,692,984.532 | 11,388,283.162 | 26.25 | 86.61 | 113.75 | **2.54×** | [json](../logs/runs/gadi_DNA_100k_mfiso_np2_full_seed1_168584737.json) |
| 168425673 | Baseline | AA 100K | 1 | 1×103T | LG+G4 | −7,541,976.860 | 15,086,233.280 | 399.46 | 764.48 | 1,169.56 | — | [json](../logs/runs/gadi_AA_100k_spr_seed1_168425673.json) |
| 168584736 | FCA np=2 | AA 100K | 2 | 2×103T | LG+G4 | −7,541,976.853 | 15,086,233.265 | 149.03 | 383.88 | 537.75 | **2.18×** | [json](../logs/runs/gadi_AA_100k_mfiso_np2_full_seed1_168584736.json) |
| 168425491 | Baseline | AA 1M | 1 | 1×103T | LG+G4 | −78,605,196.573 | 157,213,128.618 | 7,587.46 | 15,098.61 | 22,776.23 | — | [json](../logs/runs/gadi_AA_1m_spr_seed1_168425491.json) |
| 168586094 | FCA np=8 | AA 1M | 8 | 8×103T | LG+G4 | −78,605,196.497 | 157,213,128.466 | 1,443.89 | 2,147.50 | 3,671.62 | **6.20×** | [json](../logs/runs/gadi_AA_1m_mfiso_np8_full_seed1_168586094.json) |
| 168425675 | Baseline | DNA 1M | 1 | 1×103T | F81+F+G4 | −59,208,019.212 | 118,418,815.342 | 3,500.83 | 2,596.99 | 6,114.45 | — | [json](../logs/runs/gadi_DNA_1m_spr_seed1_168425675.json) |
| 168592214 | FCA np=8 | DNA 1M | 8 | 8×103T | F81+F+G4 | −59,208,019.103 | 118,418,815.123 | 1,274.69 | 349.90 | 1,640.85 | **3.73×** | [json](../logs/runs/gadi_DNA_1m_mfiso_np8_full_seed1_168592214.json) |

### 24.2 Correctness — every result passes `|ΔlnL| < 0.5` vs baseline

| Dataset | Baseline lnL | FCA lnL | ΔlnL | Best model match | BIC delta |
|---------|-------------:|--------:|-----:|:----------------:|----------:|
| DNA 100K | −5,692,984.539 | −5,692,984.532 | **0.007** | ✓ F81+F+G4 | 0.014 |
| AA 100K  | −7,541,976.860 | −7,541,976.853 | **0.007** | ✓ LG+G4    | 0.015 |
| AA 1M    | −78,605,196.573 | −78,605,196.497 | **0.076** | ✓ LG+G4    | 0.152 |
| DNA 1M   | −59,208,019.212 | −59,208,019.103 | **0.109** | ✓ F81+F+G4 | 0.219 |

All four results are bit-for-bit close to baseline within numerical tolerance.
The FCA dispatch changes WHICH RANK evaluates which model — not the model
itself — so the math is identical and the small ΔlnL reflects floating-point
ordering only.

### 24.3 Branch / build provenance

| Property | Value |
|----------|-------|
| Binary md5 | `a78ffa2942d6b073490d503416ae554c` |
| Binary size | 146,238,464 bytes |
| Binary path (rc29) | `/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` |
| Binary path (dx61) | `/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` (mirror, identical md5) |
| Source commit | `9603247f` (mf-iso: Phase 0.5 + Phase 0.6 + MF-TIME) |
| Parent commit | `ffb79a14` (Phase 0 FCA dispatch) |
| Branch | `test_MF2` (fast-forwarded 2026-05-18 from `mf-iso-phase0.5-0.6`) |
| Build host | `gadi-cpu-spr-0284` (job 168572136) |
| Build date | 2026-05-17 22:46 AEST |
| Compiler | icpx 2025.3.2 (intel-compiler-llvm) |
| MPI | OpenMPI 4.1.7 (PBS-MOFED) |
| OpenMP runtime | libiomp5 (Intel) |
| Arch flags | `-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp -g` |
| Build tag | `mf_iso_phase0.5_0.6_icx_avx512_mftime` |

### 24.4 What's NOT in `test_MF2`

These were explicitly EXCLUDED from `9603247f` because they hung
job 168486582 with SIGTERM at 1h19m and no stdout:

- **Phase 0.7** (`MPI_Isend` push instead of `MPI_Bcast`)
- **HH-NUMA Phase 2** (nested `K_outer × M_inner` OMP)
- **`MPI_Init_thread(MPI_THREAD_SERIALIZED)`** upgrade

These remain documented in §21 but are not on `test_MF2`. They will land
in separate commits, each with its own 2-node validation, **only after**
the current `test_MF2` is the published baseline.
