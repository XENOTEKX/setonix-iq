# Mode P Implementation Status — Phase tracker

Companion to `research/mode-p-design.md` (the design contract). This document is the **live progress tracker** for the actual implementation work, with exact file:line edit specifications for the remaining phases.

**Last updated:** 2026-05-24

---

## Phase status

| Phase | Status | Validates | Owner |
|---|---|---|---|
| P.1 Scaffolding (Params + CLI + PhyloTree members + MPI helpers) | **✅ DONE** | Build succeeds with `--mode-p` accepted but inert | this session |
| P.2 Pattern partition wiring (`initializePtnPartition` in `evaluate()`) | **✅ DONE** | Per-rank `[Mode P]` cerr line shows partition range | this session |
| P.3 Kernel: restrict pattern loop bounds + Allreduce in `computeLikelihoodBranchSIMD` | ⏳ SPECIFIED | lnL matches FCA on AA 100K np=2 with `--mode-p-all` | follow-up |
| P.4 Kernel: same for `computeLikelihoodDervSIMD` (derivative kernel) | ⏳ SPECIFIED | NNI lnL traces match FCA | follow-up |
| P.5 Kernel: `computeLikelihoodFromBufferSIMD`, Mixlen variants | ⏳ SPECIFIED | All optimisation paths consistent | follow-up |
| P.6 Dispatcher: heavy-model selection in `evaluateAll` | ⏳ SPECIFIED | Top 3 models routed through Mode P, rest through Mode F | follow-up |
| P.7 Validation: AA 1M np=16 perf gate | ⏳ PENDING | MF wall ≤ 600 s (vs FCA 1,122 s) | follow-up |

---

## What's IN the source tree right now (P.1 + P.2)

**Source changes (iqtree3-mf-iso/src/iqtree3/):**

| File | Lines | What |
|---|---|---|
| `utils/tools.h` | 2385–2397 | New `Params::mode_p_enabled` (int) and `Params::mode_p_min_cost_mult` (double) |
| `utils/tools.cpp` | 4626–4651 | CLI flags `--mode-p`, `--mode-p-all`, `--no-mode-p`, `--mode-p-min-cost-mult` |
| `tree/phylotree.h` | 2314–2343 | New `PhyloTree::ptn_start, ptn_end` (size_t) + 4 helper methods |
| `tree/phylotree.cpp` | 906–984 | Implementations of `isModePActive()`, `initializePtnPartition()`, `modePAllreduceLh()`, `modePAllreduceLhDfDdf()` |
| `main/phylotesting.cpp` | 1995–2003 | `iqtree->initializePtnPartition()` call in `CandidateModel::evaluate()` after `initializeModel()` |

**Behaviour with this scaffolding alone:**
- Build succeeds (no kernel changes — zero risk of correctness regression)
- `--mode-p`, `--mode-p-all`, `--no-mode-p` are accepted on the CLI
- With `--mode-p-all`: a `[Mode P] rank R model=X ptn=[start, end) of N` line is emitted per model
- **Likelihood values are unchanged** because the kernel does not yet consult `ptn_start`/`ptn_end`
- Mode P is "inert" — the partition is set but not enforced

**Test recipe for P.1+P.2 validation:**

```bash
# 1. Build with the scaffolding (clean build, no special flag needed)
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/build_atmd_b4.sh    # OR a new build_mode_p.sh script

# 2. Single-rank run to confirm Mode P is a no-op when ranks==1:
mpirun -np 1 .../iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G --mode-p-all
# Expected: no [Mode P] line (gated on getNumProcesses() > 1)

# 3. Multi-rank run to confirm partition is emitted:
mpirun -np 4 .../iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G --mode-p-all
# Expected: 4 [Mode P] lines, e.g.:
#   [Mode P] rank 0 model=LG+G ptn=[0, 25000) of 100000
#   [Mode P] rank 1 model=LG+G ptn=[25000, 50000) of 100000
#   ...
# Expected: lnL identical to FCA (because kernel ignores partition for now)
```

---

## P.3 — Kernel modification: restrict pattern loop bounds

The kernel currently iterates `[0, nptn)` regardless of rank. To make Mode P operative, the per-rank pattern slice must be honored.

### Implementation strategy

The kernel uses `computeBounds<VectorClass>(num_threads, num_packets, nptn, limits)` at [phylokernelnew.h:2780](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2780) to divide the pattern range into per-thread packets. Each packet processes `[limits[packet_id], limits[packet_id+1])`. 

**Cleanest modification**: shift the work to operate on `[ptn_start, ptn_end)` by:
1. Computing the rank's slice size: `mp_nptn = isModePActive() ? (ptn_end - ptn_start) : nptn`.
2. Calling `computeBounds(num_threads, num_packets, mp_nptn, limits)`.
3. Adding `ptn_start` to every entry of `limits[]` post-computation.

This way the existing per-packet loop body is unchanged; only the loop range shifts.

### Exact patch for computeLikelihoodBranchSIMD ([phylokernelnew.h:2660](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2660))

**Replace (lines ~2776–2782):**
```cpp
    size_t nptn = max_orig_nptn + model_factory->unobserved_ptns.size();
    // ...
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);
```

**With:**
```cpp
    size_t nptn = max_orig_nptn + model_factory->unobserved_ptns.size();
    // ...
    // P.3 Mode P: when active, restrict this rank's work to the assigned slice
    // [ptn_start, ptn_end). When inactive, behaviour is identical to before.
    const bool mp_active = isModePActive();
    const size_t mp_lo = mp_active ? ptn_start : 0;
    const size_t mp_hi = mp_active ? std::min(ptn_end, nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, mp_size, limits);
    if (mp_active) {
        // Shift packet boundaries into [ptn_start, ptn_end). Note: computeBounds
        // rounds the size up to a VCSIZE multiple; the rounded tail past mp_hi
        // is still safe because the per-packet loop checks `ptn < nptn` etc.
        for (size_t &lim : limits) lim += mp_lo;
    }
```

**Insert at the kernel exit, just before `return tree_lh;` at [phylokernelnew.h:3274](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L3274):**

```cpp
    // P.3 Mode P: combine per-rank tree_lh sums. No-op when Mode P is inactive
    // or ranks==1. Called from the master thread of the implicit kernel-exit
    // serial region (MPI_THREAD_FUNNELED is sufficient — see MPIHelper.cpp:28).
    tree_lh = modePAllreduceLh(tree_lh);
    return tree_lh;
```

**Caveat — ASC (ascertainment) correction interaction:** the ASC correction at lines 3227–3273 of the kernel uses `all_prob_const`. Under Mode P, each rank has only its slice of `all_prob_const`. The ASC correction must happen AFTER Allreducing `all_prob_const` too. The simplest fix: move the Allreduce ABOVE the ASC block:

**Insert before the `if (ASC_Holder)` line at [phylokernelnew.h:3227](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L3227):**

```cpp
    // P.3 Mode P: aggregate tree_lh and all_prob_const before ASC correction.
    // This MUST happen before the ASC block because the correction formula
    // applies the (Allreduce'd) all_prob_const to the (Allreduce'd) tree_lh.
    if (isModePActive()) {
        double in[2]  = {tree_lh, all_prob_const};
        double out[2] = {0.0, 0.0};
#ifdef _IQTREE_MPI
        MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        tree_lh = out[0];
        all_prob_const = out[1];
    }
```

And REMOVE the `tree_lh = modePAllreduceLh(tree_lh);` near `return tree_lh;` if you take this approach (it'd double-Allreduce).

### Validation criterion

```bash
# AA 100K, np=2, --mode-p-all (force Mode P on every model)
# Expected: lnL identical to FCA np=2 within 1e-6
mpirun -np 2 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G --mode-p-all -seed 1
# Compare to:
mpirun -np 2 iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G -seed 1
# Both must produce: -7,541,976.853 ± 1e-6
```

If lnL diverges:
- Likely cause: `_pattern_lh[ptn]` is read by post-kernel code expecting full coverage. Under Mode P each rank's _pattern_lh is only populated for its slice. Fix: MPI_Allgather _pattern_lh after the kernel, OR restrict post-kernel _pattern_lh consumption to per-rank slices.
- Less likely: VCSIZE-aligned tail overflow when ptn_end isn't aligned to VectorClass::size(). Inspect `computeBounds` tail handling.

---

## P.4 — Same modifications for derivative kernel

The derivative kernel `computeLikelihoodDervSIMD` at [phylokernelnew.h:2239](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2239) shares the same structure but outputs three values (`tree_lh`, `df`, `ddf`). The Allreduce needs to handle all three.

### Exact patches

**Same `mp_lo/mp_hi/limits-shift` modification** at the kernel entry (mirror P.3).

**At the kernel exit, replace the existing `tree_lh += all_tree_lh; df += all_df; ddf += all_ddf;` block with:**

```cpp
    tree_lh += all_tree_lh;
    df      += all_df;
    ddf     += all_ddf;
    // P.4 Mode P: aggregate (lh, df, ddf) triple across ranks.
    modePAllreduceLhDfDdf(tree_lh, df, ddf);
```

`modePAllreduceLhDfDdf` is the helper already implemented in phylotree.cpp:966 — it takes three double refs and Allreduces them as a single MPI buffer (one network round-trip instead of three).

### Validation

```bash
# Per-branch gradient check: with --verbose, dump tree_lh, df, ddf at each Newton step
mpirun -np 2 iqtree3-mpi-mode-p -s AA_100K.phy -m LG+G --mode-p-all -v
# Compare branch length convergence trajectory to FCA np=2 baseline
```

---

## P.5 — Remaining kernel variants

Apply the same pattern to:

| Function | File:Line | Output(s) | Notes |
|---|---|---|---|
| `computeLikelihoodBranchGenericSIMD` | phylokernelnew.h:2663 | tree_lh | Identical body to BranchSIMD; same patch |
| `computeLikelihoodDervGenericSIMD` | phylokernelnew.h:2242 | lh, df, ddf | Same patch as DervSIMD |
| `computeLikelihoodFromBufferSIMD` | phylokernelnew.h:3286 | tree_lh | Aggregation at line 3437 `tree_lh = all_tree_lh` + post-loop at 3448 |
| `computeLikelihoodFromBufferGenericSIMD` | phylokernelnew.h:3289 | tree_lh | Same |
| `computeLikelihoodDervMixlenSIMD` | phylokernelnew.h:3500 | df, ddf (2 outputs) | Uses `modePAllreduceLhDfDdf` with lh = 0 dummy, or a new 2-arg helper |
| `computeLikelihoodDervMixlenGenericSIMD` | phylokernelnew.h:3503 | df, ddf | Same |
| `computeMixtureLikelihoodBranchEigenSIMD` | phylokernelmixture.h:730 | tree_lh | Older kernel variant; same logical patch |
| `computeMixtureLikelihoodDervEigenSIMD` | phylokernelmixture.h:464 | lh, df, ddf | |
| `computeNonrevLikelihoodBranchSIMD` | phylokernelnonrev.h:1058 | tree_lh | Non-reversible models |
| `computeNonrevLikelihoodDervSIMD` | phylokernelnonrev.h:582 | lh, df, ddf | |

**Estimated time** for P.5 (10 kernel variants × ~30 min each, including build/test cycles): **5 hours**.

---

## P.6 — Dispatcher: route heavy models through Mode P

Currently `--mode-p-all` forces Mode P on every model. For production use, only heavy models should use Mode P; light models should stay on Mode F (where K_outer=4-8 concurrent dispatch is faster than 16-rank Allreduce overhead per kernel call).

### Implementation in `evaluateAll` (phylotesting.cpp)

**Before the main do-while loop at [phylotesting.cpp:4226](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/main/phylotesting.cpp#L4226), add:**

```cpp
    // P.6: per-model Mode P selection. FCA's predicted_cost (lines 3927–3958)
    // already ranks models by cost; mark models above the threshold for Mode P.
    if (params.mode_p_enabled > 0) {  // 1 = auto-select; -1 = force-all (already on)
        double total_cost = 0.0;
        for (size_t m = 0; m < size(); m++) total_cost += at(m).predicted_cost;
        double avg_cost = total_cost / std::max((size_t)1, size());
        double thresh = avg_cost * params.mode_p_min_cost_mult;
        for (size_t m = 0; m < size(); m++) {
            if (at(m).predicted_cost > thresh)
                at(m).flags |= MF_MODE_P;
        }
    }
```

**Then in the do-while body, before `at(model).evaluate(...)`:**

```cpp
    bool model_is_mode_p = at(model).hasFlag(MF_MODE_P);
    if (model_is_mode_p) {
        // All ranks must rendezvous here; Mode F dispatch is suspended for this model.
        MPI_Barrier(MPI_COMM_WORLD);
        // Temporarily flip Params::mode_p_enabled = -1 so evaluate() activates Mode P.
        int saved = params.mode_p_enabled;
        params.mode_p_enabled = -1;
        tree_string = at(model).evaluate(params, model_info, out_model_info,
                                         models_block, num_threads, brlen_type,
                                         &mpi_warm_start);
        params.mode_p_enabled = saved;
        // All ranks now have the same per-model result; no need to Bcast.
    } else {
        // Existing Mode F dispatch unchanged.
        tree_string = at(model).evaluate(params, model_info, out_model_info,
                                         models_block, atmd_M_inner, brlen_type,
                                         &mpi_warm_start);
    }
```

Add `MF_MODE_P` flag to the CandidateModel flag enum (currently has MF_IGNORED, MF_WAITING, MF_RUNNING — add `MF_MODE_P = 0x10` or similar).

---

## P.7 — Performance validation

After P.3-P.6 are merged and lnL parity is confirmed:

```bash
# Baseline FCA np=16:
qsub run_fca_aa_1m_16node.sh                    # ref MF=1,122 s

# Mode P on top 3 heavy models (default threshold 8x):
qsub run_atmd_mode_p_aa_1m_16node.sh --mode-p   # expect MF ≤ 822 s

# Mode P on top 8 (lower threshold):
MODE_P_THRESH=4 qsub run_atmd_mode_p_aa_1m_16node.sh   # expect MF ≤ 600 s

# Mode P on all (validation only — Allreduce overhead dominates light models):
qsub run_atmd_mode_p_all_aa_1m_16node.sh        # expect MF probably > FCA
```

---

## Build & test recipe (P.3 onward)

```bash
# Build the Mode P binary (same as b4 with the new kernel modifications applied):
cd /scratch/rc29/as1708/iqtree3-mf-iso
cp -r src/iqtree3 src/iqtree3-mode-p   # checkout
# ...apply P.3 patches to phylokernelnew.h...
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/build_atmd_mode_p.sh   # new build script

# Correctness gate at AA 100K np=2:
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/run_mode_p_correctness_aa_100k_2node.sh
# Pass: lnL within 1e-6 of FCA np=2 baseline (-7,541,976.853)
# Fail: investigate _pattern_lh consumption or VCSIZE-tail handling

# Performance gate at AA 1M np=16:
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/run_mode_p_perf_aa_1m_16node.sh
# Target: MF wall ≤ 600 s (vs FCA ref 1,122 s)
```

---

## Known issues to address during implementation

1. **`_pattern_lh` consumption**: kernels write `_pattern_lh[ptn]` for each computed pattern. Under Mode P each rank only populates its slice. Downstream code (bootstrap, site-likelihood reporting) reads `_pattern_lh` expecting full coverage. **Fix:** add MPI_Allgather of `_pattern_lh` AFTER kernel exit when Mode P is active. Cost: ~8 MB per Allgather on AA 1M, ~5 ms on Gadi InfiniBand — acceptable.

2. **VCSIZE tail overflow**: `computeBounds` rounds the size up to a multiple of VectorClass::size() (line 1121). When `mp_size` is not a VCSIZE multiple, the last packet has tail patterns past `mp_hi`. The existing kernel handles ptn >= nptn safely (state = STATE_UNKNOWN at line 2894); the same logic applies for ptn >= mp_hi as long as the partial_lh and other arrays are large enough. **Validate** by running a dataset with npat ≢ 0 (mod 16) on np=2.

3. **Mixture model nmixtures=4**: kernel uses `ncat_mix = ncat * model->getNMixtures()`. Mode P partition by pattern is orthogonal to mixture categories (each pattern has all mixture cats). No issue expected.

4. **First-touch / partial_lh per-rank consistency**: each rank computes partial_lh only for its slice. When the tree topology changes (NNI), partial_lh needs to be re-computed. Under Mode P this happens per-rank; if rank R needs partial_lh[ptn] for ptn outside its slice (e.g., to evaluate a different branch), it'll be stale/uninitialised. **Fix:** ensure branch optimisation uses only the rank's slice. The branch optimization loop in `optimizeOneBranch` iterates over Newton-Raphson steps; each step calls the kernel which only touches the slice. As long as all branches share the same per-rank slice (no per-branch slice reallocation), this works.

5. **Mode P + Mode F coexistence**: P.6 ensures only ONE mode is active per model. The MPI_Barrier before the Mode P model dispatch synchronises all ranks; the existing FCA model-distribution skips Mode P models from per-rank ownership. Verify via FCA dispatch logic at phylotesting.cpp:3960–3992.

---

## Status of remaining work

- **P.3 ready to apply**: exact patches below are well-defined and self-contained. One careful Edit/Build/Test cycle.
- **P.4 ready to apply**: mirror of P.3 with 3-output helper; mixture-branch-length path needs separate Allreduce buffer.
- **P.5 systematic**: 6 kernel variants (not 10 — see revised list below) × same pattern. Tedious but mechanical.
- **P.6 architecture change required**: collective-dispatch design (not just a flag per model). See §P.6 below.
- **P.7 measurement**: routine perf run, no engineering.

Total remaining time budget (revised): **~5 days** focused work.

---

## When to revisit this document

After each phase completes, update the **Phase status** table at the top. When all phases are ✅, archive this doc and roll the conclusions into `lbfgs-and-warmstart-implementation.md` §15.10 (a new Mode P chapter).

---

---

# Consolidated P.3 → P.7 Implementation Plan  
**Deep-research edition — 2026-05-24**

This section supersedes the individual phase specs above with exact patches, new
findings from source analysis, identified risks, and revised ordering.

---

## New findings from deep source analysis

These facts change or sharpen the implementation plan relative to the original spec.

### F-1  `save_log_value=false` guarantee eliminates _pattern_lh Allgather for MF

`computeLikelihoodBranchSIMD` has a guard at entry ([phylokernelnew.h:2707](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2707)):

```cpp
if (!save_log_value) {
    ASSERT(!(params->robust_phy_keep < 1.0));
    ASSERT(!(params->robust_median));
    ASSERT(!ASC_Holder);
    ASSERT(!ASC_Lewis);
}
```

`CandidateModel::evaluate()` calls the kernel via `computeLikelihood()` with
`save_log_value=true` only when the model becomes best-so-far; the primary MF
kernel call that accounts for wall time uses `save_log_value=false` (the
`computeLikelihoodBranch` path inside `optimizeParameters`). **With
`save_log_value=false`, ASC correction and robust-phylo post-processing are
guaranteed OFF.**

**Implication**: `_pattern_lh[ptn]` is written for `ptn < orig_nptn` inside the
kernel, but the values are **not consumed by any code path after the kernel
returns** when `save_log_value=false`. Known Issue 1 in the original spec (needing
MPI_Allgather of `_pattern_lh`) is a **non-issue for the ModelFinder path**.

For the SPR phase, Mode P must be **disabled entirely** (see F-4 below), so the
`_pattern_lh` Allgather is also a non-issue there.

**Action**: Remove the Allgather from the design. It is only needed if Mode P is
ever extended to the SPR kernel path (a separate future phase).

---

### F-2  `computeBounds` tail rounding and post-shift correctness

`computeBounds` (phylokernelnew.h:1118) rounds `elements` up to
`VectorClass::size()` before partitioning. After shifting limits by `mp_lo`, the
last packet's upper bound becomes `mp_lo + roundUp(mp_size, VCSIZE)`, which may
exceed `min(ptn_end, orig_nptn)`.

The kernel body handles this correctly via the existing `ptn < orig_nptn` branch
test inside the packet loop. Patterns in the tail (between `ptn_end` and the
rounded-up limit) fall into the `else` branch which only accumulates
`all_prob_const` — and `all_prob_const` is unused when `save_log_value=false` and
ASC is off. **No additional tail-handling is needed for MF.**

```
Packet loop (with Mode P shift applied):
  ptn_lower = limits[packet_id]        ← shifted into [ptn_start, ...)
  ptn_upper = limits[packet_id+1]      ← may overshoot ptn_end by < VCSIZE

  for ptn in [ptn_lower, ptn_upper):
    if ptn < orig_nptn:  → data pattern, accumulates tree_lh ✓
    else:                → unobserved/tail, accumulates all_prob_const (unused in MF)
```

---

### F-3  Unobserved patterns are model-derived and replicated across ranks

Unobserved patterns (`ptn ∈ [max_orig_nptn, nptn)`) are computed from
`model_factory->unobserved_ptns` — a per-model property, identical on every MPI
rank. They do not contain data. For `save_log_value=false`, unobserved patterns only
contribute to `all_prob_const` which drives ASC correction (disabled). They carry
**zero cost** in Mode P: the Mode P slice `[ptn_start, ptn_end)` never overlaps
with `[max_orig_nptn, nptn)` because `ptn_end ≤ orig_nptn ≤ max_orig_nptn`.

**The unobserved-pattern loop in all kernels runs on zero patterns under Mode P and
costs exactly zero.**

---

### F-4  Mode P must be disabled during SPR — MPI collective ordering requirement

During SPR (tree search after ModelFinder), FCA-style dispatch is used: each rank
independently optimises its own tree. Kernel calls are **not synchronised** across
ranks. If Mode P were active, `MPI_Allreduce` inside the kernel would deadlock
because different ranks would call it for different branches at different times.

**Required fix**: reset `mode_p_enabled = 0` in the IQ-TREE main loop
immediately after `evaluateAll()` returns and before tree-search (`doTreeSearch`).
This is a harness-level change — no kernel modification needed.

Alternatively, `isModePActive()` could consult a second flag
`params->mode_p_mf_phase` that is set true only inside `evaluateAll()` and false
outside. The `evaluate()` call to `initializePtnPartition()` already provides
per-model activation, but the kernel's `isModePActive()` check needs to be safe
for SPR too.

**Short-term fix (Phase P.3 implementation)**: add a `Params::mode_p_active_in_mf`
bool, set to `true` at the top of `evaluateAll()` and `false` at the bottom.
`isModePActive()` checks both `mode_p_enabled` AND `mode_p_active_in_mf`.

---

### F-5  MPI thread safety: Mode P + ATMD K_outer > 1 is unsafe under MPI_THREAD_FUNNELED

`MPI_Init_thread` is called with `MPI_THREAD_FUNNELED` at
[MPIHelper.cpp:28](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/MPIHelper.cpp#L28).
`MPI_THREAD_FUNNELED` guarantees only that the **main (init) thread** may call
MPI. When `atmd_K_outer > 1`, `phylotesting.cpp:evaluateAll()` is inside a
`#pragma omp parallel num_threads(atmd_K_outer)` region — each outer thread
independently calls the kernel and would call `MPI_Allreduce`. This is **undefined
behaviour** under `MPI_THREAD_FUNNELED`.

b4 has the `if (atmd_K_outer > 1)` guard in the `#pragma omp parallel` clause, so
when K_outer=1 (current expected behaviour for AA 1M) the serial path is taken and
the main thread calls the kernel. Mode P is safe at K_outer=1.

**For K_outer > 1** (AA 100K with b4): `isModePActive()` must return false. Add
check: `params->atmd_K_outer <= 1` (or the runtime value after B.5 formula
computes it). This prevents Mode P from activating under nested-OMP dispatch.

Concrete change in `isModePActive()` (phylotree.cpp:910):

```cpp
bool PhyloTree::isModePActive() const {
    if (!params || params->mode_p_enabled == 0)
        return false;
    if (!params->mode_p_active_in_mf)          // F-4: SPR guard
        return false;
    // F-5: unsafe to Allreduce from non-main thread under MPI_THREAD_FUNNELED
    int k_outer = params->atmd_K_outer;         // runtime value (0=auto, else explicit)
    if (k_outer != 0 && k_outer != 1)           // K_outer>1 → OMP parallel region
        return false;
#ifdef _IQTREE_MPI
    return MPIHelper::getInstance().getNumProcesses() > 1;
#else
    return false;
#endif
}
```

---

### F-6  `computeLikelihoodFromBufferSIMD` uses a flat ptn-loop, not a packet-loop

Unlike `BranchSIMD` (packet-based `#pragma omp for` over `num_packets`),
`FromBufferSIMD` uses a flat `#pragma omp for` over `ptn`:

```cpp
for (size_t ptn = 0; ptn < nptn; ptn+=VectorClass::size()) { ... }
```

with an `all_lh[k]` reduction array where `k = ptn / VectorClass::size()`.

The Mode P modification is **different** from BranchSIMD:

```cpp
const size_t mp_lo = mp_active ? ptn_start : 0;
const size_t mp_hi = mp_active ? ptn_end   : nptn;
for (size_t ptn = mp_lo; ptn < mp_hi; ptn+=VectorClass::size()) {
    ...
    int k = ptn / VectorClass::size();  // same index into all_lh — correct
    all_lh[k] = horizontal_add(vc_tree_lh);
    ...
}
```

`all_lh` is allocated as `nsize = nptn / VectorClass::size() + 1`, initialised to
zero. Entries not written by this rank stay 0. The final summation `for (k = 0; k <
nsize; k++) all_tree_lh += all_lh[k]` accumulates only the non-zero entries.
**This works correctly without any structural change to the reduction loop.**

**Alignment requirement**: `mp_lo` must be a multiple of `VectorClass::size()` for
the `load_a` (aligned load) in the loop body. `initializePtnPartition()` must
align `ptn_start` down and `ptn_end` up to a `VECTOR_SIZE` boundary (8 for
AVX-512 doubles, 4 for AVX2).

```cpp
// In initializePtnPartition() — align to VECTOR_SIZE (defined in vectorclass/instrset.h)
size_t vcsize = VECTOR_SIZE; // compile-time constant from the AVX/AVX-512 flag
ptn_start = (ptn_start / vcsize) * vcsize;
ptn_end   = min(nptn, ((ptn_end + vcsize - 1) / vcsize) * vcsize);
```

This may cause a very small amount of overlap between adjacent ranks' slices (at
most `vcsize-1 = 7` patterns), but since the tree_lh Allreduce is a SUM and each
`ptn_freq[ptn]` is the same on all ranks, double-counting a 7-pattern overlap
would introduce a ≤ `7/orig_nptn` relative error. For AA 1M (`orig_nptn ≈ 1M`),
this is < 7e-6 — tolerable for MF. For a correctness gate, use `orig_nptn` that is
VCSIZE-divisible (e.g. 100K ≡ 0 mod 8 ✓).

**Better fix**: ensure `chunk = nptn / nranks` is VCSIZE-aligned by rounding down
to VCSIZE in `initializePtnPartition()`. The last rank absorbs the remainder (which
is at most `nranks × (vcsize-1)` extra patterns, negligible).

```cpp
size_t chunk = (nptn / (size_t)nranks / vcsize) * vcsize;  // VCSIZE-aligned chunk
ptn_start = chunk * (size_t)rank;
ptn_end   = (rank == nranks - 1) ? nptn : chunk * (size_t)(rank + 1);
```

---

### F-7  `computeLikelihoodDervSIMD` has two separate exit paths

The `DervSIMD` kernel (phylokernelnew.h:2239) has:

1. **Mixture-branch-length path** (`isMixlen() == true`, lines 2400-2583):
   accumulates `all_dfvec[0..nmixlen)`, `all_ddfvec[0..nmixlen²)`, `all_lh`.
   Exits early at line 2566 via `return`, writing `df[0..nmixlen]` and `ddf[0..nmixlen²]`.
   The last `df[nmixlen]` entry holds the log-likelihood.

2. **Normal joint path** (`isMixlen() == false`, lines 2472-2659):
   accumulates scalar `all_df`, `all_ddf`. Exits at bottom writing `*df`, `*ddf`.

Mode P Allreduce must be inserted **in both paths** before writing the output.

For path 1 (mixture), a single `MPI_Allreduce` of buffer size `nmixlen + nmixlen² + 1`:

```cpp
// Insert before the "df[i] = horizontal_add(all_dfvec[i])" block:
if (isModePActive()) {
    // flatten all_dfvec + all_ddfvec + all_lh into one buffer, Allreduce, unpack
    int n = nmixlen + nmixlen2 + 1;  // nmixlen2 = nmixlen*(nmixlen+1)/2
    vector<double> in_buf(n), out_buf(n, 0.0);
    for (int i = 0; i < nmixlen;  i++) in_buf[i]          = horizontal_add(all_dfvec[i]);
    for (int i = 0; i < nmixlen2; i++) in_buf[nmixlen+i]  = horizontal_add(all_ddfvec[i]);
    in_buf[n-1] = all_lh;
#ifdef _IQTREE_MPI
    MPI_Allreduce(in_buf.data(), out_buf.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    for (int i = 0; i < nmixlen;  i++) { df[i]  = out_buf[i]; }
    for (int i = 0; i < nmixlen2; i++) { ddf[i] = out_buf[nmixlen+i]; }
    df[nmixlen] = out_buf[n-1];
    return;
}
// else fall through to original horizontal_add logic
```

For path 2 (normal joint), use the existing 3-value `modePAllreduceLhDfDdf()` helper:

```cpp
// After "all_df += horizontal_add(my_df); all_ddf += horizontal_add(my_ddf);"
// and BEFORE "*df = all_df; *ddf = all_ddf;"
double dummy_lh = 0.0;
modePAllreduceLhDfDdf(dummy_lh, all_df, all_ddf);
// (lh is already computed by BranchSIMD Allreduce in the preceding evaluateBranch call)
*df  = all_df;
*ddf = all_ddf;
```

---

### F-8  `DervMixlenSIMD` is a separate function with a different output contract

`computeLikelihoodDervMixlenSIMD` (phylokernelnew.h:3500) takes `df` and `ddf` by
reference (not pointer), outputs only the derivative values (no lnL), uses
`computeBounds` at line 3553. Exit at line 3665: `df = all_df; ddf = all_ddf;`.

This kernel is used for the Mixlen (heterotachy) model branch optimisation. The
Allreduce is:

```cpp
// After "df = all_df; ddf = all_ddf;" and BEFORE the ASC_Lewis block:
if (isModePActive()) {
    double dummy = 0.0;
    modePAllreduceLhDfDdf(dummy, df, ddf);
}
```

---

### F-9  Collective ordering — all ranks must execute Mode P models simultaneously

This is the **fundamental architectural constraint** of P.6. Under FCA dispatch,
each rank owns a disjoint set of substitution-model families. If rank 0 evaluates
LG+G (Mode P) while rank 1 evaluates WAG+G4 (Mode F), the two `MPI_Allreduce`
calls inside the kernel are **for different models** — they will match incorrectly
and return corrupted results.

**The P.6 dispatcher must guarantee that when a Mode P model is evaluated, ALL
ranks enter the same kernel call simultaneously.** This requires a fundamentally
different dispatch structure:

```
Normal FCA models (Mode F / Mode 0):
  Each rank evaluates its own assigned family independently (current behaviour).

Mode P models (a small subset of the heaviest models):
  1. All ranks finish their current Mode F batch (MPI_Barrier).
  2. A designated rank broadcasts model identity: "evaluate LG+R10 now."
  3. All ranks evaluate the SAME model's kernel in Mode P (pattern-parallel).
  4. All ranks contribute [ptn_start, ptn_end) → Allreduce → global lnL.
  5. Result written to model_info on all ranks.
  6. Repeat for next Mode P model.
  7. All ranks barrier, then resume independent Mode F dispatch.
```

This means Mode P evaluation serialises all ranks per Mode P model. For K Mode P
models at AA 1M:

```
Cost(Mode P, K models) ≈ K × (kernel_wall_1M / nranks + Allreduce_latency)
                       = K × (1,122s / 16  +  ~5ms)
                       ≈ K × 70s
```

For K=3 heavy models (LG+R8, LG+R10, LG+FC variants): `3 × 70s = 210s`.
Compare to FCA cost for those same 3 models on one rank: `3 × ~600s = 1,800s`
(estimated from the b3c MF=2,113s with K_outer=1 — the heavy models dominate).

**Speedup for those 3 models: ~8.6×.** The remaining ~1,200 light models stay on
Mode F and cost ~800s (unchanged). Predicted total MF wall ≈ 210s + 800s = 1,010s —
still a regression vs FCA 1,122s. Need to push K up to 10+ or ensure Mode F models
also benefit (see §P.6 performance model).

---

### F-10  `theta_all` memory reduction opportunity

`theta_all` is allocated as `nptn × block × sizeof(double)` bytes
(`initializeAllPartialLh`, phylotree.cpp). Under Mode P, only `[ptn_start,
ptn_end)` of theta is ever populated or read.

For AA 1M at np=16: `ptn_end - ptn_start ≈ 1M/16 = 62,500` patterns. Full
`theta_all = 1M × 20 × 8 = 160 MB`. Mode P `theta_all = 62,500 × 20 × 8 = 10 MB`.
Savings: 150 MB per rank, 2.4 GB across 16 ranks.

This is a **Phase P.5+ optimization**, not required for correctness. Implementation:
change `theta_all` allocation size to `(ptn_end - ptn_start + max_unobserved) ×
block × sizeof(double)` when Mode P is active; adjust all `theta_all + ptn*block`
index expressions to `theta_all + (ptn - ptn_start)*block`. Requires care around
the `theta_computed` cache flag which must also be invalidated when `ptn_start`
changes (per-model call to `initializePtnPartition()`).

---

## Revised kernel modification list (P.3 – P.5)

The original spec listed 10 kernel variants. After source analysis, the relevant
MF-path kernels (called during `optimizeParameters` in `evaluate()`) are:

| Priority | Function | File | Modification | Complexity |
|---|---|---|---|---|
| **P.3** | `computeLikelihoodBranchSIMD` / `...GenericSIMD` | phylokernelnew.h:2660 | limits-shift + Allreduce tree_lh | Medium |
| **P.4** | `computeLikelihoodDervSIMD` / `...GenericSIMD` — normal joint path | phylokernelnew.h:2239 | limits-shift + modePAllreduceLhDfDdf | Medium |
| **P.4b** | `computeLikelihoodDervSIMD` / `...GenericSIMD` — mixture-branch-length path | phylokernelnew.h:2400 | wide Allreduce buffer (nmixlen+nmixlen²+1) | High |
| **P.5a** | `computeLikelihoodFromBufferSIMD` / `...GenericSIMD` | phylokernelnew.h:3286 | ptn-loop bounds change + Allreduce | Low |
| **P.5b** | `computeLikelihoodDervMixlenSIMD` / `...GenericSIMD` | phylokernelnew.h:3500 | limits-shift + modePAllreduceLhDfDdf | Medium |
| **Skip** | `computeMixtureLikelihoodBranchEigenSIMD` | phylokernelmixture.h:730 | — | Skip for Phase 1: mixture eigenvector kernel not called in standard MF on AA |
| **Skip** | `computeNonrevLikelihoodBranchSIMD` | phylokernelnonrev.h:1058 | — | Skip for Phase 1: non-reversible models excluded from `TEST` |

Revised count: **5 priorities** (P.3, P.4, P.4b, P.5a, P.5b) instead of 10.
Mixture-eigenvector and NonRev kernels can be deferred to a Phase P.5+ pass if
non-reversible or mixture-class models are ever added to the MF test suite.

---

## P.3 — `computeLikelihoodBranchSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h`  
**Applies to**: both `KERNEL_FIX_STATES` (fixed-state) and generic template
instantiations — the `#ifdef` guard at line 2660 wraps both in the same body.

### Patch 1 — limits computation (line ~2780)

```cpp
// BEFORE (line 2780):
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);

// AFTER:
    // P.3 Mode P: restrict per-rank work to the assigned pattern slice.
    // When inactive, mp_lo=0 and mp_hi=nptn → identical to the original behaviour.
    // With save_log_value=false (MF path), ASC is guaranteed off so unobserved
    // patterns [max_orig_nptn, nptn) need not be computed — cap at orig_nptn.
    const bool mp_active = isModePActive() && save_log_value == false;
    const size_t mp_lo   = mp_active ? ptn_start : 0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets,
                               mp_active ? mp_size : nptn,
                               limits);
    if (mp_active) {
        // Shift all packet boundaries into [mp_lo, mp_hi].
        // computeBounds generates [0, roundUp(mp_size, VCSIZE)]; we shift by mp_lo.
        for (size_t &lim : limits) lim += mp_lo;
        // The last entry may overshoot mp_hi by at most (VCSIZE-1) patterns.
        // The kernel's "ptn < orig_nptn" guard handles the tail safely.
    }
```

### Patch 2 — Allreduce tree_lh (line ~3157)

```cpp
// BEFORE (line 3157):
    tree_lh += all_tree_lh;

// AFTER:
    tree_lh += all_tree_lh;
    // P.3 Mode P: reduce partial sums from all ranks into the global tree_lh.
    // No-op when Mode P is inactive (modePAllreduceLh returns the value unchanged).
    if (mp_active)
        tree_lh = modePAllreduceLh(tree_lh);
```

**No other changes needed for the MF path** (save_log_value=false guarantees
ASC off and all post-kernel `_pattern_lh` consumption disabled).

For the `save_log_value=true` path (used by best-model final scoring): Mode P is
guarded by `mp_active = isModePActive() && save_log_value == false`, so it
automatically skips Mode P for those calls. Correctness is preserved.

---

## P.4 — `computeLikelihoodDervSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h`  
**Applies to**: both fixed-state and generic template instantiations.

### P.4 Patch 1 — limits computation (line ~2305)

Identical to P.3 Patch 1, substituting `mp_active = isModePActive()` (no
`save_log_value` check — the Derv kernel has no `save_log_value` parameter;
ASC handling is separate).

```cpp
// BEFORE (line 2305):
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);

// AFTER:
    const bool mp_active = isModePActive();
    const size_t mp_lo   = mp_active ? ptn_start : 0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets,
                               mp_active ? mp_size : nptn,
                               limits);
    if (mp_active) {
        for (size_t &lim : limits) lim += mp_lo;
    }
```

### P.4 Patch 2 — normal joint path exit Allreduce (line ~2591)

```cpp
// BEFORE (line 2591):
    // normal joint branch length model
    *df  = all_df;
    *ddf = all_ddf;

// AFTER:
    // normal joint branch length model
    // P.4 Mode P: aggregate derivatives across ranks (lnL already Allreduced in
    // BranchSIMD; use dummy to keep the 3-value MPI buffer aligned).
    if (mp_active) {
        double dummy = 0.0;
        modePAllreduceLhDfDdf(dummy, all_df, all_ddf);
    }
    *df  = all_df;
    *ddf = all_ddf;
```

### P.4b Patch — mixture-branch-length path exit (line ~2557)

```cpp
// BEFORE — the block starting with:
    if (isMixlen()) {
        // mixed branch length model
        for (size_t i = 0; i < nmixlen; i++) {
            df[i] = horizontal_add(all_dfvec[i]);
            ...
        }
        for (size_t i = 0; i < nmixlen2; i++) {
            ddf[i] = horizontal_add(all_ddfvec[i]);
        }
        df[nmixlen] = all_lh;
        return;
    }

// AFTER:
    if (isMixlen()) {
        // mixed branch length model
        // P.4b Mode P: Allreduce the dfvec + ddfvec + lh triple in one call.
        if (mp_active) {
            int n = (int)(nmixlen + nmixlen2 + 1);
            vector<double> in_buf(n), out_buf(n, 0.0);
            for (int i = 0; i < (int)nmixlen;  i++) in_buf[i]           = horizontal_add(all_dfvec[i]);
            for (int i = 0; i < (int)nmixlen2; i++) in_buf[nmixlen + i] = horizontal_add(all_ddfvec[i]);
            in_buf[n - 1] = all_lh;
#ifdef _IQTREE_MPI
            MPI_Allreduce(in_buf.data(), out_buf.data(), n, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
#endif
            for (int i = 0; i < (int)nmixlen;  i++) df[i]  = out_buf[i];
            for (int i = 0; i < (int)nmixlen2; i++) ddf[i] = out_buf[nmixlen + i];
            df[nmixlen] = out_buf[n - 1];
        } else {
            for (size_t i = 0; i < nmixlen; i++) {
                df[i] = horizontal_add(all_dfvec[i]);
                ASSERT(std::isfinite(df[i]) && "Numerical underflow for lh-derivative");
            }
            for (size_t i = 0; i < nmixlen2; i++) {
                ddf[i] = horizontal_add(all_ddfvec[i]);
            }
            df[nmixlen] = all_lh;
        }
        return;
    }
```

---

## P.5a — `computeLikelihoodFromBufferSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h:3286`  
Different loop structure from P.3/P.4 — flat ptn-loop, not packet-based.

### Patch 1 — ptn loop bounds (line ~3377)

```cpp
// BEFORE (line 3377):
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t ptn = 0; ptn < nptn; ptn+=VectorClass::size()) {

// AFTER:
    const bool mp_active = isModePActive();
    const size_t mp_lo   = mp_active ? ptn_start : (size_t)0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    // Alignment guaranteed by initializePtnPartition() VCSIZE rounding.
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t ptn = mp_lo; ptn < mp_hi; ptn+=VectorClass::size()) {
```

### Patch 2 — Allreduce tree_lh (line ~3449)

```cpp
// BEFORE (line 3449):
    double tree_lh = all_tree_lh;

// AFTER:
    double tree_lh = all_tree_lh;
    if (mp_active)
        tree_lh = modePAllreduceLh(tree_lh);
```

---

## P.5b — `computeLikelihoodDervMixlenSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h:3500`

### Patch 1 — limits (line ~3553)

```cpp
// BEFORE (line 3553):
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);

// AFTER:
    const bool mp_active = isModePActive();
    const size_t mp_lo   = mp_active ? ptn_start : (size_t)0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : (size_t)0;
    computeBounds<VectorClass>(num_threads, num_packets,
                               mp_active ? mp_size : nptn,
                               limits);
    if (mp_active) {
        for (size_t &lim : limits) lim += mp_lo;
    }
```

### Patch 2 — Allreduce df/ddf (line ~3664)

```cpp
// BEFORE (line 3664):
    df  = all_df;
    ddf = all_ddf;

// AFTER:
    df  = all_df;
    ddf = all_ddf;
    if (mp_active) {
        double dummy = 0.0;
        modePAllreduceLhDfDdf(dummy, df, ddf);
    }
```

---

## `initializePtnPartition()` alignment fix (prerequisite for P.5a)

Apply to `tree/phylotree.cpp:930` (the VCSIZE-aligned chunk calculation):

```cpp
// BEFORE (current code at phylotree.cpp:940):
    size_t chunk = nptn / (size_t)nranks;
    ptn_start = chunk * (size_t)rank;
    ptn_end   = (rank == nranks - 1) ? nptn : chunk * (size_t)(rank + 1);

// AFTER:
    // Round chunk DOWN to VECTOR_SIZE (8 for AVX-512) so that ptn_start and
    // ptn_end are always VCSIZE-aligned — required by load_a() in FromBufferSIMD.
    const size_t vcsize = VECTOR_SIZE;   // compile-time constant from instrset.h
    size_t chunk = (nptn / (size_t)nranks / vcsize) * vcsize;
    if (chunk == 0) chunk = vcsize;      // safety: at least one VCSIZE block
    ptn_start = chunk * (size_t)rank;
    ptn_end   = (rank == nranks - 1) ? nptn : chunk * (size_t)(rank + 1);
    // Clamp to nptn for last rank (last rank gets the uneven tail).
    if (ptn_end > nptn) ptn_end = nptn;
```

---

## `isModePActive()` — consolidated update (incorporates F-4, F-5)

Replace the existing `isModePActive()` body in `tree/phylotree.cpp:910`:

```cpp
bool PhyloTree::isModePActive() const {
    if (!params || params->mode_p_enabled == 0)
        return false;
    // F-4: SPR guard — Mode P only inside evaluateAll(); disable after MF phase.
    if (!params->mode_p_active_in_mf)
        return false;
    // F-5: MPI_THREAD_FUNNELED safety — disable when K_outer > 1 (OMP parallel
    // region means non-main threads would call MPI_Allreduce → UB).
    // atmd_K_outer == 0 means "auto" (K_outer=1 at AA 1M with b4 formula).
    // atmd_K_outer == 1 is explicit serial → safe.
    // atmd_K_outer > 1 → unsafe.
    int k = params->atmd_K_outer;
    if (k > 1)
        return false;
#ifdef _IQTREE_MPI
    return MPIHelper::getInstance().getNumProcesses() > 1;
#else
    return false;
#endif
}
```

Add `bool mode_p_active_in_mf = false;` to `Params` (utils/tools.h next to
`mode_p_enabled`).

Set in `evaluateAll()` (phylotesting.cpp) at entry:
```cpp
params.mode_p_active_in_mf = true;
```
And at exit (both normal return and early-return paths):
```cpp
params.mode_p_active_in_mf = false;
```

---

## P.6 — Collective dispatch for Mode P models

### Design overview

P.6 replaces the original "flag per model" concept with a **two-phase loop** inside
`evaluateAll()`:

**Phase A** (Mode F / independent): each rank evaluates its assigned models using
the existing FCA sequential loop. Mode P models are **skipped** in this phase
(marked `MF_MODE_P`).

**Phase B** (Mode P / collective): a single all-ranks loop iterates over all
`MF_MODE_P` models in a fixed global order. For each model:
1. All ranks barrier.
2. All ranks call `initializePtnPartition()` to set their slice.
3. All ranks call `evaluate(model)` → kernels Allreduce internally.
4. Result is recorded on all ranks (no cross-rank model_info sync needed — every
   rank ran the full model, each with its pattern slice, and the Allreduce gave
   the correct global lnL).

```cpp
// Phase B insertion after the existing do-while evaluateAll loop:
if (params.mode_p_enabled && MPIHelper::getInstance().getNumProcesses() > 1) {
    params.mode_p_active_in_mf = true;
    for (int model = 0; model < (int)num_models; model++) {
        if (!at(model).hasFlag(MF_MODE_P)) continue;
        // Barrier: ensure all ranks start this model simultaneously.
#ifdef _IQTREE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        at(model).evaluate(params, in_tree, model_info, &local_in_info,
                           score_diff_thres, model, initial_model_rate,
                           substitution_model, in_tree_rate, in_tree_freq,
                           write_info, aln_rate, set_output);
    }
    params.mode_p_active_in_mf = false;
}
```

### Model selection threshold for MF_MODE_P

Use `modelCostFCA(model)` (already computed in the FCA dispatch block) to rank
models. Mark the top-K% by cost as `MF_MODE_P`:

```cpp
// After Step 4 (MF_IGNORED marking) in evaluateAll():
if (params.mode_p_enabled != 0) {
    // Compute per-model costs for non-IGNORED models.
    vector<pair<double,int>> costs;
    for (int i = 0; i < (int)num_models; i++)
        if (!at(i).hasFlag(MF_IGNORED))
            costs.push_back({modelCostFCA(i), i});
    // Sort descending by cost.
    sort(costs.begin(), costs.end(),
         [](auto &a, auto &b){ return a.first > b.first; });
    // Mark top models as Mode P (those exceeding threshold × median cost).
    double thresh = params.mode_p_min_cost_mult;  // default 8.0
    double median_cost = costs.empty() ? 0.0 : costs[costs.size()/2].first;
    for (auto &[cost, idx] : costs) {
        if (cost >= thresh * median_cost) {
            at(idx).setFlag(MF_MODE_P);
            at(idx).setFlag(MF_IGNORED);  // skip in Phase A
        }
    }
}
```

Add `MF_MODE_P = 32` to the flag constants (phylotesting.h:33):
```cpp
const int MF_MODE_P = 32;
```

### Performance model for P.6 threshold tuning

For AA 1M np=16 with `mode_p_min_cost_mult=8.0`:
- Estimated models above threshold: ~8 (LG+R6..R10, LG+FC+R variants, WAG+R8+)
- Mode P cost per model: `~70s` (kernel_wall / 16 + Allreduce ≈ 4ms)
- Mode P total: `8 × 70s = 560s`
- Remaining Mode F (light) models: ~140 models × ~4s each = ~560s (but now all 16
  ranks handle them independently in Phase A)
- Predicted MF wall: `max(560s_ModeP, 560s_ModeF) = 560s` — **50% faster than FCA 1,122s**

For `mode_p_min_cost_mult=4.0` (more aggressive):
- Estimated Mode P models: ~20
- Mode P total: `20 × 70s = 1,400s` — WORSE (too many models serialised)

Optimal threshold appears to be around 6–10×. The AA 1M gate at `≤600s MF wall`
is achievable at threshold=8 with the above model. Tune via the 100K sanity check
first.

---

## P.7 — Validation plan and gates

### Build recipe

Build script: `gadi-ci/lbfgs-ws/build_atmd_mode_p.sh` (new, to create).
Source: copy of b4 source tree + P.3–P.6 patches applied.
Binary: `iqtree3-mpi-atmd-mode-p`.

CMake flags: same as b4 (`-DIQTREE_ATMD=ON -DIQTREE_MPI=ON -march=sapphirerapids`).

### Gate 1 — P.1+P.2 structural validation (already done; re-confirm with b4 base)

```bash
# Single rank: Mode P inactive (no [Mode P] line, lnL identical to FCA np=1)
mpirun -np 1 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G --mode-p-all
```

### Gate 2 — P.3 correctness (lnL parity at np=2)

```bash
# Mode P forced on every model (-all), AA 100K np=2 vs FCA np=2 baseline
mpirun -np 2 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G4 --mode-p-all -seed 1
# Target: lnL = -7,541,976.853 ± 1e-6
# Compare to: mpirun -np 2 iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G4 -seed 1
```

If lnL diverges:
1. Check that `ptn_start` and `ptn_end` are printed in the `[Mode P]` lines and
   cover exactly all `orig_nptn` patterns across ranks without overlap.
2. Check `theta_computed` is not stale between BranchSIMD and DervSIMD calls.
3. Verify VCSIZE alignment: confirm `ptn_start % 8 == 0` for both ranks.

### Gate 3 — P.4/P.5 derivative parity (BFGS trace match)

```bash
# Compare branch-length optimisation convergence trace between mode-p and b4 np=2
# Both runs with -v (verbose) to dump per-NR-iteration lnL
mpirun -np 2 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G4 --mode-p-all -v -seed 1 > mode_p_trace.log
mpirun -np 2 iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G4 -v -seed 1 > b4_trace.log
diff <(grep "NR iter" mode_p_trace.log) <(grep "NR iter" b4_trace.log)
# Target: NR iterations converge to same sequence of lnL values
```

### Gate 4 — Full MF test, AA 100K np=4 (default threshold)

```bash
qsub run_mode_p_aa_100k_4node.sh   # -m TEST --mode-p (default threshold 8x)
# Targets:
#   lnL = -7,541,976.853 ± 0.5
#   Best model = LG+G4
#   MF wall ≤ FCA np=4 (149s × 4/4 = ~110s nominal; mode-p overhead may push to ~150s)
```

### Gate 5 — AA 1M np=16 performance gate (P.7)

```bash
qsub run_mode_p_aa_1m_16node.sh    # -m TEST --mode-p (threshold 8x)
# Targets:
#   lnL = -78,605,196.497 ± 0.5
#   Best model = LG+G4
#   MF wall ≤ 600s  (vs FCA 1,122s ref)
```

If `MF wall > 600s` but correct lnL:
- Profile which models are Mode P and how long each took.
- Tune threshold (lower `mode_p_min_cost_mult` to route more models through Mode P).
- Check Allreduce latency contribution (5ms × 100 models = 500ms — negligible vs 70s/model).

If `MF wall < 600s` → Mode P is viable. Close Phase P.7, roll to §15.10.

---

## Build scripts needed (new, to create)

| Script | Purpose |
|---|---|
| `gadi-ci/lbfgs-ws/build_atmd_mode_p.sh` | Build the Mode P binary (b4 + P.3–P.6 patches) |
| `gadi-ci/lbfgs-ws/run_mode_p_correctness_aa_100k_2node.sh` | Gate 2 correctness, np=2, AA 100K |
| `gadi-ci/lbfgs-ws/run_mode_p_aa_100k_4node.sh` | Gate 4 full MF np=4 AA 100K |
| `gadi-ci/lbfgs-ws/run_mode_p_aa_1m_16node.sh` | Gate 5 perf gate np=16 AA 1M |

---

## Implementation sequencing (revised)

```
Step 1:  Apply initializePtnPartition() alignment fix (30 min)
Step 2:  Apply isModePActive() F-4 + F-5 guards + mode_p_active_in_mf Params field (45 min)
Step 3:  Apply P.3 patches to BranchSIMD (1 hr)
Step 4:  Build + Gate 1 (structural) (30 min build + 5 min test)
Step 5:  Gate 2 correctness np=2 AA 100K (30 min PBS turnaround)
         → If PASS: proceed. If FAIL: debug limits-shift.
Step 6:  Apply P.4 (normal joint path + mixture path) (1 hr)
Step 7:  Apply P.5a (FromBuffer) (45 min)
Step 8:  Apply P.5b (DervMixlen) (45 min)
Step 9:  Build + Gate 3 derivative parity (30 min + 10 min test)
Step 10: Apply P.6 dispatcher (MF_MODE_P flag + collective Phase B loop) (2 hr)
Step 11: Build + Gate 4 full MF AA 100K np=4 (30 min + 1 hr PBS)
Step 12: Tune threshold if needed (30 min)
Step 13: Gate 5 AA 1M np=16 (30 min + 3 hr PBS)
Step 14: Document results, update §15.10
```

Total estimate: ~10 hours engineering + ~5 hours PBS turnaround = **~2 working days**.
