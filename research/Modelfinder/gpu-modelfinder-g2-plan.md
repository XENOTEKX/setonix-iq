# G.2 — In-tree integration plan (mapped 2026-06-08; deferred behind the intra-kernel perf lever)

Companion to [gpu-modelfinder-design.md](gpu-modelfinder-design.md) PART II and
[gpu-modelfinder-g1-log.md](gpu-modelfinder-g1-log.md). G.2 wires the validated standalone kernels (K1
postorder lnL, K2 single-edge df/ddf, K3 per-node CUDA graph, the from-buffer lnL(t) evaluator) behind
IQ-TREE's real likelihood seam under `--gpu`, so a real `-m TEST` ModelFinder run uses the GPU. Mapped by a
5-agent workflow over the dev tree `/scratch/rc29/as1708/iqtree3-gpu` (branch `gpu-kernel`).

> **Status: PLAN READY, IMPLEMENTATION DEFERRED.** Per the 2026-06-08 decision, the **intra-kernel compute
> perf lever** (the only lever that beats the 221.6 s wall at the compute-bound 100K count — see the perf
> pass in g1-log) is being done FIRST, in the standalone harness, so G.2 integrates an already-fast kernel.
> The correctness-integration effort below is independent of kernel speed and unchanged by the perf work.

## The seam (verified against source, with line numbers)

The four likelihood fn-pointers (`tree/phylotree.h`):
- `computePartialLikelihoodPointer` — `void(TraversalInfo&, size_t ptn_lower, size_t ptn_upper, int thread_id)` (member :904, typedef :903)
- `computeLikelihoodBranchPointer` — `double(PhyloNeighbor*, PhyloNode*, bool)` (member :951; **typedef has no default arg** even though the virtual has `save_log_value=true`)
- `computeLikelihoodFromBufferPointer` — `double()` **no args** (member :1004)
- `computeLikelihoodDervPointer` — `void(PhyloNeighbor*, PhyloNode*, double* df, double* ddf)` (member :1346)

Assigned by the ISA-specialized setters `setLikelihoodKernelAVX/FMA/AVX512/SSE` (`tree/phylotreeavx.cpp`;
for AA reversible non-site-model the SAFE/NORM branch is ~:149-195) through the **single funnel**
`PhyloTree::setLikelihoodKernel(LikelihoodKernel lk)` at `tree/phylotreesse.cpp:91` (ISA branch :156-174;
`lk` from `params.SSE`, auto-detected → `LK_AVX_FMA` on the Cascade-Lake V100 host). Consumed only via thin
virtual wrappers `tree/phylotreesse.cpp:212-236` (note `computeLikelihoodFromBuffer` at :226-236 only fires
the FromBuffer pointer when `optimize_by_newton` is true, else falls back to the Branch pointer doing a full
traversal).

**Hook:** add a non-virtual `void PhyloTree::setLikelihoodKernelGPU()` (declared `phylotree.h:2081-2085`
under `#ifdef IQTREE_GPU`; defined in a new gated host TU `tree/phylotreegpu.cpp`), called as the **LAST**
step of `setLikelihoodKernel` (before the final return), guarded `if (params && params->gpu && <narrow
gate>)`. Calling it last OVERRIDES the four pointers the ISA setter just populated, leaving
parsimony/dotproduct/safe_numeric/vector_size intact for CPU fallback. The four GPU members must be real
`PhyloTree::` members byte-matching the typedefs (FromBuffer no args; Branch typedef no default), host-
compiled (NOT `.cu`), marshaling state into `extern "C"` launchers in `tree/gpu/*.cu`. Override must be
**idempotent + re-applied at the funnel** (setLikelihoodKernel is re-invoked many times — phylotree.cpp
648/3010/3017, iqtree.cpp:2997 — or the tree silently reverts to CPU mid-run). Because the MF call at
`phylotesting.cpp:1959` runs BEFORE `initializeModel` (getModel() null there), re-apply the override at
`initializeAllPartialLh` (`phylotesting.cpp:2098`) where model/eigen/buffers exist.

## Bridge layer (harness clean-room ↔ in-tree reality)

**Ownership = device-mirror** (NOT consume-in-place): the GPU keeps a parallel device-resident set
(`d_partial`, `d_theta`, `d_tip_partial`) sized from the same `block_size = nptn_padded*nstates*ncat_mix`
(phylotree.cpp:1086-1117); the host `central_partial_lh`/`theta_all` stay CPU-canonical and own topology +
dirty-tracking; a host-side `PhyloNeighbor* → device-slot-id` map (from `indexlh*block_size`,
phylotree.cpp:1184-1196) lets the GPU address device slots in the host-built `traversal_info` postorder.

Gaps reconciled:
1. **Layout** — device-canonical is the harness coalesced `[slot][cat*NS+state][ptn]`; in-tree host is
   `partial_lh[ptn*block + c*nstates + x]` with Vec4d 4-pattern SoA interleave. Because partials live
   device-resident (produced AND consumed by GPU kernels), **no per-step transpose** — the only transposes
   are a one-time H2D of `tip_partial_lh` (built by `computeTipPartialLikelihood`, phylotreesse.cpp:243-375,
   already pre-multiplied by Uinv) reshaped to `[cat*NS+state][ptn]`, and a final D2H of the root partial if
   a host consumer needs it. `echild` is rebuilt **on-device** (K3 `build_echild` from `d_brlen`/`d_eval`/
   `d_U`), never transferred.
2. **Pattern weights** — every reduction weights by `ptn_freq[ptn] = aln->at(ptn).frequency` (integer
   multiplicity; phylotreesse.cpp:548-549). Harness was weight-1. Upload `d_ptn_freq` once per topology;
   K1 root: `tree_lh = Σ_ptn ptn_freq·(log|L_ptn| + buffer_scale_all)`; K2: `df += freq·dfrac`,
   `ddf += freq·(ddfrac − dfrac²)`. Pad patterns get `ptn_freq=0`.
3. **Scaling** — ✅ **CORRECTED:** for AA-100K the CPU uses **NORM_LH (unscaled)**, NOT SAFE_LH. Verified:
   `safe_numeric = (lk_safe_scaling || leafNum ≥ numseq_safe_scaling) || (num_states∉{4,20})`
   (phylotreesse.cpp:95); `numseq_safe_scaling=2000` (tools.cpp:7270), **`leafNum` = taxon count = 100**
   (the mapping agent misread it as the 100000 *site* count), `num_states=20` ⇒ `safe_numeric = (100≥2000)||
   (20∉{4,20}) = FALSE`. So the harnesses' unscaled FP64 path already matches the production oracle; **no
   SAFE_LH per-category ldexp-256 port is needed for the AA-100K validation gate.** (SAFE_LH only matters for
   >2000-taxon datasets — a later generalization; if/when needed, port per-category `lh_max` + conditional
   rescale + UBYTE `scale_num` + `buffer_scale_all`.)
4. **π-fold / eigen convention** — consume IQ-TREE's already-π-folded factors directly: `getEigenvectors()`
   = U = `diag(1/√π)V`, `getInverseEigenvectors()` = Uinv = `Vᵀdiag(√π)`, `getEigenvalues()` = eval
   (modelmarkov.cpp:1557-1561); `tip_partial_lh` is pre-multiplied by Uinv. The GPU root/branch sum must
   **drop the explicit `freq[x]` multiply** (reintroducing it double-counts π) — K2 already validated this
   freq-fold identity to rel ~1e-12. Re-upload U/Uinv/eval per model (and per `optimizeParameters` iteration
   when rate params move); re-key echild inputs.
5. **Padding** — device buffers on `nptn_padded = roundUp(nptn,4) + tail`, `tail =
   max(nstates, unobserved_ptns)` (phylotree.cpp:912-913,1089); native-20 (no 32-pad), compatible.

**Theta coherence (highest runtime risk):** host owns `theta_computed` (set false per branch at
optimizeOneBranch phylotree.cpp:2642) and `partial_lh_computed`/`clearReversePartialLh` (phylonode.cpp:22-33).
Contract: (1) GPU derv rebuilds device theta only when host `theta_computed==false` (mirror the
phylokernelnew.h:2393 gate); (2) GPU partial re-sweep executes exactly the host `traversal_info` order
(GPU never owns topology/dirty-tracking); (3) a device-slot dirty bitmap clears in lockstep with host
`partial_lh_computed=0`. K3 graph reuse is valid only WITHIN one branch's NR steps (theta stays true); across
branches the dirtied path changes → re-capture/re-key the graph per distinct traversal pattern.

## Phased sub-plan (each gated on a number; mirrors the G.1 discipline)

- **G.2.-1** (harness, cheapest, no seam risk): add `ptn_freq` pattern weights + `nptn_padded` padding to the
  standalone harness; confirm the π-fold (no explicit freq). **Scaling port dropped** (NORM_LH, see gap 3).
  **Gate:** standalone K1 lnL on the real AA-100K fixed topology for one model (LG+G4) matches the CPU
  `computeLikelihoodBranchSIMD<Vec4d,NORM_LH,20>` oracle to rel ≤ 1e-12, no NaN; K2 FD-gate still holds.
- **G.2.0** (lnL-only behind the seam). **Confirmed lnL contract** (computeLikelihoodBranchSIMD,
  phylokernelnew.h:2660; val build :2747 `val=exp(eval·len)·prop`; reduction :2433-2455):
  `lh_ptn = |Σ_{c,x} val_c[x]·partial| + ptn_invar`, then **`tree_lh = Σ_ptn ptn_freq[ptn]·(log lh_ptn +
  buffer_scale_all[ptn])`**. For AA-100K NORM_LH: `buffer_scale_all=0`, `ptn_invar=0` (non-+I) ⇒
  `tree_lh = Σ_ptn ptn_freq[ptn]·log|lh_ptn|` — exactly the harness math (the harness summed 100000
  *uncompressed* sites at weight 1 ≡ Σ_pattern freq·log, so the math is already oracle-validated; the in-tree
  path just uses IQ-TREE's *compressed* patterns + `ptn_freq`).
  - **G.2.0a — ✅ PASS (job 170203514, 2026-06-08): clean-room cross-check de-risks the bridge BEFORE the seam.**
    `GPU lnL = CPU lnL = -7541977.778778`, **rel = 1.235e-16** (machine epsilon, ≫ tighter than the 1e-12 gate),
    no transpose needed (eigen/tip/weights/π-fold conventions all exact), NORM_LH confirmed in vivo, CPU run
    byte-unchanged. Build clean (`make exit=0`). See g1-log "✅ G.2.0a". Original plan ↓:
    add a gated
    `PhyloTree::gpuLnLCrossCheck()` (new TU `tree/phylotreegpu.cpp`, `#ifdef IQTREE_GPU`) called once after the
    first full `computeLikelihood` of a single-model `--gpu` run. It rebuilds the K1 sweep clean-room from the
    LIVE tree — `model->getEigenvalues()/getEigenvectors()(=U)/getInverseEigenvectors()(=Uinv)`,
    `site_rate->getNRate()/getRate(c)/getProp(c)`, the topology+brlen walked from `PhyloNode/PhyloNeighbor`,
    tip states + `ptn_freq` from `aln` — runs an `extern "C"` K1 launcher in `tree/gpu/`, and prints
    `GPU_lnL` vs `tree->getCurScore()`. This proves the eigen/tip/weights/π-fold bridge with ZERO fn-pointer/
    partial-buffer/TraversalInfo coupling. **Gate:** `GPU_lnL` matches the CPU lnL rel ≤ 1e-12 for LG+G4
    (−7541976.9391). (No CPU behaviour change — pure additive read-only hook.)
  - **G.2.0b — ✅ PASS (job 170205301, 2026-06-08): Branch pointer wired at the funnel, GPU≡CPU bit-identical.**
    `setLikelihoodKernelGPU()` overrides `computeLikelihoodBranchPointer` with `computeLikelihoodBranchGPU`
    (calls the extracted `gpuComputeTreeLnLCleanRoom()` helper, **mirrors `_pattern_lh`**, zeroes
    `lh_scale_factor`, CPU-fallback on NaN). Validated lnL-only under `-blfix`: 5/5 gates — install+active
    markers fired, in-process self-check (GPU vs independent CPU recompute) rel 1.235e-16, **GPU final lnL+s.e.
    == CPU bit-identical** (rel 0.0, `-7541976.8566` / s.e. `15407.1942`), CPU path unperturbed, build clean.
    Adversary caught that `computeLogLVariance` (phyloanalysis.cpp:3946) unconditionally re-reads `_pattern_lh`
    after Branch returns → the override now mirrors it (no phyloanalysis.cpp edits). Gate rejects
    `-wsl/-wpl/-alrt/-abayes/-b/-bb/-asr/dating/pll` + non-FIX/supertree/non-reversible/mixture/site-specific.
    See g1-log "✅ G.2.0b". Original plan ↓:
    once 0a passes, add `setLikelihoodKernelGPU()` overriding **only**
    Partial+Branch pointers (Derv/FromBuffer stay CPU), the narrow per-evaluate gate (`gpu &&
    !model_test_and_tree && !isSuperTree() && reversible && num_states∈{4,20} && !openmp_by_model && VRAM
    fits`), persistent GPU instance keyed on (topology,nptn,nstates,maxNCAT), re-applied at
    `initializeAllPartialLh` (phylotesting.cpp:2098). **Gate:** `iqtree --gpu -m LG+G4` prints the same lnL as
    CPU (rel ≤ 1e-12); CPU (`--gpu` off / `IQTREE_GPU=OFF`) byte-unchanged.
- **G.2.1 — ✅ COHERENCE CONTRACT VERIFIED (4-agent workflow, 2026-06-08) → correctness-first design is STATELESS.**
  The adversary found the device-mirror's #1 hole: `clearAllPartialLH` (phylotree.cpp:683, fired on every
  alpha/model-param change via rategamma.cpp:182) clears all partials + nulls `current_it` but does NOT touch
  `theta_computed` — so a theta_computed-watching GPU silently reuses stale device theta/partials. theta is also
  edge-bound (`theta_computed==true` on a *different* edge after `current_it` is re-picked). **Decisive de-risk:
  the recommended correctness-first variant carries NO device-resident state — `computeLikelihoodDervGPU` /
  `computeLikelihoodFromBufferGPU` each do a fresh clean-room sweep per call (like the G.2.0b Branch override),
  so every staleness hole is structurally impossible and the entire device-mirror coherence problem (Risk #2)
  evaporates for the correctness gate.** Cost: full sweep per NR evalAt (no theta-reuse) — add the
  theta-reuse/dirty-bitmap speedups (gated by a NEW `clearAllPartialLH` generation counter — the one host signal
  that plugs the hole) ONLY if G.2.2 misses the wall, each behind a bit-identical A/B gate. Split: **G.2.1a**
  clean-room derivative cross-check (de-risk arbitrary-edge directed partials + df/ddf sign vs IQ-TREE's
  `computeLikelihoodDerv` output, read-only) → **G.2.1b** wire Derv+FromBuffer stateless, lift `-blfix`. See
  g1-log "G.2.1 — coherence contract VERIFIED". Original plan ↓:
  add `computeLikelihoodDervGPU` (K2) + `computeLikelihoodFromBufferGPU`; override
  all four pointers co-consistently (FromBuffer MUST be GPU or the wrapper at phylotreesse.cpp:233 silently
  falls back to a full GPU traversal); device-theta rebuild gated on host `theta_computed`; device-slot dirty
  bitmap in lockstep with `clearReversePartialLh`; derv writes **un-negated** df/ddf (computeFuncDerv
  phylotree.cpp:2566 negates). Wrap within-branch NR steps in the K3 graph. **Gate:** ≥3 models (LG+G4,
  WAG+I+G4, a DNA GTR+G4) converged per-model lnL matches CPU rel ≤ 1e-9 AND optimised branch vector rel ≤ 1e-6.
  - **G.2.1a — ✅ PASS (job 170258836, 2026-06-08):** GPU single-edge df/ddf == IQ-TREE's own
    `computeLikelihoodDerv`, df rel 3.99e-12 / ddf rel 4.54e-15, sign un-negated. Read-only. See g1-log "✅ G.2.1a".
  - **G.2.1b — ✅ PASS (jobs 170259046 read-only + 170259325 full `-te`, 2026-06-08): full GPU branch-opt is
    end-to-end GPU≡CPU bit-identical.** All three pointers overridden (Branch+Derv+FromBuffer, stateless clean-room
    + persistent device-buffer pool), `-blfix` dropped, `k_leaf_eig` synthesises leaf endpoints so all edges run on
    GPU, `+I` gates out to CPU. Derivative regression: INT-INT df rel 3.99e-12/ddf 4.54e-15 + LEAF df rel
    5.93e-13/ddf 9.85e-16. Full `--gpu -te -m LG+G4`: **GPU lnL = CPU = −7541976.8530 rel=0.0, all 197 optimised
    brlen worst_rel=0.0** (bit-identical to written precision; gradient ~1e-12/1e-15, NR same optimum) — stronger
    than the rel≤1e-9/1e-6 gates. CPU byte-unchanged. **Wall: GPU 1063 s vs CPU 225 s (4.7× slower) = the stateless
    re-sweep cost this contract predicted** — but `-te` full-brlen-opt is the heaviest per-model workload, so this
    is NOT the MF slowdown (G.2.2 measures real `-m TEST`). Device-mirror coherence (Risk #2) confirmed structurally
    moot. theta-reuse + generation-counter dirty-bitmap = gated lever IF G.2.2 misses wall. See g1-log "✅ G.2.1b".
- **G.2.2** (full `-m TEST`): persistent GPU instance reused across all 224 AA / 176 DNA candidates (re-upload
  only U/Uinv/eval + cat rates/props + brlen per model); per-evaluate CPU fallback for every non-{single tree,
  reversible, bifurcating, 4/20-state, non-ASC} candidate; force serial `test()` (phylotesting.cpp:3317) under
  `--gpu` (disable `openmp_by_model`; steer single-GPU non-MPI); AIC/BIC cross-check. **Gate:** same best model
  as CPU, displayed lnL −7541976.86 rel ≤ 1e-12, identical top-model AIC/BIC ranking, **MF wall < 221.6 s**.

## Top risks
1. **Wall vs correctness** — the perf-pass deep-ladder/compute-bound finding means G.2.2 may pass lnL but land
   at wall PARITY until the intra-kernel lever lands (hence that lever is being done first).
2. **Theta/device coherence** — a missed invalidation → NR optimises against stale device state → silently
   wrong df/ddf/branch lengths/lnL, no crash. Device dirty bitmap must track host `partial_lh_computed` exactly.
3. **Gate altitude + model lifecycle** — a `params->gpu`-only gate would GPU-enable supertrees/transient
   distance/bootstrap trees and thrash/OOM the ~15 GB device buffer; need a narrow per-evaluate gate
   re-checked at `initializeAllPartialLh` + a persistent instance (a fresh IQTree is built/deleted per model;
   modelomatic flips `num_states` mid-set).
4. **ABI / re-clobber** — shims must byte-match typedefs; override must be idempotent + re-applied at the
   funnel (setLikelihoodKernel re-invoked many times).
5. **Pattern weights** — omitting `ptn_freq` yields a plausible-but-wrong lnL off by the multiplicity ratio.
6. **Unsupported classes** — gate must reject rooted/multifurcating, nonreversible, SITE_MODEL, mixlen,
   BRLEN_SCALE (flips `optimize_by_newton` false), and +ASC/+I unless those derv terms are ported (else CPU
   fallback within the same `-m TEST` run).

## Default decisions (project discipline; revisit at G.2.0)
device-mirror ownership · lnL-only (G.2.0) before branch-opt · LG+G4 as the G.2.0 gate model
(−7541976.9391, all harnesses used it) · CPU/OFF build kept byte-identical · +I/+ASC excluded from the GPU
gate → CPU fallback · serial under `--gpu`.
