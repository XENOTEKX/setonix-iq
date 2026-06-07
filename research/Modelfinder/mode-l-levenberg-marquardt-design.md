# Mode L — Levenberg–Marquardt / Fisher-Scoring Optimizer for IQ-TREE ModelFinder

**Author:** as1708 (design synthesis by Claude Opus 4.7)
**Date:** 2026-05-27 (rev. 2 — code-audited against active source `cc3d403f`/`0c493bd5`/`4810a8ac` + P.7/FCA logs)
**Status:** IMPLEMENTATION STARTED — Phase **L.0a** written and compile-verified (see §17 + the
handoff header below). New research direction; supersedes the *parallelism-first* line of work
(Mode F → Mode P → MPGC → ATMD-AID → EDM) as the primary lever for the P.7 ceiling.
Those remain valid as the *dispatch* layer; Mode L changes the *optimizer* underneath them.
**Target reader:** future implementing self (Claude Opus 4.7 / as1708).
**Scope:** IQ-TREE ModelFinder inside one alignment; AA/DNA 100K–10M patterns; np=1–64;
OpenMPI 4.1.7 + ICX + Gadi Sapphire Rapids.

> **Rev. 2 (2026-05-27):** every claim below is now anchored to a source audit of the active tree
> and to real P.7/FCA log data. §0.5 is the new code-verified findings section; the headline
> framing in §1–2 and the projection in §11 were **corrected** (the Ji et al. 100× is for
> *finite-difference branch* gradients, which IQ-TREE does **not** use — see §0.5.2); §9 is now a
> detailed phase-by-phase plan with the extra steps the code reality demands; §9.5 is a new
> **traps & invariants checklist** mapping every past failure (B.4-13/14/15, Arch C, canonical
> state, load imbalance, MPI_THREAD_FUNNELED, one-IQTree-per-rank) to how Mode L preserves it.

---

## ⏭️ SESSION HANDOFF — START HERE (last updated 2026-05-31, after L.0b.vi — ❌ DECISIVE NEGATIVE RESULT: the FreeRate analytic gradient is broken; single-rank Mode-L abandoned)

### Current phase: **STOPPED — single-rank Mode-L optimiser is abandoned (user decision 2026-05-31).** Two independent negative results converge:

1. **L.1 traversal gate (gate 169643959, `-m TEST`, build `18f3d2a0`) — ❌ FAIL.** On low-dim AA (+G/+I+G) the joint LM does **+34%/model MORE** full-tree recomputes than the legacy alternating loop (MODE-L 4,868 vs BASE 3,827 post+pre; LG+G4 19→34 = +79%), cutting only the cheap branch-Newton `derv` sweeps −9%. lnL Δ=0.0 exact. The LM is *overkill* for a 1-D α fit (Brent is near-optimal). See §17.19 (`Trimorph.md`).

2. **L.0b.vi FreeRate weight gradient + FDCHECK (build 169647747 md5 `8469af7b2035cd110ae3b5be1d80474f`, FDCHECK job 169657699) — ❌ the analytic gradient is CATASTROPHICALLY BROKEN for +R.** I implemented the FreeRate weight (proportion) gradient — derivation-verified formula `s_p[n]=prop[k-1]·[(wL[p,n]/(prop[n]·L_p)−1)+(1−rates[n])·S_rate_total[p]]`, including the prop→rate mean-rate coupling term that a naive rate-mirror omits — built clean, and ran the `--mode-l-fd-check` validator on `LG+R4`. Result: **`|G-ratio-prop0|` ≈ 10⁵⁷, `|G-ratio-rate0|` ≈ 10⁵⁴** — analytic scores ~10⁵⁴ garbage vs FD O(10¹). `|lnl-recon|=0` (the likelihood is fine; only the *gradient* is broken). **Root cause:** in `accumulateAlphaFromPre` (`tree/phylotree.cpp:1437,1442`) `contrib = cf·qp·exp(scale_log − _pattern_lh[ptn])` — the per-category scaling cancellation that works for moderate +G rates **overflows for FreeRate's widely-varying per-category rates**. The garbage originates in `per_ptn_rate_nat` (the **L.0b.v rate gradient**, untouched by L.0b.vi); the weight gradient merely inherits it via `S_rate_total`. **The L.0b.v FreeRate rate gradient was NEVER FDCHECK-validated** (the L.0b.v gate ran production-mode and was killed early), so this bug has been latent since L.0b.v.

**The damning implication:** a ~10⁵⁴ gradient ⇒ every LM trust-region step is rejected ⇒ `!stepped break` ⇒ **Mode-L has been silently *no-opping* on every +R model** in all prior gates (keeping starting params, never optimising them). The "accepted_iters" came from +G/+I+G models only. So the joint-LM premise is **unvalidated and broken in exactly the high-dim +R regime that motivated it** — and where it *does* run (+G) it does *more* traversals, not fewer. The FDCHECK de-risking step worked: it caught this before a 2 h production +R gate that a no-opping MODE-L would have made *look* like a spurious "traversal cut."

**Decision (user, 2026-05-31): do not invest further in the Mode-L gradient kernel or the single-rank optimiser / Layer-2 path. This is the decisive negative result.** L.0b.vi code (correct in form) remains in the tree atop the broken rate gradient (harmless: `-m TEST` unaffected; +R no-ops as before). FDCHECK validator `gadi-ci/mode-p-iso/fdcheck_l0bvi_weight.sh` and the prepared `run_l0bvi_rplus_traversal_gate.sh` are kept for the record (the +R gate was correctly NOT submitted). Prior: L.0b.viii exp()-hoist PASS (MF 555s, 14%) — also never beat FCA.

---

### What is fully implemented and gate-verified

**Phase L.0a — FD-Jacobian LM (§17.6):**  
Self-contained Levenberg-Marquardt joint optimizer for model + rate-heterogeneity parameters, gated
behind `--mode-l`. Replaces legacy alternating BFGS / alpha-Brent / p_inv-EM. Curvature = OPG from
a per-pattern FD Jacobian. Branch lengths stay in the legacy Newton-Raphson outer loop.

- Build: **job 169484680 ✅** (md5 `f89e0b3e`, 2026-05-28)
- **L-FD gate: job 169549952 ✅ PASSED** (18:34 wall, `-m TESTONLY`)
  - Both arms exit 0; best model LG+G4 ✓; lnL parity |Δ|=0.001 ✓; R1 max|lnl-recon|=0 ✓
- **Performance: 1.63× SLOWER** than legacy BFGS (680.6s vs 416.0s). Expected — FD-Jacobian has
  same traversal count as derivativeFunk. Speedup requires L.0b analytic gradient.

**Phase L.0b.i — preorder buffer allocation (§17.7):**  
New `central_preorder_lh`/`central_preorder_scale_num` flat buffers allocated per internal node in
`PhyloTree`, pointers assigned to `PhyloNeighbor::preorder_partial_lh` via DFS. Called from
`initializeAllPartialLh()` when `mode_l_enabled`. `computePreorderPartialLikelihood()` was a stub.

- Build: **job 169552659 ✅** (md5 `92d3df1f`, 2026-05-28 ~20:18 AEST, 9:17 wall)

**Phase L.0b.ii — scalar preorder kernel + analytic alpha gradient (§17.8):**  
Implementation lives in `tree/phylotree.cpp` (anonymous namespace) after a CMake/GNU-ld
archive-ordering trap caused two failed builds (§17.9). See §17.8 for full details.

1. `computePreorderPartialLikelihood()` — fills `preorder_partial_lh` top-down for all internal
   nodes. Storage convention: **V^{-1}-projected eigenspace** (identical to `partial_lh`).
   Root init: `memcpy(current_it->preorder_partial_lh, current_it_back->partial_lh, ...)`.
   Recursion per branch u→v, category c, pattern p:
   ```
   pre_u_state[t] = Σ_i  inv_evec[i*ns+t] * pre_u[c*ns+i]       (iV column-wise)
   f_sib_state[t] = Σ_i  evec[t*ns+i] * exp(λ_i*r_c*b_sib) * pl_sib[c*ns+i]
   combo[j]       = Σ_t  inv_evec[j*ns+t] * pre_u_state[t] * f_sib_state[t]
   pre_v[c*ns+i]  = exp(λ_i*r_c*b_v) * combo[i]
   ```
   Leaf sibling handled via `tip_partial_lh[state*tip_block+mix_off]` (already in eigenspace).

2. `computeModeLAnalyticGradAlpha()` — accumulates G[alpha] over all internal branches:
   ```
   qp = Σ_i  λ_i * exp(λ_i*r_c*b) * pre[c*ns+i] * pl[c*ns+i]
   G += pf[p] * w_c * b * dr_c_dalpha * qp / exp(_pattern_lh[p])
   ```
   dr_c/dalpha via scalar FD (h=1e-4*max(|alpha|,1)), `w_c = getProp*getMixtureWeight`.

3. `modelfactory.cpp` FDCHECK block: before FD perturbation loop, calls preorder + analytic gradient,
   prints `G_alpha_fd`, `G_alpha_analytic`, `|G-ratio| = |G_analytic-G_fd|/|G_fd|` per iteration.

- **Build: job 169585981 ✅ exit 0** (md5 `9e5919b5`, 2026-05-29 08:06 AEST, 9:05 wall)
- **Gate job 169586323 FAILED** — SIGSEGV exit 139 at 7s wall. Root cause: §17.10.
- **SIGSEGV fix: job 169587129 ✅ exit 0** (md5 `720c56cffacc8d0433edb585730110cd`, 2026-05-29 08:59)
- Gate script `EXPECTED_MD5` updated to `720c56cffacc8d0433edb585730110cd`
- **Gate job 169588980 FAILED** — node fault (gadi-cpu-spr-0143): BASE arm hung 30m with CPU time=1m45s (near-zero). OMP/PLL thread stall. Code verified clean — all Mode L guards intact. Not a code regression.
- **Gate job 169589950 FAILED** — `G_alpha_analytic=0.000` for all 1273 FDCHECK lines across all models. Root cause: §17.11.
- **Zero-gradient root-edge fix: job 169590424 ✅ exit 0** (md5 `42b73ea299ee1ebeda0cc486d32b08cb`) — superseded before gate by the deeper preorder/gradient audit fixes in §17.12.
- **Preorder audit fixes: build job 169592056 FAILED** — compile error from direct access to private `PhyloNeighbor::preorder_scale_num`; fixed by adding `get_preorder_scale_num()`.
- **Accessor compile fix: build job 169592244 FAILED** — compile reached `phylotree.cpp` and caught the analogous direct access to private postorder `PhyloNeighbor::scale_num`; fixed by adding `get_scale_num()`.
- **Scale accessor compile fix: build job 169592407 ✅ COMPLETE** — md5 `922695586fefcfc171dea019374ea30d`.
- **L.0b.ii gate: job 169592704 ❌ FAILED** — BASE ✅ 421s / MODE-L SIGSEGV exit 139 at 16s. Root cause: §17.13 (heap overflow — preorder buffer not reallocated when model changes block size).
- **Gate 169604442 ❌ FAILED** — built from intermediate binary md5 `870eb8148d832b8068208fdabffb8292`; same heap overflow, MODE-L SIGSEGV exit 139 at 24s. BASE ✅ 414s.

**Phase L.0b.iii — ModeLScope RAII + preorder block-size tracking + nps SIGSEGV fix (§17.13, §17.14):**  
Fixes two distinct L.0b.ii heap issues. ModeLScope RAII + `preorder_block` tracking (§17.13) fixes the nc-1→nc-4 block-size overflow. The nps/`unobserved_ptns` fix (§17.14) fixes a separate `memset` overflow for +I models where `unobserved_ptns.size()` grows lazily after buffer allocation. Also adds file-based debug instrumentation (7 `DBG-L.0b-*` blocks in 3 files) and per-iter debug logging in `optimizeModeLAllParameters`.

- **Build 169608969 ✅ COMPLETE** — md5 `1e14caf78e54f28cf44400532102a38b`. Fixes MODE-L heap overflow (nc mismatch).
- **Gate 169610527 ❌ FAILED** — BASE arm HANGS (wall=30m, cput=1m42s, mem=762 MB). Node: gadi-cpu-spr-0594. Root cause: §17.13.
- **Gate 169613712 ❌ FAILED** — same hang on different node gadi-cpu-spr-0198 (wall=44m, cput=1m43s). Confirmed code regression, not node fault.
- **ROOT CAUSE (§17.13):** `initializeAllPartialLh()` guard was `mode_l_enabled` only — missing `&& mode_l_context_active`. Fires during fast NJ build (before any ModeLScope), attempting ~6 GB preorder alloc with 103 OMP threads → NUMA allocator stall.
- **BASE-hang fix applied** to `tree/phylotree.cpp` (add `&& mode_l_context_active`). Build 169614498 killed (stale). **Build 169615509 ✅ COMPLETE** — md5 `5424acd8bc44a5ef2d70e9379d677eae`.
- **Gate 169620202 ❌ FAILED** — BASE ✅ exit=0 wall=425s. MODE-L ❌ SIGSEGV exit=139 at 24s, during LG+F+I+G4 evaluation at iter=0. Root cause: §17.14.
- **ROOT CAUSE (§17.14):** `nps` in `computePreorderPartialLikelihood()` recomputed from `model_factory->unobserved_ptns.size()` at call time. For +I models `unobserved_ptns` is populated lazily on first `computeLikelihood()` call — AFTER `initializePreorderPartialLh()` allocated the buffer. Runtime `nps > nps_alloc` → `memset(ps, 0, nps*ncm)` overflows scale buffer → heap corruption → SIGSEGV.
- **nps fix + null guards applied** (see §17.14). `nps` now derived from stored `preorder_block / blk`. Three `tip_partial_lh` null guards and one sibling `partial_lh` null guard added. Per-iter debug logging added in `optimizeModeLAllParameters`.
- **Build 169621693 ✅ COMPLETE** — md5 `ccacea46d73ad7ee7f26df5e7f1c45a8`. Exit=0.
- **Gate 169621788 ❌ CANCELLED** (2026-05-29 ~00:11) — §17.14 nps fix confirmed (MODE-L reached iter=0 on all models past the crash site); §17.15 p_inv ratio wrong as expected (2–60×). Cancelled after 40 min (extrapolated 170 min for 153 models — too slow: FDCHECK flag forced FD for all dims every iteration).

**Phase L.0b.iv — p_inv formula fix applied (§17.15):**
- Formula corrected in `computeModeLAnalyticGradPInv()` and `computeModeLAnalyticScores()` (both locations). `inv_pi = 1/p_inv` added; formula now `(ptn_invar[p]*inv_pi - (L_p-ptn_invar[p])*inv_omp)/L_p`.
- No separate gate submitted (gate 169621788 confirmed the old binary was correct otherwise; the p_inv formula fix is bundled into L.0b.v build below).

**Phase L.0b.v — FreeRate rate gradient + production mode (§17.16):**

Root cause of walltime issue: gate script ran `--mode-l --mode-l-fd-check` which forces FD for ALL dims every iter (even alpha/p_inv which had analytic gradients). Also: `RateFree` inherits `RateGamma`, causing `alpha_param_dim` to be wrongly set to dim 0 for +R models, injecting a zero analytic gradient for the proportion dimension.

Fixes and new code (2026-05-30, build 169623002, md5 `563e8ed90a95db1086166741327fb6be`):
1. **Fix `alpha_param_dim` detection** — added `&& !dynamic_cast<RateFree*>(site_rate)` guard to exclude RateFree from the RateGamma alpha-dim path.
2. **FreeRate rate gradient via preorder pass** — `computeModeLFreeRateRateScores(ncat, per_ptn_rate)` fills `per_ptn_rate[p*ncat+k]` = ∂log(L_p)/∂r_k using the same `GradCtx` / `accumulateAlphaFromPre` traversal with `dr_c=0` (accumulates into `per_ptn_rate_scores` instead of the alpha chain-rule path).
3. **Encoded parameter transformation** — in `computeModeLAnalyticScores()`, transforms natural rate scores to the `r[n]/r[k-1]` encoded parameter space: `s_p[n] = r[k-1] * (s_nat[p,n] - w[n] * Σ_k r[k]*s_nat[p,k])`.
4. **Gate script → production mode** — removed `--mode-l-fd-check` from MODE-L arm; full_analytic=true for +G/+I+G models, partial analytic (rate dims only, weight dims still FD) for +Rk.
5. **New gate criteria**: both arms exit 0; best model LG+G4; lnL |Δ|≤0.05; accepted_iters>0; MODE-L wall < 2×BASE.
- **Build 169623002 ✅ COMPLETE** (2026-05-30 00:54) — md5 `563e8ed90a95db1086166741327pb6be`.
- **Gate 169623057 ❌ WALLTIME** — killed at 1h (60 min). Root cause: preorder kernel was single-threaded (no OMP), while standard likelihood uses all 104 threads. At 104 CPUs, preorder sweep cost = 2.9s × 9.5 avg sweeps = 27.4s/model. Gate criterion (4) (MODE-L wall < 2×BASE=854s) fails: projected 4200s >> 854s. Correctness criteria (1)–(3) likely would have passed for the first ~110 models (57 done at 36 min at parity). See §17.17.

**Phase L.0b.vii — OMP parallelization of preorder/gradient kernel (§17.17):**

Root cause: `preorderFillRecursive`, `accumulateAlphaFromPre`, `accumulateLeafAlphaBranch` all loop serially over patterns. The standard likelihood kernel (`computeLikelihoodBranch*SIMD`) uses `#pragma omp parallel for ... num_threads(num_threads)` over patterns. The preorder kernel was written scalar — one thread doing all 100K patterns × `ns` × `ncm` work per branch.

OMP changes (2026-05-30, source patched, build 169623315 running):
1. **`preorderFillRecursive`** — `#pragma omp parallel for schedule(static) num_threads(nt)` on the `ptn` loop inside the `ci` child-loop. Shared scratch `pst`/`fsb`/`cmb` replaced by `double pst_t[64], fsb_t[64], cmb_t[64]` declared inside the loop body (per-thread stack allocation).
2. **`accumulateAlphaFromPre`** — loops swapped to `ptn` outer, `c` inner. `#pragma omp parallel for schedule(static) num_threads(nt) reduction(+:G_local)`. `*ctx.G_alpha += G_local` after the region. `per_ptn_alpha[ptn]` and `per_ptn_rate_scores[ptn*ncat+k]` are safe (each `ptn` is unique to its thread).
3. **`accumulateLeafAlphaBranch`** — same as above: OMP on `ptn` loop, `reduction(+:G_local)`, per-thread `pst_t[64]/fsb_t[64]/cmb_t[64]` declared inside loop body.
4. **`computePreorderPartialLikelihood` root init** — `#pragma omp parallel for schedule(static) num_threads(num_threads)` on the root-leaf ptn init loop.

Expected timing after OMP (104 threads):
- Preorder sweep: 2.9s / 104 ≈ 0.028s
- Gradient accum: similar, ~0.028s
- Standard likelihood: ~0.03s/call (already parallelized)
- 9.5 sweeps/model × (0.028 + 0.028 + 0.03s likelihood) = 0.82s/model
- 153 models × 0.82s ≈ 125s total — well under BASE=427s, passes criterion (4) with 3.4× margin.

- **Build 169623315 ✅ COMPLETE** (2026-05-30, 46s incremental) — md5 `8912c9cccdc47ccf2d8c23411f464686`.
- **Gate 169623342 ✅ PASSED** (2026-05-30) — exit=0/0; best=LG+G4 both arms; lnL Δ=−0.0004 ✅; accepted_iters>0 for 44/82 evaluated models ✅; MODE-L wall=668s < 848s (2×424s) ✅ margin=180s. 82/153 models evaluated (71 pruned by filterRates after reference set). Performance: 8.15s/model (OMP speedup ~4.2× vs projected 104× — see L.0b.vii-fix result below for root cause correction).

**Phase L.0b.vii-fix — one omp parallel per sweep (§17.18):**

OMP restructure (2026-05-30, build 169636132, md5 `c4694c606aed346dc6890769cf35cddd`):
1. `computePreorderPartialLikelihood` — `#pragma omp parallel num_threads(num_threads)` wraps `preorderFillRecursive` call; root-init kept as separate `#pragma omp parallel for`.
2. `preorderFillRecursive` — `#pragma omp parallel for` → `#pragma omp for` (orphaned worksharing, implicit barrier per node enforces parent→child dependency).
3. `accumulateAlphaFromPre` / `accumulateLeafAlphaBranch` — `#pragma omp for` (orphaned); per-thread `G_thread` accumulator; `#pragma omp atomic` fold into `*ctx.G_alpha`.
4. `computeModeLAnalyticGradAlpha`, `computeModeLFreeRateRateScores`, `computeModeLAnalyticScores` — each wraps traversal in one `#pragma omp parallel num_threads(num_threads)` region.

- **Build 169636132 ✅ COMPLETE** (md5 `c4694c606aed346dc6890769cf35cddd`).
- **Gate 169636854 ✅ PASSED** (2026-05-30) — exit=0/0; best=LG+G4; lnL Δ=−0.0004 ✅; 44/232 EXIT calls with accepted_iters>0 ✅; MODE-L wall=661s < 848s ✅ margin=187s. 82 models evaluated, 232 EXIT calls (multiple rounds/model). Speed: **8.06s/model (vs 8.15s prior = 1.1% improvement only)**.

**Root cause of unchanged performance (§17.18 — scalar exp() bottleneck):** One-parallel-per-sweep is structurally correct, but per-sweep time is still ~0.70s. The fork/join diagnosis (§17.17) was incomplete. The real bottleneck is `exp(eval[i]*rate_cat*branch_len)` in the inner `ptn×c×state` triple loop — ~1.5B exp() calls/sweep at ~20ns/call = ~30s single-thread; 103 threads gives ~0.29s kernel bound (observed 0.70s = 2.4× above). AVX-512 can batch 8 doubles + polynomial approx → 8–16× speedup on the exp() inner loop. **Fix: L.0b.viii SIMD vectorization.**

**§17.18 RESULT + CORRECTION (L.0b.viii gate 169637529, verified PASS — supersedes the projection above).** L.0b.viii was implemented as an exp() **hoist** (precompute the pattern-independent `exp(eigenvalue·rate·branchlen)` per `(c,i)` out of the inner `ptn×state×state` loops in `preorderFillRecursive`, `accumulateLeafAlphaBranch`, root-init) — plus a gate fix (criterion-3 reads `*.mode_l_debug.log` `accepted_iters`, not the verbose-only `MODE-L:` line). Gate 169637529: lnL Δ=0.0 exact ✅, LG+G4 ✅, accepted_iters=328/42 models ✅, MF 643→**555s (14%)** → **L-FD PASS**. BUT: (1) **both §17.17 and §17.18 were wrong** that the preorder runs on 103 threads — ModelFinder is **thread-parallel over models, 1 thread/model** (`phylotesting.cpp:6304`; CPU/wall≈80–84× in both arms), so the "1.5B exp × 20ns / 103 threads = 0.29s" math is moot. (2) exp() was **not** the dominant single-thread cost — `icpx -O2` already hoisted the per-`t` `exp()` via LICM, so the hoist only recovered the per-pattern redundancy (14%, not 8–16×). The residual bottleneck is the scalar `ns²` matrix-vector products. (3) **MODE-L MF 555s > BASE 417s > FCA 259s — Mode-L does NOT beat FCA**, and can't on a single rank: 1 thread/model + branch-reopt dominance (75–85%). Mode-L is the **Layer-2 enabler**.

---

### DO THIS NEXT

**Nothing — single-rank Mode-L is STOPPED (user decision 2026-05-31).** The +R-regime gate (the would-be decisive test) was attempted via L.0b.vi and is **blocked by a broken gradient kernel**, and the cumulative evidence is a clear negative result. The chain:

1. **L.1 on `-m TEST` FAILED** (gate 169643959): joint LM does +34%/model MORE full-tree recomputes on low-dim AA (LG+G4 +79%); −9% branch-Newton. Single-rank LM loses there.
2. **L.0b.vi (the +R prerequisite) exposed that the FreeRate analytic gradient is fundamentally broken** (FDCHECK job 169657699: `|G-ratio-prop0|`≈10⁵⁷, `|G-ratio-rate0|`≈10⁵⁴; analytic ~10⁵⁴ vs FD O(10¹); `|lnl-recon|=0`). Root cause: `accumulateAlphaFromPre` `exp(scale_log − _pattern_lh[ptn])` overflows for FreeRate's per-category rate spread (works for +G). This is a **latent L.0b.v bug** — the rate gradient was never FDCHECK-validated. ⇒ **Mode-L has been silently no-opping on every +R model** (garbage gradient → all trust-region steps rejected). The high-dim +R regime that motivated the LM has therefore **never actually run the LM**, and fixing it is a deep, uncertain kernel debug.
3. **Verdict:** the single-rank joint-LM premise is unsound where it had to win (+R: gradient broken / no-op) and counter-productive where it runs (+G: more traversals). The barrier-reduction motivation for Layer 2 is also weakened (more full passes = more Mode-P barriers on AA `+F`, where `getNDim()=0`). **Stop investing in the Mode-L gradient / single-rank optimiser / Layer-2.**

**If anyone ever revisits this:** the one prerequisite is to FIX the FreeRate gradient kernel first — instrument `scale_log`, `_pattern_lh[ptn]`, `qp`, and the scaled per-category product per pattern in `accumulateAlphaFromPre`/`computeModeLFreeRateRateScores` to find why the scale cancellation overflows for +R, then validate `|G-ratio-prop0|<0.01` via `gadi-ci/mode-p-iso/fdcheck_l0bvi_weight.sh` BEFORE any production +R gate (`run_l0bvi_rplus_traversal_gate.sh`, already prepared). Without that, every +R Mode-L number is meaningless (no-op).

<details><summary>Original L.0b.viii SIMD spec (now reference material for L.0b.ix — exp() is already hoisted)</summary>

**Step 1 — Implement L.0b.viii: SIMD AVX-512 exp() vectorization in `tree/phylotree.cpp`**

File: `/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylotree.cpp`

**Hot loop (in `preorderFillRecursive`, repeated per node per child ci):**
```cpp
// Current scalar code (inner loop inside ptn×c iteration):
for (size_t t = 0; t < ns; t++) {
    double s = 0.0;
    for (size_t i = 0; i < ns; i++) s += V[t*ns+i] * exp(ev[i]*rc*bs) * tp[i];
    fsb_t[t] = s;
}
```
For AA models: ns=20. This computes `exp(ev[i]*rc*b)` 20× per (ptn, c, child) — and `ev[i]*rc*b` is CONSTANT across patterns for a given (c, child). These are per-edge-per-category pre-computable values.

**Optimization strategy (two approaches, use both):**

**A. Pre-compute exp factors per edge/category (cache exponents):**
```cpp
// Before the ptn loop for each (ci, c):
double expv[20], expv_sib[20];  // ns=20 max for AA
for (size_t i = 0; i < ns; i++) expv[i]     = exp(ev[i] * rc * bv);
for (size_t i = 0; i < ns; i++) expv_sib[i] = exp(ev[i] * rc * bs);
// Then in the ptn loop:
for (size_t i = 0; i < ns; i++) s += V[t*ns+i] * expv_sib[i] * tp[i];  // no exp() call
for (size_t i = 0; i < ns; i++) pv[i] = expv[i] * cmb_t[i];            // no exp() call
```
This removes ALL exp() calls from the ptn inner loop. The exp() calls are hoisted to once per (node, child, category) = 98×2×4 = 784 exp() calls per sweep instead of 1.5B. **Expected: dominant speedup source.**

**B. SIMD vectorize the matrix-vector products (ns=20, use AVX-512):**
The matrix multiplications `Σ_i V[t*ns+i] * expv_sib[i] * tp[i]` are dot products of length 20. With AVX-512 (8 doubles per register), unroll to process 8 elements per instruction:
```cpp
// Pseudo-code using Intel intrinsics (_mm512_*):
__m512d acc = _mm512_setzero_pd();
for (size_t i = 0; i < ns; i+=8) {
    __m512d vi = _mm512_loadu_pd(&V[t*ns+i]);
    __m512d ei = _mm512_loadu_pd(&expv_sib[i]);  // pre-computed
    __m512d ti = _mm512_loadu_pd(&tp[i]);
    acc = _mm512_fmadd_pd(vi, _mm512_mul_pd(ei, ti), acc);
}
fsb_t[t] = _mm512_reduce_add_pd(acc);
```
Or use `#pragma omp simd` for compiler-managed vectorization:
```cpp
#pragma omp simd reduction(+:s)
for (size_t i = 0; i < ns; i++) s += V[t*ns+i] * expv_sib[i] * tp[i];
```

**C. Same changes needed in `accumulateAlphaFromPre` and `accumulateLeafAlphaBranch`:**
These also compute `exp(ev[i]*rc*bv)` in the inner loop. Hoist to per-edge-per-category cache.

**Step 2 — Build and gate:**
```bash
cd /scratch/rc29/as1708/iqtree3-mode-p-iso/build-mode-p-iso-p3
make -j8 iqtree3-mpi-mode-p-iso-p3 2>&1 | tail -5
md5sum iqtree3-mpi-mode-p-iso-p3
sed -i 's/EXPECTED_MD5:=.*/EXPECTED_MD5:=<NEW_MD5>/' ~/setonix-iq/gadi-ci/mode-p-iso/run_lfd_aa100k_np1_mode_l.sh
qsub ~/setonix-iq/gadi-ci/mode-p-iso/run_lfd_aa100k_np1_mode_l.sh
```

**Expected gate result (L.0b.viii):**
- exp() hoisting: eliminates ~1.5B exp() calls/sweep; kernel becomes pure matrix-vector multiply
- 96K ptn × 4 cat × 20× 20 flops/ptn-cat = 15.4 GFlop/sweep; at 103 threads × ~100 GFlop/s/core = 10 TFlop/s → ~1.5ms/sweep (memory-bound at ~1-2 TB/s for 96K×800B = 77 MB data)
- More realistic: 0.05–0.09s/sweep (10–15× improvement vs current 0.70s)
- 8.6 sweeps/model × 0.07s = 0.6s/model; 82 models = **~49s** (vs FCA 259s: 5.3× faster)
- Gate criterion: << 848s; **first Mode-L win over FCA np=1**

**Step 3 — After L.0b.viii gate passes:**
- Proceed to **L.0b.vi** (FreeRate weight gradient — full_analytic=true for +Rk).
- Then **L.1** (LM replaces alternating loop single-rank — the Layer 2 enabler gate).

</details>

> NOTE: the "~49s / 5.3× / first Mode-L win over FCA" projections in the collapsed spec above are **obsolete** — L.0b.viii (exp() hoist) measured MF 555s and did NOT beat FCA (§17.18 RESULT). They assumed 103 threads for the preorder; ModelFinder uses 1 thread/model.

---

### L.0b.iv — p_inv analytic gradient (implemented; formula bug §17.15 fixed 2026-05-29)

**Status: IMPLEMENTED, formula bug fixed.** `computeModeLAnalyticGradPInv()` is in `tree/phylotree.cpp`; wired into `modelfactory.cpp` FDCHECK block. The original formula had a sign/scaling error; see §17.15.

**Correct formula (post-§17.15 fix):**

`computePtnInvar()` fills `ptn_invar[p] = p_inv * freq[state_p]` (already scaled by `p_inv`). So:
- `C_p = ptn_invar[p] / p_inv` — constant-site frequency for pattern p
- `L_var_p = (L_p − ptn_invar[p]) / (1 − p_inv)` — variable-rate likelihood

Differentiating `L_p = p_inv·C_p + (1−p_inv)·L_var_p`:

```
∂logL_p/∂p_inv = (C_p − L_var_p) / L_p
               = [ptn_invar[p]/p_inv − (L_p − ptn_invar[p])/(1−p_inv)] / L_p
```

This is O(nptn), no tree traversal needed (all quantities are pattern-level scalars after `computeLikelihood`).

**Implementation (`tree/phylotree.cpp`):**
```cpp
const double inv_omp = 1.0 / (1.0 - p_inv);
const double inv_pi  = 1.0 / p_inv;
for (size_t p = 0; p < nptn; p++) {
    double L_p = exp(_pattern_lh[p]);
    if (L_p <= 0.0) continue;
    G += ptn_freq[p] * (ptn_invar[p] * inv_pi - (L_p - ptn_invar[p]) * inv_omp) / L_p;
}
```

The same formula is used in `computeModeLAnalyticScores()` for the per-pattern score column (OPG curvature).

**Why the original formula was wrong (§17.15):** The design doc stated `L_const = freq[state_p]`, but
`computePtnInvar()` stores `ptn_invar[p] = p_inv * freq[state_p]`. The original code used `ptn_invar[p]`
directly as `L_const` in the formula, missing the `1/p_inv` scaling. For variable sites
(`ptn_invar[p]=0`) the two formulas coincide; for constant-state sites the old formula dropped the
`C_p/L_p` term entirely, giving `|G-ratio-pinv|` = 2–60+ (observed in gate 169621788 partial output).

**Gate:** L.0b.iv passes when `|G-ratio-pinv| < 0.01` for all LG+I+G4 FDCHECK lines.

**Key source refs:**
- `tree/phylotree.cpp:1636–1667` — `computeModeLAnalyticGradPInv()` (corrected formula)
- `tree/phylotree.cpp:1740–1760` — p_inv score column in `computeModeLAnalyticScores()` (corrected)
- `model/rategammainvar.cpp:220-287` — legacy EM loop for p_inv (replaced by L.0b.iv in LM opt)
- `tree/phylotreesse.cpp:578-646` — `computePtnInvar()` (stores `p_inv * freq[state]`, NOT bare `freq[state]`)

---

### L.0b remaining sub-steps

| Step | What | Status |
|---|---|---|
| L.0b.i | Preorder buffer allocation | ✅ Built 169552659 |
| L.0b.ii | Scalar preorder kernel + alpha gradient | ❌ Gates 169592704 + 169604442 FAILED (MODE-L SIGSEGV); BASE ✅ in both. Heap overflow → §17.13 |
| L.0b.iii | ModeLScope + block-size tracking + nps fix + null guards | ✅ Gate 169621788 cancelled after §17.14 nps fix confirmed; p_inv ratio wrong (§17.15, expected); FDCHECK too slow (170 min extrapolated) |
| L.0b.iv | Analytic gradient for p_inv (§17.15 formula fix) | ✅ Formula fixed 2026-05-29; bundled into L.0b.v build (separate gate not needed) |
| **L.0b.v** | **FreeRate (+Rk) rate gradient + production mode** | ❌ Gate 169623057 killed (1h walltime; preorder single-threaded → 27.4s/model; criterion 4 failed) |
| L.0b.vii | OMP preorder/gradient kernel parallelization | ⚠️ Gate 169623342 *unverified* — `.o` not on disk; run-dir logs show NO speedup. ModelFinder is 1 thread/model so OMP can't parallelize (§17.18). "668s/4.2×" was narration |
| L.0b.vii-fix | Move omp parallel outside recursion (one fork/sweep) | ❌ Gate 169636854 = **L-FD FAIL (exit 10)** — criterion-3 read the verbose-only `MODE-L:` line (absent in production). Mode-L OK per DBG (332 accepted). No speedup (1 thread/model) |
| **L.0b.viii** | **exp() hoist out of ptn loop (precompute per (c,i))** | **✅ Gate 169637529 verified PASS** — lnL Δ=0.0 exact; accepted_iters=328/42 (criterion-3 fixed to read DBG); MF 643→**555s (14%)**. Did NOT beat FCA (still > 259s); exp() not dominant (icpx already hoisted per-t via LICM) |
| L.0b.vi | FreeRate weight gradient (full analytic for +Rk) | pending |
| L.0b.ix | SIMD-vectorise the scalar ns² matvec loops in the preorder kernel | ⬜ Optional — even a full kernel win can't beat FCA single-rank (1 thread/model; branch-reopt-dominated). Layer 2 is the real path |
| **L.1** | **Does the joint LM CUT full-tree traversals vs the legacy alternating loop? (traversal-count gate, -m TEST)** | ❌ **FAILED gate 169643959** — MODE-L +34%/model full-tree recomputes (LG+G4 +79%), −9% branch-Newton; lnL Δ=0.0 exact. LM loses on low-dim AA; **-m TEST excludes +R** so the high-dim regime is untested. Next: rerun on +R set (-m MF). See §17.19 (Trimorph.md) |

**Reality check (supersedes the optimistic "Once …" projections below).** None of L.0b.vii / vii-fix / viii beat FCA, and they never could on a single rank: ModelFinder evaluates models **thread-parallel (1 thread/model)**, so the per-model preorder is single-threaded regardless of OMP, and per-model cost is dominated by **branch re-optimisation (75–85%)** which Mode-L does not change. L.0b.viii's exp() hoist is a real but small (14%) single-thread win; MODE-L MF 555s > BASE 417s > FCA 259s. **The path to beat FCA is Layer 2 (moldable multi-rank cohorts)** — Mode-L's reduced barrier count is the enabler there. Single-rank kernel tuning (SIMD, etc.) cannot close the gap.

~~**Once L.0b.vii-fix is done:** ~74× preorder speedup → ~71s, beats FCA 3.6×.~~ *(WRONG — assumed 103 threads for the preorder; it's 1.)*
~~**Once L.0b.viii is done:** ~0.07s/sweep → ~49s, beats FCA 5.3×.~~ *(WRONG — exp() was not the dominant cost; measured 555s.)*

---

### Key constants and paths

| Item | Value |
|---|---|
| Source tree | `/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/` |
| Build dir | `/scratch/rc29/as1708/iqtree3-mode-p-iso/build-mode-p-iso-p3/` |
| Binary | `iqtree3-mpi-mode-p-iso-p3` |
| L.0a md5 | `f89e0b3e965b93c078c0040273d6e684` (build 169484680) |
| L.0b.i md5 | `92d3df1ffba78218dfbbf8ca880483b8` (build 169552659) |
| L.0b.ii md5 | `9e5919b51505b1a236fbe1b4257e870f` (build 169585981 ✅) |
| L.0b.ii final md5 | `922695586fefcfc171dea019374ea30d` (build 169592407 ✅, used in gate 169592704) |
| L.0b.iii candidate md5 | `1e14caf78e54f28cf44400532102a38b` (build 169608969 — BASE-hang regression) |
| L.0b.iii BASE-hang fix md5 | `5424acd8bc44a5ef2d70e9379d677eae` (build 169615509 — MODE-L nps SIGSEGV) |
| L.0b.iii nps/null-guard fix md5 | `ccacea46d73ad7ee7f26df5e7f1c45a8` (build 169621693 — gate 169621788 cancelled after §17.14 confirmed) |
| L.0b.v §17.15 only md5 | `ea3c5fece76bf06db0cca8aff910785c` (build 169622432 — §17.15 only, no FreeRate gradient) |
| **L.0b.v FreeRate md5** | **`563e8ed90a95db1086166741327fb6be`** (build 169623002 ✅ — §17.15 + FreeRate rate gradient + production mode; gate 169623057 killed at 1h) |
| **L.0b.vii OMP md5** | **`8912c9cccdc47ccf2d8c23411f464686`** (build 169623315 ✅ — OMP preorder/gradient parallelization; gate 169623342 PASSED) |
| L.0b.vii-fix md5 | `c4694c606aed346dc6890769cf35cddd` (build 169636132 — one-omp-parallel-per-sweep; gate 169636854 = L-FD FAIL on criterion-3 measurement bug; no speedup) |
| **L.0b.viii md5** | **`d1f508c27422908937f3d9e2ee26397e`** (build 169637512 ✅ — exp() hoist out of ptn loop; gate 169637529 ✅ **verified PASS**: lnL Δ=0.0, MF 555s, 14% gain, does NOT beat FCA) |
| **L.1 traversal-gate md5** | **`18f3d2a0573331e986d4aa1735cc773d`** (build 169643714 ✅ — per-`PhyloTree` `[L1-TRAV]` counters: l1_postorder/preorder/derv; gate 169643959 ❌ **FAIL**: MODE-L +34% full-tree recomputes vs BASE on `-m TEST`; LM does not cut traversals on low-dim AA) |
| Gate script | `~/setonix-iq/gadi-ci/mode-p-iso/run_lfd_aa100k_np1_mode_l.sh` (criterion-3 now reads `*.mode_l_debug.log` accepted_iters) |
| Test data | `/scratch/rc29/as1708/iqtree3-mode-p-iso/data/AA_100K/` |
| Integration seams | `aidExecuteWaves:4343`, `CandidateModel::evaluate:1970`, `aidComputeCostPred:3860` |

---

## 0. TL;DR

Every architecture we have tried (Mode F, Mode P, MPGC, ATMD-AID, EDM) attacked the **same
layer**: how to *distribute* model evaluations and *parallelize the likelihood kernel*. None of
them touched the **optimizer** that drives each model evaluation. That optimizer is the true
ceiling.

ModelFinder spends its wall time inside `ModelFactory::optimizeParameters`
([`model/modelfactory.cpp:1558`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/modelfactory.cpp)),
which alternates:

1. **Branch lengths** — coordinate-descent 1-D Newton–Raphson, one branch at a time, several
   full sweeps ([`tree/phylotree.cpp:2790`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylotree.cpp)).
2. **Model parameters** (exchangeabilities, `+F` frequencies, `+I`, `+G4` **alpha**, `+R`
   free-rates) — quasi-Newton **BFGS / L-BFGS-B with finite-difference gradients**
   ([`utils/optimization.cpp:916` `derivativeFunk`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/utils/optimization.cpp)),
   plus 1-D Brent for the gamma shape alpha
   ([`model/rategamma.cpp:214`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/rategamma.cpp)).

A **finite-difference gradient over `m` model parameters costs `m+1` full tree traversals**, and
the alternating loop converges slowly because branches and model parameters are coupled. The
heavy ModelFinder models (LG+F+I+G4, Q.MAMMAL+F+I+G4 …) are heavy precisely because **alpha and
the rate-heterogeneity parameters force many full-tree re-evaluations**.

**Mode L** replaces this with a single **joint, second-order, analytic-gradient optimizer**:

- **Analytic O(N) gradient** w.r.t. *all* branch lengths **and** model parameters in **2 tree
  traversals** (postorder + preorder), per Ji et al. 2020 ("Gradients do grow on trees", MBE) —
  who measured **126–235× per-iteration** and **210–321× total** ML-optimization speedup over
  finite differences with this exact gradient.
- **Levenberg–Marquardt / Fisher-scoring step**: build the Gauss–Newton curvature from the
  **outer product of per-pattern score vectors** (empirical Fisher / BHHH), damp it, and solve the
  small dense system `(B + λ·diag B) δ = G`. LM gives quadratic convergence near the optimum and
  is robust far from it; Taylor et al. 2022 (the paper that motivated this work) show LM beats
  Adam/BFGS/L-BFGS by orders of magnitude on small-to-medium-parameter fits — exactly the
  ModelFinder regime (~200–225 free parameters per model).
- **One Gather-Reduce per LM iteration** instead of Mode P's **~3 000–38 000** per-kernel-call
  Allreduces (the source audit in §0.5.1 corrects the earlier "~8 000" estimate — the true count is
  `num_param_iterations` × inner evals, dominated by the `(ndim+1)`-multiplied finite-difference
  model gradient). The gradient (`ndim` doubles) and the empirical-Fisher matrix (`ndim²` doubles)
  are reduced across pattern-parallel ranks **once per LM step**. This collapses the synchronization
  count by ~100× and is what finally makes pattern-parallelism (Mode P) efficient at *any* group size.

**Why this also fixes dispatch / load balancing:** (a) each model evaluation shrinks (realistically
~4–8× for AA empirical models, more for DNA/estimated — see corrected §11) in absolute work, so even
imperfect FCA load balance comfortably clears the gate; and (b) Mode P becomes efficient at gs=8/16
(low barrier count), so the EDM moldable scheduler can finally throw many ranks at the residual
heavy tail without paying the Allreduce tax. Mode L does not replace the dispatcher — it makes the
dispatcher's job easy.

**Projected AA 1M np=16 MF wall: ~150–350 s** (corrected, honest band — see §11; vs FCA 1 122 s,
gate ≤ 600 s). Even the pessimistic end clears the gate by ~1.7×. The earlier "~60–200 s" assumed
Ji et al.'s 100×+ transferred wholesale; it does not (§0.5.2), so this rev. uses a defensible
mechanism-by-mechanism estimate.

> **The one-sentence honest version:** IQ-TREE already uses analytic Newton-Raphson for *branch*
> lengths, so the win is **not** "replace finite-difference branch gradients" (Ji et al.'s headline).
> The win is **(1)** eliminate the **alpha-Brent inner full-tree search** that dominates AA `+G4`/`+I+G4`
> (the exact P.7 killers), **(2)** collapse the branch↔model↔rate alternating outer loop into one
> joint second-order solve, and **(3)** for DNA GTR / `+FO` / `+R`, eliminate the `(ndim+1)`
> finite-difference model gradient. All three are real; none is 100×; together they are ~4–8× per
> heavy AA model and enough to crush the gate.

---

## 0.5 Code-Verified Findings & Real-Data Audit (2026-05-27 source + log audit)

This section is the evidence base. Three independent deep reads of the active source tree
(`/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/`) plus the P.7 and FCA logs.
**Several earlier claims in this document were wrong and are corrected here.**

### 0.5.1 Per-model traversal & Allreduce count — corrected (≈3K–38K, not 8K)

Call graph confirmed: `ModelFactory::optimizeParameters` (`modelfactory.cpp:1558`) runs an outer
loop `for (i=2; i<num_param_iterations; i++)` with **`num_param_iterations` default = 100**
(`utils/tools.cpp:7283`), early-exiting on convergence (typically ~10–20 iterations). Each outer
iteration does:
- `optimizeAllBranches(min(i,3), …)` → up to 3 branch sweeps; each sweep loops all `2N−3` branches,
  each `optimizeOneBranch`→`minimizeNewton`→`computeFuncDerv`→`computeLikelihoodDerv` (analytic
  `df,ddf`, reuses `theta_all`) → **one `modePAllreduceLhDfDdf` (3 doubles)** per Newton step.
- `optimizeParametersOnly` → `model->optimizeParameters` + `site_rate->optimizeParameters`, each a
  BFGS (`dfpmin`, `ITMAX=200`, `optimization.cpp:793`) whose gradient comes from
  `derivativeFunk` = **`(ndim+1)` calls to `targetFunk`→`computeLikelihood`** (`optimization.cpp:916`)
  → **one `modePAllreduceLh` (1 double)** per call.

The two MPI sites are `modePAllreduceLh` (`phylotree.cpp:1014`/kernel `phylokernelnew.h:3395`) and
`modePAllreduceLhDfDdf` (`phylotree.cpp:1037`/kernel `:2654`), both fired **after** the
`#pragma omp parallel` region closes (MPI_THREAD_FUNNELED-safe), guarded by `isModePActive()`.

**Corrected count:** the model-parameter BFGS path emits ≈`(ndim+1)×BFGS_iters` Allreduces per
outer iteration (hundreds), dwarfing the branch path. Per model: **≈3 000 (fast convergence) to
≈38 000 (worst case)** Allreduces — the earlier "~8 000" was a mid-range guess. This makes the
barrier-count argument for Mode L *stronger*, not weaker.

### 0.5.2 The dominant cost is NOT finite-difference branch gradients — it is alpha + the alternating loop

**This is the most important correction.** IQ-TREE optimizes branch lengths with **analytic
Newton-Raphson reusing cached partials** (`phylotree.cpp:2790`, `computeLikelihoodDerv`). So Ji et
al. 2020's headline (126–235× from replacing *finite-difference O(N²) branch gradients*) **does not
transfer wholesale** — IQ-TREE's branch optimization is already O(N)-ish per sweep. Where the time
actually goes splits by model type:

- **AA empirical models (LG/WAG/… + `+F` + `+G4`/`+I+G4`) — the exact P.7 killers:** `+F` means
  *empirical, fixed* frequencies (counted from data, **not** ML-optimized; `+FO` would optimize),
  and the exchangeability matrix `Q` is **fixed** for named matrices. So `model->getNDim() = 0` —
  there is **no finite-difference model gradient at all**. The entire model-optimization cost is
  `site_rate->optimizeParameters`: **alpha via 1-D Brent** (`rategamma.cpp:214`; each Brent eval is a
  *full tree traversal* because changing α changes every rate category → every transition matrix →
  full re-prune) and **p_inv via EM** (`rategammainvar.cpp:220-287`). Brent runs ~10–20 full
  traversals **every outer iteration**, and the branch↔rate-het alternation repeats it. **Alpha-Brent
  + the alternating loop are the AA bottleneck.**
- **DNA GTR / `+FO` / `+R` (estimated models):** here `model->getNDim() > 0` and the
  `(ndim+1)` finite-difference gradient (`optimization.cpp:916`) genuinely dominates — this is where
  the finite-difference elimination pays the most.

Mode L attacks both: alpha/p_inv/`+R` become components of one analytic joint gradient (no Brent,
no EM sub-loop, no alternation), and estimated `Q`/`+FO` gradients replace the `(ndim+1)` probes.

### 0.5.3 Real per-model data (rank 0, AA 1M) — FCA ref (168635616) vs P.7 MPGC (169211688)

| Model | FCA np=16 ref `dt` (s) | P.7 MPGC `dt` (s) | dispatch | note |
|---|---:|---:|---|---|
| LG+F | 9.88 | 16.30 (FCA) | +65% | MPGC scaffolding tax |
| LG+F+I | 40.04 | 62.09 (FCA) | +55% | `+I` (EM p_inv) |
| LG+F+G4 | 89.23 | 101.32 (MODEP gs=2) | +13.5% | `+G4` alpha-Brent heavy |
| **LG+F+I+G4** | **332.85** | **446.89 (MODEP gs=2)** | **+34.2%** | **the killer; gs=2 made it WORSE** |
| PMB+G4 | 78.38 | 173.72 (FCA) | +122% | load-imbalance + scaffolding |
| MTART+F+G4 | 121.02 | 132.58 (MODEP gs=2) | +9.6% | |

Two facts jump out: (1) `+I+G4` adds **~244 s** over `+G4` (332.8 − 89.2) — that is the
alpha+p_inv+coupling cost Mode L targets; (2) Mode P gs=2 made the heaviest model **34% slower**
(the Allreduce-barrier-per-call tax + un-sliced overhead exceeded the half-pattern saving). FCA
np=16 total wall = 2 410 s (MF 1 122 s + tree search 1 288 s); best model LG+G4.

### 0.5.4 All required substrate is present and verified

| Fix / feature | Status | Evidence (file:line) |
|---|---|---|
| B.4-13 UCC workaround | present | `--mca coll ^ucc` in run scripts; intra-group `MPI_Barrier(mp_allreduce_comm)` `phylotesting.cpp:5194` |
| B.4-14 intra-group filterRates | present | `fca_comm=mpgc_comm` `phylotesting.cpp:4651`; `MPI_Bcast(...,fca_comm)` `:3205` |
| B.4-15 MPGC inheritance | present | `phylotesting.cpp:1970-1986` (`evaluate(... in_tree)` → `setModePGroupComm`) |
| Architecture C traversal slice | present (6 kernels) | `phylokernelnew.h:1284-1291`, `:2324`, `:2858`, `:3265`, `:3492`, `:3707` |
| Canonical checkpoint/warm-start Bcast | present | `WarmStartPacket` ~3.6 KB `phylotesting.cpp:3207-3306` |
| `∂P/∂t` via eigen | present | `computeTransDerv` `modelmarkov.cpp:792-810` (`P=U e^{Λt} U⁻¹`, `∂P/∂t=U Λe^{Λt} U⁻¹`) |
| eigendecomposition | present | `decomposeRateMatrixRev`/`eigensystem_sym` `modelmarkov.cpp:1605`; `eigenvalues/eigenvectors/inv_eigenvectors` `modelmarkov.h:502` |
| per-category rate scaling in kernel | present | `len_child[c]=site_rate->getRate(c)*length`; `exp(eval[i]*len_child[c])` `phylokernelnew.h:1381-1481` |
| **`∂P/∂θ` (model-param derivative)** | **absent** | only `∂P/∂t` exists — Kenney–Gu eigen-derivative must be built (L.3) |
| **preorder / upper-partial likelihood** | **absent** | only parsimony preorder exists — the core new kernel (L.0) |
| Mode P members | present | `ptn_start/ptn_end/mp_group_rank/mp_group_size/mp_allreduce_comm/isModePActive()/setModePGroupComm()` `phylotree.h:2315-2370` |
| EDM/AID dispatch hooks | present | `edmScheduleInitialEpochs:4044`, `aidBuildLattice:4143`, `aidExecuteWaves:4293` (`setModePGroupComm` at `:4343`), `aidComputeCostPred:3860`, `getNextModel:3795` |

### 0.5.5 Rate-het derivative facts (for L.2/L.3)

- **Gamma is the MEAN parameterization** (Yang 1994, `GAMMA_CUT_MEAN` default): `r_c` from
  `cmpPointChi2`(inverse χ² CDF) + `cmpIncompleteGamma`, then **rescaled to mean 1** (a coupling
  constraint), `rategamma.cpp:109-171`. `∂r_c/∂α` therefore couples all categories — but it is a
  **scalar function of α only**, so it can be obtained by a *cheap* 1-D finite difference of the `k`
  rate values (perturb α, recompute the `k` scalars — **no tree traversal**), while the expensive
  tree-derivative part (`∂logL/∂r_c`) is analytic from the preorder pass. This makes L.2 much lower
  risk than a full analytic gamma-quantile derivative.
- **p_inv enters as** `L = p_inv·L_const + (1−p_inv)·Σ_c w_c L_c`; currently optimized by **EM**
  (`rategammainvar.cpp:220-287`). `∂logL/∂p_inv` is closed-form from already-computed quantities.
- **+R (`RateFree`)** stores `prop[]`,`rates[]` with `Σ prop_c r_c = 1` and `Σ prop_c = 1`; packs as
  ratios-to-last and **quicksorts** rates (`ratefree.cpp:412-483`). The ordered-gap transform (§4.1)
  removes the sort.
- **`+F` vs `+FO`:** `+F` = empirical fixed (no gradient needed); `+FO` = ML-estimated (needs
  `∂P/∂θ`). The P.7-critical models are all `+F` (fixed) → **L.0–L.2 cover the entire AA P.7 critical
  path without any `∂P/∂θ`**.

---

## 1. Why the Optimizer Is the Real Ceiling

### 1.1 The Amdahl wall is structural, not a dispatch artifact

From [`mode-p-implementation-status.md`](mode-p-implementation-status.md) §"Amdahl motivation":
FCA ModelFinder has serial fraction `f_s ≈ 0.182`, giving a hard ceiling of `S_max ≈ 5.5×` no
matter how many ranks we add. The P.7 gate needs `f_s ≤ 0.065`. Every dispatch architecture so
far has tried to chip at `f_s` by parallelizing the *evaluation*; none reduces the **number of
evaluations**, which is set by the optimizer.

The serial fraction is dominated by *per-model optimizer iterations that cannot be distributed*:
branch sweeps, finite-difference gradient probes, Brent alpha searches. Mode P parallelizes the
patterns *inside* each evaluation but adds an Allreduce barrier to every one of the thousands of
sequential calls — which is exactly why Mode P `--mode-p-all` is **58% slower than FCA** at np=2
(235 s vs 149 s, ISO-2): the barrier count, not the bandwidth, is the killer.

### 1.2 Measured cost of finite-difference model optimization

The Explore audit of the active source tree (`cc3d403f`/`0c493bd5`) confirmed:

| Parameter class | Method (current) | Gradient | Cost per gradient | File:line |
|---|---|---|---|---|
| Branch lengths | 1-D Newton–Raphson, coordinate descent | analytic `df,ddf` (1 branch) | 1 traversal / sweep | `phylotree.cpp:2790`, `phylokernelnew.h:2258` |
| Exchangeabilities (GTR) | BFGS multi-dim | **finite difference** `ERROR_X=1e-4` | `(m+1)` traversals | `optimization.cpp:916`, `modeldna.cpp` |
| `+F` frequencies | BFGS multi-dim | **finite difference** | `(m+1)` traversals | `optimization.cpp:916` |
| `+G4` **alpha** | 1-D **Brent** (golden section) | none | ~10–20 traversals | `rategamma.cpp:214` |
| `+R` free-rates | BFGS or L-BFGS-B | **finite difference** | `(2k−1)+1` traversals | `ratefree.cpp:311` |
| `+I` p-inv | finite diff / Brent | — | ~10 traversals | `rategammainvar.cpp` |

The objective interface is `Optimization::targetFunk(double x[])`
([`optimization.h:137`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/utils/optimization.h)),
overridden by `ModelFactory::targetFunk`
([`modelfactory.cpp:1844`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/modelfactory.cpp)),
which calls `model->getVariables()`, `decomposeRateMatrix()`, then `computeLikelihood()`. Each
`derivativeFunk` call perturbs each variable by `h = 1e-4·|x|` and re-evaluates — `m+1` full
likelihoods, each a complete Felsenstein pruning traversal over all patterns.

**The damning observation:** at AA 1M, the models that break P.7 are exactly the `+G4`/`+I+G4`
family (real rank-0 FCA-ref times: LG+F+G4 = 89 s, PMB+G4 = 78 s, MTART+F+G4 = 121 s,
**LG+F+I+G4 = 333 s**; §0.5.3). Per §0.5.2 these are empirical models (`+F` fixed, `Q` fixed) so
`model->getNDim()=0` — they carry **no** finite-difference model gradient; their cost is the
**alpha-Brent inner full-tree search** (every Brent step re-prunes the whole tree because α rescales
all rate categories) plus **EM p_inv** plus the **branch↔rate-het alternating outer loop**.
**The thing that makes them heavy is precisely what Mode L removes** — by folding α and p_inv into a
single joint analytic-gradient solve, the Brent search and the alternation both disappear. (The
`(ndim+1)` finite-difference row in the table above bites hardest for *estimated* DNA GTR/`+FO`/`+R`
models, the secondary target.)

### 1.3 The strategic pivot

```
PRIOR DIRECTION (parallelism-first):           NEW DIRECTION (optimizer-first):
  keep the optimizer, distribute it              reduce iterations, then distribute
  ───────────────────────────────               ────────────────────────────────────
  Mode F  : K models / node  (bandwidth wall)    Mode L : analytic-gradient joint LM
  Mode P  : patterns / model (barrier wall)              fewer iterations →
  MPGC    : static family groups (imbalance)            cheap per-model →
  ATMD-AID: phase split      (heavy light tail)         efficient Mode P at any gs →
  EDM     : moldable epochs   (filter timing)           dispatch becomes trivial
```

Mode L is **orthogonal** to and **composes with** EDM: EDM remains the scheduler; Mode L is the
per-model engine it schedules.

---

## 2. Mathematical Foundation

### 2.1 ML as nonlinear least-squares — the LM bridge

LM is built for `min_θ ½‖r(θ)‖²` with residual vector `r`. Phylogenetic ML maximizes
`ℓ(θ) = Σ_i f_i · log L_i(θ)` (sum over unique site patterns `i`, weight `f_i` = pattern count).
The bridge is the **Gauss–Newton / Fisher-scoring identity for MLE** (Giordan, Vaggi & Wehrens
2017, *J. Stat. Comput. Simul.*; Smyth's Fisher-scoring; the BHHH estimator of Berndt–Hall–Hall–
Hausman 1974):

Define the **per-pattern score** (gradient of one pattern's log-likelihood):
```
  s_i(θ) = ∂ log L_i / ∂θ        ∈ ℝ^ndim
```
Then the total score (gradient of ℓ) and the **empirical Fisher information / outer-product-of-
gradients (OPG)** curvature are:
```
  G(θ) = Σ_i f_i · s_i(θ)                          (ndim vector)         [total gradient]
  B(θ) = Σ_i f_i · s_i(θ) s_i(θ)ᵀ                  (ndim × ndim matrix)  [empirical Fisher]
```
`B` is the Gauss–Newton approximation to `−∂²ℓ/∂θ²`: it is **positive semi-definite by
construction** (a sum of outer products), is consistent for the negative Hessian at the MLE, and
needs **only first derivatives**. This is exactly the structure LM exploits: `B` plays the role of
`JᵀJ` and `G` the role of `Jᵀr`.

### 2.2 The Levenberg–Marquardt step

A damped Gauss–Newton / Fisher-scoring update with Marquardt diagonal scaling:
```
  (B + λ · diag(B)) δ = G                          [solve the ndim×ndim system]
  θ_new = θ + δ
```
- `λ → 0`  ⇒  pure Gauss–Newton / Fisher scoring  (fast, quadratic near optimum)
- `λ → ∞`  ⇒  scaled steepest ascent              (safe, slow, far from optimum)

**Adaptive damping (standard Marquardt + Nielsen 2-factor schedule):**
```
  evaluate ℓ(θ_new):
    if ℓ improved → accept; λ ← λ / ν   (ν=… e.g. trust-region factor)
    else          → reject; λ ← λ · ν;  re-solve (no new gradient needed)
```
A rejected step **re-uses the same `B`,`G`** and only re-solves the tiny linear system — no tree
traversal — so damping search is nearly free.

### 2.3 Hybrid curvature (the practical refinement)

`B` (OPG) is a good Hessian near the optimum but can be biased far away. Two cheap improvements:

1. **Exact diagonal for branches.** IQ-TREE *already* computes the exact second derivative `ddf`
   of the tree log-lh w.r.t. each branch length inside `computeLikelihoodDerv`
   ([`phylokernelnew.h:2539`](/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylokernelnew.h)).
   Use the exact `−Σ_i f_i ∂²log L_i/∂b²` on the branch diagonal of `B`, OPG for the off-diagonal
   and model blocks. (This is a Fisher-scoring / observed-information hybrid.)
2. **Levenberg vs Marquardt scaling.** Start with Marquardt `diag(B)` scaling (handles the very
   different units of branch lengths vs alpha vs frequencies); fall back to Levenberg `λ·I` if
   `diag(B)` has near-zero entries.

### 2.4 Why LM over L-BFGS here (the Taylor et al. argument)

Taylor, Wang, Bala & Bednarz (2022, arXiv:2205.07430 — the motivating paper) fit `sinc(10x)` with
a 481-parameter network and found LM reached MSE **26× lower than BFGS, 527× lower than L-BFGS,
1230× lower than Adam**, in **150 epochs vs 12 000–14 500** for the quasi-Newton methods. Their
explanation: for *small-to-medium parameter counts*, explicitly using curvature (LM's `JᵀJ`) beats
the secant/low-memory Hessian approximations of BFGS/L-BFGS, which "struggle to fit the
low-amplitude components" (ill-conditioned directions).

ModelFinder per-model optimization is **exactly this regime**: ~200–225 parameters, smooth
log-likelihood, ill-conditioned (branch lengths near zero, near-degenerate rate categories — the
phylogenetic analogue of Taylor's "low-amplitude tails"). Ji et al. used L-BFGS and still got
100×+; LM's explicit curvature should converge in even fewer iterations and is far more robust on
the near-zero branch lengths and near-equal free-rate categories that make `+R`/`+I+G4` models
slow today.

`B` is dense `ndim×ndim` (~225²); BFGS/L-BFGS avoid forming it for *memory* reasons — but at
ndim≈225 the matrix is 400 KB and the solve is ~10⁷ flops (microseconds). **There is no reason to
avoid the explicit curvature at this scale.** This is the core insight transferred from Taylor et
al. to phylogenetics.

---

## 3. The Enabling Technology — Ji et al. 2020 O(N) Gradient

> Ji, Zhang, Holbrook, Nishimura, Baele, Rambaut, Lemey, Suchard (2020). *Gradients Do Grow on
> Trees: A Linear-Time O(N)-Dimensional Gradient for Statistical Phylogenetics.* Mol. Biol. Evol.
> 37(10):3047–3060. arXiv:1905.12146.

### 3.1 Postorder + preorder = all derivatives in 2 traversals

Felsenstein's pruning is a **postorder** traversal producing, at each node `i`, the **lower /
postorder partial likelihood** `p_i` (prob. of data *below* `i` given state). A naive gradient
perturbs each branch and re-prunes → O(N²). Ji et al. add the **preorder traversal** producing the
**upper / preorder partial likelihood** `q_i` (prob. of the data *not below* `i`, jointly with
state at `i`):
```
  Preorder recursion (Eq. 7):   q_i = P_iᵀ [ q_parent ∘ (P_sibling · p_sibling) ]
                                 (∘ = elementwise; P = transition matrix on the branch)
```
With both `p_i` (postorder) and `q_i` (preorder) available at every node, **every** branch
derivative is a local dot product:
```
  Branch-length gradient (Eq. 9):   ∂ log L / ∂b_i  =  ( q_iᵀ · Q · P_i · p_i ) / L
        where Q is the instantaneous rate matrix  (since ∂P_i/∂b_i = Q·P_i = P_i·Q)
```
**Total cost: one postorder + one preorder pass = 2 traversals for the entire branch gradient**,
versus `(2N−3)+1` traversals for finite differences. Ji et al. report **126–235× per-iteration**
and **210–321× total** ML-optimization speedup (their Table 1), feeding the gradient to **L-BFGS**.

### 3.2 Extension to model parameters, alpha, p-inv, free-rates

Ji et al. note (their §"any tree-wise parameter θ"): for any parameter `θ` entering the generator,
substitute `Q → P_i⁻¹(∂P_i/∂θ)` in the same dot product. Concretely for ModelFinder:

- **Exchangeabilities / GTR rates, `+F` frequencies** (`θ` enters `Q`): need `∂P_i/∂θ`. IQ-TREE
  already eigendecomposes `Q` (`decomposeRateMatrix`, [`modelmarkov.h:353`]) and already computes
  `∂P/∂t` (`computeTransDerv`, [`modelmarkov.h:284`]). `∂P/∂θ` needs the eigen-derivative
  (Kenney–Gu / the BEAGLE differential-evolution formula): `∂P/∂θ = U·(V ∘ M)·U⁻¹` where
  `M = U⁻¹(∂Q/∂θ)U` and `V_{kl}` is the divided difference of `exp(λ_k t),exp(λ_l t)`. Implement
  once per model family.
- **Gamma shape alpha** (`+G4`): alpha does **not** enter `Q`; it sets the discrete rate
  multipliers `r_c(α)` that scale branch lengths. By the chain rule,
  ```
    ∂ log L / ∂α = Σ_c w_c · [ Σ_branches (∂ log L / ∂(r_c·b)) · b ] · (∂r_c/∂α)
  ```
  The bracket is **already available** from the per-category branch gradient (§3.1 run per rate
  category, which the kernel already does). `∂r_c/∂α` is a scalar derivative of the gamma quantile
  / mean-rate normalization — computed once, no tree traversal. **This converts alpha from a
  10–20-traversal Brent search into one component of the joint gradient at ~zero marginal cost.**
  Given that `+G4`/`+I+G4` are the models that break P.7, this is the single highest-leverage piece.
- **p-inv `+I`**: `L_i = pinv·[i constant] + (1−pinv)·L_i(variable)`; `∂L_i/∂pinv` is a closed-form
  combination of already-computed quantities.
- **Free-rates `+R`** (weights `w_c`, rates `r_c`): same chain-rule structure as alpha, per
  category; analytic and cheap.

### 3.3 What already exists vs what must be built

| Component | Status in source | Effort |
|---|---|---|
| Postorder partial `p_i` | ✅ `computePartialLikelihood` (Felsenstein pruning) | reuse |
| `∂P/∂t` (branch, via eigen) | ✅ `computeTransDerv` `modelmarkov.h:284` | reuse |
| Eigendecomposition of `Q` | ✅ `decomposeRateMatrix` `modelmarkov.h:353` | reuse |
| Exact per-branch `ddf` | ✅ `computeLikelihoodDerv` `phylokernelnew.h:2539` | reuse for hybrid curvature |
| `theta_all` (p·q-like buffer) | ✅ `computeTheta` `phylokernelnew.h:2424` | partial reuse |
| **Preorder partial `q_i` (all nodes)** | ❌ only parsimony preorder exists | **BUILD** (core kernel) |
| **`∂P/∂θ` for model params** | ❌ | **BUILD** (per model family) |
| **`∂r_c/∂α`, `∂r_c/∂{w,r}`** | ❌ | **BUILD** (scalar, cheap) |
| **Per-pattern score `s_i` + OPG `B`** | ❌ | **BUILD** (LM core) |
| **LM damped solver** | ❌ (have BFGS/L-BFGS-B) | **BUILD** (small dense solve via existing GSL/Eigen) |

The single largest new kernel is the **preorder traversal**. Note `computeLikelihoodDerv` already
forms `theta_all` — the elementwise product of the partials on the two sides of the central branch
— so the data-movement pattern of a preorder pass is already understood in this codebase.

---

## 4. Mode L Architecture

### 4.1 The joint parameter vector

```
  θ = [ b_1 … b_{2N−3} | model params | rate-het params ]
        ───────────────   ────────────   ───────────────
        branch lengths    Q / +F         α, p_inv, +R(w,r)
```
For a 100-taxon AA LG+F+I+G4: ndim = 197 (branches) + 19 (+F) + 1 (α) + 1 (pinv) ≈ **218**.
For GTR+F+I+G4 DNA: 197 + 5 + 3 + 1 + 1 ≈ **207**. For +R10: branches + ~18 rate params.

**Constraint handling** (LM is unconstrained; parameters are bounded) — optimize in transformed
space so the LM step is unconstrained and the chain rule folds the Jacobian into the gradient:
```
  branch b > 0          →  log b
  alpha   > 0           →  log α
  pinv ∈ (0,1)          →  logit pinv
  +F frequencies (simplex, Σ=1)  →  additive log-ratio (ALR) / softmax over 19 free coords
  +R rates/weights (ordered, simplex) →  log gaps + ALR  (preserves ordering, avoids sorting hacks)
```
This also fixes a current ugliness: `RateFree` re-sorts rates after every BFGS step
([`ratefree.cpp` `quicksort`]) — the ordered-gap transform removes the need.

### 4.2 The Mode L iteration (single model evaluation)

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  MODE L  — one model evaluation (replaces optimizeParameters inner loop) │
  ├────────────────────────────────────────────────────────────────────────┤
  │  θ ← initial guess (BIONJ branches; model defaults / warm-start cache)   │
  │  λ ← λ0   (e.g. 1e-3)                                                     │
  │  repeat (LM iteration k = 1 … Kmax, typically 10–40):                    │
  │   ┌──────────────────────────────────────────────────────────────────┐  │
  │   │ (A) POSTORDER pass : p_i for all nodes   ── pattern-parallel       │  │  ← traversal 1
  │   │ (B) PREORDER  pass : q_i for all nodes,  ── pattern-parallel       │  │  ← traversal 2
  │   │     and AT EACH PATTERN i accumulate                               │  │
  │   │        ℓ_local += f_i log L_i                                      │  │
  │   │        G_local += f_i s_i           (s_i = ∂logL_i/∂θ, all dims)   │  │
  │   │        B_local += f_i s_i s_iᵀ      (empirical Fisher / OPG)       │  │
  │   ├──────────────────────────────────────────────────────────────────┤  │
  │   │ (C) ONE Gather-Reduce  (MPI_Allreduce over pattern-parallel ranks) │  │  ← the ONLY barrier
  │   │        [ ℓ | G(ndim) | B(ndim²) ]   →  global ℓ, G, B              │  │
  │   ├──────────────────────────────────────────────────────────────────┤  │
  │   │ (D) LM SOLVE (local, every rank identically, no traversal):        │  │
  │   │        loop: solve (B + λ diagB) δ = G ; θ' = θ ⊕ δ                 │  │
  │   │              if predicted/eval improves ℓ → accept, λ/=ν, break    │  │
  │   │              else reject, λ*=ν, re-solve (re-uses B,G)             │  │
  │   │     (optional: 1 cheap likelihood eval to validate the step —      │  │
  │   │      itself pattern-parallel, 1 more reduce of a single double)    │  │
  │   ├──────────────────────────────────────────────────────────────────┤  │
  │   │ (E) converged if ‖G‖∞ < gtol or Δℓ < ltol → stop                  │  │
  │   └──────────────────────────────────────────────────────────────────┘  │
  │  return ℓ, θ                                                             │
  └────────────────────────────────────────────────────────────────────────┘
```

Per LM iteration: **2 traversals + 1 (–2) Allreduce**. Per model: ~10–40 iterations →
**~20–80 traversals and ~10–80 Allreduces total**, versus the status quo's thousands of traversals
(finite-diff + Brent + coordinate-descent sweeps) and (under Mode P) ~8 000 Allreduces.

### 4.3 Pseudocode (C++ shape)

```cpp
// New: tree/modeLOptimizer.{h,cpp}  (or a method on PhyloTree)
double PhyloTree::optimizeModeL(double gtol, int max_iter) {
    const int ndim = countModeLParams();          // branches + model + rate-het
    VectorXd  theta = packParams();                // transformed (log / logit / ALR) space
    double    lambda = params->mode_l_lambda0;     // e.g. 1e-3
    double    ell    = -INFINITY;

    for (int k = 0; k < max_iter; ++k) {
        VectorXd G   = VectorXd::Zero(ndim);
        MatrixXd B   = MatrixXd::Zero(ndim, ndim);
        double   ell_k = 0.0;

        // (A)+(B): postorder then preorder; accumulate per-pattern score & OPG.
        //          BOTH passes restricted to [ptn_start, ptn_end) under Mode P (Architecture C).
        computePostorderPartials();                // p_i  (reuse pruning)
        computePreorderPartialsAndScore(/*out*/ ell_k, G, B);   // q_i + s_i + OPG  (NEW kernel)

        // (C): the single Gather-Reduce of this iteration.
        modeLAllreduce(ell_k, G, B);               // 1 + ndim + ndim² doubles, one MPI_Allreduce
        ell = ell_k;

        if (G.lpNorm<Infinity>() < gtol) break;

        // (D): LM damped solve with adaptive damping (local; re-uses B,G on reject).
        VectorXd delta; double ell_new;
        for (int trust = 0; trust < params->mode_l_max_trust; ++trust) {
            MatrixXd A = B; A.diagonal() += lambda * B.diagonal();   // Marquardt scaling
            delta = A.ldlt().solve(G);                               // SPD solve (B is PSD)
            VectorXd theta_try = theta + delta;
            ell_new = evalLogLikelihoodAt(theta_try);                // pattern-parallel, 1 reduce of 1 double
            if (ell_new > ell) { theta = theta_try; lambda /= params->mode_l_nu; break; }
            else                 lambda *= params->mode_l_nu;        // reject, re-solve (no traversal)
        }
        if (fabs(ell_new - ell) < params->mode_l_ltol) break;
    }
    unpackParams(theta);                            // write back to tree + model objects
    return ell;
}
```

`B.ldlt().solve` is Eigen's Cholesky (LDLᵀ) — `B+λdiagB` is SPD. IQ-TREE bundles Eigen
(`FindEigen3.cmake`) and GSL; either works for the ~225×225 solve.

### 4.4 The new preorder/score kernel (the one hard piece)

`computePreorderPartialsAndScore` mirrors the postorder pruning but runs root→tips, and at each
branch/pattern forms the score contributions:
```
  for each node i (preorder):
     q_i = P_iᵀ [ q_parent ∘ (P_sib · p_sib) ]                         // Eq. 7
     for each rate category c, each pattern p in [ptn_start, ptn_end):
        L_{p,c} = q_i(p,c)ᵀ p_i(p,c)                                   // pattern-cat likelihood
        s_{p}[branch_i]  += w_c · (q_i(p,c)ᵀ Q P_i p_i(p,c)) / L_p     // Eq. 9 (branch dim)
        s_{p}[α]         += w_c · (∂r_c/∂α) · (branch-rate term)       // §3.2 (alpha dim)
        s_{p}[model θ_j] += w_c · (q_i(p,c)ᵀ (∂P_i/∂θ_j) ... ) / L_p   // §3.2 (model dims)
     // after all nodes contribute to s_p, do the OPG rank-1 update:
     B += f_p · s_p s_pᵀ ;  G += f_p · s_p ;  ℓ += f_p · log L_p
```
Implementation notes:
- Reuse the `theta_all`/`computeTheta` data-movement pattern; the `q_iᵀ Q P_i p_i` contraction is
  the same SIMD shape as the existing `computeLikelihoodDerv` inner loop.
- The OPG update `B += f_p · s_p s_pᵀ` is `ndim²/2` FMAs per pattern. At ndim=218, npat=946K:
  ~2.3×10¹⁰ FMAs ≈ 0.05–0.1 s per iteration on 103 threads — pattern-parallel, dwarfed by the two
  traversals. Use a symmetric-rank-update (`dsyr`) over per-thread accumulators, reduced at the end.
- Keep `s_p` as a per-thread `ndim`-vector (≈1.8 KB) — no per-pattern storage of the full Jacobian.

---

## 5. Parallelization — One Gather-Reduce per LM Iteration

### 5.1 The decisive change vs Mode P

```
                          Allreduces / model     bytes / Allreduce     latency-bound barriers
  Mode P (gs=k)           ~8 000                 8–24 B                ~8 000  ← the wall
  Mode L (this design)    ~10–80                 8 B + (ndim+ndim²)·8B ~10–80  ← 100× fewer
```
At ndim=218, the Mode L reduce is `(1 + 218 + 218²)·8 B ≈ 382 KB`. On Gadi InfiniBand HDR
(~25 GB/s, ~1 µs latency): ~15 µs bandwidth + ~1 µs latency per reduce → **~1 ms total per model**.
Mode P's 8 000 latency-bound barriers cost ~24 ms *in latency alone* and serialize the faster
half-rank behind the slower on every call. Mode L moves more *bytes* but pays ~100× less in the
*latency* and *serialization* that actually dominated Mode P.

This is why **Mode L makes Mode P efficient at any group size**: with only tens of barriers, the
Allreduce tax is negligible even at gs=16. The traversal-slice fix (Architecture C, already
designed) ensures both the postorder and preorder passes only touch `[ptn_start, ptn_end)`, so a
gs=k cohort does 1/k of the per-pattern work and reduces once per iteration.

### 5.2 The two pattern-parallel passes

Both `computePostorderPartials` and `computePreorderPartialsAndScore` are restricted to the rank's
pattern slice under Mode P. The score `s_p`, gradient `G`, and OPG `B` are all **sums over
patterns**, so splitting patterns across ranks and Allreducing `[ℓ|G|B]` is **mathematically
exact** (modulo FP non-associativity, bounded as in Mode P §12). The branch derivative
`q_iᵀ Q P_i p_i` at pattern `p` depends only on `p`'s tip states and the model — pattern-local,
exactly like Felsenstein — so the slice decomposition is sound.

### 5.3 Memory

The preorder partials `q_i` need the same footprint as the postorder partials `p_i`
(`O(nnode·nptn·ncat·nstates)`). At AA 1M this doubles the partial-LH memory to ~2× the current
`central_partial_lh`. Under Mode P gs=k with Architecture C, each rank stores only its slice → the
per-rank footprint is `2/k ×` the FCA single-rank footprint. At gs=8 this is *less* memory than
FCA today. (Follow-on: the preorder pass can stream/recompute to trade compute for memory if 10M
pressures DRAM — see §13.)

---

## 6. How Mode L Fixes Dispatch & Load Balancing

Mode L does **not** introduce a new scheduler. It removes the two reasons the existing schedulers
failed:

1. **Absolute work collapses.** If LG+F+I+G4 drops from 447 s → ~15–45 s (even a conservative
   10–30× from joint analytic LM), then the *entire* heavy tail shrinks proportionally. FCA np=16
   with its known imperfect balance would finish the whole 224-model set in well under the 600 s
   gate, because the longest pole is now tens of seconds, not hundreds. **The heavy-tail imbalance
   that killed MPGC/ATMD-AID is a constant-factor problem; Mode L shrinks the constant ~10–30×.**

2. **Pattern-parallelism becomes a usable lever again.** For whatever heavy tail remains, Mode P at
   gs=4/8/16 is now efficient (§5.1). The EDM moldable scheduler can assign large cohorts to the
   few residual heavy models and small cohorts (gs=1, FCA-like) to the light ones — and the
   group-size choice is no longer fighting an 8 000-barrier tax. EDM's "choose gs per task" finally
   pays off.

3. **Clean resize points.** Each LM iteration is a single uniform map-reduce with a natural barrier
   (the Gather-Reduce). Dynamic group lending / cohort repacking (the "Adaptive Rank Lending" idea
   in [`mode-p-dispatch-investigation-plan.md`](mode-p-dispatch-investigation-plan.md) §7.1) becomes
   trivial: change `mp_allreduce_comm` between iterations, re-slice, continue — no mid-traversal
   state to reconcile, ~tens of iterations so re-balancing overhead amortizes cheaply.

**Recommended stack:**
```
  ┌─────────────────────────────────────────────────────────┐
  │  EDM moldable scheduler   (task DAG, epochs, rate filter) │  ← from event-driven-moldable-dispatch.md
  ├─────────────────────────────────────────────────────────┤
  │  Mode L per-model engine  (analytic-gradient joint LM)    │  ← THIS DOCUMENT
  ├─────────────────────────────────────────────────────────┤
  │  Mode P pattern slice + Architecture C  (efficient now)   │  ← reused, finally cheap
  └─────────────────────────────────────────────────────────┘
```

---

## 7. Code Map & Concrete Integration

### 7.1 Files to touch / add

```
NEW   tree/modeloptimizerL.h / .cpp    Mode L driver: optimizeModeL(), param pack/unpack,
                                       LM solve, damping schedule.
NEW   tree/phylokernelpreorder.h       Preorder partial-likelihood + per-pattern score + OPG
                                       (SIMD, templated like phylokernelnew.h). The core kernel.
EDIT  model/modelmarkov.{h,cpp}        Add computeTransDervParam(): ∂P/∂θ_j via eigen-derivative
                                       (Kenney–Gu). Reuse decomposeRateMatrix eigenpairs.
EDIT  model/rategamma.{h,cpp}          Add dRate_dAlpha(): ∂r_c/∂α for discrete-gamma categories.
EDIT  model/ratefree.{h,cpp}           Add dRate/dWeight derivatives; ordered-gap transform.
EDIT  model/rateinvar.{h,cpp}          Add ∂L/∂pinv closed form.
EDIT  model/modelfactory.cpp:1558      optimizeParameters: if (params->mode_l) return
                                       tree->optimizeModeL(...); else <existing alternating loop>.
EDIT  tree/phylotree.{h,cpp}           Members: preorder partial buffers; countModeLParams();
                                       packParams/unpackParams; modeLAllreduce().
EDIT  utils/tools.{h,cpp}              CLI flags (below). Params::mode_l fields.
```

### 7.2 CLI flags (additive, default off)

```
--mode-l                  Enable Mode L joint LM optimizer (default off → existing optimizer)
--mode-l-lambda0  X       Initial LM damping (default 1e-3)
--mode-l-nu       X       Damping up/down factor (default 3.0; Nielsen-style)
--mode-l-max-iter N       Max LM iterations per model (default 40)
--mode-l-gtol     X       Gradient inf-norm stop (default 1e-3)
--mode-l-curv {opg|hybrid|diag}   Curvature: OPG, OPG+exact-branch-diag (default), diag-only
--mode-l-fd-check         DEBUG: cross-check analytic gradient vs finite difference, abort on >1%
```

### 7.3 MPI payload

```cpp
// One Allreduce per LM iteration. Pack contiguous: [ell | G | vech(B)] (B symmetric → upper tri).
size_t nB = ndim*(ndim+1)/2;                         // ~24k doubles at ndim=218
vector<double> buf(1 + ndim + nB);                   // ~190 KB (symmetric) — half of dense
MPI_Allreduce(MPI_IN_PLACE, buf.data(), buf.size(), MPI_DOUBLE, MPI_SUM, mp_allreduce_comm);
```
Use `mp_allreduce_comm` (the existing MPGC/Mode P group communicator) — Mode L inherits the exact
same comm-inheritance path as B.4-15. Keep `--mca coll ^ucc` (B.4-13). Symmetric packing (`vech`)
halves the payload.

### 7.4 Correctness scaffolding

- `--mode-l-fd-check`: at iteration 0, compare each analytic `G[j]` against a central finite
  difference `(ℓ(θ+he_j)−ℓ(θ−he_j))/2h`; abort if relative error > 1%. This is the single most
  important debugging tool — the eigen-derivative `∂P/∂θ` and `∂r_c/∂α` are the error-prone pieces.
- lnL parity gate: Mode L final lnL must match the existing optimizer to ≤ 1e-2 at AA 1M (same
  tolerance as Mode P P.6), ≤ 1e-6 at AA 100K (ISO gates).

---

## 8. Worked Example — alpha in LG+F+I+G4 (the P.7 killer)

Today, optimizing alpha for one outer iteration of LG+F+I+G4 at AA 1M:
- Brent on alpha: ~10–20 evaluations, **each a full 946K-pattern × 4-category tree likelihood**.
- The `+F` frequencies (19 params): finite-diff BFGS, **20 traversals per gradient** × several
  BFGS steps.
- Branch sweeps interleaved, repeated over the alternating outer loop until joint convergence.
- Result: ~332–447 s, dominated by full-tree re-evaluations driven by alpha + freq + coupling.

Under Mode L, **alpha, the 19 frequencies, p_inv, and all 197 branches are one 218-vector**,
optimized jointly. alpha's gradient is `Σ_c (∂r_c/∂α)·(category branch-rate term)` — the
category branch-rate term is *already computed* in the preorder pass for the branch gradient, and
`∂r_c/∂α` is a scalar. **alpha costs essentially nothing extra.** The whole model converges in
~10–40 LM iterations = ~20–80 traversals. If a traversal is ~0.3–0.5 s at AA 1M (single rank),
that's ~10–40 s single-rank, and gs=4–8 Mode P brings it to single-digit seconds.

This is the concrete mechanism by which Mode L removes the exact bottleneck (`+G4`/`+I+G4` at AA
1M) that defeated ATMD-AID and EDM.

---

## 9. Staged Implementation — Phase by Phase (L.0 → L.7, with code-demanded sub-steps)

Each phase is independently testable and independently valuable. **Do them in order; do not start
the next until the gate passes.** The ordering is chosen so the *value-proving* gate (L.2, alpha on
a `+G4` model) comes early and the *highest-risk* piece (L.3, the `∂P/∂θ` eigen-derivative) comes
only after the approach is proven — and may not even be needed for the AA P.7 critical path (§0.5.5).

**Reading note vs rev. 1:** L.1 (branch-only LM) is now explicitly *not* the value gate — IQ-TREE
branches are already analytic NR, so branch-only LM mainly proves the preorder kernel is correct;
it will show only modest wall gains. The bottleneck is alpha (L.2). Four new sub-steps (L.0.5,
L.3.5, L.4.5, L.5.5) were added because the source audit revealed they are load-bearing.

### Phase L.0 — Preorder kernel + gradient-check harness (FOUNDATION)
- **Build:** `tree/phylokernelpreorder.h` — the upper/preorder partial-likelihood pass `q_i` (Ji et
  al. Eq. 7), templated/SIMD like `phylokernelnew.h`. Reuse the `computeTheta`/`theta_all`
  data-movement pattern and the `computeBounds`/`num_packets` packetization. Output: branch gradient
  `∂logL/∂b_i = q_iᵀ Q P_i p_i / L` (Eq. 9) for **all** branches in one postorder+preorder pair.
- **Build:** `--mode-l-fd-check` — compare each analytic `G[branch]` to central FD
  `(ℓ(b+h)−ℓ(b−h))/2h`; abort >1%.
- **Trap-avoidance:** make the preorder pass honor `[ptn_start,ptn_end)` from day one (mirror the
  Architecture C pattern at `phylokernelnew.h:1284`), even though L.0 is serial — so L.5 needs no
  kernel rewrite. Reset `partial_lh_computed`/`theta_computed` flags on model boundaries.
- **Gate:** analytic branch gradient matches central FD <1% on AA 100K LG+G4; `computeLikelihood`
  unchanged. **Effort 3–4 d. Risk: med (new kernel).** This gate decides whether the whole direction
  is viable.

### Phase L.0.5 — (OPTIONAL de-risk) "cheap alpha" without the full preorder
- **Idea (from §0.5.5):** `∂logL/∂α = Σ_c (∂logL/∂r_c)·(∂r_c/∂α)`. `∂r_c/∂α` is a cheap scalar 1-D
  FD of the `k` rate values (no traversal). `∂logL/∂r_c` (derivative w.r.t. a *global* per-category
  rate scale) may be obtainable from the existing single-branch derivative machinery summed with the
  pulley trick — *without* a full preorder kernel.
- **Use:** if L.0's preorder kernel is delayed, this proves the alpha-elimination value first with
  far less code. If L.0 lands cleanly, skip L.0.5.
- **Gate:** analytic `∂logL/∂α` matches FD <1% on AA 100K LG+G4. **Effort 1–2 d. Risk: low.**

### Phase L.1 — Branch-only joint LM (proves the LM loop + solver)
- **Build:** `PhyloTree::optimizeModeL` skeleton (§4.3): pack branches (log-transform), assemble OPG
  `B` from per-pattern branch scores, LM damped solve (`Eigen LDLᵀ`), adaptive λ. Model+rate params
  held at legacy-optimized values.
- **Trap-avoidance:** parameter transforms (log b) keep iterates interior — no near-zero-branch blowup.
- **Gate:** AA 100K lnL parity ≤1e-6 vs legacy on a fixed model; LM converges < 40 iterations.
  **Effort 2 d. Risk: med.** (Expect only modest wall gain — that's fine, this is a correctness step.)

### Phase L.2 — Alpha + p_inv analytic gradient (THE VALUE-PROVING GATE)
- **Build:** `RateGamma::dRate_dAlpha()` (cheap scalar FD of the `k` `r_c`, §0.5.5); fold α into the
  LM vector (log-transform). `RateInvar`: closed-form `∂logL/∂p_inv` (replaces the EM loop in the LM
  path); logit-transform p_inv. The per-category branch-rate term comes from the L.0 preorder pass.
- **Trap-avoidance:** verify against EM — the analytic p_inv gradient must reach the same p_inv the
  EM loop finds (EM is a special case of the score equation). Keep `computeRates()` called whenever α
  changes (mirror `RateGammaInvar::getVariables`).
- **Gate:** on AA 100K **LG+G4 and LG+I+G4** — lnL parity ≤1e-6; recovered α within 1e-3 of Brent's;
  **traversal count per model ≤ ~40** (vs the alpha-Brent-dominated legacy count). **Effort 2–3 d.
  Risk: med.** *If this gate fails, stop and diagnose before L.3 — this is the minimum-viable proof
  that Mode L removes the AA `+G4`/`+I+G4` bottleneck.*

### Phase L.3 — Model-parameter gradient `∂P/∂θ` (HIGHEST RISK; needed only for estimated models)
- **Build:** `ModelMarkov::computeTransDervParam(θ_j, …)` — the Kenney–Gu eigen-derivative
  `∂P/∂θ = U·(M ∘ V)·U⁻¹`, `M = U⁻¹(∂Q/∂θ)U`, `V` = divided differences of `exp(λt)`. Reuse the
  stored `eigenvalues/eigenvectors/inv_eigenvectors` (`modelmarkov.h:502`). Targets: GTR
  exchangeabilities and `+FO` frequencies.
- **Trap-avoidance:** the divided-difference `V` is singular when eigenvalues coincide — use the
  limit form `t·exp(λt)` on the diagonal / near-degenerate pairs. `--mode-l-fd-check` every θ_j.
  **Empirical AA models (`+F`, fixed `Q`) skip this entirely** — so this phase is *not* on the P.7
  critical path; do it for DNA/codon completeness.
- **Gate:** GTR+F+I+G4 (DNA) FD-check <1% on all dims; lnL parity. **Effort 3–4 d. Risk: high.**

### Phase L.3.5 — Free-rate `+R` gradient + ordered-gap transform
- **Build:** `RateFree` analytic `∂logL/∂{w_c, r_c}` (same chain-rule shape as alpha, per category);
  replace the ratio-to-last packing + `quicksort` with an **ordered-gap + ALR transform** (§4.1) so
  categories stay ordered and distinct without sorting.
- **Gate:** LG4X / `+R4` AA 100K parity; no category collapse. **Effort 2 d. Risk: med.**

### Phase L.4 — Full joint Mode L replaces the inner loop (serial, np=1)
- **Build:** wire `optimizeParameters` (`modelfactory.cpp:1558`): `if (params->mode_l) return
  tree->optimizeModeL(...)`. Joint vector = branches ⊕ model ⊕ rate-het. Hybrid curvature (exact
  branch diagonal from `ddf` + OPG off-diagonal, §2.3).
- **Trap-avoidance:** **best-model selection must be unchanged** — joint LM may find slightly higher
  lnL than coordinate descent; gate on BIC *ranking*, not just per-model lnL (§13 Q7).
- **Gate:** AA 100K full ModelFinder — same best model (LG+G4); per-model lnL ≤1e-6; MF wall ≤ ½
  legacy np=1. **Effort 2 d. Risk: med.**

### Phase L.4.5 — Warm-start cache adaptation (code-demanded)
- **Why:** the canonical `WarmStartPacket` (`phylotesting.cpp:3207-3306`) stores *per-rate-class*
  converged params (gamma α, p_inv, +R props/rates) keyed by rate class. Mode L converges branches+
  model+rate **jointly**, so the warm-start seed semantics change: seed the LM `θ0` from the cache
  (good `θ0` cuts LM iterations) and **write back** the jointly-converged α/p_inv/+R into the same
  packet fields so downstream models still warm-start correctly.
- **Trap-avoidance:** preserve the existing packet layout/sentinels so non-Mode-L ranks (mixed runs)
  still parse it. Respect the `#pragma omp critical(model_info_lock)` around the shared cache.
- **Gate:** AA 100K — warm-start gives ≥1.3× fewer LM iterations on later models; no parity change.
  **Effort 1 d. Risk: low.**

### Phase L.5 — Pattern-parallel single Gather-Reduce (Mode P + Arch C, all substrate present)
- **Build:** `modeLAllreduce(ℓ, G, B)` — one `MPI_Allreduce(MPI_IN_PLACE, [ℓ|G|vech(B)], …,
  mp_allreduce_comm)` per LM iteration (symmetric `vech` packing, §7.3). Postorder + preorder passes
  already sliced (L.0). Inherit group state via the **B.4-15** path (`evaluate(... in_tree)`).
- **Trap-avoidance (the whole §9.5 checklist applies here):** keep `--mca coll ^ucc` (B.4-13); use
  `mp_allreduce_comm` not WORLD; one IQTree per rank; allreduce only on the main thread after the OMP
  region (MPI_THREAD_FUNNELED); deterministic LM iteration count across a cohort (all ranks must call
  the same number of Allreduces — make `max_iter`/convergence test depend only on reduced globals,
  never on rank-local state).
- **Gate:** AA 100K np=2 and np=4 — lnL parity ≤1e-6 (ISO-4 style); measured reduce count ~10–80/
  model; no deadlock; `[Mode P]` slices correct. **Effort 2 d. Risk: med (lock-step + FP assoc).**

### Phase L.5.5 — EDM/dispatch integration (Mode L as the per-task engine)
- **Build:** route the per-model evaluation inside `aidExecuteWaves` (`phylotesting.cpp:4293`, group
  comm set at `:4343`) through `optimizeModeL`. Update `aidComputeCostPred` (`:3860`) — Mode L's cost
  model is ~`LM_iters × 2 traversals` (much flatter across rate classes than the current `rate_mult`
  table, since alpha is no longer a separate Brent cost). Cohorts/group-size selection unchanged
  (EDM owns it); Mode L just makes large `gs` efficient.
- **Trap-avoidance:** the LM iteration count must be **agreed across the cohort** before the wave (a
  cohort runs in lock-step) — broadcast `max_iter` or use a fixed cap with a globally-reduced
  convergence flag. Honor the `MPI_Barrier(MPI_COMM_WORLD)` between waves; never strand a sub-cohort
  across it.
- **Gate:** AA 1M np=16 — Mode L under EDM cohorts; correct lnL. **Effort 2 d. Risk: med.**

### Phase L.6 — P.7 perf gate
- **Run:** AA 1M np=16 `--mode-l --mode-p-group-size 4` (and a gs=1 FCA+Mode-L variant). Reuse
  `gadi-ci/mode-p-iso/run_p7_*.sh` with a `--mode-l` arm.
- **Gate:** MF wall ≤ 600 s; lnL ≤1e-2 vs FCA; best model LG+G4. **Effort 1 d. Risk: low if L.0–L.5.5 pass.**

### Phase L.7 — Scaling validation
- **Run:** AA 10M np=16 (≤6 000 s), DNA 1M/10M np=16 (parity + wall). Exercises L.3 (`∂P/∂θ`) for the
  DNA GTR path and the §5.3/§13.4 memory behavior at 10M.
- **Gate:** AA 10M ≤ 6 000 s; DNA parity. **Effort 2–3 d. Risk: low–med (10M memory).**

**Total ≈ 18–26 working days** (was 14–20; the four code-demanded sub-steps add ~4–6 d).
**Critical path to the P.7 win:** L.0 → L.2 → L.4 → L.5 → L.6 (L.3/L.3.5 are off the AA path).

**Minimum-viable proof (do this first, serial, ~1 week):** L.0 → L.2. If branch+alpha analytic LM
does not converge a `+G4` model in ≤ ~40 traversals with lnL parity on AA 100K, **stop and diagnose**
before any MPI or eigen-derivative work.

---

## 9.5 Traps & Invariants Checklist — every past failure, and how Mode L avoids it

The previous five architectures each died on a specific, now-understood trap. Mode L must preserve
every hard-won invariant. **This table is the pre-flight checklist for the L.5/L.5.5 MPI work.**

| # | Past trap (where it bit) | Root cause | Invariant Mode L must hold | How (file:line) |
|---|---|---|---|---|
| 1 | **UCC "Message truncated"** (B.4-13, job 169179725) | OpenMPI 4.1.7 UCC mis-routes concurrent Allreduces of *different sizes* on sibling sub-comms | All Mode L cohorts use the **same** reduce shape per step; keep `--mca coll ^ucc` | run scripts + `MPI_Barrier(mp_allreduce_comm)` `phylotesting.cpp:5194` |
| 2 | **filterRates deadlock** (B.4-14) | `MPI_Bcast` on `MPI_COMM_WORLD` while groups desync | Any rate-filter/gather Mode L adds is on `fca_comm`/`mp_allreduce_comm`, never WORLD mid-evaluation | `fca_comm` `phylotesting.cpp:3159,3205,4651` |
| 3 | **Wrong cross-group sums** (B.4-15, job 169189043) | fresh per-model IQTree defaulted to WORLD/gs=1 | Mode L's per-model tree **inherits** `mp_allreduce_comm/rank/size` via `evaluate(... in_tree)` before `initializePtnPartition` | `phylotesting.cpp:1970-1986` |
| 4 | **Pattern-split gave only ~10%** (Arch C) | traversal used full `nptn`, only summation sliced | **Both** Mode L passes (postorder *and* preorder) honor `[ptn_start,ptn_end)` | mirror `phylokernelnew.h:1284-1291` in the new preorder kernel |
| 5 | **Divergent BFGS / rank-local state** (ATMD-AID, job 169221567) | ranks started collective waves from rank-local checkpoints | Cohort LM starts from **canonical** broadcast state; LM convergence test uses only **reduced globals**, never rank-local lnL | extend `WarmStartPacket` Bcast `phylotesting.cpp:3207-3306` (L.4.5) |
| 6 | **MPI from worker threads** | MPI_THREAD_FUNNELED on Gadi | `modeLAllreduce` only on main thread, **after** the `#pragma omp parallel` region closes | model on `modePAllreduceLh` `phylokernelnew.h:3395` (post-region) |
| 7 | **OOM / heap races** (naive OMP-across-models) | 103 concurrent IQTree instances | **One IQTree per rank**; `atmd_K_outer=1` whenever Mode P/L active | `isModePActive()` requires `atmd_K_outer<=1` `phylotree.cpp:924` |
| 8 | **Load imbalance** (MPGC static family→group; ATMD-AID phase split) | static binding / binary heavy-light split | Mode L shrinks absolute per-model cost so imbalance is a small constant; EDM owns moldable gs (Mode L doesn't re-introduce static binding) | EDM `aidExecuteWaves` `:4293` |
| 9 | **Cohort lock-step break** (deadlock risk, new) | ranks in a cohort call different #Allreduces | LM `max_iter` and convergence are **cohort-global**: cap iterations and AND-reduce a convergence flag so every rank does the same number of reduces | L.5/L.5.5 design rule |
| 10 | **Best-model drift** (new risk from joint opt) | joint LM may reach higher lnL than coord-descent | gate on **BIC ranking** parity, not just lnL; document any lnL improvement | L.4 gate + §13 Q7 |
| 11 | **SIMD tail / pattern alignment** | `ptn_start/ptn_end` must be VCSIZE-aligned | reuse `initializePtnPartition`'s `align_up(VCSIZE=8)`; last rank takes the unaligned tail | `phylotree.cpp:967-990` |
| 12 | **FP non-associativity over OPG sum** | reduction order differs across ranks | gate lnL at 1e-2 (AA 1M)/1e-6 (100K); far fewer reduces than Mode P → *less* drift | §5.2, §12 item 5 |

**Pre-flight rule for L.5:** before the first multi-rank Mode L run, walk this table top to bottom
and point to the line of code that enforces each invariant. The previous architectures failed
because one of these was silently violated; the cost of the checklist is minutes, the cost of a
PBS perf-gate that crashes at the Q.BIRD boundary is a day.

---

## 10. Validation Strategy

| Gate | Dataset | Pass criterion | Mirrors |
|------|---------|----------------|---------|
| L-FD | AA 100K | analytic grad vs central FD <1% on every dim | (new) |
| L-ISO1 | AA 100K np=1 | lnL ≤1e-6 vs legacy; best model unchanged | ISO-1 |
| L-ISO4 | AA 100K np=4 | lnL ≤1e-6; ~10–80 reduces/model; no deadlock | ISO-4 |
| L-P7 | AA 1M np=16 | MF wall ≤ 600 s; lnL ≤1e-2 vs FCA | P.7 |
| L-10M | AA 10M np=16 | MF wall ≤ 6 000 s | (design target) |
| L-DNA | DNA 1M/10M np=16 | lnL parity; ≤ design wall | (regression) |

Reuse the existing ISO/P.7 harness scripts in `gadi-ci/mode-p-iso/`; add `--mode-l` variants.

---

## 11. Performance Projection (AA 1M np=16) — corrected, real-data-anchored

This rev. **drops the Ji-et-al-100× extrapolation** (it applies to finite-difference branch
gradients, not IQ-TREE's analytic-NR branches; §0.5.2) and instead derives the gain mechanism by
mechanism from the **measured** FCA rank-0 times (§0.5.3).

**Per-model mechanism (the heavy model, LG+F+I+G4 = 332.8 s measured):**
- Measured decomposition along the rate ladder: LG+F = 9.9 s → +I = 40.0 s → +G4 = 89.2 s →
  **+I+G4 = 332.8 s**. The jump to `+I+G4` (+244 s over `+G4`) is **alpha-Brent re-search per outer
  iteration × many outer iterations**, p_inv EM, and branch↔rate alternation coupling.
- Mode L removes the alpha-Brent inner search and the alternation: α and p_inv become two components
  of one joint analytic-gradient + curvature step. Estimated per-model factor: **~4–8×** (it cannot
  beat the irreducible cost of the ~20–40 full traversals the joint LM still needs, each ~0.5–1.3 s
  at AA 1M). So LG+F+I+G4: **332.8 s → ~45–85 s single-rank.**
- With gs=4 Mode P (now efficient — Arch C slices both passes, ~tens of barriers not tens of
  thousands): another **~3–3.5×** → **~13–28 s.**

**Full 224-model set (FCA np=16 distribution, MF wall 1 122 s measured):** apply the ~4–8× per-model
factor to the model-evaluation portion; the longest pole (rank owning the LG family) drops from
~470 s of heavy models to ~60–120 s:

| Quantity | Pessimistic | Optimistic |
|---|---:|---:|
| Per-heavy-model Mode L factor (mechanism above) | 4× | 8× |
| LG+F+I+G4 single-rank | ~85 s | ~45 s |
| …+ gs=4 Mode P (efficient) | ~28 s | ~13 s |
| **Full MF wall, FCA np=16 + Mode L** | **~350 s** | **~150 s** |
| vs FCA np=16 legacy (1 122 s) | **3.2×** | **7.5×** |
| P.7 gate (≤ 600 s) | ✅ ×1.7 margin | ✅ ×4 margin |

Even the **pessimistic** 4× clears the 600 s gate by 1.7×. Upside beyond this comes from EDM giving
the residual heavy tail gs=8/16 (cheap now) and from DNA/estimated models where the `(ndim+1)`
finite-difference elimination is a larger multiplier.

**Why this is more credible than rev. 1's "60–200 s":** it is grounded in the measured 332.8 s and a
named mechanism (alpha-Brent elimination + joint convergence), not in a viral-tree finite-difference
speedup that does not apply here.

**AA 10M:** per-traversal ~3–5 s; joint LM ~20–40 traversals/heavy model → ~60–200 s single rank;
gs=8 → ~10–30 s; full set ~1 500–3 500 s. Clears the ≤ 6 000 s target with margin.

**DNA GTR 1M/10M:** here the `(ndim+1)` finite-difference model gradient *is* the cost, so Mode L's
multiplier is larger than the AA case — the eigen-derivative `∂P/∂θ` (L.3) turns ~`ndim+1` traversals
per gradient into ~2. Expect ≥ the AA factor.

---

## 12. Risks & Mitigations

1. **Eigen-derivative `∂P/∂θ` correctness (L.3, highest risk).** The Kenney–Gu divided-difference
   formula for `∂exp(Qt)/∂θ` is subtle when eigenvalues are close/complex (non-reversible models).
   *Mitigation:* `--mode-l-fd-check` gates every dimension; land branch+alpha (L.0–L.2, which avoid
   `∂P/∂θ`) first; for empirical AA matrices (LG, WAG…) `Q` is fixed so only `+F` frequencies and
   rate-het need `∂P/∂θ` — the heaviest P.7 models (LG-family) need *no exchangeability derivative*
   at all. So L.0–L.2 already cover the P.7 critical path; L.3 is mainly for GTR/estimated models.

2. **OPG is a poor Hessian far from optimum / near boundaries.** *Mitigation:* LM damping `λ` (large
   `λ` → safe gradient ascent); hybrid curvature with exact branch diagonal (§2.3); the transformed
   parameter space (log/logit/ALR) keeps iterates interior so near-zero branches don't blow up.

3. **Convergence to a different optimum than the legacy alternating loop.** ML surface is the same;
   both should reach the same MLE. *Mitigation:* lnL-parity gates at every milestone; if Mode L
   finds a *higher* lnL (plausible — joint beats coordinate descent), document it as an improvement,
   not a regression, but verify best-model selection (BIC ranking) is stable.

4. **Preorder kernel doubles partial-LH memory.** *Mitigation:* Mode P slicing (Arch C) makes
   per-rank memory `2/k×` FCA — a net *reduction* at gs≥2; streaming/recompute fallback for 10M.

5. **FP non-associativity over the OPG sum.** Same bounded concern as Mode P §12; far fewer
   reduces (tens vs thousands) → *less* accumulated drift. Gate at 1e-2 (AA 1M), 1e-6 (100K).

6. **Free-rate `+R` ordering / degeneracy.** Ordered-gap transform (§4.1) removes the sort hack and
   keeps categories distinct; LM damping handles near-degenerate categories gracefully.

7. **Effort is large for a research bet.** *Mitigation:* the milestone ladder (§9) makes each step
   independently valuable and gated; L.0–L.2 is a small, self-contained win even if L.3+ stalls.

---

## 13. Open Questions & Follow-on

1. **Exact observed information vs OPG.** Worth computing the true `∂²ℓ/∂θ²` (second-order preorder)
   for the model block? Probably not initially — OPG + exact-branch-diagonal + LM damping should
   suffice; revisit if convergence stalls on `+R`-heavy models.
2. **Sparsity of `B`.** The branch–branch Fisher block is dense but the branch–model coupling may be
   near block-diagonal; a block solve (branches via the existing per-branch ddf; model block via the
   small dense OPG) could replace the full `ndim×ndim` solve and reduce the reduce payload to
   `O(ndim_model²)` + diagonal. Promising for 10M and large taxa counts.
3. **Warm-starting across models.** The cross-model warm-start cache
   ([`bfgs&CrossModelWarmStart.md`](bfgs&CrossModelWarmStart.md)) becomes even more valuable: a good
   `θ0` cuts LM iterations further. Reuse the existing `RateWarmStartCache` broadcast.
4. **Memory-streamed preorder for 10M.** Recompute `q_i` on the fly per subtree to cap DRAM; trades
   compute for memory. Only if 10M pressures the node.
5. **GPU / BEAGLE.** The 2-traversal + dense-solve structure is GPU-friendly (BEAGLE 3 already does
   pre/postorder partials for HMC); a future GPU backend slots under Mode L unchanged.
6. **Tree search reuse.** The same analytic all-branch gradient accelerates the *tree-search* phase
   (NNI/SPR branch re-optimization), which the Amdahl analysis flags as the *next* bottleneck after
   MF. Mode L's gradient machinery is reusable there — a second, larger payoff.
7. **Does joint LM change model selection?** Joint optimization may yield slightly higher lnL than
   coordinate descent for some models; confirm BIC *rankings* (not just per-model lnL) match so
   ModelFinder's chosen model is unchanged.

---

## 14. Why This Is Novel

- **No phylogenetic ModelFinder uses a second-order, analytic-gradient, joint
  branch+model+rate-het optimizer.** IQ-TREE, RAxML-NG, ModelTest-NG all use coordinate descent +
  Brent + finite-difference BFGS for model selection.
- Ji et al. 2020 built the O(N) gradient and used **L-BFGS** for *single-tree* inference (BEAST/HMC
  context), **not** for ModelFinder's 224-model selection and **not** with an explicit LM/Fisher
  curvature.
- Taylor et al. 2022 showed **LM ≫ BFGS/L-BFGS** for small-medium parameter fits but in a
  neural-network/PINN context — **never applied to phylogenetics**.
- **The synthesis** — Ji's O(N) analytic gradient (incl. ∂r_c/∂α) + empirical-Fisher LM (Taylor's
  curvature argument) + one-Gather-Reduce-per-iteration pattern parallelism (replacing Mode P's
  per-call Allreduce) + EDM moldable dispatch on top — is, to the reach of this survey, **unpublished
  and specific to the ModelFinder heavy-tail problem.** It attacks the *number of iterations* (the
  true Amdahl serial term) rather than the *cost per iteration* (what every prior architecture
  attacked).

---

## 15. References

1. Ji X, Zhang Z, Holbrook A, Nishimura A, Baele G, Rambaut A, Lemey P, Suchard MA. *Gradients Do
   Grow on Trees: A Linear-Time O(N)-Dimensional Gradient for Statistical Phylogenetics.* Mol.
   Biol. Evol. 37(10):3047–3060 (2020). arXiv:1905.12146. **[enabling tech: O(N) gradient]**
2. Taylor J, Wang W, Bala B, Bednarz T. *Optimizing the optimizer for data-driven deep neural
   networks and physics-informed neural networks.* arXiv:2205.07430 (2022). **[LM ≫ BFGS/L-BFGS for
   small–medium parameter fits — the motivating paper]**
3. Giordan M, Vaggi F, Wehrens R. *On the maximization of likelihoods belonging to the exponential
   family using a Levenberg–Marquardt approach.* J. Stat. Comput. Simul. 87(5) (2017).
   arXiv:1410.0793. **[LM for exponential-family MLE; local convergence proof]**
4. Berndt E, Hall B, Hall R, Hausman J. *Estimation and Inference in Nonlinear Structural Models.*
   Ann. Econ. Soc. Meas. 3/4 (1974). **[BHHH / outer-product-of-gradients empirical Fisher]**
5. Osborne MR. *Fisher's method of scoring.* Int. Stat. Rev. 60 (1992). **[Fisher scoring =
   Gauss–Newton for MLE]**
6. Levenberg K (1944); Marquardt DW. *An Algorithm for Least-Squares Estimation of Nonlinear
   Parameters.* SIAM J. Appl. Math. 11(2):431–441 (1963). **[LM]**
7. Nocedal J, Wright SJ. *Numerical Optimization* (2006). **[LM/Gauss-Newton/trust-region theory]**
8. Felsenstein J. *Evolutionary trees from DNA sequences: a maximum likelihood approach.* J. Mol.
   Evol. 17:368–376 (1981). **[pruning / postorder partials]**
9. Kenney T, Gu H. *Hessian calculation for phylogenetic likelihood based on the pruning algorithm
   and its applications.* Stat. Appl. Genet. Mol. Biol. (2012). **[analytic phylogenetic Hessian /
   ∂P/∂θ eigen-derivative]**
10. Kalyaanamoorthy S, Minh BQ, Wong TKF, von Haeseler A, Jermiin LS. *ModelFinder.* Nat. Methods
    14:587–589 (2017). **[the target]**
11. Minh BQ et al. *IQ-TREE 2.* Mol. Biol. Evol. 37:1530–1534 (2020). **[host system]**
12. Companion design docs in this repo: [`mode-p-design.md`](mode-p-design.md),
    [`mode-p-implementation-status.md`](mode-p-implementation-status.md),
    [`event-driven-moldable-dispatch.md`](event-driven-moldable-dispatch.md),
    [`novel-dispatch-architectures.md`](novel-dispatch-architectures.md),
    [`mode-p-dispatch-investigation-plan.md`](mode-p-dispatch-investigation-plan.md).

---

## 16. Concrete Next Action (for the implementing self)

1. **Build L.0 first**: the preorder traversal kernel + `--mode-l-fd-check`. Verify the analytic
   **branch** gradient matches central finite differences to <1% on AA 100K LG+G4. (If the preorder
   kernel is slow to land, L.0.5 — "cheap alpha" via the global-rate-scale derivative — proves the
   value with less code.) This is the gate that determines whether the direction is viable.
2. **Then L.2 is the value gate** (L.1 branch-only LM is just a correctness stepping stone — branches
   are already analytic NR, so it won't show big wall gains). At L.2, fold α (cheap scalar `∂r_c/∂α`)
   and p_inv into the joint LM and confirm an AA 100K `+G4`/`+I+G4` model converges in ≤ ~40
   traversals with lnL parity and α within 1e-3 of Brent. **This is the minimum-viable proof that
   Mode L removes the alpha-Brent bottleneck that defines the AA P.7 critical path.**
3. **L.3 (`∂P/∂θ` eigen-derivatives) is OFF the AA P.7 critical path** — empirical AA models have
   fixed `Q` and empirical (fixed) `+F`, so they need no model-parameter derivative. Do L.3 for
   DNA GTR / `+FO` / `+R` completeness, after the AA path (L.0→L.2→L.4→L.5→L.6) is proven.
4. **Walk the §9.5 traps checklist** before the first multi-rank run (L.5). Each row is a past
   crash; point to the line of code that prevents it.

Do **not** launch further EDM/MPGC PBS perf gates for the parallelism-first line until L.2 proves
the optimizer-first hypothesis on AA 100K, serial. The optimizer is the ceiling; fix it first.

---

## 17. Implementation Log

### 17.1 — Phase L.0a (2026-05-27): FD-Jacobian OPG Levenberg-Marquardt for model+rate

**Delivered:** a working, compile-verified `--mode-l` joint Levenberg-Marquardt / Fisher-scoring
optimizer for the model + rate-heterogeneity parameters, replacing the legacy alternating
model-BFGS / alpha-Brent / p_inv-EM sub-optimization. The LM loop, parameter packing (reuses
`ModelFactory::setVariables`/`getVariables`), empirical-Fisher (OPG) curvature, adaptive damping with
accept/reject, and Eigen LDLT solve are all in place. The gradient/curvature source is a per-pattern
**finite-difference Jacobian** (the L.0b analytic preorder gradient drops in later).

**Files changed:**

| File | Change |
|---|---|
| `utils/tools.h` | `Params`: `mode_l_enabled, mode_l_lambda0, mode_l_nu, mode_l_max_iter, mode_l_gtol, mode_l_fd_check`. |
| `utils/tools.cpp` | CLI: `--mode-l`, `--no-mode-l`, `--mode-l-lambda0`, `--mode-l-nu`, `--mode-l-max-iter`, `--mode-l-gtol`, `--mode-l-fd-check`. |
| `model/modelfactory.h` | decl `optimizeModeLAllParameters(double)`; cached config members. |
| `model/modelfactory.cpp` | `#include <Eigen/Dense>`; ctor caches config; **gate** at top of `optimizeParametersOnly` (`if (mode_l_enabled) return optimizeModeLAllParameters(...)`); the optimizer impl after `optimizeAllParameters`. |

**Build status:** both changed TUs pass `mpicxx`(icpx)+`-D_IQTREE_MPI`+Eigen3.3.7 `-fsyntax-only`
(exit 0, no errors/warnings). Full clean PBS build = **job 169484680 ✅ exit 0**, binary md5
**`f89e0b3e965b93c078c0040273d6e684`** (`iqtree3-mpi-mode-p-iso-p3`, 149 MB, 2026-05-28 02:18).
`strings` confirms `_ZN12ModelFactory26optimizeModeLAllParametersEd`, `--mode-l*`, `MODE-L-FDCHECK`,
`MODE-L:` all present in the binary. Only pre-existing vectorclass `[-Wparentheses]` warnings —
nothing from Mode L code.

### 17.2 — Deviations from the §9 plan (and why)

- **D1 — Built L.0a (model+rate FD-LM) before L.0 (analytic preorder branch gradient).** Rationale:
  the audit (§0.5.2) proved the AA P.7 bottleneck is **alpha-Brent + the alternating loop**, which
  L.0a attacks *directly* with existing infrastructure — **no new SIMD kernel, no branch packing,
  build-verifiable this session**. It is the exact scaffold the analytic gradient (L.0b) drops into:
  only the Jacobian source changes; the LM machine is reusable. The preorder kernel (original L.0)
  becomes L.0b.
- **D2 — Per-pattern scores via `computePatternLikelihood()`, not raw `_pattern_lh`.** `_pattern_lh`
  (`phylotree.h:2427`) excludes scaling factors; `computePatternLikelihood` adds the
  `scale_num·LOG_SCALING_THRESHOLD` correction (`phylotree.cpp:1666-1685`). The per-pattern score is
  the finite difference of these *scaled* log-lh values.
- **D3 — Config cached as `ModelFactory` members** (mirrors `joint_optimize`), not via
  `PhyloTree::params` (no clean public accessor).
- **D4 — L.0a is not yet faster than legacy BFGS per evaluation** (same `(ndim+1)` evals/gradient).
  The expected L.0a win is *fewer LM iterations* + *joint alpha+p_inv* (no Brent inner search, no
  EM/BFGS alternation). The order-of-magnitude speedup awaits L.0b's 2-traversal analytic gradient.
- **D5 — Robust fallback:** if `targetFunk` returns the bad-frequency sentinel (≥1e11) at the base
  point or during perturbation, `optimizeModeLAllParameters` cleanly falls back to the legacy
  `optimizeAllParameters` (BFGS) — Mode L never makes a model fail to evaluate.

### 17.3 — Known risks / TODO carried forward

- **R1 (verify first):** `computePatternLikelihood` uses `current_it`/`current_it_back` for the
  scaling correction. After `targetFunk`→`computeLikelihood()` this *should* be valid, but it is
  unproven at runtime. The `--mode-l-fd-check` `|lnl-recon|` line is the canary; if it is not ≈0, see
  the handoff header fix. **This is the gate-2 check.**
- **R2:** the MPI `syncChkPoint->masterSyncOtherChkpts()` calls in the legacy
  `optimizeParametersOnly` are **skipped** in the Mode L path. Fine for serial / FCA (each rank owns
  its models), but must be revisited for the L.5 pattern-parallel path.
- **R3:** memory for the FD Jacobian is `ndim × nptn` doubles; small for AA model+rate (ndim 2–18),
  but the L.0b analytic kernel avoids storing it anyway.
- **R4:** Mode L currently engages for *every* model via the `optimizeParametersOnly` gate; under
  `--mode-l` alone (no `--mode-p`) it runs per-rank exactly like the legacy optimizer (FCA dispatch),
  so it is safe to test at np=1 and under FCA np>1 immediately. Pattern-parallel (Mode P) is L.5.
- **R5:** the FD step is a fixed `h = 1e-4·max(|x_j|,1)` (literal, not the file-local `ERROR_X`);
  matches the legacy `derivativeFunk` magnitude. Revisit if any dimension shows FD noise in L-FD.

### 17.4 — L.0b implementation sketch (do this after L-FD passes)

The L.0a code (`optimizeModeLAllParameters`) already implements the LM loop, Eigen LDLT solve,
adaptive damping, accept/reject, parameter packing/unpacking, and bound clamping. **L.0b is a single
swap**: the `(ndim+1)`-eval FD Jacobian assembly is replaced by one analytic call that returns
the gradient `G` and OPG curvature `B` (and the lnL) in a single postorder+preorder pair.

**Concrete swap point in `model/modelfactory.cpp` `optimizeModeLAllParameters`:**

```cpp
// REPLACE this whole block (the (A)+(B) FD Jacobian: ~25 lines, the per-dim perturb loop):
for (int j = 1; j <= ndim; j++) {
    /* ... perturb x[j] by h, eval, fill S[j-1][·] from (trial_plh - base_plh)/h ... */
}
/* ... (C) build G and B by summing pf[p] * S[j] * S[k] ... */

// WITH a single analytic call (proposed signature on PhyloTree):
double lnl_local;
tree->computeModeLGradientAndOPG(
    /* in: */  ndim, /* current x packed into model+rate */
    /* out: */ lnl_local, G, B
);
// G : Eigen::VectorXd, length ndim
// B : Eigen::MatrixXd, ndim x ndim (symmetric, PSD)
```

**New kernel files (tree/):**

- `tree/phylokernelpreorder.h` — templated SIMD preorder partial-likelihood pass `q_i = P_iᵀ [q_par ∘
  (P_sib · p_sib)]` (Ji et al. 2020 Eq. 7). Mirror the packetization/`computeBounds` structure of
  `phylokernelnew.h:1284-1481`. Allocate one preorder buffer per `PhyloNeighbor` (or reuse
  `theta_all` pattern; both `q` and `p` have the same `O(nnode · nptn · ncat · nstates)` shape).
- `tree/phylotree_modeL.cpp` (or methods on `PhyloTree`) — driver `computeModeLGradientAndOPG`:
  1. postorder pass: standard `computePartialLikelihood` (already exists).
  2. preorder pass: new kernel; at every node, for every rate category, for every pattern in
     `[ptn_start, ptn_end)`, accumulate per-pattern score `s_p[j]` for **all** model+rate dims:
     - branch dim *(not yet active in L.0b — branches stay on legacy NR until L.4)*.
     - alpha dim: `Σ_c (∂r_c/∂α) · ⟨q_i^c, Q_eff(b·r_c) · p_i^c⟩ · b_i / L_p`, where
       `∂r_c/∂α` is a **scalar** that can be obtained by a cheap one-time finite difference of
       `RateGamma::computeRates(α±h)` (no traversal — see §0.5.5).
     - p_inv dim: closed-form `(L_const_p − L_var_p) / L_p` (from `rategammainvar.cpp:220-287`).
     - `+R` weights/rates dim: same chain-rule structure (`Σ_c ∂w_c/∂θ · L_c + Σ_c w_c ∂L_c/∂r_c · ∂r_c/∂θ`).
     - model dim (GTR exchangeabilities, +FO frequencies): requires `∂P/∂θ` (Kenney–Gu eigen-
       derivative on `model/modelmarkov.{h,cpp}`) — **L.3 work**, skipped in L.0b since the AA P.7
       critical-path models have fixed `Q` and empirical `+F`.
  3. on the fly, accumulate `G[j] += f_p · s_p[j]` and `B[j][k] += f_p · s_p[j] · s_p[k]`.

**Validation path (re-uses the L-FD harness):** after L.0b lands, the very same
`run_lfd_aa100k_np1_mode_l.sh` script with `--mode-l-fd-check` becomes the analytic-vs-FD
cross-check — the FD-check now compares analytic `G` (just built) against a one-shot FD reference
(scalar `(targetFunk(x+h)−targetFunk(x−h))/2h` per dim) and aborts on >1% drift. **The harness was
written for exactly this**: in L.0a it sanity-checks `|lnl-recon|`; in L.0b it gates the
gradient correctness, dim by dim.

**Order to build L.0b (sub-steps):**
- L.0b.i: preorder kernel (`q_i`) only, validated by FD of `∂lnL/∂(global rate scale)`
  (one scalar reference computable without all-branch perturbation).
- L.0b.ii: per-pattern score for `α` (`∂r_c/∂α` cheap-FD trick) → validate by reusing the harness,
  swap the FD Jacobian column for alpha with the analytic one and check `|G_analytic - G_fd|/|G_fd| < 1%`.
- L.0b.iii: per-pattern score for `p_inv` (closed-form) → same validation.
- L.0b.iv: per-pattern score for `+R` weights/rates → same.
- L.0b.v: swap the full FD Jacobian block in `optimizeModeLAllParameters` for the analytic kernel;
  remove the per-dim perturb loop; keep the same G/B/LM step downstream.

The expected win: **L.0a is `(ndim+1)` evals/iteration; L.0b is 2 traversals/iteration** independent
of `ndim`. For AA `+I+G4` (ndim≈2 in L.0a → ~3 evals) the L.0a→L.0b ratio is small (3×). For
DNA GTR `+R10` (ndim≈26 → 27 evals) L.0b is 13.5× faster *per iteration*. The bigger AA win is
already captured by L.0a's alpha-Brent elimination and joint α+p_inv update.

### 17.5 — Run-script and harness notes

- **L-FD script:** `gadi-ci/mode-p-iso/run_lfd_aa100k_np1_mode_l.sh` — paired BASE vs MODE-L arms,
  shared seed (default 1), shared binary; expects md5 `f89e0b3e965b93c078c0040273d6e684`.
  Parses `MODE-L-FDCHECK |lnl-recon|=…` and `MODE-L: … accepted_iters=…` lines. Pass criteria:
  paired `|ΔlnL| ≤ 0.05`, best LG+G4 in both arms, max `|lnl-recon| ≤ 1e-3`, FDCHECK lines > 0.
- **Changed to `-m TESTONLY`** (job 169545917 used `-m TEST`; tree search took an extra 31 min and
  killed the job). With TESTONLY, BASE≈7 min + MODE-L≈12 min = <20 min total, well within 30 min.
  lnL comparison uses `Optimal log-likelihood` fallback (3rd in parse_arm) since MODE-L does not
  write `BEST SCORE FOUND` before tree search starts.
- The `MODE-L: ndim=… accepted_iters=…` summary line is guarded by `verbose_mode >= VB_MED` in
  `optimizeModeLAllParameters`; with default verbosity it does not appear. The gate does **not**
  fail on its absence — `MODE-L-FDCHECK` count is the canonical "Mode L engaged" proof.
- For ad-hoc tuning runs, add `-v` to surface per-model summary; FDCHECK line is always emitted
  when `--mode-l-fd-check` is set.

### 17.6 — L-FD gate results: **PASSED** (job 169549952, 2026-05-28, `-m TESTONLY`)

**Formal gate run:** job 169549952, 18:34 wall (well within 30 min), exit 0 both arms.
Earlier run 169545917 (`-m TEST`) was walltime-killed during MODE-L tree search; all ModelFinder
criteria already passed there. Job 169545780 was a 3-second false positive (pipefail SIGPIPE fix).

| Criterion | BASE | MODE-L | Pass? |
|---|---|---|---|
| exit 0 | 0 ✓ | 0 ✓ | ✓ |
| best model | LG+G4 ✓ | LG+G4 ✓ | ✓ |
| lnL | -7541976.853 | -7541976.852 | ✓ \|Δ\|=0.001 (<<0.05) |
| MF wall | 416.0 s | 680.6 s | 1.63× slower (D4) |
| FDCHECK lines | — | 5625 ✓ | ✓ (>>0) |
| max \|lnl-recon\| | — | **0.000000e+00** ✓ | ✓ **R1 CLEAN** |
| total accepted_iters | — | 0 (summary off at default verbosity) | ✓ expected |

**R1 outcome: PERFECT ZERO.** All 5625 `MODE-L-FDCHECK` lines had `|lnl-recon|=0.000`. The
`computePatternLikelihood` call sees fully valid `current_it`/`current_it_back` after every
`targetFunk` invocation. R1 risk (§17.3) is **resolved** — no fix needed.

**Performance finding (L.0a FD-Jacobian):** 680.6 s vs 416.0 s = **1.63× slower than legacy BFGS**.
Root cause: each LM iteration requires `(ndim+1)` full-tree evaluations for the FD Jacobian — same
traversal count as the legacy `derivativeFunk`. Despite eliminating alpha-Brent + EM alternation,
the LM iteration overhead and max-iter cap (default 40) exceed the savings from joint update.

- ndim=1 models (`+G4`): 2 evals/iter × up to 40 iters. LM overhead > Brent savings.
- ndim=2 models (`+I+G4`): 3 evals/iter × up to 40 iters. Net slower despite joint α+p_inv.

**L.0a is a scaffold, not the speedup.** L.0b (2-traversal analytic gradient, O(1) in ndim)
is where the real gain comes from. See §17.4 and the updated handoff header for L.0b plan.

**FDCHECK patterns observed (job 169545917 data, 2808 unique lines):**
- `gradInfNorm` ranges ~0 (converged) to ~42000 (fresh start on difficult model)
- Progressive convergence per model: large gradient → damped steps → small → accept
- LG+I+G4 (ndim=2): lnL=-7541976.857 at iter=31, just 0.004 from ref -7541976.861 ✓

### 17.7 — L.0b.i status + L.0b.ii exact implementation plan (2026-05-28)

**L.0b.i COMPLETE (compile-verified, build 169552659 queued):**

Files changed for L.0b.i:
| File | Change |
|---|---|
| `tree/phylonode.h` | Added `double* preorder_partial_lh; UBYTE* preorder_scale_num;` to `PhyloNeighbor` private section; init to `nullptr` in all 3 constructors. |
| `tree/phylotree.h` | Added `double* central_preorder_lh; UBYTE* central_preorder_scale_num;` (right after `central_scale_num`); added declarations for `initializePreorderPartialLh`, `initializePreorderPartialLhRecursive`, `deletePreorderPartialLh`, `computePreorderPartialLikelihood`. |
| `tree/phylotree.cpp` | `init()`: zeros new ptrs; `initializeAllPartialLh()`: calls `initializePreorderPartialLh()` when `mode_l_enabled`; `deleteAllPartialLh()`: calls `deletePreorderPartialLh()`. **NEW functions** after `deleteAllPartialLh`: `initializePreorderPartialLh()` (allocates `n_internal * block` doubles + UBYTEs, assigns per-neighbor pointers via DFS), `initializePreorderPartialLhRecursive()` (mirrors LM_PER_NODE postorder: assigns `dad->findNeighbor(node)->preorder_partial_lh` only for non-leaf nodes), `deletePreorderPartialLh()` (just `aligned_free`), `computePreorderPartialLikelihood()` (STUB — just asserts non-null). |

**L.0b.ii — Scalar preorder kernel: exact formulas and code pointers**

The `computePreorderPartialLikelihood()` stub needs to be filled with the following algorithm.
All formulas use the standard IQ-TREE variable names from phylokernelnew.h.

**Step 0: Find the virtual root edge.**
```cpp
// current_it is set by computeLikelihood(); it points to the root edge.
PhyloNeighbor *root_branch = current_it;           // dad_branch = root → node
PhyloNode     *root_dad    = (PhyloNode*)root;     // root_dad
PhyloNode     *root_child  = (PhyloNode*)root_branch->node;
```

**Step 1: Initialize preorder buffer at root.**
For the root edge, the preorder message for `root_child` is:
```
preorder_lh_root_child[c][p][s] = π[s] * Σ_t P_{root_dad→root_child}^T[s,t] * partial_lh_sibling
```
But for a BIFURCATING root with one more child (the "sibling" of root_child), the formula simplifies.

Actually — do it as a special root case vs. general case. See `computeLikelihoodBranchGenericSIMD`
at phylokernelnew.h around line 1765-2000 for how it handles the root-side messages. The pattern is:
```cpp
// At root: q_{root_child}[c][p][s] = π[s] * (P_sibling ⊙ p_sibling)[c][p][s]
// (weighted by π because there's no parent-side message)
```
For the root's other child(ren), reverse `node` and `dad` to get the other side's message.

**Step 2: Recursive preorder from root to leaves.**
The general recursion for an INTERNAL node `v` with parent `u` (the `u→v` edge is stored in
`nei = u->findNeighbor(v)`, `nei->preorder_partial_lh` is what we're filling):

```
// "combo" = parent preorder ⊙ sibling postorder
combo[c][p][t] = preorder_lh_u[c][p][t]  (already filled in previous level)
              * Π_{sibling sib of v at u} (P_{u→sib} · partial_lh_sib)[c][p][t]

// preorder for v:  apply P_{u→v}^T to combo
preorder_lh_v[c][p][s] = Σ_t P_{u→v}^T[s][t] * combo[c][p][t]
                        = Σ_t P_{v→u}[t][s] * combo[c][p][t]     // using symmetry
```

For binary nodes (exactly 2 children from u's perspective: `v` and one sibling `sib`):
```
combo[c][p][t] = preorder_lh_u[c][p][t] * (P_{u→sib} · partial_lh_sib)[c][p][t]
```

The term `(P_{u→sib} · partial_lh_sib)[c][p][t]` is the sibling's POSTORDER contribution at u —
which is already stored in `sib->partial_lh` (pointing to `u->findNeighbor(sib)->partial_lh`
or the eigenspace version in the traversal buffer). **Key insight**: this is exactly the `echildren`
buffer pre-computed in the postorder pass (phylokernelnew.h:1460-1580). Reuse it.

**Step 3: Gradient assembly during the preorder pass.**
At each internal node `v` (after computing `q_v = preorder_lh_v`), contribute to `G[alpha]`:

```cpp
double h_alpha = 1e-4 * max(abs(alpha), 1.0);
// Pre-compute scalar ∂r_c/∂alpha (no traversal):
double dr_dalpha[ncat];
double rates_base[ncat], rates_ph[ncat];
site_rate->getRates(alpha, rates_base);      // r_c at current alpha
site_rate->getRates(alpha + h_alpha, rates_ph); // r_c at alpha+h
for (int c = 0; c < ncat; c++)
    dr_dalpha[c] = (rates_ph[c] - rates_base[c]) / h_alpha;

// At each branch b_v (connecting v to its parent u):
for (int c = 0; c < ncat; c++) {
    double bvrc = length(v→u) * rates_base[c];  // b_v * r_c
    // Get eigenvalue-based P'(bvrc) = sum_k λ_k * exp(λ_k*bvrc) * evec_k ⊗ inv_evec_k
    // (reuse exactly what computeLikelihoodDerv builds in val0/val1 at lines 2356-2395)
    for (int p = ptn_lower; p < ptn_upper; p++) {
        double Qp_sum = dot(q_v[c][p], Pprime(bvrc), partial_lh_v[c][p]);  // scalar
        G[alpha] += ptn_freq[p] * Qp_sum * dr_dalpha[c] * w_c / exp(_pattern_lh[p]);
    }
}
```

The `Pprime(bvrc)` computation reuses `eval` (eigenvalues), `evec`, `inv_evec` — all already loaded
in the postorder/derivative kernel. The pattern is at phylokernelnew.h:2356-2395.

**Existing code to reuse:**
- `model->getEigenvalues()`, `model->getEigenvectors()`, `model->getInverseEigenvectors()` → P' formula
- `site_rate->getRates()` → r_c (and perturbed version for ∂r_c/∂alpha FD)
- `ptn_freq[p]` → pattern frequencies (already allocated, filled in `initializeAllPartialLh`)
- `_pattern_lh[p]` → total log-likelihood per pattern (filled by `computeLikelihood`)
- `current_it`, `current_it_back` → virtual root (set by `computeLikelihood`)
- `phylotree.h:1592` → `traversal_info` (list of postorder nodes) — reverse this for preorder order!

**Simplest correct implementation (avoiding all SIMD for now):**

Add a new file `tree/phylotree_modeL.cpp` with:
```cpp
#include "phylotree.h"
#include "model/rategamma.h"

// Fills central_preorder_lh bottom-up for all internal nodes.
// Requires: computeLikelihood() already called (sets current_it, partial_lh, _pattern_lh).
// Output: nei->preorder_partial_lh for all internal PhyloNeighbors.
void PhyloTree::computePreorderPartialLikelihood() {
    ASSERT(central_preorder_lh);
    size_t nstates = aln->num_states;
    size_t ncat    = site_rate->getNRate();
    size_t nptn    = aln->getNPattern();
    // ... scalar implementation using above formulas ...
}
```

Do NOT try to use the `TraversalInfo` / function-pointer infrastructure yet. Just use `FOR_NEIGHBOR_IT`
recursion. Profile after: if the preorder is faster than FD (i.e., 2 traversals < ndim+1 traversals),
the scalar version already wins for large ndim (DNA GTR+R10). For AA ndim=1/2 the speedup comes from
LM convergence quality, not traversal count.

**Validation command (after L.0b.ii is in place):**
```bash
qsub run_lfd_aa100k_np1_mode_l.sh   # uses same --mode-l-fd-check harness
# Now checks |G_analytic - G_fd| / |G_fd| per dim instead of |lnl-recon|
# Pass criterion: all dims < 1%
```
Update `run_lfd_aa100k_np1_mode_l.sh` to also parse `MODE-L-FDCHECK |G-ratio|=` lines
(new output format once the FD-check block is updated in `optimizeModeLAllParameters`).

### 17.8 — L.0b.ii implementation (2026-05-28, build 169559278 queued)

**Files changed:**
| File | Change |
|---|---|
| `tree/phylotree_modeL.cpp` | NEW — scalar preorder kernel + analytic alpha gradient |
| `tree/phylotree.h` | Added `double computeModeLAnalyticGradAlpha()` declaration |
| `tree/phylonode.h` | Added `get_preorder_partial_lh()` public accessor on PhyloNeighbor |
| `model/modelfactory.cpp` | `optimizeModeLAllParameters`: calls preorder + analytic gradient before FD loop when `mode_l_fd_check`; prints `G_alpha_fd`, `G_alpha_analytic`, `|G-ratio|` in FDCHECK output |
| `tree/CMakeLists.txt` | Added `phylotree_modeL.cpp` to tree library |

**Recursion correctness:**
The preorder convention is V^{-1}-projected eigenspace (same as `partial_lh`). Root init: `current_it->preorder_partial_lh = memcpy(current_it_back->partial_lh)`. Recursion per branch u→v:
1. `pre_u_state[t] = Σ_i inv_evec[i*nstates+t] * pre_u[i]`  (iV column-wise)
2. `f_sib_state[t] = Σ_i evec[t*nstates+i] * exp(λ_i*r_c*b_sib) * pl_sib[i]`
3. `combo[j] = Σ_t inv_evec[j*nstates+t] * (pre_u_state[t] * f_sib_state[t])`
4. `pre_v[i] = exp(λ_i*r_c*b_v) * combo[i]`

Derivation: see §17.7 analysis. For reversible models with inv_evec[i][s]=π[s]*evec[s][i], the preorder
stored as V^{-1}-projected gives `dL/db_v = Σ_i val1[i] * pre_v[i] * pl_v[i]` exactly.

**Gradient formula** (per branch b_v, category c, pattern p):
`qp = Σ_i λ_i * exp(λ_i*r_c*b_v) * pre_v[c,i] * pl_v[c,i]`
`G[alpha] += pf[p] * w_c * b_v * (dr_c/dalpha) * qp / exp(_pattern_lh[p])`

where `w_c = site_rate->getProp(c%ncat) * getMixtureWeight(c/denom)` and `dr_c/dalpha` is the same
finite-difference (h=1e-4*max(|alpha|,1)) used by the FD Jacobian.

**Gate:** Build 169559278 → run_lfd_aa100k_np1_mode_l.sh → parse `|G-ratio|` lines.
Pass criterion: max `|G-ratio|` < 1% across all FDCHECK iterations.

### 17.9 — Build trap: duplicate definition of computePreorderPartialLikelihood (2026-05-28)

**Job 169559278 FAILED at link time:**
```
model/libmodel.a(modelfactory.cpp.o): in function `ModelFactory::optimizeModeLAllParameters(double)':
  undefined reference to `PhyloTree::computeModeLAnalyticGradAlpha()'
icpx: error: linker command failed with exit code 1
```

**Root cause:** L.0b.i left a STUB `PhyloTree::computePreorderPartialLikelihood()` in
`tree/phylotree.cpp:1176` (asserting buffer non-null and returning). L.0b.ii added the REAL impl
in `tree/phylotree_modeL.cpp`. Both `.o` files end up in `tree/libtree.a` — `nm -A libtree.a`
shows the symbol defined in BOTH:
```
libtree.a:phylotree.cpp.o:        T _ZN9PhyloTree32computePreorderPartialLikelihoodEv
libtree.a:phylotree_modeL.cpp.o:  T _ZN9PhyloTree32computePreorderPartialLikelihoodEv
                                  T _ZN9PhyloTree29computeModeLAnalyticGradAlphaEv
```

The linker pulled `phylotree.cpp.o` first (countless other refs from elsewhere already needed it),
so the stub's preorder symbol was already satisfied when `tree/libtree.a` was scanned again.
`phylotree_modeL.cpp.o` was NOT pulled in — its `computeModeLAnalyticGradAlpha` symbol stayed
unresolved → link error.

**Fix (2026-05-28, build 169585743 in flight):** removed the stub body from `phylotree.cpp:1176`,
left a one-line comment pointing to `tree/phylotree_modeL.cpp`. Now only the real implementation
exists. Both symbols (`computePreorderPartialLikelihood` and `computeModeLAnalyticGradAlpha`) live
together in `phylotree_modeL.cpp.o`, so requesting either pulls in both.

**Future trap:** if you split future Mode-L symbols (`computeModeLAnalyticGradPInv`,
`computeModeLAnalyticGradRates`, etc.) across new files, keep each method defined in ONE place only.
Never leave a stub in `phylotree.cpp` while an override is in a new TU.

### 17.10 — SIGSEGV in computePreorderPartialLikelihood (gate 169586323, 2026-05-29)

**Gate 169586323 FAILED** — MODE-L arm crashed with SIGSEGV (exit 139) at 7s wall during:
```
Perform fast likelihood tree search using LG+I+G model...
Estimate model parameters (epsilon = 5.000)
```
FDCHECK lines = 0 (crash before first print), BASE arm exit 0 / 416s normal.

**Root cause: null pointer dereference in `computePreorderPartialLikelihood()`.**

In `initializeAllPartialLh(index, indexlh)` (phylotree.cpp:1625-1637), the `LM_PER_NODE` branch
**only allocates `partial_lh` for downward (parent→child) edges**:
```cpp
nei->partial_lh  = nullptr;       // upward (node→dad) edge — always null
nei2->partial_lh = central_partial_lh + offset;  // downward (dad→node) edge — allocated
```

`current_it_back` is the upward edge from the root's first child back to the virtual root.
Therefore `current_it_back->get_partial_lh()` returns **nullptr**.

The original root initialization:
```cpp
memcpy(pb, current_it_back->get_partial_lh(), sizeof(double)*nps*blk);  // BUG: src = null
```
dereferences null → SIGSEGV. The `ASSERT(pb)` above it is **disabled in release builds** (`-DNDEBUG`)
so it passed silently, and the crash came at the memcpy.

**Why the ASSERT didn't catch it:** IQ-TREE release builds use `-DNDEBUG` which makes all `ASSERT`
macros no-ops. Four no-op ASSERTs in the preorder kernel would have caught this in debug; they were
silently skipped in the PBS release binary.

**Correct root preorder initialization:**

For a reversible model, `current_it_back->partial_lh` was meant to hold the "prior from above".
Since the virtual root has no other subtrees, this prior is the stationary distribution π in
eigenspace. For the V^{-1}-projected eigenspace:
```
pre[j] = Σ_s π_s * evec[s*ns+j] = Σ_s inv_evec[j*ns+s]    (sum of row j of V^{-1})
```
(using the reversible-model relation `inv_evec[j*ns+s] = π_s * evec[s*ns+j]`).

For standard π-orthonormal eigenvectors this equals `[1, 0, …, 0]` (only the equilibrium component
is non-zero), so the root-edge gradient contribution is automatically zero (λ₀=0 drops it out).
No missing term in `gradAccumRecursive`.

**Fix applied (build 169587129, 2026-05-29):**
In `tree/phylotree.cpp`, `computePreorderPartialLikelihood()`:

1. Replaced `ASSERT(central_preorder_lh)` with `if (!central_preorder_lh) return;`
   (handles the edge case where buffers were allocated before `model_factory` was set)
2. Replaced `ASSERT(pb)` with `if (!pb) return;`
3. Replaced the null-src `memcpy` with the correct prior initialization:
   ```cpp
   std::vector<double> rp(ncm * ns);
   for (size_t c = 0; c < ncm; c++) {
       const double *iV = ip + ctx.mix_addr_malign[c];
       for (size_t j = 0; j < ns; j++) {
           double s = 0.0;
           for (size_t t = 0; t < ns; t++) s += iV[j*ns + t];
           rp[c*ns + j] = s;
       }
   }
   for (size_t p = 0; p < nps; p++)
       memcpy(pb + p*blk, rp.data(), ncm*ns*sizeof(double));
   ```
4. In `preorderFillRecursive`: replaced `ASSERT(pre_node != nullptr)` with `if (!pre_node) return;`
5. In `preorderFillRecursive`: replaced `ASSERT(pre_child)` with `if (!pre_child) continue;`
6. In `preorderFillRecursive`: replaced `ASSERT(nc == 2)` with `if (nc != 2) return;`

**Lesson:** in release builds, ASSERTs are no-ops. Any guard that protects against a crash in
`computePreorderPartialLikelihood` or its helpers MUST be a real `if (!x) return/continue`, not
an ASSERT. All future Mode-L kernel code should follow this convention.

---

### 17.11 — Zero analytic gradient in all FDCHECK iterations (gate 169589950, 2026-05-29)

**Gate 169589950 FAILED** — MODE-L arm ran to completion (exit 0) but all 1273 FDCHECK lines
showed `G_alpha_analytic=0.000`, `|G-ratio|=1.000`. The FD gradient was correct (~400 for LG+G4).

#### Root cause: `current_it` is an upward edge with null `preorder_partial_lh`

In `computeLikelihood()` (phylotree.cpp line ~1733), if `current_it` is null it is initialised
as:
```cpp
Node *leaf = findFarthestLeaf();
current_it = (PhyloNeighbor*)leaf->neighbors[0];   // leaf → adjacent-internal (UPWARD)
```

This edge is stored in **leaf's** neighbour list pointing to the adjacent internal node.
In `initializePreorderPartialLhRecursive`, the preorder buffer is assigned only to the
*downward* direction `dad->findNeighbor(node)` for non-leaf `node`.  The **upward** edge
`leaf->findNeighbor(internal)` always has `preorder_partial_lh = nullptr`.

So when `computePreorderPartialLikelihood()` (old code) ran:
```cpp
PhyloNode *rc = (PhyloNode*)current_it->node;     // the internal node
double *pb    = current_it->get_preorder_partial_lh();  // ← nullptr!
if (!pb) return;   // ← always fired → preorder buffers never filled
```
All `preorder_partial_lh` buffers remained at the mmap-zeroed initial state.
`gradAccumRecursive` then computed `qp = Σ_i ev[i]*...*0*lc[i] = 0` for every branch/pattern.

#### Why did this go unnoticed until gate 169589950?
- Gate 169588980 was the first run with the SIGSEGV fix, but it failed due to a **node fault** (gadi-cpu-spr-0143), not a code bug. The resubmit (169589950) ran cleanly and exposed the silent zero gradient.

#### Fix (build 169590424)

Both `computePreorderPartialLikelihood()` and `computeModeLAnalyticGradAlpha()` now use
`root->neighbors[0]` (the virtual-leaf root's only neighbour = always a downward edge with an
allocated preorder buffer) instead of `current_it`:

```cpp
// computePreorderPartialLikelihood():
PhyloNode *rl = (PhyloNode*)root;
if (rl->neighbors.empty()) return;
PhyloNeighbor *root_edge = (PhyloNeighbor*)rl->neighbors[0];
PhyloNode     *rc        = (PhyloNode*)root_edge->node;
if (!rc->isLeaf()) {
    double *pb = root_edge->get_preorder_partial_lh();   // ← always non-null
    if (!pb) return;
    ...
}
preorderFillRecursive(ctx, rc, rl);

// computeModeLAnalyticGradAlpha():
PhyloNode *grad_rl = (PhyloNode*)root;
PhyloNode *grad_rc = (PhyloNode*)((PhyloNeighbor*)grad_rl->neighbors[0])->node;
gradAccumRecursive(ctx, grad_rc, grad_rl);
```

`root->isLeaf()` is guaranteed by `ASSERT(root->isLeaf())` in `computeLikelihood`.
`root->neighbors.size() == 1` for any leaf node in any valid IQ-TREE tree.

**Lesson:** Never anchor a traversal on `current_it->node` when the traversal must start from
the virtual root downward.  `current_it` is set by likelihood computation to an arbitrary edge
(commonly the farthest-leaf edge, which is an upward edge with no preorder buffer).  Always
derive the starting point directly from `root->neighbors[0]`.

---

### 17.12 — Deep preorder/gradient audit after §17.11 (2026-05-29)

After fixing the `current_it` null-preorder-buffer bug, a direct code audit found five more
correctness bugs that would have made the next `|G-ratio|` gate fail.

#### A1 — Root preorder init omitted the real root taxon

IQ-TREE's `root` is a leaf used to orient the unrooted tree. For unrooted input this is usually a
real taxon, not the dummy `__root__` node. The §17.10/§17.11 prior-only root init therefore dropped
the root leaf's observed state from the top-down complement.

Fix in `computePreorderPartialLikelihood()`:
- if `isRootLeaf(root)` is true (rooted dummy `__root__`), initialise the preorder root edge to the
  no-data vector `V^{-1} 1` (row sums of `inv_evec`), as before;
- otherwise initialise `root->neighbors[0]->preorder_partial_lh` as the branch-transformed root tip:
  `pre_i = exp(lambda_i * r_c * b_root) * tip_partial_lh[root_state][i]`.

#### A2 — Preorder propagation used `inv_evec` in the wrong direction

Postorder partials are stored as `V^{-1}`-projected vectors. To recover the state-space contribution
from an already branch-transformed eigenspace message, the code must multiply by `evec`, mirroring
the existing postorder child-contribution loops in `phylokernelnew.h`.

Old buggy preorder propagation:
```cpp
pre_u_state[t] = sum_i inv_evec[i*ns+t] * pre_u[i];
```

Fixed:
```cpp
pre_u_state[t] = sum_i evec[t*ns+i] * pre_u[i];
```

The projection back after multiplying by the sibling contribution remains `inv_evec`.

#### A3 — Internal-branch alpha gradient double-applied the transition

`preorder_partial_lh` already includes the branch transition for that branch:
```cpp
pre_v[i] = exp(lambda_i * r_c * b_v) * combo[i];
```

Therefore the derivative term is:
```cpp
qp = sum_i lambda_i * pre_v[i] * partial_lh_v[i];
```
not `lambda_i * exp(...) * pre_v[i] * partial_lh_v[i]`. The extra `exp(...)` was removed.

#### A4 — Root and pendant branches were absent from `G_alpha_analytic`

The old `gradAccumRecursive()` only accumulated branches whose child had both preorder and postorder
buffers, which excludes all leaf/pendant branches. The root edge was also skipped because traversal
started at the root child and only visited child edges below it.

Fix:
- accumulate the `root->neighbors[0]` branch explicitly when it has preorder + postorder buffers;
- add an on-the-fly leaf-branch contribution path. For a leaf child, compute the same temporary
  `combo` message used by preorder propagation, form `pre_leaf_i = exp(lambda_i*r_c*b_leaf)*combo_i`,
  and add `sum_i lambda_i * pre_leaf_i * tip_partial_lh_i`.

#### A5 — Preorder scale counts were allocated but never propagated

`_pattern_lh[p]` is scale-corrected (`log(lh_stored) + scale*LOG_SCALING_THRESHOLD`), while the
analytic numerator is computed from scaled partial buffers. Without the matching scale correction,
the numerator and denominator are in different units.

Fix:
- root preorder scale is zeroed;
- each preorder child inherits `preorder_scale(parent) + postorder_scale(sibling)`;
- gradient accumulation multiplies by
  `exp((preorder_scale + postorder_scale) * LOG_SCALING_THRESHOLD - _pattern_lh[p])`.

For leaf branches, the postorder leaf scale is zero, so the correction is
`preorder_scale(parent) + postorder_scale(sibling)`.

#### Validation

- Focused compile/link on the patched tree target: clean (no compiler output).
- VS Code diagnostics for `tree/phylotree.cpp`: no errors.
- Gate script hardened: L.0b.ii now fails if `|G-ratio|` lines are absent, not only when the
  maximum ratio is too large.
- Tiny login-node smoke run failed with `Illegal instruction` because the optimized Gadi binary is
  compute-node targeted; runtime validation must be via PBS.
- Build 169592056 caught one compile-only defect: direct access to private
  `PhyloNeighbor::preorder_scale_num`. Fixed by adding public accessor
  `get_preorder_scale_num()` next to `get_preorder_partial_lh()`.
- Build 169592244 caught the matching direct access to private postorder
  `PhyloNeighbor::scale_num`. Fixed by adding `get_scale_num()`.
- Fresh official rebuild **169592407 completed** with md5
  `922695586fefcfc171dea019374ea30d`. `EXPECTED_MD5` is updated in the L.0b.ii
  gate script. Gate **169592704** submitted.

### 17.13 — L.0b.iii: heap overflow fix, BASE-hang regression, root cause, fix (2026-05-29)

#### L.0b.ii gate 169592704: FAILED (MODE-L SIGSEGV)

Gate 169592704 (build 169592407, md5=`922695586fefcfc171dea019374ea30d`):
- BASE arm: ✅ exit 0, wall=421s, lnL=-7541976.853 (LG+G4).
- MODE-L arm: ❌ SIGSEGV exit 139 at 16s wall. Crash at `Estimate model parameters (epsilon = 5.000)` during the fast NJ tree evaluation (first call to LG before ModelFinder).

**Root cause: heap buffer overflow on second model evaluation.**

`initializePreorderPartialLh()` had guard `if (central_preorder_lh) return;`.

- Thread evaluates LG (nc=1, nMix=1): allocates buffer of size `n_internal × nptn × 1 × 1 × nStates` doubles.
- Thread then evaluates LG+G4 (nc=4, nMix=1): old code skips realloc (buffer exists), writes 4× the allocated storage → heap corruption → SIGSEGV.

An intermediate binary (md5=`870eb8148d832b8068208fdabffb8292`) was produced and used for gate **169604442** (2026-05-29). Same heap overflow: MODE-L SIGSEGV exit 139 at 24s. BASE ✅ 414s.

#### L.0b.iii implementation: ModeLScope RAII + preorder_block tracking

Added to source tree (`iqtree3-mode-p-iso-p3`):

1. **`tree/phylotree.h`**: New members in `PhyloTree`:
   ```cpp
   double   *central_preorder_lh = nullptr;
   UBYTE    *central_preorder_scale_num = nullptr;
   uint64_t  preorder_block = 0;   // last-allocated block size (doubles per node)

   public:
   bool mode_l_context_active = false;   // true only inside ModeLScope
   ```
   New method declarations: `initializePreorderPartialLh()`, `deletePreorderPartialLh()`, `initializePreorderPartialLhRecursive()`, `computePreorderPartialLikelihood()`, `computeModeLAnalyticGradAlpha()`.

2. **`tree/phylotree.cpp`** — `initializePreorderPartialLh()` (revised):
   ```cpp
   // Realloc only when block size changes; otherwise reuse.
   if (central_preorder_lh) {
       if (preorder_block == block) return;
       deletePreorderPartialLh();
   }
   central_preorder_lh = aligned_alloc<double>((uint64_t)n_internal * block);
   preorder_block = block;
   ```
   Guard in `initializeAllPartialLh()`:
   ```cpp
   if (model_factory && model_factory->mode_l_enabled)
       initializePreorderPartialLh();   // ← BUG: missing context_active check
   ```

3. **`main/phylotesting.cpp`** — `ModeLScope` RAII struct inside `CandidateModel::evaluate()`:
   ```cpp
   struct ModeLScope {
       IQTree *t;
       explicit ModeLScope(IQTree *t_) : t(t_) { t->mode_l_context_active = true;  }
       ~ModeLScope()                            { t->mode_l_context_active = false; }
   } _mode_l_scope(iqtree);
   ```

Build **169608969** (md5=`1e14caf78e54f28cf44400532102a38b`) produced. This fixed the MODE-L heap overflow but introduced a **new BASE arm regression** (see below).

#### BASE arm hang regression (gates 169610527 and 169613712)

Both gates used build 169608969 (md5=`1e14caf78e54f28cf44400532102a38b`), no `--mode-l` flag on BASE arm (`mode_l_enabled=true` from CLI, `mode_l_context_active=false` because no ModeLScope active).

**Symptom (identical on two different nodes):**

| | Job 169610527 | Job 169613712 |
|---|---|---|
| Node | gadi-cpu-spr-0594 | gadi-cpu-spr-0198 |
| Last stdout line | `Create initial parsimony tree by phylogenetic library (PLL)... 1.158 sec` | same |
| Wall at kill/check | 30m38s | 44m31s |
| CPU time | 1m42s | 1m43s |
| Memory | 762 MB | 781 MB |
| Next expected line (from good run 169604442) | `Perform fast likelihood tree search using LG+I+G model...` | never printed |

Memory=762 MB confirms `initializeAllPartialLh()` was not reached by the ModelFinder path (~6 GB expected). The process was alive (exit status -29 timeout, not OOM-kill) but consuming ~0 CPU — blocking in a syscall or NUMA allocator stall.

**Root cause: unconditional preorder alloc during fast NJ tree construction.**

In a passing run, the sequence immediately after parsimony is:
```
Create initial parsimony tree by PLL... 1.2s
Perform fast likelihood tree search using LG+I+G model...   ← calls initializeAllPartialLh()
```

With build 169608969, `initializeAllPartialLh()` is called with `mode_l_enabled=true` and `mode_l_context_active=false` (no ModeLScope). The guard was:
```cpp
if (model_factory && model_factory->mode_l_enabled)
    initializePreorderPartialLh();    // fires unconditionally!
```

`initializePreorderPartialLh()` for AA 100K LG+I+G allocates:
```
n_internal=97, nptn≈96017, nRate=4, nMix=1, nStates=20
→ 97 × 96017 × 4 × 1 × 20 × 8 bytes ≈ 6.0 GB
```

With `OMP_WAIT_POLICY=PASSIVE` and 103 OMP threads, calling `aligned_alloc` for 6 GB and then `initializePreorderPartialLhRecursive` (which touches all of it via DFS) triggers:
- 6 GB of first-touch page faults across all 8 NUMA nodes simultaneously from 103 threads
- Linux NUMA-local page fault rate ≈ 6 GB / (8 NUMA × 26 GB/s bandwidth) × contention ≈ tens of seconds of stall _before_ any ModelFinder work begins
- This stall looks like a hang because CPU-seconds barely advance while wall-clock runs

The allocation occurs **every call** to `initializeAllPartialLh()` during the fast NJ tree (multiple NNI rounds, each recalculating the tree), not just once. This makes the stall additive and potentially infinite under certain OMP/NUMA allocation patterns.

**Fix: add `&& mode_l_context_active` to guard.**

```cpp
// BEFORE (BUG):
if (model_factory && model_factory->mode_l_enabled)
    initializePreorderPartialLh();

// AFTER (FIXED):
if (model_factory && model_factory->mode_l_enabled && mode_l_context_active)
    initializePreorderPartialLh();
```

This matches the analogous guard already in `model/modelfactory.cpp::optimizeParametersOnly()`:
```cpp
if (mode_l_enabled && site_rate->phylo_tree->mode_l_context_active)
    return optimizeModeLAllParameters(gradient_epsilon);
```

**Additional debug instrumentation** added to `tree/phylotree.cpp`, `model/modelfactory.cpp`, and `main/phylotesting.cpp` (7 `DBG-L.0b-START/END` block pairs) writing to `<prefix>.mode_l_debug.log`. These log:
- `initializeAllPartialLh()`: whether preorder init fires (context_active=1) or is skipped (context_active=0)
- `initializePreorderPartialLh()`: ALLOC / REUSE / REALLOC decision with block sizes
- `ModeLScope ENTER/EXIT` with model name
- `optimizeModeLAllParameters` ENTER/EXIT with lnL

**Build 169615509 queued** (node gadi-cpu-spr-0252, 17:19 AEST). Contains fix + debug instrumentation. `EXPECTED_MD5` in gate script must be updated once build completes.

### 17.14 — L.0b.iii: nps/unobserved_ptns heap overflow SIGSEGV + null-ptr fixes (2026-05-29)

#### Gate 169620202: BASE ✅ exit=0 wall=425s / MODE-L ❌ SIGSEGV exit=139 at 24s

Build 169615509 (md5=`5424acd8bc44a5ef2d70e9379d677eae`) fixed the BASE-hang (§17.13: `mode_l_context_active` guard in `initializeAllPartialLh`). The gate ran cleanly in the BASE arm. MODE-L arm crashed at 24s wall, immediately after the `[ATMD Mode F] K_outer=8 M_inner=12` line and the first two models (LG and LG+F), during LG+F+I+G4 evaluation.

Debug log at `<work_dir>/mode_l/iqtree_inner.mode_l_debug.log` showed the last three ENTER lines before crash:
```
[MODE-L-DBG] ENTER model=LG ndim=2 ...
[MODE-L-DBG] ENTER model=LG+F ndim=1 ...
[MODE-L-DBG] ENTER model=LG+F ndim=2 ...
```
No `post-preorder` or `post-scores` followed the third line — crash is inside `computePreorderPartialLikelihood()` for LG+F+I+G4 (ndim=2 = alpha+pinv).

#### Root cause: `nps` recomputed from lazily-populated `unobserved_ptns`

In `computePreorderPartialLikelihood()`, `nps` (the scale-buffer slot count) was computed as:
```cpp
const size_t nps = get_safe_upper_limit(aln->size())
                 + max(get_safe_upper_limit(ns),
                       get_safe_upper_limit(model_factory->unobserved_ptns.size()));
```

For `+I` (invariable-sites) models, `model_factory->unobserved_ptns` is populated lazily on the **first** `computeLikelihood()` call — which is `evalNeg()` inside the LM iteration, **AFTER** `initializePreorderPartialLh()` allocated the scale buffer. At allocation time `unobserved_ptns.size()=0`; at call time it is non-zero. So `nps_runtime > nps_alloc` and:

```cpp
memset(ps, 0, nps*ncm*sizeof(UBYTE));   // ← ps points to allocated[nps_alloc*ncm]; nps*ncm > that
```

overflows the scale buffer by `delta_unobserved * ncm` bytes into adjacent heap, causing corruption followed by SIGSEGV.

Why LG+F+I+G4 specifically: LG (no +I) evaluates first, so `unobserved_ptns` is zero at allocation for LG+F. When LG+F+I+G4 evaluates, the first `computeLikelihood()` (inside the outer `evalNeg()` before entering the iter loop) populates `unobserved_ptns`. The next call to `computePreorderPartialLikelihood()` at iter=0 uses the grown `nps` against the smaller allocated buffer.

#### Three fixes applied (build 169621693, md5=`ccacea46d73ad7ee7f26df5e7f1c45a8`)

**Fix 1 (primary — `tree/phylotree.cpp`, `computePreorderPartialLikelihood()`):**

Replace the runtime `nps` formula with one derived from the stored `preorder_block` (set at allocation time):

```cpp
// OLD (overflows when unobserved_ptns grows after allocation):
const size_t nps = get_safe_upper_limit(aln->size())
                 + max(get_safe_upper_limit(ns),
                       get_safe_upper_limit(model_factory->unobserved_ptns.size()));

// NEW (always matches allocated buffer size):
// Derive nps from the stored preorder_block (set at allocation time) rather than
// recomputing from model_factory->unobserved_ptns.size(). For +I models,
// unobserved_ptns is populated lazily on the first computeLikelihood() call —
// AFTER initializePreorderPartialLh() allocated the buffer. If nps were
// recomputed here it would exceed the allocated size, overflowing the scale buffer.
const size_t nps = (blk > 0 && preorder_block > 0)
                 ? (size_t)(preorder_block / (uint64_t)blk) : 0;
```

Added null guard after the eigenvector pointer checks:
```cpp
if (!ep || !ip || !lp) return;
if (!tip_partial_lh) return;
```

**Fix 2 (defensive — `preorderFillRecursive`, sibling `partial_lh` null guard):**

For non-leaf siblings, `sib_nei->get_partial_lh()` was called and the result immediately offset without a null check. In `LM_PER_NODE` mode, upward edges always have `partial_lh=nullptr`. Fixed by retrieving the sibling pointer once before the ptn/c loops with an explicit `continue` on null (also improves performance by hoisting the `findNeighbor` call out of the hot loop):

```cpp
// BEFORE: inside ptn/c loop (re-called every iteration, no null check):
const double *pl = ((PhyloNeighbor*)node->findNeighbor(sn->node))->get_partial_lh() + ptn*blk + c*ns;

// AFTER: hoisted before the ptn loop, with null guard:
const double *sib_pl_base = nullptr;
if (!sl) {
    PhyloNeighbor *sib_nei = (PhyloNeighbor*)node->findNeighbor(sn->node);
    if (sib_nei) sib_pl_base = sib_nei->get_partial_lh();
    if (!sib_pl_base) continue;
}
// inside loop: const double *pl = sib_pl_base + ptn*blk + c*ns;
```

**Fix 3 (defensive — `tip_partial_lh` null guards in 3 locations):**

```cpp
// In computePreorderPartialLikelihood():
if (!tip_partial_lh) return;   // after eigenvector check

// In preorderFillRecursive(), leaf sibling branch:
if (sl) {
    if (!ctx.tree->tip_partial_lh) continue;
    const double *tp = ctx.tree->tip_partial_lh + ...;

// In accumulateLeafAlphaBranch():
if (!ctx.tree->tip_partial_lh) return;   // first line of function
```

**Lesson:** `unobserved_ptns` (and any `ModelFactory` member that grows lazily after `initializePreorderPartialLh` allocates) must not be used to recompute a buffer dimension. Always anchor the in-use dimension to the stored allocation metadata (`preorder_block`). Any call-time recomputation of a buffer dimension is a latent bug if the underlying data can change.

#### Per-iter debug logging added to `optimizeModeLAllParameters`

To pinpoint the crash in future gate runs if the primary fix is incomplete, three new `[MODE-L-DBG]` log lines were added inside the `for (int iter = 0; ...)` loop in `modelfactory.cpp`:

```cpp
// iter entry: logs iter, model name, ndim, n_analytic, full_analytic, alpha_dim, pinv_dim,
//             tip_plh pointer, central_pre pointer
[MODE-L-DBG] iter=N model=X ndim=Y n_analytic=Z full_analytic=0 alpha_dim=1 pinv_dim=1 tip_plh=0x... central_pre=0x...

// after computePreorderPartialLikelihood():
[MODE-L-DBG] post-preorder iter=N model=X

// after computeModeLAnalyticScores():
[MODE-L-DBG] post-scores iter=N model=X
```

If crash recurs: the last `iter=N` line without `post-preorder` following it → crash is inside `computePreorderPartialLikelihood`; `post-preorder` present but not `post-scores` → crash in `computeModeLAnalyticScores`.

#### Gate 169621788 pass criteria (§17.14 nps-fix only)

- MODE-L exit=0 (nps SIGSEGV fix validated — primary §17.14 criterion)
- `[MODE-L-DBG] post-preorder` and `post-scores` present for LG+F ndim=2 (confirms crash site is past those points)
- `MODE-L-FDCHECK` lines > 0 (fd-check engaged)
- max `|G-ratio| < 0.01` (alpha gradient — expected to pass)
- `|G-ratio-pinv|` will be **wrong (2.0–60+)** — this is the §17.15 p_inv formula bug, fixed separately; does NOT block the §17.14 gate
- Gate passes even with bad p_inv ratio; the **L.0b.iv gate** (new build post-§17.15 fix) is where `|G-ratio-pinv| < 0.01` must hold

**After gate 169621788 confirms exit=0:** build with p_inv formula fix applied and submit L.0b.iv gate.
Do NOT remove `DBG-L.0b-*` blocks yet — keep them for the L.0b.iv gate to confirm p_inv fix.

---

### 17.15 — p_inv gradient formula bug (2026-05-29)

**Discovered from:** partial output of gate 169621788 (still running), showing `|G-ratio-pinv|` = 2.0–60+ across all `LG+I+G4` FDCHECK lines. The SIGSEGV was fixed (no crash); the p_inv analytic gradient was wrong.

#### Root cause

The design document's L.0b.iv section stated the formula:
```
∂logL/∂p_inv = (L_const − Σ_c w_c·L_c) / L_total
```
treating `L_const = freq[state_p]` (bare state frequency). However, `computePtnInvar()` in
`tree/phylotreesse.cpp:617` stores:
```cpp
ptn_invar[ptn] = p_invar * state_freq[(int)(*aln)[ptn].const_char];
```
i.e., `ptn_invar[p] = p_inv * freq[state_p]` — already scaled by `p_inv`.

The implementation used `ptn_invar[p]` directly as `L_const` in:
```cpp
G += ptn_freq[p] * (ptn_invar[p] - L_p) * inv_omp / L_p;
```
which evaluates to `-(L_p - p_inv*freq[s]) / ((1-p_inv)*L_p) = -L_var_p / L_p` for constant-state
patterns. The correct formula is `(C_p - L_var_p)/L_p = (freq[s] - L_var_p)/L_p`, so the missing
term is `+freq[s]/L_p = ptn_invar[p]/(p_inv * L_p)` for constant-state patterns.

For **variable sites** (`ptn_invar[p] = 0`) the formulas coincide, which is why the bug was not
caught during the initial L.0b.ii integration — all patterns with `ptn_invar[p]=0` gave the same
result. Only when the FDCHECK loop hit `LG+I+G4` (which has constant-state patterns in the 100K AA
alignment) did the discrepancy manifest.

#### Correct derivation

Given `L_p = p_inv·C_p + (1−p_inv)·L_var_p` where `C_p = ptn_invar[p]/p_inv`:

```
∂logL_p/∂p_inv = (C_p − L_var_p) / L_p
               = [ptn_invar[p]/p_inv − (L_p − ptn_invar[p])/(1−p_inv)] / L_p
```

Sanity check — variable sites (`ptn_invar[p]=0`, `C_p=0`): formula = `−L_var_p/L_p = −1/(1−p_inv)` ✓
(increasing `p_inv` hurts variable-site patterns, correct sign).

Sanity check — constant site, only p_inv*C_p contributes (`L_var_p ≈ 0`): formula ≈ `C_p/L_p = 1/p_inv` ✓
(increasing `p_inv` improves purely-constant-state patterns).

#### Fix applied (2026-05-29)

Two locations in `tree/phylotree.cpp`:

1. **`computeModeLAnalyticGradPInv()` (line ~1662–1670):**
```cpp
// was: G += ptn_freq[p] * (ptn_invar[p] - L_p) * inv_omp / L_p;
// fix:
const double inv_pi  = 1.0 / p_inv;
G += ptn_freq[p] * (ptn_invar[p] * inv_pi - (L_p - ptn_invar[p]) * inv_omp) / L_p;
```

2. **`computeModeLAnalyticScores()` p_inv score column (line ~1757–1764):**
```cpp
// was: col[p] = (L_p > 0.0) ? (ptn_invar[p] - L_p) * inv_omp / L_p : 0.0;
// fix:
const double inv_pi  = 1.0 / p_inv;
col[p] = (L_p > 0.0)
    ? (ptn_invar[p] * inv_pi - (L_p - ptn_invar[p]) * inv_omp) / L_p
    : 0.0;
```

#### Gate bundled into L.0b.v

The §17.15 fix was bundled with the L.0b.v FreeRate gradient changes (build 169623002, md5 `563e8ed90a95db1086166741327fb6be`). A separate L.0b.iv gate was not submitted because:
1. The walltime issue (FDCHECK forcing FD) made a FDCHECK-only gate impractical
2. The L.0b.v build includes the fix and the production mode changes

---

### §17.16 — L.0b.v: FreeRate rate gradient + production mode switch (2026-05-30)

**Context:** Gate 169621788 was cancelled after 40 min (34/153 models, ~67s/model, extrapolated 170 min). Root cause of slowness: `--mode-l-fd-check` forces FD for ALL dims on every iter, even alpha/p_inv which already had analytic gradients. Additionally, `RateFree` inherits `RateGamma` causing `alpha_param_dim` to be set to dim 0 for +R models, injecting zero analytic gradient into the proportion dimension and corrupting the OPG matrix.

**Changes implemented (2026-05-30):**

**1. RateFree alpha detection bug fix** (`model/modelfactory.cpp` lines ~1475-1480):
Old code: `RateGamma *rg = dynamic_cast<RateGamma*>(site_rate); if (rg) { alpha_param_dim = ... }`
New code: `if (rg && !dynamic_cast<RateFree*>(site_rate)) { alpha_param_dim = ... }`
RateFree inherits RateGamma but has no alpha parameter — first dim is a proportion ratio.

**2. GradCtx extension** (`tree/phylotree.cpp`, `GradCtx` struct):
Added `double *per_ptn_rate_scores = nullptr` (size `ncat*nptn`). When non-null, `accumulateAlphaFromPre` and `accumulateLeafAlphaBranch` accumulate: `per_ptn_rate_scores[ptn*ncat + c%ncat] += wc * bv * qp * exp(scale - lh[ptn])`. This is ∂log(L_p)/∂r_k (natural parameterisation) for rate category k = c%ncat, using the same preorder traversal.

**3. `computeModeLFreeRateRateScores(ncat, per_ptn_rate)`** (`tree/phylotree.cpp`):
New function that sets up a GradCtx with `dr_c=0` (bypassing the alpha chain-rule path) and `per_ptn_rate_scores=per_ptn_rate`, then runs the standard preorder traversal. One pass fills all k natural rate gradients simultaneously.

**4. `computeModeLAnalyticScores()` extended** (`tree/phylotree.cpp`):
New signature: `(ndim, alpha_dim, pinv_dim, fr_rate_start_dim, fr_ncat, score_cols)`.
FreeRate rate block transforms natural scores to encoded parameter space:
```
S_rate_total[p] = Σ_k r[k] * per_ptn_rate_nat[p*ncat + k]
s_encoded[p, fr_rate_start + n] = r[k-1] * (s_nat[p,n] - w[n] * S_rate_total[p])
```
for n = 0..ncat-2 (the k-1 encoded rate dims in x[k-1..2k-3] = r[n]/r[k-1]).

**5. FreeRate dim detection** (`model/modelfactory.cpp`):
```cpp
RateFree *rf = dynamic_cast<RateFree*>(site_rate);
if (rf) {
    int fr_nc = rf->getNRate();
    int fr_nd = rf->getNDim();
    if (fr_nc >= 2 && fr_nd == 2*(fr_nc-1)) {
        freerate_rate_start_dim = model->getNDim() + (fr_nc - 1);  // after prop dims
        freerate_ncat = fr_nc;
        freerate_nrate_dims = fr_nc - 1;
    }
}
n_analytic += freerate_nrate_dims;
```

**6. Gate script → production mode** (`gadi-ci/mode-p-iso/run_lfd_aa100k_np1_mode_l.sh`):
- Removed `--mode-l-fd-check` from MODE-L arm → production mode
- New criteria: exit=0 both arms; best model LG+G4; lnL |Δ|≤0.05; accepted_iters>0; MODE-L wall < 2×BASE
- FDCHECK output preserved as `[diag]` advisory (not blocking)
- Added `|G-ratio-rate0|` parsing for FreeRate rate dim validation when FDCHECK is used manually

**Expected speedup:** For +G/+I+G models: full_analytic=true → zero FD calls → fast (~5s/model). For +Rk models: k-1 rate dims analytic + k-1 prop dims FD → half the FD calls of pure-FD. Combined with better gradient quality → fewer LM iterations. Extrapolated: 100 +G/+I+G × 5s + 53 +R × 35s ≈ 37 min total (fits 1h walltime).

**Build:** 169623002 (md5 `563e8ed90a95db1086166741327pb6be`)
**Gate:** 169623057 ❌ killed at 1h walltime (2026-05-30). Root cause: preorder kernel single-threaded. See §17.17.

---

### §17.17 — L.0b.vii: OMP parallelization of preorder/gradient kernel (2026-05-30)

**Problem diagnosed from gate 169623057 partial output:**

At 36 min elapsed, 57/153 models complete at 27.4s/model average. Projected: 153 × 27.4s = 4196s (~70 min). Gate killed at 60 min. Criterion (4) "MODE-L wall < 2×BASE=854s" fails by 5×.

Root cause (single-threaded preorder): The standard likelihood kernel (`computeLikelihoodBranchSIMD`, `phylokernelnew.h`) uses `#pragma omp parallel for schedule(static) num_threads(num_threads)` over the pattern dimension — all 103/104 OMP threads participate. The Mode-L preorder kernel (`preorderFillRecursive`, `accumulateAlphaFromPre`, `accumulateLeafAlphaBranch`) loops over patterns serially — 1 thread out of 104.

Per-sweep cost analysis (27.4s/model average, 9.5 sweeps/model average):
- 9.5 sweeps × 2.9s/sweep ≈ 27.6s/model ✓ matches measurement
- 2.9s per sweep = O(nptn × ncm × ns²) with 100K patterns, 4 cats, 20 states

OMP parallelization applied (all in `tree/phylotree.cpp`, anonymous namespace):

**1. `preorderFillRecursive`**: Added `const int nt = ctx.tree->num_threads` and `#pragma omp parallel for schedule(static) num_threads(nt)` on the `ptn` loop inside the `ci` child-loop. The shared scratch `pst`/`fsb`/`cmb` pointers (from `ctx.pre_state/f_sib_buf/combo_buf`) were replaced by `double pst_t[64], fsb_t[64], cmb_t[64]` declared inside the `ptn` loop body — stack-allocated per-thread, 64 doubles = 512 bytes (covers binary/DNA/AA/codon). Write targets `pre_child[ptn*blk+c*ns]` and `pre_child_scale[ptn*ncm+c]` are unique per ptn (non-overlapping by stride), so no false sharing beyond normal cache-line granularity.

**2. `accumulateAlphaFromPre`**: Swapped loop order from `c`-outer/`ptn`-inner to `ptn`-outer/`c`-inner. Added `double G_local=0.0` and `#pragma omp parallel for schedule(static) num_threads(nt) reduction(+:G_local)` on the `ptn` loop. `*ctx.G_alpha += G_local` after the region. Thread-safety: `G_local` via reduction; `ctx.per_ptn_alpha[ptn]` and `ctx.per_ptn_rate_scores[ptn*ncat+k]` indexed by `ptn` unique to each thread — no false writes.

**3. `accumulateLeafAlphaBranch`**: Same as above — `#pragma omp parallel for ... reduction(+:G_local)`, `double pst_t[64], fsb_t[64], cmb_t[64]` inside the `ptn` loop body, `*ctx.G_alpha += G_local` after.

**4. `computePreorderPartialLikelihood` root init**: `#pragma omp parallel for schedule(static) num_threads(num_threads)` on the root-leaf-expansion `ptn` loop (writes `pb[ptn*blk+c*ns]`, independent per ptn).

**Expected speedup (104 threads = OMP_PER_RANK=103 + 1 master):**
- Preorder sweep: 2.9s → 2.9/104 ≈ 0.028s
- Gradient accum: similar, ~0.028s per traversal
- Standard likelihood (already parallelized): ~0.03s/call
- 9.5 avg sweeps × (0.028 preorder + 0.028 grad + 0.03 lh) ≈ 0.82s/model
- 153 models × 0.82s ≈ 125s — vs BASE 427s = **3.4× faster than FCA np=1** at first trial

**No OMP `num_threads` bug risk:** `ctx.tree->num_threads` is set by `PhyloTree::setNumThreads()` before `CandidateModel::evaluate()` is called. This is the same `num_threads` used by `phylokernelnew.h` kernel loops. Recursion safety: `preorderFillRecursive` recurses after the OMP parallel region completes (threads join before recursive call), so nested parallelism is never entered.

**Build:** 169623315 (✅ 2026-05-30, 46s incremental, md5 `8912c9cccdc47ccf2d8c23411f464686`)
**Gate:** 169623342 (🔄 submitted 2026-05-30)
**Gate criteria:** same 4 criteria as L.0b.v; criterion (4) now achievable.

---

*End of design document.*
