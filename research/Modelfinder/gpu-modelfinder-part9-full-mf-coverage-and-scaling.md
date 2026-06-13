# PART IX — The next phase (G.5/G.6): full `-m MF` coverage (DNA+AA), A100<8 nodes, and the VRAM ceiling

**Author:** as1708 / Claude Fable 5, 2026-06-12. The ONE condensed design doc for the next phase (per the user's
instruction: not a doc-per-issue). Produced from a Fable multi-agent research+design workflow (9 research lenses →
synthesis → correctness + effort/scope red-teams; the perf-honesty red-team and the auto-finalize hit the session
limit, finalized by the author) **plus an empirical coverage audit** (job 170602983) that actually ran
`--jolt --gpu -m MF` on 5 k-site subsamples of both 1M alignments. **Status: PLAN. Implementation starting at G.5.0.**

---

## IX.0 Verdict in one paragraph

The "ultimate test" is `--jolt -m MF` on **1M DNA and 1M AA** with the full candidate set (which, unlike `-m TEST`,
**includes +R FreeRate**). The audit shows the gap is **asymmetric and now precisely scoped: AA `-m MF` is already
95% on the GPU; DNA `-m MF` is ~8%.** So **AA-MF is a `+R` problem** (the only `-m MF`-vs-`-m TEST` delta) and
**DNA-MF is a free-Q problem** (every HKY/TN/…/GTR matrix has free exchangeabilities that move the eigensystem).
The phase therefore: (G.5.0) lands the measured **on-device reduction** that speeds *every* JOLT model and is the
A100<np8 lever; (G.5.1) adds **+R FreeRate**, reusing the *already-validated* rate-gradient kernel — but gated by a
new **cold/warm convergence harness + a CPU-optimum comparison gate**, because the red-team is right that +R's risk
is OPTIMIZER convergence on a multimodal surface, not the gradient; (G.5.2) **VRAM-tiles** the postorder arena so
the +R10 ladder still fits A100-80 at 1M; then (G.5.4) the **AA `-m MF` ultimate-test run**. **Free-Q DNA is split
out into its own phase G.6** — it is the riskier eigen-moving track (a full re-eigendecomposition per FD step) and
bolting it onto an already multi-week phase endangers shippability. **Honest ceiling (unchanged from part5):** the
per-model GPU is mutex-serialized (S≈4.8) vs N CPU-concurrent, so the `-m MF` win lives in the **CTF top-k≤3 refine**
(depth), not in full-set breadth — G.5.4 runs +R coverage *inside* CTF, never as a block-saturated full-set sweep.

---

## IX.1 The measured coverage map (job 170602983 — the empirical anchor)

`--jolt --gpu -m MF`, `JOLT_DEBUG=1`, 5 k-site subsample of each 1M alignment, per-model engage/decline tallied:

| data | candidates | **engage JOLT** | decline | decline breakdown |
|---|---:|---:|---:|---|
| **AA** | 122 | **116 (95%)** | 9 | 8 `non-mean-gamma` (+R) + 1 `pure-pinvar` |
| **DNA** | 98 | **8 (8%)** | 90 | **62 `free-subst-params`** + 8 +R + 1 pure-+I |

- **AA** engages the *entire* empirical-matrix family — LG, WAG, JTT, Q.PFAM/BIRD/MAMMAL/INSECT/PLANT/YEAST,
  JTTDCMUT … × {+G4, +I+G4, +F+G4, +F+I+G4} — all at rel ~1e-11…1e-16 PASS. The only non-engaged AA models are
  `+R` (8) and one pure `+I`. **AA-MF is one capability (+R) away from ~100%.**
- **DNA** engages only the **zero-free-parameter** models (JC, F81(+F) × {+G4, +I+G4}). Every K80/HKY/TN/TPM/TIM/
  TVM/**GTR** variant declines at `getNDim()!=0`. The audit "winner" was F81+F+G4 only because a 5 k subsample can't
  resolve the free-Q edge — **on the real GTR-generated 1M data the true winner is a GTR-family model that runs on
  CPU today.** DNA-MF is meaningless on the GPU without free-Q.

**Gate lines (verified):** `phylotreegpu.cpp:494` `getNDim()!=0 → "free-subst-params"` (the DNA wall + AA +FO/GTR20);
`phylotreegpu.cpp:502` `isGammaRate()!=GAMMA_CUT_MEAN → "non-mean-gamma"` (all +R/+I+R, median-gamma);
`:513` pure-+I. The `+R/+I+R` set is the only `-m MF`-vs-`-m TEST` difference (`phylotesting.cpp:1237`).

---

## IX.2 The phased plan (G.5 = AA-MF + perf + VRAM; G.6 = DNA free-Q)

| id | title | deliverable (files) | gate | risk | days |
|---|---|---|---|---|---:|
| **G.5.0** | On-device reduction (perf lever, speeds EVERY model) | Move the per-edge host reduction on-device in `gpu_lnl_intree.cu`: one-time `ptn_freq` H2D (~:438 next to `d_baseinvar`); deterministic FP64 **pairwise** block-reduce appended to `kj_derv` (~:130) emitting weighted {lnL,df,ddf}/edge; `reduceDerv` → single 24-byte D2H (:484-493). **WIDENED (red-team):** also fold the `d_rnum`/`gradR` host loop (:534-535) and the `invL=exp(-patlh)` loop (:529) on-device — else the +R rate-grad reduction G.5.1 hammers stays host-bound. | LG+G4 **and** LG+I+G4 `-te` self-check rel≤1e-9 (expect ~1e-12) **AND identical `outIters`+accept/reject sequence** vs the host-Kahan path. A100 wall is a **REPORTED metric, not a gate** (it is topology/CTF-dependent — see IX.4). | low | 3 |
| **G.5.1a** | +R standalone cold/warm **convergence** harness (NEW, red-team-mandated) | A `gpu_k8`-style standalone harness (like G.4.1): joint diagonal-LM over (branches + `log` rates + `softmax` weights), gauge-fixed by a `rescaleRates`-equivalent (Σw·r=1) after every accepted step; the **new weight-gradient `gW_c` FD-validated** (central diff). | cold-start == warm-start == **CPU-EM MLE** rel≤1e-9 on LG+R4 *and* LG+R6. If the diagonal-LM stalls (expected per the +I 39.5-nat precedent), add an EM-warm-start or 3–4-point multi-start as a *planned* sub-task. | med | 4 |
| **G.5.1b** | +R / +I+R in-tree JOLT coverage | Flip `phylotreegpu.cpp:502` to admit `isFreeRate()` (seed `catRate=getRate(c)`, `catProp_v=getProp(c)`, bypass `applyAlpha`); fold the `2·ncat−2` params into the LM step (G.5.1a design); writeback `setRate`/`setProp`+`clearAllPartialLH`. **+I+R DECLINES to CPU initially** (red-team: RateFreeInvar's prop structure ≠ G.4.3b's `(1−pinv)/K`; the `applyPinv` coupling is wrong for it — N-parallel-cheap on CPU). | **CPU-optimum comparison gate** (closes the +I-silent-regression pattern): after JOLT writeback, assert JOLT lnL ≥ CPU-refined lnL − `modelfinder_eps`; if worse, return NaN → CPU fallback for that candidate. Then AA-1M `-m MF` best-by-BIC == CPU oracle + per-candidate lnL parity. | med | 6 |
| **G.5.2** | VRAM tiling of the postorder arena | Tile `gbj_partial` (`nInternal·ncat·ns·nptn`, the 59 GB-of-88.6 GB dominant buffer; the +R ladder multiplies it by ncat up to 10) into T pattern-tiles; loop postorder/preorder+reductions per tile, FP64-accumulate the weighted scalars across tiles; auto-T from `cudaMemGetInfo` gated by (ns,ncat). | tiled lnL/gradient == untiled rel≤1e-12 (cross-tile order-stable); +R10 AA-1M < 80 GB on A100-80; wall within launch/reload overhead (NOT T-fold — MEASURE, don't assume). | med | 4 |
| **G.5.4** | **The AA ultimate test** — `--jolt -m MF` 1M AA, GPU↔CPU parity + energy | End-to-end inside **CTF** (rank-on-subsample → refine top-k≤3 with full +R coverage), CPU-spooling the residual (pure +I, +FO, median-γ) via the existing OMP queue + `jolt_gpu_mtx`. Report coverage %, best-by-BIC==oracle, A100 wall vs np8 1443.9 s, peak VRAM, energy (the part5/§V.13 harness). | best-by-BIC == CPU `-m MF` oracle (AA); engaged-coverage ≥90%; A100 wall + VRAM<80 GB reported. | med | 3 |
| **G.6** (own phase) | **Free-Q DNA** via FD-over-Q (kappa…GTR) | Relax `:494` for reversible `getNDim()≤8` (exclude +FO/GTR20); generalize the LM param vector to `(α,pinv,q_1..q_nQ)`; each q-grad by forward-FD like the shipping pinv FD, each step `applyQ→model->decomposeRateMatrix()→re-upload eval/U/Uinv + rebuildEchild`; freqs held fixed (empirical/equal). | HKY (nQ=1) FD-grad validated vs CPU `computeFuncDerv` **and FD-LM converges to the HKY MLE**; then **GTR (nQ=5) cold/warm convergence + boundary-stall** check (coupled rates — pre-commit to a dense Q-block fallback, NOT mid-phase); DNA GTR+G4 `-te` rel≤1e-9; DNA-1M `-m MF` best-by-BIC == oracle. | med-high | 7–10 |

**First step (now): G.5.0**, widened scope + demoted wall gate. Highest value (every downstream +R/free-Q model
inherits it), lowest risk (no CPU path touched; FP64-reorder caught by a fast existing self-check).

---

## IX.3 Correctness disciplines (red-team-mandated, non-negotiable)

1. **+R risk is OPTIMIZER convergence, not the gradient.** IQ-TREE's default +R optimizer is **EM** (Wang-Li-Susko-
   Roger 2008, `ratefree.cpp:323`) with a monotone-lnL guarantee; +R is the most **multimodal** surface in the set.
   "Same gradient" ≠ "same MLE". → the standalone **cold/warm convergence harness (G.5.1a)** runs *before* any
   in-tree wiring, and a **CPU-optimum comparison gate** in G.5.1b makes a worse-than-CPU JOLT minimum fall back to
   CPU instead of silently shipping (the exact +I 39.5-nat failure mode, `modelfactory.cpp:1374`).
2. **The gauge null-direction.** +R has TWO coupled constraints — Σw=1 (softmax handles it) **and** Σw·r=1
   (`rescaleRates`). A diagonal LM with no off-diagonal curvature can drift along the null manifold → **gauge-fix
   (`rescaleRates`-equivalent) after every accepted step**, not just at the end.
3. **+I+R does NOT reuse G.4.3b.** RateFreeInvar: free props + `pinv=1−Σprop` with its own EM M-step — not
   `(1−pinv)/K`. → **decline +I+R to CPU in G.5.1**; implement it properly (with its own FD validation) only if it
   ever lands on the critical path.
4. **G.5.0 FP64 reduction:** deterministic **pairwise/two-stage** reduce, **never `atomicAdd`**; the final cross-block
   combine in the same order as the host loop (or kept on host) so the sum is bit-reproducible. Gate on **identical
   `outIters` + accept/reject sequence**, not just final-lnL parity — a drift past the `ln>lnL+1e-9` accept test
   (`gpu_lnl_intree.cpp:571`) could flip a near-degenerate BIC tie. FP64 parity on reduced lnL/gradient stays
   non-negotiable (never TF32/FP16).
5. **Free-Q (G.6):** FD over exchangeabilities is NOT structurally identical to the α/pinv FD — it re-runs a full
   **eigendecomposition** per FD eval (GTR nQ=5 ⇒ ~6 decompose+re-upload/iter). FD-validate HKY (nQ=1) and prove
   FD-LM *convergence* before GTR; coupled GTR rates may need a **dense Q-block** (pre-committed fallback, not scope-
   creep). Defer analytic dQ/dparam (Kenney-Gu) — the FD eval sweeps are the GPU's strength.

---

## IX.4 The honest performance ceiling (do not overclaim)

Per part5 (V.10/V.11, P3.0): the per-model GPU is **mutex-serialized** (S≈4.8) while the CPU runs **N** models
concurrently; the lnL kernel is **memory-latency + occupancy-bound** (not bandwidth-bound — tiling is a *capability*
lever, not throughput). Therefore **moving more models onto the serial GPU does not win `-m MF` by breadth** — that
is the CPU cluster's strength (FCA). The win is **per-model DEPTH** (JOLT 4.8×), realized **inside CTF's top-k≤3
refine**. So:
- **G.5.0's A100<np8 wall is a REPORTED metric, not a pass/fail gate** (red-team): the reduction is a real structural
  speedup, but the *wall* also depends on the serial-vs-N-parallel topology and the CTF framing. The hard wall target
  lives in **G.5.4 with CTF top-k explicitly in scope**.
- **G.5.4 runs +R coverage INSIDE CTF**, never as a full-set block-saturated sweep (which P3.0 shows loses). The
  honest claim remains: *CTF+JOLT beats the stock tool / a small node count on one accessible GPU; it does not
  overturn the breadth verdict vs a large cluster.* +R/free-Q make the *refine* model-complete; they do not change
  the topology.

---

## IX.5 What is CUT (and why)

cuSOLVER/cuBLAS batched eigendecomp (host decompose is µs at ns≤20 — not the bottleneck); **+FO on GPU** (19 FD
dims, and the default AA/DNA MF freq sets carry no `+FO` — off the critical path); **GTR20** (189 dims); **grid.z
cross-model batching** (block-saturated at full data, P3.0); the **occupancy moonshot** (thread-per-(ptn×cat×state)
— honest coin-flip, and not needed for AA-MF); **MMAP/double-buffer** (load is one-time, not the hot loop). All stay
CPU by design — `jolt_gpu_mtx` + the existing OMP across-model work-queue already overlap them (no new plumbing).

---

## IX.6 Status & next action

Coverage audit (170602983) banked. Plan finalized with the correctness + effort/scope red-teams folded in.

**✅ G.5.0 PART A VALIDATED (job 170634239, 2026-06-12):** the on-device `kj_reduce3` pairwise reduction (replacing
`reduceDerv`'s 3×nptn D2H + host Kahan) is CORRECT — AA `-m MF` subsample (seed 1, == audit) gives **116/116
engaged models matching a fresh CPU `computeLikelihood`, max rel 2.17e-10** (gate ≤1e-9, 0 fails), coverage +
declines + best-model (LG+G4) **identical** to the pre-change audit. Max rel rose ~1e-11→2.17e-10 (the predicted
FP64 reduce-reorder drift, 5× under gate). **GPU util 61%→96%** on the same subsample (host-reduction stall gone).
Code: `gpu_lnl_intree.cu` `kj_reduce3` + `gbj_ptnfreq`/`gbj_redpart` + invL base-edge D2H. **Part B** (move the
once-per-sweep `gradR`/`invL` host loops on-device too — the +R ladder hammers them at ncat≤10) is the remaining
G.5.0 scope.

**✅ A100<np8 ACHIEVED (job 170636493, the G.5.0 payoff):** A100 1M-AA CTF total **1355 s** (coarse 168 + refine
1187), down from **1504 s** pre-G.5.0 (~10%, the host-reduction stall removed) → **beats np8 (1443.9 s) at 1.07×**,
np4 1.46×, np2 2.27× (np16 1122 s still leads). Winner LG+G4 ✓; GPU energy 73.24 Wh (was 81.69). The user's
"A100 beats 8 nodes" target is met by the on-device reduction alone.

**⏳ G.5.0 PART B + KERNEL FUSION implemented 2026-06-12 (validating, job 170726673):**
- **Part B** — `gradR`/`invL` moved on-device: `kj_invl` (1/L_p from the base-edge patlh) + `kj_reduce_gradnum`
  (per-category deterministic block reduction of `ptn_freq·rnum[c]·invL`), replacing the once-per-sweep `ncat×nptn`
  `d_rnum` D2H (75 MB at +R10) + host long-double loop. The +R ladder hammers this ncat-fold, so it had to leave
  the host.
- **Kernel fusion (part8 #3, user-requested)** — `kj_derv_fused` FUSES `kj_theta`+`kj_derv`+`kj_ratenum`: `theta =
  node·dad` is computed in registers and consumed in-place for {lnL,df,ddf} AND the rate numerator, **never
  materialised to the 601 MB `d_theta`** (eliminates 1 write + 2 reads = 3×slotSz/edge of VRAM traffic on the
  bandwidth-bound kernel). **Bit-identical** to the unfused path (FP64 store/load is lossless; same per-(c,x) product
  order) ⇒ the rel≤1e-9 gate is expected to hold at ~1e-12. Builds clean (binary 02:03). `kj_theta`/`d_theta`/
  `edgeThetaInto` are now dead — reclaim the 601 MB `d_theta` alloc in cleanup (helps the +R10 / A100-80 margin).

**Next:** validate the combined Part B + fusion (170726673), re-benchmark AA `-m MF` (expect a wall improvement from
the fusion's bandwidth saving), then G.5.1a's +R convergence harness (the load-bearing correctness piece — NOT
rushed into the benchmark binary). DNA free-Q is G.6, a separate phase, to keep G.5 (AA-MF) shippable.

---

## IX.7 The AA `-m MF` benchmark TIMED OUT — root cause was the CTF ranking, not the kernel (2026-06-13)

**Both AA-1M `--jolt -m MF` benchmarks (H200 170728179, A100 170728182) hit the 2 h walltime.** Honest root cause,
and it is NOT a GPU/correctness problem — it is the CTF **coarse-rank gate**:

- The coarse stage ranked the top-k by the **scale-consistent BIC PROJECTION** `−2·(N/m)·logL + p·ln(N)`. On `-m MF`
  this amplifies sub-nat subsample overfit by `2·N/m ≈ 378×` and ranked top-3 = **[LG+I+G4, LG+R5, LG+I+R5]** — the
  true winner **LG+G4 dropped to rank 4**, and two **+R FreeRate** models were promoted in. (Full table + mechanism:
  PART X §X.5.5 — this is the §X.3.2 projection-amplification bug, now empirically confirmed on the real pipeline.)
- The promoted +R models **decline JOLT** (non-mean-gamma, IX.1) → refine on the **CPU EM optimiser at full 1M**
  (945k patterns) → each runs for hours. Both jobs died in refine #2 (LG+R5 on CPU). **GPU util was 0 % at kill** —
  the GPU sat idle while the CPU ground a +R model the projection should never have shortlisted.

**Fix (landed, re-running 170756438/170756440), red-team-reviewed:**
1. **Rank the top-k by NATIVE subsample BIC over ALL candidates** (penalty `ln m`). Native BIC ranks LG+G4 #1 (verified
   at all 23 sweep runs + the live coarse table) — because on 5000 sites the rate-model fits are within <1 nat, the
   decision is penalty-dominated and native BIC trusts the penalty, not the amplified noise. (Do NOT pre-exclude +R —
   that would hide the coverage gap.)
2. **Rate-heterogeneity detector:** flag the run if a +R/+I model genuinely leads the best eligible by more than the
   ~Δp/2 overfit cushion. Here it does NOT (LG+G4 leads LG+R4 by 43 nat) → safe to eligible-refine. On genuinely
   rate-heterogeneous data the flag fires and demands a +R refine (G.5.1) or CPU escalation — surfaced, not silent.
3. **Per-model wall budget** on the refine: an ineligible CPU-at-1M refine that exceeds budget is carried unrefined;
   a detector-confirmed-losing ineligible model is skipped outright. The GPU-eligible refine ({LG+G4, LG+I+G4}) is
   ~610 s → total ≈ coarse 544 s + 610 s ≈ 1150 s, finishes and picks LG+G4.

**This makes G.5.1 (+R JOLT coverage) the empirically-urgent next milestone, not just a coverage nicety:** until +R
refines on the GPU, any genuinely +R-winning dataset forces the CPU-at-1M landmine (or a detector escalation). The
benchmark dodges it only because the AA winner is +G. **Also banked:** (a) validate the EXACT shipped GPU pipeline
end-to-end, not a CPU proxy (the sweep used the MPI fork, which ranks +R ~190 nat worse and never exercised the bug);
(b) the unexplained CPU-vs-GPU +R subsample-fit discrepancy is an open investigation.

### IX.7.1 Ranked next steps (post-finding)
| # | step | why now | status |
|--:|---|---|---|
| 1 | ~~Confirm the fixed-gate AA `-m MF` benchmark~~ **✅ PASS (H200 170756438)** | end-to-end, exit 0, **767 s** (coarse 467 + refine 300), **WINNER=LG+G4** ✓ (BIC 157213275.8, beats LG+I+G4 by 14); detector RATE_HET=False → LG+R4 skipped (no CPU landmine); **1.46× np16 / 1.88× np8**, 43.6 Wh, 67.25 GB. The log shows the OLD projected top-5 putting LG+G4 at rank 4 — the fix dodges it. (A100 170756440 queued = 2nd-device confirm.) | ✅ done |
| 2 | **G.5.1 +R JOLT coverage** (standalone convergence harness → in-tree) | the +R refine landmine + the coverage gap make this the load-bearing capability, not optional | ⬜ designed (IX.2) |
| 3 | ~~CPU-vs-GPU +R gap~~ **RESOLVED**: not a fit gap — the forks evaluate **different +R sets** (MPI fork: R2 only; GPU fork: R2→R5). Residual: confirm which fork's `-cmax`/FreeRate-search default is correct | the proxy never generated R4/R5 ⇒ couldn't see the bug (strengthens "validate the shipped pipeline") | 🟡 mostly closed |
| 4 | **Port the native-BIC gate + detector + budget into the production CTF path** (not just the benchmark script) | the bug lives in pipeline code; harden it where it ships | ⬜ |
| 5 | **DNA `-m MF` (G.6 free-Q)** | the 8 %-coverage track; separate phase | ⬜ |

---

## IX.8 G.5.1 +R FreeRate JOLT coverage — implementation design (2026-06-13, STARTED)

**Coverage scope (measured, IX.1).** +R closes the **AA** gap: AA `-m MF` 116/122 → ~99–100 % (adds the 8 `+R`/`+I+R`
models; only pure-`+I` remains). It barely moves **DNA** (8 → ~16/98) — **DNA is a free-Q problem (G.6), NOT a +R
problem.** So G.5.1 is the *AA-completion* milestone; DNA still needs G.6. Honest: **DNA is NOT meaningfully on the GPU
today** and +R does not change that.

**Eligibility (the models G.5.1 admits).** Currently `phylotreegpu.cpp:502` declines `isGammaRate()!=GAMMA_CUT_MEAN`
(all +R/+I+R). G.5.1b admits **`isFreeRate()` with uniform-free-rate (`+Rk`, k=2..10)**; **`+I+R` STILL DECLINES to
CPU** (IX.3 #3: `RateFreeInvar` has its own `pinv=1−Σprop` EM M-step, not G.4.3b's `(1−pinv)/K` — wrong coupling).

### IX.8.1 The gradient (derived; the weight gradient is the only NEW quantity)
FreeRate: `L_p = Σ_c w_c·V_c(p)`, weights `w_c=prop[c]` (Σw=1, softmax), rates `r_c=rates[c]` (gauge Σw·r=1 via
`rescaleRates`), `ndim = 2·ncat−2`. In the in-tree code `g_val0[c,x]` already folds in `w_c`, so the per-category
contribution is `Lc(p) = Σ_x g_val0[c,x]·θ = w_c·V_c(p)` and `lh = Σ_c Lc(p) = L_p`.

- **Weight gradient (softmax param `z_c`):** with `WN_c := Σ_p ptn_freq·Lc(p)·invL[p]`,
  **`∂lnL/∂z_c = WN_c − w_c·N`** (`N = Σ_p ptn_freq`). Derivation: `∂L_p/∂w_c = V_c = Lc/w_c` ⇒ `∂lnL/∂w_c = WN_c/w_c`;
  softmax `∂w_c/∂z_d = w_c(δ−w_d)` ⇒ `∂lnL/∂z_d = WN_d − w_d·Σ_c WN_c`, and the identity `Σ_c WN_c = Σ_p ptn_freq·L_p·invL = N`.
  Checks: `Σ_d ∂lnL/∂z_d = 0` (gauge null direction, as required); `WN_c` reuses **`kj_reduce_gradnum`** fed `Lc(p)`
  instead of `rnum[c]`.
- **Rate gradient (log param `y_c=log r_c`):** `∂lnL/∂y_c = r_c·gradR[c]` — `gradR[c]` is the **already-validated**
  G.4.0b rate gradient (`kj_reduce_gradnum` over `rnum`). No new kernel.
- **Branch gradient:** unchanged (Ji preorder, `g_df`/`g_ddf`).

### IX.8.2 Kernel + LM changes (build on existing machinery, minimal new code)
1. `kj_derv_fused`: add ONE per-category accumulator `Lc = Σ_x g_val0[c,x]·θ` and (when `wnum!=null`) write
   `wnum[c·nptn+ptn]=Lc` — alongside the existing `rnum`. Register-cheap (one more accumulator). Bit-exact to lnL.
2. `computeGradient`: launch `kj_reduce_gradnum(wnum)` → `WN_c`; host forms `gz_c=WN_c−w_c·N` and `gr_c=r_c·gradR[c]`.
3. The LM loop: a **+R branch** (when `isFreeRate`) replacing the alpha/pinv block — fold `{y_c}`+`{z_c}` into the
   joint diagonal-LM step (per-param secant `ddf`, μ-damped `Δ=g/(|ddf|+μ)`), then **`r=exp(y)`, `w=softmax(z)`**, and
   **`rescaleRates` (Σw·r=1) after every ACCEPTED step** (IX.3 #2 gauge, not just at the end).
4. Launcher `gpu_jolt_optimize`: a `+R` mode — seed `catRate=getRate(c)`, `catProp_v=getProp(c)`, bypass `applyAlpha`.

### IX.8.3 Validation (G.5.1a — BEFORE the eligibility gate flips; IX.3 #1)
A gated `gpuFreeRateConvergenceCheckOnce` (the `gpuDervCrossCheckOnce` pattern): run the +R JOLT from **cold AND warm**
starts and compare to IQ-TREE's **CPU EM MLE** (`RateFree::optimizeWithEM`, runs anyway). **Gate: cold==warm==CPU-EM
rel≤1e-9 on LG+R4 AND LG+R6.** Plus FD-validate `gz_c` (central diff) and the `Σ_c WN_c=N` identity. If the diagonal-LM
**stalls** (the +I 39.5-nat precedent — +R is the most multimodal surface, IX.3 #1), add an EM-warm-start or 3–4-pt
multi-start as a *planned* sub-task — do NOT flip the gate on a stalling optimiser. Only after PASS: G.5.1b flips
`:502` to admit `isFreeRate()` + the **CPU-optimum comparison gate** (assert JOLT lnL ≥ CPU−eps else NaN→CPU fallback).

#### IX.8.4 STATUS — G.5.1a increment 1 (the weight gradient) VALIDATED 2026-06-13 (job 170777660)
The **only NEW quantity** for +R, the softmax weight gradient `gz_c = WN_c − w_c·N`, is **FD-validated on the real GPU
path** (commit `65e45c4c`). `kj_derv_fused` gained the per-category `Lc(p)` accumulator (+ optional `wnum` output) and
stays **32 registers** (cuobjdump → 100% occupancy preserved); the gated `JOLT_RGRADCHECK` hook computes `gz_c` and
central-FD-checks it. **LG+R4 (5000-site AA subsample):** `gz_c == FD` maxrel **1.03e-8** (cold) / **4.87e-6** (non-uniform
weights), `Σ_c WN_c = N` **exact** (relWN 0), `Σ_c gz_c ≈ 0` (≤6e-12, the gauge null direction). `g50val` (170777846)
confirms the `lcc` reassociation did NOT regress the gamma `-m MF` path: best = **LG+G4**, all `[JOLT]` self-checks rel
≤ **2.4e-12**. **This validates the gradient math only** — the +R LM optimiser branch + the cold/warm-vs-CPU-EM
convergence gate (IX.8.3, the multimodal risk) is increment 2 and the eligibility gate stays unflipped until it passes.

---

## IX.9 G.6 Free-Q DNA — implementation design (2026-06-13, STARTED)

**The DNA gap, scoped from source (`modeldna.cpp`, `phylotesting.cpp`, `modelmarkov.cpp`).** The audit (IX.1) measured
DNA `-m MF` at **8 % GPU coverage**, and the dominant decline (62 of 90) is **`free-subst-params`** — every model with
free rate-matrix exchangeabilities. The DNA `-m MF` candidate space is **22 base models × {`+FQ`,`+F`} × rate-het**:

| free-Q (nQ) | rate-class string | models | note |
|---:|---|---|---|
| **0** | `000000` | JC, F81 | ✅ already on GPU (G.2) |
| **1** | `010010` | K80, HKY | κ (ti/tv) |
| **2** | `010020`/`012210`/`121020`/`120120` | TNe, TN, K81, K81u, TPM2, TPM2u, TPM3, TPM3u | **largest tier (8)** |
| **3** | `012230`/`121030`/`120130` | TIMe, TIM, TIM2e, TIM2, TIM3e, TIM3 | (6) |
| **4** | `412310` | TVMe, TVM | |
| **5** | `123450` | SYM, GTR | most general |

So free-Q (nQ=1..5) unlocks the **20 base-model families** that decline today — the whole DNA gap except `+R`/`+I+R`
(G.5.1) and pure-`+I`. **The decisive, source-confirmed simplification: `+FO` (ML-estimated freqs, `FREQ_ESTIMATE`) is
NOT in the default `-m MF` set** (`dna_freq_names[] = {"FQ","F"}`, `phylotesting.cpp:107`; `+FO` only with `-mfreq FULL`).
Both `+FQ`/`+F` hold freqs **fixed** during optimisation (`getNDim()` adds 0 freq dims) ⇒ **a GPU that holds freqs fixed
covers the entire default DNA candidate space with no `+FO` gap.** DNA-MF becomes *meaningful* on the GPU for the first
time (the audit "winner" F81+F+G4 was an artifact of a 5 k subsample that can't resolve free-Q; the GTR-generated 1M data's
true winner is a GTR-family model that runs on **CPU** today).

### IX.9.1 The gradient strategy — FD over Q (there is NO CPU analytic oracle)
Source fact: neither `ModelDNA` nor `ModelMarkov` implements an analytic Q gradient — **IQ-TREE itself optimises Q by
forward-FD BFGS** (`Optimization::derivativeFunk`, step `h=ERROR_X·|x|`, `ERROR_X=1e-4`). So the validation oracle is NOT
"GPU grad == analytic grad"; it is **(a) GPU lnL == CPU lnL at every perturbed Q** (bit-parity, which makes the FD grads
match by construction) and **(b) the GPU FD-LM converges to the same MLE as the CPU BFGS**. The GPU therefore mirrors the
CPU's own method: **FD over Q inside the JOLT LM loop.** Per free param `q_k`: perturb → `decomposeRateMatrix()` (4×4,
~µs) → re-upload `g_U`/`g_Uinv`/`g_UinvRowSum` → `rebuildEchild()` → full sweep → `gq_k=(lnL(q_k+h)−lnL)/h`. nQ ∈ {1..5}.

- **The gauge & the one-code-path insight.** G-T (`rates[5]`) is **always fixed = 1.0** and never in the optimiser vector;
  the free params are `model->getVariables()[1..num_params]`, **raw** rates in `[1e-4,100]`. By perturbing in this
  *variable* space (not the raw 6-entry `rates[]`), the model's **`param_spec`** rate-class mapping is applied
  automatically — HKY's one κ moves **both** A-G and C-T; TPM2's two params move their classes — so **one code path
  covers HKY..GTR and every equal/empirical-freq variant** with no per-model logic. (Implemented behaviour-neutrally as
  two public virtual wrappers `gpuGetFreeParams`/`gpuSetFreeParamsDecompose` on `ModelSubst`→`ModelMarkov`, over the
  protected `(set/get)Variables`; CPU path untouched.)
- **Cost.** GTR+I+G4 = 1 base sweep (branches, analytic) + 1 α-FD + 1 pinv-FD + **5 Q-FD** = 8 sweeps/iter (vs +G4's 2).
  Heavy, but DNA is **ns=4** ⇒ the postorder arena is **~5× smaller** than AA ns=20 (GTR+G4 1M ≈ 12 GB, fits any GPU —
  **VRAM tiling G.5.2 is NOT needed for DNA**), and the ns=4 kernels are fast. The 4×4 decompose + the ~128-byte symbol
  re-upload per FD step are negligible vs the sweep.

### IX.9.2 Ported AA optimisations (inherited for free; the kernels are model-agnostic)
The G.5.0 levers all apply to DNA with **zero DNA-specific work**: on-device reduction (`kj_reduce3`/`kj_invl`/
`kj_reduce_gradnum`), the fused `kj_derv_fused` (601 MB `d_theta` eliminated), the **base-sweep skip** (part8 #2), the
`d_theta` reclaim. **One DNA caveat for the base-sweep skip:** `devValid` tracks (brlen,α,pinv) only — a Q-FD step changes
the eigensystem without changing those, so the skip must be **invalidated around every Q-FD step** (re-decompose back to
base + reset the flag), else a stale-eigensystem sweep would corrupt the next gradient. This is the one new state-coherence
rule G.6 adds (analogous to the G.2.1 `theta_computed` finding).

### IX.9.3 Eligibility (what G.6.1 admits)
Relax `phylotreegpu.cpp:494` `getNDim()!=0 → "free-subst-params"` to admit: `ns==4` **reversible** free-Q with
`getFreqType() != FREQ_ESTIMATE` (exclude `+FO`) and `getNDim() ≤ 5`. **Keep** the `:502` non-mean-gamma decline (`+R`/
`+I+R` → CPU until G.5.1) and the `:513` pure-`+I` decline. (AA `+FO`/GTR20, `getNDim() > 5`, stays declined — off the
default critical path, IX.5.)

### IX.9.4 Validation plan (correctness-first, phased; gate stays unflipped until G.6.0b passes)
- **G.6.0a — Q-FD gradient cross-check ✅ PASSED 2026-06-13 (job 170787044, commit `525c4186`).** Gated
  `gpuFreeQGradCheckOnce` (`JOLT_QGRADCHECK`, the `gpuDervCrossCheckOnce` pattern): perturb each free exchangeability,
  compare GPU clean-room lnL vs CPU `computeLikelihood` at the perturbed Q + FD-grad; the model is fully restored
  (read-only). **Reuses the existing clean-room sweep** (`gpuComputeTreeLnLCleanRoom` reads `model->getEigenvalues/…`
  fresh each call ⇒ a changed Q needs **zero kernel work**). The two public model wrappers `gpuGetFreeParams`/
  `gpuSetFreeParamsDecompose` (ModelSubst→ModelMarkov, over the protected `(set/get)Variables`) give the **one-code-path**
  param_spec mapping. **RESULT (5000-site DNA subsample of the GTR+I+G4 1M data, V100): HKY+F+G4 (nQ=1), TNe+G4 (nQ=2,
  equal), TVM+F+G4 (nQ=4), SYM+G4 (nQ=5, equal), GTR+F+G4 (nQ=5, empirical) — ALL `maxrel_lnL = 0.000e+00`
  (BIT-IDENTICAL, *stronger* than the 1e-9 gate) at base AND every perturbed Q; gradients nonzero/meaningful (dL/dq up to
  ±2390), GPU==CPU to the last bit** (clean-room sweep ≡ CPU NORM_LH, identical FP64). Agent-reviewed correct &
  side-effect-free. **The eigendecompose→reupload→resweep pipeline the G.6.0b optimiser drives is bit-perfect.**
- **G.6.0b — free-Q FD-LM + convergence gate ✅ PASSED 2026-06-13 (job 170792611, commit `a12e6dde`).** The nQ Q-params
  are folded into `gpu_jolt_optimize`'s joint diagonal-LM via a host **decompose-callback** (`jolt_qdecompose_fn`, plain
  C-ABI, wrapping `gpuSetFreeParamsDecompose` + the eigen accessors — launcher stays model-agnostic). eval/U/Uinv became
  mutable host buffers (evalB/UB/UinvB via evalP/UP) refreshed per Q change by `qApply`; each free param is FD-gradiented
  (mirroring the pinv-FD) with a secant curvature and folded into the backtracking joint step; the base-sweep skip is
  disabled for free-Q (§IX.9.2). **RESULT (5000-site DNA subsample of GTR+I+G4 1M, V100, `-te` fixed topology, JOLT vs
  IQ-TREE's own BFGS Q-optimiser):**

  | model (nQ) | CPU MLE | JOLT MLE | jolt−cpu |
  |---|---|---|---|
  | HKY+F+G4 (1) | −298235.6309 | −298235.6302 | **+0.0007** |
  | TN+F+G4 (2) | −298235.5244 | −298235.5236 | **+0.0008** |
  | TVM+F+G4 (4) | −298233.2957 | −298233.2951 | **+0.0006** |
  | GTR+F+G4 (5) | −298233.1908 | −298233.1879 | **+0.0029** |
  | SYM+G4 (5, equal) | −299183.2635 | −299183.2613 | **+0.0022** |

  **JOLT ≥ CPU in every case** (joint LM marginally *beats* the alternating BFGS/brlen by 0.0006–0.0029 nat) ⇒ **NO
  stall — even GTR's 5 COUPLED exchangeabilities converge under the diagonal LM in 13–34 joint iters, so the pre-committed
  dense-5×5-Q-block fallback is NOT needed.** In-tree `[JOLT]` write-back self-check rel **~5e-12** all five. The +I
  39.5-nat stall mode does NOT recur for free-Q. Agent-reviewed (device coherence, Q-FD restore, secant timing, reject
  path, aliasing, callback lifetime/thread-safety, fixed-Q non-regression) — all OK, no bugs. Gate stays env-gated
  (`JOLT_FREEQ`) until G.6.1 makes it permanent + adds the CPU-optimum safety gate.
- **G.6.1 — coverage + gate flip ✅ PASSED 2026-06-13 (job 170795329).** DNA `-m MF` on the 5000-site subsample,
  GPU (`--jolt`, free-Q) vs CPU: **JOLT engagements 8 → 70** (every free-Q family — GTR, TVM(e), TIM/TIM2/TIM3(e),
  TPM2/3(u), TN(e), K3P(u), SYM, HKY × {+G4, +F+G4, **+I+G4**}), only **9 decline (8 +R/+I+R `non-mean-gamma` + 1
  pure-`+I`** = the expected residual, +R is G.5.1). **GPU best-by-BIC == CPU best (F81+F+G4)** — the 5000-site
  subsample under-powers free-Q so BIC picks the simplest adequate model, but the GPU↔CPU *parity* is the gate.
  **Worst write-back rel 6.224e-12 across all 70, ZERO MISMATCH.** Even **GTR+F+I+G4** (5 free-Q + pinv + α jointly)
  engages. Then the eligibility gate (`phylotreegpu.cpp:494`) was flipped — free-Q is **ON BY DEFAULT** (escape hatch
  `JOLT_NO_FREEQ`), Q write-back via `gpuSetFreeParamsDecompose(outQ)`, plus a **safety gate**: write-back rel > 1e-6 ⇒
  NaN → CPU fallback (catches a kernel/regime failure; convergence-to-CPU-MLE is validated by G.6.0b, not a per-candidate
  CPU re-run — the honest scope). DNA `-m MF` is now ~89% on the GPU (the residual is +R/+I+R = G.5.1, and pure-+I).

- **G.6.2 — DNA-1M `-m MF` CTF payoff ✅ PASSED 2026-06-13 (job 170843136, A100).** The DNA analog of the AA
  ultimate test, on the **full 1 M-site** GTR+I+G4-generated alignment. **WINNER: F81+F+G4** (full BIC 118418931.0),
  which is **EXACTLY IQ-TREE's own full `-m MF` BIC winner** (CPU ground-truth `.iqtree`: F81+F+G4, **w-BIC 0.998**) —
  and the **native-subsample-BIC top-3 = full-data top-3 in order** (F81+F+G4, HKY+F+G4, F81+F+I+G4). **My prior
  "oracle = GTR-family" expectation was WRONG**: generative ≠ BIC-selected. On this data the GTR exchangeabilities buy
  only **~3 nat over 1 M sites** (GTR+F+G4 lnL −59208016 vs F81+F+G4 −59208019), so BIC's parameter penalty correctly
  demotes **GTR+F+G4 to 18th** (and the literal generative model-freq GTR+I+G4 to far worse via `+F` capturing the
  signal). **The native-BIC coarse gate is VINDICATED**: the *old projected-BIC* ranking (the §X.3.2 bug) would have
  refined GTR+F+G4 / GTR+F+I+G4 / TVM+F+G4 — true-BIC ranks **18 / 20 / 14** — a catastrophic recall miss; native BIC
  recalls the true top-3 exactly. **Coverage in production CTF: 70 GPU engagements / 9 CPU declines** on the full set
  (free-Q genuinely on GPU). **Wall 152 s** (subsample 0 + coarse `-m MF` 56 + refine 96) **vs CPU `-m MF` 1122–3077 s
  = 7.4–13×**; **GPU energy 7.53 Wh** (181 W mean, 14.2 GB, exit 0, GPU 60 %). *Honest caveat:* CTF refines model params
  on the **fixed coarse tree**, so the winner's full lnL (−59208077) is ~58 nat below CPU full-MFP tree-search
  (−59208019) — but **model SELECTION (ModelFinder's actual job) is exact**; downstream tree search is a separate step.
  Strong support for the [subsample-sufficiency] hypothesis: 5 k-site ranking reproduces the 1 M-site ranking.

### IX.9.5 Honest positioning (no new throughput claim)
G.6 is a **coverage/capability** milestone: DNA `-m MF` goes from **8 % → near-full** GPU coverage (minus +R/+I+R/pure-+I),
making DNA-MF *meaningful* on the GPU. It does **NOT** change the part5 mutex-serialization ceiling — the `-m MF` wall win
still lives in the **CTF top-k≤3 refine** (depth), not full-set breadth. Free-Q makes the *refine* model-complete for DNA;
the throughput verdict (1 GPU beats a small node count / the stock tool, not a large cluster at full-data breadth) is
unchanged. The clean unconditional GPU win remains 1M/10M scale (bandwidth/efficiency), still gated on the unbuilt tiling.

---

## IX.10 Independent code audit + hardening (2026-06-13, commit `3ec1b5c8`)

An independent adversarial code audit of the G.5.1a + G.6 changeset (commits `65e45c4c..d5d69b48`) was run before the
GitHub source release. **Verdict: the core machinery is CLEAN** — write-back ordering (Q → pinv → α → `clearAllPartialLH`
→ fresh `computeLikelihood`), the process-mutex coverage of the *entire* `gpu_jolt_optimize` body (device constant
symbols + the persistent `DevBuf` pool + the host decompose-callback that mutates the per-thread candidate model),
the base-sweep-skip ↔ Q-FD state coherence (`qApply(qcur)` restores base Q after the FD block AND on the no-accept
break; skip disabled for `nFreeQ>0`), FP64 throughout the reduction/gradient paths, the 1-based `getVariables()`
indexing, and the NaN→CPU fallback wiring (`modelfactory.cpp`) were all verified correct. The safety-gate `rel` was
confirmed to be a *genuine GPU-vs-independent-CPU recompute* (not a tautology). **Two real holes were found and fixed:**

- **RISK-1 (silent-correctness, non-default exposure) — `phylotreegpu.cpp:583`.** The free-Q eligibility predicate
  excluded only `FREQ_ESTIMATE` (+FO), **not the DNA tied-frequency types** (`+FRY`/`+F1112`/… = `FREQ_DNA_*`). For
  those, `getNDim()` includes 1–3 **free frequency** parameters which `gpuGetFreeParams` packs into the tail of the `q`
  vector; the launcher then treats every entry as an **exchangeability** — FD-stepping it and clamping to `[1e-4,100]`
  instead of the correct frequency bound `~[0,1]`. The result is a *coherent-but-suboptimal* lnL that **passes the
  write-back coherence gate** (which checks GPU≡CPU at the written-back point, not optimality) and feeds a wrong value
  into BIC/AIC ranking. These types are **not in the default `-m MF` DNA freq set** (`{FQ,F}`), so there is no live
  regression — but the gate was not defensive. **Fix:** add `nFreqParams(model->getFreqType()) == 0` to `freeQok`.
  `nFreqParams` returns 0 for `+FQ`/`+F` (the default set is unchanged — still 70 engage) and >0 only for tied types
  (which now correctly decline to the CPU). Statically confirmed against `tools.cpp:8207`.

- **RISK-3 (defensive) — `phylotreegpu.cpp:751`.** `if (rel > 1e-6)` is **false when `rel` is NaN** (e.g. the CPU
  recompute underflowed), so the safety gate did **not** fire — and worse, it then fell through to
  `setCurScore(cpuLnL)` = `setCurScore(NaN)`, **poisoning `_cur_score`** before returning. **Fix:**
  `if (!(rel <= 1e-6)) return NAN;` — a NaN/inf `rel` now trips the CPU fallback and returns **before** `setCurScore`.

**Validation (job 170863975, V100, rebuilt binary md5 `091942126e…`):** `GTR+F+G4 -te` engages at rel **5.193e-12**
(identical to the pre-fix `170796516` rel 5.19e-12 ⇒ the RISK-1 predicate did not touch the `+F` path),
`GTR+F+I+G4 -te` engages (free-Q+I jointly), `JOLT_NO_FREEQ` declines. CPU path untouched (all changes inside the
`params->jolt` GPU path). *NIT acknowledged:* the `kj_derv_fused` `lcc` reassociation (`65e45c4c`) makes the
post-G.5.1a fixed-Q +G path agree within ~1e-16 (not literally bit-identical); the docs already report it as
rel ≤ 2.4e-12, so no overclaim.

### IX.10.1 Next steps (ranked, audit-informed)
1. **G.5.1b — +R / +I+R in-tree JOLT (THE critical path).** The only remaining AA `-m MF` gap (the 8 `+R` declines) and
   the load-bearing multimodal-convergence piece (standalone cold/warm-vs-CPU-EM harness → gate flip + the CPU-optimum
   comparison gate). The gradient (`gz_c`) is FD-validated (G.5.1a); the optimiser branch is increment 2.
2. **Close RISK-2 (coherence-vs-optimality) generally.** The free-Q safety gate currently checks GPU≡CPU *coherence*,
   relying on G.6.0b's offline JOLT≥CPU validation for *optimality*. Fold the **CPU-optimum comparison gate** (assert
   JOLT lnL ≥ CPU-refined − `modelfinder_eps`, else NaN→CPU) — designed in IX.8.3 for +R — into the free-Q path too,
   so a future regime that converges to a worse-than-CPU optimum is caught per-candidate at runtime, not just offline.
3. **Runtime confirmation of the RISK-1 fix.** A `JOLT_DEBUG=1 -m MF` DNA run to confirm `freeQok` never fires with
   `nFreqParams>0` on the default set, plus an explicit `GTR+FRY+G4 -te` that must now log `decline reason=free-subst-params`.
4. **G.5.2 — VRAM tiling of the postorder arena.** Gated on the AA-10M `-m MF` result (job 170856902): if 10M LG+G4/
   LG+I+G4 fits H200 (~58 GB est.) tiling is deferrable; if it OOMs, tiling moves onto the critical path for scale.
5. **Port the native-BIC gate + rate-het detector + wall budget into the production CTF path** (IX.7.1 #4) — the §X.5.5
   fix currently lives only in the benchmark scripts.
6. **Verify the audit's two static-only items:** `cuobjdump` confirming `kj_derv_fused` stays 32 regs / 100 % occupancy,
   and a wider fixed-Q self-check sweep confirming the ~1e-16 reassociation bound holds across (ncat, nptn) regimes.
