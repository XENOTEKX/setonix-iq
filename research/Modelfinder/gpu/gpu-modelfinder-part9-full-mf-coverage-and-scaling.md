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
path** (commit `4cb639a7`). `kj_derv_fused` gained the per-category `Lc(p)` accumulator (+ optional `wnum` output) and
stays **32 registers on the production cards (A100 sm_80 / H200 sm_90) → 100% occupancy** (cuobjdump verification,
IX.10.1 #6 — note V100 sm_70 is 40 regs/~80% occ, acceptable as the dev card; FP64 results are bit-exact regardless);
the gated `JOLT_RGRADCHECK` hook computes `gz_c` and
central-FD-checks it. **LG+R4 (5000-site AA subsample):** `gz_c == FD` maxrel **1.03e-8** (cold) / **4.87e-6** (non-uniform
weights), `Σ_c WN_c = N` **exact** (relWN 0), `Σ_c gz_c ≈ 0` (≤6e-12, the gauge null direction). `g50val` (170777846)
confirms the `lcc` reassociation did NOT regress the gamma `-m MF` path: best = **LG+G4**, all `[JOLT]` self-checks rel
≤ **2.4e-12**. **This validates the gradient math only** — the +R LM optimiser branch + the cold/warm-vs-CPU-EM
convergence gate (IX.8.3, the multimodal risk) is increment 2 and the eligibility gate stays unflipped until it passes.

#### IX.8.5 STATUS — G.5.1a increment 2a (the standalone convergence harness) 2026-06-17
The standalone +R **optimiser-convergence** harness (`gpu_k8c_jolt_freerate.cu`, the make-or-break IX.8.3 gate that must
pass *before* any in-tree eligibility flip — IX.3 #1) is built and run on the LG+I+G4 1M-AA alignment (100K patterns,
fixed `-te` topology), with the **CPU-EM MLE** (IQ-TREE's own RateFree EM on the same topology, GPU-fork CPU path) as
the gold reference. The harness does a joint diagonal-LM over (branches + log-rates + softmax-weights), gauge-fixed
(`rescaleRates`-equivalent Σw·r=1) after every accepted step, from **4 starts** {warm-cpuem, cold-geomA, cold-geomB,
cold-linC}; **PASS = reproducible (≥2 starts agree on the best within rel 1e-9) AND optimal (best ≥ CPU−eps) AND the WN
identity Σ_c WN_c = N exact**. The single-start gate (`cold==warm`) was corrected to this multi-start best-of after run
171515819 showed it was conflating "the optimiser is a *better* optimiser than CPU-EM" (it is, it beats CPU) with a
convergence failure.

| model | best lnL | starts agreeing | spread (nats) | JOLT − CPU-EM | WN identity | verdict |
|---|---|---:|---:|---:|---|---|
| **LG+R4** | −7541972.2410 | **4 / 4** | 0.0000 | **+0.2289** | relWN 0 | **PASS** (reproducible + beats CPU) |
| **LG+R6** | −7541971.7262 | 1 / 4 | 0.5022 | +2.7631 | relWN 0 | **CHECK** (distinct local optima; best still beats CPU) |

**Honest read (job 171516319):** the **gradient is exact** at every ncat (WN identity bit-exact, FD informational on the
full-data lnL), and JOLT **beats CPU-EM at every ncat** (CPU-EM itself gets stuck on local optima at high ncat — JOLT's
best is +0.23/+2.76 nats *above* it). The OPEN issue is purely **reproducibility at high ncat**: at 6 categories the +R
surface has genuinely distinct local optima (the classic near-degenerate-category / label-switching multimodality — R6's
cold solution shows near-duplicate rate pairs), and 4 *spread cold* starts do not all funnel to one global. So per the
pre-commitment (IX.3 #1, *do NOT flip the gate on a non-robust optimiser*), the in-tree scope is the open decision:
**(a) EM-warm-start** seeding (CPU-EM → JOLT-polish; guaranteed ≥ CPU and reproducible-from-the-deterministic-seed for
*all* ncat, but pays the CPU-EM cost and only *polishes* rather than replaces it); **(b) scope in-tree +R to the
reproducible low-ncat range** (R2–R4 confirmed pure-GPU multi-start, beats CPU) with **high-ncat declining to CPU** (as
+I+R already does); or **(c)** more/better-spread cold starts (cheap on GPU) to try to make high-ncat reproducible. In
all cases the **runtime CPU-optimum comparison gate** (assert JOLT lnL ≥ CPU−eps, else NaN→CPU fallback — IX.8.3) is the
correctness backstop: we can never ship a +R model *worse* than the CPU baseline. **The eligibility gate (`:502`) stays
unflipped pending this scope decision.**

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
- **G.6.0a — Q-FD gradient cross-check ✅ PASSED 2026-06-13 (job 170787044, commit `e9498baa`).** Gated
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
- **G.6.0b — free-Q FD-LM + convergence gate ✅ PASSED 2026-06-13 (job 170792611, commit `afc1c5a1`).** The nQ Q-params
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

## IX.10 Independent code audit + hardening (2026-06-13, commit `2b80bec0`)

An independent adversarial code audit of the G.5.1a + G.6 changeset (commits `4cb639a7..d6d68943`) was run before the
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
`params->jolt` GPU path). *NIT acknowledged:* the `kj_derv_fused` `lcc` reassociation (`4cb639a7`) makes the
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
4. **Scale (10M) — there are TWO walls (host RAM AND GPU VRAM); both now addressed (G.7.0 + G.7.1).** ⚠️ **Correction
   to the original job-170856902 reading:** that run (`RSS 193 GB > 180 GB`, GPU 520 MB / util 0 %) was killed in
   **host setup before reaching the GPU**, so it exposed only the HOST wall — and the earlier "the ~58 GB VRAM estimate
   held, VRAM was never the limit" claim was **wrong for 10M**: ~58 GB is the *1M* JOLT footprint; the 10M GPU arena is
   **~886 GB** (part7 §VII.1, the O(nptn) postorder+preorder), which fits NO single GPU. The host wall simply *masked*
   the GPU wall.
   - **Wall 1 (HOST) — FIXED, G.7.0 (commit `b43d5a97`, validated job 170934922).** IQ-TREE sized the `LM_PER_NODE`
     arena (558 GB at 10M) against *physical* RAM, ignoring the cgroup. cgroup-aware `getAvailableMemory()` + the
     `--jolt` lean LM_MEM_SAVE tier brought the host to **78 GB RSS, exit 0** (the GPU does the likelihood; the host
     only needs the recompute-exact self-check). The host no longer kills the run.
   - **…which EXPOSED Wall 2 (GPU VRAM).** With the host fixed, the 10M `-te` reached the GPU `DEVB` calls — and the
     886 GB arena failed to allocate on the H200 (141 GB) → NaN → CPU fallback (**GPU 533 MiB, util 0 %, no `[JOLT]`
     line**, job 170934922). This is the first run that actually saw the GPU wall.
   - **Wall 2 (GPU VRAM) — FIX = G.7.1 PATTERN TILING (part7 §VII, V.A PASSED job 170976732; V.C 10M-on-H200 in flight
     job 170977748).** Splitting patterns into `nTile` chunks shrinks the arena ~`nTile`× exactly (886 GB → fits H200
     at T≥8 / A100 at T≥15), bit-identical to one-shot. **So tiling DOES address the 10M wall — the opposite of the
     pre-host-fix conclusion**, because removing Wall 1 is what made Wall 2 reachable.
4b. **The host self-check at extreme nptn is the remaining WALL-TIME lever (not a correctness wall).** Under G.7.0's
   lean tier the host self-check `computeLikelihood` recomputes on LM_MEM_SAVE slots at 9.4M patterns — lnL-exact but
   slow, and it will dominate the 10M `--jolt -te` wall (a HOST cost, after the GPU optimise finishes). Sampling/skipping
   it at extreme nptn (keeping the GPU's tiled VRAM path) is the throughput follow-up; capability (JOLT engages,
   lnL-exact) is the G.7.1 V.C gate and comes first.
5. **Port the native-BIC gate + rate-het detector + wall budget into the production CTF path** (IX.7.1 #4) — the §X.5.5
   fix currently lives only in the benchmark scripts.

   **🟡 IN PROGRESS 2026-06-17 — single tested helper extracted: `gadi-ci/gpu-modelfinder/ctf_rerank.py`.** The
   native-subsample-BIC rerank + rate-het detector + refine/skip action (the §X.5.5 fix) was re-implemented inline as a
   ~30-line Python heredoc in ≥4 shipping CTF scripts (`run_ctf_1m_mf_energy.sh`, `_dna_energy.sh`,
   `run_ctf_10m_mf_aa_h200.sh`, `run_ctf_freerate_recall_sweep.sh`). It is now lifted **verbatim** into one
   parameterized module with the **same CLI** (`ctf_rerank.py <coarse.iqtree> <m> <N> <K>` → `MODEL:<name>:<action>`
   on stdout, `[rerank]`/`[detector]` diagnostics on stderr) so each script's heredoc collapses to a one-line call.
   **Proven a faithful drop-in:** on the real `ctf1mmfd_a100dnamf/coarse.iqtree` (DNA G.6.2) and `ctf1mmf_a100mf3/`
   (AA-1M) coarse tables, its stdout **and** stderr are **byte-identical** to the live heredoc; native#1 = `F81+F+G4`
   (DNA, G.6.2 winner in order) and `LG+G4` (AA) while the projected diagnostic reproduces the rank-18 / rank-4 bug.
   Carries a `--selftest` (4 fixtures: the §X.5.5 ranking flip, both detector branches, exact integer-p recovery, the
   DNA free-Q case) — ALL PASS. **Remaining:** rewire the bench scripts to call it (one-line replacement each;
   reviewable, no GPU needed) once a CTF run can re-validate end-to-end.
6. **Verify the audit's two static-only items:** `cuobjdump` confirming `kj_derv_fused` stays 32 regs / 100 % occupancy,
   and a wider fixed-Q self-check sweep confirming the ~1e-16 reassociation bound holds across (ncat, nptn) regimes.

   **✅ ITEM-6a — `cuobjdump` register/occupancy DONE 2026-06-17** (production binary `build-gpu-on/iqtree3`,
   md5 `8dd57cfb…`, Jun-14, free-Q + RISK-1/3 fixes; per-arch cubins extracted with `cuobjdump -xelf`):

   | kernel | sm_70 (V100) | sm_80 (A100) | sm_90 (H200) | occupancy verdict |
   |---|---:|---:|---:|---|
   | `kj_derv_fused` | **40** | **32** | **32** | 100% on **A100/H200** (production); ~80% on V100 (dev) |
   | `kj_ratenum` | 31 | 32 | 32 | ≤32 → 100% all arch |
   | `kj_reduce3` | 23 | 22 | 24 | ≤32 → 100% all arch |
   | `kj_reduce_gradnum` | 22 | 18 | 20 | ≤32 → 100% all arch |
   | `kj_invl` | 16 | 16 | 14 | ≤32 → 100% all arch |
   | `k1_node` / `kj_pre` | 56 (+160/320 B stack) | 56 | 40–56 | ~57% — latency-bound by design, never claimed 100% |

   **Honest correction to IX.8.4:** the "32 regs / 100% occupancy" claim holds on the **production cards (A100 sm_80,
   H200 sm_90)** but `kj_derv_fused` is **40 regs / ~80% occ on V100 sm_70** (sm_70 needs 32 regs/thread for full
   occupancy: 65536/2048; the fused `Lc`/`wnum` accumulators push V100 to 40). This is **not a correctness issue**
   (FP64 results are bit-exact on every arch) and V100 is the dev/test card, not a deploy target. Chasing 100% on V100
   via `__launch_bounds__` is **declined** — prior occupancy sweeps (MEMORY: project_gpu_modelfinder) showed reg caps
   regress these kernels via spilling, and the kernel already beats BEAGLE (37.8 vs 45 ms). The all-≤32 reduction
   kernels and the latency-bound `k1_node`/`kj_pre` (never a 100%-occ target) are unchanged. **ITEM-6b** (the wider
   fixed-Q self-check sweep across (ncat, nptn)) needs a GPU run and is queued behind the live G.5.1b +R job.

---

## IX.11 G.8 — Profile-mixture JOLT (C60 / MEOW80 / UDM): a genuine per-model DEPTH case (with honest occupancy caveats) — design + phased plan (2026-06-17, PLAN; red-teamed)

**Trigger.** The first real scientific dataset (Williamson et al., *Nature* 640:974 (2025), the eukaryote root; supermatrix
`CAT_100S93F.phy` = 100 taxa × 22,462 AA) infers its tree under **LG + MEOW80 + G4** — an **80-class profile mixture**
(`MEOW6020.nex`: `frequency ESmodel = FMIX{EShighclass1…ESlowclass20}`, 60+20 = 80 classes). JOLT **declines it to CPU
today** because the eligibility gate (`phylotreegpu.cpp:573`) rejects `getNMixtures() != 1`. This section scopes the
extension that makes profile mixtures run on the JOLT GPU path. **It is an engineering boundary, not a feasibility wall —
and it is a genuine per-model DEPTH case (the regime the GPU already wins). It is NOT an automatic escape from the measured
occupancy/latency ceiling — see the honest caveats in IX.11.2, which a red-team review corrected from an earlier overclaim.**

Source-grounded by a read-only research pass over `/scratch/rc29/as1708/iqtree3-gpu` (`model/modelmixture.*`,
`tree/phylokernelmixture.h`, `model/modelmarkov.cpp`, `model/modelset.*`); all file:line claims below verified there.

### IX.11.0 ⚠ FIRST REAL-DATA RUN (job 171521161, 2026-06-18, H200) — the GPU path did NOT engage on the real supermatrix (a prerequisite finding that gates G.8)

Before any mixture work, the **base** model was run on the real `CAT_100S93F.phy` (100 × 22,462). Honest outcome — **JOLT
silently declined everything to CPU; the GPU never did likelihood work:**
- **A** (fixed tree): `LG+F+G4 -fast` built, lnL −1763745.766, tree length 39.06 (a deeply divergent real supermatrix,
  ~74% occupancy / high per-taxon gaps).
- **B** (`LG+F+G4 --jolt --gpu -te`): GPU lnL −1763745.763 vs CPU −1763745.765, rel 1.13e-9 — **BUT this is CPU≡CPU, NOT a
  GPU validation.** The log shows `Kernel: AVX+FMA - 12 threads`, **`[JOLT]` markers = 0**, and the job's **GPU util = 0% /
  520 MB** (bare CUDA context). The two runs agree only because *both* used the CPU kernel.
- **C** (`-m MF`): also no `[JOLT]`, GPU 0%. And the harness asked for the **full 1232-protein candidate set** (incl.
  +I+R8) on 22,462 sites — it ground through ~768 models in 100 min and **timed out** (so did the CPU baseline). Poor
  harness choice (should have been CTF + a restricted set); but the JOLT-non-engagement is the real issue.
- **D** (`LG+MEOW80+G4 -mdef MEOW6020.nex -mwopt`): the 80-class FMIX **loaded correctly** ("90 models, 295 frequency
  vectors"; aliased to `LG+FMIX{EShighclass1…ESlowclass20}+G4`) and **optimised on CPU** (lnL −1677158 → −1665815,
  climbing) exactly as a mixture should (declines to CPU), then hit the 40-min cap. So the mixture *path* behaves as
  designed — it just isn't accelerated (the whole point of G.8).
- The **447.87 Wh "energy" is an idle-GPU artifact** (H200 ~95 W × 4.7 h with util 0%), not useful work — disregard it.

**The finding:** the *identical* `--jolt --gpu` command engages JOLT on the **synthetic AliSim** AA data (G.7.2: 116/122
engage) but the eukaryote run showed **0 `[JOLT]` engage markers + GPU util 0%.**

**⚠ DIAGNOSIS UPDATE (2026-06-18, jobs 171577643 + 171578515) — NOT the binary, and the first read was over-stated:**
- **Ruled OUT — binary:** `build-gpu-on/iqtree3` (md5 8dd57cfb) is *exactly* the binary the G.7.2 sweep used
  (`run_bench_sweep_clean_h200.sh` → `BIN=$BUILD_ON/iqtree3`), and `bench_h200_clean/AA_1000000/refine_2.stdout`
  carries `[JOLT]` markers. So the binary carries JOLT and engages on synthetic AA — same card (H200), same flags.
- **Ruled OUT — SAFE_LH (the earlier "leading hypothesis"):** retracted as unsupported; 100 taxa ≪ the 2000-taxon
  SAFE_LH trigger, and no scaling-switch line appears. Do not cite SAFE_LH as the cause.
- **Two harness flaws found in *my own* tests, not the GPU path:** (1) the eukaryote `-m MF` job **never set
  `JOLT_DEBUG=1`**, so any per-candidate `[JOLT-GATE] decline reason=…` was suppressed — "0 engage" there could be hidden
  eligibility declines, not a non-engagement. (2) diag-v1 (171577643, `JOLT_DEBUG=1 -te -m LG+F+G4`) fed a tree **already
  optimal under the same model**, so `optimizeParameters` may have had nothing to do ⇒ the JOLT hook
  (`modelfactory.cpp:1597 if(params->jolt) optimizeParametersJOLT`) was never invoked — `[JOLT-GATE] reached hook` absent
  ≠ data-ineligible.
- **Decisive test RUNNING (diag-v2, job 171578515):** `JOLT_DEBUG=1` with **forced** optimization — EUK `-m LG+G4 -te`
  (model ≠ the tree's LG+F+G4 ⇒ real re-opt), EUK `-m LG+G4` full search, and a **synthetic control** at the same
  22,462-site size. Verdict logic: SYN engages + EUK doesn't ⇒ a genuine real-data issue (read the decline reason); both
  engage ⇒ the original "didn't engage" was the `-te`-already-optimal + `JOLT_DEBUG`-off artifact (good news — real data
  works). **Pending that result, treat "JOLT declines on real data" as UNCONFIRMED.**

**NEXT (gates everything below):** read diag-v2 (171578515). **G.8 (profile mixtures) is moot until the
BASE JOLT path engages on real data**; if real supermatrices systematically trip SAFE_LH, the log-sum-exp/SAFE_LH port
(the >2000-taxon caveat noted in G.4.0b) becomes the real blocker, not the mixture kernels. **Do NOT cite this run as a
GPU result.**

### IX.11.1 What a profile mixture *is*, from the source (the 5 facts that define the design)

1. **Per-class eigendecomposition, packed contiguously.** `ModelMixture` *is* a `ModelMarkov` **and** a
   `vector<ModelMarkov*>` (`modelmixture.h:37`); `getNMixtures() = size()` (`:105`). Each class is a full component with its
   **own** eigensystem: `decomposeRateMatrix()` loops the classes (`modelmixture.cpp:2466`) and each calls
   `eigensystem_sym(rate_matrix, state_freq, …)` (`modelmarkov.cpp:1653`) — the **shared LG exchangeabilities** S with the
   **per-class** π_c ⇒ a *distinct* Q_c = S·diag(π_c) and a distinct {eval, evec, inv_evec} per class. They live in **one
   concatenated buffer** sliced per class (`initMem`, `modelmixture.cpp:1552-1595`): class `m`'s eigenvalues at offset
   `m·nstates`, eigenvectors at `m·nstates²`.
2. **The category dimension is the FLATTENED (mixture × gamma) product.** The partial-LH tensor is indexed
   `[ptn][m·ncat + c][state]`, block = `nstates·ncat·nmixture` (`phylokernelmixture.h:55-56,226-227`); the kernel selects
   class `m`'s eigen inside the loop (`eval[m·nstates]`, `evec[m·nstates²]`, tip profile
   `tip_partial_lh[state·nstates·nmixture + m·nstates]`). For `LG+C60+G4`/`LG+MEOW80+G4` `fused_mix_rate == false`
   (`isFused()` is true only when every prop == 1.0, `modelmixture.cpp:2231`) ⇒ the **full N×G Cartesian product** applies.
   So MEOW80+G4 = **80×4 = 320 effective categories** per site.
3. **Under `-mwopt` the only free model params are the N−1 weights.** `ModelMixture::getNDim() = (size()-1) +
   Σ_c (*it)->getNDim()` (`modelmixture.cpp:1894-1923`); for **profiles loaded from a .nex**, each class is fixed-Q with
   `freq_type != FREQ_ESTIMATE` ⇒ per-class `getNDim()==0` (`modelmarkov.cpp:964`). `-mwopt` sets `fix_prop=false`
   (`:1502-1503`) ⇒ the model side optimises **only the N−1 mixture weights** (ratio parametrisation `prop[i]/prop[N-1]`,
   `:2496`). The 80 profiles are **pre-estimated constants, never written by the optimiser.** α and branches are optimised
   in `site_rate`/the tree as usual. **⇒ the N eigensystems are FIXED across the whole optimisation — decompose once, cache
   forever** (unlike free-Q DNA, which re-eigendecomposes every FD step).
4. **CPU's DEFAULT weight optimiser is BFGS (ratio parametrisation), not EM — but EM exists, and BOTH need only the
   per-class likelihood `L_{ptn,c}`.** ⚠ *Red-team correction:* for a non-fused mixture the CPU default is **BFGS**
   (`optimize_alg_qmix="BFGS"`, `tools.cpp:7317`; `-mwopt` does NOT select EM). An **EM** path also exists —
   `optimizeWeights()`/`optimizeWithEM` (`modelmixture.cpp:1983`/`2057`, Wang-Li-Susko-Roger), used when the mixture
   `isFused()` or `-optalg_qmix EM` — using the per-class pattern likelihoods `_pattern_lh_cat`
   (`computePatternLhCat(WSL_MIXTURE)`, `getNumLhCat(WSL_MIXTURE)=getNMixtures()` ⇒ entry `[ptn·N+c]` is the per-class
   likelihood already summed over the G gamma sub-categories): E-step posterior `w_c·L_{ptn,c}/L_ptn`, M-step
   `w_c ← (1/N)·Σ_ptn freq[ptn]·w_c·L_{ptn,c}/L_ptn` (`:2008-2024`). **Either optimiser reaches the same MLE, and the only
   ingredient the GPU must supply is the per-class pattern likelihood `L_{ptn,c}`** — a by-product of the per-class partials.
   The gradient form (if folded into the LM instead) is `∂lnL/∂w_c = Σ_ptn freq·L_{ptn,c}/L_ptn` — identical in form to the
   validated +R weight gradient G.5.1a. (Source caveat: EM is monotone in lnL but `modelmixture.cpp:2457` warns of
   near-zero-weight overfitting and floors weights at `MIN_MIXTURE_PROP` — monotone ≠ failure-free.)
5. **Profile mixture ≠ site-specific (PMSF).** `isSiteSpecificModel()` is `false` for `ModelMixture` and `true` ONLY for
   `ModelSet` (`modelset.h:37`), which is **one model per site** via `pattern_model_map`/`getPtnModelID(ptn)` — the PMSF /
   CAT-PMSF case (no class sum). **G.8 admits profile mixtures and STILL declines PMSF.** (The paper's `GTR+CAT-PMSF+R4` is
   out of G.8 scope on two counts — site-specific AND +R.)

### IX.11.2 Why this is a genuine DEPTH case — and the honest occupancy caveats (red-team-corrected)

A profile mixture is `L_p = Σ_c w_c·Σ_g (1/G)·L_p(c,g)` — for MEOW80+G4 that is **80× the arithmetic** of the LG+G4 we
already accelerate. An earlier draft of this section claimed that 80× arithmetic "breaks the occupancy/latency ceiling" and
that "grid.z finally multiplies here." **A red-team review correctly rejected both as overclaims that contradict part5's own
measured findings.** The honest argument is narrower, and rests on three *structural* advantages plus one *contingent*
opportunity — not on an automatic escape from the ceiling.

**What is genuinely true and load-bearing:**
- **It is a DEPTH case, so the N/S breadth ceiling does NOT apply.** The 21× honest ceiling (part5 §V.2) is a **breadth**
  property — N=103 candidate models serialised on one mutex GPU. The paper's use case is **depth**: infer **one** tree under
  **one** heavy mixture model. There is no breadth-competition (one model, not 103), so the relevant comparison is the
  per-model speedup vs the CPU on that *same heavy model* — the regime JOLT already wins (4.8× single-matrix DEPTH).
- **Fixed profiles ⇒ eigens cached once** (IX.11.1 #3): the N eigensystems never change during weight/α/branch optimisation,
  so decompose all N classes ONCE and reuse — no per-iteration eigen cost, unlike the free-Q DNA path (which re-decomposes
  every FD step). A real structural win specific to fixed-profile mixtures.
- **The branch gradient is linear across classes** (`∂lnL_p/∂b = (1/L_p)·Σ_c w_c·∂L_p(c)/∂b`) ⇒ the existing Ji
  postorder+preorder sweep just **sums the per-class branch gradients** (one extra reduction axis, no new math). JOLT's joint
  all-branch optimiser extends to mixtures with no new gradient theory.

**The honest occupancy caveats (the overclaim the red-team caught):**
- **80× more arithmetic does NOT automatically escape the latency bound.** The shipped single-matrix kernel is
  **memory-latency-bound + scheduler-starved** (part5 P3.0b job 170399634: `math_pipe` 0.08%, `issue_active` ~8%,
  occupancy-capped at 4 blocks/SM by a 56-register footprint). A *naive* mixture kernel — an outer `m` class loop inside
  `k1_node` — just does **80× more latency-bound work at the same low throughput**, and *adds* register pressure (the wrong
  direction for occupancy). The `math_pipe≈0` headroom is a real *opportunity* to hide latency with more arithmetic, **but
  only if the classes are mapped as independent LOW-register resident warps** (the unbuilt P2∥-style occupancy redesign) —
  not by adding an inner loop. **This is a contingent opportunity, not a free win, and it is the open kernel-design risk of
  G.8.0.**
- **grid.z over classes does NOT "finally multiply" at full data.** At the 22,462-pattern width G.8.4 itself uses, **one
  class already block-saturates the SMs** (part5 line 77: `ceil(nptn/256)≈88` blocks per class ≫ the ~2.44-wave saturation
  point; part5 explicitly *discarded* full-pattern grid.z as block-saturated). Stacking 80 classes as extra blocks
  **serialises them across already-saturated SMs** — recreating N/S-style serialisation on the class axis. grid.z over
  classes only multiplies on a **small-pattern subsample** (part5 saturation-inversion, ~480–1000 ptn), e.g. the CTF coarse
  stage — not on the full supermatrix. So G.8 inherits CTF's depth framing, not a new breadth win.
- **Net honest claim:** G.8 runs the *exact* heavy model the paper used, at the GPU's per-model DEPTH advantage (≈4.8×,
  possibly more if the math-pipe headroom is later exploited by a low-register class mapping), with no breadth penalty
  because it is one model. It does **not** escape the occupancy ceiling and does **not** make grid.z newly multiply at full
  data. The win is "the GPU does this one heavy model faster than the CPU does this one heavy model," not "mixtures unlock a
  new class of GPU win."

**Memory: fits H200 with the O(depth) recycling we own, but mind the padding and the non-recycled footprint.** Measured
(ptn=22,462, nodes=100, ns=20, FP64): single LG+G4 O(depth≈42) frontier = **0.6 GB**; MEOW80+G4 **all-class** O(depth)
frontier = **48 GB (unpadded ns=20) → fits H200 (141 GB)**; **class-by-class** (one class resident, accumulate) =
**0.6 GB → fits any GPU**. ⚠ *Caveats (red-team):* (a) if the GPU mirrors the CPU's **AVX-padded ns=24** layout the
all-class figure rises to **~58 GB** (still fits H200) — state which layout the kernel uses; (b) the **non-recycled**
all-node all-class frontier is **~115 GB**, which would NOT comfortably co-reside on a 141 GB H200 ⇒ the O(depth) recycling
(G.4.0b) or class tiling is **mandatory**, not optional, for MEOW80. We own both levers (O(depth) recycling + pattern tiling
G.7.1); **class tiling is *cleaner*** because classes are fully independent (each contributes an additive `w_c·L_p(c)`, no
cross-tile FP64 subtlety). **The 320-regime case overflows fixed-capacity device state** — see the G.8.0 row (the 64-entry
`__constant__` coefficient arrays and the two `ncat>64` guards must move to global memory).

### IX.11.3 The phased plan (G.8.0 → G.8.4) — correctness-first, every phase gated like G.4/G.6

| id | title | deliverable (files) | gate | risk | days |
|---|---|---|---|---|---:|
| **G.8.0** | **N-class mixture lnL** (the K1-for-mixtures gate) | Generalise the eigen upload to **N slices** read from the CPU's **AVX-padded** layout (`modelmixture.cpp:1571-1595` uses `get_safe_upper_limit(ns)` strides — for AA under AVX-512 class `m` eigenvalues sit at `m·24` and eigenvectors at `m·(24·20)=m·480`, **NOT** `m·20`/`m·400`; the GPU mirror must read the padded offsets or mis-slice every class after 0). ⚠ **Fixed-capacity device state must move to GLOBAL memory** (red-team): the per-category coefficient arrays `__constant__ g_val0/1/2[64·NS_MAX]`, `g_catw[64]`, `g_rscale[64]` are **hard-sized for 64 categories** and overflow the 64 KB `__constant__` budget at MEOW80+G4's **320 regimes** (`g_val0` alone = 320·20·8 = 51 KB); AND **two** `ncat>64` guards must be relaxed — the launcher (`gpu_lnl_intree.cu:274`) AND the in-gate `ncat-range` decline (`phylotreegpu.cpp:589`). Add the **mixture axis** to `k1_node` (class-`m` eigen + tip profile `tip[state·ns·N + m·ns]`); class accumulate `L_p = Σ_m w_m·Σ_c (1/G)L_p(m,c)`. **Kernel-design risk (IX.11.2):** prefer a low-register class mapping over a naive inner `m` loop. Reuse the **clean-room** cross-check (`gpuLnLCrossCheckOnce`, reads live padded `model->getEigenvalues()` slices). | GPU mixture lnL == fresh CPU `computeLikelihood` for **LG+C20+G4, LG+C60+G4, LG+MEOW80+G4** on a fixed tree, **rel ≤ 1e-9** (expect ~1e-12, the G.2.0a machine-eps result generalised). NORM_LH confirmed; CPU byte-unchanged. | **high** | 5 |
| **G.8.1** | **Mixture branch gradient + per-class likelihood** | Extend `kj_pre`/`kj_derv_fused` over the mixture axis: branch df/ddf **sum over classes** (linear, IX.11.2); emit per-class `L_{ptn,c}` (the EM/weight-grad numerator — the `wnum` accumulator from G.5.1a, now indexed by mixture class not +R category). | GPU df/ddf (class-summed) == CPU `computeLikelihoodDerv` **rel ≤ 1e-9** (df/ddf, the G.2.1a gate); per-class `L_{ptn,c}` == CPU `_pattern_lh_cat[ptn·N+c]` (`WSL_MIXTURE`) **rel ≤ 1e-9**; identity `Σ_c L_{ptn,c}(+invar) = L_ptn` exact. | med | 4 |
| **G.8.2** | **The weight optimiser** — EM-for-weights inside the joint LM | Branches + α ride the **joint diagonal-LM** (existing G.4.1b machinery, gradient now class-summed); **weights `w_c` updated by the EM closed-form M-step** (IX.11.1 #4) from the GPU's `L_{ptn,c}` — **monotone in lnL, no simplex-gauge or multimodality risk** (the de-risked choice given the +R lesson IX.8.5; the softmax-weight-gradient-in-LM alternative is a *later* optimisation). ⚠ *Red-team:* EM is NOT the CPU's default (default is BFGS, `tools.cpp:7317`), so the gate below is an **MLE-equality** gate, not an algorithm-replication gate; and EM has its own boundary failure mode (near-zero-weight overfitting, `MIN_MIXTURE_PROP` floor) — keep the CPU-optimum backstop. | From **cold AND warm** starts, JOLT mixture MLE (branches + α + weights) **== CPU MLE** (IQ-TREE's own BFGS+brlen reference) **rel ≤ 1e-9** for LG+C60+G4 and LG+MEOW80+G4; **CPU-optimum comparison gate** (assert JOLT lnL ≥ CPU − `modelfinder_eps`, else NaN→CPU — the IX.8.3 backstop). No stall. | med-high | 6 |
| **G.8.3** | **In-tree `--jolt` seam for mixtures** + gate relax | Relax `phylotreegpu.cpp:573`: admit `getNMixtures()>1 && !isSiteSpecificModel() && isMixtureSameQ()` (shared-S profile mixtures), STILL decline PMSF/`ModelSet` and per-class-free-Q mixtures (`Σ_c getNDim()>0`). Write-back: `setProp(w_c)` weights + α via `setGammaShape` + `clearAllPartialLH`; the G.2.1 `theta_computed` coherence rule + the G.6.1 write-back safety gate (rel>1e-6 ⇒ NaN→CPU). | LG+MEOW80+G4 `-te`: `[JOLT]` GPU lnL == fresh CPU `computeLikelihood` **rel ≤ 1e-9** (write-back coherence, no stale cache); CPU path **byte-unchanged**; PMSF + free-profile mixtures still log `decline`. | med | 5 |
| **G.8.4** | **The eukaryote payoff** — LG+MEOW80+G4 on the REAL Williamson supermatrix | `--jolt --gpu -te <tree> -mdef MEOW6020.nex -m LG+ESmodel+G4 -mwopt` on `CAT_100S93F.phy` (100×22,462). GPU vs CPU lnL parity + **wall**. (This is exactly the model the paper used for its main ML tree.) | GPU lnL == CPU lnL **rel ≤ 1e-9**; report wall vs CPU (the *honest* DEPTH win: one heavy mixture model, 320 class×rate regimes — the regime that escapes the occupancy ceiling). | low | 2 |

**First step: G.8.0** (the N-class lnL kernel) — highest value (every later phase needs it). The *validation* is low-risk
(read-only clean-room cross-check, no CPU path touched, no optimiser change), but the *kernel redesign* is the real
engineering risk (high): the AVX-padded eigen-slice stride, moving the 64-entry `__constant__` coefficient state to global
memory for 320 regimes, and the low-register class mapping (IX.11.2) all land here before any optimiser work.

### IX.11.4 Honest positioning (what G.8 does and does NOT claim)

- **G.8 accelerates ML *evaluation* under the mixture (branch + α + weight optimisation), NOT the topology search or the
  bootstrap count.** The paper's heaviest compute was the mixture-model **tree search** (SPR/NNI moves) + **2,000 bootstraps
  (1000 UFBoot2 + 1000 SH-aLRT)**. JOLT makes each per-tree evaluation *inside* that search much faster — and for an 80-class
  mixture that inner evaluation *is* the dominant cost — but it does not change the search algorithm or the replicate count.
  The honest claim is "we massively speed up ML evaluation under the mixture," **not** "we speed up the whole paper."
- **This is a DEPTH win; the N/S breadth ceiling simply does not apply (IX.11.2) — and it does NOT overturn part5.** The
  mutex-serialisation verdict was always about ModelFinder *breadth* (N candidates / one GPU). Running one mixture model is
  the depth case the GPU already wins. The 80× arithmetic is **more per-model work the GPU does at its existing
  (latency-bound) throughput — NOT an 80× speedup**; the per-model advantage over the CPU is the same ≈4.8× DEPTH ratio on a
  heavier model (possibly more only if a low-register class mapping later exploits the math-pipe headroom). No contradiction
  with the 21× breadth ceiling, and no claim to escape it.
- **CAT-PMSF is a separate, later track** (site-specific `ModelSet`, ~22K per-site π — lighter per-site, no class sum, but
  needs per-site π indexing; and the paper pairs it with +R4 = G.5.1b). **UDM** profile mixtures are in-scope (same
  `ModelMixture` shape as C60/MEOW80) once G.8.0-G.8.3 land. **+G is in scope; +R4-on-a-mixture inherits the G.5.1b +R
  optimiser** (compose G.8 × +R after both ship).
- **FP64 parity stays non-negotiable** (the 320-regime sum has more terms; the per-class `L_{ptn,c}` can be small but the
  per-class self-cancellation that saved +R G.4.0b applies — `qp/L_p` stays O(100)); deterministic class-axis reduction,
  never `atomicAdd`. The clean-room cross-check (G.8.0) and the CPU-optimum gate (G.8.2) are the build-gating discipline,
  same as every prior phase.

**Estimated total: ~22 days, 5 phases, gated** (G.8.0 5 + G.8.1 4 + G.8.2 6 + G.8.3 5 + G.8.4 2). The eukaryote dataset
gives G.8 a concrete, citable, real-data payoff (G.8.4) instead of a synthetic one — the first time the GPU path would run
the *exact* model a published study used for its headline result. **Honest framing (IX.11.2, IX.11.4): a per-model DEPTH
win on one heavy model, not an escape from the occupancy ceiling and not a new breadth win.**

### IX.11.5 RESULTS — G.8.0 + G.8.1a ✅ (2026-06-18, commit `2277273d`, iqtree3-gpu local)

**G.8.0 — profile-mixture lnL on GPU is BIT-EXACT vs CPU**, the G.2.0a machine-eps result generalised to mixtures (gate
was rel ≤ 1e-9, expected ~1e-12; got ~1e-16). Job 171604565 (A100, 5000-site Williamson euk subsample, fixed FastTree
tree, `--gpu -te` no `--jolt`):

| model | N | regimes N·ncat | GPU lnL == CPU `computeLikelihood` | rel |
|---|---|---|---|---|
| LG+C20+G4 | 20 | 80 | −380523.970000 | 3.06e-16 |
| LG+C60+G4 | 60 | 240 | −377699.344541 | 1.54e-16 |
| LG+MEOW80+G4 | 80 | 320 | −373611.897285 | 1.56e-16 |

Implementation deltas vs the IX.11.3 plan:
- **New `k1_node_mix` kernel** (`gpu_lnl_intree.cu`) + `accum_child_mix` helper + `gpu_lnl_crosscheck_mix` launcher; host
  bridge `gpuComputeTreeLnLCleanRoomMix` (`phylotreegpu.cpp`). Regime flattening `r = m·ncat + c`; each regime an
  independent Felsenstein sweep with class-`m` eigen; root fold `L_p = Σ_m w_m Σ_c catProp_c (π_m·prod)`,
  `wreg[m·ncat+c]=w_m·catProp_c`.
- **The `__constant__` 64-category overflow (IX.11.3 red-team) did NOT bind the clean-room path** — per-regime arrays
  (`Uinv`/`UinvRowSum`/`freq`/`wreg`/`echild`) are uploaded to **global** device memory, so the 320-regime MEOW80 case
  runs unmodified. ⚠ *This is a property of the clean-room cross-check, not a production kernel.* When the kernel moves
  in-tree (G.8.3) any `__constant__`-resident per-category state still must move to global, and the two `ncat>64` guards
  still must relax for 320 regimes — the IX.11.3 warning stands for the production path.
- **AVX-padded eigen stride handled** in the bridge (reads each class's `getEigenvalues()`/`getEigenvectors()`/
  `getInverseEigenvectors()` via the component `ModelMarkov*` pointers, so padding is intrinsic). `+I` / fused (LG4M/LG4X,
  `isFused()`) / PMSF (`isSiteSpecificModel()`) decline to CPU (return NaN).
- Validation is a **gated one-shot diagnostic** `gpuMixLnLCrossCheckOnce` under `params->gpu && !params->jolt` (mirrors
  G.2.0a; leaves the single-model one-shot unconsumed for `nmix==1`). Used a **separate `.console`** output file — pointing
  it at IQ-TREE's own `-pre <p>.log` clobbers the raw-`printf` cross-check output (a double-logging artifact, same class as
  the G.4.3a "+F never broken" false alarm; cost ~2 diag jobs to re-learn).

**G.8.1a — per-class likelihood for the G.8.2 EM weight numerator is correct.** Two checks (the IX.11.3 G.8.1 per-class
gate, ahead of the branch-gradient half = G.8.1b):
- **GPU self-consistency** `Σ_m L_{p,m} = L_p` (CPU-independent, proves the per-class emission is internally exact):
  max rel **1.4e-14** on all three models (job 171604565).
- **Per-class posterior** `γ_{p,m} = L_{p,m}/Σ_m' L_{p,m'}` vs CPU `_pattern_lh_cat` (`WSL_MIXTURE`): PASS on all three —
  **|Δγ| = 6.84e-13 / 4.32e-12 / 1.00e-12** for C20/C60/MEOW80 (job 171633488; the C20 400-site spot-check was 5.76e-14,
  job 171633391). **The posterior is the correct metric, NOT raw `L_{p,m}`:** CPU `_pattern_lh_cat` is per-pattern
  **scaled** (`scale_num`·`LOG_SCALING_THRESHOLD` underflow protection, active even on the NORM_LH 100-taxon path) while the
  GPU clean-room is **unscaled**, so the raw per-class values differ by `exp(scale_p)` per pattern (this is exactly why the
  lnL still matches bit-exact — the factor returns in the log domain). Self-normalising each side cancels `exp(scale_p)`,
  and `γ` (the EM E-step responsibility) is precisely what the G.8.2 M-step `w_m_new = Σ_p f_p·γ_{p,m}/Σ_p f_p` consumes.

**Still ahead:** **G.8.1b** = the mixture branch-length gradient (`k2_derv_mix`: df/ddf **summed over classes** vs CPU
`computeLikelihoodDerv`, the G.2.1a rel ≤ 1e-9 gate) — the other half of G.8.1 — then G.8.2 → G.8.3 → G.8.4 as planned.
