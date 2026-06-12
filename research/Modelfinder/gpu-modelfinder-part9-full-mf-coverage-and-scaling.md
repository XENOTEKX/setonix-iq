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

**Next:** A100 1M wall measurement (the A100<np8 lever check), then G.5.0 Part B, then G.5.1a's standalone +R
convergence harness before any in-tree +R wiring. DNA free-Q is G.6, a separate phase, to keep G.5 (AA-MF) shippable.
