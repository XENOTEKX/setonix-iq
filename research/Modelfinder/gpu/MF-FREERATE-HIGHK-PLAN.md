# Making high-K FreeRate (`+R5`–`+R10`) fast in GPU ModelFinder — research verdict + gated plan

**Date:** 2026-07-10 · **Status:** RESEARCH COMPLETE, plan gated on one decider (job 173506053) + one mechanism probe (173503842). No product code written. · **Author trail:** 3 research agents (source-forensics / optimisation-literature / log-forensics-Amdahl) + 1 hostile red-team, all source- or job-grounded; synthesis + two from-existing-data derivations by the lead.

This closes the loop the user opened: *"why do tree-search `+R`/`+I` run fast (they use JOLT too) but ModelFinder `+R` grinds to the walltime, and can we port the fix — even a novel one — keeping ModelFinder's constraints?"*

---

## ⭐⭐⭐ RESULTS 2026-07-11 — GATE A = GO. High-K `+R` IS on ModelFinder's real critical path; the Amdahl "NO" is OVERTURNED.

**The Amdahl question is answered — and it reverses the prior verdict.** Gate A (`gems_mf_ladder_realdata.sh`, job **173506691**, A100; the first job `173506390` died on a missing-numpy import, fixed via `module load python3/3.11.7`) ran the fixed-tree `GTR+F+R2..R10` BIC-vs-K ladder on **real avian TENT** subsamples:

| N sites | argmin_K BIC | vs `+G4` | nesting |
|---|---|---|---|
| gamma-sim DNA-100K (A3 control) | **R4** | +G4 wins | R7/R10 tiny viol | ← control PASSES (must bottom ≤4) |
| **avian 100K** | **R5** | argmin beats +G4 by **7,942 BIC** | viol@R6 |
| **avian 1M** | **R10** | argmin beats +G4 by **82,655 BIC** | viol@R6 |
| gamma-sim AA-22K (A3-AA control) | **R4** | +G4 wins (−45 BIC) | ULP viol@5,6,10 | ← control PASSES |
| **real AA-22K** (mf-ladder job **173515407**, A100) | **R9** | argmin beats +G4 by **12,467 BIC** | **none (clean)** |

- **The AA arm (job 173515407, ✅ exit 0) EXTENDS Gate A to amino acids — the "high-K +R is on the REAL-DATA critical path" verdict is no longer DNA-only.** Real 22K-site protein selects **R9** (R10 only 0.262 BIC behind, ladder still descending at the cap), beating +G4 by 12,467 BIC, with a **fully clean converged ladder (zero nesting violations)** — a *stronger* result than the DNA/avian cells, which won *despite* R6 violations. The gamma-sim AA control bottoms at R4 (harness valid; the ULP viol@5,6,10 are optimiser noise at the saturated top where all K≥4 sit at lnL≈−1691117.58).
- **BONUS — A4 GPU-JOLT vs CPU-EM on real AA (quality AND speed, both win):** GPU reaches **equal-or-better lnL at every K** and the gap GROWS with K (R4 =0.000, R6 +0.071, R8 +0.288 nats better) while running **9–11× faster** (R6 15s vs CPU-EM 137s; R8 51s vs 562s). So on real protein the GPU optimiser is not merely faster — it finds a *better* high-K optimum than the CPU-EM reference. Directly reinforces L7 Stage A (the cap-lift ships a win, not just parity).
- **High-K `+R` is decisively on the real critical path, and the need CLIMBS with N** (DNA: R5 @100K → R10 @1M; AA: R9 @22K) — the measured realisation of `penalty∝ln N` vs `gain∝N` (⇒ `argmin K→∞` with N). At 1M the **maximum tested category (R10) wins and the ladder is still descending** ⇒ the true optimum on full 37.35M-site avian is likely **> R10** (the cap itself may bind).
- **The gamma-simulated evidence that produced the earlier "NO" was tautological** (confirmed: the A3 control bottoms at R4 exactly as gamma data must). The lead's objection held.
- **GUARD-1 makes the case STRONGER:** the R6 nesting violations prove high-K is *non-converged* ⇒ BIC *under-credits* it ⇒ it wins **despite** being handicapped. A converged high-K fit wins by more. **The models ModelFinder needs are exactly the ones pinned at the 401-iter cap.** ⇒ the speed lever is now JUSTIFIED, not hypothetical.

**fr-profile (job 173506349, H200) CONFIRMS the mechanism + closes the red-team's Claim-2 gap.** Full-ladder `--jolt-diag`, R5/R7/R9/R10 **measured** (was extrapolated): `iters=401` for **all** of R5–R10 (`nRej`≈200–207, `nLnLEval`≈602–608); R4=38, G4=13. Wall affine in K (H200: R5 56s→R10 103s, ~9.4s/cat). **The host-vs-GPU question is settled by the diag's own `device_s`/`host_s` columns: `host_s` is FLAT at 0.058–0.060s across all K while `device_s` is ~97% of wall (R8: 81.7/84s).** ⇒ **the +R8 sweeps are GPU-bound; host-serial LM is NOT a bottleneck** ⇒ the ~10× prize is pure GPU-compute from cutting iterations, exactly as assumed; the conditioning fix need not chase host time. (nsys sub-phases emitted no kernel data — harness issue — but `device_s/host_s` answered the load-bearing question without them; the kernel-TYPE split is re-queued.)

**Frozen-rate probe (job 173506391, A100) = INCONCLUSIVE, re-running.** R4=24, R6=388, R8=**401**. But only 1 diag line/K and R8 at the *model* cap (401) not the *brlen* cap (390) ⇒ per GUARD-2 the frozen `freeRate==3` path did not cleanly engage/separate. Cause: the same `-pre`/stdout-redirect **filename collision** (both write `probe_R$K.log`) that was fixed in fr-profile is still in this pre-existing script, clobbering all but one diag line. **Vacuous, not NO-GO.** Now that Gate A = GO this gate is load-bearing ⇒ re-run with the redirect fixed (capture ALL diag lines; the ~390 frozen cluster then separates from the 401 joint cluster by its cap value).

**grad-validate (re-run job 173514788, A100, COMPLETE 7h05m) = ✅ GRADUATION SCIENTIFICALLY VALIDATED; the harness "❌ do-NOT-promote" is a TEST-ARTIFACT, not a regression (2026-07-11).** All 6 cells (AA_R4/AA_R3/DNA_R4/AA_IG/DNA_IG + AA_IGB `-B 1000`) run to completion. Two tiers:
  - **CLAIM 1 (LOAD-BEARING: default-ON == the validated flag-ON GPU path) = PASS, bit-identical on ALL 6 cells** — RF=0, lnL string-identical, GPU-engage decline=0 on both arms. This is the claim that actually proves the graduation safe: flipping *which side* of `JOLT_RBRLEN`/`JOLT_IBRLEN` is the default does NOT change the validated GPU behavior — incl. the newly-engaging **L5 R3, L5 DNA+R (GTR+R4), and L6 -B +I+G** paths. Clean.
  - **CLAIM 2 (killswitch == prod, CPU-decline path) = 3/6 cells trip the harness's byte-identity bar, but ALL are RF=0 (topology-identical) with IMMATERIAL drift:** `DNA_R4` rel **5.27e-13**, `DNA_IG` rel **8.61e-12**, `AA_IGB` **lnL Δ=0** (identical) with only a ULP branch-length digit differing — every one **≥10 orders below the `loglh_epsilon=1e-3` accept bar**. The console's "RF=differ" is a *parse fallback* (the `rf()` fn byte-`diff`s treefiles first, and its RF-string grep didn't match) — the actual `rf_*.rfdist` files read **`Tree0 0`** = RF **0** on all three.
  - **ROOT CAUSE (source-audited this pass): GRAD (`56ff1e95` @ `05d8ab61`) and BL8 (`fe5ce648` @ `55743479`) are DIVERGENT source trees, NOT "build-l8 + a 2-line flip" as the script's line-4 comment claims** (`git merge-base --is-ancestor 05d8ab61 55743479` = false; build-l8 carries REM/high-K commits GRAD's tree lacks). Byte-identical CPU-decline output across two *different* builds' codegen (inlining / FP-contraction / reduction order) was never achievable — the `memcmp`-strict bar is only meaningful for a *same-source* pair. **GUARD-8 (below) pre-flagged this exact class** ("sub-printed-precision drift... do NOT upgrade to bitwise proven"). NET: graduation is topology- and within-tolerance-identical everywhere; the "fail" is the wrong binary *pair*, not a behavior change.
  - **Clean confirming re-run (cheap, DEFERRED behind the kernel decider):** rebuild GRAD from EXACTLY build-l8's tree (`55743479`) + only the 2-line default flip → CLAIM 2 then must be byte-identical; **or** re-run `bl8_def` twice to measure the multithreaded CPU-decline path's own run-to-run reproducibility at `-nt 12` (isolates codegen-divergence from OpenMP-reduction nondeterminism). Neither changes the graduation decision — CLAIM 1 already carries it.

**NET:** the workstream is no longer gated on "is it worth building" (Gate A = YES, decisively). L7 Stage A (ship the cap-lift) is doubly justified — real data needs R10.

**⚠️ FROZEN-RATE PROBE (re-run 173514787) = VACUOUS, KILLED (2026-07-11).** Diag confirms it runs the **JOINT** fit, not the frozen-rate path: `[JOLT] model=LG+R6 … 401 joint iters` (`fixedlen=0`). `JOLT_RBRLEN=1` does NOT force `freeRate==3` inside a model-*fit* context (`-m LG+RK -ninit 20`) — that path only lives in tree-search brlen reopt. So R4=55(conv)/R6=401(cap)/R8=401 merely REPRODUCE the known joint behavior (GUARD-2 realized: the probe never engaged the frozen path). **The conditioning-fix build-or-bury is therefore UNRESOLVED by this probe.** A clean test needs a redesign (fix +R rates via model string + `-te`, brlen-only, count iters) OR a code-level call to the L5 path. **Soft signal:** L5 (`freeRate==3`) is already VALIDATED at 64.6× in tree search (job 173428381) — a capping frozen-rate branch-LM could not be that fast, weakly implying the branch problem is NOT the wall (⇒ the rate-coupling is ⇒ conditioning fix not obviously dead). **DECISION: DEFER** — conditioning fix is SECONDARY to the CTF-coarse-overhead work (the active lever); L7 Stage A already delivers the +R correctness win. Do not build the conditioning fix without a clean frozen-rate measurement.

### mfp-parity 100K — deployment pipeline vs Hashara's GPU port (`-m MFP -B 1000`, jobs 173500685 DNA / 173500686 AA, H200)

GUARD-7 applied: **correctness verified before speed.** Both arms select the **same model**, produce **RF=0 (identical) trees** at nt1 AND nt12, and JOLT's lnL ≥ Hashara's — so the comparison is legitimate.

| data | model (both) | JOLT lnL | Hash lnL | JOLT nt1 | JOLT **nt12** | Hash nt1 | Hash nt12 |
|---|---|---|---|---|---|---|---|
| DNA-100K | F81+F+G4 | −5692984.526 | −5692984.532 | 261.6s | **223.2s** | 520.7s | 2630.9s |
| AA-100K | LG+G4 | −7541976.852 | −7541976.860 | 1328.9s | **635.2s** | 1199.6s | 1950.4s |

**PER-PHASE decomposition (wall-clock) — the win is NOT uniform; it separates into an algorithm win and a kernel win:**

| data | thr | Parsimony (JOLT / Hash) | ModelFinder (JOLT / Hash / ×) | Tree search+1000 UFBoot (JOLT / Hash / ×) | Total (×) |
|---|---|---|---|---|---|
| DNA | nt1  | 0.10s / 0.82s+9.5s fML  | 35.9s / 245.8s / **6.8×**  | 212.3s / 271.5s / 1.3× | 261.6/520.7 = 2.0× |
| DNA | **nt12** | 0.10s / 0.82s+20.5s fML | **47.3s / 2346.9s / 49.6×** | 172.2s / 280.8s / **1.6×** | 223.2/2630.9 = 11.8× |
| AA  | nt1  | 0.22s / 1.30s+12.0s fML | 503.2s / 376.7s / **0.75×** (JOLT slower) | 729.3s / 814.6s / 1.1× | 1328.9/1199.6 = 0.90× |
| AA  | **nt12** | 0.22s / 1.30s+18.5s fML | **220.1s / 1092.9s / 5.0×** | 399.6s / 848.7s / **2.1×** | 635.2/1950.4 = 3.07× |

- **ModelFinder phase = the CTF *algorithm* win, not a kernel win** (JOLT subsamples to 5000 sites, Hashara tests on full 100000 — GUARD-7's exact point; Hashara could adopt CTF too). Further INFLATED on DNA by Hashara's ModelFinder **threading pathology** (nt1 245.8s → nt12 2346.9s = 9.5× SLOWER with threads). At nt1 on AA, JOLT's MF is actually **slower** (503 vs 377s) — CTF subsample overhead doesn't pay single-threaded on 20-state data; JOLT wins AA only once threads feed the GPU.
- **Tree-search phase = the defensible *kernel* win: a steady 1.6–2.1× at nt12** (DNA 1.6×, AA 2.1×) — the part attributable to JOLT's GPU likelihood kernels, robust across data type.
- **Parsimony:** negligible; JOLT uses stepwise-addition (0.1–0.2s), Hashara uses PLL parsimony (0.8–1.3s) **plus** an extra fast-ML tree-search step (9–20s) JOLT skips.
- **Two total framings, both honest:** same-thread nt12 = DNA **11.8×** / AA **3.07×** (inflated by Hashara neg-scaling); best-config (JOLT@nt12 vs Hash@nt1) = DNA **2.3×** / AA **1.9×** (defensible deployment number). **Corrected headline:** the *kernel* is a robust **1.6–2.1×** (tree search); the giant ModelFinder ratios are a **CTF-algorithm** win amplified by Hashara's threading regression, not a 50× kernel. Simulated-gamma data ⇒ model is `+G4`; validates the *deployment pipeline*, NOT the high-K `+R` selection question (`173515407`, AA arm).

### mfp-parity 1M — DNA (job 173500688) — the win SHRINKS at scale (honest)

GUARD-7 passes: same model (F81+F+G4; CTF path obscures JOLT's model string but lnL+RF confirm identical), **RF=0**, JOLT lnL (−59208019.102) ≥ Hashara (−59208019.248).

| DNA-1M | ModelFinder (JOLT / Hash / ×) | Tree search (JOLT / Hash / ×) | Total (JOLT / Hash) |
|---|---|---|---|
| nt1  | 74.9s / 1010.3s / **13.5×** | 2917.2s / 1951.3s / **0.67×** | 3193.6s / 2995.4s = 0.94× |
| **nt12** | 69.4s / 2374.9s / **34.2×** | 2394.1s / 894.0s / **0.37×** | 2514.4s / 3292.4s = 1.31× |

- **The result INVERTS vs 100K.** CTF wins ModelFinder by an even bigger **34×** at 1M (coarse subsamples 5000 of 1M = 200× data reduction), BUT **JOLT *loses* the tree-search phase (0.37×: Hashara 2.7× faster on DNA-1M)** — the opposite of DNA-100K where JOLT tree search *won* 1.6×. At 1M the tree search dominates the total, so the huge CTF win only nets **1.31× same-thread / 1.19× best-config** (JOLT nt12 2514s vs Hash nt1 2995s). At nt1 JOLT even loses overall (0.94×).
- **Mechanism:** DNA tree search is JOLT's weak axis (4-state kernels amortize the GPU poorly; consistent with the tree-search project's "DNA marginal ~1.03×"). At 1M sites the full-data tree search is the long pole, and JOLT's DNA tree-search kernel is slower than Hashara's there. Hashara's threading pathology also flips: at 1M its tree search scales WELL (nt1 1951→nt12 894s = 2.2×) while its ModelFinder still regresses (1010→2375s).
- **HONEST DEPLOYMENT HEADLINE (DNA-1M): ~1.2×**, driven entirely by CTF ModelFinder; the tree-search phase is a net LOSS for JOLT on DNA at scale. **Contrast pending: AA-1M (687) — AA tree search is JOLT's STRONG axis (NS-template 2.3×), so the AA-1M total should hold up far better than DNA-1M.** This is the DNA-vs-AA tree-search asymmetry (banked in the tree-search project) showing up in the MFP deployment number.

### AA arm of Gate A — the open question the DNA Gate A did NOT answer (user's catch, 2026-07-11)

The Gate-A win above is **DNA-only** (avian). "Does a REAL AA dataset ever *select* high-K `+R`?" is UNMEASURED. Correction banked: the "15–25× GPU beats CPU-EM" (job 173435097) **is** a real AA number (AA-100K LG+R5–R10, GPU faster + strictly better lnL, CPU-EM even violates nesting) — but it is a **FIT-speed** win, not proof AA *selects* high-K. The only AA high-K data prior was gamma-sim (true model +G4 by construction; lnL moves just 0.55 nats R4→R10). ⇒ **job 173515407** (`gems_mf_ladder_aa.sh`, dgxa100): `LG+F+R2..R10` ladder on **real** Williamson CAT_100S93F (100tx×22,462 AA) vs a same-N gamma-sim control. Asymmetric test: **argmin≥R5 ⇒ closes it affirmatively**; **argmin≤R4 ⇒ inconclusive** (small-N suppression, 22462 is ~1600× smaller than avian) ⇒ needs a bigger real protein matrix (Misof 2014 ~584K AA sites identified; Dryad now auth-walls downloads — deferred).

### CTF COARSE-STAGE OVERHEAD — a SEPARATE lever from +R (debugged 2026-07-11, plan+red-team)

**Symptom:** AA-100K `-m MFP` coarse ModelFinder = 1232 protein models on a 5000-site subsample at **388 ms/model** (nt1), SLOWER per model than Hashara on FULL 100000 sites (306 ms/model). Winner is **LG+G4 — NOT +R** (only +R2 in the coarse set, it loses). So this is **NOT the +R workstream** — L5/L6/L7 do not touch it. JOLT on 1/20th the data costs more/model ⇒ **overhead-bound, not compute.**

**Mechanism (source-verified, `iqtree3-l0`):** the eigensystem is a single process-global `__constant__` slot (`gpu_lnl_intree.cu:34-43`), re-uploaded per model (`:2628`), so a **process-wide mutex** (`:2613`) forces "JOLT models run one at a time" — the comment names cross-model batching as deferred "PHALANX grid.z, G.4.3." Per model also pays host eigendecomp + echild rebuild (`:2793`). On ~20-block subsample launches this is launch/host-overhead-bound; nt12 recovers 2.3× by parallelising host launch-issue.

**Two candidate fixes (Plan agent, red-teamed):**
- **FIX #3 — CPU-coarse / GPU-fine hybrid (CHEAP, front-runner).** Route the coarse `evaluateAll` (`phylotesting.cpp:1574`) to CPU (the GPU/CPU switch is dynamic per-model at `modelfactory.cpp:1597`), keep the full-data fine refits on GPU. 🔴 **RED-TEAM SEV-1 (SHOWSTOPPER, corrected):** flipping **only** `params.jolt` false while `gpu=true` does NOT go to CPU — it defeats the guard `phylotreegpu.cpp:1893 if(params->jolt) return;` and installs the **stateless GPU funnel** = "~25 min/model = 50-100× (timeout)" (documented at `:1887-1892`). **The fix MUST save/restore BOTH `params.jolt` AND `params.gpu` false** for the coarse window. Correctness: coarse only RANKS (winner re-scored on full data), CPU path already load-bearing (+I/+R/mixture already CPU-fall-back at `modelfactory.cpp:1614`). SEV-3: drop "bit-identity" language — coarse rank can flip at an exact top-K tie (~0.46-nat-spaced candidates, part10 §X.5.5); default-OFF keeps production byte-identical. **Speed UNPROVEN at nt1** (SEV-2): CPU coarse is fully serial at nt1 + `n_pinv_starts` reverts 4→10 (`modelfactory.cpp:1382`) ⇒ could LOSE at nt1; gated on the diagnostic's new CPU-nt1 arm.
- **FIX #1 — batch K models per launch (EXPENSIVE, shelved).** Prior spike `spike/l2_0_spike.result.172388123.txt` = 1.11× batched / 1.22× streams vs a 3.0× gate = **NO-GO**, but at 391 blocks (SATURATED); the coarse is ~20-block launch-bound where the ceiling is higher, so that spike is the WRONG regime (rationale corrected — do NOT cite it as proof). Red-team SEV-5: **streams is NOT a cheaper substitute** (hits the identical `__constant__` de-globalisation wall + 64 KB constant budget). Whether ANY GPU-batch helps depends on the host echild/eigendecomp fraction (the diagnostic's echild tax). Shelve behind a compile-only spike + the diagnostic.

**Diagnostic (job 173516356, dgxa100, `gems_ctf_coarse_diag.sh`, red-team-corrected):** standard `-m MF` on pre-subsampled AA (faithful coarse proxy, no fine confound — validated: same `CandidateModelSet::generate`, `-m MF`=TESTONLY no tree-search). Measures: AXIS-1 wall vs **measured nptn** (SEV-4 fix, not raw M — else pattern-saturation masquerades as overhead-bound); AXIS-2 per-LAUNCH iters (SEV-6: not per-model); **CPU nt1 fair arm** (SEV-2 fix) + CPU nt16. **Gates:** CPU-coarse beats GPU-coarse *at nt1* ⇒ build FIX #3; wall-flat-vs-nptn ⇒ overhead-bound confirmed; echild tax large ⇒ no GPU-batch helps (only CPU-coarse). Build only after GREEN.

**✅ RESULT (job 173516356, COMPLETE 34min, exit 0, 80% GPU util, 2026-07-11):**
- **AXIS-1 = OVERHEAD-BOUND, CONFIRMED.** GPU nt1 wall vs measured nptn: M2500 200s/2442ptn, M5000 254s/4859, M10000 356s/9724, M20000 540s/19404 ⇒ **wall grows ~half as fast as distinct patterns (2.13× wall for 3.99× nptn; ~1.3–1.5× wall per 2× nptn across the whole range).** Fixed per-model cost dominates (1232 short fits, AXIS-2 mean ~18 iters/launch, min 1–max 165, **zero ≥300**) = launch/host-overhead-bound, NOT per-pattern compute. Same signature as the tree-search subsample-null (172501017) and the per-edge-sync tax — a project-wide pattern.
- **🔴 FIX #3 (CPU-coarse hybrid) GATE = FAILED ⇒ DO NOT BUILD.** The gate required CPU-coarse to *beat* GPU-coarse at nt1. It LOST: **GPU nt1 254s vs CPU nt1 352s = GPU 1.39× AHEAD** (and tied at threads: GPU nt12 166s vs CPU nt16 169s = 1.02×). FIX #3's premise ("GPU coarse is inefficient single-threaded ⇒ offload to CPU") is REFUTED — GPU coarse beats CPU coarse even at nt1. The cheap front-runner is dead.
- **⇒ NET:** the coarse screen IS overhead-bound (so there IS fixed per-model overhead to attack) but the ONLY surviving lever is **FIX #1 — cross-model batching** (de-globalise the process-wide `__constant__` eigensystem mutex `gpu_lnl_intree.cu:2613`, PHALANX grid.z / G.4.3), which is the EXPENSIVE/shelved one. Shrinking the subsample is also weak (M2500 vs M5000 saved only 21% wall for 50% fewer patterns = diminishing returns). **The cheap CTF-coarse-overhead path is CLOSED; the remaining win is a real de-globalisation engineering project, not a quick route.**

---

---

## 0. TL;DR (read this, then the section that matters to you)

1. **Why tree-search `+R` is fast and ModelFinder `+R` is slow — settled at source level.** Tree search only ever runs the **frozen-rate branch block** (L5, `freeRate==3`): rates are seeded and held, so the optimiser solves a `+G4`-shaped, well-conditioned branch problem. ModelFinder must **fit the rates**, so it takes the **joint** diagonal-LM path (`freeRate==1`), which is the one that stalls. Porting L5 to ModelFinder is therefore **not** a kernel port — it is an optimiser-structure problem.
2. **The stall is fully diagnosed.** The joint LM uses **additive Levenberg damping `g/(|dd|+μ)` with ONE shared μ** across branches, log-rates, softmax-weights, α, pinv, Q (`gpu_lnl_intree.cu:3246-3259`), a **single global accept** on total lnL (`:3266`), and **`μ×4` on reject** that crushes every arm together (`:3273`). The `bᵢ·r_c` product structure makes `∂²L/∂bᵢ∂r_c` dense, so the diagonal (Jacobi) model is wrong and converges at rate `ρ(I−diag(H)⁻¹H)→1`. `+G4` escapes on all counts (mean-1 rates → no gauge, 1 well-conditioned param). The **scale gauge** (`Σw_c r_c=1`, imposed only *post-accept* by `gaugeFix` at the sole call site `:3269`) is **real but secondary** (an exact symmetry has zero gradient, so it cannot drive the ⅓ rejections).
3. **Two results derived from data already on disk (no new jobs):**
   - **R5..R10 all sit at the 400-iter cap.** H200 walls are exactly affine in K (`R²=0.999994`); fit on the counter-instrumented R4/R6/R8 predicts R5/R10 to <0.75%. With `iters=401` directly measured at R6/R8, this shows the whole ladder is capped. **Ceiling for any convergence fix = ~10× on optimiser compute, uniform in K.**
   - **A free convergence proof.** `+R(K+1)` nests `+R(K)`, so lnL must be non-decreasing in K. GPU violates by 0.026 nats (R5); **CPU-EM violates by up to 2.90 nats at every K≥5**. Neither implementation checks this invariant; it explains "GPU beats CPU-EM on lnL" and is a shippable correctness win on its own.
4. **The EM family is DEAD (do not reopen).** Stage B (weight-EM) neutral; Stage C (SQUAREM) cancelled; **Stage D (rate-EM) NO-GO — 2.5× slower, FD-slope FAIL, still 401 iters** (job 173444797). Both arms hitting 401 says the disease is the **coupling structure**, not the rate math.
5. **The genuinely novel, untested lever = the CONDITIONING-FIX family** on the *existing joint loop*: **per-class / Marquardt-multiplicative μ + null-space (gauge) projection of the step + geodesic acceleration.** No new kernel, no CPU fallback. This is what the numerical-optimisation literature actually prescribes for this pathology, and **none of Stage A/B/C/D touched it.**
6. **BUT — is any of it worth building?** The Amdahl question (*is high-K `+R` on ModelFinder's critical path?*) is **UNANSWERED**, because every completed real-data model selection in this project ranked models on a **5000-site CTF subsample** (which structurally suppresses high-K), and every full-data `-m MF` was on **gamma-simulated** data (where `+G4` winning is tautological). **Gate A (job 173506053)** answers it on real avian data. Nothing gets built until it returns.
7. **Shippable regardless:** L7 Stage A (lift the cap for correctness/reproducibility) — R5–R10 on GPU, 3/3 bit-identical, beats CPU-EM 15–25× on wall AND on lnL. This is insurance, independent of the speed question.

---

## 1. The mechanism (source-forensics agent, all `file:line` in `/scratch/rc29/as1708/iqtree3-l0/`)

`gpu_jolt_optimize` (`tree/gpu/gpu_lnl_intree.cu:2596`), inner LM backtrack `:3245-3273`:

| parameter class | step (all `g/(|dd|+μ)`) | line |
|---|---|---|
| branch lengths | `base[v] + g_df[e]/(|g_ddf[e]|+μ)` | :3246 |
| α (+G) | `baseA + ga/(|ddA|+μ)` | :3247 |
| pinv (+I) | `baseP + gradPinv/(|ddP|+μ)` | :3248 |
| Q / exchange. | `qcur[k] + gradQ[k]/(|ddQ[k]|+μ)` | :3250 |
| **log-rates `y_c`** | `baseY[c] + g_y[c]/(|ddY[c]|+μ)`, `r=exp(ny)` | :3257 |
| **weight logits `z_c`** | `baseZ[c] + g_z[c]/(|ddZ[c]|+μ)`, softmax | :3259 |

- **One shared μ** (`μ=1.0` at `:3138`), **one global accept** `if(ln>lnL+1e-9)` (`:3266`), `μ×0.5` accept (`:3270`) / `μ×4` reject (`:3273`). Curvatures `ddY/ddZ/ddA/ddP/ddQ` are **secant** estimates (`:3200-3207`) — so drift along a flat direction *corrupts* them.
- **The gauge:** rates are unconstrained `log r_c`; `Σw_c r_c=1` is imposed only by `gaugeFix()` (rescale rates to mean 1, fold reciprocal into branches — lnL-invariant), whose **sole call site is `:3269`, post-accept**. So `(b·κ, r/κ)` slides lnL-invariantly through every trial + all 14 backtracks. **Third null direction found by the red-team:** softmax shift-invariance `z_c→z_c+δ` (`:3263`) — same character, secondary.
- **+G4 is gauge-free:** `jolt_discreteGammaMean` (`:2557-2564`) telescopes to mean exactly 1; α is the only free param. ⇒ 13 iters vs `+R`'s 401.
- **Cap:** `brlenMaxIter=400` default (`tree/phylotree.h:2123`); ModelFinder enters at `modelfactory.cpp:1613` `optimizeParametersJOLT(fixed_len)` with `brlenOnly=false` ⇒ the joint `freeRate==1` path. Convergence test = **ΔlnL<1e-7 only** (`:3138`); **no gradient-norm test, no trust-region gain-ratio ρ anywhere** (grep-confirmed). `--jolt-diag` prints only an end-of-solve summary (`:3293`): `iters= nRej= nLnLEval= echild=` — **no per-iteration lnL trace exists.**

**Why tree-search is fast:** the frozen-rate branch block is `optimizeAllBranchesJOLT` → `freeRate==3` (L5), gated by `freeRateBrlenOK` requiring `brlenOnly==true` (`phylotreegpu.cpp:2042`), reachable only from the NNI tree-**search** loop. Under a `-te`/ModelFinder model-**fit**, `brlenOnly=false` ⇒ that gate is unreachable ⇒ the joint path runs. (Verified; `JOLT_RBRLEN` is a no-op under `-te`, which is why go/no-go 173439307's "frozen" arm was invalid.)

---

## 2. What the literature says (optimisation-literature agent — primary sources)

- **Diagonal LM = damped Jacobi.** Converges linearly at rate `ρ(I−diag(H)⁻¹H)`, →1 as diagonal dominance is lost (Saad 2003; Golub & Van Loan). The `b·r` off-diagonal block is exactly what destroys diagonal dominance, and it grows with K.
- **Additive μ is NOT scale-invariant; Marquardt's `(1+μ)|H_ii|` is** (Marquardt 1963; Transtrum & Sethna 2012, arXiv:1201.5885 — a diagonal reparam leaves the Marquardt iterate sequence unchanged). One additive μ across branch/rate/weight classes (wildly different curvature scales) **cannot damp all three at once** → the ⅓ rejections. Transtrum's recommended form: **Marquardt scaling with a per-coordinate floor**, plus "delayed gratification" μ-updates (JOLT's `μ×4` reject is far more aggressive than the recommended ~×2).
- **Geodesic acceleration** (Transtrum & Sethna, arXiv:1207.4999): one extra directional 2nd-derivative per step, **2–10× (up to 70×) fewer Jacobian evals**, helps *specifically* in narrow-canyon regimes — ours.
- **Two-block alternation beats a joint method when the joint Hessian is ill-conditioned but each block is well-conditioned** — Beck & Tetruashvili 2013 (SIAM J Optim 23:2037) + cyclic-BCD (JMLR 18:17-157): alternating minimisation is *the* case whose convergence time is **independent of the least-smooth block**. This is why the CPU reference tools converge where the joint diagonal LM stalls.
- **Gauge / quotient-manifold** (Absil-Mahony-Sepulchre 2008; Mishra 2014): the principled cure for a scale symmetry is to optimise on the quotient; the **first-order surrogate is projecting the step orthogonal to the gauge direction** — cheap, and predicted to remove the rank-1 singularity (but not the dense coupling; that needs the Marquardt/per-class metric).
- **IQ-TREE's own CPU has this exact wart, acknowledged:** GitHub issue #38 — users hit a hard 99-iter cap on `+R4`; Minh Bui: *"not a bug, rather a heuristic"*; `-nparam 1000` converged after 159 rounds. Capping-instead-of-converging is documented reference behaviour.
- **No prior art batches many *models* or *trial points* into one GPU launch** (BEAGLE batches sites/categories/partitions/chains; Ayres 2019, Gangavarapu 2026). Genuinely open — but the JOLT regime axis `r=m·ncat+c` carries one **shared branch length per edge** (`:1458-1460`), so it multiplexes *rates* cheaply, **not** trial branch vectors.

---

## 3. The Amdahl question — the decider (log-forensics agent + red-team, then Gate A)

**Claim under test:** high-K `+R` (R5–R10) is on ModelFinder's critical path.

**State of evidence — UNANSWERED, honestly:**
- ModelFinder has a **greedy BIC stop**: halts at the first K where BIC worsens (`phylotesting.cpp:3971-3976`), marks all higher `+R` `MF_IGNORED` (`:4091-4102`); `filterRates` prunes rate families >10 BIC off the best (`:3485`). BIC uses **`ssize = getNSite()` = SITES** (`:299/:3842`), penalty **`+2·ln(N)` per category** (df +2/cat, verified from `.iqtree`).
- **The structural-unreachability rescue FAILS (red-team, decisive):** penalty grows like `ln N`, the lnL gain from a genuinely-needed category grows like `N`, so `gain/penalty ∝ N/ln N → ∞`. **At large N, BIC becomes *more* permissive — high-K is NOT structurally unreachable.** AA-100K penalty 23.0/cat; avian TENT (37.35M sites) 34.9/cat — only 1.5× more, while evidence is 373× more.
- **So the only reason high-K never appears in our logs is the DATA:** every completed real selection used a **5000-site CTF subsample** (penalty 17/cat on tiny per-category gains → picks R2/+G4, e.g. avian → GTR+F+I+R2, job 172810407), and every full-data `-m MF` was on **gamma-simulated** data (`complex_data_shared`, `+G4` tautological). The one time R5 was "promoted" was a **now-fixed projected-BIC bug** on simulated AA-1M (part9 §IX.7). **No completed, full-data, non-subsample `-m MF` on real data exists.**

**⇒ GATE A (job 173506053, `gems_mf_ladder_realdata.sh`):** on **real avian TENT** subsampled to N=1e5 and 1e6 (+ a gamma-sim DNA control that must bottom out at K≤4), fit a **fixed-tree `GTR+F+R2..R10` BIC-vs-K ladder** (NOT CTF, NOT greedy — compute the whole curve, find argmin ourselves; a non-converged R5 could otherwise trigger a spurious greedy stop). Recompute BIC from `df` and lnL to confirm `N=#sites`. **Decision:**
- **argmin_K ≥ 6 at any N, or argmin_K climbing with N** ⇒ high-K IS on the real critical path ⇒ the convergence lever is justified. (A subsample *understates* depth, so any argmin≥6 at small N is a lower bound.)
- **argmin_K ≤ 5 flat AND the BIC margin to K=6 grows with N** ⇒ genuinely off-path ⇒ **retire the speed lever**, keep L7 Stage A as insurance. Honest negative, now *actually* supported by real data.

Follow-up if GO: one real-data `-m MF` at N=5e5 to record the literal **wall fraction in K≥5**.

---

## 4. The lever families — ranked, with what's dead and why

| # | family | status | why |
|---|---|---|---|
| — | **L7 Stage A: lift the cap** | ✅ SHIP NOW (insurance) | R5–R10 on GPU, 3/3 bit-id, 15–25× CPU-EM on wall+lnL. Correctness, not speed. Independent of Gate A. |
| — | **EM family (Stage B weight-EM / C SQUAREM / D rate-EM)** | ❌ DEAD | B neutral (173438602); C cancelled; **D NO-GO 2.5× slower + FD FAIL + still 401 iters (173444797)**. Both arms 401 ⇒ disease is coupling structure, not rate math. Do not reopen. |
| **1** | **CONDITIONING FIX on the joint loop** — per-class/Marquardt-multiplicative μ + null-space (gauge) projection of the step + geodesic acceleration | 🟡 **NOVEL, UNTESTED, the recommendation** | Literature's actual prescription for shared-μ + `b·r` coupling. **No new kernel, no CPU fallback** — modifies `:3245-3273` only. Not touched by any closed stage. **Gated by probe 173503842** (if frozen-rate branches don't converge, nothing helps). |
| **2** | Full block-alternation (branch block ⇄ rate block) | 🔴 HARDER THAN IT LOOKS | Frozen-rate **branch** block exists (L5). Frozen-branch **rate** block **does NOT** — `phylotreegpu.cpp:2705-2721` is **weight-only** EM; building the rate block = Stage D (dead). Hybrid GPU-branch + CPU-rate reinstates the 1350–1885 s CPU-EM cost. Only viable if family #1's frozen-branch step can be made GPU-native cheaply. |
| **3** | Batch trial-points / models in one launch | ⛔ NO CLEAN PATH | Regime axis shares one branch length/edge (`:1458-1460`); can't batch trial branch vectors without M× arenas. Process-global `__constant__` eigensystem + `jolt_gpu_mtx` serialise model fits. |

### 4.1 Family #1 in detail (the build, if Gate A = GO and probe 173503842 = converges)
All inside `gpu_jolt_optimize`, default-OFF flag (`JOLT_LM_COND`), byte-identical when unset:
1. **Instrumentation FIRST (its own cheap job, no product change): `JOLT_LM_MAXITER` override + per-iteration lnL/μ/ρ trace.** Compute the LM **gain ratio** `ρ = actual ΔlnL / predicted ΔlnL` (predicted `= Σgᵢδᵢ − ½Σ|ddᵢ|δᵢ²`, all in registers). `ρ≪1` ⇒ diagonal model wrong ⇒ coupling; `ρ≈1` but wandering ⇒ multimodality. Separates *stopping-rule waste* from *bad direction* from *hard surface*, and maps the lnL-vs-iter curve that governs both the speed prize and the truncation/BIC-bias risk. **This is a citable contribution by itself** (issue #38 is "raise the cap and grind"; nobody has instrumented *why*).
2. **Null-space projection:** project the staged step orthogonal to the analytic gauge direction `d=(−b/κ,+r/κ,0)` (and optionally the softmax shift) before the accept. A few lines; removes the rank-1 singularity.
3. **Per-class / Marquardt μ with floor:** replace the single additive μ with `|dd|+μ_class·|dd|+ε` per class (branch/rate/weight), Transtrum's delayed-gratification update. Directly attacks the rejection mechanism.
4. **Geodesic acceleration** (if 2+3 leave a tail): one extra directional 2nd-derivative term.
Each step **byte-identical when off, gated on ΔlnL rel≤1e-6 + the nesting invariant per K + reproducibility ≥ CPU**. Wall metric = GPU high-K wall vs the measured 60–183s baseline (and vs `+G4`'s ~10–15s), NOT vs CPU-EM (already beaten). Human-only push; every number → a job ID.

---

## 5. Corrections banked this pass (honesty ledger)
- **My "both halves of block-alternation already ship" was WRONG** (red-team broke it): the frozen-branch **rate** block does not exist; `:2705-2721` is weight-only EM. Retracted; block-alternation demoted to family #2.
- **My "no `-m MF` on real data ever run" was OVERSTATED:** real avian **CTF** runs exist (172810407 completed → GTR+F+I+R2; eukaryote 171521161 `-m MF` with +I+R8 timed out). Precise claim: no *completed full-data non-CTF* `-m MF` on real data.
- **My R4≈75-iters back-out was WRONG:** R4 = **38 iters** measured (173439307); the "74–141" I matched is a *different* cell (in-tree +R4, 172444201). The affine-in-K wall result stands on the *measured-counter* fit (R4/R6/R8).
- **Doc errors fixed in FULL-GPU-END-TO-END-PLAN.md:** line 493 CPU-EM R8 `−78.425 → −75.410` (A100)/`−75.370` (H200); line 502 R10 "timeout" → H200 CPU-EM R10 **completed 2893.6s**.
- **Still to annotate:** `Wang-Li-Susko-Roger 2008` is the EM machinery IQ-TREE reuses (`ratefree.cpp:540` cites it), **not** the FreeRate *model* primary source (Yang 1995 / Soubrier 2012) — re-scope the citation, don't delete it. `S≈4.8` is **retired** (June microbench, job 170361630; re-runs ≈1.0×), not unsourced — annotate wherever the N/S-ceiling argument leans on it.

---

## 6. Sequencing (gated, cheapest-first — the discipline that killed the EM family cheaply)
1. **Probe 173503842** (already queued): does a FROZEN-rate branch-LM at R8 converge (<<385) or cap (~390)? Gates family #1 (converges ⇒ coupling is the disease ⇒ conditioning fix can work; caps ⇒ branches intrinsically hard ⇒ family #1 dead too).
2. **Gate A 173506053** (already queued): is high-K on the real critical path? Gates *whether to build anything at all*.
3. **Only if BOTH point GO:** build family #1 step-1 (instrumentation + `JOLT_LM_MAXITER` + ρ trace) — cheap, decisive, publishable on its own. Then steps 2→4 as each clears its gate.
4. **Independently, now:** ship L7 Stage A (correctness insurance) once its validation (173504024, esp. G2) is GREEN.

**Confidence:** mechanism ~95% (source+literature converge); "conditioning fix helps" ~50% (gated by 173503842 — genuinely unknown until frozen-rate branch convergence is measured); "high-K is on the real critical path" — the honest coin the project has never flipped (Gate A). **The EM family is closed; the conditioning family is the one live, novel bet, and it is cheaper than the block-alternation the earlier plan assumed.**

---

## 7. ⚠️ COLLECTION-TIME INTERPRETATION GUARDS (red-team of the in-flight jobs, 2026-07-10)
*Full audit of every submitted job. NO fatal bugs (the one that existed — numpy import in the Gate-A subsampler — is fixed via `module load python3/3.11.7`, repo pattern). All binaries verified correct md5 (build-l8 `fe5ce648`, build-grad `56ff1e95`, build-grad-l7 `5c23fe63` @ source HEAD `05d8ab61` kill-switch present; Hashara `e713866b` sm_90-only ⇒ parity correctly H200-locked). BUT several gates PASS/FAIL for the WRONG reason — each job logs enough raw data to apply the REAL check post-hoc. Current job IDs after the A100 move: GateA=**173506691**(A100), l7-altprobe=**173506391**(A100), grad-validate=**173506392**(A100), l7-validate=**173504024**(H200), fr-profile=**173506349**(H200), mfp-smoke=**173500684**(H200)→parity 173500685-688.*

**🔴 GUARD-1 — Gate A (173506691): non-convergence CONTAMINATES the argmin.** R5–R10 hit the 401 cap (control already showed R7/R10 nesting violations 0.043/0.095 nat). A non-converged high-K fit has UNDER-estimated lnL ⇒ OVER-estimated BIC ⇒ biased **against** high-K. So "argmin stays at R4" is ambiguous: data-doesn't-need-high-K **or** high-K-would-win-but-didn't-converge. **CHECK AT COLLECTION:** a "retire" verdict is trustworthy ONLY if high-K is flat **AND** shows NO nesting violations. If high-K is close to winning **AND** violates nesting ⇒ **INCONCLUSIVE (needs a converged high-K fit), NOT retire.** The harness prints per-K nesting violations — read them before concluding.

**🔴 GUARD-2 — l7-altprobe (173506391): NO hard engagement gate (inherits the go/no-go 173439307 flaw).** The probe buckets `iters` but never ASSERTS `freeRate==3` engaged. It uses `-ninit 20` (a real search) to make it engage (unlike the invalid `-te` go/no-go), but if it doesn't, the probe silently shows all-401 and reads as "NO-GO." **CHECK AT COLLECTION:** confirm a DISTINCT `freeRate==3` population exists — a cluster capped at ~390 (`JOLT_BRLEN_MAXITER=390`) or converged <385, SEPARATE from the ~401 model-fit population. If there is no separate population ⇒ the frozen path never engaged ⇒ result is **VACUOUS, not NO-GO.**

**🔴 GUARD-3 — l7-validate G2 (173504024): the LOAD-BEARING gate only checks the WINNER NAME, not the lnL spread.** G2 decides default-ON vs opt-in. As CODED, `G2b` only checks BIC-winner stability across seeds — but on gamma-sim data with `-cmin 5`, R5 always wins by penalty (lnL flat) ⇒ G2 passes TRIVIALLY even if high-K lnL is NOT reproducible. The "spread < 0.5 nat" check is in the design comment but NOT the code. **CHECK AT COLLECTION:** extract R8's (and R6's) per-seed lnL from the `g2_*.iqtree` model tables (not just the winner's `BEST SCORE`) and compute the real cross-seed spread. Winner-stability alone does NOT justify default-ON.

**🟠 GUARD-4 — l7-validate timeout budget.** Internal timeouts sum past 10h if caps hit (G2 6×2h + G3 CPU 4×40m + G4 3×90m). Realistic << that, but if `-m MF` is slow the job may be KILLED before G4 ⇒ partial results. If it dies mid-run, the G1/G2/G3 that completed are still valid; only re-run the missing gate.

**🟠 GUARD-5 — l7-validate G4 RF=0 may FALSE-FAIL on search stochasticity.** G4 requires `RF(GPU-brlen, CPU-brlen)=0`; tiny GPU-vs-CPU brlen deltas amplified through NNI can flip a topology for reasons unrelated to correctness. Same-seed mitigates. **CHECK:** if G4 fails ONLY on `RF(default,killswitch)!=0` while lnL matches ≤1e-3 and engagement is clean, treat as search-stochastic (re-run with a fixed `-te` brlen-only check), not a brlen-correctness bug.

**🟠 GUARD-6 — Gate A subsample understates depth + single dataset.** A 100K/1M subsample of 37.35M sites has less power to detect high-K than full data ⇒ "flat at 1M" is a LOWER BOUND, not conclusive for 37M; and it's ONE alignment (avian). A CLIMBING result (argmin rising with N) is strong; a FLAT result is suggestive, not final — if flat-but-close, the honest verdict is "no evidence high-K is on the path at ≤1M," not "retire forever."

**🟡 GUARD-7 — mfp-parity is a SYSTEM comparison, report it as such.** JOLT runs `--ctf` (subsample model selection) + `maxiter=2` brlen; Hashara runs full-data MFP. JOLT wins partly BECAUSE of CTF (which Hashara could also adopt) — the honest headline is "JOLT+CTF **system** faster," not "JOLT kernel faster." The parity tracks winner/lnL/RF, so VERIFY JOLT's tree is as good as Hashara's (RF≈0, lnL≥) before quoting any speed number — a faster-but-worse tree is not a win.

**🟡 GUARD-8 — lnL-string-identity ≠ true byte-identity.** G1 (l7-validate) and grad-validate check `lnL string == + RF=0`, not `memcmp` of partials. Strong proxy; a sub-printed-precision drift would slip through. Adequate for the byte-identical L5/L6 and the killswitch claims, but do NOT upgrade the wording to "bitwise proven" on this evidence alone. Also: grad-validate on A100 proves **A100** byte-identity — H200 is near-certain to match but strictly extrapolated.
