# A parameter-free, self-scaling curvature for the JOLT +I+R optimizer — OPG empirical-Fisher

**Status:** DESIGN RECORDED, BUILD DEFERRED (2026-07-15). The ④ 0.07-nat AA +I+R gap is **ACCEPTED as selection-irrelevant**
(user decision) — proven by the real-data measurement below — so this optimizer redesign is **not built**. The doc is kept as
a grounded plan for IF the gap ever proves selection-relevant. curvFloor was REVERTED (optimizer byte-identical to ②a).
Author-run; assistant does NOT push GPU source.

## VERDICT (zero-job real-data measurement, 2026-07-15): the gap does NOT flip real-data selection ⇒ accept it, don't build
Avian (Jarvis TENT, real DNA, `-m MF`, ctfavab 173832078 oracle tables):
- 100k: winner GTR+F+I+R4 BIC 2689238.750 vs runner-up GTR+F+R4 2689241.425 ⇒ **margin 2.675 BIC**.
- 500k: winner GTR+F+I+R4 BIC 13397004.997 vs TVM+F+I+R4 13398964.221 ⇒ **margin 1959 BIC**.
The 0.07-nat gap = 0.14 BIC ⇒ selection margin is **19×–14000×** the gap. AND the avian winner is **DNA** GTR+F+I+R4, which
JOLT scores at the **EXACT CPU MLE (zero gap** — the 0.07 gap is AA-only; DNA +I+R is byte-exact, hardening gate 173910972 H0).
So the real-data winner has no gap, and even a hypothetical AA-sized gap is dwarfed by the margin. **④ CLOSED: accept the gap.**
The OPG plan below stands unbuilt; revisit only if a real AA dataset ever shows a &lt;~0.5-BIC +I+R selection tie.

---

**Original design (retained for the record):**

**Motivation:** the ④ AA +I+R convergence gap cannot be fixed by ANY hardcoded floor constant (flat `κ·rN` regresses
high-ncat +R by up to 2.6 nats — hardening gate 173910972; weight-aware `κ·w_c·rN` still needs a swept κ = still a magic
number that won't generalize across datasets/models). The fix must be **self-scaling and parameter-free**. It is.

---

## 0. Why every hardcoded floor is doomed (formalizing the user's objection)

The curvFloor tries to hand-approximate the per-parameter Fisher information: `κ·rN` ≈ N (flat) or `κ·w_c·rN` ≈ w_c·N
(weight-aware). But the TRUE curvature of each model arm varies with **data, model, ncat, and rate-degeneracy** — there
is no constant that is right for LG+I+R4 AND LG+I+R10 AND a partition AND real avian data. Proven empirically:
- flat κ=1 fixes R4 (−7541972.260) but REGRESSES R8 (−0.41) / R10 (−2.6) — over-damps low-weight categories (173910972).
- weight-aware still needs a κ sweep {1,2,4}, and its knee moves with the weight skew.

**The scaling has to come from the DATA, not a constant.** That is exactly what a proper curvature matrix provides.

## 1. Root cause (grounded, gpu_lnl_intree.cu)

JOLT's model arms (rate ddY, weight ddZ, pinv ddP, free-Q ddQ) use **noisy diagonal FINITE-DIFFERENCE secants**
(`ddY[c]=(g_y[c]-rgyPrev[c])/(baseY[c]-ryPrev[c])`, :3282) with a **single SHARED damping mu** (:3169). A noisy/small/
wrong-signed secant → the Newton-ish step `g/(|dd|+mu)` overshoots → reject → `mu*=4` ratchets 1e3→1e9 (CONVTRACE
173907382) → every arm's step vanishes → premature `dl<tol` exit 0.07 below the MLE. The curvFloor is a band-aid on the
SYMPTOM (noisy secants); it cannot cure the DISEASE (a bad curvature estimate + a shared, monotone-inflating damping).

## 2. The fix (novel direction): OPG empirical-Fisher curvature + Marquardt diag-scaling + Nielsen gain-ratio

Three classical, parameter-free ingredients — **none is JOLT's current design; all three were specified for JOLT's
predecessor (Mode-L, Trimorph.md §120/§358) but JOLT substituted FD secants instead.**

**(a) Curvature = OPG / BHHH empirical Fisher.** Replace the per-arm FD secant with the outer product of per-pattern
scores over the model-arm block θ=(rates, weights, pinv, Q):
```
H = Σ_p ptn_freq[p] · s_p · s_pᵀ ,   s_p = ∂ log L_p / ∂θ   (per-pattern score vector)
```
- **Always PSD** (a sum of outer products) ⇒ no negative curvature, no `|·|`, no floor, no sign trap.
- **Self-scaling by construction**: `diag(H)` for rate-c is automatically `~w_c·N` (the per-category Fisher info) —
  the *exact* scaling the weight-aware floor hand-approximated, now READ FROM THE DATA, parameter-free.
- **Correct at the optimum** (information-matrix equality; Berndt-Hall-Hall-Hausman 1974).
- **Off-diagonal terms** capture the rate↔weight↔pinv coupling the diagonal secants ignore — this is what actually
  cures the +I+R zigzag (the arms stop fighting each other), not just the floor's per-arm damping.

**(b) Damping = Marquardt's `λ·diag(H)`** (NOT `λ·I`; Marquardt 1963). Scaling the damping by each parameter's own
curvature is *inherently* weight-aware — it is why no κ is needed. Step: `Δθ = (H + λ·diag(H))⁻¹ g` (a ~10–25-dim solve).

**(c) `λ` adapted by the Nielsen gain ratio** `ρ = (actual ΔlnL)/(predicted ΔlnL)` (Nielsen 1999, Madsen-Nielsen-
Tingleff). ρ good ⇒ shrink λ (accept, trust the model); ρ poor ⇒ grow λ (toward gradient-descent, safe). This REPLACES
the ×4-up/×0.5-down mu ratchet (the thing that inflates to 1e9) with a self-correcting trust region — **parameter-free**,
and it is exactly what makes BHHH robust despite its known "poor far from the optimum" caveat (search: BHHH is a good
Hessian only near the MLE; the gain-ratio λ handles the far regime by falling back to gradient descent).

**Brlen stays on its exact curvature** (`g_ddf`, already correct) — OPG replaces ONLY the fragile model-arm block. This
also respects the Mode-L lesson (the LM lost on low-dim +G; scope OPG to the high-dim +I+R model arms where it wins).

## 3. Why it is GPU-parallel and novel for ModelFinder

- **The per-pattern scores are ALREADY on-device.** `kj_ratenum` → `rnum[c][p]`; `accR[c]`/`accW[c]` are per-pattern
  sums (:3104/:3111); the comment at :2681 states "*every JOLT quantity … is a SUM OVER PATTERNS*." Building `H` reuses
  those exact partials — accumulate their **outer products** (Σ_p ptn_freq·s_p·s_pᵀ) instead of their plain sums. That is
  a small (~10–25²) batched reduction over patterns = a GPU-friendly parallel accumulation (one warp-reduced GEMM-let),
  reusing the existing `kj_reduce*` deterministic-reduction pattern. **No new likelihood machinery.**
- **The GPU win is FEWER SEQUENTIAL ITERATIONS.** ModelFinder is iteration-sequential-bound (bfgs&CrossModelWarmStart.md
  §521: "cannot parallelise across iterations"). A well-conditioned OPG-Newton step converges in far fewer iterations
  than the damped-gradient LM's reject-laden path (reject_frac 0.41→? , iters 22/401→?). Trading a cheap parallel
  per-iteration OPG-reduction for many fewer sequential iterations is *the* right GPU trade — and it attacks the one
  thing the walltime study (mflstorm 173912384) showed is NOT memory-bound: the sequential optimizer iteration count.
- **Novelty:** no phylogenetic tool does GPU-parallel OPG empirical-Fisher *joint* model-parameter optimization inside
  ModelFinder. BEAGLE parallelizes the per-model likelihood, not a joint OPG-Newton solve; CPU Fisher-scoring exists only
  for branch lengths (Ji et al. 2020, "Gradients do grow on trees", O(N) pre/post-order gradient). This pairs the O(N)
  phylogenetic gradient with a GPU-reduced empirical-Fisher curvature over the rate-het block — a genuinely new combination.

## 4. What our docs already record as attempted (do not re-tread)

- **Trimorph Mode-L (JOLT's predecessor) SPECIFIED this** (§120: "LM joint solve … analytic gradient (Ji et al.) and an
  OPG/BHHH empirical-Fisher curvature"; §358: "fall back to Fisher-scoring … if ill-conditioned [near-degenerate rate
  eigenvalues]"). It was NOT disproven — it was abandoned for (i) a BROKEN +R weight analytic gradient (10⁵⁴, §17.20) and
  (ii) losing on LOW-dim `-m TEST` (+G, +34% traversals) where the LM is overkill. **The decisive HIGH-dim +R test was
  never run** (Rec 2: "re-run the +R traversal gate — that is the decisive test of the LM premise"). ④ is that regime.
- **JOLT already de-risks the (i) blocker:** `JOLT_RGRADCHECK` (:2869) FD-checks the +R gradient (pass `maxrel<1e-4`).
  OPG is only as good as the gradient it squares, so this is the mandatory Phase-0 gate-0.
- **EM rate M-step** was tried (JOLT_REM_EN) and a full EM redesign was a NO-GO (2.5× slower, tail = branch-LM not rate
  arm; memory project_gpu_freerate_handicap). OPG is NOT EM — it keeps the joint Newton step, only fixes the curvature.
- **Model-arm optimization is 8–30% of per-model wall** (branch is 75–85%; Trimorph AA-1M decomp). ⇒ OPG is a
  CORRECTNESS/convergence fix (reach the +I+R MLE, parameter-free, no floor) with BOUNDED walltime upside — not a headline
  speed lever. Honest scope. (It still matters: the avian winner is GTR+F+I+R4, and +I+R quality gates GPU selection.)

## 5. Honest risks

- **BHHH is a poor Hessian far from the optimum** (literature). Mitigation = the gain-ratio λ trust region (falls back to
  gradient descent when ρ is bad). Must gate on warm-seeded starts (JOLT's real regime) AND a cold start.
- **Near-degenerate rate categories** (R8/R10) ⇒ `H` can be near-singular. Mitigation = `λ·diag(H)` regularization (never
  singular) + optionally the Fisher-scoring/Kenney-Gu fallback Trimorph §358 named. This is the SAME regime the flat floor
  failed on — the test is whether OPG's data-driven scaling succeeds where the constant failed.
- **OPG reduction cost**: ~10–25² doubles/pattern accumulation. Must confirm it does not dominate the per-iteration wall
  (should be ≪ the O(nptn·ns²) likelihood; measure).
- **Off-diagonal solve**: a ~10–25-dim dense `(H+λ diag)⁻¹g` per LM iteration, on-device or host. Tiny, but must be
  deterministic (bit-reproducible) to preserve the project's reproducibility posture.

## 6. Phase-0 de-risk (cheapest-kill-first; PARAMETER-FREE — no κ anywhere)

0. **Gradient soundness (prerequisite):** run `JOLT_RGRADCHECK` on AA LG+I+R4/R8/R10 + DNA. Gate: `maxrel<1e-4`. If the
   +R/pinv/Q gradients aren't clean, OPG is built on sand — fix the gradient first (or STOP). Cheap, existing path.
1. **Diagonal-OPG spike (no off-diagonal yet):** replace `cfloor(ddY[c])`/`ddZ`/`ddP` with the OPG DIAGONAL
   `H_cc = Σ_p ptn_freq·s_{p,c}²` (reuse accR/accW per-pattern partials; NO κ, NO floor), damped by `λ·H_cc` with a fixed
   λ. Gate: does it fix R4 AND hold R8/R10/WAG/JTT AND stay DNA byte-id, with ONE code path and ZERO tuned constants? This
   is the decisive parameter-free test — if the diagonal empirical-Fisher alone reconciles R4↔R10, we are done without the
   dense solve.
2. **Nielsen gain-ratio λ:** replace the mu ladder with `ρ`-adaptive λ. Gate: mu no longer inflates (CONVTRACE), iters
   drop, reject_frac drops, R4–R10 all reach MLE.
3. **Full block-OPG (off-diagonal)** only if the diagonal leaves residual coupling: build the dense `H` + solve. Gate:
   +I+R reaches MLE on R4–R10 + real avian GTR+F+I+R4, DNA/AA byte-id when the block is inactive, iteration-count win.
4. **GO/NO-GO:** GO = one parameter-free path fixes R4↔R10 uniformly + no low-dim (+G) regression + reproducible.
   NO-GO = OPG can't beat the constant floor parameter-free ⇒ retire, keep curvFloor DEFAULT-OFF, document the 0.07-nat
   AA gap as an accepted (selection-irrelevant) limitation.

## 7. Immediate ④ disposition

The flat/weight-aware curvFloor is a **band-aid that regresses high-ncat +R** ⇒ **DO NOT graduate it. Keep DEFAULT-OFF**
(env-only for A/B). The 0.07-nat AA +I+R gap stays a documented, selection-irrelevant known-issue until the OPG fix lands.
②a (the +I+R WRITEBACK fix) is unaffected and stays graduated. This is the honest posture: ship nothing that regresses.

**Literature:** Berndt-Hall-Hall-Hausman 1974 (OPG); Marquardt 1963 (`λ·diag(H)`); Nielsen 1999 / Madsen-Nielsen-Tingleff
(gain-ratio LM); Ji et al. 2020 MBE "Gradients do grow on trees" (O(N) phylogenetic gradient); Amari (natural gradient).
