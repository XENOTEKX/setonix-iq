# +R FreeRate on GPU ModelFinder — the plateau, the wobble, and the instrument we never had

**Status: PLAN. Nothing built, nothing gated, nothing promoted. 2026-07-21.**
Produced by a research → red-team → blue-team cycle. **Most of this plan's first draft was destroyed by
its own red-team; the surviving evidence is marked and the dead evidence is retained as §2 so it is not
re-derived.** Every load-bearing number was re-verified by me on disk, including the numbers the
red-team and blue-team supplied.

---

## 0. THE ONE-PARAGRAPH VERSION

We have spent four workstreams (dense-OPG, HARP, Phase-A EM, tolerance/cap sweeps) trying to make `+R`
converge, and every one was judged by a stopping flag that **cannot detect convergence**. The GPU stop
test is `dl < tol` on a single accepted step, but the step is `g/(|dd|+mu)` and `mu` moves by ×4/÷2 —
so a small step means *"near the optimum"* **or** *"heavily damped"*, and nothing in the code
distinguishes them. The cap flag is no better: four simulated cells hit the 400-iteration cap while
gaining **0.001 nats over their last ten iterations** — done, but reported as failures — while avian
`GTR+R6` hits the same cap still gaining **8.478 nats per ten iterations**. **The plateau is real, but we
have been measuring it with an instrument that reports the damping parameter.** The plan is therefore
not another step policy. It is: build the instrument (predicted remaining gain, which is free), use it to
find out where `+R` fits actually stop, then fix the stop rule — and gate that fix as a *selector*,
because a candidate's stopping point propagates into which other candidates are ever fitted.

---

## 1. WHAT IS ACTUALLY WRONG — diagnosis, in the order the evidence supports

### 1.1 ✅ `+R` fits on real degenerate data stop far from stationarity *(survives)*
avian `GTR+R6` at the 400-iteration cap is still gaining **8.478 nats per 10 accepted iterations**
(re-measured by me from `gems-verify/opgp2_174158208/on_av_r6.console`). Where a gradient was recorded
it terminates at `maxgy = 212.2` — with **healthy damping**, `lamIn=0.467`, *not* the documented
Phase-2 lam-ceiling freeze. Simulated `LG+R4` terminates at `maxgy = 0.219`.

### 1.2 ✅ At tight tolerance the fit is deterministically truncated, not converged *(survives, estimator-free)*
The `a1_*` ladder gives **byte-identical `−11,216,886.230053` at `it=400 conv=0` across all six
tolerances 1e-2 … 1e-7**. The shipped default's reproducibility is *deterministic truncation at an
arbitrary height*, which is why it looks stable and is still wrong.

### 1.3 ✅ At loose tolerance the stop is a lottery *(survives, n=1 on the shipping path)*
`off_av8`'s last 12 accepted steps: `0.26 0.25 0.51 0.024 0.42 0.51 0.80 0.14 0.52 0.11 0.23` then
**0.00314** — a stationary noisy plateau with no downward trend, ended by one outlier dip tripping
`tol=1e-2`. Read off the *shape*, not extrapolated. ⚠️ **7 of the 8 avian `conv=1` runs carry
`[HARPCFG]`** (the closed WS1.5 algorithm), so this is **n=1** for the optimiser that ships.

### 1.4 ⭐⭐ THE ROOT PROBLEM — **there is no instrument**
Two independent blind spots, both verified:

| signal | why it fails | evidence |
|---|---|---|
| `dl < tol` | `dl` scales with `1/mu`; `mu` swings ×4 on reject, ÷2 on accept | `gpu_lnl_intree.cu:`g_y[c]/(fabs(ddY[c])+mu)``, `gpu_lnl_intree.cu:`lnL=ln; mu=fmax(mu*0.5,1e-9); acc=true;`` |
| cap exhaustion (`conv=0`) | conflates *"still climbing"* with *"crawling below tolerance"* | 4 sim cells cap at **0.001–0.016** nats/10-iter; avian caps at **8.478** |

And the gradient — the one damping-free observable — **is not emitted on the shipping path at all**:
`maxgy` appears in `tolladder` **0/48** consoles, `harpws15` **0/32**. It exists only in a throwaway
instrumented binary (`gems-verify/opgp2bin/iqtree3`) whose **source was never committed** —
`grep maxgy` over all 37 worktrees = 0 files, `git log --all -S maxgy` = 0 commits.

> **This is the finding that reframes the project.** Four workstreams ranked `+R` step policies by
> terminal lnL at a stop that cannot tell convergence from damping. Fixing the instrument is not
> "Phase 2 of a plan" — it is the precondition for any `+R` experiment to be interpretable.

### 1.5 Why it matters for selection — the shortfall dwarfs the margin
avian R6/R7/R8 are ordered by BIC margins of **~11–14 nats** (top-2 was 82 BIC), while fits terminate
with hundreds of nats still on the table. **The quantity that decides the answer is smaller than the
error in each input to that decision.** That is the wobble mechanism, stated without any extrapolated
estimator.

---

## 2. THE GRAVEYARD — do not re-open (with the reason each died)

| direction | verdict | the reason that actually killed it |
|---|---|---|
| Dense OPG / empirical-Fisher solver | DEAD, 3 repair routes | floor non-monotone ⇒ untunable; **shrinking a step does not bound a wrong direction, it licenses it** (avian accepted 1/6 dense steps at floor 1e-1 but 206/214 at 1e0, at the worst loss of the ladder) |
| HARP hard rank projection | CLOSED by pre-registered rule | τ=1e-3 gains +88.7 on avian R6, loses −705.6 on avian R8 — **oscillating knob ⇒ untunable, not under-tuned** |
| Phase-A EM weight M-step | REFUTED | revived an *exact convex* block solve and **widened** spread 216→535 nats. **Convexity of a subproblem confers nothing on the alternating scheme** |
| Hashara's BFGS (external) | REFUTED | 3 seeds → 3 winners (R5/R6/R8) on avian ⇒ instability is the **data/tolerance**, not any one optimiser |
| Truncated-Newton / CG *(new, killed this cycle)* | DEAD before build | ① the only theorem licensing tCG at singular minima needs Polyak–Łojasiewicz, and PŁ ⟺ **quadratic** growth, but over-specified mixtures have **fourth-order** growth (this is *why* the rate is `n^{-1/4}`); ② where it does hold the constant is `1/λ_min = 1.3e8`; ③ the CG forcing tolerance maps ~1:1 onto HARP's τ — **η=1e-4 reproduces the step that lost 23,414 nats**; ④ the quadratic model is measured **850×** wrong (42,320 nats predicted vs ~49.7 available) ⇒ radius collapse ⇒ returns the Cauchy point ≈ the step we already ship |
| SQUAREM / DAAREM extrapolation *(considered, killed this cycle)* | DEAD before build | the claimed escape was "monotone guarding" — but Phase A **already** ran inside a 14-step backtracking safeguard (`gpu_lnl_intree.cu:`for(int bt=0;bt<14;bt++){``, accept test `ln > lnL+1e-9`) and still widened the spread. **The guard was never the differentiator.** |
| Loose / relative tolerance | DEAD, measured | avian accepted `dl` never drops below 0.0216 over 400 iters and bounces non-monotonically ⇒ any reltol loose enough to stop avian wrecks clean fits |
| λ_min or NPMLE as a *k*-selector | DEAD | λ_min anti-correlates with BIC on avian; NPMLE support grows `O(log n)` |

**The pattern across all of them:** every failure was a **single scalar knob with a non-monotone,
dataset-specific response**. That is now a project law, and it is why §3 changes *no* step policy.

---

## 3. THE PLAN

Sequenced so that the cheap measurement that can kill the plan comes **first**, and so that no build
bundles a bit-identical change with a number-changing one.

### BUILD 1 — THE INSTRUMENT *(edit only; everything default-OFF or no-op-when-unset)*

**(a) Predicted remaining gain — the diagnostic that replaces `dl`.** Emit the diagonal Newton decrement
inside the existing `JOLT_IR_CONVTRACE` guard:

```
dec = ½ ( Σ_e g_df[e]²/|g_ddf[e]| + Σ_c g_y[c]²/|ddY[c]| + Σ_c g_z[c]²/|ddZ[c]| )
```

It is **denominated in nats** (directly comparable to BIC margins), **damping-free** (no `mu`), and
**free**: `g_y`, `g_z`, `ddY`, `ddZ`, `g_df`, `g_ddf` are all already host-resident at the same scope as
the emit — `gpu_lnl_intree.cu:`g_y[c]=catRate[c]*gradR[c];``. No kernel, no sync, no D2H.
🔴 **Guard `it>=2`, not `haveSec`** — `haveSec` is set before the accept block in iteration 1 while
`ddY` still holds its `-1e6` sentinel, which would report a spuriously tiny decrement.

**(b) Gradient sup-norms** `maxgy`/`maxgz` in the same emit (free `ncat` host reductions).
⚠️ **Both are one step stale** — `computeGradient` runs at the *base* point and the emit fires after the
accept, so on `exit=accept`/`CONV` these describe the second-to-last iterate. Exact only on
`REJECT-EXIT`. A truthful *terminal* value needs one post-loop `computeGradient`, behind its **own** env
(it costs a real eval and must not ride the CONVTRACE flag).

**(c) MF-path cap override.** `JOLT_MF_BRLEN_MAXITER` at the one caller that inherits the default —
verified still the only one: `model/modelfactory.cpp:`double jolt_lh = tree->optimizeParametersJOLT(fixed_len);``
against `tree/phylotree.h:`double optimizeParametersJOLT(int fixed_len, bool brlenOnly = false, bool leanTail = false, int brlenMaxIter = 400);``.
Unset ⇒ 400 ⇒ byte-identical.

**Gate:** bit-identical lnL on DNA + AA + avian + **one partitioned cell**; `strings | grep maxgy`
present in the new binary and **absent** in canonical.

---

### ✅ BUILD 1 — DONE AND GATED (job `174362419`, 2026-07-22, 5.08 SU)

Built in worktree `iqtree3-irdec`, branch `ir-decrement`, commit `d0a7170e` (+87/−2, two files).
Binaries: instrument **`3cdee521`**, purpose-built pristine same-commit control **`80a52f91`**.

| gate | result |
|---|---|
| G0 preflight (both binaries start) | ✅ 0 unresolved libs |
| G1 proof-of-build | ✅ 4 sentinels in NEW, **0 in CTRL**; shared `JOLT_IR_*` in both ⇒ control is a real JOLT build |
| G2 bit-identity, all new envs unset | ✅ **DNA, AA, avian, AND partitioned all byte-identical** |
| G2b JOLT actually engaged | ✅ DNA/AA/avian engaged; partitioned declines (expected, no-regression only) |
| G3 instrument emits | ✅ 400/400 `[IRCONV]` lines carry `dec`/`nvd`/`nsd`; `nsd=17` at `it=1` exactly as predicted |
| G4 `[IRTERM]` terminal gradient | ✅ emitted, and **lnL unchanged** ⇒ genuinely read-only |
| G5 cap override, measured | ✅ **400 → 50** iterations |

⚠️ G5 initially reported FAIL — a **parser bug in the gate script**, not the code: `grep -oE 'it=[0-9]+'`
also matched the `rej_it=0` field on the same line, so `tail -1` returned 0 for both arms. Same class as
the previously-recorded RF parser that matched `29` inside a storage path. Fixed to anchor on the
`[IRCONV]` prefix. *(Attempt 1, job `174361261`, died in 3 s: the script loaded no runtime modules, so
`libcudart.so.12` was unresolvable. Build-time and run-time environment are separate problems. A G0
preflight and a GPU-engagement assertion were added so neither can recur silently.)*

### 🔴 FIRST MEASURED RESULT — AND IT IS A WARNING ABOUT THE DECREMENT ITSELF

The gate's avian trace (`GTR+F+R6`, fixed tree, `tol=1e-7`, cap 400) is the first `dec` data ever taken.
**It is correctly implemented but it is NOT trustworthy as a single-iteration stop criterion.**

| it | lnL | `dec` (predicted remaining) | actual gain over next 10 it |
|--:|--:|--:|--:|
| 1 | −11,562,820.597 | **2.589e+05** | 335,678 |
| 50 | −11,217,209.193 | 1.994e+01 | 48.66 |
| 200 | −11,216,985.160 | 4.034e+01 | 12.35 |
| 300 | −11,216,861.120 | 2.932e+01 | 9.35 |
| **350** | −11,216,814.841 | **1.898e−01** | 5.00 |
| 390 | −11,216,792.745 | 2.120e+00 | 5.56 |
| 400 | −11,216,787.185 | **7.430e−01** | — |

**① Correctly implemented.** At `it=1`, `dec` = 258,860 nats vs 346,033 actually gained over the run —
right order of magnitude. The quantity is real.

**② But it UNDER-PREDICTS on the degenerate cell — by 7.5×.** At `it=400` it claims **0.743 nats**
remain while the run is still harvesting **5.56 nats per 10 iterations** and is (§1.1) hundreds short.
*Mechanistic, not a bug:* `dec` is built from the same **diagonal** secants `ddY`/`ddZ` the optimiser
uses, and on a degenerate ridge the remaining gain lives in the **off-diagonal** directions a diagonal
model cannot see. It is blind in exactly the direction that matters.

**③ 🔴 AND IT OSCILLATES — it is a lottery on the same shape as `dl<tol`.** Over the last 50 iterations
`dec` runs **0.19 → 2.12 → 0.74**, an 11× swing. A single-iteration threshold anywhere near 0.5 would
have declared avian "converged" at `it=350` with hundreds of nats still on the table — **reproducing
precisely the transient-dip failure this whole plan exists to remove, on a new scalar.**

⇒ **CONSEQUENCE FOR PHASE 2, adopted now:** no stop predicate may read `dec` at a single iteration. It
must be **sustained over N consecutive accepted steps** (start N=5, covering the observed swing), and the
AND-gate must retain `params_settled`. This is a design change forced by measurement before any tuning.
⇒ **CONSEQUENCE FOR PHASE 1:** the decisive question is now sharper — not "is `dec` calibrated" (it is
not, on the ridge) but **"does sustained `dec` still SEPARATE the regimes?"** A separator does not need
to be a calibrated absolute estimate. n=1 so far; the simulated negative controls are what settle it.

### 🔴🔴 PHASE 1 RESULT — **`dec` FAILED ITS PRE-REGISTERED TEST. THE KILL FIRES.** (job `174416929`, 12.85 SU)

Not a marginal overlap — **an inversion.** The cells that are DONE report a *larger* decrement than the
cells that are still climbing:

| cell | group | last-10 gain | **sustained `dec`** | `termgy` |
|---|---|--:|--:|--:|
| `av_r6` | **still climbing** | 5.5598 | **6.93** | 211.9 |
| `av_r8` | **still climbing** | 3.5496 | **1.88** | 217.5 |
| `sd_r8` | converged | 0.0011 | **112.2** | 2.505 |
| `sd_r10` | converged | 0.0102 | **112.1** | 7.055 |
| `sa_r8` | converged | 0.0082 | 6.01 | 3.829 |
| `sa_r10` | converged | 0.0002 | 5.94 | 1.428 |
| `sd_r4` | converged @48 it | 0.0018 | **111.3** | 0.587 |
| `sa_r4` | converged @44 it, `conv=1` | 0.0000 | 5.96 | 0.093 |

Lowest sustained `dec` among climbing cells **1.65e−02** vs highest among converged **1.12e+02**.
**No threshold exists. FAIL.**

**MECHANISM — verified, and it is my design error, not a data property.** `dec` is
**branch-dominated and k-independent**: every DNA cell reads ~**1.11–1.12e2** whether *k*=4, 8 or 10,
every AA cell ~**5.94–5.96**. A six-fold change in the number of rate categories moves it by <1%.
With ~197 edges against 2*k*≈8–20 model channels, and `g²/|dd|` inflating without bound as a branch
curvature approaches zero, the sum measures **branch conditioning, not `+R` convergence** — it is blind
to the arm the entire pathology lives in. My `|dd| < 1e-12` skip guard was far too permissive.
🔴 **AND I DESTROYED THE ATTRIBUTION MYSELF** by summing branch + rate + weight into one scalar. Had the
arms been emitted separately (`decb` / `decm`), this run would have answered both questions at once.
**Any future decrement must be emitted per-arm.**

⇒ **`dec` as specified is NOT an instrument. Phase 2 must NOT be built on it.** Recorded as designed.

### ✅ BUT THE PRE-REGISTERED *SECONDARY* QUESTION RETURNED A REAL RESULT — reachability, measured at last

avian `GTR+F+R6`, identical start, cap 400 vs cap 3200:

| cap | terminal lnL | last-10 gain | `termgy` |
|--:|--:|--:|--:|
| 400 | −11,216,787.185 | 5.5598 | 211.9 |
| **3200** | **−11,216,097.088** | **0.1832** | 44.3 |

**8× headroom bought 690.10 further nats and cut the residual rate ~30×.** ⇒ avian `+R` is **not a hard
plateau**; it is a slow crawl that substantially resolves given budget. This is the first clean evidence
on reachability after both prior "existence proofs" died (§7), and it stands on its own because it was
pre-registered as the secondary question.
🔴 **It also confirms, by direct measurement rather than extrapolation, that "just raise the cap" is NOT
a clean fix:** 690 nats is **~50× the 11–14-nat BIC margins** that order R6/R7/R8. Raising the cap moves
each candidate by far more than the differences that rank them, so it can reorder selection
unpredictably — exactly the concern already on record, now quantified.

### 🟡 POST-HOC OBSERVATION — `termgy` separates where `dec` does not. **This is a NEW hypothesis, not a rescue.**

Climbing 211.9 / 217.5 vs converged 0.093 – 7.06 ⇒ **~30× margin**; and the cap-3200 arm sits *between*
at 44.3, which is the correct ordering for a cell that is genuinely part-converged. This is what the
red-team argued from the start ("the gradient does what you wanted `dl` to do") and what I under-weighted
in favour of a nats-denominated quantity.
**It must not be claimed as a pass.** It was generated *from* the data collected to test a different
hypothesis; with the cap-3200 arm included the margin is 6.3×, **below the 10× bar**. Promoting it now
would be precisely the post-hoc fitting this project has buried six times. It needs **its own
pre-registered test on fresh cells** (a real-AA high-k cell and a second real DNA dataset at minimum),
or it does not ship.

### PHASE 1 — DESIGN AS RUN *(~1.5–2 GPU-h; this phase could honestly end the plan — and for `dec`, it did)*

🔴 **avian is PRIMARY. The simulated cells are the NEGATIVE control.** This reverses an earlier
recommendation and it is the single most important sequencing decision here:

| cell | last-10-iter gain | correct answer for a stop rule |
|---|--:|---|
| `on_dna_r8` / `on_dna_r10` | **0.001** / **0.001** | **stop early — it is done** |
| `on_aa_r10` / `on_aa_r8` | **0.002** / **0.016** | **stop early — it is done** |
| **`on_av_r6`** | **8.478** | **keep going — it is hundreds of nats short** |

Tuning a stop rule on cells whose right answer is *"you're done"* and shipping it to a cell whose right
answer is *"keep going"* is inverted selection pressure. A correct rule must **stop the sim cells early
AND keep avian running.** Both directions are gate criteria.

**The one decisive question:** *does the decrement separate 8.478-nats-still-climbing from
0.001-nats-done?* Secondary: does raising the cap drive the decrement toward zero on avian, or does it
plateau? — that is the honest replacement for the two "is the value reachable" existence proofs that
died in §2 of this cycle.

> **PRE-REGISTERED KILL: if the decrement does not separate those regimes, STOP.** There is then no
> instrument, Phase 2 has nothing to stand on, and the honest deliverable is reporting only. Write that
> up as the result.

Run with a CONVTRACE-off control arm and a `--no-jolt` pin, per this project's rule that *a gate must
prove its control is a control* — the defect that spoiled the previous cycle's evidence.

---

### BUILD 2 + PHASE 2 — FIX THE STOP RULE *(the deliverable)*

Patch **one** site. `+R` is unconditionally on the diagonal branch
(`gpu_lnl_intree.cu:`const int    lbM   = (freeRate==1) ? 0 : g_lbfgs_m;``, and the L-BFGS block is
entered only `if(lbM>0)`), so the L-BFGS-branch copy of the test is **dead code for `+R`** and touching
it only risks `+G`/`+I` bit-identity.

```
stationary      = (it >= 2) && (dec < TOL_NATS);          // TOL_NATS = 1e-3
params_settled  = max_c|Δrate_c| < 1e-4 && max_c|Δprop_c| < 1e-4;
damping_healthy = (mu < MU_CEIL);
conv =  ( dl<tol && stationary && damping_healthy )       // tighten the existing stop
     || ( stationary && params_settled && damping_healthy );  // NEW: early exit — this is the speed win
```

Three deliberate choices:
- **`TOL_NATS = 1e-3` is anchored, not invented**: `utils/tools.cpp:`score_diff_thres = 10.0;`` is the
  BIC window that gates model pruning, so a residual ≪1 nat cannot move a selection. Four orders of
  headroom.
- **`params_settled` matches upstream's own EM criterion verbatim in form** —
  `model/ratefree.cpp:`converged = converged && (fabs(prop[c]-new_prop[c]) < 1e-4);`` — which makes our
  stop comparable to stock's rather than a private invention.
- 🔴 **Do NOT match upstream's `pgtol` constant.** Upstream routes `+R` to L-BFGS-B with
  `max(gradient_epsilon, TOL_FREE_RATE)`, `TOL_FREE_RATE = 0.0001`, a projected-gradient sup-norm. Tested
  against the on-disk traces, that bar fires early and costs **767.7 nats on `LG+R4`** and **519.1 on
  avian R6** — 2–75× the pruning window. Cite it as provenance for the *form*, never the threshold.
- **The AND-gate alone is monotonically cost-increasing** (it can only make `conv` harder, converting
  false `conv=1` into cap-failure). **The disjunct is what buys the speed** and must be written in
  deliberately.

#### 🔴 GATE IT AS A SELECTOR, NOT A DIAGNOSTIC
A candidate's stopping point is a **selection channel**, confirmed at source. `filterSubst()` computes
`main/phylotesting.cpp:`double ok_score = best_score + Params::getInstance().score_diff_thres;`` and then
sets `MF_IGNORED` on models **that have not been fitted yet**
(`main/phylotesting.cpp:`for (model = finished_model+1; model < size(); model++)``). This is the
documented mechanism behind *"1e-2 fits 139 `+R` candidates, 1e-7 fits 42."*

**And the `+R` ladder is a coupled feedback loop, which is the biggest risk in this plan.** A candidate's
terminal lnL becomes the *acceptance bar* for the next rung
(`main/phylotesting.cpp:`if (prev_info.logl < new_logl + params.modelfinder_eps) break;``) **and** its
rates/props become that rung's *warm seed* (`model/ratefree.cpp:`void RateFree::initFromCatMinusOne(Checkpoint &ckp, double scale_factor) {``).
Those two effects have **opposite signs**: a better-converged `R{k}` gives `R{k+1}` a better seed *and* a
higher bar, and can burn all five re-seeds and leave `R{k+1}` worse.

⇒ **the gate must log the whole ladder** — per-*k* `(terminal lnL, decrement, re-seed count, final k)` —
on ≥2 seeds, plus cross-seed winner stability and RF on real data. **A winner-only gate will pass or fail
for the wrong reason and you will not be able to tell which.** Default-OFF, byte-identical when OFF.
Cost: ~8 h/arm ⇒ **30–35 GPU-h**. This is the real bill and it should not be started before Phase 1
returns.

---

### BUILD 3 — REPORT + FLOORS *(separate build; do not bundle)*

- **Report per candidate: predicted remaining gain**, certificate status, terminal `‖g‖`, and
  `λ_min/λ_max` emitted **after** the LM loop (today the emit precedes the loop on branch
  `opg-lambda-min-diagnostic` = `1bb82e14`, **which is pushed** but is **not** in mainline — `grep
  OPGLMIN` → 0 hits — so a build must merge it).
  🔴 **Do NOT report cap-exhaustion as the failure flag** — §1.4 shows it false-alarms on four
  substantively-converged sim cells. Report the *decrement*.
- **Floors:** our `1e-4` clamps are **GPU-path only** (`gpu_lnl_intree.cu:`if(r<1e-4)r=1e-4;``);
  `model/ratefree.cpp` is byte-identical to upstream at `MIN_FREE_RATE = 0.001`. Tightening the GPU clamp
  **shrinks the feasible set — it is not byte-identical** and needs its own selection gate. Keep it out
  of Build 2 so a winner flip stays attributable.

---

## 4. THE SPEED STORY (correctness first, but the speed is real and free)

Speed here is a **consequence** of a correct stop, not a separate lever:

1. **Early exit on converged candidates.** Iterations spent after a fit is already within 0.1 nat of its
   own final value: `on_dna_r10` **233 of 400**, `on_aa_r10` **184**, `on_dna_r8` **167**. On the
   simulated cells **~1.4–2.4× of the `+R` LM budget buys <0.1 nat** — recovered *exactly* by the
   disjunct, with no approximation.
2. **Unblocks the pure-`+R` CPU self-check removal.** Every `+R` candidate currently pays a full
   CPU `computeLikelihood()`: the GPU-mirror shortcut is gated `!freeRateOK`
   (`tree/phylotreegpu.cpp:`if (mf_devuse && !mf_devcheck && !freeRateOK && site_rate->getPInvar() <= 0.0 && !omp_in_parallel()) {``),
   and the source states the blocker is that *"pure-+R agreement is UNVALIDATED on real +R data
   (avian)"*. **Phase 1 produces exactly that cross-check as a by-product.** ⚠️ the cost of this for
   `+R` specifically has not been isolated on disk — the known 2.92×→1.10× figure is DNA-MF-wide.
3. **Not pursued:** replacing the `nFreeQ+1` finite-difference evals per iteration (~2,400 over a capped
   GTR run) with analytic derivatives. It is not exact work-removal and falls under this project's
   4-of-4 law that approximate ranking fails on real data.

---

## 5. WHAT THIS DELIVERS, STATED HONESTLY

**Claimable:** *reproducible, instrumented `+R` model selection* — every candidate carries a measured
statement of how far it is from its own optimum, the stop rule can tell convergence from damping, and
candidates that are done stop early.

**NOT claimable, and must not be implied:**
- *"Recovers the true number of rate classes."* avian shows a continuous rate density with no discrete
  true `k`; converged R8 beats R6; BIC is misspecified on the `k→k−1` singular boundary.
- *"Beats stock's stability."* Stock's stable `I+R6` is **seed-stable truncation** — upstream truncates
  silently too (`for (int step = 0; step < ncategory; step++)`, plus IQ-TREE Issue #38 reporting this
  exact failure mode, closed unresolved). Matching stock's stability would reproduce stock's
  under-optimisation.
- *"Fixes the optimiser."* No step policy is changed. Six have been tried and buried (§2).

**The honest novelty:** everybody's `+R` fits truncate — ours, upstream's, RAxML-NG's (its own wiki
concedes selection uses "a faster but usually less accurate EM") — **and nobody reports it.** Shipping a
per-candidate remaining-gain number is a contribution independent of whether the stop rule ships.

---

## 6. OPEN / DEFERRED

- **NPMLE fixed-grid reference.** ✅ The `+I` convexity blocker is **refuted**:
  `L_p = Σ_c catProp_v[c]·S_p + pinv·I_p` with `Σ catProp_v + pinv = 1` is linear in the weight vector
  (the `(1−pinv)` ratio is only IQ-TREE's *storage* convention); pose it in `(catProp_v, rate-grid)` with
  `pinv` as the rate-0 atom. 🔴 But it is **not a ceiling on `+R_k`** — `sup_G L(t₀,Q₀,G)` does not bound
  `sup_{t,Q,G_k} L`, and on disk `off_av8` beats `off_av6` by 600 nats *with different branch lengths*.
  Valid only as a fixed-`(t₀,Q₀)` reference. 🔴 And `m=128` **exceeds the 64 KB `__constant__` budget**
  (`3×128×20×8 = 61,440 B` + ~7.7 KB); `m=64` sits at ~38 KB, at the ceiling. **Deferred: real
  engineering, not "measurement only."**
- **Whether a converged avian `+R` value exists at all.** Genuinely open. Phase 1's cap-raise arm is the
  first honest test; both prior "existence proofs" died this cycle (one was a false convergence, the
  other a category error — a *measured* lnL does prove reachability, so "a much better value is
  reachable" still stands, just not "converged").

---

## 7. PROVENANCE OF THE CORRECTIONS IN THIS DOCUMENT

Kept because the project's rule is that negatives are the asset. In this cycle **the red-team destroyed
most of my evidence and the blue-team then reversed one of the red-team's own recommendations**:

| claim | fate |
|---|---|
| "39–1190× step ratio proves non-convergence" | **DEAD** — arithmetic identity of the stop rule; a DNA cell shows ratio 225× at a true shortfall of **0.058 nats** |
| "Every avian `conv=1` is false" | **RETRACTED** — `t1e-3_av`'s last 8 steps are a clean monotone decay; it may be near-converged |
| "Both existence proofs are dead" | **CATEGORY ERROR** — a measured lnL needs no convergence claim |
| my control cells | **NOT A CONTROL** — all from the OPG-instrumented build; matched arms emit no `[IRCONV]` |
| "regime boundary is `k`, not real-vs-sim" | **WRONG on the right metric** — on nats-still-being-gained it *is* real-vs-sim (8.478 vs ≤0.016) |
| red-team's "use sim DNA-100k as primary testbed" | **REVERSED** — those cells are substantively converged; using them inverts the selection pressure |
| "`maxgy` was dropped, not never-written" | **FALSE** — 0 files, 0 commits across all 37 worktrees; the binary was a throwaway |
| "we are 10× looser than upstream on floors" | **WRONG FILE** — `ratefree.cpp` is byte-identical to upstream; our 1e-4 clamps are GPU-path only |
