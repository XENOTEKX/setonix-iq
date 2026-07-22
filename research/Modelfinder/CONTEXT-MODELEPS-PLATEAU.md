# Context pack: the `modelEps` stopping-rule problem in IQ-TREE ModelFinder, and what we tried

**Self-contained briefing. No prior exposure to this codebase assumed.** Every code reference is
`file:line` in IQ-TREE 3 (fork base commit `ccabc96e`); every number is from a job artifact on disk.
Written 2026-07-23.

---

## 1. The one-paragraph problem

IQ-TREE's ModelFinder fits candidate substitution models (including FreeRate `+Rk`), computes BIC from
each fitted likelihood, and selects a model. Parameter optimisation runs as a loop of "rounds", and the
loop stops when one round fails to improve the log-likelihood by more than an epsilon. **On hard `+R`
cells this stop rule fires while the optimiser is still climbing.** The reported likelihood is then a
truncation point rather than an optimum, and because BIC is computed from it, the *selected model* can be
an artefact of where the optimiser gave up. That is the defect.

---

## 2. The stopping rule, exactly as implemented

`model/modelfactory.cpp:1676` — the outer parameter loop:

```cpp
for (i = 2; i < tree->params->num_param_iterations; i++) {
    ...
    new_lh = optimizeParametersOnly(i, gradient_epsilon, new_lh);
    ...
    if (new_lh > cur_lh + logl_epsilon) {      // modelfactory.cpp:1724  <-- THE STOP TEST
        cur_lh = new_lh;                       // improved: keep going
    } else {
        site_rate->classifyRates(new_lh);
        cur_lh = tree->optimizeAllBranches(100, logl_epsilon);
        break;                                 // did not improve enough: STOP
    }
}
```

Two things to notice:

1. **The test is on the *most recent round's* gain only.** There is no gradient test, no KKT residual, no
   projected-improvement bound — nothing that measures *how far from stationary the point is*. It asks
   only "did the last step help?".
2. **The loop is also capped** at `num_param_iterations`, default **100** (`utils/tools.cpp:7469`). Exiting
   by cap and exiting by convergence are not distinguished in the reported result.

### 2.1 ⚠ Which epsilon — this is a real trap, get it right

Three distinct knobs are routinely conflated. They are **not** one axis:

| variable | default | flag | where it applies |
|---|---|---|---|
| `modelEps` | **0.01** (`tools.cpp:7643`) | `-me` / `--epsilon` | the general model-parameter stop test |
| `modelfinder_eps` | **0.1** (`tools.cpp:7645`) | `--mf-epsilon` | **what ModelFinder actually passes when evaluating candidates** |
| `TOL_GRADIENT_MODELTEST` | — | — | the *gradient* tolerance passed alongside; a different quantity |

And the subtlety that catches people:

```cpp
// main/phylotesting.cpp:790-791   -- inside the ModelFinder window
double saved_modelEps = params.modelEps;
params.modelEps = params.modelfinder_eps;     // modelEps is OVERWRITTEN with 0.1
...
params.modelEps = saved_modelEps;             // :867 restored afterwards
```

Candidate evaluation calls
`optimizeParameters(brlen_type, false, params.modelfinder_eps, TOL_GRADIENT_MODELTEST)`
(`main/phylotesting.cpp:2619`, `:2630`, `:2650`).

⇒ **During ModelFinder candidate fitting the effective stop threshold is 0.1 nats, ten times looser than
the 0.01 that `modelEps` suggests.** A looser threshold breaks the loop no later, so it leaves *no less*
unrecovered slack.

*(There is also a GPU-side `JOLT_IR_TOL`, default `1e-7`, in this fork's accelerated path. It is a fourth,
unrelated knob. Never plot or compare these on one axis.)*

### 2.2 The other cap, which is not what its name suggests

Separately from `num_param_iterations = 100`, this fork's GPU path carries
`int brlenMaxIter = 400` as a **hard-coded default argument** at `tree/phylotree.h:2123`. It has no CLI
flag (verified: no occurrence in `utils/tools.cpp`), is undocumented, and is not an upstream constant. A
lot of historical discussion about "the 400 cap" refers to this inner branch-length cap, **not** to the
outer 100-round parameter cap. Conflating them produces wrong diagnoses.

---

## 3. Why the rule structurally cannot work — the plateau

The stop test assumes the likelihood trajectory is monotone-decelerating: small gain now ⇒ small gain
forever. **On real `+R` cells it is not.**

Measured on the avian dataset (48 taxa × 37.35 M sites, Jarvis 2014), `GTR+F+R8`, fixed tree:

- ~2 nats/round for about 150 rounds — a plateau, easily below any sensible epsilon
- then **~159 nats gained in six rounds**
- then decay to ~0.24 nats/round

A per-round threshold **cannot distinguish "near-stationary" from "about to find 159 nats."** This is a
mechanism-level argument, independent of any particular residual number: whatever epsilon you choose,
that trajectory defeats it.

The underlying geometry is a near-flat ridge, and we have measured it directly (§6.3).

### 3.1 A cautionary result: the plateau can be a truncation artefact

An earlier measurement of the avian weight-block residual gave **20.745 nats**. Raising the outer round
cap 100 → 500 collapsed it to **1.161 nats (−94%)**. The cell had never converged; the "residual" was
mostly distance-still-to-travel.

⇒ **Never quote a residual measured at a capped endpoint as if it were a stationarity residual.** Report
cap-exhaustion explicitly. (The 20.7-nat figure is withdrawn and must not be cited.)

---

## 4. Simple fixes we attempted, and why each failed

This is the important part: **the obvious fixes were tried, and they do not work.** Each entry is a real
experiment with a job ID.

### 4.1 ❌ Tune the tolerance

The most natural fix: tighten the epsilon so the optimiser stops later.

- A global `1e-2` tolerance change was gated end-to-end and **failed** (job `174142647`, RF criterion).
  `1e-3` also failed.
- Tightening does not monotonically help: on the avian cell **`1e-5` beats `1e-7` by 2.47 nats**. Tighter
  is not better.
- **The apparent "tolerance effect" is largely a candidate-count effect.** ModelFinder has a skip rule
  that abandons a rate family early; changing the tolerance changes *how many candidates get fitted at
  all* (`1e-2` fitted 139 `+R` candidates; `1e-7` fitted 42). Comparisons across tolerances are therefore
  comparisons across **different candidate sets**, not the same experiment.

⇒ Tolerance tuning is confounded and does not address the mechanism.

### 4.2 ❌ Raise the cap

Raising the cap does recover likelihood (§3.1: 20.745 → 1.161 nats), which proves the cells were
truncated. But it does not tell you *when to stop* — you have simply moved the arbitrary stopping point
further out, at real cost. **The cap is a symptom, not the disease.**

### 4.3 ❌ Add a cheap in-loop decrement as a stop rule

We built a diagnostic "decrement" statistic from quantities already resident in the optimiser loop, hoping
it could serve as a convergence test.

**Result: FAILED, and inverted** (job `174416929`). Climbing cells scored 1.9–6.9; converged cells scored
5.9–112 — so no threshold separates them, and the ordering is backwards. Root cause: the statistic was
**branch-dominated and k-independent**, so it measured branch conditioning rather than `+R` convergence.
Summing three arms into one number destroyed attribution.

⇒ Do not rebuild a stop rule on that statistic.

### 4.4 ❌ Replace the optimiser (three separate attempts)

- **OPG / empirical-Fisher curvature** — avian R6 *lost* 162.875 nats and capped; a second phase lost
  51.885 nats. Retired.
- **HARP (a damping/step-direction policy)** — a one-shot direction lost **23,413.927 nats**. A tuned
  variant helped one cell while neighbouring `k`/τ combinations regressed (R8 by 705.617 nats). No
  transferable policy. Closed.
- **One-step EM weight update** — *widened* the avian R6 endpoint spread from 216.760 to 535.313 nats
  (job `174342520`). ⭐ The lesson generalises: **convexity of a subproblem confers nothing on the
  alternating scheme** — an exact block-solve against a co-moving ridge can amplify drift.

### 4.5 ❌ Approximate/subsampled ranking

Ranking candidates on a cheaper approximation and re-polishing the winner fails on real data (3/3
precedent). One mechanism: ranking on an under-optimised state lets the `+I`/`+R` confound absorb the
error, producing spurious `+I` selection. **Simulated data cannot gate this class of change.**

---

## 5. What we are doing instead: measure the residual, don't guess the stop

The reframe: **the stop rule cannot be repaired by choosing a better threshold on the wrong quantity.**
Replace "did the last round help?" with "**how much likelihood is provably still available here?**"

This is possible because the `+Rk` weight subproblem is **convex**. With
`G = Σ_j w_j δ(r_j)`, `w_j ≥ 0`, `Σ w_j = 1`, `Σ w_j r_j = 1`, and pattern likelihood
`ℓ(w) = Σ_p n_p log(Σ_j w_j F_pj)`, the Hessian `H_jl = −Σ_p n_p F_pj F_pl / s_p²` is negative
semi-definite. So at fixed rates, the optimal weights can be found **exactly**, with a certificate:

- a **Frank–Wolfe gap** `G_w = max_{v ∈ C} ∇ℓ(w)ᵀ(v − w)`, giving `0 ≤ ℓ* − ℓ(w) ≤ G_w`; and
- an independent **self-concordant Newton-decrement bound** `ω*(λ) = −λ − log(1−λ)`, which is far tighter
  on near-degenerate high-`k` problems where the FW gap overstates by ≥1e7×.

That converts a boolean ("converged") into a measured number in nats ("at most this much is left").

### 5.1 The measurement, on cells IQ-TREE had already declared converged

Two simulated 100-taxon × 100,000-site cells, `+R8`, fixed tree, CPU. Both ran with
`Estimate model parameters (epsilon = 0.010)` — confirmed in the run logs, i.e. the **0.01** path, not the
looser ModelFinder path.

| cell | weight block | rate block | cycle total |
|---|---|---|---|
| DNA-100K `GTR+F+R8` | 1.482204356 | 4.198188321 | **5.680392677 nats** |
| AA-100K `LG+R8` | 1.490762723 | 2.009270684 | **3.500033407 nats** |

Both are **lower bounds** on recoverable slack (a feasible point was reached with that much more
likelihood; it is not claimed to be the optimum).

Against the stop threshold actually in force for those runs (0.01): **568×** and **350×**.
Against ModelFinder's own candidate threshold (0.1): **57×** and **35×** — and since a looser epsilon
breaks the loop no later, the ModelFinder path leaves *no less* than this.

**Reproducibility:** an identical rerun reproduced the gain, solve count and endpoint **bit-exactly**
(`0.000e+00` difference), so these are not optimiser noise.

---

## 6. Three findings that constrain any fix

### 6.1 The convex weight solver is not the bottleneck

Across all cells: zero iteration-capped weight solves; every `NUMERICAL_STALL` exit has its tightest
*valid* bound at the arithmetic floor (`9.773e-13` against a `1e-8` request). Those are converged solves
wearing a failure label. **Methodological warning:** the same solves report a raw Frank–Wolfe signed
residual of `4.516e-04`, a factor **4.6e8** larger. Classify on the tightest *valid* bound (Newton when
globally valid, else FW when valid) — classifying on the raw residual gives the opposite verdict from
identical data.

### 6.2 The real difficulty is the outer rate block, not the weights

The rate-location block is non-convex and is currently driven by quasi-Newton with **forward differences**
(`h = 1e-4·|x|`) applied to a *profiled* objective `φ(r) = max_w ℓ(w;r)`. Consequences we measured:

- inner-solve gap `G` injects gradient error `~G/h`, so **the endpoint depends on the search tolerance**;
- trial points leave the domain where the weight problem is soluble (10 rejections on DNA, 19 on AA, the
  first at the very first line-search step);
- if such a rejection lands on a *derivative probe*, the gradient component becomes `~1e10` and the search
  **freezes** — measured: an injected fault at that position lost the entire 4.198-nat gain while every
  health indicator still read "converged, monotone".

### 6.3 `+R` parameters are weakly identified — a flat ridge, measured

A **7.05 % move in one rate ratio (0.239281 → 0.256140) costs 0.0061 nats.** Two endpoints that far apart
in parameters are the *same basin* by any sensible likelihood threshold.

**Consequences for users, which matter more than the ridge itself:**
- **Protected:** the selected model, tree, branch lengths and support. BIC keys on the maximised
  likelihood, and one extra `+R` pair costs `2·ln(100000) ≈ 23` BIC units ≈ 11.5 nats — three orders of
  magnitude above this ridge.
- **Not trustworthy:** the printed rate/weight table. Those numbers move percent-scale at ~zero likelihood
  cost. Do not read them as point estimates; **never compare two runs, versions, or CPU-vs-GPU on
  parameter values** — compare likelihoods and fitted pattern probabilities.

⇒ On a well-converged cell the ridge is benign. The danger is that **the same flatness is what makes
"converged" and "still climbing" look identical to a per-round threshold.**

---

## 7. Summary for a reader deciding what to do

1. The stop test is `new_lh > cur_lh + logl_epsilon` on the **last round's gain only**
   (`modelfactory.cpp:1724`), with an undistinguished 100-round cap.
2. ModelFinder passes **0.1**, not the 0.01 that `modelEps` implies, and overwrites `modelEps` while doing
   so. Four different epsilon knobs exist; conflating them has produced several wrong conclusions.
3. Real `+R` trajectories are **non-monotone** (~2 nats/round for 150 rounds, then ~159 nats in six), so
   *no* per-round threshold is safe. This is structural, not a tuning problem.
4. **Every simple fix was tried and failed**: tolerance tuning (confounded by candidate count),
   cap-raising (moves the arbitrary point), a cheap in-loop decrement (failed inverted), and three
   optimiser replacements (OPG, HARP, one-step EM — all lost likelihood).
5. The workable direction is to **measure a certified residual** rather than threshold a gain — the weight
   block is convex and admits an exact gap certificate; the rate block needs a well-posed gradient and an
   in-domain trust region, which is the current work.
6. On two cells IQ-TREE called converged, **5.68 and 3.50 nats** were still available — reproducible
   bit-exactly.

---

## 8. Provenance

| item | value |
|---|---|
| Fork base | `ccabc96e` (`iqtree3-jolt-merge`, branch `jolt-gpu-merge`) |
| Measurement branch / HEAD | `freerate-profile-impl` @ `e8dac3eb` |
| Final gate job | `174469880`, binary sha256 `381e214d797c3b51`, 2026-07-23T05:36+10:00 |
| Artifacts | `/scratch/rc29/as1708/gems-verify/fr_p1s2_174469880/` |
| DNA cell base lnL | `-5697284.528522684` |
| AA cell base lnL | `-7608079.835423090` |
| Full record | `MODELFINDER-FULL-GPU-PLAN-REVIEW.md` Part II |

**Withdrawn claims — do not cite:** the 20.745-nat avian residual (truncation artefact); "weights dominate
the residual" (truncated control); "run-length invariance" (n=2, sign-flipped). A fuller retraction log is
in the review document, §9.
