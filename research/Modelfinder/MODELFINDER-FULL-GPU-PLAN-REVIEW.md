# Review — MODELFINDER-FULL-GPU-PLAN.md (certified FreeRate convergence plan)

> **This document has two parts.**
> **Part I** is a design review of the plan document, written 2026-07-21 *before* implementation began.
> **Part II** (from §5 onward) is the implementation record as of 2026-07-23 — everything actually built,
> every number measured, every claim retracted, and the current decision point. Part II is written to be
> read cold by someone with no prior exposure to this codebase.
>
> Where Part I and Part II disagree, **Part II wins** — it is grounded in measurements that did not exist
> when Part I was written.

---

# PART I — Design review of the plan (2026-07-21, pre-implementation)

**Reviewer pass 2026-07-21.** Every claim below was checked on disk or verified numerically; nothing is
accepted from the document's own narration. Verdict up front: **the core is sound and the direction is
right — this is the strongest `+R` document in the project. Four substantive defects, one of which
(C1) inverts the plan's own intent and is cheap to fix.**

---

## 1. VERIFIED STRENGTHS

### 1.1 ✅ The central production primitive is mathematically correct
I verified §5.1's quotient identity numerically. Holding every kernel `r_j·b_e` fixed, proposing weights
`u`, setting `m = rᵀu`, then committing `r' = r/m`, `b' = m·b`:

| claim | result |
|---|---|
| `r'_j·b'_e == r_j·b_e` (likelihood columns unchanged) | ✅ max error **4.44e-16** |
| `Σu_j = 1` | ✅ |
| `Σu_j r'_j = 1` (unit-mean gauge restored **without clamping**) | ✅ exact to 1e-12 |

And §4.2's concavity claim: with `ℓ(w)=Σ_p n_p log(Σ_j w_j F_pj)`, the Hessian
`H_jl = −Σ_p n_p F_pj F_pl / s_p²` is negative semi-definite — verified over 2,000 random directions,
worst quadratic form **−39.5**. ⇒ **the inner weight problem is genuinely convex and admits an exact gap
certificate.** This is the right primitive and it is correctly derived.

### 1.2 ✅ Dimension bookkeeping is right in both formulations
Quotient: `k−1` weight DOF + `k−1` rate ratios = **2k−2** ✓. Literal: `k−2` weight DOF + `k` absolute
rates = **2k−2** ✓. And §5.4's admission that *"for R2 the literal block normally has zero weight degrees
of freedom"* is correct (k−2 = 0) and honest — a lesser document would have hidden that.

### 1.3 ✅ Load-bearing evidence claims check out
- §1.1 *"iteration 400 still gained 0.270 nat"* — **exact**, re-measured from `a1_1e-7_av.console`
  (final lnL `−11216886.230053`).
- §1.2 *"OPG lambda-min was printed before optimization and describes a warm seed"* — **confirmed
  independently**: the emit sits at `:3559`, the LM loop at `:3626`, on branch
  `opg-lambda-min-diagnostic`.
- §1.1 *"Stock IQ-TREE v3.1.2 job 174338165 selected +I in 9 of 9 arms"* — matches the record.
- §10.1 seam — `model/modelfactory.cpp:`double jolt_lh = tree->optimizeParametersJOLT(fixed_len);``
  confirmed as the ModelFinder dispatch.
- §10.4's insistence on a new unweighted-column path is **justified, not paranoia**: the kernel comment
  says so itself — `gpu_lnl_intree.cu:`wnum[(size_t)c*nptn+ptn]=lcc;   // G.5.1: Lc(p)=Σ_x g_val0[c,x]·θ (category weight w_c already folded into g_val0)``.

### 1.4 ✅ The correctness posture is exactly right
*"The word MLE must not appear in a success enum"* (§8.1); typed statuses; §0.6/§11.2 forbidding a
cap-bound candidate from being scored, pruned from, or used as a warm seed. This is the single most
important rule in the document and it is stated without hedging.

### 1.5 ✅ τ_L is better calibrated than my own parallel proposal
`τ_L = max(1e-4 nat, 100·E_num)` with a 0.01-nat promotion ceiling, **after measuring the numerical
floor `E_num` first** (§8.3). Anchored against the 10-nat pruning window and 11–14-nat BIC margins with
3 orders of headroom. Measuring `E_num` before choosing the bar is the part I got wrong elsewhere.

### 1.6 ⭐ A theoretical foundation the plan has but does not claim
The plan targets the **value** and explicitly refuses to chase weight labels (§4.2 *"fitted pattern
probabilities can be identifiable even when individual weights are not"*; §8.2 *"compare the canonical
fitted mixture… not arbitrary labels"*). **That is precisely what mixture theory licenses:** Ghosal &
van der Vaart (2001, *Ann. Statist.* 29(5)) prove the mixture MLE converges in Hellinger at
near-parametric rate `(log n)^κ/√n` while the *parameters* converge only at `n^{−1/4}` (Chen 1995;
Dwivedi et al. 2020). **Cite this.** It is a stronger foundation than any reference currently in §17,
and it converts §4.2 from an engineering convenience into a theorem-backed design choice.

---

## 2. DEFECTS

### 🔴 C1 — Stage 0 is not a cheap kill gate, and it is supposed to be *(most important; cheap to fix)*
§3.5 says *"Stage 0 is a kill gate. The rest of this document is authorized only for the block that the
measurements implicate."* Correct intent. **But §3.4's attribution requires items 1 and 2 — the quotient
and literal weight profiles with exact convex gaps — and those are Phase 0B deliverables.** So the gate
that is meant to authorize the solver build **requires building two nontrivial constrained solvers
first.** The intent is inverted.

**Three genuinely free pre-gates exist and should run before Phase 0B:**
1. **"Is it still climbing?" is already answered, with no new code** — last-10-accepted-iteration lnL
   gain: avian `on_av_r6` **8.478 nats** vs `on_dna_r8`/`on_dna_r10` **0.001**, `on_aa_r10` **0.002**,
   `on_aa_r8` **0.016**. Pure lnL differences, no instrumentation.
2. **Block attribution without any profiler** — `JOLT_IR_FREEZE_MODEL` and `JOLT_IR_FREEZE_BRLEN` **both
   still exist** in the canonical source (verified). Freezing one arm and measuring the lnL the other
   recovers gives model-vs-branch attribution for the cost of a few short runs.
3. **A free in-loop residual** — the diagonal Newton decrement
   `½(Σ_e g_df²/|g_ddf| + Σ_c g_y²/|ddY| + Σ_c g_z²/|ddZ|)` is denominated in nats and uses only values
   already host-resident at the `[IRCONV]` emit. Crude relative to a profile gap, but free, and it
   answers the gating question.

⇒ **Insert a "Stage 0-minus-1" using these, before authorizing Phase 0B.** If they already implicate a
block, Phase 0B starts targeted; if they refute the weight hypothesis, the project saves the entire 0B
build.

### 🔴 C2 — the cap policy will wrongly refuse to score fits that are actually converged
§0.6 says *"A cap… is a typed failure"* and §8.2 requires *"no cap… occurred"* for certification.
**But four simulated cells hit the 400-iteration cap while gaining 0.001–0.016 nats over their last ten
iterations — they are done, and the cap is incidental.** Under this policy they would be refused
certification, and §11.2 would escalate to `INCOMPLETE` for a fit that is correct to well inside `τ_L`.

**Fix (small, and the machinery is already specified):** make the failure criterion the **residual/gap**,
not the cap. A capped fit with all residuals `≤ τ_L` should certify (optionally as a distinct status
recording that the budget was exhausted); a capped fit with a material residual is `MAXITER`. §8.2
already measures every residual needed — only the conjunct *"no cap occurred"* has to go.

### 🟡 C3 — the Phase-A rebuttal is stated too weakly, and the strong form is available
§1.1 rebuts `jolremA_174342520` with *"one EM update is not an exact observed-likelihood weight solve."*
True — EM's M-step maximises the expected complete-data likelihood, not the observed-data likelihood —
**but it does not answer the recorded Phase-A lesson**, which was: *convexity of a subproblem confers
nothing on the alternating scheme; an exact block-solve against a co-moving ridge amplifies drift.*
That objection applies to a **more** exact block solve, possibly more strongly.

**The real answer is architectural, and the plan already implements it without arguing it: this is
variable projection, not alternation.** The outer objective is `φ(r,b) = max_w ℓ(w,r,b)`; §7.1 re-profiles
weights at *every* rate trial; §7.6 accepts on profile *intervals*. Monotonicity therefore holds on the
profiled objective, which is exactly what Phase A's alternating scheme lacked. **State this explicitly**
— it is the difference between "we believe it differs" and "here is why that failure mode cannot recur."
(Golub–Pereyra is already in §17 but demoted to *"conceptual precedent only"*; it is doing more work than
that.)

### 🟡 C4 — boundary certification is the largest practical risk and is unquantified
§7.3 requires, for any empty slot, a *"globally bounded continuous insertion-pricing"* solve over the
entire admissible rate domain via interval branch-and-bound or a proved Lipschitz envelope, and correctly
refuses to accept dense scanning as a bound. **That is a hard global-optimisation subproblem on a
phylogenetic likelihood.** Without it the best available status is `BOUNDARY_DIRECTIONALLY_TESTED` =
unresolved, and §11.2 escalates an unresolved candidate to `INCOMPLETE` for the whole analysis.

**Zero weights and rate collisions are common at high k on real data** — the fitted avian R6 already
carries two categories at rate 1.289/1.312. So the plausible failure mode is: **certified mode routinely
returns INCOMPLETE on exactly the real high-k datasets that motivated the work.** The plan needs (a) a
pre-registered estimate of what fraction of avian R2–R10 candidates land on a boundary, and (b) a
degraded-but-useful posture short of "whole analysis incomplete."

### 🟢 C5 — scope realism and the minimum shippable unit
Phases 0A→7 — CPU reference solver, then a new versioned GPU ABI, then full ModelFinder integration — is
a multi-month program with the GPU evaluator not arriving until Phase 4. That may be correct, but the
document never names a **minimum shippable unit**. My read: **Stage 0 + typed status + refusal to score
uncertified candidates is independently shippable and captures most of the correctness value**, whether
or not the profile solver ever lands. Worth stating, so the correctness win is not hostage to the
solver's success.

### 🟢 C6 — minor
- §1.1's **0.270 nat** (the `a1_*` PARS cell) and other documents' **0.737 nat** (`off_av6`, fixed-tree
  cell) are *different cells*. Both verified; label them, or they will be mixed later.
- §1.2's *"--thread-model fails to evaluate R3+ in the observed path"* is load-bearing for §11.3 and I did
  **not** verify it. Mark as unverified or cite the artifact.
- §1.2's caution that the HARP R6-vs-R8 comparison used different τ policies is **correct and matches an
  independent finding** — the two endpoints are max-over-τ vs max-over-τ. Worth keeping prominent,
  because the "true avian optimum is ≥ R8" claim circulates elsewhere in the project in a stronger form
  than the evidence supports. *(The same-policy `off_av8 − off_av6` = 600 nats comparison does survive.)*

---

## 3. RELATIONSHIP TO THE INSTRUMENT PLAN

`PLUSR-CONVERGENCE-INSTRUMENT-PLAN.md` and this document **reach the same diagnosis from opposite ends
and do not conflict**:

| | this plan | instrument plan |
|---|---|---|
| diagnosis | *"Current convergence reporting is insufficient… no KKT, projected-gradient, restart, or block residual certificate"* (§1.1) | the stop test cannot distinguish convergence from damping (`dl ∝ 1/mu`) |
| fix | fix the **solve** — certified convex profiling | fix the **test** — free in-loop residual |
| cap policy | typed failure, never scored | same intent, and **both need C2's correction** |
| cost | multi-month, 8 phases | 1 build + ~1.5–2 GPU-h to a kill gate |

⇒ **Merge them: the instrument plan is the cheap front-end of this plan's Stage 0** (it is essentially
Phase 0A, but free and available now). It does not replace §3.4's block attribution — it *precedes* it,
and it can redirect or kill the Phase 0B build before that build is paid for.

---

## 4. RECOMMENDATION

**Proceed — with C1 and C2 applied first.**

1. Apply **C2** (cap ⇒ residual-based, not cap-based, failure). One conjunct.
2. Insert **C1's Stage 0-minus-1** — free lnL-gain attribution, the existing `FREEZE_*` flags, and the
   in-loop decrement — *before* authorizing Phase 0B.
3. Strengthen **C3** to the variable-projection argument, so the Phase-A precedent is answered
   structurally rather than by a technicality.
4. Add a pre-registered boundary-frequency estimate for **C4** before Phase 1 commits to insertion
   pricing being achievable.
5. Split the document: Stage 0 + status contract as the live plan; §4–§11 as a conditional annex marked
   *authorized only if Stage 0 implicates the weight block.* At present ~90% of the text is detailed
   design for a hypothesis the document itself has not yet established — which is honest, but creates
   sunk-cost pressure toward building it regardless.

**What I would not change:** the quotient primitive, the typed-status contract, the refusal to score
uncertified candidates, the separation of `JOLT_IR_TOL` / `--mf-epsilon` / `-eps`, and the insistence on
same-state CPU/GPU parity. Those are the durable parts and they are right.

---
---

# PART II — Implementation record and current position (2026-07-23)

**Audience: a reader with no prior exposure to this codebase.** Everything in Part II was re-verified
against on-disk artifacts on 2026-07-23 immediately before this document was written. Numbers are quoted
to the precision the instrument emits. Where a claim was made earlier in the project and later withdrawn,
it appears in §9 (the retraction log) rather than being silently deleted.

---

## 5. Orientation — what problem this is, and what has been built

### 5.1 The problem in one paragraph

IQ-TREE's FreeRate (`+Rk`) model fits a discrete rate distribution `G = Σ_j w_j δ(r_j)` subject to
`w_j ≥ 0`, `Σ w_j = 1`, `Σ w_j r_j = 1` (public dimension `2k−2`). On hard real datasets the optimiser
terminates at points that are *not* stationary: it exhausts an iteration cap while still climbing, and
its stop rule (a per-round likelihood-gain threshold — see §7.5 for which epsilon, it is not one knob)
cannot distinguish
"near-stationary" from "about to find a lot more". Because ModelFinder then computes BIC from those
endpoints and selects `k` from them, an under-optimised `+Rk` fit can change the selected model. The plan
(`MODELFINDER-FULL-GPU-PLAN.md`) proposes replacing the boolean convergence test with a **certified**
solver: an exact convex weight profile with a likelihood-gap certificate, plus typed statuses, plus a
refusal to score any candidate that is not certified.

### 5.2 Where the code is

| item | location |
|---|---|
| Implementation tree | `/scratch/rc29/as1708/iqtree3-freerate-profile` |
| Branch / HEAD | `freerate-profile-impl` @ `e8dac3eb` (pushed to GitHub `XENOTEKX/setonix-iq`) |
| Base commit | `ccabc96e` (`iqtree3-jolt-merge`, branch `jolt-gpu-merge`) |
| Build dir | `build-baseline-head/` — **⚠ misleading name, it is NOT a baseline**, it contains the probe (`strings … \| grep -c FRSOLVE` = 12). The true control is `baseline-bin/iqtree3-head-ccabc96e` (count = 0). |
| Gate script | `setonix-iq/gadi-ci/gems/gems_fr_phase1_step2.sh` |
| Gate artifacts | `/scratch/rc29/as1708/gems-verify/fr_p1s2_<JOBID>/` |

**Push topology (relevant if you are asked to push):** `origin` is a *local* bare hub
(`/scratch/rc29/as1708/iqtree3-jolt-merge`); GitHub is a second hop from there and needs interactive
credentials. Two hops, not one.

### 5.3 Source files added by this work

| file | role |
|---|---|
| `model/freerateprofile.{h,cpp}` | The convex weight profiler. Active-set Newton + Frank–Wolfe, with **two** independent gap certificates: the FW gap and a self-concordant Newton-decrement bound `ω*(λ) = −λ − log(1−λ)` with a dual-feasibility (globality) test. |
| `model/freerateeval.cpp` | Unweighted component extraction (`F_pj` at uniform proportions) + the one-block diagnostics (`[FRATTRIB]` lines). |
| `model/freerateinternal.h` | Shared `RateBlockOracle`: `writeRates`, the anchor/`frozen_ratio` parameterisation, `FR_RATIO_LOWER = 1e-7`, `FR_RATIO_UPPER = 1.0`. |
| `model/freeratesolver.{h,cpp}` | **The step-2 probe** — the re-profiled rate-block objective under audit here. Emits `[FRSOLVE]` lines. |
| `model/freeratefit.{h,cpp}` | The typed-status / certification apparatus. **Present but entirely unwired** — see §10.3. |

### 5.4 How to run it

All diagnostics are **inert unless explicitly enabled**, so an ordinary run is byte-for-byte unchanged
(this is gated, see G3b in §7.2):

```bash
IQ_FR_ATTRIB=1 IQ_FR_SOLVE=1 \
IQ_FR_SOLVE_GAP=1e-8 \          # inner forcing gap for SEARCH solves
IQ_FR_SOLVE_MEASURE_GAP=1e-11 \ # inner gap for MEASUREMENT solves (see §8.4)
IQ_FR_SOLVE_TRIALS=800 \        # rate-trial budget
IQ_FR_SOLVE_FAULT_AT=<n> \      # fault injection at evaluation n (gate use only)
  iqtree3 -s <aln> -m GTR+F+R8 -n 0 -seed 1 -starttree PARS --no-jolt -nt 104
```

---

## 6. The two fixed test cells

Both are simulated 100-taxon / 100,000-site alignments; both `+R8`, both CPU-only (`--no-jolt`), both
from a fixed parsimony start with `-n 0` (no tree search — the tree is fixed, so the objective is fixed).

| cell | alignment | model | `k` | `ndim` | anchor | `base_lnl` |
|---|---|---|---|---|---|---|
| DNA-100K | `datasets/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` | `GTR+F+R8` | 8 | 5 | 7 | `-5697284.528522684` |
| AA-100K | `datasets/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` | `LG+R8` | 8 | 6 | 7 | `-7608079.835423090` |

`ndim < k−1` because zero/low-weight categories are frozen out of the rate block (they have no
identifiable location — §7.2 of the plan). DNA freezes 2, AA freezes 1.

---

## 7. What the probe measures, and the verified results

### 7.1 The step-2 objective

The probe runs **one** major cycle of the §7.1 block cycle at fixed tree, branches and Q:

1. **`w1`** — profile weights at the incumbent rates, commit if improving.
2. **rate pass** — maximise the **profiled** objective `φ(r) = max_w l(w; r)`. Each rate trial runs a full
   component extraction *and* a full convex weight solve. Driven by IQ-TREE's `dfpmin`
   (`utils/optimization.cpp`), which computes gradients by **forward differences** with
   `h = ERROR_X·|x| = 1e-4·|x|`.
3. **realise** — re-evaluate at the endpoint to get a coherent `(r, w)` pair.
4. **`w2`** — re-profile weights at the new rates.
5. **restore** — element-wise, and prove it.

It **certifies nothing** (`STATUS=LEGACY_UNCERTIFIED`) and **commits nothing**. It exists to measure cost
and gain before the outer machinery (§7.6 acceptance, `χ_r`, support events, start portfolio) is built on
top of it.

### 7.2 Final gate — job `174469880`, binary sha256 `381e214d797c3b51`, 2026-07-23T05:36+10:00

**All structural checks passed.** Per-cell (verified from the `[FRSOLVE]` lines, not from the verdict
summary):

| cell | search gap | `rate_gain` | `w1_gain` | solves | domain rejects | stalls | realised | suspect |
|---|---|---|---|---|---|---|---|---|
| `dna_gap8` | 1e-8 | **4.198188321** | 1.482204356 | 280 | 10 | 101 | 1 | 0 |
| `dna_rep1` | 1e-8 | **4.198188321** | 1.482204356 | 280 | 10 | 101 | 1 | 0 |
| `dna_gap5` | 1e-5 | 4.192094513 | 1.482204356 | 169 | 10 | 8 | 1 | 0 |
| `dna_gap2` | 1e-2 | 4.192038548 | 1.482204356 | 170 | 10 | 2 | 1 | 0 |
| `aa_gap8` | 1e-8 | **2.009270684** | 1.490762723 | 618 | 19 | 128 | 1 | 0 |
| `dna_fault6` | 1e-8 | **0.000000000** | 1.482204356 | 35 | 11 | 11 | 1 | **1** |
| `dna_fault12` | 1e-8 | 4.198014217 | 1.482204356 | 228 | 12 | 98 | 1 | 0 |

Supporting gate arms, all passing:
- **G3b inert-when-off** — with the switches unset, published lnL (`-5697284.5285`) and the `.treefile`
  are byte-identical to the instrumented run.
- **G4 restore integrity** — `max_param_err = 0.000e+00` and `restore_err = 0.000e+00` on all 7 cells.
- **G5b** — the three DNA rungs share an identical `base_lnl` (they are the same cell and state).
- **`moment_out`** — the §4.1 contract `Σ w_j r_j = 1` asserts to `1.000000000000` on every realised cell.

### 7.3 Endpoints (emitted for the first time in this gate)

```
dna_gap8  anchor_rate=2.37481365735  box_lo=0 box_hi=0
          0.0563378154933, 0.101825867165, 0.204817387828, 0.239280955944, 0.416021202814
dna_rep1  anchor_rate=2.37481365735  box_lo=0 box_hi=0
          0.0563378154933, 0.101825867165, 0.204817387828, 0.239280955944, 0.416021202814   <- BIT-IDENTICAL
dna_gap5  anchor_rate=2.37483907937  box_lo=0 box_hi=0
          0.0562828579649, 0.0997437488574, 0.204648181746, 0.256139803828, 0.414422385939
dna_gap2  anchor_rate=2.37485118164  box_lo=0 box_hi=0
          0.0562893072993, 0.0999367284347, 0.204652634233, 0.255896360704, 0.414364201576
aa_gap8   anchor_rate=2.39046706623  box_lo=0 box_hi=0
          0.00711779116736, 0.0554779688983, 0.16200675435, 0.278629997721, 0.217189439882, 0.403370947619
```

Values are `r_c / r_anchor` for the free categories, in ascending category-index order. No box bound is
active on any cell, so the `[1e-7, 1]` ratio envelope is not what is stopping the search.

### 7.4 Cost, decomposed (seconds, one major cycle)

| cell | extract | convex solve | realise | dominant | s / trial |
|---|---|---|---|---|---|
| `dna_gap8` | 11.467 | **117.457** | 5.981 | convex-solve | 0.4195 |
| `dna_gap5` | 8.048 | 65.147 | 4.597 | convex-solve | 0.3855 |
| `dna_gap2` | 7.383 | 62.209 | 4.381 | convex-solve | 0.3659 |
| `aa_gap8` | 169.630 | **253.835** | 90.536 | convex-solve | 0.4107 |

**Both data types are convex-solve-bound** once allowed to converge. An earlier "AA is extract-bound"
reading came from budget-truncated runs and is withdrawn (§9). A 10⁶× looser inner gap saves only ~45% of
wall time (134 s → 73 s): **the trial count dominates cost, not the inner gap.**

---

## 8. The four established findings

### 8.1 ✅ Material recoverable slack exists, and it is a lower bound

On DNA-100K R8, after committing the weight block (`w1_gain = 1.482204356`), the re-profiled rate block
recovers a further **`4.198188321` nats**; on AA-100K R8, **`2.009270684`**. Both realised, unsuspect,
`profile_capped = 0`, and bit-reproducible on rerun.

This is an **existence result**: a feasible point was reached whose likelihood, evaluated directly and
backed by a certified weight solve, sits that far above the weight-committed incumbent. It is **not** a
claim that this is `φ`'s optimum. The corresponding one-block diagnostic on the same AA cell reports
`STATUS=UNCONVERGED … gain is a LOWER BOUND on rate-block slack, not a block optimum`.

### 7.5 Which epsilon these cells actually stopped at — the multiple depends on the path

Both gate cells logged `Estimate model parameters (epsilon = 0.010)`, i.e. they ran the explicit-model
path at `modelEps = 0.01` (`utils/tools.cpp:7643`). Against that, the DNA cycle total
(`w1 + rate = 5.680392677`) is **568×** the threshold and AA (`3.500033407`) is **350×**.

**But ModelFinder's own candidate loop does not use that value.** It passes
`params.modelfinder_eps`, default **0.1** (`utils/tools.cpp:7645`), at `main/phylotesting.cpp:2619`,
`:2630`, `:2650` — and it *overwrites* `params.modelEps` with it for the duration
(`main/phylotesting.cpp:790-791`, restored `:867`). Against 0.1 the same slack is **57×** and **35×**.

A looser epsilon breaks the loop no later on the same trajectory, so the ModelFinder path leaves **no
less** unrecovered slack than measured here — but the *multiple* is 10× smaller, and quoting "568× the
ModelFinder threshold" would be wrong. The stop test itself is at `model/modelfactory.cpp:1724`
(`if (new_lh > cur_lh + logl_epsilon)`), and `modelEps` never appears in `modelfactory.cpp` at all.

### 8.2 ✅ The convex weight solver is exonerated

`profile_capped = 0` on every cell of every gate. Every `NUMERICAL_STALL` has its **tightest valid bound**
at the arithmetic floor — `dna_gap8`: 101 stalls, worst bound `9.773e-13` against a `1e-8` forcing gap,
`stall_no_valid_bound = 0`. These are *converged solves wearing a failure label*, not a degenerate tail.

**Methodological note that matters:** the same solves report a raw signed directional residual of
`4.516e-04` — a factor of **4.6e8** larger. The signed residual is the Frank–Wolfe quantity, which
`freerateprofile.h` documents as overstating the true shortfall by ≥1e7× on near-degenerate high-`k`
problems. Classifying on it produces the *opposite* verdict ("a real degenerate tail") from identical
solves. **The stall test must use the tightest valid bound (Newton-when-global, else FW-when-valid), and
must count the no-valid-bound case rather than silently skipping it.**

### 8.3 ✅ A flat ridge — weak identifiability, measured

With the measurement tolerance controlled (§8.4), the DNA rungs land at genuinely different endpoints:

| comparison | max relative endpoint difference | Δ lnL |
|---|---|---|
| `gap8` vs `rep1` (same binary, same gap) | **0.000e+00** | 0.000000 |
| `gap8` vs `gap5` | 7.046e-02 | 6.093808e-03 |
| `gap8` vs `gap2` | 6.944e-02 | 6.149773e-03 |

**A 7.05% move in `r3` (0.239281 → 0.256140) costs 0.0061 nats.** The bit-exact rerun proves this is
deterministic, not optimiser noise.

**Correct interpretation, per the plan's own thresholds:** §8.3 defines `τ_restart = 0.1` nat as the
*basin* policy threshold and states explicitly that it "is not a numerical stationarity certificate or a
bound on the unknown global candidate optimum". `0.0061` is **16× below** `τ_restart` ⇒ these are the
**same basin, at different points on a flat direction**. This is textbook `+R` weak identifiability and it
*confirms* the plan's §4.2/§8.2 insistence on comparing the fitted mixture and likelihood rather than
parameter labels.

**Consequence for an end user:** the selected model, tree, branch lengths and support are protected —
BIC keys on the maximised likelihood, and one extra `+R` pair costs `2·ln(100000) ≈ 23` BIC units ≈ 11.5
nats, three orders of magnitude above this ridge. What is *not* trustworthy is the printed rate/weight
table: those numbers move percent-scale at ~zero likelihood cost, so they must not be read as point
estimates, and **runs or engines must never be compared on parameters.**

### 8.4 ✅ The measurement tolerance must be separated from the search tolerance

`lnl_after_rate` is the exact tree likelihood at weights certified only to the gap they were solved at. If
the realise step uses the *search* gap, a loose rung reports a value up to `forcing_gap` below the true
profiled value **at the identical endpoint** — an artefact that can be as large as 1e-2 nats.

Fixed in `e8dac3eb`: `w1`, the seed identity check, the realise, and `w2` all solve at `measure_gap`
(default `1e-11`, clamped never looser than the search gap) while the rate pass searches at
`forcing_gap`. **Search cheap, measure tight.**

---

## 9. Retraction log — claims made and withdrawn

Every entry was withdrawn on evidence, and the evidence is on disk. This list exists because the *pattern*
matters more than any individual error (see §11).

| claim | why it died |
|---|---|
| "avian weight-block residual = 20.745 nats" | Truncation artefact. Raising the outer cap 100→500 collapsed it to 1.161 (−94%). **Never quote it.** |
| "weights dominate the residual" | Artefact of a truncated rate control (2–3 BFGS iterations on a 5–7-dim problem). Corrected, the blocks are comparable, and on the best-optimised avian cell the **rate** block leads 8:1. |
| "the residual is invariant to run length" | `n = 2`, sign-flipped: one cell held, the other collapsed. Not a law. |
| "`moment_out = 1.0` proves the §4.1 contract holds" | **Tautological.** It was computed *after* `restoreAll()`, and `RateFree` renormalises `Σ w_j r_j → 1`, so it printed exactly 1 for any input — including on four *aborted* cells, which is what exposed it. Quoted twice before the tell was noticed. |
| "zero capped solves ⇒ no degenerate tail" | The counter was blind to `NUMERICAL_STALL`, which is the tail's actual exit. |
| "these stalls **are** the degenerate tail" | Over-correction of the above: classified on the raw signed residual, which overstates by ~4.6e8×. They are converged solves at the arithmetic floor. |
| "the forcing ladder refutes over-solving" | The ladder cannot compare gains at all — each rung optimises a different approximation of `φ`, and a loose inner gap `G` injects gradient error `~G/h`. |
| "the tight rung finds a genuinely better optimum, +0.0061 nats = 61× τ_L" | Three faults: it used `τ_L` (a *stationarity* resolution) as a *basin/optimum* yardstick when `τ_restart = 0.1` is the applicable one; the spread was inside the measurement envelope at the time; and it was **a claim about endpoints made without ever observing an endpoint**. |
| "the 4.192 result is not reproducible / chaotically sensitive" | Withdrawn *in the other direction*. It was concluded from one budget-aborted row while the reproducing rows sat two lines below in the same table. `dna_rep1` later reproduced the endpoint bit-exactly. |
| "AA is extract-bound" | Artefact of budget truncation. With an adequate budget AA is convex-solve-bound (253.8 s solve vs 169.6 s extract). |

---

## 10. Open defects and unimplemented plan requirements

### 10.1 🔴 The search leaves the domain on every real cell

10 rejected trial points on DNA, 19 on AA, **the first at the first line-search evaluation** (eval 8 at
`ndim=5`, eval 9 at `ndim=6`). All are `INVALID_INPUT` from `prepareProblem` — component values
non-finite.

**Mechanism (inferred, consistent with all evidence, not yet directly instrumented):** the ratio box
bounds the *ratios* but not the induced *rate scale*. `writeRates` normalises `r_j = ρ_j / s` with
`s = Σ_j w_j ρ_j`; as ratios approach `FR_RATIO_LOWER = 1e-7`, `s` collapses, the anchor rate `1/s` blows
up, and `exp(eval · r · b)` overflows in the kernel. **A one-line dump of `max_j r_j` at the rejection
would confirm this directly and has not been done.**

### 10.2 🔴 A rejected *gradient probe* destroys the pass; the guard is structurally blind to it

Rejected trials return a large finite penalty. `derivativeFunk` never clamps and applies no domain
awareness, so if a rejection lands on a forward-difference probe the gradient component becomes
`~penalty/h ~ 1e10` and the search freezes. Measured, four orders of magnitude apart:

| injection site | `rate_gain` | vs control `4.198188321` |
|---|---|---|
| eval 6 (gradient block, `ndim=5` ⇒ evals 2–7) | **0.000000000** | loses **everything** |
| eval 12 (line search) | 4.198014217 | loses 1.7e-4, `endpoint_shortfall = 4.0e-05` |

`endpoint_shortfall` **cannot detect the first case**: it measures "walked past a better point", and a
frozen search walks past nothing, so its shortfall is zero by construction. The frozen run reported
`rate_monotone = 1, shortfall = 0.005` — indistinguishable from healthy. A separate `gain_suspect` flag
(fires when a pass rejected a point *and* ended with no material gain) was added to cover it and is proven
to fire by injection.

**Evaluation index layout** (needed to read any fault result): eval 1 = seed; eval 2 = `derivativeFunk`
base `fx`; evals 3 … `ndim+2` = the forward-difference probes; eval `ndim+3` onward = first `lnsrch`
trial.

### 10.3 🔴 The endpoint may violate the §4.1 nondecreasing-rate contract — new, 2026-07-23

The AA endpoint ratios are **not monotone**: `… 0.278629997721, 0.217189439882 …`. The published AA rates
are sorted ascending (`0.1179 … 2.3906`, anchor `2.3906` matching `anchor_rate = 2.39047`) and `free_cat`
is built in ascending category-index order, so **the search crossed two rate categories**.

This is not a likelihood error — a permutation is the same model — but it leaves the state non-canonical.
`model/freeratefit.cpp` already contains `canonicalizeFreeRatePoint` (`:469`) and a certification check
that refuses a "final point … not in canonical rate order" (`:705`). **Neither is reachable:**
`FreeRateFitResult` is referenced **zero times** in `freeratesolver.cpp`. The entire typed-status apparatus
is built and unwired. A crossed endpoint would fail certification the moment it is wired.

### 10.4 Unimplemented plan requirements (audited against Phase-1 deliverables and §8.2)

| requirement | status | is its absence *causing* an observed problem? |
|---|---|---|
| §7.2 ordered rate trust region | **absent** (stand-in: `dfpmin`'s box) | **Yes** — §10.1 domain excursions |
| §7.6 interval acceptance | **absent** | **Yes** — §10.2, and endpoint-below-best |
| §7.6 forcing *schedule* (loose-early, tighten geometrically) | **absent** — `forcing_gap` is a single fixed value | **Yes** — the whole ladder ambiguity |
| `χ_r` (projected rate first-order improvement, §8.2) | **absent** | **Yes** — no stationarity measure, so "converged" vs "truncated" is undecidable |
| §7.3 support events (merge / split / insertion pricing) | absent; zero-weight *freezing* is present | Deferred — correctly caps status at `BOUNDARY_DIRECTIONALLY_TESTED` |
| §7.7 deterministic start portfolio | absent | Deferred; blocks any globality claim |
| §7.1 multi-cycle | absent (`major_cycles = 1` is hard scope) | Deferred |
| Quotient pocket (§5.1/5.2) | profiler supports it; solver only ever builds `LITERAL_MASS_MEAN` | Deliberate scope choice |
| Boundary activity / metrics population | absent | Deferred |
| `bound_check[]` passed all-false ⇒ `dfpmin`'s own restart is inert | present but disabled | Minor |

### 10.5 Known cost inefficiencies (verified, unfixed)

1. **`extractComponentsAtUniformProp` runs an extra full `tree->computeLikelihood()` on every trial**
   purely to self-validate the uniform-prop identity (`total_likelihood_abs_error`). It has never fired on
   production data (`abs_err` ≈ 3.5e-8 / 6.0e-8 across all artifacts). ~1 of ~3 traversals per trial.
2. **The convex solve is always started from the same constant weight vector**
   (`freeratesolver.cpp:247` passes the pass-fixed `w`), never from the previous trial's *solved* weights,
   despite adjacent rate vectors having adjacent optima. This is the dominant cost (117 of 134 s on DNA).
   Safe to warm-start **because** every accepted solve is gated on `certifiesTo()`, so the start cannot
   change a certified optimum — only the iteration count.

---

## 11. Root cause, and the process failure

### 11.1 One technical root

> **The rate block maximises a profiled objective `φ(r) = max_w l(w;r)` by forward-differencing it,
> through a noisy inner solve, under a driver with no domain constraint.**

Every symptom in §10 is downstream of that single sentence:

- forward differences of `φ` + inner gap `G` ⇒ gradient error `~G/h` ⇒ **the endpoint depends on the
  search gap**;
- no bound on the induced rate *scale* ⇒ **out-of-domain trials** (§10.1);
- out-of-domain + a penalty return + a probe `derivativeFunk` never clamps ⇒ **~1e10 gradient, frozen
  search** (§10.2).

No amount of tuning gaps, penalties, counters or gate parsers can fix any of these. They are properties of
the driver.

### 11.2 One process root — stated plainly because it shaped the whole record

Five gates were run; **each one found a defect in the measurement rather than in the science.** The
recurring error has a single shape:

> **A number was reported as a finding before it was established to be a property of the thing measured
> rather than of the instrument.**

`moment_out` (a property of `RateFree`'s normaliser), the ladder "better optimum" (a property of the
realise tolerance), "run-length invariance", the 20.7-nat figure — all the same shape. A second-order
version also occurred: a *retraction* was issued on contaminated data and then had to be partly
un-retracted.

**The discipline correction, adopted for step 3:** the acceptance criterion is written down numerically
**before** any code is written, and the result is reported against it verbatim. If it fails, it failed —
it does not get reinterpreted afterwards. This is exactly the Stage-0 kill-gate discipline the plan
already mandates (§3.5); the failure was not following it inside the step-2 sub-loop.

---

## 12. The decided next step (step 3) and its pre-registered gate

### 12.1 Make the objective well-posed, then measure it

**(1) Eliminate the forward-difference on `φ`.** By the envelope theorem on
`L = l − α(1ᵀw − 1) − β(rᵀw − 1)`, at the constrained weight optimum `w*(r)`:

```
dφ/dr_j  =  dl/dr_j |_{w = w*}  −  β · w*_j
```

Two ingredients, with very different availability — **both verified on disk 2026-07-23**:

- **`β`** is *already computed* as the least-squares multiplier `nu` inside `computeNewtonCertificate`
  (`freerateprofile.cpp:~992-1005`, guarded by a `well_conditioned` test) and then **discarded** —
  `ProfileResult` has no multiplier member of any kind, and the header already declares
  `PROFILE_EXPLICIT_DUAL_MULTIPLIERS_UNAVAILABLE` as a known limitation. **One struct field.**
- **`dl/dr_j` at fixed `w*`** requires `∂F_pj/∂r_j`, which **does not exist in the tree API**.
  `computeLikelihoodDerv` is per-branch (`double *df`); the `Mixlen` variants are scalar (`double &df`)
  and mixlen-only; `_pattern_lh_cat` holds per-pattern-per-category *values* with no derivative
  counterpart. A full analytic derivative needs a new kernel ⇒ **Phase-4 scale, out of scope.**

⇒ **Pragmatic route: an F-space finite difference.** Re-extract `F` at `r_j + h`, hold weights at `w*`,
recompute `l` analytically from the extracted columns, and difference. `F` is bounded, positive, and
produced by the tree directly, so (i) no rejection penalty can enter a difference — the ~1e10 mode is
structurally gone — and (ii) **the gradient no longer depends on the inner weight-solve gap at all**,
which is precisely what makes the endpoint gap-dependent today.

**(2) In-domain trust region (§7.2).** Add the missing constraint — a lower bound on the gauge scale
`s(ρ)` (equivalently an upper bound on `max_j r_j`) computed once at setup from the actual branch lengths
and eigenvalues, plus a log-space radius with shrink/expand. Critically, the effective bounds must carry a
**margin against the finite-difference stencil**, since a probe at `x_j·(1 + 1e-4)` is never clamped.

**(3) §7.6 interval acceptance.** `φ(x) ∈ [L(x), L(x) + G_w(x)]`. Accept iff
`L(y) > L(x) + G_w(x) + τ_noise`; reject iff `L(y) + G_w(y) < L(x) − τ_noise`; on overlap tighten both
gaps geometrically and retest; still overlapping at final precision ⇒ numerically null, shrink the radius.
This makes the incumbent **always a previously-evaluated in-domain point**, which converts "the
catastrophic mode is unlikely" into "it cannot occur".

Plus two independently-revertable cost changes (kept separate so they cannot contaminate the correctness
result): move the self-validation likelihood behind a debug flag, and warm-start the convex solve.

### 12.2 Pre-registered acceptance gate — written before implementation

| # | criterion | current value | bar |
|---|---|---|---|
| 1 | **Endpoint invariance to the search gap** across 1e-2 / 1e-5 / 1e-8 | **7.046e-02** relative | **≤ 1e-6** |
| 2 | `domain_rejects` on every cell | 10 (DNA), 19 (AA) | **0** |
| 3 | Fault injection at **every** eval index in a swept range leaves the gain within `τ_L` of the un-injected control | eval 6 loses all 4.198 nats | **all indices within `τ_L = 1e-4`** |
| 4 | No regression in achieved gain | DNA `4.198188321`, AA `2.009270684` | **≥ control − `τ_L`** |
| 5 | Inert-when-off | passes | **byte-identical** |
| 6 | Canonical rate order at the endpoint | **violated on AA** (§10.3) | **nondecreasing** |
| 7 | Cost | 0.42 s/trial DNA | report; ≤2× or explain |

Criterion 1 is the thesis of step 3: *if the endpoint still depends on the search gap, the objective is
still ill-posed and nothing downstream is worth building.*

### 12.3 Explicitly **not** in step 3

`χ_r` and full block metrics; §7.3 support events and continuous insertion pricing; the §7.7 start
portfolio; multi-cycle; the quotient pocket; wiring `FreeRateFitResult`; anything GPU; anything avian.

---

## 13. Position against the plan's phase gates

| phase | status |
|---|---|
| **Phase 0A** — evidence instrumentation | **Done.** Provenance, typed exits, fixed-tree injection, per-block residual hooks. |
| **Phase 0B** — CPU fixed-support oracles | **~80%.** Both profilers, both certificates, scipy cross-check on a 21-case synthetic battery, writeback-confirmed reconstruction on 8/8 real cells, envelope containment over 607 archived endpoints. **Open:** clause 3b (envelope *widening* re-fits) and clause 4 (fitted pattern probabilities agree when weights are non-unique). |
| **Stage 0** — the diagnosis | **Complete**, repeatedly red-teamed. Governing row of §3.5: *"weight block certified but rate residual is material"* ⇒ the §7.1 alternating cycle **with** the §7.2 ordered rate trust region as co-equal work, not a follow-on. |
| **Phase 1** — CPU pure-`+R` fixed-tree solver | **In progress.** Step 1 (expose machinery, proven inert) and step 2 (measure the re-profiled objective) **done**. **Step 3 is the actual fix and is not built.** |
| **Phases 2–7** | Not started. |

**Honest scope caveat carried from the plan:** this increment gates on the *cheap* cells (DNA-100K,
AA-100K). Avian is report-only and expected to return `MAXITER`. The increment therefore does **not** meet
§12's Phase-1 gate as literally written ("every fixed-tree hard/easy cell terminates without a cap") and
says so rather than redefining the gate.

---

## 14. What an outside reviewer could most usefully challenge

1. **Is the F-space finite difference actually sound here?** It gives `dl/dr_j` at fixed `w*`; the `β·w*_j`
   correction supplies the constraint term. Does the gauge pin in `writeRates` (which renormalises *all*
   rates by `s`, so a one-coordinate perturbation moves every column) break the "hold `w` fixed" step? The
   intended answer is to difference in an un-gauged literal rate coordinate and let the profiler enforce
   the mean — **this has not been validated.**
2. **Is a trust region sufficient, or is `dfpmin` the wrong driver entirely?** A projected-gradient or
   genuine trust-region method owns its own step acceptance; wrapping `dfpmin` may be fighting it.
3. **§10.1's overflow mechanism is inferred, not instrumented.** One `max_j r_j` dump at the rejection
   would settle it.
4. **The flat ridge (§8.3) is one measured direction between two endpoints on one cell** — consistent with
   the `+R` identifiability literature (Morel et al. 2021, cited in the plan §17), but not a curvature map.
5. **`gap8` also ran 280 trials vs ~170** for the loose rungs, so "tighter gap" and "searched further" are
   not cleanly separated. A fixed-trial-count comparison at both gaps would isolate it.
6. **Part I's C1/C2 recommendations were never formally applied to the plan document.** C2 in particular
   (make the failure criterion the *residual*, not the *cap*) remains a live correction.

---

## 15. Step-3 implementation spec (written, reviewed, not yet built)

An independent architectural pass produced the spec below and **corrected the design in four places**.
Recorded here because the corrections are the most useful part.

### 15.1 Design corrections found before any code was written

**(a) "The F-space gradient is gap-free" is only half true.** The difference itself is gap-free, but it is
evaluated *at* `ŵ` and uses `β(ŵ)`, both from an inexact solve. A gap bounds `l* − l(w)`, **not**
`‖ŵ − w*‖`. φ's *value* error is second order in `‖ŵ − w*‖`; its *gradient* error is **first** order. So
gap-dependence drops from `O(G_w/h) ≈ 1e4·G_w` to `O(√(2G_w/λ_min))` — roughly 200× better and,
importantly, *continuous* rather than penalty-discontinuous — but **not zero**. This is why the gate bar
is `τ_L` and not exact invariance.

**(b) The warm start as I stated it is a no-op.** `w_prev` satisfies `r_prev'w_prev = 1`, not
`r_cur'w_prev = 1`, so the profiler refuses it as infeasible and silently falls back to the vertex
centroid — buying nothing while inflating `refused_start_count`. It needs a 2×2 affine projection onto
`{1'w = 1, r_cur'w = 1}` plus a non-negativity blend toward the known-feasible pass vector. Falsifiable
prediction: implement it naively and `warm_start_used ≈ 0`.

**(c) β is not "free".** `computeNewtonCertificate` returns *before* the `mu`/`nu` block at three points,
including `min_multiplicity < 1.0` — which is real under bootstrap/ascertainment weighting, where β simply
does not exist. A `dφ/dm` fallback (two convex solves on already-extracted columns, zero tree work) is
therefore **required**, not contingency.

**(d) The most likely implementation error is invisible at the seed.** The gauge chain rule uses the
oracle's *fixed* `w`; the envelope term uses the profiler's `w*`. **At the seed these coincide**, so a
gradient verified only at the seed passes with the wrong vector everywhere else. Verification must run at
the **endpoint** too.

Two further structural findings: `F` is **column-separable in `r`** (category `c`'s partials use
`len = rate(c)·branch` only), so **one** extraction with every rate perturbed simultaneously yields every
`∂F_pj/∂r_j` — the gradient costs one extra extraction, not `ndim`. And `Optimization::derivativeFunk` is
**virtual**, so overriding it deletes the forward-difference probes entirely without touching
`utils/optimization.cpp`.

### 15.2 Sequence (each independently buildable, own kill switch, all-off ⇒ bit-exact step 2)

| step | switch | content |
|---|---|---|
| 3a | — (additive) | Export `β`/`α` + validity + conditioning from the profiler |
| 3b | `IQ_FR_EXTRACT_VALIDATE` | Self-validation likelihood off the hot path, **sampled** every 64 trials and *asserted* |
| 3c | `IQ_FR_WARMSTART` | Warm start **with** the affine projection of 15.1(b) |
| 3d | `IQ_FR_GRAD_ANALYTIC` | Override `derivativeFunk`: envelope gradient, one extraction, gauge chain rule |
| 3e | `IQ_FR_TR` | Safe scale bound `s_min` + log-space trust region |
| 3f | `IQ_FR_ACCEPT` | §7.6 interval acceptance |

### 15.3 The pre-registered gate (numbers fixed *before* implementation)

| gate | criterion | current | bar |
|---|---|---|---|
| **G0** | all switches off reproduces step 2 **as printed strings** | — | `4.198188321` / `2.009270684`, 280/618 solves, 10/19 rejects |
| **G1** | **endpoint invariance to the search gap** | Δ lnL `6.093808e-03`; endpoint `7.046e-02` | **≤ 1e-4 nats** (fail > 1e-3); endpoint ≤ 1e-3, or `FLAT_RIDGE` if ΔlnL ≤ τ_L |
| **G2** | `domain_rejects == 0` **non-vacuously** | 10 / 19 | 0 with TR on **and** 10/19 with TR off **and** `safe_bound_binding == 1` |
| **G3** | fault injection swept **2…40** + `{2,3,8,12,20}` real | eval 6 loses all 4.198 | every index: gain ≥ 0, `gain_suspect == 0`, within τ_L of control |
| **G4** | no regression | — | ≥ control − τ_L, `base_lnl` byte-identical |
| **G5** | new branches provably reached | — | `forward_diff_pattern_evals == 0`, `grad_extracts == grad_calls`, `tr_shrinks > 0` **and** `tr_expands > 0`, forced-overlap ⇒ `overlap_rounds > 0` and `null_events > 0` |
| **G6** | cost | 280 / 618 solves | **≤ 168 / 371** (0.6×), `box_face_evals ≤ 30%` |
| **G7** | inert-when-off | passes | byte-identical, **cross-binary** against frozen `e8dac3eb` |
| **G8** | profiler unchanged | — | 21-case battery hash equality; `\|dφ/dm − β\| ≤ 1e-5`; analytic vs 1e-13 central difference ≤ 1e-2 **at seed and endpoint** |

**G1 is the thesis.** If the endpoint still depends on the search gap, the objective is still ill-posed
and nothing downstream is worth building.

Two anti-vacuity clauses are load-bearing and were added by the review: **G2** requires the trust-region-off
arm to still reject 10/19 and the safe bound to actually bind, because `domain_rejects == 0` is equally
consistent with "the box works" and "the box is inert"; **G5** requires each new branch to be *observed*,
because several (`overlap`, `null`, β-fallback) never execute on healthy cells and would otherwise ship
untested.

### 15.4 Known risk carried into step 3

`dfpmin` takes the Hessian as *input* and never writes it back, so **every outer trust-region iteration
restarts from the identity** — affordable only because 3d makes an iteration ~`ndim`× cheaper. G6's
`trials_attempted` bar is the tripwire: if it fails, the outer-loop shape is wrong and the box must be
adjusted in place inside one long `dfpmin` call, which needs a `dfpmin` change and a separate gate.
