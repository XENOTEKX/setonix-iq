# Full-GPU ModelFinder — certified FreeRate convergence plan

**Status:** Authoritative design, 2026-07-21. Correctness and stability precede speed.

**Source baseline:** /scratch/rc29/as1708/iqtree3-jolt-merge, branch jolt-gpu-merge, commit ccabc96e111b08460e1e5e3acf55ac281624987e. The existing build-merge/iqtree3 binary predates that source state and must not be treated as a same-commit executable.

**Historical plan:** MODELFINDER-FULL-GPU-PLAN-LEGACY-2026-07-16.md preserves the superseded performance and coverage plan.

---

## Implementation progress (updated 2026-07-22)

Implementation tree: `/scratch/rc29/as1708/iqtree3-freerate-profile`, branch `freerate-profile-impl`,
HEAD `79727c2e`, pushed to `setonix-iq` GitHub. All diagnostics are inert unless `IQ_FR_ATTRIB` (and, for
the joint arm, `IQ_FR_JOINT`) is set, so an ordinary run is byte-for-byte unchanged. This section records
status against the phase gates in §12 and is the honest running log; the design body below is unchanged.

**Phase 0A — evidence instrumentation: DONE.** Provenance/state capture, typed exit reasons, fixed-tree
and full-state injection, per-block residual hooks. The Stage-0 kill gate (§3.5) passed and was re-run
under two adversarial audits.

**Phase 0B — CPU fixed-support oracles: ~80%.** Delivered and gated: quotient and literal active-set
weight profilers (`model/freerateprofile.{h,cpp}`); the exact convex gap by **two** certificates — the
Frank–Wolfe gap AND a self-concordant Newton-decrement bound with a dual-feasibility (globality) test;
an independent scipy constrained-solver cross-check on a 21-case synthetic battery (k=2…10, duplicate,
collision, zero-weight, boundary, infeasible); the unweighted common-scale component reconstruction
(`model/freerateeval.cpp`), confirmed against production by IQ-TREE's own writeback (write profiled
weights, recompute, compare) to the reconstruction-error floor on 8/8 real cells; the archived-endpoint
envelope containment (607 endpoints, min q = 1.67e-5 against the 1e-7 floor); and the Stage-0 residual
attribution. **Newton is not optional on real data:** on two avian cells the FW gap fails τ_w by 549× and
275,000× while the Newton bound certifies — without it §11.2 would mark a correct fit INCOMPLETE.
**Open:** clause 3b (envelope *widening* re-fits — tests §5.2's feasible-interval design) and clause 4
(fitted pattern probabilities agree when weights are non-unique). Neither is ceremonial.

**Stage 0 — the diagnosis (the load-bearing scientific output): COMPLETE and repeatedly red-teamed.**
1. **The avian cells never converge — the "400-cap plateau" is a truncation artefact.** They hit the
   *outer* parameter-round cap (`num_param_iterations`, default 100; the historical "400" is a separate
   inner brlen cap) still climbing. Proven: raising the cap 100→500 on avian R8 collapsed the measured
   weight-block residual from **20.745 nats to 1.161 (−94%)**. The 20.7-nat figure must never be quoted
   as a weight-block residual. A companion R6 cell instead held its residual across +604 nats of
   progress, so run-length invariance is cell-specific, not a law — an over-generalisation, now retracted.
2. **The optimiser declares stationarity with material gain unrecovered.** At IQ-TREE's *own* stopping
   threshold (`modelEps = 0.010` nats), each of two independent blocks separately holds **1.36–1.49 nats
   ≈ 136–149× that threshold**, on the two cells that genuinely converged (DNA-100K, AA-100K R8).
3. **Its stop rule structurally cannot work.** The avian R8 trajectory is non-monotone: ~2 nats/round for
   150 rounds, then **~159 nats in six rounds**, then decay. `modelEps` stops on the most recent round's
   gain and therefore cannot distinguish "near-stationary" from "about to find 159 nats" — a
   mechanism-level argument for §8.2's typed residual certificate, independent of any residual number.
4. **"Weights dominate" is RETRACTED** (it was an artefact of a truncated rate control — 2–3 BFGS
   iterations on a 5–7-dim problem). Corrected, the two one-block gains are comparable (0.94–1.37 on most
   cells) and on the best-optimised avian cell the **rate** block leads 8:1.
5. **The joint measurement (conditional second-block gain, `surv_r`/`surv_w`): the blocks overlap, and
   the overlap grows with cell difficulty.** DNA (easy) `surv` 0.68–0.89 — largely independent; avian
   (hard) 0.13–0.37 — largely the same nats. But on **every** cell, after committing one block the other
   still holds a material gain (avian `G_w|r`=2.72, `G_r|w`=6.63), and the one-block gains are **not
   additive** (joint ≈27 nats vs the 38.5-nat naive sum).

⇒ **§3.5 reading, refined by measurement:** the "weight block certified but rate residual is material"
row governs. Both blocks are load-bearing on every cell, so the fix is the §7.1 alternating block cycle
with the §7.2 ordered rate trust-region **co-equal, not a follow-on**; a weight-only solver addresses at
most half the recoverable slack. There is real, directional path dependence on avian (weight-first beats
rate-first by 6.9 nats), consistent with §7.1's weights-first ordering.

**Phase 1 — CPU pure-+R fixed-tree solver: IN PROGRESS (this is the actual fix).**

*Scope correction found in planning, before any code was written.* The first increment builds the
**literal** mass-and-mean pocket (§5.4, k−2 weight DOF) at genuinely fixed branches, **not** the quotient
pocket (§5.1, k−1 DOF). Reason: schema v1's `certifiedForSelection` admits only
`weight_formulation = LITERAL_MASS_MEAN` with `branch_mode = BRLEN_FIX`, so a quotient solver would emit a
result that **cannot pass its own gate** without a schema bump plus a certificate extension for the
`m_L/m_U` interval and its bound-activity semantics. §5.2 agrees on the mathematics: at genuinely fixed
branches the gauge transform `b' = mb` is illegal, so the mean constraint stops being a gauge and becomes
a real constraint — which *is* the literal geometry. The quotient pocket is a later increment; the
endpoint of this one must not be described as the production optimum.

*Cost, stated honestly.* §7.1 step 3 re-profiles weights at **every rate trial**, so one BFGS iteration at
k=8 costs ~9 extractions plus 9 convex solves. §14's component-column cache does **not** rescue this: the
cached columns are keyed on the structural state `{topology, branches, Q, rates}`, and a rate trial
changes the rates, so F genuinely changes. The real avian blocker is the convex solver's degenerate tail
(10,000 iterations at ~20 min on one cell) plus over-solving every trial at the default `1e-8` gap when
§7.6 explicitly permits a loose early forcing tolerance. ⇒ **the first increment gates on the cheap cells
(DNA-100K, AA-100K, synthetic R2–R10); avian is report-only and expected to return `MAXITER`.** The
increment therefore does *not* meet §12's Phase-1 gate as literally written ("every fixed-tree hard/easy
cell terminates without a cap"), and says so rather than redefining the gate.

*Status ceiling.* With §7.6 profile-interval acceptance implemented (non-negotiable — without it the
solver is a diagnostic and has moved the wobble into profile noise rather than fixing it), interior points
can legitimately reach `LOCAL_STATIONARY_CERTIFIED`. Deferred: continuous insertion pricing ⇒ every
empty-slot/boundary cell returns `BOUNDARY_DIRECTIONALLY_TESTED` and is reported unresolved (§7.3 forbids
the shortcuts: *"dense scanning plus local Brent searches is not a bound"*); the true trust region (BFGS-
in-a-box adjudicated at the outer level stands in, which costs robustness, not certificate validity);
most of the §7.7 start portfolio; and `FALLBACK_CERTIFIED`, which would be a provenance lie with no GPU
arm to fall back from.

*Step-by-step status.* **Step 1** (expose the rate machinery; no behaviour change) is gated and confirmed
inert two independent ways. **Step 2** (build the re-profiled rate objective alone and measure what it
costs, before the outer machinery is layered on it) has now run three times, and each run killed a defect
that would otherwise have been buried under four more layers:

- *Run 1* — the pass **lost 95 nats**. Cause: the weight problem's mean constraint was being set from the
  *live* proportions, which the previous trial had overwritten, so the feasible set at trial *n* was
  `{w : rₙ'w = rₙ·p₍ₙ₋₁₎}`. The constraint level moved with history ⇒ the objective was **not a function
  of its argument**, and the realised mean drifted off the §4.1 contract, re-admitting the global-scale
  direction §7.2 assigns to the branch block. Fixed by pinning `target_moment = 1.0`.
- *Run 2 (gate 174459875)* — contract restored (`moment_out = 1.000000000000` on 4/4 cells) but
  **rate_gain = 0 on 4/4**: the pass aborted, blaming `refused-start`. **That reason was wrong.** One
  weight solve per cell returned a hard failure; those exits return early with `supplied_start_used`
  still false; and the start check sat *in front of* the exit-reason check — so a failure of the weight
  *problem* was reported as a failure of its *start*. Fixed by classifying the exit reason first and
  **rejecting the trial point** (φ is genuinely undefined there) instead of killing the pass.
- *Run 3 (gate 174461894)* — **the increment's first real number.** On DNA-100K R8 the re-profiled rate
  block recovers **`rate_gain = 4.192102582` nats**, realised, over 176 weight solves, after a
  `w1_gain = 1.482204356` weight commit (which reproduces the known `G_w` exactly). The fixed-weight
  one-block arm predicted `G_r|w ≈ 0.89 × 1.489 ≈ 1.33`, so **re-profiling recovers ≈3.2× what the
  fixed-weight arm could see** — direct evidence §7.1 step 3 is load-bearing, not an expensive nicety.
  Cycle total 5.674 nats on a cell IQ-TREE declared converged at `modelEps = 0.010`: **567× its own
  stopping threshold.**
- *Run 4 (gate 174464179)* — in flight, re-measuring the above through a corrected instrument.

**Cost, measured.** DNA: 176 trials, 81.0 s, 0.40 s/trial, **convex-solve-bound** (70.8 s of 81.0).
AA: hit the 200-trial budget and aborted, **extraction-bound** (57.4 s extract vs 55.0 s solve). The cost
profile therefore *flips with data type*, which changes what §14's column cache buys on each. And the
forcing ladder — valid for cost, never for gain — shows a **10⁶× looser inner gap saves only ~15%**
(81→69 s): **the inner gap is not the cost driver, the trial count is**, so §7.6's loose forcing schedule
is not the cost lever it was expected to be.

**The §7.2 trust region is missing, and that is now measured rather than assumed.** The *un-injected*
DNA cell rejects **10 trial points as out-of-domain** (AA: 17), the first at eval 8 — the first
line-search trial. The optimiser walks out of the declared ratio box on ordinary real cells.

**Five reporting defects found by adversarial review of run 3, each verified on disk, all fixed in
`10694646`.** ① `moment_out` was a **tautology** — computed after the state restore, where `RateFree`
renormalises `Σ w_j r_j` to 1, so it printed `1.000000000000` for any input; the tell was that all four
*aborted* cells printed it exactly. Now captured from the live fitted state (a budget-aborted pass reads
`0.999999999448`), and **every claim previously made from it is withdrawn**. ② A rejection *at the seed*
returned a **reward**: the penalty fell back to a flat `1e6` against `|lnL| = 5.7e6`, scoring better than
every feasible point. ③ The penalty base moved with the pass, re-introducing the "φ is not a function of
its argument" defect *inside the fix for it*. ④ **Uncertified solves were accepted as φ(r)** —
`MAX_ITERATIONS`/`NUMERICAL_STALL` fell through and were written back with the gap never consulted; now
gated on `certifiesTo()`. ⑤ The not-realised path measured against a mismatched `(r, w)` pair, which is
how `dna_fault12` produced `rate_gain = 0.000, w2_gain = 4.198` — **the "weights dominate" signature this
track has already retracted twice, manufactured on demand.**

**A guard's blind spot, now covered.** `endpoint_shortfall` detects only "walked past a better point". A
rejected FD probe poisons a gradient component by `≈1e10` and the search then *never moves*, so shortfall
is zero **by construction** — on real data `dna_fault6` reported `rate_monotone=1, shortfall=0.005` while
the un-injected control on the same cell recovered 4.19 nats. The new `gain_suspect` flag covers it and is
proven to reach its branch by injection.

**The step-2 result that matters, and it is a negative one.** Fault injection (a forced rejection at a
chosen evaluation index, swept 2→13 on `example.phy` GTR+F+R4, `ndim=3`) shows the rejection fix is a
**workaround, not a repair** — and the mechanism is now *verified against the optimiser source*, not
inferred. `Optimization::derivativeFunk` costs `1 + ndim` evaluations — one `fx` at `p`, then `ndim`
forward-difference probes, combined as `dfx[dim] = (dfx[dim] − fx)/h[dim]` — and `dfpmin` calls it once
before its loop and once per iteration. With the probe's seed call as eval 1 that predicts the measured
pattern exactly:

| fault eval | role in dfpmin | `rate_gain` | monotone |
|---|---|---|---|
| 2 | `fx` base of the 1st gradient | **−0.268** | 0 |
| 3, 4, 5 | the three FD probes | 0.000 (stalls) | 1 |
| 6–11 | line search, iteration 1 | +0.0377 (full) | 1 |
| 12 | `fx` base of the 2nd gradient | **−0.425** | 0 |
| 13 | FD probe of the 2nd gradient | +0.0014 (4%) | 1 |

The two catastrophic indices are exactly the `fx` bases: `fx` enters **every** gradient component, so one
poisoned evaluation corrupts all `ndim` at once, at magnitude `≈1e6/(1e-4·|x|) ≈ 1e10` (`ERROR_X = 1e-4`).
A single probe corrupts one component and merely stalls the pass. Line-search evaluations backtrack
correctly. ⇒ **where a rejection lands decides whether the answer is right**, and dfpmin can exit *below
the best point it visited* while still reporting a clean realised endpoint. §7.2's bounded trust region
(keep trials **in** the domain) and §7.6 acceptance (never publish a point worse than the incumbent) are
therefore **measured requirements of step 3**, not design tidiness. The probe now emits `rate_monotone`
and `endpoint_shortfall` rather than a single gain.

*And the mechanism dictates the step-3 design, which is why measuring it was worth a run.* The
catastrophic case is a rejection of `fx` — the value **at the current iterate**. Under §7.6 acceptance the
iterate is always a previously-accepted point that was evaluated successfully, so **`fx` can never be
rejected** and the catastrophic mode becomes structurally impossible rather than merely unlikely. The FD
probes sit at `p + h·e_j` with `h = 1e-4·|p_j|`, so a trust region that keeps the iterate strictly
interior to the ratio box also keeps every probe in-domain, removing the stall mode. **Neither needs a
larger penalty, a smaller penalty, or any penalty tuning** — which matters, because a penalty is a free
parameter and this plan does not get to add one silently.

*Two claims withdrawn from the step-2 write-up.* (a) "Zero capped ⇒ no degenerate tail" — the counter only
caught `MAX_ITERATIONS`, while the degenerate tail exits `NUMERICAL_STALL`, so it was blind to the exact
failure it existed to detect. (b) "3 `NUMERICAL_STALL` solves ⇒ **this** is the degenerate tail" — every
finite gap in that run was ≤ 1.77e-15 against a 1e-8 forcing gap, i.e. **converged solves tripping the
negative-gap guard at the arithmetic floor**. `max_gap_stall`/`stall_below_forcing` now decide this per
run instead of by assertion. The avian tail remains real but **unmeasured** by this gate. Separately, the
forcing ladder is **cost-only** and cannot compare gains: a looser inner gap `g` changes the objective and
injects forward-difference gradient error of order `g/h` (`h = 1e-4·|x|`) — at `g=1e-2` that is ~1e3–1e5
nats per unit ratio against a 1e-3 target, so the loose rungs differentiate noise.

**Phases 2–7 (branches/Q, +I+R, GPU evaluator, full ModelFinder, rollout, partitions): not started.**

**Honesty log — claims retracted after audit, all verified on disk:** the 20.7-nat avian headline
(truncation); "weights dominate" (truncated control); run-length invariance as a class property (n=2,
sign-flipped); an "11–14-nat BIC margin" figure (dropped as unlocatable); "zero capped ⇒ no degenerate
tail" (the counter was blind to `NUMERICAL_STALL`); "these stalls **are** the degenerate tail" (they were
converged solves at the arithmetic floor); "the forcing ladder refutes over-solving" (the ladder cannot
compare gains at all); and **"`moment_out = 1.0` proves the fit holds the §4.1 contract"** (the field was
computed after the state restore and was 1 by construction — it was quoted twice before the tell, four
*aborted* cells all printing it exactly, was noticed). Separately, the reporting/gate layers have produced a wrong verdict several
times (a string-equality verdict column, a `grep -v` selecting the wrong line, a survival ratio printing a
refusal as a conclusion, a stale gate reference, and a cost column that printed **blank** for four runs
because the emitter had been renamed and the gate's grep never was) while **the raw measurements survived
every audit** — the standing rule is to read the bytes, not the verdict line.

**Two rules this increment added, both from failures above.** ① *A test must be proven to reach the
branch it tests.* The out-of-domain path fires only when a trial makes the weight problem insoluble,
which on a cheap cell never happens — so the cheap-cell gate exercised none of it and passed blind. The
gate now injects a fault and **fails if the injection did not fire**. ② *A gate must assert its fields
exist.* Every extracted field is now checked non-empty, because a silently blank column is
indistinguishable from a healthy one in the verdict text.

---

## 0. Decision

Do not extend JOLT's diagonal secants, HARP, OPG/Fisher, one-step EM, floors, or tolerance tuning.

Build a host-controlled, blockwise FreeRate profile solver with these properties:

1. The first production primitive is an exact convex weight profile for the current fixed k support points.
2. For unpartitioned pure +R under BRLEN_OPTIMIZE, the preferred inner coordinates hold every rate × branch kernel fixed, optimize all k−1 weight degrees of freedom, and restore the unit-mean gauge exactly without clamping.
3. A literal-rate, literal-branch mass-and-mean profile is the independent reference and the production fallback whenever a quotient moment bound is active.
4. Continuous rates, branches, Q, frequencies, and pinv remain nonconvex outer blocks. They use separate safeguarded trust regions and exact re-profiling, not one shared damping parameter.
5. Success means LOCAL_STATIONARY_CERTIFIED under a declared deterministic start policy. It never means a globally proved phylogenetic MLE.
6. A cap, reject stall, arithmetic mismatch, or unresolved basin is a typed failure. ModelFinder must not score, prune from, or silently discard that candidate.
7. The GPU initially evaluates likelihood components and derivatives. The small constrained optimizer stays on the host until the CPU reference is proven.
8. Dense-grid or unrestricted rate-measure fitting is diagnostic and seed-generating only. It is not +Rk, has no +Rk BIC, and cannot replace the R2–R10 ladder.

This direction is conditional on Stage 0. If the fixed avian cell reaches a certified stationary point by increasing only the iteration budget, the correct first repair is adaptive budgeting plus a real certificate. If the terminal weight residual is negligible, weight profiling is not the causal repair and work redirects to the rate, branch, Q, or globalization block identified by the residual audit.

---

## 1. What the evidence actually establishes

### 1.1 Trusted findings

| Finding | Verified evidence | Design consequence |
|---|---|---|
| +R is the dominant optimization cost | Job devusent1_174126193: 48 +R candidates consumed 11,021 of 12,122 JOLT joint iterations, 90.9%; 21 cap-bound fits consumed 8,421 iterations | Fixing FreeRate convergence is both a correctness and eventual runtime priority |
| The clean avian tolerance ladder did not wobble | tolladder_174328851, fixed GTR+R6: six arms nominally labeled JOLT_IR_TOL 1e−2 through 1e−7 all completed exactly 400 accepted iterations, conv=0, final lnL −11216886.230053; iteration 400 still gained 0.270 nat | The current hard fact is truncation at the cap; tolerance is not reached |
| The GPU likelihood kernel is not the leading suspect | Relevant terminal GPU likelihoods match fresh CPU reevaluation to approximately machine precision | Preserve evaluator parity as a gate, but investigate the optimizer and its state transitions first |
| Current convergence reporting is insufficient | HARP conv=1 is a single accepted gain below tolerance; there is no KKT, projected-gradient, restart, or block residual certificate | Replace boolean convergence with typed status and measured residuals |
| OPG did not repair the hard cell | Avian R6 Phase 2 lost 162.875 nat and capped; Phase 3 lost 51.885 nat and capped | Empirical Fisher/OPG is not the observed profile curvature and is retired as the production direction |
| HARP did not transfer | A one-shot direction lost 23,413.927 nat; a tuned tau helped one R6 cell but neighboring k/tau combinations regressed, including R8 by 705.617 nat | No dataset- or k-specific damping policy |
| One/few EM weight updates did not repair the path | jolremA_174342520 widened the avian R6 endpoint spread from 216.760 to 535.313 nat and made its best endpoint 185.592 nat worse | One EM update is not an exact observed-likelihood weight solve and is not evidence for or against exact profiling |
| Stock +I is not spurious | Stock IQ-TREE v3.1.2 job 174338165 selected +I in 9 of 9 arms | Pure +R is only Milestone 1; +I+R is mandatory before full avian/ModelFinder promotion |
| Full-MF tolerance and seed comparisons are confounded | Candidate admission changes across mf-epsilon; seeds change the PARS tree; ninit/optalg changes the fitted family | Diagnose fixed model, tree, start state, work, and candidate manifest before selection |

### 1.2 Corrections that this plan adopts

- The earlier “six tolerances, 229 iterations, 323-nat spread” account mixed HARP tau policies with the pure tolerance ladder. It is invalid as a measure of ordinary tolerance noise.
- The exact “32/32 cap” census is not reproducible from the named files. The defensible statement is that cap exhaustion is near-universal in that batch.
- The HARP R6 and R8 endpoints used different tau policies and weak stopping. They show only that one reachable R8 point beats the tested R6 points. They do not establish that the true avian BIC optimum is at least R8.
- Tighter stock mf-epsilon selected a lower k, but the candidate sets differed and tight arms often never fitted R6. This is not a causal proof that underoptimization inflates k.
- Custom CPU --no-jolt is not pristine stock IQ-TREE and cannot prove that the GPU alone causes the instability.
- OPG lambda-min was printed before optimization and describes a warm seed, not the terminal fit.
- --thread-model fails to evaluate R3+ in the observed path and must not be used.
- CTF approximate ranking changed real-data outcomes and its starting-topology explanation was refuted. It remains outside this correctness plan.

JOLT_IR_TOL, ModelFinder --mf-epsilon, and IQ-TREE -eps control different layers. They must never be plotted, compared, or described as one tolerance. The tolladder_174328851 binary predates the self-describing tolerance banner, so its six JOLT_IR_TOL settings are nominal script/source labels corroborated by the easy-control trajectory response, not banner-proven runtime values.

### 1.3 Still unknown

- The globally optimal avian R6, R8, or +I+R likelihood.
- The globally BIC-optimal avian k.
- Whether the fixed avian JOLT trajectory eventually converges, crawls indefinitely, cycles, or reaches a local point that depends on its start.
- Which block—weights, rate locations, global scale, individual branches, Q/frequencies, or globalization—dominates the iteration-400 residual.
- Whether labeled parameter variation represents different fitted mixtures or only equivalent zero/colliding support representations.

No implementation or result may assume answers to these questions.

### 1.4 Raw evidence index

The following raw artifacts, rather than WORK-LOG summaries, are the evidence of record:

- /scratch/rc29/as1708/gems-verify/tolladder_174328851 — pure fixed-cell tolerance ladder and HARP-policy arms. The six a1 avian runs share alignment MD5 prefix 24ed6e1e, PARS-tree MD5 prefix 324748af, and binary MD5 prefix 8f8ce05e.
- /scratch/rc29/as1708/gems-verify/devusent1_174126193/dna1m_nt1_off.console — +R iteration-cost census.
- /scratch/rc29/as1708/gems-verify/jolremA_174342520 — exact recorded one-step EM/REM multistart comparison.
- /scratch/rc29/as1708/gems-verify/harpws1_174254673, harpws1_174266861, and harpws15_174323861 — HARP one-shot and tuned-policy evidence.
- /scratch/rc29/as1708/gems-verify/opgp2_174158000, opgp2_174158208, and opgp3_174159336 — OPG Phase-2/3 outcomes.
- /scratch/rc29/as1708/gems-verify/avirtolrf_174142647 — end-to-end RF comparison, retained only with its ModelFinder/harness confounds.
- /scratch/rc29/as1708/gems-verify/stockup_174338165[0-2].gadi-pbs — pristine stock controls and candidate-list evidence.
- /scratch/rc29/as1708/iqtree3-jolt-merge at ccabc96e111b08460e1e5e3acf55ac281624987e — canonical behavior source for this plan.

research/WORK-LOG.md and the earlier design documents remain useful chronology, but any disagreement is resolved in favor of raw console, checkpoint, tree, script, binary, and source evidence.

---

## 2. Required scientific claim

The deliverable is not “global MLE for every +Rk model.” Fixed-k continuous mixtures, branch lengths, substitution parameters, and topology are nonconvex.

The defensible production claim is:

> For every requested +Rk candidate, the exact fixed-support weight block is globally solved to a measured likelihood gap; all supported outer blocks meet directional stationarity tests; the result is reproducible under the declared deterministic start portfolio; and an unresolved fit is reported rather than silently ranked.

The operational benefit is permanent even when a difficult candidate remains nonconvex: the 400-step cap can no longer masquerade as a usable optimum.

---

## 3. Stage 0 — falsify the diagnosis before redesign

### 3.1 Provenance and state capture

Add a versioned, immutable FreeRate state record containing:

- source commit, binary SHA-256, compiler/CUDA flags, dirty-state indicator, solver schema, and runtime engagement marker;
- alignment path and checksum, pattern count, topology string and checksum, starting checkpoint checksum, model name, k, branch mode, and all fixed/free flags;
- all rates, weights, branches, Q/frequencies, pinv, bounds, tile count, device model, driver/runtime versions, thread count, and effective tolerance/cap values;
- total likelihood, per-pattern likelihood digest, accepted/rejected step counts, evaluation counts, and optimizer status.

Capture the full state at the seed, every accepted iteration for the diagnostic run, iteration 400, and the terminal point. The trace must distinguish 400 completed iterations from the existing terminal loop counter of 401.

Trial evaluations must be side-effect-free. Current JOLT records the accepted trial likelihood before gaugeFix, whose branch clamp can change the point. Therefore every trace point must be reevaluated at the exact serialized post-gauge parameters. Record the pre-gauge likelihood, post-gauge likelihood, their delta, and every would-clamp event; residual attribution uses only the post-gauge score/state pair.

### 3.2 Fixed-objective cells

Use exactly one injected tree and full parameter checkpoint per cell:

- avian GTR+F+R6;
- avian GTR+F+R8;
- avian GTR+F+I+R6;
- the existing determinate DNA-100K and AA-100K controls;
- later, the real high-k CAT_100S93F amino-acid alignment.

Pin JOLT_NTILE=1 until tiled reduction invariance passes. Do not use --thread-model. A random seed is metadata only; it must not regenerate the topology or starting parameters.

### 3.3 Cap and restart experiment

At one fixed inner tolerance:

1. Run fresh starts with caps 400, 800, 1,600, 3,200, and 25,600.
2. Verify that every fresh longer run has a state-for-state identical first 400 trajectory, or explain and fix the provenance difference.
3. Separately resume the exact iteration-400 checkpoint and prove that continuation matches the corresponding fresh longer run.
4. Reevaluate all terminal states on CPU.
5. Restart each endpoint under tighter final conditions and from deterministic small rate/weight perturbations.

Record per accepted step:

- likelihood and certified/estimated gain;
- weight, rate, global-scale, branch, and Q residuals;
- trust/damping state;
- accepted and rejected trial counts;
- projected step, parameter displacement, active bounds, support collisions, and zero weights;
- CPU/GPU arithmetic delta at selected checkpoints.

### 3.4 Residual attribution

At the seed, iteration 400, and the longest terminal state, measure:

1. The quotient fixed-kernel weight profile and its exact convex gap.
2. The literal-rate/literal-branch mass-and-mean weight profile and its exact convex gap.
3. A rate-location projected residual and best certified one-block gain.
4. The global branch-scale residual and gain.
5. Individual branch residuals and best coordinate/block gain.
6. Q and frequency residuals and best one-block gain.
7. Same-state CPU/GPU component-likelihood, total-likelihood, and derivative parity.

The two weight profiles answer different coordinate questions; their one-block gains need not agree.

### 3.5 Binding decision tree

| Stage-0 result | Action |
|---|---|
| GPU/CPU value, component, or gradient mismatch | Stop optimizer work; repair evaluator/scaling first |
| A larger cap reaches one restart-stable stationary equivalence class | Advance adaptive budgeting plus the new certificate through Phases 1–6; this fixed-cell result alone does not authorize shipping |
| Material quotient or absolute weight gap at iteration 400/terminal | Authorize the fixed-support profile solver and replay from iteration 400 |
| Weight block certified but rate residual is material | Build the ordered rate trust-region; do not claim weights caused the plateau |
| Global scale or branch residual dominates | Repair branch/gauge block scheduling |
| Q/frequency residual dominates | Keep Q/frequencies in a separate CPU block and repair that block |
| Repeated trial rejection precedes exit | Treat as a globalization defect; return REJECT_STALL and use block-specific trust regions |
| All block residuals are small but starts disagree | Treat as multimodality; use deterministic continuation, splits, swaps, and multistarts |
| Labeled parameters drift but fitted pattern probabilities and lnL agree | Canonicalize the equivalence class; do not chase label equality |
| Any residual remains above its declared tolerance at a cap | Return MAXITER; never use the candidate for scoring or pruning |

Stage 0 is a kill gate. The rest of this document is authorized only for the block that the measurements implicate.

---

## 4. Mathematical model and boundary semantics

### 4.1 Public +Rk contract

For k positive-rate categories:

\[
G=\sum_{j=1}^{k} w_j\delta_{r_j},
\qquad
w_j\ge 0,
\qquad
\sum_j w_j=1,
\qquad
\sum_j w_jr_j=1.
\]

Use nondecreasing rates:

\[
0<r_1\le\cdots\le r_k.
\]

Current model/ratefree.cpp constants do not bound every canonical physical rate: depending on the path they bound unsorted optimizer coordinates, including ratios to a then-reference category, before mean normalization. The redesign must not reinterpret them as physical-rate bounds or claim that one sorted interval is identical to every legacy chart.

Define a new, explicit, versioned certified-solver restriction instead. After sorting and choosing \(r_k\) as reference, let \(q_i=r_i/r_k\). The initial compatibility envelope is \(10^{-7}\le q_i\le1\): the legacy CPU joint box can canonicalize as low as \(10^{-6}\), while current JOLT's pinv-free 1e−4…1000 clamp permits a \(10^{-7}\) ratio. This envelope includes both source-visible domains but is not mathematically identical to either. Phase 0 must audit every archived endpoint, run lower-bound expansion tests, and reject promotion if this declared restriction changes a competitive likelihood or BIC result. Record the domain version in every checkpoint and job. Quotient scaling leaves every q unchanged.

Zero weights and coincident positive rates belong to the closure; a zero positive-category rate does not under this declared operational domain. The invariant component remains the separate +I mechanism, avoiding an unidentifiable zero-rate +R/+I allocation. Changing the scale-free rate-ratio domain is a model/optimizer semantics change and requires a separate gate.

The implemented model is the closure of the k-class family:

- zero weights are valid;
- rate collisions are valid;
- coincident active atoms may be merged internally;
- an empty slot may later be split or relocated;
- no hidden positive weight floor or separation penalty is permitted.

An Rk boundary fit remains nominal Rk for compatibility and keeps IQ-TREE's existing \(2k-2\) parameter count. Output also records effective_support and boundary_solution. An unrestricted measure or collapsed support never receives a reduced ad hoc BIC penalty.

Sorting a state must permute rates, weights, component diagnostics, restart metadata, and checkpoint fields together.

### 4.2 Pattern likelihood representation

For pattern p with multiplicity \(n_p\), component likelihood \(F_{pj}\), and mixture likelihood

\[
s_p(w)=\sum_j w_jF_{pj},
\]

the objective is

\[
\ell(w)=\sum_p n_p\log s_p(w).
\]

All component columns for one pattern must share one category-independent scaling offset. Raw category buffers that already include weights cannot be used. The implementation must export either:

- unweighted component log-likelihoods plus one common per-pattern offset; or
- an algebraically equivalent scaled representation whose recomposed likelihood, gradient, and Hessian match the production CPU objective.

For active columns:

\[
g_j=\sum_p n_p\frac{F_{pj}}{s_p},
\qquad
-H_{jl}=\sum_p n_p\frac{F_{pj}F_{pl}}{s_p^2}.
\]

The likelihood is concave in weights. It may not be strictly concave when columns are duplicate or observationally equivalent; fitted pattern probabilities can be identifiable even when individual weights are not.

---

## 5. Production inner solve: bound-aware quotient profiling

### 5.1 Exact coordinate map

At a canonical pure-+R point \((w,r,b)\), hold every component transition kernel \(r_jb_e\) fixed and optimize weights over the simplex.

For proposed weights \(u\), let

\[
m=r^Tu.
\]

Commit the canonical state

\[
r'_j=\frac{r_j}{m},
\qquad
b'_e=mb_e.
\]

Then \(r'_jb'_e=r_jb_e\), \(\sum u_j=1\), and \(\sum u_jr'_j=1\). The likelihood columns remain exactly the same. With all k slots assigned, this profiles all k−1 weight degrees of freedom while rate ratios contribute the other k−1 degrees, preserving the public \(2k-2\) dimension. A boundary state with unassigned empty slots is a union of fixed-support subproblems and is handled by the explicit insertion-pricing logic in Section 7.3, not by pretending that arbitrary empty-slot rates form one smooth block.

### 5.2 Bound-aware feasible interval

The current gaugeFix rescales branches and then clamps values above 20, which breaks likelihood invariance. The new solver must never clamp a gauge transform.

Because quotient scaling leaves rate ratios unchanged, only rescaled physical branch bounds constrain m. Using runtime branch bounds \(B_{\min},B_{\max}\) and the current canonical b, compute

\[
m_L=
\max_e\frac{B_{\min}}{b_e},
\]

\[
m_U=
\min_e\frac{B_{\max}}{b_e}.
\]

Every assigned rate must separately pass the scale-free q-domain before the weight solve. The branch extrema range only over real physical edges that the transform will uniformly rescale; exclude the root's zero parent-length entry and virtual/unidentifiable root edges. If any real physical edge is fixed or cannot share that scale, quotient profiling is illegal and the literal formulation is used. An unassigned empty slot is excluded from this profile and cannot receive weight. Each insertion proposal first assigns that slot a rate, validates the full canonical q-domain, and creates a separate fixed-support subproblem.

The inner feasible polytope is

\[
\mathcal C_q=
\left\{
u\ge0:
\mathbf1^Tu=1,\;
m_L\le r^Tu\le m_U
\right\}.
\]

Use the actual runtime branch bounds from the IQ-TREE parameter state, not JOLT's hard-coded 1e−6 and 20. Every valid canonical start must satisfy \(1\in[m_L,m_U]\); otherwise return INFEASIBLE_START. If either moment bound is active at the quotient optimum, use the literal mass-and-mean profiler for production outer derivatives. This avoids omitting the multiplier and generalized derivatives of the parameter-dependent max/min bounds. A singleton quotient interval is equivalent to the unit-mean literal face only when that singleton is \(m=1\).

### 5.3 Active-set Newton

For k≤10, use a deterministic safeguarded active-set Newton/primal-dual solve:

1. Construct a feasible starting weight vector.
2. Solve the exact observed-likelihood Newton system on the equality and active-inequality tangent space.
3. Use pivoted LDLT or QR, with SVD fallback for rank-deficient duplicate columns.
4. Take a fraction-to-boundary step.
5. Apply an Armijo line search to the unregularized likelihood.
6. Drop weights that reach zero.
7. Add a violated assigned inactive component or activate a moment bound.
8. Repeat until primal feasibility and the exact concavity gap pass.

Damping may stabilize the linear system, but acceptance and termination always use the original objective and certificate.

The exact inner upper gap is

\[
G_w=
\max_{v\in\mathcal C_q}
g(w)^T(v-w),
\]

and concavity gives

\[
0\le\ell^\star-\ell(w)\le G_w.
\]

The linear oracle is a tiny LP. Its vertices contain one or two support points under the simplex and moment interval, so enumerate them deterministically. Frank–Wolfe is used for this certificate and for large-grid diagnostics, not as the production optimizer.

Return likelihood, weights, active set, numerical rank, \(m,m_L,m_U\), primal residuals, multipliers, KKT residuals, \(G_w\), evaluation count, and exit reason.

If the solution activates \(m_L\) or \(m_U\), label the quotient result diagnostic, run the literal profile from the same canonical point, and let the literal formulation drive the outer step and stationarity test. A future semismooth quotient implementation may replace this fallback only after its active-bound envelope derivative is derived and finite-difference validated.

### 5.4 Literal-rate reference and fallback

Hold absolute rates and branches fixed and solve

\[
\mathcal C_a=
\left\{
w\ge0:
\mathbf1^Tw=1,\;
r^Tw=1
\right\}.
\]

This block has k−2 weight degrees of freedom, while all k absolute rates remain outer coordinates; total FreeRate dimension is still \(2k-2\). Its gap uses the same LP definition over \(\mathcal C_a\). The two-equality oracle enumerates feasible one-point \(r=1\) and two-point distributions that bracket one.

Use this formulation:

- as the independent Stage-0 diagnostic;
- whenever a quotient moment bound is active, including a pinned interval;
- for fixed-branch/reference modes in which uniform branch scaling is illegal;
- to cross-check quotient stationarity after full outer cycles.

For R2 the literal block normally has zero weight degrees of freedom; this is expected, not a solver failure.

---

## 6. Diagnostic grid measure

A nested deterministic log-rate grid may be solved with unrestricted weights under the relevant affine constraints. It can:

- expose missed support regions;
- propose split, insertion, and replacement events;
- seed fixed-k starts;
- provide a global optimum and dual gap only for that finite-grid unrestricted problem.

It cannot:

- certify a continuous-rate optimum;
- certify a cardinality-constrained Rk optimum when more than k grid atoms are active;
- be stopped at k support points and called an Rk MLE;
- receive the \(2k-2\) Rk BIC;
- provide a continuous upper bound without a globally validated affine-majorant/pricing oracle over the entire rate interval.

Name its successful status GRID_CERTIFIED. It is never a production candidate status.

---

## 7. Outer fixed-k solver

### 7.1 Block cycle

Run deterministic monotone major cycles:

1. Profile quotient weights to the current forcing tolerance.
2. Update nondecreasing rate locations in the common-scale quotient tangent.
3. Re-profile weights at every rate trial.
4. Optimize branches in a separate block.
5. Re-profile weights.
6. Optimize Q/frequencies in a separate CPU block initially.
7. Re-profile weights and evaluate every terminal residual.

No block shares another block's damping or curvature history.

### 7.2 Rate block

Retain all k physical rates in state, but project out the common-scale null direction. Use a bounded projected or semismooth trust region with:

- nondecreasing rates, including equality;
- explicit canonical scale-free rate-ratio domain;
- feasible quotient gauge interval;
- exact profile-gradient or finite-difference validation;
- certified profile intervals for acceptance;
- history reset on zero-weight entry/exit, collision, merge, or split; an active quotient gauge bound triggers the literal-profile fallback.

Do not use strict exponential gaps: they exclude legitimate collisions. A zero-weight atom has no identifiable location and is omitted from ordinary gradient convergence checks.

For the literal mass-and-mean reference, the envelope derivative includes the mean multiplier. With Lagrangian sign convention

\[
\mathcal L=\ell-\alpha(\mathbf1^Tw-1)-\beta(r^Tw-1),
\]

the log-rate derivative for an interior active atom is

\[
\frac{\partial\phi}{\partial\log r_j}
=
r_jw_j
\left[
\sum_p n_p\frac{\partial_rF_{pj}}{s_p}
-\beta
\right].
\]

Validate this and the quotient Jacobian with central finite differences that re-solve both displaced weight problems to a substantially tighter gap.

### 7.3 Support-event handler

On a zero weight, collision, or ordering face:

1. Merge coincident active atoms and sum their weights.
2. Mark remaining nominal Rk slots empty.
3. Re-solve the reduced active support.
4. Generate a fixed set of split candidates around every active atom.
5. Add the deterministic R(k−1) continuation split.
6. Add proposals from the diagnostic grid.
7. If all slots are occupied, scan deterministic one-for-one replacements.
8. Re-profile every proposal and continuously refine only certified improvements.
9. Accept the best certified event and reset all rate curvature history.

Finite grid, split, and swap proposals are basin-search mechanisms only. When a zero weight or collision creates an empty slot, local stationarity in the model closure additionally requires a globally bounded one-dimensional pricing solve over the entire positive-rate domain allowed by the canonical scale-free ratio contract. It must bound the best equality/gauge-feasible infinitesimal insertion gain, using interval branch-and-bound or a proved analytic/Lipschitz envelope. Dense scanning plus local Brent searches is not a bound.

If the continuous insertion upper bound is at most \(\tau_L\), the boundary may receive BOUNDARY_LOCAL_STATIONARY_CERTIFIED. Without that bound, the strongest status is BOUNDARY_DIRECTIONALLY_TESTED; it is unresolved for certified ModelFinder and cannot be scored. No-improving finite events alone never certify the boundary.

### 7.4 Branch block

Correctness reference:

- retain the CPU Gauss–Seidel/Brent branch optimizer;
- re-profile weights after branch moves;
- measure a separate exact global-scale residual;
- require monotone full profiled likelihood.

GPU phase:

- use exact all-edge gradients;
- use its own projected L-BFGS or trust-region state;
- line-search the fully re-profiled objective;
- reset on active branch bounds or support events;
- retain final fresh CPU same-state reevaluation.

Do not reuse HARP's mixed rate/branch subspace or empirical OPG curvature.

### 7.5 Q and frequency block

Keep Q/frequencies on the CPU initially. Remove them from the current monolithic FreeRate inner loop. Every accepted Q/frequency update is followed by an exact weight profile. Only add GPU Q derivatives after value/gradient parity and side-effect-free trial evaluation are proven.

### 7.6 Inexact-profile forcing and monotone acceptance

An inner solve gives

\[
\phi(x)\in[L(x),L(x)+G_w(x)].
\]

For trial y from incumbent x:

- accept only if \(L(y)>L(x)+G_w(x)+\tau_{\rm noise}\);
- reject if \(L(y)+G_w(y)<L(x)-\tau_{\rm noise}\);
- if intervals overlap, tighten both inner gaps and reevaluate;
- if they still overlap at final precision, treat the step as numerically null and shrink the trust radius.

Early in the solve require

\[
G_w\le c\max(\text{predicted gain},\tau_L),
\qquad c<1,
\]

then tighten geometrically until \(G_w\le\tau_w\), with \(\tau_w\le0.1\tau_L\).

This prevents approximate inner solves from moving the old wobble into profile-evaluation noise.

### 7.7 Deterministic starts

Use a preregistered portfolio, independent of candidate completion order:

- current IQ-TREE gamma-quantile/legacy start;
- deterministic R(k−1) continuation and every admissible atom split;
- clustered support from the GRID_CERTIFIED diagnostic;
- the archived best fixed-cell checkpoint where available;
- controlled small endpoint perturbations for the restart certificate.

The best completed local-stationary point is retained. Start agreement is evidence of repeatability, not proof of globality. Completion-order “first fit wins” warm caching is prohibited.

---

## 8. Status and termination contract

### 8.1 Typed status

Use at least:

- LOCAL_STATIONARY_CERTIFIED;
- BOUNDARY_LOCAL_STATIONARY_CERTIFIED;
- BOUNDARY_DIRECTIONALLY_TESTED;
- FALLBACK_CERTIFIED;
- MAXITER;
- REJECT_STALL;
- MULTIBASIN_UNRESOLVED;
- INFEASIBLE_START;
- GPU_CPU_MISMATCH;
- NUMERICAL_FAILURE;
- UNSUPPORTED;
- LEGACY_UNCERTIFIED.

The word MLE must not appear in a success enum.

### 8.2 Terminal requirements

A start is locally certified only when all conditions hold:

- inner convex gap \(G_w\le\tau_w\);
- mass, mean/gauge interval, and nonnegativity residuals pass;
- projected rate first-order improvement \(\chi_r\le\tau_L\);
- projected branch/global-scale improvement \(\chi_b\le\tau_L\);
- projected Q/frequency improvement \(\chi_q\le\tau_L\);
- best tested split/insertion/swap gain \(\le\tau_L\), as a basin-search gate rather than a proof;
- whenever an empty slot exists, the globally bounded continuous insertion-pricing gain is \(\le\tau_L\);
- full profiled likelihood change and scaled parameter step pass for two consecutive major cycles;
- no member of the preregistered finite restart portfolio produced a certified gain above \(\tau_{\rm restart}\);
- fresh CPU and GPU evaluation of the same canonical point pass;
- no cap, failed line search, unresolved support event, or arithmetic error occurred.

Define the scaled feasible sets explicitly:

- rate directions lie in the active order/bound tangent cone, have zero common-scale component, and satisfy \(\max_j|d\log r_j|\le1\) over positive active atoms;
- branch directions lie in the active branch-bound tangent cone and satisfy \(\max_e|d\log b_e|\le1\) over real optimizable edges;
- positive Q rates use \(\max_i|d\log q_i|\le1\), while frequency directions use the corresponding simplex-logit tangent with infinity norm at most one.

Each \(\chi\) is the maximum linearized likelihood gain in nats over its stated set. Use \(\tau_x=10^{-6}\) for the maximum accepted step in these scaled coordinates. Do not use a raw gradient norm whose interpretation changes with site count.

For zero/colliding atoms compare the canonical fitted mixture, total likelihood, and per-pattern likelihood digest—not arbitrary labels.

### 8.3 Research tolerances

First measure the repeated-evaluation numerical floor \(E_{\rm num}\) on CPU and GPU.

Initial preregistered rules:

- same-state CPU/GPU total likelihood discrepancy must be no larger than 1e−4 nat for promotion;
- \(\tau_{\rm noise}\ge10E_{\rm num}\);
- \(\tau_L=\max(10^{-4}\text{ nat},100E_{\rm num})\), with a hard promotion ceiling of 0.01 nat;
- \(\tau_w\le0.1\tau_L\);
- mass, unit-mean, and negativity residuals target 1e−12 in FP64-scaled coordinates;
- if \(100E_{\rm num}>0.01\) nat, promotion fails rather than weakening the resolution target;
- define the separate empirical policy threshold \(\tau_{\rm restart}=0.1\) nat; independent deterministic starts and endpoint restarts must finish within it or return MULTIBASIN_UNRESOLVED;
- same hardware, binary, point, and JOLT_NTILE=1 must reproduce the canonical trace digest exactly in three runs;
- A100/H200 and later tiled evaluations must agree within twice the measured arithmetic floor and return the same status.

These values are calibrated on synthetic and easy controls, then frozen before the hard avian run. They are not tuned by k, dataset, or observed winner. The 0.1-nat restart/basin policy is not a numerical stationarity certificate or a bound on the unknown global candidate optimum.

---

## 9. +I+Rk is a production gate

Do not initially collapse +I into an ordinary rate-zero atom.

IQ-TREE stores:

\[
\sum_j p_j=1-\pi,
\qquad
\sum_j p_jr_j=1.
\]

Use pinv-free positive-category coordinates:

\[
w_j=\frac{p_j}{1-\pi},
\qquad
\rho_j=(1-\pi)r_j,
\qquad
\sum_jw_j=1,
\qquad
\sum_jw_j\rho_j=1.
\]

At fixed \(\pi\), quotient profiling operates on \((w,\rho)\). If \(m=w_{\rm new}^T\rho\), commit:

\[
\rho'=\rho/m,
\qquad
b'=mb,
\qquad
p'=(1-\pi)w_{\rm new},
\qquad
r'=\rho'/(1-\pi).
\]

Pinv remains a separate bounded outer variable under IQ-TREE's existing data-derived domain, intersected with the strict requirement \(\pi<1\). If the data-derived upper bound reaches one, the all-invariant singular endpoint is UNSUPPORTED unless a separate model is explicitly defined. Every pinv trial changes the invariant contribution, positive-category mass, and physical rate mapping, and therefore re-runs the positive-category profile. The canonical rate-ratio contract applies unchanged because \(\rho_i/\rho_k=r_i/r_k\).

At fixed \(\pi\), recompute the quotient interval from rescaled physical branch bounds:

\[
m_L=
\max_e\frac{B_{\min}}{b_e},
\]

\[
m_U=
\min_e\frac{B_{\max}}{b_e}.
\]

Use real rescalable physical edges only. The quotient transform changes the rescaled branches \(mb\) and leaves all positive-rate ratios unchanged; it imposes no invented absolute physical-rate bound.

Required +I gates:

- \(\pi=0\) reduces exactly to pure +R;
- normalization and nominal dimension are \(2k-1\);
- constant, nonconstant, ambiguous, and missing patterns match the current CPU semantics;
- the additive invariant path and common pattern scaling match on CPU/GPU;
- fixed pinv is separately validated before enabling it;
- no_rescale_gamma_invar remains unsupported;
- +ASC remains unsupported until its conditional likelihood is proved concave in the proposed block;
- the strict pinv domain, canonical rate-ratio domain, and rescaled-branch quotient interval are feasible;
- the known pinv finite-difference state-leak class cannot occur because trial state is immutable.

Pure +R completion does not authorize an avian or full-ModelFinder success claim.

---

## 10. Source architecture

### 10.1 Existing seams

At source commit ccabc96e111b:

- model/modelfactory.cpp:1591–1617 dispatches JOLT and returns early, bypassing the legacy CPU block loop.
- tree/phylotreegpu.cpp:2125–2800 performs eligibility, marshaling, monolithic call, writeback, and CPU check.
- tree/gpu/gpu_iqtree.h:238–274 declares the current nullable-argument C ABI.
- tree/gpu/gpu_lnl_intree.cu:2626–3515 contains the monolithic solver.
- model/ratefree.cpp defines the public \(2k-2\) dimensions and current CPU EM/BFGS paths.
- main/phylotesting.cpp controls candidate construction, warm starts, adaptive rate pruning, checkpoint restore, and completion.

Intercept only the FreeRate arm. Non-R JOLT behavior remains unchanged.

### 10.2 New host components

Add:

- model/freerateprofile.h/.cpp — convex quotient and literal weight profilers, certificates, canonicalization;
- model/freeratesolver.h/.cpp — deterministic starts, support events, rate/branch/Q block coordinator, typed results;
- an IFreeRateEvaluator interface with CPU and GPU implementations;
- a versioned FreeRate checkpoint/certification schema.

RateFree receives one import/export/canonicalize method that atomically validates and sets rates plus matching weights. Do not scatter raw setRate/setProp calls.

Add a typed path:

\[
\text{CandidateModel::evaluate}
\rightarrow
\text{ModelFactory::optimizeFreeRateCertified}
\rightarrow
\text{PhyloTree/IFreeRateEvaluator}
\rightarrow
\text{FreeRateFitResult}.
\]

Do not route certified fitting through the existing double-return optimizeParametersJOLT contract, and do not store status in thread-local or process-global side state. CandidateModel must own the returned FreeRateFitResult until publication. Gate all of the following on its certified status:

- early checkpoint trust in main/phylotesting.cpp:2449–2452;
- in-loop model/tree saves around :2639–2655;
- final candidate save around :2685;
- warm-cache population at :2690 onward;
- IC computation, rate-family pruning, MF_DONE, and any R(k+1) continuation seed.

This typed result is also the only path that may produce FALLBACK_CERTIFIED. A scalar likelihood from a legacy path remains LEGACY_UNCERTIFIED.

### 10.3 New GPU evaluator ABI

Do not append more nullable arguments to gpu_jolt_optimize. Add a versioned POD API with an opaque, instance-owned workspace:

- FreeRateProblemView — topology, patterns, eigen/Q state, invariant column, frequencies, bounds, branch mode, scaling contract;
- FreeRatePoint — all physical rates, weights, branches, pinv, and Q snapshot;
- FreeRateEvalRequest — VALUE, COMPONENT_LOGS, GRADIENT, BRANCH_DIAG, PATTERN_SCORES, optional HVP;
- FreeRateEvalResult — likelihood, common pattern offsets or stream handle, rate/branch gradients, diagonal diagnostics, counters, and status;
- FreeRateWeightProfileResult — weights, gain, gap, multipliers, active set, gauge interval, residuals;
- FreeRateFitResult — typed status, final point, all block metrics, starts, evaluations, trace digest, schema/source identifiers.

Create/destroy the workspace separately from evaluation. Trials own their point and never mutate IQ-TREE model objects. Commit exactly once after:

1. solver success;
2. canonical sorting and boundary handling;
3. final GPU evaluation;
4. fresh CPU same-state parity;
5. checkpoint validation.

A failed call leaves the IQ-TREE object and caches unchanged.

### 10.4 Reusable GPU work

Reuse:

- topology flattening, deterministic tiling, and persistent buffers;
- postorder/preorder and all-edge sweeps;
- aggregate physical-rate and branch gradients;
- deterministic pattern reductions.

Do not reuse:

- weight-folded category arrays as unweighted columns;
- softmax weight floors;
- hard-coded branch bounds;
- one shared mu;
- coordinate secants called a Hessian;
- mutable pinv/Q finite-difference state;
- process-global solver state as the new API contract.

Add an unweighted component-output path with one common per-pattern scale. Never divide a weight-folded column by a floored or zero weight.

For k≤10, reduce value, k-gradient, and k×k weight Hessian in a deterministic streamed pass and transfer only O(k²) statistics per Newton step.

### 10.5 Branch modes and partitions

V1 supports only unpartitioned BRLEN_OPTIMIZE.

- BRLEN_FIX cannot use quotient gauge motion; it is UNSUPPORTED in certified mode and retains only LEGACY_UNCERTIFIED CPU behavior outside that mode.
- BRLEN_SCALE has one common scale and needs a separate coordinator; it is likewise UNSUPPORTED in certified mode until validated.
- -p/-spp uses PartitionModelPlen and shared/proportional branch logic; a child solver must not write independent branches.
- -sp/-Q may reach BRLEN_OPTIMIZE, but process-global GPU state currently serializes candidates.
- ASC, unsupported mixtures, site-specific/nonreversible models, and partially fixed +R parameters continue to decline.

Until the joint partition-aware coordinator passes Phase 7, partitions and BRLEN_SCALE return UNSUPPORTED/INCOMPLETE in certified mode. Outside certified mode they may use the existing CPU path only as LEGACY_UNCERTIFIED. The literal weight oracle alone does not certify a partitioned, BRLEN_FIX, or BRLEN_SCALE full fit; current -q/-spj behavior that suppresses +R remains unchanged.

---

## 11. ModelFinder integration and failure policy

### 11.1 Candidate lifecycle

In certified mode:

1. Construct an explicit candidate manifest.
2. Disable adaptive rate-family skipping for initial validation.
3. Assign deterministic keyed starts independent of scheduling order.
4. Fit every requested R2–R10 and +I+R candidate.
5. Publish its likelihood, IC, warm-start state, and completion flag only after certification.
6. Compute BIC post hoc from certified endpoints using existing nominal parameter counts.

An old checkpoint without a certification record is LEGACY_UNCERTIFIED and forces a +R refit.

### 11.2 Unresolved candidate

If GPU fitting returns anything other than a certified result:

1. Restore the unchanged base point.
2. Run the higher-budget CPU implementation of the same profile solver and start policy.
3. Accept it only if it meets the same certificate and is no worse than the GPU endpoint.
4. Mark it FALLBACK_CERTIFIED.
5. If CPU also fails, mark the entire model-selection analysis INCOMPLETE and exit diagnostically.

Never:

- turn MAXITER into a low likelihood;
- omit the candidate and continue;
- allow it to trigger pruning;
- use its checkpoint as an R(k+1) seed;
- report a unique winning model from the remaining candidates.

Outside certified mode, legacy fallback may remain available, but its status is explicitly LEGACY_UNCERTIFIED and cannot support the new correctness claim.

### 11.3 Selection wording

Report:

> Best certified model under the declared deterministic start policy.

Do not report “global best +Rk MLE.” Inner profile intervals bound numerical error only at an attained outer point; finite-start basin spread is empirical and is not an upper bound on the global candidate likelihood. Report a selection tie or INCOMPLETE when certified numerical score bounds overlap the winning BIC margin, or when the preregistered basin results make the winner policy-unstable.

Adaptive pruning may return only after it consumes certified scores or rigorous safe bounds. The existing --thread-model/MF_WAITING R3+ omission must be fixed or bypassed before any evidence from that mode is accepted.

---

## 12. Implementation phases and gates

### Phase 0A — evidence instrumentation

Deliver:

- immutable point serialization and provenance;
- full cap/restart traces;
- per-block residual hooks;
- typed current-JOLT exit reasons;
- fixed tree and full-state injection.

Gate:

- identical first-400 trajectories for matched longer runs;
- exact restart continuity;
- exact serialized post-gauge rescoring and an explicit pre/post-gauge delta;
- same-state CPU/GPU parity;
- no silent runtime flag or binary mismatch.

### Phase 0B — CPU fixed-support oracles

Deliver:

- quotient active-set Newton with moment interval;
- literal mass-and-mean active-set Newton;
- exact primal/KKT/FW gap;
- independent generic constrained-solver harness on small cases;
- unweighted common-scale component reconstruction.

Gate:

- synthetic k=2…10 objective and gap agreement;
- duplicate, collision, zero-weight, boundary, and infeasible cases;
- every archived competitive endpoint lies in the declared canonical rate-ratio envelope, and expanding that envelope does not change a competitive endpoint beyond \(\tau_L\);
- fitted pattern probabilities agree when weights are nonunique;
- Stage-0 residual attribution completed.

No-go:

- profiler hypothesis not implicated and no outer use requires it;
- component scaling cannot reconstruct the production objective;
- hidden floor or unexplained certificate violation.

### Phase 1 — CPU pure-+R fixed-tree solver

Deliver:

- quotient rate trust region;
- support-event handler;
- globally bounded continuous insertion pricing for empty-slot boundary certification, or an explicit refusal to certify such boundaries;
- deterministic start portfolio;
- profile-interval acceptance;
- local-stationary status and restart certificate.

Gate:

- every fixed-tree hard/easy R2–R10 cell terminates without a cap; a boundary cell is certified only when continuous insertion pricing passes;
- best endpoint is no worse than every archived JOLT/HARP/stock endpoint for the identical objective;
- all residuals and repeatability thresholds pass.

### Phase 2 — CPU branches and Q

Deliver:

- monotone branch block with separate global-scale test;
- Q/frequency CPU block;
- composite stationarity sweep;
- rollback and checkpoint recovery.

Gate:

- fixed-topology avian, DNA, AA, and high-k eukaryote cells are locally certified;
- candidate-order and thread-order independence;
- controlled endpoint restarts pass.

### Phase 3 — CPU +I+R

Deliver:

- pinv-free coordinate bridge;
- free and fixed pinv handling;
- invariant-column validation;
- +I boundary tests.

Gate:

- normalization, likelihood, gradients, dimensions, and checkpoint round trips pass;
- avian +I+R hard cell is locally certified.

Without this phase there is no full avian or ModelFinder promotion.

### Phase 4 — GPU evaluator

Deliver:

- side-effect-free workspace/evaluation ABI;
- unweighted common-scale component output;
- value, weight statistics, rate/branch gradients;
- deterministic tiling;
- atomic host commit.

Gate:

- CPU/GPU component, recomposed likelihood, gradient, Hessian statistic, KKT gap, endpoint, and status parity;
- three exact repeats per hardware;
- A100/H200 and tile-count invariance within declared arithmetic tolerance;
- final fresh CPU parity remains enabled.

### Phase 5 — full single-alignment ModelFinder

Deliver:

- explicit identical R2–R10/+I+R candidate manifest;
- pruning disabled;
- certification-aware checkpoint and warm starts;
- incomplete-analysis behavior;
- post-hoc BIC.

Gate:

- every candidate certified or certified-fallback;
- candidate order does not change endpoint/status;
- selected model and final topology stable under the declared start policy;
- bare stock defaults, project harness flags, and GPU certified mode remain separate arms.

### Phase 6 — production rollout

Deliver:

- default-off experimental flag;
- same-commit OFF/ON gates;
- runtime engagement marker;
- kill switch and schema migration;
- explicit unsupported-mode fallback.

Promotion:

- pure +R alone may be promoted only for its declared narrow scope;
- the full ModelFinder claim requires +I+R and all Phase-5 gates;
- default-on occurs only after two independent hard datasets and the full regression matrix pass.

### Phase 7 — partitions and performance

Only after correctness:

- design the joint PartitionModelPlen coordinator;
- validate BRLEN_SCALE and linked/proportional branches;
- cache component columns within immutable structural states;
- stream pattern blocks at fixed reduction order;
- batch rate columns, then starts;
- make workspaces instance-owned to remove static mutex serialization;
- move repeated immutable setup to device;
- benchmark time-to-certificate.

Mixed precision, approximate candidate ranking, CTF, candidate batching, and relaxed convergence are not authorized by this phase without a fresh correctness gate.

---

## 13. Validation matrix

### 13.1 Algebra and unit cases

- k=2 through k=10 interior solutions;
- all rates equal to one;
- duplicate and nearly duplicate component columns;
- zero weights, rate collisions, empty slots, and scale-free rate-ratio boundaries;
- quotient interval interior, active lower/upper bound, singleton, and infeasible interval;
- all-constant, no-constant, ambiguous, missing, and extreme-frequency patterns;
- simulated true R4 fitted as R4, R8, and R10;
- pinv zero, near maximum, and fixed pinv;
- checkpoint canonicalization and restart identity.

Compare the small convex problems with an independent high-precision constrained optimizer. Compare objective and fitted mixture probabilities, not arbitrary nonunique weights.

### 13.2 Real fixed cells

| Dataset/cell | Purpose |
|---|---|
| Avian GTR+F+R6 | Primary 400-cap failure |
| Avian GTR+F+R8 | Higher-k support behavior without claiming it is the true winner |
| Avian GTR+F+I+R6 | Mandatory +I production gate |
| DNA-100K | Easy nucleotide and free-Q control |
| AA-100K | Easy fixed-Q control |
| CAT_100S93F amino-acid alignment | Real high-k transfer test |

Each run uses an explicit tree and full point. Seed changes may be tested only after the fixed-objective suite and are labeled topology/start-policy experiments.

### 13.3 End-to-end arms

Run separately:

- pristine stock IQ-TREE with bare defaults;
- pristine stock with -ninit 2 -optalg 2-BFGS;
- canonical legacy GPU JOLT;
- CPU certified profile solver;
- GPU certified profile solver.

Do not combine their results as if they were the same implementation.

### 13.4 Required reports

For every candidate:

- source/binary/state provenance;
- status and effective support;
- likelihood and nominal BIC;
- every block residual and inner gap;
- active rate/branch/pinv bounds;
- start portfolio and basin spread;
- evaluation counts and time-to-certificate;
- CPU/GPU parity;
- checkpoint and trace digests.

For ModelFinder:

- exact candidate manifest;
- pruning state;
- all candidate statuses;
- BIC margins, overlap with certified numerical score bounds, and the separate empirical basin-policy result;
- selected model, final tree lnL, and RF as a downstream result—not as an optimizer certificate.

---

## 14. Performance plan after correctness

The relevant metric is time-to-certificate, not iterations per second.

Safe first optimizations:

1. Cache k≤10 unweighted component columns for an immutable structural state; avian k=10 is approximately 55 MB at the audited pattern count.
2. Compute value, weight gradient, and k×k Hessian statistics in one deterministic streamed GPU pass.
3. Transfer O(k²) statistics, not pattern-sized arrays, per Newton iteration.
4. Reuse topology, tip, pattern-frequency, eigen, and workspace buffers across side-effect-free evaluations.
5. Batch rate columns and analytic rate derivatives.
6. Replace process-global buffers/mutex with instance-owned workspaces after parity.
7. Batch deterministic starts only after independent-start equivalence is proven.

Forbidden speed shortcuts until separately proven:

- mixed precision;
- approximate ranking;
- CTF or subsampled likelihood;
- skipping final CPU parity;
- lowering inner/outer accuracy;
- scoring cap-bound candidates;
- post-hoc clamps;
- candidate-family pruning from uncertified scores.

The final performance comparison reports:

- certified CPU reference wall time;
- certified GPU wall time;
- current JOLT wall time only on cells where JOLT itself obtains the new certificate;
- legacy unresolved time separately.

---

## 15. Risk register

| Risk | Detection | Mitigation |
|---|---|---|
| Weight profiling is not the causal defect | Stage-0 block gains | Redirect before building the wrong outer solver |
| Category buffers contain weights or incompatible scales | Component reconstruction and gradient tests | Dedicated unweighted common-offset API |
| Gauge transform hits a bound | Explicit m interval | Never clamp; use the literal formulation for production outer derivatives |
| Profile nonsmoothness at support changes | Zero/collision/order event | Semismooth event handler and history reset |
| Duplicate columns make Newton singular | Rank-revealing factorization | QR/SVD, merge equivalent columns, gap-based stop |
| Inexact profile corrupts outer line search | Overlapping profile intervals | Tighten forcing tolerance before accept/reject |
| Local modes survive | Preregistered starts and restart spread | Best certified local basin or MULTIBASIN_UNRESOLVED |
| +I changes model semantics | Algebraic bridge and pattern tests | Keep pinv separate; no zero-atom shortcut initially |
| Candidate ordering changes warm starts | Randomized schedules and keyed starts | Deterministic start portfolio independent of completion |
| Old checkpoints are trusted | Missing certification schema | Mark legacy and refit |
| Partition child mutates shared branches | Partition-mode integration tests | UNSUPPORTED/INCOMPLETE in certified mode until the joint coordinator |
| GPU solver partially writes failed state | Fault-injection tests | Immutable trials and one atomic commit |
| BIC is nonregular on mixture boundaries | Boundary report | Retain legacy nominal count for compatibility; do not claim statistical regularity |

---

## 16. Definition of done

The FreeRate convergence work is complete only when:

1. The historical record is corrected and all jobs are provenance-complete.
2. The fixed avian cap behavior has a measured block-level cause.
3. The production inner weight block has an exact finite-dimensional likelihood-gap certificate.
4. Every supported outer block passes directional stationarity and restart tests.
5. Zero weights, collisions, and bounds are valid states rather than hidden floors or numerical exceptions; an empty-slot boundary is certified only by globally bounded continuous insertion pricing.
6. Pure +R and +I+R both pass CPU reference gates.
7. GPU component values, derivatives, gaps, endpoints, and statuses match CPU.
8. No unresolved candidate is scored, pruned from, checkpointed as complete, or used as a warm seed.
9. Full ModelFinder with an identical explicit manifest is stable under candidate ordering, repeats, threads, and supported GPUs.
10. The result is faster only after all nine correctness conditions remain true.

If fixed-k nonconvexity still produces incompatible certified local basins, the honest completed behavior is MULTIBASIN_UNRESOLVED plus an incomplete ModelFinder result. Concealing that uncertainty is not an acceptable optimization.

---

## 17. Primary references

- Susko et al. (2003), fixed-grid phylogenetic rate-distribution estimation with nonnegative mass and unit-mean constraints: https://doi.org/10.1080/10635150390235395
- Lindsay (1983), mixture-likelihood geometry and finite-support representations: https://doi.org/10.1214/aos/1176346059
- Groeneboom, Jongbloed, and Wellner (2008), support reduction for convex measure problems: https://doi.org/10.1111/j.1467-9469.2007.00588.x
- Wang (2007), constrained Newton methods for mixing distributions: https://doi.org/10.1111/j.1467-9868.2007.00583.x
- Wang (2010), profile/support algorithms for semiparametric mixtures: https://doi.org/10.1007/s11222-009-9117-z
- Feng and Dicker (2018), finite-grid approximate NPMLE as a computational device, not a phylogenetic error bound: https://arxiv.org/abs/1606.02011
- Golub and Pereyra (1973), variable projection as conceptual precedent only: https://doi.org/10.1137/0710036
- Morel et al. (2021), empirical FreeRate optimization difficulty: https://doi.org/10.1093/molbev/msaa314
- Systematic Biology (2026), current evidence that FreeRate fitting remains difficult and nominally uses \(2(k-1)\) parameters: https://doi.org/10.1093/sysbio/syag037

These references justify the convex inner decomposition and the need for better FreeRate optimization. None proves global convergence for fixed-k continuous supports, joint branch/Q optimization, topology search, or full ModelFinder.
