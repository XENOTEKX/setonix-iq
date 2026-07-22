# JOLT HARP — closed-line post-mortem

*Consolidates the nine `JOLT-HARP-*.md` design/review/result documents into one standalone record. Those nine are
preserved unmodified as the provenance trail; **this file is the one to read.** Every number below carries a job ID,
a basis label (for spectra), or an explicit "unverifiable" mark.*

---

# 0. VERDICT

> ## 🔴🔴 HARP IS CLOSED — 2026-07-21
>
> | | |
> |---|---|
> | **Status** | **CLOSED.** Not paused, not pending, not awaiting a gate. |
> | **Closed by** | Job **`174328851`** ARM 3, applying the *pre-registered* decision rule written into the gate script **before** the run. |
> | **Closing evidence** | τ=1e-3 gains **+88.72 nats** on avian `GTR+R6` and loses **−705.62 nats** on avian `GTR+R8`, same fixed tree, same τ, same job. The τ response **oscillates**; it is not a basin. |
> | **Promoted** | **Nothing.** No source, no binary, no flag, no default. |
> | **Source disposition** | Branch `harp-ws15` @ **`ce99e337`** in `/scratch/rc29/as1708/iqtree3-harpspec` — **local and unpushed.** (`git branch -r --contains ce99e337` → empty. Verified on disk 2026-07-21; this hash appears in **none** of the nine source docs.) |
> | **Tier-3 successor** | True-Fisher / GGN replacement of the OPG is **documented and UNAUTHORISED.** Do not open it. |
> | **Re-open bar** | Only on **new evidence of a kind none of these gates produced** — i.e. a *cross-cell* result, not another single-cell win. Four independent gates, five τ values, three DYMAX values, two tolerances, and four datasets have already been spent. A sixth τ is not new evidence. |
>
> **Two things that must not be misread:**
> 1. **Job `174328851` is not a tolerance sweep.** Its on-disk directory is `gems-verify/tolladder_174328851/`, its
>    PBS name is `tolladder`, and arms `a1_*`/`a2_*` *are* tolerance-ladder arms — but the **110 `a3_*` files are
>    HARP-ON** (`JOLT_HARP=1 JOLT_HARP_TAU=…`). Anyone reading that job as a pure tolerance experiment inherits a HARP
>    confound. See §9.
> 2. **The WS2 design (`JOLT-HARP-WS2-DESIGN.md`) was NEVER BUILT.** Its verdict line is
>    `# VERDICT: DO NOT BUILD — NEVER BUILT.` (`JOLT-HARP-WS2-DESIGN.md:231`). Its body reads like a live design; it is
>    not. Tikhonov, the ρ-test, κ-adaptation and Armijo exist only as text. See §10.

---

# 1. WHAT HARP WAS

## 1.1 The hypothesis

**HARP = Hard Active-Rank Projection.** One line:

> *A dense empirical-Fisher (OPG) step is useful in the statistically identified tangent space, but its near-null
> components are harmful; delete those components **exactly**, rather than flooring, damping, capping, or using the
> smallest eigenvalue to enable/disable the entire solve.*
> — `JOLT-HARP-PHASE1-DESIGN.md:5-7`

The numerical primitive (truncated symmetric eigensolve / TSVD) is textbook. The research claim was narrower and was
always stated as such: applying a **GPU-built empirical-Fisher effective-rank tangent space to the joint FreeRate
block, with an exact-zero null step and a same-iterate canonical competitor**, appeared not to have been used in
phylogenetic model selection. The design explicitly disclaimed inventing truncated SVD
(`JOLT-HARP-PHASE1-DESIGN.md:9-12`).

## 1.2 The `+R` problem it targeted

FreeRate (`+Rk`) fits on real data do not converge — they get **truncated by an iteration cap**. On avian `GTR+R6`
(48 taxa × 1M sites) the GPU joint optimiser terminates `reason=maxiter` at `it=401`, having burned
`nLnLEval=2609` likelihood evaluations and `nRej=206` rejected backtracks, at `lnL=-11216886.230053`
(job `174266861`, `off_can_av.console`, `tol=1e-7`, `-n 0 -starttree PARS`).

The mission metric was therefore **avian joint iterations**, not matched-`R4` speed: a method that halves R4 while
leaving avian at 401/401 does not solve the commissioned bottleneck (ultra-review finding **R1**).

## 1.3 THE FORMULA — WS1, unscaled / physical `H_r`

This is **the** HARP step. It is the only formula that was ever built. It is from
`JOLT-HARP-RECONCILIATION.md:65-75` (frozen §3.1), and it eigendecomposes the **raw physical tangent matrix**:

```text
Q       = diag(I_k, Helmert_k)          # removes only the softmax-logit null; n = 2k-1
H_r     = Qᵀ H Q                        # RAW / PHYSICAL. No D. No correlation scaling.
g_r     = Qᵀ (g_y, g_z)
H_r     = V diag(λ_i) Vᵀ
τ       = 1e-4                          # WS1: compiled constant. WS1.5: runtime JOLT_HARP_TAU.
keep_i  = finite(λ_i) && λ_i > 0 && λ_i/λ_max ≥ τ
a_i     = v_iᵀ g_r
δ_r     = Σ_{i ∈ keep} v_i · a_i / λ_i          # dropped coefficient = the literal float 0.0
δ       = Q δ_r
```

There is **no `D`**, no spectral floor, no `λ_LM`, no norm cap, no gain ratio, no persistent state, and no nonzero
denominator for a dropped mode.

> ### ⚠️ BOXED WARNING — the PHASE1 `D`-scaled formula was KILLED and NEVER BUILT
>
> `JOLT-HARP-PHASE1-DESIGN.md:62-100` specifies a **different** step, in `D`-scaled correlation coordinates:
> `D_i = max((H_r)_ii, 1e-12·max_j(H_r)_jj)`, `C = D^(-1/2) H_r D^(-1/2)`, keep on `C`'s spectrum, then
> `δ_r = D^(-1/2) δ_h`.
>
> **That formula was killed by red-team finding 2** (`JOLT-HARP-RED-TEAM.md:54-84`): the permitted
> `D_i/max(D) = 1e-12` plus two `D^(-1/2)` maps means the advertised `10⁴` retained-spectrum condition bound exists
> **only in correlation coordinates** — the physical step can carry an extra scale factor approaching `10¹²`. Blue
> team concurred (`JOLT-HARP-BLUE-TEAM.md:47-51`); reconciliation accepted it
> (`JOLT-HARP-RECONCILIATION.md:24-27`); ultra-review **R3** confirmed it for the scaled solve
> (`JOLT-HARP-ULTRA-REVIEW.md:34-36`).
>
> **`λ_LM` never existed.** The Phase-1 formula contains `δ_h = Σ v_i a_i/(λ_i + λ_LM)`, but `λ_LM` had **no**
> initialization, bounds, reject multiplier, Nielsen state, gain-ratio predictor, or interaction rule with canonical
> `mu` (red finding 3, `JOLT-HARP-RED-TEAM.md:86-114`). It was declared a **pre-build blocker** — "there is no single
> algorithm for the gate to test" — and was **removed entirely** in WS1, not specified.
> `JOLT-HARP-RECONCILIATION.md:27-30`: *"WS1 has no lambda, floor, gain ratio, Nielsen state, latch, norm cap, or
> damping override."*
>
> The `D`-scaled reduction survives in the codebase **only as a read-only diagnostic** — it is what `[OPGLMIN]`
> prints and what `[HARPSPEC-DSCALED]` dumps for comparison. It never selected a mode in any state-changing run.
> Full text preserved in **Appendix A**, provenance only.

## 1.4 The six-route triage

From `JOLT-HARP-PHASE1-DESIGN.md:39-46`. HARP was chosen as the smallest experiment testing the one distinction the
dead OPG solver left untried: **zero is qualitatively different from a small wrong step.**

| Direction | Escape mechanism | Disposition |
|---|---|---|
| **Hard active-rank projection** | Set each empirically unidentifiable eigencomponent to exactly zero; solve only in the retained tangent space. | **SELECTED.** Built as WS1 → WS1.5. Closed. |
| BIC/rank-aware early stop | Avoid optimising surplus precision. | **Rejected.** `gpu_jolt_optimize` has no incumbent/BIC state; the phylogenetic branch objective lacks the needed safe global bound. A local `dl` rule is tolerance loosening by another name. |
| Ordered / spacing coordinates | Remove label permutations and rate crossings. | **Rejected.** Ordering moves component collisions to a boundary; it cannot make a statistically unidentified model identifiable. |
| Weight mirror descent / natural gradient | Respect simplex geometry, avoid weight clipping. | **Rejected as primary.** Mirror descent does not remove rate/weight nonidentifiability; a full natural gradient explicitly inverts Fisher curvature ⇒ dead family. |
| Weak penalty / MAP | Lift flat directions by changing the objective. | **Rejected.** Changes the scored objective; makes selection invariance a tuning claim. |
| Functional support reduction | Optimise the rate mixing measure, delete support atoms, use a vertex-direction gap rather than local curvature. | **Banked as a separate project.** Avian's fitted R6 has six distinct positive-weight rates; a safe fixed-`k`, mean-constrained continuous oracle plus joint branch coupling is far larger than a minimal JOLT change. Refs: Lindsay 1983 (`10.1214/aos/1176346059`), Groeneboom et al. 2008 (`10.1111/j.1467-9469.2007.00588.x`). **Still banked. Still unauthorised.** |

## 1.5 How HARP differed from the dead OPG solver

The dense OPG solver had already failed all three repair routes (floor non-monotone; norm-cap falsified pre-build;
λ-ratio precondition non-separable). HARP's claim to be outside that family
(`JOLT-HARP-PHASE1-DESIGN.md:142-146`):

| Dead mechanism | HARP's stated distinction | Held up? |
|---|---|---|
| Eigenvalue **floor** | A floor leaves a non-zero null component. HARP emits **exactly zero** in every discarded mode. | **Yes** — genuinely different, and never falsified as a *mechanism*. Red team banked this explicitly (`JOLT-HARP-RED-TEAM.md:280-281`). |
| Step-norm **cap** | A cap preserves every direction and rescales. HARP changes the **direction** by orthogonally deleting unsupported components. | **Yes.** And §4.2 shows the failure really was directional, not length. |
| `λ_min` **whole-solver gate** | The failed gate chooses dense-all or diagonal-all from one scalar. HARP makes an independent keep/drop decision **per eigenmode**, recomputed at each accepted base. | **Yes.** Red team banked the "stale warm-seed-only" attack as not landing. |
| **Dense / BFGS** solve | Those invert every local direction including near-null ones. HARP's inverted spectrum has a pre-registered finite condition bound. | **Partly.** True for raw `H_r`; **false** for the `D`-scaled Phase-1 formula (red finding 2). |
| **Loose tolerance** | HARP retains `tol=1e-7`; it changes the tangent problem rather than declaring convergence earlier. | **Yes for WS1.** WS1.5 deliberately ran the efficacy sweep at `JOLT_IR_TOL=1e-2` because `conv` is unreachable at `1e-7` (finding **F4**, §8). |

**What actually killed it was none of these.** It was the same *shape* — a non-monotone, dataset-specific response to
a single scalar knob. That is the third occurrence of the pattern (dense-solver floor ladder → WS1.5 τ sweep →
`174328851` fine grid). See §6.3.

---

# 2. BASIS AND NOTATION — read this before quoting any spectrum number

**This is the most confusable material in the whole document set.** The same avian matrix is reported in two
different bases, and the numbers differ by ~11 orders of magnitude.

## 2.1 The chart

- **Coordinates `(y, z)`**: `y_c = log(rate_c)` (log-rate), `z_c = ` softmax logit of weight `w_c`.
  Full parameter vector `θ = (y_0..y_{k-1}, z_0..z_{k-1})`, dimension `2k`.
- **`Q = diag(I_k, Helmert_k)`**: the Helmert projector that removes **only** the softmax-logit null direction.
- **`n = 2k-1`** is the **GPU fixed-branch local step dimension**. It removes only the softmax null and lets the
  common rate scale move jointly with branch lengths.
- **`2k-2`** is the **nominal FreeRate model dimension** (after both the weight simplex *and* the joint
  rate/branch-scale gauge). HARP operates in the `2k-1` step space and never claimed the `2k-2` shorthand
  (`JOLT-HARP-PHASE1-DESIGN.md:25-28`; red team banked this as correct,
  `JOLT-HARP-RED-TEAM.md:288-289`).
- avian `GTR+R6`: `k = ncat = 6` ⇒ `nr = 2k-1 = 11`. DNA/AA `R4`: `k = 4` ⇒ `nr = 7`.

## 2.2 The two matrices

| | **RAW / PHYSICAL** | **D-SCALED / CORRELATION** |
|---|---|---|
| Matrix | `H_r = Qᵀ H Q` | `C = D^(-1/2) H_r D^(-1/2)`, `D_i = max((H_r)_ii, 1e-12·max_j (H_r)_jj)` |
| Units | mixed — log-rate and logit blocks have different natural scales | dimensionless correlation |
| Log tag | `[HARPSPEC-EIG]`, `[HARPSPEC]` | `[HARPSPEC-DSCALED]`, `[OPGLMIN]` |
| Used by | **WS1 / WS1.5 keep decision** (state-changing) | **diagnostic only** — the shipped λ_min identifiability diagnostic, and the T1/T2 cross-check |
| Status | live in the closed line | shipped separately as a diagnostic; the *step* built on it was killed pre-build |

## 2.3 EVERY avian number, labelled

All from job **`174235905`**, `gems-verify/harpspec_174235905/hs_av.console`, avian-1M `GTR+R6`, seed 12345,
warm seed, `maxiter=0` (read-only).

| Quantity | **RAW `H_r`** | **D-scaled `C`** | Which docs quote it |
|---|---|---|---|
| `λ_max` | **1.677738555e+06** | **6.560828e+00** | ULTRA-REVIEW R3 quotes the D-scaled `6.560828`; WS2 §2.2 quotes the raw `1.678e6` |
| `λ_min` | **7.893055471e-09** | **6.220391e-13** | — |
| `λ_min/λ_max` ratio | **4.704580e-15** | **9.481107e-14** | PHASE1 §0 and ULTRA-REVIEW §4 quote **9.481107e-14** (D-scaled, the archived `[OPGLMIN]` value); WS15 §A.1 quotes the **raw** ladder |
| `rank_kept` @ τ=1e-4 | **6** | **6** | the basis-invariance result (§4.1) |
| `rank_drop` @ τ=1e-4 | **5** | **5** | — |
| `margin_min` @ τ=1e-4 | **3.153e-05** | — | WS15 §A.4 |

**The full raw avian eigen-ladder** (`λ_i/λ_max`, raw basis, sorted; `[HARPSPEC-EIG]` prints them in solver order —
re-sorted here):

```
1.000000e+00 | 1.154225e-01 | 3.744039e-02 | 6.999864e-03 | 1.347093e-03 | 1.315260e-04
             | 2.757704e-06 | 2.676108e-08 | 2.412987e-10 | 1.377864e-12 | 4.704580e-15
```

For comparison, the well-identified controls (raw basis, same job): DNA `GTR+R4` `λ_min/λ_max = 8.085954e-03`,
AA `LG+R4` `= 7.555956e-03` — both **full rank at τ=1e-4** (`rank_kept=7/7`, `rank_drop=0`).

> ### ⚠️ CHART CAVEAT — invariance is only *within* this chart
> IQ-TREE's own `ratefree.cpp` parameterises `+R` as **ratio-to-last-category**, not log-rate + softmax logits. Every
> spectral finding in this document set is **chart-specific to the `(y,z)` chart**. The Helmert basis-invariance check
> (§4.1) establishes invariance only *within* that chart's reduction — raw vs D-scaled — **not across chart choices**.
> A ratio-to-last-category parameterisation would produce a different `H`, a different spectrum, and possibly a
> different effective rank. Source: `JOLT-HARP-WS2-DESIGN.md:223-227`. **State this wherever the spectrum is quoted.**

---

# 3. EVIDENCE SPINE — the four gates, in order

| # | Gate | Job | Binary md5 | Cells | Headline | Verdict |
|---|---|---|---|---|---|---|
| **1** | **`[HARPSPEC]`** read-only spectrum probe | **`174235905`** (gpuhopper, 14:50 wall) | **`d6b9e4d0`** (`iqtree3-harpspec/build-harpspec/iqtree3`, verified on disk) | avian-1M `GTR+R6`; DNA-100k `GTR+R4`; AA-100k `LG+R4`; all seed 12345, `-n 0`, `maxiter=0`, **no step applied** | avian `rank_kept=6 rank_drop=5`, `cutoff_ambig=0`, no clamp, all 10 numerical gates at machine precision (`recon=1.76e-15`, `leak=7.32e-16`, `qq=4.44e-16`). DNA/AA correctly drop nothing (`7/0`). | **PASS** — avian `GO=PASS`, basis-robust. DNA/AA `GO=FAIL` (correct: nothing to drop). |
| **2** | **WS1** one-shot rank-projected step at `it==1` | build `174254673` → gate **`174266861`** | **`583cec03`** (`build-harpws1/iqtree3`, verified on disk) | avian `GTR+R6`, DNA `GTR+R4`, AA `LG+R4`, seed 12345, nt12, `tol=1e-7` | The projected step was **mathematically immaculate and lost catastrophically**: `ln_A = -11573973.124573` vs `ln_B = -11550559.197332` ⇒ **A lost by −23,413.927 nats**. avian still `reason=maxiter it=401`. | **NO-GO.** |
| **3** | **WS1.5** runtime-τ, fire every eligible `it≥2`, DYMAX cap | **`174323861`** | **`8f8ce05e`** (`build-harpws15/iqtree3`, verified on disk) | avian/DNA/AA × τ ∈ {1e-4, 1e-3, 5e-3, 1.5e-2, 7e-2}, plus OFF baselines, kill-switch, probe byte-identity, CPU cross-check | **All safety pre-gates PASS.** One striking cell: avian τ=1e-3 → **+49.74 nats**, `it=229` vs OFF `400`, `exit=canonical_conv` — **the first avian `+R` convergence ever recorded** — at **0.598× loop wall**, `cos_mean=0.3789`, `dyclip=2/228`. **But 4 of 5 τ lost lnL** (1e-4 −41.62 \| 5e-3 −95.61 \| 1.5e-2 −273.24 \| 7e-2 −24.43). | **AMBIGUOUS** → escalate to a robustness gate. |
| **4** | **Robustness / τ-transfer** — fine τ grid × 4 cells, **fixed `-te` tree**, matched OFF | **`174328851`** ARM 3 | **`8f8ce05e`** (same binary) | avian R6, avian R8, DNA R4, AA R4 × τ ∈ {5e-4, 7e-4, 1e-3, 2e-3, 3e-3}, all `JOLT_IR_TOL=1e-2` | τ=1e-3: avian R6 **+88.72** / avian R8 **−705.62** / DNA **+0.0000** / AA **+0.0000**. The R6 win **reproduced and grew** on a fixed tree; it **does not transfer**. | 🔴 **FAIL ⇒ HARP CLOSED** by the pre-registered rule. |

### The pre-registered rule that fired, verbatim

From `gadi-ci/gems/gems_tolladder_taurobust.sh`, written into the script **before** the run:

```
════ DECISION RULE (pre-registered) ════
tau=1e-3 is REAL only if it is non-negative on lnL across ALL FOUR cells (dlnL >= -0.05) AND reduces iterations on
at least the two avian cells. Anything else and it is the n=1 artifact the dense-solver post-mortem predicts:
  * wins on avian R6 only, loses elsewhere      => post-hoc fit on one cell. HARP CLOSES.
  * peak is a SPIKE (5e-4 and 2e-3 both lose)   => untunable, same shape that killed the dense solver. HARP CLOSES.
```

**Both clauses fired.** avian R8 lost 705.62 nats at the winning τ; and 5e-4 (−133.33) and 2e-3 (−248.60) both lose on
avian R6, making the peak a spike.

---

# 4. THE FOUR RESULTS THAT SURVIVE

These are the durable, reusable findings. Everything else in the nine documents is process.

## 4.1 The avian `+R6` spectrum — measured, verified four ways, in both bases

Job **`174235905`**, binary `d6b9e4d0`. The full raw eigenpairs with their gradient projections `a_i = v_iᵀ g_r`
(raw basis, sorted by λ descending; `keep` at τ=1e-4):

| rank | `λ_i` (**RAW**) | `λ_i/λ_max` (**RAW**) | `\|a_i\|` (**RAW**) | keep @ τ=1e-4 |
|---|---|---|---|---|
| 1 | 1.67773855489e+06 | 1.000000e+00 | 2.02169509026e+05 | ✔ |
| 2 | 1.93648819702e+05 | 1.154225e-01 | 7.46339526335e+04 | ✔ |
| 3 | 6.28151924297e+04 | 3.744039e-02 | 2.13315654300e+04 | ✔ |
| 4 | 1.17439423539e+04 | 6.999864e-03 | 1.49092699545e+04 | ✔ |
| 5 | 2.26007036150e+03 | 1.347093e-03 | 3.04888337129e+03 | ✔ |
| 6 | 2.20666218091e+02 | 1.315260e-04 | 3.62090916088e+02 | ✔ |
| 7 | 4.62670574320e+00 | 2.757704e-06 | 3.71568490755e+01 | ✘ |
| 8 | 4.48980955464e-02 | 2.676108e-08 | 3.61736271037e+00 | ✘ |
| 9 | 4.04836139471e-04 | 2.412987e-10 | 1.24847918587e-01 | ✘ |
| 10 | 2.31169597633e-06 | 1.377864e-12 | 1.70788682970e-03 | ✘ |
| 11 | 7.89305547141e-09 | 4.704580e-15 | 1.86630361605e-04 | ✘ |

**In the D-scaled basis** the same matrix has `λ_max = 6.560828e+00`, `λ_min = 6.220391e-13`,
`ratio = 9.481107e-14` — and **the keep-set is still exactly 6**. Raw drops the bottom 5 of `[7.9e-9 … 1.68e6]`;
D-scaled drops the bottom 5 of `[6.2e-13 … 6.56]`.

**Four-way verification** (`JOLT-HARP-HARPSPEC-RESULT.md:24-32`):
1. **Code red-team** Q2–Q6 clean (gradient basis byte-matches the LM loop; rank computed on raw `H_r`; leakage on the
   *final applied* step; byte-identity OFF; indexing consistent). One defect, **Finding 1**, found and fixed —
   ungated `maxiter=0` would have neutered non-`+R` candidates on a multi-model `-m MF` run; fix
   `if (g_harpspec && opgOK) maxiter = 0;` proven complete (probe-ran ⟺ `opgOK && g_harpspec`).
2. **Offline verifier** `harpspec_verify.py` — independent Jacobi, no numpy, parses only the raw `[HARPSPEC-MAT]` /
   `[HARPSPEC-GR]` dumps and recomputes rank@τ from scratch: **PASS on all 3 cells** (avian 6=6, DNA/AA 7=7);
   trace/reconstruction/residual invariants at 1e-15.
3. **Statistical red-team** raised T1–T4 (see §7).
4. **Decisive empirical tests** falsified T1/T2 *on avian* (below) and bounded T3 (`‖g_r‖²/N = 4.7e4` vs
   `λ_max = 1.68e6` ⇒ gradient contributes **2.8%** of λ_max ⇒ curvature-dominated).

**Build gates for `174235905`**, all green: proof-of-build (7 `[HARPSPEC` sentinels; canonical `opg3` binary
`9b6b4519` = 0); `[OPGLMIN]` **bit-exact** (`ratio=9.481107e-14`, the archived value ⇒ shared reduction unperturbed);
OFF-identity exact (`off_new` = `off_can` = `-11216886.2301`).

## 4.2 Directional misallocation — same curvature budget, 9× the parameter movement

This is the single most valuable diagnostic result in the line. Source `JOLT-HARP-WS2-DESIGN.md:38-52`, from the WS1
gate `174266861`.

| | canonical **B** (`it==1`) | HARP **A** (WS1, τ=1e-4) | ratio |
|---|---|---|---|
| `‖δ‖²_H = δᵀH_rδ` | 6.968e4 | 8.401e4 | **1.1×** |
| implied ½δᵀHδ ("KL") | 34,842 nats | 42,003 nats | 1.2× |
| **max\|dy\| (log-rate)** | **0.1914** (rate ×1.21) | **1.7124** (rate ×5.54) | **9×** |
| outcome | accepted | **−23,413.927 nats** | — |

A and B spend **essentially the same curvature budget**, yet A converts it into **9× more parameter movement** by
concentrating on low-λ modes. In parameter space WS1 proposed to restructure the entire mixture in one step: rate
c=0 ×0.18, rate c=2 ×3.55, weight `w0` 0.167→0.374, `w1` 0.167→0.068. It passed eligibility because the only bounds
were the box `[1e-4, 1000]` and `w > 1e-4` — **bounds that never constrain step trustworthiness.**

> **Consequence — a Fisher/KL trust region would NOT have caught this.** A TRPO-style cap
> `α = √(2Δ/δᵀHδ)` is *self-referential*: it is calibrated by the very matrix that is wrong, and both steps have
> essentially the same H-norm. This **falsified the project's own first hypothesis** and is why the (never-built) WS2
> design did not lead with it.

*Corroboration from a different job:* WS1.5 telemetry adds a `maxdy_max` field. Job `174328851`,
`a3_1e-3_av6.console` reports `maxdy_max=1.716844` — the max over 333 firing iterations, consistent with WS1's
single `it==1` value of 1.7124.

## 4.3 The discrete Picard violation — the **measurement** survives, the **inference** did not

Over the 6 kept modes, `|a_i|/λ_i` **grows** as λ falls (4 of 5 consecutive steps growing). λ spans
`1.678e6 → 2.207e2` (÷7,603) while `|a_i|` spans `2.022e5 → 3.621e2` (÷558) — **`|a|` decays 13.6× slower than λ**.
All values **raw basis**, job `174235905`; I recomputed every ratio from the `[HARPSPEC-EIG]` dump and they reproduce.

| rank | `λ_i` (raw) | `\|a_i\|` (raw) | `\|a_i\|/λ_i` | |
|---|---|---|---|---|
| 1 | 1.678e6 | 2.022e5 | 1.205e-1 | |
| 2 | 1.936e5 | 7.463e4 | 3.854e-1 | ↑ |
| 3 | 6.282e4 | 2.133e4 | 3.396e-1 | |
| 4 | 1.174e4 | 1.491e4 | 1.270e0 | ↑ |
| 5 | 2.260e3 | 3.049e3 | 1.349e0 | ↑ |
| 6 | 2.207e2 | 3.621e2 | 1.641e0 | ↑ |

This is the **ill-determined-rank regime** of Hansen (1990; 1998 Ch. 4) — no spectral gap, so a hard TSVD filter
`f_i ∈ {0,1}` is the wrong regulariser and a smooth Tikhonov filter is indicated.

> 🔴 **The operational conclusion drawn from this table was FALSIFIED.** WS2 §2.2 asserted *"the solution is dominated
> by the smallest kept modes, so **no value of τ fixes it**."* That is **FALSE**. The red-team of WS2 recomputed the
> hard-truncation sweep offline against the same archived spectrum (`rank_kept`, `max|dy|`):
> `τ=1e-4`→6, **1.7124** \| `1e-3`→5, 0.9473 \| `5e-3`→4, 0.8746 \| `7e-3`–`3e-2`→3, **0.3810** \| `1e-1`→2, 0.2118,
> against canonical's **accepted** 0.191376. **τ was simply set ~100× too small.**
> (`JOLT-HARP-WS2-DESIGN.md:237-240`.)
>
> Related: WS2's own κ₀=3e-2 Tikhonov setting has filter factors `0.97, 0.79, 0.56, 0.19, 0.04, 0.004` ⇒ **effective
> rank 2.56** — a *soft rank-3 truncation*, i.e. exactly the thing §2.2 argued could not work
> (`JOLT-HARP-WS2-DESIGN.md:241-243`).
>
> **Keep the measurement. Discard the "no τ works" inference.** WS1.5 then went and measured τ directly, and found
> something worse than "no τ works": τ works, but on *one cell*.

## 4.4 The eval-budget model — validated to the unit

`JOLT-HARP-WS15-PLAN.md:441-453`, checked against job `174266861`'s `[HARPEXIT]` lines. I re-read all three from disk
and they match exactly.

Model: per iteration = `nFreeQ` (free-Q finite differences) + `[optPinv==1]` (pinv FD) + 1 accepted trial + rejected
backtracks; plus one seed eval; plus WS1's A eval and B-restoration eval.

| cell | `nFreeQ` | iters | rejects | **predicted** `nLnLEval` | **measured** | terminal |
|---|---|---|---|---|---|---|
| avian `GTR+R6` | 5 | 400 (`it=401`, `maxiter=400`) | 206 | `1 + 400·5 + 400 + 206 + 2` = **2609** | **2609** ✅ | `maxiter` |
| DNA `GTR+R4` | 5 | 48 (last rejects out) | 44 | `1 + 48·5 + 47 + 44 + 0` = **332** | **332** ✅ | `reject_stall` |
| AA `LG+R4` | 0 | 44 | 28 | `1 + 0 + 44 + 28` = **73** | **73** ✅ | `canonical_conv` |

The `+2` on avian is WS1's A eval **plus** its B-restoration eval — the exact confirmation that an A-loss costs
**+2** evals, not +1.

Derived quantities that any future `+R` step experiment can reuse directly:
- **evals per iteration `n_off`**: avian **6.515**, DNA **6.917**, AA **1.636**.
- ⇒ **AA can only ever be a SAFETY cell, never an efficacy cell**: `Δn/n_off = 1.222` ⇒ per-iteration cost ratio
  `ψ ≈ 2.22×`. Break-even would need `I_on ≤ 20` against its 44 — a 55% cut. Not credible.
- ⇒ **DNA R4 ends `reject_stall@48`** — a shorter reject-stall is not a win.
- ⇒ **avian `GTR+R6` is the only efficacy cell that exists.**
- **Gram/eigen overhead is NOT the cost, and this is proven**: WS1's ON/OFF wall ratio was **1.005×**
  (26.592 / 26.453 s, job `174266861`) *while building the Gram on all 400 iterations*.
  `[OPGCOST] mean_ms = 0.1253` × 400 = 50 ms of a 23.95 s loop = **0.2%**. 400 host-side 11×11 Jacobi eigensolves add
  O(100 µs) each ⇒ ≤40 ms. **All of the cost is the extra likelihood evaluations.**

## 4.5 The two lessons

> ### ⭐ Lesson 1 — Converging FASTER is not evidence of a better step.
> Job `174328851`, avian `GTR+R8` at τ=1e-3: finished in **139 iterations** instead of OFF's **342**, at
> `reason=canonical_conv`, at a point **705.62 nats worse**. Fast **and** confidently wrong, reporting clean
> convergence, with **nothing downstream to flag it**. A speed gate that reads `iterations` and `exit=CONV` and stops
> there will certify this as a win.

> ### ⭐ Lesson 2 — an OSCILLATING knob response means UNTUNABLE, not under-tuned.
> Job `174328851` ARM 3, `dlnL` vs matched OFF, fixed `-te` tree, `JOLT_IR_TOL=1e-2`, τ = 5e-4 … 3e-3:
>
> | τ | 5e-4 | 7e-4 | 1e-3 | 2e-3 | 3e-3 |
> |---|---|---|---|---|---|
> | **avian R6** | −133.33 | +82.19 | **+88.72** | −248.60 | +63.62 |
> | **avian R8** | −438.34 | −438.34 | **−705.62** | −117.61 | +38.37 |
> | DNA R4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.046 |
> | AA R4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
>
> The two avian cells **peak at different τ**. This is the **3rd occurrence** of this shape (dense-solver floor ladder
> → WS1.5 τ sweep → this fine grid), confirming `FREERATE-CONDITIONING-AND-IDENTIFIABILITY.md`'s *"no constant
> generalises"* law. τ=3e-3 is the only non-negative-everywhere arm (+63.62 / +38.37 / +0.046 / 0) but it saves **no
> work on avian R6** (`it=401`, `reason=maxiter`, `conv=0`) ⇒ **nothing shippable.**

---

# 5. ENGINEERING WORTH KEEPING

This section is the reusable payload. HARP is closed; these artifacts are not.

## 5.1 ⭐ THE DEVICE-COHERENCE TRANSACTION — designed, proven, and MEASURED WORKING

**This is the most reusable thing the line produced.** Any future experiment that compares two `+R` proposals at the
same base on the GPU needs it, or it will silently corrupt state.

### The defect

`evalLnL` is the only likelihood evaluator. It (a) stages host params, (b) `rebuildEchild()` uploads `d_echild` from
`brlen*catRate`, (c) runs the postorder over tiles, and (d) sets residency keys `devB/devA/devP/devValid=true`,
leaving `d_echild` / `d_partial` / `d_patlh` describing **its last call**.

**Its residency key records only branch lengths, alpha, and pinv. It omits the FreeRate `(y,z)`.**

Therefore an A/B pair that differs **only** in rates and weights — which is exactly what a same-base `+R` step
competition is — has an *identical* residency key. Consequences:

- `computeGradient` trusts that incomplete key for fixed-Q models and can skip both `rebuildEchild` and, at
  `nTile=1`, `postorderFill` ⇒ the **next gradient is computed at a mixed state**.
- `out_patlh` snapshot-on-accept `cudaMemcpy`s `d_patlh` — i.e. it copies **whatever proposal was evaluated last on
  the device**. A naive "evaluate A, then B, then restore host winner A" leaves **`d_patlh` describing the LOSER**,
  and the saved per-pattern vector — the one bootstrap support is computed from — is deterministically wrong.
- Free-Q GTR avoids the *next-gradient* reuse (`devMatch` is disabled when `nFreeQ>0`) but the **losing `d_patlh`
  problem remains**. Fixed-Q LG takes the full mixed-state path.
- The CPU self-check does **not** catch this: it can pass after later consistent evaluations without proving that the
  intermediate gradient or the saved pattern vector came from the accepted proposal.

Found by the hostile red team as finding 1 (`JOLT-HARP-RED-TEAM.md:17-52`), confirmed independently by blue
(`JOLT-HARP-BLUE-TEAM.md:41-44`) and accepted verbatim in reconciliation
(`JOLT-HARP-RECONCILIATION.md:19-22`).

### The fix — an ordering discipline, not a lock

```
1. Capture an immutable complete base (branches, alpha, pinv, Q, meanR, bprop, zR,
   all gradients/secants, the Gram/eigen data).
2. Build the common non-(y,z) arms — brlen, alpha, pinv, Q — SHARED by A and B.
3. Evaluate canonical B FIRST. Save its complete proposal and likelihood. Do not accept yet.
4. Restore the base host mixture/Q state.
5. Evaluate HARP A LAST, if eligible.
6. Winner: A wins iff finite(ln_A) && ln_A > ln_B && ln_A > lnL + 1e-9.
   Else B wins iff ln_B > lnL + 1e-9. Else neither.
   ── EXACT TIE GOES TO B. Never A. ──
7. Commit so the winner is the LAST DEVICE EVAL:
   • A wins            → A already is the last eval. Commit A on the host. Snapshot reads coherent A d_patlh.
   • B wins AFTER A ran → device holds A. RE-STAGE AND RE-EVALUATE B, then commit. (count a "restoration eval")
   • B wins, A skipped  → B was the only and last eval. Commit directly, no re-eval.
8. Neither wins → restore base, mu *= 4, next backtrack. Self-healing: the NEXT backtrack's B eval
   re-syncs the device before anything reads it.
```

### Why the alternatives were rejected

- "Merely invalidating the cache is insufficient because `out_patlh` can still belong to the loser. **The winner must
  be the last fully evaluated proposal before accept/snapshot.**" (`JOLT-HARP-BLUE-TEAM.md:41-44`)
- "**No host-only winner restore is permitted.**" (`JOLT-HARP-RECONCILIATION.md:22`)

### It was MEASURED WORKING

Gate `174266861` verified: *"the device-coherence A/B transaction (winner is always the last device eval,
B-restoration eval, reject self-heal) works exactly as designed"* (`JOLT-HARP-WS2-DESIGN.md:14-17`). The
`[HARPEXIT]` line for avian records `A_tried=1 A_acc=0 B_acc=1 restoreEval=1` — B won after A ran, and the
restoration eval fired, exactly once, as designed. The eval-budget model (§4.4) reproduces `nLnLEval=2609` **only if**
that restoration eval is counted, which is independent confirmation that it executed.

WS1.5 (`174323861`, `174328851`) then ran the same transaction **hundreds of times per invocation** (e.g.
`fires=398 A_tried=256 restoreEval=188`) with no coherence failure and no CPU-fallback event in any arm.

**Reusability:** this discipline is orthogonal to HARP's step policy. It applies to any GPU `+R` A/B experiment —
Tikhonov, line search, EM, or a future true-Fisher step. Preserve it.

## 5.2 The 10 probe checks (`[HARPSPEC]`)

Blocking numerical gates, all measured at machine precision on all three cells in job `174235905`:

```text
‖QᵀQ − I‖_inf                                 ≤ 1e-12     measured 4.44e-16
‖VᵀV − I‖_inf                                 ≤ 1e-10     measured 4.44e-15 (avian)
max normalized eigen residual                 ≤ 1e-10     measured 1.13e-15 (avian)
relative reconstruction error                 ≤ 1e-10     measured 1.76e-15 (avian)
λ_min                                         ≥ −1e-10·λ_max      psd=1
max_drop |v_iᵀδ_r| / max(1,‖δ_r‖₂)            ≤ 1e-12     measured 7.32e-16 (avian)
|λ_i/λ_max − τ|                               > max(1e-6, 100·r_eig), every i   cutoff_ambig=0
g_rᵀδ                                         > 0 and finite
quadratic predicted gain                      > 0 and finite
predicted rate clamp / weight floor           == false
```

⚠️ **The last two are decorative** — see threat **T4** in §7. `grdd = δᵀHδ = Σ a_i²/λ_i` and `qgain = ½·that` are
**algebraic tautologies** of a PSD pseudo-inverse step. The gate observed exactly `grTd=8.400519e+04`,
`qgain=4.200260e+04 = grTd/2`. **The quadratic model is structurally incapable of predicting a bad step.** This is
precisely why the trust-region literature puts the *actual* reduction in the numerator of ρ. Never treat them as
ascent evidence.

## 5.3 The 6 WS1 invariants (`JOLT-HARP-RECONCILIATION.md:88-95`)

Required **before A can be eligible**:

```text
‖QᵀQ − I‖_F                             ≤ 1e-12
‖VᵀV − I‖_F                             ≤ 1e-10
‖H_r V − V Λ‖_F / max(1,‖H_r‖_F)        ≤ 1e-10
‖H_r − V Λ Vᵀ‖_F / max(1,‖H_r‖_F)       ≤ 1e-10
max_drop |v_iᵀ Qᵀ δ|                    ≤ 1e-12 · max(1,‖δ_r‖₂)
‖Qᵀ δ − δ_r‖₂                           ≤ 1e-12 · max(1,‖δ_r‖₂)
```

**The design point that matters:** the last two **reproject the FINAL APPLIED `(y,z)` step**. They do *not* inspect
the pre-map zero slots. This directly answers red finding 6 — a `null_step_norm == 0` diagnostic that sums the
already-zeroed coefficient array is **self-fulfilling** and proves only that an assignment happened, not that the
applied step is free of discarded-subspace leakage.

## 5.4 Proof-of-build discipline

Every build gate in this line asserted a sentinel **stock cannot fake**, using **measured counts** (`grep -c`), never
`grep -q`:

| Gate | Sentinel(s) | Negative control |
|---|---|---|
| `174235905` | 7 × `[HARPSPEC` | canonical `opg3` binary `9b6b4519` → **0** |
| `174266861` | `JOLT_HARP_WS1_TAU_1E-4`, `[HARPSTEP]`, `[HARPEXIT]` | canonical → 0 |
| `174323861` | `JOLT_HARP_WS15_TAUENV`, `JOLT_HARP_TAU`, `JOLT_HARP_DYMAX`, `[HARPCFG`, `[HARPEXIT` | **WS1 binary `583cec03` → 0** and canonical `9b6b4519` → 0 |

Plus a **measured-effect** proof no static string can fake: job `174323861` ran avian at `JOLT_HARP_TAU=1e-4` and
`3e-1` and required different `rank_kept` (8 vs 4). A binary that ignored the env would have printed the same rank
twice. The `[HARPCFG]` line prints the value **actually in force** — *"never trust the exported env, read the log"*.

Also: `JOLT_HARP_TAU` out-of-range values are **rejected** (default retained + printed), not clamped, because a
typo'd sweep arm that silently disables HARP would look like a clean negative.

**Byte-identity OFF was proven, not assumed**, at every stage: `off_new == off_can == -11216886.2301`
(`174235905`); the `[OPGLMIN]` reduction stayed bit-exact at `9.481107e-14` (D-scaled) proving the shared reduction
was unperturbed; and the `[HARPSPEC]` probe dump was diffed byte-for-byte against the archived
`harpspec_174235905/hs_av.console` in the WS1.5 gate — **run with `JOLT_HARP_TAU=1.5e-2` also exported**, to prove
positively that the probe is pinned at its hard-coded 1e-4 and cannot be perturbed by the sweep.

## 5.5 S1 / S2 / S3 — what replaced "the controls are inert"

WS1 asserted that the DNA/AA R4 controls were **inert** (ineligible ⇒ clean fall-back to canonical). WS1.5 finding
**F5** showed that assertion **cannot be kept** once HARP fires every iteration: inertness is a property of *one
measured spectrum at one iterate*, not a code invariant — `computeGradient` rebuilds `opgH` every iteration, so the
spectrum at `it=37` is not the spectrum at `it=1`. Asserting `rank_drop==0` for 400 iterations at a τ sitting
`2.6e-4` from the AA arming threshold would false-pass or false-fail on drift — exactly the *"a gate must prove its
control is a control"* failure mode.

The replacement is **stronger**:

- **S1 — Behavioural safety** (every cell, every arm): `lnL_ON ≥ lnL_OFF − 0.05` **and** no CPU fallback **and**
  every accepted-A backtrack clean (`psd=1, cutoff_ambig=0, basis_ok=1, aClamp=0`).
  *Inertness proves "nothing happened"; S1 proves "whatever happened did not hurt".*
- **S2 — `A_acc` accounting** (every cell): `A_acc + B_acc == committed iterations` and `A_tried ≥ A_acc`, read off
  the aggregated `[HARPEXIT]`. Catches a mis-wired winner rule.
- **S3 — EXACT structural identity** on any cell where `A_tried == 0`: when A never runs the transaction degenerates
  to the canonical path with the same `mu` ladder and the same accept predicate. So assert **`it`, `nRej` and
  `nLnLEval` all bit-equal to the OFF arm**, not merely `|ΔlnL| < 1e-6`.
  *WS1's own numbers show the target (DNA `it=48 nLnLEval=332 nRej=44`; AA `it=44 nLnLEval=73 nRej=28`) — but the
  WS1 gate never ran the OFF arm with `--jolt-diag`, so it could not make this assertion.* Fixed in the WS1.5 gate.

**S3 is the general lesson: when a feature is supposed to be inert, assert bit-equality of the counters, not
approximate equality of the answer.**

---

# 6. WHY IT FAILED — the causal chain

## 6.1 WS1 was structurally incapable of achieving its own goal

Independent of the overshoot. Source-verified in `JOLT-HARP-WS2-DESIGN.md:23-30`:

- **The firing point was wrong.** `ddY`/`ddZ` are initialised to **−1e6** and only receive real secant curvature
  `if(haveSec)`, with `haveSec=true` set at the **end** of the iteration. So at **`it==1` the canonical model arms
  are deliberately near-frozen**: B's model step is `g_y/(1e6+μ)`. **WS1 competed a full undamped Newton step against
  an intentionally frozen baseline.** That is not a fair same-base competition; it is a competition against a
  deliberately hobbled incumbent, and A still lost by 23,414 nats.
- **One step out of 401 cannot move a 401-iteration metric.** The mission metric was avian 401 → ≤320 joint
  iterations. A one-shot step at `it==1` is ~0.25% of the budget. **WS1 could not have passed its own gate even if
  the step had been perfect.**
- **The step was ~9× too long** (§4.2), and its eligibility bounds (`rate ∈ [1e-4,1000]`, `w > 1e-4`) do not
  constrain trustworthiness.

The one thing WS1 *did* establish beyond doubt: **the algebra was immaculate.** `harp_project` reproduced the probe's
residuals at runtime (`qq=4.44e-16 leak=7.32e-16 recon=1.76e-15`, identical to the archived probe) ⇒ "probe predicts
A" holds. The failure was never numerical.

## 6.2 WS1.5 showed τ IS load-bearing — but it does not transfer

WS1.5 fixed the firing point (`it≥2`, every eligible iteration), made τ runtime-tunable, added a curvature-independent
`DYMAX` magnitude cap, and added a `defer` rule so a sub-tol A-win cannot pin the loop at `maxiter`.

**It worked. Once.** Job `174323861`, avian `GTR+R6`, τ=1e-3:

| | |
|---|---|
| lnL | `-11216836.494` vs OFF `-11216886.230` ⇒ **+49.74 nats** |
| iterations | **229** vs OFF **400** |
| exit | **`canonical_conv`** — *the first avian `+R` convergence ever recorded*, instead of truncation |
| loop wall | **0.598×** (16.558 s vs 25.994 s total; `C ≈ 2.46 s` ⇒ 14.10 s vs 23.53 s loop) |
| `cos_mean` | **0.3789** — a **genuinely different direction**, not a rescaled canonical step |
| `dyclip` | **2 / 228 fires** — the DYMAX cap barely engaged ⇒ **τ did the work**, not the magnitude cap |

That last pair of numbers matters: it rules out kill-criterion **K5** ("HARP is just a rescaled canonical step"). The
win, such as it was, came from the *direction change* that truncation produced.

**And then it did not transfer.** Job `174328851`, fixed `-te` tree, matched OFF, same τ=1e-3:

| cell | dlnL vs OFF | iters (OFF) | conv |
|---|---:|---:|:--:|
| avian R6 | **+88.72** | 333 (400) | yes |
| avian R8 | 🔴 **−705.62** | 139 (342) | yes |
| DNA R4 | +0.0000 | 31 (31) | inert |
| AA R4 | +0.0000 | 20 (20) | inert |

**The WS1.5 win itself REPRODUCED — and grew** (+88.7 on a fixed tree vs +49.7 on a per-seed PARS tree). It is a real
effect on avian R6. It simply **does not generalise to avian R8, the nearest possible neighbour** — same alignment,
same tree, same τ, same binary, same job, one more rate category.

## 6.3 The deeper reason — the empirical Fisher is not the curvature away from the optimum

This is the mechanism, and it is why no amount of τ or damping was ever going to work.

`H = Σ_p freq_p · s_p s_pᵀ` is the **empirical Fisher** (OPG). Its equality with curvature requires **correct model
specification AND evaluation at the optimum**. We satisfy neither: `GTR+R6` on avian is misspecified, and the warm
seed is by construction *not* the optimum.

- **Kunstner, Balles & Hennig (NeurIPS 2019)**, *Limitations of the empirical Fisher approximation*: the equality
  "only holds close to the minimum"; away from it, EF-preconditioned steps have magnitude ~inversely proportional to
  ‖g‖ and directions that can be **near-orthogonal to the true natural gradient**. ⚠️ **They warn explicitly that
  damping fixes MAGNITUDE, not DIRECTION.**
- **Martens (JMLR 2020) §11**: the empirical Fisher is *"considerably less useful as an approximation to the Fisher,
  or as a curvature matrix"*; the Fisher↔Gauss-Newton equivalence does **not** extend to the empirical variant. §10:
  natural-gradient invariance holds only for *infinitesimal* steps ⇒ at α=1 there is no invariance guarantee.
- **White (Econometrica 1982)**: the information-matrix equality `A = −B` holds only under correct specification at
  the true parameter. The OPG estimates `B`; we use it as `−A`. Under a misspecified `GTR+R6` they differ **even at
  the MLE**.
- **BHHH (1974)**: our algorithm *is* BHHH — inverse-OPG preconditioned ascent — **with the line search deleted**.
  The econometrics literature has known since the 1970s that OPG "performs poorly far from the maximum".

**Put together:** §4.2 measured that our failure *is* directional (same H-norm, 9× the movement). Kunstner et al. say
damping cannot fix direction. Therefore **the entire Tikhonov/damping family — WS2's core proposal — was predicted to
fail before it was written**, and hard truncation (WS1/WS1.5) can only change direction by *deleting* modes, which is
a one-parameter family that we then measured to be non-monotone and dataset-specific.

Note the incumbent's status in this light: the canonical `g_j/(|d_j|+μ)` step is a **diagonal-scaled
Levenberg–Marquardt step** — i.e. already the method the optimisation literature endorses. **WS1 was a strictly
*less*-regularised version of the incumbent.**

---

# 7. THREAT REGISTER — every ID, with verdict AND scope

**Read the SCOPE column.** Most "falsified" entries are falsified **on avian, at one iterate, at one τ**. None is
falsified as a general mathematical statement.

| ID | Source | Threat | Verdict | 🔬 **SCOPE of that verdict** |
|---|---|---|---|---|
| **T1** | HARPSPEC stat red-team | Raw-spectrum **block-scale bias** — a single τ on the raw matrix favours the higher-curvature block | **FALSIFIED** | **avian `GTR+R6` only, at the `it==1` warm seed, at τ=1e-4, job `174235905`.** Raw and D-scaled keep-sets both = 6. ⚠️ **Along the trajectory the bases DO diverge**: job `174328851` `a3_1e-3_av6` reports `basis_ok=0` with `rank_drop=3` vs `dkeep=9` — i.e. at some firing iteration the raw and D-scaled keep-counts disagreed by 6 modes. **The invariance is a warm-seed property, not a trajectory property.** |
| **T2** | HARPSPEC stat red-team | τ=1e-4 was **calibrated on the D-scaled gap** but **applied to the raw spectrum** | **FALSIFIED** | Same scope as T1, and the same caveat applies. The calibration mismatch is real; it just did not change avian's *warm-seed* rank decision. |
| **T3** | HARPSPEC stat red-team | OPG at a **non-stationary warm seed** has gradient-inflated `λ_max` | **CONFIRMED, minor on avian** | **avian only.** `‖g_r‖²/N = 4.7e4` vs `λ_max = 1.68e6` ⇒ gradient contributes **2.8%** of λ_max ⇒ curvature-dominated. Unmeasured on any other dataset. |
| **T4** | HARPSPEC stat red-team | The `grdd>0` / `qgain>0` gates are **algebraic tautologies** of a PSD pseudo-inverse step | **CONFIRMED — valid, and decorative** | **General — this is a theorem, not a measurement.** `grdd = Σ a_i²/λ_i`, `qgain = ½·grdd`, both guaranteed >0 for PSD `H`. Observed exactly: `grTd=8.400519e4`, `qgain=4.200260e4`. Never usable as ascent evidence anywhere. |
| **RED 1** | Phase-1 red team | A/B competition can accept host state A while **device state describes B** (residency key omits `(y,z)`) | **CONFIRMED — fixed** | **Source-general** (`gpu_lnl_intree.cu`). Free-Q avoids next-gradient reuse; **fixed-Q LG takes the full mixed-state path**; the losing-`d_patlh` problem is universal. Fix = §5.1, measured working in `174266861`. |
| **RED 2** | Phase-1 red team | The step **still inverts near-null directions**; `D`-scaling can amplify by ~`10¹²` | **CONFIRMED for the D-scaled formula** | **Kills the PHASE1 formula only.** WS1 eigendecomposes raw `H_r` ⇒ no `D`, no amplifier. The *knife-edge* half of the finding (a mode just above τ gets ~`10⁴` inverse) remains true for **any** hard cutoff and was never eliminated — it is what `DYMAX` was later added to bound. |
| **RED 3** | Phase-1 red team | `λ_LM` damping/backtracking algorithm **is not specified** (no init, bounds, multiplier, Nielsen state, gain-ratio, tie rule) | **CONFIRMED — pre-build blocker** | **Design-level, general.** *"There is no single algorithm for the gate to test."* Resolved by **deleting λ_LM entirely** in WS1, not by specifying it. |
| **RED 4** | Phase-1 red team | Same-base one-step dominance **does not guarantee endpoint safety**; also, a rank-restricted optimiser may not still score the nominal `+Rk` model | **CONFIRMED as a risk; never disproved** | **General.** Historically grounded on avian: one accepted dense step cost 26–52 nats at the cap; a shrunken step gave 206/214 locally-improving accepts and the *worst* endpoint (−383.6 nats). WS1.5's own §B.5 dominance theorem explicitly **does not** exclude trajectory divergence after an A-win. **This threat is what ultimately closed HARP** (avian R8, −705.62). |
| **RED 5** | Phase-1 red team | **Premature convergence can mechanically pass as the headline speedup** — a tiny accepted A step trips `dl<tol` | **CONFIRMED — mitigated, then re-confirmed in a new form** | **General.** Mitigated in WS1 (HARP iteration forbidden from setting `conv`) and refined in WS1.5 (a **B-win may** set conv since its step *is* canonical's; an **A-win defers** for one iteration). But **Lesson 1 (§4.5) is this threat resurfacing at the endpoint level**: avian R8 reported clean `canonical_conv` at 139 iterations, 705 nats wrong. |
| **RED 6** | Phase-1 red team | `null_step_norm == 0` is a **self-fulfilling, non-independent** gate | **CONFIRMED — fixed** | **General.** Replaced by reprojecting the **final applied** step (§5.3, last two invariants) plus independent eigen-residual / orthogonality / reconstruction checks. Measured `leak=7.32e-16`. |
| **RED 7** | Phase-1 red team | Gate **coverage omissions**: unspecified seeds, no per-partition verdict, no `out_patlh`/bootstrap cell, vacuous RF condition, no rank-stability check | **CONFIRMED — partly cleared** | **Gate-design.** Seeds/fixtures were frozen in reconciliation §5. ⚠️ **The partitioned cell, the bootstrap/`out_patlh` cell, the multi-seed selection gate and the `-nt 1`/`-nt 12` cells were NEVER RUN** — HARP died before reaching §5.3. See §8. |
| **RED 8** | Phase-1 red team | **OFF-path identity is empirical, not proven** by source structure; other `JOLT_OPG*` vars can still allocate/launch Gram work | **PLAUSIBLE — cleared empirically** | **Per-gate.** Every gate sanitised `JOLT_OPG*`/`JOLT_HARP*` and asserted zero HARP/Gram telemetry. Byte-identical OFF measured at every stage. The word *"provably"* remains unjustified; *"measured"* is correct. |
| **RED 9** | Phase-1 red team | **Phase-0 provenance gap** — cited job `174227868` / `ratios.tsv` md5 not locatable by the reviewer | **CLEARED** | **Artifact-specific.** Blue team located both: `gems-verify/opgp3sep_174227868/ratios.tsv` and `opgp3sep_174160263/ratios.tsv`, both md5 `8e2351518e387214ca0079b97cab0e0d`; PBS output exit 0, wall `00:07:20`, 11.00 SU. |
| **R1** | User ultra-review | A matched-R4 speedup **does not solve the real bottleneck** if avian/R8 stay at 401 | **CONFIRMED — became the mission metric** | **General; drove the whole gate design.** Escalated the bar from `wall ≤ 1.03×` to **avian ≤320 iterations + ≥10% wall cut**. |
| **R2** | User ultra-review | Per-iteration A/B would add hundreds of postorders; but a one-shot WS1 then makes `wall ≤ 1.03×` **too weak** — 401/401 can pass | **CONFIRMED** | **avian-specific arithmetic**, generalised by the §4.4 eval model. Exactly what happened: WS1 ran at 1.005× wall and 401/401. |
| **R3** | User ultra-review | In the **D-scaled** basis, `λ_max = 6.560828` makes the τ=1e-4 boundary `6.560828e-4`, undamped inverse ≈ `1.52e3` — a barely-retained mode can still generate a clamp-sized proposal | **CONFIRMED for the scaled solve; PLAUSIBLE for raw WS1** | **D-scaled basis only.** Raw `H_r` removes the two `D^{-1/2}` amplifiers. But the *knife-edge* concern transferred: WS1's raw step had `max\|dy\| = 1.7124` vs canonical's accepted 0.1914. |
| **R4** | User ultra-review | **Dump the full spectrum before any state-changing run** — only `lmin`/`lmax` were ever preserved | **CONFIRMED blocker — cleared** | **Artifact-level.** This finding is what created `[HARPSPEC]` (job `174235905`), which produced §4.1 and §4.3 — **the two results that outlived the method.** Best single decision in the line. |
| **F1** | WS1.5 blue team | Red-team's flagship `τ=7e-3` **silently self-disables** | **CONFIRMED on disk** | **avian only, exact.** avian raw eigen-ratio #4 = **6.999864e-03**; at τ=7e-3, `margin_min = 1.36e-7 < 1e-6` ⇒ `cutoff_ambig=1` ⇒ `gates=false` ⇒ `go=false` ⇒ **HARP never fires.** Replaced by τ=1.5e-2 (`margin_min=8.00e-3`, same `rank_kept=3`). |
| **F2** | WS1.5 blue team | Cost is **+2 evals** per B-win iteration, not +1 (the B-restoration eval already exists) | **CONFIRMED to the unit** | **General, validated on all 3 WS1 cells** (§4.4). Break-even iteration count is ~10% tighter than assumed. |
| **F3** | WS1.5 blue team | "A-win never sets conv" is sound for correctness but **unbounded for termination** — a run of sub-tol A-wins pins the loop at `maxiter` at 1.2–1.4× cost | **CONFIRMED** | **General.** Formal bound on A-wins is `(lnL*−lnL_0)/1e-9 ≈ 10¹³` ⇒ the only effective bound is `maxiter`. Fixed by the **one-iteration defer**. Measured: `defers` stayed ≤7 in every arm ⇒ K7 never fired. |
| **F4** | WS1.5 blue team | The **`≤0.90×` wall gate is UNACHIEVABLE BY CONSTRUCTION at `tol=1e-7`** | **CONFIRMED — structural** | **avian, `tol=1e-7`, any step policy whatsoever.** avian's `I_off = 400` **is the `maxiter` cap**, not a convergence. `I_on` cannot fall; WS1.5 can only add evals. ⇒ efficacy must be measured at `JOLT_IR_TOL=1e-2` on **loop wall**, not total wall. See §8. |
| **F5** | WS1.5 blue team | "Keep the DNA/AA R4 controls inert" **cannot be asserted** under `it≥2` firing | **CONFIRMED** | **General under the new firing policy.** The spectrum is rebuilt every iteration; no τ can guarantee `rank_drop==0` for 400 iterations. Replaced by S1/S2/S3 (§5.5). |
| **F6** | WS1.5 blue team | AA has `nFreeQ=0` ⇒ `n_off = 1.636` vs avian's 6.515 ⇒ +2 evals is **2.2× per-iteration on AA** vs 1.31× on avian | **CONFIRMED, measured exactly** | **Per-dataset, exact.** ⇒ **AA can only ever be a SAFETY cell, never an efficacy cell**; DNA R4 ends `reject_stall@48`, and a shorter reject-stall is not a win. **avian `GTR+R6` is the only efficacy cell in the entire fixture set.** |
| **WS2-SF1** | WS2 self-falsification | §2.2's *"no value of τ fixes it"* | 🔴 **FALSE** | **avian warm seed, offline recompute against `harpspec_174235905/hs_av.console`, reproducing `[HARPSPEC-STEP]` to 14 digits.** τ sweep gives `max\|dy\|` 1.7124 → 0.9473 → 0.8746 → 0.3810 → 0.2118 vs canonical's accepted 0.191376. **τ was set ~100× too small.** The Picard *measurement* stands. |
| **WS2-SF2** | WS2 self-falsification | §3.1's κ₀=3e-2 Tikhonov is a *smooth* regulariser, not a truncation | 🔴 **FALSE** | **avian warm seed.** Its filter factors are `0.97, 0.79, 0.56, 0.19, 0.04, 0.004` ⇒ **effective rank 2.56** — a soft rank-3 truncation, i.e. exactly what §2.2 argued could not work. |
| **WS2-SF3** | WS2 self-falsification | §3.2.1's ρ trust-region test would have rejected WS1's catastrophic step | 🔴 **FALSE — it ACCEPTS it by 156×** | **avian `it==1`, WS1 step.** Base = `-11,901,270.395` ⇒ **`ρ_A = 7.79`** vs the design's reject threshold **η=0.05** (a factor of 156). Worse: `ρ > ¾` then fires `κ ← κ/2`, **loosening damping after the worst step in the run.** Root cause = a **subspace category error**: the trial brlen/α/pinv/Q arms are **SHARED by A and B**, so `ln_A − ln_base` is dominated by arms the quadratic model does not describe. **The same defect voids the Armijo guarantee.** ⚠️ Note the doc's own provenance caveat: the base value was *inferred*, then read from `ws1_cpu_av.log`. |
| **WS2-SF4** | WS2 self-falsification | §3.4's fire-every-iteration policy is compatible with §4's PRIMARY gate | 🔴 **FALSE — makes it unreachable** | **Source-general.** `conv=true` is set **only** in the L-BFGS and canonical μ-ladder branches; the HARP transaction sets it nowhere ⇒ firing every iteration means the loop can **only** exit at `maxiter`. (WS1.5 fixed this with the B-win-may-converge rule.) |

---

# 8. GATES AS EXECUTED (not as designed)

## 8.1 The pre-registered ladder, and how far it actually got

| Stage | Pre-registered in | Status |
|---|---|---|
| `[HARPSPEC]` read-only probe | ULTRA-REVIEW §3 | ✅ **RAN** — job `174235905`, PASS |
| §5.1 avian decisive spike | RECONCILIATION §5.1 | ✅ **RAN** — job `174266861`, **NO-GO** |
| §5.2 matched R4 controls | RECONCILIATION §5.2 | ⛔ **NEVER REACHED** (gated behind §5.1) |
| §5.3 full multi-seed selection gate | RECONCILIATION §5.3 | ⛔ **NEVER REACHED** |
| WS1.5 τ sweep | WS15-PLAN §D | ✅ **RAN** — job `174323861`, ambiguous |
| WS1.5 robustness / τ-transfer | WS15-PLAN §E + gate script | ✅ **RAN** — job `174328851`, **CLOSED** |
| WS2 (Tikhonov/ρ/κ/Armijo) | WS2-DESIGN | ⛔ **DO NOT BUILD — NEVER BUILT** |

**Never executed, and therefore never evidence about anything:** the partitioned `-p` cell; the
bootstrap/`out_patlh` snapshot cell; the multi-seed (1/2/3) real-data selection gate on avian MFP and
`CAT_100S93F.phy`; the DNA-1M / AA-1M scale controls; the R4/R8/R10 single-model ladder; the `-nt 1` / `-nt 12` paired
cells. **No claim about HARP's effect on model selection was ever tested.** HARP died on likelihood safety long
before selection invariance was on the table.

## 8.2 Gates later shown UNREACHABLE

> 🔴 **The `≤0.90×` wall gate at `tol=1e-7` is unachievable by construction — WS1.5 finding F4.**
>
> At the compiled default `tol=1e-7`, avian's OFF arm terminates `reason=maxiter` at the 400 cap. **It is not
> converging; it is being truncated.** So `I_off = 400` is a *constant*, and `I_on < 400` requires the method to
> *cause* either `conv` or a `reject_stall` — and a `reject_stall` is explicitly disqualified as a speed win. With
> `I_on = I_off = 400` the wall ratio is exactly `ψ ∈ [1.17, 1.31]`.
>
> **⇒ The ≤0.90× gate was unreachable at `tol=1e-7` for ANY step policy whatsoever** — including a hypothetical
> perfect one. It was inherited from the WS1 plan and the addendum and carried forward unexamined through **three**
> documents before anyone checked whether it could be satisfied.
>
> Consequences, both applied in job `174323861`/`174328851`: (1) run the efficacy sweep at **`JOLT_IR_TOL=1e-2`**,
> where `conv` is reachable, with the OFF baseline at the **same** tol; (2) gate on **loop wall** (`total − C`, with
> `C ≈ 2.46–2.50 s` measured in-job by the `maxiter=0` probe) **and iterations**, never total wall — because at
> `tol=1e-2` the fixed setup cost starts to dominate a 26 s total, so a total-wall ratio would be measuring tree
> setup.

Also worth recording as a near-miss: **F1** showed that the red-team's flagship τ=7e-3 arm would have **silently
self-disabled** (`cutoff_ambig=1` ⇒ ineligible ⇒ `fires>0` but `A_tried==0`), producing a "clean negative" that was
actually a no-op. The WS1.5 decision rule was therefore ordered so that **step 1 is "did HARP actually arm?"** and a
non-arming sweep row is **VOID, not a negative**.

## 8.3 K1–K7 — which fired

Kill criteria from `JOLT-HARP-WS15-PLAN.md:599-615`. Deliberately easy to hit, because a cheap decisive negative was
the good outcome.

| ID | Criterion | Fired? | Evidence |
|---|---|:--:|---|
| **K1** | **No win anywhere** — `A_acc == 0` across all non-void avian arms | ❌ no | `A_acc` was positive in most arms (e.g. `174323861` τ=1.5e-2: `A_acc=96/187`; τ=7e-2: `A_acc=63/143`). HARP *did* win same-base competitions. |
| **K2** | **Safe but useless** — `A_acc ≥ 1` somewhere but `I_on ≥ 0.95·I_off` everywhere at `tol=1e-2` | ❌ no | τ=1e-3 reached `I_on=229` vs 400 (`174323861`) and 333 vs 400 (`174328851`). Real iteration reduction occurred. |
| **K3** | 🔴 **Trajectory harm** — any arm with `lnL_ON < lnL_OFF − 0.05` | ✅ **FIRED, hard** | avian R8 @ τ=1e-3: **−705.62 nats** (`174328851`). Also avian R6 @ τ=5e-4 (−133.33) and τ=2e-3 (−248.60), and every WS1.5 non-1e-3 arm in `174323861`. |
| **K4** | **No wall headroom** — best `loop_wall_on/loop_wall_off > 1.00×` in all arms | ❌ no | τ=1e-3 achieved **0.598×** loop wall on avian R6 (`174323861`). |
| **K5** | **No novelty** — wins only where `dyscale < 1` **and** `cos > 0.90` (a rescaled canonical step) | ❌ no | The winning arm had `cos_mean=0.3789` and `dyclip=2/228` ⇒ genuinely different direction, DYMAX barely engaged. |
| **K6** | **Truncation not well-posed** — every `rank_kept ≤ 3` arm void on `margin_min`/`basis_ok` | ❌ no | τ=1.5e-2 ran cleanly at `rank_kept=6, margin_min=5.057e-04, basis_ok=1`. |
| **K7** | **Defer fires constantly** — `defers/fires > 0.5` | ❌ no | Max observed `defers=7` against `fires=283` (`174323861` τ=5e-3); most arms `defers=0`. |

**Plus the ARM-3 rule (§3), which is the one that actually closed the line** and fired on **both** clauses: "wins on
avian R6 only, loses elsewhere" **and** "peak is a SPIKE (5e-4 and 2e-3 both lose)".

> **Read this table carefully — it is the most interesting thing about the closure.** HARP did **not** die of K1
> (never wins) or K5 (not novel) or K2 (useless). It won same-base competitions, it changed the direction genuinely,
> it cut iterations 43%, it cut loop wall to 0.598×, and it produced the **first avian `+R` convergence ever
> recorded**. It died of **K3 + non-transferability**: the same knob that bought all of that on avian R6 cost 705
> nats on avian R8. **A method that is fast, novel, convergent and 705 nats wrong is unshippable.**

---

# 9. EVIDENCE LEDGER

## 9.1 VERIFIED ON DISK (by me, 2026-07-21, reading the bytes)

### Jobs and raw paths

| Job | Directory | Contents | Verified |
|---|---|---|---|
| `174235905` | `/scratch/rc29/as1708/gems-verify/harpspec_174235905/` | 39 files: `hs_av`/`hs_dna`/`hs_aa` `.console/.log/.iqtree/…`, `off_can*`, `inv_lmin*`, `harp.npz`, `cmake.log`, `make.log` | ✅ read `hs_av.console`, all 11 `[HARPSPEC-EIG]` lines, `[HARPSPEC-DSCALED]`, all three `[HARPSPEC]` summary lines |
| `174254673` | WS1 gate, **first attempt — telemetry corrupt** | `gems-verify/harpws1_174254673/` — **76 files, 616 K, 18 `ws1_on*` arms** | ✅ located and explained (below) |
| `174266861` | `/scratch/rc29/as1708/gems-verify/harpws1_174266861/` | 76 files: `hs_av_new*`, `hs_av_arch.harp`, `off_can_{av,dna,aa}*`, `kill_av*` | ✅ read all `[HARPSTEP]` and `[HARPEXIT]` lines |
| `174323861` | `/scratch/rc29/as1708/gems-verify/harpws15_174323861/` | 196 files: `base_{av,dna,aa}`, `t{1e-4,1e-3,5e-3,1p5e-2,7e-2}_{av,dna,aa}`, `eff_lo`, `eff_hi`, `cpu_av`, `hs_av_new`, `hs_av_arch.harp` | ✅ read all `[HARPEXIT]` lines + walls + BEST SCORE |
| `174327276` | `/scratch/rc29/as1708/gems-verify/avfixtree_174327276/` | 91 files: `A_s1/A_s2/A_s3`, `B_n13/B_n104`, … | ✅ directory exists; ⚠️ **see 9.3** |
| `174328851` | `/scratch/rc29/as1708/gems-verify/tolladder_174328851/` | **282 files**; arms `a1_*` (18), `a2_*` (6), **`a3_*` (110)**, `off_{av6,av8,dna,aa}` | ✅ read all ARM-3 `[HARPEXIT]`/`[HARPCFG]`/BEST SCORE lines |

### Binaries (md5, first 8 — computed by me)

| md5 | Path | Role |
|---|---|---|
| `d6b9e4d0` | `/scratch/rc29/as1708/iqtree3-harpspec/build-harpspec/iqtree3` | `[HARPSPEC]` read-only probe |
| `583cec03` | `/scratch/rc29/as1708/iqtree3-harpspec/build-harpws1/iqtree3` | WS1 one-shot state-changing |
| `8f8ce05e` | `/scratch/rc29/as1708/iqtree3-harpspec/build-harpws15/iqtree3` | WS1.5 runtime-τ (used by both `174323861` and `174328851`) |
| `9b6b4519` | `iqtree3-opg/build-opg3/iqtree3` | canonical OPG negative control (0 HARP sentinels) |
| `f3f7875f` | canonical `iqtree3-jolt-merge@30c0faf9` | canonical baseline named in PHASE1 §0 |

### Source

| Item | Value | Verified |
|---|---|---|
| Worktree | `/scratch/rc29/as1708/iqtree3-harpspec` | ✅ |
| Base commit | `1bb82e14a5e652145a007e89401e7da71af6b99f` (*OPG Phase 1: empirical-Fisher Gram H build + validate*) | ✅ |
| **HARP commit** | **`ce99e337faaf02f7226b747087bd7d43e833a632`** — *"HARP: rank-projected +R step, runtime tau (default-OFF)"*, 2026-07-21 20:36:34 +1000 | ✅ **appears in NO source doc** |
| Branch | `harp-ws15` (HEAD = `ce99e337`) | ✅ |
| **Push status** | **UNPUSHED.** `git branch -r --contains ce99e337` → **empty** | ✅ |
| Untracked in worktree | `build-harpspec/`, `build-harpws1/`, `build-harpws15/`, `harp-ws1.patch`, `harpspec_selftest{,.cpp}`, `harpspec_selftest_emit{,.cpp}`, `harpspec_verify.py` | ✅ |
| Edited file | `tree/gpu/gpu_lnl_intree.cu` **only** | per plan; single-file scope |

### Datasets and fixtures

| Item | Path |
|---|---|
| avian-1M (48 × 1,000,000) | `/scratch/dx61/as1708/shared-jolt/avian-convergence/avian_1000000.phy` |
| DNA-100k (100 taxa × 100k) | `/scratch/rc29/as1708/datasets/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` |
| AA-100k (100 taxa × 100k) | `/scratch/rc29/as1708/datasets/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` |
| **Fixed avian PARS tree** | `/scratch/rc29/as1708/gems-verify/fixedtree_src/avian_pars_seed1.tre`, **md5 `da93f0c9`** ✅ verified |
| Partition fixture (never used) | `gadi-ci/gems/opg_p2_parts.nex` |

### Scripts

| Script | Role |
|---|---|
| `gadi-ci/gems/gems_harpspec.sh` | `[HARPSPEC]` probe gate (`174235905`) |
| `gadi-ci/gems/gems_harp_ws1.sh` | WS1 gate (`174266861`) |
| `gadi-ci/gems/gems_harp_ws15.sh` | WS1.5 τ sweep (`174323861`) |
| **`gadi-ci/gems/gems_tolladder_taurobust.sh`** | **job `174328851`** — carries the pre-registered ARM-3 decision rule verbatim; `BIN=…/build-harpws15/iqtree3 # 8f8ce05e`; `AVTREE=…/avian_pars_seed1.tre # md5 da93f0c9` ✅ read |
| `harpspec_verify.py` (in worktree, untracked) | independent offline Jacobi verifier, no numpy |

## 9.2 🔴 THE `174328851` NAMING CONFOUND — read before citing this job

**Job `174328851` is named `tolladder` and its on-disk directory is `gems-verify/tolladder_174328851/`. It is NOT
solely a tolerance sweep.**

| Arm prefix | Files | Environment | What it is |
|---|---:|---|---|
| `a1_*` | 18 | `JOLT_IR_TOL=<1e-2…1e-7>` | **tolerance ladder**, 3 datasets × 6 tolerances |
| `a2_*` | 6 | `JOLT_IR_TOL=<…>` | tolerance ladder, model-selection winner check |
| **`a3_*`** | **110** | **`JOLT_HARP=1 JOLT_HARP_TAU=<τ> JOLT_IR_TOL=1e-2`** | 🔴 **HARP-ON.** 5 τ × 4 cells, fixed `-te` tree |
| `off_*` | — | `JOLT_IR_TOL=1e-2` (no HARP) | matched OFF baselines for ARM 3 |

Verified by reading `[HARPCFG] JOLT_HARP_WS15_TAUENV tau=… dymax=… allbt=…` in the `a3_*` consoles and by reading
`E="JOLT_HARP=1 JOLT_HARP_TAU=$TA JOLT_IR_TOL=1e-2 …"` in the gate script.

**⇒ Anyone treating `tolladder_174328851` as a pure tolerance experiment inherits a HARP confound on 110 of its 282
files.** The `a1_*`/`a2_*` arms are clean; the `a3_*` arms are not.

## 9.3 CITED BUT UNLOCATABLE / UNVERIFIABLE FROM THE DOCS

| Item | Problem |
|---|---|
| 🔴 **Job `174327276`** | The directory `gems-verify/avfixtree_174327276/` **exists on disk** and **ARM 3's entire fixed-`-te` design depends on it** — the script comment reads *"Uses `-te` on the FIXED avian tree so the starting topology cannot confound the comparison (RF between per-seed PARS trees is 40-52 of 90 -- see job `174327276`)"*. **But `174327276` is named in NONE of the nine source documents.** The "RF 40–52 of 90" figure that justifies fixing the tree is therefore **unverifiable from the doc set** — it is traceable only to a comment in `gems_tolladder_taurobust.sh`. **Flagged: the closing gate's most important design choice rests on a citation the documentation does not carry.** |
| ~~Job `174254673`~~ **RESOLVED 2026-07-22 — not unlocatable; it is the corrupt-telemetry first attempt** | `gems-verify/harpws1_174254673/` **does exist** (76 files, 616 K, 18 `ws1_on*` arms), so `174254673` was a real WS1 gate run, not merely a build. Its `[HARPSTEP]` lines are **corrupt**: `mu=nan base=0.000000`, against the re-run's `mu=5.000e-01`. That is the known `[HARPSTEP]` printf format/arg bug (a missing `mu` argument shifted every subsequent float field). ⇒ **`174254673` is superseded by the clean re-run `174266861`, whose telemetry is the only version that may be quoted.** Nothing load-bearing is lost, and the WS1 verdict is unaffected — it was banked from the clean run. |
| `ws1_cpu_av.log` base value `-11,901,270.395` | Used to compute WS2-SF3's `ρ_A = 7.79`. The doc itself flags the provenance: *"I inferred it instead of reading it"* before correcting. **Treat the 156× figure as directionally sound, magnitude approximate.** |
| Commit `ce99e337` | **Verified by me on disk** but appears in **no** source document. Recorded here for the first time. |
| §5.3 full-gate cells | Partitioned `-p`, bootstrap/`out_patlh`, multi-seed selection, DNA-1M/AA-1M scale, R4/R8/R10 ladder, `-nt 1`/`-nt 12`. **Designed in detail, never run.** Any statement about HARP and model selection is unsupported. |
| WS2 §6 references | Moré 1978 page range; Armijo 1966 volume; K-FAC's ω₁/T₁ constants; Hansen 1997-vs-1998 year; `CTR-LoRA` arXiv:2510.15962 (snippet only). **Partially verified only** — see §11. |

## 9.4 🔴 NUMBERS THAT APPEAR WITH DIFFERENT VALUES IN DIFFERENT DOCS

**Every one of these is a real discrepancy with a real explanation. None is an error. All are traps.**

| Quantity | Value A | Value B | Explanation |
|---|---|---|---|
| **avian λ ratio** | `9.481107e-14` (PHASE1 §0, ULTRA-REVIEW §4, HARPSPEC-RESULT) | `4.704580e-15` (WS15 §A.1) | **DIFFERENT BASES.** A = **D-scaled** `C` (the archived `[OPGLMIN]` value). B = **raw** `H_r` (`[HARPSPEC-EIG]`). Both correct. Ratio between them ≈ 20×. |
| **avian `λ_max`** | `6.560828` (ULTRA-REVIEW R3) | `1.677738555e+06` (WS2 §2.2, HARPSPEC) | **DIFFERENT BASES.** A = D-scaled, B = raw. ~11 orders of magnitude apart. **R3's `1e-4 → 6.560828e-4` boundary arithmetic is a D-scaled statement and does not apply to WS1's raw solve.** |
| **avian OFF iterations** | **400** | **401** | Same run. `400` = `maxiter`, the number of *completed* iterations. `401` = the terminal `it` counter value at loop exit (`for(it=1; it<=maxiter; it++)` exits with `it = maxiter+1`). `[HARPEXIT]` prints `it=401`; §C.1's eval model uses `400`. **Both refer to job `174266861`'s single avian OFF run.** |
| **avian OFF lnL** | **`-11216886.230`** | **`-11216787.185`** | 🔴 **DIFFERENT EXPERIMENTS — NEVER COMPARE ACROSS.** A = `-n 0 -starttree PARS`, `tol=1e-7`, jobs `174266861` (`off_can_av`) and `174323861` (`base_av`). B = **fixed `-te` tree** on `avian_pars_seed1.tre`, `JOLT_IR_TOL=1e-2`, job `174328851` (`off_av6`). Different starting topology **and** different tolerance. The WS1.5 "+49.7 nats" is against A; the robustness gate's "+88.72" is against B. |
| **avian OFF wall** | `25.994 s` | `26.453 s` / `26.592 s` | **JOB-SCOPED.** `25.994` = `174323861` `base_av`. `26.453` (OFF) / `26.592` (ON) = `174266861`. Same binary family, same node type, different job ⇒ ~2% run-to-run spread. **Any wall ratio must use the OFF arm from its own job.** The 0.598× loop-wall claim correctly uses `174323861`'s own 25.994 s. |
| **matched-R4 iteration targets** | `≤36` (DNA) / `≤35` (AA) | **`≤23` (DNA) / `≤33` (AA)** | **SUPERSEDED.** PHASE1 §5.1 set the loose pair. Blue team recomputed from Phase-3 data: DNA R4 `48→17`, so 80% retention requires **`≤23`, not `≤36`**; AA R4 `44→30`, so **`≤33`, not `≤35`** (`JOLT-HARP-BLUE-TEAM.md:129-134`). Reconciliation §5.2 adopted the tight pair. **Use `≤23`/`≤33`.** Moot in the end — §5.2 was never reached. |
| **avian efficacy bar** | `wall ≤ 1.03× OFF` | `≤320 iters` + `≤0.90× wall` | **ESCALATED**, then shown **unreachable at `tol=1e-7`**. PHASE1/RECONCILIATION §5.1 set 1.03×; ultra-review R1/R2 escalated to ≤320 + ≥10% wall; WS1.5 **F4** proved the 0.90× total-wall form unachievable by construction and re-specified it as `I_on ≤ 0.75·I_off` **and** `loop_wall ≤ 0.90×` at matched `tol=1e-2`. |
| **HARP eval overhead** | `+1` per iteration | **`+2`** per B-win iteration | **F2 correction**, validated to the unit (§4.4). `+1` when A wins (A is already last), `+2` when B wins after A ran (A eval + B-restoration eval). |
| **`τ=7e-3`** | red-team's flagship sweep value | **BANNED** | **F1.** `margin_min = 1.36e-7 < 1e-6` ⇒ `cutoff_ambig=1` ⇒ ineligible ⇒ **HARP never fires.** Replaced by `1.5e-2` (same `rank_kept=3`, `margin_min=8.00e-3`). |

---

# 10. BANKED AND EXPLICITLY UNAUTHORISED

**Everything in this section exists as text only. None of it was built. None of it is authorised.** Listed so a
future reader knows it was considered and *why it is not simply "the obvious next step"*.

| Item | Origin | Status | 🔴 Why not |
|---|---|---|---|
| **Tikhonov damped solve** `δ(μ) = Σ_i v_i a_i/(λ_i+μ)`, `μ = κ·λ_max` | WS2-DESIGN §3.1 | **BANKED, NEVER BUILT** | Its own calibrated κ₀=3e-2 has effective rank **2.56** ⇒ it *is* a soft truncation (WS2-SF2), so it is not categorically different from what WS1.5 already measured. And Kunstner et al.: **damping fixes magnitude, not direction** — §4.2 proved the failure is directional. |
| **ρ trust-region test** `ρ = (ln_A − ln_base)/qgain(μ)`, reject if `ρ ≤ η=0.05` | WS2-DESIGN §3.2.1 | **BANKED — and AS SPECIFIED it would have been WORSE THAN NOTHING** | 🔴 **It ACCEPTS the catastrophic WS1 step by 156×**: `ρ_A = 7.79` vs `η = 0.05`. Then `ρ > ¾` fires `κ ← κ/2`, **loosening damping right after the worst step in the run.** Root cause: a **subspace category error** — the trial brlen/α/pinv/Q arms are **shared** by A and B, so `ln_A − ln_base` is dominated by arms the quadratic model does not describe. Any future ρ-test **must** be computed on the `(y,z)` sub-step alone, or it measures the wrong thing. |
| **Armijo backtracking** on α along the damped direction | WS2-DESIGN §3.2.2 | **BANKED, NEVER BUILT** | **Voided by the same subspace category error as the ρ-test.** The Armijo guarantee ("for a genuine ascent direction, backtracking terminates at an improving step") is about the *objective along the direction being searched*; with shared non-`(y,z)` arms it is not that objective. |
| **κ adaptation** (LM / K-FAC ρ-rule, ω=0.5, κ∈[1e-4,10], expand only if α-unclipped) | WS2-DESIGN §3.3 | **BANKED, NEVER BUILT** | Depends entirely on ρ being meaningful. It is not (above). |
| **`JOLT_HARP_FIRST`** — also fire at `it==1` | WS15-PLAN §E | **BANKED, NEVER BUILT** | WS1 *was* the `it==1` experiment. `it==1` competes against **deliberately frozen** canonical model arms (`ddY/ddZ = −1e6`), so it is not an honest same-base competition. |
| **`JOLT_HARP_MAXFIRE`** — cap fires per invocation | WS15-PLAN §E | **BANKED, NEVER BUILT** | A fire-count cap is another scalar knob on a response already measured to be non-monotone in τ. |
| 🔴 **Tier-3: true Fisher / GGN** — replace the OPG with model-simulated scores or a Generalized Gauss-Newton | WS2-DESIGN §5 | **DOCUMENTED and EXPLICITLY UNAUTHORISED** | This is the mathematically correct fix for §6.3 — and it is exactly why it must not be started casually. It is a **new curvature subsystem**, not a step-policy tweak: it needs sampling from the fitted model per pattern (or a GGN factorisation of the phylogenetic likelihood), new kernels, new validation, and a new correctness argument. `JOLT-HARP-WS15-PLAN.md:618-619` is unambiguous: *"**Do not** open the banked Tier-3 (true-Fisher / GGN) direction — it is documented and unauthorised."* |
| **Functional support reduction** — optimise the rate mixing measure, delete support atoms, vertex-direction gap instead of local curvature | PHASE1 §1 triage row 6 | **BANKED as a separate research project** | Genuinely different from everything above (it changes *what is optimised*, not *how*). Also genuinely large: a safe fixed-`k`, mean-constrained, continuous-rate oracle plus joint branch coupling is far beyond a minimal JOLT change. Primary refs: **Lindsay 1983**, `10.1214/aos/1176346059`; **Groeneboom et al. 2008**, `10.1111/j.1467-9469.2007.00588.x`. |
| Weak penalty / MAP; ordered-rate coordinates; mirror descent / natural gradient; BIC-aware early stop | PHASE1 §1 triage | **Rejected at triage** | See §1.4. |

---

# 11. NOVELTY AND CITATION HYGIENE

## 11.1 Methods novelty is LOW — say so

Every ingredient is standard: truncated-Newton + trust region (CG-Steihaug); truncation + damping hybrids (Hansen);
OPG + line search (BHHH 1974); Fisher/KL trust regions (TRPO 2015); LM ρ-adapted Tikhonov (K-FAC 2015). **TRPO is
structurally the closest prior art.**

> **Do not claim we invented truncation + trust region.** (`JOLT-HARP-WS2-DESIGN.md:207`)

## 11.2 What IS defensible — in phylogenetics specifically

1. **No mainstream package uses a second-order / Fisher method for `+R`.** IQ-TREE uses L-BFGS-B /
   `minimizeMultiDimen` / EM; RAxML-NG uses two-step L-BFGS-B with an EM alternative. All first-order,
   quasi-Newton, or EM.
2. **A quantified conditioning characterisation of the FreeRate likelihood.** This is the durable contribution and it
   survives HARP's closure intact:
   - the **λ_min identifiability spectrum** across 12 model orders (already shipping as a diagnostic);
   - the **discrete Picard violation** table (§4.3) — `|a|` decays 13.6× slower than λ over the kept modes;
   - the **H-norm-vs-parameter-space decomposition** (§4.2) — same curvature budget, 9× the parameter movement;
   - the **`+R` cap-truncation** characterisation: avian `GTR+R6` terminates `reason=maxiter` at 401 iterations
     rather than converging.
3. **External corroboration that this is a real, unresolved field problem:** IQ-TREE GitHub **Issue #38**
   (Woodhams, 2017) — `+R` hitting a hard 99-cycle cap while still improving, with no developer resolution —
   independently matching our own 401-cap finding on different data eight years earlier.
4. **Two transferable negative results** (§4.5): *converging faster is not evidence of a better step*, and *an
   oscillating knob response means untunable, not under-tuned*. The second is now a **three-occurrence** law within
   this project.

## 11.3 ⚠️ CITATION HYGIENE — mandatory before any thesis or publication use

> **Several references in the WS2 research pass were only PARTIALLY VERIFIED. Re-check each one before use:**
> - **Moré 1978** — page range unverified.
> - **Armijo 1966** — volume unverified.
> - **K-FAC (Martens & Grosse, ICML 2015)** — the ω₁/T₁ constants were not confirmed against the paper.
> - **Hansen** — 1997-vs-1998 edition/year ambiguity unresolved.
> - **`CTR-LoRA`, arXiv:2510.15962** — **snippet only**, and flagged as *the most likely closest prior art*. This one
>   matters most: an unverified nearest-prior-art citation is a novelty-claim risk, not just a formatting risk.
>
> Also treat as needing first-hand checking: Kunstner/Balles/Hennig (NeurIPS 2019), Martens (JMLR 2020) §§10–11,
> White (Econometrica 1982), BHHH (1974), Lindsay (1983), Groeneboom et al. (2008), and IQ-TREE Issue #38's date and
> content.
>
> **Our own measured numbers are first-hand and are NOT subject to this caveat** — every one carries a job ID and a
> preserved raw path in §9.

---

# Appendix A — the dead PHASE1 `D`-scaled formula (provenance only)

**⚠️ THIS FORMULA WAS NEVER BUILT. It is reproduced solely so that the D-scaled numbers appearing in PHASE1 §0,
ULTRA-REVIEW R3, and `[OPGLMIN]`/`[HARPSPEC-DSCALED]` output can be traced to their source.** It was killed by red
finding 2 before any implementation; see the boxed warning in §1.3.

From `JOLT-HARP-PHASE1-DESIGN.md:62-107`:

```text
# --- reduce and scale ---
H_r = Qᵀ H Q                                       # Q = Helmert projector, n = 2k-1
g_r = Qᵀ g
D_i = max((H_r)_ii, 1e-12 · max_j (H_r)_jj)         # ← the amplifier: permits D_i/max(D) = 1e-12
C   = D^(-1/2) H_r D^(-1/2)                         # correlation matrix
g_h = D^(-1/2) g_r
C   = V diag(λ_i) Vᵀ

# --- hard effective-rank rule ---
tau_rank = 1e-4                                     # pre-registered, fixed before the state-changing run
keep(i)  = isfinite(λ_i) && λ_i > 0 && λ_i/max_j(λ_j) >= tau_rank

# --- the projected step ---
a_i     = v_iᵀ g_h
delta_h = Σ_{i: keep(i)} v_i · a_i / (λ_i + lambda_LM)     # ← lambda_LM WAS NEVER SPECIFIED
delta_r = D^(-1/2) delta_h                                  # ← second D^(-1/2): the 10^12 leak
delta   = Q delta_r
```

**Two fatal defects, both found pre-build:**

1. **The `D`-scaling leak (red finding 2).** With `D_i/max(D) = 1e-12` permitted, and **two** `D^{-1/2}` maps applied,
   the advertised `10⁴` retained-spectrum condition bound holds **only in correlation coordinates**. The *physical*
   applied step can carry an additional scale factor approaching `10¹²`. An uncorrelated coordinate with tiny absolute
   Fisher information gets a scaled correlation eigenvalue near 1 — so HARP **keeps** it and then amplifies it twice.
   The claim *"the null spectrum is never in a denominator"* is false in the coordinates actually applied to the model.
2. **`lambda_LM` had no algorithm (red finding 3).** No initialization, no min/max, no reject multiplier, no Nielsen
   `nu` state, no gain-ratio predictor, no rule when neither proposal wins, no "increase" factor when B wins, no
   interaction with canonical `mu` when A wins. Two conforming implementations would produce different branch
   trajectories and different endpoint likelihoods. **Declared a pre-build blocker: there was no single algorithm for
   a gate to test.**

**Rationale for `tau_rank = 1e-4`, preserved for the record:** fixed before the state-changing run; not chosen from
dense-step win/loss labels; sits inside the reproduced identifiability gap (matched R4 ≥ `8.62e-2`; measured
over-parameterised cells ≤ `1.98e-6`); and is 10× the independent per-pattern score gate's `1e-5` relative tolerance.
**This calibration was performed on the D-scaled gap and then inherited by WS1's raw solve — which is exactly threat
T2** (§7). WS1.5's measured answer: on avian the best τ was `1e-3`, an order of magnitude larger — and it still did
not transfer to R8.

---

# Appendix B — process history, compressed

**Nine documents, five review rounds, four gates, six weeks of design for two builds. The process worked: it killed
two bad designs before they were built, and the one thing it insisted on adding — the read-only probe — produced the
only results that outlived the method.**

| # | Document | Date | Role | Outcome |
|---|---|---|---|---|
| 1 | `JOLT-HARP-PHASE1-DESIGN.md` | 2026-07-20 | Pre-review design. `D`-scaled formula + `λ_LM` + per-iteration A/B. | **Superseded.** Preserved as the pre-review deliverable. |
| 2 | `JOLT-HARP-RED-TEAM.md` | 2026-07-20 | Hostile review, 9 ranked findings. | **NO-GO — do not build as written.** 3 pre-build blockers (1, 3, 5). |
| 3 | `JOLT-HARP-BLUE-TEAM.md` | 2026-07-20 | Adversarial defence + cheaper variant. | **"NO-GO is justified"** on 1/3/5, but red **overreaches** treating exact-zero truncation as closed. Proposes **HARP-WS1**: one shot, raw `H_r`, no λ, transactional A/B. Also corrects the R4 iteration targets `≤36/≤35` → **`≤23/≤33`**. |
| 4 | `JOLT-HARP-RECONCILIATION.md` | 2026-07-20 | Adjudicates all 9 findings; freezes the WS1 algorithm. | **Authorises WS1 only.** Rejects blue's own `λ_H=1e-2` backup (returns to direction-preserving shrinkage) and blue's `TOL_GRADIENT_MODELTEST` KKT test (a caller tolerance, not a validated KKT norm ⇒ a new ungrounded tuning surface). |
| 5 | `JOLT-HARP-ULTRA-REVIEW.md` | 2026-07-20 | User review → red addendum → blue addendum. **R1–R4.** | **Restricts further: read-only `[HARPSPEC]` probe ONLY.** R1 makes avian iterations the mission metric; R4 blocks any state-changing run until the full spectrum is dumped. **This is the decision that produced §4.1/§4.3.** |
| 6 | `JOLT-HARP-HARPSPEC-RESULT.md` | 2026-07-20 | Probe result + 4-way verification. Job `174235905`, bin `d6b9e4d0`. | **avian `GO=PASS`, basis-robust.** Statistical red-team raises T1–T4; T1/T2 empirically falsified on avian, T3 bounded at 2.8%, T4 confirmed decorative. **GO to WS1 with two refinements** (keep-basis-invariance guard; never use `grdd`/`qgain` as ascent evidence). One code defect (**Finding 1**) found and fixed. |
| 7 | `JOLT-HARP-WS1-IMPL-PLAN.md` | 2026-07-20 | Blue-team of the probe + pre-staged WS1 implementation, line-by-line. | Finding-1 fix proven **CORRECT and COMPLETE** (probe-ran ⟺ `opgOK && g_harpspec`). Contains **§2.0, the anatomy of the device-coherence defect** — the most reusable section in the set (§5.1). Built as `583cec03`; gate `174266861` → **NO-GO, −23,414 nats**. |
| 8 | `JOLT-HARP-WS2-DESIGN.md` | 2026-07-21 | First-principles redesign: Tikhonov + ρ-test + κ-adaptation + Armijo. | 🔴 **DO NOT BUILD — NEVER BUILT.** Red-teamed against the archived spectrum; **four of its own load-bearing claims falsified** (WS2-SF1…SF4, §7). Its §2.1/§2.2/§2.3 evidence and §6/§7 novelty framing survive; its §3 design does not. |
| 9 | `JOLT-HARP-WS15-PLAN.md` | 2026-07-21 | Successor: ~15 lines against already-verified code. Runtime τ, fire every `it≥2`, `DYMAX`, defer rule, telemetry. | Corrects **six** things in the red-team's WS1.5 proposal (**F1–F6**, §7). Built as `8f8ce05e` @ `ce99e337`. Gate `174323861` ambiguous → robustness gate `174328851` → 🔴 **HARP CLOSED** by this plan's own pre-registered rule. |

### What the process got right

- **Ordered adversarial review caught a real state-corruption bug before a single line was written** (red finding 1),
  and the fix was proven complete and then measured working.
- **The ultra-review's insistence on a read-only probe first (R4) was the single best decision.** It cost one 15-minute
  job and produced the avian spectrum and the Picard table — the two artifacts that outlived the method.
- **Pre-registered kill criteria, written into the gate script before the run, closed the line cleanly.** No post-hoc
  rescue was attempted. Nothing was tuned after the fact.
- **Negative results were checked as hard as positive ones.** F1 caught a sweep arm that would have silently
  self-disabled and been mistaken for a clean negative.

### What the process got wrong

- **An unreachable gate (`≤0.90×` wall at `tol=1e-7`) was carried unexamined through three documents** before F4
  proved it unsatisfiable by *any* step policy. Nobody checked whether the baseline could move.
- **WS1 was authorised despite being structurally incapable of hitting its own mission metric** — one step out of 401
  cannot move a 401-iteration number, and this was arithmetic available at design time, not a measurement.
- **The `it==1` firing point competed against deliberately frozen canonical arms** (`ddY/ddZ = −1e6`), and this was
  only discovered *after* WS1 failed, by reading the source during the WS2 post-mortem.
- **τ was calibrated in one basis (D-scaled) and applied in another (raw)** — threat T2 — and survived to the built
  binary because the warm-seed keep-sets happened to agree on avian.
- **A gate job was given a misleading name.** `tolladder_174328851` contains 110 HARP-ON files (§9.2).

---

*End of post-mortem. HARP is closed. `harp-ws15` @ `ce99e337` stays local and unpushed. Nothing to promote.*
