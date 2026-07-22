# FreeRate (+R): why it is slow, why it wobbles, and a conditioned-step solution

**Status: DESIGN — nothing built, nothing gated, nothing promoted. Started 2026-07-19.**
**Every number below traces to a job ID or a source line. Claims I could not verify are marked UNVERIFIED and are not used to justify the design.**

---

## 0. The ask, restated precisely

Make `+R` FreeRate models (a) **time-efficient** and (b) **reproducible — RF=0 between the optimised route and the
reference route** — while keeping the joint-optimiser tolerance at `1e-2`. Answer, from evidence rather than
assertion: **why do FreeRate models take so much time, and why do they wobble at higher R?**

Two framing notes that shape everything downstream:

1. **`tol=1e-2` is the correct destination, and `tol=1e-7` is a proven bottleneck (user, confirmed on disk).** The
   shipped default `1e-7` (`gpu_lnl_intree.cu:3220`) is **10⁶× tighter than upstream IQ-TREE's own
   `modelfinder_eps = 0.1`** (`utils/tools.cpp:7645`) — gratuitous over-precision on the scale selection can resolve —
   and it costs **3.55× on DNA-1M MF** (2193s→610s at 1e-2, byte-identical selection on simulated data, job
   `174127774`). `1e-2` is still 10× tighter than upstream. So `1e-2` should be the default. 🔴 **The ONE thing that
   holds it: on real avian +R, bare `1e-2` TODAY moves the tree (RF 4,0,4, job `174142647`) and walks a disjoint
   +I-vs-pure-R model search (WORK-LOG §RF). But that failure is the +R UNDER-CONDITIONING symptom, NOT evidence `1e-2`
   is wrong** — loose tol lands the degenerate +R fit seed-dependently, flipping the early +I/pure-R screen. **This design
   is precisely what makes `1e-2` reproducible: OPG conditioning + λ_min canonicalisation reach the same +R optimum every
   seed, turning the RF 4,0,4 into RF=0 — which is what lets `1e-2` finally ship as the default.** 1e-7 hides the wobble
   by brute convergence; we remove the wobble instead, then don't need 1e-7.
2. **This is an OPTIMISER problem, not a SELECTION problem.** That distinction is load-bearing. Every previous
   attempt in this project that tried to make ModelFinder faster by changing *what gets compared* has failed on real
   data, 4 times out of 4 (§7). This design changes only *how well the optimiser solves each candidate* — the
   candidate set, the criterion, and the comparison are untouched.

---

## 1. Evidence base — what is actually measured

### 1.1 `+R` is essentially the entire optimiser cost (job `174126193`, DNA-1M nt1, 119 candidates, 12,122 joint iters)

| family | models | iters | mean iters/model | share of all optimiser work |
|---|--:|--:|--:|--:|
| **+R** | 48 | **11,021** | **229.6** | **90.9%** |
| +G | 69 | 1,046 | 15.2 | 8.6% |
| base | 2 | 55 | 27.5 | 0.5% |

**Mean +R cost is 15× mean +G cost.** And the blow-up has a sharp onset **at k=5**:

| k | +R2 | +R3 | +R4 | **+R5** | +R6 |
|---|--:|--:|--:|--:|--:|
| mean iters | 26.5 | 32.0 | 41.5 | **265.7** | 223.4 |

🔴 **21 candidates sit at EXACTLY 401 iterations — the `brlenMaxIter` cap (`phylotree.h:2123`) — accounting for
8,421 iters = 69.5% of ALL optimiser work. They never converge; they are truncated.**

### 1.2 Non-convergence is *proven*, not inferred — the GTR/SYM internal control

`GTR+R5 = −59,403,389.130` vs `SYM+R5 = −59,403,408.260`.

**GTR with equal base frequencies is mathematically identical to SYM.** Two runs of the same model landed
**19.13 nats apart**, against `modelfinder_eps = 0.1`. A converged optimiser cannot do this. This single
observation is the cleanest evidence in the whole investigation, and §5 turns it into the primary acceptance test.

### 1.3 The instability is UPSTREAM — it survives with the GPU path disabled (job `174132495`)

> ⚠️ **HEADING CORRECTED 2026-07-21.** This section previously read *"stock CPU IQ-TREE has it too"*. That
> **overstated the evidence**: job `174132495` ran **our** binary `f3f7875f` with `--no-jolt`, i.e. our GPU path
> switched off — **not unmodified upstream**. Our CPU kernels are not stock either (`tree/phylokernelnew.h`:
> `schedule(dynamic,1)`→`schedule(static)` plus a fixed-order reduction replacing `reduction(+:)`/`omp critical`,
> which changes FP summation order vs stock). The table below is valid for what it is — **GPU-disabled CPU path** —
> and the word *stock* was **UNVERIFIED** at the time of writing. See §1.3b for the source-level proof, and the
> stock-binary gate for the empirical closure.

Pure CPU (`--no-jolt`), real avian 48×1M DNA, `--mf-epsilon` sweep, three seeds:

| seed | eps=0.1 | eps=1e-2 | eps=1e-4 (tight reference) |
|---|---|---|---|
| 1 | GTR+F+I+R4 | GTR+F+I+**R6** | **VOID** — SIGKILL @ 12,000 s |
| 2 | GTR+F+I+R4 | GTR+F+I+**R7** | **VOID** — SIGKILL @ 12,000 s |
| 3 | GTR+F+I+R4 | GTR+F+**R5** | **VOID** — SIGKILL @ 12,000 s |

**Three seeds, three different +R winners, with the GPU entirely absent.** ⇒ **the +R wobble is a property of the
FreeRate model and its optimiser, not of our GPU port.** Independently, the tight CPU reference is *unobtainable* —
all three seeds died at 3h20m on a full 104-core node, while GPU-tight does the same alignment in **755.6 s**
(job `174146189`).

### 1.3b Source-level proof that the mechanism is upstream (added 2026-07-21)

Our fork point is upstream tag **`v3.1.2`** (`4e91dd61447c301a896014002b3509bec05f8ab1`, `github.com/iqtree/iqtree3`).
Every file that decides a `+R` / `+I` / `+I+R` fit is **byte-identical to upstream** — compared by git blob hash, not
by reading diffs, and confirmed across all 34 `iqtree3-*` worktrees on disk:

| file | blob (ours) | blob (v3.1.2) |
|---|---|---|
| `model/ratefree.cpp` | `93d454b9` | `93d454b9` |
| `model/ratefreeinvar.cpp` | `5f4a1907` | `5f4a1907` |
| `model/rateinvar.cpp` | `8f932c0a` | `8f932c0a` |
| `utils/optimization.cpp` | `93fb24ce` | `93fb24ce` |

`git diff v3.1.2..c80e7574` restricted to the `+R`-relevant logic returns **zero lines** — upstream has not changed it
since our fork either. Three upstream mechanisms are load-bearing, all verified in source:

1. **`+I+R` does not have a free `p_invar`.** `ratefree.cpp:612` `new_pinvar = 1.0 - new_pinvar;` (where the running sum
   was the rate-category weights), then `setPInvar(new_pinvar)`, and `:624` asserts `sum_prop + new_pinvar ≈ 1`. So
   `+I+Rk` is `+R(k+1)` with one category pinned at rate 0 — the two are **reparameterisations of one family**, which is
   why ranking slides between them instead of resolving. `RateFreeInvar::optimizeParameters` (`ratefreeinvar.cpp:107-112`)
   does nothing but delegate to `RateFree::optimizeParameters`.
2. **The EM is capped at `ncategory` steps** — `ratefree.cpp:541` `for (int step = 0; step < ncategory; step++)`. An R6
   fit gets **at most six** EM steps, not "iterate to convergence".
3. **The EM aborts on the first clipped weight** — `ratefree.cpp:599` `if (zero_prop) break;`, triggered whenever any
   weight falls below `MIN_PROP = 1e-4` (`:510`). On 38%-invariant avian this fires early, at a data- and
   start-point-dependent step. *(That it actually fires on avian is inferred from source, not yet measured — needs a
   `-v VB_MED` EM trace. **UNVERIFIED**.)*

Upstream contains **no identifiability guard of any kind** for `+I` with `+R` — no warning, no comment, no restriction —
and deliberately enables the combination in the `-m MF` candidate list (`phylotesting.cpp:1238` `rate_options[8]="+I+R"`,
`:1246` `test_options_new[8]=true`).

**What is ours, not upstream:** the `401` iteration cap (`tree/phylotree.h:2123` `brlenMaxIter=400`, reported as `401`
when exhausted — `grep -rn '\b401\b'` over upstream returns nothing), the `JOLT_IR_TOL`/`tol=1e-7` inner
Levenberg–Marquardt tolerance (`gpu_lnl_intree.cu:3220`, no upstream equivalent — upstream has no inner LM loop), and the
JOLT early-return in `modelfactory.cpp:1592-1618`. None of these can explain an instability that also appears in builds
that execute none of them.

### 1.3c ⚠️ THREE KNOBS, NOT ONE AXIS (added 2026-07-21)

| knob | source | governs | live on GPU arm? | live on `--no-jolt` CPU arm? |
|---|---|---|---|---|
| `--mf-epsilon` → `modelfinder_eps` (default **0.1**) | `utils/tools.cpp` (`setDefault`), consumed `modelfactory.cpp:1667`/`:1624` | upstream joint-loop stop | **NO** — bypassed by the JOLT early return | **YES** |
| `-eps` → `loglh_epsilon` | `utils/tools.cpp:4568` | tree-search log-lh epsilon — **unrelated to ModelFinder** | n/a | n/a |
| `JOLT_IR_TOL` → `tol` (default **1e-7**) | `gpu_lnl_intree.cu:3220/:3265` | inner LM step acceptance, CUDA only | **YES** | **NO** — silent no-op |

Job `174157174` crosses both live axes on GPU and settles it empirically: `--mf-epsilon` 0.1→1e-2 moved mean wall by
**0.1 s** and changed **no winner** (`GTR+F+R6` at both), while `JOLT_IR_TOL` 1e-7→1e-2 moved wall 778→230 s and took the
winner from a stable `GTR+F+R6` to two distinct `+I` models. **Never present a "0.1 vs 1e-7" contrast as one dial** — the
two numbers name different knobs on different code paths, and a comparison spanning them is not an A/B.

### 1.3d 🔴🔴 THE `brlenMaxIter=400` CAP HAS NO RECORDED RATIONALE — AND ON avian IT, NOT THE TOLERANCE, IS THE BINDING CONSTRAINT (added 2026-07-21)

**The measurement that forces this section.** Job `174328851` ARM 1, avian single-model `GTR+R6`, ladder over
`JOLT_IR_TOL`:

| tol | lnL | iters | conv | capped |
|---|---|---|---|---|
| `1e-2` | `-11216886.230` | 400 | 0 | 1 |
| `1e-3` | `-11216886.230` | 400 | 0 | 1 |
| `1e-4` | `-11216886.230` | 400 | 0 | 1 |
| `1e-5` | `-11216886.230` | 400 | 0 | 1 |
| `1e-6` | `-11216886.230` | 400 | 0 | 1 |
| `1e-7` | `-11216886.230` | 400 | 0 | 1 |

**Byte-identical at every tolerance across five orders of magnitude.** The tolerance NEVER FIRES on this cell — the fit
is truncated by the iteration cap every single time. ⇒ **On avian `+R`, tight-tolerance "stability" is DETERMINISTIC
TRUNCATION, not convergence.** `1e-7` is reproducible because it always stops at the same arbitrary place, and it pays
the full 400-iteration cost for a point that is *not* a converged optimum. (Contrast the easy cells in the same table:
DNA `GTR+R4` and AA `LG+R4` converge in 20–48 iterations with `conv=1`, and there the tolerance does control the result.)
Corroborated at MF scale by job `174127774`: **21 candidates sit at exactly 401 iterations = 8,421 iters = 69.5% of all
optimiser work.**

**PROVENANCE — the number was never chosen.**
- It is a **default argument**: `tree/phylotree.h:2123`
  `double optimizeParametersJOLT(int fixed_len, bool brlenOnly = false, bool leanTail = false, int brlenMaxIter = 400);`
- Introduced by commit `37a63740` (2026-06-26), whose message is *"baseline: brute-force JOLT GPU tree-search
  (**snapshot of tree-search-ts0 working tree**) … Source = iqtree3-gpu HEAD `6fce15de` + uncommitted JOLT/TS.6 mods"* —
  a wholesale working-tree import, not a deliberated choice.
- The 12-line doc comment immediately above the declaration documents the function's inputs, write-back behaviour,
  self-check, and every NaN-return condition — **and says nothing whatsoever about the 400.**
- `grep -rn '400\|401' research/**/*.md`: every hit is a *consequence* observation ("21 candidates at 401", "the
  400-iter +R crawl", "no iteration headroom"). **Zero hits justify the value.**
- 🔴 **The contrast is damning.** The sibling cap in the same file — `optimizeAllBranchesJOLT`'s `maxiter`, default 2
  (`phylotreegpu.cpp:2960-2966`) — carries a full validated rationale in its comment: *"was 12 … 12 over-converged each
  INTERMEDIATE topology … maxiter=2 holds the tight gate (RF==0 + dlnL<=1e-3) DIRECTLY on AA-100K + DNA-200tx, INFERRED
  on AA-200tx, at ~2.5× wall"*, plus an env override. **That cap was researched. This one was inherited.**
- It does **not** exist upstream: `grep -rn 'brlenMaxIter'` over the clean v3.1.2 checkout returns nothing, and
  `grep -rn '\b401\b'` over upstream `*.cpp/*.h` returns nothing. Upstream's analogous limits are
  `num_param_iterations = 100` (`tools.cpp`) and `optimizeAllBranches(min(i,3), …)` (`modelfactory.cpp:1624`).

**WHO INHERITS IT — exactly one caller, and it is the one that matters.**
`grep -rn 'optimizeParametersJOLT('` gives two call sites:
- `model/modelfactory.cpp:1613` — `tree->optimizeParametersJOLT(fixed_len);` — **no cap argument ⇒ takes the 400
  default. This is the ModelFinder candidate path**, i.e. every candidate model in `-m MF` is capped at 400.
- `tree/phylotreegpu.cpp:2971` — passes an **explicit** `maxiter` (the tree-search brlen-reopt path, default 2).

🔴 **THEREFORE THERE IS NO WAY TO SWEEP THE MF CAP TODAY.** `JOLT_BRLEN_MAXITER` is read at
`phylotreegpu.cpp:2965`, *inside* `optimizeAllBranchesJOLT` — the tree-search path only. It never reaches
`modelfactory.cpp:1613`. **A cap sweep requires a source change first** (an env-gated override on the MF path,
default 400 so it is byte-identical when unset). That is a prerequisite, not part of the experiment.

**WHAT A PRINCIPLED CAP WOULD LOOK LIKE (design space — none of this is measured yet, all UNVERIFIED).**
A single constant for every model and every dataset is the thing most likely to be wrong, because the work a fit needs
scales with what is being fitted:
1. **Parameter-count-aware.** A `+R6` fit optimises ~11 free rate/weight parameters *plus* `nedge` branch lengths
   jointly; a `+G4` fit optimises 1 shape parameter plus branches. A cap proportional to the free-parameter count
   (e.g. `base + k·nfree`) at least scales with the problem rather than against it.
2. **Convergence-first, cap-as-safety-net.** The cap should be a runaway guard set far above the working regime, with
   convergence doing the stopping. Today it is the *primary* terminator on the hardest cells, which is backwards —
   and it is precisely why the tolerance is inert there.
3. **Data-type / model-family aware**, mirroring the sibling cap's validated `auto(DNA:2/AA:3)` shape.
4. **Report it.** Whatever the value, a capped fit must be *visible*: a candidate that terminated by cap exhaustion is
   not a converged MLE and its BIC is not comparable to a converged sibling's. Nothing currently flags this in the
   `.iqtree` output, so a truncated `+I+R6` and a converged `+G4` are ranked against each other as equals.

**OPEN QUESTION THIS RAISES ABOUT EVERY TOLERANCE RESULT WE HAVE.** If the cap binds on the hard `+R` candidates and the
tolerance binds only on the easy ones, then the full-`-m MF` winner changes we attributed to tolerance
(`174142485`, `174142647`) are a *mixture*: the tolerance moved the easy candidates, the cap pinned the hard ones, and
the ranking shifted between the two groups. **The cap-vs-tolerance attribution has never been separated.** Until it is,
"tolerance X selects model Y" statements should be read as "the (tolerance X, cap 400) pair selects model Y".

### 1.3e RED-TEAM + BLUE-TEAM VERDICT ON THE SWEEP DESIGN (2026-07-21) — the tolerance effect is a CANDIDATE-COUNT effect, and 1e-7 is not ground truth

A red-team pass on the proposed 5-point ladder found — and I **independently re-verified every load-bearing number
against `gems-verify/tolladder_174328851/`** — that the whole framing was wrong. Three disk facts:

**① The tolerance does not move a fixed candidate — it moves WHICH candidates get fitted at all.** `+R` candidate count
in each run's `.model.gz`: `1e-2`→**139**, `1e-3`→42, `1e-4`→43, `1e-5`→**76**, `1e-6`→42, `1e-7`→42. The "non-monotone
wall" (`1e-5` at 1194 s) is not noise and not a tolerance effect — `1e-5` simply fitted 76 candidates where `1e-7` fitted
42. Mechanism = the upstream skip rule (`phylotesting.cpp`, `MF_IGNORED` when `BIC(+R_k) > BIC(+R_{k-1})`), byte-identical
to v3.1.2. ⇒ **"1e-7 stably picks GTR+F+R6" is partly "1e-7 never lets GTR+F+R7 compete."** Verified: `GTR+F+R7` is
**ABSENT from the fitted set at 1e-3, 1e-4, 1e-6, and 1e-7**; it is fitted only at 1e-2 and 1e-5.

**② 1e-7 is NOT the best fit — 1e-5 is.** The `GTR+F+R6` candidate, same seed/tree, `−lnL` by tolerance (from each
`.iqtree`): `1e-2` 11206355.72 · `1e-3` 11205864.42 · `1e-4` 11205848.85 · **`1e-5` 11205836.33 (best)** · `1e-6`
11205839.76 · `1e-7` 11205838.80. **1e-5 beats the shipped 1e-7 by 2.47 nats on the identical candidate**, and at 1e-5
the `GTR+F+R7` that *does* get fitted is 2.37 nats better still (losing to R6 on BIC by only ~22.9). The LM's terminal
state is path-dependent at the ~10-nat level; **a decision rule anchored on "matches 1e-7" anchors on a coin-flip.**

**③ BUT the topology reference IS stable — the RF rule is satisfiable.** I recomputed pairwise Robinson–Foulds directly
from the newick in `avirtolrf_174142647/` (my parser reproduces the job's own `4/0/4` for the `1e-7`-vs-`1e-2` cross, so
it is trustworthy): **RF(1e-7 s1,s2) = RF(s1,s3) = RF(s2,s3) = 0** — all three tight-tolerance trees are topologically
**identical**, despite RF 40–52/90 between their PARS starting trees. So "RF vs the 1e-7 tree = 0 on all seeds" is
well-posed, contrary to the earlier worry that the reference was itself unstable.

**DECIDED PLAN (blue-team, ~15–18 GPU-h vs the ~74 GPU-h of the original two-phase design):**
- **PREREQUISITE (source, not a cell):** an env-gated MF-path cap override at `model/modelfactory.cpp:1613`, default 400
  ⇒ byte-identical when unset. Lands in the `iqtree3-tolbanner` worktree ON TOP of the `ir-tol` banner change, so one
  binary carries both — but ONLY after the banner build (`174339144`) passes, so its bit-identity gate is not confounded.
- **J1 cap + trajectory probe** (~1.5 GPU-h): avian `-te` fixed tree, `{GTR+F+R6, R7, +I+R6}` × cap `{400, 3200, 25600}`,
  tol=1e-2 (loosest ⇒ tolerance can never mask the cap), `JOLT_IR_CONVTRACE=1`. **Stopping question = "does `conv=1` ever
  become reachable on avian +R?"** — NOT "does the winner change". Bonus: one uncapped CONVTRACE run per candidate yields
  the entire single-model tolerance ladder for free (the `dl(it)` trajectory is deterministic; tolerance only truncates
  it — exactly the read-out `irtolnt1.o174127774` CELL 0 already did), so the single-model ladder need not be re-run.
- **J2 attribution 2×2** (~8 GPU-h): avian full `-m MF`, 3 seeds × tol `{1e-2, 1e-5, 1e-7}` × cap `{400, best-from-J1}`.
  Record, per candidate: `[IRCONV]` exit reason (`CONV`/`accept`/`REJECT-EXIT`), **the fitted-candidate count and model
  list** (the actual selection channel, currently logged nowhere), and the top-2 BIC margin. Exit flips with tol at fixed
  cap ⇒ tolerance-driven; with cap at fixed tol ⇒ cap-driven; only in interaction ⇒ the mixture.
- **J3 second real dataset** (~3.4 GPU-h): euk22k `-m MF`, 1 seed, tol `{1e-2,1e-5,1e-7}` — presence/absence, not variance.
- **J4 (only if J2 yields a candidate default):** avian `-m MFP`, 3 seeds × `{candidate, 1e-7}`, RF vs the
  already-measured 1e-7 reference trees.
- **DROPPED with cause:** all Phase-2 defaults-vs-defaults on DNA-1M/AA-1M (already measured `174128914/15`, and no `+R`
  model wins there so the knob is selection-irrelevant); the `1e-4`/`1e-6` rungs (degenerate on 3/3 single-model cells);
  any tight CPU reference (`1e-4` was VOID×3 at 12,000 s on a full node — unobtainable, do not design it in again).

**SHIP-REGARDLESS DELIVERABLE (highest evidence, zero risk, one-line change): report cap-exhaustion in the ModelFinder
table.** At 1e-7 the avian table ranks a possibly-truncated `GTR+F+R6` against possibly-truncated rivals with **no record
of which converged**, and by the skip rule that one truncated fit decides whether ~45 further candidates are ever fitted.
A `capped` flag per candidate is trivial (the data already exists behind `JOLT_IR_CONVTRACE`) and makes a truncated
candidate's non-comparable BIC visible instead of silently ranked as an equal.

**WHAT THE CAP SHOULD BE — ranked by the evidence we actually have (all else UNVERIFIED until J1/J2):**
1. **Report-exhaustion + keep 400** — strongest, cheapest, zero-risk. Ship independent of everything else.
2. **Convergence-first, cap as a far-above runaway guard** — the loop's own exit is `dl<tol` and the measured `dl`
   trajectory decays smoothly; but it is UNKNOWN whether avian R6 converges at *any* cap. That is exactly what J1 answers.
3. **Data-type/model-family keyed (like the sibling's `auto(DNA:2/AA:3)`)** — the precedent exists, but the measured
   discriminator is NOT the alphabet: simulated DNA (48 iters) and AA (44) both converge while avian — also DNA — caps.
4. 🔴 **Parameter-count-aware (`base + k·nfree`) — NOT SUPPORTED (reason corrected 2026-07-21, see §1.3g).** On the same
   avian tree at tol=1e-2, `GTR+R8` shows `conv=1` at `it=342` while `GTR+R6` caps at 400 (`174328851` ARM 3 OFF arms). I
   originally read this as "iteration demand anti-correlated with `k`". **That reading is wrong** — the mechanism probe
   proved `conv=1` here is a *transient-dip lottery*, not convergence: both models crawl at the SAME ~0.1–0.8 nats/iter,
   and R8 tripped `dl<tol` only because one oscillating step hit `3.14e-3` while R6's best dip was `0.022` (just above the
   `0.01` threshold). Neither is converged. So a `k`-scaled cap is still not the fix — but because the pathology is
   degeneracy/tolerance-driven and `k`-agnostic at the shipped `tol=1e-7` (where EVERY high-k `+R` caps), not because
   high-k needs fewer iterations.

**AND THE STRATEGIC ANSWER TO "is a stable GTR+F+R6 worth the speed?"** — the data does not yet support that `GTR+F+R6` is
the right answer at all: at 1e-7 it wins partly because `GTR+F+R7` is never fitted (①); when R7 *is* fitted it beats R6 on
likelihood (②); on a fixed tree `GTR+R8` beats `GTR+R6` by ~600 nats (`174328851` ARM 3). So "a stable R6" may be a stable
*under-fit*. The honest deliverable is the identifiability diagnostic + the cap-exhaustion flag + an upstream bug report,
**not** a tolerance default chosen to reproduce R6. The measured price of keeping 1e-7: avian `-m MF` 778→230 s vs 1e-2
(3.38×), DNA-1M MFP nt12 2144.8→842.4 s (2.55×).

### 1.3f IMPLEMENTATION CONTRACT (so the build is mechanical once banner job `174339144` clears)

All of this lands in the `iqtree3-tolbanner` worktree (branch `tol-banner`), ON TOP of the `ir-tol` banner change, in ONE
rebuild — but only after `174339144` has proven the banner change is bit-identical-when-OFF, so a failure can be localised.

**(a) MF-path cap override — the prerequisite for J1/J2.**
- Seam: `model/modelfactory.cpp:1613` `tree->optimizeParametersJOLT(fixed_len);` — the ONLY caller that inherits the 400
  default. Read a new env there and pass it as the 4th arg.
- Env name: reuse `JOLT_BRLEN_MAXITER`? **NO** — that name already means the tree-search per-round cap
  (`phylotreegpu.cpp:2965`); overloading it would silently couple two unrelated knobs. Use a NEW name `JOLT_MF_BRLEN_MAXITER`.
- Semantics: unset ⇒ pass nothing ⇒ the 400 default is used ⇒ **byte-identical to `f3f7875f`**. Positive int ⇒ that value.
  Non-positive/garbage ⇒ fall back to 400 (mirror the `jolt_resolve_ir_tol` garbage policy; never let a bad env silently
  disable the cap).
- Proof-of-build: `strings|grep -c JOLT_MF_BRLEN_MAXITER` ≥1 in the new binary AND 0 in canonical, PLUS a **measured**
  iteration-count change on avian `GTR+R6` (400 → the override) — a string alone does not prove it is plumbed.

**(b) Cap-exhaustion flag in the `.iqtree` ModelFinder table — the ship-regardless deliverable.**
- The optimizer already knows: `[IRCONV]` (`gpu_lnl_intree.cu:3453`) emits per-candidate
  `exit={CONV(dl<tol)|accept|REJECT-EXIT}` and `conv={0,1}`; a candidate that runs to `it==maxiter` with `conv=0` is
  truncated. Surface a per-candidate `capped` boolean up to the table writer so a truncated fit is visibly marked.
- Default-ON is acceptable (it is a *report* — changes no lnL/BIC/selection), but gate it byte-identical on the numbers:
  the flag must not perturb one lnL bit, only add a column/annotation.

**(c) J1/J2 telemetry contract (already emitted, just parse it).** Per-candidate terminal `[IRCONV]` line gives `it=`,
`conv=`, `exit=`. Real sample from `tolladder_174328851/a1_1e-7_av.console`:
`[IRCONV] it=1 muIn=1.000e+00 muOut=5.000e-01 rej_it=0 acc=1 conv=0 lnL=-11550559.197332 exit=accept`. A run terminates
`capped` iff its last `[IRCONV]` line has `it==maxiter && conv==0`. J2 additionally reads the fitted-candidate count/list
from each `.model.gz` (`best_model_list_BIC:` line) — the actual selection channel, logged nowhere else today.

**(d) CONVERGENCE-TEST FIX (new, from §1.3g — a genuine correctness win, not approximate-ranking).** The accept sites
`gpu_lnl_intree.cu:3404/:3450` set `conv=true` on a SINGLE `dl<tol`. §1.3g proved that is a transient-dip lottery (R8
tripped on one 3.14e-3 step while crawling at 0.1–0.8 nats/iter; R6 didn't and capped). Change: require `dl<tol` on **N
consecutive** accepted steps (start N=3) before declaring convergence — or add a gradient-norm gate. Default-OFF behind a
flag (`JOLT_IR_CONVN`, default 1 ⇒ byte-identical to today's single-step test) so it can be A/B'd, then flipped to N=3 if
J1/J2 show it removes the non-monotone conv/cap flip without changing a converged winner. This is *strengthening* the
convergence criterion (fewer false "converged"), the safe side of [[feedback_approximate_ranking_fails_real_data]] — it
removes work-stops that were premature, it does not stop real work early. Proof-of-build: measured change in `conv`
outcome on `off_av8` (should flip `conv=1@342` → keep crawling, since its 3.14e-3 dip is not sustained 3×).

### 1.3g DEEP CAP RESEARCH — what the cap does to the models and the +R family (2 agents, 2026-07-21; every load-bearing number re-verified by me on disk)

Two disjoint research tracks (census of WHICH models cap; mechanism of WHY/whether-more-helps). They **disagreed** on
one point and I resolved it on disk. Summary of what survives verification:

**MECHANISM (verified — this is the deepest finding).** A capped `+R` fit is **not stalled and not mu-thrash — it is a
slow linear crawl still descending on a near-degenerate ridge.** Read directly from `off_av6.console`/`off_av8.console`
(avian, fixed tree, `JOLT_IR_TOL=1e-2`):
- Every one of the 400 outer LM steps is **accepted** (`acc=1`); lnL keeps improving ~0.1–0.8 nats/iter to the cap. It is
  NOT a stall. The prior "62% rejects" claim is FALSE — inner line-search reject fraction ≈0.34, outer accept = 400/400.
- 🔴 **The `conv=1` flag is a TRANSIENT-DIP LOTTERY, not convergence.** Both R6 and R8 crawl at the same magnitude. R8
  shows `conv=1` at it=342 only because one oscillating step hit `dl=3.14e-3 < tol(1e-2)`; R6's best dip was `0.022` at
  it399 — a hair above the `0.01` threshold — so it kept crawling and capped at 400 still moving `dl=0.737`. Neither is a
  converged MLE. (This corrects my earlier "iteration demand anti-correlated with k" — §1.3e item 4.) The census agent
  wrongly reported this "reverses" across a tau grid; that grid is HARP-ON (`a3_*` = `JOLT_HARP=1 JOLT_HARP_TAU=…`, tol
  held at 1e-2, `[HARPSTEP]` injecting projected steps) — a HARP confound, not clean cap behaviour. Discarded.
- **Root cause = near-duplicate rate classes.** Fitted avian `+R6` has two categories at rate 1.289/1.312 (one class split
  in two) ⇒ a near-null OFF-diagonal Hessian direction the **diagonal/Jacobi LM cannot precondition** (L-BFGS is disabled
  for `+R`, `gpu_lnl_intree.cu:3224`) ⇒ linear (arithmetic), not geometric, convergence. Control on disk: `GTR+R5` =
  −59403389.130 vs `SYM+R5` = −59403408.260 (identical model) **19.13 nats apart** — a real convergence-state gap.
- **Raising the cap recovers bounded lnL but NO STABLE OPTIMUM (extrapolation — no cap>400 MF run exists).** Est. ~100–300
  nats of remaining headroom, but the within-model lnL spread from where the crawl stops (~337 nats R6 / ~477 nats R8) is
  **20–40× the R6-vs-R7-vs-R8 BIC margin (~11–14 nats)**. So a bigger cap perturbs each candidate by MORE than the gap that
  orders them ⇒ can flip selection *unpredictably*. **"Just raise the cap" is not a clean fix.**

**CENSUS (verified — the family/data picture).** The cap is a **tolerance × degeneracy artefact, not an intrinsic model
property**:
- Same real euk-AA data: **0% capped at tol 1e-2 → 71.4% capped at tol 1e-7.** DNA-1M-sim at default = 69.5% of ALL
  optimiser work spent inside capped `+R` candidates.
- Onset is **gradual and data-dependent**, always at the top of whatever k-ladder MF climbs: **R5** on DNA-1M-sim (nothing
  ≤R4 ever caps), **R8** on real euk-AA, rising to ~84% by R10. Not a sharp threshold.
- Predictor is the **constant-site fraction / degeneracy**, NOT the alphabet: avian 37.9% const → caps; euk 12.7%;
  sim-DNA 1.5% → converges cleanly. Sim DNA `GTR+R4` and sim AA `LG+R4` both converge (~44–48 iters).
- **Selection consequence splits by data realism:** on SIMULATED data the winner is always a fast-converging `+G4` model
  and every capped `+R` **loses** ⇒ the cap is **pure wasted wall, selection untouched**. On REAL data the winner IS a
  high-k `+R` fit (euk `LG+I+R9`, avian `GTR+F+R{k}`) ⇒ the cap costs wall (up to 14× iters) AND, on avian, **selection
  stability** (the truncated lower-k fit is deflated, which flips which near-tied model wins — §1.3e ①).

**THE FIX THE EVIDENCE SUPPORTS (revised — NOT "just raise the cap"):**
1. **Fix the convergence TEST** (primary, cheap, and a genuine correctness win — not approximate-ranking): the
   single-iteration `dl<tol` check is the lottery. Require `dl<tol` over *k consecutive* iterations, or a gradient-norm
   criterion. Removes the non-monotone conv/cap coin-flip so `conv=1` means something.
2. **Report the degeneracy** (primary, ship-regardless): cap-exhaustion flag in the `.iqtree` table + the λ_min
   identifiability diagnostic ("R6/R7 statistically indistinguishable"). The flat direction is a real data property.
3. **Off-diagonal optimiser conditioning would help speed but CANNOT remove the ridge — and it is the KNOWN-DEAD path.**
   This is exactly what the OPG/empirical-Fisher +R solver attempted, and it FAILED all three repair routes
   ([[project_freerate_conditioning]]). The mechanism track independently rediscovering "off-diagonal preconditioning is
   the theoretically-right fix" is a cross-check that the honest deliverable is **report-only + conv-test-fix + upstream
   report**, not another optimiser rewrite.

⇒ **NET: the 400 cap is not the disease, it is a symptom.** The disease is a degenerate `+R` likelihood ridge that the
diagonal LM crawls linearly and a one-shot `dl<tol` test reads as converged-or-not by luck. J1/J2 still measure the
cap-raise (to bound the recoverable lnL and confirm the extrapolation), but the SHIPPABLE outputs are the conv-test fix
and the cap-exhaustion flag, both added to the §1.3f implementation contract.

### 1.4 Selection instability propagates to topology (job `174142647`)

avian `-m MFP`, tight vs loose × seeds{1,2,3}: **RF = 4, 0, 4.** The tight arm is stable (same winner all three
seeds, lnL spread **0.16 nats**); the loose arm is not (spread **704 nats**). ⇒ +R selection instability is not
purely cosmetic — it can move the tree.

---

## 2. WHY IT IS SLOW — mechanism, source-grounded

### 2.1 GPU/JOLT path: a diagonal preconditioner on a near-singular block

The +R joint optimiser carries three parameter arms: branch lengths (`nedge`), log-rates `y` (`ncat`), and
softmax weight-logits `z` (`ncat`, `gpu_lnl_intree.cu:3209` `zR[c]=log(bprop[c])`). The update
(`gpu_lnl_intree.cu:~3417`) is:

```cpp
double ny = baseY[c] + g_y[c]/(fabs(ddY[c])+mu);   // rate arm  — per-category DIAGONAL
// ... and the same shape for the z (weight-logit) arm
```

**Each rate and each weight is stepped independently, scaled only by its own diagonal second derivative.
There is zero off-diagonal coupling.** Compounding it, `gpu_lnl_intree.cu:3224`:

```cpp
const int lbM = (freeRate==1) ? 0 : g_lbfgs_m;   // +R always uses the diagonal path
```

⇒ **L-BFGS — the only convergence accelerator in the optimiser — is explicitly DISABLED for the family that is
90.9% of the work.** +R has nothing but the diagonal/LM path.

### 2.2 Why a diagonal preconditioner fails *specifically* at k > true class count

The measured onset at **k=5** is not an architectural switch (`JOLT_FREERATE_HIGHK` gates eligibility only, cap 10,
identical for R4 and R5). It is **statistical**. The fitted +R5 on data simulated under GTR+I+G4 (**4 real rate
classes**) was:

```
rate/weight:  0.132/0.221 · 0.1932/0.032 · 0.4805/0.247 · 1.002/0.250 · 2.381/0.250
                  └────────── categories 1 & 2 are ONE class split in two ──────────┘
```

When k exceeds the number of rate classes the data supports, two categories become near-duplicates. The likelihood
then has a **near-null direction**: rate/weight mass can be traded between the duplicate pair at almost no cost.
That degeneracy lives **entirely in the off-diagonal** of the Hessian. A diagonal preconditioner sees large
individual curvature for each component, takes small steps in each, and **never moves along the valley** — giving
linear convergence measured at **ρ ≈ 0.975**, i.e. ~270 iterations to gain one decimal digit. Hence the 401 cap.

### 2.3 CPU path: under-converged by construction (verified in `model/ratefree.cpp`)

The CPU is not a clean reference — it has the same disease via a different route. `RateFree::optimizeWithEM()`
(the default path, `:506`), whose own comment cites *"the EM algorithm described in Wang, Li, Susko, and Roger (2008)"*:

| property | source | consequence |
|---|---|---|
| `for (int step = 0; step < ncategory; step++)` | `:540` | **EM budget = k steps for a k-component mixture.** EM converges *linearly*; k steps is nowhere near convergence for large k. |
| `if (zero_prop) break;` | `:600` | EM **aborts** the moment any proportion hits `MIN_PROP = 1e-4`. More categories ⇒ higher chance of collapse ⇒ **higher R terminates earlier**. |
| `fabs(prop[c]-new_prop[c]) < 1e-4`, `fabs(rates[c]-scaling) < 1e-4` | `:608`, `:652` | **Absolute** convergence tests. Individual props shrink as ~1/k, so a fixed absolute bar is **relatively looser at higher R**. |
| `max(gradient_epsilon, TOL_FREE_RATE)`, `TOL_FREE_RATE = 0.0001` | `:353`, `:18` | A **hard tolerance floor** on the BFGS path — you cannot ask CPU +R for more precision than 1e-4. |
| per step, per category: `copyPhyloTree` + `initializeAllPartialLh()` + `optimizeTreeLengthScaling` | `:628-652` | cost ~**O(k²) full-tree operations** per call; the R2..R10 ladder is then ~O(R³). |

⇒ **Three independent structural reasons CPU +R is under-converged, and all three get WORSE as k rises.** This is
the mechanistic explanation for §1.3's empirical result.

🔴 **AND THE CITED JUSTIFICATION DOES NOT COVER THE CODE (literature agent, verified against Wang 2008 full text).**
The comment at `ratefree.cpp:540` cites *"the EM algorithm described in Wang, Li, Susko, and Roger (2008)"* (BMC Evol
Biol 8:331) for a global-optimum guarantee. **That paper's EM optimises mixture WEIGHTS ONLY** — a concave subproblem
on the simplex, which genuinely does have a unique global optimum. But `optimizeWithEM` **also updates the rates**
(via `optimizeTreeLengthScaling`, `:652`), and the joint (weights + rates) surface is **non-concave and multi-modal**
— outside Wang 2008's scope. The `for (step < ncategory)` budget corresponds to **nothing** in the cited paper, which
says only *"until they converge."* IQ-TREE's own release notes concede the history: v1.3.6 introduced the EM to fix a
+R *"local optimum"* problem, and v1.3.9 shipped *"a more stable EM"* for +R three releases later — the developers
already know +R optimisation is hard. So the k-step budget is an **undocumented heuristic**, and the global-optimum
claim is **defensible for weights, over-stated for +R**. EM convergence theory agrees the budget is insufficient:
mixture EM is **linearly** convergent and **"prohibitively slow when components are poorly separated"** (Redner &
Walker 1984, SIAM Review 26:195) — exactly the +R regime as rates crowd. There is **no result anywhere** that k steps
suffice for a k-component mixture.

---

## 3. WHY IT WOBBLES AT HIGHER R

Combining §2.2 and §2.3, the wobble has a single root with two expressions:

> **When k exceeds the number of rate classes the data supports, the FreeRate likelihood becomes near-unidentifiable.
> The Fisher information matrix of the rate/weight block acquires a near-zero eigenvalue. Every optimiser — CPU EM
> and GPU diagonal-LM alike — then terminates at an essentially arbitrary point along a flat valley, and *which*
> point depends on the seed, the tolerance, and the truncation budget.**

This predicts, correctly, all four observations: the k=5 onset (§1.1), the GTR/SYM 19.13-nat split (§1.2), the
three-seeds-three-winners result on real data (§1.3), and the near-degenerate BIC landscape (top-2 within
82 BIC of 22.4M ≈ 3.7e-6).

### 🔴 3.1 The one question that decides whether this is fixable

There are **two candidate diagnoses**, and the evidence to date does **not** cleanly separate them:

| | diagnosis | prediction | consequence |
|---|---|---|---|
| **T** | **Truncation** — one basin, but every run stops at a different point along a slow crawl | a better-conditioned optimiser makes all runs agree | **the wobble is FIXABLE and RF=0 is achievable** |
| **M** | **Multi-modality** — genuinely distinct local optima | better conditioning converges faster, to *different* optima | **the wobble is INHERENT; the honest deliverable becomes reporting it, not removing it** |

The GTR/SYM control (§1.2) is evidence for **T** — those two runs *should* be identical, and a 19.13-nat gap in a
model with no basin structure difference points at truncation. But that is one datapoint on one model.

🔴 **There is a published warning against blithe optimism here, and it comes from Minh's own group.** **Nguyen, von
Haeseler & Minh (2018), Syst Biol 67(3):552** ("Complex Models of Sequence Evolution Require Accurate Estimators…")
characterised the neighbouring **+I+G** likelihood surface and found **two distinct peaks separated by a flat valley**,
with PhyML/RAxML/IQ-TREE/MrBayes all sometimes failing to estimate the parameters (89.2% of RAxML's 6-taxon estimates
worse than truth). That is **diagnosis M for +I+G**. Since +I and +R's slowest category are confounded (a near-zero-rate
+R category mimics an invariable-sites class — the +I/+R confound, which the agent confirmed is **undocumented** but
must operate by the same mechanism as the well-studied +I/+G one), **the +I+R family is the most likely to be genuinely
multi-modal**, and P0 may well resolve against us *there*.

**A refinement this forces, and it is the honest core of the design.** The pure-+R near-duplicate-category degeneracy
(§2.2) is a **flat valley** — a connected continuum of near-equal optima — which is *neither* clean T *nor* clean M.
On a truly flat valley, a better-conditioned optimiser converges *faster* but still to a seed-dependent point unless it
**also picks a canonical point on the valley** (regularise the near-null direction, or take the min-norm solution). So
the deliverable is not "condition better and hope"; it is **detect the flat direction via λ_min (§4.4) and resolve it
deliberately.** That is why the identifiability diagnostic is load-bearing, not decorative.

**§5 makes P0 the pre-registered primary experiment. I will not claim the wobble is fixable until T is confirmed on
pure +R, I expect +I+R to be harder, and if M holds I will say so and the RF=0 headline claim narrows to a detection
claim (§4.4).** The Nguyen 2018 method — map the surface, count the peaks — is also the exact template for doing this
rigorously, and it is a citable precedent from the tool's own authors.

---

## 4. PROPOSED SOLUTION — OPG empirical-Fisher conditioning of the rate/weight block

### 4.0 🔴 PRIOR ART, IN THIS PROJECT — reconciled after red-team (I missed it the first time)

**The optimiser core below is NOT new — it was already designed here.** `research/Modelfinder/JOLT-OPG-FISHER-OPTIMIZER.md`
(2026-07-15) specified the identical construction: OPG/BHHH empirical-Fisher curvature from the per-pattern scores
already on-device, Marquardt `λ·diag(H)` damping, and a Nielsen gain-ratio trust region, scoped to the model arm with
branch lengths left on their existing `g_ddf` curvature. I wrote §4.1–4.4 without finding it — a failure of my own
"check the artifact on disk" rule — and in doing so **got two things wrong that the prior doc got right** (block
dimension; the "free" claim). This section now inherits that design rather than re-deriving it.

**What that prior doc was, and why it was shelved — this matters for scope.** It targeted the **④ 0.07-nat AA +I+R
convergence gap**, which was then measured **selection-irrelevant** (real avian margin 19×–14000× the gap), so it was
recorded and **not built**. It was shelved because *its target died*, not because the optimiser failed. **This document
re-points the same machinery at a target that is very much alive: the 401-cap wobble and the RF=0 reproducibility
demand.** That re-targeting, plus the identifiability diagnostic (§4.4), is what is actually new here.

**Inherited verbatim from the prior doc (do not re-invent):** ✅ curvature = OPG `H = Σ_p freq_p·s_p·s_pᵀ` (always PSD,
self-scaling, correct at the MLE by the information-matrix equality, BHHH 1974); ✅ damping = **Marquardt `λ·diag(H)`**
not `λ·I` (inherently weight-aware, no magic κ); ✅ **λ adapted by the Nielsen gain-ratio** ρ = actualΔ/predictedΔ —
this **replaces the ×4-up/×0.5-down `mu` ratchet that inflates to 1e9**, and is precisely the mitigation the red-team
flagged my version as lacking; ✅ brlen stays on `g_ddf`; ✅ Phase-0 gate-0 = `JOLT_RGRADCHECK` (`:2869`) FD-checks the
+R gradient (`maxrel<1e-4`) BEFORE any solve, because OPG is only as good as the gradient it squares.

### 4.1 The idea (corrected)

Replace the per-category **diagonal** step on the `(y, z)` arms with a **dense OPG-Newton solve on the reduced
rate/weight sub-block**, branch lengths unchanged. 🔴 **The λ_min diagnostic lives in a `(2k−1)` space, NOT `(2k−2)`
(blue-team, verified on disk 2026-07-19 — I had it wrong TWICE).** The *model* has `getNDim()=2k−2` free params
(`phylotreegpu.cpp:2214`), but the OPG Gram is built over the `(y,z)` coordinates **at fixed branch length**, and only
**ONE** of the two gauges is a structural null of *that* block: the softmax shift `n_z=(0_k;1_k)` (`:3200`) is an exact
zero of `H` (`s_p·n_z = Σ_c ∂logL_p/∂z_c = 0` every pattern; cross-checks at gradient level `Σ_c gzR[c]=0`, verified
`:3150`). The mean-rate gauge `n_y=(1_k;0_k)` (`:3193` `gaugeFix`) is **NOT** a null of the fixed-brlen block — it is a
*large-curvature* direction there (rate scale is well-determined once brlen is frozen), so it cannot contaminate
`λ_min`. ⇒ project out **only** `n_z` (rank-1, `P = I − n_z n_zᵀ/k`), then `λ_min(P H P)` over the `(2k−1)` complement
is the identifiability signal. My earlier "project both gauges → 2k−2" reasoning was factually wrong for `n_y`.

### 4.2 The curvature is built from partials already on the device

For a mixture `L_p = Σ_c w_c L_c(p)`, the per-pattern scores `s_p = ∂log L_p/∂θ` over θ = (rates, weights) are
**already computed** — `gpu_lnl_intree.cu:2820-2823` allocates the per-category `L_c(p)` numerator `d_wnum`; `:3011-3013`
reduces the responsibilities `r_c(p)=L_c(p)/L_p` to their column sums `WNc[c]`. OPG accumulates the **outer products**
`Σ_p freq_p·s_p·s_pᵀ` instead of those plain sums, reusing the existing deterministic `kj_reduce*` machinery. **No new
likelihood evaluations, no finite differences.** But it is a **new outer-product-before-reduce kernel** (the current
`kj_reduce_gradnum` is a k-fold reduce, not a (2k−2)² one) — not merely "the same reduction widened."

### 4.3 Cost — "tiny," not "free" (red-team correction)

The dense solve is `(2k−2)³` ≈ trivial. The real cost is the **outer-product accumulation**: up to `(2k−2)(2k−1)/2 ≈
150` accumulators per pattern at k=10, with a shared-memory footprint of `blockDim × (2k−2)²` that **can cut occupancy**.
For DNA (ns=4) the per-pattern Gram work (~4k²) is **comparable to the likelihood arithmetic (~16k)** at high k; for AA
(ns=20) it is dwarfed. Plausibly hidden on a memory-bound kernel, cheaper on AA than DNA — **but this must be MEASURED,
not asserted.** P5 (wall) is a live falsifier, not a formality.

### 4.4 ⭐ The genuinely new part — identifiability made measurable (corrected coordinates)

`λ_min` of the OPG block is a per-candidate measure of how over-parameterised `+R{k}` is on this data — **the one
contribution neither the prior in-project doc nor any published tool provides.** 🔴 **But it MUST be computed in the
reduced (2k−2) tangent space** — a raw 2k×2k block carries **structural** near-zero eigenvalues from the two gauge
directions (§4.1), so `λ_min ≈ 0` **always**, identifiable or not. Project out the softmax and mean-rate gauges first;
then a residual small `λ_min` genuinely signals that the data does not support k classes. Get this wrong and the
diagnostic is contaminated by construction (this was my §4.4 error).

With that fix, it yields: a cheap per-candidate identifiability score; a principled answer to *"how many rate
categories does this alignment support?"* without fitting the whole ladder; and an honest reporting mode — *"R7 selected
but λ_min≈0 ⇒ R6/R7 statistically indistinguishable"* — which is a better scientific answer than silently returning one
arbitrary basin, and is exactly what §1.3 calls for. **This, not the speedup, is the strongest novelty candidate.**

⚠️ **λ_min caveat (two parts, red-team-refined):** (1) OPG/empirical-Fisher equals the true information **only at the
MLE** (BHHH); a capped candidate is far from it, so an early λ_min is a noisy proxy — trustworthy only once converged.
(2) 🔴 **My "1e-6 gradient accuracy might corrupt λ_min" worry was MISDIRECTED (red-team):** the 1e-6 is the *FD
validation's* floor (central-diff cancellation on lnL~1e6), NOT the production gradient's accuracy — OPG uses the
*analytic* FP64/Kahan gradient (~1e-12), so `‖ΔH‖~2δ·λ_max` with δ~1e-12 is fine. **BUT the residual caveat is real:**
RGRADCHECK only *proves* correctness to ~1e-6, so a genuine bug of size 1e-12…1e-6 passes yet corrupts λ_min at
~1e-6·λ_max. ⇒ **λ_min is trustworthy for the coarse identifiable/not call, NOT for the fine near-degenerate boundary
(λ_min/λ_max ∈ 1e-6…1e-10).** Claim the former, not the latter.

### 4.5 What this design explicitly does NOT do — and the honest bound on the upside

- ❌ does not change the candidate set, the criterion, or the comparison; ❌ no screen/subsample/approximate ranking;
  ❌ does not trade CPU-authoritative scores for GPU-trusted ones; ✅ changes **only** the search direction to each MLE.
- ⚠️ **Bounded upside, stated honestly (from the prior doc):** the model arm is only **8–30% of per-model wall**; branch
  lengths are 75–85%. Conditioning (y,z) cuts iteration *count*, but each surviving iteration still pays full brlen cost.
  So this is a **convergence/reproducibility fix with a bounded speed component**, not a headline speed lever — and it
  does **not** by itself deliver "same walltime": the tol=1e-2 win (already banked, 3.55× DNA) pays for the walltime
  budget; Lever A must fit inside it. Keep the two accountings separate (§7).

---

## 5. GATES AND PRE-REGISTERED FALSIFIERS

Written **before** any build, so results cannot be rationalised afterwards.

🔴 **P0 is DEMOTED (red-team): it is necessary but NOT sufficient, and cannot alone greenlight the build.** It proves
*non-convergence exists* (verified: two evals of literally-identical `SYM+R5` differ 0.004 nats, so the 19.13-nat
GTR/SYM gap is a real convergence-state difference, not noise or label-switching) — but a 19.13-nat gap is consistent
with **both** truncation (T) **and** multimodality (M); its existence discriminates neither. And P0 tests *within-model*
agreement of one pair on one dataset, whereas the wobble that matters is the *cross-model* winner flip (R4/R6/R7),
which for avian is a **genuine BIC near-tie** (top-2 within 82 BIC of 22.4M) — **a third case the T/M split omits**,
where no optimiser improvement helps and **only the λ_min report does.** **P0 and P4 are CO-PRIMARY.**

⭐ **The strongest pro-T evidence is not P0 — it is already on disk:** at the *tight* tol=1e-7 the GPU gives a
seed-stable avian `GTR+F+R6` (lnL spread **0.16 nats** across 3 seeds, WORK-LOG §112); the loose tols wobble. Full
convergence *already* stabilises the winner — which is what "fixable" needs, and is better evidence than the GTR/SYM
pair.

| # | test | pass | 🔴 **falsifier — what kills the design** |
|---|---|---|---|
| **P0** (co-primary, cheap, no build) | GTR+R5 vs SYM+R5 agreement (§1.2), identical-model control | gap < `modelfinder_eps` = 0.1 nats (from 19.13) | gap ≫0.1 ⇒ non-convergence confirmed but *not* discriminated; read together with P4 |
| **P4** (co-primary) | **cross-seed WINNER stability on real avian, ≥3 seeds** — the actual user ask | same winner every seed | differs ⇒ wobble not fixed (or it is the genuine-near-tie third case ⇒ λ_min-report deliverable, not RF=0) |
| **P1** | iterations on the 21 cap-bound candidates | no candidate at the 401 cap | still capped ⇒ conditioning did not address the crawl |
| **P2** | **selection invariance** vs the all-tight reference, DNA + AA + **real avian** | identical winner **and** top-10 BIC order | 🔴 **this is a genuine recall risk, NOT a formality** — a state-changing optimiser that moves a winner is on the risk side of §7's law (see §7); a moved winner must be provably the correct AND seed-stable MLE, not merely "more converged" |
| **P3** | **RF=0** vs the tight reference, real avian, **≥3 seeds** | RF=0 on every seed | RF>0 ⇒ headline claim fails |
| **P5** | wall time | +R MF wall materially below the diagonal path | flat ⇒ iterations were not the wall after all; redirect and say so |
| **P6** | OFF byte-identity | flag-off run byte-identical to canonical | any drift ⇒ not shippable |
| **P7** | `-p` partitioned cell | ON == OFF winner **per partition** | **mandatory** — this project has shipped 3 partition-blind gates |
| **P8** | λ_min diagnostic sanity | λ_min small exactly where selection is unstable (avian R5–R7), large where stable (sim +G) | no correlation ⇒ the diagnostic claim (§4.4) is unsupported and must be dropped |

**Gate-construction rules carried forward from this project's own failures:** proof-of-build sentinel via
`strings | grep -c` (never `grep -q` — SIGPIPEs under `pipefail`; RF=0 is **not** a proof-of-build); every marker
must report a **measured count**, never a set size or an intent; assert **scored-count == candidate-count**; the
CPU control must pin `--no-jolt` and assert `Kernel: AVX+FMA`; **fail-closed** with an accumulator and a non-zero
exit; and **real data is mandatory** — simulated cells cannot gate this class (§7).

---

## 6. LITERATURE

### 6.1 Competitive tools — ✅ LANDED (agent, 2026-07-19; source code read, not just abstracts)

| tool | +R? | +R optimiser | k-selection | reproducibility discussed? |
|---|---|---|---|---|
| **IQ-TREE / ModelFinder** (Kalyaanamoorthy 2017) | yes | EM on weights → optimise-all-branches; +I+R | ladder R2..R10, BIC/AIC | not in paper; only GH Issue #38 |
| **RAxML-NG** (Kozlov 2019) | **yes** | **`--opt-freerate {bfgs, em-bfgs, em-brent}`**; default **EM-Brent** in selection; EM `max_steps=10`, weight-tol 1e-4 | ladder, BIC/AIC | paper minimal; **wiki: EM is "faster but usually less accurate" during model selection**; v1.0.2 fixed FreeRate memory corruption |
| **ModelTest-NG** (Darriba 2020) | yes | pll-modules (same stack as RAxML-NG); not stated in paper | not stated | no |
| **jModelTest2** / **PartitionFinder2** | **no** (+I/+G/+I+G only) | via PhyML/RAxML | n/a | no |
| **PhyML** (Guindon 2010) | yes (FreeRate origin, Soubrier 2012) | not detailed | user k, default 4 | no |
| **ModelTamer** (Sharma & Kumar 2022) | yes (wraps IQ-MF) | inherits IQ-MF | on subsample | **no — and it ADDS randomness without analysing it** |
| **BEAGLE v4 tensor cores** (Gangavarapu 2026) | n/a (fixed +G4) | GPU likelihood + gradient only | **does not do model selection** | n/a |

### 6.2 🔴 Honest correction to my novelty framing — a joint step is NOT itself novel

**RAxML-NG already provides a joint quasi-Newton +R optimiser** (`--opt-freerate bfgs`, L-BFGS-B on rates+weights
together). So "couple the rate/weight arms instead of stepping them diagonally" is **prior art, not our idea**, and
this document must not claim it. What remains genuinely ours, and survives the literature:

1. **GPU-resident model selection at all.** The agent found **no published GPU ModelFinder / GPU model-selection
   work of any kind** — every GPU phylogenetics paper (BEAGLE 1→4, tensor cores 2026, MrBayes/GPU, Suchard-Rambaut
   2009) accelerates likelihood / gradient / MCMC / tree search under a **fixed** model. High-confidence gap.
2. **The specific cheap construction:** the empirical-Fisher block as the **Gram of responsibilities already
   resident on the device** (`d_wnum`, §4.2). RAxML-NG's `bfgs` builds curvature by other means on CPU; ours is a
   free byproduct of a reduction the kernel already runs. This is the engineering novelty.
3. **λ_min as a per-candidate identifiability diagnostic** (§4.4) — no tool exposes this; the alternatives to the
   k-ladder in the literature are all Bayesian-nonparametric (slow: Susko 2003 NPML; Gill 2025 DP/HDP/iHMM;
   Huelsenbeck & Suchard 2007) or DL shortcuts that **drop +R entirely** (ModelRevelator 2023). **No fast,
   deterministic, ML method picks the number of free-rate categories in one pass.**
4. **Pairing +R accuracy WITH reproducibility.** RAxML-NG *documents* trading +R accuracy for speed in selection but
   is silent on reproducibility; IQ-TREE has the accurate optimiser but the instability we measured. **Nobody claims
   both fast and RF=0-reproducible +R selection.**

⚠️ **A caution the agent surfaced that the red-team must weigh:** RAxML-NG had a joint L-BFGS-B option and
**chose EM-Brent as the selection default anyway** — i.e. a joint step was available and *not* preferred. Our
Fisher-block step must demonstrably beat that choice, not merely exist. If it does not, this reduces to re-deriving a
path the field already declined.

### 6.3 Reproducibility & the k-ladder — the three defensible gaps (agent, high-confidence negatives)

- **No published work documents seed-dependent / non-reproducible *model selection*** (as opposed to tree search).
  The nearest are bit-reproducibility of the likelihood across MPI core counts (RAxML-NG, bioRxiv 2025.06.02.656320)
  and cross-*software* agreement (Li et al. 2025, PLOS ONE) — both different axes. **Our avian three-seeds →
  three-+R-winners result has no published analogue.** Informal corroboration only: **IQ-TREE GitHub Issue #38**
  ("Frequent failure of optimization for free rates site model") documents directional +R non-convergence against a
  hard iteration cap — community confirmation that §1–2 is real and known, but not a peer-reviewed treatment.
- **Fast deterministic k-selection is unsolved** — every alternative to fitting the ladder is either slow (Bayesian
  nonparametric) or drops +R (DL). This is the methodological opening §4.4 aims at.

### 6.4 What to borrow / benchmark against

- **RAxML-NG `--opt-freerate` is the ready-made cited baseline.** Benchmark our step against `em-brent` (their
  default), `em-bfgs`, and `bfgs` (their joint option — the honest comparator for §4.1).
- **ModelTamer's subsample→upsample is orthogonal and composable** — but it is on the retired-in-this-project
  subsample axis (§7, CTF), so treat it as related work to cite, **not** a lever to adopt.
- **Susko 2003 (NPML adequacy LRT) and Gill 2025 (let-data-choose-k)** are the principled citations for *why* the
  fixed-k ladder is unsatisfactory — motivation for §4.4.

### 6.5 EM theory & accelerations — ✅ LANDED (agent, 2026-07-19)

**EM convergence theory (all bibliographically verified):** EM increases the likelihood **monotonically** and
converges **linearly**, at a rate set by the fraction of missing information (Dempster/Laird/Rubin 1977, corrected by
Wu 1983). For finite mixtures specifically, **Redner & Walker (1984)** state EM is "prohibitively slow when components
are poorly separated" and explicitly recommend Newton/quasi-Newton instead — poorly-separated components is exactly
the +R regime as k grows. **Balakrishnan/Wainwright/Yu (2017)** give geometric convergence **only within a basin of
the true parameter** — outside it there are no guarantees, so initialisation matters. ⇒ **the k-step budget (§2.3) is
theoretically unjustified and known-insufficient in the hard regime.**

**Proven mixture-EM accelerations, ranked by track record — and NONE has ever been applied to +R in phylogenetics
(agent's novelty gap #4):**
1. **SQUAREM** (Varadhan & Roland 2008, Scand J Statist 35:335) — squared vector extrapolation over two EM steps;
   **globally convergent, monotone, superlinear**, wraps any EM map with minimal change. Mature R package.
2. **ECME** (Liu & Rubin 1994) — matches IQ-TREE's existing weights-then-rates alternation; monotone; faster than EM.
3. **Quasi-Newton EM** (Jamshidian & Jennrich 1997) / direct BFGS on the (2k−2) params — well-founded for
   poorly-separated components; IQ-TREE already has a partial BFGS path (`TOL_FREE_RATE`).
4. **Aitken** (Louis 1982) — cheapest bolt-on; also yields the information matrix.
5. **DAAREM** (Henderson & Varadhan 2019) — reportedly fastest, least battle-tested.
6. **Problem-structure** (Susko/Lincker/Roger 2018, MBE — star-tree / composite-likelihood surrogate) — the only one
   *proven in phylogenetics*, but bespoke, not drop-in.

**Identifiability foundations:** finite-mixture identifiability (Teicher 1963); phylogenetic mixtures form a convex
polytope and non-identifiable branch-length mixtures are "not rare" (Matsen/Mossel/Steel 2008); GTR+Γ identifiable
(Allman/Ané/Rhodes 2008) but GTR+I+Γ only "likely" for the discrete implementation (Nguyen 2018; Chai & Housworth
2011). **The +I/+R confound is unstudied** — all published confound work is +I/+G.

### 6.6 🔴 LEVER B (SQUAREM) DROPPED — it is mis-targeted at a path the GPU deliverable bypasses (red-team, CONFIRMED)

I proposed SQUAREM as a "cheap first probe." The red-team killed it on source, and I verified: **SQUAREM would
accelerate `RateFree::optimizeWithEM` (the CPU EM), but the GPU +R cost is NOT there.** `modelfactory.cpp:1613-1614`
calls `optimizeParametersJOLT` and **returns before the CPU path** on success; the +R joint LM in
`gpu_lnl_intree.cu` steps rates (`g_y`) and weights (`g_z`) itself. `ratefree.cpp:325 return optimizeWithEM()` runs
only on **decline/fallback** (ineligible regime, user-fixed `+R{...}`, ncat>10) — not the 90.9%/401-cap cost. Worse,
the GPU's *own* EM weight M-step already exists and was **measured neutral and RETIRED**:
`gpu_lnl_intree.cu:3182 static constexpr bool JOLT_REM_EN = false; // RETIRED 2026-07-14 ... measured NEUTRAL (1.0×)`.

⇒ **SQUAREM-on-the-GPU-EM is a partial graveyard repeat, and SQUAREM-on-the-CPU-EM probes the wrong optimiser.**
There is only **one lever: A** (the OPG conditioned step of §4). SQUAREM survives only as *related work to cite*
(a proven accelerator never applied to +R, novelty gap #4) — not as a build path.

🔴 **AND SEQUENCE ACCELERATION HAS A THREE-TIME DEAD-END RECORD IN THIS CODEBASE (verified on disk 2026-07-19, in
response to the user's "SQUAREM was attempted" note — the memory is correct, it points here):**
1. **`TRY_AITKEN` (`utils/optimization.cpp:183` `#define TRY_AITKEN 0`).** Aitken's Δ² accelerator — of which SQUAREM is
   literally the *squared* vector generalisation — was coded into `brent_opt`, but it is **disabled**, on the wrong
   subproblem (the 1-D **Brent line search**, not the EM/mixture map), and is **printf-only**: `:257-272` computes the
   accelerated point `aitken` and prints `f(A)=...` but **never feeds it back into the search**. A diagnostic stub that
   was never wired in — i.e. sequence acceleration was *looked at* and abandoned before it did any work.
2. **`JOLT_REM` EM weight M-step** — measured NEUTRAL (1.0×), RETIRED 2026-07-14 (`gpu_lnl_intree.cu:3182`).
3. **Mode-L / Trimorph** (`mode-l-levenberg-marquardt-design.md`, `Trimorph.md`) — JOLT's predecessor, which *specified*
   the OPG/Fisher joint solve, abandoned for a broken +R weight analytic gradient (10⁵⁴) and losing on low-dim +G.

**Read together, these say: extrapolation/EM accelerators keep dying here, but the OPG *conditioned Newton step* (Lever A)
is a different animal and its decisive high-dim +R test was never actually run** (Mode-L Rec 2, unrun). That is the gap
Lever A fills — and the three corpses are why Lever A, not another accelerator, is the only build path.

---

## 7. GRAVEYARD CHECK — is this a lever that already failed?

Checked against the project's retired levers before proposing anything:

| retired lever | why it died | does this design repeat it? |
|---|---|---|
| **CTF subsample** | real-data RF **8** (avian) / **18** (euk22k), spurious `+I` | ❌ no subsampling |
| **global `tol=1e-2`** | avian `-m MFP` RF **4, 0, 4** | ❌ tolerance unchanged; this attacks the *conditioning* |
| **screen + repolish** | wrong winner + slower on avian; ranked on an under-optimised state | ❌ no screening; every candidate fully optimised |
| **MFVAL** | flips 90.8% DNA / 96.7% AA of rows to GPU-trusted; deletes the CPU refit fallback | ❌ authoritative-value path untouched |
| **model dedup** | 23.1% fewer models but wall **UP 1.0%** (`174130604`) | ❌ candidate set untouched — *and this measurement corroborates §2*: the wall is the +R crawl, not model count |
| **`-mrate G` / `--score-diff 9`** | wins by evaluating fewer models = artifact | ❌ explicitly rejected |
| **`--thread-model`** | "3.16×" was a **correctness bug** — `+R≥3` silently never dispatched in non-MPI builds | ❌ not used |
| **L-BFGS sweep (`174126911`)** | aimed at the wrong subset — `:3224` pins `lbM=0` for +R, so it could only move the 8.6% +G slice | ⚠️ **closest relative — distinction VERIFIED in source, not asserted.** `:3224` `lbM = (freeRate==1)?0:g_lbfgs_m`; the `lbS`/`lbY` ring buffers are length `nedge` (brlen only). L-BFGS could only ever move brlen; this design conditions the **(y,z) block** where the +R degeneracy lives. Genuinely different. |
| **OPG-Fisher optimiser (`JOLT-OPG-FISHER-OPTIMIZER.md`)** | **shelved unbuilt** — its target (④ 0.07-nat AA gap) was measured selection-irrelevant | ⚠️ **this IS that design, re-pointed at a LIVE target.** Not a repeat of a *failure* (it never failed — its target died). Inherited wholesale (§4.0); the new part is the re-targeting + the λ_min diagnostic. |
| **warm-start in `test()`** | NULL (0.15% wall, perturbs lnL up to 5.72 nats) | ❌ not used |
| **`JOLT_MF_DEVUSE`** | coverage-limited — guard excludes `pinv>0` and freeRate | ❌ not used |
| **SQUAREM / EM-accel (Lever B)** | GPU EM M-step `JOLT_REM_EN` measured NEUTRAL, RETIRED 2026-07-14 | 🔴 **DROPPED (§6.6)** — mis-targeted at the CPU EM the GPU bypasses |

**The governing principle, learned at the cost of four failures:**

> ✅ **Exact work-removal and exact optimiser improvements ship** (`JOLT_SCREEN_CACHE`, `MF_RESIDENT`, L5/L6,
> ns-template). 🔴 **Approximate ranking does not** (CTF, global loose tol, screen+repolish, MFVAL — 4/4 failed on
> real data). The test: *does this change which candidates are compared, or the STATE they are compared in?*

🔴 **Red-team correction — I read this law too generously the first time.** The law flags changing **the state** as a
risk *trigger*, not a safe harbour. The exact improvements that shipped byte-identical (`SCREEN_CACHE`, `MF_RESIDENT`,
L5/L6, ns-template) did **not** change the state. The one state-changing exact improvement that shipped — the high-K +R
graduation — was explicitly **NOT byte-identical** and justified only by "*the GPU fit is CORRECT and the CPU-EM was
WRONG*" (`phylotreegpu.cpp:2191`). **Lever A is in THAT category**, and its safety argument ("more converged ⇒ more
correct winner") **breaks on exactly the flat/degenerate +R candidates it targets, where there is no unique MLE.** So
**P2/P3 are genuine recall risks, not formalities.** The honest framing is not "changes neither, therefore safe" — it
is: *a state-changing optimiser improvement that ships ONLY if the new winner is verifiably the correct AND seed-stable
MLE.* That is a higher bar, and it is the right one.

⚠️ **The single largest risk is §3.1**, not the linear algebra: if the wobble is multi-modality (or the genuine-BIC-tie
third case), a better optimiser will not deliver RF=0, and the deliverable narrows from "we fixed +R" to "we can *detect*
when +R is unreliable" (§4.4) — still a real result, but a different claim. P0+P4 decide which.

---

## 8. Immediate next steps (revised post-red-team + blue-team + P0)

1. ✅ Literature (§6); ✅ red-team + blue-team landed (§7, §9); ✅ P0 ran (§9); ⏳ P4 (avian warm+tight) in flight.
2. 🔴🔴 **GATE-0.5 — THREE checks before any OPG build (red-team, 2× audited). Part 1 PASSED; parts 2–3 BLOCK.**
   - ✅ **Part 1 (weight aggregate, pure +R): PASSED (job `174152279`, first-ever run).** RGRADCHECK maxrel
     1.3e-7…1.8e-6 across AA-R4/R8/R10 + DNA-R4/R8; analytic `gz=WN−w·N` vs central-diff FD match to 6–8 sig figs;
     `sumGz=0`, `sumWN=N` (structural nulls exact). Genuine, not a false-pass. (Binary md5 `b43cb1f2` = clone
     `iqtree3-mfdevcheck`@`a07f61be`; the RGRADCHECK + `kj_derv_fused` source is **byte-identical to canonical
     `30c0faf9`**, so it transfers — re-confirm on the shipped binary before promotion.)
   - ✅ **Part 2 (RATE arm) — PASSED (job `174153578`, 2026-07-20).** New `JOLT_RGRADCHECK_RATE` (worktree
     `iqtree3-rgradrate`@`30c0faf9`, block after `:3212`) FD-checks the **REAL optimiser gradient** — it calls
     `computeGradient` then `g_y[c]=catRate[c]·gradR[c]` (`:3143`/`:3361`), not a standalone reimplementation. **maxrel
     6e-9…1.6e-7** on DNA-R4/R8 + AA-R4/R8/R10 + **real avian R6/R8**; per-category `g_y` vs FD match 6–7 sig figs,
     **verified line-by-line on disk** (ncat lines each, summary maxrel traces to a real row). 🔴🔴 **The small-`meanR`
     cats (c=0, r≈0.05–0.09) — the `1/small` `g_rscale` regime (`:772`) that blew Mode-L to 10⁵⁴ — are the CLEANEST
     (rel ~1e-9): the analytic rate gradient is SOUND, NOT Mode-L redux.** Binary `7b296034`; env-OFF byte-identical to
     canonical (Z gate: lnL+s.e. match). 1st submit `174153541` died at cmake (dropped `-DEIGEN3_INCLUDE_DIR` — module
     load ≠ `find_package(Eigen3)`; 7s/0.17SU, **no GPU burned**; fixed from canonical `CMakeCache`). **How the traps were
     dodged:** RGRADCHECK passes `nullptr` for the rate numerator (`:2916`), so
     `g_y[c]=catRate[c]·gradR[c]` (`:3143`/`:3361`) — the exact arm Mode-L died on (10⁵⁴) — has **zero FD coverage**.
     The check perturbs `y=log r` and re-evaluates through **full `evalLnL`** (echild rebuild). 🔴 **Trap dodged (the
     red-team's #1): rates feed `echild` (`len=brlen·catRate`,
     `:2860`), so the rate FD MUST do a full `rebuildEchild`+`postorderFill`+re-eval — the weight FD's partials-frozen
     shortcut is INVALID for rates and would silently FALSE-PASS.** Also: perturb the *unconstrained* `y` **without
     re-applying `gaugeFix`** (`:3168`); **cover a small-`r` AND small-`w` category** (`g_rscale=b_e/(r·w)`, `:772`, is
     the 1/small that blew Mode-L up — R8/R10 make w~1e-3); expect maxrel *worse* than 1e-6 honestly, so gate
     regime-aware.
   - 🔴 **Part 3 (per-sample decomposition) — new, OPG-specific.** RGRADCHECK validates only the **Σ_p-aggregate**
     gradient. OPG squares **per-pattern** score vectors `s_p`; a bug that redistributes score mass across patterns but
     preserves the sum passes RGRADCHECK yet corrupts every off-diagonal of `H`. Add a per-pattern spot-check.
   - ✅ **Coverage gap CLOSED (2026-07-20) — new `JOLT_RGRADCHECK_WOPT` block + coverage gate `174154357`.** Part 1
     validated only a STANDALONE weight reimpl with `pinv` hardcoded 0 (`:2916/2928`) and the pure-+R formula `gz=WN−w·N`
     — it never touched the OPTIMISER's real `gzR` (`:3150`, which the OPG SQUARES) nor the +I+R path (`wnorm=sumWN`).
     The WOPT block FD-checks the real `gzR` via `computeGradient`, perturbing `z=log(bprop)` (the optimiser's own softmax
     param, `:3209`) and re-evaluating via full `evalLnL` at `curPinv` — so +I is live. Cells: WOPT@pinv=0 (reproduces
     part 1), WOPT@pinv>0 (`GTR+I+R4`/`LG+I+R4`/avian `GTR+I+R6`), RATE@pinv>0 (`GTR+I+R4`/`LG+I+R4`), + rate-block
     regression + Z byte-identity on pure-+R AND +I+R. Committed `a6fc4d39` (worktree, local, no push).

   🔴🔴 **RED+BLUE AUDIT of WOPT (2026-07-20, both `REVISE-FIRST`, all fixes integrated before submit):**
   - ✅ **BLOCK CORRECT.** Red-team: 9/9 attacks fail with line evidence (gzR convention = ∂lnL/∂z consumed at
     `:3486/:3491`; `bprop` sums to 1 so softmax(log bprop)=bprop; `computeGradient` `devMatch` TRUE after the seed eval;
     `evalLnL` reflects perturbed `bprop` via `applyPinv`→`catProp_v` `:2847/:2877`; the +10-nat +I+R offset cancels in a
     self-consistent central diff; byte-identical when OFF). **Blue-team HAND-DERIVED `gzR[c]=WN_c−bprop[c]·sumWN=∂lnL/∂z_c`
     for +I+R and confirmed it against kernel `:480/:482`** — the +I term `p·I_p` is bprop-independent ⇒ contributes ZERO to
     the weight gradient ⇒ the formula rightly omits it and `sumWN` (not `rN`) is the exact normalizer. FD is *sensitive* to
     a `wnorm=rN` bug (~O(1e4) on avian). **⇒ the exact gradient the OPG will square is CORRECT on both arms.**
   - 🔴 **The GATE was where false-GREEN hid (fixed):** (SEV-1) `wopt` printed but never ASSERTED `optPinv` ⇒ +I+R
     coverage could green while pure-+R ran — now asserts `optPinv==expected` AND `pinv0>0` (else `sumWN≈rN`, degenerate
     +I). (SEV-3a) `1e-3` was ~4 orders too loose for the WEIGHT arm (weights don't perturb `echild`, so the FD is clean
     ~1e-7) — **but blue-team's #1: a warm-seed sits near the CPU weight optimum ⇒ some `gz≈0` ⇒ pure-relative FLAPS
     (false FAIL, worse than a false pass).** Fix = **per-category HYBRID gate `rel<1e-5 OR |gz−fd|<1e-2`** (in BOTH blocks).
     (SEV-3b) `tail -1` gated only the last of several per-run summary lines — now all-lines fail-closed. **eps stays `1e-4`
     (blue-team: near the central-diff optimum; smaller = round-off-limited, WORSE).**
   - ⭐ **Blue-team hardened part 3 from "folds in" to MANDATORY:** a total-gradient FD provably CANNOT catch per-pattern
     sign/scale errors that cancel in the sum, and the OPG squares per-pattern `s_p` — so the OPG gate MUST add a
     host-vs-device Gram cross-check on per-pattern `γ_pc`. Keep eps=1e-4; per-pattern is the OPG's own gate, not this one.
   **✅ COVERAGE GATE `174154357` PASS (binary `c2d6b63f`, 2026-07-20) — verified line-by-line on disk.** WOPT@pinv=0
   reproduces part 1 (maxrel 1.2e-8…3.2e-7); WOPT@pinv>0 real +I `gzR` on `GTR+I+R4`/`LG+I+R4`/avian `GTR+I+R6`
   (`pinv0`=0.009/0.014/**0.19**, all asserted >0 ⇒ +I genuinely engaged; maxrel 4e-9…1.7e-8); RATE@pinv>0
   (maxrel 6.8e-8/8.2e-8); rate-block regression still PASS under the new hybrid gate (1.5e-7); standalone weight
   RGRADCHECK unregressed; **Z byte-identity holds on BOTH pure-+R (`-7543892.8342`) AND +I+R (`-7543893.4986`).** Every
   cell clears the tight `1e-5` relative arm with 30–1000× margin (no near-stationary flap materialised, hybrid held as
   insurance) ⇒ the `1e-5` gate is meaningful (a 1e-4 +I bug WOULD fail).
   **⇒ GATE-0.5 FULLY CLEARED: both gradient arms (rate y, weight z) proven correct on the REAL optimiser gradients, at
   pinv=0 AND pinv>0, on DNA+AA+real-avian, byte-identical when off. The gradient the OPG will SQUARE is validated.
   Per-pattern (the only remaining check) is folded into the OPG's own gate as a MANDATORY host-vs-device Gram cross-check.**
3. Only if GATE-0.5 passes AND P4 ⇒ warm-tight-stable: prototype the OPG step in a **fresh clone**, default-OFF,
   kill-switched, inheriting the prior doc's ingredients — Marquardt `λ·diag(H)`, Nielsen gain-ratio, **`(2k−1)` reduced
   coords (softmax-only projection)**, diagonal-OPG spike before the dense solve, deterministic widened
   `kj_reduce_gradnum` Gram with **per-tile Kahan** (`nTile>1` auto-tiles for +R). Gate per §5 on DNA+AA+**real
   avian**+`-p`, fail-closed. **Bench against BOTH the 1e-7 crawl (speed) AND the 1e-2 flip (RF=0)** (§9 reframe).
4. **Fold in blue-team lever #2 (the 5× re-seed short-circuit):** `phylotesting.cpp:2649-2664` re-runs a degenerate
   +R{k} up to 5× (worst case ~5×401 iters); `λ_min≈0` is a principled gate to **skip** the re-seed (unsupported
   classes) — the diagnostic's second, concrete speed job, not just a report.
5. If P4 ⇒ M / near-tie: pivot to the **λ_min identifiability report** as the deliverable — novel regardless (§4.4).
6. No GPU-source push without user sign-off; author-run builds only.

---

## 9. AUDIT + P0 RESULTS (2026-07-19) — three agents, one live experiment

### 9.1 P0 results (job `174150315`, build-free on f3f7875f) — read WITH the red-team's confound corrections

| cell | result | reading |
|---|---|---|
| **P0b** GTR+R5 × 3 seeds (M-detector) | lnL **−59,399,837 / −59,495,862 / −59,724,692** = **spread 324,855 nats** | 🔴 too large for multimodality (distinct optima are ~tens of nats) ⇒ **catastrophic COLD-START truncation** — each seed stranded near its random init by the ρ≈0.975 crawl. **NEW confound I must own: this is COLD standalone +R5; the `-m MF` pipeline WARM-seeds +R{k} from +R{k−1} (blue-team: ladder seeding reaches the GPU). So P0b measures the cold optimiser, NOT the pipeline. The decisive cell is P4 (warm+tight).** |
| **P0a** GTR+R5 vs SYM+R5 | −59,399,837 vs −59,595,327 = 195,490 nats | 🔴 **CONFOUNDED (red-team SEV-2, confirmed):** the sim data has **unequal** freqs (0.300/0.197/0.291/0.212); bare `-m GTR`=empirical, `-m SYM`=equal ⇒ GTR ⊃ SYM, the gap is **frequency freedom + cold truncation, NOT non-convergence.** The true identical-model control is `GTR+FQ+R5` vs `SYM+R5` — a follow-up owes this. **Do NOT credit the old "19.13-nat GTR/SYM" line to non-convergence (§1.2 overclaimed).** |
| **P0c** defcap vs `-nparam 1000` | **−59,399,837 == −59,399,837** (identical) | ✅ confirms red-team SEV-3: `-nparam` is the **outer** round loop, never the inner `brlenMaxIter=400` cap — a **no-op**. The inner-cap T-test needs a build (no env lifts it); truncation is instead already established on disk via the dl-trajectory (`174127774`: dl never reaches 1e-7 at the cap). |
| **P0d** tol 1e-7 vs 1e-2 | −59,399,837.23 vs −59,399,839.31 = **2.08 nats** | loosening the kept tol costs ~2 nats on a cold +R5 fit (stops slightly earlier). |
| **P4** avian `-m MF` × 3 seeds, tight | ✅ **winner = `GTR+F+R6` / `GTR+F+R6` / `GTR+F+R6` — SEED-STABLE.** lnL −11205729.6 / −11206349.0 / −11206077.7 (spread ~619 nats on the fixed PARS tree). | ✅ **pro-T (fixable).** In the warm-seeded pipeline at tight tol the WINNER is reproducible (reproduces WORK-LOG L115), and the RF gate `174142647` separately showed the FINAL tree after search is RF=0 (finalLnL spread 0.16 nats). The residual ~619-nat spread here is a **pre-tree-search flat-valley transient** on the fixed tree — big enough to matter at loose tol, small enough at tight tol to leave winner+topology invariant. |

### 9.1a ⭐ T-vs-M VERDICT: **T (fixable) for the pipeline** — the RF=0 headline is alive

Reading the cells together, with the confounds removed:

- **P0b's 324,855-nat spread is COLD-START, not the pipeline.** Standalone `-m GTR+R5` from a random init, ρ≈0.975 crawl,
  400-cap ⇒ each seed stranded near its start. The job's own auto-interpretation line ("MULTI-MODALITY signal") read
  *this* number and is **misleading** — it measured the cold optimiser, which the `-m MF` pipeline never uses (it
  ladder-warm-seeds +R{k} from +R{k−1}).
- **P4 is the pipeline, and it is winner-stable.** Warm-seeded + tight tol ⇒ `GTR+F+R6` all three seeds; final tree RF=0
  (RF gate). ⇒ **the +R wobble that matters is NOT genuine multi-modality (M) — the winner and the topology are
  reproducible once the fit is converged.** It is the **flat-valley** case of §3.1: a connected near-degenerate valley
  where the exact fit point is seed-dependent (the residual 619 nats, and the 704-nat loose-tol spread) but the *selection
  and topology* are robust at tight convergence.
- **⇒ the design proceeds on the T branch:** OPG conditioning to reach the tight-quality point at 1e-2 speed (so 1e-2
  gives the same winner + RF=0 that 1e-7 does today), + λ_min to canonicalise the residual flat-valley wobble. The
  "narrow to a detection-only claim" fallback (§4.4) is **not** triggered — but λ_min still earns its keep on the residual
  wobble and on lever #2 (the 5× re-seed short-circuit).
- ⚠️ **Honest caveat the 619 nats forces:** "condition better and the wobble vanishes" is too strong — even tight+warm
  leaves a ~619-nat pre-search flat-valley wobble. Tree search washes it out (RF=0, 0.16-nat finalLnL), but the OPG gates
  P2/P3 must confirm the *conditioned 1e-2* reaches the same winner+tree, not merely "a" stable point.

### 9.2 🔴🔴 THE VALUE-PROPOSITION REFRAME (blue-team #1, verified) — the shipped default is 1e-7, not 1e-2

My §0 first said "keeping the tolerance at 1e-2." The shipped inner LM tol is actually `1e-7` (`gpu_lnl_intree.cu:3220`);
`1e-2` is the `JOLT_IR_TOL` override. 🔴 **CORRECTED FRAMING (user, 2026-07-19): `1e-2` is not "a retired lever" — it is
the correct DESTINATION, and `1e-7` is the proven bottleneck (3.55× on DNA, 10⁶× tighter than upstream's 0.1).** What
held the *global* flip was the avian RF 4,0,4 — but that is the **+R under-conditioning symptom**, not a fault of `1e-2`
itself: loose tol lands the degenerate +R fit seed-dependently, so the +I/pure-R early screen flips and MF walks a
disjoint model set (WORK-LOG §RF, `174142647`). ⇒ **the design's job is to make `1e-2` reproducible: OPG + λ_min reach
the same +R optimum every seed, turning RF 4,0,4 → RF=0, which is what finally lets `1e-2` ship as the default.** Honest
nuance kept: on avian `1e-7` currently finds the *best* BIC of the three tols (66 better, WORK-LOG L523) — so the target
the conditioned `1e-2` must reach is that tight optimum, not merely "any stable point." **⇒ OPG must be benchmarked BOTH
ways: as fast as the 1e-2 crawl-free speed AND landing on the 1e-7 optimum's RF=0/BIC.**

### 9.3 Agent verdicts (both verified on disk)

- **Red-team = REVISE.** SEV-1 (rate-gradient never FD-checked → GATE-0.5, §8); SEV-2 (P0a freq confound, §9.1); SEV-3
  (P0c wrong cap, §9.1); SEV-4 (the "8–30% of wall" bound is an AA figure — on the DNA target +R is **77% of wall**, so
  the design *understates* its own upside — but OPG and the tol lever are **substitutes**, not additive); SEV-5 (the
  prior doc's Kenney-Gu exact-Hessian fallback for the rank-deficient case was dropped — λ_min *detects* the degeneracy
  but `λ·diag(H)` alone does not *resolve* it to a canonical point). Confirmed the design is NOT a graveyard repeat.
- **Blue-team = BUILDABLE.** The responsibilities ARE resident (`d_wnum[c][p]=w_c·L_c(p)`, `Σ_c=L_p`; both score arms
  formable per-pattern, no new likelihood pass — verified `:478-482`, `:3148-3150`). The Gram is a **deterministic
  widening of `kj_reduce_gradnum`, ~2 KB shared — my §4.3 occupancy-cliff worry was a strawman.** Corrected the gauge to
  `(2k−1)` (§4.1). Found **lever #2** (the 5× re-seed short-circuit via λ_min, §8) and confirmed **same-walltime+RF=0 is
  NOT achievable by OPG alone** — needs OPG (conditioned convergence) + λ_min (canonical point) + the tol lever.

---

## 10. 🔴 CORRECTION TO THE PHASE-2/3 POST-MORTEM (2026-07-21, gate re-audit)

The record said, in effect, *"OPG failed for +R."* **That is too broad, and the gate evidence is weaker in two
places than it was written up. Both corrections are against the gate artifacts on disk, not against memory.**

### 10.1 What failed was the STEP POLICY, not the CURVATURE

The OPG Gram itself passed every validation it was given and **was never the thing that failed**: GATE-0.5 proved
both gradient arms (rate `y`, weight `z`) against the REAL optimiser gradients at `pinv=0` AND `pinv>0` on
DNA+AA+real-avian (`174152279`, `174153578`, `174154357`); Phase 1 then validated `H` itself by an independent
per-pattern FD vs `patlh` plus a host-vs-device Gram cross-check, with an eps-ladder confirming `|analytic−FD| ~ eps²`
(`174156911`). **Three different STEP POLICIES have now been run on that same validated Gram:**

| policy | avian GTR+R6 | job |
|---|---|---|
| diagonal OPG (Phase 2) | 401 capped, **−163 nats** | `174158208` |
| dense damped Newton (Phase 3) | 401 capped, **−51.9 nats** | `174159336` |
| **hard rank projection (HARP WS1.5, τ=1e-3)** | **229 iters, `exit=CONV(dl<tol)`, +49.7 nats** | `174323861` |

⇒ **the correct statement is "the diagonal and dense-damped step policies failed on +R", NOT "OPG failed".** The
third policy — built after this programme was wound down — produced the FIRST genuine convergence avian +R has ever
had (every prior arm, ON and OFF, terminated at the `brlenMaxIter` cap). That is not yet a promotion: WS1.5's
neighbouring `τ` values all lose lnL, which is the same non-monotone shape that killed Phase 3, and job `174328851`
is the pre-registered test of whether `τ=1e-3` is real or an n=1 artifact. But it does mean the programme was
retired on evidence about two step policies, and generalised to the curvature.

### 10.2 🔴 FINDING 1 — `dna_r8`'s Phase-3 verdict is confounded by UNEQUAL WORK

Phase 3 recorded `dna_r8` as a −10.28-nat loss and folded it into "the dense direction is wrong." The `[IRCONV]`
trace (`opgp3_174159336/on_dna_r8.console`) says otherwise:

```
it=143  rej_it=14  acc=0  conv=0  lamOut=1.000e+07  nu=32768  exit=REJECT-EXIT
```

**ON exhausted all 14 backtracks, hit the `lam` CEILING, and gave up at iteration 143 — while OFF ran the full 401.**
The −10.28 nats therefore conflates "wrong direction" with "terminated with 64% less optimiser work", and the
ceiling-latch is the *Phase-2* freeze signature, not the Phase-3 overshoot signature. `wag_r8` (−6.39) and `av_r6`
(−51.88) BOTH ran the full 401/401 like OFF, so **their** comparisons are work-matched and stand. ⇒ 1 of the 3
losing cells has a muddied verdict; the headline "direction, not length" is stated with more confidence than
`dna_r8` supports, though it survives on the other two.

### 10.3 🔴 FINDING 2 — the OFF reference was itself NOT CONVERGED

On every degenerate cell the OFF arm ran `401/401` with `conv=0`. So the Phase-2/3 gates measured
*"did ON beat a TRUNCATED baseline at equal iteration budget?"* — **not** *"did ON find a worse optimum?"* Those are
different questions and only the second justifies "the direction is wrong."

How much the reference was short by is now measured: on avian, OFF sits at `−11,216,886.23` (truncated at the cap),
while HARP WS1.5 at `τ=1e-3` reaches `−11,216,836.49` and genuinely converges — **the reference was 49.7 nats below
what is reachable on that cell.** This does not rescue Phase 3 (it lost to even the truncated reference, by 51.9),
but it does mean **the gates never asked whether the dense direction is better GIVEN convergence**, because neither
arm ever converged.

### 10.4 🟡 FINDING 3 — the 0.5-nat kill bar was never checked for SELECTION impact

This project's own precedent (the ④ 0.07-nat AA +I+R gap) was **accepted** after showing it could not flip a real
winner — margin 19×–14000× the gap. Phase 3 got no equivalent check. For avian it would not have helped (51.88 nats
≈ 104 BIC, larger than the 82-BIC avian top-2 margin ⇒ genuinely selection-relevant), but `dna_r8`'s −10.28
(≈20 BIC) may well have been selection-irrelevant. ⇒ a nat-threshold kill without a BIC-margin check is not the bar
this project holds itself to elsewhere.

### 10.5 What the gates got RIGHT (audited, not assumed — do not re-litigate these)

- **Proof-of-build**: Phase-3-unique sentinel `JOLT_OPG_DLAM0`, MEASURED `strings|grep -c`, canonical asserted to
  LACK it, and the gate aborts before burning GPU. Follows the project's own rules exactly.
- **OFF-identity (gate Z)**: byte-identical to canonical on DNA + AA + avian **and a PARTITIONED `-p` cell** —
  the partition-blind defect class was covered.
- **Spectrum regression (gate LM)**: asserts the refactor still reproduces Phase 1's `lmin=6.238907e-06` EXACTLY.
- **Metric symmetry**: `iters`/`nRej` are summed from `JOLT-DIAG-CU`, which BOTH arms emit; the `JOLT_IR_CONVTRACE`
  asymmetry (ON-only) affects wall alone and biases *against* ON ⇒ conservative, not a false kill.
- **Attribution control (`174160067`)**: `JOLT_OPG_LSAT=0` ⇒ dense step provably never fires ⇒ avian reproduces OFF
  to **+0.0001 nat**. This is the right control and it passed, so avian's −51.88 IS the accepted dense step(s) and
  NOT the Fix-A fallback. (Note it is cell-dependent: the same control reads −0.09 on dna_r8 and **+4.50** on
  wag_r8, so "the scaffolding is inert" is exact only on avian.)

### 10.6 Net effect on the verdict

**Phase 3 as designed stays dead** — three repair routes closed, and the floor's non-monotonicity means no constant
generalises. **The generalisation "OPG doesn't work for +R" is withdrawn**: the curvature was validated and never
implicated, and a third step policy on that same Gram has since converged avian. The open question is no longer
*"is the OPG curvature usable?"* but *"is there a step policy on it that is stable across cells?"* — which is
exactly what `174328851` tests, with kill criteria fixed in advance.

---

# 11. THE REAL SOLUTION (2026-07-21) — three literature tracks + code archaeology, red-teamed and verified

> Written after L1 (optimiser/convergence SOTA), L2 (mixture degeneracy in statistics), L3 (phylogenetic
> rate-category identifiability), plus my own source archaeology. **Every load-bearing number below I re-verified
> myself**; where an agent's claim failed verification I say so and give the corrected version. This section
> supersedes §4 (the OPG proposal) as the design of record.

## 11.1 THE THEOREM — three independent tracks converge, and it reframes everything

**Finite mixtures are SINGULAR statistical models.** When two components coincide the Fisher information becomes
non-invertible, and MLE convergence is **provably slower than the parametric rate** (`n^{-1/4}` vs `n^{-1/2}` for
over-specified mixtures — Chen 1995 *Ann. Statist.*; Ho & Nguyen arXiv:1609.02655; Manole & Ho 2022). L1, L2 and L3
reached this independently from the optimisation, statistics and phylogenetics literatures respectively.

Three consequences, and they are the reason §4 failed:

1. 🔴 **The OPG failure was PREDICTED BY THEORY, not an implementation bug.** We were conditioning a matrix that is
   *genuinely rank-deficient at the solution*. No amount of step repair inverts a singular matrix.
2. **No optimiser fix removes the ridge.** The ridge is a property of the *model given the data*, not of the code.
   ⇒ the fix must be **detection + model-order control**, not step repair.
3. 🔴 **BIC's penalty is misspecified at exactly the comparison we make.** Schwarz's derivation assumes a nonsingular
   FIM; mixtures violate it precisely on the `k → k−1` boundary the ladder compares across. The correct asymptotic
   penalty is governed by the real log-canonical threshold and is **strictly smaller** than `k/2`
   (Watanabe 2009/2013 *JMLR*; sBIC, Drton & Plummer 2017 *JRSS-B*). Independently, mixture *order* selection is
   non-regular — parameter on the boundary, singular FIM, non-identifiability under the null — so the LRT is not
   `χ²` either (Hartigan 1985; Chen & Chen 2001; review Celeux/Frühwirth-Schnatter/Robert arXiv:1812.09885).
   ⇒ **part of our irreproducibility is intrinsic and no optimiser will remove it.**

## 11.2 THE STRUCTURAL INSIGHT — undrawn in the literature for +R

**`+I+R_k` IS the boundary limit of `+R_{k+1}`** as one rate → 0 (an invariable class is a rate-0 class with a free
weight). Symmetrically, two collapsing `+R` categories place the fit on the boundary where `+R_k` degenerates to
`+R_{k−1}`. **So the `+I`/`+R` confound is a mixture-order non-regularity, not a numerical nuisance** — and every
symptom we measured (near-duplicate categories, a near-zero category impersonating `+I`, a linear crawl that never
converges, a `k`/`±I` choice that flips with the seed) is the textbook signature of a mixture fitted past its
supported order. L3 found **no phylogenetics paper drawing this connection for `+R`**; Nguyen, von Haeseler & Minh
2018 (*Syst Biol* 67:552), the field's definitive treatment of the neighbouring `+I+G` pathology, **does not mention
`+R` once**. Empirical corroboration in print: Jia, Lo & Ho 2014 (*PLOS ONE* 9:e95722) — marginal likelihood
**plateaus** past ~6–10 categories and pinv is *"highly susceptible to changes in the number of rate categories."*

## 11.3 🔴🔴 LAYER 1 — RESTORE THE CONVEX STRUCTURE. **The correct solve already exists in our code and was retired for the wrong reason.**

**The mathematics (I verified this myself, it is elementary):** for FIXED rates and branch lengths,
`L_p = Σ_c w_c f_c(p)` is **affine in `w`**, so `log L_p` is concave and `Σ_p freq_p log L_p` is **concave in `w`**.
The constraints — `w_c ≥ 0`, `Σ w_c = 1`, and the mean-rate gauge `Σ w_c r_c = 1` — are **all linear in `w`**. ⇒ the
weight subproblem is a **convex program**: unique optimal *value*, seed-independent, and a convex argmax set whose
max-entropy / min-norm point is unique and computable. **This is exactly where our degenerate mass-trading direction
lives.** (Classical NPMLE-in-weights structure: Lindsay 1983/1995; Koenker & Mizera 2014 *JASA*.)

🔴 **But our GPU path destroys that convexity by construction.** It parameterises weights as **softmax logits**
(`gpu_lnl_intree.cu:3209` `zR[c]=log(bprop[c])`) and steps them with a **diagonal LM**
(`:3421` `cz[c]=baseZ[c]+g_z[c]/(fabs(ddZ[c])+mu)`). A softmax reparameterisation of a concave-in-`w` problem is not
concave in `z`. **We took a convex subproblem and made it non-convex and ill-conditioned.**

🔴🔴 **AND THE FIX IS ALREADY WRITTEN, AND WAS SWITCHED OFF.** `gpu_lnl_intree.cu:3422` contains the exact EM
closed-form M-step — the convex solve — `w_c = WNc[c] / Σ_c WNc[c]`, where `WNc[c] = Σ_p freq_p·γ_pc` is the EM
posterior sum **already reduced on-device for the softmax gradient** (`:3150`), i.e. **it is FREE**. It is gated on:

```cpp
static constexpr bool JOLT_REM_EN = false;   // RETIRED 2026-07-14 (was JOLT_REM): EM closed-form weight M-step,
                                             // measured NEUTRAL (1.0x, insurance only). Env surface removed.
```

**It was judged on SPEED, found neutral, and retired — but it was NEVER judged on DETERMINISM, which is the problem
we actually have.** That is a metric error in the retirement decision, the same class as this project's other
gate-blindness failures ([[feedback_check_the_artifact_not_the_story]]). Two further defects in how it was scoped:
it was restricted to `optPinv==0` (**pure `+R` only — so `+I+R`, the worst case, never got it**), and it was
evaluated as a speed lever when its real value is a *unique, seed-independent optimum for the degenerate arm*.

⚠️ **Honest bound on what this buys.** Convexity gives a unique optimal **value** even when the argmax is a flat set;
it does **not** make the model identifiable (the flat set stays flat — that is a data property). What it delivers is
exactly what we need: **a reproducible per-candidate lnL, hence a reproducible BIC ranking**, plus a canonical
(max-entropy) parameter point instead of wherever the crawl happened to stop. The rate arm remains non-convex; this
is an *inner* solve inside the outer loop, not a global cure.

**⇒ HIGHEST-VALUE ACTION: revive `JOLT_REM`, extend it to `+I+R` (its EM normaliser is `1−pinv`), and gate it on
DETERMINISM (cross-seed lnL/winner agreement), not on wall time.**

## 11.4 LAYER 2 — FIX THE CONVERGENCE TEST PROPERLY. **Aitken (windowed), not N-consecutive.**

L1's criterion table, ranked by robustness to *our* failure mode:

| criterion | fooled by a linear ridge-crawl? |
|---|---|
| **Aitken / extrapolated-limit** | **No** — `a → 1` inflates the estimated remaining distance by `1/(1−a)` |
| **RDM** `∇Fᵀ H⁻¹ ∇F / m` (Commenges 2006; Philipps 2021) | **No** — gradient along the ridge is non-zero while crawling |
| projected-gradient norm (L-BFGS-B `pgtol`) | No, but unscaled ⇒ needs per-problem tuning |
| parameter change `Σ(Δθ)²` | **Yes** — θ crawls too |
| relative logL change | **Yes** — explicitly documented to cause premature convergence on slow progress |
| **absolute logL change ← OUR CURRENT TEST** | **Yes — the weakest criterion in the field** |
| N-consecutive-below-tol | **Partially** — defeats a single transient dip, but a *sustained* crawl still passes N in a row. L1 found **no principled citation** for it as a convergence rule |

🔴 **This corrects the proposed `JOLT_IR_CONVN` fix.** N-consecutive kills the *lottery symptom* but not the
*crawl*. It is strictly better than what we have and cheap, so keep it — but it is **not** the principled answer.

**Aitken-accelerated stopping** (Böhning et al. 1994; Lindsay 1995; McNicholas 2010 — the default in shipped mixture
packages) stops on *distance to the extrapolated limit*, not on step size:
`a⁽ᵗ⁾ = (l⁽ᵗ⁺¹⁾−l⁽ᵗ⁾)/(l⁽ᵗ⁾−l⁽ᵗ⁻¹⁾)`, `l∞ = l⁽ᵗ⁾ + (l⁽ᵗ⁺¹⁾−l⁽ᵗ⁾)/(1−a⁽ᵗ⁾)`, stop iff `l∞ − l⁽ᵗ⁾ ∈ (0,ε)`.

🔴 **CRITICAL IMPLEMENTATION FINDING — I tested this on our real traces and the naive form FAILS.** Applied to the
final consecutive pair (as the literature agent applied it), `off_av8` gives `a = 3.14e-3/0.228 = 0.0138` ⇒ remaining
`3.18e-3` ⇒ **it PASSES — the same lottery.** Our `dl` sequence *oscillates*, violating Aitken's linear-convergence
assumption pointwise. The **windowed** form (ratio of mean `dl` over consecutive 25-iteration blocks) is correct and
rejects both:

| cell | naive Aitken | **windowed Aitken (w=25)** |
|---|---|---|
| `off_av8` `GTR+R8` (currently flagged `conv=1`@342) | `a=0.0138`, remaining `3.18e-3` ⇒ **PASS (wrong)** | `a=0.879`, remaining **2.83 nats** ⇒ **REJECT** ✅ |
| `off_av6` `GTR+R6` (capped@400) | `a=33.4` ⇒ reject | `a=1.086` (**increments growing**), remaining **∞** ⇒ **REJECT** ✅ |

⇒ **implement Aitken over a window, never per-step.** Note `off_av6`'s windowed `a > 1`: it is not even linearly
converging, it is wandering — a fact the current test cannot express.

**Also restore what we lost:** upstream IQ-TREE uses parameter-change (`1e-4`, EM) and a gradient tolerance
(`TOL_FREE_RATE=1e-4`, BFGS). **Our GPU path uses neither** — disabling L-BFGS for `+R` (`:3224`) removed the only
gradient-based test and left the weakest one. Every reference implementation (L-BFGS-B, NLopt, marqLevAlg) stops on
a **composite AND-gate**. Adopt: gradient **and** parameter **and** windowed-Aitken.

## 11.5 LAYER 3 — STRUCTURAL GUARDS. We are far looser than the field, and it is nearly free.

RAxML-NG bounds rates and weights away from zero — `PLLMOD_OPT_MIN_RATE = 0.02`, `MIN_RATE_WEIGHT = 1.0e-3`
(`pll-modules/src/optimize/pll_optimize.h`). Ours are `1e-4` for both (`gpu_lnl_intree.cu:3419`, `:3423`) —
**200× looser on rate, 10× looser on weight.** Hard floors well away from zero make it structurally harder for two
categories to become numerically duplicate; this is a cheap, cited, partial defence against the degeneracy at source.

## 11.6 LAYER 4 — MODEL-ORDER CONTROL AND HONEST SELECTION (where the novelty is)

Because the ridge is a *model* property, the principled response is to **not fit past the supported order**:

- **Identifiability-gated ladder.** Stop climbing `k` when the fit stops being identifiable, instead of climbing
  blind to `kmax=10`. L2 confirms our `λ_min` idea is the established notion of **local identifiability**
  (Rothenberg 1971 *Econometrica*: local identifiability ⟺ nonsingular information matrix) operationalised as
  **parameter redundancy / near-redundancy** (Catchpole & Morgan 1997 *Biometrika*). ⚠️ **`λ_min` alone is NOT
  scale-invariant** — a bare threshold is meaningless. We already report `λ_min/λ_max` (reciprocal condition number),
  which is the defensible form; **keep it that way and never quote a bare `λ_min` threshold.**
- **Report cap-exhaustion.** The field treats a cap as a *safety net* and cap-exhaustion as a **reportable failure**;
  ours is unusual in being *load-bearing* (everyone else's caps are far smaller — RAxML-NG 32, PhyML 100 — precisely
  because they are paired with criteria that actually fire).
- **Report ICL alongside BIC** (Biernacki, Celeux & Govaert 2000). Two near-duplicate categories give every site
  ~50/50 responsibility ⇒ maximal classification entropy ⇒ ICL penalises **exactly** our failure mode, from
  responsibilities the optimiser already computes. ⚠️ Caveat: ICL's rationale is *clustering*; `+R` is a
  rate-heterogeneity model, and ICL is known to under-select when components genuinely overlap — which for rate
  heterogeneity may be physically correct. **Use as a stability diagnostic, not as a replacement selector.**
- **Merge near-duplicates** as the canonical representative of the flat set (entropy merging: Baudry, Raftery,
  Celeux, Lo & Gottardo 2010 *JCGS*; Hennig 2010) — cheapest deterministic resolution, uses responsibilities we have.
- **Name the limit honestly.** BIC is misspecified here and sBIC's learning coefficients are **unknown for
  phylogenetic rate mixtures**, so we cannot repair the criterion — but we *can* flag when a comparison sits on the
  degenerate boundary. That flag is the honest deliverable.

## 11.7 WHAT I EXPLICITLY REJECT (red-team, with reasons)

- ❌ **Reparameterisation as a CURE.** The FIM transforms as `JᵀIJ`; rank is preserved under any smooth invertible
  reparameterisation. Ordering/gap coordinates **relocate** the ridge to a detectable boundary (`Δ_c → 0`) — a real
  *diagnostic* gain — but do **not** condition it. Do not sell it as a fix.
- ❌ **Another dense-Newton / OPG variant.** Theory (11.1) says the matrix is genuinely rank-deficient at the solution.
- ❌ **Natural gradient.** It preconditions with the Fisher information — the very matrix that is singular here.
- ❌ **Relative tolerance as the primary fix.** Documented to cause premature convergence on slow progress; it would
  reproduce our bug in new units.
- ❌ **Penalised likelihood as a default.** It changes the objective ⇒ changes selection. Only defensible in the cited
  *penalise-to-select-then-refit-unpenalised* pattern (Chen & Khalili 2009), and their SCAD form is **unverified at
  source** (the agent could not open the PDF).
- ⚠️ **Truncated-Newton with CG** (Nash) is the one optimiser idea that survives — matrix-free, declines to step along
  near-null curvature instead of inverting it — but it is a rewrite of the `+R` inner loop, so it is a *later* phase,
  not the first move.

## 11.8 NOVELTY — what is actually unclaimed (L1 + L3 agree)

1. **No acknowledgement anywhere in phylogenetics that `+R` model selection is non-reproducible due to optimiser
   non-convergence.** L1: *"a genuine gap in the literature and may be publishable in its own right."* The only
   corroboration is informal — IQ-TREE GitHub Issue #38 ("+R reaches a hard limit of 99 iterations without having
   converged… parameters slowly migrating in one direction"), **our exact failure mode**, closed without resolution.
2. **No use of Aitken or RDM stopping in phylogenetic software** — both standard in statistical mixture work; the
   transfer has not happened.
3. **No fast, deterministic identifiability gate on an ML rate-category ladder.** The Bayesian-nonparametric line
   (Huelsenbeck & Suchard 2007; Gill, Baele, Suchard & Lemey 2025 *MBE*) solves it far more expensively and trades
   seed-dependence for chain-convergence dependence. ModelFinder's own docs admit the `kmax=10` endpoint is
   unprincipled — the official remedy is *"re-run manually with `-cmax 20`."*
4. **The `+I+R_k` = boundary-limit-of-`+R_{k+1}` framing** (11.2) is undrawn for `+R`.
5. ⭐ **By PAML's own published standard** — convergence is checked by running from multiple starts and comparing —
   **our three-seeds-three-winners result is already a methodology-standard diagnosis of non-convergence.**

## 11.9 PHASED PLAN

| phase | action | why first | gate |
|---|---|---|---|
| **A** | Revive `JOLT_REM` (EM weight M-step) + extend to `+I+R`; gate on **determinism**, not speed | the correct solve exists, is free, and was retired on the wrong metric | cross-seed lnL + winner agreement on avian; OFF byte-identical |
| **B** | **Windowed** Aitken + composite AND-gate (gradient ∧ parameter ∧ Aitken); keep N-consecutive as a cheap floor | ~20 lines host-side, zero GPU impact; verified to reject both our cases | `off_av6`/`off_av8` must both report NOT-converged |
| **C** | Tighten rate/weight floors toward RAxML-NG's `0.02` / `1e-3` | nearly free, cited, attacks the degeneracy at source | selection-invariance on sim + real |
| **D** | Cap-exhaustion flag + `λ_min/λ_max` + ICL reported per candidate | honesty layer; ship-regardless | reporting only, numbers unchanged |
| **E** | Identifiability-gated ladder (stop climbing `k` when unsupported) | the novelty; needs A–D to stand on | winner + RF vs the tight reference, ≥3 seeds, real data |
| **F** | *(later, optional)* Truncated-Newton/CG inner loop | only surviving optimiser idea; a rewrite | full P0–P8 |

🔴 **Gate rule carried forward:** Phase A's failure mode is that it changes a winner. Any moved winner must be proven
to be the *correct and seed-stable* MLE, not merely "more converged" — the recall risk of §7.

---

## 12. 🔴🔴 THE STOCK-UPSTREAM REFERENCE (job `174338165`, 2026-07-22) — §11 SURVIVES, BUT ONE OF OUR CRITERIA DIES

Until now every claim about what the "right" avian model is rested on comparing **our** arms against **each other**.
§11 was written without a genuine external reference. That reference now exists.

**Setup.** Stock upstream IQ-TREE v3.1.2 (`4e91dd61`), binary `4c355d8b`, built with no GPU path at all —
proof-of-control asserted in-job: `JOLT` strings **0**, `--gpu/--jolt/--no-jolt` seams **0**, `--mf-epsilon` present
**1**, log line `Kernel: AVX+FMA - 104 threads`. Whole normalsr node (104 cores / 500 GB) per cell, 497 SU each.
Alignment `avian_1000000.phy` md5 `24ed6e1e` — **byte-identical to the file our tolerance gates used** — seeds {1,2,3}.

### 12.1 Result

| arm | flags | seed 1 | seed 2 | seed 3 | MF wall (s) |
|---|---|---|---|---|---|
| **A** | upstream defaults (`--mf-epsilon` 0.1) | `GTR+F+I+R6` | `GTR+F+I+R6` | `GTR+F+I+R6` | 2328 / 2280 / 2611 |
| **C** | `-ninit 2 -optalg 2-BFGS` | `GTR+F+I+R6` | `GTR+F+I+R7` | `GTR+F+I+R4` | 2226 / 2396 / 2093 |
| **B** | defaults, `--mf-epsilon` 1e-2 | `GTR+F+I+R5` | `GTR+F+I+R5` | `GTR+F+I+R5` | 3434 / 3346 / 3550 |

**9 of 9 arms select `+I`.** In arm A seed 1 the BIC-sorted table holds **118 fitted rows, 94 of them `+I`**; the best
non-`+I` fit is `JC+R8` at −11,630,557 — **424,000 nats worse** than the winner −11,206,660. On this dataset `+I` is
not a marginal label: it wins by a landslide.

### 12.2 🔴🔴 The consequence: our own gate's reference was the bug

Our `avirtolrf_174142647` ran the **same alignment**, the **same seeds**, and — verified from its `Command:` line —
**the same `-ninit 2 -optalg 2-BFGS` flags as stock arm C**:

- `JOLT_IR_TOL=1e-2` → `+I+R6`, `+I+R5`, `+I+R5` — **`+I` 3/3, agrees with upstream**
- `JOLT_IR_TOL=1e-7` (**our shipped default**) → `GTR+F+R6` ×3 — **`+I` 0/3, the only configuration, ours or
  upstream's, that drops it**

That gate recorded *"spurious `+I` 3/3"* as a failure of the `1e-2` arm, using **our own `1e-7` arm as the reference**.
Against the real reference the `+I` is correct and **our tight default is the outlier.**
⇒ **"spurious +I" is retired as a criterion.** This is a [[feedback_gate_must_prove_its_control]] failure of the
purest kind: the control was never validated as a control, it was simply the arm we already believed.

**What this does *not* do** (stated explicitly to prevent the opposite overclaim):
- **Global `1e-2` stays dead.** Its second, *independent* failure — the tree moves in 2/3 seeds (RF 4, 0, 4) — is
  completely untouched by this result. Two criteria killed it; one has now died; the other still holds.
- **`1e-2` is not vindicated.** Its `k` wobbles (R6/R5/R5) where stock at true defaults is stable `R6`×3. Only its
  seed-1 answer matches the reference exactly.
- The reference answer for avian is **`GTR+F+I+R6`**. Ours: `1e-7` gets `k` right and `+I` wrong; `1e-2` gets `+I`
  right and `k` wobbly. **Neither of our configurations reproduces the reference across all three seeds.**

### 12.3 ⭐ Under-optimisation INFLATES the selected model order — measured, in stock, on its own knob

Arm A vs arm B is the same binary, same seeds, same data, differing **only** in `--mf-epsilon`: `0.1` (looser) → `R6`
×3; `1e-2` (tighter) → `R5` ×3, consistently, at ~1.5× the wall. **Tighter convergence lets the smaller model realise
its fit and win.**

This is the single most useful mechanistic result in this document, because it is measured *upstream*, with no GPU and
no JOLT anywhere in the picture, and it predicts our own pathology: **a capped/truncated `+R` under-optimises by
construction, so it should bias selection toward higher `k`.** It is direct empirical support for §11.2 — `+I+R_k` as
the boundary limit of `+R_{k+1}` — and it means the `brlenMaxIter=400` cap is not merely wasted wall: it is a
**systematic upward bias on model order**.

It also sharpens §11.6: reporting cap-exhaustion is not cosmetic honesty. A truncated candidate is not just "less
converged", it is **biased in a known direction**, and BIC currently ranks it as an equal of a converged one.

### 12.4 ⭐ Our harness has been contributing instability we attributed to JOLT

Arm C (`-ninit 2 -optalg 2-BFGS`) flips **R6 / R7 / R4** across seeds while arm A (same seeds, bare defaults) is
stable **R6 / R6 / R6**. Those flags destabilise **stock itself**, with no GPU involved. Our selection gates have been
passing them routinely.
⇒ **every future selection gate must carry a bare-defaults arm**, or it cannot separate "JOLT is unstable" from
"the flags we chose are unstable". Some fraction of the instability this whole workstream has been chasing is ours.

### 12.5 CPU baseline for avian (use these, never published numbers)

Stock MF wall on a **full 104-core node**: arm A 2328 / 2280 / 2611 s · arm C 2226 / 2396 / 2093 s · arm B 3434 /
3346 / 3550 s. Per [[feedback_check_the_artifact_not_the_story]], these measured same-hardware numbers supersede any
published table for ours-vs-CPU comparison on this dataset.

### 12.6 Net effect on §11

| §11 claim | status after the reference run |
|---|---|
| mixtures are singular ⇒ OPG's failure was predicted | **unchanged** (theory, not data) |
| `+I+R_k` is the boundary limit of `+R_{k+1}` | **strengthened** — §12.3 measures the predicted order-inflation upstream |
| restore convexity (Phase A) targets *reproducibility*, not identifiability | **unchanged**, and now clearly the right target: upstream at defaults is *stable*, so reproducibility is achievable |
| windowed Aitken, not N-consecutive | **unchanged** |
| tighten floors toward RAxML-NG | **unchanged** |
| report cap-exhaustion + λ_min | **upgraded from honesty to necessity** — truncation is a *directional* bias (§12.3) |
| "1e-7 is correct because it is `+I`-clean" | 🔴 **DEAD.** `+I`-clean was never a virtue. `1e-7`'s only remaining defence is seed-stability, and §1.3d already showed that stability is deterministic cap-truncation, not convergence. |
