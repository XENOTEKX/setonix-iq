# PART X — The subsample-sufficiency hypothesis (the statistical foundation of CTF ModelFinder)

**Author:** as1708 / Claude (Sonnet 4.6 sweep write-up; Opus 4.8 verification + correction 2026-06-13).
**Status: SPLIT VERDICT — the hypothesis holds under _native_ subsample BIC (n=23, validated), but the CTF pipeline's
_projected_ BIC gate FAILS recall on `-m MF` and caused both 1M benchmarks to time out (jobs 170728179/170728182).
The §X.3.2 projection-amplification risk is now EMPIRICALLY CONFIRMED, not absent — see §X.5.5. Fix landed (native-BIC
eligible gate); re-run in flight.**

> **⚠️ Correction note (Opus 4.8, 2026-06-13):** an independent re-verification of the Sonnet write-up found that
> (a) the §X.5.1 **native-BIC numbers are correct** (LG+G4 #1 at all 23 runs, ΔBIC values reproduce), but (b) the
> sweep job's *embedded analysis printed empty tables* (a regex bug, now fixed — same bug was in `analyze_subsuff.py`),
> so the recall claim rested on native BIC only; and (c) the §X.5.2/§X.5.4/§X.8 conclusion that "the §X.3.2 +R risk did
> NOT trigger" is **wrong for the actual CTF pipeline** — it didn't trigger in the sweep because the sweep ranked by
> native BIC and used the CPU reference binary, but the real CTF uses the *projected* BIC on the GPU binary, where it
> DOES trigger (§X.5.5). The corrected verdict is in §X.8.

This document states the statistical assumption the coarse-to-fine (CTF) ModelFinder rests on, derives what theory
predicts, says where it breaks, and reports the experiment (§X.5, job 170727768) plus the CTF-pipeline confirmation
of the failure mode (§X.5.5, jobs 170728179/170728182).

---

## X.0 The hypothesis in one paragraph

> **Beyond a saturation alignment length L\* — hypothesised to be on the order of a few thousand site patterns,
> far below 1 M — the _identity_ of the best substitution model (the BIC/AIC/AICc winner) stops changing. Adding
> more sites tightens the continuous parameter MLEs (branch lengths, α, π, pinv) but does not move the discrete
> model-selection decision. Therefore ranking candidate models on a ~5000-site random column subsample recovers
> the same winner the full data would select, and the full alignment is needed only to _refine the parameters of
> the already-chosen model_, not to choose it.**

If true, this is the scientific licence for CTF: rank-all-on-subsample → refine top-k on full data. If false (or
if L\* ≫ 5000 for the data in question), CTF can mis-rank, and the speed win is bought with a correctness risk.
**We currently have n = 1 of supporting evidence (the AA-1M run); that is an anecdote, not a validation.**

---

## X.1 Why this is the load-bearing assumption (not a nice-to-have)

CTF does two things, and **both** embed this hypothesis:

1. **The coarse rank** runs the full `-m MF` candidate set on the subsample and ranks by BIC. This only recovers
   the right ordering if model identity has saturated by the subsample size.
2. **The scale-consistent BIC projection** (`run_ctf_*` rerank: `p=(bic+2·logl)/ln(m)` then
   `BIC_full = −2·(N/m)·logl + p·ln(N)`) **extrapolates the per-site log-likelihood measured on `m` subsample
   sites to the full `N`**. That projection is _algebraically_ the statement "the per-site log-likelihood — and
   hence the per-site lnL _difference_ between any two models — is the same on the subsample as on the full data."
   That is the subsample-sufficiency hypothesis written as a line of Python.

So the hypothesis is not a peripheral optimisation justification — it is the definition of correctness for the
coarse stage. The top-k refine is the safety net (§X.4), but the net only has to catch what the coarse stage drops,
and how often it drops the true winner is exactly the quantity this hypothesis is about.

---

## X.2 Two hypotheses, not one — and CTF only needs the weak one

It is essential to separate these, because they have very different truth conditions:

- **H1 (strong — winner saturation):** `argmin_M BIC(M | subsample) == argmin_M BIC(M | full)` for subsample size
  ≥ L\*. The subsample picks the _exact_ winner.
- **H2 (weak — top-k recall, what CTF actually requires):** `argmin_M BIC(M | full) ∈ top-k(BIC(M | subsample))`
  for k = 3. The subsample's top-k _contains_ the true winner; the full-data refine then breaks the tie correctly.

CTF is correct under **H2 alone**. H1 can fail (two near-tied models swap rank-1 and rank-2 between subsample and
full) while H2 still holds (both are in the top-3, the full refine picks the right one). The empirical target is
therefore **recall of the full-winner into the subsample top-k**, as a function of subsample length — not exact
winner agreement. Reporting only H1 would understate CTF; reporting only "it worked once" overstates it.

---

## X.3 What the theory predicts (first-principles, BIC scaling)

Write the per-site log-likelihood difference between candidates A and B at their MLEs, and let
`δ = E[ ℓ_A(site) − ℓ_B(site) ]` be its expectation under the true data-generating process (the per-site
discriminability, a property of the _data and the model pair_, independent of N). Sites are treated as independent
given the tree and the among-site rate model — the standard phylogenetic-likelihood assumption that also licenses
a random column subsample as an unbiased estimate of per-site quantities.

By the law of large numbers, `lnL̂_A − lnL̂_B = N·δ + O_p(√N)`, so

```
ΔBIC(A,B) = BIC_A − BIC_B = −2N·δ  +  Δp·ln(N)  +  O_p(√N)        (Δp = p_A − p_B)
                              └ signal ┘   └ penalty ┘  └ sampling noise ┘
```

Three regimes fall out, and they map cleanly onto the cases that actually arise:

- **Well-separated models (δ ≠ 0, e.g. two different exchangeability matrices — LG vs WAG vs a DNA Q):** the
  `−2Nδ` term dominates linearly. The decision is locked in once the signal beats the sampling noise. The
  signal-to-noise of the _sign_ of ΔBIC is `√N · δ/σ_δ` (σ_δ = sd of the per-site difference), so to be confident
  at z standard deviations you need

  ```
  N  ≳  L*  =  z² · (σ_δ / δ)²
  ```

  For genuinely different models the per-site effect size δ/σ_δ is O(1), so **L\* is tiny — tens to hundreds of
  sites.** A subsample of 5000 is wildly sufficient to exclude the wrong matrix family. This is the easy, dominant
  case and it is why CTF works at all.

- **Nested, extra parameter truly zero (δ = 0 — e.g. +I when the data's invariant signal is absorbed by +G, so
  the recovered pinv → 0):** the signal term vanishes; `ΔBIC ≈ Δp·ln(N) − O_p(1)` (the lnL gain of the larger model
  is a _bounded_ ½χ²-type boundary fluctuation, not growing with N). The penalty wins for large N ⇒ **BIC
  consistently selects the smaller, more parsimonious model**, and it does so from small N (once `ln N` exceeds the
  bounded fluctuation, i.e. again small L\*). _This is exactly the AA-1M case_: data simulated under LG+I+G4, but
  the winner is LG+G4 because +G absorbs the invariant sites, pinv → 1e-6, and BIC penalises the dead +I parameter.
  The theory predicts this winner is stable from very modest length.

- **The dangerous regime (δ > 0 but tiny — two models with a small, real per-site difference):** here `L* = z²(σ_δ/δ)²`
  is _large_, possibly ≫ 5000. The winner can genuinely flip somewhere between the subsample size and N. **This is
  the only regime where CTF is at risk** — and even here it only fails if the flip pushes the true full-winner
  _out of the subsample top-k_, not merely from rank 1 to rank 2. Note that in this regime ΔBIC is by definition
  small, so the _likelihood cost_ of guessing wrong is also small; the harm is to model _identity_, not to fit.

**Net theoretical prediction:** L\* is small for the cases that matter most (wrong matrix family, dead nested
parameters), so H2 should hold comfortably at 5000 sites for typical data — **but L\* is unbounded for near-tied
competitors, so the hypothesis is data-dependent and cannot be assumed; it must be measured per regime.**

### X.3.1 The +R overfitting twist (why the sweep must include `-m MF`, not just `-m TEST`)
FreeRate (+Rk) models carry `2k−2` free parameters. On a _small_ subsample they have the most room to fit sampling
noise, which can **inflate their subsample lnL and rank them artificially high** — only for them to fall back at
full N where the noise averages out and the BIC penalty bites. This is the classic small-sample overfitting failure
of high-dimensional models, and it is the most plausible way the coarse rank could surface a spurious +R candidate
into (or a true winner out of) the top-k. The `-m TEST` set (no +R) will look cleaner than reality; **the honest
test of our actual `-m MF` pipeline must include the +R models.**

### X.3.2 The projection amplifies the bias — a sharpened, actionable finding (from §X.7's Q4)
The literature pass (§X.7, Q4) flagged that a subsample's BIC penalty `p·ln(m)` is ~38 % weaker than `p·ln(N)`
(ln 5000 ≈ 8.5 vs ln 946k ≈ 13.8), so the _native subsample ranking_ under-penalises rich models. Following this
into **CTF's actual scale-consistent rerank** — `BIC_full = −2·(N/m)·logl_sub + p·ln(N)` — makes it worse, not
better, and this is the most important nuance in this document:

- An over-parameterised model overfits the subsample by an lnL excess that is **sample-size-independent** (the
  AIC-bias / Wilks term: the expected spurious gain from fitting Δp extra parameters to noise is ≈ Δp nats,
  whether m = 5000 or N = 10⁶).
- The projection multiplies `logl_sub` by `N/m ≈ 189`. It therefore **amplifies that fixed overfit excess by ~189×**,
  over-crediting the model's projected lnL by `≈ (N/m)·Δp` nats — i.e. `≈ 2·(N/m)·Δp` on the −2lnL/BIC scale —
  while the penalty term `p·ln(N)` is _unchanged_ and cannot compensate. For a +R model with Δp ≈ 5 extra params
  this is a **spurious BIC improvement of order ~10³ nats** (heuristic, AIC-bias magnitude; the sweep §X.5 measures
  the real value). The projection is correct for a _well-specified_ model (its per-site lnL truly scales with N) and
  biased only on the _overfit component_ — but that component lands precisely on the +R/+Rk models we most need to
  rank honestly.
- **On the AA-1M data this is harmless** — the 17,618-nat cliff above the LG family swamps a ~10³-nat spurious boost.
  **On data where +R is genuinely competitive (real rate-heterogeneous alignments), it is a real failure mode.**
- **Actionable design consequence:** for the _top-k gate_, ranking by the **native subsample BIC** (`−2·logl_sub +
  p·ln(m)`) is _safer_ for recall of a low-parameter true winner than the projected BIC, because the projection
  differentially over-credits high-p competitors and can push a low-p true winner out of the top-k. The projection
  should arguably be reserved for _reporting_ comparable numbers, not for the selection gate. **The sweep computes
  recall under _both_ rankings (native vs projected) so we can see which the data prefers** — do not assume the
  projection is the right gate just because it makes the scales match.

---

## X.4 Where it breaks — the honest failure modes

1. **Near-tied non-nested models (δ small, L\* large):** rank instability across subsample resamples; H2 can fail if
   the true winner sits at rank 4–6 on the subsample. _Mitigation in CTF:_ larger k, or a wider subsample. _Residual
   risk:_ real.
2. **Boundary / non-regular parameters (+I at pinv→0, +R rate or weight collapse, α→∞):** the χ²/BIC asymptotics
   assume interior, identifiable parameters. On the boundary the lnL-gain distribution is non-standard (Chernoff
   mixtures), the effective Δp is ambiguous, and small-sample behaviour is erratic. The +I/+R boundary is precisely
   where our own JOLT optimiser was shown to need the 4-start robustness mechanism (PART VIII §VIII.4) — the same
   boundary that makes the _selection_ unstable makes the _estimation_ unstable.
3. **Site non-independence the model does not capture** (true partition structure, heterotachy, covarion): a random
   column subsample is then _not_ an unbiased sample of the per-site discriminability, and the `(N/m)` projection is
   biased. Single-partition +G/+R data (our benchmark) is safe by construction; real biological alignments with
   unmodelled heterogeneity are not guaranteed safe.
4. **n = 1 evidence.** Everything we have "shown" is one alignment, one true model, one subsample seed. That is an
   existence proof that CTF _can_ recover the winner, not that it _reliably_ does.

---

## X.5 The experiment — COMPLETED (job 170727768, 2026-06-13)

**23 runs: -m TEST across L ∈ {1K,2K,5K,10K,20K,50K,100K} × 3 seeds + -m MF at L=5K,20K seed=1. Full-node SPR node (104T), 1h56m wall.**

### X.5.1 -m TEST results (21 runs)

| L | seed | winner | runner-up | ΔBIC | rank(LG+G4) | top3✓ |
|---:|---:|---|---|---:|---:|:---:|
| 1000 | 1 | LG+G4 | LG+I+G4 | 6.6 | 1 | ✓ |
| 1000 | 2 | LG+G4 | LG+I+G4 | 7.4 | 1 | ✓ |
| 1000 | 3 | LG+G4 | LG+I+G4 | 7.1 | 1 | ✓ |
| 2000 | 1–3 | LG+G4 | LG+I+G4 | 7.8–8.1 | 1 | ✓ |
| 5000 | 1–3 | LG+G4 | LG+I+G4 | 8.3–9.1 | 1 | ✓ |
| 10000 | 1–3 | LG+G4 | LG+I+G4 | 7.7–10.3 | 1 | ✓ |
| 20000 | 1–3 | LG+G4 | LG+I+G4 | 11.0–11.4 | 1 | ✓ |
| 50000 | 1–3 | LG+G4 | LG+I+G4 | 10.1–12.7 | 1 | ✓ |
| 100000 | 1–3 | LG+G4 | LG+I+G4 | 11.7–14.4 | 1 | ✓ |

**Summary by L:**

| L | recall(top3) | exact(rank1) | mean ΔBIC | ln(L) | ΔBIC/ln(L) |
|---:|---:|---:|---:|---:|---:|
| 1000 | **1.00** | **1.00** | 7.0 | 6.91 | 1.013 |
| 2000 | **1.00** | **1.00** | 8.0 | 7.60 | 1.053 |
| 5000 | **1.00** | **1.00** | 8.8 | 8.52 | 1.033 |
| 10000 | **1.00** | **1.00** | 9.4 | 9.21 | 1.021 |
| 20000 | **1.00** | **1.00** | 11.3 | 9.90 | 1.141 |
| 50000 | **1.00** | **1.00** | 11.7 | 10.82 | 1.081 |
| 100000 | **1.00** | **1.00** | 13.2 | 11.51 | 1.147 |

**➤ H2 LICENSED at L = 1000 (recall = 1.00, exact = 1.00 across all 3 resamples at every tested length).**

### X.5.2 -m MF results (+R overfitting probe, 2 runs)

| L | seed | winner | runner-up | ΔBIC | top3✓ |
|---:|---:|---|---|---:|:---:|
| 5000 | 1 | LG+G4 | LG+I+G4 | 8.3 | ✓ |
| 20000 | 1 | LG+G4 | LG+I+G4 | 11.4 | ✓ |

**No +R model appeared in the runner-up or top-3 at either length.** The §X.3.2 overfitting risk (projection amplifying +R overfit by ~189×) did NOT trigger on this dataset. Runner-up is consistently LG+I+G4 — the dead +I parameter, not a +R model. H2 licensed at L=5000 for -m MF (only 2 runs, single seed; limited coverage).

### X.5.3 Mechanism verification — ΔBIC grows as ln(L), not linearly

§X.3 predicts two regimes: (a) linear in L for well-separated δ≠0 models, (b) ~Δp·ln(L) for dead nested parameters (δ=0). The measured ΔBIC/ln(L) ratios range **0.95–1.25**, clustering near 1.0 across all lengths — consistent with ΔBIC ≈ Δp·ln(L) with Δp=1 (LG+G4 vs LG+I+G4 differ by exactly 1 parameter — p_invar). This confirms **the penalty-dominated regime**.

The mechanism checked precisely against raw BIC values: ΔBIC = ln(L) − 2·δ_lnL, where δ_lnL = lnL(LG+I+G4) − lnL(LG+G4). In several runs, δ_lnL is **negative** (LG+G4 achieves *higher* lnL than the richer LG+I+G4 model). This is consistent with §X.3's boundary-parameter prediction: when p_invar→0, the LG+I+G4 optimizer gets trapped at the boundary with a zero-gradient direction and cannot improve — it reports slightly lower lnL than LG+G4 despite having an extra free parameter. BIC then penalises LG+I+G4 by both the ln(L) penalty *and* the lnL loss, giving ΔBIC > ln(L) in those runs. This is not a flaw in BIC — it is BIC correctly identifying a dead parameter that costs the optimizer more than it contributes.

**This is exactly what §X.3 predicted for the AA-1M case**, and the data confirms the mechanism quantitatively. The mean ΔBIC grows from 7.0 at L=1000 to 13.2 at L=100000, tracking ln(L) closely (ln(1000)=6.9, ln(100000)=11.5).

### X.5.4 Honest caveats (do not overstate)

1. **One dataset, one true model, one tree.** The data was simulated under LG+I+G4 with near-zero pinv — a case chosen partly because the theory predicts it should be easy. The 17,618-nat LG-family cliff means the between-family winner is trivially stable; all the action is in the within-family LG+G4 vs LG+I+G4 near-tie.
2. **MF probe is underpowered** (2 runs, 1 seed). The §X.3.2 +R amplification risk is not ruled out — it's just absent on this alignment. A genuinely rate-heterogeneous alignment with competitive +R models is needed.
3. **L\* ≤ 1000 here, but L\* is data-dependent.** On near-tied non-nested models with small true δ, L\* could be ≫ 5000. The sweep confirms our specific 5000-site CTF choice is safe on this data; it does not prove it is safe in general.
4. **Native vs projected BIC — CORRECTED (Opus 4.8):** the sweep tested only NATIVE subsample BIC, and only on the
   CPU reference binary. It did NOT test the projected BIC that the CTF pipeline actually uses, and the CPU binary's
   +R subsample fits differ from the GPU binary's. So §X.5.2's "the projection bias did not trigger" is an artifact
   of what was measured — on the real pipeline it DOES trigger and broke both benchmarks (§X.5.5).

---

## X.5.5 The projection failure, CONFIRMED on the real CTF pipeline (jobs 170728179/170728182, 2026-06-13)

The two AA-1M `--jolt -m MF` CTF benchmarks (H200, A100) **both hit the 2 h walltime**, and the root cause is exactly
the §X.3.2 projection-amplification bug — which the §X.5 sweep could not see. On the *actual* GPU `coarse.iqtree`
(L=5000), the native vs projected rankings diverge sharply:

| candidate | logL_sub | native BIC | projected BIC | native rank | projected rank |
|---|---:|---:|---:|---:|---:|
| **LG+G4** (true full winner) | −390187.678 | **782061.8** ① | 147718258.9 | **1** | **4** |
| LG+I+G4 | −390187.381 | 782069.7 ② | 147718160.2 ① | 2 | **1** |
| LG+R5 | −390187.215 | 782120.5 | 147718179.9 ② | 5 | **2** |
| LG+I+R5 | −390187.235 | 782129.0 | 147718201.3 ③ | 6 | **3** |

**Native BIC ranks LG+G4 #1; the projection drops it to #4 and promotes LG+R5, LG+I+R5 into the top-3.** The mechanism
is precisely §X.3.2: the candidates' subsample **fits are within ~0.5 nat of each other** (LG+R5 beats LG+G4 by 0.46
nat of pure overfit on 5000 sites), and the projection's `2·N/m ≈ 378×` multiplier turns that 0.46 nat into ~175 nat
of spurious BIC improvement — far above the ~96 nat penalty for R5's extra parameters. CTF then tried to refine the
promoted +R models on full 1M; +R **declines JOLT** (non-mean-gamma) → CPU EM optimiser on 945k patterns → both jobs
died in refine #2 (LG+R5 on CPU).

**Two findings, one fix.** (i) Correctness: the projection breaks H2 recall on `-m MF` (LG+G4 ∉ projected top-3).
(ii) Performance: the promoted ineligible models are a CPU-at-1M time bomb. The fix (landed, re-running 170756438/440):
**rank the top-k gate by NATIVE subsample BIC over all candidates** (LG+G4 returns to #1), add a **rate-heterogeneity
detector** (flag if a +R/+I model genuinely leads the best eligible by more than the ~Δp/2 overfit cushion — here it
does NOT: LG+G4 leads LG+R4 by 43 nat), and **cap each refine with a wall budget**; a provably-losing ineligible model
is skipped (detector-justified), never silently excluded.

**The deeper lesson (red-team).** On 5000 sites the rate-model choice (+G vs +I+G vs +R) is decided *entirely by the
penalty* — the fits are statistically indistinguishable (<1 nat). So (a) the projection, which amplifies that sub-nat
noise, is the worst possible ranking for this decision; native BIC (penalty `ln m`) is correct precisely because it
*trusts the penalty, not the noise*. And (b) this is itself a **subsample-sufficiency result**: rate-model selection
may be an intrinsically large-N decision, so a small subsample "chooses" the rate model by parsimony, not evidence —
which is fine here (the parsimonious LG+G4 IS the oracle) but is a caveat for data where a richer rate model truly wins.

**Methodology flaw banked, and the +R discrepancy RESOLVED.** The sweep validated a *proxy* (the CPU/MPI reference
binary) that never exercised the projection (pipeline code, not binary). The "CPU-vs-GPU +R gap" I flagged is now
explained, and it is sharper than "the fits differ": **the two binaries evaluate a DIFFERENT +R candidate set.** On the
identical 5000-site subsample the CPU reference `mf_5000_1.iqtree` contains **only +R2** (LG+R2 BIC 790191, LG+I+R2
789778 — both ~8000 nat behind LG+G4, no threat under any ranking), whereas the GPU `coarse.iqtree` evaluated the full
**R2→R5 ladder** (LG+R4 782105, LG+R5 782120 — within ~60 nat of LG+G4). The two IQ-TREE forks differ in their FreeRate
auto-category search depth (the MPI fork stopped at R2; the GPU fork climbed to R5). So the sweep couldn't see the bug
not because of fit quality but because **the proxy never generated the R4/R5 candidates the projection promotes.** This
*strengthens* the lesson: a proxy binary with a different candidate set cannot validate the pipeline — **validate the
exact shipped pipeline end-to-end; the CPU full-data MFP is an oracle for the final answer only.** (Open: confirm which
fork's `-cmax`/`-mrate` default is "correct" — but it is no longer a mystery, and it does not affect the fix.)

---

## X.6 Experiment design and metrics (retrospective — sweep completed, §X.5 has results)

**Design (the data-statistics test, optimiser-invariant ⇒ run on the CPU reference binary, no GPU contention):**

- For each subsample length `L ∈ {1k, 2k, 5k, 10k, 20k, 50k, 100k}` (and the full N as the reference):
  - Draw **R independent random column subsamples** (R ≥ 3, distinct seeds) — resampling is what turns a single
    ranking into a _recall probability_ and exposes rank instability.
  - Run ModelFinder (`-m TEST` first for a dense cheap sweep; `-m MF` at a few lengths to probe the +R twist of
    §X.3.1) and record the **full candidate BIC table** (model, lnL, p, BIC), the winner, and the top-3.
- **Reference winner / ranking:** the full-N `-m MF` run (AA: LG+G4, with its complete BIC table; DNA: TBD — needs a
  full DNA run, noting the free-Q coverage caveat is irrelevant to BIC, which is optimiser-invariant).

**Metrics, all as a function of L:**
1. **Recall (H2, the headline):** fraction of resamples whose top-3 contains the full-winner. CTF is licensed at the
   smallest L where recall = 1.0 across resamples.
2. **Exact-winner agreement (H1):** fraction whose rank-1 == full-winner.
3. **ΔBIC margin growth:** winner-vs-runner-up ΔBIC at each L — theory (§X.3) predicts ~linear growth in L for the
   separated case; this is the direct empirical check of the `−2Nδ` term.
4. **Kendall-τ** of the subsample BIC ranking vs the full ranking — measures whether the _whole_ ordering, not just
   the top, stabilises.
5. **L\* estimate** per competing pair: fit `ΔBIC ≈ aL + b·ln L` and read off where the sign locks; compare to the
   `z²(σ_δ/δ)²` prediction.

**What a PASS looks like:** recall reaches 1.0 by L = 5000 across resamples and data types, ΔBIC margins grow ~linearly
(confirming the mechanism), and no +R model spuriously enters the top-3 at small L. **What a FAIL looks like:** recall
< 1.0 at 5000 for some data, or a +R model that ranks top-3 on the subsample but is not competitive at full N — either
would force a larger subsample, a larger k, or a regime-aware subsample size.

---

## X.7 Literature grounding (research pass, 2026-06-13 — honest verdicts, not confirmation)

A focused literature pass against the four sub-questions. Each carries a verdict _relative to our hypothesis_;
"inference" tags my own reasoning where no direct source was found. No direct prior art exists for our exact scheme
(rank-on-site-subsample → refine-on-full → report recall) — **that remains our novel claim to validate empirically.**

**1 — BIC/AIC selection consistency & finite-sample rate. [SUPPORTS, with a near-tie caveat.]** BIC (Schwarz 1978) is
selection-_consistent_; AIC is _efficient_ but overfits, and the two cannot be jointly attained (Yang 2005; Shao 1997;
Hannan–Quinn 1979 put `ln n` just inside the consistency boundary). The true model's lnL advantage grows ~linearly in
n while the penalty grows ~`ln n`, so **well-separated models resolve at modest n; near-ties (small ΔBIC) converge
slowly and are where pre-asymptotic instability lives** (Chaurasia & Harel 2013 on selection rates). Phylogenetic
confirmation: **Luo et al. 2010 (BMC Evol Biol 10:242)** — AIC "selected a dozen different best-fit models per 100
replicates" (unstable) while BIC was stable/parsimonious, and **~1000–2000 bp sufficed** for most criteria. Matches
our regime exactly: our top-3 sit behind a 17,618-nat cliff (well-separated → recovered early).

**2 — Subsampling/subsetting sites for phylo model selection. [QUALIFIES — strong precedent-in-spirit, no direct
evidence for site-subsample-then-refine recall.]** Approximating the _scoring step_ is standard, accepted practice:
**ModelFinder itself** (Kalyaanamoorthy et al. 2017, Nat Methods 14:587) scores models on a _fixed initial parsimony
tree_, not a per-model re-optimised topology; **PartitionFinder rcluster/rclusterf** (Lanfear et al. 2014 BMC EB 14:82;
2017 MBE 34:772) evaluates only a shortlisted search space yet returns better AICc/BIC schemes; **RELL / UFBoot**
(Kishino–Hasegawa; Shimodaira; Minh et al. 2013 MBE 30:1188) reuse per-site lnL as a validated site-resampling
shortcut; Posada & Buckley 2004 (Syst Biol 53:793) note alignment "sample size" is itself subtle. But **none subsample
the number of sites n, rank models, and refine on full data with a measured recall** — they approximate the tree or
the search space, not n. Precedent in spirit, not a validation of our scheme.

**3 — Boundary-parameter instability (+I at pinv→0, +R collapse). [QUALIFIES — instability is concentrated exactly
here.]** On the parameter-space boundary the LRT null is a chi-bar-squared _mixture_, not χ² (Chernoff 1954; Self &
Liang 1987; **Susko 2013, Biometrika 100:1019** gives the phylo-facing data-dependent-df remedy). Information criteria
do _not_ apply this correction — they just count k — so a +I model with true pinv≈0 contributes a poorly-determined
parameter whose small, noisy lnL gain makes ΔBIC{base+G vs base+I+G} a near-tie that flips easily on small n. Luo et
al. 2010 found selection accuracy _worse at low pinv_. This is exactly our observed `+I+G` runner-up at ΔBIC≈14 / the
pinvar declines — the part of the ranking least safe to trust from a subsample.

**4 — High-parameter overfitting distorting ranking. [CONTRADICTS — mild but principled; see §X.3.2.]** General
overfitting of parameter-rich RHAS/+Rk models is documented (bounded category counts ~6–10 recommended); Luo et al.
2010's AIC instability _is_ small-sample criterion overfitting, which BIC resists. **No citable "minimum
sites-per-parameter" threshold was found.** The mechanism (the agent's, extended in §X.3.2): on a subsample the
penalty `k·ln(n)` is ~38 % weaker (ln 5000 vs ln 1e6), under-guarding +R — and CTF's `(N/m)` projection then
_amplifies_ the n-independent overfit excess ~189×. This biases the ranking _toward_ the rich models that should fall
back. It is the one place the literature points against the naive hypothesis, and it is why the sweep includes `-m MF`
and reports recall under both native and projected BIC.

**Bottom line for CTF (literature).** The core hypothesis is supported for the easy, dominant case — BIC is a stable,
consistent selector whose discrete winner locks in early when the field is well-separated (our 17,618-nat cliff), and
approximating the scoring step is established practice. It is **not unconditionally true**, and the danger is
_concentrated, not diffuse_: near-tied competitors within the top-k, and boundary/over-parameterised models (+I near
pinv→0, +R/+Rk), where broken boundary asymptotics _plus_ a subsample-shrunk (and projection-amplified) penalty can
reorder the shortlist. CTF's top-k refine is the right insurance — provided k is wide enough and we treat any +R/+I
model near the cut as the least trustworthy. **No one has published our exact scheme's recall; that is ours to prove.**

_Sources:_ Schwarz 1978; Shao 1997; Yang 2005 (Biometrika); Hannan–Quinn 1979; Chaurasia & Harel 2013 (EJS);
Kalyaanamoorthy et al. 2017 (Nat Methods 14:587); Lanfear et al. 2014 (BMC EB 14:82), 2017 (MBE 34:772); Minh et al.
2013 (MBE 30:1188); Posada & Buckley 2004 (Syst Biol 53:793); Luo et al. 2010 (BMC EB 10:242); Susko 2013 (Biometrika
100:1019); Chernoff 1954; Self & Liang 1987; Abadi et al. 2019 (Nat Commun, "model selection may not be mandatory").

---

## X.8 Verdict (SPLIT — corrected by the CTF runs + Opus re-verification, 2026-06-13)

**The hypothesis holds under the RIGHT ranking, and fails under the one CTF actually shipped.**

- **Under NATIVE subsample BIC (the statistically correct gate):** H2 recall = 1.00 AND exact rank-1 = LG+G4 across
  all 21 `-m TEST` runs (L ∈ {1K–100K} × 3) and both `-m MF` runs; ΔBIC grows as ~ln(L) (penalty-dominated, Δp=1).
  Full recall already at L=1000. **These numbers are verified correct.** So the subsample-sufficiency hypothesis
  itself holds on AA-1M, and CTF's L=5000 choice is safe — *provided the gate ranks by native BIC.*
- **Under the PROJECTED BIC that CTF actually used:** H2 recall **FAILS** on `-m MF` (§X.5.5) — LG+G4 drops to rank 4,
  +R models are promoted, and both 1M benchmarks timed out refining +R on CPU. The §X.3.2 amplification risk is
  **confirmed, not absent.** Sonnet's earlier "did NOT trigger" conclusion was an artifact of testing native BIC on a
  proxy binary.

**What this establishes:** (1) the hypothesis is true on AA-1M for native-BIC ranking; (2) the CTF *implementation* had
a real correctness+performance bug (the projected gate), **now fixed AND confirmed end-to-end** — the native-BIC gate
+ rate-het detector + wall budget re-run (H200 170756438) completed in **767 s** and picked **LG+G4** (BIC
157213275.8, beating LG+I+G4 by 14), with the run log showing the *old* projected gate would have dropped LG+G4 to
rank 4; (3) rate-model selection on a 5000-site subsample is penalty-dominated (fits within <1 nat), so the projection
— amplifying sub-nat noise 378× — was the worst possible ranking for it.

**What this does not establish:**
- Safety for other alignments, especially rate-heterogeneous data where +R models are genuinely competitive
- Safety for small N (the 1M benchmark is the hardest case to fail; smaller N means smaller δ, harder identification)
- Absence of +R amplification risk (§X.3.2) — the -m MF probe had only 2 runs, and LG+I+G4 is a simple nested competitor, not a true +R rivalry
- DNA models, free-rate, or non-LG families

**Correct status: _empirically supported on the AA-1M benchmark case, mechanism confirmed, extrapolation to other regimes requires further evidence._** The §X.5.4 caveats should accompany any citation of these results.
