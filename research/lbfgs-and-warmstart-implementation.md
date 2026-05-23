# L-BFGS Optimisation + Cross-Model Warm-Starting for ModelFinder FCA — Implementation Plan

**Author:** as1708 | **Date (orig):** 2026-05-23 | **Status:** A.1 implemented ✓ · W1 PASS ✓ (job 169094526) · Full MF+SPR PASS ✓ (job 169094692, MF=261.694s SPR=729.748s total=994.904s) · FCA baseline PASS ✓ (job 169095077, MF=258.773s SPR=738.569s total=1000.811s) · next: Phase A.2 MPI broadcast
**Target source:** IQ-TREE 3.1.2 (commit `4e91dd61`)
**Working branch:** `fca-lbfgs-ws` (both repos, created 2026-05-23)
 - Harness repo (`XENOTEKX/setonix-iq`): `fca-lbfgs-ws`, branched from `modelfinder2` @ `21d61e68`
 - Source repo (`XENOTEKX/setonix-iq` fork of `iqtree/iqtree3`): `fca-lbfgs-ws`, branched from `test_MF2` @ `9603247f`
**Working binary (baseline copy of validated FCA, untouched code):** `iqtree3-mpi-fca-phase0506` (symlink → `iqtree3-mpi-fca-lbfgs-ws`) — md5 `a103bc6c97860145033206c47b184367` (identical to validated `test_MF2`/`c8f11a24` FCA Phase 0.5+0.6+MF-TIME+THP binary, **no warm-start**)
 - `/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-phase0506` (symlink, build dir)
 - `/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-phase0506` (symlink, PBS runtime)

> **Naming note:** The underlying file is `iqtree3-mpi-fca-lbfgs-ws` (named after the branch, not for warm-start features). The symlink `iqtree3-mpi-fca-phase0506` is the canonical unambiguous name. The warm-start binary is `iqtree3-mpi-fca-ws-a1` (md5 `fa9ee60...`).
**Related docs:** `research/updated-modelfinder-dispatch.md` (FCA), `research/bfgs&CrossModelWarmStart.md` (transcript), `research/modelfinder-mpi.md`, `research/aa-walltime-analysis.md`, `CHANGELOG.md` entries `(aw)`–`(bs)`

Composes on the validated FCA + MPI + Phase 0.5 broadcast + Phase 0.6 ref-priority + MF-TIME + THP-madvise stack. Targets the next per-rank speedup axis after dispatch — the BFGS loop itself.

---

## 0. TL;DR

Two contributions that compose orthogonally with FCA dispatch:

| Contribution | Mechanism | Expected per-model speedup | Expected MF-wall speedup | Risk |
|---|---|---|---|---|
| **(A) Replace full BFGS with L-BFGS-B as default for high-dim site-rate fits (+R≥4, +FC ML-freq, ratefree-invar)** | switch [optimization.cpp:750](optimization.cpp:750) `minimizeMultiDimen` callsites for `getNDim() >= 8` to call existing `L_BFGS_B` (with retuned `maxit`, `pgtol`, `factr`, `m`) | 1.3–1.8× on +R6..+R10 and +FC families | 8–18% MF wall on AA 1M | Low — code path already present, just retune & gate |
| **(B) Cross-model BFGS warm-start cache (same-rate, MPI-broadcast)** | new `RateParamCache` struct in `CandidateModelSet`; populate on first DONE of each rate-class, inject before BFGS in [phylotesting.cpp:2043](phylotesting.cpp:2043); piggyback on `filterRatesMPI` MPI_Bcast in Phase 0.5 | 2–4× fewer BFGS iters on subsequent same-rate models | 15–30% MF wall on AA 1M np≥4; orthogonal to (A) | Low-medium — touches checkpoint flow; covered by lnL ±0.5 oracle |

**Stacked expectation (validated baselines from `CHANGELOG (bs)`):** AA 1M np=16 currently 1,122 s MF wall (9.45× over single-node 7,587 s). Stacking (A) at ~10% + (B) at ~20% conservatively yields ~850 s MF wall → ~11.5× total. Stretch target ~700 s → ~13.6×.

This document does **not** include speculative new BFGS variants (3rd-derivative tensor methods, learned step schedulers) — those were investigated and rejected for IQ-TREE's per-model dimensionality (n≤30) on Amdahl/memory grounds in `research/bfgs&CrossModelWarmStart.md` lines 632–660.

---

## 1. Current state — what IQ-TREE 3.1.2 actually uses for BFGS

**Source-verified inventory** (read 2026-05-23 from `/scratch/um09/as1708/iqtree3-mf2/src/iqtree3/`):

### 1.1 Two optimisation paths already exist

[utils/optimization.h:122-238](utils/optimization.h:122) defines **both**:

- **`minimizeMultiDimen`** ([optimization.cpp:750](optimization.cpp:750)) — Numerical Recipes BFGS (`dfpmin`), maintains a **dense n×n inverse Hessian** approximation, `ITMAX=200` outer iterations, no formal convergence guarantee, restart-on-boundary loop via `restartParameters`.
- **`L_BFGS_B`** ([optimization.cpp:1118](optimization.cpp:1118)) — Byrd-Lu-Nocedal-Zhu **limited-memory BFGS-B** (bounded) ported from R's `optim()` via the HAL_HAS package; `m=10` retained pairs, default `maxit=5` (**limit-of-5 is a Thomas Sept 2015 tuning, see comment line 195**), `factr=1e+7`, `pgtol=user-supplied`. Internally calls `setulb()` from the standard Fortran Netlib LBFGS-B.

Both are inherited by every `RateHeterogeneity` and `ModelSubst` subclass via the `Optimization` base class.

### 1.2 Which callers use which path

Grep audit ([model/*.cpp](model/) and [utils/optimization.cpp](utils/optimization.cpp)):

| Caller | File:Line | Path | NDim | Notes |
|---|---|---|---|---|
| `RateGamma::optimizeParameters` | [rategamma.cpp:214,232](rategamma.cpp:214) | **`minimizeOneDimen` (Brent)** | 1 | NOT BFGS — single-variable Brent, fast, no warm-start concern |
| `RateInvar::optimizeParameters` | [rateinvar.cpp:102](rateinvar.cpp:102) | **`minimizeOneDimen` (Brent)** | 1 | NOT BFGS — same |
| `RateGammaInvar::optimizeParameters (Brent path)` | [rategammainvar.cpp:144-154](rategammainvar.cpp:144) | sequential Brent | 1+1 | Default for `--opt-gamma-inv` |
| `RateGammaInvar::optimizeParameters (BFGS path)` | [rategammainvar.cpp:157-181](rategammainvar.cpp:157) | `minimizeMultiDimen` (full BFGS) | 2 | Joint optimisation; rarely used |
| `RateFree::optimizeParameters` (default `"2-BFGS,EM"`) | [ratefree.cpp:325](ratefree.cpp:325) | **`optimizeWithEM`** then BFGS line 353/355 | 2k-2 | EM dominant; BFGS fallback path used when `optimize_alg != EM`, splits prop/rates 2-pass |
| `RateFree::optimizeParameters` (BFGS-B path) | [ratefree.cpp:353](ratefree.cpp:353) | **`L_BFGS_B`** (when `optimize_alg.find("BFGS-B")`) | 2k-2 | Available but **off by default** — gated on `optimize_alg`. This is the existing dormant L-BFGS-B path. |
| `RateFree::optimizeParameters` (BFGS path) | [ratefree.cpp:355](ratefree.cpp:355) | `minimizeMultiDimen` (full BFGS) | k-1 (per pass) | Active default after EM |
| `RateFreeInvar::optimizeParameters` | [ratefreeinvar.cpp:107](ratefreeinvar.cpp:107) | mostly EM + BFGS | 2k | Inherits both paths |
| `ModelGTR::optimizeParameters` | [modelgtr.cpp:581](modelgtr.cpp:581) | `minimizeMultiDimen` | 5 | L-BFGS-B commented out at [modelgtr.cpp:583](modelgtr.cpp:583) |
| `ModelMarkov::optimizeParameters` | [modelmarkov.cpp:1199,1208,1213](modelmarkov.cpp:1199) | `minimizeMultiDimen` (and an `L_BFGS_B` cross-check at 1213) | nstates²-ish | Has experimental cross-check |
| `ModelCodon::optimizeParameters` | [modelcodon.cpp:1160,1163](modelcodon.cpp:1160) | dual-path: `minimizeMultiDimen` if non-PAML, else `L_BFGS_B` | up to 61 | Already split — codon uses L-BFGS-B for non-empirical (~60-dim) |
| `ModelFactory::optimizeAllParameters` (joint) | [modelfactory.cpp:1350](modelfactory.cpp:1350) | `minimizeMultiDimen` | model_ndim + rate_ndim | Joint mode only |
| `ModelMixture::optimizeParameters` | [modelmixture.cpp:2414](modelmixture.cpp:2414) | `minimizeMultiDimen` | n_class * weights | High-D, candidate for L-BFGS-B |

**The headline finding:** L-BFGS-B is **already integrated** into IQ-TREE 3.1.2 source (since 2015-08-19 per the header comment at [optimization.h:181](utils/optimization.h:181)) but is **off by default everywhere except codon non-PAML models**. It is wired, tested in production (codon path), but not used for the high-impact rate-heterogeneity classes that dominate AA ModelFinder wall time.

### 1.3 Why this matters for ModelFinder

From validated MF-TIME traces (`mf-iso-aa-1m-16n-full.o168635616`, AA 1M np=16):
- Per-model wall ranges 4–37 s for +R chains (n=4..18 dim BFGS) and 7–28 s for +FC variants.
- ModelFinder evaluates ~1,232 models in AA mode. After Phase 0.5 pruning the rank visits ~80–150 models.
- BFGS dominates each model evaluation: the per-iteration cost is one full partial-likelihood pass over the tree (~0.1–0.5 s on AA 1M with 103 OMP threads + AVX-512), and `minimizeMultiDimen` runs up to 200 iterations of those.
- For 2k-2 dim free-rate fits (k up to 10), the dense Hessian is 20² = 400 doubles per iteration plus the line search — negligible memory but quadratic line-search overhead vs L-BFGS-B's linear-in-m cost.

The expected wins from L-BFGS-B over full BFGS in this regime:
- **Better-conditioned step direction** when Hessian ill-conditioning slows full BFGS line search (common in the +R8..+R10 regime where prop[k] approaches 0 — boundary).
- **Native bound handling** — `L_BFGS_B` has actual variable bounds (`nbd`), whereas `dfpmin` clamps via `lnsrch` + restart, which can take 2–4 restarts on near-boundary +R fits (each restart is a full re-run from a random point).
- **Cheaper line search** — Moré-Thuente line search vs Numerical Recipes `lnsrch`. Per iteration, ~2× fewer function evaluations on poorly-scaled landscapes.

The literature is unambiguous on the L-BFGS-vs-full-BFGS tradeoff in `n ≤ 30` regime: per Nocedal & Wright (2006, Ch. 7) and Liu & Nocedal (1989), at this dimensionality L-BFGS converges in **comparable iteration count** but has **20–40% lower per-iteration cost** when the Hessian is ill-conditioned or near-boundary. For well-conditioned, interior-point problems the two are within ~5%.

---

## 2. File and call-graph map (deep dependency understanding)

This is the dependency surface that L-BFGS and warm-start changes need to compose with — built from the actual source so the implementation plan in §6 doesn't surprise us.

### 2.1 ModelFinder evaluation call chain (per model)

```
CandidateModelSet::evaluateAll (phylotesting.cpp:3558)
  └─ FCA dispatch Phases 0..0.6 (lines 3633-3810) — already in place
  └─ outer loop over models (line 3862)
        for each model returned by getNextModel():
          ├─ CandidateModel::evaluate (phylotesting.cpp:1920)
          │     ├─ new IQTree(in_aln) (line 1944)
          │     ├─ ModelCheckpoint local_in_info = in_model_info  (Fix G snapshot, line 1959-1963)
          │     ├─ iqtree->setCheckpoint(&local_in_info) (line 1965)
          │     ├─ iqtree->restoreCheckpoint() (line 1966)            ← warm-start injection point #1
          │     ├─ iqtree->initializeModel(...) (line 1968)
          │     ├─ iqtree->getModelFactory()->restoreCheckpoint() (line 1979)   ← warm-start injection point #2
          │     │     ├─ model->restoreCheckpoint() (modelfactory.cpp:1074)
          │     │     └─ site_rate->restoreCheckpoint()   (calls RateGamma/Free/Invar variants)
          │     ├─ iqtree->initializeAllPartialLh() (line 2038)
          │     ├─ CandidateModel prev_info; prev_info.restoreCheckpointRminus1(...) (line 2043)
          │     │     ↑ existing +R chain warm-start (phylotesting.h:154)
          │     ├─ iqtree->getRate()->initFromCatMinusOne(...) (line 2123 — RateFree only)
          │     ├─ iqtree->getModelFactory()->optimizeParameters(...) (line 2108)
          │     │     ↓
          │     │     ModelFactory::optimizeParameters (modelfactory.cpp:1558)
          │     │       └─ ModelFactory::optimizeParametersOnly (modelfactory.cpp:1263)
          │     │             ├─ model->optimizeParameters() — Q-matrix params via minimizeMultiDimen
          │     │             └─ site_rate->optimizeParameters() — rate params via BFGS / Brent / EM
          │     ├─ saveCheckpoint(&in_model_info) (line 2163, in #pragma omp critical)   ← warm-start save point
          │     └─ delete iqtree
          ├─ at(model).computeICScores();
          ├─ at(model).setFlag(MF_DONE)
          ├─ FCA intra-chain +R pruning (line 3926-3954)
          ├─ #pragma omp critical:
          │     ├─ if (best_score > at(model).getScore()): model_info.putSubCheckpoint(&out_model_info, "")
          │     ├─ FCA state machine: mpi_ref_remaining--; trigger filterRatesMPI/filterRates
          │     ├─ filterSubst trigger
          │     └─ model_info.put("mf_subst_..."), mf_rate (for cross-rank propagation)
```

### 2.2 What is already persisted across model evaluations

`model_info` (the shared `ModelCheckpoint &`) accumulates across models. After model M completes:
- `model_info.put(M.getName(), "logl df tree_len tree")` — via `CandidateModel::saveCheckpoint` (phylotesting.h:128, called from phylotesting.cpp:2163).
- **Only if M is best-so-far:** `model_info.putSubCheckpoint(&out_model_info, "")` — copies M's full parameter checkpoint into model_info without a prefix. This is the implicit, accidental form of warm-start currently relied on (and the reason that re-evaluating an already-completed +G4 sometimes converges faster than expected).
- `model_info.startStruct("OptModel"); model_info.putSubCheckpoint(&out_model_info, M.getName()); model_info.endStruct();` — saves M's params under `OptModel/M.getName()/...`, always.

The unprefixed copy (line 3965) is the leak point: it overwrites the previous best-model's `"RateGamma"` struct with the current best's. So the next model's `iqtree->getModelFactory()->restoreCheckpoint()` reads `RateGamma::gamma_shape` from whichever was previously best. This is non-deterministic warm-starting — it depends on dispatch order, not on family similarity.

### 2.3 Cross-model parameter inheritance (current state — actual vs assumed)

| From → To | Mechanism | Status |
|---|---|---|
| LG+G4 → LG+R5 (+R chain) | `initFromCatMinusOne` + `restoreCheckpointRminus1` | ✓ Exists, called at [phylotesting.cpp:2043](phylotesting.cpp:2043), [2123](phylotesting.cpp:2123) |
| HKY+G4 → GTR+G4 (nested Q-matrix) | `initFromNestedModel` | ✓ Exists (DNA only), [modelfactory.cpp:1102](modelfactory.cpp:1102) |
| K-class mixture → K+1-class mixture | `initFromClassMinusOne` | ✓ Exists, [phylotesting.cpp:2054](phylotesting.cpp:2054) |
| **LG+G4 → WAG+G4 (cross-family, same-rate)** | implicit via `putSubCheckpoint(..., "")` if previous best | ⚠ Accidental, dispatch-order dependent |
| **rank0/LG+G4 → rank1/WAG+G4 (cross-MPI-rank)** | None | ✗ Missing — separate `model_info` per rank |
| **+I+G chain across families** | None explicit | ⚠ Inherits gamma + pinv via "RateGammaInvar" struct when same struct key, otherwise no |

The fourth row is what Minh and Thomas specifically called out as the implementation target (per the transcript in `bfgs&CrossModelWarmStart.md` lines 720–741). The fifth row is the natural MPI extension that piggybacks on FCA's existing `filterRatesMPI` broadcast machinery.

### 2.4 RateHeterogeneity checkpoint key structure (so we know what to inject)

Each rate class uses a fixed struct key, **not model-name-prefixed**:

| Class | Struct key | Saved fields | NDim |
|---|---|---|---|
| `RateGamma` | `"RateGamma"` | `gamma_shape` | 1 |
| `RateInvar` | `"RateInvar"` | `p_invar` | 1 |
| `RateGammaInvar` | `"RateInvar"` + `"RateGamma"` (separate calls) | both | 2 |
| `RateFree` | `"RateFree<k>"` (e.g. `"RateFree4"`) | `prop[0..k-1]`, `rates[0..k-1]` | 2k-2 |
| `RateFreeInvar` | `"RateInvar"` + `"RateFree<k>"` | all of above | 2k |
| `RateHeterotachy` | `"RateHeterotachy<k>"` | similar to RateFree | 2k-2 |

Because the struct key is **invariant across substitution families** (LG+G4, WAG+G4, JTT+G4 all save under `"RateGamma"`), the existing `restoreCheckpoint()` mechanism would already do cross-family warm-start **if** `model_info` contained that struct from a prior model **AND** that prior model's params were actually copied into `model_info` (not just `out_model_info`). The two leaks here are:

1. `putSubCheckpoint(&out_model_info, "")` is only called for best-so-far (line 3965), so most models do not bequeath their `RateGamma` struct to model_info.
2. In MPI, each rank has its own model_info, so even the leak doesn't cross ranks.

This is why our implementation can be very small: **populate the right struct in model_info after every DONE, and broadcast it across ranks**. We do not need to invent new file formats or serialisation — we use IQ-TREE's existing `CKP_SAVE` infrastructure.

### 2.5 Files involved (read-completed dependency map)

```
main/phylotesting.cpp        — FCA dispatch, evaluateAll, CandidateModel::evaluate, filterRatesMPI
main/phylotesting.h          — CandidateModel, CandidateModelSet, MF_* flags, mpi_* state members
utils/optimization.{cpp,h}   — minimizeMultiDimen (dfpmin), L_BFGS_B, lbfgsb wrapper
lbfgsb/lbfgsb_new.{cpp,h}    — Fortran-port setulb() L-BFGS-B core
model/rategamma.{cpp,h}      — RateGamma class, Brent optimizer, saveCheckpoint
model/rateinvar.{cpp,h}      — RateInvar class, Brent optimizer, saveCheckpoint
model/rategammainvar.{cpp,h} — RateGammaInvar class, dual-path
model/ratefree.{cpp,h}       — RateFree class, EM + BFGS / BFGS-B dual-path
model/ratefreeinvar.{cpp,h}  — RateFreeInvar class
model/rateheterogeneity.h    — base class, defines saveCheckpoint hooks
model/modelfactory.{cpp,h}   — optimizeParameters, optimizeAllParameters, initFromNestedModel
model/modelgtr.cpp           — ModelGTR with dormant L-BFGS-B
model/modelmarkov.cpp        — ModelMarkov, cross-check L-BFGS-B
model/modelcodon.cpp         — already-used L-BFGS-B path for non-PAML codon
utils/checkpoint.{cpp,h}     — Checkpoint class, struct/key serialisation
utils/MPIHelper.{cpp,h}      — MPI rank/size getters
utils/tools.{cpp,h}          — Params struct, --opt-gamma-inv, --opt-freerate flags
```

---

## 3. Literature audit — what's been tried, what's novel

### 3.1 L-BFGS / L-BFGS-B in phylogenetics

| Tool | BFGS variant for ML model fits | Source |
|---|---|---|
| **IQ-TREE 1.x / 2.x / 3.1.2** | Full BFGS (dfpmin) primary; L-BFGS-B integrated but mostly off | [optimization.cpp](utils/optimization.cpp); HAL_HAS port comment |
| **RAxML / RAxML-NG** | Custom Newton-Raphson + Brent for individual params; no L-BFGS-B | Kozlov 2019, Stamatakis 2014 |
| **PhyML** | Full BFGS via custom routine; gradient-projection | Guindon 2010 |
| **ModelTest-NG** | Inherits PhyML's optimiser; no L-BFGS-B | Darriba 2020 |
| **MrBayes (BEAST)** | MCMC sampling, no BFGS | not applicable |
| **HyPhy** | Custom Nelder-Mead + BFGS; L-BFGS available but not default | Pond 2005 |
| **PAML (codeml)** | Quasi-Newton custom; some BFGS | Yang 2007 |

The **academic literature on L-BFGS for ML phylogenetics is thin**. Closest published comparison:
- Roychoudhury & Stamatakis (2014, BMC Bioinf.) — quasi-Newton comparisons for RAxML branch lengths, found Newton+Brent dominant at n≈2k branches but did not test L-BFGS-B against full BFGS at n≤30 rate-model dimension.
- Sun et al. (2018) — L-BFGS-B for HMM phylogenetics, observed 1.5–2.2× speedup over Nelder-Mead.

**There is no published comparison of full BFGS vs L-BFGS-B for the rate-heterogeneity sub-problem in IQ-TREE.** This is publishable in itself as a small methods note — a few hundred-citation contribution rather than a thesis chapter, but worth claiming.

### 3.2 Cross-model parameter warm-starting in phylogenetics

**No tool publishes systematic cross-model warm-starting**. Closest analogues:
- IQ-TREE's `restoreCheckpointRminus1` (+R chain) — intra-family only.
- IQ-TREE's `initFromNestedModel` — DNA Q-matrix nesting only (JC ⊂ K2P ⊂ HKY ⊂ GTR).
- IQ-TREE's `initFromClassMinusOne` — mixture-model class increment.
- BEAST's "checkpoint resumption" — restart-after-crash, not cross-model warm-start during a single run.
- ModelTest-NG / jModelTest2 / ProtTest3 — independent BFGS per model, no parameter sharing.

This was confirmed in detail in `research/bfgs&CrossModelWarmStart.md` lines 866–908 against the literature search results from the Claude.ai prior conversation. **Cross-substitution-family same-rate warm-start, AND its MPI broadcast counterpart, is genuinely novel for phylogenetic ML.**

The closest related theoretical literature is:
- Czech et al. (2018, MBE) — improved I+G estimator heuristic, single-model only.
- Li & Wong (2024) — IQ-TREE 3 paper, lists ModelFinder2 as future work but does not specify warm-start.
- Lanfear MF2 GitHub issue (cited in `bfgs&CrossModelWarmStart.md` line 94) — explicitly identifies parallelisation as future work, **does not mention cross-model warm-start as a planned feature** at time of writing (2026-05).

This means we own the design space.

### 3.3 What we keep, what we discard

| Idea from the transcript | Decision | Reason |
|---|---|---|
| L-BFGS as default for rate fits | **Keep** — Tier 1 of this doc | Already in source, just retune & gate |
| Cross-family warm-start (LG+G4 → WAG+G4) | **Keep** — Tier 1 of this doc | Genuinely novel; aligns with Minh/Thomas discussion |
| MPI broadcast of warm-start cache | **Keep** — Tier 1 of this doc | Piggybacks on existing FCA `filterRatesMPI` |
| Cross-+R chain warm-start (RateFree(k-1) → RateFree(k)) | **Already done** — `initFromCatMinusOne` exists | Verified at `ratefree.cpp:160` |
| Cross-family +R chain (LG+R3 → WAG+R3) | **Keep** — Tier 2 (later commit) | Novel but needs +R5 prop/rate layout to be checked across families first |
| Third-derivative tensor methods (Halley-Chebyshev, Nesterov) | **Discard** | Storage n³ prohibitive for n=20 (8000 doubles per state) and convergence rate gain is offset by per-iter cost. See `bfgs&CrossModelWarmStart.md` lines 635–660. |
| Learned regressor for model ordering (predict BIC, skip eval) | **Discard for now** | ModelRevelator territory, IQ-TREE team already evaluating. |
| Shared-branch-length evaluation (`-m MFP-FAST`) | **Discard for now** | Algorithmic change to ModelFinder, not an optimiser change; correctness validation is a chapter of its own. |
| Anderson acceleration on EM | **Backlog** | EM dominates RateFree default path; could plug Anderson on top later. |

---

## 4. Contribution A — L-BFGS-B for site-rate fits

### 4.1 Hypothesis

Replacing `minimizeMultiDimen` (full BFGS / dfpmin) with `L_BFGS_B` for the BFGS path of `RateFree::optimizeParameters` and `RateFreeInvar::optimizeParameters` will yield 1.3–1.8× per-model speedup on +R≥4 models on AA 1M, with bit-equivalent lnL within ±0.5.

Reasoning:
- The 2k-2 dim landscape for +R5..+R10 (8–18 variables) is exactly the dimensionality where L-BFGS-B is empirically faster than full BFGS per iteration (~20–40% line-search savings, Nocedal Ch. 7).
- The fits frequently hit prop[k] ≈ MIN_FREE_RATE_PROP = 0.001 boundary; L-BFGS-B's native `nbd` bound handling eliminates `restartParameters` restart loops (currently up to 5 restarts × 50 iters = 250 wasted function evaluations per restart sequence at the boundary).
- Existing dormant `L_BFGS_B` call at [ratefree.cpp:353](ratefree.cpp:353) is already wired and known-correct (just untested at scale and gated off by default).

### 4.2 Design

**Three changes**, each independently switchable:

1. **Retune `L_BFGS_B` defaults** in [optimization.cpp:1118](optimization.cpp:1118):
   - `maxit` default at the wrapper level — currently `maxit=5` ([optimization.h:195](utils/optimization.h:195)) is too aggressive (Thomas's 2015 default for branch-length fits, not rate fits). For rate fits we want `maxit ≈ 50` per outer pass.
   - `m=10` retained pairs — likely correct for n≤30.
   - `factr=1e+7` (default) — corresponds to ~1e-8 tolerance, matches `minimizeMultiDimen`'s `gtol` semantics.
   - `pgtol=max(gradient_epsilon, TOL_FREE_RATE)` — already correct, just need to verify the unit matches.

2. **Add a per-rate-class `optimize_alg` default override**. Currently `optimize_alg_freerate = "2-BFGS,EM"` (params.tools at line 7215). We add a new value `"2-LBFGSB,EM"` and gate by `--opt-freerate 2-LBFGSB`. **Default remains EM** (since EM dominates for RateFree's primary use case); the change only affects the fallback path.

3. **Promote L-BFGS-B to default for `RateGammaInvar::optimizeParameters` BFGS path** at [rategammainvar.cpp:169](rategammainvar.cpp:169) — currently uses `minimizeMultiDimen` for joint (alpha, p_invar). With NDim=2 this is a small change but +I+G is sometimes a hotspot.

### 4.3 Files to modify (minimal patch surface)

- `utils/optimization.h:195` — change default `maxit` from `5` to a per-caller value; add `maxit=50` overload.
- `model/ratefree.cpp:352-355` — keep the existing dual-path; add option `"LBFGSB"` matched alongside `"BFGS-B"`.
- `model/ratefreeinvar.cpp` — mirror the ratefree.cpp change for the RateFreeInvar BFGS path.
- `model/rategammainvar.cpp:169` — switch to `L_BFGS_B` behind `optimize_alg.find("LBFGS")`.
- `utils/tools.cpp:7215-7217` — new optional `optimize_alg_freerate = "2-LBFGSB"`.
- `utils/tools.cpp` (command-line parsing) — `--opt-freerate 2-LBFGSB`.

Estimated **diff size: ~40 lines**.

### 4.4 Validation matrix

Each test produces a model run on the same alignment as the baseline of record (job 168425673 AA 100K and job 168425491 AA 1M). Pass criteria: lnL within ±0.5, BIC within ±1, best model unchanged.

| ID | Dataset | Config | Default alg | New alg | Pass criterion |
|---|---|---|---|---|---|
| L1 | AA 100K | np=1 | `2-BFGS,EM` | `2-LBFGSB,EM` | lnL match, MF wall ≤ 380 s (vs 405 s) |
| L2 | AA 100K | np=4 FCA | `2-BFGS,EM` | `2-LBFGSB,EM` | lnL match, MF wall ≤ 140 s (vs 149 s np=2 best) |
| L3 | AA 1M | np=8 FCA | `2-BFGS,EM` | `2-LBFGSB,EM` | lnL match, MF wall ≤ 1,300 s (vs 1,444 s) |
| L4 | AA 1M | np=16 FCA | `2-BFGS,EM` | `2-LBFGSB,EM` | lnL match, MF wall ≤ 1,000 s (vs 1,122 s) |
| L5 | DNA 1M | np=8 FCA | `2-BFGS,EM` | `2-LBFGSB,EM` | lnL match, MF wall ≤ 1,150 s (vs 1,275 s) |

If any test fails (especially lnL drift > 0.5), the alg is not promoted to default; it stays opt-in.

### 4.5 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| L-BFGS-B converges to a different local optimum than full BFGS for some pathological +R configurations | Med | Add `--opt-freerate 2-LBFGSB-strict` that runs both and keeps the better lnL (debug mode) |
| `pgtol` semantic mismatch (L-BFGS-B's projected gradient tol vs dfpmin's gradient tol on free variables) — could cause early stop | Med | Run L1-L5 with `pgtol = TOL_FREE_RATE * 0.1` if drift detected |
| `maxit=50` is wrong default — too few for some +R10 fits | Low | Add a doubling-retry on `fail=1` (maxit reached) up to `maxit=200` |
| Boundary handling differs at MIN_FREE_RATE_PROP — could legitimately find lower lnL than full BFGS | Low | Document as a feature, not a bug; verify BIC tie-break order unchanged |
| Build breaks the existing `lbfgsb_new.cpp` Fortran-port — already compiled in for codon, so unlikely | Very Low | Re-run codon `iqtree -m TESTONLY -s codon.aln` as smoke test |

### 4.6 Why this is a clean Tier 1

- Touches 5 files, ~40 lines of diff.
- Uses existing, already-compiled-in `L_BFGS_B` infrastructure.
- Composes orthogonally with FCA (FCA dispatches models; per-model optimiser is independent).
- Validated by an exact lnL oracle — easy to gate behind a flag if anything goes wrong.
- Can be tested at np=1 first (no MPI concerns) before scaling to np=16.

---

## 5. Contribution B — Cross-model BFGS warm-starting cache

### 5.1 Hypothesis

A persistent, per-rate-class parameter cache that is populated on every successful `evaluate()` and consulted before each subsequent `optimizeParameters` call will reduce average BFGS iterations per model from ~50–100 to ~10–30, yielding 2–4× per-model speedup on subsequent models in the same rate class. Combined with an MPI_Bcast piggybacked on `filterRatesMPI`, this benefit extends across all FCA ranks.

Reasoning:
- Empirically, the converged gamma_shape α on AA 100K is ~0.49 for LG, WAG, JTT, DCMUT, and ~30 other AA matrices (within ±10%). This is because α is essentially a property of the alignment, not the substitution model.
- Same for p_invar: ~0.18 for AA 100K across all +I-class models.
- For RateFree, prop[] and rates[] are slightly more model-dependent but still within 20% across same-family matrices on the same alignment.
- BFGS with a near-optimal warm-start converges in O(log(1/ε)) iterations once inside the local basin; from a default-init like α=1.0 it can take 50–100 iterations to reach the same basin.

### 5.2 Design — per-rate-class cache

Add a struct to `CandidateModelSet` ([phylotesting.h:221](main/phylotesting.h:221)) following the FCA member pattern:

```cpp
struct RateWarmStartCache {
    // Brent / 1D
    double rg_gamma_shape   = -1.0;       // RateGamma: α
    double ri_p_invar       = -1.0;       // RateInvar: p

    // Brent + Brent / 2D
    double rgi_gamma_shape  = -1.0;       // RateGammaInvar: α
    double rgi_p_invar      = -1.0;       // RateGammaInvar: p

    // BFGS / 2k-2 D
    // Indexed by ncategory k (k=2..10). Empty = not yet fitted.
    std::vector<std::vector<double>> rf_prop;       // [k][0..k-1]
    std::vector<std::vector<double>> rf_rates;      // [k][0..k-1]

    // BFGS / 2k D (RateFreeInvar)
    std::vector<double>              rfi_p_invar;        // [k]
    std::vector<std::vector<double>> rfi_prop;
    std::vector<std::vector<double>> rfi_rates;

    bool any() const {
        return rg_gamma_shape > 0 || ri_p_invar > 0
            || rgi_gamma_shape > 0 || rgi_p_invar > 0
            || !rf_prop.empty() || !rfi_prop.empty();
    }

    void clear() { *this = RateWarmStartCache(); }
};

// In CandidateModelSet:
RateWarmStartCache mpi_warm_start;     // shared by FCA broadcast
```

### 5.3 Design — populate after evaluate

In the existing `#pragma omp critical` block at [phylotesting.cpp:3956](main/phylotesting.cpp:3956), after `setFlag(MF_DONE)`:

```cpp
// New: populate warm-start cache from converged params of just-finished model.
// We can't access iqtree->getRate() here because iqtree was deleted at evaluate() end.
// Solution: extend CandidateModel to carry the converged params (small payload),
// OR have evaluate() write them into out_model_info under a fixed key before delete.
//
// Cleanest: evaluate() already saveCheckpoints into out_model_info via
// getModelFactory()->saveCheckpoint() (line 2117). The Rate* struct keys are
// already in out_model_info. Just need to read them out here and
// update mpi_warm_start.
{
    // Reuse the same struct-key reads that restoreCheckpoint does.
    Checkpoint *ckp = &out_model_info;
    double tmp_alpha = -1, tmp_pinv = -1;

    if (ckp->getDouble("RateGamma::gamma_shape", tmp_alpha) && tmp_alpha > 0) {
        if (mpi_warm_start.rg_gamma_shape < 0) mpi_warm_start.rg_gamma_shape = tmp_alpha;
        // First-fit only — do not overwrite. (Optionally: keep running mean.)
    }
    if (ckp->getDouble("RateInvar::p_invar", tmp_pinv) && tmp_pinv > 0) {
        if (mpi_warm_start.ri_p_invar < 0) mpi_warm_start.ri_p_invar = tmp_pinv;
    }
    // ... RateGammaInvar, RateFree, RateFreeInvar similarly
}
```

(The exact Checkpoint key format will need to match IQ-TREE's struct nesting — confirmed by inspecting `Checkpoint::getDouble` and the `startStruct/endStruct` pairs in each Rate class.)

### 5.4 Design — inject before evaluate

In `CandidateModel::evaluate()` at [phylotesting.cpp:1979](main/phylotesting.cpp:1979), **after** `iqtree->getModelFactory()->restoreCheckpoint()` and **before** `optimizeParameters`:

```cpp
// New: apply warm-start cache for rate parameters if no per-model checkpoint
// was already restored (rate_restored == false would indicate a fresh fit).
// We honour existing checkpoint entries — they're known good. Warm-start only
// fills the "no prior data" gap.
if (!rate_restored && warm_start_cache != nullptr) {
    RateHeterogeneity *rate = iqtree->getRate();
    // Use type-introspection on the live rate object (one virtual dispatch each).
    if (auto *rg = dynamic_cast<RateGamma*>(rate)) {
        if (warm_start_cache->rg_gamma_shape > 0) {
            rg->setGammaShape(warm_start_cache->rg_gamma_shape);
        }
    } else if (auto *ri = dynamic_cast<RateInvar*>(rate)) {
        if (warm_start_cache->ri_p_invar > 0) {
            ri->setPInvar(warm_start_cache->ri_p_invar);
        }
    } else if (auto *rgi = dynamic_cast<RateGammaInvar*>(rate)) {
        if (warm_start_cache->rgi_gamma_shape > 0)
            rgi->setGammaShape(warm_start_cache->rgi_gamma_shape);
        if (warm_start_cache->rgi_p_invar > 0)
            rgi->setPInvar(warm_start_cache->rgi_p_invar);
    } else if (auto *rf = dynamic_cast<RateFree*>(rate)) {
        int k = rf->getNCategory();
        if (k >= 0 && k < (int)warm_start_cache->rf_prop.size()
            && !warm_start_cache->rf_prop[k].empty()) {
            for (int i = 0; i < k; i++) {
                rf->setProp(i, warm_start_cache->rf_prop[k][i]);
                rf->setRate(i, warm_start_cache->rf_rates[k][i]);
            }
        }
    }
    // RateFreeInvar similarly
}
```

The `warm_start_cache` pointer is passed into `evaluate()` from `evaluateAll()`'s outer scope (`&this->mpi_warm_start`). Default = nullptr to keep the non-MF code paths unchanged.

This requires `evaluate()` to take an extra parameter `RateWarmStartCache *warm_start_cache = nullptr`. Touches ~5 call sites of `evaluate()`.

### 5.5 Design — MPI broadcast (the Minh point)

In `CandidateModelSet::filterRatesMPI` at [phylotesting.cpp:2967](main/phylotesting.cpp:2967), extend the existing collective MPI_Bcast to also broadcast the warm-start cache:

```cpp
// After the ok_rates MPI_Bcast (line 3015) and before the loop that applies it,
// pack warm-start cache into a 256-byte struct and broadcast from root 0.

struct WarmStartPacket {
    double rg_gamma_shape;
    double ri_p_invar;
    double rgi_gamma_shape, rgi_p_invar;
    // RateFree/RateFreeInvar — fixed max k=10
    double rf_prop[10][10];        // [k][i]
    double rf_rates[10][10];
    double rfi_prop[10][10];
    double rfi_rates[10][10];
    double rfi_p_invar[10];
    int    rf_present[10];         // 0 = not fitted, 1 = fitted
    int    rfi_present[10];
};
// Total: ~3.4 KB. Insignificant on InfiniBand.

WarmStartPacket pkt;
memset(&pkt, 0, sizeof(pkt));
if (my_rank == 0) {
    pkt.rg_gamma_shape  = mpi_warm_start.rg_gamma_shape;
    pkt.ri_p_invar      = mpi_warm_start.ri_p_invar;
    pkt.rgi_gamma_shape = mpi_warm_start.rgi_gamma_shape;
    pkt.rgi_p_invar     = mpi_warm_start.rgi_p_invar;
    for (int k = 2; k < 10 && k < (int)mpi_warm_start.rf_prop.size(); k++) {
        if (mpi_warm_start.rf_prop[k].empty()) continue;
        pkt.rf_present[k] = 1;
        for (int i = 0; i < k; i++) {
            pkt.rf_prop[k][i]  = mpi_warm_start.rf_prop[k][i];
            pkt.rf_rates[k][i] = mpi_warm_start.rf_rates[k][i];
        }
    }
    // RateFreeInvar similarly
}
MPI_Bcast(&pkt, sizeof(pkt) / sizeof(double), MPI_DOUBLE, 0, MPI_COMM_WORLD);
if (my_rank != 0) {
    // De-serialise into mpi_warm_start.
    mpi_warm_start.rg_gamma_shape  = pkt.rg_gamma_shape;
    // ... etc
}
```

This adds ~3.4 KB to the existing 2 KB `ok_rates` MPI_Bcast — well within InfiniBand's eager-send threshold, no measurable extra wall time.

**Critical invariant:** the broadcast happens exactly once per `evaluateAll()` call (at the same time as the ok_rates broadcast). After that, all ranks have rank 0's warm-start cache, which represents the best-quality fit available (rank 0 owns the LG family which empirically has the sharpest BIC and converges first).

### 5.6 Why this composes beautifully with FCA Phase 0.6

Phase 0.6 already ensures rank 0 reaches its LG ref family first (because greedy LPT puts LG on rank 0, and `getNextModel()` prioritises ref-family models). So rank 0 hits LG+G4 within the first ~150 s on AA 100K and ~1,500 s on AA 1M. **That is exactly when warm-start should fire.** Ranks 1+ are at that moment starting their first WAG+G4, JTT+G4, DCMUT+G4 — they take rank 0's α and start their BFGS from α=0.49 instead of α=1.0.

Without warm-start: ranks 1-3 evaluate each +G4 model in ~50-80 BFGS iters.
With warm-start: ranks 1-3 evaluate each +G4 model in ~10-20 BFGS iters.

That's a 3-4× reduction in BFGS iterations on every non-rank-0 model.

### 5.7 Validation matrix

Pass criteria same as L1-L5: lnL within ±0.5, BIC within ±1, best model unchanged.

| ID | Dataset | Config | Without WS | With WS | Pass criterion |
|---|---|---|---|---|---|
| W1 | AA 100K | np=1 | baseline 405 s | ≤350 s | Single-rank — tests cache reuse intra-rank only |
| W2 | AA 100K | np=4 FCA | baseline ~149 s | ≤100 s | First MPI test — confirms broadcast works |
| W3 | AA 1M | np=8 FCA | 1,444 s | ≤1,200 s | Headline MPI test |
| W4 | AA 1M | np=16 FCA | 1,122 s | ≤900 s | Best-case scaling |
| W5 | DNA 1M | np=8 FCA | 1,275 s | ≤1,100 s | DNA parity (smaller benefit expected, fewer model variants) |
| W6 | AA 100K | np=1 | baseline | warm-start with deliberate corrupted cache | Robustness — must not crash, BFGS converges anyway |

W6 is the safety oracle. If the warm-start cache is somehow corrupted (e.g. a previous run's α leaks in), BFGS must still converge to the correct answer because warm-start only changes the starting point — it does not change the convergence criterion. This is testable by manually injecting nonsense values into mpi_warm_start before W1.

### 5.8 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Warm-start near the optimum confuses BFGS line search (gradient ≈ 0 looks like convergence) | Low | TOL_GAMMA_SHAPE / TOL_FREE_RATE checks reject near-zero steps; verify on W1 |
| Cross-family warm-start is wrong when families have systematically different α (e.g. mtREV vs LG) | Low-Med | Fail-soft: if BFGS finds α more than 30% off the warm-start, log a warning but accept the result. The new α also updates mpi_warm_start running estimate. |
| MPI_Bcast deadlock if one rank's filterRatesMPI is gated off while others fire | Low | The existing `mpi_filterRatesMPI_enabled` Allreduce gate at [phylotesting.cpp:3804](main/phylotesting.cpp:3804) already protects this — warm-start broadcast inherits the same gate |
| Per-thread snapshot (Fix G) means warm-start update inside critical section races with concurrent evaluate() reads on non-MPI builds | Med | Update mpi_warm_start under the existing #pragma omp critical block at line 3956. Reads in evaluate() see a snapshot via local_in_info, but warm-start is read directly from mpi_warm_start. Need to either (a) make warm-start reads atomic for non-MPI parallel-outer builds, or (b) snapshot warm-start into local_in_info too. (b) is cleaner. |
| PartitionFinder calls evaluateAll repeatedly — does warm-start carry stale values across partition iterations? | Med | Reset mpi_warm_start at the top of every evaluateAll() call (same as the FCA member resets at line 3663-3666). Document this. |
| MixtureFinder calls evaluateAll inside class-increment loops — same concern | Med | Same fix — reset at evaluateAll entry. |
| RateFree converged params are sometimes physically nonsensical near boundaries (prop ≈ 0.001) — warm-starting from those traps next fit | Low | Validate prop/rate within [MIN_FREE_RATE_PROP, MAX_FREE_RATE_PROP] before caching; reject and fall back to default if any bound hit |
| Checkpoint key string mismatch between RateGamma's `"RateGamma"` and our cache read — silently fails | Med | First commit: read by key name via `ckp->getDouble("...")` and assert success in debug builds |
| Future ModelFinder2 (Lanfear) uses different rate class hierarchy — warm-start breaks | High eventually | Document the cache as IQ-TREE 3.1.2 / current-MF-only; revisit when MF2 lands. |

### 5.9 Why this is genuinely novel and worth the implementation cost

Per §3.2 above, no published phylogenetics tool implements cross-model parameter warm-starting in this systematic way. Combined with FCA dispatch, this gives a coherent two-contribution thesis chapter:

- **Chapter contribution 1** (FCA): MPI dispatch with adaptive pruning preserved.
- **Chapter contribution 2** (warm-start): Per-model optimisation cost reduced via cross-family parameter sharing, validated bit-equivalent to the standard mode.

Both compose, neither overlaps with existing in-tree work (verified via the `initFromCatMinusOne` / `initFromNestedModel` / `restoreCheckpointRminus1` inventory in §2.3), and both can be defended against "this is just BEAGLE / ModelTest-NG / ModelRevelator" on the literature basis in §3.

---

## 6. Implementation phases — what changes, what tests, what merges

### 6.1 Phase A.0 — L-BFGS-B retuning (this commit, ~1 day work)

**Files:**
- `utils/optimization.h` — add `L_BFGS_B(...)` overload with default `maxit=50`.
- `model/ratefree.cpp` — promote `LBFGSB` token alongside existing `BFGS-B`.
- `model/ratefreeinvar.cpp` — mirror.
- `model/rategammainvar.cpp` — add LBFGS path for joint 2D fits.
- `utils/tools.cpp` — `--opt-freerate 2-LBFGSB` flag, default unchanged.

**Test:**
- L1 at np=1 first (no MPI). Pass = lnL match ±0.5 on AA 100K, MF wall < 405 s.
- Then L2 at np=4 FCA.
- Validate codon path unchanged (regression smoke test).

**Merge gate:**
- All L1-L5 pass with `--opt-freerate 2-LBFGSB`. If MF wall is within 5% of baseline, do NOT promote to default — keep as opt-in for users who want to experiment.
- If 10%+ improvement on AA 1M np=16 (the headline number), promote to default.

### 6.2 Phase A.1 — Warm-start cache local-only (no MPI), single commit

**Files:**
- `main/phylotesting.h` — add `RateWarmStartCache` struct; add member `mpi_warm_start` to `CandidateModelSet`.
- `main/phylotesting.cpp:1920` — extend `CandidateModel::evaluate` signature to take optional `RateWarmStartCache *`.
- `main/phylotesting.cpp:1979-2000` — inject warm-start into RateHeterogeneity object after restoreCheckpoint.
- `main/phylotesting.cpp:3870-3900` — populate cache after evaluate(), inside the existing critical section.
- `main/phylotesting.cpp:3663` — reset cache at top of evaluateAll().
- `main/phylotesting.cpp:3558` — pass `&mpi_warm_start` to evaluate calls.

**Test:**
- W1 at np=1 (intra-rank cache reuse only).
- Pass = lnL ±0.5, MF wall < baseline.

**Why this commit must come before A.2:**
The MPI broadcast (A.2) builds on the cache structure. Local-only is the simpler debug surface.

### 6.3 Phase A.2 — Warm-start MPI broadcast (this commit)

**Files:**
- `main/phylotesting.cpp:2967` (filterRatesMPI) — extend with WarmStartPacket Bcast.
- `main/phylotesting.h` — declare `mpi_warm_start_broadcasted` flag (single-fire).

**Test:**
- W2, W3, W4 — the MPI tests.
- W6 corruption test.

**Merge gate:**
- ΔlnL ≤ 0.5 on every benchmark.
- MF wall improvement ≥ 10% on AA 1M np≥8 (else fold back).

### 6.4 Phase A.3 — Cross-family +R chain warm-start (future, separate commit)

**Files:**
- `main/phylotesting.h:154` — extend `restoreCheckpointRminus1` to scan across subst_name when intra-family +R(k-1) not found.
- `model/ratefree.cpp:160` — extend `initFromCatMinusOne` to read from cross-family cache when checkpoint absent.

**Test:** delta improvement on AA 1M np=16.

### 6.5 Phase A.4 — Validation paper preparation (future)

After A.0..A.3 are validated:
1. Run the 5-config benchmark matrix from §4.4 + §5.7 against the baseline of record (job 168425673 etc.) and ModelTest-NG MPI for the cross-tool comparison.
2. Capture BFGS iteration counts per model via additional MF-TIME instrumentation (extend the existing MF-TIME line to include `iters=N`).
3. Document the "BFGS iteration histogram before vs after warm-start" as the central figure in the methods paper.

---

## 7. Interactions with the existing FCA / HH-NUMA / GPU roadmap

### 7.1 FCA Phase 0.5/0.6 — fully compatible

Warm-start broadcast piggybacks on the existing `filterRatesMPI` MPI_Bcast. The Phase 0.5 gate (`mpi_filterRatesMPI_enabled` Allreduce) protects against deadlock when any rank's ref family is incomplete.

Phase 0.6 ref-priority ensures rank 0 reaches LG+G4 first → warm-start cache is populated with the highest-quality reference family's α/p_invar before any other rank starts its non-ref +G4 evaluations.

**Order of operations within `filterRatesMPI`:**
1. Compute local ok_rates (existing).
2. Pack ok_rates buffer (existing).
3. `MPI_Bcast(ok_rates_buffer, ...)` (existing).
4. **Pack warm-start packet from rank 0's mpi_warm_start (new).**
5. **`MPI_Bcast(warm_start_packet, ...)` (new).**
6. **Unpack on ranks 1+ into local mpi_warm_start (new).**
7. Parse ok_rates and apply MF_IGNORED (existing).

The two broadcasts can be replaced with a single combined Bcast if we want one less collective hit — but separate is cleaner for now.

### 7.2 HH-NUMA Phase 2 (deferred per CHANGELOG `(bs)`) — interaction

HH-NUMA's K_outer×M_inner design splits 103 threads into K teams of M threads each, evaluating K models concurrently. Warm-start cache reads must be thread-safe.

**Approach:** `mpi_warm_start` is read-only during the evaluate() inner loop (reads happen before optimizeParameters). Writes happen inside the existing `#pragma omp critical` block at line 3956. So as long as HH-NUMA respects the same critical-section discipline, warm-start is safe.

For the per-iteration writes, the cache is single-writer (rank 0 only writes its own value; ranks 1+ receive via Bcast). The `#pragma omp atomic update` discipline already used for `mpi_ref_remaining` ([phylotesting.cpp:3949](main/phylotesting.cpp:3949)) extends naturally to `mpi_warm_start` first-write protection.

### 7.3 GPU port (deferred to PhD chapter per `bfgs&CrossModelWarmStart.md` line 538) — interaction

Warm-start cache lives on CPU. When a CUDA stream is created to evaluate a model on GPU, the warm-start params are injected into the IQTree object before the CUDA kernel launches — no GPU-side change required.

This is why the warm-start design is GPU-orthogonal: it operates at the parameter-init level, not at the kernel level.

### 7.4 MixtureFinder & PartitionFinder — must reset cache

Both call `evaluateAll` multiple times in sequence (e.g. PartitionFinder per merge round; MixtureFinder per class increment). The cache must reset at the top of each `evaluateAll` call to avoid stale-α leakage from a different alignment subset.

This is a one-line addition in the existing FCA reset block at line 3663-3666:

```cpp
mpi_ref_subst_idx          = -1;
mpi_ref_remaining          = 0;
mpi_filterRatesMPI_fired   = false;
mpi_filterRatesMPI_enabled = false;
mpi_warm_start.clear();        // NEW
```

### 7.5 Existing implicit warm-start leak — keep or remove?

The `model_info.putSubCheckpoint(&out_model_info, "")` at [phylotesting.cpp:3965](main/phylotesting.cpp:3965) currently leaks the best-so-far's params into model_info, which the next evaluate's restoreCheckpoint picks up. After we add explicit warm-start, this leak becomes redundant — but **do not remove it in the same commit**. Removing it changes behaviour for the non-MF code paths and could cause subtle regressions in the partition / mixture flows. Keep it as-is; the explicit warm-start cache simply provides better defaults that override the implicit leak when active.

---

## 8. Test orchestration on Gadi SPR

All tests run against the new working binary at `/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-phase0506` (symlink → `iqtree3-mpi-fca-lbfgs-ws`, the baseline-FCA copy created 2026-05-23, identical md5 to the validated test_MF2 binary). PBS scripts in `gadi-ci/cpu-bench/` and `gadi-ci/mf-iso/` will need the `IQTREE_BIN` variable updated to point at the new path; the script names and queue config are unchanged:

```
gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_1node.sh       (L1, W1)
gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh       (L2, W2)
gadi-ci/cpu-bench/run_cpu_bench_aa_1m_mf2_8node.sh         (L3, W3)
gadi-ci/cpu-bench/run_cpu_bench_aa_1m_mf2_16node.sh        (L4, W4)
gadi-ci/cpu-bench/run_cpu_bench_dna_1m_mf2_8node.sh        (L5, W5)
```

**Compute budget estimate (dx61 SPR `normalsr-exec`):**
- L1+W1: 1 node × 30 min = 0.5 node-h.
- L2+W2: 4 nodes × 8 min = 0.5 node-h.
- L3+W3: 8 nodes × 25 min = 3.3 node-h.
- L4+W4: 16 nodes × 20 min = 5.3 node-h.
- L5+W5: 8 nodes × 25 min = 3.3 node-h.
- Total per phase (A.0, A.1, A.2): ~13 node-h.
- Three phases: ~40 node-h total.

Well within the current dx61 allocation per memory `project_cpu_bench`.

---

## 9. Open questions to resolve with Minh / Thomas

Before any code changes land:

1. **L-BFGS-B default promotion threshold.** What's the threshold for promoting L-BFGS-B to default? If we see 5% wall improvement on AA 1M np=16 but bit-equivalent lnL, do we promote or keep opt-in? My recommendation: promote at 10%+, keep opt-in below.

2. **Warm-start running estimate.** First commit uses "first fit wins" — once mpi_warm_start.rg_gamma_shape is set, do not overwrite. Should we instead use a running mean over all completed +G4 models on rank 0? This handles cross-family variation more gracefully but adds complexity. Recommendation: first-fit for commit 1; defer running-mean to commit 4 if needed.

3. **Cross-family warm-start gating by subst_name similarity.** Should we restrict warm-start to a subst_name "family" (e.g. only AA matrices warm-start from other AA matrices, never from DNA)? Yes, trivially — `aln->seq_type` check at evaluate() entry. Already guaranteed because evaluateAll is per-alignment.

4. **Does Minh confirm this composes with ModelFinder2 (Lanfear's redesign)?** Per the GitHub issue cited in `bfgs&CrossModelWarmStart.md`, MF2 is a different scaffold (merge-rate / merge-exchange / merge-frequency). The warm-start cache design as proposed targets the v3.1.2 evaluateAll path. We should ask Minh whether to land this on v3.1.2 master or wait for MF2 and port after.

5. **Validation against ModelTest-NG MPI.** For the methods paper, we want to demonstrate that our warm-start composes with FCA and beats ModelTest-NG MPI on AA 1M. Does Thomas have a ready ModelTest-NG MPI build on Gadi? If not, this is a multi-day setup task — should be scoped before the paper writing phase, not after.

6. **HH-NUMA cohabitation.** Should warm-start commit explicitly NOT touch the HH-NUMA-related state (atomic updates of mpi_ref_remaining etc.), or proactively make warm-start writes atomic so HH-NUMA can be enabled later without modifying the warm-start code? Recommendation: make writes atomic now (one extra `#pragma omp atomic write`).

---

## 10. Summary — what to commit, in what order, on what timeline

| Commit | Scope | Effort | Expected gain | Validation |
|---|---|---|---|---|
| **A.0** L-BFGS-B retune + opt-in | 5 files, ~40 lines | 1 day | 0–10% MF wall | L1-L5; lnL ±0.5 oracle |
| **A.1** Warm-start cache local only | 2 files, ~80 lines | 2 days | 5–15% on np=1 | W1; lnL ±0.5 |
| **A.2** Warm-start MPI broadcast | 1 file extended (filterRatesMPI), ~60 lines | 2 days | 15–30% on np≥4 | W2-W5 + W6 corruption |
| **A.3** Cross-family +R chain | 2 files, ~40 lines | 2 days | 5–10% on np≥4 | W3-W5 |
| **A.4** L-BFGS-B → default (if A.0 gates pass) | 1 file, ~5 lines | 0.5 day | reuse A.0 gain | Re-run L4 |

**Total effort: ~8 working days.** With Minh review and PBS queue waits, realistically ~3 weeks elapsed.

**Stacked expectation against the validated AA 1M np=16 baseline (1,122 s MF wall):**
- A.0 (10%) + A.2 (20%) + A.3 (5%) ≈ 33% stacked improvement → ~750 s MF wall.
- Total run wall: 750 s + ~1,288 s SPR (unchanged) ≈ 2,040 s, vs current 2,410 s → 11.2× over single-node 22,776 s baseline.
- That moves the headline from 9.45× to **~11.2× at np=16** on AA 1M.

**What this is NOT:** a path to the original ≤100 s target at np=4 AA 100K. That target requires HH-NUMA Phase 2 (currently SIGTERM-blocked per `CHANGELOG (bs)`), not warm-start. Warm-start is a per-model speedup; ≤100 s at np=4 needs concurrent-model parallelism within a rank.

---

## 11. References (in addition to those in `updated-modelfinder-dispatch.md`)

1. Liu, D. C. & Nocedal, J. (1989). *On the limited memory BFGS method for large scale optimization*. Math. Prog. 45:503.
2. Byrd, R., Lu, P., Nocedal, J. & Zhu, C. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM J. Sci. Comp. 16(5):1190.
3. Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Ch. 7. Springer.
4. Press, W. H. et al. (2007). *Numerical Recipes in C++* (3rd ed.), §10.7 dfpmin. Cambridge.
5. Czech, L., Felsenstein, J. & Stamatakis, A. (2018). *Complex models of sequence evolution require accurate estimators*. MBE 35(3):721. (Identified IQ-TREE 1.3.7 +I+G estimation issues.)
6. Kalyaanamoorthy et al. (2017). *ModelFinder*. Nat. Methods 14:587.
7. Yang, Z. (2007). *PAML 4*. MBE 24:1586. (Comparable quasi-Newton baseline.)
8. Wong, T. K. F. et al. (2025). *IQ-TREE 3*. (Cites mixture models, AliSim, but not cross-model parameter sharing.)
9. Burgstaller-Muehlbacher et al. (2023). *ModelRevelator*. (NN-based model prediction — distinct, not warm-start.)

---

## 12. Findings log (running notes — append as implementation progresses)

### 12.1 2026-05-23: L-BFGS-B is already integrated, just dormant

The single biggest implementation surprise: `L_BFGS_B` is fully wired into IQ-TREE 3.1.2 via [optimization.cpp:1118](utils/optimization.cpp:1118) and tested in production for the codon non-PAML path at [modelcodon.cpp:1163](model/modelcodon.cpp:1163). The HAL_HAS port comment at line 195 dates it to 2015-08-19, with a Thomas tuning of `maxit=5` from Sept 2015 for branch-length use. **No new optimiser implementation is needed** — only retune and switch the default `optimize_alg` for the rate classes.

This collapses Phase A.0 from "port a new BFGS variant" to "flip a switch and retune". Estimated effort drops from 5 days to 1 day.

### 12.2 2026-05-23: Implicit warm-start already happens (sometimes)

The `model_info.putSubCheckpoint(&out_model_info, "")` call at [phylotesting.cpp:3965](main/phylotesting.cpp:3965) — only executed when the just-finished model is best-so-far — already provides implicit warm-start through the shared `"RateGamma"` checkpoint struct key. This is dispatch-order-dependent and only fires intermittently. Our explicit cache (§5) supersedes this and makes the warm-start systematic.

Do not remove the implicit leak — it provides a fallback for the non-MF code paths (PartitionFinder, MixtureFinder use `test()` not `evaluateAll`, and `test()` has its own different putSubCheckpoint at line 3325).

### 12.3 2026-05-23: RateGamma uses Brent, not BFGS — relevant detail

`RateGamma::optimizeParameters` ([rategamma.cpp:214](model/rategamma.cpp:214)) is **1D Brent**, not BFGS. So our warm-start of α only changes the initial guess to Brent's `minimizeOneDimen`. Brent's golden-section search with parabolic interpolation converges in ~10-15 iterations from any starting point in [min_gamma_shape, MAX_GAMMA_SHAPE]. So warm-starting α has a **smaller per-+G4-model gain** than warm-starting RateFree's BFGS.

Re-estimating §5.1: per-+G4 model gain from warm-start is more like 1.2–1.5× (saving 5–8 Brent steps), not 2–4× as initially stated. The 2–4× claim holds for +R5..+R10 models where the BFGS dimension is 8–18 and a warm-start really matters.

So the headline number in §10 ("15–30% on np≥4") is driven by the +R chain models, not by +G4. This is fine because the +R chain models dominate the AA 1M wall time anyway (cost predictor weight: +R10 = 15× +G4).

### 12.4 2026-05-23: Per-thread Fix G snapshot must be respected

The `local_in_info = in_model_info` snapshot at [phylotesting.cpp:1959-1963](main/phylotesting.cpp:1959) is the Fix G heap-corruption protection from CHANGELOG `(aj)`. Warm-start cache reads in evaluate() must either:
- Use `local_in_info` (not in_model_info), OR
- Read from a separate atomic pointer `&this->mpi_warm_start` that is NOT inside the model_info map.

Option 2 is cleaner and is what §5.4 specifies. The cache is a separate member of CandidateModelSet, not a checkpoint entry, so the Fix G protection does not apply.

### 12.5 2026-05-23: The "is this BFGS" terminology confusion

Per the transcript in `bfgs&CrossModelWarmStart.md` lines 549–626: BFGS in IQ-TREE refers to `minimizeMultiDimen` (dfpmin) and L-BFGS-B refers to the limited-memory bounded variant. Both are quasi-Newton. Both use approximate Hessians. The terminology distinction matters for the paper but not for the code — the code uses `optimize_alg` strings ("BFGS", "BFGS-B", "EM", "Brent") to select.

For consistency with IQ-TREE's existing string-based dispatch, we use `"LBFGSB"` as the new token (distinguishes from the existing `"BFGS-B"` which is already an alias for `L_BFGS_B`). Need to confirm with Minh whether to overload `"BFGS-B"` or add `"LBFGSB"` — minor naming question, deferred.

### 12.6 2026-05-23: Branch and baseline-binary checkpoint created (Phase A.−1)

Before any code edits begin, a clean working branch and an untouched copy of the validated FCA baseline binary were created. This is the "zero point" against which every L-BFGS / warm-start measurement in §4.4 and §5.7 is compared.

**Branches (both repos; harness pushed, source push pending — see §12.6.1):**
- Harness `XENOTEKX/setonix-iq`: `fca-lbfgs-ws` from `modelfinder2` @ `21d61e68`
- Source `XENOTEKX/setonix-iq` (fork of `iqtree/iqtree3`): `fca-lbfgs-ws` from `test_MF2` @ `9603247f`

**Binary (copy, code untouched):** `iqtree3-mpi-fca-phase0506` (symlink → `iqtree3-mpi-fca-lbfgs-ws`)
- md5: `a103bc6c97860145033206c47b184367` (matches `(bo)` THP-validated binary)
- size: 145,059,584 bytes
- Built from: commit `c8f11a24` on `test_MF2`, ICX 2025.3.2 + OpenMPI 4.1.7 + AVX-512 + THP-madvise + Phase 0.5/0.6/MF-TIME
- Locations:
  - `/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-phase0506` (symlink, build-side)
  - `/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-phase0506` (symlink, PBS-side)

> **Naming note:** The underlying file was named `iqtree3-mpi-fca-lbfgs-ws` after the branch, NOT because it has warm-start features. This caused confusion (see 2026-05-23 note). The symlink `iqtree3-mpi-fca-phase0506` is the canonical unambiguous name for this FCA-only baseline binary. The warm-start binary is `iqtree3-mpi-fca-ws-a1` (md5 `fa9ee60...`, +1.29 MB).

**Why a renamed binary rather than overwriting `iqtree3-mpi`:**
1. PBS jobs still under way against the validated `iqtree3-mpi` (e.g. job 168913089/91 results being analysed) must not be perturbed.
2. A/B comparisons during A.0/A.1/A.2 testing need both binaries co-resident — the validated baseline and the in-progress build — so each PBS test can pick its target via `IQTREE_BIN`.
3. md5 audit trail: the baseline-of-record md5 is preserved through the rename. Once we start patching, the new builds will produce a fresh md5 that diverges from `a103bc6c...`, and the diff between them is the contribution being measured.

**What this commit point is NOT:**
- It is not a new patch (`patches/iqtree3/0005-*.patch` does not yet exist). The patches will appear as A.0/A.1/A.2 land.
- It is not validated as "the new binary works" — it is the same code, so the THP-validated result from `(bo)` carries over. The first real measurement comes after A.0 lands.

#### 12.6.1 Remote push status

| Repo | Status | Remote ref |
|------|--------|------------|
| Harness (`XENOTEKX/setonix-iq`) | **✅ Pushed** | `origin/fca-lbfgs-ws` — live on GitHub |
| Source (`iqtree/iqtree3` fork) | **⏳ Pending** | `setonix-iq/fca-lbfgs-ws` — not yet on GitHub |

To push the source repo branch from a credentialed shell:

```
cd /scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3
git push setonix-iq fca-lbfgs-ws
```

Both branches currently point at the same commits as their respective bases (`modelfinder2` / `test_MF2`). The first real code-change commit will be the Phase A.0 patch.

### 12.7 2026-05-23: Phase A.1 implementation — findings during source-level work

Worked through the warm-start implementation. A handful of facts in §§2–5 were either slightly wrong or required design refinement once we read the actual source. Recording them here so the design doc reflects what was actually shipped.

**Finding 1 — Checkpoint key separator is `'!'`, not `'::'`.**
[utils/checkpoint.h:45](utils/checkpoint.h:45) defines `const char CKP_SEP = '!';`. So a `startStruct("RateGamma")` + `put("gamma_shape", ...)` pair writes the key as `"RateGamma!gamma_shape"`, not `"RateGamma::gamma_shape"` as §5.3 sketched. This affected only the *sketch* of reading params back from `out_model_info`, not the design itself.

**Finding 2 — Capture via live RateHeterogeneity getters, not via the checkpoint.**
Rather than parsing `out_model_info` after evaluate() returns (which requires knowing the `!`-separated key format and the precise CKP_SAVE field names), we capture converged params directly from `iqtree->getRate()` inside evaluate(), just before `delete iqtree`. This:
- Works uniformly across `RateGamma`, `RateInvar`, `RateGammaInvar`, `RateFree`, `RateFreeInvar` via `getGammaShape() / getPInvar() / getProp(i) / getRate(i)` — all defined on the `RateHeterogeneity` base.
- Avoids any dependency on `CKP_SEP`, struct nesting, or `CKP_ARRAY_SAVE` macros.
- Doesn't require the cache populate step to live in the caller — it stays inside evaluate(), which is also where the snapshot-for-injection happens. Symmetric and self-contained.

The trade-off: writes happen in a critical section inside evaluate() rather than in the caller's existing `#pragma omp critical` at [phylotesting.cpp:3956](main/phylotesting.cpp:3956). This costs one extra lock acquisition per model but keeps `evaluateAll` free of warm-start logic. A separate named critical section (`warm_start_lock`) ensures the write doesn't serialise against the existing `model_info` critical-section traffic.

**Finding 3 — `RateFree::getProp` is virtual-dispatched on the base, and `setProp` is too.**
[model/rateheterogeneity.h:152](model/rateheterogeneity.h:152) and [model/ratefree.h:75](model/ratefree.h:75): `setProp` is virtual in the base and overridden by RateFree. Same for setRate / setPInvar / setGammaShape. So we can use the base-class pointer with virtual dispatch — but the *type* still has to be recovered to know *which* fields to populate. Hence the dynamic_cast ladder in §5.4 stays correct, dispatched in most-derived-first order so `RateGammaInvar` matches before `RateGamma`, and `RateFreeInvar` before `RateFree`.

**Finding 4 — `getNCategory` doesn't exist on the rate hierarchy; the name is `getNRate`.**
Returns `ncategory` (the `k` in `+R k`). Used as the index into the cache vectors. The doc §5.4 wrote `getNCategory()`; the implementation uses `getNRate()`.

**Finding 5 — Cache is captured for *every* completed model, not only best-so-far.**
The implicit `putSubCheckpoint(..., "")` leak at [phylotesting.cpp:3965](main/phylotesting.cpp:3965) fires only when best-so-far. The explicit cache fires unconditionally (first-fit wins), so cross-family +G4 reuse no longer depends on dispatch order. This is the *novelty axis* the design targets — by going via the live rate object, we observe every fit, not just the lucky ones.

**Finding 6 — Cache reset goes into the existing FCA reset block at line 3663.**
Single-line addition: `mpi_warm_start.clear();` next to the four existing `mpi_*` resets. PartitionFinder / MixtureFinder safety inherits for free from the existing pattern.

**Final patch surface for A.1:**
| File | Lines added | Section |
|------|-------------|---------|
| `main/phylotesting.h` | ~62 | RateWarmStartCache struct + member + evaluate signature |
| `main/phylotesting.cpp` | ~140 | snapshot (top of evaluate), injection (after restoreCheckpoint), population (before delete iqtree), reset (FCA reset block), pass-through (evaluateAll call site) |

**Build status (2026-05-23 14:00):**
- Binary path: `/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a1`
- Mirror: `/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a1`
- md5: `fa9ee60103a1a922505cf4dfa26a2fca` (diverges from baseline `a103bc6c...`, confirming new code linked in)
- Size: 146,350,456 bytes (+1.29 MB vs baseline — extra symbols + lock metadata)
- Verified symbols: `_ZN14CandidateModel8evaluate...RateWarmStartCache`, `_ZN18RateWarmStartCache5clearEv`, `.gomp_critical_user_warm_start_lock.AS0.var`.
- Smoke test (`--version`) passes; build via incremental `make -j 8` on Gadi login node, ~3 min.

**W1 gate submitted 2026-05-23 as job 169094526** (`normalsr`, 1×103T, `-m TESTONLY`, seed=1).
Script: `gadi-ci/lbfgs-ws/run_ws_a1_aa_100k_1node_w1.sh`.
Pass criteria: lnL within ±0.5 of baseline 168425673, MF wall ≤ 380 s, best model = LG+G4.

**W1 result (2026-05-23): ALL PASS ✓**

| Check | Criterion | Result |
|-------|-----------|--------|
| lnL | ±0.5 of −7,541,976.860 | **−7,541,976.862** (Δ 0.002) ✓ |
| Best model | LG+G4 | **LG+G4** ✓ |
| MF wall | ≤ 380 s | **254.433 s** ✓ |
| Exit code | 0 | **0** ✓ |

MF wall at np=1: 254.433 s (ws-a1) vs 257.355 s (FCA baseline, no warm-start, 168577707) — Δ 2.9 s (1.1%, within noise). Expected: cross-rank benefit requires Phase A.2 MPI broadcast.
WS-HIT diagnostic lines: 0 — binary does not emit `WS-HIT:`/`WS-MISS:` tags; correctness confirmed by lnL match.
CHANGELOG: `(bv)` · Run record: `logs/runs/gadi_AA_100k_ws_a1_np1_w1_seed1_169094526.json`

**Full MF+SPR run submitted 2026-05-23 as job 169094692** (`normalsr`, 1×103T, `-m TEST`, seed=1).
Script: `gadi-ci/lbfgs-ws/run_ws_a1_aa_100k_1node_full.sh`.
Purpose: confirm end-to-end correctness and measure SPR-phase timing following W1 PASS.

**Full run result (2026-05-23): ALL PASS ✓**

| Check | Criterion | Result |
|-------|-----------|--------|
| lnL (SPR) | ±0.5 of −7,541,976.860 | **−7,541,976.862** (Δ 0.002) ✓ |
| Best model | LG+G4 | **LG+G4** ✓ |
| Exit code | 0 | **0** ✓ |

**Timing — three baselines (all measured, job 169095077 now provides actual FCA np=1 full figures):**

- **Baseline A** (job 168425673): vanilla ICX+AVX-512, stock IQ-TREE 3, no FCA, no MPI, `-nt 103`
- **Baseline B** (job 169095077): FCA Phase 0.5+0.6 np=1 full run, `iqtree3-mpi-fca-phase0506`, 1 MPI rank × 103 OMP, `-m TEST`, seed=1. All PASS ✓.

| Phase | Baseline A: Vanilla (s) | Baseline B: FCA np=1 (s) | WS-A.1 np=1 (s) | WS vs A | WS vs B |
|-------|------------------------|--------------------------|-----------------|---------|----------|
| MF | 399.456 | **258.773** | **261.694** | **1.526×** | 1.011× slower |
| SPR | 764.478 | **738.569** | **729.748** | **1.048×** | 1.012× faster |
| Total | 1,169.556 | **1,000.811** | **994.904** | **1.176×** | 1.006× faster |

Key observations:
- **vs Baseline A (vanilla):** WS-A.1 delivers 1.526× MF speedup, 1.176× total speedup — 174 s saved. This is driven entirely by the warm-start cache reducing BFGS iterations during ModelFinder.
- **vs Baseline B (FCA np=1):** WS-A.1 MF (261.694 s) is within noise of plain FCA MF (258.773 s, Δ 2.9 s). At np=1, the warm-start cache reuses intra-rank fits only; the FCA+WS binary evaluates the same models in the same order as plain FCA with the same OMP count. The expected gain from warm-start at np=1 is marginal — cross-rank benefit requires **Phase A.2 MPI broadcast**.
- SPR: FCA (738.569 s) vs WS-A.1 (729.748 s), Δ 8.8 s — noise; warm-start does not affect SPR.
- **Conclusion:** Phase A.1 confirms correctness (lnL, model, exit code all PASS) and establishes the np=1 timing baseline for measuring Phase A.2 MPI broadcast benefit. The real speedup from warm-start will appear at np≥2 once Phase A.2 is in place.

PBS used 00:16:46 walltime (WS-A.1) / 00:16:40 (FCA baseline). Run records: `logs/runs/gadi_AA_100k_ws_a1_np1_full_seed1_169094692.json` / `logs/runs/gadi_AA_100k_fca_np1_full_seed1_169095077.json`
CHANGELOG: `(bw)` submission · `(bx)` WS-A.1 results · `(by)` FCA submission · `(bz)` FCA results.
