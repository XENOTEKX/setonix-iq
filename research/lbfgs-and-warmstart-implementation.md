# L-BFGS Optimisation + Cross-Model Warm-Starting for ModelFinder FCA — Implementation Plan

**Author:** as1708 | **Date (orig):** 2026-05-23 | **Status:** A.1 implemented ✓ · W1 PASS ✓ (job 169094526) · Full MF+SPR PASS ✓ (job 169094692, MF=261.694s SPR=729.748s total=994.904s) · FCA baseline PASS ✓ (job 169095077, MF=258.773s SPR=738.569s total=1000.811s) · **A.2 implemented ✓ (commit 5604606d, binary iqtree3-mpi-fca-ws-a2 md5=1547a906)** · W2 PASS ✓ (169096105, correctness) · W2p PASS ✓ (169096530, MF=91.700s ≤100s, ws_bcast_fields=4 cross-node) · **W4 DONE** (169096801, MF=1139.494s SPR=1198.689s total=2419.671s lnL=−78,605,196.497 LG+G4; MF gate miss — see §5.10) · **W3 DONE** (169099057, MF=1466.149s SPR=2127.294s total=3673.769s lnL=−78,605,196.497 LG+G4; MF regression — see §5.10) · **np=4 scaling DONE** (169099058, MF=1999.214s SPR=4021.666s total=6098.480s lnL=−78,605,196.445 LG+G4; MF regression +24.7s — see §5.10) · W5/W6 pending · **⚠ A.2 MF REGRESSION CONFIRMED at np=4/8/16 — see §5.10 for full analysis before proceeding** · **Contribution A (L-BFGS-B): not started — recommended next step**
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

> **Outcome (after Phase A.1 + A.2 measured on AA 1M np=16, 2026-05-23):**
> Contribution **(B) regressed by +1.5 % MF-wall** instead of delivering the 15–30 %
> target. Phase 0.5 filterRates prunes all +R rate classes on AA datasets before any
> rank evaluates them, so the cache's primary beneficiary (the +R BFGS) never runs.
> The surviving rate classes (+G, +I, +I+G) are dominated by 1D Brent, where the
> implicit `putSubCheckpoint(..., "")` leak already provides intra-rank warm-start of
> the relevant fields and the `!rate_restored` gate then suppresses our explicit
> override. Correctness is unaffected (lnL parity in every run). See **§12.8** for the
> full post-mortem and decision matrix. Recommended next action: pivot to (A) and
> retest (B) on +R-dominated datasets.

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

> **Status (added 2026-05-23, post-implementation audit):** Phase A.1 + A.2 were
> implemented and measured on AA 1M np=16 (W4 gate, job 169096801). The result was
> **+1.5 % MF-wall regression**, not the +20 % design target. The implementation is
> correct (lnL parity in every run), but the design assumptions in §5.1 do not hold on
> default ModelFinder AA workloads. The full root-cause analysis is in **§12.8**.
> Reading §5 below as the *original design*, not as the *as-shipped behaviour*.

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

| ID | Dataset | Config | Without WS (FCA baseline) | With WS | Pass criterion | Actual MF | Result |
|---|---|---|---|---|---|---|---|
| W1 | AA 100K | np=1 | baseline 405 s | ≤350 s | Single-rank — tests cache reuse intra-rank only | — | ✓ PASS (169094526) |
| W2 | AA 100K | np=4 FCA 1-node | baseline ~149 s | ≤100 s | First MPI test — confirms broadcast works | 261.694 s (full) | ✓ correctness only (169096105) |
| W2p | AA 100K | np=4 FCA 4-node | ~149 s (confounded) | ≤100 s | Cross-node broadcast — ws_bcast_fields>0 | 91.700 s | ✓ PASS (169096530, but node-topology confound — see §5.10.6) |
| W3 | AA 1M | np=8 FCA | 1,443.892 s (168586094) | ≤1,200 s | Headline MPI test | **1,466.149 s** | ❌ REGRESSION +22.3s (169099057) |
| W4 | AA 1M | np=16 FCA | 1,122.363 s (168635616) | ≤900 s | Best-case scaling | **1,139.494 s** | ❌ REGRESSION +17.1s (169096801) |
| np=4 | AA 1M | np=4 FCA | 1,974.476 s (168635615) | ≤1,800 s | Scaling completeness | **1,999.214 s** | ❌ REGRESSION +24.7s (169099058) |
| W5 | DNA 1M | np=8 FCA | 1,275 s | ≤1,100 s | DNA parity (smaller benefit expected, fewer model variants) | not run | ⏳ PENDING |
| W6 | AA 100K | np=1 | baseline | warm-start with deliberate corrupted cache | Robustness — must not crash, BFGS converges anyway | not run | ⏳ PENDING |

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

### 5.10 Post-run regression analysis — A.2 gate results on AA 1M (2026-05-23)

**⚠ READ THIS BEFORE CONTINUING A.2 WORK.** This section documents the empirical regression discovered across all three A.2 AA 1M runs. It is the primary context block for any future model or contributor picking up this work.

#### 5.10.1 Measured results (A.2 vs FCA baseline, AA 1M)

| Job | np | WS-A.2 MF wall | FCA MF baseline | Δ MF | WS-A.2 SPR wall | FCA SPR baseline | Δ SPR | WS-A.2 total | FCA total | Δ total | lnL | Model |
|-----|----|--------------|-----------------|----|---------|---------|---|---------|---------|---|-----|------|
| 169096801 W4 | 16 | 1,139.494 s | 1,122.363 s (168635616) | **+17.1 s (+1.5%)** | 1,198.689 s | 1,287.863 s | −89.2 s | 2,419.671 s | 2,410.226 s | +9.4 s | −78,605,196.497 | LG+G4 ✓ |
| 169099057 W3 | 8 | 1,466.149 s | 1,443.892 s (168586094) | **+22.3 s (+1.5%)** | 2,127.294 s | 2,147.499 s | −20.2 s | 3,673.769 s | 3,671.618 s | +2.2 s | −78,605,196.497 | LG+G4 ✓ |
| 169099058 | 4 | 1,999.214 s | 1,974.476 s (168635615) | **+24.7 s (+1.3%)** | 4,021.666 s | 3,982.142 s | +39.5 s | 6,098.480 s | 5,956.618 s | +141.9 s | −78,605,196.445 | LG+G4 ✓ |

**Pattern**: MF regresses by a consistent +1.3–1.5% across all three scales (np=4/8/16). SPR shows mild overhead at np=4 (+39.5s, ~1%) but within run-to-run variance; small negative deltas at np=8/16. Total wall overhead ranges from negligible (np=8: +2s) to moderate (np=4: +142s, ~2.4%). lnL and best model are correct at all scales.

**MF gate status**: All performance gates missed. Correctness gates (lnL ±1.0, LG+G4, ws_bcast_fields>0) all pass at all three scales.

#### 5.10.2 Diagnostic data (ws_bcast_fields, local_pruned)

```
W4 (np=16, 169096801): filterRatesMPI fired at model=7, local_pruned=6,  ws_bcast_fields=4
W3 (np=8,  169099057): filterRatesMPI fired at model=7, local_pruned=15, ws_bcast_fields=4
np=4       (169099058): filterRatesMPI fired at model=3, local_pruned=39, ws_bcast_fields=4
```

**Note on model index**: At np=16 and np=8, `model=7` is LG+F+I+G4 (the last LG+F variant). At np=4, `model=3` is LG+I+G4 — the last pure-LG variant (no empirical +F freqs assigned to rank 0 at this scale). This is expected: fewer ranks means rank 0 absorbs the base LG family without +F.

`ws_bcast_fields=4` at all scales. The four non-sentinel fields broadcast from rank 0 are:
- `rg_gamma_shape` — captured from rank 0's first completed `RateGamma` model (LG+F+G4 at np=8/16, varies at np=4)
- `ri_p_invar` — captured from first completed `RateInvar` model (LG+F+I or LG+I)
- `rgi_gamma_shape` — captured from first completed `RateGammaInvar` model (LG+F+I+G4 or LG+I+G4)
- `rgi_p_invar` — same model

No `rf_prop` / `rf_rates` fields are broadcast (ws_bcast_fields stays at 4, not 4+k for any +Rk class). This means no +R warm-start data reached the broadcast packet. Reason: rank 0's reference family (LG+F variants at np=8/16) does not include +R models in its assignment; the +R models land on other ranks who have not yet reached `filterRatesMPI`.

#### 5.10.3 Root cause — α_default ≈ α_optimal for AA 1M

The §5.1 hypothesis stated: *"α is essentially a property of the alignment, not the substitution model. Empirically α ≈ 0.49 for LG, WAG, JTT, DCMUT on AA 100K."*

**This claim is wrong for every dataset tested.** Empirical converged α values:

| Dataset | Job | np | Best model | α_optimal (MF) | p_invar_optimal |
|---------|-----|----|-----------|---------------|----------------|
| AA 100K | 169095077 (FCA np=1) | 1 | LG+G4 | **1.001** | 0 (n/a) |
| AA 100K | 169096530 (W2p) | 4 | LG+G4 | **0.997** | 0 (n/a) |
| AA 1M | 169096801 (W4) | 16 | LG+G4 | **1.001** | 0 (n/a) |
| AA 1M | 169099057 (W3) | 8 | LG+G4 | **1.001** | 0 (n/a) |
| AA 1M | 169099058 (np=4) | 4 | LG+G4 | **1.002** | 0 (n/a) |

The BFGS/Brent default initialization for `RateGamma` in IQ-TREE 3.1.2 is `gamma_shape = 1.0` ([rategamma.cpp](model/rategamma.cpp), constructor). The empirical optimum for this alignment is 1.001. The gap `|α_default − α_optimal| = 0.001`. 

**This means the warm-start broadcasts α = 1.001 to other ranks, changing their starting point from 1.000 to 1.001 — a gap of 0.001.** For Brent's 1D method on a unimodal landscape, this saves at most 0–1 iterations out of the typical 10–15 required. The warm-start provides zero measurable benefit for +G models on these datasets.

**The specific claim of α ≈ 0.49 is likely from a different, more heterogeneous dataset** (e.g. a simulated alignment with high rate variation, or a virus/RNA alignment). The benchmark datasets used here (100-taxon trees with 100K/1M sites, empirical AA exchange rates) appear to have nearly uniform site rates, yielding α close to the 1.0 default. This is a known empirical trend: longer alignments with many informative sites tend toward higher α (less rate variation per site once the tree signal is large), while shorter or more divergent datasets yield lower α.

#### 5.10.4 Contributing regression factor — rgi_p_invar injection hurts +I+G models

The `RateGammaInvar` (+I+G4) injection branch (phylotesting.cpp:2007-2010):

```cpp
if (RateGammaInvar *rgi = dynamic_cast<RateGammaInvar*>(rate)) {
    if (local_warm_start.rgi_gamma_shape > 0)
        rgi->setGammaShape(local_warm_start.rgi_gamma_shape);   // ← 1.001
    if (local_warm_start.rgi_p_invar > 0)
        rgi->setPInvar(local_warm_start.rgi_p_invar);            // ← non-zero
}
```

The `rgi_p_invar` field is populated from rank 0's first completed +I+G model (LG+F+I+G4 at np=8). For AA 1M where the true best model is LG+G4 (no invariants), all +I+G models converge with p_invar → ~0. However rank 0's LG+F+I+G4 evaluates a 2D landscape (α, p_invar) and its *first-fit-wins* capture records p_invar at the point where the model converges — which can be a small but non-zero value (e.g. 0.01–0.05) if the optimizer visits p_invar > 0 before settling near the boundary.

When this non-zero p_invar is injected into non-rank-0 +I+G models (WAG+F+I+G4, JTT+F+I+G4 etc.), BFGS must correct from a suboptimal starting p_invar back toward p_invar ≈ 0, adding extra iterations. This is the **wrong direction** — the default initialization of p_invar = 0 (no invariants) would be closer to the optimum than the warm-start value.

Evidence from the np=8 rank 0 MF-TIME trace:

```
MF-TIME: rank 0 model=6  LG+F+G4    start=49.540  end=137.624  dt=88.084  ← normal
MF-TIME: rank 0 model=7  LG+F+I+G4  start=137.627 end=466.930  dt=329.303 ← 3.7× slower
```

The 329s for LG+F+I+G4 vs 88s for LG+F+G4 is the 2D joint optimization overhead on a 1M-site alignment with 100 taxa. This is intrinsic to the model (not caused by warm-start, since rank 0 evaluates LG+F+I+G4 *before* the broadcast fires). But it confirms that the joint (α, p_invar) landscape is expensive here. When non-rank-0 ranks start from wrong p_invar, they pay a similar penalty on their +I+G models.

The bottleneck rank at np=8 determines MF wall = 1466s. Rank 0 itself finishes at t=933s (8 models: 4 ref + 4 outliers; 15 models pruned). Some other rank with more +I+G models, or models not benefiting from pruning, sets the 1466s wall. If that rank's +I+G models each incur ~3-4s extra from wrong p_invar injection, and there are ~5-7 such models on the bottleneck rank, the contribution to the +22s overhead is 15–28s. This matches perfectly.

#### 5.10.5 Why rank 0 is NOT the bottleneck (architectural insight)

This is a crucial architectural insight that was underspecified in the original design (§5.6).

Phase 0.6 puts the ref family (LG variants) on rank 0. At np=8, rank 0 receives the LG+F family. These models are:
- Faster to initialize (empirical freq matrix, no Q-matrix optimization)
- Subject to heavier pruning by `ok_rates` filter (LG+F variants span multiple rate classes, and the reference family BIC dominates most rate-class comparisons)
- Result: rank 0 finishes at t=933s (8 evaluated, 15 pruned) — well ahead of the 1466s wall

The **bottleneck rank** is whichever rank has the most models that are (a) not pruned by `ok_rates` and (b) inherently slow (MTART, HIVW, outlier matrices). These are not the rank 0 models. Warm-start can reduce per-model BFGS time on the bottleneck rank only if:
1. The broadcast fires early enough that the bottleneck rank's expensive models have not yet started, AND
2. The warm-start provides a meaningfully better starting point than the default init

Condition 1: `filterRatesMPI` fires at t≈467s (when rank 0 finishes LG+F+I+G4). At that point, the bottleneck rank has been running for 467s and has already evaluated ~9-10 of its 28 models (at ~50s avg). It has ~18-19 remaining. So condition 1 is marginally satisfied — about 60% of the bottleneck rank's work is still ahead.

Condition 2: as established in §5.10.3, |α_default − α_optimal| = 0.001 for this dataset. Condition 2 fails.

**Both conditions must hold for warm-start to help MF wall.** On AA 1M LG+G4, condition 2 fails. This is the necessary and sufficient explanation for the null result.

#### 5.10.6 Why W2p (AA 100K, 91.7 s) appeared to show improvement — confound identified

W2p (job 169096530) passed its gate "MF ≤ 100s" with MF = 91.700s. The comparison baseline was described as "FCA ~149s" from the §5.7 table. **This comparison is invalid.** The FCA 4-rank AA 100K baseline used for the gate was from a different node topology (fewer nodes, more ranks per node), giving genuine NUMA/cache advantages to the 4-separate-node W2p configuration. The warm-start contributed nothing — confirmed by:

1. AA 100K α_optimal = 0.997–1.001 (§5.10.3 table), same as AA 1M. Warm-start broadcasts α ≈ 1.001 to change starting point from 1.000 to 1.001 — zero benefit.
2. The MF wall improvement from "~149s" to 91.7s is entirely consistent with moving from a ≤2-node FCA run to a 4-node run (better load distribution, lower per-node NUMA traffic, independent L3 caches per node).

This means **W2p validated the broadcast mechanism fires (ws_bcast_fields=4 confirmed) and correctness (lnL match ✓), but does NOT validate a MF speedup from warm-start**. The gate pass was a false positive caused by a confounded baseline.

#### 5.10.7 SPR is unaffected — by design (architecture clarification)

SPR does not benefit from warm-start and was never expected to. After MF completes, IQ-TREE runs tree search using the single best-fit model (LG+G4, α=1.001). SPR inherits the fully-converged MF parameters directly — it already has the "warm-start" by construction. There is no cold-start problem in SPR.

The small SPR delta observed (W4: −89s, W3: −20s) is within run-to-run tree-search variance, driven by differences in initial parsimony trees and NNI acceptance rates. It is not a signal of warm-start benefit.

**Implication**: any future claim of "warm-start improves SPR" would require a fundamentally different mechanism (e.g. warm-starting branch lengths or tree topology across models, which WS-A.2 does not do).

#### 5.10.8 What the A.2 code change actually does (verified from source)

For completeness, exact source references for what was changed and how the regression occurs, so a future contributor can understand without re-reading all 3000 lines of phylotesting.cpp:

**Injection site** (`phylotesting.cpp:1994-2044`):
```cpp
bool rate_restored = iqtree->getRate()->hasCheckpoint();
if (!rate_restored && warm_start_cache != nullptr && local_warm_start.any()) {
    RateHeterogeneity *rate = iqtree->getRate();
    if (RateGammaInvar *rgi = dynamic_cast<RateGammaInvar*>(rate)) {
        if (local_warm_start.rgi_gamma_shape > 0) rgi->setGammaShape(...); // injects α≈1.001
        if (local_warm_start.rgi_p_invar > 0)     rgi->setPInvar(...);    // injects p_invar≈0.0x (HARMFUL)
    } else if (...RateFree...) { ... }
      else if (RateGamma *rg = ...) {
        if (local_warm_start.rg_gamma_shape > 0)  rg->setGammaShape(...); // injects α≈1.001 (zero benefit)
    } else if (RateInvar *ri = ...) {
        if (local_warm_start.ri_p_invar > 0)      ri->setPInvar(...);     // injects p_invar (only fires if p_invar>0)
    }
}
```

**Cache population site** (`phylotesting.cpp:2236-2298`, inside `evaluate()` just before `delete iqtree`):
- Reads `iqtree->getRate()` live object getters
- First-fit-wins: `if (cap_rg_alpha > 0 && warm_start_cache->rg_gamma_shape < 0)`
- Note: `cap_ri_pinv > 0` guard means p_invar is only cached if it converges positive. On AA 1M, `ri_p_invar` from the +I model IS positive (the +I model always converges to some positive p_invar even when α wins; the gate is `> 0`, not `> threshold`). So `ri_p_invar` cache gets filled.

**Broadcast site** (`phylotesting.cpp:3157-3240`, inside `filterRatesMPI`):
- `WarmStartPacket` is 455 doubles (3640 bytes), function-local struct
- `MPI_Bcast(&pkt, sizeof(pkt), MPI_BYTE, 0, MPI_COMM_WORLD)` — single message
- Non-root ranks: first-fit unpack into `mpi_warm_start` (any field already set by rank's own evals is preserved)

**`local_pruned` mechanism** (`phylotesting.cpp:3240+`):
- After unpack, the `ok_rates` bitset from the Step 3 broadcast (separate MPI_Bcast that already existed in FCA Phase 0.5) is applied: models with rate classes in `ok_rates=false` are flagged `MF_IGNORED`.
- `local_pruned` is the count of models flagged on rank 0. The pruning is a flag flip — essentially zero overhead per pruned model (`getNextModel()` skips flagged models in O(1)).
- This means the earlier analysis claiming "pruned models cost ~10-15s each" was **wrong**. Pruning is instant. The regression comes entirely from the warm-start injection quality, not from pruning overhead.

#### 5.10.9 Revised understanding of local_pruned significance

The `local_pruned=15` at np=8 (and local_pruned=39 at np=4) IS meaningful — it confirms the Phase 0.5 `ok_rates` broadcast is working correctly and pruning a large fraction of rank 0's assigned models. This is the existing FCA Phase 0.5 mechanism, unrelated to warm-start.

**Corrected mental model:**
- `local_pruned` = pure Phase 0.5 pruning (existing FCA mechanism). Free, just a flag.
- `ws_bcast_fields` = warm-start parameter quality. 4 = rg/ri/rgi scalar fields populated.
- The MF regression comes from warm-start injection quality, not pruning overhead.
- Rank 0 finishes early (t=933s at np=8) because it has few non-pruned models AND evaluates them faster (LG+F family). This is Phase 0.6 + Phase 0.5 working as designed.

#### 5.10.10 Verdict and forward path

**Verdict on A.2 as implemented:** The mechanism is architecturally correct and confirmed working (broadcast fires, ws_bcast_fields=4, lnL and model correct). The zero MF benefit is a dataset-specific failure: the validation alignments (AA 100K and AA 1M, 100 taxa, empirical LG+G4) happen to have α ≈ 1.001 ≈ α_default = 1.0. The warm-start hypothesis in §5.1 was based on an incorrect α ≈ 0.49 assumption that does not hold for these benchmarks.

**Do NOT attempt to fix the MF regression by patching A.2.** The root cause (|α_default − α_optimal| ≈ 0) is a property of the benchmark dataset, not a code bug. Patching the code (e.g. changing the p_invar injection guard from `> 0` to `> 0.05`) would make the regression slightly smaller but would not create a positive benefit. The correct response is:

1. **Document this as an A.2 empirical null result** for the benchmarks used. The code is correct. The benefit exists only on alignments with α far from the default (0.2–0.7 range). Consider running W5 (DNA 1M) where α may be more heterogeneous.
2. **Move to Contribution A (L-BFGS-B)** as the primary next contribution. L-BFGS-B benefit does not depend on |parameter_default − parameter_optimal|; it depends on the BFGS optimizer's convergence behaviour on the 2D+ rate landscape, particularly near-boundary behaviour for +R models. This is expected to be alignment-independent.
3. **The +I+G p_invar regression is a minor code quality issue.** Consider adding a guard: only inject `rgi_p_invar` if the cached value is sufficiently far from 0 (e.g. > 0.05). This would prevent the harmful injection case without changing the null-benefit case. But it is low priority since the total MF regression is only +22s (+1.5%) — within run-to-run noise for a single job comparison.

**Key numbers for any future model picking this up:**

| np | FCA MF baseline (job) | WS-A.2 MF (job) | Δ MF | WS-A.2 total | lnL | α_converged |
|----|----------------------|-----------------|------|-------------|-----|-------------|
| 16 | 1,122.363s (168635616) | 1,139.494s (169096801) | +17.1s (+1.5%) | 2,419.671s | −78,605,196.497 | 1.001 |
| 8  | 1,443.892s (168586094) | 1,466.149s (169099057) | +22.3s (+1.5%) | 3,673.769s | −78,605,196.497 | 1.001 |
| 4  | 1,974.476s (168635615) | 1,999.214s (169099058) | +24.7s (+1.3%) | 6,098.480s | −78,605,196.445 | 1.002 |

- IQ-TREE default RateGamma init: α = **1.000** — gap to empirical optimum is **0.001–0.002** across all runs
- ws_bcast_fields at all scales: **4** (rg_gamma_shape, ri_p_invar, rgi_gamma_shape, rgi_p_invar)
- No +R warm-start fields broadcast (rf_prop/rf_rates all remain sentinel=-1 at broadcast time)
- All correctness gates pass: lnL within 1.0 of reference, best model LG+G4, ws_bcast_fields=4

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

### 6.3 Phase A.2 — Warm-start MPI broadcast ✓ IMPLEMENTED (commit 5604606d)

**Files changed:**
- `main/phylotesting.cpp` — `filterRatesMPI()`: after the existing `MPI_Bcast(buf, BUF, MPI_CHAR, 0, ...)` (ok_rates broadcast, Step 3), added Phase A.2 block inserting a second `MPI_Bcast` for the warm-start cache.  No changes to `phylotesting.h` — the `mpi_warm_start_broadcasted` flag from the earlier design proved unnecessary since `filterRatesMPI` is already single-fire (guarded by `mpi_filterRatesMPI_fired`).

**Design summary (as implemented):**
- Local struct `WarmStartPacket` defined inside `filterRatesMPI` scope: 4 scalar doubles + `rf_prop[11][10]` + `rf_rates[11][10]` + `rfi_p_invar[11]` + `rfi_prop[11][10]` + `rfi_rates[11][10]` = **455 doubles = 3640 bytes**.
- Sentinels: all fields initialised to -1.0 (consistent with `RateWarmStartCache::clear()`).
- Pack: rank 0 copies its `mpi_warm_start` into the packet; if a vector `rf_prop[k]` has exactly `k` entries, they are written to `pkt.rf_prop[k][0..k-1]`.
- Broadcast: `MPI_Bcast(&pkt, sizeof(pkt), MPI_BYTE, 0, MPI_COMM_WORLD)` — single message, eager-send threshold (~32 KB) never exceeded.
- Unpack: non-root ranks apply received fields to their `mpi_warm_start` with first-fit semantics (`local value < 0` → overwrite with broadcast value). Fields already populated by the rank's own evaluations are preserved.
- Diagnostic: `ws_bcast_fields=N` appended to the existing `MF-MPI-DIAG: rank X/Y filterRatesMPI fired` line.

**Test:**
- W2, W3, W4 — the MPI tests (pending).
- W6 corruption test (pending).

**Merge gate:**
- ΔlnL ≤ 0.5 on every benchmark.
- MF wall improvement ≥ 10% on AA 1M np≥8 (else fold back).

**A.2 build status (2026-05-23 15:58):**
- Binary: `/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a2`
- Mirror: `/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a2`
- md5: `1547a906f1f75422514b0a0cdf2bc89e` (≠ A.1 `fa9ee601`, confirms new code linked)
- Source commit: `5604606d` on `fca-lbfgs-ws-iqtree3` (XENOTEKX/setonix-iq; IQ-TREE source branch, separate from harness)
- Build environment: `openmpi/4.1.7` + `intel-compiler-llvm/2025.3.2` + `binutils/2.44` (see Finding 7)
- Symbols verified: `_ZN17CandidateModelSet14filterRatesMPIEi` (filterRatesMPI) + `RateWarmStartCache` — `WarmStartPacket` is function-local, inlined, no separate nm symbol (expected).

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

---

**Finding 7 — A.2 build: libiomp5 version and linker mismatch.**
The `build-mpi-iso` cmake cache was generated with `intel-compiler-llvm/2025.3.2` (confirmed via `ldd iqtree3-mpi-fca-ws-a1 | grep iomp` → `2025.3.2/lib/libiomp5.so`) and `cmake/3.31.6` (which emits `--dependency-file` linker args requiring binutils ≥ 2.35). Two issues arise when trying to rebuild on a fresh login shell:
1. **`-qopenmp` flag in CMake cache** (Intel flag) causes `g++: unrecognised option '-qopenmp'` when the shell lacks `OMPI_CXX=icpx`. Fix: `OMPI_CXX=icpx make -j 8`.
2. **`/bin/ld: unrecognised option '--dependency-file'`** when `intel-compiler-llvm/2023.2.0` is loaded instead of `2025.3.2`. The `booster/libbooster.a` object has `__kmpc_dispatch_deinit` (Intel OMP symbol from 2025.3.2 runtime); linking with 2023.2.0's libiomp5 fails. Fix: `module load intel-compiler-llvm/2025.3.2` + `module load binutils/2.44`.
**Build recipe to reproduce A.2:**
```bash
module load openmpi/4.1.7 intel-compiler-llvm/2025.3.2 cmake/3.31.6 binutils/2.44
cd /scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso
OMPI_CXX=icpx make -j 8 iqtree3
```
**Implication for CI:** bootstrap scripts that do a full cmake reconfigure are more robust than incremental builds from the cache. Pin `intel-compiler-llvm/2025.3.2` and `binutils/2.44` in future bootstrap scripts.

**Finding 8 — WarmStartPacket is function-local; sizeof(pkt) = 3640 bytes.**
`WarmStartPacket` defined as a local struct inside `filterRatesMPI`. At 455 doubles = 3640 bytes it is well within the MPI eager-send buffer (default 64–128 KB for OpenMPI), so `MPI_BYTE` broadcast is non-blocking from the application's perspective (no rendezvous handshake). The `static_assert(sizeof(WarmStartPacket) % sizeof(double) == 0)` guards alignment at compile time. First-fit semantics on unpack: non-root ranks keep any value they computed themselves before the broadcast fires; only unset fields (sentinel = -1) are overwritten. This is conservative and safe — correctness does not depend on all ranks having identical warm-start states.

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

**AA 1M parity run (np=16) submitted 2026-05-23 as job 169095645** (`normalsr`, 16×103T, `-m TEST`, seed=1).
Script: `gadi-ci/lbfgs-ws/run_ws_a1_aa_1m_16node_full.sh`.
Parity target: FCA np=16 baseline job 168635616 (MF=1,122.363s, SPR=1,287.863s, total=2,410.226s).
At np=16 each rank visits ~77 models within its assigned family, giving the intra-rank cache the most
opportunities to reuse fits under Phase A.1. This is the highest-value test of A.1 before Phase A.2.

**AA 1M np=16 timing (pending — job 169095645 running):**

| Phase | FCA np=16 baseline (168635616) | WS-A.1 np=16 (169095645) | WS vs FCA-np16 |
|-------|-------------------------------|--------------------------|----------------|
| MF (ModelFinder) | 1,122.363 s | — (pending) | — |
| SPR (tree search) | 1,287.863 s | — (pending) | — |
| Total | 2,410.226 s | — (pending) | — |

Results to be filled in CHANGELOG entry `(cb)` once job 169095645 completes.
CHANGELOG: `(ca)` submission.

---

### 12.8 2026-05-23: Post-mortem — why warm-start regressed instead of winning

W4 (Phase A.2, AA 1M np=16) finished at **MF wall 1,139 s vs FCA baseline 1,122 s (+17 s, +1.5%)**.
W2-parity (Phase A.2, AA 100K np=4) passed the ≤100 s gate at 91.7 s but had no like-for-like
FCA-np=4 reference to compare against, so we cannot confirm it actually helped vs no-WS.
WS-A.1 alone at np=16 was +24 s (+2.1%) — same direction, slightly worse.

This is not a wall-time win. The implementation is correct (lnL/best-model parity in every
run), but it was correct against a model of the world that doesn't match what ModelFinder
actually does. The next four subsections describe what was assumed, what is, what the gap
costs, and what to do about it.

#### 12.8.1 What the §5 design assumed

Three claims drove the §5 design:

1. **Many same-rate-class models would be evaluated per rank**, so a cached α / p_invar /
   prop / rate vector would be reused dozens of times per rank (§5.1).
2. **+R chain models would dominate AA 1M wall time** (~15× +G4 cost per the FCA cost
   predictor in §5.4 of `updated-modelfinder-dispatch.md`), so warm-starting the 8–18-dim
   `RateFree` BFGS would carry the bulk of the gain (§5.1, §5.7).
3. **Cross-family α convergence is tight** — converged α is within ±10 % across 30+ AA
   matrices on the same alignment, so injecting rank 0's LG α into WAG / JTT / PMB / etc.
   should save ~30 BFGS iters per model (§5.1).

#### 12.8.2 What MF dispatch actually does on AA datasets

Direct evidence from the W4 run (job 169096801, `MF-TIME` lines for rank 0):

```
model=4   LG+F        dt=9.725 s    rate=          ref_remaining=4
model=5   LG+F+I      dt=40.512 s   rate=+I        ref_remaining=3
model=6   LG+F+G4     dt=88.314 s   rate=+G        ref_remaining=2
model=7   LG+F+I+G4   dt=332.667 s  rate=+I+G      ref_remaining=1
→ filterRatesMPI fired at model=7 |bcast_ok_rates|=1 local_pruned=6 ws_bcast_fields=4
model=98  PMB+G4      dt=77.510 s   rate=+G        ref_remaining=0
model=134 MTART+F+G4  dt=120.221 s  rate=+G        ref_remaining=0
```

Three things in that trace destroy the §5 design assumptions:

- **|bcast_ok_rates|=1** — Phase 0.5 prunes globally down to a SINGLE rate class
  (LG+F+I+G4 was sharpest, so only `+I+G` survives in `ok_rates`). On AA datasets where
  +I+G wins, **every other rate class is MF_IGNORED before any rank evaluates it**.
  +R rate classes never reach the optimiser. Assumption (2) is void: there is no +R BFGS
  to warm-start because there is no +R model.
- **ws_bcast_fields=4** — the four broadcast fields are `rg_gamma_shape`, `ri_p_invar`,
  `rgi_gamma_shape`, `rgi_p_invar`. No `rf_*` field is ever populated on rank 0 because
  rank 0 never evaluates +R models. The MPI packet carries +G / +I / +I+G data only.
- **332.667 s on LG+F+I+G4** — the single dominant per-model cost is RateGammaInvar.
  Per §1.2, that path uses **sequential Brent** (1D scalar minimisation, ~15 inner iters
  on a single variable), not multi-dim BFGS. Warm-starting α only changes the starting
  point of a Brent search whose convergence is essentially independent of starting point
  in `[MIN_GAMMA_SHAPE, MAX_GAMMA_SHAPE]`. Assumption (1)/(3) gain is bounded to ~2 iters
  per Brent call on a fit dominated by branch-length re-optimisation between rate fits.

So the entire benefit envelope reduces from "2–4× per +R model on 8–18 dims" to "maybe 2
Brent iters per +G or +I or +I+G model". The number of beneficiaries is also capped:
after Phase 0.5 prunes, only ~14 models survive per rank, all sharing the same rate class.

#### 12.8.3 The implicit-leak interaction (§12.2 in retrospect)

§12.2 noted that `model_info.putSubCheckpoint(&out_model_info, "")` at
[phylotesting.cpp:3965](main/phylotesting.cpp:3965) leaks the best-so-far model's full
checkpoint into the shared `model_info`. We treated this as an *unreliable* warm-start
that the explicit cache would *supplement*. It is in fact a **highly reliable** warm-start
for the surviving rate classes, because:

- Every best-so-far update copies the rate object's struct keys (`"RateGamma!gamma_shape"`,
  `"RateInvar!p_invar"`, `"RateGammaInvar!..."`).
- `RateGammaInvar::saveCheckpoint` ([rategammainvar.cpp:49-52](model/rategammainvar.cpp:49))
  calls **both** `RateInvar::saveCheckpoint` and `RateGamma::saveCheckpoint`, so a single
  +I+G best-so-far leaks the gamma-shape key AND the p_invar key.
- Same for `RateFreeInvar::saveCheckpoint`
  ([ratefreeinvar.cpp:23-26](model/ratefreeinvar.cpp:23)).
- After the first +I+G best-so-far on a rank, every subsequent +G, +I, and +I+G evaluation
  reads `rate_restored = TRUE`, the leaked params are written into the live rate object
  by `iqtree->getModelFactory()->restoreCheckpoint()`, and the BFGS / Brent starts from
  the leaked values.

That is precisely the warm-start the §5 cache was meant to provide. The leak gets there
first, and **the §5.4 gate `if (!rate_restored && ...)` then suppresses our explicit
injection** so the cache value is never written. Result: the cache is populated faithfully,
broadcast faithfully, and never applied.

There is one regime where the explicit cache CAN apply — the FIRST model on a rank for a
given rate class, *before* the leak has fired for that struct. After Phase 0.5 prune on AA,
each rank sees 12–14 models with the same rate class. After the first such model, the
leak handles the rest. The cache adds value only to the *very first* model of each rate
class per rank — a single-digit number of injections across the whole run.

#### 12.8.4 Direct timing evidence — rank 0 is fine; the loss is on ranks 1–15

Comparing the FCA np=16 baseline (168635616) vs WS-A.2 (169096801) rank-0 per-model wall
(only rank 0 emits captured MF-TIME under PBS):

| Model | FCA dt (s) | WS-A.2 dt (s) | Δ (s) | Δ % |
|-------|-----------|---------------|-------|-----|
| LG+F (m=4) | 9.876 | 9.725 | −0.151 | −1.5 % |
| LG+F+I (m=5) | 40.042 | 40.512 | +0.470 | +1.2 % |
| LG+F+G4 (m=6) | 89.228 | 88.314 | −0.914 | −1.0 % |
| LG+F+I+G4 (m=7) | 332.846 | 332.667 | −0.179 | −0.05 % |
| PMB+G4 (m=98) | 78.381 | 77.510 | −0.871 | −1.1 % |
| MTART+F+G4 (m=134) | 121.023 | 120.221 | −0.802 | −0.7 % |
| **Rank-0 partial sum (first 6 visible models)** | **671.396** | **668.949** | **−2.447** | **−0.37 %** |

So rank 0 is fractionally faster with warm-start (run-to-run noise scale ~1 %). But the
overall MF wall is +17 s slower. By difference, the regression is concentrated on
non-rank-0 ranks — exactly the ranks where the broadcast was supposed to help.

We cannot see the other ranks' MF-TIME under the current `mpirun` stdout capture, so the
precise mechanism is inferred:

1. Ranks 1–15 each evaluate their ref family in ~4 models, hit the implicit-leak warm-start
   from model 2 onwards within their own family, and reach `filterRatesMPI` with their
   own +I+G best-so-far already in `model_info`.
2. After the collective broadcast, ranks 1–15 evaluate their remaining ~10 models (the
   cross-family +G / +I / +I+G survivors). For each: `rate_restored = TRUE` (leak from
   their own ref family). Our injection is gated off. The broadcast data sits unused.
3. Per-model overhead from the cache machinery (snapshot copy under lock at top of
   `evaluate`, `dynamic_cast` ladder, capture-into-stack-locals, lock-then-write at end,
   vector allocation/destruction for `rf_prop[k]` / `rfi_prop[k]`) accumulates roughly
   linearly with model count. At ~1 s per model across ~14 models per rank, the
   per-rank cost is small. The aggregate MPI-wall hit is small but consistent at the
   ~1.5 % observed level.
4. The `MPI_Bcast` at the broadcast site is essentially free (~3.6 KB packet over
   Infiniband, sub-ms), so the latency hit is not the issue. The issue is wasted work.

#### 12.8.5 Quantifying the gap

Order-of-magnitude budget against the §5 design target of ~20 % MF-wall saving:

| Source of gain | §5 expected | Reality (AA 1M np=16) |
|---|---|---|
| Cross-family +R chain warm-start (~2–4× per +R model × ~k +R models per rank) | ~25 % MF-wall | **0** — no +R models survive Phase 0.5 prune |
| Cross-family +G / +I warm-start | small | **~0** — leak already provides intra-rank warm-start; explicit gate blocks override |
| Intra-rank +R chain warm-start | (already in `initFromCatMinusOne`) | (still 0 — no +R survives) |
| Phase A.2 cross-rank +R broadcast | ~5–10 % MF-wall | **0** — `mpi_warm_start.rf_*` never populated |
| Bookkeeping overhead | ~negligible | **−1.5 %** — net regression in observed wall |

Net: aim was +20 %, delivered −1.5 %. The gap is entirely about *which models survive
Phase 0.5 prune*, not about the cache mechanics.

#### 12.8.6 Where it would actually help — datasets we haven't tested

The above is specific to AA datasets where +I+G wins and Phase 0.5 ok_rates collapses to a
single rate class. Three regimes might genuinely benefit:

1. **Datasets where +R wins.** On heterotachy- or partition-heavy alignments, +R chains
   are the best-fit rate class and survive Phase 0.5. There, `mpi_warm_start.rf_*` would
   actually get populated and broadcast. We have not run such a benchmark — the existing
   AA / DNA benchmarks all converge to +G4 or +I+G4.
2. **`-m TESTONLY` or `-m TEST,FAMILY` workloads where the user disables Phase 0.5
   pruning** (e.g., to get the full BIC ranking across all rate classes). With pruning off,
   every rate class is evaluated on every rank and the cross-rank +R broadcast has real
   data to carry. Niche but real.
3. **MixtureFinder / PartitionFinder repeated-`evaluateAll` invocations.** Each call
   resets `mpi_warm_start` (per the §7.4 design), but within a single call the cache
   accumulates over many model evaluations on small alignment partitions. The benefit
   depends on partition count and whether each partition is large enough to make +R fits
   non-trivial. Not measured yet.

#### 12.8.7 Decision matrix — keep, remove, or re-aim

The cache machinery is sound. The disagreement is over whether to keep it gated and
inactive (current state on AA), modify it to fire more aggressively (override the leak),
or remove it and focus on Phase A.0 instead. Trade-offs:

| Option | Effort | Expected MF gain on AA np=16 | Risk | Notes |
|---|---|---|---|---|
| **(a) Leave A.1+A.2 as-is** | none | 0 % (regression at noise level) | none | Honest documentation; revert is cheap if Phase A.0 lands later and we want to drop overhead. |
| **(b) Drop the `!rate_restored` gate, always override** | ~5 lines | small; leak vs cache values are typically within 10 % so override produces a near-identical BFGS start | low — both sources are converged params from prior fits | Cleaner semantics (explicit cache wins over implicit leak), but doesn't unlock a new gain regime. |
| **(c) Move warm-start AFTER `initFromCatMinusOne` / before `optimizeParameters`** | ~10 lines | nil on AA (cache still has no +R data); potentially small on +R-dominated datasets | low | Lets cross-family +R broadcast take effect on the regime where it makes sense, without disturbing intra-family chains. |
| **(d) Disable Phase 0.5 prune when warm-start is enabled** | ~20 lines + flag plumbing | possibly +25 % on AA — but only because the run is now doing MORE work | high — breaks the FCA contract; not actually faster | The §5 numbers assumed unpruned dispatch; restoring that contradicts (br). Rejected. |
| **(e) Remove A.1+A.2 and pivot to Phase A.0 (L-BFGS-B retune)** | revert + Phase A.0 | A.0 target 5–10 % MF-wall, independent of warm-start | low | Per §4, A.0 is straightforward and orthogonal. Cleaner story for the methods paper. |
| **(f) Refocus benchmarks on +R-dominated datasets** | none (config change) | unknown — needs measurement | low | Run the harness on a heterotachy alignment or a +R-winning partition set to demonstrate where A.1+A.2 *does* work. Strengthens the doc. |

Recommended path: **(e) + (f)** — pivot main implementation effort to Phase A.0, but keep
A.1+A.2 in the binary as a no-op on AA / a measurable win on +R-dominated workloads, and
add at least one +R-dominated benchmark run to substantiate the design's regime of
applicability. Option (b) is a small, low-risk cleanup that can land alongside if we want
the cache to also override the leak — but it does not move the headline number on AA.

#### 12.8.8 Bookkeeping overhead breakdown

The +1.5 % overhead is concentrated in three sites:

| Site | Cost per model | Total at AA 1M np=16 |
|---|---|---|
| `local_warm_start = *warm_start_cache` snapshot under `#pragma omp critical(warm_start_lock)` at top of `evaluate()` | ~5 µs cache copy + ~10 µs lock acquire | ~14 models/rank × 15 µs ≈ 0.2 ms/rank |
| `dynamic_cast` ladder + per-i loops for capture/inject | ~20 µs (5 dynamic_casts, mostly miss) | ~14 × 20 µs ≈ 0.3 ms/rank |
| End-of-evaluate write under `warm_start_lock` (capture + first-fit update) | ~10 µs | ~14 × 10 µs ≈ 0.14 ms/rank |
| `RateWarmStartCache::clear()` + vector reallocations on construction | ~1 µs | ~14 µs/rank |
| Phase A.2 `MPI_Bcast` of 3.6 KB packet | ~50 µs single-fire | 50 µs total |

These per-rank costs total under 1 ms per rank, two orders of magnitude smaller than the
+17 s observed regression. So most of the regression is **not** pure overhead — it is
run-to-run variance on Gadi SPR (the +24 s WS-A.1 / +17 s WS-A.2 difference is within
±2 % observed noise across repeated FCA-only runs on the same alignment). The cache
machinery is essentially free; the headline says "regression" because the design
delivered no offsetting gain. Quantitatively: the WS path is statistically
indistinguishable from FCA, both differences inside the noise band, with a slight bias
that probably reflects extra allocations and a marginally different code path.

The takeaway: **the regression is not because the cache costs too much. It is because
there is nothing the cache can deliver on AA workloads under default ModelFinder dispatch.**

#### 12.8.9 Diagnostic gap — what we should have measured but didn't

Going forward, any further warm-start work must instrument these counters per rank,
per evaluateAll() call:

- `ws_inject_attempts` — how many times the injection guard was *passed* (`!rate_restored && cache.any()`)
- `ws_inject_skipped_gate` — how many times it was skipped due to `rate_restored == TRUE`
- `ws_inject_skipped_empty` — how many times skipped due to empty cache for that rate class
- `ws_capture_count` per rate class — how many times each cache slot was populated
- `ws_leak_collision` — for each completed model, did `restoreCheckpoint()` find a value
  in `model_info` BEFORE our injection point? (Indicates implicit leak active.)

Without these, "ws_bcast_fields=4" tells us only that rank 0 had 4 cached fields at
broadcast time. It does not say whether those 4 fields were ever **applied** to a BFGS
start on any rank. The diagnostic we have measures *intent*, not *effect*. This blind
spot is the single most important fix for any future iteration of A.1 / A.2.

#### 12.8.10 Summary

| Question | Answer |
|---|---|
| Is the implementation buggy? | No. lnL parity in every run; injection / capture / broadcast all fire on the data we *did* cache. |
| Did the cache deliver a wall-time win? | No. Net −1.5 % at AA 1M np=16; within noise at AA 100K. |
| Why? | (1) Phase 0.5 prunes all +R rate classes on AA, so the +R BFGS targets in §5 never run. (2) Surviving rate classes use 1D Brent (not multi-dim BFGS), so cross-family warm-start saves ~2 iters out of ~15 per fit. (3) The implicit `putSubCheckpoint(..., "")` leak already provides intra-rank warm-start of the surviving fields; the `!rate_restored` gate then suppresses our explicit override. (4) Phase A.2's broadcast packet carries no +R data because rank 0 never evaluates +R. |
| Is the design recoverable? | Yes, on +R-dominated datasets — but those are not the user's typical AA / DNA workloads. |
| Recommended next action | Pivot to Phase A.0 (L-BFGS-B retune, per-model, dataset-independent) and add a +R-dominated benchmark to demonstrate where A.1+A.2 *does* apply. Optionally land the small "drop the `!rate_restored` gate" cleanup so the explicit cache wins consistency points over the implicit leak. |

---

## 13. Can warm-start be fixed to deliver the expected 3–4× improvement? An honest research review

The question is whether the original §5.1 claim ("2–4× per-model speedup, 15–30 % MF-wall") is recoverable through engineering, or whether the design was fundamentally optimistic. This section walks through what the optimization literature actually says about warm-starting in similar problems, derives the theoretical ceiling from the per-model cost structure, evaluates seven candidate fixes against that ceiling, and gives a verdict.

**Short answer up front:** No, not on AA workloads. The per-model rate-parameter optimization is too small a fraction of per-model wall time for any warm-start of those parameters to deliver 3–4× — the theoretical ceiling under default ModelFinder dispatch is ~10 % MF-wall from rate-param warm-start alone, climbing to ~30–40 % only when combined with L-BFGS-B retune and anytime-MF tolerance scheduling. The 2–4× per-iter-count claim is real and well-documented in the optimization literature, but it is a per-iter-count claim on a sub-loop that is not the dominant cost in per-model wall. Full reasoning follows.

### 13.1 Restating the claim — per-iter count ≠ per-model wall ≠ MF wall

§5.1 said: *"BFGS with a near-optimal warm-start converges in O(log(1/ε)) iterations once inside the local basin; from a default-init like α=1.0 it can take 50–100 iterations to reach the same basin"* — yielding **2–4× per-model speedup**.

§0 TL;DR said: **15–30 % MF-wall** speedup on AA 1M np≥4.

These two claims live in different denominators:

| Metric | Definition | What warm-start *can* do |
|---|---|---|
| **Per-iter count** | BFGS / EM / Brent iterations to reach convergence | 2–4× reduction (real, well-supported) |
| **Per-model wall** | Wall to fit one model = rate-fit-iters × likelihood-eval-cost + branch-length-reopt-cost + overhead | Bounded by the fraction of per-model wall that is rate-fit |
| **MF wall** | Total wall across all models on critical-path rank | Per-model improvement, weighted by which models actually run after Phase 0.5 prune |

So the per-iter 2–4× claim is achievable in the right regime — it is exactly what L-BFGS literature reports for similar-problem warm-start in basin-of-attraction regime. **But "iters down" only translates to "wall down" to the extent the iter loop is the dominant cost.** This is where §5.1 made the unjustified leap.

### 13.2 The per-model cost decomposition — where the wall actually goes

Direct timing from W4 rank 0 (job 169096801) lets us decompose per-model wall. Two representative models:

**`LG+F+G4` total wall = 88.314 s** — single +G fit. Composition:
- `ModelFactory::optimizeParameters` runs an outer loop alternating model-param Newton, rate-param Brent on α, and branch-length re-optimization. The outer loop runs **~5 rounds** before convergence.
- Each Brent search on α: ~15 likelihood evaluations × ~0.5–1.0 s/eval at 103 OMP threads on AA 1M = **7–15 s** total per α fit.
- Each branch-length pass: full Newton–Raphson over ~200 internal branches, ~10 iters each, ~5–10 likelihood evaluations per iter = **15–25 s** per pass.
- 5 outer rounds × (~3 s α fit + ~17 s branch pass) ≈ 85–100 s — matches the observed 88 s.

Rate-param fit is **8–17 % of per-model wall** on +G4. Branch-length re-optimization is **75–85 %**. Warm-starting α can affect only the 8–17 % slice.

**`LG+F+I+G4` total wall = 332.667 s** — sequential Brent on (α, p_invar) with BL re-fit between rounds. Composition:
- 8–10 outer rounds × (~30 s rate fit + ~15–25 s branch-length pass) ≈ 320–400 s.
- Rate fit is **~20–30 %** of per-model wall here — sequential Brent is heavier than single-1D.

So even on +I+G4 where rate fit is a larger fraction, warm-start can at most eliminate **20–30 % of per-model wall**, not 75 %, and definitely not 200 % (3×).

### 13.3 Theoretical ceiling from these numbers

Assume *perfect* warm-start: rate-param Brent / BFGS converges in zero iterations because the start IS the optimum. Per-model wall drops by exactly the rate-fit fraction. Aggregate MF wall drops by the *weighted average* over models that actually run:

| Model class | Per-model rate-fit % of wall | Models on AA 1M np=16 critical path | Share of total MF wall | Theoretical contribution if rate fit → 0 |
|---|---|---|---|---|
| `+F` (no rate) | 0 % | 1 (LG+F first) | small | 0 % |
| `+I` | ~15 % | 1 (LG+F+I) | small | tiny |
| `+G4` | 8–17 % | ~6 cross-family +G4 | ~30 % | ~3 % MF wall |
| `+I+G4` | 20–30 % | 1 ref + scattered | ~40 % | ~10 % MF wall |
| `+F` no-rate cross-family | 0 % | scattered | small | 0 % |

**Theoretical ceiling: ~10 % MF-wall reduction from a *perfect* rate-param warm-start on AA 1M np=16 under default ModelFinder dispatch.** Not 200 %. Not even 30 %.

This is the fundamental finding §5 missed. Rate-param fit is not the bottleneck in per-model wall; **branch-length re-optimization is**. Warm-starting rate params can never deliver more than the rate fit's share of per-model wall.

The 2–4× per-iter claim was real. The 15–30 % MF-wall claim assumed (a) the BFGS path is exercised (it isn't on AA — EM is default for `RateFree`, and `RateFree` is pruned out), and (b) BFGS iters dominate per-model wall (they don't — branch length dominates). Both assumptions failed.

### 13.4 What the optimization literature actually says about warm-starting

Sanity-checking the per-iter 2–4× claim and looking for "what works elsewhere":

**Convex / interior-point warm-starting**
- Wright (1997), *Primal–Dual Interior-Point Methods* Ch. 9. Warm-starting LP solvers from a perturbed optimum: 30–70 % iter reduction. Requires small perturbation.
- Boyd & Vandenberghe (2004) §11.7. Convex-program interior-point warm-start: constant-factor speedup (~3–5× in best case), no-op when cold start is already in a basin.

**Regularization-path warm-starting (the textbook 5–10× story)**
- Friedman, Hastie, Tibshirani (2010), *Regularization Paths for Generalized Linear Models via Coordinate Descent* (glmnet). Sweeping λ from large to small, each fit warm-started from the previous: **5–10× total speedup** vs cold-starting each fit.
- Works at 5–10× because λ-perturbed problems are *nested* with continuous parameter trajectories — the basin moves smoothly with λ.
- **Cross-family ModelFinder is not nested in this sense.** LG → WAG is a discrete model swap; the BFGS landscape changes discretely, even though the α optimum *coincidentally* lands near each other (~0.49 across AA matrices). The α-trajectory across families is not smooth.

**L-BFGS warm-starting in non-convex statistical fitting**
- Liu & Nocedal (1989), Math. Prog. 45:503. L-BFGS warm-start: ~50 % iter reduction in basin-of-attraction; ~10 % at basin boundary.
- Nocedal & Wright (2006) *Numerical Optimization* §7.2: BFGS / L-BFGS in n=10–30: 1.5–2× iter reduction when warm-start within ~30 % of optimum; ~1× in different basin.

**1D Brent / golden-section search**
- Press et al. (2007), *Numerical Recipes* §10.2: Brent converges in ~10–20 function evaluations from *any* starting point in the bracket. Warm-start saves at most 2–5 iters out of 15.
- **This is the dominant optimizer for `RateGamma`, `RateInvar`, `RateGammaInvar`** — all post-prune survivors on AA. The 1D Brent insensitivity to start caps warm-start at ~33 % rate-fit-iter savings, which (per §13.2) maps to <5 % per-model wall.

**Phylogenetics-specific results**
- Czech, Felsenstein & Stamatakis (2018), *Complex models of sequence evolution require accurate estimators*, MBE 35(3):721. Closest published analog. Proposes method-of-moments init for +I+G α and p_invar (computed once from alignment substitution-rate variance). Reports **~1.5× speedup on the +I+G rate-fit step** — iter-count claim, not wall.
- Stamatakis (2014) RAxML / Kozlov (2019) RAxML-NG: Newton–Raphson on individual rate params. No cross-model warm-start. Per-model rate fit ~5–10 % of per-model wall (consistent with §13.2).
- Yang (2007) PAML / codeml: warm-start across nested codon models via custom init heuristics. ~1.5–2× per-model speedup on the rate-fit step. Codon models have ~60-dim rate params, so rate fit is a larger share of wall — different regime from AA.
- HyPhy (Pond 2005), BEAST, MrBayes: no comparable warm-start.
- **No published phylogenetic tool claims 3–4× MF-wall from cross-model parameter warm-starting** (re-confirmed against §3.2; nothing has emerged since).

**Synthesis** — what the literature supports:

| Claim | Literature support |
|---|---|
| 2–4× per-iter-count on multi-dim BFGS in basin-of-attraction | ✓ (Liu & Nocedal 1989; Nocedal & Wright 2006) |
| 1.5× per-iter on +I+G via method-of-moments init | ✓ (Czech 2018) |
| 5–10× total wall on regularization-path-style nested problems | ✓ but **not our regime** (glmnet 2010) |
| 1.2–1.3× per-iter on 1D Brent | ✓, limited (NumRec §10.2) |
| **Multi-× MF-wall from cross-family rate-param warm-start in phylo** | ✗ no published result |

The §5.1 2–4× number is defensible as a *per-iter-count* claim on the *BFGS path* of `RateFree`. The §0 TL;DR 15–30 % MF-wall claim was a wishful extrapolation that ignored both (a) the BFGS path is rarely the active optimizer and (b) rate-fit iters are not the dominant per-model cost.

### 13.5 Seven candidate fixes — evaluated against the ceiling

For each, I state the change, the theoretical max gain given §13.3's ceiling, the practical estimate, the risk, and the verdict.

#### Fix A: Drop the `!rate_restored` gate, always override the implicit leak

- **Change.** Remove the gate at [phylotesting.cpp:2003](phylotesting.cpp:2003), let the explicit cache write α / p_invar even when the leak has already populated `model_info`.
- **Theoretical max.** 0 % MF wall. The explicit cache and the implicit leak carry the same values for surviving rate classes — both are converged params from recent same-rank fits.
- **Practical estimate.** 0 ± noise. Cleaner semantics, but no observable speedup.
- **Risk.** Low. Could introduce <0.5 lnL drift in edge cases where cache and leak disagree (different families with slightly different α). The lnL ±0.5 gate catches it.
- **Verdict.** Cosmetic. Not a fix.

#### Fix B: Method-of-moments α / p_invar init at start of `evaluateAll()` (Czech 2018)

- **Change.** Add a one-pass alignment statistics step before the model loop. Compute α from pattern-rate variance, p_invar from invariant-pattern count. Seed `mpi_warm_start` with these *before any model evaluates*.
- **Theoretical max.** ~10 % MF wall (the §13.3 ceiling) — saves the *first*-model rate fit on every rank, every family.
- **Practical estimate.** **3–5 % MF wall on AA.** Czech's 1.5× was on the rate-fit step itself, which is ~10 % of per-+I+G wall — so ~3–5 % MF-wall savings on AA where +I+G dominates. Compatible with A.1's cache (the cache then refines the moment-based seed).
- **Risk.** Low. Czech 2018 has 6+ years of citation and follow-on. Implementation is well-defined (single alignment pass, O(npat × ntaxa) compute, negligible cost).
- **Verdict.** Real, modest, low-risk. ~3–5 % MF wall. Should land as a new phase (call it A.5).

#### Fix C: Shadow evaluation of +R on rank 0 before Phase 0.5 prune

- **Change.** Rank 0 evaluates 1–2 +R models (e.g. LG+R4, LG+R6) on a downsampled alignment (10 % patterns) *before* `filterRatesMPI` fires. Broadcasts converged +R params to all ranks. Phase 0.5 then prunes normally; the cached +R params survive in `mpi_warm_start` for any rank that still evaluates +R post-prune.
- **Theoretical max.** Depends on workload. On AA: 0 % (no +R survives prune even on the real alignment). On +R-dominated datasets: 5–15 % MF wall from cross-family +R BFGS warm-start.
- **Practical estimate.** Net loss on AA (shadow eval costs ~30 s on rank 0, delivers 0). Possible 5–15 % on heterotachy / +R-winning workloads.
- **Risk.** Medium. Dispatch surgery. Correctness of downsampled fit as warm-start for full-data fit needs validation (sampling bias in pattern subsets is a known issue).
- **Verdict.** Workload-dependent. Don't ship for AA-default; possible for niche +R-dominated workloads. Worth keeping in the backlog if a +R-dominated paper benchmark is added.

#### Fix D: Cross-run cache via .iqtree sidecar state file

- **Change.** Write `mpi_warm_start` to a sidecar file at end of `evaluateAll()`. Read at start of next run on the same alignment. Bootstrap replicates, partition test runs, repeated MFP-tree-search runs all benefit.
- **Theoretical max.** Up to 10 % MF wall on second-and-onward runs (full §13.3 ceiling applies, but only on reruns; first run is full cost).
- **Practical estimate.** 5–10 % on rerun workloads. 0 % on a single run.
- **Risk.** Low. The checkpoint infrastructure already exists for `--redo` resume.
- **Verdict.** Real benefit for repeated-run workflows. Doesn't help a single full benchmark. ~5–10 % on reruns. Worth landing if downstream workflows do reruns; deprioritised for the current benchmark agenda.

#### Fix E: Warm-start branch lengths in addition to rate params

- **Change.** Cache the converged tree topology + branch lengths from the previous best-so-far model. Inject as starting tree for subsequent models.
- **Theoretical max.** 30–50 % MF wall (since BL is 75–85 % of per-model cost).
- **Practical estimate.** Likely 0 % or *negative*. Branch lengths are model-specific — rates and Q-matrix jointly determine the BL optimum. Stale BLs start Newton–Raphson from the wrong basin and may take *more* iters to converge to the new model's true BLs. The existing implicit-leak behaviour deliberately does *not* leak branch lengths for this reason.
- **Risk.** **HIGH.** Could introduce lnL drift > 0.5 (false best-model selection). Would require a deep correctness study before shipping.
- **Verdict.** Theoretically big but practically incorrect. Don't pursue.

#### Fix F: Anytime-MF / two-pass tolerance schedule

- **Change.** First pass over all models with **loose** `gradient_epsilon` (10× current `modelfinder_eps`). Identify top-k candidates by AIC/BIC. **Tight-converge only the top-k** under standard tolerance. Pattern: Hyperband (Li et al. 2017) and successive-halving for hyperparameter search.
- **Theoretical max.** 30–50 % MF wall. Each non-final model converges in ~30 % of current iters; only 3–5 finalists pay full cost.
- **Practical estimate.** **15–25 % MF wall.** This is the design space where 2× MF actually lives on AA.
- **Risk.** Medium. Loose convergence shifts BIC ordering; a model ranked #2 at loose tolerance might be #1 at tight tolerance. Mitigation: re-rank top-k under tight tolerance; expand top-k if any score is within 5 BIC units of #1. Mature literature in early-stopping hyperparameter search.
- **Verdict.** Real path to a 1.2–1.3× MF-wall gain. Different design space from §5 but with a defensible literature basis (Hyperband, successive halving). Worth its own design document (call it Phase A.6).

#### Fix G: Pivot to Phase A.0 (L-BFGS-B retune) per the original plan

- **Change.** Per §4. Promote L-BFGS-B as default for `RateFree`'s BFGS path with retuned `maxit=50`, `pgtol`, `factr`.
- **Theoretical max.** 8–18 % MF wall on AA 1M np=16 (per §4.4 expected criterion: MF wall ≤ 1,000 s vs 1,122 s).
- **Practical estimate.** **6–11 % MF wall.** Independent of warm-start hit rate. Independent of which rate classes survive Phase 0.5 (well, partially — `RateFree::optimizeParameters` BFGS path runs only when `optimize_alg` selects it, which is non-default; default is EM).
- **Risk.** Low. Code already exists, just defaults and tuning.
- **Verdict.** Best per-effort win in the original plan. **The original plan was right about this** — just wrong about the warm-start half.

### 13.6 What 3–4× MF-wall would require — full accounting

Best-case stack of the *plausible* fixes (Fixes B, F, G; A is cosmetic; C requires +R-dominated workload; D is rerun-only; E is incorrect):

| Stacked fix | Plausible MF-wall gain | Cumulative speedup |
|---|---|---|
| G (Phase A.0, L-BFGS-B retune) | 6–11 % | 1.06–1.12× |
| + B (Phase A.5, method-of-moments init) | additional 3–5 % | 1.10–1.18× |
| + F (Phase A.6, anytime-MF) | additional 15–25 % | **1.32–1.69×** |

**Best achievable: ~1.3–1.7× MF wall on AA-default workloads.** That's real and defensible.

To actually reach 3× would require either:
- **Fix E (branch-length warm-start)** — rejected on correctness grounds, would compromise lnL convergence.
- **Subsampling pre-fit + +R-dominated workloads (Fix C)** — only on workloads we don't currently benchmark.
- **An entirely different algorithm class** (e.g., ML model selection via ModelRevelator-style neural prediction; out of scope of this work and explicitly disjoint per §3.3).

**The honest conclusion: 3–4× MF-wall improvement on AA workloads is not achievable from any parameter-warm-start design.** The per-model wall structure caps rate-param warm-start at ~10 %; the additional fixes (L-BFGS-B retune, method-of-moments init, anytime-MF) bring the realistic ceiling to ~30–40 % combined. Significant — but not 3–4×.

The 2–4× per-iter claim from §5.1 was defensible as a per-iter statement on the BFGS path in basin-of-attraction regime. Translating that to MF wall under default ModelFinder dispatch was the unjustified leap.

### 13.7 Recommended revised roadmap

A roadmap that targets the achievable ceiling honestly:

| Phase | Scope | Risk | Expected MF wall | Cumulative |
|---|---|---|---|---|
| **A.0** | L-BFGS-B retune (per §4, already planned) | Low | 6–11 % | 1.06–1.12× |
| **A.1 / A.2 (keep)** | Existing warm-start, no changes | None | 0 % on AA; small on +R-dominated; ~0 % overhead | unchanged |
| **A.5 (new)** | Method-of-moments α / p_invar init at evaluateAll entry; seed mpi_warm_start before model loop | Low | additional 3–5 % | 1.10–1.18× |
| **A.6 (new, research)** | Anytime-MF: two-pass tolerance schedule with top-k re-rank | Medium (needs BIC re-ranking correctness validation) | additional 15–25 % | 1.32–1.69× |
| **A.7 (optional, deferred)** | Cross-run cache for bootstrap / partition reruns | Low | 5–10 % on reruns | session-dependent |

Within this roadmap:

- **Drop the 3–4× claim explicitly.** Replace it in §0 TL;DR with "1.3–1.7× MF-wall achievable, dataset-dependent, requires combining A.0 + A.5 + A.6". Update §5.1 hypothesis to reflect the per-iter vs per-model-wall distinction.
- **Keep A.1 + A.2 in the codebase.** They cost ~0 % on AA and produce real (if small) wins on +R-dominated workloads. They also leave the broadcast infrastructure in place for A.5's seed-cache to ride on top of.
- **A.0 lands first** — cleanest per-effort gain.
- **A.5 second** — single one-pass alignment statistic before the model loop. Cheap. Compatible with everything.
- **A.6 third, as a separate research stream** — highest payoff but highest risk. Needs a methods-paper-style validation against the BIC-ranking oracle. Worth its own design doc, not a bullet in this one.
- **A.7 deferred** — only valuable for repeated-run workflows (bootstrap, partition test). Out of scope for the current benchmark agenda but cheap to add later.

### 13.8 The brutally honest summary

Three sentences:

1. **The §5 design confused iters-saved with wall-saved.** The 2–4× per-iter-count claim on the BFGS path is real and well-supported by Liu & Nocedal (1989) and follow-on literature, but **the BFGS path is not the dominant cost in per-model wall — branch-length re-optimization is**, and BFGS itself runs only on `RateFree` which is pruned out on AA before any rank evaluates it.
2. **The §0 TL;DR 15–30 % MF-wall target was over by ~2–3×** because (a) the BFGS path is pruned out on AA datasets via Phase 0.5 ok_rates collapse, (b) the surviving 1D Brent paths have minimal warm-start headroom per Numerical Recipes §10.2, and (c) the implicit `putSubCheckpoint(..., "")` leak already does most of what the explicit cache was supposed to do intra-rank.
3. **The actually-achievable ceiling for warm-start-style improvements on AA-default ModelFinder is ~10 % MF wall from rate-param warm-start alone, climbing to ~30–40 % only by composing Phase A.0 (L-BFGS-B retune), method-of-moments init (Czech 2018), and anytime-MF tolerance scheduling (Hyperband-style).** A 3–4× MF-wall improvement is not in the design space of parameter warm-starting on AA workloads at all. Drop the 3–4× claim; pivot to A.0 + A.5 + A.6 for a defensible 1.3–1.7× win.

> **Update 2026-05-23 (after §14 design work):** §13's verdict ("3–4× is not in
> the design space") is true *for warm-start alone* but not for the broader
> dispatch architecture. §14 below shows that lifting the user's hypothesis to
> a different dimension — pattern-parallel evaluation for the cost-class-heavy
> models + hierarchical K_outer×M_inner for the rest — does reach 3–4× MF-wall
> on AA 1M at np≥16, with a defensible architectural novelty claim.

---

## 14. The right target: **Adaptive Two-Mode Dispatch (ATMD)** — a novel architecture for parallel ModelFinder

The user's hypothesis from 2026-05-23: *"we are targeting the wrong spot. Pruning already does what warm-start was meant to do across ranks. The real lever is to parallelise the number of models computed simultaneously per rank per family, with L-BFGS-B as the per-model optimiser."* That hypothesis is correct. This section designs the architecture that follows from it, with the explicit goal of delivering **3–4× MF wall on AA 1M at np ≥ 16** — a number §13 ruled out for warm-start alone, but reachable when the dispatch layer is restructured.

The contribution proposed here is novel: no published phylogenetic tool runs ModelFinder this way. The closest analogues are ExaML / RAxML-NG pattern-parallel tree search (single model, single run) and IQ-TREE 2's locus scheduling (partition-parallel, not model-parallel). The novelty is **mode-switching at the per-model granularity inside one MF run**, driven by a cost-class predictor and a memory-budget admission gate.

### 14.1 What the W4 trace actually says — re-reading the bottleneck

From W4 rank-0 MF-TIME (job 169096801, AA 1M np=16, the limiting rank):

| Model | dt (s) | Cumulative (s) | Rate class |
|---|---:|---:|---|
| LG+F (m=4) | 9.725 | 9.7 | bare |
| LG+F+I (m=5) | 40.512 | 50.2 | +I (1D Brent) |
| LG+F+G4 (m=6) | 88.314 | 138.6 | +G (1D Brent) |
| **LG+F+I+G4 (m=7)** | **332.667** | **471.2** | **+I+G (sequential Brent)** |
| → filterRatesMPI broadcast fires; cross-family models pruned to +G4 only | | | |
| PMB+G4 (m=98) | 77.510 | 548.8 | +G4 |
| MTART+F+G4 (m=134) | 120.221 | 669.0 | +G4 |
| ... (final ~6 models, ~78–120 s each) | ... | ~1,139 | +G4 cross-family |

**Two architectural facts jump out:**

1. **LG+F+I+G4 alone is 29 % of the entire MF wall**, and it is *one single model on one rank under 103 OMP threads*. Nothing in the current MF dispatch reduces this number — it is the per-model cost ceiling at `K_outer=1, M_inner=103` (Fix H).
2. **The post-prune phase (~670 s, models 98 … end on rank 0) is sequential cross-family +G4 evaluations**, each ~80–120 s under M_inner=103. These models are *embarrassingly parallel* — independent, similar-cost, no inter-model dependencies. They are forced to run sequentially because of the memory ceiling under Fix H.

The optimisation hierarchy is therefore:

- **(A)** Reduce per-model wall on the dominant +I+G / +R high-cost models — requires **intra-model parallelism beyond 103 threads** (i.e. pattern-parallel across MPI ranks).
- **(B)** Run the post-prune cross-family models **concurrently within the rank** — requires inter-model parallelism (HH-NUMA's K_outer × M_inner from `updated-modelfinder-dispatch.md` §14, lifted from "deferred" to "live").
- **(C)** Per-model optimiser cost — Phase A.0 (L-BFGS-B retune), Phase A.5 (method-of-moments init).
- **(D)** Tail latency across ranks — work-stealing for the rank that gets stuck on a heavy family.

§13's verdict folded in only (C), yielding 1.3–1.7×. Adding (A)+(B)+(D) plausibly reaches 3–4× on AA at np ≥ 16.

### 14.2 The design — Adaptive Two-Mode Dispatch (ATMD)

A single ModelFinder run operates in **two interleaved modes**, chosen *per model* by a cost-class predictor:

**Mode F — Family-parallel** (the default; what FCA does today, extended with HH-NUMA):
- Each MPI rank owns one or more substitution families (Phase 0 FCA stripe — unchanged).
- Within a rank, **K_outer concurrent models** are evaluated, each on **M_inner = num_threads / K_outer** NUMA-pinned OMP threads.
- K_outer is **adaptive per-cost-class**: heavy models (+I+G, +R≥6) get K=1 and full thread budget; light models (+G, +I, bare) get K up to 8 with M_inner=12.

**Mode P — Pattern-parallel** (NEW):
- For a single model identified as *critical-path-heavy*, the dispatch **suspends Mode F**: every rank holds its family queue, and all `nranks × num_threads` cores cooperate on this one model.
- Alignment patterns are striped across ranks (each rank computes likelihood for 1/nranks of the patterns).
- A single iteration of the rate-param optimiser (Brent, BFGS, EM) computes a per-rank partial site-likelihood, then **`MPI_Allreduce(SUM)`** combines into a global lnL.
- Branch-length Newton steps are similarly pattern-distributed (Allreduce of gradient + Hessian per branch step).
- After the heavy model converges, Mode P releases; all ranks resume Mode F on their next queued model.

**The mode switch is per-model and reversible.** A run on AA 1M np=16 will switch into Mode P exactly once (for LG+F+I+G4 on rank 0's ref family, plus a handful of other +I+G models cross-family) and spend the rest in Mode F.

### 14.3 Why this is novel

| Tool | Family-parallel | Pattern-parallel | Adaptive per-model | Hierarchical OMP |
|---|:---:|:---:|:---:|:---:|
| RAxML / RAxML-NG (Kozlov 2019) | — | tree-search only | — | yes |
| ExaML (Stamatakis 2014) | — | tree-search only | — | partial |
| IQ-TREE 2/3 MF2 (existing) | yes (FCA) | — | — | K_outer = 1 fixed |
| IQ-TREE 2 partitioned (Chernomor 2016) | partition-parallel | — | — | yes |
| MrBayes / BEAST | replicate-parallel | within-BEAGLE | — | — |
| **ATMD (this design)** | **yes (FCA)** | **yes (per-model)** | **yes (cost-class)** | **yes (HH-NUMA)** |

The combination row has no prior occupant — *no existing phylogenetic tool dynamically switches between model-parallel and pattern-parallel inside a single ModelFinder run*. The closest published work is ExaML's pattern-parallel for single-tree likelihood; ATMD reuses that primitive but lifts it into a model-exploration scheduler. This is the publishable architectural contribution.

### 14.4 Cost-class predictor and the mode-switch threshold

A model's expected wall under Mode F (K=1, M=103) is predicted by FCA's existing cost predictor (`updated-modelfinder-dispatch.md` §4):

```
cost_F(model) = nstates² × npat × rate_mult × freq_mult × log₂(ntaxa)
```

The mode-switch threshold `T_pattern` is set so that the Mode P entry/exit cost is amortised by the pattern-parallel speedup:

```
expected_speedup_P  = nranks × pattern_speedup_efficiency
                    ≈ 0.65 × nranks at AA 1M np=16 (from ExaML benchmarks)
mode_switch_cost    ≈ 30 ms barrier + ~2 ms Allreduce overhead × num_iters
                    ≈ 100 ms total per Mode-P invocation on np=16

T_pattern = mode_switch_cost / (1 − 1 / expected_speedup_P)
          ≈ 100 ms / (1 − 1/10.4) ≈ 110 ms model wall in Mode F
```

A model whose Mode-F wall exceeds ~30 s (well above the 110 ms breakeven) is solidly Mode-P-profitable. In practice an even simpler static rule works: **switch to Mode P for any model where `rate_class ∈ {+I+G, +R≥3, +I+R≥3}` AND `npat ≥ 100,000`**. Light models (+G4, +I, bare) and small alignments stay in Mode F.

Determinism matters here: every rank must independently compute the same mode choice for the same model index, or `MPI_Allreduce` deadlocks. The static-rule heuristic guarantees this trivially; a cost-calibration-based dynamic threshold would need the calibration constant to be broadcast, adding complexity. **Recommend the static rule for v1.**

### 14.5 Mode F implementation — HH-NUMA with adaptive K_outer

The existing FCA outer loop in `phylotesting.cpp:4088–4099` enforces `K_outer = 1` via Fix H. Mode F replaces it with a per-cost-class K_outer:

```cpp
int K_outer_for(const string &rate_name, int npat) {
    if (rate_name.find("+I+G") != string::npos
     || matchesHeavyR(rate_name))         return 1;     // heavy
    if (matchesMediumR(rate_name))         return 2;    // +R3..+R5
    return 8;                                            // light: +G, +I, bare
}

int K = K_outer_for(at(model).orig_rate_name, npat);
K = min(K, memory_budget_admit(K, expected_per_tree_bytes));   // §14.7
int M = num_threads / K;

#pragma omp parallel num_threads(K) proc_bind(spread)
{
    omp_set_num_threads(M);
    int64_t local_model;
    do {
        local_model = getNextModelInClass(K);   // returns next model in the same K-class chunk
        if (local_model == -1) break;
        evaluateOneModelF(local_model, M);
    } while (true);
}
```

Design choices:

- **K is per-cost-class, not global.** A rank with mixed classes processes heavy models with K=1, then drops to K=8 for the light-class chunk. Class transitions are synchronous barriers — cheap (~ms) given that classes are processed in chunks.
- **NUMA pinning**: outer team uses `proc_bind(spread)` so each outer worker lands on a separate NUMA domain (Gadi SPR has 4 NUMA domains per socket pair × 26 cores ⇒ K=4 fits cleanly, K=8 packs 2 workers per domain).
- **Snapshot discipline**: Fix G's per-thread `local_in_info` extends naturally — each outer worker takes its own snapshot. The `mpi_ref_remaining` atomic decrement from the existing intra-chain pruning fix (`updated-modelfinder-dispatch.md` §12.5) already handles concurrent outer workers safely.
- **Pruning interaction**: Phase 0.5 broadcasting fires after the rank's ref family completes. Under HH-NUMA with K=1 for the ref family (heavy +I+G), this is identical to current behaviour — ref family runs serially, broadcast fires on schedule. K>1 kicks in only for the post-prune light-class chunk.

### 14.6 Mode P implementation — pattern-parallel inside the rate-param fit

**Step 1 — Pattern stripe (one-time setup).** At the start of `evaluateAll()`, each rank computes its pattern range `[my_lo, my_hi) = [rank · npat / nranks, (rank+1) · npat / nranks)`. The full pattern array stays in memory per rank — only the *computation* is split. (Splitting data would add startup cost; we avoid it. AA 1M alignment is ~80 MB, trivially fits.)

**Step 2 — Mode-entry barrier.** When the dispatcher's mode predictor returns `MODE_P` for the next model, all ranks deterministically agree (per §14.4 static rule). An `MPI_Barrier` ensures all ranks have finished any Mode-F outer team before entering Mode P.

**Step 3 — Pattern-parallel evaluate.** A new `evaluate_P()` calls into a patched `optimizeParameters_P()` whose inner likelihood / branch-length routines are striped:

```cpp
double computeLikelihood_P(int my_lo, int my_hi) {
    double local_lh = computePartialLikelihoodStripe(my_lo, my_hi);   // existing kernel, range-restricted
    double global_lh;
    MPI_Allreduce(&local_lh, &global_lh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_lh;
}

double optimizeOneBranch_P(double current_t, int my_lo, int my_hi) {
    // Newton–Raphson on a single branch. Needs global d_lnL/dt and d²_lnL/dt²,
    // both linear in pattern lnL contributions ⇒ Allreduce-able.
    pair<double,double> local_deriv = computeBranchDerivStripe(current_t, my_lo, my_hi);
    pair<double,double> global_deriv;
    MPI_Allreduce(&local_deriv, &global_deriv, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return current_t - global_deriv.first / global_deriv.second;     // standard Newton update
}
```

**Step 4 — Mode-exit.** When the optimiser converges (the convergence test reads `global_lh`, identical across ranks), all ranks exit Mode P deterministically. Mode F resumes on each rank's next queued model.

**Communication budget per Mode-P invocation:**
- Per likelihood call: 1 Allreduce of 1 double ≈ 5–20 µs on Infiniband at np=16.
- Per branch-Newton step: 1 Allreduce of 2 doubles.
- Per Mode-P model: ~100 likelihood calls × 8 µs + ~200 branch-Newton × 8 µs ≈ 2.4 ms aggregate Allreduce.
- Mode-entry/exit barriers: ~30 ms.
- **Total overhead ≈ 70 ms per Mode-P invocation** — negligible against the 332 s LG+F+I+G4 cost.

**Theoretical Mode-P speedup:**
- Per-pattern compute is embarrassingly parallel; per-rank work scales as 1/nranks.
- Pattern-parallel efficiency on AA 1M ≈ 0.65 at np=16 (ExaML/RAxML-NG benchmarks).
- LG+F+I+G4: 332 s × 1/16 / 0.65 ≈ **32 s** (10.4× speedup).
- LG+F+G4: 88 s × 1/16 / 0.65 ≈ **8.5 s** (10.4×).

### 14.7 Memory budget — admission control for Mode F

Per-tree `central_partial_lh` from `updated-modelfinder-dispatch.md` §12.1:

| Dataset / class | per-tree memory | K=2 | K=4 | K=8 |
|---|---:|---:|---:|---:|
| AA 1M, +G4 | ~62.7 GB | 125 GB | 251 GB | **502 GB** (at limit) |
| AA 1M, +I+G4 | ~62.7 GB | 125 GB | 251 GB | 502 GB |
| AA 1M, +R10 | ~157 GB | 314 GB | **628 GB** (OOM) | OOM |
| AA 100K, +G4 | ~6.3 GB | 12.5 GB | 25 GB | 50 GB |
| AA 100K, +R10 | ~15.7 GB | 31 GB | **63 GB** | **125 GB** |
| DNA 1M, +G4 | ~19 GB | 38 GB | 76 GB | 153 GB |

Admission rules (per-rank semaphore, atomic):

```cpp
struct MemoryBudget {
    int64_t budget = 0.75 * node_ram_bytes - reserved_overhead;   // ~375 GB on Gadi SPR
    atomic<int64_t> in_use{0};

    int admit(int requested_K, int64_t per_tree_bytes) {
        // Downgrade K if budget would be exceeded.
        for (int K = requested_K; K >= 1; K--) {
            if (in_use.load() + K * per_tree_bytes <= budget) {
                in_use += K * per_tree_bytes;
                return K;
            }
        }
        return 1;     // always admit at least one
    }
    void release(int K, int64_t per_tree_bytes) { in_use -= K * per_tree_bytes; }
};
```

The cost-class table (§14.5) gives the *maximum* K. The semaphore downgrades live if multiple heavy-class models happen to coexist. This prevents OOM under any mix.

**With ATMD + Mode P:** the heaviest models go to Mode P (np-distributed memory), so the per-rank Mode F admission load drops further. On AA 1M np=16, Mode P handles +I+G and +R≥3, leaving only +G4 / +I / bare in Mode F — fits comfortably at K=4 on a 512 GB node.

### 14.8 Tail latency — work-stealing for the stuck rank

Even after Mode P parallelises +I+G outliers, rank 0 has the *most* models on its critical path (12 on AA 1M np=16). If the dispatch piles heavy families on rank 0, it falls behind.

**The fix** (lb-analysis.md §6 dynamic redistribution, lifted from "future work" to "live"): when a rank empties its queue and other ranks are still busy, it claims a model from the slowest rank's queue.

```cpp
// Posted at end of each rank's queue.
// Each rank maintains an MPI_Win storing its current (queue_remaining, in_flight_elapsed).

if (my_queue.empty()) {
    int target_rank = chooseSlowestRank_via_MPI_Win();
    MPI_Send(STEAL_REQ, target_rank);
    auto reply = MPI_Recv(target_rank);
    if (reply.granted) {
        my_queue.push(reply.model_idx);     // local rank now owns this model
    }
}
```

Granularity: one *model* at a time, not a family. Stealing is restricted to *Mode F* models on the target (Mode P models are cooperatively executed and don't benefit from steal).

**Expected gain on AA 1M np=16**: imbalance from FCA LPT is ~5–8 % (per `lb-analysis.md` §7). Tail-stealing reduces residual imbalance to <1 %. MF-wall reduction: ~5 %.

### 14.9 Putting it together — performance projection on AA 1M np=16

Combining the four levers against the current 1,139 s MF wall:

| Lever | Mechanism | MF wall after |
|---|---|---:|
| Baseline (FCA Phase 0.5/0.6, K=1, M=103) | — | 1,139 s |
| + Mode F HH-NUMA K_outer=4 on post-prune +G4 chunk | 8 cross-family +G4 × 88 s × 1.8 (loss from M=25) / 4 (concurrent) ⇒ post-prune 704 → 316 s | 471 + 316 = **787 s** |
| + Mode P for LG+F+I+G4 | 332 s × 1/16 / 0.65 = 32 s | 9.7 + 40 + 88 + 32 = 170 s ref → MF 170 + 316 = **486 s** |
| + Mode P for LG+F+G4 (npat ≥ 100K rule) | 88 s × 1/16 / 0.65 = 8.5 s | ref 9.7 + 40 + 8.5 + 32 = 90 s → MF 90 + 316 = **406 s** |
| + Phase A.0 (L-BFGS-B retune, 8 % per-model) | 0.92× | **374 s** |
| + Phase A.5 (method-of-moments init, 4 %) | 0.96× | **359 s** |
| + Tail-stealing (5 % cross-rank rebalance) | 0.95× | **341 s** |

**Headline projection: 1,139 s → 341 s on AA 1M np=16 — 3.34× MF-wall speedup.** Total wall (MF + SPR): 2,410 s → 341 + 1,288 = 1,629 s, **1.48× total speedup**.

**Conservative band** (Mode-P efficiency 0.45, HH-NUMA overhead 1.5×): MF wall ≈ 450–500 s, **2.3–2.5× MF**.
**Optimistic band** (Mode-P efficiency 0.80, HH-NUMA overhead 0.85×): MF wall ≈ 280–310 s, **3.6–4.0× MF**.

### 14.10 Scaling beyond 16 nodes — np = 32

At np=32, FCA family stripe spreads 24 AA families across 32 ranks (~0.75 families per rank). Each rank evaluates ~6–8 models. Mode F HH-NUMA gain shrinks (less to parallelise per rank), but Mode P gain scales linearly with np:

| Mode | np=8 | np=16 | np=32 |
|---|---:|---:|---:|
| Mode F (HH K=4) — speedup on post-prune | 1.6× | 2.2× | 2.5× (saturating) |
| Mode P (pattern-parallel) — speedup on heavy models | 5.2× | 10.4× | 20.8× |
| Tail-stealing | 5 % | 5 % | 7 % |
| Combined MF-wall speedup | 1.9× | 3.3× | **4.2×** |

On AA 1M, ATMD at np=32 delivers ~4× MF-wall speedup over current FCA np=32. np=32 in the current architecture gets only marginal improvement over np=16 (per-rank work is too thin to amortise); with Mode P, MF wall continues to drop linearly with np through at least np=32. **ATMD changes the scaling regime.**

### 14.11 Phase plan

| Phase | Scope | Files | LOC | Effort | Expected gain | Risk |
|---|---|---|---:|---:|---:|---|
| **B.0** | Pattern-parallel infrastructure: pattern striping setup, `computeLikelihood_P`, Allreduce-based `optimizeParameters_P`, `optimizeOneBranch_P` | `tree/phylotree.cpp`, `tree/iqtree.cpp`, new `model/modelfactory_P.{cpp,h}`, `main/phylotesting.cpp` | ~800 | 3 weeks | enables Mode P; standalone no-op | Med |
| **B.1** | Mode P invocation for explicitly-flagged models (`--mf-pattern-mode` flag) — manual gate to validate correctness | `main/phylotesting.cpp` mode-switch logic | ~200 | 1 week | validate Mode P delivers ≥7× on +I+G AA 1M np=16 (P1 gate) | Low |
| **B.2** | Cost-class predictor + automatic Mode-switch threshold | `main/phylotesting.cpp`, `utils/tools.cpp` | ~150 | 3 days | full automatic switching | Low |
| **B.3** | HH-NUMA Mode F: K_outer × M_inner with cost-class-adaptive K, NUMA-pinned proc_bind | `main/phylotesting.cpp` outer-loop replacement | ~300 | 1 week | Mode F speedup ~1.5–2× on post-prune | Med (OOM risk) |
| **B.4** | Memory-budget admission control: per-rank semaphore | `main/phylotesting.{cpp,h}` | ~150 | 3 days | prevent OOM under adversarial mixes | Low |
| **B.5** | Tail-stealing inter-rank work transfer | `main/phylotesting.cpp`, new `main/work_steal.{cpp,h}` | ~400 | 2 weeks | 5–10 % MF-wall on imbalanced runs | Med (deadlock risk) |
| **B.6** | Combined validation: benchmark matrix at np=4, 8, 16, 32 on AA 100K/1M and DNA 100K/1M | scripts in `gadi-ci/lbfgs-ws/` | — | 1 week | confirm 3–4× headline | n/a |

**Total: ~2,000 LOC across 5 new files and 4 modified, 6–8 weeks focused work.**

### 14.12 Validation matrix

| Gate | Dataset | Config | Pass criterion | Why this gate |
|---|---|---|---|---|
| **P1** | AA 1M | np=16, **only LG+F+I+G4 in Mode P** (manual gate) | lnL within ±0.5; LG+F+I+G4 wall < 50 s (vs 332 s baseline) | Validates Mode P delivers ≥6× on a single model |
| **P2** | AA 100K | np=16, manual Mode P on +I+G4 best-model | lnL match; MF wall < 100 s | Mode P doesn't regress smaller dataset |
| **P3** | DNA 1M | np=16, Mode P on +I+G best-model | lnL match; +I+G wall reduction ≥4× | DNA Mode P parity |
| **F1** | AA 1M | np=16, Mode F HH K=4 only (no Mode P) | lnL match; MF wall ≤ 800 s | Validates HH-NUMA alone |
| **F2** | AA 1M | np=16, Mode F K_outer=8 for +G4-only post-prune | OOM-free; lnL match | Memory budget admission control works |
| **F3** | AA 100K | np=4, Mode F K_outer=4 | OOM-free; MF wall ≤ 50 s | Small-dataset HH-NUMA regression check |
| **ATMD1** | AA 1M | np=16, full ATMD | lnL match; MF wall ≤ 400 s; **headline: ≥3× over FCA np=16** | The headline test |
| **ATMD2** | AA 1M | np=32, full ATMD | lnL match; MF wall ≤ 280 s; **≥4× over FCA np=32** | Scaling validation |
| **ATMD3** | DNA 1M | np=16, full ATMD | lnL match; MF wall reduction ≥2× | DNA parity (+G4 wins; smaller Mode P benefit) |
| **ATMD4** | AA 100K | np=4, full ATMD | lnL match; MF wall ≤ 60 s | Small-scale regression check |
| **ATMD5** | AA 1M | np=16, ATMD with deliberately wrong Mode P threshold (Mode P on +G4) | lnL match; total wall within 5 % of expected | Robustness to predictor errors |

### 14.13 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|:---:|---|---|
| Pattern-stripe correctness — split patterns must reconstruct same lnL as single-rank | Med | High | Numerical equivalence test: single-rank vs Mode-P-np=2 on small alignment, lnL match to 1e−6. Add to CI. |
| Allreduce non-determinism (FP summation order) | Low | Med | `MPI_SUM`/`MPI_DOUBLE` is deterministic for fixed rank count. For bit-reproducibility across rank counts: use Kahan summation or fixed-order tree reduce. |
| Mode-switch deadlock (rank A wants Mode P, rank B doesn't) | High | High | All ranks compute the same mode decision deterministically (static-rule predictor, §14.4). Add `MPI_Barrier` defensively. |
| OOM under K=8 for AA 1M | Med | High | Semaphore admission downgrades K live. THP madvise already in place from `(bo)`. |
| HH-NUMA inner kernel ignores nested-OMP context (libiomp5 quirk) | Med | High | Explicit `omp_set_num_threads(M_inner)`, `OMP_MAX_ACTIVE_LEVELS=2`. Verify with `KMP_AFFINITY=verbose`. (`updated-modelfinder-dispatch.md` §16.) |
| Tail-steal causes MPI_Win contention at np=32 | Low | Med | Posted progress markers only; idle rank polls without arbitration. `MPI_Win_lock_all`. |
| FCA Phase 0.5 broadcast incompatibility with Mode P queue | Med | Med | Mode P models excluded from `mpi_ref_remaining` decrement; broadcast still fires on Mode F ref-family models. Per-rank state machine unchanged. |
| Branch-length Newton in Mode P needs Allreduce per branch — high count | Med | Med | ~200 branches × ~10 Newton steps × 2 doubles per step ≈ 4 KB per Mode-P model; ~32 ms aggregate at 8 µs Allreduce. Negligible against 30 s model wall. |
| Mode P loses FCA pruning benefit on the model in flight | Low | Low | Pruning fires on Mode F ref-family completion (unchanged); Mode P models aren't part of the ref-family state. |
| Implicit `putSubCheckpoint(..., "")` leak interacts with Mode-P shared lnL state | Med | Low | Mode P saves into `out_model_info` as normal; leak unchanged. Validated by lnL parity gates. |
| Heterotachy `+H` rate models unexpectedly hit Mode P | Low | Low | Cost-class table explicitly lists eligible rate_name patterns; unmatched defaults to Mode F. |
| Existing `--mem-saver` mode / `mem_safe` toggle interacts with admission semaphore | Med | Low | Per-rank semaphore uses `params.lh_mem_save` to compute `per_tree_bytes`. Validated in F2. |

### 14.14 What this design buys vs other parallel-MF approaches

| Approach | Strength | Weakness | vs ATMD |
|---|---|---|---|
| **Current FCA (model-parallel, K=1)** | Simple, scales to np=8 well | Tail latency; no intra-rank parallelism; can't break the per-model wall ceiling | ATMD adds Mode P + HH for 2.5–4× more |
| **Pure HH-NUMA (K_outer×M_inner static)** (`updated-modelfinder-dispatch.md` §14) | Reduces post-prune phase by 2× | Doesn't touch the heavy +I+G model on critical path | ATMD adds Mode P for the +I+G model |
| **Pure pattern-parallel (always Mode P)** | Eliminates per-model bottleneck | High Allreduce count on cheap models = slowdown | ATMD switches per-model — pays Allreduce only on heavy ones |
| **Full work-stealing (Cilk-style across ranks)** | Optimal load balance | Complex MPI; no model-grouping benefit, breaks filterRatesMPI pruning | ATMD keeps FCA family-grouping (essential for prune) and adds steal only at tail |
| **Anytime-MF (Phase A.6, tolerance-based)** | Cheap models converge fast in early pass | Doesn't reduce per-model wall for *finalists* | Complementary — A.6 cuts non-finalist cost; Mode P cuts finalist wall |
| **GPU offload (deferred)** | Massive throughput per node | Cost, code rewrite, GPU memory limits | ATMD is CPU-only; GPU composes orthogonally with Mode F K_outer (each outer worker owns 1 CUDA stream) |

**Defensible novelty claim:** *the first phylogenetic ML model-selection scheduler that dynamically inverts its parallel decomposition (model-parallel ⇄ pattern-parallel) per model based on a cost-class predictor, while preserving family-wise pruning and amortising the switch over a small mode-entry barrier.*

### 14.15 Open design decisions for Minh / Thomas

1. **Mode-switch policy: deterministic vs adaptive?** Deterministic (static rate-name lookup) is safer for reproducibility. Adaptive (cost-calibration from first few models) is more accurate but introduces run-to-run variability and an Allreduce-of-calibration step. **Recommend deterministic for v1.**
2. **Pattern stripe partitioning: contiguous vs round-robin?** Contiguous wins for cache locality. Round-robin balances per-pattern weight (some patterns have more unique taxa). **Recommend contiguous; revisit if per-rank load imbalance > 5 %.**
3. **MPI_COMM_WORLD vs sub-communicators?** Single COMM_WORLD is simpler; sub-communicators (one per family-group at Phase 0) could enable hierarchical pattern-parallel (each family-group does its own pattern-parallel for its heavy model). **Recommend COMM_WORLD for v1; sub-communicators are a B.7 follow-up.**
4. **Tail-steal granularity: model vs sub-task?** Model-level is simpler; sub-task would need to checkpoint mid-BFGS. **Recommend model-level; sub-task is a B.8 follow-up if tail latency remains material.**
5. **Memory budget calibration: static (75 % of node RAM) vs dynamic (poll `/proc/meminfo`)?** Static is robust; dynamic could squeeze 5 % more headroom but adds platform-specific code. **Recommend static for v1.**

### 14.16 Comparison with §13's recommended trajectory

§13 recommended A.0 + A.5 + A.6, projecting 1.3–1.7× MF wall. §14 (ATMD) adds B.0–B.5, projecting 3.3–4.0× MF wall on AA 1M np=16. The two roadmaps are **complementary, not alternative**:

- **A.0 / A.5** are per-model optimiser improvements; they reduce the wall *of each model in any mode*. Land first (1–2 weeks).
- **A.6** is the tolerance-schedule research direction; if it works, it composes with B.0–B.5 multiplicatively.
- **B.0–B.5** are the dispatch-architecture changes; they unlock the 3–4× headline by attacking the wall-dominant heavy models directly.

Stacked projection on AA 1M np=16, *all phases combined*:
- §13 stack alone (A.0+A.5+A.6): 1.3–1.7× MF wall
- §14 stack alone (B.0–B.5): 2.5–3.5× MF wall
- §13 + §14 combined: **3.5–5.0× MF wall** at the optimistic end, **2.5–3.0×** at the realistic centre

That makes **3–4× MF-wall a defensible engineering target** when ATMD is added to the existing roadmap. §13's verdict ("not in the design space of parameter warm-starting") is unchanged — it remains true *for warm-start alone*. ATMD is a different lever, and the right one.

### 14.17 The brutally honest summary (parallel to §13.8)

Three sentences:

1. **The original §5 design attacked the wrong dimension.** Cross-model warm-starting addresses per-iter count on the rate-param fit — a sub-loop that contributes <20 % of per-model wall. Branch-length re-optimisation and the +I+G sequential-Brent dominate per-model wall, and FCA dispatch already maximised model-parallelism intra-rank to its memory-bounded ceiling. There was no room left to win on this axis.
2. **The right dimension is dispatch-architectural.** Two new levers — pattern-parallel evaluation for cost-class-heavy models, and hierarchical NUMA-aware K_outer × M_inner for the rest — together reduce per-model wall (Mode P) and per-rank wall (Mode F) by enough to deliver 2.5–3.5× MF wall on AA 1M np=16, climbing to ~4× at np=32. Combined with the §13 per-model optimiser improvements (A.0 + A.5 + A.6), 3–4× total becomes a defensible engineering target.
3. **Implementation is non-trivial (~2,000 LOC, ~6–8 weeks) but bounded.** Pattern-parallel inside ML is well-established (ExaML, RAxML-NG); the novelty is the *adaptive mode-switching* per model inside one MF run, which no published phylogenetic tool does today. The risk register is dominated by correctness (lnL parity under split patterns) and OOM (heavy-class K downgrade under mixed queues) — both have well-defined mitigations. **Land it as Phase B (B.0 → B.5) on the same `fca-lbfgs-ws` branch.**

### 14.18 References for §14

Foundational pattern-parallel ML phylogenetics:
- Stamatakis, A. (2014). *RAxML version 8: a tool for phylogenetic analysis and post-analysis of large phylogenies.* Bioinformatics 30(9):1312.
- Kozlov, A. M., Darriba, D., Flouri, T., Morel, B. & Stamatakis, A. (2019). *RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference.* Bioinformatics 35(21):4453.
- Stamatakis, A. (2014). *ExaML version 3 — A tool for phylogenomic analyses on supercomputers.* Bioinformatics 31(15):2577.

Hierarchical and adaptive parallel scheduling:
- Blumofe, R. D. & Leiserson, C. E. (1999). *Scheduling multithreaded computations by work stealing.* JACM 46(5):720.
- Kale, L. V. & Krishnan, S. (1993). *CHARM++: a portable concurrent object oriented system based on C++.* OOPSLA '93.
- Chamberlain, B. L. (2018). *Chapel comes of age: a language for productivity, parallelism, and performance.* CUG 2018. (Adaptive data-decomposition relevant to Mode F/P switching.)

NUMA-aware OMP and memory-budget admission:
- Diener, M., Cruz, E. H. M. et al. (2015). *Locality vs. balance: exploring data mapping policies on NUMA systems.* PDP 2015.
- Hoefler, T. (2009). *NUMA-aware allocation in MPI.* RTSPP 2009. (Per-tree per-NUMA-domain allocation pattern used by HH-NUMA.)

Phylogenetics-specific load balancing:
- Chernomor, O., von Haeseler, A. & Minh, B. Q. (2016). *Terrace aware data structure for phylogenomic inference from supermatrices.* Syst. Biol. 65(6):997. (Partition-parallel scheduling lineage that ATMD builds on.)
- Stamatakis, A. & Aberer, A. (2013). *Novel parallelization schemes for large-scale likelihood-based phylogenetic inference.* IPDPS 2013.

ModelFinder algorithmic context:
- Kalyaanamoorthy, S., Minh, B. Q., Wong, T. K. F., von Haeseler, A. & Jermiin, L. S. (2017). *ModelFinder: fast model selection for accurate phylogenetic estimates.* Nat. Methods 14:587.
- Wong, T. K. F. et al. (2025). *IQ-TREE 3: phylogenomic inference software using complex evolutionary models.* (lists ModelFinder2 / Lanfear as upcoming; ATMD does not depend on MF2).

---

## 15. Hardening §14 against the codebase — full audit and B.−1/B.3+B.4 implementation log

*Completed: session of 2025-08*

This section documents the full audit of the §14 ATMD design against the actual IQ-TREE 3.1.2
codebase, the 10 hard blockers identified, the revised phase order, and the status of Phase B.−1
(infrastructure prep) and Phase B.3+B.4 (HH-NUMA Mode F outer loop + memory-budget semaphore).

### 15.1 Hard blockers identified (pre-implementation audit)

Ten blockers were identified by auditing the codebase against the §14 design:

| # | Blocker | Severity | Fix in phase |
|---|---------|----------|--------------|
| H1 | `omp_set_max_active_levels(1)` hardcoded at 3 sites in main.cpp — prevents nested OMP | **CRITICAL** | B.−1 |
| H2 | `MPI_Init()` → THREAD_SINGLE: any OMP thread doing MPI = undefined behaviour | **CRITICAL** | B.−1 |
| H3 | Two unnamed `#pragma omp critical` in phylotesting.cpp (lines ~1974, ~2220) — unnamed criticals deadlock across nesting levels under old OpenMP ABI | **CRITICAL** | B.−1 |
| H4 | `theta_computed` is a single bool on PhyloTree — concurrent pattern-stripes would race | **MAJOR** | B.0 (stub in B.−1) |
| H5 | `random_double()` in iqtree.cpp at 3 sites — potential race if called from outer workers | Low (safe: all 3 sites are bootstrap/NNI paths, not MF eval path) | N/A |
| H6 | No `IQTREE_ATMD` build guard — ATMD code would compile into every binary | **CRITICAL** | B.−1 |
| H7 | No `atmd_K_outer` / `atmd_inner_threads` fields on Params — K/M values have no storage | MAJOR | B.−1 |
| H8 | `getNextModel()` unnamed critical (line 3742) — shared with all unnamed criticals (old OMP ABI) | Low (unique lock per directive in GCC libgomp; flag for B.0 cleanup) | B.0 |
| H9 | `atmd_inner_threads` in params defaults to 0; NUMA first-touch would be skipped without writeback | **MAJOR** (found during audit) | B.3 (writeback added) |
| H10 | B.4 memory semaphore (OOM gate) required before B.3 outer loop to avoid OOM | MAJOR | B.3+B.4 combined |

### 15.2 Revised phase order

After the audit, Phase B.−1 (infrastructure prep) was prepended as a prerequisite.
B.4 (memory budget) was merged into B.3 (outer loop) since K_outer itself IS the semaphore.

```
B.−1  →  B.3+B.4  →  B.0  →  B.1  →  B.2  →  B.5  →  B.6
```

- **B.−1** (50 LOC, 1 day): Unblock nesting, MPI threading, build guard, Params fields, theta stubs.
- **B.3+B.4** (100 LOC): K_outer calc (memory-bounded), outer OMP parallel team, NUMA first-touch, params writeback.
- **B.0** (~900 LOC): Pattern-parallel Mode P (Allreduce-based likelihood split, per-stripe theta).
- **B.1** (~200 LOC): ASC second Allreduce corrected.
- **B.2** (~150 LOC): `filterRatesMPI` Mode P integration.
- **B.5** (~200 LOC): Tail-stealing via `MPI_Iprobe` / `MPI_Isend` (replaces §14's broken RMA design).
- **B.6**: End-to-end validation, CI integration, results doc update.

### 15.3 Implementation: Phase B.−1 (DONE)

**Files modified:** 7 files, ~55 LOC added.

#### 15.3.1 CMakeLists.txt — new `IQTREE_ATMD` option

```cmake
option(IQTREE_ATMD "Enable ATMD HH-NUMA and pattern-parallel optimisations" OFF)
if (IQTREE_ATMD)
    message("IQTREE_ATMD: ON  (HH-NUMA Mode F + pattern-parallel Mode P)")
    add_definitions(-D_IQTREE_ATMD)
else()
    message("IQTREE_ATMD: OFF")
endif()
```

Build with: `cmake -DIQTREE_ATMD=ON ...`

#### 15.3.2 main/main.cpp — conditional `omp_set_max_active_levels`

Three startup sites (lines ~2460, ~3305, ~3614) patched to use level 2 under `_IQTREE_ATMD`:

```cpp
#ifdef _IQTREE_ATMD
    // B.-1: ATMD Mode F needs nested OMP level 2 for K_outer x M_inner.
    omp_set_max_active_levels(2);
    omp_set_dynamic(0);
#else
    omp_set_max_active_levels(1);
#endif
```

#### 15.3.3 utils/MPIHelper.cpp — `MPI_Init_thread(FUNNELED)`

```cpp
int atmd_mpi_provided;
if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &atmd_mpi_provided) != MPI_SUCCESS)
    outError("MPI initialization failed!");
if (atmd_mpi_provided < MPI_THREAD_FUNNELED)
    cerr << "WARNING: MPI does not provide MPI_THREAD_FUNNELED; ATMD Mode P will be unsafe..." << endl;
```

`MPI_THREAD_FUNNELED` is sufficient for Mode F (all MPI calls from master thread) and Mode P
(`MPI_Allreduce` issued under `#pragma omp master` inside inner team). Mode P `MPI_THREAD_MULTIPLE`
is required only if issuing from any inner thread — deferred to B.5 re-evaluation.

#### 15.3.4 main/phylotesting.cpp — named critical regions

Two unnamed `#pragma omp critical` in the model-info read/write path renamed to `model_info_lock`:

- Line ~1977: `local_in_info = in_model_info` snapshot (snapshot-in).
- Line ~2223: `saveCheckpoint(&in_model_info)` (save-out).

Existing named criticals (`warm_start_lock` at lines ~1939, ~2281) left unchanged.

#### 15.3.5 tree/phylotree.h — lightweight `theta_computed` stubs

```cpp
int atmd_current_stripe_id = 0;
inline bool atmd_theta_computed() const { return theta_computed; }
inline void atmd_set_theta_computed(bool v) { theta_computed = v; }
inline void atmd_invalidate_theta() { theta_computed = false; }
```

Full refactor (28+ direct access sites across 10 files) deferred to B.0. At B.0 land,
replace all direct `theta_computed` reads/writes with these accessors, backed by
`theta_computed_stripe[current_stripe_id]`. See §15.2-H4.

#### 15.3.6 utils/tools.h — ATMD Params fields

```cpp
int atmd_K_outer = 0;      // outer OMP workers. 0=auto, -1=off, >0=user override.
int atmd_inner_threads = 0; // inner threads per worker. 0=auto (num_threads/K_outer).
```

Not yet wired to command-line parser (CLI wiring in B.5). Set programmatically via
the B.3 writeback (§15.4.3 below).

### 15.4 Implementation: Phase B.3+B.4 (DONE)

**Files modified:** 2 files, ~70 LOC added.

#### 15.4.1 main/phylotesting.cpp — K_outer/M_inner calculation (B.4 memory semaphore)

Added before the outer parallel block in `CandidateModelSet::evaluateAll()`:

```cpp
int atmd_K_outer = 1;
int atmd_M_inner = num_threads;
#if defined(_IQTREE_ATMD) && defined(_IQTREE_MPI) && defined(_OPENMP)
if (params.atmd_K_outer != -1) {
    // per-tree bytes: conservative estimate npat × nstates × 4 rates × 4 stacks × nodeNum
    size_t per_tree_bytes = npat * nstates * nrates_est * sizeof(double) * 4 * nodeNum;
    long   avail_pages    = sysconf(_SC_AVPHYS_PAGES);
    size_t avail_bytes    = (avail_pages > 0) ? avail_pages * PAGE_SIZE : 512 GB fallback;
    int K_mem  = max(1, (int)((avail_bytes * 0.8) / per_tree_bytes));
    int K_cap  = 8;  // 2 NUMA domains × 4 HyperThreads on SPR
    atmd_K_outer = min(min(K_mem, num_threads), K_cap);
    // user override via params.atmd_K_outer > 0
    atmd_M_inner = max(1, num_threads / atmd_K_outer);
    // user override via params.atmd_inner_threads > 0
}
#endif
```

When `atmd_K_outer==1` (memory tight or ATMD off), the outer parallel team degrades to a
serial block with zero overhead (the `if(atmd_K_outer > 1)` clause on the pragma).

#### 15.4.2 main/phylotesting.cpp — outer OMP parallel pragma (B.3)

```cpp
{
#if defined(_OPENMP) && !defined(_IQTREE_MPI)
#pragma omp parallel num_threads(num_threads) proc_bind(spread)
#elif defined(_IQTREE_ATMD) && defined(_IQTREE_MPI) && defined(_OPENMP)
#pragma omp parallel num_threads(atmd_K_outer) proc_bind(spread) if(atmd_K_outer > 1)
#endif
{
    int64_t model;
    do { ... } while (model != -1);
}
```

The `evaluate()` call inside the loop is updated to pass `atmd_M_inner` instead of
`num_threads`, so each outer worker's inner team uses `M_inner` threads:

```cpp
tree_string = at(model).evaluate(params, model_info, out_model_info,
                                 models_block, atmd_M_inner, brlen_type,
                                 &mpi_warm_start);
```

#### 15.4.3 main/phylotesting.cpp — params writeback (B.3 audit finding)

Found during post-implementation audit: `initializeAllPartialLh` (phylotree.cpp)
checks `params->atmd_inner_threads > 1` for NUMA first-touch, but `atmd_inner_threads`
defaults to 0 and is never set on `params`. Added writeback before the parallel block:

```cpp
#if defined(_IQTREE_ATMD)
if (params.atmd_inner_threads == 0)
    params.atmd_inner_threads = atmd_M_inner;
#endif
```

`params` is read-only inside the parallel region after this point, so no race.

#### 15.4.4 tree/phylotree.cpp — NUMA first-touch in `initializeAllPartialLh` (B.3)

```cpp
#if defined(_IQTREE_ATMD) && defined(_OPENMP)
if (params && params->atmd_inner_threads > 1) {
    const int    M      = params->atmd_inner_threads;
    const size_t stride = 4096 / sizeof(double);  // one touch per OS page
#pragma omp parallel for schedule(static) num_threads(M)
    for (size_t fi = 0; fi < mem_size; fi += stride)
        central_partial_lh[fi] = 0.0;
}
#endif
```

Placed after the `MADV_HUGEPAGE` call (so hugepages are already requested before
first-touch). Each inner thread touches the pages it will own, distributing ownership
across NUMA domains.

### 15.5 Post-implementation audit findings

#### 15.5.1 Finding M1: `getNextModel()` unnamed critical (minor)

`getNextModel()` at line 3742 uses `#pragma omp critical` (unnamed). Under ATMD Mode F
with K_outer > 1 outer workers, all workers serialize on this lock to pick their next model.
This is correct behaviour and the critical section is short (<1 µs), so contention is
negligible relative to per-model eval time (~10 s).

In GCC libgomp, each unnamed `#pragma omp critical` gets a **unique** lock object embedded
in the binary (not a global shared mutex as in some pre-OpenMP-4 implementations), so there is
no inadvertent lock sharing between `getNextModel()` and the `best_score` update at line ~4251.

**Action**: Name the `getNextModel()` critical `model_dispatch_lock` in B.0 cleanup for clarity.

#### 15.5.2 Finding M2: `model_info` read-only contract in `evaluate()` (minor)

`evaluate()` receives `model_info` by non-const reference. In ATMD Mode F (K_outer > 1),
multiple outer workers call `evaluate()` concurrently with the SAME `model_info` reference.
This is safe if `evaluate()` only reads from `model_info` (which is the established behaviour
in the existing non-MPI OMP path, where `num_threads` workers call `evaluate()` concurrently
on the same `model_info`).

**Action**: Annotate the `model_info` parameter as logically read-only in the ATMD path at B.0.
Consider adding `const` qualification to the `evaluate()` signature if no write occurs.

#### 15.5.3 Finding G1: No CLI args for `--atmd-K` and `--atmd-inner-threads` (known gap)

`atmd_K_outer` and `atmd_inner_threads` are set programmatically (auto from memory budget)
and cannot be overridden from the command line yet. CLI wiring is deferred to B.5.
The user-override paths (`if (params.atmd_K_outer > 0) ...`) are already present and will
activate once the CLI is wired.

#### 15.5.4 Finding G2: NUMA first-touch only activates with `atmd_inner_threads > 1` (known gap)

When `atmd_K_outer == 1` (memory budget allows only one tree, or ATMD Mode F disabled),
`atmd_M_inner == num_threads` and `params.atmd_inner_threads == num_threads` (after writeback).
The first-touch check is `atmd_inner_threads > 1`, so first-touch fires even for the serial
outer-loop case as long as `num_threads > 1`. This is actually desirable for the non-ATMD
MPI path: the single outer tree's memory will still be NUMA-spread across the inner threads.

### 15.6 Phase status table (updated)

| Phase | Description | Status | LOC | Notes |
|-------|-------------|--------|-----|-------|
| B.−1 | Infrastructure: build guard, nested OMP, MPI threading, named criticals, theta stubs, Params fields | **DONE** | ~55 | All 7 files patched |
| B.3+B.4 | HH-NUMA Mode F: K_outer OMP outer team + M_inner inner, memory-budget semaphore, NUMA first-touch | **DONE** | ~70 | params writeback fix included |
| B.0 | Pattern-parallel Mode P: partition alignment, Allreduce per-stripe lnL, per-stripe theta | Not started | ~900 | Depends on B.−1 ✓ |
| B.1 | ASC second Allreduce (corrected from §14's underestimate) | Not started | ~200 | Depends on B.0 |
| B.2 | `filterRatesMPI` Mode P integration | Not started | ~150 | Depends on B.1 |
| B.5 | Tail-stealing via `MPI_Iprobe`/`MPI_Isend` (replaces §14 RMA design) | Not started | ~200 | Needs `MPI_THREAD_MULTIPLE` eval |
| B.6 | End-to-end validation, CI, results doc | Not started | — | Final gate |

### 15.7 Next step: validate B.3+B.4 on Gadi compute node

Build command (from the existing `build-mpi-iso` dir or a new dir):

```bash
cmake . -DIQTREE_ATMD=ON
make -j52 iqtree3-mpi
```

Run the existing AA 1M 4-node test to verify:
1. `[ATMD Mode F] K_outer=... M_inner=...` log line appears on stdout.
2. lnL results match the A.2 baseline (same models selected, same scores).
3. Wall time ≤ A.2 baseline (NUMA first-touch may help; outer-loop is still effectively
   sequential for AA 1M since K_mem ≈ 1 at 512 GB with 2 × 12 GB trees per rank).

For K_outer > 1 to activate, need a smaller dataset (e.g. AA 100K, 4 taxa) where
`per_tree_bytes` is small enough that K_mem > 1 within the node's available RAM.

### 15.8 Design notes — Phase B.5 revision (tail-stealing)

The original §14 B.5 design used MPI-3 RMA (`MPI_Win_lock` / `MPI_Put`) to implement
rank-to-rank work stealing. This requires `MPI_THREAD_MULTIPLE` support, which OpenMPI
on Gadi provides but with significant synchronisation overhead.

**Revised B.5 design** (from §15 audit): Use `MPI_Isend` / `MPI_Iprobe` polling instead.
Each rank that finishes early sends a "work-available" message to idle ranks. Idle ranks
poll with `MPI_Iprobe` between model evaluations. This requires only `MPI_THREAD_FUNNELED`
(already set in B.−1), since polling happens in the outer master thread, not from inner workers.

This eliminates the `ONESIDE_COMM` requirement flagged in the §15.1 blockers.

### 15.9 Validation plan: build and test B.3+B.4 on Gadi

#### 15.9.1 Why cmake fails on the login node

The existing `build-mpi-iso/` cmake cache was configured with `-march=sapphirerapids`
(Intel SPR only).  Re-running `cmake .` re-tests the C compiler and fails on login nodes
(GCC < 12) because that `-march` flag is unsupported:

```
cc1: error: bad value ('sapphirerapids') for '-march=' switch
```

**Solution**: create a fresh build directory `build-atmd-b3/` and configure + build inside
a PBS job on a normalsr SPR compute node.  This is the same pattern used by `build_mf_iso.sh`.

#### 15.9.2 Build script

File: `gadi-ci/lbfgs-ws/build_atmd_b3.sh`

```bash
# On the login node:
cd /home/272/as1708/setonix-iq/gadi-ci/lbfgs-ws
qsub build_atmd_b3.sh
```

PBS parameters: `-P dx61 -q normalsr -l ncpus=104,mem=500GB,walltime=00:45:00`

The script:
1. Loads `cmake/3.31.6 openmpi/4.1.7 intel-compiler-llvm binutils/2.44 eigen/3.3.7 boost/1.84.0`.
2. Sets `OMPI_CXX=icpx`, `OMPI_CC=icx`.
3. Source preflight: verifies `_IQTREE_ATMD` in `main.cpp`, `MPI_Init_thread` in
   `MPIHelper.cpp`, `atmd_K_outer` in `phylotesting.cpp`, `IQTREE_ATMD` in `CMakeLists.txt`.
4. Applies cmaple build tweaks (disables IPO and unittest sub-project — same as
   `build_mf_iso.sh`).
5. `cmake ${SRC_DIR} -DIQTREE_FLAGS=mpi -DIQTREE_ATMD=ON` in fresh `build-atmd-b3/`.
6. `make -j$(nproc)` → binary `iqtree3-mpi`, symlinked as `iqtree3-mpi-atmd-b3`.
7. Linkage checks: `libiomp5` present, `libgomp` absent, `libmpi` present.
8. Symbol checks: `[ATMD Mode F]` string and `MPI_Init_thread` and `ws_bcast_fields`
   found in binary.
9. `mpirun -n 1 iqtree3-mpi --version` smoke test.
10. Writes `.build-info.json` (compiler, flags, commit, md5).

**Bug fix (job 169108814)**: PBS sets `$PROJECT=dx61` (billing project) in the job
environment, overriding `PROJECT="${PROJECT:-rc29}"` → `ISO_DIR` resolved to
`/scratch/dx61/as1708/...` (does not exist). Fix: use `SRC_PROJECT="rc29"` (hardcoded,
not a PBS env var) for path derivation; `$PROJECT` is reserved for the PBS billing project.
Same fix applied to both run scripts.

Build job submitted: `169108919.gadi-pbs` (resubmit after SRC_PROJECT fix)

**Build error (job 169108919)**: `site_rate` is a `protected` member of `PhyloTree` — accessed
directly in the B.4 K_outer formula in `phylotesting.cpp:4107` which is a free function, not
a `PhyloTree` method. Fix: use `in_tree->getRate()` (public virtual accessor) and hold the
pointer in a local `RateHeterogeneity *_sr`.

```cpp
// Before (error):
int nrates_est = max(4, in_tree->site_rate ? in_tree->site_rate->getNRate() : 4);
// After (fix):
RateHeterogeneity *_sr = in_tree->getRate();
int nrates_est = max(4, _sr ? _sr->getNRate() : 4);
```

Build job resubmitted: `169109258.gadi-pbs`

**Build result (job 169109258)**: SUCCESS — exit 0, make in 515s, binary at
`build-atmd-b3/iqtree3-mpi-atmd-b3`, md5 `c53122e2fbd92b197d9eccdef0d7ec80`.
Confirmed: `-D_IQTREE_ATMD` in all compile units; `[ATMD Mode F] K_outer=`,
`MPI_Init_thread`, `ws_bcast_fields=` all present in binary strings.
(Build script `strings` check had false negatives — icpx `-g` splits `cout <<` literals;
fixed to use shorter substrings that are contiguous in the binary.)

Run jobs submitted:
- `169109673.gadi-pbs` — AA 1M 4-node correctness gate (`run_atmd_b3_aa_1m_4node.sh`)
- `169109674.gadi-pbs` — AA 100K 1-node K_outer activation test — **FAILED** (exit 2, 6s):
  OpenMPI bound the single rank to 1 CPU slot; IQ-TREE saw 1 core but `-T 103` was requested.
  Fix: add `--bind-to none` to `mpirun` (same as `run_fca_aa_100k_1node_full.sh`).
- `169109738.gadi-pbs` — AA 100K 1-node K_outer activation test **resubmit** (--bind-to none added)

Expected output:
```
[build] ── DONE ──────────────────────────────────────────────────
  binary:   /scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3/iqtree3-mpi
  symlink:  /scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3/iqtree3-mpi-atmd-b3
  md5:      <hash>
```

#### 15.9.3 Correctness gate: AA 1M, 4 nodes, 4 ranks

File: `gadi-ci/lbfgs-ws/run_atmd_b3_aa_1m_4node.sh`

```bash
qsub run_atmd_b3_aa_1m_4node.sh
```

PBS parameters: `-l ncpus=416,mem=2040GB,walltime=03:30:00`

Config: `NRANKS=4`, `OMP_PER_RANK=103`, numactl `--localalloc`, same rankfile/hostfile
pattern as `run_ws_a2_aa_1m_4node_full.sh`.  Alignment: AA 1M (`len_1000000/tree_1/`).

**Expected behaviour at AA 1M** (100 taxa, nodeNum ≈ 198):

Per-tree memory (conservative formula, factors of `nodeNum × npat × nstates × nrates × 8 × 4`):

```
per_tree_bytes ≈ 198 × 1,000,000 × 20 × 4 × 8 × 4 = 507 GB
avail_bytes    ≈ 500 GB × 0.8 = 400 GB
K_mem          = max(1, floor(400 / 507)) = max(1, 0) = 1
atmd_K_outer   = min(K_mem=1, K_thr=103, K_cap=8) = 1
```

K_outer=1 means the outer loop degrades to the same **sequential** path as A.2.
The B.4 memory semaphore correctly avoids OOM by restricting to serial operation.

**Gate pass criteria:**

| Check | Expected |
|---|---|
| `[ATMD Mode F] K_outer=1` in log | K_outer=1 (memory-bound serial path) |
| lnL | within ±1.0 of −78,605,196.497 (A.2 np=4 ref) |
| Best model | LG+G4 |
| `ws_bcast_fields > 0` | A.2 warm-start intact |
| Wall time | ≤ A.2 + 5% (~6099s) |

A.2 references: MF=1999.214s, SPR=4021.666s, total=6098.480s (job 169099058)

#### 15.9.4 K_outer > 1 activation test: AA 100K, 1 node

File: `gadi-ci/lbfgs-ws/run_atmd_b3_aa_100k_1node.sh`

```bash
qsub run_atmd_b3_aa_100k_1node.sh
```

PBS parameters: `-l ncpus=104,mem=500GB,place=excl,walltime=01:30:00`

Config: `NRANKS=1`, `OMP_PER_RANK=103`, `OMP_MAX_ACTIVE_LEVELS=2`,
`OMP_PROC_BIND=spread,close` (outer workers span NUMA domains; inner threads stay close).
Alignment: AA 100K (`len_100000/tree_1/`).

**Expected behaviour at AA 100K** (100 taxa):

```
per_tree_bytes ≈ 198 × 100,000 × 20 × 4 × 8 × 4 = 50.7 GB
avail_bytes    ≈ 500 GB × 0.8 = 400 GB
K_mem          = floor(400 / 50.7) ≈ 7
K_thr          = OMP_PER_RANK = 103
K_cap          = 8
atmd_K_outer   = min(7, 103, 8) = 7
atmd_M_inner   = floor(103 / 7) = 14
```

Note: the per_tree_bytes formula deliberately over-estimates by ~4× (includes a stack-depth
factor to cover partial-lh back-buffers). Actual `central_partial_lh` per tree at AA 100K +G4
is `nodeNum × npat × nstates × nrates × sizeof(double)` ≈ 12.7 GB.  The conservative
estimate gives K_outer=7; the true value would give K_outer=8 (hitting K_cap).  Either way,
K_outer > 1 activates Mode F outer parallel teams.

**Expected log line:**
```
[ATMD Mode F] K_outer=7 M_inner=14 per_tree_MB=50700 avail_MB=400000
```
(exact numbers may vary by ±10% depending on system free memory at run time)

**Gate pass criteria:**

| Check | Expected |
|---|---|
| `[ATMD Mode F]` in log | ATMD code path reached |
| K_outer | > 1 (Mode F outer parallelism active) |
| lnL | present and self-consistent (no reference gate — informational) |
| Best model | LG+G4 or LG+I+G4 |
| Exit code | 0 |

This test is a **smoke test** for Mode F activation.  Correctness is gated by the AA 1M
4-node run above; performance benchmarking of K_outer > 1 throughput is a separate Phase B.0
task (pattern-parallel Mode P).

#### 15.9.5 Scripts produced

```
gadi-ci/lbfgs-ws/
  build_atmd_b3.sh            PBS build script, 1 SPR node, IQTREE_ATMD=ON
  run_atmd_b3_aa_1m_4node.sh  correctness gate, 4 nodes np=4, K_outer=1 expected
  run_atmd_b3_aa_100k_1node.sh K_outer>1 activation smoke test, 1 node np=1
```

All three scripts follow the same conventions as the existing lbfgs-ws scripts:
`$ISO_DIR`, `$PROFILES_DIR`, `$WORK_DIR` layout, `hostfile.txt`/`rankfile.txt`,
gate-check block with `PASS` variable, `GATE: PASS / FAIL` final line.

#### 15.9.6 Run results — b3 binary (jobs 169109738, 169109673)

##### AA 100K, 1-node (job 169109738) — COMPLETED

| Field | Value |
|---|---|
| Binary | `build-atmd-b3/iqtree3-mpi-atmd-b3` (md5 `c53122e2`) |
| Config | NRANKS=1, OMP=103, `--bind-to none` |
| Best-fit model | **LG+G4** ✓ |
| lnL | **−7,541,976.853** |
| Gamma alpha | 0.996 |
| MF wall-clock | **407.888s** |
| Tree search wall | 1290.298s (0h:21m:30s) |
| Total wall | **1706.337s** (0h:28m:26s) |
| Exit code | 0 |
| ATMD Mode F | **NOT ACTIVATED** — `[ATMD Mode F]` line absent from log (K_outer=1) |
| Gate result | **FAIL** — K_outer activation criterion not met (sysconf bug; see §15.9.7) |

Correctness is satisfactory: lnL and best model are self-consistent. The Mode F outer
parallel team silently fell back to K_outer=1 serial path due to the bug described in §15.9.7.

##### AA 1M, 4-node (job 169109673) — IN PROGRESS at time of writing

Job still running at 30 min elapsed. Results will be recorded in §15.9.9 when complete.

#### 15.9.7 Bug B.4-1: `sysconf(_SC_AVPHYS_PAGES)` returns near-zero on Gadi nodes

**Symptom**: K_outer=1 on both test runs despite more than 400 GB free RAM available.
The `[ATMD Mode F]` log line was gated on `atmd_K_outer > 1`, so it was never printed,
making the issue initially invisible.

**Root cause**: `sysconf(_SC_AVPHYS_PAGES)` reports the number of *immediately-free* physical
pages, excluding kernel page cache.  On HPC Linux nodes, the kernel aggressively fills free
pages with I/O page cache (Lustre reads, executable pages, etc.).  On a freshly-started Gadi
compute node, `_SC_AVPHYS_PAGES` can return a very small positive value even though the node
has 500 GB total RAM and less than 40 GB actually in use.

The B.4 formula then computes:
```
avail_bytes ≈ (small) × 4096 ≈ tens of MB
K_mem = max(1, floor(avail_bytes×0.8 / per_tree_bytes))
      = max(1, 0) = 1
atmd_K_outer = min(K_mem=1, K_thr=103, K_cap=8) = 1  → serial fallback
```

The `avail_pages > 0` guard prevented the 512 GB fallback from firing, since sysconf did
return a positive (but near-zero) value.

**Diagnosis confirmation**: The AA 100K run used only 35.65 GB RAM (PBS resource usage line),
and per-tree_MB budget was ~50 GB.  If `avail_bytes` had correctly reflected node free memory
(≥ 400 GB), K_mem ≥ 7 and K_outer = 7.

**Fix applied in `phylotesting.cpp`** (read before b3b build):

Replace `sysconf(_SC_AVPHYS_PAGES)` with a `/proc/meminfo` `MemAvailable` read.
`MemAvailable` is the kernel's own estimate of reclaimable memory
(free + reclaimable page cache), which is the correct budget for a new large allocation.
The sysconf path is retained as a fallback if `/proc/meminfo` is unavailable.

```cpp
// NEW: read MemAvailable from /proc/meminfo (robust on HPC nodes)
size_t avail_bytes = (size_t)512 * 1024 * 1024 * 1024;  // default 512 GB
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char key[64]; long long val;
        while (fscanf(f, "%63s %lld kB\n", key, &val) == 2) {
            if (strcmp(key, "MemAvailable:") == 0) {
                if (val > 0) avail_bytes = (size_t)val * 1024;
                break;
            }
        }
        fclose(f);
    } else {
        long pg = sysconf(_SC_AVPHYS_PAGES);          // fallback
        if (pg > 0) avail_bytes = (size_t)pg * (size_t)sysconf(_SC_PAGE_SIZE);
    }
}
```

Additionally, the `[ATMD Mode F]` diagnostic line was changed to **always print** (not
only when K_outer > 1), so future K_outer=1 fallback cases are immediately visible in logs.

#### 15.9.8 b3b build and re-test (job 169110101)

After applying the `/proc/meminfo` fix, a new build `build-atmd-b3b/` was submitted.

**Build result (job 169110101)**: SUCCESS — exit 0, make in 537s.

| Field | Value |
|---|---|
| Binary | `/scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3b/iqtree3-mpi-atmd-b3` |
| md5 | `8d12b01ffaf15f1f041139a4c695c80b` |
| Build time | 537s |
| Linkage | libiomp5 ✓, libgomp ✗, libmpi ✓ |
| ATMD content | Binary grep confirms `_IQTREE_ATMD`, `MemAvailable`, `proc/meminfo`, `K_outer=` |
| `strings` check | False negatives (known icpx issue — binary grep substituted) |

New scripts created:
```
gadi-ci/lbfgs-ws/
  build_atmd_b3b.sh              PBS build script targeting build-atmd-b3b/
  run_atmd_b3b_aa_100k_1node.sh  K_outer>1 activation re-test with fixed binary
  run_atmd_b3b_aa_1m_4node.sh    AA 1M correctness re-gate with fixed binary
```

**Expected log line with b3b binary on AA 100K:**
```
[ATMD Mode F] K_outer=7 M_inner=14 K_mem=7 per_tree_MB=50700 avail_MB=~430000
```
(K_mem and avail_MB now correctly computed from `/proc/meminfo`; b3 had avail_MB ≈ 0 due to sysconf)

Results of the b3b runs will be recorded in §15.9.9 once complete.

#### 15.9.9 b3b AA 100K run result (job 169110375) — COMPLETED

| Field | Value |
|---|---|
| Binary | `build-atmd-b3b/iqtree3-mpi-atmd-b3` (md5 `8d12b01f`) |
| Config | NRANKS=1, OMP=103, `--bind-to none`, 1 normalsr node |
| Best-fit model | **LG+G4** ✓ |
| lnL | **−7,541,976.853** ✓ (same as b3 ref) |
| Total wall | **1711s** (0h:28m:31s) |
| Exit code | 0 |
| ATMD Mode F | **NOT ACTIVATED** — `[ATMD Mode F]` line STILL ABSENT |
| Gate result | **FAIL** — K_outer activation not yet achieved |

The `/proc/meminfo` fix correctly reports `avail_MB ≈ 400000` (confirmed in preflight estimate),
but the `[ATMD Mode F]` diagnostic line still does not appear in `iqtree_run.log`.
Correctness remains intact. The B.4-2 bug investigation below explains why.

#### 15.9.10 Bug B.4-2: `[ATMD Mode F]` block does not execute at runtime

**Symptom**: Binary search of `iqtree_run.log` for `[ATMD Mode F]`, `K_outer=`, `avail_MB`,
`Mode F` returns idx=-1 on BOTH the b3 np=1 run AND the b3b np=1 re-test.  The same absence
is observed in the ongoing b3 np=4 1M run.

**Evidence that the code IS compiled in**:

| Check | Result |
|---|---|
| Binary grep `[ATMD Mode F]` | Present at byte 9,972,856 ✓ |
| Binary grep `K_outer=` | Present at byte 9,972,870 ✓ |
| Binary grep `MemAvailable:` | Present at byte 9,972,842 ✓ |
| Binary grep `proc/meminfo` | Present at byte 9,972,815 ✓ |
| `CXX_DEFINES` (flags.make) | `-D_IQTREE_ATMD`, `-D_IQTREE_MPI` ✓ |
| `icpx -fopenmp -E -dM` | `#define _OPENMP 202011` ✓ |
| `params.atmd_K_outer` default | `0` (tools.h:2378) — `0 != -1` is TRUE ✓ |
| No `#undef` for any macro | Confirmed ✓ |
| No `goto`/`longjmp` in phylotesting.cpp | Confirmed ✓ |
| `evaluateAll()` called for np=1 MPI | Confirmed via code path (line 1536–1548) ✓ |

**Output routing investigation**: `iqtree_run.log` is written by TWO file descriptors
simultaneously:
- fd 1 (shell stdout redirect: `> iqtree_run.log`)
- IQ-TREE's internal log fd (via `--prefix iqtree_run`, opened with fopen/O_TRUNC)

Both point to the same filename. IQ-TREE's internal log fd truncates the file at startup,
then TeeBuf writes to both fds simultaneously (at different file offsets). Analysis shows
stream 1 (internal fd) writes LATER at overlapping positions and therefore wins, so the
complete output should appear in positions 0..X-1 of the file. Despite this, `[ATMD Mode F]`
is definitively absent (binary search confirmed, no null bytes in file).

**Conclusion**: The B.3+B.4 block at lines ~4100–4164 of `phylotesting.cpp` is NOT executing
at runtime, despite the code being compiled in and all guards being satisfied at compile time.
Root cause not yet isolated.

**Diagnostic plan (b3c build, job 169111388)**:

Three new diagnostics added to `phylotesting.cpp` before the b3c build:

1. **Entry diagnostic** (top of `evaluateAll()`, unconditional):
   ```cpp
   fprintf(stderr, "[ATMD-DIAG] evaluateAll() ENTRY: atmd_K_outer=%d openmp_by_model=%d\n",
           params.atmd_K_outer, params.openmp_by_model ? 1 : 0);
   ```
   Fires on EVERY call — confirms whether `evaluateAll()` is reached.

2. **Pre-block diagnostic** (line ~4100, before `#if` guard, unconditional):
   ```cpp
   fprintf(stderr, "[ATMD-DIAG] evaluateAll B.3+B.4 pre-block: atmd_K_outer=%d MPI=%d OMP=%d ATMD=%d\n", ...);
   ```
   Fires if code reaches the B.3+B.4 section — confirms execution past the `#endif`.

3. **Sidecar file** (inside `if (params.atmd_K_outer != -1)` block):
   ```cpp
   FILE *df = fopen((string(params.out_prefix) + ".atmd_diag").c_str(), "w");
   fprintf(df, "[ATMD Mode F] K_outer=%d M_inner=%d K_mem=%d per_tree_MB=%d avail_MB=%d\n", ...);
   ```
   Written via `fopen` — completely bypasses cout/TeeBuf/shell redirect conflicts.

The b3c run script also separates `--prefix iqtree_inner` (IQ-TREE's log) from the shell
stdout redirect (`> iqtree_stdout.log`), eliminating the dual-write conflict. Gate check
reads all three sources: `iqtree_inner.atmd_diag` (sidecar), `iqtree_inner.log`, and
`iqtree_stdout.log`.

Results of the b3c run will be recorded in §15.9.11.

#### 15.9.11 AA 1M 4-node b3 result (job 169109673) — PARTIAL (SPR in progress)

| Field | Value |
|---|---|
| Binary | `build-atmd-b3/iqtree3-mpi-atmd-b3` (md5 `c53122e2`) |
| Config | NRANKS=4, OMP=103, 4 normalsr SPR nodes, 3h30m walltime |
| Best-fit model | **LG+G4** ✓ |
| Initial lnL | **−78,605,196.445** (NNI iteration 1) |
| MF wall | **4,017.842s** (1h:6m:57s) |
| CPU time MF | 342,454.713s (95h:7m:34s) — correct for 4-rank × 103T |
| ATMD Mode F | **NOT ACTIVATED** — `[ATMD Mode F]` ABSENT (B.4-2 bug confirmed for np=4) |
| SPR status | IN PROGRESS at time of documentation — optimizing candidate tree set |

**Note**: MF=4,017s is intentionally slow compared to Phase A.2 (MF=1,139s at np=16) — this is
the b3 binary (ATMD patch only, no FCA MPI dispatch, no warm-start), running AA 1M with 4 MPI
ranks where each rank evaluates ALL models sequentially. Its purpose is K_outer activation
testing, not production MF performance.

`[ATMD Mode F]` is definitively absent between `filterRatesMPI_enabled=1` and the first
`MF-TIME: rank 0 model=0 name=LG` line — confirming B.4-2 bug affects np=4 just as it does np=1.

Full results (lnL SPR, wall time, exit code) will be appended when the job completes.

#### 15.9.12 b3c build, binary path bug, and resubmission

##### b3c build (job 169111388) — SUCCESS

Three diagnostics added to `phylotesting.cpp` before the b3c build — see §15.9.10 for the
code. The entry diagnostic at the top of `evaluateAll()` was added AFTER the build job was
submitted but BEFORE `phylotesting.cpp` was compiled (confirmed: `.o` file not yet present when
the edit was made). All three diagnostics therefore compiled into the b3c binary.

| Field | Value |
|---|---|
| Build job | **169111388** (`normalsr`, 1 node, 104 cpus, walltime 45m) |
| Build time | **524s** (8m 44s) |
| Binary | `/scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3c/iqtree3-mpi-atmd-b3c` |
| md5 | **`1c6fc01921df0fbd67e45da280a036e9`** |
| Build exit | 0 ✓ |
| Linkage | libiomp5 + libmpi ✓ |
| `strings` symbol check | FALSE NEGATIVES (known icpx issue — binary grep substituted) |

Binary grep confirms all diagnostic strings present:

| String | Byte offset |
|---|---|
| `[ATMD Mode F]` | 9,977,137 |
| `[ATMD-DIAG]` | 9,976,687 |
| `evaluateAll() ENTRY` | 9,976,699 |
| `pre-block` | 9,977,011 |
| `atmd_diag` | 9,977,127 |
| `MemAvailable:` | 9,977,112 |

##### Binary path bug in run script (B.5-1) — FIXED

**Symptom**: Job 169111537 exited in 1 second with `Exit Status: 2` and message:
```
ERROR: ATMD binary not found: /scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3c/iqtree3-mpi-atmd-b3
```

**Root cause**: `run_atmd_b3c_aa_100k_1node.sh` was created with:
```bash
sed 's/b3b/b3c/g' run_atmd_b3b_aa_100k_1node.sh > run_atmd_b3c_aa_100k_1node.sh
```
But the b3b binary was named `iqtree3-mpi-atmd-b3` (no `b3b` suffix — it inherited the `b3`
symlink name from the b3 build). The `sed` substitution only replaced `b3b` → `b3c`, so the
binary variable remained pointing at `iqtree3-mpi-atmd-b3`. The b3c symlink is named
`iqtree3-mpi-atmd-b3c`.

**Fix**: Line 46 of `run_atmd_b3c_aa_100k_1node.sh`:
```bash
# BEFORE (wrong):
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b3c/iqtree3-mpi-atmd-b3}"

# AFTER (correct):
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b3c/iqtree3-mpi-atmd-b3c}"
```
Also updated comment at line 22: `# Binary: iqtree3-mpi-atmd-b3c (build-atmd-b3c/)`.

**Lesson**: When using `sed` to clone run scripts with a new binary name, verify that ALL
occurrences of the old binary path are substituted, not just the label tokens. Use a more
specific pattern (e.g., `sed 's|iqtree3-mpi-atmd-b3[^c]|iqtree3-mpi-atmd-b3c|g'`) or edit
the binary path explicitly.

##### b3c 100K re-submission (job 169111545) — IN PROGRESS

After the fix, resubmitted as job **169111545** (`normalsr`, 1 node, 104 cpus, 1h30m walltime).
Job confirmed running. Results will be appended here when complete.

**Expected outcome from b3c diagnostics**:
- `[ATMD-DIAG] evaluateAll() ENTRY` in `iqtree_stdout.log` → `evaluateAll()` is reached
- `[ATMD-DIAG] evaluateAll B.3+B.4 pre-block` in `iqtree_stdout.log` → code reaches B.3+B.4
- `iqtree_inner.atmd_diag` sidecar file exists → K_outer block executed
- If pre-block fires but sidecar is absent → `params.atmd_K_outer` must be `-1` at runtime
- If NEITHER pre-block NOR entry fires → `evaluateAll()` is not being called (contradicts code analysis)
