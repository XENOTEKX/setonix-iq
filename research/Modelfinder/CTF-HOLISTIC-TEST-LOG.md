# CTF Holistic Test-Log — Coarse-to-Fine ModelFinder: credibility, cost, and correctness

**Living document. Every number traces to a job ID + on-disk log. Started 2026-07-18.**

## 0. What CTF is, and what this log tests

CTF (`--ctf`) is a GPU ModelFinder mode: it **coarse-ranks all candidate models on a seeded site-subsample**
(`--ctf-subsample N`, default **5000**), then **refines the top-k on full data** (fixed coarse topology), picking the
min-BIC. It is an order of magnitude faster in ModelFinder walltime, but **stochastic** (the subsample is seeded) — which
is exactly why Minh (IQ-TREE author) asked us not to ship it: a subsample-driven model choice is hard for researchers to
trust. This log tests that concern head-on, holistically:

| # | Question | Metric | Job |
|---|---|---|---|
| A | Does CTF pick the **same model** as full deterministic `-m MFP`? | model-string match | h2h/realdata/sweep |
| B | At what **subsample size** does any over-fitting drift disappear, per dataset/seed? | first size with 3/3-seed match | **174119600** |
| C | **CTF-MF walltime cost** vs full MF, per dataset/size | CTF-MF s vs full-MF s | **174119600** |
| D | Does CTF's **final tree** match/beat deterministic? | RF to TRUE generating tree | **174118622** |
| E | **Seed robustness** — is CTF more seed-variable than DET's own search noise? | RF spread across seeds | **174118511** |

Datasets span **DNA/AA × simulated/real**: DNA-1M, AA-1M (sim, known true tree), avian-1M (real DNA), euk-22K (real AA).
All runs: `f3f7875f`, `-seed S -ninit 2 -optalg 2-BFGS -nt 12 -starttree PARS --jolt --gpu`.

---

## 1. Prior work (grounded, agent-audited 2026-07-18) — this sweep is NEW; prior work asked a DIFFERENT question

**The subsample-SIZE → final-model-MATCH experiment (question B) has never been run.** A staged script existed
(`gems_ctf_subsample_sweep.sh`) but produced no output/workdir — it was unexecuted, not prior art. So §3 below is genuinely new.

**What prior work DID measure — top-k RECALL vs subsample LENGTH, on AA-1M, under native BIC (a different question + a
different `+I` mechanism):** `research/Modelfinder/gpu/gpu-modelfinder-part10-subsample-sufficiency-hypothesis.md`
("PART X — subsample-sufficiency hypothesis", **job 170727768**, 2026-06-13, CPU reference binary). It swept subsample
length L ∈ {1K,2K,5K,10K,20K,50K,100K} × 3 seeds on AA-1M (`-m TEST`, target LG+G4 vs LG+I+G4) and measured **recall(top-3)
+ exact(rank-1)**, not final-model-match on the GPU pipeline. Result: **recall = exact = 1.00 at every L including 1000**
under *native* BIC; ΔBIC grows ~ln(L). But the *shipped projected*-BIC gate FAILED recall (§X.5.5, LG+G4 → projected rank 4,
`+R` promoted; jobs 170728179/170728182). Related recall sweeps (all top-k recall, sizes ≤5000, CPU, recall@3):
`run_ctf_phase0_recall_v100.sh`, `run_ctf_overfit_recall_sweep.sh`, `run_ctf_freerate_recall_sweep.sh`.

**Why part10 does NOT answer question B:** (1) it measured *recall of the candidate set*, not whether the *final selected
model* matches deterministic; (2) its `+I` story is a **boundary pinv→0 + projection-amplified `+R`** effect on AA-1M — a
*different* mechanism from the DNA-1M **coarse-topology-error** `+I` drift documented in §2 here; (3) it ran on the CPU
reference, not the shipped GPU `--ctf` pipeline. **⇒ we CITE and EXTEND part10, we do not restate it.** The one honest overlap:
part10 already showed native-BIC recall is robust down to L=1000 on AA-1M — consistent with an *expectation* that AA-1M may not
drift in §3 (agent also found the AA-1M native `-m MF` @5000 re-run picked LG+G4, no `+I`, job 170756438). **Only single-size
walltime points exist pre-today** (no cost-vs-size curve): DNA-1M `--ctf`@5000 CTF-MF **57.7s**; AA-1M `-m MF` native@5000 **767s**.
§3 produces the first cost-vs-size curve.

---

## 2. The `+I` over-fitting finding — DNA-1M (mechanism, SOLVED) — job 174116178

CTF@5000 on DNA-1M picked **F81+F+I+G4**; full deterministic picked **F81+F+G4** (a one-param `+I` difference).

**It is NOT a BIC failure — it is a topology-conditioning artifact.** At CTF's refine step (full data, topology fixed to
the 5000-site coarse tree): `F81+F+G4 logL=−59581615.658` vs `F81+F+I+G4 logL=−59581591.158` → `+I` bought a **real +24.50
nats** on that tree, which clears the one-param BIC penalty (`ln(1e6)/2 = 6.9` nats), so BIC *correctly* took it. **But those
24.5 nats were `+I` absorbing TOPOLOGY ERROR in the rough coarse tree** — once the full tree search corrects the topology,
**pinv collapses to 1e-6** and `+I` goes inert. The final tree-searched lnL is identical (−59208019.10 both), and CTF's BIC
ends up exactly `ln(1e6)=13.84` worse = **one dead parameter**. The deterministic run never trips because it selects models
on a full-1M-site initial tree. ⇒ the drift is structural to subsample-fixed-topology selection, amplified by the `+I`/`+G`
identifiability confound. **Speed:** CTF total 663.3s (3.57×) / CTF-MF 57.7s (**28× under** full MF 1617.9s).

---

## 3. Subsample-size vs drift-cure + ModelFinder cost — job 174119600 ⏳

Hypothesis (user): a bigger subsample → better coarse topology → less error for `+I` to absorb → drift disappears, cheaply.
Prediction (mechanistic): topology error ~`1/√sites`, so 5K→20K (4×) roughly halves it → `+I` gain ~24.5→~12 nats, still
above the 6.9 bar ⇒ **narrows but may not cure at 20K; likely needs 50–100K**. Matrix: `--ctf-subsample ∈ {5000,20000,50000,
100000}` (euk capped {5000,10000,20000}, 22462 total sites) × `--ctf-seed ∈ {1,2,3}`, `-n 0` (model-selection only).

### 3.1 DNA-1M — ✅ LANDED (174119600[0]) — **a bigger subsample does NOT cure the drift, and it is nearly FREE**

Deterministic target **F81+F+G4**. ⚠ The job's own `verdict` column is buggy (the sed parse leaves a trailing space so the
`winner == DET` test always failed ⇒ everything printed "DRIFT"); the **winner** column is correct and the table below is
re-derived whitespace-trimmed from `dna1m_matrix.csv`.

| subsample | seed 1 | seed 2 | seed 3 | **seeds matching det.** | mean CTF-MF | vs full MF 1617.9s |
|--:|---|---|---|:--:|--:|--:|
| **5,000** (default) | +I | +I | +I | **0 / 3** | 60.2s | **27×** |
| **20,000** | ✅ | ✅ | +I | **2 / 3** | 65.0s | **25×** |
| **50,000** | ✅ | ✅ | +I | **2 / 3** | 61.9s | **26×** |
| **100,000** | ✅ | +I | +I | **1 / 3** | 64.1s | **25×** |
| **200,000** (20% of sites, `ctfext` 174120538) | ✅ | +I | +I | **2 / 5**ᵉ | 85.0s | **19×** |
| **500,000** (50% of sites) | ✅ | +I | +I | **1 / 3** | 100.7s | **16×** |
| 🔴 **1,000,000 = THE FULL ALIGNMENT** | +I | +I | — | **0 / 2** | 157.1s | **10×** |

ᵉ *200K was run with 5 seeds (1–5): MATCH on s1, s4; DRIFT on s2, s3, s5.*

### 3.1a 🔴 CTF drifts at **100% of the sites** — so the subsample is not the cause. But the cause is NOT topology either (mechanism CORRECTED by red-team, 2026-07-18)

At `--ctf-subsample 1000000` on a 1,000,000-site alignment the coarse rank uses **every site** — there is no subsample and no
stochasticity in the site draw. **CTF still picked the spurious `F81+F+I+G4` on both seeds (0/2).** ⇒ **the `+I` drift is NOT a
sampling artifact.** That much survives.

**But my stated MECHANISM — "caused entirely by the fixed coarse TOPOLOGY" — is REFUTED, verified on disk:**

- **The coarse topology IS DET's topology.** `md5sum` of the two runs' `.parstree` files is **byte-identical** (`89eeb057…`):
  the CTF coarse pass at full subsample and the deterministic run build the **same** parsimony start tree. So "fixed *rough*
  coarse tree" is false at full subsample — there is no topology error to absorb.
- **The real differentiator is BRANCH-LENGTH UNDER-OPTIMISATION in the coarse pass.** On the *identical* model and *identical*
  topology, the coarse fit is far worse:

  | model (identical) · on identical `.parstree` | DET logL | CTF coarse logL | gap |
  |---|--:|--:|--:|
  | **F81+F+G4** | −59208167.970 | −59399863.014 | **191,695 nats worse** |

  The deterministic path runs a pre-ModelFinder branch-length optimisation (`Estimate model parameters (epsilon = 5.0)` then
  `1.0`, confirmed in its console); **the CTF coarse pass skips it** and jumps straight to `epsilon = 0.1` (confirmed). Every
  model is therefore scored on crude branch lengths and fits ~190K nats worse.
- **`+I` absorbs a sliver of THAT misfit.** On DET's well-optimised tree, `+I` adds **exactly 0** (F81+F+G4 = F81+F+I+G4 =
  −59208167.970). On CTF's under-optimised tree, `+I` buys **+9.4 nats** (−59399863.014 → −59399853.585) — a tiny fraction of
  the 191,695-nat deficit, but enough to clear the one-param BIC penalty and flip the selection. Same `+I`/`+G` confound as §2,
  but the misfit `+I` absorbs is **branch-length under-optimisation, not topology error.**

⇒ **REVISED CONCLUSION: the `+I` drift is neither a sampling artifact NOR a structural topology property — it is an
UNDER-OPTIMISATION artifact of the coarse pass, which is a FIXABLE software defect.** Adding the pre-MF branch-length
optimisation to the coarse tree (or relaxing its `brlen-maxiter`) may remove the drift outright. **This directly reverses my
"§6.1 B: structural, SETTLED, no size fixes it" verdict — it is not structural, it is a missing optimisation step.** ⏳ UNTESTED:
the one-flag experiment that would confirm it is `-m MF-all` (`score_diff_thres = -1`, `tools.cpp:3045`) which disables
`filterRates` — plus a coarse-pass with the pre-MF optimisation restored.

### 3.1b 🔴 RETRACTED — "a deterministic full-length CTF still gives ~10× MF"

**An earlier revision of this section claimed that `--ctf-subsample = full length` answers Minh's stochasticity objection while
keeping a ~10× ModelFinder speedup, "because CTF's speed comes from refining only the top-k, not from subsampling."
That claim is WITHDRAWN — it was an inference, never verified, and the logs contradict the stated mechanism.**

**What the logs actually say.** Both runs announce the *same* work:

```
CTF @ full subsample :  ModelFinder will test 968 DNA models (sample size: 1000000)
Deterministic -m MFP :  ModelFinder will test up to 968 DNA models (sample size: 1000000, epsilon: 0.100)
```

At `--ctf-subsample = nsite` the coarse pass evaluates **all 968 models on all 1,000,000 sites** — it reduces neither the model
count nor the site count. The top-k restriction only replaces the *final refinement* step, so it cannot explain a coarse pass
that is nominally identical work to full MF. **The stated mechanism was wrong.**

**The measured gap, decomposed (this part is real, the explanation is not settled):**

| | CPU time | wall | CPU/wall |
|---|--:|--:|--:|
| CTF @ full subsample (`ctfext` 174120538) | 605.6s | 149.5s | **4.05×** |
| Deterministic `-m MFP` (h2h 174115960) | 2513.9s | 1541.0s | **1.63×** |

- **~2.5× of the wall gap is the DET path parallelising badly** (CPU/wall 1.63 vs 4.05) — the already-documented DNA-MF
  serialization gap. That is a **defect in the deterministic path, not a CTF virtue**, and quoting it as a CTF speedup would be
  misleading.
- **A residual ~4× CPU gap is UNEXPLAINED.** 968 models × 1M sites in 605s CPU vs 2514s CPU for nominally equal work. Either the
  two paths do different per-model work, or the coarse `evaluateAll` takes a cheaper route. The textual hint: DET prints
  "**up to** 968 … epsilon: 0.100" (early termination + convergence threshold), CTF prints plainly "968".

**What survives from the full-subsample run** is the scientific result, which does not depend on any speed claim: **CTF still
picked the spurious `F81+F+I+G4` at 100% of the sites (0/2 seeds)** ⇒ the `+I` drift is not a sampling artifact.

### 3.1c ⭐ ROOT CAUSE FOUND IN SOURCE — the gap is **`evaluateAll` vs `test`**, and it may not be a CTF property at all

Chasing the unexplained gap led to a **dispatch in the deterministic path itself**. All of the following is
**ESTABLISHED FROM SOURCE** (`/scratch/rc29/as1708/iqtree3-jolt-merge`, HEAD `30c0faf9`) — read the lines, they are short:

```cpp
// main/phylotesting.cpp:2004-2009  — the -m MF / -m MFP dispatch
if (params.openmp_by_model)
    best_model = model_set.evaluateAll(params, &iqtree, model_info, models_block, params.num_threads, BRLEN_OPTIMIZE);
else
    best_model = model_set.test      (params, &iqtree, model_info, models_block, params.num_threads, BRLEN_OPTIMIZE);
```

| | `openmp_by_model` | evaluator | how it threads |
|---|---|---|---|
| `--thread-model` (`tools.cpp:4667`) | `true` | **`evaluateAll()`** (`phylotesting.cpp:4257`) | `#pragma omp parallel num_threads(N) proc_bind(spread)` — **across MODELS** |
| `--thread-site` (`tools.cpp:4672`) — **THE DEFAULT** (`tools.cpp:7676`) | `false` | `test()` (`phylotesting.cpp:3779`) | serial model chain + on-disk `checkpoint->dump()` |

**And CTF's coarse pass calls `evaluateAll()` DIRECTLY** (`phylotesting.cpp:1586`), bypassing that dispatch entirely — so it
gets the parallel evaluator *regardless of the flag*. Meanwhile every deterministic `-m MFP` run we have ever benchmarked took
`test()`, because `openmp_by_model` defaults to **false**.

**`test()`'s over-models loop is plainly serial** — `phylotesting.cpp:3906` `for (model = 0; model < size(); model++)`, with
**no `#pragma omp` anywhere in its body** (the only pragma in the whole function is one `omp critical`). Its parallelism is
site-level *inside* each model evaluation. That is exactly what the flag names mean: **`--thread-site` = parallelise within one
model over sites; `--thread-model` = parallelise across models.** Measured CPU/wall **1.63** (≈1.6 of 12 cores busy) vs **4.05**.

#### 🔴 Two of my own hypotheses about the mechanism were WRONG — both killed by source

| my hypothesis | verdict | why |
|---|---|---|
| "`test()` serialises on per-model on-disk `checkpoint->dump()`" | ❌ **DEAD** | `Checkpoint::dump()` (`utils/checkpoint.cpp:145-151`) returns immediately if `filename == ""` **and** is rate-limited by `dump_interval`. It is not a per-model disk write. |
| "`evaluateAll()` skips the R−1 warm-start chain" | ❌ **WRONG** | `restoreCheckpointRminus1` / `initFromCatMinusOne` live inside `CandidateModel::evaluate()` (`:2565`, `:2645`, `:2661`), which **both** paths call. `evaluateAll` inherits the chain verbatim. |

**The real asymmetry is what each path FEEDS BACK into the warm-start pool:**

```cpp
// test():4023-4024   — UNCONDITIONAL, every model
// "BQM 2024-06-22: save checkpoint for starting values of next model"
model_info.putSubCheckpoint(&out_model_info, "");

// evaluateAll():4664-4670 — ONLY when the model improved on the best score
if (best_score > at(model).getScore()) { ...
    // only update model_info with better model
    model_info.putSubCheckpoint(&out_model_info, ""); }
```

⇒ `evaluateAll`'s warm-start pool is **far sparser**, so inside the shared `evaluate()` the flag `prev_rate_present` is false
more often, which routes a model to a **single `optimizeParameters` call** (`:2627`) instead of the **up-to-5-step refinement
loop** (`:2650`). **That is a source-grounded route to the ~4× CPU gap — but note what it implies: `evaluateAll` may be cheaper
because it optimises `+R` models LESS THOROUGHLY.** If so this is not a free 10× — it is a speed/model-fit trade, and the code
itself anticipates the failure mode (`:2665-2668` `outWarning("Log-likelihood ... worse than ...")`).

**Supporting evidence that the two are selection-equivalent (not proof):** MPI builds **already force**
`params.openmp_by_model = true` unconditionally (`:1993`) with the in-tree comment *"Always use evaluateAll() in MPI builds so
np=1 and np=N … produce consistent best-fit model selection."* An assertion in a comment is not evidence, but it means the
authors already treat `evaluateAll` as selection-equivalent, and every MPI run we have ever done took that path.

**Why `evaluateAll` only reaches 4.05/12, not 12/12:** it has its own serialisation — the whole scoring/bookkeeping tail runs
under `#pragma omp critical` (`:4661`ff), `evaluate()` copies the entire checkpoint map under `critical`
(`:2436-2438` `{ local_in_info = in_model_info; }`), plus `critical(warm_start_lock)` at `:2398`/`:2740`.

⇒ **HYPOTHESIS: the "CTF speedup" at full subsample is not a CTF property — it is partly `evaluateAll` vs `test`.** But the
red-team decomposition shows the **~10× is CONFOUNDED and cannot be attributed to parallelism alone:**

| factor in the CTF-vs-DET MF gap | size | nature | verified? |
|---|--:|---|---|
| **`evaluateAll` parallelism** (across-model OMP vs serial `test`) | ~2.5× | real, exact, deterministic | wall/CPU ratios 4.05 vs 1.63 ✓ |
| **fewer models EXECUTED** (`filterRates` prunes the `+R` ladder harder on the under-optimised coarse landscape) | ~2.2× | **a DEFECT, not a feature** — CTF evaluates fewer models *because* its fits are 190K nats worse | DET executed **143** ✓; CTF count red-team-claimed **65**, NOT independently reproduced ⚠ |
| **cheaper per-model** (crude branch-length start + `brlen-maxiter=2`) | residual | also a consequence of the under-optimisation | inferred, not measured |

So **the honest, safe, deterministic win is only the ~2.5× parallelism term** (`--thread-model`, exact same model set, same
result). The rest of the "10×" is CTF doing **less and worse work** — the same under-optimisation that causes the `+I` drift
(§3.1a). **Plain `-m MF --thread-model` should give ≈2.5×, NOT 10×** — predict ~600–900s on DNA-1M, not ~150s. My earlier
"exact deterministic ~10× MF" framing was wrong for the same reason the `+I`-topology story was: I conflated a real parallelism
win with a pruning artifact.

🔬 **TEST RUNNING — job `174122266`** (`gems_threadmodel_probe.sh`, DNA-1M + AA-1M). Arms: **A** `-m MF -n 0` (today's default,
`test()`) vs **B** `-m MF -n 0 --thread-model` (`evaluateAll()`). Pre-registered falsifiable predictions, so the result cannot
be rationalised after the fact:

| | prediction | what a failure means |
|---|---|---|
| **P1 speed** | B's MF wall ≪ A's (nearer ~150–400s than ~1541s on DNA-1M) | B ≈ A ⇒ **hypothesis WRONG**, the gap is something else |
| **P2 same work** | both print the **same** model count (968 DNA) | different counts ⇒ comparison **void** |
| **P3 correctness** | both select the **same** best model | divergence ⇒ **not a free win**, a correctness risk |
| **P4 CPU** | if B's *CPU* is also ~4× lower, parallelism alone does NOT explain it ⇒ a **second mechanism** exists | — |

#### 🔴 CRITICAL COROLLARY (user-spotted) — this lever is `-nt`-BOUND, so it does NOT serve the nt1 / full-GPU direction

`phylotesting.cpp:4559` is `#pragma omp parallel num_threads(num_threads) proc_bind(spread)`, and `num_threads` is the function
parameter fed from `params.num_threads` = **`-nt`**. ⇒ at **`-nt 1` the parallel region opens with ONE thread and all 968 models
are evaluated one at a time — the entire advantage should vanish.**

Two consequences, both important and both cutting against earlier framing:

1. **CTF's "GPU speedup" at full subsample is largely a CPU-THREAD effect, not a GPU effect.** The 12 host threads are each
   *driving* GPU kernels for a different model concurrently. It is not evidence of GPU capability.
2. **"1 H200 + `-nt 1`" is the WORST case for this lever.** One host thread issuing 968 sequential model evaluations cannot
   keep an H200 busy — consistent with the measured ~87–91% serial-host-compute profile on DNA-1M. **Any "pure GPU, low
   thread" pitch cannot claim this speedup.**

🔬 **TEST RUNNING — job `174122291`** (DNA-1M, 2×2): `{--thread-site, --thread-model} × {nt12, nt1}`. Prediction: the
`--thread-model` advantage is large at nt12 and **collapses to ~nothing at nt1**. If it does NOT collapse, then something other
than across-model OpenMP is at work and this whole section needs revisiting.

#### 🔬 SAFETY TEST — job `174122292` (the cell that can actually kill the flag)

The speed probes use DNA-1M (`F81+F+G4`) and AA-1M (`LG+G4`) — **neither exercises the `+R` ladder**, which is precisely where
the sparser warm-start pool should do damage, and precisely what wins on our **real** data (avian `GTR+F+R6`, euk `LG+I+R9`).
`174122292` runs avian-1M + euk-22K, `--thread-site` vs `--thread-model`, and measures: same best model? same logL/BIC? and
**the count of the code's own `"worse than"` cold-start warnings** (`:2665-2668`) as a direct damage metric.
**A speed win on simulated DNA/AA means nothing if `+R` degrades on real data.**

### 3.1d 🔬 GPU per-model parallelism (user's idea) — the GPU is ALREADY serialized; the honest picture

The user asked whether per-model parallelism is worth doing on GPU (no subsampling). Source answer, verified:

- **It already runs** — CTF's coarse pass IS `evaluateAll` + `--jolt --gpu` across-model-parallel, so the combination is proven
  (no crash, 149.5s). A plain `-m MF --thread-model --jolt --gpu` is the same machinery without the subsample/top-k.
- **But the GPU is SERIALIZED by design** — `gpu_lnl_intree.cu:2648` `static std::mutex jolt_gpu_mtx` wraps the entire
  `gpu_jolt_optimize`. Its own comment (`:2642-2647`): the single GPU's `__constant__` symbols + static `DevBuf` pool are
  **process-global device state**, so *"JOLT models run one at a time on the GPU while the other threads keep optimising
  CPU-fallback (+I/+R/+FO) candidates."* ⇒ the across-model "parallelism" is **one GPU model at a time + CPU-fallback models
  filling the other threads** — NOT 12 models on the device at once.
- **True concurrent multi-model-on-device ("PHALANX grid.z", G.4.3)** is referenced in comments only; **grep finds no
  implementation.** Treat as unbuilt.
- **The `+I`/`+R` families — ✅ RESOLVED, and the premise was BACKWARDS.** I twice quoted stale text (first "kernels exist ⇒
  we built it", then "+R/+I decline to CPU" from `MODELFINDER-FULL-GPU-PLAN.md:383` + the `gpu_lnl_intree.cu:2646` comment —
  the user flagged BOTH as outdated). **Verified in the actual `f3f7875f` source (`phylotreegpu.cpp`, the merge tree HEAD
  `30c0faf9`):**
  - **`+R` (R2–R10) engages by default** — `:2205-2207` `JOLT_FREERATE_MAXCAT` `if (!e) return 10; // GRADUATED default-ON:
    R5-R10 engage on GPU`. Kill-switch is `JOLT_NO_FREERATE_HIGHK`, not an enabler.
  - **pure `+I` engages by default** — `:2265` declines only `if (getenv("JOLT_NO_PUREINVAR"))`; comment *"pure +I DEFAULT-ON"*.
  - **`+I+R` engages** (`:2210-2216` G.5.1d ladder 2b), `+I+G` engages (G.4.3b). The `JOLT_IR*` strings are kill-switches/probes
    on an **already-on** path, not enablers.
  - What still declines (NONE are in the default `-m MF` set): `+FO`/free-freq, median-gamma, user-fixed pinv, R>10, mixtures,
    tied-freq DNA, AA free-Q. Default DNA freqs `{FQ,F}` and AA `{"",+F}` are all fixed (`phylotesting.cpp:108,196`).
  ⇒ **~100% of default `-m MF` candidates ALREADY run on the (serialized) GPU. There is NO +I/+R CPU-fallback gap to close —
  the whole "route +I/+R to GPU" idea is already shipped.** 🔬 `174123768` (JOLT_DEBUG on f3f7875f) is the empirical confirmation,
  but the source is now unambiguous.
  - 🔴 **The real bottleneck is elsewhere:** the per-candidate **CPU self-check** — a fresh CPU `computeLikelihood()` after each
    GPU optimise (`phylotreegpu.cpp:2644`) — is **~68% of the DNA-1M `-m MF` host wall** (`:2582`, perf job 173929005). The
    across-model parallelism (CPU/wall 4.05) works by **overlapping those self-checks with the serialized GPU compute**, NOT by
    running models on CPU-fallback. That reframes the entire opportunity — see §3.1f.

**The honest ceiling (pre-registered before the probe lands):** because the GPU is one serialized device, GPU per-model
parallelism cannot add *compute* the way CPU per-model parallelism does (12 threads ≈ 12× on CPU). Its speedup is bounded by
GPU *utilisation* — overlapping CPU-fallback work and CPU-side model setup with the one serial GPU stream. So expect the GPU
`--thread-model` win to be **smaller** than the CPU one, and possibly capped below it. The tension to resolve: routing `+I`/`+R`
onto the GPU would raise coverage but pile more contention on the single `jolt_gpu_mtx` — which could make it *slower*, not
faster, unless PHALANX-style true batching lands. 🔬 Plan + red-team in progress.

### 3.1e caveats to hold until the probes land

⚠ (a) Across-model OpenMP means each thread carries its own tree/partial-likelihood
arrays — **memory scales with threads**, which is a plausible reason it is not the default; the CTF coarse pass did survive 12
threads × 1M sites inside 180 GB, which is evidence but not proof for the general case. (b) `test()`'s warm-start chain may be
load-bearing for *result quality*, not just speed — P3 tests exactly that. (c) **"CPU time" excludes GPU time**, so a CPU-time
comparison across paths with different GPU offload is not self-interpreting; the probe records GPU util for this reason.
(d) Prior work in this project found `evaluateAll` had **partition-specific** problems, so nothing here transfers to `-p` runs
without its own test.

*(✅ = picked F81+F+G4 = matches deterministic; "+I" = picked the spurious F81+F+I+G4.)*

**Two conclusions, both important:**

1. **COST: raising the subsample is essentially FREE — the user's intuition was right.** CTF-MF is **flat at ~60–65 s from 5K to
   100K** (a 20× larger subsample costs ~0), and stays **~25× faster than full MF** at every size. Reason: the coarse rank is not
   the bottleneck — the **top-3 full-data refine** dominates the CTF-MF wall, and that is independent of subsample size.
2. **EFFICACY: it does NOT cure the over-fitting.** 5K → 0/3; 20K → 2/3 (real improvement); 50K → 2/3; **100K → 1/3 (worse than
   20K)**. **Non-monotonic and seed-dependent — even at 100,000 sites (10% of the alignment) 2 of 3 seeds still pick the spurious
   `+I`.** ⇒ **The drift is STRUCTURAL, not a subsample-size artifact — you cannot buy your way out of it with a bigger subsample.**
   Mechanistically the selection sits on a knife-edge (the `+I`/`+G` confound + residual coarse-topology error), so *which* model
   wins is driven by *which sites the subsample happened to draw*. **That is precisely the stochastic-credibility objection Minh
   raised, and this data supports it.** (My prior prediction — "narrows at 20K, cured by 50–100K" — is **half refuted**: it narrows,
   but 50–100K does not reliably cure.) **⭐ 200,000 (20% of the whole alignment) is still only 1/3 — the curve has now been pushed
   40× above the default and has NOT converged.** Best match fraction anywhere on the curve is 20K's 2/3; 200K is *worse* than 20K.
   This is the strongest single refutation of "just raise the subsample" in this log.

### 3.2 Remaining cells

### 3.2a AA-1M — ✅ LANDED (174119600[1]) — **milder drift than DNA, and again NOT cured by a 4× bigger subsample**

Deterministic target **LG+G4**. *(Same trailing-space `verdict`-column bug; re-derived from the `winner` column.)*

| subsample | seed 1 | seed 2 | seed 3 | **seeds matching det.** | mean CTF-MF | vs full MF 2452.2s |
|--:|---|---|---|:--:|--:|--:|
| **5,000** (default) | +I | ✅ | ✅ | **2 / 3** | 330.8s | **7.4×** |
| **20,000** | +I | ✅ | ✅ | **2 / 3** | 355.0s | **6.9×** |

**Same shape as DNA, different severity.** AA drifts on the *same* `+I` axis (LG+**I**+G4 vs LG+G4) but only on seed 1, and a 4×
subsample changes nothing (2/3 → 2/3) — consistent with part10's finding that AA-1M native-BIC recall is robust down to L=1000.
**Note the cost ratio is much weaker than DNA's:** CTF-MF is only **~7× under** full MF here vs DNA's ~25×, because AA's top-3
full-data refine is far more expensive relative to the coarse rank. ⇒ CTF's headline speed claim is DNA-flavoured.

### 3.2b Remaining cells

| dataset | deterministic | seeds matching @ best size | mean CTF-MF | full-MF | *status* |
|---|---|---|--:|--:|---|
| avian-1M (real) | GTR+F+R6 | | | 775.8s | ⏳ 174119600[2] |
| euk-22K (real) | LG+I+R9 | | | 6534.8s | ⏳ 174119600[3] |

---

## 4. Does CTF give a BETTER (or equal) END TREE? — job 174118622 ⏳

Model-independent test (you cannot compare raw lnL across different selected models). **Ground truth (sim):** RF distance of
each method's final tree to the TRUE alisim generating tree (`tree_1.full.treefile`), lower=better. **Fair topology (all):**
re-score both final topologies under one common richer model (GTR+F+I+G4 / LG+I+G4). 8 cells DNA/AA{10K,100K,1M}+avian+euk,
DET vs CTF@5000.

⚠ **RF values MUST be read from the `.rfdist` matrix off-diagonal (`awk '/^Tree0/{print $NF}'`), NOT the in-script verdict
line** — the script's RF parser matched the `29` in the storage path `/scratch/rc29/…` from IQ-TREE's "printed to `<path>`"
line and printed a fake `29` for every cell. The `.rfdist` files are correct and durable; all RF below is re-extracted from them.

| cell | DET model | CTF model | RF(DET→true) | RF(CTF→true) | RF(DET,CTF) | read |
|---|---|---|--:|--:|--:|---|
| **dna10k** | F81+F+G4 | F81+F+G4 | **2** | **2** | **0** | ✅ identical tree, no drift, both 1 split from truth = EQUAL |
| **dna100k** | F81+F+G4 | F81+F+G4 | **0** | **0** | **0** | ✅ **both recover the TRUE tree exactly** = EQUAL |
| dna1m | | | | | | ⏳ [2] |
| aa10k | | | | | | ⏳ [3] |
| aa100k | | | | | | ⏳ [4] |
| aa1m | | | | | | ⏳ [5] |
| avian-1M | | | *(no true)* | *(no true)* | | ⏳ [6] |
| euk-22K | | | *(no true)* | *(no true)* | | ⏳ [7] |

Decisive 3 ways: CTF RF < DET → pro-CTF proof; = → credibility rescue (model wobble cosmetic, tree no-worse at 3.6×); > → shelve.
**dna10k = EQUAL** (identical tree); **dna100k = EQUAL and both hit the true tree exactly (RF=0 all three ways)**.
**2/2 cells so far land on "EQUAL" — no pro-CTF tree evidence has appeared, and §5.2's identical-lnL result says the same thing
independently. The "better end trees" case is not materialising; plan for the "no worse" framing, not the "better nats" one.** Note the DNA-10K **no-drift** vs DNA-1M **+I drift** contrast supports the scale-dependence:
at 10K the 5000-site subsample is 50% of sites (good coarse topology) vs 0.5% at 1M (topology error → drift). The subsample
sweep §3 quantifies this directly.

---

## 5. Seed robustness — job 174118511 — ✅ 7/8 CELLS LANDED (aa1m s3 still running)

CTF's stochasticity is the credibility crux. Extra seeds {2,3,4} both-arms on cheap sim cells (**DET's own search variance =
the fair baseline**), CTF-only {2,3} on expensive cells.

⚠ **RF was NOT captured** — the script's `rf2()` deletes its `.rfdist` temps, so the tree-distance measure is lost for this job.
The `.treefile`s survive and RF is recomputable. **The evidence below is therefore (i) selected-model stability across seeds and
(ii) final lnL, which where the model matches is the stronger statement anyway** (identical lnL ⇒ same optimum).

### 5.1 Model stability across seeds — **DET is stable in every cell; CTF is not, and the REAL datasets are the worst**

| cell | deterministic | DET seeds 2,3,4 | CTF seeds 2,3,4 | CTF match |
|---|---|---|---|:--:|
| dna10k | F81+F+G4 | F81+F+G4 ×3 | F81+F+G4 ×3 | ✅ **3/3** |
| **dna100k** | F81+F+G4 | F81+F+G4 ×3 | F81+F+G4 · **TPM2u+F+G4** · F81+F+G4 | ⚠ **2/3** |
| dna1m | F81+F+G4 | *(=main table)* | F81+F+**I**+G4 ×2 | ✗ **0/2** |
| aa10k | LG+G4 | LG+G4 ×3 | LG+G4 ×3 | ✅ **3/3** |
| aa100k | LG+G4 | LG+G4 ×3 | LG+G4 ×3 | ✅ **3/3** |
| aa1m | LG+G4 | *(=main table)* | LG+G4 (s2) · ⏳ s3 | ✅ 1/1 so far |
| **avian-1M (real)** | GTR+F+R6 | *(=§3 ref)* | GTR+F+**I+R2** ×2 | ✗ **0/2** |
| **euk-22K (real)** | LG+I+R9 | *(=§3 ref)* | LG+**F**+I+**G4** ×2 | ✗ **0/2** |

**Three reads.** (1) **The baseline is clean** — DET picked the identical model on all 3 seeds in all 4 both-arm cells, so every
CTF disagreement below is CTF's variance, not shared search noise. (2) **dna100k is the sharpest single datapoint in this log:**
CTF flipped F81+F+G4 → **TPM2u+F+G4** on seed 3 alone, in a cell where DET is rock-stable and where CTF matches on the other two
seeds — *the selected model is a function of which sites the subsample drew.* That is Minh's objection reproduced in one cell.
(3) **The real datasets drift hardest and differ in KIND, not just one nuisance parameter** — avian changes the rate model
(`+R6` → `+I+R2`), euk changes base frequencies *and* the rate model (`+I+R9` → `+F+I+G4`). The §2 DNA-1M `+I` story (one dead
parameter, cosmetic) does **not** generalise: on real data CTF returns a structurally different model.

### 5.2 Final lnL — where the model matches, CTF finds the SAME optimum, not a better one

| cell | DET lnL | CTF lnL | |
|---|--:|--:|---|
| aa10k | −807350.031281 | −807350.031281 | **identical, all digits** |
| aa100k | −7541976.852167 | −7541976.852167 | **identical, all digits** |
| dna100k (s2, s4) | −5692984.526136 | −5692984.526136 | **identical, all digits** |
| dna10k | −564208.774954 | −564208.774955 | identical to 1e−6 (12th digit) |
| dna1m | −59208019.10 | −59208019.103539 | identical |
| aa1m | −78605196.44 | −78605196.435066 | identical |

⇒ **question D ("does CTF give BETTER end trees?") is answered NO on this evidence.** CTF converges to the *same* optimum
wherever it selects the same model — it is not buying better nats, so **there is no "better nats" case to put to Minh.** The
honest positive framing is the weaker one: **CTF's tree is no WORSE**, so its model wobble is (on simulated data) cosmetic.
*(dna100k seed 3 is the exception worth naming: CTF's TPM2u+F+G4 reached −5692973.65, ~10.9 nats BETTER than F81+F+G4's
−5692984.53 — but that is a RICHER model, so it is a BIC question, not a free win, and it is exactly the drift case.)*

### 5.3 ⭐ NEW, unplanned finding — **CTF's speed advantage is SCALE-DEPENDENT and is a LOSS below ~1M sites**

Mean TOTAL walltime across seeds (full runs, tree search included):

| cell | DET | CTF | CTF vs DET |
|---|--:|--:|--:|
| dna10k | 112.1s | 110.7s | 1.01× — noise |
| dna100k | 129.2s | 131.1s | **1.01× SLOWER** |
| **aa10k** | 217.2s | 303.1s | **1.40× SLOWER** |
| aa100k | 512.2s | 466.6s | 1.10× faster |
| dna1m | 2144.8s | 669.5 / 690.2s | **3.1–3.2× faster** |
| aa1m | 4927.0s | 2997.7s | **1.64× faster** |

**CTF only pays for itself at ~1M sites.** Below that the top-3 full-data refine dominates the CTF-MF wall (the same mechanism
§3.1 found makes subsample size ~free), so the coarse pass is pure added cost — at **aa10k it is a net 1.40× LOSS**. This was
not a question we set out to ask and it materially narrows CTF's honest operating envelope: **the credibility risk is paid at
every scale, but the speed is only collected at the top end.**

---

## 6. Verdict

### 6.1 INTERIM verdict (2026-07-18, ~60% of cells landed) — **the evidence is currently AGAINST making a CTF case to Minh**

Marked interim: `ctfext` 174120538 (incl. the FULL-alignment decisive cell), `ctfholi` AA/avian/euk, and better-end-tree cells
3–7 are still outstanding and could move parts of this. Nothing landed so far points the other way, though.

| Q | answer on evidence to date |
|---|---|
| **A** — same model as deterministic? | **No, unreliably.** Matches on small/medium *simulated* cells; drifts on DNA-1M (0/3 @5000), and on **both real datasets** (avian `+R6`→`+I+R2`, euk `+I+R9`→`+F+I+G4`) — drifts in KIND, not just a nuisance parameter (§5.1). |
| **B** — subsample size that cures drift? | **No SIZE cures it** (DNA-1M 5K 0/3 … 200K 2/5 … FULL 0/2; AA-1M 5K→20K 2/3), because 🔴 **the cause is NOT the subsample and NOT the topology — it is BRANCH-LENGTH UNDER-OPTIMISATION in the coarse pass** (§3.1a, verified: identical `.parstree`, 191,695-nat fit gap on identical model, DET does an `eps 5.0→1.0` pre-opt the coarse pass skips). ⇒ **REVERSAL of my earlier "structural/unfixable" verdict — it is a FIXABLE optimisation defect** (⏳ untested: restore the pre-MF optimisation / `-m MF-all`). |
| **C** — CTF-MF cost vs full MF | Coarse pass is ~free on DNA (flat 60–80s, 5K→200K, ~20–27× under full MF) but only **~7× on AA** (§3.2a) — **and that is the wrong metric anyway.** On TOTAL walltime CTF only wins at ~1M sites and is a **1.40× LOSS at aa10k** (§5.3). |
| **D** — better end tree? | **No.** Identical lnL to all printed digits wherever the model matches (§5.2); RF=0 all-ways on both landed tree cells (§4). CTF finds the *same* optimum — **no "better nats" argument exists.** |
| **E** — seed robustness | **CTF is materially less stable than DET.** DET picked one model on 3/3 seeds in all 4 both-arm cells; CTF flipped model on **dna100k seed 3** in a cell where DET is rock-stable (§5.1). |

**The honest summary.** Minh's objection is **supported, not refuted, by our own data.** CTF's model selection is a function of
which sites the subsample drew (E), the drift cannot be bought away with a bigger subsample (B), it changes the model
*structurally* on real data (A), and it buys nothing in tree quality to trade against that (D). The one real benefit — speed —
**is only collected at ~1M+ sites** (C/§5.3), so the credibility cost is paid at every scale while the payoff is not.

**🔴 A "deterministic full-length CTF" was proposed here and is now RETRACTED — see §3.1b.** The idea was that
`--ctf-subsample = full length` removes the stochasticity while keeping a ~10× MF speedup. **The speedup claim does not hold
up:** at full length the coarse pass tests all **968 models on all 1M sites** (log-confirmed), so the stated mechanism
("speed comes from the top-k refine") was wrong, and ~2.5× of the observed wall gap is just the deterministic path
parallelising badly (CPU/wall 1.63 vs 4.05), which is a defect in *our* DET path rather than a CTF benefit. A ~4× CPU gap
remains unexplained. **Do not pitch this variant until §3.1b's open test is done.** (The stochasticity half of the idea is
still sound in principle — full-length CTF *is* deterministic — but with no established speed benefit there is nothing to
trade for the model drift, which survives at 100% of sites.)

**Recommendation:** do **not** build a "CTF gives better nats" case. If CTF is argued for at all, the defensible claim is narrow
and should be stated with its limits: *at ≥1M sites CTF reaches the same optimum (identical lnL, RF=0) at 1.6–3.2× lower
walltime, at the cost of a stochastic and sometimes structurally different model choice.* Keeping it **default-OFF is the right
posture** and matches Minh's guidance. The genuinely reusable win from this work is the **mechanism** (§2) — that subsample-fixed
topology lets `+I`/`+F` absorb topology error — which is a real methodological result worth writing up independently of shipping CTF.

### 6.2 Final verdict — ⏳ pending `ctfext` 174120538 (FULL-alignment cell), `ctfholi` AA/avian/euk, better-end-tree 3–7
