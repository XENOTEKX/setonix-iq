# ModelFinder Full-GPU Coverage — getting EVERY model family onto GPU (no `--ctf`)

**Status:** design + grounded evidence + test matrix. 2026-07-15. Author-run; assistant does NOT push GPU source.
**Companion:** the running log lives in `MODELFINDER-FULL-GPU-PLAN.md` §N/§N.6 (this doc is the standalone,
red-team-corrected distillation). **Every number here traces to a real job ID; refuted claims are kept as refuted.**

---

## 0. The one-line goal and the honest bound

Reduce full `-m MF/MFP` walltime **without `--ctf`** by running every candidate family's likelihood + self-check on
GPU. **Grounded bound:** this is a decisive win on **AA** (state-count lever) and a *correctness/coverage* win
everywhere; it is **NOT automatically the DNA throughput win** vs the competitor's whole-likelihood-resident port —
DNA is memory-latency-bound at 4 states and needs an architectural change (residency/fusion), not kernel tuning.

## 1. Grounded evidence base (measured, cited to jobs)

| finding | evidence | job |
|---|---|---|
| Host self-time = the CPU likelihood recompute: **DNA-1M 43.5%, AA-1M 69.5%** | perf `.data` top symbols (`computePartialLikelihoodSIMD` + `dotProductTriple` + `bufferSIMD`) | mfoffload 173815952 |
| Per-candidate **framework cost small** (no single symbol in top-15; distributed sum bounded only by pending mfhostattr) | perf top-15 has no eigendecompose/initModel/ctor | mfoffload 173815952 (+ J.4 pending) |
| GPU self-check (mirror) win is **ns-dependent: AA 7–11×, DNA ~1.3×** | DEVCHECK wall_cpu vs wall_gpu per candidate | mfdevcheck 173802323 |
| **+I family = 27 candidates / 32.6s CPU at DNA-1M, ALL declined** (mirror pinv-gate) — biggest un-offloaded block | DEVCHECK per-family bucket | mfdevcheck 173802323 |
| **State-count lever** explains DNA-loses/AA-wins: 4-state = low arithmetic intensity; tensor-core gains codon 3× / AA 2.3× / **nucleotide only 1.1×** | BEAGLE 3 (Ayres 2019); tensor-core BEAGLE v4 (Gangavarapu 2026) | literature |
| +R self-check divergence is the **+I(pinv) coupling** (pure +R rel 0; +I+R MISMATCH) | scab `[JOLT]` lines | scab 173825354 |

## 2. Per-family GPU status + fix (source-verified `iqtree3-mfdevcheck` @ a07f61be)

| family | blocker (file:line) | fix | effort / status |
|---|---|---|---|
| **+I / +I+G** self-check | mirror declines `pinv>0` `phylotreegpu.cpp:84`; the function-value path has no invariant term; `gpu_lnl_crosscheck` sig has no pinv | add `pinv·base_invar[ptn]` before the root `log`, guarded on `base_invar!=nullptr`; build `base_invar` (copy `:2301-2315`); upload to device in `gpu_lnl_crosscheck`; thread pinv/base_invar via optional default-arg | **SOUND but surface is 3 kernels not 1** (corrected): `launch_k1_node` dispatches to `k1_node` (generic `:87`) AND `k1_node_t<4>`/`k1_node_t<20>` (templated, 128-reg) — the term must be added to all three + the launcher + crosscheck + mirror. Default-arg keeps non-+I byte-identical (the gate). LOW-MED per edit, 6 coordinated edits |
| **+I+R** | **joltLnL vs CPU postorder off by constant ~10.0 nats, INVARIANT to pinv** (scab) → self-check NaN → full CPU | **ROOT CAUSE UNRESOLVED** — see §3. NOT the `:3128` normaliser (dead pure-+R path); NOT a missing invariant term (would scale with pinv). Needs a trace spike. | **SPIKE FIRST** |
| **C10–C60 profile mix** | correct already (mix mirror+optimizer, MEOW80 rel 2.46e-13); dispatch gated `JOLT_MIX_HOSTDRIVEN` OFF for latency `modelfactory.cpp:1606` | flip/validate the flag; measure host-driven latency | **dispatch test, LOW risk** |
| pure +R | works (rel 0) — mirror computes it; DEVUSE excludes by policy (`!freeRateOK`) pending real-+R cross-check | (optional) re-enable after avian cross-check | LOW |
| EX/UL rate-mix | rate-1 guard `:2981` (per-class tns≠1) | thread `tns[]` through 3 echild builders | host-math |
| LG4X/LG4M fused | decline `:2800`/`:212` (isFused) | new fused-diagonal kernel | new kernel |
| +FO free-freq | decline `:2127` (freeQok excludes FREQ_ESTIMATE) — *confirm in default set first* | plumb free-freq dims into LM Jacobian | MED |
| codon ns=61 / binary ns=2 / morph | decline `:2102`/`:80` (ns∉{4,20}) | new `<61>`/`<2>` templates | codon LARGE |
| +ASC | decline `:2108`; kernel excludes unobserved_ptns | GPU unobserved-lh + `−N·log(1−ΣP_unobs)` | MED kernel |
| median-γ | decline `:2189` | add median-cut rate vector | small |

## 3. RESOLVED (T2, irtrace 173843042): the +I+R ~10-nat is a CONSTANT freeRate∧optPinv joltLnL bug — NOT invariant, NOT write-back

The T2 trace decomposed the offset per candidate: `diff = joltLnL−cpuLnL` vs `invContrib = cpuLnL − cpuLnL@pinv0`:
```
LG+R2 pinv 0.0228: diff=10.003  invContrib=7930.5   diff/invContrib=0.0013   restore_rel=0
LG+R3 pinv 0.0099: diff=10.002  invContrib=1440.9   diff/invContrib=0.0069   restore_rel=0
LG+R4 pinv 2.8e-5: diff=10.003  invContrib=2.79     diff/invContrib=3.59     restore_rel=0
LG+R5 pinv 4e-6:   diff=10.004  invContrib=0.38     diff/invContrib=26.6     restore_rel=0
(JC+R2..R5 DNA identical: diff ~10.00 across pinv 0.018..0.00007)
```
**DECISIVE READ:** `invContrib` swings **0.38 → 7930** with pinv while `diff` never moves off **~10.00**. If the GPU were
missing the invariant term, `diff` would EQUAL `invContrib` — it does not. ⇒ **the GPU computes the invariant term
CORRECTLY** (rules out the T1 mirror-term hypothesis for +I+R), and **`restore_rel=0` rules out a stale-pinv write-back**
(the CPU self-check state is clean; `cpuLnL` is right). **The ~10.00 is a CONSTANT systematic error in the
`gpu_jolt_optimize` freeRate∧optPinv path** — present whenever pinv is ESTIMATED (`optPinv==1`) with free-rate,
**independent of the pinv value** (still ~10 at pinv=4e-6). Pure +R (`optPinv==0`) = rel 0; +I+G (`freeRate==0`) = clean;
only the *combination* breaks. **Consequences:**
- **T1's +I mirror term does NOT fix +I+R** (the invariant isn't the problem) — T1 is scoped to **+I / +I+G only**.
- **+I+R needs its own fix**: bisect the freeRate∧optPinv `joltLnL` to find the constant-~10 source (candidates: the
  `gaugeFix` mean-rate rescale with `f=(1−pinv)`, the `out_props` `RateFreeInvar` writeback gauge, or a constant added
  once in the optimizer objective). It is a REAL-DATA priority: the **avian selects GTR+F+I+R4** (ctfavab), and this bug
  forces that winner to full CPU (rtilegate G2: AA-1M LG+I+R4 didn't finish in 6h).
- Safe today: caught by the `:2589` self-check → CPU fallback (never wrong, just slow).

### 3a. NARROWED (T1 iplus 173879879 + source audit) — ⚠️ its conclusion is CORRECTED by §3b: the written-back PARAMS are the wrong side, not the reported number
> **Read §3b first.** This subsection's original conclusion ("params correct, only the reported number carries +10")
> is **BACKWARDS** and is what produced the harmful re-eval. mfgauge + JOB A/B/C (§3b) show `joltLnL` = the **MLE** and the
> **written-back params** self-check 10 nats *below* it — so the returned number is right and the *params* are the wrong
> side. The iplus data below is still valid; only its inference is superseded.

The iplus DEVCHECK printed `joltLnL / cpuLnL / gpuLnL(mirror)` at full precision for every +I+R candidate:
```
DNA  R2 pinv0.0184  jolt=-5781382.5790  cpu=-5781392.579889  gpu(mirror)=-5781392.579889  | jolt-cpu=+10.0009  gpu-cpu=0.000000
DNA  R3 pinv0.0110  jolt=-5728096.2909  cpu=-5728106.294508  gpu=-5728106.294508           | jolt-cpu=+10.0036  gpu-cpu=0.000000
DNA  R4 pinv0.00037 jolt=-5722777.7449  cpu=-5722787.747697  gpu=-5722787.747697           | jolt-cpu=+10.0028  gpu-cpu=0.000000
AA   R2 pinv0.0228  jolt=-7609499.8948  cpu=-7609509.897686  gpu=-7609509.897686           | jolt-cpu=+10.0029  gpu-cpu=0.000000
AA   R5 pinv4e-6    jolt=-7541972.3105  cpu=-7541982.314690  gpu=-7541982.314690           | jolt-cpu=+10.0042  gpu-cpu=0.000000
```
**The offset is a clean +10.00 in ABSOLUTE nats** (identical for DNA −5.7M and AA −7.5M ⇒ NOT pattern-count / rel-scaled),
and **`gpu(mirror) − cpu = 0.000000` exactly** ⇒ the independent on-device mirror agrees with CPU to printf precision.
So **the JOLT-optimised PARAMETERS are correct** (two independent evaluators, mirror ⟂ CPU, agree at those params);
only `gpu_jolt_optimize`'s RETURNED `lnL` carries the +10. Source audit closes two sub-hypotheses:
- **Writeback round-trip is CLEAN** (NOT the +10): `RateGamma::getRate`=`rates[c]` direct (rategamma.h:106), `RateFreeInvar::getProp`=`prop[c]`
  direct, `RateInvar::setPInvar` just sets `p_invar` (no rescale). `setRate(c,catRate)`→`getRate(c)`=`catRate` is identity ⇒
  the mirror reads EXACTLY the device's `catRate/catProp_v` ⇒ the +10 is `evalLnL`'s DEVICE COMPUTE, not the params.
- Because +10 is a **constant** added to every `evalLnL`, it **cancels in the optimizer's relative `ln>lnL+1e-9` comparisons**
  ⇒ convergence path unaffected ⇒ the +I+R params are as-trustworthy as +I+G's. The self-check gate (`rel≤1e-6` on the
  +10-biased `joltLnL`, phylotreegpu.cpp:2638) is the ONLY thing that trips → NaN → the redundant full CPU re-opt.

### 3b. 🔴 RESOLVED (batch JOB A/B/C + mfgauge 173893718, 2026-07-15) — the re-eval was HARMFUL; the +10 is a return-vs-writeback inconsistency; the +I+R optimiser CONVERGES to the MLE
**Two earlier claims RETRACTED here** (kept as retracted per the prune-not-append discipline): (i) "gaugeFix pre/post
divergence ⇒ +10" and (ii) "Route A re-eval = the fix, GREEN". A grounded 3-job batch + a decisive `-m MF` gauge probe
overturned both.

- **gaugeFix is INVARIANT — `worst|d| = 0.0000`** on every +I+R accept, in BOTH `-te` (JOB A gaugetr 173890950) AND
  `-m MF` (mfgauge 173893718, DNA and AA). So the §3b-old "pre/post-gauge clamp divergence lands at +10" mechanism is
  **REFUTED** — gaugeFix is not the source.
- **The +I+R optimiser already FINDS the MLE.** `-te` fixed-topology LG+I+R4 → **−7541972.276 (= the CPU MLE) with NO
  fix** (JOB A/B/C, three independent runs). The "AA shortfall" seen earlier was **caused entirely by the re-eval**, not
  by the optimiser.
- **The re-eval (`JOLT_IR_REEVAL`) is HARMFUL, not a fix — ABANDON it.** Its loop-exit `evalLnL` re-scores the
  **written-back** state, which is ~10 nats *worse* than the optimum the loop actually reached; so it *degrades* the
  reported optimum to the bad written-back value (JOB C: reported gap == drift exactly; JOB A: MLE reached only with the
  re-eval OFF). The earlier "irreeval GREEN" reading measured only that the number now *matches the written-back params* —
  it did not notice those params were 10 below the MLE.
- **What the +10 actually is (mfgauge, decisive):** `joltLnL` (returned) = **MLE**; `cpuLnL` (CPU postorder of the
  **written-back** params) = **MLE − 10**; `gpuLnL` (independent device mirror of the written-back params) = `cpuLnL`
  **exactly** (rel ~1e-11). ⇒ the optimiser finds the MLE but the state written to `out_rates/out_props/out_pinv/out_brlen`
  self-checks 10 nats worse. `rel = 10/7.5M ≈ 1.3e-6` just clears the `1e-6` self-check gate → NaN → CPU fallback.
  **This is the SAME phenomenon in `-m MF` and `-te` — there is no separate `-m MF` writeback bug.**

**THE REAL FIX (replaces the harmful re-eval): reconcile the writeback to the joltLnL optimum** — write back the *exact*
params that scored `joltLnL` (the loop's accepted best), so the self-check passes AT the MLE, not 10 below it. This is a
targeted return/writeback-consistency fix in `gpu_jolt_optimize`, NOT a re-eval and NOT a gauge change. Related, separable
perf work: the **decoupled per-arm LM damping** ("new variable") to cut the AA zigzag (reject_frac 0.42–0.56 vs DNA
0.07–0.21; ~56% of GPU postorders wasted on rejected backtracks — JOB B irprof) — a PERFORMANCE lever, not the correctness
fix. Route B (mirror-gate) stays REJECTED (DEVUSE tautology, weakens the backstop).

### 3c. ✅ RESOLVED + IMPLEMENTED + GRADUATED (job 173898475, 2026-07-15) — root cause = props-deficit; fix = `JOLT_IR_FDFIX`
**Root cause (Plan+blue-team, source-verified):** the pinv finite-difference (`gpu_lnl_intree.cu:3212-3215`) calls
`evalLnL(base,baseA,baseP±1e-4)`, whose `applyPinv` leaves host `catRate/catProp_v` at the `(1-baseP∓1e-4)` scaling.
**When `nFreeQ==0` nothing re-evals at `baseP`** before the `baseR_save/baseW_save` capture (`:3238`), so the reject-EXIT
restore (`:3333`) writes props summing to `1-1e-4` → offset **≈ 1e-4·N_var** (npat 96017 ⇒ ≈9.6 ≈ the observed 10.00).
⚠️ **Trigger is `nFreeQ==0`, NOT "AA"** — AA **and** equal/empirical-freq DNA (**JC+I+R, F81+F+I+R**) hit it; only FREE-Q DNA
(HKY/TN/GTR) is clean (its Q-FD's last eval re-evals at `baseP`). The corrupted `catRate` also feeds `g_y=catRate·gradR`
(`:3241`). **FIX `JOLT_IR_FDFIX` (default-ON):** re-apply `applyPinv(baseP)` before `:3238` — recovers the gauged state
(`meanR` holds `catRate_gauged·(1-pinv)`), repairing both consumers. Free-Q DNA no-op. (Defensive alt `JOLT_IR_BESTWB`
snapshot-restore = default-OFF, red-team: INCOMPLETE — skips the zero-accept exit; never ship solo. The harmful re-eval is
REMOVED.)

**GATE (job 173898475, DNA+AA):** `-m MF` +I+R worst `|jolt-cpu|` **10.00 → 1e-4** (FDFIX and BESTWB), rel ~1.7e-11 ≪ 1e-6
⇒ self-check PASSES, **no CPU fallback**; **best-fit UNCHANGED** (DNA F81+F+G4, AA LG+G4 both arms). `-te`: **DNA GTR+F+I+R4
= −5692971.953 = exact CPU MLE** (incl. the avian winner); non-+I+R & free-Q DNA byte-identical.

**🔴 HONEST CAVEAT (corrects §3b's "JOLT == CPU MLE on -te" — that was the CPU FALLBACK, not JOLT's own):** the fix UNMASKS
that JOLT's **AA** LG+I+R4 optimum (`-te` = **−7541972.346**) is **0.070 nats BELOW** the CPU MLE (−7541972.276). OFF hid
this by always failing the self-check → CPU fallback → CPU MLE. 0.07 nats = 0.14 BIC = **negligible for selection** (best-fit
unchanged), but it is a real GPU-optimiser convergence shortfall = **the ④ lever** (AA zigzag), tracked separately. DNA gap = 0.
**User decision 2026-07-15: GRADUATE FDFIX now, track AA 0.07 as ④.** DEFECT-2 (red-team, low, pre-existing): the writeback
is post-gauge, so a `brlen>20` clamp saturation would break gaugeFix invariance — not exercised here, note for real data.

## 3d. 🔴🔴 **THE MF THREAD-SCALING ANOMALY — ModelFinder IS STILL CPU-BOUND** (2026-07-19, measured, jobs `174128914`/`174128915`)

> ⭐ **The cleanest evidence yet that MF offload is INCOMPLETE — and it is a *scaling* argument, not a profile.**
> **If MF were genuinely GPU-resident its wall would be thread-INDEPENDENT.** Tree search already is. ModelFinder is not.

### The anomaly (disk-verified from the parity runs, tol 1e-2, `-m MFP`)

| data | phase | nt1 | nt12 | **nt1/nt12** | reading |
|---|---|---|---|---|---|
| DNA-1M | **MF** | 605.4 s | 213.6 s | **2.83×** | 🔴 scales with threads |
| DNA-1M | TS | 669.5 s | 593.6 s | 1.13× | ✅ ~flat (thread-independent) |
| AA-1M | **MF** | 4508.1 s | 1502.1 s | **3.00×** | 🔴 scales with threads |
| AA-1M | TS | 2546.1 s | 2379.1 s | 1.07× | ✅ ~flat |

⇒ **MF gets ~3× faster on 12 threads; TS barely moves.** A fully-offloaded phase cannot behave this way. **MF still
contains substantial THREAD-PARALLEL CPU work.**

### The decomposition — IQ-TREE's OWN `CPU time` vs `Wall-clock time` for ModelFinder (the decisive split)

| cell | MF CPU-time | MF wall | CPU/wall | interpretation |
|---|---|---|---|---|
| DNA nt1 | 446.4 s | 605.4 s | **0.74** | CPU **idle ~26%** of MF wall ⇒ that 26% is GPU-wait |
| DNA nt12 | **870.0 s** | 213.6 s | **4.07** | only **4.1 of 12 cores** busy (34% parallel efficiency) |
| AA nt1 | 3793.2 s | 4508.1 s | **0.84** | CPU idle ~16% |
| AA nt12 | **4342.2 s** | 1502.1 s | **2.89** | only **2.9 of 12 cores** busy (24% efficiency) |

**Three grounded conclusions:**
1. **MF is CPU-DOMINATED, not GPU-dominated.** At nt1 the CPU is busy 74% (DNA) / 84% (AA) of the MF wall — the GPU
   finishes and *waits*. The residual CPU work, not the kernel, is the MF wall.
2. ⚠️ **Total CPU time RISES with threads** (DNA 446.4→870.0 s; AA 3793.2→4342.2 s). 🔴 **MY FIRST READING OF THIS —
   "parallel overhead, real duplicated work" — IS UNSAFE AND IS RETRACTED PENDING MEASUREMENT.** `getCPUTime()` is
   `getrusage(RUSAGE_SELF).ru_utime` = **USER TIME ONLY** (`utils/timeutil.h:104`, verified by me). GNU libgomp
   **busy-spins at barriers**, and that spin is charged as user time. The self-check postorder issues **~98 fork/join
   barriers per candidate** (one `#pragma omp parallel for schedule(static)` per internal node,
   `phylokernelnew.h:2391`) ≈ 9,600 per DNA run, with ragged per-pattern work ⇒ guaranteed imbalance at every barrier.
   So the extra ~424 user-seconds may be **SPIN, not compute.**

   ✅ **SETTLED — job `174140641`, and I was ~60% wrong.** DNA-1M nt12, same run, three arms:

   | arm | MF CPU | MF wall | cpu/wall |
   |---|---|---|---|
   | base (spin) | 829.5 s | 212.3 s | 3.91 |
   | `OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0` | **584.1 s** (−29.6%) | **223.2 s** (+5.1%) | 2.62 |
   | PASSIVE + `NOSELFCHECK` | 116.9 s | 99.2 s | 1.18 |

   ⇒ **~245 user-seconds (30%) of the reported "CPU time" was libgomp barrier spin, not work.** The honest,
   spin-corrected thread-driven rise is **451 s (nt1) → 584 s (nt12) = +29%**, NOT the +95% the raw counters suggested.
   ⚠️ **The script's own verdict line prints "SPIN ARTEFACT PROVEN"; that word is too strong and I am not adopting it** —
   wall *rose* 5.1% under PASSIVE, so the spin was also doing real latency-hiding. The honest verdict is **MIXED, spin-
   dominated**: the CPU-time counter is inflated ~30%, but spinning is not free to remove. **Consequence: every
   "MF CPU time" figure in this document is an upper bound; only WALL is trustworthy.** The wall-based conclusions
   (55.1% / 83.1% / the 2.92×→1.10× flattening) are unaffected — they never used the CPU counter.
3. **Parallel efficiency is poor (24–34%)**, so the ~3× is well short of 12×.

### 🔴 CORRECTION — the thread scaling is NOT candidate-level parallelism (source-verified)
I originally wrote that the CPU work is "thread-parallel **across candidates**." **That is WRONG.** Verified in source
(and independently re-checked by me, not just the agent):
- `params.openmp_by_model` defaults **false** (`tools.cpp:7676`); **`--thread-site` sets it false** (`:4673`) and our runs
  all use `--thread-site`. `--thread-model` would set it true (`:4668`).
- The MPI block that would force `openmp_by_model=true` (`phylotesting.cpp:1993`) is **compiled out** (`strings | grep -c
  MPI_Init` = 0).
- **`CandidateModelSet::test` (`phylotesting.cpp:3779-4195`) contains ZERO `#pragma omp`** — I counted them myself: 0.
  The candidate loop is **strictly serial**.
⇒ the ~3× comes from **intra-candidate, per-PATTERN OpenMP inside the CPU likelihood kernel**
(`computePartialLikelihoodSIMD` etc., `phylokernelnew.h:2391/2861/3028/3618`), driven by `setNumThreads` →
`num_packets = num_threads*2` (`phylotreesse.cpp:44-52`). **This matters:** the lever is not "parallelise the candidate
loop", it is "stop doing the per-candidate CPU likelihood at all."

### ✅ ANSWERED — is the MF number INFLATED by an included sub-phase? **Yes, but only ~3–7%, and it makes the anomaly WORSE**
Timer starts `phylotesting.cpp:1843-1844`, stops+prints `:2048-2053`. **Both the parsimony start tree AND the fast-ML
tree search ARE inside the window** (`computeFastMLTree` at `:1878`, which calls `computeInitialTree` at `:777`) — so the
tree-search-style trap is real. But it is small, and **no final winner re-optimisation is inside** (the tail of `test()`
only sorts/checkpoints; the "Estimate model parameters" lines are *after* the print).

| | prefix (pars + fastML) | MF wall | **candidate loop** | nt1/nt12 |
|---|---|---|---|---|
| DNA nt1 | 23.98 s (4.0%) | 605.45 | **581.5** | |
| DNA nt12 | 15.81 s (7.4%) | 213.61 | **197.8** | **2.94×** (was 2.83×) |
| AA nt1 | 132.37 s (2.9%) | 4508.14 | **4375.8** | |
| AA nt12 | 52.90 s (3.5%) | 1502.07 | **1449.2** | **3.02×** (was 3.00×) |

⇒ **stripping the prefix moves DNA 2.83→2.94× and AA 3.00→3.02×.** Unlike tree search's `cand` trap, the MF number is
**not materially inflated** — the anomaly is genuine and slightly understated. Per candidate: DNA 5.93→2.02 s;
AA 65.3→21.6 s ⇒ **AA is 11.0× more expensive per candidate than DNA** (≈ the ns² ratio 25× damped by the GPU portion).

### ⭐⭐ ATTRIBUTION CONFIRMED — the per-candidate CPU SELF-CHECK is the residual work (job `174140461` cell 1, DNA nt12)
**The mechanism, source-verified (I re-read these lines myself):** every candidate falls through to
`phylotreegpu.cpp:2703  double cpuLnL = computeLikelihood();` — a **full, COLD Felsenstein postorder on the CPU**
(cold because `clearAllPartialLH()` ran at `:2568`), used only to validate the GPU number against a `rel<=1e-6` gate
(`:2789`); the **CPU value is then what MF tables** (`:2797`). The lean early-return that skips it (`:2440`) is reachable
**only** from tree-search brlen reopt (`:2971`) — its own comment calls the self-check "the **ModelFinder-only** D4
gain-eraser". MF passes only `fixed_len` ⇒ `leanTail=false` ⇒ **never skipped on the MF path.**

**MEASURED A/B (`JOLT_MF_NOSELFCHECK`, DNA-1M `-m MF` nt12, tol 1e-2):**

| arm | MF CPU | MF wall | cpu/wall | winner | JOLT markers |
|---|---|---|---|---|---|
| base | 831.0 s | 214.3 s | 3.88 | F81+F+G4 | 79 |
| **noselfcheck** | **185.6 s** | **96.2 s** | 1.93 | F81+F+G4 | **0** |
| **delta** | **−77.7%** | **−55.1% (2.23× faster)** | | ✅ unchanged | ✅ engagement proven |

🔑 **The CPU self-check is 77.7% of MF CPU time and 55.1% of MF WALL at nt12.** Engagement is proven by the marker count
collapsing 79→0 (not by the number under test), and **selection is unchanged**. `cpu/wall` falling 3.88→1.93 confirms the
self-check was *the* OpenMP-parallel phase — what remains is far more serial.

**AA-1M `-m MF` nt12 (job `174140461` cell 3, both arms landed):**

| arm | MF CPU | MF wall | cpu/wall | winner | markers |
|---|---|---|---|---|---|
| base | 4304.0 s | 1496.4 s | 2.88 | LG+G4 | 70 |
| **noselfcheck** | **1055.0 s** | **910.1 s** | 1.16 | LG+G4 | **0** |
| **delta** | **−75.5%** | **−39.2% (586.3 s)** | | ✅ unchanged | ✅ engaged (67 candidates both arms) |

🔴 **KEY NUANCE — on AA the self-check is a SMALLER SHARE of wall (39%) than on DNA (55%), even though it is ABSOLUTELY
5× larger (586 s vs 118 s).** Reason: AA's per-candidate GPU-optimiser work (ns²=400) is itself large, so it, not the
CPU self-check, dominates AA's MF wall. ⇒ **the design (§3e) saves ~5× more wall-seconds on AA in absolute terms, but AA's
residual after removal (910 s) is far larger than DNA's (96 s)** — AA's remaining wall is candidate-count × per-candidate
GPU cost + the serial floor, which is the §3e.8 / batched-BLAS territory, not the self-check. **This tempers my earlier
"AA is where the per-candidate CPU cost is 11× DNA's" framing: the CPU cost is larger, but so is everything else.**

### ⭐ THE DECISIVE CELL — nt1 (job `174140461` cell 0): removing the self-check **FLATTENS THE THREAD SCALING**

| arm | MF wall nt1 | MF wall nt12 | **nt1/nt12** | reading |
|---|---|---|---|---|
| base | 626.6 s | 214.3 s | **2.92×** | 🔴 the anomaly |
| **noselfcheck** | **106.2 s** | **96.2 s** | **1.10×** | ✅ **FLAT — matches TS's 1.13×** |

🔑 **This is the whole argument closed.** The user's hypothesis was "a fully-offloaded phase should be thread-independent;
MF isn't." Removing **one** CPU call makes it thread-independent. The self-check is not merely *a* term — it is
**essentially the entire thread-scaling anomaly**, and what remains (96–106 s) behaves like a genuinely GPU-resident phase.
At nt1 the self-check is **83.1% of MF wall**; at nt12, **55.1%**.

**A/B FAIRNESS PROVEN (the confound that would have invalidated this):** both arms evaluated **98 candidates** (identical
BIC-table row count, both announce "up to 968 DNA models"). The noselfcheck arm did **not** do less work — it did the same
work without the CPU recompute. Winner `F81+F+G4` unchanged in every cell. `[JOLT]` markers 79 → 0 prove engagement.

### 🔴 RED-TEAM OF MY OWN §3d — THREE CORRECTIONS I OWE

**(i) I called the in-tree comment "a story". That was unfair and I retract it.** The comment at `:2783-2786` ("~25% of MF
wall at `-nt 12`", "the 68% was an `-nt 1` artefact") was a *valid measurement of a different regime*. **The denominator
changed underneath it, not the numerator.** Reconciliation from the numbers each cites:

| regime | MF wall (self-check ON) | MF wall (OFF) | self-check ABSOLUTE | self-check SHARE |
|---|---|---|---|---|
| pre-tolerance (its era) | 1501.7 s (`173995856`) | 1300.9 s (`173931905`) | **≈200.8 s** | **13.4%** |
| post-tolerance (`JOLT_IR_TOL=1e-2`, ours) | 214.3 s | 96.2 s | **≈118.1 s** | **55.1%** |

⇒ the self-check's **absolute** cost was always ~120–200 s. The **tolerance lever** (which killed the `+R`
non-convergence cap that dominated the old wall) removed ~1290 s of GPU-optimiser work, and the *same* CPU term went from
a minor 13% to the **dominant 55%**. Both measurements are right; only the conclusion drawn from the old one has expired.
⚠️ Note the comment's own cited pair implies **13.4%, not the "~25%" it asserts** — a real internal inconsistency, and the
two jobs are different runs, so even 13.4% is a cross-job inference, not a clean A/B. **Our 55.1% is a same-job A/B.**

**(ii) 🔴 ITS DIRECTIVE IS NOW INVERTED — and this is the strategically important one.** The comment concludes: *"even
removed entirely our MF (1300.9s) still trails hers (1142.4s) ⇒ a free self-check LOSES DNA. Do not chase the self-check
as a speed lever."* At `tol 1e-2` our DNA-1M MF is **213.6 s @12t / 605.4 s @1t** (parity table) — i.e. **we already beat
her 1142.4 s by ~5.3× WITH the self-check ON.** The premise that made "don't chase it" correct no longer holds.
**⇒ chasing the self-check is now exactly the right move**, and the in-tree guidance must be updated in any shipped tree.

**(iii) My "candidate loop has ZERO `#pragma omp` ⇒ strictly serial" was RIGHT FOR OUR RUNS but stated too broadly.**
IQ-TREE has **two** MF paths, selected at `phylotesting.cpp:2004`:
- `params.openmp_by_model == false` → **`CandidateModelSet::test`** (`:3779`) — serial over candidates, per-PATTERN OMP
  inside. **This is our path** (default false `tools.cpp:7676`; `--thread-site` forces false `:4673`; MPI compiled out).
- `params.openmp_by_model == true` → **`CandidateModelSet::evaluateAll`** (`:4257`) — a genuine **OMP-parallel outer loop
  across models** (`#pragma omp parallel num_threads(num_threads) proc_bind(spread)`, `:4559`), reached via
  `--thread-model` or an MPI build.
⇒ the in-tree comment's "MF's candidate loop IS OMP-parallel (`evaluateAll :4559`)" and my "it is serial" are **both
correct in their own scope**. Consequences: on our path the `JOLT_SELFCHECK_STRIDE` counter race the comment warns about
**cannot occur**; and `--thread-model` is an unexplored axis that would parallelise the self-check across candidates
instead of across patterns.

(Its sibling claim at `:2575-2577` — "finalists-only = VERIFIED 0.85× regression" — rests on jobs deleted from disk and
is **UNVERIFIABLE**; it must be re-measured, not trusted, especially now that (ii) has changed the regime. ⚠️ **That same
comment carries a THIRD figure — "~11% at nt12" — which contradicts `:2783`'s own "~25% at nt12".** So the tree contains
**three mutually inconsistent self-check-share figures** (25% at `:2783`, 11% at `:2577`, measured **55%**). A source that
contradicts itself twice is weak evidence; our 55% is a same-job A/B.)
⚠️ **`JOLT_MF_NOSELFCHECK` is not selection-neutral *by construction*** — it returns `joltLnL` (GPU) instead of `cpuLnL`,
so a diverged candidate would be trusted. **But MEASURED, the difference is trivial:** a full `diff` of the two 99-row
`.iqtree` model tables shows **2 rows differing by 0.01 nats (0.02 BIC)**; 97 rows byte-identical, top-5 BIC order
unchanged, winner lnL identical to 4 dp. So it is a **sound timing A/B**; it is still **not shippable as-is** (it removes
the backstop, and absence of divergence *here* is not a guarantee elsewhere). Gate any shipped form on best-model +
top-K BIC order.

⚠️ **STATISTICAL HONESTY: every ratio in this section is n=1.** Measured same-config repeat variance is **0.92%**
(nt12 `-m MF` base 214.274 vs 212.312 across two jobs). The parity runs are `-m MFP` and the phasemap runs are
`-m MF -n 0`; at nt12 that config change is a **null** (213.6 / 214.3 / 212.3, 0.3% spread), so the nt1 gap
(605.4 → 626.6, **+3.5%**) is best read as run-to-run variance, not configuration — but it means **2.83× and 2.92×
are the same measurement within noise** and neither is replicated. **AA cells 2/3 have NOT finished; no AA A/B number
exists yet and none should be quoted.**

### Why this reconciles the DNA-vs-AA asymmetry (see WORK-LOG head-to-head)
The tolerance lever cut DNA MF 3.6× but AA MF only ~6%. Both facts fit **one** model: MF wall = (GPU candidate
optimisation) **+ (per-candidate CPU work that scales with ns² and with candidate COUNT)**. DNA's wall was dominated by
the +R non-convergence *iterations* (tolerance fixed that); AA's is dominated by **per-candidate CPU work × 1232 protein
models**, which tolerance cannot touch. **⇒ the next AA lever is this CPU residue, not tolerance.**

### PRIME SUSPECT + the decisive zero-build test (job `174140461`, `gems_mf_phasemap.sh`, submitted)
**Hypothesis:** the per-candidate **CPU self-check** — a full CPU postorder likelihood recomputed for EVERY candidate to
validate the GPU number (`phylotreegpu.cpp`, `rel<=1e-6` gate). That is O(nptn × ntax × ns²) per candidate, is
thread-parallel across candidates, and scales with **ns²** — which would explain **both** the ~3× thread scaling **and**
why AA (ns=20) is far worse than DNA (ns=4).
**Test:** A/B on the existing `JOLT_MF_NOSELFCHECK` knob (sentinel verified present in `f3f7875f`), `-m MF` (isolates
ModelFinder, no tree-search confound), tol pinned 1e-2, DNA+AA × nt1+nt12. **Pre-registered readings:**
- CPU time collapses **and** the nt1/nt12 ratio flattens ⇒ **self-check confirmed** as the residual CPU work ⇒ the lever
  is a **GPU-RESIDENT self-check** (the per-call mirror is already REFUTED at scale — `JOLT_MF_DEVUSE`, 173837501).
- CPU time barely moves ⇒ **self-check is NOT the lever**; the cost is framework/per-candidate setup (model construction,
  eigendecomposition, tip rebuilds) ⇒ a different target. **Report either outcome honestly.**
- Engagement is proven by a DROP in per-candidate `JOLT` marker count, **not** by the number under test (a gate must
  prove its control is a control). Winner must be unchanged (`NOSELFCHECK` is validation-only, not selection).

### ⚠️ Open: is the MF number INFLATED by an included sub-phase? (the tree-search trap)
Tree search's naive wall folds in a separate `cand`/fast-ML-tree stage; **MF may have the same trap.** Console order is
`parsimony (3.681s)` → `fast ML tree search (20.294s)` → `"ModelFinder will test up to 968 DNA models"` → `Best-fit
model` → `CPU/Wall time for ModelFinder`. Whether the **fast-ML tree is inside or outside the MF timer is UNRESOLVED
from the console alone** and is being settled from source (timer start/stop trace). Until then, treat MF wall as
possibly including up to ~20 s of pre-model-loop tree work (small vs 605 s, but it must be stated, not assumed).

### 📍 THE PHASE MAP (the deliverable) — MF window = `phylotesting.cpp:1843` → `:2049`
Seconds = DNA nt1 / DNA nt12 / AA nt1 / AA nt12, disk-verified from the parity logs.

| # | phase | where | CPU/GPU | offloaded? | seconds |
|---|---|---|---|---|---|
| 0 | `.model.gz` restore, `readModelsDefinition` | `:1849-1874` | CPU | n/a | <0.1 |
| 1 | **Parsimony start tree** | `:777`→`iqtree.cpp:701` | mixed (Fitch on GPU, stepwise addition CPU) | partial | 3.68 / 3.80 / 4.62 / 4.28 |
| 2 | **Fast-ML tree search** (model opt @eps×50 + NNI) | `:794-862` | mixed | yes | 20.29 / 12.02 / 127.75 / 48.62 |
| 3 | candidate-set `generate()`, memory check | `:1908-1930` | CPU | n/a | <0.1 |
| **4** | **CANDIDATE LOOP** — `CandidateModelSet::test`, **SERIAL over candidates (0 omp pragmas)** | `:3779-4195` | — | — | **581.5 / 197.8 / 4375.8 / 1449.2** |
| 4a | ↳ `new IQTree` + `initializeModel` (incl. eigendecomposition) | `:2419,:2444` | CPU serial | **no** | not exposed |
| 4b | ↳ `ModelCheckpoint` deep copy (`omp critical`, uncontended) | `:2436` | CPU serial | no | not exposed |
| 4c | ↳ `initializeAllPartialLh` — **~11.7 GB alloc + first-touch per DNA candidate** | `phylotree.cpp:977` | CPU serial, thread-INDEP | **no** | not exposed |
| 4d | ↳ DFS reindex / flat topology arrays | `phylotreegpu.cpp:2325-2340` | CPU serial | no | `--jolt-diag` only |
| 4e | ↳ tip[]/ptnFreq[] gather | `:2357`→`:1481` | CPU serial | **cached (default-ON)** | ~0 after 1st |
| 4f | ↳ `base_invar[]` O(nptn), `+I` only | `:2373-2381` | CPU serial | no | not exposed |
| **4g** | ↳ **`gpu_jolt_optimize`** — joint LM over brlen+α+pinv+Q+R | `:2410` | **GPU** | ✅ **YES** | not exposed |
| 4h | ↳ write-back + `clearAllPartialLH()` | `:2557-2568` | CPU serial | no | not exposed |
| **4i** | ↳ 🔴 **CPU SELF-CHECK: full COLD Felsenstein postorder, EVERY candidate** | **`:2703`** | **CPU, OMP over patterns @ `-nt`** | ❌ **NO** | **≈55% of MF wall (measured)** |
| 4j | ↳ `rel<=1e-6` gate (0 fallbacks fired: `grep -c MISMATCH`=0) | `:2789` | CPU | n/a | 0 |
| 4k | ↳ checkpoint save (`omp critical`), `computeICScores` | `:2682,:3947` | CPU serial | no | ~0 |
| 4* | ↳ **`+R` re-init retry loop ×5** ⇒ up to 5 `optimizeParameters` calls (⇒ 5 JOLT calls + 5 self-checks) per `+R` row | **`phylotesting.cpp:2649-2662`** (NOT phylotreegpu) | — | — | **multiplier** |
| 5 | `dump`, `transferModelFinderParameters`, sort/rank | `:2040-2043` | CPU | no | ~0 |

**Structure:** phase 4g is the only offloaded compute. **4i is the ONLY per-candidate phase that is both thread-parallel
and ns²-scaling** — matching all three observed signatures (2.94×/3.02× thread scaling, AA 11× per-candidate, CPU idle at
nt1). Phases **4a/4c/4h are the SERIAL CPU floor** and are what remains after 4i is removed (the residual 96.2 s / 185.6 s
CPU at nt12, `cpu/wall`→1.93).

### 🎯 RANKED LEVERS (from the phase map, not from speculation)
1. **🥇 Eliminate / amortise the per-candidate CPU self-check (4i) — worth ~55% of MF wall at nt12, measured.**
   Not by trusting the GPU (`NOSELFCHECK` changes the reported lnL and removes the correctness backstop). Candidate forms,
   in increasing order of ambition: **(a) STRIDE it** — `JOLT_SELFCHECK_STRIDE` already exists (`:2596`, default 1 = never
   skips); validating every *k*-th candidate keeps a real backstop at 1/k the cost. **(b) GPU-RESIDENT mirror** — the
   per-call mirror is REFUTED at scale (`JOLT_MF_DEVUSE` 173837501: setup swamps it), so this needs tip/echild reuse
   ACROSS candidates + a tiled arena. **(c) finalists-only** — the in-tree "0.85× regression" claim (`:2575`) rests on
   deleted jobs; **re-measure before accepting or rejecting.**
2. **🥈 The serial CPU floor (4a/4c/4h)** — now the binding term once 4i is gone (`cpu/wall`→1.93). Prime target
   **4c: ~11.7 GB `initializeAllPartialLh` alloc + first-touch PER CANDIDATE**, thread-independent by construction.
   Reusing one arena across candidates is the obvious win. **Unmeasured — needs `--jolt-diag` phase timers to size it.**
3. **🥉 The `+R` ×5 retry loop (4*)** — multiplies 4a-4k five-fold on 8 rows/run. Interacts with the tolerance lever.
4. **AA-specific:** AA's wall is candidate COUNT (1232 models) × per-candidate cost. Since tolerance cannot touch it
   (measured: AA MF moves ~6% across tolerances vs DNA 3.6×), **1 and 2 above ARE the AA lever.** This is why AA ties at
   1t while DNA wins 1.51×.

### 🔴 Provenance caveat on §1's profile numbers
§1's "host self-time DNA 43.5% / AA 69.5%" (`computePartialLikelihoodSIMD`+`dotProductTriple`+`bufferSIMD`, job
mfoffload `173815952`) — **that perf data is DELETED from disk and cannot be re-verified.** It is retained as an
**unverified prior** that *predicts* the self-check hypothesis above; job `174140461` re-establishes the attribution
from scratch rather than resting on it.

## 3e. 🎯 THE GPU-FIRST REDESIGN — trust the GPU, verify only the FINALISTS (red-team-reduced from a fuller ladder)

> **Status labels used below:** ✅ MEASURED · 📄 SOURCE-VERIFIED · 🧮 DERIVED · ✏️ DESIGN (not built) · ❓ UNVERIFIED.
> Nothing here is built. This is a specification with a falsification plan.

### 🔴 RED-TEAM VERDICT (design audit, job `ab74…`, source-verified by me) — BUILD A REDUCED FORM, AND IT IS INCREMENTAL NOT ARCHITECTURAL
> The full T0–T3 stratified-per-pattern ladder as first written is **over-engineered**. An adversarial audit (independently
> re-verified against source, lines below) broke three load-bearing claims. **The surviving, honest design is lean:**
>
> | tier | keep? | why |
> |---|---|---|
> | **T0** on-device NaN/Inf/range + 2nd-order reduction of `out_patlh` | ✅ **KEEP** | genuinely ~free **at nTile==1** (DNA-1M); a real cheap tripwire |
> | **T1/T2** stratified per-pattern CPU probe every candidate | ❌ **CUT as specified** | its only unique value is catching a *diffuse* bug on a *non-finalist* — which by definition can't change selection. Heavy engineering for a fidelity-only margin. (Optionally keep a *simple uniform* T1 at nTile==1 as a cheap monitor.) |
> | **T3** exact CPU postorder on finalists | ✅ **KEEP, but CAPPED** | 🔴 **drop the unbounded "each family's best" clause** — it forces one full postorder **per family** (≈22 DNA / ≈30 AA), and on AA (8.38 s/postorder, measured) that reconverges to ~251 s, eating ~43% of the very cost being removed. Cap at leader-BIC-margin + hard K≈5–10. |
> | **tighten `rel` 1e-6 → ~1e-9** | ✅ **KEEP** | free wherever a CPU value already exists; makes the *existing* backstop meaningful (§3e.2) |
>
> **🔴 THE HEADLINE CORRECTION: the "second-order unlock" (§3e.8) is FALSIFIED.** The 11.7 GB arena is allocated
> **eagerly and upstream** (`phylotesting.cpp:2560`, *before* JOLT dispatch at `modelfactory.cpp:1613`) and is **load-bearing
> for the ~19 JOLT-declined candidates** that fall to the CPU optimiser (`modelfactory.cpp:1625`). **Proven on disk:** the
> `NOSELFCHECK` arm still floors at **96.2 s, cpu/wall 1.93** — the arena is untouched by removing the self-check. ⇒ this
> redesign is an **incremental self-check optimisation, not an alignment-resident refactor.** Getting *below* the 96 s floor
> needs a **separate, risky, eligibility-gated** refactor of `:2560`, scoped on its own.
>
> **Realistic headroom (corrected):** DNA ~55–72% (not 85–90%); **AA only ~57%** — and AA is where the self-check is a
> *smaller* share to begin with (§3d). ⇒ **this is a DNA-modest / AA-limited win; do not oversell it.** The subsections
> below are kept for the mechanism and evidence, each annotated with the correction that applies.

### 3e.1 The actual defect: the self-check does THREE jobs at once, and only one of them needs a CPU postorder

📄 Reading `phylotreegpu.cpp:2703-2798`, the single `computeLikelihood()` call is load-bearing for three separable jobs:

| job | what it does | does it need a FULL CPU postorder PER CANDIDATE? |
|---|---|---|
| **J1 DETECT** | is the GPU number wrong? | **No** — see 3e.3; a sample suffices, and is *more* sensitive |
| **J2 REPAIR** | `rel>1e-6` → `return NaN` → full CPU re-optimisation | No — only on the rare failure |
| **J3 DEFINE** | `return cpuLnL` (`:2797`) ⇒ the CPU value is what the MF table publishes | Only for candidates that **decide the answer** |

**J3 is what forces the cost.** Because the published number must be CPU-derived for *every* row, the architecture is
obliged to run a CPU postorder for every candidate even when J1 is already satisfied far more cheaply. **Un-conflating
these three is the entire design move.**

### 3e.2 🔴 The gate we are paying 55–83% of MF wall to preserve is near-decorative (MEASURED)

✅ At DNA-1M, `lnL ≈ −5.9208e7`, and the gate is **relative**: `rel = |joltLnL−cpuLnL|/|cpuLnL| ≤ 1e-6` (`:2715`, `:2789`).

| quantity | value | source |
|---|---|---|
| error the gate **admits** | **59.2 nats = 118.4 BIC units** | 🧮 from the measured lnL |
| **top-2 BIC gap in this very run** | **12.2 BIC** (`F81+F+G4` 118419112.858 vs `HKY+F+G4` 118419125.073) | ✅ `.iqtree` |
| avian real-data top-2 gap | 82 BIC | ✅ WORK-LOG `:1317` |
| **worst rel actually observed** (79 checks) | **9.255e-12** (median 3.364e-12) | ✅ console |

⇒ **The gate admits ~10× more error than separates the winner from the runner-up on its own dataset**, while the true
divergence sits **~10⁵× below** the threshold. It cannot fire until the answer is already wrong, and it never fires here
(`grep -c MISMATCH` = **0** in all four consoles). 📄 **The tree already says this** (`phylotreegpu.cpp:2781`):
*"the rel<=1e-6 DETECTION arm is near-decorative at 1M (admits ~59 nats vs ~6 that decide selection) — do not lean on it."*

🔑 **So the current design is the worst of both worlds: maximal cost, near-zero detection power.** That, not the raw
55%, is the real case for redesign — and it means a replacement does **not** have to be "as safe as today" to be an
improvement; today's bar is very low.

### 3e.3 ⭐ THE CORE IDEA — audit PER-PATTERN on a SAMPLE, not PER-CANDIDATE on the TOTAL

Both the cost and the blindness come from the same choice: comparing **one scalar** (the total) computed over **all
~935,227 patterns**. Invert both axes.

🧮 **Why a sample of patterns is enormously more sensitive than the full total.** Patterns are **statistically
independent given the tree and model** (`lnL = Σ_p f_p · log L_p`), so a per-pattern comparison has no cancellation:
- Today's total-only gate tolerates **59.2 nats** of total error, which a systematic bug spreads as **δ ≈ 5.9e-5
  nats/site** — invisible in a total, but **~10⁶× above the ~1e-11 per-pattern FP64 agreement floor** between two correct
  implementations. **A single sampled pattern would expose it.**
- Errors of opposite sign across patterns **cancel in the total** and are structurally invisible to today's gate. They
  cannot cancel in a per-pattern comparison.

⇒ auditing 4,096 patterns (0.44% of the alignment) is **~225× cheaper and, for a DIFFUSE error, orders of magnitude more
discriminating** than the check it replaces.

⚠️ **RED-TEAM CORRECTION — this is a TRADE, not an unconditional win.** Per-pattern sampling is far more sensitive to
**diffuse** errors (spread across patterns, e.g. the historical `+I+R` +10-nat bug) but **WEAKER** to **sparse-large**
errors (a few patterns badly wrong), which today's total-only gate actually *accumulates*. §3e.5's own K=10 → 4.3%
detection is that weakness. So "sampling makes the check stronger" is honest **only for the diffuse class**; net safety
against sparse errors rests **entirely on the capped T3**, not on T1. (Independence in 3e.3(a) is source-confirmed:
`gpu_lnl_intree.cu:786-792` / `:1336` build `lnL = Σ_p f_p·patlh[p]`, and the only rescaling is per-pattern indexed,
no cross-pattern coupling.)

### 3e.4 ✏️ THE LADDER — as originally conceived, with the red-team's keep/cut applied

> 🔴 **Read the verdict banner first.** T1/T2 as specified are **CUT**; T3's "each family's best" clause is **DROPPED**
> (capped). The full table is kept for the reasoning; the **keep?** column is the decision.

| tier | keep? | when | what | cost @DNA-1M | catches |
|---|---|---|---|---|---|
| **T0** | ✅ **KEEP** | every candidate, on device | NaN/Inf/range guard + **2nd reduction of `out_patlh` in a different order** ⇒ a **plausibility check** ⚠️ NOT a certificate (§3f.5); ⚠️ free only at nTile==1 (§3e.6) | ~free (DNA-1M) | reduction/rounding pathologies, silent NaN |
| **T1** | ❌ **CUT** (opt. simple-uniform monitor) | every candidate, host | ~~STRATIFIED~~ per-pattern audit: recompute log L_p on CPU for a sample S (≈4096), compare per-pattern @~1e-9 | 🧮 ~<2 s/run | diffuse model-term bug — **but only on non-finalists, which can't change selection** |
| **T2** | ❌ **CUT** | first candidate of each family | T1 with a large sample (~65k) | 🧮 ~2 s/run | family-specific term errors (subsumed by capped T3 verifying each finalist) |
| **T3** | ✅ **KEEP, CAPPED** | end of run | **EXACT full-alignment CPU postorder on finalists** = models within a BIC margin of the leader, **hard cap K≈5–10.** 🔴 **DROP "each family's best"** (§3e.7: reconverges to ~251 s on AA) | 🧮 DNA ~8–15 s / **AA ~42–84 s** | anything that could change the **selection** |
| **esc** | ✅ KEEP | on any T0 failure | fall through to today's full CPU postorder + `NaN` → CPU re-opt | rare | preserves today's backstop exactly |

⚠️ **On stratification** — I originally wrote "STRATIFICATION IS NOT OPTIONAL". With T1 cut, it is moot; *if* a simple
uniform T1 monitor is kept at nTile==1, stratification remains the honest way to cover code-path branches (constant/`+I`
patterns, gap/ambiguous `g_UinvRowSum` bucket, rescaling/underflow, extreme-frequency) — but it is **no longer
load-bearing for selection safety**, which now rests entirely on capped T3.

### 3e.5 🧮 THE GUARANTEE — stated honestly, including what it does NOT cover

Uniform sampling is weak against a **sparse-but-large** error (few patterns, big per-pattern magnitude): detection
probability is `1−(1−K/N)^S`, so K=1000 affected patterns ⇒ **98.7%**, but K=10 ⇒ **4.3%**. Stratification targets the
plausible sparse modes, but cannot exclude them all. **T3 is what closes it — by a decision argument, not a detection one:**

> A sparse error missed by T1/T2 can only harm the result if it changes the selection. If it **promotes** a wrong model,
> that model is in the finalist set and is **exactly verified on the full alignment by T3**. If it does not reach the
> finalist set, it did not change the selection. **⇒ the only residual risk is a large *negative* sparse error that
> DEMOTES the true winner below the finalist cut.**

🔴 **THE GENUINE TENSION the red-team exposed (and it must be resolved, not hidden):** the mitigation for that residual —
"T3 also verifies the best of *every family*" — is **exactly the clause that makes T3 reconverge to ~251 s on AA** (§3e.7).
So safety-against-demotion and cost pull in opposite directions. Two more holes in the naive version: (a) the finalist
margin is computed from the **possibly-corrupted GPU leader score**, so a corrupted leader shifts the margin's centre;
(b) the `+R` ×5 re-init loop (`phylotesting.cpp:2649`) breaks on the GPU value, so a corrupted-high score can early-break
to suboptimal params that T3 then certifies as the *exact lnL of the wrong optimum*.

**HONEST RESOLUTION (adopt): CAP T3, accept the residual, document it.** The residual requires a **large negative sparse
error landing specifically on the true winner's own score** — low probability, and a cheap uniform-T1 monitor (if kept)
catches the diffuse modes. Paying 251 s on AA to insure against it only makes sense under the "the argmin is sacred"
framing that **§3f.8 (Abadi 2019) shows is not scientifically load-bearing.** Holes (a)/(b) are **≈ today's behaviour**
(today's self-check also merely re-scores whatever params JOLT set) — so capped T3 is **no worse than the status quo on
those, and strictly better on the diffuse-error case.**

**Net (corrected): the new guarantee is stronger on the DIFFUSE case and equal on the sparse/`+R` cases.** Today:
*"every candidate's total agrees within 118 BIC"* — which provably does **not** protect a 12.2 BIC decision. Proposed:
*"every finalist within the leader margin is exactly CPU-verified on the full alignment; no wrong model can be **promoted**
into that set without exact verification; the residual is a large negative sparse error demoting the true winner, accepted
and documented."*

### 3e.6 ✅ FEASIBILITY — the primitives already exist (this is why it is cheap to try)

- 📄 **Per-pattern device values are already available.** `gpu_jolt_optimize` takes an `out_patlh` argument that fills
  per-pattern `log|lh_ptn|` for the accepted tree (`phylotreegpu.cpp:96`, `:2401-2421`; the write is `gpu_lnl_intree.cu:116`).
  It is currently populated only under `-B` + `JOLT_BOOT_SNAPSHOT`. **T0 needs no kernel change — just pass a buffer.**
  🔴 **RED-TEAM CORRECTION — "just pass a buffer" is free ONLY at `nTile==1`.** Verified at `gpu_lnl_intree.cu:3509`
  (`if (out_patlh && !snapDone) { … evalLnL(…) }`): when the arena is tiled (`nTile>1` — **AA-1M, avian, 10M**), filling
  `out_patlh` forces a **full per-candidate GPU re-eval**. DNA-1M fits one tile so T0 is genuinely free there; on the
  large/AA cases T0's cost is a re-eval and must be budgeted, not assumed free. ⚠️ Also: the JOLT fold has **no underflow
  rescaling**, so on very deep trees where FP64 per-pattern lh underflows (CPU rescales, GPU → `log(0)=-inf`) T0/T1 would
  raise **spurious** escalations — erodes savings on the biggest data, not a correctness hole.
- 📄 **Sub-alignment CPU evaluation is exact for the JOLT-eligible class.** Per-pattern likelihoods depend only on the
  column pattern, tree and model (rescaling is per-pattern indexed — confirmed), so a "probe alignment" reproduces each
  sampled pattern's `log L_p` **exactly**. **`+ASC` declines to CPU (`phylotreegpu.cpp:2152`** — corrected from the wrong
  `:2108`), and **mixtures / non-reversible / site-specific all decline JOLT (`:2147`)**, so the cross-pattern-coupling
  worry is **moot by construction** — those never reach the self-check. ❓ Empirical-frequency (`+F`) params must be
  **copied from the fitted model, never re-estimated from the probe sub-alignment** — an implementation risk the (unbuilt)
  probe must honour; source cannot confirm code that does not exist yet. Partitioned `-p` is not in the JOLT gate
  explicitly ⇒ **the mandatory gate cell.**
- 📄 The probe tree is built **once per run**, so T1's per-candidate work is a parameter copy plus a ~4k-pattern
  postorder — it does **not** pay the ~11.7 GB `initializeAllPartialLh` (phase 4c); its arena is ~4096/935227 of it.

### 3e.7 🧮 PROJECTED EFFECT (a projection, NOT a measurement)

Measured per-postorder cost (from the A/B slabs ÷ self-check calls): **DNA 118.1 s / 79 = 1.49 s**, **AA 586.3 s / 70 =
8.38 s**. These set both T1 (tiny fraction of one postorder) and T3 (whole postorders on finalists).

| | DNA-1M nt12 | DNA-1M nt1 | **AA-1M nt12** |
|---|---|---|---|
| today (self-check ON) | ✅ 214.3 s | ✅ 626.6 s | ✅ 1496.4 s |
| ✅ measured floor (`NOSELFCHECK`, no audit) | **96.2 s** | **106.2 s** | **910.1 s** |
| self-check absolute (the slab attacked) | 118.1 s | 520.4 s | **586.3 s** |
| 🔴 **UNCAPPED T3** ("each family's best", ≈22 DNA / 30 AA postorders) | ~145 s | ~155 s | **~1161 s** |
| ✅ **CAPPED T3** (leader-margin + K≈5–10) — RECOMMENDED | **~105–115 s** | **~115–135 s** | **~955–995 s** |

⇒ 🔴 **The originally-written "each family's best" clause was the costly mistake:** on AA it reconverges to ~1161 s = only
**~57% of the 586 s slab reclaimed**, because 30 families × 8.38 s ≈ 251 s. **Capping T3** (drop the family clause) recovers
DNA to **~55–72%** headroom and AA to **~85% of its slab** — but AA's *residual* (910 s) is GPU-optimiser-dominated, not
self-check, so even the capped win is a **modest fraction of AA's total MF wall.** Assumptions still needing a spike (§3e.10):
per-call self-check basis (79 = JOLT calls incl. `+R` retries, **not** 98 distinct BIC rows — confirmed from console
`[JOLT]`=79); T1 near-linear in pattern count with a possibly-dominant fixed per-call OMP-fork floor at 4k patterns
(**the weakest link**); finalist count small (near-degenerate BIC clusters, common on real data, inflate T3).

### 3e.8 🔴 THE "SECOND-ORDER UNLOCK" — **FALSIFIED** (red-team, source + disk verified by me)

I hypothesised that phase **4c** (`initializeAllPartialLh`, ~11.7 GB alloc + first-touch **per candidate**) exists only to
feed the self-check, so removing the self-check would collapse the serial floor. **This is WRONG on three grounds:**
1. 📄 **The alloc is eager and UPSTREAM of the self-check** — `phylotesting.cpp:2560 iqtree->initializeAllPartialLh()` runs
   *before* `optimizeParameters` (which dispatches JOLT at `modelfactory.cpp:1613`). Removing the self-check at
   `phylotreegpu.cpp:2703` leaves `:2560` untouched.
2. ✅ **Proven on disk:** the `NOSELFCHECK` arm still floors at **96.2 s, cpu/wall 1.93** — the arena is still allocated and
   first-touched. **The floor did NOT collapse.**
3. 📄 **The arena is load-bearing for the ~19 JOLT-DECLINED candidates** (79 of 98 engaged this run) that fall through to
   the CPU optimiser at `modelfactory.cpp:1625` (`cur_lh = tree->computeLikelihood()`), which reads the partials. It cannot
   be blanket-removed.

⇒ **This redesign is INCREMENTAL, not architectural.** Killing the self-check gets you to the 96 s floor; going *below* it
needs a **separate, eligibility-gated, correctness-risky** refactor of `:2560` that preserves the arena for declined
candidates — its own project, not a free consequence. The 🥇 (self-check) and 🥈 (arena floor) levers are **independent**,
as the phase map already had them.

**The framing shift (still valid as a DIRECTION, not a free unlock):** IQ-TREE's MF is **candidate-major** (a whole tree object + arena per model) because on CPU the
setup was cheap relative to the likelihood. On GPU that is inverted. A GPU-first ModelFinder is **alignment-resident,
model-streaming**: one arena, one tip upload, one topology; models stream through as parameter sets, and **the CPU
becomes a sparse auditor rather than the definition of truth.** 📄 Note `JOLT_MF_RESIDENT` (default-ON since 2026-07-17,
`gpu_lnl_intree.cu:2781`, ✅ 1.597× bit-identical) already does this **within** one `gpu_jolt_optimize` call — its own
comment states `loadedChunk` resets per call, so **"a fresh candidate always re-uploads"**. Extending residency *across*
candidates is the untaken step. ⚠️ Its safety invariant is the process-wide `jolt_gpu_mtx`; narrowing that mutex turns
this into a silent stale-chunk corruption bug.

### 3e.9 ❌ WHY THE OBVIOUS ALTERNATIVES ARE ALREADY CLOSED (do not re-propose these)

- **Per-call device mirror (`JOLT_MF_DEVUSE`) — REFUTED at scale** (job 173837501): aa-1M **3.7× SLOWER**. Cause is
  per-call setup (echild rebuild + ~100 MB tip re-upload + a **non-tiled ~59 GB** partial arena) **every candidate**.
  The audit design **needs no device mirror at all** (T0 reuses `out_patlh` already on device; T3 is a CPU postorder),
  which is precisely why it sidesteps this.
- **`JOLT_SELFCHECK_STRIDE>1` — unsafe *alone*, but the RIGHT primitive for the lean form**: it bypasses the fallback
  wholesale (a diverged optimum trusted blind), so it must be **paired with capped-T3 finalist verification + the tighter
  `rel`**. It samples in the *candidate* dimension; capped-T3 then exactly verifies the few that decide selection. This
  pairing — not the elaborate per-pattern ladder — is the red-team's recommended build.
- **`JOLT_MF_NOSELFCHECK` — a measurement, not a lever** (removes the backstop entirely).
- **"finalists-only = 0.85× regression"** (`:2575-2577`) — 🔴 rests on jobs **deleted from disk** ⇒ **UNVERIFIABLE**.
  T3 is a finalists-style tier, so **this claim must be re-measured before it is allowed to veto the design.** Note the
  tree gives **two mutually inconsistent** figures for the same quantity (**~25%** at `:2783`, **~11%** at `:2577`;
  measured **55%**) — a source that contradicts itself is weak evidence.

### 3e.10 🚦 THE FALSIFICATION PLAN (what must be measured before any of this is believed)

🔴 **STEP 0 (red-team's sharpest point — do this before building anything): prove the LEAN design isn't already enough.**
The reduced form (trust GPU + **capped T3** + `rel`→1e-9) is nearly the whole win, using primitives that *already exist*
(`JOLT_SELFCHECK_STRIDE` `:2594`, the finalists concept, a free tolerance tighten). Build **that** first and measure it;
only if a fidelity gap remains is the T0 tripwire (or a simple uniform T1 monitor) worth the engineering. If capped-T3
alone lands within a few seconds of the elaborate ladder, **the ladder is not worth building** — ship the lean form.

1. **T3 cost / finalist-count spike (cheapest, do first):** on DNA-1M and **AA-1M**, count how many models fall within a
   BIC margin of the leader, and time one full postorder (DNA 1.49 s, AA 8.38 s — measured). **If the finalist cluster is
   large on real data (avian: top-2 within 82 BIC), capped-T3 reconverges toward the full cost** — the hard cap K is then
   the whole design. Settle K here.
2. **Sensitivity oracle:** inject a known bug (`JOLT_IR_NOFDFIX` reinstates the **+10-nat `+I+R` defect** — real,
   historical, diffuse) and confirm the **capped T3** (and, if kept, a uniform T1) flags it. A design that cannot
   re-detect our own known bug is not shippable.
3. **`out_patlh` nTile>1 cost (for T0 on AA/large):** measure the per-candidate re-eval that `gpu_lnl_intree.cu:3509`
   forces when the arena is tiled — T0 is only free at nTile==1, so on AA it must be budgeted or gated off.
4. **Gate re-spec (free, do independently):** tighten `rel` 1e-6 → ~1e-9 (observed max 9.255e-12 ⇒ ~10³ headroom) —
   makes the *existing* backstop meaningful on its own, no ladder required.
5. **Selection invariance:** best model + full top-K BIC order unchanged, DNA **and** AA, `-m MF` and `-m MFP`, plus a
   **partitioned `-p` cell** (⚠️ every gate in this project was once blind to partitions).
6. **Arena floor, separately:** the 96 s `NOSELFCHECK` floor (§3e.8) is a *different* project — an eligibility-gated
   refactor of `phylotesting.cpp:2560` that must keep the arena for the ~19 declined candidates. Do not couple it to the
   self-check work.

---

## 3f. 📚 LITERATURE GROUNDING — the design is precedented, the *current* design is not

> Evidence grades: **[READ]** = source retrieved and read · **[SUMMARY]** = search summary only, verify before citing ·
> **[UNREAD]** = retrieval failed, flagged rather than filled in from memory.

### 3f.1 ⭐ Our per-candidate CPU revalidation appears to be WITHOUT PRECEDENT in production phylogenetics
- **BEAGLE 3** (Ayres et al. 2019, *Syst Biol* 68(6):1052–1061) **[READ]** — full text contains **no** runtime CPU
  revalidation, no tolerance comparison, no GPU-vs-CPU self-check. The paper is API/partitioning/parallelism.
- **MrBayes tgMC3** (Zhou et al. 2013, *PLOS ONE* 8:e60667) **[READ]** — validates **once at development time**
  (*"the accuracy of the results is not changed by porting these codes"*). No production check.
- ⚠️ **[UNREAD] — the one place a contradiction could hide:** BEAGLE v4.0.0 tensor-core (Gangavarapu et al. 2026,
  *Syst Biol*, doi:10.1093/sysbio/syag017). PDF would not extract; **its validation methodology is UNKNOWN.**
  **Obtain via institutional access and read §Methods before this claim goes in the thesis.**

⇒ the community norm is validate-once-at-test-time, or periodically at a low duty cycle. **We validate 100% of
candidates, in production, at 55–83% of phase wall.** That is a defensible novelty claim *about the defect*.

### 3f.2 ⭐⭐ THE DIRECT PRECEDENT FOR T1 — and it is stronger evidence than I expected
**ModelTamer** (Sharma & Kumar 2022, *MBE* 39(11):msac236) **[READ]** subsamples **unique site patterns** for model
selection: **98–99.9% time reduction** (mammal 1 Mbp: 106 CPU-h → **0.81 h ≈ 130×**), and — the load-bearing number —
***accuracy ≥99% (same model as the full MSA) for g ≥ 0.5% of patterns***. Their single failure (Lassa Virus) picked the
second-best model where *"the difference in BIC between the top two models was less than 10 … statistically
indistinguishable."*

🔑 **This supports T1 *a fortiori*.** They subsample to **SELECT** a model from partial data — hard. We subsample only to
**DETECT divergence between two implementations evaluated on the SAME patterns** — strictly easier, and with no
statistical estimation involved. If 0.5% of patterns suffices to pick the right model, 0.44% is ample to catch a bug.

### 3f.3 ⭐ THE DIRECT PRECEDENT FOR THE LADDER — BEAST 2 already ships it
**BEAST 2** (beast2.org, 2019-04-29) **[READ]** performs, *in normal production runs*, a from-scratch recomputation
check: *"BEAST checks every 10,000 samples whether the posterior calculation obtained while caching equals the
posterior"*, agreeing only *"up to a small error due to machine precision"*, aborting *"after 100 such messages"*, with a
debug flag that raises the rate. ⇒ **periodic + tolerant + rate-limited + debug-escalation is established practice.**
⚠️ Two honest caveats: it is **CPU-vs-CPU** (it validates cache invalidation, not a GPU port), and its duty cycle is
**1-in-10,000 (~0.01%)** — so it is a precedent for the *shape* of the ladder, not for our tolerance.

### 3f.4 🔄 THE FRAMING INVERSION — the CPU is probably the WORSE estimator
**[SUMMARY]** Higham (2002, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.): sequential summation error grows
**O(n·u)**; pairwise/tree summation **O(log₂n·u)**. NVIDIA CCCL documents its reproducible accumulator giving *"tighter
error bounds than the standard pairwise summation traditionally used in parallel reductions."*

⇒ **A GPU tree reduction over 935k patterns has a structurally BETTER error bound than the CPU's sequential loop.** The
self-check's implicit premise — *the CPU is the reference truth and the GPU is the suspect* — is **backwards on
error-bound grounds.** ⚠️ Confidence MEDIUM: from summaries, constants unverified (arXiv:2107.01604 would not extract),
and the pruning recursion is not a flat sum (rescaling complicates it; **no source found that analyses this**).

### 3f.5 ❌ A CERTIFIED ON-GPU ERROR BOUND IS AN OPEN PROBLEM — do not plan around it
**No literature found** applying Higham-style *running* error bounds to GPU reductions, and none in phylogenetics.
ABFT checksums (Wunderlich et al. 2013 IOLTS; A-ABFT 2014) **[SUMMARY]** are cheaper than recomputation but **detect
faults (bit flips), not rounding error** — they do not solve tolerance selection. **⇒ this is why T0 above is labelled a
plausibility check, not a certificate.** A running error bound would be a genuine novel contribution but carries real
risk; **T1–T3 is the de-risked primary path.**

### 3f.6 💰 EVERY PUBLISHED CERTIFICATION MECHANISM IS CHEAPER THAN OURS
| mechanism | cost | source |
|---|---|---|
| ReproBLAS reproducible summation | **≥8%** (parallel reduction) | Demmel & Nguyen **[SUMMARY]** |
| NVIDIA CCCL `gpu_to_gpu` RFA (bitwise across architectures) | **20–30%** | NVIDIA CCCL blog **[READ]** |
| NVIDIA CCCL `run_to_run` (default; same-GPU determinism) | *"slightly slower"* | same **[READ]** |
| **our per-candidate CPU postorder** | **55–83%** | ✅ this document |

⇒ **if the real requirement behind the self-check is REPRODUCIBILITY rather than accuracy, it can be bought outright for
8–30%** — and that directly addresses this project's own historical `+I`/`+R` non-determinism.

### 3f.7 🏗️ ARCHITECTURE — BEAGLE 3 already proved the alignment-resident refactor
**[READ]** BEAGLE 2 required *"multiple instances of the library, one for each data subset"*; BEAGLE 3 made subsets
*"share a library instance"* ⇒ dengue 10-subset analysis **7.2 GB → 3.3 GB**, *"run-time performance improvements as high
as 5.9-fold."* Structurally the same refactor as §3e.8 (theirs across **data subsets**, ours across **models**). Plus
**MAGMA Batched [SUMMARY]** (2.8–3× over cuBLAS/MKL): *"the overhead of launching kernels and managing resources can
dominate"* for many small tasks — the exact symptom that killed `JOLT_MF_DEVUSE`. Our per-candidate 4×4/20×20
eigen/exponential work is the archetypal batched-BLAS regime.

### 3f.8 🔴🔴 WHERE THE LITERATURE CONTRADICTS US — confront this, do not route around it
**Abadi, Azouri, Pupko & Mayrose 2019, *Nat Commun* 10:934** **[READ]**, over 7,200 simulated + empirical datasets:
- *"AIC and BIC disagreed on 62% of the empirical datasets"*;
- yet *"less than 10% of the 7200 simulated datasets resulted in different topologies"*, and criteria agreed on topology
  for **>83%**;
- using **GTR+I+G always** scored *"even better than under the true model"*; **JC** was *"only ~2% lower."*

Corroborating: **ModelTest-NG** (Darriba et al. 2020) **[SUMMARY]** recovers the true generating model only **81–85%** of
the time *with exact arithmetic*.

⇒ **the exact identity of the BIC-argmin is not, on this evidence, a scientifically load-bearing quantity.** This
undercuts not only the `1e-6` gate but **any** framing in which validating the ranking to high precision is intrinsically
valuable — including, partly, my own §3e.5 "stronger where it matters".

🔑 **THE DEFENSIBLE REFRAMING (adopt this in the thesis):** the contract we owe is **fidelity to the reference
implementation** — a user expects IQ-TREE's GPU path to reproduce IQ-TREE's CPU path — which is an **engineering
contract, not a statistical necessity.** That framing survives Abadi et al.; *"we must get the best model right"* does
not. It also sets the right target for validation: **agreement with the CPU implementation**, which is exactly what a
per-pattern comparison measures and what a 118-BIC total-only gate does not.

### 3f.9 🕳️ THE GAP — GPU model selection appears unpublished
Searched ~8 phrasings. Everything found accelerates **tree search or scoring**, never **model selection as a phase**:
a GPU IQ-TREE 2 scoring-function **poster** (Kang/Moshiri/Rosing 2021, 32× on the scoring function); an FGCS 2025
multi-platform **parsimony** scoring paper **[UNREAD — 403, authorship unverified]**; jModelTest 2 and ModelTest-NG are
**MPI/multicore only**. **No GPU ModelTest, no GPU ModelFinder, no published OpenACC IQ-TREE port** (⇒ the collaborator's
port appears **unpublished**). ⭐ **Strongest single artifact for the gap claim: IQ-TREE 3's own 2026 flagship paper
(Wong et al., *MBE* 43(5):msag117) [READ] contains NO GPU content whatsoever** and lists no ModelFinder speedup.
⚠️ Absence of evidence across 8 phrasings is not proof of absence — state it as "we found none", not "there is none".

The active non-GPU directions are **site subsampling** (ModelTamer), **search-space heuristics** (jModelTest 2 filtering,
PartitionFinder relaxed clustering — Lanfear et al. 2014), and **ML surrogates** (ModelTeller, Azouri et al. 2020;
Tinh & Vinh 2024 **[UNREAD — 403]**). **Our T1 is the subsampling idea redeployed from selection to VALIDATION — which
is, as far as this search reaches, new.**

---

## 3g. 🗺️ THE EXECUTION PLAN — phase by phase (measure → build lean → gate → graduate)

> **Governing principle (from the red-team): MEASURE BEFORE YOU BUILD.** The lean form reuses primitives that already
> exist — `JOLT_SELFCHECK_STRIDE` (`phylotreegpu.cpp:2594`, already trusts `joltLnL` on skip), the BIC ranking
> (`getBestModelID` `phylotesting.cpp:334`), and a **BIC-ordered top-K margin detector already in the tree**
> (`:1359-1404`: `margin=|df−be.df|/2`, `lead=be.BIC−bi.BIC`). So **T3 "verify the finalists" is a REUSE, not a build.**
> Constraints honoured throughout: **clone before touching source** (never canonical `iqtree3-jolt-merge`); **assistant
> does NOT push GPU source** (build+gate only, human pushes); **proof-of-build sentinel** every job; gate on
> **selection-invariance** DNA+AA+**partitioned `-p`**; new flag **default-OFF, byte-identical when unset**.

### PHASE 0 — MEASUREMENT / GO-NO-GO (no source build; zero- or one-build jobs) — *do all before writing code*
| # | spike | question it settles | kill condition |
|---|---|---|---|
| **0.1** ⭐ | **finalist-count + T3 cost** on DNA-1M, AA-1M, **avian (real, near-degenerate)**: how many models fall within BIC margins {2,6,10} of the leader? (postorder cost already measured: DNA 1.49 s, AA 8.38 s) | **sets the hard cap K, and whether capped-T3 is cheap or reconverges** | if real-data finalist clusters are routinely >20, capped-T3 ≈ full cost ⇒ the win shrinks to just "tighten `rel`" |
| **0.2** ⭐ | **"is the lean form already the whole win?"** — run pure-trust (`NOSELFCHECK`) + a *manual* offline finalist re-verify, compare best-model + top-K BIC vs the full self-check | **whether the elaborate ladder is even needed** | if capped-T3-alone preserves selection on all three ⇒ **T0/T1/T2 are dead; ship the lean form** |
| **0.3** | **`rel`-tighten is free** — rescan existing self-check logs for max observed `rel` (measured 9.3e-12), set 1e-6→1e-9, confirm ZERO false-trips on a clean DNA+AA run | independent, shippable correctness win — makes the *existing* backstop meaningful | if a clean run trips at 1e-9 ⇒ back off to the tightest zero-trip value |
| **0.4** | **diffuse-bug sensitivity oracle** (only if a fidelity tripwire is wanted) — `JOLT_IR_NOFDFIX` reinstates the real +10-nat `+I+R` bug; does capped-T3 (or a uniform-T1 monitor) catch it? | whether a continuous monitor earns its keep beyond capped-T3 | if capped-T3 already catches it ⇒ no T1 needed |

**Phase-0 decision gate:** if 0.2 shows capped-T3 preserves selection AND 0.1 shows a small K ⇒ **skip to a minimal Phase 1** (stride + capped-T3 + tighter `rel`, no T0/T1). Otherwise add T0.

### PHASE 1 — BUILD THE LEAN VALIDATOR — ✅ IMPLEMENTED (worktree `iqtree3-mfvalidate`, commit `e120cd79`, default-OFF `JOLT_MF_VALIDATE`)
🔴 **DESIGN PIVOT during implementation (grounded in source):** the doc first specified an **end-of-loop reconstruction**
of each finalist. **That is INFEASIBLE — the fitted branch lengths are NOT persisted.** In `evaluate()` the fitted tree is
assigned to a **local** `string tree_string` (`phylotesting.cpp:2675`) that is discarded; the `CandidateModel::tree`
member stays empty on the MF path, and `CandidateModel::restoreCheckpoint` (`phylotesting.h:203`) reads back only
`logl/df/tree_len`, **never the tree**. So a finalist cannot be re-evaluated at its true brlens after the loop —
reconstruction would score the wrong point and cause **false corrections.**
**⇒ Implemented instead as IN-CONTEXT competitive verification, which reuses the live fitted state:**
1. **`tree/phylotreegpu.cpp` (~:2650):** under `JOLT_MF_VALIDATE`, skip the per-candidate CPU postorder — trust `joltLnL`
   in the loop (like `NOSELFCHECK`, but paired with the verify below). `clearAllPartialLH()` at `:2568` still runs first.
2. **`main/phylotesting.cpp` `evaluate()` (after `:2675`):** with the fitted model+brlens **still live** and the partials
   already cleared by JOLT's write-back (so `computeLikelihood()` is a **cold CPU postorder == the old self-check**), run
   that CPU check **only** for candidates whose BIC is within `JOLT_MF_VALIDATE_MARGIN` (default 15) of the best-so-far;
   correct `logl` at `rel>1e-9`. Non-competitive candidates are trusted. This is the "verify the finalists" idea, done
   streaming instead of end-of-loop — **no reconstruction, no brlens problem, exact reuse of the self-check.**
3. **rel tightened to 1e-9** in that check (vs the near-decorative 1e-6, §3e.2).
4. **Scope:** single-alignment serial MF. Super-alignments (`-p`) never set the skip ⇒ keep the per-candidate check
   (byte-identical). T0 device tripwire deferred (only if Phase-0 shows a monitor is needed).
⚠️ **Known v1 limitation:** streaming over-checks candidates that were transiently near-best during the descent (safe,
never under-checks a competitor); and a **gross** divergence (`rel>1e-6`) on a competitor is corrected to the honest
`cpuLnL` but **not CPU-re-optimised** (the stock path's NaN→re-opt) — safe for selection (never over-credits), measured
divergence is ~1e-11 so this branch ~never fires. Both documented; the gate measures selection-invariance regardless.

### PHASE 2 — GATE (the selection-invariance harness; this is where every prior gate was weak)
- **Best model + full top-K BIC order UNCHANGED** vs the full self-check, on **DNA-1M, AA-1M, avian (real)**, and a
  **PARTITIONED `-p` cell** (⚠️ mandatory — every gate in this project was once partition-blind).
- **Proof-of-build sentinel** (`strings|grep` the new flag, `grep -c` not `-q`); **wall measured** vs the projected
  headroom (§3e.7); engagement proven by a marker drop, not by the number under test.
- Byte-identical when the flag is unset (the ship invariant).

### PHASE 3 — GRADUATE / DECIDE
- **`rel`→1e-9 (0.3/1.3)** graduates FIRST and independently — a free correctness win, no behaviour change to selection.
- **Lean validator** graduates default-ON (kill-switch) **iff** Phase 2 holds selection-invariance on all cells AND
  captures material wall on DNA **and** AA. **Human pushes GPU source** (assistant builds/gates only).
- **Update the stale in-tree comments** (`:2783` "~25%", `:2577` "~11%", `:2589` "loop is OMP-parallel") — they encode the
  *old* regime and now mislead (§3d).

### PHASE 4 — SEPARATE PROJECT (do NOT couple): the 96 s serial floor
The `NOSELFCHECK` floor (~96 s DNA / 910 s AA) is the arena + serial setup, **not** the self-check (§3e.8, falsified
unlock). Attacking it is an **eligibility-gated refactor of `phylotesting.cpp:2560`** that must preserve the ~11.7 GB
arena for the ~19 JOLT-declined candidates — its own scope, its own risk assessment. On **AA** this is the bigger prize
(residual 910 s ≫ DNA's 96 s) and is **batched-BLAS / cross-candidate-residency** territory (§3e.8 direction, BEAGLE-3
precedent §3f.7). Sequence it *after* Phase 3 lands.

### 📌 CRITICAL PATH (what actually blocks what)
`0.1 + 0.2` (one GPU job each, cheap) → decide K and whether T0 is needed → `1.1–1.3` (one clone, days) → `Phase 2`
(one gate job, DNA+AA+avian+partition) → graduate. `0.3/1.3` (`rel`-tighten) runs in parallel and ships first. Phase 4
is deferred and independent. **First concrete action: submit spike 0.1** (finalist-count on the three datasets — the one
number that decides whether this is a real lever or just "tighten the gate").

---

## 4. Design tiers (red-team-corrected)

**TIER 1 — coverage (build-ready / near):**
1. **+I mirror invariant term** — the one sound fix (covers +I / +I+G self-check offload). §2 row 1.
2. **Mixture dispatch flip** (`JOLT_MIX_HOSTDRIVEN`) for C10–C60.
3. **+I+R** — resolve §3 first (spike), then it inherits the +I mirror term.

**TIER 2 — walltime frontier (REFUTED as headlined; each needs a de-risk spike, NOT a build):**
- **Batched-candidate kernel** (model dimension in the grid) — **contradicted by our own J.7** ("K-batching DEAD /
  occupancy DEAD") + the eigen `__constant__` layout fits only ~2 AA models (batching → global mem → MORE DRAM/L2
  traffic, the binding resource). SPECULATIVE; needs a 2-vs-1 AA microbenchmark before it's called a lever.
- **Residency + launch/`toSymbol` fusion** — residency itself is a <2s enabler (K.2), NOT a perf lever; only the
  host-API launch storm (~13s+8s @100K) is reclaimable and it has a graveyard/race history (J.7). Secondary, sized not
  headlined.

## 5. TEST MATRIX (this doc's deliverable — each a gate job)

| # | test | what it proves | job / script | status |
|---|---|---|---|---|
| T2 | **+I+R trace spike** | resolve §3 | `gems_ir_trace.sh` | ✅ **DONE 173843042 — RESOLVED (§3): constant freeRate∧optPinv bug** |
| T3 | **mixture dispatch flip** | `JOLT_MIX_HOSTDRIVEN` C20 GPU vs CPU | `gems_ir_trace.sh` | ⚠️ **INCONCLUSIVE** — C20 GPU path ENGAGED (`[MIX-TILE] nTile=2/3`) but mis-scoped (full tree search hit 5h walltime); **re-run with `-te` fixed tree** |
| T5 | **+I+R constant-10 bisect** (NEW, from T2) | localize where ~10 enters `gpu_jolt_optimize` freeRate∧optPinv (gaugeFix / out_props / objective const) | `gems_ir_bisect.sh` | **NEXT — highest value (real-data avian winner)** |
| T1 | **+I mirror term** gate | non-+I byte-identical + +I/+I+G mirror rel≤1e-6 vs CPU | `gems_iplus_mirror.sh` | **STAGED, scoped to +I/+I+G** (T2 proved it does NOT help +I+R) |
| T4 | **batched-candidate microbench** | 2-vs-1 AA const-mem + DRAM | `gems_batch_probe.sh` | designed, low priority (de-risks a refuted lever) |

## 6. Landed results (2026-07-15) + next steps

| job | result | grounding |
|---|---|---|
| **hdna1m 173832774** | DNA-1M full MFP (Hashara base cmd): **bffdc16e 4392s vs promoted 239fb5f2 12159s = 2.77× (mi3 lever)**; both F81+F+G4; full offload CONFIRMED (posture `cache=ON +I=ON +R=ON brlen-maxiter=DNA:2`, hook=627). **Still ~2× behind Hashara ~2217s** ⇒ DNA no-ctf is architectural (residency), NOT coverage. | posture banner + wall |
| **ctfavab 173832078** | avian selects **GTR+F+I+R4** (oracle); CTF-lockstep = SAFE-NEGATIVE (legacy==fixed==GTR+F+I+R2); abort-fix held; CTF under-resolves +R order. | best-fit .iqtree |
| **rtilegate 173815930** | **G1 GREEN: +R tiling nTile1-vs-2 rel=0 (bit-identical same-device)** — stronger than the ≤1e-12 claim; the M1 concern is cross-GPU only. G2 AA-1M LG+I+R4 walltime (no OOM, but the +I+R→CPU slowness didn't finish in 6h). | gate log |
| **covgate 173824352** | 🔴 **HARNESS BUG** (reached-hook=0/rel=NA in EVERY cell — parser/env, same class as prior gate bugs; NOT a code regression). Coverage stands on census 173824353. **Fix the covgate parser before trusting it.** | gate log |
| **irbestwb 173898475** | ✅ **②a +I+R WRITEBACK FIX GRADUATED (§3c).** `JOLT_IR_FDFIX` default-ON: `-m MF` +I+R worst\|jolt-cpu\| **10.00→1e-4** DNA+AA, self-check passes, **no CPU fallback**, best-fit unchanged; DNA `-te` = **exact CPU MLE** (avian GTR+F+I+R4). Root cause = pinv-FD props-deficit (`nFreeQ==0`: AA + JC/F81 DNA). Plan+red+blue-team reviewed. ⚠️ unmasks AA 0.07-nat convergence gap = ④ (negligible-for-selection). NOT pushed (human). | gate + reviews |
| **mfgauge 173893718** | 🔴 **+I+R +10 RESOLVED (§3b).** `-m MF` gauge probe: `worst\|d\|=0.0000` (gaugeFix invariant, DNA+AA) AND `joltLnL−cpuLnL=+10.00` (R2–R5, both) — **identical to `-te`, so NO separate `-m MF` bug.** Optimiser finds MLE (`joltLnL`), writes back params 10 worse (`cpuLnL`), mirror==cpuLnL exact. Re-eval RETRACTED harmful; real fix = writeback reconciliation. | gauge trace + DEVCHECK |
| **mfdevuse 173837501** | 🔴 **DEVUSE (Direction-A self-check offload) REFUTED at scale.** Selection-invariant vs CPU oracle + hardening held (+R-skips=0), BUT wall: dna-100K +7.5%, aa-100K +13.4%, **dna-1M +0.3% (noise), aa-1M −266% (3.7× SLOWER)**. The DEVCHECK "7–11× cheaper" measured only the kernel; the mirror's PER-CALL setup (echild rebuild + ~100MB tip re-upload + **non-tiled ~59GB partial arena**, every candidate) swamps the saved CPU self-check at 1M. **Stays default-OFF; retired as a production lever.** The real self-check offload needs a GPU-RESIDENT mirror (reuse tip/echild across candidates, tile the arena) — architectural, not this per-call mirror. Confirms: per-candidate self-check offload is NOT the MF walltime lever. | gate log |

**NEXT STEPS (ranked by grounded value):**
1. ✅ **DONE — ②a +I+R writeback fix GRADUATED** (`JOLT_IR_FDFIX` default-ON, §3c, job 173898475). Coverage hole closed
   (avian GTR+F+I+R4 now exact-MLE on GPU, no fallback). **NOT pushed (human).** NEXT on this thread = **④ AA convergence**:
   the fix unmasked JOLT's AA LG+I+R4 optimum 0.07 nats below CPU (reject_frac 0.55 zigzag). Blue-team method: FIRST a
   per-arm reject-attribution print (`JOLT_IR_FREEZE_MODEL`/`FREEZE_BRLEN`); if model-arm-caused, **bound the secant
   curvature** (floor `|ddY|,|ddZ|,|ddP|`) — beats the decoupled-`mu` design (there is no grounded per-arm attribution yet).
2. **T1 — +I mirror term** (+I/+I+G self-check offload). Independent, build-ready, 6-edit kernel change.
3. **T3 re-run** with `-te` fixed tree (validate mixture dispatch correctness + latency, no tree search).
4. **covgate parser fix** (or retire — census already proves coverage).
5. DNA-beating (Hashara) remains **architectural** (residency/fusion), not in this coverage scope — do NOT conflate.

**Standing constraints:** clone before touching source (work tree `iqtree3-mfdevcheck`, never canonical); pin
`-starttree PARS`; validate DNA **and** AA; assistant does NOT push GPU source; proof-of-build sentinels on every job;
gate on **selection-invariance** (mirror value differs ~1e-11) not byte-identity, except non-+I which MUST be byte-id.
