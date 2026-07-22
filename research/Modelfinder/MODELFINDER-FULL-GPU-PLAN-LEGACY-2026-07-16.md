# ARCHIVED — Full-GPU ModelFinder: removing host↔device synchronization from the MF path

> Superseded on 2026-07-21 by `MODELFINDER-FULL-GPU-PLAN.md`. This file is retained as the historical performance and coverage plan; its corrected/overturned conclusions must not be used as the current FreeRate convergence design.

**Status:** DRAFT, 2026-07-14. Audit (§A–§C) is source-grounded (file:line, spot-verified). The offload
**design direction (§E) is now RESOLVED by nsys** (job `173763069` H200 LANDED; `173761198` A100 confirming;
no-`--ctf` MF `host_resid` sweep `173760019` H200 done / `173760016` A100 pending). **Verdict: `kern (40%) ≪
device (85%)` → BOTH Direction A (host self-check) and Direction C (per-launch upload cache + sync fusion) are
real; ~60% of the 1M wall is attackable before touching kernels (§E.1).** This mirrors how tree search was done
(audit → profile → phased device-resident offload with bit-identity gates), per
`research/Treesearch/FULL-GPU-END-TO-END-PLAN.md`.

**⚠️ 2026-07-14 — Phase-1a guard BUILT (`5ba5e2f1`) and A/B measured; live results land in §J. Two honest
corrections there: (1) the OFF baselines already carry EVERY shipped tree-search lever, so those are not a pending
MF win (the biggest, `JOLT_SCREEN_CACHE`, fires on MF but removes ~0 — §J intro); (2) the "451.5s / ~510s
attackable" of §E is transfer-VOLUME, which converts to only ~10.7s wall at 100K because H2D overlaps host —
attackable-volume ≠ wall-win (§J.5). The DNA-1M `-m MF` headline is still pending (§J.3).**

**⚠️ READ §H FIRST — red-team (2026-07-14, source-verified) found the draft AND the first audit
mis-identified the execution regime: single-alignment no-`--ctf` MF is the SERIAL `test()` path
(`phylotesting.cpp:3894`), NOT the parallel `:4547` loop. That CONFIRMS Direction A (attack the
serially-exposed self-check), makes the mutex/Direction B unreachable in the measured regime, and corrects
several §A/§D/§E mechanism claims. §I = why this wasn't caught earlier (CTF-hiding = CONFIRMED PRIMARY).**
§A–§G is the original draft; §H/§I supersede it where they conflict.

**Line numbers are for the graduate clone** `/scratch/rc29/as1708/iqtree3-graduate` (the promotion tree).
They differ from the `iqtree3-l0` clone the older docs cite (e.g. the MF OMP loop is `phylotesting.cpp:4547`
here, not `:4097`). The CUDA TU is `tree/gpu/gpu_lnl_intree.cu`.

---

## 🔴🔴 2026-07-16 (LATER, CORRECTED) — THE SELF-CHECK IS **NOT** THE DNA LEVER; FINALISTS-ONLY = A VERIFIED REGRESSION (READ FIRST — SUPERSEDES the "RED-TEAM + perf VERDICT" block just below it)

A second adversarial red-team + grounded head-to-head **overturned the "self-check is THE real lever / defer to finalists" conclusion below.** Corrections, every number re-verified from job logs this session:

- **The "67.7% self-check" was an `-nt 1` profile on a 12-core node** (job 173929005 ran `-nt 1`). At the **DEPLOYED `-nt 12`** the CPU self-check is **~25% of MF wall** (it is already GOMP-parallel there), NOT 68%. Profiling a serial config and shipping the conclusion is the exact "optimise the thing the profiler pointed at, not the thing that decides the outcome" trap.
- **Even a FREE self-check LOSES DNA.** Grounded morning head-to-head (her job 173919899, ours job 173931905, same node, exact cmd, no `--ctf`, `-nt 12`): her **OpenACC-JOLT DNA-1M MF = 1142.4s**; ours self-check **OFF = 1300.9s** (the raw ceiling) and correctness-intact ≈ **1501.7s**. Removing the self-check entirely still trails her by ~1.14×. So it is arithmetically incapable of winning DNA.
- **FINALISTS-ONLY = DO NOT BUILD — VERIFIED 0.85× REGRESSION** (design agent + independent check). `CandidateModel::evaluate` (`phylotesting.cpp:2406`) builds+destroys a fresh `IQTree`, so re-verifying a finalist = a FULL re-optimise at ~4× the self-check (15.79s vs 3.94s @nt12); the pool is ~110–142 evaluated models (filterRates/filterSubst prune first), not 968; break-even K=27.5 but the safe window needs K≈47 ⇒ **2045s = an 18% REGRESSION.** No K is both safe and profitable.
- **THE REAL DNA GAP — ✅ RESOLVED 2026-07-16 by `dnares` (job 174003961, DNA-1M `-m MF`, nt12, binary `020ff472`). VERDICT: near HOST CEILING — no clean new DNA lever.** Two arms: self-check **ON = 1464.9s**, **OFF = 1304.1s** (both best `F81+F+G4`) ⇒ the self-check at nt12 is **160.8s = 11.0%** of wall (this **settles the "25% vs 68%" question — it is ~11%, even lower than the retracted 25%; the +I+R fix cut the fallback work). The OFF-arm flat `perf` self-time (self-check already gone) is **diffuse across three buckets, none a single fixable symbol**: CUDA driver storm **~44%** (spread over ~20 unresolved `libcuda` addrs, top 4.24% — the launch/`toSymbol` storm already classed a GRAVEYARD/hidden-race item, §J.7), **CPU Felsenstein likelihood ~29%** (`computePartialLikelihoodSIMD` 15.18 + `dotProductTriple` 7.09 + `bufferSIMD` 5.79 + `DervSIMD` 1.15 — this is the **CPU-declined +I families**, i.e. a **TIER-1 COVERAGE** target, NOT a new lever), **OMP wait ~21%** (`kmp_flag_64::wait` 14.08 — enabler-only, the postorder's own parallel inefficiency, §N.6 C-3). **⇒ the pre-committed "diffuse ⇒ host ceiling ⇒ STOP chasing a new DNA lever" branch FIRES.** The only reclaimable DNA slice is the 29% CPU-declined +I, and that is closed by Tier-1 coverage (the +I mirror term + the pureinvar M.2 closures), already planned — **not** a Tier-2 win. (Original framing kept: *~933 CPU-s residual, diffuse ⇒ ceiling; a dominant host symbol would have been the lever — there was none.*)
- **SCORECARD, MF-phase to MF-phase (grounded):** **DNA we LOSE** (ours fix 1501.7s vs her JOLT 1142.4s = **1.31× behind**; but ahead of her *stable* 1679.8s). **AA we WIN** (ours `-m MF` ~2845s vs her 4605.1s = **~1.62× ahead**; AA is compute-bound / near-full-GPU). Full MFP: her DNA 2018.9s / AA 10136.0s; we have not re-run full MFP this session.
- **A SEPARATE CORRECTNESS FIND (not a speed lever) — ⚠️ CORRECTED 2026-07-16: this is a RE-DERIVATION of an ALREADY-GRADUATED fix, not a new closure.** The **+I+R export-normalisation bug** (`gpu_lnl_intree.cu`: the pinv forward-FD perturbs `catProp_v` to `baseP+ep`; for `nFreeQ==0` (JC) nothing resets it; a reject-exit exports `Σprop+pinv = 1−1e-4` ⇒ error `≈ ep·Nsites` = 10 nats @100K / 100 @1M) was **already root-caused, implemented, and GRADUATED default-ON on 2026-07-15 as `JOLT_IR_FDFIX`** (`iqtree3-mfdevcheck`; gate **job 173898475**: `-m MF` +I+R worst `|jolt-cpu|` **10.00 → 1e-4** DNA+AA, best-fit unchanged, DNA `-te` = **exact CPU MLE** on avian GTR+F+I+R4). See **`MF-FULL-GPU-COVERAGE.md` §3c**. This session I **independently re-implemented the identical `applyPinv(baseP)` fix** in `iqtree3-mfresident` under a different flag (`JOLT_NO_PINVFIX`) and **re-validated** it (job 173995856: Σ→1.0 vs 0.9999, MISMATCH 1→0, winner unchanged) — a **DUPLICATE**, not a first closure. What the re-derivation genuinely ADDED: (a) the clean `ep·Nsites` arithmetic; (b) the **+R5 cascade** characterisation (job 173999251: old deterministic; the fix shifts **42** non-competitive models — 23 +R5 / 4 +I+R / 15 +I+G — by ≤**40.56** BIC; competitive top-15 stable); (c) the +R-favouring shipping gate (job 174003950, in flight). **The gauge (rate↓/branch↑) is EXACTLY lnL-invariant** — two earlier "gauge is lossy" mechanisms RETRACTED. **⚠️ merge implication:** the fix lives in TWO trees under two flag names ⇒ **reconcile to ONE** (`JOLT_IR_FDFIX` is the graduated one). **AA +I+R still ~0.07 nat below the CPU MLE** (a separate LM-convergence issue, "④", NOT fixed by FDFIX). See `MF-FREERATE-HIGHK-PLAN.md` + memory `project-gpu-freerate-handicap`.

**NET:** the whole self-check → finalists-only direction is **retired**. DNA is decided by the unnamed ~933 CPU-s residual (under profile); AA is already a win. Everything below in the older block is kept for the record but its "THE REAL LEVER / FIX finalists-only" conclusion is WRONG.

---

## ⭐ 2026-07-16 — RED-TEAM + perf VERDICT ~~(READ FIRST)~~ **[SELF-CHECK-AS-LEVER CONCLUSION RETIRED — see the corrected block above; the three dead-ends #1–#3 below still hold]**

Three dead ends were closed by adversarial review + a **CPU-sampling reprofile** (`perf`, job `173929005`, DNA-100K/1M `-m MF` **nt1** — note: nt1, the config the corrected block above flags as unrepresentative). Every number re-derived from `mfbtl_173785379/d1m_on.sqlite` + the perf data:

1. **Constant-cache (model-const re-upload skip): DEAD ~1.00×.** DNA/AA × 100K/1M A/B all ~1.0×; redundancy print silent even at 1M ⇒ model constants are uploaded per-outer-iter (few), not the storm. (job 173922034 / sweep 173924387: DNA-1M off 3250.1s vs on 3245.3s = 1.001×.)
2. **Async val-copy port (`JOLT_MF_ASYNCVAL`, Direction C): DO NOT BUILD ~1.02× (hard-ceiling 1.04×).** The design agent found it's a cheap 3-line gate reusing the already-wired `JOLT_TS_ASYNC` valpool — **which is the trap**. `cudaMemcpyToSymbol` = **209s HOST but only 11.6s DEVICE transfer** (avg 2369B) ⇒ 95% is driver/sync overhead; same fingerprint as #1. An async copy can't help.
3. **On-device `val0/1/2` compute: DEAD.** The perf reprofile was run to size the `setVal exp()` loop (the one lever that could have been past 1.1×). **libm/exp = 0.2% of host self-time.** The `val` path (`gpu_jolt_optimize` lambdas) is ~0.5% each. Not the cost.

🔴 **RETIRED (kept for the record):** the block below concluded the per-candidate CPU self-check (`phylotreegpu.cpp:2502`, 67.7% of host self-time at **nt1**) was "THE REAL LEVER" and proposed deferring it to finalists. The corrected block at the top of this section overturns both: at nt12 the self-check is ~25%, a free self-check still loses DNA (1300.9s vs her 1142.4s), and finalists-only is a measured 0.85× regression.
> **THE REAL LEVER (perf-grounded, was mis-scoped as "sync"):** DNA-1M `-m MF` is **69.7% GPU-busy / 30.3% idle**, and the idle is dominated by **CPU Felsenstein likelihood = 67.7% of host self-time** (`computePartialLikelihoodSIMD` 59.6% + `computeLikelihoodBufferSIMD` 6.3% + `dotProductTriple` 5.3% + `computeLikelihoodDervSIMD` 1.3%; CUDA driver/sync 19.7%; exp 0.2%) — the per-candidate CPU self-check. **[nt1 profile — at nt12 this is ~25%, see corrected block.]**
> **FIX (retired):** defer the self-check to finalists only. **[VERIFIED 0.85× REGRESSION — do not build.]**

---

## 0. The one-sentence problem

"GPU ModelFinder" is **host-orchestrated per-candidate GPU offload**, not a GPU-resident loop: one CPU thread
per candidate, each candidate's GPU work serialized behind a process-global mutex, each candidate ending in a
**full CPU `computeLikelihood` self-check**. Tree search escaped this with **L0** (a GPU-resident search loop
that returns a side-effect-free scalar and drops the CPU postorder). **The self-check cannot simply be dropped
for MF the way L0 dropped it, because in MF the returned lnL IS the deliverable** (BIC ranking selects the
model) — that is the load-bearing difference this plan must solve.

### 0.1 Why optimize — the mandate + the measured problem (grounded, 2026-07-14)

**Mandate:** Minh's directive is to run ModelFinder **without `--ctf`**, and the project decision (user,
2026-07-14) is that **CTF can no longer be relied on for ModelFinder.** CTF was the proven 7–13× MF lever
(§I) — removing it exposes the full cost it was hiding.

**Measured problem (job 173760019, H200, DNA-1M `-m MF`, no-`--ctf`):** total **2768s (46 min)**. The coarse
JOLT-diag timer reported **`Σdevice`≈85%** — but that "device" bucket lumps kernel + H↔D transfer + the
per-launch `cudaDeviceSynchronize` all together, so it is **NOT** a measure of irreducible kernel compute.

**🔬 nsys now DECOMPOSES it (job `173763069`, H200, DNA-1M `-m MF` no-ctf, `cuda_gpu_kern_sum`) — this
OVERTURNS the "85% = GPU compute" reading:**
| bucket | 1M (nsys wall 2828s) | 100K (nsys wall 75.5s) |
|---|---|---|
| **GPU kernel-busy** (device-measured, accurate) | **1119.6s = 40%** | **15.6s = 21%** |
| H2D transfer (attackable — see breakdown below) | **522s = 18%** | 4.8s = 6% |
| host / sync-wait residual | ~1175s = 40% | 54.5s = 72% |
| top-3 kernels (k1_node/kj_derv/kj_pre) share of GPU-busy | 92.6% | 90.8% |

**🔬 H2D size-decomposition (nsys `.sqlite`, CUPTI_ACTIVITY_KIND_MEMCPY, 1M) — this CORRECTS my earlier
toSymbol attribution, which was WRONG-AXIS:**
| H2D size bucket | ops | device time | share of H2D |
|---|---|---|---|
| **>16MB** (the 93.5MB `d_tip` re-upload) | **57,074** | **451.5s** | **86.5%** |
| 1–16MB (`d_ptnfreq`+`d_baseinvar`, 7.5MB each) | 114,148 | 62.6s | 12.0% |
| **<1KB (the 9.7M `cudaMemcpyToSymbol` coeff storm)** | 9,738,453 | **7.6s** | **1.5%** |

Max transfer = **93,522,700 bytes = exactly `ntax·nptn` (100·935227)** = the **`d_tip` byte array**. `setChunk`
(`gpu_lnl_intree.cu:2735`) re-uploads the **alignment-constant** `d_tip`/`d_ptnfreq`/`d_baseinvar` **every LM
sweep** (~505× per candidate, 57,074× total) even though its own comment (`:2731`) says they are *"CONSTANT
across the optimise call."* **⇒ the attackable transfer is ~510s of redundant re-upload of read-only data —
NOT the toSymbol storm (7.6s, a dead lever), and NOT the `kj_derv_fused_args` async path** (which only trades
3 toSymbol for 1 async memcpy = the 1.5%, and is default-OFF anyway — runtime `getenv("JOLT_TS_ASYNC")` in the
promoted `a07f61be` (`gpu_lnl_intree.cu:833`), `constexpr false` only in the pending flag-hygiene branch).

⚠️ nsys barely inflated THIS run — nsys wall 2828s vs un-profiled sweep wall **2768s (H200) / 4614.6s (A100,
job 173760016)** = ~2%, so the sync-tracing tax I worried about is small here; kernel-busy (1119.6s) and the H2D
(522s) are device-measured and real. **What is now settled: `kern (40%) ≪ device (85%)` — the coarse "device"
bucket is ~half real kernel and ~half attackable transfer+sync.** So the no-`--ctf` MF wall is **NOT purely
GPU-compute-bound**; **~510s at 1M is the `setChunk` re-upload of the alignment-constant `d_tip`/`d_ptnfreq`/
`d_baseinvar` (read-only, uploaded ~505× per candidate)** that a resident/cache-once buffer removes — a
correctness-neutral lever. The cost CTF hid (~90 full-data model fits vs ~3) is now paid in full, and ~a fifth of
it is redundant read-only re-upload, not compute.

**⇒ Two targets, scale-dependent, BOTH real (resolves §E's open question):**
- **Small N (100K): host-bound — 62% host/sync, kernel only 21%.** Direction A (kill the redundant host
  self-check) is the lever; this is the user's "40% on small datasets is too much."
- **Large N (1M): transfer+sync-heavy — kernel 40%, attackable transfer 23%, host 37%.** Direction C (cache the
  per-launch eigen/partial uploads + fuse the per-edge syncs) AND Direction A both bite.
The open question — is the "device" bucket real kernel or launch/sync? — is now ANSWERED by nsys: ~40% real
per-edge launches / CUDA-graphing the LM loop cuts it **without** a coarse screen). That hinge determines
whether "fast MF without `--ctf`" is achievable at all.

---

## A. Audit — host vs device, per candidate (source-grounded)

### A.1 The candidate loop is host, and each candidate is independent
- `main/phylotesting.cpp:4547` — `#pragma omp parallel num_threads(...)` across-model loop; `getNextModel()`
  work-steal `:4552`; `at(model).evaluate(...)` `:4581`. **Single-alignment MF runs here too** (the
  partition loop `:3065`/`:3066` `if(parallel_over_partitions)` is a *different* region; this model loop runs
  regardless).
- Each candidate gets its **own** `new IQTree(in_aln)` with its **own** CPU partial-LH buffers
  (`phylotesting.cpp:2407`) + host `initializeModel` incl. **host eigendecompose** (`:2431`). **No device
  state is shared across candidates.**

### A.2 Dispatch to JOLT (`model/modelfactory.cpp`)
- `:1597` `if (params->jolt) tree->optimizeParametersJOLT(fixed_len)`; NaN ⇒ CPU fallback (`:1614`).
- Mixtures (`getNMixtures()>1`) ⇒ `optimizeParametersJOLTMix` **only under `JOLT_MIX_HOSTDRIVEN`, else CPU**
  (`:1599–1611`).
- **+I+G multiplies the round-trips:** `optimizeParametersGammaInvar` (`:1368`) does a multi-start pinv sweep,
  `n_pinv_starts=4` under `--jolt` (`:1388`) — so a +I+G candidate = ~4 JOLT offloads **+ 1 extra host
  `computeLikelihood`** (`:1523`), on top of A.3's self-check.

### A.3 `optimizeParametersJOLT` body (`tree/phylotreegpu.cpp:2081`) — the per-candidate offload
| step | file:line | H/D | forces sync |
|---|---|---|---|
| eligibility gate | :2082–2232 | HOST | — |
| read eigen factors + `UinvRowSum` | :2235–2240 | HOST | — |
| DFS reindex → flat topology arrays | :2252–2276 | HOST | — |
| `joltGetTipPtnFreq` → tip[]/ptnFreq[] | :2292 (impl :1437) | HOST | 🔴 cache **bypassed** under OMP (:1442) ⇒ full O(ntax·nptn) rebuild **every candidate** |
| `base_invar` build (+I) | :2301–2315 | HOST | — |
| **`gpu_jolt_optimize(...)`** | :2346–2356 | **DEVICE** | returns scalar lnL ⇒ **mandatory H↔D** |
| brlen writeback + setters + `clearAllPartialLH` | :2368–2499 | HOST | invalidates all partials |
| **`double cpuLnL = computeLikelihood();`** self-check | **:2502** | **HOST full postorder** | **the load-bearing per-candidate round-trip** |
| `rel<=1e-6` gate ⇒ NaN/CPU fallback | :2525–2531 | HOST | — |

### A.4 Kernel side (`tree/gpu/gpu_lnl_intree.cu`)
- 🔴 **Process-global serialization:** `static std::mutex jolt_gpu_mtx; lock_guard ...` (`:2607–2608`) locks
  the **entire** GPU computation, because the eigensystem is in process-global `__constant__` symbols
  `g_Uinv/g_U/g_val*` (`:34–43`) and the `DevBuf` pool is process-global. ⇒ **only one candidate on the GPU
  at a time**, regardless of the OMP thread count.
- Per-call eigensystem H2D `cudaMemcpyToSymbol` (`:2622–2624`); free-Q re-uploads per FD step (`:2633–2641`).
- All per-edge D2H/`cudaDeviceSynchronize` are **internal to one candidate call** (`:2796–2824`, `:2981`,
  `:3022`); the `reduceDerv` host round-trip was already moved on-device in G.5.0 (`:2813–2824`,
  `part8-jolt-code-audit.md:50`).

### A.5 The mandatory per-candidate HOST round-trips to remove (the target list)
1. **CPU `computeLikelihood()` self-check** — `phylotreegpu.cpp:2502` (×~4 more for +I+G, `modelfactory.cpp:1523`).
2. **tip[]/ptnFreq[] host rebuild** — `:1442/:2292`, cache-bypassed under MF's OMP region.
3. **`gpu_jolt_optimize` return-to-host** — `:2346` (one scalar; internal per-edge syncs already reduced).
4. **Per-candidate host model init + eigendecompose + `__constant__` H2D** — `phylotesting.cpp:2431`,
   `gpu_lnl_intree.cu:2622`.
5. **Process-global mutex** — `gpu_lnl_intree.cu:2607` (structural blocker to concurrent candidate residency).

---

## B. The tree-search L0 pattern — and why it does NOT transfer for free

**L0 = `computeLikelihoodGPUResident`** (`phylotreegpu.cpp:2546`), hooked at the search loop's perturb
postorder (`iqtree.cpp:3643`) and the `doNNISearch` re-sum (`:3684`).
- **Resident state:** alignment-constant `tip[]`/`ptnFreq[]` in **members** `_gpuResTip`/`_gpuResPtnFreq`,
  rebuilt only on signature change (`:2616–2623`); persistent `DevBuf` arena in `gpu_jolt_optimize`.
- **Removed:** the CPU Felsenstein postorder — L0 calls `gpu_jolt_optimize(maxiter=0)`, a **pure,
  side-effect-free** GPU lnL (`:2650–2658`), **no writeback, no `clearAllPartialLH`, no CPU self-check**
  (contract `:2537–2545`).
- **Correctness-safe because:** (i) byte-identity gate — DNA/AA-1M `-B` RF=0, lnL rel `0.000e+00`, UFBoot
  197/197 (`FULL-GPU-END-TO-END-PLAN.md:210–214`); (ii) safe-regime guard `_l0_ok = ts_fused && ...`
  (`iqtree.cpp:3633`) restricting to the fused-NNI path that **never reads the now-stale CPU partials**;
  (iii) measure-mode `JOLT_L0=1` runs BOTH and keeps CPU value (`:3645–3653`) — the A/B that proves identity.

**Why it does not transfer for free (the crux):** L0's lnL only *ranks NNIs* — a wrong value is
self-correcting (the search retries; the final tree is CPU-reconverged), so dropping the self-check is legal.
**In MF the lnL IS the deliverable** — BIC selects the model from it. The self-check (`:2502`, gate `:2525`)
exists precisely to catch a **coherent-but-wrong** GPU optimum (kernel/regime failure, or a silently dropped
free parameter). So an "L0 for MF" needs a **new** correctness guarantee, not just a safe-regime guard.

**⚠️ Correction (nsys 173763069, 2026-07-14): L0 is TWO independent mechanisms — only ONE is correctness-gated.**
(1) **Drop-the-CPU-postorder/self-check** — correctness-gated, does NOT transfer to MF (above). (2) **Resident
`_gpuResTip`/`_gpuResPtnFreq` members + persistent `DevBuf` arena** — this is **correctness-NEUTRAL** (it only
stops re-sending byte-identical alignment-constant data each launch; the kernel and the self-check are untouched)
and it **DOES transfer to MF cleanly.** nsys+sqlite pin the cost precisely: **451.5s at 1M = 57,074 re-uploads
of the 93.5MB `d_tip` byte array** (max transfer = `ntax·nptn` exactly) by `setChunk` (`gpu_lnl_intree.cu:2739`)
+ ~62s of `d_ptnfreq`/`d_baseinvar` = **~510s of redundant read-only H2D**. This is a DIFFERENT layer than the
tree-search `JOLT_SCREEN_CACHE` (which caches the host *gather* — H1, ~0.87ms in MF, correctly demoted in §H);
this is the device *upload* of the already-gathered data, uploaded every LM sweep though `setChunk`'s own comment
(`:2731`) says the data is *"CONSTANT across the optimise call."* **⇒ mechanism-(2) caching = an intra-call
`setChunk` resident guard (skip the H2D of the already-resident chunk), correctness-neutral (the buffers are
written only by `setChunk`, read-only to every kernel), NO new guarantee needed** — the first, safest build.

---

## C. Precedents — what exists vs what does NOT (grounded)

- 🔴 **Device-resident mixture optimiser does NOT exist.** `gpu_jolt_optimize_mix` is a **name in one comment**
  (`modelfactory.cpp:1604`) — no definition anywhere (verified 2026-07-14). The real `optimizeParametersJOLTMix`
  (`phylotreegpu.cpp:2727`) is **host-driven**, launch-latency-bound (~4.4 s/outer on 400 sites), gated OFF
  behind `JOLT_MIX_HOSTDRIVEN`, and still ends in a CPU self-check (`:2938`).
- **CTF is NOT device-resident.** `runCTFModelFinder` (`phylotesting.cpp:1465`) coarse pass = the **same
  host-orchestrated per-candidate JOLT loop** on a subsample (`:1572`); refine = serial full-data loop
  (`:1621`). CTF shrinks pattern count + prunes candidates; it does **not** batch candidates on-GPU.
  `MF-FREERATE-HIGHK-PLAN.md:96`: cheap CTF-coarse-overhead path CLOSED; residual = de-globalise the
  `__constant__` mutex.
- **Cross-candidate GPU batching (PHALANX-BMF, `grid.z = model`) — DESIGNED, NOT BUILT**, "de-prioritized
  post-P3.0" (`00-MASTER-gpu-modelfinder.md:298`; `part4-jolt-optimizer.md:35/84`). Flagged **NO CLEAN PATH**
  for trial-branch batching (M× arenas; process-global `__constant__` + mutex serialize model fits) —
  `MF-FREERATE-HIGHK-PLAN.md:175/146`. One spike: `1.11× batched` at saturated 391 blocks = NO-GO, but that
  is the wrong (saturated) regime for the ~20-block coarse.
- **Removing/sampling the self-check — NAMED, DEFERRED.** `part9-full-mf-coverage-and-scaling.md:472`: "the host
  self-check at extreme nptn is the remaining WALL-TIME lever (not a correctness wall)… sampling/skipping it …
  is the throughput follow-up; capability comes first." The **intra-candidate** per-edge D2H is already
  on-device (G.5.0); the **per-candidate self-check + tip rebuild are not.**

**⇒ Nothing that delivers a device-resident MF loop exists in source today. The celebrated tip-cache and
reduceDerv levers are tree-search-scoped and do not fire on the MF path.**

---

## D. The redesign surface (mechanisms; correctness is the hard part)

Ranked per-candidate round-trips + candidate GPU-resident mechanism:
1. **CPU self-check (`:2502`)** → **on-device self-check**: an *independent* device recompute at the
   written-back params. `gpuComputeTreeLnLCleanRoom` already exists (`phylotreegpu.cpp:1997`, used for the -B
   snapshot `:2451`) — a candidate mirror. Alternative: **sample** it (only top-K by preliminary rank, as
   CTF's refine already does, `phylotesting.cpp:1621`).
2. **tip/ptnFreq rebuild (`:1442`)** → **persistent per-alignment device buffer for MF**: extend the L0 member
   cache (`_gpuResTip` `:2616`) so the OMP bypass uses a per-thread/per-alignment slot instead of a private
   host rebuild every candidate.
3. **Mutex serialization (`:2607`)** → **de-globalise the `__constant__` eigensystem to global memory**, then
   batch candidates (PHALANX grid.z). Scope to the rate/regime axis it already multiplexes (trial-branch
   batching is NO CLEAN PATH).
4. **Per-candidate eigensystem H2D (`:2622`)** — already minimal; deprioritize.

**D.3 — the correctness anchor (the reason this is hard, not just plumbing).** The self-check catches a
coherent-but-wrong optimum. It must be a **genuine independent recompute**: `optimizeParametersJOLTMix`'s
comment (`phylotreegpu.cpp:2721–2725`) warns a same-path echo is a **tautology** ("recomputes CPU lnL at the
SAME un-optimised params ⇒ they agree"), and GPU overrides are suppressed under `--jolt` so the check is a real
CPU recompute (`setLikelihoodKernelGPU` `:2017–2023`). Viable replacements: (a) **independent device
recompute** (different kernel than the optimizer) + on-device rel gate; (b) the **CPU-optimum comparison gate**
already designed for +R ("assert JOLT lnL ≥ CPU-refined lnL − eps else NaN→CPU", `part9:452/58`). Removing the
check **entirely** is unsafe for MF (a wrong-*low* lnL still mis-selects; BIC only penalises wrong-*high*).

---

## E. 🚦 PROFILING GATE — RESOLVED by nsys (job 173763069, H200, 2026-07-14)

The audit says *where* the host work is; profiling says *how big* — and that picks the direction. Two numbers:

**E.1 nsys device profile — job `173763069` (H200) LANDED; `173761198` (A100) confirms.** DNA 100K + 1M,
`-m MF` no-ctf. `cuda_gpu_kern_sum` vs `cuda_gpu_mem_time_sum` vs `cuda_api_sum`. **Answer to D.4.3 — the
"device" bucket is HALF real kernel, HALF attackable transfer/sync:**
- **1M:** kernel-busy **1119.6s (40%)**; H2D **522s** — of which **86.5% (451.5s) is `setChunk` re-uploading
  the 93.5MB `d_tip`** 57,074× (sqlite CUPTI decomposition), 12% is `d_ptnfreq`/`d_baseinvar`, and only **1.5%
  (7.6s) is the 9.7M `cudaMemcpyToSymbol` coeff storm**. `cudaDeviceSynchronize` 1065s (per-edge sync — but
  overlaps the 1119.6s kernel wait, not additive wall).
- **100K:** kernel-busy **15.6s (21%)**; H2D 4.8s; host/sync **54.5s (72%)**.
- top-3 kernels (`k1_node_t<4>` 46% / `kj_derv_fused` 25% / `kj_pre_t<4>` 22%) = **92.6% of GPU-busy** — the
  same fold/derivative/pre trio as tree-search, so the tree-search kernel levers (tip-vec, cp.async) apply.
- Un-profiled sweep wall pins the fraction: H200 2768s, A100 4614.6s (both ~83–85% coarse "device").

**⇒ `kern (40%) ≪ device (85%)`. The Direction-C win is the `setChunk` redundant re-upload** — ~510s at 1M of
read-only `d_tip`/`d_ptnfreq`/`d_baseinvar`, uploaded every LM sweep though constant across the call. NOT the
toSymbol storm (7.6s, dead). The per-edge `cudaDeviceSynchronize` (1065s) overlaps the kernel wait, so removing
it needs a GPU-resident LM loop (harder, deferred), not the first lever. This is NOT irreducible compute.

**E.2 no-`--ctf` MF `host_resid` sweep — `173760019` (H200) LANDED.** `host_resid = total − Σ(--jolt-diag
device=)` (a valid GPU-vs-non-GPU split in the serial regime, H.2). Measured (H200, DNA `-m MF`, no-ctf):
10K = discard (cold-start); 100K = 27.9s / 69.7s = **40% host**; **1M = 406.7s / 2768s = 15% host / 85%
GPU-optimize** (engage 113, best F81+F+G4). ⇒ **host fraction DROPS with scale (40%→15%): at 1M, no-`--ctf`
MF is GPU-COMPUTE-DOMINATED, not host-bound.**

**E.3 — REFRAME (1M landed) + user correction (2026-07-14): BOTH Direction A and the intra-candidate sync
must go — they are the SAME "remove host↔device sync" philosophy at two levels.** My prior "Direction A caps
at 15%, not the lever" was an over-pivot; corrected here.

At 1M the wall is 85% GPU-optimize / 15% host — but **"GPU-optimize" is NOT pure kernel compute.**
`gpu_jolt_optimize` does per-edge `cudaDeviceSynchronize` (×~197 edges × ~24 LM iters), `reduceDerv` D2H, and
eigensystem H2D **inside** each candidate call (§A.4). So the 85% itself contains **intra-candidate
host↔device synchronisation** — real removable overhead, not all compute. That is exactly the "synchronisation
inside the compute" concern: it is a target, not a floor.

**Two levers, both kept (not either/or):**
- **Direction A — the host self-check (`:2502`). NOT abandoned.** It is **redundant overhead** (the GPU already
  produced the lnL; the CPU recomputes it) and it is **~40% of the wall at 100K** (jobs 173760016/019). Most
  real MF datasets are ≤1M, so 40% is not a rounding error; and its *absolute* cost grows with size even as its
  *fraction* falls (to the 15% band at 1M, which also carries declined-CPU + framework). Replace it with an
  **independent on-device** coherence check (or top-K sample) — keep the correctness guarantee, drop the CPU
  round-trip.
- **Direction C — the redundant intra-candidate H2D re-upload.** `setChunk` (`gpu_lnl_intree.cu:2735`) re-uploads
  the read-only `d_tip`/`d_ptnfreq`/`d_baseinvar` every LM sweep (~505× per candidate). **Guard it resident** —
  the safe, first Phase-1a build (§F). *(The per-edge `cudaDeviceSynchronize` — 1065s — overlaps the kernel wait,
  so it is NOT additive wall; a GPU-resident LM loop to remove it is a later, harder lever, not this one.)*

**nsys RESULT (173763069 + sqlite): `kern (1119.6s, 40%) ≪ device (85%)` — the attackable transfer is the
`setChunk` re-upload, CONFIRMED and precisely sized.** 86.5% of the 522s H2D = **451.5s re-uploading the 93.5MB
`d_tip`** (57,074×; max transfer = `ntax·nptn` exactly), +62s `d_ptnfreq`/`d_baseinvar` = **~510s of redundant
read-only H2D.** The 9.7M `cudaMemcpyToSymbol` is only **7.6s (1.5%) — a dead lever** (and the `kj_derv_fused_args`
async path that would cut it is default-OFF anyway, `gpu_lnl_intree.cu:833`). So "fast MF without `--ctf`" gets its
first, safest win from the `setChunk` resident guard (correctness-neutral), then Direction A (host self-check),
then — only if needed — the harder GPU-resident LM loop for the per-edge sync. Direction B (cross-candidate
mutex) stays deprioritized (serial regime, H.3).

---

## F. Phased plan (bit-identity-gated, default-off — mirrors the tree-search rollout)

- **Phase 0 — profiling (✅ DONE, nsys 173763069 + sqlite):** §E numbers landed. Direction verdict: the attackable
  transfer is the `setChunk` re-upload (~510s at 1M), NOT toSymbol (7.6s). Remaining Phase-0 A/B: the
  `JOLT_MF_NOSELFCHECK` split on a fixed candidate set (the MF analogue of `JOLT_L0=1`) sizes Direction A's slice
  separately from framework/declined-CPU.
- **⭐ Phase 1a — BUILT + GATING (`JOLT_MF_RESIDENT`, correctness-NEUTRAL, binary `5ba5e2f1`; results §J):** intra-call `setChunk` resident guard —
  track the currently-resident chunk index; skip the re-gather + 3× H2D (`gpu_lnl_intree.cu:2738-2741`) when the
  requested chunk is already resident. At nTile==1 (the ≤1M case) this uploads `d_tip`/`d_ptnfreq`/`d_baseinvar`
  **once per candidate** instead of ~505×. **Byte-identical by construction** (the three buffers are written ONLY
  by `setChunk` and are read-only to every kernel — verified: no kernel writes `gbj_tip`/`gbj_ptnfreq`/
  `gbj_baseinvar`). Attacks the measured ~510s with **no new correctness guarantee**. Helps BOTH MF and DNA
  tree-search (same `gpu_jolt_optimize`). Fresh clone (gate 173768871 still building the graduate tree — no
  contamination). **Gate:** byte-identical lnL DNA-1M **and** AA-1M (`-m MF` + fixed-model); partitioned-MF
  `omp_in_parallel` bypass unaffected (`setChunk` is intra-call, per-candidate, so no cross-thread state — but gate
  a partitioned cell anyway, [[feedback-gates-blind-to-partitions]]). **⚠️ proof-of-effect is SCALE-DEPENDENT
  (measured §J.1): ">16MB op-count drops ~505×" only reads at 1M (chunk = 93.5MB > 16MB); at 100K the chunk is
  <16MB so that count is 0 in BOTH arms — the guard's firing shows in H2D BYTE-VOLUME (DNA-100K 81.7GB→1.9GB = 43×,
  op-count near-flat). Below 1M, read bytes not the >16MB count.**
- **Phase 1 — on-device self-check (default-off `JOLT_MF_GPUCHECK`):** replace `:2502` `computeLikelihood()`
  with an independent device recompute (`gpuComputeTreeLnLCleanRoom` mirror). **Gate:** best-by-BIC == CPU
  `-m MF` oracle on DNA-1M **and** AA-1M; per-candidate lnL rel ≤ 1e-9; **fault-injection** must still catch the
  dropped-param failure (`phylotreegpu.cpp:2721`).
- **Phase 2 — persistent per-alignment tip buffer for MF (`JOLT_MF_RESIDENT`):** kill the per-candidate host
  rebuild (D.2). **Gate:** byte-identical lnL, no partitioned-MF regression (the `omp_in_parallel` bypass exists
  *because* of that — the resident slot must be per-thread/per-alignment).
- **Phase 3 (only if §E = Direction B):** de-globalise `__constant__` eigensystem + PHALANX grid.z candidate
  batching, rate/regime axis only. **Gate:** the shelved 3.0× bar, honest NO-GO if it stays ~1.1×.
- **Every phase:** default-off env flag, `-starttree PARS` pinned, DNA **and** AA validated symmetrically, RF=0
  / rel≤5e-16, negatives reported as prominently as wins. Assistant does NOT push GPU source — the author does.

---

## G. Open risks / honest caveats
- The hardest piece is **correctness (D.3)**, not plumbing: MF's lnL is the deliverable, so the self-check
  can't be dropped, only replaced by an *independent* device recompute or a comparison gate. If that recompute
  costs ~the same as the CPU postorder on-device, Direction A's win shrinks — Phase 0's A/B must quantify it.
- Candidate batching (Direction B) is **explicitly shelved as "no clean path"** in the repo; reopening it needs
  the de-globalisation first and must beat a real bar, not a saturated-regime spike.
- The 10K sweep point was a **cold-start artifact** (first-run CUDA init/JIT/clock-ramp inflated GPU device
  time); only warm sizes (100K+) are usable for the host/GPU split.

---

## H. 🔴 Red-team — live-data corrections (2026-07-14, supersedes §A–§G where they conflict)

Grounded in the in-flight no-`--ctf` MF run **job 173760019** (H200, DNA `-m MF`, `--jolt-diag`) + source
re-read. These correct overstatements in my own draft; recorded honestly.

**H.1 — OVERSTATED: the tip[]/ptnFreq[] rebuild is NOT a meaningful host cost (fixes §A.5 #2, §D.2).**
The `--jolt-diag` `H1` timer spans `phylotreegpu.cpp:2250→:2324` — i.e. **all** per-candidate host setup
*including* the `joltGetTipPtnFreq` gather at `:2292`. Measured at 1M: **`H1 = 0.000869 s`** (nptn=935227).
~0.9 ms cannot be an uncached 93.5 MB rebuild, so the gather is either cache-served or intrinsically cheap —
**either way it is negligible.** ⇒ Demote §D.2 (persistent tip buffer) far down the priority list; the tree-
search "cache" analogy does not carry a win here because the cost it removes is already ~0 in MF.

**H.2 — CORRECTED (regime): in the verified SERIAL regime `device=` is clean GPU time, but `host_resid` is
NOT purely the self-check (refines §E.2).** Since single-align MF is serial (H.6), the mutex
(`gpu_lnl_intree.cu:2607`) is **never contended** ⇒ `device=` carries no mutex-wait ⇒ `Σdevice` IS the true
GPU-call time and `host_resid = total − Σdevice` IS a valid GPU-vs-non-GPU split (my earlier "unreliable due
to mutex-wait" was for the wrong regime). **BUT** the non-GPU part is not just the self-check: it also includes
every candidate that **declines entirely to CPU** (`modelfactory.cpp:1606-1616`) running a full CPU optimize, plus
per-candidate framework (`new IQTree`/`initializeModel`/eigendecompose, `phylotesting.cpp:2407/2431`). ⇒ §E must
decompose `host_resid` into {self-check, declined-candidate CPU, framework}. nsys still needed to split `device=`
itself into kernel vs intra-candidate launch/sync (Direction C).
> 🔴 **STALE-CLAIM FIX 2026-07-19: the "(+R/+I/mixtures) decline; ~10 of ~88, engage=78" here was the PRE-M.2 binary
> (`9d1fd49c`).** On the **current** binary (`f3f7875f`, post-M.2 pure-+I/+I+R closures) **+I / +I+G / +I+R and pure +R
> (R2–R10) ALL engage on the GPU by default** — DNA `-m MF` declines = **0** (empirical `174123768`: DNA 85/85, AA 127/127
> engaged; `m4census 174007767`: 0 JOLT declines). So on the shipping binary the `host_resid` decline-candidate bucket is
> ~empty for standard DNA/AA; the residual host cost is self-check (~11% at nt12) + framework. The only *default* CPU-decline
> candidates that remain are `+FO`/tied-freq/AA-free-Q/codon (the `+FO` slice is the `JOLT_FREEQ_FO` fix, gate `174125462`).
> ✅ **`174125462` PASSED**: F81+FO 1.11e-8, HKY+FO 1.10e-8, GTR+FO 1.44e-8, GTR+FO+G4 4.72e-8 (all ≪1e-6, engage=1/no-decline)
> ⇒ **+FO closed on DNA; the default-set CPU-decline list is now empty for standard DNA.** Default-OFF, human graduates.

### 🔬 2026-07-19 — two mechanism findings that change this plan

**(1) `--thread-model` (evaluateAll) TRUNCATES the +R ladder — root-caused to `MF_WAITING`, and it is a CORRECTNESS bug, not a
speed feature.** On avian-1M it selects `GTR+F+I+R2` (lnL −11224813) instead of `GTR+F+R6` (−11205730) = **19,082 nats worse /
+38,070 BIC** (job `174122292[0]`). Chain (agent-pinned, disk-verified): `generate()` `phylotesting.cpp:2141-2146` inserts
`+R3..+Rmax`/`+I+R3..` with flag **`MF_WAITING`**; `getNextModel` `:4242` skips any `MF_WAITING` model; `MF_WAITING` is cleared in
**exactly one place** — `:4453`, inside `#ifdef _IQTREE_MPI` ⇒ in a **non-MPI** build it is never cleared ⇒ `+R≥3` are **never
dispatched, in any family**. `test()` is immune (index-order loop `:3906` skips only `MF_IGNORED`, climbs via `skip_model`
`:4103-4114`). REFUTED alternatives: the `:4638` `getHigherKModel` cascade (cannot fire — `getLowerKModel(+R2)=−1`, and +R
improves monotonically on avian) and `filterRates` (`:4731`, only a secondary cross-family propagation). **NOT a GPU bug** — the
same binary scores `GTR+F+R6` correctly under `--thread-site`, so the GPU +R/+I fixes (`JOLT_IR_FDFIX`, +R determinism) are not
implicated and **cannot** rescue it. ⚠ **OPEN: possible FORK REGRESSION** — the `MF_WAITING` flag is upstream (Minh Bui
`c77497a5e`), but a non-MPI clear may have been lost in our MPI/FCA refactor. **Check against a pristine upstream checkout before
attributing this to upstream IQ-TREE.**

**(2) The `JOLT_MF_DEVUSE` retirement is `-nt 12`-SCOPED and does NOT bind DNA at nt1.** Every arm of the retiring gate
`173837501` ran `-nt 12` (disk-verified). The GPU-mirror self-check cost is **thread-independent**; the CPU postorder it replaces
is ~7-9× larger at nt1 ⇒ the cost/benefit **inverts** exactly where we lose (nt1 DNA-1M MF **2193s** vs Hashara ~**1241s**; the
nt1 profile `mfhost 173929005` is ~**50%** CPU postorder — `computePartialLikelihoodSIMD` 42.84% + `dotProductTriple` 5.68% —
vs **11%** at nt12, `dnares 174003961`). dna-100K was already **+7.5%** even at nt12. **Keep "never ON" for nt12 and for AA at
any nt** (aa-1M was 3.7× slower — a GPU-side, thread-independent cost at ns=20 ⇒ retain an **ns=4/DNA scope guard**), but the
DNA-nt1 cell is **untested**: spike **`174126193`**. Honest ceiling: base/+G alone ≈ −280s → ~1900s (**still loses**); reaching
~1200s needs **+I** (mirror term already validated, `invargate 173818786` rel 5.9e-10) **and +R** (gated on an avian +R
mirror-vs-CPU cross-check). Gate on **selection-invariance**, not byte-identity (the mirror differs ~1e-11).

**H.3 — RETRACTED (my own error): the self-check is NOT hidden — it is serially EXPOSED. Direction A
confirmed.** My prior claim here assumed the parallel `:4547` loop; that is FALSE for the measured regime
(see H.6, source-verified): single-align `-m MF` runs the **serial `test()` loop** (`phylotesting.cpp:3894`,
zero omp pragmas), so each candidate runs `setup → GPU optimize → CPU self-check` **one at a time**, and the
self-check (`:2502`, internally 16-thread site-parallel) sits **on the critical path between GPU calls** —
fully exposed, adding directly to wall. ⇒ **Direction A (reduce/replace the serially-exposed self-check) is
correctly prioritized** (provided it dominates `host_resid` over the declined-candidate CPU cost, H.2), and
**Direction B is UNREACHABLE in this regime** (the mutex is never contended) — deprioritize it unless
explicitly targeting the parallel `evaluateAll` / partition regimes.

**H.4 — the self-check's cost is currently UNMEASURED.** It sits after the GPU call and is captured by neither
`H1` nor `device=`. The §F Phase-0 `JOLT_MF_NOSELFCHECK` A/B (or an added timer around `:2502`) is required
before ANY claim that the self-check is or isn't the lever. Do not assert its magnitude from memory.

**H.5 — coherence vs optimality (sharpens §D.3).** The `:2502`/`:2525` gate checks **coherence** (does a fresh
CPU lnL at the JOLT-written params match the GPU's returned lnL within `rel≤1e-6`) — NOT **optimality** (are
those params the MLE). So "the lnL is the deliverable ⇒ can't drop the check" is right for catching a *broken
kernel*, but the *wrong-optimum / dropped-param* failure is caught by a **different** mechanism (the +R
CPU-optimum comparison gate, `part9:452/58`). The plan must keep these two guarantees distinct; an on-device
replacement must reproduce **both**, and the doc §D.3 currently blurs them.

**H.6 — the THREE regimes (the root of the confusion).** (1) **Serial `test()`** (`phylotesting.cpp:3894`) —
single-alignment `-m MF`, `openmp_by_model=false` default (`tools.cpp:7676`); dispatched at `:1992`. **This is
what ALL the profiling runs** (173760019/016, nsys 173761198/069 — no `--thread-model`, non-MPI build). Cache
active, mutex uncontended, self-check exposed. (2) **Parallel `evaluateAll`** (`:4245`, the `:4547` omp loop)
— only under `--thread-model`/MPI/**the CTF coarse pass** (`:1574`). (3) **`parallel_over_partitions`**
(`:3066`) — partitioned data. The draft §A and the first audit described regime (2); the runs are regime (1).

**H.7 — CORRECTED: the self-check does NOT catch a dropped free parameter (fixes §B, §D.3, §F Phase-1 gate).**
§B claimed it catches "a silently dropped free parameter." The cited comment says the OPPOSITE
(`phylotreegpu.cpp:2721-2725`): with a dropped free-freq the GPU and the CPU recompute agree *because both use
the un-optimised freqs*. The check is **coherence-only** (GPU lnL == CPU lnL at the SAME params, `rel≤1e-6`),
NOT optimality. Dropped params are barred by the **eligibility gate** (`getNDim()==0` / free-Q handling,
`:2719-2725`), never the self-check (prose: `part9:449-451`). ⇒ Phase-1's "fault-injection must catch
dropped-param" gate tests an invariant the self-check architecturally can't provide; an on-device replacement
must reproduce the **coherence** check, while optimality stays with the eligibility gate + offline JOLT≥CPU
validation.

**H.8 — Direction A's replacement must be INDEPENDENT and is not free.** The on-device self-check must be a
*different* kernel than the optimizer (a same-path echo is a tautology, `:2721`). And the +R "CPU-optimum
comparison gate" (`part9:58/450`) offered in §D.3 checks *optimality* via a **CPU refit** — reintroducing the
host cost the offload removes. Honest Direction-A menu: (a) independent on-device coherence recompute, or (b)
**sample** the self-check to top-K (`ctf_topk`-style) accepting a small correctness-coverage trade. Phase-0
must price both.

**Net effect on direction:** source-verified regime restores **Direction A** as the lever (serially-exposed
self-check + declined-candidate CPU cost), with **Direction B deprioritized** (mutex uncontended in serial).
§E's gate now serves two specific measurements, not a fork: (a) the self-check's share of `host_resid` via a
Phase-0 `JOLT_MF_NOSELFCHECK` A/B, and (b) the kernel-vs-launch/sync split of `device=` via nsys
(173761198/069) — the latter decides whether Direction C (fuse per-candidate launches) is also needed.

---

## I. Why this wasn't caught earlier — CTF-hiding = CONFIRMED PRIMARY (source + provenance grounded)

**Mechanism.** `runCTFModelFinder` (`phylotesting.cpp:1465`): subsample to `min(ctf_subsample, nsite)`
(`:1516`), default **`ctf_subsample=5000`** (`tools.cpp:7864`); evaluate **all ~90 candidates on the 5000-site
subsample** (`:1574`); refine only **`ctf_topk`, default 3** (`tools.cpp:7865`) on **full data** (`:1621`).
⇒ the O(nptn) per-candidate host round-trip (the `:2502` self-check at full nptn + any declined-candidate
full-CPU optimize) is paid **~3× under CTF, ~90× without.** The no-`--ctf` full-MF path is the FIRST time the
×90 full-data cost has run at scale — and it exists only because of Minh's no-`--ctf` directive this session.

**Provenance — every headline MF number is a CTF run:** AA-1M CTF **893s** (`00-MASTER:93`, jobs
170517590/170581208); DNA-1M `-m MF` CTF **152s = 7.4–13×** (`00-MASTER:101`, job 170843136); AA-1M `-m MF`
CTF **767s** (`00-MASTER:173`). No no-`--ctf` 1M MF benchmark existed before this session.

**Secondary reasons (ranked):**
1. **Config compounding.** No-`--ctf` single-alignment also loses cross-model parallelism — it routes through
   **serial `test()`** (`openmp_by_model=false`, H.6), while CTF-coarse uses **parallel `evaluateAll`**. So the
   no-ctf runs pay the full-data cost ×90 *and* serially. (`--thread-model` parallelizes it but re-enters the
   mutex-contention/cache-bypass regime.)
2. **Known-but-deferred, mis-scoped.** `part9:472` named the host self-check as "the remaining WALL-TIME lever
   … capability comes first" — but for the **10M `-te` single-tree eval**, not the ~90-candidate MF loop. The
   tracked metric was **engage-% coverage** (job 170602983), never per-candidate wall.
3. **Tree-search fix targeted the wrong MF bottleneck.** `JOLT_SCREEN_CACHE` (2.089×, `a07f61be`) killed the
   tip rebuild in serial tree-search; on MF the tip rebuild is negligible (H1=0.9ms) — the MF host cost is the
   self-check × candidate-count, which the cache doesn't touch. (§C's "cache doesn't fire on MF" is itself
   half-wrong — it DOES fire on serial no-ctf MF, default-on, and hits; it just doesn't help.)

**Verdict:** CTF-hiding is **PRIMARY and confirmed** — by mechanism (k=3 vs ~90, `:1465/1574/1621`) and
provenance (all headline MF numbers are CTF). The per-candidate full-data host cost was invisible because the
only path that pays it ~90× had never been run at scale until now.

---

## J. Overnight profiling batch — RESULTS LANDING (2026-07-14)

Five live jobs on the guard binary **`5ba5e2f1`** (`JOLT_MF_RESIDENT` sentinel verified; cloned from graduate
`a07f61be`). **All shipped tree-search levers are already compiled INTO this binary** (sentinel grep:
`JOLT_SCREEN_CACHE`/`JOLT_L0`/`TS-FUSED`/`_gpuResTip` all present; +R/+I GPU-reopt confirmed live — +I+G
candidates show `JOLT-DIAG-HOST device=0.5–1.4s`, i.e. optimizing ON GPU, not CPU-declining). **⇒ the OFF-arm
baselines below are NOT un-optimized — they already carry every tree-search win. This isolates the NEW MF-specific
lever (the `setChunk` guard) on top of a fully-optimized binary.** Guard A/B = env `JOLT_MF_RESIDENT` unset (OFF)
vs `=1` (ON).

**⚠️ Why the tree-search optimizations do NOT already make MF fast (grounded):** the biggest one —
`JOLT_SCREEN_CACHE` (2.089× on tree search) — **fires on serial no-`--ctf` MF (default-on) but removes ~0**,
because the cost it kills (the host tip gather, `H1`) is **0.9 ms–7 ms per call in MF** (measured live), not the
0.69–1.09 s it was in tree search (§H.1, §I.3). `L0` doesn't fire at all (no tree search in `-m MF`). What DOES
carry: the shared `k1_node`/`kj_derv`/`kj_pre` kernels (so kernel levers like tip-vec would help both), and the
+R/+I GPU reopt (already keeping ~declined candidates on-GPU). **Net: MF needs its OWN two levers — Phase 1a
(`setChunk` guard) + Direction A (self-check) — there is no free ride from the tree-search work.**

### J.1 — mfbtl (job `173785379`, A100): MF bottleneck map, OFF vs ON, `-m MF` no-`--ctf`
| cell | wall OFF→ON | bit-id (best + table md5) | H2D **bytes** OFF→ON | GPU idle OFF→ON | kernel-busy |
|---|---|---|---|---|---|
| **DNA-100K** | **108.65→97.76s (1.11×)** | ✅ `F81+F+G4`, md5 `26fba2bb` (98 rows) | **81.7GB→1.9GB (43×)** | 75%→72% | 27.2s |
| **AA-100K** | 600.28→596.88s (1.006×) | ON landed (bit-id pending export) | 70.6GB OFF | 50% OFF | **298.6s** |
| DNA-1M ON | guard-ON nsys running (next-bottleneck profile) | — | — | (measures guard-ON idle) | — |
| AA-1M ON | best-effort, last | — | — | — | — |

**★ DNA-1M `-m MF` HEADLINE (job `173781194`, clean wall, no nsys): OFF 2571.5s → ON 1610.5s = 1.597×,
BIT-IDENTICAL** (best `F81+F+G4`, table md5 `c55bad83`, 142 rows, both arms). This is the Phase-1a win at the
scale where the chunk is 93.5MB > 16MB — the 100K→1M scaling held (1.11×→1.60×) exactly as predicted. 961s of
wall removed, zero correctness cost. **NB — AA barely moves (100K 1.006×)** vs DNA (1.60×): AA is compute-bound
(50% idle, busy 298.6s), so the transfer the guard removes is a tiny share; DNA is host/latency-bound so the
re-upload dominated. The guard is a **DNA lever**, mirroring [[project-reopt-rearchitecture]]'s cache (also DNA).

- **Proof-of-effect = BYTE-VOLUME, not the >16MB op-count** (see §F correction): at 100K the chunk is <16MB so the
  count is 0 in BOTH arms; the guard shows as **81.7GB→1.9GB (43×)** with op-count near-flat (1.194M→1.172M).
- **The residual idle is HOST, not transfer.** Even guard-ON, DNA is **72% GPU-idle**; the guard removed only the
  ~10.7s of transfer-idle (matching the 10.9s wall delta). **⇒ front-runs Direction A** — the dominant idle at
  100K is the host self-check / declined-CPU, not the transfer the guard removes.
- **AA vs DNA axis is clean and expected.** AA-100K: **50% idle, busy 298.6s** (compute-bound, ns=20 → ~25× the
  matvec). DNA-100K: **75% idle, busy 27.2s** (host/latency-bound). The guard saves LESS on AA (C3 = 1.02×, §J.2)
  precisely because AA has less transfer-idle to remove — a real, honest DNA≠AA result, not a defect.

### J.2 — Phase-1a same-binary gate (job `173777619`): guard OFF vs ON, `-B`/MFP + partition
| cell | wall OFF→ON | BEST SCORE (OFF == ON) | bit-id |
|---|---|---|---|
| C1 DNA-1M `-B` (tree-search) | 1341.3→1271.6s (1.055×) | `-59208015.973831` | ✅ identical |
| C2 DNA-100K MFP | 186.4→175.4s (1.063×) | `-5692984.526136` | ✅ identical |
| C3 AA-100K MFP | 690.8→677.9s (1.019×) | `-7541976.852167` | ✅ identical |
| C4 partitioned | running | — | — |

C1 is `-B` tree-search (not `-m MF`) — the guard is a smaller share there, hence flat-ish; the pure-`-m MF`
headline is J.3. All landed cells **BEST SCORE bit-identical** — correctness holds on DNA, AA, and (pending) partition.

### J.3 — mfprof (job `173781194`): DNA-1M `-m MF` OFF vs ON — **the Phase-1a headline** (PENDING)
- **OFF: 2571.5s** (best `F81+F+G4`), DONE. **ON: running** (model 840, per-call `H1`≈0.007s = guard hitting).
- This is the regime where the chunk is **93.5MB > 16MB** and the 451.5s re-upload lives — the real test of how
  much transfer-volume converts to wall. ⚠️ its in-job bit-id has the empty-md5 false-pass; **re-derive table md5
  from `.iqtree` on landing** (best-fit-model check saves correctness meanwhile).

### J.4 — mfhostattr (job `173785380`, A100): Direction-A go/no-go (PENDING, queued)
- perf host-attribution, DNA+AA-100K, guard-ON: **is `computePartialLikelihoodSIMD` (the `:2502` self-check
  postorder) the dominant host cost?** J.1's 72%/50% idle says the residual IS host — this names the function.
  **Self-check dominates → build Direction A. Declined-CPU/framework dominates → Direction A is smaller than hoped.**

### J.5 — reconciliation (HONEST — resolves the §E "451.5s" framing)
The **451.5s** `setChunk` figure (§E) is a **transfer-VOLUME** removed, **not a wall saving**. At 100K it converts
to only **~10.7s of wall** (108.65→97.76s) because the H2D **overlaps host compute** — 43× fewer bytes but 1.11×
wall. **Do NOT quote 451.5s as a speedup.** The DNA-1M ON number (J.3) measures the real wall conversion at the
93.5MB-chunk scale; the honest expectation is "bounded by host-overlap," measured per-scale, not the volume ratio.
This corrects the plan's implicit "~510s attackable ⇒ ~510s faster" reading: **attackable-volume ≠ wall-win.**

### J.6 — competitive reality: can no-`--ctf` MF beat Hashara? (grounded, honest, 2026-07-14)
**Corrects a memory error:** the "her MF = 2374.9s CPU" figure was the STALE `cudajolt` column (OUR June binary
in CPU-MF mode at nt12), NOT Hashara. Her real port **offloads the WHOLE likelihood incl. MF to the GPU, runs MF
on 1 CPU thread**, and she runs it **without `--ctf`** (plain `-ninit 2 -optalg 2-BFGS` = default MFP)
(`Hashara/2026_06_24_simulated_results/ANALYSIS_cudajolt_column_is_our_stale_binary.md:36-43`).

**Her recorded full-MFP bar (`openacc_jolt`, ratio vs CPU nt104; denominators from that doc §3):**
| cell | CPU nt104 (MFP) | her openacc_jolt | back-computed wall |
|---|---|---|---|
| DNA-1M | 3769s | **1.70×** | **~2217s** |
| AA-1M | 17610s | **2.02×** | **~8718s** |

**Our no-`--ctf` side (grounded):** DNA-1M `-m MF` **model phase alone** = **1610.5s** guard-ON (this job) — i.e.
**73% of her ENTIRE MFP (2217s) is spent by us on the model phase before any tree search.** ⇒ **on DNA we are
currently BEHIND her without `--ctf`.** Her architecture sidesteps exactly our bottleneck: no per-candidate CPU
self-check (our 72% GPU-idle), because every eval is GPU-resident. Direction A attacks that self-check directly,
but even a large Direction-A win must ALSO clear the separate DNA tree-search gap (ours ~1288s vs her 894s,
[[project-fullgpu-endtoend]]) to beat her total. **Honest: beating her on DNA no-`--ctf` is NOT guaranteed by
Phase-1a + Direction A alone.**

**AA is the opposite** — we already beat her OpenACC on AA tree-search (**1.123× ahead**, job 173059374), AA MF is
compute-bound (near-full-GPU, small host residual), so once MF is optimized we should clear her 2.02×. **AA-1M
`-m MF` head-to-head is UNMEASURED on our side (pending `a1m_on`)** — do not claim the AA MF win until it lands.

**Strategic truth:** `--ctf` was our STRUCTURAL answer to the MF cost (968 models → 3 full-data fits = 35.5×).
Remove it and we fight on per-eval GPU throughput — Hashara's whole-likelihood-resident design's home turf. Our
raw engine is competitive (14–32× vs CPU on likelihood work, doc §4), but our per-candidate-offload + self-check
orchestration carries overhead her resident design does not. **The deeper no-`--ctf` lever may be architectural
(make the MF likelihood GPU-resident across the candidate loop, like hers), not just Phase-1a + Direction A.**

### J.7 — synchronisation decomposition + the "is it compute or stall?" check (landed + in-flight)
**Landed (`mfbtl` `cuda_api_sum`, host-side CUDA API time — the SYNC map):**
| API | DNA-100K OFF | DNA-100K ON | AA-100K OFF | what it is |
|---|---|---|---|---|
| `cudaDeviceSynchronize` | 44.6% / 28.2s / 869K calls | 49.8% / 28.2s | **84.0% / 290.2s / 1.19M** | per-edge sync (CPU blocks on GPU) |
| `cudaMemcpy` | 21.5% / 13.6s | **12.0% / 6.8s** | 4.7% / 16.1s | transfer — **guard halved it** |
| `cudaMemcpyToSymbol` | 20.6% / 13.0s / 1.16M | 23.5% / 13.3s | 8.6% / 29.6s | eigen coeff storm |
| `cudaLaunchKernel` | 12.7% / 8.0s / 1.7M | 14.2% / 8.0s | 2.6% / 8.9s | launch overhead |

**Reading 1 — the guard cut ONLY `cudaMemcpy` (13.6→6.8s); sync/toSymbol/launch untouched** — exactly as designed
(it removes redundant H2D, nothing else).

**Reading 2 — `cudaDeviceSynchronize` ≈ kernel-busy** (DNA 28.2s sync vs 27.2s busy; AA 290.2s vs 298.6s busy) ⇒
it is overwhelmingly **the CPU blocking on GPU compute, not removable overhead.** Removing it needs the GPU-resident
loop (§K), and even then it doesn't speed the kernel — it frees the host.

**⚠️ Reading 3 — "kernel-busy = compute" is FALSE. ncu SETTLED IT (job `173790738`, H200, LANDED 2026-07-14).**
`cudaDeviceSynchronize ≈ kernel-busy` proves the CPU waits for the GPU, but that wait is **NOT** irreducible FP64
compute — it is memory-stall. ncu on the actual shipped MF kernels (fixed-model brlen reopt = same kernels as `-m MF`):
| kernel | Compute (SM)% | FP64 peak | binding resource | achieved occupancy |
|---|---|---|---|---|
| DNA `k1_node_t<4>` | **19.95%** | **11%** | L1/TEX 45.8% (+DRAM 18.5) | 34.2% of 62.5% theo |
| AA `k1_node_t<20>` | **18.53%** | **5%** | **DRAM 56.6% / L2 71.1%** | 34.2% of 100% theo |

Both are **~19% SM util, single-digit FP64, memory-subsystem-bound** (ncu verbatim: *"All compute pipelines are
under-utilized"*). The binding resource matches tree-search's DNA-vs-AA split exactly: **DNA<4> = L1/TEX
(shared-mem) latency** (tree-search: `short_scoreboard`, `FULL-GPU-END-TO-END-PLAN.md:222`), **AA<20> = DRAM/L2
(global) latency** (tree-search: `long_scoreboard`, `:1056`). So the GPU-busy time is memory-latency STALL, not
compute — confirmed on both types. **My Reading-2 "CPU blocks on real compute" is corrected to "CPU blocks on a
kernel that is ~80% memory-stalling."**

**🔴 BUT — this is the SAME bottleneck tree-search already mapped, and it corrects TWO overclaims I just made
([[project-gpu-tree-search]], `FULL-GPU-END-TO-END-PLAN.md:187-240`):**
- **The kernel is NOT the lever — it is near its practical ceiling.** tree-search's production ncu (job 173523960)
  found DNA<4> at **57–62% occupancy, memory-latency-bound, no bandwidth slack ⇒ "can't be made much faster."** My
  MF "occupancy-starved at 34%" was a **100K-probe tail artifact** (theoretical 62.5% matches tree-search's full-grid
  achieved) — occupancy-raising is DEAD.
- **tip-vec is NOT ~1.1–1.4× — it is ~1.075× (a marginal cleanup), and cp.async/factored-fold/shared-staging/
  K-batching are all DEAD** on these kernels (factored-fold cut 97% of loads and got SLOWER on every arch, jobs
  173470989/72). I overstated the device lever.
- **The REAL bottleneck in BOTH workloads is the per-edge host↔device sync / launch orchestration around kernels
  too CHEAP (at ns=4) to amortize it** — tree-search measured ~700K launches / 320K syncs / 160K D2H per search;
  MF's `cuda_api_sum` (§J.7) is the SAME shape: **869K `cudaDeviceSynchronize` + 1.7M `cudaLaunchKernel` + 1.16M
  `cudaMemcpyToSymbol`** at DNA-100K. **⇒ the device-side MF lever is NOT a kernel rewrite; it is killing the
  per-edge sync/launch storm** (launch-batching / level-synchronous traversal / CUDA-Graph / GPU-resident LM loop) —
  the SAME lever tree-search identified and already **CUT once as a GRAVEYARD-REATTEMPT + hidden race** (`:240`,
  `:207`). Hard, and the honest reason the "device is also a lever" hope is smaller than it looked.

**Net (synthesis, both docs agree):** the k1_node/kj_derv/kj_pre kernels are memory-latency-bound + near-ceiling in
MF *and* tree-search; kernel/occupancy/bandwidth levers are dead-or-marginal in both. The genuine levers are (1) the
**host** (MF: coverage of CPU-declining families §K.3 + the self-check; tree-search: the CPU postorder/L0) and (2)
the **per-edge sync/launch orchestration** (the resident loop) — NOT the kernel. *(Per-stall-reason breakdown needs
`--set full`; `--set detailed` gave SOL+occupancy, which already settles compute-vs-stall and matches tree-search.)*

**⚠️ Correction to §E's "toSymbol = dead lever (7.6s)":** that 7.6s was **device transfer bytes**. The **host-side
API overhead** of 1.16M tiny `cudaMemcpyToSymbol` calls is **13.0s = 20.6% of DNA-100K API time** — additive on the
serial critical path. Dead in bytes, **alive in call-count.** Batching the per-FD-step coeff uploads is a real
(if secondary) lever.

**Preliminary offload-target ranking (to be confirmed by `mfhostattr` 173785380 @100K + `mfoffload` 173790739
@1M):** (1) **host postorder / self-check** (the 72–74% GPU-idle host compute — biggest, = Direction A); (2)
`cudaMemcpyToSymbol` call-count (~13s host, batch it); (3) `cudaLaunchKernel` (~8s, CUDA-graph the LM loop); (4)
device-side kernel stall IF `ncu` confirms latency-bound (tip-vec). Sync-proper (`cudaDeviceSynchronize` beyond
kernel-busy) is small (~1s at 100K) — the resident loop's value is freeing the host, not cutting sync wall.

---

## K. Phase-1b — GPU-resident MF loop (designed + adversarially red-teamed, 2026-07-14)

Designed by a source-grounded Plan pass, then INDEPENDENTLY red-teamed against the graduate source (all 7
load-bearing claims verified to file:line). Both passes recorded honestly. **Verdict: SPIKE only Direction A;
cross-candidate residency is NOT a near-term perf lever.**

**K.1 — what Phase-1b actually is (Plan verdict): an INTEGRATION, not new physics.** It decomposes into
(a) Phase-1a's `setChunk` guard **generalized from intra-call to cross-candidate**, (b) **Direction A** (on-device
self-check replacing the CPU `computeLikelihood()` at `phylotreegpu.cpp:2502` with `gpuComputeTreeLnLCleanRoom`,
`:76`), and (c) the `DevBuf` arena — which **already persists** across candidates (`gbj_*` file-static
`gpu_lnl_intree.cu:2564`; `devbuf_ensure` grows-only `:805-810`), so hoisting it is a **no-op**. All 7 source facts
CONFIRMED: topology IS fixed across `-m MF` candidates (built once `phylotesting.cpp:777→:857`, each candidate
`restoreCheckpoint` `:2429` → fixed-tree branch `:2542`; `-mtree` is the excluded branch); `d_tip` is taxon-ID
indexed (`:2265`) ⇒ DFS/brlen-invariant; `gbj_tip`/`gbj_ptnfreq` written ONLY by `setChunk`, kernels read `const`;
`d_baseinvar` is model-dependent (per-+I candidate, `:2301-2315`), so the resident set is just `{d_tip, d_ptnfreq}`.

**K.2 — 🔴 RED-TEAM #1 (HIGH, reframes the spike): cross-candidate residency has < 2s of headroom — Phase-1a
already captured it.** The 451.5s was 57,074 `d_tip` re-uploads = **~505×/candidate**; Phase-1a already cuts that
to **1/candidate** at nTile==1 (that IS the landed 1.597×, §J.3). Per-upload = 451.5s/57,074 = **7.9ms**. Keeping
`d_tip` resident ACROSS candidates removes only the residual ~1×(engaged + extra +I+G starts) ≈ 150–250 uploads
≈ **1–2s at DNA-1M = under 0.15% of the 1610s ON wall.** `d_baseinvar` is model-dependent (ineligible), `d_ptnfreq`
~0.15s. **⇒ Cross-candidate residency is a correctness-neutral building-block + safety proof for a FUTURE
whole-loop resident redesign — NOT a standalone perf lever. Do not present it as one.** The genuine lever is
Direction A alone.

**K.3 — 🔴🔴 THE GOVERNING CONSTRAINT (user, 2026-07-14) — COVERAGE, not count. This supersedes the red-team's
"minority by count" framing, which used the WRONG metric.** In a full `-m MF`/`-m MFP` run **ALL** candidates are
forced to run (DNA ~142, AA up to 1232). The wall is governed by the **SLOWEST** candidate — so **ANY** family that
declines to CPU sets a wall FLOOR, no matter how rare it is or whether it is selected. Judging a family by its
*count* share (or "the best model won't be +I+G") is the **exact tree-search mistake** that "the best tree won't
need +R10" was — until the real avian dataset selected +R10 freerate and broke it. **Do not repeat it.**
- **GROUNDED (landed DNA-1M `-m MF` ON): 113 of 142 candidates emitted GPU-engage diagnostics; ~29 did NOT** — the
  `+I`/`+I+R` families (`JC+I`, `JC+I+R2..R5` in the table) that decline JOLT **entirely to CPU** (`ctfIneligible`,
  `phylotesting.cpp:1337-1339`; gate `phylotreegpu.cpp:2111`). Those run a **full CPU optimize at nptn=935K**, not
  merely a CPU self-check — the most expensive per-candidate cost there is.
- **Two distinct coverage holes, both wall-setting in a full run:** (1) pure `+I`/`+I+R` **decline JOLT outright**
  → full CPU optimize (Phase-1a + Direction A do NOTHING for these); (2) `+I+G` **engages JOLT but is Direction-A
  ineligible** (`gpuComputeTreeLnLCleanRoom` declines `pinv>0`, CONFIRMED `:84`) → stays on the CPU self-check, and
  is ~4× costlier each (`n_pinv_starts=4` under `--jolt`) and often the selected winner.
- **⇒ The lever for fast no-`--ctf` MFP is COVERAGE — get EVERY family (base/+I/+G/+I+G/+R/+I+R/mixtures/free-freq/
  codon) to optimize AND self-check on GPU with NO CPU decline. Phase-1a (transfer) and Direction A (self-check on
  already-engaged models) are necessary but do NOT close coverage.** The `+I`/+I+R JOLT decline and the `+I+G`
  self-check decline are now FIRST-CLASS blockers, not footnotes. `mfoffload` (173790739, DNA-1M perf) measures the
  declined-CPU wall share directly — that number, not the engaged-model speedup, is the real no-`--ctf` MFP governor.

**K.4 — 🔴 RED-TEAM #3 (HIGH, correctness): the on-device self-check is STRICTLY WEAKER than today's CPU check.**
Claim-4 CONFIRMED genuinely independent (host `exp()` echild `:134-143`, own `g_Uinv` upload `gpu_lnl_intree.cu:1252`,
host Kahan `:1294-1296`; shares ONLY the `k1_node` fold) — NOT a tautology. **But** today's `:2502` is a full CPU
postorder = independent silicon AND independent kernel; swapping to the GPU mirror **trades that away** (shares
`k1_node` + eigendecomp, same GPU). Acceptable ONLY if fault-injection exercises the surface `k1_node` still covers
independently (optimizer echild/descriptor/reduction — it can). Also: `gpu_lnl_crosscheck` takes **no mutex**
(`:1241-1300`) and writes `__constant__ g_Uinv` ⇒ race-safe ONLY in the serial `test()` regime (H.6) — silently
blocks Direction A from parallel/CTF-coarse/partition regimes unless the lock is added. (STAGE-2b `Σfreq·patlh`
identity `:2438` = genuine tautology, `-B`-only, correctly NOT the MF check.)

**K.5 — 🔴 RED-TEAM #4 (MEDIUM, correctness trap): the process-global residency key is an ABA/collision hazard.**
`_gpuResTip`/`_gpuResPtnFreq` are per-tree instance members (`phylotree.h:2137-2140`), destroyed per candidate
(CONFIRMED) ⇒ the process-global `gbj_` pool is the right home. **But** keying on `nptn/ntax/ns` alone lets two
same-dimension alignments collide → wrong resident `d_tip` → silently mis-selected model; raw aln-pointer has ABA.
Bites hardest in **partitioned MF** ([[feedback-gates-blind-to-partitions]]). Needs a **content-addressed
signature** — this validation, not the ABI plumbing, is the real cost the "~1 day" estimate under-prices.

**K.6 — the "necessary but NOT sufficient for DNA no-`--ctf`" conclusion: red-team CONFIRMS it, appropriately
hedged.** The `H1` timer (`:2250→:2324`) EXCLUDES eigendecompose + `initializeModel` (those run at
`phylotesting.cpp:2431`; `new IQTree`/`initializeAllPartialLh` `:2407/:2548` are outside both `H1` and `device=`)
⇒ per-candidate framework cost is genuinely **UNMEASURED** (that's what `mfhostattr` 173785380 / `mfoffload`
173790739 measure). Phase-1b touches NONE of it. With DNA-1M model-phase-alone (1610.5s) already ≈73% of Hashara's
ENTIRE 2217s MFP (§J.6), "very likely not sufficient" is correct — it could only flip if the pending jobs show
framework cost is small.

**K.7 — SPIKE PLAN (re-scoped per red-team AND the coverage constraint K.3).** ⭐ **Priority order is now:
(0) COVERAGE — eliminate the CPU-declining families (`+I`/`+I+R` JOLT-decline; `+I+G` self-check-decline) because
in a full MFP they set the wall floor (K.3); (1) Direction A on the already-engaged models; residency = a separate
correctness-only safety proof, NOT a perf claim. Direction A alone, with the decline families still on CPU, will
NOT make no-`--ctf` MFP fast — that was the tree-search "won't-be-selected" error.**
- **Step 1 (kill-or-continue, cheapest):** `JOLT_MF_DEVCHECK=1` at `:2502` runs BOTH `computeLikelihood()` (keep its
  value) AND `gpuComputeTreeLnLCleanRoom`, printing `wall_cpu`, `wall_gpu`, `rel` (mirrors the `JOLT_L0=1` run-both
  A/B). DNA-100K + AA-100K. **Kill criterion (sharpened from red-team): collapse iff `(GPU-postorder + D2H(nptn) +
  host-Kahan) ≥ CPU-postorder`** — not "≈"; the GPU mirror's host residue is only O(nptn), so if GPU already beats
  CPU on the fold it should be net-positive. Report honestly.
- **Step 2 (bit-id + fault-injection):** best-by-BIC == CPU `-m MF` oracle DNA-1M AND AA-1M; per-candidate rel≤1e-6
  reproduced; **fault-injection MUST catch a perturbed JOLT brlen/Q (LM/write-back class); document the `k1_node`
  shared-kernel blind spot and cover it with a sampled CPU self-check on top-K-by-BIC (K≈3, the CTF-refine
  pattern).** Note the `+I+G` winner stays on CPU self-check regardless (K.3).
- **Step 3 (residency, SEPARATE, correctness-only):** cross-call guard behind its own flag; gate = byte-identical
  DNA-1M+AA-1M + a **partitioned cell** with a content-addressed signature (K.5). Claim ONLY "safe + neutral," NOT a
  speedup (<2s, K.2).
- Default-off env flags; `-starttree PARS`; DNA AND AA symmetric; assistant does NOT push (author does).

**K.8 — unverifiable flags (red-team honesty, do not treat as source invariants):** the "~88 candidates / engage 78"
split is job-specific (source shows 6 rate cats × 22 models, doesn't cleanly = 88); the `modelfactory.cpp:1523`
extra-`computeLikelihood` citation is imprecise (real calls ~`:1396/:1409` in `optimizeParametersGammaInvar`) —
mechanism confirmed, line number off.

---

## K.9 — Direction-A IMPLEMENTED (`JOLT_MF_DEVCHECK` run-both) + red-teamed (2026-07-14)

Implemented in fresh clone `iqtree3-mfdevcheck` (`phylotreegpu.cpp:2502`): default-off, when `JOLT_MF_DEVCHECK=1`
runs BOTH `computeLikelihood()` (CPU, authoritative — returned) AND `gpuComputeTreeLnLCleanRoom(nullptr)` (the
independent device mirror), times both, prints `[MF-DEVCHECK]`. Off-mode byte-identical (impl agent + red-team
confirmed). Measures: is the on-device self-check cheaper than the CPU postorder = Direction-A go/no-go. Job
`gems_mf_devcheck.sh` (173797889 H200): G0 build (fail-fast) + G1 OFF==guard + G2 ON==OFF (mirror side-effect-free)
+ MEASURE (CPU-vs-mirror wall, decline%, rel_gpu) DNA/AA-100K + DNA-1M. **-nt 12** (see regime note).

**🔴 Red-team of the implementation — one HIGH finding, RESOLVED by regime verification (and it repeats the §H.6
trap):** the red-team flagged that `gpu_lnl_crosscheck` (the mirror's launcher, `gpu_lnl_intree.cu:1249`) takes
**NO `jolt_gpu_mtx` lock** (unlike `gpu_jolt_optimize` `:2613`) and writes the process-global `__constant__ g_Uinv`
+ `cudaFree`/`cudaMalloc`s the `static gb_*` pool — so at `-nt 12` it would RACE concurrent candidates, and G2
(best-model+md5) would NOT catch it (the pre-existing rel-gate CPU fallback masks the corruption). **BUT this
assumes the PARALLEL `evaluateAll` regime — plain `-m MF` (non-MPI, no `--thread-model`) takes the SERIAL `test()`
branch** (`phylotesting.cpp:1992` else-branch; `openmp_by_model=false` default `tools.cpp:7663`, `=true` only inside
`#ifdef _IQTREE_MPI`). **VERIFIED empirically: the real `-nt 12 -m MF` run completes models in strictly monotonic
order** (mfprof log: 818,822,839,840,844…) — impossible under a parallel outer loop. ⇒ candidates run one-at-a-time
on the GPU, the mirror has NO concurrent GPU work, **no race**, and `wall_cpu` (12-way inner site-parallel CPU
postorder = the real deployment self-check cost) vs `wall_gpu` (1-GPU mirror) is the **fair deployment comparison**.
The red-team fell into the exact regime-confusion §H.6 warns about — a useful confirmation the trap is easy.
- **⚠️ VALID productionization requirement (keep):** if Direction A is ever productionized and run under
  `--thread-model`/MPI (the parallel `evaluateAll`), `gpu_lnl_crosscheck` MUST take `lock_guard(jolt_gpu_mtx)` —
  the mirror is race-unsafe there. Not a blocker for the serial-regime measurement.
- **Other red-team findings applied:** report the per-candidate MEDIAN (not sum-ratio) as the robust headline
  (first-candidate CUDA-init inflation); build fail-fast now gates on the make exit code + a clean `rm -rf` of the
  build dir (was: binary-presence only → stale-binary footgun). Off-mode byte-inertness confirmed. Only `pinv>0`
  (+I family) declines the mirror on a standard DNA/AA `-m MF` set (the coverage lever, reported separately).
- **Build recipe bug fixed (not the patch):** first submit `173796831` failed cmake configure ("Eigen3 not found" —
  `module load` doesn't set `FindEigen3`/`Boost` dirs on Gadi); fixed with explicit `-DEIGEN3_INCLUDE_DIR`/`-DBoost_*`
  from the working `build-mfresident` cache; resubmitted `173797889`.

---

## L. Live gate/profile results + self-corrections (2026-07-14, A100 unless noted)

**L.1 — DNA-1M `-m MF` guard-ON, nsys "next-bottleneck" (mfbtl `d1m_on`, A100):** GPU **31% idle / 69% busy**
(vs the guard-OFF A100 baseline job 173761198 = **48% idle** — same arch, same workload ⇒ valid A/B: the guard cut
inter-kernel idle 48%→31%). Guard fired (>16MB H2D **113 ops** / 10.6GB, vs OFF 53,169). ⚠️ *(This 31%/69% is a
GPU-timeline metric at 1M; the earlier "31% = self-check + declines / 69% = memory-stall" was an INFERENCE — now
partly MEASURED at 100K in §L.9, but do NOT cross-apply the 100K host split to the 1M GPU-timeline: mfdevcheck L.5
shows the self-check is only ~6% of the 1M WALL.)*

**L.2 — 🔴🔧 AA-1M `-m MF` OOM ROOT-CAUSED (mfbtl `a1m_on`, A100-80GB) — NOT a memory-capacity limit; a SPURIOUS
one-line +R tiling guard. We already have device pattern tiling; it's DISABLED for +R.** `devbuf_ensure failed
(74.2GB)` (`gpu_lnl_intree.cu:2721`, the `gbj_partial` postorder arena). The crash was on **`LG+I+R4`/`+I+R5`** —
**free-rate (+R) models.** Root cause = **`:2691 if (freeRate) nTile = 1;`** — it forces `nTile=1` for EVERY +R model
(overriding `JOLT_NTILE`), so the full-nptn 74.2GB arena is allocated. Comment blames "RGRADCHECK uses full-nptn
buffers" — but **RGRADCHECK is an OPTIONAL diagnostic** (`JOLT_RGRADCHECK`-gated `:2855`). The **production +R LM is
already chunk-safe**: the lnL loop (`:2919 for t<nTile`) AND the gradient loop (`:2981 for t<nTile`, `postorderFill`
reruns per chunk, `accW`→`WNc` Kahan across chunks `:2979/:3097`) tile correctly, "rel≤1e-12 vs one-shot." ⇒ **FIX =
one line: `if (freeRate && getenv("JOLT_RGRADCHECK")) nTile = 1;`** → +R tiles via auto-pick (`:2676`), AA-1M +R fits
A100-80GB at nTile≈2-3, **no H200/141GB needed.** Non-+R AA-1M (base/+G/+I+G) ALREADY tiles fine. Gate: AA-100K +R
byte-identical `JOLT_NTILE=1` vs `=2` (rel≤1e-12), then AA-1M +R completes + matches CPU oracle. This is also the
**AA-1M-pattern-tiling** forward item — now a scoped one-liner, not a redesign. *(My earlier "needs a larger-memory
GPU / H200 fits at 141GB" was BRUTE-FORCE-WRONG — the device tiling we built for AA-10M/MEOW80 already solves it; it
was just switched off for +R. Also wrong on arch
AND wrongly implied causation — corrected.)*

**L.3 — Phase-1a gate partition cell C4d: guard is PARTITION-SAFE; the OFF/ON drift is benign (agent audit,
source-cited).** C1/C2/C3 (single-alignment) were EXACTLY bit-identical; C4d (partitioned MFP) drifted
OFF `-564212.4437` vs ON `-564212.4433` = **rel 7.1e-10**. Verdict: **NOT a guard bug.** (a) `loadedChunk` is a
**per-call local** reset to −1 every `gpu_jolt_optimize` call (`gpu_lnl_intree.cu:2749`) ⇒ cannot leak across
partitions; each partition re-uploads its own `d_tip` before any kernel reads it (mutex-serialized `:2613`). (b) A
stale chunk would give a **gross** error (~1e-2/NaN → CPU-fallback), never 7e-10. (c) The guard is **byte-identical
within every call** ⇒ it cannot inject a numerical difference *at any magnitude*. The 7e-10 is pre-existing
**OpenMP dynamic `reduction(+:)` order non-determinism on the partitioned path** (`phylosupertree.cpp:725`,
`phylotesting.cpp:3066`) — *exposed* (via a timing/scheduling shift) not *caused* by the guard; it appears OFF-vs-OFF
too. **Decisive settling run (if airtight proof wanted):** OFF-vs-OFF partitioned MFP (num_threads>1, same seed) →
should also drift ~1e-10; and num_threads=1 → ON==OFF exactly (serial reduction). ⇒ **partition bit-identity is
"exact up to OMP reduction order," and the guard is neutral** — record it that way, not as strict byte-identity.

**L.4 — C4a (AA-partition `-B 1000`) — same slow pattern as the mi3 gate G2** (predates the `-B`-drop fix); may
time out. Not a correctness signal.

**L.5 — ⭐ DIRECTION-A LANDED + DECISIVE (mfdevcheck `173802323`, H200, Exit 0, 52min). CORRECT & SAFE, but a
CONFIRMED NON-LEVER for DNA-1M; the governor is COVERAGE.** All correctness gates GREEN: **G1 OFF==REF YES/YES**
(edit byte-inert when off — DNA+AA table md5 identical), **G2 ON==OFF YES/YES** (run-both mirror side-effect-free,
CPU value stays authoritative), **rel_gpu clean n>1e-6 = 0 on all three regimes** (max 1.56e-11 DNA-100K / 1.22e-10
AA-100K / 6.78e-12 DNA-1M). Mirror-vs-CPU-self-check speed (engaged candidates only):

| regime | candidates | engaged | **declined (pinv>0→CPU)** | self-check CPU→GPU | mirror |
|---|---|---|---|---|---|
| DNA-100K | 78 | 39 | **39 = 50%** | 6.075→2.245s | 2.71× |
| AA-100K | 69 | 62 | **7 = 10%** | 53.43→4.79s | **11.16×** |
| DNA-1M | 113 | 74 | **39 = 35%** | 97.21→85.22s | **1.14×** |

Two honest takeaways: **(1) Direction A is NOT the DNA-1M closer.** The mirror wins only **1.14×** at DNA-1M, and the
self-check is only ~6% of the 1610.5s model-phase wall ⇒ Direction A moves **~0.7% of DNA-1M wall.** It IS a real AA
lever (11.16× on a larger self-check share), but AA-1M OOMs (L.2) so unmeasured at scale. This CONFIRMS the §K.9
red-team verdict verbatim. **(2) COVERAGE is the governor — now with hard numbers (grounds §K.3 / the user's
repeated point):** the mirror declines `pinv>0` on **35% of DNA-1M candidates it sees** (39/113), 50% DNA-100K, 10%
AA-100K; and the ~29 that decline JOLT optimize outright (phylotesting.cpp:1337) never reach the mirror at all. Those
declined families run FULL CPU = the wall floor, and **Direction A does nothing for them.** ⇒ **Priority-0 forward
work = coverage (get the +I / +I+G / +I+R / mixture families to optimize AND self-check on GPU), NOT more self-check
tuning.** Build md5 `9d1fd49c`, both sentinels present.

**L.6 — mi3 flag-hygiene gate `173799026` Gate-1 GREEN (promotion-blocking):** old graduate binary vs new
mi3-hygiene binary, `-m MFP -B 1000 --ctf --ts-fused` full pipeline — **DNA-1M BIT-IDENTICAL** (both lnL
−59208019.1016, treefile md5 `54c29857…`) AND **AA-1M BIT-IDENTICAL** (both −7541976.8522, treefile `9035360c…`).
The flag-hygiene changes are byte-neutral. Only G2_aa_part (partition, no-crash — the `-B`-dropped cell) still
running; Gate-1 result unblocks promotion pending G2.

**L.9 — ⭐ HOST ATTRIBUTION MEASURED (mfhostattr `173785380`, A100, `perf record`, guard-ON, `-m MF`) — replaces the
§L.1 inference; the self-check IS the top real host function at 100K, and the DNA/AA split explains the mfdevcheck
mirror ratios.** *(Harness note: the job's inline `perf report` parser was broken — printed only the total sample
count; the real breakdown was salvaged by re-running `perf report -i {dna100,aa100}.data`. Data intact, parser
buggy.)* Host self-time buckets (fraction of CPU samples; GPU kernel time is NOT sampled by perf, so the "idle"
bucket ≈ host blocked on the GPU / serial sections):

| bucket | DNA-100K | AA-100K |
|---|---|---|
| **CPU self-check** (`computePartialLikelihoodSIMD<Vec4d,ns>` + Buffer/dotProductTriple/Derv) | **26.1%** (top real fn 19.6%) | **53.5%** (top fn **44.7%**) |
| OpenMP idle/spin (`__kmp_hardware_timestamp`+`kmp_flag::wait`) = host waiting on GPU/serial | 51.5% | 25.7% |
| CUDA driver (`libcuda`) | 7.6% | 10.8% |
| other | 12.6% | 8.7% |

**Reading it:** (1) the CPU self-check (`computePartialLikelihoodSIMD` — the `:2502` postorder Direction A replaces)
is the **dominant real host function on BOTH types** — so Direction A targets the right host code. (2) The DNA/AA
asymmetry **explains L.5's mirror ratios directly:** AA's host is self-check-bound (53.5% ⇒ mirror **11.16×**), DNA's
host is half GPU-wait (51.5% idle ⇒ mirror only **2.71×**). (3) DNA is host-latency/GPU-wait-bound (threads spinning
on the GPU), AA is host-compute-bound (the postorder). 🔴 **SCALE CAVEAT (do NOT drop):** this is **100K host-self-time,
NOT 1M wall.** At DNA-1M the self-check is only ~6% of wall (L.5) ⇒ this 100K host-dominance does **not** make Direction
A a DNA-1M wall lever. It confirms Direction A is a **100K + AA** lever, and re-confirms **COVERAGE (the OMP-idle/GPU-wait
+ the ~29–35% CPU declines) is the DNA-1M governor**, not the self-check.

**⭐ 1M FLAGSHIP host-self-time ranking (mfoffload `173815952`, H200, `perf -F99`, guard-ON, `-m MF`; salvaged offline
— the inline `--sort=overhead` collapsed all samples to one 100%-total row, same class of parser bug as the 100K job,
`.data` intact).** The 100K picture HOLDS and SHARPENS at flagship scale — the likelihood postorder is even more dominant:

| bucket | DNA-1M (wall 1597.6s, F81+F+G4 bit-id) | AA-1M (wall 2765.3s, LG+G4 bit-id) |
|---|---|---|
| **likelihood-postorder** (`computePartialLikelihoodSIMD<ns>` + `dotProductTriple` + `computeLikelihoodBufferSIMD`) | **~43.5%** (top fn 34.7%) | **~69.5%** (top fn 56.9%) |
| OpenMP spin/barrier-idle (`kmp_flag::wait` + `__kmp_hardware_timestamp`) | ~24% | ~20% |
| unsymbolized (likely libcuda) | ~11% | ~1% |

⇒ **for BOTH types at 1M the removable HOST cost is the likelihood postorder** (the Direction-A self-check), NOT
eigen/model-init and NOT framework — so the first morning offload target is confirmed at flagship scale. 🔴 **SAME WALL
CAVEAT (do NOT drop):** this is host SELF-TIME, not wall — perf does not sample GPU-kernel time. At DNA-1M the self-check
is still only ~6% of WALL (L.5) and COVERAGE remains the DNA-1M wall governor; AA is more host-compute-bound so the
postorder is a larger share of AA wall. The OMP-idle (DNA ~24% / AA ~20%) is the barrier AROUND the postorder
(`schedule(static)` + implicit barrier) = the secondary lever (mfselfdeep). Also confirms the resident guard is
model-selection-correct at 1M for BOTH types (F81+F+G4 / LG+G4 == the OFF baselines).

**L.8 — Phase-1a guard gate `173777619` FINAL: verdict RED, but the guard is CORRECT — the RED is TWO harness
artifacts (verified by hand).** Single-alignment cells ALL genuinely bit-identical AND faster: **C1 DNA-1M GTR+G4
+5.1%, C2 DNA-100K MFP +5.9%, C3 AA-100K MFP +1.9%** (lnL-exact + treefile md5-identical). The two "partition
FAILs" are both false: **C4a (AA-part) — broken RF parser** (the gate's own self-test printed `rf(self,self)=1 RF
PARSER BROKEN`; the `.rfdist` header row `N N` was scanned; **the real RF=0**, lnL exact −807349.876 — verified by
re-running with `tail -n +2`); **C4d (DNA-part) — the benign 7.1e-10** (−564212.444 vs .443, **RF=0**, brlen-level
OMP-order drift, settling run 173802130 in flight). ⇒ **the guard is byte-identical on single alignments and
topology-identical (RF=0) on partitions; RED is a harness bug (broken RF parser — now FIXED in the script) + the
strict-byte-id-vs-OMP-order partition expectation.** Not a guard defect; does NOT affect promotion (separate
default-OFF binary `5ba5e2f1`, not the candidate `bffdc16e`). NOT re-run (would just re-confirm; guard isn't in this
promotion). The partition cell's correct assertion is **RF=0 + lnL rel ≤ 1e-8**, to adopt once the settling run
confirms guard-neutrality.

**L.7 — build-recipe correction (supersedes the §K.9 "Eigen3" note):** the mfdevcheck re-submits failed on a SECOND
cause — the cmake line was missing `-DUSE_CMAPLE=OFF -DUSE_CMAPLE_AA=OFF -DUSE_LSD2=OFF`, so `add_subdirectory(cmaple)`
(CMakeLists.txt:905) ran `FetchContent(googletest)` = a **network fetch at configure time**, and **Gadi compute nodes
have no network** (a login-node configure test FALSELY passes — the login node has network). Fixed by matching the
proven `mfresident` recipe (which builds fine with the same empty submodules *because* those flags skip the subdir).
Populating the git submodules was a red herring. Job `173802323` = the clean run above.

---

## M. FULL MODEL-COVERAGE AUDIT + closure plan (2026-07-15) — "get ALL models on GPU"

**M.0 — principle.** In full `-m MF/MFP` every family is evaluated; ANY family that `JOLT_DECLINE`s to CPU sets the
wall FLOOR (coverage governs the MF wall, [[feedback_coverage_governs_mf_wall]]). So "coverage" = *zero* declines
across the actual candidate set. This section is the **complete source census of every GPU decline gate**, each mapped
to its model class and its relevance to *default* `-m MF` on standard DNA/AA vs non-default data/models.

**M.1 — the decline-gate census (source: `phylotreegpu.cpp` main lnL path + the screener `DECLINE`).**

| decline reason | file:line | model class it rejects | in DEFAULT DNA/AA `-m MF`? | verdict |
|---|---|---|---|---|
| `pure-pinvar-no-gamma` | phylotreegpu.cpp:2210 | pure +I (RateInvar, ncat==1) | **YES** | ✅ CLOSED M.2 (proven) |
| `fixed-pinvar` | :2203 | user-pinned +I{v} | no (MF estimates pinv) | ✅ CLOSED M.2 (optPinv=2) |
| `+R` tiling force-nTile=1 | gpu_lnl_intree.cu:2696 | +R / +I+R at scale (OOM) | **YES** (at 1M) | ✅ CLOSED M.2 (tiling) |
| `invar-sites-brlenonly` | :2242 | +I in the lean/brlen-only reopt | n/a (full path covers +I) | legit-conservative (keep) |
| `no-const-sites` | :2212 | +I when frac_const≈0 (pinv unestimable) | edge/degenerate | legit CPU (keep) |
| `no-rescale-gamma-invar` | :2211 | user `--no-rescale-gamma-invar` (nonstd math) | no (non-default flag) | legit CPU (keep) |
| `non-mean-gamma` | :2189 | **median-gamma** (+G{median}), non-mean cut | no (mean-γ is the default) | GAP-LOW (M.3) |
| `free-subst-params` | :2127 | **+FO / free-optimised Q** beyond freeQok | partial (named GTR/HKY engage via freeQok) | GAP-MED (M.3) |
| `num-states` | :2102 | **codon ns=61, binary ns=2, morph/multistate** | only `-st CODON/BIN/MORPH` | GAP-LARGE (M.3) |
| `ascertainment-bias` | :2108 | **+ASC** (SNP/no-const data) | only +ASC data | GAP-MED (M.3) |
| `not-nonfused-mixture` / `free-per-class-or-linked-gtr` | screener | **profile mixtures C10–C60, EX/UL** | no (needs `-madd`/`-m MIX`) | GAP: kernel EXISTS (JOLTMix), screener not wired (M.3) |
| `null-ptr`, `brlen-mode`, `ncat-range`, `bad-root` | various | defensive / reopt-mode / ncat∉[1,64] | never (guards) | keep |

**M.2 — CLOSED THIS TURN (`iqtree3-pureinvar` @ a07f61be + 3 edits; kill-switches for each; gate `covgate 173822572`).**
For **standard DNA/AA `-m MF`** the candidate set is {named reversible subst models} × {base, +I, +G, +I+G, +R, +I+R}.
After these 3 edits every one of those engages:
1. **pure +I default-ON** (phylotreegpu.cpp:2210, kill `JOLT_NO_PUREINVAR`). Empirically proven: invargate `iP_dna`
   CPU −6054636.2564 vs GPU −6054636.2600 ⇒ **rel 5.9e-10, RF=0, engaged**. Mechanism: meanR init 1.0 + applyAlpha
   ncat>1-guarded ⇒ applyPinv(p) gives catRate[0]=1/(1-p), catProp[0]=(1-p) == RateInvar exactly, no double-rescale.
2. **fixed +I → optPinv=2 apply-don't-step** (:2203, kill `JOLT_NO_FIXINVAR`). Not in MF but requested "ALL". RISK: the
   full-path optPinv==2 was previously brlen-only — red-team + covgate must confirm base_invar populated + rescale
   applied + the 4 pinv-optimise arms (==1-gated) held. Held behind kill-switch; gate is the arbiter.
3. **+R tiling default-ON** (gpu_lnl_intree.cu:2696, kill `JOLT_RTILE_OFF`). The LM lnL(:2919)+gradient(:2981) loops
   Kahan-accumulate across chunks. ⚠️ **CORRECTION (red-team M1, was mis-stated as "bit-identical"):** tiling is
   `rel<=1e-12` to nTile=1, NOT bit-identical (FP64 reductions regroup at chunk boundaries). Because `nTile` is
   auto-picked from free VRAM (:2685), the SAME +R run on A100 (nTile~2) vs H200 (nTile=1) can differ ~1e-6 ABSOLUTE at
   |lnL|~1e6 — above the LM accept threshold (~1e-9) — so accept/reject can flip and the converged tree can differ
   ACROSS GPUs. This merely extends the pre-existing +G/+I auto-tiling non-determinism to +R, and is backstopped by the
   FULL-path CPU self-check (:2512) which returns `cpuLnL` not the GPU value. Pin `JOLT_NTILE=1` for bitwise
   reproducibility. The old force-nTile=1 caused the AA-1M +R 74.2GB OOM (only load-bearing on A100-80GB; on H200-139GB
   +R already fits at nTile=1 — so `rtilegate`'s OOM-buster CONTROL cannot OOM there; the OOM-avoidance VALUE needs an
   A100 run). `rtilegate` quantifies the nTile=1-vs-2 divergence (M1) and confirms AA-1M +I+R4 completes on tiling.

⇒ **claim (to be confirmed by the definitive census M.4): default DNA/AA `-m MF` is now DECLINE-FREE.** The census is a
SOURCE audit; the empirical proof is covgate (the +I/+R families directly) + M.4 (a real full `-m MF` decline count).

**M.3 — remaining gaps = NON-default data/models, prioritized (each a bounded kernel/wiring project, NOT a one-liner).**
- **P1 free-frequency (+FO / estimated base freqs)** — GAP-MED. `free-subst-params` fires when ndim≠0 and not freeQok.
  Named models engage; estimated-Q variants don't. Closure = extend freeQok to the estimated-Q Jacobian (plumb the
  extra dims through the LM). Effort: moderate (LM dimension bookkeeping, no new kernel). *Confirm it's even in MF's set first.*
- **P2 mixtures — GROUNDED CORRECTION (Plan-agent, source-verified; supersedes my earlier "screener
  `not-nonfused-mixture` decline" which was WRONG — that string is in the FULL-path `JMIX_DECLINE` block
  phylotreegpu.cpp:2746, NOT the screener; the screener declines ALL mixtures wholesale at :1488 `getNMixtures()!=1`).**
  Three surfaces, three statuses:
  - **C10–C60 (fixed AND estimated weights), +G/+I+G — ALREADY CORRECT on the full GPU path** (kernels `k1_node_mix`
    gpu_lnl_intree.cu:115 …; self-check rel≤1e-6 :2948; MEOW80 rel 2.46e-13). MF dispatch gates it behind
    `JOLT_MIX_HOSTDRIVEN` (modelfactory.cpp:1606), default-**OFF for LATENCY** (host-driven ~4.4s/outer), NOT
    correctness. **CLOSABLE NOW = pure dispatch, NO kernel work** (validate that flag as the supported path; optional
    fixed-weight fast-path predicate split). Value VERY HIGH (C-series = the AA deep-phylogeny standard).
  - **EX2/EX3/EHO/UL2/UL3 rate mixtures — engage-then-DECLINE** at the rate-1 guard (:2927): a missing per-class
    `total_num_subst` scaling in the echild builders (:306/:1791, device :1458/:1634). **Host math fix, NO new kernel**
    (thread `tns[]` through 3 builders + 2 launchers, then relax the guard behind a flag). Currently SAFE (declines,
    never silently wrong). Value low-med.
  - **LG4X/LG4M (fused N==ncat diagonal) + the mixture SCREENER (:1488, launchers carry no `nmix`) — genuine KERNEL
    projects, NOT wiring.** Defer. The device-resident `gpu_jolt_optimize_mix` (the graduation path to make C-series
    default-on, killing the host-driven latency) is the real perf unlock — large. Full tiered plan + per-class
    red-team targets (EM weight normalization, per-class rate coupling, +I×mixture, underflow) = Plan-agent output.
- **P3 +ASC (ascertainment bias)** — GAP-MED. Add the ascertainment-correction term (conditional-lnL normaliser) to the
  kernel objective + gradient. Effort: medium kernel. Only for SNP/morph data.
- **P4 codon ns=61** — GAP-LARGE. New `k1_node_t<61>` template + genetic-code plumbing + register/occupancy re-tune
  (ns=61 ≫ 20 ⇒ likely tiling). Effort: large. Only for `-st CODON`.
- **P5 binary ns=2 / morph multistate** — GAP-MED. ns=2 template (cheap) + morph multistate + usually paired with +ASC.
- **P6 median-gamma** — GAP-LOW. Non-default rate cut; add the median-cut rate vector. Effort: small. Rarely selected.

**M.4 — the DEFINITIVE coverage test (forward gate): a full `-m MF` DECLINE CENSUS.** Run real `-m MF` (DNA + AA,
`--jolt --gpu`, `JOLT_DBG=1`) and count `[JOLT-GATE] decline reason=` by reason across the WHOLE candidate sweep. The
audit's claim ("default MF decline-free after M.2") is TRUE iff that count is 0 for standard DNA/AA. This is the honest
replacement for the memory's unaudited "DNA ~89% GPU" — it measures coverage directly instead of inferring it.

**M.5 — RED-TEAM AUDIT of the M.2 diff (agent, source-grounded, 2026-07-15). Verdict: GO-WITH-GATE; no CRITICAL/HIGH
wrong-likelihood bug; ONE overclaim corrected.**
- **Load-bearing backstop (governs pure/fixed +I):** both DECLINE the brlen-only/leanTail path (`phylotreegpu.cpp:2242`,
  `invarBrlenOK` needs `optPinv==1 && ncat>1`) so they reach ONLY the FULL path, which recomputes a CPU likelihood at
  the written-back params (`:2512`), returns NaN→CPU if `!(rel<=1e-6)` (`:2535`), and **returns `cpuLnL` not the GPU
  value** (`:2543`). A latent +I error can at worst silently decline to CPU — never ship a wrong likelihood.
- **(A) pure +I — CONFIRMED CORRECT in code:** `meanR` init 1.0 (`gpu_lnl_intree.cu:2769`), all 3 `applyAlpha`
  `ncat>1 && !freeRate`-gated (`:2772/2910/3154`) so `meanR[0]≡1.0`; `applyPinv` → `catRate[0]=1/(1-p)`,
  `catProp[0]=(1-p)` == RateInvar (`rateinvar.h:77,84`); no double-rescale; no div-by-zero (`p<frac_const<1`). Matches
  the empirical rel 5.9e-10.
- **(B) fixed +I `optPinv==2` FULL path — CONFIRMED CORRECT:** APPLY sites truthy-`optPinv?` (`:2779/2786/2787/3090/
  3095/3126/3283/3291`); ALL 4 pinv-STEP arms strictly `==1`-gated (`:3176/3234/3241/3253`) so pinv is HELD; no pinv
  Jacobian column enters the LM; alpha gradient carries `f=(1-curPinv)` so branches+alpha move against the correct
  objective (the historical 21-nat bug was the OPPOSITE — `optPinv=0` dropping the term in an un-self-checked path).
- **(C) +R tiling — buffers correct** (chunk-sized `d_rnum/d_wnum` `:2732/2764`, write & read both stride `Pn`, no
  cross-chunk reads; full-nptn buffers only under RGRADCHECK which the guard forces to nTile=1). **BUT M1 (MED,
  confirmed): NOT bit-identical** — tiling is `rel<=1e-12`, and auto-nTile-from-VRAM (`:2685`) makes +R non-reproducible
  ACROSS GPUs (A100 nTile~2 vs H200 nTile=1 → ~1e-6 absolute at |lnL|~1e6 → LM accept/reject can flip → tree can
  differ). Backstopped by the FULL-path CPU `cpuLnL` return; same class as existing +G/+I auto-tiling. **Action: pin
  `JOLT_NTILE=1` for reproducible runs; `rtilegate` (nTile=1 vs 2) quantifies the divergence.**
- **L1 (LOW):** the leanTail +R-brlen path (`freeRate==3`) now tiles with NO CPU self-check → intermediate `curScore`
  VRAM-dependent at ~1e-12 (final tree still CPU-re-optimized). **L2:** no compile issue (`jolt_fixp` scope correct).
- **Recommendation (adopted):** ship behind the 3 kill-switches (current posture); add a +R cross-nTile reproducibility
  cell before removing the gate; +I edits safe given the FULL-path self-check backstop.

**M.6 — CENSUS LANDED + a methodology correction I have to own + the CTF-lockstep gap (2026-07-15, grounded).**

**🔴 Methodology correction (my error, disclosed):** the FIRST census (`173822920`) and the first covgate/invargate
runs set `JOLT_DBG=1`, but the code reads `getenv("JOLT_DEBUG")` (phylotreegpu.cpp:2086) — so `[JOLT-GATE]` logging was
NEVER ON. My earlier "engaged, 0 declines" readings were therefore **ungrounded** (rel≤1e-6 + RF=0 only proves the two
arms agree; it cannot distinguish GPU-engage from a silent CPU-fallback). Fixed to `JOLT_DEBUG` + an explicit
**`reached-hook`** check (engage = the model ENTERED `optimizeParametersJOLT`, not merely "no decline logged").

**Census result (CORRECTED `173824353`, `JOLT_DEBUG`, no-`--ctf` `-m MF`, 100K):**
| arm | reached-JOLT-hook | hook declines | best-fit |
|---|---|---|---|
| DNA treatment (edits ON) | **79** | **0** | F81+F+G4 |
| AA treatment | **72** | **0** | LG+G4 |
| DNA control (all kill-switches) | 79 | **1 (pure-pinvar-no-gamma)** | F81+F+G4 (unchanged) |
| AA control | 70 | **1 (pure-pinvar-no-gamma)** | LG+G4 (unchanged) |

⇒ **no-`--ctf` `-m MF` DNA/AA is DECLINE-FREE at the JOLT hook** (treatment=0), and flipping the kill-switch makes
EXACTLY the pure-+I decline reappear = **causal proof the pure-+I edit closed the last hole.** The +I+G/+I+R/+R/+G
families already reached the hook and engaged (hook lines confirm, e.g. `GTR+F pinv=0.0094 ncat=4` = +I+G4 engaging).
Best-fit model unchanged. **`ctfIneligible` (below) does NOT fire in no-ctf** — verified it's called only in the CTF
rerank, so every rate family reaches the in-tree gate here. *(Corrects the earlier `reached-hook=0` reading, which was
purely the `JOLT_DBG` artifact.)*

**🔧 The CTF-lockstep gap — the concrete "`-m MF` with CTF" item + EDIT + RED-TEAM (2026-07-15, grounded).** The census
is no-ctf. **⚠️ Mechanism CORRECTED (red-team F1-C — my first framing was wrong):** `ctfIneligible`
(phylotesting.cpp:1337) is consulted ONLY in `selectCTFTopK`'s refine-SKIP decision (:1385/:1416), NOT the coarse pass.
The coarse subsample ranking (`evaluateAll`→`optimizeParameters`→`optimizeParametersJOLT`) engages +R via the IN-TREE
gate (the M.2 edits) — so it does NOT route +R to CPU. What `ctfIneligible` did was **SKIP +R/+I in the full-data refine
step** (`skip = ctfIneligible(name) && behind-best-eligible`), i.e. a +R model that leads on the subsample would never
get full-data refined ⇒ **the +R winner is dropped** (the detector at :1398 warns of exactly this — the avian case).
```
was:  return name.find("+R") || (name.find("+I") && !name.find("+G"));   // skips +R/+I in refine
```
**EDIT LANDED (phylotesting.cpp:1337):** `ctfIneligible` → `false` by default (kill-switch `JOLT_CTF_LEGACY` restores the
old skip); +R/+I are now REFINED like any eligible model (on GPU via the in-tree gate). **RED-TEAM VERDICT: GO** — the
refined set becomes a strict SUPERSET, winner = min-crit over it, so selection can only tie-or-improve; declined families
(mixture/codon) NaN→CPU-fallback, no crash (F1-D). **🔴 Self-audit caught + fixed a CRITICAL abort I introduced:**
always-false broke the Release-active `ctfSelfTest` fixture (`LG+R5 must be skipped` ASSERT → `abort()` on the first
`--ctf` run); fixed by making the fixture env-aware (`LG+R5.skip == legacy_mode`, :1459-1462) — confirmed sound by the
red-team. LOW residual: the +R-winner *detector warning* (:1398) is now permanently dead — fine, it warned about the
skip this edit removes. **Validation DONE — `--ctf` A/B on the avian (ctfavab 173832078, 2026-07-15):** the avian 100k genuinely selects a
+R model (**no-ctf oracle = GTR+F+I+R4**, confirmed at 500k). Result = **SAFE-NEGATIVE**: **legacy `--ctf` = GTR+F+I+R2
== fixed `--ctf` = GTR+F+I+R2** (both arms agree). ⇒ the lockstep fix is **harmless but not decisive here** — the +R
model LED the coarse subsample, so legacy's skip (which only drops models BEHIND the best eligible) never bit it; both
refined +I+R2. **My `ctfSelfTest` abort fix HELD** (both `--ctf` arms completed clean, no abort). **Separate finding
(not the lockstep):** `--ctf` UNDER-RESOLVES the +R order — both CTF arms pick R2 while the full oracle picks R4 (both
correctly in the +I+R FAMILY; the coarse subsample just under-resolves the rate-category count). A decisive lockstep
test still needs data where a +R model is BEHIND on the coarse pass yet wins on full data. **KEY GROUNDING: the avian
real data selects GTR+F+I+R — exactly the family with the ~10-nat offload gap (§N.6/§3 of MF-FULL-GPU-COVERAGE.md), so
+I+R coverage is a REAL-DATA priority, not synthetic-only.**

---

## N. FULL-GPU MF DESIGN — get EVERY family's likelihood+self-check on GPU, and the walltime frontier (2026-07-15)

Synthesis of three grounded streams this session: (1) **profiling** (perf `.data` mfoffload 173815952 @1M + DEVCHECK
173802323 per-candidate timing), (2) a **source cause-map** (agent, every claim file:line-verified this turn), and
(3) a **literature survey** (BEAGLE lineage + model-selection tools, cited in N.2). It supersedes the "which lever"
guesswork in §E/§J with measured facts.

### N.0 — The grounded evidence base (measured, not inferred)

- **§K.6 RESOLVED — per-candidate FRAMEWORK cost (new IQTree / eigendecompose / initModel) is NEGLIGIBLE.** It does
  not appear in the top-15 perf symbols on DNA-1M or AA-1M. The host self-time is almost entirely the CPU likelihood
  recompute: **DNA-1M 43.5%** (`computePartialLikelihoodSIMD<4>` 34.72% + `bufferSIMD` 4.48% + `dotProductTriple`
  4.29%) + **~24% OpenMP idle**; **AA-1M 69.5%** (`…SIMD<20>` 56.86% + 7.87% + 4.73%) + **~20% OMP idle**. ⇒ **the lever
  is the self-check + the CPU-declining families, NOT framework and NOT a kernel rewrite.** This retires the "framework
  might dominate ⇒ Phase-1b insufficient" hedge — the offloadable term is the whole host cost.
- **The DEVUSE self-check win is ns-DEPENDENT (measured, DEVCHECK 173802323):** AA (ns=20) mirror **7–11×** cheaper/
  candidate (0.88s→0.12s); DNA (ns=4) only **~1.3×** (rate-het 110.9s→84.1s over 85 candidates). The mirror's per-call
  setup (g_Uinv upload + echild build) does not amortize against cheap 4-state compute. **This is the state-count
  lever (N.2-Q1), not a defect** — and it is exactly why we win AA and struggle on DNA.
- **Per-family self-check cost (DNA-1M on_d1m, 113 candidates):** `+I` family = **27 candidates / 32.6s CPU, ALL
  declined** (mirror pinv-gate) = the single biggest un-offloaded block; rate-het = 85 / 110.9s CPU→84.1s GPU; base =
  1 / negligible. **⇒ the +I self-check gap is the top coverage target on DNA.**

### N.1 — Per-family cause + bounded fix (source-verified this turn; full table in the agent record)

Four gate surfaces: optimizer-single `optimizeParametersJOLT` (`phylotreegpu.cpp:2081`), **mirror-single**
`gpuComputeTreeLnLCleanRoom` (`:76`, gates `:78–84`), optimizer-mixture `optimizeParametersJOLTMix` (`:2780`),
mirror-mixture `gpuComputeTreeLnLCleanRoomMix` (`:192`, already does +I via `clsinv`). The CPU self-check at
`:2555`+`rel≤1e-6` gate `:2589` backstops everything (coherent-but-wrong ⇒ NaN ⇒ CPU), so every gap below is *safe but
slow*, never wrong.

| family | blocker (verified file:line) | bounded fix | effort/risk |
|---|---|---|---|
| **+I+R** (default MF set) | optimizer ENGAGES, **reaches the MLE** (`joltLnL`), but **writes back params ~10 nats worse** → self-check `rel≈1.3e-6` trips → full CPU. **`:3128` root cause REFUTED (§N.6 CORRECTION-1); `gaugeFix` invariant `worst\|d\|=0` in `-te` AND `-m MF` (mfgauge 173893718); re-eval RETRACTED harmful. RESOLVED in `MF-FULL-GPU-COVERAGE.md` §3b.** | **writeback reconciliation**: write back the exact accepted-best params that scored `joltLnL` (NOT a re-eval, NOT a gauge change). Separable perf: decoupled per-arm LM damping (AA zigzag). | host-side, no kernel / **LOW-MED** |
| **+I / +I+G** (self-check) | mirror **declines pinv>0** `:84`; `gpu_lnl_crosscheck` sig has **no pinv/base_invar** (verified); `k1_node` root `:630` `log(fabs(lh))` has no invariant term (verified) | add `pinv·base_invar[ptn]` before the log at `:630`; add `pinv,base_invar` params to `gpu_lnl_crosscheck`; copy the `base_invar` build from `:2301-2315`. **+I = one extra rate-0 category, identity P, no branch** (N.2-Q2) | host-math + 1 kernel term / **LOW** |
| **C10–C60 profile mix** | **CORRECT already** (mix optimizer+mirror, MEOW80 rel 2.46e-13); blocked only by dispatch `JOLT_MIX_HOSTDRIVEN` OFF for **latency** (`modelfactory.cpp:1606`) | flip/validate the dispatch flag; real unlock = device-resident `gpu_jolt_optimize_mix` (**does not exist**) | dispatch LOW-MED; resident optimizer LARGE |
| **EX/UL rate-mix** | engage-then-decline at rate-1 guard `:2981` (per-class `tns≠1`) | thread `tns[]` through 3 echild builders + 2 launchers | host-math / LOW-MED |
| **LG4X/LG4M fused** | decline `:2800`/`:212` (`isFused`) — 1:1 class↔rate diagonal | new fused-diagonal kernel | new kernel / MED |
| **+FO free-freq** | decline `:2127` (freeQok excludes FREQ_ESTIMATE) — *confirm it's even in default MF set* | plumb free-freq dims through the LM Jacobian | MED |
| **codon ns=61 / binary ns=2 / morph** | decline `:2102`/`:80` (ns∉{4,20}); kernels only `<4>/<20>` | new `<61>`/`<2>` templates + genetic code; codon likely tiled | codon LARGE, binary small |
| **+ASC** | decline `:2108`; kernel `nptn` excludes `unobserved_ptns` | GPU unobserved-pattern lh + subtract `N·log(1−ΣP_unobs)` from objective+grad | MED kernel |
| **median-γ** | decline `:2189` (only mean-cut) | add median-cut rate vector | small |

### N.2 — Literature synthesis (cited; full source list in the agent record)

- **Q1 — the memory-latency-bound nature is the DOCUMENTED CONSENSUS, not our bug.** Suchard&Rambaut 2009 spawn
  (rate×pattern×state) threads that "spend most effort on memory operations"; Gangavarapu 2024 (Bioinformatics
  btae030) states the algorithm is *"memory-bound on both the CPU and the GPU"*; the 2026 FP64 **tensor-core** BEAGLE
  v4 (Syst Biol syag017) *still* "exhaust[s] the memory bandwidth… saturation of performance." Our ncu (~19% SM,
  single-digit FP64) is the expected regime.
- **THE STATE-COUNT LEVER (decisive).** BEAGLE 3 (Ayres 2019): higher state count "increases the ratio of computation
  to data transfer ⇒ increased GPU performance." Tensor-core gains: **codon ~3×, AA ~2.3×, nucleotide only ~1.1×.**
  This is the mechanistic explanation of our DNA-loses / AA-wins split and of the DEVUSE ns-dependence (N.0).
- **Q2 — +I is SOLVED & STANDARD:** one extra **rate-0 category, weight p_inv, identity P-matrix**, folded into the
  same category sum (BEAST2, MrBayes, BEAGLE). Removes the "is-site-constant" branch divergence. Matches our
  `base_invar` fix. The only numerical care is invariant-class × rescaling (standard per-site max).
- **Q3 — +R is KERNEL-TRIVIAL, OPTIMIZER-HARD.** Free-rate = +G with different (rate,weight) vectors (Q fixed), so
  the offloaded likelihood is byte-identical shape. The hard part is bit-reproducible optima: Höhler/Stamatakis 2023
  (MBE, PMC10518076) show IQ-TREE is ~10× more threshold-sensitive than RAxML-NG — FP-associativity/threshold changes
  move the +R optimum. No published "reproducible GPU free-rate" recipe. Confirms our +I+R normaliser is the fix and
  our OMP-reduction-order determinism work was necessary.
- **Q4 — mixtures EXPRESSIBLE but UNPORTED:** per-class eigen/P folded as a class dimension (Le/Gascuel/Lartillot
  2008; GTRpmix 2024); BEAGLE runs multi-eigen on GPU but **PhyloBayes CAT and IQ-TREE mixtures are CPU-only** — no
  ML-world GPU mixture to copy. 20-state ⇒ GPU-favorable (state-count lever). Our JOLTMix already works (2.46e-13).
- **Q5 — THE NOVEL FRONTIER (model-selection loop).** Every selection tool (jModelTest/ProtTest/ModelTest-NG) is
  one-model-per-CPU-worker; **none is GPU-resident or batches candidates.** Building blocks exist: MrBayes tgMC3 (Zhou
  2013) keeps CLVs resident + **fuses 5 kernels→1** (3.7–5.7× on memory-bound DNA via residency+transfer-elimination,
  NOT compute); BEAGLE 3 (Ayres 2019) batches "multiple CLV arrays in one launch" for under-utilized few-pattern data.
  **The unpublished step = redirect that batching from partitions to CANDIDATE MODELS.**

### N.3 — THE DESIGN (two tiers, honest about what each buys)

**TIER 1 — COVERAGE: get every family's self-check on GPU (closes the 43.5%/69.5% host term for ALL families).**
Ordered by value/effort AND by a hard correctness dependency:
1. **FIX the +I+R optimizer FIRST** (`gpu_lnl_intree.cu:3128` deferred `(1−pinv)` normaliser). Highest value (it's a
   default-set family running full CPU today) AND a **prerequisite landmine**: if we enable the mirror self-check for
   `pinv>0` *before* this, GPU-optimizer-vs-GPU-mirror becomes a **tautology** (both wrong the same ~10 nats, the K.4
   blind spot) → silent mis-selection. Gate: scab-style rel≤1e-6 restored on +I+R, DNA+AA, bit-id vs CPU oracle.
2. **THEN add the +I term to the single-model mirror** (`:630` + `gpu_lnl_crosscheck` sig). Enables Direction-A self-
   check for +I/+I+G (35–50% of candidates) — turns the mirror from base/+G-only into all-single-model. Gate:
   selection-invariant + rel≤1e-9 on +I/+I+G vs CPU. **Only after (1)**, so the mirror check on pinv>0 is meaningful.
3. **Flip mixture dispatch** (`JOLT_MIX_HOSTDRIVEN`) for C10–C60 (correct already; latency-gated). Gate: C-series
   best-fit vs CPU oracle on a real AA set; measure the host-driven latency to decide if the resident optimizer (Tier 2)
   is needed.

**TIER 2 — WALLTIME FRONTIER: the batched/resident candidate loop (the DNA lever + the novel angle).**
- **BATCHED-CANDIDATE KERNEL (novel, N.2-Q5).** On the fixed MF topology, the K candidates share identical tip data,
  tree, and pattern-weight vector; they differ ONLY in P-matrices (Q-eigen × brlen × rate). Add a **model dimension to
  the launch grid** (`models × R × ⌈patterns/CBS⌉`): read tip data + pattern indices ONCE, reuse across all K models,
  only the K P-matrix sets differ. For our 19%-SM latency-bound kernel this multiplies independent latency-hiding work
  **without** growing the dominant (shared) tip-data bandwidth — precisely BEAGLE 3's "multiple CLV arrays in one
  launch," redirected to candidates. **First batch = models sharing peeling shape** (same ns, same ncat, differ only in
  Q — e.g. all nucleotide +G4 candidates GTR/HKY/TN…). This is the highest-upside DNA lever and is unpublished.
- **GPU-RESIDENT + FUSED MF LOOP (tgMC3 precedent).** Keep {d_tip, d_ptnfreq, tree} resident across candidates (K.1:
  the arena already persists; residency is the building block), fuse the per-candidate launch storm (J.7: 1.7M
  `cudaLaunchKernel` + 1.16M `cudaMemcpyToSymbol` at 100K), drive from one host thread. Literature predicts this — not
  kernel tuning — is the DNA lever, and it's exactly what the competitor's whole-likelihood-resident OpenACC port does
  to win DNA. Reclaims the measured **16–24% OMP idle** (host waiting on serial single-GPU work).

### N.4 — Honest verdict (what's standard vs genuinely hard)

- **+I: standard/solved** (rate-0 category). **+R kernel: trivial; +R/+I+R OPTIMA: genuinely hard** (reproducibility —
  literature-confirmed; our normaliser fix + determinism work is the right track). **Mixtures: expressible, no GPU
  precedent to copy** (engineering, not research; 20-state favorable). **Batched-candidate model-selection loop: NOT
  solved, NOT standard — open ground, and the right frontier for the DNA walltime we're losing.**
- **Strategic truth:** Tier 1 makes coverage complete and correct (necessary; wins AA decisively via the state-count
  lever). But **beating the competitor on DNA no-`--ctf` needs Tier 2** (residency + batching + fusion), because DNA is
  memory-bound/low-intensity where only transfer-elimination and occupancy help — a kernel that is already near its
  ceiling cannot be tuned into the win. Coverage ≠ the DNA throughput win; both docs and the literature agree.

### N.5 — Landmines (carry into implementation; to be red-teamed)

1. **Ordering: +I+R optimizer fix BEFORE mirror-pinv self-check** (tautology risk, N.3-Tier1). Non-negotiable.
2. **+R tiling cross-GPU non-determinism** (M.5-M1): pin `JOLT_NTILE=1` for reproducible +R/+I+R; auto-nTile can flip
   LM accept/reject across A100/H200.
3. **Mixture rate-1 guard `:2981` is load-bearing** — keep it when extending EX/UL; a slipped `tns≠1` mis-scales all
   branch lengths.
4. **Clone reality:** DEVUSE lives in `iqtree3-mfdevcheck`; the M.2 coverage closures (pure-+I, +R-tiling) live in
   `iqtree3-pureinvar`. A combined binary needs an explicit **merge**, not an assumption that one tree has both.
5. **Batched-candidate kernel correctness:** each candidate's rescaling/underflow is independent — a shared-launch must
   not let one model's scale factors leak into another's; per-(model,pattern) scale state required.

### N.6 — RED-TEAM of §N (agent, source-verified 2026-07-15). Verdict: coverage math SOUND; THREE overclaims corrected.

The red-team caught §N reaching past its own measured evidence in exactly the three places to distrust. All verified in source this turn; the affected claims above are **superseded by this subsection**.

**CORRECTION-1 (CRITICAL) — the +I+R "root cause = deferred `(1−pinv)` normaliser @ `gpu_lnl_intree.cu:3128`" is REFUTED (misattribution).** `JOLT_REM_EN` (`:3129`) is consumed at exactly ONE site, `:3267` `remW=(JOLT_REM_EN && optPinv==0)` — a **default-OFF, pure-+R-only** EM feature. **+I+R (optPinv≠0) never enters it**; the `:3270` comment even labels the gradient arm "OFF path / +I+R". The default +I+R path ALREADY handles `(1−pinv)`: forward `:236` `Lp=lh+pinv*baseinvar`, weight-gradient `:3092-3094` derives the +I normaliser explicitly (`wnorm=optPinv?sumWN:rN`). **⇒ there is no `:3128` fix, and §N.3-Tier1's "FIX +I+R optimizer FIRST" + §N.5#1 "non-negotiable ordering landmine" are DELETED (phantom dependency).**
   - **BUT the red-team's counter ("scab ~10-nat = the mirror omitting `ptn_invar`") is ALSO not fully right.** scab's `[JOLT]` line compares `joltLnL` (from `gpu_jolt_optimize` `:2346`) vs the **CPU postorder** `cpuLnL` (`:2554`) — the mirror is NOT in that comparison. And the signature falsifies any simple missing-invariant-term story: the offset is **~10.00 nats, INVARIANT to pinv** (JC+R2 pinv 0.0184 → +10.0008; JC+R4 pinv 0.00036 → +10.0069; a 50× pinv change, same offset) and **identical for DNA and AA**. A missing `pinv·base_invar` would scale with pinv; a near-constant ~10.00 does not. **⇒ +I+R root cause RESOLVED by T2 (irtrace 173843042, 2026-07-15) — see `MF-FULL-GPU-COVERAGE.md` §3.** The trace
decomposed the offset: `diff` stays ~10.00 while `invContrib` (the real invariant contribution) ranges 0.38→7930 with
pinv, and `restore_rel=0`. ⇒ the GPU computes the invariant CORRECTLY (not a missing term) and the CPU self-check state
is clean (not a write-back bug). **The ~10.00 is a CONSTANT systematic error in `gpu_jolt_optimize`'s freeRate∧optPinv
path**, present whenever pinv is ESTIMATED with free-rate, independent of the pinv value. Next = bisect the gauge/return
(`gaugeFix` / `out_props`); the +I mirror term (T1) becomes the independent forward-lnL oracle for it. Real-data
priority (avian = GTR+F+I+R4).** Safe today: the `:2589` self-check catches it → CPU fallback (so +I+R runs full CPU — a real wall cost, but never wrong).

**CORRECTION-2 (HIGH) — the batched-candidate kernel is NOT "the highest-upside DNA lever"; it is contradicted by our own §J.7 + the constant-memory layout.** §J.7:583 already lists "K-batching DEAD on these kernels" and :579-582 "occupancy-raising is DEAD … near practical ceiling." Worse, the eigen state (`g_Uinv/g_val0-2/…`) is single-model `__constant__` memory (`gpu_lnl_intree.cu:34-43`, overwritten per model via `cudaMemcpyToSymbol`); the 64KB budget holds **only ~2 AA models** (~30KB/model at ns=20), so batching forces eigen into GLOBAL memory — ADDING traffic to the exact binding resource (AA `<20>` = DRAM/L2). And the dominant per-thread traffic is the **per-model** double-precision CLV/echild reads (grow ×K), not the 1-byte tip data. **⇒ downgraded from "lever" to "SPECULATIVE — needs a 2-model-vs-1-model AA microbenchmark (constant-mem overflow + DRAM traffic) BEFORE it can be called a lever." Likely neutral-to-negative on these kernels.** (thread-per-pattern means batching adds blocks not registers — that part is benign; the memory-subsystem saturation is the killer.)

**CORRECTION-3 (HIGH) — "residency+fusion reclaims the 16-24% OMP idle" is an overclaim (contradicts K.2 + J.7).** K.2: cross-candidate residency has "<2s headroom … NOT a standalone perf lever." J.7:609: "the resident loop's value is freeing the host, not cutting sync wall." The 16-24% idle is the **16-way CPU-postorder's own parallel inefficiency** (inside the 43.5%/69.5% self-time), which residency/fusion of the GPU launch storm does NOT touch; in the serial `-m MF` regime freeing the host has nothing to overlap. **⇒ residency reframed as an ENABLER only (K.1 language). The one genuinely reclaimable slice = the launch/`toSymbol` host-API storm (~13s+8s @100K, J.7) — which has a GRAVEYARD/hidden-race history (J.7:591-592). Sized as a secondary, not headlined.**

**Sound + build-ready (red-team-confirmed):**
- **The +I mirror invariant term** (add `pinv·base_invar[ptn]` before the `:630` log) — math verified vs CPU (`phylotreesse.cpp:1196`/`:588-616`), no double-count (`lh` already carries `(1−pinv)` via `g_catw`). **Effort corrected UP from "1 kernel term/LOW":** `k1_node` is a SHARED kernel (isRoot callers at `:1289,1759,1866,1971,1992,2097…`) → thread a default-no-op `pinv/base_invar` param through all sites or make a variant. Gate: verify `gpu_lnl_crosscheck` uploads the `(1−pinv)`-scaled `catProp`; note the mirror runs unscaled (NORM_LH `:74`). **This ONE fix covers +I, +I+G AND +I+R self-check** (N.1 rows 1+2 merge; there is no separate optimizer fix).
- **Mixture dispatch flip** (`JOLT_MIX_HOSTDRIVEN`) — grounded (MEOW80 2.46e-13); measure host-driven latency before claiming it as the path.
- **State-count-lever / AA-win framing (N.2/N.4)** — well-cited, consistent with ncu.

**Also carry (red-team):** §N.0 "framework NEGLIGIBLE" softened to "no single framework symbol in top-15; the distributed sum (per-candidate `initializeAllPartialLh` first-touch of ~hundreds-MB CLV) stays bounded only by the pending `mfhostattr` J.4 measurement" — do NOT fully retire K.6's hedge. And the mirror coverage "for ALL families" holds **serial-regime only** (`!omp_in_parallel()` `:2530`; the `__constant__ g_Uinv` write needs the `jolt_gpu_mtx` lock for CTF-coarse/`--thread-model`/partition).

**Net:** the honest deliverable is **ONE sound coverage fix (+I mirror term, covers all pinv>0) + a mixture dispatch flip**, plus **two UNRESOLVED questions that each need a cheap spike before any build** (the +I+R constant-~10 offset; the batched-candidate microbenchmark). The two "novel Tier-2 levers" as headlined were refuted by our own prior evidence — corrected here, not defended.
