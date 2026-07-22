# PROFILE — full `-m MFP` device+host bottleneck campaign (2026-07-17)

**Goal (user):** find the *remaining deep bottlenecks* in GPU-MF offloading — the suspected **large synchronisations
+ residual host CPU** that persist even with the offload. Decompose the full `-m MFP` wall into DEVICE (GPU kernels |
H2D/D2H | cudaAPI sync/launch) via **nsys** and HOST (residual CPU self-time by symbol | OS-runtime waits) via **perf**.

## Method (grounded — every number traces to a job ID)
- **Script:** `gadi-ci/gems/gems_mfp_profile_campaign.sh` (8-cell PBS array). Base cmd = the lead's:
  `-m MFP -seed 1 -ninit 2 -optalg 2-BFGS -nt 12 -starttree PARS --jolt --gpu -redo` (NO `--ctf`).
- **Device pass (nsys):** whole-run `-t cuda[,osrt],nvtx --wait=all` on the **EXACT shipped `f3f7875f`**
  (`iqtree3-jolt-merge/build-merge/iqtree3`). Reports: `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`,
  `cuda_api_sum`, `osrt_sum`. osrt dropped only on the 3 long 1M cells to bound the rep.
- **Host pass (perf):** `-e task-clock -F 99 --no-children` (self-time by symbol) on an **unstripped twin
  `iqtree3-prof`** (md5 `f1e428f2`) — relinked in 11 s from the **identical `.o` set** that built `f3f7875f`
  (link.txt has `-g -rdynamic`, no strip; same per-function codegen, full `.symtab`). The shipped binary is
  host-stripped, so perf on it leaves the hottest host fn as a raw `0x…` offset; the twin NAMES it. Device numbers
  stay on the exact shipped binary; only host **symbol names** come from the twin.
- **Jobs:** cells 0–1 = **`174072991`** (final script); cells 2–7 = **`174073365`**.
- **Red-team fixes baked in** (see §method-notes): the `--duration` **orphan** blocker, the `readelf` compute-node
  gate, the perf `--children` self-vs-cumulative trap, GPU-engage + empty-kernel-block VOID.

## Status — ✅ CAMPAIGN COMPLETE (8 cells; euk_22k VOID)
| # | cell | job | wall (s) | GPU util | best | host top-1 | state |
|---|------|-----|----------|----------|------|-----------|-------|
| 0 | dna_10k  | 174072991[0] | 148.1 | 29% | F81+F+G4 | kmp_wait 57.3% | ✅ done |
| 1 | dna_100k | 174072991[1] | 154.1 | 13% | F81+F+G4 | kmp_wait 52.0% / cP⟨4⟩ 8.1% | ✅ done |
| 2 | dna_1m   | 174073365[2] | 1683 (MF ~1597) | 52-66% | F81+F+G4 | **cP⟨4⟩ 15.0%** | ✅ done |
| 3 | aa_10k   | 174073365[3] | 230.6 | 40% | LG+G4 | kmp_wait 47.8% / cP⟨20⟩ 16.0% | ✅ done |
| 4 | aa_100k  | 174073365[4] | 546.4 | 46% | LG+G4 | **cP⟨20⟩ 30.7%** | ✅ done |
| 5 | aa_1m    | 174073365[5] | 5062.0 | 30% | LG+G4 | **cP⟨20⟩ 40.4%** | ✅ done |
| 6 | avian_1m (real DNA 48×1M) | 174073365[6] | 1216.1 | 30% | GTR+F+R6 | **cP⟨4⟩ 22.8%** | ✅ done |
| 7 | euk_22k (real AA) | 174073365[7] | — | — | — | — | 🔴 **VOID** (nsys exit=124 timeout, 7.9G rep, no wall) |

*(cP = `computePartialLikelihoodSIMD`; ⟨4⟩ DNA / ⟨20⟩ AA. kmp_wait = `kmp_flag_64::wait` OMP idle-spin.)*

### ⭐ Cross-cell pattern (all 7 live cells — the anti-tunnel-vision read)
1. **`computePartialLikelihoodSIMD` is the universal host residual and GROWS with alignment size in BOTH data types:**
   DNA 10k ~0 → 100k 8.1% → 1m 15.0% → **avian(real) 22.8%**; AA 10k 16.0% → 100k 30.7% → **1m 40.4%**. Real data
   (avian GTR+F+R6, GPU-engaged via the graduated +R ladder — `nTile=1 ncat=6`) confirms the DNA residual is not a
   synthetic artifact.
2. **`kmp_flag::wait` (OMP idle-spin) DOMINATES small cells then FADES at scale:** dna_10k 57.3% / aa_10k 47.8% →
   aa_1m 8.4%. Small alignments = launch-latency-bound (tiny GPU work, threads idle-spin between syncs); at 1M real
   compute dominates and the spin recedes. ⇒ the "sync storm / idle-spin" framing is a **small-N property**, not the
   1M-scale story — at 1M the residual is real CPU partial-likelihood, not spin. (Corrects any read that generalised
   the 10k/100k idle-spin to the 1M headline.)
3. **AA cP > DNA cP at every matched size** (aa_1m 40% vs dna_1m 15%). But AA is the space we WIN (memory
   `project_mf_noctf_offload`: AA MF ~1.62× ahead) — AA's cP is the postorder likelihood (AA's own L0/postorder lever),
   a DIFFERENT call-path from DNA's branch-length **derivative** cP. Do not conflate the two levers.

---

## HEADLINE (from the 4 completed cells) — the DNA vs AA split is the whole story

The **synchronisation** diagnosis (flagged up-front by the user, repeatedly) is **confirmed**: full `-m MFP` is
**synchronisation-dominated on the device**. `cudaDeviceSynchronize` is the top CUDA-API cost in *every* cell
(2–4 M calls; **42% of API on DNA → 77–86% on AA**).

**Why this is genuine sync-latency WASTE, not "CPU correctly waiting on a busy GPU":** GPU utilisation is only
**13–46%** — the device is **idle 54–87% of the time**. So the millions of `cudaDeviceSynchronize` calls are
blocking on round-trip *latency* between tiny launches, not on saturated compute. Both sides stall in lockstep:
the host spins in `kmp_flag_64::wait` (OMP idle) while the GPU sits idle between launches. This is a latency-bound
ping-pong, and it is the mechanism behind DNA losing the head-to-head. On the host the residual is either **OMP
idle-spin (DNA)** or **real CPU-side partial-likelihood (AA)** — see below.

### DNA (4-state) — sync-latency-bound; host is mostly IDLE-SPIN
| metric | dna_10k | dna_100k |
|---|---|---|
| GPU util | 29% | 13% |
| `cudaDeviceSynchronize` (% of API) | **42.4%** (4.37M calls) | (same shape) |
| `cudaMemcpyToSymbol` | 23.8% (5.3M) | — |
| osrt `pthread_cond_wait` | 89.2% | — |
| host `kmp_flag_64::wait` (OMP spin) | 57.3% | (spin-dominated) |
| host `__kmp_hardware_timestamp` | 15.0% | — |
| host **real compute** (`computePartialLikelihoodSIMD<…,4,…>`) | **1.3%** | ~ |
| host `gpu_screen_nni_tile_crosscheck` lambda | 1.0% | — |

**Read:** ~**72% of DNA host "CPU" is OpenMP threads busy-waiting for the GPU** (`kmp_flag::wait` + timestamp), not
compute. Tiny 4-state kernels, **millions of `cudaDeviceSynchronize`**, host spinning between them. This is a
launch/sync-**latency**-bound workload — the signature behind DNA losing the head-to-head. Batching would just
multiplex the same host bottleneck.

### AA (20-state) — REAL CPU-side residual that GROWS with size
| metric | aa_10k | aa_100k |
|---|---|---|
| GPU util | 40% | 46% |
| `cudaDeviceSynchronize` (% of API) | **77.1%** (2.34M calls) | **86.0%** (2.27M calls) |
| `cudaMemcpyToSymbol` | 10.2% | 6.2% |
| host `computePartialLikelihoodSIMD<…,20,…>` (**real CPU compute**) | **16.0%** | **30.7% (⇐ #1 host symbol)** |
| host `kmp_flag_64::wait` (OMP spin) | 47.8% | 27.3% |
| host `__kmp_hardware_timestamp` | 12.6% | 7.0% |
| host `gpu_screen_nni_tile_crosscheck` lambda | — | 1.7% |

**Read — the actionable finding:** unlike DNA, AA has a **named, growing residual host compute**:
`PhyloTree::computePartialLikelihoodSIMD<Vec4d, false, 20, true, false>` (AVX2 double, 20-state partial likelihood)
**runs on the CPU** and rises **16% → 31% of host time as sites grow 10K → 100K**, overtaking OMP-spin to become
the #1 host consumer at 100K. So a slice of the AA partial-likelihood is **not offloaded** — it's computed
host-side. GPU util is higher than DNA (40–46%) because the 20-state kernels are bigger (more work between the same
~2.3 M syncs), but the host is doing meaningful likelihood work in parallel with the GPU.

---

## ⚠️ SCALE-DEPENDENCE — the FULL 6-cell pattern (and my refuted 2-point extrapolation)

🔴 **RETRACTION (be explicit, per the retrospective's discipline):** an earlier version of this section fit a
2-point trend (dna_10k→dna_100k: GPU-busy 29%→21%, toSymbol 5.24M→2.24M) and extrapolated "sync storm shrinks with
scale ⇒ DNA-1M is host-bound/GPU-idle." **The DNA-1M cells REFUTE it.** That was exactly the "reason about DNA from a
partial frame" trap the retrospective warns against — committed here, caught by the actual 1M measurement.

The real pattern across ALL six completed cells (every number from the nsys reps + perf profiles):

| cell | wall | GPU kern-time | **GPU busy** | `cudaMemcpyToSymbol` | host `computePartialLikSIMD<4/20>` | host `kmp_flag::wait` (OMP-spin) |
|---|---|---|---|---|---|---|
| dna_10k  | 148 s | 43.6 s | **29%** | 5.24M | 1.3% | 57% |
| dna_100k | 154 s | 32.2 s | **21%** | 2.24M | 8.1% | 52% |
| **dna_1m**   | **2364 s** | **1236 s** | **52%** | **10.5M** | **22.3%** | **15%** |
| **avian_1m (real DNA)** | **1216 s** | **799 s** | **66%** | **7.4M** | **22.8%** | **21%** |
| aa_10k   | 231 s | 146.7 s | **64%** | 2.62M | 16.0% | 48% |
| aa_100k  | 546 s | 364.4 s | **67%** | 2.51M | 30.7% | 27% |

**What the DNA-1M cells actually say (synthetic AND real avian agree):**
- **GPU is busy 52–66%, NOT idle.** GPU-busy is not monotone in size (29%→21%→**52%**): at 1M the 4-state kernels
  are finally large enough to dominate. So DNA-1M full MFP is **not** "host-bound/GPU-idle" — it is a **GPU/host
  BALANCE**, roughly half device compute.
- **The toSymbol count is NOT monotone** (5.24M→2.24M→**10.5M**; real avian 7.4M) — it tracks *models × LM-iters ×
  edges* per dataset (convergence dynamics), not site count. "Shrinks with scale" was wrong.
- **The one MONOTONE trend is the host CPU likelihood:** `computePartialLikelihoodSIMD<4>` 1.3%→8.1%→**22.3%**
  (real avian **22.8%**) — at 1M it is the **#1 host self-time symbol**, consistent synthetic + real. OMP-spin falls
  (57%→15–21%) as the host does more real work.
- **The sync/toSymbol storm OVERLAPS the 52%-busy GPU** at 1M, so it is **not** the dominant waste there (unlike
  dna_10k). The GPU-driving thread's syncs are largely hiding real kernel work.

⚠️ **The 22% `computePartialLikelihoodSIMD` is a RED HERRING (see FINAL DIAGNOSIS below) — it is the overlapped JOLT
self-check (+G) / the L0 `ncat>4` decline (+R), not the wall driver.** Reconciliation with the retrospective's DNA-1M
nsys (`173531152`: GPU ~93–97% idle) still holds *for tree-search-proper* (fixed model, no MF): the DNA-TS phase of
THIS full-MFP run is likewise GPU ~85% idle. But full MFP adds a GPU-heavy ModelFinder (70% busy, serialized), which
is why the *whole-run* GPU-busy is 52% not ~5%. Both true; the levers are per-phase (see below).

## WHERE the storm is (source-grounded, Plan-agent map — verified against the merge tree that built f3f7875f)
The per-edge storm is emitted by ONE loop: `gpu_jolt_optimize` (`tree/gpu/gpu_lnl_intree.cu:2626`), whose
`computeGradient`→`proc` DFS visits every edge and per edge does **4× `cudaMemcpyToSymbol`** (`setVal` 3×`g_val0/1/2`
`:2881` + `g_rscale` `:3091`) + **3× `cudaDeviceSynchronize`** (`:3051/:3091/:3092`) + 3 launches + 1 D2H
(`reduceDerv` `:2888`). Arithmetic closes on the aggregates: `4·E_g ≈ 5.3M toSymbol`, `3·E_g+… ≈ 4.37M sync`,
`kj_derv_fused` 1.317M ≈ the derv count. **Both storm-mitigations are dead by default in f3f7875f:**
- `cc_skip_toSymbol`/`CC_TOSYM` guard: env `JOLT_CONSTCACHE` unset ⇒ returns false (`:56-57`); even if set,
  `omp_in_parallel()` is true under the MF `#pragma omp parallel` (`phylotesting.cpp:4559`) ⇒ bypassed (`:58`); and it
  only guards `g_Uinv/g_U/g_UinvRowSum` (one-time per model), NOT the branch-dependent `g_val*/g_rscale` storm.
- Packed-async ON-path: `static constexpr bool g_ts_async = false;` (`:873`) with no assignment and no getenv ⇒
  compile-time dead; every gated site takes the legacy per-edge branch. The console's own
  `[TS-ASYNC] … valpool_async=0` (summing to 5,244,471 ≈ the 5.31M nsys toSymbol) is the runtime proof it is OFF.

## ⭐ FINAL DIAGNOSIS (red-team confirmed against source + console — the second hypothesis I had to retract)

🔴 **RETRACTION #2: "the DNA lever is offloading the CPU `computePartialLikelihoodSIMD` postorder" is REFUTED.**
That 22% is a **red herring** — overlapped CPU shadow-work, not the critical path. Verified from source + console:

1. **What `computePartialLikelihoodSIMD` actually IS in DNA MF = the per-candidate JOLT SELF-CHECK, not declining
   models.** Console (`JOLT_DEBUG=1`): **0 declines, 0 write-back MISMATCH, 0 NaN→CPU-fallback, 121 JOLT successes** —
   every eligible candidate engages JOLT and succeeds (this binary graduated +I/+R/free-Q). The MF `computePartial`
   is the fresh CPU postorder verifying JOLT's write-back (`phylotreegpu.cpp:2703`; dispatch returns JOLT's result
   without the CPU path, `modelfactory.cpp:1613`, and the `:1624` CPU fallback is never reached). ~6.2 CPU-s/self-check
   × 121 ≈ 746 CPU-s. The source's OWN comment already said this (`:2783-2786`): even removed entirely (free
   self-check), MF **1300.9s still LOSES** to Hashara's 1142.4s. So it cannot be the lever.
2. **The AA paradox DISSOLVES.** AA's *higher* `computePartial%` (30.7%) is simply more overlapped shadow-work: the
   self-check runs OUTSIDE `jolt_gpu_mtx` and overlaps the *next* candidate's serialized GPU-optimize. `computePartial%`
   measures CPU work hidden in the GPU's shadow, **not** the wall. The real win/lose split is **tree-search GPU
   utilization**: AA TS ~73% GPU-busy (WINS), DNA TS ~15% GPU-busy / **85% IDLE** (LOSES).
3. **The +R (avian) TS `computePartial` = the L0 `ncat≤4` cliff** (`phylotreegpu.cpp:2831` `L0_FREERATE_MAXCAT=4`,
   `:2834` declines `ncat>4`): avian `GTR+F+R6` (ncat=6) → L0 returns NaN → CPU `computeLogL`. `+G4` (dna_1m, ncat=4)
   → L0 engages on GPU (`computePartial`≈0, `computeLikelihoodGPUResident` 12.7% instead). A genuine but **NARROW**
   lever (high-K `+R`/`+I+R` only), NOT the DNA-1M `+G4` headline.

### The two named DNA-1M levers — 🔴 RED-TEAM AUDIT VERDICT: BOTH DEAD AS FRAMED (job-a2fa688f, disk-verified this pass)
The audit was tasked with **feasibility**, not just attribution — because a correctly-diagnosed lever that can't be
built is still a dead end. Every load-bearing citation below was re-verified against the source bytes this turn.

- **ModelFinder — `jolt_gpu_mtx` serialization: DIAGNOSIS REAL, FIX (batching) UNBUILDABLE, reclaimable ≈ 0.**
  - *Diagnosis reproduces exactly:* the `static std::mutex jolt_gpu_mtx` `lock_guard` (`gpu_lnl_intree.cu:2649`) is held
    to the function's **sole `return` at `:3516`** — the ENTIRE GPU body (upload + all kernels + +R FD path + D2H
    write-back) serializes. MF CPU 2618.2s / MF wall 1596.8s = **1.640×** parallelism; GPU-busy **69.8%** (1130s/1618s);
    idle **≈488 s**. All confirmed.
  - *The fix is not buildable as framed.* The per-model eigensystem lives in **process-global `__constant__`**
    (`g_Uinv/g_U/g_val0/1/2/g_rscale`, `:35-44`) and the partial pools are process-global static `DevBuf`s — two
    candidates with different Q **cannot** be in flight; they clobber one `__constant__` bank. The source comment
    (`:2642-2647`) states this is deliberate and defers batching to "PHALANX grid.z, **not this**." And **grid.z=K
    batching was already spiked NO-GO** (1.22×, spike 172388123, memory `project_gpu_tree_search`): one DNA-1M candidate
    already launches **gridX≈3654 blocks vs 132 SMs = ~27.7× oversubscribed** (nsys: gridZ=1 for every kernel), so a
    second candidate just queues. Batching MODELS is strictly worse than the TREES that NO-GO was measured on (needs K
    independent eigensystems + K× the ~2.7 GB arena).
  - *The 478 s idle is NOT GPU-reclaimable.* It is CPU-bound gaps — 10+ of 12 threads doing CPU self-checks /
    CPU-fallback candidates while `kmp_flag_64::wait` (15.25%) spins on the GPU mutex. GPU concurrency cannot fill a gap
    that has no GPU work. The self-check does **not** steal GPU bandwidth (it is a host postorder, `phylotreegpu.cpp:2703`,
    outside the lock). **Reclaimable ≈ 0.**
- **Tree search — discarded-screener: DEFAULT + 85%-idle both REAL, but MIS-FRAMED (not free headroom).**
  - *Default, not a profiling flag, and NOT a wrong-binary artifact.* `ts_screen_topk=0` is the default in **both** trees
    (`tools.cpp:7865` in profiled `f3f7875f` AND in production `9d845205`). The profiled binary auto-enables the fused
    screener (`main.cpp:2298-2302`) but leaves `topk=0` ⇒ discarded-validator mode. **Production `9d845205`'s `main.cpp`
    does NOT auto-enable the screener at all** (grep empty) — so production never drives NNI from the GPU ranking either;
    a plain `--gpu` prod DNA run wouldn't even run the screener. The "GPU-drives" (Step 2) mode is opt-in everywhere.
    ⇒ **the config caveat is RESOLVED: DNA-TS GPU-idle is a genuine property of the default GPU path, not a wrong-binary
    artifact.** (nsys TS window: ~99s busy / 696s wall = **14.2% busy → 85.8% idle**; `screener_wall_s 358.8` with
    `branches_checked 0` — the validator validated nothing.)
  - *"Reclaiming" it is an unvalidated trajectory change, not an offload.* Using the idle GPU means flipping
    `ts_screen_topk>0`, which **alters the NNI search trajectory** and is research-gated on recall≥0.95 / final-lnL
    within 0.5 (`iqtree.cpp:3109`) — unvalidated for DNA. The cheap alternative (`--no-ts-fused`, stop the discarded
    358.8s validator) saves host wall but leaves the GPU ~100% idle, because **DNA-TS is host-bound regardless** — an
    already-characterized problem (memory `project_reopt_rearchitecture`: DNA-1M TS = 87-91% serial host compute).

### Where this leaves DNA — and what `dnares` (job 174003961) NAMED (disk-read 2026-07-17)
Both audited levers are closed. The residual DNA-MF headroom is **CPU-side**, not a GPU-offload — and the one CPU-side
move the audit surfaced (turn on `JOLT_MF_DEVUSE` to delete the CPU self-check postorders) is **refuted as the DNA
closer** on two counts now: `dnares` measures the self-check at **only 11%** at nt12 (161s of 1465s, even smaller than
the memory's ~25%), and even FREE (1300.9s) still loses to Hashara's 1142.4s.

**`dnares` (mfresident `020ff472`, `-m MF -seed 1 -ninit 2 -nt 12 --jolt --gpu`, self-check OFF via
`JOLT_MF_NOSELFCHECK=1`, wall 1304.1s) NAMES the residual for the first time:** the top host symbol is
`computePartialLikelihoodSIMD<Vec4d,false,4,…>` at **15.18% self-time**, driven *entirely* by the CPU **derivative**
path (`computeLikelihoodDervSIMD` 9.66%, via `computeFuncDerv`) + the CPU **branch-length** path
(`computeLikelihoodBranchSIMD` 5.47%). Everything below it is diffuse (driver/OMP-spin/idle) ⇒ DNA-1M `-m MF` at nt12
is **~15% one named CPU cluster over a mostly host-ceiling wall.**
✅ **RESOLVED — job 174078758 (`dnaconfirm`, disk-read 2026-07-17): H2 MIGRATION confirmed, H1 COVERAGE refuted.**
Single-binary (dnares's exact `020ff472`), single-variable A/B, both `JOLT_DEBUG=1 JOLT_MF_NOSELFCHECK=1` + perf `-F 499`:
- **Arm A `-m MF`** (full set): MF-wall **1304.45s** (reproduces dnares 1304.12s ✅), best `F81+F+G4`, **114 hooks /
  1 decline** (`pure-pinvar-no-gamma`) / 113 engaged. `computePartial+Derv+Branch` cluster = **16.08%** (`computePartial`
  14.95% alone ≈ dnares 15.18% ✅), flowing `computePartial ← computeLikelihoodBufferSIMD ← computeLikelihoodDervSIMD`.
- **Arm B `-m MF -mrate G`** (`+G`-only): MF-wall 129.3s, **same winner** `F81+F+G4`, **36 hooks / 0 declines** (every
  candidate engaged JOLT). Cluster **23.01%** — it did **NOT collapse; it grew** — via `dotProductTriple ←
  computeLikelihoodDervSIMD` (the branch-length **derivative**).
- **Pre-committed rule fired, but ⇒ H2 is CONFOUNDED (caught 2026-07-17, anti-tunnel-vision):** the cluster PERSISTS on
  the 0-decline `+G`-only set — which rules out **coverage** (H1) cleanly: `+R`/`+I+R`/`+I+G` all ENGAGE JOLT (live
  markers: `ncat=6`=`+R6` reached the hook AND `JOLT-TILE` fired 3×; 113/114 engaged; 1 pure-`+I` decline in the whole
  run). **BUT "persists ⇒ per-candidate migration" does NOT follow.** The one-time **"fast ML tree search using GTR+I+G"
  is a FIXED ~64s in BOTH arms** (full 64.4s = 4.9% of wall; `+G`-only 63.7s = **49%** of the 129s wall). That fast
  search runs `optimizeModelParameters` (GPU) **then `doNNISearch(true)`** (`phylotesting.cpp:807/814`) — and
  `doNNISearch` on DNA is the KNOWN **host-bound tree search** (GPU screener idle, CPU drives NNI branch reopt =
  `computeLikelihoodDervSIMD`). So the `+G`-only cluster is inflated by that fixed initial NNI, not proof of a
  per-candidate lever. A 2-point split can't separate them (RETRACTION #1's mistake), and the fp caller stacks came back
  topologically scrambled. **Control job 174080780 (`dnanni`, `gems_dna_nni_isolate.sh`) SETTLES it:** baseline
  (`-m MF -starttree PARS`, includes `doNNISearch`) vs fixedtree (`-m MF -t <MLE tree>`, `start_tree=USER` ⇒ **skips**
  `doNNISearch` per `:812`, per-candidate branch-opt intact). fixedtree cluster COLLAPSES ⇒ the derivative was the
  initial NNI (tree-search lever, = `project_reopt_rearchitecture`); PERSISTS ⇒ real per-candidate branch-LM (H2).
- ✅ **`dnanni` RESULT — H2 VALIDATED, co-existing with a 39% tree-search component (disk-read 2026-07-18):** absolute
  CPU-s (MF-CPU × cluster%): baseline **556.8** (22.60% × 2463.8s, fast-search 65.6s) → fixedtree **341.3** (17.44% ×
  1957.2s, fast-search 28.9s). **~39% (~215 CPU-s) = initial `doNNISearch`** (fixedtree `children%` shows it → 0.00% ✅
  isolation worked) = the host-bound DNA tree search. **~61% (~341 CPU-s) PERSISTS** with the tree fixed AND every
  candidate engaging JOLT (119 hooks / 1 decline; the initial fit **GTR+F+I+G4** freqtype=3 ndim=5 ENGAGES, so the
  persist is NOT a declined-fit CPU fallback).
- ✅ **RESOLVED (dnadwarf 174084797 + brlenprobe 174086555, disk-read 2026-07-18) — the persisting per-candidate
  derivative is NOT in `optimizeAllBranches`; the DNA-MF wall is GPU-serialization-bound, NOT this CPU cluster; the
  CPU-derivative chase is EXHAUSTED (3rd independent confirmation the DNA-1M `-m MF` loss is ARCHITECTURAL).**
  - Two prior mechanism hypotheses stay FALSIFIED (kept compact): *"inside the engaged JOLT path"* — FALSE
    (`optimizeParametersJOLT` has no `computeLikelihoodDerv` before its `:2539` NOSELFCHECK return; `gpu_jolt_optimize` is
    `extern "C"`, its only host callback = eigendecomp); *"CPU-fallback of declined models"* — FALSE (disk: exactly 1
    decline, `pure-pinvar-no-gamma`; the graduated ladders engage `+I+G`/`+I+R`/`+R`). The fp children% "under
    optimizeParametersJOLT" was call-graph misattribution.
  - *dnadwarf did NOT de-scramble the caller* (as feared): even `--call-graph dwarf` roots `computeLikelihoodDervSIMD` at
    `__kmp_GOMP_microtask_wrapper` — the OMP fork orphans the worker stacks; top-down children% captures only the SERIAL
    main-thread slice (`doNNISearch` 2.55%, `optimizeAllBranches` 1.28%), under-counting the parallel derivative beneath.
    Perf (even dwarf) cannot name the phase — a COMPILED counter can.
  - *brlenprobe (a `JOLT_BRLEN_PROBE`-gated RAII wall-timer compiled INTO `optimizeAllBranches`, tagged by phase, sees
    through the OMP fork) SETTLES it.* DNA-1M `-m MF`, both arms (`JOLT_MF_NOSELFCHECK=1 JOLT_MF_RESIDENT=1`, nt12):
    - baseline (`-starttree PARS`): OTHER(init/final)=17.8s | **NNI(doNNISearch)=15.4s/5calls** | **MF(candidate-eval)=0.000s / 0 CALLS**. wall 1294.8s.
    - fixedtree (`-t <MLE>`, skips doNNISearch): OTHER=17.9s | **NNI=0** (isolation clean) | **MF=0.000s/0calls**. wall 1222.0s.
  - **DECISIVE READ (pre-registered red-team #1 FIRED):** the `MF(candidate-eval)` bucket is **0 CALLS in BOTH arms** ⇒ the
    per-candidate model evaluation NEVER calls `optimizeAllBranches` (confirms `modelfactory.cpp:1613` returns JOLT's result
    directly, no CPU branch-polish). So dnanni's persisting ~341 CPU-s derivative enters via a **non-`optimizeAllBranches`
    path** — still unnamed, but now proven OFF this function (so any tip-vec / `optimizeAllBranches` framing can't touch it).
  - **Cross-check (red-team #2 ✓):** NNI bucket 15.4s WALL × ~12 threads ≈ dnanni's 215 CPU-s doNNISearch — two independent
    instruments agree.
  - **Wall-criticality (red-team #5):** removing doNNISearch (the MORE critical-path, GPU-idle initial search) drops the MF
    wall by only **73s (1294.8→1222.0 = 5.6%)**. The wall is dominated by the GPU-serialized candidate loop (`jolt_gpu_mtx`,
    69.8% GPU-busy); the CPU derivative is OMP-parallel shadow overlapping it. ⇒ **the CPU-derivative lever cannot close the
    ~160s Hashara gap. The forward DNA move is GPU-RESIDENT ModelFinder (tip/echild/partials on-device across candidates,
    drop the per-candidate mutex serialization), NOT another CPU-side micro-lever.**
  - 🔴 **HONEST CAVEAT:** brlenprobe's `reached-hook=0/declines=0` AND its 0 `JOLT-DIAG-CU` lines are FALSE ZEROS — I set
    `JOLT_BRLEN_PROBE=1` but not `JOLT_DEBUG=1`, so no `[JOLT-GATE]`/`JOLT-DIAG-CU` markers printed (the "assert on a marker
    whose switch you didn't set" trap; the review agent caught the identical omission in ovlevers). Engagement is
    corroborated by GPU-util **90%** + best=**F81+F+G4** both arms + wall≈known 1300s + dnanni's `JOLT_DEBUG=1` (119
    hooks/1 decline). The `optimizeAllBranches` buckets are `JOLT_BRLEN_PROBE`-gated, independent of the omission — VALID.
- 🔴 **GATE MIS-CITATION CORRECTED (THE MAINSTAY — running markers beat my source read):** I earlier cited
  `phylotesting.cpp:1337 ctfIneligible` (`"+R" OR "+I"-without-"+G"`) as the operative MF gate. It is **not** — it is the
  **CTF selection** gate. The non-CTF MF path (this run, no `--ctf`) has the **graduated +R ladder**, so `+R` engages the
  GPU. The `+R6`-engages marker overrules the source rule. (The `m4census` `eed14b92`@100K "0 declines" prior was
  directionally right for a different reason: near-zero declines because the families engage, not because a `+I` closure
  removed them.)
- **Cross-cutting:** `kmp_flag_64::wait` OMP idle-spin is large in BOTH arms (14.6% full / 26.2% `+G`-only) — threads
  idling while the serial CPU branch work (and the `jolt_gpu_mtx` GPU serialization) runs. Some of that spin is
  downstream of the un-migrated branch step; how much is reclaimable is a Phase-1 measurement, not an assumption.

AA is untouched and still wins; the depth-axis kernel lever (tip-vec, Track A of the standing plan) is an **AA-side**
improvement — a GPU-kernel speedup cannot help a host-bound DNA phase.

### Track A/B overnight levers — ovlevers 174087249 (disk-read 2026-07-18)
- **Track A / depth (tip-vec Phase-0 step 1 — echild-rebuild tax):** the per-eval HOST echild tax is **NEGLIGIBLE** —
  DNA-100K mean 0.003s/call (Σ0.009s over 3 calls, 210 rebuilds); AA-100K mean 0.006s/call (Σ0.012s over 2 calls, timed
  out early). ⇒ the plan's "one unmeasured risk" (that adding `make_tabP` host-side work would compound the echild tax) is
  REFUTED — the tax is milliseconds. **`make_tabP` host-vs-device placement is de-risked** (both fine on the tax axis).
  This does NOT validate tip-vec itself; the make-or-break Phase-0 gate is still the compile-only register/spill check
  (`tabp_lever3_probe.cu`, `nvcc -Xptxas -v`, bar: NS=20 ≤128 reg / 0 spill vs the shipped `k1_node_t<20>`). NEXT depth step.
- **④ AA-zigzag nRej:** INCONCLUSIVE — AA timed out at 2 optimize calls (nRej=0, unrepresentative); DNA nRej=48/187 lnL
  evals ≈ **26%** (memory's "DNA reject_frac 0.07-0.21" was a touch low; same ballpark). The AA≫DNA prediction is UNTESTED
  here (need AA data with an iteration cap so the AA search doesn't hit the 1800s per-run timeout).
- **Track B / breadth (`-pers`×`-nbest` convergence sweep, GPU wide-beam gate):** UNDER-POWERED / NEUTRAL. All four
  DNA-100K cells (base 0.5/5, 0.1/20, 0.15/10, 0.05/30) reached the **IDENTICAL** optimum −5692972.985 in the **IDENTICAL
  102 iterations** — each finds ONE improvement early ("BETTER TREE FOUND at iteration 1") then grinds 100 unsuccessful
  iters to the nstop floor. So small-perturbation/wide-pool **doesn't regress**, but the dataset is too easy to
  discriminate whether wide-beam HELPS (the parsimony start is already ~optimal). AA cells VOID (both timed out; AA-100K
  tree search needs ~1.5h — 1800s cap too tight). ⇒ the breadth gate needs a HARD search landscape + capped/downsized AA
  to actually decide; on easy data the lever is a no-op. Bank as an honest under-powered negative (breadth was always the
  speculative half of the bet).

⚠️ **`aa_1m` is NOT dead** (earlier "VOIDed on walltime" was wrong): its MF finished (best model LG+G4, MF CPU 5283s)
and it is in the tree-search perf pass. The AA paradox is resolved on `aa_100k`; a clean `aa_1m` would close it at
matched 1M size.

---

## ✅ Scale + real-data test — ANSWERED (cells landed 2026-07-17)
- **Does AA `cP⟨20⟩` keep growing past 100K?** YES — aa_1m **40.4%** (100k was 30.7%). AA's postorder residual is the
  biggest single host symbol anywhere in the campaign; it's the AA lever (L0/postorder), and AA still WINS on wall.
- **Does the DNA sync-storm / OMP-spin ratio hold at 1M / on real data?** NO — the OMP-spin story is a **small-N**
  property (dna_10k 57% → dna_1m ~15%); at 1M the DNA residual is real `cP⟨4⟩` (15%), and **avian (real DNA R6)
  confirms it at 22.8%** with the +R ladder engaging the GPU. So the "sync storm" is real at 10k/100k but is NOT the
  1M-scale lever — that framing must not be generalised to the headline (it was, early; corrected above).
- **euk_22k VOID** — the real-AA 100×22462 cell's nsys pass hit the 8400s guard (exit=124, 7.9G rep) and produced no
  usable wall/host data. Re-run needs a longer nsys guard or perf-only (skip nsys) for the big real-AA case.

---

## Method-notes / red-team trail (why the mechanics are trustworthy)
- **Orphan BLOCKER (independent red-team, verified on-node):** `nsys --duration … --kill none` STOPS+RETURNS while
  the app runs on as an orphan ⇒ false VOID + PASS2 racing the orphan on one GPU. Fixed by **whole-run `--wait=all`,
  no `--duration`**; rep bounded by dropping osrt on 1M cells (nsys-rep is **event-count-bound, site-INDEPENDENT** —
  dna_100k rep 361 MB < dna_10k 780 MB, so 1M stays in the 100s-of-MB range).
- **`readelf` gate killed canary #1** in 9 s: `readelf` is absent from the compute-node base image and gives
  false-negative symbol checks (column truncation, per the red-team). Switched to `nm`, moved after `module load`,
  made **non-fatal** (perf reads symbols itself; a tool quirk must not kill a valid run). `nm` is *also* flaky on
  the compute image — the real proof is perf's output naming `kmp_flag_64::wait`.
- **perf self-vs-cumulative:** default `perf report` with `-g` sorts by Children% ⇒ the "top symbols" would be
  `main()`/roots. Fixed with `--no-children` + `-e task-clock` (SW event, no PMU/paranoid gating) + no `-g`.
- **VOID hardening:** a cell is LIVE only with a real wall AND a non-empty `cuda_gpu_kern_sum` block AND GPU-engage
  markers — a silent CPU fallback or corrupt rep cannot print a clean DONE.
