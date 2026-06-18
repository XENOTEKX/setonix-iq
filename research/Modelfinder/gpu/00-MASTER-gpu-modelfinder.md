# GPU ModelFinder for IQ-TREE 3 — MASTER REFERENCE (condensed)

**Purpose.** ONE self-contained reference that condenses the 13 `gpu-modelfinder-*.md` docs (5,224 lines) in this folder
into the load-bearing facts, numbers, verdicts, and open work. Written 2026-06-17 (author as1708) for fast future
reference; the source docs remain here for full detail (index in §9). **Every job ID / commit / lnL / rel below is quoted
from the source docs and cross-checked against `setonix-iq/CHANGELOG.md`.** When in doubt, the dated source doc wins.

---

## 1. Orientation — what this project is

**Goal.** Run IQ-TREE 3 **ModelFinder** (and per-tree ML optimisation) on **one GPU** instead of an MPI CPU cluster, by
replacing IQ-TREE's CPU-shaped *sequential* per-edge Newton branch optimiser with a GPU-shaped **joint, parallel,
linear-time-gradient** optimiser — **JOLT** — and wrapping ModelFinder in a **coarse-to-fine (CTF)** pipeline.

**The two inventions:**
- **JOLT** = joint Levenberg–Marquardt over *all* branch lengths + gamma α (+ free-Q, + later +R weights / mixture weights)
  using the **Ji-2020 linear-time all-branch gradient** (1 postorder + 1 preorder sweep), optimised jointly with a
  damped second-order step. Reaches the **same MLE** as IQ-TREE (correctness = same optimum, not same trajectory).
- **CTF** = rank the whole candidate set on a **5,000-site subsample** (native-BIC gate + rate-het detector) → **refine only
  the top-k ≤ 3** on full data. Turns a breadth problem into a depth problem.

**Live code (authoritative):**
- Dev tree: `/scratch/rc29/as1708/iqtree3-gpu/` branch **`gpu-kernel`**.
- **Published 2026-06-18** to GitHub **`XENOTEKX/setonix-iq` branch `gpu-kernel` @ `a972cefc`** (the full 1070-file fork).
  ⚠ History was rewritten before this public push to strip `Co-Authored-By` trailers (author `as1708` unchanged; source
  tree **byte-identical**, same tree-hash `f683c3b1`) — so **all commit hashes in these docs are the post-rewrite
  (published) hashes**. The pre-rewrite history is retained locally on branch `gpu-kernel-backup @ 6d7f7483`.
- Production binary: `/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3` (md5 `8dd57cfb`) — fat binary
  **sm_70 + sm_80 + sm_90** (V100/A100/H200), CPU path = upstream-identical (no `--jolt`/`--gpu` ⇒ stock IQ-TREE).
  **Full binary registry (live vs frozen, which script pins which): [`GPU-BINARIES.md`](GPU-BINARIES.md).**
- Kernels: `tree/gpu/gpu_lnl_intree.cu`; GPU driver `tree/phylotreegpu.cpp`; eligibility gate `phylotreegpu.cpp` (see §6).
- The full GPU source is published on GitHub (`gpu-kernel`); the old stale `gadi-ci/gpu-modelfinder/src/...` backup was
  removed 2026-06-18 — edit the live tree at `/scratch/rc29/as1708/iqtree3-gpu/` only.
- Activate: `--jolt --gpu`. Env escape hatches: `JOLT_NO_FREEQ` (disable free-Q DNA), `JOLT_DEBUG=1` (gate decisions),
  `JOLT_QGRADCHECK` / `JOLT_RGRADCHECK` (gated FD cross-checks).

**Gadi compute.** `gpuvolta` (V100-32 GB, sm_70, dev card), `dgxa100` (A100), `gpuhopper` (H200-141 GB, sm_90). The
**MPI fork** `iqtree3-mpi` is `-march=sapphirerapids` → **SIGILLs on non-SPR CPUs** (gpuvolta/login); use `build-gpu-on/iqtree3`
on GPU nodes (runs on any CPU, no mpirun).

---

## 2. How JOLT works (the kernel + optimiser pipeline)

**Likelihood (postorder), eigen-space.** `k1_node` computes the Felsenstein partial likelihoods in the **eigenbasis** of Q
(transform once, prune with diagonal `exp(λt)`, transform back), looping `for c < ncat` (gamma categories share **one**
eigendecomposition with scaled eigenvalues; eigenvectors live in `__constant__ g_U`). NORM_LH (unscaled) for ≤2000 taxa.

**Gradient (Ji-2020 linear time).** `kj_pre` is the top-down **preorder** "rest-of-tree" partial; combined with the postorder
partials it yields **all 197 branch df/ddf from one postorder + one preorder sweep** (vs IQ-TREE's 197-deep *sequential*
Gauss-Seidel per-edge Newton). `kj_derv_fused` FUSES theta+lnL+df+ddf (+ `rnum` rate numerator, + `wnum` per-category
likelihood) in registers — `d_theta` (601 MB) never materialised. `kj_ratenum`/`kj_reduce_gradnum`/`kj_invl`/`kj_reduce3`
are the on-device deterministic FP64 reductions (no `atomicAdd`).

**Optimiser.** Joint diagonal-LM: `b_e += df_e/(|ddf_e|+μ)` for all branches at once; α rides the same step (analytic
`∂lnL/∂α` via the validated rate gradient); free-Q rides via **forward-FD** per exchangeability (re-decompose 4×4 + re-upload
+ rebuild echild per FD step). Accept/reject on `ln > lnL+1e-9` (no line search). Gauge-fix simplex params after each
accepted step.

**Memory recycling.** O(depth) frontier pool: store partials only for the active frontier (= tree height ≈42), not all
nodes (G.4.0b). **Pattern tiling** (G.7.1): split patterns into `nTile` chunks, shrink the arena exactly `nTile×`,
bit-identical to one-shot (auto-`nTile` from `cudaMemGetInfo`).

**The in-tree seam.** `--jolt` installs `optimizeParametersJOLT` at the `ModelFactory::optimizeParameters` hook;
write-back = brlen + α (`setGammaShape`) + `clearAllPartialLH`, then a **fresh CPU `computeLikelihood` self-check** (`[JOLT]`
line). Under `--jolt` the stateless G.2 GPU overrides do NOT install (else ineligible models fall to the *slow* stateless
path); a process-wide `std::mutex` serialises JOLT on the 1 GPU (ModelFinder is across-model OpenMP, `phylotesting.cpp:4097`).

---

## 3. Phase timeline G.0 → G.8 (verdicts + provenance)

| Phase | Verdict (one line) | Job | Commit |
|---|---|---|---|
| **G.0** | BEAGLE lnL bit-parity CPU≡GPU AA-100K, 43–66×; r10split bit-exact 57×; gradient FD-OK on CPU but **BEAGLE-CUDA gradient broken for 20-state** | 169679422, 169680691 | — |
| **G.1.1 (K1)** | Custom eigen-space postorder lnL; r10 ONE-PASS −7554280.5776 (BEAGLE's NCAT≤8 cap gone); 37.8 vs 45 ms beats BEAGLE | 170188276 | — |
| **G.1.2 (K2)** | Single-edge df/ddf FD-validated (df ~2e-9); evalAt 1.1–1.3 ms theta-cached | 170188743 | — |
| **G.1.3 (K3)** | CUDA-graph + on-device echild; graph≡naive bit-identical; **wall PARITY 1.00–1.01×** (structural win only, 104→1 API calls) | 170189528 | — |
| **K4 fusion** | 1.79× @1k but **wash-to-loss at 100K** (real ML tree = height-42 ladder) ⇒ production bound = compute, not launches | 170194367 | — |
| **G.2.0b** | Branch pointer wired in-tree; GPU lnL+s.e. == CPU **bit-identical rel 0.0** (−7541976.8566) | 170205301 | — |
| **G.2.1b** | Full GPU branch-opt rel 0.0 / brlen worst_rel 0.0; **WALL 1063 s vs CPU 225 s = 4.7× SLOWER** (stateless re-sweep — the pivot trigger) | 170259325 | — |
| **G.2.2a** | -m TESTONLY: 6 models/2.5 h (~25 min each), all 6 **bit-identical** to CPU ⇒ selection correct, problem is **throughput** | 170265661 | — |
| **G.4.0 (K7)** | Ji preorder all-branch gradient works on GPU; FD rel ~2.5e-8 | 170279700 | `98b0cd50`/`ebe8cd14` |
| **G.4.0b** | **MAKE-OR-BREAK: +R rate gradient does NOT overflow unscaled** (1/L_ptn 1e92 yet finite; qp/L_p self-cancels O(100)); O(depth) recycling fits r8/r10 | 170281211 | — |
| **G.4.1** | Joint LM **cold→MLE in 27 iters**, cold==warm rel **2.47e-16** (PROVES per-model parallelizes) | 170302036 | — |
| **G.4.1b** | Joint α: full +G MLE (197 branches + α) in 27 iters, **α for free, no Brent** | 170303658 | — |
| **G.4.2a** | In-tree `--jolt` single model; write-back rel 2.77e-12; **WALL 47 s vs 224 s = 4.8× FASTER** (reverses G.2.1b) | 170361630 | `505b52d1` |
| **G.4.2b** | Full -m TESTONLY -nt12; best==LG+G4 (BIC/AIC/AICc); parity worst 1.185e-09; mutex thread-safe, no GPU corruption | 170367630 | `4d801e7e` |
| **Coverage** | 58/62 engage JOLT (94%, incl 28 +F); **GPU util 96%, CPU 2/12 ⇒ GPU-serialization-bound** | 170386010 | — |
| **CTF P0** | Subsample recall **3/3** at 1000/2000/5000 sites → CTF GO | 170396778 | — |
| **CTF P3.0** | **Bandwidth thesis FALSIFIED** (DRAM flat ~33%); latency/occupancy-bound | 170398260 | — |
| **AA-1M CTF** | H200 1994 s → **893 s** after +I 4-start fix = 1.26× vs np16 (1122 s) | 170517590, 170581208 | — |
| **G.5.0** | On-device reduction; 116/116 rel 2.17e-10; util 61→96%; **A100 1355 s beats np8 (1443.9 s)** | 170634239, 170636493 | — |
| **G.5.1a i1** | +R weight gradient `gz_c=WN_c−w_c·N` FD-validated 1.03e-8; ΣWN_c=N exact | 170777660 | `4cb639a7` |
| **G.5.1a i2a** | Standalone +R convergence: **R4 4/4 PASS** (+0.2289 vs CPU-EM); **R6 1/4 CHECK** (multimodal); gate UNFLIPPED | 171516319 | — |
| **G.5.1b boundary** | +R reproducibility cutoff MAPPED: **R3 & R4 PASS** (4/4 starts agree, match/beat CPU-EM, grad WN exact); **R5/R6/R8 CHECK** (1/4–1/8, multimodal — JOLT's *best* beats CPU-EM ~10 nats but not reproducible). **Cutoff = R4↔R5.** Ship R2–R4, decline R5+ to CPU (or EM-warm-start). Gate still UNFLIPPED | 171518800 | — |
| **G.6.0a** | Q-FD gradient cross-check **bit-identical** (maxrel_lnL 0.0) HKY/TNe/TVM/SYM/GTR | 170787044 | `e9498baa` |
| **G.6.0b** | Free-Q FD-LM: **JOLT ≥ CPU all 5** (GTR +0.0029); no stall even GTR 5 coupled rates (13–34 iters) | 170792611 | `afc1c5a1` |
| **G.6.1** | DNA coverage **8→70 engage**; worst write-back rel 6.224e-12, ZERO mismatch; free-Q **ON BY DEFAULT** | 170795329 | `d6d68943` |
| **G.6.2** | DNA-1M -m MF CTF: **WINNER F81+F+G4 == IQ-TREE's own -m MF winner**; **152 s vs 1122–3077 s = 7.4–13×**, 7.53 Wh | 170843136 | — |
| **IX.7 fix** | Native-BIC gate fixes -m MF timeout; H200 **767 s, LG+G4**, 1.46× np16 | 170756438 | — |
| **G.7.0** | cgroup-aware host-memory fix (10M host wall) | 170934922 | `b43d5a97` |
| **G.7.1** | Pattern tiling: chunked==one-shot rel 0.0; **AA-10M on 1 H200** rel 4.465e-13 | 170976732, 170977748 | `a972cefc` |
| **n=30 recall** | Native **30/30** recall@3; projected DNA **0/15** (overfit) | 171258771 | — |
| **BEAGLE bench** | JOLT **2.3–2.43×** over CUDA-core BEAGLE, ≈parity vs FP64 tensor-core; runs AA-1M where BEAGLE client OOMs. **Follow-up lever → part12** | 171265226+ | `960121f8` (local; CHANGELOG §A/B) |
| **FP64-TC lever** | **CLOSED — T.0 kill-switch fired STOP** (part12 §XII.6): DMMA matvec **0.36× @1M H200 / 0.60× A100** = 1.6–2.8× SLOWER than JOLT scalar (parity bit-identical). Gate was ≥1.3×. Confirms JOLT's register-fused scalar already captured the gain; 20→32 pad + latency-bound negate the TC FLOP edge. Cost 0.43 SU vs 6–10 days saved | 171587052/3 | `tc_decider.cu` |
| **G.8.0** | **Profile-mixture lnL on GPU bit-exact**: `k1_node_mix` GPU lnL == CPU for LG+**C20**/**C60**/**MEOW80**+G4 rel **3.06e-16 / 1.54e-16 / 1.56e-16** (320-regime MEOW80 via global-mem per-regime arrays, dodges `__constant__` 64-cat limit) | 171604565 | `2277273d` |
| **G.8.1a** | **Mixture per-class** for EM numerator: GPU self-consistency Σ_m L_{p,m}=L_p **1.4e-14** (×3); posterior γ_{p,m} vs CPU `_pattern_lh_cat` **\|Δγ\|=6.84e-13/4.32e-12/1.00e-12** (C20/C60/MEOW80) — scale-invariant metric (CPU per-pattern scaled, GPU clean-room unscaled) | 171633488 | `2277273d` |
| **G.8.1b** | **Mixture branch DERIVATIVE** df/ddf == CPU `computeLikelihoodDerv` ~machine-eps: INT-INT **df 8.1e-14/4.8e-14/1.1e-13** ddf ~1e-14; LEAF **df 1.5e-14/3.2e-14/1.2e-13** ddf ~1e-15 (C20/C60/MEOW80, both edge types). `k2_derv_mix` regime-axis + per-class global-mem coeffs; π_m absorbed via theta trick. Red-team clean | 171637224/171637348 | `b855c3fe` |
| **G.8 →** | next: G.8.2 EM weight optimiser (G.8.1a posterior + G.8.1b gradient in joint LM) → G.8.3 seam+gate relax → G.8.4 eukaryote LG+MEOW80+G4 payoff (part9 §IX.11); production still declines mixtures to CPU | — | — |

---

## 4. The load-bearing HONEST claims (never overclaim these)

1. **N/S mutex-serialisation ceiling.** GPU is mutex-serialised at per-model speedup **S≈4.8**; CPU runs **N=103** concurrent.
   Aggregate **N/S = 103/4.8 ≈ 21× slower at ANY coverage**. Even batched: A100 g4 B=12×S=4.8=57 < 103 ⇒ full-data 100K
   breadth batching is coin-flip-to-loss. **This is a BREADTH property.**
2. **100K is not a clean GPU-specific win — but that's a REGIME property, NOT a parallelization failure.** JOLT's 27 cold
   iters → same MLE rel 2.5e-16 **proves** the per-model algorithm parallelizes; a 100K loss measures GPU occupancy at
   small N, not parallelizability. (Mode-L's L.1 gate, re-stated as **critical-path length not traversal count**, WON.)
3. **The bandwidth thesis is DEAD (P3.0).** DRAM% flat ~33% (not climbing to saturation) ⇒ memory-**latency**-bound +
   scheduler-starved, a third category. "1 GPU beats 16 nodes" does NOT rest on bandwidth (16 nodes ≈ 10,000 GB/s ≈ 5× an
   A100); it rests on the cluster's **measured 28.5% parallel efficiency** at np16 (Amdahl f_s=0.182, S_max≈5.5×).
   Triangulated: V100 308 GB/s vs SPR node ~350 GB/s = **0.88× wash** — one GPU ≈ one CPU node on full-data likelihood.
4. **CTF beats the TOOL, not the CPU; it is DEPTH not BREADTH; it is CPU-portable.** On the fine-refine step a 103-core node
   refines top-k concurrently while the GPU mutex-serialises ⇒ CTF-on-GPU vs CTF-on-CPU = **wash-to-CPU-favourable**. The
   GPU's honest edge is **per-model DEPTH (4.8×)**, realised inside CTF's top-k≤3 refine.
5. **The native-vs-projected BIC gate.** SHIP the **native** subsample BIC `−2·lnL_sub + k·ln m` (un-amplified). The
   **projected** gate `−2·(N/m)·lnL'_sub + p·ln N` amplifies subsample overfit by ≈(N/m)·(k/2) and **demoted the true
   winner LG+G4 to rank 4** behind +R5/+I+R5 → caused both 1M -m MF benchmark timeouts (170728179/182). n=30: native 30/30,
   **projected DNA 0/15**. The panel's overfitting fear is the *projected* gate's behaviour, NOT the native gate we ship.
6. **CTF cannot make the output over-complex** — only mis-recall. The winner is chosen by EXACT full-data BIC over the
   re-optimised top-k; coarse optimism is discarded. A recall miss biases toward dropping the *simpler* model (under-fit),
   the opposite of the over-fit fear.
7. **FP64 parity is non-negotiable.** Never TF32/FP16/fast-math on the reduced lnL or gradient (likelihood terraces demand it;
   breaks the 1e-8 gate). Deterministic block-local pairwise reduction, no atomics.
8. **FD-validate every gradient before shipping** — "a wrong-but-plausible 20-state kernel is easy to ship (BEAGLE proved it)."
9. **Generative ≠ BIC-selected.** DNA-1M (GTR+I+G4-generated): BIC winner is F81+F+G4 (GTR exchangeabilities buy ~3 nat/1M
   ⇒ demoted to 18th). The old "oracle = GTR-family" expectation was WRONG.
10. **The 4.7×-slower G.2.1b wall is NOT the MF slowdown** — `-te` full-brlen-opt is the heaviest per-model workload; don't
    extrapolate. The JOLT in-tree wall is **4.8× FASTER** (G.4.2a).
11. **Report speedup as a curve vs pattern count, never one headline number; route small alignments to CPU.**

**Phrases to AVOID:** "1 GPU beats a 103-core node at 100K" (false by N/S); "100K throughput is a coin-flip" (for the
full-data engine it's a *loss*); any 1M/10M number as a *throughput* win (tiling is built = *capability*, the bandwidth
throughput win is unproven/falsified); "1M times out" unqualified (only -m MF with the OLD projected gate did).

---

## 5. Key reusable numbers

**Anchored CPU baselines (AA-100K unless noted):** stock `-m MFP` **399 s** (job 168425673, 103T); AVX-512 single-model
floor ≈ **221 s**; vanilla -m TEST 264.2 s; FCA np1 = 1289 s (full) / 258.8 s (MF-phase) — *qualify by scope when reused*.
**AA-1M:** FCA np16 -m MF **1122 s** (np1 5119.9 s, 4.56×, **28.5% parallel efficiency**); node walls np2 3076.9 / np4
1974.5 / np8 1443.9 / np16 1122 s. **Reference lnL:** LG+G4 AA-100K = **−7541976.853**, s.e. 15407.1942.

**GPU walls (measured):** AA-100K JOLT `-te` **47 s V100** (4.8×); AA-1M CTF H200 **893 s** (1.26× vs np16), A100 **1355 s**
(1.07× vs np8); **DNA-1M -m MF CTF A100 152 s** (7.4–13×); AA-1M -m MF CTF H200 **767 s** (1.46× vs np16).

**Energy (per-device):** H200 CTF **67.89 Wh** (~280 W), A100 81.69 Wh; DNA-1M 7.53 Wh; CPU np1 MF-only 791.8 Wh
⇒ **MF-phase 11.7× vs np1, ~29× vs np8**.

**VRAM (per-model, native-20, FP64):** AA-100K g4 6.16 GB, r10 14.93 GB. AA-1M JOLT (postorder+preorder) **~88.6 GB**
(measured peak 67.8 GB @946k patterns; postorder arena 59 GB). AA-10M **~886 GB** (fits no single GPU one-shot ⇒ tiling
mandatory). Tiling: AA-1M T=10 → 8.9 GB (V100/RTX4090), T=40 → 2.2 GB (any GPU); AA-10M on H200 auto-T=6 → 112.8 GB.

**Registers/occupancy (production cubin, per-arch, part9 IX.10.1):** `kj_derv_fused` 40 (V100 sm_70) / **32 (A100 sm_80,
H200 sm_90)** ⇒ 100% occ on production cards, ~80% V100. Reduction kernels (kj_ratenum/reduce3/reduce_gradnum/invl) all
≤32 ⇒ 100% all arch. `k1_node`/`kj_pre` 56 regs (+stack spill) ⇒ ~50–57%, **latency-bound by design, never a 100% target**.
`__launch_bounds__` caps make these kernels SLOWER (spilling) — declined.

**Convergence iterations:** JOLT cold **27 joint iters** (branches+α), warm 14; vs IQ-TREE ~197-deep sequential
Gauss-Seidel + α-Brent. Per-model traversals: +I+G4 178, +G4 ~19–27, bare ~3.

**CTF structure (AA-100K):** top-3 all LG within ΔBIC≤264, then a **17,618-nat cliff** to #4 ⇒ recall near-certain.
**FP64-danger:** the per-parameter BIC penalty `ln(m)` = **11.5 (100K) / 13.8 (1M)** is the same scale as the inter-model
**lnL** differences (AA-100K lnL diff 6e-4, AA-1M 4.7e-3) — a sub-nat lnL error can flip a near-tie ⇒ FP64 mandatory.
(11.5/13.8 are the `ln(m)` penalties, NOT inter-model ΔBIC; the actual nearest-neighbour full-data ΔBIC ≈13 at 100K, and
LG+G4→LG+I+G4 = +14.3.)

---

## 6. Coverage state + the eligibility gates (`phylotreegpu.cpp`)

**Coverage (measured, job 170602983 + G.6.1):** **AA -m MF ~95% → ~near-full** (only +R/+I+R/pure-+I decline after G.6;
+R is G.5.1b). **DNA -m MF 8% → ~89%** (free-Q ON by default since G.6.1; residual = +R/+I+R/pure-+I).

**The gates (decline → CPU):**
- `getNDim()!=0 → "free-subst-params"` — RELAXED by G.6.1 for `ns==4` reversible free-Q (HKY…GTR), `getNDim()≤5`, fixed
  freqs (excludes +FO `FREQ_ESTIMATE` AND tied DNA freqs `FREQ_DNA_*` — RISK-1 fix); AA +FO/GTR20 still decline.
- `isGammaRate()!=GAMMA_CUT_MEAN → "non-mean-gamma"` — declines ALL +R/+I+R (median-gamma). **This is the gate G.5.1b would
  flip for +R.** Currently UNFLIPPED (R6 multimodality).
- pure-`+I` declines; `getNMixtures()!=1 || isSiteSpecificModel()` declines (the **G.8 target** — profile mixtures + PMSF).
- Safety: write-back rel > 1e-6 ⇒ NaN→CPU fallback (RISK-3 fix: `if(!(rel<=1e-6))` so NaN trips it, not `>1e-6`).

**Correctness discipline (every phase):** standalone FD/clean-room cross-check BEFORE the gate flips; **CPU-optimum
comparison gate** (JOLT lnL ≥ CPU−eps else NaN→CPU); deterministic FP64 reductions; CPU path byte-identical; gate on
`outIters`+accept/reject sequence not just final lnL.

---

## 7. Open work (ranked)

1. **G.5.1b — +R / +I+R in-tree JOLT.** The last AA -m MF gap. Gradient FD-validated (G.5.1a i1) and exact at every ncat
   (WN identity `relWN=0.0`). **Boundary now MAPPED (job 171518800):** R3 & R4 reproducible (4/4 starts agree, match/beat
   CPU-EM); **R5/R6/R8 multimodal** (1/4–1/8 starts; JOLT's best beats CPU-EM ~10 nats but not reproducibly) — **cutoff =
   R4↔R5.** ⇒ data-justified scope: **engage in-tree +R for ncat≤4 only, decline ncat≥5 to CPU** (EM-warm-start could lift
   the cap later). The gate flip (`phylotreegpu.cpp:502`) is now justified *for ncat≤4* but is an invasive in-tree change —
   surface + CPU-optimum-gate backstop before flipping; do NOT flip wholesale on the multimodal high-ncat path. +I+R
   declines initially (RateFreeInvar's `pinv=1−Σprop` ≠ G.4.3b coupling).
2. **G.8 — Profile-mixture JOLT (C60/MEOW80/UDM).** part9 §IX.11. The eukaryote LG+MEOW80+G4 case. A genuine per-model
   **DEPTH** case (one heavy model, no breadth competition) — **NOT** an occupancy-ceiling escape and NOT a new grid.z
   breadth win (red-team-corrected; 80× arithmetic = more latency-bound work, not 80× speedup). Real structural wins: fixed
   profiles → cache eigens once; branch gradient linear across classes. Weights by EM (de-risked vs +R multimodality), but
   CPU default is BFGS ⇒ MLE-equality gate. **✅ G.8.0 (lnL) + G.8.1a (per-class posterior) + G.8.1b (branch derivative)
   DONE — all ×3 at machine-eps, commits `2277273d` + `b855c3fe`** (C20/C60/MEOW80 lnL rel ~1e-16; posterior |Δγ| ~1e-12;
   df/ddf vs CPU `computeLikelihoodDerv` ~1e-13/1e-14 INT-INT + LEAF; 320-regime MEOW80 via global-mem per-regime arrays,
   so the 64-entry `__constant__` cap was NOT hit by the clean-room path — it WILL bind a production `k1_node`). **NEXT: G.8.2
   EM weight optimiser** (the G.8.1a posterior M-step + G.8.1b branch gradient ride the joint diagonal-LM) → G.8.3 seam + gate
   relax (`phylotreegpu.cpp:573`) → G.8.4 eukaryote payoff. **Watch (still ahead):** low-register class map, the `__constant__`
   cap when the kernel goes in-tree, EM near-zero-weight overfitting floor. ~12 days remaining.
3. **CAT-PMSF** (site-specific `ModelSet`, +R4) — separate later track (per-site π, no class sum).
4. **10M throughput follow-up** — the host self-check at extreme nptn dominates wall (a HOST cost after GPU optimise);
   sample/skip it. Capability (JOLT engages, lnL-exact) is shipped (G.7.1).
5. **Port the native-BIC gate + rate-het detector + wall budget into the production CTF path** (lives only in bench scripts;
   single tested helper `gadi-ci/gpu-modelfinder/ctf_rerank.py` extracted, byte-identical, selftest passes — rewire pending).
6. **Commercial-card precision (part11)** — df64/compensated-FP32 + precision-tiering for 1/64-FP64 consumer GPUs; design
   done, experiment unbuilt.
7. **FP64 tensor-core MMA (part12) — ✅ CLOSED, T.0 said STOP.** The BEAGLE-benchmark "confirmed lever" was tested with a
   standalone decider (`tc_decider.cu`, jobs 171587052 H200 / 171587053 A100): DMMA matvec **0.36× @1M H200, 0.60× A100**
   = 1.6–2.8× SLOWER than JOLT's scalar matvec (parity bit-identical rel 0.0). Gate ≥1.3× → STOP. Confirms the register-
   fused eigenspace scalar already captured the gain (20→32 pad ~2.56× FLOPs + latency-bound negate the TC edge); BEAGLE's
   tuned ceiling ~13%<gate corroborates. **Kill-switch worked: 0.43 SU vs 6–10 days saved.** Citable NO. Done, not pursued.

---

## 8. Contradictions / corrections across the source docs (so future-me doesn't re-introduce them)

1. **VRAM "126 GB" (AA-1M lnL) → 63 GB native / 101 GB padded** (design self-correction); JOLT gradient-bearing footprint is
   a *different* quantity (~88.6 GB).
2. **Bandwidth thesis (parts 3/4 "decisive 1M/10M HBM-bandwidth win") FALSIFIED in part5 P3.0.** Reframe = compute-throughput
   + cluster inefficiency, lever = **occupancy** not tiling; single-GPU full-data = wash vs one CPU node.
3. **"32 regs / 100% occ" (part8) → V100 is 40 regs / ~80%** (part9 IX.10.1 cuobjdump); production A100/H200 are 32/100%.
4. **"oracle = GTR-family" WRONG** (G.6.2) — BIC winner is F81+F+G4; generative ≠ BIC-selected.
5. **10M: the host wall masked the VRAM wall.** Original "VRAM was never the limit" was wrong for 10M (886 GB arena);
   G.7.0 fixed host → exposed VRAM → G.7.1 tiling fixed it. Tiling DOES address 10M (opposite of pre-fix conclusion).
6. **The projected "scale-consistent BIC" (part5 §V.4, part6) is the OVERFITTING BUG**, superseded by the native gate
   (part5 §V.14, part10 §X.5.5, part9 IX.7). part6 predates the fix and still endorses it — ignore that recommendation.
7. **Coverage "5% (12/224)" was a double logging artifact** (model->name dropped +F; cap report_count<12); real = 58/62 (94%).
8. **+I single-start "fix" was itself a regression** (lost 39.5 nat at pinv≈0.5); corrected to **4-start** (the restart
   sweep IS the robustness mechanism). The same multimodal lesson drives the +R caution (G.5.1b) and the EM-for-weights
   choice in G.8.
9. **§IV.7.2 self-supersedes**: "coverage before batching" → "batching is the structural gate" (N/S arithmetic).
10. **Minor unreconciled:** G.4.2b TESTONLY wall appears as both 3493 s and 3541 s across docs (~48 s, scope/run diff).

---

## 9. Source-doc index (full detail lives here)

| doc | purpose | status |
|---|---|---|
| `gpu-modelfinder-design.md` | PART I/II founding design (BEAGLE de-risk → custom in-tree kernel) | foundational; optimizer dir superseded by part4 |
| `gpu-modelfinder-g0-log.md` | G.0 BEAGLE de-risk execution log | historical; BEAGLE-gradient path abandoned |
| `gpu-modelfinder-g1-log.md` | G.1+ custom kernel build log (K1–K4, G.2 seam) | historical record through G.2 |
| `gpu-modelfinder-g2-plan.md` | G.2 in-tree integration plan (4-pointer seam, bridge) | superseded by part3→part4 |
| `gpu-modelfinder-part3-architecture.md` | PHALANX-BMF cross-model batching (grid.z) | DESIGNED, NOT BUILT; de-prioritized post-P3.0 |
| `gpu-modelfinder-part4-jolt-optimizer.md` | **JOLT optimizer — CENTRAL** | algorithm done, in-tree, correct |
| `gpu-modelfinder-part5-coarse-to-fine-and-the-100K-verdict.md` | **CTF + 100K verdict — CENTRAL, most honest** | active, most-updated |
| `gpu-modelfinder-part6-per-pattern-racing-feasibility.md` | per-pattern streaming/racing feasibility | closed NEGATIVE |
| `gpu-modelfinder-part7-vram-space-complexity.md` | VRAM/pattern-tiling | BUILT (`a972cefc`) |
| `gpu-modelfinder-part8-jolt-code-audit.md` | honest JOLT code audit + perf levers | active audit log |
| `gpu-modelfinder-part9-full-mf-coverage-and-scaling.md` | **G.5/G.6/G.7 coverage+scaling + G.8 plan — CENTRAL, most recent** | active |
| `gpu-modelfinder-part10-subsample-sufficiency-hypothesis.md` | CTF statistical foundation (subsample sufficiency) | active; native-BIC confirmed |
| `gpu-modelfinder-part11-commercial-card-precision.md` | FP64/consumer-card precision (df64/tiering) | research verdict; experiment unbuilt |
| `gpu-modelfinder-part12-fp64-tensor-core-lnl-kernel.md` | FP64 tensor-core MMA lever (vs BEAGLE-4.0 tensor cores) | PLAN; kill-switch-first; conditional/likely-marginal; unbuilt |

**Realised wins (honest summary):** per-model DEPTH 4.8×; CTF-vs-stock-tool 1.5–13×; energy 12–29× (MF-phase); capability
(AA-10M now runs on one H200; DNA -m MF meaningful on GPU). **Not** a breadth win vs a large cluster at full-data 100K
(N/S≈21×); **not** a bandwidth win (falsified). Next frontiers: +R (G.5.1b), profile mixtures (G.8). FP64 tensor-core MMA
(part12) was tested + CLOSED (T.0 STOP: DMMA 1.6–2.8× slower than JOLT scalar; register-fused eigenspace already wins).
