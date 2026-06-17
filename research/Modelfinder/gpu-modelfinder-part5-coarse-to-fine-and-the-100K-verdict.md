# PART V — The 100K Verdict + Coarse-to-Fine: an honest, first-principles answer to "break GPU ModelFinder at 100K"

**Author:** as1708 (multi-agent research synthesis by Claude Opus 4.8, 2026-06-10)
**Status:** RESEARCH VERDICT + DESIGN. Built on the validated G.0→G.4.2 foundation (JOLT optimizer correct +
in-tree + thread-safe; PHALANX grid.z designed). Produced by two adversarial multi-agent workflows
(understand+literature sweep; 6-lens judge panel + red-teams) + advisor review + a structural BIC pre-check +
the empirical coverage re-measure (job 170386010).
**Scope:** the question the user posed — *can we break the AA-100K `-m TESTONLY` ModelFinder wall-clock on one
GPU, and does failing to do so disprove that the algorithm parallelizes?*

> **The one-paragraph verdict.** There is **no clean GPU-*specific* throughput win at 100K** — and that is a
> property of the regime (one GPU under-saturated at 96K patterns + N=103 CPU cores running models
> concurrently), **not** a failure of parallelization. The per-model algorithm *provably* parallelizes (JOLT:
> 27 cold parallel iterations → the same MLE, replacing IQ-TREE's un-parallelizable 197-deep sequential
> Gauss-Seidel). We **can** beat the *current tool's* 100K wall-clock — via **Coarse-to-Fine (CTF)**, which
> ranks all candidates on a tiny pattern subsample then refines only the top-k≤3 on full data (~57–151 s vs the
> CPU's 221 s floor / 399 s `-m MFP`). But CTF is an **algorithmic** restructuring that is **CPU-portable** —
> it beats the *tool*, it does not prove the *GPU* beats the *CPU* at 100K. **The GPU's structural advantage is
> bandwidth, and it is realized — decisively and unconditionally — at 1M/10M, where one A100 beats ~16 CPU
> nodes.** That, plus JOLT's validated parallelization, is the real answer to "does this scale."

---

## V.1 The spine — one question governs everything

**Does the workload saturate GPU memory bandwidth?**

- **At 100K it does NOT.** Measured (ncu, job 170195112): DRAM 16–40%, Compute(SM) 36%, Memory 48%, L1 71–83%
  — *nothing saturated*; achieved occupancy **23.85%** (theoretical 25%) from **128 regs/thread → Block Limit
  (Registers) = 2 = 16/64 warps**. Too few resident warps to hide memory latency. The kernel is
  **latency/occupancy-bound**, so the GPU cannot realize its only structural edge (bandwidth), and it loses on
  aggregate serialization.
- **At 1M/10M one model DOES saturate** — *IF the kernel becomes bandwidth-bound as nptn grows* (this is a
  HYPOTHESIS gated by P3.0; see V.7). The per-device bandwidth ratio is 3× (V100 900 GB/s) / 6.7× (A100 2 TB/s)
  over one SPR socket (~300–400 GB/s).
  > **CORRECTION (2026-06-10, advisor — the headline arithmetic):** "1 A100 beats **16 nodes**" does NOT rest on
  > a bandwidth ratio. Sixteen nodes ≈ **~10,000 GB/s aggregate** — roughly **5× the A100's 2,000**. A purely
  > bandwidth-bound contest, the *cluster wins*. The GPU win rests entirely on the cluster's **measured 28.5%
  > parallel efficiency** at np16 (AA-1M FCA: np1 5119.9 s → np16 1122.4 s = 4.56× of 16; Amdahl f_s=0.182,
  > S_max≈5.5×, 83% of ceiling). The honest framing is **"GPU bandwidth-bound and ~100% efficient (one model
  > saturates the device; JOLT removes the 75–85% serial branch-opt fraction) vs CPU cluster ~28% efficient
  > (Amdahl/dispatch waste squanders most of its aggregate bandwidth)."** P3.2 must therefore be a **direct wall
  > measurement** (A100 full-MF wall vs the CPU's *measured* 1122 s), NOT a bandwidth-ratio projection, with the
  > gap attributed to cluster inefficiency. And the whole thing is **gated on P3.0** proving the GPU kernel
  > actually becomes bandwidth-bound at scale — occupancy (16/64 warps) caps latency-hiding regardless of nptn,
  > so this is genuinely uncertain, not given.

**Therefore a lever helps at 100K ONLY if it (i) converts latency-bound → bandwidth-bound (breaks the 25%
occupancy ceiling), or (ii) shrinks the precision-critical work itself.** Every other lever is a *measured*
wash — proven by our own history: CUDA graphs (K3) = parity; kernel fusion (K4, 98→42 launches) = wash-to-loss
at 100K (g4 1.05×, r8 0.96×, r10 0.94×); `__launch_bounds__` register caps = all slower (spilling).

---

## V.2 The arithmetic that kills the naive GPU win at 100K — measured, not theorized

**N/S aggregate serialization.** A mutex-serialized single GPU at per-model speedup **S ≈ 4.8** processes
candidates one-at-a-time; the CPU runs **N = 103** concurrently (1 thread/model). Aggregate ratio = **N/S =
103/4.8 ≈ 21× slower** — *at any coverage*.

**Confirmed directly** (job 170386010, `--jolt -m TESTONLY -nt 12`, fixed logging + gate instrumentation):

| measurement | value | meaning |
|---|---|---|
| candidates reaching the JOLT hook | 62 | |
| **JOLT engagements** | **58** (incl. 28 `+F`) | **coverage ≈ 94%** — earlier "5%/12-of-224" was a pure logging artifact (capped print + dropped `+F` suffix) |
| declines | **4, all `reason=pinvar`** | only `+I`/`+I+G4` fall to CPU |
| **GPU utilisation** | **96%** | GPU busy the whole wall, serializing 58 models |
| CPU busy | **~2.05 / 12 cores** (~83% idle) | threads blocked on the JOLT mutex |
| wall | **~59 min** | ≈ 58 models × ~61 s serialized |

So the wall is **GPU-serialization-bound, not the CPU tail, not coverage.** Coverage is solved (94%); `+I`
coverage (the deferred G.4.3b p_inv gradient) would only *lengthen the mutex queue* and is therefore **off the
critical path** until concurrency exists.

**Why batching (PHALANX grid.z) does NOT rescue full-data 100K.** Every per-node kernel launches
`ceil(96017/256) = 391` blocks **regardless of tree depth** — the SMs are **block-saturated** (2.44 waves/SM).
Adding a model plane (`grid.z`) multiplies *queued* blocks, not resident warps, so B does **not** multiply
throughput at saturated levels. Even the optimistic A100 ceiling **g4 B=12 × S=4.8 = 57 < N=103** → still
~1.8× slower. **Batching at full-data 100K is a coin-flip-to-loss.** (The deep 33-node serial tail is *also*
not SM-starved at 100K — 391 blocks/node even there — so its problem is the dependency chain, which JOLT's
2-sweep gradient already shortens; launch latency is already hidden, per K3 parity.)

---

## V.3 The one regime where the constraint INVERTS — the novel idea

**Saturation-inversion (the single best novel idea).** The very fact that kills 100K — block-saturation —
**inverts at small pattern counts.** At a ~480–1000-pattern subsample each node launches only ~2–4 blocks;
one model occupies ~1.3% of the device. Now `grid.z` batching of **all ~28–60 candidate models** (each its
own eigensystem — the literature white space) runs in ~2 waves instead of ~80 serial passes, and the *whole*
batch fits in ~3.7 GB VRAM. **This is the one place the central constraint flips in the GPU's favor:** the
GPU's massive cross-model parallelism (every candidate advancing at once) finally multiplies, exactly where
the CPU's one-thread-per-model cannot keep pace per-model.

**Honest bound (the advisor's catch, stated up front, not buried):** the CPU never *had* the block-saturation
problem — it parallelizes across 103 cores. So saturation-inversion lets the GPU **catch up to where the CPU
already was at small N; it does not pass it.** It is real GPU physics, and it is what makes the coarse phase
cheap on the GPU — but it is **not** a GPU-beats-CPU result by itself.

---

## V.4 Coarse-to-Fine (CTF) — the way to beat the current tool's 100K wall-clock

**Mechanism.** Restructure ModelFinder from "fully optimize all ~58 candidates on 96K patterns" into:
1. **Coarse rank** — optimize *all* candidates on a small frequency-weighted pattern subsample (~480–1000),
   rank by BIC. At subsample scale this is cheap on either device; on the GPU it is the saturation-inversion
   batched pass.
   > ⚠️ **SUPERSEDED (see §X.5.5 and §V.14):** this section originally specified the **projected** "scale-consistent
   > BIC" `BIC' = −2·(N/m)·lnL'_sub + p·ln N`. That gate is **complexity-biased** — the `N/m` factor amplifies a
   > model's overfitting optimism by ≈`(N/m)·k/2 ∝ k`, and it demonstrably demoted the true winner LG+G4 to rank 4
   > behind +R5/+I+R5 on the real AA-1M subsample. **The shipped gate is now the NATIVE subsample BIC**
   > `−2·lnL_sub + p·ln m` (penalty `ln m`, *not* amplified), which restores LG+G4 to rank 1. The final winner is
   > always the **exact full-data BIC** over the refined top-k, so even the old biased gate could not over-fit the
   > *output* (it only risked recall). §V.14 is the full treatment of the panel's overfitting concern.
2. **Fine refine** — fully optimize only the **top-k≤3** on the full 96K patterns (FP64, JOLT for eligible,
   CPU for `+I`), pick the BIC winner.

**Why it wins the wall-clock — it dodges the heavy tail.** The expensive families are `+I+G4`/`+F+I+G4` at
**178 traversals/model** at full data (vs `+G4` 19–27, bare 3). CTF kills the non-competitive heavy-tail
models at 480 patterns for ~0.4 s each **instead of** paying 178 traversals × 96K patterns. The saving is N→k
**plus** the disproportionate avoidance of the most expensive models.

**Honest projected wall** (tagged ⏳ — rides on two unmeasured inputs; see V.6):

| top-k refined | wall ≈ k×47 s + ~10 s coarse | vs 399 s `-m MFP` | vs 221 s AVX-512 floor |
|---|---|---|---|
| 1 | ~57 s | 7.0× | 3.9× |
| 2 | ~104 s | 3.8× | 2.1× |
| 3 | ~151 s | 2.6× | 1.5× |

**The honest positioning (lead, do not bury).** CTF beats the **current CPU tool**. It is an **algorithmic**
restructuring that is **CPU-portable** — and on the dominant fine-refine step the CPU is actually *better*
(a 103-core node refines the top-k concurrently; the GPU mutex-serializes k×47 s). So the fair **CTF-on-GPU vs
CTF-on-CPU** comparison is **wash-to-CPU-favorable**. CTF is the answer to "break the *tool's* 100K
wall-clock," **not** to "the GPU beats the CPU at 100K." The GPU-specific residual at 100K is only the
warm-started single-model JOLT refine (4.8×, banked).

**The structural pre-check strongly de-risks the coarse phase (free, from existing logs):** the full-data BIC
table has top-3 all LG-family within **ΔBIC ≤ 264** (LG+G4; LG+I+G4 +14.3; LG+F+G4 +263.7), then a
**17,618-nat cliff** to Q.PFAM+F+G4 (#4). So a subsample need only preserve LG's 17,600-nat family dominance
to recall the top-3 into top-k; the tight within-LG ordering is exactly what the full refine re-resolves.
**Caveat (C-eligible):** the 14-BIC runner-up LG+I+G4 is `+I` → JOLT-ineligible → refined on **CPU** — priced
at the CPU heavy-tail rate, not 47 s, and run concurrently on the otherwise-idle node.

---

## V.5 The honest scorecard — what one GPU actually wins

| Claim | AA-100K TESTONLY | AA-1M / 10M |
|---|---|---|
| **GPU wall-clock vs stock CPU ModelFinder** | ✅ **WIN via CTF** (~57–151 s vs 221/399 s, k≤3) — but *algorithmic*, CPU-portable | ⏳ **WIN (projected, gated on P3.0)**: target A100 full-MF wall < FCA-np16's measured 1122 s. **Win = EFFICIENCY, not bandwidth ratio** (16 nodes have ~5× the A100's aggregate bandwidth but run at 28.5% efficiency). **Tiling kernel UNBUILT; bandwidth-bound-at-scale UNPROVEN (P3.0).** |
| **GPU-*specific* delta (vs CPU running the same CTF)** | ⚠️ **modest/wash** — GPU refine mutex-serialized; CPU refines top-k concurrent. Banked GPU gain = single-model JOLT 4.8× warm. | ✅ **decisive IF P3.0 holds** — one model saturates the device + JOLT removes the serial fraction → ~100% efficient vs cluster's 28.5%. |
| **Capability frontier** | n/a (CPU finishes 100K) | ✅ AA-1M `-m MF` times out on 16 nodes; 1 GPU + tiling finishes. (AA-1M *TESTONLY* finishes on CPU — do not over-claim.) |

**Defensible claims:** (1) "CTF reproduces the exact full-data BIC ranking at ~57–151 s on one GPU, beating
stock CPU ModelFinder ~1.5–7×." (2) "JOLT is a GPU-native joint all-branch+α optimizer reaching the same MLE
(cold==warm rel 1.5e-15, FD-validated gradients) in 27 cold / 14 warm parallel iters vs IQ-TREE's
un-parallelizable 197-deep sequential Gauss-Seidel — 4.8× single-model." (3) "The clean, unconditional
GPU-beats-CPU win is 1M/10M (bandwidth ratio)."

**Claims to AVOID:** "1 GPU beats a 103-core node at 100K" (false by N/S); "100K throughput is a coin-flip"
(for the full-data engine it is a *loss* — block-saturation); perf/watt at 100K (wash: A100 700 W/CPU 400 W =
1.75× vs ~1.8× time → no win); "1M times out" unqualified (only `-m MF` does, not TESTONLY); any 1M/10M number
as *achieved* (tiling is UNBUILT — tag ⏳).

---

## V.6 The answer to the actual fear — does a 100K loss disprove parallelizability? **NO.**

The per-model algorithm provably parallelizes: **G.4.1 banked 27 cold parallel iterations reaching the same
MLE at rel 2.5e-16**, each iteration *one parallel sweep over all 197 branches*, replacing the 197-deep
sequential Gauss-Seidel chain that cannot run on a GPU. The Mode-L "L.1" parallelization gate, re-stated in
the correct GPU metric (**critical-path length, not traversal count**), WON. The 100K loss has a
fully-identified, **regime-bounded** cause — **under-saturation + mutex serialization** (N=103 concurrent CPU
models vs 1 GPU), a throughput-*packing* artifact at small N — that **vanishes at 1M/10M** where one model
saturates the device. **A 100K loss measures GPU occupancy at small N, not algorithmic parallelizability.**
The scaling curve is the proof; the 100K point sitting below the bandwidth knee is *consistent with* the
thesis, not contrary to it.

---

## V.7 Recommended phased plan — cheapest-decisive kill-switch first

| Phase | Deliverable | Gate (concrete) | Cost / Risk |
|---|---|---|---|
| **P0 — Subsample-recall DECIDER ✅ PASS (job 170396778, 2026-06-10)** | Stock `-m TESTONLY` on 1000/2000/5000-site frequency-weighted subsamples of the existing AA-100K data, fixed tree; rank by BIC; compare to the existing full-data table (NO baseline re-run). | **✅ ALL THREE K (1000/2000/5000 = 1.0/2.1/5.2% of patterns): recall = 3/3 — subsample top-3 = full-data top-3 {LG+G4, LG+I+G4, LG+F+G4} in EXACT order at every K, and subsample best-fit = LG+G4 (the correct overall winner) at every K. The coarse phase reproduces the full ranking on as little as 1% of the data — robust.** (The run script's auto-recall line misprinted "0/3" — an awk early-`exit` bug on the blank line after the BIC header; the raw `.iqtree` tables confirm 3/3.) | minutes CPU. The CTF gate is unconditionally PASSED → GO. |
| **P1 — `outIters` warm-start measure** | Instrument `outIters` (`phylotreegpu.cpp:556`); measure subsample-warm vs forced-cold refine iters in real `-m TESTONLY`. **Do NOT re-run the full CPU baseline** (standing constraint) — anchor on 221 s/399 s, flag the missing same-scope denominator. | subsample-warm refine ≤ ~14 iters (else coarse phase is redundant with cheap CPU subsampling). | ~1 job. Risk: poor warm init → GPU residual value dies. |
| **P2 — Eligibility-aware CTF pipeline** | P0 rank → top-k shortlist → full-96K `--jolt` refine (eligible) + CPU refine (`+I`), warm-started from coarse; report each top-k member's param-class so `+I` is priced at the CPU rate. *Optional:* GPU-batched coarse phase (saturation-inversion) only if P1 shows the coarse phase is on the critical path. | end-to-end wall < 221 s producing the exact full-data BIC winner; honest CTF-on-GPU vs CTF-on-CPU comparison. | the build. Risk: see V.4 honest positioning. |
| **P2∥ — Occupancy microbench (moonshot, ~1 SU, parallel)** | Standalone `gpu_k1tpcs_occ.cu`: thread-per-(pattern×category×output-state) collapses `prod[NS=20]` to a scalar (~2 regs vs ~40) — an *algorithmic* footprint cut, unlike the failed `__launch_bounds__` sweep. | ptxas spill=0 AND ncu warps/SM ≥ 48% AND g4 sweep ≤ 29 ms at rel ~1e-12 vs G.0 oracle. | ~1 SU. **The ONLY lever attacking the real bound** → if it lands, 100K becomes a clean GPU win (recreates the 1M regime at 100K). **Honest: coin-flip** (NS=20 matvec likely too small to amortize the reduction). Do NOT block CTF on it. |
| **P3.0 — Bandwidth-knee KILL-SWITCH ❌ FALSIFIED (job 170398260, 2026-06-10)** | Profiled a SINGLE WIDE-LEVEL postorder `k1_node` launch **in steady state** at growing nptn, lnL-only on V100, 8 launches/size median. | **VERDICT = FLAT (the pre-registered falsification branch).** Steady-state DRAM% sits at **34 → 33 → 34%** across nptn 100K→200K→300K (NO climb toward saturation); GB/s plateaus at **~300 of ~900 peak (~33%)**; achieved **warps/SM dead-flat at ~48.6%** (occupancy-capped: 56 regs → `occ_limit_registers`=4 blocks/SM). The only metric that moves is **SM% (compute throughput) RISING 38 → 49 → 56%** ⇒ the kernel is **latency/occupancy-bound, drifting *compute*-bound** as nptn grows — **NOT memory-bound, at any scale in range.** | ~2.4 SU. **The bandwidth thesis is dead; tiling would optimize a non-bottleneck.** |
| ~~P3.1 — pattern-tiling kernel~~ **CANCELLED** | Pre-condition was "P3.0 climbs." It stayed flat at 33% DRAM. Tiling optimizes HBM traffic — which is NOT the binding resource (occupancy is). Tiling remains needed *only* as a **capability** enabler (fit AA-10M in VRAM), never as a *throughput* lever. | — | — |
| **P3.2 → re-grounded: the win is COMPUTE-throughput + cluster inefficiency, not bandwidth** | The AA-1M direct-wall measurement still stands as the headline, but its *mechanism* is corrected: the GPU win is **(a) compute throughput** (SM% rising with scale, kernel fills up at large nptn) **and (b) the cluster's measured 28.5% efficiency** (Amdahl f_s=0.182) — NOT HBM bandwidth. The binding GPU lever is **occupancy** (P2∥), not tiling. | direct wall < 1122 s; report SM% (not DRAM%) as the saturating metric. | gated on P2∥ lifting occupancy. |

**Discarded (one line each):** full-pattern grid.z throughput-engine at 100K (block-saturated → B doesn't
multiply → decisive loss); occupancy-attack as a *standalone* 100K throughput win (raises S, never touches N →
still ~10× short); bandwidth-headline *at 100K* (kernel stays latency-bound until ~1M — it is the scale win).

---

## V.8 The immediate next experiment (running now)

**P0 subsample-recall test — job 170396778.** The highest-information, lowest-cost decision in the program:
minutes of CPU, zero new code, zero GPU dependence. It gates the one path that can break the 100K wall-clock.
- **3/3 recall → GO** on CTF (P1: `outIters` warm-start).
- **<3/3 → the only positive-arithmetic 100K path is dead → pivot cleanly to the unconditional 1M/10M
  bandwidth win** (build the G.3.5 tiling kernel; bit-parity first, then per-model AA-1M wall with the ncu
  DRAM-throughput fraction).

**Bottom line for the user:** we *can* break the current tool's 100K wall-clock (CTF, ~1.5–7×), and that is
worth building — but be precise that it is an algorithmic win, not a GPU-beats-CPU-at-100K win. The thing you
actually feared — that a 100K shortfall means the algorithm doesn't parallelize — is **false and disproven**:
JOLT parallelizes the per-model optimizer to the same MLE. **CORRECTION (P3.0): the GPU's structural advantage
at scale is *compute throughput + cluster inefficiency*, NOT HBM bandwidth.** **Optimize 100K with CTF for
usability; claim the GPU victory at scale — but frame it as compute/efficiency, and attack occupancy not bandwidth.**

---

## V.10 Batched-CTF 1M MFONLY decomposition (2026-06-10) — grid.z is the margin, CTF is the lever

Before building grid.z cross-model batching (the user's choice toward a 1M `-m TESTONLY` JOLT-vs-FCA win), the
end-to-end wall was decomposed from MEASURED numbers (FCA np16 1M MF = 1122 s; JOLT LG+G4 -te 100K = 47 s V100;
A100 = 1.7× V100; 1 model's +G4 partials ≈ 60 GB at 1M):

| phase | serialized | grid.z-batched | note |
|---|---:|---:|---|
| **Coarse** (rank 224 on ~2 k-ptn subsample) | ~129 s | **~30 s** | grid.z saves only ~99 s here |
| **Refine** (top-3 {LG+G4, LG+I+G4, LG+F+G4} at full 1M) | **~541 s** | ~541 s | **DOMINATES**; 3×60 GB > 80 GB ⇒ GPU *serializes* the refine; +I→CPU |
| **Total (single A100)** | | **~571 s** | vs FCA 1122 s ⇒ **1.96× faster than FCA-the-TOOL** |

**Counterfactual — a 103-core CPU node running the SAME CTF ≈ 860 s** (coarse ~60 s + refine top-3 *concurrently*
~800 s; CPU has no 60 GB co-residence cap, so it parallelizes the refine the GPU must serialize). So GPU-CTF
(571 s) leads CPU-CTF (860 s) — **but that lead is JOLT's per-model refine *depth* (4.8×), NOT grid.z.**

**Conclusions:**
1. **grid.z batching is a ~99 s coarse-phase margin, not the win.** A multi-day kernel restructure for ~99 s is
   poor ROI *unless* the 1M recall-subsample must be large (at 2 k-ptn coarse is cheap either way; at ~20 k-ptn
   serialized coarse ≈ 1300 s and grid.z would save ~1200 s — so grid.z's ROI is *gated on a cheap 1M-recall-size
   test*, not on the build).
2. **The win is CTF (refine 3 not 224) + JOLT per-model depth — both largely already exist.** "batched-CTF" is
   mostly orchestration over the built JOLT, not new kernels.
3. **The GPU's honest edge is DEPTH (per-model 4.8×), not BREADTH.** MFONLY is a breadth/dispatch problem that
   CPU clusters (FCA) own by design; the GPU loses breadth to serialization (mutex, 60 GB co-residence). Depth
   pays in: the refine, the deferred ~50% NNI tree-search phase, single-large-tree inference, and multi-GPU.
   **Chasing an MFONLY-breadth win on one GPU fights the GPU's weakness against the CPU's strength.**
4. **P3.0 removed the only reason 1M would differ from the 100K verdict.** 1M now *inherits* it: no clean
   GPU-specific MFONLY win; batched-CTF beats the TOOL (FCA's exhaustive 224-model dispatch), not the CPU.

**Recommendation:** do NOT start the grid.z build for a ~99 s margin. Either (a) build the cheap CTF pipeline
over existing JOLT for the honest ~2×-vs-FCA-tool headline, or (b) invest in where depth unlocks what a CPU node
can't cheaply do — the tree-search GPU hook (the deferred ~50%) and/or multi-GPU per-model-depth scaling.

---

## V.9 P3.0 VERDICT (2026-06-10, job 170398260) — the bandwidth thesis is FALSIFIED

`ncu` profiled the postorder lnL kernel `k1_node` in steady state (8 launches/size, median) at growing pattern
counts on a V100 (HBM2 ~900 GB/s peak, 64 warps/SM max):

| nptn | grid (blocks) | DRAM % | GB/s | warps %/SM | SM % | dur (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 000 | 378  | 34.3 | 308 | 48.6 | 37.9 | 5.0 |
| 200 000 | 752  | 32.6 | 299 | 48.9 | 49.2 | 7.6 |
| 300 000 | 1123 | 34.0 | 308 | 48.7 | 56.3 | 10.1 |

**Reading (pre-registered):** DRAM% is **FLAT at ~33%** — it does *not* climb toward the ~70–90% that a
bandwidth-bound kernel would show at scale. Achieved occupancy is **pinned at ~48.6% warps** (the 56-register
footprint caps residency at 4 blocks/SM = 32 of 64 warps), *independent of nptn* — exactly the mechanism the
kill-switch was designed to detect.

**The kernel is memory-*latency*-bound — a third category, distinct from bandwidth-bound and compute-bound.**
The rising **SM% (38→56%)** is NOT approach-to-compute-saturation (if it were, occupancy couldn't help — you
can't issue more math than the pipes allow). It is **improved latency-hiding as the grid fills**: at 100K only
~1.2 occupancy waves (378 blocks / ~320 resident) overlap; at 300K ~3.5 waves overlap and hide more memory
latency → SM% climbs. It will **plateau below 100% precisely because occupancy is capped at half the warps**
needed to cover the kernel's memory-latency. Low DRAM% + low occupancy + latency-limited throughput ⇒
**memory-latency-bound, and occupancy is exactly the right lever for that category.** *(This stall-reason
reading is itself being measured — see P3.0b below — before it is treated as established.)*

**Consequences:**
1. **The "unconditional 1M/10M HBM-bandwidth win" is falsified.** Bandwidth is not the binding resource at any
   scale we can profile; a pattern-**tiling** kernel (P3.1) would optimize HBM traffic that sits at 33% — a
   non-bottleneck. **P3.1 is cancelled as a throughput lever** (it survives only as a *capability* enabler:
   fitting AA-10M into VRAM, which is real but separate).
2. **The GPU-at-scale win is not dead, but its mechanism is now a HYPOTHESIS, not a verdict.** The candidate
   story — a single near-saturated GPU beats the cluster's measured **28.5% efficiency** (Amdahl f_s=0.182 on
   FCA-np16) — has **no throughput arithmetic behind it yet**, and the GPU itself runs at ~49% occupancy / 33%
   DRAM / ≤56% SM, i.e. it is *also* far from saturated. "Inefficient GPU beats inefficient cluster" requires
   the raw lnL-evals/s (or FLOPS) comparison spelled out before it is a claim. **Marked HYPOTHESIS** until the
   direct AA-1M wall (P3.2) supplies the number — same discipline the bandwidth-ratio claim was held to.
3. **The lever that touches the real bound is occupancy** — exactly **P2∥** (thread-per-(pattern×category×
   output-state), collapsing the `prod[NS=20]` accumulator from ~40 regs to ~2, breaking the 4-block/SM ceiling).
   **But promotion of P2∥ to critical path is GATED on P3.0b** (the stall-reason profile): occupancy only helps
   if warp-issue is *starved* (`warps_eligible_per_active_cycle` < 1) and the dominant stall is memory-latency
   (`stall_long_scoreboard`), NOT math/MIO-pipe throttle. If math-pipe dominates, P2∥ is dead on arrival and the
   build is saved.

**Pivot (gated):** P3.1 cancelled. **P3.0b stall-reason profile runs first** (~1 SU); *if* it confirms
latency-starvation, **P2∥ occupancy attack becomes the critical path** for any GPU-specific scaling claim. CTF
(V.4) is unaffected — it was always an algorithmic/CPU-portable win and remains the usability story at 100K.

> **P3.0b — stall-reason discriminator ✅ PASS (job 170399634, 2026-06-10).** `k1_node` warp-stall breakdown
> (median, 6 steady launches): **long_scoreboard (memory latency) = 50.1% @100K / 42.2% @300K** (dominant);
> short_scoreboard = 37.6 / 43.9% (also memory-dependency); **math_pipe (compute) = 0.08%** (≈zero);
> mio_throttle 7.3 / 8.7%; **issue_active = 7.8 / 8.8%** (schedulers issue on <9% of active cycles ⇒ starved for
> eligible warps ~92% of the time). **VERDICT: memory-latency-bound + scheduler-starved — NOT compute-bound
> (math_pipe ≈ 0). The advisor's "3rd category" reframe is empirically confirmed.** ⇒ **P2∥ occupancy attack is
> CONFIRMED worth building by measurement**: more resident warps directly fill the idle issue slots that memory
> latency is starving. The lever is occupancy (thread-per-(ptn×cat×state), ~40→~2 regs, break the 4-block/SM
> cap), NOT tiling (bandwidth, sits at 33%) and NOT anything compute-side. *(Optional bulletproofing: the A100
> 1M lnL-only point converts the 1M extrapolation, 33× beyond the 300K top measurement, into a measurement.)*

---

## V.11 TRIANGULATED VERDICT (2026-06-10) — one physics, three probes; the win is depth + multi-GPU

Three independent attempts to find a single-GPU, full-data win over the CPU all returned the **same physics**:

| Probe | Result | Mechanism |
|---|---|---|
| MFONLY `grid.z` cross-model batch | block-saturated at full data → B doesn't multiply | one model's pattern grid already fills the SMs (376 blocks @100K, 4.7× SM count) |
| 1M HBM-bandwidth thesis (P3.0) | FALSIFIED — DRAM% flat ~33%, no climb | memory-LATENCY-bound, 49% occ, 308 GB/s achieved |
| Tree-search GPU hook | wash-to-loss before building | **same** kernel, block-saturated; `optimizeAllBranches(1,…)` single-pass by design |

**The arithmetic:** V100 *achieved* 308 GB/s (P3.0, the real `k1_node`) vs one Sapphire-Rapids node's ~350 GB/s
DRAM ⇒ **0.88× = a wash. One GPU ≈ one CPU node on full-data phylogenetic likelihood, because the kernel is
memory-bandwidth-bound.** This is a *regime property*, robustly triangulated — NOT three failures, and NOT a
parallelizability failure (JOLT proved per-model parallelism: 4.8×, same MLE, by breaking the sequential branch
chain).

**Where the genuine, non-falsified win lives (depth, not breadth, not one-GPU full-data):**
1. **Multi-GPU throughput** — M GPUs × JOLT 4.8×/model = the honest GPU analog of FCA's M MPI ranks. The "beat
   the cluster" play; needs a cost-normalized comparison. **This is the user's stated north star** ("validate
   scaling at multiple GPUs on massive datasets"). UNBUILT.
2. **A single very-large analysis** — one huge tree that can't be cheaply MPI-split (the CPU cluster's breadth
   advantage evaporates when there's only one tree), where one GPU's parallel branch-opt (JOLT depth) wins
   outright. UNTESTED — the one single-GPU regime NOT yet falsified.
3. **JOLT as the banked per-model-depth result** — ship/benchmark honestly: 4.8× single-model fixed-topology
   optimization, same MLE, in-tree `--jolt` correct + thread-safe. The 100K full `-m TEST` run (job 170406241)
   gives the honest phase-decomposed baseline + parity.

**Discarded as single-GPU full-data throughput wins (all measured/triangulated):** MFONLY grid.z, 1M tiling
(bandwidth-bound falsified), tree-search hook (same saturated kernel). The tree-search ALGORITHMIC angle
(JOLT making full-convergence-per-round cheap → fewer topology rounds) is a *results-changing gamble*, not a
clean win — do not sell it as one.

---

## V.12 MEASURED AA-1M CTF on one GPU (2026-06-11, job 170517590, H200) — supersedes the V.10 projection

The V.10 decomposition **projected** ~571 s for a single-A100 1M CTF, assuming +I→CPU at ~180 s and refine ~541 s.
Both assumptions were wrong; here is the **measurement** (H200, +I now GPU-eligible via G.4.3b):

| phase | measured wall | note |
|---|---:|---|
| subsample 1M→5000 | 0 s | |
| **coarse** (stock `-m TESTONLY` on 5 k-ptn) | **158 s** | builds the fixed tree; top-3 = {LG+I+G4, LG+G4, LG+F+I+G4} |
| refine LG+I+G4 (full 1M, JOLT) | **869 s** | **10 JOLT calls** (random-restart pinv sweep) |
| refine LG+G4 (full 1M, JOLT) | **78 s** | **1 JOLT call** |
| refine LG+F+I+G4 (full 1M, JOLT) | **889 s** | **10 JOLT calls** |
| **TOTAL (1 H200)** | **1994 s** | **1.54× vs np2 (3076.9 s) ✓ ; 0.99× vs np4 (1974.5 s)** |

**Winner = LG+G4 (correct, matches the FCA oracle).** Peak GPU mem **67.8 GB** for 946,439 distinct patterns
(A100-80GB viable; the 88 GB estimate was high). GPU util 62 %.

**The V.10 projection's two errors, corrected by measurement:**
1. **+I is NOT ~180 s.** It is **~870 s** — not because the GPU is slow (~87 s/JOLT-call, same as +G4's 78 s) but
   because IQ-TREE's `RateGammaInvar` runs **10 pinv restarts**, each a full JOLT call, and on this 2.2 %-constant
   data **all 10 converge to the identical pinv→0 optimum** (9 wasted). This is the dominant cost and the clear
   lever: a single-start +I+G path under `--jolt` ⏳ → total ~412 s (**7.5× vs np2, 4.8× vs np4**). See
   PART VIII §VIII.4 #0.
2. **The win is bigger than "1.96× vs the tool" but on a different mechanism.** It is **measured 1.54× vs an actual
   2-node CPU run** (not a projection vs FCA-16's 1122 s) — and it is *still* the CTF algorithm + JOLT depth, which
   remains **CPU-portable** (a 2-node cluster could run the same CTF). The honest claim is unchanged from V.5/V.11:
   **CTF+JOLT beats the stock tool / a small node count on one accessible GPU; it does not overturn the
   breadth-vs-depth verdict vs a large cluster.** What G.4.3b added is that the +I tail no longer falls to a
   core-starved CPU — the single-device story now actually holds end-to-end.

**Status against the reduced bar:** **2 nodes — BEATEN (1.54×), measured, correct model.** 4 nodes — one validated
optimization away (the +I restart skip, or the on-device reduction). The PART VIII audit ranks both.

---

## §V.13 — The +I 4-start fix lands, and the measured GPU↔CPU energy parity (2026-06-12)

**The +I 4-start fix re-measured the headline.** Cutting `optimizeParametersGammaInvar`'s pinv restart sweep 10→4
under `--jolt` (G.4.3c, single-start rejected by the multimodal gate — 39.5-nat loss at pinv≈0.5, job 170580368;
4-start re-validated rel ~1e-3 on both collapsed AND pinv=0.5 data) **cut the 1M-AA CTF wall 1994 s → 893 s on the
H200** (job 170581208). At 893 s the GPU ModelFinder now **beats every measured CPU node count** on MF wall —
3.45× vs np2, 2.21× vs np4, 1.62× vs np8, **1.26× vs np16 (1122 s)** — still selecting LG+G4. A100-80: 1504 s
(job 170581209). This is a real shift from V.12's "2 nodes beaten / 4 nodes one-opt-away": the +I fix alone took it
past 16 nodes on the MF phase.

**Measured per-device energy (the decisive axis).** GPU via `nvidia-smi power.draw` (no counter wrap); CPU via
direct RAPL `energy_uj` (pkg+dram, both sockets), one sampler/node over `mpirun -rf rankfile`, on **fully-allocated
nodes** (`ncpus=104·N`, `mem=500GB·N` — RAPL sums the whole socket, so a shared node is contaminated).

| Phase / config | wall | energy | per-device power |
|---|---:|---:|---:|
| **H200 CTF (MF)** | 893 s | **67.89 Wh** | ~280 W |
| A100 CTF (MF) | 1504 s | 81.69 Wh | ~199 W |
| CPU np1 **MF only** (job 170582791) | 4068 s | **791.8 Wh** | 701 W/node |
| CPU np4 **full -m TEST** (170582814) | 5994 s | **4.18 kWh** | ~626 W/node |
| CPU np8 **full -m TEST** (170582815) | 3642 s | **4.93 kWh** | ~607 W/node |

**MF-phase energy benefit (H200 67.89 Wh):** 11.7× vs np1, ~20× vs np4 (MF fraction ~1.38 kWh), ~29× vs np8
(~1.97 kWh). The H200 draws ~280 W vs ~600–700 W *per* SPR node and finishes the MF faster than 16 nodes ⇒ **no CPU
node count is simultaneously faster and lower-energy than the single GPU.** CPU energy grows ~linearly with node
count while wall saturates (np16 only 2.7× faster than np2), so pushing the cluster harder only widens the gap.
np4/np8 reproduced the baseline walls to <1% (5994 vs 5956.6 s; 3642 vs 3671.6 s) → validated parity-matches.

**Methodology bug found & fixed (banked).** First RAPL integrator summed all 4 domains into one counter and added
the *combined* 655 kJ range on any wrap → ~3× over-count (~1900 W/node, non-physical). `energy_uj` wraps
**independently per domain** (pkg 262.1 kJ, dram 65.7 kJ); fix matches each negative delta to the specific domain
range that wrapped → physical ~600–700 W/node. The GPU number was never affected (power×dt, no wrap). Also: Gadi
`pbsdsh` has **no `-u`** (use `mpirun -rf rankfile` to put one sampler per node); `perf-report --no-mpi` pins the
process to 1 core via the OpenMPI singleton, so it cannot measure a full-node multithreaded run — direct RAPL is the
same counter Forge reads internally, without that artifact.

**Still honest about scope:** CTF is MF-equivalent only; the GPU does **not** yet run the 1M tree search (deferred
JOLT tree-search hook), so the *full* `-m TEST` energy (CPU-measured: np4 4.18 kWh, np8 4.93 kWh) is not yet
contested end-to-end on GPU. The MF-phase win (12–29× energy, faster than 16 nodes) is measured and stands.

---

## §V.14 — DOES CTF FAVOUR OVERFITTING UNDER BIC? — the rigorous answer (the IQ-TREE panel's main concern, 2026-06-15)

**Synthesis of a 2-agent research workflow (statistics/phylo literature sweep + adversarial red-team, both code- and
literature-grounded) + a first-principles decomposition + a decisive real-data demonstration on the AA-1M coarse
table. Author as1708; multi-agent synthesis by Claude Opus 4.8.**

> **The panel's concern, verbatim in spirit:** *"Does the coarse-to-fine procedure favour overfitting — selecting
> models that are too complex — because BIC is being applied to a subsample?"*
>
> **The one-paragraph verdict.** The concern is **correct about the coarse RANKING and wrong about the OUTPUT.** A
> subsample-rescaled fit term genuinely amplifies a model's overfitting "optimism" by ≈(N/m)·(k/2) ∝ k, so a
> *projected* coarse BIC tilts toward complex models — and ours **once did, demonstrably** (it demoted the true
> winner LG+G4 to rank 4 behind LG+R5/+I+R5; §X.5.5). **But the CTF output is chosen by EXACT full-data BIC over the
> fully re-optimised top-k** — the identical number stock ModelFinder computes — so coarse optimism is *discarded*
> and **cannot make the output over-complex.** A subset-minimisation of an exact criterion can never invert the BIC
> ordering of two retained candidates. The *only* way CTF can differ from full ModelFinder is by **screening the true
> winner out of the top-k (a recall miss)** — and the coarse complexity bias pushes that miss toward dropping the
> *simpler* model (under-fitting), the **opposite** of the panel's fear. We further (i) **replaced the projected gate
> with the native subsample BIC** (penalty `k·ln m`, un-amplified), which restores LG+G4 to rank 1, and (ii) note the
> method is algebraically **ModelTamer** (Sharma & Kumar 2022, *MBE*), peer-reviewed with **≥99 % recall and no
> documented complexity bias**. Net: **no over-fitting bias at the output; manage recall, not overfitting.**

### V.14.1 The two-level decomposition — the whole argument in one move

Let 𝒞 = the ~120-model candidate set; T = the fixed tree (from the subsample); `BIC_full(M)` = the exact full-data
BIC of model M re-optimised on all N sites (the live pipeline literally computes `bic = -2*lnL + p*ln(N)`,
`run_ctf_1m_mf_energy.sh:142`, with `lnL` the full-N re-optimised value — bit-for-bit what stock ModelFinder uses).
CTF outputs `argmin_{M ∈ 𝒮} BIC_full(M)` where 𝒮 ⊆ 𝒞 is the coarse top-k shortlist. Stock ModelFinder outputs
`W = argmin_{M ∈ 𝒞} BIC_full(M)`. Two exhaustive cases:

- **Case A — W ∈ 𝒮 (true winner shortlisted).** Then `argmin` over 𝒮 ⊇ {W} returns **exactly W**, because
  `BIC_full(W) ≤ BIC_full(M)` for every M ∈ 𝒮 by definition. **CTF output == stock output, identically.** A complex
  model can sit in 𝒮 beside W, but the exact full-data BIC rejects it precisely as stock ModelFinder does. *Restricting
  the candidate set cannot make BIC prefer a complex model it would otherwise reject.*
- **Case B — W ∉ 𝒮 (true winner screened out).** Then CTF outputs W′ ≠ W with `BIC_full(W′) > BIC_full(W)`. **This is
  the only way CTF differs from stock**, and whether W′ is more or less complex than W is not decided by BIC here — it
  is whatever survived the coarse screen.

**Therefore "CTF favours overfitting" requires CTF to *output* an over-complex model when the simpler one was in the
refine set — impossible by Case A. The entire concern reduces to: *can the coarse screen drop the true winner?* — a
one-sided RECALL question.** And §V.14.2 shows the coarse bias makes that miss fall toward the *simpler* model, i.e.
toward **under**-fitting, not over-fitting.

### V.14.2 The coarse ranking IS complexity-biased — the optimism math, and our documented failure

The panel's mechanism is real and classical. A k-parameter model's maximised log-likelihood on its *own* fitting
sample is upward-biased ("optimism") by ≈ **k/2** in lnL units — **constant in sample size** (Akaike 1973; Efron 2004
*JASA*; Hastie–Tibshirani–Friedman *ESL* §7.4). The old **projected** coarse criterion
`BIC' = −2·(N/m)·lnL_sub + k·ln N` multiplies the subsample lnL by N/m — which correctly rescales the *signal*
(`(N/m)·E[lnL_sub] ≈ E[lnL_full]`) **but also amplifies the optimism to ≈ (N/m)·(k/2)**, ∝ k, while the penalty
`k·ln N` is not amplified. A parameter is then spuriously favoured in the coarse rank whenever `N/m ≳ ln N` — at our
operating point `N/m = 10⁶/5000 = 200 ≫ ln N ≈ 14`, so the amplified optimism *dominates* the penalty. **The coarse
projected ranking is materially over-fitting-prone — established, not speculative.**

**This is not hypothetical — it fired, and we caught it (§X.5.5, jobs 170728179/182).** Re-run *live* on the real
AA-1M coarse table (122 models, m = 5000 sites, N = 10⁶; k recovered as `AIC/2 + lnL`):

| Rank | **NATIVE gate** `−2·lnL_sub + k·ln m` (CURRENT) | **PROJECTED gate** `−2·(N/m)·lnL_sub + k·ln N` (OLD, superseded) |
|---:|---|---|
| 1 | **LG+G4** (k=198) ✅ the true full-data winner | LG+I+G4 (k=199) |
| 2 | LG+I+G4 (k=199) | LG+R5 (k=205) |
| 3 | LG+R4 (k=203) | LG+I+R5 (k=206) |
| 4 | LG+I+R4 (k=204) | **LG+G4 (k=198)** ← true winner DEMOTED |

Under the projected gate the true winner LG+G4 is pushed to **rank 4**, leapfrogged by **+1, +7 and +8-parameter**
models (LG+I+G4, LG+R5, LG+I+R5) — **3 textbook overfit inversions**, exactly the (N/m)·k/2 amplification. Under the
**native gate** (which penalises by `ln m`, *not* amplified) LG+G4 returns to **rank 1** and the +R ladder sits
correctly below. **The fix is shipped:** the coarse gate is the native subsample BIC, in both
`run_ctf_1m_mf_energy.sh` and the benchmark sweeps (which rank on IQ-TREE's own `n=m` BIC column). *(This corrects
§V.4, which still printed the projected `BIC'` formula — see the note there.)*

### V.14.2b The n=30 multi-seed confirmation — native gate is robust, projected gate fails *catastrophically* on the DNA ladder (job 171258771, 2026-06-16)

§V.14.2's table is n=1. The decisive precursor experiment proposed in §V.14.7 has now **run to completion** —
`run_ctf_overfit_recall_sweep.sh` on **real 100K alignments** (AA simulated under LG+I+G4; DNA under GTR+I+G4),
computing the full-data `-m MF` BIC oracle and then ranking the candidate set under **both** gates across
**m ∈ {1000, 2000, 5000} × seeds {1…5} = 30 runs** (CPU reference binary, BIC is optimiser-invariant; exit 0, 36 min,
6.6 GB). For each run we record the oracle winner, each gate's top-1, recall@3 of the oracle winner, and whether the
gate's top-1 has **more parameters** than the oracle (the "over-fit-top1" flag):

| Dataset (oracle) | gate | **recall@3** | **over-fit-top1** | what the gate's top-1 actually was |
|---|---|---:|---:|---|
| **AA** (LG+G4, k=198) | **NATIVE** (shipped) | **15/15** | **0/15** | LG+G4 every time |
| AA (LG+G4) | projected (old) | 15/15 | 1/15 | LG+G4 ×14, LG+I+G4 ×1 (the lone AA overfit, +1 param) |
| **DNA** (F81+F+G4, k=201) | **NATIVE** (shipped) | **15/15** | **1/15** | F81+F+G4 ×14; one TPM2u+F+G4 (k=203) at m=5000 — **+2 params, and still recalled F81+F+G4 in top-3** |
| DNA (F81+F+G4) | projected (old) | **0/15** | **15/15** | **never the true winner** — GTR+F+G4 ×7, GTR+F+I+G4 ×2, TIM/TVM/TIM2/TPM3u/TPM2u-family ×6 |

**Three things this nails down:**

1. **The shipped native gate is robust on both data types:** recall@3 = **30/30**, over-fit-top1 = **1/30** — and that
   single native "over-fit" (DNA, m=5000) is a **+2-parameter within-family neighbour (TPM2u+F vs F81+F) that *still
   recalled the true winner into the top-3*** → the exact full-data BIC refine then demotes it and outputs F81+F+G4.
   The screen-then-clean safety (§V.14.3) is observed working, not just argued.

2. **The projected gate fails exactly where the model family has a complexity ladder — and DNA is the worst case.**
   On AA fixed-Q matrices (no exchangeability ladder, only +I/+F/+G to bolt on) the projected gate overfits only 1/15.
   On **DNA it is catastrophic: 0/15 recall, 15/15 over-fit** — the GTR exchangeability ladder
   (JC→K80→HKY→TN→TIM→TVM→GTR, ±I, ±F) gives the amplified (N/m)·k/2 optimism a continuum of nested richer models to
   climb, so the gate climbs it *every single time* and never even keeps the true F81+F+G4 in the top-3. This is the
   live, multi-seed realisation of the §V.14.5 prediction that **exchangeability/+R ladders are the genuine adversarial
   lever** — measured, not theorised.

3. **The mechanism is doubly illuminating because of *which* model is truth.** The DNA data is *generated* under
   GTR+I+G4, yet the full-data **BIC** oracle is the *simpler* **F81+F+G4** (GTR's exchangeabilities buy too little lnL
   to pay their `k·ln N` at this length — the same "generative ≠ BIC-selected" finding as G.6.2). So the projected gate
   chases the **generative** model (GTR-family) while the actual selection criterion (full-data BIC) wants the simpler
   one — and **only the native gate agrees with BIC.** The panel's fear ("CTF favours overfit models") is precisely the
   *projected* gate's behaviour, and precisely *not* the native gate we ship.

This upgrades §V.14.2's single demonstration to a 30-run, two-data-type result with a clean separation
(native 30/30 vs projected 15/30, dominated by the DNA 0/15). Summary at
`/scratch/rc29/as1708/iqtree3-gpu/ctf_overfit_recall/RECALL_SUMMARY.tsv`.

### V.14.2c — THE +R-FAVOURING regime, the last open gap, now CLOSED (job 171466576, 2026-06-17)

§V.14.2b's data was all **+G-winning** (AA→LG+G4, DNA→F81+F+G4) — the native gate never had to *recall a genuine +R
winner*, which §V.14.7/8 named as the one untested regime (and the one the panel's fear actually targets:
under-fitting away from a true +R model). This sweep closes it. We **simulated** (AliSim) 100K alignments under
strongly **bimodal FreeRate** generative models that unimodal gamma structurally cannot fit, so the full-data BIC
oracle is a genuine +R-family model:

- **AA** — `LG+R4{0.45,0.1, 0.05,0.4, 0.05,1.6, 0.45,1.9}` (two well-separated rate clusters at 0.1 and 1.9)
- **AAI** — `LG+I{0.2}+R3{0.5,0.2, 0.1,1.0, 0.4,2.0}` (genuine invariant fraction + bimodal variable rates)
- **DNA** — `GTR{2,5,1.5,1.2,4.5}+F{...}+R4{…bimodal…}`

**Precondition verified before spending the job (not vacuous):** on the AA bimodal data even at 10K sites, LG+R4
beats LG+G4 by **2,847 nats (ΔBIC ≈ −5,648)** despite +5 params — and since the lnL-gain/penalty ratio only grows
with N, +R wins by a landslide at 100K. For each regime, m ∈ {1000,2000,5000} × seeds {1..5} = 15 cells; per cell we
rank under both gates and compute the **actual CTF output** = `argmin BIC_full` over the coarse top-3 (the exact
refine, computed from the oracle's own full-data BIC — non-circular, since the coarse top-3 is built independently on
the subsample), plus an **under-fit** flag (CTF output simpler than oracle), an **over-fit** flag, and a rate-het
**class-downgrade** flag (oracle +R-family but CTF drops to +G/+I).

| regime | oracle (full-data BIC) | cells | native recall@3 | **CTF output correct** | under-fit | over-fit | class-down |
|---|---|---:|---:|---:|---:|---:|---:|
| AA (LG+R4 bimodal) | **LG+I+R2** | 15 | 15/15 | **15/15** | **0** | **0** | **0** |
| AAI (LG+I+R3 bimodal) | **LG+I+R2** | 15 | 15/15 | **15/15** | **0** | **0** | **0** |
| DNA (GTR+R4 bimodal) | **GTR+F+I+R2** | 15 | 15/15 | **15/15** | **0** | **0** | **0** |

**45/45 perfect: the shipped native gate recalls the genuine +R winner every time, zero under-fit, zero over-fit,
zero class-downgrade.** Three honest observations:

1. **The predicted under-fit never materialises.** §V.14.5 warned the one residual risk was data where +R *legitimately*
   wins, where a small subsample could under-determine the rate categories and drop the genuine +R winner *toward
   parsimony*. It does not happen even at m=1000 (1% of patterns): recall@3 = 15/15 in every regime.
2. **Screen-then-clean observed, not just argued.** The oracle is **+I+R2** (not the generated +R4 — "generative ≠
   BIC-selected" again, cf. G.6.2: the big slow rate-cluster is captured as an invariant + 2 free rates), and in
   several cells the *coarse* top-1 is the simpler `LG+R2`/`TVM+F+I+R2`, yet the +I+R2 oracle is always in the coarse
   top-3, so the exact-BIC refine recovers it — the Case-A finish (§V.14.1) working on live +R data.
3. **The projected gate also scored 45/45 here — and that is consistent, not a contradiction.** The projected gate's
   *failure* mode is promoting complexity when the truth is **simple** (the DNA 0/15 of §V.14.2b). Here the truth IS
   the complex +I+R model, so the projected bias is harmless — it cannot over-shoot past the actual winner. So this
   sweep is *not* a native-vs-projected discriminator (that was §V.14.2b's job on +G data); it is the **under-fit
   safety proof** for the shipped native gate on the adversarial +R regime — which is exactly what was missing.

Net: the last honest gap in §V.14.7/8 is closed — **on genuinely +R-favouring data CTF neither under-fits nor
over-fits; recall@3 = 45/45.** Summary at
`/scratch/rc29/as1708/iqtree3-gpu/ctf_freerate_recall/FREERATE_RECALL_SUMMARY.tsv`; harness
`gadi-ci/gpu-modelfinder/run_ctf_freerate_recall_sweep.sh`.

### V.14.3 Why the bias does NOT reach the output — screening theory + the exact-BIC finish

This is the architecture the statistics literature calls **screen-then-clean**, and CTF matches its "safe" pattern:

- **Sure Independence Screening (Fan & Lv 2008, *JRSS-B* 70:849)** — a cheap, biased, noisy screen is admissible
  *provided it does not drop the truth* ("sure screening property"); an exact criterion then finishes on the reduced
  set. **Over-inclusion is the designed-for, benign direction.**
- **Screen-and-clean (Wasserman & Roeder 2009, *Ann. Statist.*)** — the screen is allowed to return a *supermodel* of
  the truth w.p.→1; the clean stage removes the excess.
- CTF's coarse complexity bias **over-includes complex models** into 𝒮 (the safe direction), and the exact full-data
  BIC refine **discards all coarse optimism** for shortlisted models — a model that "looked good" coarsely gets its
  true, un-inflated `BIC_full` and loses if it is genuinely over-parameterised. **BIC is selection-consistent and
  anti-overfitting at full N** (Schwarz 1978; Haughton 1988); in phylogenetics it is *the* recommended
  anti-overfitting criterion (Posada & Buckley 2004, *Syst. Biol.*; Luo et al. 2010, *BMC Evol. Biol.*;
  Kalyaanamoorthy et al. 2017, *Nat. Methods* — "BIC consistently outperformed AIC/AICc in identifying the true
  model"). **Consistency is preserved at the output as long as recall holds.**

### V.14.4 The decisive prior art — CTF's coarse criterion is published, and validated

**ModelTamer (Sharma & Kumar 2022, *MBE* 39:msac236) is algebraically the same method.** It subsamples site patterns
then **up-samples** (resamples with replacement back to N sites) and runs standard ModelFinder/BIC — which multiplies
each pattern's lnL contribution by ≈ N/m and keeps the full-N penalty: **identically** `−2·(N/m)·lnL_sub + k·ln N`,
reached by data replication instead of an explicit factor. Their empirical findings answer the panel directly:
**≥99 % recall** of the full-data optimal model at ≥0.5 % subsample, **no systematic complexity bias**, and — the key
inversion — their dominant failure was **UNDER-fitting** (raw subsampling without upsampling gave 12 % accuracy
because the substitution signal collapsed); the N/m rescaling is the *cure* for under-fitting. **The practical hazard
of subsample model selection is under-fitting, and the rescale fixes it — the opposite of the panel's worry.**

### V.14.5 The phylo overfitting traps, and the one place to stay vigilant

- **+I+G near-non-identifiability** (Sullivan & Joyce 2005; Yang 2006; Jia et al. 2014) — p_inv and α are confounded
  along a flat ridge. **Susko & Roger 2020 (*MBE* 37:549): near boundaries a model "has less freedom than its
  parameter count suggests" — its *effective* DoF < nominal k.** Since optimism ∝ effective DoF, the +I+G / boundary-+I
  cases are **partially self-protected** (less amplified than naive k predicts); and the confounding *keeps the simple
  twin +G in the shortlist*, which *helps* recall. Benign-to-helpful for output correctness.
- **+R (FreeRate, 2(k−1) params)** is the genuine adversarial lever: near-nominal effective DoF ⇒ **maximal**
  (N/m)·k/2 amplification, small inter-model BIC gaps. It caused the one documented projected-gate failure (§X.5.5).
  The native gate contains it on tested data; the residual risk is **data where +R *legitimately* wins** — there a
  small subsample under-determines the rate categories and could drop a genuine +R winner *toward parsimony*
  (under-fitting). Again the wrong direction for "overfitting," but the one regime to test.
- **The 17,618-nat between-family cliff is the structural safety net.** Coarse noise can scramble only the
  *within-family* near-ties (top-3 within ΔBIC ≤ 264); it cannot promote a wrong-matrix-family model (O(0.2 nat/site)
  away — signal swamps noise at any m ≥ a few hundred). So the worst realistic native-gate miss is a within-LG-family
  swap (e.g. output LG+I+G4 instead of LG+G4) — **+14 BIC, a single dead pinv≈0 parameter — cosmetic, not a fit
  catastrophe.**

### V.14.6 Safeguards — from "cannot over-fit" to "cannot mis-recall"

| # | Safeguard | Status | Closes |
|---|---|---|---|
| **S1** | **Native-BIC coarse gate** (`−2·lnL_sub + k·ln m`), never the projected BIC | ✅ **shipped** | the entire (N/m)·k/2 amplification — the only observed overfitting-looking failure (§X.5.5) |
| **S2** | **Adaptive-k / ΔBIC-band refine** — refine every coarse candidate within a band Δ (≪ the 17,618 cliff) of the coarse leader | ⬜ recommended | Case-B recall miss: guarantees the true winner is refined if it is anywhere near the coarse leader → output reverts to exact-BIC-correct |
| **S3** | **Force-include the simplest competitive model per rate-class** in the refine set | ⬜ recommended | the "three rich models crowd out the bare winner" miss — the simple model is refined *by construction*, then exact BIC demotes any dead-parameter complex twin |
| **S4** | **Recall certificate** — certify from the coarse ΔBIC(rank-1→rank-(k+1)) vs a one-sided `z·σ_δ·√(N/m)` bound that no lower candidate can overtake at full N | ⬜ optional | upgrades "recall=1.0 empirically" to "recall=1.0 with certificate, else widen k" |
| **S5** | **Rate-het detector + per-model wall budget** (never silently exclude an ineligible +R/+I leader; cap each refine) | ✅ shipped | the +R-spurious-promotion **time bomb** (CPU-at-1M timeout) AND the under-fit miss of a genuine +R leader |

**S1 + S3 give a provable-non-overfitting statement:** with the un-amplified native gate *and* the simplest
competitive model per rate-class force-refined, the output is `argmin BIC_full` over a set guaranteed to contain the
simplest within-family candidate; since exact full-data BIC demotes any dead-parameter complex model (it did so to +I
in *every* test), **CTF's output cannot be more complex than full ModelFinder's, except in the cliff-excluded
measure-zero event of a wrong-family flip.**

### V.14.7 The defensible statement to the panel + the one decisive experiment

**Statement:** *"CTF's coarse ranking can be complexity-biased — and demonstrably was, via the lnL-rescaling
projection, which we caught (it demoted the true LG+G4 to rank 4 behind +R5/+I+R5) and replaced with the native
subsample BIC. But that bias cannot make CTF output an over-fit model: the final selection is the exact full-data BIC
over fully re-optimised candidates — the same number ModelFinder itself uses — which rejects dead parameters
regardless of how the shortlist was built. The only way CTF can differ from full ModelFinder is by failing to
shortlist the true winner, and our coarse bias pushes toward dropping the *simpler* model, not selecting a complex
one. So the concern is a recall question about parsimony, not an overfitting question about fit; we bound it with the
native gate, the 17,618-nat between-family cliff, top-k ≥ 3, and (recommended) adaptive-k + a forced simplest model
per rate-class. The algebraically identical published method, ModelTamer, reports ≥99 % recall with no complexity
bias."*

**The single decisive experiment (the one untested regime):** run CTF end-to-end (native gate) on a **genuinely
rate-heterogeneous alignment simulated under a true +Rk model where +R legitimately beats +G at full data** — the only
regime where the coarse stage could either drop a genuine +R winner (under-fit) or be tempted to promote one
spuriously. Measure top-3 recall of the full-data BIC winner across ≥5 subsample seeds at m=5000, and whether the
output ever differs from a full `-m MF` reference. **Recall = 1.0 there closes the last honest gap; recall < 1.0
localises the failure to shortlist construction (fixed by S2/S3), and the output stays non-overfitting by
construction.** ✅ **The cheap precursor — a multi-seed native-vs-projected recall sweep on the existing AA/DNA data
(m ∈ {1000, 2000, 5000} × 5 seeds) — has now run (§V.14.2b, job 171258771): the shipped native gate scored 30/30
recall@3 with 1/30 trivial over-fit, while the old projected gate collapsed to 0/15 recall on the DNA exchangeability
ladder.** ✅ **The decisive experiment itself has now ALSO run (§V.14.2c, job 171466576): on genuinely +R-favouring
data — AA `LG+R4`, AAI `LG+I+R3`, DNA `GTR+R4`, all strongly bimodal so the full-data BIC oracle is a true +I+R-family
model — the shipped native gate scored recall@3 = 45/45 with ZERO under-fit, ZERO over-fit, ZERO class-downgrade, and
the exact-BIC CTF output matched the +R oracle in all 45 cells. The under-fit direction the panel feared does not
materialise even at m=1000.** The last open recall check is closed.

### V.14.8 Established vs uncertain (intellectual honesty)

**Established:** optimism ≈ k/2, constant in n (Akaike 1973; ESL §7); rescaling amplifies it to ≈(N/m)·k/2 (direct
algebra, **and demonstrated live** on the AA-1M table); BIC consistent/anti-overfitting at full N (Schwarz 1978;
Kalyaanamoorthy 2017); biased screen + exact finish is safe, over-inclusion benign (Fan & Lv 2008; Wasserman & Roeder
2009); the equivalent method (ModelTamer) gives ≥99 % recall, no complexity bias, under-fitting the real hazard
(Sharma & Kumar 2022); effective DoF < nominal for +I+G (Susko & Roger 2020); **and — now measured at n=30 (§V.14.2b)
— the shipped native gate recalls the full-data BIC winner 30/30 with 1/30 trivial over-fit, while the projected gate
collapses to 0/15 on the DNA exchangeability ladder, confirming both the amplification mechanism and the native fix on
real data.** **Uncertain:** no paper analyses the explicit `−2(N/m)lnL+k·ln N` estimator's optimism directly (the
derivation here is sound first-principles, **though the n=30 sweep now empirically matches its prediction**, not a
cited result for this exact form); worst-case recall for near-degenerate +R ladders under subsampling is not formally
bounded **(though the genuinely-+R-favouring generative regime — once the last untested gap — is now measured at
45/45 recall@3, §V.14.2c)**; ModelTamer's 99 % is empirical, its one miss ΔBIC<10; the fixed-subsample-tree ×
model-ranking interaction is uncharacterised (a separate risk, orthogonal to overfitting).

---

## §V.15 — The architectural boundary: multi-locus / partitioned data (where CTF+JOLT is the WRONG tool — said plainly)

**The one-line answer to the panel.** *If the dataset is many small loci (a partitioned / multi-gene analysis), one
GPU is the wrong tool and a CPU cluster already wins — by design, not by accident. CTF+JOLT is built for the opposite
shape: one massive single alignment (or a few genome-scale blocks) that a cluster cannot cheaply split.*

**Why — the one reason that matters (it is just our own depth-vs-breadth verdict, applied to loci).** Model selection
on K loci is K *independent* per-locus selections. A cluster hands one whole locus to each node — embarrassingly
parallel, ~100 % efficient. One GPU is **mutex-serialized** (per-model speedup S≈4.8), so it does the loci one after
another. This is the same `N/S ≈ 21×` serialization that makes one GPU lose the 122-model *single-alignment* race
(§V.2), now multiplied by the number of loci. The GPU's edge is **depth** (a deep per-model optimisation), and
multi-locus is the purest **breadth** problem there is (§V.11) — the cluster's home turf.

**CTF specifically adds nothing for small loci.** CTF's whole job is to subsample a *huge* N down to ~5000 sites. A
gene partition is *already* ~500–2000 sites, so there is nothing to subsample — CTF degenerates into ordinary
ModelFinder plus a redundant re-fit. The value proposition is simply vacuous when N is already small.

**Two tempting arguments to AVOID (they are wrong, and a panel will dismantle them):**
1. *"~400 s of coarse overhead per partition."* **False.** The measured coarse cost (≈158–467 s) is for a **5000-site,
   100-taxon, ~122-model** pass — not a small gene. CTF uses `min(5000, L)` sites, so a 500-site locus is far cheaper,
   and (per above) CTF doesn't even help it. The honest objection is "no benefit," **not** a 400 s-per-gene tax.
2. *"Blended-signal error if partitions aren't declared."* That is garbage-in/garbage-out — **every** tool, the stock
   CPU one included, fits one model to an undeclared concatenation. It is user error, not a CTF-specific failure, so it
   does not define our boundary.

**The boundary is precise: it is *many small loci*, not "partitioning."** A few **large** partitions (e.g. 3–4
whole-genome blocks of ~1M sites) are just 3–4 single large alignments — each is exactly what CTF+JOLT is for, and the
cluster has only 3–4 loci to spread, so the GPU's per-locus depth can compete. The failure regime is breadth (many
tiny loci), not the word "partition."

**The honest non-claim.** There *is* a possible GPU direction for many small loci — batching them concurrently on an
under-occupied device (the saturation-inversion idea, §V.3). It is **unbuilt**, and it would still have to beat the
cluster's near-perfect locus parallelism, which is a high bar — so we **do not claim** multi-locus.

**Scope honesty.** No partitioned experiment was run; the project's measured data is all single-alignment. The
statement above is architectural reasoning grounded in the *measured* `N/S` serialization and depth-vs-breadth physics
(§V.2/§V.11), not a partition benchmark. A quick empirical confirmation (concatenate K genes; CTF-per-partition vs the
cluster's locus-parallel run) would make it bulletproof-measured, but the argument already follows from the banked
physics.
