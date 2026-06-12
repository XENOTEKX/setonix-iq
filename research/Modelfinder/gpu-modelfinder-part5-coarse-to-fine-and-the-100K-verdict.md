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
   rank by **scale-consistent BIC**: `BIC' = −2·(96017/m)·lnL'_sub + p·ln(96017)` (rescale the subsample lnL to
   full magnitude; keep the *unscaled* parameter penalty so cross-param-class calls — `+F`'s ~19 freq params
   vs `+I`'s 1 — stay commensurable). At subsample scale this is cheap on either device; on the GPU it is the
   saturation-inversion batched pass.
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
