# PART III — PHALANX-BMF: Cross-Model Batched GPU ModelFinder for IQ-TREE 3

**Author:** as1708 (lead-architect synthesis by Claude Opus 4.8, 2026-06-08)
**Status:** ARCHITECTURE + IMPLEMENTATION PLAN. Builds on the validated G.0→G.2.1b foundation
(standalone K1/K2/K3/K4/K5 + in-tree clean-room lnL/df/ddf bit-parity). Supersedes the
single-model-serial assumption behind G.2.2.
**Scope:** one GPU first (multi-GPU deferred); AA/DNA; 100K → 10M patterns; Gadi V100/A100/H100.

> **Provenance note (honesty).** This plan was synthesized by a multi-agent workflow (3 independent
> architecture proposals + adversarial reviews) grounded in the validated G.0→G.2.1b facts. The 8 parallel
> deep-research sub-agents (GPU math libraries, dispatch, tuning, optimizers, batching, literature, codebase,
> scaling) FAILED on a session/usage limit, so §III.5's library/literature conclusions are reasoned from
> arithmetic-intensity first principles + the design-doc's prior 9-agent literature sweep, NOT a fresh search
> — flagged for enrichment (cuSOLVER/cuTENSOR/tensor-core specifics, Gangavarapu btae030 batching numbers).
> The architecture itself is driven by the measured profiling facts, not the literature, so the gap does not
> change the design — but the §III.5 library verdicts should be re-checked when the research phase is re-run.

> **One-sentence thesis.** IQ-TREE's ModelFinder is a stream of B independent candidate models that
> share topology + patterns + tip states and differ only in eigensystem + category rates/props; the
> validated single-model GPU kernels lose to CPU at 100K not because the GPU is slow but because one
> model is **register-capped at 25 % occupancy AND re-sweeps statelessly at ~38 ms/derivative**. PHALANX-BMF
> attacks **both** with one coherent engine: a **warp-cooperative kernel restructure** that lowers the
> 128-reg ceiling so occupancy can actually rise, a **cross-model `grid.z` batch** that supplies the
> independent warps to fill it (where the grid is *not* already block-saturated) and amortises the host
> serial tail, and an **intra-NR-burst device-resident theta cache** that restores the CPU's ~1 ms
> evalAt — all behind the proven stateless bit-parity baseline, with regime-aware routing so the honest
> speedup is reported as a **curve** and the decisive single-GPU win lands at 1M/10M where HBM bandwidth
> dominates SPR DRAM.

---

## III.1 Executive summary + the named architecture

**Name: PHALANX-BMF** (Persistent-residency, HARdware-ALigned, cross-model-batched eNgine for the
X-candidate set — Batched ModelFinder). A phalanx because the independent candidate models advance
shoulder-to-shoulder across the device, and the deep serial branch-opt ladder that no single model can
parallelise is widened B-fold by the models beside it.

The architecture has **five composable layers**, each independently testable and each gated on a concrete
number against an existing oracle:

1. **Warp-cooperative state-distributed kernels (K1c/K2c).** The validated K1/K2 put one thread per
   pattern and *loop the 20 states inside the thread* — the `prod[NS]` 20-double working set is what makes
   the kernel irreducibly 128 regs/thread (Block Limit Registers = 2 → 25 % occupancy; profile job
   170195112, confirmed by the K5 `__launch_bounds__` sweep job 170195272 where every register cap
   spilled and ran slower). The restructure distributes the NS=20 inner matvec across cooperating threads
   (warp-shuffle reduction over states), cutting registers/thread so Block Limit Registers rises from 2 toward
   3–4. **This is the lever g1-log:389-391 named as "what a real occupancy win would need," and it is the
   prerequisite that makes batching able to add resident warps rather than just queued blocks.**

2. **Cross-model batch axis `grid.z = model` (K1b/K2b).** B independent candidates co-resident; each
   carries its own eigensystem/rates/brlen/theta in `[model][...]` device arrays and reads ONE shared
   tip-state + `ptn_freq` buffer. This is the centerpiece and the genuinely novel architecture — FCA's
   model-level MPI dispatch realised as on-device **data** parallelism. **Honest scoping (per the
   reviews): the occupancy win from `grid.z` is real only where the grid is not already block-saturated
   — the deep-ladder serial tail (33 single-node levels at 100K), small alignments, and combined with the
   K1c register drop. Where the grid is block-saturated (wide shallow levels at 100K), `grid.z`'s value is
   host-tail amortisation + scheduling-latency amortisation, not occupancy.** The two layers (1)+(2)
   together are what convert the latency-bound regime.

3. **Intra-NR-burst device-resident theta cache + generation-counter coherence.** The stateless design
   (G.2.1b, bit-parity rel 0.0) is the correctness floor but pays ~38 ms/evalAt → 4.7× slower than 1 CPU
   thread on `-te`. The validated K2 already shows theta-cached evalAt = **1.21 ms** (job 170188743). We
   restore that cache *within one `optimizeOneBranch` NR burst* (2–5 steps, the window where
   `theta_computed` legitimately stays true), keyed by a **per-tree `clearAllPartialLH` generation
   counter** that plugs the one staleness hole G.2.1 found. We do NOT cache theta across the whole
   optimiser (the `clearAllPartialLH`-per-Brent-trial frequency makes that a silent-staleness trap;
   reviews of Proposals 2+3 are correct).

4. **Batched optimiser driver + retire-and-compact + filterRates batch pruning.** A host control plane over
   B live `IQTree` instances drives `optimizeParameters`/`optimizeAllBranches` with per-edge B-wide
   K1b/K2b launches but per-model independent `minimizeNewton` decisions (the batch fuses only kernel
   launches, never an optimisation decision — this is the bit-parity anchor). Models converge at different
   iteration counts → retire-and-compact keeps `grid.z` dense; refill from the cheapest-first candidate
   queue. **Batch membership respects the candidate dependency graph: independent across substitution
   families, sequential within a +R ladder** (must-fix from Proposal 3).

5. **Regime-aware router + library-accelerated math.** A host cost-model routes GPU vs CPU and chooses B
   from pattern count, NCAT, and live VRAM; pattern tiling raises B at 1M/10M. cuSOLVER batched eigendecomp
   and cuBLAS/CUTLASS are adopted **only where measured to beat the hand kernels** — which, per the
   arithmetic-intensity analysis below, is NOT the NS=20 inner contraction. The speedup is reported as a
   curve vs pattern count, never one number.

**What is novel vs validated vs speculative (stated up front, honesty discipline §III.8):**

| Element | Status |
|---|---|
| Cross-model `grid.z` batch axis for independent candidate models on one GPU | **NOVEL** (no precedent: BEAGLE is one-model-per-instance; FCA is across CPU ranks) |
| Warp-cooperative state distribution to break the 128-reg ceiling | **NOVEL for this kernel** (lever named but not built; g1-log:390) — **HIGH-RISK, the K5 sweep shows the naive register knob backfires** |
| Eigen-space lnL/df/ddf math, +R10-one-pass, native-20, FP64-unscaled NORM_LH, four-pointer seam, stateless bit-parity | **VALIDATED** (G.0→G.2.1b) — built on, not re-claimed |
| Intra-NR-burst theta cache | **VALIDATED standalone** (K2 1.21 ms); the in-tree coherence wiring is **new + medium-risk** |
| 1M/10M HBM-bandwidth dominance | **PHYSICALLY GROUNDED** (HBM 900 GB/s V100 / 2 TB/s A100 vs SPR 300–400 GB/s); tiling kernel **UNBUILT** |
| Occupancy actually rising with B at 100K | **SPECULATIVE** — the make-or-break Nsight gate (G.3.0); the grid is already block-saturated at 100K, so this is conditional on the K1c register drop opening block slots |

---

## III.2 Diagnosis — why stateless single-model loses at 100K −m TEST

Three measured facts, none of which the centerpiece may contradict.

**(a) The kernel is register-capped at 25 % occupancy — and the simple knob backfires.** The decisive
profile (ncu, job 170195112) on the production g4 `k1_node` launch:

```
(391, 1, 1) x (256, 1, 1)      Registers Per Thread = 128
Block Limit Registers = 2      Block Limit Warps = 8     Block Limit SM = 32
Theoretical Occupancy = 25 %   Achieved = 23.85 %        Waves Per SM = 2.44
Compute (SM) = 36 %   Memory = 48 %   DRAM = 16–40 %   L1 hit = 71–83 %   (NONE saturated)
```

The K1/K2 thread map is **one thread per pattern**, with the 20 states and the NCAT categories looped
*inside* the thread (`gpu_k1_lnl.cu:165` `int ptn = blockIdx.x*blockDim.x + threadIdx.x`; the `double
prod[NS]` array at `k1_body:133`). That 20-double working set + the matvec accumulators + the 9 child
pointers make 128 regs **irreducible by `__launch_bounds__`**: the K5 sweep (job 170195272) A/B'd every cap
{85,64,51,42} regs and *all spilled and ran slower* (g4 base 37.8 ms; LB256/4 @64reg = 55.4 ms; LB256/6 @42reg
= 305.8 ms). Conclusion (g1-log:386-391): the 25 % diagnosis is right but the register knob is the wrong
fix; **a real win needs an algorithmic restructure that distributes the NS=20 states across cooperating
threads so the per-thread footprint genuinely shrinks.**

**(b) `grid.z = model` alone does NOT lift occupancy at 100K — and the reviews are right about this.**
`Waves Per SM = 2.44` means the (391,1,1) grid already places 391 blocks over 80 SMs × 2-resident = 160
slots, i.e. ~2.4 waves deep. The SMs are **block-saturated**; adding a model plane multiplies *queued*
blocks (391·B) but the per-SM resident warp count is still pinned to 2 blocks × 8 warps = 16 of 64 (25 %)
by the register limit, which is invariant to `grid.z`. **Therefore batching's occupancy benefit at the
production 100K count is contingent on layer (1) first opening block slots (K1c register drop → Block Limit
Registers 2→3/4 → more resident warps per SM), OR on the workload not being block-saturated** — which is
exactly the deep-ladder serial tail and small alignments. This is the single most important correction to
the three proposals and it reshapes the gating (G.3.0 is the kill-switch).

**(c) The dominant cost is the stateless re-sweep, not launches.** G.2.1b measured the full `-te`
branch-opt at **GPU 1063 s vs CPU 225 s (4.7× slower)** because each NR `evalAt` re-runs a full ~38 ms
postorder sweep where the CPU does ~1 ms cached-theta. CUDA graphs (K3) gave wall **parity** (1.00–1.01×):
the 98-dependent-kernel chain is GPU-side scheduling-latency-bound, which a graph collapses on the *host*
submission side but not on the device. Fusion (K4) is **wash-to-loss at 100K** (r8 0.96×, r10 0.94×)
because the real tree is a **height-42 ladder** — only ~8 shallow levels batch (L0–L8), then L9–L41 are a
33-deep single-node serial tail no same-depth strategy can parallelise (g1-log:322-327). Branch re-opt is
75–85 % of per-model wall (Trimorph:142).

**(c′) Direct MF-loop confirmation (G.2.2a, job 170265661, 2026-06-08).** A real GPU `-m TESTONLY -mrate G`
(all-GPU +G4 candidates, NO +I fallback) scored only **6 models in 2.5 h before walltime kill (~25 min/model)**
— each candidate runs a full `optimizeParameters` (≈ the 17.7-min `-te` cost). Extrapolated, the full 224-model
`-m TEST` on the stateless path is **tens of hours vs the CPU's 221.6 s — a ~50–100× wall miss, not 4.7×.**
**Crucially, the 6 completed models are BIT-IDENTICAL to the CPU baseline** (LG+G4 −lnL 7541976.853 == baseline,
ranked #1 by BIC 15086233.266; LG+F+G4 7541999.323 exact; WAG+G4 7602067.428; JTT+F+G4 7650982.369 rel 4e-10),
so **model selection in the real MF loop is already correct — the entire problem is throughput, not correctness.**
This is the most direct evidence that the architecture's job is to convert the latency-bound serial per-model
optimiser into a saturated throughput engine, exactly what layers C1/C2/C3 target.

**Synthesis of the diagnosis.** The wall has three independent components and the architecture must hit all
three:
- **C1 — per-evalAt cost** (38 ms stateless vs 1 ms CPU): fixed by the **theta cache** (layer 3), already
  validated at 1.21 ms. This is the biggest single lever and needs no batching.
- **C2 — 25 % occupancy on the sweep** (the partial-LH recompute that seeds theta): fixed by **K1c
  register drop** (layer 1) + **`grid.z` batch** (layer 2) where slots open.
- **C3 — the 33-deep serial ladder tail** (per-kernel ~85 µs scheduling latency × 33 levels, untouchable by
  fusion or graphs within one tree): fixed ONLY by the **`grid.z` batch** — at a single-node ladder level,
  B models give B independent blocks that co-schedule, the one axis fusion provably could not touch.

The 1M/10M regimes are different: there a single model is already SM/bandwidth-saturated (K4 found 100K
already saturates), so C2 is moot, C1 still matters, and the win is **raw HBM bandwidth** over SPR DRAM —
the regime where one GPU can beat 16 CPU nodes.

---

## III.3 The batched-model engine in detail

### III.3.1 Data layout with the model dimension

Device-resident, model-outermost so a model's arena is contiguous and a retired model is simply not
launched:

```
d_partial[model][slot][cat*NS + state][ptn]     // postorder partials, eigen coordinates
d_theta  [model][cat*NS + state][ptn]            // ONE edge's theta (the active NR edge), not a 98-edge arena
d_brlen  [model][edge]                           // device-resident branch lengths (K3)
d_echild [model][node][cat][NS*NS]               // rebuilt on-device from d_brlen/d_eval/d_U
d_U, d_Uinv, d_eval [model][...]                 // per-model eigensystem (tiny: 20x20+20 = 3.3 KB/model)
d_catRate, d_catProp [model][NCAT]               // per-model rate categories (tiny)
--- SHARED across the whole batch (uploaded once per topology) ---
d_tip[ntax][ptn]   (compact states, ~9.6 MB)     d_ptn_freq[ptn] (~0.8 MB)
```

**Critical correction (must-fix, Proposals 1+3): the "shared arena uploaded once" saves ~10 MB, three
orders of magnitude below the multi-GB per-model partials. VRAM is ~100 % per-model partials and scales
with B.** The shared buffer is a coalescing/H2D convenience, NOT a VRAM lever. The d_theta is ONE edge's
worth (the validated K2 caches only the active NR edge's `theta = node_eig ⊙ dad_eig`, `gpu_k2_derv.cu:245`
`d_theta` sized `slotSz`), ~0.12–0.31 GB/model — NOT a 98-edge arena (the Proposal-2/3 review correctly
caught that IQ-TREE's `theta_all` is a single `block_size` buffer, `phylotree.cpp:939`). So theta caching
does **not** double the per-model footprint.

Warp-within-one-model invariant: `blockIdx.z = model`, never `threadIdx`. A warp therefore spans patterns
of ONE model with ONE eigensystem → no intra-warp divergence across eigensystems (must-keep from all three
proposals; correct and necessary).

### III.3.2 Batched eigendecomposition + on-device echild rebuild

Per-model rate matrices are eigendecomposed on the **host** (the validated default; IQ-TREE already does
this in `modelmarkov.cpp` `decomposeRateMatrix`, consumed π-folded as `U=getEigenvectors()`,
`Uinv=getInverseEigenvectors()`, `eval=getEigenvalues()` per the G.2.0a bridge, machine-eps validated). For
the batch, B host eigendecomps run per param update — **this is a host serial tail and is gated as a hard
<20 %-idle Nsight Systems check at G.3.1** (must-fix). cuSOLVER `syevjBatched` (FP64) is the fallback only
if that idle fraction is exceeded, and only after re-validating FP64 device-eigendecomp parity vs the host
(deferred until measured-necessary). `echild = U·exp(eval·rate_c·t)` is rebuilt **on-device** per model
from `d_brlen` (K3 `build_echild`, ULP-clean V0 with the FP-grouping `len = brlen·catRate` first), batched
over models in one launch — never round-trips the host.

### III.3.3 The postorder/derivative kernels gaining `grid.z = model`

**K1b (batched postorder lnL).** The K1 body is unchanged numerically; the launch gains `blockIdx.z =
model`. Each plane reads its own `d_echild[model]` / `d_eval[model]` and writes its own
`d_partial[model]`. Root reduction is per-model (deterministic block-local pairwise reduction → per-model
`d_lnL[model]`, ptn_freq-weighted, FP64 unscaled — the G.1.3 deterministic reduction, NOT atomics). Bit-parity
per model is inherited from K1 (rel ~1e-12 vs G.0 oracle).

**K2b (batched single-edge df/ddf).** `grid.z = model`; per-model `theta[model]` cached;
per-model `val0/val1/val2` (NCAT·NS doubles each, tiny) for that model's current branch length; per-model
triple-dot reduction → per-model `{lh, df, ddf}`. df/ddf written **un-negated** (`computeFuncDerv`
negates downstream, `phylotree.cpp:2566`). **Must-fix (Proposal 3): replace the standalone K2's `3×
cudaMemcpyToSymbol + 3× D2H + host reduction` (`gpu_k2_derv.cu:274-281`) with an on-device batched
reduction returning B×{lh,df,ddf} in one D2H** — the symbol-copy/D2H path is latency-bound and will not
scale to B models. The per-model `val` arrays move from `__constant__` (which cannot hold B planes) to a
small `__global__` array indexed by `blockIdx.z`.

**K1c/K2c (warp-cooperative variant — layer 1, the occupancy restructure).** Instead of one thread holding
`prod[NS=20]`, a **cooperating group of threads** (e.g. a 4- or 8-thread sub-warp, or the full warp across
a small pattern tile) splits the state dimension: each thread owns NS/g states, the matvec
`pk[x] = Σ_i echild[x][i]·L_child[i]` is computed with a warp-shuffle reduction over the partitioned `i`,
and `prod[x] *= pk[x]` folds across children. This cuts the per-thread `prod[]` from 20 doubles to NS/g,
shrinking the register footprint so Block Limit Registers rises. **HIGH-RISK and explicitly speculative:**
the K5 sweep proves the compiler cap backfires; this is a genuine source restructure whose payoff is
uncertain (warp-shuffle adds latency; the matvec is small). It is gated standalone at G.3.0 BEFORE any
batch/integration sunk cost — if K1c does not raise achieved warps/SM net of shuffle overhead, layer 1 is
cut and the architecture falls back to "batching helps only the serial tail + host-tail amortisation +
1M/10M bandwidth," which is still a defensible (smaller) win.

### III.3.4 Per-model streams + retire-and-compact + filterRates batch pruning

**Streams.** B per-model CUDA streams overlap the host eigendecomp/echild-rebuild of model m+1 with kernel
execution of model m's plane, hiding the host serial tail. The batched launches themselves are single
grid.z kernels on one stream (co-scheduling is the point); streams are for host/device overlap.

**Retire-and-compact.** `optimizeParameters` runs ~22 outer iterations and DIFFERENT models converge at
different counts. A model hitting `logl_epsilon` is retired (its `grid.z` plane dropped); the batch is
compacted (live models renumbered dense) so launches never waste SM cycles on done models. When live count
< refill threshold, the collector tops up from the cheapest-first queue. **Honest caveat (review of all
three): a batch that decays B=8→2 re-exposes the C2/C3 problems; the refill+compaction is the mitigation
and its effective-utilisation must be measured (G.3.3 gate), not assumed.**

**filterRates batch pruning — corrected per the Proposal-1 must-fix.** `filterRates`
(`phylotesting.cpp:3035`) hard-returns until the *entire* reference substitution family (including the
+R2..+R10 chain) is `MF_DONE` (`if (!at(model).hasFlag(MF_DONE+MF_IGNORED)) return`). So **the +R long-pole
cannot be pruned before it is evaluated.** Consequence for batching:
- **The independent batch axis is ACROSS substitution families** (LG, WAG, JTT, … base + their +G4 rows) —
  the ~20–40 truly independent models, NOT 224. This is the real B.
- **Within a +R ladder the models are SEQUENTIALLY DEPENDENT** (+R[k+1] is initialised from +R[k] via
  `initFromCatMinusOne` / `restoreCheckpointRminus1`, `phylotesting.cpp:2183/2103`, and the ladder is
  BIC-pruned on +R[k]'s score). So a +R family is a B=1 serial tail INSIDE the batch, not a co-resident
  block. The heavy +R8/+R10 long-pole is therefore the least-batchable, most-VRAM-hungry case — exactly
  inverting where batching helps. **This is honestly the centerpiece's weakest reach and is why the 1M/10M
  bandwidth play (which does not need batching) carries the decisive single-GPU win.**

filterRates/filterSubst pruning + cheapest-first ordering run on the CPU control plane *between* batches,
exactly as today, so the `MF_IGNORED` table must match CPU bit-for-bit (G.3.4 gate). The entire `_IQTREE_MPI`
LPT dispatch / cost-predictor cohort layer is deleted (single GPU = no dispatch).

### III.3.5 B vs VRAM tradeoff + pattern tiling

Per-model partial arena (measured, native-20, FP64, compact tips, AA-100K = 96017 ptn, 98 internal;
`gpu_k5_occ.cu` VRAM print confirms r10 = 14.93 GB):

| Model | NCAT | Arena/model | V100-32GB (≈28 usable) | A100-80GB (≈76 usable) |
|---|---|---|---|---|
| g1 | 1 | 1.78 GB | B=15 | B=42 |
| g4 | 4 | 6.16 GB | B=4 | B=12 |
| r8 | 8 | 12.0 GB | B=2 | B=6 |
| r10 | 10 | 14.93 GB | B=1 | B=5 |

**The router's B-budget uses the theta-inclusive arena** (partials + one-edge theta + per-model echild +
scratch), NOT the bare partial figure, or it OOMs at the exact B it targets (must-fix). The +G4 families —
the bulk of the candidate set and the eventual LG+G4 winner — batch B=4 on V100 / B=12 on A100, which is
where the latency-kill is both needed and feasible. r10 cannot batch on V100 (B=1) — confirming the
inversion above; r10 batches only on A100 (B=5) or via tiling.

**Pattern tiling (the B-recovery lever + the 1M/10M enabler).** lnL is a per-site sum, so stream pattern
blocks through a fixed per-model buffer set; the SAME captured K3 graph replays per tile with swapped
tip/partial pointers (composes cleanly with graph replay, design II.4). VRAM = B × tile_arena, so **tiling
trades tile-loop iterations for batch width** — it recovers B at 100K for heavy +R, AND makes AA-1M
(~63 GB native-20 single model, fits A100 at B=1) and AA-10M (~630 GB, tiling mandatory) feasible.
**Honest: no tiling kernel exists yet (K1–K4 are whole-arena); the 1M/10M curve is UNBUILT and is a
phase G.5 deliverable, not an asserted number** (must-fix, Proposal 1).

---

## III.4 Device-residency + the optimizer redesign

The optimiser's **numerics are IQ-TREE's own** — this is the non-negotiable correctness anchor (G.2.1b:
full `-te` GPU lnL = CPU = −7541976.8530, rel 0.0, all 197 optimised brlen worst_rel 0.0). The redesign
changes only the **granularity** of work submitted to the GPU and the **caching** of theta within a branch.

### III.4.1 Theta-reuse + the clearAllPartialLH generation counter

The stateless clean-room sweep (G.2.1b) is the shippable correctness floor and the A/B baseline every
faster path must match bit-identically. The speedup lever:

- **Cache theta ONLY within one `optimizeOneBranch` NR burst.** `theta_computed` is set false per branch
  (`phylotree.cpp:2642`); `minimizeNewton` (`optimization.cpp:422`) then does 2–5 NR steps on that edge.
  Across those steps the partials are unchanged → theta is valid → each `evalAt` is the **1.21 ms** K2
  triple-dot (job 170188743), not a 38 ms re-sweep. This is the validated window; it needs no cross-call
  device state.
- **Plug the one staleness hole with a per-tree generation counter.** `clearAllPartialLH`
  (`phylotree.cpp:683`, fired on alpha/rate change via `rategamma.cpp:182`) clears partials but does NOT
  touch `theta_computed`. **Critical correction (reviews of Proposals 2+3): `clearAllPartialLH` is a
  per-`PhyloTree` method and fires per Brent alpha TRIAL, dozens of times per model — and each candidate is
  a DISTINCT `IQTree*` (`phylotesting.cpp:1955` `new IQTree(in_aln)`).** So the counter must be a member of
  each tree instance and the device theta cache must be keyed by **(tree-identity, edge, generation)**, not
  by a single global `this->theta_computed`. A missed per-model invalidation silently corrupts ONE model's
  df/ddf with no crash — and that model could be the BIC winner. Therefore:
  - The theta cache is invalidated whenever the tree's generation counter advances OR the active edge
    changes — which, given the per-Brent-trial frequency, means **in practice theta survives only the NR
    burst, exactly the validated window.** We do NOT attempt cross-Brent-trial reuse.
  - **The A/B bit-identical gate (cached path == stateless path, rel 0.0) runs on EVERY GPU-eligible model
    class, not a 3-model sample** (the failure is data-dependent, firing only when alpha/rate moves
    mid-optimisation; must-fix).

### III.4.2 Tree-edge graph-coloring (Jacobi branch updates) — OPTIONAL, gated, likely-fails

`optimizeAllBranches` is Gauss-Seidel: `optimizeOneBranch` reads the previous branch's freshly-updated
partials (`phylotree.cpp:2763`). A 2-coloring of the edge-adjacency graph would let non-adjacent
(independent-theta) edges update in the same Jacobi wave — the ONE lever that attacks the 33-deep serial
ladder tail *within* a model. **But this CHANGES the convergence order and will generically converge to a
different branch vector, failing the rel-0.0 bit-parity gate** (reviews of Proposals 2+3 are correct). It
is therefore **explicitly speculative, gated, and NOT counted toward the wall**: engaged only if it passes
a bit-identical-optimum A/B test, otherwise the exact Gauss-Seidel order is kept (already bit-identical).
The real serial-tail fix is the cross-model batch (C3), which widens the tail B-fold WITHOUT changing any
model's convergence order. We rely on C3, not coloring.

### III.4.3 Batched line-search + the device-side / conditional-graph loop

**Batched cross-model Gauss-Seidel.** The driver visits branch index e across all B live models together:
K1b re-sweeps the dirty path for all models (theta-cached within the burst), then `minimizeNewton` runs
**per-model** but its `computeFuncDerv` calls K2b once for edge e across the batch. Each model gets its own
df/ddf, its own safeguarded NR step (Newton vs bisection branch, `optimization.cpp:450`), its own step
count. **Must-fix (Proposal 3): models cannot share a step schedule** — model A may need 1 NR step on edge
e while B needs 5 and takes a bisection branch. So the batch is **decoupled at the optimisation decision**:
the batch shares only the kernel launch (one K2b serving B models' CURRENT edge with per-model t), and a
model that has converged on edge e contributes a masked no-op. Lockstep over `minimizeNewton` is explicitly
rejected; the B-fold throughput comes from the kernel-launch fusion + occupancy, not from forcing models
into a shared step count. **Effective batch utilisation is a measured gate (G.3.3), not an assumption.**

**Conditional-graph / persistent-kernel device-side loop — DEMOTED to explicitly-speculative, out-of-band.**
A CUDA-12.4 conditional-WHILE node driving the inner NR/Brent loop on-device would remove per-iteration host
chatter (extending the K3 device-resident-brlen win), but `minimizeNewton` has data-dependent
Newton/bisection control flow and the alpha Brent calls host `clearAllPartialLH`+`computeLikelihood` per
trial (`rategamma.cpp:182`) — porting that control loop on-device is open research, NOT a phase deliverable
(reviews of Proposal 2 are correct). The default device-side loop is **host-driven K3 graph replay per
outer iteration** (validated capability), with the conditional-WHILE node verified by a 3-line Gadi toy
*before* any wiring (design II.5). We do not bank wall on it.

### III.4.4 Correctness argument: it lands on the same optimum

The chain of bit-parity is:
1. **Per-model lnL/df/ddf are bit-identical to CPU** (G.2.0a rel 1.235e-16; G.2.1a df rel 3.99e-12 / ddf
   4.54e-15; G.2.1b full `-te` rel 0.0 / brlen worst_rel 0.0). The `grid.z` batch is B independent copies
   of these validated reductions — each model's plane is numerically isolated (own echild/eval/rates, own
   per-model deterministic reduction), so K1b/K2b reduce to B invocations of the proven K1/K2 (G.3.0 gate:
   each model's batched lnL == its standalone oracle rel ≤ 1e-12).
2. **The theta cache is bit-identical to the stateless sweep** within the NR burst (A/B gate, all model
   classes), so caching changes nothing numerically.
3. **Each model's `minimizeNewton` decisions are IQ-TREE's own, unchanged** — the batch fuses only kernel
   launches, never an optimisation decision. So each model walks the identical NR sequence it would walk
   serially, reaching the identical optimum (G.2.1b standard, now under batching: per-model converged lnL
   rel ≤ 1e-9, brlen vector rel ≤ 1e-6).
4. **FP64, deterministic block-local pairwise reduction, no atomics, no fast-math, never TF32/FP16 on the
   reduced lnL** (the likelihood surface's terraces demand FP64; syag017). Graph-coloring (which WOULD
   change the optimum) is gated out unless it passes bit-parity.

The architecture therefore lands on the same optimum **by construction**: the only things the batch shares
are read-only (topology, tips, ptn_freq) or per-model-private (eigensystem, brlen, theta, NR decisions).

---

## III.5 Library-accelerated math + kernel tuning — honest about where libraries do NOT help

**Where libraries help:**
- **cuSOLVER `syevjBatched` (FP64)** for B eigendecompositions — ONLY as the fallback if the host
  eigendecomp serial tail exceeds the 20 %-idle gate (G.3.1). FP64 device-eigendecomp parity vs host must
  be re-validated before adoption. Host is the validated default.
- **CUDA Graphs (K3)** — collapses the 104→1 host API calls and holds brlen device-resident; the
  *structural* integration unblock (validated), composes with tiling. Not a standalone speedup at 100K
  (parity), but the batch makes one `cudaGraphLaunch` advance B models, which is where its host-side
  collapse finally pays.

**Where libraries do NOT help (decisive, measured):**
- **cuBLAS `DgemmStridedBatched` / CUTLASS / cuTENSOR for the NS=20 inner contraction.** The contraction
  `pk[x] = Σ_i echild[x][i]·L_child[i]` is a skinny M=K=NS=20 matvec with arithmetic intensity ~2.5
  FLOP/byte regardless of the pattern (N) tile — it is **bandwidth-bound, not compute-bound**, exactly the
  regime where FP64 tensor cores (syag017, AA ~2–3× but memory-bandwidth-saturated, HIGH-AI-only) give no
  upside, and where design II.3 already found the hand kernel beats `cublasDgemmStridedBatched` at this
  size. **The hand kernel (K1c) is primary; the library is at most a fallback.** Also, GEMM reorders the
  FP64 reduction (split-K, tensor-core path) → would break the rel-0.0 bit-parity bar to ~1e-12 — a policy
  regression we explicitly reject for the reduced lnL (must-fix, Proposal 3: do not silently relax parity).
- **Note on the eigen-space structure (must-fix, Proposal 3):** per node it is NOT one clean two-GEMM. It
  is (one skinny matvec per child) → **element-wise Hadamard `prod[x] *= pk[x]` across children**
  (`gpu_k1_lnl.cu:147`) → a second skinny matvec for the Uinv re-projection. The Hadamard/prod-fold is a
  non-GEMM step a library cannot absorb; the hand kernel already fuses it.

**Kernel tuning that IS the lever:**
- **Warp-cooperative state distribution (K1c, layer 1)** — the only validated-as-named path to break the
  128-reg ceiling (g1-log:390). HIGH-RISK; gated standalone.
- **Per-leaf-edge P-matrix lookup** — for leaf children, replace the 400-FMA `echild·Uinv[:,s]` matvec with
  a precomputed `P[x][s]` gather (g1-log:391); cuts both registers and FLOPs on the ~half of edges that are
  leaf edges. Lower-risk than the warp-shuffle restructure; build it first within layer 1.
- **Native-20 (no 32-pad)** — 37.5 % VRAM+bandwidth saving on a bandwidth-bound kernel (validated); the
  binding constraint at scale.
- **Pattern-innermost coalesced layout** — retained (validated 128-B coalesced loads, L1 hit 71–83 %); rules
  OUT shared-mem echild staging (broadcast-served, already L1-resident).

---

## III.6 Regime-aware routing + the honest speedup curve

Report the speedup as a **curve vs pattern count, never one number** (USER DECISION 2, design II.7). The
router picks (a) GPU vs CPU and (b) batch size B from `nptn`, `nstates`, NCAT/+R profile, and live VRAM
(nvml).

**Routing thresholds:**
1. **GPU vs CPU:** route GPU iff `nptn > KNEE` (the saturation knee, measured early against the single-AA
   AVX-512 baseline; expect ~2×10³–5×10³ for AA-20) AND the model is GPU-eligible (reversible, 4/20-state,
   no +I/+ASC/mixture/site-specific/non-rev/multifurcating; the validated G.2.0b gate). Below the knee,
   CPU-SSE wins and the router sends it to CPU.
2. **Batch size B:** `B = min(floor(VRAM_budget / theta_inclusive_arena(nptn,NCAT)), B_cap)`, NCAT-aware —
   low-NCAT/+G4 batch deep (most latency-bound, most to gain, fit), high-NCAT/+R batch shallow/single
   (VRAM-limited, sequentially-dependent ladder). `B_cap` ~ the occupancy-saturation factor measured at
   G.3.0.
3. **Tiling** engages when `theta_inclusive_arena > VRAM_budget` (AA-1M+), recovering B via the tile loop.
4. **GPU contention from existing concurrency (must-fix, Proposal 1):** the openmp_by_model path
   (`evaluateAll`) runs candidates across OMP threads today; under `--gpu` we **force serial `test()` /
   disable `openmp_by_model`** so the batch is the single GPU consumer — multiple host threads must never
   contend for one device's state. The CPU-fallback candidates may run on spare CPU threads concurrently
   with GPU batches (they touch no device state).

**The honest curve (three regimes):**

| Regime | CPU reference (measured) | GPU expectation | Confidence |
|---|---|---|---|
| **100K** (96017 ptn, −m TEST gate) | vanilla 264.2 s; **R1+R2+AVX512 221.594 s** (the honest 103-thread floor); FCA-np2 149.256 s (2 nodes) | C1 (theta cache, 1 ms evalAt) closes most of the 4.7× stateless gap; +C2/C3 (K1c + batch) target sub-221.6 s. **RISKIEST number** — if K1c does not raise occupancy and the batch only amortises the host tail + serial ladder, 100K may land parity-to-modest-win and the value is the scale regimes. | **Coin-flip on V100; plausible on A100. Self-flagged as the riskiest claim.** |
| **1M** (−m MF; CPU vanilla np1 ~126 min, **FCA-np16 MF 18.71 min on 16 nodes**) | One model already SM/bandwidth-saturated → batching less critical (B=1–2 via tiling); the **HBM ratio (A100 2 TB/s vs SPR 300–400 GB/s) is the win**, +R10 in one pass (no Amdahl pole). One A100 finishing −m MF in finite time already beats the timeout-prone CPU; tiled-B should approach/beat the 18.71 min 16-node wall on ONE GPU. | **The strongest, most defensible claim.** |
| **10M** (CPU: FCA-np16 −m MF would be days; AA-1M FCA-np16 −m MF TIMED OUT at 3 h) | Tiling mandatory, B=1, pure HBM-bandwidth. Any correct finite result is a win; **1 GPU beating 16 CPU nodes is the headline target.** | **GPU-dominance territory; bounded by single-GPU VRAM/tiling.** |

**How batching shifts the knee LEFT:** at small/medium nptn a single model underfills the device (the
latency-bound regime); B models supply the independent work that pushes the GPU-vs-CPU crossover to lower
pattern counts than a single-model port would. But — honestly — the shift is bounded by the block-saturation
fact (§III.2b): at 100K the wide shallow ladder levels are already block-saturated, so the leftward shift
there depends on K1c. The clean leftward shift is in the deep-ladder tail and mid-nptn range.

---

## III.7 Phased implementation plan (G.3.x) — each phase independently testable, strict dependency order

Numbering continues from the completed G.2.1b. Every gate is a concrete number against an existing oracle.
**Standalone phases (G.3.0–G.3.2) come BEFORE any phylotesting.cpp restructure** — the hot-loop restructure
is the riskiest change in the program and must not start until the microbench proves the throughput is real
(reviews of all three).

| Phase | Deliverable | Gate (concrete number vs oracle) | Novel / Risk |
|---|---|---|---|
| **G.3.0 — Occupancy kill-switch (standalone): K1c warp-cooperative + grid.z** | Add (a) per-leaf P-matrix lookup, then (b) warp-cooperative state distribution to `gpu_k1_lnl.cu`; add `grid.z=model` (B copies of LG+G4 + B distinct families LG/WAG/JTT, one shared topology/tips/ptn_freq). | **(1) Bit-parity: each model's lnL == standalone K1 / G.0 oracle rel ≤ 1e-12** (g4 −7541976.9391, r10 −7554280.5776 one-pass). **(2) Nsight: at nptn=100K, B≥2, achieved warps/SM and occupancy RISE above the 25 %/16-warp baseline (Block Limit Registers ≥ 3), net of warp-shuffle overhead. (3) per-model effective lnL time drops monotonically with B until VRAM/saturation.** If (2) fails on V100, layer-1 is CUT and the centerpiece value falls back to serial-tail + host-tail + 1M/10M. | **NOVEL + HIGHEST-RISK. The go/no-go before any sunk cost. The K5 sweep is the warning: the naive knob backfired.** |
| **G.3.1 — Batched K2b + on-device reduction + host-tail bound (standalone)** | `grid.z` K2b (per-model theta cache, per-model val0/1/2 in global mem, **on-device batched reduction → B×{lh,df,ddf} in one D2H**); B host eigendecomps + echild rebuild on B streams. | **(1) Per-model df/ddf FD-validated** (df rel ≤ 2e-9, ddf rel ≤ 5e-6, matching K2). **(2) Batched evalAt for B=8 < 2 ms total** (< 0.25 ms/model effective vs 1.21 ms single). **(3) Nsight Systems: GPU idle fraction from B host eigendecomps + scheduler < 20 %** (the host serial tail hard gate). | NOVEL batch reduction; MEDIUM-risk (host tail). |
| **G.3.2 — Batched graph + decoupled optimiser driver + retire/compact (standalone)** | Capture batched postorder+derivative as one CUDA graph; drive B independent `minimizeNewton`-shaped sweeps DECOUPLED (per-model step counts, masked converged edges) with retire-and-compact + refill. | **(1) For B=8 perturbed-brlen sweeps: each model's converged lnL + brlen vector bit-identical (|dt|=0) to its single-model K3 sweep** (extends G.1.3 V6). **(2) Compaction never changes a live model's result.** **(3) Effective batch utilisation measured + reported** (the lockstep-divergence honesty gate). | NOVEL scheduler; MEDIUM-risk. |
| **G.3.3 — In-tree batched optimiser (theta cache + generation counter)** | Extend the G.2.1b stateless overrides to a B-batch over B live `IQTree` instances; theta cache gated by the **per-tree `clearAllPartialLH` generation counter**; A/B vs stateless. | **For a batch of ≥3 +G4 models (LG+G4, WAG+G4, DNA GTR+G4): each converged lnL matches CPU rel ≤ 1e-9 AND 197-brlen rel ≤ 1e-6** (G.2.1b standard, now batched). **A/B cached==stateless rel 0.0 on EVERY GPU-eligible model class.** Batched per-model MF time < the 4.7×-slower serial-GPU baseline by ≥ the occupancy factor. | NOVEL; HIGH-risk (coherence). |
| **G.3.4 — Batch collector in phylotesting.cpp + router (full −m TEST AA-100K)** | Restructure `CandidateModelSet::test` into batch-collect/evaluate; batch axis ACROSS subst families, +R ladder as in-batch B=1 serial tail; filterRates/filterSubst preserved between batches; per-candidate CPU fallback; force serial test() / disable openmp_by_model under --gpu; strip MPI dispatch; router selects B + GPU/CPU. | **Best model == LG+G4; displayed lnL == CPU rel ≤ 1e-12 (−7541976.86); `MF_IGNORED` filterRates table matches CPU bit-for-bit; identical AIC/BIC top ranking; IQTREE_GPU=OFF + --gpu-off byte-identical. MF wall < 221.594 s (the honest R1+R2+AVX512 floor); stretch < 149.256 s (FCA-np2 2-node).** | NOVEL + HIGH-risk (hot-loop restructure). |
| **G.3.5 — Scale regime: heavy +R, then AA-1M/10M pattern tiling + honest curve** | Build the pattern-TILING kernel (same captured graph per tile); +R2..+R10 single-pass (high-NCAT shallow/single B); AA-1M native-20 (~63 GB A100); produce the curve vs pattern count. | **(1) Tiled lnL == untiled bit-parity where untiled fits.** **(2) −m MF AA-100K < FCA np=1 1341 s, ideally < np=2 481 s. (3) AA-1M −m MF finishes on ONE A100 (vs SPR FCA-np16 3 h timeout); tiled-B approaches/beats 18.71 min 16-node wall. (4) curve published, never one number.** | NOVEL tiling (UNBUILT); the decisive single-GPU win. |

**Dependency order (strict):** G.3.0 (kill-switch) → G.3.1 → G.3.2 are standalone and gate the bet
cheaply. G.3.3 needs G.3.0+G.3.1+G.3.2. G.3.4 needs G.3.3. G.3.5 needs G.3.4. **If G.3.0(2) fails, do NOT
proceed to integration on the occupancy thesis** — re-scope to the serial-tail + 1M/10M value and go
straight to a B=1-large-pattern G.3.5.

**Toolchain (unchanged from II.9):** `module load cuda/12.5.1 gcc/12.2.0 cmake/3.24.2 eigen/3.3.7
boost/1.84.0`; configure+build inside a PBS job on gpuvolta/dgxa100 (login has no nvcc/icpx);
`CMAKE_CUDA_HOST_COMPILER=g++` (all-GCC host) for the GPU lib; `CUDA_ARCHITECTURES "70;80;90"`; drop
`-static`; guard the GPU lib out of `BUILD_LIB`. Dev tree `/scratch/rc29/as1708/iqtree3-gpu` branch
`gpu-kernel`; setonix-iq branch `gpu-modelfinder`. NOTHING committed until a gate passes.

---

## III.8 Top risks + honesty discipline

**Non-negotiable discipline (carry from the whole program):**
- **FP64 parity is non-negotiable.** Never TF32/FP16/fast-math on the reduced lnL. Deterministic
  block-local pairwise reduction, no atomics. The bit-parity bar is rel 0.0 for per-model branch-opt and
  rel ≤ 1e-12 for lnL vs oracle; we do NOT silently relax to library-GEMM 1e-12 (Proposal-3 must-fix).
- **FD-validate every gradient** vs the CPU oracle at every step — a wrong-but-plausible 20-state kernel is
  easy to ship (BEAGLE proved it; design II.10). Build-gating.
- **Measure GPU idle %** (Nsight Systems) at every batched phase; the host eigendecomp/scheduler serial
  tail is a hard < 20 % gate (G.3.1).
- **Report speedup as a curve** vs pattern count, never one headline number; route small alignments to CPU.
- **Keep the CPU path byte-identical** (`#ifdef IQTREE_GPU` + `--gpu` gate; the G.2.0b/G.2.1b invariant) and
  a per-candidate CPU fallback for every unsupported model (+I/+ASC/non-rev/mixture/site-specific).

**Top risks (ranked, with mitigation and honest UNVALIDATED flags):**

1. **Occupancy may not rise with B at 100K (the centerpiece's core bet).** The grid is already
   block-saturated (Waves/SM 2.44; Block Limit Registers 2). Batching adds resident warps ONLY if K1c first
   opens block slots — and the K5 sweep shows the naive register knob backfires. **MITIGATION:** K1c is
   gated standalone at G.3.0 with an Nsight warps/SM number as the kill-switch; if it fails, re-scope to
   serial-tail + host-tail + 1M/10M (still defensible, smaller). **UNVALIDATED — the make-or-break gate.**
2. **The +R long-pole is sequentially dependent and least-batchable.** filterRates can't prune it before
   evaluation; +R[k+1] inits from +R[k]; r10 is B=1 on V100 (14.93 GB). **MITIGATION:** batch across subst
   families (real B ~20–40, not 224), +R ladder as in-batch B=1 serial tail; lean on tiling (A100) + the
   1M/10M bandwidth win where +R does not need batching. **The centerpiece's weakest reach — stated.**
3. **Theta-cache coherence under batching = silent-staleness trap.** Per-tree `clearAllPartialLH`
   generation counter must be tracked across B DISTINCT `IQTree*` and keyed by (tree, edge, gen); a missed
   per-model invalidation corrupts one model's df/ddf with no crash. **MITIGATION:** restrict reuse to the
   NR burst (validated window); A/B bit-identical gate on ALL model classes (data-dependent failure), not a
   sample. **MEDIUM-HIGH risk.**
4. **B live `IQTree` instances = B host object graphs co-resident.** Each candidate is `new IQTree(in_aln)`
   with its own central_partial_lh + ModelFactory + checkpoint (`phylotesting.cpp:1955`); B models = B×~6 GB
   HOST RAM (fine on 512 GB nodes, but the batched driver must thread B trees through one device context, a
   real `evaluate()` lifecycle rewrite). **MITIGATION:** the batch path is additive behind `--gpu`; CPU
   evaluate() stays intact for fallback. **The most invasive change; gated at G.3.4.**
5. **VRAM/host-RAM budget optimistic if mis-counted.** Router must use the theta-inclusive, both-direction
   arena (not the 6.16 GB single-plane figure) or OOM at the target B. **MITIGATION:** budget formula uses
   the measured-inclusive arena + nvml live-free admission control.
6. **CUDA graph / fusion gave parity, register caps backfired — prior intuition-defying results.** Every
   prior speedup lever came back parity-or-worse. **MITIGATION:** G.3.0/G.3.1 standalone gates kill the bet
   early; the architecture is structured so the cheap microbench, not the narrative, decides integration.
7. **Conditional-WHILE device-side loop maturity / graph-coloring optimum-shift** — both DEMOTED to
   speculative/out-of-band; not counted toward the wall. Default is host-driven K3 graph replay + exact
   Gauss-Seidel order.
8. **Single-GPU only:** 10M and wide batches bounded by one device's VRAM; the 10M regime leans entirely on
   tiling (serialised tile loops) — beats a CPU cluster on accessibility/cost, not necessarily total
   throughput. Multi-GPU deferred.

**Bottom line.** The validated foundation (G.0→G.2.1b) is rock-solid: the math, the seam, the bit-parity,
the kernels. The novel architecture is cross-model batching as on-device data parallelism — the right
direction and the only untried lever consistent with the decisive profile. The brutal honesty: the 100K
−m TEST gap-closing is a coin-flip gated on the K1c occupancy restructure (the naive knob already backfired
once); the +R long-pole is the least-batchable case; and the decisive, defensible single-GPU win is the
1M/10M HBM-bandwidth regime, carried by bandwidth more than by batching. The phasing is built so G.3.0 kills
the occupancy bet cheaply before the invasive phylotesting.cpp restructure, and so the value never collapses
to zero: even if layer 1 fails, the serial-tail batch + theta cache + 1M/10M bandwidth deliver a real win.
