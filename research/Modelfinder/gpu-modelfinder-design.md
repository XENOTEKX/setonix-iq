# GPU ModelFinder — Single-GPU, CUDA-Graph, BEAGLE-Kernel Design for IQ-TREE 3

**Author:** as1708 (design synthesis by Claude Opus 4.8)
**Date:** 2026-06-01
**Status:** Phase **G.0 (de-risk) PASSED** (2026-06-01) — standalone BEAGLE harness on a V100 computes
AA-100K likelihoods at **bit-parity with the CPU plugin** (and 1e-8 of IQ-TREE for LG+G4) across
**LG+G4, LG+F+I+G4, LG+R8, and LG+R10** (the last via a bit-exact 5+5 category-split), at **43–66× the
CPU-SSE eval rate** (e.g. LG+G4 45 ms vs 2973 ms/lnL), 10–20 GB VRAM. **Key findings:** double precision
needs **no scaling** on this tree (unscaled = scaled, ~2× faster); **compact tip states** cut VRAM/time;
BEAGLE 4.0.1 CUDA **hard-caps NCAT≤8** for 20 states (rescale-grid limit) → +R9/+R10 need category-split
or a BEAGLE rebuild. **The +R10 long-pole — which killed every CPU dispatch architecture — runs ~57× on
one GPU.** **Gradient (2026-06-06):** the Ji-et-al O(N) pre-order branch-length gradient — the exact piece
Mode-L overflowed at 10⁵⁴ — is **FD-validated correct on the CPU plugin** (g4 rel 2.8e-3, single-rate g1 rel
1.1e-7), confirming the *algorithm*. **But BEAGLE 4.0.1's CUDA gradient is unusable for protein (20-state)
models**: even with BEAGLE's own `hmctest` transpose recipe the A100 gradient is wrong for *all* NCAT
(NCAT=1 fails too ⇒ not a category issue, it is the 20-state CUDA pre-order/edge-derivative kernel; `hmctest`
only validates 4-state nucleotides). The GPU gradient *infrastructure* runs (A100-80GB, 50.3 GB, ~100–240 ms,
~100–150× CPU) — only the values are wrong. ⇒ production GPU gradient needs a **fixed/newer libhmsbeagle or a
custom CUDA kernel** (integration phase), NOT off-the-shelf 4.0.1. lnL (dominant cost) is proven on GPU, so
the bet holds; proceed toward G.1 using the lnL path. Full record:
[gpu-modelfinder-g0-log.md](gpu-modelfinder-g0-log.md).

> **➡ 2026-06-07 — the project now moves from the standalone BEAGLE de-risk to a CUSTOM in-tree GPU
> CUDA kernel for IQ-TREE 3 ModelFinder. The full phase-by-phase implementation plan — grounded in a
> 9-agent codebase+literature+failure research sweep — is [PART II](#part-ii--custom-gpu-cuda-kernel-implementation-plan-2026-06-07) at the
> end of this doc. Headline reframe: IQ-TREE optimises branches one-at-a-time by Newton-Raphson
> (`computeLikelihoodDerv`, a single-edge df/ddf), NOT the Ji O(N) all-branch pre-order gradient — so the
> broken-BEAGLE pre-order kernel and O(depth) pre-order buffer recycling are an OPTIONAL advanced track,
> not the critical path. GPU dev tree: `/scratch/rc29/as1708/iqtree3-gpu` (branch `gpu-kernel`, cloned
> from the FCA source HEAD `5604606d`).**
>
> **Progress (2026-06-08): G.1.0 build scaffold ✅, G.1.1 lnL kernel K1 ✅, G.1.2 derivative kernel K2 ✅,
> G.1.3 CUDA-graph + on-device echild K3 ✅ (correctness/capability)** — the two custom CUDA kernels at the
> heart of the bet (postorder lnL + single-edge df/ddf) are validated standalone on a V100: lnL parity to
> the G.0 oracle at rel ~1e-12 (incl. **+R10 in one pass**), FD-validated df/ddf that **rediscover
> IQ-TREE's optimized branch length**, and the full sweep now **captured as a replayable CUDA graph fed by
> a device-resident branch-length buffer** — a 197-edge optimizeAllBranches-shaped sweep converges
> **bit-identically** under graph replay (clearing the "same converged branch lengths" gate). **Honest
> caveat:** graph wall-clock is *parity* (1.00–1.01×) — the 98-dependent-kernel chain is GPU-scheduling-
> latency-bound, so the graph's win is **structural** (104→1 host API calls + no per-iteration host
> re-feed), and a wall-clock speedup needs same-depth kernel fusion (a perf pass). Critical-path numerics
> + the device-resident replay capability proven; remaining work is the perf pass + in-tree integration
> (G.2). See the [G.1 execution log](gpu-modelfinder-g1-log.md).
New go-forward direction after the **Mode-L** single-rank optimiser was abandoned (broken FreeRate
gradient ~10⁵⁴ on +R; +34% traversals on +G) and the entire CPU-dispatch line (Mode-F → MPGC → ATMD →
EDM → Trimorph L2/3) was exhausted on the same walls.
**Target reader:** future implementing self (Claude / as1708).
**Scope:** IQ-TREE 3 ModelFinder, **one GPU**, AA/DNA 100K–1M patterns; Gadi (V100/A100/H100).

> One-sentence thesis: every CPU architecture died on the **dispatch/load-balance** problem (Amdahl
> +R10 pole, MPI-barrier tax, ~300–400 GB/s SPR DRAM). A **single GPU dissolves dispatch entirely**
> (one worker → no pole, no FCA/EDM) and trades the bandwidth wall (~2–3 TB/s HBM) for a **VRAM**
> wall. The bet: port the *likelihood/gradient kernel* to GPU (BEAGLE v4), keep the legacy
> lnL-correct optimiser and FCA's cheap CPU pruning, and make the hot loop **CPU-free via CUDA Graph
> replay** over the fixed-topology scoring phase.

---

## 0. Feasibility recon (2026-06-01) — what is and isn't already true

| Question | Finding | Source |
|---|---|---|
| Gadi GPU queues for dx61? | ✓ `gpuvolta` (V100 32GB), `dgxa100` (A100 80GB), `gpuhopper` (H100) live; dx61 ~561 KSU avail | `qstat -Q`, `nci_account -P dx61` |
| BEAGLE + CUDA toolchain? | ✓ modules `beagle-lib/4.0.1` (+3.1.2), `cuda/12.9.0`…`10.1`, cudnn | `module avail` |
| **Does IQ-TREE 3 use BEAGLE / any GPU?** | **✗ NO** — zero `beagle`/`libhmsbeagle` refs, no `.cu`/`cudaMalloc`/`__global__`, no CMake CUDA/BEAGLE option. CPU SIMD kernels only. | grep of source tree |
| Existing GPU work in this repo? | ✗ none (no scripts/docs/builds) — fresh start | filesystem scan |
| CPU kernels to mirror | `PhyloTree::computePartialLikelihood` (postorder, phylotree.h:902); `computeLikelihoodDerv` (phylotree.h:1377); eigen-decomp in `modelmarkov` | source |

**The load-bearing consequence:** because IQ-TREE 3 has **no GPU path at all**, "port ModelFinder to
GPU via BEAGLE" is **not a flag flip** — there is nothing to enable. A GPU likelihood path must be
*built*. Therefore the de-risk (Phase G.0) is a **standalone `libhmsbeagle` harness**, not an
IQ-TREE invocation. Wiring BEAGLE into IQ-TREE's `PhyloTree` is a *later* phase (G.2+), justified
only if G.0 shows the kernel speedup is real.

---

## 1. Architecture

```
          ┌─────────────────────── CPU (cheap control) ───────────────────────┐
          │  ModelFinder driver (phylotesting.cpp): candidate-model list,      │
          │  FCA cost-predictor + filterRates pruning  ── NO MPI dispatch,      │
          │  NO Mode-L.  One model at a time, in order.                        │
          │  Legacy alternating optimiser (Brent α / EM p_inv / NR branches) — │
          │  drives parameter updates; each lnL/gradient eval is a GPU call.   │
          └───────────────┬────────────────────────────────────────────────────┘
                          │  (per LM/Brent/NR step: push params, replay graph)
          ┌───────────────▼────────────────────────────────────────────────────┐
          │  GPU (BEAGLE v4 instance, one per resident model)                   │
          │   • tip partials/states, eigen-decomp (U,Λ,U⁻¹), category rates     │
          │   • updateTransitionMatrices  →  P_i = U e^{Λ r_c t_i} U⁻¹          │
          │   • updatePartials (postorder)  →  root log-lh                      │
          │   • pre-order partials + edge derivatives (Ji et al O(N) gradient)  │
          │   • CPU-FREE HOT LOOP: capture the fixed-topology scoring traversal │
          │     as a CUDA Graph; each optimiser step = graph replay (params in  │
          │     device buffers, no per-call kernel-launch/H2D storm).           │
          └─────────────────────────────────────────────────────────────────────┘
```

**Why each choice:**
- **Keep the legacy optimiser** (not Mode-L): Mode-L's whole premise was "fewer full-tree evals."
  On GPU each eval is cheap (HBM-bound, graph-replayed), so eval *count* stops mattering — and
  Mode-L's analytic gradient is the exact thing that broke (10⁵⁴ overflow). BEAGLE computes the
  gradient correctly. So: legacy optimiser for control, BEAGLE for the kernel.
- **Keep FCA pruning** as cheap CPU control: filterRates + cost-predictor still cut the candidate
  set; they cost ~nothing and reduce GPU work. No MPI, no cohorts — single GPU = no dispatch.
- **CUDA Graphs over the scoring phase**: ModelFinder's per-model scoring runs on a **fixed reference
  tree** (topology constant; only branch lengths + model params change). That fixed topology is
  exactly what makes the traversal graph-capturable: capture once, replay per optimiser step,
  eliminating per-eval launch overhead and CPU↔GPU chatter. (The post-MF **SPR** phase *changes
  topology* → graph must be re-captured per topology; out of scope for G.0.)

---

## 2. The binding constraint is now VRAM

Postorder partial-lh buffer ≈ `(#internal nodes) × nptn × ncat × nstates × 8B`.
- AA, 100 taxa (~98 internal), 1M patterns, 20 states, 4 cat: `98 × 1e6 × 4 × 20 × 8 ≈ 126 GB` —
  **exceeds even A100-80GB**. AA-**100K**: `≈ 12.6 GB` — fits V100-32GB / A100 comfortably.
- ⇒ AA-1M needs **pattern tiling** (stream pattern blocks through the GPU) or fewer resident node
  buffers (BEAGLE scaling/rescaling buffers add overhead). G.0 should run **AA-100K first** (fits),
  then probe AA-1M tiling. (BEAGLE supports partials in tiles via multiple instances / partial
  operations; confirm in G.0.)
- Single-precision (float) halves VRAM and is much faster on consumer/V100; phylo lnL usually needs
  the scaling buffers + occasional double for the root reduction. BEAGLE supports both
  (`BEAGLE_FLAG_PRECISION_SINGLE/DOUBLE`); measure accuracy vs the CPU reference lnL in G.0.

---

## 3. Phased plan with gates

| Phase | What | Gate |
|---|---|---|
| **G.0 de-risk** | **Standalone `libhmsbeagle` GPU harness** (`gadi-ci/gpu-modelfinder/gpu_derisk.cpp`): AA-100K + fixed reference tree; LG+G4, LG+F+I+G4, LG+R8, LG+R10. **✅ lnL DONE** — bit-parity CPU≡GPU all models (LG+G4 also 1e-8 vs IQ-TREE), **43–66×**, +R10 via bit-exact 5+5 split (57×). **✅ Gradient DONE** — algorithm FD-validated on CPU (g4 2.8e-3, g1 1.1e-7); GPU gradient infra runs on A100-80GB (~100–150×) but **BEAGLE-4.0.1 CUDA gives wrong 20-state values** (bug 16) ⇒ deferred to a fixed BEAGLE / custom kernel. | lnL matches CPU within 1e-3 ✅ (bit-exact); GPU eval ≥ a few× faster ✅ (43–66×); gradient correct ✅ on CPU, ⚠ GPU = BEAGLE library gap; justify G.1 ✅ |
| G.1 CUDA-Graph hot loop | Capture the fixed-topology scoring traversal as a CUDA Graph; drive ~N optimiser steps by graph replay. Measure CPU-free hot-loop speedup vs naive per-call BEAGLE. | replay correct (same lnL) AND materially faster than per-call; launch overhead amortised |
| G.2 IQ-TREE integration (scoring) | Wire BEAGLE behind `computePartialLikelihood`/`computeLikelihoodDerv` for the **MF scoring phase only**, gated by a `--gpu` flag. Legacy CPU path stays default/fallback. | full `-m TEST` AA-100K best-model + lnL parity vs CPU; MF wall < CPU |
| G.3 heavy regime | `-m MF` (+R chains) AA-100K on GPU — the regime that killed every CPU architecture. | MF wall beats FCA np=1 (259s) and ideally np=16; +R10 no longer the pole |
| G.4 AA-1M / tiling | pattern tiling for AA-1M VRAM; SPR-phase graph re-capture. | AA-1M MF within VRAM; end-to-end speedup |

**G.0 is the whole bet.** If a single heavy model is not materially faster on one GPU (kernel +
overhead) than its CPU slice, the project stops — with *zero* dispatch/integration sunk cost.

---

## 4. The G.0 de-risk experiment (concrete)

**Harness (`gpu-ci/derisk_beagle_aa100k.cpp` + PBS `-q gpuvolta`):**
1. `module load beagle-lib/4.0.1 cuda/12.x`; link `-lhmsbeagle`.
2. Read the AA-100K alignment (same `…/alignment_100000.phy` used by the CPU gates) → patterns +
   weights; read a fixed tree (export one from an IQ-TREE CPU run, or NJ).
3. `beagleCreateInstance(..., BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_FRAMEWORK_CUDA, ...)` — pick
   the GPU resource; set tip partials, category rates/weights (+G4 / +R), eigen-decomposition
   (U,Λ,U⁻¹) from the LG matrix (export from IQ-TREE `modelmarkov`, or recompute).
4. `updateTransitionMatrices` → `updatePartials` (postorder) → `calculateRootLogLikelihoods`.
   Confirm lnL matches the IQ-TREE CPU lnL for that tree+model (parity check).
5. **Gradient**: exercise BEAGLE v4's pre-order partials + `beagleCalculateEdgeDerivatives` (the
   Ji-et-al O(N) gradient BEAGLE implements correctly) — confirm it returns finite, sane values
   (the thing Mode-L got wrong).
6. **Measure**: wall per lnL eval, per gradient eval, H2D transfer, kernel-launch overhead; repeat
   K times to model an optimiser inner loop. Compare to the CPU per-model cost (LG+G4 ≈ 89s,
   LG+F+I+G4 ≈ 333s at AA-1M; scale to 100K).

**Decision rule:** GPU lnL+gradient eval must be clearly faster than the CPU equivalent *and* the
gradient must be numerically sound. Then proceed to G.1 (CUDA-Graph replay).

---

## 4b. G.0 empirical findings (2026-06-01) — resolve/refine §2 and §5

- **Precision/scaling (resolves §2 precision question):** at AA-100K, 100 taxa, `PRECISION_DOUBLE`
  needs **no rescaling** — unscaled lnL is bit-identical to manually-scaled and ~2× faster. Run the GPU
  hot path unscaled in double. Single precision would still need scaling (future float/VRAM-halving).
- **Compact tip states:** use `beagleSetTipStates` (not partials) — NCAT-independent tip memory, faster
  state kernels, identical lnL. Cut AA-100K LG+G4 VRAM 19.9→10.4 GB and eval 121→86 ms (then →45 ms unscaled).
- **⚠ NEW HARD CONSTRAINT — BEAGLE 4.0.1 CUDA caps category count at `kMatrixBlockSize`=8 (20 states).**
  `beagleCreateInstance` with NCAT>8 hard-`exit(-1)`s (`bgScaleGrid.y = NCAT/kMatrixBlockSize > 1`,
  rescale path unimplemented) — **category-driven, not VRAM/pattern-driven**. Affects **+R9/+R10**, i.e.
  exactly the long-pole models. Two fixes, both viable: (a) **category-split** the >8-cat model into
  ≤8-cat sub-passes and combine per-site (validated bit-exact in G.0 — `r10split` Δ=0.0000, 57×); (b)
  **rebuild BEAGLE** with larger `kMatrixBlockSize` or the slow-reweighing kernel implemented. G.2+
  integration must choose one; (a) needs no patched dependency.
- **Speedup is per-eval, GPU(unscaled) vs BEAGLE-CPU-SSE single-instance** — 43–66× across the model
  set. Still not "vs IQ-TREE AVX/threaded" (that needs a single-lnL IQ-TREE timing). The *direction* is
  unambiguous and large; the +R10 long-pole runs ~57× on one GPU.

## 5. Open questions / risks (to resolve in G.0/G.1)

- **BEAGLE v4 gradient API surface — ✅ RESOLVED (2026-06-06, mostly negative for GPU).** Pre-order partials
  + `beagleCalculateEdgeDerivatives` (plural; the singular is deprecated → `NO_IMPLEMENTATION`) give the
  branch-length gradient. The differential matrix is the **generator Q·r_c** (per-category rate baked in;
  `categoryRates` ignored by the kernel), set once via `beagleSetDifferentialMatrix`; `outDerivatives` must
  be NULL (per-pattern, size count·nptn) — use `outSumDerivatives`. **CPU plugin: FD-correct** (g4 2.8e-3,
  g1 1.1e-7). **CUDA plugin: `setRootPrePartials` is a stub** (seed via `beagleSetPartials`, needs
  `partialsBufferCount>max-preorder-index`), and the **20-state pre-order/edge-derivative path is broken**
  (wrong values for all NCAT even with the `hmctest` manual-transpose recipe — bug 16 in the g0-log).
  ⇒ off-the-shelf BEAGLE-4.0.1 gives a usable GPU gradient only for nucleotides; **protein GPU gradients
  need a fixed/newer libhmsbeagle or a custom kernel.** Model params (α, p_inv, +R) still use legacy
  Brent/EM as planned (BEAGLE gives only branch-length derivatives directly).
- **CUDA Graphs + BEAGLE** — does BEAGLE expose a capturable async stream, or must we capture at the
  CUDA driver level around BEAGLE calls? Conditional-node device-side optimiser loop (CUDA 12.4+
  conditional graph nodes) is the aspirational "CPU-free" form; a simpler win is capture-once /
  replay-per-step. Confirm which is achievable.
- **VRAM** — AA-1M (~126 GB partials) needs tiling; AA-100K fits. Single vs double precision tradeoff.
- **Eigen-decomposition transfer** — recompute on host (cheap, once per model-param update) and push
  U,Λ,U⁻¹, or compute on device. Host is simplest for G.0.
- **IQ-TREE integration cost (G.2)** — `computePartialLikelihood` is a deeply-templated SIMD kernel;
  wiring BEAGLE behind it cleanly (without disturbing the CPU path) is non-trivial but bounded.
- **dx61 GPU registration** — first `qsub -q gpuvolta` confirms the project can use GPUs.

---

## 6. What this explicitly does NOT do

- No multi-GPU (deferred — single-GPU accessibility first).
- No Mode-L resurrection (analytic-gradient optimiser premise is dead; BEAGLE owns the kernel gradient).
- No MPI / FCA cohorts / EDM dispatch (single GPU = no dispatch problem).
- No SPR-phase GPU graph (topology changes) until G.4.

---

---

# PART II — Custom GPU CUDA Kernel: Implementation Plan (2026-06-07)

**How this was produced.** A 9-agent research workflow read the cloned IQ-TREE source
(`/scratch/rc29/as1708/iqtree3-gpu`, branch `gpu-kernel`, HEAD `5604606d` = the FCA+WS-A.2 source behind
the headline CHANGELOG runs), the GPU phylo literature (Gangavarapu 2024 btae030, Ji 2020 msaa130,
Tensor-Cores 2025 syag017, BEAGLE 3/4, PhyloGrad, AD-no-Panacea), the BEAGLE GitHub state, and the
CPU-failure post-mortems (Trimorph, Mode-L, CHANGELOG). Every claim below is grounded in a `file:line` or
a cited source; full findings are archived in the run transcript. Setonix-IQ branch: `gpu-modelfinder`.

**Implementation status:** **G.1.0 (build scaffold) ✅ PASS** (2026-06-07, job 170176864) — in-tree CUDA
toolchain proven end-to-end. **G.1.1 (postorder lnL kernel K1) ✅ PASS** (2026-06-08, job 170188276 on a
V100) — custom **eigen-space** CUDA kernel matches the G.0 oracle to rel ~1e-12 for g4/r8/r10/g1; **+R10
runs in ONE pass** (−7554280.5776, the NCAT>8 capability stock BEAGLE lacked); native-20 VRAM r10 14.9 GB;
g4 37.8 ms already beats BEAGLE 45 ms (all unoptimised, standalone harness `gpu_k1_lnl.cu`). **G.1.2
(single-edge derivative kernel K2) ✅ PASS** (2026-06-08, job 170188743 on a V100) — custom eigen-space
df/ddf kernel, **FD-validated** (df rel ~2e-9, ddf rel ~4e-6, all 4 models), lnL(t0) cross-check rel
~1e-12, and **bisection-on-df independently rediscovers IQ-TREE's optimized edge length** (|t*−t0|≤2e-4,
ddf<0) for g4/g1/r8/r10; each NR `evalAt` is **~1.1–1.3 ms** (theta cached). **G.1.3 (CUDA-graph capture +
on-device echild rebuild K3) ✅ PASS (correctness/capability)** (2026-06-08, job 170189528 on a V100) —
on-device `build_echild` from a device-resident `d_brlen` (ULP-clean vs host), full sweep captured as a
CUDA graph and replayed without re-capture; graph lnL = G.0 oracle, **patlh bit-identical** to the naive
path, deterministic replay, and a **197-edge optimizeAllBranches-shaped sweep converges bit-identically**
(graph vs naive, |dt|=0) — clearing the literal "same converged branch lengths" gate. **⚠ Speed reframed:
wall-clock is parity (1.00–1.01×) at every pattern count** — the 98-dependent-kernel chain is bound by
GPU-side per-kernel scheduling latency the graph cannot remove; the graph's win is **structural** (104→1
host API calls + device-resident brlen, the integration unblock), not a speedup. **Perf pass (same-depth
kernel fusion K4, job 170194367)** done — bit-identical to oracle, and it **wins in the launch-bound regime
(1.79× at nptn=1000)** but is a **wash-to-slight-loss at the compute-bound production count** (g4 1.05×, g1
1.14×, r8 0.96×, r10 0.94×). The decisive finding: the real ML tree is a **deep height-42 ladder** (only ~8
shallow levels batch; a 33-deep serial tail can't), and at 100K patterns the V100 is already SM-saturated,
so **the production bottleneck is intra-kernel compute/bandwidth, not launch overhead** — relocating the
speedup lever to coalescing/shared-mem/register opts (K1 headroom; K1 already beats BEAGLE 37.8 vs 45 ms),
*not* launch batching. ⇒ G.2 uses the **per-node K3 graph** (not fusion) as the production sweep. Running
execution log: [gpu-modelfinder-g1-log.md](gpu-modelfinder-g1-log.md). Next: **G.2 in-tree integration**.

## II.0 The decisive reframe (read this first)

The standalone G.0 harness explored the **Ji-et-al O(N) all-branch pre-order gradient** because it was
de-risking BEAGLE's gradient capability in general. **But IQ-TREE's ModelFinder does not use that.**
The codebase map is unambiguous:

- Per-model optimisation is the **alternating** loop `ModelFactory::optimizeParameters` (`model/modelfactory.cpp:1558`):
  branch re-optimisation (`tree->optimizeAllBranches`, :1629/:1686) alternated with rate/model-param fit
  (`optimizeParametersOnly`, :1642). **Branch re-opt = 75–85 % of per-model wall** (Trimorph.md:142).
- `optimizeAllBranches` (`tree/phylotree.cpp:2732`) sweeps each branch in a **fixed pre-order** calling
  `optimizeOneBranch` (:2628) → `minimizeNewton` (`utils/optimization.cpp:422`) → `computeFuncDerv` (:2563)
  → **`computeLikelihoodDerv`** — a **single-edge** lnL + 1st + 2nd derivative, computed from
  `theta = partial_lh_node ⊙ partial_lh_dad` and the eigen-space diagonals `val0=exp(eval·r·t)·prop`,
  `val1=(r·eval)·val0`, `val2=(r·eval)·val1`, reduced by `dotProductTriple` (`tree/phylokernelnew.h:2338-2424`).

So IQ-TREE's branch gradient is a **per-edge, postorder-partials-based df/ddf** — there is **no pre-order
traversal**. Consequences that reshape the whole plan:

1. **The broken BEAGLE 20-state CUDA pre-order/edge-derivative kernel (g0-log bug 16) is NOT on the
   critical path.** IQ-TREE never calls that code shape. The GPU gradient we must build is
   `computeLikelihoodDervSIMD` (single-edge), which is structurally just *the postorder lnL kernel + a
   per-pattern triple dot-product* — far lower risk than the pre-order gradient the harness fought.
2. **O(depth) pre-order buffer recycling is NOT needed for the critical path** (there is no pre-order pass
   to recycle). It returns only in the **optional advanced track** (§II.8 G.5) if a joint/all-branch or
   device-side LM optimiser is ever pursued. The user's "single biggest VRAM fix" is real — but it applies
   to the Ji-gradient track, not to porting IQ-TREE's existing per-edge optimiser.
3. **The dominant GPU workload is therefore: (a) postorder partial-likelihood recompute along the dirty
   path, and (b) the single-edge df/ddf reduction.** Both reuse the *same* partials and P-matrices; the
   gradient adds one `dotProductTriple` over `theta`. This is what 75–85 % of wall actually is.

The Ji pre-order gradient stays relevant only as: the G.0 correctness oracle we already have, and a future
option if we replace IQ-TREE's alternating optimiser with a joint solver (explicitly **out of scope** —
the design keeps the legacy lnL-correct optimiser, §1).

## II.1 Build-vs-reuse: custom in-tree kernels, not BEAGLE-behind-IQ-TREE

**Decision: write custom CUDA kernels inside the IQ-TREE GPU tree; do NOT bolt BEAGLE behind IQ-TREE's
`computePartialLikelihood`.** Rationale:

- IQ-TREE stores node partials in **eigen coordinates** (post `U⁻¹·L` multiply), layout
  `partial_lh[ptn*block + c*nstates + x]`, `block = nstates*ncat_mix` (`tree/phylokernelnew.h:1335,1466`).
  A child's contribution is `vchild = Σ_i echild[i]·partial_lh_child[i]` with `echild = evec·exp(eval·r·t)`
  (:1026-1034) — i.e. IQ-TREE never forms full `P(t)`; it splits `Q=UΛU⁻¹` across the post-order. Bolting
  BEAGLE (which works in probability space with its own buffer model + 20→32 padding) behind this seam
  means translating coordinate systems every call — awkward and slow.
- BEAGLE owns buffer allocation as **O(nodes)**; that is exactly what forced the 402-buffer / 50.3 GB
  gradient onto an A100 (g0-log:242-247). The headline optimisations (CUDA-Graph capture of the fixed
  topology, owning our own device buffers) are **hard or impossible through BEAGLE's public C API**
  (its `cuda-graph-test` branch is unmerged prior art to read first, finding [4]).
- The G.0 harness already proved the **math** and gives us **bit-parity oracles** (CPU IQ-TREE kernel +
  the BEAGLE lnL numbers). So a custom in-tree kernel is validated, not speculative.
- BEAGLE-in-IQ-TREE has **zero precedent** (no tool wires BEAGLE into IQ-TREE/RAxML-NG; finding [6]); it
  would be novel integration work *anyway*, with less control than owning the kernel.

We keep the **lnL math identical to G.0** (reversible eigendecomposition, compact tip states, double
unscaled) so the custom kernel inherits the proven bit-parity.

## II.2 The integration seam (surgical, no call-site churn)

IQ-TREE dispatches the kernel through **four `PhyloTree` member function pointers** (`tree/phylotree.h:904,
951,1004` and — corrected G.1.0 — **`computeLikelihoodDervPointer` at 1346**, sibling
`computeLikelihoodDervMixlenPointer` at 1349; NOT all in the 902-1004 block),
assigned per state-count/ISA in `setLikelihoodKernel*` (`tree/phylotreeavx.cpp:50-145`) and invoked through
thin virtual wrappers (`tree/phylotreesse.cpp:212-236`):

```
computePartialLikelihoodPointer      // postorder partials for one TraversalInfo
computeLikelihoodBranchPointer       // root lnL at a branch (convergence checks)
computeLikelihoodDervPointer         // single-edge lnL + df + ddf  ← the 75–85% hot path
computeLikelihoodFromBufferPointer   // lnL from cached theta buffer
```

Add `setLikelihoodKernelGPU()` assigning all four to GPU member functions **with identical signatures**
(`(TraversalInfo&, ptn_lower, ptn_upper, packet_id)` and `(PhyloNeighbor*, PhyloNode*, df, ddf)`). The
whole-tree entry `computeLikelihood()` (`phylotree.cpp:1288`) and the NR loop are **untouched**. Runtime
switch = a `Params::gpu` bool parsed in `utils/tools.cpp` and a `--gpu` token; CPU path stays default.
Build-time: `#cmakedefine IQTREE_GPU` so call sites compile out entirely in a CPU-only build.

**Hard guards (else fall back to CPU):** `params.model_test_and_tree == false` (no NNI/SPR topology
changes — only then is the topology fixed and graph-capturable); reversible bifurcating models (+ cherry +
tip-internal cases); default contiguous slot allocation (assert `lh_mem_save`/`buffer_mem_save` off).
Multifurcating start trees, non-reversible (`computeNonrevPartialLikelihoodSIMD`), and site-specific models
fall back to the existing CPU pointers.

## II.3 Kernel architecture (custom CUDA, 20-state AA)

Two core kernels, both **pattern-parallel** and **memory-bandwidth-bound** (confirmed by btae030 + syag017:
the GPU saturates HBM, so optimise for *bandwidth/coalescing*, not FLOPs):

- **K1 — postorder partial-likelihood.** Thread map `threadIdx.x = state`, `threadIdx.y = pattern in a
  column-block (CBS)`; grid `blockIdx.x = pattern block`, `blockIdx.y = rate category` — **loop categories
  on the y-grid, NOT a MATRIX_BLOCK_SIZE tile**, which is what dodges BEAGLE's `NCAT≤8` cap so **+R10 runs
  in one pass** (g0-log finding 7). Stage the per-edge per-category `echild`/`P` in **shared memory**
  (BLOCK_PEELING_SIZE-tiled, `__syncthreads`, FMA-accumulate over child states), stream patterns from HBM.
  Store partials in **eigen coordinates**, layout `[ptn][cat][state]` with **state contiguous** so a warp
  of consecutive patterns issues fully-coalesced 128-B loads.
- **K2 — single-edge derivative** (`computeLikelihoodDerv` analog). Build `theta = partial_node ⊙ partial_dad`
  once (cache, reuse across the 2–5 NR steps on the same edge — mirror `theta_computed`,
  `phylokernelnew.h:2393`), then per NR step recompute only `val0/val1/val2` for the new branch length and
  reduce `lh,df,ddf = dotProductTriple(val0,val1,val2,theta)` as a device-wide pattern reduction
  (warp-shuffle + block reduce; **no global atomics** on the accumulator).

Numerics (all G.0-validated): **FP64, unscaled** for AA-100K/100-taxa (per-site L≈e⁻⁷⁵ ≫ double's e⁻⁷⁰⁸
floor; unscaled == scaled bit-exact, ~2× faster). Keep a **BEAGLE-style log-space per-pattern scale-buffer
path as a runtime/compile option** (reset → accumulate → `+min_scale·(−177.4456782233459932741)`,
matching `phylotree.h:74-80`) that auto-engages on a detected `−inf/NaN` root lnL for deeper/larger trees.
**Never** TF32/FP16 on the parity-critical reduced lnL (the likelihood surface's "terraces" demand FP64 —
syag017; would break the 1e-8 parity gate). **Tip data** = compact states (one int/site, NCAT-independent;
ambiguous AA → "any"/all-ones; gather the precomputed `partial_lh_leaf = E·tip_partial_lh[state]` by state
code) — bit-identical to partials, cuts VRAM 19.9→10.4 GB and eval 121→86 ms (g0-log finding 5).

**State padding:** prefer **native 20** (skip BEAGLE's forced 20→32 pad → saves 37.5 % VRAM+bandwidth on a
memory-bound kernel, ~AA-1M 101→63 GB). Validate coalescing empirically (pad-24 = 1.5×16 risks misaligned
16-value transactions); fall back to pad-32 only if profiling shows native-20 breaks coalescing.

**Transition matrices:** eigendecompose LG **once on host** (constant matrix), push `U,Λ,U⁻¹` (tiny,
resident per model). Rebuild `echild`/`P_c = U·exp(Λ r_c t)·U⁻¹` **on-device** when branch lengths change
(custom fused batched 20×20 kernel — fuses the diagonal `exp` between two GEMMs, beats
`cublasDgemmStridedBatched` at this size and is trivially graph-capturable with params in device buffers).

**UBYTE = `unsigned short` (16-bit)** scale counters (`phylonode.h:17`) — a naive 8-bit assumption breaks
bit-parity; copy width must match.

## II.4 Memory plan (corrected numbers)

- **VRAM = #internal × nptn × ncat × nstates × 8 B.** AA-100K lnL (postorder, compact tips, double) ≈
  **10.4 GB** (observed) — fits a 32 GB V100. The design's earlier "AA-1M ≈ 126 GB" is ~2× the literal
  formula; **corrected: AA-1M lnL ≈ 63 GB native / ~101 GB if 20→32-padded** (finding [5]). Native-20
  brings AA-1M to ~63 GB → fits A100-80GB **without tiling**; padded needs tiling.
- IQ-TREE's per-edge NR (the critical path) needs **only postorder partials** — no pre-order doubling — so
  the **gradient does NOT inflate VRAM** here (unlike the G.0 Ji-gradient's 50 GB). The 50 GB / A100-only
  number was a property of the *all-branch pre-order* exploration, not IQ-TREE's optimiser.
- **AA-1M pattern tiling (G.4):** lnL is a per-site sum, so stream pattern blocks through a fixed buffer set;
  the SAME captured graph replays per tile with only the tile's tip-state/partials pointers swapped. Tiling
  composes cleanly with graph replay.

## II.5 CUDA-Graph plan (G.1)

- Capture target = the body of **`optimizeAllBranches`** (`phylotree.cpp:2732`): the fixed pre-order branch
  list (`computeBestTraversal`, called once), the postorder dirty-path recompute, and the per-branch df/ddf
  reduction — all on a **fixed topology**. CUDA Graph update *requires* identical topology, so this is
  exactly the capturable case.
- **Params in fixed device buffers** (branch lengths[~197], catRates[NCAT], `U/Λ/U⁻¹`): an optimiser step
  writes new values via one small `cudaMemcpyAsync` (or an in-graph memcpy node) and **replays** — no
  `cudaGraphExecUpdate`, no re-capture. Re-capture only when **model params change category count**
  (between outer iterations) — a separate captured graph per NCAT.
- Collapse the ~300 launches/eval (99 postorder ops + ~197 matrix builds + reduction; g0-log:102) into one
  `cudaGraphLaunch` (~2.5 µs Ampere / O(n) single-submission V100), vs ~1–3 ms naive per-call submission.
- **Divergent paths fall back to non-graph:** `optimizeAllBranches`'s lnL-decrease rollback (:2771-2797) and
  any data-dependent NR step count. Stretch goal (G.1b): CUDA 12.4+ conditional-WHILE node or device-graph
  tail-launch for a CPU-free inner loop — verify the Gadi `cuda/12.x` module actually runs WHILE nodes on
  A100 with a 3-line toy *before* wiring the optimiser.

## II.6 CPU-failure walls → GPU design constraints

| CPU wall (where it killed us) | On one GPU |
|---|---|
| **Amdahl +R10 pole** (f_s≈0.182 → FCA asymptote ~5.5×; Trimorph:146) — a *dispatch/load-balance* wall | **Dissolves.** One worker, models sequential, full device each. No cost-predictor/cohort/moldable layer — **delete it.** +R10 is just the longest single eval (~57× in G.0). |
| **NCAT≤8 BEAGLE CUDA cap** (kMatrixBlockSize; +R9/+R10 `exit(-1)`) | **Gone in a custom kernel** — loop categories on the grid-y; +R10 in one pass (reproduce r10split −7554280.5776 Δ=0). |
| **MPI barrier tax** (3K–38K Allreduce/model; filterRatesMPI 2.5× straggler) | **Gone.** Only reductions left are *on-device* (root lnL pattern-sum, per-edge df/ddf) — warp/block reductions, no network. |
| **DRAM ~300–400 GB/s** (IPC 1.88→1.3, LLC miss 85 % at AA-1M) | **Replaced by HBM** ~900 GB/s (V100) / ~2 TB/s (A100). The kernel MUST be coalesced to realise it — the bandwidth ratio is the whole thesis. |
| **1 thread/model** (`setNumThreads(...,1)`, `phylotesting.cpp:2391`; defeated all intra-model parallelism) | **Inverts:** one model → the WHOLE GPU's cores, parallel over pattern×category×state — exactly what the CPU could not exploit. Sequential models, full device (G.0 measured 43–66× this way). |
| **Branch re-opt = 75–85 % of wall** (Mode-L failed because it didn't touch it) | **The dominant GPU target.** Accelerate the postorder-lnL + single-edge df/ddf NR loop (K1+K2 behind `computeLikelihoodDerv`), not the rate fit. |
| **FreeRate analytic gradient 10⁵⁴ overflow** (Mode-L §17.20) | Irrelevant to per-edge NR; but the lesson — **FD-validate every gradient** — is the non-negotiable gate (§II.8). |

Keep **only** FCA's *cheap CPU* `filterRates`/`filterSubst` BIC pruning + the cost-predictor formula
(`phylotesting.cpp:3035/3290/3918`) to cut the candidate set and order models cheapest-first so pruning
fires before the +R long-pole. Strip the entire `_IQTREE_MPI` LPT dispatch block.

## II.7 Honesty discipline (baseline + speedup reporting)

- The G.0 **43–66× is vs single-thread BEAGLE-CPU-SSE**, *not* IQ-TREE's AVX-512 multithreaded kernel.
  **Measure the honest baseline early:** a single AA-100K lnL eval in IQ-TREE's own `-nt {1,8,16}` AVX-512
  kernel. Expect the real single-GPU AA margin to be **low-double-digit**, concentrated in large-pattern /
  heavy-+R models; the GPU can **lose** to CPU-SSE below the ~10³–10⁴-pattern saturation knee.
- **Gate the GPU path on pattern count** — route small alignments to the CPU kernel; report speedup as a
  **curve vs pattern count**, never a single headline number.

## II.8 Phase-by-phase plan (each phase independently testable + validated against a concrete number)

| Phase | Deliverable | Independent test | Validates against |
|---|---|---|---|
| **G.1.0 Build scaffold ✅ PASS** (job 170176864) | `option(IQTREE_GPU)` + gated `enable_language(CUDA)` + `iqtree_gpu` .cu static lib linked into `iqtree3` (mirror `kernelavx`; option ~297, CUDA block before `add_subdirectory(main)` ~913, link after 1023) + `#cmakedefine IQTREE_GPU` + `--gpu`/`-gpu` flag + a hello-world `.cu` launched from the diag hook (`main.cpp` after `parseArg`). In-job build (`cuda/12.5.1 + gcc/12.2.0 + cmake/3.24.2`, all-GCC host). | `-DIQTREE_GPU=OFF` build behaviourally identical; `-DIQTREE_GPU=ON` w/o `--gpu` runs CPU path; `--gpu` launches the kernel + clean `cudaGetLastError` on gpuvolta. | Pure plumbing (no numerics). **✅ 8/8: ON+OFF build; V100 kernel launch (marker 0xC0DE, `cudaGetLastError=cudaSuccess`); `libcudart` linked ON only; OFF==ON-no-gpu==ON-`--gpu` lnL −21152.524 (example.phy). Cost: clone submodules `cmaple`/`lsd2` had to be populated from the FCA source first.** |
| **G.1.1 Postorder lnL kernel (K1) ✅ PASS** (job 170188276) | Custom CUDA **eigen-space** postorder partial-LH, layout `[slot][cat·NS+state][ptn]` (pattern-contiguous, coalesced), compact tips, FP64 unscaled, NCAT≤10 single-pass, native-20. Standalone harness `gpu_k1_lnl.cu` (validated before the in-tree wiring, which becomes part of G.2). | full-tree lnL parity vs G.0 oracle; **NCAT=10 in ONE pass = r10split −7554280.5776**. (Nsight HBM% deferred.) | G.0 lnL: −7541976.9391 (g4). **✅ 4/4 models rel ~1e-12 to oracle (g4/r8/r10/g1); r10 ONE-PASS exact (NCAT>8 cap GONE); native-20 VRAM r10 14.9 GB < V100; K1 g4 37.8 ms BEATS BEAGLE 45 ms — all unoptimised. g1-log.** |
| **G.1.2 Single-edge derivative kernel (K2) ✅ PASS** (job 170188743) | `computeLikelihoodDervSIMD` analog: t-independent `theta = node_eig ⊙ dad_eig` cache (computed once on GPU) + `val0/val1/val2 = exp(eval·r·t)·prop·{1,r·eval,(r·eval)²}` uploaded per-step + `k2_derv` triple-dot → lh/df/ddf. Standalone harness `gpu_k2_derv.cu` (in-tree `computeLikelihoodDervPointer` wiring deferred to G.2). | df/ddf vs CPU FD oracle (swept-ε); lnL(t0) cross-check vs K1/G.0; bisection-on-df → converged length matches IQ-TREE's. | **✅ 4/4 models: FD df rel ~2e-9 (meets g1<1e-6 by 3 orders), ddf rel ~4e-6; lnL(t0) cross-check rel ~1e-12 (freq-fold identity OK); bisection rediscovers t0 (\|t*−t0\|≤2e-4, ddf<0). evalAt ~1.1–1.3 ms (theta cached). g1-log.** |
| **G.1.3 CUDA-Graph capture ✅ PASS (correctness/capability; speed reframed)** (job 170189528) | On-device `build_echild` rebuilds echild from a device-resident `d_brlen`; full sweep `[brlen H2D→build_echild→memset→98×k1_node→reduce→lnL D2H]` captured on a non-default stream (Global mode), replayed per brlen change with one `cudaGraphLaunch` (Pattern-A pinned-buffer brlen node; no re-capture/SetParams); deterministic block-local pairwise reduction; fallback→naive on capture error. Standalone `gpu_k3_graph.cu`. | Graph lnL == naive == G.0 oracle; **patlh bit-identical** graph-vs-naive (pre-reduction); deterministic replay; **197-edge optimizeAllBranches-shaped sweep converges bit-identically** to naive (|dt|=0, ΔlnL=0); timing curve graph-vs-naive across nptn. | **✅ 4/4 models: V0 echild ULP-clean (≤4e-16); V1 oracle rel ~1e-12; V2 patlh 0/100000 diff; V3 deterministic; V4 perturb Δ≤9e-10; V5/V6 graph≡naive convergence (|dt|=0). ⚠ wall-clock PARITY 1.00–1.01× (98-kernel-chain scheduling-latency-bound, not host-submission-bound) → win is STRUCTURAL: 104→1 API calls + device-resident brlen, NOT a speedup. Speedup needs same-depth kernel batching (perf pass). g1-log.** |
| **G.2 ModelFinder scoring integration** | Full `-m TEST` AA-100K on GPU, models sequential, keep CPU `filterRates`; strip MPI dispatch. | Best model = **LG+G4**; per-model lnL parity; `filterRates` prunes the same models (`MF_IGNORED` table matches CPU). | lnL −7,541,976.86; **MF wall < 221.6 s** (R1+R2+AVX512, honest single-node floor; vanilla 264 s / FCA-np1 258 s are the easy bar; FCA-np2 149 s on 2 nodes is the stretch). |
| **G.3 Heavy regime** | `-m MF` AA-100K, +R2..+R10 single-pass, +R warm-start chain order preserved. | No +R10 pole; r10 one-pass. | **MF wall < FCA np=1 1341 s**, ideally < np=2 481 s (CHANGELOG:136-137). |
| **G.4 AA-1M tiling** | Pattern tiling (stream blocks; same graph per tile); native-20 to fit ~63 GB on A100. | Tiled lnL == untiled (bit-parity where untiled fits); AA-1M finishes. | The regime where **FCA np=16 timed out at 3 h** (CHANGELOG:158) — any correct finite result is a win. |
| **G.5 (OPTIONAL) Ji O(N) all-branch gradient + O(depth) recycling** | Only if a joint/device-side optimiser is pursued: custom pre-order edge-derivative kernel (NOT BEAGLE's broken one) + O(tree-depth) recycled buffer pool sized by measured max DFS depth. | FD vs the G.0 CPU oracle (g4<3e-3, g1<1e-6); peak VRAM <32 GB on V100 (vs BEAGLE's 50.3 GB). | g0-log gradient FD results; the "99→~7 buffer" VRAM lever. **Not on the ModelFinder critical path.** |

**Dependency note:** G.1.0→G.1.1→G.1.2 are strictly ordered; G.1.3 needs G.1.1+G.1.2; G.2 needs G.1.3;
G.3/G.4 need G.2. Each is shippable and testable alone. G.5 is independent and optional.

## II.9 Toolchain & build (Gadi)

`module load cuda/12.5.1 gcc/12.2.0 intel-compiler-llvm/2024.2.1 cmake/3.24.2 eigen/3.3.7 boost/1.84.0`.
**All configure+build run inside a PBS job** on gpuvolta/dgxa100 — login has neither `nvcc` nor `icpx`, and
login gcc 8.5 is both too old for CUDA-12 host and below the project's `GCC_MIN_VERSION 9`. For Phase 0 use
the **all-GCC host path** (`CMAKE_CUDA_HOST_COMPILER=g++`) to remove the icpx+nvcc ABI variable; introduce
icpx as host only after it links+runs, and diff lnL between the two. `CUDA_ARCHITECTURES "70;80;90"`
(V100/A100/H100) for a fat binary, or `70` only for fast gpuvolta iteration. **Drop `-static`** (cudart
can't fully static-link); guard the GPU lib out of the `BUILD_LIB` aggregation.

## II.10 Top risks & open questions

- **Highest risk = K2 correctness** — a wrong-but-plausible 20-state kernel is easy to ship (BEAGLE proved
  it). **FD-validation vs the CPU oracle is a build-gating, non-negotiable test** at every step.
- **Honest speedup unknown** until G.2 measures end-to-end MF wall vs the AVX-512 CPU; the per-eval win must
  survive H2D, launch overhead (G.1.3 protects this), host eigendecomp, and the CPU control loop.
- **Host serial tail** — eigendecomp + P-matrix H2D + CPU optimiser loop could stall the GPU on +R chains
  with frequent param updates; measure GPU idle fraction (Nsight Systems), move P-assembly on-device if >20 %.
- **Read the BEAGLE `cuda-graph-test` branch** before G.1.3 (prior art for graph capture around phylo kernels).
- Open: whether to *eventually* go full-custom lnL is already answered (yes, custom in-tree from the start);
  whether to ever pursue G.5 depends on a future joint-optimiser decision (currently out of scope).

---

## References
- Ayres et al. BEAGLE 3 (2019), *Syst. Biol.* 68(6); BEAGLE v4 (libhmsbeagle 4.0.1, Gadi module).
- Gangavarapu, Ji, Baele, Fourment, Lemey, Matsen, Suchard (2024), *Bioinformatics* 40(2):btae030 —
  many-core high-dimensional phylogenetic gradients (GPU pre/post-order, >128× codon / >8× nucleotide).
- Ji, Zhang, Holbrook, … Suchard (2020), *MBE* 37(10):3047 — "Gradients do grow on trees" (the O(N) gradient).
- IQ-TREE 3: Wong et al. (2025), doi:10.32942/X2P62N.
- Gangavarapu & Suchard (2025/26), *Syst. Biol.* adv. syag017 — "Tensor Cores Unlock… Phylogenetic Trees"
  (AA ~2–3× over CUDA cores, FP64-only, memory-bandwidth saturated; BEAGLE v4.0.0 `tensor-cores` branch).
- Lieser, Belousov, Söding (2026), *BMC Bioinformatics*, doi:10.1186/s12859-025-06353-4 — PhyloGrad
  (stable matrix-exp-derivative via symmetric diagonalization + Hadamard form; analytic, double precision).
- Fourment et al. (2023), *GBE* 15(6):evad099 — "Automatic Differentiation is no Panacea for Phylogenetic
  Gradient Computation" (hand-coded analytic gradients ≥8× faster than AD; do NOT autodiff the traversal).
- Berger & Stamatakis (2010) — single vs double precision for ML phylogeny (single OK for search but needs
  much heavier rescaling; double still accumulates error up deep trees).
