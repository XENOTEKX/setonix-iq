# PART IV — JOLT: A GPU-Native Joint-Gradient Optimizer for ModelFinder (first-principles redesign)

**Author:** as1708 (first-principles synthesis by Claude Opus 4.8, 2026-06-08)
**Status:** ARCHITECTURE / NEW RESEARCH DIRECTION. Complements PART III (PHALANX-BMF). Builds on the
validated G.0→G.2.1b kernels + the abandoned Mode-L CPU optimizer (`mode-l-levenberg-marquardt-design.md`)
**re-read through GPU eyes**. Literature-grounded (Ji 2020; Gangavarapu 2024; torchimize/GPU-LMFit).
**Scope:** one GPU first; AA/DNA; 100K → 10M patterns; Gadi V100/A100/H100.

> **JOLT** = **J**oint **O**(N)-gradient **L**evenberg-marquardt optimizer for **T**rees.

> **One-sentence thesis.** IQ-TREE's ModelFinder optimizer (per-edge Gauss-Seidel Newton-Raphson +
> alpha-Brent + EM, alternating) is a **CPU-shaped algorithm** — low-memory, few-traversal, *sequential* —
> and it is precisely the sequentiality that starves the GPU (25 % occupancy, latency-bound, 50–100× slower
> on full `-m TEST`). The GPU-optimal algorithm is the **opposite**: compute the gradient w.r.t. **all**
> branch lengths + rate/shape parameters in **two fully-parallel traversals** (postorder + preorder, the
> Ji-2020 / Gangavarapu-2024 linear-time gradient), then take a **joint second-order step** (Levenberg–
> Marquardt / L-BFGS) over the whole ~200-dim parameter vector — trading "more FLOPs and one extra
> traversal" (which is *free* on a bandwidth-rich GPU) for the **elimination of the 197-edge sequential
> chain and the 10–20-traversal alpha-Brent line search** (which is the entire wall). This is the Mode-L
> idea that *failed on CPU* — reborn in the regime where its failure modes invert.

---

## IV.0 Why a new direction (and how it relates to PHALANX-BMF)

PART III (PHALANX-BMF) keeps IQ-TREE's **exact** optimizer (for rel-0.0 bit-parity) and attacks the wall by
running **B models at once** (`grid.z`) + a theta cache. It is the *substrate* layer: how to fill the GPU.
Its honest verdict: the 100K gap-closing is a coin-flip gated on an occupancy restructure, because batching
hides — but does not remove — the per-model sequential cost.

JOLT attacks the **other axis**: it changes *what the optimizer computes*, removing the sequential cost at
the root rather than hiding it. The two are **orthogonal and compose**: JOLT shortens each model's
critical path from "197 sequential edge-Newtons + 10–20 sequential Brent traversals per outer iteration" to
"2 parallel traversals + 1 small solve per joint iteration"; PHALANX then runs B of those concurrently.
JOLT is the deeper, riskier, higher-ceiling bet — and it is the one the user's first-principles framing
points at. **This document is the optimizer redesign; PART III remains the kernel/batching substrate.**

The conceptual pivot in one line: the design doc (II.0, II.8 G.5) parked the Ji all-branch pre-order
gradient as an **"optional advanced track, NOT on IQ-TREE's per-edge critical path."** That judgement was
correct for *replicating IQ-TREE's CPU optimizer* and **wrong for designing a GPU-optimal one.** JOLT
**promotes G.5 to the critical path** for the GPU.

---

## IV.1 First principles — the maximum-likelihood problem on a fixed tree

For a fixed topology τ with branch lengths **b** = (b₁…b_{2N−3}) and model parameters **φ** (substitution
matrix / exchangeabilities, base frequencies π, gamma shape α, invariant fraction p_inv, FreeRate rates
**r** and weights **w**), the phylogenetic log-likelihood over compressed site patterns is

```
lnL(θ) = Σ_ptn  ptn_freq[ptn] · log L_ptn,         θ = (b, φ)
L_ptn  = Σ_c prop_c · Σ_x π_x · F_root[c][x][ptn]      (Felsenstein pruning)
```

where the postorder conditional likelihoods F obey, per node, `F_parent = Π_children (P(b_child) · F_child)`
and the substitution Markov chain is `P(t) = exp(Q t) = U exp(Λ t) U⁻¹` (reversible Q ⇒ symmetric
eigendecomposition; IQ-TREE folds π into U=getEigenvectors / U⁻¹=getInverseEigenvectors and works in
**eigen coordinates**, the G.0/G.2.0a-validated convention `echild[c][x][i] = U[x][i]·exp(λ_i r_c t)`).

The ModelFinder optimization is: **maximize lnL(θ) over θ on the fixed τ**, separately for each candidate
model, then rank by AIC/BIC. θ has **~200–225 free dims** for a heavy AA model (≈197 branches + α + p_inv
+ a few rate/weight params; empirical-matrix `+F` models fix Q and π so `model.getNDim()=0` and the rate
params dominate; DNA GTR/`+FO` add exchangeability/frequency dims).

**The likelihood surface is smooth and (on a fixed topology, for these models) has a unique interior MLE.**
This is the load-bearing fact that licenses changing the optimizer: *any* correct optimizer that ascends
lnL converges to the **same** θ* (hence the same lnL, AIC, BIC, model ranking) — bit-identical trajectory
is **not** required, only the same optimum within tolerance.

---

## IV.2 What IQ-TREE does, and WHY it is shaped that way (CPU-optimal)

`ModelFactory::optimizeParameters` (`modelfactory.cpp:1558`) runs an **alternating coordinate-descent**:

1. **Branch lengths** — `optimizeAllBranches` (`phylotree.cpp:2790`): a **Gauss-Seidel** sweep visiting each
   of the 2N−3 branches *in order*, each via `optimizeOneBranch → minimizeNewton → computeFuncDerv →
   computeLikelihoodDerv` — an analytic 1-D df/ddf from the cached `theta = ∂F_node ⊙ ∂F_dad`, 2–5 Newton
   steps per edge. Each edge **reads the freshly-updated partials of the previous edge** (the Gauss-Seidel
   dependency). 75–85 % of per-model wall (Trimorph:142).
2. **Rate/shape params** — `optimizeParametersOnly`: **1-D Brent for α** (`rategamma.cpp:214`; *each Brent
   eval is a full-tree traversal* because α changes every category's rate → every P(t) → full re-prune;
   ~10–20 sequential traversals), **EM for p_inv**, **BFGS with (m+1)-traversal finite-difference gradients**
   for estimated Q/`+FO`/`+R`.
3. **Alternate** 1↔2 for ~10–20 outer iterations until convergence.

**Why this is the right CPU design** (and it genuinely is): (a) per-edge analytic NR **reuses cached
partials** so each edge is cheap and needs *no* preorder buffer → **minimal memory**; (b) Brent is
near-optimal for a 1-D α fit; (c) the alternating loop does **few full traversals**; (d) everything is
scalar/SIMD-friendly. On a CPU — DRAM-bandwidth-starved (~300–400 GB/s SPR), latency-*tolerant*, with only
~100 threads parallelised **across models** (1 thread/model, `phylotesting.cpp:6304`) — minimizing memory
and total traversals while staying sequential is exactly optimal.

---

## IV.3 Why that design is GPU-PESSIMAL (the inversion)

The GPU's cost model is the **inverse** of the CPU's on every axis:

| Resource | CPU (SPR) | GPU (V100/A100) | IQ-TREE optimizer's relationship |
|---|---|---|---|
| Bandwidth | ~300–400 GB/s (scarce) | 900 GB/s–2 TB/s HBM (abundant) | CPU minimizes traversals; GPU has bandwidth to spare → **traversal count is the wrong thing to minimize** |
| FLOPs | scarce | abundant (idle at 25 % occ) | CPU avoids redundant work; GPU **wants** more parallel work |
| Latency | tolerant (OoO, caches) | **intolerant** (kernel-launch + dependent-chain scheduling, ~85 µs/kernel) | the 197-edge Gauss-Seidel chain + Brent line search = a long **dependent** chain → GPU poison |
| Parallel work needed | ~100 threads, **across models** | **thousands** of threads, within+across models | 1-D edge-Newton + 1-D Brent expose **almost no** parallelism per step |

Measured consequences (this program): the single-model stateless GPU path is **25 % occupancy,
latency-bound** (profile 170195112), **4.7× slower than ONE CPU thread** on full `-te` (G.2.1b), and
**~50–100× too slow** on full `-m TEST` (G.2.2a: 6 models in 2.5 h). The CUDA-graph (K3) and fusion (K4)
levers came back **parity** because the bottleneck is the **dependent-kernel critical path**, not host
submission. **The CPU's "few sequential traversals" virtue is the GPU's fatal flaw.** Minimizing traversal
count (the CPU metric) is the wrong objective; the GPU objective is **minimizing critical-path length**
(dependent steps) while **maximizing parallel work per step**.

---

## IV.4 The GPU-optimal reformulation — joint analytic gradient + second-order step

Replace coordinate-descent with a **joint** optimizer over the whole θ vector, driven by the **linear-time
all-parameter analytic gradient**:

**Step A — one postorder + one preorder traversal yields the ENTIRE gradient ∇lnL(θ).**
- Postorder (= the validated K1 kernel) fills F (conditional likelihoods bottom-up).
- **Preorder** fills the complementary "rest-of-tree" conditionals `Fᵖ` top-down (Ji 2020; the Mode-L
  L.0b.ii kernel, eigen-coordinate `Fᵖ`, validated for the α gradient on +G). Then **every** branch
  derivative is read off in one pass:
  ```
  ∂lnL/∂b_e = Σ_ptn ptn_freq[ptn] · ( Σ_c w_c · r_c · qp_e[c,ptn] ) / L_ptn ,
      qp_e[c,ptn] = Σ_i λ_i · exp(λ_i r_c b_e) · Fᵖ_e[c,i,ptn] · F_e[c,i,ptn]
  ```
  This is the **O(N) all-branch gradient** (Ji 2020: replaces the O(N²) per-edge pruning gradient; **126–234×
  ML-optimization speedup**; Gangavarapu 2024 btae030: **many-core/GPU**, linear in N). The α, p_inv, and
  +R rate/weight gradients fold into the **same** preorder pass via chain rule (`∂r_c/∂α`, the p_inv O(nptn)
  scalar, the FreeRate score) — **no Brent, no EM, no (m+1) finite differences.**

**Step B — one joint second-order step over θ.** Build the **empirical-Fisher (BHHH/OPG)** curvature
`B = Σ_ptn ptn_freq · s_ptn s_ptnᵀ` from the per-pattern score vectors `s_ptn = ∂log L_ptn/∂θ` (already
produced as a by-product of Step A), damp it (Levenberg–Marquardt) and solve the small dense system
`(B + μ·diag B) δ = g` for the full update δ (ndim≈200 → a ~200×200 solve, trivial on host or cuSOLVER).
LM gives quadratic convergence near θ* and trust-region robustness far from it; **L-BFGS** (gradient-only,
no matrix) is the simpler fallback. Batched LM/Gauss-Newton on GPU is established prior art (torchimize,
GPU-LMFit).

**The transformation in one table:**

| | IQ-TREE (CPU coordinate-descent) | JOLT (GPU joint-gradient) |
|---|---|---|
| Branch lengths | 197 **sequential** edge-Newtons / sweep × several sweeps | **all 197 derivatives in 1 parallel preorder pass** |
| α (gamma shape) | 1-D Brent = **10–20 sequential full traversals** / outer iter | **1 analytic gradient component** (same preorder pass) |
| p_inv, +R | EM / (m+1)-FD traversals | analytic components, same pass |
| outer structure | alternate branches↔params, ~10–20 iters | **1 joint LM step** per iter, ~5–15 iters |
| GPU critical path / iter | ~197 + 10–20 **dependent** traversals | **2** parallel traversals + 1 small solve |
| parallel work / step | ~1 edge (tiny) | **all patterns × cats × branches** (saturating) |

**JOLT converts the optimizer from a long dependent chain into ~2 wide parallel sweeps per step** — exactly
what a latency-bound, occupancy-starved, bandwidth-rich GPU wants.

---

## IV.5 Why JOLT wins on GPU where Mode-L failed on CPU (the post-mortem, reframed)

Mode-L was the *same idea* on CPU and was abandoned (2026-05-31) for three reasons. **Each inverts on GPU:**

1. **L.1: joint LM did +34 % MORE full-tree traversals** than the alternating loop on low-dim AA (LG+G4
   19→34). *On CPU* every extra traversal is paid **serially**, so "more traversals" = slower; and the
   preorder pass is pure overhead the per-edge NR doesn't need. **On GPU the metric is wrong:** traversals
   are cheap and parallel; the cost is *critical-path length*, which JOLT **slashes** (197+Brent dependent
   steps → 2 parallel sweeps). +34 % more *parallel* traversals that remove a 200-long dependent chain is a
   massive net win. **The L.1 gate must be re-defined for GPU: measure critical-path length / GPU wall, not
   traversal count.**
2. **The FreeRate analytic gradient overflowed (~10⁵⁴), killing Mode-L.** Root cause (Mode-L §0):
   `accumulateAlphaFromPre`'s `contrib = cf·qp·exp(scale_log − _pattern_lh[ptn])` — the **per-category
   log-scaling cancellation** overflows for FreeRate's wide per-category rate spread (fine for +G).
   **On GPU we run FP64-UNSCALED in the validated eigen-space** (NORM_LH; AA-100K leafNum=100<2000 ⇒ no
   `scale_log` term at all). The exact overflow mode is **structurally absent** for the 100K/native-20
   regime. **This is the single most important re-derivation: the bug that killed Mode-L is a CPU-scaling
   artifact our GPU path does not have.** (Must be FD-validated — see G.4.0b; for >2000-taxon SAFE_LH
   regimes, use a log-sum-exp-stable per-category reduction.)
3. **Single-rank: 1 thread/model + branch-reopt = 75–85 %** ⇒ Mode-L couldn't beat FCA on CPU and was demoted
   to a "Layer-2 MPI-barrier-reduction enabler." **On GPU there is no MPI and no 1-thread/model limit:** the
   2 traversals run the *whole device* (pattern×category×branch parallel), and across models via PHALANX
   `grid.z`. The branch-reopt that dominated is now **the parallel preorder pass**, not a serial sweep.

**The throughline:** Mode-L was a GPU-shaped algorithm tried on a CPU. Its three failure modes are a CPU
metric (traversal count), a CPU numerical artifact (log-scaling overflow), and a CPU parallelism limit
(1 thread/model) — **none of which exist on the GPU path we have already validated.**

---

## IV.6 The gradient kernels (concrete, building on validated code)

- **K6 = K1** (postorder partials), reused unchanged (bit-parity rel ~1e-12 vs G.0 oracle).
- **K7 — preorder gradient-partials** (new): top-down sweep filling `Fᵖ` in eigen coordinates, per the
  Mode-L recursion (`pre_v[c,i] = exp(λ_i r_c b_v)·Σ_t U⁻¹[i,t]·preU_state[t]·fSib_state[t]`), pattern-
  parallel (the K1 thread map) + category-parallel (grid.y) + model-batched (grid.z). Leaf siblings via the
  precomputed `tip_partial_lh`. This is the kernel the design doc deferred as "G.5"; JOLT makes it primary.
- **K8 — all-parameter gradient reduction** (new): from F (K6) + Fᵖ (K7), reduce per-pattern → the full
  gradient vector **g** (all branches via the `qp` formula; α via `∂r_c/∂α` chain rule; p_inv via the
  O(nptn) scalar `(ptn_invar/p_inv − (L−ptn_invar)/(1−p_inv))/L`; +R rates/weights via the encoded-param
  score) **and** the per-pattern score matrix **S** (ndim × nptn) for the BHHH curvature. FP64, deterministic
  block-local pairwise reduction (no atomics, no fast-math).

**VRAM cost — the honest tradeoff JOLT pays.** The preorder buffer roughly **doubles** the per-model partial
arena (`Fᵖ` is the same size as F). So the B-vs-VRAM table from PART III §III.3.5 **halves**: g4 ≈ 12.3 GB
→ B=2 on V100 / B=6 on A100; r10 needs tiling on V100. This is precisely the memory cost the CPU optimizer
avoided (and why IQ-TREE doesn't keep a preorder buffer) — **the GPU pays it because bandwidth/VRAM is the
abundant resource and the latency it buys back is the scarce one.** Native-20 + pattern tiling (PART III
§III.3.5) recover B and enable 1M/10M.

---

## IV.7 The joint step (LM / L-BFGS), per-model, decoupled

Per model, per joint iteration: K6→K7→K8 give **g** and **S**; assemble `B = S·diag(ptn_freq)·Sᵀ`
(a `ndim×ndim` GEMM, ~200² — small, cuBLAS or host), solve `(B + μ diag B) δ = g` (cuSOLVER `getrf/getrs`
or host LAPACK), line-search / trust-region update μ (accept if lnL increases, else grow μ — the LM
safeguard), iterate to `‖g‖`/ΔlnL tolerance. **L-BFGS** is the matrix-free alternative (store m≈10 (s,y)
pairs; no B, no solve) — lower memory, often as good for ~200-dim smooth problems, and trivially batched.

**Decoupled across the PHALANX batch** (PART III §III.4.3): B models each run their **own** LM trust region
with their own μ, step count, and convergence flag; the batch shares only the K6/K7/K8 **kernel launches**
(grid.z), never an optimization decision. Models retire as they converge (retire-and-compact). This is the
torchimize/GPU-LMFit "many independent least-squares in parallel" pattern, specialized to phylogenetic ML.

---

## IV.8 Correctness model — same optimum, not same trajectory

JOLT **deliberately breaks** PHALANX's rel-0.0 bit-parity-of-trajectory constraint (it is a *different*
optimizer). The correctness contract is **same MLE within tolerance**:

1. **Gradient correctness is bit-parity-gated** against the CPU oracle by **finite differences** (the
   non-negotiable Mode-L lesson + the design's II.10 discipline): every component of **g** must match a
   swept-ε central difference to rel ≤ 1e-6 (and the all-branch gradient must match IQ-TREE's per-edge
   `computeLikelihoodDerv` df to rel ≤ 1e-9, which we already validated single-edge in G.2.1a). **No
   wrong-but-plausible gradient ships** (BEAGLE proved that failure; Mode-L proved the overflow one).
2. **Convergence is validated against the CPU MLE**: per-model converged **lnL rel ≤ 1e-9** and **identical
   best model + AIC/BIC ranking** vs IQ-TREE's alternating optimizer on the same fixed τ. The branch vector
   may differ at ~optimizer-tolerance (different path, same basin) — gate on lnL/ranking, **not** brlen
   bit-parity.
3. **Risk owned:** a different optimizer could land a hair off on a near-degenerate model and **flip a close
   AIC/BIC call**. Mitigation: tight convergence tolerance + a per-model A/B against the CPU lnL on the full
   candidate set at G.4.2; if any ranking flips, fall back that model to the (validated, slower) PHALANX
   stateless path. FP64 throughout; never TF32/FP16 on the reduced lnL or the gradient.

---

## IV.9 Composition with PHALANX-BMF + regime behaviour

- **They stack.** JOLT = the per-model algorithm (2 parallel sweeps + joint solve); PHALANX = the substrate
  (B models via grid.z, theta/buffer residency, router, tiling). The PHALANX kernels K1b/K2b generalise to
  K6b/K7b/K8b (add grid.z to the preorder + gradient kernels). The theta cache (PHALANX C1) becomes **less
  central** under JOLT — JOLT does not do per-edge NR, so there is no per-edge re-sweep to cache; the
  dominant cost is the 2 batched traversals, which is what the GPU wants.
- **JOLT directly fixes C3** (the 33-deep serial ladder tail): JOLT traverses the ladder ~2× per joint
  iteration (postorder+preorder) instead of 197×NR-steps×Brent times, so the ladder is hit an order of
  magnitude fewer times, and each hit is a wide parallel sweep (plus grid.z width).
- **Regime story (sharper than PART III):** JOLT's win is **largest exactly where the sequential optimizer
  hurts most** — many branches + heavy α/+R (the AA `+I+G4`/`+R` killers that dominated every CPU wall). At
  **1M/10M**, JOLT (≈2 bandwidth-bound sweeps/iter) + native-20 + tiling on one A100/H100 is the **decisive
  single-GPU win** (the "1 GPU beats 16 nodes" target), because the all-branch gradient is the *minimal*
  bandwidth-bound work and the GPU HBM ratio (2 TB/s vs 300–400 GB/s) is the lever. At **100K −m TEST**,
  JOLT closes the gap by **removing** the sequential cost rather than hiding it — a structurally stronger
  position than PHALANX's occupancy coin-flip (though it pays the preorder VRAM → smaller B).

---

## IV.10 Phased plan G.4.x (new optimizer track; strict dependency order; cheap standalone gates first)

Numbered as a **parallel track to G.3** (PART III batching). G.4.0/0b are the make-or-break standalone
gates — they re-test the **exact** thing that killed Mode-L (the FreeRate gradient overflow) on the GPU's
unscaled path, BEFORE any optimizer-driver or integration sunk cost.

| Phase | Deliverable | Gate (concrete, vs oracle) | Risk |
|---|---|---|---|
| **G.4.0 — Standalone preorder gradient kernel K7 ✅ PASS (job 170279700, 2026-06-09)** | `gpu_k7_grad.cu` = the validated K1/K2 harness + ONE new kernel `k7_pre` (top-down preorder `pre_v`); per-edge df = `theta_e=pre_v⊙pl_v` through the bit-validated `k2_derv`. All 197 branch derivatives from ONE postorder + ONE preorder sweep. | **✅ g4+g1: (1) lnL edge-invariance vs K1 oracle rel 5.8e-12/5.2e-12 (validates `pre_v`); (2) all-branch df FD rel 2.5e-8/2.0e-8 (swept-eps off-optimum); (3) central edge == G.1.2/G.2.1a K2 df.** Debug: bug#1 step-1 used `inv_evec` (stale doc pseudocode) → must be `evec`; bug#2 branch double-count → store `pre_v` w/o own branch, apply parent branch step-1. r8/r10 OOM the naive 1-buf/node arena (→ O(depth) recycling, G.4.0b). | **WAS HIGH (the Mode-L bug); now RETIRED — the gradient algorithm is proven on the unscaled GPU path.** |
| **G.4.0b — FreeRate (+R) gradient on the unscaled GPU path + O(depth) recycling ✅ PASS (job 170281211, 2026-06-09)** | `gpu_k7b_freerate.cu` = the G.4.0 K7 harness + ONE new kernel `k_ratenum` + an O(depth) pre-slot POOL. (A) Ji recycling: a single interleaved preorder DFS recycles `treeHeight+2` slots — **r8/r10 now FIT the V100** (were OOM in G.4.0). (B) the +R rate gradient `dlnL/dr_k = w_k·Σ_ptn(Σ_e b_e·qp_e[k])/L_ptn`, the exact reduction that overflowed ~10⁵⁴ on CPU, recomputed unscaled. | **✅ ALL PASS r4/r8/r10. (A) pre-pool peak = 42/44 = tree height (vs 198 nodes); g4 regression bit-IDENTICAL to G.4.0 (lnL self-inv rel 0.0, df FD 2.5e-8) ⇒ recycling numerically identical; r8/r10 lnL-inv 0.0 + oracle 3.8e-12/6.1e-12 + df FD 2.5e-8. (B1) `dlnL/dr_k` FINITE & bounded (max 3–7×10⁴, ≪ 1e8) — NO overflow; (B2) the EXACT scaling identity `Σ_k r_k·gr_k == Σ_e b_e·gb_e` to rel 5e-15…2e-13 (machine eps — ties +R grad to the validated branch grad); (B3) FD `|G-ratio|` = 1.3e-8…5.0e-8 ≪ 0.01 (the Mode-L FDCHECK that read 10⁵⁴), every category. **DECISIVE: `1/L_ptn` reaches 1e92 — FAR past Mode-L's 1e54 overflow — yet the gradient is finite because `qp∝L_p` makes `qp/L_p` self-cancel to O(100); the unscaled eigen path has no `scale_log` factor to blow it up.** Worst `lnL_ptn ≈ −212` ⇒ `L_p ≈ e⁻²¹²`, huge margin to the e⁻⁷⁰⁸ floor (confirms NORM_LH safety for 100-taxon/100K). | **WAS HIGHEST (the make-or-break); now RETIRED — the hypothesis of the whole new direction is CONFIRMED on the unscaled GPU path.** |
| **G.4.1 — Standalone joint optimiser driver ✅ PASS (job 170302036, 2026-06-09)** | `gpu_k8_jolt.cu` = the validated K1/K7/k2_derv + O(depth) pool byte-for-byte + a **joint LM-damped diagonal-Newton** driver (all 197 branches stepped at once: `b_e += df_e/(\|ddf_e\|+μ)`, the validated per-edge `ddf` as the diagonal preconditioner; accept-if-lnL-increases else grow μ — **no line search to balloon**, advisor #3) + mmap/pinned data load. α/rates FIXED at the MLE (the load-bearing +G case; joint-α = G.4.1b). | **✅ PASS g4 + g1, COLD start (b=0.1, deliberately non-optimal — advisor #1). g4: pre-check at θ* reproduces oracle (rel 5.8e-12) + calibrates ‖g‖=34.8; cold start lnL −8,008,561 (6.2% off) → MLE in **27 joint iterations**, reaching the WARM (.treefile) optimum to **rel 2.47e-16** (machine zero — same optimum, not just close); 91 dependent full-tree traversals on the critical path; 9 backtrack-rejects (no blowup); ‖g‖ 34.8→0.28 (found the branch-MLE at the fixed rates, *better* than the .treefile which is MLE at unrounded α). g1: 21 iters, cold==warm rel 1.2e-16. **HEADLINE (the JOLT thesis verdict, advisor #2): 27 cold-start joint iterations** — each ONE parallel preorder sweep updating all 197 branches — vs IQ-TREE's `optimizeAllBranches` ~197-deep × several-sweeps *sequential* Gauss-Seidel chain (un-parallelisable on GPU). The Mode-L L.1 gate, re-stated in the correct GPU metric (critical-path length, not traversal count), is decisively WON. | **WAS MEDIUM-HIGH (convergence robustness); now RETIRED for +G branches — the joint parallel optimiser converges to the same MLE from a cold start in a modest, non-blowing-up iteration count.** |
| **G.4.1b — joint α (full +G MLE from cold start)** | Add the gamma shape α to the joint vector: α-gradient = `Σ_c (∂lnL/∂r_c)·(dr_c/dα)` (the validated +R rate-grad reduction × a host FD of Yang's mean-rate gamma discretisation) → converge (b, α) cold (b=0.1, α=1.0) to the full CPU MLE −7541976.853. | **(1) cold (b,α) → −7541976.853 rel ≤ 1e-9 (or cold==warm self-consistent); (2) α folded into the joint step (NO Brent line search); (3) joint-iteration count reported.** | MEDIUM (gamma discretisation must match IQ-TREE's mean-rate variant). |
| **G.4.2 — In-tree integration (JOLT behind `--gpu --jolt`)** | Wire JOLT as an alternative `optimizeParameters` for GPU-eligible models; per-candidate fall back to PHALANX-stateless if a ranking would flip; full `-m TESTONLY` AA-100K. | **Best model == LG+G4; per-model lnL rel ≤ 1e-9; identical AIC/BIC top ranking + `MF_IGNORED` table; CPU/OFF byte-unchanged; MF wall reported vs 221.6 s.** | HIGH (optimizer-swap in the real loop). |
| **G.4.3 — Compose with PHALANX grid.z + scale regimes** | K6b/K7b/K8b (add model batch axis); B-model batched JOLT; native-20 + pattern tiling for 1M/10M. | **(1) batched == single-model per-model lnL.** **(2) −m MF AA-100K < FCA np=1 1341 s. (3) AA-1M −m MF finishes on one A100 (vs SPR FCA-np16 3 h timeout); honest speedup curve published.** | NOVEL tiling; the decisive single-GPU win. |

**Dependency:** G.4.0 → G.4.0b are the cheap kill-switches (re-test the Mode-L killer on the unscaled path).
**If G.4.0b overflows even unscaled and the stable reduction can't fix it, JOLT for +R is cut** and the
direction narrows to +G/+I+G (still the bulk of `-m TEST`) + the PHALANX path for +R. G.4.1 needs 0/0b;
G.4.2 needs 0/0b/1; G.4.3 needs G.4.2 + PHALANX G.3.0–G.3.1 (the grid.z substrate).

**Toolchain / dev tree:** unchanged (PART III §III.7 / design II.9). NOTHING committed until a gate passes.

---

## IV.11 Top risks + honesty discipline

1. **The FreeRate gradient may still be numerically hard even unscaled** (the Mode-L killer). G.4.0b is the
   hypothesis test; if it fails, +R falls back to PHALANX-stateless. **The unscaled-eigen-space-avoids-the-
   overflow claim is a HYPOTHESIS, not yet validated — flagged.**
2. **Different optimizer ⇒ could flip a near-degenerate AIC/BIC call.** Mitigation: tight tolerance + full
   candidate-set A/B vs CPU lnL + per-model fallback. Gate on ranking, not trajectory.
3. **Preorder buffer doubles VRAM → halves B** (the tradeoff for removing sequentiality). Native-20 + tiling
   mitigate; the router accounts for the 2× arena.
4. **LM curvature (BHHH) can be ill-conditioned far from θ*** → trust-region/L-BFGS fallback; never ship a
   step that decreases lnL (the LM accept/reject safeguard).
5. **The +R ladder is still sequentially dependent** (PART III §III.3.4) — JOLT speeds each rung but +R[k+1]
   still inits from +R[k].
6. **Re-state the L.1 gate for GPU.** The Mode-L L.1 "traversal count" metric is meaningless on GPU; G.4.1
   gates on **critical-path length / wall**, not traversal count. Carrying the CPU metric forward would
   wrongly reject JOLT exactly as it (correctly, for CPU) rejected Mode-L.
7. **Honesty (carried from the whole program):** FP64 parity non-negotiable; FD-validate every gradient
   (build-gating); report speedup as a curve vs pattern count; keep the CPU path byte-identical; per-model
   CPU/PHALANX fallback for anything unsupported or ranking-ambiguous.

**Bottom line.** PHALANX-BMF hides the sequential per-model cost behind batch width; JOLT **removes** it by
replacing IQ-TREE's CPU-shaped coordinate-descent with the GPU-shaped joint O(N)-gradient + second-order
step — the Mode-L idea reborn where its three failure modes (CPU traversal metric, CPU log-scaling overflow,
CPU 1-thread/model) all invert. The make-or-break is cheap and early: **G.4.0b re-tests the exact gradient
that overflowed at 10⁵⁴ on CPU, on the GPU's validated unscaled eigen-space path.** If it holds, JOLT is the
higher-ceiling path and the decisive 1M/10M single-GPU win; if it doesn't, we fall back to +G-only JOLT +
PHALANX for the rest. Either way the two architectures compose: **JOLT changes the algorithm to be
GPU-shaped; PHALANX fills the GPU with B of them.**

---

---

# IV.12 REFINEMENT (2026-06-08) — adversarial + evidence-grounded

A new architecture that claims an unoccupied niche needs **coding evidence**, not narrative. This section
grounds every JOLT claim in already-validated artifacts from this program's logs, sharpens the make-or-break
overflow argument from "hopeful" to "mechanistic," corrects the VRAM analysis (it was too pessimistic), and
red-teams the whole thing. Where JOLT is speculative it is said so.

## IV.12.1 Coding evidence — JOLT is ASSEMBLED FROM VALIDATED PARTS, not a from-scratch bet

The single most important honesty point: **4 of JOLT's 5 components are already validated; only one kernel
is genuinely new, and it is the top-down twin of a kernel we already validated bottom-up.**

| JOLT component | Already-validated artifact | Evidence | What is NEW |
|---|---|---|---|
| All-branch gradient **direction** (the Ji O(N) gradient) | The gradient **algorithm** is FD-validated on CPU | **G.0-log: g1 rel 1.07e-7, g4 rel 2.76e-3 PASS** (BEAGLE CPU plugin); "mathematically correct and FD-validated" | nothing in the math — only the kernel impl |
| Per-edge reduction `dL/db = Σ val1·pre·pl` | The `val0/val1/val2` triple reduction is **bit-validated on our OWN GPU kernel in the real binary** | **G.2.1a single-edge df/ddf GPU==CPU rel 3.99e-12/4.54e-15; G.2.1b all 197 branches rel 0.0**; Mode-L:1718 confirms `dL/db_v = Σ_i val1[i]·pre_v[i]·pl_v[i]` uses the SAME `val1=(r·eval)·val0` | supplying `pre_v` for ALL edges via ONE preorder sweep, vs K2's two-sub-root DFS per edge |
| Preorder partial kernel (`pre_v` recursion) | **Written + compile-verified + alpha-gradient FD-validated for +G on CPU** (Mode-L L.0b.ii) | Mode-L:66-69,1676,1712-1718 (the exact eigen-space recursion); the +G/+I+G alpha FDCHECK passed | port the validated recursion to the GPU eigen-space (identical machinery to the GPU-validated postorder K1) |
| Eigen-space postorder partials | **GPU-validated** | K1 rel ~1e-12 vs G.0 oracle; in-tree G.2.0a rel 1.235e-16 | reuse unchanged |
| Joint LM / L-BFGS solver | **`L_BFGS_B` ALREADY EXISTS in IQ-TREE** (Byrd-Lu-Nocedal-Zhu, m=10, `optimization.cpp:1118`); `dfpmin` BFGS at :750 | source-verified (lbfgs-and-warmstart doc §1.1) | feed it the analytic JOINT gradient instead of finite differences |

**Conclusion:** JOLT's risk is concentrated in exactly one new kernel — the GPU preorder sweep `K7` — which
is the top-down mirror of the postorder `K1` we have already validated on the GPU, computing a quantity
(`pre_v`) whose downstream reduction (`val1·pre·pl`) we have already validated bit-identically on the GPU
(G.2.1a/b). **The gradient math is CPU-FD-proven (G.0); the reduction is GPU-bit-proven (G.2.1); the solver
exists (optimization.cpp:1118).** This is the opposite of a speculative rewrite.

## IV.12.2 The FreeRate overflow — from "hopeful" to MECHANISTIC (the make-or-break, sharpened)

Mode-L died on `contrib = cf·qp·exp(scale_log − _pattern_lh[ptn])` overflowing to ~10⁵⁴ for +R
(`phylotree.cpp:1437,1442`). Decompose **why the GPU unscaled path is structurally different**, term by term:

1. **`scale_log` is the accumulated per-category LOG-SCALING factor.** On the GPU we run **FP64-UNSCALED**
   (NORM_LH; AA-100K leafNum=100<2000, validated in vivo G.2.0a) ⇒ **`scale_log ≡ 0`**. The term collapses
   to `contrib = cf·qp·exp(−_pattern_lh) = cf·qp / L_p`. **The overflow factor literally does not exist.**
2. **`qp = Σ_i λ_i·exp(λ_i r_c b)·pre·pl`.** Rate-matrix (CTMC generator) eigenvalues **λ_i ≤ 0**; r_c, b > 0
   ⇒ **`exp(λ_i r_c b) ∈ (0,1]` — bounded, cannot overflow.** (This is why the *likelihood* never overflowed,
   only Mode-L's `exp(scale_log−lh)` term did.)
3. **`qp ∝ L_p` in magnitude** (qp is ∂L_p/∂param), so **`qp/L_p` is O(1)** — a well-conditioned ratio even
   when both numerator and denominator are tiny.
4. **The only residual risk is UNDERFLOW (`L_p → 0`), not overflow** — and for AA-100K per-site
   `L_p ≈ e⁻⁷⁵ ≫ ` double's `e⁻⁷⁰⁸` floor (the very reason the unscaled lnL is bit-validated). So the
   100K/native-20 regime is safe by the same argument that already validated the likelihood.
5. **HONEST caveat:** >2000-taxon trees re-enter SAFE_LH scaling ⇒ `scale_log` returns ⇒ the overflow risk
   returns *there*. For that regime, compute the gradient in **log-space with sign tracking**
   (log-sum-exp-stable per-category reduction) — a known fix, deferred until the >2000-taxon regime is needed.

⇒ The claim "GPU unscaled avoids the Mode-L overflow" is now **mechanistic** (no `scale_log`; `λ_i≤0` bounds
the exp; `qp/L_p` is conditioned; AA-100K is far from underflow), not hopeful. **It is still a hypothesis
until FD-checked** — which is precisely what **G.4.0b** does (`|G-ratio|<0.01` on LG+R4, the exact Mode-L gate
that read 10⁵⁴). This is the cheapest, most decisive experiment in the whole plan.

**✅ CONFIRMED (job 170281211, 2026-06-09).** The G.4.0b run validated this mechanistically AND empirically on
LG+R4/+R8/+R10. The decisive observation: `1/L_ptn` (the term that, multiplied by Mode-L's extra `exp(scale_log)`,
hit 10⁵⁴) **reaches 1e92 here** — *larger* than the Mode-L overflow magnitude — yet **`max|rnum/L_ptn|` stays at
O(100)** and `max|dlnL/dr_k| ≈ 3–7×10⁴` (finite, ≪ 1e8). The self-cancellation is exactly point 3 above: `rnum`
(the `Σ_e b_e·qp_e[k]` numerator) is computed on-device in linear FP64 as a sum of `λ_i≤0`-bounded terms (max
`|rnum| ≈ 0.8`, no overflow), and the `÷L_ptn` is the matched O(1) ratio. The worst per-pattern `lnL_ptn ≈ −212`
⇒ `L_p ≈ e⁻²¹²`, a vast margin to the `e⁻⁷⁰⁸` FP64 floor — so the NORM_LH (leafNum<2000) regime is safe by the
same argument that validated the likelihood. Two independent absolute checks closed it: the **exact scaling
identity** `Σ_k r_k·dlnL/dr_k == Σ_e b_e·dlnL/db_e` held to rel 5e-15…2e-13 (ties the +R grad to the bit-validated
branch grad, no FD needed), and **per-category FD** `|G-ratio| = 1.3e-8…5.0e-8` (the exact Mode-L FDCHECK, now
8 orders of magnitude inside its 0.01 gate). **The honest caveat (point 5) is undischarged: this validates only
NORM_LH; >2000-taxon SAFE_LH reintroduces `scale_log` and needs the log-sum-exp-stable reduction.**

## IV.12.3 VRAM — CORRECTED: O(depth) recycling, not 2× (this STRENGTHENS JOLT)

§IV.6 said the preorder buffer "doubles" the per-model arena. **That is the naive one-buffer-per-node
allocation** — and exactly what made BEAGLE's gradient need 50.3 GB (A100-only, 402 buffers, 32-pad;
G.0-log:224-246). The G.0-log states the fix verbatim (line 227): **"Ji's O(N) needs only O(depth) live
pre-partials, not one/node."** Our real ML tree is **height-42 with 98 internal nodes**, so with
depth-recycled preorder buffers the extra arena is **≈ depth/n_internal ≈ 42/98 ≈ 0.43×** the postorder —
**under half**, not 1.0×. The PART III B-vs-VRAM table therefore shrinks by ~1.43×, not 2×: g4 ≈ 8.8 GB
(B=3 on V100 / B=8 on A100), not 12.3 GB. (This recycling is the design-doc's parked "G.5 O(depth) buffer
pool" — JOLT promotes it.) **Honest cost:** recycling adds scheduling logic (free a node's pre-partial once
both children have consumed it) — bounded, and the depth-42 ladder makes the live set small.

## IV.12.4 Adversarial red-team (attack JOLT; concede or defend each, with evidence)

1. **"Joint LM did +34% MORE iterations on CPU (Mode-L L.1) — maybe it just needs more iterations, not
   fewer."** **CONCEDE** the converged iteration count is the key empirical unknown. **DEFEND:** the GPU win =
   (critical-path reduction/iter) × (iteration-count ratio). Mode-L L.1 measured LG+G4 19→34 iters (+79%
   worst case); at 1.79× more iters but ~100× shorter critical path per iter (197+Brent dependent traversals
   → 2 parallel sweeps), net ≈ 56×. The bet fails ONLY if the iter count *blows up* (>10×), which a
   line-searched LM/L-BFGS should not. **GATE G.4.1 measures the real converged iter count** — this is the
   honest decider, and it re-states the Mode-L L.1 gate in the correct GPU metric (critical-path, not
   traversal count).
2. **"A joint/Jacobi all-branch step may oscillate or land in a different basin."** **DEFEND:** damped Newton
   (LM μ) + accept-only-if-lnL-increases guarantees **monotone ascent** to the unique interior MLE on a fixed
   topology; the per-branch diagonal `ddf` (cheap, validated in K2) damps each step. **CONCEDE:** convergence
   *rate* is empirical (G.4.1); if Jacobi oscillates, fall back to colored Gauss-Seidel waves.
3. **"The empirical-Fisher (BHHH) curvature is a 227×nptn score matrix = ~174 MB/model — it kills batch B."**
   **CONCEDE** for full BHHH. **DEFEND:** default to **L-BFGS** (matrix-free; m=10 (s,y) pairs ≈ 227×10×8×2 ≈
   36 KB/model) — **already in IQ-TREE (`optimization.cpp:1118`)**. Reserve the dense LM solve for the small
   model-param sub-block (n≤30, where dense methods were already studied — lbfgs doc). So JOLT's default is
   **L-BFGS over the joint vector**, not BHHH-LM. The name keeps "LM" for the optional second-order sub-block.
4. **"+R is where JOLT most needs the high-dim win AND where the overflow lives AND where `filterRates`
   prunes it early on AA, so it rarely even runs."** **HONEST:** correct — on AA `-m TEST`, `filterRates`
   prunes +R classes before evaluation (lbfgs doc §12.8: the warm-start cache regressed because +R never
   ran). So JOLT's **+R coverage matters most for `-m MF` / DNA / `-mrate`, not the core AA `-m TEST` win**,
   which is dominated by +G/+I+G — the **overflow-free, CPU-alpha-FD-validated** path. ⇒ the G.4.0b kill-switch
   gates the `-m MF`/DNA *extension*; the AA `-m TEST` win does not depend on it. This is a strength, not a
   weakness: the riskiest piece (+R gradient) is NOT on the critical path for the headline benchmark.
5. **"Same-optimum-not-same-trajectory could flip a near-degenerate AIC/BIC call."** **CONCEDE** — a real
   risk. **MITIGATE:** tight convergence tolerance + a full candidate-set A/B of JOLT lnL vs CPU lnL at
   G.4.2; any model whose ranking would flip falls back to the (validated, bit-parity) PHALANX-stateless
   path. FP64 throughout; never TF32/FP16 on the reduced lnL or gradient.
6. **"You claim 'better than BEAGLE/BEAST' — substantiate or retract."** **HONEST LANDSCAPE** (web + the
   FCA-landscape transcript in `updated-modelfinder-dispatch.md`): **BEAGLE** is a likelihood *library* — one
   model per instance, no model selection, and its **GPU gradient is broken for 20-state** (G.0 bug 16).
   **BEAST** is *Bayesian* (HMC sampling; it uses the Ji gradient via BEAGLE for posterior sampling, NOT ML
   model selection). **ModelTest-NG** dispatches a *fixed, non-prunable* model grid (every model evaluated,
   no GPU, no `filterRates`). **RAxML-NG has NO ModelFinder** (GTR default + ModelTest-NG as a separate step;
   its MPI is site-parallel within one tree). ⇒ **NO existing tool does GPU-accelerated ML ModelFinder with
   cross-model batching + a joint analytic-gradient optimizer.** The honest claim is **not** "faster BEAGLE at
   BEAGLE's job" — it is "a novel architecture in a niche no tool occupies": a GPU model-selection engine.
   That is defensible and bleeding-edge; the inflated framing is not.
7. **"Every prior speedup lever in this program (CUDA graph, fusion, register caps, Mode-L) came back
   parity-or-worse — why is JOLT different?"** **CONCEDE** the base rate is humbling. **DEFEND:** those levers
   all attacked *secondary* costs (host submission, launch count, occupancy knobs) while leaving the
   **sequential dependency structure** intact; JOLT is the first lever that attacks the structure itself
   (the 197+Brent dependent chain). And unlike Mode-L, JOLT runs in the regime where Mode-L's three failure
   modes provably invert (§IV.5). **But the base rate is why G.4.0/0b/1 are cheap standalone kill-switches
   BEFORE any integration sunk cost.**

## IV.12.5 The minimal first experiment (sharpened — it reuses validated code)

**G.4.0 reuses the G.2.1a harness as its oracle.** `gpuComputeEdgeDervCleanRoom` already computes the
single-edge df via the two-sub-root extraction, **bit-validated == CPU (rel 3.99e-12)**. G.4.0 computes the
SAME df for **all 197 edges via one preorder sweep** (K7) and checks it equals the per-edge result — whose
correctness we ALREADY own. **So G.4.0 needs no new oracle: its ground truth is our own validated GPU
single-edge output.** This is the cheapest possible first step and the tightest possible de-risk: if the
preorder all-branch gradient matches the validated per-edge gradient to rel ≤ 1e-9 for all 197 edges, the
single new kernel is proven, and JOLT rests entirely on already-validated pieces + the existing solver.

---

## References (this part)
- Ji, Zhang, Holbrook, Nishimura, Baele, Rambaut, Lemey, Suchard (2020), *MBE* 37(10):3047 — "Gradients do
  grow on trees: a linear-time O(N)-dimensional gradient" (arXiv 1905.12146). **126–234× ML-opt speedup.**
- Gangavarapu, Ji, Baele, Fourment, Lemey, Matsen, Suchard (2024), *Bioinformatics* 40(2):btae030
  (arXiv 2303.04390) — **many-core/GPU** linear-time all-branch-length gradient.
- Random-Effects Substitution Models via scalable gradient approximations (2024), PMC11498053 — gradients
  for substitution-model parameters.
- torchimize (github.com/hahnec/torchimize) — batched Gauss-Newton / Levenberg-Marquardt on GPU (PyTorch);
  GPU-LMFit — scalable parallel LM on CUDA. **Prior art for B independent LM problems in parallel.**
- Fourment et al. (2023), *GBE* 15(6):evad099 — hand-coded analytic gradients ≥8× faster than autodiff
  (⇒ hand-code the Ji gradient; do NOT autodiff the traversal).
- Gangavarapu & Suchard (2025/26), *Syst. Biol.* syag017 — FP64 tensor cores for AA likelihood (bandwidth-
  bound; informs why the NS=20 contraction stays a hand kernel).
- "Much Ado About Nothing: early stopping" (2024), bioRxiv 2024.07.04.602058 — converge-to-MLE, avoid
  over-optimization (supports the same-optimum-not-same-trajectory contract).
- Internal: `mode-l-levenberg-marquardt-design.md` (the CPU attempt + its decisive negative result);
  `gpu-modelfinder-design.md` PART II (kernels); `gpu-modelfinder-part3-architecture.md` (PHALANX-BMF).
