# Phase G.0 Execution Log — BEAGLE GPU de-risk (single-GPU ModelFinder)

Companion to [gpu-modelfinder-design.md](gpu-modelfinder-design.md). Running record of the G.0
de-risk: does one heavy AA model run materially faster on one GPU (BEAGLE) than on CPU, at parity?

**Date:** 2026-06-01. **GPU:** V100 (`gpuvolta`) first, per user. **Model:** LG+G4 (the AA-100K winner).

---

## Decisions & corrections

- **Reuse existing CPU baselines — do NOT re-run** (user, firm): the 100K AA CPU ModelFinder data
  already exists from FCA testing. References used:
  - **FCA np=1, AA-100K** (`logs/runs/gadi_AA_100k_fca_np1_full_seed1_169095077.json`): best=LG+G4,
    lnL **−7,541,976.861**, **MF wall 258.8s**, SPR 738.6s, total 1000.8s.
  - **Gate 169643959** (`-m TEST` BASE arm, AA-100K, mode-p-iso binary): best=LG+G4, lnL
    **−7,541,976.853**, MF 418s. Its `.treefile` is the **fixed reference tree** for the harness;
    its lnL is the **precise parity target**.
  - (Three redundant baseline jobs were briefly submitted then immediately `qdel`'d — ~2 min each,
    negligible SU — once the user pointed out the existing data.)
- **Parity target:** GPU/BEAGLE lnL of the reference tree under LG+G4 must match **−7,541,976.853**
  (to ≤1e-3 relative) — the gate of correctness before any speed claim.

## The harness: `gadi-ci/gpu-modelfinder/gpu_derisk.cpp`

Standalone `libhmsbeagle` (4.0.1) program; builds with `icpx -O2 -std=c++17 -lhmsbeagle`. Switchable
CPU vs CUDA plugin (`cpu|gpu` arg). Parity-critical pieces, all matched to IQ-TREE:
- **LG matrix + freqs** embedded verbatim from IQ-TREE src `pll/models.c` (190 exchangeabilities +
  20 freqs, standard AA order `ARNDCQEGHILKMFPSTWYV`).
- **Reversible-model eigendecomposition** (IQ-TREE's method): Q=R·diag(π), mean-rate-normalised;
  symmetrise `B=diag(√π)·Q·diag(1/√π)`; symmetric Jacobi eigensolver (self-contained, no MKL link);
  `U=diag(1/√π)·V`, `U⁻¹=Vᵀ·diag(√π)`.
- **Gamma:** exact 4-cat MEAN rates `{0.1362,0.4756,0.9994,2.3887}`, weights 0.25 (alpha 0.9963) —
  taken verbatim from the reference `.iqtree` to guarantee gamma-discretisation parity.
- **Tip data:** all 100K sites used as patterns, weight 1 (Σ-over-sites lnL == pattern-compressed
  lnL, so parity holds without replicating IQ-TREE's compression); AA→20-state partials, ambiguous
  (B/Z/J/X/gap) → all-ones partials.
- **Tree:** general Newick parser; N-ary (trifurcating) root handled by pairwise `updatePartials`
  combines with a zero-length **identity** transition matrix for the 3rd+ child.
- **Scaling:** `BEAGLE_FLAG_SCALING_MANUAL | SCALERS_LOG`; per-operation scale buffers accumulated
  (`beagleResetScaleFactors`+`beagleAccumulateScaleFactors`) into a cumulative buffer for the root
  lnL — required or the 100-taxon partials underflow to −inf.

## Bugs found & fixed during bring-up

1. **Tip-partial indexing (first attempt — WRONG, superseded by bug 4)** — partials were initially
   set at alignment-tip index; operations reference parser **node IDs**, so I switched tip partials
   to node-ID buffers. That was the wrong direction (see bug 4): BEAGLE forbids tip indices ≥
   tipCount.
2. **No scaling** → −inf lnL on a 100-taxon tree. Added manual scaling (above).
3. **Login-node OpenCL crash** — `beagleCreateInstance` segfaults during BEAGLE's resource
   enumeration on the **login node** (no GPU driver/OpenCL ICD): `GPUInterfaceOpenCL.cpp:105`.
   ⇒ the CPU-plugin parity check can't run on login either. Resolution: run **both** plugins inside
   the `gpuvolta` job, where the driver stack makes enumeration succeed. (A later refinement could
   restrict `resourceList` to avoid the probe.)
4. **Tip-buffer index OUT_OF_RANGE (the real one — found by instrumentation, job 169679354).**
   Symptom: `beagleSetTipPartials` returned **rc=−5 (`BEAGLE_ERROR_OUT_OF_RANGE`) for every tip**,
   leaving tip buffers unset → garbage partials → **segfault** (CPU) / **CUDA 700** (GPU) downstream
   in the compute, *after* instance creation and setup succeeded. Root cause: BEAGLE's hard contract
   — **`beagleSetTipPartials`/`beagleSetTipStates` require the buffer index ∈ `[0, tipCount)`**; tip
   buffers are reserved at the bottom of the index space. The Newick parser assigns DFS node IDs
   (root=0, interleaved), so tips land at IDs ≥ tipCount → rejected. Bug 1's "use node IDs" was
   exactly backwards. **Fix:** a buffer-index remap applied consistently everywhere —
     - leaf node → its alignment tip index (already unique in `[0, ntax)`),
     - internal node → `tipCount, tipCount+1, …` (in `[tipCount, nnodes)`),
     - N-ary-root scratch → `nnodes, nnodes+1, …`,
   and `buf[node]` used for tip-partials, `BeagleOperation` partial refs, transition-matrix indices,
   and the root index passed to `calculateRootLogLikelihoods`. Buffer counts (`partialsBufferCount =
   nnodes+rootdeg+2`, with `compactBufferCount=0` so the index space is `[0, partialsBufferCount)`)
   already cover this. Also added: a setup-error counter that **aborts before compute** if any BEAGLE
   setup call failed (turns the confusing downstream segfault into a clean diagnostic), and the build
   now happens **in-job** (`icpx` in the PBS module env) so the running binary always matches source.

## Runs

| Job | Node | Arm | Result |
|---|---|---|---|
| (login) | login | cpu plugin | ✗ crash in resource enumeration (no GPU stack) — see bug 3 |
| 169678407 | gpuvolta (V100) | cpu + gpu | ✗ instance OK, both plugins crash in compute (CPU segfault / CUDA 700) before lnL — uninstrumented |
| 169679354 | gpuvolta (V100) | cpu + gpu | ✗ **instrumented** → localized: `setTipPartials` rc=−5 OUT_OF_RANGE for every tip (bug 4) → garbage → crash |
| 169679397 | gpuvolta (V100) | cpu + gpu | ✗ in-job build failed — wrong `-I` (header is under `include/libhmsbeagle-1/`, not `include/`) |
| **169679422** | gpuvolta (V100) | cpu + gpu | ✅ **PARITY + 29× SPEEDUP** (LG+G4, tip-partials) — see results below |
| 169679582 | gpuvolta (V100) | g4/fig4/r10 | g4+fig4 bit-parity; r10 GPU instance fails (NCAT=10) — first sight of finding 7 |
| 169680581 | gpuvolta (V100) | compact tips | compact tip states: VRAM 19.9→10.4 GB, g4 GPU 121→86 ms; r10 still fails (not VRAM) |
| 169680647 | gpuvolta (V100) | scaled vs unscaled | unscaled lnL bit-identical to scaled (finding 6); g4 GPU 86→45 ms; r10 fails even unscaled |
| **169680691** | gpuvolta (V100) | g4/fig4/r8/r10/r10split | ✅ **heavier models complete** — all bit-parity, 43–66×; r10split (5+5) Δ=0.0000 exact, 57× |

## ✅ G.0 RESULT (job 169679422, 2026-06-01) — de-risk PASSES

| | CPU plugin (`CPU-SSE-Double`) | GPU plugin (V100 `CUDA-Double`) |
|---|---|---|
| lnL | **−7541976.9391** | **−7541976.9391** |
| lnL eval, min | 3521.7 ms | **121.5 ms** |
| lnL eval, mean (reps) | 3623.1 ms (×5) | 122.0 ms (×30) |
| warmup | 4396 ms | 147 ms |

- **Parity:** vs IQ-TREE reference **−7541976.853**, |Δ| = 0.086 absolute → **relative 1.1e-8**, far
  inside the 1e-3 gate ⇒ **PASS**. The CPU and GPU plugins agree to displayed precision (both
  −7541976.9391), so the harness is internally exact; the ~0.086 residual vs IQ-TREE is a model-spec
  rounding (gamma category rates hardcoded to 4 dp from the `.iqtree`), not a likelihood-kernel bug.
- **Speedup ≈ 29×** per full lnL evaluation (postorder partials over 99 internal ops + 197 transition
  matrices + root). This is a *controlled* number: identical harness/code path, only the BEAGLE
  plugin differs (single-instance CPU-SSE vs one V100). It is **not** "29× vs IQ-TREE's AVX/threaded
  kernel" — that comparison needs a single-lnL timing from IQ-TREE itself (a later refinement).
- **VRAM = 19.93 GB** for AA-100K (PBS `GPU Memory Used`). Confirms the design's prediction that the
  binding constraint flips to **VRAM**: AA-100K fits a 32 GB V100 comfortably; **AA-1M (~10× partials)
  would exceed it** ⇒ pattern tiling on V100, or use A100-80GB / H100. Job SU = 0.41 (trivial).
- **Caveats for honesty:** (a) BEAGLE CPU-SSE is single-threaded per instance — not the multicore SIMD
  IQ-TREE uses, so the 29× overstates the real-vs-IQ-TREE gap; (b) lnL eval here re-derives transition
  matrices each call (as MF does when params change) but does **not** include eigendecomposition (done
  once on host); (c) no gradient yet — next.

## Bugs / findings 5–7 (heavier-model bring-up, 2026-06-01)

5. **Compact tip states >> full tip partials (perf + VRAM).** Initially set tips via
   `beagleSetTipPartials` (full NS×NCAT×nptn buffer/tip). Switched to `beagleSetTipStates` (compact:
   one int/site, NCAT-independent) with `compactBufferCount=ntax` and `partialsBufferCount` reduced to
   internals+scratch. Effects on g4: **VRAM 19.9 → 10.4 GB**, GPU lnL eval **121.5 → 86 ms** (BEAGLE's
   state kernels are faster than partial kernels), lnL **unchanged** (bit-identical — tips are
   unambiguous states; ambiguous→state `NS`=20 = "any", same as the old all-ones partial). This is how
   IQ-TREE/BEAST actually run.
6. **Double precision needs NO scaling on this tree (perf).** Manual log-scalers were added (bug 2) to
   stop underflow. But with `BEAGLE_FLAG_PRECISION_DOUBLE`, the 100-taxon partials do **not** underflow
   (per-site L≈e^−75, well inside double's e^−708): unscaled lnL is **bit-identical** to scaled for g4
   and fig4. Dropping scaling is also ~2× faster (g4 GPU **86 → 45 ms**; fewer kernels). ⇒ run the GPU
   hot path unscaled in double precision. (Single precision *would* need scaling — relevant for a future
   float/VRAM-halving variant.)
7. **BEAGLE 4.0.1 CUDA hard-caps category count at `kMatrixBlockSize` (=8) for 20 states.** `r10`
   (NCAT=10) makes `beagleCreateInstance` print *"Not yet implemented! Try slow reweighing."* and
   `exit(-1)` — **unconditionally, regardless of scaling flags or VRAM** (only 10 GB used). Root cause
   (BEAGLE src `KernelLauncher.cpp::SetupKernelBlocksAndGrids`): the rescale-kernel grid
   `bgScaleGrid.y = kCategoryCount / kMatrixBlockSize`; when `>1` (i.e. NCAT>8) the fast path is
   unimplemented and the "slow reweighing" alternative *also* hard-fails when combined with scale
   accumulation. It is **category-driven, not pattern-driven** — pattern tiling does NOT help. `r8`
   (NCAT=8) works, pinning the boundary. **Workaround (validated, see below): category-splitting** —
   evaluate the k>8 rate categories as ≤8-category sub-passes (same instance, re-set rates/weights
   between passes), retrieve per-site lnL via `beagleGetSiteLogLikelihoods`, combine
   `L_site = Σ_groups exp(logL_group)` then `lnL = Σ_site log L_site`. Mathematically identical to a
   single NCAT=k model ⇒ bit-exact. For the real GPU ModelFinder, +R9..+R10 either category-split or
   need a BEAGLE rebuild with larger `kMatrixBlockSize` / the slow-reweighing kernel implemented.

## ✅ Heavier-model results (job 169680691, V100, 2026-06-01) — parity holds, speedup holds

CPU plugin = scaled (always works); GPU = **unscaled double precision** (finding 6). Parity = CPU≡GPU.

| Model | NCAT | CPU lnL | GPU lnL | parity | GPU eval | CPU eval | **GPU speedup** |
|---|---|---|---|---|---|---|---|
| `g4` LG+G4 | 4 | −7541976.9391 | −7541976.9391 | **bit-identical** | 45.1 ms | 2973 ms | **66×** |
| `fig4` LG+F+I+G4 | 5 | −7551774.2140 | −7551774.2140 | **bit-identical** | 56.3 ms | 3240 ms | **58×** |
| `r8` LG+R8 | 8 | −7556251.9185 | −7556251.9185 | **bit-identical** | 89.9 ms | 3855 ms | **43×** |
| `r10` LG+R10 | 10 | −7554280.5776 | ✗ instance fails | — | — | (CPU 4272 ms) | — (finding 7) |
| `r10split` LG+R10, 5+5 | 5×2 | −7554280.5776 | −7554280.5776 | **Δ=0.0000 exact** | 114 ms (2-pass) | 6488 ms (2-pass) | **57×** |

- **`fig4`/`r8` use representative (not IQ-TREE-fitted) +I/+F/+R params** — the rigorous GPU-correctness
  test for heavier models is CPU-plugin ≡ GPU-plugin on the same fixed tree, which holds **bit-exactly**.
  (`g4` additionally matches IQ-TREE's own number to 1e-8.)
- **The +R10 long-pole is GPU-tractable.** `r10split` proves the category-split workaround is **bit-exact**
  (Δ=0.0000 vs the NCAT=10 CPU lnL) and still **57×** faster than CPU even with the 2-pass overhead. This
  is the model class that killed every CPU dispatch architecture (Amdahl pole) — on one GPU it is ~57×.
- VRAM (r10split, NCAT=5 + per-site buffers): 20.4 GB. Job SU = 1.25.

## Verdict (lnL + heavier models)

**G.0 de-risk PASSES decisively.** On one V100, BEAGLE computes AA-100K likelihoods at **bit-parity with
the CPU plugin** (and 1e-8 of IQ-TREE for LG+G4) across LG+G4, LG+F+I+G4, LG+R8, and LG+R10 (via 5+5
split), at **43–66× the CPU-SSE eval rate**. Double precision needs no scaling here; compact tip states
cut VRAM/time; the only wrinkle is BEAGLE's NCAT≤8 CUDA cap, surmounted bit-exactly by category-splitting.
The GPU-ModelFinder bet is strongly supported. Remaining G.0 work before G.1 (CUDA-Graph): **gradient timing**.

## Gradient bring-up (2026-06-05) — bugs 8–10

8. **`beagleCalculateEdgeDerivative` (singular) is deprecated in BEAGLE 4.0.1** — it prints
   *"Depricated"* and returns `BEAGLE_ERROR_FLOATING_POINT` (-7) without computing anything; all
   gradient values stay 0. Fix: use `beagleCalculateEdgeDerivatives` (plural, the BEAGLE 4 API).
9. **`partialsBufferCount = 2*nnodes + 4 = 402` causes `beagleCreateInstance` to fail on both CPU
   and GPU** — 402 partial buffers × nptn × NCAT × NS × 8B = 25.7 GB exceeds available CPU RAM (silently)
   and V100 VRAM. Correct formula: `2*nnodes - ntax + 4 = 302` (19.3 GB, fits).
   Derivation: BEAGLE partial buffer global index = `tipCount + j` where `j ∈ [0, partialsBufferCount)`;
   preB_(id) = nnodes + id so j = (nnodes-ntax) + id; max j = 2*nnodes - ntax - 1 = 297 < 302 ✓.
10. **GPU `beagleSetRootPrePartials` returns rc=-7 (BEAGLE_ERROR_FLOATING_POINT)** — possibly this
    function is not implemented in the BEAGLE 4.0.1 CUDA backend. CPU succeeds; GPU fails. The
    Gangavarapu 2024 paper may require a newer BEAGLE build. Pending confirmation (job 170093588).

## Gradient diagnosis (2026-06-06) — root causes nailed against BEAGLE 4.0.1 source (bugs 11–13)

Job 170093588 reproduced two distinct failures inside `evalGrad()`: **CPU plugin hard SEGFAULT**, and
**GPU `setRootPrePartials` rc=-7** then `CUDA "Out of memory" at GPUInterfaceCUDA.cpp:596`. A 4-way
parallel source investigation (raw `beagle-dev/beagle-lib` v4.0.1: `BeagleCPUImpl.hpp`,
`BeagleGPUImpl.hpp`, `GPUInterfaceCUDA.cpp`, `examples/hmctest/hmctest.cpp`) + direct WebFetch
verification pinned all root causes. **Two earlier guesses were wrong** and are corrected here:

- **Bug 8 correction:** rc=-7 is `BEAGLE_ERROR_NO_IMPLEMENTATION`, **not** `_FLOATING_POINT` (that is -8).
- **Bug 10 correction:** `setRootPrePartials` is a *pure CUDA stub* — `return BEAGLE_ERROR_NO_IMPLEMENTATION;`
  in `BeagleGPUImpl.hpp`. It is NOT an OOM and NOT a numerical issue. **But** `updatePrePartials`
  (→`upPrePartials`) and `calculateEdgeDerivatives` (→`calcEdgeFirstDerivatives`) **are** fully
  implemented on CUDA. So only the root-pre *seeding* convenience fn is missing — see bug 13 workaround.

11. **CPU SEGFAULT = `outDerivatives` heap overflow (THE crash).** `beagleCalculateEdgeDerivatives`'
    7th arg `outDerivatives` is the **per-edge × per-pattern** array; `BeagleCPUImpl::accumulateDerivatives`
    writes `outDerivatives[k]` for `k∈[0,kPatternCount)` at offset `nodeNum·kPatternCount`, i.e. it needs
    `count·kPatternCount = 198·100000` doubles. The harness passed a **198-element** `grad` buffer →
    heap overflow → SIGSEGV *inside* libhmsbeagle (so no rc could be checked). **Fix:** pass `outDerivatives = NULL`
    (it is optional; the `DoDerivatives=false` template path skips it). The gradient is `outSumDerivatives`
    (size = `count` = nE). `outSumSquaredDerivatives` also NULL.
12. **Wrong differential matrix (correctness — would FD-fail even without the crash).** The plural
    `beagleCalculateEdgeDerivatives` consumes a **differential (infinitesimal-generator) matrix** set via
    `beagleSetDifferentialMatrix` — *not* the dP/dt computed through `beagleUpdateTransitionMatrices(…firstDerivativeIndices…)`.
    Evidence: `setDifferentialMatrix → setTransitionMatrix(idx,m,0.0)`; `calcEdgeLogDerivativesPartials`
    **ignores its `categoryRates` argument**, so the per-category rate must be **baked into** the matrix.
    Reference `hmctest.cpp` builds `scaledQ[c·S²+i·S+j] = Q[i][j]·rate[c]` (Q = generator, negative diag,
    rows sum 0) and sets it **once** (branch-independent). **Fix:** `diff[c·NS²+i·NS+j] = Q[i][j]·catRates[c]`
    using the SAME normalized Q from the eigendecomposition; `beagleSetDifferentialMatrix(inst,dQidx,diff)`;
    point every `derivativeMatrixIndices[k]` at the single `dQidx`. Drop the dP/dt computation entirely.
    Pre-order **op convention is CORRECT as-is** (independently verified vs `upPrePartials` kernel + Ji et al.
    Eq.7: mat1 = edge-of-child matrix applied transposed against parent pre-partial; mat2 = sibling edge
    matrix forward against sibling post-partial; top-down order).
13. **GPU needs (a) a root-seed workaround and (b) the transpose flag.** (a) Since CUDA `setRootPrePartials`
    is a stub, seed the root pre-order partial directly: `beagleSetPartials(inst, preB_(root), freqs)` with
    the state-frequency vector replicated over every pattern×category (`seed[(c·nptn+p)·NS+s]=f[s]`).
    (b) For `stateCount>4`, the CUDA pre-order kernel only auto-transposes the transition matrices when
    `BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO` is set (`kUsingAutoTranspose = kPaddedStateCount>4 && flag`); the
    harness must pass it at create (CPU ignores it). *Open:* whether AUTO also transposes the differential
    matrix on GPU — `hmctest` used a transposed index for it, so a Qᵀ·r_c variant may be needed if the GPU
    FD-check fails. GPU VRAM: AA padded **20→32 states**, so a partial buffer = `nptn·32·NCAT·8` = 102.4 MB
    (NCAT=4); 302 buffers ≈ 30.9 GB (observed 29.4 GB) just fits 32 GB — the edge-deriv scratch is only
    `kPaddedPatternCount·kBufferCount·2·8` ≈ 643 **MB** (not GB), so the V100 OOM is the thin partials margin,
    fixable by recycling pre-order buffers (Ji's O(N) needs only O(depth) live pre-partials, not one/node).

14. **BEAGLE `createInstance` enumerates the GPU/OpenCL stack even for a CPU-plugin request** → on a
    pure-CPU node it dies at `GPUInterfaceOpenCL.cpp:105` ("OpenCL error: Unknown error", exit 255) before
    any compute (job 170128687 on `normal`/clx). Same root as bug 3 (login node). **Consequence:** even
    CPU-plugin validation must run on a node that has the GPU/OpenCL stack — i.e. **gpuvolta**. The
    "frugal CPU-only on the `normal` queue" plan does not work; CPU FD reference + GPU run share one
    gpuvolta job (cost ~1 SU).

15. **GPU root-seed `beagleSetPartials` rc=-5 OUT_OF_RANGE, and the partial-buffer-count interaction.**
    After the bug 11–13 fixes the CPU gradient computes (job 170128746, see results below) but the GPU
    root-seed `beagleSetPartials(inst, preB_(root)=397, …)` returns **rc=-5**. Source: CUDA `setPartials`
    checks `bufferIndex >= kPartialsBufferCount` where `kPartialsBufferCount == partialsBufferCount`
    (NOT `tipCount+partialsBufferCount`). With the (memory-saving) `partialsBufferCount=302`, the global
    pre-order index 397 ≥ 302 → rejected. CPU's `setRootPrePartials` checks the wider global space, so CPU
    tolerated 302. **Fix:** set `partialsBufferCount = 2·nnodes+4 = 402` so every pre-order index (max 397)
    is `< partialsBufferCount`. Cost: 402 partials, **GPU pads AA 20→32 states** ⇒ 402·(nptn·32·NCAT·8) =
    **41 GB at NCAT=4 — exceeds the 32 GB V100, fits the 80 GB A100** ⇒ GPU gradient must run on **dgxa100**.
    (The doubled partial footprint of the pre-order pass is itself the headline VRAM finding: lnL fits V100,
    the *gradient* does not.) CPU at 20 states = 25.7 GB, fits a 90 GB node. Validation split: CPU FD on
    gpuvolta (job 170129094), GPU gradient on dgxa100 (job 170129095).

16. **BEAGLE 4.0.1's CUDA pre-order/edge-derivative path is BROKEN for 20-state (protein) models** — the
    final, decisive finding. After applying the exact BEAGLE `hmctest` GPU recipe (manual transpose of the
    pre-order matrix1 via `beagleTransposeTransitionMatrices`, transposed differential Qᵀ·r_c, forward
    sibling matrix, `setPartials` root-seed; **not** the AUTO flag — AUTO silently failed to transpose
    matrix1, jobs 170129095/618), the A100 gradient is STILL wrong (worst rel = 1.00, sign flips), while the
    CPU plugin computes the same algorithm correctly. **NCAT-isolation diagnostic (job 170130580):** a
    single-rate model (`g1`, NCAT=1) **also fails on GPU (rel 1.00) but PASSes on CPU (rel 1.07e-7)** ⇒ the
    bug is NOT category handling (so a category-split à la r10split cannot rescue it) — it is the **20-state
    CUDA pre-order/edge-derivative kernel itself**. `hmctest` only validates 4-state nucleotide gradients on
    GPU; the protein path is under-tested and incorrect in this build. The GPU gradient *infrastructure* is
    sound (fits A100-80GB at 50.3 GB, runs at ~100–240 ms = ~100–150× the CPU eval) — only the VALUES are wrong.

**Final harness state (`gpu_derisk.cpp`, 2026-06-06):** differential = generator matrix at `dQidx=nnodes+1`,
**Qᵀ·r_c on GPU / Q·r_c on CPU**; `updateTransitionMatrices` computes P only (no dP/dt); `derivativeMatrixIndices`
all → `dQidx`; `outDerivatives`/`outSumSqDerivatives` → NULL (fixes the CPU heap-overflow SEGFAULT);
GPU root-seed via `beagleSetPartials` with `partialsBufferCount=402`; **manual** transpose of each edge P into
`TP(c)` per eval (matrix1) — AUTO flag dropped; swept-eps FD validator (best of {1e-2,1e-3,1e-4}, PASS<5e-3).
Added a `g1` (NCAT=1) diagnostic model.

## ✅/⚠ Gradient results (2026-06-06) — algorithm validated on CPU; BEAGLE-4.0.1 CUDA gradient unusable for protein

| model | device | lnL | lnL eval | grad eval | FD worst rel | gradient |
|---|---|---|---|---|---|---|
| g1 (NCAT=1) | CPU-SSE-Double | −7974816.4323 | 1383 ms | 9886 ms | **1.07e−07** | ✅ **PASS** |
| g1 (NCAT=1) | A100 CUDA-Double | −7974816.4323 | **9.0 ms** | **100.6 ms** | 1.00 | ❌ FAIL (values wrong) |
| g4 (NCAT=4) | CPU-SSE-Double | −7541976.9391 | 1414 ms | 35839 ms | **2.76e−03** | ✅ **PASS** |
| g4 (NCAT=4) | A100 CUDA-Double | −7541976.9391 | **34.5 ms** | **237 ms** | 1.00 | ❌ FAIL (values wrong) |

(jobs 170129094 CPU-gpuvolta, 170129955/170130580 A100-dgxa100. lnL is bit-parity CPU≡GPU on both models;
g4 also 1e-8 of the IQ-TREE reference. GPU VRAM 50.3 GB — confirms the gradient needs A100-80GB, not V100.)

**Bottom line:** (1) The Ji-et-al O(N) pre-order branch-length gradient — the exact computation Mode-L
overflowed at 10⁵⁴ — **is mathematically correct and FD-validated** (CPU, rel 1e-7 to 3e-3). (2) BEAGLE's
**GPU likelihood** is correct + fast (29–153×). (3) BEAGLE **4.0.1's GPU gradient is unusable for protein
models** (wrong values for all NCAT, even with the reference recipe) → the production GPU gradient must come
from a **newer/fixed libhmsbeagle** or a **custom CUDA kernel in the IQ-TREE port**, NOT off-the-shelf 4.0.1.
This does not threaten the GPU-ModelFinder bet: the lnL (dominant cost) is proven on GPU, and the gradient
*algorithm* is proven; only BEAGLE's protein-CUDA-gradient implementation is the gap.

## Pending (post-PASS)
- **Gradient — ✅ ALGORITHM VALIDATED (CPU), ⚠ BEAGLE-4.0.1 GPU gradient broken for protein (bug 16).**
  CPU FD PASS for g4 (2.8e-3) and g1 (1.1e-7). GPU gradient values wrong (20-state CUDA kernel bug, NCAT-
  independent). Next-phase options: (a) build libhmsbeagle from a revision that fixes the 20-state CUDA
  pre-order path; (b) write the gradient CUDA kernel in the IQ-TREE GPU port directly. GPU gradient memory =
  402 partials = 50.3 GB ⇒ A100-80GB (lnL alone fits V100).
- **Heavier models — ✅ DONE** (job 169680691): g4/fig4/r8 bit-parity 43–66×; r10 via 5+5 split bit-exact 57×.
- **Tighten parity (optional polish)** — recompute the 4 gamma category rates from α=0.9963 at full
  precision to drive LG+G4 |Δ| from 1e-8 toward machine epsilon (cosmetic; gate already passes).
- **vs IQ-TREE kernel (optional)** — a single-lnL timing from IQ-TREE itself for a true apples-to-apples
  speedup (vs AVX/threaded), rather than vs BEAGLE-CPU-SSE.
- Then **G.1**: CUDA-Graph capture of the fixed-topology scoring traversal (CPU-free hot loop).
