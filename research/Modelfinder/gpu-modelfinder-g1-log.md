# Phase G.1+ Execution Log — Custom in-tree CUDA kernel implementation

Companion to [gpu-modelfinder-design.md](gpu-modelfinder-design.md) **PART II** (the implementation
plan) and [gpu-modelfinder-g0-log.md](gpu-modelfinder-g0-log.md) (the G.0 BEAGLE de-risk). This file is
the running record of the **in-tree CUDA kernel build** in the GPU dev clone
`/scratch/rc29/as1708/iqtree3-gpu` (branch `gpu-kernel`; setonix-iq branch `gpu-modelfinder`).

Each phase is independently testable against a concrete number (design §II.8). FP64 unscaled, native-20,
compact tips, NCAT≤10 one-pass; **non-negotiable gate: FD-validate every GPU gradient** (g4<3e-3, g1<1e-6).

---

## Build environment (Gadi, in-PBS-job — login lacks nvcc/icpx)

- **Toolchain (design §II.9 all-GCC host path):** `cuda/12.5.1` + `gcc/12.2.0` + `cmake/3.24.2`
  + `eigen/3.3.7` + `boost/1.84.0`. Host C++ = gcc (`CC=gcc CXX=g++`); CUDA host
  `-DCMAKE_CUDA_HOST_COMPILER=g++` (system g++ 8.5 is too old for CUDA 12). No intel, no MPI.
- **cmake flags:** `-DCMAKE_BUILD_TYPE=Release -DIQTREE_GPU=ON -DCMAKE_CUDA_COMPILER=$(which nvcc)
  -DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3 -DBOOST_ROOT=/apps/boost/1.84.0
  -DBoost_NO_SYSTEM_PATHS=ON`.
- **Queue:** `gpuvolta` (V100-SXM2-32GB) — **Cascade Lake host ⇒ NEVER `-march=sapphirerapids`**
  (would SIGILL, see ref_spr_binary_login_sigill memory); generic build, IQ-TREE runtime ISA dispatch
  selects AVX/AVX512 safely at run time. dgxa100 (A100) is AMD EPYC → a cascadelake binary won't run
  there; rebuild per target, or build generic.
- **Build script:** `gadi-ci/gpu-modelfinder/run_g1_0_build_gpuvolta.sh` (builds ON + OFF, runs 5 tests).

### Clone-setup gotcha (cost the first job) — submodules not populated
`git clone` does **NOT** populate submodules. The clone's `cmaple` and `lsd2` are git submodules
(`git submodule status` shows `-<sha>` = uninitialised) and were **empty** → first configure (job
170176345, 31 s) failed: `set_target_properties Can not find target to add properties to: maple` /
`lsd2` (CMakeLists.txt ~1000/1008). **Fix:** rsync the populated dirs from the FCA source tree
(`rsync -a --exclude=.git /scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/{cmaple,lsd2}/ ./{cmaple,lsd2}/`).
This also **preserves configuration parity with the FCA binary** (cmaple/lsd2 compiled in but unused by
MF) — important for the later G.2/G.3 wall-clock comparisons. The copied `cmaple/CMakeLists.txt` already
carries the Gadi offline patches (IPO off, `unittest`/`FetchContent(googletest)` commented out), so the
build-script seds are guarded no-ops. (terrace/terraphast/booster are regular dirs, already populated.)

---

## ✅ G.1.0 — Build scaffold (job 170176864, V100, 2026-06-07) — PASS (8/8)

**Deliverable (design §II.8):** `option(IQTREE_GPU)` (default OFF) + gated `enable_language(CUDA)` +
`iqtree_gpu` static `.cu` lib linked into `iqtree3` + `#cmakedefine IQTREE_GPU` + `--gpu`/`-gpu` flag +
a hello-world `.cu` launched from a diagnostic hook. Pure plumbing, **no numerics**.

### Change set (8 edits, 2 new files — all OFF-inert)
| File | Change | Anchor |
|---|---|---|
| `CMakeLists.txt` | `option(IQTREE_GPU "..." OFF)` + ON/OFF message | after `option(BUILD_LIB ...)` (~297) |
| `CMakeLists.txt` | `-static`→`-rdynamic` guard when `IQTREE_GPU` (cudart can't fully static-link) | UNIX else() branch (~333) |
| `CMakeLists.txt` | gated CUDA block: `CMP0104 NEW` + `CMAKE_CUDA_ARCHITECTURES "70;80;90"` **before** `enable_language(CUDA)` + `find_package(CUDAToolkit)` + `add_library(iqtree_gpu STATIC tree/gpu/gpu_diag.cu)` + `CUDA::cudart` PUBLIC | before `add_subdirectory(main)` (~913) |
| `CMakeLists.txt` | gated `target_link_libraries(iqtree3 iqtree_gpu)` | after master link (1023) |
| `iqtree_config.h.in` | `#cmakedefine IQTREE_GPU` | after `#cmakedefine Backtrace_FOUND` |
| `utils/tools.h` | `bool gpu;` field (public) | after `string intree_str;` (2914) |
| `utils/tools.cpp` | `gpu = false;` in `Params::setDefault()`; `--gpu`/`-gpu` parse block | setDefault end (7557); after `-mrbayes` (5456) |
| `main/main.cpp` | guarded `#include "tree/gpu/gpu_iqtree.h"`; `if (params.gpu) iqtree_gpu_diag()` else "built without GPU" | after `#include <iqtree_config.h>` (25); after `parseArg` (2279) |
| **new** `tree/gpu/gpu_iqtree.h` | C-linkage decl `void iqtree_gpu_diag();` | — |
| **new** `tree/gpu/gpu_diag.cu` | device-prop print + hello-world kernel + marker readback + `cudaGetLastError` | — |

### Findings / decisions during implementation
1. **`cmake_minimum_required(VERSION 3.5)` < 3.18 needed for `CUDA_ARCHITECTURES`.** Conditionally
   bumping line 76 is **fragile** (the `option()` is evaluated later). **Chosen fix:** inside the gated
   block, `if(POLICY CMP0104) cmake_policy(SET CMP0104 NEW)` + `set(CMAKE_CUDA_ARCHITECTURES ...)` **before**
   `enable_language(CUDA)`. cmake/3.24 honours it; line 76 untouched. Worked first try.
2. **Single macro mechanism.** Used the `#cmakedefine IQTREE_GPU` config-header route (main.cpp already
   `#include <iqtree_config.h>` at line 25). Did **not** also `add_definitions(-DIQTREE_GPU)` — that would
   redefine the macro. (The codebase's `_IQTREE_MPI`/`IQTREE_TERRAPHAST` use the add_definitions route; the
   config-header route is equally valid and self-documenting.)
3. **Explicit `CUDA::cudart` link** (`find_package(CUDAToolkit)` + `target_link_libraries(iqtree_gpu PUBLIC
   CUDA::cudart)`) rather than relying on CMake's implicit propagation across the static-lib boundary —
   removes a known foot-gun. ldd confirms `iqtree3` (ON) links `libcudart.so.12`; OFF links none.
4. **`-static` is not the default on Gadi/UNIX** — `-rdynamic` is used unless `IQTREE_FLAGS MATCHES "static"`.
   So the default GPU build was never at risk; the guard only matters for an explicit `static`+`IQTREE_GPU`
   combo.
5. **Seam line-number correction:** `computeLikelihoodDervPointer` lives at **`tree/phylotree.h:1346`**
   (typedef 1345; sibling `computeLikelihoodDervMixlenPointer` at 1349), **NOT** inside the 902-1004 block
   as the design/GPU_PORT.md stated. The other three pointers are confirmed at 904 / 951 / 1004.
   `setLikelihoodKernelAVX()` assigns all four at `phylotreeavx.cpp:50-145`; virtual wrappers at
   `phylotreesse.cpp:212-236`. (Corrected in design.md §II.2 and GPU_PORT.md.)
6. **Benign pre-existing warning:** `-Wpsabi "AVX vector return without AVX enabled changes the ABI"` in
   `cmaple/.../utils/matrix.h:54` (`simde__m256d mul4`). Appears **identically in ON and OFF** builds (it is
   cmaple's own simde code compiled without a global `-mavx`); not introduced by the GPU changes, harmless
   (inlined within the TU). Not chased.

### Validation (5 tests, all PASS) — job 170176864, wall 5:50, SU 3.50
| Test | What | Result |
|---|---|---|
| T2 ON configure | finds nvcc + CUDAToolkit 12.5.82, arch 70;80;90 | ✅ exit 0 |
| T2 ON build+link | `libiqtree_gpu.a` built; `iqtree3` links `libcudart.so.12` | ✅ exit 0 |
| T1 OFF configure+build | no CUDA enabled; binary links **no** cudart | ✅ exit 0 |
| T3 ON, no `--gpu` | normal CPU ModelFinder on example.phy | ✅ lnL −21152.524 |
| T4 ON, `--gpu` | GPU diag: V100, 31.7 GB / 80 SMs / ~898 GB/s, `marker=0xC0DE`, `cudaGetLastError=cudaSuccess`, "diagnostic PASSED", **then completed CPU run** | ✅ lnL −21152.524 |
| T5 OFF, `--gpu` | prints "built WITHOUT GPU support" + completes CPU run | ✅ lnL −21152.524 |
| **Behavioural identity** | OFF lnL == ON-no-gpu lnL | ✅ −21152.524 == −21152.524 |
| **GPU-hook non-interference** | ON-no-gpu lnL == ON `--gpu` lnL (the diag does not perturb the CPU likelihood) | ✅ −21152.524 == −21152.524 |

**Verdict: G.1.0 PASSES.** The in-tree CUDA toolchain is proven end-to-end — nvcc compiles a `.cu`, it
links into `iqtree3`, a kernel runs on the V100 with a clean error state, the `--gpu` flag parses, the
`#ifdef IQTREE_GPU` guard cleanly degrades the CPU-only build, and the GPU hook leaves the CPU path
byte-faithful. This is the foundation for G.1.1 (postorder lnL kernel K1).

### Reproduce
`qsub gadi-ci/gpu-modelfinder/run_g1_0_build_gpuvolta.sh` → `g1-0-gpu-scaffold.o<jobid>` in the clone root.
Binaries: `build-gpu-on/iqtree3` (CUDA), `build-gpu-off/iqtree3` (CPU). Runs land in `g1_0_runs/`.

---

## ✅ G.1.1 — Postorder lnL kernel K1 (job 170188276, V100, 2026-06-08) — PASS (4/4 models)

**Deliverable:** a **custom CUDA eigen-space postorder partial-likelihood kernel** (NOT BEAGLE), validated
standalone against the G.0 oracle. Harness `gadi-ci/gpu-modelfinder/gpu_k1_lnl.cu` reuses the G.0
BEAGLE-free CPU scaffolding (LG matrix, reversible eigendecomp → U/Λ/U⁻¹ in IQ-TREE convention, Newick
parser, compact tip states, gamma/FreeRate rates) and swaps BEAGLE's probability-space compute for the
eigen-space kernel. Built standalone (`nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo`, **no fast-math**) — no
IQ-TREE rebuild, ~seconds to iterate.

**The exact algorithm implemented** (= IQ-TREE `computePartialLikelihoodSIMD`, derived in the G.1.1
research workflow): partials stored in **eigen space**.
- leaf state `s`: `L[c][i] = Uinv[i][s]` (column `s` of U⁻¹; ambiguous → row-sum over `s`).
- internal node, per cat `c`, per child `k`: `pk[x] = Σᵢ echild_k[c][x][i]·L_child_k[c][i]`;
  `prod[x] = Πₖ pk[x]`; store `L_node[c][r] = Σₓ Uinv[r][x]·prod[x]`.
- root: `lh_ptn = Σ_c prop_c·Σₓ freq[x]·prod_root_c[x]`; `lnL = Σ_ptn log|lh_ptn|` (all sites, weight 1).
- `echild_k[c][x][i] = U[x][i]·exp(eval[i]·rate_c·len_k)` precomputed on host (fixed tree).
The eigen-space invariant `L_eigen = U⁻¹·L_prob` makes the parent's `echild·L_child = U·exp(Λ·rate·len)·U⁻¹·L_prob = P(t)·L_prob` — i.e. exactly Felsenstein.
- **Layout** `partial[slot][c·NS+state][ptn]` (pattern innermost/contiguous) → coalesced loads with one
  thread per pattern. One kernel launch per internal node in postorder (same default stream ⇒ ordered;
  one `cudaDeviceSynchronize` before the host-side log-sum). FP64, **unscaled**, **native-20** (no 20→32 pad).

**Results — every model matches the G.0 oracle to rel ~1e-12 (FP64 summation-noise floor):**
| model | NCAT | lnL (K1 custom) | G.0 oracle | rel err | eval (V100) | VRAM |
|---|---|---|---|---|---|---|
| g4  | 4  | −7541976.9391 | −7541976.9391 | 5.78e-12 | **37.8 ms** | 6.16 GB |
| r8  | 8  | −7556251.9185 | −7556251.9185 | 3.82e-12 | 74.0 ms | 12.0 GB |
| **r10** | **10** | **−7554280.5776** | −7554280.5776 | 6.09e-12 | 93.0 ms | 14.9 GB |
| g1  | 1  | −7974816.4323 | −7974816.4323 | 5.23e-12 | 10.4 ms | 1.78 GB |

**Headline findings:**
1. **Custom eigen-space kernel is correct** — g4 rel 5.8e-12 (|Δ|=4.4e-5 on −7.5e6 = pure FP64
   reduction-order noise; "bit-parity" in the design = parity to the FP64 noise floor, NOT bit-identical,
   since GPU reduction order ≠ CPU SIMD FMA order — an honest restatement of the gate).
2. **+R10 in ONE pass (NCAT=10) = −7554280.5776, rel 6e-12** — exactly the r10split number, with NO
   category-split and NO `kMatrixBlockSize≤8` cap. The +R10 long-pole that broke every CPU dispatch arch
   (Amdahl) and that stock BEAGLE-CUDA could not run, now executes in a single custom kernel. **The §II.6
   "NCAT≤8 cap → gone in a custom kernel" claim is now demonstrated.**
3. **K1 already beats BEAGLE unoptimized** — g4 37.8 ms vs BEAGLE-CUDA 45 ms on the same V100, and
   **native-20 keeps VRAM well under 32 GB** (r10 14.9 GB vs the doc's prediction). No shared-mem `echild`
   staging, no batched independent-node launches, no CUDA graph yet — all headroom for G.1.3 / a perf pass.
4. cost: SU 0.26, wall 26 s (build + 4 model runs).

**Deferred (not blocking the correctness gate):** formal Nsight HBM%/coalescing/occupancy profiling — the
timing already beating BEAGLE confirms the memory-bandwidth thesis, but the >80 % HBM target is unmeasured.
Kernel perf optimizations (shared-mem `echild`, batch same-depth nodes into one launch to cut the 98
launches, reduce `prod[NS]` register pressure) tracked for a perf pass / G.1.3.

**Reproduce:** `qsub gadi-ci/gpu-modelfinder/run_g1_1_k1_v100.sh` → `g1-1-k1-v100.o<jobid>`.

---

## ✅ G.1.2 — Single-edge derivative kernel K2 (job 170188743, V100, 2026-06-08) — PASS (4/4 models)

**Deliverable:** a **custom CUDA single-edge derivative kernel** (the `computeLikelihoodDervSIMD` analog,
NOT BEAGLE) returning lnL, df=∂lnL/∂t and ddf=∂²lnL/∂t² for one branch, FD-validated against a CPU
finite-difference oracle and used to **independently rediscover** IQ-TREE's optimized edge length by
bisection-on-df. Harness `gadi-ci/gpu-modelfinder/gpu_k2_derv.cu` reuses all of K1's BEAGLE-free
scaffolding + the K1 postorder kernel; built standalone (`nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo`,
**no fast-math**), ~seconds to iterate, no IQ-TREE rebuild.

**The exact algorithm** (= IQ-TREE `computeLikelihoodDervSIMD`, eigen space):
- Pick a central edge `(root, c0)` where `c0` = the root's first **internal** child. Run K1 to get the
  eigen-space partials for **both endpoints**: `node_eig` = c0's partial; `dad_eig` = the partial of the
  rest of the tree as seen from c0 (a K1 `k1_node` over the root's children **except** c0).
- **`theta = node_eig ⊙ dad_eig`** (elementwise over the `NCAT·NS` eigen components per pattern) is
  **t-independent** → computed once on the GPU (`k2_theta` kernel) and reused for every t. This is the
  derivative-specific cache; it is what makes each NR step cheap.
- Per category `c`, per eigen-index `i`: `val0 = exp(eval[i]·rate_c·t)·prop_c`, `val1 = (rate_c·eval[i])·val0`,
  `val2 = (rate_c·eval[i])·val1`. Uploaded to `__constant__` (`setVal(t)`) each NR step (tiny, `NCAT·NS` doubles).
- **`k2_derv` kernel** (one thread/pattern): `lh = Σ val0·theta`, `d1 = Σ val1·theta`, `d2 = Σ val2·theta`;
  then `patlh = log|lh|`, `pdf = d1/lh`, `pddf = d2/lh − (d1/lh)²`. Host reduces (Kahan for lnL):
  `lnL = Σ patlh`, `df = Σ pdf`, `ddf = Σ pddf`.
- The eigen-space **freq-fold identity** `freq_a·U[a][x] = U⁻¹[x][a]` means the derivative formula needs no
  explicit `freq` — proven by the lnL(t0) cross-check below matching the K1/G.0 oracle to ~1e-12.

**Results — job 170188743, wall 22 s, exit 0:**
| model | lnL(t0) cross-check vs K1/G.0 | df rel (FD) | ddf rel (FD) | bisection t* | \|t*−t0\| | ddf(t*)<0 | derv eval |
|---|---|---|---|---|---|---|---|
| g4  | −7541976.9391 (rel 5.78e-12) | 2.50e-9 | 3.29e-6 | 0.014217 | 0.00000 | yes | **1.21 ms** |
| g1  | −7974816.4323 (rel 5.23e-12) | 2.38e-9 | 4.99e-6 | 0.014160 | 0.00006 | yes | 1.10 ms |
| r8  | −7556251.9185 (rel 3.82e-12) | 2.19e-9 | 3.51e-6 | 0.014009 | 0.00021 | yes | 1.24 ms |
| r10 | −7554280.5776 (rel 6.09e-12) | 2.30e-9 | 3.73e-6 | 0.014195 | 0.00002 | yes | 1.26 ms |

**Three independent gates, all PASS:**
1. **lnL(t0) cross-check** — evaluating the derivative kernel's `lh` at the tree's branch length and
   summing `log|lh|` reproduces the K1/G.0 oracle to rel ~1e-12 for all 4 models. Confirms the derivative
   kernel's likelihood path (and the freq-fold identity) is exact, independent of the partials path.
2. **FD-validation (the non-negotiable gate)** — central finite differences of lnL(t) (swept ε) vs the
   kernel's analytic df/ddf: **df rel ~2–2.5e-9** (meeting even g1's tight <1e-6 gate by 3 orders), **ddf
   rel ~3–5e-6**. The analytic gradient is correct, not just plausible.
3. **Bisection-on-df optimum-find** — bracket `df(1e-6)≈+4.4e6 → df(5.0)≈−1.3e4` (a single interior sign
   change ⇒ unique max), 60 bisection iters → `t*` within rounding of IQ-TREE's optimized `t0` for all 4
   models (|t*−t0| ≤ 2.1e-4), with `ddf(t*)<0` (true concave max) and `lnL(t*) ≥ lnL(t0)`. The GPU
   derivatives genuinely **locate** the optimum, end-to-end.

**Headline findings:**
1. **Derivative kernel is correct and the NR step is cheap** — one `evalAt(t)` (val upload + `k2_derv` +
   D2H reduce) is **~1.1–1.3 ms** on g4–r10, vs the ~38–93 ms one-shot K1 lnL. The expensive partials are
   computed once; each NR iteration is just the theta-dot. This is exactly the cost structure branch
   re-optimization (75–85 % of per-model wall) needs.
2. **Bisection chosen over raw Newton.** The prior run (170188563) drove **naive** Newton `t -= df/ddf`
   from `t=0.5`, which **diverged** — at that start the curve is **convex** (`ddf=+3.16e4>0`, the far side
   of the optimum), so the raw step runs away. This is an optimizer-driver artifact, **not** a derivative
   error (the FD gate PASSED in that run too). Replaced with **bisection-on-df** (robust from any bracket
   with a df sign change) for a clean convergence record; real integration will use IQ-TREE's safeguarded
   `minimizeNewton` (bracketing + bisection fallback), which already handles this.
3. **Central-edge choice.** Validated on `(root, first-internal-child)` so both endpoints are internal
   (the general case: two `prod[NS]` vectors folded by theta). Leaf-endpoint edges are a strict
   simplification (one side is a tip column) and are covered by the same kernel.
4. cost: SU ~0.22, wall 22 s (build + 4 models × {cross-check, FD-sweep, bisection, 20-rep timing}).

**Deferred (not blocking the gate):** wiring K2 into IQ-TREE's real `minimizeNewton` (G.2 integration);
on-device `echild`/`val` rebuild + theta recompute when a *neighbouring* branch changes during a full
tree NR sweep (G.1.3); Nsight profiling of `k2_derv` (it is a trivially bandwidth-bound triple-dot, so
HBM% should be near the K1 figure).

**Reproduce:** `qsub gadi-ci/gpu-modelfinder/run_g1_2_k2_v100.sh` → `g1-2-k2-v100.o<jobid>`.

---

## ✅ G.1.3 — CUDA-graph capture + on-device echild rebuild K3 (job 170189528, V100, 2026-06-08) — PASS (correctness/capability; speed reframed by evidence)

**Deliverable:** take the GPU off the host's critical path for the postorder lnL sweep — (1) a custom
**on-device `build_echild` kernel** that rebuilds `echild = U·exp(eval·rate·t)` from a **device-resident**
branch-length buffer `d_brlen` (K1 built echild once on the host), and (2) a **CUDA graph** capturing the
whole pipeline `[brlen H2D → build_echild → blocksum memset → 98× k1_node postorder → reduce_patlh →
reduce_final → lnL D2H]`, instantiated once and **replayed per branch-length change with a single
`cudaGraphLaunch`** (no re-capture, no `SetParams` — only `d_brlen` contents change, read from global
memory). Harness `gadi-ci/gpu-modelfinder/gpu_k3_graph.cu`; standalone nvcc, no IQ-TREE rebuild.

**Design adversarially audited before implementation** (3-auditor workflow: CUDA-graph semantics / FP64
numerics / gate-intent → synthesis, `go=true`). Load-bearing fixes folded in (all confirmed by the run):
- **`build_echild` FP-grouping** must mirror the host *exactly* — `len = brlen·catRate` first (single
  rounding), then `exp(eval·len)`, then `U·ex`. The bare 3-factor product `exp(eval·brlen·catRate)` groups
  as `(eval·brlen)·rate` and bit-differs in FP64 (the auditor measured 12/125 representative combos differ,
  max rel 1.4e-14). With the fix, device echild matches host to ≤1 ULP (V0 below).
- **No synchronizing CUDA API inside the capture region** — only kernel launches + `cudaMemcpyAsync`/
  `cudaMemsetAsync` on the capture stream; the four `__constant__` model arrays are set **once** before
  capture, never refreshed inside it. Capture on a `cudaStreamNonBlocking` stream in
  **`cudaStreamCaptureModeGlobal`** (strictest — surfaces any stray sync call), not the legacy default stream.
- **Replay race fix (Pattern A)** — the `d_brlen` H2D is the **first captured node**, copying from a
  **pinned** host staging buffer; each replay rewrites the pinned buffer then launches (strictly ordered
  before `build_echild`). Pinned-buffer reuse is safe (prior `cudaStreamSynchronize` before overwrite).
- **Deterministic reduction** — block-local shared-mem **pairwise** halving → `d_blocksum[nblocks]` →
  single-block pairwise final; **not** atomicAdd (non-deterministic) and **not** a long sequential
  accumulator (worst-case rel ~2e-11 would fail the oracle gate). `d_blocksum` zeroed by a *captured* memset.
- Bracket bounded to **[1e-6, 10]** + `isfinite(lnL)` guard (UNSCALED FP64: `exp(−745)=0 → log=−inf` trap).

**Results — job 170189528, exit 0, wall 6:53, SU 4.13, GPU util 97 %. All gates PASS, 4/4 models:**

| gate | g4 | r8 | r10 | g1 |
|---|---|---|---|---|
| **V0** device `build_echild` vs host echild (max rel) | 3.4e-16 | 3.9e-16 | 4.1e-16 | 3.2e-16 |
| **V1** graph lnL = G.0 oracle (rel) | −7541976.9391 (5.8e-12) | −7556251.9185 (3.8e-12) | −7554280.5776 (6.1e-12) | −7974816.4323 (5.2e-12) |
| **V1** graph (device-reduce) vs naive-host (Kahan) \|Δ\| | **0.0** | **0.0** | 9.3e-10 | **0.0** |
| **V2** patlh bit-identity graph-vs-naive (pre-reduction) | 0/100000 | 0/100000 | 0/100000 | 0/100000 |
| **V3** deterministic replay (same brlen → bit-identical) | yes | yes | yes | yes |
| **V4** one-edge perturbation, graph vs naive \|ΔlnL\| | 0.0 | 9.3e-10 | 9.3e-10 | 9.3e-10 |
| **V5** single-branch opt, graph t* vs naive t* (\|dt\|) | 0.0 | 0.0 | 0.0 | 0.0 |

**V6 — multi-branch convergence (g4; the literal "same converged branch lengths" gate):** perturb **all
197 branches ×1.3**, run one optimizeAllBranches-shaped Gauss-Seidel golden-section pass (≈4300 full-tree
**graph replays**) and the identical pass with naive (non-graph) host evals. Result: graph-final ≡
naive-final lnL = **−7546671.8370**, **converged-vector max\|dt\| = 0.0**, **\|ΔlnL\| = 0.0** — the
graph-driven sweep converges **bit-identically** to the naive sweep (far stronger than the Δ<1e-4 gate).
(One pass from a ×1.3 perturbation does not return to the global optimum — hence −7546671 < oracle
−7541977 — but graph≡naive convergence is exact, which is what the gate asks.)

**Findings:**
1. **Correctness/capability gate: decisively PASS.** Device-resident `build_echild` is ULP-clean (V0); the
   captured graph replays the full sweep with lnL = the proven K1/G.0 oracle (V1) and **bit-identical
   per-pattern `patlh`** to the naive device path (V2); replay is **deterministic** (V3); branch-length
   changes flow through correctly (V4) and drive **identical optimisation convergence**, single- and
   multi-branch (V5/V6). The device tree-reduction even matched the host Kahan sum to 0 (g4/r8/g1) or 1 ULP
   (r10) — the feared reduction-order noise is a non-issue for this data.
2. **⚠ Speed reframed by evidence — wall-clock is PARITY, not "materially faster".** The timing curve is
   **1.00–1.01× graph-vs-naive at *every* pattern count** (g4 8.37 ms @1k · 8.31 ms @10k · 37.73 ms @100k;
   r8 73.9; r10 93.0; g1 10.4). The predicted small-nptn launch-bound win **did not materialise**: even at
   nptn=1000 the sweep is 8.4 ms because it is bound by **GPU-side per-kernel scheduling latency** of the 98
   sequential dependent `k1_node` launches (~85 µs each), which a CUDA graph **does not remove** — the graph
   only collapses *host submission*, and on the default stream that submission was already overlapped behind
   GPU execution. So 104 host API calls → 1, but wall-clock unchanged.
3. **⇒ The graph's real, demonstrated value is STRUCTURAL, not a speedup:** (a) **device-resident `d_brlen`**
   — the GPU is no longer re-fed from the host each iteration (the actual G.1.3 deliverable, and the
   integration unblock the design named), and (b) **104 → 1 host CUDA API calls per eval**, freeing the CPU
   control loop. A *wall-clock* win for this kernel chain requires **fewer/bigger kernels** — fuse the
   postorder into same-depth batched launches (the perf opt already flagged as K1 headroom: "batch
   same-depth nodes to cut the 98 launches"), which reduces the per-kernel scheduling latency the graph
   cannot touch. That is a perf pass, tracked below — *not* a correctness blocker.
4. VRAM (native-20): g4 6.16 GB, r8 12.0 GB, r10 14.93 GB, g1 1.78 GB — all well under the V100's 31.7 GB.
5. **Fallback path** (`useGraph=false` → naive host `evalOnce` + Kahan) is wired and error-checked, but was
   not exercised (capture+instantiate succeeded on every model). Nsight HBM%/gap profiling still deferred.

**Honest accuracy statement:** device echild bit-matches host to ≤1 ULP (V0); per-pattern `patlh` is
bit-identical graph-vs-naive (V2, both device `build_echild`); the final lnL **scalar** is **not** claimed
bit-identical to the host Kahan value in general — the in-graph device tree-reduction differs only by
summation order (here it happened to match to 0–1 ULP) and agrees with the G.0 oracle at rel ~1e-12.

**Reproduce:** `qsub gadi-ci/gpu-modelfinder/run_g1_3_k3_v100.sh` → `g1-3-k3-v100.o<jobid>`.

### Perf pass — same-depth kernel fusion K4 (job 170194367, V100, 2026-06-08) — correct, but characterises a different bottleneck

**Goal:** the wall-clock win the CUDA graph could *not* deliver — cut the GPU-side per-kernel scheduling
latency by **fusing** the 98 per-node `k1_node` launches. Internal nodes grouped by **tree height**
(longest path to a leaf) are mutually independent (every node at height *h* has all children at height
*< h*), so a whole height level runs in **one** `k4_level` launch (2D grid: `blockIdx.y` = node-in-level,
`blockIdx.x` = pattern). Harness `gadi-ci/gpu-modelfinder/gpu_k4_fused.cu`; adversarially reviewed (the
height-leveling proven a valid parallel postorder schedule). Built standalone; also captured as a
fused-graph (the K3 machinery with `k4_level` in place of the 98 launches).

**Correctness — bit-identical, all 4 models (fusion changes dispatch, not numerics):** fused lnL ≡
per-node lnL (|ΔlnL| = 0), per-pattern `patlh` bit-identical (0/100000), oracle rel ~1e-12, fused-graph
deterministic. ✓

**The decisive finding — real ML trees are deep ladders, not balanced.** The AA-100K
`iqtree_inner.treefile` has **height 42** for 100 taxa: 97 non-root internal nodes spread over 41 levels,
but only the *shallow* part batches — `L1=22, L2=12, L3=8, L4=8, L5=6, L6=3, L7=3, L8=2` (64 nodes → 8
launches), then **`L9…L41` are all single-node levels** (a 33-deep serial caterpillar tail). So fusion
collapses 98 → **42** launches (2.3×), nowhere near the ~10× a balanced tree would give — and the deep
serial tail is *inherently* unparallelisable by any same-depth strategy.

**Timing (min of 50 reps):**
| nptn | per-node (98 launches) | fused (42) | fused-graph (1) | note |
|---|---|---|---|---|
| 1000 (g4) | 8.49 ms | **4.74 ms (1.79×)** | 4.73 ms | launch-bound → fusion wins |
| 10000 (g4) | 8.35 ms | **6.25 ms (1.34×)** | 6.24 ms | partially launch-bound |
| 100000 (g4) | 37.79 ms | 36.02 ms (1.05×) | 36.01 ms | compute-bound → ~wash |
| 100000 (g1) | 10.41 ms | 9.13 ms (1.14×) | 9.12 ms | compute-bound |
| 100000 (r8) | 73.93 ms | 76.94 ms (**0.96×**) | 76.94 ms | **fusion slightly SLOWER** |
| 100000 (r10) | 92.89 ms | 99.07 ms (**0.94×**) | 99.03 ms | **fusion slightly SLOWER** |

**Conclusions:**
1. **Fusion does win in the launch-bound regime** — **1.79× at nptn=1000**, 1.34× at 10000 — confirming the
   98 → 42 launch collapse cuts the scheduling-latency floor where it dominates.
2. **At the *production* pattern count (100K+) it is a wash-to-slight-loss** (g4 1.05×, g1 1.14×, but **r8
   0.96× and r10 0.94× — fusion is *slower***). At 100K patterns each kernel already saturates the V100's
   80 SMs, so collapsing launches buys little; for high-NCAT (r8/r10) the larger register-heavy 2D-grid
   `k4_level` launches schedule slightly *worse* than the many smaller per-node grids.
3. **⇒ At production scale the bottleneck is the kernel's COMPUTE/bandwidth, not launch overhead.** This is
   the real, useful result of the perf pass: it relocates the speedup lever. A wall-clock win at 100K+
   patterns needs **intra-kernel** optimisation (coalescing, shared-mem `echild` staging, cutting the
   `prod[NS]` register pressure — K1's other deferred opts, ideally Nsight-guided), *not* launch batching.
   And K1 already **beats BEAGLE** at this compute-bound size (g4 37.8 ms vs 45 ms), so this is headroom,
   not a blocker.
4. **For G.2, the production sweep should be the per-node graph (K3)**, not the fused sweep — fusion does
   not help (and can hurt) at the compute-bound production pattern count on real deep trees. The fused
   machinery is validated and retained for the launch-bound / balanced-tree regimes, but it is not the
   default. (cost: SU ~0.4, wall 1:43.)

**Reproduce:** `qsub gadi-ci/gpu-modelfinder/run_g1_3_k4_fused_v100.sh` → `g1-3-k4-fused.o<jobid>`.

## Intra-kernel perf lever — profile + occupancy sweep (jobs 170195112, 170195272, V100, 2026-06-08)

The perf pass proved launch-level work (graphs, fusion) gives wall PARITY at the compute-bound 100K count;
the only lever that can beat the 221.6 s wall is the kernel's own compute/bandwidth. So before in-tree
integration, profile + optimise `k1_node` (the dominant per-model cost) in the standalone harness.

**Profile (ncu, job 170195112) — decisive, single diagnosis: LATENCY-bound at 25 % occupancy, capped by
register pressure.** Per representative `k1_node` launch (g4, 391 blocks × 256 threads): **128 registers/
thread → Block Limit (Registers) = 2 → Theoretical/Achieved occupancy 25 %/~24 %**. Compute (SM) ~35 %,
Memory ~48 %, DRAM ~16–40 % — **none saturated**: there simply aren't enough warps resident to hide memory
latency. L1/TEX hit 70–82 % (so the broadcast `echild` reads are already L1-served — **shared-mem echild
staging would NOT help**, ruling out that candidate). The roofline confirmed it's not FLOP-bound
(37.8 ms vs ~6.7 ms FP64 floor). ⇒ the lever is **register reduction → higher occupancy**.

**Occupancy sweep (`gpu_k5_occ.cu`, job 170195272) — clean NEGATIVE result: the simple lever backfires.**
A/B'd the identical K1 body under `__launch_bounds__(threads, minBlocks)` register caps; every config is
SLOWER than baseline, monotonically worse as the cap tightens, and `ptxas -v` shows why — the kernel's true
working set is ~128 regs, so capping forces heavy spilling:
| config | regs | spill (st/ld) | g4 sweep | r10 sweep |
|---|---|---|---|---|
| **base (natural)** | **128** | **0 / 0** | **37.78 ms** | **92.90 ms** |
| LB256/3 (37.5 % occ) | 80 | 552 / 504 B | 41.1 ms | 102.6 ms |
| LB256/4 (50 %) | 64 | 904 / 792 B | 55.4 ms | 138.0 ms |
| LB256/5 (62.5 %) | 48 | 5.1 / 7.8 KB | 148.8 ms | 374.7 ms |
| LB256/6 (75 %) | 40 | 8.6 / 14.2 KB | 305.8 ms | 779.7 ms |
(LB128/{4,6,8} mirror this; lnL bit-identical to the G.0 oracle for **all** configs — body unchanged.)

**Conclusion:** the 25 %-occupancy diagnosis is right but `__launch_bounds__` is the WRONG fix — the
`prod[NS]`=20-double working set + matmul accumulators + 9 child pointers make ~128 regs irreducible by
compiler cap; spilling costs far more than the occupancy gain. **Baseline K1 (128 regs, 0 spill, 25 % occ)
is optimal in this family, and already beats BEAGLE (37.8 ms vs 45 ms).** A real occupancy win would need an
*algorithmic* restructure — distribute the NS=20 states across cooperating threads (warp-shuffle the
matvec → fewer regs/thread), or precompute per-leaf-edge P-matrix lookups (`P[x][s]` replaces the 400-FMA
leaf matmul) — both high-effort with uncertain payoff. **Decision pending (user):** attempt a restructure,
or proceed to G.2 and judge the 221.6 s wall in situ (the per-model GPU budget — ~38 ms/sweep + ~1 ms K2
evals vs the CPU's ~1 s/model — suggests the wall may already be beatable without further kernel work).

## Then — G.2 in-tree integration (PLAN READY, deferred)

Full plan in **[gpu-modelfinder-g2-plan.md](gpu-modelfinder-g2-plan.md)** (seam mapped + verified, bridge
design, phased gates G.2.-1→G.2.2, risks). Headline: a gated `setLikelihoodKernelGPU()` overriding the four
`computeLikelihood*Pointer`s at the `setLikelihoodKernel` funnel; **device-mirror** ownership; the harness↔
in-tree bridge is pattern-weights (`ptn_freq`) + padding + π-fold — and **NOT** scaling (corrected:
AA-100K/100-taxa is NORM_LH/unscaled, `leafNum=100 < numseq_safe_scaling=2000`, so the harnesses already
match the production oracle). Uses the **per-node K3 graph** (not fusion). Wall target (G.2.2) depends on
the intra-kernel lever above; correctness (G.2.0/G.2.1) does not.

## ✅ G.2.0a — in-tree clean-room GPU lnL cross-check (job 170203514, V100, 2026-06-08) — PASS (bridge proven at machine epsilon)

First in-tree GPU code, executed inside the **real iqtree3 binary** (dev tree
`/scratch/rc29/as1708/iqtree3-gpu`, branch `gpu-kernel`). De-risks the harness↔in-tree **bridge** (eigen
convention, tip-ambiguity fold, `ptn_freq` pattern weights, π-fold, NORM_LH) with **zero coupling** to the
fn-pointer seam, the host partial buffers, or `TraversalInfo` — that wiring is G.2.0b. Pure additive,
read-only diagnostic hook.

**What it does.** A gated one-shot `PhyloTree::gpuLnLCrossCheckOnce(curScore)` (new TU
`tree/phylotreegpu.cpp`, whole body `#ifdef IQTREE_GPU`), called at the end of `computeLikelihood`
(phylotree.cpp:1310, guarded `if (params && params->gpu)`) on the first full evaluation of a `--gpu` run. It
rebuilds the validated K1 eigen-space postorder sweep **clean-room from the LIVE IQ-TREE objects** —
`model->getEigenvalues()/getEigenvectors()(=U)/getInverseEigenvectors()(=Uinv)`, `model->getStateFrequency`,
`site_rate->getNRate()/getRate(c)/getProp(c)`, topology + branch lengths walked from `PhyloNode/PhyloNeighbor`
(rooted clean-room at the internal node adjacent to IQ-TREE's leaf-root; lnL is reversible-invariant), tip
states + `aln->at(p).frequency` from the alignment — marshals flat int descriptor arrays into an `extern "C"`
launcher `gpu_lnl_crosscheck` (`tree/gpu/gpu_lnl_intree.cu`, the K1 kernels parameterised by `ns` with a
`ptn_freq`-weighted Kahan reduction `lnL += ptn_freq[p]·log|lh_ptn|`), and compares to IQ-TREE's own
`curScore`.

**Result (LG+G4 on AA-100K, the standing gate dataset, 100 taxa / 96017 patterns / NCAT=4 / native-20):**

```
[GPU-XCHECK] ns=20 nptn=96017 ncat=4 ntax=100 nNodes=198 nInternal=98 (root@internal node id=100)
[GPU-XCHECK] GPU lnL = -7541977.778778   CPU lnL = -7541977.778778   |d|=9.3132e-10   rel=1.235e-16   -> PASS (bridge OK)
```

- **rel = 1.235e-16** — machine epsilon, **ten orders of magnitude tighter than the 1e-12 gate** (`|d|`=9.3e-10
  on a −7.5M magnitude). The clean-room conventions match IQ-TREE **exactly, no transpose needed**: `U`
  row-major with `P=U·exp(Λt)·U⁻¹`, full-ambiguity tip fold (`st<ns ? st : ns`, ambiguous → Uinv row-sum),
  integer `ptn_freq` weighting, and π-folded factors consumed directly (no explicit `freq[x]` multiply).
- **NORM_LH correction holds in vivo:** the run used the unscaled path (leafNum=100 < `numseq_safe_scaling`
  =2000, num_states=20) — the harness math (which the mapping agent had nearly mis-scoped as SAFE_LH) is the
  production oracle. No SAFE_LH ldexp-256 port needed for the AA-100K gate.
- **CPU path byte-unchanged:** the `--gpu` run and a plain CPU run both report identical
  `-7541976.8530 (s.e. 15407.1763)` (optimised tree). The hook is genuinely additive/read-only; the
  `#ifdef IQTREE_GPU` + `params->gpu` gate means a CPU build/run never touches it. (The xcheck compares at the
  *first* `computeLikelihood`, pre-NR brlen = −7541977.78; both runs also agree on the final −7541976.85.)
- **Build:** incremental rebuild of `build-gpu-on` (configured at G.1.0), `make exit=0`, 11.36 MB binary; the
  CMake wiring (`gpu_lnl_intree.cu` → `iqtree_gpu` lib; `phylotreegpu.cpp` → `tree` lib) compiles and links
  clean. Job walltime 10m26s (incl. two full AA-100K runs). Script
  `gadi-ci/gpu-modelfinder/run_g2_0a_xcheck_v100.sh`.

**Significance.** The math bridge between IQ-TREE's live objects and the validated K1 kernel is exact. G.2.0b
(overriding the `computeLikelihoodBranchPointer` at the `setLikelihoodKernel` funnel so IQ-TREE's **own**
`computeLikelihood` routes through the GPU) is now a pure *plumbing* problem — the numbers are proven.

## Next — G.2.0b: wire the Branch pointer at the funnel (lnL-only, `-blfix`)

Seam re-verified in this tree: funnel `PhyloTree::setLikelihoodKernel` (phylotreesse.cpp:91) takes the
AVX_FMA path → `setLikelihoodKernelFMA()` → `return` at :173 (the GPU hook insertion point — re-applied on
every funnel re-invocation, so idempotent without a separate `initializeAllPartialLh` re-apply). IQ-TREE's
`computeLikelihood` (phylotree.cpp:1289) routes the whole-tree lnL through `computeLikelihoodBranch(current_it
…)`, so overriding **only the Branch pointer** captures it. `computeLikelihoodFromBuffer`
(phylotreesse.cpp:230) only fires under NR branch-opt → **`-blfix` (tools.cpp:3363, fixes brlen of the `-te`
tree) makes Branch the *sole* exercised pointer**, giving a coherent lnL-only test with no host-partial
dependency (Derv/FromBuffer never run; alpha is still tuned by derivative-free Brent calling Branch). Plan:
extract the clean-room sweep into a reusable `gpuComputeTreeLnLCleanRoom()` helper; add
`computeLikelihoodBranchGPU` (byte-matching the Branch typedef) that calls it (delegating to a **saved CPU
branch pointer** when the per-call gate fails); `setLikelihoodKernelGPU()` saves+overrides at the funnel; a
one-shot in-process self-check compares GPU(curScore) vs an independent CPU recompute via the saved pointer.
**Gate:** `--gpu -te TREE -m LG+G4 -blfix` lnL == CPU rel ≤ 1e-12, GPU pointer proven fired, CPU byte-unchanged.

## ✅ G.2.0b — Branch pointer wired at the funnel (job 170205301, V100, 2026-06-08) — PASS (5/5 gates; GPU≡CPU bit-identical)

IQ-TREE's **own** `computeLikelihood` now routes the whole-tree lnL through the GPU under `--gpu … -blfix`.
A gated non-virtual `setLikelihoodKernelGPU()` is called LAST in the `setLikelihoodKernel` funnel
(phylotreesse.cpp:~176, `#ifdef IQTREE_GPU`); it saves the ISA-set CPU Branch pointer and overrides
`computeLikelihoodBranchPointer` with a new `PhyloTree::computeLikelihoodBranchGPU` member (byte-matches
`ComputeLikelihoodBranchType`). The override calls the reusable `gpuComputeTreeLnLCleanRoom()` helper (the
G.2.0a sweep, extracted), **mirrors the per-pattern `log|lh_ptn|` into host `_pattern_lh[]`** and zeroes the
branch `lh_scale_factor` (NORM_LH no-scaling path), then returns the scalar lnL; on any unsupported regime /
CUDA error the helper returns NaN and the override delegates to the saved CPU pointer.

**Adversarial pre-verification (5-agent workflow over the dev tree) shaped the design — and caught a real
bug.** It confirmed that under `-te -m LG+G4 -blfix` branch-length Newton-Raphson is unreachable
(`fixed_branch_length==BRLEN_FIX` gates out `optimizeAllBranches` at modelfactory.cpp:1628; `-te` zeroes
`min_iterations` so no tree search), so `computeLikelihoodDerv`/`computeLikelihoodFromBuffer` **never fire**
and +G4 alpha is tuned by derivative-free Brent (rategamma.cpp:240) which only calls Branch — so a Branch-only
override is coherent. BUT the adversary found a **counterexample**: `computeLogLVariance()`
(phyloanalysis.cpp:3946) is **unconditional** (gated only by `!pll`) and re-reads host `_pattern_lh[]` via
`computePatternLikelihood` (phylotree.cpp:1515-1528) *after* the branch lnL returns — a scalar-only override
would leave `_pattern_lh` stale → wrong/NaN `(s.e. …)` on the reported "Log-likelihood of the tree" line.
**Fix = mirror `_pattern_lh` from the launcher's already-computed per-pattern values** (option 1) → the
existing CPU variance/report path yields the correct s.e. with **zero edits to phyloanalysis.cpp**. The gate
also rejects `-wsl/-wpl/-alrt/-abayes/-b/-bb/-asr/dating/pll` and non-`BRLEN_FIX` / supertree / non-reversible
/ mixture / site-specific (→ CPU fallback) so the lnL-only override is safe by construction.

**Result — `--gpu -te TREE -m LG+G4 -blfix` on AA-100K, vs the same binary without `--gpu`:**

```
[GPU-KERNEL] setLikelihoodKernelGPU: Branch pointer -> GPU (clean-room lnL, -blfix lnL-only); fixed_branch_length=1 num_states=20
[GPU-BRANCH] computeLikelihoodBranchGPU active (clean-room full sweep; _pattern_lh mirrored)
[GPU-XCHECK] GPU lnL = -7541977.778778   CPU lnL = -7541977.778778 (CPU-recompute)   |d|=9.3132e-10   rel=1.235e-16   -> PASS (bridge OK)
GPU  Log-likelihood of the tree: -7541976.8566 (s.e. 15407.1942)
CPU  Log-likelihood of the tree: -7541976.8566 (s.e. 15407.1942)
GPU lnL = -7541976.8566   CPU lnL = -7541976.8566   |d|=0.0000e+00  rel=0.000e+00  -> PASS (rel<=1e-12)
```

- **GATE 1 (install):** funnel hook fired, gate passed (`fixed_branch_length=1`=BRLEN_FIX, 20-state).
- **GATE 2 (active):** the GPU pointer genuinely computed branch lnLs (not a silent CPU fallback).
- **GATE 3 (in-process self-check):** GPU clean-room sweep == an **independent CPU recompute** (via the saved
  CPU pointer, same process, same alpha/brlen) at **rel 1.235e-16** — machine epsilon.
- **GATE 4 (final lnL + s.e.):** GPU final `-7541976.8566 (s.e. 15407.1942)` == CPU final, **bit-identical
  (rel = 0.000e+00)** to all printed digits — including the s.e., proving the `_pattern_lh` mirror is faithful
  (the adversary's counterexample is resolved: `computeLogLVariance` read the GPU-populated buffer correctly).
- **GATE 5 (CPU unperturbed):** the no-`--gpu` run shows NO `[GPU-*]` markers (gate `params->gpu` false → not
  installed) and the identical lnL/s.e. CPU path byte-unchanged.
- **Build:** incremental `make exit=0`, 11.36 MB; job walltime **3:16** (recompiled the seam TUs + relinked +
  two fast `-blfix` runs). Script `gadi-ci/gpu-modelfinder/run_g2_0b_seam_v100.sh`.

(The `-blfix` lnL `-7541976.8566` / s.e. `15407.1942` differ slightly from G.2.0a's `-7541976.8530` /
`15407.1763` because `-blfix` fixes branch lengths and optimises only alpha, vs G.2.0a's full `-te` branch
optimisation — the GPU and CPU runs use identical settings, which is the correct comparison.)

**Significance.** IQ-TREE's production likelihood entry point is GPU-backed and **provably exact** (bit-identical
final lnL + s.e., independent CPU cross-check at machine epsilon, CPU path untouched). The remaining work is
purely about **branch-length optimisation on the GPU** (G.2.1: Derv + FromBuffer + device-theta coherence) so
the `-blfix` restriction can be lifted, then full `-m TEST` (G.2.2).

## G.2.1 — coherence contract VERIFIED → correctness-first design is STATELESS (de-risked the "highest-risk phase")

Before writing any device-mirror code, a 4-agent adversarial workflow (3 source tracers + 1 adversary over the
dev tree) mapped the CPU's own theta/partial staleness contract and hunted for stale-state holes a GPU
device-mirror could fall into. **Contract (all high-confidence, file:line):**
- `theta_all[ptn,c,s] = node_partial ⊙ dad_partial` (the two endpoint partials, phylokernelnew.h:2141 tip /
  2192 internal), **bound to a specific `(current_it,current_it_back)` edge**; filled in the Buffer-fill loop
  AFTER the endpoint partials are (re)swept.
- `theta_computed` flips **false** only at `optimizeOneBranch` entry (phylotree.cpp:2649); **true** only at the
  Derv/Branch tail (phylokernelnew.h:2571/3662). Reused across NR steps within one branch (only the edge
  length changes per step — `exp(eval·r·t)` is applied fresh inside the derivative kernel; the partials are
  fixed). `computeLikelihoodFromBuffer` `ASSERT(theta_all && theta_computed)` (3292).
- `partial_lh_computed & 1` = up-to-date (set at phylotree.cpp:6081 as `computeTraversalInfo` enqueues a
  branch; the postorder reuse/recompute decision is at 6051). Branch-length change → `clearReversePartialLh`
  (phylonode.cpp:22-33, **directional**, subtree away from each endpoint, only when `current_len!=optx`).
  **alpha / any model-rate-param change → `clearAllPartialLH`** (phylotree.cpp:683, full tree, called from
  rategamma.cpp:182, ratefree, all model/*.cpp, revert at 2787).

**Stale-state holes a naive device-mirror would hit (why theta_computed alone is insufficient):**
1. **#1: `clearAllPartialLH` (phylotree.cpp:683) clears every partial + nulls `current_it` but does NOT touch
   `theta_computed`.** After any alpha/model change, a theta_computed-watching GPU keeps stale device theta +
   partials → silently wrong df/ddf. No theta_computed signal exists for this event.
2. **theta is edge-bound:** `theta_computed==true` can be observed on a *different* edge (computeLikelihood picks
   a new `current_it` at 1272 after `clearAllPartialLH` nulls it) → old-edge theta reused for the new edge.
3. `computeTraversalInfo(...,false)` sets `partial_lh_computed|=1` *before* the partial VALUE is recomputed
   (the value is filled later in the packet loop) → snapshotting dirty bits after the traversal call wrongly
   reports partials fresh. The re-sweep set must come from `traversal_info` itself, not post-call bits.
4. `clearReversePartialLh` is directional + conditional (`current_len!=optx`) → must not over- or
   under-invalidate.

**THE DE-RISK (decisive):** the adversary's recommended **correctness-first variant is STATELESS** — make
`computeLikelihoodDervGPU` and `computeLikelihoodFromBufferGPU` each do a *fresh clean-room sweep from the live
tree on every call* (exactly as the validated G.2.0b Branch override already does), carrying **NO
device-resident partials/theta across calls**. With nothing persistent on the device, **every hole H1–H6 is
structurally impossible** — there is no stale state to be stale. The whole "device-mirror coherence" problem
(plan Risk #2, the reason G.2.1 was "highest-risk") simply evaporates for the correctness gate. Cost: each NR
`evalAt` is a full sweep (~38 ms) instead of the ~1 ms theta-cached K2 eval — slower, but a real `-m TEST`
per-model branch-opt is still GPU-bound, and per the user's directive we **judge the wall in situ and add the
theta-reuse/dirty-bitmap speedups only if G.2.2 misses 221.6 s** (each speedup layered behind its own
bit-identical A/B gate, gated by a NEW `clearAllPartialLH` generation counter — the one host signal that plugs
hole #1 — per the adversary's INV1–INV8). `risk_level` after the stateless design: **medium**.

### Plan — same a/b split that worked for G.2.0

- **G.2.1a — clean-room derivative cross-check (de-risk the NEW math: arbitrary-edge directed partials + df/ddf
  sign):** a read-only one-shot that, for one edge, computes GPU `df/ddf` clean-room and compares to IQ-TREE's
  own `computeLikelihoodDerv` output (in-process, CPU pointer still installed). The new piece vs G.2.0a is
  rooting the sweep at an **arbitrary edge** `(A=dad_branch->node, B=dad)`: root at A, `node_eig` = A's
  eigen-space partial over A's neighbours **except B** (leaf A ⇒ tip partial `Uinv[·][stateA]`), `dad_eig` =
  B's normal postorder partial (leaf B ⇒ tip), `theta=node_eig⊙dad_eig`; derivative kernel reuses the
  G.1.2-validated `val0=exp(eval·r·t)·prop, val1=(r·eval)·val0, val2=(r·eval)·val1`, `lh=Σval0·θ, d1=Σval1·θ,
  d2=Σval2·θ`, `pdf=d1/lh, pddf=d2/lh−(d1/lh)²`, host Kahan-sum weighted by `ptn_freq`. **Gate:** GPU `df/ddf`
  match CPU `computeLikelihoodDerv` rel ≤ 1e-9 (the sign/convention must match IQ-TREE's *returned* df/ddf, not
  just FD of lnL). Pure additive, CPU unchanged.
- **G.2.1b — wire Derv + FromBuffer (stateless), lift `-blfix`:** `setLikelihoodKernelGPU` additionally
  installs `computeLikelihoodDervGPU` + `computeLikelihoodFromBufferGPU` (stateless clean-room; FromBuffer =
  whole-tree lnL = Branch, since nothing carries theta), and the gate drops the `fixed_branch_length==BRLEN_FIX`
  requirement. derv writes **un-negated** df/ddf (computeFuncDerv phylotree.cpp:2570 negates). **Gate:** ≥3
  models (LG+G4, WAG+I+G4, a DNA GTR+G4) `--gpu -te TREE -m … ` (no `-blfix`) converged per-model lnL matches
  CPU rel ≤ 1e-9 AND optimised branch-length vector rel ≤ 1e-6; CPU byte-unchanged.

## ✅ G.2.1a — clean-room single-edge derivative cross-check (job 170258836, V100, 2026-06-08) — PASS

The new G.2.1 math validated read-only, in the real binary, vs IQ-TREE's own `computeLikelihoodDerv`:

```
[GPU-DERV-XCHECK] edge(R=100,C=101) t=0.0142171
  df:  GPU=3.323413e+01   CPU=3.323413e+01   rel=3.988e-12
  ddf: GPU=-5.124683e+06  CPU=-5.124683e+06  rel=4.543e-15   -> PASS (derivative bridge OK)
```

- **df rel 3.99e-12, ddf rel 4.54e-15** — both far inside the 1e-9 gate. The arbitrary-edge directed-partial
  sweep (two sub-root DFS passes, central edge excluded from both → `node_eig`/`dad_eig` exclude the central
  transition, which `val0=exp(eval·r·t)·prop` reapplies) + the K2 `val0/val1/val2` reduction are **exact**.
- **Sign convention confirmed un-negated:** GPU `df = Σ ptn_freq·(d1/lh) = d(lnL)/dt` matches CPU
  `computeLikelihoodDerv`'s *returned* df directly (computeFuncDerv negates downstream, phylotree.cpp:2576) —
  no flip needed in G.2.1b.
- New code (all in the existing `iqtree_gpu` lib / `phylotreegpu.cpp`, no new files → no CMake change):
  `gpu_lnl_intree.cu` gained the `k2_derv` kernel + `gpu_derv_crosscheck` launcher (`g_val0/1/2` constants);
  `phylotreegpu.cpp` gained `gpuComputeEdgeDervCleanRoom` (two-sub-root extraction) + `gpuDervCrossCheckOnce`.
- **Harness gotcha fixed (1st run, job 170258583):** the stateless GPU Branch override never populates host
  partials, so calling CPU `computeLikelihoodDerv` for the cross-check hit `!isfinite(df)` →
  "Numerical underflow (lh-derivative)" abort (phylokernelnew.h:2595). Fix: seed the R–C host partials with a
  CPU branch eval (`cpuComputeLikelihoodBranchPointer(db,R,false)` + `theta_computed=false`) before the CPU
  Derv — mirrors the normal flow where a full `computeLikelihood` precedes any derivative. (This is a
  *cross-check* artifact only; the production G.2.1b GPU Derv is stateless and needs no host partials.)
- G.2.0b lnL seam regression stayed green (rel 1.235e-16). Build incremental, walltime ~1 min. Script
  `gadi-ci/gpu-modelfinder/run_g2_1a_derv_v100.sh`.

**Significance.** Both halves of the GPU likelihood — the postorder lnL (G.2.0a/b) and the single-edge
derivative (G.2.1a) — are now proven bit-equivalent to IQ-TREE's own kernels *inside the real binary*. G.2.1b
is the mechanical step: wire the Derv/FromBuffer pointers (stateless), add leaf-endpoint synthesis so ALL edges
(incl. the ~100 leaf edges) run on GPU, drop the `-blfix` gate, validate full branch optimisation end-to-end.

### G.2.1b notes (refined by G.2.1a)
- **Leaf endpoints REQUIRED (not deferrable):** a real `optimizeAllBranches` visits leaf edges too; if those
  fell back to CPU Derv mid-GPU-opt they would hit the same stale-host-partial underflow. So ALL edges must run
  on GPU. A leaf endpoint's directed partial is the tip eigen vector `Uinv[·][s]` (or `UinvRowSum` if ambiguous),
  rate-independent — synthesised by a small `k_leaf_eig` kernel into a scratch slot; the internal endpoint is a
  normal sub-root. (≤1 leaf per edge for a bifurcating tree.)
- **+I must gate out:** the clean-room sweep omits the `ptn_invar` term, so `site_rate->getPInvar() > 0` ⇒
  return NaN ⇒ CPU fallback (add to both helpers). ⇒ the 3-model gate uses **+G4-only** GPU-handled models
  (LG+G4, WAG+G4, DNA GTR+G4); +I/+R/+ASC stay CPU (a `-m TEST` run mixes GPU + CPU-fallback candidates).
- Stateless ⇒ each Derv/FromBuffer call is a full sweep (no theta cache) ⇒ slow but correct; judge wall at G.2.2.

## G.2.1b — full GPU branch optimisation, end-to-end (✅ PASS, bit-identical, 2026-06-08)

The whole Newton-Raphson branch-length optimisation now runs **entirely on the GPU**. The `-blfix` gate is
dropped; `setLikelihoodKernelGPU()` installs all three overrides (`computeLikelihoodBranchGPU` +
`computeLikelihoodDervGPU` + `computeLikelihoodFromBufferGPU`), each a stateless clean-room sweep, backed by a
persistent device-buffer pool so the ~6 GB partial arena is allocated once and reused (contents recomputed every
call ⇒ statelessness preserved — only the *allocation* persists).

**Validation in two stages.**

1. **Read-only derivative regression** (job 170259046, still under `-blfix` so the seam stays inert) — confirms
   the two edge types a real `optimizeAllBranches` hits, against IQ-TREE's OWN `computeLikelihoodDerv`:
   - INT-INT edge (node=101, dad=100, t=0.0142): df rel **3.99e-12**, ddf rel **4.54e-15**
   - LEAF edge (node=98, dad=101, t=0.0724): df rel **5.93e-13**, ddf rel **9.85e-16**

   The LEAF pass validates `k_leaf_eig` (tip eigen partial `Uinv[·][s]` / `UinvRowSum`, synthesised into a scratch
   slot, rate-independent). The G.2.0b lnL seam self-check stayed green (rel 1.235e-16).

2. **Full integration** (job 170259325): `--gpu -te -m LG+G4 -nt 1` (NO `-blfix`) vs the same binary without
   `--gpu`. All four markers fired — `[GPU-KERNEL] … fixed_branch_length=0`, `[GPU-BRANCH]`, `[GPU-DERV]`,
   `[GPU-FROMBUF]` — proving every likelihood path ran on the GPU through full branch-opt.

   | gate | result |
   |---|---|
   | GPU final lnL vs CPU | **−7541976.8530 == −7541976.8530, rel = 0.0** |
   | 197 optimised branch lengths | **worst_rel = 0.0** (all bit-identical) |
   | convergence path | identical round-by-round: −7541977.779 → −7541976.940 → −7541976.853 |
   | CPU run (no `--gpu`) | byte-unchanged, no `[GPU-*]` markers |

   **Honest reading of "0.0":** the `.treefile` writes branch lengths to ~6–7 significant figures, so worst_rel=0.0
   means GPU and CPU agree to *every written digit*. Underneath, the gradient agrees to ~1e-12 (df) / ~1e-15 (ddf)
   and Newton-Raphson lands on the same optimum within the optimiser's ε, so the true sub-1e-7 FP differences fall
   below the treefile's precision. This is **stronger** than the rel≤1e-9 (lnL) / rel≤1e-6 (brlen) gates the phase
   demanded.

**Wall — the signal that drives G.2.2.** GPU **1063 s** vs CPU **225 s** for one model ⇒ GPU **4.7× slower**. This
is exactly the stateless re-sweep cost the G.2.1 coherence contract predicted (a full ~38 ms postorder sweep per
NR `evalAt`, with no theta cache). **Caveat against over-extrapolation:** `-te` does *full* branch optimisation
(197 branches × NR) — the heaviest per-model workload there is. ModelFinder per-candidate is much lighter (fixed
tree, mostly rate/shape parameters), so "4.7×" must NOT be read as the MF slowdown — G.2.2 has to measure the real
`-m TEST` wall. (PBS reported "GPU Utilisation 0%": a sampling artifact — 6.05 GB GPU memory confirms the kernels
ran; the path is latency-bound at ~25 % occupancy, as the K3/K5 profiling already established.)

**What this de-risks.** The device-mirror coherence problem (Risk #2, flagged in G.2.1 as the "highest-risk phase")
never materialised — there is no device-resident state to go stale. The theta-reuse + `clearAllPartialLH`
generation-counter dirty-bitmap remain the gated speedup lever, to be applied **only if G.2.2 misses the wall**
(per the user's standing "revisit kernel only if wall missed"), each behind a bit-identical A/B gate.

**Code** (all in existing files; no new TUs beyond G.2.0a's): `tree/gpu/gpu_lnl_intree.cu` += `k_leaf_eig` kernel
+ persistent `DevBuf` pool (`devbuf_ensure`/`DEVB`, 8 static buffers, no `cudaFree` at exit);
`tree/phylotreegpu.cpp` += `computeLikelihoodDervGPU` + `computeLikelihoodFromBufferGPU` +
`gpuComputeEdgeDervCleanRoom` leaf-endpoint support + `setLikelihoodKernelGPU` 3-pointer install (BRLEN_FIX gate
dropped) + `+I` gate (`getPInvar()>0` ⇒ NaN ⇒ CPU). Script `gadi-ci/gpu-modelfinder/run_g2_1b_full_v100.sh`.
**Nothing committed — all local edits to `/scratch/rc29/as1708/iqtree3-gpu` (branch `gpu-kernel`).**

**Next:** multi-model gate (WAG+G4 AA, DNA GTR+G4; confirm WAG+I+G4 falls back to CPU), then **G.2.2** full
`-m TEST` with a persistent GPU instance reused across candidates + per-evaluate CPU fallback for unsupported
models — gate = same best model as CPU + displayed lnL rel≤1e-12 + identical AIC/BIC ranking + MF wall vs 221.6 s.
