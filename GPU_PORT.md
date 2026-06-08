# IQ-TREE 3 — GPU ModelFinder port (dev tree)

This tree is the **GPU dev clone** of the FCA IQ-TREE source (cloned from
`/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3`, HEAD `5604606d` "Phase A.2: WarmStartPacket"),
on branch **`gpu-kernel`**. It builds the CPU FCA binary today; GPU CUDA kernels are being added here.

**Status: G.1.0 (build scaffold) ✅ PASS** (2026-06-07, job 170176864 on a V100). `-DIQTREE_GPU=ON` builds
a `.cu` into `iqtree3` and `--gpu` launches a kernel with a clean `cudaGetLastError`; `-DIQTREE_GPU=OFF`
is the unchanged CPU build. Build+test job: `setonix-iq/gadi-ci/gpu-modelfinder/run_g1_0_build_gpuvolta.sh`;
log: `setonix-iq/research/Modelfinder/gpu-modelfinder-g1-log.md`. **NOTE: the clone's `cmaple`/`lsd2` git
submodules must be populated** (rsync from the FCA source) before configuring — `git clone` leaves them
empty and configure dies on `Can not find target maple/lsd2`.

## The plan lives in setonix-iq
The authoritative, phase-by-phase implementation plan is **PART II** of
`setonix-iq/research/Modelfinder/gpu-modelfinder-design.md` (branch `gpu-modelfinder`). Read it before
touching code. Companion records: `gpu-modelfinder-g0-log.md` (G.0 de-risk + BEAGLE bugs 1–16),
`Trimorph.md` (CPU-failure post-mortems), `CHANGELOG.md` (the wall-clock validation targets).

## The one thing to internalise first
IQ-TREE optimises branches **one at a time** by Newton-Raphson:
`optimizeAllBranches` (tree/phylotree.cpp:2732) → `optimizeOneBranch` (:2628) → `computeFuncDerv` (:2563)
→ **`computeLikelihoodDerv`** (a *single-edge* lnL+df+ddf from `theta`, tree/phylokernelnew.h:2239-2424).
There is **no pre-order all-branch gradient pass** on the critical path — so BEAGLE's broken 20-state
pre-order kernel and O(depth) pre-order buffer recycling are an OPTIONAL advanced track (plan §II.8 G.5),
NOT the ModelFinder critical path. The GPU must accelerate **postorder partials + single-edge df/ddf**
(75–85 % of per-model wall is branch re-opt).

## Integration seam (surgical)
Override the 4 `PhyloTree` function pointers via a new `setLikelihoodKernelGPU()`:
`computePartialLikelihoodPointer`, `computeLikelihoodBranchPointer`, `computeLikelihoodDervPointer`,
`computeLikelihoodFromBufferPointer` (tree/phylotree.h:904,951,1004 + **computeLikelihoodDervPointer at 1346** — NOT all in 902-1004; assigned in phylotreeavx.cpp:50-145;
invoked via wrappers in phylotreesse.cpp:212-236). Runtime switch = `Params::gpu` / `--gpu`; build switch
= `option(IQTREE_GPU)` + `#cmakedefine IQTREE_GPU`. CPU path stays default. Gate GPU on
`model_test_and_tree==false` (fixed topology) + reversible bifurcating models; else fall back to CPU.

## Build (Gadi, in a PBS GPU job — never on login)  [G.1.0 DONE — recipe below is proven]
All-GCC host (design §II.9): `module load cuda/12.5.1 gcc/12.2.0 cmake/3.24.2 eigen/3.3.7 boost/1.84.0`
(NO intel for now). `CC=gcc CXX=g++`, `-DCMAKE_CUDA_HOST_COMPILER=g++` (system g++ 8.5 too old for CUDA 12).
gpuvolta = **Cascade Lake** ⇒ NO `-march=sapphirerapids` (SIGILL); generic build + runtime ISA dispatch.
The `iqtree_gpu` `.cu` static lib sits in a gated block before `add_subdirectory(main)`: `if(IQTREE_GPU)
{ CMP0104 NEW; set CMAKE_CUDA_ARCHITECTURES "70;80;90" BEFORE enable_language(CUDA); find_package(CUDAToolkit);
add_library(iqtree_gpu STATIC tree/gpu/*.cu); target_link_libraries(iqtree_gpu PUBLIC CUDA::cudart) }`,
linked into `iqtree3` after the master link; `-static`→`-rdynamic` guard (cudart can't static-link).

## Phase ladder (each independently testable — see plan §II.8 for tests + validation numbers)
G.1.0 build scaffold **✅DONE** · G.1.1 postorder lnL kernel · G.1.2 single-edge derivative kernel ·
G.1.3 CUDA-Graph capture of optimizeAllBranches · G.2 `-m TEST` integration (MF wall < 221.6 s, lnL
−7541976.86) · G.3 `-m MF` heavy (+R10 one-pass, < FCA np1 1341 s) · G.4 AA-1M tiling · G.5 (optional)
Ji O(N) gradient + buffer recycling. FP64 unscaled; native-20 states; compact tip states; NCAT≤10 one pass.
**Non-negotiable gate: FD-validate every GPU gradient against the CPU oracle (g4 <3e-3, g1 <1e-6).**
