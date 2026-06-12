# GPU ModelFinder — source code

This directory holds the **GPU ModelFinder source code** (custom in-tree CUDA kernels + the JOLT optimizer) for the
IQ-TREE 3 GPU offload project. The development fork lives at `/scratch/rc29/as1708/iqtree3-gpu` (branch `gpu-kernel`,
HEAD `d3e5cd82`), whose `origin` is a *local* path — so this branch is how the source reaches GitHub. The narrative,
benchmarks, and phased design are in [`../research/Modelfinder/`](../research/Modelfinder/) (part3–part9) and
[`../CHANGELOG.md`](../CHANGELOG.md).

## What this is

A from-scratch CUDA implementation of phylogenetic-likelihood ModelFinder for IQ-TREE — **not** BEAGLE. The core is
**JOLT**: a GPU-native joint Levenberg–Marquardt optimizer that optimizes all branch lengths + α (+ pinv) in parallel
sweeps (the Ji-2020 linear-time 2-traversal gradient), reaching the **same MLE** as IQ-TREE's sequential per-edge
Newton + α-Brent. It is wired in-tree behind a `--jolt` flag, is thread-safe (a process mutex serializes the GPU while
CPU-fallback candidates run concurrently), and is bit-parity-validated against the CPU likelihood engine (GPU≡CPU
per-pattern log-likelihood to rel ~1e-12; FP64 throughout — never reduced precision on the exact path).

## Layout

```
gpu-src/
  README.md                     this file
  gpu_modelfinder_full.patch    the COMPLETE GPU ModelFinder changeset (git diff vs the pre-GPU base
                                5604606d..d3e5cd82) — 15 files, applyable with `git apply`
  src/
    tree/gpu/gpu_lnl_intree.cu  the CUDA kernels + the gpu_jolt_optimize launcher (the heart)
    tree/gpu/gpu_iqtree.h       device/launcher declarations
    tree/gpu/gpu_diag.cu        device diagnostics
    tree/phylotreegpu.cpp       host-side integration: PhyloTree::optimizeParametersJOLT (the eligibility
                                gate + base_invar/+I + write-back) and the GPU branch-opt overrides
```

The four files under `src/` are **GPU-native** (they exist only for this project) and are copied here in full for
direct reading. The remaining integration is a set of hooks into existing IQ-TREE files — captured precisely by the
patch (see below).

## The full changeset (`gpu_modelfinder_full.patch`)

`git diff` from the last pre-GPU commit (`5604606d`, an FCA/MPI CPU commit) to the current GPU HEAD (`d3e5cd82`).
It isolates *only* the GPU ModelFinder work from the CPU/FCA fork, across all 15 touched files:

| file | role |
|---|---|
| `tree/gpu/gpu_lnl_intree.cu` | **all CUDA kernels** (k1_node postorder lnL, kj_pre preorder, kj_derv per-edge derivative, kj_ratenum FreeRate rate-gradient, kj_reduce3 on-device reduction, k_leaf_eig) + `gpu_jolt_optimize` |
| `tree/phylotreegpu.cpp` | `optimizeParametersJOLT` eligibility gate, +I base_invar, write-back, the GPU branch-opt overrides |
| `tree/gpu/gpu_iqtree.h`, `gpu_diag.cu` | launcher API + diagnostics |
| `model/modelfactory.cpp` | the `optimizeParameters` JOLT hook + the +I+G `n_pinv_starts` 10→4 restart logic |
| `tree/phylotree.{cpp,h}`, `tree/phylotreesse.cpp` | the GPU kernel install seam (gated off under `--jolt`) |
| `main/main.cpp`, `utils/tools.{cpp,h}` | the `--jolt` / `--gpu` command-line flags |
| `CMakeLists.txt`, `tree/CMakeLists.txt`, `iqtree_config.h.in`, `GPU_PORT.md` | the `IQTREE_GPU` CMake option + build wiring |

## Build (in the full fork)

```bash
module load gcc/12.2.0 cuda/12.5.1 eigen/3.3.7 boost/1.84.0 cmake/3.24.2
cmake -DIQTREE_GPU=ON ..        # enables the CUDA path; CPU build is byte-unchanged when OFF
make -j4 iqtree3
# run:  iqtree3 --jolt --gpu -m MF -s alignment.phy -nt N
```

## Status (see part9 for the full plan)

Validated: JOLT correct + in-tree + thread-safe; AA `-m MF` is **95% on the GPU** (the empirical-matrix family ×
{+G4,+I+G4,+F+G4,+F+I+G4}); +I via a 4-start restart (G.4.3c); the on-device reduction `kj_reduce3` (G.5.0,
116/116 models GPU≡CPU rel ≤ 2.17e-10). Next: +R FreeRate coverage (G.5.1), VRAM tiling (G.5.2), and free-Q DNA
(G.6 — the DNA `-m MF` enabler).
