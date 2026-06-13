# GPU ModelFinder — source code

This directory holds the **GPU ModelFinder source code** (custom in-tree CUDA kernels + the JOLT optimizer) for the
IQ-TREE 3 GPU offload project. The development fork lives at `/scratch/rc29/as1708/iqtree3-gpu` (branch `gpu-kernel`,
HEAD `3ec1b5c8`), whose `origin` is a *local* path — so this branch is how the source reaches GitHub. The narrative,
benchmarks, and phased design are in [`../research/Modelfinder/`](../research/Modelfinder/) (part3–part10) and
[`../CHANGELOG.md`](../CHANGELOG.md).

## What this is

A from-scratch CUDA implementation of phylogenetic-likelihood ModelFinder for IQ-TREE — **not** BEAGLE. The core is
**JOLT**: a GPU-native joint Levenberg–Marquardt optimizer that optimizes all branch lengths + α (+ pinv) (+ for DNA,
the free-Q exchangeabilities) in parallel sweeps (the Ji-2020 linear-time 2-traversal gradient), reaching the **same
MLE** as IQ-TREE's sequential per-edge Newton + α-Brent + BFGS-over-Q. It is wired in-tree behind a `--jolt` flag, is
thread-safe (a process mutex serializes the single GPU while CPU-fallback candidates run concurrently across models),
and is bit-parity-validated against the CPU likelihood engine (GPU≡CPU per-pattern log-likelihood to rel ~1e-12;
**FP64 throughout — never reduced precision on the exact path**). The CPU build is byte-unchanged when `IQTREE_GPU` is
OFF and when `--jolt` is not passed.

It is delivered as **Coarse-to-Fine (CTF) ModelFinder**: rank the full `-m MF` candidate set on a ~5000-site column
subsample by *native subsample BIC*, then JOLT-refine the top-3 on full data. The statistical licence for CTF (the
subsample-sufficiency hypothesis) and a confirmed projection-amplification bug + fix are in part10.

## Layout

```
gpu-src/
  README.md                     this file
  gpu_modelfinder_full.patch    the COMPLETE GPU ModelFinder changeset (git diff vs the pre-GPU base
                                5604606d..3ec1b5c8) — 18 files, applyable with `git apply`
  src/
    tree/gpu/gpu_lnl_intree.cu  the CUDA kernels + the gpu_jolt_optimize launcher (the heart)
    tree/gpu/gpu_iqtree.h       device/launcher declarations + the free-Q decompose-callback ABI
    tree/gpu/gpu_diag.cu        device diagnostics
    tree/phylotreegpu.cpp       host-side integration: PhyloTree::optimizeParametersJOLT (the eligibility
                                gate + base_invar/+I + free-Q write-back + the GPU↔CPU self-check/safety gate)
                                and the GPU branch-opt overrides
```

The four files under `src/` are **GPU-native** (they exist only for this project) and are copied here in full for
direct reading. The remaining integration is a set of hooks into existing IQ-TREE files — captured precisely by the
patch (see below).

## The full changeset (`gpu_modelfinder_full.patch`)

`git diff` from the last pre-GPU commit (`5604606d`, an FCA/MPI CPU commit) to the current GPU HEAD (`3ec1b5c8`).
It isolates *only* the GPU ModelFinder work from the CPU/FCA fork, across all 18 touched files (2102 insertions):

| file | role |
|---|---|
| `tree/gpu/gpu_lnl_intree.cu` | **all CUDA kernels** (k1_node postorder lnL, kj_pre preorder, kj_derv/kj_derv_fused per-edge derivative + per-category accumulators, kj_ratenum FreeRate rate-gradient, kj_reduce3/kj_invl/kj_reduce_gradnum on-device reductions, k_leaf_eig) + `gpu_jolt_optimize` (the joint LM, incl. the free-Q FD branch driven by a host decompose-callback) |
| `tree/phylotreegpu.cpp` | `optimizeParametersJOLT`: the eligibility gate, +I `base_invar`, **free-Q write-back** (`gpuSetFreeParamsDecompose`), and the **GPU↔CPU self-check + safety gate** (write-back rel > 1e-6 → NaN → CPU fallback); plus the GPU branch-opt overrides |
| `tree/gpu/gpu_iqtree.h`, `gpu_diag.cu` | launcher API + the `jolt_qdecompose_fn` C-ABI + diagnostics |
| `model/modelmarkov.{cpp,h}`, `model/modelsubst.h` | the two behaviour-neutral free-Q wrappers `gpuGetFreeParams`/`gpuSetFreeParamsDecompose` (over the protected `(set/get)Variables`, so the model's `param_spec` gives **one code path for HKY..GTR**); virtual no-ops on the base class |
| `model/modelfactory.cpp` | the `optimizeParameters` JOLT hook + the +I+G `n_pinv_starts` 10→4 restart logic |
| `tree/phylotree.{cpp,h}`, `tree/phylotreesse.cpp` | the GPU kernel install seam (gated off under `--jolt`) + the diagnostic cross-check hooks |
| `main/main.cpp`, `utils/tools.{cpp,h}` | the `--jolt` / `--gpu` command-line flags |
| `CMakeLists.txt`, `tree/CMakeLists.txt`, `iqtree_config.h.in`, `GPU_PORT.md` | the `IQTREE_GPU` CMake option + build wiring |

## Build (in the full fork)

```bash
module load cmake/3.24.2 gcc/12.2.0 cuda/12.5.1 eigen/3.3.7 boost/1.84.0
cmake -DIQTREE_GPU=ON ..        # enables the CUDA path; CPU build is byte-unchanged when OFF
make -j8 iqtree3
# NOTE: an incremental relink needs the CUDA lib dir on the linker path, else
#   /bin/ld: cannot find -lcudadevrt / -lcudart_static
export LIBRARY_PATH="$CUDA_HOME/lib64:$LIBRARY_PATH"
# run:  iqtree3 --jolt --gpu -m MF -s alignment.phy -nt N
```

## Validated status

| phase | capability | status |
|---|---|---|
| G.1–G.4 | in-tree CUDA kernels; JOLT joint optimizer (branches + α + pinv); bit-parity vs CPU; in-tree `--jolt` seam; thread-safe under across-model OpenMP | ✅ |
| G.4.3c | +I via a multimodal-gated **4-start** pinv restart (single-start loses 39.5 nat at pinv≈0.5) | ✅ |
| G.5.0 | on-device reduction (Part A+B) + kernel fusion + base-sweep-skip + `d_theta` reclaim → **A100 beats np8** | ✅ |
| G.5.1a | +R FreeRate **weight gradient** `gz_c = WN_c − w_c·N` FD-validated (`Σ WN_c = N` exact; `Σ gz_c = 0`) | ✅ *(gradient only)* |
| G.6.0a/b | **free-Q DNA** lnL pipeline BIT-IDENTICAL GPU==CPU at every perturbed Q; FD-LM optimizer converges to the CPU MLE on HKY..GTR (incl. GTR's 5 **coupled** exchangeabilities — no dense-Q fallback needed) | ✅ |
| G.6.1 | free-Q **ON BY DEFAULT** (`JOLT_NO_FREEQ` escape hatch) + write-back safety gate; DNA `-m MF` coverage **8 → 70** engage | ✅ |
| G.6.2 | **DNA-1M `-m MF` CTF payoff** — winner **F81+F+G4 == IQ-TREE's own full `-m MF` BIC winner** (w-BIC 0.998); subsample top-3 == full-data top-3 in order | ✅ |
| audit | independent audit (`3ec1b5c8`): core machinery CLEAN; RISK-1 (tied-freq eligibility) + RISK-3 (NaN-rel) fixed | ✅ |

**Coverage:** AA `-m MF` ~95 % on GPU (the whole empirical-matrix family × {+G4,+I+G4,+F+G4,+F+I+G4}; the residual is
`+R`); DNA `-m MF` ~89 % on GPU after free-Q (residual `+R`/`+I+R`/pure-`+I`). FP64 parity is bit-level on the lnL and
gradient paths in every engaged model.

## Measured benchmarks (seed 1, full-node CPU baselines)

- **AA-1M `-m MF` CTF** (full set incl. +R): H200 **767 s** / A100 1122 s, winner **LG+G4** (== CPU oracle), 116/122 GPU
  engagements; vs CPU `-m TEST` MF: np16 1.46× (H200).
- **AA-1M `-m TEST` CTF**: H200 **893 s** = **1.26× np16** (the only GPU that beats 16 nodes); A100 1139 s with
  PartB+fusion = **0.99× np16** (honest: improved 1355→1139 s but still does *not* beat 16 nodes), 1.27× np8.
- **DNA-1M `-m MF` CTF**: A100 **152 s** vs the measured single-node CPU `-m MTEST` (176 models) **1714.9 s = 11.3×**
  wall / **43.6× energy** (7.53 vs 328.5 Wh); winner **F81+F+G4 == CPU BIC winner**.

**Honest ceiling (part5, unchanged):** the per-model GPU is mutex-serialized (S≈4.8) vs N CPU-concurrent, so the
`-m MF` win lives in the **CTF top-k≤3 refine** (depth) + the **1M/10M bandwidth regime**, *not* full-set breadth (the
CPU cluster's strength). CTF refines model params on a fixed coarse tree, so the GPU's absolute lnL sits below the
CPU's fully-tree-searched lnL — but the model **selection** (ModelFinder's actual job) is exact in every benchmark.

## Independent code audit (2026-06-13, commit `3ec1b5c8`)

An adversarial audit of the G.5.1a + G.6 changeset (`65e45c4c..d5d69b48`) ran before this release. **Core machinery
verdict: CLEAN** — write-back ordering (Q → pinv → α → `clearAllPartialLH` → fresh `computeLikelihood`), the
process-mutex coverage of the *whole* `gpu_jolt_optimize` body (device symbols + the persistent DevBuf pool + the
decompose-callback), the base-sweep-skip ↔ Q-FD state coherence, FP64 parity, the 1-based `getVariables()` indexing,
and the NaN→CPU fallback were all verified; the safety-gate `rel` is a genuine GPU-vs-independent-CPU recompute (not a
tautology). **Two real holes were found and fixed:**

- **RISK-1** (`phylotreegpu.cpp`): the free-Q eligibility gate excluded only `FREQ_ESTIMATE`, not the DNA
  **tied-frequency** types (`+FRY`/`+F1112`/… = `FREQ_DNA_*`) whose `getNDim()` packs 1–3 *frequency* params into the
  Q-vector tail — the launcher would mis-clamp them as exchangeabilities (`[1e-4,100]` vs the correct `~[0,1]`),
  yielding a coherent-but-**suboptimal** lnL that passes the coherence gate. Not in the default `-m MF` set (no live
  regression), but a silent-correctness hole for explicit user models. **Fix:** require `nFreqParams(getFreqType())==0`
  (0 for the default `+FQ`/`+F`; >0 only for tied types, which now decline to the CPU).
- **RISK-3** (`phylotreegpu.cpp`): `rel > 1e-6` is *false* for NaN `rel`, so the safety gate didn't fire and then
  `setCurScore(NaN)` poisoned `_cur_score`. **Fix:** `if (!(rel <= 1e-6)) return NAN;` (returns before `setCurScore`).

Validated regression-free (job 170863975, V100): `GTR+F+G4` engages rel 5.193e-12 (== pre-fix), `GTR+F+I+G4` engages,
`JOLT_NO_FREEQ` declines.

## Next steps (ranked, audit-informed)

1. **G.5.1b — +R / +I+R in-tree JOLT (the critical path).** The only remaining AA `-m MF` gap and the load-bearing
   multimodal-convergence piece: a standalone cold/warm-vs-CPU-EM convergence harness → flip the `non-mean-gamma` gate
   → the CPU-optimum comparison gate. The gradient (`gz_c`) is already FD-validated (G.5.1a); the optimiser branch is
   increment 2. `+I+R` declines to CPU initially (its `pinv=1−Σprop` coupling differs from the +I+G form).
2. **Close RISK-2 (coherence-vs-optimality) generally.** Fold the CPU-optimum comparison gate (assert JOLT lnL ≥
   CPU-refined − `modelfinder_eps`, else NaN→CPU) into the free-Q path too, so a future regime that converges below the
   CPU MLE is caught per-candidate at runtime (currently relied on offline G.6.0b validation).
3. **Runtime-confirm the RISK-1 fix** (`JOLT_DEBUG=1 -m MF` DNA: `freeQok` never fires with `nFreqParams>0`; explicit
   `GTR+FRY+G4 -te` declines).
4. **G.5.2 — VRAM tiling of the postorder arena** — gated on the AA-10M `-m MF` result: if 10M LG+G4/LG+I+G4 fits H200
   (~58 GB est.) tiling is deferrable; if it OOMs, tiling moves onto the critical path for scale.
5. **Port the native-BIC gate + rate-het detector + wall budget into the production CTF path** (currently only in the
   benchmark scripts).
6. **Verify the audit's two static-only items**: `cuobjdump` 32-reg / 100 % occupancy on `kj_derv_fused`, and a wider
   fixed-Q self-check sweep confirming the ~1e-16 reassociation bound across (ncat, nptn) regimes.
