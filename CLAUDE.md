# IQ-TREE GPU Offload — Development Guide

## What is this?

GPU offloading of IQ-TREE's phylogenetic likelihood kernels from `phylokernelnew.h` to AMD Instinct MI250X GPUs on Setonix (Pawsey). Five functions consume >95% of total CPU time and are excellent GPU candidates. A proof-of-concept GPU implementation exists in `poc-gpu-likelihood-calculation-main/` with CUDA/cuBLAS/OpenACC/OpenMP-GPU backends; these need porting to HIP/ROCm for Setonix's AMD GPUs.

## Quick reference

- **IQ-TREE source:** `ANUHPC-iqtree/` (IQ-TREE 3.1.0, C++17, AVX SIMD, OpenMP)
- **GPU PoC:** `poc-gpu-likelihood-calculation-main/` (multi-backend GPU likelihood calculator)
- **Core kernel:** `ANUHPC-iqtree/tree/phylokernelnew.h` (3,688 lines)
- **Profiling report:** `ANUHPC-iqtree/PROFILING_REPORT.html`
- **Progress log:** `CHANGELOG.md` ← **read this first when resuming**
- **Dataset:** `medium_dna.phy` — 50 taxa × 5,000 sites
- **Profiling HPC:** hpc-01, Intel Xeon E5-2670 @ 2.60GHz, 32 cores, 62 GB DDR3
- **Target HPC:** Setonix (Pawsey) — AMD EPYC 7A53 "Trento" + 8× AMD Instinct MI250X per node, HIP/ROCm

## Orientation (read this first when starting a session)

1. **Read `CHANGELOG.md`** to see what's done, what failed, and what's next.
2. Run the current test suite (if one exists) to see status: `cd build && ctest --output-on-failure 2>&1 | tail -20`
3. Pick the next unchecked item from CHANGELOG.md.
4. **When you finish a unit of work, update CHANGELOG.md before stopping.**
5. Commit and push after every meaningful unit of work.

## Principles for autonomous development

These are adapted from the [long-running Claude](https://www.anthropic.com/research/long-running-Claude) pattern and the clax Boltzmann solver project.

### 1. IQ-TREE CPU output is the oracle — tests are everything

The existing CPU SIMD kernels in `phylokernelnew.h` produce known-good results. GPU kernels must match them. This is our "GCC oracle" — the CPU is always right, GPU must match it.

Rules:
- Never commit GPU code that breaks agreement with CPU output.
- Every new kernel must have a corresponding correctness test BEFORE implementation. Write the test (specifying what the CPU produces), then make it pass.
- When a discrepancy is found, add a test that reproduces it before fixing it.
- Numerical tolerance: ≤1e-10 relative error for double precision. Document any larger deviations with root-cause analysis (FP reordering, fused multiply-add differences, etc.).
- When debugging a discrepancy, bisect: is it in partial likelihood, buffer, or derv? Compare intermediate values at specific patterns.

### 2. Concise test output (context window hygiene)

LLMs have finite context windows. Every line of noisy output displaces useful information.

Rules:
- Tests print at most 5-10 lines on success, ~20 lines on failure.
- Print summary statistics: max relative error, speedup ratio, pass rate.
- Log verbose diagnostics to files, not stdout.
- When comparing arrays, print: max relative error, the pattern index where it occurs, and the overall pass rate. Not the full arrays.

Good:
```
PASS computeLikelihoodDervSIMD — max rel err 2.3e-15 at ptn=1847 (5000/5000 patterns match, tolerance 1e-10)
GPU: 0.42s | CPU 1T: 92.6s | speedup: 220×
```

Bad:
```
FAILED — arrays not equal:
  [1.0183e-4, 1.0182e-4, 1.0181e-4, ...]  (5000 more lines)
```

### 3. Keep CHANGELOG.md current (agent memory across sessions)

Each agent session starts with zero memory of previous sessions. CHANGELOG.md is the shared memory. Without it, agents waste time re-discovering what's done and what's broken.

Rules:
- Update CHANGELOG.md after every meaningful unit of work.
- Check off completed items with dates.
- Note what worked, what didn't, what's blocked.
- **Record failed approaches so they aren't re-attempted.** E.g.: "Tried hipify-perl on MatrixKernels.cu — broke template specializations. Manual port required."
- Add new tasks discovered during implementation.
- Track accuracy tables and speedup ratios at key checkpoints.

### 4. Small, testable commits with git coordination

Rules:
- Each commit implements one thing (one kernel, one bugfix).
- Each commit passes all existing tests.
- Run `ctest --output-on-failure -q` (or equivalent) before every commit. Never commit code that breaks existing passing tests.
- Commit and push after every meaningful unit of work. This gives a recoverable history and makes progress visible.
- Avoid large commits that change multiple kernels at once.

### 5. Prevent regressions

- Run likelihood comparison tests before every commit.
- If GPU output diverges from CPU after a change, fix it before committing.
- If a new feature requires changing behavior in an existing test, update the test explicitly (don't delete or skip it).
- Track test pass rates and speedup ratios over time in CHANGELOG.md.

### 6. Structure work for parallelism

Easy to parallelize (independent tasks):
- `computeLikelihoodBufferSIMD` GPU kernel (trivial element-wise multiply)
- `computeLikelihoodBranchSIMD` GPU kernel (single dot product + log)
- `computeLikelihoodDervSIMD` GPU kernel (triple dot product)
- Correctness tests for each kernel (independent)

Hard to parallelize (coupled tasks):
- `computePartialLikelihoodSIMD` (tree-order dependencies, must be post-order)
- Integration with IQ-TREE tree search loop (touches many files)
- Full end-to-end validation (depends on all kernels)

## Profiling summary (VTune + perf, April 2026)

### The five hot functions (>95% of CPU time)

| Priority | Function | Default 1T | GPU Suitability | Inner Work |
|----------|----------|-----------|-----------------|------------|
| #1 | `computeLikelihoodDervSIMD` | 92.6s (40.1%) | ★★★★★ | 3 dot products/pattern, ~60 FLOPs |
| #2 | `computePartialLikelihoodSIMD` | 42.2s (18.3%) | ★★★★☆ | ~480 FLOPs/pattern, tree-order deps |
| #3 | `computeLikelihoodBufferSIMD` | 8.7s (3.8%) | ★★★★★ | Element-wise multiply, trivial |
| #4 | `computeLikelihoodBranchSIMD` | 0.76s (0.3%) | ★★★★★ | 1 dot product + log reduction |
| #5 | `computeLikelihoodFromBufferSIMD` | 0.41s (0.2%) | ★★★★★ | Wrapper for BranchSIMD |

### Why GPU beats multi-threaded CPU here

| Problem | Evidence | GPU Solution |
|---------|----------|-------------|
| OpenMP overhead | 22-29% of CPU at 4T+ | No fork-join; persistent kernel execution |
| Frontend stalls | 52% (1T) → 70% (8T) | Simple GPU kernels; no I-cache pressure |
| Backend stalls | 17% (1T) → 48% (8T) | No cache coherence (thread-private registers) |
| IPC degradation | 1.63 (1T) → 0.89 (8T) | Thousands of wavefronts hide latency |
| Parallel efficiency | 45% at 8T | >90% with 5,000 patterns mapped to GPU threads |

### Data sizes (all fit in MI250X 128 GB HBM2e)

| Dataset | Taxa | Sites | Total (50-taxon tree) |
|---------|------|-------|-----------------------|
| medium_dna | 50 | 5,000 | ~30 MB |
| large_dna | 100 | 10,000 | ~120 MB |
| stress_dna | 200 | 20,000 | ~500 MB |

### Baseline wall-clock times (pre-GPU, hpc-01 Xeon)

| Dataset / config | 1T | 4T | 8T |
|-----------------|-----|-----|-----|
| medium_dna (Default) | 230.9s | 89.2s | 64.2s |
| medium_dna (GTR+G4) | 166.8s | 68.2s | — |

## Architecture

### IQ-TREE kernel structure (CPU, SIMD — phylokernelnew.h)

```
phylokernelnew.h (3,688 lines):
├── Helper functions: sumVec, dotProductVec, dotProductDualVec, dotProductTriple, productVecMat
├── computePartialLikelihoodSIMD()  — partial likelihood at internal nodes (post-order traversal)
├── computeLikelihoodBufferSIMD()   — theta = plh_node × plh_dad (element-wise)
├── computeLikelihoodDervSIMD()     — df, ddf via triple dot product (Newton-Raphson)
├── computeLikelihoodBranchSIMD()   — tree log-likelihood (single dot product + log)
└── computeLikelihoodFromBufferSIMD() — wrapper using precomputed buffer
```

### Call hierarchy (tree search — how kernels are invoked)

```
doTreeSearch
  └─ doNNISearch
      └─ evaluateNNIs
          └─ getBestNNIForBran
              └─ optimizeOneBranch
                  └─ minimizeNewton
                      └─ computeFuncDerv
                          ├─ computeLikelihoodBufferSIMD  [once per branch — builds theta cache]
                          └─ computeLikelihoodDervSIMD    [3-5× per branch — Newton iterations]

computePartialLikelihoodSIMD    [pre-search, via computeTraversalInfo]
computeLikelihoodBranchSIMD     [ModelFinder, via testModel → computeLikelihood]
```

### Hot loop details (DNA, nstates=4, ncat=4, block=16)

**computeLikelihoodDervSIMD** (Priority #1):
```cpp
for (ptn = ptn_lower; ptn < ptn_upper; ptn += VectorClass::size()) {
    theta = theta_all + ptn * block;
    dotProductTriple(val0, val1, val2, theta, lh_ptn, df_ptn, ddf_ptn, block);
    lh_ptn = 1.0 / (abs(lh_ptn) + ptn_invar[ptn]);
    df_frac  = df_ptn  * lh_ptn;
    ddf_frac = ddf_ptn * lh_ptn;
    my_df  += df_frac  * ptn_freq[ptn];
    my_ddf += (ddf_frac - df_frac * df_frac) * ptn_freq[ptn];
}
// Reduction: all_df += horizontal_add(my_df);  all_ddf += horizontal_add(my_ddf);
```

**computeLikelihoodBufferSIMD** (Priority #3):
```cpp
for (ptn = ptn_lower; ptn < ptn_upper; ptn += VectorClass::size()) {
    for (i = 0; i < block; i++)
        theta[i] = plh_node[i] * plh_dad[i];   // element-wise multiply
}
```

**computeLikelihoodBranchSIMD** (Priority #4):
```cpp
for (ptn = ptn_lower; ptn < ptn_upper; ptn += VectorClass::size()) {
    dotProductVec(val, theta, lh_ptn);
    lh_ptn = abs(lh_ptn) + ptn_invar[ptn];
    tree_lh += log(lh_ptn) * ptn_freq[ptn];
}
```

### GPU PoC structure (poc-gpu-likelihood-calculation-main/)

```
├── main.cpp              — Reads alignment/tree → builds model → calculates likelihood
├── Params.h              — Configuration singleton
├── alignment/            — Pattern and sequence handling
├── model/                — Evolutionary models (JC, POISSON)
├── tree/                 — Tree structure + LikelihoodCalculator (post-order traversal)
└── helper/               — Matrix operations with GPU backends
    ├── MatrixOpCPU.h/cpp       — CPU fallback
    ├── MatrixOpCUDA.cu/cuh     — Direct CUDA kernels (K-specialized)
    ├── MatrixOpCuBLAS.h/cpp    — cuBLAS + async streams
    ├── MatrixOpOpenACC.h/cpp   — OpenACC
    ├── MatrixOpOpenMPGPU.h/cpp — OpenMP GPU offloading
    ├── MatrixKernels.cu/cuh    — Fused hadamard+scaling, warp-shuffle reductions
    └── TipLikelihoodKernel.cu  — Tip node one-hot encoding on GPU
```

## GPU kernel designs

### Priority 1: computeLikelihoodDervSIMD → HIP kernel

- One GPU thread per alignment pattern
- Each thread: `dotProductTriple` (3 dot products of size `block` against `theta`) → normalize → accumulate
- Warp-level reduction for `df`/`ddf` sums (AMD: `__shfl_down` via wavefront-64)
- `val0/val1/val2` (16 doubles for DNA) in LDS (shared memory); `theta_all` in global HBM
- Returns 2 doubles (`df`, `ddf`) to CPU

### Priority 2: computePartialLikelihoodSIMD → HIP kernel

- One thread per pattern per internal node
- Inner work: `dotProductDualVec` + `productVecMat` per rate category
- One kernel launch per node in post-order traversal
- `partial_lh` arrays stay resident on GPU (avoid PCIe round-trips)

### Priority 3: computeLikelihoodBufferSIMD → HIP kernel

- Trivial element-wise multiply kernel
- If `partial_lh` already on GPU (from Priority 2), zero data transfer
- Can be fused with the DervSIMD kernel launch

## Setonix build environment

### Loading modules

```bash
module load rocm           # HIP/ROCm compiler and runtime
module load cmake          # Build system
module load gcc/12.2.0     # Host compiler
```

### GPU PoC build (HIP)

CUDA kernels must be ported to HIP. Options:
1. **hipify-perl** — automated CUDA→HIP translation (`hipify-perl MatrixKernels.cu > MatrixKernels.hip`)
2. **Manual port** — replace `__global__`/`__shared__`/`cudaMalloc` with HIP equivalents (mostly identical API)
3. **HIP can compile CUDA** — `hipcc` can compile `.cu` files directly in many cases

```bash
cd poc-gpu-likelihood-calculation-main
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc -DUSE_HIP=ON
make -j$(nproc)
```

### IQ-TREE build (CPU baseline for oracle comparison)

```bash
cd ANUHPC-iqtree
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
# Binary: build/iqtree3
```

### Test commands

```bash
# CPU oracle — 1 thread, default pipeline
./iqtree3 -s medium_dna.phy -seed 1

# CPU oracle — GTR+G4 fixed model
./iqtree3 -s medium_dna.phy -m GTR+G4 -seed 1

# CPU oracle — multi-threaded
./iqtree3 -s medium_dna.phy -seed 1 -nt 4
```

### Setonix GPU job submission

```bash
#!/bin/bash
#SBATCH --job-name=iqtree-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=gpu_test_%j.log

module load rocm cmake gcc/12.2.0
cd $MYSCRATCH/Iqtree

# Build GPU PoC
cd poc-gpu-likelihood-calculation-main/build
cmake .. -DCMAKE_CXX_COMPILER=hipcc -DUSE_HIP=ON
make -j16

# Run tests
ctest --output-on-failure 2>&1 | tail -30
```

## Implementation plan

### Phase 1: Port GPU PoC CUDA kernels to HIP for Setonix
- [ ] Run hipify-perl on all .cu files; fix any template specialization breakage
- [ ] Add CMake HIP/ROCm build path (detect `hipcc`, set `USE_HIP`)
- [ ] Build and run existing PoC on Setonix GPU node
- [ ] Validate PoC output matches CPU for JC model

### Phase 2: GPU kernel for computeLikelihoodDervSIMD (highest impact)
- [ ] Write correctness test first (CPU oracle output at known inputs)
- [ ] Implement per-pattern HIP kernel with triple dot product
- [ ] Add wavefront-level reduction for df/ddf accumulation
- [ ] Validate against CPU output (tolerance ≤1e-10)
- [ ] Benchmark: measure speedup vs 1T and 8T CPU

### Phase 3: GPU kernel for computePartialLikelihoodSIMD
- [ ] Implement post-order node traversal with HIP kernel per node
- [ ] Handle TIP-TIP, TIP-INTERNAL, INTERNAL-INTERNAL cases
- [ ] Keep partial_lh arrays GPU-resident
- [ ] Validate against CPU output

### Phase 4: GPU kernels for Buffer/Branch (lower priority, quick wins)
- [ ] computeLikelihoodBufferSIMD — trivial element-wise multiply
- [ ] computeLikelihoodBranchSIMD — single dot product + log reduction
- [ ] Fuse with DervSIMD launch if partial_lh already on GPU

### Phase 5: Integration with IQ-TREE
- [ ] Add HIP build option to IQ-TREE CMakeLists.txt
- [ ] Create GPU dispatch layer (CPU fallback when no GPU)
- [ ] End-to-end test: full tree search with GPU kernels
- [ ] Benchmark on Setonix MI250X

## Known limitations of the GPU PoC

- Only JC and POISSON models (need GTR, HKY for real workloads)
- CUDA-only kernels (need HIP port for Setonix AMD GPUs)
- No integration with IQ-TREE tree search loop
- Test scripts are stubs (not implemented)
- No benchmark results documented

## Confirmed correct (do not re-investigate)

- (Nothing confirmed yet — project is starting)

## Failed approaches (do not re-attempt)

- (None yet — record failures here as they occur)

## File locations

| Item | Path |
|------|------|
| IQ-TREE source | `ANUHPC-iqtree/` |
| Core kernel | `ANUHPC-iqtree/tree/phylokernelnew.h` |
| GPU PoC | `poc-gpu-likelihood-calculation-main/` |
| CUDA kernels (need HIP port) | `poc-gpu-likelihood-calculation-main/helper/MatrixKernels.cu` |
| Profiling report | `ANUHPC-iqtree/PROFILING_REPORT.html` |
| This file | `CLAUDE.md` |
| Progress log | `CHANGELOG.md` |
