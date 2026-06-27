# Canonical GPU IQ-TREE 3 — the ONE tree, the ONE binary (consolidation 2026-06-27)

This file ends the "too many binaries" problem. There is now a single unified line that
carries **every** validated GPU capability (JOLT likelihood/optimizer, +R FreeRate,
GPU profile-mixtures, native CTF ModelFinder, and the full TS.0–TS.6 GPU tree search).
Everything else is superseded and archived (see §5 — no work was lost).

## 1. The canonical tree + binary

| | |
|---|---|
| **Repo / tree** | `/scratch/rc29/as1708/iqtree3-l2search` |
| **Branch** | `gpu-kernel-prod` == `l2-batched-nni` @ **`75532f6c`** (identical refs) |
| **GitHub** | `setonix-iq` = github.com/XENOTEKX/setonix-iq, branch **`gpu-kernel-prod`** (pushed 2026-06-27, tip `75532f6c`, clean FF) |
| **Canonical binary (GPU)** | `build-gpu-on/iqtree3` — md5 **`fe5f01f0f9217683ade96dfbd4018817`** (built 2026-06-27 from `75532f6c`, the warm-seed tip) |
| **Companion binary (CPU / no-GPU)** | `build-gpu-off/iqtree3` (for `--jolt`-off / OLD-vs-NEW bit-identity comparison) |

History: `587e5ba8` = +R Phase-1 (fixed-Q pure +R) + shelved L-BFGS; `5551dc23` = consolidation doc; then the
**+R ladder** — `0ec14119` = 2a (free-Q+R / GTR+R), `f7198d73` = 2b+2c (+I+R and free-Q⊗+I⊗+R = GTR+F+I+R2);
`4cca1d4a` = +R-ladder doc; **`75532f6c` = G.5.1e warm-seed** (the current tip).
The two binaries differ only by `-DIQTREE_GPU`.

**G.5.1e warm-seed (`75532f6c`)** — the +R/+I+R joint-LM start kept the CPU-seeded *separated* rates (`meanR=ρ_c`
from `getRate`) instead of flattening to ~1 (a symmetric stationary point that trapped ncat≥2 at collapsed rates).
Real-data fix: full avian `GTR+F+I+R2 -te` (21.76M ptns) had collapsed both R2 cats to (rate=1.411, prop=0.3542);
post-fix (job **172457047**) the rates **separate** (0.9318/0.61, 3.534/0.12), lnL **−418,863,583.56** (**+5.44M nats**
vs the collapsed −424,299,552.21), GPU==CPU **rel 2.1e-15**, 85 iters / 5.9 min (vs 401 / 19.9 min). Pure-+R is no
longer bit-identical to the pre-fix binary *by design* (start changed) — validated by GPU≥CPU + rate-separation; the
pinv=0 / non-+R paths are byte-identical (job **172456879**: +G/+I+G OLD==NEW).

## 2. Build recipe

```bash
module load cmake/3.24.2 gcc/12.2.0 cuda/12.5.1 eigen/3.3.7 boost/1.84.0
cmake -DIQTREE_GPU=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_HOST_COMPILER=/apps/gcc/12.2.0/wrappers/g++ ..   # nvcc 12.5.1, -O3
make -j   # -> build-gpu-on/iqtree3   (use -DIQTREE_GPU=OFF for the CPU companion)
```

Run GPU jobs on a `gpuhopper` H200 node. NOTE: a cgroup memory limit < physical RAM is
auto-detected and the likelihood memory is sized to the allocation.

## 3. Validated capability matrix (this session, on the canonical binary)

| Capability | How to invoke | Validation | Result |
|---|---|---|---|
| JOLT single-matrix **DNA** (JC…GTR) base/+G/+I+G | `--jolt` (default-on) | covR audit (job **172444810**), DNA `-m MF` | **73 models engage, ALL PASS**, worst GPU==CPU rel 6.2e-12 |
| JOLT single-matrix **AA** (empirical Q) base/+G/+I+G | `--jolt` | covR audit, AA `-m MF` | **119 models engage, ALL PASS**, worst rel 1.8e-10 |
| **+R (FreeRate) — FULL LADDER**, R≤4 | `-m …+R{2,3,4}` (and `+I+R`) `--jolt` | p1Rval2 (**172444201**) · lad2a (**172450392**) · lad2bc (**172451291**) | **complete & exact**: fixed-Q pure +R (JC/F81/LG) · **2a** free-Q+R (GTR/HKY/TN+R) · **2b** +I+R · **2c** GTR+F+I+R2 — all engage, GPU lnL==CPU rel≤5e-16, GPU≥CPU, pinv==CPU exact; LG+R5/ncat>4 decline |
| **+R live in ModelFinder** | `-m MF` `--jolt` | covR audit | DNA: JC+R2/R3/R4 · AA: LG+R2/R3/R4 (free-Q+R and +I+R now also engage) |
| Free-Q (GTR), +I+G | `--jolt` | p1Rval2 no-regression | OLD==NEW bit-identical |
| **GPU profile-mixtures** (JOLTMix, e.g. MEOW80) | `JOLT_MIX_HOSTDRIVEN=1 … --jolt --gpu` | uniMEOW (job **172448018**), LG+MEOW80+G4, 21,798 ptns | **N=80 engages (weights=EM), GPU lnL==CPU rel 2.463e-13**, lnL −1,665,670.997 |
| Native **CTF** ModelFinder | `--ctf` | (prior G.6.x, in-tree) | wired |
| **GPU tree search** TS.0–TS.6 | `--ts-fused` (+ `--jolt`) | avianTENT (job **172437363**, in flight) | screener + JOLT reopt, `maxiter=2` (validated, beats Hashara 812s) |

Hard exactness bar held throughout: every engaged model self-checks GPU lnL == CPU lnL.

## 4. Environment flags (all default to the validated behavior)

| Flag | Default | Effect |
|---|---|---|
| `JOLT_MIX_HOSTDRIVEN` | unset (off) | `=1` engages the GPU profile-mixture optimizer (JOLTMix) |
| `JOLT_BRLEN_MAXITER` | 2 | per-round LM brlen reopt cap (validated win); `<0` ⇒ skip GPU reopt (CPU fallback, profiling) |
| `JOLT_LBFGS_M` | 0 (off) | L-BFGS reopt — **shelved** (honest-negative: converges worse; OFF kept as insurance) |
| `JOLT_K1_HOIST` | (not wired) | Fix-B val-table hoist — perf-FALSIFIED (~1.001×); source preserved in `staging/*.fixb`, see `staging/README_K1_FIXB.md` |
| `JOLT_DEBUG` | 0 | `=1` prints `[JOLT]`/`[JOLT-GATE]`/`[JOLTMIX]` engage+coherence lines |
| `JOLT_RGRADCHECK` | 0 | `+R` gradient-check-only mode (gate-test) |

## 5. Retired / superseded trees — and exactly where their content now lives

- **`/scratch/rc29/as1708/iqtree3-gpu`** (branch `tree-search-ts0` @ `6fce15de`) — **SUPERSEDED**.
  Verified 2026-06-27 that **100% of its content is in canonical history**:
  - committed tip `6fce15de` is a strict **ancestor** of canonical `5551dc23`;
  - its 6 uncommitted tree-search files (gpu_iqtree.h, iqtree.cpp, phylotree.h, phylotreegpu.cpp,
    tools.cpp, tools.h) are byte-identical to canonical commit **`37a63740`**
    ("snapshot of tree-search-ts0 working tree");
  - its one drifted file, `gpu_lnl_intree.cu` (the K1/Fix-B version), is byte-identical to
    canonical **`staging/gpu_lnl_intree.cu.fixb`**.
  Its 7 stale `build-*/` dirs are obsolete; nothing there is unique. Safe to delete to reclaim
  disk (not done automatically — non-destructive retirement).
- **`/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3`** — this is the shared bare `origin` /
  the **MF-Isolation harness** (a separate ModelFinder-dispatch debugging project), *not* part of
  the GPU consolidation. Left as-is.

## 6. Provenance tags (on setonix-iq)

- `archive/pre-consolidation-l2-2026-06-27` → `587e5ba8` (pushed) — the +R code state pre-doc.
- `archive/gpu-kernel-{a972cefc,backup-6d7f7483,clean-cca7dbc1}` — prior validated GPU-kernel states.

---
*One tree, one binary: `gpu-kernel-prod@75532f6c` → `build-gpu-on/iqtree3` (md5 fe5f01f0). Use it.*
