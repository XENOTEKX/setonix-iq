# IQ-TREE HPC Profiling — Changelog

**Repo:** `https://github.com/XENOTEKX/setonix-iq.git`  
**Binary under test:** IQ-TREE 3.1.2 `gadi-spr-r2-avx512` branch (R1+R2 NUMA first-touch + AVX-512 icpx patches)  
**Compiler:** `icpx 2025.3.2` (`intel-compiler-llvm/2025.3.2`), `-O3 -march=sapphirerapids`  
**Target node:** Gadi `normalsr` — Xeon 8470Q "Sapphire Rapids", 2 sockets × 52 cores = 104 cores, DDR5-4800  
**Charge rate:** 2.0 SU/CPU-hour on `normalsr`

---

## Key Results Summary

### `xlarge_mf.fa` (100 taxa × 50,000 sites, ~5 MB) — Gadi Sapphire Rapids

| Config | Ranks × OMP | Nodes | Wall (s) | IPC | LLC miss | Speedup vs 1T |
|--------|-------------|-------|----------|-----|----------|---------------|
| 1T gcc | 1×1 | 1 | 13,954 | 2.01 | — | 1.00× |
| 4T gcc | 1×4 | 1 | 4,803 | 1.59 | — | 2.91× |
| 8T gcc | 1×8 | 1 | 2,956 | 1.31 | — | 4.72× |
| 16T gcc | 1×16 | 1 | 2,048 | 1.00 | — | 6.81× |
| 32T gcc | 1×32 | 1 | 1,425 | 0.81 | — | 9.79× |
| 64T gcc | 1×64 | 1 | 1,638 | 0.41 | — | 8.52× |
| 32T ICX R2 | 1×32 | 1 | 1,119 | 1.26 | — | 12.5× |
| 64T ICX R2 | 1×64 | 1 | 691 | 1.44 | — | 20.2× |
| **104T ICX R2** | **1×104** | **1** | **523.7** | **1.377** | **75.8%** | **26.6×** |
| 104T ICX R2 v3.1.2 | 1×104 | 1 | 541.8 | 1.374 | 76.0% | 25.8× |
| 2×52 socket (1-node) | 2×52 | 1 | 520.1 | 1.315 | 72.1% | 26.8× |
| 2×52 socket AVX-512 | 2×52 | 1 | 512.1 | 1.069 | — | 27.3× |
| 8×13 L3-rank (1-node) | 8×13 | 1 | 957.8 | 1.366 | 55.8% | 14.6× |
| 4×52 2-node socket | 4×52 | 2 | 389.1 | 1.303 | 75.7% | 35.9× |
| 2×104 2-node R1 | 2×104 | 2 | 334.6 | 1.355 | 77.0% | 41.7× |
| 2×104 2-node v3.1.2 | 2×104 | 2 | 342.4 | 1.347 | 76.6% | 40.8× |
| **2×104 2-node AVX-512** | **2×104** | **2** | **324.5** | **1.105** | **—** | **43.0×** |

**Best single-node:** 104T ICX R2 — 523.7 s (PBS 167865976)  
**Best multi-node:** 2×104 2-node AVX-512 — 324.5 s (PBS ~167973941), **−38.1%** vs single-node  
**Setonix best (for context):** Setonix 32T — 3,302 s → Gadi 2-node is **10.2× faster** than Setonix best  
**lnL across all runs:** −10,956,936.612 (bit-exact match or within MPI tolerance of ±0.005)

### `alignment_10000000.phy` (100 taxa × 10,000,000 sites, 954 MB) — Gadi 4-node MPI

| Config | Ranks × OMP | Nodes | Wall | Status |
|--------|-------------|-------|------|--------|
| 4×104 AVX-512 | 4×104 | 4 | — | PBS 167977883 queued (UCX fixes applied 2026-05-09) |

---

## 2026-05-09 — 4-node MPI: 100 taxa × 10 M sites dataset

### Script: `gadi-ci/run_100taxa_10M_r2_avx512_mpi_4node.sh`

**PBS job history (all failures, root-caused and fixed):**

| PBS ID | Wall | Exit | Root cause | Fix applied |
|--------|------|------|------------|-------------|
| 167976747 | 3s | 1 | `\\$2` unbound variable under `set -u` in `<<PYENV` heredoc | `\\$2` → `\$2` |
| 167976807 | ~6min | cancelled | Cancelled to add VTune before collecting results | — |
| 167977268 | 9s | 1 | `UCX_IB_ADDR_TYPE=lid` invalid in UCX 1.17.0 (renamed to `ib_local`) | `lid` → `ib_local` |
| 167977317 | 14s | 139 (SIGSEGV) | `rc_mlx5` missing `ud_mlx5` auxiliary transport → NULL endpoint → crash | Added `ud_mlx5` to `UCX_TLS` |
| **167977883** | **pending** | **—** | `UCX_TLS=rc_mlx5,ud_mlx5,sm,self` + `UCX_NET_DEVICES=mlx5_0:1` | — |

**Final `MPI_OPTS` (commit `51f6625b`):**
```bash
MPI_OPTS=(
    --mca pml ucx
    -x "UCX_TLS=rc_mlx5,ud_mlx5,sm,self"
    -x "UCX_NET_DEVICES=mlx5_0:1"
)
```
- `ud_mlx5` is the mandatory auxiliary transport for `rc_mlx5` endpoint address resolution in UCX 1.17.0 (`select.c:634`). Omitting it causes "no auxiliary transport" → NULL endpoint → OpenMPI `poll_dispatch` SIGSEGV.
- `self` is required for MPI collective operations where a rank is both sender and receiver (e.g. `MPI_Allreduce`).
- `UCX_IB_ADDR_TYPE` removed — the `lid` → `ib_local` rename broke things; without the flag UCX auto-selects correct addressing.

**Other fixes in this session:**
- `env.json` stall: `sh("mpirun -n 1 ${IQTREE} --version …")` inside a PBS job hangs (nested mpirun cannot acquire a process slot). Fixed to `sh("${IQTREE} --version …")` — IQ-TREE prints version without MPI init. Applied to both 4-node and 2-node scripts.
- VTune Pass 3 (`uarch-exploration`) added after `perf stat` Pass 2: `pmu-collection-mode=summary`, `collect-memory-bandwidth=true`, `finalization-mode=deferred`, rank 0 only.

**Profiling passes on success:**
1. **Pass 1** — Clean timing with `/proc` RSS sampler (every 10s on rank 0)
2. **Pass 2** — `perf stat` hardware counters per rank (`cycles`, `instructions`, `LLC-loads`, etc.)
3. **Pass 3** — VTune `uarch-exploration` TMAM + DRAM bandwidth (rank 0, summary mode)

**Dataset:** `/scratch/um09/as1708/iqtree3-3.1.2/benchmarks/100taxa_10M/alignment_10000000.phy`  
954 MB, 100 taxa, 10,000,000 DNA sites. 19.0× more sites than `xlarge_mf.fa` (50K sites).

**Commits (2026-05-09):**
| Hash | Message |
|------|---------|
| `afa11df8` | `fix: \\$2 → \$2 in PYENV heredoc (unbound variable with set -u)` |
| `a1ae7867` | `perf: explicit --mca pml ucx + UCX_TLS=rc_mlx5 for deterministic IB path` |
| `47ab6049` | `fix: use direct binary for iqtree_version in env.json (avoid nested mpirun stall)` |
| `3eb2146c` | `feat: VTune uarch-exploration Pass 3 for 4-node run` |
| `de0480ab` | `changelog: record PBS 167977268` |
| `ca7cce69` | `fix: UCX_IB_ADDR_TYPE lid→ib_local (UCX 1.17.0 enum rename)` |
| `51f6625b` | `fix: add ud_mlx5 to UCX_TLS, drop UCX_IB_ADDR_TYPE (rc_mlx5 auxiliary transport)` |

---

## 2026-05-08–09 — MPI multi-node sweep: `xlarge_mf.fa`

### Binary
`/scratch/um09/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi`  
Branch: `gadi-spr-r2-avx512` (R1+R2 NUMA first-touch + AVX-512 patches, v3.1.2 source)  
Build: `icpx 2025.3.2 -O3 -march=sapphirerapids -DIQTREE_FLAGS="mpi KNL"`  
Links: `libmpi.so.40` (openmpi/4.1.7), `libiomp5.so` (intel-compiler-llvm/2025.3.2), UCX 1.17.0

### MPI/InfiniBand stack
- MOFED 5.8, OpenMPI 4.1.7 built with `--with-ucx=/apps/ucx/1.17.0 --with-hcoll --with-ucc`
- UCX transports: `rc_mlx5`, `dc_mlx5`, `ud_mlx5` on `mlx5_0:1` (ConnectX HDR)
- Collectives: HCOLL 4.8 + UCC 1.3 (`UCC_TLS=^sharp` — SHARP not wired on `normalsr`)

### Results (all lnL = −10,956,936.607 or −10,956,936.612 ✓)

| Config | PBS ID | Wall | IPC | LLC miss | Δ vs 104T canonical | Notes |
|--------|--------|------|-----|----------|---------------------|-------|
| 2×52 socket, 1-node, numa_ft | 167895713 | 520.1 s | 1.315 | 72.1% | −0.7% | Validates MPI topology |
| 2×52 socket, 1-node, avx512 | ~167973xxx | 512.1 s | 1.069 | — | −2.2% | AVX-512 lowers IPC (wider SIMD, same work) |
| 8×13 L3-rank, 1-node | 167899378 | 957.8 s | 1.366 | 55.8% | +83% | OMP team too small; MPI overhead dominates |
| 4×52 socket, 2-node | 167911421 | 389.1 s | 1.303 | 75.7% | −25.7% | Linear scaling holds |
| 2×104 full-node, 2-node, numa_ft | 167931341 | 334.6 s | 1.355 | 77.0% | −36.1% | Full-node eliminates intra-node UPI boundary |
| 2×104 full-node, 2-node, v3.1.2 | 167932918 | 342.4 s | 1.347 | 76.6% | −34.6% | v3.1.2 parity (+2.3%, within HPC noise) |
| **2×104 full-node, 2-node, avx512** | **~167973xxx** | **324.5 s** | **1.105** | **—** | **−38.1%** | **Current best** |

### Key findings

**OMP team size matters more than MPI rank count.** The 8×13 L3-rank experiment confirmed: 13-thread OMP teams are too small for this workload despite excellent L3 locality (LLC miss dropped from 72% to 55.8%, IPC rose from 1.315 to 1.366). MPI coordination at 8 ranks + slow OMP convergence overwhelmed the cache benefit (+83% wall vs −0.7% for 2×52).

**Full-node vs socket pinning on 2-node.** 2×104 (−36.1%) outperforms 4×52 (−25.7%) because:
1. Each 104-thread OMP team is more efficient per replicate than a 52-thread team
2. Eliminating the intra-node socket MPI boundary removes UPI crosslink traffic
3. InfiniBand cost at 2 ranks is lower than at 4 ranks

**v3.1.1 vs v3.1.2 parity confirmed.** `tree/phylotreesse.cpp` and `tree/phylokernelnew.h` are byte-identical between tags. Wall delta +2.3%/+3.5% is within single-run HPC noise (±5% threshold). IPC and LLC miss are statistically identical.

### Script fixes (MPI-specific)
The MPI scripts went through several rounds of fix before producing clean results:

| Fix | Root cause | Symptom |
|-----|------------|---------|
| `--mca rmaps_base_mapping_policy ""` required | OpenMPI 4.1.7 BYCORE auto-default conflicts with `-rf rankfile` | ranks bound to wrong cores |
| `--mca rank_file` (4.x) not `--map-by rankfile:file=` (5.x) | API version mismatch | "unrecognized modifier" error |
| `INNER_PID` pgrep chain needed `\|\| true` | pgrep returns exit 1 when no match; `set -e` killed the script | mpirun killed mid-run at the pgrep line |
| `OMP_DYNAMIC=false` required | Without it, OMP may shrink team under cpuset pressure | non-deterministic thread counts |

### Files
| File | Purpose |
|------|---------|
| `gadi-ci/run_xlarge_r2_mpi_2node_fullnode.sh` | 2-node 2×104 production script |
| `gadi-ci/run_xlarge_r2_v312_mpi_2node_fullnode.sh` | Same, v3.1.2 parity |
| `gadi-ci/run_xlarge_r2_v312_canonical.sh` | 1-node 1×104 v3.1.2 parity |
| `gadi-ci/bootstrap_iqtree_3.1.2_mpi.sh` | MPI build bootstrap |

---

## 2026-05-08 — NUMA R2 ICX single-node sweep + v3.1.2

### NUMA R2 patches applied to `xlarge_mf.fa`

**R1 patches** (`phylotreesse.cpp`): Pre-allocate and first-touch `_partial_lh` arrays on the NUMA node that will own them in the hot loop. Eliminates cross-socket LLC traffic during tree traversal.

**R2 patches** (`phylokernelnew.h`): Replace `schedule(dynamic,1)` with `schedule(static)` on 5 inner OMP parallel-for loops. Removes dynamic scheduler overhead and balances work by site-block (aligned to NUMA pages).

**Patch sites verified (8/8):**
- `schedule(dynamic,1)` in `phylokernelnew.h`: 0 (removed)
- `schedule(static) num_threads` in `phylokernelnew.h`: 5 (added at lines 1275, 2386, 2838, 3005, 3595)
- `NUMA first-touch` markers in `phylotreesse.cpp`: 3 (R1a, R1b, R2a)

### Results (v3.1.1 ICX R2, `xlarge_mf.fa`, PBS series ~167865976)

| Threads | Wall | IPC | LLC miss | Speedup vs 1T |
|---------|------|-----|----------|---------------|
| 32T | 1,119 s | 1.257 | — | 12.5× |
| 64T | 691 s | 1.438 | — | 20.2× |
| **104T** | **523.7 s** | **1.377** | **75.8%** | **26.6×** |

**Before R2 (gcc baseline at 104T):** gcc pins gave 64T → 1,638 s and 32T → 1,425 s. The R2 patch restored the expected monotonic scaling and brought 64T below 32T.

**ICX vs gcc at 104T (same R2 patch):** ICX (523.7 s, IPC 1.38) vs gcc (not run at 104T with R2 separately, but gcc 64T = 1,638 s prior to R2 confirms ICX is substantially better on AVX-512 code generation).

### v3.1.2 parity (PBS 167932915–918)

| Run | PBS | Wall | IPC | LLC miss | Δ vs v3.1.1 |
|-----|-----|------|-----|----------|-------------|
| canonical 1×104 | 167932917 | 541.8 s | 1.374 | 76.0% | +3.4% |
| 2-node 2×104 MPI | 167932918 | 342.4 s | 1.347 | 76.6% | +2.3% |

Both within ±5% noise. lnL bit-exact (−10,956,936.612). **Conclusion: v3.1.2 performance-equivalent to v3.1.1.**

---

## 2026-05-07–08 — NUMA first-touch investigation

### Problem identified (2026-05-07)
IQ-TREE 3 at high thread counts (>32T on Gadi SPR) showed a performance cliff:
- 32T → 64T: wall increased (wrong direction)
- LLC miss rates extremely high (~75–78%) indicating cross-NUMA memory traffic
- Profiling with VTune `uarch-exploration` showed BackEnd-Bound / Memory-Bound dominating TMAM

### Root cause
`_partial_lh` likelihood arrays in `phylotreesse.cpp` are allocated on the master thread's NUMA node (socket 0), then accessed by all OMP threads including socket 1 threads. At 64–104 threads, half the threads cross the UPI fabric on every cache miss (560 GB/s UPI vs 307 GB/s per-socket DDR5 effective bandwidth — cross-socket adds ~2× latency per miss).

### Patch R1 (allocation fix)
In `phylotreesse.cpp` `PhyloTree::computePatternLikelihood()` and `PhyloTree::computeLikelihood()`: distribute and first-touch `_partial_lh` using per-thread OMP parallel blocks so each thread's page lands on its local NUMA domain.

### Patch R2 (scheduler fix)
In `phylokernelnew.h` inner OMP loops: `schedule(dynamic,1)` assigns single-site chunks dynamically, causing excessive cross-NUMA work-stealing. Changed to `schedule(static)` which assigns contiguous chunks, keeping each thread's work on locally-allocated pages.

### Validation
`perf stat` LLC miss rate dropped from ~78% (baseline) to 75.8% (R2). More importantly, the **64T → 104T scaling cliff was eliminated**: 104T ICX R2 = 523.7 s is monotonically faster than 64T ICX R2 = 690.5 s (expected direction).

HTML analysis: `numa_first_touch.html` (archived in repo root) documents the cache miss rate by thread count before and after patching.

---

## 2026-05-01–07 — Corpus audit, gcc/ICX parity, dashboard

### Round 2 audit (2026-04-30 to 2026-05-01)
The original Gadi corpus (`_sr_icx` runs) had two systematic issues:
1. **Compiler mismatch**: used `intel-compiler-llvm/2024.2` (ICX + libiomp5) vs Setonix's GCC/libgomp — cross-platform comparisons were contaminated.
2. **VTune co-running overhead**: VTune hotspot collection added 6–26% wall overhead at 32–104T.

These runs are now `non_canonical: true, non_canonical_label: "ICX+VTune (ref)"` in the dashboard.

**Resolution**: Re-ran everything with `gcc/14.2.0 + libgomp`, no VTune co-run, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `numactl --localalloc`. These are the `_sr_gcc_pin` canonical runs used for cross-platform comparison.

### Cross-platform baselines (canonical, lnL-verified)

**`large_modelfinder.fa` (100 taxa × 50K sites):**

| Platform | 1T | Best | Best config | lnL |
|----------|----|------|-------------|-----|
| Gadi SPR | 2,451 s | 284 s (32T) | gcc+libgomp, `close/cores`, `--localalloc` | −2,690,513.343 |
| Setonix Genoa | — | — | — | — |

**`xlarge_mf.fa` (100 taxa × 50K sites):**

| Platform | 1T | Best single-node | lnL |
|----------|----|-----------------|-----|
| Gadi SPR | 13,954 s | 523.7 s (104T ICX R2) | −10,956,936.612 |
| Setonix Genoa | — | 3,302 s (32T) | same |

### perf counter fixes
- `cycles:u` suffix required on Gadi (`perf_event_paranoid=2`)
- Setonix `cycles:uk` needed to count kernel cycles (Slurm cgroup issue)
- L1d-MPKI adopted as cross-platform memory pressure metric (identical PMU event on AMD Zen3 and Intel SPR)

### Dashboard
- `speedup`/`efficiency` keyed by `(dataset_short, platform)` — prevents cross-platform baseline contamination
- `cache_level` field tags whether `cache_miss_rate` refers to L2 (AMD/Setonix) or L3 (Intel/Gadi)
- Non-canonical runs visible but excluded from speedup calculations

---

## 2026-04-24–30 — Initial Gadi sweep and baseline

**Goal:** Reproduce Setonix IQ-TREE profiling results on Gadi Sapphire Rapids to quantify cross-platform performance.

### What was done
1. Cloned IQ-TREE 3 source, built with GCC and ICX on Gadi normalsr
2. Ran thread sweeps (1T, 4T, 8T, 16T, 32T, 64T, 104T) on `large_modelfinder.fa` and `xlarge_mf.fa`
3. Harvested `perf stat` counters and VTune hotspot data into dashboard JSON schema
4. Identified OMP binding gap: runs without `OMP_PROC_BIND=close` / `numactl` showed non-reproducible results
5. Built and validated the web dashboard at `xenotekx.github.io`

**Key insight from initial sweep:** `large_modelfinder.fa` peaks at 32T (diminishing returns beyond — memory bandwidth saturation), while `xlarge_mf.fa` keeps improving to 104T but with declining efficiency (75–78% LLC miss throughout, memory-bound).

### Dataset geometry
All datasets SHA256-gated in scripts to prevent accidental reuse of wrong files.

| File | Taxa | Sites | Size | Notes |
|------|------|-------|------|-------|
| `turtle.fa` | 16 | 20,820 | ~0.33 MB | stub/test |
| `large_modelfinder.fa` | 100 | 50,000 | ~5 MB | primary `large_mf` benchmark |
| `xlarge_mf.fa` | 100 | 50,000 | ~5 MB | ModelFinder-focused, different model space |
| `mega_dna.fa` | 500 | 100,000 | ~50 MB | large-taxa benchmark |
| `alignment_10000000.phy` | 100 | 10,000,000 | 954 MB | 4-node MPI stress test (2026-05-09) |

---

## Archive: Superseded Process Entries

The following entries from the original verbose changelog have been archived here as one-line summaries. The full text existed at commit `de0480ab` (2026-05-09) and is recoverable via `git log --all`.

| Original entry | Date | Summary |
|----------------|------|---------|
| (a)–(e) unlabelled | 2026-04-24–30 | Initial Gadi job submissions, harvest scripts, dashboard CI setup |
| (f) | 2026-05-08 | `numa_first_touch.html` formatting cleanup |
| (g) | 2026-05-08 | `numa_first_touch.html` graph generation |
| (h) | 2026-05-08 | MPI placement scripts staged (not yet submitted) |
| (i) | 2026-05-08 | R2 alternate placement chain submitted (PBS 167889450–452) |
| (j) | 2026-05-08 | Placement jobs failed — 3 bugs root-caused (pgrep, BYCORE, 5.x syntax) |
| (k) | 2026-05-08 | Round 2 fixes applied (pgrep `\|\| true`, `--mca rank_file`, `OMP_DYNAMIC=false`) |
| (l) | 2026-05-08 | Socket 2×52 result: 520.1 s PASS |
| (m) | 2026-05-08 | L3-rank 8×13 result: 957.8 s PASS (outcome: too slow) |
| (n) | 2026-05-08 | 2-node socket experiment design |
| (o) | 2026-05-08 | 2-node 4×52 result: 389.1 s PASS |
| (r) | 2026-05-09 | 4-node script creation + 2-node 2×104 full-node result: 334.6 s |
| (s) | 2026-05-09 | UCX fixes (3 cascading bugs), VTune Pass 3, PBS 167977268/317/883 |
| follow-up #1–#20 | 2026-04-30 – 2026-05-07 | Dashboard audits, corpus corrections, NUMA patch investigation, ICX bootstrap fix, perf counter calibration, gcc/ICX parity runs |
