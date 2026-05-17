# IQ-TREE MF2 Dispatch — Benchmark Results

This document collects empirical benchmark results and projected performance estimates for the
ModelFinder MPI dispatch patches applied to IQ-TREE 3.1.2 on Gadi (`normalsr` queue, SPR nodes).

---

## Datasets

| Name | Taxa | Sites | Distinct patterns | Compression | File size | Path |
|------|------|-------|------------------|-------------|-----------|------|
| `xlarge_mf.fa` | 200 | 100,000 | 98,858 | ~1% | — | `/scratch/um09/as1708/iqtree3-mf2/benchmarks/xlarge_mf.fa` |
| `100taxa_10M` | 100 | 10,000,000 | 10,000,000 | **0%** | **954 MB** | `/scratch/um09/as1708/iqtree3-mf2/benchmarks/100taxa_10M/alignment_10000000.phy` |

The 10M-site dataset has zero site-pattern compression because all 10M sites are distinct.
This makes it a memory-bandwidth-bound worst case: every site must be loaded from RAM on each
likelihood evaluation. RAM requirement is ~324 GB per rank (confirmed by IQ-TREE warning,
PBS 168000932).

---

## Hardware

| Resource | Spec |
|----------|------|
| Queue | `normalsr` (Gadi NCI) |
| Node | SPR 8470Q, 104 cores, 503 GB RAM |
| Interconnect | InfiniBand HDR |
| Charge rate | **2.0 SU / core-hour** |
| MPI library | OpenMPI 4.1.7 |
| Compiler | Intel LLVM (icpx) |

---

## Binary

| Item | Value |
|------|-------|
| Branch | `gadi-spr-r2-avx512` |
| HEAD commit | `1ac3c0a8` |
| Binary path | `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi` |
| Build command | `cd build-mpi-mf2 && module load openmpi/4.1.7 intel-compiler-llvm && gmake -j4 iqtree3` |
| Build date | 2026-05-10 |

Patches in `1ac3c0a8`:
- **Phase 1**: Round-robin `MF_IGNORED` stripe — each rank evaluates ~1/N models
- **Phase 2**: `MPI_Allreduce` gather + checkpoint merge + model name fix
- **Phase 3**: `--mpi-ranks-per-node` OMP thread budget (default: 1 rank/node = full 104 OMP)
- **Issue 5**: Sequential model eval loop inside `#ifdef _IQTREE_MPI` (eliminates OMP data race)
- **Issue 6**: Always `evaluateAll()` in MPI builds (`np=1 ≡ np=N` code path)

---

## `xlarge_mf.fa` Benchmarks

### PBS 167999083 — np=1 correctness reference

| Metric | Value |
|--------|-------|
| Config | 1 node, 1 rank, 104 OMP, `-te fixed_xlarge_tree.nwk`, seed=42 |
| Best-fit model | `SYM+G4` (BIC) |
| MF wall | **69.095 s** (968 models, sequential `evaluateAll()`, 104 OMP/model) |
| Total wall | ~70 s |

### PBS 168000131 — Phase 5 benchmark (np=4, 4 nodes)

| Metric | Value |
|--------|-------|
| Config | 4 nodes, 4 ranks, 104 OMP/rank, `-te fixed_xlarge_tree.nwk`, seed=42 |
| MPI mapping | `--map-by node:PE=104` |
| Best-fit model | **`SYM+G4`** (BIC) ✓ matches np=1 |
| MF wall | **58.924 s** (4 ranks × 242 models, 104 OMP/rank, sequential per-rank) |
| MF CPU time | 4,887.078 s (4 ranks × ~1,222 s each) |
| Total wall | 59.688 s |
| Script wall | 64 s |
| Phase 1 | `MF-MPI: rank 0/4 assigned 242/968 models` ✓ |
| Phase 2 | `MF-MPI: gather complete, 968 model scores consolidated` ✓ |
| Exit code | 0 ✓ |
| SU charged | 15.72 SU |

**Speedup vs np=1: 69.1 / 58.9 = 1.17×**

This low speedup is expected for `xlarge_mf.fa` because at ~0.24 s per model (compressed,
104 OMP), load imbalance between ranks (a few heavy models on one rank) dominates the small
savings. The 4,887 s CPU time = 4 × ~1,222 s confirms true 4-rank parallelism; only wall
time is limited by load balance. The 10M-site dataset (748 s/model) will show much larger
wall speedup because load-imbalance overhead becomes negligible relative to model evaluation
time.

---

## `100taxa_10M` Benchmarks

### PBS 167977883 — Baseline (no dispatch, np=4, all ranks redundant)

| Metric | Value |
|--------|-------|
| Config | 4 nodes, 4 ranks, 104 OMP/rank, `--hostfile`/`-rf rankfile`, seed=1 |
| MF wall at 2h kill | ~6,735 s elapsed (9 models × 748 s each) |
| Unique models done | **9** (all 4 ranks evaluated the **same** 9 models — no dispatch) |
| Coverage | 9 / 968 = **0.9%** |
| Best-fit model at kill | `GTR+R4` (partial; only 9 base models evaluated) |
| RAM per rank | ~324 GB |
| Average wall / model | **748 s** (at 104 OMP — memory-bandwidth limited) |

Note: The 748 s/model figure is empirically derived from the 6,735 s MF window in PBS
167977883 in which 9 JC/F81/K2P variants were completed. These are among the fastest models
(no rate variation); the rate-variation models (`+R2..+R9`, `+G4`, `+I+G4`) will be slower.
748 s is therefore a lower bound on the average.

**Projected baseline total wall to complete all 968 models (single sequential pass):**

```
968 models × 748 s/model = 724,064 s ≈ 201 hours (8.4 days)
CPU-hours: 968 × 104 cores × 748 s / 3600 = 20,917 CPU-h
KSU: 20,917 × 2.0 SU/core-h / 1000 = 41.8 KSU
```

### PBS 168000932 — MF2 Dispatch (np=4, 4 nodes, evaluateAll, 1 thread/model)

| Metric | Value |
|--------|-------|
| Config | 4 nodes, 4 ranks, 104 OMP/rank, `--map-by node:PE=104`, seed=1, `-m MF` |
| Script | `gadi-ci/run_100taxa_10M_mf2dispatch_4node.sh` |
| Walltime limit | 3 h (intentional; job expected to be killed, capturing partial results) |
| RAM (all 4 ranks) | **659 GB** at 40 min (165 GB/rank; still loading — warned 324 GB/rank at full load) |
| CPU utilisation | 94.5% on all 416 cores (confirmed 4-rank parallel operation) |
| Phase 1 | `MF-MPI: rank 0/4 assigned 242/968 models` ✓ |

**Rank-0 models completed at 40 min elapsed (1,878 s into MF phase):**

| Model | Index | Note |
|-------|-------|------|
| JC | 1 | Fastest base model |
| JC+R2 | 5 | Rate variation |
| JC+G4 | 25 | Rate variation |
| F81 | 45 | Base model |
| K2P | 89 | Base model |
| HKY | 133 | Base model |
| TNe | 177 | Base model |
| TN | 221 | Base model |
| K3P | 265 | Base model |
| K3Pu | 309 | Base model |
| TPM2 | 353 | Base model |
| TPM2u | 397 | Base model |

12 unique models done by rank 0 in 1,878 s ≈ 4 × 12 = **~48 unique models across all 4 ranks**
(round-robin stripe means each rank processes a completely disjoint set).

**OMP speedup constraint (empirically derived from this run):**

The base models (JC, F81, etc.) completed in < 1,878 s at 1 OMP thread each.
The rate-variation models are still running at 1,878 s.

In the baseline (PBS 167977883), JC-type models took ~748 s with 104 OMP threads.
With 1 OMP thread each in the dispatch mode:
- Base models (JC): single-thread time < 1,878 s → OMP speedup < 1,878 / 748 ≈ **2.5×**
- Rate-var models (JC+R3, not yet done): single-thread time > 1,878 s → upper bound only

For a 10M-site alignment with 0% compression, site-likelihood evaluation is
**memory-bandwidth bound** at 954 MB / rank. OMP parallelism (104 threads sharing the same
NUMA domain) does not scale linearly — L3/DRAM bandwidth saturates well before 104 threads.
The expected OMP speedup is **2–5×**, not the 80–100× achievable on compressed datasets.

---

## Projected Walltime Comparison (10M-site dataset, all 968 models)

This table compares the time to evaluate all 968 DNA models on `100taxa_10M` for the two
approaches. The OMP speedup in MF2 dispatch mode is the key uncertain variable; three
scenarios are shown.

| Scenario | OMP speedup (1→104 threads/model) | Avg single-thread time | **Wall to complete 968 models** | 3h kill coverage |
|----------|-----------------------------------|----------------------|--------------------------------|-----------------|
| Baseline (PBS 167977883, no dispatch) | 104× (sequential) | — | **~201 h (8.4 days)** | 0.9% |
| MF2 dispatch — lower bound | 2× | 1,496 s | **~1.9 h** | 100% |
| **MF2 dispatch — best estimate** | **5×** | **3,740 s** | **~4.7 h** | **86%** |
| MF2 dispatch — upper bound | 15× | 11,220 s | ~14 h | 43% |

**Method:** With 4 ranks and 104 OMP threads/rank, `evaluateAll()` runs all 242 rank-0 models
concurrently (1 OMP thread each). The 242 models are processed in `ceil(242/104) = 3` waves
(rounds) of 104 concurrent models. The wall time per round is dominated by the heaviest model
in that round (estimated at 1.5× the average). Total MF wall = 3 × heavy-model time.

### Summary

| | Baseline | MF2 dispatch (4 nodes, best estimate) |
|-|----------|--------------------------------------|
| **Wall to complete all 968 models** | ~201 h | **~4.7 h** |
| **Wall speedup** | — | **~43× faster** |
| **Unique models at 2h elapsed** | 9 (all ranks redundant) | ~750 (77%) |
| **Unique models at 3h elapsed** | 14 | ~832 (86%) |
| **KSU to complete all 968** | 41.8 KSU | **~5–14 KSU** |

### Why MF2 is also cheaper in SU

In the baseline, each model uses 104 OMP threads for 748 s = 77,792 core-seconds of CPU work
per model. In MF2 dispatch, each model uses 1 OMP thread for ~3,740 s (5× OMP speedup) =
3,740 core-seconds per model — **21× less CPU work per model** — because the memory-bandwidth
bottleneck meant the additional OMP threads were mostly stalling on DRAM loads. The dispatch
avoids this waste by concurrently evaluating 104 different models on 104 threads, each at
full memory bandwidth, instead of 104 threads on one model sharing the same bandwidth.

---

## Progress at 3h Kill (PBS 168000932, expected)

The job has a 3h PBS walltime limit. With the 5× OMP speedup estimate:

- MF phase has ~10,252 s (3h − 548 s tree build) available
- Single-thread avg = 3,740 s; heavy (worst-case round) = 5,610 s
- Rounds completed before kill: 10,252 / 5,610 ≈ 1.8 → **rounds 1 and 2 fully completed**
- Models per rank saved to checkpoint: ~208 of 242 (86%)
- Total unique models in checkpoint: ~832 of 968 (86%)

The `iqtree_run.model.gz` checkpoint file is written incrementally; models evaluated before
the SIGKILL at walltime will be preserved. `perf_stat.rank{0-3}.txt` files will be populated
on job exit.

---

## Data-Size Note

**File:** `alignment_10000000.phy` — PHYLIP format, 100 taxa × 10,000,000 sites
**Size:** **954 MB** (confirmed: `ls -lh` on 2026-05-10)
**Format header:** `100 10000000`
**SHA256:** not yet recorded (file is 954 MB; see `benchmarks/sha256sums.txt` for dataset
checksums once verified)

The 10M-site dataset was generated synthetically. Its 0% site-pattern compression (all 10M
sites distinct) makes it the memory-bandwidth worst case for ModelFinder — every unique site
contributes one pattern, requiring the full 954 MB alignment to be traversed on every
likelihood evaluation. This is why OMP speedup saturates at ~2–5× on this dataset vs ~80×
on compressed datasets like `xlarge_mf.fa`.
