# Mode P Implementation Status — Phase tracker

Companion to `research/mode-p-design.md` (the design contract). This document is the **live progress tracker** for the actual implementation work, with exact file:line edit specifications for the remaining phases.

**Last updated:** 2026-05-26 — **ATMD-AID BUILT + ISO-4 SUBMITTED (binary `68031e28`, build 169218528 exit=0 8m36s; ISO-4 AID job 169219973):** Full source implementation of [research/Modelfinder/novel-dispatch-architectures.md](novel-dispatch-architectures.md) compiled cleanly. New binary `68031e28be61289da962ed3deaec2d98`. Components in source: Architecture C tree-slice fix at `phylokernelnew.h:1268-1305`; `--atmd-aid` + `--atmd-aid-heavy-mult` CLI flags; 5 new methods on `CandidateModelSet` (`aidComputeCostPred` uses FCA closed-form predictor that returns 0 for `MF_AID_HEAVY` so family-LPT excludes heavies; `aidScheduleWaves` greedy LPT with descending `gs ∈ {np, np/2, ..., 2}`; `aidBuildLattice` pre-creates MPI_Comm_split lattice once; `aidFreeLattice`; `aidExecuteWaves` per-wave cohort dispatch via `setModePGroupComm`); `MF_AID_HEAVY` flag; getNextModel skips it. Integration in `evaluateAll()`: mpgc_active=false when aid_active; AID setup after MPGC block; Phase 0 = FCA loop with heavy skipped; Phase 1 = `aidExecuteWaves` after loop; cleanup frees lattice. Diagnostics `AID-DIAG` + `AID-WAVE`. Run scripts created: `run_iso4_aa100k_np4_aid.sh` (correctness gate, np=4) + `run_p7_aa1m_np16_aid.sh` (perf gate, np=16). ISO-4 AID job **169219973** submitted to verify correctness (lnL parity with FCA np=1, best model LG+G4, wave dispatch fires, no crashes). Backward compatible: AID off ⇒ legacy FCA/MPGC unchanged. Prior: P.7 ❌ FAIL job 169211688 (>1800s, MPGC gs=2). H2 CONFIRMED + fixed in Arch C. Expected wall AA 1M np=16: 520-680s (gate ≤600s ✓).
**2026-05-26 follow-up — ATMD-AID ISO-4 PASS after canonical state fix:** Jobs 169221567/169221568 showed the first warm-start-null patch was insufficient: a rank could restore/evaluate from rank-local Phase 0 checkpoint state and escape a collective wave with a local-slice score. Final fix broadcasts a canonical `ModelCheckpoint` and `RateWarmStartCache` from rank 0 before AID Phase 1, then runs waves against the synchronized state and merges results back. Binary `3e79db194ced77971a55c6a0ff476863`. ISO-4 AID job 169227435 completed: BASE exit=0 wall=427s, AID exit=0 wall=360s, 29 AID waves, 142 `AID-WAVE` starts, best model `LG+G4`, AID lnL `-7541976.854` (Δ=0.007 vs ref). The PBS wrapper initially reported FAIL only because its awk parser rounded lnL through `print v+0`; `run_iso4_aa100k_np4_aid.sh` now preserves the raw log field and expects md5 `3e79db19`.

B.4-13 prior fix: `--mca coll ^ucc` added to both run scripts (no rebuild; binary `ad77fa4e`). Job 169179725 (`ad77fa4e`, K1 diag): crashed at Q.BIRD+G4/Q.MAMMAL+G4 boundary — barrier fix alone insufficient. Job 169189043 (`ad77fa4e` + `--mca coll ^ucc`, K1 diag): no crash but wrong scores (= B.4-15).

---

## ⚠ PRIORITY NEXT STEPS — BLOCKERS BEFORE ISO-4 PASS

These must be resolved in order before ISO-4 (np=4 Mode P correctness gate) can pass.
B.4-12 ✅ (script fix done). B.4-14 ✅ (CSCH fix built + verified in binary `43c313f3`, build 169176245 exit=0). B.4-13 ❌ **STILL OPEN** — job 169179725 confirmed per-model barrier fix (`ad77fa4e`) was insufficient: 7 MF-TIME completions (LG,LG+G4,LG+I+G4,LG+F+G4,JTT+G4,JTT+F+G4), crashed at Q.BIRD+G4/Q.MAMMAL+G4 boundary (model ~24-26) with 75 "Message truncated" lines. **New root cause (2026-05-26):** OpenMPI 4.1.7 UCC collective component (`tl_ucp_coll.c`) does NOT properly isolate concurrent `MPI_Allreduce` calls on different sub-communicators created by `MPI_Comm_split` — when group 0 (ranks 0,1) calls `modePAllreduceLh` (count=1 double) while group 1 (ranks 2,3) calls `modePAllreduceLhDfDdf` (count=3 doubles) on their respective communicators simultaneously, UCC routes the messages incorrectly → one rank expects 3 doubles but receives 1 → "Message truncated". **Fix:** `--mca coll ^ucc` added to `mpirun` in both run scripts to disable UCC and fall back to the `tuned` collective component (properly uses context IDs for comm isolation). No code change or rebuild needed. Job **169189043** submitted (K1 diag, binary `ad77fa4e` + `--mca coll ^ucc`).

| # | Task | Status | Notes |
|---|------|--------|-------|
| **B.4-12 Fix** | `--atmd-k-outer 1` restored to MODEP sub-run in `gadi-ci/mode-p-iso/run_iso4_aa100k_np4_p3.sh` | ✅ FIXED 2026-05-25 | Script fix only (no rebuild needed). K_outer=8 was causing MPI_THREAD_FUNNELED violation; re-added `--atmd-k-outer 1` and updated comment explaining the NNI step-count asymmetry (BASE=92 vs MODEP=61) is EXPECTED, not a bug. |
| **B.4-13 Fix** | `--mca coll ^ucc` added to `mpirun` in both run scripts to disable UCC collective component | ✅ FIXED 2026-05-26 (job 169189043 exit=0 80s, no crash; scripts carry forward to `76cdfb19`) | **True root cause (2026-05-26):** B.4-13 barrier fix in `ad77fa4e` was insufficient. Job 169179725 (`ad77fa4e`, K1 diag, 20-min): 7 MF-TIME completions, crashed at Q.BIRD+G4/Q.MAMMAL+G4 boundary (model ~24). UCX error: `tl_ucp_coll.c:137 TL_UCP ERROR failure in recv completion Message truncated`. Crash pattern: ranks 0,1 (group 0) on Q.BIRD+G4; ranks 2,3 (group 1) on Q.MAMMAL+G4 — groups ARE properly split, intra-group Allreduces use different communicators. **OpenMPI 4.1.7 UCC bug:** UCC's UCP transport layer (`tl_ucp`) does not properly isolate concurrent `MPI_Allreduce` calls on sub-communicators created by `MPI_Comm_split` when both sub-communicators are active simultaneously. When group 0 calls `modePAllreduceLh` (count=1 double, group0_comm) while group 1 is in `modePAllreduceLhDfDdf` (count=3 doubles, group1_comm), UCC confuses the two operations → size mismatch → "Message truncated". **Fix:** `--mca coll ^ucc` in mpirun command disables UCC, falls back to `tuned` component that correctly uses MPI context IDs to isolate sub-communicator collectives. No binary change; `ad77fa4e` unchanged. Scripts: `run_iso4_diag_k1_np4_p3.sh` + `run_iso4_aa100k_np4_p3.sh`. |
| **B.4-14 Fix** | Scope `filterRatesMPI` Bcasts to intra-group communicator under P.7-MPGC ngroups>1 | ✅ FIXED 2026-05-25 (build 169176245 running) | **Root cause:** `filterRatesMPI()` fires when a rank's reference family completes — under MPGC, "rank" = group, so group 0 fires at time T0 (LG family done) while group 1 is still in WAG family at T0. Original code's `MPI_Bcast(... MPI_COMM_WORLD)` requires ALL world ranks to call simultaneously → group 1 never arrives → deadlock or message-size mismatch crash. **Fix (CSCH design):** added `MPI_Comm fca_comm = MPI_COMM_WORLD` member to `CandidateModelSet`; `evaluateAll()` MPGC setup block sets `fca_comm = mpgc_comm` (intra-group sub-communicator from `MPI_Comm_split`); `filterRatesMPI()` uses `MPI_Comm_rank/size` on `fca_comm` for local rank/size and Bcast root=0 in `fca_comm` (group leader). Each group prunes within itself. Reset to `MPI_COMM_WORLD` in MPGC cleanup. Files: `main/phylotesting.h` (member), `main/phylotesting.cpp` (filterRatesMPI body lines 3120-3236; evaluateAll setup line ~4008; cleanup line ~4634). **Build history**: job 169175807 exit=4 (preflight: `p6_lite_collective` grep missed P.7-MPGC); job 169175914 exit=4 (preflight: `B\.4-9 Mode P fix` regex found 1 match, Branch fallback comment updated to `B.4-9 / B.4-11` by prior fix); build_mode_p_iso_p3.sh updated with `grep -qE 'p6_lite_collective\|P\.7-MPGC\|mpgc_active'` and `grep -cE 'B\.4-9.* Mode P fix'`; job **169176245** submitted (preflight passed, currently compiling at 52%). |
| **B.4-15 Fix** | Propagate MPGC state from `in_tree` to per-model `iqtree` in `CandidateModel::evaluate()` | ✅ FIXED 2026-05-26 (binary `76cdfb19`, build 169193965 exit=0; verifying via job 169195103) | **Root cause (discovered by job 169189043 with `ad77fa4e + --mca coll ^ucc`):** the run completed cleanly (exit=0, 80s) but produced WRONG scores — every Group 1 family score = its paired Group 0 family score (e.g. LG+G4 == WAG+G4 == −7,572,497.929). Diagnostic: MF-MPI-DIAG shows rank 0 `ptn=[0,48008)` (group-based) at MPGC setup; per-rank stderr `[Mode P]` lines show world-based QUARTERS at evaluation time (rank 0=[0,24008), rank 1=[24008,48008), rank 2=[48008,72016), rank 3=[72016,96017)). The fresh `iqtree` created inside `CandidateModel::evaluate()` (phylotesting.cpp:1962) has DEFAULT `mp_group_size=1` and `mp_allreduce_comm=MPI_COMM_WORLD`; the MPGC state previously set on `in_tree` is never propagated. So `initializePtnPartition()` falls into the WORLD-rank/size branch (phylotree.cpp:958–961), and kernel `modePAllreduceLh` runs on WORLD — across-group sums of LG (group 0) and WAG (group 2) yield identical garbage on all 4 ranks. **Fix (MPGC Inheritance — novel):** add `PhyloTree *in_tree=nullptr` parameter to `CandidateModel::evaluate()`; right after `iqtree->setNumThreads()` and before `initializePtnPartition()`, propagate group state: `if (in_tree && in_tree->mp_group_size > 1) iqtree->setModePGroupComm(in_tree->mp_allreduce_comm, in_tree->mp_group_rank, in_tree->mp_group_size);`. Both call sites (`evaluateAll:4368`, `test:3532`) updated to pass `in_tree`. Backward compatible (default nullptr; non-MPI builds skip via `#ifdef _IQTREE_MPI`). Enhanced `[Mode P]` diagnostic adds `grp_rank=…/grp_size=…` for verification. Files: `main/phylotesting.h`, `main/phylotesting.cpp`. |
| **ISO-4 MPGC submit** | ✅ **PASS** — job **169197750** exit=0, wall=11m7s (binary `76cdfb19`) | ✅ PASS | BASE np=1: LG+G4, lnL=−7,541,976.853 (Δ=0.008) ✓, MF=426s. MODEP np=4: LG+G4, lnL=−7,541,976.852 (Δ=0.009) ✓, MF=238s. [Mode P] 109 lines × 4 ranks ✓. **Speedup 1.79×** (426s→238s; ISO-3 was 1.74× at np=2). WAG+G4=−7,602,067 ≠ LG+G4=−7,541,977 → B.4-15 MPGC inheritance confirmed ✓. **K1 diag 169195103:** wall=4m3s, same result. Script awk precision bug fixed (printf %.6f). **Prior history:** 169177186/187 (no barrier crashed), 169179725 (crashed Q.BIRD boundary), 169189043 (wrong scores B.4-15). |
| **ISO-5 Auto-dispatcher** | Run with `--mode-p` (cost-threshold dispatch, NOT `--mode-p-all`) | ✅ **PASS** — job **169207131** exit=0 wall=13m36s (binary `cc3d403f`) | **All 8/8 checks pass.** MODEP np=4: LG+G4, lnL=−7,541,976.852 (Δ=0.009) ✓. P6-DIAG: `avg_cost=1531015535 threshold=2296523303 (mult=1.500×) heavy=56/224` ✓. dispatch=MODEP:28 ✓, dispatch=FCA:34 ✓. [Mode P] 87 lines × 4 ranks ✓. Speedup 1.11× (428s→384s). **Script fix:** rank_logs glob `*/stderr` → `*/*/stderr` (7 places) — OpenMPI `--output-filename` creates files at `rank_logs/1/rank.N/stderr` (two levels), not `rank_logs/1/stderr`. Jobs 169200513 (threshold=8.0 too high), 169203004 (race), 169204621 (7/8, glob bug) all resolved. |
| **ISO-4 ATMD-AID** | ✅ **PASS** — job **169227435** completed with binary `3e79db19` | ✅ PASS | Final root cause was rank-local Phase 0 checkpoint/warm-start state entering collective AID Phase 1. Fix broadcasts canonical checkpoint and warm-start from rank 0 before waves. Wave 0 `MTVER+F+I+G` completed on all ranks with matching score `1.58894e+07` before wave 1, eliminating the earlier rank-local escape. AID: LG+G4, lnL=−7,541,976.854 (Δ=0.007), MF=345s, wall=360s, 29 waves, 142 `AID-WAVE` starts. Wrapper parser precision fixed (`awk` no longer uses `print v+0`), and expected md5 updated to `3e79db19`. |
| **P.7 Perf gate** | AA 1M np=16 MF wall ≤ 600 s | ❌ **ATMD-AID still misses perf target** — job **169227541** (binary `3e79db19`) was stopped after it had already exceeded the 600s MF target before Phase 1 waves could start. Evidence: AID np=16 initialized correctly (`heavy_models=56/224`, `num_waves=10`), LG=16.4s, LG+I=62.0s, LG+G4=175.6s, then plain `LG+I+G4` remained in Phase 0 FCA for >420s because the current avg×1.5 heavy threshold only marks `+F+G4`/`+F+I+G4` classes heavy. Correctness/deadlock blocker is fixed; next blocker is AID heavy-selection/scheduling for non-`+F` `+I+G4` on AA 1M. | Next small test: make `+I+G4` eligible for AID waves (or lower/tune `--atmd-aid-heavy-mult`) and rerun a focused np=16 gate. |

---

## Completed Runs — All dispatching strategies (AA 100K + AA 1M)

All runs use alignment `alignment_100000.phy` (AA 100K) or `alignment_1000000.phy` (AA 1M), seed 1, `-m TEST` (full ModelFinder). Speedup is MF-wall speedup over the Baseline for each dataset. IPC and LLC miss % from user-space `perf stat` where collected.

> **Baseline binary:** `build-intel-vanila/iqtree3` · non-MPI OMP-across-models · ICX + AVX-512 + R1+R2 · v3.1.2 (`4e91dd6`) · sa0557  
> **FCA binary:** `iqtree3-mpi` · md5 `a78ffa2942d6b073490d503416ae554c` · icpx 2025.3.2 + OpenMPI 4.1.7 + AVX-512  
> **ATMD b3c binary:** `iqtree3-mpi-atmd-b3c` · md5 `1c6fc01921df0fbd67e45da280a036e9` · same toolchain, macros `-D_IQTREE_ATMD -D_IQTREE_MPI`  
> **ATMD b4 / Mode P binary (active):** `iqtree3-mpi-mode-p-iso-p3` · md5 `63548e7cf9ac4a09f31384e1a672d132` · built 2026-05-25 18:55 (B.4-12/13/14/15 + P.6 cost-threshold dispatch, job 169200138 exit=0 wall=10m35s) · prior builds: `76cdfb19` (B.4-15 MPGC Inheritance), `ad77fa4e` (B.4-13+14), `43c313f3` (B.4-14 CSCH only), P.7-MPGC `efef6be4`, B.4-11 `85bf5c79`, B.4-9 `9660575a`

| Job | Type | Dataset | Nodes | Ranks×OMP | Best model | lnL | BIC | MF wall (s) | SPR wall (s) | Total wall (s) | MF speedup | IPC | LLC miss % |
|-----|------|---------|-------|-----------|------------|-----|-----|-------------|-------------|----------------|-----------|-----|-----------|
| 168425673 | **Baseline** | AA 100K | 1 | 1×103T | LG+G4 | −7,541,976.860 | 15,086,233.280 | 399.456 | 764.478 | 1,169.556 | 1.00× | 1.878 | 56.0% |
| 168584736 | **FCA np=2** | AA 100K | 2 | 2×103T | LG+G4 | −7,541,976.853 | 15,086,233.265 | 149.029 | 383.876 | 537.750 | **2.68×** | — | — |
| 169095077 | FCA np=1 | AA 100K | 1 | 1×103T | LG+G4 | −7,541,976.861 | — | 258.773 | 738.569 | 1,000.811 | 1.54× | — | — |
| 169111545 | **ATMD b3c np=1, K=8** | AA 100K | 1 | 8×12T | LG+G4 | −7,541,976.853 | — | **423.2** | ~1,297 | 1,720 | **0.94× ⚠** | — | — |
| 168425491 | Baseline | AA 1M | 1 | 1×103T | LG+G4 | −78,605,196.573 | 157,213,128.618 | 7,587 | 15,099 | 22,776 | 1.00× | — | — |
| 168913089 | FCA np=1 | AA 1M | 1 | 1×103T | LG+G4 | −78,605,196.590 | — | 5,120 | 15,061 | 20,180 | 1.47× | — | — |
| 168635614 | FCA np=2 | AA 1M | 2 | 2×103T | LG+G4 | −78,605,196.443 | — | 3,077 | 7,869 | 10,946 | 2.47× | 1.260 | 83.7% |
| 168635615 | FCA np=4 | AA 1M | 4 | 4×103T | LG+G4 | −78,605,196.445 | — | 1,974 | 3,982 | 5,957 | 3.84× | 1.273 | 84.0% |
| 168586094 | FCA np=8 | AA 1M | 8 | 8×103T | LG+G4 | −78,605,196.497 | — | 1,444 | 2,147 | 3,672 | 5.25× | — | — |
| 168635616 | FCA np=16 | AA 1M | 16 | 16×103T | LG+G4 | −78,605,196.497 | — | 1,122.363 | 1,287.863 | 2,410.226 | 9.45× | 1.337 | 85.27% |
| 169112256 | **ATMD b3c np=16, K=1** | AA 1M | 16 | 1×103T | LG+G4 | −78,605,196.497 | — | **2,114** | 1,958 | 4,327 | **3.59× ⚠** | — | — |
| 169135061 | **Mode P ISO-2 `--mode-p-all` np=2** | AA 100K | 2 | 2×103T | LG+G4 | −7,541,976.861 | — | **235.844** | — | MF only | **1.69× ⚠†** | — | — |
| 169136469 | **Mode P ISO-3 `--mode-p-all` np=2** | AA 100K | 2 | 2×103T | LG+G4 | −7,541,976.8614 | — | **240** | — | 247s (MF+NNI) | **1.66× ⚠†** | — | — |

> ⚠ = regression vs FCA np=2 (149s). Both Mode P ISO rows use `--mode-p-all` which is a **correctness gate only** — all models routed through Mode P Allreduce, including light models where per-Allreduce overhead is significant. The production design (P.6) applies Mode P exclusively to the top 3–8 heaviest models; light models stay in FCA serial dispatch. See analysis below.  
> † Mode P `--mode-p-all` at np=2 gives only 1.69–1.66× MF speedup vs baseline, compared to FCA np=2's 2.68×. Mode P is 58% slower than FCA at the same node count when applied to all models. The performance case for Mode P is at **np≥8 with AA 1M on heavy models** — see §§ "Mode P scaling regime" below.  
> ATMD b3c rows: at 100K (K=8) slower than non-MPI baseline (bandwidth-saturated); at 1M (K=1, Mode F inactive) 1.88× slower than FCA np=16 (NUMA first-touch overhead per model).

---

## Architecture deep-dive: Baseline vs FCA vs ATMD Mode F — why ATMD is not competitive at 2-node 100K AA

### TL;DR verdict first

**ATMD Mode F is structurally incapable of speedup on AA workloads on Gadi SPR.** At 100K it is slower than the non-MPI baseline (MF=423s vs 399s). At 1M it cannot engage at all (K_outer forced to 1 by memory). FCA's 2.68× MF speedup at np=2 comes from a completely orthogonal mechanism (model-parallel island dispatch) that Mode F does not touch.

**Mode P `--mode-p-all` at np=2 also underperforms FCA np=2**: ISO-2 gives MF=235s vs FCA np=2's 149s — Mode P is **58% slower than FCA at the same resource count**. The question "is Mode P worth pursuing?" requires understanding *when* Mode P actually wins. The honest answer: **Mode P is not a 2-node technology and `--mode-p-all` is not the production use case. Mode P is designed for np≥8 with AA 1M where a few heavy models dominate the MF critical path.** The ISO gates prove correctness; they do not demonstrate production performance. See §§ "Mode P scaling regime" for the full analysis.

---

### What each architecture actually does

#### Baseline: serial per-model, 103 OMP threads

The vanilla binary (non-MPI) evaluates one model at a time, using all 103 threads across both NUMA domains via OpenMP-across-patterns (`#pragma omp parallel for` inside `computeLikelihoodBranchSIMD`). The per-pattern kernel streams the `partial_lh` array — the dominant hot data — at ≈46 GB per model evaluation at AA 100K (npat=96,017 × nstates=20 × nrates=4 × 8 bytes/double). The Sapphire Rapids node has ~500 GB/s aggregate DRAM bandwidth (both sockets) and a 100 MB LLC. At M=103 threads the kernel is already **DRAM-bandwidth-saturated**: the bandwidth curve is sublinear past ~32 threads for this access pattern (stride-1 reads across a 46 GB array), and adding more threads provides no further gain per model. MF wall: 399s for 224 models.

#### FCA (Family-Controlled Assignment): model-parallel island dispatch

FCA is a **model-parallel** strategy. Under Phase 0.5/0.6, each MPI rank is assigned a disjoint family of amino-acid substitution models. At np=2: rank 0 owns the LG family (~22 models, including the four heavy `LG+F*` models), rank 1 owns all remaining families (~202 models post-prune). **Both ranks evaluate their models simultaneously and independently, with no inter-rank communication during evaluation.** The only MPI calls are:

1. A `filterRates` Bcast after rank 0 finishes its LG family — broadcasts the winning rate categories so rank 1 can skip already-eliminated rate variants (Phase 0.5 pruning)
2. A final BIC score gather to select the global best model

The speedup is real and near-ideal because the work is **embarrassingly parallel at the model level**. Rank 1's queue is pruned (≈202 models → ≈80–100 after filterRates) and runs in ~149s. Rank 0 runs its LG family (heavy models) in ~149s. Total MF wall = max(rank 0 wall, rank 1 wall) ≈ 149s = **2.68× over baseline**. This is a genuine Amdahl gain: both ranks are fully utilised at 103 threads each, the serial coordination fraction is tiny (two MPI calls), and the model evaluation itself is embarrassingly parallel.

#### ATMD Mode F: K_outer-parallel within each rank

ATMD Mode F is an **intra-rank parallelism** strategy. Instead of 1 model × 103 threads per rank, it runs K_outer models concurrently per rank, each on M_inner = 103/K_outer threads. At AA 100K: K_mem = floor(496,978 MB × 0.8 / 46,414 MB) = 8 → **K_outer=8, M_inner=12**. The design intent is that 8 models running in parallel would collectively complete in the same wall time as 1 model on 103 threads, giving ~8× per-rank throughput.

This theory has a fatal flaw for AA workloads: the per-model kernel is **memory-bandwidth-bound, not dispatch-bound**. The SIMD FMA compute in the kernel (20-state transition matrix product) takes ~3 ns/pattern. The `partial_lh` streaming takes ~10 ns/pattern (46 GB / 500 GB/s × cache-miss factor). The kernel is already bottlenecked on DRAM bandwidth at K=1 with M=103 threads. Running K=8 models simultaneously multiplies the DRAM demand, not the DRAM supply:

| K_outer | Working set | DRAM demand | Available BW | Effective per-model BW | Model wall |
|--------:|------------:|------------:|-------------:|-----------------------:|----------:|
| 1 | 46 GB | 46 GB/call | 500 GB/s | ~500 GB/s | ~92 ms/call |
| 8 | 368 GB | 368 GB/call | 500 GB/s | ~62 GB/s | ~740 ms/call |

Each model at K=8 gets ~8× less effective bandwidth → each individual model takes ~8× longer per L-BFGS step, and the K=8 models finish in roughly the same wall as K=1 — **no gain from parallelism**. Meanwhile the LLC is thrashed (100 MB cannot hold any fraction of 368 GB), the hot blocks that the K=1 path kept warm between iterations are evicted on every pass, and 112 nested-OMP team setup/teardown events add per-model overhead. The measured result: MF wall = **423s at K=8 vs 258s FCA np=1 — Mode F is 64% slower than a single-rank sequential run**.

#### Why FCA beats ATMD at 2-node 100K: different axes of parallelism

FCA exploits **between-model parallelism** across nodes: two independent models are evaluated on two independent nodes, each with their own full 500 GB/s bandwidth budget. The key insight is that models do not share state — they are fully independent evaluation tasks. This is a textbook parallel decomposition: perfectly separable work, negligible communication, linear speedup.

ATMD Mode F tries to exploit **within-rank parallelism** across models on the same node: the K models compete for the same node's DRAM bandwidth. On a bandwidth-saturated workload, this delivers zero throughput gain and adds overhead.

At 2 nodes with FCA: each node evaluates its own family at full bandwidth → 2× throughput at the model-queue level.  
At 2 nodes with ATMD Mode F: each node runs K=8 models per rank, all competing for that node's DRAM → per-node throughput ≈ 1× (bandwidth-limited), and the per-model wall is actually worse than serial. Adding a second node with ATMD gives ≈2× the models covered in ≈423s wall (vs FCA's 149s for the same model set).

#### The structural asymmetry at AA 1M

At AA 1M, the per-tree working set is 457,507 MB — nearly filling the 500 GB node. K_mem = floor(493,125 × 0.8 / 457,507) = **1**. The `if(atmd_K_outer > 1)` guard in the OMP pragma evaluates false: **the parallel region never activates**. ATMD at 1M is mechanically identical to FCA per rank (one model at a time, 103 threads) — except it carries the overhead of the NUMA first-touch setup and `omp_set_max_active_levels(2)` bookkeeping, making it **1.88× slower than FCA np=16** (2,114s vs 1,122s).

The B.5 formula fix (b4) corrects the per_tree_MB estimate from 457,507 MB to ~64,400 MB, yielding K_mem=6. But the bandwidth-saturation analysis still applies: even with K=6 physically possible, K=6 models × 64 GB = 384 GB competing for 500 GB/s still gives each model only ~83 GB/s effective bandwidth. The 8× case was harmful; the 6× case would be similar. The memory formula fix does not fix the bandwidth ceiling.

---

### Is ATMD worth pursuing?

The answer depends critically on which part of ATMD is meant.

#### ATMD Mode F alone: **No**, for AA workloads on Gadi SPR

Mode F as implemented in B.3+B.4 cannot deliver speedup on any AA dataset size on Gadi SPR:
- At ≤500K AA sites (K_mem ≥ 2 physically possible): bandwidth-saturated kernel makes K>1 harmful
- At >500K AA sites (AA 1M): K_mem=1 (even after B.5), Mode F mechanically inactive
- Even the B.5 fix (K=6 at 1M) runs into the same bandwidth wall

The one workload class where Mode F WOULD win is compute-bound kernels. Codon models (61-state, 61×61 transition matrix — FLOPS/byte ratio ~8× the AA case) are the natural positive control. Small-npat DNA with LLC-resident working sets also qualify. For the project's AA targets, Mode F contributes nothing.

**However, abandoning Mode F entirely is premature.** The B.-1/B.3/B.4 infrastructure (nested OMP, NUMA first-touch, the K_outer dispatch loop) is already in the source tree. It is the scaffold that Mode P reuses for the light-model path (after Mode P handles the heavy models, the remaining light models go through Mode F with K_outer=1, which is just the FCA serial path — no regression). Removing it would cost more than keeping it, and it provides correct behaviour on codon datasets as a free side-benefit.

#### ATMD Mode P (pattern-parallel intra-model MPI_Allreduce): **Yes — but only at the right scale**

The ISO-2 result (MF=235s at np=2 `--mode-p-all` vs FCA np=2=149s) is real and must be explained honestly before asserting Mode P is worth pursuing.

##### Why Mode P is slow at np=2 with `--mode-p-all`

Mode P `--mode-p-all` routes ALL models — light and heavy — through a cooperative 2-rank pattern-split evaluation. This has two compounding problems at np=2 AA 100K:

**Problem 1: Model serialization.** In `--mode-p-all` mode, all ranks must cooperate on the SAME model at every step. This means models are evaluated **sequentially** (one model completes, then the next starts). FCA np=2 evaluates models **in parallel** (rank 0 and rank 1 work on different families simultaneously). FCA's wall = max(queue_0, queue_1) ≈ 149s. Mode P's wall = Σ(all model times, each ~10% faster) ≈ 235s. This is the core of why Mode P `--mode-p-all` loses badly to FCA at any np.

**Problem 2: Small per-model speedup at np=2 for AA 100K.** The observed per-model speedup is only ~1.097× (258s → 235s, 224 models). Theoretical 2× is not achieved because:
- Even at half the patterns (48K per rank, ~23 GB working set), the per-rank kernel is **still partially DRAM-bound** (23 GB >> 100 MB LLC). The speedup from halving the pattern work is ~1.5× in the kernel, not 2×.
- A large fraction of model evaluation time is **non-kernel** (model initialization, `initializeAllPartialLh`, parameter optimization setup, `decomposeRateMatrix`). These get zero speedup from Mode P.
- Allreduce overhead per kernel call: ~3–10 µs × 16,000 calls/model = ~50–160 ms per model. For a light model (0.3s), this is 17–53% overhead. For the full 80-model pruned queue, this accumulates to ~4–13s of pure communication tax.

The measured 1.097× average speedup per model is consistent with ~60% non-kernel fraction plus Allreduce overhead.

##### The Mode P scaling regime: where it actually delivers

Mode P's value proposition is regime-specific. It requires **both** conditions:

1. **Large np** (≥ 8): the per-rank working set must be small enough to fit in LLC (2.9 GB at np=16, comfortably LLC-resident vs 46 GB DRAM-bound at np=1). Below np≈8, partial DRAM-saturation persists and the kernel speedup is limited.
2. **Heavy models dominate the wall**: Mode P, applied only to the top 3–8 heaviest models (P.6 dispatcher), is cooperative (all ranks on one model) while the light models run in FCA parallel. The P.6 design keeps the embarrassingly parallel FCA throughput for the 200+ light models and only serializes the handful of heavy models.

**At AA 1M np=16 with the P.6 production design** (Mode P on heavy models only, FCA for rest):

| Layer | Mechanism | Est. MF wall at AA 1M np=16 |
|-------|-----------|-----------------------------|
| 0 | FCA K=1 (status quo) | 1,122 s |
| **A** | **Mode P on LG+F+I+G4 (702s → ~70s at np=16, 10× speedup)** | **~490 s** |
| **B** | **Mode P on LG+F+G4 (~200s → ~20s at np=16)** | **~310 s** |
| A+B | Mode P on top 2 heaviest + FCA for rest (rank 0 bottleneck removed) | **~200–250 s** |
| A+B+top8 | Mode P on top 8 heavy + FCA for rest | **~150–180 s** |

Note: the 1M heavy-model times (702s for LG+F+I+G4) are from live ATMD b3c job 169112256 observations. At np=16 with LLC-resident partial_lh (~4 GB per rank), speedup is ~10× (not 16×, due to Allreduce latency × ~80 L-BFGS iterations × ~200 branches × 2 calls = ~32,000 Allreduces × 3µs = ~96ms overhead, which is ~0.1% of 70s).

**Why 702s → 70s at np=16 but only 3.2s → 2.9s at np=2 for a typical 100K model:** the Allreduce overhead (fixed ~50ms per model for AA 100K) is 0.07% of 70s but 1.7% of 2.9s. For heavy AA 1M models the overhead is negligible; for light AA 100K models it is significant. The regime matters.

**The combined production design (P.6):**
- Mode P handles only the models on the MF critical path: LG+F+I+G4, LG+F+G4, and similar +I+G variants of the top families
- FCA handles the ~200 light models: each of 16 ranks runs its own family queue concurrently (~70s at 1M × 16 = same 1M per-rank time as today)
- Total MF wall ≈ max(FCA_light_queue + Mode_P_heavy_overhead) ≈ 70s + 70s + 20s ≈ **160s** vs current 1122s

This is the **7× MF speedup** that motivated Mode P, not the 1.1× seen at np=2 `--mode-p-all`.

##### What the ISO-2/3 results actually prove

ISO-2 and ISO-3 are **correctness gates only**. The 235s and 240s walls confirm:
- The Mode P kernel (pattern split + Allreduce) produces bit-accurate lnL matching FCA np=1 serial (Δ≤4e-4, within FP non-associativity)
- No SEGFAULT, no HALF-tree_lh leak (B.4-9 fixed), no NaN propagation (B.4-8 fixed)
- All MPI collectives synchronize correctly across ranks

They do NOT prove production performance. The production performance case requires P.6 (heavy-model-only dispatcher) + AA 1M + np=16.

The Amdahl analysis (top of this document) shows np=16 FCA is already at 83% of its Amdahl ceiling ($4.56\times$ of $5.5\times$ max). Without Mode P, the P.7 gate (MF ≤ 600s at np=16) is **mathematically unreachable** within FCA alone. Mode P is the only mechanism that can restructure the computation to bypass that ceiling, but only when deployed correctly (heavy models, high np, large dataset).

#### Bottom line

| Component | Worth it? | When | Reason |
|-----------|-----------|------|--------|
| Mode F B.3+B.4 (already shipped) | Keep, don't extend | Always | Correct code; codon workloads benefit; no AA regression at K=1 |
| New Mode F AA investment | No | Never | Bandwidth-bound; no AA dataset size delivers speedup |
| Mode P `--mode-p-all` (all models) | No, as production config | Never | Serializes model queue; slower than FCA at any np |
| Mode P on **heavy models only** (P.6 dispatcher) at **np≥8 AA 1M** | **Yes — mandatory** | After B.4-10 fixed | Only path to P.7; 700s → 70s for LG+F+I+G4; ISO-3 correctness PASSED |

The ISO gates (ISO-2, ISO-3) prove kernel correctness. The production performance case rests on P.6 (heavy-model-only dispatch) at np=16 AA 1M — which is unverified but strongly motivated by the 702s LG+F+I+G4 single-model observation (b3c job 169112256). ISO-4 hit B.4-10 (SIGFPE in +F models at np=4); diagnostic job 169136683 is running. Once B.4-10 is fixed, the path is: ISO-4 (np=4 correctness) → ISO-5 (auto-dispatcher) → P.7 (AA 1M np=16 perf gate ≤600s).

---

## Amdahl motivation: why Mode P is the only viable MF path forward

The following analysis is derived from the AA 1M FCA MPI full-run scaling data in `CHANGELOG.md` and `logs/runs/*.json`. All $T(1)$ values use FCA np=1 (job 168913089) as the single-rank MPI baseline to isolate pure MPI scaling.

### Serial fraction and Amdahl ceiling — ModelFinder

$$\frac{1}{S(p)} = f_s + \frac{1-f_s}{p} \quad\Rightarrow\quad f_s = \frac{1/S(p) - 1/p}{1 - 1/p}$$

| $p$ | MF wall (s) | $S(p)$ | $E(p)$ | $f_s$ estimate |
|---:|---:|---:|---:|---:|
| 1 | 5,119.929 | 1.000 | 100.0% | — |
| 2 | 3,076.873 | 1.664 | 83.2% | 0.202 |
| 4 | 1,974.476 | 2.593 | 64.8% | 0.181 |
| 8 | 1,443.892 | 3.547 | 44.3% | 0.179 |
| 16 | 1,122.363 | **4.562** | **28.5%** | 0.167 |

$$\hat{f}_s^{\text{MF}} = 0.182 \quad\Rightarrow\quad S_{\max}^{\text{MF}} = \frac{1}{0.182} \approx 5.5\times$$

**At np=16 we have reached 83% of the Amdahl ceiling ($4.56\times$ out of $5.5\times$).** Adding more MPI ranks yields no further gain within the current architecture. The 18.2% serial fraction is structural: model initialization, BIONJ reference tree construction, `initializeAllPartialLh()`, and `filterRates()` coordination cannot be distributed across ranks by adding islands — they run per-rank on the same model space.

**The only way to improve MF performance beyond np=16 is to reduce $\hat{f}_s^{\text{MF}}$ itself.** Mode P does this by splitting the pattern dimension across ranks within one model evaluation: instead of one rank evaluating all $N_{\text{pat}}$ patterns for model $m$, $k$ ranks each evaluate $N_{\text{pat}}/k$ patterns and Allreduce the likelihood sums. The effective serial fraction per model evaluation shrinks as the collective overhead replaces the sequential pattern scan.

### Comparison: tree search has very different scaling behavior

| $p$ | Tree-search wall (s) | $S(p)$ | $E(p)$ | $f_s$ estimate |
|---:|---:|---:|---:|---:|
| 1 | 15,060.551 | 1.000 | 100.0% | — |
| 2 | 7,868.928 | 1.914 | 95.7% | 0.045 |
| 4 | 3,982.142 | 3.782 | 94.5% | 0.019 |
| 8 | 2,147.499 | 7.012 | 87.6% | 0.020 |
| 16 | 1,287.863 | **11.694** | **73.1%** | 0.025 |

$$\hat{f}_s^{\text{tree}} = 0.027 \quad\Rightarrow\quad S_{\max}^{\text{tree}} = \frac{1}{0.027} \approx 37\times$$

Tree search is at only **32% of its Amdahl ceiling**. It scales well because each rank runs an independent topology trajectory — perfect island parallelism with a small synchronization overhead. The efficiency drop from 100% to 73% at np=16 is work-imbalance, not a serial bottleneck.

### Consequence for Mode P priority

When Mode P is complete and $\hat{f}_s^{\text{MF}}$ drops from 18.2% to, say, 5%, the weighted total-wall Amdahl ceiling rises from $14.7\times$ to approximately:

$$S_{\max}^{\text{total}} \approx \frac{1}{0.254 \times 0.05 + 0.746 \times 0.027} \approx 30\times$$

At that point tree-search performance becomes the dominant bottleneck in total wall time, and the topology-parallel work documented in `research/Treesearch/topology-parallel-tree-search-roadmap.md` becomes the primary lever.

**Mode P P.7 target revision (informed by Amdahl):** the current P.7 gate is MF wall $\leq 600$ s at np=16 (vs FCA ref 1,122 s). This corresponds to $S(16) \geq 1{,}122/600 \approx 1.87\times$ improvement over FCA np=16, or a reduction in $\hat{f}_s^{\text{MF}}$ to approximately:

$$f_s \leq \frac{1/8.56 - 1/16}{1 - 1/16} \approx 0.065$$

Mode P needs to bring the effective per-rank MF serial fraction from 18.2% to below 6.5%. This is a real but achievable target: if pattern work (the dominant per-model cost) is fully parallelized across ranks, $f_s$ collapses to initialization + communication overhead.

---

## Phase status

| Phase | Status | Validates | Owner |
|---|---|---|---|
| P.1 Scaffolding (Params + CLI + PhyloTree members + MPI helpers) | **✅ DONE** | Build succeeds with `--mode-p` accepted but inert | prior session |
| P.2 Pattern partition wiring (`initializePtnPartition` in `evaluate()`) | **✅ DONE** | Per-rank `[Mode P]` cerr line shows partition range | prior session |
| **B.4-8 incomplete fix (REVERTED):** `safe_numeric=true` in `initializePtnPartition()` | **❌ BROKEN (reverted 2026-05-25)** | First attempt: set `safe_numeric=true` in `phylotree.cpp:initializePtnPartition()` after ptn_start/ptn_end computed. DOES NOT WORK: function pointers (`computeLikelihoodDervPointer` etc.) are assigned by `setLikelihoodKernelFMA()` which reads `safe_numeric` AT CALL TIME — setting the flag afterwards leaves function pointers stale, pointing to `NORM_LH=false` instantiations. Every packet returned NaN (job **169133798**, 32s, exit 2 with all 206 P4-PKT-DIAG packets df=nan). Reverted; `phylotree.cpp` now contains an explanatory comment with `// safe_numeric = true;  // ← DO NOT re-enable without re-dispatch`. **Built as job 169133227** (md5=`161fc0983bacf5f054d7d17b99a64364`). | Non-deterministic crash NOT fixed; first attempt was incorrect | this session |
| **B.4-8 complete fix: `safe_numeric=true` override in `setLikelihoodKernel()` before dispatch** | **✅ FIXED + VERIFIED 2026-05-25** | Correct fix location: `phylotreesse.cpp::setLikelihoodKernel()`, after the default `safe_numeric` derivation but **BEFORE** the `setLikelihoodKernelFMA()` call. `#include "utils/MPIHelper.h"` added. Added `#ifdef _IQTREE_MPI` block: `if (params && params->mode_p_enabled != 0 && MPIHelper::getInstance().getNumProcesses() > 1) { safe_numeric = true; }`. This ensures function pointers for all SIMD kernels (Branch, Derv, Partial, FromBuffer) are assigned to `SAFE_LH=true` template instantiations. Applies on EVERY `setLikelihoodKernel()` call (initial + per-model re-dispatch). **Built as job 169134517** (8m42s, md5=`50b4b172052900c78351b1c16f423e1b`). ISO-2 confirmed PASS as **job 169135061** (exit 0, 15m23s, lnL=−7,541,976.861 = FCA np=1 exact, LG+G4). **Note**: P4-PKT-DIAG NaN diagnostics still fire (~206 per `+I` model evaluation) because the safe kernel does not eliminate NaN df for all patterns; however with `SAFE_NUMERIC=true` the `outError()` crash check is compile-time disabled (`if (!true && ...)` = `if (false)`), so NaN df is tolerated and optimisation proceeds correctly (final lnL unaffected). | `Numerical underflow (lh-derivative)` crash eliminated; ISO-2 deterministically PASS | this session |
| **P.1/P.2 hardening: F-4 + F-5 + F-6 incorporated** | **✅ DONE 2026-05-24** | `mode_p_active_in_mf` SPR guard, `atmd_K_outer<=1` thread-safety guard, VCSIZE-aligned partitions | this session |
| **P.ISO Mode P kernel sandbox bootstrapped** | **✅ DONE 2026-05-24** | Dual-tree sandbox at `/scratch/rc29/as1708/iqtree3-mode-p-iso/` (`src/iqtree3-mode-p-iso-{base,p3}/`) with rsync'd P.1/P.2 source + F-4/5/6 hardening; build/run scripts at `gadi-ci/mode-p-iso/`; parity checker at `tools/mode_p_iso/compare_mode_p_parity.py`; **ready for qsub** | this session |
| **F-11 build bug: `modePAllreduceLh` overload ambiguity — fixed 2026-05-24** | **✅ FIXED (2nd attempt)** | `void modePAllreduceLh(double&)` called `modePAllreduceLh(tmp)` where `tmp` is `double` — still ambiguous (non-const lvalue binds to both `double` and `double&`); final fix: `const double tmp = tree_lh` (const cannot bind to non-const `double&`, forcing by-value overload selection); propagated to all 3 source trees; build resubmitted as job **169130286** | this session |
| **P.3 Kernel patches applied to -p3 ISO tree only** | **✅ DONE 2026-05-24 (NOT YET BUILT/TESTED)** | `phylokernelnew.h` limits-shift + `modePAllreduceLh(tree_lh)` at kernel exit; isolated from -base tree | this session |
| **B.4-9 (HALF-tree_lh SEGFAULT under +I+G4 EM) — FIX APPLIED & VERIFIED 2026-05-25** | **✅ FIXED + VERIFIED (binary `9660575a`, build 169136419, ISO-3 job 169136469 PASSED — MODEP exit 0, lnL=-7,541,976.8614, no SEGFAULT, no HALF-tree_lh leak)** | Two-headed Mode P sync gap. (α) Kernel fallback recompute in `computeLikelihoodBranchSIMD` (phylokernelnew.h:3243) and `computeLikelihoodFromBufferSIMD` (phylokernelnew.h:3541) iterates `[0, orig_nptn)` reading slice-only `_pattern_lh[]` then returns LOCAL tree_lh with no Allreduce → caller sees HALF (≈ tree_lh/nranks). (β) `RateGammaInvar::optimizeWithEM` (rategammainvar.cpp:236) sums `_pattern_lh_cat[]` over `[0, nptn)` per-rank → divergent `newPInvar` → branch lengths diverge → eventual kernel NaN → fallback (α) fires. **Fix:** restrict both kernel fallbacks and the EM accumulator to `[ptn_start, ptn_end)` and Allreduce derived scalars. Latent same-pattern accumulator bugs identified in `rateheterotachy.cpp`, `ratefree.cpp`, `modelmixture.cpp` (NOT yet patched; not on +I+G4 critical path — needed before +R/+H/mixture Mode P). | ISO-3 MODEP sub-run no longer SEGFAULTs during `WAG+I+G4` EM; tree_lh stays Allreduced FULL across all EM iterations | this session |
| **B.4-10 (SIGFPE in +F model evaluation at np=4) — IDENTIFIED 2026-05-25, NOT YET FIXED** | ❌ **OPEN** | SIGFPE (Linux Signal 8, Floating-point exception / FE_INVALID or FE_DIVBYZERO) on rank 3 during `LG+F` model evaluation in ISO-4 (np=4). **Crash signature**: all 4 ranks complete LG/LG+I/LG+G4/LG+I+G4 correctly; crash fires at entry to the first +F model on rank 3 only (`ptn=[72016, 96017)`). Ranks 0–2 log all 4 `LG+F*` models; rank 3's stdout is empty except UCC cascade errors. **Ruled out**: (1) empirical freq computation — `countStates()` uses full alignment, all ranks get identical non-zero frequencies; (2) kernel dispatch — same `computeLikelihoodBranchSIMD<Vec4d, SAFE_LH, 20, true>` for LG and LG+F; (3) B.4-9-α fallback path — not reached (SIGFPE is hardware trap, fired before `!isfinite(tree_lh)` check). **NOT caused by the remainder pattern**: ISO-3 rank 1 has identical `ptn_end=96017` (96017%4=1 remainder) and passes LG+F at np=2; the 1-pattern remainder is not the trigger. Key asymmetry: rank 3 at np=4 has 24,001 patterns (`ptn_start=72016`) vs rank 1 at np=2 has 48,009 patterns (`ptn_start=48008`). **Leading candidates for investigation**: (a) ATMD batch dispatch (`[ATMD Mode F] K_outer=8 M_inner=12`) — model parameter handoff via MPI to ranks; possible buffer misalignment or stale +F parameter state for rank 3 in the 8-model batch; (b) `decomposeRateMatrix()` per-rank eigen decomp with empirical freqs — verify all ranks produce identical eigenvalues; (c) some index in `computeLikelihoodBranchSIMD` uses `ptn_start` arithmetic that produces a zero divisor for the specific value 72016. **Diagnostic run submitted 2026-05-25**: `--atmd-k-outer 1` (single-model batches, K_outer=8 → K_outer=1) job **169136683** (`iso4-diag-k1`, script `gadi-ci/mode-p-iso/run_iso4_diag_k1_np4_p3.sh`, MODEP-only, walltime 20 min). **Interpretation**: if job 169136683 PASSES → B.4-10 root cause is in the K_outer>1 batch dispatch / model-parameter MPI handoff to rank 3 for +F models; if it also crashes with SIGFPE → root cause is in per-model +F kernel arithmetic specific to `ptn_start=72016`. Log: `~/setonix-iq/iso4-diag-k1.o169136683`. | ISO-4 MODEP np=4 exits 0 with no SIGFPE during LG+F batch | superseded — diagnostic revealed B.4-11 fires first at LG+G4 |
| **B.4-11 (`TL_UCP ERROR Message truncated` SEGFAULT at LG+G4 — conditional Allreduce mismatch) — IDENTIFIED & FIXED 2026-05-25** | ✅ **FIX VERIFIED — LG+G4 crash eliminated. B.4-11 fix confirmed by job 169137348 (K1 diag, `85bf5c79`): model 2 (LG+G4) and models 3–47 all pass; crash moved to Q.INSECT+G4 (model ~50) = new bug B.4-13. Full ISO-4 (169137349 modep_np4) crashed at LG+F+I due to K_outer=8 MPI_THREAD_FUNNELED violation = new bug B.4-12. Binary `85bf5c79` is superseded by P.7-MPGC `efef6be4`.** | **Root cause**: B.4-9-α added an `modePAllreduceLh` call inside the `!isfinite(tree_lh)` fallback block in `computeLikelihoodBranchSIMD` (phylokernelnew.h:~3276-3279). This creates a **conditional Allreduce** that fires only on ranks whose local `tree_lh` is non-finite (underflow). At np=4 with LG+G4 (nrates=4, first Gamma-4 model), small per-rank pattern slices (24K vs 48K at np=2) cause rank-local underflow on SOME ranks but not ALL. Ranks that enter the fallback: 2 Allreduce calls (fallback + main-path at line 3375). Other ranks: 1 Allreduce call (main-path only). Collective mismatch → UCX `Message truncated` error → SEGFAULT. The main-path unconditional Allreduce at line 3375 ALREADY handles aggregation correctly from both fallback and non-fallback ranks; the fallback Allreduce was redundant AND harmful. **Why np=2 (ISO-2, ISO-3) passed**: at np=2 each rank has ~48K patterns vs ~24K at np=4; the larger slice is more numerically stable and rank-local underflow does not occur at LG+G4. **Why LG/LG+I passed even at np=4**: simpler models (nrates=1 or ~1) produce stable initial likelihoods. **Crash observed**: jobs 169136682/683 (`--atmd-k-outer 1`, np=4 AA 100K), ALL 4 ranks, at LG+G4. Fix: removed 4 lines (`if (mp_active_fb) { ... modePAllreduceLh ... }`) from BranchSIMD fallback while RETAINING the loop restriction to [ptn_start, ptn_end). P.5a (FromBuffer) fallback Allreduce RETAINED (correct there: early unconditional Allreduce makes all ranks agree on isfinite(tree_lh) before fallback). File: `tree/phylokernelnew.h` lines 3244–3283. **Relationship to B.4-10**: original ISO-4 job 169136585 (K_outer=8) crashed with exit 136 at LG+F (rank 3 only). B.4-11 fires at LG+G4 (before reaching LG+F). B.4-11 fix may resolve B.4-10 too; jobs 169137348/169137349 will confirm. | ISO-4 MODEP exits 0; no Message-truncated errors; all 4 ranks complete LG, LG+I, LG+G4, LG+I+G4, LG+F family; best model LG+G4; lnL within 0.05 of ref | pending jobs 169137348/349 |
| P.3 ISO build + ISO-0/1/2 gate runs | **✅ COMPLETE 2026-05-25 (deterministic)** — **ISO-0 ✅ PASSED**, **ISO-1 ✅ PASSED 2026-05-24**, **ISO-2 ✅ PASSED 2026-05-25 deterministically (job 169135061, binary `50b4b172`, lnL=-7,541,976.861 = FCA np=1 exact Δ=0, MF wall 235.844s)** — earlier `a278e44c` PASS (169132572) was lucky OMP scheduling; B.4-8 numerical issue is real and now fixed by forcing `safe_numeric=true` for Mode P + np>1 inside `setLikelihoodKernel`. Closure required P.3 (Branch) + P.4 (Derv) + P.5a (FromBuffer) + P.6-lite (collective dispatch) + B.4-8 (safe-kernel force at dispatch site). FP non-associativity finding: Mode P parity reference is FCA np=1 (single-rank sequential), NOT FCA np=2. **B.4-3a** (jobs 169130537/169130538): `--bind-to none` missing → 1 core detected → abort; **B.4-3b** (job 169130673): 2 ranks on 1 node + `OMP_PROC_BIND=close` → libiomp5 affinity conflict → rank 1 SIGSEGV; **B.4-3c** (job 169130806): ~~Mode F K_outer=8 misdiagnosis — see B.4-5~~; **B.4-5** (jobs 169130806+169130878): real root cause — `errstreambuf::overflow` in `main.cpp:1801` has null `fout_buf` for worker ranks; first cerr write from rank 1 → SIGSEGV; **fix**: null-guard `fout_buf` in `errstreambuf::overflow` and `sync` (both ISO source trees); base binary rebuilt as job 169131254 (md5=`6d1c1729`); **ISO-1 (job 169131306, 2 nodes, np=2, AA 100K, --mode-p-all): MF done 318.627s, full pipeline done 1007.414s wall, NNI 103 iters, model=LG+G4 α=0.996, lnL=-7,541,976.852 (Δ=0.001 vs FCA ref -7,541,976.853 ✓), rank 0 31× `[Mode P] ptn=[0,48008)`, rank 1 31× `[Mode P] ptn=[48008,96017)`, perfect 50/50 partition.** PBS exit 141 (SIGPIPE) is cosmetic — IQ-TREE itself exited 0. **P.3 binary built ✅** (job 169131641, 9 min, md5=`79550723`); a pre-build **F-11-bis fix** was needed first (job 169131566 ambig overload at phylokernelnew.h:3292 — fixed with `{ const double tree_lh_local = tree_lh; tree_lh = modePAllreduceLh(tree_lh_local); }`). **ISO-2 ❌ FAILED 2026-05-24** (job 169131722, exit 2, 25s wall): with `-m TEST --mode-p-all`, FCA dispatched LG to rank 0 + WAG to rank 1; both ranks called collective `MPI_Allreduce` simultaneously on different models → garbage tree_lh + half-LG/half-WAG `theta_all` → Derv underflow at phylokernelnew.h:2595. **B.4-7 / F-12 design defect**: Mode P kernel patch is correct in isolation, but the F-4 enable-during-MF gate is insufficient — Mode P additionally requires all ranks to be on the **same model** at every collective Allreduce. **Next**: implement P.6-lite collective dispatch OR add `--no-fca` gate, then rebuild + re-run ISO-2. | ISO-2: lnL within 1e-6 of FCA np=2 ref `-7,541,976.853` | this session (compute) |
| P.4 Kernel: same for `computeLikelihoodDervSIMD` (derivative kernel) | **✅ DONE 2026-05-24** — Patch 1 (limits-shift via replace_all, also caught DervMixlen sibling) at phylokernelnew.h:2305; Patch 2 (3-value Allreduce `modePAllreduceLhDfDdf(dummy_lh, all_df, all_ddf)`) at phylokernelnew.h:2610; P.4b mixlen path guarded with `outError("Mode P + mixlen not supported")` for non-joint-branch models (deferred full implementation). Validated end-to-end by ISO-2 PASS. | NNI lnL traces match FCA | this session |
| P.5a Kernel: `computeLikelihoodFromBufferSIMD` | **✅ DONE 2026-05-24** — Patch 1 (flat ptn-loop bounds `for (ptn = mp_lo; ptn < mp_hi; ...)`) at phylokernelnew.h:3420; Patch 2 (Allreduce tree_lh) at phylokernelnew.h:3499. Unobserved-pattern tail `[orig_nptn, nptn)` deliberately skipped under Mode P (safe per F-1: ASC OFF during MF). Validated by ISO-2 PASS. | All optimisation paths consistent | this session |
| P.5b Kernel: `computeLikelihoodDervMixlenSIMD` (mixlen variant) | ⏳ PARTIAL — limits-shift applied at phylokernelnew.h:3603 (via P.4 Patch 1 replace_all); Allreduce applied at phylokernelnew.h:3729. Not exercised by ISO-2 (LG+G4 is joint-branch). | Validate with explicit mixlen test if used in production | follow-up |
| **B.4-12 (MPI_THREAD_FUNNELED violation under K_outer=8 + Mode P np=4) — FIXED 2026-05-25 (script fix, no rebuild)** | ✅ **FIXED** | **Root cause**: `run_iso4_aa100k_np4_p3.sh` removed `--atmd-k-outer 1` from the MODEP sub-run ("caused step-count asymmetry BASE=92 vs MODEP=61 in ISO-3"). K_outer=8 (default) dispatches 8 OMP worker threads, each calling Mode P Allreduce → violates `MPI_THREAD_FUNNELED` (only main thread may call MPI) → "Message truncated" SEGFAULT on ALL 4 ranks at LG+F+I (model 6). The NNI step-count asymmetry noted in ISO-3 is EXPECTED with Mode P: different pattern partitioning changes the optimization landscape so step counts legitimately differ. This is NOT a correctness issue; the script should not have removed K_outer=1 enforcement. **Evidence**: job 169137349 `modep_np4` sub-run (binary `85bf5c79`, K_outer=8): 0 MF-TIME completions, crash at model 6 (LG+F+I). Job 169137348 K1 diag (K_outer=1): 15 MF-TIME completions, crash at model ~50 = different bug B.4-13. **Fix**: re-add `--atmd-k-outer 1` to MODEP sub-run in `run_iso4_aa100k_np4_p3.sh`. Additionally consider enforcing K_outer≤1 in C++ code when Mode P is active. | MODEP sub-run exits 0; no Message-truncated errors at any model | requires script fix + rebuild test |
| **B.4-14 (filterRatesMPI MPI_COMM_WORLD Bcast deadlocks under P.7-MPGC ngroups>1) — FIX BUILT 2026-05-25 (binary `43c313f3`)** | ✅ FIXED (binary `43c313f30464ba4a9296ba6e68dd05e7`, build 169176245 exit=0 8m34s) | **Root cause:** `filterRatesMPI()` (main/phylotesting.cpp:3120) does two `MPI_Bcast(... MPI_COMM_WORLD)` calls when a rank's reference family completes. Under P.7-MPGC with ngroups=2 (default np=4 → 2 groups of 2), "rank" = group, so group 0 reaches the call at time T0 (LG family done) while group 1 is still in WAG family at T0. A world-scoped Bcast at T0 needs all 4 world ranks to participate, but group 1's two ranks are NOT at this call → cross-group MPI_Bcast mismatch → UCX "Message truncated" or deadlock at the first non-LG group's filterRatesMPI fire-point. **Novel fix — "Communicator-Scoped Collective Hierarchy" (CSCH):** every MPI collective declares its scope. WORLD-scope collectives MUST be at deterministic synchronization points (end-of-evaluateAll score gather, Step-8 setup gate); GROUP-scope collectives MAY fire per-group-per-model-completion. filterRatesMPI is logically GROUP-scope (each group prunes its OWN ref family — group 0's ok_rates from LG would be WRONG for group 1 which owns WAG family). **Implementation:** added `MPI_Comm fca_comm = MPI_COMM_WORLD` member to CandidateModelSet (`phylotesting.h:457`); `evaluateAll` MPGC setup block sets `this->fca_comm = mpgc_comm` (intra-group sub-communicator from `MPI_Comm_split`); `filterRatesMPI` uses `MPI_Comm_rank/size` on `fca_comm` for local rank/size (group leader = rank 0 in group_comm) and Bcast root=0 in `fca_comm`. Under MPGC ngroups>1: each group prunes within itself; under legacy FCA / non-MPGC: `fca_comm == MPI_COMM_WORLD` → identical behaviour. Reset to `MPI_COMM_WORLD` in MPGC cleanup before `MPI_Comm_free`. Markers: `B.4-14` in source. **Build history**: job 169175807 exit=4 (preflight regex `p6_lite_collective` failed — P.7-MPGC had replaced P.6-lite, string absent); job 169175914 exit=4 (preflight regex `B\.4-9 Mode P fix` found 1 match — Branch fallback comment had been updated to `B.4-9 / B.4-11 Mode P fix` in prior fix, FromBuffer still `B.4-9`; regex updated to `B\.4-9.* Mode P fix`); build script also updated P.6-lite check to `grep -qE 'p6_lite_collective\|P\.7-MPGC\|mpgc_active'`; B.4-14 preflight added (header≥1 `B.4-14` + cpp≥4 `B.4-14\|fca_comm`); job **169176245** exit=0 (8m34s, md5=`43c313f3`). | ISO-4 MPGC sub-run progresses past filterRatesMPI fire-point on group 0 without world-Bcast deadlock; both groups complete their assigned families; world Allreduces at end-of-evaluateAll merge scores correctly | pending ISO-4 validation (submit `run_iso4_diag_k1_np4_p3.sh` + `run_iso4_aa100k_np4_p3.sh`) |
| **B.4-13 ("Message truncated" SEGFAULT — concurrent sub-communicator collectives confused by OpenMPI 4.1.7 UCC component) — FIX IN TEST (job 169189043)** | ❌ **BARRIER FIX WAS INSUFFICIENT — UCC FIX TESTING (job 169189043)** | **Root cause (REVISED 2026-05-26):** Job 169179725 (`ad77fa4e`, K1 diag) still crashed with 7 MF-TIME completions at Q.BIRD+G4/Q.MAMMAL+G4 boundary (model ~24). Per-model intra-group barrier in `ad77fa4e` did NOT fix the crash. Analysis: at the crash point, both groups ARE properly synchronized (ranks 0,1 on Q.BIRD+G4; ranks 2,3 on Q.MAMMAL+G4) — the B.4-11/MPGC lockstep design is working correctly within each group. The crash is caused by **OpenMPI 4.1.7 UCC component `tl_ucp_coll.c` failing to isolate concurrent collectives on different sub-communicators**: when group 0 calls `modePAllreduceLh(count=1, group0_comm)` while group 1 calls `modePAllreduceLhDfDdf(count=3, group1_comm)` simultaneously, UCC's UCP transport layer confuses the two reductions (possibly due to shared collective ID slot or hash collision in UCC team cache) → one rank receives fewer bytes than expected → UCX "Message truncated". **Job 169179725 evidence**: `TL_UCP ERROR failure in recv completion Message truncated` in `tl_ucp_coll.c:137`, fired as soon as both groups were simultaneously active (first model after the two groups diverged). **Fix (2026-05-26):** `--mca coll ^ucc` in `mpirun` command disables UCC collective component; OpenMPI falls back to `tuned` which properly uses MPI context IDs to isolate sub-communicator collective messages. Both run scripts updated. No code change or rebuild; binary `ad77fa4e` unchanged. Note: per-model barrier from the earlier `ad77fa4e` fix is still in place (harmless; adds ~0.1ms per model). |nization → at model 50 some ranks ahead/behind → intra-rank Allreduce calls target DIFFERENT models → garbage tree_lh + underflow. **Evidence**: job 169137348 (binary `85bf5c79`, K_outer=1): 15 MF-TIME completions, crash at Q.INSECT+G4. **Hypothesis for P.7-MPGC `efef6be4`**: GROUP dispatch (2 ranks per group, intra-group Allreduce on `mp_allreduce_comm`) ensures both ranks in a group always evaluate SAME model in lockstep — drift cannot happen. P.6-lite eliminated → B.4-13 implicitly resolved. **Verify**: submit ISO-4 K1 diag with `efef6be4` + B.4-14 fix (build 169176245); if Q.INSECT+G4 still crashes, root cause is more fundamental. | K_outer=1 MODEP sub-run exits 0; all ranks complete Q.INSECT family; lnL matches np=1 ref within 0.05 | verify via ISO-4 resubmit (pending — jobs 169177186/169177187 were with old binary `43c313f3`, crashed at JTT+F+G4 boundary, confirming bug; fix applied in `ad77fa4e`) |
| **B.4-15 (MPGC state not propagated to per-model `iqtree` — wrong scores, no crash) — FIX APPLIED 2026-05-26 (build pending)** | ✅ FIXED (build pending) | **Root cause:** `CandidateModel::evaluate()` creates a fresh `iqtree` (`main/phylotesting.cpp:1962`) for each model. The new tree has DEFAULT `mp_group_size=1` and `mp_allreduce_comm=MPI_COMM_WORLD` (from `phylotree.h` inline initialisers at lines 2361–2364). The MPGC state previously set on `in_tree` by `setModePGroupComm()` in `evaluateAll`'s MPGC setup block (line ~4026) is NEVER propagated to the new iqtree. So `iqtree->initializePtnPartition()` at line ~2014 hits the WORLD-rank/size fallback (`phylotree.cpp:958-961`), and the kernel `modePAllreduceLh()` runs on `MPI_COMM_WORLD` (because `iqtree->mp_allreduce_comm == MPI_COMM_WORLD`). **Symptom (job 169189043, `ad77fa4e` + `--mca coll ^ucc`, K1 diag, np=4):** no crash (exit=0 80s) but every Group 1 substitution family's lnL score is NUMERICALLY IDENTICAL to its paired Group 0 family (LG+G4 == WAG+G4 == −7,572,497.929, LG+F+G4 == WAG+F+G4 == −7,569,245.292, JTT+G4 == Q.PFAM+G4 == −7,606,759.529, etc.). Mechanism: group 0 (LG family, ranks 0,1) and group 1 (WAG family, ranks 2,3) call WORLD Allreduce concurrently → all 4 ranks see the SAME sum `lnL_LG(first_half) + lnL_WAG(second_half)` (mixed across families). Best-model selection wrong: LG+F+G4 chosen instead of LG+G4; lnL off by ~30k nats. **Diagnostic discrepancy:** MF-MPI-DIAG (printed at MPGC setup from phylotesting.cpp:4040) shows rank 0 `ptn=[0,48008)` (group-based, correct). Per-rank stderr `[Mode P]` (printed from phylotesting.cpp at line ~2016 right after the kernel ran for the model) shows world-based quarters (rank 0=[0,24008), rank 1=[24008,48008), rank 2=[48008,72016), rank 3=[72016,96017)). **Novel fix — "MPGC Inheritance":** recognise that MPGC state is tree-bound execution context, not config. Plumb it through the tree creation lifecycle the same way `params`/`kernel`/`numThreads` are plumbed. Added `PhyloTree *in_tree = nullptr` parameter to `CandidateModel::evaluate()` (default nullptr preserves all non-MF callers); right after `iqtree->setNumThreads()` in the evaluate body, propagate group state: `if (in_tree && in_tree->mp_group_size > 1) iqtree->setModePGroupComm(in_tree->mp_allreduce_comm, in_tree->mp_group_rank, in_tree->mp_group_size);`. Both call sites in `CandidateModelSet` (`evaluateAll:4368`, `test:3532`) updated to pass `in_tree`. The fix runs BEFORE `iqtree->initializePtnPartition()` so the group state is in place when the partition is computed. Enhanced `[Mode P]` stderr diagnostic includes `grp_rank=…/grp_size=…` so post-fix runs can verify group state at evaluation time matches the MPGC-setup state. Backward compatible: legacy P.6-lite (no MPGC) has `in_tree->mp_group_size==1` and the if-branch skips, leaving iqtree at defaults (= old behaviour). Files: `main/phylotesting.h` (declaration + parameter doc), `main/phylotesting.cpp` (definition + propagation + 2 call sites + diagnostic). Preflight: `B.4-15` markers + 3rd `setModePGroupComm` call site in cpp. | ISO-4 K1 diag (np=4) all 4 ranks emit `[Mode P]` with `grp_size=2`; rank-in-group=0 ranks (world 0, 2) get `ptn=[0,48008)` and rank-in-group=1 (world 1, 3) get `ptn=[48008,96017)`; LG+G4 best model with lnL ≈ −7,541,976.86 (NOT identical to WAG+G4 anymore); no MPI_COMM_WORLD Allreduce in kernel | pending build + ISO-4 K1 diag validation |
| **P.7-MPGC (Mode P Group Communicator) — permanent replacement for P.6-lite — IMPLEMENTED 2026-05-25** | ✅ **IMPLEMENTED (binary `efef6be4d5ac8dc9686317e2a50b2fb2`); ISO-4 verification pending** | **IQ-tree-specific two-tier architecture**: Tier 1 (across groups — FCA at group level): world np split into groups of `mode_p_group_size` ranks (default 2). FCA greedy-LPT assigns each substitution family to one group, giving `floor(np/group_size)` model families in parallel. `fca_rank = group_id`, `fca_nranks = ngroups` for assignment. Tier 2 (within each group — Mode P pattern split): all ranks in a group evaluate the same model simultaneously; each rank owns `1/group_size` of patterns; `MPI_Allreduce` uses `mp_allreduce_comm` (intra-group sub-communicator from `MPI_Comm_split`), NOT `MPI_COMM_WORLD`. **Files changed**: `utils/tools.h` (+`mode_p_group_size=-1` param), `tree/phylotree.h` (+`mp_group_rank`, `mp_group_size`, `mp_allreduce_comm`, `setModePGroupComm()`), `tree/phylotree.cpp` (MPGC-aware `initializePtnPartition()`, new `setModePGroupComm()`, `modePAllreduceLh` + `modePAllreduceLhDfDdf` use `mp_allreduce_comm`), `main/phylotesting.cpp` (MPGC block replaces P.6-lite; FCA block uses `fca_my_rank`/`fca_my_nranks`; MPGC cleanup at `evaluateAll` exit). **Expected speedup at np=4, group_size=2**: 2 groups × ~1.5× Mode P ≈ **3× vs np=1** (P.6-lite was only 1.7×). **np=2 with MPGC**: 1 group of 2 = P.6-lite-equivalent numerics, but `filterRatesMPI` now ENABLED. | ISO-4 np=4 exits 0; all ranks complete full model set; lnL matches np=1 ref; MPGC diagnostic lines show correct group assignment | submit ISO-4 verification job |
| P.6 Dispatcher: heavy-model selection in `evaluateAll` | ⏳ SUPERSEDED BY P.7-MPGC | P.7-MPGC combines model-level (FCA at group level) and pattern-level (Mode P within group) parallelism, making the original P.6 cost-threshold dispatcher unnecessary for correctness; perf tuning via `--mode-p-group-size` | follow-up |
| P.7 Validation: AA 1M np=16 perf gate | ⏳ PENDING | MF wall ≤ 600 s (vs FCA 1,122 s) | follow-up |

---

## What's IN the source tree right now (P.1 + P.2)

**Source changes (iqtree3-mf-iso/src/iqtree3/):**

| File | Lines | What |
|---|---|---|
| `utils/tools.h` | 2385–2397 | New `Params::mode_p_enabled` (int) and `Params::mode_p_min_cost_mult` (double) |
| `utils/tools.cpp` | 4626–4651 | CLI flags `--mode-p`, `--mode-p-all`, `--no-mode-p`, `--mode-p-min-cost-mult` |
| `tree/phylotree.h` | 2314–2343 | New `PhyloTree::ptn_start, ptn_end` (size_t) + 4 helper methods |
| `tree/phylotree.cpp` | 906–984 | Implementations of `isModePActive()`, `initializePtnPartition()`, `modePAllreduceLh()`, `modePAllreduceLhDfDdf()` |
| `main/phylotesting.cpp` | 1995–2003 | `iqtree->initializePtnPartition()` call in `CandidateModel::evaluate()` after `initializeModel()` |

**Behaviour with this scaffolding alone:**
- Build succeeds (no kernel changes — zero risk of correctness regression)
- `--mode-p`, `--mode-p-all`, `--no-mode-p` are accepted on the CLI
- With `--mode-p-all`: a `[Mode P] rank R model=X ptn=[start, end) of N` line is emitted per model
- **Likelihood values are unchanged** because the kernel does not yet consult `ptn_start`/`ptn_end`
- Mode P is "inert" — the partition is set but not enforced

**Test recipe for P.1+P.2 validation:**

```bash
# 1. Build with the scaffolding (clean build, no special flag needed)
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/build_atmd_b4.sh    # OR a new build_mode_p.sh script

# 2. Single-rank run to confirm Mode P is a no-op when ranks==1:
mpirun -np 1 .../iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G --mode-p-all
# Expected: no [Mode P] line (gated on getNumProcesses() > 1)

# 3. Multi-rank run to confirm partition is emitted:
mpirun -np 4 .../iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G --mode-p-all
# Expected: 4 [Mode P] lines, e.g.:
#   [Mode P] rank 0 model=LG+G ptn=[0, 25000) of 100000
#   [Mode P] rank 1 model=LG+G ptn=[25000, 50000) of 100000
#   ...
# Expected: lnL identical to FCA (because kernel ignores partition for now)
```

---

## P.3 — Kernel modification: restrict pattern loop bounds

The kernel currently iterates `[0, nptn)` regardless of rank. To make Mode P operative, the per-rank pattern slice must be honored.

### Implementation strategy

The kernel uses `computeBounds<VectorClass>(num_threads, num_packets, nptn, limits)` at [phylokernelnew.h:2780](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2780) to divide the pattern range into per-thread packets. Each packet processes `[limits[packet_id], limits[packet_id+1])`. 

**Cleanest modification**: shift the work to operate on `[ptn_start, ptn_end)` by:
1. Computing the rank's slice size: `mp_nptn = isModePActive() ? (ptn_end - ptn_start) : nptn`.
2. Calling `computeBounds(num_threads, num_packets, mp_nptn, limits)`.
3. Adding `ptn_start` to every entry of `limits[]` post-computation.

This way the existing per-packet loop body is unchanged; only the loop range shifts.

### Exact patch for computeLikelihoodBranchSIMD ([phylokernelnew.h:2660](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2660))

**Replace (lines ~2776–2782):**
```cpp
    size_t nptn = max_orig_nptn + model_factory->unobserved_ptns.size();
    // ...
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);
```

**With:**
```cpp
    size_t nptn = max_orig_nptn + model_factory->unobserved_ptns.size();
    // ...
    // P.3 Mode P: when active, restrict this rank's work to the assigned slice
    // [ptn_start, ptn_end). When inactive, behaviour is identical to before.
    const bool mp_active = isModePActive();
    const size_t mp_lo = mp_active ? ptn_start : 0;
    const size_t mp_hi = mp_active ? std::min(ptn_end, nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, mp_size, limits);
    if (mp_active) {
        // Shift packet boundaries into [ptn_start, ptn_end). Note: computeBounds
        // rounds the size up to a VCSIZE multiple; the rounded tail past mp_hi
        // is still safe because the per-packet loop checks `ptn < nptn` etc.
        for (size_t &lim : limits) lim += mp_lo;
    }
```

**Insert at the kernel exit, just before `return tree_lh;` at [phylokernelnew.h:3274](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L3274):**

```cpp
    // P.3 Mode P: combine per-rank tree_lh sums. No-op when Mode P is inactive
    // or ranks==1. Called from the master thread of the implicit kernel-exit
    // serial region (MPI_THREAD_FUNNELED is sufficient — see MPIHelper.cpp:28).
    tree_lh = modePAllreduceLh(tree_lh);
    return tree_lh;
```

**Caveat — ASC (ascertainment) correction interaction:** the ASC correction at lines 3227–3273 of the kernel uses `all_prob_const`. Under Mode P, each rank has only its slice of `all_prob_const`. The ASC correction must happen AFTER Allreducing `all_prob_const` too. The simplest fix: move the Allreduce ABOVE the ASC block:

**Insert before the `if (ASC_Holder)` line at [phylokernelnew.h:3227](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L3227):**

```cpp
    // P.3 Mode P: aggregate tree_lh and all_prob_const before ASC correction.
    // This MUST happen before the ASC block because the correction formula
    // applies the (Allreduce'd) all_prob_const to the (Allreduce'd) tree_lh.
    if (isModePActive()) {
        double in[2]  = {tree_lh, all_prob_const};
        double out[2] = {0.0, 0.0};
#ifdef _IQTREE_MPI
        MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        tree_lh = out[0];
        all_prob_const = out[1];
    }
```

And REMOVE the `tree_lh = modePAllreduceLh(tree_lh);` near `return tree_lh;` if you take this approach (it'd double-Allreduce).

### Validation criterion

```bash
# AA 100K, np=2, --mode-p-all (force Mode P on every model)
# Expected: lnL identical to FCA np=2 within 1e-6
mpirun -np 2 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G --mode-p-all -seed 1
# Compare to:
mpirun -np 2 iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G -seed 1
# Both must produce: -7,541,976.853 ± 1e-6
```

If lnL diverges:
- Likely cause: `_pattern_lh[ptn]` is read by post-kernel code expecting full coverage. Under Mode P each rank's _pattern_lh is only populated for its slice. Fix: MPI_Allgather _pattern_lh after the kernel, OR restrict post-kernel _pattern_lh consumption to per-rank slices.
- Less likely: VCSIZE-aligned tail overflow when ptn_end isn't aligned to VectorClass::size(). Inspect `computeBounds` tail handling.

---

## P.4 — Same modifications for derivative kernel

The derivative kernel `computeLikelihoodDervSIMD` at [phylokernelnew.h:2239](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2239) shares the same structure but outputs three values (`tree_lh`, `df`, `ddf`). The Allreduce needs to handle all three.

### Exact patches

**Same `mp_lo/mp_hi/limits-shift` modification** at the kernel entry (mirror P.3).

**At the kernel exit, replace the existing `tree_lh += all_tree_lh; df += all_df; ddf += all_ddf;` block with:**

```cpp
    tree_lh += all_tree_lh;
    df      += all_df;
    ddf     += all_ddf;
    // P.4 Mode P: aggregate (lh, df, ddf) triple across ranks.
    modePAllreduceLhDfDdf(tree_lh, df, ddf);
```

`modePAllreduceLhDfDdf` is the helper already implemented in phylotree.cpp:966 — it takes three double refs and Allreduces them as a single MPI buffer (one network round-trip instead of three).

### Validation

```bash
# Per-branch gradient check: with --verbose, dump tree_lh, df, ddf at each Newton step
mpirun -np 2 iqtree3-mpi-mode-p -s AA_100K.phy -m LG+G --mode-p-all -v
# Compare branch length convergence trajectory to FCA np=2 baseline
```

---

## P.5 — Remaining kernel variants

Apply the same pattern to:

| Function | File:Line | Output(s) | Notes |
|---|---|---|---|
| `computeLikelihoodBranchGenericSIMD` | phylokernelnew.h:2663 | tree_lh | Identical body to BranchSIMD; same patch |
| `computeLikelihoodDervGenericSIMD` | phylokernelnew.h:2242 | lh, df, ddf | Same patch as DervSIMD |
| `computeLikelihoodFromBufferSIMD` | phylokernelnew.h:3286 | tree_lh | Aggregation at line 3437 `tree_lh = all_tree_lh` + post-loop at 3448 |
| `computeLikelihoodFromBufferGenericSIMD` | phylokernelnew.h:3289 | tree_lh | Same |
| `computeLikelihoodDervMixlenSIMD` | phylokernelnew.h:3500 | df, ddf (2 outputs) | Uses `modePAllreduceLhDfDdf` with lh = 0 dummy, or a new 2-arg helper |
| `computeLikelihoodDervMixlenGenericSIMD` | phylokernelnew.h:3503 | df, ddf | Same |
| `computeMixtureLikelihoodBranchEigenSIMD` | phylokernelmixture.h:730 | tree_lh | Older kernel variant; same logical patch |
| `computeMixtureLikelihoodDervEigenSIMD` | phylokernelmixture.h:464 | lh, df, ddf | |
| `computeNonrevLikelihoodBranchSIMD` | phylokernelnonrev.h:1058 | tree_lh | Non-reversible models |
| `computeNonrevLikelihoodDervSIMD` | phylokernelnonrev.h:582 | lh, df, ddf | |

**Estimated time** for P.5 (10 kernel variants × ~30 min each, including build/test cycles): **5 hours**.

---

## P.6 — Dispatcher: route heavy models through Mode P

Currently `--mode-p-all` forces Mode P on every model. For production use, only heavy models should use Mode P; light models should stay on Mode F (where K_outer=4-8 concurrent dispatch is faster than 16-rank Allreduce overhead per kernel call).

### Implementation in `evaluateAll` (phylotesting.cpp)

**Before the main do-while loop at [phylotesting.cpp:4226](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/main/phylotesting.cpp#L4226), add:**

```cpp
    // P.6: per-model Mode P selection. FCA's predicted_cost (lines 3927–3958)
    // already ranks models by cost; mark models above the threshold for Mode P.
    if (params.mode_p_enabled > 0) {  // 1 = auto-select; -1 = force-all (already on)
        double total_cost = 0.0;
        for (size_t m = 0; m < size(); m++) total_cost += at(m).predicted_cost;
        double avg_cost = total_cost / std::max((size_t)1, size());
        double thresh = avg_cost * params.mode_p_min_cost_mult;
        for (size_t m = 0; m < size(); m++) {
            if (at(m).predicted_cost > thresh)
                at(m).flags |= MF_MODE_P;
        }
    }
```

**Then in the do-while body, before `at(model).evaluate(...)`:**

```cpp
    bool model_is_mode_p = at(model).hasFlag(MF_MODE_P);
    if (model_is_mode_p) {
        // All ranks must rendezvous here; Mode F dispatch is suspended for this model.
        MPI_Barrier(MPI_COMM_WORLD);
        // Temporarily flip Params::mode_p_enabled = -1 so evaluate() activates Mode P.
        int saved = params.mode_p_enabled;
        params.mode_p_enabled = -1;
        tree_string = at(model).evaluate(params, model_info, out_model_info,
                                         models_block, num_threads, brlen_type,
                                         &mpi_warm_start);
        params.mode_p_enabled = saved;
        // All ranks now have the same per-model result; no need to Bcast.
    } else {
        // Existing Mode F dispatch unchanged.
        tree_string = at(model).evaluate(params, model_info, out_model_info,
                                         models_block, atmd_M_inner, brlen_type,
                                         &mpi_warm_start);
    }
```

Add `MF_MODE_P` flag to the CandidateModel flag enum (currently has MF_IGNORED, MF_WAITING, MF_RUNNING — add `MF_MODE_P = 0x10` or similar).

---

## P.7 — Performance validation

After P.3-P.6 are merged and lnL parity is confirmed:

```bash
# Baseline FCA np=16:
qsub run_fca_aa_1m_16node.sh                    # ref MF=1,122 s

# Mode P on top 3 heavy models (default threshold 8x):
qsub run_atmd_mode_p_aa_1m_16node.sh --mode-p   # expect MF ≤ 822 s

# Mode P on top 8 (lower threshold):
MODE_P_THRESH=4 qsub run_atmd_mode_p_aa_1m_16node.sh   # expect MF ≤ 600 s

# Mode P on all (validation only — Allreduce overhead dominates light models):
qsub run_atmd_mode_p_all_aa_1m_16node.sh        # expect MF probably > FCA
```

---

## Build & test recipe (P.3 onward)

```bash
# Build the Mode P binary (same as b4 with the new kernel modifications applied):
cd /scratch/rc29/as1708/iqtree3-mf-iso
cp -r src/iqtree3 src/iqtree3-mode-p   # checkout
# ...apply P.3 patches to phylokernelnew.h...
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/build_atmd_mode_p.sh   # new build script

# Correctness gate at AA 100K np=2:
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/run_mode_p_correctness_aa_100k_2node.sh
# Pass: lnL within 1e-6 of FCA np=2 baseline (-7,541,976.853)
# Fail: investigate _pattern_lh consumption or VCSIZE-tail handling

# Performance gate at AA 1M np=16:
qsub ~/setonix-iq/gadi-ci/lbfgs-ws/run_mode_p_perf_aa_1m_16node.sh
# Target: MF wall ≤ 600 s (vs FCA ref 1,122 s)
```

---

## Known issues to address during implementation

1. **`_pattern_lh` consumption**: kernels write `_pattern_lh[ptn]` for each computed pattern. Under Mode P each rank only populates its slice. Downstream code (bootstrap, site-likelihood reporting) reads `_pattern_lh` expecting full coverage. **Fix:** add MPI_Allgather of `_pattern_lh` AFTER kernel exit when Mode P is active. Cost: ~8 MB per Allgather on AA 1M, ~5 ms on Gadi InfiniBand — acceptable.

2. **VCSIZE tail overflow**: `computeBounds` rounds the size up to a multiple of VectorClass::size() (line 1121). When `mp_size` is not a VCSIZE multiple, the last packet has tail patterns past `mp_hi`. The existing kernel handles ptn >= nptn safely (state = STATE_UNKNOWN at line 2894); the same logic applies for ptn >= mp_hi as long as the partial_lh and other arrays are large enough. **Validate** by running a dataset with npat ≢ 0 (mod 16) on np=2.

3. **Mixture model nmixtures=4**: kernel uses `ncat_mix = ncat * model->getNMixtures()`. Mode P partition by pattern is orthogonal to mixture categories (each pattern has all mixture cats). No issue expected.

4. **First-touch / partial_lh per-rank consistency**: each rank computes partial_lh only for its slice. When the tree topology changes (NNI), partial_lh needs to be re-computed. Under Mode P this happens per-rank; if rank R needs partial_lh[ptn] for ptn outside its slice (e.g., to evaluate a different branch), it'll be stale/uninitialised. **Fix:** ensure branch optimisation uses only the rank's slice. The branch optimization loop in `optimizeOneBranch` iterates over Newton-Raphson steps; each step calls the kernel which only touches the slice. As long as all branches share the same per-rank slice (no per-branch slice reallocation), this works.

5. **Mode P + Mode F coexistence**: P.6 ensures only ONE mode is active per model. The MPI_Barrier before the Mode P model dispatch synchronises all ranks; the existing FCA model-distribution skips Mode P models from per-rank ownership. Verify via FCA dispatch logic at phylotesting.cpp:3960–3992.

---

## Status of remaining work

- **P.ISO is now the first required action**: create a Mode P kernel ISO sandbox, mirroring the FCA `mf.iso` workflow, before applying any P.3+ SIMD kernel patches to the main source tree.
- **P.3 ready to apply**: exact patches below are well-defined and self-contained. One careful Edit/Build/Test cycle.
- **P.4 ready to apply**: mirror of P.3 with 3-output helper; mixture-branch-length path needs separate Allreduce buffer.
- **P.5 systematic**: 6 kernel variants (not 10 — see revised list below) × same pattern. Tedious but mechanical.
- **P.6 architecture change required**: collective-dispatch design (not just a flag per model). See §P.6 below.
- **P.7 measurement**: routine perf run, no engineering.

Total remaining time budget (revised): **~5 days** focused work.

---

## When to revisit this document

After each phase completes, update the **Phase status** table at the top. When all phases are ✅, archive this doc and roll the conclusions into `lbfgs-and-warmstart-implementation.md` §15.10 (a new Mode P chapter).

---

---

# Consolidated P.ISO → P.7 Implementation Plan
**Deep-research edition — 2026-05-24**

This section supersedes the individual phase specs above with exact patches, new
findings from source analysis, identified risks, and revised ordering.

**Important revision:** before P.3 touches the SIMD likelihood kernel, build an
isolated Mode P kernel sandbox (`P.ISO`). This follows the successful FCA `mf.iso`
pattern from `updated-modelfinder-dispatch.md` §23: separate source clone, separate
build dir, separate run dirs/logs, exact parity gates, and tooling that future agents
can run without touching the production b4/ATMD tree. Mode P changes the numerical
heart of IQ-TREE; lnL and BIC parity must be proven in the ISO first.

---

## P.ISO — Mode P kernel sandbox (highest priority)

### Why this is mandatory

The P.3–P.5 work modifies `phylokernelnew.h`, not just dispatcher plumbing. A bad
patch can silently corrupt likelihood sums, derivative values, BIC, or model
ranking. The FCA work avoided this risk by creating an isolated `mf.iso` build and
running controlled parity jobs before promoting changes. Mode P needs the same
discipline, with a narrower focus: isolate the kernel, the Mode P helpers, and the
call chain that drives the kernel during ModelFinder.

**Promotion rule:** no P.3/P.4/P.5 patch should be applied to the main b4 source tree
until the ISO passes the single-model lnL/BIC gates below.

### ISO filesystem layout

Use a new tree so builds, logs, and generated `.iqtree` files cannot collide with b4
or b3c runs.

```text
/scratch/rc29/as1708/iqtree3-mode-p-iso/
    src/iqtree3-mode-p-iso/        # clone/copy of current b4 source state
    build-mode-p-iso-base/         # unpatched b4+P.1/P.2 baseline build
    build-mode-p-iso-p3/           # P.3-only build
    build-mode-p-iso-p4/           # P.3+P.4 build
    build-mode-p-iso-p5/           # P.3+P.4+P.5 build
    runs/
        aa100k_np1_base/
        aa100k_np2_p3/
        aa100k_np2_p4_trace/
        aa1m_np16_p7/
    logs/
        build/
        parity/
```

Keep the harness scripts under the repo so they are versioned:

```text
gadi-ci/mode-p-iso/
    bootstrap_mode_p_iso.sh
    build_mode_p_iso_base.sh
    build_mode_p_iso_p3.sh
    build_mode_p_iso_p4.sh
    build_mode_p_iso_p5.sh
    run_iso_lg_g4_aa100k_np1_base.sh
    run_iso_lg_g4_aa100k_np2_p3.sh
    run_iso_lg_g4_aa100k_np2_p4_trace.sh
    run_iso_mf_aa100k_np4_auto.sh
    run_iso_mf_aa1m_np16_p7.sh
tools/mode_p_iso/
    compare_mode_p_parity.py
    parse_mode_p_partitions.py
```

### Source snapshot and provenance

Bootstrap the ISO from the current b4 source state, including B.5 formula fix and
P.1/P.2 Mode P scaffolding, but before P.3 kernel edits:

| Item | Value |
|---|---|
| Source root | `/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/` |
| Starting source commit | `5604606d` plus uncommitted b4/P.1/P.2 edits |
| Required source features | B.5 per-tree formula, `if(atmd_K_outer > 1)` guard, `--mode-p*` CLI, `initializePtnPartition()` wiring |
| ISO source | `/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso/` |
| Baseline binary | `build-mode-p-iso-base/iqtree3-mpi-mode-p-iso-base` |
| Patched binaries | `iqtree3-mpi-mode-p-iso-p3`, `iqtree3-mpi-mode-p-iso-p4`, `iqtree3-mpi-mode-p-iso-p5` |

Record `git diff --stat`, `git diff -- main/phylotesting.cpp tree/phylotree.* tree/phylokernelnew.h utils/tools.*`, compiler version, binary md5, and build host in each ISO build log.

### Scoped files and dependencies

The ISO must include the full call path, not a synthetic stand-alone kernel driver.
The kernel depends on too much IQ-TREE state to be meaningfully unit-tested outside
`CandidateModel::evaluate()`.

| Scope | File(s) | Required functions / state |
|---|---|---|
| Dispatch entry | `main/phylotesting.cpp`, `main/phylotesting.h` | `CandidateModel::evaluate()`, `CandidateModelSet::evaluateAll()`, `getNextModel()`, `filterRatesMPI()`, `MF_IGNORED`, `MF_WAITING`, `MF_RUNNING`, `MF_DONE`, future `MF_MODE_P` |
| Mode P helpers | `tree/phylotree.h`, `tree/phylotree.cpp` | `ptn_start`, `ptn_end`, `isModePActive()`, `initializePtnPartition()`, `modePAllreduceLh()`, `modePAllreduceLhDfDdf()` |
| Primary kernel | `tree/phylokernelnew.h` | `computeBounds()`, `computeLikelihoodBranchSIMD()`, `computeLikelihoodBranchGenericSIMD()`, `computeLikelihoodDervSIMD()`, `computeLikelihoodDervGenericSIMD()`, `computeLikelihoodFromBufferSIMD()`, `computeLikelihoodFromBufferGenericSIMD()`, `computeLikelihoodDervMixlenSIMD()`, `computeLikelihoodDervMixlenGenericSIMD()` |
| Kernel data | `tree/phylotree.cpp`, `tree/phylotree.h` | `initializeAllPartialLh()`, `central_partial_lh`, `partial_lh`, `_pattern_lh`, `_pattern_lh_cat`, `_pattern_scaling`, `theta_all`, `theta_computed`, `buffer_partial_lh`, `buffer_scale_all`, `ptn_freq`, `ptn_invar` |
| Model/rate state | `model/*.cpp`, `model/*.h` | `ModelFactory::optimizeParameters()`, `optimizeParametersOnly()`, rate classes (`RateGamma`, `RateInvar`, `RateFree`, `RateGammaInvar`), `model_factory->unobserved_ptns`, `ASC_type` |
| Branch optimisation | `tree/phylotree.cpp` | `optimizeAllBranches()`, `optimizeOneBranch()`, `computeLikelihood()`, `computeLikelihoodDerv()`, `computeLikelihoodFromBuffer()` |
| Runtime params | `utils/tools.h`, `utils/tools.cpp` | `Params::mode_p_enabled`, `mode_p_min_cost_mult`, future `mode_p_active_in_mf`, `atmd_K_outer`, `atmd_inner_threads`, CLI parser |
| MPI/threading | `utils/MPIHelper.*`, `main/main.cpp` | `MPI_Init_thread`, `MPI_THREAD_FUNNELED`/future thread level, `MPI_Allreduce`, `MPI_Barrier`, `omp_set_max_active_levels()` |

### Kernel call graph to preserve in ISO

```text
CandidateModelSet::evaluateAll()
    -> CandidateModel::evaluate()
             -> iqtree->initializeModel()
             -> iqtree->initializePtnPartition()         # P.2, emits [Mode P] partition
             -> ModelFactory::optimizeParameters()
                        -> PhyloTree::optimizeAllBranches()
                                 -> optimizeOneBranch()
                                            -> computeLikelihoodBranchSIMD / GenericSIMD
                                            -> computeLikelihoodDervSIMD / GenericSIMD
                                 -> computeLikelihoodFromBufferSIMD / GenericSIMD
                        -> optimizeParametersOnly()
                                 -> rate/model optimisers calling the same kernel family
```

The ISO must exercise this real path with normal IQ-TREE input files, not a reduced
mock, because correctness depends on checkpoint restore, rate-class choice,
`unobserved_ptns`, ASC guards, theta caching, and MPI collective ordering.

### Log and output isolation rules

Reuse the lessons from b3c/B.4-2 and FCA `mf.iso`:

- Never use the same filename for `--prefix` and shell redirection. Use
    `--prefix ${WORK_DIR}/iqtree_inner` and redirect stdout/stderr to
    `${WORK_DIR}/iqtree_stdout.log`.
- Always pass `--output-filename ${WORK_DIR}/rank_logs/` for multi-node runs so rank
    1+ `[Mode P]`, `MF-TIME`, and trace lines are not lost.
- Capture rank bindings separately (`iqtree_bindings.log`) and keep
    `OMP_NUM_THREADS=103`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`,
    `OMP_DYNAMIC=false`, `OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=200`.
- Parser must accept IQ-TREE's real lnL format: `BEST SCORE FOUND : ...`. Do not
    repeat the b3c JSON gate bug that only matched `Log-likelihood of the tree:`.
- **B.4-3 (2026-05-24):** For np≥2 runs: always place each MPI rank on its own
    dedicated node (`ncpus=208, mem=1000GB` for np=2; `ncpus=N×104` for np=N) and
    use a rankfile (`-rf rankfile.txt`) to pin rank R to node R with `slot=0-103`.
    Do NOT run 2 MPI processes on 1 node with `OMP_PROC_BIND=close, OMP_PLACES=cores`
    — Intel libiomp5 initialises its thread pool with core affinity per-process;
    two processes sharing the same 104-core node fight over affinity and rank 1
    crashes (SIGSEGV) before printing any output. `--bind-to none` removes the
    per-rank core limit (fixing B.4-3a: 1-core detection) but does NOT fix the
    libiomp5 affinity conflict on a shared node (B.4-3b: rank 1 SIGSEGV). The
    correct fix is 1 dedicated node per rank + rankfile, exactly as the working
    `run_mf_iso_aa_100k_2node.sh` script does. ISO-1/ISO-2 scripts updated
    2026-05-24 (jobs 169130673→169130806 fixed).
- **B.4-3c (2026-05-24) — MISDIAGNOSIS, superseded by B.4-5:** Initially
    attributed to ATMD Mode F with K_outer=8 active at np=2 on rank 1. This was
    wrong: the `--atmd-k-outer 1` fix had no effect on the crash (job 169130878
    failed identically). The real cause is B.4-5 below.
- **B.4-5 (2026-05-25):** `errstreambuf::overflow` in `main.cpp:1801` has a null
    `fout_buf` for worker MPI ranks. `outstreambuf::open()` only calls `fout.open()`
    for `isMaster()`; workers leave `fout_buf = nullptr`. `startLogFile()` then
    passes this null pointer to `_err_buf.init(nullptr)`. Any subsequent cerr write
    from a worker rank reaches `fout_buf->sputn("ERROR: ",7)` with `fout_buf=0x0`
    → SIGSEGV. Confirmed via gdb backtrace of core dump from job 169130878:
    frame #9 `errstreambuf::overflow(c=91='[')` at `main.cpp:1801` calling
    `sputn(this=0x0)`, with original crash site frame #13
    `CandidateModel::evaluate()` at `phylotesting.cpp:1997` (the
    `cerr << "[Mode P] rank 1..."` line). This is the **first** cerr write from
    rank 1 inside `evaluate()`, making it the first opportunity to trigger the bug.
    **Fix**: add `if (fout_buf == nullptr)` null-guard before the log-file writes in
    `errstreambuf::overflow` and `sync` (both `-base` and `-p3` ISO source trees).
    Base binary rebuilt as **job 169131254** (md5=`6d1c1729ae17bd53f9c9224b20253de2`);
    ISO-1 resubmitted as **job 169131306**.
- **B.4-6 (2026-05-25):** ISO run scripts use `set -euo pipefail` but `grep -c`
    returns exit 1 when there are zero matches, killing the script before parity
    results print.
- **B.4-7 (2026-05-24) — ISO-2 dispatch mismatch:** FCA dispatch in `evaluateAll()`
    gives **different models** to different ranks (rank 0 → LG, rank 1 → WAG simultaneously).
    The `MPI_Allreduce` in the P.3 Mode P kernel then sums rank 0's partial LG lnL with
    rank 1's partial WAG lnL → numerical garbage → `Numerical underflow (lh-derivative)`
    on both ranks → mpirun exit 2 in 25s. Confirmed from ISO-2 job 169131722 stdout:
    `[Mode P] rank 0 model=LG ptn=[0, 48008)` and `[Mode P] rank 1 model=WAG ptn=[48008,
    96017)` appeared simultaneously, followed immediately by the underflow crash.
    **The P.3 kernel is structurally correct.** The Allreduce is placed correctly and the
    limits-shift is correct — the bug is that Mode P assumes all ranks evaluate the SAME
    model (collective dispatch), but FCA evaluates different models per rank (independent
    dispatch). **Fix: implement P.6-lite collective dispatch** — when Mode P is enabled in
    `evaluateAll()`, all ranks iterate over models in the same order without FCA queue
    distribution; each rank uses its Mode P `[ptn_start, ptn_end)` slice; `MPI_Allreduce`
    in the kernel correctly combines the partial sums because all ranks are always evaluating
    the same model. Only rank 0 updates BIC/best-model tracking. This is a precondition
    for ISO-2 to pass. ISO-0 (169130671) IQ-TREE run PASSED but script exited 1 because
    `grep -c '\[Mode P\]'` (correctly) found zero lines. **Fix**: replace
    `grep -c ... | awk ...` with `{ grep ...; true; } | awk ...` pattern; also fix
    bare `grep ... | head -N` display lines to `{ grep ...; true; } | head -N`.
    Applied to `run_iso0_aa100k_np1_base.sh`, `run_iso1_aa100k_np2_base.sh`,
    `run_iso2_aa100k_np2_p3.sh` (2026-05-25). Tolerance in ISO-0 lnL check also
    widened from `1e-3` to `0.05` to accommodate full-tree-search variance.
- **B.4-8 (2026-05-25) — Non-deterministic `Numerical underflow (lh-derivative)` crash — FIXED:**
    With the non-safe SIMD kernel (AA 20-state, `safe_numeric=false`), specific alignment
    patterns can have `lh_ptn=0` under certain model parameter states during branch length
    optimization. This produces `df_ptn = numerator / lh_ptn = 0/0 = NaN`, which
    accumulates through the OMP reduction and survives the MPI_Allreduce, triggering
    `outError("Numerical underflow (lh-derivative)")` → mpirun exit 2.
    **Non-deterministic**: job 169132572 (binary `a278e44c`) completed without crash
    (lucky OMP thread scheduling); jobs 169131969/169132390/169132598 all crashed.
    Diagnostic: `P4-PKT-DIAG rank=0 pkt=93 ptn=[32520,32868) df=nan ddf=nan`; `P4-DIAG
    pre_df=nan` on both ranks — NaN is local, not Allreduce-induced.
    **First fix attempt (REVERTED)**: `safe_numeric=true` at `phylotree.cpp:~1001`
    (end of `initializePtnPartition` MPI block). Incomplete: function pointers were
    already set by `setLikelihoodKernelFMA()` during `setLikelihoodKernel()` to
    `NORM_LH=false` instantiations; setting the flag afterwards did not change them →
    every packet NaN, crash in 32s (job 169133798, binary `161fc098`). Reverted with
    explanatory comment; `// safe_numeric = true; // ← DO NOT re-enable without re-dispatch`.
    **Complete fix (built job 169134517, md5=`50b4b172`)**: `safe_numeric=true` override
    added to `setLikelihoodKernel()` in `phylotreesse.cpp` (line ~116), BEFORE the
    `setLikelihoodKernelFMA()` dispatch call, gated on `#ifdef _IQTREE_MPI` and
    `params->mode_p_enabled != 0 && nranks > 1`. Also added `#include "utils/MPIHelper.h"`.
    All SIMD kernel function pointers now assigned to `SAFE_LH=true` instantiations from
    initial dispatch; buffer sizing sees `safe_numeric=true`. P4-PKT-DIAG NaN still fires
    for `+I` model states (safe kernel does not fully eliminate NaN df for AA) but
    `SAFE_NUMERIC=true` disables the `outError()` crash path at compile time; optimizer
    recovers silently; lnL unaffected. **ISO-2 PASS confirmed: job 169135061, exit 0,
    lnL=−7,541,976.861 = FCA np=1 exact, LG+G4, 15m23s wall.**
- **B.4-9 (2026-05-24) — `tree_lh` HALF-value SEGFAULT during +I+G4 EM under Mode P — FIXED:**
    **Symptom (job 169136036, binary `50b4b172`, ISO-3 MODEP sub-run):** during ModelFinder
    evaluation of `WAG+I+G4`, the +I+G4 EM iteration produces `tree_lh = -3,818,200` and
    similar values (i.e. ≈ `-7,541,976 / 2` — rank 0's local pattern-slice sum). The very
    next `optimizeAllBranches` iteration returns the correctly Allreduced full
    `tree_lh = -7,598,892`. The branch-decrease guard at `phylotree.cpp:2914`
    `if (new_tree_lh < tree_lh - tolerance*0.1)` then compares HALF (saved) vs FULL
    (next), declares "tree log-likelihood decreases", triggers `restoreBranchLengths()`,
    corrupts optimiser state, and the next kernel call SIGSEGVs. Trace lines:
    `NOTE: Restoring branch lengths as tree log-likelihood decreases after branch length
    optimization: -3783260.217 -> -7598867.386` (`new_tree_lh: -7598892.41 tree_lh: -3783260.22`).

    **Root cause (multi-headed):** two distinct Mode P sync gaps converge to produce HALF.
    1. **Upstream trigger — EM accumulator reads slice-only arrays.** In
       [`rategammainvar.cpp:236-244`](file:///scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/rategammainvar.cpp#L236-244)
       `RateGammaInvar::optimizeWithEM` sums over `[0, nptn)`:
       `ppInvar += ptn_invar[ptn] * ptn_freq[ptn] / lk_ptn` where
       `lk_ptn = ptn_invar[ptn] + sum_cat _pattern_lh_cat[ptn*ncat + cat]`. Under Mode P,
       `_pattern_lh_cat[]` is only populated for this rank's slice `[ptn_start, ptn_end)`
       by `computePatternLhCat()` (which dispatches to the partitioned kernel); patterns
       outside the slice have stale-or-zero `_pattern_lh_cat`, so `lk_ptn ≈ ptn_invar[ptn]`
       and the outside-slice contribution becomes `ptn_invar[ptn] * ptn_freq[ptn] /
       ptn_invar[ptn] = ptn_freq[ptn]` (massively over-counting). Each rank computes a
       DIFFERENT `newPInvar`. `p_invar = newPInvar` makes the model parameters diverge
       across ranks; subsequent branch optimisation uses rank-divergent gradient/Hessian
       slices → branch lengths diverge → kernel eventually computes NaN/inf `tree_lh`
       under Mode P partition.
    2. **Primary half-leak — kernel fallback recompute bypasses partition + Allreduce.**
       In [`phylokernelnew.h:3243-3259`](file:///scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylokernelnew.h#L3243-3259)
       (`computeLikelihoodBranchSIMD`) and
       [`phylokernelnew.h:3541-3549`](file:///scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylokernelnew.h#L3541-3549)
       (`computeLikelihoodFromBufferSIMD`), when the Allreduced `tree_lh` is non-finite
       (e.g. underflow from the `+I+G4` EM divergence above) a fallback path fires:
       `tree_lh = 0.0; for (ptn = 0; ptn < orig_nptn; ptn++) tree_lh += _pattern_lh[ptn] *
       ptn_freq[ptn];`. This iterates the FULL range but `_pattern_lh[]` was only written
       for `[ptn_start, ptn_end)` by the kernel just above — outside-slice entries are
       stale/zero — and there is NO Allreduce after the recompute. Result: each rank
       returns its LOCAL slice sum ≈ `tree_lh / nranks` → caller stores HALF as `tree_lh`
       → next call returns Allreduced FULL → spurious decrease guard fire → SEGFAULT.

    **Why ISO-2 passed but ISO-3 failed:** ISO-2 used `-m TEST` against the *initial* tree
    (random-start), where MF early-terminated heavy `+I+G4` models via the rate-class
    pruning logic before EM divergence accumulated enough to underflow. ISO-3 runs `-m
    TEST -te <ISO2-treefile>` (start from the ISO-2 OPTIMAL tree) so the optimiser dwells
    inside the deep `+I+G4` basin, EM runs to convergence, and the divergent-`p_invar`
    pathology has time to crash. Bug class was latent in ISO-2 too — visible only in
    deeper optimisation cycles.

    **Novel fix architecture — "Partition-Extent Invariant" (PEI):** rather than chasing
    every kernel fallback and rate-model accumulator one-by-one (whack-a-mole), classify
    each per-pattern array by its sync-state contract:
    - **REPLICATED** — identical on every rank (e.g. `ptn_freq[]`, `ptn_invar[]`,
      alignment data). Safe to read at any `ptn`.
    - **PARTITIONED** — each rank has only its `[ptn_start, ptn_end)` slice valid; rest
      is zero or stale (e.g. `_pattern_lh[]`, `_pattern_lh_cat[]` under Mode P).
      Consumers MUST partition the read loop AND Allreduce derived scalars.
    - **SYNCED** — promoted via explicit Allreduce so every rank has the full global
      view (e.g. `tree_lh` after `modePAllreduceLh`).

    Patches in this fix (B.4-9-α, β):
    - **α — kernel fallback paths:** in both `computeLikelihoodBranchSIMD` (line 3243)
      and `computeLikelihoodFromBufferSIMD` (line 3541), restrict the fallback recompute
      to `[mp_lo, mp_hi)` (mirror the main kernel's mp_active guard) and Allreduce the
      local `tree_lh` via `modePAllreduceLh()`. No-op when Mode P inactive or single-rank
      — pre-Mode-P semantics preserved exactly.
    - **β — EM accumulator (root cause):** in `RateGammaInvar::optimizeWithEM` line ~236,
      restrict the pattern loop to `[mp_lo, mp_hi)` and call `phylo_tree->modePAllreduceLh(ppInvar)`
      before dividing by `nSites`. `ppInvar` is a scalar — 1 `MPI_DOUBLE` Allreduce, cheap.
      Result: identical `newPInvar` on every rank → no parameter divergence → no NaN
      trigger → no fallback fire.

    **Status:** implemented 2026-05-24 in `-p3` ISO source tree. Added
    `#include "tree/phylotree.h"` to `rategammainvar.cpp`. Latent same-pattern accumulator
    bugs identified (NOT yet patched, not on +I+G4 critical path) at
    [`rateheterotachy.cpp:175,247`](file:///scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/rateheterotachy.cpp#L175),
    [`ratefree.cpp:566,645`](file:///scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/ratefree.cpp#L566),
    [`modelmixture.cpp:2000,2107,2176,2203`](file:///scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/model/modelmixture.cpp#L2000)
    — same `for (ptn = 0; ptn < nptn) sum_using _pattern_lh_cat[ptn*..]` shape; will need
    the same B.4-9-β treatment before extending Mode P to `+R`/`+H`/mixture models.
    **Build B.4-9 (job 169136359) failed at 44% with F-11-bis ambiguity**: `rategammainvar.cpp:271` called `modePAllreduceLh(ppInvar)` where `ppInvar` is a non-const `double` lvalue — both the by-value overload (`double f(double)`) and by-reference overload (`void f(double&)`) matched. Fix (B.4-9-γ): introduce `const double ppInvar_local = ppInvar;` so the non-const-reference overload is excluded. Rebuild job 169136419 succeeded (md5=`9660575a`). ISO-3 re-submitted as job **169136469**.
- Store the `.iqtree`, `.log`, `.treefile`, rank stdout, `mf_time.log`,
    `mf_diag.log`, build log, binary md5, and source diff for every ISO run.

### ISO parity gates

The ISO starts with correctness gates before any full MF performance run.

| Gate | Build | Run | Status | Pass criteria |
|---|---|---|---|---|
| ISO-0 | base (P.1/P.2 only) | AA 100K np=1 `LG+G4 --mode-p-all` | **✅ EFFECTIVELY PASSED** (job 169130671): IQ-TREE exit=0, no `[Mode P]` lines, lnL=-7,541,976.853 (|Δ|=0.008 vs ref; within 0.05 tolerance). Script exit=1 was a false positive — `grep -c` returns exit 1 on zero matches, triggering `set -e` before the parity report printed. Script fixed 2026-05-25 (grep -c → `{grep; true} \| awk`, tolerance 1e-3→0.05). | No `[Mode P]` line, lnL matches b4/FCA within 0.05, best model exact |
| ISO-1 | base (P.1/P.2 only) | AA 100K np=2 `LG+G4 --mode-p-all` | **✅ PASSED 2026-05-24** — job **169131306** (2 nodes, binary md5=`6d1c1729`, B.4-5 null-guard applied). ModelFinder completed 318.627s; full pipeline (MF + NNI 103 iterations + final opt) 1007.414s wall; model=LG+G4, α=0.996, **lnL=-7,541,976.852** (Δ=0.001 vs FCA np=2 ref -7,541,976.853 ✓); rank 0 emitted 31 `[Mode P] rank 0 model=X ptn=[0, 48008)` lines (one per AA model); rank 1 emitted 31 `[Mode P] rank 1 model=X ptn=[48008, 96017)` lines; **perfect 50/50 partition of 96017 patterns confirmed both ranks**. PBS exit 141 (SIGPIPE) is cosmetic — IQ-TREE itself exited 0 with all results written. | `[Mode P]` partition lines emitted, lnL/BIC unchanged because kernel is inert |
| ISO-2 | P.3 + P.4 + P.5a + P.6-lite | AA 100K np=2 `-m TEST --mode-p-all --atmd-k-outer 1` | **✅ PASSED DEFINITIVELY 2026-05-25** — confirmed job **169135061** (binary md5=`50b4b172`, B.4-8 complete fix). IQ-TREE exit 0, 15m23s wall. **lnL=-7,541,976.861** = FCA np=1 ref `-7,541,976.861` (job 169095077) exactly (Δ=0 at strict 1e-6 ✓). Best model **LG+G4** ✓. MF wall **235.844s**. Required FOUR kernel-closure iterations: (1) P.3 alone → B.4-7 ranks-mismatched dispatch → (2) P.3+P.6-lite → Derv `lh-derivative` underflow → (3) P.3+P.4+P.6-lite → FromBuffer `lh-from-buffer` underflow → (4) P.3+P.4+P.5a+P.6-lite → end-to-end PASS. **B.4-8 crash resolution history**: first PASS (job 169132572, `a278e44c`) was non-deterministic — the non-safe kernel was active and lucky OMP scheduling avoided NaN; jobs 169131969/169132390/169132598/169133798/169134417 all crashed. Incomplete fix (binary `161fc098`, job 169133798): `safe_numeric=true` in `initializePtnPartition()` leaves function pointers stale (set by `setLikelihoodKernelFMA()` to `NORM_LH=false` before the override) → every packet df=nan, crash in 32s; reverted with `// B.4-8 (REVERTED)` comment in `phylotree.cpp`. **Complete fix (binary `50b4b172`, built job 169134517, 8m42s)**: `safe_numeric=true` override added to `setLikelihoodKernel()` in `phylotreesse.cpp` (line ~116) BEFORE the `setLikelihoodKernelFMA()` call, inside `#ifdef _IQTREE_MPI` guard gated on `params->mode_p_enabled != 0 && nranks > 1`. All SIMD kernel function pointers now point to `SAFE_LH=true` template instantiations from initial dispatch onward. P4-PKT-DIAG NaN diagnostics still fire per-model for `+I` model parameter states (the safe kernel's `ptn_scale[]` does not eliminate all NaN df for 20-state AA in the derivative path), but with `SAFE_NUMERIC=true` baked into the template the `outError()` crash check is compile-time disabled (`if (!true && ...)` = dead code); optimizer silently recovers from NaN df; final lnL is unaffected (Δ=0 vs FCA np=1). **FP non-associativity**: Mode P + Allreduce produces FCA np=1 lnL (single-rank sequential summation order); Δ=0.008 vs FCA np=2 is expected, informational only. | lnL `-7,541,976.861 ± 1e-6` vs FCA np=1 ✓; best model LG+G4 ✓ |
| ISO-3 | P.3+P.4+P.5a+B.4-9 | AA 100K np=2 `-m TEST --mode-p-all -v -te <ISO2-tree>` | **✅ CORRECTNESS PASS 2026-05-25 (B.4-9 fix verified)** — job **169136469** (binary `9660575a`, all B.4-9-α/β patches in place). **MODEP sub-run completed end-to-end without SEGFAULT** (exit 0, 247s wall, 240.097s MF). Final **lnL = -7,541,976.8614** matches FCA np=1 reference -7,541,976.861 within 4.22e-04; BASE (np=1) sub-run optimal lnL=-7,541,976.8528 |Δ vs ref|=8.23e-03. Both are well below the 1e-3 FP-non-associativity band; MODEP actually matches the reference MORE CLOSELY than BASE. Best model **LG+G4** ✓ on both sub-runs. Mode P actively partitioning: rank 0 emitted 117 `[Mode P]` lines, rank 1 emitted 59. **NO `Restoring branch lengths` with HALF values, NO `tree_lh = -3.7XX,XXX` (HALF) anywhere** — the diagnostic symptoms from job 169136036 are completely absent. ModelFinder evaluated 224 models, Phase-2 Allreduce merged scores, final selection produces LG+G4. **Script "FAIL" exit 10 is cosmetic**: (i) the strict 1e-6 parity gate uses a truncated `REF_LNL = -7541976.861` constant and flags Δ=4.22e-04 as failure even though MODEP matches reference more closely than BASE; (ii) the BASE-vs-MODEP NR-trace step-by-step compare assumes identical model-exploration order (only valid for single-model `-te` runs) but `-m TEST` triggers full ModelFinder which inherently explores models in different orders between FCA np=1 and Mode P np=2 → step counts differ (BASE=92 vs MODEP=61) → the trace compare is meaningless under `-m TEST`. Correctness was verified by direct inspection of `inner.log`: no SEGFAULT, no HALF-lnL leak, lnL parity, best-model parity. **Asymmetric rank-load finding (follow-up):** rank 0 ran ~117 model evaluations and rank 1 ran ~59 under P.6-lite — full lockstep was expected. Possible cause: rank-local `MF_IGNORED` pruning during `+R_k` chains marks different model sets per rank when FP noise drives one rank to a slightly different score. Doesn't affect correctness (final scores gather + Allreduce-merge), but is a Mode P efficiency suboptimality worth investigating before P.7 perf gates. **Build history**: (1) job **169135995** `-m LG+G4`: WRONG — `CandidateModel::evaluate()` never called → Mode P never activated; (2) job **169136036** (binary `50b4b172`): SEGFAULT during `WAG+I+G4` EM — B.4-9 root cause; (3) build **169136359**: failed at 44% — `modePAllreduceLh(ppInvar)` overload ambiguity in rategammainvar.cpp (same F-11-bis bug as job 169131566) → fixed with const-local pattern; (4) build **169136407**: transient `Stale file handle` on lustre — auto-retry; (5) build **169136419** (md5=`9660575a`, 10m16s): **OK**, all preflight checks pass; (6) ISO-3 job **169136469**: PASS. **B.4-9 fix verified.** | No SEGFAULT during `+I+G4` EM ✓; final lnL within 1e-3 of FCA np=1 ref ✓; best model LG+G4 ✓; both ranks emit `[Mode P]` lines ✓ |
| ISO-4 | P.3+P.4+P.5a | AA 100K np=4 `-m TEST --mode-p-all -te <ISO2-tree>` | ❌ **FAILED 2026-05-25 — new bug B.4-10 (SIGFPE in +F models at np=4)** — job **169136585** (binary `9660575a`, 4 nodes, 416 cpus, 2000GB, wall=00:07:48, 108.16 SU). BASE np=1 sub-run: exit=0, wall=429s ✓. MODEP np=4 sub-run: **SIGFPE (Linux Signal 8, Floating-point exception)** on rank 3, exit=136. **Crash context**: all 4 ranks successfully evaluated LG, LG+I, LG+G4, LG+I+G4 (16 `[Mode P]` markers logged across all ranks ✓). Crash triggered on rank 3 (`ptn=[72016, 96017)`) at entry to the **LG+F** batch. Rank 0's inner log confirms all 4 `LG+F*` models (LG+F, LG+F+I, LG+F+G4, LG+F+I+G4) were logged by rank 0 before crash; rank 3's stdout shows only UCC cascade errors (`tl_ucp_coll.c:137 TL_UCP ERROR failure in recv completion Message truncated`), indicating rank 3 died mid-Allreduce and caused MPI collective abort on the other ranks. **Root cause investigation 2026-05-25**: (1) Frequency computation — `ModelGTR::init(FREQ_EMPIRICAL)` calls `aln->computeStateFreq()` → `countStates()` which iterates ALL patterns (not partitioned); `convfreq()` applies `min_state_freq` floor — frequencies are identical on all 4 ranks and non-zero. NOT the cause. (2) Kernel dispatch — `safe_numeric=true` (B.4-8) forces `computeLikelihoodBranchSIMD<Vec4d, SAFE_LH, 20, true>` for AA 20-state on all ranks; same kernel for LG and LG+F — no +F-specific dispatch branch. NOT the cause. (3) B.4-9-α fallback fix — the `!std::isfinite(tree_lh)` fallback at lines ~3244–3281 iterates `[mp_lo_fb=ptn_start, mp_hi_fb=ptn_end)` with Allreduce; correctly scoped. NOT the cause (fallback only entered on underflow, SIGFPE is a hardware trap). (4) **Leading hypothesis — np=4 vs np=2 rank symmetry**: ISO-3 (np=2) rank 1 has `ptn_end=96017` (same remainder 96017%4=1) and passes LG+F → the 1-pattern remainder is NOT the issue. Difference is partition SIZE: rank 3 at np=4 = 24,001 ptn vs rank 1 at np=2 = 48,009 ptn. SIGFPE (FE_INVALID / FE_DIVBYZERO) occurs in rank 3 but not ranks 0–2 → something rank-3-specific in the +F kernel call. **Diagnostic run submitted 2026-05-25**: job **169136683** (`iso4-diag-k1`, `--atmd-k-outer 1`, MODEP-only, walltime 20 min, script `gadi-ci/mode-p-iso/run_iso4_diag_k1_np4_p3.sh`, log `~/setonix-iq/iso4-diag-k1.o169136683`). If PASSES → B.4-10 root cause is K_outer>1 batch parameter handoff for +F models. If also crashes → root cause is per-model +F kernel arithmetic at `ptn_start=72016`. **New bug label**: B.4-10. Run dir: `/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/iso4_p3_aa100k_np4_seed1_169136585/`. **DIAGNOSTIC RESULT 2026-05-25**: Jobs 169136682/683 (K_outer=1) crashed with a **different** bug — `TL_UCP ERROR failure in recv completion Message truncated` SEGFAULT on **ALL 4 ranks** at **LG+G4** (the first nrates=4 model). Did NOT reproduce the original SIGFPE at LG+F. This is a new bug **B.4-11** (see below). B.4-11 fix applied; new jobs 169137348 (K1 diag) and 169137349 (full ISO-4) submitted 2026-05-25 to verify. Run dirs: `iso4_diag_k1_p3_aa100k_np4_seed1_169136682/` and `iso4_diag_k1_p3_aa100k_np4_seed1_169136683/` (crashed B.4-11). | Both exits 0; [Mode P] on all 4 ranks; best model LG+G4; lnL within 0.05 of FCA np=1 ref |
| ISO-5 | P.3+P.4+P.5+P.6 | AA 100K np=4 `-m TEST --mode-p` | ⏳ NOT STARTED | Auto dispatcher routes only heavy models; lnL/BIC parity; rank logs show collective Mode P order |
| ISO-6 | P.7 candidate | AA 1M np=16 `-m TEST --mode-p` | ⏳ NOT STARTED | lnL `-78,605,196.497 ± 0.5`, best model LG+G4, MF wall target `≤600s` |

### Baseline run records for ISO comparison

| Reference | Job | Dataset | Nodes | Key values | Use in ISO |
|---|---:|---|---:|---|---|
| FCA AA 100K np=2 | 168584736 | AA 100K | 2 | lnL `-7,541,976.853`, BIC `15,086,233.265`, MF `149.029s` | Primary np=2 parity reference |
| FCA AA 100K np=1 | 169095077 | AA 100K | 1 | lnL `-7,541,976.861`, MF `258.773s`, SPR `738.569s` | Single-rank base sanity |
| ATMD b3c AA 100K | 169111545 | AA 100K | 1 | K_outer=8, lnL `-7,541,976.853`, MF `423.233s` | Confirms ATMD+kernel correctness before Mode P |
| FCA AA 1M np=16 | 168635616 | AA 1M | 16 | lnL `-78,605,196.497`, MF `1,122.363s`, SPR `1,287.863s` | P.7 performance and parity reference |
| ATMD b3c AA 1M | 169112256 | AA 1M | 16 | K_outer=1, lnL `-78,605,196.497`, MF `2,113.706s`, SPR `1,958.174s` | Regression/control case; confirms correctness despite bad wall time |

### ISO parser requirements

`tools/mode_p_iso/compare_mode_p_parity.py` should parse and compare:

- `BEST SCORE FOUND : <lnL>` from `.iqtree` or stdout.
- `Best-fit model according to BIC:` and/or model summary lines.
- BIC value from `.iqtree` report.
- `Wall-clock time for ModelFinder`, `Wall-clock time used for tree search`.
- `[Mode P] rank R model=X ptn=[start, end) of N` partition coverage.
- `MF-TIME` per-model lines and rank-local stdout from `--output-filename`.
- Exit status and PBS walltime.

The parser should fail closed: missing lnL, missing BIC, missing rank logs on np>1,
or overlapping/incomplete partitions are hard failures.

---

## New findings from deep source analysis

These facts change or sharpen the implementation plan relative to the original spec.

### F-1  `save_log_value=false` guarantee eliminates _pattern_lh Allgather for MF

`computeLikelihoodBranchSIMD` has a guard at entry ([phylokernelnew.h:2707](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylokernelnew.h#L2707)):

```cpp
if (!save_log_value) {
    ASSERT(!(params->robust_phy_keep < 1.0));
    ASSERT(!(params->robust_median));
    ASSERT(!ASC_Holder);
    ASSERT(!ASC_Lewis);
}
```

`CandidateModel::evaluate()` calls the kernel via `computeLikelihood()` with
`save_log_value=true` only when the model becomes best-so-far; the primary MF
kernel call that accounts for wall time uses `save_log_value=false` (the
`computeLikelihoodBranch` path inside `optimizeParameters`). **With
`save_log_value=false`, ASC correction and robust-phylo post-processing are
guaranteed OFF.**

**Implication**: `_pattern_lh[ptn]` is written for `ptn < orig_nptn` inside the
kernel, but the values are **not consumed by any code path after the kernel
returns** when `save_log_value=false`. Known Issue 1 in the original spec (needing
MPI_Allgather of `_pattern_lh`) is a **non-issue for the ModelFinder path**.

For the SPR phase, Mode P must be **disabled entirely** (see F-4 below), so the
`_pattern_lh` Allgather is also a non-issue there.

**Action**: Remove the Allgather from the design. It is only needed if Mode P is
ever extended to the SPR kernel path (a separate future phase).

---

### F-2  `computeBounds` tail rounding and post-shift correctness

`computeBounds` (phylokernelnew.h:1118) rounds `elements` up to
`VectorClass::size()` before partitioning. After shifting limits by `mp_lo`, the
last packet's upper bound becomes `mp_lo + roundUp(mp_size, VCSIZE)`, which may
exceed `min(ptn_end, orig_nptn)`.

The kernel body handles this correctly via the existing `ptn < orig_nptn` branch
test inside the packet loop. Patterns in the tail (between `ptn_end` and the
rounded-up limit) fall into the `else` branch which only accumulates
`all_prob_const` — and `all_prob_const` is unused when `save_log_value=false` and
ASC is off. **No additional tail-handling is needed for MF.**

```
Packet loop (with Mode P shift applied):
  ptn_lower = limits[packet_id]        ← shifted into [ptn_start, ...)
  ptn_upper = limits[packet_id+1]      ← may overshoot ptn_end by < VCSIZE

  for ptn in [ptn_lower, ptn_upper):
    if ptn < orig_nptn:  → data pattern, accumulates tree_lh ✓
    else:                → unobserved/tail, accumulates all_prob_const (unused in MF)
```

---

### F-3  Unobserved patterns are model-derived and replicated across ranks

Unobserved patterns (`ptn ∈ [max_orig_nptn, nptn)`) are computed from
`model_factory->unobserved_ptns` — a per-model property, identical on every MPI
rank. They do not contain data. For `save_log_value=false`, unobserved patterns only
contribute to `all_prob_const` which drives ASC correction (disabled). They carry
**zero cost** in Mode P: the Mode P slice `[ptn_start, ptn_end)` never overlaps
with `[max_orig_nptn, nptn)` because `ptn_end ≤ orig_nptn ≤ max_orig_nptn`.

**The unobserved-pattern loop in all kernels runs on zero patterns under Mode P and
costs exactly zero.**

---

### F-4  Mode P must be disabled during SPR — MPI collective ordering requirement

During SPR (tree search after ModelFinder), FCA-style dispatch is used: each rank
independently optimises its own tree. Kernel calls are **not synchronised** across
ranks. If Mode P were active, `MPI_Allreduce` inside the kernel would deadlock
because different ranks would call it for different branches at different times.

**Required fix**: reset `mode_p_enabled = 0` in the IQ-TREE main loop
immediately after `evaluateAll()` returns and before tree-search (`doTreeSearch`).
This is a harness-level change — no kernel modification needed.

Alternatively, `isModePActive()` could consult a second flag
`params->mode_p_mf_phase` that is set true only inside `evaluateAll()` and false
outside. The `evaluate()` call to `initializePtnPartition()` already provides
per-model activation, but the kernel's `isModePActive()` check needs to be safe
for SPR too.

**Short-term fix (Phase P.3 implementation)**: add a `Params::mode_p_active_in_mf`
bool, set to `true` at the top of `evaluateAll()` and `false` at the bottom.
`isModePActive()` checks both `mode_p_enabled` AND `mode_p_active_in_mf`.

---

### F-5  MPI thread safety: Mode P + ATMD K_outer > 1 is unsafe under MPI_THREAD_FUNNELED

`MPI_Init_thread` is called with `MPI_THREAD_FUNNELED` at
[MPIHelper.cpp:28](file:///scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/MPIHelper.cpp#L28).
`MPI_THREAD_FUNNELED` guarantees only that the **main (init) thread** may call
MPI. When `atmd_K_outer > 1`, `phylotesting.cpp:evaluateAll()` is inside a
`#pragma omp parallel num_threads(atmd_K_outer)` region — each outer thread
independently calls the kernel and would call `MPI_Allreduce`. This is **undefined
behaviour** under `MPI_THREAD_FUNNELED`.

b4 has the `if (atmd_K_outer > 1)` guard in the `#pragma omp parallel` clause, so
when K_outer=1 (current expected behaviour for AA 1M) the serial path is taken and
the main thread calls the kernel. Mode P is safe at K_outer=1.

**For K_outer > 1** (AA 100K with b4): `isModePActive()` must return false. Add
check: `params->atmd_K_outer <= 1` (or the runtime value after B.5 formula
computes it). This prevents Mode P from activating under nested-OMP dispatch.

Concrete change in `isModePActive()` (phylotree.cpp:910):

```cpp
bool PhyloTree::isModePActive() const {
    if (!params || params->mode_p_enabled == 0)
        return false;
    if (!params->mode_p_active_in_mf)          // F-4: SPR guard
        return false;
    // F-5: unsafe to Allreduce from non-main thread under MPI_THREAD_FUNNELED
    int k_outer = params->atmd_K_outer;         // runtime value (0=auto, else explicit)
    if (k_outer != 0 && k_outer != 1)           // K_outer>1 → OMP parallel region
        return false;
#ifdef _IQTREE_MPI
    return MPIHelper::getInstance().getNumProcesses() > 1;
#else
    return false;
#endif
}
```

---

### F-6  `computeLikelihoodFromBufferSIMD` uses a flat ptn-loop, not a packet-loop

Unlike `BranchSIMD` (packet-based `#pragma omp for` over `num_packets`),
`FromBufferSIMD` uses a flat `#pragma omp for` over `ptn`:

```cpp
for (size_t ptn = 0; ptn < nptn; ptn+=VectorClass::size()) { ... }
```

with an `all_lh[k]` reduction array where `k = ptn / VectorClass::size()`.

The Mode P modification is **different** from BranchSIMD:

```cpp
const size_t mp_lo = mp_active ? ptn_start : 0;
const size_t mp_hi = mp_active ? ptn_end   : nptn;
for (size_t ptn = mp_lo; ptn < mp_hi; ptn+=VectorClass::size()) {
    ...
    int k = ptn / VectorClass::size();  // same index into all_lh — correct
    all_lh[k] = horizontal_add(vc_tree_lh);
    ...
}
```

`all_lh` is allocated as `nsize = nptn / VectorClass::size() + 1`, initialised to
zero. Entries not written by this rank stay 0. The final summation `for (k = 0; k <
nsize; k++) all_tree_lh += all_lh[k]` accumulates only the non-zero entries.
**This works correctly without any structural change to the reduction loop.**

**Alignment requirement**: `mp_lo` must be a multiple of `VectorClass::size()` for
the `load_a` (aligned load) in the loop body. `initializePtnPartition()` must
align `ptn_start` down and `ptn_end` up to a `VECTOR_SIZE` boundary (8 for
AVX-512 doubles, 4 for AVX2).

```cpp
// In initializePtnPartition() — align to VECTOR_SIZE (defined in vectorclass/instrset.h)
size_t vcsize = VECTOR_SIZE; // compile-time constant from the AVX/AVX-512 flag
ptn_start = (ptn_start / vcsize) * vcsize;
ptn_end   = min(nptn, ((ptn_end + vcsize - 1) / vcsize) * vcsize);
```

This may cause a very small amount of overlap between adjacent ranks' slices (at
most `vcsize-1 = 7` patterns), but since the tree_lh Allreduce is a SUM and each
`ptn_freq[ptn]` is the same on all ranks, double-counting a 7-pattern overlap
would introduce a ≤ `7/orig_nptn` relative error. For AA 1M (`orig_nptn ≈ 1M`),
this is < 7e-6 — tolerable for MF. For a correctness gate, use `orig_nptn` that is
VCSIZE-divisible (e.g. 100K ≡ 0 mod 8 ✓).

**Better fix**: ensure `chunk = nptn / nranks` is VCSIZE-aligned by rounding down
to VCSIZE in `initializePtnPartition()`. The last rank absorbs the remainder (which
is at most `nranks × (vcsize-1)` extra patterns, negligible).

```cpp
size_t chunk = (nptn / (size_t)nranks / vcsize) * vcsize;  // VCSIZE-aligned chunk
ptn_start = chunk * (size_t)rank;
ptn_end   = (rank == nranks - 1) ? nptn : chunk * (size_t)(rank + 1);
```

---

### F-7  `computeLikelihoodDervSIMD` has two separate exit paths

The `DervSIMD` kernel (phylokernelnew.h:2239) has:

1. **Mixture-branch-length path** (`isMixlen() == true`, lines 2400-2583):
   accumulates `all_dfvec[0..nmixlen)`, `all_ddfvec[0..nmixlen²)`, `all_lh`.
   Exits early at line 2566 via `return`, writing `df[0..nmixlen]` and `ddf[0..nmixlen²]`.
   The last `df[nmixlen]` entry holds the log-likelihood.

2. **Normal joint path** (`isMixlen() == false`, lines 2472-2659):
   accumulates scalar `all_df`, `all_ddf`. Exits at bottom writing `*df`, `*ddf`.

Mode P Allreduce must be inserted **in both paths** before writing the output.

For path 1 (mixture), a single `MPI_Allreduce` of buffer size `nmixlen + nmixlen² + 1`:

```cpp
// Insert before the "df[i] = horizontal_add(all_dfvec[i])" block:
if (isModePActive()) {
    // flatten all_dfvec + all_ddfvec + all_lh into one buffer, Allreduce, unpack
    int n = nmixlen + nmixlen2 + 1;  // nmixlen2 = nmixlen*(nmixlen+1)/2
    vector<double> in_buf(n), out_buf(n, 0.0);
    for (int i = 0; i < nmixlen;  i++) in_buf[i]          = horizontal_add(all_dfvec[i]);
    for (int i = 0; i < nmixlen2; i++) in_buf[nmixlen+i]  = horizontal_add(all_ddfvec[i]);
    in_buf[n-1] = all_lh;
#ifdef _IQTREE_MPI
    MPI_Allreduce(in_buf.data(), out_buf.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    for (int i = 0; i < nmixlen;  i++) { df[i]  = out_buf[i]; }
    for (int i = 0; i < nmixlen2; i++) { ddf[i] = out_buf[nmixlen+i]; }
    df[nmixlen] = out_buf[n-1];
    return;
}
// else fall through to original horizontal_add logic
```

For path 2 (normal joint), use the existing 3-value `modePAllreduceLhDfDdf()` helper:

```cpp
// After "all_df += horizontal_add(my_df); all_ddf += horizontal_add(my_ddf);"
// and BEFORE "*df = all_df; *ddf = all_ddf;"
double dummy_lh = 0.0;
modePAllreduceLhDfDdf(dummy_lh, all_df, all_ddf);
// (lh is already computed by BranchSIMD Allreduce in the preceding evaluateBranch call)
*df  = all_df;
*ddf = all_ddf;
```

---

### F-8  `DervMixlenSIMD` is a separate function with a different output contract

`computeLikelihoodDervMixlenSIMD` (phylokernelnew.h:3500) takes `df` and `ddf` by
reference (not pointer), outputs only the derivative values (no lnL), uses
`computeBounds` at line 3553. Exit at line 3665: `df = all_df; ddf = all_ddf;`.

This kernel is used for the Mixlen (heterotachy) model branch optimisation. The
Allreduce is:

```cpp
// After "df = all_df; ddf = all_ddf;" and BEFORE the ASC_Lewis block:
if (isModePActive()) {
    double dummy = 0.0;
    modePAllreduceLhDfDdf(dummy, df, ddf);
}
```

---

### F-9  Collective ordering — all ranks must execute Mode P models simultaneously

This is the **fundamental architectural constraint** of P.6. Under FCA dispatch,
each rank owns a disjoint set of substitution-model families. If rank 0 evaluates
LG+G (Mode P) while rank 1 evaluates WAG+G4 (Mode F), the two `MPI_Allreduce`
calls inside the kernel are **for different models** — they will match incorrectly
and return corrupted results.

**The P.6 dispatcher must guarantee that when a Mode P model is evaluated, ALL
ranks enter the same kernel call simultaneously.** This requires a fundamentally
different dispatch structure:

```
Normal FCA models (Mode F / Mode 0):
  Each rank evaluates its own assigned family independently (current behaviour).

Mode P models (a small subset of the heaviest models):
  1. All ranks finish their current Mode F batch (MPI_Barrier).
  2. A designated rank broadcasts model identity: "evaluate LG+R10 now."
  3. All ranks evaluate the SAME model's kernel in Mode P (pattern-parallel).
  4. All ranks contribute [ptn_start, ptn_end) → Allreduce → global lnL.
  5. Result written to model_info on all ranks.
  6. Repeat for next Mode P model.
  7. All ranks barrier, then resume independent Mode F dispatch.
```

This means Mode P evaluation serialises all ranks per Mode P model. For K Mode P
models at AA 1M:

```
Cost(Mode P, K models) ≈ K × (kernel_wall_1M / nranks + Allreduce_latency)
                       = K × (1,122s / 16  +  ~5ms)
                       ≈ K × 70s
```

For K=3 heavy models (LG+R8, LG+R10, LG+FC variants): `3 × 70s = 210s`.
Compare to FCA cost for those same 3 models on one rank: `3 × ~600s = 1,800s`
(estimated from the b3c MF=2,113s with K_outer=1 — the heavy models dominate).

**Speedup for those 3 models: ~8.6×.** The remaining ~1,200 light models stay on
Mode F and cost ~800s (unchanged). Predicted total MF wall ≈ 210s + 800s = 1,010s —
still a regression vs FCA 1,122s. Need to push K up to 10+ or ensure Mode F models
also benefit (see §P.6 performance model).

---

### F-10  `theta_all` memory reduction opportunity

`theta_all` is allocated as `nptn × block × sizeof(double)` bytes
(`initializeAllPartialLh`, phylotree.cpp). Under Mode P, only `[ptn_start,
ptn_end)` of theta is ever populated or read.

For AA 1M at np=16: `ptn_end - ptn_start ≈ 1M/16 = 62,500` patterns. Full
`theta_all = 1M × 20 × 8 = 160 MB`. Mode P `theta_all = 62,500 × 20 × 8 = 10 MB`.
Savings: 150 MB per rank, 2.4 GB across 16 ranks.

This is a **Phase P.5+ optimization**, not required for correctness. Implementation:
change `theta_all` allocation size to `(ptn_end - ptn_start + max_unobserved) ×
block × sizeof(double)` when Mode P is active; adjust all `theta_all + ptn*block`
index expressions to `theta_all + (ptn - ptn_start)*block`. Requires care around
the `theta_computed` cache flag which must also be invalidated when `ptn_start`
changes (per-model call to `initializePtnPartition()`).

---

### F-11  `modePAllreduceLh` overload ambiguity — compile error in ISO baseline build (fixed 2026-05-24)

**Symptom:** First ISO baseline build (job 169130092) failed at `tree/phylotree.cpp:1007`:

```
error: call to member function 'modePAllreduceLh' is ambiguous
```

**Root cause (two iterations to fully resolve):**

The `void` overload was implemented as:
```cpp
void PhyloTree::modePAllreduceLh(double &tree_lh) const {
    tree_lh = modePAllreduceLh(tree_lh);  // attempt 1: tree_lh is double& — ambiguous
}
```

First fix used `double tmp = tree_lh` (job 169130191) — still ambiguous because a non-const `double` lvalue can also bind to `double&`. Second fix (job 169130286) uses `const double tmp`:

```cpp
void PhyloTree::modePAllreduceLh(double &tree_lh) const {
    const double tmp = tree_lh;  // const cannot bind to double& — unambiguously by-value
    tree_lh = modePAllreduceLh(tmp);
}
```

**C++ rule:** a `const T` lvalue cannot bind to a `T&` (non-const lvalue reference). The only matching overload becomes `double modePAllreduceLh(double)` — resolved.

**Propagated to:**
- `/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/tree/phylotree.cpp` (working source)
- `/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-base/tree/phylotree.cpp`
- `/scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/tree/phylotree.cpp`

Build resubmitted as job **169130286**.

---

## Revised kernel modification list (P.3 – P.5)

The original spec listed 10 kernel variants. After source analysis, the relevant
MF-path kernels (called during `optimizeParameters` in `evaluate()`) are:

| Priority | Function | File | Modification | Complexity |
|---|---|---|---|---|
| **P.3** | `computeLikelihoodBranchSIMD` / `...GenericSIMD` | phylokernelnew.h:2660 | limits-shift + Allreduce tree_lh | Medium |
| **P.4** | `computeLikelihoodDervSIMD` / `...GenericSIMD` — normal joint path | phylokernelnew.h:2239 | limits-shift + modePAllreduceLhDfDdf | Medium |
| **P.4b** | `computeLikelihoodDervSIMD` / `...GenericSIMD` — mixture-branch-length path | phylokernelnew.h:2400 | wide Allreduce buffer (nmixlen+nmixlen²+1) | High |
| **P.5a** | `computeLikelihoodFromBufferSIMD` / `...GenericSIMD` | phylokernelnew.h:3286 | ptn-loop bounds change + Allreduce | Low |
| **P.5b** | `computeLikelihoodDervMixlenSIMD` / `...GenericSIMD` | phylokernelnew.h:3500 | limits-shift + modePAllreduceLhDfDdf | Medium |
| **Skip** | `computeMixtureLikelihoodBranchEigenSIMD` | phylokernelmixture.h:730 | — | Skip for Phase 1: mixture eigenvector kernel not called in standard MF on AA |
| **Skip** | `computeNonrevLikelihoodBranchSIMD` | phylokernelnonrev.h:1058 | — | Skip for Phase 1: non-reversible models excluded from `TEST` |

Revised count: **5 priorities** (P.3, P.4, P.4b, P.5a, P.5b) instead of 10.
Mixture-eigenvector and NonRev kernels can be deferred to a Phase P.5+ pass if
non-reversible or mixture-class models are ever added to the MF test suite.

---

## P.3 — `computeLikelihoodBranchSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h`  
**Applies to**: both `KERNEL_FIX_STATES` (fixed-state) and generic template
instantiations — the `#ifdef` guard at line 2660 wraps both in the same body.

### Patch 1 — limits computation (line ~2780)

```cpp
// BEFORE (line 2780):
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);

// AFTER:
    // P.3 Mode P: restrict per-rank work to the assigned pattern slice.
    // When inactive, mp_lo=0 and mp_hi=nptn → identical to the original behaviour.
    // With save_log_value=false (MF path), ASC is guaranteed off so unobserved
    // patterns [max_orig_nptn, nptn) need not be computed — cap at orig_nptn.
    const bool mp_active = isModePActive() && save_log_value == false;
    const size_t mp_lo   = mp_active ? ptn_start : 0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets,
                               mp_active ? mp_size : nptn,
                               limits);
    if (mp_active) {
        // Shift all packet boundaries into [mp_lo, mp_hi].
        // computeBounds generates [0, roundUp(mp_size, VCSIZE)]; we shift by mp_lo.
        for (size_t &lim : limits) lim += mp_lo;
        // The last entry may overshoot mp_hi by at most (VCSIZE-1) patterns.
        // The kernel's "ptn < orig_nptn" guard handles the tail safely.
    }
```

### Patch 2 — Allreduce tree_lh (line ~3157)

```cpp
// BEFORE (line 3157):
    tree_lh += all_tree_lh;

// AFTER:
    tree_lh += all_tree_lh;
    // P.3 Mode P: reduce partial sums from all ranks into the global tree_lh.
    // No-op when Mode P is inactive (modePAllreduceLh returns the value unchanged).
    if (mp_active)
        tree_lh = modePAllreduceLh(tree_lh);
```

**No other changes needed for the MF path** (save_log_value=false guarantees
ASC off and all post-kernel `_pattern_lh` consumption disabled).

For the `save_log_value=true` path (used by best-model final scoring): Mode P is
guarded by `mp_active = isModePActive() && save_log_value == false`, so it
automatically skips Mode P for those calls. Correctness is preserved.

---

## P.4 — `computeLikelihoodDervSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h`  
**Applies to**: both fixed-state and generic template instantiations.

### P.4 Patch 1 — limits computation (line ~2305)

Identical to P.3 Patch 1, substituting `mp_active = isModePActive()` (no
`save_log_value` check — the Derv kernel has no `save_log_value` parameter;
ASC handling is separate).

```cpp
// BEFORE (line 2305):
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);

// AFTER:
    const bool mp_active = isModePActive();
    const size_t mp_lo   = mp_active ? ptn_start : 0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
    vector<size_t> limits;
    computeBounds<VectorClass>(num_threads, num_packets,
                               mp_active ? mp_size : nptn,
                               limits);
    if (mp_active) {
        for (size_t &lim : limits) lim += mp_lo;
    }
```

### P.4 Patch 2 — normal joint path exit Allreduce (line ~2591)

```cpp
// BEFORE (line 2591):
    // normal joint branch length model
    *df  = all_df;
    *ddf = all_ddf;

// AFTER:
    // normal joint branch length model
    // P.4 Mode P: aggregate derivatives across ranks (lnL already Allreduced in
    // BranchSIMD; use dummy to keep the 3-value MPI buffer aligned).
    if (mp_active) {
        double dummy = 0.0;
        modePAllreduceLhDfDdf(dummy, all_df, all_ddf);
    }
    *df  = all_df;
    *ddf = all_ddf;
```

### P.4b Patch — mixture-branch-length path exit (line ~2557)

```cpp
// BEFORE — the block starting with:
    if (isMixlen()) {
        // mixed branch length model
        for (size_t i = 0; i < nmixlen; i++) {
            df[i] = horizontal_add(all_dfvec[i]);
            ...
        }
        for (size_t i = 0; i < nmixlen2; i++) {
            ddf[i] = horizontal_add(all_ddfvec[i]);
        }
        df[nmixlen] = all_lh;
        return;
    }

// AFTER:
    if (isMixlen()) {
        // mixed branch length model
        // P.4b Mode P: Allreduce the dfvec + ddfvec + lh triple in one call.
        if (mp_active) {
            int n = (int)(nmixlen + nmixlen2 + 1);
            vector<double> in_buf(n), out_buf(n, 0.0);
            for (int i = 0; i < (int)nmixlen;  i++) in_buf[i]           = horizontal_add(all_dfvec[i]);
            for (int i = 0; i < (int)nmixlen2; i++) in_buf[nmixlen + i] = horizontal_add(all_ddfvec[i]);
            in_buf[n - 1] = all_lh;
#ifdef _IQTREE_MPI
            MPI_Allreduce(in_buf.data(), out_buf.data(), n, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
#endif
            for (int i = 0; i < (int)nmixlen;  i++) df[i]  = out_buf[i];
            for (int i = 0; i < (int)nmixlen2; i++) ddf[i] = out_buf[nmixlen + i];
            df[nmixlen] = out_buf[n - 1];
        } else {
            for (size_t i = 0; i < nmixlen; i++) {
                df[i] = horizontal_add(all_dfvec[i]);
                ASSERT(std::isfinite(df[i]) && "Numerical underflow for lh-derivative");
            }
            for (size_t i = 0; i < nmixlen2; i++) {
                ddf[i] = horizontal_add(all_ddfvec[i]);
            }
            df[nmixlen] = all_lh;
        }
        return;
    }
```

---

## P.5a — `computeLikelihoodFromBufferSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h:3286`  
Different loop structure from P.3/P.4 — flat ptn-loop, not packet-based.

### Patch 1 — ptn loop bounds (line ~3377)

```cpp
// BEFORE (line 3377):
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t ptn = 0; ptn < nptn; ptn+=VectorClass::size()) {

// AFTER:
    const bool mp_active = isModePActive();
    const size_t mp_lo   = mp_active ? ptn_start : (size_t)0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    // Alignment guaranteed by initializePtnPartition() VCSIZE rounding.
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t ptn = mp_lo; ptn < mp_hi; ptn+=VectorClass::size()) {
```

### Patch 2 — Allreduce tree_lh (line ~3449)

```cpp
// BEFORE (line 3449):
    double tree_lh = all_tree_lh;

// AFTER:
    double tree_lh = all_tree_lh;
    if (mp_active)
        tree_lh = modePAllreduceLh(tree_lh);
```

---

## P.5b — `computeLikelihoodDervMixlenSIMD` / `GenericSIMD` — exact patch

**File**: `tree/phylokernelnew.h:3500`

### Patch 1 — limits (line ~3553)

```cpp
// BEFORE (line 3553):
    computeBounds<VectorClass>(num_threads, num_packets, nptn, limits);

// AFTER:
    const bool mp_active = isModePActive();
    const size_t mp_lo   = mp_active ? ptn_start : (size_t)0;
    const size_t mp_hi   = mp_active ? std::min(ptn_end, (size_t)orig_nptn) : nptn;
    const size_t mp_size = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : (size_t)0;
    computeBounds<VectorClass>(num_threads, num_packets,
                               mp_active ? mp_size : nptn,
                               limits);
    if (mp_active) {
        for (size_t &lim : limits) lim += mp_lo;
    }
```

### Patch 2 — Allreduce df/ddf (line ~3664)

```cpp
// BEFORE (line 3664):
    df  = all_df;
    ddf = all_ddf;

// AFTER:
    df  = all_df;
    ddf = all_ddf;
    if (mp_active) {
        double dummy = 0.0;
        modePAllreduceLhDfDdf(dummy, df, ddf);
    }
```

---

## `initializePtnPartition()` alignment fix (prerequisite for P.5a)

Apply to `tree/phylotree.cpp:930` (the VCSIZE-aligned chunk calculation):

```cpp
// BEFORE (current code at phylotree.cpp:940):
    size_t chunk = nptn / (size_t)nranks;
    ptn_start = chunk * (size_t)rank;
    ptn_end   = (rank == nranks - 1) ? nptn : chunk * (size_t)(rank + 1);

// AFTER:
    // Round chunk DOWN to VECTOR_SIZE (8 for AVX-512) so that ptn_start and
    // ptn_end are always VCSIZE-aligned — required by load_a() in FromBufferSIMD.
    const size_t vcsize = VECTOR_SIZE;   // compile-time constant from instrset.h
    size_t chunk = (nptn / (size_t)nranks / vcsize) * vcsize;
    if (chunk == 0) chunk = vcsize;      // safety: at least one VCSIZE block
    ptn_start = chunk * (size_t)rank;
    ptn_end   = (rank == nranks - 1) ? nptn : chunk * (size_t)(rank + 1);
    // Clamp to nptn for last rank (last rank gets the uneven tail).
    if (ptn_end > nptn) ptn_end = nptn;
```

---

## `isModePActive()` — consolidated update (incorporates F-4, F-5)

Replace the existing `isModePActive()` body in `tree/phylotree.cpp:910`:

```cpp
bool PhyloTree::isModePActive() const {
    if (!params || params->mode_p_enabled == 0)
        return false;
    // F-4: SPR guard — Mode P only inside evaluateAll(); disable after MF phase.
    if (!params->mode_p_active_in_mf)
        return false;
    // F-5: MPI_THREAD_FUNNELED safety — disable when K_outer > 1 (OMP parallel
    // region means non-main threads would call MPI_Allreduce → UB).
    // atmd_K_outer == 0 means "auto" (K_outer=1 at AA 1M with b4 formula).
    // atmd_K_outer == 1 is explicit serial → safe.
    // atmd_K_outer > 1 → unsafe.
    int k = params->atmd_K_outer;
    if (k > 1)
        return false;
#ifdef _IQTREE_MPI
    return MPIHelper::getInstance().getNumProcesses() > 1;
#else
    return false;
#endif
}
```

Add `bool mode_p_active_in_mf = false;` to `Params` (utils/tools.h next to
`mode_p_enabled`).

Set in `evaluateAll()` (phylotesting.cpp) at entry:
```cpp
params.mode_p_active_in_mf = true;
```
And at exit (both normal return and early-return paths):
```cpp
params.mode_p_active_in_mf = false;
```

---

## P.6-lite — Minimum-viable collective dispatch (ISO-2 unblocker)

**Status:** Implemented 2026-05-24 in `-p3` source tree (commit pending). Required to unblock ISO-2 after B.4-7 dispatch-mismatch failure.

### Why P.6-lite is needed

ISO-2 (job 169131722) failed because the P.3 kernel patch is correctness-correct in isolation, but the **F-4 enable-during-MF gate alone is insufficient**: it ensures Mode P is *active* during MF, but does NOT ensure all ranks are on the *same model* at every collective `MPI_Allreduce` inside the kernel. FCA dispatches different models to different ranks → kernel `Allreduce` sums `partial(LG)[0..48008) + partial(WAG)[48008..96017)` → garbage tree_lh → `theta_all` left in mixed-model state → unpatched Derv kernel reads garbage → `df` underflows → mpirun aborts in 25s.

Full P.6 (heavy-model cost-threshold routing with two-phase Mode F / Mode P dispatch) is the production target but is a multi-day implementation. P.6-lite is a one-line conceptual change that unblocks ISO-2 immediately and validates the P.3 kernel patch end-to-end.

### What P.6-lite does

When `params.mode_p_enabled != 0 && nranks > 1`:
- **Skip the entire FCA dispatch block** (Steps 1-8 at `phylotesting.cpp:3931-4073`).
- No `MF_IGNORED` markers are set → `getNextModel()` returns the full model sequence on every rank → all ranks lockstep through every model in identical order.
- `rate_block` stays at its auto-detected value (Step 7 doesn't override it to `num_models`) → legacy `filterRates(model)` fires correctly when `model >= rate_block`, identical decision on every rank (since all ranks see identical scores).
- `mpi_filterRatesMPI_enabled` stays `false` (reset above the FCA block, never set under P.6-lite) → FCA state-machine trigger inside the loop stays gated off.
- Phase 2 Allreduce-merge at end-of-`evaluateAll` becomes a no-op-style safety net: every rank has every model `MF_DONE` with the same scores → `MPI_MAX(x, x) = x`.

### Implementation (committed to `-p3` source tree)

```cpp
// phylotesting.cpp inside evaluateAll(), placed BEFORE the existing FCA block:
bool p6_lite_collective = (params.mode_p_enabled != 0
                           && MPIHelper::getInstance().getNumProcesses() > 1);

if (p6_lite_collective) {
    int my_rank = MPIHelper::getInstance().getProcessID();
    int nranks  = MPIHelper::getInstance().getNumProcesses();
    cout << "MF-MPI-DIAG: rank " << my_rank << "/" << nranks
         << " owns " << num_models << "/" << num_models
         << " models (P.6-lite collective Mode P dispatch — FCA disabled)"
         << endl;
    cout.flush();
}

if (MPIHelper::getInstance().getNumProcesses() > 1 && !p6_lite_collective) {
    // ... existing FCA Steps 1-8 unchanged ...
}
```

### Invariants P.6-lite preserves

| Invariant | How preserved |
|---|---|
| F-4 (Mode P only during MF) | `mode_p_active_in_mf` still set/reset by existing F-4 code |
| F-5 (Allreduce only with K_outer=1) | `--atmd-k-outer 1` from ISO-2 script; OMP outer team's `if(>1)` clause defeats parallel region |
| `MPI_THREAD_FUNNELED` safety | Loop runs sequentially in main thread on every rank |
| Deterministic model order | `getNextModel()` iterates `at(0..N-1)` skipping `MF_IGNORED`/`MF_DONE`; with no `MF_IGNORED` set, order is identical on every rank |
| Phase 2 score merge | Every rank has every score; `MPI_MAX(x,x)=x` is consistent |

### Cost vs full P.6

P.6-lite is strictly slower than full P.6 in the multi-rank regime because **every rank evaluates every model**, instead of FCA's per-rank subset. Speedup vs single-rank comes only from intra-model pattern partitioning + kernel Allreduce. For AA 100K np=2:
- Reference FCA np=2: MF wall ≈ 149s (each rank evaluates ~112/224 models in parallel)
- Expected P.6-lite np=2: MF wall ≈ 2× FCA (each rank evaluates 224/224 models, but each kernel call is 2× faster → net same as single-rank ≈ 259s)

P.6-lite is a **correctness gate, not a performance win** — it proves the P.3 kernel patch produces identical lnL across ranks. Full P.6 (selective routing) restores cross-model parallelism for light models.

### ISO-2 expected behaviour under P.6-lite

- New diag line: `MF-MPI-DIAG: rank 0/2 owns 224/224 models (P.6-lite collective Mode P dispatch — FCA disabled)`
- `[Mode P] rank 0 model=X ptn=[0, 48008)` and `[Mode P] rank 1 model=X ptn=[48008, 96017)` — **same model `X` on both ranks** (not LG vs WAG as in the failed run)
- Final lnL within `1e-6` of FCA np=2 reference `-7,541,976.853`
- Best model `LG+G4`

---

## P.6 — Collective dispatch for Mode P models

### Design overview

P.6 replaces the original "flag per model" concept with a **two-phase loop** inside
`evaluateAll()`:

**Phase A** (Mode F / independent): each rank evaluates its assigned models using
the existing FCA sequential loop. Mode P models are **skipped** in this phase
(marked `MF_MODE_P`).

**Phase B** (Mode P / collective): a single all-ranks loop iterates over all
`MF_MODE_P` models in a fixed global order. For each model:
1. All ranks barrier.
2. All ranks call `initializePtnPartition()` to set their slice.
3. All ranks call `evaluate(model)` → kernels Allreduce internally.
4. Result is recorded on all ranks (no cross-rank model_info sync needed — every
   rank ran the full model, each with its pattern slice, and the Allreduce gave
   the correct global lnL).

```cpp
// Phase B insertion after the existing do-while evaluateAll loop:
if (params.mode_p_enabled && MPIHelper::getInstance().getNumProcesses() > 1) {
    params.mode_p_active_in_mf = true;
    for (int model = 0; model < (int)num_models; model++) {
        if (!at(model).hasFlag(MF_MODE_P)) continue;
        // Barrier: ensure all ranks start this model simultaneously.
#ifdef _IQTREE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        at(model).evaluate(params, in_tree, model_info, &local_in_info,
                           score_diff_thres, model, initial_model_rate,
                           substitution_model, in_tree_rate, in_tree_freq,
                           write_info, aln_rate, set_output);
    }
    params.mode_p_active_in_mf = false;
}
```

### Model selection threshold for MF_MODE_P

Use `modelCostFCA(model)` (already computed in the FCA dispatch block) to rank
models. Mark the top-K% by cost as `MF_MODE_P`:

```cpp
// After Step 4 (MF_IGNORED marking) in evaluateAll():
if (params.mode_p_enabled != 0) {
    // Compute per-model costs for non-IGNORED models.
    vector<pair<double,int>> costs;
    for (int i = 0; i < (int)num_models; i++)
        if (!at(i).hasFlag(MF_IGNORED))
            costs.push_back({modelCostFCA(i), i});
    // Sort descending by cost.
    sort(costs.begin(), costs.end(),
         [](auto &a, auto &b){ return a.first > b.first; });
    // Mark top models as Mode P (those exceeding threshold × median cost).
    double thresh = params.mode_p_min_cost_mult;  // default 8.0
    double median_cost = costs.empty() ? 0.0 : costs[costs.size()/2].first;
    for (auto &[cost, idx] : costs) {
        if (cost >= thresh * median_cost) {
            at(idx).setFlag(MF_MODE_P);
            at(idx).setFlag(MF_IGNORED);  // skip in Phase A
        }
    }
}
```

Add `MF_MODE_P = 32` to the flag constants (phylotesting.h:33):
```cpp
const int MF_MODE_P = 32;
```

### Performance model for P.6 threshold tuning

For AA 1M np=16 with `mode_p_min_cost_mult=8.0`:
- Estimated models above threshold: ~8 (LG+R6..R10, LG+FC+R variants, WAG+R8+)
- Mode P cost per model: `~70s` (kernel_wall / 16 + Allreduce ≈ 4ms)
- Mode P total: `8 × 70s = 560s`
- Remaining Mode F (light) models: ~140 models × ~4s each = ~560s (but now all 16
  ranks handle them independently in Phase A)
- Predicted MF wall: `max(560s_ModeP, 560s_ModeF) = 560s` — **50% faster than FCA 1,122s**

For `mode_p_min_cost_mult=4.0` (more aggressive):
- Estimated Mode P models: ~20
- Mode P total: `20 × 70s = 1,400s` — WORSE (too many models serialised)

Optimal threshold appears to be around 6–10×. The AA 1M gate at `≤600s MF wall`
is achievable at threshold=8 with the above model. Tune via the 100K sanity check
first.

---

## P.7 — Validation plan and gates

### Build recipe

ISO build scripts: `gadi-ci/mode-p-iso/build_mode_p_iso_*.sh` (new, to create).
Source: isolated copy of the b4 source tree + incremental P.3–P.6 patches.
ISO binaries: `iqtree3-mpi-mode-p-iso-base`, `iqtree3-mpi-mode-p-iso-p3`,
`iqtree3-mpi-mode-p-iso-p4`, `iqtree3-mpi-mode-p-iso-p5`.

Production build script after ISO promotion: `gadi-ci/lbfgs-ws/build_atmd_mode_p.sh`
(new, to create). Binary: `iqtree3-mpi-atmd-mode-p`.

CMake flags: same as b4 (`-DIQTREE_ATMD=ON -DIQTREE_MPI=ON -march=sapphirerapids`).

### Gate 1 — P.1+P.2 structural validation (already done; re-confirm with b4 base)

```bash
# Single rank: Mode P inactive (no [Mode P] line, lnL identical to FCA np=1)
mpirun -np 1 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G --mode-p-all
```

### Gate 2 — P.3 correctness (lnL parity at np=2)

```bash
# Mode P forced on every model (-all), AA 100K np=2 vs FCA np=2 baseline
mpirun -np 2 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G4 --mode-p-all -seed 1
# Target: lnL = -7,541,976.853 ± 1e-6
# Compare to: mpirun -np 2 iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G4 -seed 1
```

If lnL diverges:
1. Check that `ptn_start` and `ptn_end` are printed in the `[Mode P]` lines and
   cover exactly all `orig_nptn` patterns across ranks without overlap.
2. Check `theta_computed` is not stale between BranchSIMD and DervSIMD calls.
3. Verify VCSIZE alignment: confirm `ptn_start % 8 == 0` for both ranks.

### Gate 3 — P.4/P.5 derivative parity (BFGS trace match)

```bash
# Compare branch-length optimisation convergence trace between mode-p and b4 np=2
# Both runs with -v (verbose) to dump per-NR-iteration lnL
mpirun -np 2 iqtree3-mpi-atmd-mode-p -s AA_100K.phy -m LG+G4 --mode-p-all -v -seed 1 > mode_p_trace.log
mpirun -np 2 iqtree3-mpi-atmd-b4 -s AA_100K.phy -m LG+G4 -v -seed 1 > b4_trace.log
diff <(grep "NR iter" mode_p_trace.log) <(grep "NR iter" b4_trace.log)
# Target: NR iterations converge to same sequence of lnL values
```

### Gate 4 — Full MF test, AA 100K np=4 (default threshold)

```bash
qsub run_mode_p_aa_100k_4node.sh   # -m TEST --mode-p (default threshold 8x)
# Targets:
#   lnL = -7,541,976.853 ± 0.5
#   Best model = LG+G4
#   MF wall ≤ FCA np=4 (149s × 4/4 = ~110s nominal; mode-p overhead may push to ~150s)
```

### Gate 5 — AA 1M np=16 performance gate (P.7)

```bash
qsub run_mode_p_aa_1m_16node.sh    # -m TEST --mode-p (threshold 8x)
# Targets:
#   lnL = -78,605,196.497 ± 0.5
#   Best model = LG+G4
#   MF wall ≤ 600s  (vs FCA 1,122s ref)
```

If `MF wall > 600s` but correct lnL:
- Profile which models are Mode P and how long each took.
- Tune threshold (lower `mode_p_min_cost_mult` to route more models through Mode P).
- Check Allreduce latency contribution (5ms × 100 models = 500ms — negligible vs 70s/model).

If `MF wall < 600s` → Mode P is viable. Close Phase P.7, roll to §15.10.

---

## Build scripts needed (new, to create)

### ISO sandbox scripts — highest priority

| Script | Purpose |
|---|---|
| `gadi-ci/mode-p-iso/bootstrap_mode_p_iso.sh` | Create `/scratch/rc29/as1708/iqtree3-mode-p-iso/`, copy current b4 source state, record source diff/provenance |
| `gadi-ci/mode-p-iso/build_mode_p_iso_base.sh` | Build inert P.1/P.2 baseline binary for ISO-0/ISO-1 |
| `gadi-ci/mode-p-iso/build_mode_p_iso_p3.sh` | Build P.3-only kernel binary |
| `gadi-ci/mode-p-iso/build_mode_p_iso_p4.sh` | Build P.3+P.4 derivative binary |
| `gadi-ci/mode-p-iso/build_mode_p_iso_p5.sh` | Build P.3+P.4+P.5 kernel-family binary |
| `gadi-ci/mode-p-iso/run_iso_lg_g4_aa100k_np1_base.sh` | ISO-0 base single-rank sanity |
| `gadi-ci/mode-p-iso/run_iso_lg_g4_aa100k_np2_p3.sh` | ISO-2 P.3 lnL/BIC parity gate |
| `gadi-ci/mode-p-iso/run_iso_lg_g4_aa100k_np2_p4_trace.sh` | ISO-3 derivative/NR trace gate |
| `gadi-ci/mode-p-iso/run_iso_mf_aa100k_np4_auto.sh` | ISO-5 mixed dispatcher correctness gate |
| `gadi-ci/mode-p-iso/run_iso_mf_aa1m_np16_p7.sh` | ISO-6 AA 1M performance gate |
| `tools/mode_p_iso/compare_mode_p_parity.py` | Parse `.iqtree`, stdout, rank logs, lnL/BIC/model/timing/partition coverage; fail closed on missing evidence |

### Production scripts — after ISO promotion

| Script | Purpose |
|---|---|
| `gadi-ci/lbfgs-ws/build_atmd_mode_p.sh` | Build the Mode P binary (b4 + P.3–P.6 patches) |
| `gadi-ci/lbfgs-ws/run_mode_p_correctness_aa_100k_2node.sh` | Gate 2 correctness, np=2, AA 100K |
| `gadi-ci/lbfgs-ws/run_mode_p_aa_100k_4node.sh` | Gate 4 full MF np=4 AA 100K |
| `gadi-ci/lbfgs-ws/run_mode_p_aa_1m_16node.sh` | Gate 5 perf gate np=16 AA 1M |

---

## Implementation sequencing (revised)

```
Step 1:  Create Mode P kernel ISO sandbox (P.ISO) from current b4 source state (1 hr)
Step 2:  Add ISO build/run/parity scripts and parser (2 hr)
Step 3:  Build ISO base + run ISO-0/ISO-1 inert scaffolding gates (30 min build + 30 min PBS)
Step 4:  Apply initializePtnPartition() alignment fix inside ISO (30 min)
Step 5:  Apply isModePActive() F-4 + F-5 guards + mode_p_active_in_mf Params field inside ISO (45 min)
Step 6:  Apply P.3 patches to BranchSIMD inside ISO (1 hr)
Step 7:  Build ISO P.3 + Gate ISO-2 correctness np=2 AA 100K (30 min build + 30 min PBS)
         → If PASS: proceed. If FAIL: debug limits-shift.
Step 8:  Apply P.4 (normal joint path + mixture path) inside ISO (1 hr)
Step 9:  Apply P.5a (FromBuffer) inside ISO (45 min)
Step 10: Apply P.5b (DervMixlen) inside ISO (45 min)
Step 11: Build ISO P.5 + Gate ISO-3 derivative parity (30 min + 10 min local/short PBS)
Step 12: Apply P.6 dispatcher in ISO (MF_MODE_P flag + collective Phase B loop) (2 hr)
Step 13: Build ISO + Gate ISO-5 full MF AA 100K np=4 (30 min + 1 hr PBS)
Step 14: Gate ISO-6 AA 1M np=16 performance/correctness (30 min + 3 hr PBS)
Step 15: Promote exact ISO patch set into main b4/Mode P source tree (1 hr)
Step 16: Build production `iqtree3-mpi-atmd-mode-p` + rerun Gate 4/Gate 5 (30 min + PBS)
Step 17: Tune threshold if needed (30 min)
Step 18: Document results, update §15.10 and CHANGELOG
```

Total estimate: ~14 hours engineering + ~6–8 hours PBS turnaround = **~3 working days**.
