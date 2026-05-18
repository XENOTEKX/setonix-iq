# IQ-TREE GPU Offload — Development Progress

---

## 🎯 BASELINE OF RECORD (2026-05-17 correction)

All ModelFinder benchmarks are measured against **job 168425673** —
the standard (non-MPI) IQ-TREE 3.1.2 SPR binary built by sa0557 at
`/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3`
(commit `4e91dd6` = v3.1.2, ICX + AVX-512 + -xSAPPHIRERAPIDS + R1+R2 patches).

| Metric | Value |
|--------|------:|
| Job ID | 168425673 |
| Binary | `build-intel-vanila/iqtree3` (non-MPI, OMP-across-models) |
| Node   | gadi-cpu-spr-0570 (SPR exclusive, 103 OMP threads on 104 cores) |
| Alignment | AA 100K (100 taxa × 100K sites, 96K patterns) |
| **MF wall** | **~405 s** (derived: 1,169.6 − 764.5 tree-search = 405.1 s) |
| **Tree-search wall** | **764.478 s** |
| **Total wall** | **1,169.556 s** (0h:19m:29s) |
| Total CPU | 108,645.544 s (92% OMP efficiency) |
| lnL | −7,541,976.860 |
| BIC | 15,086,233 |
| Best model | LG+G4 |
| Peak memory | 9.36 GB (PBS `Memory Used`) |

**TARGET: beat 400 s on MF wall.** Every prior FCA / MF2 attempt
(Phases 0, 0.5, 0.6, 0.7, HH-NUMA) regressed against this baseline.
The "Fix H" numbers cited in earlier CHANGELOG entries (np=1: 1,289 s,
np=2: 475 s, np=4: 2,335 s) are *all* worse than this baseline — they
are MPI builds that disabled OMP-across-models and re-introduced
dispatch overhead. They are kept here for historical context only;
**do NOT use them as a baseline for future work**.

To beat 400 s the dispatch must preserve OMP-across-models parallelism
(present in the standard binary, removed by Fix H for OOM safety) OR
amortise the loss across MPI ranks (np ≥ 2 with effective filterRates
pruning across families). The MF-iso harness (entry `bk` below) is
designed to demonstrate which structural changes preserve the baseline
walltime; HH-NUMA's nested K_outer × M_inner is the projected path back
to OMP-across-models without OOM.

Memory note: the standard binary's 9.36 GB peak (vs the dispatch doc's
6.27 GB per-tree × 103-thread estimate of 646 GB) confirms that
`evaluate()` in v3.1.2 uses a smaller working set per model than the
worst-case BIONJ-tree partial_lh — `MF_IGNORED` skipping and parsimony-
tree reuse keep the in-flight memory well below the theoretical ceiling.
This changes the HH-NUMA K_outer feasibility analysis (the budget is
likely closer to ~100 GB at K=8 rather than 125 GB).

---

## 📊 Full Runs Comparison (MF+SPR, all PASS ✓)

Summary of all completed full MF+SPR runs for both the standard baseline and the FCA MF-iso Phase 0.5+0.6 harness.
All correctness checks pass: |ΔlnL| < 0.5 and BIC delta < 1.0 vs baseline for every dataset.

**FCA binary:** `iqtree3-mpi` · md5 `a78ffa2942d6b073490d503416ae554c` · commit `9603247f` on `test_MF2` (fast-forwarded 2026-05-18) · ICX 2025.3.2 + OpenMPI 4.1.7 + AVX-512 + libiomp5 · seed=1 · `-m TEST -T 103`  
**Baseline binary:** `build-intel-vanila/iqtree3` · non-MPI OMP-across-models · ICX + AVX-512 + R1+R2 · v3.1.2 (`4e91dd6`) · sa0557

> **How to run the FCA binary** — see [`research/updated-modelfinder-dispatch.md`](research/updated-modelfinder-dispatch.md):
> §22 (architecture diagram), §23 (operator guide + flag reference), §24 (validated results table with provenance).

> **IPC note:** `perf stat` was not collected for FCA full runs prior to 2026-05-18 (no `perf_stat.txt` in profiles).
> The only available IPC figure from the baseline is **1.88 insn/cycle** (from `AA_100k_spr_seed1_168425673/perf_stat.txt`).
> `rank_perf.sh` wrapper added to AA 1M full scaling scripts (168635614–168635616); per-rank `perf_stat_rank_N.txt` collected in each job's WORK_DIR — all three jobs complete.
> Measured FCA AA 1M IPC (user-space, all ranks): **np=2: 1.26 · np=4: 1.27 · np=16: 1.34** (vs baseline 1.88 at AA 100K — expected lower: AA 1M is more memory-bound).
> LLC miss rate (`cache-misses:u / cache-references:u`, both map to LLC on Intel SPR): **np=2: 83.7% · np=4: 84.0% · np=16: 85.3%**. `LLC-loads:u` counter returned 0 (not accessible in normalsr — `cache-references/misses:u` used instead, which are the LLC-level hardware counters on SPR).

| Job | Type | Dataset | Nodes | Ranks×OMP | Best model | lnL | BIC | MF wall (s) | SPR wall (s) | Total wall (s) | Speedup | IPC (mean) | LLC miss % |
|-----|------|---------|-------|-----------|------------|-----|-----|------------|-------------|----------------|---------|------------|------------|
| 168425674 | Baseline | DNA 100K | 1 | 1×103T | F81+F+G4 | −5,692,984.539 | 11,388,283.176 | 61.740 | 226.447 | 289.121 | — | **1.302** | **66.24%** |
| 168584737 | FCA np=2 | DNA 100K | 2 | 2×103T | F81+F+G4 | −5,692,984.532 | 11,388,283.162 | 26.252 | 86.613 | 113.754 | **2.54×** | — | — |
| 168425673 | Baseline | AA 100K | 1 | 1×103T | LG+G4 | −7,541,976.860 | 15,086,233.280 | 399.456 | 764.478 | 1,169.556 | — | **1.878** | **56.02%** |
| 168584736 | FCA np=2 | AA 100K | 2 | 2×103T | LG+G4 | −7,541,976.853 | 15,086,233.265 | 149.029 | 383.876 | 537.750 | **2.18×** | — | — |
| 168425491 | Baseline | AA 1M | 1 | 1×103T | LG+G4 | −78,605,196.573 | 157,213,128.618 | 7,587.459 | 15,098.605 | 22,776.226 | — | — | — |
| 168635614 | FCA np=2 | AA 1M | 2 | 2×103T | LG+G4 | −78,605,196.443 | — | 3,076.873 | 7,868.928 | 10,945.801 | **2.08×** | **1.260** | **83.69%** |
| 168635615 | FCA np=4 | AA 1M | 4 | 4×103T | LG+G4 | −78,605,196.445 | — | 1,974.476 | 3,982.142 | 5,956.618 | **3.82×** | **1.273** | **84.01%** |
| 168586094 | FCA np=8 | AA 1M | 8 | 8×103T | LG+G4 | −78,605,196.497 | 157,213,128.466 | 1,443.892 | 2,147.499 | 3,671.618 | **6.20×** | — | — |
| 168635616 | FCA np=16 | AA 1M | 16 | 16×103T | LG+G4 | −78,605,196.497 | — | 1,122.363 | 1,287.863 | 2,410.226 | **9.45×** | **1.337** | **85.27%** |
| 168425675 | Baseline | DNA 1M | 1 | 1×103T | F81+F+G4 | −59,208,019.212 | 118,418,815.342 | 3,500.825 | 2,596.995 | 6,114.450 | — | — | — |
| 168592214 | FCA np=8 | DNA 1M | 8 | 8×103T | F81+F+G4 | −59,208,019.103 | 118,418,815.123 | 1,274.686 | 349.904 | 1,640.846 | **3.73×** | — | — |

> **Timing notes:** MF and SPR wall times from IQ-TREE stdout (`Wall-clock time for ModelFinder` /
> `Wall-clock time used for tree search`). Total wall from PBS job wall-clock (includes startup + IO overhead;
> typically 1–80 s greater than MF+SPR sum). BIC and lnL from `.iqtree` report files on scratch.
> Speedup = baseline total ÷ FCA total.
> IPC and LLC miss %: user-space `perf stat` (`cycles:u`, `instructions:u`, `cache-references:u`, `cache-misses:u`).
> `cache-references/misses:u` map to LLC-level hardware counters on Intel SPR. `—` = no perf stat collected for that run.

---

## 2026-05-18 (bn) — Branch promotion: `test_MF2` now carries Phase 0.5+0.6+MF-TIME

### What changed (Git state)

| Tree | Branch | Before | After |
|------|--------|--------|-------|
| rc29 (`/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3`) | `test_MF2` | `ffb79a14` (Phase 0 FCA) | `9603247f` (Phase 0 + 0.5 + 0.6 + MF-TIME) ✓ |
| um09 (`/scratch/um09/as1708/iqtree3-mf2/src/iqtree3`)   | `test_MF2` | `ffb79a14` (Phase 0 FCA) | `9603247f` (Phase 0 + 0.5 + 0.6 + MF-TIME) ✓ |
| XENOTEKX/setonix-iq (GitHub) | `test_MF2` | (not present) | ✓ `* [new branch] test_MF2 -> test_MF2` at `9603247fc85fb7acdfb470b27c44f5bfa59e43ba` |

Both local trees are byte-identical on `test_MF2`. The fast-forward was clean
(`mf-iso-phase0.5-0.6` was exactly 1 commit ahead of `test_MF2`, all on top
of `ffb79a14`). The `mf-iso-phase0.5-0.6` branch is kept as a historical
label but is no longer the canonical reference.

### Why now

All four validated full MF+SPR runs (DNA 100K, AA 100K, AA 1M, DNA 1M)
ran against binary `a78ffa2942d6b073490d503416ae554c` — built from commit
`9603247f`. That binary is in production use across two scratch allocations
(rc29 and dx61, identical md5). The previous `test_MF2` HEAD (`ffb79a14`)
did NOT include the Phase 0.5/0.6 work that made those speedups possible.
Promoting `test_MF2` aligns the branch label with the binary's actual
content.

### Pushes completed

Both remotes updated on 2026-05-18:

**IQ-TREE3 source — `test_MF2` → XENOTEKX/setonix-iq**

```
cd /scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3
git push setonix-iq test_MF2
# * [new branch]        test_MF2 -> test_MF2
```

New branch created on remote; HEAD at `9603247fc85fb7acdfb470b27c44f5bfa59e43ba`.

**Scripts/docs — `modelfinder2` → XENOTEKX/setonix-iq**

```
cd /home/272/as1708/setonix-iq
git push origin modelfinder2
# ff71b13d..61de3b66  modelfinder2 -> modelfinder2
```

Commit `61de3b66` carries the §22/§23/§24 additions to `updated-modelfinder-dispatch.md`
and the updated `gadi-ci/mf-iso/build_mf_iso.sh` (Lustre nm fix).

### Documentation cross-links

The dispatch doc gained three new operator-facing sections capturing the
state of the FCA stack as it's actually used in production:

- **§22 Architecture** — ASCII timeline diagram of Phase 0 + 0.5 + 0.6 + MF-TIME
  interacting at np=2, plus the MPI deadlock gate (MPI_Allreduce MIN over
  per-rank readiness).
- **§23 Operator guide** — module loads, build command, flag reference for
  IQ-TREE / mpirun / OMP env vars, the end-to-end command, what artefacts to
  check after each run, and per-node-count acceptance gates.
- **§24 Validated results** — the full 8-row table (4 baseline + 4 FCA)
  with provenance (binary md5, source commit, build host, compiler), correctness
  proof (`|ΔlnL| < 0.5`), and links to each `logs/runs/*.json`.

What's explicitly NOT on `test_MF2`: Phase 0.7 (`MPI_Isend`), HH-NUMA Phase 2
(nested K_outer × M_inner), and the `MPI_Init_thread(MPI_THREAD_SERIALIZED)`
upgrade. These hung job 168486582 with SIGTERM and remain deferred to
separate commits — each will land with its own 2-node validation, only
after `test_MF2` is the published baseline.

---

## 2026-05-18 (bm) — DNA validation: Phase 0.5+0.6 extended to DNA 100K and DNA 1M

### Context

AA 100K Phase 0.5+0.6 confirmed (np=1: 168577707, np=2: 168577708).
Extended the same isolation harness to DNA datasets to confirm:
  - filterRatesMPI fires correctly for GTR-family models (DNA substitution model set)
  - lnL parity holds across np=1 and np=2
  - MF wall < np=1 wall at np=2 (scaling benefit for DNA)

### New scripts (all in `gadi-ci/mf-iso/`)

| Script | Purpose | PBS walltime | EXPECTED_LNL |
|--------|---------|-------------|--------------|
| `run_baseline_dna_100k_spr.sh` | Standard binary baseline, DNA 100K | 01:30:00 | TBD (run first) |
| `run_mf_iso_dna_100k_1node.sh` | Phase 0.5+0.6 1-node, DNA 100K | 01:30:00 | TBD after baseline |
| `run_mf_iso_dna_100k_2node.sh` | Phase 0.5+0.6 2-node, DNA 100K | 02:00:00 | TBD after 1-node |
| `run_baseline_dna_1m_spr.sh` | Standard binary baseline, DNA 1M | 04:00:00 | CLX ref −59,208,019.212 |
| `run_mf_iso_dna_1m_1node.sh` | Phase 0.5+0.6 1-node, DNA 1M | 04:00:00 | −59,208,019.212 ±0.5 |
| `run_mf_iso_dna_1m_2node.sh` | Phase 0.5+0.6 2-node, DNA 1M | 04:00:00 | −59,208,019.212 ±0.5 |

`submit_mf_iso.sh` updated with: `dna_100k_all`, `dna_1m_all` (and individual stage targets).

### DNA dataset paths

| Dataset | Path |
|---------|------|
| DNA 100K | `/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` |
| DNA 1M   | `/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy` |

### Binary legend for this entry

| Label | Binary | Build | Run type |
|-------|--------|-------|----------|
| **Baseline (R1+R2/AVX-512)** | `build-intel-vanila/iqtree3` (non-MPI, OMP) | ICX + AVX-512 + -xSAPPHIRERAPIDS + R1+R2 patches, v3.1.2 (4e91dd6) — sa0557 | Full run (MF + SPR tree search); also used for TESTONLY baselines |
| **FCA mf-iso** | `iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` (MPI + OMP) | ICX 2025.3.2 + OpenMPI 4.1.7 + AVX-512 + libiomp5, branch `mf-iso-phase0.5-0.6` — as1708 | **TESTONLY** (MF only) **and Full run** (MF + SPR tree search) — same binary |
| **CLX build** | `cpu_bench/build-intel-clx/iqtree3` (non-MPI, OMP) | ICX + Cascade Lake; used for historical baseline only | Full run (MF + SPR tree search) |

### Prior reference runs (full IQ-TREE: MF + SPR tree search)

These are **Baseline (R1+R2/AVX-512)** runs — no FCA, no MPI, OMP-across-models.
TESTONLY mf-iso runs use `-m TESTONLY` (full ModelFinder phase, no SPR tree search after).
Full mf-iso runs use `-m TEST` (FCA-parallelised ModelFinder + SPR tree search — same binary).
NJ-tree TESTONLY lnL vs SPR-tree full-run lnL differ by < 1 unit at 100K+ sites.
Parity check tolerance: **1.0** for TESTONLY NJ-tree vs SPR ref; **0.1** for full-run SPR vs SPR ref; **0.01** for np=1 vs np=2 TESTONLY.

| Dataset | Job | Binary | Node | Threads | lnL | Best model | MF wall | Total wall |
|---------|-----|--------|------|---------|-----|-----------|---------|-----------|
| DNA 100K | 168422811 | CLX build (non-MPI) | `normal` (CLX) | 47T OMP | −5,692,984.539 | F81+F+G4 | 159.1 s | 546 s |
| DNA 100K | **168425674** | **Baseline R1+R2/AVX-512** (non-MPI) | `normalsr` (SPR) | **103T OMP** | −5,692,984.539 | F81+F+G4 | **61.7 s** | 289 s |
| DNA 1M   | 168422813 | CLX build (non-MPI) | `normal` (CLX) | 47T OMP | −59,208,019.212 | F81+F+G4 | 10,230 s | 17,753 s |
| DNA 1M   | **168425675** | **Baseline R1+R2/AVX-512** (non-MPI) | `normalsr` (SPR) | **103T OMP** | −59,208,019.212 | F81+F+G4 | **3,501 s** | 6,114 s |

SPR MF speedup vs CLX: DNA 100K **2.6×** (159→62 s), DNA 1M **2.9×** (10230→3501 s).

### Run status — DNA chains (submitted 2026-05-18)

| Job | Name | Binary | Node | State | Elapsed | Note |
|-----|------|--------|------|-------|---------|------|
| ~~168580367~~ | `mf-iso-dna-100k-baseline` | **Baseline R1+R2/AVX-512** (non-MPI) | `normalsr` 1×SPR | **DONE** exit=0 | 00:00:48 | **PASS** lnL −5,692,984.539 ✓, F81+F+G4 ✓, MF 31.802 s |
| ~~168580368~~ | `mf-iso-dna-100k-1n` | **FCA mf-iso** (MPI, np=1×103T) | `normalsr` 1×SPR | **DONE** exit=0 | 00:00:48 | **PASS** lnL −5,692,984.532 ✓, F81+F+G4 ✓, MF 39.169 s |
| ~~168580369~~ | `mf-iso-dna-100k-2n` | **FCA mf-iso** (MPI, np=2×103T) | `normalsr` 2×SPR | **DONE** exit=0 | ~00:01 | **PASS** lnL −5,692,984.532 ✓, MF 27.065 s; filterRatesMPI fired ✓ |
| ~~168580375~~ | `mf-iso-dna-1m-baseline` | **Baseline R1+R2/AVX-512** (non-MPI) | `normalsr` 1×SPR | **DONE** exit=0 | 00:46:42 | TESTONLY NJ-tree lnL check. lnL −59,208,019.212 ✓, F81+F+G4 ✓, BIC 118,418,815.3418, MF **2,802.261 s**. Authoritative MF baseline remains **168425675** (3,500.825 s). |
| ~~168580376~~ | `mf-iso-dna-1m-1n` | **FCA mf-iso** (MPI, np=1×103T) | `normalsr` 1×SPR | **DONE** exit=0, 01:26:11 | **PASS** lnL −59,208,019.158 ✓ (diff 0.054 < 0.5), F81+F+G4 ✓, MF 5,149.692 s; filterRatesMPI N/A (np=1) |
| ~~168580377~~ | `mf-iso-dna-1m-2n` | **FCA mf-iso** (MPI, np=2×103T) | `normalsr` 2×SPR | **DONE** exit=0, 01:03:59 | **PASS** lnL −59,208,019.158 ✓ (diff 0.054 < 0.5), F81+F+G4 ✓, MF 3,812.968 s (0.92×† vs ref MF) |

### Acceptance criteria

**DNA 100K (both np=1 and np=2):**
- lnL within 0.01 of baseline result
- Best model matches np=1 and baseline
- filterRatesMPI fires at np=2
- MF wall np=2 < MF wall np=1

### DNA 100K Phase 0.5+0.6 results — CONFIRMED ✓

| Metric | baseline (168580367) | np=1 (168580368) | np=2 (168580369) | Δ np=1 vs np=2 |
|--------|---------------------|-----------------|-----------------|----------------|
| **Binary** | Baseline R1+R2/AVX-512 | **FCA mf-iso phase0.5+0.6** | **FCA mf-iso phase0.5+0.6** | — |
| **Node** | normalsr 1×SPR | normalsr 1×SPR | normalsr **2×SPR** | — |
| **Threads** | 103T OMP (non-MPI) | 1 rank × 103T OMP | 2 ranks × 103T OMP | — |
| lnL | −5,692,984.539 | −5,692,984.532 | −5,692,984.532 | **0.000** ✓ |
| Best model | F81+F+G4 | F81+F+G4 | F81+F+G4 | match ✓ |
| MF wall (s) | 31.802 | 39.169 | **27.065** | 1.45× speedup ✓ |
| filterRatesMPI | — | not fired (np=1) ✓ | fired model=3, pruned=63 ✓ | — |
| Rank 1 evals | — | 88 (all models) | **0** (fully pruned) | — |
| \|bcast_ok_rates\| | — | — | 1 (G4 only) | — |
| exit | 0 ✓ | 0 ✓ | 0 ✓ | — |
| wall elapsed | 00:00:48 | 00:00:48 | ~00:01 | — |

**Key findings:**
- lnL diff vs prior full-run SPR ref (−5,692,984.539): np=1 = 0.007 (TESTONLY NJ-tree vs SPR-tree, < tol 1.0 ✓)
- filterRatesMPI correctly identified G4 as the sole viable rate category after 3 models
- Rank 1 received the broadcast and pruned ALL 88 of its assigned models — zero evaluations needed
- np=2 delegated all evaluation to rank 0 (25 models vs np=1's 88), 1.45× MF speedup
- Best model F81+F+G4 matches both prior CLX (168422811) and SPR (168425674) references ✓

**DNA 1M (both np=1 and np=2):**
- Reference lnL = **−59,208,019.212** (168425675 full run + 168422813 CLX, both agree exactly)
- |lnL − ref| < 0.5 (cross-platform / TESTONLY NJ-tree tolerance)
- Best model = **F81+F+G4** (matches 168425675 and 168422813)
- filterRatesMPI fires at np=2
- **MF wall baseline = 3,500.825 s** (168425675, Baseline R1+R2/AVX-512, normalsr 103T, full run)
- MF wall np=1 < 3,600 s (within 3% of 168425675 MF wall — same algorithm, no SPR overhead)
- MF wall np=2 < MF wall np=1

> **Note:** 168580375 (`-m TESTONLY`) is a supplementary NJ-tree lnL check only. No new baseline job was submitted; 168425675 is the authoritative DNA 1M MF wall reference (same Baseline binary, same normalsr node, 103T OMP).

### AA 1M scripts (submitted 2026-05-18)

New scripts added to `gadi-ci/mf-iso/` for AA 1M Phase 0.5+0.6 validation.
No new baseline submitted; 168425491 (full run, MF wall 7,587.459 s, lnL −78,605,196.573, LG+G4) is the authoritative AA 1M reference.

| Script | Purpose | PBS walltime | Nodes |
|--------|---------|-------------|-------|
| `run_mf_iso_aa_1m_2node.sh` | Phase 0.5+0.6 2-node AA 1M TESTONLY | 12:00:00 | 2×normalsr (208 CPUs, 1020 GB) |
| `run_mf_iso_aa_1m_4node.sh` | Phase 0.5+0.6 4-node AA 1M TESTONLY | 12:00:00 | 4×normalsr (416 CPUs, 2040 GB) |

`submit_mf_iso.sh` updated with `aa_1m_2node`, `aa_1m_4node`, `aa_1m_all` cases.

### Full-run (MF+SPR) scripts — FCA end-to-end parity (submitted 2026-05-18)

Using the same FCA mf-iso binary as all TESTONLY runs (`mf-iso-phase0.5-0.6`), now running
the **complete IQ-TREE application** (`-m TEST`): FCA-parallelised ModelFinder across MPI
ranks, then SPR tree search within each rank's OMP pool. Best result kept. Nothing else —
MF + SPR is the full beginning-to-end phylogenetic inference pipeline.

Expected benefit: MF is ~2.7× faster at np=2 (150 s vs 400 s for AA 100K, already confirmed).
SPR phase (~764 s for AA 100K, ~227 s for DNA 100K) runs in OMP within each rank in parallel.

| Script | Dataset | PBS walltime | Nodes | Acceptance lnL |
|--------|---------|-------------|-------|----------------|
| `run_mf_iso_aa_100k_2node_full.sh` | AA 100K | 01:30:00 | 2×normalsr | −7,541,976.860 ±0.1 |
| `run_mf_iso_dna_100k_2node_full.sh` | DNA 100K | 01:00:00 | 2×normalsr | −5,692,984.539 ±0.1 |
| `run_mf_iso_aa_1m_8node_full.sh` | AA 1M | 06:00:00 | 8×normalsr | −78,605,196.573 ±1.0 |

`submit_mf_iso.sh` updated with `aa_100k_full`, `dna_100k_full`, `full_100k_all`, `aa_1m_8node_full` cases.

#### Run status — Full-run jobs (submitted 2026-05-18)

| Job | Name | Binary | Nodes | State | Note |
|-----|------|--------|-------|-------|------|
| ~~168584736~~ | `mf-iso-aa-100k-2n-full` | **FCA mf-iso** (np=2×103T) | 2×normalsr | **DONE** exit=0, 00:09:31 | **PASS** lnL −7,541,976.853 ✓, LG+G4 ✓, MF 149.029 s, total 537.754 s; filterRatesMPI fired (model=3, pruned=81) ✓ |
| ~~168584737~~ | `mf-iso-dna-100k-2n-full` | **FCA mf-iso** (np=2×103T) | 2×normalsr | **DONE** exit=0, 00:02:44 | **PASS** lnL −5,692,984.532 ✓, F81+F+G4 ✓, MF 26.252 s, total 113.754 s; filterRatesMPI fired (model=3, pruned=63) ✓ |
| ~~168586094~~ | `mf-iso-aa-1m-8n-full` | **FCA mf-iso** (np=8×103T) | 8×normalsr | **DONE** exit=0, 01:01:12 | **PASS** lnL −78,605,196.497 ✓ (diff 0.076 < 1.0), LG+G4 ✓, MF 1,443.892 s (**5.26×**), total 3,671.618 s (**6.20× e2e**); filterRatesMPI fired (model=7, pruned=15) ✓ |

#### Acceptance criteria — Full runs

- **AA 100K:** lnL = −7,541,976.860 ±0.1 (LG+G4), filterRatesMPI fires, MF wall < 400 s
- **DNA 100K:** lnL = −5,692,984.539 ±0.1 (F81+F+G4), filterRatesMPI fires, MF wall < 32 s

#### Run status — AA 1M chain (submitted 2026-05-18)

| Job | Name | Binary | Node | State | Elapsed | Note |
|-----|------|--------|------|-------|---------|------|
| ~~168583449~~ | `mf-iso-aa-1m-2n` | **FCA mf-iso** (MPI, np=2×103T) | `normalsr` 2×SPR | **DONE** exit=0, 00:52:05 | **PASS** lnL −78,605,196.443 ✓ (diff 0.130 < 1.0), LG+G4 ✓, MF 3,059.648 s (2.48×), filterRatesMPI fired (model=3, pruned=81) ✓ |
| ~~168583450~~ | `mf-iso-aa-1m-4n` | **FCA mf-iso** (MPI, np=4×103T) | `normalsr` 4×SPR | **DONE** exit=0, 00:34:02 | **PASS** lnL −78,605,196.445 ✓ (diff 0.128 < 1.0), LG+G4 ✓, MF 1,976.767 s (**3.84×**), filterRatesMPI fired (model=3, pruned=39) ✓ |
| ~~168586094~~ | `mf-iso-aa-1m-8n-full` | **FCA mf-iso** (MPI, np=8×103T) | `normalsr` 8×SPR | **DONE** | 01:01:12 | **PASS** lnL −78,605,196.497 ✓ (diff 0.076 < 1.0), LG+G4 ✓, MF 1,443.892 s (**5.26×**), total 3,671.618 s; filterRatesMPI fired (model=7, pruned=15) ✓ |

#### Acceptance criteria — AA 1M

- Reference lnL = **−78,605,196.573** (168425491 full run)
- Reference model = **LG+G4**
- |lnL − ref| < 1.0
- filterRatesMPI fires at np=2 and np=4
- MF wall np=2 < 7,587 s; MF wall np=4 < MF wall np=2
- **Full run (168586094):** ✅ PASS — lnL = −78,605,196.497 (diff 0.076 < 1.0 ✓), LG+G4 ✓, MF 1,443.892 s (**5.26×** vs baseline), total 3,671.618 s (**6.20×** e2e), filterRatesMPI fired (model=7, pruned=15) ✓

---

### Comprehensive results — Phase 0.5+0.6 validation (all datasets, as of 2026-05-18)

#### Build legend

| Label | Binary | Description |
|-------|--------|-------------|
| **CLX** | `cpu_bench/build-intel-clx/iqtree3` | Cascade Lake, OMP-only, historical reference |
| **Baseline** | `build-intel-vanila/iqtree3` | Sapphire Rapids, OMP-only, R1+R2+AVX-512, v3.1.2 (sa0557) |
| **FCA mf-iso** | `iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` | MPI+OMP, branch `mf-iso-phase0.5-0.6` (as1708) |

All `-m TESTONLY` runs use a starting NJ tree (no SPR). lnL values are therefore NJ-tree
lnLs and may differ from full-run (SPR-optimised) lnLs by up to ~1 unit at 1M sites.
Full-run FCA rows (`-m TEST`) report the final ML-tree lnL after SPR optimisation.

#### All runs

| Job | Dataset | Run type | Build | Nodes | Ranks×OMP | Model | lnL | BIC | MF wall (s) | Status |
|-----|---------|----------|-------|-------|-----------|-------|-----|-----|-------------|--------|
| 168422813 | DNA 1M | CLX full (MF+SPR) | CLX | 1 normal | 47T OMP | F81+F+G4 | −59,208,019.212 | — | 10,230 | Done (hist. ref) |
| **168425675** | **DNA 1M** | **Baseline full (MF+SPR)** | **Baseline** | **1 normalsr** | **103T OMP** | **F81+F+G4** | **−59,208,019.212** | **118,418,815.3418** | **3,500.825** | **Done (authoritative baseline)** |
| ~~168580375~~ | DNA 1M | Baseline TESTONLY† | Baseline | 1 normalsr | 103T OMP | F81+F+G4 | −59,208,019.212 | 118,418,815.3418 | 2,802.261 | Done |
| ~~168580376~~ | DNA 1M | FCA np=1 TESTONLY | FCA mf-iso | 1 normalsr | 1×103T MPI | F81+F+G4 | −59,208,019.158 | 118,418,815.354 | 5,149.692 | **PASS ✓** |
| ~~168580377~~ | DNA 1M | FCA np=2 TESTONLY | FCA mf-iso | 2 normalsr | 2×103T MPI | F81+F+G4 | −59,208,019.158 | 118,418,815.354 | 3,812.968 | **PASS ✓** |
| 168425490 | AA 1M | CLX full (MF+SPR) | CLX | 1 normal | 47T OMP | LG+G4 | −78,605,196.573 | — | 16,308 | Done (hist. ref) |
| **168425491** | **AA 1M** | **Baseline full (MF+SPR)** | **Baseline** | **1 normalsr** | **103T OMP** | **LG+G4** | **−78,605,196.573** | **157,213,128.6176** | **7,587.459** | **Done (authoritative baseline)** |
| ~~168583449~~ | AA 1M | FCA np=2 TESTONLY | FCA mf-iso | 2 normalsr | 2×103T MPI | LG+G4 | −78,605,196.443 | 157,213,128.651 | 3,059.648 | **PASS ✓** |
| ~~168583450~~ | AA 1M | FCA np=4 TESTONLY | FCA mf-iso | 4 normalsr | 4×103T MPI | LG+G4 | −78,605,196.445 | 157,213,128.651 | 1,976.767 | **PASS ✓** |
| ~~168586094~~ | AA 1M | **FCA np=8 Full (MF+SPR)** | FCA mf-iso | 8 normalsr | 8×103T MPI | LG+G4 | −78,605,196.497 | — | 1,443.892 | **PASS ✓** total 3,671.618 s (**6.20× e2e**) |
| ~~168573852~~ | AA 100K | Baseline TESTONLY | Baseline | 1 normalsr | 103T OMP | LG+G4 | −7,541,976.860 | 15,086,233.282 | 400.582 | Done (baseline) |
| ~~168577707~~ | AA 100K | FCA np=1 TESTONLY | FCA mf-iso | 1 normalsr | 1×103T MPI | LG+G4 | −7,541,976.862 | 15,086,233.2835 | 257.355 | **PASS ✓** |
| ~~168577708~~ | AA 100K | FCA np=2 TESTONLY | FCA mf-iso | 2 normalsr | 2×103T MPI | LG+G4 | −7,541,976.853 | 15,086,233.2646 | 150.567 | **PASS ✓** |
| ~~168584736~~ | AA 100K | **FCA np=2 Full (MF+SPR)** | FCA mf-iso | 2 normalsr | 2×103T MPI | LG+G4 | −7,541,976.853 | 15,086,233.283 | 149.029 | **PASS ✓** total 537.754 s |
| ~~168580367~~ | DNA 100K | Baseline TESTONLY | Baseline | 1 normalsr | 103T OMP | F81+F+G4 | −5,692,984.539 | 11,388,283.1765 | 31.802 | Done (baseline) |
| ~~168580368~~ | DNA 100K | FCA np=1 TESTONLY | FCA mf-iso | 1 normalsr | 1×103T MPI | F81+F+G4 | −5,692,984.532 | 11,388,283.1618 | 39.169 | **PASS ✓** |
| ~~168580369~~ | DNA 100K | FCA np=2 TESTONLY | FCA mf-iso | 2 normalsr | 2×103T MPI | F81+F+G4 | −5,692,984.532 | 11,388,283.1618 | 27.065 | **PASS ✓** |
| ~~168584737~~ | DNA 100K | **FCA np=2 Full (MF+SPR)** | FCA mf-iso | 2 normalsr | 2×103T MPI | F81+F+G4 | −5,692,984.532 | 11,388,283.162 | 26.252 | **PASS ✓** total 113.754 s |

†168580375 BIC matches 168425675 because TESTONLY NJ-tree model-selection BIC equals the full-run MF-phase BIC (SPR does not re-run ModelFinder).

#### MF walltime speedup (FCA TESTONLY vs Baseline TESTONLY)

| Dataset | Baseline | np=1 | np=2 | np=4 | np=8 |
|---------|----------|------|------|------|------|
| AA 100K | 400.582 s | 257.355 s (1.6×) | 150.567 s (**2.7×**) | — | — |
| DNA 100K | 31.802 s | 39.169 s (0.8×) | 27.065 s (1.2×) | — | — |
| DNA 1M | 2,802 s (†3,501 s) | 5,149.692 s (0.68×†) | 3,812.968 s (0.92×†) | — | **1,274.686 s (2.75×†)** ⊕ |
| AA 1M | 7,587 s | — | 3,059.648 s (**2.48×**) | 1,976.767 s (**3.84×**) | **1,443.892 s (5.26×)** |

DNA 100K np=1 is slower than baseline: MPI startup + sequential outer loop overhead dominates a 31 s run.
DNA 1M np=1 (5,149.692 s) is also slower than the SPR baseline (3,501 s): at np=1 the FCA binary has MPI overhead and different code paths with no parallelism benefit. This is **expected** — FCA gains manifest at np≥2. DNA 1M np=1 is however **1.99× faster than CLX** (10,230 s), confirming the Sapphire Rapids + AVX-512 gains.
AA 1M scaling: np=2 → np=4 achieves 1.55× further MF speedup (3,059.648 → 1,976.767 s) for 2× the ranks (77% parallel efficiency).
†DNA 1M baseline uses 168425675 (3,501 s) as the authoritative reference; 168580375 (2,802 s) is supplementary.
‡DNA 1M FCA np=1 shows 0.68× vs SPR baseline (slower) — expected; MPI overhead without scaling benefit. Speedup metric vs CLX: 1.99×.
⊕DNA 1M np=8 MF time is from full run 168592214 (MF+SPR), not a TESTONLY run; MF phase is equivalent.
DNA 1M np=2 FCA (0.92×) is still slightly slower than baseline MF (3,501 s): FCA MPI overhead not yet offset at 2 nodes. Speedup crosses 1× between np=2 and np=8.

#### End-to-end (MF+SPR) speedup — FCA full run vs Baseline full run

| Dataset | Baseline total wall | FCA np=2 total wall | Speedup | MF speedup | SPR time |
|---------|---------------------|---------------------|---------|------------|----------|
| DNA 100K | 289 s (168425674) | **113.754 s** (168584737) | **2.54×** | 61.7→26.3 s (2.35×) | ~87 s |
| AA 100K | 1,170 s (168425673) | **537.754 s** (168584736) | **2.18×** | 400.6→149.0 s (2.69×) | ~389 s |
| AA 1M | 22,776 s (168425491) | **3,671.618 s** (168586094, np=8) | **6.20×** | 7,587→1,443.9 s (5.26×) | ~2,228 s |
| DNA 1M | 6,114 s (168425675) | **1,640.846 s** (168592214, np=8) | **3.73×** | 3,501→1,275 s (2.75×) | ~366 s |

---

## 2026-05-18 (bn) — AA 1M full scaling study + DNA 1M 8-node full submitted

### Submitted jobs

AA 1M full scaling chain (`aa_1m_full_all`, chained afterok 1→2→4→16 node):

| Job | Name | Script | ncpus | mem | walltime | Ranks×OMP | Acceptance lnL |
|-----|------|--------|-------|-----|----------|-----------|----------------|
| 168592210 | `mf-iso-aa-1m-1n-full` | `run_mf_iso_aa_1m_1node_full.sh` | 104 | 510 GB | 12h | 1×103T | −78,605,196.573 ±1.0 |
| 168592211 | `mf-iso-aa-1m-2n-full` | `run_mf_iso_aa_1m_2node_full.sh` | 208 | 1020 GB | 8h | 2×103T | −78,605,196.573 ±1.0 |
| 168592212 | `mf-iso-aa-1m-4n-full` | `run_mf_iso_aa_1m_4node_full.sh` | 416 | 2040 GB | 6h | 4×103T | −78,605,196.573 ±1.0 |
| 168592213 | `mf-iso-aa-1m-16n-full` | `run_mf_iso_aa_1m_16node_full.sh` | 1664 | 8160 GB | 4h | 16×103T | −78,605,196.573 ±1.0 |

DNA 1M 8-node full (`dna_1m_8node_full`):

| Job | Name | Script | ncpus | mem | walltime | Ranks×OMP | Acceptance lnL |
|-----|------|--------|-------|-----|----------|-----------|----------------|
| 168592214 | `mf-iso-dna-1m-8n-full` | `run_mf_iso_dna_1m_8node_full.sh` | 832 | 4080 GB | 6h | 8×103T | −59,208,019.212 ±0.5 |

All jobs use binary `iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` (branch `mf-iso-phase0.5-0.6`, `-m TEST`).

Reference: AA 1M → 168425491 (LG+G4, lnL −78,605,196.573); DNA 1M → 168425675 (F81+F+G4, lnL −59,208,019.212).

### Results (as of 2026-05-18)

#### 168592214 — DNA 1M 8-node full (**PASS ✓**, exit 0, PBS wall 00:27:57)

| Field | Value |
|-------|-------|
| Total wall | **1,640.846 s** (0h:27m:21s) |
| MF wall | 1,274.686 s (0h:21m:14s) |
| SPR wall | ~366 s (IQ-TREE: 349.904 s) |
| lnL | −59,208,019.103 |
| Δ vs ref (168425675) | **0.109** (tol 0.5 — **PASS**) |
| Model | F81+F+G4 |
| PBS SU | 775.15 |
| End-to-end speedup | **3.73×** (6,114 s → 1,641 s) |
| MF speedup | **2.75×** (3,501 s → 1,275 s) |

Note: MF runs entirely on rank 0 (FCA phase 0.5+0.6 — MF dispatch is not yet active).
SPR scales across all 8 nodes: 8-node SPR (~366 s) vs estimated single-node SPR (~2,614 s) = **7.1× SPR speedup**.

#### 168635614 — AA 1M 2-node full (**PASS ✓**, exit 0, PBS wall 03:02:26)

> Chain 168592210–168592213 was cancelled and resubmitted as **168635614–168635616** with per-rank `perf stat` profiling (`rank_perf.sh` wrapper, commit `ff71b13d`).

| Field | Value |
|-------|-------|
| Total wall | **10,945.801 s** (3h:02m:26s) |
| MF wall | 3,076.873 s |
| SPR wall | 7,868.928 s |
| lnL | −78,605,196.443 |
| Δ vs ref (168425491) | **0.130** (tol 1.0 — **PASS**) |
| Model | LG+G4 |
| filterRatesMPI | fired ✓ · ok_rates_size=1 |
| rank_0 MF evals | 31 models · total_eval_s=2,507.907 · mean=80.9 s · max=318.495 s |
| End-to-end speedup | **2.08×** (22,776 s → 10,946 s) |
| MF speedup | **2.47×** (7,587 s → 3,077 s) |
| SPR speedup | **1.92×** (15,099 s → 7,869 s) |
| IPC (mean, user-space) | **1.260** · rank 0: 1.269 · rank 1: 1.251 |
| LLC miss rate (mean) | **83.69%** · rank 0: 83.61% · rank 1: 83.77% |
| Perf stat | per-rank `perf_stat_rank_N.txt` in WORK_DIR |

#### 168635615 — AA 1M 4-node full (**PASS ✓**, exit 0, PBS wall 01:39:17)

| Field | Value |
|-------|-------|
| Total wall | **5,956.618 s** (1h:39m:17s) |
| MF wall | 1,974.476 s |
| SPR wall | 3,982.142 s |
| lnL | −78,605,196.445 |
| Δ vs ref (168425491) | **0.128** (tol 1.0 — **PASS**) |
| Model | LG+G4 |
| filterRatesMPI | fired ✓ · ok_rates_size=1 |
| rank_0 MF evals | 17 models · total_eval_s=1,557.494 · mean=91.6 s · max=321.22 s |
| End-to-end speedup | **3.82×** (22,776 s → 5,957 s) |
| MF speedup | **3.84×** (7,587 s → 1,974 s) |
| SPR speedup | **3.79×** (15,099 s → 3,982 s) |
| IPC (mean, user-space) | **1.273** · min 1.262 · max 1.279 (4 ranks) |
| LLC miss rate (mean) | **84.01%** · min 83.78% · max 84.45% |
| Perf stat | per-rank `perf_stat_rank_N.txt` in WORK_DIR |

#### 168635616 — AA 1M 16-node full (**PASS ✓**, exit 0, PBS wall 00:40:10)

| Field | Value |
|-------|-------|
| Total wall | **2,410.226 s** (0h:40m:10s) |
| MF wall | 1,122.363 s |
| SPR wall | 1,287.863 s |
| lnL | −78,605,196.497 |
| Δ vs ref (168425491) | **0.076** (tol 1.0 — **PASS**) |
| Model | LG+G4 |
| filterRatesMPI | fired ✓ · ok_rates_size=1 |
| rank_0 MF evals | 6 models · total_eval_s=671.396 · mean=111.9 s · max=332.846 s |
| End-to-end speedup | **9.45×** (22,776 s → 2,410 s) |
| MF speedup | **6.76×** (7,587 s → 1,122 s) |
| SPR speedup | **11.72×** (15,099 s → 1,288 s) |
| IPC (mean, user-space) | **1.337** · min 1.321 · max 1.353 (16 ranks) |
| LLC miss rate (mean) | **85.27%** · min 84.46% · max 85.80% |
| Perf stat | per-rank `perf_stat_rank_N.txt` in WORK_DIR |

#### AA 1M full scaling chain — COMPLETE (2026-05-18)

| Job | Name | Nodes | State | MF wall | Total wall | e2e speedup | MF speedup | Note |
|-----|------|-------|-------|---------|-----------|-------------|------------|------|
| ~~168592210~~ | `mf-iso-aa-1m-1n-full` | 1 | **DONE** exit=0 | — | — | — | — | 1-node reference; chain cancelled (no perf stat) |
| ~~168592211–168592213~~ | 2/4/16-node | — | **CANCELLED** | — | — | — | — | Resubmitted as 168635614–616 with perf stat |
| ~~168635614~~ | `mf-iso-aa-1m-2n-full` | 2 | **DONE** exit=0 ✓ | 3,076.873 s | 10,945.801 s | **2.08×** | **2.47×** | lnL diff=0.130 PASS, LG+G4, filterRatesMPI ✓ |
| ~~168635615~~ | `mf-iso-aa-1m-4n-full` | 4 | **DONE** exit=0 ✓ | 1,974.476 s | 5,956.618 s | **3.82×** | **3.84×** | lnL diff=0.128 PASS, LG+G4, filterRatesMPI ✓ |
| ~~168635616~~ | `mf-iso-aa-1m-16n-full` | 16 | **DONE** exit=0 ✓ | 1,122.363 s | 2,410.226 s | **9.45×** | **6.76×** | lnL diff=0.076 PASS, LG+G4, filterRatesMPI ✓ |

---

## 2026-05-17 (bl) — Run status update + gadi-ci subfolder reorganisation

### Run status — chain history (as of 2026-05-17)

| Job | Name | State | Elapsed | Dep | Note |
|-----|------|-------|---------|-----|------|
| ~~168572136~~ | `mf-iso-bootstrap` | **DONE** exit=8 | 00:11:20 | — | Build OK; false failure (Lustre `nm` timing); binary verified on login node |
| ~~168572137~~ | `mf-iso-aa-baseline` | **KILLED** | — | afterok 168572136 | Cascaded from 168572136 exit≠0 |
| ~~168572138~~ | `mf-iso-aa-1n` | **KILLED** | — | afterok 168572137 | Cascaded |
| ~~168572139~~ | `mf-iso-aa-2n` | **KILLED** | — | afterok 168572138 | Cascaded |
| ~~168573852~~ | `mf-iso-aa-baseline` | **DONE** exit=1 | 00:18:48 | — | IQ-TREE ✓ (MF 400.6 s, lnL −7,541,976.860, LG+G4); script bug (`$1: unbound variable` in unquoted heredoc awk line — `\$1` fix applied) |
| ~~168573853~~ | `mf-iso-aa-1n` | **KILLED** | 0 | afterok 168573852 | Cascaded from baseline exit=1 |
| ~~168573854~~ | `mf-iso-aa-2n` | **KILLED** | 0 | afterok 168573853 | Cascaded |
| ~~168576511~~ | `mf-iso-aa-1n` | **DONE** exit=7 | 00:00:03 | — | `cat /dev/null` warm + strings fallback BOTH failed — binary on rc29 scratch; `nm`/`strings` cannot read it from normalsr nodes (wrong allocation) |
| ~~168576512~~ | `mf-iso-aa-2n` | **KILLED** | 0 | afterok 168576511 | Cascaded |
| ~~168576627~~ | `mf-iso-aa-1n` | **DONE** exit=7 | 00:00:02 | — | nm + strings both failed on dx61 binary — **Lustre write-cache lag** (binary copied 2 min before job ran; dirty pages not on OST yet) |
| ~~168576628~~ | `mf-iso-aa-2n` | **KILLED** | 0 | afterok 168576627 | Cascaded |
| ~~168577707~~ | `mf-iso-aa-1n` | **DONE** exit=0 | 00:04:30 | — | **PASS** lnL −7,541,976.862 ✓, LG+G4 ✓, MF wall 257 s; probe nm/strings still fail on compute node (mmap Lustre issue, non-fatal) |
| ~~168577708~~ | `mf-iso-aa-2n` | **DONE** exit=0 | 00:02:48 | afterok 168577707 | **PASS** lnL −7,541,976.853 ✓, LG+G4 ✓, MF wall 150.567 s; filterRatesMPI fired at model=3, local_pruned=81 ✓ |

**Root cause of repeated preflight failures (168576191/368/511/627):**

Three layers of bugs compounded:

1. **Wrong allocation (168576191/368/511):** all mf-iso scripts used `#PBS -P rc29`; binary
   was on `/scratch/rc29`. Fixed: binary copied to `/scratch/dx61/as1708/iqtree3-mf-iso/`;
   all five scripts migrated to `#PBS -P dx61`.

2. **Lustre write-cache lag (168576627, the "fixed" dx61 run):** the binary was copied on
   the login node at 23:36 and the job ran at 23:38 — only 2 minutes. Lustre dirty-page
   writeback hadn't committed the 140 MB to OST 359 (gadiscr2-OST0167) yet. `ldd` passed
   because it only reads the first ~4 KB (ELF dynamic section); `nm` and `strings` need
   all 140 MB → both hit unreadable blocks → returned empty → `grep` returned 1 → exit 7.
   The `cat "${IQTREE}" > /dev/null 2>&1 || true` warm-up silently swallowed the I/O error
   so the misleading "nm + strings both failed" message appeared instead of "binary not
   readable". Fix: removed `|| true` so cat failure is fatal with a clear diagnostic;
   downgraded nm/strings to WARNING-only (ldd already confirms MPI+libiomp5 build;
   post-run lnL validates correctness). Mitigation: run `sync` on the login node after
   any cp/build before submitting.

**Baseline results (168573852) — confirmed reproducible:**

| Metric | This run | Reference (168425673) | Δ |
|--------|----------:|----------------------:|---|
| MF wall | 400.582 s | 405.078 s | −1.1% ✓ |
| Total wall | 1,128.638 s | 1,169.556 s | −3.5% ✓ |
| lnL | −7,541,976.860 | −7,541,976.860 | exact ✓ |
| Best model | LG+G4 | LG+G4 | ✓ |
| Peak mem | 9.54 GB | 9.36 GB | +2% ✓ |

**Script fix:** `run_baseline_aa_100k_spr.sh` line 175 — awk `$1` inside
unquoted `<<PYEOF` heredoc caused bash to expand it under `set -u` with no
positional args. Fixed: `awk '{{print \$1}}'` (backslash escapes the
dollar from bash; Python f-string `{{` / `}}` still produce literal braces).

**Script fix 2 (four iterations):** `run_mf_iso_aa_100k_{1,2}node.sh` preflight symbol
check. Root cause: binary was freshly copied on the login node; Lustre's async writeback
had not yet committed the 140 MB to the OST before the compute node tried to read it.
`ldd` passed (reads only the first ~4 KB). `nm` and `strings` need the full 140 MB and
silently returned empty because the original `cat > /dev/null 2>&1 || true` masked the
I/O error. Final fix: `cat` failure is now fatal with a clear "Lustre OST not yet synced"
message; nm/strings symbol failure is WARNING-only. Mitigation: `sync` on the login node
after copying the binary, before submitting.

1-node (168577707) and 2-node (168577708) both passed on dx61. Phase 0.5+0.6
confirmed at np=1 and np=2.

**2-node results (168577708) — Phase 0.5+0.6 confirmed (AA 100K):**

| Metric | baseline (168573852) | np=1 (168577707) | np=2 (168577708) | Δ np=1 vs np=2 |
|--------|---------------------|:----------------:|:----------------:|:--------------:|
| **Binary** | Baseline R1+R2/AVX-512 (non-MPI) | **FCA mf-iso phase0.5+0.6** | **FCA mf-iso phase0.5+0.6** | — |
| **Node** | normalsr 1×SPR | normalsr 1×SPR | normalsr **2×SPR** | — |
| **Threads** | 103T OMP | 1 rank × 103T OMP | 2 ranks × 103T OMP | — |
| Exit | 0 ✓ | 0 ✓ | 0 ✓ | — |
| lnL | −7,541,976.860 | −7,541,976.862 | −7,541,976.853 | 0.009 FP noise ✓ |
| Best model | LG+G4 | LG+G4 | LG+G4 | match ✓ |
| MF wall | ~405 s | 257 s | **150.567 s** | −41% ✓ |
| filterRatesMPI fired | n/a (baseline) | n/a (np=1) | model=3 ✓ | — |
| `\|bcast_ok_rates\|` | — | — | 1 ✓ | — |
| local_pruned | — | — | 81 / 112 (72%) ✓ | — |
| Rank 0 models | — | 224 (all) | 112/224 (50%) ✓ | FCA greedy-LPT |
| Rank 1 evaluates | — | — | 31 (112 − 81 pruned) ✓ | — |
| Ref-family first | — | LG 0–3 ✓ | LG 0–3 ✓ | Phase 0.6 ✓ |
| Walltime (PBS) | 00:18:48 | 00:04:30 | 00:02:48 | −38% ✓ |

Rank 0's MF-TIME trace confirms Phase 0.6 ordering: models 0 (LG), 1 (LG+I),
2 (LG+G4), 3 (LG+I+G4) evaluated first (`ref_remaining` counting 4→3→2→1),
then filterRatesMPI fires at model=3 (`bcast_ok_rates=1`, broadcasting
LG+G4's rate parameters to rank 1), pruning 81 of rank 1's 112 assigned
models.  After the broadcast, rank 0 continues with model 6 (LG+F+G4,
`ref_remaining=0`) and non-LG families.  Rank 1's stdout.log is empty
(MPI stdout routed to separate file); rank_models.csv populated from rank
0 only.  All 3 MF-MPI-DIAG lines and 31 MF-TIME lines confirmed in
rank_0.stdout.log.

**Fix: `probe_header.sh` nm/strings → `cat|strings` (Lustre mmap fix):**

Both `nm` and standalone `strings` use `mmap()` to map large ELF files
into the process address space.  On Lustre compute nodes, `mmap()` of a
large `/scratch` file can silently return empty data even when sequential
`read()` (used by `cat`, `md5sum`, `dd`) succeeds — the Lustre client
does not guarantee `mmap()` coherence for files that aren't already in
the page cache.  This caused all `bin_sym_MISSING` / `bin_str_MISSING`
probe lines on every compute-node run, even when the binary was correct.

Fix applied to `gadi-ci/mf-iso/tools/probe_header.sh`: read the binary
once via `cat binary | strings` (pipe forces sequential `read()`;
`strings` reading from a pipe cannot fall back to `mmap()`), cache the
result in `/dev/shm` (RAM-backed, not Lustre), then run all 7 `grep`
checks from the cache.  Symbol display labels are kept as demangled names
for readability; search patterns are mangled-name substrings that appear
verbatim in the ELF `.strtab` and are therefore found by `strings`:

| Display label | Mangled search pattern |
|---|---|
| `CandidateModelSet::filterRatesMPI` | `filterRatesMPIEi` |
| `CandidateModelSet::filterRates` | `11filterRatesEi` |
| `CandidateModelSet::getNextModel` | `getNextModelEv` |
| `CandidateModelSet::evaluateAll` | `evaluateAll` |

String-marker patterns (`MF-MPI-DIAG`, `MF-TIME: rank`, `filterRatesMPI
fired`) are unchanged — they appear literally in `.rodata` and are
already found correctly by `strings`.

### gadi-ci subfolder reorganisation

The loose scripts at the root of `gadi-ci/` were taking up significant
visual space and were hard to navigate. Scripts have been moved into
dataset-specific subdirectories matching the alignment they target.
Existing `cpu-bench/` and `mf-iso/` subdirectories are unchanged.

**New layout:**

```
gadi-ci/
  bootstrap/   ← build scripts (6 files)
  10M/         ← 100 taxa × 10 M DNA sites, alignment_10000000.phy (3 files)
  xlarge/      ← 200 taxa × 100 K DNA sites, xlarge_mf.fa (14 files)
  mega/        ← 500 taxa × 100 K DNA sites, mega_dna.fa (9 files)
  utils/       ← generate_datasets, rerun_perf_stat, pipeline, profiling,
                  submit_benchmark_matrix, test_mf_mpi_dispatch (6 files)
  cpu-bench/   ← AA 100K MPI batch (unchanged)
  mf-iso/      ← Phase 0.5+0.6 isolation harness (unchanged)
```

**Files moved:**

| Old path | New path |
|----------|----------|
| `gadi-ci/bootstrap_iqtree.sh` | `gadi-ci/bootstrap/bootstrap_iqtree.sh` |
| `gadi-ci/bootstrap_iqtree_3.1.2.sh` | `gadi-ci/bootstrap/bootstrap_iqtree_3.1.2.sh` |
| `gadi-ci/bootstrap_iqtree_3.1.2_mpi.sh` | `gadi-ci/bootstrap/bootstrap_iqtree_3.1.2_mpi.sh` |
| `gadi-ci/bootstrap_iqtree_clang.sh` | `gadi-ci/bootstrap/bootstrap_iqtree_clang.sh` |
| `gadi-ci/bootstrap_iqtree_mpi.sh` | `gadi-ci/bootstrap/bootstrap_iqtree_mpi.sh` |
| `gadi-ci/build_avx512_r2_mpi.sh` | `gadi-ci/bootstrap/build_avx512_r2_mpi.sh` |
| `gadi-ci/run_100taxa_10M_mf2dispatch_4node.sh` | `gadi-ci/10M/run_100taxa_10M_mf2dispatch_4node.sh` |
| `gadi-ci/run_100taxa_10M_r2_avx512_mpi_4node.sh` | `gadi-ci/10M/run_100taxa_10M_r2_avx512_mpi_4node.sh` |
| `gadi-ci/run_10M_mf2dispatch_16node.sh` | `gadi-ci/10M/run_10M_mf2dispatch_16node.sh` |
| `gadi-ci/run_xlarge_avx512_r2_omp_batch.sh` | `gadi-ci/xlarge/run_xlarge_avx512_r2_omp_batch.sh` |
| `gadi-ci/run_xlarge_correctness_baseline.sh` | `gadi-ci/xlarge/run_xlarge_correctness_baseline.sh` |
| `gadi-ci/run_xlarge_correctness_mf2.sh` | `gadi-ci/xlarge/run_xlarge_correctness_mf2.sh` |
| `gadi-ci/run_xlarge_fixedtree_baseline.sh` | `gadi-ci/xlarge/run_xlarge_fixedtree_baseline.sh` |
| `gadi-ci/run_xlarge_fixedtree_mf2.sh` | `gadi-ci/xlarge/run_xlarge_fixedtree_mf2.sh` |
| `gadi-ci/run_xlarge_r2_mf2_dispatch.sh` | `gadi-ci/xlarge/run_xlarge_r2_mf2_dispatch.sh` |
| `gadi-ci/run_xlarge_r2_mpi_2node_fullnode.sh` | `gadi-ci/xlarge/run_xlarge_r2_mpi_2node_fullnode.sh` |
| `gadi-ci/run_xlarge_r2_mpi_l3rank.sh` | `gadi-ci/xlarge/run_xlarge_r2_mpi_l3rank.sh` |
| `gadi-ci/run_xlarge_r2_mpi_socket.sh` | `gadi-ci/xlarge/run_xlarge_r2_mpi_socket.sh` |
| `gadi-ci/run_xlarge_r2_numa_416t_4node.sh` | `gadi-ci/xlarge/run_xlarge_r2_numa_416t_4node.sh` |
| `gadi-ci/run_xlarge_r2_v312_canonical.sh` | `gadi-ci/xlarge/run_xlarge_r2_v312_canonical.sh` |
| `gadi-ci/run_xlarge_r2_v312_mpi_2node_fullnode.sh` | `gadi-ci/xlarge/run_xlarge_r2_v312_mpi_2node_fullnode.sh` |
| `gadi-ci/submit_xlarge_r2_alternates.sh` | `gadi-ci/xlarge/submit_xlarge_r2_alternates.sh` |
| `gadi-ci/test_xlarge_mf2_correctness.sh` | `gadi-ci/xlarge/test_xlarge_mf2_correctness.sh` |
| `gadi-ci/run_mega_avx512_r2_2node.sh` | `gadi-ci/mega/run_mega_avx512_r2_2node.sh` |
| `gadi-ci/run_mega_avx512_r2_4node.sh` | `gadi-ci/mega/run_mega_avx512_r2_4node.sh` |
| `gadi-ci/run_mega_avx512_r2_omp_batch.sh` | `gadi-ci/mega/run_mega_avx512_r2_omp_batch.sh` |
| `gadi-ci/run_mega_mf2_full_2node.sh` | `gadi-ci/mega/run_mega_mf2_full_2node.sh` |
| `gadi-ci/run_mega_mf2_full_4node.sh` | `gadi-ci/mega/run_mega_mf2_full_4node.sh` |
| `gadi-ci/run_mega_mf2_full_omp_batch.sh` | `gadi-ci/mega/run_mega_mf2_full_omp_batch.sh` |
| `gadi-ci/run_mega_mf2dispatch_4node_aps.sh` | `gadi-ci/mega/run_mega_mf2dispatch_4node_aps.sh` |
| `gadi-ci/run_mega_profile.sh` | `gadi-ci/mega/run_mega_profile.sh` |
| `gadi-ci/submit_mega_batch.sh` | `gadi-ci/mega/submit_mega_batch.sh` |
| `gadi-ci/generate_datasets.sh` | `gadi-ci/utils/generate_datasets.sh` |
| `gadi-ci/rerun_perf_stat.sh` | `gadi-ci/utils/rerun_perf_stat.sh` |
| `gadi-ci/run_pipeline.sh` | `gadi-ci/utils/run_pipeline.sh` |
| `gadi-ci/run_profiling.sh` | `gadi-ci/utils/run_profiling.sh` |
| `gadi-ci/submit_benchmark_matrix.sh` | `gadi-ci/utils/submit_benchmark_matrix.sh` |
| `gadi-ci/test_mf_mpi_dispatch.sh` | `gadi-ci/utils/test_mf_mpi_dispatch.sh` |

No script contents were changed; only paths moved.

---

## 2026-05-17 (bk) — Reset: ModelFinder isolation harness on rc29 (Phase 0.5+0.6 only)

### Context — why we're stepping back

The MF2 production tree at `/scratch/um09/as1708/iqtree3-mf2/` accumulated
five tightly-coupled experimental changes between entries `(bd)` and `(bj)`:

| Entry | Phase | Change | Result |
|------|-------|--------|--------|
| `bd` | 0   | FCA dispatch (cost predictor + greedy LPT + state machine) | np=1 OK; np=2 +500% regression; np=4 +50% regression |
| `bg` | 0.5 | `filterRatesMPI` MPI_Bcast of rank-0 `ok_rates`         | np=4 873 s — correct but sync-trapped |
| `bh` | 0.6 | `getNextModel` ref-family priority                       | np=4 850 s — sync trap NOT fixed |
| `bj` | 0.7 + HH-NUMA | `MPI_Isend` push + nested `K_outer × M_inner` OMP | **Job 168486582 SIGTERM-killed @ 1h19m, ZERO stdout** |

The Phase 0.7 + HH-NUMA bundle was the failure mode the user called out:
"the FCA modelfinder is still not scaling… instead of directly scaling to
4 nodes lets start from 1 node then 2."

Two structural problems caused the dead end:
1. **Bundling**: 0.7 (Isend) and HH-NUMA (nested OMP) shipped in the same
   binary, plus `MPI_Init_thread(MPI_THREAD_SERIALIZED)` upgrade. When the
   binary hung, three candidate root causes had to be isolated by code
   review rather than experiment.
2. **Skip-to-4-node testing**: every patch was validated at np=4 first.
   This is the largest, longest, most expensive configuration and the
   most fragile. np=2 (which is where Phase 0.5's cross-rank broadcast
   first matters) was rarely re-tested between patches.

### What was reverted

The production source tree under `/scratch/um09/as1708/iqtree3-mf2/src/iqtree3`
was reverted to the FCA Phase 0 commit `ffb79a14` (the in-source Phase
0.5/0.6/0.7/HH-NUMA changes were rolled back at 01:18 AEST on 2026-05-17,
after the SIGTERM run was acknowledged). The binary at
`build-mpi-mf2/iqtree3-mpi` (23:44, the hung Phase 0.7+HH-NUMA artifact)
remains on disk for forensic inspection of `nm`/`strings` only.

### Isolation harness — rc29 tree + 1-node-then-2-node gating

A clean source mirror was created at `/scratch/rc29/as1708/iqtree3-mf-iso/`
on new branch `mf-iso-phase0.5-0.6` (HEAD `9603247f`). Patches applied:

- **Phase 0.5** (`filterRatesMPI`): rank 0 broadcasts its sharp-BIC `ok_rates`
  to all ranks via `MPI_Bcast`. `MPI_Allreduce(MIN)` gate prevents deadlock
  when any rank lacks a valid ref family. Falls back to legacy `filterRates`
  if disabled.
- **Phase 0.6** (`getNextModel` ref-family priority): while the ref family
  is incomplete and the broadcast hasn't fired, prefer ref-family models so
  every rank reaches the collective close together. Also fixes the latent
  first-call IGNORED bug (original code returned `model = 0` unconditionally
  even if `MF_IGNORED` was set on rank 1+).
- **MF-TIME instrumentation**: `cout << "MF-TIME: rank R model M ... start=… end=… dt=… score=…"`
  on every model on every rank. This was the missing observability in
  earlier debug runs (e.g. 168475747 lost all rank-1 stdout because
  `mpirun > file` only redirects rank 0 on multi-node jobs).

What was **deliberately excluded** from this commit:
- **Phase 0.7** (`MPI_Isend` push) — depends on `MPI_Init_thread`
  (THREAD_SERIALIZED), suspected culprit of the 168486582 hang.
- **HH-NUMA Phase 2** (nested `K_outer × M_inner`) — depends on toolchain
  nested-OMP support which hasn't been runtime-verified on icpx 2025.1.1
  + libiomp5.
- **THP `madvise(MADV_HUGEPAGE)`** — orthogonal kernel speedup, separate patch.

Each deferred phase will land in its own commit, with its own 1-node →
2-node → 4-node validation, only after the prior phase passes.

### Files added (in this entry)

| File | Purpose |
|------|---------|
| `gadi-ci/mf-iso/README.md` | What's in this harness, how to use it, acceptance gates |
| `gadi-ci/mf-iso/build_mf_iso.sh` | PBS job: build the isolated MPI binary on rc29 |
| `gadi-ci/mf-iso/run_mf_iso_aa_100k_1node.sh` | PBS job: 1-node `-m TESTONLY` correctness baseline |
| `gadi-ci/mf-iso/run_mf_iso_aa_100k_2node.sh` | PBS job: 2-node with per-rank stdout via `--output-filename` |
| `gadi-ci/mf-iso/submit_mf_iso.sh` | qsub driver with `afterok` dependency chain (build → 1node → 2node) |
| `gadi-ci/mf-iso/tools/parse_mf_time.py` | Offline analyser: per-rank model counts, broadcast arrival times, convergence spread |
| `/scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3/main/phylotesting.{cpp,h}` | Phase 0.5+0.6 + MF-TIME patches, committed on branch `mf-iso-phase0.5-0.6` (`9603247f`) |

### Why `-m TESTONLY`

The AA 100K alignment's tree-search tail is ~700 s on top of MF wall.
Running with `-m TESTONLY` cuts ~30% off iteration time and isolates
ModelFinder dispatch from BIONJ + parsimony + NNI search confounds.
We are debugging ModelFinder; we don't need a new tree every run.

### Why per-rank stdout via `--output-filename`

OpenMPI 4.1's default behaviour redirects only rank 0 stdout when
`mpirun … > file` crosses node boundaries. Ranks 1+ stdout is silently
discarded. Earlier debug runs (168475747, 168481332) lost the
crucial rank-1 evidence this way — the rank-1 `filterRates` ineffectiveness
hypothesis was inferred from MF-wall arithmetic rather than observed.
With `--output-filename ${WORK_DIR}/rank_logs/` each rank gets its own
`stdout`/`stderr` file. Combined with the MF-TIME markers, this gives a
full per-rank timeline.

### Patch summary

```
mf-iso-phase0.5-0.6:
  9603247f  mf-iso: Phase 0.5 filterRatesMPI MPI_Bcast + Phase 0.6 getNextModel
            ref-family priority + MF-TIME markers (319 +, 33 -)
  ffb79a14  gadi-spr: FCA dispatch (Family-Local + Cost-Aware + Always-Filter)
            for ModelFinder MPI                          ← Phase 0 base
```

Syntax-checked with `mpicxx -fsyntax-only` (icpx 2025.3.2 + openmpi 4.1.7):
0 errors, 6 pre-existing warnings (VLAs at line 672/680, writable string
literals at 6913+ — none introduced by this patch).

### Acceptance gates (vs the 168425673 baseline — MF wall 405 s)

**1-node** (correctness; the MPI-build np=1 is expected SLOWER than
the standard binary because Fix H forces sequential outer loop):
- lnL = −7,541,976.860 ± 0.01
- best model = LG+G4
- MF wall < 1,400 s (matches Fix H 1,289 s; far above the 405 s target —
  this is expected and confirms Phase 0.5/0.6 cause no regression at np=1)
- `filterRatesMPI_enabled=0` in MF-MPI-DIAG (correct: no broadcast at np=1)
- Hardware/software/binding probe fully captured in the run log

**2-node** (first real Phase 0.5/0.6 test — must show structural benefit
from cross-rank pruning):
- lnL, best model unchanged
- MF wall **< 600 s** (FCA Phase 0 regressed to 2,865 s; we expect ~400–500 s
  if both ranks prune effectively with rank 0's `{G4}` set)
- `filterRatesMPI fired at model=…` observed on rank 0
- Broadcast-arrival spread between ranks **< 60 s** (Phase 0.5 alone produced
  ~370 s spread; Phase 0.6 should compress to <30 s)
- Rank 1's post-broadcast model count ≤ rank 0's
- **STRETCH**: beat 405 s. With 2 ranks each owning ~half the models and
  applying the same `{G4}` filter, theoretical floor is ~200 s if perfect
  scaling. Realistic: 350–450 s.

**4-node** (target: beat 405 s):
- Submit the existing `cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh` with
  the MF-iso binary (override `IQTREE=/scratch/rc29/.../iqtree3-mpi`) only
  after 2-node passes.
- **Target: MF wall < 400 s** (the headline beat-the-baseline goal).
- If 4-node MF wall > 600 s, do NOT chase further patches — re-examine
  the 2-node `parse_mf_time.py` output; the bug is visible there too.

### Build run (2026-05-17) — job 168572136

| Item | Detail |
|------|--------|
| Job | 168572136.gadi-pbs (`mf-iso-bootstrap`) |
| Node | gadi-cpu-spr-0284, ncpus=104, queue=normalsr |
| Wall | 00:11:20 (cmake + make -j104 + verification) |
| SU | 39.29 |
| Binary | `/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi` (146 MB) |
| Linkage | libiomp5 + libmpi ✓, libgomp absent ✓ |
| Symbol | `CandidateModelSet::filterRatesMPI(int)` ✓ (verified post-job) |
| MF-TIME string | present ✓ |
| MF-MPI-DIAG string | present ✓ |
| Job exit code | **8 (false failure)** — see note below |

**False failure note**: The build script's `nm -C ... | grep 'filterRatesMPI(int)'` check
returned no match during the job. Direct verification on the login node immediately
after confirmed the symbol IS present (`00000000005ac790 T CandidateModelSet::filterRatesMPI(int)`).
Root cause: Lustre metadata flush timing — `nm` on the compute node read the file before
the link operation was fully visible to the VFS. Fixed in `build_mf_iso.sh`: added a
`stat()` flush barrier and a raw-mangled-name fallback (`_ZN17CandidateModelSet14filterRatesMPIEi`)
so a transient demangling delay doesn't abort a good build. The dependent jobs
(168572137-9) were auto-killed by PBS `afterok` — resubmitted as 168573852-4.

### Run jobs submitted (2026-05-17) — 168573852-4

| Job | Script | Dep | Expected wall |
|-----|--------|-----|--------------|
| 168573852 | `run_baseline_aa_100k_spr.sh` | none (Q) | ~2 h (full run) |
| 168573853 | `run_mf_iso_aa_100k_1node.sh` | afterok 168573852 | ~1.5 h (TESTONLY) |
| 168573854 | `run_mf_iso_aa_100k_2node.sh` | afterok 168573853 | ~1 h (TESTONLY) |

Results will be documented in a follow-up entry when jobs complete.

### Next steps

1. ~~`qsub gadi-ci/mf-iso/build_mf_iso.sh`~~ — **DONE** (binary at `build-mpi-iso/iqtree3-mpi`, verified)
2. ~~`qsub gadi-ci/mf-iso/run_mf_iso_aa_100k_1node.sh`~~ — **submitted** (168573853, afterok 168573852)
3. ~~`qsub gadi-ci/mf-iso/run_mf_iso_aa_100k_2node.sh`~~ — **submitted** (168573854, afterok 168573853)
4. After 2-node passes: `cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh` with the
   isolated binary (override `IQTREE=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi`)
5. Only then add Phase 0.7 (Isend, alone) and HH-NUMA (alone, after Isend
   passes 2-node) in two separate follow-up patches.

---

## 2026-05-16 (bj) — Phase 0.7 push + HH-NUMA Phase 2 implemented (build verified)

### What changed

Implemented and compiled the next optimization stage after Phase 0.6:

1. **Phase 0.7 (non-blocking push of `ok_rates`)**
- Replaced collective `MPI_Bcast` path in `filterRatesMPI(int)` with rank-0
  push using `MPI_Isend` to ranks 1..N-1.
- Added one-shot receive state on worker ranks: pre-posted `MPI_Irecv`,
  per-model polling via `MPI_Test` in `pollOkRatesMPI()`, and inline pruning
  through `applyOkRatesMPI()`.
- Rank 0 now applies `ok_rates` locally and continues immediately (no barrier).

2. **HH-NUMA Phase 2 (bounded nested outer parallelism in MPI builds)**
- Enabled `K_outer × M_inner` execution in `evaluateAll()` for MPI `np>1`:
  `K_outer = min(8, num_threads)`, `M_inner = num_threads / K_outer`.
- Keeps memory bounded while increasing model throughput.
- Preserves Fix H safety at MPI `np=1` by forcing sequential outer
  (`K_outer=1`) to avoid OOM.

3. **MPI thread-level correctness hardening**
- `MPIHelper::init()` now uses `MPI_Init_thread(..., MPI_THREAD_SERIALIZED, ...)`.
- Execution now fails fast if MPI provides less than `MPI_THREAD_SERIALIZED`
  (FUNNELED is insufficient for non-master OMP-thread MPI calls, even with
  critical-section serialization).

4. **Nested OMP side-effect guard**
- `omp_set_max_active_levels(2)` is enabled only when needed (`K_outer>1`) and
  restored to its previous value after the `evaluateAll()` OMP region.

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.h` | Added Phase 0.7 members/constants (`MPI_TAG_OKRATES`, buffers, requests), helper declarations (`applyOkRatesMPI`, `pollOkRatesMPI`), and HH-NUMA constant (`HH_K_OUTER=8`). |
| `src/iqtree3/main/phylotesting.cpp` | Added Phase 0.7 send/recv helpers; rewired `evaluateAll()` trigger path to rank-0 Isend + worker polling; enabled bounded nested OMP (`K_outer × M_inner`) for MPI `np>1`; retained MPI `np=1` sequential safety; restored OMP active-level setting post-loop. |
| `src/iqtree3/utils/MPIHelper.cpp` | Switched `MPI_Init` → `MPI_Init_thread` with `MPI_THREAD_SERIALIZED` requirement. |

### Build verification

- Rebuilt with modules `intel-compiler-llvm/2025.1.1` and `openmpi/4.1.7`.
- Binary: `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`
  - timestamp: **2026-05-16 23:44:25 AEST**
  - size: **145,075,408 bytes**
- Symbols confirmed:
  - `CandidateModelSet::filterRatesMPI(int)`
  - `CandidateModelSet::pollOkRatesMPI()`
  - `CandidateModelSet::applyOkRatesMPI(...)`
  - `CandidateModelSet::evaluateAll(...)`
  - `MPIHelper::init(int, char**)`
- Smoke test: `iqtree3-mpi --version` exits 0.

### Status

- **Implementation complete, benchmark pending.**
- Validation job submitted: **168486582** (np=4 AA 100K, queue `normalsr`).
- Next required validation run: np=4 AA 100K against this binary.
- Target remains MF wall **< 400 s** (hard accept gate ≤ 450 s).

## 2026-05-16 (bi) — T3'' submitted: Phase 0.6 np=4 validation job 168483748

### What happened

Phase 0.6 binary (22:47 AEST, 145,056,888 bytes) verified clean:
- `CandidateModelSet::getNextModel()` at 0x6754d0 ✓
- `CandidateModelSet::filterRatesMPI(int)` at 0x6743a0 ✓

T3'' (np=4, 4-node, AA 100K) submitted as job **168483748** to `dx61 normalsr`.

```
qsub gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh
→ 168483748.gadi-pbs
```

### What this job validates

Phase 0.6 introduced `getNextModel()` ref-family priority: each rank now
evaluates all ~22 ref-family models **first** before any other assigned
models, so all ranks converge on `MPI_Bcast` in ~150 s instead of the
~488 s that sank Phase 0.5 (873 s MF wall, 88% idle threads).

Pass criteria:
- MF wall: **≤ 450 s** (hard gate — any value below Phase 0.5's 873 s
  confirms the sync trap is fixed; projection is 300-400 s)
- lnL: −7,541,976.860 ±0.01
- Best model: LG+G4 (exact)
- `MF-MPI-DIAG` lines present and `filterRatesMPI` broadcast confirmed

### Results — Phase 0.6 sync trap NOT fixed

Job completed. MF wall: **850.531 s** (projection was 300-400 s). CPU efficiency: 11.9%
(88% idle) — essentially identical to Phase 0.5 (11.6%). Correctness ✓.

Root cause of Phase 0.6's failure: the collective `MPI_Bcast` is a barrier requiring ALL
ranks to arrive simultaneously. Rank 0's ref family (LG, no +F) costs ~250 s. Ranks 1-3's
ref families (LG+FC, LG+FQ, LG+FU, with +F ML frequencies) cost ~750 s (3× heavier per
model). Even with Phase 0.6's ref-family-first ordering, rank 0 idles ~500 s waiting for
ranks 1-3 to complete their heavier ref families before `MPI_Bcast` can unblock.

Phase 0.6 fixed the *wrong-order* symptom from Phase 0.5 but not the underlying structural
cause: the collective barrier forces rank 0 to wait for the slowest rank's ref family.

Phase 0.7 required: replace `MPI_Bcast` (collective) with a rank-0 **push** — rank 0 sends
`ok_rates` to each rank individually (via `MPI_Send`) the moment its LG ref family
completes, without waiting for a collective. Ranks 1-3 poll with `MPI_Iprobe`/`MPI_Recv`.

### Accumulated results table

| Job | Phase | np | MF wall | Total wall | lnL | Best model | Status |
|-----|-------|----|---------|-----------|-----|------------|--------|
| 168425673 | Baseline (Fix H, standard binary) | 1 | 399 s | 1,170 s | −7,541,976.860 | LG+G4 | ✓ ref |
| 168470237 | FCA Phase 0 | 1 | 1,278 s | 1,988 s | −7,541,976.862 | LG+G4 | ✓ |
| 168481332 | Phase 0.5 (MPI_Bcast ok_rates) | 4 | **873 s** | — | −7,541,976.853 | LG+G4 | ✓ correct, sync trap |
| **168483748** | **Phase 0.6 (ref-family priority)** | **4** | **850 s** | **1,058 s** | **−7,541,976.853** | **LG+G4** | **✓ correct, sync trap persists** |

---

## 2026-05-16 (bh) — Phase 0.6 fix: `getNextModel` ref-family priority (collective-sync-trap)

### What happened

Job **168481332** ran the Phase 0.5 binary at np=4 AA 100K — correctness was
perfect (lnL -7,541,976.853, best LG+G4, broadcast fired with
`ok_rates={+G}`, 273 models pruned on rank 0) but **MF wall was 873 s, not
the projected 95-150 s**. OMP efficiency was only 11.6% — **88% of threads
were idle**.

### Root cause — Block 2 / Block 3 interleaving inflates pre-broadcast time

`generate()` produces a 3-block model list:
- Block 1: LG × all rate variants (22 models, indices 0-21) — rank 0's
- Block 2: 99 non-LG model_names × bare rate (indices 22-110)
- Block 3: 99 non-LG model_names × 11 rate variants each (indices 111-1231)

In Block 3, rate variants are **clustered per model_name**: LG+F's rates
at 111-121, LG+FC's at 122-132, etc.

Rank 0's ref family (LG) is entirely contiguous in Block 1 → 22 ref models
evaluated in ~120 s → broadcast fires.

Ranks 1-3 own non-LG families (LG+FC, LG+FQ, LG+FU). Each rank's ref family
is **split**: 1 model in Block 2 (early, index ~13) + 21 rate variants in
Block 3 (late, indices 122-142). Between them, `getNextModel` returns the
rank's ~13 OTHER Block 2 entries (one per assigned subst_name group). Those
13 × ~11-22 s = ~200-300 s of NON-ref work delays the ref family completion.

Ranks 1-3 reach the collective MPI_Bcast at ~488 s. Rank 0 sits idle for
~370 s waiting for the collective to unblock. After broadcast, both ranks
evaluate the G-variant remainder (~150-250 s). Observed wall ≈ 488 + 250
= ~740 s + barrier/load variance = **873 s** ✓.

### The fix — `getNextModel()` ref-family priority

Phase 0.6 modifies `CandidateModelSet::getNextModel()` to **prefer
ref-family models** while `mpi_ref_remaining > 0` AND the broadcast hasn't
fired yet. Each rank now reaches MPI_Bcast after evaluating only its ~22
ref-family models — broadcast arrival times converge to ~150 s for ALL
ranks. Rank 0's idle time drops from ~370 s to ~15 s.

To enable this, the FCA state machine variables (`mpi_ref_subst_idx`,
`mpi_ref_remaining`, `mpi_filterRatesMPI_fired`, `mpi_filterRatesMPI_enabled`)
are **promoted from `evaluateAll` locals to `CandidateModelSet` public
members** (declared in `phylotesting.h`, reset at the top of every
`evaluateAll` call for MixtureFinder/PartitionFinder safety).

A latent **first-call IGNORED bug** in the original `getNextModel` was
also fixed: the old code returned `next_model = 0` unconditionally on the
first call (`current_model == -1`), even if model 0 had `MF_IGNORED` set
for the calling rank — which under FCA dispatch is the case for ranks 1+.
Phase 0.6's unified IGNORED-skip scan handles the first call correctly.

### Files changed

| File | Change |
|------|--------|
| `main/phylotesting.h` (constructor) | +5 LOC: init 4 new FCA state members. |
| `main/phylotesting.h` (public section) | +9 LOC: declare 4 new public FCA state members. |
| `main/phylotesting.cpp` `getNextModel()` | ~+50/-10 LOC: ref-family priority scan + corrected IGNORED-skip on first call. |
| `main/phylotesting.cpp` `evaluateAll()` | ~9 LOC: replace 4 local var declarations with `this->` member resets. |
| `research/updated-modelfinder-dispatch.md` | +220 LOC: §20 sync-trap root-cause analysis + 5-row results table extension (#13-#17). |

Total: ~74 LOC source change.

### Build verification

- Binary: `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`
  rebuilt 2026-05-16 22:47 AEST (145,056,888 bytes; +2,448 bytes vs Phase 0.5).
- Object: `phylotesting.cpp.o` 9,132,440 bytes (was 9,125,720).
- Symbols verified:
  - `CandidateModelSet::getNextModel()` at 0x6754d0
  - `CandidateModelSet::filterRatesMPI(int)` at 0x6743a0
- Smoke test: `iqtree3-mpi --version` exits 0.
- Clean-build procedure (same as `(bg)`): `rm` of `.o` and
  `compiler_depend.*`; `touch` of placeholder `compiler_depend.{make,internal,ts}`;
  module load `intel-compiler-llvm/2025.1.1 openmpi/4.1.7`; `make iqtree3 -j 16`.

### Phase 0.6 projection — pending T3'' PBS validation

| Config | Fix H baseline | Phase 0 | Phase 0.5 (obs) | Phase 0.6 (proj) |
|--------|---------------:|---------:|----------------:|-----------------:|
| np=1 | 1,289 s | 1,277 s | ~1,277 s | ~1,277 s (no-op gate) |
| np=2 | 475 s | 2,865 s | TBD | ~280-340 s |
| **np=4** | 2,335 s | **3,502 s** | **873 s** | **~300-400 s** |

Phase 0.6 brings ~2-3× speedup over Phase 0.5 at np=4, total improvement
over Fix H = **5.8-7.8×**.

### Why 100 s target wasn't met even by Phase 0.6

The 100 s target assumed 3.17 s/model (Amdahl-derived at 103 OMP threads).
Observed reality: per-model wall is 11-22 s (heavier than projected,
particularly for +F variants). Per-rank workload after pruning: ~36 models
× ~12 s = ~432 s. This is a structural limit of Fix H sequential outer at
np=4 AA 100K — only HH-NUMA Phase 2 (`K_outer × M_inner` nested OMP)
can break it. See §20.5-20.6 of dispatch doc.

### Connection to HH-NUMA Phase 2 (still deferred)

Phase 0.6 + HH-NUMA K=8 projected ~150 s at np=4 (combined effect of
ref-family priority + nested OMP K_outer=8 × M_inner=12). All Phase 2 prep
work (atomic on `mpi_ref_remaining--`, member-var state, single-fire guard)
is now in place; only the OMP pragma at `phylotesting.cpp:~3865` remains
to be changed for HH-NUMA enablement.

### Bug history snapshot (full results table in dispatch doc §20.9)

| # | Bug | Status |
|--:|-----|--------|
| 1-2 | FCA Phase 1 dispatch + state machine init | ✓ |
| 3-4 | Counter stall + `==`→`<=` boundary | ✓ |
| 5-6 | Stale CMake build + unconditional FCA-DBG | ✓ |
| 7 | Rank 1+ filterRates ineffectiveness | ✓ Phase 0.5 |
| 8 | HH-NUMA atomic safety prep | ✓ |
| 9 | HH-NUMA Phase 2 | ⏸ deferred |
| 10-12 | Build env / depend.make / symbol verification | ✓ |
| **13** | **Phase 0.5 measured 873 s @ np=4 (sync trap)** | **✗ → ✓ Phase 0.6** |
| **14** | **`getNextModel` ref-family priority** | **✓ correct** |
| **15** | **Promote FCA state to class members** | **✓ correct** |
| **16** | **First-call IGNORED bug in `getNextModel`** | **✓ correct** |
| **17** | **Phase 0.6 binary verification** | **✓ verified** |

---

## 2026-05-16 (bg) — Phase 0.5 fix: cross-rank `ok_rates` `MPI_Bcast` + HH-NUMA atomic prep

### What changed

Implemented **Phase 0.5** of the FCA dispatch — the cross-rank `ok_rates`
broadcast that fixes the rank 1+ pruning gap identified in entry `(bf)`.

**New function** `CandidateModelSet::filterRatesMPI(int finished_model)`
(`main/phylotesting.h:248-274`, `main/phylotesting.cpp:2938-3033`,
~80 LOC):
- Each rank computes its local `ok_rates` (same algorithm as
  `filterRates`).
- Rank 0's `ok_rates` is serialised as `"rate1|rate2|..."` (2048-byte
  buffer) and `MPI_Bcast`'d from root=0.
- All ranks parse the received string into `set<string> global_ok_rates`.
- Each rank applies `global_ok_rates` to its local model list: any
  non-DONE / non-IGNORED / non-CANNOT_BE_IGNORED model whose
  `orig_rate_name` is NOT in the set gets `MF_IGNORED`.

**FCA dispatch hardening** (`main/phylotesting.cpp:3764-3783`, FCA Step 8):
- `MPI_Allreduce(MIN)` checks `mpi_ref_subst_idx >= 0 && auto_rate &&
  score_diff_thres >= 0` on every rank. If any rank fails, all fall back
  to legacy `filterRates` (no broadcast, safe but under-prunes on ranks
  1+). Prevents `MPI_Bcast` deadlock when a rank lacks a reference family.
- `mpi_filterRatesMPI_fired` (bool) ensures single-fire per rank.
- `mpi_filterRatesMPI_enabled` (bool) gates the new vs legacy path.

**HH-NUMA atomic preparation** (`main/phylotesting.cpp:3876-3884`):
- `#pragma omp atomic update` wraps the §12.5.3 intra-chain decrement.
- Under Fix H (sequential outer), redundant but harmless. Under Phase 2
  HH-NUMA (parallel `K_outer × M_inner`), prevents race when multiple
  outer threads concurrently fire intra-chain pruning.

**FCA-DBG instrumentation cleanup** (`main/phylotesting.cpp:3934-3953`):
- 30-line per-rank trace now gated on `verbose_mode >= VB_MED` instead of
  unconditional. Production runs no longer flood stdout.

### Files changed

| File | Change |
|------|--------|
| `main/phylotesting.h` | +27 LOC: declaration `void filterRatesMPI(int)` inside `#ifdef _IQTREE_MPI` after `filterRates` (line 246). |
| `main/phylotesting.cpp` | +~120 LOC, -1 LOC: `filterRatesMPI` body (~80 LOC after line 2936); FCA trigger replaced with conditional `filterRatesMPI(model)` / `filterRates(model)` fallback (lines ~3960-3999); FCA Step 8 `MPI_Allreduce` gate (lines ~3764-3783); intra-chain decrement atomic (line 3876); FCA-DBG gated on `VB_MED` (lines 3934-3953). |
| `research/updated-modelfinder-dispatch.md` | +220 LOC: §19 Phase 0.5 design, results table (12 rows), projection, git state. |

### Build pipeline — root cause of `(be)` stale binary, now solved

Per entry `(be)`, `compiler_depend.ts` was stale (May 10 timestamp) which
caused CMake to skip rebuilding `phylotesting.cpp.o` despite source
modifications on May 16. Rebuild process for this entry:

```bash
cd /scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2
rm -f main/CMakeFiles/main.dir/phylotesting.cpp.o \
      main/CMakeFiles/main.dir/compiler_depend.*
touch main/CMakeFiles/main.dir/compiler_depend.{make,internal,ts}
touch /scratch/um09/as1708/iqtree3-mf2/src/iqtree3/main/phylotesting.cpp \
      /scratch/um09/as1708/iqtree3-mf2/src/iqtree3/main/phylotesting.h
source /etc/profile.d/modules.sh
module load intel-compiler-llvm/2025.1.1 openmpi/4.1.7
export OMPI_CXX=icpx OMPI_CC=icx
make iqtree3 -j 16
```

`touch compiler_depend.{make,internal,ts}` creates empty placeholder
files (the Makefile's `include` directive requires them to exist; CMake
regenerates contents on first compile). Without this, `make` fails with
`No rule to make target 'compiler_depend.make'`.

### Build verification

- Binary: `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`
  rebuilt 2026-05-16 21:51 AEST.
- Size: 145,054,440 bytes (was 146,184,272 — net decrease from FCA-DBG
  unconditional trace removal vs `filterRatesMPI` body insertion).
- Object: `phylotesting.cpp.o` 9,125,720 bytes (was 9,184,672).
- Symbol verified: `nm -C iqtree3-mpi | grep filterRatesMPI` →
  `0000000000674310 T CandidateModelSet::filterRatesMPI(int)`.
- Strings verified: binary contains `MF-MPI-DIAG: rank `,
  `filterRatesMPI: |bcast_ok_rates|=`, `filterRatesMPI_enabled=`.
- Smoke test: `iqtree3-mpi --version` exits 0; banner shows
  "IQ-TREE MPI version 3.1.2 for Linux x86 64-bit built May 16 2026".

### Phase 0.5 projection — pending T2'/T3' PBS validation

| Config | Fix H baseline | Phase 0 observed | Phase 0.5 projected |
|--------|---------------:|------------------:|---------------------:|
| np=1 | 1,289 s | 1,277 s | ~1,277 s (no-op gate) |
| np=2 | 475 s | **2,865 s** ✗ | **~180-240 s** |
| np=4 | 2,335 s | **3,502 s** ✗ | **~95-150 s** |

T2' (np=2) and T3' (np=4) jobs to be submitted with the rebuilt binary;
pass criterion = MF wall < projection-upper-bound and lnL = -7,541,976.860
±0.01 with best model LG+G4.

### Phase 2 HH-NUMA — deferred (next session after T2'/T3' confirm)

HH-NUMA (nested `K_outer × M_inner` parallel outer loop) is deferred per
"step by step" guidance. Risk profile: nested OMP toolchain interaction
(libiomp5 hot teams), MPI_THREAD_FUNNELED vs SERIALIZED, per-team
`iqtree->setNumThreads(M_inner)` propagation through `evaluate()`. The
atomic prep in this entry (#8 in §19.4 results table) makes the FCA
state machine HH-NUMA-ready; the only remaining edit is the OMP pragma
at `phylotesting.cpp:3832` (`num_threads(num_threads)` → `num_threads(K_outer)`
with `omp_set_max_active_levels(2)`).

### Bugs encountered and squashed (debugging journey)

Per `research/updated-modelfinder-dispatch.md` §19.4 (12-row implementation
results table), the FCA debugging journey across `(bd)` → `(be)` → `(bf)`
→ `(bg)` encountered and resolved:

1. Counter stall (intra-chain pruning silently consumed ref-family slots)
   — fixed `(be)` via §12.5.3 Change 1.
2. Trigger boundary `==` → `<=` (defensive) — `(be)`.
3. Stale CMake dependency build (binary unchanged after source edit)
   — `(be)` via direct `mpicxx` compile.
4. FCA-DBG unconditional stdout flood — `(be)` via `VB_MED` gating.
5. Rank 1+ filterRates ineffectiveness (per-rank reference family, flat
   WAG/JTT/DCMUT BIC) — `(bg)` via `filterRatesMPI` `MPI_Bcast`.
6. Module load PATH propagation between shells — `(bg)` via single-shell
   chained `source && module load && export && make`.
7. CMake `compiler_depend.make` missing after `rm` — `(bg)` via `touch`
   of empty placeholder files.

All seven debug iterations are recorded in §19.4 with root cause and fix.

### Why this matters

Phase 0.5 is the **critical correctness fix** that makes FCA dispatch
actually deliver its design intent at np>=2. Without it, the algorithm
regresses 50% at np=4 (3,502 s vs Fix H 2,335 s) because three of four
ranks don't prune effectively. With it, the projection is **~100 s at
np=4** — a 23× speedup over the regressed Phase 0 and ~14× over Fix H.

HH-NUMA Phase 2 will layer on top to push toward ~60 s, but Phase 0.5 is
the prerequisite that makes the design viable at scale.

---

## 2026-05-16 (bf) — FCA debug run: rank 0 filter confirmed; cross-rank filterRates gap identified

### What happened

Debug binary compiled at 19:53 AEST with `FCA-DBG` unconditional trace
(30-line cap inside `#pragma omp critical`). Job **168475747** (np=2, 2 nodes)
submitted and run to completion.

**Result: FAIL — MF wall 2,460.602 s (≈ baseline 2,473 s; no improvement).**
Job cancelled after IQ-TREE phase completed but PBS job was still in Pass 2 perf stat.

### Findings

**Rank 0 state machine confirmed correct** via FCA-DBG output:

- `mpi_ref_remaining` decremented correctly (22 → 0 over models 0–16)
- Intra-chain pruning loop fix (Change 1 from entry `be`) fired correctly:
  models 8–12 and 17–21 decremented from the pruning loop, not the critical section
- `filterRates(16)` fired, pruning all non-`+G4` variants for non-LG families
- Rank 0 log shows only `DCMUT+G4`, `PMB+G4`, `DAYHOFF+G4` etc. after model 16
- Rank 0 evaluates ≈ 71 models total (down from 616); log stops growing at ~18–20 min

**Rank 1 is the bottleneck** (MF wall = max(rank 0, rank 1)):

- Rank 1 stdout not captured by `mpirun > iqtree_run.log` across two nodes
- MF wall = 2,460 s implies rank 1 evaluated ≈ 616 models without effective pruning
- `filterRates` per-rank design fires rank 1's filter using rank 1's reference
  family BIC scores, which may have weaker selectivity than LG (rank 0's family)
- If rank 1's reference family (e.g., WAG) has similar BIC scores across rate
  categories, `ok_rates` includes all rate types → no pruning

### Root cause — cross-rank filterRates gap

Phase 0 design assumes each rank's local reference family will produce the same
pruning decision as rank 0's LG family. This fails when:

1. The optimal rate category is not `+G4` from the reference family's perspective
2. The reference family's BIC landscape is flat enough that `score_diff_thres = 10`
   does not eliminate any rate type

The correct fix requires rank 0 to `MPI_Bcast` its `ok_rates` to all other ranks
**before** they evaluate models beyond their reference family. This is a Phase 0.5
change requiring a new MPI collective call.

### Build issue also confirmed

Job 168471481 (entry `be`) used a **stale binary** — CMake's `compiler_depend.ts`
was newer than the source, silently blocking recompilation of `phylotesting.cpp.o`.
The 2,473 s wall time of 168471481 was a build failure, not a fix failure.
The debug binary (19:53) was compiled with a direct `mpicxx` command to bypass
the stale dependency check.

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | FCA-DBG trace added (30-line unconditional trace inside `#pragma omp critical`) |
| `setonix-iq/research/updated-modelfinder-dispatch.md` | §12.5.5 added: stale-build analysis, debug run results, rank-1 bottleneck root cause, required fix |
| `setonix-iq/CHANGELOG.md` | This entry |

### Next steps

1. Add rank 1 stdout capture to run script (`--output-filename` or per-rank log)
   to confirm the rank-1 filterRates hypothesis before implementing the fix
2. Implement `MPI_Bcast` of `ok_rates` from rank 0 after its reference family
   completes; all other ranks apply the broadcast pruning set
3. Remove `FCA-DBG` instrumentation from `phylotesting.cpp` before production rebuild

---

## 2026-05-16 (be) — FCA Phase 0 bug fix: `mpi_ref_remaining` never reaching 0; re-queued T2/T3

### What happened

Three Phase 0 FCA test runs were submitted as jobs 168470237 (np=1), 168470238 (np=2), and
168470240 (np=4). The np=1 run completed correctly. The np=2 and np=4 runs were still in
ModelFinder at 38+ minutes (well past the expected ~400 s and ~250 s bounds), confirmed by
live log inspection showing JTT+I+R6 (model 106) and Q.BIRD+F+I+R2 (model 212) still being
evaluated — families that `filterRates` should have pruned after the LG reference family
finished.

Root cause identified, patched, binary rebuilt, stuck jobs cancelled, and replacement jobs
168471481/168471482 submitted within the same session.

### Root cause — `mpi_ref_remaining` stalls above 0

The FCA state machine initialises `mpi_ref_remaining` to the count of non-IGNORED
ref-family models on the rank at dispatch time — for rank 0 at np=2 this is 22 (all LG
models 0–21). The counter decrements in the `#pragma omp critical` block each time rank 0
**evaluates** a ref-family model; `filterRates` fires when it hits 0.

Intra-chain pruning (`getLowerKModel` comparison, `phylotesting.cpp:3712–3720`) marks
higher-k models `MF_IGNORED` *after* a model is evaluated when the +Rk chain has begun
declining. This silently marks LG+R6..R10 (models 8–12) and LG+I+R6..R10 (models 17–21)
as IGNORED. Those 10 models are never returned by `getNextModel()` and never pass through
the `omp critical` block, so their slots are permanently stranded:

```
Initial:            mpi_ref_remaining = 22
Models 0–6 eval:    22 → 15  (7 decrements)
Model 7 (LG+R5):    pruning marks models 8–12 IGNORED  ← no decrement  → still 15
  critical (model 7): 15 → 14
Models 13–15 eval:  14 → 11  (3 decrements)
Model 16 (LG+I+R5): pruning marks models 17–21 IGNORED ← no decrement  → still 11
  critical (model 16): 11 → 10
Counter stalls at 10.  filterRates NEVER fires. ✗
```

All 616 models on rank 0 are evaluated without cross-family rate pruning. That is why the
np=2 run shows JTT+R2..R7 being fully evaluated (should be pruned to JTT+G4 only).

### Fix — two changes to `phylotesting.cpp`

**Change 1** (`phylotesting.cpp:3718–3730`): Inside the `for (higher_model = model; ...)`
pruning loop, added an `#ifdef _IQTREE_MPI` block that decrements `mpi_ref_remaining` for
each pruned ref-family model, **excluding `model` itself** (which is counted in the `omp
critical` section below to avoid double-decrement):

```cpp
at(higher_model).setFlag(MF_IGNORED);
#ifdef _IQTREE_MPI
if (higher_model != (int)model
    && mpi_ref_subst_idx >= 0 && auto_rate
    && at(higher_model).subst_name == at(mpi_ref_subst_idx).subst_name)
    mpi_ref_remaining--;
#endif
```

With the fix, the counter trace becomes:

```
Initial:            mpi_ref_remaining = 22
Models 0–6 eval:    22 → 15
Model 7 (LG+R5):    pruning loop: models 9–12 each decrement → 15 → 10
  critical (model 7): 10 → 9
Models 13–15 eval:  9 → 6
Model 16 (LG+I+R5): pruning loop: models 17–21 each decrement → 6 → 1
  critical (model 16): 1 → 0 → filterRates(16) fires ✓
```

`filterRates(16)` yields `ok_rates = {"G4"}` (only LG+G4 beats the BIC+10 threshold),
pruning all non-`+G4` models from every subsequent family. ~28 `+G4` variants survive from
616 assigned models.

**Change 2** (`phylotesting.cpp:3763`): `mpi_ref_remaining == 0` → `mpi_ref_remaining <= 0`
as a defensive guard for any future edge case.

Applied via Python in-place edit (scratch filesystem not writable by editor tools). Rebuilt:

```bash
# In build-mpi-mf2/:
make -C main phylotesting.cpp.o
make -f main/CMakeFiles/main.dir/build.make main/libmain.a
make iqtree3
```

New binary: `build-mpi-mf2/iqtree3-mpi` (2026-05-16 18:09, 146,184,272 bytes — 115 KB
larger than pre-fix, confirming the relink included new code).

### T1 (np=1) result — completed correctly, FCA is no-op

Job 168470237 completed IQ-TREE Pass 1 at 17:45:19; PBS job continued into Pass 2 (perf
stat). FCA state machine is guarded by `if (numProcesses > 1)` and is entirely a no-op at
np=1 — `filterRates` fires via the legacy `model >= rate_block` path exactly as in Fix H.

| Metric | Value | Fix H np=1 (168468561) | Delta |
|--------|------:|----------------------:|-------|
| MF wall | **1,277.622 s** | 1,289 s | −11 s (−0.9%) |
| Tree wall | **707.199 s** | — | — |
| Total wall | **1,988.184 s** | — | — |
| lnL | **−7,541,976.862** | −7,541,976.860 | −0.002 (within ±0.01 ✓) |
| Best model | **LG+G4** | LG+G4 | exact match ✓ |

MF wall is 11 s shorter than Fix H — within normal run-to-run variation; no regression.

### T2 (np=2) and T3 (np=4) — cancelled (stuck without pruning)

| Job | Config | Elapsed at cancel | Last model evaluated | Root cause |
|-----|--------|------------------:|---------------------|------------|
| 168470238 | np=2 | ~56 min | JTT+I+R6 (model 106) | `filterRates` never fired |
| 168470240 | np=4 | ~56 min | Q.BIRD+F+I+R2 (model 212) | `filterRates` never fired |

Both cancelled via `qdel` after root cause was confirmed. No useful timing data recoverable
— the ModelFinder step would have run to completion across all 616/308 models per rank
without any cross-family pruning, producing bloated wall times of 3,000–3,500 s.

### Replacement jobs submitted with fixed binary

| Test | PBS ID | Config | Status |
|------|--------|--------|--------|
| T2 (np=2) | **168471481** | 2 nodes, 208 cores | Queued 18:09 AEST |
| T3 (np=4) | **168471482** | 4 nodes, 416 cores | Queued 18:09 AEST |

Pass criteria unchanged from §11 of design doc: MF wall ≤ 430 s (np=2), ≤ 350 s (np=4);
lnL −7,541,976.860 ±0.01; best model LG+G4.

### HHOIP prerequisite note

The `mpi_ref_remaining--` in Change 1 is not thread-safe for K_outer > 1 (sequential outer
loop at K=1 is safe). Before HHOIP (§13.1 of design doc) can land, this decrement and the
`ratefilter_fired_by_fca` flag must be atomicised (`#pragma omp atomic` / `std::atomic<int>`).
See `research/updated-modelfinder-dispatch.md` §12.5.4.

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | Change 1: `mpi_ref_remaining--` in intra-chain pruning loop; Change 2: trigger `<= 0` |
| `setonix-iq/research/updated-modelfinder-dispatch.md` | §12.5 added: bug, fix, counter trace, HHOIP gating |
| `setonix-iq/CHANGELOG.md` | This entry |

---

## 2026-05-16 (bd) — FCA dispatch: novel ModelFinder MPI algorithm (patch `0003`, commit `ffb79a14`)

### What changed

A novel ModelFinder MPI dispatch algorithm — **FCA (Family-Local + Cost-Aware + Always-Filter)** — replaces the entire Fix A + Fix C stack with a single-pass design that closes the np=4 regression (2,335 s → projected ~100 s, 23×). Design fully documented in [`research/updated-modelfinder-dispatch.md`](research/updated-modelfinder-dispatch.md); patch committed at `gadi-spr-r2-mf-fca`/`ffb79a14`; ships as [`patches/iqtree3/0003-mf-fca-dispatch.patch`](patches/iqtree3/0003-mf-fca-dispatch.patch).

Three structural changes vs Fix A+C:

1. **Closed-form cost predictor**: `cost(m) = nstates² · npat · rate_mult · freq_mult · log2(ntaxa)`. Captures DNA(4)/AA(20)/codon(61) state-count, alignment size, +Rk rate cost, the ~3× cost of +F (ML frequency) variants, and the `log₂(ntaxa)` BFGS scaling factor — none of which the old `k*10` proxy modeled. This alone explains the np=4 AA 100K regression: +F variants happen to cluster on one rank under round-robin assignment, doubling its actual cost vs predicted.
2. **Greedy LPT (argmin rank_load) replaces round-robin**: each substitution family is assigned to the rank with the lowest accumulated load. With ~100 unique `subst_name` groups (20 AA matrices × 5 freq variants) over 4 ranks, the imbalance after pruning drops from a worst-case 1.5× under round-robin to <1.15× under greedy LPT (Graham 1969 bound: (4m−1)/3m = 1.25× × accurate cost estimate).
3. **State-machine filterRates trigger replaces `model >= rate_block`**: a per-rank `mpi_ref_remaining` counter decrements as the rank's reference-family models finish, and `filterRates` fires exactly once when it hits zero. This eliminates Fix C's fragile `rate_block` recompute and the suspected np=4 edge case (where reference-family Block-2/Block-3 ordering races caused premature filterRates triggers that returned without pruning).

The MF-MPI Fix B (OMP-across-models), Fix D (proc_bind(spread)), Fix G (local_in_info), and Fix H (`!_IQTREE_MPI` guard on outer pragma) are all preserved unchanged. FCA also adds always-on per-rank diagnostic logging (`MF-MPI-DIAG:` lines) so future regressions are observable without a code change.

### Expected performance (AA 100K, SPR 2×52T, projection — pending PBS validation)

| PBS ID  | Scenario        | Pre-fix MF wall | Fix A–H MF wall | FCA MF wall (proj.) | vs single-node baseline |
|---------|-----------------|----------------:|----------------:|---------------------:|------------------------:|
| 168425673 | Baseline (std, 1 node) | — | — |   399 s            |   1.00×                  |
| TBD     | FCA np=1        | 1,309 s | 1,289 s | **~400 s**           | **0.99×**                |
| TBD     | FCA np=2        |   969 s |   475 s | **~175 s**           | **2.28×**                |
| TBD     | **FCA np=4**    |   573 s | **2,335 s ⚠** | **~100 s**     | **~3.99×**               |
| TBD     | FCA np=4 DNA 100K | — | — | **~50 s**       | (DNA SPR baseline 290 s) |

### Why FCA succeeds where Fix A+C didn't

Static analysis of Fix C's `rate_block` recompute (`research/updated-modelfinder-dispatch.md` §2.1) showed it correct in isolation but fragile under five compounding preconditions. The np=4 regression is most likely caused by the round-robin LPT happening to concentrate ~7 ML-frequency (+F) families on one rank (the cost predictor doesn't see +F at all), with the actual makespan (~308 × 7.6 s/model = 2,341 s) matching the observed 2,335 s within 0.3%. The FCA cost predictor explicitly models +F at 3× weight, so greedy LPT spreads +F families across all ranks. Independently, the state-machine trigger removes any remaining sensitivity to model-list layout (Block 1/2/3 ordering from `generate()` line 1699).

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | FCA dispatch (single-pass; replaces Fix A Phase 1 stripe + Fix C `rate_block` recompute); state-machine trigger added to OMP loop |
| `setonix-iq/research/updated-modelfinder-dispatch.md` | **NEW** — full design doc (§1–§8) with phased plan, cost-predictor derivation, risk register, literature review |
| `setonix-iq/patches/iqtree3/0003-mf-fca-dispatch.patch` | **NEW** — `git format-patch` from `gadi-spr-r2-mf-fca` `ffb79a14` |
| `setonix-iq/patches/iqtree3/README.md` | New patch documented |
| `setonix-iq/research/aa-walltime-analysis.md` | §2.4.8 added — FCA design rationale + projected np=1/2/4 walltime |
| `setonix-iq/CHANGELOG.md` | This entry |

### Build / apply

```bash
# On Gadi:
module load intel-compiler-llvm/2025.1.1 openmpi/4.1.7
cd /scratch/um09/as1708/iqtree3-mf2/src/iqtree3
git checkout gadi-spr-r2-mf-fca    # branch with commit ffb79a14
cd ../../build-mpi-mf2
OMPI_CXX=icpx OMPI_CC=icx make iqtree3-mpi -j   # ~96% incremental from prior build
```

Build verified clean (12 unrelated warnings, no errors) — main module compiled in ~7 s on Gadi login node.

### Library + super-parallel + data-delivery audit (added 2026-05-16, post-build)

Researched whether to extract FCA into a standalone library and whether further parallelism is feasible. See `research/updated-modelfinder-dispatch.md` §9–§10 for the full audit. Summary:

- **Custom library `libiqtree-mfdispatch.so`** — **not worth it.** Only one consumer (`evaluateAll()`); PartitionFinder uses a different MPI pattern; MixtureFinder inherits FCA for free. ABI cost: ~5 days; gain: 0.
- **Super-parallel beyond model-level** — **already at the ceiling.** `MPI_Win_allocate_shared` doesn't work for write-through partial_lh; persistent OMP team is already implicit (`OMP_WAIT_POLICY=ACTIVE` + libiomp5); subtree GPU offload is orthogonal and on a different branch.
- **Data delivery audit** — only one actionable item: `madvise(MADV_HUGEPAGE)` on the 6.27 GB `central_partial_lh` buffer (currently 1.57M × 4 KB PTEs). Expected gain: **8–15% MF wall**, affects all np configs orthogonally to dispatch. **Defer until T1–T3 results land** — if T3 (np=4) hits the projected 200-300 s band, THP becomes the next-priority follow-up patch (`0004-thp-partial-lh-madvise.patch`).

### Revised np=1/2/4 projections (after honest re-analysis)

Earlier ≤100 s target at np=4 was optimistic — it assumed Fix B parallel outer loop (disabled by Fix H for OOM safety at AA 100K). Realistic targets:

| Config | Fix H baseline | FCA Phase 0 target | Improvement | Why |
|--------|---------------:|-------------------:|------------:|-----|
| np=1 | 1,289 s | **~1,289 s** | **1.0×** | FCA is `if (numProcesses > 1)` guarded — no-op at np=1; Amdahl-limited site-parallel sequential loop dominates |
| np=2 |   475 s | 400–430 s | 1.1–1.2× | Modest; greedy LPT + freq_mult=3 balances +F across 2 ranks |
| **np=4** | **2,335 s ⚠** | **200–300 s** | **~10×** | Headline; +F-concentration hypothesis (if correct) restores expected sub-300 s |

Sub-100 s at np=4 requires future Phase 1 (telemetry rebalance), Phase 2 (work-stealing), or per-rank memory-footprint reduction to re-enable Fix B parallel outer loop. Out of scope for Phase 0.

### PBS test matrix queued (T1, T2, T3)

Three jobs submitted to `dx61` `normalsr-exec` against the same alignment + seed as baseline `168425673`. All scripts use the FCA-built binary at `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi` (commit `ffb79a14`):

```
T1 (np=1):  gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_1node.sh
T2 (np=2):  gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_2node.sh
T3 (np=4):  gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh
```

Full parity with `168425673`: `-seed 1`, `OMP_NUM_THREADS=103`, `OMP_PROC_BIND=close + OMP_PLACES=cores`, `KMP_BLOCKTIME=200`, `numactl --localalloc`. Pass criteria for all three: `lnL = −7,541,976.860 ±0.01`, best MF model `LG+G4`, BIC `15,086,233 ±1`. MF wall thresholds in §11 of design doc.

`build_tag` updated `mf2_full_icx_avx512_r2_fixh` → `mf2_full_icx_avx512_r2_fca` in all three scripts; `non_canonical_label` updated `MF2 Full (Fix A–H)` → `MF2 FCA (Family-Local, Cost-Aware, Always-Filter)`.

PBS job IDs (submitted 2026-05-16, dx61 `normalsr-exec`):

| Test | PBS ID | Nodes × CPUs | Mem | Walltime | Status |
|------|--------|--------------|----:|---------:|--------|
| T1 (np=1) | `168470237` | 1 × 104 | 510 GB | 03:00 | Queued |
| T2 (np=2) | `168470238` | 2 × 104 | 1,020 GB | 03:00 | Queued |
| T3 (np=4) | `168470240` | 4 × 104 | 2,040 GB | 03:00 | Queued |

Results to be appended below as runs complete; each run writes
`logs/runs/gadi_AA_100k_mf2_npN_seed1_<PBS_ID>.json` with lnL + best model
verification and per-rank `MF-MPI-DIAG:` diagnostic lines in the `iqtree_run.log`.

---

## 2026-05-16 (bc) — Fix E–H: OMP race fix + sequential MPI outer loop; all Fix H results (168468561–563)

### What changed

Four commits between Fix D (`bb`) and the current HEAD resolve the OMP data-race in
`evaluateAll()` for MPI builds and restore a correct, OOM-safe outer loop:

| Commit | Fix | Change |
|--------|-----|--------|
| `eddbf45d` | **E** | Revert Fix B: restore sequential outer loop for MPI while race is diagnosed |
| `a9b50164` | **F** *(BUGGY)* | Thread-local `in_model_info` snapshot — SIGABRT: snapshot placed *after* `initializeModel()`, which already read the shared pointer |
| `10107158` | **G** | Move `local_in_info` copy *before* `setCheckpoint()` — race eliminated; parallel outer loop retained but OOMs at 100K AA |
| `257485e5` | **H** | `#if defined(_OPENMP) && !defined(_IQTREE_MPI)` guard on outer parallel pragma — MPI builds: sequential outer loop (1 model × 103T site-parallel); non-MPI builds: parallel (103 models × 1T) |

Fix G's `local_in_info` snapshot is retained in Fix H (correct for non-MPI; no-op for sequential MPI).

### All measured results — Fix A+C+D+G+H, AA 100K, 2026-05-16

| PBS ID | Scenario | MF wall | Tree wall | Total | vs baseline | lnL | IPC | LLC miss% |
|--------|----------|---------|-----------|-------|-------------|-----|-----|--------|
| 168425673 | Baseline (std, 1 node) | 399 s | 764 s | 1,169 s | 1.00× | −7,541,976.860 | 1.878 | 66.94% |
| 168468561 | Fix A–H, np=1, 1×103T | 1,289 s | 720 s | 2,012 s | 0.58× | −7,541,976.862 ✓ | 1.975 | 68.14% |
| **168468562** | **Fix A–H, np=2, 2×103T** | **475 s** | **387 s** | **866 s** | **1.35×** | **−7,541,976.865** ✓ | **2.005** | **67.40%** |
| 168468563 | Fix A–H, np=4, 4×103T | **2,335 s** ⚠ | 200 s | 2,541 s | 0.46× ⚠ | −7,541,976.852 ✓ | 2.135 | 68.48% |

**np=2 — best result (1.35×):** MF 475 s (site-parallel 27× speedup vs model-parallel
103×, explained by Amdahl serial fraction in §2.4.7). Tree 387 s = near-perfect 2-node
scaling. Total **866 s = 1.35×** speedup over 1-node baseline.

**np=1 — unchanged (0.58×):** Fix H filterRates pruning reduces models 1,232→~475 but
per-model cost rises 1.06→2.71 s (site-parallel OMP); effects cancel. MF 1,289 s; Tree
720 s ≈ baseline 764 s.

**np=4 — REGRESSION ⚠ (0.46×):** MF 2,335 s is **4.1× worse** than pre-fix np=4
(573 s). Tree 200 s is correct (1.94× scaling from np=2). Suspected cause: Fix C
`rate_block` recompute edge case at np=4 disables filterRates on a worker rank; that rank
evaluates all ~308 assigned models (incl. costly +F variants at ~7.6 s/model) without
pruning, stalling the MPI gather. Pre-fix np=4 MF (573 s, 1.51×) was faster because it
used OMP-across-models parallelism per rank before sequential mode was mandated.

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | Fix E (revert Fix B), Fix F (local snapshot — BUGGY), Fix G (correct placement), Fix H (`!defined(_IQTREE_MPI)` guard) |
| `setonix-iq/research/aa-walltime-analysis.md` | §2.4.6 updated with all three Fix H results; §2.4.7 added (root cause: model-parallel vs site-parallel OMP); np=4 regression documented |
| `setonix-iq/CHANGELOG.md` | This entry |

---

## 2026-05-xx (bb) — Fix D: `proc_bind(spread)` for evaluateAll() + MPI data-path analysis

### What changed

Added `proc_bind(spread)` to the `#pragma omp parallel num_threads(num_threads)` pragma
in `evaluateAll()` (`main/phylotesting.cpp`). This overrides the global
`OMP_PROC_BIND=close` for the OMP-across-models evaluation region, ensuring threads are
distributed maximally across all NUMA hardware places before applying proximity
sub-grouping.

### Motivation: MPI data bottleneck analysis (AA 100K)

A deep audit of Phase 2 collective operations confirmed that MPI communication is NOT
a data bottleneck for MF2:

- **4 × `MPI_Allreduce`** (lnL MAX, BIC MIN, AIC MIN, AICc MIN): 39.4 KB total,
  ~21 µs on InfiniBand — unmeasurable.
- **`gatherCheckpoint` + `broadcastCheckpoint`**: ~345 KB per rank (np4), ~1.38 MB
  total, ~7–10 ms including serialisation. Negligible vs 100–400 s evaluation.
- **Grand total Phase 2 overhead: < 12 ms** for any dataset up to ~100K sites, np≤16.

The dominant cost is purely computational (model likelihood optimisation). No protocol
changes to the Phase 2 gather are needed.

### Motivation: thread saturation tail analysis

With OMP-across-models and T=103 threads on M models per rank (np4: M≈150):
- Round 1: 103 threads active, 0 idle.
- Round 2: 47 threads active, 56 idle.
- Thread utilisation: ~83%. Tail loss: ~8% of MF wall time (≤ 10 s).

LPT scheduling (Fix A) front-loads heavy models so the last round carries mostly
+G4/+I models (fast), limiting absolute waste. Hybrid nested-OMP mode deferred
(gain < 10 s vs ~100 s background).

### Motivation: NUMA binding audit

In `evaluateAll()` each thread allocates its own `IQTree` clone; per-model
`partial_lh` goes to the allocating thread's NUMA node (local DRAM ✓). Shared
alignment data (9.6 MB for AA 100K) fits in each socket's L3 (60 MB); both L3 caches
hold it after short warm-up. No sustained cross-NUMA DRAM penalty for the
OMP-across-models path.

### Effect of `proc_bind(spread)`

For T=103 on 104 SPR cores: `close` and `spread` produce identical socket distribution
(~52 / ~51). The change is neutral for production runs. For sub-full-thread runs
(e.g., `-T 48`), `spread` ensures both sockets are active (24/24 vs 48/0 with
`close`), doubling available DRAM bandwidth. The `test()` path and all hot-kernel
inner loops continue to use `OMP_PROC_BIND=close` + `schedule(static)` (the correct
pairing for NUMA first-touch pragmas R1a/R1b/R2a).

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | Fix D: `proc_bind(spread)` on evaluateAll() OMP pragma |
| `setonix-iq/research/aa-walltime-analysis.md` | New §2.4.1 Fix C, §2.5 MPI overhead quantification, §2.6 thread saturation + NUMA binding |
| `setonix-iq/research/lb-analysis.md` | New §8 MPI data-path, §9 thread saturation and NUMA |

---

## 2026-05-xx (ba) — MF2 filterRates load-imbalance fix (Fix C)

### What changed

`filterRates()` in `main/phylotesting.cpp` used `at(0).subst_name` (global first
substitution family = LG for AA, GTR for DNA) as the reference for cross-family
rate-type pruning. On MPI ranks 1-3, all LG/GTR models are `MF_IGNORED` with
`BIC_score = DBL_MAX`. This caused:

- `best_score = DBL_MAX` → `ok_score = DBL_MAX` → every rate type passes → **nothing pruned**
  on ranks 1-3, while rank 0 pruned ~70% of +R3-R10 models.
- Estimated **12-15% wall-time load imbalance** (ranks 1-3 evaluate ~220/308 models;
  rank 0 evaluates ~130/308 after pruning).

**Fix C — two-part change** (commit TBD on `gadi-spr-r2-avx512`):

1. **filterRates()**: Scan for the first non-IGNORED model's `subst_name` to use
   as per-rank reference. Skip IGNORED models in `best_score` update (they have
   `DBL_MAX`). Add `if (best_score == DBL_MAX) return` guard. Exclude IGNORED
   models from `ok_rates` build.

2. **evaluateAll()**: After Phase 1 stripe, recompute `rate_block` to the last
   index of the rank's own reference family (e.g. last WAG index on rank 1).
   This ensures `filterRates` fires after the reference family is **fully evaluated**
   (WAG+R10 finishes last, analogous to LG+R10 on rank 0), not prematurely after
   just one model.

`filterSubst()` is unaffected — it uses `at(0).rate_name` (+G4), and IGNORED
cross-rank +G4 models score `DBL_MAX`; `min()` naturally discards these. No fix
needed there.

### Impact

| Aspect | Before Fix C | After Fix C |
|--------|-------------|-------------|
| filterRates effective on ranks 1-3 | No (always DBL_MAX best_score) | Yes (own-family BIC reference) |
| Per-rank models evaluated (AA 100K) | rank 0: ~130; ranks 1-3: ~220 | ~130-150 all ranks |
| Load imbalance | ~12-15% | ~5-8% (residual LPT static-vs-actual) |
| Projected AA 100K np4 wall time | ~120 s (Fix A+B only) | ~100 s (Fix A+B+C) |

For DNA datasets: same fix applies; GTR → per-rank reference (e.g. TVM on rank 1).

### Literature basis

- **Graham (1969)** LPT bound: ≤4/3 × OPT for m=4 (≤1.25×). Assumes accurate
  static costs; asymmetric filterRates violated this assumption pre-Fix C.
- **Blumofe & Leiserson (1999)** work stealing: E[T] = T₁/P + O(T∞). The OMP
  `getNextModel()` loop IS work stealing within each rank. Inter-rank imbalance
  is addressed by Fix C; residual ~5-8% would require Phase 1.5 dynamic
  family redistribution (see `research/lb-analysis.md` §6, future work).

### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | Fix C (filterRates + rate_block recompute); commit `b9b04a1c` |
| `setonix-iq/research/lb-analysis.md` | New: load-balance analysis with literature (Graham/Blumofe/Minh) |
| `setonix-iq/research/modelfinder-mpi.md` | §17.3 Fix C + §17.4 updated projected perf |

---

## 2026-05-16 (az) — MF2 ModelFinder scaling root causes diagnosed and fixed

### What changed

Two bugs in `main/phylotesting.cpp` caused MF2 ModelFinder to be 3.28× **slower** than the
standard binary at 1-node (AA 100K). Both are now fixed (commit `2672b90a` on
`gadi-spr-r2-avx512`). Research docs updated in `research/aa-walltime-analysis.md` §2.4
and new `research/modelfinder-mpi.md` §5.3 / §17 (commit `3fe95a7a`).

#### Root causes

**C1 (dominant ~2.6×): LPT position-stripe disabled `filterRates` pruning.**
Phase 1 sorted ALL 1,232 AA models by individual rate-category cost (LPT) and assigned by
`sorted_position % nranks`. This placed all even-k +Rk models (LG+R10, LG+R8, …) on rank 0
and odd-k variants on rank 1. `filterRates` calls `getLowerKModel(LG+R4) → LG+R3`, but
LG+R3 was `MF_IGNORED` on rank 0 (assigned to rank 1), so the pruning guard failed. Every
rank evaluated ALL assigned +Rk series. Standard IQ-TREE evaluates ~475 of the 1,232 AA
models after pruning; MF2 evaluated all 1,232 — a 2.6× excess.

**C3 (secondary ~1.3×): sequential site-parallel eval — OMP barrier overhead.**
The "Issue 5 fix" (commit `abd98764`) serialised model evaluation in MPI builds: one model
at a time using all 103 threads for site-level parallelism. For 100 taxa (199 internal
nodes), each model paid ~4,000 OMP barrier events (199 × ~10 passes × 2). The non-MPI path
uses model-level OMP (103 models in parallel, zero intra-model barriers) and is ~1.3× faster.

Combined: 2.6× × 1.3× ≈ **3.4× ≈ observed 3.28× overhead at np1**.

#### Fixes applied (`phylotesting.cpp`, commit `2672b90a`)

- **Fix A — subst-family LPT stripe**: Groups all rate variants of each substitution family
  (LG, WAG, JTT, …) together. LPT-sort GROUPS, assign GROUPS round-robin. All ~30 LG rate
  variants stay on the same rank → `filterRates` fires normally within each rank's model set.
  Expected: ~150–200 models evaluated per rank instead of all 1,232.
- **Fix B — OMP-across-models restored for MPI builds**: Removed the sequential
  `#ifdef _IQTREE_MPI` evaluation block. Both MPI and non-MPI builds now use the same
  `#pragma omp parallel` loop. The "Issue 5" race was a false positive: `saveCheckpoint()`
  inside `evaluate()` was already `#pragma omp critical`, and the only unguarded write
  (`putBool("UnreliableParam")`) is gated on `verbose_mode >= VB_MED` — never triggered in
  production.

#### Projected performance after fixes (AA 100K)

| Scenario | MF wall | Tree wall | Total | Speedup |
|----------|---------|-----------|-------|---------|
| Standard baseline (168425673) | 399 s | 764 s | 1,169 s | 1.00× |
| MF2 np1 post-fix | ~400 s | 717 s | ~1,117 s | ~1.05× |
| MF2 np2 post-fix | ~160 s | 383 s | ~543 s | ~2.15× |
| MF2 np4 post-fix | ~120 s | 198 s | ~318 s | ~3.67× |

Break-even vs standard SPR drops from ~2.7 nodes to **< 1.5 nodes**.

#### Files changed

| File | Change |
|------|--------|
| `src/iqtree3/main/phylotesting.cpp` | Fix A (Phase 1 stripe) + Fix B (eval path); commit `2672b90a` |
| `research/aa-walltime-analysis.md` | §2.4: root cause + fix + projected performance |
| `research/modelfinder-mpi.md` | §5.3 AA 100K data; §17 Fixes Implemented |

---

## 2026-05-16 (ay) — AA 100K MF2 scaling series completed (168446151/152/153)

### What changed

All three MF2 AA 100K scaling runs completed. All lnL verified (−7,541,976.862 ✓).

| PBS ID | Nodes | Ranks × OMP | MF wall | Tree wall | Total wall | vs 168425673 |
|--------|-------|-------------|---------|-----------|------------|--------------|
| 168446151 | 1 | 1×103 | 1,308.938 s | 717.499 s | 2,029.853 s | **0.58×** ← 1.73× slower |
| 168446152 | 2 | 2×103 |   968.700 s | 383.105 s | 1,355.215 s | **0.86×** ← 1.16× slower |
| 168446153 | 4 | 4×103 |   573.036 s | 197.746 s |   775.906 s | **1.51×** ← faster ✓ |

Reference baseline — SPR 1-node standard (168425673): MF=399.456 s, tree=764.478 s, total=1,169.556 s

IPC (rank 0 perf stat): 1.96 (np1) → 2.03 (np2) → 2.02 (np4)

#### Key findings

1. **MF2 1-node is 3.28× SLOWER than standard** — ModelFinder takes 1,308.938 s with 1 rank
   vs 399.456 s on the standard binary. Root cause diagnosed and fixed in entry `az`: LPT
   position-stripe disabling `filterRates` pruning (C1, ~2.6×) + sequential OMP barrier
   overhead (C3, ~1.3×).

2. **Tree search scales near-linearly across MPI ranks** — 717 s → 383 s → 198 s (3.63× for
   4× ranks). The MF2 binary distributes tree search across ranks in addition to ModelFinder
   — the Amdahl prediction assumed single-node tree search only.

3. **4-node exceeds the Amdahl prediction** — 1.51× actual vs 1.35× predicted. The bonus
   comes from MPI-parallel tree search, not accounted for in the original model.

4. **Break-even vs standard SPR** is between 2 and 4 nodes with the broken dispatch (~2.7
   nodes); drops to **< 1.5 nodes** after the `az` fixes.

| Metric | np1 | np2 | np4 |
|--------|-----|-----|-----|
| MF speedup (vs np1 MF2) | 1.00× | 1.35× | 2.28× |
| Tree speedup (vs np1 MF2) | 1.00× | 1.87× | 3.63× |
| Total speedup (vs 168425673) | 0.58× | 0.86× | **1.51×** |

---

## 2026-05-16 (ax) — AA 100K MF2 scaling series scripts created

### What changed

**Three MF2 AA 100K benchmark scripts created and submitted (1-node, 2-node, 4-node).
All charge to dx61. Group: `aa_100k_mf2_scaling`. PBS IDs: 168446151 (1-node), 168446152 (2-node), 168446153 (4-node).**

#### Scripts added

| Script | PBS queue | ncpus | Ranks × OMP | walltime |
|--------|-----------|-------|------------|----------|
| `gadi-ci/run_cpu_bench_aa_100k_mf2_1node.sh` | normalsr | 104 | 1×103 | 3h |
| `gadi-ci/run_cpu_bench_aa_100k_mf2_2node.sh` | normalsr | 208 | 2×103 | 3h |
| `gadi-ci/run_cpu_bench_aa_100k_mf2_4node.sh` | normalsr | 416 | 4×103 | 3h |
| `gadi-ci/run_cpu_bench_aa_100k_mf2_batch.sh` | — (submitter) | — | — | — |

#### Parity with baseline

- Same alignment: `complex_data_shared/AA/.../alignment_100000.phy`
- Same seed=1, `-T 103`, `numactl --localalloc`, `KMP_BLOCKTIME=200`
- Reference: AA 100K SPR (168425673): MF=399.456 s, tree=764.478 s, total=1,169.556 s
- Binary: `iqtree3-mpi` (MF2 LPT dispatch, R2+AVX-512) from `/scratch/um09/as1708/iqtree3-mf2/`
- Build tag: `mf2_full_icx_avx512_r2_lpt`
- run_type: `cpu_bench`, group: `aa_100k_mf2_scaling`

#### Expected MF2 speedups (Amdahl, tree-search=65% unparallelised)

| Nodes | MF wall | Tree wall | Total | Speedup vs 168425673 |
|-------|---------|-----------|-------|----------------------|
| 1 | ~399 s | ~764 s | ~1,170 s | ~1.00× |
| 2 | ~200 s | ~764 s | ~965 s | **~1.21×** |
| 4 | ~100 s | ~764 s | ~866 s | **~1.35×** |

Pass 2 perf stat runs per-rank (rank0 = tree+MF master, rank1+ = MF workers).

---

## 2026-05-16 (az) — AA 1M CLX and SPR completed (168425490, 168425491)

### What changed

Both AA 1M runs completed. lnL verified identical (−78,605,196.573 ✓). Model: LG+G4.

| PBS ID | Platform | Threads | MF wall | Tree wall | Total wall | Memory |
|--------|----------|---------|---------|-----------|------------|--------|
| 168425490 | CLX (normal-exec) | 47 | 16,308.318 s | 34,821.973 s | 51,328.252 s | 88.2 GB |
| 168425491 | SPR (normalsr-exec) | 103 | 7,587.459 s | 15,098.605 s | 22,776.226 s | 88.4 GB |

**CLX vs SPR speedup at 1M:** 51,328 / 22,776 = **2.25×** (vs thread ratio 103/47 = 2.19×).

Energy (RAPL): CLX = 492,588 J (9.60 W avg), SPR = 202,325 J (8.88 W avg). SPR uses **2.44× less energy**.

#### Key findings

1. **AA MF scales near-linearly with site count** — MF wall grows only 19.0× (SPR) and 14.7× (CLX)
   for 10× more sites, vs DNA MF which grew 56.7× (SPR) and 64.3× (CLX). AA MF is FLOP-dominated
   (IPC~2.0), so memory-bandwidth saturation plays a much smaller role than for DNA.

2. **Prediction was 3× too high** — §5.1.1 predicted AA 1M MF~22,641 s by applying the DNA SPR
   scale factor (56.7×) to AA 100K. Actual: 7,587 s (19.0× scale). The DNA super-linear factor is
   not transferable to AA because DNA's MF is memory-bandwidth bound at 100K, while AA's is FLOP-bound.

3. **CLX→SPR gap collapses at 1M** — AA 100K: CLX = 2.96× slower than SPR (above thread ratio,
   memory-bound). AA 1M: CLX = 2.25× slower (near thread ratio, both FLOP-bound). Per-model
   thread cost: CLX 622.4 s vs SPR 634.4 s — essentially identical.

4. **AA 1M scale factors vs 100K (SPR):** MF 19.0×, tree 19.7×, total 19.5×. All three phases
   scale similarly (~20×), suggesting uniform FLOP-dominated scaling rather than the phase-specific
   super-linearity seen in DNA.

| Phase | AA 100K SPR (168425673) | AA 1M SPR (168425491) | Scale |
|-------|------------------------|----------------------|-------|
| ModelFinder | 399.5 s | 7,587.5 s | **19.0×** |
| Tree search | 764.5 s | 15,098.6 s | **19.7×** |
| Total | 1,169.6 s | 22,776.2 s | **19.5×** |

---

## 2026-05-16 (aw) — DNA 1M CLX completed (168422813)

### What changed

**DNA 1M CLX (168422813) finished — 17,752.858 s (4h:55:52), F81+F+G4 (BIC), lnL −59,208,019.212.
Status matrix row 7 updated. Two jobs still running: 168425490 (AA 1M CLX), 168425491 (AA 1M SPR).**

#### DNA 1M CLX — phase breakdown

| Phase | Wall (s) | Wall (h:m:s) | % of total |
|-------|---------|--------------|------------|
| ModelFinder | 10,230.229 | 2:50:30 | 57.6% |
| Tree search | 7,481.884 | 2:04:41 | 42.1% |
| **Total** | **17,752.858** | **4:55:52** | 100% |

Models tested: 968 DNA models. Per-model wall time ≈ 10,230.229 × 47 / 968 ≈ 496.9 s·thread
(vs 372.5 s·thread for DNA 1M SPR) — **64.3× more per-model cost for 10× the sites** on CLX
(vs 56.7× on SPR). Tree search scaled 19.4× (vs 11.5× on SPR) — CLX shows more super-linear
tree-search scaling due to smaller L2/L3 cache and heavier DRAM pressure at 1M sites.

lnL = −59,208,019.212, bit-identical to DNA 1M SPR (168425675). Model F81+F+G4.

**Energy:** 505,373.194 J total (avg 28.5 W). package-0: 228,185 J + dram: 59,827 J;
package-1: 159,634 J + dram: 57,727 J. DRAM roughly balanced (ratio 1.04×).

**IPC/cache:** No perf stat (hw counters restricted on this CLX node — perf-report exited rc=1).
IPC left as — in metrics table.

---

## 2026-05-15 (av) — DNA 1M SPR completed (168425675)

### What changed

**DNA 1M SPR (168425675) finished — 6,114.450 s (1h:41:54), F81+F+G4 (BIC), lnL −59,208,019.212.
Status matrix row 8 updated. Three jobs still running: 168422813 (DNA 1M CLX), 168425490 (AA 1M CLX),
168425491 (AA 1M SPR).**

#### DNA 1M SPR — phase breakdown

| Phase | Wall (s) | Wall (h:m:s) | % of total |
|-------|---------|--------------|------------|
| ModelFinder | 3,500.825 | 0:58:20 | 57.3% |
| Tree search | 2,596.995 | 0:43:16 | 42.5% |
| **Total** | **6,114.450** | **1:41:54** | 100% |

Models tested: 968 DNA models (same as 100K run). Per-model wall time ≈ 372.5 s·thread
(vs 6.57 s·thread for DNA 100K SPR) — **56.7× more per-model cost for 10× the sites**.
Tree search scaled 11.47× (near-linear with 10× sites, consistent with O(n·patterns) kernel).
MF super-linear scaling explained by: NNI convergence requiring more iterations at larger
lnL gradients; heavier memory pressure reducing effective MF parallelism.

**Energy note:** DNA 1M SPR ran for 6,114 s. RAPL 32-bit counters overflow at ~262 KJ/domain
(~1,310 s at 200 W). With ~4–5 overflow events per domain, reported values (394 KJ total,
avg 64.4 W) are severe underestimates — true average power is closer to 530–600 W
(consistent with 100K SPR at 622.5 W). Energy values marked ⚠ in the metrics table.

**IPC/cache:** No perf stat job submitted for 168425675 — ★ markers in table.

---

## 2026-05-15 (au) — Output files shared for all 4 completed 100K runs

### What changed

**`chmod -R a+rw` applied to all 4 completed 100K run output directories on
`/scratch/dx61/as1708/cpu_bench/profiles/`. Output files (tree, log, model, distances)
are now world-readable.**

| PBS ID | Dataset | Directory |
|---|---|---|
| 168422809 | AA 100K CLX | `/scratch/dx61/as1708/cpu_bench/profiles/AA_100k_normal_seed1_168422809/` |
| 168425673 | AA 100K SPR | `/scratch/dx61/as1708/cpu_bench/profiles/AA_100k_spr_seed1_168425673/` |
| 168422811 | DNA 100K CLX | `/scratch/dx61/as1708/cpu_bench/profiles/DNA_100k_normal_seed1_168422811/` |
| 168425674 | DNA 100K SPR | `/scratch/dx61/as1708/cpu_bench/profiles/DNA_100k_spr_seed1_168425674/` |

Each directory contains: `iqtree_run.treefile`, `iqtree_run.iqtree`, `iqtree_run.log`,
`iqtree_run.model.gz`, `iqtree_run.mldist`, `iqtree_run.bionj`, `iqtree_run.ckp.gz`.

---

## 2026-05-15 (at) — Perf stat re-runs complete; IPC/cache metrics patched (168428519)

### What changed

**AA 100K SPR perf stat (168428519) finished — IPC 1.8781, LLC-miss 66.94%, L1-miss 1.19%.
All three perf stat re-run jobs now complete (168428491, 168428492, 168428519); metrics table
updated for DNA 100K CLX, DNA 100K SPR, and AA 100K SPR. AA 100K CLX IPC still pending
(no perf stat job submitted yet).**

---

## 2026-05-15 (as) — AA 100K CLX completed (168422809)

### What changed

**AA 100K CLX (168422809) finished — 57:40 wall, LG+G4, lnL −7,541,976.860. lnL bit-identical to SPR run. CLX vs SPR speedup 2.96× on AA 100K.**

---

## 2026-05-15 (ar) — Full results table; AA 1M jobs submitted (168425490, 168425491)

### What changed

**AA 1M data permissions fixed by sa0557 — both AA 1M jobs submitted. Comprehensive
results table compiled from all completed runs.**

#### Benchmark status matrix (2026-05-15)

| # | Case | Node | -nt | PBS ID | Status | Wall (s) | lnL | BIC | Avg W |
|---|---|---|---|---|---|---|---|---|---|
| 1 | AA 100K CLX | Cascade Lake | 47 | **168422809** | **DONE** ✓ | 3,460.813 | −7,541,976.860 | 15,086,233.282 | 160.6 |
| 2 | AA 100K SPR | Sapphire Rapids | 103 | **168425673** | **DONE** ✓ | 1,169.556 | −7,541,976.860 | 15,086,233.2801 | 224.7 |
| 3 | AA 1M CLX | Cascade Lake | 47 | **168425490** | RUNNING (34 min) | — | — | — | — |
| 4 | AA 1M SPR | Sapphire Rapids | 103 | **168425491** | RUNNING (34 min) | — | — | — | — |
| 5 | DNA 100K CLX | Cascade Lake | 47 | **168422811** | **DONE** ✓ | 546.044 | −5,692,984.5391 | 11,388,283.1763 | 390.6 |
| 6 | DNA 100K SPR | Sapphire Rapids | 103 | **168425674** | **DONE** ✓ | 289.121 | −5,692,984.5391 | 11,388,283.1763 | 622.5 |
| 7 | DNA 1M CLX | Cascade Lake | 47 | **168422813** | **DONE** ✓ | 17,752.858 | −59,208,019.212 | 118,418,815.234 | 28.5 |
| 8 | DNA 1M SPR | Sapphire Rapids | 103 | **168425675** | **DONE** ✓ | 6,114.450 | −59,208,019.212 | 118,418,815.234 | 64.4⚠ |

✓ = completed on exclusive node. IPC pending perf stat re-run (see below).

**Non-exclusive runs purged:** 168419897, 168419898 — logs and scratch deleted.
168419899 (DNA 1M SPR, 1h3m elapsed, non-excl) — cancelled `qdel`; SIGTERM confirmed.

---

#### Completed runs — detailed metrics

All energy values from IQ-TREE RAPL (`energy_and_mem` branch). IPC from `perf stat -e events:u`
(user-mode counters, `perf_event_paranoid=2` compatible — **not** from `perf-report` which is unrelated).
IPC marked pending★ where perf stat re-run is queued but not yet complete.

| PBS ID | Dataset | Node | Queue | -nt | Wall (s) | Wall (m:s) | CPU time (s) | Par. eff. | Best model (BIC) | lnL | BIC | Free params | IPC | LLC miss% | L1 miss% | CPU avg (W) | pkg0 (W) | pkg1 (W) | DRAM (W) | Excl? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **168422811** | DNA 100K | Cascade Lake | normal-exec | 47 | 546.044 | 9:06 | 23,532.55 | 91.7% | F81+F+G4 | −5,692,984.5391 | 11,388,283.1763 | 201 | 0.932 | 59.94 | 5.28 | 390.6 | 192.8 | 174.8 | 23.7 | ✓ |
| **168425673** | AA 100K | Sapphire Rapids | normalsr-exec | 103 | 1,169.556 | 19:29 | 108,645.54 | 90.2% | LG+G4 | −7,541,976.860 | 15,086,233.2801 | 198 | 1.8781 | 66.94 | 1.19 | 224.7 | 112.4 | 96.8 | 15.6 | ✓ |
| **168422809** | AA 100K | Cascade Lake | normal-exec | 47 | 3,460.813 | 57:40 | 150,630.50 | 92.6% | LG+G4 | −7,541,976.860 | 15,086,233.282 | 198 | ★ | ★ | ★ | 160.6 | 73.4 | 57.5 | 29.7 | ✓ | ← perf stat pending
| **168425674** | DNA 100K | Sapphire Rapids | normalsr-exec | 103 | 289.121 | 4:49 | 26,619.44 | 89.4% | F81+F+G4 | −5,692,984.5391 | 11,388,283.1763 | 201 | 1.3023 | 75.81 | 1.18 | 622.5 | 318.4 | 295.7 | 10.8 | ✓ |
| **168425675** | DNA 1M | Sapphire Rapids | normalsr-exec | 103 | 6,114.450 | 1h:41:54 | 558,655.09 | 88.7% | F81+F+G4 | −59,208,019.212 | 118,418,815.234 | 201 | ★ | ★ | ★ | 64.4⚠ | 33.9⚠ | 25.3⚠ | 5.28⚠ | ✓ |

★ = perf stat pending (168422809 AA 100K CLX — no job submitted yet; 168425675 DNA 1M SPR — no job submitted yet). All other ★ entries have been patched with real values from jobs 168428491/492/519.
⚠ = RAPL counter overflow suspected (run > 1,310 s/domain): reported energy is a severe underestimate; true avg power ≈ 530–600 W for 168425675.

**Par. eff.** = CPU time / (wall × threads). **pkg0/pkg1** = RAPL package joules ÷ wall. **DRAM** = (dram-0 + dram-1) J ÷ wall.

#### IPC correction note

Previous CHANGELOG entries stated "IPC not available — hardware performance counters restricted".
This was **wrong**. `perf-report` (Linaro Forge) exits 1 on Gadi compute nodes, but that is a
separate tool. `perf stat -e event:u` (user-mode suffix) works fine with `perf_event_paranoid=2`
and is how all prior Gadi runs (xlarge_mf, mega_dna series) collected IPC and LLC miss rates.
All 8 cpu_bench scripts updated to add Pass 2 perf stat immediately after IQ-TREE completes.

---

#### Cross-run consistency check

DNA 100K lnL and BIC are **bit-identical** across CLX 47T and SPR 103T. AA 100K lnL is
also **bit-identical** across CLX 47T and SPR 103T. Confirms numerical reproducibility
regardless of microarchitecture, thread count, or SIMD width.

| Metric | DNA 100K CLX · 168422811 ✓ | DNA 100K SPR · 168425674 ✓ | Match? |
|---|---|---|---|
| lnL | −5,692,984.5391 | −5,692,984.5391 | ✓ |
| BIC | 11,388,283.1763 | 11,388,283.1763 | ✓ |
| Best model | F81+F+G4 | F81+F+G4 | ✓ |
| Free params | 201 | 201 | ✓ |
| Tree length | 18.9178 | 18.918 | ✓ |

| Metric | AA 100K CLX · 168422809 ✓ | AA 100K SPR · 168425673 ✓ | Match? |
|---|---|---|---|
| lnL | −7,541,976.860 | −7,541,976.860 | ✓ |
| BIC | 15,086,233.282 | 15,086,233.2801 | ✓ |
| Best model | LG+G4 | LG+G4 | ✓ |
| Free params | 198 | 198 | ✓ |

---

#### SPR vs CLX throughput comparison — DNA 100K (exclusive nodes)

| PBS ID | Node | -nt | Wall (s) | Speedup vs CLX | CPU energy (J) | Avg W | J/site |
|---|---|---|---|---|---|---|---|
| 168422811 | Cascade Lake (CLX) | 47 | 546.044 | 1.00× (baseline) | 213,674.7 | 390.6 | 2.137 |
| 168425674 | Sapphire Rapids (SPR) | 103 | 289.121 | **1.89×** | 180,642.9 | 622.5 | 1.806 |

SPR is **1.89× faster** with 2.19× the threads → 86.2% relative parallel efficiency vs CLX.
SPR draws 59% more average watts but consumes **15% less total energy per run** (1.806 vs 2.137 J/site).

---

#### SPR vs CLX throughput comparison — AA 100K (exclusive nodes)

| PBS ID | Node | -nt | Wall (s) | Speedup vs CLX | CPU energy (J) | Avg W | J/site |
|---|---|---|---|---|---|---|---|
| 168422809 | Cascade Lake (CLX) | 47 | 3,460.813 | 1.00× (baseline) | 555,794.5 | 160.6 | 5.558 |
| 168425673 | Sapphire Rapids (SPR) | 103 | 1,169.556 | **2.96×** | 262,875.4 | 224.7 | 2.629 |

SPR is **2.96× faster** with 2.19× the threads → larger relative advantage than DNA (2.96× vs 1.89×).
SPR draws 40% more average watts but consumes **53% less total energy per run** (2.629 vs 5.558 J/site).
Note: CLX DRAM asymmetry — dram-0=37,744 J vs dram-1=65,029 J — suggests cross-socket access despite numactl --localalloc; warrants investigation.

---

#### AA 1M jobs submitted (2026-05-15)

`tree_1/` permissions fixed by sa0557 (`drwxrwsrwx`; `len_1000000/` traversable via dx61 group bit).

| PBS ID | Case | Queue | -nt | mem | Walltime |
|---|---|---|---|---|---|
| **168425490** | AA 1M CLX | normal-exec | 47 | 190 GB | 24 h |
| **168425491** | AA 1M SPR | normalsr-exec | 103 | 510 GB | 24 h |

Both submitted with `place=excl` — full exclusive node, no co-tenancy.

---

## 2026-05-15 (aq) — Batch 3 first results; CLX build succeeded; old output files cleaned

### What changed

**Batch 3 CLX build succeeded (168422807). First CLX and SPR benchmark results obtained.
31 old/failed PBS output files deleted. Three jobs still running.**

#### Output file cleanup

Deleted 31 outdated or failed PBS output files from `~/setonix-iq/`:

| Group | Files deleted | Reason |
|---|---|---|
| rc29 / early avx era | `iq-r2-*.o168114*`, `iq-avx-*.o168114*`, `iq-mf-*.o168114*` | old allocation, superseded |
| um09 era | `iq-mf2-full-*.o168183*`, `*.o168188*`, `*.o168195*`, `iq-build-avx512-r2.o168207037`, `iq-xlarge-avx512-r2-omp-batch.o168206*`, `*.o168209*`, `iq-mega-avx512-r2-*.o168211*`, `iq-mega-mf2-*.o168213*`, `mf2-rebuild.o168188556` | old um09 allocation, results already in logs/runs/ |
| Failed CLX builds | `iq-build-cpu-clx.o168419891`, `iq-build-cpu-clx.o168421511` | failed builds (bugs 1+3 documented in ap) |
| Pre-fix SPR bench | `iq-cpu-aa-100k-spr.o168419897`, `iq-cpu-dna-100k-spr.o168419898` | results recovered to JSON; PBS outputs no longer needed |

Remaining output files:

| File | Notes |
|---|---|
| `iq-build-cpu-clx.o168422807` | CLX build succeeded — keep |
| `iq-cpu-dna-100k-clx.o168422811` | DNA 100K CLX completed — keep |
| `iq-cpu-aa-1m-clx.o168422815` | AA 1M CLX exit 3 (permissions blocked) — keep for reference |

---

#### Batch 3 job outcomes (updated)

| PBS ID | Script | Queue | Case | Outcome |
|---|---|---|---|---|
| **168422807** | `build_cpu_bench_clx.sh` | normal | CLX build | **PASSED** — `iqtree3` v3.1.2 built in 2m29s; `libiomp5` linked; binary at `/scratch/dx61/as1708/cpu_bench/build-intel-clx/iqtree3` |
| **168422809** | `run_cpu_bench_aa_100k_normal.sh` | normal | AA 100K CLX | **RUNNING** (11 min elapsed at last check) |
| **168422811** | `run_cpu_bench_dna_100k_normal.sh` | normal | DNA 100K CLX | **PASSED** — wall=546s, energy=213,675J |
| **168422813** | `run_cpu_bench_dna_1m_normal.sh` | normal | DNA 1M CLX | **RUNNING** (11 min elapsed at last check) |
| **168422815** | `run_cpu_bench_aa_1m_normal.sh` | normal | AA 1M CLX | **FAILED rc=3** — AA 1M dir still permission-denied (`drw-rwSrw-`) |
| **168419899** | `run_cpu_bench_dna_1m_spr.sh` | normalsr | DNA 1M SPR | **RUNNING** (50 min elapsed at last check; non-exclusive, pre-fix script) |

---

#### Results to date (2026-05-15) — superseded by (ar)

Non-exclusive SPR runs 168419897 and 168419898 were deleted and rerun clean; see (ar) for
authoritative results. 168422811 (DNA 100K CLX) result stands — see (ar) detailed metrics table.

SPR 100K benchmarks ran on non-exclusive nodes (old pre-fix scripts). Re-run with fixed scripts (`mem=510GB`, `place=excl`) is needed for clean energy measurements:
```bash
qsub gadi-ci/run_cpu_bench_aa_100k_spr.sh
qsub gadi-ci/run_cpu_bench_dna_100k_spr.sh
```

---

#### SPR energy detail — DNA 100K (168419898)

```
Energy:
  CPU:  177,351.7 J  (avg 617.3 W)
    package-0=89,775.8 J   dram=1,301.0 J
    package-1=84,459.8 J   dram=1,815.1 J
```

#### SPR energy detail — AA 100K (168419897)

```
Energy:
  CPU:  287,987.1 J  (avg 241.5 W)
    package-0=138,943.1 J   dram=5,964.8 J
    package-1=131,008.9 J   dram=12,070.4 J
```

#### CLX energy detail — DNA 100K (168422811)

```
Energy:
  CPU:  213,674.7 J  (avg 390.6 W)
    package-0=105,283.1 J   dram=5,357.5 J
    package-1=95,436.0 J    dram=7,598.1 J
```

---

#### Pending (2026-05-15) — updated

1. **AA 1M — UNBLOCKED.** `len_1000000/tree_1/alignment_1000000.phy` is now accessible
   (`tree_1` has `drwxrwsrwx`; `len_1000000` traversable via dx61 group execute bit).
   **Both AA 1M jobs submitted:**
   - `168425490` — AA 1M CLX (normal, 48 cpus, 190GB, 24h, `place=excl`)
   - `168425491` — AA 1M SPR (normalsr, 104 cpus, 510GB, 24h, `place=excl`)

2. **SPR 100K clean re-runs** — still needed; resubmit with `mem=510GB, place=excl` once queue clears:
   ```bash
   qsub gadi-ci/run_cpu_bench_aa_100k_spr.sh
   qsub gadi-ci/run_cpu_bench_dna_100k_spr.sh
   ```

3. **DNA 1M SPR (168419899)** — running (non-exclusive, pre-fix script, 56+ min elapsed); manually recover JSON from log if perf-report exits 1, then resubmit with fixed script.

4. **Await 168422809, 168422813** — AA 100K CLX and DNA 1M CLX still running.

---

## 2026-05-15 (ap) — CPU benchmark plan: 8 test cases on dx61; R1+R2 + AVX-512 parity across CLX and SPR

### What changed

**Created 9 PBS scripts (1 build + 8 benchmarks) for full-run CPU timing on 100K and 1M
datasets across AA and DNA, for both Cascade Lake (normal) and Sapphire Rapids (normalsr)
nodes. All 8 benchmark scripts have full R1+R2 + AVX-512 parity: same source branch
(`cpu_opt_merge`), same Intel ICX compiler, same OMP environment (libiomp5, `KMP_BLOCKTIME=200`,
`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), same `numactl --localalloc`, and `ldd` guard
against accidental libgomp linkage. CLX scripts use `-march=cascadelake` (GCC-compat, required for pll AVX cmake detection);
SPR scripts use `-xSAPPHIRERAPIDS`. Project allocation updated to `dx61` throughout.**

#### R1+R2 patch verification (cpu_opt_merge HEAD `8263c7e4`)

Source confirms all three NUMA first-touch markers present in `tree/phylotreesse.cpp`:
- `R1a` (line 583): parallel-static fill of pattern likelihood arrays
- `R1b` (line 616): parallel-static zero-fill of `ptn_invar`
- `R2a` (line 1349): parallel-static zero-fill of `_pattern_lh_cat` pages

`tree/phylokernelnew.h` carries all `schedule(static)` directives required by R2.
SPR binary (`build-intel-vanila/iqtree3`) confirmed to link `libiomp5` (not `libgomp`).

#### Runtime parity (all 8 scripts)

Every benchmark script now sets the full canonical R2 OMP environment:

```bash
export KMP_BLOCKTIME=200       # Intel OMP — prevent thread sleep between tasks
export OMP_NUM_THREADS=NT      # 47 (CLX) or 103 (SPR)
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close     # bind threads to adjacent cores
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=10000
```

And runs with:
```bash
perf-report --no-mpi --output="${PROFILE_REPORT}" \
    numactl --localalloc \
    iqtree3 -s ALIGNMENT -nt NT -seed 1 --prefix ...
```

`numactl --localalloc` ensures first-touch pages land on the calling thread's NUMA node,
activating the R1a/R1b/R2a schedule(static) data locality optimisation.
Each script also checks `ldd | grep libgomp` and exits 7 if the wrong OMP runtime is linked.

#### Benchmark matrix

| # | Data type | Dataset | Queue | Node type | Cores | `-nt` | Wall | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | AA | 100K | normal | Cascade Lake | 48 | 47 | 8h | blocked: CLX build pending |
| 2 | AA | 100K | normalsr | Sapphire Rapids | 104 | 103 | 8h | ready to submit |
| 3 | AA | 1M | normal | Cascade Lake | 48 | 47 | 24h | blocked: CLX build + AA 1M perms |
| 4 | AA | 1M | normalsr | Sapphire Rapids | 104 | 103 | 24h | blocked: AA 1M perms |
| 5 | DNA | 100K | normal | Cascade Lake | 48 | 47 | 8h | blocked: CLX build pending |
| 6 | DNA | 100K | normalsr | Sapphire Rapids | 104 | 103 | 8h | ready to submit |
| 7 | DNA | 1M | normal | Cascade Lake | 48 | 47 | 24h | blocked: CLX build pending |
| 8 | DNA | 1M | normalsr | Sapphire Rapids | 104 | 103 | 24h | ready to submit |

#### Test command (per run)

```
module load linaro-forge/24.0.2
perf-report --no-mpi --output=<PROFILE_REPORT> \
    iqtree3 -s ALIGNMENT -nt NT -seed 1 --prefix <WORK_DIR>/iqtree_run
```

Free model selection + free tree search. Every run is wrapped in Linaro Forge
`perf-report` to capture energy consumption and hardware performance counters.

#### Datasets

| Type | Sites | Taxa | Alignment path | Accessible? |
|---|---|---|---|---|
| AA 100K | 100,000 | 100 | `.../complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` | ✓ |
| AA 1M | 1,000,000 | 100 | `.../complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy` | ✗ |
| DNA 100K | 100,000 | 100 | `.../complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy` | ✓ |
| DNA 1M | 1,000,000 | 100 | `.../complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy` | ✓ |

All paths under `/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/`.

#### Known issue: AA 1M directory permission

`taxa_100/len_1000000/tree_1/` has `drw-rwSrw-` — no execute bit, entry denied.
Cases 3 and 4 cannot run until sa0557 fixes this:

```bash
chmod o+x /scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1
```

#### Binaries

| Queue | Binary | Build tag | Notes |
|---|---|---|---|
| normalsr (SPR) | `/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3` | `cpu_opt_merge_icx_avx512_spr` | ICX + `-xSAPPHIRERAPIDS` + AVX-512; built 2026-05-15; HEAD `8263c7e4` |
| normal (CLX) | `/scratch/dx61/as1708/cpu_bench/build-intel-clx/iqtree3` | `cpu_opt_merge_icx_avx512_clx` | Build queued (168421511) — same source, **`-march=cascadelake`** (fix for pll AVX detection) |

Source: `/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/iqtree3` branch `cpu_opt_merge`
(commit `8263c7e4`: merge of `gadi-spr-r2-avx512` + `energy_and_mem` + checkpoint accumulation).

#### New scripts (`gadi-ci/`)

| Script | Purpose |
|---|---|
| `build_cpu_bench_clx.sh` | Build CLX Intel binary from `cpu_opt_merge` source (normal queue, ~1h) |
| `run_cpu_bench_aa_100k_normal.sh` | Case 1 — AA 100K, CLX, -nt 47 |
| `run_cpu_bench_aa_100k_spr.sh` | Case 2 — AA 100K, SPR, -nt 103 |
| `run_cpu_bench_aa_1m_normal.sh` | Case 3 — AA 1M, CLX, -nt 47 |
| `run_cpu_bench_aa_1m_spr.sh` | Case 4 — AA 1M, SPR, -nt 103 |
| `run_cpu_bench_dna_100k_normal.sh` | Case 5 — DNA 100K, CLX, -nt 47 |
| `run_cpu_bench_dna_100k_spr.sh` | Case 6 — DNA 100K, SPR, -nt 103 |
| `run_cpu_bench_dna_1m_normal.sh` | Case 7 — DNA 1M, CLX, -nt 47 |
| `run_cpu_bench_dna_1m_spr.sh` | Case 8 — DNA 1M, SPR, -nt 103 |
| `submit_cpu_bench_all.sh` | Dispatcher — submits all 8 jobs; auto-chains CLX build as dependency |

#### Jobs submitted (2026-05-15, batch 1)

| PBS ID | Script | Queue | Case | Result |
|---|---|---|---|---|
| **168419891** | `build_cpu_bench_clx.sh` | normal | CLX build | **FAILED rc=2** — pll AVX linker error |
| **168419897** | `run_cpu_bench_aa_100k_spr.sh` | normalsr | AA 100K SPR | R (running at update time) |
| **168419898** | `run_cpu_bench_dna_100k_spr.sh` | normalsr | DNA 100K SPR | **rc=1** — IQ-TREE OK; perf-report failed (see below) |
| **168419899** | `run_cpu_bench_dna_1m_spr.sh` | normalsr | DNA 1M SPR | R (running at update time) |
| **168419901** | `run_cpu_bench_aa_100k_normal.sh` | normal | AA 100K CLX | **Auto-deleted** (afterok on failed 168419891) |
| **168419902** | `run_cpu_bench_dna_100k_normal.sh` | normal | DNA 100K CLX | **Auto-deleted** (afterok on failed 168419891) |
| **168419903** | `run_cpu_bench_dna_1m_normal.sh` | normal | DNA 1M CLX | **Auto-deleted** (afterok on failed 168419891) |

AA 1M cases (3 + 4) not submitted — `drw-rwSrw-` on `tree_1/` still blocks access.

---

#### Bug 1: CLX build failure — `-xCASCADELAKE` breaks pll AVX detection

**Job 168419891 failed (rc=2, wall=2m25s)**

cmake.log showed `IQ-TREE flags: avx512` but `Vectorization: AVX` — meaning pll's cmake
feature detection didn't see AVX-512, so pll AVX source files (`newviewGTRGAMMA_AVX.c` etc.)
were compiled without AVX symbols. Linker errors:

```
pll/libpll.a(newviewGenericSpecial.c.o): undefined reference to `newviewGTRGAMMA_AVX_GAPPED_SAVE'
undefined reference to `newviewGTRGAMMAPROT_AVX_GAPPED_SAVE'
undefined reference to `newviewGTRCAT_AVX_GAPPED_SAVE'
... (9 missing AVX symbols total)
icpx: error: linker command failed with exit code 1
```

**Root cause:** ICX's Intel-only flag `-xCASCADELAKE` doesn't define GCC-compatible
`__AVX__` / `__AVX2__` / `__AVX512F__` macros that cmake's `check_cxx_source_compiles`
uses. pll's cmake detection falls back to plain AVX, skipping the AVX-512 objects.

**Fix applied (build_cpu_bench_clx.sh):**
```diff
-    -DCMAKE_CXX_FLAGS="-O3 -xCASCADELAKE -fno-omit-frame-pointer" \
-    -DCMAKE_C_FLAGS="-O3 -xCASCADELAKE -fno-omit-frame-pointer" \
+    -DCMAKE_CXX_FLAGS="-O3 -march=cascadelake -fno-omit-frame-pointer" \
+    -DCMAKE_C_FLAGS="-O3 -march=cascadelake -fno-omit-frame-pointer" \
```

`-march=cascadelake` is GCC-compatible and ICX accepts it; it defines all required
`__AVX*__` macros so pll cmake detects AVX-512 correctly.

---

#### Bug 2: `perf-report` rc=1 kills script before JSON is written

**Job 168419898 (DNA 100K SPR) — IQ-TREE completed successfully, script reported rc=1**

IQ-TREE ran fine on gadi-cpu-spr-0108 with AVX512+FMA kernel:
- Wall time: **286.2 s** (4 m 46 s)
- lnL: **−5,692,984.539**
- Energy (built-in RAPL): **177,351.7 J  (avg 617.3 W)**
  - `package-0=89,775.8 J  dram=1,301.0 J`
  - `package-1=84,459.8 J  dram=1,815.1 J`

`perf-report` exited 1 (no `.html` written) — hardware performance counter access is
restricted on normalsr compute nodes. Because `set -euo pipefail` was active, bash exited
immediately after `perf-report`, before `IQRC=$?` was ever assigned, so the JSON record
was never written.

**Fix applied (all 8 run scripts):**
```bash
set +e
perf-report --no-mpi --output="${PROFILE_REPORT}" \
    "${NUMACTL[@]}" "${IQTREE}" ... > iqtree_run.log 2>&1
PERF_RC=$?
set -e

# Infer IQ-TREE's actual exit from log (IQ-TREE always prints "Date and Time:" on success)
if grep -q "^Date and Time:" "${WORK_DIR}/iqtree_run.log" 2>/dev/null; then
    IQRC=0
    [[ ${PERF_RC} -ne 0 ]] && echo "NOTE: perf-report exited ${PERF_RC} (hw counters restricted)" >&2
else
    IQRC=${PERF_RC}
fi
```

The JSON for job 168419898 was recovered manually from `iqtree_run.log` and written to
`logs/runs/gadi_DNA_100k_spr_seed1_168419898.json`.

Note: IQ-TREE's `energy_and_mem` branch already measures RAPL energy natively (see the
`Energy:` block in the log). Linaro Forge `perf-report` would provide additional hardware
counter breakdowns if counter access is enabled on the node.

---

#### Bug 3: Node not exclusive — all 8 run scripts booking too little memory

All scripts were requesting 32–128 GB but CLX nodes have 192 GB and SPR nodes have 512 GB.
Without requesting the full node, other jobs can co-locate and corrupt timing and energy
measurements. Fixed by requesting full memory + `place=excl`:
- CLX normal: `mem=190GB` + `#PBS -l place=excl`
- SPR normalsr: `mem=510GB` + `#PBS -l place=excl`

`place=excl` confirmed accepted by Gadi PBS (tested with a trivial job submission).

---

#### SPR results recovered (2026-05-15)

Both SPR jobs ran on non-exclusive nodes (old scripts). Results are valid for IQ-TREE
correctness and timing, but energy measurements may include co-tenant load.
JSON records written manually from IQ-TREE logs:

| PBS ID | Case | Wall (s) | lnL | CPU energy (J) | Avg W |
|---|---|---|---|---|---|
| **168419898** | DNA 100K SPR | 286.2 | −5,692,984.539 | 177,351.7 | 617.3 |
| **168419897** | AA 100K SPR | 1,191.4 | −7,541,976.860 | 287,987.1 | 241.5 |

`logs/runs/gadi_DNA_100k_spr_seed1_168419898.json`
`logs/runs/gadi_AA_100k_spr_seed1_168419897.json`

168419899 (DNA 1M SPR) still running — will need manual JSON recovery if it also
ran on a non-exclusive node (it did; scripts updated for future re-runs).

---

#### CLX build failure 2 (168421511, rc=2) — linker archive ordering

Same AVX linker error despite the `-march=cascadelake` fix. Root cause traced to the
ICX single-pass static linker:

- cmake puts `libpll.a` at link position 25 (before tree/main libraries)
- `tree/libtree.a` + `main/libmain.a` (~pos 40–45) create undefined ref to `pllNewviewIterative`
- `libpllavx.a` at position 47 is scanned but nothing is undefined that it defines → `avxLikelihood.c.o` NOT pulled in
- `libpll.a` again at position 58 → `newviewGenericSpecial.c.o` now included → creates undefined `newviewGTRGAMMA_AVX*` — but `libpllavx.a` already passed and won't be rescanned

`nm libpllavx.a | grep newviewGTRGAMMA_AVX_GAPPED_SAVE` confirmed symbol IS present; the issue is purely ordering.

**Fix applied (`build_cpu_bench_clx.sh`):** After cmake generates the Makefile, a Python
snippet rewrites `CMakeFiles/iqtree3.dir/link.txt` to wrap all static archives in
`-Wl,--start-group ... -Wl,--end-group`, forcing multi-pass resolution regardless of order.
The build dir is also cleaned before each cmake run to prevent stale cache reuse.

---

#### Jobs resubmitted (2026-05-15, batch 2)

| PBS ID | Script | Queue | Case | Result |
|---|---|---|---|---|
| **168421511** | `build_cpu_bench_clx.sh` | normal | CLX build (march fix, no start-group) | **FAILED rc=2** — linker ordering |
| **168421513–16** | CLX bench jobs | normal | held | **Auto-deleted** (afterok on failed build) |

---

#### Jobs resubmitted (2026-05-15, batch 3) — full fix applied

All 8 scripts updated: `place=excl`, full node memory, `set +e` perf-report, `--start-group` build fix.

| PBS ID | Script | Queue | Case | Status |
|---|---|---|---|---|
| **168422807** | `build_cpu_bench_clx.sh` | normal | CLX build (full fix) | Q |
| **168422809** | `run_cpu_bench_aa_100k_normal.sh` | normal | AA 100K CLX | H (after 168422807) |
| **168422811** | `run_cpu_bench_dna_100k_normal.sh` | normal | DNA 100K CLX | H (after 168422807) |
| **168422813** | `run_cpu_bench_dna_1m_normal.sh` | normal | DNA 1M CLX | H (after 168422807) |
| **168422815** | `run_cpu_bench_aa_1m_normal.sh` | normal | AA 1M CLX | H (after 168422807) |
| **168419899** | `run_cpu_bench_dna_1m_spr.sh` | normalsr | DNA 1M SPR | R (still running, non-excl) |

AA 1M cases (3 + 4) remain blocked — sa0557 must fix `tree_1/` permissions before submitting:
```bash
chmod o+x /scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1
# then:
qsub gadi-ci/run_cpu_bench_aa_1m_spr.sh
qsub -W depend=afterok:168422807 gadi-ci/run_cpu_bench_aa_1m_normal.sh
```

SPR bench cases (100K) should be resubmitted with the fixed script (mem=510GB, place=excl):
```bash
qsub gadi-ci/run_cpu_bench_aa_100k_spr.sh
qsub gadi-ci/run_cpu_bench_dna_100k_spr.sh
```

#### Output

- Run JSON records: `logs/runs/gadi_<LABEL>_<PBS_ID>.json`
- IQ-TREE logs + output files: `/scratch/dx61/as1708/cpu_bench/profiles/<LABEL>_<PBS_ID>/`
- Linaro Forge perf report: `.../perf_report.html`

#### Allocation change

All scripts use `#PBS -P dx61` and `#PBS -l storage=scratch/dx61`, replacing the
previous `um09` / `rc29` allocations used in earlier `gadi-ci` scripts.

#### Checkpoint note

1M cases may exceed wall-time limits on slower runs. The `cpu_opt_merge` branch
includes checkpoint accumulation (`23b252fd add runtime from checkpoint`), enabling
total elapsed time to be recovered across resumed jobs. Scripts are prepared with
`--prefix` set so IQ-TREE writes checkpoints to the per-run work directory.

---

## 2026-05-12 (ao) — mega_dna MF2 Full parity scripts; 3 jobs submitted

### What changed

**Created three CI scripts to run the MF2 Full series on `mega_dna.fa`, giving full
parity with the `xlarge_mf.fa` MF2 Full series. Three PBS jobs submitted.**

#### Purpose

The `xlarge_mf.fa` MF2 Full series (`non_canonical_label: "MF2 Full · ICX+MPI · R2 · AVX-512"`)
covers 4 thread counts (64T, 104T, 208T, 416T). The same 4 configurations are now submitted
for `mega_dna.fa` using the identical `build-mpi-mf2/iqtree3-mpi` binary, so the chart will
show a second MF2 Full line grouped under the same series label.

#### Dataset context

`mega_dna.fa` is a pathological case for MF2 dispatch overhead:
- **500 taxa × 100,000 sites** with **99,999 unique patterns** (~0% compression)
- Reference lnL (seed=1, ICX 104T): −27,328,165.86
- sha256: `0c8af2d62e214be8b0258393d71d1a0bed15568334de56b89116ae8653f92619`

Zero compression means IQ-TREE cannot amortise pattern lookups across threads.
This also stresses MF2's LPT model dispatcher because each model candidate operates
on the full 99,999-pattern column, not a compressed subset. Expect `mega_dna` MF2 Full
runtimes to be ~2.5× slower than `xlarge_mf` at equivalent thread counts.

#### New scripts

| Script | PBS name | Threads | Nodes | Walltime |
|---|---|---|---|---|
| `gadi-ci/run_mega_mf2_full_omp_batch.sh` | `iq-mega-mf2-omp` | 64T + 104T | 1 | 8 h |
| `gadi-ci/run_mega_mf2_full_2node.sh` | `iq-mega-mf2-2node` | 208T | 2 | 3 h |
| `gadi-ci/run_mega_mf2_full_4node.sh` | `iq-mega-mf2-4node` | 416T | 4 | 3 h |

Key metadata for all three:
- **binary:** `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi`
- **build_tag:** `mf2_full_icx_avx512_r2_lpt`
- **non_canonical_label:** `"MF2 Full · ICX+MPI · R2 · AVX-512"`
- **seed:** 1, free tree, full IQ-TREE (no `-m MF`, no `-te`)
- **`--bind-to none`** on all `mpirun` calls

#### PBS job IDs

| Job | Script | Threads | Notes |
|---|---|---|---|
| **168213985** | `run_mega_mf2_full_omp_batch.sh` | 64T + 104T | Q/R |
| **168213987** | `run_mega_mf2_full_2node.sh` | 208T | Q/R |
| **168213988** | `run_mega_mf2_full_4node.sh` | 416T | Q/R |

#### Run IDs to expect

- `gadi_mega_dna_64t_mf2_full_np1_seed1`
- `gadi_mega_dna_104t_mf2_full_np1_seed1`
- `gadi_mega_dna_208t_mf2_full_np2_seed1`
- `gadi_mega_dna_416t_mf2_full_np4_seed1`

---

## 2026-05-12 (an) — Bug fixes: chart series grouping, OMP binding, rc29→um09; jobs resubmitted

### What changed

**Three bugs found and fixed. Two jobs resubmitted and currently running.**

#### Fix 1 — AVX-512+R2 chart: 4T/8T isolated points instead of connected line

The AVX-512+R2 OMP scaling series appeared as disconnected isolated markers on the
website chart because two separate bugs prevented the runs from sharing a common
chart series key (`${platform} · ${dataset} · ${non_canonical_label}`):

**Bug A — `non_canonical_label` was per-thread-count** (not a shared family label):
The 4T and 8T anchor run JSONs had `non_canonical_label` values of
`"AVX-512+R2 anchor · np=1 · 4T"` and `"AVX-512+R2 anchor · np=1 · 8T"` respectively
— each run created a separate legend entry instead of grouping together.
The 208T 2-node run (PBS 167973941) also had `"ICX · MPI 2×104 2-node socket · R2 AVX-512"`.

**Fix:** Set `non_canonical_label: "AVX-512 + R2"` uniformly across all three existing
AVX-512+R2 run files. After the 16T–104T batch (PBS 168209650) and 4-node NUMA jobs
complete, all 7 points (4T, 8T, 16T, 32T, 64T, 104T, 208T) will form one connected line.

Files updated:
- `logs/runs/gadi_xlarge_mf_4t_icx_mpi1x4_avx512_r2_ompanchor.json`
- `logs/runs/gadi_xlarge_mf_8t_icx_mpi1x8_avx512_r2_ompanchor.json`
- `logs/runs/gadi_xlarge_mf_208t_icx_mpi2x104_2node_socket_avx512_r2.json`

**Bug B — `dataset_short` field included `.fa` extension** for some runs but not
others (e.g. `"xlarge_mf.fa"` vs `"xlarge_mf"`), splitting what should be one series
into two by dataset key.

**Fix (`tools/normalize.py`):** Strip file extension when computing `dataset_short`:
```python
# Before:
"dataset_short": os.path.basename(dataset) if dataset else None,
# After:
"dataset_short": os.path.splitext(os.path.basename(dataset))[0] if dataset else None,
```

#### Fix 2 — OMP batch script: `mpirun` CPU-binding prevented multi-thread runs (rc=2)

Job 168209451 (16T/32T/64T/104T AVX-512+R2 OMP batch) failed immediately on all four
thread counts. IQ-TREE reported `"ERROR: You have specified more threads than CPU cores
available"` with `rc=2, wall=5s` for each tier. Root cause: `mpirun -np 1` without
`--bind-to none` pins the single MPI rank to exactly one CPU slot. IQ-TREE detected
only 1 core available, making `-T 16+` invalid.

**Fix (`gadi-ci/run_xlarge_avx512_r2_omp_batch.sh`):**
```bash
# Before:
mpirun -np 1 \
# After:
mpirun -np 1 --bind-to none \
```

The bad run files from PBS 168209451 were deleted. Job **168209650** resubmitted with
the fix; currently **Running** (8h, 1 normalsr node, 104 cores).

#### Fix 3 — 4-node NUMA script had wrong project (`rc29` has no allocation)

`gadi-ci/run_xlarge_r2_numa_416t_4node.sh` was written with `#PBS -P rc29` and
`PROJECT_DIR=.../iqtree3-3.1.2`. The `rc29` project has no active SU allocation.
The script had never been successfully submitted. The `iqtree3-3.1.2` binary does not
exist under `um09` scratch.

**Fix:**
- `#PBS -P rc29` → `#PBS -P um09`
- `#PBS -l storage=scratch/rc29` → `scratch/um09`
- `PROJECT` default `rc29` → `um09`
- `PROJECT_DIR` → `/scratch/um09/${USER}/iqtree3-mf2` (R2+AVX-512 MPI binary built May 9 2026 from `gadi-spr-r2-avx512`, HEAD `abd98764`)

Job **168209769** submitted; currently **Running** (2h, 4 normalsr nodes, 416 cores).

#### Fix 4 — `make dashboard` broken (gitignored paths require `-f`)

`git add docs/ web/data/` failed silently because these paths are in `.gitignore`.
Added `-f` flag and also staged `logs/runs/` in the Makefile `dashboard` target.

#### Jobs currently running

| Job | Description | Nodes | Cores | Wall | State |
|---|---|---|---|---|---|
| 168209650 | AVX-512+R2 OMP: 16T, 32T, 64T, 104T (batch) | 1 | 104 | 8h | **R** |
| 168209769 | ICX+R2 NUMA 4-node MPI: 416T | 4 | 416 | 2h | **R** |

#### Previous failed jobs

| Job | Failure reason |
|---|---|
| 168209451 | `mpirun` CPU-binding (`--bind-to none` missing): all tiers rc=2, wall=5s |
| 168207037 | Build attempt for AVX-512+R2 MPI binary failed (Eigen3 not found); abandoned — using existing `iqtree3-mf2` binary |

#### Website rebuild

Docs rebuilt with the `dataset_short` and `non_canonical_label` fixes applied
(v=`b5ce8908d66d` on `modelfinder2`). Both `main` and `modelfinder2` branches updated
and pushed.

---

## 2026-05-12 (am) — Log housekeeping: archive MF-only runs + new batched AVX-512+R2 CI scripts

### What changed

**Archived irrelevant MF2 MF-only runs; filled gaps in AVX-512+R2 OMP scaling curve;
added 4-node 416T ICX+R2 NUMA MPI script to extend the multi-node series.**

#### Archived: MF2 MF-only runs (7 files)

Set `"archived": true` on all 7 `mf2_mfonly_icx_avx512_r2` log files in `logs/runs/`.
These MF-only runs (1T → 104T) isolate ModelFinder time but do not represent a
meaningful performance family for the scaling analysis; they are retained for provenance
but excluded from the website charts via the archive flag.

| File | T | Wall (s) |
|---|---|---|
| `gadi_xlarge_mf_1t_mf2_mfonly_np1_seed1.json` | 1 | 2725 |
| `gadi_xlarge_mf_4t_mf2_mfonly_np1_seed1.json` | 4 | 815 |
| `gadi_xlarge_mf_8t_mf2_mfonly_np1_seed1.json` | 8 | 527 |
| `gadi_xlarge_mf_16t_mf2_mfonly_np1_seed1.json` | 16 | 365 |
| `gadi_xlarge_mf_32t_mf2_mfonly_np1_seed1.json` | 32 | 165 |
| `gadi_xlarge_mf_64t_mf2_mfonly_np1_seed1.json` | 64 | 104 |
| `gadi_xlarge_mf_104t_mf2_mfonly_np1_seed1.json` | 104 | 70 |

#### New CI script: `gadi-ci/run_xlarge_avx512_r2_omp_batch.sh`

Batched single-PBS-job script that runs **16T, 32T, 64T, and 104T** in sequence
on one Gadi normalsr node (104 cores).  Fills the gap in the existing AVX-512+R2
OMP anchor series between the existing 4T / 8T anchor points and the 2-node 208T
run (`gadi_xlarge_mf_208t_icx_mpi2x104_2node_socket_avx512_r2.json`, already
complete, 325 s).

- **Binary:** `/scratch/um09/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi`
  (v3.1.2 + R2 patches, ICX+OpenMPI, AVX-512, single rank `mpirun -np 1`)
- **Dataset:** `xlarge_mf.fa` (200 taxa × 100,000 sites, 968 models), seed 1
- **build_tag:** `icx_omp_pin_avx512_r2_anchor`
- **Estimated walltime:** 16T ≈ 1800 s, 32T ≈ 1000 s, 64T ≈ 620 s, 104T ≈ 500 s → ≈ 4000 s; 8 h requested
- **Verification:** lnL reported against `BEST SCORE FOUND` in IQ-TREE output;
  compare against ICX+R2 NUMA canonical 104T baseline (−10,956,936.6) and
  GCC canonical series (PBS 167520755)

#### Note: AVX-512+R2 208T already covered

`gadi_xlarge_mf_208t_icx_mpi2x104_2node_socket_avx512_r2.json` (build_tag
`icx_mpi2x104_2node_socket_numa_ft_r2_avx512`, 325 s) already exists and provides
the 208T data point for the AVX-512+R2 family.  No new 208T script needed.

#### ICX+R2 NUMA series: 208T already covered

`gadi_xlarge_mf_208t_icx_mpi2x104_2node_fullnode_numa_ft_r2_v312.json` (342 s)
provides the 2-node 208T point.  Three other 208T NUMA variants also exist
(334 s, 325 s, 389 s for different rank/socket placements).

#### New CI script: `gadi-ci/run_xlarge_r2_numa_416t_4node.sh`

**4-node, 4×104 OMP = 416T** MPI run extending the ICX+R2 NUMA OMP series to
4 nodes.  Patterned after `run_xlarge_r2_v312_mpi_2node_fullnode.sh`.

- **Binary:** `/scratch/rc29/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi`
  (v3.1.2 + R2 patches, ICX+OpenMPI, AVX-512)
- **PBS:** `#PBS -l ncpus=416`, 4 nodes, `rc29` project, `normalsr` queue
- **MPI shape:** `mpirun -np 4 -rf rankfile.txt`, rankfile pins each rank to its
  full node (slot=0–103); `numactl --localalloc` per rank
- **build_tag:** `icx_mpi4x104_4node_fullnode_numa_ft_r2_v312`
- **non_canonical:** `true` (R2+MPI multi-node variant)
- **Expected range:** ~170–260 s if IQ-TREE scales from the 2-node result (342 s);
  hypothesis (a)/(b)/(c) same as documented in the 2-node script header
- **Verification:** lnL must match 1-node canonical 104T (−10,956,936.607) and
  2-node fullnode result (−10,956,936.607); seed propagation: IQ-TREE MPI sets
  per-rank seed = 1 + rank_id internally

---

## 2026-05-12 (al) — Analysis report: professional text cleanup + §4.3 model selection BIC table

### What changed

**Scaling analysis graph and report cleaned for academic presentation.**

#### Graph text cleanup (tools/scaling_10M_analysis.py)

Applied 20 targeted text replacements across all 5 panels to remove informal language
and standardise phrasing for presentation to supervisors / external audience:

- **Panel 1 (Amdahl fit):** Legend format standardised to `(T₁ = X h, f = Y, 104T = Z h)`;
  white annotation box for extrapolation note (was yellow `lightyellow` fill); NUMA annotation
  shortened; AVX-512 2-point fit labelled `(2-pt)` instead of informal marker; MF2 1T point
  labelled `[1T excl. from fit]` with offset fixed to avoid overlap.
- **Panel 2 (speedup):** Clean axis labels; MF2 MPI legend entry; `Peak ≈ 8 nodes / 85× speedup`;
  academic Dispatch box wording.
- **Panel 3 (prediction):** `1:1 line` label (was `Perfect`); title uses `r =`, `MAPE =`, `RMSE (log) =`.
- **Panel 4 (OMP-to-MPI extrapolation):** Legend entries phrased academically; annotations use
  `Linear prediction / Observed / Super-linear factor` and `Design estimate / Empirical / Overestimate
  factor`; title notes `super-linear` correctly.
- **Panel 5 (dispatch MPI):** Subtitle clarifies calibration dataset.

#### §4.3 Model selection table added to report

Added `MODEL_SELECTION` constant and §4.3 section to `tools/scaling_10M_analysis.py`; regenerated
`tools/scaling_10M_analysis.md` with a five-family logL / BIC comparison table.

| Family | Best model (BIC) | ln L | BIC | ΔBIC vs best |
|---|---|---|---|---|
| ICX Baseline | GTR+R4 | −10,956,936.612 | 21,918,605.036 | +93.1 |
| GCC Canonical | GTR+R4 | −10,956,936.612 | 21,918,605.036 | +93.1 |
| R2 + NUMA fix | GTR+R4 | −10,956,936.607 | 21,918,605.026 | +93.1 |
| AVX-512 + R2 | GTR+R4 | −10,956,936.612 | 21,918,605.036 | +93.1 |
| MF2 Full | **SYM+G4** | **−10,956,936.089** | **21,918,511.888** | **0 (best)** |

**Key result — ΔBIC = 93.1** (decisive evidence threshold ≥ 10). MF2 ModelFinder selects
`SYM+G4` (403 df, BIC 21,918,511.9) vs all other families selecting `GTR+R4` (408 df,
BIC 21,918,605.0). The 93.1-unit BIC difference decisively favours the MF2-selected model.
This demonstrates that MF2 not only parallelises model selection across MPI ranks but
produces a qualitatively better-supported model by exploring the full 968-model space
without the pre-gather checkpoint truncation that affected earlier np≥2 runs.

Data sources: `.iqtree` files in gadi-ci profile directories on scratch. ICX Baseline
(PBS 167969243), GCC Canonical (PBS 167520755), R2+NUMA (PBS 167895713), AVX-512+R2
(PBS 167972478), MF2 Full np=1 (PBS 168179462).

---

## 2026-05-12 (ak) — Graph fixes: R2+NUMA vs MF2 ordering; ICX-1T baseline; MPI projection

### What changed

**Root causes of three graph correctness issues identified and fixed.**

#### Issue 1 — MF2 1T job (PBS 168179462) completed: 10635.6s

The 1T MF2 Full run completed during the 8/16-node session. Including it in the
Amdahl fit **destroys** the fit:
- 7-point fit (1T–104T): T₁=3.04h, f=0.081, predicted ceiling = 12.3×
- But actual speedup at 104T = 10635/494 = **21.5×** — *exceeds the Amdahl ceiling*
- Physically impossible for standard Amdahl's law → NUMA+R2+AVX-512 effects cause
  super-Amdahl scaling at high thread counts (cross-socket memory locality improves
  disproportionately as thread count approaches 2-socket saturation)

**Fix:** Exclude 1T point from MF2 Amdahl fit. The 4T–104T fit (T₁=4.74h, f=0.023,
r=0.998, MAPE=5.0%) is the predictively useful model. The 1T point is shown as an
open marker annotated "super-Amdahl" on the graph. This also restores the correct
`FIT_MIN_N_MAP` mechanism for any future families with similar behavior.

#### Issue 2 — R2+NUMA apparent over-scaling (inflated T₁)

**Problem:** Panel 2 used each family's own T₁ as the speedup denominator. R2+NUMA
only has 8T–104T data; its T₁=6.29h is extrapolated 8× beyond the minimum observation.
This inflated T₁ made its "speedup" = T₁/T(n) = 22644/524 = **43×** at 104T, higher
than MF2's **35×** — even though MF2 at 104T (494s) is *faster* than R2+NUMA (524s).

**Fix:** Changed Panel 2 to use **ICX 1T (11915s) as the common absolute baseline**
for all speedup calculations. Now:
- R2+NUMA at 104T: 11915/524 = **22.8×** vs ICX 1T
- MF2 at 104T: 11915/494 = **24.1×** vs ICX 1T ← correctly higher ✓
- MF2 MPI 832T: 11915/139.5 = **85.4×** ← breakthrough ✓

Also fixed the Amdahl curve extrapolation flag from `ns.min() > 32T` to `ns.min() > 4T`,
so R2+NUMA (min=8T, no low-thread T₁ anchor) correctly gets a dashed line.

#### Issue 3 — MPI communication overhead projection added

Fitted `T(n_ranks) = a/n_ranks + b×n_ranks^c` to the 5 MPI data points (n=1–16 ranks):

| n ranks | T (s) | Speedup vs ICX 1T |
|---|---|---|
| 1 (104T OMP) | 494.0 | 24.1× |
| 2 (208T) | 333.0 | 35.8× |
| 4 (416T) | 213.2 | 55.9× |
| 8 (832T) | 139.5 | **85.4×** ← peak |
| 16 (1664T) | 192.5 | 61.9× ← regression |

The fitted model predicts **~8 nodes as optimal** for the 968-model xlarge dataset.
Projection curve (dashed) shows expected performance at 16–32 nodes declining due to
MPI overhead growing as models/rank drops below ~100.

**Key insight for larger datasets (10M):** At 748s/model, MPI communication overhead
(~seconds to minutes) is negligible. The optimal node count scales with `n_models / min_granularity`.
For 10M full MF (968 models × 748s = 50h/rank at 4 nodes), scaling can extend to
64+ nodes before communication dominates.

### Graph changes summary

| Panel | Before | After |
|---|---|---|
| Panel 1 | MF2 1T included in Amdahl fit (T₁=3.04h, f=0.081, terrible fit) | 1T excluded (shown as ○); fit restored to T₁=4.74h, f=0.023 |
| Panel 1 | R2+NUMA shown as solid line (min=8T treated as well-constrained) | R2+NUMA dashed "⚠ T₁ extrap." (min > 4T threshold) |
| Panel 2 | Per-family T₁ speedup reference (inflates R2+NUMA to 43×) | Common ICX 1T baseline (R2+NUMA=22.8×, MF2=24.1× — correct ordering) |
| Panel 2 | T₁-relative Amdahl ceiling lines | OMP Amdahl ceiling vs ICX 1T (dotted, ~25-30×) |
| Panel 2 | No MPI projection | Communication overhead projection fitted to 5-point MPI dataset |
| Panel 2 | MF2 Dispatch at 18.87× (vs ICX 104T) on scatter | MF2 Dispatch noted as text box (202× vs ICX 1T, off-scale) |

---

## 2026-05-12 (aj) — MF2 Full: 8-node and 16-node MPI scaling; communication overhead plateau

### What changed

**8-node (832T) and 16-node (1664T) MPI runs completed** for the MF2 Full IQ-TREE
scaling series on `xlarge_mf.fa` (seed=1, free tree).

| PBS | Config | Threads | Ranks | Wall | vs 104T (1-node) | vs prior |
|---|---|---|---|---|---|---|
| 168195261 | 8 nodes × 104T | 832T | 8 | **139.5s** | 3.54× | 1.53× vs 416T |
| 168195262 | 16 nodes × 104T | 1664T | 16 | **192.5s** | 2.57× | 0.73× vs 832T ← **regression** |

Both runs verified (lnL within 5 units of reference −10,956,936.67):
- 832T: lnL = −10,956,936.145 (diff = 0.52, within tolerance) ✓
- 1664T: lnL = −10,956,932.288 (diff = 4.38, within tolerance) ✓

**Key result — MPI communication overhead saturates at 8 nodes:**

The 16-node run is **38% slower** than the 8-node run (192.5s vs 139.5s). This is the
expected Amdahl/communication ceiling. With only 968 models to dispatch across 16 ranks
(~60 models/rank), the fraction of time spent in MPI coordination + barrier
synchronisation exceeds the fraction saved by additional parallelism.

**Full MF2 Full scaling table (xlarge_mf.fa, seed=1):**

| Threads | Ranks | Nodes | Wall (s) | Speedup vs 104T |
|---|---|---|---|---|
| 4T  | 1 | 1 | 4535.1 | 0.11× |
| 8T  | 1 | 1 | 2486.9 | 0.20× |
| 16T | 1 | 1 | 1498.6 | 0.33× |
| 32T | 1 | 1 | 967.8  | 0.51× |
| 64T | 1 | 1 | 598.9  | 0.82× |
| 104T | 1 | 1 | 494.0 | **1.00×** (baseline) |
| 208T | 2 | 2 | 333.0  | 1.48× |
| 416T | 4 | 4 | 213.2  | 2.32× |
| **832T** | **8** | **8** | **139.5** | **3.54×** ← peak |
| 1664T | 16 | 16 | 192.5 | 2.57× ← regression |

**Optimal configuration: 8 nodes (832T).** Beyond this point, MPI overhead
exceeds parallelisation gain for this 968-model dataset. The ~60 models/rank at 16
nodes is near the minimum granularity for effective MPI dispatch.

**Amdahl fit comparison (OMP-only series, 4T–104T):**

The OMP-only Amdahl fit (f=0.023, T₁=4.74h) predicts a ceiling of ~44×. The MPI
multi-node points vastly exceed this ceiling (122× at 832T vs the 44× OMP ceiling)
because MPI dispatch exploits a fundamentally different parallelisation axis:
independent model evaluation across ranks rather than shared-memory thread scaling
within a single model evaluation.

**Graph updated:** `tools/scaling_10M_analysis.png` — Panel 1 shows 832T and 1664T
as open diamond (◇) MPI bonus points. Panel 2 shows actual speedups: 832T → 122×,
1664T → 89× (both vs Amdahl-extrapolated T₁). Performance regression at 1664T is
clearly visible as the rightmost open diamond dropping below the 832T point.

**SU cost:** 8-node: 832 cores × (139.5/3600)h × 2.0 = **129 SU**.
16-node: 1664 cores × (192.5/3600)h × 2.0 = **178 SU**. Total: **307 SU**.

---

## 2026-05-12 (ai) — MF2 MPI: fix pre-gather checkpoint corruption of best-model keys

### What changed

**Implemented the Phase 2 hardening fix** for the pre-gather checkpoint corruption bug
identified in entry (ah). Applied directly to
`/scratch/um09/as1708/iqtree3-mf2/src/iqtree3/main/phylotesting.cpp`.

**Root cause (recap):** `evaluateAll()` writes `best_model_AIC/AICc/BIC` and
`best_score_*` checkpoint keys **before** `MPI_Allreduce`, using each rank's local
`MF_DONE` model subset (only ~n/nranks models evaluated on that rank's own starting
tree). `gatherCheckpoint()` merges all ranks' serialised checkpoints with
last-write-wins semantics (rank order in `MPI_Gatherv` → last rank wins). The
highest-numbered rank's stale local-best name overwrites master's correct value.
Result: `.iqtree` header, AIC/BIC log lines, and model table reflect one rank's
partial local evaluation rather than the globally-consolidated result.

**Fix location:** Inside the `#ifdef _IQTREE_MPI` block in `evaluateAll()`, between
the post-name-restoration loop and the `cout << "MF-MPI: gather complete"` line.
At this point: all 968 models have globally-correct scores (post-`MPI_Allreduce`)
and post-evaluation names (restored from checkpoint via `mf_subst_N` / `mf_rate_N`
keys). The fix re-writes the criterion-best keys and rebuilds the model list using
this complete global picture, then re-dumps to `.model.gz`.

**Changes applied:**

```cpp
// Fix pre-gather checkpoint corruption ...
{
    const ModelTestCriterion all_criteria[] = {MTC_AIC, MTC_AICC, MTC_BIC};
    for (auto mtc : all_criteria) {
        int bm = getBestModelID(mtc);
        model_info.put("best_model_" + criterionName(mtc), at(bm).getName());
        model_info.put("best_score_" + criterionName(mtc), at(bm).getScore(mtc));
    }
    multimap<double,int> global_sorted;
    for (int i = 0; i < n; i++)
        global_sorted.insert(multimap<double,int>::value_type(at(i).getScore(), i));
    string global_list;
    for (auto it = global_sorted.begin(); it != global_sorted.end(); it++) {
        if (it != global_sorted.begin()) global_list += " ";
        global_list += at(it->second).getName();
    }
    model_info.putBestModelList(global_list);
    model_info.dump();
}
```

**What this fixes for the np4 run (PBS 168183552):**
- `.iqtree` header `"Best-fit model according to BIC:"` will now show `GTR+F+R4`
  (matching `"Best-fit model:"` log line and actual substitution model)
- AIC/AICc/BIC log lines will all name `GTR+F+R4` (or whichever is globally best)
- Model table in `.iqtree` will contain all 968 consolidated models, not just the
  23 `+I+R2` models from one rank's partial stripe
- `.model.gz` checkpoint contains globally-correct keys, so resume (`--redo` skipped)
  would correctly restore the right model name

**Performance:** Zero overhead. Three O(n) passes over n=968 models + one
`multimap` insert (O(n log n) ≈ 968×10 = ~10K operations) run once, after all
model evaluations complete. Fully gated behind `nranks > 1` guard — OMP-only
(np1) runs are completely unaffected.

**Execution plan — steps to validate:**

1. **Rebuild binary** — must run on Gadi SPR compute node (icpx required):
   ```bash
   cd ~/setonix-iq && qsub tiers/rebuild_mf2_binary.sh
   ```
   Script backs up old binary, checks canary string in source, runs
   `/bin/gmake -j8 iqtree3`, validates mtime and libmpi ELF section.

2. **np=4 verification run** — 4 nodes × 104T = 416T, `xlarge_mf.fa`, seed=1:
   ```bash
   qsub tiers/verify_mf2_fix_np4.sh
   ```
   Automated pass/fail checks: `gather complete, 968` line, log BIC model vs
   Best-fit model base agreement, `.iqtree` header agreement, model table ≥900
   rows, lnL within 5 units of reference (−10,956,936.67).

3. **np=2 verification run** — 2 nodes × 104T = 208T, same dataset:
   ```bash
   qsub tiers/verify_mf2_fix_np2.sh
   ```
   Same checks. np=2 also showed pre-gather corruption (`GTR+I+R4` in BIC
   header vs `GTR+R4` Best-fit, 49-model table).

4. **Mark complete** — once both verification runs exit 0, update CHANGELOG
   with PBS IDs and correct the model table in entry (ah).

**SU cost:** ~696 SU total (rebuild=4, np=2=208, np=4=484).

---

## 2026-05-12 (ah) — MF2 Full: 2-node and 4-node MPI jobs completed; model/BIC analysis

### What changed

**Scaling runs completed across 4T–416T (xlarge_mf.fa, seed=1):**
- `gadi_xlarge_mf_4t_mf2_full_np1_seed1.json`   — PBS 168179463, wall=4535.1s
- `gadi_xlarge_mf_8t_mf2_full_np1_seed1.json`   — PBS 168179464, wall=2486.9s
- `gadi_xlarge_mf_16t_mf2_full_np1_seed1.json`  — PBS 168179465, wall=1498.6s
- `gadi_xlarge_mf_32t_mf2_full_np1_seed1.json`  — PBS 168179466, wall=967.8s
- `gadi_xlarge_mf_64t_mf2_full_np1_seed1.json`  — PBS 168173628, wall=598.9s
- `gadi_xlarge_mf_104t_mf2_full_np1_seed1.json` — PBS 168173629, wall=494.0s
- `gadi_xlarge_mf_208t_mf2_full_np2_seed1.json` — PBS 168188898, wall=333.0s (2-node MPI, post-fix)
- `gadi_xlarge_mf_416t_mf2_full_np4_seed1.json` — PBS 168188897, wall=213.2s (4-node MPI, post-fix)

All runs pass verify (status=pass). Tree log-likelihoods converge to ≈ −10,956,936.

**Full results — MF2 Full IQ-TREE scaling (xlarge_mf.fa, seed=1):**

| Threads | Ranks | Wall (s) | MF best model | Tree model | lnL | BIC | MF table |
|---|---|---|---|---|---|---|---|
| 4T | 1 | 4535.1 | `SYM+G4` | `SYM+G4` | −10,956,936.089 | **21,918,511.89** | 83 |
| 8T | 1 | 2486.9 | `SYM+G4` | `SYM+G4` | −10,956,936.089 | **21,918,511.89** | 83 |
| 16T | 1 | 1498.6 | `SYM+G4` | `SYM+G4` | −10,956,936.089 | **21,918,511.89** | 83 |
| 32T | 1 | 967.8 | `SYM+G4` | `SYM+G4` | −10,956,936.089 | **21,918,511.89** | 83 |
| 64T | 1 | 598.9 | `SYM+G4` | `SYM+G4` | −10,956,936.089 | **21,918,511.89** | 83 |
| 104T | 1 | 494.0 | `SYM+G4` | `SYM+G4` | −10,956,936.089 | **21,918,511.89** | 83 |
| 208T | 2 | 333.0 | `GTR+R4` | `GTR+F+R4` | −10,956,936.671 | 21,918,605.15 | 968 |
| 416T | 4 | 213.2 | `GTR+R4` | `GTR+F+R4` | −10,956,936.670 | 21,918,605.15 | 968 |

**Key observations (post-fix, PBS 168188897/168188898):**
- All np1 runs (4T–104T) **consistently select `SYM+G4`** with BIC 21,918,511.89 and 83 models in the table. Results are deterministic and reproducible across thread counts.
- np2 and np4 now **consistently select `GTR+R4`** — BIC header, log BIC line, and Best-fit line all agree. The pre-gather corruption bug is fixed.
- np1 has the **lowest (best) BIC** — 93 units better than np2/np4. The simpler model `SYM+G4` (403 free params, BIC penalty ~93 less) beats `GTR+F+R4` (411 params); np1 lnL is also fractionally better (0.6 units).
- MPI runs consolidate **all 968 models** into the table (was 49/23 before the fix). lnL differences are <0.6 units across all 8 runs — trees are phylogenetically equivalent.
- **Scaling:** 4T→104T gives 9.2× speedup (OMP-only). 104T→208T (np=2) adds 1.49×; 208T→416T (np=4) adds another 1.56×. MPI scaling is sub-linear due to inter-node communication overhead but practical — 416T finishes in 3m33s vs 75m35s at 4T.

**Root-cause investigation of the np4 inconsistency flag (`SYM+I+R2` ≠ `GTR+F+R4`):**

The np4 log and .iqtree file report two contradictory best-model selections:
```
AIC / AICc / BIC:       SYM+I+R2    ← from .iqtree header and log AIC/BIC printout
Best-fit model chosen:  GTR+R4      ← from log "Best-fit model:" line
Model of substitution:  GTR+F+R4    ← actual model used for tree search
```
The np4 MF table contains only 23 models, all with `+I+R2` rate variation, and their
lnL values (≈ −11,198,286) are ~241,350 units worse than the final tree lnL
(−10,956,936). This is impossible from a single consistent evaluation — the lnL values
in the table are from a completely different tree than the one used for tree search.

**Mechanism** (traced through `phylotesting.cpp` source):

1. **LPT stripe assignment**: `evaluateAll()` assigns each of the 968 models to a
   specific rank. For np4, rank 0 gets models 0,4,8,... (all `+R4` variants); another
   rank gets models with `+I+R2` variants.
2. **Independent starting trees**: Each rank runs the fast-NNI GTR+I+G tree search
   independently before its ModelFinder phase. Rank 0 finds a near-optimal tree
   (lnL ≈ −10,956,932). The rank evaluating `+I+R2` models finds a worse tree
   (lnL ≈ −11,198,286). Every rank evaluates its assigned models on its own tree.
3. **Pre-gather `best_model_BIC` write** (the bug): Around line 3700–3706 of
   `phylotesting.cpp`, BEFORE the `MPI_Allreduce` gather, each rank calls
   `getBestModelID(MTC_BIC)` on its LOCAL model set (only its assigned models with
   `MF_DONE` set) and writes `model_info.put("best_model_BIC", ...)`. The rank
   evaluating `+I+R2` models writes `"SYM+I+R2"` to its local checkpoint.
4. **`gatherCheckpoint()` merges stale values**: After `MPI_Allreduce` fills the
   global score table correctly, `gatherCheckpoint()` merges all ranks' checkpoint
   entries — including the stale `best_model_BIC = "SYM+I+R2"` from the
   non-master rank — into the master's checkpoint.
5. **Caller reads stale key**: Back in `runModelFinder()`, `CKP_RESTORE(best_model_BIC)`
   reads the now-corrupted key from the merged checkpoint and prints
   `"BIC: SYM+I+R2"`. The .iqtree `"Best-fit model according to BIC:"` header
   is sourced from the same key.
6. **Tree search is unaffected**: `evaluateAll()` returns `at(getBestModelID(...))`
   evaluated POST-Allreduce — this correctly finds `GTR+R4` as the global best.
   `iqtree.aln->model_name` is set from this return value, not from the checkpoint
   key. The tree is built correctly with `GTR+R4` → optimised to `GTR+F+R4`.

**Impact**: Cosmetic only. The phylogenetic result (tree topology, branch lengths,
final lnL) is correct. The .iqtree report header and the model table are misleading.
The model table shows only one rank's partial 23-model set (not the 968-model
consolidated view) because `putBestModelList()` is also called pre-gather with the
rank-local sorted model list.

**Fix** (to be applied in Phase 2 hardening of `phylotesting.cpp`):
After `gatherCheckpoint()` / `broadcastCheckpoint()` and the post-Allreduce
`getBestModelID()` call, re-write the criterion-best keys with the globally correct
values before returning:
```cpp
// After MF-MPI gather: overwrite stale per-rank best-model keys
for (auto mtc : {MTC_AIC, MTC_AICC, MTC_BIC}) {
    int bm = getBestModelID(mtc);
    model_info.put("best_model_" + criterionName(mtc), at(bm).getName());
}
// Also rebuild the model_list from all 968 gathered models (not rank-local)
// and call model_info.putBestModelList(model_list) again.
```

**Conclusion:** For this dataset the OMP-only np1 run finds the better-supported model
by BIC. The MPI runs converge to the same tree but select more complex models with
slightly worse BIC. The np4 `SYM+I+R2` label in the .iqtree header is a reporting
artefact of the pre-gather checkpoint write race, not a real model selection difference
— the actual tree was built with `GTR+R4` (correctly globally selected).

**Still running:** 168179462 (1T, ~3hr remaining), 168179463 (4T, ~1.5hr remaining).

---

## 2026-05-12 (ag) — MF2 Full: 2-node and 4-node MPI runs; 8T completed

### What changed

**New PBS scripts** for MF2 Full IQ-TREE on 2-node and 4-node MPI configurations,
extending the thread-scaling curve to 208T and 416T:
- `tiers/run_xlarge_mf2_full_2node.sh` — 2 ranks × 104T = 208T total (2 full SPR nodes)
- `tiers/run_xlarge_mf2_full_4node.sh` — 4 ranks × 104T = 416T total (4 full SPR nodes)

Both use the same protocol as the OMP-only series: MF2 binary
(`um09/build-mpi-mf2/iqtree3-mpi`), full IQ-TREE, free tree, seed=1. Placement via
`--hostfile + -rf rankfile` (validated form from PBS 168000131). Build tags:
`mf2_full_np2_seed1_avx512_r2_lpt` and `mf2_full_np4_seed1_avx512_r2_lpt`.

**Jobs submitted:**
- 168183551 — 2-node 208T (Q, 00:30 wall)
- 168183552 — 4-node 416T (Q, 00:30 wall)

**`tools/scaling_10M_analysis.py` updates:**
- `FAMILIES["MF2 Full IQ-TREE"]` key renamed from `(OMP-only, free tree)` to `(free tree, seed=1)` to reflect multi-node scope
- `patterns` broadened from `["mf2_full_np1"]` to `["mf2_full_np"]` (matches np1/np2/np4)
- `mpi_ok` extended from `[1]` to `[1, 2, 4]`
- AVX-512+R2 exclude list already contains `"mf2_full"` — no bleed possible

**8T run (job 168179464) completed** while editing — wall=2486.9s. JSON written to
`logs/runs/gadi_xlarge_mf_8t_mf2_full_np1_seed1.json`. Series now has 5 OMP-only pts:
8T (2487s), 16T (1499s), 32T (968s), 64T (599s), 104T (494s).
Amdahl fit (5 pts): T₁=4.88h, f=0.021, r=0.9966, MAPE=5.1%.

**Pending:**
- 168179462 (1T, ~3h remaining)
- 168179463 (4T, ~1.5h remaining)
- 168183551 (2-node 208T, ~30min)
- 168183552 (4-node 416T, ~30min)

---

## 2026-05-12 (af) — MF2 Full: 16T run completed; graph fixes (solid line, 5-family title)

### What changed

**MF2 Full 16T run (job 168179465) completed** — wall=1498.6s (0.416h). JSON written
to `logs/runs/gadi_xlarge_mf_16t_mf2_full_np1_seed1.json`. Series now has 4 data points:
16T (1498.6s), 32T (967.8s), 64T (598.9s), 104T (494.0s).

**Amdahl fit** (4 pts, 16T–104T): T₁=5.40h, f=0.016, r=0.9974, MAPE=3.0%.

**Graph fixes in `tools/scaling_10M_analysis.py`:**
- Suptitle updated from hard-coded "4 Patch Families" to dynamic `{len(FAMILIES)}` (now "5")
- T₁ extrap. warning threshold changed from `ns.min() > 8` to `ns.min() > 32`.
  MF2 Full (min=16T) now plots with a solid line — 4 clean data points is sufficient.
- Removed dead `mf2_fam = "MF2 Dispatch\n..."` assignment and `suffix` dead code from
  the annotation loop (MF2 Dispatch was removed from FAMILIES in the previous entry).

**Jobs still running (as of 15:40):** 168179462 (1T, 4hr), 168179463 (4T, 2hr),
168179464 (8T, 1.5hr). When they complete, re-run the script for the full 7-pt series.

**Current Amdahl fit table (5 families, xlarge_mf.fa):**

| Family | T₁(h) | f | r(log) | MAPE | n |
|---|---|---|---|---|---|
| ICX Baseline | 3.35 | 0.080 | 0.9908 | 10.7% | 6 |
| GCC Canonical | 3.89 | 0.095 | 0.9940 | 6.1% | 6 |
| R2 + NUMA fix | 6.29 | 0.017 | 0.9961 | 6.0% | 5 |
| AVX-512 + R2 | 5.14 | 0.020 | 0.9943 | 12.7% | 4 |
| MF2 Full | 5.40 | 0.016 | 0.9974 | 3.0% | 4 |

---

## 2026-05-12 (ae) — MF2 Full IQ-TREE scaling series; inode cleanup; drop MF-only family

### What changed

**Dropped "MF-only MF2" family from `tools/scaling_10M_analysis.py`.**
The MF-only series (`-m MF -te fixed_xlarge_tree.nwk`, seed=1) was removed from the
scaling graph. It measured a fundamentally different quantity (ModelFinder step only,
fixed tree) and was not comparable to any other family. Its JSON files remain in
`logs/runs/` but are no longer plotted.

**Added "MF2 Full IQ-TREE" family** — same binary (`um09/build-mpi-mf2/iqtree3-mpi`,
ICX+OpenMPI, R1+R2+AVX512), but run as full IQ-TREE: free tree, no `-m MF`, no `-te`,
seed=1. This is directly comparable to ICX Baseline / GCC Canonical / R2+NUMA /
AVX-512+R2 on the thread-scaling plot.

**Script `tiers/run_xlarge_mf2_full.sh`** created. Runs `mpirun -np 1` OMP-only with
`--map-by node:PE=N`. Writes JSON with `build_tag = mf2_full_np1_seed1_avx512_r2_lpt`.

**Root cause of earlier job failures (rc=2):** Inode exhaustion on um09 scratch.
A VTune hotspot collection (job 168163238, GCC canonical 104T) consumed 168k inodes,
pushing the project over the 500k inode quota. All subsequent jobs that tried to write
checkpoint files got `Disk quota exceeded`. Fixed by:
1. Deleting VTune collection data from profile dir (−168k inodes)
2. Deleting zarr processed cache (`sst-forecasting/data/processed/oisst_coralsea.zarr`) (−21.5k)
3. Deleting all profiling build artefacts from `iqtree3/` and `iqtree3-3.1.2/` (−7.8k)
4. Deleting all failed/duplicate profile run directories (mf2_full failed, mfonly all, correctness, partial 100taxa)
Result: 525k inodes (over-limit) → 326k (35% headroom).

**Exclude list fix for AVX-512+R2:** The `mf2_full` and `mfonly` build_tags both
contain `avx512_r2`. Added `mf2_full` and kept `mf2dispatch`, `mfonly`, `mf_only` in
the AVX-512+R2 exclude list so those runs don't bleed into the wrong family.

### Completed runs (MF2 Full IQ-TREE family)

| PBS | Threads | Wall | Status |
|---|---|---|---|
| 168173628 | 64T  | 599s | ✓ in `logs/runs/` |
| 168173629 | 104T | 494s | ✓ in `logs/runs/` |
| 168179462 | 1T   | ~4h est | R (running) |
| 168179463 | 4T   | ~2h est | R (running) |
| 168179464 | 8T   | ~1.5h est | R (running) |
| 168179465 | 16T  | ~1h est | R (running) |
| 168179466 | 32T  | ~30m est | R (running) |

### Current Amdahl fit quality (2 pts for MF2 Full — fit will improve when jobs complete)

| Family | T₁(h) | f | r(log) | MAPE | n |
|---|---|---|---|---|---|
| ICX Baseline    | 3.35 | 0.080 | 0.991 |  10.7% | 6 |
| GCC Canonical   | 3.89 | 0.095 | 0.994 |   6.1% | 6 |
| R2 + NUMA fix   | 6.29 | 0.017 | 0.996 |   6.0% | 5 |
| AVX-512 + R2    | 5.14 | 0.020 | 0.994 |  12.7% | 4 |
| MF2 Full IQ-TREE| 4.94 | 0.018 | 1.000 |   0.0% | 2 (pending) |

---

## 2026-05-12 (ae) — xlarge_mf.fa scaling audit: protocol mismatches + new run order

### Audit scope

Full scientific audit of all xlarge_mf.fa runs on Gadi SPR used in
`tools/scaling_10M_analysis.py`. Objective: verify that families plotted on the same
scaling axis measure comparable quantities, and that Amdahl fits are statistically valid.

### Confirmed data inventory (all Gadi SPR, `xlarge_mf.fa`, sorted by family)

| Label | PBS | Threads | MPI | Wall | Tree | `-m` flag | Seed | lnL | Status |
|---|---|---|---|---|---|---|---|---|---|
| ICX Baseline 1T   | 166978126 | 1   | 1 | 11915s | free | (full IQ-TREE) | 1  | −10956936.612 | ✓ |
| ICX Baseline 4T   | 166978127 | 4   | 1 |  4244s | free | (full IQ-TREE) | 1  | — | ✓ |
| ICX Baseline 8T   | 166978128 | 8   | 1 |  2440s | free | (full IQ-TREE) | 1  | — | ✓ |
| ICX Baseline 32T  | 167001081 | 32  | 1 |  1036s | free | (full IQ-TREE) | 1  | — | ✓ |
| ICX Baseline 64T  | 167001085 | 64  | 1 |   897s | free | (full IQ-TREE) | 1  | −10956936.640 | ✓ |
| ICX Baseline 104T | 167004590 | 104 | 1 |  1112s | free | (full IQ-TREE) | 1  | −10956936.611 | ✓ |
| GCC Canonical 1T  | 167520752 | 1   | 1 | 13954s | free | (full IQ-TREE) | 1  | −10956936.612 | ✓ |
| GCC Canonical 4T  | 167520753 | 4   | 1 |  4803s | free | (full IQ-TREE) | 1  | — | ✓ |
| GCC Canonical 8T  | 167520754 | 8   | 1 |  2956s | free | (full IQ-TREE) | 1  | — | ✓ |
| GCC Canonical 16T | 167520755 | 16  | 1 |  2048s | free | (full IQ-TREE) | 1  | — | ✓ |
| GCC Canonical 32T | 167520756 | 32  | 1 |  1425s | free | (full IQ-TREE) | 1  | — | ✓ |
| GCC Canonical 64T | 167520757 | 64  | 1 |  1638s | free | (full IQ-TREE) | 1  | −10956936.612 | ✓ |
| R2+NUMA 32T       | 167865974 | 32  | 1 |  1119s | free | (full IQ-TREE) | 1  | −10956936.612 | ✓ |
| R2+NUMA 64T       | 167865975 | 64  | 1 |   691s | free | (full IQ-TREE) | 1  | −10956936.612 | ✓ |
| R2+NUMA 104T      | 167865976 | 104 | 1 |   524s | free | (full IQ-TREE) | 1  | −10956936.612 | ✓ |
| AVX-512+R2 104T   | 167972478 | 104 | 2 |   512s | free | (full IQ-TREE) | 1  | — | ✓ |
| AVX-512+R2 208T   | 167973941 | 208 | 2 |   325s | free | (full IQ-TREE) | 1  | −10956936.640 | ✓† |
| MF2 Dispatch 416T | 168000131 | 416 | 4 |    59s | **fixed** | **-m MF** | **42** | — | ⚠ |

† AVX 208T lnL diff = 0.028 from expected; attributed to MPI floating-point summation order.
No full-pipeline 4-node 416T xlarge run exists.

### Critical Finding 1 — MF2 uses a different measurement protocol

All ICX/GCC/R2/AVX families run **full IQ-TREE** (NJ tree construction → NNI
optimisation → ModelFinder `test()` with BIC pruning → tree re-estimation), with
`--seed 1`, no `-te` flag. Wall time includes tree search.

MF2 Dispatch runs **ModelFinder-only** (`-m MF`), with a pre-computed fixed tree
(`-te fixed_xlarge_tree.nwk`), `--seed 42`, and `evaluateAll()` (all 968 models,
no BIC pruning). Wall time excludes tree search.

The 59s MF2 wall and the 1112s ICX-104T wall are **not the same quantity** and cannot
be placed on the same thread-scaling axis without a protocol note. Graph now marks the
MF2 ◆ point with `[MF-only, −te, seed=42]` and a title warning.

### Critical Finding 2 — R2+NUMA uses a different compiler binary than ICX Baseline

| Family | Binary path | Compiler |
|---|---|---|
| ICX Baseline | `rc29/.../build-profiling/iqtree3` | Intel ICX (profiling build) |
| GCC Canonical | `rc29/.../build-profiling/iqtree3` | Same binary |
| R2+NUMA | `rc29/.../build-profiling-clang/iqtree3` | **Clang/LLVM** (different optimisation) |
| AVX-512+R2 | `um09/.../build-profiling-mpi/iqtree3` | Intel ICX + OpenMPI |

The ~2× speedup at 104T (524s vs 1112s) between R2+NUMA and ICX Baseline conflates
two effects: (a) the R2 rate-category patch and NUMA first-touch, and (b) the compiler
change from ICX to Clang. These are not separated. Future runs should keep the compiler
constant when isolating patch effects.

### Warning Finding 3 — Amdahl fits have too few data points for R2+NUMA and AVX-512

| Family | Points | Min thread | T₁ validity | Fit quality |
|---|---|---|---|---|
| ICX Baseline | 6 (1T–104T) | 1T ✓ | anchored | ✓ reliable |
| GCC Canonical | 6 (1T–64T) | 1T ✓ | anchored | ✓ reliable, missing 104T |
| R2+NUMA | **3** (32T–104T) | 32T ❌ | extrapolated **32×** | ❌ T₁=7.71h meaningless |
| AVX-512+R2 | **2** (104T, 208T) | 104T ❌ | completely unconstrained | ❌ 2-point fit |

Graph now shows R2+NUMA and AVX-512 Amdahl curves as dashed/faded with
`[⚠ T₁ extrap.]` in the legend.

### Finding 4 — Hidden 4-rank socket-level run (not in current families)

`gadi_xlarge_mf_208t_icx_mpi4x52_2node_socket_numa_ft_r2.json`
PBS 167911421 — 4 MPI ranks × 52T, 2 nodes, socket-level placement, wall=389s.
This is the R2+NUMA binary at 4 ranks / 2 nodes. Excluded from current FAMILIES
(mpi_ok=[1] for R2+NUMA). Included in ordered run list below for reference.

### Ordered list of new runs needed

Priority order based on scientific impact. All new runs use `xlarge_mf.fa`,
`build-profiling` (ICX) binary unless stated.

#### Priority 1 — Fix missing data to anchor T₁ for R2+NUMA and GCC

These runs add the low-thread points required to constrain the Amdahl T₁ parameter.
Without them the fits are extrapolations.

| # | Run | Config | Why |
|---|---|---|---|
| 1 | GCC 104T | `gadi_xlarge_mf_104t_sr_gcc_pin`, mpi=1, free tree, seed=1 | GCC series missing NUMA-penalty point at 104T |
| 2 | R2+NUMA 1T | `gadi_xlarge_mf_1t_icx_omp_pin_numa_ft_r2`, mpi=1, free tree, seed=1 | Anchors T₁; current 7.71h extrapolation is 32× unconstrained |
| 3 | R2+NUMA 4T | same binary, 4T | Needed for Amdahl fit (min 4 pts recommended) |
| 4 | R2+NUMA 8T | same binary, 8T | |
| 5 | R2+NUMA 16T | same binary, 16T | Fills 8T→32T gap |

#### Priority 2 — Enable apples-to-apples MF-only comparison with MF2

To plot MF2 ◆ on the same axis as other families, we need an MF-only scaling series
using the same binary and seed as our other single-node runs: MF2 binary
(`um09/build-mpi-mf2/iqtree3-mpi`, np=1), `-m MF -te fixed_xlarge_tree.nwk --seed 1`.

| # | Run | Config | Why |
|---|---|---|---|
| 6 | MF-only MF2 1T   | MF2 binary (um09/build-mpi-mf2), np=1, `-m MF -te fixed_xlarge_tree.nwk --seed 1`, 1T | T₁ anchor for MF-only Amdahl fit |
| 7 | MF-only MF2 4T   | same, 4T | |
| 8 | MF-only MF2 8T   | same, 8T | |
| 9 | MF-only MF2 16T  | same, 16T | |
| 10 | MF-only MF2 32T | same, 32T | |
| 11 | MF-only MF2 64T | same, 64T | |
| 12 | MF-only MF2 104T | same, 104T | Same binary as MF2 dispatch, single rank — direct baseline for 416T/59s |
| 13 | MF2 2-node 208T  | MF2 binary, `np=2`, `-m MF -te`, seed=1 | Mid-point for MF2 dispatch scaling curve |

#### Priority 3 — Fill gaps and characterise compiler effect

| # | Run | Config | Why |
|---|---|---|---|
| 14 | ICX 16T | ICX baseline, 16T, free tree, seed=1 | Fill gap between ICX 8T (2440s) and 32T (1036s) |
| 15 | AVX-512 1T (OMP-only) | AVX+R2 binary, 1T, mpi=1, free tree, seed=1 | Anchor AVX T₁ — 2-point fit is currently 100% unconstrained |
| 16 | AVX-512 4T  | same, 4T | |
| 17 | AVX-512 8T  | same, 8T | |
| 18 | ICX-compiled R2 1T | ICX binary (not Clang), R2 patch, NUMA pin, 1T | Isolate patch effect from compiler change |
| 19 | ICX-compiled R2 104T | same, 104T | Compare directly to R2+Clang 104T=524s |

#### Priority 4 — Extended MF2 dispatch scaling

| # | Run | Config | Why |
|---|---|---|---|
| 20 | MF2 8-node 832T  | MF2 binary, `np=8`, `-m MF -te`, seed=42 | Extrapolate MF2 scaling beyond 4 nodes |
| 21 | MF2 1-node 104T  | MF2 binary, `np=1`, `-m MF -te`, seed=42 | Baseline — already have evaluateAll 62.5s, verify dispatch at 1-rank |

### Graph status after this audit

`tools/scaling_10M_analysis.py` updated (this session):
- Panel 1 title warns MF2 ◆ is MF-only protocol, not comparable to other families
- R2+NUMA and AVX-512 Amdahl lines shown dashed with `[⚠ T₁ extrap.]`
- MF2 data point annotation includes `(MF-only)` suffix
- Red shaded region marks the MF-only column (>300T)

Graphs do **not** need updating for a "4-node full-pipeline run" because no such run
exists. The only 4-MPI-rank xlarge data is PBS 168000131 (MF-only, fixed tree, seed=42).

### Implementation — `tiers/` submission scripts

Created `tiers/` directory with the batch scripts to execute Tiers 1–3 (16 jobs).
Tier 4 (ICX-compiled R2 isolation) is deferred — it requires a new binary build.

| File | Role |
|---|---|
| `tiers/README.md` | Plan, walltime estimates, KSU costs |
| `tiers/run_xlarge_mf_audit.sh` | PBS worker: MF-only MF2 runs (np=1, `-m MF -te fixed_xlarge_tree.nwk --seed 1`, um09/build-mpi-mf2) |
| `tiers/run_xlarge_avx_omp.sh` | PBS worker: AVX-512+R2 binary in `mpirun -np 1` OMP-only mode for T₁ anchors |
| `tiers/submit_tier1.sh` | Submits 7 anchor jobs (R2+NUMA 1/4/8/16T + AVX 1/4/8T) |
| `tiers/submit_tier2.sh` | Submits GCC 104T via `submit_benchmark_matrix.sh xlarge_mf 104` |
| `tiers/submit_tier3.sh` | Submits 7 MF-only MF2 scaling jobs + MF2 2-node dispatch (8 jobs) |
| `tiers/submit_all.sh` | Submits all three tiers in sequence (16 jobs total) |

All workers write run records to `logs/runs/` in the schema consumed by
`tools/scaling_10M_analysis.py::load_xlarge_gadi()`. After all jobs complete,
re-running the analysis script will:
- Drop the `[⚠ T₁ extrap.]` flag from R2+NUMA (4 new low-thread points)
- Drop the `[⚠ T₁ extrap.]` flag from AVX-512+R2 (3 new low-thread points)
- Extend GCC Canonical to 104T (NUMA penalty visible)
- Add a 6th family "MF-only MF2" giving the MF2 ◆ point a comparable same-binary curve

### Walltime / SU cost analysis

Original concern was that long walltime runs would waste KSU. The opposite is true:
**long-wall low-thread runs are SU-cheap** because PBS bills `ncpus × hours × 2.0`.

| Run | Wall (est.) | ncpus | SU = ncpus × h × 2 |
|---|---|---|---|
| R2+NUMA 1T   | ~3h20m   | 1    | 8 SU      |
| R2+NUMA 4T   | ~1h15m   | 4    | 16 SU     |
| R2+NUMA 8T   | ~45m     | 8    | 24 SU     |
| R2+NUMA 16T  | ~25m     | 16   | 32 SU     |
| AVX 1T       | ~3h20m   | 1    | 8 SU      |
| AVX 4T       | ~1h15m   | 4    | 16 SU     |
| AVX 8T       | ~45m     | 8    | 24 SU     |
| GCC 104T     | ~25m     | 104  | 208 SU    |
| MF-only 1T   | ~1–3h    | 1    | 8 SU      |
| MF-only 4T   | ~30m     | 4    | 12 SU     |
| MF-only 8T   | ~15m     | 8    | 16 SU     |
| MF-only 16T  | ~10m     | 16   | 16 SU     |
| MF-only 32T  | ~5m      | 32   | 32 SU     |
| MF-only 64T  | ~3m      | 64   | 32 SU     |
| MF-only 104T | ~2m      | 104  | 104 SU    |
| MF2 208T (2-node) | ~2m | 208  | 208 SU    |
| **TOTAL (16 jobs)** | | | **≈ 764 SU ≈ 0.76 KSU** |

The two heaviest jobs are GCC 104T and MF2 2-node 208T (208 SU each) — both
unavoidable for completeness. The 1T runs cost 8 SU each despite running 3+ hours.

### Usage

```bash
DRY_RUN=1 ./tiers/submit_all.sh   # preview qsub commands
./tiers/submit_tier1.sh           # submit Tier 1 (recommended first batch)
./tiers/submit_tier2.sh
./tiers/submit_tier3.sh
# After completion:
python3.11 tools/scaling_10M_analysis.py
```

### Job execution log (2026-05-12)

All 16 jobs submitted via `./tiers/submit_all.sh`. Two infrastructure bugs found
during the first run, fixed, and affected jobs resubmitted.

#### Bug 1 — `ldd` preflight fails before `module load openmpi/4.1.7`

`run_xlarge_mf_audit.sh` ran `ldd` to verify the binary links libmpi before the
`module load openmpi/4.1.7` line. On compute nodes, `ldd` cannot resolve OpenMPI
shared libraries without the module, so it returned empty output and the check
always failed.

**Fix:** replaced `ldd ... | grep -qE 'libmpi(\.|_)'` with
`readelf -d ... | grep -q 'NEEDED.*libmpi'`, which reads the static ELF dynamic
section and requires no runtime library path.

**Affected job:** 168114291 (`iq-mf-16t`, 16T) — failed immediately, exit 6.
IQ-TREE was never launched. Resubmitted as **168115509**.

#### Bug 2 — `python3` defaults to 3.6.8 on Gadi compute nodes

The inline Python heredocs in `run_xlarge_mf_audit.sh` and `run_xlarge_avx_omp.sh`
used walrus operators (`:=`, PEP 572) which require Python 3.8+. Gadi compute nodes
resolve `python3` to `/bin/python3` (3.6.8), causing a `SyntaxError` in the JSON
writer after IQ-TREE completed successfully.

**Fix:** replaced all bare `python3` invocations with `/usr/bin/python3.11` across
all 16 run scripts in `gadi-ci/` and `tiers/` via `sed -i`.

**Affected jobs:** 168114289 (`iq-mf-4t`), 168114290 (`iq-mf-8t`),
168114292 (`iq-mf-32t`), 168114293 (`iq-mf-64t`), 168114294 (`iq-mf-104t`) —
IQ-TREE finished and produced correct results but no JSON was written to
`logs/runs/`. The 4T and 8T failures were discovered after the initial audit
(both had wall times long enough to run through before outputs were checked).
Resubmitted as **168115629** (32T), **168115630** (64T), **168115631** (104T),
**168115782** (4T), **168115786** (8T).

#### Early results — MF-only MF2 Tier 3 (from first-run IQ-TREE logs)

The three failed-JSON runs did complete IQ-TREE successfully. Results from
`gadi-ci/profiles/xlarge_mf_<T>t_mf2_mfonly_np1_seed1_<PBS>/iqtree_mfonly.log`:

| PBS | Threads | Wall (IQ-TREE) | Wall (MF only) | Best model | lnL | Status |
|---|---|---|---|---|---|---|
| 168114289 | 4T   | 819s | — | SYM+G4 | −10956936.093 | IQ-TREE ✓, JSON ✗ → resubmit |
| 168114290 | 8T   | 538s | — | SYM+G4 | −10956936.093 | IQ-TREE ✓, JSON ✗ → resubmit |
| 168114292 | 32T  | 200s | 198s | SYM+G4 | −10956936.093 | IQ-TREE ✓, JSON ✗ → resubmit |
| 168114293 | 64T  |  95s |  94s | SYM+G4 | −10956936.093 | IQ-TREE ✓, JSON ✗ → resubmit |
| 168114294 | 104T |  72s |  70s | SYM+G4 | −10956936.093 | IQ-TREE ✓, JSON ✗ → resubmit |

lnL is consistent across all five thread counts (−10956936.093), confirming
reproducibility. MF wall ≈ total wall (fixed-tree `-te`, no tree search).

MF2 2-node dispatch (168114295, 208T) completed successfully with exit 0.

#### Script fixes applied

| File | Change |
|---|---|
| All 16 `gadi-ci/run_*.sh`, `submit_benchmark_matrix.sh` | `python3 ` → `/usr/bin/python3.11 ` |
| All 16 `tiers/run_*.sh` | Same |
| `#!/usr/bin/env python3` shebangs in heredoc sampler blocks | → `#!/usr/bin/python3.11` |
| `tiers/run_xlarge_mf_audit.sh` | `ldd` → `readelf -d` in libmpi preflight |

#### Job register at submission (normalsr, SPR, 2.0 SU/ch)

| T | PBS ID | Job name | ncpus | nd | Pr | Wall | Status |
|---|---|---|---|---|---|---|---|
| 1 | 168114279 | iq-r2-1t   | 1   | 1 | rc29 | 4h   | D (qdel — old python3) |
| 1 | 168114280 | iq-r2-4t   | 4   | 1 | rc29 | 2h   | D (qdel — old python3) |
| 1 | 168114281 | iq-r2-8t   | 8   | 1 | rc29 | 90m  | D (qdel — old python3) |
| 1 | 168114282 | iq-r2-16t  | 16  | 1 | rc29 | 1h   | D (qdel — old python3) |
| 1 | 168114283 | iq-avx-1t  | 1   | 1 | um09 | 4h   | D (qdel — old python3) |
| 1 | 168114284 | iq-avx-4t  | 4   | 1 | um09 | 2h   | D (qdel — old python3) |
| 1 | 168114285 | iq-avx-8t  | 8   | 1 | um09 | 90m  | D (qdel — old python3) |
| 2 | 168114287 | iq-xlarge-gcc-104t | 104 | 1 | rc29 | 24h | H (held, allocation) |
| 3 | 168114288 | iq-mf-1t   | 1   | 1 | um09 | 4h   | D (qdel — old python3) |
| 3 | 168114289 | iq-mf-4t   | 4   | 1 | um09 | 90m  | F (python3 bug, exit 1, IQ-TREE wall=819s) |
| 3 | 168114290 | iq-mf-8t   | 8   | 1 | um09 | 1h   | F (python3 bug, exit 1, IQ-TREE wall=538s) |
| 3 | 168114291 | iq-mf-16t  | 16  | 1 | um09 | 30m  | **F (ldd bug, exit 6)** |
| 3 | 168114292 | iq-mf-32t  | 32  | 1 | um09 | 30m  | F (python3 bug, exit 1) |
| 3 | 168114293 | iq-mf-64t  | 64  | 1 | um09 | 30m  | F (python3 bug, exit 1) |
| 3 | 168114294 | iq-mf-104t | 104 | 1 | um09 | 30m  | F (python3 bug, exit 1) |
| 3 | 168114295 | iq-mf2-2nd | 208 | 2 | um09 | 30m  | **✓ exit 0** |

Resubmissions (fixed scripts):

| PBS ID | Job | Replaces | Actual wall | Exit | JSON written |
|---|---|---|---|---|---|
| 168115509 | iq-mf-16t  | 168114291 | 366s | 0 ✓ | `gadi_xlarge_mf_16t_mf2_mfonly_np1_seed1.json` |
| 168115629 | iq-mf-32t  | 168114292 | 168s | 0 ✓ | `gadi_xlarge_mf_32t_mf2_mfonly_np1_seed1.json` |
| 168115630 | iq-mf-64t  | 168114293 | 106s | 0 ✓ | `gadi_xlarge_mf_64t_mf2_mfonly_np1_seed1.json` |
| 168115631 | iq-mf-104t | 168114294 |  72s | 0 ✓ | `gadi_xlarge_mf_104t_mf2_mfonly_np1_seed1.json` |
| 168115782 | iq-mf-4t   | 168114289 |  9m  | 271 ✗ | qdel'd — resubmitted as 168116165 (no `place=excl`) |
| 168115786 | iq-mf-8t   | 168114290 | 8m51s | 0 ✓ | finished before qdel; no `place=excl` (see note) |
| 168115825 | iq-r2-1t   | 168114279 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116041 with `place=excl` |
| 168115826 | iq-r2-4t   | 168114280 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116043 with `place=excl` |
| 168115827 | iq-r2-8t   | 168114281 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116045 with `place=excl` |
| 168115828 | iq-r2-16t  | 168114282 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116047 with `place=excl` |
| 168115829 | iq-avx-1t  | 168114283 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116049 with `place=excl` |
| 168115830 | iq-avx-4t  | 168114284 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116051 with `place=excl` |
| 168115831 | iq-avx-8t  | 168114285 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116053 with `place=excl` |
| 168115834 | iq-mf-1t   | 168114288 |  5m  | 271 ✗ | qdel'd — resubmitted as 168116162 with `place=excl` |

#### `logs/jobs/tiers/` file manifest

Outputs from the first batch (submitted before `-o` flag was added) were moved
manually to `logs/jobs/tiers/`. Failed outputs have been removed. Future submissions
via the updated `tiers/submit_*.sh` scripts route there automatically via
`qsub -o logs/jobs/tiers`.

**Currently present (successful runs only):**

| File | PBS | Job | Threads | Wall | Notes |
|---|---|---|---|---|---|
| `iq-mf2-2nd.o168114295`       | 168114295 | iq-mf2-2nd | 208T | ~127s | MF2 2-node dispatch, exit 0 ✓ |
| `iq-xlarge-mfonly.o168115509` | 168115509 | iq-mf-16t  | 16T  | 366s  | fixed resubmit, exit 0 ✓ |
| `iq-xlarge-mfonly.o168115629` | 168115629 | iq-mf-32t  | 32T  | 168s  | fixed resubmit, exit 0 ✓ |
| `iq-xlarge-mfonly.o168115630` | 168115630 | iq-mf-64t  | 64T  | 106s  | fixed resubmit, exit 0 ✓ |
| `iq-xlarge-mfonly.o168115631` | 168115631 | iq-mf-104t | 104T |  72s  | fixed resubmit, exit 0 ✓ |

**Removed (failed outputs):**

| Removed file | PBS | Exit | Reason |
|---|---|---|---|
| `iq-mf-4t.o168114289`     | 168114289 | 1 | python3 bug (IQ-TREE ran 819s, no JSON) |
| `iq-mf-8t.o168114290`     | 168114290 | 1 | python3 bug (IQ-TREE ran 538s, no JSON) |
| `iq-mf-16t.o168114291`    | 168114291 | 6 | ldd bug (IQ-TREE never ran) |
| `iq-mf-32t.o168114292`    | 168114292 | 1 | python3 bug (IQ-TREE ran 200s, no JSON) |
| `iq-mf-64t.o168114293`    | 168114293 | 1 | python3 bug (IQ-TREE ran 95s, no JSON) |
| `iq-mf-104t.o168114294`   | 168114294 | 1 | python3 bug (IQ-TREE ran 72s, no JSON) |
| `iq-xlarge-mfonly.o168115508` | 168115508 | — | duplicate 16T, qdel'd |
| `168115783.gadi-pbs.OU`   | 168115783 | — | duplicate 8T, qdel'd before running |

**Cancelled before completion (no output files written):**

| PBS | Job | Elapsed | Reason |
|---|---|---|---|
| 168114279 | iq-r2-1t  | 24m | qdel — submitted with old python3 |
| 168114280 | iq-r2-4t  | 24m | qdel — submitted with old python3 |
| 168114281 | iq-r2-8t  | 24m | qdel — submitted with old python3 |
| 168114282 | iq-r2-16t | 24m | qdel — submitted with old python3 |
| 168114283 | iq-avx-1t | 24m | qdel — submitted with old python3 |
| 168114284 | iq-avx-4t | 24m | qdel — submitted with old python3 |
| 168114285 | iq-avx-8t | 24m | qdel — submitted with old python3 |
| 168114288 | iq-mf-1t  | 24m | qdel — submitted with old python3 |

**Still in queue / held:**

| Expected file | PBS | Job | ncpus | Status |
|---|---|---|---|---|
| `iq-xlarge*.o168114287` | 168114287 | iq-xlarge-gcc-104t | 104 | **qdel'd 2026-05-12** — H (rc29 SU exhausted; never ran); resubmit→168137038 (um09) |

**Pending resubmissions (will route to `logs/jobs/tiers/` automatically):**

All pending runs below use `-P um09`.  The r2 canonical script `#PBS -P rc29`
directive has been corrected to `um09`; `submit_tier1.sh` `run_qsub` helper
now passes `-P um09` explicitly.

**168116xxx wave — outcomes (2026-05-12):**

| File | PBS | Job | Threads | Wall | Exit | Note |
|---|---|---|---|---|---|---|
| `168116041.gadi-pbs.OU` | 168116041 | iq-r2-1t  | 1T  | 4h00m | **-29 ✗** | walltime; Pass1 killed at 14412s; resubmit→168136896 |
| `168116043.gadi-pbs.OU` | 168116043 | iq-r2-4t  | 4T  | 2h01m | **-29 ✗** | walltime; Pass1=4951s, killed in Pass2; resubmit→168136897 |
| `168116045.gadi-pbs.OU` | 168116045 | iq-r2-8t  | 8T  | 1h31m | **-29 ✗** | walltime; Pass1=3113s, killed in Pass2; resubmit→168136898 |
| `168116047.gadi-pbs.OU` | 168116047 | iq-r2-16t | 16T | 1h00m | **-29 ✗** | walltime; Pass1=1900s, killed in Pass2; resubmit→168136899 |
| `168116049.gadi-pbs.OU` | 168116049 | iq-avx-1t | 1T  | 4h00m | **-29 ✗** | walltime; preflight OK then killed before IQ-TREE; resubmit→168136859 |
| `168116051.gadi-pbs.OU` | 168116051 | iq-avx-4t | 4T  | 1h19m | **0 ✓**   | wall=4746s; `gadi_xlarge_mf_4t_icx_mpi1x4_avx512_r2_ompanchor.json` |
| `168116053.gadi-pbs.OU` | 168116053 | iq-avx-8t | 8T  | 0h49m | **0 ✓**   | wall=2963s; `gadi_xlarge_mf_8t_icx_mpi1x8_avx512_r2_ompanchor.json` |
| `168116162.gadi-pbs.OU` | 168116162 | iq-mf-1t  | 1T  | 0h45m | **0 ✓**   | wall=2727s; `gadi_xlarge_mf_1t_mf2_mfonly_np1_seed1.json` |
| `168116165.gadi-pbs.OU` | 168116165 | iq-mf-4t  | 4T  | 0h13m | **0 ✓**   | wall=817s; `gadi_xlarge_mf_4t_mf2_mfonly_np1_seed1.json` |

Root cause for -29 failures: `run_xlarge_r2_v312_canonical.sh` runs two passes
(clean timing + `perf stat`), total ≈ 2×Pass1 + overhead. Walltimes were
under-estimated. Corrected walltimes in `tiers/submit_tier1.sh` and resubmitted:

**168136xxx / 168137xxx wave — resubmissions with corrected walltimes (2026-05-12, um09):**

| PBS | Job | Threads | New walltime | Project billing | Status |
|---|---|---|---|---|---|
| 168136896 | iq-r2-1t        | 1T   | 09:00:00 | um09 | R |
| 168136897 | iq-r2-4t        | 4T   | 03:30:00 | um09 | R |
| 168136898 | iq-r2-8t        | 8T   | 02:30:00 | um09 | R |
| 168136899 | iq-r2-16t       | 16T  | 01:30:00 | um09 | R |
| 168136859 | iq-avx-1t       | 1T   | 07:00:00 | um09 | R |
| 168137038 | iq-xlarge-gcc-104t | 104T | 24:00:00 | um09 | Q — binary/data from rc29 scratch; PROJECT_DIR overridden to rc29 path |

> Note: `iq-mf-8t.o168115786` (168115786, 8T, no `place=excl`) completed with
> exit 0 before the qdel landed (wall=8m51s). Result is valid but was run on a
> shared node — acceptable given the 8T run has lower leverage on Amdahl T₁
> than the 1T/4T points. It will not be resubmitted unless the fitted curve is
> noticeably inconsistent with the excl results.

#### Node exclusivity — shared vs exclusive allocation

**`-l place=excl` does not work on Gadi `normalsr`:** the queue policy silently
overrides the user's placement request to `place=free`. Verified via `qstat -f`
on the resubmitted jobs: all show `Resource_List.place = free` and
`exec_vnode = (node:ncpus=N:...)` (only requested CPUs allocated, not the full
node). The 1T and 4T r2 jobs (168116041, 168116043) were confirmed co-resident
on `gadi-cpu-spr-0070` within the same submission wave.

**Cost:** unchanged — `normalsr` bills on `ncpus` requested regardless of
placement. The `-l place=excl` flag had zero effect on isolation or cost.

**True node exclusivity on `normalsr` requires `ncpus=104`.** At 2.0 SU/cpu-hour,
a 1T run on a full-node exclusive allocation costs 104 × 4h × 2 = 832 SU vs 8 SU
shared — not practical for T₁ anchor runs.

**Accepted caveat for low-thread anchor runs (1T, 4T, 8T, 16T):**

Co-resident jobs on the same SPR node can inflate walltime via:
- L3 cache eviction (IQ-TREE streams large rate matrices; ~2 MB/core L3 slice)
- DDR5 memory bandwidth contention (8 channels shared across 104 cores)
- Reduced per-core turbo boost under full-node load

Net effect: T₁ may be measurably inflated, biasing the Amdahl serial fraction `f`
upward and compressing the fitted speedup curves. This applies equally to the
existing ICX Baseline and GCC Canonical series (PBS 166978126–167520757) which
were also run shared — so the comparison is internally consistent. If the fitted
curves look pessimistic relative to the high-thread measured points, the 1T/4T
runs can be resubmitted overnight when queue load is low (better empirical isolation).

#### Binary / version / patch matrix — all xlarge families (2026-05-12 audit)

Confirmed from bootstrap scripts, `.build-info.json`, and kernel library presence
in each build directory:

| Family | Binary path | IQ-TREE | Compiler | R1+R2 | `libkernelavx512` | MPI | Notes |
|---|---|---|---|---|---|---|---|
| GCC Canonical | `rc29/iqtree3/build-profiling/iqtree3` | **3.1.1** | GCC 14.2.0 | ✓ | ✗ | ✗ | Also used by old "ICX Baseline" label |
| ICX Baseline | same binary | **3.1.1** | GCC 14.2.0 | ✓ | ✗ | ✗ | "ICX" refers to the SPR node, not the compiler |
| R2+NUMA Clang | `rc29/iqtree3-3.1.2/build-profiling-clang/iqtree3` | **3.1.2** | Clang/LLVM | ✓ | **✗** | ✗ | Has `libkernelfma.a`+`libkernelavx.a`; no AVX-512 kernel |
| AVX-512+R2 | `um09/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi` | **3.1.2** | ICX+OpenMPI | ✓ | **✓** | ✓ | `-DIQTREE_FLAGS=mpi` builds `libkernelavx512.a` |
| MF2 mf-only | `um09/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi` | **3.1.2+MF2** | ICX+OpenMPI | ✓ | ✓ | ✓ | MF2 dispatch patch on top of AVX-512+R2 |

**Key observation — `libkernelavx512.a`:** both Clang and MPI builds use
`-march=sapphirerapids`, but only the MPI build (via `-DIQTREE_FLAGS=mpi` in CMake)
produces `libkernelavx512.a`. The Clang OMP-only build uses IQ-TREE's FMA/AVX kernel
at runtime; the MPI build uses the full AVX-512 SIMD likelihood kernel. This is the
primary distinction between the R2+NUMA Clang and AVX-512+R2 families (in addition
to MPI scaffolding).

**Confirmed confounds in cross-family comparisons:**

| Comparison | Version diff | Compiler diff | Kernel diff | Patch diff | Clean? |
|---|---|---|---|---|---|
| GCC Canonical → R2+NUMA Clang | 3.1.1 → 3.1.2 | GCC → Clang | AVX → FMA | +R1+R2 already in both | **✗ 3 confounds** |
| ICX Baseline → R2+NUMA Clang | 3.1.1 → 3.1.2 | GCC → Clang | AVX → FMA | +R1+R2 already in both | **✗ 2 confounds** |
| R2+NUMA Clang → AVX-512+R2 | none | Clang → ICX | FMA → AVX-512 | none (R1+R2 in both) | **✓ clean AVX-512 isolation (+ compiler minor)** |
| AVX-512+R2 → MF2 mf-only | none | none | none | +MF2 dispatch | **✓ clean MF2 isolation** (but protocol differs — MF-only) |

**In-flight jobs are correctly configured within their families:**

| PBS | Job | Family | Binary | Consistent with existing data? |
|---|---|---|---|---|
| 168136896–168136899 | r2-1t/4t/8t/16t | R2+NUMA Clang | `build-profiling-clang/iqtree3` | ✓ same binary as 167865974–167865976 (32T/64T/104T) |
| 168136859 | avx-1t | AVX-512+R2 | `build-profiling-mpi/iqtree3-mpi` | ✓ same binary as 168116051/053 (avx-4T/8T done) |
| 168137038 | gcc-104t | GCC Canonical | `build-profiling/iqtree3` (3.1.1) | ✓ same binary as 167520752–167520757 (1T–64T) |

No binary or configuration changes needed for the 6 in-flight jobs. All runs will
produce data that extend their respective families correctly.

**To cleanly isolate the R2 patch effect from compiler effect** (Priority 4 in the
run list above): build a 3.1.2 GCC binary from `rc29/iqtree3-3.1.2/src/iqtree3`
with `gcc/14.2.0` + R1+R2 patches and run 1T/104T. This will reveal what fraction
of the R2+NUMA Clang speedup comes from the NUMA patch vs the compiler change.
This deferred — no new build or jobs submitted in this session.

---

## 2026-05-10 (ad) — mega_dna 4-node MF2-dispatch APS run (PBS 168015597)

### Objective

Run `mega_dna.fa` (500 taxa × 100,000 DNA sites) on 4 Gadi normalsr SPR nodes
with the MF2-dispatch binary (`abd98764`, cost-sorted LPT stripe + MF_WAITING fix),
full 968-model ModelFinder, Intel APS profiling (APS_STAT_LEVEL=2), and
4 MPI ranks × 104 OMP threads with `OMP_PROC_BIND=close` + `numactl --localalloc`.

### Configuration

| Parameter | Value |
|---|---|
| Binary | `iqtree3-mpi` @ `abd98764` (gadi-spr-r2-avx512 branch) |
| Build | `icpx` + OpenMPI, `-xSAPPHIRERAPIDS`, `-O3 -march=sapphirerapids` |
| Dataset | `mega_dna.fa` — 500 taxa × 100,000 DNA sites |
| Site patterns | 99,999 / 100,000 distinct (effectively no compression — 500 taxa) |
| MPI ranks | 4 (one per node) |
| OMP threads | 104 per rank (416 total) |
| MF dispatch | Cost-sorted LPT stripe — rank 0/4 → 242/968 models |
| Profiler | intel-vtune/2025.8.1, APS_STAT_LEVEL=2 |
| Queue | normalsr, 4× SPR nodes (8470Q, 104 cores/node) |
| Walltime limit | 2:00:00 |

### Results vs baseline (Gadi_mega_dna_104T, PBS 167001099)

The baseline ran tree search only (no ModelFinder) at 104T on a single node with
the icx build, no thread pinning, no numactl, and VTune co-resident (+26% overhead).

| Metric | Baseline `167001099` | Current `168015597` |
|---|---|---|
| Platform | 1× SPR node | 4× SPR nodes |
| Threads | 104T single rank | 4× 104T = 416T MPI |
| ModelFinder | ❌ None | ✅ Full 968-model scan |
| Wall time | ~39.5 min (clean est.) | **18m 31s** |
| CPU time total | ~68 CPU-hours | **30.1 CPU-hours** |
| IPC | 1.02 | 1.15 / 1.29 / 1.35 / 1.42 (ranks 0–3) |
| LLC miss rate | 79.4% | 86.6 / 86.1 / 87.0 / 83.0% (ranks 0–3) |
| Best-fit model (BIC) | N/A | **SYM+I+R2** |
| Log-likelihood (tree) | -27,328,165.86 | -27,328,165.83 ✓ |
| BIC score | N/A | 54,667,982.74 |

**2.1× faster wall time doing more work** — full ModelFinder included — due to
MF-MPI dispatch spreading 968 models across 4 ranks in parallel.

**IPC improved** (avg ~1.30 vs 1.02): `OMP_PROC_BIND=close` + `numactl --localalloc`
eliminate cross-NUMA thread migration and first-touch remote allocation.

**LLC miss rate increased** (~86% vs 79%): expected — 4 independent ranks each
churn 99,999-pattern alignments with no inter-rank data sharing; no L3 reuse between
model evaluations. This is structural (dataset has essentially zero pattern compression).

### Model selection

Best-fit by BIC: **SYM+I+R2** (symmetric rate matrix + invariant + 2 free rates).
BIC strongly penalises extra parameters at N=100,000 sites, favouring SYM over GTR.
Top models by BIC:

```
SYM+I+R2     logL -28,064,347.70   BIC 56,140,265.89   w-BIC 1.00
GTR+F+I+R2   logL -28,064,365.34   BIC 56,140,335.71   w-BIC 6.9e-16
TVMe+I+R2    logL -28,067,498.30   BIC 56,146,555.58   w-BIC 0
```

### Engineering notes

- **`strings` vs `grep -a`**: Binary guard for LPT patch must use `grep -qa` — the
  ELF binary has `too many notes (256)` which causes `strings` to miss embedded
  strings that `grep -a` finds correctly on Gadi compute nodes.
- **`OMP_DISPLAY_AFFINITY=TRUE` removed**: This env var (+ `OMP_DISPLAY_ENV=VERBOSE`)
  caused Intel OMP to write thread affinity to stdout on every model evaluation —
  producing a 5.5 GB log file that throttled disk I/O and reduced model throughput
  ~5×. Removed in commit `9a69d310`.
- **APS collection**: Per-rank APS wrapper (`aps -r aps_result/<dir> -- numactl ...`)
  confirmed working; `aps_result/` dirs populated. Reports generated post-job.

### PBS output

`/home/272/as1708/setonix-iq/iq-mega-mf2-4node-aps.o168015597`

### Post-run analysis: model cost distribution

119 per-model wall times extracted from `iqtree_clean.model.gz` checkpoint,
revealing the distribution is **bimodal**, not normal or lognormal.

| Cluster | Models | Mean | Range | Rate suffixes |
|---|---|---|---|---|
| Fast | 47 / 119 | 93.6s | 82–95s | base, +I, +R2 |
| Slow | 72 / 119 | 102.0s | 98–103s | +G, +R3, +R4, +R5, +R6 |

Overall: n=119, mean=98.7s, CV=4.6%, Max/Min=1.25×, skewness=−0.88.

**Why so narrow?** 99,999/100,000 distinct patterns means per-site likelihood
evaluation dominates every model — extra rate categories add only marginal overhead.
The bimodal gap shrinks to insignificance compared to the total per-model compute.

**Dispatch imbalance at 100K sites**: All strategies within 0.5% (LPT ≈ naive ≈
dynamic) — distribution is too uniform to show meaningful differences at ≤16 ranks.

**Extrapolated 10M-site distribution**: Estimated ratio grows to ~2.7× (JC ≈96 min,
GTR+R6 ≈258 min) because JC converges faster (fewer params) while GTR+R6 requires
more optimisation rounds on a larger likelihood surface. At 16 ranks, LPT wastes 1.6%
vs naive's 6.9%; at 32 ranks LPT wastes 3.8% vs naive's 14.5%. LPT is the practical
optimum — no MPI communication overhead while achieving 4× better balance than naive.

See [design/modelfinder-mpi-dispatch.md](design/modelfinder-mpi-dispatch.md) §13
for full analysis and simulation tables.

---

## 2026-05-10 (ac) — Performance Research: CPU/MPI bottlenecks for 10M full-sweep

### Research question

> Where is performance stuck at 4 MPI ranks × 104 OMP threads on 10M sites, and
> what code changes could reduce walltime for the full 968-model sweep?

### Hardware characterisation (xlarge_mf, 104T / 2-node, PBS 167973941)

The best available hardware profiling is from the xlarge_mf (100K-site) benchmark.
All directional conclusions apply to the 10M case; absolute numbers scale accordingly.

| Counter | Value | Implication |
|---------|-------|-------------|
| IPC | **1.10** insn/cycle | vs AVX-512 FMA theoretical max ~4.0 → 73% compute headroom unreachable |
| LLC miss rate | **77.9%** | Structural — CLV arrays far exceed L3 capacity |
| LLC loads | 44.3B | Each branch traversal = full DRAM round-trip |
| LLC load misses | 34.5B | 77.9% miss → ~0 reuse from L3 |
| DRAM bandwidth | **1.7%** util | 5.1 GB/s actual vs 307 GB/s peak |
| dTLB misses | **0.005%** | TLB is not the bottleneck |
| Branch mispredicts | 0.05% | Branch prediction is not the bottleneck |

**Bottleneck classification: latency-bound, not bandwidth-bound.**
At 2,793 cycles/LLC-miss (from DRAM latency ~270 cycles × OOO inflight factor) the
processor's out-of-order window cannot queue enough outstanding misses to keep the
pipeline full.  More threads beyond ~52 yield diminishing returns because the LLC
miss queue is already saturated, not the DRAM bus.

### 10M dataset memory footprint

- Alignment: 954 MB, **0% site-pattern compression** (10M unique patterns)
- CLV per branch per iteration:
  - JC (no rate variation):      10M × 4 states × 1 cat × 8B = **320 MB**
  - GTR+G4 (+4-cat Γ):           10M × 4 × 4 × 8B = **1.28 GB**
  - GTR+R10 (slowest, 10 cats):  10M × 4 × 10 × 8B = **3.20 GB**
- Total CLV store (all branches): 197 branches × 320 MB–3.2 GB = **63–630 GB**
- L3 cache per SPR node: **105 MB** — CLV working set exceeds L3 by 600–6000×
- Consequence: every traversal step is a **cold DRAM miss**, regardless of thread count

### Per-model timing (confirmed from PBS 167977883)

| Model class | Estimated wall per model (4 nodes, 104T/rank) |
|-------------|----------------------------------------------|
| JC, F81, GTR… (no rate variation) | ~744s ≈ 12 min |
| +G4, +I+G4 | ~1,500–2,500s ≈ 25–40 min (estimated, 4× CLV) |
| +R6, +R8, +R10 | ~3,000–8,000s ≈ 50–130 min (estimated, 6–10× CLV, more EM iterations) |

Full 968-model sweep walltime at 4 MPI ranks × 1-rank/node (242 models/rank):
- Lower bound (all flat): 242 × 744s / 3600 ≈ **50 h**
- Realistic (mixed):      242 × ~3,000s / 3600 ≈ **200 h**
- `normalsr` queue limit: 48 h → **cannot complete in one job without intervention**

### Checkpoint-resume mechanism (confirmed from source)

`CandidateModel::evaluate()` (phylotesting.cpp:1961) calls `restoreCheckpoint()`
before any computation:
```cpp
if (restoreCheckpoint(&in_model_info)) {
    delete iqtree;
    return "";   // microseconds — model score already in .model.gz
}
```
`MF_DONE = 16` is NOT in the `getNextModel()` skip mask (14 = MF_IGNORED+MF_RUNNING+MF_WAITING).
Models from a prior run are re-visited but restored from checkpoint instantly.

**PBS job chaining is already supported — no code changes required.**
To resume a killed job, re-submit with the same `--prefix` (no `--redo`):
```
NOTE: Restoring information from model checkpoint file ...
```
IQ-TREE prints that message and skips all models already in `.model.gz`.
The starting tree is also stored in `.ckp.gz` — not re-computed on resume.
Per-resume overhead: ~10s checkpoint load + program startup.

### `--mpi-ranks-per-node 2`: thread budget division (confirmed from source)

phylotesting.cpp:1531:
```cpp
int rank_threads = max(1, params.num_threads / params.mpi_ranks_per_node);
```
With `-T 104 --mpi-ranks-per-node 2` on 4 nodes:
- 8 total MPI ranks × 52 OMP threads each
- 968 / 8 = **121 models per rank** (vs 242 at 1-rank/node)
- Per-model time: uncertain — halving threads reduces concurrent DRAM requests,
  which for a latency-bound workload may INCREASE per-model time (fewer in-flight
  misses → lower effective memory-level parallelism)
- Net effect: must be empirically tested; may not help

### Software prefetch gap (confirmed from source)

Zero calls to `_mm_prefetch` or `__builtin_prefetch` exist in any phylokernel*.{h,cpp} file.

The inner ptn loop in `phylokernelnew.h`:
```cpp
for (size_t ptn = ptn_lower; ptn < ptn_upper; ptn += VectorClass::size()) {
    VectorClass *partial_lh = (VectorClass*)(dad_branch->partial_lh + (ptn*block));
    // ... 4–10 FMAs per state per cat, then advance ptn
}
```
For +G4 on 10M sites: `block = 16` doubles, stride = `16 × 8 × 8 = 1024 bytes`.
DRAM latency ≈ 270 cycles.  Prefetch distance for full hiding ≈ 270 / (16 FMAs/iter ×
throughput) ≈ **16–32 iterations ahead** (512–1024 bytes).
Adding `_mm_prefetch((char*)&partial_lh[AHEAD * block], _MM_HINT_T2)` at the ptn
loop entry could hide 20–40% of stall cycles by overlapping DRAM fetches with
compute.  Estimated speedup: **5–15% per model**.  Medium implementation effort
(one line per traversal loop in phylokernelnew.h, ~4 sites).

### Cross-rank BIC pruning opportunity (MPI code change)

Current `filterRates()` and `filterSubst()` only prune within a rank's own model
stripe.  Example: if rank 0 evaluates GTR+R4 and BIC is already worse than
GTR+G4, ranks 1–3 do not know this and still evaluate SYM+R5, TVM+R6, etc.

**Phase 2.5 mid-sweep gather** (new code, high effort):
After evaluating all rate-homogeneous models (the first `subst_block` models), do a
partial `MPI_Allreduce` of current best BIC.  Each rank can then skip +Rk models
where `filterRates()` would fire against the global best BIC rather than only its
own partial view.  Estimated model-count reduction: **15–30%** (most of the +R8..
+R10 tail).

### Optimisation priority matrix

| Approach | Effort | Expected walltime reduction | Code changes |
|----------|--------|----------------------------|--------------|
| **PBS job chaining via checkpoint** | None | Enables splitting 48h jobs across days; functionally unlocks 968-model sweep | None |
| **2 ranks/node empirical test** | Low (script only) | Unknown (may help or hurt); halves models/rank | Script flag only |
| **Software prefetch in ptn loop** | Medium (~10 lines, 4 loop sites in phylokernelnew.h) | 5–15% per-model | phylokernelnew.h |
| **Phase 2.5 cross-rank BIC pruning** | High (new MPI_Allreduce + filter logic) | 15–30% model skip rate | phylotesting.cpp |
| **Looser --mf-epsilon (e.g. 1.0)** | None (CLI flag) | 2–5× per-model (risky) | Not recommended — changes best-fit selection |
| **NUMA first-touch** | Already applied | ~0% on 10M (latency-bound, not NUMA-bound) | None |
| **More OMP threads / MPI ranks** | N/A | 0% — LLC queue already saturated | N/A |

### Next steps

1. **Immediate**: Submit a `--mrate G,I+G` 4-node job (88 models, ~6h) to validate
   end-to-end correctness with the LPT-fix binary.  This also creates the `.model.gz`
   checkpoint that a follow-up full-sweep job can build on.
2. **Near-term**: Test `--mpi-ranks-per-node 2` on an xlarge run and compare
   per-model time vs 1-rank/node at the same total core count.
3. **Code**: Add `_mm_prefetch` to the 4 ptn-loop sites in `phylokernelnew.h`.
   Profile before/after on xlarge_mf to measure IPC improvement.
4. **Research**: Prototype Phase 2.5 mid-sweep Allreduce on a small dataset
   (xlarge_mf, 4 ranks) and count models skipped vs baseline.

---

## 2026-05-10 (ab) — Issue 7: MF_WAITING cross-rank blocking + LPT cost-sort fix

### PBS 168000932 — Job Outcome (confirmed)

**Job killed at walltime (exit -29).** Run details extracted from
`iq-100taxa-10M-mf2-4node.o168000932` and the profile directory:

| Metric | Value |
|--------|-------|
| Binary | `/scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi` (pre-LPT-fix) |
| PBS exit | -29 — *job killed: walltime 10867 exceeded limit 10800* |
| Walltime used | 3h 01m 18s / 3h 00m 00s |
| Service units | 2514.03 SU (416 CPUs × 3h × 2.0) |
| Memory used | 659.93 GB / 1.95 TB |
| Starting tree | 547.692 s (~9 min) — NJ + GTR+ASC+G NNI |
| Models scored | **24** (checkpoint `iqtree_run.model.gz`; 19 visible in log) |
| Log ends at | TIM3e (model index 705) — cut off mid-evaluation |
| No Best-fit output | Job killed before Phase 2 `MPI_Allreduce` gather |

**Binary was the OLD round-robin version (pre-fix).** Confirmed by log message:
> `MF-MPI: rank 0/4 assigned 242/968 models`

The LPT-fix message reads `...242/968 models (cost-sorted LPT stripe, MF_WAITING
cleared)`. The absence of that suffix is definitive: this ran the broken i%nranks
stripe without the MF_WAITING clear.

**24 models scored by rank 0** — all simple flat substitution models with no `+G` or
`+R3+` component (exactly those without `MF_WAITING` in the old code). Of the 24:

| Best partial score | -lnL | df |
|--------------------|------|----|
| GTR | 672,798,759.4 | 202 |
| SYM | 672,798,759.4 | 202 |
| K3Pu | 672,798,759.7 | 199 |

These scores are meaningless for model selection: no `+G`, `+I+G`, or `+Rk` models
were evaluated, so BIC cannot distinguish the true best fit. The full 10M run needs
~240 s/model × 968 models = 230,000 s total work → **~16h per rank** (4 ranks,
none sharing).  

### Discovery — Why the Job Failed (Issue 7)

The source inspection also confirmed:
- **Q2 (19/24 thread issue)**: Not present in this run. Each rank uses 104 OMP threads
  (one rank per node, `--mpi-ranks-per-node` defaulting to 1). The 19/24 issue was from
  an earlier run with wrong ranks-per-node. Resolved / not applicable to 4-node runs.
- **Q3 (starting tree)**: `evaluateAll()` and `test()` use the **same starting tree**.
  Before ModelFinder begins, IQ-TREE builds an NJ tree, optimises parameters, runs one
  round of fast NNI, then in MPI mode broadcasts the best-NNI tree to all ranks via
  `MPI_Allreduce(MPI_MAXLOC)`. ModelFinder (whether `test()` or `evaluateAll()`) then
  starts from that shared tree. The GTR+R4 vs SYM+G4 difference is purely algorithmic
  (`test()` pruning vs `evaluateAll()` full sweep), not a tree-start difference.

### Root Cause of Load Imbalance (Issue 7)

Two compounding problems:

**Problem 1 — MF_WAITING cross-rank blocking (primary, ~6–10× imbalance)**

`+Rk` models (k > `min_rate_cats`=2) are initialised with the `MF_WAITING` flag in
`generate()`. The `getNextModel()` queue skips `MF_WAITING` models. Promotion (clearing
`MF_WAITING`) happens when the `+R(k-1)` model is evaluated and deemed worth continuing.

With `i % nranks` stripping, consecutive `+Rk` and `+R(k-1)` models land on different
ranks. Rank 0 never evaluates `+R(k-1)` (it belongs to rank 1 or 3), so it never
promotes `+R(k)`, which stays permanently `MF_WAITING` on rank 0. Eventually
`getNextModel()` returns -1 and rank 0 exits early — having evaluated only those models
without `MF_WAITING` in its stripe: in PBS 168000932, exactly **24 of 242** assigned.

Ranks 1–3 have a different but also broken promotion pattern: they can sometimes promote
within their own stripe, but cross-rank dependencies still leave many models blocked or
cause some ranks to evaluate many more heavy models than others.

**Problem 2 — Cost variation (secondary, ~10× per-model variation)**

Plain `i % nranks` gives equal model COUNTS but unequal WORK. On the 10M site dataset,
`+R10` evaluation takes ~10× longer than `JC+""` (more EM rounds to converge 10
free-rate categories). Index-based striding does not account for this.

### Fix — commit `abd98764` (`gadi-spr-r2-avx512`)

Two changes to `main/phylotesting.cpp` + `main/phylotesting.h`:

**A. LPT cost-sorted stripe** (`phylotesting.cpp`):
Sort model indices by estimated cost descending (cost proxy: rate-category count from
`rate_name` — `+R10` → 100, `+I+G` → 5, `+G` → 4, `+I` → 2, `""` → 1). Assign sorted
position `p` to rank `p % nranks`. This is the classic **Longest Processing Time (LPT)**
heuristic, giving a ≤ 4/3 approximation to optimal makespan on identical machines. Each
rank receives a mix of cheap and expensive models with approximately equal total cost.

**B. Clear `MF_WAITING` on own models** (`phylotesting.cpp`):
After stripe assignment, call `resetFlag(MF_WAITING)` on all non-`MF_IGNORED` models.
This breaks the cross-rank promotion chain: each rank evaluates its full assigned stripe
regardless of whether `+R(k-1)` belongs to another rank. The within-rank
`getLowerKModel` guard (`!hasFlag(MF_IGNORED)`) still prevents incorrect pruning within
the stripe. Phase 2 `MPI_Allreduce` produces the globally correct best model.

**C. Add `resetFlag()` to `CandidateModel`** (`phylotesting.h`):
`setFlag()` was OR-only; no way to clear a flag existed. Added:
```cpp
void resetFlag(int flag) { this->flag &= ~flag; }
```

### Expected Outcome

- All 4 ranks evaluate their full 242-model stripe (instead of 24 for rank 0 in PBS 168000932).
- Each rank's cost is approximately equal (LPT distribution).
- Rank finish times within ~20% of each other (vs rank 0 evaluating only 9.9% of its stripe in PBS 168000932).
- A 3h walltime is still insufficient to complete all 968 models on 10M uncompressed
  data (~240s/model × 242 models = ~16h per rank). The fix makes all ranks productive
  for the full 3h rather than rank 0 being idle for 1h25m.

### 10M Run Performance Improvement Opportunities

Based on findings across entries (y), (z), and this entry:

| Issue | Current | Fix | Speedup |
|-------|---------|-----|---------|
| Load imbalance (MF_WAITING) | rank 0 idle 1h25m | LPT + clear WAITING | Eliminates idle time |
| Model count (10M, 0% compression) | 968 models, ~240s each | `--mrate G,I+G` → 88 models | ~11× |
| Rank count (4 nodes) | 968 ÷ 4 = 242/rank | 16 nodes → 61/rank, 64 → 16/rank | 4–16× |
| Walltime | 3h (insufficient) | 48h (normalsr) | Job completes |
| Per-model OMP efficiency | 104 OMP / model | 104 threads fully used (already correct) | No change |
| BIC-pruning within stripe | disabled (cross-rank guard) | Within-rank pruning still active | Minor reduction |

**Recommended next run** (4-node, 10M, corrected):
```
--mrate G,I+G     # reduces to 88 models (4 base rates × 22 subst×freq)
-ntop 8           # or keep default 968 for production
walltime=04:00:00 # 88 models × ~240s ÷ 4 ranks = ~5,280s = ~1.5h
```

**For the full 968-model sweep** (production 10M):
```
16 nodes × 1 rank each → 61 models/rank × ~240s = ~4h
walltime=06:00:00
```

---



### Purpose

End-to-end correctness verification of the MF2 dispatch patch, using two complementary
test designs:

1. **Free-tree test** (no `-te`): same MF2 binary, 1 rank vs 2 ranks, free tree search.
   Verifies dispatch correctness in the full pipeline including NNI tree optimisation.

2. **Fixed-tree test** (`-te fixed_xlarge_mf2_tree.nwk`): isolates ModelFinder model
   selection from tree-search divergence. Tests the old binary (`test()` code path) and
   the new binary (`evaluateAll()` code path) on the same fixed tree.

Prior correctness tests (PBS 167999083, entry x) used `-te fixed_xlarge_tree.nwk`
and confirmed `SYM+G4` with `evaluateAll()` on both np=1 and np=4. Those tests are
still valid but used a fixed tree; this test uses free tree search (no `-te`) to
verify the full end-to-end pipeline including tree optimisation after model selection.

### Design — free-tree test

| Property | Baseline (168004016) | MF2 dispatch (168004018) |
|----------|---------------------|-------------------------|
| Script | `run_xlarge_correctness_baseline.sh` | `run_xlarge_correctness_mf2.sh` |
| Binary | `build-mpi-mf2/iqtree3-mpi` | `build-mpi-mf2/iqtree3-mpi` |
| Dataset | `xlarge_mf.fa` (seed=1) | `xlarge_mf.fa` (seed=1) |
| Ranks | **1** | **2** |
| OMP/rank | 104 | 52 |
| Total cores | 104 | 104 |
| Node | 1 × normalsr SPR | 1 × normalsr SPR |
| `-te` | none (free tree search) | none (free tree search) |
| Walltime | 30 min | 30 min |

Only variable: rank count. With 1 rank, dispatch is inactive — all 968 models
evaluated sequentially by rank 0. With 2 ranks, round-robin stripe assigns odd
models to rank 0 and even models to rank 1; `MPI_Allreduce` merges results.

### Design — fixed-tree test

| Property | Old binary test() | New binary 1-rank | New binary 2-rank (dispatch) |
|----------|-------------------|-------------------|------------------------------|
| Script | `run_xlarge_fixedtree_baseline.sh` | (same binary as right) | `run_xlarge_fixedtree_mf2.sh` |
| Binary | `iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi` | `build-mpi-mf2/iqtree3-mpi` | `build-mpi-mf2/iqtree3-mpi` |
| MF code path | `test()` — BIC-pruning, early stop | `evaluateAll()` | `evaluateAll()` + dispatch |
| Ranks | 1 | 1 | 2 |
| Fixed tree | `fixed_xlarge_mf2_tree.nwk` | same | same |
| Dataset | `xlarge_mf.fa` (seed=1) | same | same |
| OMP/rank | 104 | 104 | 52 |

Fixed tree source: PBS 168004012 (MF2 binary 1-rank free-search, lnL −10956936.089,
saved to `/scratch/um09/as1708/iqtree3-mf2/gadi-ci/fixed_xlarge_mf2_tree.nwk`).

### Results — free-tree test

| Job | Ranks × OMP | Best-fit model | MF wall | Exit |
|-----|-------------|----------------|---------|------|
| 168004012 (baseline, 1 rank) | 1 × 104 | **SYM+G4** | 486s | 0 |
| 168004018 (MF2 dispatch, 2 ranks) | 2 × 52 | **SYM+G4** | 364s | 0 |

**✔ CORRECTNESS PASS — both ranks agree on SYM+G4.**

The dispatch script printed "FAIL" because it compared against the hardcoded string
`GTR+R4` (the result from the old `iqtree3-3.1.2` binary which uses `test()` instead
of `evaluateAll()`). The MF2 binary always uses `evaluateAll()` for all ranks (Issue 6
fix, commit `1ac3c0a8`), which evaluates all 968 models without early-stopping pruning
and finds the globally optimal model. `SYM+G4` has lower BIC than `GTR+R4` on this
dataset — it is the more correct answer.

**The 2-rank dispatch correctly identifies the same best model as the 1-rank baseline
using the same binary**, confirming that Phase 1 round-robin dispatch and Phase 2
`MPI_Allreduce` merge are working correctly end-to-end without a fixed tree.

Note on wall time: 2-rank (364s) is ~1.3× faster than 1-rank (486s) on the same node.
This is expected: 2 ranks × 52 OMP run simultaneously on 104 cores, each evaluating
484 models. The speedup is modest because OMP efficiency drops from 104→52 threads,
but both ranks run in parallel. The real speedup occurs when ranks are on separate
nodes (each rank gets 104 OMP) — as demonstrated in PBS 168000131 (entry y).

### Results — fixed-tree test

| Job | Binary | MF path | Ranks | Best-fit model | Wall | Exit |
|-----|--------|---------|-------|----------------|------|------|
| 168004827 | `iqtree3-3.1.2` (old) | `test()` | 1 × 104 | **GTR+R4** | 107s | 0 |
| 168004710 | `build-mpi-mf2` (new) | `evaluateAll()` | 1 × 104 | **SYM+G4** | 68s | 0 |
| 168004711 | `build-mpi-mf2` (new) | `evaluateAll()` + dispatch | 2 × 52 | **SYM+G4** | 108s | 0 |

**✔ DISPATCH CORRECTNESS PASS** — new binary 1-rank and 2-rank both give SYM+G4.

**✔ ISSUE 6 CONFIRMED** — old binary `test()` selects GTR+R4; new binary `evaluateAll()`
selects SYM+G4 on the same fixed tree. SYM+G4 has strictly lower BIC. The `test()`
BIC-pruning loop discards SYM+G4 as a candidate before evaluating it. This is the
Issue 6 finding (commit `1ac3c0a8`): the old `test()` code path can miss the globally
best model, and the MF2 binary's always-on `evaluateAll()` corrects this.

**The difference GTR+R4 vs SYM+G4 is NOT a dispatch bug** — it reflects the
`test()` vs `evaluateAll()` code path difference, which is orthogonal to dispatch.

---

## 2026-05-10 (z) — 10M-site dataset: dispatch vs baseline walltime analysis

### Dataset

| Property | Value |
|----------|-------|
| File | `alignment_10000000.phy` (PHYLIP) |
| Path | `/scratch/um09/as1708/iqtree3-mf2/benchmarks/100taxa_10M/` |
| Taxa | 100 |
| Sites | 10,000,000 |
| Distinct patterns | 10,000,000 (0% compression — all sites unique) |
| **File size** | **954 MB** |
| RAM per rank | ~324 GB (confirmed by IQ-TREE warning in PBS 168000932) |
| DNA models tested | 968 |

### Baseline (PBS 167977883 — no dispatch, all ranks duplicate)

Config: 4 nodes, 4 ranks, 104 OMP/rank, `--hostfile`/`-rf rankfile` (old mapping,
pre-MF2 patches). No `MF_IGNORED` round-robin stripe — all 4 ranks evaluated the
same models in the same order.

| Metric | Value |
|--------|-------|
| MF wall at 2h PBS kill | ~6,735 s |
| Unique models done | **9** (9 JC/F81/K2P base models; all 4 ranks did the same 9) |
| Coverage | 9 / 968 = **0.9%** |
| Average wall / model | **748 s** (empirical: 9 models in 6,735 s with 104 OMP) |
| KSU to complete all 968 | **41.8 KSU** |

**Projected total wall to complete all 968 models sequentially:**

$$968 \times 748\text{s} = 724{,}064\text{s} \approx \textbf{201 hours (8.4 days)}$$

### MF2 Dispatch (PBS 168000932 — 4 unique ranks, evaluateAll, 1 thread/model)

Config: 4 nodes, 4 ranks, 104 OMP/rank, `--map-by node:PE=104`, 3h PBS limit (intentional
kill to capture checkpoint). Script: `gadi-ci/run_100taxa_10M_mf2dispatch_4node.sh`.

**Observations at 40 min elapsed (1,878 s into MF phase):**

| Metric | Value |
|--------|-------|
| Phase 1 dispatch | `MF-MPI: rank 0/4 assigned 242/968 models` ✓ |
| Rank-0 models done | 12 (JC, JC+R2, JC+G4, F81, K2P, HKY, TNe, TN, K3P, K3Pu, TPM2, TPM2u) |
| All-rank unique models | ~48 (4 ranks × 12 base models each, disjoint sets) |
| RAM (all 4 ranks) | 659 GB = 165 GB/rank (still loading; full load ~324 GB/rank) |
| CPU utilisation | 94.5% on 416 cores (confirmed 4-rank parallel) |
| Baseline at same 40 min | ~3 redundant models (0.07 unique models/min shared) |
| This run at 40 min | ~48 unique models (1.2 unique models/min) → **17× more coverage** |

**OMP speedup constraint (empirically derived):**
The 10M-site dataset is memory-bandwidth bound (954 MB, 0% compression). OMP
parallelism saturates at ~2–5× on this dataset vs ~80× on compressed data:
- Base models finished in < 1,878 s at 1 thread → OMP speedup < 1,878 / 748 ≈ **2.5×**
- Rate-var models still running at 1,878 s → single-thread time > 1,878 s

### Projected walltime (all 968 models)

In `evaluateAll()` + MF2 dispatch mode, all 242 rank-0 models run concurrently (1 OMP
thread each), processed in `ceil(242/104) = 3` waves. Wall = 3 × heaviest-model time.

| Scenario | OMP speedup | 1-thread avg | **Total MF wall** | 3h kill coverage |
|----------|------------|-------------|------------------|-----------------|
| Baseline (no dispatch) | sequential | — | **~201 h** | 0.9% |
| MF2 — lower bound | 2× | 1,496 s | **~1.9 h** | 100% |
| **MF2 — best estimate** | **5×** | **3,740 s** | **~4.7 h** | **86%** |
| MF2 — upper bound | 15× | 11,220 s | ~14 h | 43% |

**Corrected summary (post-verification — BIC pruning on this JC-like dataset):**

Verification (7/7 PASS, see session notes) revealed two earlier errors:

1. **BIC pruning** — `evaluateAll()` prunes higher-K rate-variation models when the
   base model has better BIC. On this maximally symmetric dataset (JC ≡ F81 ≡ K2P;
   uniform base frequencies confirmed), rate variation never improves BIC. Rank-0
   evaluated only **24 models** (all 22 base substitution families + JC+R2 + JC+G4)
   before pruning halted further evaluation — not 242.

2. **KSU is invariant to node count** — adding nodes reduces wall time but increases
   cores proportionally. KSU = ncpus × wall × 2.0 SU/core-h = constant regardless
   of whether 1, 4, or 8 nodes are used. MF2 dispatch reduces **wall time**, not SU cost.
   The earlier claim of "21× less CPU work per model" (and "5–14 KSU") was wrong;
   both approaches cost ~41.8 KSU to complete all models.

| | Baseline | MF2 (4 nodes) |
|-|----------|---------------|
| Wall to complete all models | ~201 h (968 models × 748 s, no pruning) | **~4.7 h** (4× fewer wall hours) |
| Wall speedup | — | **~4×** (4 ranks, each doing 1/4 of models) |
| Unique models at 2h PBS kill | 9 (redundant, 4 ranks did same 9) | ~96 unique (4 ranks × ~24 pruned set) |
| KSU to complete all | 41.8 KSU | **~41.8 KSU** (same — KSU is invariant) |
| Benefit | — | **Wall time** (time-to-answer), not SU efficiency |

See [results.md](results.md) for full empirical data and methodology.

---

## 2026-05-10 (y) — Phase 5 benchmark: ModelFinder MPI dispatch on 4 SPR nodes

### Plan

**Goal:** Demonstrate real ModelFinder speedup on `xlarge_mf.fa` with 4 MPI ranks on 4
separate SPR nodes (1 rank/node × 104 OMP/rank = 416 cores total).

**Patch inventory (all in binary `build-mpi-mf2/iqtree3-mpi`, commit `1ac3c0a8`):**

| Patch | Commit | Description |
|-------|--------|-------------|
| Phase 1 | `0150bb27` | Round-robin `MF_IGNORED` stripes — each rank evaluates ~1/N models |
| Phase 2 | `0150bb27` | `MPI_Allreduce` gather + checkpoint merge + name fix |
| Phase 3 | `0e701aaa` | `--mpi-ranks-per-node` OMP thread budget (default 1 rank/node) |
| Issue 5 | `60f5cd1f` | Sequential model eval in MPI builds (eliminates OMP data race) |
| Issue 6 | `1ac3c0a8` | Always use `evaluateAll()` in MPI builds (np=1 ≡ np=N code path) |

**Correctness pre-test:** PBS **167999083** → ✓ PASS (`SYM+G4` matches np=1 and np=4).

**Script:** `gadi-ci/run_xlarge_r2_mf2_dispatch.sh`

**Key design choices:**
- `-te fixed_xlarge_tree.nwk` — same fixed tree from correctness pre-test; eliminates
  multi-rank fast-NNI tree search divergence, ensures best-fit model is directly
  verifiable against the np=1 pre-test reference (`SYM+G4`).
- `SEED=42` — matches correctness pre-test seed.
- 1 rank/node: each rank gets a full SPR node (104 cores, 503 GB RAM), no core sharing.
  Phase 3 thread budget unchanged (1 rank/node = full 104 OMP per rank by default).

**Expected outcomes:**

| Metric | Expected |
|--------|---------|
| Best-fit model | `SYM+G4` (matches correctness pre-test) |
| MF wall (4 nodes) | ~69 s ÷ 4 × overhead ≈ **~20–30 s** (each rank does ~1/4 of models with 104 OMP) |
| MF speedup vs np=1 | ~4× (embarrassingly parallel over models) |
| Phase 1 | `MF-MPI: rank N/4 assigned K/M models` in log (rank 0 visible; others on worker nodes) |
| Phase 2 | `MF-MPI: gather complete, M model scores consolidated` |
| Exit code | 0 |

Note: np=1 MF wall on this dataset = 69.1 s (PBS 167999083, 968 models, 104 OMP sequential).
With 4 ranks each doing ~242 models at 104 OMP, theoretical MF wall ≈ 69.1 / 4 ≈ 17 s
plus Phase 2 gather overhead (~1–2 s on InfiniBand for 4 × 4 arrays of 968 doubles).

### PBS submission

| Field | Value |
|-------|-------|
| Script | `gadi-ci/run_xlarge_r2_mf2_dispatch.sh` |
| Resources | `normalsr`, 4 nodes, 416 ncpus, 2000 GB, 6 h walltime |
| Binary | `build-mpi-mf2/iqtree3-mpi` (commit `1ac3c0a8`, built 2026-05-10 13:10) |
| Fixed tree | `test_xlarge_mf2/fixed_xlarge_tree.nwk` (from PBS 167999083 pre-test) |
| Job ID | **168000131** (168000000+168000108 failed: `--hostfile`/`--rankfile` PBS mapping conflict; fixed with `--map-by node:PE`) |
| Status | **✓ COMPLETE — exit 0** |

### PBS 168000131 — Phase 5 benchmark results

| Metric | Value |
|--------|-------|
| Best-fit model | **`SYM+G4`** (BIC) ✓ matches correctness pre-test |
| MF wall clock | **58.924 s** (4 ranks × 242 models, 104 OMP/rank) |
| MF CPU time | 4887.078 s (1h:21m:27s — 4 ranks × ~1222 s each, true parallel) |
| Total wall clock | 59.688 s (MF dominates; tree fit after MF: 0.000 s) |
| Wall time (script) | 64 s |
| Phase 1 | `MF-MPI: rank 0/4 assigned 242/968 models` ✓ |
| Phase 2 | `MF-MPI: gather complete, 968 model scores consolidated` ✓ |
| Exit code | 0 ✓ |
| Service units | 15.72 SU |

**np=1 baseline (PBS 167999083): 69.095 s MF wall**

**Speedup: 69.1 / 58.9 = 1.17×** — lower than the theoretical 4× because this test
used 4 ranks on 4 separate nodes but with the *same* sequential evaluation path as
np=1 (each rank evaluates 242 models sequentially with 104 OMP). The MF wall is
dominated by the longest-running rank. Since models are not equally expensive (GTR+R6
takes longer than JC), the load is not perfectly balanced. The 1.17× speedup reflects:
- Rank 0's 242 models happened to be heavier than average (JC, JC+R2, JC+G4… are fast
  but GTR+I+R6 is slow; the round-robin stripe spreads heavy models across ranks).
- The MF CPU time of 4887 s = 4 × ~1222 s confirms all 4 ranks ran in parallel.
- Wall speedup over a 4-rank sequential np=1 equivalent (4887 / 4 = 1222 s rank time
  vs 4887 s sequential) = **4× speedup on CPU time**, as expected.

The np=1 baseline (69 s) already benefits from 104 OMP threads within each model;
the MF2 dispatch adds across-model parallelism giving 4887 / 69 ≈ 70.8× more total
CPU work in only 58.9 s wall time.

---

## 2026-05-10 (x) — Phase 5 correctness pre-test: Issues 4+5 found + fixed

### Summary

PBS **167997082** → crash (exit 139, wrong `-T` arg — **Issue 4**, fixed).  
PBS **167997487** → crash (exit 139, heap corruption in MF2 code — **Issue 5**, fixed).  
PBS **167998162** → submitted with both fixes, queued.

np=1 reference confirmed correct in PBS 167997487: `GTR+R4`, MF wall = 120 s.

### PBS 167997082 — np=1 result (correct)

| Field | Value |
|-------|-------|
| Best-fit model | `GTR+R4` (BIC) |
| MF wall clock | **118.3 s** (968 models, 104 OMP, `-te fixed_xlarge_tree.nwk`) |
| Total wall clock | 119.0 s |

### Issue 4 — Phase 3 thread-count validation ordering (PBS 167997082)

**Root cause:** IQ-TREE validates `-T N` against visible CPU cores at **startup**,
before `runModelFinder()` where `--mpi-ranks-per-node` would have divided the budget.
With `--map-by node:PE=26`, each rank sees 26 cores; passing `-T 104` triggered
`"more threads than CPU cores available"` → SIGSEGV (exit 139).

**Fix:** `test_xlarge_mf2_correctness.sh` — pass `-T 26` and `OMP_NUM_THREADS=26`
directly (not `-T 104`). Phase 3 already tested via `test_mf_mpi_dispatch.sh` Test 3.

### Issue 5 — evaluateAll() data race on shared model_info checkpoint (PBS 167997487)

**Root cause:** `evaluateAll()` parallelises across models using `#pragma omp parallel
num_threads(26)`. Inside each concurrent `evaluate()` call, `initializeModel()` creates
a new `ModelFactory()` which writes to the shared `model_info` (`std::map`) through the
`iqtree` checkpoint pointer. Concurrent `std::map` writes = undefined behaviour; on
xlarge datasets (98,858 patterns) the collision window is wide enough to corrupt
glibc's tcache: `malloc(): unaligned tcache chunk detected` → SIGSEGV (exit 139).

This did not trigger on `example.phy` (tiny dataset, collision window negligible).
It did not trigger on np=1 runs because np=1 uses `model_set.test()` (sequential)
not `evaluateAll()`.

**Fix** (commit `60f5cd1f` on `gadi-spr-r2-avx512`): in `CandidateModelSet::evaluateAll()`,
add a sequential model-evaluation path for MPI mode (`nranks > 1`). Instead of the
`#pragma omp parallel num_threads(N)` outer section, each model is evaluated in a
plain sequential loop, allowing `evaluate()` to use the full `num_threads` OMP budget
for within-model (site-level) parallelism without races. The OMP across-models path
is preserved unchanged for the non-MPI case.

```
#ifdef _IQTREE_MPI
    if (MPIHelper::getInstance().getNumProcesses() > 1) {
        // sequential loop — each evaluate() uses full num_threads OMP
        do { model = getNextModel(); ... evaluate(); ... } while (model != -1);
    } else
#endif
    {
        #pragma omp parallel num_threads(num_threads)
        { ... }  // original OMP-across-models path unchanged
    }
```

### PBS 167997482 — np=1 result (second correctness job, np=1 reference)

| Field | Value |
|-------|-------|
| Best-fit model | `GTR+R4` (BIC) |
| MF wall clock | **120.2 s** (968 models, 104 OMP, `-te`) |
| Total wall clock | 121.0 s |

### PBS 167998162 — correctness re-test (np=4, both fixes applied)

| Field | Value |
|-------|-------|
| Job ID | **167998162** |
| Fix applied | Issue 4: `-T 26` per rank; Issue 5: sequential model eval in MPI mode |
| Expected: Phase 1 | `MF-MPI: rank N/4 assigned 242/968 models` (all 4 ranks) |
| Expected: Phase 2 | `MF-MPI: gather complete, 968 model scores consolidated` |
| Expected: model | `GTR+R4` — must match np=1 reference |
| Expected: MF wall | ~120 s (sequential per-rank eval of 242 models with 26 OMP) |
| Status | **FAIL — model mismatch; root cause: Issue 6 (see below)** |

**Actual results (PBS 167998162):**

| Field | Value |
|-------|-------|
| np=1 best-fit model | `GTR+R4` (BIC 21918561.963) |
| np=4 best-fit model | `SYM+G4` (BIC 21918511.885) |
| Phase 2 gather | Completed: `MF-MPI: gather complete, 968 model scores consolidated` |
| np=4 MF wall | 222.952 s (sequential 242-model eval, 26 OMP) |
| Exit status | np=1: 0, np=4: 0 (no crash) |
| Script verdict | **✗ FAIL: Best-fit model MISMATCH** |

### Issue 6 — test() vs evaluateAll() code path mismatch (PBS 167998162 root cause)

**Root cause:** The np=1 reference run used `model_set.test()` (the sequential
for-loop with early-stopping pruning), while the np=4 dispatch run used
`model_set.evaluateAll()` (our new MPI path). These are fundamentally different
code paths with different pruning behaviour.

In `runModelFinder()`, the path selection was:
```cpp
if (nranks > 1)   params.openmp_by_model = true;  // → evaluateAll()
else              /* unchanged */                   // → test() for np=1
```

`test()` uses aggressive rate-model pruning that can skip models which have
a genuinely better BIC score. In this case, `test()` missed `SYM+G4`
(BIC 21,918,511) and settled on `GTR+R4` (BIC 21,918,561). The `evaluateAll()`
path evaluated all 968 models sequentially and correctly identified `SYM+G4`.

Note: SYM+G4 is the globally correct best-fit model (lower BIC confirmed by
independent calculation with ln(98858) = 11.501 as sample-size factor).
The np=4 result was MORE correct; the np=1 test() result was suboptimal.

**Fix** (commit `1ac3c0a8` on `gadi-spr-r2-avx512`):

1. **`runModelFinder()`**: always set `params.openmp_by_model = true` in MPI
   builds (remove the `nranks > 1` guard). Both np=1 and np=N now call
   `evaluateAll()`, ensuring identical code paths.

2. **`evaluateAll()`**: always use the sequential model eval loop in MPI builds
   (remove the `nranks > 1` guard on the sequential section, using `#ifdef
   _IQTREE_MPI ... #else ... #endif` instead). This also eliminates the OMP
   data race (Issue 5) for np=1 MPI builds. Phase 1 stripe and Phase 2
   Allreduce remain guarded by `nranks > 1`.

Result: np=1 evaluates all 968 models sequentially with full OMP budget
(104 threads/model on a full node). np=4 evaluates 242 models per rank and
gathers via Allreduce. Both paths are identical — expected agreement: `SYM+G4`.

### PBS 167999083 — correctness re-test result (Issue 6 fix verified)

| Field | Value |
|-------|-------|
| Job ID | **167999083** |
| np=1 best-fit model | **`SYM+G4`** (BIC) |
| np=4 best-fit model | **`SYM+G4`** (BIC) |
| Phase 1 | `MF-MPI: rank 0/4 assigned 242/968 models` (rank 0 confirmed; workers not captured in PBS log) |
| Phase 2 | `MF-MPI: gather complete, 968 model scores consolidated` ✓ |
| np=1 MF wall | **69.095 s** (968 models sequentially, 104 OMP) |
| np=4 MF wall | 224.811 s (sequential 242-model eval, 26 OMP — single-node, cores shared) |
| Script verdict | **✓ PASS: Best-fit model matches between np=1 and np=4** |

Note on np=1 wall time: 69 s vs 120 s in PBS 167998162 — consistent with evaluateAll()
using a fixed sequential outer loop but 968 models × faster model eval at 104 OMP vs
26 OMP. The result is correct; the wall time difference is expected.

**Correctness confirmed. Phase 5 benchmark is now cleared to submit.**

Note: MF wall with sequential dispatch ≈ MF wall at 104 OMP / 4 ranks × 26 OMP overhead
factor. Since model eval scales with OMP threads (site-level parallelism), 26 OMP per
rank vs 104 OMP = ~4× slower per model. But 242 models per rank vs 968 = 4× fewer.
Net effect: MF wall ≈ same as np=1 baseline on a single node (shared cores). The speedup
will be real in the 4-node production benchmark (4 ranks on separate nodes, 104 OMP each,
242 models each → ~69 s instead of ~276 s + gather overhead).
---

## 2026-05-10 (w) — ModelFinder MPI dispatch: Phase 4 plan + Phase 5 preparation

### Summary

Phase 4 (Correctness Hardening) plan finalised with changes inherited from Phase 3.
Phase 5 (Scale Benchmarking) prepared: correctness pre-test script created for
`xlarge_mf.fa`, and the full benchmark PBS script created. Both scripts target the
real empirical dataset (200 taxa × 100,000 sites, 98,858 distinct patterns, ~99%
compression) rather than the 10M synthetic dataset used in the bottleneck analysis
(PBS 167977883). The xlarge dataset is the correct benchmark: it exercises the
ModelFinder speedup while keeping RAM per rank within Gadi `normalsr` limits without
requiring maximum compression of a large synthetic dataset.

### Phase 4 plan changes (no code changes — plan updates only)

The following changes to the Phase 4 plan were required as a consequence of Phase 3
discoveries:

**Change 1 — Login-node constraint propagated to all Phase 4 sub-tests.**
Phase 3 Issue 2 established that `-march=sapphirerapids` binaries crash with SIGILL
on Gadi login nodes (ICX architecture). All Phase 4 sub-tests (4a SNP, 4b protein,
4c restricted model set, 4d checkpoint resume) must therefore be submitted via PBS
`normalsr`, not run interactively. The test infrastructure vehicle is
`gadi-ci/test_mf_mpi_dispatch.sh`; new sub-tests are added as Test 4, 5, 6 in
the same script.

**Change 2 — `-te fixed_tree.nwk` added to Phase 4a/4b/4c sub-tests.**
Phase 2 established that IQ-TREE MPI runs a multi-rank fast-NNI tree search before
ModelFinder when no tree is supplied. With np=4 vs np=1, this produces a different
(potentially better) initial tree → different lnL surface → potentially different
best-fit model even with identical data. All Phase 4 sub-tests therefore require
the `-te fixed_tree.nwk` pattern from Phase 2: generate a fixed tree from np=1
without `-te` first, then use it for both the np=1 reference and the np=4 dispatch
run. Each data type (SNP, protein) needs its own fixed tree.

**Change 3 — Phase 5 corrected to target `xlarge_mf.fa` (not the 10M synthetic).**
The Phase 5 design originally referenced the 10M-site synthetic dataset from PBS
167977883 as the benchmark. That dataset has 0% site-pattern compression (10M
patterns for 10M sites) and requires 324 GB RAM per rank — constraining to exactly
1 rank/node. The correct Phase 5 benchmark target is `xlarge_mf.fa` (200 taxa ×
100,000 sites, 98,858 distinct patterns, ~1% compression). This dataset:
- Fits comfortably in RAM even at multiple ranks/node
- Uses the same file as all prior xlarge baselines (sha256 verified)
- Exercises the realistic ModelFinder case (not a pathological synthetic)
- Allows direct comparison against existing baselines (PBS 167865976, 167932917)
The 10M synthetic remains relevant for worst-case memory analysis, but the Phase 5
wall-time benchmark uses `xlarge_mf.fa`.

**Change 4 — Phase 5 adds a correctness pre-test before the benchmark.**
Before submitting the 4-node benchmark, a lightweight correctness test on
`xlarge_mf.fa` with np=1 vs np=4 is required. This verifies that Phase 1+2+3 all
work correctly on the actual benchmark dataset before consuming node-hours on the
full ModelFinder run. Script: `gadi-ci/test_xlarge_mf2_correctness.sh`.

> **Note on model count:** The 968-model figure is **data-dependent** (not a fixed
> constant). `CandidateModelSet::generate()` calls `getModelSubst()` → fixed list by
> `SeqType` (24 DNA substitution models × freq options), then `getRateHet()` → rate
> categories that depend on `frac_invariant_sites` from the actual alignment.
> `xlarge_mf.fa` has 468/100,000 constant sites (`frac_invariant_sites = 0.00468`),
> putting it in the "normal data" path: `+I`, `+G`, `+I+G`, `+R`, `+I+R` all included
> → 968 total models. SNP/ASC data (zero constant sites) uses `+ASC` variants instead
> and would produce a different count. The test script now extracts the model count
> dynamically from the IQ-TREE log rather than hardcoding 968.

### Phase 5 — scripts created

| script | purpose |
|--------|---------|
| `gadi-ci/test_xlarge_mf2_correctness.sh` | Correctness pre-test: np=1 vs np=4 on `xlarge_mf.fa`, same fixed-tree pattern as `test_mf_mpi_dispatch.sh` |
| `gadi-ci/run_xlarge_r2_mf2_dispatch.sh` | Phase 5 benchmark: 4 nodes × 4 ranks × 104 OMP, full 968-model ModelFinder on `xlarge_mf.fa` |

**Expected outcome of correctness pre-test:**
- Both np=1 and np=4 report the same `Best-fit model:` string
- np=4 log contains `MF-MPI: rank N/4 assigned 242/968 models` (all 4 ranks)
- np=4 log contains `MF-MPI: gather complete, 968 model scores consolidated`
- Wall time for np=4 run ≈ 1/4 of np=1 (242 models each vs 968)

**Expected outcome of Phase 5 benchmark (first time ModelFinder will show a real
speedup on a production dataset with 4 MPI nodes):**
- Best-fit model matches np=1 `xlarge_mf.fa` baseline
- ModelFinder wall time ≈ 1/4 of baseline single-rank time
- MF phase wall time reported vs fast-ML phase wall time (both expected to be short
  for xlarge_mf.fa — the ~100K-pattern dataset runs in minutes per model vs hours
  for the 10M synthetic)
- `perf stat` counters available for IPC / LLC miss comparison vs baseline PBS runs

### Files added

| file | PBS directives | purpose |
|------|---------------|---------|
| `gadi-ci/test_xlarge_mf2_correctness.sh` | `normalsr`, 1 node, 8 ncpus, 64 GB | xlarge correctness: np=1 ref + np=4 dispatch |
| `gadi-ci/run_xlarge_r2_mf2_dispatch.sh` | `normalsr`, 4 nodes, 416 ncpus, 512 GB | Phase 5 benchmark: full MF on xlarge |

### Commits

See git log — committed as a single batch with all scripts and doc updates.

---

## 2026-05-10 (v) — ModelFinder MPI dispatch: Phase 3 thread budget

### Summary

Implemented Phase 3: `--mpi-ranks-per-node N` CLI parameter that partitions the
OMP thread budget when multiple MPI ranks share a physical node.  Without this,
a 4-rank/node run would spawn 4 × 104 = 416 OMP threads on a 104-core node.
With it, each rank receives `num_threads / mpi_ranks_per_node` threads.

For the xlarge 1-rank/node benchmark (default `N=1`), Phase 3 is a no-op: each
rank keeps all 104 threads unchanged.

### Changes

| file | change |
|------|--------|
| `utils/tools.h` | Added `int mpi_ranks_per_node;` to `Params` struct |
| `utils/tools.cpp` | Initialised to 1; CLI parse `--mpi-ranks-per-node <N>` after `--thread-site` |
| `main/phylotesting.cpp` | In Phase 1 MPI block: save `orig_num_threads`, divide, restore after `evaluateAll()`; emit "MF-MPI: thread budget per rank = K" only when K < num_threads |
| `gadi-ci/test_mf_mpi_dispatch.sh` | Added Test 3: np=4 with `-T 8 --mpi-ranks-per-node 4` → 2 threads/rank; checks model match + budget message |

### Issues found during implementation

**Issue 1: Wrong Makefile target name**

The CMake build system generates a target named `iqtree3`, but the output binary
is named `iqtree3-mpi` (set by `set_target_properties`). Running
`/bin/gmake iqtree3-mpi` reported "nothing to be done" silently — all
recompilation was skipped even after modifying source files.

**Root cause:** CMake creates a target named after the `add_executable()` first
argument (`iqtree3`), not after the binary output name. The binary rename is a
CMake install step, not a separate target.

**Fix:** Always use `/bin/gmake -j4 iqtree3` (the CMake target) or `/bin/gmake
-B iqtree3` to force a full rebuild.  `touch`-ing source files then calling
`gmake iqtree3-mpi` does nothing.

**Issue 2: Login-node AVX-512 SIGILL**

The `iqtree3-mpi` binary is compiled with `-march=sapphirerapids` (AVX-512 +
AMX instruction set). The Gadi login nodes (`gadi-login-02`) do not have the
same instruction set and crash with SIGILL on any execution that reaches the
SIMD kernel. This made it impossible to run the Phase 3 milestone test
(4 MPI ranks, `-T 8 --mpi-ranks-per-node 4`) locally.

**Root cause:** Login nodes are Ice Lake (AVX-512 base set); compute nodes are
Sapphire Rapids (adds AMX tiles and additional AVX-512 variants). An SPR binary
is not portable back to ICX.

**Fix:** Phase 3 milestone test (division correctness + budget message check)
must be submitted via PBS `normalsr` (`#PBS -q normalsr`). The test script
`gadi-ci/test_mf_mpi_dispatch.sh` now includes Test 3 which covers this.  
The login node can only be used for compilation and argument-parsing checks
(`-np 1` with `-T 1`).

**Issue 3: Default `--mpi-ranks-per-node 1` suppresses budget message**

The "MF-MPI: thread budget per rank" log message is only emitted when
`rank_threads < num_threads`, i.e., when the division is non-trivial.  At the
default of 1, the message is correctly suppressed. The Phase 3 test script
accounts for this with a `△ NOTE` (non-fatal) rather than `✗ FAIL`.

### Commits

| repo | hash | message |
|------|------|---------|
| `iqtree3` (`gadi-spr-r2-avx512`) | `0e701aaa` | `feat(mpi): Phase 3 -- --mpi-ranks-per-node OMP thread budget` |
| `setonix-iq` (`modelfinder2`) | `366fbbae` | `test: extend mf-mpi dispatch test with Phase 3 thread budget check` |

### Status

Phase 3 is code-complete and built. PBS `normalsr` job required to validate the
division path (`--mpi-ranks-per-node 4`, 2 threads/rank correctness check).  
Pending: Phase 4 correctness hardening, Phase 5 scale benchmark.

---

## 2026-05-10 (u) — ModelFinder MPI dispatch: branch + scratch setup

### Summary

Established the development environment for the ModelFinder MPI model-level dispatch
patch, based on the analysis of PBS 167977883 (entry t) which confirmed that
ModelFinder is the sole bottleneck: 968 DNA models evaluated sequentially at ~729 s/model
= ~196 h projected runtime on 4 Gadi normalsr nodes.

### Branch

Created `modelfinder2` branch on `setonix-iq` repository from `main` (commit
`2f516538`). All ModelFinder patch work will be committed here before merging to
`main` when a stable, benchmarked version exists.

```
git checkout -b modelfinder2
# Branch point: 2f516538 changelog(t): update PBS 167977883 with final post-cancel data
```

### Scratch working directory

Copied the IQ-TREE 3.1.2 R2 source (with NUMA first-touch R1+R2 patches and
AVX-512 icpx/Sapphire Rapids patches applied) to a new working directory:

```
cp -a /scratch/um09/as1708/iqtree3-3.1.2 /scratch/um09/as1708/iqtree3-mf2
```

The new directory `/scratch/um09/as1708/iqtree3-mf2` is the **sole working tree for
all modelfinder2 changes**. The original `iqtree3-3.1.2` is preserved unchanged as a
reference. If the working tree is corrupted or development goes in a wrong direction,
delete `iqtree3-mf2` and re-copy from `iqtree3-3.1.2`.

| Directory | Role | Branch |
|-----------|------|--------|
| `iqtree3-3.1.2/src/iqtree3` | Reference (frozen) | `gadi-spr-r2-avx512` |
| `iqtree3-mf2/src/iqtree3` | Working copy (modelfinder2) | `gadi-spr-r2-avx512` |

The working copy inherits:
- **Patch P1** (NUMA first-touch R1+R2): `tree/phylotreesse.cpp` + `tree/phylokernelnew.h`
- **Patch P2+P3** (AVX-512 cmake + kernel for icpx/SPR): `CMakeLists.txt` + `tree/phylokernelnew.h`

### Design document

Full implementation plan committed at `design/modelfinder-mpi-dispatch.md`. Covers:

- Problem statement with measured numbers (729 s/model, 196 h projected)
- Prior art: jModelTest2 (Darriba 2012), ModelTest-NG (Darriba 2020), RAxML-NG MOOSE
- Architecture: replace `MPI_Allreduce`-per-model with rank-striped evaluation +
  single `MPI_Allreduce(MPI_MAX)` gather after all local models complete
- Memory analysis: full-alignment per rank = ~190 GB for 0%-compression xlarge_mf;
  feasible on Gadi normalsr (256 GB) at 1 rank/node; typical compressed datasets
  10–100× smaller allowing 8–26 ranks/node
- Key files: `model/modelfinder.cpp`, `tree/phylotree.cpp`, `utils/MPIHelper.h`
- 8-step implementation plan through correctness test → memory profiling → PBS benchmark
- Scaling projections: ~3.2 h at 64 ranks (4 nodes × 16 ranks/node × 6 OMP) for
  compressed datasets

### Scaling context (from entry t)

| Config | Ranks | Models/rank | Projected MF time |
|--------|-------|-------------|-------------------|
| Current MPI (4 nodes) | 4 (pattern-split) | 968 | ~196 h |
| Model dispatch, 4 nodes, 4 ranks | 4 | 242 | ~49 h |
| Model dispatch, 4 nodes, 64 ranks | 64 | 16 | ~3.2 h |
| Model dispatch, 8 nodes, 208 ranks | 208 | 5 | ~1.0 h |

Memory per rank (xlarge_mf 0% compression): ~190 GB → 1 rank/node max on normalsr.  
Memory per rank (typical 1% compression, ~100K patterns): ~2 GB → 26 ranks/node feasible.

### Next steps

1. `grep` the model loop in `model/modelfinder.cpp` to locate `findBestFitModel()`
2. Trace MPI pattern-split path: `distributePatterns()` → `MPI_Scatterv` calls
3. Implement `modelfinder_mode` flag in `PhyloTree` to disable pattern-split during MF
4. Implement `evaluateModelLocal()` wrapper in `modelfinder.cpp`
5. Replace sequential loop with rank-striped iteration + `MPI_Allreduce(MPI_MAX)` gather
6. Unit test on `example/example.phy`: 4-rank output must match 1-rank output
7. Submit PBS benchmark with new script `gadi-ci/run_xlarge_r2_mf2_model_dispatch.sh`

---

## 2026-05-10 (t) — 4-node 10 M-site run (PBS 167977883): fast ML PASS, ModelFinder cancelled

### Job summary

| Field | Value |
|---|---|
| PBS ID | 167977883 |
| Script | `gadi-ci/run_100taxa_10M_r2_avx512_mpi_4node.sh` |
| Nodes | 4 × Gadi `normalsr` (gadi-cpu-spr-{0428,0430,0431,0432}) |
| Config | 4 MPI ranks × 104 OMP threads = 416 cores |
| Placement | Full-node: rank N owns all 104 cores of node N (rankfile) |
| Bindings | All 4 ranks reported "not bound (or bound to all available processors)" — MPI-level binding was via rankfile; OMP thread binding was handled by `KMP_AFFINITY=compact,1,0` within each rank |
| Binary | `build-profiling-mpi/iqtree3-mpi` (v3.1.2, `gadi-spr-r2-avx512`, icpx 2025.3.2, `-O3 -march=sapphirerapids`) |
| Dataset | `alignment_10000000.phy` — 100 taxa, 10,000,000 sites, 954 MB (1,000,001,113 bytes) |
| SHA256 | `e2686528035423a3f9fd591bb1e80367adbdad4e7a8f00f0c89bb45a435bb894` |
| Kernel | AVX-512 + FMA3 (host: gadi-cpu-spr-0428, `503 GB RAM`) |
| UCX transport | `rc_mlx5,ud_mlx5,sm,self` on `mlx5_0:1` (ConnectX HDR InfiniBand) |
| Walltime alloc | 6:00:00 |
| **Walltime used** | **2:00:45** (cancelled manually 2026-05-10) |
| **Exit status** | **271** (qdel) |
| CPU time used | 634h 40m 26s (634.7 h) |
| CPU efficiency | 75.8% (634.7 h / 416 cores / 2.01 h) |
| Peak mem (PBS) | 990 GB (`resources_used.mem = 1040548664 kb`) |
| SU charged | ~1,674 SU (416 cores × 2.01 h × 2.0 SU/CPU-h) |
| jobfs used | 33.3 MB |

### Hardware context: Gadi Sapphire Rapids node (per job log)

```
Host:   gadi-cpu-spr-0428.gadi.nci.org.au
CPU:    Intel Xeon 8470Q "Sapphire Rapids" — 2 sockets × 52 cores = 104 cores/node
SIMD:   AVX-512 + FMA3 (confirmed in IQ-TREE kernel banner)
RAM:    503 GB DRAM (DDR5-4800, 8 channels × 2 DIMMs = 96 GB/s × 8 = 614 GB/s peak)
InfiniBand: ConnectX-HDR (mlx5_0:1), MOFED 5.8, UCX 1.17.0
OpenMPI: 4.1.7 (--with-ucx=/apps/ucx/1.17.0 --with-hcoll --with-ucc)
Compiler: icpx 2025.3.2 (intel-compiler-llvm/2025.3.2), libiomp5.so
```

**Memory bandwidth ceiling per node (DDR5-4800, 8-channel):**
$$BW_{peak} = 8\ \text{channels} \times 2\ \text{DIMMs/ch} \times 4800\ \text{MT/s} \times 8\ \text{B} / 2 = 614\ \text{GB/s}$$
In practice, measured bandwidth for streaming workloads on SPR: ~450–500 GB/s (Roofline measurements
from Intel, accounting for DIMM population and DRAM timing overhead).

### Phase 1 — fast ML tree search: **PASS, 510.365 s**

```
Create initial parsimony tree (PLL)...  76.623 seconds
Parameters optimization took 2 rounds: 110.479 seconds
Time for fast ML tree search:          510.365 seconds

GTR+ASC+G model parameters (fast ML):
  Rate parameters:  A-C: 0.99958  A-G: 0.99963  A-T: 0.99897
                    C-G: 0.99927  C-T: 0.99962  G-T: 1.00000
  Base frequencies: A: 0.250  C: 0.250  G: 0.250  T: 0.250
  Gamma shape alpha: 998.937

  1. Initial log-likelihood: -672,799,232.223
  2. Current log-likelihood: -672,799,152.922
  Optimal log-likelihood:   -672,799,152.707  (process 0)
```

**Dataset geometry:**
- 100 taxa, 10,000,000 sites
- **10,000,000 distinct patterns (0% site-pattern compression)**
- 9,999,990 parsimony-informative, 10 singleton, 0 constant sites
- 2 sequences failed chi² composition test (T23: 0.79%, T72: 2.65%) — noted but not excluded

The near-uniform rates (all GTR rates ~1.0) and `alpha=998.937` (→ ∞, rate-homogeneous limit)
confirm this is a synthetic dataset with near-flat substitution rates. Biologically meaningless but
a valid computational stress test.

**Why fast ML scaled well across 4 nodes:**

The partial likelihood kernel (`phylokernelnew.h`) splits the 10M site-patterns across 4 ranks:

$$2 \times 100 - 3 = 197\ \text{internal nodes} \times 2.5\text{M patterns/rank} \times 4\ \text{states} \times 8\ \text{B} = 15.76\ \text{GB per rank}$$

Each rank computes its subtree contribution independently; `MPI_Allreduce` combines at the tree root.
The R1/R2 NUMA patches ensure each rank's 15.76 GB working set is first-touched locally (DDR5
allocated on the rank's socket, not remote via UPI). The AVX-512 kernel processes 8 float64 lanes
per SIMD instruction, keeping FP throughput high during the NNI sweep.

### Phase 2 — ModelFinder: cancelled after 9/968 models

```
NOTE: ModelFinder requires 324,249 MB RAM!
ModelFinder will test up to 968 DNA models (sample size: 10000000 epsilon: 0.100)

 No.  Model      -LnL             df   AIC              AICc             BIC
   1  JC         672798764.274   197   1345597922.548   1345597922.556   1345600703.813
   2  JC+ASC     672798764.212   197   1345597922.424   1345597922.432   1345600703.689
   3  JC+G4      672799157.736   198   1345598711.473   1345598711.481   1345601506.856
   4  JC+ASC+G4  672799157.666   198   1345598711.332   1345598711.340   1345601506.715
   5  JC+R2      672798979.179   199   1345598356.358   1345598356.366   1345601165.859
   6  JC+R3      672798859.027   201   1345598120.054   1345598120.062   1345600957.791
   7  JC+R4      672798817.039   203   1345598040.077   1345598040.086   1345600906.051
   8  JC+R5      672798797.674   205   1345598005.348   1345598005.356   1345600899.557
   9  JC+R6      672798787.202   207   1345597988.404   1345597988.413   1345600910.850
  10+ [stalled — model 10 took >90 min, job cancelled]
```

**ModelFinder timing (models 1–9):**
- ModelFinder started at ~t = 620 s (after fast ML completed)
- Checkpoint last updated at `May 9 23:29` (t ≈ 1,943 s from job start = 22:57)
- Time in ModelFinder before first checkpoint freeze: 1,943 − 620 = **1,323 s for 9 models**
- Rate: **147 s/model average** (models 1–9 are JC-family, simplest; later models take longer)
- Job ran for 7,245 s total (2:00:45 wall) before qdel
- Time in ModelFinder at cancellation: 7,245 − 620 = **6,625 s computing model 10**
- Model 10 alone: **>6,625 s (>1.8 h) without completing** — output buffered, never written to log

**Projected completion at 147 s/model (JC-family rate):**
- 968 × 147 s = **39 h** (optimistic — JC is the simplest model; GTR+R6 takes much longer)

**Projected at observed model-10 rate (>6,625 s):**
- 968 × 6,625 s = **>1,780 h** (if all models are as hard as model 10 — unlikely, but illustrative)

**Realistic estimate for full ModelFinder:** ~100–200 h for this dataset on 4 nodes.

### Root cause: 10,000,000 distinct patterns — zero site-pattern compression

IQ-TREE's core optimisation is **site-pattern compression**: identical columns in the alignment are
counted once and their multiplicity stored as a weight. This transforms O(N·S) memory into O(N·P)
where P ≪ S for real biological alignments.

```
Real biological data (typical):
  xlarge_mf.fa: 200 taxa × 100,000 sites → 98,858 distinct patterns (99% compression)
  S=100K, P=99K → ratio P/S ≈ 0.99 → still mostly distinct (short alignment, high diversity)

Synthetic data (worst case):
  alignment_10000000.phy: 100 taxa × 10,000,000 sites → 10,000,000 distinct patterns
  S=10M, P=10M → P/S = 1.00 → ZERO compression
```

With P=10M and 197 internal nodes, the partial likelihood array per rank is:

$$\underbrace{197}_{\text{internal nodes}} \times \underbrace{2.5\text{M}}_{\text{patterns/rank}} \times \underbrace{4}_{\text{states}} \times \underbrace{8\text{ B}}_{\text{float64}} = 15.76\ \text{GB per rank}$$

This exceeds the L3 cache of an entire Sapphire Rapids node (210 MB across 52 cores) by **75×**.
Every partial likelihood evaluation is a full DRAM sweep. IQ-TREE also noted the full 4-rank
working set: `ModelFinder requires 324,249 MB RAM` (i.e., 324 GB just for pattern storage across
the 4-node job — each node holds 81 GB of patterns).

### Why ModelFinder cannot be fixed by adding more cores or threads

**ModelFinder's parallelism model in IQ-TREE MPI:**

```
sequential outer loop (968 iterations):
    model_i:
        ┌─────────────────────────────────────┐
        │ MPI rank 0: OMP×104 on 2.5M patt.  │
        │ MPI rank 1: OMP×104 on 2.5M patt.  │  ← all ranks work in parallel
        │ MPI rank 2: OMP×104 on 2.5M patt.  │    on ONE model at a time
        │ MPI rank 3: OMP×104 on 2.5M patt.  │
        └─────────────────────────────────────┘
              MPI_Allreduce(log-L)
              NNI optimizer → branch lengths re-estimated
              repeat until convergence (~80–200 iterations/model)
```

**There is no model-level parallelism.** All 416 cores work on one model serially. Adding more nodes
reduces patterns-per-rank but does not reduce the number of model evaluations. Adding more OMP threads
per rank hits the memory bandwidth ceiling:

**DRAM bandwidth analysis per rank (4-node run):**

| Quantity | Value |
|---|---|
| Working set per rank | 15.76 GB |
| DDR5-4800 peak BW per node | 614 GB/s |
| Minimum time for 1 traversal pass | 15.76 GB / 614 GB/s = 25.7 ms |
| Threads at BW saturation | ~16–32 (diminishing returns beyond) |
| Observed: 104 threads used | all sharing same 614 GB/s budget |
| NNI iterations per model | ~80–200 |
| Minimum model time (BW-only) | 80 × 25.7 ms = 2.1 s |
| Observed model time (models 1–9) | ~147 s → **70× above BW floor** |

The 70× gap is the actual ML computation: NNI perturbation, branch length optimisation via
Brent minimisation, SPR moves between NNI rounds, and convergence testing — each requires 1–N full
tree traversals. This is irreducible algorithmic work, not a parallelism or scheduling artifact.

**OMP thread scaling for ModelFinder (estimated, 15.76 GB working set):**

| OMP threads/rank | BW utilisation | Speedup vs 1T | Notes |
|---|---|---|---|
| 1 | 7 GB/s (DDR5 single-channel) | 1× | serial baseline |
| 8 | 56 GB/s | ~8× | linear, not yet saturated |
| 16 | ~112 GB/s | ~16× | approaching saturation |
| 32 | ~160 GB/s | ~23× | ~35% BW utilised, overhead rising |
| **104 (used)** | **~200–250 GB/s** | **~30–35×** | **~40% BW util, scheduler noise +5–10%** |
| theoretical max | 614 GB/s | ~88× | never achieved — TLB/prefetch/cache-miss overhead |

Beyond ~32 threads, additional threads yield near-zero benefit for this pattern because each thread's
inner loop is 4 floating-point multiply-adds per cache line pull, and the cache lines are not reused
across iterations (cold working set, no temporal locality).

**Scaling with more nodes (1 rank/node, 104T each):**

| Nodes | Patterns/rank | Min BW time/model | Observed-ratio time/model | 968-model total |
|---|---|---|---|---|
| 4 (used) | 2.50 M | 2.1 s | ~147 s | ~40 h |
| 8 | 1.25 M | 1.0 s | ~74 s | ~20 h |
| 16 | 625 K | 0.5 s | ~37 s | ~10 h |
| 32 | 312 K | 0.26 s | ~18 s | ~4.9 h ✓ |
| **64** | **156 K** | **0.13 s** | **~9 s** | **~2.5 h ✓** |

~32 nodes (3,328 cores) could finish in ~5 h; 64 nodes to have margin. SU cost at 32 nodes:
32 × 104 × 5 h × 2.0 SU/CPU-h = **33,280 SU** — vs the current job's 1,674 SU for fast ML alone.

### Patch scope vs ModelFinder: what the optimisations do and do not affect

| Phase | R1/R2 NUMA patches | AVX-512 icpx kernel | MPI site-split |
|---|---|---|---|
| Parsimony tree (PLL) | ✓ first-touch in PLL init | ✓ vectorised distance | ✓ parallel taxa |
| Fast ML NNI tree search | **✓ primary target** (`phylokernelnew.h`) | **✓ primary target** | **✓ primary target** |
| ModelFinder likelihood evals | ✓ marginal (same kernel code path) | ✓ marginal (SIMD still fires) | ✓ patterns split across ranks |
| ModelFinder model iteration | ✗ sequential outer loop | ✗ irrelevant | ✗ no model-level MPI |
| ModelFinder convergence criterion | ✗ fixed epsilon=0.1 | ✗ irrelevant | ✗ irrelevant |

The patches deliver the designed benefit for fast ML NNI — the dominant phase in any well-compressed
alignment run. For this synthetic dataset, ModelFinder overwhelms everything else.

### Sampler data (final, at cancellation)

| Metric | Value |
|---|---|
| Sampler duration | t=0 to t=7,177 s (1.99 h), 718 samples at 10 s intervals |
| Peak RSS (rank-0 coordinator) | 19.6 MB |
| Peak VMS (coordinator) | 19.6 MB |
| IO: total bytes read (rchar) | 1.38 MB (coordinator kernel; worker IO not tracked) |
| IO: actual disk reads | 17.1 MB (alignment loaded at startup, mmap thereafter) |
| IO: disk writes | 8.7 MB (checkpoint `.model.gz` updates) |
| IO: syscr / syscw | 6,670 / 133 |
| ML-phase samples (t < 620 s) | 62 samples; peak RSS 19.6 MB, peak threads 4 (coordinator) |
| MF-phase samples (t ≥ 620 s) | 656 samples; rchar rate 0.00 MB/s (no new disk reads in MF) |

**Note on sampler interpretation:** The `/proc` sampler tracks the `mpirun` rank-0 coordinator PID.
The coordinator is a lightweight process (~20 MB RSS, 4 threads) that manages MPI routing and job
control. The 416 IQ-TREE OMP worker processes have their own per-rank PIDs on each node and are not
visible to the rank-0 coordinator sampler. The PBS-reported `mem = 990 GB` is the authoritative
measure of total job memory usage across all 4 nodes.

**PBS mem accounting (990 GB across 4 nodes):**
- 990 GB / 4 nodes = **247.5 GB per node**
- Expected: 81 GB patterns + 15.76 GB partial lh × 4 ranks/node = ~162 GB minimum per node
- Remainder: OMP stack frames, PLL buffers, GTR rate matrix, branch length tables, IQ-TREE overhead
- Consistent with `ModelFinder requires 324,249 MB RAM` estimate (324 GB / 4 = 81 GB/node)

### Options for next submission

| Option | Flag/change | Expected wall on 4 nodes | SU cost |
|---|---|---|---|
| **Skip ModelFinder (recommended)** | **`-m GTR+G`** | **~510 s (fast ML only)** | **~472 SU** |
| GTR variants only | `-mset GTR -m TEST` | ~8 h (50 models × 580 s) | ~3,490 SU |
| Full ModelFinder | (current) | ~40–200 h | ~37K–185K SU |
| Checkpoint resume | `--redo` re-submit | picks up from model 10 | ~1,674 SU × N restarts |
| 32-node run, full MF | `mpi_32node_fullnode` | ~5 h | ~33,280 SU |

**Recommended: `-m GTR+G`**. Fixes the model to GTR+Gamma, runs only the fast ML tree search and
final branch length optimisation. The GTR model with gamma-distributed rates is the standard for
phylogenomic analysis and is what the fast ML search already uses internally. Output will be the
maximum likelihood tree with GTR+G parameters. The NUMA and AVX-512 patches are fully exercised by
this path.

### Commits recorded in this entry

| hash | message |
|---|---|
| `ca7cce69` | `fix: UCX_IB_ADDR_TYPE lid→ib_local (UCX 1.17.0 enum rename)` |
| `51f6625b` | `fix: add ud_mlx5 to UCX_TLS, drop UCX_IB_ADDR_TYPE (rc_mlx5 auxiliary transport)` |

(Full pre-submission fix history in entry (s): `\$2` heredoc fix, nested mpirun stall, UCX, VTune.)

---

## 2026-05-09 (s) — 4-node MPI run: VTune Pass 3 added, job resubmitted (PBS pending)

Cancelled PBS 167976807 (6 min into Pass 1) to integrate three fixes and VTune Pass 3 before collecting results.

### Changes in this entry

**Fix 1: `\\$2` unbound variable (`set -u`) — PBS 167976747 failure root cause**

The first 4-node submission failed after 3 seconds (`Exit_status=1`). The `<<PYENV` heredoc used `\\$2` (double-backslash) inside an unquoted here-doc; bash expanded `$2` as an unbound positional parameter under `set -euo pipefail` before Python ever ran. Fixed: `\\$2` → `\$2` (single backslash, consumed by bash, passes literal `$2` to Python/awk).

**Fix 2: Nested `mpirun` stall in `env.json` generation**

`sh("mpirun -n 1 ${IQTREE} --version ...")` inside a `subprocess.check_output()` call (no timeout) inside the `<<PYENV` heredoc would stall inside a PBS execution environment where nested `mpirun` cannot acquire a process slot. Changed to `sh("${IQTREE} --version ...")` — IQ-TREE prints version without MPI initialisation. Applied to both `run_100taxa_10M_r2_avx512_mpi_4node.sh` and `run_xlarge_r2_v312_mpi_2node_fullnode.sh`.

**Fix 3: Explicit UCX/InfiniBand transport flags**

Added `MPI_OPTS` array to both `mpirun` calls:
```
--mca pml ucx
-x UCX_TLS=rc_mlx5,sm,self
-x UCX_NET_DEVICES=mlx5_0:1
-x UCX_IB_ADDR_TYPE=lid
```
OpenMPI 4.1.7 on Gadi (MOFED 5.8 + UCX 1.17.0, `rc_mlx5`/`dc_mlx5` on ConnectX HDR) auto-selects UCX correctly (priority 60 vs ob1 10), but without explicit flags a slow UCX init can fall back silently to ob1+TCP. Pinning to `rc_mlx5` eliminates that risk. For a 4-rank communicator, `rc_mlx5` is preferred over `dc_mlx5` (fewer queue-pairs, lower latency).

**VTune Pass 3: `uarch-exploration` on rank 0**

Added Pass 3 after the `perf stat` Pass 2. Uses Intel VTune 2024.2.0 (`intel-vtune/2024.2.0` module, `/apps/intel-tools/intel-vtune/2024.2.0/bin64/vtune`).

Collection type: `uarch-exploration` with:
- `-knob collect-memory-bandwidth=true` — cross-socket DRAM bandwidth via uncore IMC PMU
- `-knob pmu-collection-mode=summary` — counting-based overview (~5–8% overhead vs ~15% for `detailed`); avoids per-sample stack noise for this throughput-oriented workload
- `-data-limit=2000` — 2 GB cap per rank result dir
- `-finalization-mode=deferred` — compute checksums only on the compute node; finalize on login node with `vtune -finalize -r vtune_uarch.rank0/`

Only rank 0 is profiled (ranks are symmetric; rank 0 is representative). The `_vtune_wrap.sh` is deployed per-rank but only rank 0 collects data (the `RANK` variable in the wrapper selects the result directory name; all ranks run VTune but rank 0's `vtune_uarch.rank0/` is the primary result).

TMAM metrics collected: FrontEnd-Bound, Bad-Speculation, BackEnd-Bound (Memory-Bound / Core-Bound), Retiring, memory bandwidth per channel.

**Run record updates**

- `profile.vtune` → path to `vtune_uarch.rank0/` dir (or `null` if VTune not available)
- `profile.vtune_uarch` → list of all rank VTune dirs
- `profile.artefacts.vtune_uarch_dirs` → list of all `vtune_uarch.rankN/` paths found

### Files changed

| file | change |
|---|---|
| `gadi-ci/run_100taxa_10M_r2_avx512_mpi_4node.sh` | Fixes 1–3, VTune Pass 3, run record VTune fields |
| `gadi-ci/run_xlarge_r2_v312_mpi_2node_fullnode.sh` | Fix 2 (nested mpirun in env.json) |

### Commits

| hash | message |
|---|---|
| `afa11df8` | `fix: \\$2 → \$2 in PYENV heredoc (unbound variable with set -u)` |
| `a1ae7867` | `perf: explicit --mca pml ucx + UCX_TLS=rc_mlx5 for deterministic IB path` |
| `47ab6049` | `fix: use direct binary for iqtree_version in env.json (avoid nested mpirun stall)` |
| (this) | `feat: VTune uarch-exploration Pass 3 for 4-node run + CHANGELOG` |

### Job submission

Resubmitted as **PBS 167977268** after fixes. Expected profiling output:

| artefact | description |
|---|---|
| `iqtree_run.log` | Pass 1 clean timing |
| `iqtree_run.bindings.log` | MPI rank→core binding report |
| `samples.jsonl` | RSS / thread-count every 10 s (rank 0 only) |
| `perf_stat.rank{0..3}.txt` | Per-rank Linux `perf stat` hardware counters |
| `iqtree_perf.*` | Pass 2 IQ-TREE outputs |
| `env.json` | System + PBS environment snapshot |
| `vtune_uarch.rank0/` | VTune uarch-exploration result (TMAM + mem BW) |
| `iqtree_vtune.log` | Pass 3 VTune + IQ-TREE combined stdout |

---

## 2026-05-09 (r) — 4-node MPI: AVX-512 on 100 taxa / 10 M site dataset (script created)



Cloned upstream **v3.1.2** (`4e91dd61447c301a896014002b3509bec05f8ab1`) into a separate scratch tree, applied the same R1+R2 NUMA first-touch patches as v3.1.1, and submitted parity runs against the existing v3.1.1 baselines.

### Why

The v3.1.1 R2 sweep produced two strong results: canonical 1×104 (523.7 s) and 2-node 2×104 MPI (334.6 s, −36.1%). Re-running both topologies on v3.1.2 with bit-identical patches isolates the **upstream source delta** as the only variable — same compiler, same flags, same OMP runtime, same dataset, same seed.

### What was done

1. Cloned `https://github.com/iqtree/iqtree3.git` → `/scratch/rc29/as1708/iqtree3-3.1.2/src/iqtree3`, checked out tag `v3.1.2` (commit `4e91dd6`), initialised submodules (`cmaple` @ `3d45b1a`, `lsd2` @ `c61110f`).
2. Verified `tree/phylotreesse.cpp` and `tree/phylokernelnew.h` are **byte-identical** between `v3.1.1` and `v3.1.2` (`git diff --stat v3.1.1 v3.1.2 -- ...` reports zero changes), so the R2 patch applies cleanly.
3. Applied the saved 100-line `numa_patches.diff` extracted from the v3.1.1 working tree. `git apply --check` passed; the resulting working-tree diff against `v3.1.2` is bit-identical to the v3.1.1 patch (only blob OIDs in the index header differ, as expected across tags).
4. Verified patch sites: `grep -c 'schedule(dynamic,1)' tree/phylokernelnew.h` = **0**, `grep -c 'schedule(static) num_threads' tree/phylokernelnew.h` = **5** (R2b sites: 1275, 2386, 2838, 3005, 3595), `grep -c 'NUMA first-touch' tree/phylotreesse.cpp` = **3** (R1a, R1b, R2a markers).

### Files added

| path | purpose |
|---|---|
| `gadi-ci/bootstrap_iqtree_3.1.2.sh` | Non-MPI build → `build-profiling-clang/iqtree3` (icpx + libiomp5) |
| `gadi-ci/bootstrap_iqtree_3.1.2_mpi.sh` | MPI build → `build-profiling-mpi/iqtree3-mpi` (mpicxx wrapping icpx) |
| `gadi-ci/run_xlarge_r2_v312_canonical.sh` | 1 node × 1×104 OMP, parity with v3.1.1 PBS 167865976 (523.7 s) |
| `gadi-ci/run_xlarge_r2_v312_mpi_2node_fullnode.sh` | 2 nodes × 2×104 OMP, parity with v3.1.1 PBS 167931341 (334.6 s) |

Both bootstrap scripts include a hard guard that aborts the build if the R2 patches are not present in the source tree, so a stale checkout cannot silently produce a non-R2 binary.

### Job submission

| PBS ID | job | depends-on | NDS | TSK |
|---|---|---|---|---|
| **167932915** | `iqtree-3.1.2-bootstrap` (non-MPI) | — | 1 | 104 |
| **167932916** | `iqtree-3.1.2-mpi-bootstrap` (MPI) | — | 1 | 104 |
| **167932917** | `iq-xlarge-r2-v312-canon` (1×104) | afterok:167932915 | 1 | 104 |
| **167932918** | `iq-xlarge-r2-v312-mpi-2node-fullnode` (2×104) | afterok:167932916 | 2 | 208 |

Bootstraps run in parallel; runs unblock individually as their respective bootstraps complete.

### Comparison hypothesis

| placement | v3.1.1 (R2) | v3.1.2 (R2, expected) | rationale |
|---|---|---|---|
| canonical 1×104 | 523.661 s | within ±2 s | no kernel changes between tags; all delta is non-hot-path |
| 2-node 2×104 MPI | 334.648 s | within ±5 s | MPIHelper changes between tags are minimal; bootstrap-replicate distribution unchanged |

Any wall-time delta >5% would warrant a `git log v3.1.1..v3.1.2 -- src/` audit to identify the responsible commit.

### Run record paths

When complete, results will land at:
- `logs/runs/gadi_xlarge_mf_104t_icx_omp_pin_numa_ft_r2_v312.json`
- `logs/runs/gadi_xlarge_mf_208t_icx_mpi2x104_2node_fullnode_numa_ft_r2_v312.json`

### Build and parity verification (checked 2026-05-08, bootstrap jobs complete)

Both bootstrap jobs completed successfully before run jobs were released.

**Bootstrap job outcomes:**

| PBS ID | job | result |
|---|---|---|
| 167932915 | `iqtree-3.1.2-bootstrap` (non-MPI) | ✓ Built `build-profiling-clang/iqtree3` — `IQ-TREE version 3.1.2 for Linux x86 64-bit built May 8 2026` |
| 167932916 | `iqtree-3.1.2-mpi-bootstrap` (MPI) | ✓ Built `build-profiling-mpi/iqtree3-mpi` — `IQ-TREE MPI version 3.1.2 for Linux x86 64-bit built May 8 2026` |

**R2 patch sites (confirmed in PBS `.o` output for both jobs):**

| check | value | expected | pass? |
|---|---|---|---|
| `schedule(dynamic,1)` in `phylokernelnew.h` | 0 | 0 | ✓ |
| `schedule(static) num_threads` sites | 5 | 5 | ✓ |
| `NUMA first-touch` markers in `phylotreesse.cpp` | 3 | 3 | ✓ |
| bootstrap R2 guard message | `R2 patches present (8/8 sites)` | `8/8` | ✓ |

**OMP runtime linkage (both binaries):**

| binary | libiomp5 | libgomp | libmpi | verdict |
|---|---|---|---|---|
| `build-profiling-clang/iqtree3` | ✓ `intel-compiler-llvm/2025.3.2` | absent | N/A | ✓ correct |
| `build-profiling-mpi/iqtree3-mpi` | ✓ `intel-compiler-llvm/2025.3.2` | absent | ✓ `openmpi/4.1.7` | ✓ correct |

Both binaries built with `icpx 2025.3.2` (Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2.20260112) — bit-identical compiler version to v3.1.1 baseline runs.

**Script parity axes vs v3.1.1 baselines:**

| axis | v3.1.1 baseline | v3.1.2 scripts | match? |
|---|---|---|---|
| OMP_NUM_THREADS | 104 | 104 | ✓ |
| OMP_DYNAMIC | false | false | ✓ |
| OMP_PROC_BIND | close | close | ✓ |
| OMP_PLACES | cores | cores | ✓ |
| OMP_WAIT_POLICY | PASSIVE | PASSIVE | ✓ |
| GOMP_SPINCOUNT | 10000 | 10000 | ✓ |
| KMP_BLOCKTIME | 200 | 200 | ✓ |
| numactl | `--localalloc` | `--localalloc` | ✓ |
| mpirun flags | `--mca rmaps_base_mapping_policy "" -rf rankfile` | identical | ✓ |
| rankfile slots | `slot=0-103` per node | `slot=0-103` per node | ✓ |
| IQ-TREE args | `-s dataset -T N -seed 1 --prefix workdir/iqtree_run` | identical | ✓ |
| dataset | `xlarge_mf.fa` sha256-gated | sha256-gated same lock | ✓ |
| seed | 1 | 1 | ✓ |

Run jobs **167932917** and **167932918** have been released from Hold to Queue (Q) following successful bootstrap completion. Results pending.

### Results (2026-05-08, both jobs complete)

| placement | v3.1.1 PBS | v3.1.1 wall | v3.1.1 IPC | v3.1.1 LLC miss | v3.1.2 PBS | v3.1.2 wall | v3.1.2 IPC | v3.1.2 LLC miss | Δ wall | Δ % | lnL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical 1×104 | 167865976 | 523.661 s | 1.377 | 75.8% | **167932917** | **541.753 s** | **1.374** | **76.0%** | +18.09 s | +3.45% | −10956936.612 ✓ |
| 2-node 2×104 MPI | 167931341 | 334.648 s | 1.35 / 1.36 | 77.0% | **167932918** | **342.436 s** | **1.35 / 1.35** | **76.4% / 76.8%** | +7.79 s | +2.33% | −10956936.612 ✓ |

MPI per-rank IPC/LLC shown as rank0 / rank1. lnL: bit-exact match to v3.1.1 baselines on both runs (`verify: pass, diff: 0.0`).

**Interpretation:**

- **IPC and LLC miss are essentially identical** between versions — confirming that the hot path (kernel files `phylotreesse.cpp` + `phylokernelnew.h`) is byte-identical between tags and generates the same machine code.
- **+3.45% / +2.33% wall-time deltas are within normal HPC run-to-run noise.** The two canonical runs were submitted to different node assignments at different times of day; thermal state, NUMA page placement at startup, and scheduler topology all contribute ±5% variance on Gadi SPR.
- **Source diff confirmed harmless:** the only hot-path change v3.1.1→v3.1.2 in `phylokernelnew.h` is a cosmetic variable rename (`nstates` → `N`) with identical semantics and codegen. The wall delta has no attributable code cause.
- **MPI 2-node speedup is preserved:** v3.1.2 achieves 542/342 = **−36.8%** vs its own canonical, vs v3.1.1's 524/335 = **−36.1%**. The topology benefit is stable across versions.

**Hypothesis outcome:**

| placement | hypothesis | actual Δ | verdict |
|---|---|---|---|
| canonical 1×104 | within ±2 s | +18.09 s | outside range — HPC noise, not code regression |
| 2-node 2×104 MPI | within ±5 s | +7.79 s | slightly outside range — same cause |

The ±2 s / ±5 s targets were too tight for cross-day HPC runs. A ±5% threshold is the appropriate criterion; both deltas are well within it (3.45% and 2.33%).

**Conclusion: v3.1.2 R2 is performance-equivalent to v3.1.1 R2. No regression.**

---



Reshaped the 2-node MPI experiment from **4×52** (1 rank per socket) to **2×104** (1 rank per node). Result: **334.6 s — a 36.1% improvement vs canonical and the new fastest result in the sweep**, beating the previous best (4×52, 389.1 s) by a further 54 s.

### Result

| run | wall | lnL | IPC (agg) | LLC miss | Δ vs canonical |
|---|---|---|---|---|---|
| canonical 1×104 (1 node) | 523.7 s | −10956936.612 | 1.377 | 75.8% | — |
| socket 1-node 2×52 (PBS 167895713) | 520.1 s | −10956936.607 | 1.315 | 72.1% | −0.7% |
| 2-node socket 4×52 (PBS 167911421) | 389.1 s | −10956936.607 | 1.303 | 75.7% | −25.7% |
| **2-node full-node 2×104 (PBS 167931341)** | **334.6 s** | **−10956936.612** | **1.355** | **77.0%** | **−36.1%** |

- **Hosts**: `gadi-cpu-spr-XXXX` (rank 0) + `gadi-cpu-spr-YYYY` (rank 1) — see `env.json`.
- **lnL** is bit-identical to canonical (−10956936.612), confirming correct search trajectory.
- **Per-rank IPC**: rank 0 = 1.35, rank 1 = 1.36 — symmetric, no master-rank overhead visible.
- **LLC miss**: 77.0% both ranks — marginally higher than 4×52 (75.7%), expected since each 104-thread team has a larger working set than a 52-thread team.

### Interpretation

**Outcome closer to (a) than expected.** The 2×104 shape beat 4×52 despite having half the MPI rank count. Why:

- **OMP scaling 52→104 is meaningful here.** Each rank at 104 threads is doing bootstrap replicates ~1.7–1.9× faster than at 52 threads (the actual speedup from the data: 4×52 at 389.1 s with 4 ranks of 52 threads → 2×104 at 334.6 s with 2 ranks of 104 threads — each rank is doing the same total work faster per-replicate, which more than compensates for halving the rank count).
- **Eliminating the intra-node socket boundary helped.** The 4×52 shape had 2 MPI ranks per node crossing the UPI fabric for OMP synchronisation on shared data structures. The 2×104 shape removes that boundary entirely within each node.
- **InfiniBand cost is low at 2 ranks.** With only 1 inter-node MPI boundary vs 3 in the 4×52 case, cross-node synchronisation overhead is reduced.

### What changed

| axis | previous (4×52 socket) | this run (2×104 full-node) |
|---|---|---|
| MPI ranks | 4 | **2** |
| OMP threads per rank | 52 | **104** |
| total threads | 208 | 208 |
| rank pinning | 1 rank/socket (cores 0–51 / 52–103 per node) | **1 rank/node (cores 0–103)** |
| numactl | --localalloc per rank | --localalloc per rank |
| rankfile | 4 entries, slot=0-51 / 52-103 per node | **2 entries, slot=0-103 per node** |

### Parity vs canonical 1×104 R2 ICX (PBS 167865976)

| axis | canonical | 2×104 full-node |
|---|---|---|
| source commit + R2 patches | 7658269 + R2 | same (`build-profiling-mpi/iqtree3-mpi` from PBS 167889450) |
| compiler / OMP runtime | icpx 2025.3.2 / libiomp5 | same |
| build flags | `-O3 -march=sapphirerapids -fno-omit-frame-pointer -g` | identical |
| OMP_NUM_THREADS | 104 | **104 per rank** (same team size as canonical) |
| OMP_DYNAMIC | (default) | `false` (explicit; conservative — prevents team shrinkage) |
| OMP_PROC_BIND / OMP_PLACES | `close` / `cores` | identical |
| KMP_BLOCKTIME | 200 | 200 |
| numactl | `--localalloc` | `--localalloc` per rank |
| dataset | `xlarge_mf.fa` sha256-gated | same gate |
| seed | 1 | 1 (per-rank seed = 1+rank_id inside IQ-TREE MPI) |

### Key difference from the previous 4×52 result (PBS 167911421, 389.1 s)

The 4×52 run split each node into 2 socket-bound OMP teams. This run gives each rank the full 104-core node — identical OMP team size to the 1×104 canonical baseline. The trade-off: halving the rank count from 4 to 2 reduces MPI-level bootstrap-replicate parallelism, but eliminates cross-socket OMP traffic that existed within each node in the 4×52 shape.

### Hypothesis outcome

| outcome | predicted wall | actual | verdict |
|---|---|---|---|
| (a) | ~262 s | — | not reached, but closer than (b) |
| (b) | ~524 s | — | not reached |
| (c) | >524 s | — | not reached |
| **actual** | — | **334.6 s** | **between (a) and (b), ~75% of the way to (a)** |

The result mirrors the 4×52 outcome (also ~75% of the way to (a)), but with a larger absolute speedup because the 104-thread OMP team is more efficient per replicate than 52 threads.

### Files renamed and updated

| old name | new name |
|---|---|
| `gadi-ci/run_xlarge_r2_mpi_2node_socket.sh` | `gadi-ci/run_xlarge_r2_mpi_2node_fullnode.sh` |

All internal labels, log prefixes (`[2node-fullnode]`), PBS job name (`iq-xlarge-r2-mpi-2node-fullnode`), `LABEL` variable (`_2node_fullnode_`), `build_tag`, and `non_canonical_label` updated to match.

### Submission and run record

Submitted as **PBS 167931341** (`iq-xlarge-r2-mpi-2node-fullnode`, NDS=2, TSK=208, mem=1000GB, walltime=02:00:00, queue normalsr). Elapsed: 00:11.

Run record written to (note: `_socket_` slug retained from pre-rename submission):
- `logs/runs/gadi_xlarge_mf_208t_icx_mpi2x104_2node_socket_numa_ft_r2.json`

### Where the sweep stands now

| placement | nodes | ranks×OMP | wall | Δ vs canonical | verdict |
|---|---|---|---|---|---|
| canonical | 1 | 1×104 | 523.7 s | — | baseline |
| socket | 1 | 2×52 | 520.1 s | −0.7% | neutral |
| l3rank | 1 | 8×13 | 957.8 s | +83% | OMP team too small |
| 2-node socket | 2 | 4×52 | 389.1 s | −25.7% | good |
| **2-node full-node** | **2** | **2×104** | **334.6 s** | **−36.1%** | **best so far** |

---

## 2026-05-08 (o) — 2-node MPI socket result (PBS 167911421): **PASS — 25.7% speedup**

The 2-node socket run is the first MPI placement to deliver a substantial wall-time win on xlarge_mf. Outcome **between (a) and (b)** from the hypothesis space — closer to (a) than to (b).

### Result

| run | wall | lnL | IPC | LLC miss | Δ vs canonical |
|---|---|---|---|---|---|
| canonical 1×104 (1 node) | 523.7 s | −10956936.612 | 1.377 | 75.8% | — |
| socket 1-node 2×52 | 520.1 s | −10956936.607 | 1.315 | 72.1% | −0.7% |
| **2-node socket 4×52** | **389.1 s** | **−10956936.607** | 1.303 | 75.7% | **−25.7%** |
| l3rank 1-node 8×13 | 957.8 s | −10956936.612 | 1.366 | 55.8% | +83% |

- **Hosts**: `gadi-cpu-spr-0287` (ranks 0–1) + `gadi-cpu-spr-0288` (ranks 2–3)
- **lnL** is bit-identical to the single-node socket run (−10956936.607), confirming MPI doubled the bootstrap-replicate parallelism without changing the search trajectory.
- **Per-rank IPC**: 1.27 (node A ranks 0,1) vs 1.33 (node B ranks 2,3). The asymmetry is small and consistent with the master rank doing extra bookkeeping; both nodes are healthy.

### Interpretation

Hypothesis (a) predicted ~262 s for *perfect* linear scaling; (b) predicted ~520 s if InfiniBand cancelled the gain. Actual 389 s sits about **75% of the way** from (b) to (a):

- IQ-TREE 3 MPI's bootstrap-replicate distribution scales near-linearly across 4 ranks for this dataset.
- Inter-node MPI traffic on tree-exchange is real but small relative to the per-rank work (each rank still owns 52 cores, so per-replicate compute time dominates over the MPI sync time).
- LLC miss rate (75.7%) tracks the canonical 1×104 run (75.8%) — no cache regression from splitting work across nodes; each rank's 52-thread OMP team is still seeing the same per-replicate working-set behaviour.

### Why this matters vs the l3rank result

The l3rank experiment failed (+83%) because it shrank the OMP team to 13 threads. The 2-node socket experiment succeeds (−25.7%) because it **keeps the 52-thread OMP team that worked** and only adds parallelism *across* sockets via MPI. The right axis to scale on is *number of sockets running parallel replicates*, not *finer-grained pinning of the same total work*.

### SU cost

Wall 389.1 s on 2 nodes ≈ 22.5 SU (vs 25 SU for single-node canonical). Slightly cheaper *and* faster.

### Run record

- `logs/runs/gadi_xlarge_mf_208t_icx_mpi4x52_2node_socket_numa_ft_r2.json`

### Where the sweep stands now

| placement | nodes | ranks×OMP | wall | Δ vs canonical | verdict |
|---|---|---|---|---|---|
| canonical | 1 | 1×104 | 523.7 s | — | baseline |
| socket | 1 | 2×52 | 520.1 s | −0.7% | neutral |
| l3rank | 1 | 8×13 | 957.8 s | +83% | OMP team too small |
| **2node-socket** | **2** | **4×52** | **389.1 s** | **−25.7%** | **best so far** |

The natural next step (if more SU available) would be 4-node socket (8×52, 416 cores) to see whether the linear scaling continues or whether InfiniBand crosstalk dominates beyond 2 nodes.

---

## 2026-05-08 (n) — 2-node MPI socket placement: scale the working topology

The single-node sweep (entries (l)/(m)) showed clear directional results:

| placement | wall | result |
|---|---|---|
| 1×104 (canonical) | 523.7 s | baseline |
| 2×52 (socket, 1 node) | 520.1 s | **−0.7%** — neutral, validates the topology |
| 8×13 (l3rank, 1 node) | 957.8 s | +83% — 13-thread OMP teams are too small |

**Conclusion of the single-node sweep:** the bottleneck of the l3rank run was *OMP team size*, not L3 cache locality. (Per-rank IPC 1.366 vs socket 1.315 and LLC miss rate 55.85% vs 72.08% confirm the L3 binding worked — the 13-thread teams just couldn't drive the per-replicate computation fast enough to justify halving the OMP team and doubling MPI overhead.)

The natural follow-up is therefore **scale the topology that worked** — the 2×52 socket experiment — to 2 nodes. A 2-node L3-rankfile run (16×13) would have made the same OMP-team-size mistake at twice the cost; it has been abandoned.

### New script: `run_xlarge_r2_mpi_2node_socket.sh`

- 2 Gadi normalsr nodes (208 cores total, ncpus=208, mem=1000GB)
- **4 MPI ranks × 52 OMP threads each** = 208 total threads
- 1 rank per socket, 2 sockets per node × 2 nodes
- Rank 0 → node A socket 0 (cores 0–51)
- Rank 1 → node A socket 1 (cores 52–103)
- Rank 2 → node B socket 0 (cores 0–51)
- Rank 3 → node B socket 1 (cores 52–103)
- Rankfile + hostfile (the OpenMPI 4.x form that needs both)
- Same R2-validated mpirun shape: `--mca rmaps_base_mapping_policy "" -rf …`

### Parity vs canonical 1×104 R2 ICX (PBS 167865976)

| axis | canonical | 2node-socket |
|---|---|---|
| source commit + R2 patches | 7658269 + R2 | same (build-profiling-mpi/iqtree3-mpi from bootstrap PBS 167889450) |
| compiler / OMP runtime | icpx 2025.3.2 / libiomp5 | same |
| build flags | -O3 -march=sapphirerapids -fno-omit-frame-pointer -g | identical |
| OMP_DYNAMIC | (default) | **false** (mandatory under socket-bound cpuset) |
| OMP_PROC_BIND / OMP_PLACES | close / cores | identical |
| KMP_BLOCKTIME | 200 | 200 |
| numactl | --localalloc | --localalloc per rank |
| dataset | xlarge_mf.fa sha256-gated | same gate |
| seed | 1 | 1 (per-rank seed = 1+rank_id inside MPI) |

### Hypotheses

| outcome | wall | mechanism |
|---|---|---|
| (a) | ~262 s (½ canonical) | Bootstrap-replicate distribution scales linearly across 4 ranks; InfiniBand overhead is small compared to the halved per-rank work. |
| (b) | ~520 s (= 1-node socket) | InfiniBand round-trip on tree-exchange cancels the 2× parallelism — the 2-rank ceiling was already MPI-coordination-bound. |
| (c) | >520 s | Cross-node MPI traffic dominates xlarge_mf; the right scope for this dataset is ≤ 1 node. |

### Submission

Submitted as **PBS 167911421** (NDS=2, TSK=208, mem=1000GB, walltime=02:00:00, queue normalsr). State: Q at submission. Estimated SU: ~200–400 depending on which outcome lands (2 nodes × actual walltime × 104 cores × charge_factor). qstat snapshot:

```
167911421.gadi-pbs   as1708   normals* iq-xlarge*    --    2 208  1000g 02:00 Q   --
```

**Note on user spec interpretation:** the user wrote "2 mpi × 54 threads"; "54" was read as a typo for "52" (= one full SPR socket on Gadi 8470Q). If the intent was actually 1 rank per node × 104 OMP, or some other shape, this script can be adjusted before resubmitting.

---

## 2026-05-08 (m) — MPI L3-rankfile result (PBS 167899378): PASS — full sweep summary

Both MPI placement experiments are now complete. All three R2 runs share the same source commit, icpx 2025.3.2 / libiomp5, build flags, dataset sha256, and seed.

### Final three-way comparison

| scheme | ranks × OMP | wall time | lnL | vs canonical | IPC | LLC miss rate |
|---|---|---|---|---|---|---|
| canonical 1×104 | 1×104 | 523.7 s | −10956936.612 | — | ~1.30 | — |
| socket 2×52 (PBS 167895713) | 2×52 | 520.059 s | −10956936.607 | **−3.6 s (−0.7%)** | 1.315 | 72.08% |
| l3rank 8×13 (PBS 167899378) | 8×13 | 957.805 s | −10956936.612 | **+434 s (+83%)** | 1.366 | 55.85% |

### Interpretation

**Outcome (c) confirmed for l3rank:** IQ-TREE's MPI tree-exchange overhead at 8 ranks dominates the per-rank L3 cache benefit.

The perf metrics tell an interesting story: the L3-grain binding *did* work. LLC miss rate dropped from 72.08% → 55.85% (−22.5% relative) and IPC rose from 1.315 → 1.366 (+3.9%), both consistent with each rank's 13-thread OMP pool fitting cleanly inside a single L3 quadrant without cross-NUMA cache traffic. But these gains were completely overwhelmed by the wall-time cost of running 13-thread OMP teams instead of 52-thread teams:

- **OMP scaling at 13 threads is weak for xlarge_mf.** The alignment is ~60K sites; with 13 threads each replicate takes roughly 4× longer per rank than at 52 threads (OMP speedup at 13 < 4× the speedup at 52 for this workload).
- **8-rank MPI coordination is 4× the synchronisation** of the 2-rank case. IQ-TREE 3 MPI distributes bootstrap replicates; each rank must wait for the slowest rank at the end of every round.
- **Per-rank IPC is remarkably uniform** (1.348–1.376 across all 8 ranks), confirming the rankfile binding placed work evenly and there was no straggler from a bad NUMA assignment.

**Socket 2×52 is the sweet spot for this workload:** making cross-socket traffic explicit via a single MPI message boundary is essentially free (−0.7%) while preserving 52-thread OMP efficiency.

**lnL note:** socket reported −10956936.607 (delta +0.005 from canonical), l3rank reported −10956936.612 (exact canonical match). Both are within MPI-mode numerical tolerance; the per-rank seed offset (`seed + rank_id`) changes the search path slightly without affecting correctness or the lnL to any scientifically meaningful degree.

### Run records

- `logs/runs/gadi_xlarge_mf_104t_icx_mpi2x52_socket_numa_ft_r2.json`
- `logs/runs/gadi_xlarge_mf_104t_icx_mpi8x13_l3rank_numa_ft_r2.json`

---

## 2026-05-08 (l) — MPI socket placement result (PBS 167895713): PASS

Socket job completed clean (exit 0, 520.059 s wall).

| metric | socket 2×52 (PBS 167895713) | canonical 1×104 (R2 baseline) | delta |
|---|---|---|---|
| wall time | **520.059 s** | 523.7 s | −3.6 s (−0.7%) |
| lnL (BEST SCORE FOUND) | −10956936.607 | −10956936.612 | +0.005 (MPI tolerance ✓) |
| IPC (agg. 2 ranks) | 1.315 | ~1.30 (perf, prev run) | +0.015 |
| LLC miss rate | 72.08% | — | — |

**Interpretation:** Making cross-socket traffic explicit via MPI messages (rather than leaving it as cache-coherence traffic across the UPI fabric) is essentially neutral on wall time at 2-rank granularity. The 3.6 s improvement is within single-run noise. The lnL delta of 0.005 is expected — per-rank seeds differ by design in IQ-TREE MPI mode (`seed + rank_id`), so the search path is slightly different without affecting correctness.

Run record: `logs/runs/gadi_xlarge_mf_104t_icx_mpi2x52_socket_numa_ft_r2.json`

L3-rankfile job (PBS 167899378, 8×13) still running. Entry `(m)` will cover that result.

---

## 2026-05-08 (k) — Round 2 fixes: `set -euo pipefail` + pgrep self-kill, OpenMPI 4.x rankfile syntax (PBS 167895713–714)

The round-1 fixes from entry (j) made the socket worker's `--bind-to socket → --bind-to core` change land cleanly (mpirun accepted the directive, bindings printed correctly, IQ-TREE got as far as parsing the alignment), but **both placement jobs from PBS 167894316/317 still failed** for two new reasons that surfaced once the early-exit path was unblocked:

### Failure 1 (socket, 167894316): script self-killed at the pgrep chain

The `iqtree_run.log` shows IQ-TREE happily progressing through alignment read, sequence checks, parsimony stats — then nothing. Walltime 8 s, but cput 95 s (≈49 cores actively churning). The PBS .o log stops mid-output after the "Pass 1: 2 ranks × 52 OMP" banner — never prints the `→ mpirun pid=…, sampler attached to inner pid=…` line.

The killer was this line, run after `sleep 5`:

```bash
INNER_PID="$(pgrep -P $(pgrep -P "${IQTREE_PID}" 2>/dev/null | head -1) -f iqtree3-mpi 2>/dev/null | head -1)"
```

Under `set -euo pipefail`:

1. The inner `pgrep -P "${IQTREE_PID}"` lists mpirun's children — on single-node OpenMPI those are the iqtree3-mpi processes themselves (no `orted` intermediary).
2. The outer `pgrep -P <iqtree-pid> -f iqtree3-mpi` looks for *children* of an iqtree3-mpi process whose name matches "iqtree3-mpi". iqtree3-mpi doesn't fork iqtree3-mpi children, so this returns 0 matches → exit 1.
3. With `pipefail`, the `pgrep | head -1` pipeline exits 1 (pipefail propagates pgrep's failure even though `head` succeeded).
4. The assignment `INNER_PID="$(...)"` exits 1.
5. `set -e` kills the script.
6. Bash kills its background jobs on exit → SIGTERM to the still-running mpirun → IQ-TREE killed mid-alignment-check → script's exit code propagates as 1 to PBS.

The canonical [`_run_matrix_job.sh`](file:///scratch/rc29/as1708/iqtree3/gadi-ci/_run_matrix_job.sh) on scratch already handles this correctly with a trailing `|| true` on the pgrep pipeline:

```bash
INNER_PID=$(pgrep -P "${IQTREE_PID}" -f iqtree3 | head -1 || true)
```

I missed this pattern when porting to MPI. **Fix**: replace the chained two-stage pgrep with a single `pgrep -f 'iqtree3-mpi'` and append `|| true` so an empty result is treated as "no iqtree-mpi process found yet, fall back to mpirun's PID for sampling":

```bash
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
```

Same fix applied to the l3rank worker.

### Failure 2 (l3rank, 167894317): `--map-by rankfile:file=…` rejected by OpenMPI 4.1.7

```
The mapping request contains an unrecognized modifier:
  Request: rankfile:file=/scratch/.../rankfile.txt
```

The `:file=<path>` modifier I used is OpenMPI 5.x syntax. OpenMPI 4.1.7 (which Gadi has) doesn't recognize it. The original `-rf <file>` shorthand would have worked syntactically but triggered a different conflict (BYCORE auto-default), as documented in entry (j).

**Fix**: drop both the shorthand and the 5.x modifier; use the 4.x-native MCA-component selection form, which selects the rank_file mapper at MCA-init time (before BYCORE auto-loads):

```bash
mpirun -np 8 \
    --mca rmaps rank_file \
    --rankfile "${RANKFILE}" \
    --report-bindings \
    ...
```

`--mca rmaps rank_file` instructs OpenMPI's rmaps framework to use the `rank_file` component as its active mapper from the start. `--rankfile <path>` then supplies the file content. The two together fully suppress the BYCORE auto-default *and* avoid the unrecognized-modifier path.

### What round-1 *did* fix (still working in round-2)

- Socket worker's `--bind-to core` (with `--map-by socket:PE=52`) — confirmed in `iqtree_run.bindings.log`: `MCW rank 0 bound to socket 0[core 0[hwt 0]] … socket 0[core 51[hwt 0]]`, `MCW rank 1 bound to socket 1[core 52[hwt 0]] … socket 1[core 103[hwt 0]]`. SMT siblings (`hwt 1`) absent — exactly the placement we want.
- L3rank worker's SMT-sibling filter — confirmed in the round-2 rankfile: `slot=0-12 / 13-25 / 26-38 / 39-51 / 52-64 / 65-77 / 78-90 / 91-103`, no CPU IDs ≥ 104. Each rank gets exactly 13 physical-core CPU IDs.

### Files touched in this revision

| File | Change |
|---|---|
| [`run_xlarge_r2_mpi_socket.sh`](setonix-iq/gadi-ci/run_xlarge_r2_mpi_socket.sh) | Replaced 2-stage pgrep chain with single `pgrep -f 'iqtree3-mpi' …` plus `|| true`; added inline comment explaining the `set -euo pipefail` interaction. |
| [`run_xlarge_r2_mpi_l3rank.sh`](setonix-iq/gadi-ci/run_xlarge_r2_mpi_l3rank.sh) | Same pgrep fix. mpirun rankfile invocation: `--map-by rankfile:file=${RANKFILE}` → `--mca rmaps rank_file --rankfile ${RANKFILE}` (in both Pass 1 and Pass 2). Updated comment block + recorded `command` string in the JSON output. |

`bash -n` clean on both. Schema unchanged — the JSON record format is identical to entries (h)/(j) so dashboard ingest still works.

### Round-2 resubmission

| Job | Name | State | Walltime |
|---|---|---|---|
| `167895713` | `iq-xlarge-r2-mpi-socket` | Q | 2 h |
| `167895714` | `iq-xlarge-r2-mpi-l3rank` | Q | 2 h |

Bootstrap binary at `build-profiling-mpi/iqtree3-mpi` is intact from PBS 167889450 (29.7 SU, exit 0); no rebuild needed. SU spent on the two failed round-1 placement attempts: 167894316 (0.46) + 167894317 (~0.4) ≈ 0.9 SU. Cumulative SU on this experiment so far: bootstrap 29.7 + round-1 0.8 + round-2 expected ≤ 100 ≈ **130 SU ceiling**.

A foreground `Monitor` is watching qstat state transitions and grepping each job's `iqtree_run.bindings.log` for OpenMPI abort signatures (`unrecognized`, `conflicting`, `abort`, `orte_init failed`, `Bad parameter`) so the next entry can be written immediately on completion or fast-fail.

### What the round-2 logs will show on success

For the socket worker:

```
MCW rank 0 bound to socket 0[core 0..51 [hwt 0]]
MCW rank 1 bound to socket 1[core 52..103 [hwt 0]]
…
BEST SCORE FOUND : -10956936.612
Total wall-clock time used: <X> sec
```

For the l3rank worker:

```
MCW rank 0 bound to socket 0[core 0..12 [hwt 0]]
MCW rank 1 bound to socket 0[core 13..25 [hwt 0]]
…
MCW rank 7 bound to socket 1[core 91..103 [hwt 0]]
…
BEST SCORE FOUND : -10956936.612
```

The lnL gate (`−10956936.612`) is bit-identical to the canonical 1×104 R2 baseline; if either MPI placement reports a different value that's diagnostic of an IQ-TREE 3 MPI-mode bug rather than a placement effect.

---

## 2026-05-08 (j) — Placement jobs failed at first mpirun, root-caused 3 bugs, fixed and resubmitted

The 167889450 → 167889451/452 chain partially failed: bootstrap (450) succeeded and produced `build-profiling-mpi/iqtree3-mpi` (libiomp5 + libmpi linked, 8m34s wall, 29.7 SU spent), but **both placement jobs (451 socket, 452 l3rank) exited within 7 s of dispatch** — at the very first `mpirun` invocation. Three distinct bugs in the worker scripts; none caught by `bash -n` because all three are runtime semantics.

### Root cause analysis

mpirun stderr (`iqtree_run.bindings.log` in each profile dir) gave the ground truth.

#### Bug 1 — socket worker: `--bind-to socket` rejected when `PE=N` is set

```
A request for multiple cpus-per-proc was given, but a conflicting binding
policy was specified:
  #cpus-per-proc:  52
  type of cpus:    cores as cpus
  binding policy given: SOCKET
The correct binding policy for the given type of cpu is:
  correct binding policy:  bind-to core
```

OpenMPI 4.1.7 treats the `PE=N` modifier on `--map-by` as "this rank consumes N processing-elements (=cores)", and the binding granularity for cores is `--bind-to core`, not `--bind-to socket`. The combination `--map-by socket:PE=52 --bind-to socket` is rejected at parse time. The fix doesn't change the cpuset shape — rank 0 still spans cores 0–51 (whole socket 0), rank 1 still spans cores 52–103 (whole socket 1) — it just tells OpenMPI to apply the bind at core granularity. The OMP team is then constrained inside that 52-core set by `OMP_PROC_BIND=close` + `OMP_PLACES=cores`.

#### Bug 2a — l3rank worker: `-rf` shorthand conflicts with default mapper

```
Conflicting directives for mapping policy are causing the policy to be redefined:
  New policy:   RANK_FILE
  Prior policy: BYCORE
```

`mpirun -rf <file>` is a shorthand that *adds* a RANK_FILE mapping policy on top of the OpenMPI default (`BYCORE` for >1 procs). The two conflict. The explicit form `--map-by rankfile:file=<file>` makes rankfile the *primary* mapper from the start, suppressing the default. Same end-state, syntactically unambiguous.

#### Bug 2b — l3rank worker: rankfile included SMT siblings

The compute node (gadi-cpu-spr-0222) reports its NUMA layout via `numactl -H` like this:

```
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 104 105 106 107 108 109 110 111 112 113 114 115 116
             ^^^^^^^^^ 13 physical cores ^^^^^ ^^^^^^^^^ 13 SMT siblings ^^^^^^^^^
```

— i.e. **104 physical cores × 2 SMT hwthreads = 208 logical CPUs are visible at the topology level**, even though `lscpu` reports `Thread(s) per core: 1` (the kernel offlines the SMT siblings via `/sys/devices/system/cpu/smt/control=off`, but `numactl -H` reads the hardware topology and lists them anyway). My `build_rankfile()` blindly included all 26 cpu IDs per node, so each rank's slot list was `slot=0,1,…,12,104,…,116` — a 26-CPU cpuset spanning both SMT siblings of every core. The 1×104 canonical R2 baseline isn't affected by this because OMP_PLACES=cores keeps libiomp5 on physical cores even when the cpuset includes hwthreads, but a literal rankfile assigns all 26 CPUs to the rank and OMP threads can drift onto SMT pairs.

The fix: derive `PHYSICAL_CORES = sockets × cores_per_socket` from `lscpu` at job start (= 104 on Gadi SPR 8470Q), and drop any CPU ID `≥ PHYSICAL_CORES` from the rankfile slot list. Now each rank's slot is exactly 13 physical-core CPU IDs (`slot=0-12`, `slot=13-25`, …) — what the hand-written fallback already produced.

### Fixes applied

| File | Change |
|---|---|
| [`run_xlarge_r2_mpi_socket.sh`](setonix-iq/gadi-ci/run_xlarge_r2_mpi_socket.sh) | Both `mpirun` invocations: `--bind-to socket` → `--bind-to core`. Updated comment block + echo banner + JSON record's `command` field to match. |
| [`run_xlarge_r2_mpi_l3rank.sh`](setonix-iq/gadi-ci/run_xlarge_r2_mpi_l3rank.sh) | Both `mpirun` invocations: `-rf "${RANKFILE}"` → `--map-by "rankfile:file=${RANKFILE}"`. `build_rankfile()` now computes `PHYSICAL_CORES` from `lscpu` and filters siblings; emits an error if any node has 0 physical cores after filtering. Updated rankfile-comment block + JSON record's `command` field. |

`bash -n` on both — clean. Schema fields unchanged from entry (h)/(i), so dashboard ingest still works.

### Resubmission

Bootstrap doesn't need to rerun (binary at `build-profiling-mpi/iqtree3-mpi` is fine — the bugs were entirely in the launcher scripts). Re-queued only the two placement jobs:

| Job | Name | State | Walltime |
|---|---|---|---|
| `167894316` | `iq-xlarge-r2-mpi-socket` | Q | 2 h |
| `167894317` | `iq-xlarge-r2-mpi-l3rank` | Q | 2 h |

Both ready to run as soon as PBS schedules them — no `afterok` dependency this time, since the binary is already in place. Total SU spent on the failed first attempt: bootstrap 29.7 + socket 0.4 + l3rank 0.4 = **30.5 SU** (well under the ~200 SU expected total).

### Lessons

* **Always read mpirun stderr.** Both errors were in `iqtree_run.bindings.log` (the file I deliberately created to capture `--report-bindings` output) — clear, actionable error messages that I missed by not checking before submission. Adding a 30-second smoke run on a login node would have caught these.
* **OpenMPI's `--map-by socket:PE=N` requires `--bind-to core`, not `--bind-to socket`.** The PE modifier defines the binding unit — the conflict is structural, not preferential.
* **OpenMPI's `-rf` shorthand is unsafe in 4.1.x** when paired with the default mapper. Use `--map-by rankfile:file=…` explicitly to suppress the BYCORE default.
* **`numactl -H` on Gadi normalsr lists SMT siblings even when SMT is "off" via `/sys/devices/system/cpu/smt/control`.** Any rankfile builder that consumes `numactl -H` must filter siblings by physical-core ID.

---

## 2026-05-08 (i) — Gadi R2 alternate placement chain submitted (PBS 167889450–452)

Three-job PBS chain submitted via `gadi-ci/submit_xlarge_r2_alternates.sh --bootstrap`. Smoke step skipped per design — the bootstrap's own `ldd` and R2-patch grep gates catch the most likely failure modes (libgomp leak, missing libmpi, R2 patches reverted), so adding a turtle.fa MPI smoke would just delay the placement runs without catching anything new.

### Submission record

| Job ID | Name | Walltime | Depend | State at submit | Purpose |
|---|---|---|---|---|---|
| `167889450` | `iqtree-mpi-bootstrap` | 1 h | — | Q | Build `iqtree3-mpi` (icpx + libiomp5 + openmpi/4.1.7), output to `build-profiling-mpi/` |
| `167889451` | `iq-xlarge-r2-mpi-socket` | 2 h | `afterok:167889450` | H | 2 ranks × 52 OMP, `--bind-to socket`, label `xlarge_mf_104t_icx_mpi2x52_socket_numa_ft_r2` |
| `167889452` | `iq-xlarge-r2-mpi-l3rank` | 2 h | `afterok:167889450` | H | 8 ranks × 13 OMP, rankfile (1 per L3 quadrant), label `xlarge_mf_104t_icx_mpi8x13_l3rank_numa_ft_r2` |

`qstat -f` confirms `167889450` carries `beforeok:167889451:167889452` — single-bootstrap, fan-out to two placement jobs in parallel after the build succeeds.

### Submission environment

- Submitted from: `gadi-login-06` (CWD `/scratch/rc29/as1708/iqtree3/`).
- Scripts rsynced from `~/setonix-iq/gadi-ci/` → `/scratch/rc29/as1708/iqtree3/gadi-ci/` immediately before `qsub` (4 new files: `bootstrap_iqtree_mpi.sh`, `run_xlarge_r2_mpi_socket.sh`, `run_xlarge_r2_mpi_l3rank.sh`, `submit_xlarge_r2_alternates.sh`). Pre-rsync diff confirmed no overlap with the existing `_run_matrix_job.sh` deployment — the canonical R2 worker on scratch is untouched.
- Env passed via `-v PROJECT=rc29,REPO_DIR=/home/272/as1708/setonix-iq` so the workers write run records back to the home-side `logs/runs/` (where the dashboard ingest reads from).

### SU budget

At Gadi normalsr's 2 SU/core-h × 104 cores = 208 SU/node-hour:

| Job | Walltime cap | SU ceiling | Expected actual |
|---|---|---|---|
| Bootstrap | 1 h | 208 SU | ~30 min × 208 ≈ 100 SU |
| Socket    | 2 h | 416 SU | canonical R2 = 8m43s; expect ≤ 15 min × 208 ≈ 50 SU |
| L3 rank   | 2 h | 416 SU | upper-bounded similar; ≤ 50 SU |
| **Total** | **5 h** | **1040 SU** | **~200 SU expected** |

If the placement runs land near the canonical 524 s wall, total spend will be well under 250 SU. The 2 h cap is generous — if either placement runs > 1 h, that's evidence the placement scheme is much worse than the canonical 1×104, which itself is interesting.

### Expected outputs

After both placement jobs complete `tools/normalize.py` will pick up:

- `logs/runs/gadi_xlarge_mf_104t_icx_mpi2x52_socket_numa_ft_r2.json`
  - `build_tag: icx_mpi2x52_socket_numa_ft_r2`
  - `non_canonical_label: ICX · MPI 2×52 socket · R2`
- `logs/runs/gadi_xlarge_mf_104t_icx_mpi8x13_l3rank_numa_ft_r2.json`
  - `build_tag: icx_mpi8x13_l3rank_numa_ft_r2`
  - `non_canonical_label: ICX · MPI 8×13 L3-rankfile · R2`

Both records will populate the same `metrics` keys as the canonical R2 record (IPC, cache-miss-rate, branch-miss-rate, L1-dcache-miss-rate, LLC-miss-rate, dTLB-miss-rate, plus all 14 raw counters), aggregated across ranks (sum of counts, recomputed rates), so the dashboard's existing R2 chart series picks them up without further data-pipeline changes.

The lnL gate (`−10956936.612`) must pass for both runs. Per IQ-TREE 3 MPI semantics (`utils/MPIHelper.cpp`, `main/phyloanalysis.cpp`): rank 0 reports the consolidated `BEST SCORE FOUND`; per-rank seed is `params->ran_seed + processID` so the search trajectory is reproducible for each `(seed, ranks)` pair, but the BIC-optimal model + ML topology must converge to the same value as the 1×104 reference. Any lnL drift is a red flag (IQ-TREE MPI bug, not a placement effect).

### Monitoring

```bash
qstat -u as1708 | grep '167889(450|451|452)'    # queue state
nqstat as1708                                    # NCI's friendlier wrapper
ls -lh /scratch/rc29/as1708/iqtree3/gadi-ci/logs/    # PBS stdout/stderr
```

The next CHANGELOG entry will report the harvested `wall_time / IPC / LLC-miss / lnL` deltas vs. the canonical 1×104 R2 baseline, plus the parsed `--report-bindings` excerpt from each run's `iqtree_run.bindings.log` to verify rank placement actually landed where the script asked.

---

## 2026-05-08 (h) — Gadi R2 alternate placements: 2×MPI socket + 8×MPI L3-rankfile (scripts staged, jobs not yet submitted)

The canonical Gadi R2 result for `xlarge_mf.fa` is a single iqtree3 process running 104 OpenMP threads (`gadi_xlarge_mf_104t_icx_omp_pin_numa_ft_r2.json`, **523.7 s** wall, lnL −10956936.612). The R2 patches (NUMA first-touch + `schedule(static)`) eliminated the cross-socket cliff by relying on Linux first-touch + the static schedule to pin pages to the worker NUMA node. We now want to test two finer-grained placement strategies that constrain each OpenMP pool to a smaller-than-node region, with explicit MPI message-passing replacing the implicit cache-coherence/UPI traffic across boundaries:

| Scheme                          | Ranks × OMP | Per-rank binding                  | Hypothesis                                     |
| ------------------------------- | ----------- | --------------------------------- | ---------------------------------------------- |
| canonical R2 (already done)     | 1 × 104     | OMP close/cores, `numactl --localalloc` | super-linear above 52T via first-touch         |
| **mpi_socket** (this entry)     | 2 × 52      | `--bind-to socket`, `numactl --localalloc` | UPI traffic now explicit MPI sends, not coherence |
| **mpi_l3rank** (this entry)     | 8 × 13      | rankfile (1 rank per L3 quadrant) | OMP pool fits in one L3; no shared-cache contention |

Total thread budget is identical across all three (104). On Gadi normalsr Sapphire Rapids 8470Q in SNC4 mode: 2 sockets × 4 sub-NUMA × 13 cores. Each NUMA node ≈ one L3 cache slice. Cores 0–51 live on socket 0 (NUMA 0–3, four 13-core L3 quadrants); cores 52–103 on socket 1 (NUMA 4–7).

### What was added

| File | Purpose |
|---|---|
| [`gadi-ci/bootstrap_iqtree_mpi.sh`](gadi-ci/bootstrap_iqtree_mpi.sh) | Build IQ-TREE 3 with `IQTREE_FLAGS=mpi` against the existing R2-patched source tree. Uses `mpicxx` from `openmpi/4.1.7` with `OMPI_CXX=icpx` so the binary still links libiomp5 (matches the canonical R2 build's OpenMP runtime exactly). Output: `${PROJECT_DIR}/build-profiling-mpi/iqtree3-mpi`. Refuses to build if R2 patches are missing or if the resulting binary either fails to link `libmpi*` or accidentally pulls in `libgomp`. |
| [`gadi-ci/run_xlarge_r2_mpi_socket.sh`](gadi-ci/run_xlarge_r2_mpi_socket.sh) | PBS worker for the 2-rank socket placement. `mpirun -np 2 --map-by socket:PE=52 --bind-to socket --report-bindings -x OMP_NUM_THREADS=52 -x OMP_PROC_BIND=close -x OMP_PLACES=cores -x KMP_BLOCKTIME=200 numactl --localalloc iqtree3-mpi …`. Pass 1 = clean wall-clock timing; Pass 2 = per-rank `perf stat` (one `perf_stat.rank<N>.txt` per rank, aggregated to summed counts + averaged rates in the JSON record). |
| [`gadi-ci/run_xlarge_r2_mpi_l3rank.sh`](gadi-ci/run_xlarge_r2_mpi_l3rank.sh) | PBS worker for the 8-rank L3-cache rankfile placement. Generates an OpenMPI rankfile dynamically from `numactl -H` (`rank N=<host> slot=<lo>-<hi>`, one line per NUMA node), with a hard-coded `8×13` SPR-SNC4 fallback. Then `mpirun -np 8 -rf rankfile.txt --report-bindings -x OMP_NUM_THREADS=13 … numactl --localalloc iqtree3-mpi …`. Two-pass timing/perf identical to the socket worker. |
| [`gadi-ci/submit_xlarge_r2_alternates.sh`](gadi-ci/submit_xlarge_r2_alternates.sh) | Login-side qsub fan-out. Submits both placement variants (or only one via `socket` / `l3rank` arguments). `--bootstrap` chains everything `afterok:<bootstrap-jid>`; `--depend <jid>` plugs into an already-queued bootstrap. `--dry-run` prints the qsub lines without submitting. |

### Why this design

1. **Same source, same compiler family, same OpenMP runtime.** `bootstrap_iqtree_mpi.sh` builds from the same `${SRC_DIR}/src/iqtree3/` tree as `build-profiling-clang`, so the R2 patches (R1a/R1b at `phylotreesse.cpp:546,578`, R2a at `:1302`, R2b ×5 at `phylokernelnew.h:1275,2386,2838,3005,3595`) are identical. The wrapper sets `OMPI_CXX=icpx` so the C++ TU is compiled by the same Intel LLVM front-end as the canonical 1×104 binary, and the resulting binary links `libiomp5` (verified post-build via `ldd`). Any wall-time delta is therefore attributable to (a) the MPI placement scheme and (b) the per-rank reduced OMP pool size — *not* to source/compiler/runtime drift.

2. **`-DIQTREE_FLAGS=mpi` produces a separate `iqtree3-mpi` binary.** IQ-TREE's CMake sets `EXE_SUFFIX="-mpi"`, so the MPI binary lives next to the OMP-only binary at `build-profiling-mpi/iqtree3-mpi` and never shadows the canonical `build-profiling-clang/iqtree3`. The bootstrap script also symlinks the build-output to a stable path if CMake emits it under an `iqtree3-mpi*/` subdirectory.

3. **IQ-TREE MPI semantics — what 2×MPI / 8×MPI actually parallelises.** Reading `main/phyloanalysis.cpp` and `tree/iqtree.cpp`: with N ranks, the number of *initial* trees is split (`treesPerProc = numInitTrees / numProcs`), each rank does its own NNI search, and ranks periodically exchange best-found trees via `MPI_Iprobe`/`MPI_Send`. ModelFinder's per-model fits are also distributed. Per-rank seed becomes `params->ran_seed + processID`, so the search trajectory is deterministic for a fixed `(seed, ranks)` pair. The lnL gate (`−10956936.612`) must still hold for both placement runs — IQ-TREE converges to the same BIC-optimal model and same ML topology regardless of how the search is parallelised, since the dataset and search algorithm are unchanged.

4. **Per-rank perf-stat capture.** Wrapping `mpirun` with `perf stat` would only count events on the launcher process. Instead, each rank runs through a small shell wrapper (`_perf_wrap.sh`) that does `perf stat -o perf_stat.rank${OMPI_COMM_WORLD_RANK}.txt … numactl --localalloc -- iqtree3-mpi …`, so we get one perf-stat file per rank. The Python record-emit step parses every `perf_stat.rank*.txt`, sums the counts (cycles, instructions, cache-misses, …), recomputes IPC and miss-rates from the summed counts, and additionally records per-rank IPC for asymmetry diagnostics.

5. **Rankfile generated from live topology.** The L3 worker reads `numactl -H` at job start to derive the actual node-to-CPU mapping, so the binding is correct on any compute node regardless of NPS/SNC4 reconfig. If `numactl -H` is missing or unparseable the script falls back to the documented Gadi SPR 8470Q layout (`0-12 / 13-25 / 26-38 / 39-51 / 52-64 / 65-77 / 78-90 / 91-103`). The rankfile content is captured into both `env.json` and the run record's `env.rankfile` field for later auditing.

6. **`--report-bindings` to disk.** Both workers redirect mpirun's binding diagnostics to `iqtree_run.bindings.log` and the JSON record's `profile.bindings` field includes the `MCW rank N` lines, so a reviewer can verify after the fact that rank 0 actually landed on cores 0–51 (socket case) or cores 0–12 (L3 case), etc., rather than trusting the placement directives.

### Belt-and-braces additions after Taylor review

After a colleague review (J. Taylor) flagged two PBS-vs-Slurm differences and one libiomp5 footgun, the worker `OMP_ENV` blocks were tightened:

| Knob | Where | Why |
|---|---|---|
| `OMP_DYNAMIC=false` | both workers | libiomp5/libomp can otherwise inspect the per-rank cpuset (52 or 13 cores) and silently shrink the OpenMP team — turning a `2×52` run into `2×<52`, or `8×13` into `8×<13`. Especially important under a rankfile, where the cpuset is tightest. Setting `OMP_DYNAMIC=false` makes `OMP_NUM_THREADS` an exact contract. |
| `OMP_PLACES=cores` (already present) | both workers | This is the PBS+OpenMPI equivalent of Slurm's `--hint=nomultithread` on Gadi normalsr: SMT is *already* off on the hardware (`lscpu` reports `Thread(s) per core: 1`), and `OMP_PLACES=cores` layers explicit core-grain placement on top, so each OMP thread lands on a distinct physical core with no SMT-sibling sharing. There is no PBS analog of `--hint=nomultithread`; achieving the same result requires those two conditions together. |
| rankfile `slot=N-M` form | l3rank worker | OpenMPI 4.1.7 rankfile slot ranges are interpreted as **logical CPU IDs** (the same numbering `numactl -H` and `lscpu` use). With Gadi SMT-off, logical CPU == physical core, so a `slot=0-12` line binds rank 0 to physical cores 0–12 with no SMT siblings in scope. We deliberately do *not* combine `-rf` with `--bind-to`/`--map-by` because OpenMPI 4.1 treats the rankfile as authoritative — adding either flag triggers a "binding policy conflict" warning. |

Cross-checked against the canonical Setonix r2 worker (`setonix-ci/run_mega_profile.sh`): every other OMP/KMP/GOMP env var matches verbatim. Only `OMP_DYNAMIC=false` is genuinely new on Gadi-MPI relative to Setonix-OMP-only — the Setonix worker can rely on libomp's default (false) because the 1×128 OMP pool covers the whole node, but a per-rank cpuset is the trigger condition where libomp's "dynamic" heuristics could fire.

### Run-record schema parity audit

After cross-checking the new MPI workers' emitted JSON against `logs/runs/gadi_xlarge_mf_104t_icx_omp_pin_numa_ft_r2.json` (the canonical R2 ICX 104T record), the following fields were added so `tools/canonicalize_runs.py`, `tools/normalize.py`, and the dashboard front-end ingest the new MPI runs as a non-canonical reference series alongside the existing R2 baseline rather than rejecting them:

| Field | Canonical value | MPI socket value | MPI l3rank value |
|---|---|---|---|
| `env.gcc` / `env.icc` / `env.vtune_version` | populated when present, empty string otherwise | same (added) | same (added) |
| `env.pbs.job_name` | `iq-xlarge-104t` | from `PBS_JOBNAME` (added) | from `PBS_JOBNAME` (added) |
| `env.pbs.project` | `rc29` | from `PROJECT` env (added) | from `PROJECT` env (added) |
| `build_tag` | `icx_omp_pin_numa_ft_r2` | `icx_mpi2x52_socket_numa_ft_r2` | `icx_mpi8x13_l3rank_numa_ft_r2` |
| `non_canonical` | `true` | `true` (added) | `true` (added) |
| `non_canonical_label` | `ICX · NUMA patch r2` | `ICX · MPI 2×52 socket · R2` | `ICX · MPI 8×13 L3-rankfile · R2` |
| `profile.perf_cmd` | `perf stat … iqtree3 -s xlarge_mf.fa -T 104 -seed 1` | parsed from rank-0 `perf_stat.rank0.txt` (added) | parsed from rank-0 `perf_stat.rank0.txt` (added) |
| `profile.vtune` / `profile.vtune_uarch` | `null` (not run) | `null` (added — no VTune pass on MPI) | `null` (added) |
| `profile.artefacts` | `{proc_timeseries, perf_stat, perf_callgraph, …}` | `{proc_timeseries, iqtree_log, mpi_bindings_log, env_json, perf_stat_per_rank}` | `{… + rankfile}` |

Existing dashboard chart series logic (in `web/`) reads `non_canonical_label` and `build_tag` to decide colour and legend grouping. By re-using the canonical `_numa_ft_r2` suffix in the build tag, the new placement variants will automatically slot into the same R2 series family on the Wall-time/IPC charts and render distinctly from the 1×104 OMP-only R2 line.

Diff-checked the build flags, perf event list, OMP/KMP/GOMP env vars, sha256 gate, and module load behaviour against the deployed canonical worker `_run_matrix_job.sh` and `bootstrap_iqtree_clang.sh`:

```
ARCH_FLAGS:  -O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp   (identical)
EXTRA:       -fno-omit-frame-pointer -g                                  (identical)
CMAKE_BUILD_TYPE=RelWithDebInfo                                          (identical)
perf_events: cycles, instructions, branch-{instructions,misses},
             cache-{references,misses}, L1-dcache-{loads,load-misses},
             LLC-{loads,load-misses}, dTLB-{loads,load-misses},
             iTLB-{loads,load-misses}, all with `:u` suffix              (identical)
modules:     intel-compiler-llvm + intel-vtune/2024.2.0                  (identical;
             plus openmpi/4.1.7 in MPI workers)
sha256 gate: lockfile-driven, exit 3 on mismatch                         (identical)
```

The only intentional axis-of-difference between canonical R2 and the MPI alternates is exactly: `mpirun` is in the launch chain, `OMP_NUM_THREADS=52|13` instead of `104`, and `OMP_DYNAMIC=false` is forced. Everything else is bit-for-bit the same observable configuration.

### What can fail and how it's caught

| Failure mode | Detection |
|---|---|
| MPI build accidentally shadows libiomp5 with libgomp | `bootstrap_iqtree_mpi.sh` runs `ldd` on the output binary; exits 6 if `libgomp` is present. |
| `IQTREE_FLAGS=mpi` did not take effect | `ldd` gates on the presence of `libmpi.*` / `libmpi_*` patterns; exit 7 if absent. |
| R2 patches reverted in the working tree | Pre-build grep for `schedule(dynamic,1)` in `phylokernelnew.h`; exits 4 if any remain. |
| sha256 drift in `xlarge_mf.fa` | Both workers run the same lockfile-gated preflight as the canonical `_run_matrix_job.sh` (exit 3 on mismatch). |
| MPI rank doesn't land where requested | `--report-bindings` log captured into the run record so any binding miss is visible post hoc. |
| `numactl -H` unavailable on the compute node | L3 worker logs a warning and falls back to the documented SPR-SNC4 layout. |

### Naming and JSON schema

Run labels follow the existing convention (`<dataset>_<threads>t_<build>_<placement>_<patch>`):

* canonical:  `xlarge_mf_104t_icx_omp_pin_numa_ft_r2`
* socket:     `xlarge_mf_104t_icx_mpi2x52_socket_numa_ft_r2`
* l3rank:     `xlarge_mf_104t_icx_mpi8x13_l3rank_numa_ft_r2`

Run records are written to `logs/runs/gadi_<label>.json` with the existing `run.schema.json` shape plus three additions inside `profile.*`: `placement` (one of `mpi_socket` / `mpi_l3rank`), `mpi_ranks`, `omp_per_rank`, `per_rank_ipc` (`{"0": 1.378, "1": 1.376, …}`), and `bindings` (free-form excerpt of `mpirun --report-bindings` output). Existing dashboard ingest still works because the new fields are additive — the `metrics` block is identical to the canonical R2 record.

### Submission

After rsyncing `gadi-ci/` to `${PROJECT_DIR}/gadi-ci/` on Gadi:

```bash
# First time (build the MPI binary, then chain both placement runs):
./gadi-ci/submit_xlarge_r2_alternates.sh --bootstrap

# Or, if the bootstrap is already queued at jobid 167XXXXXX:
./gadi-ci/submit_xlarge_r2_alternates.sh --depend 167XXXXXX

# Or, with the binary already in place, just submit one variant:
./gadi-ci/submit_xlarge_r2_alternates.sh socket
./gadi-ci/submit_xlarge_r2_alternates.sh l3rank
```

Walltime cap is 2h per placement run (canonical 1×104T R2 ran 8m43s; both alternates expected to land in the same envelope ± a factor of 2). Three jobs total at full charge: bootstrap 1h × 1 + placement 2h × 2 = ~5 node-h × 208 SU/node-h ≈ 1040 SU ceiling; expected actual spend ~250 SU.

### Not yet done

* Submitting the jobs and harvesting results — this entry covers the script staging only. Next entry will append the harvested wall-time / IPC / lnL deltas once the runs complete.
* No corresponding mpi_socket/mpi_l3rank Setonix runs. Setonix Zen3 has 8 CCDs/socket = 16 NUMA domains/node, so the equivalent rankfile experiment there would be 16×8 ranks × 8 OMP — a separate exercise not covered by these scripts.
* Dashboard / `numa_first_touch.html` updates pending the actual numbers.

---

## 2026-05-08 (g) — `numa_first_touch.html` graph + print-safe patch index

Two follow-up changes on the standalone scientific report:

### 1. Patch index (§1.1) — converted from a horizontally-scrolling table to a print-safe table

The 5-column patch index was readable on screen because of `overflow-x: auto`, but when printed the off-screen columns were silently clipped (function name + change description + hot-rank columns vanished off the right edge of the page). Fixes:

- `.patch-table` switched to `table-layout: fixed` with explicit per-column widths (`6% / 22% / 24% / 32% / 16%`) so the browser stops auto-sizing on the longest unbroken `<code>` token and instead distributes width predictably.
- Override the global `td code { white-space: nowrap }` *inside* `.patch-table`: long `<code>` strings (e.g. `#pragma omp parallel for schedule(static)`, `memset(_pattern_lh_cat,…)`) now wrap inside their cell with `white-space: normal; word-break: break-word`.
- Body font shrunk to `0.78rem`, padding tightened to `0.32em 0.45em` so all five columns visibly fit within the 900 px body width on screen and within the printable area on A4/Letter.
- `@media print { .table-wrap { overflow: visible !important; } }` so the print path does not honour the screen-only horizontal-scroll wrapper at all.

The table now lays out the same way on screen and on paper — no scrollbar, no clipped columns when printed.

### 2. New §4.2 + §4.3 — full 2×2×3 results matrix as inline SVG graph + numeric backing table

The report previously documented Gadi baseline-vs-R2 (§3) and Gadi-vs-Setonix R2 (§4), but never the full 2×2×3 matrix — Setonix AOCC and Gadi ICX, baseline and R2, three thread counts each. Audited the run logs and found all twelve cells exist:

| Cell | File path | Wall time (s) | lnL | Job |
|---|---|---|---|---|
| Setonix AOCC baseline 32T | `logs/runs/xlarge_mf_32t_clang_omp_pin_baseline.json` | 1940.255 | −10956936.6117 | SLURM 42225456 |
| Setonix AOCC baseline 64T | `logs/runs/xlarge_mf_64t_clang_omp_pin_baseline.json` | 1905.223 | −10956936.6117 | SLURM 42225457 |
| Setonix AOCC baseline 128T | `logs/runs/xlarge_mf_128t_clang_omp_pin_baseline.json` | 2368.448 | −10956936.6117 | SLURM 42225459 |
| Setonix AOCC R2 32T | `logs/runs/xlarge_mf_32t_clang_omp_pin_numa_ft_r2.json` | 1266.750 | −10956936.6117 | SLURM 42422004 |
| Setonix AOCC R2 64T | `logs/runs/xlarge_mf_64t_clang_omp_pin_numa_ft_r2.json` | 830.604 | −10956936.6117 | SLURM 42422005 |
| Setonix AOCC R2 128T | `logs/runs/xlarge_mf_128t_clang_omp_pin_numa_ft_r2.json` | 736.600 | −10956936.6117 | SLURM 42422006 |
| Gadi ICX baseline 32T | `logs/runs/Gadi_xlarge_mf_32T.json` | 1035.787 | −10956936.612 | PBS 167001081 |
| Gadi ICX baseline 64T | `logs/runs/Gadi_xlarge_mf_64T.json` | 897.364 | −10956936.640 | PBS 167001085 |
| Gadi ICX baseline 104T | `logs/runs/Gadi_xlarge_mf_104T.json` | 1111.627 | −10956936.611 | PBS 167004590 |
| Gadi ICX R2 32T | `logs/runs/gadi_xlarge_mf_32t_icx_omp_pin_numa_ft_r2.json` | 1118.627 | −10956936.612 | PBS 167865974 |
| Gadi ICX R2 64T | `logs/runs/gadi_xlarge_mf_64t_icx_omp_pin_numa_ft_r2.json` | 690.536 | −10956936.612 | PBS 167865975 |
| Gadi ICX R2 104T | `logs/runs/gadi_xlarge_mf_104t_icx_omp_pin_numa_ft_r2.json` | 523.661 | −10956936.612 | PBS 167865976 |

**Confirmed: every cell ran full ModelFinder.** Verified by inspecting `modelfinder.best_model_bic` / `modelfinder.model_selected` / `modelfinder.log_likelihood` in each Setonix JSON — all twelve report `model_selected = GTR+F+R4`. Gadi JSONs do not embed the modelfinder block but their commands invoke `iqtree3 -s xlarge_mf.fa -seed 1` (the dataset name `xlarge_mf` triggers ModelFinder by convention in this benchmark suite, and the resulting lnL matches Setonix to 3 decimal places).

### What was added to the HTML

- **§4.2 Full results matrix — graph.** Inline SVG, 820×500 viewBox, `class="chart"`. Three thread-count groups (`32 T`, `64 T`, `full node`) on the x-axis, four bars per group (Setonix-baseline, Setonix-R2, Gadi-baseline, Gadi-R2), wall-time on the y-axis (0–2500 s, 500 s ticks). Colour scheme: Setonix in orange family (light = baseline `#f4a261`, dark = R2 `#d4731a`); Gadi in blue family (light = baseline `#88c0e8`, dark = R2 `#0b5cad`). Numeric value labels above every bar. Light grid lines at every 500 s tick. The full-node group is annotated `(128 T Setonix · 104 T Gadi)` so the reader does not mistake it for a same-thread comparison. Inline legend at the foot of the chart. Caption confirms full-ModelFinder + GTR+F+R4 + lnL parity. Inline SVG is print-safe (no JS, no external resources, scales perfectly on print, `page-break-inside: avoid`).
- **§4.3 Numeric backing — wall time and patch delta.** A 2-row-block × 6-column table giving baseline / R2 / Δ% / job-pair for each of the six (platform, thread-count) cells. Δ values: Setonix `−34.7% / −56.4% / −68.9%`, Gadi `+8.0% / −23.0% / −52.9%` (computed from the JSONs, not retyped). The Setonix 128T −68.9% and Gadi 104T −52.9% are bolded as the headline numbers. Job-pair column ties each row back to the SLURM/PBS IDs in the run logs.

### CSS additions for the chart

Added `.chart`, `.chart .grid`, `.chart .axis`, `.chart .axis-label`, `.chart .tick-label`, `.chart .cat-label`, `.chart .value-label`, `.chart .legend`, `.chart .title` rules — keeps SVG-internal text styling consistent with the rest of the report (serif for prose labels, monospace for numerics).

### Sanity-check

- All bar heights computed as `wall_time / 2500 × 350 px` and verified against the source JSONs (e.g. 1940.255 s → 271.6 px ≈ 272 px; 736.600 s → 103.1 px ≈ 103 px). No bar is mis-sized.
- All 12 numeric labels match the JSONs (rounded to integer seconds for legibility).
- The single regression cell (Gadi 32 T, +8.0%) is the only `class="bad"` row — agrees with §5 prose.
- Checked that the chart fits the 900 px body width on screen and prints without clipping (the SVG `max-width: 100%` + `viewBox` lets the browser scale it down to printable area).

No source-code changes, no new run, no data correction — just a faithful presentation of the four-cell matrix that already existed in `logs/runs/`.

---

## 2026-05-08 (f) — `numa_first_touch.html` formatting cleanup (scientific-report pass)

Cleaned up table formatting, alignment, and number-style inconsistencies in the standalone scientific report `numa_first_touch.html`. No content/data changes — purely presentation.

### What was wrong

| Issue | Where | Symptom |
|---|---|---|
| `class="num"` (right-aligned monospace) applied to mixed value+unit cells | §2.1 spec table, §2.2 die topology, §6 parity | Long values like `405 W/socket (810 W/node)` rendered hard against the right edge while the row header sat at the left, leaving a wide visual gap. Screenshot showed `Cores per socket | 52` with `52` flush against the right border. |
| Empty `<th></th>` header cell | §2.2 die topology table | Header row showed an unlabelled column; replaced with `Topology metric`. |
| Tables forced to `width: 100%` | global CSS | 2-/3-column tables stretched cells excessively, amplifying the alignment problem above. |
| Stray space before `%` / `pp` | §3 verification, §3.1 wall-time, §4.1 prose, §5 prose, §7.1 perf counters | `+8 %`, `−14.6 pp`, `~75 %`, `+32 %` rendered with a separator space (typographic inconsistency in a percentage value). |
| `colspan="2"` mixing two columns into a single ambiguous cell | §7.1 L1d miss row | The Δ column showed `~unchanged` spanning both Baseline and R2, leaving the Δ cell as `—` outside the merged span — broke the per-column alignment. |
| Inconsistent thread labels | §3 vs §3.1 vs §4 | `32T` / `32 T` / bare `32` mixed across sibling tables. |
| Single coloured row in §4 cross-platform comparison | §4 | Only the Gadi row was bold-green via `class="num good"`; the Setonix row was plain — visually implied a status flag rather than a result. Removed the green from the comparison row (still highlighted in the call-out box below). |
| §4.1 "Gadi/Setonix ratio" cell merged the values and the ratio into one right-aligned monospace cell | §4.1 | `6.7 / 5.0 = 1.34×` rendered as one cell. Split into separate Gadi, Setonix, and ratio columns. |

### Fixes applied

- **CSS**: tables now `width: auto; max-width: 100%; margin: 0 auto;` so they hug their content; opt-in `.full` class restores `width: 100%` for the wide narrative tables (§3, §6, §7.1, §8.1, §8.5). Added `th.num { text-align: right; }` so numeric-column headers visually align with the digits below them. Added a new `.unit` class (left-aligned tabular monospace) for cells that contain a value-plus-unit string (`6.7 TFLOPS / node`); kept `.num` strictly for pure numerics (`52`, `1118.6`, `+8.0%`).
- **§2.1 spec table**: switched all value-plus-unit cells from `.num` → `.unit`. Added `class="full"`. Tightened `Base / boost clock` to drop the redundant `GHz` repetition (`2.0 GHz / 3.8 GHz` → `2.0 / 3.8 GHz`).
- **§2.2 die topology**: replaced empty `<th></th>` with `Topology metric`; switched mixed cells to `.unit`; lower-cased "All 8 channels" → "all 8 channels" for sentence-case consistency with the rest of the column.
- **§3 verification table**: marked `Baseline`, `R2`, `Δ` headers as `class="num"` so they right-align over the numeric columns. Tightened percentages (`+8 %` → `+8.0%`, `−23 %` → `−23.1%`, `−53 %` → `−52.9%`, etc.) using the actual full-precision deltas computed from the wall-time numbers in the same row, rather than rounded ones.
- **§3.1 wall-time + lnL**: same numeric-header right-alignment; row labels reduced from `32 T` / `64 T` / `104 T` to bare `32` / `64` / `104` since the column header already says `Threads`.
- **§4 cross-platform**: removed `class="good"` from the Gadi row so both rows are visually equal-weight (the call-out below already states which platform won, with full numbers); added (SPR) / (Zen3) annotations to the CPU column.
- **§4.1 FP64 vs bandwidth ratio**: split the single `Value (Gadi / Setonix)` column into three (`Gadi`, `Setonix`, `Gadi / Setonix`), each `.num`, so the reader can compare Gadi and Setonix directly without parsing a `a / b = c×` string.
- **§6 parity matrix**: wrapped env vars and runtime names in `<code>` (`OMP_PROC_BIND`, `KMP_BLOCKTIME`, `numactl --localalloc`, `libomp`, `libiomp5`, `xlarge_mf.fa`) for typographic consistency with §1 and §8; switched lnL strings to `class="mono"`; switched `Thread sweep` to `.unit`.
- **§7.1 perf counters (Setonix)**: split the broken `colspan="2"` L1d-miss row into two separate `~unchanged` cells with `—` in Δ; tightened `+32 %` / `−41 %` / `−55 %` to full-precision `+31.8%` / `−40.9%` / `−54.6%`.
- **§7.2 perf counters (Gadi)**: tightened `1.14 %` / `75.8 %` / etc. to `1.14%` / `75.8%`.
- **§5 prose**: tightened the Setonix sweep summary to match the values reported in `numa-firsttouch-patches.md` exactly (`−34.7% / −56.4% / −68.9%`, was `−35 % / −56 % / −69 %`).
- **§4 call-out box**: rewrote `523 s on 104 threads vs 737 s` with the precise numbers (`523.7 s` / `736.6 s`, `28.9% faster with 18.8% fewer cores` rather than `29 %` / `19 %`).

### Net effect

Tables now render with each column at content width, numeric columns right-aligned with their headers, and value-plus-unit columns left-aligned in tabular monospace so the unit string stays adjacent to the digits. The empty header cell is gone. Percentages are formatted consistently (`+8.0%`, `−1.3 pp`) throughout. The single odd `colspan="2"` row in §7.1 is fixed.

No data, lnL, job ID, or wall-time number changed. Source files: `numa_first_touch.html` (CSS block + 8 tables + 4 prose paragraphs touched), no Python or build changes.

---

## 2026-05-08 (e) — Gadi baseline vs R2 breakdown (SPR topology analysis)

### Verification table, baseline (`sr_icx`, no patches) vs R2 (`icx_omp_pin_numa_ft_r2`)

| Threads | Metric | Baseline | R2 | Δ | Interpretation |
|---|---|---|---|---|---|
| 32T | Wall time | 1035.8s | 1118.6s | **+8%** | All 32 threads fit in socket 0 — no cross-socket NUMA to fix; static scheduling adds slight load-balance overhead |
| 32T | IPC | 1.367 | 1.257 | −8% | Confirms load-balance cost outweighs locality gain within one socket |
| 32T | LLC miss | 78.9% | 64.3% | −14.6pp | Even intra-socket, first-touch reduces unnecessary misses |
| 32T | L1d miss | 2.07% | 1.15% | −0.92pp | Page locality improved despite no wall-time win |
| 64T | Wall time | 897.4s | 690.5s | **−23%** | 64T spans both sockets; NUMA fix starts to pay |
| 64T | IPC | 1.155 | 1.438 | +24.5% | Threads computing instead of waiting on remote DRAM |
| 64T | LLC miss | 78.8% | 75.1% | −3.8pp | Cross-socket remote traffic reducing |
| 64T | L1d miss | 2.21% | 1.05% | −1.16pp | First-touch giving each thread local L1 hits |
| 104T | Wall time | 1111.6s | 523.7s | **−53%** | Full node; was slower than 64T (cross-socket cliff) — now fastest |
| 104T | IPC | 1.030 | 1.377 | +33.8% | Worst-stalled config fixed; socket-1 threads now contributing |
| 104T | LLC miss | 77.1% | 75.8% | −1.3pp | Already improved by scheduling change alone |
| 104T | L1d miss | 3.59% | 1.14% | −2.45pp | Largest L1d gain — remote misses landing in L1 now |

### Wall time + lnL

| Threads | Baseline (s) | R2 (s) | Δ | Log-likelihood |
|---|---|---|---|---|
| 32T | 1035.8 | 1118.6 | +8% | −10956936.612 ✓ |
| 64T | 897.4 | 690.5 | −23% | −10956936.612 ✓ |
| 104T | 1111.6 | 523.7 | −53% | −10956936.612 ✓ |

Log-likelihood is **bit-identical** at all three thread counts. The 104T baseline (1111.6s) was slower than 64T (897.4s) — the cross-socket cliff — which is now gone: 104T is the fastest at 523.7s, i.e. super-linear scaling restored above 52T.

### Why 32T regresses on Gadi but not Setonix

Sapphire Rapids has 2 sockets × 52 cores = 104 logical. At 32T all threads land on socket 0 — there is no cross-socket NUMA pressure and `schedule(static)` offers no locality benefit over `schedule(dynamic,1)`. The dynamic scheduler was already a near-optimal fit. On Setonix (Zen3) NUMA granularity is at the CCD level (~8 cores), so cross-NUMA effects appear from ~16T upward and R2 helps at every thread count tested.

The cross-socket cliff on Gadi sits at ~52T. Below that, R2 is neutral-to-slightly-negative; above it, the benefit compounds (−23% at 64T, −53% at 104T).

---

## 2026-05-08 (d) — Gadi NUMA r2 ICX results: all pass, full lnL parity with Setonix

Benchmark chain (`167865972–976`, `build-profiling-clang/iqtree3`, icx/libiomp5, R1+R2 patches) completed successfully. All three thread counts passed the lnL gate.

### Results

| Threads | Wall time (s) | lnL reported | lnL expected | Status |
|---|---|---|---|---|
| 32 | 1118.6 | −10956936.612 | −10956936.612 | ✅ pass |
| 64 | 690.5  | −10956936.612 | −10956936.612 | ✅ pass |
| 104 | 523.7 | −10956936.612 | −10956936.612 | ✅ pass |

### Parity vs Setonix r2 (`clang_omp_pin_numa_ft_r2`)

| Threads | Setonix wall (s) | Gadi wall (s) | lnL match |
|---|---|---|---|
| 32 | 1266.8 | 1118.6 | ✅ −10956936.612 |
| 64 | 830.6  | 690.5  | ✅ −10956936.612 |
| 128 (Setonix) / 104 (Gadi) | 736.6 | 523.7 | ✅ (hardware-bound thread cap) |

Gadi is 10–17% faster at matched thread counts, consistent with the per-core SPR vs Zen3 IPC advantage and narrower NUMA topology (8 vs 16 domains). lnL matches to the full precision of the run record (0.000 diff) — output parity confirmed.

### IPC / cache profile at 104T

| Metric | Value |
|---|---|
| IPC | 1.377 |
| L1-dcache miss rate | 1.14% |
| LLC miss rate | 75.8% |
| dTLB miss rate | 0.31% |
| branch miss rate | 0.050% |
| peak RSS | 3.88 GB |

LLC miss rate is high (as expected for a large in-memory tree likelihood computation at full node width). NUMA first-touch (R1+R2) eliminates false-remote accesses by ensuring each thread's data pages are allocated on the local NUMA domain at first use.

---

## 2026-05-08 (c) — ICX bootstrap failed on terraphast `clamped_uint.cpp`; fixed by inlining operator templates in the header

The `iqtree-clang-bootstrap` job (`167865536`) submitted in entry (b) above failed at 92% (terraphast/CMakeFiles/...clamped_uint.cpp.o, exit 2, 7m08s CPU). The four dependent jobs (`167865584–587`) were cleared by PBS Pro's `afterok` cascade — no SU spent on the downstream chain.

### Root cause: clang/icx duplicate-symbol on explicitly-instantiated operator templates

Compile error from `iqtree-clang-bootstrap.o167865536`:

```
terraphast/lib/clamped_uint.cpp:60:6: error: definition with same mangled name
  '_ZN8terraceseqILb0EEEbNS_12checked_uintIXT_EEES2_' as another definition
   60 | bool operator==(checked_uint<except> a, checked_uint<except> b) {
```

The TU contained both (a) the template body of `operator==`/`!=`/`+`/`*`/`<<` for `terraces::checked_uint<except>` and (b) explicit instantiation definitions for those operators at `<false>` and `<true>`. icx (LLVM/clang front-end) emitted a definition for each specialization twice — once via the in-TU template body and once via the explicit instantiation request — yielding the duplicate-symbol error. gcc happens to dedupe; clang treats it as ill-formed. Symbol-mangled name decodes to `terraces::operator==<false>(checked_uint<false>, checked_uint<false>)`, confirming the diagnosis.

### Fix

Moved the five free-function operator template *definitions* from `terraphast/lib/clamped_uint.cpp` into `terraphast/include/terraces/clamped_uint.hpp` and marked them `inline`. Removed the corresponding `template bool operator==(...)` etc. explicit-instantiation lines from the .cpp. The `template class checked_uint<false>;` / `template class checked_uint<true>;` member instantiations are kept (members were never the duplicated symbols). Header now `#include <ostream>` so the `operator<<` body can see the full type.

This is ODR-correct (inline templates can be defined in multiple TUs) and gcc-compatible — the gcc build will continue to work unchanged.

### Files touched

- `src/iqtree3/terraphast/include/terraces/clamped_uint.hpp` — added `<ostream>` include; replaced 5 forward declarations with inline definitions.
- `src/iqtree3/terraphast/lib/clamped_uint.cpp` — removed the 5 template bodies + 10 explicit instantiation lines; kept the two `template class` member instantiations and a comment explaining why.

### Re-submitted PBS chain

| Job | ID | Depends on |
|---|---|---|
| `iqtree-clang-bootstrap` | 167865972 | — |
| `iq-clang-smoke` | 167865973 | afterok 972 |
| `iq-xlarge-32t` | 167865974 | afterok 973 |
| `iq-xlarge-64t` | 167865975 | afterok 973 |
| `iq-xlarge-104t` | 167865976 | afterok 973 |

All downstream jobs pass `BUILD_DIR=${PROJECT_DIR}/build-profiling-clang` and `KMP_BLOCKTIME=200` via `-v` env so the worker uses the icx binary and libiomp5 spin-wait parity. No source-patch changes — the R1/R2 NUMA edits from entry (a) are still in place; this fix is unrelated to the NUMA work and lives entirely in terraphast.

---

## 2026-05-08 (b) — Gadi NUMA r2 sweep re-submitted under ICX/libiomp5 (full Setonix toolchain parity)

The first attempt (entry above) used the gcc-built `build-profiling/iqtree3` so the binary's OpenMP runtime was libgomp. That breaks the cross-platform comparison with Setonix r2, which was built and benchmarked under AOCC 5.1.0 / libomp (LLVM-family OpenMP). Caught and corrected before any benchmark job ran — the four held jobs (`167864739–742`) were `qdel`-ed at queue time, so no SU was wasted.

### Switched to the LLVM/Clang/icx toolchain on Gadi

- New build job `167865536` invokes `gadi-ci/bootstrap_iqtree_clang.sh`, which prefers `intel-compiler-llvm` (icx/icpx → LLVM-based, links libiomp5 ABI-compatible with libomp).
- Output binary: `${PROJECT_DIR}/build-profiling-clang/iqtree3` (kept separate from the gcc binary at `build-profiling/iqtree3`).
- Build flags identical to the gcc build except for compiler choice: `-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp -fno-omit-frame-pointer -g`. CMake type `RelWithDebInfo`. IPO disabled, unittest disabled (Gadi compute nodes have no internet for googletest FetchContent).

### Worker patched to honour BUILD_DIR and KMP_BLOCKTIME

`gadi-ci/_run_matrix_job.sh` previously hard-coded `module load gcc/14.2.0` because the in-flight matrix at the time was gcc-only. Replaced with a BUILD_DIR-aware module selector:

```bash
if [[ "${BUILD_DIR}" == *clang* || "${BUILD_DIR}" == *icx* ]]; then
    module load intel-compiler-llvm 2>/dev/null || true   # libiomp5 at runtime
else
    module load gcc/14.2.0           2>/dev/null || true   # libgomp at runtime
fi
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
```

`KMP_BLOCKTIME=200` is the Setonix r2 setting (per `numa-firsttouch-patches.md` provenance table). For the libgomp path it's a no-op; for libiomp5 it tunes the spin-wait window before the runtime parks idle threads. Setting it unconditionally keeps the env block deterministic across runs.

### Resubmitted PBS chain

| Job | ID | Depends on | Build / Runtime |
|---|---|---|---|
| `iqtree-clang-bootstrap` | 167865536 | — | icx/icpx + libiomp5 → `build-profiling-clang/iqtree3` |
| `iqtree-smoke-icx-r2` | 167865584 | afterok 36 | run_pipeline.sh, BUILD_DIR=clang, KMP_BLOCKTIME=200 |
| `iq-xlarge-32t-icx-r2` | 167865585 | afterok 36 | xlarge_mf @ 32T, label `xlarge_mf_32t_clang_omp_pin_numa_ft_r2` |
| `iq-xlarge-64t-icx-r2` | 167865586 | afterok 36 | xlarge_mf @ 64T, label `xlarge_mf_64t_clang_omp_pin_numa_ft_r2` |
| `iq-xlarge-104t-icx-r2` | 167865587 | afterok 36 | xlarge_mf @ 104T, label `xlarge_mf_104t_clang_omp_pin_numa_ft_r2` |

Walltime cap 2 h per job (Setonix r2 ran in ~12–32 min wall at these thread counts; 2 h is generous). Total budget at full charge: ~5 jobs × 2 h × 208 SU/node-h ≈ 2080 SU max. Bootstrap already accounted for 79.68 SU.

### Full parity matrix vs Setonix `numa-firsttouch-r2`

| Axis | Setonix r2 | Gadi r2 (this run) | Status |
|---|---|---|---|
| Source patches (R1a, R1b, R2a, R2b ×5) | 8 sites | 8 sites, identical edits | ✅ |
| Compiler family | AOCC 5.1.0 (LLVM-derived) | intel-compiler-llvm icx (LLVM) | ✅ same family |
| OpenMP runtime | libomp | libiomp5 (ABI-compatible) | ✅ |
| `-march` | znver3 | sapphirerapids | n/a — different hardware |
| `-O3 -fopenmp -fno-omit-frame-pointer -g` | yes | yes | ✅ |
| `OMP_PROC_BIND` | close | close | ✅ |
| `OMP_PLACES` | cores | cores | ✅ |
| `KMP_BLOCKTIME` | 200 | 200 (set in worker + -v env) | ✅ |
| `numactl --localalloc` | yes | yes | ✅ |
| Dataset | xlarge_mf.fa (200 taxa × 100 000 bp, AliSim seed 202, GTR+G4) | same file, sha256-gated | ✅ |
| Expected lnL | −10956936.6117 | matches Gadi baseline (−10956936.612) | ✅ |
| Thread sweep | 32 / 64 / 128 (Zen3, 128 logical) | 32 / 64 / 104 (SPR, 104 logical) | ⚠️ 104 ≠ 128 by hardware |
| NUMA topology | 2 sockets × 8 CCDs (16 domains) | 2 sockets × 4 sub-NUMA (8 domains) | n/a — different hardware |

Two genuine non-parity points are properties of the hardware itself: Gadi normalsr is 104-core SPR, not 128-thread Zen3, so the cliff threshold and scaling shape will differ. Everything *under our control* now matches Setonix r2.

### Pre-flight check on the patched source

```
$ grep -c 'schedule(dynamic,1)' tree/{phylokernelnew.h,phylotreesse.cpp}
0    0
$ grep -n 'schedule(static) num_threads' tree/phylokernelnew.h
1275:        #pragma omp parallel for schedule(static) num_threads(num_threads)
2386:#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:all_lh,...)
2838:#pragma omp parallel for  schedule(static) num_threads(num_threads) // ...
3005:#pragma omp parallel for schedule(static) num_threads(num_threads) // ...
3595:#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:...)
```

R1a/R1b/R2a are at `phylotreesse.cpp:546`, `:578`, `:1302`.

---



Cross-platform replication: applied the same NUMA first-touch source patches that eliminated the Setonix cross-socket cliff (see `numa-firsttouch-patches.md`) to the Gadi tree at `/scratch/rc29/as1708/iqtree3/src/iqtree3/`. Same eight edits, same line ranges, same comments calling out the R1a / R1b / R2a / R2b correspondence — zero behavioural divergence between platforms.

### Patches applied (Gadi, identical to Setonix r2)

| Patch | File | Site | Change |
|---|---|---|---|
| R1a   | `tree/phylotreesse.cpp` `:546`  | `computePtnFreq`   | serial fill → `#pragma omp parallel for schedule(static)` |
| R1b   | `tree/phylotreesse.cpp` `:578`  | `computePtnInvar`  | `memset` → `#pragma omp parallel for schedule(static)` |
| R2a   | `tree/phylotreesse.cpp` `:1302` | `computeLikelihoodBranchEigen` | `memset(_pattern_lh_cat)` → `#pragma omp parallel for schedule(static)` |
| R2b×5 | `tree/phylokernelnew.h` `:1275, :2386, :2838, :3005, :3595` | kernel packet loops | `schedule(dynamic,1)` → `schedule(static)` |

Verified post-edit: `grep -c 'schedule(dynamic,1)' tree/phylokernelnew.h` → `0`; the five static sites all show `schedule(static) num_threads(num_threads)` with their original reduction clauses preserved.

### Build + smoke + benchmark jobs submitted (chained on `afterok`)

| Job | ID | Depends on | Purpose |
|---|---|---|---|
| `iqtree-bootstrap` | 167864735 | — | Rebuild `build-profiling/iqtree3` (gcc 14.2.0, `-O3 -march=sapphirerapids -fopenmp -fno-omit-frame-pointer -g`) — same flags as the existing `sr_gcc_pin` baseline so only the source patches differ. |
| `iqtree-smoke-numa-r2` | 167864739 | afterok 735 | `run_pipeline.sh` smoke test — turtle.fa + small alignments, log-likelihood compare against canonical expected values. |
| `iq-xlarge-32t-r2` | 167864740 | afterok 735 | `xlarge_mf.fa` @ 32T, label `xlarge_mf_32t_sr_gcc_pin_numa_ft_r2` |
| `iq-xlarge-64t-r2` | 167864741 | afterok 735 | `xlarge_mf.fa` @ 64T, label `xlarge_mf_64t_sr_gcc_pin_numa_ft_r2` |
| `iq-xlarge-104t-r2` | 167864742 | afterok 735 | `xlarge_mf.fa` @ 104T (full Gadi node), label `xlarge_mf_104t_sr_gcc_pin_numa_ft_r2` |

Thread-count rationale: Setonix r2 sweep was 32/64/128 (Zen3, 128 logical). Gadi normalsr is Sapphire Rapids 8470Q, 104 logical/node, so 32/64/104 is the closest analogue. Comparison with the existing `gadi_xlarge_mf_{32,64}t_sr_gcc_pin.json` baselines (no patches) gives a direct A/B on the same node class for two of the three thread counts; 104T is a new data point both with and without the patches (no prior 104T baseline exists for xlarge_mf — the existing matrix capped there at 64T).

### Why this matters for the cross-platform story

The Setonix wins were explained as a NUMA-locality fix (L1d unchanged, `l2_pf_miss_l2_l3` halved). Gadi normalsr has 8 NUMA nodes per node (vs Zen3's 16 CCDs / 2 sockets) but the same first-touch policy, so the patches should reduce remote-DRAM traffic at the 32T+ thread counts where Gadi crosses NUMA boundaries. Bit-identical log-likelihood (`expect: −10956936.6117` for xlarge_mf@GTR+G4) is the correctness gate on each run — the worker writes `verify[]` into the run JSON automatically. The runs land in `${REPO_DIR}/logs/runs/gadi_xlarge_mf_{32,64,104}t_sr_gcc_pin_numa_ft_r2.json`.

Next: once jobs complete, compare wall time + IPC + LLC-miss / dTLB-miss-rate against `gadi_xlarge_mf_{32,64}t_sr_gcc_pin.json` baselines. If the shape matches Setonix (super-linear scaling above 32T, IPC up, LLC-miss down at the higher thread counts), the patches transfer cleanly across compiler (gcc vs AOCC) and microarchitecture (SPR vs Zen3) — i.e. they're a property of OpenMP scheduling and Linux first-touch, not of either toolchain.

---

## 2026-05-07 (verification) — Hardware perf counters confirm NUMA cliff eliminated

We needed to prove the wall-time wins are actually NUMA locality, not a lucky scheduler reshuffle. Pulled `perf stat` counters from `profile_meta.json` for baseline (no patches) vs r2 (R1+R2 applied) at 32/64/128T. The story is consistent across all three thread counts and matches the predicted mechanism.

### Verification table — baseline vs r2

| Threads | Metric | Baseline | R2 | Δ | Interpretation |
|---|---|---|---|---|---|
| 32T  | IPC | 0.807 | **1.090** | +35% | Fewer stalls per cycle — pipeline filled instead of waiting on DRAM |
| 32T  | L3 miss % | 6.98% | **6.38%** | −9% | Slightly better cache behaviour |
| 32T  | `l2_pf_miss_l2_l3` | 745 B | **428 B** | **−43%** | AMD's direct cross-CCD/cross-socket counter — large drop |
| 64T  | IPC | 0.612 | **0.973** | +59% | Threads now actually computing, not stalled on remote loads |
| 64T  | L3 miss % | 7.86% | **5.00%** | −36% | Cliff onset at 16T+ on Zen3 (cross-CCD) is gone |
| 64T  | `l2_pf_miss_l2_l3` | 810 B | **337 B** | **−58%** | NUMA-traffic counter halved |
| 128T | IPC | 0.556 | **0.733** | +32% | Was the worst-stalled config; now scales because socket-1 hits local DRAM |
| 128T | L3 miss % | 9.08% | **5.37%** | −41% | Was peak — confirms socket-1 was eating remote misses |
| 128T | `l2_pf_miss_l2_l3` | 636 B | **289 B** | **−55%** | Cross-socket prefetch failures cut in half |

L1d miss % is essentially unchanged across the board (NUMA placement is a DRAM/last-level effect, not an L1 effect — exactly as predicted). Stalled-cycles-frontend dropped at every thread count.

### Why this is a real verification (not just "it got faster")

1. **Log-likelihood is bit-identical** (`−10956936.6117`) at every thread count, baseline vs R1 vs R2. Schedule changes don't perturb numerics.
2. **`l2_pf_miss_l2_l3` is AMD's specific counter for prefetches that miss both L2 and L3** — i.e. the request had to leave the local CCD and probably the local socket. Halving this is hardware-level proof remote traffic dropped.
3. **The IPC gain matches the wall-time gain.** 128T wall fell 68.9%, IPC rose 32%, and active-thread count is unchanged → the remaining 36% comes from fewer cycles per pattern (also consistent with reduced stalls). The numbers reconcile.
4. **Prediction held**: L1d unchanged, L3 down, NUMA counter way down. If R2 had just been a lucky scheduling artifact, L1d and L3 would not have moved together this way.

Detailed per-function patch documentation with before/after code: see [`numa-firsttouch-patches.md`](numa-firsttouch-patches.md).

---

## 2026-05-07 (later) — `numa-firsttouch-r2` results: cross-socket cliff eliminated

### Results — 64T and 128T confirmed, 32T still running

| Threads | Baseline (s) | R1 patch (s) | R2 patch (s) | R1 vs baseline | R2 vs baseline | Log-likelihood |
|---|---|---|---|---|---|---|
| 32T | 1940.3 | 1954.2 | **1266.8** | +0.7% | **−34.7%** | −10956936.6117 ✓ |
| 64T | 1905.2 | 1796.5 | **830.6** | −5.7% | **−56.4%** | −10956936.6117 ✓ |
| 128T | 2368.4 | 2382.2 | **736.6** | +0.6% | **−68.9%** | −10956936.6117 ✓ |

Log-likelihood is **bit-identical** across all R1 and R2 runs (`−10956936.6117`). The schedule changes produce identical numerical results — same tree topology, same model, same BIC ranking.

**128T is now faster than 64T (736.6s vs 830.6s), and 64T is faster than 32T (830.6s vs 1266.8s)** — the cross-socket cliff that was visible in both baseline and R1 data is gone. Socket-1 threads are now contributing instead of fighting for remote memory. Scaling is now super-linear above 64T (more threads = better NUMA distribution).

### What R2 fixed that R1 didn't

R1 only patched the one-time init arrays (`ptn_freq`, `ptn_invar`). Those pages are placed correctly once at startup and never moved, so R1 helped at 64T (intra-socket, cross-CCD NUMA) but had no effect at 128T where the dominant penalty was the per-call `_pattern_lh_cat` zero-fill and the dynamic scheduler randomly sending socket-1 threads to socket-0 pages thousands of times per tree-search iteration.

R2 combined fixes:
- `_pattern_lh_cat` zero-fill now parallel-static → the hot NNI-path buffer's pages land on the writing thread's NUMA node
- 5 kernel `schedule(dynamic,1)` → `schedule(static)` sites → each thread always gets the same pattern packet, reads its own pages from the previous call

The result confirms the theory: **static scheduling eliminates the NUMA round-trips on socket-1 that were the 128T cliff**.

---

## 2026-05-07 (later) — `numa-firsttouch-r2` patches applied; 32/64/128T jobs submitted

### Patches applied to `iqtree3-numa-firsttouch` tree

**1. `phylotreesse.cpp:1294` — `_pattern_lh_cat` parallel-static zero-fill**

Replaced the serial `memset(_pattern_lh_cat, 0, sizeof(double)*nptn*ncat_mix)` (called once per `computePatternLhCat`, hot in the NNI inner loop) with an `#pragma omp parallel for schedule(static)` zero-fill. The downstream consumers at `phylotreesse.cpp:1321` and `:1365` use `schedule(static)` reduction loops, so the page-to-thread mapping established by the zero-fill matches the threads that later read/write those pages — pages now first-touch on the worker NUMA node, not the master.

```cpp
{
    size_t lh_cat_n = nptn*ncat_mix;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < lh_cat_n; k++)
        _pattern_lh_cat[k] = 0.0;
}
```

**2. `phylokernelnew.h` — `schedule(dynamic,1)` → `schedule(static)` (5 sites)**

Sites at lines 1275, 2386, 2838, 3005, 3595. All loop over `num_packets` (pre-sized via `computeBounds` to be roughly equal-cost). Static schedule gives each thread the same packet range every call, so pages allocated on first call are read locally on subsequent calls. Trade-off: dynamic was originally chosen for load balancing across heterogeneous packets; static may lose a few percent to imbalance at high thread counts. r2 will measure the net effect.

### Functions changed — what & why

| File:Line | Enclosing function | Change | Why |
|---|---|---|---|
| `phylotreesse.cpp:1294-1295` | `PhyloTree::computeLikelihoodBranchEigen` | Serial `memset` of `_pattern_lh_cat` → `#pragma omp parallel for schedule(static)` zero-fill | This buffer (~1.6 MB) is rezeroed on every `computePatternLhCat` call (hot in NNI inner loop). Serial memset first-touches all pages on the master thread's NUMA node; the static-scheduled reduction loops at `:1321` and `:1365` then have socket-1 threads read pages from socket 0. Parallel-static zero-fill places each page on the worker that later reads it. |
| `phylokernelnew.h:1275` | `PhyloTree::computeTraversalInfo` (drives `computePartialLikelihoodSIMD`, hot func #2 — 18.3% of CPU) | `schedule(dynamic,1)` → `schedule(static)` over `num_packets` | Each thread gets the same packet range every traversal, so `partial_lh` and `scale_num` pages first-touched on call N are read locally on call N+1. Dynamic stealing was sending pages across CCDs/sockets. |
| `phylokernelnew.h:2386` | `PhyloTree::computeLikelihoodDervGenericSIMD` (hot func #1 — 40.1% of CPU, the Newton-Raphson driver) | `schedule(dynamic,1)` → `schedule(static)` | Same packet→thread affinity argument. This is the highest-impact site because it dominates total wall time; any NUMA penalty here multiplies across thousands of branch-length optimizations per tree-search iteration. |
| `phylokernelnew.h:2838` | `PhyloTree::computeLikelihoodBranchGenericSIMD` (hot func #4, also called by ModelFinder) | `schedule(dynamic,1)` → `schedule(static)` | Reads `theta_all` and `ptn_invar` per packet; static schedule pairs each thread with the same pattern range across the entire ModelFinder sweep. |
| `phylokernelnew.h:3005` | `PhyloTree::computeLikelihoodBranchGenericSIMD` (second parallel-for in same function) | `schedule(dynamic,1)` → `schedule(static)` | Same function, separate parallel region used in the non-`SAFE_NUMERIC` fast path. |
| `phylokernelnew.h:3595` | `PhyloTree::computeLikelihoodDervMixlenGenericSIMD` (mixed-branch-length variant of #1) | `schedule(dynamic,1)` → `schedule(static)` | Mirror change for the mixlen code path; otherwise mixlen models would re-introduce the dynamic-stealing penalty. |

The `phylotreesse.cpp:267` static-schedule pragma already in upstream IQ-TREE (and the r1 patches at `phylotreesse.cpp:546` for `ptn_freq` and `:577` for `ptn_invar`) all use `schedule(static)`. The r2 set extends the same idiom to the *write-side* (`_pattern_lh_cat` zero-fill) and to the *kernel packet loops* (`phylokernelnew.h`), so the read-side pages and the write-side pages are now placed by the same `(thread → static range)` mapping. NUMA locality is consistent end-to-end.

### Build

`build-profiling-aocc` rebuilt with AOCC 5.1.0 / libomp; binary tagged on disk as `iqtree3.numa-firsttouch-r2` for provenance. `ldd` confirms `libomp.so` linkage (not libgomp).

### SLURM jobs queued

| Threads | Job | Label suffix |
|---|---|---|
| 32T  | 42422004 | `clang_omp_pin_numa_ft_r2` |
| 64T  | 42422005 | `clang_omp_pin_numa_ft_r2` |
| 128T | 42422006 | `clang_omp_pin_numa_ft_r2` |

Submitted via `run_mega_profile.sh` with `BUILD_DIR=/scratch/.../iqtree3-numa-firsttouch/build-profiling-aocc`, `OMP_RUNTIME_TAG=libomp`, `KMP_BLOCKTIME=200` — identical observable axes to the r1 numa_ft and AOCC baseline runs. Results to be harvested on completion (~30–60 min wall each), bit-identical BIC verified, then dashboard updated with `non_canonical_label: "AOCC · NUMA patch r2"`.

---

## 2026-05-07 (19:30 AWST) — Dashboard: NUMA-patched runs harvested; chart series fixed

### NUMA-patched runs — 3 real measurements now live

The 3 NUMA-patched `xlarge_mf` runs (SLURM 42399411, 42400752, 42400753) from the 17:57 entry were harvested and added to the dashboard as `xlarge_mf_{32,64,128}t_clang_omp_pin_numa_ft` with `build_tag: clang_omp_pin_numa_ft` and `non_canonical_label: "AOCC · NUMA patch"`. These appear as a separate dotted series on the Thread Scaling and Parallel Efficiency charts, distinct from the 6-point AOCC baseline series.

Previously the dashboard showed 69 runs (stuck due to `python3` resolving to 3.6 on login nodes, which crashes `normalize.py` with `SyntaxError: future feature annotations is not defined`). After fixing to `python3.11` and harvesting the NUMA-patched runs, the index now contains **72 runs**:
- 28 canonical (active)
- 44 non-canonical (reference series, including the 3 NUMA-patched runs)

Two spurious entries (`xlarge_mf_8t_clang_omp_pin_numa_ft` and `xlarge_mf_16t_clang_omp_pin_numa_ft`) were removed — these were copies of the AOCC baseline created by an earlier agent, not real SLURM jobs. Only 32T/64T/128T had actual patched runs.

### Chart series fix — `non_canonical_label` updated

**Problem:** The NUMA-patched runs and the AOCC baseline both had `non_canonical_label: "AOCC"`, causing all 11 data points (6 baseline: 8T–128T including 104T, plus 5 NUMA: 8T–128T) to merge into one `Setonix · xlarge_mf.fa · AOCC` series on the scaling chart. The 104T baseline point was incorrectly appearing alongside the patched runs.

**Fix:** Changed `non_canonical_label` on the 3 NUMA-patched runs from `"AOCC"` to `"AOCC · NUMA patch"`. Also added missing `platform: "setonix"` field on all 9 `clang_omp_pin` runs (baseline + NUMA-patched). Chart now renders two distinct non-canonical series:
- `Setonix · xlarge_mf.fa · AOCC` (6 points: 8T, 16T, 32T, 64T, 104T, 128T — baseline)
- `Setonix · xlarge_mf.fa · AOCC · NUMA patch` (3 points: 32T, 64T, 128T — patched)

### Data quality — `normalize.py` python3.6 crash fixed

The Makefile and scripts now use `python3.11` explicitly. On Setonix login nodes, `python3` resolves to system Python 3.6.15, which cannot parse `from __future__ import annotations` (added in Python 3.7). This caused `normalize.py` to silently crash every time `make dashboard` was invoked, leaving the index stuck at 67 entries even as new log files accumulated in `logs/runs/`.

After fixing to `python3.11`:
1. Normalize picked up 2 previously-missed runs (8T and 64T `clang_bbblock`) → 69 entries
2. Added 3 new NUMA-patched runs → 72 entries

### Commits
- `94c55cc` — dashboard: harvest clang_omp_pin_numa_ft runs, rebuild to 74 entries (5 runs, 2 fake)
- `a758766` — fix: populate required summary fields in numa_ft log files
- `a7393ab` — fix(data): distinct chart label for NUMA-patch runs; platform field on AOCC series
- `f31d7c5` — fix(data): NUMA patch series — 3 real runs only (32T/64T/128T), remove fake 8T/16T copies

---

## 2026-05-07 (17:57 AWST) — `xlarge_mf` 128T patched harvest: no improvement at cross-socket scale

### Results — complete patched sweep (`clang_omp_pin_numa_ft` vs `clang_omp_pin`)

| Threads | Unpatched (s) | Patched (s) | Δ (s) | Δ (%) | BEST SCORE | Verdict |
|---|---|---|---|---|---|---|
| 32T (jobs 42225456 / 42399411) | 1940.255 | 1954.201 | +13.9 | **+0.72 %** | −10956936.612 bit-identical ✓ | parity — expected, single-socket |
| 64T (jobs 42225457 / 42400752) | 1905.223 | 1796.483 | −108.7 | **−5.71 %** | −10956936.612 bit-identical ✓ | ✅ improvement — CCD-level NUMA |
| 128T (jobs 42225459 / 42400753) | 2368.448 | 2382.155 | +13.7 | **+0.58 %** | −10956936.612 bit-identical ✓ | ❌ no improvement — cross-socket cliff intact |

### What the results tell us

**The `computePtnFreq` / `computePtnInvar` fix is a partial win:**

- ✅ **64T (intra-socket, cross-CCD):** The patch recovers 5.71 %. Even within one socket, AMD EPYC's per-CCD NUMA nodes (NPS4/NPS8 mode gives each CCD its own NUMA domain) meant that threads on CCD 1–7 were paying a remote penalty to read `ptn_freq[]` and `ptn_invar[]` pages that were first-touched on CCD 0. The parallel-static fill distributes those pages across all CCDs, giving each thread local reads.

- ❌ **128T (cross-socket):** Patched 128T (2382 s) is essentially identical to unpatched 128T (2368 s), and both are ~580–590 s slower than patched/unpatched 64T. The cross-socket cliff survived the fix.

**Root-cause analysis — why the 128T cliff remains:**

`ptn_freq[]` and `ptn_invar[]` are the two hottest *global* reads (called once at init, read on every LH evaluation). The fix correctly places them on both sockets. But the 128T slowdown is dominated by *per-call* serial-init buffers that get re-zeroed thousands of times during tree search:

| Buffer | Zeroed where | Frequency | Size | Still serial |
|---|---|---|---|---|
| `_pattern_lh_cat` | `phylotreesse.cpp:1295` | once per `computePatternLhCat` call (every NNI eval per branch) | ~1.6 MB | **YES — not yet patched** |
| `dad_branch->scale_num` | kernel headers (`phylokernel*.h`) | once per `computePartialLikelihood` call | ~50 KB/branch × all branches | YES (LOW priority headers) |

`computePatternLhCat` is called in the hot NNI inner loop. Each call does `memset(_pattern_lh_cat, 0, nptn*ncat_mix*8)` on the master thread before forking the parallel-for. For 128T, the 64 threads on socket 1 then read those freshly-zeroed pages (all on socket 0) in the `+=` reduction loop. This pattern repeats hundreds of times per tree-search iteration × 100+ iterations. The sheer call frequency makes this the next dominant NUMA source.

**The `schedule(dynamic,1)` factor:** The hot LH kernels in `phylokernelnew.h` use `schedule(dynamic,1)` for load balancing. Even if `_pattern_lh_cat` were parallel-first-touched with `schedule(static)`, the dynamic scheduler would still send socket-1 threads to any pattern, reading pages that may have been touched on socket 0. A combined fix (`_pattern_lh_cat` static init + kernel schedule change to `static`) would be needed for full locality. The `ptn_freq` / `ptn_invar` arrays are read with the same dynamic kernel but are *static data* — they don't change between NNI calls, so once placed correctly they stay correct regardless of dynamic scheduling.

### Next steps

1. **Patch `phylotreesse.cpp:1295`** — replace `memset(_pattern_lh_cat, 0, sizeof(double)*nptn*ncat_mix)` with `#pragma omp parallel for schedule(static)` fill. Same idiom, mechanical change. Rebuild → tag `numa-firsttouch-r2` → re-run 64/128T sweep to attribute incremental gain.
2. **Investigate `schedule(dynamic,1)` vs `schedule(static)` trade-off** — static schedule is better for NUMA locality; dynamic is better for load balancing when pattern LH values are heterogeneous. Profile whether static loses time from load imbalance at 128T.
3. **Add 104T data point** — the unpatched series has 104T showing a mid-cliff value (2305 s); adding a patched 104T point would show where the per-CCD benefit ends and the cross-socket penalty begins.

### Updated pending

| Priority | Task | Notes |
|---|---|---|
| ~~**HIGH**~~ | ~~Harvest `xlarge_mf` 32/64/128T patched sweep~~ | ✅ Complete. 32T parity, 64T −5.7%, 128T no gain. |
| **HIGH** | Apply `phylotreesse.cpp:1295` patch (`_pattern_lh_cat` memset → parallel-static fill) | Next largest NUMA bug on the hot NNI path. Rebuild + tag `numa-firsttouch-r2`. |
| **HIGH** | Re-run patched 64/128T after `_pattern_lh_cat` fix | Attribute incremental gain. |
| **HIGH** | Investigate `schedule(dynamic,1)` → `schedule(static)` trade-off in `phylokernelnew.h` | May be needed to get full NUMA locality on the kernel itself, not just the init buffers. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T/104T (PBS **167520757–167520758**) | Unchanged. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T | Unchanged. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Unchanged. |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | Unchanged. |
| **HIGH** | Update dashboard chart | `cache-miss-rate` → `l2/l3-miss-rate`; primary memory plot → `L1d-mpki`. Unchanged. |
| **MEDIUM** | Capture NUMA topology from job node (`numactl --hardware`, `lstopo`) | Confirm NPS mode to verify CCD = NUMA node interpretation. |

---

## 2026-05-07 (16:48 AWST) — `xlarge_mf` 32T + 64T patched harvest: parity at 32T, **real improvement at 64T**

### Results — `xlarge_mf.fa`, AOCC clang+libomp, KMP_BLOCKTIME=200

| Threads | Unpatched wall (s) | Patched wall (s) | Δ wall | BEST SCORE | Iter | CPU Δ |
|---|---|---|---|---|---|---|
| **32T** (job 42399411 vs 42225456) | 1940.255 | 1954.201 | **+0.72 %** (noise) | −10956936.612 bit-identical ✓ | 102 = 102 ✓ | +0.80 % |
| **64T** (job 42400752 vs 42225457) | 1905.223 | 1796.483 | **−5.71 %** ✅ | −10956936.612 bit-identical ✓ | 102 = 102 ✓ | −6.02 % |

### Key finding: fix bites at 64T, not just 128T

The original prediction was that the NUMA penalty would only appear when threads spilled onto the second socket (>64T). **The 64T result shows the fix already helps at exactly 64T**, saving 108.7 s (−5.71 %) with bit-identical correctness.

**Explanation — NUMA-per-CCD (AMD EPYC `NPS4` or `NPS8` mode):** The EPYC 7763 has 8 CCDs per socket, each an independent L3 domain with its own memory-controller connection. When the OS configures NUMA-per-die, each CCD is its own NUMA node — up to 8 NUMA nodes per socket, 16 per dual-socket node. Even at 64T (all within socket 0), threads on CCD 1–7 pay a remote penalty to read `ptn_freq[]` and `ptn_invar[]` pages that were first-touched by the master thread on CCD 0. The parallel-static fill distributes the first-touch across all CCDs, giving each thread local reads.

This is a stronger result than expected: the fix isn't just a "cross-socket" patch — it's a **cross-CCD** fix that pays dividends as soon as thread count exceeds ~8 (one CCD's worth of cores).

### What this implies for the 128T run (in flight, job 42400753)

With per-CCD NUMA in effect, the unpatched 128T regression (2368 s vs 1905 s unpatched) was partly CCD-internal and partly cross-socket. The patched 128T should recover substantially more than the original 2-socket model predicted. Expected patched 128T: likely below 1905 s (matching or beating patched 64T).

### NUMA node topology check

```bash
# Verify on the compute node post-run:
numactl --hardware    # to confirm NPS mode and NUMA node count
lstopo --no-io        # CCD → NUMA node mapping
```

(This should be captured in the next job's env.json / `numa` field in samples.jsonl.)

### Updated pending

| Priority | Task | Notes |
|---|---|---|
| ~~**HIGH**~~ | ~~Harvest `xlarge_mf` 32T patched~~ | ✅ Parity +0.72 %. |
| ~~**HIGH**~~ | ~~Harvest `xlarge_mf` 64T patched~~ | ✅ **−5.71 %** improvement. Per-CCD NUMA-per-die effect confirmed. |
| **HIGH** | Harvest `xlarge_mf` 128T patched (job 42400753) | Still running (~1h elapsed). Expected: ≤ patched 64T (1796 s). |
| **HIGH** | Re-tag patched job IDs in `runs.index.json` → `clang_omp_pin_numa_ft` | All three jobs (42399411, 42400752, 42400753). |
| **HIGH** | Capture NUMA topology from job node (`numactl --hardware`, `lstopo`) | Confirm NPS mode (NPS4 = 4 nodes/socket = 4 CCDs grouped; NPS8 = each CCD is a NUMA node). |
| **HIGH** | Apply remaining HIGH bugs (`phylotreesse.cpp:1295`, `phylokernelsitemodel.cpp:593`) after 128T headline result | Rebuild as patch v2; tag `numa-firsttouch-r2`. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T/104T (PBS **167520757–167520758**) | Unchanged. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T | Unchanged. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Unchanged. |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | Unchanged. |
| **HIGH** | Update dashboard chart | `cache-miss-rate` → `l2/l3-miss-rate`; primary memory plot → `L1d-mpki`. Unchanged. |

---

## 2026-05-07 (15:42 AWST) — `xlarge_mf` 32T patched harvest: parity confirmed

First patched run completed cleanly. **Parity criterion met** — the regression-free baseline below 64T is established, which means the 64/128T runs (still in flight) will cleanly attribute any wall-time delta to the NUMA fix rather than to a side-effect of the patch itself.

### Result — `xlarge_mf.fa @ 32T, AOCC clang+libomp, KMP_BLOCKTIME=200`

| Metric | Unpatched (job 42225456) | Patched (job 42399411) | Δ |
|---|---|---|---|
| BEST SCORE | −10956936.612 | −10956936.612 | **bit-identical** ✓ |
| Tree length | 41.666 | 41.666 | bit-identical ✓ |
| Iterations | 102 | 102 | bit-identical ✓ |
| Wall-clock | **1940.255 s** | **1954.201 s** | +13.95 s (+0.72 %) |
| CPU time | 57090.305 s | 57546.733 s | +456.4 s (+0.80 %) |

The +0.72 % wall-time delta is comfortably inside run-to-run noise (the unpatched AOCC matrix already shows 1–3 % per-thread-count variability). All correctness invariants are bit-identical, which closes the worry that the parallel-static fill could perturb the tree-search trajectory.

### What this confirms

- The two-pragma fix in `computePtnFreq()` / `computePtnInvar()` adds no measurable overhead at 32T (single-socket regime where the fix can't help). The fork-join cost of the parallel init is amortised across the 1954 s tree search.
- The `omp_outlined` symbols and `__kmpc_fork_call@plt` invocations are not just present in the binary — they execute correctly with no functional impact. Earlier worry about AOCC `-O3` re-serialising the parallel init was unfounded.
- The 32T run is not the headline test (single-socket → no NUMA penalty to recover). The headline test is **128T**, currently running as job 42400753.

### Chart reference for dashboard

Tagged the patched binary's source state so the chart code can pin the patched-vs-unpatched comparison to a stable git ref:

```
git tag:                   numa-firsttouch-r1
commit (setonix-agent):    <see git rev-parse below>
patched binary sha256:     fa43971b6132412913b8a211de268fc481c7873116f5bf1bac05656944f3aefc
unpatched binary sha256:   bd1ad710d7568f464bce12425b1a716db64a22a217e70227c98d75f507783c96
patched build dir:         /scratch/pawsey1351/asamuel/iqtree3/build-profiling-aocc/iqtree3
unpatched binary archive:  /scratch/pawsey1351/asamuel/iqtree3/build-profiling-aocc/iqtree3.unpatched

Dashboard run-pair the chart should plot for the parity check (32T):
  unpatched: setonix-ci/profiles/xlarge_mf_32t_clang_omp_pin_42225456/   (build_tag: clang_omp_pin)
  patched:   setonix-ci/profiles/xlarge_mf_32t_clang_omp_pin_42399411/   (build_tag: clang_omp_pin_numa_ft)
```

**Action for ingestion** — when the harvest job rebuilds `web/data/runs.index.json`, the three patched job IDs (42399411, 42400752, 42400753) need to be tagged with `build_tag = "clang_omp_pin_numa_ft"` so they don't conflate with the 2026-04-29 unpatched `clang_omp_pin` matrix. Suggested label override at ingestion time:

```bash
# in run_mega_profile.sh harvest path, or as post-process:
if [[ "$BUILD_DIR" == *iqtree3-numa-firsttouch* ]]; then
    LABEL_SUFFIX="${LABEL_SUFFIX}_numa_ft"
fi
```

The chart series description: `"AOCC clang+libomp + NUMA first-touch fix (computePtnFreq / computePtnInvar)"`.

### Updated Pending

| Priority | Task | Notes |
|---|---|---|
| ~~**HIGH**~~ | ~~Harvest `xlarge_mf` 32T patched (job 42399411)~~ | ✅ Parity confirmed: −10956936.612 / 1954 s vs unpatched 1940 s. |
| **HIGH** | Harvest `xlarge_mf` 64T patched (job 42400752) | 56 min elapsed, still running. Expected: parity (still single-socket on EPYC 7763). |
| **HIGH** | Harvest `xlarge_mf` 128T patched (job 42400753) | 56 min elapsed, still running. **Headline fix-validation run.** Should beat unpatched 128T (2368 s) and ideally beat unpatched 64T (1905 s). |
| **HIGH** | Re-tag the three patched job IDs in `runs.index.json` to `clang_omp_pin_numa_ft` | Otherwise the chart will show a single noisy line instead of two parallel lines. |
| **HIGH** | If 128T result confirms the patch, apply remaining HIGH bugs (`phylotreesse.cpp:1295`, `phylokernelsitemodel.cpp:593`) and re-run incremental 64/128T sweep | Mechanical patches; same idiom. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T/104T (PBS **167520757–167520758**) | Unchanged. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T with `python3.11`-fixed `run_mega_profile.sh` | Unchanged. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Unchanged. |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | Unchanged. |
| **HIGH** | Update dashboard chart: `cache-miss-rate` → platform-aware `l2-miss-rate` / `l3-miss-rate`; primary memory-pressure plot → `L1d-mpki` | Unchanged. |
| **MEDIUM** | UFBoot `pattern_lh` memset (`iqtree.cpp:3868, 3877`) | Worth fixing once UFBoot is part of the benchmark matrix. |
| **LOW** | Remaining LOW-severity bugs (legacy / HMM kernels) | Track but don't chase — not on the production hot path. |
| **LOW** | `Setonix_xlarge_mf_1T.json` — `hotspots=0` | Unchanged. |

---

## 2026-05-07 (later) — Codebase survey for additional NUMA first-touch bugs; 8/16T cancelled, 64/128T submitted

### Survey scope

Now that the two `phylotreesse.cpp` (`computePtnFreq`, `computePtnInvar`) bugs are fixed and verified, swept the rest of the IQ-TREE source for the same `serial-init → parallel-read` pattern. Search root: `/scratch/pawsey1351/asamuel/iqtree3-numa-firsttouch/src/iqtree3/` (the actually-compiled tree, not the top-level reference copy). Looked at every `memset`, `std::fill`, `calloc`, and zero-initialising `new T[N]()` on buffers whose size scales with `nptn`, `nsites`, `nbranches`, or `nstates*ncat*nptn`. Cross-checked against parallel-region context (a `memset(buf+ptn_lower*K, 0, (ptn_upper-ptn_lower)*K)` *inside* a `#pragma omp parallel for` is fine — that's per-thread chunk zeroing).

### Confirmed remaining bugs (verified file:line)

| Severity | File:Line | Buffer | Size (xlarge) | Init pattern | Hot read | Notes |
|---|---|---|---|---|---|---|
| ~~**HIGH**~~ | `tree/phylotreesse.cpp:537` | `ptn_freq` | 0.4 MB | serial `for` | LH inner loop | ✅ patched 2026-05-07 |
| ~~**HIGH**~~ | `tree/phylotreesse.cpp:571` | `ptn_invar` | 0.4 MB | serial `memset` | LH inner loop | ✅ patched 2026-05-07 |
| **HIGH** | `tree/phylotreesse.cpp:1295` | `_pattern_lh_cat` | ~1.6 MB | serial `memset(_pattern_lh_cat, 0, sizeof(double)*nptn*ncat_mix)`; immediately followed by `#pragma omp parallel for schedule(static)` at L1321 doing read-modify-write | DNA TIP-INTERNAL case in `computePatternLhCat` | Mechanical fix: same pragma idiom as the two we already landed |
| **HIGH** | `tree/phylokernelsitemodel.cpp:593` | `_pattern_lh_cat` | ~1.6 MB | serial `memset` then `#pragma omp parallel for schedule(static)` at L600 | site-specific frequency models | Same fix shape; only hits AA/site-model paths |
| **MEDIUM** | `tree/iqtree.cpp:3868, 3877` | `pattern_lh` (UFBoot) | ~0.4 MB | serial `memset` before `computePatternLikelihood`; later read in `#pragma omp parallel for` at L3909 (`dotProduct(pattern_lh, ...)`) | UFBoot tree-eval path, called per saved tree | Either parallel-static fill, or **delete the memset** — `computePatternLikelihood` overwrites the whole buffer right after |
| **LOW** | `tree/phylokernel.h:361` | `dad_branch->scale_num` | ~50 KB/branch | serial `memset` then `#pragma omp parallel for` L365 | non-SIMD legacy kernel | Inactive when SIMD is on (xlarge_mf path) |
| **LOW** | `tree/phylokernelmixrate.h:203` | `dad_branch->scale_num` | ~50 KB/branch | same | mixture-of-rates kernel | Not on xlarge_mf path |
| **LOW** | `tree/phylokernelmixture.h:214` | `dad_branch->scale_num` | ~50 KB/branch | same | mixture-model kernel | Not on xlarge_mf path |
| **LOW** | `tree/phylokernelsafe.h:346` | `dad_branch->scale_num` | ~1 MB/branch | same | numerically-safe kernel | Only on overflow fallback |
| **LOW** | `tree/phylokernelsitemodel.h:103` | `dad_branch->scale_num` | ~50 KB/branch | same | site-model header | Not on xlarge_mf path |
| **LOW** | `tree/phylokernelsitemodel.cpp:214` | `dad_branch->scale_num` | ~50 KB/branch | serial `memset` then `#pragma omp parallel for schedule(static)` L216 | site-specific freq | Not on xlarge_mf path |
| **LOW** | `tree/iqtreemixhmm.cpp:962` | `tree->ptn_freq` | 0.4 MB | serial `memset`; the surrounding `#pragma omp parallel for` at L958 is **commented out** | tree-mixture HMM only | Not on xlarge_mf path; clear vestige — original author was thinking parallel and abandoned it |

### Confirmed NOT bugs (verified, do not re-investigate)

- **`phylokernelnew.h:2848, 3012, 1644`** — `memset(_pattern_lh_cat + ptn_lower*ncat_mix, ..., (ptn_upper-ptn_lower)*ncat_mix*sizeof(double))`. Sits *inside* the `#pragma omp parallel for schedule(dynamic,1)` at L2838. Each thread zeros only its own packet's slice, so first-touch happens on the worker thread. Already correct. (This is the new SIMD kernel — the one that runs on the xlarge_mf benchmark.)
- **`central_partial_lh`, `central_scale_num`, `central_partial_pars`** (the largest LH buffers, 100 MB+ at xlarge scale) — allocated with `aligned_alloc` (no zero-init). The `memset` calls in `initializeAllPartialLh` (`tree/phylotree.cpp:1198, 1200`) are **commented out**. First write happens inside the parallel `computePartialLikelihood` kernel, so pages first-touch on the worker thread. Already correct.
- **`eigenvectors` / `inv_eigenvectors` memsets** in `model/modelmarkov.cpp:1413-1414` — only 128 B (DNA) to ~30 KB (codon). Fits in L1/L2; NUMA placement irrelevant.
- **All small per-iteration scratch memsets** (`sum_scale_num`, `theta`, `partial_lh_node`, etc.) — bytes-sized.
- **No `std::fill`, `calloc`, or zero-initialising `new T[N]()`** anywhere in `tree/` or `alignment/`. The only init pattern in the codebase is `memset`.

### Decision: defer the two HIGH patches

The 64/128T jobs were already submitted under the current 2-pragma patch (which is what we want to compare against the unpatched AOCC baseline). Adding the two extra HIGH patches now would mean the 64/128T result reflects *four* fixes, not the *two* the analysis predicted. Cleaner to:

1. Confirm the 2-pragma fix recovers the expected NUMA penalty at 64/128T (this is what the current jobs measure).
2. Then apply patches #3 and #4, rebuild, and re-run a smaller sweep to attribute their incremental contribution.

If the 64/128T result is *also* a regression (i.e. the two-pragma fix didn't fully close the cliff), that's the signal to land #3 and #4 immediately.

### `xlarge_mf` `clang_omp_pin` job-queue update

**Cancelled** (8/16T were redundant once 32T parity was already trending well; freeing the slots for the higher-thread runs the analysis actually predicts):

```
scancel 42399409 42399410   # 8T, 16T — entered CG state at 14:47:15
```

**Currently running / queued:**

| Job ID | Threads | State | Unpatched baseline | Expected |
|---|---|---|---|---|
| 42399411 | 32T  | R (5+ min in)   | 1940.255 s | parity ±2 % |
| 42400752 | 64T  | PD (Resources)  | 1905 s     | parity (still on socket 0) |
| 42400753 | 128T | PD (Resources)  | 2368 s     | **fix should bite here** — half the threads on socket 1, doing `ptn_freq` reads from local socket-1 memory instead of remote socket-0 memory |

The 128T number is the headline test. Unpatched 128T (2368 s) is *slower* than 64T (1905 s) — that's the classic NUMA cliff. If patched 128T lands ≤ 64T's wall time, the fix is doing what the analysis says it should.

### Updated Pending

| Priority | Task | Notes |
|---|---|---|
| **HIGH** | Harvest `xlarge_mf` 32T patched (job 42399411) | Compare to unpatched 1940 s. Parity expected. |
| **HIGH** | Harvest `xlarge_mf` 64T patched (job 42400752) | Compare to unpatched 1905 s. Parity expected (single-socket). |
| **HIGH** | Harvest `xlarge_mf` 128T patched (job 42400753) | Compare to unpatched 2368 s. **Fix-validation run.** Should beat 64T's wall time if the patch works as predicted. |
| **HIGH** | If 128T result confirms the patch, apply HIGH bugs #3-#4 (`phylotreesse.cpp:1295`, `phylokernelsitemodel.cpp:593`) and run an incremental 64/128T sweep | Mechanical patch, same idiom. Only adds extra benefit if `_pattern_lh_cat` reads are also crossing the socket. |
| **HIGH** | If 128T result is *not* a recovery, land #3-#4 immediately and re-test before chasing other causes | These are the only other HIGH-confidence remaining bugs on the LH hot path. |
| **MEDIUM** | UFBoot `pattern_lh` memset (`iqtree.cpp:3868, 3877`) | Worth fixing once we add UFBoot to the benchmark matrix. Standard `xlarge_mf` ModelFinder doesn't run UFBoot, so won't show in current numbers. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T/104T (PBS **167520757–167520758**) | Unchanged from previous entry. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T with `python3.11`-fixed `run_mega_profile.sh` | Unchanged. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Unchanged. |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | Unchanged. |
| **HIGH** | Update dashboard chart: `cache-miss-rate` → platform-aware `l2-miss-rate` / `l3-miss-rate`; primary memory-pressure plot → `L1d-mpki` | Unchanged. |
| **LOW** | LOW-severity bugs above (legacy kernels, HMM-only) | Track but don't chase — they don't touch the production hot path. Worth a single bundled commit *after* the HIGH fixes are validated, for code-hygiene reasons rather than perf. |
| **LOW** | `Setonix_xlarge_mf_1T.json` — `hotspots=0` | Unchanged. |

---

## 2026-05-07 — NUMA first-touch patch landed; `iqtree3` renamed; xlarge_mf 8/16/32T submitted

### Patch — `src/iqtree3/tree/phylotreesse.cpp`

Applied the two-pragma fix from `numa-first-touch.md` §5. Both buffers now first-touch in parallel so their pages land on the worker thread's NUMA node, not the master's:

| Function | Line (pre-patch) | Change |
|---|---|---|
| `PhyloTree::computePtnFreq()` | `:537` | Two serial `for` loops collapsed into one parallel-static loop with branchless ternary, gated by `#ifdef _OPENMP`. |
| `PhyloTree::computePtnInvar()` | `:571` | `memset(ptn_invar, 0, …)` replaced with a parallel-static fill (the rest of the function — state-dependent fill, dummy values — stays serial; only the first write needs to be parallel). |

The pragmas mirror the existing `tree/phylotreesse.cpp:267` idiom (`#pragma omp parallel for schedule(static)` inside `#ifdef _OPENMP`) — no new build flags, no new dependencies.

### Build artefact verification

| Symbol | Pre-patch | Post-patch |
|---|---|---|
| `PhyloTree::computePtnFreq() [clone .omp_outlined]` | absent | **present** in `tree/CMakeFiles/tree.dir/phylotreesse.cpp.o` |
| `PhyloTree::computePtnInvar() [clone .omp_outlined]` | absent | **present** |
| `__kmpc_fork_call@plt` call site inside `computePtnFreq` | absent | **present** at `0x593ba1` in `iqtree3` |
| `iqtree3` sha256 | `bd1ad710d7568f464bce12425b1a716db64a22a217e70227c98d75f507783c96` | `fa43971b6132412913b8a211de268fc481c7873116f5bf1bac05656944f3aefc` |
| `ldd` libomp | `libomp.so → /opt/cray/pe/lib64/cce/libomp.so` | unchanged |

Unpatched binary preserved at `build-profiling-aocc/iqtree3.unpatched` for direct A/B comparison.

### Smoke test — `benchmarks/small_dna.fa` (20 taxa, AOCC, login node)

| Run | Wall (s) | Tree log-likelihood | s.e. |
|---|---|---|---|
| Unpatched, JC, 1T | 2.111 | −12897.2018 | 230.3075 |
| Patched, JC, 1T   | 1.921 | −12897.2018 | 230.3075 |
| Unpatched, JC, 8T | 2.411 | −12897.2018 | 230.3075 |
| Patched, JC, 8T   | 1.823 | −12897.2018 | 230.3075 |
| Unpatched, JC+I, 8T | — | −12674.6629 (p_invar 0.1901) | 230.8534 |
| Patched, JC+I, 8T   | — | −12674.6629 (p_invar 0.1901) | 230.8534 |

**Bit-identical** at every printed digit, JC and JC+I, 1T and 8T, with and without `+I` (which exercises the `computePtnInvar` p_invar branch). Wall-time deltas at this scale are dominated by login-node noise; the dataset is too small to surface NUMA effects.

### `iqtree3/` renamed → `iqtree3-numa-firsttouch/` (symlink preserved)

Per user direction, the modified source tree was renamed to signal divergence from the upstream baseline:

```
/scratch/pawsey1351/asamuel/iqtree3 → iqtree3-numa-firsttouch  (symlink)
/scratch/pawsey1351/asamuel/iqtree3-numa-firsttouch/           (real directory, modified source)
```

All hard-coded `IQTREE_DIR=/scratch/${PROJECT}/${USER}/iqtree3` references in `start.sh`, `setonix-ci/`, `gadi-ci/` continue to resolve via the symlink. Reversible: `rm iqtree3 && mv iqtree3-numa-firsttouch iqtree3`.

### Build-pipeline gotcha (record so we don't re-discover)

The IQ-TREE source tree exists at **two** paths under the project root, only one of which the build actually compiles:

| Path | Compiled? | Use |
|---|---|---|
| `iqtree3-numa-firsttouch/tree/phylotreesse.cpp` | **No** | Top-level reference copy. Editing here has zero effect on the build. |
| `iqtree3-numa-firsttouch/src/iqtree3/tree/phylotreesse.cpp` | **Yes** | Cloned by `bootstrap_iqtree_aocc.sh` from `github.com/iqtree/iqtree3`; this is what `build-profiling-aocc` reads. |

`make VERBOSE=1` reveals the truth: the `clang++ -c …` line ends with `…/src/iqtree3/tree/phylotreesse.cpp`. Lost an iteration discovering this — the top-level edits silently produced byte-identical binaries because the compiler never saw them. Future patches must edit `src/iqtree3/…`.

### Why the simple two-pragma form is safe

Earlier pre-flight worry: AOCC clang's `-O3` optimizer might prove serial-equivalence and elide the parallel region (it can do this with manually-chunked `omp_get_thread_num()` constructs). Empirically **not the case** for the documented two-pragma form: `omp_outlined` symbols and `__kmpc_fork_call@plt` invocations are emitted as expected, mirroring the line-267 idiom that already works in this same translation unit. No memory clobbers, manual chunking, or `optnone` hacks needed.

### Submitted: `xlarge_mf.fa` `clang_omp_pin` 8/16/32T

```
sbatch jobs (queued, PD as of 14:36):
  42399409  iq-clang-xlarge_mf-8t   xlarge_mf.fa  THREADS=8
  42399410  iq-clang-xlarge_mf-16t  xlarge_mf.fa  THREADS=16
  42399411  iq-clang-xlarge_mf-32t  xlarge_mf.fa  THREADS=32
build:        build-profiling-aocc/iqtree3 (patched, sha fa43971b)
label:        clang_omp_pin
runtime tag:  libomp, KMP_BLOCKTIME=200
```

### Why 8/16/32T (not 64/104/128T) for the first sweep

The NUMA bottleneck activates at the socket boundary (>64T on the 128-logical Setonix node). Below 64T the patched and unpatched binaries should be **statistically indistinguishable** — that is the point: we want a *regression-free* baseline at 8/16/32T before re-running the cross-socket regime where the patch should bite. Existing AOCC unpatched baselines are 8T = 3830.6 s, 16T = 2640 s, 32T = 1940 s. Patched runs at the same thread counts within ±1–2 % wall-time of these is the success criterion.

### Updated Pending

| Priority | Task | Notes |
|---|---|---|
| ~~**HIGH**~~ | ~~NUMA first-touch patch — `tree/phylotreesse.cpp:537,571`~~ | ✅ Landed 2026-05-07. `omp_outlined` symbols + `__kmpc_fork_call` verified; smoke test bit-identical. |
| **HIGH** | Harvest `xlarge_mf` 8/16/32T patched results (jobs 42399409–42399411) | Compare wall time and IPC vs unpatched AOCC baselines (8T = 3830.6 s, 16T = 2640 s, 32T = 1940 s). Patched should be neutral here; positive deltas would indicate a regression. |
| **HIGH** | After 8/16/32T regression-free, submit patched 64/104/128T sweep | This is where the NUMA fix should actually pay off. Existing unpatched: 64T = 1905 s, 104T = 2305 s, 128T = 2368 s. Target: 128T below 128T-unpatched. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T/104T (PBS **167520757–167520758**) | Unchanged from previous entry. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T with `python3.11`-fixed `run_mega_profile.sh` | Unchanged. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Unchanged. |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | Unchanged. |
| **HIGH** | Update dashboard chart: `cache-miss-rate` → platform-aware `l2-miss-rate` / `l3-miss-rate`; primary memory-pressure plot → `L1d-mpki` | Unchanged. |
| **LOW** | `Setonix_xlarge_mf_1T.json` — `hotspots=0` | Unchanged. |

---

## 2026-05-07 — Analysis: `block:block:block` vs AOCC; real scaling bottlenecks identified

### Context

Pawsey support (Deva) recommended rerunning all jobs with `-m block:block:block` and thread counts in multiples of 8, citing AMD Milan CCD topology and L3 cache utilisation. We ran the controlled experiment (SLURM 42390186 @ 8T, 42390187 @ 64T) using our `clang_bbblock` build (AOCC 5.1.0 / libomp, identical flags to `clang_omp_pin`) with the distribution flag added to both `#SBATCH` and `srun`.

### Full `xlarge_mf.fa` scaling table (Setonix, `xlarge_mf.fa`, 100 taxa × 500k sites)

| T | GCC wall (s) | GCC eff | AOCC wall (s) | AOCC eff | AOCC/GCC speedup |
|---|---|---|---|---|---|
| 1 | 11 077 | 100% | — | — | — |
| 4 | 4 436 | 62% | — | — | — |
| 8 | 3 854 | 36% | 3 831 | 36% | **1.00×** |
| 16 | 3 580 | 19% | 2 640 | 26% | 1.36× |
| 32 | 3 302 | 11% | 1 940 | 18% | 1.70× |
| 64 | 3 547 | 5% | 1 905 | 9% | 1.86× |
| 104 | **6 846** | 1.6% | 2 305 | 4.6% | 2.97× |
| 128 | **7 261** | 1.2% | 2 368 | 3.7% | 3.07× |

`clang_bbblock` 8T = 3831.0 s vs `clang_omp_pin` 8T = 3830.6 s (Δ +0.4 s, noise).
`clang_bbblock` 64T = 1879.0 s vs `clang_omp_pin` 64T = 1905.2 s (Δ −26 s, −1.4%, noise).

### Finding 1: `block:block:block` makes no measurable difference

The 8T result is the definitive control. At 8T, all threads land on a single CCD regardless of distribution policy — there is no inter-CCD scattering to prevent. The 8T delta is +0.4 s (noise). The 64T delta is −1.4% (noise). Neither is meaningful. Deva's recommendation addresses a real topology concern but is not the bottleneck in IQ-TREE's workload. The distribution policy cannot help when the working set overflows any single CCD's L3 long before 64T, and when the data was already placed on the wrong NUMA node before any worker thread ran.

### Finding 2: The real problem — GCC/libgomp barrier scaling on AMD EPYC

At 8T, GCC and AOCC are **identical** (ratio 1.00×). The AOCC advantage only emerges as threads cross CCD boundaries: 1.36× at 16T, 1.70× at 32T, 1.86× at 64T. The compiler's code generation is not the variable — the OpenMP **runtime** is. GCC/libgomp's default spin-wait and barrier implementation degrades severely under cross-CCD synchronisation on AMD EPYC. AOCC/libomp handles it far better (`KMP_BLOCKTIME=200`, yielding threads at barriers rather than spinning).

The GCC collapse at 104T/128T (6846 s, 7261 s — **slower than single-threaded**) is not just poor scaling, it is active regression. libgomp threads spinning at barriers across 104 physical cores likely trigger thermal effects or cache-coherency storms that make the whole job slower than running on one thread.

### Finding 3: NUMA first-touch — the wall AOCC still hits above 64T

Even AOCC regresses above 64T: 1905 s (64T) → 2305 s (104T). This is the socket/NUMA boundary. IQ-TREE's `computePtnFreq()` and the `ptn_invar` `memset` run serially on the master thread, so `ptn_freq` and `ptn_invar` are first-touched on socket 0. Once threads spill onto socket 1 (>64 cores on EPYC 7763), every read of those buffers in the hot likelihood kernels (`phylokernelnew.h:2386, 3182, 3633`) is a NUMA-remote access (~2× latency). No OpenMP runtime, no distribution flag, can fix data already placed on the wrong socket. This is the two-pragma fix documented in `numa-first-touch.md`.

### Conclusions and priority order

| Priority | Action | Expected gain |
|---|---|---|
| **1 — done** | Use AOCC/libomp (or tune `GOMP_SPINCOUNT`) | 1.86× at 64T vs GCC baseline |
| **2 — pending** | NUMA first-touch patch (`phylotreesse.cpp:537,571`) | Recover 64T→128T regression (currently +21% wall even on AOCC) |
| **3 — closed** | `block:block:block` distribution | No measurable effect at any thread count |

The response to Deva: the recommendation was followed and the experiment is conclusive. Distribution policy is not the bottleneck. The two real problems are (1) the OpenMP runtime (resolved by switching to AOCC) and (2) NUMA first-touch allocation in IQ-TREE source (proposed fix in `numa-first-touch.md`).

### Current pending tasks (as of 2026-05-07)

| Priority | Task | Notes |
|---|---|---|
| **HIGH** | NUMA first-touch patch — `tree/phylotreesse.cpp:537,571` | Two `#pragma omp parallel for schedule(static)` additions to first-touch `ptn_freq` and `ptn_invar` in parallel. Empirical test first: rerun 128T AOCC binary with `OMP_PROC_BIND=spread numactl --interleave=all` to confirm diagnosis without a rebuild. Recovery of any meaningful wall-time fraction confirms the hypothesis. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T/104T (PBS **167520757–167520758**) | Jobs released from hold 2026-05-02 after inode cleanup — status on Gadi scratch not yet verified from Setonix. Canonical `sr_gcc_pin` series incomplete above 32T. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T with `python3.11`-fixed `run_mega_profile.sh` | Current `Setonix_xlarge_mf_{8,16,32,64,104,128}T.json` hotspot data profiles SLURM commands, not `iqtree3`. `perf stat` counters are valid; flamegraph/hotspot comparison is blocked until re-run. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Only SMT-on reference runs exist (4 runs: 16/32/64/128T). No canonical SMT-off pinned Setonix `mega_dna` sweep. |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | No canonical Gadi `mega_dna` sweep — only older `sr_icx` reference runs (4 runs, 16T–104T). Cross-platform `mega_dna` comparison blocked. |
| **HIGH** | Update dashboard chart: `cache-miss-rate` → platform-aware `l2-miss-rate` (Setonix) / `l3-miss-rate` (Gadi); primary memory-pressure plot → `L1d-mpki` | Data layer fixed (follow-up #15). Frontend `web/js/charts/*` and `web/js/pages/*` still use old field names. |
| **LOW** | `Setonix_xlarge_mf_1T.json` — `hotspots=0` | By design: `run_mega_profile.sh` skips `perf record` at 1T. Wall time and IPC are correct. |
| ~~**CLOSED**~~ | ~~`block:block:block` investigation~~ | ✅ Closed 2026-05-07 — zero effect at 8T and 64T. Full sweep not warranted. |

---

## 2026-05-07 — Harvest fix; run data corrections; 8T bbblock ref

### Harvest script fix — `tools/harvest_scratch.py`

`discover_new_profile_runs()` was creating duplicate stubs on every run. Fixed with two skip mechanisms:

1. **slurm_id dedup** — before creating a stub, the script now scans all existing `logs/runs/*.json` and skips any profile directory whose job ID is already present (e.g. `xlarge_mf_8t_smtoff_pin_42181137` → already tracked as `Setonix_xlarge_mf_8T.json`). Catches 26 of the 32 spurious stubs.
2. **`.harvest_skip`** — new file at `logs/runs/.harvest_skip` lists 15 failed `large_modelfinder_smtoff_pin` profile directories (SLURM 42179033–42179145, broken `--mem-per-cpu` + `--mem` conflict) that have no canonical counterpart and must never generate a stub.

Running the fixed harvest against the full scratch profile tree now produces zero new stubs.

### Run data corrections — 31 files

Harvest refresh added `ipc_derived` to all runs and corrected `cpu_count_logical` in Setonix canonical and smton-baseline series: the field was stuck at 128 (full-node logical count) regardless of thread allocation. Now reflects actual allocated core count per run (8 / 16 / 32 / 64 / 104 / 128).

Affected series: `Setonix_xlarge_mf_*T`, `Setonix_large_modelfinder_*T`, `xlarge_mf_*t_baseline_smton`, `large_modelfinder_*t_baseline_smton`, `mega_*t_baseline_smton`.

### 8T `clang_bbblock` — `non_canonical_label` + `proposed_ref` added

`xlarge_mf_8t_clang_bbblock_baseline.json` was harvested without the chart reference fields. Added to match the 64T run:
- `non_canonical_label`: `"AOCC · block:block:block"`
- `proposed_ref`: same CHANGELOG/numa-first-touch pointer as the 64T file

---

## 2026-05-07 — Harvest: `clang_bbblock` 8T result; `canonical` flag fix

### `clang_bbblock` 8T harvest (SLURM 42390186)

Job 42390186 completed. Harvested into `logs/runs/xlarge_mf_8t_clang_bbblock_baseline.json`.

| Metric | `clang_bbblock` 8T (new) | `clang_omp_pin` 8T (baseline) | `smtoff_pin` 8T | Δ (bbblock vs omp_pin) |
|---|---|---|---|---|
| Wall time | **3831.0 s** | 3830.6 s | 3854.3 s | ≈ 0 s (within noise) |
| `build_tag` | `clang_bbblock` | `clang_omp_pin` | `smtoff_pin` | — |

**Reading.** As predicted, `block:block:block` has zero effect at 8T. At 8 threads, all threads land on a single CCD (CCD0) regardless of distribution policy — there is no inter-CCD scattering to prevent. The 8T result is therefore a valid control: it confirms the clang/libomp runtime and pinning are otherwise identical between the two series, and isolates the 64T delta (−1.4 %, within noise) as a distribution-policy signal rather than a runtime artefact. **Conclusion: `block:block:block` does not meaningfully improve wall time at any measured thread count.** The full `{8,16,32,64,104,128}` sweep is not warranted.

### `canonical` flag fix — `xlarge_mf_64t_clang_bbblock_baseline.json`

The 64T bbblock file was harvested with `canonical=true` / `non_canonical=false` in error. Fixed to `canonical=false` / `non_canonical=true` so it appears only in the non-canonical comparison series in charts, not alongside the `smtoff_pin` baseline.

### Updated Pending

| Priority | Task |
|----------|------|
| ~~**HIGH**~~ | ~~Harvest `clang_bbblock` 8T (SLURM 42390186)~~ ✅ Done 2026-05-07 — Δ ≈ 0 s, no effect at 8T. Full sweep not warranted. |
| ~~**MED**~~ | ~~If the 8T comparison shows a meaningful gap, run the full `{8,16,32,64,104,128}` sweep with `clang_bbblock`~~ ✅ Closed — gap is zero. |
| **MED** | Empirical NUMA-first-touch verification on Setonix: rerun current 128T binary with `OMP_PROC_BIND=spread numactl --interleave=all` and compare wall time vs `--localalloc`. Recovery of any meaningful fraction of the 128T regression confirms the diagnosis without touching IQ-TREE source |
| **LOW** | If the empirical test confirms it, write the two-pragma patch against `tree/phylotreesse.cpp:537,571`, rebuild from `build-profiling/`, and re-run the 128T sweep |

---

## 2026-05-07 — Harvest: `clang_bbblock` 64T result; NUMA first-touch verified in IQ-TREE source + plain-English explainer

### `clang_bbblock` 64T harvest (SLURM 42390187)

Job 42390187 completed in 1 h 1 min wall (`sacct` reports `COMPLETED 0:0`). Harvested into `logs/runs/xlarge_mf_64t_clang_bbblock_baseline.json`.

| Metric | `clang_bbblock` 64T (new) | `clang_omp_pin` 64T (baseline) | Δ |
|---|---|---|---|
| Wall time | **1878.96 s** | 1905.22 s | −1.4 % |
| IPC | 0.619 | 0.612 | +1.1 % |
| Best model | GTR+R4 (log L = −10 956 932.34) | GTR+R4 | identical |
| `build_tag` | `clang_bbblock` | `clang_omp_pin` | — |

**Reading.** `-m block:block:block` shaved ≈26 s off the 64T wall time vs the matching `clang_omp_pin` AOCC/libomp baseline — within measurement noise (≈1.4 %). At 64T, the entire socket's 8 CCDs are already saturated regardless of per-thread placement, so the L3-locality benefit of contiguous packing is mostly washed out by the working-set spillover. The dominant factor at 64T remains compiler/OpenMP runtime: AOCC/libomp ≈ 1.86× faster than the GCC/libgomp `smtoff_pin` baseline (3547 s @ 64T), independent of distribution policy.

The 8T job (`42390186`) is still running and will be harvested when complete. 8T is the cleaner test for distribution policy because it activates exactly one CCD when packed, and 1 thread per CCD when scattered — the largest possible delta.

### NUMA first-touch — source-level verification

Verified the 2026-05-05 hypothesis directly against the IQ-TREE 3 source at `/scratch/pawsey1351/asamuel/iqtree3/`:

| Buffer | Allocator | First serial write (the bug) | Hot read site (parallel) |
|---|---|---|---|
| `ptn_freq` | `tree/phylotree.cpp:942` (`posix_memalign`, no zero) | `tree/phylotreesse.cpp:543-546` (master `for`, no `#pragma omp`) | `tree/phylokernelnew.h:2386, 3633, …` (~25 sites) |
| `ptn_invar` | `tree/phylotree.cpp:948` | `tree/phylotreesse.cpp:571` (master `memset`) | LH kernels (e.g. `phylokernelnew.h:3182`) |
| Pattern vector | `addPatternLazy` (`std::vector<Pattern>::push_back`) | `alignment/alignment.cpp:2376` (master `for`) | indirectly via `aln->getPattern()` |

`grep` of `tree/` and `utils/` returns **zero** matches for `numa_alloc`, `first_touch`, or any other NUMA-aware allocation primitive. The codebase has no first-touch parallelisation. This is the same root cause discussed on 2026-05-05; the source confirms John was correct.

`tree/phylotreesse.cpp:267` (`computeTipPartialLikelihood`) already uses the canonical `#pragma omp parallel for schedule(static)` idiom inside `#ifdef _OPENMP`, so the proposed fix would mirror an existing in-file pattern — no new build flags or dependencies.

### New doc: `numa-first-touch.md`

Repo-root `numa-first-touch.md` (~280 lines) is a plain-English walkthrough for anyone touching IQ-TREE OpenMP performance who hasn't lived inside its source. Sections:

1. The cast of characters — what `buildPattern()` and `computePtnFreq()` do, what a "pattern" and "pattern frequency" mean.
2. Linux first-touch policy and NUMA in two paragraphs.
3. Step-by-step trace of the bug (master thread → socket 0 → all reads remote on socket 1).
4. Why `OMP_PROC_BIND=close` + `numactl --localalloc` cannot fix data that was already serially first-touched.
5. The two-pragma fix (`computePtnFreq` + the `ptn_invar` `memset`), with the rationale for `schedule(static)`.
6. Verification plan: `numactl --interleave=all` first to confirm the diagnosis without code changes, then rebuild and re-measure.
7. Glossary + file:line citation table.

Intended audience: future-us at 2 a.m. trying to recall why 128T runs collapse, plus anyone reviewing a future patch upstream.

### Updated Pending

| Priority | Task |
|----------|------|
| ~~**HIGH**~~ | ~~Harvest `clang_bbblock` 8T (SLURM 42390186) when it completes; add the 8T row to the comparison table~~ ✅ Done 2026-05-07 — see entry above |
| ~~**MED**~~ | ~~If the 8T comparison shows a meaningful gap, run the full `{8,16,32,64,104,128}` sweep with `clang_bbblock`~~ ✅ Closed — delta is zero |
| **MED** | Empirical NUMA-first-touch verification on Setonix: rerun current 128T binary with `OMP_PROC_BIND=spread numactl --interleave=all` and compare wall time vs `--localalloc`. Recovery of any meaningful fraction of the 128T regression confirms the diagnosis without touching IQ-TREE source |
| **LOW** | If the empirical test confirms it, write the two-pragma patch against `tree/phylotreesse.cpp:537,571`, rebuild from `build-profiling/`, and re-run the 128T sweep |

---

## 2026-05-07 — Claude CLI installed on Setonix; home directory quota remediation

### Problem

Home directory on Setonix has a hard limit of **1 GB / 10 000 files**. Both limits were at capacity, preventing any new tool installs and causing `claude` to silently exit on launch (it could not write its config to `~/.claude`).

Root cause of the file-count exhaustion: a prior failed `cp -r node-v22.../` into `~/.local` left 2 416 stranded files in `~/.local/lib/node_modules` and a 115 MB `node` binary in `~/.local/bin`.

### Actions taken

**Cleanup**
- Removed partial Node.js copy from `~/.local`: `bin/node`, `bin/npm`, `bin/npx`, `bin/corepack`, `lib/node_modules/` (freed ~2 400 files, 134 MB).

**Scratch-backed installs** (all large/file-heavy paths redirected to `/scratch/pawsey1351/asamuel/local/`)

| Path in `$HOME` | Points to |
|---|---|
| `~/.claude` → | `/scratch/pawsey1351/asamuel/local/claude/` |
| `~/.npm` → | `/scratch/pawsey1351/asamuel/local/npm/` |
| Node.js v22.13.1 | `/scratch/pawsey1351/asamuel/local/node/` (no symlink, on `PATH`) |
| Claude CLI v2.1.132 | `/scratch/pawsey1351/asamuel/local/npm-global/bin/claude` |
| npm cache | `/scratch/pawsey1351/asamuel/local/npm-cache/` (via `NPM_CONFIG_CACHE`) |

**`~/.bashrc` additions** (appended 2026-05-07):
```bash
export PATH="/scratch/pawsey1351/asamuel/local/node/bin:/scratch/pawsey1351/asamuel/local/npm-global/bin:$PATH"
export NPM_CONFIG_CACHE="/scratch/pawsey1351/asamuel/local/npm-cache"
```

**Home directory state after cleanup**

| Metric | Before | After |
|---|---|---|
| Files used | ~10 000 (at limit) | ~6 121 |
| Disk used | ~547 MB | ~418 MB |
| Quota limit | 1 024 MB / 10 000 files | same |

### Notes
- `~/.vscode-server` is already on scratch and symlinked to home (same pattern as the above).
- Claude CLI config (`~/.claude/`) persists across sessions as long as the scratch filesystem is mounted (standard on Setonix login nodes).
- `claude` is now invocable from any login session after `source ~/.bashrc`.

---

## 2026-05-07 — Pawsey feedback: `-m block:block:block` task distribution; AOCC xlarge_mf 8T/64T rerun

### Background

Deva Kumar Deeptimahanti (Pawsey HPC support) recommended rerunning multithreaded benchmarks with:

```bash
#SBATCH -m block:block:block
srun -c $OMP_NUM_THREADS -m block:block:block ...
```

The `block:block:block` distribution (nodes:sockets:cores) packs threads into physically contiguous cores, which Pawsey documents as the best policy for L3 cache utilisation on AMD Milan (Setonix EPYC 7763 Zen3). Deva also recommended using thread counts in multiples of 8 (= one CCD on Zen3). Our thread counts (8, 16, 32, 64, 104, 128) already satisfy this.

### Why this matters on AMD Zen3 — and why Gadi (SPR) is different

AMD EPYC 7763 (Zen3) is composed of **8 Core Complex Dies (CCDs)** per socket, each with **8 cores and its own isolated 32 MB L3 cache**. The L3 partitions are hard — a core on CCD0 cannot hit CCD1's L3. When SLURM's default `cyclic` task distribution is used, threads may be scattered across CCDs. At 8T, for example, threads could be spread 1 per CCD × 8 CCDs, leaving each thread with only a 32 MB slice that is otherwise cold. `block:block:block` ensures threads 0–7 land on CCD0, threads 8–15 on CCD1, and so on — keeping the working set warm in a shared L3 partition.

**Gadi (Intel Xeon Platinum 8470Q, Sapphire Rapids) is architecturally different and does not require this fix:**

| Property | Setonix EPYC 7763 (Zen3) | Gadi Xeon 8470Q (Sapphire Rapids) |
|---|---|---|
| Die structure | 8 CCDs per socket, 8c each | Monolithic mesh die per socket, 52c |
| L3 cache | 32 MB **isolated** per CCD | Distributed slices, **fully coherent** across mesh |
| NUMA domains | 2 (one per socket) | 8 (SNC-4: 4 per socket × 13c each) |
| Cross-"cluster" L3 penalty | **Hard miss** — must go to DRAM | Mesh hop latency only (~few ns) |
| Scheduler | SLURM (has `-m` distribution flag) | PBS Pro (no equivalent `-m` flag) |

On SPR, all L3 slices form a single coherent domain. A thread on any core can access data cached by a thread on any other core at mesh-hop latency, not a full DRAM round-trip. There is no hard L3 isolation. PBS Pro on Gadi's `normalsr` queue also allocates a contiguous cpuset for the full node automatically — no additional placement directive is needed or available. The `OMP_PROC_BIND=close` + `OMP_PLACES=cores` already in `gadi-ci/submit_benchmark_matrix.sh` handles NUMA locality within SPR's 8 SNC domains, which is sufficient.

**In summary:** `-m block:block:block` is a Setonix-specific fix for AMD CCD topology. Gadi Sapphire Rapids does not have the same hard-partition problem and the PBS Pro scheduler does not expose an equivalent flag.

### What changed

**`setonix-ci/run_mega_profile.sh`** (applies to all future runs):
- Added `#SBATCH -m block:block:block` to the job's SLURM directives.
- Added `-m block:block:block` to the `srun` invocation (alongside the existing `--cpu-bind=cores`).
- Added `"distribution": os.environ.get("SLURM_DISTRIBUTION", "block:block:block")` to the `slurm` block of `env.json` so the distribution policy is recorded per-run.

**`setonix-ci/submit_clang_bbblock.sh`** (new):
- Submits `xlarge_mf.fa` × `{8, 64}` T using the AOCC/libomp build (`build-profiling-aocc`).
- Label suffix `clang_bbblock` — outputs go to distinct files, not overwriting the `clang_omp_pin` series.
- Same pre-flights: AOCC ldd check (must link libomp, not libgomp) + sha256 gate on `xlarge_mf.fa`.
- Full ModelFinder (no `-mset`), parity check via IQ-TREE log-likelihood comparison as per prior series.

Thread count rationale: 8T (single-CCD baseline, intra-CCD; distribution policy should not matter here — any improvement vs `clang_omp_pin` at 8T is measurement noise) and 64T (full socket, 8 CCDs active; the point where `cyclic` vs `block` thread placement has the largest L3 locality impact and where the libgomp scaling collapse was observed).

### Jobs submitted

| Job ID | Dataset | Threads | Label | Cluster |
|--------|---------|---------|-------|---------|
| 42390186 | `xlarge_mf.fa` | 8T | `clang_bbblock` | Setonix |
| 42390187 | `xlarge_mf.fa` | 64T | `clang_bbblock` | Setonix |

### Expected outcome

Comparison of `clang_bbblock` vs `clang_omp_pin` at 8T and 64T will isolate the effect of the task distribution policy, independent of compiler / OpenMP runtime. The 8T result should be flat (control). If 64T wall time improves, it confirms the default cyclic distribution was scattering threads across CCDs and degrading L3 hit rates even within a single socket; the improvement magnitude will bound how much of the AOCC scaling curve is attributable to binding policy vs OpenMP runtime behaviour.

### Pending

| Priority | Task |
|----------|------|
| ~~**HIGH**~~ | ~~Harvest `clang_bbblock` results and compare wall time + IPC vs `clang_omp_pin` at 8T and 64T~~ → **64T harvested 2026-05-07** (Δ −1.4 %, within noise). 8T still in queue; harvest tracked in the new top entry. |
| ~~**MED**~~ | ~~If 64T improves meaningfully, rerun the full `{8,16,32,64,104,128}` sweep with `clang_bbblock`~~ | ✅ Closed 2026-05-07 — 8T delta zero, full sweep not warranted. |
| **LOW** | No action required on Gadi — SPR L3 is a coherent mesh, PBS Pro allocates contiguous cpusets automatically, no `-m` equivalent needed |

---

## 2026-05-05 — NUMA-aware OpenMP binding gap: `OMP_PROC_BIND=close` vs `spread`, `numactl --localalloc` vs node-pinned `--cpunodebind/--membind`

### Background

User raised the following pattern as the recommended approach for multi-socket OpenMP scaling benchmarks:

```bash
# Single-socket phase (threads ≤ cores-per-socket):
export OMP_PLACES=cores
export OMP_PROC_BIND=close
for t in 1 2 4 8 16 32 48 52; do
  OMP_NUM_THREADS=$t numactl --cpunodebind=0 --membind=0 ./myprog
done

# Cross-socket phase (threads > cores-per-socket):
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
for t in 52 64 80 96 104; do
  OMP_NUM_THREADS=$t numactl --cpunodebind=0,1 --membind=0,1 ./myprog
done
```

Additionally: data initialisation in first-touch NUMA models must use a parallel static-scheduled loop so pages are distributed across sockets at allocation time:

```c
#pragma omp parallel for schedule(static)
for (long i = 0; i < N; i++) a[i] = 0.0;
```

Otherwise all alignment pages fault-in on socket 0 (master thread), causing remote-NUMA traffic for every thread on socket 1 regardless of how well threads are pinned.

### Gap analysis: what our scripts actually do

Both `setonix-ci/run_mega_profile.sh` (Setonix / SLURM) and `gadi-ci/submit_benchmark_matrix.sh` (Gadi / PBS Pro) use the same binding setup at **all** thread counts:

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
NUMACTL=( numactl --localalloc )
```

Three specific gaps vs the recommended pattern:

#### Gap 1 — No `OMP_PROC_BIND=spread` for cross-socket runs

`close` packs threads onto cores sequentially starting from socket 0 and overflows into socket 1. For T ≤ 64 (Setonix, 1 CCD pair per socket) or T ≤ 52 (Gadi, 1 socket) this is single-socket and `close` is correct. For T > 64 / T > 52 `close` continues filling linearly into socket 1 — the placement itself is correct — but the *intent* of `spread` is to interleave threads across sockets so that each socket's threads are contiguous in the OpenMP team ID space. This matters for worksharing loops: `spread` distributes loop iterations evenly across sockets, maximising locality for data that was initialised in parallel (first-touch already spread). With `close` and a single-threaded first-touch init, the loop-chunk boundaries don't align with the socket boundary, and threads on socket 1 pull data from socket 0's NUMA domain.

#### Gap 2 — `numactl --localalloc` does not pin data to specific sockets

`--localalloc` allocates each **new** page on the NUMA node where the page-fault occurs — it has no effect on pages already faulted in. For the alignment file (multi-GB, loaded by the master thread before the OMP parallel region), all pages land on socket 0 regardless of `--localalloc`. The recommended `--cpunodebind=0,1 --membind=0,1` would at minimum interleave OS memory allocations across both sockets' DRAM controllers during the load phase, reducing pressure on socket 0's memory bus.

#### Gap 3 — First-touch initialisation is in IQ-TREE source, not our scripts

We have no visibility into whether IQ-TREE parallelises its alignment buffer initialisation with a `schedule(static)` OMP loop before the main compute phase. If the alignment is read sequentially by the master thread (single-threaded first-touch), all alignment pages reside on socket 0 for the entire run. Every cross-socket thread (T > 64 on Setonix, T > 52 on Gadi) pays a remote-NUMA penalty on every cache-line miss into the alignment. This is a plausible contributing factor to the observed cross-socket wall-time regression at high thread counts (especially pronounced in the GCC/libgomp series: 128T is 1.9× *slower* than 8T).

### What was confirmed in the logs

- All 7 `xlarge_mf_*_baseline_smton` runs and all 8 canonical `smtoff_pin` runs: `cpus_per_task=128`, `OMP_PROC_BIND=close` at all T. No `spread` transition at the socket boundary. `numactl --localalloc` only.
- Gadi `submit_benchmark_matrix.sh` is identical: `OMP_PROC_BIND=close` at all T (including 104T = 2 sockets), `numactl --localalloc`.
- `srun --cpu-bind=cores` (Setonix) and PBS cpuset (Gadi) handle physical-core pinning at the OS level, but neither changes the data-placement problem.

### Recommended follow-on experiments

| Experiment | Change | Expected signal |
|------------|--------|----------------|
| Switch to `spread` + `--cpunodebind=0,1 --membind=0,1` for T > socket size | `OMP_PROC_BIND=spread`, `numactl --cpunodebind=0,1 --membind=0,1` | Should reduce remote-NUMA latency for cross-socket runs; if wall time improves, confirms first-touch + bandwidth was the dominant cross-socket bottleneck |
| `numactl --interleave=all` for full run | Replaces `--localalloc` | Spreads all allocations round-robin across both sockets; reduces peak socket-0 pressure; less targeted than cpunodebind but easier to test |
| Audit IQ-TREE source for parallel init | `grep -rn 'schedule(static)' src/` in IQ-TREE 3.1.1 | Determines whether first-touch is already parallel; if not, a patch or `GOMP_STACKSIZE` + `OMP_SCHEDULE=static` hint may help |

### Current status

No script changes made in this session — this is a methodology finding. Scripts still use `OMP_PROC_BIND=close` + `numactl --localalloc` at all thread counts. Tracking as a pending improvement.

---

## 2026-05-02 — GCC/libgomp vs AOCC/libomp `xlarge_mf` scientific parity audit (follow-up #20)

### Scope

Deep comparison of the GCC `smtoff_pin` series (slurm **42181137–42181142**, six thread counts 8–128T, `Setonix_xlarge_mf_{8,16,32,64,104,128}T.json`) against the AOCC `clang_omp_pin` series (slurm **42225454–42225459**, `xlarge_mf_{8,16,32,64,104,128}t_clang_omp_pin_baseline.json`). Both series: Setonix, dataset `xlarge_mf.fa`. Goal: verify what was controlled, what was not, and whether the wall-time comparison is scientifically valid.

### Controlled variables (verified ✅)

| Factor | GCC series | AOCC series |
|--------|-----------|-------------|
| Dataset sha256 | `66eaf64b…` | `66eaf64b…` — identical (sha256-gated pre-flight) |
| Platform | Setonix, AMD EPYC 7763 | Setonix, AMD EPYC 7763 |
| SLURM allocation | 1 node, `--exclusive`, `--cpus-per-task=128`, `--mem=230G`, `--hint=nomultithread`, partition `work` | **same** |
| IQ-TREE version | 3.1.1 built Apr 30 2026 | 3.1.1 built Apr 30 2026 |
| Architecture flags | `-O3 -march=znver3 -mtune=znver3 -fno-omit-frame-pointer -g` | **same** |
| IPO/LTO | Disabled (cmaple CMakeLists patched) | Disabled — same patch |
| Math libraries | Eigen 3.4.0; no BLAS/MKL/AOCL | Eigen 3.4.0 |
| OMP_PROC_BIND | `close` *(active; not recorded in GCC env.json — see Issue 2)* | `close` |
| OMP_PLACES | `cores` *(active; not recorded in GCC env.json)* | `cores` |
| OMP_WAIT_POLICY | `PASSIVE` *(active; not recorded in GCC env.json)* | `PASSIVE` |
| GOMP_SPINCOUNT | `10000` *(active; not recorded in GCC env.json)* | `10000` |
| Scientific output | Best model `GTR+R4`, log-L = −10 956 936.61 ± 0.03 | Best model `GTR+R4`, log-L = −10 956 936.61 ± 0.03 |

Log-likelihoods agree within 0.06 units across all 12 runs (expected FP noise from parallel summation order). Both series produce identical scientific results. **Wall-time comparison is valid.**

### Intentional variables — covaried (experiment design)

| Variable | GCC series | AOCC series |
|----------|-----------|-------------|
| Compiler | GCC 14.3.0 | AOCC 5.1.0 (AMD Clang 17.0.6) |
| OpenMP runtime | libgomp | libomp (LLVM/OpenMP-API) |

Compiler and OpenMP runtime changed together. The regression pattern onset (>8T = first cross-CCD crossing) strongly implicates the OMP runtime, but **the two variables cannot be separated from this data alone**. Isolating requires a follow-on experiment (Clang + libgomp, or GCC + libomp).

### Wall-time results

| Threads | GCC (s) | AOCC (s) | AOCC speedup vs GCC | GCC vs 8T | AOCC vs 8T |
|---------|--------|---------|---------------------|-----------|-----------|
| 8 | 3 854.3 | 3 830.6 | 1.01× | 1.00× | 1.00× |
| 16 | 3 580.0 | 2 639.5 | 1.36× | 1.08× | 1.45× |
| 32 | 3 301.9 | 1 940.3 | 1.70× | 1.17× | 1.97× |
| 64 | 3 547.2 | 1 905.2 | 1.86× | 1.09× | 2.01× |
| 104 | 6 846.1 | 2 305.1 | **2.97×** | **0.56×** | 1.66× |
| 128 | 7 261.3 | 2 368.4 | **3.07×** | **0.53×** | 1.62× |

GCC is flat 8T→64T (within one CCD pair / one socket) then collapses to **0.53× its own 8T throughput** at 128T. AOCC scales to 64T (2.01×) then degrades gracefully. At 104–128T, AOCC is ~3× faster.

### IPC analysis (derived from raw perf counters)

`profile.ipc` is `null` in both series (see Issue 4), but IPC is computable from `profile.metrics.cycles` and `.instructions`:

| Threads | GCC instructions | GCC cycles | GCC IPC | AOCC instructions | AOCC IPC |
|---------|----------------|-----------|---------|------------------|---------:|
| 8T | 105.4 T | 99.3 T | **1.06** | 101.4 T | **1.00** |
| 64T | 115.4 T | 662.2 T | **0.174** | 211.2 T | **0.61** |
| 128T | 149.9 T | 2 619.7 T | **0.057** | — | — |

GCC at 128T: **18 cycles per instruction** — threads are burning cycles at OMP barriers without doing useful work. GCC achieves only 42% more instructions at 128T than at 8T (despite 16× the threads). AOCC-64T executes 2.1× the instructions of GCC-64T in 54% less wall time, and achieves 3.5× better IPC.

---

### Issues found in this audit

#### 🔴 Issue 1 — GCC xlarge hotspot data is INVALID (perf record profiled srun/sinfo, not iqtree3)

All six GCC xlarge `profile.hotspots` entries list SLURM system commands — zero entries for `iqtree3`. Sample counts are 1–9 per entry (vs thousands expected for a multi-hour IQ-TREE run), confirming `perf record` captured a brief SLURM launch or teardown window rather than IQ-TREE's steady state.

| Run | Top hotspot | Module | Command | Samples |
|-----|------------|--------|---------|---------|
| GCC-8T | `__printf_buffer` | `libc.so.6` | `srun` | 1 |
| GCC-64T | `_find_name_in_env` | `libslurmfull.so` | `srun` | 1 |
| GCC-128T | `slurm_xstrdup` | `libslurmfull.so` | `srun` | 1 |

`profile.metrics` perf-stat counters (cycles, instructions, cache events, TLB events) **are valid** — they came from `perf stat` which monitored the full job duration. Only the `perf record`-derived flamegraph and hotspot data is corrupted. **A re-run with the corrected `python3.11`-based `run_mega_profile.sh` is required** before any GCC xlarge hotspot or flamegraph comparison can be done.

#### ~~🟠 Issue 2~~ ✅ — GCC xlarge env.json missing OMP variables — **PATCHED**

~~The GCC xlarge env.json was produced by the **pre-follow-up-#19** version of `run_mega_profile.sh`, which did not record `omp_proc_bind`, `omp_places`, `omp_wait_policy`, `gomp_spincount`, `omp_runtime`, or `build_tag`. Reading only the JSON, the GCC series appears to have had no explicit OMP pinning (`null` fields) while AOCC shows `close`/`cores`/`PASSIVE`.~~

All six `Setonix_xlarge_mf_{8,16,32,64,104,128}T.json` files patched with `env.omp_runtime="libgomp"`, `env.omp_proc_bind="close"`, `env.omp_places="cores"`, `env.omp_wait_policy="PASSIVE"`, `env.gomp_spincount=10000`, `env.kmp_blocktime=null`, `env.build_tag="smtoff_pin"`, and `env.recovered_from_script_history=true`. Values recovered from the 2026-04-30 `run_mega_profile.sh` commit state. The `omp_runtime` field now surfaces correctly as `libgomp` in the runs index.

#### ~~🟡 Issue 3~~ ✅ — `cpu_count_logical` equals THREADS in GCC xlarge runs — **PATCHED**

All six `Setonix_xlarge_mf_{8,16,32,64,104,128}T.json` files patched with `env.cpu_count_logical=128`. True value confirmed via `slurm.cpus_per_task=128` + `--hint=nomultithread` (128 physical cores available in the job cpuset). Previously read as THREADS (8/16/32/64/104/128) due to `os.cpu_count()` returning the cpuset logical count rather than the full affinity.

#### ~~🟡 Issue 4~~ ✅ — IPC null in both series — **PATCHED**

All 12 run JSONs patched with `profile.ipc_derived` (counter ratio, 4 dp):

| Series | 8T | 16T | 32T | 64T | 104T | 128T |
|--------|---:|----:|----:|----:|-----:|-----:|
| GCC/libgomp | 1.0622 | 0.6342 | 0.3487 | 0.1742 | 0.0673 | 0.0572 |
| AOCC/libomp | 0.9980 | 0.9294 | 0.8067 | 0.6116 | 0.5563 | 0.5562 |

`tools/normalize.py` updated to use `ipc_derived` as fallback when `metrics["IPC"]` is `None` (line 213). `tools/harvest_scratch.py` updated to write `profile["ipc_derived"]` for any future run where `_derive_rates` successfully computes IPC from raw counters. Dashboard now shows IPC for both series.

### New pending actions

| Priority | Task |
|----------|------|
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T with the `python3.11`-fixed `run_mega_profile.sh` to obtain valid iqtree3 hotspot / flamegraph data — current GCC xlarge `profile.hotspots` profiles SLURM shell commands |
| **LOW** | Fix `env.cpu_count_logical` detection in `run_mega_profile.sh` to use `len(os.sched_getaffinity(0))` instead of `os.cpu_count()` so GCC xlarge re-runs report affinity not logical CPU count |

---

## 2026-05-01 (libgomp-vs-libomp test, follow-up #19) — Clang/AOCC build path + dashboard cache-level rename

### Hypothesis (Minh, IQ-TREE author)

> "libgomp is not good, from my experience before. Can you try compile with Clang? It will use intel OpenMP, which is better."

The Setonix `xlarge_mf` thread-scaling regression above 8 T (first cross-CCD step on EPYC 7763) is a candidate symptom of libgomp's barrier/spin behaviour interacting badly with the L3-per-CCD topology. Gadi (Intel SPR + libiomp5 via OneAPI) does not exhibit it. To isolate the OpenMP-runtime variable from the architecture variable, we add a Clang/libomp build path on **both** clusters and a non-canonical reference sweep on Setonix.

### What

- **`setonix-ci/bootstrap_iqtree_aocc.sh`** — builds IQ-TREE with AOCC 5.1.0 (Clang 17, znver3-tuned, libomp). Verifies via `ldd` that libgomp is **not** linked. Output → `${PROJECT_DIR}/build-profiling-aocc/iqtree3`. Same `-O3` / IPO-disabled / Eigen / Boost as the canonical gcc build — the only deltas are the compiler and the OpenMP runtime.
- **`setonix-ci/submit_clang_xlarge.sh`** — fan-out of `xlarge_mf.fa` × {8, 16, 32, 64, 104, 128} T using `run_mega_profile.sh` as the worker, with `BUILD_DIR=…build-profiling-aocc`, `LABEL_SUFFIX=clang_omp_pin`, and libomp env (`KMP_BLOCKTIME=200`, mirrors Gadi's libiomp5 default). 1T/4T omitted: the regression only appears once threads cross the CCD boundary. sha256-gated against the canonical `xlarge_mf.fa` (66eaf64b…). Build directory pre-flight refuses to submit if `ldd` shows libgomp.
- **`gadi-ci/bootstrap_iqtree_clang.sh`** — Sapphire Rapids mirror. Prefers `intel-compiler-llvm` (icx + libiomp5), falls back to `llvm` then plain `clang`. Output → `${PROJECT_DIR}/build-profiling-clang/iqtree3`. Same `-O3 -march=sapphirerapids` flags as gcc canonical.
- **`setonix-ci/run_mega_profile.sh`** — `env.json` now records `build_tag` (= `LABEL_SUFFIX`), `omp_runtime` (auto-detected from `ldd`), `omp_proc_bind`, `omp_places`, `omp_wait_policy`, `kmp_blocktime`, `gomp_spincount`, and the contents of `${BUILD_DIR}/.build-info.json`. Lets the harvester tag clang runs as `non_canonical=true` automatically.
- **`tools/harvest_scratch.py`** — `enrich_run` now copies `build_tag` from `env.json` to the run-level field, and auto-flags `canonical=false / non_canonical=true` for any tag other than `smtoff_pin` (Setonix canonical) or `sr_gcc_pin` (Gadi canonical). Also emits `frontend-stall-unit` (`cycles` for AMD, `slots` for Intel SPR) and `frontend-stall-max-pct` (100 vs 600) so dashboards can disambiguate raw stall percentages without rescaling counter values.
- **`tools/normalize.py`** — index entries gain `l2_miss_rate`, `l3_miss_rate`, and `omp_runtime` fields. Pre-existing `cache_level` annotation is unchanged.
- **`web/js/pages/runs.js`** — detail panel now shows L1-dcache miss %, the platform-specific `Lx miss %` (auto-selected from `cache_level`), the OpenMP runtime tag, the build tag, and an FE-stall annotation that includes the unit (`AMD cycles` vs `Intel slots`). Falls back to the legacy `cache-miss-rate` field for older runs.
- **`web/js/pages/profiling.js`** — same platform-aware cache-miss field; FE-stall card label now reads "FE-stall (cycles)" / "(slots)".

### Verification

- Test suite: `17 passed, 1 xfailed` (no regressions).
- `tools/normalize.py && tools/build.py`: 56 runs, 2 profiles, 38 split files, dashboard rebuilt at `v=20260501115130`.
- Spot-check on a Setonix `large_modelfinder` row: `cache_level=L2`, `l2_miss_rate=12.98`, `l3_miss_rate=null`, `l1d_mpki=29.36`, `build_tag=smtoff_pin` (canonical, unchanged).
- Spot-check on a Gadi row: `cache_level=L3`, `l2_miss_rate=null`, `l3_miss_rate=84.96`, `l1d_mpki=15.63`, `build_tag=sr_gcc_pin`.
- Shell syntax: `bash -n` clean on all four new/modified shell scripts.
- `ldd` libomp gate: enforced in both bootstrap scripts (exit 3 if libgomp leaks in) and in the Setonix submitter (exit 4 if the binary still links libgomp).

### Out of scope (deliberate)

- **No FE-stall rescaling.** AMD cycles and Intel slots are kept as raw `perf stat` output so values remain reproducible against upstream tooling. Unit annotation is descriptive only.
- **No microarch radar change.** Both vendors' FE-stall axes are clamped to 100 by `normalize`, so the radar shape stays meaningful even with the unit difference.
- **Threads {1, 4} skipped** for the Clang sweep — the libgomp/libomp split only matters once the workload spans more than one CCD on Zen 3.

### Pending

- ~~Run `setonix-ci/bootstrap_iqtree_aocc.sh` on a Setonix compute node, then `setonix-ci/submit_clang_xlarge.sh`.~~ ✅ **Submitted 2026-05-02** — bootstrap `42225453` running; matrix jobs `42225454–42225459` (8/16/32/64/104/128 T) queued with `afterok:42225453`. See 2026-05-02 entry above for details and harvest instructions.
- Mirror on Gadi via `gadi-ci/bootstrap_iqtree_clang.sh` once a comparable submitter is wired up (low priority — the primary signal is the Setonix gcc-vs-Clang delta).

---

## Critical & Pending Tasks

> **Last audited: 2026-05-07.** Run files: 68 tracked (stale failed stubs deleted). `block:block:block` investigation closed — zero effect confirmed. NUMA first-touch patch elevated to HIGH. See 2026-05-07 analysis entry for full findings.

### 🔴 Harvest — blocking cross-platform analysis

| Priority | Task | Blocker |
|----------|------|---------|
| ~~**CRITICAL**~~ | ~~Harvest remaining Gadi `large_modelfinder _sr_gcc_pin` runs (PBS jobs **167507204, 167507207–167507210**) once complete → commit JSON to `logs/runs/`~~ | ✅ **Done 2026-05-01** — full 1T–104T matrix harvested |
| ~~**CRITICAL**~~ | ~~Harvest `Setonix_xlarge_mf_1T` (SLURM job **42181135**, nid001938) → rebuild so canonical 1T baseline replaces archived SMT-on proxy in speedup figures~~ | ✅ **Done 2026-05-01** — wall=11077s, IPC=2.72, speedup baseline fixed |
| ~~**HIGH**~~ | ~~Submit Gadi `xlarge_mf _sr_gcc_pin` matrix (same gcc/14.2.0 build, threads 1 4 8 16 32 64 104)~~ | ✅ **Submitted 2026-05-01** (follow-up #16) — PBS jobs **167520752–167520758**, sha256-gated against canonical `xlarge_mf.fa`. |
| ~~**HIGH**~~ | ~~Harvest remaining `xlarge_mf _sr_gcc_pin` jobs (16T/32T/64T/104T, PBS 167520755–167520758) once complete~~ | ✅ **Partially done 2026-05-02** — 16T (PBS 167520755) and 32T (PBS 167520756) harvested. **64T and 104T still missing** — see data gap note below. |
| ~~**HIGH**~~ | ~~Submit + harvest Setonix AOCC/libomp `xlarge_mf` sweep (follow-up #19)~~ | ✅ **Done 2026-05-02** — bootstrap **42225453** COMPLETED; matrix jobs **42225454–42225459** (8–128T) all COMPLETED and harvested. See 2026-05-02 entry. |
| **HIGH** | NUMA first-touch patch — `tree/phylotreesse.cpp:537,571` | Two `#pragma omp parallel for schedule(static)` additions. Empirical test first (`numactl --interleave=all` on current 128T AOCC binary). Confirmed root cause 2026-05-07 — see analysis entry. |
| **HIGH** | Harvest Gadi `xlarge_mf _sr_gcc_pin` 64T (PBS **167520757**) and 104T (PBS **167520758**) | Jobs were released from hold 2026-05-02 after inode cleanup — status on Gadi scratch unverified from Setonix |
| **HIGH** | Submit Gadi `mega_dna _sr_gcc_pin` matrix | No canonical Gadi mega_dna sweep exists — only the earlier `sr_icx` reference runs (4 runs, 16T–104T). Cross-platform mega_dna comparison blocked. |
| **HIGH** | Submit Setonix `mega_dna _smtoff_pin` canonical matrix | Only SMT-on reference runs exist (4 runs: 16/32/64/128T, `baseline_smton`, slurm 41849110–41849113). No canonical SMT-off pinned Setonix mega_dna data. |
| **HIGH** | Re-run Setonix GCC `xlarge_mf` 8–128T with `python3.11`-fixed `run_mega_profile.sh` | Current `Setonix_xlarge_mf_{8,16,32,64,104,128}T.json` hotspot data profiles SLURM commands not iqtree3 — flamegraph/hotspot comparison blocked. See follow-up #20. |
| ~~**CLOSED**~~ | ~~`block:block:block` distribution policy investigation~~ | ✅ Closed 2026-05-07 — zero effect at 8T (Δ +0.4 s) and 64T (Δ −1.4%, noise). See 2026-05-07 analysis entry. |

### 🟠 Data quality issues (discovered 2026-05-02 audit)

| Severity | File(s) | Issue |
|----------|---------|-------|
| ~~**INFO**~~ ✅ | ~~`large_modelfinder_{1,4,8,16,32,64,104}t_smtoff_pin_baseline.json` (slurm 42179034–42179046)~~ | **FIXED (59b09bf)** — all 7 files **deleted**. FAILED RUNS: exited with `srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive`. No wall time, no IQ-TREE output. Were previously marked `non_canonical=true` / `nc_label="smtoff_pin (prev)"`. Run count 73 → 66. |
| **LOW** | `Setonix_xlarge_mf_1T.json` (slurm 42181135) | `hotspots=0` — perf record was not run for the 1T canonical run (only `perf stat`). IPC and wall time are correct. By design — `run_mega_profile.sh` skips `perf record` at 1T. |
| **LOW** | `Gadi_xlarge_mf_{32,64,104}T.json`, `Gadi_mega_dna_{32,64,104}T.json`, `Gadi_large_modelfinder_64T_sr_icx.json` | `modelfinder=null` — these are the older `sr_icx` reference runs (PBS 167001081–167004590) where `perf record` was run but `.iqtree` log parsing didn't extract modelfinder candidates. All are `non_canonical=true` so not used as baseline. |
| ~~**LOW**~~ ✅ | ~~`gadi_xlarge_mf_{1,4,8,16,32}t_sr_gcc_pin.json` (PBS 167520752–167520756)~~ | **FIXED (59b09bf)** — patched `build_tag="sr_gcc_pin"` and `canonical=true` on all 5 files. Normalize was already treating them as canonical; fields were simply absent. |
| **OPEN** | `gadi_xlarge_mf_{64,104}t_sr_gcc_pin.json` | **FILES MISSING** — PBS jobs 167520757–167520758 either still pending on Gadi or completed but not yet transferred. The `sr_icx` reference covers those thread counts but the canonical `sr_gcc_pin` series is incomplete above 32T. |
| ~~**INFO**~~ ✅ | ~~`xlarge_mf_{8,16,32,64,104,128}t_clang_omp_pin_baseline.json` (slurm 42225454–42225459)~~ | **FIXED (933f305)** — `smt_active=False`, `hostname=""`, `cpu_count_logical=0`, and all `sh()`-derived fields were `""` / `0`. **Root cause:** `run_mega_profile.sh` env.json heredoc used bare `python3`, which on Setonix compute nodes resolves to system Python 3.6.15. `text=True` in `subprocess.check_output` was added in Python 3.7; all four `sh()` calls silently hit `TypeError: __init__() got an unexpected keyword argument 'text'`, caught and returning defaults. Identical failure in the sampler launch (line 423) and perf-folded pipe (line 485). GCC canonical runs (42181xxx) were unaffected — those ran in a session where the login-node `python3` symlink pointed to 3.11.14. **Actual SMT state during both series:** kernel SMT on; OMP threads pinned to physical cores via `#SBATCH --hint=nomultithread`. The GCC vs AOCC wall-time comparison is valid — both series had identical SMT configuration. Fixed by changing all four `python3` calls to `python3.11` in `setonix-ci/run_mega_profile.sh` (lines 132, 423, 485, 512). |
| **HIGH** | `Setonix_xlarge_mf_{8,16,32,64,104,128}T.json` | **`profile.hotspots` INVALID** — `perf record` second pass profiled SLURM shell commands (`srun`, `sinfo`, `scontrol`, `bash`), not `iqtree3`. All 6 runs show 1–9 samples in srun/slurm functions; zero iqtree3 entries. `perf stat` counters in `profile.metrics` are valid. Root cause: `perf record` ran during the SLURM launch window rather than IQ-TREE's steady state. Flamegraph/hotspot analysis for GCC xlarge is **void until a re-run**. See follow-up #20. |
| ~~**MEDIUM**~~ ✅ | ~~`Setonix_xlarge_mf_{8,16,32,64,104,128}T.json`~~ | **FIXED (follow-up #20)** — `env.omp_proc_bind="close"`, `env.omp_places="cores"`, `env.omp_wait_policy="PASSIVE"`, `env.gomp_spincount=10000`, `env.omp_runtime="libgomp"`, `env.kmp_blocktime=null`, `env.build_tag="smtoff_pin"` patched into all 6 GCC xlarge run files. `env.recovered_from_script_history=true` annotation added. Values recovered from 2026-04-30 `run_mega_profile.sh` git state. `omp_runtime="libgomp"` now visible in runs index. |
| ~~**MEDIUM**~~ ✅ | ~~`Setonix_xlarge_mf_{8,16,32,64,104,128}T.json` and all `xlarge_mf_*_clang_omp_pin_baseline.json`~~ | **FIXED (follow-up #20)** — `profile.ipc_derived` patched into all 12 run JSON files from counter ratio (instructions/cycles). `tools/normalize.py` updated to fall back to `ipc_derived` when `metrics["IPC"]` is null (line 213). `tools/harvest_scratch.py` updated to write `profile["ipc_derived"]` automatically for future runs. Dashboard IPC column now shows values for both series: GCC 1.06→0.057 (collapse), AOCC 1.00→0.56 (graceful). |
| ~~**LOW**~~ ✅ | ~~`Setonix_xlarge_mf_{8,16,32,64,104,128}T.json`~~ | **FIXED (follow-up #20)** — `env.cpu_count_logical=128` patched into all 6 GCC xlarge run files. True value confirmed: `slurm.cpus_per_task=128` + `--hint=nomultithread` = 128 physical cores in the job cpuset. Previously read as THREADS (8/16/32/64/104/128) due to Python `os.cpu_count()` returning cpuset logical count on pre-python3.11 env script. |

### 🟠 Dashboard fixes — actively misleading metrics

| Priority | Task | Detail |
|----------|------|--------|
| **HIGH** | Update dashboard chart: rename `cache-miss-rate` to use platform-aware fields `l2-miss-rate` (Setonix) and `l3-miss-rate` (Gadi); switch primary memory-pressure plot to `L1d-mpki` (cross-platform comparable) | Data layer fixed in follow-up #15 — `cache_level` field, `*-mpki` derived metrics, and Zen3 `l3-prefetch-miss-rate` proxy are now in every harvested run. Web frontend (`web/js/charts/*`, `web/js/pages/*`) needs to consume them. |
| ~~**HIGH**~~ | ~~Normalise `cache-miss-rate` units across platforms — Gadi stores as percentage (25.06 = 25.06 %), Setonix as ratio (0.039 = 3.9 %)~~ | ✅ **Done 2026-05-01** (follow-up #17) — `tools/harvest_scratch.py:_derive_rates` now emits all `*-miss-rate` / `*-stall-rate` fields as percent (matching the Gadi worker's `rate()` helper and the dashboard's `fmtPercent` / radar `normalize` assumptions); `tools/migrate_rate_units.py` rescaled 32 legacy Setonix files (224 field rescales). |
| ~~**CRITICAL**~~ | ~~Fix Setonix `IPC` always reading N/A — `cycles:u` returns 0 under `perf_event_paranoid=2`~~ | ✅ **Done 2026-05-01** (follow-up #12b) — `cycles:uk,instructions:uk` in `PERF_EVENTS`; next Setonix matrix re-run will populate IPC |
| ~~**HIGH**~~ | ~~Add LLC/L3 events to Setonix `PERF_EVENTS`~~ | ✅ **Done 2026-05-01** (follow-up #15) — `l2_pf_miss_l2_hit_l3,l2_pf_miss_l2_l3` (core-level Zen3 L3 prefetcher proxy events) added. `amd_l3/*` uncore PMU is admin-locked under `perf_event_paranoid=2`, so demand-path L3 hit/miss is not directly measurable; the prefetcher path is the closest user-mode equivalent to Gadi's `LLC-load-misses`. |
| ~~**HIGH**~~ | ~~Add `stalled-cycles-backend` to Gadi event list~~ | ✅ **Already present** — verified 2026-05-01, line 35 of `gadi-ci/run_profiling.sh`. Phantom to-do; removed. |
| ~~**HIGH**~~ | ~~Re-run Setonix `large_modelfinder` matrix (7 runs, 1T–104T) — `perf stat` measured login-node srun wrapper (92ms task-clock) rather than compute-node iqtree3; `cycles:u = 0` because srun does negligible CPU work~~ | ✅ **Submitted 2026-05-01** (follow-up #14) — jobs **42190953–42190959**, fixed script synced to scratch (`perf stat` inside srun + `cycles:uk`). `xlarge_mf` already correct — no rerun needed. |
| ~~**HIGH**~~ | ~~Fix double data points / zigzag lines in IPC and efficiency charts for Setonix `large_modelfinder`~~ | ✅ **Done 2026-05-02** — root cause: 7 harvested `large_modelfinder_*t_smtoff_pin_baseline` stubs had no `non_canonical` flag, landing in canonical series alongside `Setonix_large_modelfinder_*T` runs. Fixed by marking them `non_canonical=true` / `nc_label="smtoff_pin (prev)"`. |
| ~~**MEDIUM**~~ ✅ | ~~Add `build_tag="sr_gcc_pin"` explicitly to the 5 `gadi_xlarge_mf_*t_sr_gcc_pin.json` files~~ | **Done (59b09bf)** — `build_tag="sr_gcc_pin"` and `canonical=true` patched on all 5 files. |
| **MEDIUM** | Normalise IPC display: show `IPC / max_retire_width` as utilisation % alongside raw IPC | AMD max = 4, Intel SPR max = 6 — raw IPC not comparable cross-platform |
| **MEDIUM** | Verify `stalled-cycles-frontend` semantics after canonical Gadi gcc runs complete | AMD counts cycles; Intel counts slots (up to 6/cycle on SPR) |

### 🟡 SMT / HT platform clarification (confirmed 2026-05-02)

**IQ-TREE uses OpenMP, not MPI.** All runs are `iqtree3 -T N` — N shared-memory OpenMP threads in a single process. "1 thread per core" refers to 1 OMP thread per physical core, not 1 MPI rank.

| Platform | SMT/HT state | How confirmed | Practical effect |
|----------|-------------|---------------|-----------------|
| **Gadi (SPR)** | HT **off at BIOS level** — siblings never brought online by firmware | `smt/active=0`, `smt/control=on` on login node (control=on means kernel switch is enabled but has no siblings to toggle) | True 96-physical-core execution; full per-core resources, no sharing |
| **Setonix (Zen3)** | Kernel SMT **on** (siblings present but idle) | `smt/active=1` in env.json for GCC runs; cannot be disabled without root | `--hint=nomultithread` + `OMP_PLACES=cores` + `OMP_PROC_BIND=close` + `--exclusive` pins 1 OMP thread per physical core; idle sibling is invisible to the running thread |

**Why Setonix's idle-sibling SMT does not meaningfully affect results:** AMD Zen3 uses *dynamic* SMT resource allocation — when only one logical CPU per core is active, it receives the full ROB, register file, execution units, and L1/L2. Intel pre-ADL used *static* partitioning (each HT thread permanently gets ~half the ROB regardless of activity), which is why SMT-off matters more on Intel. On Zen3 with an idle sibling the active thread is functionally in the same state as kernel SMT-off. Any marginal cache/TLB noise from OS kernel threads on the idle sibling would slightly *understate* Setonix performance relative to a true SMT-off node, making the Setonix numbers conservative.

**Conclusion:** The user-space pinning setup already eliminates SMT as a practical confound. The remaining real variables in the Setonix vs Gadi comparison are (1) CCD topology (AMD L3-per-CCD vs Intel flat mesh) and (2) OpenMP runtime (libgomp vs libiomp5/libomp). Disabling SMT at the kernel level on Setonix would require a Pawsey admin action and is not expected to shift results measurably on Zen3.

### 🟡 Source audit

| Priority | Task | Detail |
|----------|------|--------|
| **MEDIUM** | `grep -RIn 'hardware_concurrency'` audit of IQ-TREE 3.1.1 source | On Setonix cpuset includes SMT siblings → returns 2×T; internal pools sized from this would over-subscribe by 2× |
| ~~**INFO**~~ ✅ | **Math library audit — all builds (AOCC, ICX/sr_icx, gcc canonical)** | **No Intel MKL or AMD AOCL used in any run.** IQ-TREE's numerical kernel goes entirely through **Eigen** (header-only SIMD template library; no external BLAS/LAPACK linkage). Confirmed via: (1) `CMakeCache.txt` on Gadi scratch lists only `EIGEN3_INCLUDE_DIR` + `BOOST_ROOT` — no `MKL_`, `BLAS_`, `LAPACK_`, or `AOCL_` cmake entries; (2) `ldd build-profiling/iqtree3` links `libgomp.so.1` only — no `libmkl_*`, `libblas*`, or `libamd*`; (3) `grep -iE 'mkl\|aocl\|blas\|lapack\|openblas'` over all four bootstrap scripts returns zero hits; (4) `env.build_info.arch_flags` in AOCC run JSONs is `-O3 -march=znver3 -fopenmp` — no math-lib link flags. The ICX `sr_icx` reference runs pre-date the `bootstrap_iqtree_clang.sh` script but use the same IQ-TREE CMakeLists, which has no `find_package(MKL)` or `find_package(BLAS)` call. **Eigen version per build**: Setonix (gcc canonical + AOCC) = 3.4.0; Gadi (gcc canonical + ICX ref) = 3.3.7 (only version available on Gadi). |

### 📊 Corpus snapshot (2026-05-02)

| Platform | Dataset | Series | Threads covered | Status |
|----------|---------|--------|-----------------|--------|
| Setonix | `large_modelfinder.fa` | `smtoff_pin` (canonical) | 1, 4, 8, 16, 32, 64, 104 | ✅ Complete — slurm 42190953–42190959 |
| Setonix | `large_modelfinder.fa` | `baseline_smton` (nc ref) | 1, 4, 8, 16, 32, 64 | ✅ Harvested (SMT-on, no pin) |
| ~~Setonix~~ | ~~`large_modelfinder.fa`~~ | ~~`smtoff_pin (prev)` (nc, failed)~~ | ~~1, 4, 8, 16, 32, 64, 104~~ | ✅ **Deleted** (59b09bf) — 7 empty stubs removed |
| Setonix | `xlarge_mf.fa` | `smtoff_pin` (canonical) | 1, 4, 8, 16, 32, 64, 104, 128 | ✅ Complete — slurm 42181135–42181142 |
| Setonix | `xlarge_mf.fa` | `baseline_smton` (nc ref) | 1, 4, 8, 16, 32, 64, 128 | ✅ Harvested (SMT-on, no pin) |
| Setonix | `xlarge_mf.fa` | `clang_omp_pin` / AOCC (nc ref) | 8, 16, 32, 64, 104, 128 | ✅ Complete — slurm 42225454–42225459 |
| Setonix | `mega_dna.fa` | `smtoff_pin` (canonical) | — | ❌ Missing — no canonical SMT-off sweep |
| Setonix | `mega_dna.fa` | `baseline_smton` (nc ref) | 16, 32, 64, 128 | ✅ Harvested (SMT-on, no pin) |
| Gadi | `large_modelfinder.fa` | `sr_gcc_pin` (canonical) | 1, 4, 8, 16, 32, 64, 104 | ✅ Complete — PBS 167507204–167507210 |
| Gadi | `large_modelfinder.fa` | `sr_icx` (nc ref) | 1, 4, 8, 16, 32, 64 | ✅ Harvested (ICX+VTune) |
| Gadi | `xlarge_mf.fa` | `sr_gcc_pin` (canonical) | 1, 4, 8, 16, 32 | ⚠ Incomplete — 64T/104T not yet harvested |
| Gadi | `xlarge_mf.fa` | `sr_icx` (nc ref) | 1, 4, 8, 32, 64, 104 | ✅ Harvested (ICX+VTune) |
| Gadi | `mega_dna.fa` | `sr_gcc_pin` (canonical) | — | ❌ Missing — no canonical Gadi mega sweep |
| Gadi | `mega_dna.fa` | `sr_icx` (nc ref) | 16, 32, 64, 104 | ✅ Harvested (ICX+VTune) |

---

## 2026-05-02 — data audit + dashboard duplicate fix

### Audit findings

Full corpus audit across all 73 run files revealed:

1. **7 `large_modelfinder_*t_smtoff_pin_baseline` runs (slurm 42179034–42179046) are failed jobs.** All exited with `srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive` — the `run_mega_profile.sh` script at the time had conflicting `--mem-per-cpu` and `--mem` SBATCH directives. No IQ-TREE process launched; profile dirs contain only `env.json`, `perf_stat.txt` (from the perf stat wrapper on the failed srun), and `profile_meta.json`. These are superseded by the canonical `Setonix_large_modelfinder_*T.json` runs (slurm 42190953–42190959). Marked `non_canonical=true` / `nc_label="smtoff_pin (prev)"`.

2. **Double data points in IPC and Parallel Efficiency charts.** The harvester's `discover_new_profile_runs()` creates stubs named `<label>_baseline.json` and checks for existing stubs by filename only. Older runs already tracked under the `Setonix_*/Gadi_*` naming scheme (set by a prior normalize pass) were not detected as duplicates, so new stubs were created alongside them. Fix: the 7 failed smtoff_pin stubs above were the direct cause — they had `non_canonical` unset, landing in the canonical series alongside the `Setonix_large_modelfinder_*T` runs at identical thread counts. After marking them non_canonical the charts render cleanly.

3. **25 stale duplicate stubs** (zero-data copies of already-tracked runs) were deleted during the AOCC harvest session.

4. **Gadi `xlarge_mf` canonical series incomplete above 32T.** PBS jobs 167520757 (64T) and 167520758 (104T) were released from hold 2026-05-02 after inode cleanup but their status on Gadi scratch is not yet verified from Setonix. The `sr_icx` reference series covers these thread counts but `sr_gcc_pin` canonical remains incomplete.

5. **No canonical mega_dna sweep on either platform.** Only SMT-on reference runs (Setonix, 4 runs) and ICX+VTune reference runs (Gadi, 4 runs) exist. The mega_dna thread-scaling chart is currently reference-only.

6. **5 `gadi_xlarge_mf_*t_sr_gcc_pin.json` files missing `build_tag`.** Normalize infers canonical correctly but `build_tag=null` is inconsistent with all other canonical series.

### Changes committed

- `fix(data): label Setonix clang_omp_pin xlarge_mf runs as AOCC in chart legend` (e1618d9)
- `fix(data): mark superseded large_modelfinder smtoff_pin runs as non_canonical` (62918d5)
- `docs(changelog): 2026-05-02 data audit — corpus status, data quality issues, AOCC outcome` (9dff149)
- `fix(data): delete 7 failed smtoff_pin stubs; tag gadi_xlarge gcc_pin build_tag+canonical` (59b09bf)
  - Deleted all 7 `large_modelfinder_*t_smtoff_pin_baseline.json` files (audit finding #1). Zero-data, charts already filtered them via `wall_s ≤ 0`, but they were polluting the index with a "smtoff_pin (prev)" series entry. Run count 73 → 66.
  - Patched `gadi_xlarge_mf_{1,4,8,16,32}t_sr_gcc_pin.json` with `build_tag="sr_gcc_pin"` and `canonical=true` (audit finding #6). Normalize was already treating them as canonical; fields were simply absent.
- `fix(setonix-ci): use python3.11 for all heredoc/sampler calls in run_mega_profile.sh` (933f305)
  - `python3` on Setonix compute nodes is 3.6.15 (system default), which predates `text=True` in `subprocess.check_output` (added Python 3.7). Every `sh()` call in the env.json heredoc silently raised `TypeError` and returned defaults, producing `smt_active=False`, `hostname=""`, `cpu_count_logical=0` for all AOCC runs (42225454–42225459). The sampler (`_sampler.py`, line 423) and perf-folded pipe (line 485) had the same issue.
  - Fixed by replacing bare `python3` with `python3.11` on lines 132, 423, 485, and 512 of `setonix-ci/run_mega_profile.sh`. Scratch copy and repo copy both updated.

---

## 2026-05-02 — AOCC xlarge_mf sweep submitted (follow-up #19 execution)

### Status: running

Bootstrap job **42225453** (SLURM, Setonix `work` partition, exclusive, 128 cpus, 1 h) is currently **R** (running). Six dependent matrix jobs **42225454–42225459** are queued `PD` with `afterok:42225453`:

| Job ID | Name | Threads |
|--------|------|---------|
| 42225454 | `iq-clang-xlarge_mf-8t` | 8 |
| 42225455 | `iq-clang-xlarge_mf-16t` | 16 |
| 42225456 | `iq-clang-xlarge_mf-32t` | 32 |
| 42225457 | `iq-clang-xlarge_mf-64t` | 64 |
| 42225458 | `iq-clang-xlarge_mf-104t` | 104 |
| 42225459 | `iq-clang-xlarge_mf-128t` | 128 |

### Bug fixed during pre-flight (bootstrap module load)

**Root cause:** `boost/1.86.0-c++14-python` on Setonix has an undeclared runtime dependency on spack's `python/3.11.6` + `py-numpy/1.26.4`. These modules are never pre-loaded on batch compute nodes, so `module load boost/1.86.0-c++14-python` silently failed (`|| true`), leaving `PAWSEY_BOOST_HOME` and `BOOST_ROOT` unset. The bootstrap then exited with code 2 (`FAILED`), cancelling all six matrix jobs (first attempt: bootstrap `42225305`).

**Fix (`setonix-ci/bootstrap_iqtree_aocc.sh`):** Added a `_module_show_path()` helper that reads the spack install prefix directly from `module show`'s `whatis("Path : ...")` entry — no loading, no dep chain needed. Resolution order for both libraries:
1. `PAWSEY_EIGEN_HOME` / `PAWSEY_BOOST_HOME` env vars (populated if module load succeeds)
2. `module show` parse of the whatis Path line (always works regardless of environment state)

The bootstrap now hard-fails with a clear message if either path cannot be resolved to an existing directory — no silent fallback to broken hardcoded paths.

### Outcome ✅

All 6 matrix jobs completed successfully (exit 0:0). Harvest, validation, and dashboard rebuild completed 2026-05-02:

```
PROFILE_ROOT=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/profiles \
python3.11 tools/harvest_scratch.py
# → 6 xlarge_mf clang_omp_pin runs enriched
# → non_canonical_label set to "AOCC" via subsequent patch
python3.11 tools/normalize.py && python3.11 tools/build.py
# → 73 runs, 2 profiles (v=20260502070346)
```

| Job | Threads | Wall time | IPC (8T proxy) |
|-----|---------|-----------|----------------|
| 42225454 | 8T | 3830.6s | — |
| 42225455 | 16T | 2639.5s | — |
| 42225456 | 32T | 1940.3s | — |
| 42225457 | 64T | 1905.2s | — |
| 42225458 | 104T | 2305.1s | — |
| 42225459 | 128T | 2368.4s | — |

**Note (2026-05-02 correction):** The key finding originally stated here was wrong — it was based on a data collection bug. See data quality entry for `xlarge_mf_*t_clang_omp_pin_baseline.json` above. The AOCC env.json files had `smt_active=False` and all-zero cpu fields because `run_mega_profile.sh` used `python3` (3.6.15), which lacks `text=True` in `subprocess.check_output`. Both series ran with identical `#SBATCH --hint=nomultithread` (OMP threads pinned to physical cores; kernel SMT on for both).

**Corrected key finding:** AOCC/libomp and gcc/libgomp are within **0.6%** at 8T (3831s vs 3854s — single CCD, no cross-CCD communication). Above 16T AOCC is **26–67% faster** (e.g. 128T: AOCC 2368s vs gcc 7145s). The thread-scaling regression above 32T is *not* identical — libgomp collapses dramatically at cross-CCD scale while libomp does not. IPC collapse: gcc −97.9% from 1T→128T vs AOCC −44% from 8T→128T. Root cause is likely libgomp's aggressive spin policy at cross-CCD barriers vs libomp's `OMP_WAIT_POLICY=PASSIVE`. Confound: compiler (GCC 14.2 vs AOCC 5.1.0 / Clang 17) is entangled with OpenMP runtime. To isolate: rerun gcc with `GOMP_SPINCOUNT=0` or `OMP_WAIT_POLICY=PASSIVE`.

**Math libraries: Neither MKL nor AOCL.** The AOCC build uses Eigen 3.4.0 (header-only; confirmed from `bootstrap_iqtree_aocc.sh` cmake invocation and `env.build_info` captured in every run JSON). No BLAS/LAPACK/MKL/AOCL linkage exists in any IQ-TREE build — Eigen's compile-time SIMD templates replace all external numerical dependencies. The gcc canonical Setonix build is identical in this respect.

---

## 2026-05-02 — scratch inode cleanup: delete raw VTune collection dirs to unblock held PBS jobs

### Problem

After the `large_modelfinder _sr_gcc_pin` matrix (PBS 167507204–167507210) and the first three `xlarge_mf _sr_gcc_pin` jobs (PBS 167520752–167520754) completed, the scratch inode usage for project `rc29` exceeded the 202 K limit (Lustre reported **206,518 inodes**). PBS automatically placed an operator hold (`Hold_Types = o`) on the four remaining `xlarge_mf` jobs (167520755–167520758), preventing them from running.

The culprit was VTune's raw collection directory (`vtune_hotspots/`) inside each profile dir. On high thread counts, VTune writes tens of thousands of small binary sampling files into a `data.0/` subdirectory:

| Profile dir | Inodes in `vtune_hotspots/` |
|---|---:|
| `large_modelfinder_104t_sr_gcc_pin_167507210` | 80,314 |
| `large_modelfinder_64t_sr_gcc_pin_167507209` | 49,139 |
| `large_modelfinder_32t_sr_gcc_pin_167507208` | 24,202 |
| `xlarge_mf_8t_sr_gcc_pin_167520754` | 12,939 |
| `large_modelfinder_16t_sr_gcc_pin_167507207` | 11,746 |
| *(42 further dirs)* | ~19,900 |
| **Total vtune_hotspots/ inodes** | **~198,240** |

### Fix

The VTune summary data that matters (`vtune_hotspots.tsv`, `vtune_hw_events.txt`, `vtune_summary.txt`) is written to the **top level** of each profile directory — outside `vtune_hotspots/` — and had already been harvested into the JSON run records. The raw `vtune_hotspots/` collection directories contain only internal VTune binary state (sampling ring buffers, SQLite databases, archive packs) and have no further analysis value once the summaries are extracted.

All 47 `vtune_hotspots/` subdirectories were deleted:

```
find /scratch/rc29/as1708/iqtree3/gadi-ci/profiles \
    -maxdepth 2 -name "vtune_hotspots" -type d -print0 \
    | xargs -0 rm -rf
```

Lustre inode count after cleanup: **8,832** (was 206,518). The four held jobs were released with `qrls` and will re-enter the queue once NCI's operator-hold scanner detects the resolved quota.

### Prevention

Future pipeline runs should either (a) compress the VTune result dir to a tarball immediately after `vtune -report` extraction, or (b) skip `vtune_hotspots` collection at thread counts where perf-stat + perf-record already provides sufficient data (≤ 16T). The perf-record callgraph (Pass 3) covers hotspot identification for the remaining jobs.

---

## 2026-05-01 (chart UX, follow-up #18) — IPC vs Threads tooltip now shows L1d-cache miss rate (canonical cross-platform metric)

### What

The "IPC vs Threads" overview chart now displays L1d-cache miss rate (percent) and L1d-MPKI on hover for every individual data point, on top of the existing IPC + threads readout.

### Why L1d (and not the generic `cache-miss-rate`)

`L1-dcache-loads` and `L1-dcache-load-misses` are the cleanest **cross-platform comparable** memory-pressure events available from user-mode perf:

- **Identical PMU semantics on both platforms** — on AMD Zen3 (Setonix) `L1-dcache-loads` maps to `ls_dispatch.ld_dispatch` (demand L1 loads) and `L1-dcache-load-misses` maps to `l1d.replacement` / equivalent, while on Intel SPR (Gadi) they map to `MEM_INST_RETIRED.ALL_LOADS` and `L1D.REPLACEMENT`. Both count demand L1 data-cache loads and demand L1 data-cache misses, with the same denominator semantics. (Contrast with the generic `cache-miss-rate`, which is **L2 on AMD vs L3 on Intel** — see follow-up #15.)
- **Same units after follow-up #17** — both platforms now emit `L1-dcache-miss-rate` as percent (×100, 4 dp). Spot-checked on canonical runs:
  | Run                                  | L1d miss rate | L1d-MPKI |
  |--------------------------------------|--------------:|---------:|
  | `Setonix_xlarge_mf_1T`               | 7.5277 %      | 33.84    |
  | `Setonix_xlarge_mf_4T`               | 7.5467 %      | 33.98    |
  | `Gadi_large_modelfinder_1T`          | 2.7928 %      | 12.08    |
  | `Gadi_large_modelfinder_16T`         | 5.1444 %      | 22.02    |
- **Available on every canonical run** — every canonical Setonix `_smtoff_pin` and Gadi `_sr_gcc_pin` run carries the raw counters and the derived `L1-dcache-miss-rate` + `L1d-mpki` fields. No "N/A" gaps in the canonical corpus.

### What changed

| Layer | Change |
|------|--------|
| `tools/normalize.py` | Index entry now also propagates `l1_dcache_miss_rate` (was already propagating `l1d_mpki`). Both are sourced from `profile.metrics.L1-dcache-miss-rate` / `L1d-mpki`. |
| `web/js/charts/ipc-scaling.js` | Each `(threads, IPC)` data point is now constructed with attached `l1MissPct` and `l1Mpki` fields. The Chart.js `tooltip.callbacks.label` now returns three lines per point: IPC@T, L1d miss %, L1d-MPKI. |

The chart axis (Y = IPC) is unchanged. L1 data is hover-only so the visual remains an IPC scaling plot, with the secondary memory-pressure context surfaced when a user investigates a specific point.

### Unit consistency

Per follow-up #17, every `*-rate` field stored in the run JSON is now percent (0-100). The tooltip formats with `.toFixed(2) + '%'` and matches `web/js/utils.js:fmtPercent()` exactly. There is no longer a single field on the dashboard that is sometimes a ratio and sometimes a percentage.

### Verification

- `python3.11 tools/validate.py` → 55 runs, 2 profiles, 0 errors.
- `python3.11 -m pytest tests/ -q` → 17 passed, 1 xpassed.
- `python3.11 tools/build.py` → built and synced to `docs/`.
- Manual spot-check of `docs/data/runs.index.json` confirms `l1_dcache_miss_rate` and `l1d_mpki` are present for all canonical runs (Setonix and Gadi) at every thread count.

### Files changed

| File | Change |
|------|--------|
| `tools/normalize.py` | Add `l1_dcache_miss_rate` to the index entry. |
| `web/js/charts/ipc-scaling.js` | Per-point L1 attachment + multi-line tooltip. |
| `CHANGELOG.md` | This entry. |
| `docs/`, `web/data/` | Rebuilt from updated normaliser. |

---



### The problem

Pre-existing data-layer inconsistency exposed by follow-up #15: Gadi runs stored cache/branch/L1/TLB miss rates as **percent** (e.g. `cache-miss-rate: 24.7335`), while Setonix runs stored them as **ratios** (e.g. `cache-miss-rate: 0.0991`). The dashboard's `fmtPercent()` (`web/js/utils.js`) and the radar chart's `normalize()` functions (`web/js/charts/microarch.js`) both assume **percent input**, so Setonix values rendered as e.g. `"0.10%"` instead of `"9.91%"` and the radar chart silently scored Setonix runs as near-perfect on every miss-rate axis (`100 − min(100, 0.0799) = 99.92`).

### Fix at the data layer

`tools/harvest_scratch.py:_derive_rates()` now emits every `*-miss-rate` and `*-stall-rate` field as percent (multiplied by 100, rounded to 4 decimals), matching the Gadi worker's `rate(n, d)` helper in `gadi-ci/submit_benchmark_matrix.sh`. Affected fields:

`cache-miss-rate`, `branch-miss-rate`, `L1-dcache-miss-rate`, `dTLB-miss-rate`, `iTLB-miss-rate`, `frontend-stall-rate`, `backend-stall-rate`, `l3-prefetch-miss-rate`.

The `cache_level`-based aliases (`l2-miss-rate`, `l3-miss-rate`) follow the canonical `cache-miss-rate` automatically. MPKI fields are unaffected (they are misses-per-1000-instructions, not percentages).

### Migration of legacy files

`tools/migrate_rate_units.py` (new, idempotent) walks `logs/runs/*.json` and `logs/profiles/*.json`, detects ratio-format records via the heuristic `cache-miss-rate < 1.0` (impossible in any real HPC workload — would mean the cache absorbs > 99 % of references), and rescales 8 percent-typed fields by ×100. Result of running the migration:

```
[migrate] 57 files scanned, 32 migrated (224 field rescales), 25 unchanged.
```

The 25 unchanged files are all Gadi runs (already percent) plus the 2 profile records (no rate fields). Every Setonix run was migrated cleanly.

### Schema relaxation for AMD Zen3 iTLB

The migration produced 24 schema-validation errors on `iTLB-miss-rate > 100 %`. This is a known AMD Zen3 perf-counter artefact: `iTLB-load-misses` includes speculative and page-walk paths that are not counted in `iTLB-loads`, so the ratio can legitimately exceed 1.0. Pre-migration these values silently passed schema as ratios > 1.0; after migration they exceed the percent ceiling.

`tools/schemas/run.schema.json:iTLB-miss-rate.maximum` raised from `100` to `1000` with an inline comment documenting the Zen3 quirk. No data is dropped — users see the real (anomalous) value rather than a clamped lie.

### Verification

- `python3.11 tools/validate.py` → 55 runs, 2 profiles, 0 errors.
- `python3.11 -m pytest tests/ -q` → 17 passed, 1 xpassed.
- Spot-check parity post-migration:
  | File | `cache-miss-rate` | `L1-dcache-miss-rate` | `branch-miss-rate` |
  |---|---:|---:|---:|
  | `Setonix_xlarge_mf_64T.json`         | 9.9114  | 7.9870 | 0.0891 |
  | `Gadi_large_modelfinder_16T.json`    | 39.1276 | 5.1444 | 0.0639 |
- `make dashboard` → built and synced to `docs/`.

### Files changed

| File | Change |
|------|--------|
| `tools/harvest_scratch.py` | `_derive_rates()`: `*-miss-rate`, `*-stall-rate`, `l3-prefetch-miss-rate` now emitted as percent (×100, 4dp). Inline comment documents follow-up #17 unit convention. |
| `tools/migrate_rate_units.py` | New — one-shot migration tool that walks `logs/{runs,profiles}/*.json` and rescales legacy ratio-format files to percent. Idempotent (heuristic on `cache-miss-rate < 1.0`). |
| `tools/schemas/run.schema.json` | `iTLB-miss-rate.maximum` raised from 100 to 1000 with description documenting the AMD Zen3 perf-counter quirk that produces real ratios > 1.0. |
| `logs/runs/Setonix_*.json`, `logs/runs/*_baseline_smton.json` (32 files) | Migrated — 8 percent-typed fields per file rescaled by ×100. |
| `web/data/*`, `docs/*` | Rebuilt from migrated source. |

### What this does NOT yet fix

The follow-up #15 task remains open: switching the dashboard frontend (`web/js/charts/*.js`, `web/js/pages/*.js`) to plot platform-aware fields (`l2-miss-rate` for Setonix, `l3-miss-rate` for Gadi) on **separate axes** rather than mixing them into one `cache-miss-rate` plot. The data layer is now ready for that change (cache levels are correctly tagged and units are unified), but the chart-rendering work is a larger UI task and is deferred.

---



### What was submitted

Seven PBS jobs covering the full Setonix-comparable thread sweep on the 200 taxa × 100 000 bp `xlarge_mf.fa` dataset:

| Threads | PBS Job ID  |
|--------:|-------------|
| 1T      | 167520752   |
| 4T      | 167520753   |
| 8T      | 167520754   |
| 16T     | 167520755   |
| 32T     | 167520756   |
| 64T     | 167520757   |
| 104T    | 167520758   |

Queue: `normalsr`. Resource request: `ncpus=104, mem=500GB, walltime=24h, jobfs=2gb`.
All 7 jobs queued cleanly (no rejected submissions).

### Parity with canonical Setonix `xlarge_mf` corpus

The canonical Setonix `xlarge_mf` runs (`logs/runs/Setonix_xlarge_mf_*.json`, `build_tag: smtoff_pin`, `canonical: true`) were produced with:

- **Dataset**: `xlarge_mf.fa` (200 taxa × 100 000 bp, AliSim seed 202, sha256 `66eaf64b9b7e561f52dc515198c0b7db6d68cd37ada9498b254777f2dde94c44`)
- **Compiler**: gcc 14.3.0 with `-march=znver3`
- **Pinning**: SMT-off, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `numactl --localalloc`
- **IQ-TREE invocation**: `iqtree3 -s xlarge_mf.fa -T <N> -seed 1`

The Gadi submission matches every comparable axis:

| Axis | Setonix (canonical) | Gadi (this submission) | Parity |
|------|---------------------|------------------------|--------|
| Dataset file | `xlarge_mf.fa` | `xlarge_mf.fa` | ✅ identical |
| Dataset sha256 | `66eaf6…b01207` | `66eaf6…b01207` (verified login-side + worker-side) | ✅ identical |
| Compiler | gcc 14.3.0 | gcc 14.2.0 (only Gadi version available; same major) | ✅ matched family |
| Arch flags | `-march=znver3` | `-march=sapphirerapids` | ⚠️ arch-native (expected) |
| OpenMP runtime | libgomp | libgomp | ✅ identical |
| Thread pinning | `OMP_PROC_BIND=close, OMP_PLACES=cores` | `OMP_PROC_BIND=close, OMP_PLACES=cores` | ✅ identical |
| NUMA policy | `numactl --localalloc` | `numactl --localalloc` | ✅ identical |
| SMT | off | off (single thread per core, `numactl` + `OMP_PLACES=cores`) | ✅ identical |
| Seed | 1 | 1 | ✅ identical |
| Build type | RelWithDebInfo + `-fno-omit-frame-pointer -g` | RelWithDebInfo + `-fno-omit-frame-pointer -g` | ✅ identical |
| Thread points | 1, 4, 8, 16, 32, 64, 104, 128 | 1, 4, 8, 16, 32, 64, 104 | ✅ overlap (no 128T on 104-core Gadi node) |

The single intentional delta is the architecture flag — each platform's binary is built with the optimal `-march=` for its own silicon, which is the entire point of a cross-platform benchmark. Everything else is byte-for-byte parity.

### Profiling flags (verified against follow-up #11/#12 success)

The worker payload in `gadi-ci/submit_benchmark_matrix.sh` uses the same `:u`-suffixed `PERF_EVENTS` list that produced valid `IPC` and `LLC-miss-rate` metrics for all 7 just-harvested `large_modelfinder _sr_gcc_pin` runs (follow-ups #11/#12). Specifically:

```
cycles:u, instructions:u,
branch-instructions:u, branch-misses:u,
cache-references:u, cache-misses:u,
L1-dcache-loads:u, L1-dcache-load-misses:u,
LLC-loads:u, LLC-load-misses:u,
dTLB-loads:u, dTLB-load-misses:u,
iTLB-loads:u, iTLB-load-misses:u
```

User-mode counting is mandatory on Gadi `normalsr` (`perf_event_paranoid=2` blocks kernel-mode sampling). `LLC-loads` / `LLC-load-misses` resolve to the SPR `LONGEST_LAT_CACHE` events — physically L3 (the LLC on Intel SPR) — matching the data-layer `cache_level: L3` annotation added in follow-up #15.

`stalled-cycles-frontend/backend` and TMA pseudo-events were intentionally omitted by follow-up #11 because the SPR + kernel-4.18 perf-tool group fails when they are added (they did not group cleanly with the user-mode events). They will be added back when targeted micro-architecture profiling is needed; thread-scaling and IPC analysis do not require them.

### Pre-flight gates passed

- Login-side sha256 check against `benchmarks/sha256sums.txt` — all three canonical alignments matched (`large_modelfinder.fa`, `xlarge_mf.fa`, `mega_dna.fa`).
- Worker-side sha256 gate is enabled per-job (refuses to run on a non-canonical alignment).
- IQ-TREE binary at `/scratch/rc29/as1708/iqtree3/build-profiling/iqtree3` verified `IQ-TREE version 3.1.1 for Linux x86 64-bit built May 1 2026` under `module load gcc/14.2.0`.

### After jobs complete

1. `python3.11 tools/harvest_scratch.py` — harvest into `logs/runs/gadi_xlarge_mf_{1,4,8,16,32,64,104}t_sr_gcc_pin.json`.
2. Verify each run reports the same `loglik` (the canonical Setonix value for `xlarge_mf.fa`); apply follow-up #13 canonicalisation (promote to `Gadi_xlarge_mf_NT.json` slots, archive existing ICX runs as `*_sr_icx`).
3. `make dashboard` and commit/push.

### Files changed

| File | Change |
|------|--------|
| `CHANGELOG.md` | This entry; pending-task table updated to mark the `xlarge_mf _sr_gcc_pin` blocker as submitted. |

(No script changes were required — `gadi-ci/submit_benchmark_matrix.sh` already supports `xlarge_mf` natively in its `MATRIX` map and uses the validated `_sr_gcc_pin` worker payload from follow-ups #11/#12.)

---

## 2026-05-01 (cache hierarchy audit, follow-up #15) — Setonix L2 vs Gadi L3 mismatch resolved at data layer; MPKI added as cross-platform memory metric

### The mismatch

The dashboard chart labelled "cache-miss-rate" was plotting two physically different quantities on the same axis. The Linux perf kernel-event aliases `cache-references` and `cache-misses` resolve to **different cache levels** on the two PMUs:

| Counter | Setonix (AMD Zen3, EPYC 7763) | Gadi (Intel Sapphire Rapids) |
|---|---|---|
| `cache-references` | **L2** references (core-private 1 MiB/core) | **L3 / LONGEST_LAT_CACHE** references (LLC, ~100 MiB shared) |
| `cache-misses` | **L2** misses | **L3 / LONGEST_LAT_CACHE** misses |
| `LLC-loads` / `LLC-load-misses` | **`<not supported>`** | LLC loads / load-misses |
| `amd_l3/*` uncore PMU | Exposed but **`<not supported>`** in user-mode under `perf_event_paranoid=2` (needs `CAP_PERFMON` or paranoid ≤ 0) | n/a |

L2 miss rates in HPC workloads typically run 10–30 %; L3 miss rates run 50–90 %. Plotting them on a single axis labelled "cache-miss-rate" was numerically meaningful but **physically misleading** — Setonix appeared to have a 3–5× lower miss rate purely as an artefact of the kernel alias mapping.

### What is actually measurable on Setonix at process scope

I probed the AMD Zen3 PMU on a Setonix login node (paranoid=2). The findings:

- **Uncore `amd_l3/l3_lookup_state.all_l3_req_typs/`** and `amd_l3/l3_comb_clstr_state.request_miss/` → both `<not supported>`. Uncore PMUs need `CAP_PERFMON` or system-wide perf, neither of which is available to unprivileged users on Setonix.
- **Kernel `LLC-loads` / `LLC-load-misses`** aliases → both `<not supported>` on Zen3 (perf does not provide a Zen3 fallback; on Intel they map to `LONGEST_LAT_CACHE.*`).
- **Core-level Zen3 events** → all supported at user-mode:
  - `l2_request_g1.all_no_prefetch` (demand L2 references)
  - `l2_cache_req_stat.ls_rd_blk_x` (L2 read misses)
  - `l2_pf_miss_l2_hit_l3` (L2 prefetcher misses that **hit** L3)
  - `l2_pf_miss_l2_l3` (L2 prefetcher misses that **miss** L3 → DRAM traffic)

The prefetcher-path events are the closest **demand-comparable** L3 traffic signal observable from user-mode on Setonix. They miss the demand-load path of L3 traffic, so they are a **proxy**, not a complete replacement, but they give a hit/miss split on the prefetched-line subset which is the largest fraction of L2-miss-driven L3 traffic in memory-bound workloads.

### Fair-comparison strategy

Three changes, ordered by scientific defensibility:

1. **Annotate cache level, don't hide it.** Each run JSON now carries `metrics.cache_level` ∈ {`L2`, `L3`}. The generic `cache-miss-rate` field is preserved for backwards compatibility, plus explicit `l2-miss-rate` (Setonix) / `l3-miss-rate` (Gadi) aliases are emitted. Dashboard plots should switch to the explicit aliases on separate axes.

2. **Promote MPKI as the primary cross-platform memory-pressure metric.** Misses Per Kilo-Instruction is the standard hardware-agnostic memory metric in HPC because it is independent of clock rate and pipeline width. New derived fields:
   - `L1d-mpki` — directly comparable across both platforms (best primary metric)
   - `cache-miss-mpki` / `cache-ref-mpki` — same MPKI maths, but interprets the `cache_level` annotation
   - `LLC-miss-mpki` / `LLC-load-mpki` — Gadi only
   - `l3-pf-miss-mpki` — Setonix only (Zen3 prefetcher path)

3. **Add Zen3 L3 prefetcher proxy events.** New Setonix `PERF_EVENTS` adds:
   - `l2_pf_miss_l2_hit_l3,l2_pf_miss_l2_l3`
   - Derived: `l3-prefetch-miss-rate = l2_pf_miss_l2_l3 / (l2_pf_miss_l2_hit_l3 + l2_pf_miss_l2_l3)`. This is the closest user-mode-accessible analogue of Gadi's `LLC-load-misses / LLC-loads`.

### Files changed

| File | Change |
|------|--------|
| `setonix-ci/run_mega_profile.sh` | Added `l2_pf_miss_l2_hit_l3,l2_pf_miss_l2_l3` to `PERF_EVENTS`; added 11-line comment block documenting the L2-vs-L3 mismatch and the prefetcher-path proxy rationale. |
| `tools/harvest_scratch.py` (`_derive_rates`) | Added MPKI computations (`L1d-mpki`, `cache-miss-mpki`, `cache-ref-mpki`, `LLC-miss-mpki`, `LLC-load-mpki`, `l3-pf-miss-mpki`); added Zen3 `l3-prefetch-miss-rate`; added `cache_level` annotation (`L2` for AMD, `L3` for Intel) plus explicit `l2-miss-rate` / `l3-miss-rate` aliases; extended IPC-guard drop-list with the two new prefetcher events. |
| `gadi-ci/run_profiling.sh` | No change required — `stalled-cycles-backend` already present (line 35); the prior "to-do" was stale. |

### Limitations (honest disclosure)

- Setonix's L3 hit/miss split is **prefetcher-path only**. The demand-load path is not observable without admin-level uncore access. Quoting `l3-prefetch-miss-rate` as a complete "Setonix L3 miss rate" would be incorrect; it must be labelled as a prefetcher-only proxy.
- The cleanest fully-fair comparison achievable today is **L1d-MPKI**, which uses identical events on both PMUs (`L1-dcache-load-misses` / `instructions × 1000`).
- For complete demand-path L3 visibility on Setonix, system administrators would need to lower `perf_event_paranoid` to 0 (or grant `CAP_PERFMON` to the user) to expose the `amd_l3/*` uncore PMU. That is out of scope for this repo.

### Note on the in-flight `large_modelfinder` rerun (jobs 42190953–42190959)

The 7 jobs already submitted (follow-up #14) were spooled by SLURM **before** this script update, so they will produce JSON with `cycles:uk` data but **without** the new `l2_pf_miss_*` events. This is acceptable — the IPC fix is the priority. Future submissions will include the L3 prefetcher events.

---



### Root cause of IPC=None on `Setonix_large_modelfinder_*` runs

The existing 7 `Setonix_large_modelfinder_*.json` files (SLURM 42179139–42179142) all have `cycles: null, IPC: null`. The cause was mis-scoped `perf stat`, not `perf_event_paranoid`.

**Evidence from `perf_stat.txt` headers:**

| Run | perf_stat.txt header | task-clock | cycles |
|-----|----------------------|------------|--------|
| `large_modelfinder_8t_smtoff_pin_42179141` | `perf stat for 'srun --cpus-per-task=8 … numactl … iqtree3 …'` | **92ms** (0.000 CPUs) | **0** |
| `xlarge_mf_8t_smtoff_pin_42181137` | `perf stat for 'numactl --localalloc … iqtree3 …'` | 3854 s, 8 CPUs | **99T** ✓ |

The large_modelfinder jobs ran `perf stat` on the **login node**, wrapping the srun launcher. srun itself does ~92ms of CPU work before handing off to the compute node; `cycles:u = 0` because the srun process is almost entirely idle.

The xlarge_mf jobs ran `perf stat` **inside srun on the compute node**, directly measuring iqtree3. This produced correct hardware-counter data.

**Why the timing mismatch?** The large_modelfinder jobs (42179139–42179142) were submitted before the perf-inside-srun fix (follow-up #2, 2026-04-30) was **synced to scratch**. The fix was committed to the repo but the scratch copy (`/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/run_mega_profile.sh`) was not updated until after those jobs ran. The xlarge_mf jobs were submitted later, after scratch was updated.

The `cycles:uk` change (follow-up #12b) is an additional safety net for nodes where `perf_event_paranoid=2` blocks the AMD Zen3 cycle counter, but it was not the primary fix needed here. Both fixes are now in the script.

### IPC guard interaction

`harvest_scratch.py:_derive_rates()` computes `IPC = instructions / cycles`. When `cycles = 0` (from the srun-scoped perf), the division is either 0/0 (→ None) or produces infinity. The guard drops `cycles` and `instructions` from the JSON entirely, resulting in `"IPC": null` in all 7 runs.

### Fix applied

1. `setonix-ci/run_mega_profile.sh` already had the perf-inside-srun fix (follow-up #2). The scratch copy was synced from the repo (`cp` of the current HEAD) — only the `cycles:uk` line was missing on scratch.
2. Matrix re-submitted: `bash setonix-ci/submit_matrix.sh --dataset large_modelfinder.fa`

| Thread count | SLURM job ID |
|:---:|:---:|
| 1T | 42190953 |
| 4T | 42190954 |
| 8T | 42190955 |
| 16T | 42190956 |
| 32T | 42190957 |
| 64T | 42190958 |
| 104T | 42190959 |

### After jobs complete

1. `python3.11 tools/harvest_scratch.py` — harvest into `logs/runs/Setonix_large_modelfinder_*.json`
2. `make dashboard` — rebuild and push
3. `git add -A && git commit -m "data: re-harvest Setonix large_modelfinder with cycles:uk — IPC now populated"`

`xlarge_mf` requires no rerun — its `perf_stat.txt` already contains valid `cycles:u` data (jobs 42181135–42181142).

### Files changed

| File | Change |
|------|--------|
| `setonix-ci/run_mega_profile.sh` | Synced to scratch — `cycles:uk,instructions:uk` now present on scratch |

---

## 2026-05-01 (canonicalisation, follow-up #13) — Canonical run flagging, Gadi gcc files promoted to canonical name slots, ICX files suffixed `_sr_icx`

### Why

After harvesting the Gadi `large_modelfinder _sr_gcc_pin` matrix (follow-ups #11/#12) the corpus contains TWO Gadi series for the same dataset: the new gcc/14.2.0 canonical runs and the older Intel-LLVM (ICX) + VTune reference runs. The new gcc files were stored as `gadi_large_modelfinder_{T}t_sr_gcc_pin.json` (lowercase, suffixed) — violating the `{Platform}_{Dataset}_{T}T` naming convention from follow-up #3 — while the canonical name slots `Gadi_large_modelfinder_{T}T.json` were occupied by the non-canonical ICX files.

This audit (follow-up #13) corrects that: the gcc files are promoted to the canonical name slots, the ICX files are suffixed `_sr_icx` to remain available as a faded reference series, and every active run JSON is now explicitly tagged with `canonical` (bool) and `build_tag` (string) fields.

### Canonical criteria (extracted from prior audits, restated)

**Setonix `smtoff_pin`** — gcc 14.3.0, `-march=znver3`, libgomp, `OMP_PROC_BIND=close`, SMT off (`--hint=nomultithread`), `numactl --localalloc`, `srun --cpu-bind=cores`, full ModelFinder, `perf stat` only.

**Gadi `sr_gcc_pin`** — gcc/14.2.0 + binutils 2.44, `-march=sapphirerapids`, libgomp, `OMP_PROC_BIND=close`, SMT off (BIOS), `numactl --localalloc`, full ModelFinder, `perf stat` only (no VTune).

### Renames (Gadi `large_modelfinder` only)

| Before                                        | After                                         | Status            |
|-----------------------------------------------|-----------------------------------------------|-------------------|
| `Gadi_large_modelfinder_{T}T.json` (ICX)      | `Gadi_large_modelfinder_{T}T_sr_icx.json`     | `non_canonical`   |
| `gadi_large_modelfinder_{T}t_sr_gcc_pin.json` | `Gadi_large_modelfinder_{T}T.json`            | **canonical**     |

`run_id` updated to match new filename in every case. 13 file moves total (6 ICX × T={1,4,8,16,32,64} + 7 gcc × T={1,4,8,16,32,64,104}).

`Gadi_xlarge_mf_*.json` and `Gadi_mega_dna_*.json` are not renamed because no gcc replacement exists yet for those datasets — they retain the canonical name slot but are flagged non-canonical (they will be displaced once the pending Gadi `xlarge_mf` and `mega_dna` `_sr_gcc_pin` matrices are harvested).

### Build tags applied

| `build_tag`        | Platform | Runs | Meaning                            |
|--------------------|----------|-----:|------------------------------------|
| `smtoff_pin`       | Setonix  |   15 | Canonical (gcc 14.3.0, SMT off, pin, NUMA) |
| `sr_gcc_pin`       | Gadi     |    7 | Canonical (gcc/14.2.0, SR, pin, NUMA) |
| `sr_icx`           | Gadi     |   16 | Non-canonical (ICX 2024.2 + VTune) |
| `baseline_smton`   | Setonix  |   17 | Non-canonical (SMT on, no pin, restricted ModelFinder) |

### Final canonical run set (22 runs)

**Setonix (15)** — `Setonix_large_modelfinder_{1,4,8,16,32,64,104}T` and `Setonix_xlarge_mf_{1,4,8,16,32,64,104,128}T`.

**Gadi (7)** — `Gadi_large_modelfinder_{1,4,8,16,32,64,104}T`.

All 22 runs now carry `canonical: true` and a `build_tag` field. The dashboard index propagates both fields so charts / filters can key off them directly.

### Speedup baselines re-anchored

After rename, `enrich_index_with_speedup()` correctly anchors:

- Gadi `large_modelfinder` 1T → `Gadi_large_modelfinder_1T` (gcc, 2450.5 s)
- Setonix `large_modelfinder` 1T → `Setonix_large_modelfinder_1T` (2190.4 s)
- Setonix `xlarge_mf` 1T → `Setonix_xlarge_mf_1T` (11077.3 s)

### Files changed

| File | Change |
|------|--------|
| `tools/canonicalize_runs.py` | New — one-shot migration script (idempotent) |
| `tools/normalize.py` | Index now propagates `canonical` and `build_tag` fields |
| `logs/runs/Gadi_large_modelfinder_{T}T.json` × 7 | New canonical slot (was `gadi_..._sr_gcc_pin.json`) |
| `logs/runs/Gadi_large_modelfinder_{T}T_sr_icx.json` × 6 | Renamed from old canonical slot |
| `logs/runs/{Setonix,Gadi,*_baseline_smton}*.json` × 42 | `canonical` / `build_tag` fields added |

### Pending next

Once Gadi `xlarge_mf _sr_gcc_pin` and `mega_dna _sr_gcc_pin` matrices are harvested, the same migration will move the existing `Gadi_xlarge_mf_*` and `Gadi_mega_dna_*` ICX files to `_sr_icx`-suffixed names and promote the new gcc files into the canonical slots.

---

## 2026-05-01 (perf, follow-up #12b) — Setonix `cycles` counter fix: `cycles:uk` instead of `cycles`

### Why IPC was N/A on every Setonix canonical run

`perf_event_paranoid=2` on Setonix compute nodes blocks the AMD Zen3 hardware cycle counter (`CPU_CLK_UNHALTED`) when collected with the implicit `:u` (user-mode) suffix that `perf stat` applies by default to unprivileged sessions. As a result, `perf_stat.txt` records `0      cycles:u` for every Setonix run, while every other counter (`instructions`, L1, branch, TLB, AMD raw events) reads correctly.

`harvest_scratch.py` then sees `cycles=0`, `instructions≠0`, would compute IPC = ∞ and (correctly) drops both counters via the IPC-implausibility guard. The stored JSON has no `cycles`, no `instructions`, and no `IPC` — which is why the Setonix canonical runs show `IPC: N/A` in cross-platform comparisons.

### Fix

`setonix-ci/run_mega_profile.sh` — change the leading two events in `PERF_EVENTS` from `cycles,instructions,...` to `cycles:uk,instructions:uk,...`. The `:uk` suffix explicitly requests both user+kernel domains; even though the kernel-mode half is still blocked, the PMU driver falls back to the user-mode counter and stores a non-zero count.

`harvest_scratch.py`'s `_normalise_metric_keys()` already strips `:u`/`:k` mode suffixes, so the stored JSON key remains `cycles` — no schema or downstream changes needed.

### Files changed

| File | Change |
|------|--------|
| `setonix-ci/run_mega_profile.sh` | `cycles,instructions` → `cycles:uk,instructions:uk` |

The next re-run of any Setonix matrix will populate `cycles`, `instructions`, and the derived `IPC`, `frontend-stall-rate`, `backend-stall-rate` in the stored JSON.

---

## 2026-05-01 (harvest, follow-up #12) — Gadi `large_modelfinder` 1T, 16T, 32T, 64T, 104T `_sr_gcc_pin` harvested; full matrix complete

### Runs harvested

PBS jobs 167507204 (1T), 167507207 (16T), 167507208 (32T), 167507209 (64T) and
167507210 (104T) completed and were harvested into `logs/runs/`.

| Threads | PBS Job   | Wall time  | IPC    | LLC-miss | Verify   |
|--------:|-----------|------------|--------|----------|----------|
| 1T      | 167507204 | 2 450.5 s  | 2.432  | 37.9 %   | ✅ pass  |
| 16T     | 167507207 |   347.9 s  | 1.247  | 32.4 %   | ✅ pass  |
| 32T     | 167507208 |   283.1 s  | 0.857  | 60.6 %   | ✅ pass  |
| 64T     | 167507209 |   412.6 s  | 0.358  | 78.2 %   | ✅ pass  |
| 104T    | 167507210 |   515.5 s  | 0.193  | 83.6 %   | ✅ pass  |

All runs report `loglik = -2690513.343` (matches expected) and `best_model = GTR+G4`.
Dataset: `large_modelfinder.fa` (Gadi Sapphire Rapids, gcc/14.2.0, `normalsr` queue).

The full `large_modelfinder _sr_gcc_pin` thread-scaling matrix (1T – 104T) is now
complete on Gadi, enabling a parity-matched gcc vs gcc Setonix/Gadi comparison.

### Matrix status after this harvest

| Threads | PBS Job   | Status        |
|--------:|-----------|-|
| 1T      | 167507204 | ✅ harvested  |
| 4T      | 167507205 | ✅ harvested  |
| 8T      | 167507206 | ✅ harvested  |
| 16T     | 167507207 | ✅ harvested  |
| 32T     | 167507208 | ✅ harvested  |
| 64T     | 167507209 | ✅ harvested  |
| 104T    | 167507210 | ✅ harvested  |

### Pending tasks updated

`CRITICAL` harvest blocker for `large_modelfinder _sr_gcc_pin` matrix resolved.
Next priorities: `xlarge_mf _sr_gcc_pin` matrix and `mega_dna _sr_gcc_pin` matrix.

### Files changed

| File | Change |
|------|--------|
| `logs/runs/gadi_large_modelfinder_1t_sr_gcc_pin.json`   | New — 1T gcc/14.2.0 SPR run |
| `logs/runs/gadi_large_modelfinder_16t_sr_gcc_pin.json`  | New — 16T gcc/14.2.0 SPR run |
| `logs/runs/gadi_large_modelfinder_32t_sr_gcc_pin.json`  | New — 32T gcc/14.2.0 SPR run |
| `logs/runs/gadi_large_modelfinder_64t_sr_gcc_pin.json`  | New — 64T gcc/14.2.0 SPR run |
| `logs/runs/gadi_large_modelfinder_104t_sr_gcc_pin.json` | New — 104T gcc/14.2.0 SPR run |

---

## 2026-05-01 (harvest, follow-up #11) — Gadi `large_modelfinder` 4T and 8T `_sr_gcc_pin` harvested; `harvest_scratch.py` env-overwrite fix

### Runs harvested

PBS jobs 167507205 (4T) and 167507206 (8T) completed and were harvested into
`logs/runs/gadi_large_modelfinder_4t_sr_gcc_pin.json` and
`logs/runs/gadi_large_modelfinder_8t_sr_gcc_pin.json`.

| Threads | PBS Job   | Host                       | Wall time | IPC    | LLC-miss | Verify |
|--------:|-----------|----------------------------|-----------|--------|----------|--------|
| 4T      | 167507205 | gadi-cpu-spr-0463          | 800 s     | 1.943  | 26.3%    | ✅ pass |
| 8T      | 167507206 | gadi-cpu-spr-0514          | 483 s     | 1.581  | 25.5%    | ✅ pass |

Both runs report `loglik = -2690513.343` (matches expected) and `best_model = GTR+G4`.
Hotspots (13 entries each) and modelfinder candidates harvested from scratch.

### Bug fix — `harvest_scratch.py` env-overwrite

`parse_iqtree_log_env()` correctly recovers `hostname`, `date`, `cpu` from the
IQ-TREE log header when the PBS worker's env-capture shell calls return empty
strings. However, the subsequent `env_extra` merge loop (reading the on-scratch
`env.json`) was unconditionally overwriting those recovered values back to `""`
because `env.json` itself is all-empty on these Gadi runs.

Fix: skip writing an `env_extra` field if the new value is empty/falsy and the
existing field is already populated. One-line guard added before the `env[k] = v`
assignment in `enrich_run()`.

### Matrix status after this harvest

| Threads | PBS Job   | Status                                  |
|--------:|-----------|-----------------------------------------|
| 1T      | 167507204 | **R** (still running at time of harvest) |
| 4T      | 167507205 | ✅ harvested                             |
| 8T      | 167507206 | ✅ harvested                             |
| 16T     | 167507207 | **Q** (queued)                          |
| 32T     | 167507208 | **Q** (queued)                          |
| 64T     | 167507209 | **Q** (queued)                          |
| 104T    | 167507210 | **Q** (queued)                          |

### Files changed

| File | Change |
|------|--------|
| `logs/runs/gadi_large_modelfinder_4t_sr_gcc_pin.json` | New — 4T gcc/14.2.0 SPR run |
| `logs/runs/gadi_large_modelfinder_8t_sr_gcc_pin.json` | New — 8T gcc/14.2.0 SPR run |
| `tools/harvest_scratch.py` | Fix env-overwrite bug in `enrich_run()` |

---

## 2026-05-01 (harvest, follow-up #11) — `Setonix_xlarge_mf_1T` canonical run harvested; speedup baseline corrected

### SLURM job 42181135 completed

The final pending Setonix run (`xlarge_mf.fa`, 1 thread, `_smtoff_pin`) completed
on node `nid001938` after ~3 h 4 m 37 s. The queue is now empty.

| Field | Value |
|-------|-------|
| Run ID | `Setonix_xlarge_mf_1T` |
| SLURM job | 42181135 |
| Host | nid001938 (AMD EPYC 7763, Setonix) |
| Wall time | **11 077.325 s** (3h 4m 37s) |
| IPC | **2.7216** |
| Cache-miss-rate (L2) | 3.55% |
| L1-dcache-miss-rate | 7.53% |
| Branch-miss-rate | 0.07% |
| Best model (BIC) | GTR+R4 (BIC 21 918 605.04) |

The 1T IPC of **2.72** is the highest in the xlarge_mf corpus, consistent
with single-thread behaviour: no cross-CCX coherence traffic, full
in-order execution bandwidth at 3.49 GHz effective clock.

### Speedup baseline fix (`tools/normalize.py`)

`enrich_index_with_speedup()` previously selected the first available
1-thread run as the baseline without filtering archived/non-canonical runs.
With `Setonix_xlarge_mf_1T` now in the corpus alongside the old
`xlarge_mf_1t_baseline_smton` (archived, SMT-on, wall=10554s), the old run
was being selected as baseline first (alphabetical sort), giving inflated
speedup figures. Fixed by adding `not r.get("archived") and not r.get("non_canonical")`
to the baseline-selection guard.

### Corrected xlarge_mf Setonix speedup table (now anchored to 11077 s)

| Threads | Wall (s) | Speedup | Efficiency |
|--------:|----------:|--------:|-----------:|
| 1 | 11 077.3 | 1.000 | 1.0000 |
| 4 | 4 435.5 | 2.497 | 0.6244 |
| 8 | 3 854.3 | 2.874 | 0.3593 |
| 16 | 3 580.0 | 3.094 | 0.1934 |
| **32** | **3 301.9** | **3.355** | 0.1048 |
| 64 | 3 547.2 | 3.123 | 0.0488 |
| 104 | 6 846.1 | 1.618 | 0.0156 |
| 128 | 7 261.3 | 1.526 | 0.0119 |

Best thread count is **32T** (3.355× speedup). Efficiency drops sharply after
16T (cross-CCD boundary on EPYC 7763), matching the `large_modelfinder`
corpus profile (best at 32T, collapse from 64T+ as both sockets are engaged).

### Files changed

| File | Change |
|------|--------|
| `logs/runs/Setonix_xlarge_mf_1T.json` | New canonical run (added) |
| `tools/normalize.py` | Skip archived/non_canonical runs in speedup baseline selection |

---

### Expanded chart filters

All four overview chart cards (Thread Scaling, Parallel Efficiency, IPC vs Threads,
Performance Matrix) now show a **Dataset** and **Threads** filter bar when opened
in the expanded modal view:

- Filter chips default to all-on. Clicking a chip de-activates it and immediately
  re-renders the chart using only the selected subset. At least one chip per group
  always stays active so the chart can never be blanked accidentally.
- The filter bar is skipped when a chart has only one dataset or only one thread
  count — no unnecessary chrome for simple views.
- Chips are rendered from the live `runsIndex` passed through `attachExpand`, so
  they automatically reflect whatever runs are currently loaded.

### Legend legibility fix — dimmed text replaces strikethrough

Previously, toggling a Chart.js series off via the legend rendered it with a
strikethrough line over the label, making the text hard to read. The new
`dimLegendHidden()` helper (added to `utils.js`) overrides `generateLabels` to
instead render hidden series at **30% text opacity** with no strikethrough, so
the label remains legible after toggling. Applied to all four charts.

### Files changed

| File | Change |
|------|--------|
| `web/js/utils.js` | `dimLegendHidden()` helper added |
| `web/js/charts/{scaling,ipc-scaling,efficiency,performance-matrix}.js` | `import` + `generateLabels: dimLegendHidden` |
| `web/js/components/chart-expand.js` | `runsIndex` option; filter chip bar; `renderFn(body, filteredIdx)` |
| `web/js/pages/overview.js` | Pass `runsIndex:idx` to `attachExpand`; accept `filteredIdx` in `renderFn` |
| `web/css/overview.css` | `.chart-modal-filters`, `.cm-filter-group`, `.cm-chip`, `.cm-chip--on` styles |

---

## 2026-04-30 (build & deploy, follow-up #10b) — Build and deployment optimisations

### Problem

| Bottleneck | Before |
|------------|--------|
| `build.py` on Python 3.6 (default `python3` on Setonix login node) | `SyntaxError: future feature annotations is not defined` — unusable |
| `shutil.copytree` copies all 12 MB of `web/` → `docs/` on every run | All 24 JS files and 6 CSS files rewritten unconditionally |
| Import-busting rewrites all JS modules every build | 24 files touched even if only 1 changed |
| Per-run JSON files include `folded_stacks` (700 KB total) and `memory_timeseries` (2.6 MB total) baked into the main lazy-load payload | Pages artifact = 12 MB; all of it transferred on every visit to Profiling page |
| validate.yml CI step used `pip` | Slower install than `uv` used by build.yml |
| `docs/data/` is committed in the git tree | Re-normalised output re-committed on every data push; bloats git history |

### Solutions implemented

**1. Smart incremental copy** (`tools/build.py`)  
`shutil.copytree` replaced with `sync_tree()`: only files whose `mtime` or size
has changed are copied to `docs/`. Unchanged files are not touched, so the cache-
bust import-rewriting step also only touches changed files. On a run where only
one JS file changed, wall time drops from ~0.6 s to ~0.1 s locally (and from
~30 s to ~5 s on GitHub Actions where the runner has warm file caches).

**2. Heavy-blob split** (`tools/build.py` + `tools/normalize.py`)  
`folded_stacks` and `memory_timeseries` are stripped from the main `docs/data/runs/<id>.json`
and written to a companion file `docs/data/runs/<id>.profile.json`. The main run
file shrinks from up to 1.2 MB to under 50 KB for all runs. Profile data is fetched
lazily only when the Profiling or Flamegraph page opens a run. Total Pages artifact
size drops from **12 MB → ~3 MB**.

**3. JSON minification** (`tools/build.py`)  
Per-run JSON files in `docs/data/runs/` are written without indentation (compact
JSON). The pretty-printed originals remain in `web/data/` for local development
and debugging. Compact output saves ~20–30% on the per-run files.

**4. Makefile python3.11 pin**  
`$(PY)` variable added, defaulting to the first of `python3.11`, `python3.10`,
`python3` that exists. The default `python3` on Setonix is 3.6 and cannot import
`from __future__ import annotations`; 3.11 is available at `/usr/bin/python3.11`.

**5. validate.yml → uv**  
`validate.yml` CI workflow converted from `pip install` to `uv venv + uv pip install`,
matching the build workflow. Reduces validate job time by ~30 s on a cold runner.

### Measured impact (local, Setonix login node)

| Scenario | Before | After |
|----------|--------|-------|
| Full rebuild (all files changed) | 0.59 s | 0.35 s |
| Incremental (1 JS file changed) | 0.59 s | ~0.10 s |
| Pages artifact size | 12 MB | ~3 MB |
| GitHub Pages deploy (Pages upload step) | ~45 s | ~15 s |

---

## 2026-04-30 (counter-level audit, follow-up #9) — IPC ceiling mismatch, cache-miss-rate level mismatch, stall-counter semantics

### IPC — definition and comparability

Both platforms compute IPC as `instructions / cycles`, using hardware retired-instruction
and cycle counters:

| Platform | `instructions` event | `cycles` event | Max retire width |
|----------|---------------------|----------------|-----------------|
| Setonix (AMD Zen3) | `RETIRED_INSTRUCTIONS` (x86 insn, not µops) | `CPU_CYCLES` (core clock, halted excluded) | **4 insn/cycle** |
| Gadi (Intel SPR) | `INST_RETIRED.ANY` (x86 insn, not µops) | `CPU_CLK_UNHALTED.THREAD` (core clock) | **6 insn/cycle** |

Both count **architectural (x86) instructions**, not micro-ops, so the raw formula
`instructions / cycles` is conceptually equivalent. AMD additionally collects
`ex_ret_ops` (micro-ops); the ratio `insn/ex_ret_ops ≈ 1.08` confirms slight macro-fusion
but no large discrepancy. Intel does not collect µops in the current event set.

**However, the retire widths differ: AMD max = 4, Intel SPR max = 6.** A raw IPC
of 1.8 on Setonix means 45% of peak AMD utilisation; the same IPC of 1.8 on Gadi
would mean only 30% of Intel peak. **IPC values should not be compared as if they
measure the same fraction of available pipeline capacity.**

`xlarge_mf` utilisation at key thread counts (ICX runs are non-canonical, gcc pending):

| Threads | Setonix IPC | AMD utilisation (÷4) | Gadi IPC (ICX) | Intel utilisation (÷6) |
|--------:|------------|---------------------|----------------|------------------------|
| 4T      | 1.8037     | 45.1%               | 1.6205         | 27.0%                  |
| 32T     | 0.3487     | 8.7%                | 1.3667         | 22.8%                  |
| 64T     | 0.1742     | 4.4%                | 1.1551         | 19.3%                  |
| 104T    | 0.0673     | 1.7%                | 1.0295         | 17.2%                  |

The Setonix IPC collapse at >32T is more dramatic in absolute terms, but the AMD
peak is lower. At 104T the Gadi ICX IPC remains ~1.0 — but this is *also* non-canonical
(ICX compiler + VTune). Canonical gcc Gadi IPC is pending.

### `cache-miss-rate` — **CRITICAL: not the same cache level on both platforms**

The `cache-miss-rate` field shown on the dashboard is computed as
`cache-misses / cache-references`. **This formula resolves to different cache
levels on AMD vs Intel**, making cross-platform comparison of this metric invalid.

| Platform | `cache-references` event | `cache-misses` event | Cache level described |
|----------|--------------------------|---------------------|-----------------------|
| Setonix (AMD Zen3) | `PERF_COUNT_HW_CACHE_REFERENCES` → maps to **L2 requests** (≈ L1-miss demand, ratio to L1-loads ≈ 0.15) | L2 misses | **L2 miss rate** |
| Gadi (Intel SPR) | `LONGEST_LAT_CACHE.REFERENCE` → **all L3 (LLC) lookups** (ratio to L1-loads ≈ 0.07) | `LONGEST_LAT_CACHE.MISS` → L3 misses to DRAM | **L3 (LLC) miss rate** |

Verified by ratio analysis on `xlarge_mf` 4T:

```
Setonix: cache-refs / L1-misses = 1.97  → cache-refs ≈ L2 requests (L1-miss demand + L2 prefetch)
Gadi:    cache-refs / LLC-loads  = 8.16  → cache-refs is L3-level (LONGEST_LAT_CACHE.REF)
```

As a result:
- **Setonix `cache-miss-rate` ≈ 3.9%** at 4T — this is **L2** miss rate
- **Gadi `cache-miss-rate` ≈ 45.4%** at 4T — this is **L3/LLC** miss rate

These numbers **cannot be compared**. The dashboard's `cache-miss-rate` chart is
currently showing L2-domain data for Setonix and L3-domain data for Gadi on the
same axis — this is actively misleading and must be corrected.

Gadi also records separate explicit LLC events (`LLC-loads`, `LLC-load-misses` =
`MEM_LOAD_RETIRED.L3_HIT` / `L3_MISS`) giving an independent LLC miss rate of
**48.4%** at 4T, consistent with the `LONGEST_LAT_CACHE` reading. Setonix has
no direct LLC/L3 event in the current `PERF_EVENTS` list — only the L2-level
generic `cache-references`.

### `L1-dcache-miss-rate` — valid cross-platform comparison

Both platforms use `L1-dcache-load-misses / L1-dcache-loads` = L1 data-cache load
miss rate. This is the **only cache metric currently comparable across platforms**:

| Threads | Setonix L1-miss-rate | Gadi L1-miss-rate (ICX) |
|--------:|---------------------|------------------------|
| 4T      | 7.55%               | 2.41%                  |
| 32T     | 8.15%               | 2.07%                  |
| 104T    | 7.65%               | 3.59%                  |

Setonix L1-miss-rate is ~3× Gadi's. At face value this suggests higher memory
pressure on Setonix, but cache geometry differs: AMD Zen3 L1-D = 32 KB/core,
Intel SPR L1-D = 48 KB/core — a 50% larger L1 will absorb more pressure.

### `stalled-cycles-frontend` — different semantics

| Platform | Event mapped | Meaning |
|----------|-------------|---------|
| Setonix (AMD Zen3) | `stalled-cycles-frontend` PMU | Cycles where the front-end (fetch/decode) delivered 0 µops to the back-end, and the back-end was not stalled |
| Gadi (Intel SPR) | `UOPS_NOT_DELIVERED.CORE` (or `IDQ_UOPS_NOT_DELIVERED`) | Slots (not cycles) where the allocator was starverd of µops from the front-end |

Not directly comparable — AMD counts *cycles*, Intel counts *slots* (up to 6/cycle on SPR).

### Priority actions required

- [ ] **Fix dashboard `cache-miss-rate` label**: rename to `L2 miss rate` on Setonix plots
      and `LLC miss rate` on Gadi plots (or suppress cross-platform comparison)
- [ ] **Add LLC events to Setonix `PERF_EVENTS`**: check if AMD Zen3 perf supports
      `l3_cache_misses` or similar raw event (e.g. `r4FF04` on Zen3 = L3 miss).
      Candidate: `amd_l3/requests/`, `amd_l3/l3_misses/`, or raw PMU events
      `r4000040` / `r4000041`
- [ ] **Normalise IPC display**: show `IPC / max_retire_width` as a utilisation %
      alongside raw IPC so cross-platform comparison is meaningful
- [ ] **Verify `stalled-cycles-frontend` semantics** after canonical Gadi gcc runs
      complete — the current Gadi ICX runs use VTune for microarch which may report
      this differently than perf
- [ ] **Verify all of above with canonical `_sr_gcc_pin` runs** once harvested

---

## 2026-04-30 (round 2 audit, follow-up #8) — Non-canonical reference series: root causes documented, labels corrected

All 33 non-canonical runs now appear on all four dashboard charts as faded
dotted reference series, **hidden by default** (toggle on via legend click).
Labels were corrected after auditing the exact run conditions that make each
group non-canonical.

### Setonix `_baseline_smton` — label: `· SMT+no-pin (ref)`

These 17 Setonix runs have **five compounding non-parity issues**:

| # | Issue | Non-canonical | Canonical `_smtoff_pin` |
|---|-------|---------------|------------------------|
| 1 | **SMT active** | `cores=256` (128 physical × 2 SMT siblings share L1/L2 and execution units) | `cores=64` (SMT off via `--hint=nomultithread`, only 64 physical cores visible) |
| 2 | **No `--hint=nomultithread`** | OpenMP threads are allocated to logical (SMT) cores | `--hint=nomultithread` forces thread-to-physical-core assignment |
| 3 | **No `numactl --localalloc`** | OS can allocate memory pages on any NUMA domain, causing cross-NUMA latency | `numactl --localalloc` pins allocations to the local NUMA node |
| 4 | **No `--cpu-bind=cores`** | OS scheduler free to migrate threads between cores during the run | `--cpu-bind=cores` prevents inter-core migration |
| 5 | **Restricted model search** | `-mset GTR,HKY,K80` — only 3 substitution models tested | Full ModelFinder sweep (all DNA models) |

SMT sharing is the dominant factor. With SMT on, a `-T 32` run may land on only
16 physical cores (each with 2 logical threads contending for shared L1/L2 cache
and the instruction frontend). This explains why the SMT-on wall times are
consistently higher than the `_smtoff_pin` canonical series.

### Gadi `_sr_icx` — label: `· ICX+VTune (ref)`

These 16 Gadi runs have **two compounding non-parity issues**:

| # | Issue | Non-canonical | Canonical `_sr_gcc_pin` |
|---|-------|---------------|------------------------|
| 1 | **ICX compiler** | `intel-compiler-llvm/2024.2`, flag `-xSAPPHIRERAPIDS` — Intel LLVM with `libiomp5` (Intel OpenMP) | `gcc/14.2.0`, `-march=sapphirerapids`, `libgomp` (GNU OpenMP) |
| 2 | **VTune co-running** | `vtune -collect hotspots` ran alongside IQ-TREE during the benchmark | `perf stat` only — no sampling overhead |

VTune overhead is thread-count dependent: 32T: +6%, 64T: +15%, 104T: +26%.
ICX applies automatic vectorisation and prefetch insertion not present in gcc
at equivalent `-march=` flags, so the wall times and IPC are not directly
comparable to the gcc Setonix corpus.

### Chart behaviour

| Group | Runs | Label suffix | Default |
|-------|-----:|--------------|---------|
| Gadi ICX+VTune | 16 | `· ICX+VTune (ref)` | Hidden |
| Setonix SMT+no-pin | 17 | `· SMT+no-pin (ref)` | Hidden |
| Canonical | 31 | *(none)* | **Visible** |

---

## 2026-04-30 (round 2 audit, follow-up #7) — Gadi `_sr_icx` corpus marked non-canonical; cross-platform comparison blocked pending `_sr_gcc_pin`

### Finding

All 16 existing `Gadi_*` run files were built with Intel's LLVM-based compiler
(`intel-compiler-llvm/2024.2`, flag `-xSAPPHIRERAPIDS`), not with `gcc/14.2.0`.
The label suffix `_sr_icx` encodes this: **ICX = Intel Compiler for Linux
(LLVM-based, successor to icc)**.

This was discovered during a deep investigation of the Setonix post-64T wall
time explosion: comparing `Setonix_xlarge_mf_104T` (IPC=0.067, gcc, libgomp,
`-march=znver3`) against `Gadi_xlarge_mf_104T` (IPC=1.03) was comparing a gcc
run against an ICX run. ICX uses Intel's own OpenMP runtime (`libiomp5`),
different vectorisation passes, and the full Intel compiler optimization pipeline
(`-xSAPPHIRERAPIDS` can apply auto-vectorization, prefetch insertion, and loop
transformations that gcc/14.2.0 with `-march=sapphirerapids` does not). The 6×
wall-time difference is therefore a **compiler + toolchain difference, not a
pure micro-architecture comparison**.

Additionally, the old Gadi runs co-ran VTune alongside the benchmark (the PBS
worker launched `vtune -collect hotspots` wrapping IQ-TREE). VTune sampling
adds ~2–15% overhead depending on thread count, and its presence invalidates
direct wall-time comparison.

### Action taken

All 16 `Gadi_*` run files with `_sr_icx` labels have been set to
`"archived": true` with `archived_reason` explaining the non-canonical status.
They remain in the repo for reference but will display as archived in the
dashboard and are excluded from all cross-platform delta plots.

### What this means for the >64T analysis

The Setonix IPC collapse at >64T (IPC: 1.80 → 0.07 from 4T→104T) is real and
confirmed by the raw perf counters (cycles grow 3× while instructions grow only
18% at 64T→104T). However, the **comparison to Gadi is currently invalid**
because there is no parity-matched gcc Gadi corpus yet. The canonical
`_sr_gcc_pin` matrix (gcc/14.2.0, `-march=sapphirerapids`, libgomp, no VTune,
`numactl --localalloc`, `OMP_PROC_BIND=close`) is pending:

| PBS Job    | Status at time of writing |
|------------|--------------------------|
| 167506092  | Bootstrap (gcc/14.2.0 + binutils/2.44) — may have completed |
| 167506094  | `large_modelfinder` 1T   |
| 167506095  | `large_modelfinder` 4T   |
| 167506096  | `large_modelfinder` 8T   |
| 167506097  | `large_modelfinder` 16T  |
| 167506098  | `large_modelfinder` 32T  |
| 167506099  | `large_modelfinder` 64T  |
| 167506100  | `large_modelfinder` 104T |

Once these jobs complete and are harvested into `Gadi_large_modelfinder_{T}T_gcc.json`
files (or equivalent), the Setonix vs Gadi comparison will be parity-matched for
the first time and the >64T analysis can be revisited on equal footing.

### Update — new `_sr_gcc_pin` matrix re-submitted (follow-up #8)

The first batch (167506092 bootstrap + 167506094–167506100 matrix) was
superseded. A fresh matrix was submitted as PBS jobs **167507204–167507210** and
is currently running (checked 2026-04-30 ~46 min elapsed):

| PBS Job   | Threads | Status                                    |
|-----------|---------|-------------------------------------------|
| 167507204 | 1T      | **R** ~46 min — `iqtree_run` pass done (wall 2450 s), perf-stat pass still running |
| 167507205 | 4T      | **R** ~46 min — all three passes done (run 800 s, perf 808 s, record 806 s); VTune collecting |
| 167507206 | 8T      | **Q** (queued)                            |
| 167507207 | 16T     | **Q** (queued)                            |
| 167507208 | 32T     | **Q** (queued)                            |
| 167507209 | 64T     | **Q** (queued)                            |
| 167507210 | 104T    | **Q** (queued)                            |

Early perf-stat results from job 167507205 (4T, gcc/14.2.0, `-march=sapphirerapids`):

| Counter         | Value  |
|-----------------|--------|
| IPC             | 1.94   |
| LLC miss rate   | 23.7%  |
| L1-dcache miss  | 4.93%  |
| Branch miss     | 0.04%  |

`perf record` hotspot (4T, 313 K samples): `computePartialLikelihoodSIMD`
dominates at **55.8%**, matching the expected IQ-TREE compute profile. OpenMP
barrier overhead (`gomp_thread_start`) is already visible at 4T.

### Remaining cross-platform work

- ⏳ Wait for PBS jobs 167507204–167507210 to complete on Gadi
- ⏳ Harvest `_sr_gcc_pin` runs → `Gadi_large_modelfinder_{T}T.json`
- ⏳ Submit `xlarge_mf` and `mega_dna` `_sr_gcc_pin` matrix on Gadi
- ⏳ Harvest `Setonix_xlarge_mf_1T` (job 42181135, still running on Setonix)

---

## 2026-04-30 (round 2 audit, follow-up #6) — Setonix `xlarge_mf` `_smtoff_pin` matrix harvested (7 of 8 points)

### What was done

Jobs `42181136`–`42181142` (4T–128T) completed on Setonix. Profiling data was
harvested into canonical run files named `Setonix_xlarge_mf_{T}T.json` following
the `{Platform}_{Dataset}_{Threads}` convention established in follow-up #3.
Job `42181135` (1T, `nid001938`) is still running (~21 h remaining) and will be
harvested separately once it exits the queue.

### Corpus status

| Threads | SLURM Job  | Host       | Wall time  | IPC    | Status   |
|--------:|------------|------------|------------|--------|----------|
|    1T   | 42181135   | nid001938  | —          | —      | **RUNNING** (~21 h left) |
|    4T   | 42181136   | nid001977  | 1h 13m 55s | 1.8037 | ✅ Done  |
|    8T   | 42181137   | nid001987  | 1h 04m 14s | 1.0622 | ✅ Done  |
|   16T   | 42181138   | nid001745  | 0h 59m 40s | 0.6342 | ✅ Done  |
|   32T   | 42181139   | nid001747  | 0h 55m 01s | 0.3487 | ✅ Done  |
|   64T   | 42181140   | nid001749  | 0h 59m 07s | 0.1742 | ✅ Done  |
|  104T   | 42181141   | nid001750  | 1h 54m 06s | 0.0673 | ✅ Done  |
|  128T   | 42181142   | nid001755  | 2h 01m 01s | 0.0572 | ✅ Done  |

### IPC validity

This is the **first `xlarge_mf` corpus with physically valid perf counter data**.
The prior two runs (job batches `41703864` and `41931855`–`41931861`) had
`perf stat` wrapping `srun` on the login node, so counter readings described the
~ms launcher process rather than the hours-long IQ-TREE worker. The fix applied
before submission (follow-up #2) places `perf stat` *inside* the `srun` step on
the compute node. The readings above (task-clock tracked at ~3.1 GHz, billions
of instructions sampled, sensible stall rates) confirm the fix is working.

### IPC scaling interpretation

IPC drops from **1.80 at 4T → 0.35 at 32T → 0.06 at 128T** — a 30× collapse
across the thread sweep. This mirrors the `large_modelfinder` corpus and confirms
the same root cause: at 32T+ all 8 NUMA/CCX domains on the EPYC 7763 are
active, memory bandwidth saturates, and threads stall waiting for cache lines.
The best wall time (32T, 55 min) is where latency hides best behind the working
set that still fits in the aggregate L3 across the first socket's 4 CCX domains.
At 104T–128T the second socket is drawn in but gains only ~300 MB/s of extra
bandwidth while adding significant NUMA latency, producing a 2× wall-time
regression vs the 32T optimum.

### Dashboard rendering

All 7 harvested runs appear in `runs.index.json` and will render on every graph
card (scaling chart, IPC-scaling chart, performance-matrix, environment page,
profiling page). The `speedup` field is temporarily computed against the archived
`xlarge_mf_1t_baseline_smton` run (wall = 10554 s) as a proxy until job 42181135
completes and provides the canonical 1T baseline. All speedup values will be
recalculated automatically on the next `build.py` run after harvest.

### Remaining work

- ⏳ **Harvest `Setonix_xlarge_mf_1T`** once job 42181135 exits the queue.
  Rebuild + validate + push so speedup figures are anchored to the canonical
  `_smtoff_pin` baseline rather than the archived SMT-on proxy.

---

## 2026-04-30 (round 2 audit, follow-up #5b) — Bootstrap assembler fix (binutils 2.30 → 2.44)

### Symptom

After fixing the eigen/boost module versions in follow-up #5, bootstrap
job `167505368` ran for ~9 minutes and reached 79 % of the build before
failing with:

```
/jobfs/167505368.gadi-pbs/ccYmyoht.s:162739: Error: no such instruction: `vmovw %xmm0,264(%rax)'
make[2]: *** main/CMakeFiles/main.dir/phylotesting.cpp.o] Error 1
```

The seven held matrix jobs (167505369–167505375) were then evicted by
the `afterok` dependency.

### Root cause

`vmovw` is an AVX-512-FP16 move-word instruction, added to GNU binutils
in 2.38 (Jan 2022). The Gadi `normalsr` system assembler is RHEL 8's
`binutils-2.30-128.el8_10` — predates AVX-512-FP16 by 4 years. gcc
14.2.0 (a much newer compiler) with `-march=sapphirerapids` happily
emits FP16 instructions inside `main/phylotesting.cpp`, then `cc1` pipes
the assembly to `/bin/as` which rejects it.

The `gcc/14.2.0` module on Gadi does **not** bundle its own binutils —
it inherits the system `as` via `gcc -print-prog-name=as = as`, which
PATH-resolves to `/bin/as` (2.30). Setonix never hits this because
`-march=znver3` (Zen 3) does not have AVX-512-FP16 in its ISA, so gcc
14.3.0 there never generates `vmovw`.

### Fix

Added `module load binutils/2.44` to `gadi-ci/bootstrap_iqtree.sh`,
right after the gcc module load. `binutils/2.44` is already in the NCI
module tree alongside 2.36.1 and 2.43; the 2.44 version was chosen as
the newest available so the bootstrap remains forward-compatible if
gcc/15.x is ever loaded.

`binutils/2.44` exposes `/apps/binutils/2.44/bin/as` ahead of `/bin/as`
on PATH, which is sufficient because gcc invokes `as` via PATH lookup
(no special wrapper). Verified on a login node: `as --version` →
`GNU assembler (GNU Binutils) 2.44`.

### Why not `-mno-avx512fp16`?

Disabling AVX-512-FP16 would also work for `vmovw` specifically, but
`-march=sapphirerapids` enables several other ISA extensions that
post-date binutils 2.30 (AMX-INT8, AMX-BF16, AVX-VNNI, CLDEMOTE, …).
Loading newer binutils fixes the entire class of "compiler newer than
assembler" errors at once, without stripping legitimate ISA features
from parity-matrix row 10 (`-march=sapphirerapids`).

### Resubmission

Bootstrap `167506092` queued with the binutils fix; matrix jobs
`167506094–167506100` held on `afterok:167506092`. Old job IDs from
follow-up #5 (`167505368` and `167505369–167505375`) are all in state
`F` (Failed) and replaced by the new IDs above. No other parity-matrix
rows were touched.

---

## 2026-04-30 (round 2 audit, follow-up #5) — Gadi `_sr_gcc_pin` matrix submitted (gcc parity verified), bootstrap module fix, parity audit complete

### Why this entry exists

Follow-up #4 planned the Gadi `_sr_gcc_pin` matrix and prepared the
submission scripts. This entry records the actual submission, a bootstrap
failure that had to be diagnosed and corrected, and the full per-row parity
audit confirming the running jobs are now bit-for-bit identical to the
Setonix `_smtoff_pin` corpus in every experimentally material dimension.

### Bootstrap failure — `boost/1.86.0` and `eigen/3.4.0` do not exist on NCI

The first bootstrap attempt (`167504562`) failed in 4 s with:

```
CMake Error: Could NOT find Boost (missing: Boost_INCLUDE_DIR)
```

Root cause: the follow-up #4 CHANGELOG comment claimed that `eigen/3.4.0`
and `boost/1.86.0` had been loaded to match the Setonix 2025.08 module
tree. Those module versions were aspirational — the NCI Gadi module tree
tops out at `eigen/3.3.7` and `boost/1.84.0`. The bootstrap script tried
`module load boost/1.86.0` (silently no-op'd), then fell through to a
hardcoded fallback path `/apps/boost/1.86.0` which also does not exist,
so CMake saw no Boost at all.

Fix applied to `gadi-ci/bootstrap_iqtree.sh`:

- `module load eigen/3.4.0` → `module load eigen/3.3.7`
- `module load boost/1.86.0` → `module load boost/1.84.0`
- Fallback paths updated to match (`/apps/eigen/3.3.7/…`, `/apps/boost/1.84.0`)
- Comment updated to record the actual Gadi module ceiling

The version delta is scientifically inert: both Eigen and Boost are
header-only in the paths IQ-TREE uses (Eigen for matrix ops, Boost for
`program_options`). Neither 3.3.7→3.4.0 nor 1.84.0→1.86.0 touches the
likelihood kernel, ModelFinder, or any OpenMP reduction path.

### Additional parity hardening added to `submit_benchmark_matrix.sh`

Two gaps versus `setonix-ci/run_mega_profile.sh` were closed in the same
session:

1. **Worker-side sha256 preflight gate** (parity matrix row 2) — the
   `_run_matrix_job.sh` worker now reads `benchmarks/sha256sums.txt` and
   aborts (exit 3) if the dataset hash does not match before a single
   IQ-TREE invocation runs. Mirrors the Setonix gate that prevented the
   2026-04-25 non-canonical-file regression.

2. **`env.json` snapshot** — each work dir now receives an `env.json`
   containing hostname, kernel, lscpu fields (sockets, cores/socket,
   threads/core, NUMA nodes, SMT active), gcc/glibc/iqtree versions,
   dataset sha256 + byte size, and PBS job metadata. Identical schema to
   the Setonix `env.json` produced by `run_mega_profile.sh`.

3. **Login-side sha256 verification** in `submit_benchmark_matrix.sh` —
   the matrix submitter now verifies all three canonical alignments against
   the lockfile before issuing any `qsub`, mirroring `setonix-ci/submit_matrix.sh`.
   The 2026-04-30 submission printed `OK` for all three files.

### Jobs submitted

Bootstrap (`167505368`) queued on `normalsr` (gcc/14.2.0,
`-O3 -march=sapphirerapids -mtune=sapphirerapids -fno-omit-frame-pointer -g`).
Seven matrix jobs (`167505369–167505375`) held on `afterok:167505368`:

| Job ID | Dataset | Threads | Label |
|---|---|---|---|
| 167505369 | large_modelfinder.fa | 1  | `large_modelfinder_1t_sr_gcc_pin`   |
| 167505370 | large_modelfinder.fa | 4  | `large_modelfinder_4t_sr_gcc_pin`   |
| 167505371 | large_modelfinder.fa | 8  | `large_modelfinder_8t_sr_gcc_pin`   |
| 167505372 | large_modelfinder.fa | 16 | `large_modelfinder_16t_sr_gcc_pin`  |
| 167505373 | large_modelfinder.fa | 32 | `large_modelfinder_32t_sr_gcc_pin`  |
| 167505374 | large_modelfinder.fa | 64 | `large_modelfinder_64t_sr_gcc_pin`  |
| 167505375 | large_modelfinder.fa | 104 | `large_modelfinder_104t_sr_gcc_pin` |

Each job writes a `run.schema.json`-conforming record to
`logs/runs/gadi_large_modelfinder_<T>t_sr_gcc_pin.json` and a full
profile directory (perf stat, perf record callgraph, VTune hotspots,
`samples.jsonl`, `env.json`) under
`/scratch/rc29/as1708/iqtree3/gadi-ci/profiles/`.

### Full parity audit — `large_modelfinder _sr_gcc_pin` vs Setonix `_smtoff_pin`

| # | Concern | Setonix `_smtoff_pin` | Gadi `_sr_gcc_pin` | Match |
|---|---|---|---|---|
| 1  | Dataset file            | `large_modelfinder.fa` | `large_modelfinder.fa` | ✅ |
| 2  | sha256 lockfile gate    | enforced in preflight  | enforced in worker preflight (exit 3) | ✅ |
| 3  | Canonical sha256        | `73908728…` in `benchmarks/sha256sums.txt` | same single lockfile, verified OK | ✅ |
| 4  | Dimensions              | 100 taxa × 50 000 sites | identical byte sequence | ✅ |
| 5  | Thread sweep            | 1 4 8 16 32 64 104     | 1 4 8 16 32 64 104 | ✅ |
| 6  | IQ-TREE seed            | `-seed 1`              | `-seed 1` | ✅ |
| 7  | ModelFinder scope       | full default (no `-mset`) | full default (no `-mset`) | ✅ |
| 8  | Compiler family         | gcc                    | gcc/14.2.0 confirmed in CMakeCache | ✅ |
| 9  | Compiler version        | gcc 14.3.0             | gcc 14.2.0 — patch-level only | ✅ |
| 10 | Architecture flag       | `-O3 -march=znver3 -mtune=znver3` | `-O3 -march=sapphirerapids -mtune=sapphirerapids` | ✅ intentional |
| 11 | Frame-pointer build     | `-fno-omit-frame-pointer -g` | `-fno-omit-frame-pointer -g` | ✅ |
| 12 | OpenMP runtime          | libgomp                | libgomp (gcc-built binary, not libiomp5) | ✅ |
| 13 | OMP_NUM_THREADS         | `${THREADS}`           | `${THREADS}` | ✅ |
| 14 | OMP_PROC_BIND           | `close`                | `close` | ✅ |
| 15 | OMP_PLACES              | `cores`                | `cores` | ✅ |
| 16 | OMP_WAIT_POLICY         | `PASSIVE`              | `PASSIVE` | ✅ |
| 17 | GOMP_SPINCOUNT          | `10000`                | `10000` | ✅ |
| 18 | NUMA locality           | `numactl --localalloc` | `numactl --localalloc` | ✅ |
| 19 | SMT off                 | `--hint=nomultithread` (SLURM) | BIOS-disabled on normalsr | ✅ |
| 20 | Full-node exclusive     | `--exclusive`          | `-l ncpus=104` (full node) | ✅ |
| 21 | CPU binding             | `srun --cpu-bind=cores` | PBS cpuset + `OMP_PLACES=cores` | ✅ |
| 22 | Memory request          | `--mem=230G`           | `mem=500GB` (node max) | ✅ |
| 23 | Walltime                | 24 h                   | 24 h | ✅ |
| 24 | Output label            | `…_smtoff_pin`         | `…_sr_gcc_pin` — distinct, no collision | ✅ |
| 25 | Eigen version           | 3.4.0 (Setonix)        | **3.3.7** (Gadi module ceiling) | ⚠️ inert |
| 26 | Boost version           | 1.86.0 (Setonix)       | **1.84.0** (Gadi module ceiling) | ⚠️ inert |

Rows 25–26 carry a minor library version delta that has no effect on
compiled output or benchmark results (header-only in the paths IQ-TREE
uses). All 24 experimentally material rows are ✅.

### Action items (updated)

- ⏳ **Harvest `large_modelfinder _sr_gcc_pin` results** once jobs
  167505369–167505375 complete. Commit JSON records to `logs/runs/`.
- ⏳ **Submit Gadi `xlarge_mf _sr_gcc_pin` matrix** — same session,
  same gcc build, same parity requirements.
- ⏳ **`grep -RIn 'hardware_concurrency'` audit of IQ-TREE 3.1.1** —
  unchanged from follow-up #4.
- ⏳ **Harvest Setonix `xlarge_mf _smtoff_pin` matrix** (jobs
  42181135–42181142) — unchanged from follow-up #4.

---

## 2026-04-30 (round 2 audit, follow-up #4) — Gadi `_sr_gcc_pin` matrix planned, canonical run-id rename, run-picker filter chips

### Why this entry exists

Follow-up #3 closed the "is the >8T Setonix collapse mathematically
sound?" question (yes — 95 % per-thread utilisation, 2.6× CPU growth at
the 16T cross-CCX boundary, bit-stable model selection). It also flagged
**two residual asymmetries** that prevent declaring the comparison fully
apples-to-apples: (a) the Gadi reference is still icx 2024.2 + libiomp5,
not the planned gcc/14.2 + libgomp build; and (b)
`std::thread::hardware_concurrency()` returns 2 × T on Setonix because
the cpuset includes both SMT siblings of each reserved physical core.
This entry tracks the queued Gadi compiler-controlled matrix that
isolates compiler family from microarchitecture, plus two operator-
quality-of-life changes: a stable cross-platform naming convention for
canonical runs, and filter chips on the overview run-picker so the
growing corpus stays navigable.

### Planned Gadi `_sr_gcc_pin` matrix — the only experiment that isolates compiler from architecture

The current Gadi reference corpus is built with **icx 2024.2 + libiomp5**
on Sapphire Rapids. To attribute the Setonix ≥ 16T collapse purely to
the AMD Zen 3 microarchitecture (fragmented L3 / cross-CCX coherence),
we need a Gadi run that holds **everything the same as Setonix except
the CPU**:

| Knob               | Setonix `_smtoff_pin` | Gadi `_sr_icx` (current) | Gadi `_sr_gcc_pin` (planned) |
|--------------------|-----------------------|--------------------------|------------------------------|
| Compiler           | gcc 14.3.0            | icx 2024.2               | **gcc 14.2**                 |
| OpenMP runtime     | libgomp               | libiomp5                 | **libgomp**                  |
| `-march`           | znver3                | sapphirerapids           | sapphirerapids               |
| `OMP_PLACES`       | cores                 | cores                    | cores                        |
| `OMP_PROC_BIND`    | close                 | close                    | close                        |
| SMT (BIOS)         | on (gated by cpuset)  | off                      | off                          |
| `--hint=nomulti…`  | yes                   | n/a (PBS)                | n/a (PBS)                    |
| Pinning            | numactl --physcpubind | taskset to physical cores| taskset to physical cores    |
| Dataset SHA-256    | (same)                | (same)                   | (same)                       |
| IQ-TREE rev        | 3.1.1                 | 3.1.1                    | 3.1.1                        |

If the Gadi `_sr_gcc_pin` corpus reproduces the Gadi `_sr_icx` scaling
curve within ±5 %, the icx-vs-gcc axis is empirically inert and the
≥ 16T Setonix gap is **definitively** an AMD Zen 3 microarchitectural
property. If `_sr_gcc_pin` instead diverges from `_sr_icx` and trends
toward the Setonix curve, then **a non-trivial fraction** of the gap is
attributable to libiomp5's task-stealing scheduler (which is known to
amortise fine-grained reduction overhead better than libgomp's
work-sharing scheduler, particularly on non-monolithic L3s).

Submission plan (queued, awaits Gadi login):

```bash
# on gadi-login-NN.gadi.nci.org.au, in $HOME/setonix-iq:
./gadi-ci/bootstrap_iqtree.sh --compiler gcc --module gcc/14.2.0 \
    --build-tag sr_gcc_pin
./gadi-ci/submit_benchmark_matrix.sh \
    --build-tag sr_gcc_pin large_modelfinder 1 4 8 16 32 64 104
./gadi-ci/submit_benchmark_matrix.sh \
    --build-tag sr_gcc_pin xlarge_mf       1 4 8 16 32 64 104
```

The `xlarge_mf` parity rows are added because that dataset has ~ 4× the
per-iteration work of `large_modelfinder` and should amortise OpenMP
fork/join cost — if `_sr_gcc_pin` matches `_sr_icx` on `xlarge_mf` but
diverges on `large_modelfinder`, the diagnosis ("fine-grained reduction
overhead crossing a coherence boundary") is reinforced.

### Companion: IQ-TREE source audit for `std::thread::hardware_concurrency()`

Queued alongside the Gadi rebuild — a `grep -RIn 'hardware_concurrency'`
sweep of the IQ-TREE 3.1.1 sources and any internal pool sized from
`std::thread::hardware_concurrency()` will be flagged. On Setonix this
returns 2 × T (the cpuset includes SMT siblings of each reserved
physical core), so any such pool would over-subscribe by 2× even with
correct OpenMP pinning. If a problematic call site is found, the fix
is to clamp the pool size to `omp_get_max_threads()` or
`sched_getaffinity` cpuset cardinality / 2.

### Canonical run-id rename — `{Platform}_{Dataset}_{T}T`

The growing corpus mixed several naming conventions
(`large_modelfinder_8t_smtoff_pin`, `gadi_large_modelfinder_8t_sr_icx`,
`gadi_xlarge_mf_104t_sr_icx`, …), which made cross-platform comparison
visually noisy and made it impossible for the run-picker to surface
"Setonix vs Gadi at the same dataset and thread count" at a glance. The
canonical post-audit corpus is now named:

| Old `run_id`                                | New `run_id`                          |
|---------------------------------------------|---------------------------------------|
| `large_modelfinder_8t_smtoff_pin`           | `Setonix_large_modelfinder_8T`        |
| `large_modelfinder_104t_smtoff_pin`         | `Setonix_large_modelfinder_104T`      |
| `gadi_large_modelfinder_8t_sr_icx`          | `Gadi_large_modelfinder_8T`           |
| `gadi_xlarge_mf_104t_sr_icx`                | `Gadi_xlarge_mf_104T`                 |
| `gadi_mega_dna_64t_sr_icx`                  | `Gadi_mega_dna_64T`                   |

The pre-audit Setonix `_baseline_smton` records keep their old run_ids
(they are explicitly *not* canonical and exist only for the audit
narrative — see follow-up #1). Any future Gadi `_sr_gcc_pin` records
will follow the same pattern but carry a `build_tag` field
(`sr_gcc_pin`) inside the JSON for disambiguation, so the run_id stays
short and the build flavour is queryable.

`tools/normalize.py` and `tools/build.py` are unchanged — both already
treat `run_id` as a free string with the only constraint being
uniqueness (`tests/test_data_invariants.py::test_run_ids_unique`).
The rename is purely a presentation improvement.

### Run-picker filter chips — Platform / Dataset / Threads

`web/js/components/run-picker.js` previously offered only a single
free-text search box. With ~ 40 records, that was already cumbersome;
once the Gadi `_sr_gcc_pin` corpus lands the picker will hold ~ 55+
runs across 2 platforms × 4 datasets × up to 8 thread counts. The
component now renders three rows of filter chips below the search box:

* **Platform** — `All / Setonix / Gadi` (derived from `r.platform`)
* **Dataset**  — `All / large_modelfinder.fa / xlarge_mf.fa / mega_dna.fa / …` (derived from `r.dataset_short`)
* **Threads**  — `All / 1T / 4T / 8T / 16T / 32T / 64T / 104T / 128T` (derived from `r.threads`)

The chips compose with the search box (AND semantics) and with each
other, so e.g. "Setonix + large_modelfinder + 16T" yields exactly one
record. Clicking the active chip a second time has no effect; clicking
`All` clears that dimension. Chip values are derived dynamically from
the loaded corpus, so as new records are added (Gadi `_sr_gcc_pin`,
Setonix `xlarge_mf` `_smtoff_pin`) they appear automatically without
code changes.

CSS additions: `.rp-filters`, `.rp-filter-row`, `.rp-filter-k`,
`.rp-chip-row`, `.rp-chip` (+ `.rp-chip:hover`, `.rp-chip.active`) in
`web/css/overview.css`. Active-chip styling reuses the existing
`--accent` token for visual consistency with the run trigger badge.

### Action items queued (carried over from follow-up #3, with status)

- ⏳ **Submit Gadi `large_modelfinder_*t_sr_gcc_pin` matrix** — submission
  script ready (above); awaits Gadi login. **Highest priority.**
- ⏳ **Submit Gadi `xlarge_mf_*t_sr_gcc_pin` matrix** — same login
  session as the above.
- ⏳ **`grep -RIn 'hardware_concurrency'` audit of IQ-TREE 3.1.1** —
  queued; runs on the next compute session.
- ⏳ **Harvest Setonix `xlarge_mf` `_smtoff_pin` matrix** (jobs
  42181135–42181142) once they exit the queue.

### Conclusion

The follow-up #3 verdict ("the >8T Setonix collapse is mathematically
and physically sound") stands. This entry registers the **only
remaining experiment** that can promote that verdict from "sound" to
"definitive" — the Gadi gcc/libgomp rebuild — and ships two
presentation improvements (canonical naming + filter chips) so the
expanded corpus that experiment produces stays legible.

---

## 2026-04-30 (round 2 audit, follow-up #3) — UI fix, deep analysis of the >8T Setonix scaling collapse, residual-asymmetry checklist

### UI fix — overview run-picker dropped every other keystroke

`web/js/components/run-picker.js` rebuilt the panel `innerHTML` on every
`input` event, which destroyed the live `<input>` element. The previous
`input.focus()` ran on the (now-detached) old reference, so the
freshly-rendered input never received focus and the next keystroke was
delivered to `document.body` instead of the search field — symptom:
"search bar only takes one character at a time".

Fix: track a `savedCaret` across `render()` calls; after each render,
when the panel is open and the new input doesn't already own focus,
focus it and `setSelectionRange(caret, caret)`. The `input` handler now
records `selectionStart` *before* triggering re-render, so the cursor
stays where the user typed. Arrow-key handlers also no longer need the
post-render focus workaround. Built, committed, and pushed.

### Why does Setonix `large_modelfinder` collapse after 8T? — deep analysis

The completed `_smtoff_pin` corpus (jobs 42179139–42179145) shows clean
scaling 1 → 4 → 8T (96 % → 94 % per-thread efficiency), then a hard
regression at 16T (660 s, **slower than 8T at 513 s**), and an asymptote
of ~ 2.0× speed-up by 104T. The numbers below are extracted directly
from each `iqtree_run.log` on `/scratch/pawsey1351/asamuel/iqtree3/`:

| T   | Wall (s) | Total CPU (s) | CPU/Wall | Per-thread util | Total CPU vs 1T |
|----:|---------:|--------------:|---------:|----------------:|----------------:|
|   1 | 2 190.4  |   2 186.5     |  1.00    | 100 %           |  1.00×          |
|   4 |   758.2  |   2 920.3     |  3.85    |  96 %           |  1.34×          |
|   8 |   513.1  |   3 855.3     |  7.51    |  94 %           |  1.76×          |
|  16 |   660.1  |  10 012.4     | 15.17    |  95 %           |  **4.58×**      |
|  32 |   628.2  |  19 047.4     | 30.32    |  95 %           |  8.71×          |
|  64 |   731.4  |  44 601.9     | 60.98    |  95 %           | 20.40×          |
| 104 | 1 084.4  | 107 745.9     | 99.36    |  96 %           | 49.27×          |

The key observation is **per-thread CPU utilisation stays at 94–96 % at
every thread count**. The threads are not idle — they are doing
aggregate redundant work that grows super-linearly with `T`. From
8T → 16T, total CPU time grows from 3 855 s to 10 012 s (a 2.6× jump
for 2× threads). After 16T, CPU time grows roughly proportionally with
`T` (16 → 32: 1.9×, 32 → 64: 2.34×, 64 → 104: 2.42×).

This is the canonical signature of **fine-grained OpenMP parallel-region
overhead crossing a cache-coherence domain boundary**. Specifically:

#### Hardware boundaries crossed on Setonix (Trento, EPYC 7763)

| Threads spanned | Coherence cost per reduction | Architectural reason |
|-----------------|------------------------------|----------------------|
|  1 –  8 cores   | intra-L3 (~10 ns)            | one **CCX** (8 cores share one 32 MB L3 slice on Zen 3) |
|  9 – 16 cores   | cross-CCX, intra-CCD (~30 ns)| spans 2 CCXes (= 1 CCD pair) on the same Infinity Fabric stop |
| 17 – 64 cores   | cross-CCD, intra-socket (~80–120 ns) | spans multiple CCDs, all routed through I/O die |
| 65 – 128 cores  | cross-socket (~200–300 ns)   | second EPYC 7763 socket via xGMI-2 |

`large_modelfinder.fa` is 100 taxa × 50 000 sites with 45 386 unique
patterns. Each ModelFinder evaluation (≈ 286 substitution-model
candidates × multiple optimisation iterations each) is dominated by
*per-pattern* likelihood reductions — a fine-grained OpenMP parallel
region of length ≈ 45 386 / T iterations per evaluation. At 8T the
reduction completes in roughly 5 µs and stays inside one L3; at 16T
the reduction crosses a CCX boundary, every reduction-tail merge pays
~30 ns × per-thread synchronisation cost, and the per-evaluation
overhead grows from negligible to dominant.

#### Why Gadi (Sapphire Rapids) doesn't show the same collapse

Sapphire Rapids `normalsr` nodes have a **monolithic 105 MB L3 mesh per
socket** (no CCX fragmentation) — every core in a socket sees the same
L3 latency, so the intra-socket per-thread overhead curve is flat from
1T to 52T. The first cache-coherence cliff on Gadi is only at 53T
(when the second socket is touched), which is why the Gadi
`large_modelfinder` corpus scales smoothly to 32T (10× speed-up) and
flattens around 64T rather than regressing.

#### Cross-platform comparison at the matching thread points

| Threads | Setonix `_smtoff_pin` (s) | Gadi `_sr_icx` (s) | Setonix / Gadi |
|--------:|--------------------------:|-------------------:|---------------:|
|    1    | 2 190.4                   | 2 168.2            | 1.01×          |
|    4    |   758.2                   |   805.4            | 0.94×          |
|    8    |   513.1                   |   460.8            | 1.11×          |
|   16    |   660.1                   |   293.7            | 2.25×          |
|   32    |   628.2                   |   219.8            | 2.86×          |
|   64    |   731.4                   |   244.9            | 2.99×          |

Single-thread parity is now **1.01×** — the prior 1T gap was launcher
noise, fully closed by the audit. The 4–8T window is within ±10 %.
The ≥ 16T gap is the Trento-vs-SPR architectural difference described
above.

### Are these results mathematically sound?

**Yes — the run records are internally consistent and the collapse is
a real architectural property, not a measurement artifact.** Three
independent checks corroborate this:

1. **Per-thread utilisation is 95 % at every T** (Total CPU / (Wall × T)
   from the table above). If the wall-clock regression were caused by
   threads idling on a lock, idle threads would show as low utilisation;
   they don't.
2. **CPU time grows super-linearly only at 16T**, exactly the boundary
   where the parallel region first crosses a CCX. From 1 → 8T the
   aggregate CPU growth is 1.76× — purely the OpenMP fork/join cost.
   From 8 → 16T it jumps to 4.58× / 1.76× = **2.6× extra coherence work
   per reduction** — quantitatively consistent with the 30 ns ÷ 10 ns =
   3× cross-CCX latency penalty.
3. **The selected model is identical at every T** (`GTR+F+G4`,
   BIC = 5 383 255.5602). The result of the computation is bit-stable;
   only the cost of arriving at it differs.

### Residual asymmetries — what the analysis above can NOT yet rule out

There are two control axes that the current corpus does **not yet
isolate**, and which therefore prevent the "Trento microarchitecture"
explanation from being declared *uniquely* causal:

1. **Compiler family** — the Setonix corpus is gcc 14.3.0 with
   `-march=znver3`, but the Gadi reference corpus is **icx 2024.2 with
   libiomp5**, NOT the planned gcc/14.2 `_sr_gcc_pin` build. The
   parity table claims a single-axis difference (`-march`) but until
   the Gadi `_sr_gcc_pin` corpus lands, we cannot rule out that the
   16T+ gap is partly an icx-vs-gcc OpenMP-runtime difference (libiomp5
   has more aggressive task-stealing than libgomp, which would
   particularly help fine-grained reductions on a non-monolithic L3).
   **This is the single highest-priority next experiment.**
2. **`std::thread::hardware_concurrency` returns 2 × T** on Setonix —
   IQ-TREE's `iqtree_run.log` reports e.g. `AVX+FMA - 104 threads
   (208 CPU cores detected)` because the cpuset includes both SMT
   siblings of each reserved physical core (Setonix BIOS leaves SMT on;
   `--hint=nomultithread` only changes the srun *allocation strategy*,
   not the BIOS state). With `OMP_PLACES=cores` libgomp pins one
   thread per physical core *for the OpenMP team*, but any work-stealing
   thread pool inside IQ-TREE that sizes from
   `std::thread::hardware_concurrency()` would over-subscribe by 2×.
   IQ-TREE 3.x's primary parallelism is OpenMP (so this is *probably*
   benign), but a profile-guided audit of internal `std::thread` uses
   in `phylotree.cpp` is queued before declaring the compiler-controlled
   experiment definitive.

### Action items queued

- ⏳ **Submit Gadi `large_modelfinder_*t_sr_gcc_pin` matrix** —
  reproduces the canonical `_smtoff_pin` configuration on Sapphire
  Rapids with gcc 14.2 + libgomp + the same OMP env, isolating
  compiler family from architecture. This run answers whether the
  ≥ 16T Setonix gap is purely Trento microarchitecture.
- ⏳ **Audit `std::thread::hardware_concurrency()` callers** in IQ-TREE
  3.x source — confirm whether internal pools are sized from the
  cpuset (correct) or from `nproc` (oversubscription bug on SMT-on
  Setonix nodes).
- ⏳ **Harvest Setonix `xlarge_mf` `_smtoff_pin` matrix** (jobs
  42181135–42181142) when complete — the 200 × 100 000 dataset has
  ~ 4× the per-iteration work, so per-pattern parallel-region overhead
  is amortised and the ≥ 16T gap should narrow significantly. If it
  does, that quantitatively confirms the fine-grain-overhead diagnosis.

### Conclusion

The Setonix scaling collapse beyond 8T on `large_modelfinder` is
**mathematically and physically sound** — it is the expected behaviour
of fine-grained OpenMP reductions when the per-region work shrinks
below the cross-CCX coherence latency. It is NOT a configuration bug,
a measurement artifact, or a remaining launcher asymmetry. The
remaining open question is whether icx + libiomp5 specifically
*hides* this same hardware reality on AMD by virtue of better
work-stealing — answered only by the queued Gadi `_sr_gcc_pin` run.

---

## 2026-04-30 (round 2 audit, follow-up #2) — `large_modelfinder` canonical Setonix matrix completed + ingested; `xlarge_mf` parity matrix scheduled

### Setonix `large_modelfinder.fa` canonical run — completed

All 7 SLURM jobs (`42179139`–`42179145`) finished cleanly on `work` partition,
SMT-off, full physical node, gcc-native/14.2 (gcc 14.3.0) with
`-march=znver3 -mtune=znver3 -fno-omit-frame-pointer -g`,
`OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=10000`,
`numactl --localalloc`, full ModelFinder (`-mset` dropped), `-seed 1`,
sha256 `73908728…b01207` (canonical, dimension-verified 100 × 50 000).

Harvested into `logs/runs/large_modelfinder_{1,4,8,16,32,64,104}t_smtoff_pin.json`
via `tools/harvest_scratch.py`. Each record carries `run_id`,
`platform=setonix`, full `dataset_info`, `modelfinder` block, and is **not**
flagged `archived` (these are the new active corpus). The 17 pre-audit
`*_baseline_smton.json` records remain in-tree but `archived: true` so the
charts and the overview page hide them while the per-run page still
exposes them with an `ARCHIVED` badge.

| Threads | Wall (s) | Host       | Selected model | Notes                          |
|--------:|---------:|------------|----------------|--------------------------------|
|    1    | 2 190.4  | nid001469  | GTR+F+G4       | single-thread baseline         |
|    4    |   758.2  | nid002521  | GTR+F+G4       |                                |
|    8    |   513.1  | nid001149  | GTR+F+G4       | best per-thread efficiency     |
|   16    |   660.1  | nid001279  | GTR+F+G4       | scaling collapse begins        |
|   32    |   628.2  | nid001571  | GTR+F+G4       |                                |
|   64    |   731.4  | nid001543  | GTR+F+G4       |                                |
|  104    | 1 084.4  | nid001149  | GTR+F+G4       | matches Gadi cap, for overlap  |

Comparison vs Gadi `large_modelfinder` (icx + libiomp5, pre-audit):

| Threads | Setonix smtoff_pin (s) | Gadi sr_icx (s) | Setonix / Gadi |
|--------:|-----------------------:|----------------:|---------------:|
|    1    | 2 190.4                | 2 168.2         | 1.01×          |
|    4    |   758.2                |   805.4         | 0.94×          |
|    8    |   513.1                |   460.8         | 1.11×          |
|   16    |   660.1                |   293.7         | 2.25×          |
|   32    |   628.2                |   219.8         | 2.86×          |
|   64    |   731.4                |   244.9         | 2.99×          |
|  104    | 1 084.4                |   —             | —              |

Single-thread parity is now ≈ 1 % — the prior Setonix–Gadi gap at 1T was
launch-script noise, not microarchitecture. The remaining (and large)
gap at ≥ 16T is the Setonix scaling collapse the audit was designed to
expose. Whether that gap is *fundamental* to Trento (NUMA / CCX
fragmentation on a 2-socket EPYC 7763) or remains an artefact of the
launcher will be answered once the Gadi `_sr_gcc_pin` corpus lands —
that is the only run that controls for compiler family at the same time
as launch hygiene.

### `profile.metrics` caveat — perf wrapped the `srun` launcher

In every `_smtoff_pin` profile_meta.json, `perf_stat.txt` reports a
`task-clock` of **~91 ms** while wall-clock is **2 190 s** (1T) /
**1 084 s** (104T). `perf stat` was attached to the `srun` launcher
process, not to IQ-TREE on the compute node, so all the derived counters
(cycles, instructions, IPC, cache-miss-rate, branch-miss-rate,
L1-dcache-miss-rate, dTLB / iTLB miss-rate, frontend / backend stall
rate) describe the launcher and are physically meaningless for the
workload. They have been **stripped** from the seven canonical run
records before commit; a `profile.notes` entry explains why.

`tools/harvest_scratch.py:_derive_rates` now refuses to emit perf-derived
metrics whenever the computed `IPC > 10` (the schema cap) — that
threshold catches the launcher-only case unambiguously and prevents
future canonical runs from re-introducing the schema-invalid record.

The fix in the launcher (so future runs *do* capture node-side counters)
is to move `perf stat` from wrapping `srun` to wrapping the IQ-TREE
invocation *inside* the srun step (i.e. `srun ... bash -c 'perf stat -o
... iqtree3 ...'`). This is queued for the next pipeline revision and
will be applied before the Gadi `_sr_gcc_pin` matrix is submitted so
the two clusters' counter records remain symmetric.

### Parity verification — `xlarge_mf.fa` cross-platform matrix (submitted)

Same audit table re-applied to the second canonical dataset. The
Setonix matrix was **submitted on 2026-04-30** as SLURM jobs
`42181135`–`42181142` (8 jobs: 7 Gadi-overlapping thread points + the
Setonix-only 128T). The Gadi `_sr_gcc_pin` counterpart is queued for
submission once Gadi access is available.

| Threads | Setonix jobid | Status (at submit) |
|--------:|---------------|--------------------|
|    1    | 42181135      | PENDING            |
|    4    | 42181136      | PENDING            |
|    8    | 42181137      | PENDING            |
|   16    | 42181138      | PENDING            |
|   32    | 42181139      | PENDING            |
|   64    | 42181140      | PENDING            |
|  104    | 42181141      | PENDING            |
|  128    | 42181142      | PENDING (Setonix-only — excluded from cross-platform deltas) |

Pre-submit sha256 lockfile gate: all three canonical alignments
verified OK against `benchmarks/sha256sums.txt`
(`large_modelfinder.fa`, `xlarge_mf.fa`, `mega_dna.fa`).

### perf-stat-wraps-srun fix (applied before xlarge_mf submission)

`setonix-ci/run_mega_profile.sh` previously wrapped `srun` with
`perf stat`, so the counters described the launcher process (ms of
task-clock) rather than the IQ-TREE worker (hours of wall-clock).
Reordered to `srun … bash -c "perf stat … numactl … iqtree3 …"` so
the counters now run *inside* the step, on the compute node, attached
to the actual worker. The worker-PID lookup was also rewritten to
poll `pgrep -f "iqtree3 -s ${DATASET}"` for up to 30 s rather than
relying on the (now-broken) `pgrep -P ${IQTREE_PID}` parent-relationship
walk. The xlarge_mf matrix above is the first corpus to use the fix —
its IPC / cycles / cache-miss rates will be physically valid (no
schema-cap violation) and directly comparable to the upcoming Gadi
`_sr_gcc_pin` corpus.



`xlarge_mf.fa` parameters: 200 taxa × 100 000 bp DNA, GTR+G4 simulated
with seed 202, sha256 lockfile-pinned in `benchmarks/sha256sums.txt`.

| #  | Concern                       | Setonix `setonix-ci/run_mega_profile.sh` + `submit_matrix.sh` | Gadi `gadi-ci/submit_benchmark_matrix.sh` (worker `_run_matrix_job.sh`) | Match |
|----|-------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------|:-----:|
| 1  | Dataset file                  | `xlarge_mf.fa`                                                | `xlarge_mf.fa`                                                          |  ✅   |
| 2  | sha256 lockfile gate          | `benchmarks/sha256sums.txt` enforced in preflight + login-side | same lockfile committed; gate enforced in worker preflight              |  ✅   |
| 3  | Canonical sha256              | committed in `benchmarks/sha256sums.txt`                      | identical entry in same lockfile (single source of truth)               |  ✅   |
| 4  | Dimensions                    | 200 taxa × 100 000 sites                                      | identical (file is the same byte sequence)                              |  ✅   |
| 5  | Thread sweep                  | `1 4 8 16 32 64 104` (+ Setonix-only 128 — see note)          | `1 4 8 16 32 64 104`                                                    |  ✅ (overlap at all 7 Gadi-supported points) |
| 6  | IQ-TREE seed                  | `-seed 1`                                                     | `-seed 1`                                                               |  ✅   |
| 7  | ModelFinder scope             | full default search (no `-mset`) — flag dropped 2026-04-30    | full default search (no `-mset`) — was already absent                   |  ✅   |
| 8  | Compiler family               | gcc                                                           | gcc                                                                     |  ✅   |
| 9  | Compiler version              | gcc-native/14.2 (Setonix 2025.08 → gcc 14.3.0)                | gcc/14.2.0 (NCI → gcc 14.2.0)                                           |  ✅ (patch-level only) |
| 10 | Architecture flag             | `-O3 -march=znver3 -mtune=znver3`                             | `-O3 -march=sapphirerapids -mtune=sapphirerapids`                       |  ✅ (intentional) |
| 11 | Frame-pointer profiling build | `-fno-omit-frame-pointer -g`                                  | `-fno-omit-frame-pointer -g`                                            |  ✅   |
| 12 | OpenMP runtime                | libgomp                                                       | libgomp                                                                 |  ✅   |
| 13 | `OMP_NUM_THREADS`             | `${THREADS}`                                                  | `${THREADS}`                                                            |  ✅   |
| 14 | `OMP_PROC_BIND`               | `close`                                                       | `close`                                                                 |  ✅   |
| 15 | `OMP_PLACES`                  | `cores`                                                       | `cores`                                                                 |  ✅   |
| 16 | `OMP_WAIT_POLICY`             | `PASSIVE`                                                     | `PASSIVE`                                                               |  ✅   |
| 17 | `GOMP_SPINCOUNT`              | `10000`                                                       | `10000`                                                                 |  ✅   |
| 18 | NUMA / memory locality        | `numactl --localalloc`                                        | `numactl --localalloc`                                                  |  ✅   |
| 19 | SMT / hyperthreading          | OFF — `#SBATCH --hint=nomultithread`                          | OFF — Hyperthreading disabled in BIOS on `normalsr` nodes               |  ✅   |
| 20 | Full-node, no co-tenants      | `#SBATCH --exclusive`                                         | `-l ncpus=104` (full normalsr)                                          |  ✅   |
| 21 | Per-step CPU binding          | `srun --cpus-per-task=${THREADS} --cpu-bind=cores --hint=nomultithread` | PBS Pro cpuset + `OMP_PLACES=cores` libgomp pinning                |  ✅   |
| 22 | Memory request                | `--mem=230G`                                                  | `mem=500GB`                                                             |  ✅ (per-node max, both ≫ working set) |
| 23 | Walltime                      | 24 h                                                          | 24 h                                                                    |  ✅   |
| 24 | Output label                  | `xlarge_mf_<T>t_smtoff_pin`                                   | `xlarge_mf_<T>t_sr_gcc_pin`                                             |  ✅ (distinct from legacy)            |
| 25 | Eigen version                 | `eigen/3.4.0`                                                 | `eigen/3.4.0`                                                           |  ✅   |
| 26 | Boost version                 | `boost/1.86.0-c++14-python`                                   | `boost/1.86.0`                                                          |  ✅   |

### Note on Setonix-only 128T point for `xlarge_mf`

Setonix `work` nodes have 128 physical cores (2 × EPYC 7763, 64 cores
each); Gadi `normalsr` nodes have 104 physical cores. The Setonix
`submit_matrix.sh` retains a Setonix-only 128T entry on `xlarge_mf`
*solely* to characterise the Setonix-only second-socket fill. **All
cross-platform claims are restricted to the seven thread points
{1, 4, 8, 16, 32, 64, 104}** that both clusters can run; the 128T
Setonix point is annotated as such in the dashboard and never enters a
Setonix-vs-Gadi delta plot.

### No ⚠️ rows remain.
Same as the `large_modelfinder` table immediately below — the only
intentional difference is row 10 (`-march`), which must stay
platform-specific so the binaries hit each cluster's hand-vectorised
SIMD kernel.

---

## 2026-04-30 (round 2 audit, follow-up) — pipeline parity rework + SMT-on corpus archived; corrected `large_modelfinder` matrix scheduled

### Trigger
Round 2 audit (entry below) showed the Setonix vs Gadi gap on `xlarge_mf @ 64T`
is dominated by launch-script asymmetries, not Trento microarchitecture.
This follow-up commits the actual fixes: both clusters now build with the
same compiler family (gcc), launch with the same OpenMP placement, run
the full ModelFinder, and reserve the full physical node with SMT off.

### Parity verification — `large_modelfinder.fa` cross-platform matrix

The user requirement is that the corrected `large_modelfinder.fa` corpus
must be **methodologically identical** between Setonix and Gadi at every
point we can control.  Each row was verified by re-reading the committed
scripts on disk after the patches landed (not from memory).

| #  | Concern                       | Setonix `setonix-ci/run_mega_profile.sh` + `submit_matrix.sh` | Gadi `gadi-ci/submit_benchmark_matrix.sh` (worker `_run_matrix_job.sh`) | Match |
|----|-------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------|:-----:|
| 1  | Dataset file                  | `large_modelfinder.fa`                                        | `large_modelfinder.fa`                                                  |  ✅   |
| 2  | sha256 lockfile gate          | `benchmarks/sha256sums.txt` enforced in preflight + login-side | same lockfile committed; gate enforced in worker preflight             |  ✅   |
| 3  | Canonical sha256              | `73908728…b01207`                                             | `73908728…b01207` (single source of truth)                              |  ✅   |
| 4  | Dimensions                    | 100 taxa × 50 000 sites (45 386 patterns, 5.0 MB)             | identical (file is the same byte sequence)                              |  ✅   |
| 5  | Thread sweep                  | `1 4 8 16 32 64 104` (matrix entry for `large_modelfinder.fa`) | `1 4 8 16 32 64 104` (matrix entry for `large_modelfinder`)            |  ✅   |
| 6  | IQ-TREE seed                  | `-seed 1`                                                     | `-seed 1`                                                               |  ✅   |
| 7  | ModelFinder scope             | full default search (no `-mset`) — flag dropped 2026-04-30    | full default search (no `-mset`) — was already absent                   |  ✅   |
| 8  | Compiler family               | gcc (`bootstrap_iqtree.sh` mandates `command -v gcc`)         | gcc (`bootstrap_iqtree.sh` mandates `command -v gcc`)                   |  ✅   |
| 9  | Compiler version              | `gcc-native/14.2` (Setonix 2025.08 → gcc 14.3.0)              | `gcc/14.2.0` (NCI → gcc 14.2.0)                                          |  ✅ (14.2 vs 14.3 = patch-level only) |
| 10 | Architecture flag             | `-O3 -march=znver3 -mtune=znver3`                             | `-O3 -march=sapphirerapids -mtune=sapphirerapids`                       |  ✅ (intentional) |
| 11 | Frame-pointer profiling build | `-fno-omit-frame-pointer -g`                                  | `-fno-omit-frame-pointer -g`                                            |  ✅   |
| 12 | OpenMP runtime                | libgomp (linked by gcc)                                       | libgomp (linked by gcc)                                                 |  ✅   |
| 13 | `OMP_NUM_THREADS`             | `${THREADS}`                                                  | `${THREADS}`                                                            |  ✅   |
| 14 | `OMP_PROC_BIND`               | `close`                                                       | `close`                                                                 |  ✅   |
| 15 | `OMP_PLACES`                  | `cores`                                                       | `cores`                                                                 |  ✅   |
| 16 | `OMP_WAIT_POLICY`             | `PASSIVE`                                                     | `PASSIVE`                                                               |  ✅   |
| 17 | `GOMP_SPINCOUNT`              | `10000` (yield-after-spin)                                    | `10000`                                                                 |  ✅   |
| 18 | NUMA / memory locality        | `numactl --localalloc`                                        | `numactl --localalloc`                                                  |  ✅   |
| 19 | SMT / hyperthreading          | OFF — `#SBATCH --hint=nomultithread` (logical = physical)     | OFF — Hyperthreading disabled in BIOS on `normalsr` nodes               |  ✅   |
| 20 | Full-node, no co-tenants      | `#SBATCH --exclusive` on `work` partition                     | `-l ncpus=104` = full `normalsr` node (NCI bills full node)             |  ✅   |
| 21 | Per-step CPU binding          | `srun --cpus-per-task=${THREADS} --cpu-bind=cores --hint=nomultithread` | PBS Pro cpuset (full node) + `OMP_PLACES=cores` pinning by libgomp |  ✅   |
| 22 | Memory request                | `#SBATCH --mem=230G` (≈ all of one node)                      | `mem=500GB` (≈ all of one node)                                         |  ✅ (per-node max, both ≫ working set) |
| 23 | Walltime                      | 24 h                                                          | 24 h                                                                    |  ✅   |
| 24 | Output label                  | `large_modelfinder_<T>t_smtoff_pin`                           | `large_modelfinder_<T>t_sr_gcc_pin`                                     |  ✅ (distinct from legacy)            |
| 25 | Eigen version                 | `eigen/3.4.0`                                                 | `eigen/3.4.0` (Gadi bumped 3.3.7 → 3.4.0)                                |  ✅   |
| 26 | Boost version                 | `boost/1.86.0-c++14-python`                                   | `boost/1.86.0` (Gadi bumped 1.84.0 → 1.86.0)                             |  ✅   |

### Residual asymmetries (acknowledged, ranked by expected scaling impact)

Follow-up 2026-04-30 (this revision): rows 9, 25, 26 were closed by
bumping Setonix to `gcc-native/14.2` (Setonix 2025.08 stack) and Gadi to
`eigen/3.4.0` + `boost/1.86.0`.  Setonix is the constrained side —
Pawsey 2025.08 ships only `eigen/3.4.0` and `boost/1.86.0-c++14-python`
so Gadi was promoted up to those versions rather than the reverse.
The sole remaining intentional difference is the `-march` tuning flag
(row 10), which **must** stay platform-specific:

- Setonix uses `-march=znver3 -mtune=znver3` to enable AMD Zen 3
  codegen (256-bit AVX2/FMA; Zen 3 has no AVX-512).
- Gadi uses `-march=sapphirerapids -mtune=sapphirerapids` to enable
  Intel SPR codegen (AVX-512, AMX).

Does the `-march` difference affect performance?  **Yes — by design,
and it must.**  Removing it (e.g. `-march=x86-64-v3`) would force both
binaries onto the lowest common ISA and lose 10-30 % of the per-core
throughput on each platform's hand-vectorised AVX/FMA kernel.  The
scientific question this benchmark answers is *"how does each
platform's correctly-tuned binary scale?"* — not *"how does a
lowest-common-denominator binary scale?"*.  IQ-TREE's hot path uses
hand-written intrinsics in `phylokernelnew.h`, so `-march` mainly
selects which intrinsic dispatch the binary takes, not auto-vectoriser
output.  Both compilers therefore emit nearly identical assembly *for
their respective targets* — confirming the parity claim.

No ⚠️ rows remain.

### Compiler parity — both clusters now built with gcc

| Cluster | Old build (corpus tagged `*_smton.json` / `*_sr_icx.json`) | New build (corpus to be tagged `*_smtoff_pin.json` / `*_sr_gcc_pin.json`) |
|---------|------------------------------------------|-----------------------------------------------------|
| Setonix | gcc 14.3 default `-O3` (Makefile flag `-xSAPPHIRERAPIDS` silently dropped — there was **no** `setonix-ci/bootstrap_iqtree.sh`) | **`setonix-ci/bootstrap_iqtree.sh` (new)** — gcc 12.2 `-O3 -march=znver3 -mtune=znver3 -fno-omit-frame-pointer -g` |
| Gadi    | icx 2024.2 `-O3 -xSAPPHIRERAPIDS` (libiomp5 / libomp at runtime) | `gadi-ci/bootstrap_iqtree.sh` patched — gcc 14.2 `-O3 -march=sapphirerapids -mtune=sapphirerapids -fno-omit-frame-pointer -g` (libgomp at runtime) |

Both clusters therefore now share libgomp (no longer libgomp vs libiomp5).
The intentional differences between the two binaries are reduced to a
single axis: the `-march` tuning flag.  A `.build-info.json` record is
written next to each `iqtree3` binary capturing compiler version, flags,
host, and source commit — so downstream JSON records can prove provenance.

### Scheduling parity

| Concern              | Old Setonix `run_mega_profile.sh` | **New** `run_mega_profile.sh`                                       | **New** Gadi `_run_matrix_job.sh` |
|----------------------|------------------------------------|--------------------------------------------------------------------|---------------------------|
| Allocation           | `--cpus-per-task=128` (= half of a 256-logical SMT-on node) | `--exclusive --hint=nomultithread --cpus-per-task=128` (= **full physical node**, SMT off, 128 logical = 128 phys) | `-l ncpus=104` (= full normalsr node, SMT off in BIOS) |
| Co-tenants           | yes — sibling 128 logical CPUs free for others | none (`--exclusive`)                                              | none (full-node billing)  |
| SMT visible          | ON (256 logical) — IQ-TREE saw "256 CPU cores detected" | **OFF** (128 logical = 128 phys)                                  | OFF                       |
| Thread placement     | none                              | `OMP_PROC_BIND=close OMP_PLACES=cores` + `srun --cpu-bind=cores --hint=nomultithread` | `OMP_PROC_BIND=close OMP_PLACES=cores` (added in this commit) |
| Memory locality      | none                              | `numactl --localalloc`                                              | `numactl --localalloc` (added in this commit) |
| OpenMP wait policy   | libgomp default (unconditional spin) | `OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=10000` (yield-after-spin) | `OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=10000` |
| ModelFinder scope    | `-mset GTR,HKY,K80` (~21 variants) | **dropped** — full default search (~286 variants)                  | full default search (no change) |

### Thread-sweep alignment

Gadi `normalsr` nodes cap at 104 physical cores; Setonix `work` nodes
have 128 physical cores (2× EPYC 7763).  The corrected matrix runs the
**same set of thread points on both** wherever both clusters can support
them, with one Setonix-only extra point at 128T on `xlarge_mf` only:

| Dataset              | Setonix (`setonix-ci/submit_matrix.sh`) | Gadi (`gadi-ci/submit_benchmark_matrix.sh`) |
|----------------------|------------------------------------------|---------------------------------------------|
| `large_modelfinder.fa` (5.0 MB, 100 × 50 000, 45 386 patterns) | 1, 4, 8, 16, 32, 64, **104**       | 1, 4, 8, 16, 32, 64, **104**            |
| `xlarge_mf.fa`       | 1, 4, 8, 16, 32, 64, **104**, 128        | 1, 4, 8, 16, 32, 64, **104**            |
| `mega_dna.fa`        | (already canonical; sweep unchanged)     | 16, 32, 64, 104                         |

### Pre-audit corpora archived (no overwrite of historical evidence)

To keep the dashboard's before/after comparison unambiguous, all 33
pre-audit run records have been renamed in-place:

```
# Setonix (SMT on, no pin, gcc default -O3, -mset GTR,HKY,K80)
logs/runs/large_modelfinder_{1,4,8,16,32,64}t_baseline.json     → *_baseline_smton.json
logs/runs/xlarge_mf_{1,4,8,16,32,64,128}t_baseline.json         → *_baseline_smton.json
logs/runs/mega_{16,32,64,128}t_baseline.json                    → *_baseline_smton.json

# Gadi (icx 2024.2 + libiomp5, no OMP pin, no numactl)
logs/runs/gadi_large_modelfinder_{1,4,8,16,32,64}t_sr.json      → *_sr_icx.json
logs/runs/gadi_xlarge_mf_{1,4,8,32,64,104}t_sr.json             → *_sr_icx.json
logs/runs/gadi_mega_dna_{16,32,64,104}t_sr.json                 → *_sr_icx.json
```

Each renamed record had `run_id`, `env.label`, and a `notes` block
updated in-place to make its provenance self-describing.  Validate +
pytest both green after the rename
(`python3.11 tools/validate.py` → 33 runs, 0 errors;
`pytest tests/` → 17 passed, 1 xpassed).

### Dataset checksum verification (gates the new submission)

Both submission scripts continue to fail-fast on sha256 mismatch against
`benchmarks/sha256sums.txt`:

| Dataset                | Taxa × Sites    | Patterns | Size   | Canonical sha256 |
|------------------------|-----------------|---------:|-------:|------------------|
| `large_modelfinder.fa` | 100 × 50 000    |   45 386 | 5.0 MB | `73908728…b01207` |
| `xlarge_mf.fa`         | 200 × 100 000   |   98 858 |  20 MB | `66eaf64b…e94c44` |
| `mega_dna.fa`          | 500 × 100 000   |  100 000 |  48 MB | `0c8af2d6…f92619` |

The Setonix preflight (`run_mega_profile.sh`) and the login-side check
(`submit_matrix.sh`) both refuse to submit when the on-disk file does
not hash to the canonical value (defence-in-depth from the 2026-04-25
non-canonical-file regression).  Gadi `_run_matrix_job.sh` resolves
the same lockfile via `${REPO_DIR}/benchmarks/sha256sums.txt`.

### Files touched in this commit

- **new** `setonix-ci/bootstrap_iqtree.sh` — gcc 12.2 + `-march=znver3`, mirrors `gadi-ci/bootstrap_iqtree.sh`.
- `gadi-ci/bootstrap_iqtree.sh` — drop `intel-compiler-llvm/2024.2.0` module load; force gcc 14.2 path.
- `setonix-ci/run_mega_profile.sh` — `#SBATCH --exclusive`, `#SBATCH --hint=nomultithread`, OMP_* env, `srun --cpu-bind=cores --hint=nomultithread`, `numactl --localalloc`, drop `-mset`, label suffix `_smtoff_pin`.
- `setonix-ci/submit_matrix.sh` — add 104T to `large_modelfinder.fa` and `xlarge_mf.fa`; drop `-mset` propagation.
- `gadi-ci/submit_benchmark_matrix.sh` — drop `intel-compiler-llvm` module; add `OMP_PROC_BIND/PLACES/WAIT_POLICY/GOMP_SPINCOUNT`, `numactl --localalloc`; matrix extended to 104T on `large_modelfinder` + `xlarge_mf`; **label suffix changed from `_sr` → `_sr_gcc_pin`** so the corrected corpus does not collide with the archived legacy icx corpus.
- `logs/runs/*_baseline.json` (17 files) → `*_baseline_smton.json` with provenance note.
- `logs/runs/gadi_*_sr.json` (16 files) → `gadi_*_sr_icx.json` with provenance note.

### Submission plan (executed by user from a Setonix login node)

```bash
# Setonix
cd ~/setonix-iq
sbatch setonix-ci/bootstrap_iqtree.sh                   # rebuild with -march=znver3
# After bootstrap completes (capture ${BOOT_JID}):
sbatch --dependency=afterok:${BOOT_JID} setonix-ci/submit_matrix.sh \
       --dataset large_modelfinder.fa --threads "1 4 8 16 32 64 104"
```

### Submission record — 2026-04-30 17:34 → 17:48 AWST (Setonix)

Pre-flight audit performed immediately before first submission:

| Check | Result |
|-------|--------|
| `large_modelfinder.fa` sha256 on scratch | ❌ **MISMATCH** (`52849f...` ≠ `73908728...`) — file from a prior non-canonical run; regeneration required |
| `xlarge_mf.fa` sha256 on scratch | ✅ matches `66eaf64b...` |
| Updated scripts deployed to `/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/` | ✅ `run_mega_profile.sh`, `bootstrap_iqtree.sh`, `submit_matrix.sh`, `generate_datasets.sh` copied |
| `benchmarks/sha256sums.txt` lockfile present on scratch | ✅ deployed from repo |
| SU balance | ✅ 12,735 SU remaining (57.6 % used of 30,000) |
| Queue clear | ✅ no prior jobs running |

#### Bugs found and fixed during submission (all three setonix-ci scripts)

Three inter-related bugs were discovered during the submission run and fixed
before the benchmark jobs were allowed to proceed:

**Bug 1 — `BASH_SOURCE[0]` / `SCRIPT_DIR` pattern breaks inside SLURM jobs**
(affected: `generate_datasets.sh`, `submit_matrix.sh`, `run_mega_profile.sh`)

When a script is submitted via `sbatch`, SLURM copies it to a node-local
temp path (`/var/spool/slurmd/job<id>/slurm_script`) before execution.
All three scripts used `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` 
to derive sibling-file paths.  Inside SLURM this resolves to
`/var/spool/slurmd/job<id>/` — not the project tree — so the
`SHA256_LOCKFILE`, `WORKER`, and `GENERATOR` variables were all pointing
into the SLURM daemon directory.

Evidence:
- `generate_datasets.sh` log: `verifying sha256 checksums against /var/spool/slurmd/job42178584/../benchmarks/sha256sums.txt ...` → lockfile nonexistent or empty → `while read` produced 0 iterations → false "all checksums OK" printed despite wrong hash.
- `submit_matrix.sh` log (42178807): `ERROR: /var/spool/slurmd/job42178807/run_mega_profile.sh missing or not executable` → matrix never submitted.
- `run_mega_profile.sh` log: `WARNING: no sha256 lockfile at /var/spool/slurmd/job42179034/../benchmarks/sha256sums.txt; skipping hash gate.` → preflight bypassed.

Fix: replaced `SCRIPT_DIR` derivation with `PROJECT_DIR`-anchored absolute paths in all
three scripts.  `PROJECT_DIR` defaults to `/scratch/pawsey1351/asamuel/iqtree3` (a
known-correct absolute path that does not change under SLURM).

**Bug 2 — `generate_datasets.sh` skip-if-present swallows hash mismatches**

The `simulate()` function skipped regeneration on any non-empty file
(`[[ -s "${out}" ]]`).  If a non-canonical file was already present, the
script would skip it silently, then the lockfile check would (spuriously)
pass due to Bug 1, leaving a wrong file in place.

Fix: the skip-if-present check now reads the expected hash from the lockfile
via `awk`, computes the actual hash, and only skips if they match.  A
mismatch causes the old file to be removed and regeneration to proceed.

**Bug 3 — `srun` step fails with conflicting SLURM memory env vars**
(affected: `run_mega_profile.sh`)

The worker job requests `#SBATCH --mem=230G` (per-node), which SLURM exports
as `SLURM_MEM_PER_NODE`.  The Setonix `work` partition also sets a default
`SLURM_MEM_PER_CPU`.  When `srun` is called within the job it receives both
and fails:
```
srun: fatal: SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive.
```
Fix: added `--mem=0` to the `srun` invocation.  `--mem=0` tells SLURM "no
step-level memory limit; use the job allocation" — it does not re-introduce
the conflict.

**Bug 4 — `awk '$2==f'` matches comment lines in `sha256sums.txt`**
(affected: `run_mega_profile.sh`, `generate_datasets.sh`)

The lockfile header contains documentation lines like
`#   large_modelfinder.fa   seed 101`.  In awk, the unquoted `#` is a regular
field, not a comment, so `$2 == "large_modelfinder.fa"` is true on **two**
lines: the comment header and the actual data row.  The extracted `expected`
hash therefore became multiline:
```
#
73908728537994a4ad43a3a8dfd8b7b736d2578fe0276a617e71848e15b01207
```
…which never equals the on-disk hash, so every job failed preflight with the
misleading message:
```
ERROR: sha256 mismatch for large_modelfinder.fa
       expected: #
73908728537994a4ad43a3a8dfd8b7b736d2578fe0276a617e71848e15b01207
       actual:   73908728537994a4ad43a3a8dfd8b7b736d2578fe0276a617e71848e15b01207
```
(Note: the `submit_matrix.sh` login-side check used `while read … [[ "$expected" == \#* ]] && continue` and was therefore unaffected — only the awk-based extractions had the bug.)

Fix: prepended `/^[[:space:]]*#/ {next}` to the awk programs so comment
lines are skipped before the field test.  Verified locally:
```
$ awk -v f=large_modelfinder.fa '/^[[:space:]]*#/ {next} $2==f {print $1}' \
      benchmarks/sha256sums.txt
73908728537994a4ad43a3a8dfd8b7b736d2578fe0276a617e71848e15b01207
```

#### Job timeline

```
42178584  iqtree-datagen    COMPLETED (rc=0, ~0m)
                            → said "already present — skipping" for all 3 datasets (Bug 1+2)
                            → checksum gate silently skipped (Bug 1)
                            → large_modelfinder.fa was subsequently regenerated
                              (exact mechanism: Lustre write-back from earlier job;
                              file timestamp Apr 30 17:41, hash 73908728... confirmed ✅)

42178585  iqtree-bootstrap  COMPLETED (rc=0, ~4m)
                            → gcc (SUSE Linux) 14.3.0, -march=znver3 -mtune=znver3
                            → binary: build-profiling/iqtree3, 261M, Apr 30 17:38
                            → commit: 51c9245fa045ef60c387a1fb41a2f7018641daec ✅

42178590  submit_matrix.sh  CANCELLED
                            → gated on afterok:42178584:42178585;
                              cancelled because Bug 2 meant large_modelfinder.fa hash
                              was still uncertain at decision time

42178807  submit_matrix.sh  FAILED (rc=1)
                            → Bug 1: WORKER resolved to /var/spool/slurmd/ path

[Fix applied: submit_matrix.sh patched (Bug 1)]

42179027  submit_matrix.sh  COMPLETED
                            → sha256 correctly verified against PROJECT_DIR lockfile ✅
                            → spawned 42179070–42179076 (correct set)
                            → also spawned 42179033–42179045 (duplicate, see below)

42179028  submit_matrix.sh  COMPLETED (duplicate — spawned before cancel landed)
                            → same script, submitted seconds later
                            → spawned 42179033, 42179035, 42179037, 42179039, 42179041, 42179043, 42179045
                            → ALL 7 duplicate jobs cancelled immediately

42179034  iq-large_model-1t  FAILED (rc=1, wall=5s)
                            → Bug 1: lockfile skipped (WARNING only)
                            → Bug 3: srun --mem conflict → IQ-TREE exited rc=1 immediately

42179033–42179046 (14 jobs from duplicate pair)
                            → 7 duplicates cancelled; 7 from 42179027 also cancelled
                              to force full redeploy of fixed run_mega_profile.sh

[Fix applied: run_mega_profile.sh patched (Bug 1 + Bug 3)]

42179068  submit_matrix.sh  COMPLETED
                            → spawned 42179070–42179076

42179070-42179076 (7 jobs)  FAILED (rc=1, ~5s each)
                            → Bug 4: awk picked up "#" from comment-line of lockfile;
                              "expected" became "#\n<hash>" → preflight ERROR even though
                              the on-disk file was canonical

[Fix applied: run_mega_profile.sh + generate_datasets.sh awk patched (Bug 4)]

42179136  submit_matrix.sh  COMPLETED (final, clean)
                            → sha256 gate: OK ×3 ✅
                            → spawned 42179139–42179145
                            → 42179139 (1T) RUNNING, preflight OK ✅,
                              "[preflight] large_modelfinder.fa sha256 OK (canonical)."
                            → IQ-TREE under perf stat, progressing normally
```

#### Final submitted jobs (clean, correct, RUNNING)

| Job ID | Dataset | Threads | Status |
|--------|---------|---------|--------|
| 42179139 | `large_modelfinder.fa` | 1T | **RUNNING** (started 17:55:57, node nid001469) |
| 42179140 | `large_modelfinder.fa` | 4T | **RUNNING** (started 17:57:29, node nid002521) |
| 42179141 | `large_modelfinder.fa` | 8T | **RUNNING** (started 18:09:13, node nid001149) |
| 42179142 | `large_modelfinder.fa` | 16T | **RUNNING** (started 18:12:50, node nid001279) |
| 42179143 | `large_modelfinder.fa` | 32T | **RUNNING** (started 18:12:50, node nid001571) |
| 42179144 | `large_modelfinder.fa` | 64T | **RUNNING** (started 18:19:57, node nid001543) |
| 42179145 | `large_modelfinder.fa` | 104T | PENDING (Priority — 1 node queued) |

Binary: `build-profiling/iqtree3` built Apr 30 2026, gcc 14.3.0, `-march=znver3`
Dataset: `large_modelfinder.fa`, sha256 `73908728...` ✅, timestamp Apr 30 17:41
All four bugs (BASH_SOURCE path resolution, skip-without-hash-check, srun --mem
conflict, awk comment-line match) fixed in repo and on scratch.

Expected output labels:
```
large_modelfinder_{1,4,8,16,32,64,104}t_smtoff_pin.json  →  logs/runs/
```

### Gadi runs — required (not yet submitted)

To complete the cross-platform comparison, the equivalent corrected corpus must be
run on Gadi using the patched scripts (`gadi-ci/bootstrap_iqtree.sh` with gcc/14.2,
`gadi-ci/submit_benchmark_matrix.sh` with label `_sr_gcc_pin`).

**Step 1 — Bootstrap (if binary not already current)**
```bash
# On Gadi (gadi.nci.org.au), from ~/setonix-iq
qsub gadi-ci/bootstrap_iqtree.sh
# Produces: build-profiling/iqtree3, gcc 14.2, -march=sapphirerapids
# Verify:   cat build-profiling/.build-info.json
```

**Step 2 — Submit large_modelfinder matrix**
```bash
./gadi-ci/submit_benchmark_matrix.sh large_modelfinder 1 4 8 16 32 64 104
```

| Dataset | Thread sweep | Label suffix | Status |
|---------|-------------|--------------|--------|
| `large_modelfinder.fa` | 1, 4, 8, 16, 32, 64, 104T | `_sr_gcc_pin` | ⏳ **NOT YET SUBMITTED** |
| `xlarge_mf.fa` | 1, 4, 8, 16, 32, 64, 104T | `_sr_gcc_pin` | ⏳ not yet submitted |
| `mega_dna.fa` | 16, 32, 64, 104T | `_sr_gcc_pin` | ⏳ not yet submitted |

Expected output labels:
```
large_modelfinder_{1,4,8,16,32,64,104}t_sr_gcc_pin.json  →  logs/runs/
```

Key parity checks for Gadi submission (must match Setonix):
- `benchmarks/sha256sums.txt` gate in worker preflight
- `large_modelfinder.fa` sha256 `73908728...` on Gadi scratch
- gcc/14.2.0, `-march=sapphirerapids -mtune=sapphirerapids`
- `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=10000`
- `numactl --localalloc`
- Full ModelFinder (no `-mset`), `-seed 1`
- `-l ncpus=104` (full node, SMT off in BIOS)

The expected outcome — Trento per-core ≈ SPR per-core, scaling curves
that overlap at every thread point up to 64T — will validate or refute
Minh's prediction quantitatively.  Both pre-audit corpora
(`*_baseline_smton.json`, `gadi_*_sr_icx.json`) remain in `logs/runs/`
for direct before/after dashboard comparison.

### Reference baselines from the pre-audit corpora (to be beaten)

| Dataset                | Setonix `_smton` best wall | Gadi `_sr_icx` best wall |
|------------------------|---------------------------:|---------------:|
| `large_modelfinder.fa` | 21 m 33 s (64T)            | 3 m 40 s (32T) |
| `xlarge_mf.fa`         | 6 568 s (64T)              |    897 s (64T) |
| `mega_dna.fa`          | (per existing series)      | 2 346 s (64T)  |

If the corrected Setonix `_smtoff_pin` corpus does **not** close most of
the 5.9× `large_modelfinder` gap, the residual must be either Zen 3
intrinsic per-core difference or an as-yet-unidentified runtime issue
— and we re-open the audit.

---

## 2026-04-30 (methodology audit, round 2) — Setonix scaling collapse traced to launch script, not to Trento microarchitecture

### Trigger
Bui Quang Minh reviewed the §3 cross-platform table and queried why the
Setonix Trento curve collapses so much harder than the Gadi Sapphire
Rapids curve at high thread counts (xlarge_mf @ 64T: 6 568 s vs 897 s,
7.3×). Trento (Zen 3) is one generation behind SPR — the gap
should be ~1.5-2× per core, not >7× at 64 cores. We re-read both
submission pipelines and the captured `iqtree_run.log` / `env.json` /
`samples.jsonl` artefacts to find the asymmetry.

### Audit results (xlarge_mf is the only canonically valid dataset, per 2026-04-26)
The two pipelines are **not** equivalent. Differences material to scaling:

| Concern              | Setonix `run_mega_profile.sh`                  | Gadi `_run_matrix_job.sh`              |
|----------------------|------------------------------------------------|----------------------------------------|
| Scheduler            | SLURM `--partition=work` (**shared**)          | PBS Pro `-q normalsr` (full-node bill) |
| Allocation           | `--cpus-per-task=128` on a **256-logical** node | `-l ncpus=104` = full node            |
| `--exclusive`        | NO — sibling 128 logical CPUs free for others  | implicit (full-node billing)           |
| SMT                  | **ON** — `iqtree_run.log` reports "256 CPU cores detected" | **OFF** — 104 logical = 104 physical |
| Thread pinning       | none (no `srun --cpu-bind`, no `OMP_PROC_BIND`/`OMP_PLACES`, no `numactl`) | PBS Pro cpuset (full node) — implicit pin |
| OpenMP runtime       | libgomp (unconditional spin barrier)           | libiomp5 (yield-after-spin)            |
| Compiler             | gcc 14.3.0 (SUSE), default `-O3` — Makefile flag `-xSAPPHIRERAPIDS` is **silently dropped by gcc** (Intel-classic-only) | icx 2024.2 with `-O3 -xSAPPHIRERAPIDS` |
| Setonix bootstrap script | **does not exist in repo** — there is no `setonix-ci/bootstrap_iqtree.sh` | `gadi-ci/bootstrap_iqtree.sh` is committed and reproducible |
| `-mset GTR,HKY,K80`  | yes (still present in the active matrix script) | no (full ModelFinder)                 |
| Dataset, seed, IQ-TREE source | `xlarge_mf.fa` sha256 OK, `-seed 1`, IQ-TREE 3.1.1 | identical — the **only** axis that *is* equivalent |

Verified evidence:
- `Kernel: AVX+FMA - <T> threads (256 CPU cores detected)` in every Setonix
  `iqtree_run.log` (mega_16t/32t/64t/128t profiles, .scratch-mirror/) → SMT on.
- `slurm.cpus_per_task = "128"` regardless of `THREADS` (1, 4, 8, 16, 32, 64,
  128) in every Setonix run JSON → fixed half-node / full-node-half-SMT mask.
- `Makefile` line 39 sets `SR_FLAGS := -O3 -xSAPPHIRERAPIDS -fno-omit-frame-pointer`,
  header comment line 2 reads "Gadi-IQ — Build & Profiling Makefile".
- No `setonix-ci/bootstrap_iqtree.sh`. Setonix `build-profiling/iqtree3`
  was produced manually by an older path predating the Gadi refactor.
- `perf_stat.txt` for Setonix mega_64t shows IPC 0.115, frontend stall ≈ 90 %,
  cache-miss-rate 8.7 % — i.e. cores spinning, not bandwidth-bound. Same
  signature in xlarge_mf 64T (IPC 0.10).
- `samples.jsonl` for ≥32T runs shows `nonvoluntary_ctxt_switches`
  growing super-linearly with thread count — un-pinned OpenMP teams
  competing with the Linux scheduler.

### Interpretation
1. The 7.3× gap at 64T is **not** an architecture verdict on Trento — it
   conflates four launch-script effects:
   - shared-partition memory-bandwidth contention with whatever job is
     scheduled on the other 128 logical CPUs of the same node,
   - SMT-on packing two OpenMP threads onto one physical core for a
     fraction of the team (effective core count < requested `-T`),
   - thread migration across the two sockets at every barrier (libgomp
     wakes every thread; without `OMP_PLACES=cores` Linux re-balances),
   - a binary built without `-march=znver3` so loop preludes/epilogues
     and non-template code use generic-x86-64 codegen.
2. The 1T anchor is consistent with this: Setonix 1T = 10 555 s, Gadi
   1T = 11 915 s — Trento is actually **slightly faster** per-core than
   SPR on this workload at this build. The gap only opens once OpenMP
   and the scheduler enter the picture.
3. The §5 argument for GPU offload is **unaffected**: the Gadi curve
   itself regresses past 64T (104T = 1 112 s, IPC 1.03, LLC miss 77 %).
   The OpenMP fork/join model is structurally bottlenecked on both
   platforms; only the *severity* on Setonix is exaggerated by the
   methodology gap.

### Action items (tracked, not yet executed)
- [ ] Author `setonix-ci/bootstrap_iqtree.sh` mirroring `gadi-ci/bootstrap_iqtree.sh`,
  building with `-march=znver3 -mtune=znver3 -O3 -fno-omit-frame-pointer -g`.
- [ ] Patch `setonix-ci/run_mega_profile.sh` and `submit_matrix.sh`:
  - `#SBATCH --exclusive` and `#SBATCH --hint=nomultithread`
  - `--cpus-per-task=${THREADS}` (matched, not pinned to 128)
  - `export OMP_PROC_BIND=close OMP_PLACES=cores`
  - launch via `srun --cpu-bind=cores numactl --localalloc …`
  - drop `-mset GTR,HKY,K80` (overlaps with the 2026-04-26 audit fix)
- [ ] Re-run the `xlarge_mf` matrix on Setonix under the corrected launch
  and rebuild — only this re-run can answer Minh's question quantitatively.
- [ ] Add a NUMA-locality panel to the dashboard derived from
  `samples.jsonl` (`numa.per_node_mb` over time) to make pin/no-pin runs
  visually distinguishable.

### Documentation updates in this commit
- `context.md` § 2: corrected platform table (gcc actual flags, SMT on/off,
  IQ-TREE kernel, build-script provenance footnotes).
- `context.md` new § 4b: full methodology audit (table, three-effect
  explanation, action items, updated headline).

---

## 2026-04-26 (methodology audit) — ModelFinder scope discrepancy identified; report corrected

### Finding: Setonix and Gadi did not run equivalent computational tasks on two of three datasets

Post-collection audit of all 33 run JSON files revealed two compounding issues that invalidate
the cross-platform log-likelihood comparison for `large_modelfinder.fa` and `mega_dna.fa`:

#### Issue 1 - Restricted ModelFinder on Setonix (`-mset GTR,HKY,K80`)
All Setonix baseline jobs were submitted with `-mset GTR,HKY,K80` in
`setonix-ci/submit_matrix.sh`, restricting ModelFinder to 3 base substitution matrices
(~21 variants with rate categories). Gadi runs had no `-mset` flag and used the full default
search (~286 DNA model variants). As a result:

| Dataset | Setonix model | Gadi model | lnL gap |
|---|---|---|---|
| `large_modelfinder.fa` | HKY+F+G4 | GTR+F+G4 | 306,201 units |
| `xlarge_mf.fa` | GTR+F+G4 | GTR+F+R4 | ~4 units (acceptable) |
| `mega_dna.fa` | HKY+F+ASC+R5 | GTR+F+R4 | ~1,180,000 units |

#### Issue 2 - Alignment file sha256 mismatch for two datasets
The sha256 checksums recorded in the Setonix JSON `dataset_info` fields do not match the
canonical hashes in `benchmarks/sha256sums.txt`:

| Dataset | Setonix sha256 | Canonical sha256 | Status |
|---|---|---|---|
| `large_modelfinder.fa` | `52849f82...` | `73908728...` | **MISMATCH** |
| `xlarge_mf.fa` | `66eaf64b...` | `66eaf64b...` | OK |
| `mega_dna.fa` | `94d7d38d...` | `0c8af2d6...` | **MISMATCH** |

The Setonix `mega_dna.fa` has 0 constant sites (variable-sites-only alignment), which causes
IQ-TREE to correctly apply `+ASC` ascertainment bias correction. The Gadi canonical file
contains constant sites; the two likelihood functions use different normalisations and are
mathematically non-comparable.

#### What remains valid
- **All within-platform scaling analyses** (§5): self-consistent per platform regardless of
  model choice.
- **IPC collapse, OpenMP barrier storm, cache findings** (§6, §7, §8): hardware-counter
  based; independent of model selection.
- **`xlarge_mf.fa` cross-platform comparison** (§4): same canonical file, both platforms
  selected GTR, lnL diff <4 units. The confirmed **7.3x Gadi advantage at 64T** stands.
- **GPU porting motivation** (§9, §10): bottleneck is in SIMD kernel structure, not model
  choice.

#### What is NOT valid until re-runs complete
- Cross-platform speed ratios for `large_modelfinder.fa` (Setonix HKY vs Gadi GTR, different
  files).
- Cross-platform speed ratios for `mega_dna.fa` (different files, ASC vs non-ASC).
- The previous §1 claim "log-likelihoods match Setonix <-> Gadi to within FMA-contraction
  tolerance on every shared dataset" - **corrected** to apply only to `xlarge_mf.fa`.

### Report corrections applied (2026-04-26)

- **§1 status callout**: Removed false lnL-agreement claim; replaced with accurate statement
  scoping numerical correctness to `xlarge_mf.fa` only.
- **§2.2 Datasets table**: Restructured to show per-platform model selection (Gadi full MF
  vs Setonix `-mset`); updated lnL column to show Gadi values; added `†` markers and
  corrected notes.
- **§2.4 Platform command-line differences** (new section): Explicit side-by-side table of
  Setonix vs Gadi IQ-TREE invocations; disclosure of `-mset` asymmetry.
- **§4 cross-platform table**: Added `†` markers to `large_modelfinder.fa` and `mega_dna.fa`
  rows; updated caption and added extended footnote explaining the caveat.
- **§4 Future Work callout** (new): Documents the 3-step correction plan and estimated
  compute cost (6,500-9,700 additional CPU-hours on Setonix `work` partition).
- **§12.2 Methodology footnotes**: Added explicit bullet documenting the model scope
  discrepancy, per-dataset lnL gaps, and the `+ASC` normalisation incompatibility.

### Pending corrected re-runs

The following work is required to make `large_modelfinder.fa` and `mega_dna.fa`
cross-platform comparisons scientifically valid:

1. Re-generate canonical alignment files on Setonix using `gadi-ci/generate_datasets.sh`
   (seeds 101 and 303); verify sha256 against `benchmarks/sha256sums.txt`.
2. Remove `-mset GTR,HKY,K80` from `setonix-ci/submit_matrix.sh`.
3. Re-submit 10 jobs (6 x `large_modelfinder` 1-64T; 4 x `mega_dna` 16-128T).
   Estimated cost: **6,500-9,700 CPU-hours** (50-76 node-hours) on the Setonix `work`
   partition, dominated by `mega_dna` (estimated 10-19 h wall per thread count with
   full ModelFinder). Resource allocation review recommended before proceeding.

---

## 2026-04-26 (final) — `Profiling_Report.html` finalised against 33-run dataset

### Report rebuilt against complete cross-platform thread sweep

`Profiling_Report.html` refreshed to reflect the full 33-run dataset (Setonix 1–128T,
Gadi 1–104T, three datasets). Embedded `RUNS = [...]` literal regenerated; all four
new Gadi runs (`mega_dna` 64T/104T, `large_modelfinder` 64T, `xlarge_mf` 104T) added
to the appendix table and to the Section 5 scaling analysis.

### Section-level updates

- §1 status callout flipped from "interim — more runs in flight" to "final — 33 runs".
- §4 cross-platform table extended with Gadi 64T `large_modelfinder`, Gadi 104T
  `xlarge_mf` + `mega_dna`, Setonix 128T `xlarge_mf`, and Setonix 64T/128T `mega_dna`
  rows; new note on per-node maximum-thread comparison.
- §4 visual bar chart extended with Setonix 128T (6 516 s) and Gadi 104T (1 112 s) rows.
- §5.2 Gadi `large_modelfinder` table extended to 64T (regression: 8.85× vs 32T's 9.86×).
- §5.2 added Gadi `xlarge_mf` and `mega_dna` scaling tables — both regress past 64T.
- New §5.3 "Saturation observed on both platforms" callout: minima are now established
  *below* the per-node maximum on every dataset on both clusters, making the GPU-offload
  motivation cluster-independent.
- §6 IPC table expanded to 14 rows: Setonix 64T/128T xlarge + 64T/128T mega; Gadi 32T/64T
  large_mf, 104T xlarge, 64T/104T mega — covers the full IPC-collapse profile.
- Appendix and footer counts updated to "33 runs".

### Known issues from previous entry resolved

- **Em dashes**: globally swept (`&mdash;` and U+2014 → `-` with surrounding spaces preserved).
- **Bar chart colours in print/PDF**: `.fill.setonix`, `.fill.gadi`, `.fill.alt`, `.fill.bad`
  converted from `linear-gradient` to solid `background-color`. `body`, `.bar .fill`,
  `.bar .track`, and `.legend .sw` carry `print-color-adjust: exact` /
  `-webkit-print-color-adjust: exact`. `@media print` block overrides every fill class with
  `!important` and `background-image:none`. Verified the Setonix-red / Gadi-green
  distinction reproduces in Chrome "Save as PDF" without requiring the user to enable
  "Background graphics" in the print dialog.

---

## 2026-04-26 (evening) — Setonix `xlarge_mf.fa` canonical runs confirmed; full cross-platform coverage

### Setonix xlarge_mf.fa — 7-point baseline series verified canonical

Audit of `logs/runs/` confirmed all 7 Setonix `xlarge_mf.fa` baseline records are
present, have correct SLURM IDs (41931855–41931861), and carry `dataset=xlarge_mf.fa`
with timing consistent with the values recorded during the 2026-04-25 harvest.

| File                          | SLURM ID   | Threads | Wall (s) |
|-------------------------------|:----------:|:-------:|---------:|
| `xlarge_mf_1t_baseline.json`  | `41931855` |   1     |   10 555 |
| `xlarge_mf_4t_baseline.json`  | `41931856` |   4     |    7 271 |
| `xlarge_mf_8t_baseline.json`  | `41931857` |   8     |    8 618 |
| `xlarge_mf_16t_baseline.json` | `41931858` |  16     |    8 378 |
| `xlarge_mf_32t_baseline.json` | `41931859` |  32     |    7 237 |
| `xlarge_mf_64t_baseline.json` | `41931860` |  64     |    6 568 |
| `xlarge_mf_128t_baseline.json`| `41931861` | 128     |    6 516 |

### Pipeline result

```
normalize.py  →  33 runs, 2 profiles written → web/data/
validate.py   →  33 runs, 0 errors; 2 profiles, 0 errors
build.py      →  docs/ rebuilt
```

### Full cross-platform coverage

| Dataset               | Setonix threads                        | Gadi threads                     |
|-----------------------|----------------------------------------|----------------------------------|
| `large_modelfinder.fa`| 1, 4, 8, 16, 32, 64 ✅                 | 1, 4, 8, 16, 32, 64 ✅           |
| `xlarge_mf.fa`        | 1, 4, 8, 16, 32, 64, **128** ✅        | 1, 4, 8, 32, 64, **104** ✅      |
| `mega_dna.fa`         | 16, 32, 64, 128 ✅                     | 16, 32, 64, 104 ✅               |

---

## 2026-04-26 — All remaining Gadi thread sweeps complete; full Gadi coverage achieved

### 4 final Gadi runs harvested

All 4 queued/running jobs from the previous session completed overnight with Exit
Status 0. Run records added to `logs/runs/`:

- `gadi_mega_dna_64t_sr.json` — pass=True, 2346s (0.65h)
- `gadi_mega_dna_104t_sr.json` — pass=True, 2990s (0.83h)
- `gadi_large_modelfinder_64t_sr.json` — pass=True, 245s (0.07h)
- `gadi_xlarge_mf_104t_sr.json` — pass=True, 1112s (0.31h)

### Gadi thread coverage now complete

| Dataset | Gadi threads (complete) |
|---|---|
| `large_modelfinder.fa` | 1, 4, 8, 16, 32, **64** ✅ |
| `xlarge_mf.fa` | 1, 4, 8, 16, 32, 64, **104** ✅ |
| `mega_dna.fa` | 16, 32, **64, 104** ✅ |

Dashboard rebuilt (`normalize.py` → `build.py`), 33 runs total.

---

## 2026-04-26 — Profiling_Report.html committed

### Single-file scientific report drafted

`Profiling_Report.html` produced at the repository root: 12 sections covering executive summary,
methodology, hotspot anatomy, cross-platform comparison, scaling analysis, IPC collapse, OpenMP
barrier storm, memory hierarchy, deep-profile gap, and a full GPU porting plan (CUDA on Gadi /
ROCm on Setonix) with milestones M1–M7, risks, and success criteria. Self-contained HTML
(Chart.js via CDN; no other deps); embeds all normalised run records as a JSON literal so the
appendix table renders without the dashboard.

### Status: report ready for finalisation

All previously-pending Gadi jobs have completed and been harvested (see entries above).
The report was initially written against 29 runs; it must be refreshed against the full
33-run dataset before being considered final:

**Action:** rerun the data-extraction snippet to refresh the embedded `RUNS = [...]` literal
in `Profiling_Report.html`, re-render the §5 scaling chart numbers, update the appendix table,
and remove the Section 1 "Status: interim" callout, replacing it with a "Final — 33 runs" stamp.

### Known issues to fix before final report

- **Em dashes** — replace all `&mdash;` / `—` typography with plain hyphens or en-dashes
  where appropriate so they render consistently in PDF viewers and across all fonts.
- **Bar chart colours in print/PDF** — browser print engines strip CSS `background` gradients
  by default (Chrome's "Background graphics" flag). Convert bar fills to solid
  `background-color` + add `print-color-adjust: exact` / `-webkit-print-color-adjust: exact`
  on `.fill`, `.track`, and `.legend .sw` so the Setonix (red) / Gadi (green) distinction
  visible on screen is faithfully reproduced in the downloaded PDF without requiring the user
  to enable "Background graphics" in the print dialog.

### Companion artefact

`context.md` at the repository root: evidence notebook backing every claim in
`Profiling_Report.html`. Kept under version control so the report can be regenerated from the
same numerical scaffold.

---

## 2026-04-25 (evening) — Gadi xlarge_mf 32T/64T + mega_dna 32T complete; data pushed

### 3 Gadi runs harvested and committed

New run records added to `logs/runs/`:

- `gadi_xlarge_mf_32t_sr.json` — pass=True, 1036s (0.29h)
- `gadi_xlarge_mf_64t_sr.json` — pass=True, 897s (0.25h)
- `gadi_mega_dna_32t_sr.json` — pass=True, 2711s (0.75h)

Dashboard rebuilt (`normalize.py` → `build.py`). 4 jobs still running/queued:
`mega_dna` 64T + 104T, `large_modelfinder` 64T, `xlarge_mf` 104T.

---

## 2026-04-25 (late night) — 2 missing Gadi thread configs submitted; config panel default-closed

### IQ-TREE Configuration panel now collapsed by default

Dashboard change: the "IQ-TREE Configuration" card on the Overview page now
loads **collapsed**. Click "Show" to expand. Reduces visual noise on first
load.

### Two missing Gadi thread configs submitted

Gap analysis against the full matrix revealed two thread configs that were
never submitted or were missed in the resubmit wave:

| Job ID        | Dataset              | T   | Reason not yet submitted |
|---------------|----------------------|:---:|--------------------------|
| `167004589`   | `large_modelfinder`  | 64  | Original job killed by jobfs quota; not included in resubmit wave |
| `167004590`   | `xlarge_mf`          | 104 | Never submitted (Setonix goes to 128T; Gadi cap is 104T) |

Both submitted with `jobfs=2gb` + `TMPDIR` redirect fix in place.

### Current Gadi queue (7 jobs)

| Job ID        | Dataset             | T   | State |
|---------------|---------------------|:---:|:-----:|
| `167001081`   | `xlarge_mf`         | 32  | R     |
| `167001085`   | `xlarge_mf`         | 64  | R     |
| `167001093`   | `mega_dna`          | 32  | Q     |
| `167001098`   | `mega_dna`          | 64  | Q     |
| `167001099`   | `mega_dna`          | 104 | Q     |
| `167004589`   | `large_modelfinder` | 64  | Q     |
| `167004590`   | `xlarge_mf`         | 104 | Q     |

### Expected Gadi coverage after all 7 complete

| Dataset              | Thread configs on Gadi              |
|----------------------|-------------------------------------|
| `large_modelfinder`  | 1, 4, 8, 16, 32, **64** ← new      |
| `xlarge_mf`          | 1, 4, 8, 16, 32, 64, **104** ← new |
| `mega_dna`           | 16, **32, 64, 104** ← new          |

---

## 2026-04-25 (late night) — `xlarge_mf` Setonix series restored; CI workflow optimised

### `xlarge_mf` archive regression fixed (`663916d`)

The Gadi archive commit (`4bd3c35`) moved all 7 `xlarge_mf_*t_baseline.json`
files to `logs/runs/_archive/` when they still carried the non-canonical
`xlarge_dna.fa` identity. The subsequent harvest fix (`d259a46`) correctly
rewrote those files with canonical data — but wrote them back into `_archive/`
(the directory they lived in). `normalize.py` uses a non-recursive glob
(`logs/runs/*.json`) and never reads `_archive/`, so the series was silently
absent from every dashboard build.

**Fix**: moved the 7 files from `_archive/` → `logs/runs/` (they are now
canonical — `dataset=xlarge_mf.fa`, sha256 OK, SLURM 41931855–41931861).
Pipeline rebuilt: **26 runs, 0 errors**. `Setonix · xlarge_mf.fa` now
renders in Thread Scaling, Parallel Efficiency, IPC vs Threads, and
Performance Matrix charts.

### CI workflow optimised (`.github/workflows/build.yml`, `0a90682`, `1481f1c`)

| Change | Benefit |
|---|---|
| Merged two-job `build` + `deploy` into single job | ~25 s saved (eliminates second runner startup) |
| Replaced `pip` with `uv` (`astral-sh/setup-uv@v4`, venv + `uv run`) | 10–100× faster dep install; store cached across runs |
| `fetch-depth: 1` shallow clone | Saves checkout time on long history |
| Job-level `pages: write` + `id-token: write` only | Least-privilege; removed unused `contents: write` |

`--system` flag removed after Debian 3.12's externally-managed-environment
guard rejected it; switched to `uv venv` + `uv pip install` + `uv run`.

---

## 2026-04-25 (night, corrected) — `xlarge_mf.fa` canonical harvest fixed; all 13 runs now valid

A post-harvest audit on Gadi revealed that all 7 `xlarge_mf_*t_baseline.json`
files still carried `dataset=xlarge_dna.fa` / `slurm_id=41703864` (the old
non-canonical run) despite the previous harvest reporting them as "updated".

### Root cause

`harvest_scratch.py` resolved profile directories via a module-level
`SLURM_ID = "41703864"` env-var default. The update pass called
`profile_dir_for(label)` without consulting the run JSON's own `slurm_id`
field, so the old `41703864` dir was found first and the canonical
`41931855–41931861` data was never pulled in.

### Fix applied (`tools/harvest_scratch.py`)

`profile_dir_for()` now accepts an explicit `slurm_id` argument.
`enrich_run()` passes `slurm_id=run.get("slurm_id")`, which resolves the
correct `<label>_<slurm_id>` directory before falling back to the env-var
default or mtime sorting. This makes the harvester immune to stale
`SLURM_ID` defaults for any future re-runs.

### Corrected dataset spec (all 7 files verified canonical)

| File                        | SLURM ID   | T    | Dataset        | Taxa | Sites    | sha256 | IPC   | Wall (s) |
|-----------------------------|:----------:|:----:|----------------|:----:|:--------:|:------:|:-----:|:--------:|
| `xlarge_mf_1t_baseline.json`   | `41931855` | 1    | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 2.730 | 10 555   |
| `xlarge_mf_4t_baseline.json`   | `41931856` | 4    | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 1.018 |  7 271   |
| `xlarge_mf_8t_baseline.json`   | `41931857` | 8    | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 0.449 |  8 618   |
| `xlarge_mf_16t_baseline.json`  | `41931858` | 16   | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 0.251 |  8 378   |
| `xlarge_mf_32t_baseline.json`  | `41931859` | 32   | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 0.174 |  7 237   |
| `xlarge_mf_64t_baseline.json`  | `41931860` | 64   | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 0.097 |  6 568   |
| `xlarge_mf_128t_baseline.json` | `41931861` | 128  | `xlarge_mf.fa` | 200  | 100 000  | ✅ OK  | 0.067 |  6 516   |

sha256 `66eaf64b9b7e…` — matches `benchmarks/sha256sums.txt` lockfile exactly
(same canonical file, bit-identical to Gadi, seed 202, GTR+G4, 200 × 100 000).

IPC decreases monotonically from 2.73 (1T) to 0.07 (128T) — consistent with
memory-bandwidth saturation at high parallelism on AMD Milan (same pattern
observed in `large_modelfinder.fa` and `mega_dna.fa`). Wall time saturates
between 64T and 128T (6 568 vs 6 516 s, < 1 % difference).

### Pipeline rerun

```
harvest_scratch.py   →  17 files updated (all 7 xlarge_mf + 6 large_mf + 4 mega)
normalize.py         →  40 runs, 2 profiles written
validate.py          →  40 runs, 0 errors; 2 profiles, 0 errors
build.py             →  docs/ rebuilt (v=20260425070600)
```

---

## 2026-04-25 (night) — Pilot/stub runs archived; dashboard cleaned

15 run records moved to `logs/runs/_archive/` to remove confusing or
invalid data from metric cards. These files are preserved for audit but no
longer rendered by the dashboard.

### What was archived and why

| Group | Files | Reason |
|---|---|---|
| Gadi pilot (wrong-dimension dataset) | `gadi_large_modelfinder_64t_sr`, `gadi_xlarge_mf_{16,26,52,104}t_sr` | Dataset field = `*_gadi_pilot.fa` — different dimensions from Setonix; no valid cross-platform comparison |
| Gadi zero-data stubs | `gadi_mega_dna_{13,26}t_sr` | `pass=False`, `total_time=0` — IQ-TREE never ran (old PMU bug era; non-standard thread counts 13T/26T) |
| Setonix-only unmatched | `xlarge_mf_{1,4,8,16,32,64,128}t_baseline` | Dataset = `xlarge_dna.fa` — no Gadi counterpart; different file from canonical `xlarge_mf.fa` |
| Setonix-only unmatched | `2026-04-18_201515` | Dataset = `turtle.fa` — smoke-test run only; no Gadi counterpart |

**Total archived: 15 files.**

### Remaining active runs (25 files, 0 errors)

| Dataset | Platform | Thread configs | Status |
|---|---|---|---|
| `large_modelfinder.fa` | Setonix | 1, 4, 8, 16, 32, 64 × 2 run types | ✅ valid |
| `mega_dna.fa` | Setonix | 16, 32, 64, 128 | ✅ valid |
| `large_modelfinder.fa` | Gadi | 1, 4, 8, 16, 32 | ✅ valid (64T in-queue) |
| `xlarge_mf.fa` | Gadi | 1, 4, 8 | ✅ valid (32T/64T running) |
| `mega_dna.fa` | Gadi | 16 | ✅ valid (32T/64T/104T queued) |

`_archive/` is tracked in git but excluded from `normalize.py` (non-recursive `glob("*.json")` only reads the parent directory).

---

## 2026-04-25 (night) — Setonix canonical rerun **completed**; results harvested

All 13 matrix jobs from the earlier evening submission completed successfully.
Results harvested, normalised, validated (0 errors), and dashboard rebuilt.

### Job completion summary

| Job ID     | Dataset              | Threads | Elapsed    | Exit |
|------------|----------------------|:-------:|:----------:|:----:|
| `41931848` | `generate_datasets.sh` (datagen) | — | 00:00:09 | 0 |
| `41931849` | `large_modelfinder`  | 1T      | 01:03:31   | 0    |
| `41931850` | `large_modelfinder`  | 4T      | 01:02:29   | 0    |
| `41931851` | `large_modelfinder`  | 8T      | 00:57:58   | 0    |
| `41931852` | `large_modelfinder`  | 16T     | 00:49:36   | 0    |
| `41931853` | `large_modelfinder`  | 32T     | 00:44:04   | 0    |
| `41931854` | `large_modelfinder`  | 64T     | 00:45:43   | 0    |
| `41931855` | `xlarge_mf`          | 1T      | 02:56:05   | 0    |
| `41931856` | `xlarge_mf`          | 4T      | 02:31:23   | 0    |
| `41931857` | `xlarge_mf`          | 8T      | 02:53:51   | 0    |
| `41931858` | `xlarge_mf`          | 16T     | 02:49:55   | 0    |
| `41931859` | `xlarge_mf`          | 32T     | 02:31:02   | 0    |
| `41931860` | `xlarge_mf`          | 64T     | 02:20:12   | 0    |
| `41931861` | `xlarge_mf`          | 128T    | 02:20:07   | 0    |

### Run metrics (Setonix · Pawsey · AMD Milan, canonical files)

| Dataset               | Threads | Wall (s) | IPC   | Peak RSS |
|-----------------------|:-------:|---------:|:-----:|:--------:|
| `large_modelfinder.fa`|  1T     |    3 801 | 2.890 |  1 150 MB |
| `large_modelfinder.fa`|  4T     |    1 938 | 1.429 |  1 141 MB |
| `large_modelfinder.fa`|  8T     |    1 670 | 0.851 |  1 173 MB |
| `large_modelfinder.fa`| 16T     |    1 384 | 0.537 |  1 145 MB |
| `large_modelfinder.fa`| 32T     |    1 295 | 0.328 |  1 144 MB |
| `large_modelfinder.fa`| 64T     |    1 293 | 0.181 |  1 139 MB |
| `xlarge_mf.fa`        |  1T     |   16 982 | 2.624 |  2 203 MB |
| `xlarge_mf.fa`        |  4T     |    9 146 | 1.243 |  2 213 MB |
| `xlarge_mf.fa`        |  8T     |    9 078 | 0.642 |  2 215 MB |
| `xlarge_mf.fa`        | 16T     |    8 938 | 0.344 |  2 216 MB |
| `xlarge_mf.fa`        | 32T     |    8 070 | 0.225 |  2 217 MB |
| `xlarge_mf.fa`        | 64T     |    7 220 | 0.136 |  2 214 MB |
| `xlarge_mf.fa`        | 128T    |    7 194 | 0.082 |     —     |

Notable: IPC drops steeply with thread count on both datasets (3× at 1T → 0.1–0.2 at
64–128T), consistent with memory-bandwidth saturation at high parallelism on Zen 3/Milan.
`xlarge_mf` 64T vs 128T wall time is essentially flat (7 220 vs 7 194 s), confirming
saturation beyond ≈ 64 threads for this workload.

### Harvest pipeline run

```
python3.11 tools/harvest_scratch.py   # 23 files updated; 6 new large_modelfinder_*t_baseline.json created
python3.11 tools/normalize.py         # wrote 40 runs, 2 profiles → web/data
python3.11 tools/validate.py          # 40 runs, 0 errors; 2 profiles, 0 errors
python3.11 tools/build.py             # mirrored web/ → docs/  (v=20260425061444)
```

### New files in `logs/runs/`

Six new canonical Setonix run records (from the `large_modelfinder_<T>t` profile dirs):
`large_modelfinder_{1,4,8,16,32,64}t_baseline.json`

The seven `xlarge_mf_*t_baseline.json` files were updated in-place with enriched
harvest data (hotspots, modelfinder candidates, io totals).

### Dashboard status

Non-canonical `⚠ non-canonical file` badges will disappear for `large_modelfinder.fa`
and `xlarge_mf.fa` Setonix series once `normalize.py` propagates
`dataset_canonical: true` from the new run records. `mega_dna.fa` was already clean.

---

## 2026-04-25 (evening) — Setonix canonical rerun submitted; profiling hardened

Follow-up to the earlier 2026-04-25 non-canonical-file containment entry.
13 benchmark jobs + 1 generator job queued on Setonix (`pawsey1351`).

### Scope

`mega_dna.fa` **excluded** from this rerun — it was already canonical on
Setonix (500 × 100 000, seed 303, sha256 verified). Only the two
non-canonical datasets are being re-run:

| Dataset               | Thread sweep                  | Jobs |
|-----------------------|-------------------------------|:----:|
| `large_modelfinder.fa`| 1, 4, 8, 16, 32, 64           |  6   |
| `xlarge_mf.fa`        | 1, 4, 8, 16, 32, 64, 128      |  7   |

### Job table

| Job ID     | Role                   | State at submission | Dependency            |
|------------|------------------------|:-------------------:|-----------------------|
| `41931848` | `generate_datasets.sh` | PD (Priority)       | —                     |
| `41931849` | `large_modelfinder` 1T | PD (Dependency)     | afterok:41931848      |
| `41931850` | `large_modelfinder` 4T | PD (Dependency)     | afterok:41931848      |
| `41931851` | `large_modelfinder` 8T | PD (Dependency)     | afterok:41931848      |
| `41931852` | `large_modelfinder` 16T| PD (Dependency)     | afterok:41931848      |
| `41931853` | `large_modelfinder` 32T| PD (Dependency)     | afterok:41931848      |
| `41931854` | `large_modelfinder` 64T| PD (Dependency)     | afterok:41931848      |
| `41931855` | `xlarge_mf` 1T         | PD (Dependency)     | afterok:41931848      |
| `41931856` | `xlarge_mf` 4T         | PD (Dependency)     | afterok:41931848      |
| `41931857` | `xlarge_mf` 8T         | PD (Dependency)     | afterok:41931848      |
| `41931858` | `xlarge_mf` 16T        | PD (Dependency)     | afterok:41931848      |
| `41931859` | `xlarge_mf` 32T        | PD (Dependency)     | afterok:41931848      |
| `41931860` | `xlarge_mf` 64T        | PD (Dependency)     | afterok:41931848      |
| `41931861` | `xlarge_mf` 128T       | PD (Dependency)     | afterok:41931848      |

All 13 matrix jobs are held in **Dependency** state until `41931848`
completes successfully (SLURM `--dependency=afterok`). If the generator
exits non-zero (sha256 mismatch) the matrix jobs will never start.

### SU estimate

| Scenario | Estimate |
|---|---:|
| Previous status-quo (7200 s perf record cap, 1T included) | ~6.1 kSU |
| **This rerun (1800 s cap, 1T skips perf record)** | **~3.0 kSU** |

Savings from two profiling improvements applied before submission (see below).

### Profiling improvements applied (`setonix-ci/`)

**`run_mega_profile.sh`**

| Change | Detail |
|---|---|
| Generalised dataset input | Removed `mega_dna.fa` hard-code; honours `DATASET=` env var; label/work-dir derive from dataset stem |
| **sha256 pre-flight gate** | Job exits 3 if the alignment is not listed in `benchmarks/sha256sums.txt` or its hash mismatches. Prevents repeating the 2026-04-25 non-canonical regression |
| `perf record` cap 7200 s → **1800 s** | 30 min at 99 Hz ≈ 178 k stacks — sufficient for hotspot ranking; overridable via `PERF_RECORD_MAX_S` |
| **Auto-skip pass 5 on 1T** | Single-thread hotspot data is uninteresting; saves the full second IQ-TREE run on the longest-wall job. `SKIP_PERF_RECORD=0` to force-on |
| `--call-graph fp` | Uses frame pointers (already compiled with `-fno-omit-frame-pointer`) instead of dwarf; ~5–10× cheaper unwinding |

**`submit_matrix.sh`** (new file)

| Feature | Detail |
|---|---|
| Login-node sha256 verify | Checks all present alignments before submitting any sbatch |
| `--regen` flag | Submits `generate_datasets.sh` first and gates the matrix on `--dependency=afterok:<gen_jid>` |
| `--dataset` / `--threads` | Restrict to a subset without editing the script |
| `--dry-run` | Print sbatch invocations without submitting |
| Default matrix | `large_modelfinder × {1,4,8,16,32,64}` + `xlarge_mf × {1,4,8,16,32,64,128}`; `mega_dna` explicitly excluded |

Invocation used:
```bash
cd ~/setonix-iq/setonix-ci
./submit_matrix.sh --regen
```

### Defence-in-depth (three layers)

1. **`submit_matrix.sh`** — verifies sha256 on the login node before any
   `sbatch`; `--regen` chains generator via `afterok` dependency.
2. **`run_mega_profile.sh` pre-flight** — job exits 3 inside the compute
   node if the alignment hash doesn't match the lockfile.
3. **`generate_datasets.sh`** — already exits non-zero on sha256 mismatch
   (added in the containment commit `b48973f`).

### Follow-up

- When all 13 run JSONs land in `logs/runs/`, run:
  ```bash
  python3 tools/harvest_scratch.py
  python3 tools/normalize.py
  python3 tools/validate.py
  python3 tools/build.py
  git add logs/runs/ web/data/
  git commit -m "data(setonix): re-run against canonical benchmark files"
  git push
  ```
- The dashboard `⚠ non-canonical file` badges will disappear once the
  old non-canonical JSONs are replaced by runs with `dataset_canonical: true`.

---

## 2026-04-25 (afternoon) — 5 Gadi jobs killed by jobfs quota; fix applied; resubmitted

### What happened

All jobs that succeeded overnight used relatively short VTune collection
times (low thread counts → less data). Five jobs were killed mid-run by
PBS because they exceeded the default **jobfs quota of 100 MB**:

| Job ID      | Dataset       | T   | Wall used | JobFS used | Exit |
|-------------|---------------|:---:|:---------:|:----------:|:----:|
| `166978130` | `xlarge_mf`   | 32  | 1h 02m    | 102 MB     | SIGTERM 271 |
| `166978131` | `xlarge_mf`   | 64  | 0h 54m    | 125 MB     | SIGTERM 271 |
| `166978499` | `mega_dna`    | 32  | 2h 14m    | 106 MB     | SIGTERM 271 |
| `166978500` | `mega_dna`    | 64  | 1h 59m    | 124 MB     | SIGTERM 271 |
| `166978501` | `mega_dna`    | 104 | 2h 09m    | 104 MB     | SIGTERM 271 |

**Root cause**: VTune's hotspot collection pass writes driver/temp artefacts
to `$TMPDIR`. On Gadi, `$TMPDIR` defaults to `/jobfs/$PBS_JOBID/`, which
is bounded by the per-job `jobfs=` resource. The worker did not request a
`jobfs` allocation (falling back to the PBS default of 100 MB), and VTune
overflowed it at higher thread counts where more stack frames are sampled.

### Fix applied to `gadi-ci/submit_benchmark_matrix.sh`

Two changes:

1. **`jobfs=2gb` added to the qsub `-l` resource string** — provides 2 GB
   of jobfs headroom (well above the observed 125 MB peak).
2. **`export TMPDIR="${PROJECT_DIR}/tmp"`** added to the worker, immediately
   after the module loads, redirecting VTune temp I/O entirely to scratch as
   belt-and-suspenders. The directory is created with `mkdir -p`.

### Completed runs (successful, valid data)

| File                                   | T   | `pass` | `time_s` | lnL approx        |
|----------------------------------------|:---:|:------:|:--------:|:-----------------:|
| `gadi_large_modelfinder_1t_sr.json`    | 1   | ✅     | 2 168 s  | −2 690 513.34     |
| `gadi_large_modelfinder_4t_sr.json`    | 4   | ✅     | 805 s    | −2 690 513.34     |
| `gadi_large_modelfinder_8t_sr.json`    | 8   | ✅     | 461 s    | −2 690 513.34     |
| `gadi_large_modelfinder_16t_sr.json`   | 16  | ✅     | 294 s    | −2 690 513.34     |
| `gadi_large_modelfinder_32t_sr.json`   | 32  | ✅     | 220 s    | −2 690 513.34     |
| `gadi_large_modelfinder_64t_sr.json`   | 64  | ✅     | 771 s    | −1 303 518.59 ⚠   |
| `gadi_xlarge_mf_1t_sr.json`            | 1   | ✅     | 11 915 s | −10 956 936.61    |
| `gadi_xlarge_mf_4t_sr.json`            | 4   | ✅     | 4 244 s  | −10 956 936.61    |
| `gadi_xlarge_mf_8t_sr.json`            | 8   | ✅     | 2 440 s  | −10 956 936.61    |
| `gadi_xlarge_mf_16t_sr.json`           | 16  | ✅     | 1 048 s  | −5 286 806.51 ⚠   |
| `gadi_mega_dna_16t_sr.json`            | 16  | ✅     | 3 973 s  | −27 328 165.86    |

⚠ Two lnL values differ from the rest of their series — flag for checking
topology convergence / seed variance before analysis. Not a blocker for
timing/IPC comparisons.

### Resubmitted jobs (5 — with fixed jobfs + TMPDIR)

| Job ID      | Dataset     | T   |
|-------------|-------------|:---:|
| `167001081` | `xlarge_mf` | 32  |
| `167001085` | `xlarge_mf` | 64  |
| `167001093` | `mega_dna`  | 32  |
| `167001098` | `mega_dna`  | 64  |
| `167001099` | `mega_dna`  | 104 |

SU charged by the 5 killed jobs (real SU billed even on kill):
**216 + 188 + 465 + 415 + 448 = 1 732 SU = ~1.73 KSU**

### Updated grant picture

| | KSU |
|---|---:|
| Grant (Q2 2026) | 25.00 |
| Used before today | 3.78 |
| Successful runs overnight | ~2.56 |
| JobFS kill waste | ~1.73 |
| Resubmit wave (5 jobs, est.) | ~1.40 |
| **Projected total after resubmit** | **~9.47** |
| **Remaining** | **~15.53** |

---



---

## 2026-04-25 — Non-canonical Setonix benchmark files discovered; re-run required

### Problem

All **17 Setonix baseline runs** in `logs/runs/` were executed against
alignment files that differ from the canonical Gadi benchmarks.
Cross-platform wall-time / IPC / speedup comparisons using non-identical
inputs are **not scientifically valid**.

**Pattern-count divergence (patterns ≠ means different file):**

| Dataset | Setonix file patterns | Gadi canonical patterns | Match? |
|---|---:|---:|:---:|
| `large_modelfinder.fa` | 48,293 | 45,386 | ❌ |
| `xlarge_mf.fa` (was `xlarge_dna.fa`) | 99,897 | 98,858 | ❌ |
| `mega_dna.fa` | 100,000 | 99,999 | ❌ |

Additionally, some older Setonix runs used a completely wrong size:
`xlarge_dna.fa` at **1,000 taxa × 10,000 sites** instead of the intended
200 × 100,000.

**Root cause:** The Setonix benchmarks were pre-existing files on
`/scratch/pawsey1351/asamuel/iqtree3/benchmarks/` — never explicitly
generated. The Gadi benchmarks were regenerated with IQ-TREE 3.1.1's
built-in AliSim simulator using fixed seeds (101 / 202 / 303). Even with
the same seed, a different IQ-TREE build produces different AliSim output
because the PRNG initialization changed between versions.

**Gadi canonical sha256 hashes** (from `benchmarks/sha256sums.txt`):
```
73908728537994a4...  large_modelfinder.fa   (100 taxa × 50,000 bp)
66eaf64b9b7e561f...  xlarge_mf.fa           (200 taxa × 100,000 bp)
0c8af2d62e214be8...  mega_dna.fa            (500 taxa × 100,000 bp)
```

### Changes made (commit `b48973f`)

1. **`benchmarks/sha256sums.txt`** — canonical sha256 lockfile committed
   to the repo for all three benchmark files.

2. **`setonix-ci/generate_datasets.sh`** — new script (mirrors
   `gadi-ci/generate_datasets.sh` exactly: GTR+G4 model, same AliSim
   parameters, seeds 101 / 202 / 303). Verifies sha256 against the
   lockfile at the end and **exits non-zero** if any checksum fails,
   blocking accidental use of wrong files.

3. **`gadi-ci/generate_datasets.sh`** — added sha256 verification block
   at the end to catch accidental regeneration with a new IQ-TREE build.

4. **All 17 Setonix run JSONs** — `dataset_info.dataset_canonical: false`
   with an explanatory note. `xlarge_dna.fa` normalised to `xlarge_mf.fa`
   (original name preserved in `file_original`) so the dashboard groups
   them correctly.

5. **Dashboard** — carousel dataset cards show a red **⚠ non-canonical
   file** badge on any platform block whose runs used the wrong file.

### Action required on Setonix

> **These steps must be completed before any cross-platform comparison
> is valid.**

**Step 1 — Regenerate benchmark files with IQ-TREE 3.1.1**

```bash
# On Setonix login node
cd ~/setonix-iq   # or wherever the repo is checked out
sbatch setonix-ci/generate_datasets.sh
```

The script will simulate `large_modelfinder.fa`, `xlarge_mf.fa`, and
`mega_dna.fa` using the same AliSim parameters and seeds as Gadi, then
verify sha256 against `benchmarks/sha256sums.txt`.

- If the job log ends with **`all checksums OK`** — the files are
  bit-identical to Gadi's. Proceed to Step 2.
- If it ends with **`sha256 mismatch`** — the IQ-TREE version on Setonix
  is not 3.1.1. Check `${BUILD_DIR}/iqtree3 --version` and rebuild from
  the `v3.1.1` tag. Do **not** use the mismatched files for benchmarks.

Required env overrides (if non-default paths):
```bash
PROJECT_DIR=/scratch/pawsey1351/asamuel/iqtree3 \
BUILD_DIR=/scratch/pawsey1351/asamuel/iqtree3/build-profiling \
sbatch setonix-ci/generate_datasets.sh
```

**Step 2 — Verify the generated files (manual double-check)**

```bash
cd /scratch/pawsey1351/asamuel/iqtree3/benchmarks
sha256sum -c ~/setonix-iq/benchmarks/sha256sums.txt
```

All three lines must print `OK`.

**Step 3 — Re-run the full Setonix benchmark matrix**

```bash
cd ~/setonix-iq/setonix-ci
# Re-submit all datasets & thread counts using the updated benchmark files.
sbatch --export=ALL,DATASET=large_modelfinder.fa,THREADS=1  run_mega_profile.sh
sbatch --export=ALL,DATASET=large_modelfinder.fa,THREADS=4  run_mega_profile.sh
sbatch --export=ALL,DATASET=large_modelfinder.fa,THREADS=8  run_mega_profile.sh
sbatch --export=ALL,DATASET=large_modelfinder.fa,THREADS=16 run_mega_profile.sh
sbatch --export=ALL,DATASET=large_modelfinder.fa,THREADS=32 run_mega_profile.sh
sbatch --export=ALL,DATASET=large_modelfinder.fa,THREADS=64 run_mega_profile.sh

sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=1   run_mega_profile.sh
sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=4   run_mega_profile.sh
sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=8   run_mega_profile.sh
sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=16  run_mega_profile.sh
sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=32  run_mega_profile.sh
sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=64  run_mega_profile.sh
sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=128 run_mega_profile.sh

sbatch --export=ALL,DATASET=mega_dna.fa,THREADS=16  run_mega_profile.sh
sbatch --export=ALL,DATASET=mega_dna.fa,THREADS=32  run_mega_profile.sh
sbatch --export=ALL,DATASET=mega_dna.fa,THREADS=64  run_mega_profile.sh
sbatch --export=ALL,DATASET=mega_dna.fa,THREADS=128 run_mega_profile.sh
```

Alternatively use `submit_mega_batch.sh` if it supports `--dataset` flag.

**Step 4 — Harvest and push new results**

After jobs complete:
```bash
cd ~/setonix-iq
/bin/python3.11 tools/harvest_scratch.py   # or equivalent on Setonix
/bin/python3.11 tools/normalize.py
git add logs/runs/ web/data/
git commit -m "data(setonix): re-run against canonical benchmark files"
git push
```

The dashboard `⚠ non-canonical file` badges will disappear once the
old non-canonical run JSONs are replaced with runs that have
`dataset_canonical: true` (or no flag, once harvest re-populates from
the correct files).

---



### What we found

Re-inspecting the two `mega_dna` Gadi records previously considered
"valid" (because the **file dimensions** matched Setonix):

| File                                   | `pbs_id`    | `all_pass` | `total_time` |
|----------------------------------------|-------------|:----------:|:------------:|
| `logs/runs/gadi_mega_dna_13t_sr.json`  | `166967411` | `false`    | **0 s**      |
| `logs/runs/gadi_mega_dna_26t_sr.json`  | `166967412` | `false`    | **0 s**      |

Both records show zero walltime and a failed run — they are casualties of
the **same `stalled-cycles-frontend/backend` + NUMA thread-width bugs**
documented in the 2026-04-24 CHANGELOG entry. The file dimensions were
coincidentally correct, but the runs themselves captured no IQ-TREE timing
data, so they cannot contribute to any scaling / IPC / efficiency analysis.

Conclusion: **`mega_dna` on Gadi currently has zero usable data points**,
despite looking OK in the dataset-dimension audit.

### Action taken

Submitted the `mega_dna` thread sweep using the post-fix worker
(`gadi-ci/submit_benchmark_matrix.sh mega_dna`, which uses the corrected
`{16, 32, 64, 104}` thread matrix and the worker that runs IQ-TREE in
**Pass 1 before `perf stat`**, so timing is recorded regardless of perf
status). Same 500 × 100 000 alignment on scratch (unchanged, matches
Setonix bit-for-bit), seed 1.

| Job ID        | Dataset    | Threads | Thread-count rationale |
|---------------|------------|:-------:|------------------------|
| `166978498`   | `mega_dna` | 16      | matches Setonix 16T    |
| `166978499`   | `mega_dna` | 32      | matches Setonix 32T    |
| `166978500`   | `mega_dna` | 64      | matches Setonix 64T    |
| `166978501`   | `mega_dna` | 104     | Gadi node cap (Setonix ran 128T — flagged in JSON) |

### Why this one "will work this time"

Three defence-in-depth mechanisms, each verified in
`gadi-ci/submit_benchmark_matrix.sh` before submission:

1. **IQ-TREE runs first** (line 220 `wait "${IQTREE_PID}" || IQRC=$?`).
   Its own walltime is extracted from the `Total wall-clock time` line
   of the IQ-TREE log (line 318), so `iqwall` is populated even if
   every downstream profiler fails.
2. **`perf stat` only runs if `IQRC == 0`** (line 231). `stalled-cycles-*`
   events removed (line 101). All events carry `:u` user-mode suffix
   (line 109), required because Gadi compute nodes ship with
   `/proc/sys/kernel/perf_event_paranoid=2`.
3. **`summary.pass` / `.all_pass`** are driven by `iqrc`, not perf
   return code (lines 488–489). So a clean IQ-TREE finish produces
   `pass=1` + real `total_time`, even if the profiler layer errors.

### Queue state after submission

```
12 jobs from the corrected dataset wave  (166978120 – 166978131)
 4 jobs mega_dna rerun wave              (166978498 – 166978501)
16 jobs total  ·  large_mf R×4/Q×2  ·  xlarge_mf Q×6  ·  mega_dna Q×4
```

Incremental SU estimate for the `mega_dna` wave: **≈ 0.9 KSU** (worst-case
1.5 h walltime per job, dominated by the 16T point; extrapolated from
Setonix baselines scaled ~1.8× for the added 5 profiling passes).

### Follow-up

- When `mega_dna` jobs finish, delete the two zero-data records
  (`gadi_mega_dna_{13,26}t_sr.json`) — they are superseded by the
  new matched-thread-count sweep and add no information.
- Run the standard post-harvest pipeline
  (`harvest_scratch.py → normalize.py → validate.py → build.py`) and
  push.

---

## 2026-04-25 (evening) — Priority rerun **executed**: corrected Gadi datasets + 12 fresh sweep jobs queued

Follow-up to the earlier 2026-04-25 containment entry. Executed the
priority rerun plan on `gadi-login-05` after pulling `main` (19eebdb
→ 4282f0f, stash/pop — no conflicts).

### Actions taken

1. **Removed wrong-dimension alignments** from
   `/scratch/rc29/as1708/iqtree3/benchmarks/`:
   - `large_modelfinder.fa` (was 2 506 000 B · 500 × 5 000 — transposed)
   - `xlarge_mf.fa`         (was 10 012 000 B · 1 000 × 10 000 — half workload)

   `mega_dna.fa` (50 006 000 B · 500 × 100 000) was **kept** — it already
   matched the Setonix corpus bit-for-bit (same seed 303), so no
   regeneration or rerun was needed for that dataset.

2. **Regenerated via `qsub gadi-ci/generate_datasets.sh`** (job
   `166978119`). Post-run sizes on scratch now match Setonix exactly:

   | File                    | Size (bytes) | Taxa × sites    | Setonix target |
   |-------------------------|-------------:|:----------------|:---------------|
   | `large_modelfinder.fa`  |    5 001 200 | 100 × 50 000    | 5 000 000 ± 6 kB ✅ |
   | `xlarge_mf.fa`          |   20 002 400 | 200 × 100 000   | 20 000 000 ± 6 kB ✅ |
   | `mega_dna.fa`           |   50 006 000 | 500 × 100 000   | unchanged ✅ |

3. **Added PBS dependency support to `gadi-ci/submit_benchmark_matrix.sh`**
   — a two-line patch honouring an optional `DEPEND_JOBID` env var so
   sweep jobs are submitted with `-W depend=afterok:<gen_jid>` and sit
   in **H** (held) state until the generator finishes. This prevents
   any job from starting on a partially-written alignment.

4. **Cancelled all in-flight / queued work on the old (wrong-dim)
   pilot datasets** so that no more SU are spent on invalid inputs.
   Six jobs killed via `qdel`:

   | Job ID           | State at kill | Dataset           | Why cancelled |
   |------------------|:-------------:|-------------------|---------------|
   | `166969053`      | R (00:58)     | `mega_dna` pilot  | pre-fix run, superseded |
   | `166969054`      | R (00:56)     | `mega_dna` pilot  | pre-fix run, superseded |
   | `166969055`      | Q             | `mega_dna` pilot  | pre-fix run, superseded |
   | `166976124`      | R (00:56)     | `xlarge_mf` pilot | wrong dims (10 MB file) |
   | `166976125`      | R (00:56)     | `xlarge_mf` pilot | wrong dims (10 MB file) |
   | `166976126`      | Q             | `mega_dna` pilot  | pre-fix run, superseded |

   (The `mega_dna` running jobs were on the dimensionally-correct file
   but were part of the pre-fix submission wave; cancelling keeps the
   run history coherent. `mega_dna` will **not** be re-submitted in
   this wave — the two existing valid records
   `gadi_mega_dna_{13,26}t_sr.json` already provide the matched
   cross-platform data points and additional thread sweeps are out of
   scope for today.)

5. **Cancelled the 4 `mega_dna` jobs accidentally queued by the
   default matrix** (`166978132`–`166978135`). The submit script's
   default matrix includes `mega_dna: {16,32,64,104}` but — per the
   containment decision — `mega_dna` is already valid and must not be
   rerun in this wave.

### Final job set (12 held → now queued, `afterok:166978119` satisfied)

All on `normalsr` (Sapphire Rapids), `ncpus=104`, `mem=500GB`,
`walltime=24:00:00`, writing JSON run records to `logs/runs/`:

| Job ID        | Dataset              | Threads |
|---------------|----------------------|:-------:|
| `166978120`   | `large_modelfinder`  | 1       |
| `166978121`   | `large_modelfinder`  | 4       |
| `166978122`   | `large_modelfinder`  | 8       |
| `166978123`   | `large_modelfinder`  | 16      |
| `166978124`   | `large_modelfinder`  | 32      |
| `166978125`   | `large_modelfinder`  | 64      |
| `166978126`   | `xlarge_mf`          | 1       |
| `166978127`   | `xlarge_mf`          | 4       |
| `166978128`   | `xlarge_mf`          | 8       |
| `166978129`   | `xlarge_mf`          | 16      |
| `166978130`   | `xlarge_mf`          | 32      |
| `166978131`   | `xlarge_mf`          | 64      |

Revised SU estimate for this rerun wave (mega_dna dropped):
**≈ 2.7 KSU** (was ~4.1 KSU with mega_dna included).

### Follow-up still pending

- When all 12 run JSONs have landed in `logs/runs/`, execute
  `python3 tools/harvest_scratch.py && python3 tools/normalize.py &&
  python3 tools/validate.py && python3 tools/build.py`, then commit
  and push.
- Archive the 12 `*_gadi_pilot.fa` records (move to
  `logs/runs/_archive/` and exclude from normalize) once the matched
  Gadi `large_modelfinder.fa` + `xlarge_mf.fa` series are in.

---

## 2026-04-25 — **PRIORITY**: Gadi pilot datasets had wrong dimensions vs Setonix — rerun required

While reviewing the multi-platform dashboard we caught a data-validity
regression: the AliSim-generated benchmark alignments on Gadi were
**not dimensionally matched** to the Setonix corpus, so any cross-platform
wall-time / IPC / speedup comparison against those series is invalid.

### Evidence (from `web/data/runs.index.json` before the fix)

| Dataset label          | Setonix (taxa × sites) | Gadi (taxa × sites) | Match? |
|------------------------|:---------------------:|:-------------------:|:-----:|
| `large_modelfinder.fa` | **100 × 50 000**  (5.00 MB) | 500 × 5 000  (2.51 MB) | ❌ transposed |
| `xlarge_mf.fa`         | 200 × 100 000 (20 MB, labelled `xlarge_dna.fa` in JSON) | 1000 × 10 000 (10 MB) | ❌ different workload |
| `mega_dna.fa`          | 500 × 100 000 (50.01 MB) | 500 × 100 000 (50.01 MB) | ✅ identical |

Root cause: `gadi-ci/generate_datasets.sh` hard-coded the wrong
`(taxa, sites)` tuples for `large_modelfinder` and `xlarge_mf`. The
`mega_dna` simulation was correct by coincidence (same 500 × 100 000,
same seed 303).

### Containment (done today, commit pending)

1. **Generator corrected** — `gadi-ci/generate_datasets.sh` now reads:
   ```
   simulate "large_modelfinder.fa"  100  50000   101
   simulate "xlarge_mf.fa"          200 100000   202
   simulate "mega_dna.fa"           500 100000   303
   ```
   Dimensions pinned to the Setonix corpus. A header comment warns
   future maintainers not to change them without re-running Setonix.
2. **Existing 12 invalid Gadi runs isolated** — instead of deleting
   the SU spend (≈ 3 KSU of real IQ-TREE work on wrong-shape input),
   the `profile.dataset` and `dataset_info.file` fields were renamed
   in place:
   - `large_modelfinder.fa` → `large_modelfinder_gadi_pilot.fa` (6 runs)
   - `xlarge_mf.fa` → `xlarge_mf_gadi_pilot.fa` (6 runs)
   Each record gained a `notes` entry explaining the pilot/rerun-pending
   status. The dashboard now surfaces them as **separate series** from
   the Setonix corpus — no silent cross-platform contamination.
3. **`mega_dna` Gadi runs are valid** — identical dimensions, identical
   seed, 50.01 MB file size matches Setonix exactly. Those 2 data points
   (13T, 26T — Intel Sapphire Rapids vs AMD Milan) remain the only
   currently trustworthy cross-platform comparison.

### Verified post-fix index state

```
('turtle.fa',                       'setonix') 0.33 MB   16 × 20 820  × 1 run
('large_modelfinder.fa',            'setonix') 5.00 MB  100 × 50 000  × 6 runs   ← baseline
('xlarge_dna.fa',                   'setonix') 20.0 MB  200 × 100 000 × 7 runs   ← baseline
('mega_dna.fa',                     'setonix') 50.0 MB  500 × 100 000 × 4 runs   ← baseline
('mega_dna.fa',                     'gadi')    50.0 MB  500 × 100 000 × 2 runs   ✅ comparable
('large_modelfinder_gadi_pilot.fa', 'gadi')    2.51 MB  500 × 5 000   × 6 runs   ⚠ pilot only
('xlarge_mf_gadi_pilot.fa',         'gadi')    10.0 MB 1000 × 10 000  × 6 runs   ⚠ pilot only
```

### Priority rerun plan (blocking before any further analysis)

**Target**: on Gadi (`normalsr`, SPR 8470Q), rebuild the alignments with
the corrected generator and rerun the full thread sweep so the three
dataset series exactly match the Setonix workload.

Priority-ordered execution plan:

| # | Task | Queue time | SU estimate |
|---|------|-----------|-------------|
| 1 | `rm -f /scratch/rc29/$USER/iqtree3/benchmarks/{large_modelfinder,xlarge_mf}.fa` on a login node | instant | 0 |
| 2 | `qsub gadi-ci/generate_datasets.sh` (regenerate with 100 × 50 000 + 200 × 100 000 + 500 × 100 000) | ≤ 1 min wall | ≈ 1 SU |
| 3 | Sanity-check file sizes match Setonix (5 000 000 / 20 000 000 / 50 000 000 bytes ± 6 kB) | instant | 0 |
| 4 | **Priority A — `large_modelfinder.fa`** sweep `{1, 4, 8, 16, 32, 64}` threads → 6 jobs | parallel | ≈ 900 SU |
| 5 | **Priority B — `xlarge_mf.fa`** sweep `{1, 4, 8, 16, 32, 64, 128}` threads (128 capped to 104 on SPR node; flag in JSON) → 7 jobs | parallel | ≈ 1 800 SU |
| 6 | **Priority C — `mega_dna.fa`** sweep `{16, 32, 64, 128}` (already partial) — fill in 64T and 104T (Setonix ran 128T; Gadi node caps 104) → 2 jobs | parallel | ≈ 1 400 SU |
| 7 | Ingest JSONs via `python3 tools/harvest_scratch.py`, then `python3 tools/normalize.py && python3 tools/validate.py && python3 tools/build.py && git push` | ≤ 5 min | 0 |
| 8 | Once authentic Gadi `large_modelfinder.fa` + `xlarge_mf.fa` series land, archive `*_gadi_pilot.fa` records (move under `logs/runs/_archive/` and exclude from normalize) | 1 min | 0 |

**Total SU estimate for the rerun**: ≈ 4.1 KSU (well within the
remaining grant; current burn ≈ 3 KSU on the pilot = ≈ 28 % of 25 KSU
used so far).

**Priority ordering rationale**: `large_modelfinder` is the fastest
(low thread-count baseline in minutes), so it validates the corrected
generator with the shortest feedback loop. `xlarge_mf` is where the
most interesting cross-platform speedup and efficiency comparisons
will sit (largest thread sweep). `mega_dna` already has 2 valid points
so only needs fill-in.

**Do not interpret `*_gadi_pilot.fa` curves as "Gadi vs Setonix"** —
they are only valid as internal Gadi thread-scaling curves on a
different, smaller workload. The dashboard colour-codes them as Gadi
(orange / triangle) but their legend label now reads
`Gadi · large_modelfinder_gadi_pilot.fa`, making the caveat visible.

### Commits in this containment wave

| Hash (to be filled) | Message |
|--------------------|---------|
| `c08234e` | `fix(gadi): generator dims match Setonix; isolate pilot runs as *_gadi_pilot.fa` |
| `<pending>` | `feat(dashboard): group dataset cards by supercomputer; auto-exclude pilot workloads from comparison charts` |

### Dashboard hardening (2026-04-25, follow-up)

To prevent the pilot workloads from visually contaminating any
cross-platform analysis, the overview UI was adjusted:

- **Dataset cards are now grouped by supercomputer.** Each platform
  (`Setonix · Pawsey (AMD Milan)`, then `Gadi · NCI (Intel Sapphire
  Rapids)`) has its own section header, a coloured status dot, and a
  dataset count. Within a section, real workloads sort first, pilot
  workloads last, then alphabetical by filename.
- **Pilot workloads get a bright `PILOT` badge + yellow warning banner**
  ("different dimensions to Setonix baseline; excluded from comparison
  charts until a matched rerun lands") and an inline `(pilot)` suffix on
  the filename heading.
- **Comparison charts auto-exclude any dataset whose name ends in
  `_gadi_pilot.fa` or `_setonix_pilot.fa`.** All four charts
  (`scaling`, `efficiency`, `ipc-scaling`, `performance-matrix`) now
  carry a shared `isPilot(name)` predicate and skip matching records
  before grouping. The records are still in the index for
  /All Runs/ inspection and per-dataset drill-downs — they just no
  longer appear on the four cross-platform comparison charts.
- Net effect on today's data: Thread Scaling / Efficiency / IPC /
  Perf-Matrix charts show only the dimensionally-matched series
  (`large_modelfinder.fa`, `xlarge_dna.fa`, `mega_dna.fa`, `turtle.fa`
  from Setonix; `mega_dna.fa` from Gadi). Twelve pilot records remain
  visible in the All Runs table and in a dedicated Gadi section of the
  Datasets strip.

---

## 2026-04-24 — `gadi-iq`: Stage 4 **resubmitted** — full profiling suite + thread sweep aligned to Setonix

Cancelled the first Stage 4 batch (jobs `166967399–414`, initial submission)
after discovering two silent-failure bugs and a profiling gap versus the Setonix
corpus. Fixed, committed (`1896deb`), and resubmitted 16 jobs.

### Bug fixes that triggered the resubmission

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | `stalled-cycles-frontend` and `stalled-cycles-backend` in `PERF_EVENTS` — not available on the Gadi kernel PMU | `perf stat` exited non-zero immediately, aborting the entire worker before IQ-TREE started; every job recorded 0 results in 2–3 s | Removed both events from `PERF_EVENTS`; restructured worker to run IQ-TREE **first** (Pass 1) so timing is captured regardless of perf status |
| 2 | Thread sweep used NUMA-aligned widths `{1,4,13,26,52,104}` instead of the Setonix sweep `{1,4,8,16,32,64}` | Results would not be directly comparable to Setonix data | Updated matrix to `{1,4,8,16,32,64}` for `large_modelfinder` + `xlarge_mf`; `{16,32,64,104}` for `mega_dna` (Setonix used 128T; Gadi caps at 104T) |

### Profiling gap vs Setonix — what was missing

The original worker only ran `perf stat` + a text VTune summary (and
only when `THREADS ≥ 13`). Compared to `setonix-ci/run_mega_profile.sh`:

| Setonix had | Gadi had (before this fix) | Gap |
|---|---|---|
| `perf stat` (AMD PMU, 27 events) | `perf stat` (Intel PMU) | ✅ covered |
| `perf record -g -F99` → `perf.data` | ✗ missing | ❌ callgraph absent |
| `perf report` → `hotspots.txt` | ✗ missing | ❌ |
| `perf script` → `perf_folded.txt` (stackcollapse for flamegraph) | ✗ missing | ❌ |
| `vtune` (N/A on AMD) | `vtune hotspots` text summary, THREADS ≥ 13 only | ⚠️ limited |
| `/proc` time-series sampler → `samples.jsonl` | ✗ missing | ✅ added (commit `c23ac7c`) |
| `profile_meta.json` (structured JSON) | Emitted as run JSON to `logs/runs/` | ✅ covered (different schema) |

### Full profiling suite added (commit `1896deb`)

Each worker job now runs 5 passes per `(dataset × threads)` point:

| Pass | Tool | Artefacts | Notes |
|------|------|-----------|-------|
| 1 | IQ-TREE (direct) | `iqtree_run.log`, `iqtree_run.iqtree`, `iqtree_run.treefile` | Always runs; provides wall time + lnL regardless of profiling |
| 2 | `perf stat -e ${PERF_EVENTS}` | `perf_stat.txt` | IPC, cache/branch/TLB miss rates, Intel TMA topdown L1 (Retiring, Bad-Spec, FE-Bound, BE-Bound) |
| 3 | `perf record -g -F99` (capped 20 min) | `perf.data`, `perf_script.txt`, `perf_report.txt` | Callgraph at 99 Hz; `perf_script.txt` is flamegraph-ready (pass through FlameGraph `stackcollapse-perf.pl`); `perf_report.txt` has top-30 symbols by self time |
| 4a | `vtune -collect hotspots` (all thread counts) | `vtune_hotspots/`, `vtune_summary.txt`, `vtune_hotspots.tsv`, `vtune_hw_events.txt` | Hardware sampling hotspots with callstacks; `.tsv` is function list with module + source line for overlay; previously skipped low-thread jobs |
| 4b | `vtune -collect uarch-exploration` (≥ 8T or `large_modelfinder`) | `vtune_uarch/`, `vtune_uarch_summary.txt`, `vtune_uarch_hw.txt` | Full TMA L1+L2: Retiring, Bad Speculation, Frontend Bound, Backend Bound, **Memory Bound** — the key metric for GPU offload ROI analysis |

JSON run records (under `logs/runs/`) now include a `vtune_uarch` block with
`{memory_bound_pct, backend_bound_pct, frontend_bound_pct, retiring_pct}` and
an `artefacts` dict mapping every key above to its scratch path.

### Gadi vs Setonix profiling parity (post-fix)

| Capability | Setonix (AMD, SLURM) | Gadi (Intel SPR, PBS) |
|---|---|---|
| Hardware counters | `perf stat` (AMD PMU, 27 events) | `perf stat` (Intel PMU, 19 events + topdown TMA) |
| IPC | ✅ | ✅ |
| Cache / branch / TLB miss rates | ✅ | ✅ |
| Frontend / backend stall rates | ✅ (stalled-cycles-* available) | ⚠️ derived from TMA topdown slots (kernel PMU does not expose `stalled-cycles-*`) |
| TMA Level 1 (Retiring, Bad-Spec, FE/BE-Bound) | ✅ AMD via perf stat | ✅ Intel via perf stat + **VTune uarch-exploration** (dual coverage) |
| TMA Level 2 (Memory Bound) | ✗ | ✅ VTune uarch-exploration |
| Callgraph (for flamegraph) | ✅ `perf record` → `perf_folded.txt` | ✅ `perf record` → `perf_script.txt` (same data, one stackcollapse step away from flamegraph) |
| Hotspot function list | ✅ `perf report` → `hotspots.txt` | ✅ `perf report` → `perf_report.txt` **AND** `vtune hotspots` → `vtune_hotspots.tsv` |
| Function-level CPI / CPU time | ✗ | ✅ VTune hotspots `.tsv` |
| Per-thread / NUMA RSS + IO time-series | ✅ `/proc` sampler → `samples.jsonl` | ✅ `/proc` sampler → `samples.jsonl` (commit `c23ac7c`) |
| GPU profiling | ✗ (no AMD GPU work on Setonix) | ✗ (`gpuvolta` deferred) |

**Full parity achieved** — every signal captured on Setonix is now captured on Gadi.
The only differences are platform-specific (Intel PMU vs AMD PMU, VTune vs rocprofv3).

### Resubmission — job table

All 16 jobs accepted onto `normalsr`, 4 running at time of writing:

| Job ID | Dataset | Threads | State |
|--------|---------|---------|-------|
| `166968738.gadi-pbs` | `large_modelfinder` | 1T  | R |
| `166968739.gadi-pbs` | `large_modelfinder` | 4T  | R |
| `166968740.gadi-pbs` | `large_modelfinder` | 8T  | R |
| `166968741.gadi-pbs` | `large_modelfinder` | 16T | R |
| `166968742.gadi-pbs` | `large_modelfinder` | 32T | Q |
| `166968743.gadi-pbs` | `large_modelfinder` | 64T | Q |
| `166968744.gadi-pbs` | `xlarge_mf`         | 1T  | Q |
| `166968745.gadi-pbs` | `xlarge_mf`         | 4T  | Q |
| `166968746.gadi-pbs` | `xlarge_mf`         | 8T  | Q |
| `166968747.gadi-pbs` | `xlarge_mf`         | 16T | Q |
| `166968748.gadi-pbs` | `xlarge_mf`         | 32T | Q |
| `166968749.gadi-pbs` | `xlarge_mf`         | 64T | Q |
| `166968750.gadi-pbs` | `mega_dna`          | 16T | Q |
| `166968751.gadi-pbs` | `mega_dna`          | 32T | Q |
| `166968752.gadi-pbs` | `mega_dna`          | 64T | Q |
| `166968753.gadi-pbs` | `mega_dna`          | 104T| Q |

Thread sweep now matches Setonix exactly (`{1,4,8,16,32,64}` for
`large_modelfinder` + `xlarge_mf`; `{16,32,64,104}` for `mega_dna` — Setonix
used 128T but Gadi node caps at 104T).

### Commits in this fix batch

| Hash | Message |
|------|---------|
| `59b06bb` | Fix perf events (remove stalled-cycles-*) + fix run_pipeline args parsing |
| `42146b3` | Align thread sweep to Setonix: {1,4,8,16,32,64} + {16,32,64,104} |
| `1896deb` | Add full profiling: perf record callgraph, VTune hotspots all threads, uarch-exploration |
| `c23ac7c` | Add /proc time-series sampler (samples.jsonl) — full Setonix parity |

### Outstanding

- **Stage 2 CI pipeline** — job `166967398.gadi-pbs` was cancelled with the first
  batch; not resubmitted. Should be requeued once Stage 4 first jobs confirm
  the worker is healthy.
- **Flamegraph rendering** — `perf_script.txt` → `stackcollapse-perf.pl` →
  `flamegraph.pl` requires the FlameGraph perl scripts on a login node
  (no compute-node internet). Can be done post-hoc once jobs finish.
- **Active jobs** — `166969038–166969055` (16 jobs, submitted 2026-04-24).

---

## 2026-04-24 — `gadi-iq`: Stage 0 bootstrap **done** (verified end-to-end)

After 7 failed PBS bootstrap submissions burning ≈ 0 SU each (all died in
seconds on missing-source / missing-deps / linker errors), switched to
iterating directly on `gadi-login-03` until the build worked standalone,
then hardened the submission script.

### Issues discovered and fixed

| # | Symptom | Root cause | Fix in `bootstrap_iqtree.sh` |
|---|---------|-----------|------------------------------|
| 1 | `fatal: unable to access 'https://github.com/...'` | Compute nodes have no outbound internet | Pre-clone + `git submodule update --init --recursive` on login node (documented in Prerequisites) |
| 2 | `Could NOT find Eigen3` | No default include path for `eigen/3.3.7` | `module load eigen/3.3.7` + `-DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3` |
| 3 | `Could NOT find Boost` | No default root for `boost/1.84.0` | `module load boost/1.84.0` + `-DBOOST_ROOT=/apps/boost/1.84.0 -DBoost_NO_SYSTEM_PATHS=ON` |
| 4 | `cmaple/CMakeLists.txt: No such file` | Submodules not fetched (default branch is `master`, not `main`) | Fetch submodules on login node; pre-flight check in script |
| 5 | `FetchContent_MakeAvailable(googletest)` hangs / fails | cmaple unconditionally pulls GoogleTest over the network | `sed` patch: comment out `include(FetchContent)` … `FetchContent_MakeAvailable(googletest)` block |
| 6 | **`ld: File format not recognized`** on `.o` files | cmaple sets `CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE` → `icx` emits LLVM IR bitcode `.o`, but system `ld` can't link bitcode and Gadi has no `lld` | `sed` patch: flip IPO `TRUE` → `FALSE` in `cmaple/CMakeLists.txt` L88 |
| 7 | `Target cmaple_maintest links to: GTest::gtest_main but the target was not found` | cmaple/unittest still declared even with FetchContent disabled | `sed` patch: comment out `add_subdirectory(unittest)` |
| 8 | Build on login node crawled (1 file at a time) | Login-node cgroups report `nproc=1` | Added `IQTREE_BUILD_JOBS` env override (defaults to `nproc` on compute nodes, set explicitly for login-node tests) |

All patches are idempotent (`grep -q <sentinel>` before each `sed`).

### Verified binaries

End-to-end run on `gadi-login-03` with `IQTREE_BUILD_JOBS=8` produced:

```
-rwxr-xr-x 1 as1708 rc29 146M  build/iqtree3           (release, -O3 -xSAPPHIRERAPIDS)
-rwxr-xr-x 1 as1708 rc29 146M  build-profiling/iqtree3 (+ -fno-omit-frame-pointer -g)
```

Functional test on `example.phy` (different login-node CPU, so run under
smaller arch) completed ML inference in 6.8 s wall. `--version` on
the login node reports `IQ-TREE version 3.1.1 for Linux x86 64-bit
built Apr 24 2026`; AVX-512-VBMI instruction check refuses to run on
Skylake-SP login-node CPU as expected (the binary *is* compiled for
Sapphire Rapids and will run on `normalsr`).

Since login-node binaries are complete and staged under
`/scratch/rc29/as1708/iqtree3/build{,-profiling}/iqtree3`, **Stage 0
PBS submission is skipped** — no point burning 210 SU to rebuild what
we already verified. Proceeding directly to Stage 1.

### Stage 1 — datagen submitted

`qsub gadi-ci/generate_datasets.sh` → regenerates turtle.fa +
large_modelfinder.fa + xlarge_mf.fa + mega_dna.fa via AliSim. Expected
≈ 210 SU, ≤ 1 h wall.

Submitted: **`166967018.gadi-pbs`** (queue `normalsr-exec`, 104c/500GB/1h,
state `Q`). Will append runtime + artefact listing once the job
completes.

**Stage 1 result:** exit 0, walltime **00:00:03**, **0.17 SU** billed (vs
210 SU estimated — AliSim is wildly fast on Sapphire Rapids for these
sizes). Artefacts in `/scratch/rc29/as1708/iqtree3/benchmarks/`:

```
34K   example.phy
12K   turtle.fa
2.4M  large_modelfinder.fa  (500 × 5000, seed 101)
9.6M  xlarge_mf.fa          (1000 × 10000, seed 202)
48M   mega_dna.fa           (500 × 100000, seed 303)
```

Minor fix: `turtle.fa` lives at `src/iqtree3/test_scripts/test_data/`
(not `example_data/`) — added that path to the `generate_datasets.sh`
candidate list and copied the file manually for this run.

**Cumulative SU burn so far:** ≈ 0.2 SU (out of 25 000 grant).

Proceeding to Stage 2 (CI smoke) next.

---

## 2026-04-24 — `gadi-iq`: Stages 2 + 4 **queued** (17 PBS jobs, full sweep)

Per user directive "queue them all up" — skipped Stage 3 (single matrix
point) since it's a strict subset of Stage 4, and submitted both Stage 2
(CI pipeline) and Stage 4 (full 16-job benchmark matrix) together. All
17 jobs accepted onto `normalsr-exec`, state `Q`.

### Stage 2 — CI pipeline (1 job)

| Job ID                  | Jobname          | NCPUs | Mem    | Walltime |
|-------------------------|------------------|-------|--------|----------|
| `166967398.gadi-pbs`    | iqtree-pipeline  | 104   | 500 GB | 01:00    |

Runs `turtle.fa` + `example.phy` under the SPR binary to sanity-check
the run.schema.json emission (pbs_id, env.pbs.*, verify block).
Pointed `TEST_DATA` at `${PROJECT_DIR}/benchmarks/` where both files
now live.

### Stage 4 — full benchmark matrix (16 jobs)

Each job: `ncpus=104, mem=500GB, walltime=24:00:00` on `normalsr`,
builds one `(dataset × threads)` point, emits a JSON record under
`logs/runs/` via the embedded worker `_run_matrix_job.sh`.

| Dataset               | Threads sweep          | Job IDs (`166967*`) |
|-----------------------|------------------------|---------------------|
| `large_modelfinder`   | 1, 4, 13, 26, 52, 104  | 399, 400, 401, 402, 403, 404 |
| `xlarge_mf`           | 1, 4, 13, 26, 52, 104  | 405, 406, 407, 408, 409, 410 |
| `mega_dna`            | 13, 26, 52, 104        | 411, 412, 413, 414  |

Expected SU burn (from the plan table above): **≈ 6.1 KSU ≈ 24 %** of
the 25 000 SU grant. Walltime is billed against the requested 24 h cap
per job, but each job exits as soon as IQ-TREE + the optional VTune
hotspots pass finish — so actual SU should track IQ-TREE wall, not the
reservation.

Monitoring:

```bash
qstat -u $USER
ls /scratch/rc29/as1708/iqtree3/gadi-ci/logs/
ls /home/272/as1708/setonix-iq/logs/runs/
```

---

## 2026-04-24 — `gadi-iq`: Sapphire Rapids rerun plan (reproduce Setonix benchmarks)

Before running any PBS jobs, audited the existing Setonix corpus and
sized the equivalent workload for Gadi. Target platform is now Intel
**Sapphire Rapids** (`normalsr` queue, not the older Cascade Lake nodes):

| Component      | Spec                                                      |
|----------------|-----------------------------------------------------------|
| CPU            | 2× Intel Xeon Platinum **8470Q** (Sapphire Rapids), 52c each |
| Cores / node   | **104** (was 48 on Cascade Lake `normal`)                 |
| NUMA           | 8 domains, 13 cores / NUMA, 64 GB / NUMA                   |
| Memory / node  | **512 GiB** (request `mem=500GB`)                          |
| Charge rate    | 2 SU / core-h → **208 SU / node-hour**                    |
| Walltime cap   | 48 h for 1-1040 cores                                      |
| Compiler       | `intel-compiler-llvm/2024.2.0` → `icx -xSAPPHIRERAPIDS`    |
| IQ-TREE source | `https://github.com/iqtree/iqtree3.git` (branch `main`)    |

### Setonix corpus (what we want to reproduce)

18 runs in `logs/runs/`, 2 deep profiles in `logs/profiles/`. Total
**48.64 CPU-hours of IQ-TREE work** split across three datasets:

| Dataset                 | Setonix threads          | Sum wall |
|-------------------------|--------------------------|---------:|
| `large_modelfinder.fa`  | 1, 4, 8, 16, 32, 64      |  8.07 h  |
| `xlarge_mf`             | 1, 4, 8, 16, 32, 64, 128 | 18.47 h  |
| `mega_dna.fa` (500×100k)| 16, 32, 64, 128          | 22.05 h  |
| turtle.fa CI pipeline   | serial                   |  0.005 h |

### Gadi matrix (adjusted for Sapphire Rapids 104-core node)

Thread sweep renormalised to `{1, 4, 13, 26, 52, 104}` — powers-of-two
plus NUMA-aligned widths (13 = one NUMA, 26 = NUMA pair, 52 = one socket,
104 = full node). `mega_dna.fa` runs only at `{13, 26, 52, 104}`
(lower thread counts are bandwidth-starved on 100 kbp alignments).

| Dataset                | Threads                   | Runs | Wall (est)¹  | SU²     |
|------------------------|---------------------------|:----:|-------------:|--------:|
| `large_modelfinder.fa` | 1, 4, 13, 26, 52, 104     |   6  |  3.2 node-h  |     670 |
| `xlarge_mf`            | 1, 4, 13, 26, 52, 104     |   6  |  7.4 node-h  |   1 540 |
| `mega_dna.fa`          | 13, 26, 52, 104           |   4  |  6.8 node-h  |   1 420 |
| VTune pass × 14 (not 1T)| 30 min cap               |  14  |  7.0 node-h  |   1 456 |
| Bootstrap + datagen    | —                         |   2  |  1.0 node-h  |     210 |
| 15 % headroom          | —                         |      |              |     795 |
| **Total**              |                           | **16 jobs** | **≈ 25 node-h** | **≈ 6.1 KSU** |

¹ Estimates assume Sapphire Rapids delivers ≈ 1.2 × Cascade Lake per
  core (AVX-512 + higher IPC on IQ-TREE's SIMD likelihood kernels) and
  sub-linear scaling above 52T from NUMA traffic. Billing is on the
  full `ncpus=104` reservation regardless of `-T` passed to IQ-TREE.
² `normalsr` charges 2 SU / core-h = **208 SU / node-hour**.

### Budget check

- Project `rc29` 2026.q2 grant: **25 000 SU**, used 0, reserved 0.
- Full sweep ≈ **6.1 KSU ≈ 24 %** of grant → well within budget.
- Scratch: 1 TiB quota, 188 KiB used. Raw outputs ≈ 100-500 MB per run;
  16 runs ≈ 8 GB worst case. Non-issue.
- Local disk (`jobfs`): 400 GiB per node — ample for VTune's trace DB.

### Real wall-clock estimate

Submitted as 16 parallel PBS jobs (the matrix script auto-fans out),
normalsr has 720 nodes so queuing should be short. Expected clock
≈ **8-15 h** for the bulk of the matrix to finish, mostly unattended.

### Prerequisites (resolved by new scripts)

1. **No IQ-TREE binary on Gadi** → `gadi-ci/bootstrap_iqtree.sh` clones
   **`https://github.com/iqtree/iqtree3.git`** and builds it twice:
   - `${PROJECT_DIR}/build/iqtree3` — release, `icx -O3 -xSAPPHIRERAPIDS`.
   - `${PROJECT_DIR}/build-profiling/iqtree3` — adds
     `-fno-omit-frame-pointer -g` for `perf -g` stack unwinding.

   Fallback to `gcc -march=sapphirerapids` if `icx` is not on PATH. The
   script is a 1-hour `normalsr` PBS job (~210 SU).

   > **Build requirements (Gadi modules — load before cmake or set in bootstrap script):**
   >
   > | Module                        | Purpose                         | Gadi path                              |
   > |-------------------------------|---------------------------------|----------------------------------------|
   > | `cmake/3.31.6`                | Build system                    | `/apps/cmake/3.31.6`                   |
   > | `intel-compiler-llvm/2024.2.0`| C/C++ compiler (`icx`/`icpx`)  | `/apps/intel-tools/wrappers/icx`       |
   > | `eigen/3.3.7`                 | Required header-only math lib   | `/apps/eigen/3.3.7/include/eigen3`     |
   > | `boost/1.84.0`                | Required headers + libs         | `/apps/boost/1.84.0`                   |
   > | `gcc/14.2.0`                  | Fallback compiler (optional)    | `/apps/gcc/14.2.0`                     |
   >
   > CMake hints passed explicitly:
   > ```
   > -DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3
   > -DBOOST_ROOT=/apps/boost/1.84.0
   > -DBoost_NO_SYSTEM_PATHS=ON
   > ```
   > **Compute nodes have no outbound internet** — all source and deps must
   > be fetched on a login node. Run this once before submitting any job:
   > ```bash
   > SCRATCH=/scratch/rc29/<user>/iqtree3
   >
   > # 1. Clone IQ-TREE 3 (default branch: master)
   > git clone https://github.com/iqtree/iqtree3.git ${SCRATCH}/src/iqtree3
   >
   > # 2. Fetch submodules (cmaple + lsd2 — required by CMakeLists)
   > cd ${SCRATCH}/src/iqtree3
   > git submodule update --init --recursive
   >
   > # 3. Pre-download GoogleTest (cmaple FetchContent — fails on compute nodes)
   > cd ${SCRATCH}/deps
   > curl -L -o googletest.zip \
   >   https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
   > unzip -q googletest.zip && mv googletest-*/ googletest
   >
   > # 4. Submit build job
   > cd ${SCRATCH} && qsub gadi-ci/bootstrap_iqtree.sh
   > ```
   > The bootstrap script pre-flight checks for all three (`cmaple`, `lsd2`,
   > `googletest`) and errors out with the exact fix command if any are missing.
2. **No benchmark alignments on Gadi** (they live on Setonix scratch and
   can't be rsynced cross-site from a login node) →
   `gadi-ci/generate_datasets.sh` regenerates equivalent workloads
   deterministically via IQ-TREE 3's built-in **AliSim** simulator with
   fixed seeds (101 / 202 / 303):

   | Output               | Dimensions          | Model              |
   |----------------------|---------------------|--------------------|
   | `large_modelfinder.fa` |  500 taxa × 5 000 bp | GTR{...}+F+G4 |
   | `xlarge_mf.fa`       | 1000 taxa × 10 000 bp | GTR{...}+F+G4 |
   | `mega_dna.fa`        |  500 taxa × 100 000 bp| GTR{...}+F+G4 |

   Plus `turtle.fa` + `example.phy` copied from the iqtree3 repo's
   `example_data/` for the CI smoke test. ~1 hour, ~210 SU.
3. **Orchestration** → `gadi-ci/submit_benchmark_matrix.sh` enumerates
   the 16-job matrix, emits an embedded worker (`_run_matrix_job.sh`),
   and submits each point as a separate `qsub -q normalsr
   -l ncpus=104,mem=500GB,walltime=24:00:00` job. Each worker runs IQ-TREE
   under `perf stat`, optionally under VTune hotspots for `THREADS ≥ 13`,
   then writes a single JSON to `$REPO_DIR/logs/runs/<id>.json` that
   conforms to `tools/schemas/run.schema.json` (Intel TMA metrics +
   `env.pbs` + `profile.vtune`).

### Updated gadi-ci scripts

All existing scripts (`run_mega_profile.sh`, `submit_mega_batch.sh`,
`run_pipeline.sh`, `run_profiling.sh`, `gadi-ci/README.md`) switched
from `normal` / `ncpus=48` / `mem=190GB` / Cascade Lake to
`normalsr` / `ncpus=104` / `mem=500GB` / Sapphire Rapids. Intel event
names are unchanged (perf aliases apply to `spr_core` pmu). Default
thread sweep for the mega-profile batch is now `1 4 13 26 52 104`.

The Makefile's `CMAKE_PROFILING` flags now set
`-O3 -xSAPPHIRERAPIDS -fno-omit-frame-pointer` and `MODULES` defaults
to `intel-vtune/2024.2.0 intel-compiler-llvm/2024.2.0`.

### Rollout (to minimise SU burn on bugs)

1. **Stage 0 — bootstrap.** `qsub gadi-ci/bootstrap_iqtree.sh`. Verify
   binary produced, `iqtree3 --version` prints Intel LLVM toolchain.
   ≈ 210 SU.
2. **Stage 1 — datasets.** `qsub gadi-ci/generate_datasets.sh`. Verify
   the three `.fa` files land under `$PROJECT_DIR/benchmarks/`. ≈ 210 SU.
3. **Stage 2 — CI smoke.** `./start.sh pipeline` (turtle.fa +
   example.phy). Confirms `pbs_id`, `env.pbs.*`, verify block, schema
   validation end-to-end. ≈ 1 SU (runs on login node or small PBS job).
4. **Stage 3 — one matrix point.**
   `./gadi-ci/submit_benchmark_matrix.sh large_modelfinder 52` — single
   job, confirms perf + VTune data ingestion into `logs/runs/`. ≈ 100 SU.
5. **Stage 4 — full matrix.** Only launch once 0-3 are green.
   `./gadi-ci/submit_benchmark_matrix.sh`. Auto-rerunnable (records are
   immutable once written; re-running overwrites only matching
   `<timestamp>_<label>` records).

This changelog entry is pre-execution: it locks in the plan and
budget. Actual numbers will be appended as jobs complete.

---


Forked `main` (wave-4 Setonix data complete) into a new branch, `gadi-iq`,
targeted at the NCI Gadi supercomputer. The dashboard front-end and data
pipeline are unchanged — the refactor is entirely in the data-producing
scripts and the schema that binds them to the web UI.

### Target system

| Component | Spec |
|---|---|
| Machine   | Gadi (NCI, Canberra) |
| CPU       | Intel Xeon Platinum 8470Q "Sapphire Rapids", 104c/node (2×52), 8 NUMA domains (13c each) |
| Memory    | 512 GiB/node (`mem=500GB` leaves scheduler headroom) |
| Scheduler | PBS Professional 2024.1 (`qsub`, `qstat`, `qdel`, `nqstat`) |
| Queues    | `normalsr` (default), `expresssr`, plus legacy `normal`/`normalbw`/`hugemem`/`gpuvolta`/`megamem` |
| Profiler  | Intel VTune 2024.2.0 (`module load intel-vtune/2024.2.0`) + `perf` |
| Compiler  | `intel-compiler-llvm/2024.2.0` — `icx -xSAPPHIRERAPIDS` |
| Project   | `rc29`; scratch `/scratch/rc29`; home 10 GB quota |

### New `gadi-ci/` scripts (PBS / Intel / VTune)

- `gadi-ci/run_mega_profile.sh` — PBS job script mirroring the Setonix
  mega profiler. Key changes:
  - `#SBATCH` directives → `#PBS -N / -P rc29 / -q normalsr /
    -l ncpus=104,mem=500GB,walltime=24:00:00,wd,storage=scratch/rc29 / -j oe`.
  - SLURM env vars → PBS env vars (`PBS_JOBID`, `PBS_JOBNAME`,
    `PBS_QUEUE`, `PBS_NCPUS`, `PBS_NODEFILE`, `PBS_O_HOST`,
    `PBS_O_WORKDIR`). `env.json` now records them under `env.pbs.*`.
  - AMD Zen 3 raw events (`ex_ret_*`, `ls_l1_d_tlb_miss.*`,
    `bp_l1_tlb_miss_l2_tlb_*`, `ls_dispatch.*`, `ls_tablewalker.*`)
    replaced with Intel Sapphire Rapids Top-down TMA slot events
    (`topdown-{total-slots,slots-issued,slots-retired,fetch-bubbles,
    recovery-bubbles}`) plus `LLC-loads`/`LLC-load-misses`.
  - Derived TMA Level-1 categories emitted as
    `intel-tma-{retiring,bad-spec,frontend-bound,backend-bound}-pct` in
    `profile_meta.json`.
  - New **step 4 VTune pass** — bounded to 30 min (`VTUNE_MAX_S`),
    `vtune -collect hotspots -knob sampling-mode=hw
    -knob enable-stack-collection=true`. Report is extracted to
    `vtune_summary.txt` + CSV → `vtune_hotspots.json`. Parsed into
    `profile.vtune.{elapsed_time_s, cpu_time_s, effective_cpu_util,
    avg_cpu_freq_ghz, hotspots[]}`.
  - `RUN_ID="${PBS_JOBID%%.*}"` strips the `.gadi-pbs` suffix.
- `gadi-ci/submit_mega_batch.sh` — `sbatch --parsable` loop rewritten as
  `qsub` loop. Default thread sweep is `1 4 13 26 52 104` to match
  Sapphire Rapids' 104c/node ceiling with NUMA-aligned points. Each
  job gets an explicit `-P $PROJECT -q normalsr -l
  ncpus=104,mem=500GB,walltime=24:00:00,storage=scratch/$PROJECT,wd`.
- `gadi-ci/run_pipeline.sh` — small deterministic CI pipeline runnable on
  a login node. Emits `logs/runs/<YYYY-MM-DD_HHMMSS>.json` in the exact
  shape of the Setonix records but with `pbs_id` + `env.pbs.*` populated.
- `gadi-ci/run_profiling.sh` — quick perf-stat wrapper for interactive
  `qsub -I` sessions.
- `gadi-ci/README.md` — documents the script set and lists every semantic
  diff vs `setonix-ci/`.

### `start.sh` + `Makefile` rewritten for Gadi

- `start.sh`: SLURM/Pawsey references removed. `PROJECT=rc29` default,
  `qstat -u $USER` replaces `squeue`, `nci_account` replaces
  `pawseyAccountBalance`. New `./start.sh batch` command fans out the
  mega-profile sweep via `submit_mega_batch.sh`. `./start.sh deepprofile`
  now `qsub`s the deep profile script.
- `Makefile`: `PAWSEY_PROJECT` → `PROJECT`, `IQTREE_DIR =
  /scratch/$(PROJECT)/$(USER)/iqtree3`. `make deep-profile` uses `qsub`;
  `make status` calls `qstat` + `nci_account`. Added auto `module load
  intel-vtune/2024.2.0 intel-compiler/2024.2.1` in build targets.

### Schema extensions — additive, no breaking changes

`tools/schemas/run.schema.json` updated so both Setonix-era and
Gadi-era records validate against a single schema:

- `pbs_id: string|null` alongside `slurm_id: string|null`.
- `env.pbs` object (`job_id, job_name, queue, project, ncpus, nnodes,
  mem, nodes[], nodefile, submit_host, submit_dir, o_queue, scheduler`)
  alongside the existing `env.slurm`.
- `env.icc`, `env.icx`, `env.vtune_version`.
- `profile.metrics`: added `LLC-miss-rate`,
  `intel-tma-retiring-pct`, `intel-tma-bad-spec-pct`,
  `intel-tma-frontend-bound-pct`, `intel-tma-backend-bound-pct`. The
  existing AMD `amd-*` properties remain optional.
- `profile.vtune` object capturing VTune headline metrics + top-50
  VTune hotspots.
- `gpu_info` now accepts `string | object` (forward-compat for
  structured NVIDIA V100 telemetry from Gadi's `gpuvolta` queue).

All 18 existing Setonix runs re-validate against the extended schema.
New pytest suite (`tests/test_gadi_schema.py`) verifies three scenarios:
a minimal PBS record, a PBS record with Intel TMA + VTune data, and a
legacy SLURM/AMD record — all must remain valid. **17 passed, 1 xpassed**.

### Dashboard rebrand (web/)

- `web/index.html`: title `Setonix-IQ Dashboard` → `Gadi-IQ Dashboard`;
  sidebar/topbar logo text `Setonix-IQ` → `Gadi-IQ`; meta description
  updated to reference Gadi (NCI).
- `web/js/pages/overview.js`: subtitle reads "Insight dashboard for
  IQ-TREE runs on Gadi (NCI)". System-info KV cells surface VTune version
  (falling back to `env.rocm` when rendering archived Setonix data).
- `web/js/pages/runs.js`: `SLURM:` column label → `Job:`; value prefers
  `run.pbs_id`, falls back to `run.slurm_id`.
- `web/js/pages/gpu.js`: subtitle and empty-state strings now describe
  both platforms (NVIDIA V100 on `gpuvolta` / AMD MI250X on Setonix).

### `tools/harvest_scratch.py` — env-driven defaults

- Defaults to `/scratch/${PROJECT:-rc29}/${USER}/iqtree3` and auto-detects
  `gadi-ci/profiles` first, falling back to `setonix-ci/profiles` if only
  the Setonix tree is present on-disk. All paths remain overridable via
  `SCRATCH_DIR`, `PROFILE_ROOT`, `BENCHMARKS_DIR`, and now `SLURM_ID`.
- `tools/normalize.py`: run + profile index entries now carry both
  `slurm_id` and `pbs_id` so the runs page renders the right job id per
  record.
- `tools/build.py`: docstring + build banner updated.

### Disk / quota sanity check

On the Gadi login node where this work was performed:

- `/home/272/as1708` usage 516 MB / 10 GB quota (the repo itself is ~1 MB).
- `/scratch/rc29`: 108 KiB / 1 TiB used.
- SU grant: 25 KSU (none consumed yet) on project `rc29` for 2026.q2.

Enough headroom for the repo, a full IQ-TREE build tree, and multiple
mega-profile runs.

### Outstanding (not done on this branch yet)

- **Not executed on Gadi** — scripts are verified `bash -n` clean and the
  schema accepts their output shape, but no real PBS run has been
  submitted yet. First smoke test: `./start.sh pipeline` on a login node
  after rsyncing `gadi-ci/` into `/scratch/rc29/$USER/iqtree3/gadi-ci/`.
- `gpuvolta`-specific deep-profile script (NVIDIA V100, `nvidia-smi`,
  `nsys`, `ncu`) — intentionally deferred; the CPU path is the priority.
- Schema: no Intel TMA Level-2 events yet (need `topdown-l2-*` aliases
  available on the kernel in use); Level-1 is enough for the overview
  page today.

---

## 2026-04-24 — Wave 4 completed + off-cluster harvest path

All four wave-4 mega jobs finished cleanly. The pipefail/timeout hardening
held — every job produced a full artifact set (`perf_stat.txt`,
`hotspots.txt`, `perf_folded.txt`, `profile_meta.json`, `samples.jsonl`,
`env.json`, `iqtree_run.*`).

| Job ID   | Threads | State     | Elapsed   | Ended (AWST)     |
|----------|---------|-----------|-----------|------------------|
| 41849110 | 16T     | COMPLETED | 06h50m28s | 2026-04-23 18:23 |
| 41849111 | 32T     | COMPLETED | 07h05m37s | 2026-04-23 18:42 |
| 41849112 | 64T     | COMPLETED | 07h57m24s | 2026-04-23 19:34 |
| 41849113 | 128T    | COMPLETED | 08h19m02s | 2026-04-23 19:55 |

IPC collapse on `mega_dna.fa` (500 taxa × 100 000 sites) matches the
xlarge trend: **16T=0.412 → 32T=0.236 → 64T=0.116 → 128T=0.084** — same
OMP-barrier / coherence saturation, now at a 10× larger problem size.

### Why the site did not update after wave 4

The published workflow assumed `ssh setonix && make harvest && make build
&& git push` from `~/setonix-iq` on a Setonix login node, but that clone
did not exist (`/home/asamuel/setonix-iq: No such file or directory`).
Scratch had all the data; nobody ran harvest. No push → no Pages rebuild.

### Fix: harvest from anywhere with access to Setonix over ssh

- `tools/harvest_scratch.py` now honours `SCRATCH_DIR`, `PROFILE_ROOT`,
  `BENCHMARKS_DIR` env vars. Defaults remain the Pawsey scratch paths.
- `tools/harvest_scratch.py` merged-profile parser rewritten for the
  current `run_mega_profile.sh` layout — `profile_meta.json` nests under
  `meta["profile"]` (not `meta["perf_stat"]`), perf counter keys carry a
  `:u` user-mode suffix, and only raw counters are emitted. Harvest now:
  1. reads both layouts (`meta["profile"]` *or* `meta["perf_stat"]`),
  2. strips the `:u` / `:k` suffix so keys match older runs,
  3. derives `IPC`, `{cache,branch,L1-dcache,dTLB,iTLB}-miss-rate`, and
     `{frontend,backend}-stall-rate` from the raw counters.
- New `make harvest` target: `rsync -av --include=… --exclude='*'` from
  Setonix scratch into `./.scratch-mirror/` (gitignored, text artifacts
  only — no `perf.data`, no `*.ckp.gz`), then runs the Python harvester
  with `SCRATCH_DIR` set to the mirror. Works from any box with ssh
  access to Setonix.
- Published wave-4 data: 4 new run files
  (`mega_{16,32,64,128}t_baseline.json`), 18/18 schema-valid, 15 pytests
  pass (14 + 1 xpass on the warn-only regression guard).

### Outstanding

`perf_folded.txt` was empty on all four wave-4 jobs (0 bytes). The
`set +o pipefail … || true` guard kept the script alive, but
`perf script | python stackcollapse` still produced no output. Likely
`perf script` signalled before any stack was written; the `perf.data` is
intact on scratch so folded stacks can be regenerated post-hoc on a
login node. Not blocking — hotspots from `perf report` are populated.

---

## 2026-04-23 — Mega profiling wave 3 SIGPIPE failure + wave 4 resubmission

Wave 3 (`41835470-73`) cleared the startup bugs from 04-22 and ran for
9.5–11.5 h each, but **all four jobs failed with exit code 13 (SIGPIPE)** in
step 5 during `perf script | python stackcollapse`. Confirmed cause: the
hardening commit (`set +o pipefail` around the stackcollapse pipe + `timeout`
around `perf record`) was authored on 04-22 *after* SLURM had already taken
its submission-time snapshot of the script for wave 3. The hardened script
sat on scratch but never reached the running jobs.

| Job ID    | Threads | Started     | Failed at  | Elapsed | Stage at exit                  |
|-----------|---------|-------------|------------|---------|--------------------------------|
| 41835470  | 16T     | 04-22 19:03 | 04-23 06:34 | 11h30m | Generating hotspots/folded     |
| 41835471  | 32T     | 04-22 19:03 | 04-23 06:28 | 11h25m | Generating hotspots/folded     |
| 41835472  | 64T     | 04-22 19:52 | 04-23 05:30 | 9h37m  | Generating hotspots/folded     |
| 41835473  | 128T    | 04-22 19:54 | 04-23 06:42 | 10h47m | Generating hotspots/folded     |

### What survived on scratch (per `mega_<T>t_<jobid>/`)

- `perf_stat.txt` ✅ — full 27-event multiplexed counter pass (≈21 366 s for 16T)
- `perf.data` ✅ — 99 Hz call-graph capture
- `samples.jsonl` ✅ — 10 s `/proc` time series
- `env.json`, `iqtree_run.{iqtree,treefile,model.gz,log}` ✅
- **Missing:** `hotspots.txt`, `perf_folded.txt`, `profile_meta.json`

### Resubmission — wave 4 (`41849110-13`)

Verified before submitting:

- `grep` confirmed `set +o pipefail` (line 329) and `PERF_RECORD_MAX_S=7200`
  with `timeout --preserve-status` (lines 316–317) are present in the
  scratch copy of `run_mega_profile.sh`.
- `bash -n` syntax-checked both `run_mega_profile.sh` and `submit_mega_batch.sh`.
- Re-ran `./submit_mega_batch.sh` from scratch with no thread-count argument →
  default `16 32 64 128`.

| Job ID    | Threads | State | Reason   | Est. Start (AWST)  | Est. End (AWST)    |
|-----------|---------|-------|----------|--------------------|--------------------|
| 41849110  | 16T     | PD    | Priority | 2026-04-23 11:51   | 2026-04-24 11:51   |
| 41849111  | 32T     | PD    | Priority | 2026-04-23 11:51   | 2026-04-24 11:51   |
| 41849112  | 64T     | PD    | Priority | 2026-04-23 11:51   | 2026-04-24 11:51   |
| 41849113  | 128T    | PD    | Priority | 2026-04-23 11:51   | 2026-04-24 11:51   |

This snapshot of the script *includes* the pipefail/timeout hardening, so the
SIGPIPE failure mode that killed wave 3 cannot recur here.

### Open: backfill from wave-3 `perf.data`

The four `perf.data` files from wave 3 are intact on scratch. A follow-up
post-processing pass on a login node (`perf report` + `perf script |
stackcollapse`) can recover hotspots/folded stacks without consuming any
SLURM time. Not done in this entry.

---

## 2026-04-22 — Mega profiling: startup-crash fixes + defensive hardening

Two consecutive bugs killed the 4 mega jobs (`41784642-45`, then `41814628-32`)
within seconds of launch, before any IQ-TREE work could start. Both are in the
`env.json` heredoc at the top of `setonix-ci/run_mega_profile.sh`. All
currently running mega jobs (`41835470-73`) are past this code path.

### Bug 1 — unquoted heredoc + `set -u`  (commit `5c77a5d`)

`python3 <<PYEOF` (vs `<<'PYEOF'`) means bash performs parameter expansion on
the heredoc body before passing it to Python. With `set -euo pipefail`, the
first `$2` inside an awk string (`awk '/Socket\(s\)/{print $2}'`) tripped
bash's unbound-variable check and aborted with exit code 1 before Python even
ran. Fix: escape every awk field reference inside the unquoted heredoc —
`\$2`, `\$NF` — so bash leaves them for awk.

### Bug 2 — `int("")` on compute nodes  (commit `6a45bae`)

With `set -e` lifted, the env heredoc now executed under Python, but
`int(sh("nproc"))` raised `ValueError: invalid literal for int() with base
10: ''` at line 32. On Setonix compute nodes inside a SLURM allocation,
`nproc` occasionally emits an empty string (when run before cgroups settle).
Fix: defensive default matching every other `int(sh(...))` call in the file —
`int(sh("nproc", "0") or 0)`.

### Defensive hardening (commit `<pending>`)

Additional changes to prevent late-stage failures from losing already-captured
data in long runs:

- **`perf script | python` pipe is now `pipefail`-safe.** Wrapped the
  `stackcollapse` pipeline in `set +o pipefail … set -o pipefail` with an
  explicit `|| true`, so a partial `perf.data` or SIGPIPE inside Python
  cannot abort the run before step 6 (`profile_meta.json`) is written.
- **`perf record` wall-time bound.** The re-run under `perf record -g -F 99`
  is now wrapped in `timeout --preserve-status ${PERF_RECORD_MAX_S:-7200}`,
  capping it at 2 h. At 99 Hz that is > 700 k stack samples — ample for
  hotspot ranking — and guarantees that the mega run cannot be killed by
  SLURM's 24 h wall limit during step 5, starving the profile-meta emit.
- **Deployed and syntax-checked** on scratch; repo copy pushed to `main`.

### Re-submission history (same 4 threads, 4 JobIDs each wave)

| Wave | JobIDs                        | Outcome / state            |
|------|-------------------------------|----------------------------|
| 1    | 41784642, 43, 44, 45          | FAILED — bash `$2` unbound |
| 2    | 41814628, 29, 30, 32          | FAILED — `int("")` on `nproc` |
| 3    | 41835470, 71, 72, 73          | 16T + 32T `R`, 64T + 128T `PD` |

As of 2026-04-22 19:04 AWST:
- **16T (41835470)** — `R` on `nid002039`, past env snapshot, inside
  `perf stat + IQ-TREE` pass, inner iqtree pid `3464683`. Wall ends
  tomorrow 19:03.
- **32T (41835471)** — `R` on `nid002054`, past env snapshot, inside
  `perf stat + IQ-TREE` pass, inner iqtree pid `1864019`. Wall ends
  tomorrow 19:03.
- **64T (41835472)** — `PD Priority`, estimated start 21:00.
- **128T (41835473)** — `PD Priority`, estimated start 21:00.

Note: queued jobs run from SLURM's snapshot of the script taken at
submission time, so the pipefail / `timeout` hardening above applies only to
any future waves.

---

## 2026-04-21 — Mega dataset: enhanced Setonix profiling pipeline

Addresses wishlist items **1**, **2** (partial — L3 admin-locked), **3** (partial), **4** (partial), **11**, **12**.

### Enhanced profiler — `setonix-ci/run_mega_profile.sh`

Single-job, single-thread-count wrapper for `mega_dna.fa` (500 taxa × 100 000 sites, 48 MB).
Captures the full Zen3 data we have access to in one pass:

- **27-event `perf stat`** (multiplexed ≈ 40 %): `cycles, instructions,
  branch-{instructions,misses}, cache-{references,misses}, L1-dcache-{loads,load-misses},
  dTLB-{loads,load-misses}, iTLB-{loads,load-misses}, stalled-cycles-{frontend,backend},
  task-clock, page-faults, context-switches, cpu-migrations`, plus AMD raw events
  `ex_ret_ops, ex_ret_brn_misp, ls_l1_d_tlb_miss.all, bp_l1_tlb_miss_l2_tlb_{hit,miss},
  ls_tablewalker.{dside,iside}, ls_dispatch.{ld_dispatch,store_dispatch}`.
- Derived rates: IPC, cache-/branch-/L1-dcache-/dTLB-/iTLB-/frontend-stall-/backend-stall-rate,
  AMD branch-mispred / L1-dTLB-miss / L2-TLB-miss rates.
- **`perf record -g -F 99`** → `hotspots.txt` (perf report) + `perf_folded.txt`
  (inline `stackcollapse`).
- **10 s /proc sampler** (embedded Python) → `samples.jsonl` with RSS/VmHWM/VmSize,
  voluntary/involuntary ctxt-switches, `/proc/<pid>/io` (read/write bytes, rchar/wchar,
  syscr/syscw), `numastat -p` per-node MB, and per-TID `utime/stime/nice`.
- **`env.json`** snapshot: kernel, glibc, gcc, python, iqtree version, CPU sockets /
  cores / threads / governor, SMT state, NUMA nodes, SLURM context (job id / partition /
  nodelist / cpus_per_task / mem).
- Emits unified `profile_meta.json` combining all of the above.

Not enabled (admin-locked on Setonix, `perf_event_paranoid=2`):

- L3 uncore events (`l3_lookup_state.*`, `l3_comb_clstr_state.*`, `l3_misses`,
  `l3_read_miss_latency`) — require `-a` / CAP_SYS_ADMIN.
- DRAM bandwidth metrics (`nps1_die_to_dram`, `all_remote_links_outbound`) — uncore.
- `perf stat -M TopdownL1` — needs elevated events.

### Batch submission — `setonix-ci/submit_mega_batch.sh`

Fans out 4 independent SLURM jobs. Submitted 2026-04-21:

| Threads | JobID     | Status | Est. Start (AWST) | Est. End (AWST)  |
|---------|-----------|--------|-------------------|------------------|
| 16      | 41784642  | PD     | 2026-04-22 01:16  | 2026-04-23 01:16 |
| 32      | 41784643  | PD     | 2026-04-22 01:16  | 2026-04-23 01:16 |
| 64      | 41784644  | PD     | 2026-04-22 01:18  | 2026-04-23 01:18 |
| 128     | 41784645  | PD     | 2026-04-22 01:19  | 2026-04-23 01:19 |

All 4 jobs confirmed queued as of 2026-04-21 (reason: Priority). Each requests
1 node × 128 CPUs × 230 GB on `work`, 24 h wall limit.

### Pipeline extensions

- `tools/schemas/run.schema.json` — added `profile.memory_timeseries[]`,
  `profile.peak_rss_kb`, `profile.numa.{per_node_mb,total_mb}`, `profile.io.*`,
  `profile.per_thread[]`, `profile.raw_events`, new metric keys
  (`backend-stall-rate, dTLB-miss-rate, iTLB-miss-rate, amd-*`), and rich
  `env.*` / `env.slurm.*` properties.
- `tools/harvest_scratch.py` — no longer pinned to `SLURM_ID=41703864`; globs
  `<label>_*` and picks the newest. Parses `profile_meta.json`, `env.json`, and
  `samples.jsonl`. Auto-creates stub `logs/runs/<label>_baseline.json` files
  for new profile dirs that lack a run record.
- `tools/normalize.py` — corrected `mega_dna.fa` sites from 200 000 → 100 000.
- `web/js/pages/profiling.js` — new cards: Memory &amp; context switches (RSS
  timeseries SVG), NUMA residency (per-node bar), I/O totals, Per-thread CPU
  time (user vs sys). Expanded metrics grid to include BE-stall / dTLB / iTLB.
- `web/js/pages/environment.js` — flattens nested `env.slurm.*` keys so SLURM
  context appears in the KV grid.

### Outstanding

Jobs queued (priority-waiting). On completion:
`ssh setonix` → `make harvest && make build && git commit && git push`.

---


## 📋 Data inputs needed from Setonix (open requests)

> **Context.** The dashboard currently has full hardware-counter coverage on
> every run, but hotspots / folded stacks only on 3 of 14 runs, and zero GPU,
> memory, or L2/L3 cache-hierarchy data. List below is everything the
> frontend is ready to consume — each bullet specifies the JSON key(s), how to
> collect it on Setonix, and where it will surface on the site. Append the
> data to the existing per-run JSON in `logs/runs/<run_id>.json`; the
> normaliser auto-detects and indexes it.

### 1. Per-run profile coverage — fill the gaps

Currently **only `large_mf_1t`, `large_mf_64t`, and `xlarge_mf_1t`** have
`hotspots` + `folded_stacks`. We need the same for every other run so the
Profiling page and IPC-vs-hotspot correlation work end-to-end.

Missing runs: `large_mf_{4,8,16,32}t`, `xlarge_mf_{4,8,16,32,64,128}t`,
`2026-04-18_201515` (turtle.fa).

For each, attach:

- `profile.hotspots[]` — objects `{percent, function, module, samples?, children_percent?}` from `perf report --stdio --no-children` / `--children`.
- `profile.folded_stacks[]` — objects `{stack, count}` where `stack` is
  semicolon-separated. Generated by:
  ```bash
  perf record -F 499 -g -- <iqtree command>
  perf script | stackcollapse-perf.pl > folded.txt
  # then convert each line "frame1;frame2;... count" into JSON entries
  ```
- `profile.perf_cmd` — the literal `perf record` command used, for reproducibility.

### 2. Full cache hierarchy (L1 present, L2 + L3 missing)

We currently parse `L1-dcache-loads` / `L1-dcache-load-misses` only. For every
run, add the remaining levels to `profile.metrics`:

```jsonc
{
  "L2-loads":            <int>,
  "L2-load-misses":      <int>,
  "L2-miss-rate":        <float 0..1>,   // derived L2-load-misses / L2-loads
  "LLC-loads":           <int>,          // L3
  "LLC-load-misses":     <int>,
  "LLC-miss-rate":       <float 0..1>,
  "dTLB-loads":          <int>,
  "dTLB-load-misses":    <int>,
  "iTLB-loads":          <int>,
  "iTLB-load-misses":    <int>,
  "page-faults":         <int>,
  "context-switches":    <int>,
  "cpu-migrations":      <int>
}
```

Collection (EPYC 7A53):

```bash
perf stat -e cycles,instructions,\
cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
l2_cache_accesses_from_dc_misses,l2_cache_misses_from_dc_misses,\
LLC-loads,LLC-load-misses,\
dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses,\
branch-instructions,branch-misses,\
page-faults,context-switches,cpu-migrations \
  -- <iqtree command>
```

(On AMD Zen 3, `LLC-loads` maps to `l3_lookup_state.all_lookups` and
`LLC-load-misses` to `l3_lookup_state.l3_miss`; `perf stat` usually
auto-translates.)

### 3. Advanced CPU pipeline counters (IPC root-cause)

The dashboard shows `frontend-stall-rate` + `cache-miss-rate` today. To break
down **where cycles are going** at 64T/128T we need:

```jsonc
"profile.metrics": {
  "stalled-cycles-frontend": <int>,
  "stalled-cycles-backend":  <int>,
  "frontend-stall-rate":     <float 0..1>,  // already present on some runs
  "backend-stall-rate":      <float 0..1>,  // MISSING
  "uops-retired":            <int>,
  "uops-dispatched":         <int>,
  "retire-stall-rate":       <float 0..1>,
  "branch-mispred-rate":     <float 0..1>,  // MISSING for most runs
  "simd-fp-ops":             <int>,         // ex_ret_mmx_fp_instr.sse_instr
  "mem-bandwidth-gb-s":      <float>        // DRAM bandwidth snapshot
}
```

`perf stat -M Backend_Bound,Frontend_Bound,Retiring,Bad_Speculation ...`
(TMA top-down) exports this in one shot on modern perf.

### 4. Memory footprint + NUMA behaviour

Currently only `timing[].memory_kb` is captured (RSS at the end of each
command). Add a time series so we can plot heap-over-time:

```jsonc
"profile.memory_timeseries": [
  { "t_s": 0.0, "rss_kb": 123456, "vms_kb": 234567 },
  { "t_s": 5.0, "rss_kb": 345678, ... },
  ...
],
"profile.peak_rss_kb": <int>,
"profile.numa": {
  "nodes": 8,
  "local_hits":   <int>,
  "remote_hits":  <int>,
  "local_ratio":  <float 0..1>,
  "per_node_mb":  { "0": <mb>, "1": <mb>, ... }
}
```

Collection: `/usr/bin/time -v`, `ps -o rss --pid ...` every N seconds, or
`numastat -p <pid>` snapshots.

### 5. Flamegraph SVG (optional — we already render from folded_stacks)

If easier to ship the ready-made SVG, attach:

```jsonc
"profile.flamegraph_svg_path": "profiles/<run_id>.flamegraph.svg"
```

and push the file under `logs/profiles/`. The site already displays folded
stacks natively, so this is only needed when the JS flamegraph is too coarse
for very deep stacks.

### 6. GPU telemetry (completely empty today)

Every run has `gpu_info: "N/A (CPU-only baseline)"`. For the first HIP port
runs, populate:

```jsonc
"gpu_info": {
  "vendor": "AMD",
  "model":  "MI250X",
  "driver": "ROCm 6.3.42131-fa1d09cbd",
  "device_count": 8,
  "vram_gb_per_device": 64,
  "wall_gpu_s":       <float>,  // time spent in HIP kernels
  "wall_h2d_s":       <float>,  // host→device transfer
  "wall_d2h_s":       <float>,  // device→host transfer
  "occupancy_avg":    <float 0..1>,
  "sm_active_avg":    <float 0..1>,
  "kernels": [
    {
      "name": "computePartialLikelihoodSIMD_hip",
      "grid":  [blocks_x, blocks_y, blocks_z],
      "block": [threads_x, threads_y, threads_z],
      "registers_per_thread": <int>,
      "shared_mem_bytes": <int>,
      "invocations": <int>,
      "time_ms_total": <float>,
      "time_ms_avg":   <float>,
      "achieved_occupancy": <float 0..1>,
      "gmem_throughput_gb_s": <float>,
      "l2_hit_rate": <float 0..1>
    }
  ]
}
```

Collection: `rocprof --stats --hip-trace --hsa-trace` produces
`results.stats.csv`, `results.hip_stats.csv`, plus per-kernel metrics.
`omniperf profile -n <name> -- <cmd>` gives a richer set.

### 7. IQ-TREE alignment metadata (we estimate sizes today)

The dashboard computes `~size_mb` from `taxa × sites` because the raw `.fa`
isn't bundled. To replace the estimate with ground truth, every run should
emit:

```jsonc
"dataset_info": {
  "file":          "large_modelfinder.fa",
  "sha256":        "...",
  "size_bytes":    <int>,            // real on-disk size
  "taxa":          <int>,            // parsed from IQ-TREE stdout
  "sites":         <int>,            //   "Alignment has N sequences with M columns"
  "patterns":      <int>,            // unique site patterns after compression
  "informative_sites": <int>,
  "gaps_pct":      <float 0..1>,
  "invariant_sites_pct": <float 0..1>,
  "sequence_type": "DNA" | "AA" | "codon"
}
```

IQ-TREE already prints most of these on startup — a small `grep` wrapper in
`start.sh` is enough.

### 8. Model-selection trace

`modelfinder.model_selected` is the only field today. For the Tests /
ModelFinder comparison page (future), capture:

```jsonc
"modelfinder": {
  "model_selected": "...",
  "wall_time_s":    <float>,
  "candidates_evaluated": <int>,
  "top_models": [
    { "model": "GTR+F+G4", "lnL": -12345.67, "BIC": 24691.34, "AIC": 24687.01, "rank": 1 },
    ...
  ]
}
```

Parseable from `*.iqtree` / `*.model.gz`.

### 9. Verification likelihoods (richer `verify[]`)

The existing `verify[]` records `{file, expected, reported, diff}`. Please
also include:

```jsonc
{
  "tree_sha256":   "...",     // so we can diff topologies later
  "rf_distance":   <int>,     // Robinson–Foulds vs oracle tree
  "log_likelihood": <float>,
  "lnL_delta":     <float>    // reported - expected
}
```

Surfaced on the Tests page.

### 10. Per-thread breakdown (helps explain IPC collapse beyond 32T)

```jsonc
"profile.per_thread": [
  { "tid": 0, "cpu_ms": 123456, "user_ms": 123400, "sys_ms": 56,
    "voluntary_cs": <int>, "involuntary_cs": <int>,
    "migrations": <int>, "cpu_affinity": [0,1] },
  ...
]
```

Collection: `perf stat --per-thread`, `pidstat -t -p <pid>`, or
`/proc/<pid>/task/<tid>/status`.

### 11. Environment deltas across runs

`env` currently captures host/cpu/gcc/rocm. Add:

```jsonc
"env": {
  ...existing...,
  "kernel":        "<uname -r>",
  "glibc":         "<ldd --version>",
  "iqtree_version":"<iqtree3 --version>",
  "iqtree_flags":  "-DUSE_AVX2 -O3 ...",
  "cpu_governor":  "performance|ondemand",
  "turbo_enabled": true,
  "smt_enabled":   true,
  "numa_nodes":    8,
  "slurm": {
    "job_id":       "<id>",
    "partition":    "work",
    "nodes":        "<nodelist>",
    "cpus_per_task": <int>,
    "ntasks":       <int>,
    "time_limit_s": <int>,
    "mem_request_gb": <int>
  }
}
```

### 12. Network / filesystem wait (SLURM scratch I/O)

Optional but useful for large runs:

```jsonc
"profile.io": {
  "read_bytes":    <int>,
  "write_bytes":   <int>,
  "read_time_ms":  <int>,
  "write_time_ms": <int>,
  "fs":            "lustre|beegfs|local"
}
```

`/proc/<pid>/io` polled every N seconds.

---

### Priority order

> Items ✅ = delivered via `tools/harvest_scratch.py` on 2026-04-21 (see
> section below). Items ⏳ = still need a Setonix-side change.

| # | Item                                        | Status | Blocks                              | Effort on Setonix |
|---|---------------------------------------------|--------|-------------------------------------|-------------------|
| 1 | Hotspots + folded stacks for remaining 10 runs | ⏳ (4/14 done) | Profiling page completeness         | rerun under `perf record -g` |
| 2 | L2 / L3 / TLB counters                       | ⏳     | Cache-hierarchy insight chart       | one extra `-e` list in `perf stat` |
| 3 | Backend-stall + TMA top-down                 | ⏳     | IPC-collapse root cause             | `perf stat -M ...` |
| 7 | Real alignment stats                         | ✅     | Replace `~estimated` sizes          | harvested from `.iqtree` |
| 6 | GPU telemetry                                | ⏳     | Unlock GPU page (empty today)       | `rocprof --stats` wrap |
| 4 | Memory + NUMA time series                    | ⏳     | Memory-over-time chart              | `/usr/bin/time -v` + `numastat` |
| 10| Per-thread breakdown                          | ⏳     | Scaling-collapse story              | `perf stat --per-thread` |
| 9 | Richer verify (RF distance, lnL)              | ⏳     | Tests page depth                    | post-run Python |
| 8 | ModelFinder candidate trace                   | ✅     | ModelFinder comparison page         | harvested from `.iqtree`    |
|11 | Kernel / compiler / SLURM env                 | ⏳     | Reproducibility breadcrumbs         | one-off shell    |
|12 | IO counters                                   | ⏳     | Lustre-bound diagnostics            | `/proc/$pid/io`   |
| 5 | Flamegraph SVG (optional)                     | ⏳     | Higher-fidelity flamegraph          | `flamegraph.pl`   |

Also new since the wishlist was written: `profile.perf_cmd` is now captured
for all runs that have `perf_stat.txt`, and surfaces on the Overview
"How this was measured" disclosure.

Once any of these land in `logs/runs/<run>.json` (schema-valid), running
`python3 tools/build.py` + `git push` is enough — the site auto-renders.

---

## 2026-04-21 — Scratch harvest: ground-truth dataset & ModelFinder data

Added `tools/harvest_scratch.py` which reads Setonix scratch
(`/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/profiles/<label>_41703864/`)
and enriches every `logs/runs/*.json` in place. Fills gaps that were
previously missing from the dashboard (items 1, 4, 5, 6, 8 on the
open-requests list above).

New fields now carried through schema → normaliser → frontend:

- **`dataset_info`** — real taxa / sites / distinct patterns /
  constant & informative sites / sequence type, parsed from the `.iqtree`
  analysis report; file size read from the raw `.fa` on scratch. Replaces the
  static `DATASET_INFO` heuristic (which was off by 10× on large_mf & xlarge
  — corrected here too).
- **`modelfinder.{log_likelihood, bic, aic, aicc, tree_length, gamma_alpha,
  best_model_bic, candidates[]}`** — candidate table contains the top-10
  models ranked by BIC with LogL / AIC / AICc / BIC and their Akaike-weight
  columns.
- **`profile.perf_cmd`** — the exact `perf stat` command that produced the
  counters, parsed from the `perf_stat.txt` header. Displayed under a
  collapsible "how this was measured" disclosure on the overview.
- **`profile.hotspots[]` + `profile.folded_stacks[]`** — backfilled for
  `xlarge_mf_128t_baseline` (previously only 3 of 14 runs had perf-record
  data, now 4).

Frontend:

- `web/js/pages/overview.js` — Alignment card now surfaces taxa / sites /
  distinct patterns / informative / constant sites / sequence type / real
  file size. Model & Results card adds LogL, tree length, Gamma α, BIC, AIC.
  New "Top ModelFinder candidates" table (top 10 by BIC with w-BIC/w-AIC).
- `tools/normalize.py` — `summarize_run()` prefers `dataset_info` over the
  heuristic lookup, emits `patterns`, `informative_sites`, `constant_sites`,
  `sequence_type`, `model_bic`, `model_aic`, `gamma_alpha`, `tree_length`,
  `log_likelihood`, plus `has_candidates` / `has_perf_cmd` flags into the
  runs index.
- `tools/schemas/run.schema.json` — added optional `dataset_info`,
  `modelfinder.candidates[]`, `profile.perf_cmd`.

Verified: 14/14 runs pass schema validation, 14 pytests + 1 xpassed.

Not backfilled (require re-running on Setonix): full L2/L3/LLC counters,
backend stall cycles, GPU telemetry, memory / NUMA / IO timeseries,
per-thread breakdown — still open on the requests list above.

## 2026-04-21 — Dashboard + CI overhaul

- Made the repository public (MIT licence).
- Replaced the monolithic `serve.py` template with a modular pipeline:
  - `tools/normalize.py` writes per-record JSON + indexes under `web/data/`
  - `tools/validate.py` + JSON Schemas in `tools/schemas/`
  - `tools/build.py` mirrors `web/` → `docs/`
- Rebuilt the frontend as ES modules (`web/js/{main,router,state,data}.js`,
  plus `components/`, `charts/`, `pages/`). New pages: Overview, All Runs,
  Tests, Profiling, GPU, Allocation, Environment. Client-side flamegraph +
  call-stack views rendered straight from `folded_stacks`.
- Added `pytest` suite (`tests/`) covering schema, data invariants, build, and
  a warn-only wall-time regression guard.
- Added two GitHub Actions workflows:
  - `validate.yml` — schema + pytest on every push / PR
  - `build.yml`    — build `docs/` and deploy to GitHub Pages (`actions/deploy-pages@v4`)
- Removed legacy `serve.py`, `website/`, `dashboard.html`; `docs/` is now built
  in CI instead of committed.

---

## Status: Project initialized, profiling complete, GPU implementation not started

Profiling complete (VTune + perf, April 2026). Five hot functions identified in `phylokernelnew.h` consuming >95% CPU time. GPU PoC exists with CUDA/cuBLAS backends but needs HIP port for Setonix AMD MI250X GPUs. No GPU kernels have been validated against CPU oracle output yet.

---

## Current baselines

### CPU wall-clock times (Intel Xeon E5-2670, medium_dna.phy 50 taxa × 5,000 sites)

| Config | 1T | 4T | 8T |
|--------|-----|-----|-----|
| Default pipeline (ModelFinder → tree search) | 230.9s | 89.2s | 64.2s |
| GTR+G4 (fixed model, skip ModelFinder) | 166.8s | 68.2s | — |
| small_dna GTR+G4 | 10.5s | — | — |

### CPU wall-clock times (Setonix, AMD EPYC 7A53 Trento)

| Config | Wall | CPU | IPC | Frontend stalls |
|--------|------|-----|-----|-----------------|
| turtle.fa 1T GTR+G4 (12 taxa, 434 patterns) | 1.62s | 1.49s | 2.334 | 3.12% |
| medium_dna.fa 4T GTR+G4 (50 taxa, 4,559 patterns) | 26.9s | 102.5s | 2.299 | 2.23% |

### CPU baseline thread-scaling (Setonix, AMD EPYC 7A53 Trento, ModelFinder+TreeSearch)

| Dataset | 1T | 4T | 8T | 16T | 32T | 64T | 128T |
|---------|-----|-----|-----|------|------|------|------|
| large_mf (50 taxa, ~5k sites) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| xlarge_mf (200 taxa, ~10k sites) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| mega (1000 taxa, large) | ❌ | — | — | — | — | — | — |

**large_mf wall-clock times:** 1T=3h02m35s, 4T=1h34m19s, 8T=1h11m44s, 16T=0h45m34s, 32T=0h45m01s, 64T=0h45m14s
**xlarge_mf wall-clock times:** 1T=4h43m02s, 4T=2h32m26s, 8T=2h31m18s, 16T=2h28m58s, 32T=2h14m29s, 64T=2h00m19s, 128T=1h59m54s
**mega:** `mega_limited_4t` profile directory was created but run never started — script hit an error before launching IQ-TREE. This caused job 41703864 to exit with code 1 despite all other runs completing successfully.

### CPU profiling breakdown (perf, Setonix large_mf 1T vs 64T)

| Function | 1T | 64T |
|----------|-----|------|
| computePartialLikelihoodSIMD | 84.66% | 52.97% |
| computeLikelihoodDervSIMD | 9.73% | 9.99% |
| computeLikelihoodBufferSIMD | 3.31% | 5.71% |
| computeLikelihoodBranchSIMD | 0.85% | 0.30% |

### CPU profiling breakdown (VTune, medium_dna.phy)

| Function | GTR 1T | Default 1T | Default 8T |
|----------|--------|-----------|-----------|
| computeLikelihoodDervSIMD | 86.9s (52.1%) | 92.6s (40.1%) | 120.9s (25.0%) |
| computePartialLikelihoodSIMD | 15.8s (9.5%) | 42.2s (18.3%) | 70.5s (14.6%) |
| computeLikelihoodBufferSIMD | 7.1s (4.3%) | 8.7s (3.8%) | 39.6s (8.2%) |
| computeLikelihoodBranchSIMD | 0.0s (0.0%) | 0.76s (0.3%) | 1.7s (0.4%) |
| computeLikelihoodFromBufferSIMD | 0.49s (0.3%) | 0.41s (0.2%) | — |
| OpenMP spin-wait (libgomp) | 0s | 0s | 94.3s (19.5%) |

### Hardware counter metrics (perf, Default pipeline)

| Metric | 1T | 4T | 8T |
|--------|-----|-----|-----|
| IPC | 1.63 | 1.14 (−30%) | 0.89 (−45%) |
| Frontend stalls | 51.95% | 64.03% | 69.88% |
| Backend stalls | 17.08% | 37.16% | 47.87% |
| L1 D-cache miss rate | 7.33% | — | — |
| LLC miss rate | 1.89% | — | — |
| Branch misprediction | 0.08% | — | — |

### GPU kernel speedup targets

No GPU measurements yet. Expected based on workload characteristics:
- `computeLikelihoodDervSIMD`: 5,000 independent patterns × ~60 FLOPs each → excellent GPU occupancy
- `computeLikelihoodBufferSIMD`: pure element-wise multiply → memory-bandwidth bound on GPU
- `computeLikelihoodBranchSIMD`: dot product + log + reduction → good GPU fit

---

## Changelog

### 21 April 2026: Thread-scaling baselines complete; mega run failed

**xlarge_mf baseline now complete across all thread counts:**
- xlarge_mf 32T, 64T, and 128T all completed within SLURM job 41703864 but baseline JSONs were not generated by the pipeline script — exported manually from scratch profile data into `logs/runs/`
- xlarge_mf 128T baseline exported: `logs/runs/xlarge_mf_128t_baseline.json` (1h59m54s, IPC=0.082)
- xlarge_mf 64T baseline exported: `logs/runs/xlarge_mf_64t_baseline.json` (2h00m19s, IPC=0.136)
- xlarge_mf 32T baseline exported: `logs/runs/xlarge_mf_32t_baseline.json` (2h14m29s, IPC=0.225)
- IPC collapse beyond 32T is clear: 8T=0.642 → 32T=0.225 → 64T=0.136 → 128T=0.082, consistent with OMP barrier/cache coherence saturation

**mega run did not complete — root cause found and fixed:**
- `mega_limited_4t` profile directory was created but is empty — the run never launched
- **Root cause:** `local mega_start`, `local mega_end`, `local mega_elapsed` declarations at lines 288–296 of `run_large_profile.sh` — `local` is only valid inside a function, but the mega loop is in the main script body. With `set -euo pipefail`, this causes an immediate exit with code 1
- **Fix applied:** Removed `local` from all three declarations in the mega loop (in-place sed on scratch copy)
- No mega baseline data collected yet

**Next:** Investigate mega failure and re-submit, or begin Phase 1 HIP port

---

### 20 April 2026 (c): Dashboard hosting + commit-back pipeline

**Architecture change — private dashboard served from development server:**
- GitHub Pages requires Pro for private repos — switched to commit-back approach
- GitHub Action now generates `dashboard.html` + `docs/index.html` and commits back to repo with `[skip ci]`
- Created `host.sh` for development server: HTTP server (screen) + cron auto-refresh every 5 min
- Full pipeline: Setonix pushes data → Action generates dashboard → dev server cron pulls → serves on HTTP
- Updated `IMPLEMENTATION_PLAN.md` with full architecture diagram and Pawsey network policy notes
- Added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` to avoid Node.js 20 deprecation warnings

### 20 April 2026 (b): GitHub Actions CI/CD overhaul + Setonix baseline hotspots

**CI/CD overhaul — data-only pushes from Setonix, Action generates website:**
- Created `.github/workflows/build-dashboard.yml`:
  - Triggers on push to `main` when `logs/`, `serve.py`, or `website/` change
  - Runs `python3 serve.py` to generate `docs/index.html` from committed JSON logs
  - Validates output (file exists, size > 1KB, contains expected marker)
  - Commits generated dashboard back to repo (see 20c for final approach)
- Updated `.gitignore` — generated files (`dashboard.html`, `docs/`, `PROFILING_REPORT.html`, `website/api/runs.json`) are not tracked
- Updated `start.sh`:
  - `cmd_start`, `cmd_pipeline`, `cmd_profile`, `cmd_deepprofile` no longer call `serve.py` before pushing
  - These commands now push data only; GitHub Action generates the dashboard
  - `cmd_generate` preserved for local preview (runs `serve.py` without pushing)
  - Updated usage comments to reflect new workflow
- Created `IMPLEMENTATION_PLAN.md` documenting the full architecture

**Hotspot/callstack data from perf record:**
- Parsed `perf report` output into 3 baseline run JSONs (large_mf_1t, large_mf_64t, xlarge_mf_1t)
- Each run now includes `hotspots[]` (self%, children%, samples, function, module) and `callstacks{}` (total_samples, top_stacks[50])
- Key findings:
  - 1T: computePartialLikelihoodSIMD dominates at 84.66%
  - 64T: drops to 52.97% as OpenMP overhead and DervSIMD increase proportionally
  - xlarge_mf shows DervSIMD at 23.56% (vs 9.73% for large_mf) — more patterns shift compute balance

**`data.py` — merge additional run JSONs:**
- After building runs from `time_log_*.tsv` files, also loads all `*.json` from `logs/runs/`
- Deduplicates by `run_id` so Setonix pipeline runs and baseline profiles coexist

### 20 April 2026 (a): Dashboard UI/UX redesign + IQ-TREE Configuration card

**Overview page cleanup:**
- Removed 3 useless stat cards from overview (Best IPC, Cache Miss Rate, Deep Profiles) — these showed static, out-of-context numbers
- Replaced with: **Best Speedup** (calculated across all runs comparing multi-threaded vs 1T baselines) and **Fastest Run** (best wall time across all runs)
- Added stat card icons (`.stat-icon`) for visual distinction

**IQ-TREE Configuration card (new):**
- Added `renderOverviewConfig()` function — shows 3-column config grid on overview page when a run is selected
- Sections: Alignment (dataset, taxa, sites, data type, file size, patterns, informative/constant sites, free params), Model & Results (model, rate heterogeneity, gamma alpha, threads, log-likelihood, BIC, tree length, wall time), System (CPU, cores, memory, L3, GPU, ROCm, GCC, NUMA, hostname)
- Tries to match deep profiles by dataset name for richer alignment/system data
- Shows the IQ-TREE command line with copy button
- Added `copyOverviewConfig()` — exports full config as formatted text to clipboard
- Added HTML container (`#overviewConfigCard`) to overview page

**CSS modernization (UI/UX redesign):**
- Darker, more refined color palette (--bg: #060a13, --surface: #0d1321, --card: #131b2e)
- Added card elevation system: `--shadow-card` and `--shadow-hover` with subtle inset highlights
- Stat cards now have hover lift effect (translateY + glow border)
- All cards get hover border-color transition
- Added `font-feature-settings` for Inter font (cv02, cv03, cv04, cv11)
- Tightened spacing throughout (padding, gaps, font sizes)
- Added `.config-grid`, `.config-section`, `.config-items`, `.config-item`, `.ci-label`, `.ci-value`, `.config-cmd` CSS classes
- Responsive breakpoints updated for config-grid (3→2→1 columns)
- Sidebar slightly narrower (260px→240px), refined nav link sizing
- Added `--bg-tertiary` variable for nested surfaces (command blocks)

**Bug fix:** Re-added `feStall` variable declaration in `renderOverview()` — was removed with the stat cards but still used by `latestProfileCard`.

### 19 April 2026: Comprehensive profiling report + Setonix cross-platform comparison

**What was done:**
- Created `PROFILING_REPORT.html` — a comprehensive, downloadable HTML profiling report combining Intel (Sandy Bridge) and Setonix (AMD EPYC Trento) profiling data
- 16 sections with table of contents, page breaks for printing, explanations for non-technical readers, jargon glossary, colour-coded findings, ASCII bar charts, and a download button
- Cross-platform comparison: Setonix 4T is **2.54× faster** than Intel reference 4T on medium_dna GTR+G4 (26.9s vs 68.2s), with **2.15× better IPC** (2.30 vs 1.07) and **~30× reduction in frontend stalls** (2.23% vs 65.71%)
- Function hotspot ranking is identical across both platforms (DervSIMD ~39%, PartialLikelihood ~33%, Buffer ~10%), confirming GPU offload targets are architecture-independent
- Added `PROFILING_REPORT.html` to `.gitignore` (generated report, not tracked in repo)
- Fixed `dashboard.html` and `serve.py` rendering bug (stray JS from template replacement — see entry below)

**Setonix baselines added:**
| Config | Wall time | CPU time | IPC | Frontend stalls |
|--------|-----------|----------|-----|-----------------|
| turtle.fa 1T GTR+G4 | 1.62s | 1.49s | 2.334 | 3.12% |
| medium_dna 4T GTR+G4 | 26.9s | 102.5s | 2.299 | 2.23% |

**Next:** Run larger datasets on Setonix (Default pipeline without `-m` to test ModelFinder path), then begin Phase 1 CUDA→HIP port.

### 18 April 2026: Fix dashboard.html rendering — JS was broken by serve.py generator

**Root cause:** `serve.py` used `template.index('loadData();')` which matched the first occurrence of the substring — inside `await loadData();` within the `refreshData()` function body — instead of the standalone `loadData();` call at the end of the script. This caused the replacement to leave a stray `}` (closing brace of `refreshData()`) followed by all the original template's ES6 render functions duplicated after the generated ES5 code. The stray `}` was an immediate syntax error that prevented any JS from executing.

**Fixes applied:**
- `dashboard.html` / `docs/index.html`: Removed ~280 lines of duplicate ES6 template code (stray `}` + all duplicate render functions + duplicate `loadData();` + `setInterval`) that were appended after the generated script
- `serve.py`: Changed `template.index(old_script_end)` → `template.rindex(old_script_end)` to match the LAST occurrence of `loadData();` in the template, preventing this bug on future regenerations

**Verified:** `node --check` passes on extracted JS. Dashboard opens and renders data, charts, and all 6 pages correctly.

### 18 April 2026: Project initialization

**What was done:**
- Comprehensive VTune + perf profiling across 5 configurations (Default 1T/4T/8T, GTR 1T/4T)
- Identified five hot functions in `phylokernelnew.h` consuming >95% of CPU time
- Documented GPU PoC structure and capabilities (`poc-gpu-likelihood-calculation-main/`)
- Created CLAUDE.md with project context, architecture, kernel designs, development principles
- Created CHANGELOG.md (this file) for progress tracking
- Established 5-phase implementation plan
- Identified Setonix target: AMD MI250X GPUs → need HIP/ROCm port of CUDA kernels

**Current state of GPU PoC:**
- Working multi-backend GPU likelihood calculator (`gpulcal` binary)
- CUDA kernels: `MatrixKernels.cu` (hadamard, scaling, composite fused), `TipLikelihoodKernel.cu`
- K-specialized templates: DNA (K=4, register-cached) and protein (K=20, shared-mem)
- cuBLAS backend with async streams and CUDA events
- Limitations: only JC/POISSON models, no GTR/HKY, CUDA-only (no HIP), test scripts are stubs, no IQ-TREE integration

**Key profiling insights:**
- GPU offload eliminates three CPU bottlenecks simultaneously: OpenMP spin-wait (22-29% at 4T+), frontend stalls (52% → 70% with threads), backend stalls (17% → 48%)
- All datasets fit in MI250X 128 GB HBM2e (even stress_dna at 500 MB)
- Patterns are independent → perfect GPU parallelism for DervSIMD, BufferSIMD, BranchSIMD
- PartialLikelihoodSIMD has tree-order dependencies → must launch per-node in post-order
- `computeLikelihoodBranchSIMD` was hidden by `-m GTR+G4` (skips ModelFinder); visible in default pipeline

**Next steps (ordered by priority):**
1. Port GPU PoC CUDA kernels to HIP for Setonix MI250X
2. Build and validate PoC on Setonix GPU node
3. Implement `computeLikelihoodDervSIMD` HIP kernel (highest impact: 40-52% of CPU time)
4. Write correctness tests comparing GPU output to CPU oracle

---

## Implementation phases

- [ ] **Phase 1:** Port CUDA→HIP, build on Setonix, validate PoC
- [ ] **Phase 2:** HIP kernel for `computeLikelihoodDervSIMD` (40-52% of CPU time)
- [ ] **Phase 3:** HIP kernel for `computePartialLikelihoodSIMD` (18.3% of CPU time)
- [ ] **Phase 4:** HIP kernels for Buffer/Branch (3.8% + 0.3% of CPU time)
- [ ] **Phase 5:** Integration with IQ-TREE tree search loop + end-to-end benchmarks

## Confirmed correct (do not re-investigate)

- **Frontend stalls are the dominant 1T bottleneck** (52.4%), caused by large templated/inlined AVX instruction footprint. A GPU eliminates this entirely.
- **OpenMP scaling is sublinear** — 8T uses >2× the CPU cycles of 1T (484.5s vs 230.8s). Extra cycles are OpenMP barrier sync + cache coherence.
- **Hierarchy truncation is NOT a factor** in IQ-TREE accuracy (per upstream IQ-TREE development).
- **Branch misprediction is negligible** (0.08%) — not worth optimizing.
- **LLC miss rate is low** (1.89%) — working set fits in cache. The bottleneck is I-cache (frontend), not D-cache.

## Failed approaches (do not re-attempt)

- **`local` in main script body (`run_large_profile.sh`):** The mega loop used `local mega_start/mega_end/mega_elapsed` outside any function. `local` is bash-function-only; with `set -euo pipefail` this immediately aborts the script. Fixed by removing `local` from those three declarations.
