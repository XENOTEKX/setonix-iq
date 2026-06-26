# L2 Batched-NNI Search-Axis Redesign — Plan + Honest Red-Team Verdict

**Tree:** `iqtree3-l2search` (branch `l2-batched-nni`, baseline commit `37a63740` = brute-force JOLT `--ts-fused` source snapshot).
**Frozen baseline binary:** `frozen-binaries/iqtree3-jolt-bruteforce.1a924889` (current production brute-force JOLT, post-A3).
**Date:** 2026-06-26. Built with agents (2× code-map, 1× Plan, 1× hostile red-team) under "ultracode".

---
## ⚠️ DECISION: NO-GO — measured by the L2.0 spike (job 172388123, 2026-06-26)
Single-GPU K-batching is falsified. On a real H200, `cudaOccupancy` on the verbatim kernels:
`k1_node` 40 regs → 6 blk/SM → **K-to-fill 2.03**; `kj_derv_fused` 32 regs → 8 blk/SM → **K-to-fill 2.70**.
One AA-100K tree (391 blocks) already saturates the GPU; K=8–12 oversubscribes 3–4×.
Throughput at K=8: serial 490.7ms / batched 440.9ms (1.11×) / streams 403.8ms (**1.22×**); `batched_ms(K)/batched_ms(1)≈K` = time-sliced.
**Best K=8 = 1.22× vs the 3.0× gate → NO-GO.** Do NOT build L2.1–L2.5. **Pivot to GPU+MPI hybrid** (R ranks × 1 GPU, MPI search-axis scaling on top of per-GPU Fix-B). The staged plan below is retained as the record of *why* the single-GPU path was rejected.

---

---

## 0. The problem & the two axes

IQ-TREE tree search has two parallelism axes:
- **Likelihood axis** — scoring many candidate moves *within one tree*. This is what the GPU `--ts-fused` path accelerates today (screener `gpu_screen_nni_tile_crosscheck` `gpu_lnl_intree.cu:1418`; JOLT all-branch reopt `gpu_jolt_optimize:1643`).
- **Search axis** — the `-ninit` independent starts + ~100 stop-rule perturbation iterations (`iqtree.cpp:doTreeSearch:2448`), embarrassingly parallel. **This is what IQ-TREE-MPI exploits** (per-rank seed `base+rankID` `iqtree.cpp:691,835`; merge best trees `syncCurrentTree:5394`).

**Established state of play:** GPU `--ts-fused` accelerates the *wrong axis* for wall-clock competitiveness. On AA-100K LG+G4: current 1 GPU ≈ 1688s; projected ≈ 600–900s after **Fix B** (removes within-tree per-edge syncs); 1 CPU node ≈ 950s; **4-node MPI ≈ 202s**. Goal: beat 202s on **one** H200.

**What is batched today:** MOVE-level within ONE tree only. Zero multi-tree concurrency — enforced by a process-global mutex (`gpu_lnl_intree.cu:1663`), global `__constant__` model symbols (`:34-43`), and static `DevBuf` pools (`:510-515,1631-1634`). The code explicitly defers multi-tree batching to future "PHALANX grid.z" work.

## 1. The proposed redesign (Design A)

Run **K independent seeded perturbation searches resident + concurrent on one H200**, batching the screener and JOLT kernels over a tree index (`grid.z=K`) so each launch is filled with K trees' work and per-launch/per-sync latency amortizes across K. Merge by keeping the best ML tree (MPI search semantics). Staged:

| Stage | Deliverable | Gate | Effort | Risk |
|---|---|---|---|---|
| **L2.0** spike | K-concurrency speedup table on the *real* kernels at AA-100K | per-replica bit-identity | 2–3 d | **the whole bet** |
| **L2.1** de-globalize | `__constant__`+`DevBuf`→per-batch `[K]` arrays; mutex→per-batch barrier; K=1 no-regress | pass@1e-9, lnL/RF==0 vs frozen | 1.5–2 wk | med-high |
| **L2.2** batch screener | `grid.z=K` screener; flatten per-node DFS sync (`:1535`) via level-batched ragged launches+mask | per-slot bit-identity, 28/28 | 2–3 wk | high |
| **L2.3** batch JOLT | `grid.z=K` LM reopt w/ ragged-convergence mask; composes w/ Fix B | JOLT_AUDIT 1e-9 | 3–4 wk | high |
| **L2.4** host K-loop | `--l2-k K` end-to-end; per-search `rstream=base+id`; merge via CandidateSet | beats 202s @ K=8–12 | 2 wk | med |
| **L2.5** K/tile budget | joint (K,nTile) auto-pick; graceful degrade | no OOM, gates hold | 1 wk | low |

Total ≈ **10–14 weeks**.

## 2. HONEST RED-TEAM VERDICT — ⚠️ likely cannot beat MPI on one GPU

A hostile red-team (and my own re-derivation) found the plan's load-bearing assumption is **false at the scale that matters**:

- **BLOCKER 1 — Saturation ceiling.** One AA-100K launch = `ceil(100000/256)=391` blocks. H200 = 132 SMs × (4–5 blocks/SM for the register-heavy `k1_node`) ≈ 528–660 resident blocks → **K-to-fill ≈ 1.35**. One tree already fills the GPU; `grid.z=K` (K=8–12) just **time-slices** → total compute scales ~K×. The only saving is per-launch overhead, so **max per-tree speedup = 1 + T_overhead/T_compute**. The "SQUEEZE": small Pn = launch-bound (K helps) but trivial work (MPI wins, GPU starved); large Pn = compute-bound (one tree fills GPU, K barely helps). **No useful middle regime at AA-100K.** *(Robust across 32–64 reg/thread: even at 8 blocks/SM, K-to-fill ≈ 2.7, so K=8–12 still oversubscribes 3–4×.)*

- **BLOCKER 2 — Fix B already ate the overhead.** The 1688s is ~83% per-edge-sync-bound JOLT LM (`:1957,1960,1961`); Fix B removes those syncs (1688→600–900s) — i.e. Fix B **already recovers the per-launch/per-sync slack** L2 wants. Post-Fix-B is compute-bound, residual overhead ~10–30% → **L2 ceiling ≈ 1.1–1.4×**. Beating 202s from a 600–900s post-Fix-B baseline needs **3.0–4.5×**. The plan only looks viable because it measures K-speedup against the *pre-Fix-B 1688s*, double-counting overhead Fix B already removed.

- **BLOCKER 3 (strategic) — GPU+MPI hybrid is strictly better.** R MPI ranks × **1 GPU each**, each running the existing single-GPU JOLT+Fix-B path. Zero new device code, reuses MPI's *proven near-linear search scaling* (`syncCandidateTrees:5324`), composes multiplicatively with Fix B. On a 4-GPU node: 4 × (600–900s concurrent) with MPI search scaling lands at/below 202s — without any of L2.1–L2.5's de-globalization/ragged-DFS/budgeting risk.

- **MAJOR — exactness-gate contradiction.** L2.3/L2.4 "RF==0 vs frozen single-search baseline" is **wrong**: per-search seeds explore different trajectories → generally different (or equal-lnL-different) topology. Correct gate = **(a)** per-slot per-move/per-tree likelihood pass@1e-9 / JOLT_AUDIT≤1e-9 on fixed topologies, **(b)** final best lnL ≥ baseline (recomputed exactly). RF==0 only applies to the fixed-topology likelihood-engine check, never the search outcome.

- **MAJOR — ragged-DFS fragility.** K trees have wildly different depths (balanced ≈7 vs caterpillar ≈100 for 100 taxa). Level-batching stalls all K on the deepest tree; masking wastes blocks at deep levels; per-slot Kahan order (0..nptn-1) must be proven preserved under `grid.z` scheduling — the likeliest correctness regression.

- **MINOR — memory fits but isn't throughput.** ~7.8–9.5 GB/tree → K_max ≈ 14 in 141 GB. Loading 12 trees is possible; they then time-slice (Blocker 1).

### Verdict
**Do NOT commit to L2.1–L2.5 on the current premise.** The GPU is already near its useful limit for ONE tree search (screener + JOLT + Fix B); single-GPU K-batching recovers overhead Fix B already removed and is occupancy-capped. **Recommended: pivot to GPU+MPI hybrid.** If L2 evidence is still wanted, gate the whole program on **one** cheap measurement:

> **L2.0 DECISION SPIKE:** run the real `k1_node`/`kj_derv_fused` at AA-100K with `grid.z=K`, report **per-tree wall-clock speedup, K=8 vs K=1**, measured against the **post-Fix-B** kernel structure.
> **GO** only if ≥ **3×**. ~1.3× (the predicted result) ⇒ L2 cannot beat MPI ⇒ redirect the 10–14 weeks to the hybrid.

This file is the durable record; the implementation log + `project_gpu_tree_search` memory carry the pointer.
