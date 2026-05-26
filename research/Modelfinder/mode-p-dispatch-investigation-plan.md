# Mode P Dispatch: Root-Cause Investigation & Novel Architecture Design Plan

**Author:** as1708  
**Date:** 2026-05-25  
**Status:** ACTIVE — written after P.7 perf gate failure (job 169211688)  
**Target reader:** Claude Opus 4.7 (code-and-research subagent)

---

## 1. Executive Summary of the Problem

P.7 perf gate (AA 1M, np=16, `--mode-p`, group_size=2) was **killed at walltime 30:38**
(exit -29). Estimated MF wall > 1,800 s. The FCA np=16 baseline is **1,122 s**.
Mode P with group_size=2 is at minimum **60% SLOWER than FCA** on the same hardware.

This is not merely a regression — it is a structural failure of the current MPGC
design. All further P.X perf gates are blocked until this is understood and fixed.

---

## 2. Measured Performance Data

### 2.1 P.7 MF-TIME records (rank 0 only — all that completed before kill)

```
model             dispatch   dt (s)   FCA np=16 ref   Δ
─────────────────────────────────────────────────────────
LG+F              FCA         16.3          —           —
LG+F+I            FCA         62.1          —           —
LG+F+G4           MODEP      101.3        ~89.2       +13.5%  ← regression
LG+F+I+G4         MODEP      446.9        ~332.8      +34.2%  ← severe regression
Q.YEAST+F+G4      MODEP       54.9          —           —
PMB+G4            FCA        173.7          —           —
MTART+F+G4        MODEP      132.6          —           —
MTINV+G4          FCA        276.9          —           —
HIVW+F+G4         MODEP      109.7    (in progress at kill)
```

**The heaviest model in the workload (LG+F+I+G4) takes 34% MORE time in Mode P
than in FCA np=16**, even though Mode P group_size=2 should halve the pattern work.

### 2.2 Per-rank Mode P completions at walltime kill (t = 1,800 s)

```
Group  Ranks  Mode P models completed at kill   Last model in progress
──────────────────────────────────────────────────────────────────────────
0      0,1    5 (LG+F+G4, LG+F+I+G4, Q.YEAST+F+G4, MTART+F+G4, HIVW+F+G4)
1      2,3    4 (WAG+F+G4, WAG+F+I+G4, JTTDCMUT+F+I+G4, MTZOA+F+I+G4)
2      4,5    4 (JTT+F+G4, JTT+F+I+G4, DCMUT+F+I+G4, MTMET+F+I+G4)
3      6,7    4 (Q.PFAM+F+G4, Q.PFAM+F+I+G4, VT+F+G4, VT+F+I+G4)
4      8,9    2 (PMB+F+G4, [another])
5      10,11  1 (Q.MAMMAL+F+I+G4 started at t≈989, still running at kill)
6      12,13  1 (Q.INSECT+F+I+G4 started at t≈989, still running at kill)
7      14,15  0  ← ZERO Mode P models in 30 minutes
```

Groups 5, 6, 7 are catastrophically behind Groups 0–3. Groups 5 and 6 each started
their first +F+I+G4 model at t=989 s, and were still running at kill (~811+ s per model).
Group 7 never reached a Mode P model at all in 30 minutes.

### 2.3 Key numbers for the design target

| Metric                              | Value           |
|-------------------------------------|-----------------|
| FCA np=16 MF wall                   | 1,122.363 s     |
| FCA np=16 total wall                | 2,410.226 s     |
| P.7 MF wall (estimated, gs=2)       | > 2,000 s       |
| P.7 perf gate target                | ≤ 600 s         |
| Required MF speedup over FCA np=16  | ≥ 1.87×         |
| Alignment                           | AA 1M, 946,439 unique patterns |
| Pattern split at gs=2               | [0, 473,224) and [473,224, 946,439) |
| Heavy models (P6-DIAG)              | 56 / 224 (threshold = 1.5× avg_cost) |
| MPGC groups at gs=2, np=16          | 8 groups of 2 ranks |
| OMP threads per rank                | 103              |
| Nodes                               | 16 Gadi SPR (104-core, 4 NUMA) |

---

## 3. Source Code Architecture (as-built, P.7 binary `cc3d403f`)

**Key files:**
```
src/iqtree3-mode-p-iso-p3/
  main/phylotesting.cpp    — evaluateAll(), filterRatesMPI(), P.6 dispatcher, MPGC setup
  main/phylotesting.h      — CandidateModelSet, CandidateModel declarations
  tree/phylokernelnew.h    — computeLikelihoodBranchGenericSIMD (P.3),
                             computeLikelihoodDervGenericSIMD (P.4),
                             computeLikelihoodBranchSIMD / P.5a summation loop
  tree/phylotree.cpp       — modePAllreduceLh / modePAllreduceLhDfDdf (line 1014),
                             computeTraversalInfo (line 6201),
                             initializePtnPartition (line ~960)
  tree/phylotree.h         — PhyloTree MPGC members: ptn_start, ptn_end,
                             mp_group_rank, mp_group_size, mp_allreduce_comm
```

### 3.1 MPI Allreduce payload (confirmed small)

```cpp
// phylotree.cpp:1014 — 1 double per modePAllreduceLh call
MPI_Allreduce(&tree_lh_local, &tree_lh_global, 1, MPI_DOUBLE, MPI_SUM, mp_allreduce_comm);

// phylotree.cpp:1043 — 3 doubles per modePAllreduceLhDfDdf call  
MPI_Allreduce(in, out, 3, MPI_DOUBLE, MPI_SUM, mp_allreduce_comm);
```

The allreduce payload is trivially small (8–24 bytes). At 2 µs InfiniBand latency,
even 10,000 allreduces per model = 20 ms total allreduce overhead. This CANNOT explain
the 114 s regression for LG+F+I+G4 (446 s vs 332 s).

### 3.2 Pattern slice enforcement in computeLikelihoodBranchGenericSIMD (P.3)

```cpp
// phylokernelnew.h ~line 2839–2854 (P.3 Mode P):
const bool   mp_active = isModePActive();
const size_t mp_lo     = mp_active ? ptn_start : 0;
const size_t mp_hi     = mp_active ? std::min(ptn_end, nptn) : nptn;
const size_t mp_size   = (mp_hi > mp_lo) ? (mp_hi - mp_lo) : 0;
vector<size_t> limits;
computeBounds<VectorClass>(num_threads, num_packets, mp_size, limits);
if (mp_active) {
    for (size_t &lim : limits) lim += mp_lo;
}
```

The SUMMATION loop (log-lh accumulation) IS restricted to [ptn_start, ptn_end).
But `computeTraversalInfo` (tree traversal to compute partial_lh) is called BEFORE
this restriction with no slice arguments. Whether the partial_lh pass is restricted
to the Mode P slice must be confirmed by profiling.

### 3.3 `num_packets` allocation concern

`num_packets` is set once at tree initialization (likely = num_threads = 103), based
on the FULL `orig_nptn = 946,439`. With Mode P active, `mp_size = 473,224`, and
`computeBounds(103, num_packets, 473224, limits)` produces `num_packets` buckets of
`473224 / num_packets` ≈ 4,594 patterns each.

If `num_packets` was computed using `orig_nptn = 946,439` patterns, the OpenMP task
structure has 103 packets × (internal tree nodes ≈ 98) = 10,094 `computePartialLikelihood`
calls per model evaluation. In Mode P each call covers 4,594 patterns.
In FCA each call covers 9,188 patterns.
The OVERHEAD per call (function setup, scheduler, OMP barrier, NUMA miss on partial_lh
array boundaries) might dominate the 4,594-pattern payloads.

---

## 4. Root-Cause Hypotheses (ordered by likelihood)

### H1 — Load Imbalance: static MPGC group assignment ← CONFIRMED DOMINANT

The 8 MPGC groups are assigned to substitution model families at Phase 1.
Group 0 = LG family (reference, includes LG+F+I+G4 as model 7).
Groups 1–7 = WAG, JTT, Q.PFAM, Q.MAMMAL, Q.INSECT, etc.

Group 7 (ranks 14,15) was assigned a family with NO models above the P.6 cost
threshold. They processed only light FCA models for the entire 30-minute run.
Groups 5 and 6 did not reach their first heavy model until t=989 s, implying a
very long queue of light FCA models before the first heavy one.

Under FCA np=16, the same 56 heavy models are distributed across 16 ranks via
LPT (Longest Processing Time first) dynamic dispatch. FCA has NO static family
assignment — any rank picks up any model from the global queue. This gives nearly
perfect load balance.

MPGC's static group ↔ family binding eliminates the benefit of dynamic work-stealing,
causing Groups 5, 6, 7 to idle on light models while Groups 0–3 are overloaded.

**Impact**: Even if Mode P were 2× faster per heavy model, the load imbalance alone
makes the total MF wall bounded by Groups 5–7 which finish at ≥ 1,440+ seconds.

### H2 — Pattern-Parallel Regression: work not halved as expected ← SUSPECTED

Despite the Mode P slice restriction in the summation loop, `computeTraversalInfo`
runs unconditionally on the FULL tree. If the resulting calls to
`computePartialLikelihood(info, ptn_lower, ptn_upper)` use the SLICED range
(from `limits[]` above), the tree traversal IS halved. But if the tree traversal
uses a separately-computed `limits[]` based on `orig_nptn` (a possible code path
in the threaded scheduler), both ranks would compute partial_lh for ALL 946K patterns.

**Evidence**: LG+F+G4 = 101 s in Mode P vs 89 s in FCA. If both ranks are computing
full 946K patterns (same as FCA), the overhead of 12 s matches plausible allreduce
and OMP barrier costs. For LG+F+I+G4: 447 s vs FCA's 332 s — if both ranks do full
work AND there is additional synchronization overhead from the allreduce barrier
(one rank finishes patterns [0,473K) faster than the other finishes [473K,946K)),
the slower rank's time could be 332 s × ~1.35 = 448 s. This matches exactly.

**This would explain the mystery**: if Mode P does NOT actually split the
computePartialLikelihood work (tree traversal), both ranks do full 946K-pattern
computations AND block on each allreduce until the other rank is ready.
The slower rank dominates → total time > FCA serial time.

### H3 — OMP Thread Scaling: 103 threads × 473K patterns → cache underutilization

Each OMP thread in Mode P handles 473K/103 ≈ 4,594 patterns per packet.
The key hot data is `theta_all`: 946K (full!) × block (80) × 8 bytes ≈ 590 MB.
In Mode P only [ptn_start, ptn_end) of `theta_all` is written, but the array is
allocated at FULL size. The working set for a 103-thread OMP team touching only
half of a 590 MB array is ~295 MB — still larger than the Sapphire Rapids LLC (105 MB).
DRAM bandwidth is the bottleneck either way, but the cold half of `theta_all`
may cause additional TLB pressure and prefetcher confusion.

### H4 — NUMA First-Touch: fresh iqtree per model, NUMA-0 allocation

`CandidateModel::evaluate()` creates a fresh `IQTree` per model (as confirmed by the
B.4-15 fix). `initializeAllPartialLh()` allocates `central_partial_lh` and
`theta_all` on the calling thread's NUMA node (NUMA 0 of rank 0's socket). When
103 OMP threads then access this array, 75 of them (on NUMA 1–3) incur cross-NUMA
latency (~50–120 ns vs ~13 ns local). At 590 MB and ~30 GB/s intra-NUMA bandwidth,
a cold DRAM pass takes ~20 ms. Over 1,000+ optimization steps this could be
significant, but IQ-TREE uses the `numactl --localalloc` binding — which allocates
on the first-touch NUMA node of the thread doing the allocation (the main thread),
not necessarily NUMA-aware. This is a pre-existing issue in FCA too, however, so
it cannot explain the REGRESSION relative to FCA.

### H5 — filterRatesMPI Barrier: Phase 1 serial dependency for Group 0

filterRatesMPI fires when Group 0 finishes the LG reference family. All 8 groups
wait at the fca_comm barrier in filterRatesMPI. Group 0 finishes LG at t≈627 s
(MF-TIME confirms: ref_remaining=0 starts at model Q.YEAST+F+G4, t=627).
Groups 1–7 must have finished their own reference families by t=627 (since they're
blocked on fca_comm Bcast). After filterRatesMPI, Phase 2 dispatch reassigns models
globally.

The fca_comm Bcast is intra-group (B.4-14 fix), NOT world-wide. So filterRatesMPI
does NOT cause a world barrier — it fires independently per group. This means H5
is NOT a blocking issue; groups 4–7 can fire their own filterRatesMPI whenever
their own reference family is done.

---

## 5. What Must Be Profiled and Measured

The investigation agent (Claude Opus 4.7) should collect the following data before
proposing architectural changes.

### 5.1 Verify H2 (does computePartialLikelihood see the slice?)

**Task:** Instrument `computePartialLikelihoodGenericSIMD` (phylokernelnew.h:1299)
with a diagnostic that prints `ptn_lower, ptn_upper` for the FIRST call per model
per rank (guard with a static/atomic `once_per_model` flag). Submit a short diagnostic
run (100K AA, np=2, --mode-p-all, K_outer=1). Check whether ptn_lower/ptn_upper
for rank 0 are [0, 48K) and for rank 1 are [48K, 96K), or if both see [0, 96K).

**Expected outcome if H2 correct:** Both ranks see [0, 96K) → tree traversal
is NOT restricted to the Mode P slice → both ranks do full work → allreduce is
a barrier that serializes the faster half behind the slower → total > FCA.

**Expected outcome if H2 wrong:** Rank 0 sees [0, 48K), rank 1 sees [48K, 96K) →
tree traversal IS halved → overhead from H3/H4/Allreduce latency is the bottleneck.

### 5.2 Measure per-model OMP timing breakdown

**Task:** Add `omp_get_wtime()` markers around:
1. `computeTraversalInfo` call
2. The OpenMP packet loop (partial_lh computation + summation)
3. `modePAllreduceLh` call

Print per-model timing for one reference run (AA 100K, np=2, --mode-p-all, LG+F+G4
and LG+F+I+G4 only). Compare Mode P vs FCA wall times for each sub-phase.

### 5.3 Count allreduce calls per model

**Task:** Add an atomic counter to `modePAllreduceLh` and `modePAllreduceLhDfDdf`.
Print the total count at model completion via MF-TIME line. For LG+F+I+G4 at AA 1M,
record the allreduce count. Multiply by 2 µs to estimate total allreduce overhead.

### 5.4 Model cost assignment per MPGC group (AA 1M, np=16)

**Task:** Add a per-group MF-MPI-DIAG2 dump at the start of Phase 2 that lists:
for each MPGC group (group_id), the ordered list of model costs assigned to that
group's Phase 2 queue. Identify the bottleneck group (the one with the largest
total cost in Phase 2).

**Key question:** Why does Group 7 have zero heavy models in 30 minutes?
Is it because its model queue has no models above the P.6 threshold, or because
the LPT assignment gave it a very long queue of cheap models?

---

## 6. The Fundamental Architecture Problem

### 6.1 The tension between pattern-parallelism and model-parallelism

IQ-TREE's ModelFinder at AA 1M has two independent sources of computational work:

```
Total work W = Σ(models i) cost(i)

With FCA np=16 (pure model-parallel):
  Each rank handles ~W/16 models independently
  No synchronization — perfect Amdahl
  Wall ≈ W/16 = 1,122 s

With Mode P gs=2, np=16 (pattern × model hybrid):
  8 groups of 2 ranks
  Each group handles ~W/8 models in FCA-parallel across groups
  For heavy models within a group: 2-rank pattern split → 2× speedup
  But groups are statically bound to model families → load imbalance
  And the pattern split appears not to speed up individual model evaluation
  Wall > W/8 = 2,244 s (worse than FCA np=16!)
```

The core insight: **at AA 1M, the workload is NOT dominated by a few
heavy models** to an extent that justifies the group_size=2 overhead.
P6-DIAG shows 56/224 models are "heavy" — but the bottleneck is the few
very heavy ones (LG+F+I+G4, Q.MAMMAL+F+I+G4, Q.INSECT+F+I+G4 each ~350–450 s)
surrounded by hundreds of ~5–200 s models. Dynamic load balancing (FCA) handles
this naturally. Static family binding (MPGC) does not.

### 6.2 The 10M scaling projection

At AA 10M (×10 more patterns):
- FCA np=16 MF wall ≈ 10 × 1,122 = ~11,220 s (assuming linear scaling — optimistic)
- FCA np=32 would give ~5,610 s (if more nodes available)
- The heaviest models (LG+F+I+G4-class) would take ~3,300 s each
- With 50+ models of this cost, no amount of model-parallel FCA can finish in < 1 hr

At 10M DNA:
- Different model space (GTR+G4 etc.), fewer AA models
- Pattern count similar order of magnitude
- +I+G4 and other rate-het models still dominate

**The goal**: achieve ≤ 600 s MF wall at AA 1M, and ≤ 6,000 s at AA 10M, with np=16.

---

## 7. Novel Architecture Investigation: Design Brief for Claude Opus 4.7

### 7.1 Core principle: Adaptive Rank Lending (ARL)

The investigation agent should design, prototype, and evaluate a novel dispatch
architecture called **Adaptive Rank Lending (ARL)** (working title).

**Key insight**: In the current MPGC, each group has a FIXED group_size=2.
The group_size should be VARIABLE and DYNAMIC. A rank that is "idle" (its FCA
model queue is empty or all remaining models are very cheap) should be LOANED to
another group that is evaluating a very expensive model.

ARL design principles:
1. **Dynamic group formation**: Pre-allocate MPI communicators for all feasible
   group sizes {1, 2, 4, 8, 16}. At dispatch time, assign a group_size based on
   the model's estimated cost.
2. **Work-stealing across groups**: Idle ranks steal model assignments from
   overloaded groups. The stealing protocol must not break the filterRatesMPI
   intra-group broadcast invariant.
3. **No-starve guarantee**: Every rank must always have work to do. If no heavy
   model needs pattern-parallel help, the rank evaluates light FCA models
   independently.

### 7.2 Literature survey task

The investigation agent should survey the following areas and identify relevant
techniques applicable to this problem:

**A. Dynamic task scheduling in distributed ML / HPC:**
- Charm++ work-stealing and asynchronous MPI
- StarPU (heterogeneous task graph scheduling)
- OmpSs task dependency annotations
- Reference: "Adaptive Task Scheduling with Work-Stealing" (Blumofe & Leiserson 1999)

**B. Site-parallel phylogenetics:**
- ExaML (RAxML site-parallel, Kozlov et al. 2015): distributes sites across MPI ranks
  during full inference — not ModelFinder. What overheads did they measure?
  Why does ExaML require ≥100K sites per rank for efficiency?
- PhyML-MPI: topology-parallel (different NNI moves per rank), not site-parallel
- RAxML-NG (Kozlov et al. 2019): partition-parallel — how does it avoid the
  Allreduce bottleneck?

**C. ModelFinder parallel design:**
- IQ-TREE 2 paper (Minh et al. 2020 Mol Biol Evol): §ModelFinder section
  describes the FCA island dispatch. Why was site-parallelism NOT used in MF?
- The B-series prior work in this codebase (see design/modelfinder-mpi-dispatch.md):
  what design options were rejected and why?

**D. Scheduling heavy tails in heterogeneous workloads:**
- Critical-path scheduling for DAGs with variable task weights
- The "big rock" heuristic (schedule heaviest tasks first, lending idle workers)
- ListSchedule and LPT algorithms: known to give ≤ (4/3 - 1/3m)×OPT makespan
- Reference: Graham (1969), Coffman et al. scheduling theory

**E. Memory-efficient tree likelihood:**
- "Checkpointing" partial_lh to reduce memory pressure during pattern-parallel execution
- Tiled matrix × vector products with NUMA-aware allocation
- Can partial_lh be computed "on the fly" for [ptn_start, ptn_end) without storing
  the full tree's partial_lh array? (Would change memory from O(ntax×nptn) to O(nptn))

### 7.3 Specific designs to evaluate

The investigation agent should produce a written design proposal (as a new document
at `research/Modelfinder/novel-dispatch-architectures.md`) evaluating the following
three candidate architectures. For each, estimate wall time at AA 1M np=16.

#### Architecture A: Dynamic Group-Size MPGC (DGMPGC)

Extension of current P.7-MPGC with variable group sizes.

Setup phase (before model eval):
1. Estimate cost of each model using P.6's `p6_model_costs[]`.
2. Bin models into: ULTRA-HEAVY (> 5× avg), HEAVY (1.5–5×), LIGHT (< 1.5×).
3. Assign models to "slots" using a 3-tier LPT:
   - ULTRA-HEAVY models → group_size = max(available), e.g. 8 or 16
   - HEAVY models → group_size = 2 or 4
   - LIGHT models → group_size = 1 (single rank, no allreduce)
4. Communicators for each size pre-created via `MPI_Comm_split`.

Critical constraint: for the group-level filterRatesMPI (P.6/B.4-14), the reference
family (LG) must complete on all group members before broadcast. With variable group
sizes, this requires careful communicator management.

**Expected wall time improvement**: Eliminates the group_size=2 overhead for light
models (removes spurious allreduces). For the few ULTRA-HEAVY models, larger
group sizes may amortize overhead better (8× pattern split = 8× less work per rank,
making allreduce overhead < 5% of total). Estimate: 700–900 s at AA 1M np=16
(still misses 600 s target, but correct direction).

**Feasibility**: Medium. Requires pre-computing model costs (already available from
P.6's `p6_model_costs[]`) and a multi-size communicator pool. filterRatesMPI must
be adapted to the variable-size group world.

#### Architecture B: Two-Phase Critical-Path Dispatch (TCPD)

Eliminates MPGC groups entirely. Pure dynamic dispatch with a critical-path oracle.

Phase 1: **Model profile scan** (before ModelFinder)
- Run a brief "probe" evaluation of each model using a random subsample of 1,000
  patterns (< 1 s total). Record estimated per-model cost.
- Build a priority queue ordered by estimated cost (heaviest first).

Phase 2: **Critical-path-aware dispatch**
- Each rank polls the priority queue for its next model.
- For a model with estimated cost > k × (avg remaining cost), the dispatcher
  "splits" it across k ranks via pattern parallelism:
  - k = min(available_idle_ranks, 8)
  - Creates a temporary MPI communicator of size k for that model's evaluation
  - After evaluation, communicator is freed and ranks return to the pool
- For a model with cost < threshold, the dispatcher assigns it to a single rank.
- filterRatesMPI fires globally when the reference family's last model is picked up
  (not when it completes — fire-and-forget with eventual consistency broadcast).

This architecture requires a shared-memory lock-free queue (for single-node
scenarios) or an MPI-3 RMA queue for multi-node work-stealing.

**Expected wall time improvement**: Near-optimal load balance. Heavy models
auto-get k=4–8 ranks when idle ranks are available. Estimated speedup: 3–4× over
FCA np=16 = 280–374 s. Beats 600 s target.

**Feasibility**: High complexity. The probe phase adds ~1 s overhead (negligible).
The RMA work-stealing queue requires MPI-3 atomics (available in OpenMPI 4.1.7).
filterRatesMPI "eventual consistency" mode needs careful proof of correctness (the
ok_rates broadcast must still happen before any post-filter model is evaluated).

**Literature reference**: This closely resembles the "HEFT" algorithm (Heterogeneous
Earliest Finish Time, Topcuoglu et al. 2002 IEEE TPDS), which is standard in
scientific workflow engines and consistently achieves ≤ 10% above optimal makespan.

#### Architecture C: Asynchronous Pattern-Streaming MPGC (APSGC)

Addresses the root cause of Mode P slowness (if H2 is confirmed correct).

If `computePartialLikelihood` currently computes ALL patterns on both ranks:
- Fix: pass [mp_lo, mp_hi) to `computeTraversalInfo` → the tree traversal scheduler
  ONLY computes partial_lh for the rank's pattern slice.
- This requires changing the `partial_lh_computed` flag semantics: a flag that
  means "partial_lh is valid for [ptn_start, ptn_end)" not "valid for all patterns".
- Add a `partial_lh_slice_start` / `partial_lh_slice_end` per-branch metadata.

Then, at log-lh summation time, the rank only reads from [ptn_start, ptn_end)
of each partial_lh (already done by P.3 Mode P), which is consistent.

**But there is a correctness hazard**: when a model in Mode P evaluates the same
branch as a previous FCA-dispatched model, the partial_lh cache may hold stale
data for [0, ptn_start) and [ptn_end, nptn). The `partial_lh_computed` bit must
be cleared when switching between Mode P and FCA dispatch for the same branch.

**Expected wall time improvement**: If H2 is confirmed, fixing the tree traversal
to use the Mode P slice would give ~2× speedup on individual model evaluation.
Combined with current load imbalance (H1), total MF wall estimate:
- With tree traversal fix only: LG+F+I+G4 ≈ 166 s (vs 447 s) = 2.7× speedup
- But load imbalance still limits total MF wall to ~900–1200 s
- Combination of APSGC + better group assignment → estimate 500–700 s

**Feasibility**: Medium. Core kernel change, but localized. Requires extensive
correctness testing (ISO-1 through ISO-5 must all still pass).

---

## 8. Immediate Action Items for Claude Opus 4.7

**Priority 1: Confirm H2 (2 hours)**

1. Read `tree/phylokernelnew.h` lines 2720–2900 (`computeLikelihoodBranchGenericSIMD`)
   and lines 1299–1500 (`computePartialLikelihoodGenericSIMD`).
2. Trace: does `computeLikelihoodBranchGenericSIMD` call `computePartialLikelihood`
   with the SLICED `limits[]` (based on `mp_size`) or with the FULL `[0, nptn)` range?
3. If the tree traversal sees the FULL range → H2 is confirmed → Architecture C is
   the highest-priority fix.
4. Document findings in `research/Modelfinder/h2-partial-lh-slice-audit.md`.

**Priority 2: Profile allreduce count (1 hour)**

1. Add atomic counter to `modePAllreduceLh` and `modePAllreduceLhDfDdf` in
   `tree/phylotree.cpp`. Print in MF-TIME line.
2. Run AA 100K, np=2, `--mode-p-all`, to get allreduce counts for LG+F+G4 and LG+F+I+G4.
3. Estimate total allreduce overhead (count × 2 µs) and compare to observed regression.

**Priority 3: Design novel dispatch (4 hours)**

1. Survey the literature references in §7.2.
2. Design Architecture B (TCPD) in detail, including:
   - Probe phase protocol (how to run 1,000-pattern subsample)
   - RMA work-stealing queue design
   - filterRatesMPI eventual-consistency adaptation
   - MPI communicator lifecycle management
3. Write design doc at `research/Modelfinder/novel-dispatch-architectures.md`.

**Priority 4: Implement Architecture C fix if H2 confirmed (4 hours)**

1. Modify the `computeBounds` dispatch in `computeLikelihoodBranchGenericSIMD`
   to pass the Mode P slice limits to `computePartialLikelihood`.
2. Add `partial_lh_slice_start/end` to `PhyloNeighbor` or use the existing
   `partial_lh_computed` flag with a more granular bitmask.
3. Update `partial_lh_computed` invalidation to handle slice transitions.
4. Run ISO-1 through ISO-5 correctness gates on the modified binary.
5. Submit a new P.7 perf gate with group_size=2 (to measure Architecture C speedup
   in isolation before attempting Architecture B).

---

## 9. Performance Targets and Success Criteria

| Target                              | Wall (s)  | MF speedup vs FCA np=16 |
|-------------------------------------|-----------|-------------------------|
| P.7 gate (current, failed)          | > 1,800   | < 0.62×  ← FAIL         |
| Architecture C alone (H2 fix)       | ~900–1200 | ~0.94–1.25×             |
| Architecture B (TCPD)               | ~280–400  | ~2.8–4.0×  ← ≥1.87× ✅ |
| Architecture A + C combined         | ~500–700  | ~1.6–2.2×               |
| Ultimate target (10M AA)            | ≤ 6,000   | ≥ 1.87× (relative)      |

The 600 s (≥ 1.87×) target for P.7 requires Architecture B (TCPD) or a combination.
Architecture C alone is necessary but not sufficient.

---

## 10. Codebase Context for the Investigation Agent

```
Repository roots (both identical binaries, different PBS setups):
  /home/272/as1708/setonix-iq/     — primary workspace (Gadi/Setonix)
  /home/272/as1708/um09-iqtree/    — secondary workspace

Active source tree:
  /scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3/

Active build dir:
  /scratch/rc29/as1708/iqtree3-mode-p-iso/build-mode-p-iso-p3/

P.7 workdir (completed run data):
  /scratch/rc29/as1708/iqtree3-mode-p-iso/runs/p7_p3_aa1m_np16_seed1_169211688/

FCA np=16 reference workdir (job 168635616):
  /scratch/dx61/as1708/mf_iso/profiles/AA_1m_mfiso_np16_full_seed1_168635616/

Alignments:
  AA 1M:  /scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
  AA 100K: /scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
  AA 10M:  (location TBD — check with sa0557 or generate from the 1M with x10 duplication)
  DNA 10M: (location TBD)

CI scripts:
  setonix-iq/gadi-ci/mode-p-iso/run_iso*.sh   — ISO correctness gates 0–5
  setonix-iq/gadi-ci/mode-p-iso/run_p7_aa1m_np16_p3.sh  — P.7 perf gate (failed)

Research docs:
  setonix-iq/research/Modelfinder/mode-p-implementation-status.md  — phase tracker
  setonix-iq/research/Modelfinder/mode-p-design.md                 — design contract
  setonix-iq/design/modelfinder-mpi-dispatch.md                    — earlier dispatch design

Build script:
  setonix-iq/gadi-ci/mode-p-iso/build_mode_p_iso_p3.sh

Binary checksums:
  P.7 binary: cc3d403f9aac4eb44f3ef022efcce8d8 (iqtree3-mpi-mode-p-iso-p3)
  FCA ref:    a78ffa2942d6b073490d503416ae554c (iqtree3-mpi)
```

---

## 11. Summary of Known Constraints

1. **MPI_THREAD_FUNNELED**: Only the main thread may call MPI. All allreduces must
   happen outside OMP parallel regions or use a dedicated MPI thread.
   `--atmd-k-outer 1` enforces this for the ATMD phase; Mode P must do the same.

2. **`--mca coll ^ucc`**: Required in all mpirun invocations due to OpenMPI 4.1.7
   UCC bug (concurrent Allreduces on sub-communicators created by MPI_Comm_split
   cause "Message truncated"). Any architecture using MPI_Comm_split must include
   this flag.

3. **`numactl --localalloc`**: Each rank should run with `numactl --localalloc` to
   avoid NUMA cross-domain memory access. Any novel architecture that redistributes
   pattern ownership between ranks must ensure the new owner's NUMA domain has the
   relevant portion of the alignment and partial_lh arrays.

4. **`np` must divide `group_size` evenly**: For DGMPGC/TCPD, ensure communicator
   splits never leave a rank without a group.

5. **filterRatesMPI correctness invariant**: No rank may evaluate a post-prune model
   (Phase 2 model) until its group's filterRatesMPI Bcast has completed.
   Any architecture that modifies group membership mid-run must ensure this invariant
   is preserved.

6. **Sapphire Rapids nodes**: 1 rank per node, 103 OMP threads, 4 NUMA domains
   (26 cores each), ~500 GB aggregate DRAM bandwidth. The per-rank memory limit
   used in PBS scripts is `mem=500GB` per node.
