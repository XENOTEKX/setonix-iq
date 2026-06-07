# Event-Driven Moldable Dispatch for IQ-TREE ModelFinder

**Date:** 2026-05-27  
**Status:** Design reset after ATMD-AID v1/v2 P.7 failures. Supersedes the static ATMD-AID Phase-0/Phase-1 split as the next architecture direction.  
**Scope:** IQ-TREE ModelFinder inside one alignment; AA/DNA 100K-10M patterns; np=2-64; OpenMPI 4.1.7 + ICX + Gadi SPR.

**Implementation start (2026-05-27):** EDM v0 is now present in the active ATMD/Mode-P source copy behind `--mf-edm`. It adds scheduler state/CLI flags, creates a sentinel epoch plus an LPT-packed moldable tail epoch, and executes those epochs through the existing pre-created Mode-P lattice and canonical checkpoint/warm-start path. Binary md5: `4810a8ac73e3b92b1f93b3f03ec04d57`. This is not the final event loop yet: explicit filter firing after the sentinel epoch and telemetry-driven replan remain the next implementation steps.

**ISO-4 EDM PASS (2026-05-27, job 169350768):** AA 100K np=4 correctness gate confirmed. lnL=−7,541,976.864 (Δ=0.003), model=LG+G4, EDM-DIAG/EDM-EPOCH/wave markers all present, [Mode P] on all 4 ranks. Wall 14m52s.

**P.7 EDM FAIL (2026-05-27, job 169352556, PBS-killed):** AA 1M np=16, sentinel gs=16 + tail gs=4 (4 cohorts). Scheduler plan emitted correctly but MF did not complete. Two root causes measured: (1) Mode P gs=4 at AA 1M averages 150–210s/model vs FCA ~80s/model — allreduce overhead exceeds the gain from 4-way pattern split; (2) the tail LPT queue contains the expensive `+R4`/`+I+R4` models (150–241s each) that the sentinel-boundary `filterRatesMPI` event (`note=explicit_filter_event_pending`) would prune before they run. Without the filter, the 4-cohort tail needs ~8,000–10,000s; with the filter, pruned tail ≈ 50–70 models × ~40s / 4 cohorts ≈ 500–550s. **Immediate next step: implement explicit filter event at sentinel epoch boundary.**

---

## 0. TL;DR

ATMD-AID failed because it made a static binary decision before runtime:

- **Phase 0:** models below a threshold run through FCA as "light".
- **Phase 1:** models above the threshold run later through Mode P waves.

That split is not responsive enough. At AA 1M, `+G4` and `+F+G4` are not truly light: rank 0 measured `LG+F+G4=209s`, `PMB+G4=174s`, and `MTART+F+G4=277s` before Phase 1 started. With 56 such models, the ideal Phase 0 lower bound is about **945s**, already beyond the **600s** P.7 gate. Moving them into Phase 1 is also fatal because the gs=2 tail becomes too large.

The reset architecture is **Event-Driven Moldable Dispatch (EDM)**:

1. ModelFinder is represented as a **task DAG**, not as a light/heavy phase split.
2. Every model task can run at `group_size` in `{1, 2, 4, 8, np}`; `gs=1` is FCA-like, `gs>1` is Mode P.
3. Scheduling happens in **short epochs**. At each epoch boundary, the scheduler gathers completed scores, fires `filterRates`/`filterSubst` events, updates cost estimates from actual timings, prunes queued work, and replans.
4. `filterRatesMPI` becomes an explicit **scheduler event**, not a barrier hidden inside the per-rank model loop.
5. Large substitution families are no longer atomic. They are soft-affinity groups of model tasks; rate-chain and filter dependencies preserve correctness without pinning a whole family to one rank or one cohort.

The short-term P.7-MPGC `--mode-p --mode-p-group-size 4` run may still be useful as a triage/performance proof, but it is not the final architecture. Static all-model `gs=4` can pass one AA 1M gate while still failing to adapt to small datasets, rate-filter timing, and family heavy tails.

---

## 1. Why the Previous Architectures Failed

### 1.1 FCA: good island parallelism, poor responsiveness

FCA fixed the worst early dispatch bugs by assigning substitution families with greedy LPT and preserving ModelFinder rate pruning. It still has two structural limits:

- **Family granularity:** a large family such as LG or a family with expensive `+G4`/`+I+G4` variants can dominate a rank.
- **Filter barrier:** `filterRatesMPI` requires all ranks in its communicator to finish their reference families before global `ok_rates` can be broadcast. At AA 100K np=4, rank 0 evaluated fewer models than np=2 but took longer because the excess wall was barrier wait, not model computation.

FCA is the right fallback for genuinely cheap tasks. It is not a sufficient global scheduler.

### 1.2 ATMD Mode F: wrong parallel axis for AA on SPR

Mode F runs multiple IQTree model evaluations concurrently on one node. AA likelihood evaluation is already DRAM-bandwidth-bound at one model x 103 threads. Running K models on the same node divides the same bandwidth K ways.

Measured consequences:

| Case | Result | Reason |
|------|--------|--------|
| AA 100K, K=8 | MF=423s vs FCA np=1 MF=259s | 8 concurrent 46GB working sets thrash DRAM/LLC |
| AA 1M, K=1 | MF=2114s vs FCA np=16 MF=1122s | Mode F inactive but still carries setup/NUMA overhead |

Mode F can remain in the tree for compute-bound codon workloads, but it should not be extended for AA P.7.

### 1.3 MPGC group_size=2: correct idea, static ownership

MPGC introduced the right primitive: a group of MPI ranks cooperates on one model via Mode P. The failed P.7 run exposed two problems:

1. **Static family-to-group binding:** some groups completed several Mode P models while other groups completed none in 30 minutes.
2. **Traversal was not sliced at first:** both ranks computed all patterns, then summed only a slice. Architecture C fixed this, but static group ownership remained.

MPGC provided the communicator and inheritance machinery we should keep. It did not provide a responsive scheduler.

### 1.4 ATMD-AID v1/v2: static threshold and static phase boundary

ATMD-AID improved MPGC by creating a communicator lattice and routing predicted-heavy models through Mode P waves. It failed for a deeper reason: **the threshold is doing too much work**.

| Version | Failure | Lesson |
|---------|---------|--------|
| AID v1 | `+I+G4` had cost multiplier 5 < threshold 9, so it stayed light and ran >420s in Phase 0 | static predictor missed a known expensive class |
| AID v2 | `+G4`/`+F+G4` had multipliers 11/13.2 < threshold 18.15, so 56 medium-expensive models stayed light | even a better predictor cannot make a binary split adapt |

At AA 1M, the "light" tail is itself expensive. A scheduler must be able to assign medium models to `gs=4` or `gs=2` while the run is in progress, not after the light queue is drained.

---

## 2. Hard Constraints from the Current IQ-TREE Implementation

These are non-negotiable for a viable design:

1. **One IQTree per MPI rank at a time.** Naive OMP-across-models in MPI builds causes checkpoint map races and OOM. Keep `atmd_K_outer=1` whenever Mode P is active.
2. **Mode P collectives require cohort lockstep.** Every rank in a Mode P cohort must evaluate the same model and call the same Allreduces in the same order.
3. **No MPI from OMP worker threads.** Gadi/OpenMPI production is effectively `MPI_THREAD_FUNNELED` for this path. The scheduler must run collectives from the main thread.
4. **Disable OpenMPI UCC for concurrent sub-communicators.** Run scripts must keep `--mca coll ^ucc`; UCC confused simultaneous 1-double and 3-double Allreduces on different sub-communicators.
5. **World collectives only at deterministic epoch boundaries.** Per-model collectives must be cohort-scoped; rate-filter or gather collectives that use world scope must happen when every world rank is known to participate.
6. **MPGC inheritance must remain.** `CandidateModel::evaluate()` creates a fresh per-model IQTree; it must inherit `mp_allreduce_comm`, `mp_group_rank`, and `mp_group_size` from the scheduling context before `initializePtnPartition()`.
7. **Architecture C must remain.** Tree traversal and summation must both honor `[ptn_start, ptn_end)` under Mode P, or pattern splitting does not reduce the real work.
8. **Canonical checkpoint/warm-start state is required before collective waves.** Rank-local Phase 0 checkpoint state caused divergent BFGS paths; scheduler epochs must start from a canonical state when ranks cooperate.

---

## 3. New Direction: Event-Driven Moldable Dispatch (EDM)

### 3.1 Core idea

Replace this:

```text
classify once -> run all light work -> barrier -> run heavy waves
```

with this:

```text
build task DAG -> schedule a short epoch -> gather results -> fire filters -> update costs -> schedule next epoch
```

A task is not a family. A task is one candidate model evaluation with metadata:

```cpp
struct MfTask {
    int model_idx;
    string subst_name;
    string rate_name;
    double predicted_cost;
    double observed_cost;
    int min_group_size;
    int max_group_size;
    vector<int> dependencies;
    bool ready;
    bool pruned;
    bool done;
    bool cannot_ignore;
};
```

The scheduler is free to run `LG+F+G4` at `gs=4`, `PMB+G4` at `gs=2`, and a cheap bare model at `gs=1` in the same overall run. There is no global heavy/light label.

### 3.2 ModelFinder as a task DAG

Dependencies encode ModelFinder semantics:

| Dependency | Purpose |
|------------|---------|
| Reference sentinel -> global rate filter | Evaluate a small canonical reference tranche early enough to derive `ok_rates` and prune rate variants before they enter the queue. |
| `R_k` -> `R_{k+1}` within a family | Preserve lower-k-before-higher-k pruning and avoid evaluating high FreeRate categories speculatively. |
| Canonical checkpoint/warm-start epoch state -> cooperative task | Ensure every rank in a Mode P cohort starts from the same BFGS/checkpoint state. |
| Family completion -> `filterSubst` event | Preserve substitution-model pruning without binding the whole family to one rank. |

This turns rate filtering into explicit scheduler events. It removes the hidden barrier where `filterRatesMPI` fires inside the per-rank model loop.

### 3.3 Reference sentinel epoch

The first epoch should not try to drain a whole family. It should evaluate only the models needed to obtain a sharp global `ok_rates` decision.

Candidate sentinel tranche for AA:

```text
canonical family: LG / LG+F chosen by existing generate-order or by calibration
sentinel models: bare, +I, +G4, +I+G4, and any required cannot-ignore variants
```

Run this tranche with a moldable plan, typically `gs=np` for the expensive sentinel and `gs=4` or `gs=8` for medium sentinel tasks. When the tranche completes:

1. Build `ok_rates` exactly as current `filterRates` does.
2. Broadcast the result once at a world epoch boundary.
3. Mark not-yet-started tasks with disallowed rates as pruned.
4. Rebuild the ready queue.

At AA 1M, this prevents the run from spending 800-900s on a supposed light phase before rate pruning is known.

### 3.4 Moldable group-size choice for every ready task

For each ready task, choose `group_size` from `{1, 2, 4, 8, np}` by minimizing predicted finish time, not by crossing a fixed threshold.

A practical scoring function:

```text
finish(task, g, cohort) = cohort_load[cohort]
                       + predicted_cost(task) / speedup(task, g)
                       + collective_overhead(task, g)
                       + cache_penalty(task, g)
```

Where:

- `speedup(task, g)` is learned from MF-TIME telemetry by `(nstates, npat, rate_class, freq_class, g)`.
- `collective_overhead` is from Mode P Allreduce count and communicator size.
- `cache_penalty` prevents routing tiny models through large Mode P groups.

For AA 1M P.7, this should naturally choose `gs=4` for the broad `+G4`/`+F+G4` medium tail rather than leaving it in FCA or serializing it in a gs=2 tail.

### 3.5 Epoch planning instead of wave planning

An epoch is a short scheduling window, not a phase. Example target: 30-90 seconds predicted wall or 1-4 tasks per cohort.

Each epoch has:

```cpp
struct MfEpochAssignment {
    int epoch_id;
    int group_size;
    int cohort_id;
    MPI_Comm comm;
    vector<int> model_queue;  // sequential within this cohort for this epoch
};
```

Rules:

1. Every world rank belongs to exactly one cohort for the epoch.
2. A cohort may run several short tasks sequentially if they fit within the epoch budget.
3. Cohorts are independent during the epoch.
4. At epoch end, `MPI_Barrier(MPI_COMM_WORLD)` synchronizes before any world-scope event.
5. Completed scores are gathered; queued tasks may be pruned or reweighted before the next epoch.

This keeps the proven AID property that all communicator switching happens at safe points, while removing AID's fatal property that all light work must finish before heavy work can start.

### 3.6 Legal communicator tilings

Do not create communicators mid-run. Pre-create a small set of legal tilings at `evaluateAll()` entry.

For np=16, useful tilings include:

| Tiling | Use case |
|--------|----------|
| `16` | one huge sentinel/heavy task |
| `8+8` | two dominant heavy tasks |
| `4+4+4+4` | broad AA 1M `+G4` medium tail; likely P.7 workhorse |
| `4+4+2+2+2+2` | mixed medium and light queue |
| `2x8` | many moderate tasks after pruning |
| `1x16` | cheap FCA-like tail, no Allreduce |

This is more expressive than AID's sequential waves by group size, but still avoids MPI process malleability or dynamic `MPI_Comm_split`.

### 3.7 Cost model: priors plus online correction

The current empirical AID v2 multipliers are useful priors, not a final classifier.

Initial prior dimensions:

```text
nstates^2, npat, ntaxa log factor,
rate_class: bare, +I, +G4, +I+G4, +Rk, +I+Rk,
freq_class: fixed/empirical/bare +F,
family adjustment: learned from observed family residuals,
group_size speedup curve: S_g(rate_class, npat)
```

Online update at each epoch boundary:

```text
observed_multiplier = observed_wall * group_size / base_predicted_cost
update EMA for (rate_class, freq_class, family, group_size)
```

This means a PMB or MTART `+G4` taking 174-277s at AA 1M changes the next epoch's plan immediately. It is not trapped in a phase chosen before the run began.

---

## 4. Why EDM Addresses the Persistent Issues

| Persistent issue | Why ATMD-AID failed | EDM response |
|------------------|---------------------|--------------|
| Smaller models | Mode P on all small models serializes the queue; Mode F saturates DRAM | `gs=1` or small `gs=2` chosen per task; no all-model Mode P requirement |
| Medium `+G4` tail | Below threshold, so stuck in Phase 0 FCA | Medium tasks can get `gs=4` immediately if predicted wall justifies it |
| Rate filters | Hidden `filterRatesMPI` barrier inside model loop | Rate filter is explicit sentinel event at epoch boundary |
| Large families | Family assigned as an atomic ownership unit | Family is soft affinity; individual model tasks can spread across cohorts |
| Cost prediction errors | Wrong threshold causes wrong phase forever | Epoch telemetry updates costs before the next scheduling window |
| MPGC imbalance | Static family-to-group binding | Cohorts are repacked every epoch based on remaining work |
| UCC/collective mismatches | Concurrent comms plus wrong collective scope | Pre-created comms, `--mca coll ^ucc`, group collectives inside epoch, world collectives only at boundaries |

---

## 5. Expected P.7 Behavior

The P.7 target is AA 1M np=16 MF wall <= 600s.

A static `gs=4` all-model MPGC run estimates:

```text
FCA np=4 MF wall = 1974s
all models at gs=4 with 4 cohorts ~= 1974 / 4 = 494s
```

EDM should match or beat that lower-risk static configuration while avoiding its generality problems:

1. Sentinel epoch obtains `ok_rates` early.
2. Broad `+G4`/`+F+G4` tail is scheduled mostly as `4+4+4+4` tilings.
3. Cheap post-filter tail falls back to `gs=1`/`gs=2` to avoid Allreduce tax.
4. Epoch replanning absorbs family residuals such as MTART+F+G4 taking 277s instead of the LG-based 209s prediction.

### 5.1 2026-05-27 FCA 1M cancellation sweep (new evidence)

After the non-running subset was harvested, the remaining FCA jobs were canceled with `qdel` to avoid spending further allocation on runs that were not converging to usable parity rows.

| Jobs | Scenario | End state | Last usable MF evidence |
|------|----------|-----------|--------------------------|
| 169332780, 169332785 | AA 1M, np=2 (`-m MF`, `-m MFP`) | SIGTERM at ~03:07 wall | rank 0 reached `model=1168 RTREV+F+G4` at t~9,8xxs; `filterRatesMPI` fired at model 16 and pruned 567 local models; no `ModelFinder took`/best-model block. |
| 169332781, 169332786 | AA 1M, np=4 (`-m MF`, `-m MFP`) | SIGTERM at ~03:25 wall | rank 0 reached `model=1080 HIVW+F+G4` at t~8,9xxs; `filterRatesMPI` fired at model 16 and pruned 273 local models; no final MF/MFP output block. |
| 169332788 | AA 1M, np=16 (`-m MFP`) | SIGTERM at 03:25:56 wall | `filterRatesMPI` fired much later (model 38, `LG+F`) after heavy `+I+R5` tail; rank 0 only reached `model=728 MTART+F+G4`; no tree-search start. |
| 169332790, 169332797 | DNA 1M, np=2 (`-m MF`, `-m MFP`) | SIGTERM at ~03:07 wall | rank 0 reached `model=905 SYM+I+G4` at t~2,49xs; `filterRatesMPI` fired at model 36 with two accepted rates and 407 local prunes; no completion block. |
| 169332799 | DNA 1M, np=4 (`-m MFP`) | SIGTERM at 03:07:38 wall | rank 0 at `model=390 TPM2+I+R5` (t=10,950s), `ref_remaining=10`; `filterRatesMPI` never fired before cancel. |

Implication: the remaining failure mode is not just static `+G4` load. Medium/high FreeRate tails can consume walltime before globally useful pruning decisions propagate. This directly supports EDM's event-first design where filter sentinels and group-size changes are scheduler-level actions rather than hidden loop side effects.

Initial target for an offline replay simulator: **predicted P.7 <= 550s** using observed MF-TIME costs from jobs 168635615, 168635616, 169227541, and 169343365.

---

## 6. Implementation Plan

### D.0: Offline schedule replay before C++ changes

Build a simulator that consumes existing logs:

```text
tools/mode_p_iso/parse_mf_time.py output
FCA np=4/np=16 timing tables
AID v1/v2 MF-TIME traces
```

Simulator outputs:

- chosen epoch tilings,
- per-cohort load,
- predicted wall,
- rate-filter event time,
- tasks pruned before start,
- sensitivity to `speedup(gs)` assumptions.

Gate: replay predicts P.7 <= 550s with conservative Mode P speedup curves.

### D.1: Scheduler state in C++

Add new state to `CandidateModelSet`:

```cpp
vector<MfTask> mf_tasks;
vector<MfEpochAssignment> mf_epoch;
MfSchedulerTelemetry mf_telemetry;
MfFilterState mf_filter_state;
```

New methods:

```cpp
void mfBuildTaskGraph(PhyloTree *in_tree, Params &params);
void mfBuildCommTilings(int np, int world_rank);
void mfPlanNextEpoch();
void mfExecuteEpoch(Params&, PhyloTree*, ModelCheckpoint&, ModelsBlock*, int, int);
void mfApplyCompletedScoresAndFilters(ModelCheckpoint&);
void mfUpdateCostTelemetry();
```

Reuse from AID/MPGC:

- `setModePGroupComm`,
- `initializePtnPartition`,
- MPGC inheritance into per-model IQTree,
- Architecture C kernel slicing,
- canonical checkpoint/warm-start broadcast,
- `MF-TIME` instrumentation.

### D.2: Explicit rate-filter event

Refactor the current `filterRatesMPI()` call path so the scheduler can invoke it at an epoch boundary using completed sentinel scores.

Required behavior:

1. `filterRates` logic computes `ok_rates` from canonical sentinel/reference scores.
2. World or scheduler-scope broadcast happens only at a boundary.
3. Not-started tasks are pruned in the task DAG.
4. In-progress epoch tasks are allowed to finish; no mid-model cancellation in the first implementation.

This avoids collective deadlocks and preserves ModelFinder correctness.

### D.3: Epoch execution path

Add a new flag, default off:

```text
--mf-edm                 Enable Event-Driven Moldable Dispatch
--mf-edm-epoch-sec N     Predicted epoch budget, default 60
--mf-edm-gs-list 1,2,4,8,16
--mf-edm-dry-run         Print schedule without evaluating models
```

Execution loop:

```cpp
while (scheduler.hasReadyOrRunningWork()) {
    scheduler.planNextEpoch();
    scheduler.executeEpoch(...);
    MPI_Barrier(MPI_COMM_WORLD);
    scheduler.gatherCompletedScores();
    scheduler.fireFilterEvents();
    scheduler.updateTelemetry();
}
```

### D.4: Correctness gates

| Gate | Dataset | Pass criterion |
|------|---------|----------------|
| EDM-Dry | AA 100K np=4 | deterministic identical schedule on all ranks; no evaluation |
| EDM-ISO4 | AA 100K np=4 | lnL/model parity with FCA np=1; no deadlock; filters fire once |
| EDM-ISO5 | AA 100K np=4 auto | same as ISO4 plus mixed `gs` schedule exercised |
| EDM-P7-sim | AA 1M np=16 replay | predicted MF <= 550s |
| EDM-P7 | AA 1M np=16 | measured MF <= 600s |

---

## 7. Near-Term Triage vs Final Architecture

P.7-MPGC with `--mode-p --mode-p-group-size 4` is a reasonable **triage experiment** because it removes the ATMD-AID phase split and may hit the gate quickly.

But it should not be mistaken for the new architecture:

| Static gs=4 MPGC | EDM |
|------------------|-----|
| one group size for every model | group size selected per task/epoch |
| can overpay Allreduce tax on small models | routes cheap tasks to `gs=1`/`gs=2` |
| no online cost correction | updates telemetry each epoch |
| no explicit rate-filter event | rate filters are scheduler events |
| fixed cohort layout | cohorts repacked each epoch |

If a static gs=4 run passes P.7, it validates the Mode P performance envelope. It does not solve dynamic scheduling.

---

## 8. Decision

Stop tuning ATMD-AID thresholds. The persistent failure is architectural, not parametric.

Keep:

- Mode P kernel and Architecture C,
- MPGC communicator inheritance,
- canonical checkpoint/warm-start broadcast,
- AID communicator-lattice code as a starting point,
- `MF-TIME` telemetry.

Replace:

- `MF_AID_HEAVY` static tagging,
- Phase 0 light drain,
- Phase 1 heavy waves,
- threshold-based architecture claims.

Next concrete action: implement the offline EDM schedule replay and use it to choose the first C++ epoch scheduler scope. No more PBS perf gates should be launched for ATMD-AID until replay shows the new dispatcher can beat the P.7 lower bound.
