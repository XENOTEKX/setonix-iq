# MF-Isolation harness — ModelFinder MPI dispatch debugging (rc29)

## Why this exists

The FCA Phase 0 dispatch is correct at np=1 (1,277 s, matches Fix H 1,289 s
to within run-to-run variance) but regressed at np=2 (2,865 s vs Fix H 475 s)
and np=4 (3,502 s vs Fix H 2,335 s). Five rounds of patching on the
production tree (`/scratch/um09/as1708/iqtree3-mf2/`) ended with job
**168486582 hung — SIGTERM after 1h19m, zero stdout** because the experimental
Phase 0.7 (MPI_Isend) + HH-NUMA (nested K_outer × M_inner) work was bundled
together. Each new change was tested at np=4 before np=2 validated, hiding
which addition actually broke things.

This harness:

1. Lives in an **isolated** source tree on rc29
   (`/scratch/rc29/as1708/iqtree3-mf-iso/`) so the production binary stays
   stable while we iterate.
2. Includes **only** the two proven-correct structural changes —
   Phase 0.5 (filterRatesMPI MPI_Bcast) and Phase 0.6 (getNextModel
   ref-family priority) — on top of Phase 0 FCA at commit `ffb79a14`.
3. **Skips** the experimental Phase 0.7 (MPI_Isend), HH-NUMA (nested OMP),
   and MPI_Init_thread upgrade. Those land in a later patch, one at a
   time, only after the np=2 validation passes.
4. Adds per-model `MF-TIME` markers so we can see what each rank actually
   does — the missing instrumentation that made earlier debugging slow.
5. Uses `-m TESTONLY` to skip the tree-search tail (~700 s on AA 100K),
   giving ~3× faster iteration cycles for ModelFinder-only development.
6. Captures **per-rank stdout** via `mpirun --output-filename` — without
   this, ranks 1+ logs were silently dropped on cross-node MPI runs
   (which is exactly what hid the rank-1 filter ineffectiveness in
   job 168475747 until the fix-by-inspection workaround was applied).

## Layout

```
~/setonix-iq/gadi-ci/mf-iso/
├── README.md                              # this file
├── build_mf_iso.sh                        # PBS: build the isolated binary
├── run_mf_iso_aa_100k_1node.sh            # PBS: 1-node MF-only run
├── run_mf_iso_aa_100k_2node.sh            # PBS: 2-node MF-only run, per-rank logs
├── submit_mf_iso.sh                       # qsub driver with afterok deps
└── tools/
    └── parse_mf_time.py                   # parse MF-TIME / MF-MPI-DIAG offline

/scratch/rc29/as1708/iqtree3-mf-iso/
├── src/iqtree3/                           # full source mirror @ branch mf-iso-phase0.5-0.6
└── build-mpi-iso/                         # cmake build (created by build_mf_iso.sh)
    └── iqtree3-mpi                        # the binary
```

## Workflow

```bash
cd ~/setonix-iq/gadi-ci/mf-iso

# Stage 1+2+3 chained with afterok dependencies:
./submit_mf_iso.sh all

# Or one stage at a time:
./submit_mf_iso.sh build         # ~30 min PBS build
./submit_mf_iso.sh 1node         # ~22 min PBS run (after build done)
./submit_mf_iso.sh 2node         # ~45 min PBS run (after 1node done)

# Inspect a finished run:
WORK=/scratch/rc29/as1708/mf_iso/profiles/AA_100k_mfiso_np2_seed1_<jobid>
./tools/parse_mf_time.py "${WORK}"
```

## Acceptance gates

**1-node** (correctness only; FCA is a no-op at np=1):
- `lnL` = -7,541,976.860 ± 0.01
- best model = `LG+G4`
- MF wall in 1,200–1,400 s band
- `filterRatesMPI_enabled=0` (no broadcast at np=1 — correct)

**2-node** (Phase 0.5/0.6 first real test):
- `lnL`, best model unchanged
- MF wall **< 600 s** (Fix H baseline 475 s; FCA Phase 0 regressed to 2,865 s; we expect close to Fix H)
- `filterRatesMPI fired at model=…` line present on rank 0 in `mf_diag.log`
- broadcast-arrival spread between ranks **< 60 s**
  (Phase 0.5 alone produced ~370 s spread; Phase 0.6 should compress it to <30 s)
- rank 1's post-broadcast model count ≤ rank 0's (both should drop to surviving `+G4` variants)

**4-node** (only after the above pass):
- submit `~/setonix-iq/gadi-ci/cpu-bench/run_cpu_bench_aa_100k_mf2_4node.sh`
  **manually**. The expectation revisits: Phase 0.5/0.6 should land in the
  300–500 s band (vs Fix H 2,335 s, FCA Phase 0 3,502 s).
- If 4-node MF wall > 600 s, do NOT chase it with further patches until
  the 2-node parse_mf_time.py output is re-examined — the bug is likely
  visible there first.

## What's deferred (and why)

- **Phase 0.7 (MPI_Isend push instead of Bcast)** — needs `MPI_Init_thread`
  (THREAD_SERIALIZED) which may be the actual culprit behind 168486582's
  hang. Defer until 2-node Phase 0.5+0.6 is validated, then add Isend
  alone (no HH-NUMA), and test 2-node before 4-node.
- **HH-NUMA Phase 2 (nested K_outer × M_inner)** — `proc_bind` + per-team
  `iqtree->setNumThreads(M_inner)` propagation through `evaluate()` needs
  empirical verification. Land separately after Phase 0.7 confirmed.
- **THP `madvise(MADV_HUGEPAGE)` on `central_partial_lh`** — orthogonal
  kernel speedup. Independent patch line `0004-thp-partial-lh-madvise.patch`.

See `~/setonix-iq/research/updated-modelfinder-dispatch.md` for the full
design (§19 Phase 0.5, §20 Phase 0.6, §21 Phase 0.7, §§13-15 HH-NUMA).
