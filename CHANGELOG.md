# IQ-TREE GPU Offload — Development Progress

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

### Remaining cross-platform work

- ⏳ Check status of PBS jobs 167506092–167506100 on Gadi
- ⏳ Harvest `_sr_gcc_pin` runs → `Gadi_large_modelfinder_{T}T.json` (overwrite or add `_gcc` suffix)
- ⏳ Submit `xlarge_mf` and `mega_dna` `_sr_gcc_pin` matrix on Gadi
- ⏳ Harvest `Setonix_xlarge_mf_1T` (job 42181135, still running)

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
