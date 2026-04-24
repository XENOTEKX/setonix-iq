# IQ-TREE GPU Offload — Development Progress

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
   > **Compute nodes have no outbound internet** — clone the source AND its
   > submodules (`cmaple`, `lsd2`) on a login node first, then submit the
   > build job:
   > ```bash
   > git clone https://github.com/iqtree/iqtree3.git \
   >     /scratch/rc29/<user>/iqtree3/src/iqtree3
   > cd /scratch/rc29/<user>/iqtree3/src/iqtree3
   > git submodule update --init --recursive   # fetches cmaple + lsd2
   > qsub gadi-ci/bootstrap_iqtree.sh
   > ```
   > The bootstrap script pre-flight checks for `cmaple/CMakeLists.txt` and
   > `lsd2/CMakeLists.txt` and errors out with a clear message if either is
   > missing, so you never waste SUs on a build that cannot configure.
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
