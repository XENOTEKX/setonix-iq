# gadi-ci/ — NCI Gadi pipeline scripts

Gadi/PBS/Intel counterpart to `setonix-ci/`. These scripts live on
`/scratch/$PROJECT/$USER/iqtree3/gadi-ci/` on Gadi and produce the same
JSON artefacts the dashboard consumes.

## Target system — Intel Sapphire Rapids (`normalsr`)

| Component | Spec |
|-----------|------|
| Machine   | Gadi (NCI, Canberra) |
| CPU       | 2× Intel Xeon Platinum **8470Q Sapphire Rapids**, 52 cores each → **104 cores/node**, 2.1 GHz base / 3.8 GHz turbo |
| NUMA      | 2 sockets × 4 NUMA domains = 8 NUMA nodes, **13 cores / NUMA**, 64 GB / NUMA |
| Memory    | **512 GiB / node** (request `mem=500GB` to leave scheduler headroom) |
| Local SSD | 400 GiB `jobfs` |
| Scheduler | PBS Professional 2024.1 (`qsub`, `qstat`, `qdel`, `nqstat`) |
| Queue     | **`normalsr`** (CPU, 2 SU/core-h = 208 SU/node-h), `expresssr` (6 SU/core-h), `gpuvolta` (V100) |
| Profiler  | Intel VTune 2024.2 (`module load intel-vtune/2024.2.0`) + Linux `perf` |
| Compiler  | `intel-compiler-llvm/2024.2.0` → `icx -xSAPPHIRERAPIDS` |
| Storage   | `/home` (10 GB quota), `/scratch/$PROJECT` (1 TiB on `rc29`), `/g/data/$PROJECT` |

## Scripts

| File | Purpose |
|------|---------|
| `bootstrap_iqtree.sh`        | One-shot PBS job: clone `iqtree/iqtree3`, build with `-xSAPPHIRERAPIDS` + frame pointers into `$PROJECT_DIR/build-profiling/iqtree3`. |
| `generate_datasets.sh`       | Deterministic AliSim simulation producing `large_modelfinder.fa`, `xlarge_mf.fa`, `mega_dna.fa` into `$PROJECT_DIR/benchmarks/`. |
| `run_pipeline.sh`            | Small deterministic CI pipeline — login-node friendly. Emits `logs/runs/<YYYY-MM-DD_HHMMSS>.json`. |
| `run_profiling.sh`           | Quick single-run perf stat wrapper (for interactive `qsub -I` sessions). |
| `run_mega_profile.sh`        | Full deep-profile PBS job — perf stat + VTune hotspots + perf record + sampler. |
| `submit_mega_batch.sh`       | Fan out `run_mega_profile.sh` across thread counts via `qsub`. |
| `submit_benchmark_matrix.sh` | Reproduce the Setonix benchmark corpus: datasets × thread sweep → one `qsub` per point, each emitting a schema-conforming JSON. |

## Key differences vs `setonix-ci/`

- `#SBATCH` directives → `#PBS -N / -P / -q / -l`.
- SLURM env vars (`SLURM_JOB_ID`, etc.) → PBS env vars (`PBS_JOBID`,
  `PBS_JOBNAME`, `PBS_QUEUE`, `PBS_NCPUS`, `PBS_NODEFILE`, `PBS_O_HOST`,
  `PBS_O_WORKDIR`).
- AMD Zen 3 raw events (`ex_ret_*`, `ls_l1_d_tlb_miss.*`,
  `bp_l1_tlb_miss_l2_tlb_*`, `ls_dispatch.*`, `ls_tablewalker.*`) replaced
  with Intel Sapphire Rapids Top-down TMA slot events
  (`topdown-total-slots`, `topdown-slots-issued`, `topdown-slots-retired`,
  `topdown-fetch-bubbles`, `topdown-recovery-bubbles`) plus `LLC-loads` /
  `LLC-load-misses` (Intel publishes LLC alias events, AMD Setonix did not).
- New **VTune hotspots pass** (`vtune -collect hotspots`) with bounded
  wall-time — summary + CSV are parsed into `profile.vtune.*` in the
  emitted `profile_meta.json`.
- PBS job-id strips the `.gadi-pbs` suffix: `RUN_ID="${PBS_JOBID%%.*}"`.
- Queue default: **`normalsr`** (Sapphire Rapids 104c/node). Thread sweep
  uses `1 4 13 26 52 104` (powers-of-two plus NUMA-aligned 13 and 26)
  instead of Setonix's `16/32/64/128`. 13 = one NUMA domain, 52 = one
  socket, 104 = full node.

All artefact file names (`env.json`, `samples.jsonl`, `perf_stat.txt`,
`hotspots.txt`, `perf_folded.txt`, `profile_meta.json`) are identical to
the Setonix scripts so `tools/harvest_scratch.py` and the dashboard
front-end consume both transparently.

---

## First-time setup on Gadi (run once on a login node)

> **Why login node?** Gadi compute nodes have **no outbound internet**.
> All source code and build dependencies must be fetched on a login node
> before submitting any PBS jobs.

```bash
# 0 — set your project and scratch root
PROJECT=rc29
USER=$(whoami)
SCRATCH=/scratch/${PROJECT}/${USER}/iqtree3
mkdir -p ${SCRATCH}/{src,deps,benchmarks,build,build-profiling,logs,gadi-ci}

# 1 — clone IQ-TREE 3 (default branch: master)
git clone https://github.com/iqtree/iqtree3.git ${SCRATCH}/src/iqtree3

# 2 — fetch git submodules (cmaple + lsd2 — required by CMakeLists)
cd ${SCRATCH}/src/iqtree3
git submodule update --init --recursive

# 3 — pre-download GoogleTest
#     cmaple's CMakeLists calls FetchContent(googletest URL github...) at
#     configure time, which fails on compute nodes. Download manually:
cd ${SCRATCH}/deps
curl -L -o googletest.zip \
  https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
unzip -q googletest.zip
mv googletest-*/ googletest

# 4 — sync the gadi-ci scripts from the repo
rsync -a ~/setonix-iq/gadi-ci/ ${SCRATCH}/gadi-ci/

# 5 — submit the build job (PBS, normalsr, 1 h, ~210 SU)
cd ${SCRATCH}
qsub gadi-ci/bootstrap_iqtree.sh
```

When the build job finishes (~30-45 min), verify:
```bash
ls -lh ${SCRATCH}/build/iqtree3 ${SCRATCH}/build-profiling/iqtree3
${SCRATCH}/build/iqtree3 --version
```

Then run the remaining stages:
```bash
# Stage 1 — generate benchmark datasets
cd ${SCRATCH} && qsub gadi-ci/generate_datasets.sh

# Stage 2 — single matrix point (validation)
./gadi-ci/submit_benchmark_matrix.sh large_modelfinder 52

# Stage 3 — full matrix (16 jobs, ~6.1 KSU)
./gadi-ci/submit_benchmark_matrix.sh
```

## Build requirements (Gadi modules)

The bootstrap script loads these automatically. Listed here for reference.

| Module | Purpose | Gadi path |
|--------|---------|----------|
| `cmake/3.31.6` | Build system | `/apps/cmake/3.31.6` |
| `intel-compiler-llvm/2024.2.0` | C/C++ compiler (`icx`/`icpx`) | `/apps/intel-tools/wrappers/icx` |
| `eigen/3.3.7` | Required header-only math lib | `/apps/eigen/3.3.7/include/eigen3` |
| `boost/1.84.0` | Required headers + libs | `/apps/boost/1.84.0` |
| `gcc/14.2.0` | Fallback compiler (optional) | `/apps/gcc/14.2.0` |

CMake flags passed by the bootstrap script:
```
-DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3
-DBOOST_ROOT=/apps/boost/1.84.0
-DBoost_NO_SYSTEM_PATHS=ON
-DFETCHCONTENT_SOURCE_DIR_googletest=<scratch>/deps/googletest
-DFETCHCONTENT_FULLY_DISCONNECTED=ON
```

## PBS quick reference (equivalents of SLURM commands)

| SLURM | PBS equivalent |
|-------|---------------|
| `squeue -u $USER` | `qstat -u $USER` |
| `squeue -j <id>` | `qstat -x <id>` |
| `sinfo` | `qstat -Q` |
| `sacct` (history) | `qstat -x <id>` (includes finished with `-x`) |
| `scancel <id>` | `qdel <id>` |
| `scontrol show job` | `qstat -xf <id>` |
| `watch squeue` | `watch -n 10 'qstat -u $USER'` |
