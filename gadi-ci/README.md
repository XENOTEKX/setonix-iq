# gadi-ci/ — NCI Gadi pipeline scripts

Gadi/PBS/Intel counterpart to `setonix-ci/`. These scripts live on
`/scratch/$PROJECT/$USER/iqtree3/gadi-ci/` on Gadi and produce the same
JSON artefacts the dashboard consumes.

## Target system

| Component | Spec |
|-----------|------|
| Machine   | Gadi (NCI, Canberra) |
| CPU       | Intel Xeon Platinum 8268 "Cascade Lake", 48 cores/node, 2 sockets, 4 NUMA |
| Memory    | 192 GB/node (request `mem=190GB` to leave headroom) |
| Scheduler | PBS Professional 2024.1 (`qsub`, `qstat`, `qdel`, `nqstat`) |
| Queue     | `normal` (CPU), `gpuvolta` (NVIDIA V100 CUDA nodes) |
| Profiler  | Intel VTune (`module load intel-vtune/2024.2.0`) + Linux `perf` |
| Storage   | `/home` (10 GB quota), `/scratch/$PROJECT` (time-limited), `/g/data/$PROJECT` |

## Scripts

| File | Purpose |
|------|---------|
| `run_pipeline.sh`       | Small deterministic CI pipeline — login-node friendly. Emits `logs/runs/<YYYY-MM-DD_HHMMSS>.json`. |
| `run_profiling.sh`      | Quick single-run perf stat wrapper (for interactive `qsub -I` sessions). |
| `run_mega_profile.sh`   | Full deep-profile PBS job — perf stat + VTune hotspots + perf record + sampler. |
| `submit_mega_batch.sh`  | Fan out `run_mega_profile.sh` across thread counts via `qsub`. |

## Key differences vs `setonix-ci/`

- `#SBATCH` directives → `#PBS -N / -P / -q / -l`.
- SLURM env vars (`SLURM_JOB_ID`, etc.) → PBS env vars (`PBS_JOBID`,
  `PBS_JOBNAME`, `PBS_QUEUE`, `PBS_NCPUS`, `PBS_NODEFILE`, `PBS_O_HOST`,
  `PBS_O_WORKDIR`).
- AMD Zen 3 raw events (`ex_ret_*`, `ls_l1_d_tlb_miss.*`,
  `bp_l1_tlb_miss_l2_tlb_*`, `ls_dispatch.*`, `ls_tablewalker.*`) replaced
  with Intel Cascade Lake Top-down TMA slot events
  (`topdown-total-slots`, `topdown-slots-issued`, `topdown-slots-retired`,
  `topdown-fetch-bubbles`, `topdown-recovery-bubbles`) plus `LLC-loads` /
  `LLC-load-misses` (Intel publishes LLC alias events, AMD Setonix did not).
- New **VTune hotspots pass** (`vtune -collect hotspots`) with bounded
  wall-time — summary + CSV are parsed into `profile.vtune.*` in the
  emitted `profile_meta.json`.
- PBS job-id strips the `.gadi-pbs` suffix: `RUN_ID="${PBS_JOBID%%.*}"`.
- Queue default: `normal` (Cascade Lake 48c/node). Thread sweep uses
  `4 8 16 24 48` (full-node ceiling) instead of Setonix's `16/32/64/128`.

All artefact file names (`env.json`, `samples.jsonl`, `perf_stat.txt`,
`hotspots.txt`, `perf_folded.txt`, `profile_meta.json`) are identical to
the Setonix scripts so `tools/harvest_scratch.py` and the dashboard
front-end consume both transparently.
