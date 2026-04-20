# Setonix IQ-TREE Dashboard

Self-contained dashboard for monitoring IQ-TREE pipeline runs on the [Setonix supercomputer](https://pawsey.org.au/systems/setonix/) (Pawsey Supercomputing Centre). Automatically collects timing, verification, profiling, GPU, and environment data from each run and renders it as a static HTML dashboard deployed to GitHub Pages.

**Live dashboard:** [https://xenotekx.github.io/setonix-iq/](https://xenotekx.github.io/setonix-iq/)

---

## Architecture

```
┌──────────────────────── Setonix (Pawsey HPC) ────────────────────────┐
│                                                                       │
│  /scratch/pawsey1351/asamuel/iqtree3/                                │
│  ├── setonix-ci/                                                      │
│  │   ├── run_pipeline.sh    ← runs IQ-TREE tests, writes logs        │
│  │   ├── run_profiling.sh   ← perf stat profiling                    │
│  │   ├── results/           ← time_log_*.tsv, verify_*.txt, etc.     │
│  │   └── profiles/          ← perf_stat_*.json                       │
│  └── test_scripts/test_data/ ← alignment datasets                    │
│                                                                       │
│  ~/setonix-iq/  (this repo, cloned on Setonix)                       │
│  ├── website/results → symlink to setonix-ci/results                 │
│  ├── website/profiles → symlink to setonix-ci/profiles               │
│  ├── start.sh          ← entry point: pipeline → generate → push     │
│  ├── serve.py          ← parses logs, generates dashboard HTML       │
│  ├── logs/runs.json    ← cached parsed data (committed to git)       │
│  ├── docs/index.html   ← generated dashboard (GitHub Pages source)   │
│  └── dashboard.html    ← same file, for local viewing                │
│                                                                       │
│  ./start.sh pipeline  →  git push  ─────────────────────────┐        │
└──────────────────────────────────────────────────────────────┼────────┘
                                                               │
                                                               ▼
                                                      GitHub (main branch)
                                                      ├── docs/index.html → GitHub Pages
                                                      └── logs/runs.json  → offline cache
                                                               │
                      ┌────────────────────────────────────────┘
                      ▼
┌──────────── Local Machine (Mac/Linux) ────────────┐
│                                                    │
│  git pull                                          │
│  python3 serve.py   ← uses logs/runs.json fallback │
│  open dashboard.html                               │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Requirements

### HPC node (Setonix / Pawsey)

| Requirement | Version | Notes |
|---|---|---|
| **OS** | RHEL 8 / CentOS Stream 8 | Any Linux with perf support will work |
| **Python** | ≥ 3.8 | Used by `serve.py` and `data.py` |
| **Git** | ≥ 2.x | For committing + pushing dashboard |
| **CMake** | ≥ 3.16 | Required to build IQ-TREE |
| **GCC** | ≥ 10 (recommended 12) | C++17 support; load via `module load gcc/12.2.0` |
| **Make** | any | Standard GNU Make |
| **perf** | Linux kernel perf tools | For CPU profiling (`perf stat`, `perf record`) |
| **ROCm** | ≥ 6.3.0 | AMD GPU support; load via `module load rocm` |
| **rocprofiler-compute** | 3.0.0 | GPU kernel profiling; `module load rocprofiler-compute/3.0.0` |
| **rocprofiler-systems** | 6.3.0 | GPU trace collection; `module load rocprofiler-systems/6.3.0` |
| **SLURM** | any | Job scheduler — required for GPU node access |

> **Build flag:** Always build IQ-TREE with `-fno-omit-frame-pointer` for profiling so `perf -g` can unwind call stacks:
> ```bash
> make build-profiling   # sets -fno-omit-frame-pointer automatically
> ```
> Without this flag, ~72% of profiling samples will appear as `[unknown]` in the flamegraph.

### Local machine (macOS / Linux — dashboard only)

| Requirement | Version | Notes |
|---|---|---|
| **Python** | ≥ 3.8 | Runs `serve.py` to regenerate HTML |
| **Git** | ≥ 2.x | `git pull` to sync logs from HPC |
| **Web browser** | any modern | Open `dashboard.html` — no server needed |
| **Node.js** | ≥ 16 *(optional)* | Only needed for JS syntax checking during dev |

No Python packages beyond the standard library are required — `serve.py` and `data.py` use only `json`, `os`, `re`, `datetime`, and `pathlib`.

---

## Why static HTML?

Setonix disables SSH TCP forwarding on shared login nodes for security. Port forwarding doesn't work, so a live server is not viable. Instead, `serve.py` bakes all run data directly into a self-contained HTML file with no external dependencies (Chart.js is loaded from CDN).

---

## Setup on Setonix

### 1. Clone the repo

```bash
cd ~
git clone https://github.com/XENOTEKX/setonix-iq.git
cd setonix-iq
```

### 2. Symlinks are created automatically

`start.sh` calls `check_links()` which creates symlinks from `website/results` and `website/profiles` to the pipeline output directories on scratch:

```
website/results  →  /scratch/pawsey1351/asamuel/iqtree3/setonix-ci/results
website/profiles →  /scratch/pawsey1351/asamuel/iqtree3/setonix-ci/profiles
```

These symlinks are gitignored (they only work on Setonix).

### 3. Configure git push access

```bash
# If using HTTPS with a PAT:
git remote set-url origin https://<TOKEN>@github.com/XENOTEKX/setonix-iq.git

# Or use SSH if configured:
git remote set-url origin git@github.com:XENOTEKX/setonix-iq.git
```

---

## Usage on Setonix

All commands are run via `start.sh`:

```bash
# Run the full pipeline, generate dashboard, push to GitHub
./start.sh pipeline

# Just regenerate dashboard from existing logs and push
./start.sh generate

# Run profiling on a dataset, generate, push
./start.sh profile

# Run deep CPU+GPU profiling (submits GPU SLURM job)
./start.sh deepprofile

# Check SLURM jobs and allocation balance
./start.sh status

# Only push (no regeneration)
./start.sh sync
```

### What `./start.sh pipeline` does

1. Runs `run_pipeline.sh` — executes IQ-TREE test cases, writes:
   - `results/time_log_<RUN_ID>.tsv` — command + wall time + memory per test
   - `results/verify_<RUN_ID>.txt` — PASS/FAIL with expected vs reported likelihood
   - `results/env_<RUN_ID>.txt` — hostname, CPU, cores, GCC, ROCm versions
   - `results/gpu_info_<RUN_ID>.txt` — `rocm-smi` output
   - `profiles/perf_stat_<RUN_ID>.json` — `perf stat` counters (IPC, cache misses, etc.)
2. Runs `serve.py` which:
   - Reads all `time_log_*.tsv` files via `data.py` to build a list of runs
   - Exports parsed data to `logs/runs.json` (committed to git for offline use)
   - Injects the JSON into `website/index.html` template → produces `docs/index.html` + `dashboard.html`
3. Commits everything and pushes to GitHub
4. GitHub Pages serves `docs/index.html` at the live URL

---

## Usage on Local Machine

```bash
# Pull the latest data (including logs/runs.json)
git pull

# Regenerate dashboard from cached logs
python3 serve.py

# Open in browser
open dashboard.html        # macOS
xdg-open dashboard.html    # Linux
```

`data.py` detects that `website/results/` doesn't exist (no Setonix symlink) and automatically falls back to reading `logs/runs.json`.

---

## Repository Structure

```
setonix-iq/
├── README.md              ← this file
├── CHANGELOG.md           ← project progress log
├── CLAUDE.md              ← AI agent development guide
├── .gitignore
│
├── start.sh               ← Setonix entry point (pipeline/generate/push)
├── serve.py               ← Dashboard generator (template + data → HTML)
│
├── website/
│   ├── index.html          ← Dashboard template (fetches data via JS)
│   ├── api/
│   │   └── data.py         ← Parses raw log files into structured JSON
│   ├── results/            ← SYMLINK to scratch (gitignored)
│   ├── profiles/           ← SYMLINK to scratch (gitignored)
│   └── assets/             ← Flamegraph SVG symlink (gitignored)
│
├── logs/
│   ├── runs/               ← One JSON file per run (COMMITTED to git)
│   │   ├── 2026-04-18_201515.json
│   │   └── ...
│   └── profiles/           ← Deep profile JSON files (COMMITTED to git)
│       ├── deep_profile_41686771.json
│       └── ...
│
├── docs/
│   └── index.html          ← Generated dashboard (GitHub Pages source)
│
└── dashboard.html          ← Generated dashboard (convenience copy)
```

---

## Data Flow

### On Setonix (data available)

```
Pipeline scripts write raw files to scratch
         │
         ▼
website/results/ ──symlink──► time_log_*.tsv, verify_*.txt, env_*.txt, gpu_info_*.txt
website/profiles/ ─symlink──► perf_stat_*.json
         │
         ▼
data.py  ──parses──►  structured JSON (list of run objects)
         │
         ├──► logs/runs/<date-time>.json  (one file per run, committed to git)
         ├──► website/api/runs.json       (reference copy, gitignored)
         └──► embedded in HTML            (docs/index.html, dashboard.html)
```

### On local machine (no Setonix)

```
logs/runs/*.json  ──loaded by data.py──►  same structured JSON
         │
         └──► embedded in HTML (dashboard.html)
```

---

## Deep Profiling

`./start.sh deepprofile` submits a GPU SLURM job (`run_deep_profile.sh`) that runs 5 profiling stages:

| Stage | Tool | Output |
|-------|------|--------|
| 1. CPU counters | `perf stat` | 21 hardware counters (IPC, cache, TLB, branch, stalls) |
| 2. CPU hotspots | `perf record` + `perf report` | Function-level CPU time breakdown |
| 3. GPU metrics | `rocm-smi --json` | Temperature, power, VRAM, utilization, clocks |
| 4. GPU traces | `rocprofv3` | Kernel dispatches, HIP API calls, memory copies |
| 5. System info | `uname`, `lscpu`, `rocm-smi` | CPU/GPU hardware identification |

All output is consolidated into a single JSON file: `deep_profiles/deep_profile_<SLURM_ID>.json`

**Modules used:** `rocm/6.3.0`, `rocprofiler-compute/3.0.0`, `rocprofiler-systems/6.3.0`

The dashboard's **Profiling** page automatically renders deep profile data when available, including:
- CPU performance metrics (IPC, stall rates, cache/TLB/branch miss rates)
- Hardware counter bar chart (log scale)
- Pipeline stall breakdown (doughnut chart)
- Function hotspot table from `perf report`
- GPU hardware metrics grid
- GPU kernel dispatch and memory copy tables
- IPC and stall trend charts across profiling runs
- Collapsible section showing exact profiling commands used

---

## Raw Log File Formats

### `time_log_<RUN_ID>.tsv`

Tab-separated, one row per test command:

```
command	time_s	memory_kb
/path/to/iqtree3 -s turtle.fa -m GTR+G4 -seed 1	45.230	524288
/path/to/iqtree3 -s example.phy -seed 1	12.890	262144
```

### `verify_<RUN_ID>.txt`

Likelihood verification results:

```
PASS: turtle.fa -- Expected: -6138.123, Reported: -6138.123, Abs-diff: 0.000
FAIL: broken.phy -- Expected: -1234.567, Reported: -1234.999, Abs-diff: 0.432
```

### `env_<RUN_ID>.txt`

Key-value pairs:

```
date: 2026-04-18 14:30:00
hostname: nid002145
cpu: AMD EPYC 7A53 64-Core Processor
cores: 64
gcc: 12.2.0
rocm: 5.4.3
```

### `gpu_info_<RUN_ID>.txt`

Raw `rocm-smi` output (parsed for temperature, power, VRAM, utilization).

### `perf_stat_<RUN_ID>.json`

```json
{
  "metrics": {
    "IPC": 1.63,
    "cache-miss-rate": "2.14",
    "L1-dcache-miss-rate": "0.87",
    "branch-miss-rate": "0.45",
    "instructions": 45000000000,
    "cycles": 27600000000,
    "cache-references": 120000000,
    "cache-misses": 2568000
  }
}
```

---

## Dashboard Features

- **Overview** — current run status, best speedup (multi-thread vs 1T baselines), fastest run time, IQ-TREE configuration card (alignment, model & results, system info with command line), performance leaderboard, hotspot/microarch/scaling charts, latest deep profile summary
- **All Runs** — leaderboard-style list with search, filter (pass/fail), sort (date/time/IPC), expandable detail panels per run
- **Per-run details** — commands with copy-to-clipboard, verification table, perf counters, environment info
- **Copy All as Script** — one-click copy of all commands from a run for re-execution on Setonix
- **Tests** — verification results with expected vs reported likelihood, pass/fail filtering
- **Timing** — bar/doughnut chart of per-command wall time, trend chart across runs
- **Profiling** — IPC, cache miss rate, branch mispredict, hardware counter bar chart, IPC trend across runs
- **GPU** — temperature, power, VRAM, utilization from `rocm-smi`
- **Allocation** — CPU/GPU SU balance for `pawsey1351`
- **Environment** — full key-value dump per run

---

## Target HPC Environment

| Component | Spec |
|-----------|------|
| System | Setonix (Pawsey Supercomputing Centre, Perth) |
| CPU | AMD EPYC 7A53 "Trento" 64-core |
| GPU | 8× AMD Instinct MI250X per node (128 GB HBM2e each) |
| GPU Stack | HIP/ROCm |
| Project | `pawsey1351` / `pawsey1351-gpu` |
| Scratch | `/scratch/pawsey1351/asamuel/` |
| Scheduler | SLURM |

---

## Run Naming Convention

Runs are identified by **date-time** strings derived from the pipeline's start timestamp:

```
2026-04-18_201515   →   April 18 2026, 20:15:15
```

Format: `YYYY-MM-DD_HHMMSS` — sortable, filesystem-safe, human-readable.

The SLURM job ID is preserved inside each run's JSON as `slurm_id` for cross-referencing
with raw log files on scratch (which still use SLURM IDs: `time_log_41683322.tsv`).

### Log storage

Each run is stored as an individual JSON file:

```
logs/runs/
├── 2026-04-18_201515.json   ← one complete run
├── 2026-04-19_143022.json
└── ...
```

Benefits over a monolithic `runs.json` array:
- **Clean git diffs** — new run = new file, no rewrite of existing data
- **No merge conflicts** — independent files don't collide
- **Easy archival** — delete old runs by removing individual files
- **Small commits** — ~5KB per run instead of rewriting the full history

### Adding a run manually

1. Place raw files in `website/results/` (or scratch on Setonix):
   - `time_log_<SLURM_ID>.tsv`
   - `verify_<SLURM_ID>.txt`
   - `env_<SLURM_ID>.txt` (must contain `date:` line for naming)
   - Optionally: `gpu_info_<SLURM_ID>.txt`
2. Place profiling in `website/profiles/`:
   - `perf_stat_<SLURM_ID>.json`
3. Run `python3 serve.py` — this parses, creates `logs/runs/<date-time>.json`, and regenerates the dashboard

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Dashboard shows "No pipeline runs" locally | Run `git pull` to get latest `logs/runs/*.json` files |
| Symlink errors on Setonix | Verify `/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/results` exists |
| `git push` fails on Setonix | Check remote URL has valid token: `git remote -v` |
| Dashboard doesn't update on GitHub Pages | Check `docs/index.html` was committed; Pages serves from `docs/` on `main` |
| Charts not rendering | Requires internet for Chart.js CDN (`cdn.jsdelivr.net`) |
