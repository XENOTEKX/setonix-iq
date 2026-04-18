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
│   └── runs.json           ← Cached parsed data (COMMITTED to git)
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
         ├──► logs/runs.json        (cached for offline, committed to git)
         ├──► website/api/runs.json (reference copy, gitignored)
         └──► embedded in HTML      (docs/index.html, dashboard.html)
```

### On local machine (no Setonix)

```
logs/runs.json  ──loaded by data.py──►  same structured JSON
         │
         └──► embedded in HTML (dashboard.html)
```

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

- **Overview** — current run status, test pass rate, pipeline time, IPC, cross-run averages
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

## Adding a New Run

Runs are auto-detected by filename pattern. To add data manually:

1. Place files in `website/results/` (or the scratch directory on Setonix):
   - `time_log_<YOUR_RUN_ID>.tsv`
   - `verify_<YOUR_RUN_ID>.txt`
   - `env_<YOUR_RUN_ID>.txt`
   - Optionally: `gpu_info_<YOUR_RUN_ID>.txt`
2. Place profiling in `website/profiles/`:
   - `perf_stat_<YOUR_RUN_ID>.json`
3. Run `python3 serve.py` to regenerate

The `<RUN_ID>` must be consistent across all files for a run — `data.py` uses `time_log_*.tsv` filenames as the primary index.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Dashboard shows "No pipeline runs" locally | Run `git pull` to get latest `logs/runs.json` |
| Symlink errors on Setonix | Verify `/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/results` exists |
| `git push` fails on Setonix | Check remote URL has valid token: `git remote -v` |
| Dashboard doesn't update on GitHub Pages | Check `docs/index.html` was committed; Pages serves from `docs/` on `main` |
| Charts not rendering | Requires internet for Chart.js CDN (`cdn.jsdelivr.net`) |
