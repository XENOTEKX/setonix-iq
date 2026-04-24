# Setonix IQ-TREE Dashboard

Static dashboard for monitoring [IQ-TREE](http://www.iqtree.org/) pipeline runs on
the [Setonix supercomputer](https://pawsey.org.au/systems/setonix/) at the
Pawsey Supercomputing Centre. Each run pushes structured JSON to this repo; a
GitHub Actions workflow validates it, builds a modern client-side dashboard, and
deploys to GitHub Pages.

**Live dashboard:** `https://xenotekx.github.io/setonix-iq/` *(once Pages is enabled — see below)*

---

## Branches

| Branch | Target system | Scheduler | CPU | Profiler |
|---|---|---|---|---|
| `main`    | **Setonix** (Pawsey) | SLURM | AMD EPYC 7A53 / MI250X | ROCm `rocprof`, `perf` |
| `gadi-iq` | **Gadi** (NCI)       | PBS Pro | Intel Xeon Platinum 8268 | Intel **VTune 2024.2**, `perf` |

Both branches share the same web front-end, data schema, and Python tooling.
Only the job-submission scripts (`setonix-ci/` vs `gadi-ci/`), default paths,
and platform-specific performance counters differ. Records from either system
validate against the unified schema — a single dashboard can ingest mixed
Setonix + Gadi runs. See [CHANGELOG.md](CHANGELOG.md) for the `gadi-iq`
refactor details.

---

## Architecture

```
┌──────────── Setonix (Pawsey HPC) ─────────────┐        ┌────────── GitHub Actions ──────────┐
│                                                │        │                                    │
│  ~/setonix-iq/                                 │        │  .github/workflows/validate.yml    │
│   └── start.sh  ─► run_pipeline.sh             │        │   • jsonschema validation          │
│                    └─► logs/runs/*.json        │  push  │   • pytest tests/                  │
│                    └─► logs/profiles/*.json    │ ─────► │                                    │
│                    └─► git push origin main    │        │  .github/workflows/build.yml       │
│                                                │        │   • tools/build.py → docs/         │
└────────────────────────────────────────────────┘        │   • actions/deploy-pages@v4        │
                                                          │                                    │
                                                          └────────────┬───────────────────────┘
                                                                       │
                                                                       ▼
                                                            GitHub Pages (static site)
                                                            docs/ served as root
```

The dashboard itself is a small set of ES modules (no framework, no build step)
that fetches JSON from `data/` and renders everything client-side.

---

## Repository layout

```
setonix-iq/
├── README.md
├── CHANGELOG.md
├── LICENSE                 MIT
├── Makefile                build/profile/dashboard convenience targets
├── start.sh                Setonix entry point — runs pipeline + git push
├── host.sh                 optional: serve docs/ locally with periodic refresh
│
├── logs/                   SOURCE OF TRUTH — one JSON per run (committed)
│   ├── runs/               .json per run (timing + verify + env + summary)
│   └── profiles/           .json per deep profile (perf counters + hotspots)
│
├── tools/                  data pipeline (Python stdlib + jsonschema)
│   ├── schemas/            Draft-7 JSON schemas for runs & profiles
│   ├── normalize.py        logs/  →  web/data/{runs,profiles}.index.json + per-record files
│   ├── validate.py         schema-validates every file under logs/
│   └── build.py            runs normalize, mirrors web/ → docs/
│
├── tests/                  pytest suite run in CI (validate.yml)
│   ├── test_schemas.py          every run/profile matches its schema
│   ├── test_data_invariants.py  summary consistency, IPC/miss-rate ranges, etc.
│   ├── test_regression.py       wall-time regression guard (xfail by default)
│   └── test_build.py            tools/normalize.py end-to-end
│
├── web/                    dashboard source (deployed as-is to docs/)
│   ├── index.html
│   ├── css/   tokens, layout, components, charts, pages
│   └── js/    main.js, router, state, data, utils
│       ├── components/   copy-button, toast, run-selector
│       ├── charts/       hotspot, microarch, scaling, callstack, flamegraph, timing
│       └── pages/        overview, runs, tests, profiling, gpu, allocation, environment
│
├── docs/                   build output (gitignored; published by Pages workflow)
│
└── .github/workflows/
    ├── validate.yml        schema + pytest on every push/PR to logs/, tools/, tests/
    └── build.yml           build + deploy Pages on every push to logs/, web/, tools/
```

---

## Requirements

### HPC node (Setonix / Pawsey)

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.8 | stdlib only (`tools/` uses `jsonschema` — `pip install -r tools/requirements.txt` in CI) |
| Git | ≥ 2.x | for push |
| CMake / GCC | 3.16+ / 12+ | to build IQ-TREE (`make build-profiling`) |
| perf | kernel perf tools | CPU profiling |
| ROCm / SLURM | latest | GPU + scheduler |

Build IQ-TREE with `-fno-omit-frame-pointer` so `perf -g` can unwind stacks:

```bash
make build-profiling
```

### Local machine (dashboard preview)

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.8 | `pip install jsonschema pytest` (or `tools/requirements.txt`) |
| Web browser | modern | Chart.js 4.4 via CDN; ES modules |

---

## Usage on Setonix

All commands are thin wrappers in `start.sh`:

```bash
./start.sh                 # push current logs/ to GitHub
./start.sh pipeline        # run CI pipeline, then push
./start.sh profile FILE    # run perf profiling, then push
./start.sh deepprofile     # CPU + GPU deep profile, then push
./start.sh generate        # local preview — validate + build → docs/
./start.sh status          # SLURM jobs + pawseyAccountBalance
```

The Setonix pipeline scripts write **one JSON file per run** directly to
`logs/runs/<YYYY-MM-DD_HHMMSS>.json` (and `logs/profiles/<id>.json` for deep
profiles). They're validated by `tools/validate.py` against the schemas in
`tools/schemas/` before being committed.

## Usage locally (preview)

```bash
git pull

# Validate + build static site into docs/
python3 tools/validate.py
python3 tools/build.py

# Serve
python3 -m http.server -d docs 8000
# → open http://localhost:8000
```

Or just:

```bash
make dashboard   # validate + build + commit + push
make test        # run pytest
```

---

## Data format

Every run is a single JSON file conforming to `tools/schemas/run.schema.json`:

```jsonc
{
  "run_id": "2026-04-18_201515",          // required — YYYY-MM-DD_HHMMSS
  "slurm_id": "41683322",
  "run_type": "pipeline",                 // pipeline | profile | deep
  "label": "large_mf_8t_baseline",
  "description": "50 taxa, ~5k sites",
  "timing": [                             // required
    { "command": "iqtree3 -s turtle.fa …", "time_s": 17.71 }
  ],
  "verify": [
    { "file": "turtle.fa", "status": "pass", "expected": -5681.1, "reported": -5681.1, "diff": 0.0 }
  ],
  "env": {                                // required — hostname + date + …
    "hostname": "setonix-node",
    "date": "2026-04-18T20:15:15+08:00",
    "cpu": "AMD EPYC 7A53 64-Core Processor",
    "cores": 64, "gcc": "12.2.0", "rocm": "6.3.0"
  },
  "summary": {                            // required — counts + total_time
    "pass": 7, "fail": 0, "total_time": 17.71
  },
  "profile": {
    "dataset": "turtle.fa", "threads": 1,
    "metrics": { "IPC": 2.945, "cache-miss-rate": 3.17, … },
    "hotspots":     [ { "function": "computeLikelihood", "percent": 42.1, "module": "iqtree3" }, … ],
    "folded_stacks":[ { "stack": "main;foo;bar",         "count":   1234                          }, … ]
  }
}
```

Deep profiles follow `tools/schemas/profile.schema.json` and add a `gpu` section
with parsed `rocm-smi` output. Schemas are the source of truth — both are enforced
in CI on every push.

---

## Dashboard features

The dashboard is organised into seven pages (sidebar nav, hash-routed):

- **Overview** — hero stats, run picker with dataset/IPC/wall summary, config card
  (alignment / model / system), hotspot breakdown, multi-run microarch radar,
  per-dataset thread-scaling curves, top call stacks, copy-all commands button.
- **All Runs** — searchable / sortable / status-filterable list. Each row expands
  to show env, metrics, verification table, and commands with per-command wall time.
- **Tests** — aggregated verification across every run (pass/fail, |Δlnℒ|).
- **Profiling** — deep dive: CPU counters, top hotspots, top call stacks, and a
  lightweight flamegraph rendered client-side from `folded_stacks`.
- **GPU** — `rocm-smi` key/value grid per profile.
- **Allocation** — Pawsey SU balance (checked live via `./start.sh status`).
- **Environment** — full env dump per run.

All charts use Chart.js 4.4 via CDN. The dashboard is keyboard navigable, honours
`prefers-reduced-motion`, and reports copy actions via a toast live region.

---

## CI / CD

Two workflows under `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `validate.yml` | push / PR touching `logs/`, `tools/`, `tests/` | `pip install` deps, run `tools/validate.py`, run `pytest tests/` |
| `build.yml`    | push to `main` touching `logs/`, `web/`, `tools/` | build `docs/` via `tools/build.py`, upload Pages artifact, deploy |

Tests verify:

- every run / profile matches its JSON schema
- `run_id`s are unique
- `summary.total_time` matches the sum of `timing[].time_s` (±1 s + 1 %)
- IPC ∈ (0, 10); miss / stall rates ∈ [0, 100]
- hotspot percents sum to ≤ 100 within a small tolerance
- wall time does not regress more than 20 % (warn-only `xfail`)

Run locally with `make test`.

---

## Enabling GitHub Pages

After the first successful `build.yml` run:

1. Go to the repo → **Settings** → **Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions** (not a branch).
3. The site will be served at `https://<org>.github.io/setonix-iq/` after the next
   push to `main` touching `web/`, `tools/`, or `logs/`.

---

## Run naming

`YYYY-MM-DD_HHMMSS` derived from the pipeline start time — sortable,
filesystem-safe, SLURM-ID-independent. The SLURM ID is preserved as
`slurm_id` inside each JSON for correlation with scratch log files.

---

## License

MIT — see [`LICENSE`](LICENSE).
# Setonix IQ-TREE Dashboard

Self-contained dashboard for monitoring IQ-TREE pipeline runs on the [Setonix supercomputer](https://pawsey.org.au/systems/setonix/) (Pawsey Supercomputing Centre). Automatically collects timing, verification, profiling, GPU, and environment data from each run and renders it as a static HTML dashboard deployed to GitHub Pages.

**Live dashboard:** *(deployed via GitHub Pages)*

---

## Architecture

```
┌──────────────────────── Setonix (Pawsey HPC) ────────────────────────┐
│                                                                       │
│  /scratch/$PAWSEY_PROJECT/$USER/iqtree3/                             │
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
git clone https://github.com/<YOUR_USERNAME>/setonix-iq.git
cd setonix-iq
```

### 2. Symlinks are created automatically

`start.sh` calls `check_links()` which creates symlinks from `website/results` and `website/profiles` to the pipeline output directories on scratch:

```
website/results  →  /scratch/$PAWSEY_PROJECT/$USER/iqtree3/setonix-ci/results
website/profiles →  /scratch/$PAWSEY_PROJECT/$USER/iqtree3/setonix-ci/profiles
```

These symlinks are gitignored (they only work on Setonix).

### 3. Configure git push access

```bash
# If using HTTPS with a PAT:
git remote set-url origin https://<TOKEN>@github.com/<YOUR_USERNAME>/setonix-iq.git

# Or use SSH if configured:
git remote set-url origin git@github.com:<YOUR_USERNAME>/setonix-iq.git
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
hostname: setonix-node
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
- **Allocation** — CPU/GPU SU balance
- **Environment** — full key-value dump per run

---

## Target HPC Environment

| Component | Spec |
|-----------|------|
| System | Setonix (Pawsey Supercomputing Centre, Perth) |
| CPU | AMD EPYC 7A53 "Trento" 64-core |
| GPU | 8× AMD Instinct MI250X per node (128 GB HBM2e each) |
| GPU Stack | HIP/ROCm |
| Project | `$PAWSEY_PROJECT` / `$PAWSEY_PROJECT-gpu` |
| Scratch | `/scratch/$PAWSEY_PROJECT/$USER/` |
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
| Symlink errors on Setonix | Verify `/scratch/$PAWSEY_PROJECT/$USER/iqtree3/setonix-ci/results` exists |
| `git push` fails on Setonix | Check remote URL has valid token: `git remote -v` |
| Dashboard doesn't update on GitHub Pages | Check `docs/index.html` was committed; Pages serves from `docs/` on `main` |
| Charts not rendering | Requires internet for Chart.js CDN (`cdn.jsdelivr.net`) |
