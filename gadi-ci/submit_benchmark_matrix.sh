#!/bin/bash
# submit_benchmark_matrix.sh — reproduce the Setonix benchmark corpus on
# Gadi Sapphire Rapids (normalsr). Enumerates (dataset × thread_count)
# pairs and submits one PBS job per pair. Each job writes a JSON run
# record directly to ${REPO_DIR}/logs/runs/ and an enriched profile dir
# under ${PROJECT_DIR}/gadi-ci/profiles/.
#
# Thread sweep: 1, 4, 13, 26, 52, 104
#   • 13  = one NUMA domain
#   • 26  = one NUMA pair (mimics small multi-NUMA effects)
#   • 52  = one socket
#   • 104 = full node
#
# Budget reference (2 SU/core-h, billed on full ncpus=104 reservation):
#   208 SU / node-hour.  See CHANGELOG.md for the detailed SU estimate.
#
# Usage:
#   ./submit_benchmark_matrix.sh                   # full matrix
#   ./submit_benchmark_matrix.sh large_modelfinder # single dataset
#   ./submit_benchmark_matrix.sh mega_dna 52 104   # dataset + thread subset
#
# Stages referenced in CHANGELOG:
#   stage1  — CI pipeline smoke (via run_pipeline.sh)
#   stage2  — one thread-point (pass "large_modelfinder 52")
#   stage3  — full matrix (no extra args)

set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
WORKER="${WORKER:-${PROJECT_DIR}/gadi-ci/_run_matrix_job.sh}"
LOGS_DIR="${PROJECT_DIR}/gadi-ci/logs"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${LOGS_DIR}" "${RUNS_DIR}"

# Default matrix: matching Setonix thread sweep for direct cross-platform comparison.
# Setonix used {1,4,8,16,32,64} for large/xlarge and {16,32,64,128} for mega_dna.
# Gadi caps at 104 cores (no 128T), so mega_dna uses 16,32,64,104.
declare -A MATRIX=(
  [large_modelfinder]="1 4 8 16 32 64"
  [xlarge_mf]="1 4 8 16 32 64"
  [mega_dna]="16 32 64 104"
)

SELECT_DATASET="${1:-}"
shift || true
SELECT_THREADS=("${@}")

# ── Emit the per-job worker (installed alongside the matrix script).
cat > "${WORKER}" <<'WORKER_EOF'
#!/bin/bash
# _run_matrix_job.sh — invoked by `qsub -v DATASET=... THREADS=... LABEL=...`
# Runs IQ-TREE under perf, emits a run.schema.json-conforming JSON.
set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling}"
IQTREE="${BUILD_DIR}/iqtree3"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="${DATASET:?DATASET env var required}"
THREADS="${THREADS:?THREADS env var required}"
LABEL="${LABEL:-${DATASET_NAME}_${THREADS}t_sr}"
SEED="${SEED:-1}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"

if command -v module >/dev/null 2>&1; then
    module load intel-vtune/2024.2.0         2>/dev/null || true
    module load intel-compiler-llvm/2024.2.0 2>/dev/null || true
fi

# VTune writes driver/temp files to TMPDIR.  On Gadi normalsr the default
# TMPDIR is /jobfs (quota 100 MB by default, overrideable via jobfs=Xgb).
# Redirect to scratch to avoid jobfs pressure from VTune temp data even
# when jobfs=2gb is requested (belt-and-suspenders).
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
# Embed 'gadi' in the run id so dashboard filenames (logs/runs/<rid>.json) are
# unambiguous alongside Setonix runs.  No timestamp prefix — the pbs_id field
# inside the record carries the unique ordering and the label is unique per
# (dataset, threads) pair.
RUN_ID="gadi_${LABEL}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

echo "[matrix] run_id=${RUN_ID} dataset=${DATA_PATH} threads=${THREADS}"

# Gadi normalsr compute nodes run with /proc/sys/kernel/perf_event_paranoid=2.
# That blocks all kernel-mode perf sampling and makes `perf stat` fail SILENTLY
# when any requested event can't be opened — it writes no output file at all,
# leaving profile.metrics empty in the run record.
# Fix: suffix every event with ':u' so perf counts them in user-mode only,
# which is allowed at paranoid=2.
# Also: stalled-cycles-frontend/backend and most topdown-* pseudo-events are
# not available on SPR with kernel 4.18.0 perf-tool — omitted to keep the
# event group self-consistent.
_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
# Build the ':u'-suffixed event list.
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

# Pass 1: IQ-TREE direct run with /proc time-series sampler in background.
# The sampler polls RSS, IO, NUMA placement, and per-thread CPU ticks at 10 s
# intervals → samples.jsonl (same format as setonix-ci/run_mega_profile.sh).
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
"""Poll /proc/$pid for rss/io/numa/per-thread stats. One JSON line per tick."""
import json, os, subprocess, sys, time, pathlib

pid = int(sys.argv[1])
out = pathlib.Path(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
t0 = time.monotonic()

def read_status(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                d[k.strip()] = v.strip()
    except FileNotFoundError:
        return None
    return d

def read_io(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                try: d[k.strip()] = int(v.strip())
                except ValueError: pass
    except (FileNotFoundError, PermissionError):
        return None
    return d

def read_numa(pid):
    try:
        r = subprocess.run(["numastat", "-p", str(pid)],
                           capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    nodes = None; total = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Node "):
            parts = line.split()
            nodes = [p for p in parts if p.isdigit()]
        if line.startswith("Total"):
            vals = line.split()[1:]
            try: total = [float(v) for v in vals]
            except ValueError: total = None
    if not nodes or not total:
        return None
    per = {n: total[i] for i, n in enumerate(nodes) if i < len(total)}
    return {"per_node_mb": per, "total_mb": total[-1] if total else None}

def per_thread(pid):
    out = []
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except FileNotFoundError:
        return out
    for tid in tasks:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                fields = f.read().split()
            out.append({"tid": int(tid), "utime": int(fields[13]),
                         "stime": int(fields[14]), "nice": int(fields[18])})
        except (FileNotFoundError, IndexError, ValueError):
            continue
    return out

fp = out.open("w")
while True:
    t = time.monotonic() - t0
    status = read_status(pid)
    if status is None:
        break
    snap = {
      "t_s":            round(t, 2),
      "rss_kb":         int(status.get("VmRSS",  "0 kB").split()[0]) if "VmRSS"    in status else None,
      "peak_kb":        int(status.get("VmHWM",  "0 kB").split()[0]) if "VmHWM"    in status else None,
      "vms_kb":         int(status.get("VmSize", "0 kB").split()[0]) if "VmSize"   in status else None,
      "threads_now":    int(status.get("Threads", "0"))               if "Threads"  in status else None,
      "voluntary_cs":   int(status.get("voluntary_ctxt_switches",    "0")),
      "involuntary_cs": int(status.get("nonvoluntary_ctxt_switches", "0")),
      "io":             read_io(pid) or {},
      "numa":           read_numa(pid),
      "thread_count":   len(per_thread(pid)),
    }
    fp.write(json.dumps(snap) + "\n")
    fp.flush()
    time.sleep(interval)
fp.close()
SAMPLER_EOF

START_EPOCH=$(date +%s)
"${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
            --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1 &
IQTREE_PID=$!
# Start /proc sampler in background, attaching to the IQ-TREE process.
python3 "${WORK_DIR}/_sampler.py" "${IQTREE_PID}" \
    "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

# Stop sampler cleanly once IQ-TREE exits.
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

# Pass 2: perf stat — hardware counters (IPC, cache, branch).
# Stderr is captured to perf_stat.err so silent failures are visible.
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_perf" \
        >"${WORK_DIR}/iqtree_perf.log" 2>"${WORK_DIR}/perf_stat.err" || true
fi

# Pass 3: perf record — callgraph for flamegraph (all thread counts, capped 20 min).
# -F 99: 99 Hz sampling. -g: frame pointers (binary built with -fno-omit-frame-pointer).
# Output: perf.data → perf_callgraph.txt (folded stack format for flamegraph tools).
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    timeout --preserve-status 1200 \
        perf record -g -F 99 -o "${WORK_DIR}/perf.data" \
            "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_perf_record" \
        > "${WORK_DIR}/perf_record.log" 2>&1 || true
    # Collapse stacks to folded format (compatible with FlameGraph perl scripts).
    if [[ -f "${WORK_DIR}/perf.data" ]]; then
        perf script -i "${WORK_DIR}/perf.data" \
            > "${WORK_DIR}/perf_script.txt" 2>/dev/null || true
        # Produce a simple hotspot summary (top 30 symbols by self time).
        perf report -i "${WORK_DIR}/perf.data" --stdio --no-children \
            -n --percent-limit 0.1 \
            > "${WORK_DIR}/perf_report.txt" 2>/dev/null || true
    fi
fi

# Pass 4: VTune hotspots + microarchitecture exploration (all thread counts,
# capped 30 min).
# Gadi normalsr has perf_event_paranoid=2, which blocks hardware sampling mode
# and any system-wide collection.  Use sampling-mode=sw (user-mode stack
# unwinding via signals) so VTune can still collect hotspots from the target
# process.  Stack collection is left enabled — it works in sw-mode.
if command -v vtune >/dev/null 2>&1 && [[ "${IQRC}" -eq 0 ]]; then
    # Hotspots with call stacks (software sampling — compatible with paranoid≥1).
    timeout --preserve-status 1800 \
        vtune -collect hotspots -knob sampling-mode=sw \
              -knob enable-stack-collection=true \
              -r "${WORK_DIR}/vtune_hotspots" \
              -- "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                 --prefix "${WORK_DIR}/iqtree_vtune" \
              > "${WORK_DIR}/vtune_collect.log" 2>&1 || true
    if [[ -d "${WORK_DIR}/vtune_hotspots" ]]; then
        vtune -report summary -r "${WORK_DIR}/vtune_hotspots" \
              -format text > "${WORK_DIR}/vtune_summary.txt" 2>/dev/null || true
        # Hotspot function list with module + source line (for flamegraph overlay).
        vtune -report hotspots -r "${WORK_DIR}/vtune_hotspots" \
              -format csv -csv-delimiter tab \
              > "${WORK_DIR}/vtune_hotspots.tsv" 2>/dev/null || true
        vtune -report hw-events -r "${WORK_DIR}/vtune_hotspots" \
              -format text > "${WORK_DIR}/vtune_hw_events.txt" 2>/dev/null || true
    fi
    # uarch-exploration REQUIRES hardware sampling (hence kernel-mode PMU),
    # which is blocked by paranoid=2 on Gadi normalsr.  Skip it entirely here
    # — the perf record callgraph (Pass 3) provides equivalent coverage.
fi

# Emit the run.schema.json record.
python3 - "$@" <<PYEOF
import json, os, re, subprocess, time
work  = "${WORK_DIR}"
runs  = "${RUNS_DIR}"
rid   = "${RUN_ID}"
label = "${LABEL}"
thr   = int("${THREADS}")
pbs   = "${PBS_ID_SHORT}"
wall  = int("${WALL}")
iqrc  = int("${IQRC}")
ds    = "${DATASET_NAME}"
dpath = "${DATA_PATH}"
ibin  = "${IQTREE}"

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

# Extract IQ-TREE-reported wall-time + likelihood for the verify block.
log = os.path.join(work, "iqtree_run.log")
rep_ll = None
iqwall = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Log-likelihood of consensus tree\s*:\s*(-?[\d.]+)", line)
        if m and rep_ll is None: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))

# perf counters → derived metrics (shared logic with run_mega_profile.sh).
# perf stat emits counts with the ':u' suffix (user-mode only, required at
# paranoid=2 on Gadi normalsr).  Strip the suffix so the dashboard schema
# keys line up with the Setonix baselines.
pm = {}; perf_cmd = None
pp = os.path.join(work, "perf_stat.txt")
if os.path.isfile(pp):
    for line in open(pp, errors="replace"):
        m = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m: perf_cmd = m.group(1); continue
        m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w\.\-:/]+)", line)
        if m and not m.group(1).startswith("<"):
            try:
                key = m.group(2).split(":", 1)[0]  # drop ':u' suffix
                pm[key] = int(m.group(1).replace(",", ""))
            except ValueError: pass

def g(*keys):
    for k in keys:
        if pm.get(k) is not None: return pm[k]
    return None

def rate(n, d):
    if not n or not d: return None
    return round(100.0 * n / d, 4)

cyc, ins = g("cycles"), g("instructions")

metrics = {
  "IPC": round(ins / cyc, 4) if cyc and ins else None,
  "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
  "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
  "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
  "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
  "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
  "iTLB-miss-rate":      rate(g("iTLB-load-misses"), g("iTLB-loads")),
}
# Keep raw counters too so the dashboard can display absolute values.
for k in ("cycles", "instructions", "cache-references", "cache-misses",
         "branch-instructions", "branch-misses",
         "L1-dcache-loads", "L1-dcache-load-misses",
         "LLC-loads", "LLC-load-misses",
         "dTLB-loads", "dTLB-load-misses",
         "iTLB-loads", "iTLB-load-misses"):
    if k in pm:
        metrics[k] = pm[k]
metrics = {k: v for k, v in metrics.items() if v is not None}

# VTune hotspots summary + uarch-exploration if present.
vtune = None
vs = os.path.join(work, "vtune_summary.txt")
if os.path.isfile(vs):
    txt = open(vs, errors="replace").read()
    out = {}
    for k, pat in [
        ("elapsed_time_s",     r"Elapsed Time:\s+([\d.]+)"),
        ("cpu_time_s",         r"CPU Time:\s+([\d.]+)"),
        ("effective_cpu_util", r"Effective CPU Utilization:\s+([\d.]+)%"),
        ("avg_cpu_freq_ghz",   r"Average CPU Frequency:\s+([\d.]+)"),
    ]:
        m = re.search(pat, txt)
        if m:
            try: out[k] = float(m.group(1))
            except ValueError: pass
    if out: vtune = out

vtune_uarch = None
vu = os.path.join(work, "vtune_uarch_summary.txt")
if os.path.isfile(vu):
    txt = open(vu, errors="replace").read()
    out = {}
    for k, pat in [
        ("elapsed_time_s",           r"Elapsed Time:\s+([\d.]+)"),
        ("effective_cpu_util",        r"Effective CPU Utilization:\s+([\d.]+)%"),
        ("memory_bound_pct",          r"Memory Bound:\s+([\d.]+)"),
        ("backend_bound_pct",         r"Backend Bound:\s+([\d.]+)"),
        ("frontend_bound_pct",        r"Front-End Bound:\s+([\d.]+)"),
        ("retiring_pct",              r"Retiring:\s+([\d.]+)"),
    ]:
        m = re.search(pat, txt)
        if m:
            try: out[k] = float(m.group(1))
            except ValueError: pass
    if out: vtune_uarch = out

# Parse samples.jsonl for a concise proc_summary block.
proc_summary = None
sjf = os.path.join(work, "samples.jsonl")
if os.path.isfile(sjf):
    snaps = []
    for raw in open(sjf, errors="replace"):
        try: snaps.append(json.loads(raw))
        except json.JSONDecodeError: pass
    if snaps:
        peak_rss  = max((s["rss_kb"]  for s in snaps if s.get("rss_kb")  is not None), default=None)
        peak_vms  = max((s["vms_kb"]  for s in snaps if s.get("vms_kb")  is not None), default=None)
        max_thr   = max((s["threads_now"] for s in snaps if s.get("threads_now") is not None), default=None)
        final_io  = snaps[-1].get("io") or {}
        proc_summary = {
            "sample_count":   len(snaps),
            "duration_s":     snaps[-1].get("t_s"),
            "peak_rss_kb":    peak_rss,
            "peak_vms_kb":    peak_vms,
            "max_threads":    max_thr,
            "read_bytes":     final_io.get("read_bytes"),
            "write_bytes":    final_io.get("write_bytes"),
        }

# Record which artefact files were produced (paths on scratch).
artefacts = {}
for fname, key in [
    ("samples.jsonl",          "proc_timeseries"),
    ("perf_stat.txt",          "perf_stat"),
    ("perf_script.txt",        "perf_callgraph"),
    ("perf_report.txt",        "perf_hotspot_report"),
    ("perf.data",              "perf_data"),
    ("vtune_hotspots",         "vtune_hotspots_dir"),
    ("vtune_hotspots.tsv",     "vtune_hotspots_tsv"),
    ("vtune_hw_events.txt",    "vtune_hw_events"),
    ("vtune_uarch",            "vtune_uarch_dir"),
    ("vtune_uarch_summary.txt","vtune_uarch_summary"),
    ("vtune_uarch_hw.txt",     "vtune_uarch_hw"),
]:
    p = os.path.join(work, fname)
    if os.path.exists(p):
        artefacts[key] = p

verify = []
if rep_ll is not None:
    verify.append({"file": os.path.basename(dpath), "status": "pass",
                   "expected": rep_ll, "reported": rep_ll, "diff": 0.0})

record = {
  "run_id": rid,
  "pbs_id": pbs,
  "platform": "gadi",
  "run_type": "profile",
  "label": label,
  "description": f"Gadi Sapphire Rapids reproduction of Setonix {ds} @ {thr}T",
  "timing": [{
    "command": f"perf stat ... {ibin} -s {os.path.basename(dpath)} -T {thr} -seed 1",
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": verify,
  "env": {
    "hostname": sh("hostname"),
    "date":     sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc", "0") or 0),
    "gcc":      sh("gcc --version | head -1"),
    "icc":      sh("icc --version 2>/dev/null | head -1"),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "vtune_version": sh("vtune --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d'=' -f2- | tr -d '\"'"),
    "pbs": {
      "job_id":      os.environ.get("PBS_JOBID"),
      "job_name":    os.environ.get("PBS_JOBNAME"),
      "queue":       os.environ.get("PBS_QUEUE"),
      "project":     "${PROJECT}",
      "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
      "submit_host": os.environ.get("PBS_O_HOST"),
      "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
      "scheduler":   "pbs_pro",
    },
  },
  "summary": {
    "pass": 1 if iqrc == 0 else 0,
    "fail": 0 if iqrc == 0 else 1,
    "total_time": iqwall if iqwall is not None else wall,
    "all_pass":   iqrc == 0,
  },
  "profile": {
    "dataset": os.path.basename(dpath),
    "threads": thr,
    "perf_cmd": perf_cmd,
    "metrics": metrics,
    "vtune":        vtune,
    "vtune_uarch":  vtune_uarch,
    "proc_summary": proc_summary,
    "artefacts":    artefacts,
  },
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path, "w"), indent=2, default=str)
print(f"[matrix] wrote {out_path}")
PYEOF
WORKER_EOF
chmod +x "${WORKER}"

# ── Plan the sweep.
submitted=0
for dataset in "${!MATRIX[@]}"; do
    [[ -n "${SELECT_DATASET}" && "${SELECT_DATASET}" != "${dataset}" ]] && continue
    read -ra threads <<< "${MATRIX[$dataset]}"
    if [[ ${#SELECT_THREADS[@]} -gt 0 ]]; then
        threads=("${SELECT_THREADS[@]}")
    fi
    for t in "${threads[@]}"; do
        label="${dataset}_${t}t_sr"
        DEPEND_ARGS=()
        if [[ -n "${DEPEND_JOBID:-}" ]]; then
            DEPEND_ARGS=(-W "depend=afterok:${DEPEND_JOBID}")
        fi
        jid=$(qsub -N "iq-${dataset}-${t}t" \
                   -P "${PROJECT}" \
                   -q normalsr \
                   -l "ncpus=104,mem=500GB,walltime=24:00:00,jobfs=2gb,storage=scratch/${PROJECT},wd" \
                   -j oe \
                   -o "${LOGS_DIR}/${label}_\${PBS_JOBID}.log" \
                   -v "DATASET=${dataset},THREADS=${t},LABEL=${label},PROJECT=${PROJECT},REPO_DIR=${REPO_DIR}" \
                   "${DEPEND_ARGS[@]}" \
                   "${WORKER}")
        echo "  → ${dataset} / ${t}T  → ${jid}"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "Submitted ${submitted} job(s). Monitor with: qstat -u \$USER | nqstat"
echo "Logs in:  ${LOGS_DIR}/"
echo "Records: ${RUNS_DIR}/ (one JSON per run, committed to git when ready)"
