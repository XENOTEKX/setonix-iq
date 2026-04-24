#!/bin/bash
# run_mega_profile.sh — Gadi/NCI port of the Setonix mega profiler.
#
# Deep profiling of IQ-TREE on mega_dna.fa with Intel hardware counters,
# Top-down Microarchitecture Analysis (TMA) events, Intel VTune hotspot
# collection, and software-polled RSS/NUMA/IO/per-thread timeseries.
#
# Target system:  Gadi (NCI) — Intel Xeon Platinum 8268 Cascade Lake,
#                 48 cores/node, 192 GB/node, PBS Professional scheduler.
#
# Usage (qsub):
#   qsub -v THREADS=48 run_mega_profile.sh
#
# Or via submit_mega_batch.sh for a sweep across thread counts.
#
#PBS -N iqtree-mega
#PBS -P rc29
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=190GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=scratch/rc29
#PBS -j oe
#PBS -o /scratch/rc29/as1708/iqtree3/gadi-ci/logs/

set -euo pipefail

THREADS="${THREADS:-${1:-48}}"
PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-as1708}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling}"
PROFILE_ROOT="${PROJECT_DIR}/gadi-ci/profiles"
DATASET="${DATASET:-${BENCHMARKS}/mega_dna.fa}"
RUN_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
# PBS_JOBID has the form "12345678.gadi-pbs" — strip the suffix for cleaner paths.
RUN_ID="${RUN_ID%%.*}"
LABEL="mega_${THREADS}t"
WORK_DIR="${PROFILE_ROOT}/${LABEL}_${RUN_ID}"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Load Intel VTune + compilers (NCI module system). Best-effort: continue if
# module environment is absent (e.g. running under "bash" outside PBS).
if command -v module >/dev/null 2>&1; then
    module load intel-vtune/2024.2.0 2>/dev/null || true
    module load intel-compiler/2024.2.1 2>/dev/null || true
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Gadi mega_dna.fa profiling (Intel Cascade Lake)"
echo "║  Threads:   ${THREADS}"
echo "║  Run ID:    ${RUN_ID}"
echo "║  Work dir:  ${WORK_DIR}"
echo "║  Project:   ${PROJECT}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment snapshot (env.json) — captures PBS context + host details
# ─────────────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Snapshotting environment..."

ENV_JSON="${WORK_DIR}/env.json"
python3 <<PYEOF > "${ENV_JSON}"
import json, os, subprocess, hashlib, pathlib

def sh(cmd, default=""):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                        stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default

def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def read_nodefile():
    nf = os.environ.get("PBS_NODEFILE")
    if nf and os.path.isfile(nf):
        try:
            with open(nf) as f:
                return sorted({ln.strip() for ln in f if ln.strip()})
        except Exception:
            return []
    return []

dataset = "${DATASET}"
env = {
  "run_id": "${RUN_ID}",
  "label": "${LABEL}",
  "threads": ${THREADS},
  "hostname": sh("hostname"),
  "kernel":   sh("uname -r"),
  "os":       sh("cat /etc/os-release | grep PRETTY_NAME= | cut -d'=' -f2- | tr -d '\"'"),
  "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets": int(sh("lscpu | awk -F: '/Socket\\(s\\)/{print \$2}' | xargs", "0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\\(s\\) per socket/{print \$2}' | xargs", "0") or 0),
  "cpu_threads_per_core": int(sh("lscpu | awk -F: '/Thread\\(s\\) per core/{print \$2}' | xargs", "0") or 0),
  "cpu_count_logical": int(sh("nproc", "0") or 0),
  "numa_nodes": int(sh("lscpu | awk -F: '/NUMA node\\(s\\)/{print \$2}' | xargs", "0") or 0),
  "smt_active": sh("cat /sys/devices/system/cpu/smt/active 2>/dev/null") == "1",
  "cpu_governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"),
  "mem_total_kb": int(sh("awk '/MemTotal/{print \$2}' /proc/meminfo", "0") or 0),
  "glibc":    sh("ldd --version | head -1 | awk '{print \$NF}'"),
  "gcc":      sh("gcc --version | head -1"),
  "icc":      sh("icc --version 2>/dev/null | head -1"),
  "icx":      sh("icx --version 2>/dev/null | head -1"),
  "python":   sh("python3 --version"),
  "vtune_version": sh("vtune --version 2>&1 | head -1"),
  "iqtree_binary":  "${BUILD_DIR}/iqtree3",
  "iqtree_version": sh("${BUILD_DIR}/iqtree3 --version 2>&1 | head -1"),
  "date":     sh("date -Iseconds"),
  "dataset": {
    "path": dataset,
    "file": os.path.basename(dataset),
    "size_bytes": os.path.getsize(dataset) if os.path.isfile(dataset) else None,
    "sha256": sha256(dataset),
  },
  "pbs": {
    "job_id":       os.environ.get("PBS_JOBID"),
    "job_name":     os.environ.get("PBS_JOBNAME"),
    "queue":        os.environ.get("PBS_QUEUE"),
    "project":      os.environ.get("PROJECT") or os.environ.get("PBS_PROJECT"),
    "ncpus":        os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    "nnodes":       os.environ.get("PBS_NNODES") or os.environ.get("PBS_NUM_NODES"),
    "mem":          os.environ.get("PBS_VMEM") or os.environ.get("PBS_MEM"),
    "nodefile":     os.environ.get("PBS_NODEFILE"),
    "nodes":        read_nodefile(),
    "submit_host":  os.environ.get("PBS_O_HOST"),
    "submit_dir":   os.environ.get("PBS_O_WORKDIR"),
    "o_queue":      os.environ.get("PBS_O_QUEUE"),
    "scheduler":    "pbs_pro",
  },
}
print(json.dumps(env, indent=2))
PYEOF

echo "  → ${ENV_JSON}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Background sampler: RSS, NUMA, IO, per-thread  (every 10s → JSONL)
#    Identical shape to the Setonix version so tools/harvest_scratch.py
#    and the dashboard front-end can ingest Gadi data unchanged.
# ─────────────────────────────────────────────────────────────────────────────
SAMPLER_PY="${WORK_DIR}/_sampler.py"
cat > "${SAMPLER_PY}" <<'PYEOF'
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
                d[k.strip()] = int(v.strip())
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
    nodes, total = None, None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Per-node"):
            continue
        if line.startswith("Node "):
            parts = line.split()
            nodes = [p for p in parts if p.isdigit()]
        if line.startswith("Total"):
            vals = line.split()[1:]
            try: total = [float(v) for v in vals]
            except ValueError: total = None
    if not nodes or not total: return None
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
            out.append({
                "tid": int(tid),
                "utime": int(fields[13]),
                "stime": int(fields[14]),
                "nice":  int(fields[18]),
            })
        except (FileNotFoundError, IndexError, ValueError):
            continue
    return out

fp = out.open("w")
while True:
    t = time.monotonic() - t0
    status = read_status(pid)
    if status is None:
        break
    io = read_io(pid) or {}
    numa = read_numa(pid)
    threads = per_thread(pid)
    snap = {
      "t_s": round(t, 2),
      "rss_kb":  int(status.get("VmRSS", "0 kB").split()[0]) if "VmRSS" in status else None,
      "peak_kb": int(status.get("VmHWM", "0 kB").split()[0]) if "VmHWM" in status else None,
      "vms_kb":  int(status.get("VmSize", "0 kB").split()[0]) if "VmSize" in status else None,
      "threads_now": int(status.get("Threads", "0")) if "Threads" in status else None,
      "voluntary_cs":   int(status.get("voluntary_ctxt_switches", "0")),
      "involuntary_cs": int(status.get("nonvoluntary_ctxt_switches", "0")),
      "io": io,
      "numa": numa,
      "thread_count": len(threads),
    }
    fp.write(json.dumps(snap) + "\n")
    fp.flush()
    time.sleep(interval)
fp.close()
PYEOF

SAMPLE_JSONL="${WORK_DIR}/samples.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# 3. perf stat pass — Intel Cascade Lake events + Top-down TMA slots
# ─────────────────────────────────────────────────────────────────────────────
# Notes on Intel event selection (Cascade Lake / Skylake-SP perf alias names):
#   • Core: cycles, instructions, branches/misses, cache-refs/misses
#   • Memory: L1-dcache-*, LLC-loads/misses, dTLB-*/iTLB-*
#   • Stalls: stalled-cycles-frontend/backend
#   • Top-down slots (TMA Level-1): topdown-{total-slots,slots-issued,
#     slots-retired,fetch-bubbles,recovery-bubbles} — aliases exist on
#     Cascade Lake for the skx_core pmu. perf will multiplex if more events
#     than counters.
#   • `perf_event_paranoid=2` on Gadi → process-scope only, no system-wide
#     uncore (LLC hit/miss via CHA), same restriction as Setonix.
echo "[$(date +%H:%M:%S)] Starting perf stat + IQ-TREE (pass 1: counters)..."
PERF_STAT_TXT="${WORK_DIR}/perf_stat.txt"

PERF_EVENTS="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses,\
stalled-cycles-frontend,stalled-cycles-backend,\
task-clock,page-faults,context-switches,cpu-migrations,\
topdown-total-slots,topdown-slots-issued,topdown-slots-retired,\
topdown-fetch-bubbles,topdown-recovery-bubbles"

IQTREE_LOG="${WORK_DIR}/iqtree_run.log"
START_EPOCH=$(date +%s)

perf stat -e "${PERF_EVENTS}" \
    -o "${PERF_STAT_TXT}" \
    "${BUILD_DIR}/iqtree3" -s "${DATASET}" -T "${THREADS}" -seed 1 \
    -mset GTR,HKY,K80 \
    --prefix "${WORK_DIR}/iqtree_run" > "${IQTREE_LOG}" 2>&1 &
IQTREE_PID=$!

sleep 5
INNER_PID=$(pgrep -P "${IQTREE_PID}" -f iqtree3 | head -1 || true)
if [[ -z "${INNER_PID:-}" ]]; then
    INNER_PID="${IQTREE_PID}"
fi
echo "  → perf pid=${IQTREE_PID} inner iqtree pid=${INNER_PID}"

python3 "${SAMPLER_PY}" "${INNER_PID}" "${SAMPLE_JSONL}" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQTREE_RC=$?
IQTREE_RC="${IQTREE_RC:-0}"
END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
WALL_TIME="$(( ELAPSED/3600 ))h$(( (ELAPSED%3600)/60 ))m$(( ELAPSED%60 ))s"

kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] IQ-TREE finished rc=${IQTREE_RC} wall=${WALL_TIME}"
echo ""
echo "=== perf stat output ==="
cat "${PERF_STAT_TXT}" || true
echo "========================"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Intel VTune pass — hotspots + performance snapshot
#    Skipped if VTune is not available. Bounded to 30 min wall by default so
#    it cannot starve step 5 or 6 of PBS wall-time.
# ─────────────────────────────────────────────────────────────────────────────
VTUNE_DIR="${WORK_DIR}/vtune_hotspots"
VTUNE_SUMMARY="${WORK_DIR}/vtune_summary.txt"
VTUNE_JSON="${WORK_DIR}/vtune_hotspots.json"
VTUNE_MAX_S="${VTUNE_MAX_S:-1800}"

if command -v vtune >/dev/null 2>&1 && [[ "${IQTREE_RC}" -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] Starting VTune hotspots collection (timeout ${VTUNE_MAX_S}s)..."
    timeout --preserve-status "${VTUNE_MAX_S}" \
        vtune -collect hotspots \
              -knob sampling-mode=hw \
              -knob enable-stack-collection=true \
              -r "${VTUNE_DIR}" \
              -- "${BUILD_DIR}/iqtree3" -s "${DATASET}" -T "${THREADS}" -seed 1 \
                 -mset GTR,HKY,K80 --prefix "${WORK_DIR}/iqtree_vtune" \
          > "${WORK_DIR}/vtune_collect.log" 2>&1 || true

    if [[ -d "${VTUNE_DIR}" ]]; then
        vtune -report summary -r "${VTUNE_DIR}" \
              -format text > "${VTUNE_SUMMARY}" 2>/dev/null || true
        vtune -report hotspots -r "${VTUNE_DIR}" \
              -format csv -report-output "${WORK_DIR}/vtune_hotspots.csv" \
              -csv-delimiter=comma 2>/dev/null || true
        # CSV → JSON for harvester consumption
        python3 <<'PYEOF' > "${VTUNE_JSON}" || true
import csv, json, os, sys
csv_path = os.environ.get("WORK_DIR", ".") + "/vtune_hotspots.csv"
if not os.path.isfile(csv_path):
    print("[]"); sys.exit(0)
rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(json.dumps(rows[:200], indent=2))
PYEOF
    fi
else
    echo "[$(date +%H:%M:%S)] VTune unavailable or IQ-TREE failed — skipping VTune pass"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. perf record for hotspots + folded stacks (bounded wall-time)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${IQTREE_RC}" -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] Starting perf record (call-graph)..."
    PERF_DATA="${WORK_DIR}/perf.data"
    PERF_RECORD_MAX_S="${PERF_RECORD_MAX_S:-7200}"
    timeout --preserve-status "${PERF_RECORD_MAX_S}" \
        perf record -g -F 99 -o "${PERF_DATA}" \
        "${BUILD_DIR}/iqtree3" -s "${DATASET}" -T "${THREADS}" -seed 1 \
        -mset GTR,HKY,K80 \
        --prefix "${WORK_DIR}/iqtree_flame" > "${WORK_DIR}/iqtree_flame.log" 2>&1 || true

    if [[ -s "${PERF_DATA}" ]]; then
        echo "[$(date +%H:%M:%S)] Generating hotspots.txt + perf_folded.txt..."
        perf report -i "${PERF_DATA}" --stdio --no-children -n --percent-limit 0.5 \
            2>/dev/null | head -120 > "${WORK_DIR}/hotspots.txt" || true

        set +o pipefail
        perf script -i "${PERF_DATA}" 2>/dev/null | python3 - <<'PYEOF' > "${WORK_DIR}/perf_folded.txt" || true
import sys, collections
stacks = collections.Counter()
current = []
for line in sys.stdin:
    line = line.rstrip()
    if not line:
        if current:
            stacks[";".join(reversed(current))] += 1
            current = []
    elif line.startswith("\t"):
        parts = line.strip().split()
        if parts:
            func = parts[0] if len(parts) == 1 else parts[1]
            func = func.split("+")[0]
            current.append(func)
for stack, count in stacks.most_common():
    print(f"{stack} {count}")
PYEOF
        set -o pipefail
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Emit structured profile_meta.json (machine-readable, for harvester)
# ─────────────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Writing profile_meta.json..."
export WORK_DIR
python3 <<PYEOF > "${WORK_DIR}/profile_meta.json"
import json, os, re

work = "${WORK_DIR}"

env = {}
try:
    env = json.load(open(os.path.join(work, "env.json")))
except Exception: pass

def parse_perf(path):
    metrics = {}
    perf_cmd = None
    if not os.path.isfile(path): return metrics, perf_cmd
    for line in open(path, errors="replace"):
        line = line.rstrip()
        m = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m: perf_cmd = m.group(1); continue
        m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w\.\-:/]+)", line)
        if m:
            raw, name = m.group(1), m.group(2)
            if raw.startswith("<"): continue
            try: metrics[name] = int(raw.replace(",", ""))
            except ValueError: pass
    return metrics, perf_cmd

metrics, perf_cmd = parse_perf(os.path.join(work, "perf_stat.txt"))

def rate(n, d, pct=True, prec=4):
    if not n or not d or d == 0: return None
    r = n / d
    return round(r * (100.0 if pct else 1.0), prec)

def getany(*keys):
    for k in keys:
        v = metrics.get(k)
        if v is not None: return v
    return None

cyc = getany("cycles")
ins = getany("instructions")
# Intel TMA Level-1 (formula from Intel SDM):
#   Retiring           = slots-retired / total-slots
#   Bad Speculation    = (slots-issued - slots-retired + recovery-bubbles) / total-slots
#   Frontend Bound     = fetch-bubbles / total-slots
#   Backend Bound      = 1 - (above three)
slots       = getany("topdown-total-slots")
issued      = getany("topdown-slots-issued")
retired     = getany("topdown-slots-retired")
fetch_bub   = getany("topdown-fetch-bubbles")
recov_bub   = getany("topdown-recovery-bubbles")
tma = {}
if slots:
    if retired is not None:
        tma["intel-tma-retiring-pct"]      = round(100.0 * retired / slots, 4)
    if issued is not None and retired is not None and recov_bub is not None:
        tma["intel-tma-bad-spec-pct"]      = round(100.0 * (issued - retired + recov_bub) / slots, 4)
    if fetch_bub is not None:
        tma["intel-tma-frontend-bound-pct"] = round(100.0 * fetch_bub / slots, 4)
    if all(k in tma for k in ("intel-tma-retiring-pct","intel-tma-bad-spec-pct","intel-tma-frontend-bound-pct")):
        tma["intel-tma-backend-bound-pct"] = round(
            max(0.0, 100.0 - tma["intel-tma-retiring-pct"]
                           - tma["intel-tma-bad-spec-pct"]
                           - tma["intel-tma-frontend-bound-pct"]), 4)

derived = {
    "IPC": round(ins / cyc, 4) if (cyc and ins) else None,
    "cache-miss-rate":       rate(getany("cache-misses"), getany("cache-references")),
    "branch-miss-rate":      rate(getany("branch-misses"), getany("branch-instructions")),
    "L1-dcache-miss-rate":   rate(getany("L1-dcache-load-misses"), getany("L1-dcache-loads")),
    "LLC-miss-rate":         rate(getany("LLC-load-misses"), getany("LLC-loads")),
    "dTLB-miss-rate":        rate(getany("dTLB-load-misses"), getany("dTLB-loads")),
    "iTLB-miss-rate":        rate(getany("iTLB-load-misses"), getany("iTLB-loads")),
    "frontend-stall-rate":   rate(getany("stalled-cycles-frontend"), cyc),
    "backend-stall-rate":    rate(getany("stalled-cycles-backend"), cyc),
    **tma,
}
metrics_all = {**metrics, **{k: v for k, v in derived.items() if v is not None}}

# Parse optional VTune summary text for headline figures.
vtune = {}
vt_summary = os.path.join(work, "vtune_summary.txt")
if os.path.isfile(vt_summary):
    text = open(vt_summary, errors="replace").read()
    for key, pattern in [
        ("elapsed_time_s",        r"Elapsed Time:\s+([\d.]+)"),
        ("cpu_time_s",            r"CPU Time:\s+([\d.]+)"),
        ("effective_cpu_util",    r"Effective CPU Utilization:\s+([\d.]+)%"),
        ("avg_cpu_freq_ghz",      r"Average CPU Frequency:\s+([\d.]+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            try: vtune[key] = float(m.group(1))
            except ValueError: pass

vt_hotspots = []
vt_json = os.path.join(work, "vtune_hotspots.json")
if os.path.isfile(vt_json):
    try:
        rows = json.load(open(vt_json))
        for row in rows[:50]:
            func = row.get("Function") or row.get("Source Function") or ""
            mod  = row.get("Module") or ""
            cpu  = row.get("CPU Time") or row.get("CPU Time:Self") or "0"
            try: cpu_s = float(str(cpu).replace("s","").strip())
            except ValueError: cpu_s = 0.0
            if func:
                vt_hotspots.append({"function": func, "module": mod, "cpu_time_s": cpu_s})
    except Exception: pass
if vt_hotspots:
    vtune["hotspots"] = vt_hotspots

# Sampler timeseries → summary
mem_ts, peak_rss_kb, io_first, io_last, numa_last = [], 0, None, None, None
sjf = os.path.join(work, "samples.jsonl")
if os.path.isfile(sjf):
    for line in open(sjf):
        try: rec = json.loads(line)
        except: continue
        mem_ts.append({"t_s": rec.get("t_s"),
                       "rss_kb": rec.get("rss_kb"),
                       "peak_kb": rec.get("peak_kb")})
        if rec.get("peak_kb"): peak_rss_kb = max(peak_rss_kb, rec["peak_kb"])
        if io_first is None and rec.get("io"): io_first = rec["io"]
        if rec.get("io"): io_last = rec["io"]
        if rec.get("numa"): numa_last = rec["numa"]

io_delta = None
if io_first and io_last:
    io_delta = {k: io_last.get(k, 0) - io_first.get(k, 0) for k in io_last}

wall_time = None
mem_kb = 0
iqlog = os.path.join(work, "iqtree_run.log")
if os.path.isfile(iqlog):
    for line in open(iqlog, errors="replace"):
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)\s+sec", line)
        if m: wall_time = float(m.group(1))
        m = re.search(r"memory usage:\s+([\d.]+)\s+MB", line, re.I)
        if m:
            try: mem_kb = max(mem_kb, int(float(m.group(1)) * 1024))
            except: pass

meta = {
  "run_id": env.get("run_id"),
  "label": env.get("label"),
  "threads": env.get("threads"),
  "wall_time_s": wall_time,
  "dataset": env.get("dataset"),
  "env": env,
  "profile": {
    "perf_cmd": perf_cmd,
    "metrics": metrics_all,
    "vtune": vtune if vtune else None,
    "peak_rss_kb": peak_rss_kb or None,
    "memory_timeseries": mem_ts,
    "io_totals": io_delta,
    "numa_last": numa_last,
  },
}
print(json.dumps(meta, indent=2, default=str))
PYEOF

echo "[$(date +%H:%M:%S)] Done. Artefacts in ${WORK_DIR}:"
ls -lh "${WORK_DIR}/" || true
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Gadi mega profiling complete"
echo "║  Threads:  ${THREADS}"
echo "║  Wall:     ${WALL_TIME}"
echo "║  Exit rc:  ${IQTREE_RC}"
echo "╚══════════════════════════════════════════════════════════════╝"
