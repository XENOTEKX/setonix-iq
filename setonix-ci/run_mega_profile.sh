#!/bin/bash
# run_mega_profile.sh — Enhanced profiling of IQ-TREE on Setonix.
#
# Designed for SLURM. One job per (dataset × thread-count). Captures CPU
# counters (Zen 3 core-level), software-polled RSS/NUMA/IO/per-thread
# timeseries, perf record call stacks, and a rich environment snapshot.
#
# Inputs (env vars, all overridable on the sbatch line):
#   DATASET                — basename under ${BENCHMARKS}, e.g. xlarge_mf.fa
#                            (default: mega_dna.fa for backward-compat)
#   THREADS                — IQ-TREE -T argument (default: 128)
#   PERF_RECORD_MAX_S      — cap on the perf-record second pass (default: 1800)
#   SKIP_PERF_RECORD       — set to 1 to skip pass 5 entirely (auto-on for THREADS=1)
#   SHA256_LOCKFILE        — defaults to ../benchmarks/sha256sums.txt;
#                            if the dataset is listed there, its hash MUST match
#                            or the job aborts (correctness gate against the
#                            2026-04-25 non-canonical-file regression)
#
# Usage (sbatch):
#   sbatch --export=ALL,DATASET=xlarge_mf.fa,THREADS=64 run_mega_profile.sh
#
# Or use submit_mega_batch.sh.
#
#SBATCH --job-name=iqtree-mega
#SBATCH --account=pawsey1351
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=230G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/mega_%x_%j.out
#SBATCH --error=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/mega_%x_%j.err

# 2026-04-30 (methodology audit, round 2): added --exclusive,
# --hint=nomultithread (SMT off so 128 logical = 128 physical cores), and
# OMP_PROC_BIND/OMP_PLACES env below.  This brings the Setonix submission
# to functional parity with the Gadi PBS Pro full-node cpuset.  The label
# suffix flipped from "" (now treated as SMT-on legacy) to "_smtoff_pin"
# so the new corpus does not collide with the old _baseline_smton.json
# series.

set -euo pipefail

THREADS="${THREADS:-${1:-128}}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/pawsey1351/asamuel/iqtree3}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling}"
PROFILE_ROOT="${PROFILE_ROOT:-${PROJECT_DIR}/setonix-ci/profiles}"

# Dataset selection — basename only (no path); resolved against ${BENCHMARKS}.
# Default kept as mega_dna.fa for backward-compat with the original script name.
DATASET_NAME="${DATASET:-mega_dna.fa}"
DATASET="${BENCHMARKS}/${DATASET_NAME}"

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
# Label: stem of the dataset filename + thread count + binding tag
# (e.g. xlarge_mf_64t_smtoff_pin).  See 2026-04-30 changelog entry.
DATASET_STEM="${DATASET_NAME%.fa}"
LABEL_SUFFIX="${LABEL_SUFFIX:-smtoff_pin}"
LABEL="${DATASET_STEM}_${THREADS}t_${LABEL_SUFFIX}"
WORK_DIR="${PROFILE_ROOT}/${LABEL}_${RUN_ID}"

# OpenMP pinning — matches Gadi PBS cpuset behaviour and stops Linux from
# migrating threads across the two NUMA sockets at every barrier.
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# libgomp wait policy — Pawsey libgomp is unconditional spin which is what
# starves cores at high T.  PASSIVE flips it to a yield-after-spin so threads
# release their core when blocked at a barrier.  This is the closest match to
# Gadi libiomp5's default.
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=10000

# ─────────────────────────────────────────────────────────────────────────────
# 0. Pre-flight: dataset must exist AND its sha256 must match the canonical
#    lockfile (if listed). This is the correctness gate that prevents a
#    repeat of the 2026-04-25 non-canonical-file rerun.
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -s "${DATASET}" ]]; then
    echo "ERROR: dataset ${DATASET} not found or empty." >&2
    echo "       Run setonix-ci/generate_datasets.sh first." >&2
    exit 2
fi

# NOTE: Do NOT use BASH_SOURCE[0] / SCRIPT_DIR for path resolution when
# submitted via sbatch — SLURM copies the script to a temp path before
# execution, so a SCRIPT_DIR-relative path resolves into the SLURM daemon
# directory.  Use PROJECT_DIR (hardcoded default) instead.
SHA256_LOCKFILE="${SHA256_LOCKFILE:-${PROJECT_DIR}/benchmarks/sha256sums.txt}"
if [[ -s "${SHA256_LOCKFILE}" ]]; then
    expected="$(awk -v f="${DATASET_NAME}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}")"
    if [[ -n "${expected}" ]]; then
        actual="$(sha256sum "${DATASET}" | awk '{print $1}')"
        if [[ "${actual}" != "${expected}" ]]; then
            echo "ERROR: sha256 mismatch for ${DATASET_NAME}" >&2
            echo "       expected: ${expected}" >&2
            echo "       actual:   ${actual}"   >&2
            echo "       Refusing to run on a non-canonical alignment." >&2
            echo "       Regenerate via setonix-ci/generate_datasets.sh." >&2
            exit 3
        fi
        echo "[preflight] ${DATASET_NAME} sha256 OK (canonical)."
    else
        echo "[preflight] ${DATASET_NAME} not in lockfile — proceeding without hash gate."
    fi
else
    echo "[preflight] WARNING: no sha256 lockfile at ${SHA256_LOCKFILE}; skipping hash gate."
fi

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE profiling (Setonix)"
echo "║  Dataset:   ${DATASET_NAME}"
echo "║  Threads:   ${THREADS}"
echo "║  Run ID:    ${RUN_ID}"
echo "║  Work dir:  ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment snapshot (env.json)
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
  "python":   sh("python3 --version"),
  "iqtree_binary":  "${BUILD_DIR}/iqtree3",
  "iqtree_version": sh("${BUILD_DIR}/iqtree3 --version 2>&1 | head -1"),
  "date":     sh("date -Iseconds"),
  "dataset": {
    "path": dataset,
    "file": os.path.basename(dataset),
    "size_bytes": os.path.getsize(dataset) if os.path.isfile(dataset) else None,
    "sha256": sha256(dataset),
  },
  "slurm": {
    "job_id":       os.environ.get("SLURM_JOB_ID"),
    "job_name":     os.environ.get("SLURM_JOB_NAME"),
    "partition":    os.environ.get("SLURM_JOB_PARTITION"),
    "account":      os.environ.get("SLURM_JOB_ACCOUNT"),
    "nodelist":     os.environ.get("SLURM_JOB_NODELIST"),
    "num_nodes":    os.environ.get("SLURM_JOB_NUM_NODES"),
    "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "ntasks":       os.environ.get("SLURM_NTASKS"),
    "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    "submit_host":  os.environ.get("SLURM_SUBMIT_HOST"),
    "submit_dir":   os.environ.get("SLURM_SUBMIT_DIR"),
  },
}
print(json.dumps(env, indent=2))
PYEOF

echo "  → ${ENV_JSON}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Background sampler: RSS, NUMA, IO, per-thread  (every 10s → JSONL)
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
    # Last line "Total" contains per-node MB totals; columns are nodes.
    nodes = None
    total = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Per-node"):
            continue
        if line.startswith("Node "):
            # header e.g. "Node 0   Node 1   Node 2 ...   Total"
            parts = line.split()
            nodes = [p for p in parts if p.isdigit()]
        if line.startswith("Total"):
            vals = line.split()[1:]
            try:
                total = [float(v) for v in vals]
            except ValueError:
                total = None
    if not nodes or not total:
        return None
    # vals has one more column for Total
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
            # fields 14=utime 15=stime 18=priority 19=nice
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
# 3. Comprehensive perf stat pass — events + AMD -M metrics
# ─────────────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting perf stat + IQ-TREE (pass 1: counters)..."
PERF_STAT_TXT="${WORK_DIR}/perf_stat.txt"

# Kernel events + AMD Zen 3 raw events (process-scope only; L3 uncore is
# admin-locked on Setonix so skipped). Multiplexed — ~35–40% measurement time
# each, which is plenty given the multi-hour runtime.
PERF_EVENTS="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses,\
stalled-cycles-frontend,stalled-cycles-backend,\
task-clock,page-faults,context-switches,cpu-migrations,\
ex_ret_ops,ex_ret_brn_misp,\
ls_l1_d_tlb_miss.all,bp_l1_tlb_miss_l2_tlb_hit,bp_l1_tlb_miss_l2_tlb_miss,\
ls_tablewalker.dside,ls_tablewalker.iside,\
ls_dispatch.ld_dispatch,ls_dispatch.store_dispatch"

IQTREE_LOG="${WORK_DIR}/iqtree_run.log"
START_EPOCH=$(date +%s)

# Detect srun availability so the script still works in interactive (salloc)
# sessions where srun would re-allocate.  Inside an sbatch script `srun` is
# the supported launcher and takes care of CPU binding via --cpu-bind=cores.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    # --mem=0 tells srun to use the full node allocation memory without
    # re-specifying a step-level limit.  Without it, SLURM can fail with
    # "SLURM_MEM_PER_CPU and SLURM_MEM_PER_NODE are mutually exclusive"
    # when the partition has a default mem-per-cpu and the job also sets --mem.
    SRUN=( srun --cpus-per-task="${THREADS}" --cpu-bind=cores --hint=nomultithread --mem=0 )
else
    SRUN=( )
fi
NUMACTL=( )
if command -v numactl >/dev/null 2>&1; then
    NUMACTL=( numactl --localalloc )
fi

# Launch IQ-TREE under perf stat in background so the sampler can attach.
# 2026-04-30: dropped `-mset GTR,HKY,K80` so ModelFinder runs the full
# default search — matches Gadi (see 2026-04-26 audit).
perf stat -e "${PERF_EVENTS}" \
    -o "${PERF_STAT_TXT}" \
    "${SRUN[@]}" "${NUMACTL[@]}" \
    "${BUILD_DIR}/iqtree3" -s "${DATASET}" -T "${THREADS}" -seed 1 \
    --prefix "${WORK_DIR}/iqtree_run" > "${IQTREE_LOG}" 2>&1 &
IQTREE_PID=$!

# perf stat wraps iqtree, so the inner PID we want to sample is the perf child.
# Walk the children to find the iqtree3 PID.
sleep 5
INNER_PID=$(pgrep -P "${IQTREE_PID}" -f iqtree3 | head -1 || true)
if [[ -z "${INNER_PID:-}" ]]; then
    INNER_PID="${IQTREE_PID}"
fi
echo "  → perf pid=${IQTREE_PID} inner iqtree pid=${INNER_PID}"

# Start the sampler (10s interval)
python3 "${SAMPLER_PY}" "${INNER_PID}" "${SAMPLE_JSONL}" 10 &
SAMPLER_PID=$!

# Wait for IQ-TREE
wait "${IQTREE_PID}" || IQTREE_RC=$?
IQTREE_RC="${IQTREE_RC:-0}"
END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
WALL_TIME="$(( ELAPSED/3600 ))h$(( (ELAPSED%3600)/60 ))m$(( ELAPSED%60 ))s"

# Stop sampler
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] IQ-TREE finished rc=${IQTREE_RC} wall=${WALL_TIME}"
echo ""
echo "=== perf stat output ==="
cat "${PERF_STAT_TXT}" || true
echo "========================"

# ─────────────────────────────────────────────────────────────────────────────
# 4. AMD -M metrics pass  (short re-run: first ~5 min only, for proportions)
#    We can't re-run the whole thing, so we capture metrics via a short
#    sub-workload using the same binary (iqtree3 --help) just to record what
#    the metric values look like on this system. These are supplementary.
# ─────────────────────────────────────────────────────────────────────────────
# Skipped — the counter events above already include cache + branch data.

# ─────────────────────────────────────────────────────────────────────────────
# 5. perf record for hotspots + folded stacks (skipped on 1T runs by default —
#    single-thread hotspot data is uninteresting and the second IQ-TREE pass
#    dominates SU spend on 1T jobs).
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${SKIP_PERF_RECORD:-0}" != "1" && "${THREADS}" -eq 1 ]]; then
    echo "[$(date +%H:%M:%S)] Skipping perf record on 1T job (auto; set SKIP_PERF_RECORD=0 force-on with caution)."
    SKIP_PERF_RECORD=1
fi

if [[ "${IQTREE_RC}" -eq 0 && "${SKIP_PERF_RECORD:-0}" != "1" ]]; then
    echo "[$(date +%H:%M:%S)] Starting perf record (call-graph fp)..."
    PERF_DATA="${WORK_DIR}/perf.data"
    # Bound the perf-record re-run so it cannot consume the remaining SLURM
    # wall-time and starve step 6 (profile_meta.json). At 99 Hz, 30 min ≈
    # 178 k stacks — ample for hotspot ranking. (Was 7200s; cut to 1800s
    # 2026-04-25 to halve rerun SU spend.)
    PERF_RECORD_MAX_S="${PERF_RECORD_MAX_S:-1800}"
    # --call-graph fp uses frame pointers (built in via -fno-omit-frame-pointer)
    # which is ~5–10× cheaper than dwarf unwinding and matches how the binary
    # was compiled.
    timeout --preserve-status "${PERF_RECORD_MAX_S}" \
        perf record --call-graph fp -F 99 -o "${PERF_DATA}" \
        "${SRUN[@]}" "${NUMACTL[@]}" \
        "${BUILD_DIR}/iqtree3" -s "${DATASET}" -T "${THREADS}" -seed 1 \
        --prefix "${WORK_DIR}/iqtree_flame" > "${WORK_DIR}/iqtree_flame.log" 2>&1 || true

    if [[ -s "${PERF_DATA}" ]]; then
        echo "[$(date +%H:%M:%S)] Generating hotspots.txt + perf_folded.txt..."
        perf report -i "${PERF_DATA}" --stdio --no-children -n --percent-limit 0.5 \
            2>/dev/null | head -120 > "${WORK_DIR}/hotspots.txt" || true

        # pipefail-safe: either side of the pipe may fail without aborting the run
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
python3 <<PYEOF > "${WORK_DIR}/profile_meta.json"
import json, os, re

work = "${WORK_DIR}"

# Load env
env = {}
try:
    env = json.load(open(os.path.join(work, "env.json")))
except Exception: pass

# Parse perf_stat.txt
def parse_perf(path):
    metrics = {}
    perf_cmd = None
    if not os.path.isfile(path): return metrics, perf_cmd
    for line in open(path, errors="replace"):
        line = line.rstrip()
        m = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m: perf_cmd = m.group(1); continue
        # "    123,456      event-name   ..."
        m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w\.\-:/]+)", line)
        if m:
            raw, name = m.group(1), m.group(2)
            if raw.startswith("<"): continue
            try: metrics[name] = int(raw.replace(",", ""))
            except ValueError: pass
    return metrics, perf_cmd

metrics, perf_cmd = parse_perf(os.path.join(work, "perf_stat.txt"))

# Derived rates
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
derived = {
    "IPC": round(ins / cyc, 4) if (cyc and ins) else None,
    "cache-miss-rate":       rate(getany("cache-misses"), getany("cache-references")),
    "branch-miss-rate":      rate(getany("branch-misses"), getany("branch-instructions")),
    "L1-dcache-miss-rate":   rate(getany("L1-dcache-load-misses"), getany("L1-dcache-loads")),
    "dTLB-miss-rate":        rate(getany("dTLB-load-misses"), getany("dTLB-loads")),
    "iTLB-miss-rate":        rate(getany("iTLB-load-misses"), getany("iTLB-loads")),
    "frontend-stall-rate":   rate(getany("stalled-cycles-frontend"), cyc),
    "backend-stall-rate":    rate(getany("stalled-cycles-backend"), cyc),
    "amd-branch-mispred-rate": rate(getany("ex_ret_brn_misp"), getany("ex_ret_ops")),
    "amd-l1-dtlb-miss-rate":   rate(getany("ls_l1_d_tlb_miss.all"), getany("ls_dispatch.ld_dispatch")),
    "amd-l2-tlb-miss-rate":    rate(getany("bp_l1_tlb_miss_l2_tlb_miss"),
                                     (getany("bp_l1_tlb_miss_l2_tlb_hit") or 0) + (getany("bp_l1_tlb_miss_l2_tlb_miss") or 0)),
}
metrics_all = {**metrics, **{k: v for k, v in derived.items() if v is not None}}

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

# Per-thread from final /proc snapshot (we only have periodic counts here)
# The sampler stored thread_count per tick; detailed per-thread is only
# useful for live debugging — skipped in meta JSON.

# Parse iqtree log for wall time + memory
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
echo "║  Mega profiling complete"
echo "║  Threads:  ${THREADS}"
echo "║  Wall:     ${WALL_TIME}"
echo "║  Exit rc:  ${IQTREE_RC}"
echo "╚══════════════════════════════════════════════════════════════╝"
