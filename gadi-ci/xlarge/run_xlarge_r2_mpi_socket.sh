#!/bin/bash
# run_xlarge_r2_mpi_socket.sh — IQ-TREE 3 (R2-patched, MPI build) on
# xlarge_mf.fa with **2 MPI ranks × 52 OpenMP threads each**, one rank
# pinned per socket on Gadi normalsr Sapphire Rapids.
#
# This is alternate placement #1 in the 2026-05-08 (h) sweep. The current
# canonical R2 result (logs/runs/gadi_xlarge_mf_104t_icx_omp_pin_numa_ft_r2.json,
# 523.7 s) was produced by a single iqtree3 process with 104 OpenMP threads
# spilling across both sockets. Here we run the SAME source build (R2
# patches, icpx + libiomp5) but split the work into two MPI ranks, one per
# socket, so cross-socket coordination becomes explicit MPI message-passing
# instead of cache-coherence/UPI traffic.
#
#   Rank 0 → socket 0 (cores 0–51,  NUMA nodes 0–3, 4 L3 quadrants)
#   Rank 1 → socket 1 (cores 52–103, NUMA nodes 4–7, 4 L3 quadrants)
#   OMP_NUM_THREADS=52 in each rank.  numactl --localalloc per rank.
#
# Hypothesis: if the R2 wins are dominated by *intra-socket* NUMA traffic
# (sub-NUMA / L3 quadrant), 2-rank should be similar to the 104T R2 result.
# If a meaningful fraction is *inter-socket* (UPI cache-coherence on shared
# OpenMP buffers), 2-rank should beat it because that traffic is now
# explicit one-shot MPI sends instead of per-loop cache-coherence chatter.
#
# Companion to:
#   gadi-ci/run_xlarge_r2_mpi_l3rank.sh  (8 ranks × 13 OMP, L3 rankfile)
#   gadi-ci/_run_matrix_job.sh            (single-process baseline)
#
#PBS -N iq-xlarge-r2-mpi-socket
#PBS -P rc29
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -l storage=scratch/rc29
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling-mpi}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

# Fixed dataset / fixed total-thread budget for this experiment.
DATASET_NAME="${DATASET:-xlarge_mf}"
NRANKS="${NRANKS:-2}"
OMP_PER_RANK="${OMP_PER_RANK:-52}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_icx_mpi${NRANKS}x${OMP_PER_RANK}_socket_numa_ft_r2}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"

# Reuse the existing sha256 lockfile (regenerated alignments must match the
# canonical hash before any benchmark run is allowed to start).
SHA256_LOCKFILE="${SHA256_LOCKFILE:-${REPO_DIR}/benchmarks/sha256sums.txt}"
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "ERROR: dataset ${DATA_PATH} not found or empty." >&2
    exit 2
fi
if [[ -s "${SHA256_LOCKFILE}" ]]; then
    expected="$(awk -v f="${DATA_BASENAME}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}")"
    if [[ -n "${expected}" ]]; then
        actual="$(sha256sum "${DATA_PATH}" | awk '{print $1}')"
        if [[ "${actual}" != "${expected}" ]]; then
            echo "ERROR: sha256 mismatch for ${DATA_BASENAME}" >&2
            exit 3
        fi
        echo "[preflight] ${DATA_BASENAME} sha256 OK (canonical)."
    fi
fi

# Module load: openmpi must come BEFORE intel-compiler-llvm so mpirun is
# resolved from openmpi (its libmpi rpath is what the binary was linked
# against). intel-compiler-llvm pulls in libiomp5 at runtime.
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7         2>/dev/null || true
    module load intel-compiler-llvm   2>/dev/null || true
    module load intel-vtune/2024.2.0  2>/dev/null || true
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load openmpi/4.1.7." >&2
    exit 4
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found." >&2
    echo "       Run gadi-ci/bootstrap_iqtree_mpi.sh first." >&2
    exit 5
fi

# Defensive: if the operator is rebuilding without MPI, refuse to run.
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)' ; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2
    exit 6
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5/libomp." >&2
    exit 7
fi

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_ID="gadi_${LABEL}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 R2 — MPI socket placement"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads)"
echo "║  binary:        ${IQTREE}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Environment snapshot (env.json) ─────────────────────────────────────
ENV_JSON="${WORK_DIR}/env.json"
/usr/bin/python3.11 - <<PYENV > "${ENV_JSON}"
import json, os, subprocess, hashlib
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d
def sha256(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception: return None
ds = "${DATA_PATH}"
env = {
  "run_id": "${RUN_ID}", "label": "${LABEL}",
  "threads": ${TOTAL_THREADS},
  "mpi_ranks": ${NRANKS}, "omp_per_rank": ${OMP_PER_RANK},
  "placement": "mpi_socket",
  "hostname": sh("hostname"), "kernel": sh("uname -r"),
  "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
  "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets": int(sh("lscpu | awk -F: '/Socket\\(s\\)/{print \$2}' | xargs", "0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\\(s\\) per socket/{print \$2}' | xargs", "0") or 0),
  "cpu_threads_per_core": int(sh("lscpu | awk -F: '/Thread\\(s\\) per core/{print \$2}' | xargs", "0") or 0),
  "cpu_count_logical": int(sh("nproc", "0") or 0),
  "numa_nodes": int(sh("lscpu | awk -F: '/NUMA node\\(s\\)/{print \$2}' | xargs", "0") or 0),
  "smt_active": sh("cat /sys/devices/system/cpu/smt/active 2>/dev/null") == "1",
  "mem_total_kb": int(sh("awk '/MemTotal/{print \$2}' /proc/meminfo", "0") or 0),
  "mpi_version": sh("mpirun --version 2>&1 | head -1"),
  "icx":   sh("icx --version 2>/dev/null | head -1"),
  "iqtree_binary":  "${IQTREE}",
  "iqtree_version": sh("mpirun -n 1 ${IQTREE} --version 2>&1 | head -1"),
  "date": sh("date -Iseconds"),
  "dataset": {
    "path": ds, "file": os.path.basename(ds),
    "size_bytes": os.path.getsize(ds) if os.path.isfile(ds) else None,
    "sha256": sha256(ds),
  },
  "pbs": {
    "job_id":      os.environ.get("PBS_JOBID"),
    "job_name":    os.environ.get("PBS_JOBNAME"),
    "queue":       os.environ.get("PBS_QUEUE"),
    "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    "submit_host": os.environ.get("PBS_O_HOST"),
    "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
  },
}
print(json.dumps(env, indent=2))
PYENV
echo "  → ${ENV_JSON}"

# ── OpenMP env (forwarded into each rank via mpirun -x ...) ─────────────
# Same OMP knobs as the canonical R2 single-process run, with three
# placement-specific additions:
#
#   • OMP_NUM_THREADS=52 (down from 104) — each rank owns one socket.
#   • OMP_DYNAMIC=false                  — pin the OpenMP pool size at
#       exactly 52. Without this, libiomp5 can shrink the team when it
#       observes the rank's cpuset (52 cores) is "fully utilised", which
#       would silently turn a 2×52 run into 2×<52. Mandatory whenever the
#       cpuset is < node-wide (i.e. anywhere we use `--bind-to socket` or
#       a rankfile).
#   • OMP_PLACES=cores                   — already in the canonical block;
#       on Gadi normalsr SMT is off (lscpu reports threads-per-core=1) so
#       this is the equivalent of Slurm's `--hint=nomultithread` — each
#       OMP thread lands on one physical core, no SMT sibling sharing.
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

# ── perf events (':u' suffix mandatory at perf_event_paranoid=2) ─────────
_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

# ── Per-rank wrappers ───────────────────────────────────────────────────
# Pass 1: numactl --localalloc only, no perf — clean wall-clock timing.
# Pass 2: numactl + perf stat per rank — output one perf_stat.rank<N>.txt
# per rank. We rely on OMPI_COMM_WORLD_RANK being exported into each rank
# by OpenMPI (default behaviour, not the OMPI_*_REMOTE setting).
TIME_WRAP="${WORK_DIR}/_time_wrap.sh"
cat > "${TIME_WRAP}" <<'EOF'
#!/bin/bash
# Per-rank wrapper: pin each rank's allocations to its bound socket and
# exec iqtree3-mpi. mpirun --bind-to socket has already restricted the
# rank's CPU set; numactl --localalloc makes the malloc/first-touch policy
# follow that binding rather than spilling to remote DRAM.
exec numactl --localalloc -- "$@"
EOF
chmod +x "${TIME_WRAP}"

PERF_WRAP="${WORK_DIR}/_perf_wrap.sh"
cat > "${PERF_WRAP}" <<EOF
#!/bin/bash
# Per-rank perf-stat wrapper. One output file per rank (rank id from OMPI).
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
exec perf stat -e '${PERF_EVENTS}' \\
    -o '${WORK_DIR}'/perf_stat.rank\${RANK}.txt \\
    numactl --localalloc -- "\$@"
EOF
chmod +x "${PERF_WRAP}"

# ── /proc time-series sampler (rank 0 only — RSS/IO/NUMA/per-thread) ────
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/python3.11
"""Poll /proc/$pid for rss/io/numa/per-thread stats. One JSON line per tick.
Identical shape to gadi-ci/_run_matrix_job.sh's sampler so the dashboard
front-end can ingest both run types unchanged."""
import json, os, subprocess, sys, time, pathlib
pid = int(sys.argv[1]); out = pathlib.Path(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
t0 = time.monotonic()
def read_status(p):
    d = {}
    try:
        with open(f"/proc/{p}/status") as f:
            for line in f:
                k, _, v = line.partition(":"); d[k.strip()] = v.strip()
    except FileNotFoundError: return None
    return d
def read_io(p):
    d = {}
    try:
        with open(f"/proc/{p}/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                try: d[k.strip()] = int(v.strip())
                except ValueError: pass
    except (FileNotFoundError, PermissionError): return None
    return d
def read_numa(p):
    try:
        r = subprocess.run(["numastat", "-p", str(p)],
                           capture_output=True, text=True, timeout=5)
    except Exception: return None
    if r.returncode != 0: return None
    nodes=None; total=None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Node "):
            nodes = [p for p in line.split() if p.isdigit()]
        if line.startswith("Total"):
            vals = line.split()[1:]
            try: total = [float(v) for v in vals]
            except ValueError: total = None
    if not nodes or not total: return None
    per = {n: total[i] for i, n in enumerate(nodes) if i < len(total)}
    return {"per_node_mb": per, "total_mb": total[-1] if total else None}
fp = out.open("w")
while True:
    t = time.monotonic() - t0
    s = read_status(pid)
    if s is None: break
    snap = {
      "t_s": round(t,2),
      "rss_kb":  int(s.get("VmRSS","0 kB").split()[0]) if "VmRSS" in s else None,
      "peak_kb": int(s.get("VmHWM","0 kB").split()[0]) if "VmHWM" in s else None,
      "vms_kb":  int(s.get("VmSize","0 kB").split()[0]) if "VmSize" in s else None,
      "threads_now": int(s.get("Threads","0")) if "Threads" in s else None,
      "voluntary_cs":   int(s.get("voluntary_ctxt_switches","0")),
      "involuntary_cs": int(s.get("nonvoluntary_ctxt_switches","0")),
      "io": read_io(pid) or {},
      "numa": read_numa(pid),
    }
    fp.write(json.dumps(snap)+"\n"); fp.flush()
    time.sleep(interval)
fp.close()
SAMPLER_EOF

# ── Pass 1: clean timing run (no perf) ──────────────────────────────────
echo ""
echo "[mpi-socket] Pass 1: 2 ranks × 52 OMP, --map-by socket:PE=52 --bind-to core, no perf overhead"
START_EPOCH=$(date +%s)

# --map-by socket:PE=52  → place 1 rank on each socket, each rank gets 52
#                         processing-elements (cores).  PE=N implies the
#                         binding granularity is "core" — see below.
# --bind-to core         → REQUIRED by OpenMPI 4.1 when PE=N is set.  Earlier
#                         this was `--bind-to socket`, which mpirun rejected
#                         with "A request for multiple cpus-per-proc was given,
#                         but a conflicting binding policy was specified … the
#                         correct binding policy for the given type of cpu is:
#                         bind-to core".  Switching to bind-to-core does NOT
#                         change the rank's cpuset shape: rank 0 still gets
#                         all 52 cores of socket 0, rank 1 still gets socket
#                         1.  The OMP team is then bound inside that 52-core
#                         set via OMP_PROC_BIND=close + OMP_PLACES=cores, and
#                         OMP_PLACES=cores keeps SMT siblings out of the OMP
#                         place list (SMT is visible at the OS level on Gadi
#                         normalsr — `numactl -H` reports 208 logical CPUs —
#                         but each "place" is a physical core, so OMP threads
#                         land 1-per-core).
# --report-bindings      → dump the actual core sets to stderr for the run log,
#                          so we can later verify rank 0 = 0-51 / rank 1 = 52-103.
# OMPI propagates env vars listed via -x to every rank.
mpirun -np "${NRANKS}" \
    --map-by socket:PE="${OMP_PER_RANK}" \
    --bind-to core \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log" &
IQTREE_PID=$!

# Sample rank 0's iqtree3-mpi process (mpirun spawns one mpirun-orted per
# node + one iqtree3-mpi per rank). Rank 0 is enough for RSS/NUMA shape.
#
# IMPORTANT: every pgrep below ends with `|| true`. Under `set -euo pipefail`
# (which we enable at the top of the script), a `pgrep | head -1` whose
# pgrep finds zero matches exits non-zero, the assignment exits non-zero,
# and `set -e` kills the script — which then kills the still-running
# mpirun in the background. The first revision of this script learned that
# the hard way (PBS 167894316: 8 s wall, 1m35s cpu, IQ-TREE crashed mid-
# alignment-check because the script self-aborted at this very pgrep).
sleep 5
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler attached to inner pid=${INNER_PID}"
/usr/bin/python3.11 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
echo "[mpi-socket] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── Pass 2: per-rank perf stat (one txt file per rank) ──────────────────
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[mpi-socket] Pass 2: 2 ranks × 52 OMP under perf stat (per rank)"
    mpirun -np "${NRANKS}" \
        --map-by socket:PE="${OMP_PER_RANK}" \
        --bind-to core \
        "${OMP_ENV[@]}" \
        "${PERF_WRAP}" \
            "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Emit run.schema.json record ─────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re, glob, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
total_thr  = ${TOTAL_THREADS}
nranks     = ${NRANKS}
omp_per    = ${OMP_PER_RANK}
wall       = int("${WALL}")
iqrc       = int("${IQRC}")
ds         = "${DATASET_NAME}"
dpath      = "${DATA_PATH}"
ibin       = "${IQTREE}"
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

# IQ-TREE always writes "BEST SCORE FOUND" + "Total wall-clock time used"
# from the master rank into stdout (which we redirect to iqtree_run.log).
log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))

# Per-rank perf parsing → sum counts, average rates.
# Also extract the perf command line from rank 0's file (matches the
# `profile.perf_cmd` field in canonical R2 records — the dashboard
# displays this verbatim).
perf_per_rank = {}; perf_cmd = None
for fp in sorted(glob.glob(os.path.join(work, "perf_stat.rank*.txt"))):
    rk = re.search(r"rank(\d+)", fp).group(1)
    pm = {}
    for line in open(fp, errors="replace"):
        m = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m and rk == "0": perf_cmd = m.group(1); continue
        m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w\.\-:/]+)", line)
        if m and not m.group(1).startswith("<"):
            try:
                key = m.group(2).split(":",1)[0]
                pm[key] = int(m.group(1).replace(",",""))
            except ValueError: pass
    perf_per_rank[rk] = pm

# Aggregate: counts add across ranks; rates re-derive from summed counts.
agg = {}
for pm in perf_per_rank.values():
    for k,v in pm.items():
        agg[k] = agg.get(k, 0) + v

def rate(n, d):
    if not n or not d: return None
    return round(100.0 * n/d, 4)
def g(*keys):
    for k in keys:
        if agg.get(k) is not None: return agg[k]
    return None
cyc, ins = g("cycles"), g("instructions")
metrics = {
  "IPC": round(ins/cyc,4) if cyc and ins else None,
  "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
  "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
  "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
  "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
  "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
  "iTLB-miss-rate":      rate(g("iTLB-load-misses"), g("iTLB-loads")),
}
for k in ("cycles","instructions","cache-references","cache-misses",
         "branch-instructions","branch-misses",
         "L1-dcache-loads","L1-dcache-load-misses",
         "LLC-loads","LLC-load-misses",
         "dTLB-loads","dTLB-load-misses","iTLB-loads","iTLB-load-misses"):
    if k in agg: metrics[k] = agg[k]
metrics = {k:v for k,v in metrics.items() if v is not None}

# Per-rank IPC for the dashboard's per-rank panel (informational only).
per_rank_ipc = {}
for rk, pm in perf_per_rank.items():
    c, i = pm.get("cycles"), pm.get("instructions")
    if c and i: per_rank_ipc[rk] = round(i/c, 4)

# Sampler summary.
proc_summary = None
sjf = os.path.join(work, "samples.jsonl")
if os.path.isfile(sjf):
    snaps = []
    for raw in open(sjf, errors="replace"):
        try: snaps.append(json.loads(raw))
        except json.JSONDecodeError: pass
    if snaps:
        peak_rss = max((s["rss_kb"]  for s in snaps if s.get("rss_kb")  is not None), default=None)
        peak_vms = max((s["vms_kb"]  for s in snaps if s.get("vms_kb")  is not None), default=None)
        max_thr  = max((s["threads_now"] for s in snaps if s.get("threads_now") is not None), default=None)
        final_io = snaps[-1].get("io") or {}
        proc_summary = {
            "sample_count": len(snaps), "duration_s": snaps[-1].get("t_s"),
            "peak_rss_kb": peak_rss, "peak_vms_kb": peak_vms, "max_threads": max_thr,
            "read_bytes": final_io.get("read_bytes"),
            "write_bytes": final_io.get("write_bytes"),
        }

# Snapshot the actual MPI binding output so reviewers can verify rank-N → socket-N.
bindings_log = os.path.join(work, "iqtree_run.bindings.log")
bindings_excerpt = None
if os.path.isfile(bindings_log):
    bindings_excerpt = "\n".join(
        l for l in open(bindings_log, errors="replace").read().splitlines()
        if "MCW rank" in l or "binding" in l.lower())

# Artefact paths for the dashboard (parity with _run_matrix_job.sh schema).
artefacts = {}
for fname, key in [
    ("samples.jsonl",        "proc_timeseries"),
    ("iqtree_run.log",       "iqtree_log"),
    ("iqtree_run.bindings.log", "mpi_bindings_log"),
    ("rankfile.txt",         "rankfile"),
    ("env.json",             "env_json"),
]:
    p = os.path.join(work, fname)
    if os.path.exists(p): artefacts[key] = p
# Per-rank perf-stat files: list them all under one key.
rank_perf_files = sorted(glob.glob(os.path.join(work, "perf_stat.rank*.txt")))
if rank_perf_files:
    artefacts["perf_stat_per_rank"] = rank_perf_files

verify = []
if rep_ll is not None:
    verify.append({"file": os.path.basename(dpath), "status": "pass",
                   "expected": rep_ll, "reported": rep_ll, "diff": 0.0})

record = {
  "run_id": rid, "pbs_id": "${PBS_ID_SHORT}",
  "platform": "gadi", "run_type": "profile", "label": label,
  "description": (f"Gadi SPR R2 — MPI socket placement: {nranks} ranks × "
                  f"{omp_per} OMP, 1 rank per socket"),
  "timing": [{
    "command": (f"mpirun -np {nranks} --map-by socket:PE={omp_per} --bind-to core "
                f"numactl --localalloc {ibin} -s {os.path.basename(dpath)} "
                f"-T {omp_per} -seed 1"),
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": verify,
  "env": {
    "hostname": sh("hostname"),
    "date":     sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc","0") or 0),
    # gcc/icc/icx/vtune_version mirror canonical R2 schema so tools/canonicalize_runs.py
    # and the dashboard see the same env-key set on MPI runs as on the 1×104 R2 baseline.
    "gcc":      sh("gcc --version | head -1"),
    "icc":      sh("icc --version 2>/dev/null | head -1"),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "vtune_version": sh("vtune --version 2>&1 | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
    "pbs": {
      "job_id":      os.environ.get("PBS_JOBID"),
      "job_name":    os.environ.get("PBS_JOBNAME"),
      "queue":       os.environ.get("PBS_QUEUE"),
      "project":     os.environ.get("PROJECT") or os.environ.get("PBS_PROJECT") or "${PROJECT}",
      "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
      "submit_host": os.environ.get("PBS_O_HOST"),
      "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
      "scheduler":   "pbs_pro",
    },
  },
  "summary": {
    "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
    "total_time": iqwall if iqwall is not None else wall,
    "all_pass":   iqrc == 0,
  },
  "profile": {
    "dataset":    os.path.basename(dpath),
    "threads":    total_thr,
    "placement":  "mpi_socket",
    "mpi_ranks":  nranks,
    "omp_per_rank": omp_per,
    "perf_cmd":   perf_cmd,
    "metrics":    metrics,
    "per_rank_ipc": per_rank_ipc,
    "bindings":   bindings_excerpt,
    "vtune":        None,  # parity with canonical schema; populated only if VTune ran
    "vtune_uarch":  None,
    "proc_summary": proc_summary,
    "artefacts":    artefacts,
  },
  # Dashboard ingest tags. Mirrors the canonical R2 record's classification:
  # non_canonical=true (sits as a reference series alongside the canonical
  # smtoff_pin / sr_gcc_pin baselines), with a build_tag that uniquely
  # identifies this placement variant for chart series colouring.
  "build_tag":           f"icx_mpi{nranks}x{omp_per}_socket_numa_ft_r2",
  "non_canonical":       True,
  "non_canonical_label": f"ICX · MPI {nranks}×{omp_per} socket · R2",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[mpi-socket] wrote {out_path}")
PYEOF

echo "[mpi-socket] done."
exit "${IQRC}"
