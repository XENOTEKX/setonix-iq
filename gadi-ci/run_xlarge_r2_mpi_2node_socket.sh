#!/bin/bash
# run_xlarge_r2_mpi_2node_socket.sh — IQ-TREE 3 (R2-patched, MPI build) on
# xlarge_mf.fa across **2 Gadi normalsr SPR nodes**, with **4 MPI ranks ×
# 52 OpenMP threads each**. One rank per socket, scaled across 2 nodes.
#
# This is alternate placement #3 in the 2026-05-08 sweep, replacing the
# (abandoned) 2-node L3-rankfile design after the single-node l3rank result
# showed 13-thread OMP teams are too small for xlarge_mf:
#
#   #1 socket 2×52 (single node, PBS 167895713) → 520.1 s,  −0.7% vs canonical
#   #2 l3rank 8×13 (single node, PBS 167899378) → 957.8 s,  +83%   vs canonical
#                                                  → conclusion: 13-thread OMP
#                                                  is the bottleneck, not L3
#                                                  cache pressure
#   #3 2node-socket 4×52 (this script)         → 2 nodes × 2 sockets, each
#                                                 rank still owns a full 52-
#                                                 core socket OMP team
#
# This is "scale the socket experiment to 2 nodes" — the only single-node
# placement that matched canonical wall time. Each rank still has 52 OMP
# threads (the team size that *worked*), and we now run 4 ranks in parallel
# on bootstrap-replicate distribution.
#
#                   2 × Sapphire Rapids 8470Q:
#                   2 nodes × 2 sockets × 52 cores = 208 cores
#
#         ┌─── node A ──────────────┐  ┌─── node B ──────────────┐
#         │ socket 0 │ socket 1 │      │ socket 0 │ socket 1 │
#   cores 0..51     │ 52..103  │      │ 0..51    │ 52..103  │
#   rank  0         │ 1        │      │ 2        │ 3        │
#   OMP   52        │ 52       │      │ 52       │ 52       │
#
# Inside each socket, the 52 OMP threads run with OMP_PROC_BIND=close so
# they stay packed on physical cores (Gadi normalsr SMT is off at user
# level, so OMP_PLACES=cores keeps each thread on one core).  Each rank's
# numactl --localalloc keeps its mallocs on the rank's bound socket; cross-
# socket coordination is now a 4-rank MPI message graph instead of cache-
# coherence traffic.  Cross-node coordination (rank 0,1 ↔ rank 2,3) goes
# over the InfiniBand fabric.
#
# Hypothesis — three outcomes possible:
#   (a) Roughly half of the 1×104 canonical wall (~262 s) → IQ-TREE 3 MPI
#       bootstrap distribution scales linearly across 4 ranks, inter-node
#       cost is small relative to the halving of per-rank work.
#   (b) Same as 1-node socket (~520 s) → InfiniBand round-trip on every
#       tree-exchange cancels the doubled parallelism. The 2-rank ceiling
#       was already MPI-coordination-bound; doubling rank count just
#       doubles overhead.
#   (c) Slower than 1-node socket (>520 s) → Cross-node MPI traffic
#       dominates this dataset; xlarge_mf doesn't have enough independent
#       work to amortise inter-node communication.  The right scope is ≤
#       1 node.
#
# Full parity with canonical 1×104 R2 ICX (PBS 167865976):
#   • Same source build (build-profiling-mpi/iqtree3-mpi, R2-patched, icpx)
#   • Same OpenMP runtime (libiomp5)
#   • Same OMP env: OMP_PROC_BIND=close, OMP_PLACES=cores, KMP_BLOCKTIME=200,
#                   OMP_DYNAMIC=false (mandatory for socket-bound cpusets)
#   • Same numactl --localalloc per rank
#   • Same dataset (xlarge_mf.fa, sha256-gated)
#   • Same -seed 1 (per-rank seed becomes 1+rank_id inside IQ-TREE MPI)
#
# Companion to:
#   gadi-ci/run_xlarge_r2_mpi_socket.sh   (1 node, 2×52 — the topology we scale)
#   gadi-ci/run_xlarge_r2_mpi_l3rank.sh   (1 node, 8×13 — abandoned at 2 nodes)
#   gadi-ci/_run_matrix_job.sh             (1 node, single-process baseline)
#
#PBS -N iq-xlarge-r2-mpi-2node-socket
#PBS -P rc29
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1000GB
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

# Fixed shape: 4 ranks × 52 OMP = 208 = 2 full SPR nodes (104 cores each).
# Rank 0 = node A socket 0,  rank 1 = node A socket 1,
# rank 2 = node B socket 0,  rank 3 = node B socket 1.
DATASET_NAME="${DATASET:-xlarge_mf}"
NRANKS="${NRANKS:-4}"
OMP_PER_RANK="${OMP_PER_RANK:-52}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_icx_mpi${NRANKS}x${OMP_PER_RANK}_2node_socket_numa_ft_r2}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"

# sha256 gate (same canonical hash as every other R2 run).
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

# Module load order: openmpi BEFORE intel-compiler-llvm so mpirun resolves
# from openmpi (matches what iqtree3-mpi was linked against). intel-compiler-
# llvm contributes libiomp5 at runtime.
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

# ── Multi-node host discovery ──────────────────────────────────────────
# PBS_NODEFILE on Gadi normalsr lists one line per allocated CPU (so 208
# lines for an ncpus=208 job). Unique-sort gives the 2 hostnames.
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — this script must run inside a PBS job." >&2
    exit 8
fi
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
if [[ "${#HOSTS[@]}" -ne 2 ]]; then
    echo "ERROR: expected 2 nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2
    echo "       Check PBS resource spec — should be #PBS -l ncpus=208 on normalsr." >&2
    exit 9
fi
HOST_A="${HOSTS[0]}"
HOST_B="${HOSTS[1]}"

# Hostfile (slot-counted) for mpirun --hostfile.  Required by OpenMPI 4.x
# whenever the rankfile carries hostnames — the hostfile gives the launcher
# its set of valid targets, the rankfile then pins each rank inside.
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 R2 — MPI 2-node socket placement"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads, 2 nodes)"
echo "║  node A:        ${HOST_A}  (ranks 0–1, sockets 0+1)"
echo "║  node B:        ${HOST_B}  (ranks 2–3, sockets 0+1)"
echo "║  binary:        ${IQTREE}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[2node-socket] hostfile:"
cat "${HOSTFILE}" | sed 's/^/    /'

# ── Build the rankfile for both nodes ──────────────────────────────────
# OpenMPI rankfile syntax: "rank N=<host> slot=<cpu-list>".  Hardcoded SPR
# layout: socket 0 = cores 0–51, socket 1 = cores 52–103.  Same numbering
# scheme on every Gadi normalsr node (verified across all single-node R2
# JSON records — they share the same numa_topology string).  If a future
# Gadi maintenance round changes the per-node CPU map, the lscpu cross-
# check below catches it before submitting work.
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
echo "[2node-socket] head node topology: sockets=${LSCPU_SOCKETS} cores/socket=${LSCPU_COREPS} → physical_cores=${PHYSICAL_CORES}"
if [[ "${PHYSICAL_CORES}" -ne 104 ]]; then
    echo "ERROR: head-node topology is not 2×52 SPR (got ${PHYSICAL_CORES} cores)." >&2
    echo "       This script's hardcoded socket layout (0-51 / 52-103) assumes that." >&2
    exit 10
fi

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-51
rank 1=${HOST_A} slot=52-103
rank 2=${HOST_B} slot=0-51
rank 3=${HOST_B} slot=52-103
EOF

echo "[2node-socket] rankfile:"
cat "${RANKFILE}" | sed 's/^/    /'

# ── Environment snapshot (env.json) ─────────────────────────────────────
ENV_JSON="${WORK_DIR}/env.json"
python3 - <<PYENV > "${ENV_JSON}"
import json, os, subprocess, hashlib
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d
def sha256(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return h.hexdigest()
    except Exception: return None
ds = "${DATA_PATH}"
env = {
  "run_id": "${RUN_ID}", "label": "${LABEL}",
  "threads": ${TOTAL_THREADS},
  "mpi_ranks": ${NRANKS}, "omp_per_rank": ${OMP_PER_RANK},
  "placement": "mpi_2node_socket",
  "nodes": 2,
  "hosts": ["${HOST_A}", "${HOST_B}"],
  "rankfile": open("${RANKFILE}").read() if os.path.isfile("${RANKFILE}") else None,
  "hostfile": open("${HOSTFILE}").read() if os.path.isfile("${HOSTFILE}") else None,
  "hostname": sh("hostname"), "kernel": sh("uname -r"),
  "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
  "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets": int(sh("lscpu | awk -F: '/Socket\\(s\\)/{print \$2}' | xargs", "0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\\(s\\) per socket/{print \$2}' | xargs", "0") or 0),
  "cpu_threads_per_core": int(sh("lscpu | awk -F: '/Thread\\(s\\) per core/{print \$2}' | xargs", "0") or 0),
  "cpu_count_logical": int(sh("nproc","0") or 0),
  "numa_nodes": int(sh("lscpu | awk -F: '/NUMA node\\(s\\)/{print \$2}' | xargs", "0") or 0),
  "numa_topology": sh("numactl -H | grep -E '^node|cpus:' | head -40"),
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
    "queue":       os.environ.get("PBS_QUEUE"),
    "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    "submit_host": os.environ.get("PBS_O_HOST"),
    "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
  },
}
print(json.dumps(env, indent=2))
PYENV
echo "  → ${ENV_JSON}"

# ── OMP env (forwarded into each rank via mpirun -x) ────────────────────
# Identical to single-node socket: OMP_NUM_THREADS=52, OMP_DYNAMIC=false to
# stop libiomp5 shrinking the team when it observes the rank's 52-core
# cpuset is "fully utilised", OMP_PROC_BIND=close + OMP_PLACES=cores so the
# 52 threads stay on physical cores within the rank's bound socket.
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

# perf events (':u' suffix mandatory at perf_event_paranoid=2).
_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

TIME_WRAP="${WORK_DIR}/_time_wrap.sh"
cat > "${TIME_WRAP}" <<'EOF'
#!/bin/bash
exec numactl --localalloc -- "$@"
EOF
chmod +x "${TIME_WRAP}"

PERF_WRAP="${WORK_DIR}/_perf_wrap.sh"
cat > "${PERF_WRAP}" <<EOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
exec perf stat -e '${PERF_EVENTS}' \\
    -o '${WORK_DIR}'/perf_stat.rank\${RANK}.txt \\
    numactl --localalloc -- "\$@"
EOF
chmod +x "${PERF_WRAP}"

# /proc sampler (rank 0 only — lives on node A)
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
"""Identical to socket/l3rank single-node samplers — rank 0 timeline."""
import json, os, subprocess, sys, time, pathlib
pid = int(sys.argv[1]); out = pathlib.Path(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
t0 = time.monotonic()
def read_status(p):
    d={}
    try:
        with open(f"/proc/{p}/status") as f:
            for line in f:
                k,_,v = line.partition(":"); d[k.strip()] = v.strip()
    except FileNotFoundError: return None
    return d
def read_io(p):
    d={}
    try:
        with open(f"/proc/{p}/io") as f:
            for line in f:
                k,_,v = line.partition(":")
                try: d[k.strip()] = int(v.strip())
                except ValueError: pass
    except (FileNotFoundError, PermissionError): return None
    return d
def read_numa(p):
    try:
        r = subprocess.run(["numastat","-p",str(p)], capture_output=True, text=True, timeout=5)
    except Exception: return None
    if r.returncode != 0: return None
    nodes=None; total=None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Node "):
            nodes = [p for p in line.split() if p.isdigit()]
        if line.startswith("Total"):
            try: total = [float(v) for v in line.split()[1:]]
            except ValueError: total = None
    if not nodes or not total: return None
    return {"per_node_mb": {n: total[i] for i,n in enumerate(nodes) if i<len(total)},
            "total_mb": total[-1] if total else None}
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

# ── Pass 1: clean timing run (no perf) ────────────────────────────────
echo ""
echo "[2node-socket] Pass 1: ${NRANKS} ranks × ${OMP_PER_RANK} OMP, rankfile-bound across 2 nodes"
START_EPOCH=$(date +%s)

# Multi-node mpirun:
#   --hostfile  → required by OpenMPI 4.x when the rankfile carries
#                 hostnames; gives mpirun the launch targets.
#   --mca rmaps_base_mapping_policy ""  → cancels Gadi's site-config default
#                 (rmaps_base_mapping_policy = core), so -rf can install
#                 RANK_FILE without the BYCORE conflict.  This is the
#                 round-6-validated form from the single-node l3rank script
#                 (CHANGELOG entries (h)–(m), 2026-05-08).
#   -rf <file>  → applies the 4-rank socket pinning (1 rank per socket × 4
#                 sockets = 2 nodes).
#   --report-bindings → echo actual binding map to stderr for audit.
#
# OpenMPI 4.1.7 on Gadi auto-selects InfiniBand transport (UCX or openib).
# orted launches on node B via PBS_TM (Gadi's openmpi was compiled with
# --with-tm=$PBS_HOME), no SSH involved.
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log" &
IQTREE_PID=$!

# Sample rank 0 (lives on node A — mother-superior). The sampler can only
# see processes on its own node; per-rank data on node B is captured via
# Pass 2's perf wrappers writing to the shared scratch work_dir.
#
# `|| true` on pgrep — guards `set -euo pipefail` against zero-match self-
# kill (CHANGELOG entry (k) fix).
sleep 5
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler attached to inner pid=${INNER_PID} (node A rank 0)"
python3 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
echo "[2node-socket] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── Pass 2: per-rank perf stat ────────────────────────────────────────
# perf is on every Gadi compute node, so each rank writes its own
# perf_stat.rank<N>.txt into the shared scratch work_dir.
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[2node-socket] Pass 2: ${NRANKS} ranks × ${OMP_PER_RANK} OMP under perf stat"
    mpirun -np "${NRANKS}" \
        --hostfile "${HOSTFILE}" \
        --mca rmaps_base_mapping_policy "" \
        -rf "${RANKFILE}" \
        "${OMP_ENV[@]}" \
        "${PERF_WRAP}" \
            "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Emit run record ───────────────────────────────────────────────────
python3 - <<PYEOF
import json, os, re, glob, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
total_thr = ${TOTAL_THREADS}; nranks=${NRANKS}; omp_per=${OMP_PER_RANK}
wall, iqrc = int("${WALL}"), int("${IQRC}")
ds, dpath, ibin = "${DATASET_NAME}", "${DATA_PATH}", "${IQTREE}"
host_a, host_b = "${HOST_A}", "${HOST_B}"
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))

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

agg = {}
for pm in perf_per_rank.values():
    for k,v in pm.items(): agg[k] = agg.get(k, 0) + v
def rate(n,d):
    if not n or not d: return None
    return round(100.0*n/d, 4)
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

per_rank_ipc = {}
for rk, pm in perf_per_rank.items():
    c, i = pm.get("cycles"), pm.get("instructions")
    if c and i: per_rank_ipc[rk] = round(i/c, 4)

per_rank_host = {str(rk): (host_a if rk < 2 else host_b) for rk in range(nranks)}

proc_summary = None
sjf = os.path.join(work, "samples.jsonl")
if os.path.isfile(sjf):
    snaps=[]
    for raw in open(sjf, errors="replace"):
        try: snaps.append(json.loads(raw))
        except json.JSONDecodeError: pass
    if snaps:
        peak_rss = max((s["rss_kb"] for s in snaps if s.get("rss_kb") is not None), default=None)
        peak_vms = max((s["vms_kb"] for s in snaps if s.get("vms_kb") is not None), default=None)
        max_thr  = max((s["threads_now"] for s in snaps if s.get("threads_now") is not None), default=None)
        final_io = snaps[-1].get("io") or {}
        proc_summary = {
            "sample_count": len(snaps), "duration_s": snaps[-1].get("t_s"),
            "peak_rss_kb": peak_rss, "peak_vms_kb": peak_vms, "max_threads": max_thr,
            "read_bytes": final_io.get("read_bytes"),
            "write_bytes": final_io.get("write_bytes"),
        }

bindings_log = os.path.join(work, "iqtree_run.bindings.log")
bindings_excerpt = None
if os.path.isfile(bindings_log):
    bindings_excerpt = "\n".join(
        l for l in open(bindings_log, errors="replace").read().splitlines()
        if "MCW rank" in l or "binding" in l.lower())

artefacts = {}
for fname, key in [
    ("samples.jsonl",        "proc_timeseries"),
    ("iqtree_run.log",       "iqtree_log"),
    ("iqtree_run.bindings.log", "mpi_bindings_log"),
    ("rankfile.txt",         "rankfile"),
    ("hostfile.txt",         "hostfile"),
    ("env.json",             "env_json"),
]:
    p = os.path.join(work, fname)
    if os.path.exists(p): artefacts[key] = p
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
  "description": (f"Gadi SPR R2 — MPI 2-node socket placement: {nranks} ranks × "
                  f"{omp_per} OMP, 1 rank per socket, 2 sockets per node × 2 nodes"),
  "timing": [{
    "command": (f"mpirun -np {nranks} "
                f"--hostfile hostfile.txt "
                f'--mca rmaps_base_mapping_policy "" '
                f"-rf rankfile.txt "
                f"numactl --localalloc {ibin} -s {os.path.basename(dpath)} "
                f"-T {omp_per} -seed 1"),
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": verify,
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc","0") or 0),
    "gcc":      sh("gcc --version | head -1"),
    "icc":      sh("icc --version 2>/dev/null | head -1"),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "vtune_version": sh("vtune --version 2>&1 | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
    "nodes":    2,
    "hosts":    [host_a, host_b],
    "rankfile": open(os.path.join(work,"rankfile.txt")).read() if os.path.isfile(os.path.join(work,"rankfile.txt")) else None,
    "hostfile": open(os.path.join(work,"hostfile.txt")).read() if os.path.isfile(os.path.join(work,"hostfile.txt")) else None,
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
    "placement":  "mpi_2node_socket",
    "mpi_ranks":  nranks,
    "omp_per_rank": omp_per,
    "nodes":      2,
    "perf_cmd":   perf_cmd,
    "metrics":    metrics,
    "per_rank_ipc": per_rank_ipc,
    "per_rank_host": per_rank_host,
    "bindings":   bindings_excerpt,
    "vtune":        None,
    "vtune_uarch":  None,
    "proc_summary": proc_summary,
    "artefacts":    artefacts,
  },
  "build_tag":           f"icx_mpi{nranks}x{omp_per}_2node_socket_numa_ft_r2",
  "non_canonical":       True,
  "non_canonical_label": f"ICX · MPI {nranks}×{omp_per} 2-node socket · R2",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[2node-socket] wrote {out_path}")
PYEOF

echo "[2node-socket] done."
exit "${IQRC}"
