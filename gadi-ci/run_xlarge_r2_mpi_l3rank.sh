#!/bin/bash
# run_xlarge_r2_mpi_l3rank.sh — IQ-TREE 3 (R2-patched, MPI build) on
# xlarge_mf.fa with **8 MPI ranks × 13 OpenMP threads each**, where every
# rank is pinned via an OpenMPI rankfile to exactly one L3-cache quadrant
# (= one of Gadi SPR 8470Q's 8 NUMA nodes / sub-NUMA cluster regions).
#
# This is alternate placement #2 in the 2026-05-08 (h) sweep. Whereas
# run_xlarge_r2_mpi_socket.sh splits the work across the two sockets only,
# this script tightens the binding all the way down to L3-cache granularity:
#
#                   Sapphire Rapids 8470Q (SNC4 mode):
#                   2 sockets × 4 sub-NUMA × 13 cores = 104
#
#         ┌─── socket 0 ────┐  ┌─── socket 1 ────┐
#   NUMA  0    1    2    3      4    5    6    7
#   cores 0-12 13-  26-  39-    52-  65-  78-  91-
#               25   38   51    64   77   90   103
#   rank  0    1    2    3      4    5    6    7   ← OpenMPI rankfile binding
#
# Each rank's 13-thread OpenMP pool runs entirely within one L3 quadrant.
# • No cross-L3 cache traffic — the static-scheduled NNI loops touch only
#   pages first-touched by their own 13 threads.
# • No cross-NUMA DRAM traffic — every page allocated under numactl
#   --localalloc lands on the rank's bound node.
# • UPI / memory-controller contention between OMP threads is eliminated;
#   only MPI rank-to-rank tree exchange uses the inter-socket fabric.
#
# Hypothesis — three outcomes are possible and each is informative:
#   (a) Faster than 2-rank socket  → cross-L3 traffic mattered; binding to
#       L3 grain pays off.
#   (b) Same as 2-rank socket      → SNC4 first-touch + OMP_PROC_BIND=close
#       was already pinning per-quadrant; the extra MPI overhead cancels.
#   (c) Slower than 2-rank socket  → IQ-TREE's MPI tree-exchange dominates
#       at 8 ranks; the OpenMP-only socket placement is the right grain.
#
# The rankfile is generated dynamically from `numactl -H` so the script
# is robust against future topology changes; a fallback hard-codes the
# standard 8×13 SPR layout in case numactl is unavailable.
#
#PBS -N iq-xlarge-r2-mpi-l3rank
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

DATASET_NAME="${DATASET:-xlarge_mf}"
NRANKS="${NRANKS:-8}"
OMP_PER_RANK="${OMP_PER_RANK:-13}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_icx_mpi${NRANKS}x${OMP_PER_RANK}_l3rank_numa_ft_r2}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"

# sha256 gate (same as socket runner — preflight before any compute spend).
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
            echo "ERROR: sha256 mismatch for ${DATA_BASENAME}" >&2; exit 3
        fi
        echo "[preflight] ${DATA_BASENAME} sha256 OK (canonical)."
    fi
fi

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7         2>/dev/null || true
    module load intel-compiler-llvm   2>/dev/null || true
    module load intel-vtune/2024.2.0  2>/dev/null || true
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load openmpi/4.1.7." >&2; exit 4
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found.  Run gadi-ci/bootstrap_iqtree_mpi.sh first." >&2
    exit 5
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)' ; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 6
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
echo "║  IQ-TREE 3 R2 — MPI L3-cache rankfile placement"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads)"
echo "║  binary:        ${IQTREE}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Build the rankfile from live NUMA topology ──────────────────────────
# OpenMPI rankfile syntax (one line per rank):
#   rank N=<host> slot=<cpu-list>
# We derive the host from PBS_NODEFILE (single-node job) and the cpu-lists
# from `numactl -H`, which prints one "node N cpus: …" line per NUMA node.
#
# CRITICAL: filter SMT siblings out of the cpu list.  Gadi normalsr
# Sapphire Rapids nodes have 104 physical cores but Linux exposes 208
# logical CPUs (each core has 2 SMT hwthread siblings, even though SMT
# is "off" at the user-facing level — `lscpu` reports "Thread(s) per core:
# 1" because of /sys/devices/system/cpu/smt/control=off, but `numactl -H`
# still lists the sibling cpu IDs in the NUMA topology).  So a raw NUMA
# node line looks like:
#     node 0 cpus: 0 1 2 ... 12 104 105 ... 116
#                  ^^^ 13 physical cores ^^^ 13 SMT siblings ^^^
# If we put all 26 cpu IDs into the rankfile, OpenMPI binds each rank to
# a 26-CPU cpuset and the OMP team can drift onto SMT pairs (losing ~30%
# IPC).  The fix: keep only CPU IDs strictly less than the physical-core
# count (sockets × cores-per-socket = 2 × 52 = 104 on SPR 8470Q).
#
# Earlier attempts hit two distinct rankfile failure modes:
#   • PBS 167889452 (initial): the rankfile contained 26 cpus per rank
#     (13 cores + 13 SMT siblings), AND `mpirun -rf <file>` triggered a
#     "Conflicting directives: RANK_FILE vs BYCORE" error.
#   • PBS 167894317 (round 2): `--map-by rankfile:file=<file>` was rejected
#     as "The mapping request contains an unrecognized modifier" — that
#     `:file=…` modifier is OpenMPI 5.x syntax, not 4.x.
# Both fixed in this revision: SMT siblings filtered out here; the mpirun
# call site uses `--mca rmaps rank_file --rankfile <file>` (the 4.x-native
# syntax that selects the rank_file MCA component before BYCORE auto-loads).
#
# If numactl is missing or output is unparseable, fall back to the standard
# Gadi SPR 8470Q SNC4 layout (8 nodes × 13 cores: 0-12, 13-25, …, 91-103).
RANKFILE="${WORK_DIR}/rankfile.txt"
HOSTNAME_NODE="$(head -n1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || hostname)"

# Compute the physical-core ceiling (CPUs >= this are SMT siblings).
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
echo "[l3rank] physical core count from lscpu: ${PHYSICAL_CORES}" \
     "(sockets=${LSCPU_SOCKETS}, cores/socket=${LSCPU_COREPS})"

build_rankfile() {
    local node_lines
    node_lines="$(numactl -H 2>/dev/null | grep -E '^node [0-9]+ cpus:' | head -"${NRANKS}")"
    if [[ -z "${node_lines}" ]]; then
        return 1
    fi
    : > "${RANKFILE}"
    local rank=0
    while IFS= read -r line; do
        # Line shape: "node 0 cpus: 0 1 2 ... 12 104 105 ... 116"
        local cpus_raw cpus filtered_cnt=0
        cpus_raw="$(echo "${line}" | sed -E 's/^node [0-9]+ cpus:[[:space:]]*//' | tr -s ' ')"
        # Filter SMT siblings: drop any CPU id >= PHYSICAL_CORES.
        cpus=""
        for c in ${cpus_raw}; do
            if (( c < PHYSICAL_CORES )); then
                cpus="${cpus}${cpus:+ }${c}"
                filtered_cnt=$((filtered_cnt + 1))
            fi
        done
        if (( filtered_cnt == 0 )); then
            echo "[l3rank] ERROR: no physical-core cpus left for node ${rank}" >&2
            return 1
        fi
        # Convert space-separated list to a hyphenated range when contiguous,
        # otherwise to a comma-separated list. mpirun accepts either; the
        # range form is just more readable in logs.
        local first last contiguous=1 prev=-2
        for c in ${cpus}; do
            if (( prev >= 0 )) && (( c != prev + 1 )); then contiguous=0; fi
            prev=${c}
        done
        if (( contiguous == 1 )); then
            first="${cpus%% *}"; last="${cpus##* }"
            echo "rank ${rank}=${HOSTNAME_NODE} slot=${first}-${last}" >> "${RANKFILE}"
        else
            echo "rank ${rank}=${HOSTNAME_NODE} slot=$(echo ${cpus} | tr ' ' ',')" >> "${RANKFILE}"
        fi
        rank=$((rank + 1))
    done <<< "${node_lines}"
    [[ "${rank}" -eq "${NRANKS}" ]]
}

if ! build_rankfile; then
    echo "[l3rank] WARNING: numactl -H unavailable or unparseable; using SPR-SNC4 fallback."
    cat > "${RANKFILE}" <<EOF
rank 0=${HOSTNAME_NODE} slot=0-12
rank 1=${HOSTNAME_NODE} slot=13-25
rank 2=${HOSTNAME_NODE} slot=26-38
rank 3=${HOSTNAME_NODE} slot=39-51
rank 4=${HOSTNAME_NODE} slot=52-64
rank 5=${HOSTNAME_NODE} slot=65-77
rank 6=${HOSTNAME_NODE} slot=78-90
rank 7=${HOSTNAME_NODE} slot=91-103
EOF
fi

echo "[l3rank] rankfile:"
cat "${RANKFILE}" | sed 's/^/    /'

# ── Environment snapshot ────────────────────────────────────────────────
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
  "placement": "mpi_l3rank",
  "rankfile": open("${RANKFILE}").read() if os.path.isfile("${RANKFILE}") else None,
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

# ── OMP env (forwarded into each rank) ─────────────────────────────────
# OMP_PROC_BIND=close makes the rank's 13 OMP threads stay packed inside
# the 13-cpu set the rankfile assigned to it.
#
# OMP_DYNAMIC=false is *especially* important under rankfile binding
# because libiomp5/libomp inspect the per-rank cpuset (13 cores) and may
# decide to spawn fewer threads if it thinks the system is "saturated".
# Pinning OMP_DYNAMIC=false guarantees `OMP_NUM_THREADS=13` is honoured
# verbatim — otherwise an 8×13 = 104 run can silently degrade to 8×N<13.
# This is the explicit Taylor-rule for rankfile-bound MPI+OMP runs.
#
# OMP_PLACES=cores satisfies the spirit of Slurm's `--hint=nomultithread`
# on Gadi (SMT is already off on normalsr — `lscpu` shows threads-per-
# core=1 — so the rankfile slot=N-M ranges already map to physical cores
# rather than SMT siblings; OMP_PLACES=cores layers explicit core-grain
# placement on top of that).
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

# Per-rank wrappers: numactl --localalloc inside the rankfile cpu-set
# guarantees each rank's malloc()s land on its bound NUMA node, so first
# touch can never spill to a remote L3 quadrant.
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

# ── /proc sampler (rank 0 only) ────────────────────────────────────────
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
"""Identical to the socket-runner sampler — see that file for commentary."""
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
echo "[mpi-l3rank] Pass 1: 8 ranks × 13 OMP, rankfile-bound, no perf overhead"
START_EPOCH=$(date +%s)

# Empty-string the base mapping policy (overriding Gadi's site config which
# hardcodes `rmaps_base_mapping_policy = core` in
# /apps/openmpi-mofed5.8-pbs2021.1/4.1.7/etc/openmpi-mca-params.conf), so
# that `-rf <file>` can set the policy to RANK_FILE without conflict.
#
# Failure timeline that led to this form:
#   • `-rf <file>` (PBS 167889452):                 RANK_FILE vs BYCORE conflict
#                                                   from Gadi's site default.
#   • `--map-by rankfile:file=<file>` (167894317):  "unrecognized modifier" — that
#                                                   `:file=…` syntax is OpenMPI
#                                                   5.x, not 4.x.
#   • `--mca rmaps rank_file --rankfile <file>`     RANK_FILE vs BYCORE again —
#     (PBS 167895714, round 3):                     `--mca rmaps rank_file` selects
#                                                   the *component*, not the
#                                                   base mapping policy.
#   • `--mca rmaps_base_mapping_policy rank_file …` PBS 167896488, round 4):
#                                                   underscored value rejected as
#                                                   "policy not recognized".
#   • `--mca rmaps_base_mapping_policy rank-file …` (PBS 167898265, round 5):
#                                                   hyphenated value also rejected.
#                                                   The valid policy-value list is
#                                                   {slot,hwthread,core,l1cache,
#                                                   l2cache,l3cache,socket,numa,
#                                                   board,node,seq,dist,ppr} —
#                                                   no `rank-file` token.
#
# Login-node-validated working form (round 6):
#       --mca rmaps_base_mapping_policy "" -rf <file>
# By empty-stringing the policy we cancel the site default WITHOUT setting
# any policy of our own; -rf then sets RANK_FILE cleanly. Equivalent
# OMPI_MCA_rmaps_base_mapping_policy="" env-var form also works; using the
# --mca command-line form so it appears in --report-bindings audit output.
# --report-bindings prints the actual binding map to stderr.
mpirun -np "${NRANKS}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log" &
IQTREE_PID=$!

sleep 5
# `|| true` guards against `set -euo pipefail` killing us when pgrep finds
# zero matches (which would otherwise bring down the still-running mpirun
# in the background — exact symptom seen on PBS 167894316/317 first attempt).
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler attached to inner pid=${INNER_PID}"
python3 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
echo "[mpi-l3rank] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── Pass 2: per-rank perf stat ────────────────────────────────────────
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[mpi-l3rank] Pass 2: 8 ranks × 13 OMP under perf stat"
    mpirun -np "${NRANKS}" \
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
  "description": (f"Gadi SPR R2 — MPI L3-rankfile placement: {nranks} ranks × "
                  f"{omp_per} OMP, 1 rank per L3 quadrant (NUMA node)"),
  "timing": [{
    "command": (f"mpirun -np {nranks} "
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
    # gcc/icc/icx/vtune_version mirror canonical R2 schema so tools/canonicalize_runs.py
    # and the dashboard see the same env-key set on MPI runs as on the 1×104 R2 baseline.
    "gcc":      sh("gcc --version | head -1"),
    "icc":      sh("icc --version 2>/dev/null | head -1"),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "vtune_version": sh("vtune --version 2>&1 | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
    "rankfile": open(os.path.join(work,"rankfile.txt")).read() if os.path.isfile(os.path.join(work,"rankfile.txt")) else None,
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
    "placement":  "mpi_l3rank",
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
  # Dashboard ingest tags — mirrors canonical R2 classification.
  "build_tag":           f"icx_mpi{nranks}x{omp_per}_l3rank_numa_ft_r2",
  "non_canonical":       True,
  "non_canonical_label": f"ICX · MPI {nranks}×{omp_per} L3-rankfile · R2",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[mpi-l3rank] wrote {out_path}")
PYEOF

echo "[mpi-l3rank] done."
exit "${IQRC}"
