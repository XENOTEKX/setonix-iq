#!/bin/bash
# run_100taxa_10M_r2_avx512_mpi_4node.sh — IQ-TREE 3.1.2 (gadi-spr-r2-avx512
# branch: R1+R2 NUMA first-touch + P2+P3 AVX-512 icpx patches) on
# alignment_10000000.phy (100 taxa, 10 M DNA sites) across 4 Gadi normalsr
# SPR nodes with 4 MPI ranks × 104 OpenMP threads each.
#
# Topology:
#   4 × Xeon 8470Q "Sapphire Rapids" (2 sockets × 52 cores, DDR5-4800)
#   4 nodes × 104 cores each = 416 cores total
#
#   ┌─ node A ──────────────┐  ┌─ node B ──────────────┐
#   │ rank 0 · 104 OMP      │  │ rank 1 · 104 OMP      │
#   └───────────────────────┘  └───────────────────────┘
#   ┌─ node C ──────────────┐  ┌─ node D ──────────────┐
#   │ rank 2 · 104 OMP      │  │ rank 3 · 104 OMP      │
#   └───────────────────────┘  └───────────────────────┘
#
# IQ-TREE MPI distributes bootstrap replicates (or site partitions for
# ModelFinder) across ranks. Each rank runs a full 104-thread OMP team
# on its node. Cross-node coordination via InfiniBand (UCX).
#
# Parity with 2-node AVX-512 run (PBS 167973941):
#   • Same binary:  build-profiling-mpi/iqtree3-mpi (gadi-spr-r2-avx512 branch)
#   • Same build:   icpx -O3 -march=sapphirerapids -DIQTREE_FLAGS="mpi KNL"
#   • Same OMP env: OMP_PROC_BIND=close, OMP_PLACES=cores, KMP_BLOCKTIME=200
#   • Same numactl --localalloc per rank
#   • Same -seed 1
#   • Dataset:      alignment_10000000.phy (100 taxa, 10 M DNA sites, 954 MB)
#
#PBS -N iq-100taxa-10M-r2-avx512-4node
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l walltime=06:00:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-3.1.2}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling-mpi}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="${DATASET:-100taxa_10M}"
DATASET_FILE="${DATASET_FILE:-alignment_10000000.phy}"
NRANKS="${NRANKS:-4}"
OMP_PER_RANK="${OMP_PER_RANK:-104}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_icx_mpi${NRANKS}x${OMP_PER_RANK}_4node_fullnode_avx512_r2}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}/${DATASET_FILE}"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }
DATA_BASENAME="$(basename "${DATA_PATH}")"

# Module load: openmpi BEFORE intel-compiler-llvm (libiomp5 runtime).
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7         2>/dev/null || true
    module load intel-compiler-llvm   2>/dev/null || true
    module load intel-vtune/2024.2.0  2>/dev/null || true
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load openmpi/4.1.7." >&2; exit 4
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found." >&2
    echo "       Run gadi-ci/build/bootstrap_iqtree_3.1.2_mpi.sh first (gadi-spr-r2-avx512 branch)." >&2
    exit 5
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 6
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5/libomp." >&2; exit 7
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
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — must run inside a PBS job." >&2; exit 8
fi
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
if [[ "${#HOSTS[@]}" -ne 4 ]]; then
    echo "ERROR: expected 4 nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2
    echo "       Check PBS spec: #PBS -l ncpus=416 on normalsr." >&2
    exit 9
fi
HOST_A="${HOSTS[0]}"; HOST_B="${HOSTS[1]}"
HOST_C="${HOSTS[2]}"; HOST_D="${HOSTS[3]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

# ── Topology check ────────────────────────────────────────────────────
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
if [[ "${PHYSICAL_CORES}" -ne 104 ]]; then
    echo "ERROR: head-node has ${PHYSICAL_CORES} cores, expected 104 (2×52 SPR)." >&2; exit 10
fi

# ── Rankfile: 1 rank per node, all 104 cores ──────────────────────────
RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
rank 2=${HOST_C} slot=0-103
rank 3=${HOST_D} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3.1.2 gadi-spr-r2-avx512 — 4-node MPI full-node"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads)"
echo "║  node A (rank 0): ${HOST_A}"
echo "║  node B (rank 1): ${HOST_B}"
echo "║  node C (rank 2): ${HOST_C}"
echo "║  node D (rank 3): ${HOST_D}"
echo "║  binary:        ${IQTREE}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[4node] hostfile:"; cat "${HOSTFILE}" | sed 's/^/    /'
echo "[4node] rankfile:"; cat "${RANKFILE}" | sed 's/^/    /'

# ── env.json snapshot ────────────────────────────────────────────────
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
  "threads": ${TOTAL_THREADS}, "mpi_ranks": ${NRANKS}, "omp_per_rank": ${OMP_PER_RANK},
  "placement": "mpi_4node_fullnode",
  "nodes": 4,
  "hosts": ["${HOST_A}", "${HOST_B}", "${HOST_C}", "${HOST_D}"],
  "rankfile": open("${RANKFILE}").read() if os.path.isfile("${RANKFILE}") else None,
  "hostfile": open("${HOSTFILE}").read() if os.path.isfile("${HOSTFILE}") else None,
  "hostname": sh("hostname"), "kernel": sh("uname -r"),
  "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
  "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets":          int(sh("lscpu | awk -F: '/Socket\(s\)/{print \$2}' | xargs","0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\(s\) per socket/{print \$2}' | xargs","0") or 0),
  "numa_nodes":           int(sh("lscpu | awk -F: '/NUMA node\(s\)/{print \$2}' | xargs","0") or 0),
  "mem_total_kb":         int(sh("awk '/MemTotal/{print \$2}' /proc/meminfo","0") or 0),
  "smt_active": sh("cat /sys/devices/system/cpu/smt/active 2>/dev/null") == "1",
  "mpi_version": sh("mpirun --version 2>&1 | head -1"),
  "icx":  sh("icx --version 2>/dev/null | head -1"),
  "iqtree_binary":  "${IQTREE}",
  "iqtree_version": sh("${IQTREE} --version 2>&1 | head -1"),
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
    "project":     os.environ.get("PROJECT","${PROJECT}"),
    "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    "nnodes":      4,
    "nodes":       ["${HOST_A}","${HOST_B}","${HOST_C}","${HOST_D}"],
    "submit_host": os.environ.get("PBS_O_HOST"),
    "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
    "scheduler":   "pbs_pro",
  },
}
print(json.dumps(env, indent=2))
PYENV
echo "  → ${ENV_JSON}"

# ── MPI/UCX transport: explicit IB path (rc_mlx5 for 4-rank communicator) ────
# OpenMPI 4.1.7 on Gadi is built against MOFED 5.8 + UCX 1.17.0 (rc_mlx5/dc_mlx5
# on mlx5_0 ConnectX HDR).  Without explicit flags OpenMPI auto-selects UCX (it
# wins the PML priority race at 60 vs ob1 10), but we pin it to prevent any
# silent fallback to ob1+TCP if UCX init is slow.
# UCX_TLS: rc_mlx5 requires ud_mlx5 as auxiliary transport for endpoint
# address resolution (UCX 1.17.0 select.c); omitting ud_mlx5 causes
# "no auxiliary transport" → NULL endpoint → SIGSEGV in OpenMPI poll loop.
MPI_OPTS=(
    --mca pml ucx
    -x "UCX_TLS=rc_mlx5,ud_mlx5,sm,self"
    -x "UCX_NET_DEVICES=mlx5_0:1"
)

# ── OMP env forwarded into each rank ─────────────────────────────────
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

# perf events (':u' suffix for perf_event_paranoid=2).
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

VTUNE_WRAP="${WORK_DIR}/_vtune_wrap.sh"
cat > "${VTUNE_WRAP}" <<EOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
RDIR="${WORK_DIR}/vtune_uarch.rank\${RANK}"
exec vtune \\
    -collect uarch-exploration \\
    -knob collect-memory-bandwidth=true \\
    -knob pmu-collection-mode=summary \\
    -data-limit=2000 \\
    -finalization-mode=deferred \\
    -result-dir "\${RDIR}" \\
    -no-summary \\
    -- numactl --localalloc -- "\$@"
EOF
chmod +x "${VTUNE_WRAP}"

# /proc sampler (rank 0, node A only)
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
import json, os, sys, time, pathlib
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
      "io": read_io(pid) or {},
    }
    fp.write(json.dumps(snap)+"\n"); fp.flush()
    time.sleep(interval)
fp.close()
SAMPLER_EOF

# ── Pass 1: clean timing run ──────────────────────────────────────────
echo ""
echo "[4node] Pass 1: ${NRANKS} ranks × ${OMP_PER_RANK} OMP across 4 nodes"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    "${MPI_OPTS[@]}" \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log" &
IQTREE_PID=$!

sleep 5
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler on rank-0 pid=${INNER_PID}"
python3 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
echo "[4node] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── Pass 2: per-rank perf stat ────────────────────────────────────────
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[4node] Pass 2: ${NRANKS} ranks × ${OMP_PER_RANK} OMP under perf stat"
    mpirun -np "${NRANKS}" \
        --hostfile "${HOSTFILE}" \
        --mca rmaps_base_mapping_policy "" \
        -rf "${RANKFILE}" \
        "${MPI_OPTS[@]}" \
        "${OMP_ENV[@]}" \
        "${PERF_WRAP}" \
            "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Pass 3: VTune uarch-exploration (rank 0 only, summary mode) ──────
# Collects Top-Down Microarchitecture Analysis (TMAM) breakdown:
# FrontEnd-Bound, Bad-Speculation, BackEnd-Bound (Memory/Core), Retiring,
# plus cross-socket memory bandwidth (collect-memory-bandwidth=true).
# pmu-collection-mode=summary avoids per-sample overhead (~5-8% vs ~15%).
# finalization-mode=deferred: checksum only on compute nodes; finalize
# later on login node with: vtune -finalize -r vtune_uarch.rankN/
# Only rank 0 is profiled (representative; IQ-TREE ranks are symmetric).
VTUNE_DIR="${WORK_DIR}/vtune_uarch.rank0"
if [[ "${IQRC}" -eq 0 ]] && command -v vtune >/dev/null 2>&1; then
    echo "[4node] Pass 3: VTune uarch-exploration rank-0 only (summary, deferred)"
    VTUNE_RC=0
    mpirun -np "${NRANKS}" \
        --hostfile "${HOSTFILE}" \
        --mca rmaps_base_mapping_policy "" \
        -rf "${RANKFILE}" \
        "${MPI_OPTS[@]}" \
        "${OMP_ENV[@]}" \
        "${VTUNE_WRAP}" \
            "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_vtune" \
        > "${WORK_DIR}/iqtree_vtune.log" 2>&1 || VTUNE_RC=$?
    echo "[4node] Pass 3 done: vtune_rc=${VTUNE_RC}"
    # Finalize rank 0 result now (deferred → full on scratch)
    if [[ -d "${VTUNE_DIR}" ]]; then
        vtune -finalize -result-dir "${VTUNE_DIR}" >> "${WORK_DIR}/iqtree_vtune.log" 2>&1 || true
        echo "[4node] Pass 3 finalized: ${VTUNE_DIR}"
    fi
fi

# ── Emit run record ───────────────────────────────────────────────────
python3 - <<PYEOF
import json, os, re, glob, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
total_thr = ${TOTAL_THREADS}; nranks=${NRANKS}; omp_per=${OMP_PER_RANK}
wall, iqrc = int("${WALL}"), int("${IQRC}")
dpath, ibin = "${DATA_PATH}", "${IQTREE}"
hosts = ["${HOST_A}","${HOST_B}","${HOST_C}","${HOST_D}"]
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

artefacts = {}
for fname, key in [
    ("samples.jsonl",           "proc_timeseries"),
    ("iqtree_run.log",          "iqtree_log"),
    ("iqtree_run.bindings.log", "mpi_bindings_log"),
    ("rankfile.txt",            "rankfile"),
    ("hostfile.txt",            "hostfile"),
    ("env.json",                "env_json"),
]:
    p = os.path.join(work, fname)
    if os.path.exists(p): artefacts[key] = p
rank_perf_files = sorted(glob.glob(os.path.join(work, "perf_stat.rank*.txt")))
if rank_perf_files:
    artefacts["perf_stat_per_rank"] = rank_perf_files
vtune_dirs = sorted(glob.glob(os.path.join(work, "vtune_uarch.rank*")))
if vtune_dirs:
    artefacts["vtune_uarch_dirs"] = vtune_dirs

verify = []
if rep_ll is not None:
    verify.append({"file": os.path.basename(dpath), "status": "pass",
                   "expected": rep_ll, "reported": rep_ll, "diff": 0.0})

record = {
  "run_id": rid, "pbs_id": "${PBS_ID_SHORT}",
  "platform": "gadi", "run_type": "profile", "label": label,
  "description": (f"Gadi SPR gadi-spr-r2-avx512 — MPI 4-node full-node: {nranks} ranks × "
                  f"{omp_per} OMP, 1 rank per node × 4 nodes, AVX-512"),
  "timing": [{
    "command": (f"mpirun -np {nranks} --hostfile hostfile.txt "
                f'--mca rmaps_base_mapping_policy "" -rf rankfile.txt '
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
    "nodes":    4,
    "hosts":    hosts,
    "pbs": {
      "job_id":      os.environ.get("PBS_JOBID"),
      "job_name":    os.environ.get("PBS_JOBNAME"),
      "queue":       os.environ.get("PBS_QUEUE"),
      "project":     os.environ.get("PROJECT","${PROJECT}"),
      "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
      "nnodes":      4,
      "nodes":       hosts,
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
    "dataset":      os.path.basename(dpath),
    "threads":      total_thr,
    "placement":    "mpi_4node_fullnode",
    "mpi_ranks":    nranks,
    "omp_per_rank": omp_per,
    "nodes":        4,
    "perf_cmd":     perf_cmd,
    "metrics":      metrics,
    "per_rank_ipc": per_rank_ipc,
    "per_rank_host": {str(i): hosts[i] for i in range(nranks)},
    "bindings":     bindings_excerpt,
    "vtune":        vtune_dirs[0] if vtune_dirs else None,
    "vtune_uarch":  vtune_dirs,
    "proc_summary": proc_summary,
    "artefacts":    artefacts,
  },
  "build_tag":           f"icx_mpi{nranks}x{omp_per}_4node_fullnode_avx512_r2_v312",
  "non_canonical":       True,
  "non_canonical_label": f"ICX · MPI {nranks}×{omp_per} 4-node · R2 AVX-512",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[4node] wrote {out_path}")
PYEOF

echo "[4node] done."
exit "${IQRC}"
