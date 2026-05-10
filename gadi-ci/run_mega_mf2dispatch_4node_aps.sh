#!/bin/bash
# run_mega_mf2dispatch_4node_aps.sh
# IQ-TREE 3.1.2 + MF2 dispatch — mega_dna.fa on 4 Gadi normalsr SPR nodes
# Full ModelFinder (968 models) with Intel APS profiling for MPI performance.
#
# Dataset: mega_dna.fa — 500 taxa × 100 K DNA sites, 48 MB
# Layout:  4 MPI ranks (1 rank/node, 1 rank/2-socket node), 104 OMP threads/rank
# Binary:  abd98764 (LPT + MF_WAITING fix, branch gadi-spr-r2-avx512)
# Profiling: Intel APS/VTune (APS_STAT_LEVEL=2) + perf stat + /proc sampler
#
# ┌─ Timing estimates (from xlarge_mf 1-node 104T: 968 models in 212s wall) ─┐
# │  xlarge_mf: 200 taxa → 397 branches; mega_dna: 500 taxa → 997 branches  │
# │  Per-model wall: 212s/968 × (997/397) ≈ 0.55s/model                     │
# │  4-rank dispatch: 242 models/rank × 0.55s ≈ 133s MF wall                │
# │  + starting tree (500 taxa) ≈ 90–180s; total ≈ 7–15 min                 │
# │  APS overhead: +10–20% → budget 30 min per pass, 3 passes = 90 min      │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Parity with previous 4-node runs (PBS 168000932, 167977883):
#   - Same 4-node layout (1 rank/node), same seed=1, same build binary path
#   - Different dataset: mega_dna.fa (500t×100K sites) instead of 10M or xlarge
#   - Full MF, no --mrate restriction (mega is fast enough to complete)
#
# Intel APS (Application Performance Snapshot):
#   module load intel-vtune/2025.8.1
#   APS wraps each MPI rank: mpirun ... aps -r aps_result -- iqtree3-mpi ...
#   Metrics captured:
#     - MPI time per rank, imbalance, message size distribution
#     - CPU utilization, vectorization, IPC
#     - Memory bandwidth (DRAM BW)
#     - Function-level hotspots
#   Reports generated post-run:
#     aps_summary.txt, aps_mpi_time.txt, aps_functions.txt,
#     aps_messages.txt, aps_counters.txt
#
# Usage:
#   qsub run_mega_mf2dispatch_4node_aps.sh
#   # Or with overrides:
#   qsub -v MRATE="G,I+G" run_mega_mf2dispatch_4node_aps.sh   # restricted models
#   qsub -v SEED=42       run_mega_mf2dispatch_4node_aps.sh   # different seed
#
#PBS -N iq-mega-mf2-4node-aps
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=800GB
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-mpi-mf2}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="${DATASET_NAME:-mega_dna}"
DATASET_FILE="${DATASET_FILE:-mega_dna.fa}"
NRANKS="${NRANKS:-4}"
OMP_PER_RANK="${OMP_PER_RANK:-104}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_mf2dispatch_aps_mpi${NRANKS}x${OMP_PER_RANK}_4node_fullnode}"

DATA_PATH="${BENCHMARKS}/${DATASET_FILE}"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }

# ── Module loading ─────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7              2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
    # Intel VTune/APS — provides the 'aps' command for MPI performance profiling
    module load intel-vtune/2025.8.1       2>/dev/null || true
fi

# ── Preflight checks ──────────────────────────────────────────────────
if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load openmpi/4.1.7." >&2; exit 4
fi
if ! command -v aps >/dev/null 2>&1; then
    echo "ERROR: 'aps' not found after module load intel-vtune/2025.8.1." >&2
    echo "       Intel Application Performance Snapshot is required for this run." >&2
    exit 3
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found or not executable." >&2
    echo "       Build with: cd build-mpi-mf2 && gmake -j4 iqtree3" >&2
    exit 5
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 6
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5 (icpx build)." >&2; exit 7
fi
# LPT fix verification (Issue 7): binary must contain the MF_WAITING clear string.
# Use grep -a (binary-as-text) for reliable cross-node detection;
# 'strings' may silently miss strings on some Gadi compute nodes.
if ! grep -qa 'cost-sorted LPT stripe' "${IQTREE}"; then
    echo "ERROR: ${IQTREE} is missing the LPT fix (Issue 7 — commit abd98764)." >&2
    echo "       Broken binary will stall: rank 0 evaluates only ~24/242 models." >&2
    echo "       Rebuild from branch gadi-spr-r2-avx512, HEAD abd98764." >&2
    exit 11
fi

echo "[preflight] binary:  ${IQTREE}"
echo "[preflight] LPT fix: CONFIRMED (cost-sorted LPT stripe + MF_WAITING cleared)"
echo "[preflight] aps:     $(aps --version 2>&1 | head -1)"
echo "[preflight] mpirun:  $(mpirun --version 2>&1 | head -1)"
echo "[preflight] dataset: ${DATA_PATH} ($(du -sh "${DATA_PATH}" | cut -f1))"

# ── OMP and KMP tuning ─────────────────────────────────────────────────
# Per user's APS config: close binding, 104 threads/rank, PASSIVE wait policy.
export OMP_NUM_THREADS="${OMP_PER_RANK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=PASSIVE
export OMP_DISPLAY_ENV=VERBOSE
export OMP_DISPLAY_AFFINITY=TRUE
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export GOMP_SPINCOUNT=10000
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

# ── APS tuning ─────────────────────────────────────────────────────────
# APS_STAT_LEVEL=2 → full MPI tracing + hardware counters + function summary.
# Level 1 = lightweight; Level 2 = standard (MPI + counters); Level 3 = deepest.
export APS_STAT_LEVEL="${APS_STAT_LEVEL:-2}"

# ── Job ID and directories ─────────────────────────────────────────────
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
APS_DIR="${WORK_DIR}/aps_result"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}" "${APS_DIR}"
cd "${WORK_DIR}"

# ── Node discovery ────────────────────────────────────────────────────
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — must run inside a PBS job." >&2; exit 8
fi
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
if [[ "${#HOSTS[@]}" -ne 4 ]]; then
    echo "ERROR: expected 4 nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2; exit 9
fi
HOST_A="${HOSTS[0]}"; HOST_B="${HOSTS[1]}"
HOST_C="${HOSTS[2]}"; HOST_D="${HOSTS[3]}"

# Hostfile (reference only — we use --map-by not hostfile+rankfile)
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

# ── Topology check ────────────────────────────────────────────────────
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
if [[ "${PHYSICAL_CORES}" -ne 104 ]]; then
    echo "ERROR: head-node has ${PHYSICAL_CORES} cores, expected 104 (2×52 SPR)." >&2; exit 10
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3.1.2 + MF2 dispatch + Intel APS — 4-node mega"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads)"
echo "║  seed:          ${SEED}"
echo "║  full_mf:       968 DNA models (no --mrate restriction)"
echo "║  node A (rank 0): ${HOST_A}"
echo "║  node B (rank 1): ${HOST_B}"
echo "║  node C (rank 2): ${HOST_C}"
echo "║  node D (rank 3): ${HOST_D}"
echo "║  binary:        ${IQTREE}"
echo "║  aps version:   $(aps --version 2>&1 | head -1)"
echo "║  APS_STAT_LEVEL: ${APS_STAT_LEVEL}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[mega-aps] hostfile (reference):"; cat "${HOSTFILE}" | sed 's/^/    /'

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
  "mf2_dispatch": True,
  "full_modelfinder": True,
  "mf2_commit": "abd98764",
  "mf2_branch": "gadi-spr-r2-avx512",
  "mf2_patches": [
    "phase1_lpt_stripe", "phase2_allreduce", "phase3_thread_budget",
    "issue5_sequential_eval", "issue6_evaluateall_always",
    "issue7_lpt_mf_waiting_fix"
  ],
  "profiling": {
    "aps_version": sh("aps --version 2>&1 | head -1"),
    "aps_stat_level": int("${APS_STAT_LEVEL}"),
    "aps_result_dir": "${APS_DIR}",
    "perf_events": "cycles,instructions,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses",
    "proc_sampler_interval_s": 10,
  },
  "mpirun_mapping": "--map-by node:PE=${OMP_PER_RANK}",
  "seed": ${SEED},
  "placement": "mf2_mpi_4node_fullnode",
  "nodes": 4,
  "hosts": ["${HOST_A}", "${HOST_B}", "${HOST_C}", "${HOST_D}"],
  "hostname": sh("hostname"), "kernel": sh("uname -r"),
  "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
  "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets":          int(sh("lscpu | awk -F: '/Socket\\(s\\)/{print \$2}' | xargs","0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\\(s\\) per socket/{print \$2}' | xargs","0") or 0),
  "numa_nodes":           int(sh("lscpu | awk -F: '/NUMA node\\(s\\)/{print \$2}' | xargs","0") or 0),
  "mem_total_kb":         int(sh("awk '/MemTotal/{print \$2}' /proc/meminfo","0") or 0),
  "smt_active": sh("cat /sys/devices/system/cpu/smt/active 2>/dev/null") == "1",
  "mpi_version": sh("mpirun --version 2>&1 | head -1"),
  "icx":  sh("icx --version 2>/dev/null | head -1"),
  "iqtree_binary":  "${IQTREE}",
  "iqtree_version": sh("${IQTREE} --version 2>&1 | head -1"),
  "date": sh("date -Iseconds"),
  "dataset": {
    "path": ds, "file": "${DATASET_FILE}",
    "taxa": 500, "sites": 100000, "seq_type": "DNA",
    "size_bytes": os.path.getsize(ds) if os.path.isfile(ds) else None,
    "sha256": sha256(ds),
    "note": "500 taxa × 100K sites, 2.5× more branches than xlarge_mf (200t×100K)",
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

# ── /proc sampler ─────────────────────────────────────────────────────
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
import json, os, sys, time, pathlib
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
fp = out.open("w")
while True:
    t = time.monotonic() - t0
    s = read_status(pid)
    if s is None: break
    snap = {
      "t_s":         round(t, 2),
      "rss_kb":      int(s.get("VmRSS",  "0 kB").split()[0]) if "VmRSS"  in s else None,
      "peak_kb":     int(s.get("VmHWM",  "0 kB").split()[0]) if "VmHWM"  in s else None,
      "vms_kb":      int(s.get("VmSize", "0 kB").split()[0]) if "VmSize" in s else None,
      "threads_now": int(s.get("Threads", "0"))               if "Threads" in s else None,
      "io":          read_io(pid) or {},
    }
    fp.write(json.dumps(snap) + "\n"); fp.flush()
    time.sleep(interval)
fp.close()
SAMPLER_EOF

# ── perf events (':u' suffix for perf_event_paranoid=2) ───────────────
_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

# ── MPI/UCX transport options ─────────────────────────────────────────
MPI_OPTS=(
    --mca pml ucx
    -x "UCX_TLS=rc_mlx5,ud_mlx5,sm,self"
    -x "UCX_NET_DEVICES=mlx5_0:1"
)

# ── OMP environment forwarding ────────────────────────────────────────
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
    -x "APS_STAT_LEVEL=${APS_STAT_LEVEL}"
)

# ════════════════════════════════════════════════════════════════
# PASS 1: Intel APS profiling (MPI performance + hardware counters)
# ════════════════════════════════════════════════════════════════
# APS wraps each MPI rank: mpirun ... aps -r <dir> -- iqtree3-mpi ...
# This captures per-rank MPI time, imbalance, message sizes, CPU utilization.
echo ""
echo "[mega-aps] Pass 1: Intel APS collection (APS_STAT_LEVEL=${APS_STAT_LEVEL})"
echo "[mega-aps] Expected: MF-MPI: rank 0/4 assigned 242/968 models (cost-sorted LPT stripe, MF_WAITING cleared)"
START_EPOCH=$(date +%s)
IQRC_APS=0

# APS per-rank wrapper: each mpirun rank executes 'aps -r <dir> -- numactl ... iqtree'
APS_WRAP="${WORK_DIR}/_aps_wrap.sh"
cat > "${APS_WRAP}" <<APSEOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
exec aps -r '${APS_DIR}' \\
    numactl --localalloc -- "\$@"
APSEOF
chmod +x "${APS_WRAP}"

mpirun -np "${NRANKS}" \
    --map-by "node:PE=${OMP_PER_RANK}" \
    --report-bindings \
    "${MPI_OPTS[@]}" \
    "${OMP_ENV[@]}" \
    "${APS_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -m MF -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    ${MRATE:+--mrate "${MRATE}"} \
                    --prefix "${WORK_DIR}/iqtree_aps" \
    > "${WORK_DIR}/iqtree_aps.log" 2> "${WORK_DIR}/iqtree_aps.bindings.log" &
IQTREE_APS_PID=$!

# Launch /proc sampler on rank-0 mpirun process
sleep 5
INNER_PID="$(pgrep -P "${IQTREE_APS_PID}" 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_APS_PID}"
echo "  → mpirun pid=${IQTREE_APS_PID}, sampler tracking pid=${INNER_PID}"
python3 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples_aps.jsonl" 10 &
SAMPLER_APS_PID=$!

wait "${IQTREE_APS_PID}" || IQRC_APS=$?
IQRC_APS="${IQRC_APS:-0}"
END_EPOCH_APS=$(date +%s)
WALL_APS=$(( END_EPOCH_APS - START_EPOCH ))
kill "${SAMPLER_APS_PID}" 2>/dev/null || true
wait "${SAMPLER_APS_PID}" 2>/dev/null || true
echo "[mega-aps] Pass 1 done: rc=${IQRC_APS} wall=${WALL_APS}s ($(( WALL_APS/60 ))m$(( WALL_APS%60 ))s)"

# ── MF-MPI dispatch diagnostics ───────────────────────────────────────
echo ""
echo "[mega-aps] MF-MPI dispatch diagnostics (Pass 1):"
grep "MF-MPI:" "${WORK_DIR}/iqtree_aps.log" 2>/dev/null | sed 's/^/    /' || \
    echo "    (no MF-MPI: lines)"
echo "[mega-aps] Best-fit model:"
grep "Best-fit model:" "${WORK_DIR}/iqtree_aps.log" 2>/dev/null | sed 's/^/    /' || \
    echo "    (not completed)"

# ════════════════════════════════════════════════════════════════
# PASS 2: perf stat (hardware counters — independent of APS)
# ════════════════════════════════════════════════════════════════
# Separate pass for clean perf counters. Uses existing .model.gz checkpoint —
# models already evaluated will be restored from checkpoint instantly,
# so Pass 2 effectively just measures the Phase 2 gather + tree search
# unless the checkpoint was wiped. We use a different --prefix for Pass 2.
echo ""
echo "[mega-aps] Pass 2: perf stat hardware counters"
START_EPOCH_PERF=$(date +%s)
IQRC_PERF=0

PERF_WRAP="${WORK_DIR}/_perf_wrap.sh"
cat > "${PERF_WRAP}" <<EOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
exec perf stat -e '${PERF_EVENTS}' \\
    -o '${WORK_DIR}'/perf_stat.rank\${RANK}.txt \\
    numactl --localalloc -- "\$@"
EOF
chmod +x "${PERF_WRAP}"

mpirun -np "${NRANKS}" \
    --map-by "node:PE=${OMP_PER_RANK}" \
    "${MPI_OPTS[@]}" \
    "${OMP_ENV[@]}" \
    "${PERF_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -m MF -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    ${MRATE:+--mrate "${MRATE}"} \
                    --prefix "${WORK_DIR}/iqtree_perf" \
    > "${WORK_DIR}/iqtree_perf.log" 2>&1 &
IQTREE_PERF_PID=$!

sleep 5
PERF_INNER="$(pgrep -P "${IQTREE_PERF_PID}" 2>/dev/null | head -1 || true)"
[[ -z "${PERF_INNER:-}" ]] && PERF_INNER="${IQTREE_PERF_PID}"
python3 "${WORK_DIR}/_sampler.py" "${PERF_INNER}" "${WORK_DIR}/samples_perf.jsonl" 10 &
SAMPLER_PERF_PID=$!

wait "${IQTREE_PERF_PID}" || IQRC_PERF=$?
IQRC_PERF="${IQRC_PERF:-0}"
END_EPOCH_PERF=$(date +%s)
WALL_PERF=$(( END_EPOCH_PERF - START_EPOCH_PERF ))
kill "${SAMPLER_PERF_PID}" 2>/dev/null || true
wait "${SAMPLER_PERF_PID}" 2>/dev/null || true
echo "[mega-aps] Pass 2 done: rc=${IQRC_PERF} wall=${WALL_PERF}s"

# ════════════════════════════════════════════════════════════════
# PASS 3: clean timing (only if passes 1 and 2 succeeded)
# ════════════════════════════════════════════════════════════════
IQRC_CLEAN=0
WALL_CLEAN=0
if [[ "${IQRC_APS}" -eq 0 && "${IQRC_PERF}" -eq 0 ]]; then
    echo ""
    echo "[mega-aps] Pass 3: clean timing (numactl only)"
    START_EPOCH_CLEAN=$(date +%s)
    mpirun -np "${NRANKS}" \
        --map-by "node:PE=${OMP_PER_RANK}" \
        "${MPI_OPTS[@]}" \
        "${OMP_ENV[@]}" \
        numactl --localalloc -- \
            "${IQTREE}" -s "${DATA_PATH}" -m MF -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        ${MRATE:+--mrate "${MRATE}"} \
                        --prefix "${WORK_DIR}/iqtree_clean" \
        > "${WORK_DIR}/iqtree_clean.log" 2>&1 || IQRC_CLEAN=$?
    WALL_CLEAN=$(( $(date +%s) - START_EPOCH_CLEAN ))
    echo "[mega-aps] Pass 3 done: rc=${IQRC_CLEAN} wall=${WALL_CLEAN}s"
fi

# ════════════════════════════════════════════════════════════════
# APS REPORT GENERATION
# ════════════════════════════════════════════════════════════════
echo ""
echo "[mega-aps] Generating Intel APS reports from ${APS_DIR}..."
APS_REPORT_DIR="${WORK_DIR}/aps_reports"
mkdir -p "${APS_REPORT_DIR}"

# Check APS result directory has data
APS_OK=false
if ls "${APS_DIR}"/*.aps 2>/dev/null | head -1 | grep -q . 2>/dev/null; then
    APS_OK=true
elif ls "${APS_DIR}" 2>/dev/null | grep -qE "\.aps$|aps-\d"; then
    APS_OK=true
fi

if [[ "${IQRC_APS}" -eq 0 ]]; then
    APS_OK=true
fi

if "${APS_OK}" || [[ -d "${APS_DIR}" ]]; then
    echo "[mega-aps] APS result dir: ${APS_DIR}"
    ls "${APS_DIR}" 2>/dev/null | sed 's/^/    /'

    # Generate all report types (capture errors gracefully)
    for report_flag in "" "-t" "-f" "-m" "--counters"; do
        case "${report_flag}" in
            "")           RNAME="aps_summary.txt"       ;;
            "-t")         RNAME="aps_mpi_time.txt"      ;;
            "-f")         RNAME="aps_functions.txt"     ;;
            "-m")         RNAME="aps_messages.txt"      ;;
            "--counters") RNAME="aps_counters.txt"      ;;
        esac
        RPATH="${APS_REPORT_DIR}/${RNAME}"
        if [[ -z "${report_flag}" ]]; then
            aps --report "${APS_DIR}" > "${RPATH}" 2>&1 && \
                echo "  ✓ ${RNAME}" || echo "  ✗ ${RNAME} (aps --report failed)"
        else
            aps --report "${APS_DIR}" ${report_flag} > "${RPATH}" 2>&1 && \
                echo "  ✓ ${RNAME}" || echo "  ✗ ${RNAME} (aps --report ${report_flag} failed)"
        fi
    done

    # Print APS summary to PBS log for immediate visibility
    echo ""
    echo "════ APS SUMMARY ════"
    cat "${APS_REPORT_DIR}/aps_summary.txt" 2>/dev/null | head -60 || echo "(no summary generated)"
    echo "════════════════════"
else
    echo "[mega-aps] WARNING: APS result directory appears empty — skipping reports."
    echo "           APS may not have collected data (binary/APS compatibility issue?)."
fi

# ════════════════════════════════════════════════════════════════
# RUN RECORD HARVESTING
# ════════════════════════════════════════════════════════════════
echo ""
echo "[mega-aps] Building run record..."
WALL="${WALL_APS}"
IQRC="${IQRC_APS}"

python3 - <<PYEOF
import json, os, re, glob, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
total_thr = ${TOTAL_THREADS}; nranks=${NRANKS}; omp_per=${OMP_PER_RANK}
wall_aps, wall_perf, wall_clean = int("${WALL_APS}"), int("${WALL_PERF}"), int("${WALL_CLEAN}")
iqrc_aps, iqrc_perf, iqrc_clean = int("${IQRC_APS}"), int("${IQRC_PERF}"), int("${IQRC_CLEAN}")
dpath, ibin = "${DATA_PATH}", "${IQTREE}"
hosts = ["${HOST_A}","${HOST_B}","${HOST_C}","${HOST_D}"]
aps_dir, aps_report_dir = "${APS_DIR}", "${WORK_DIR}/aps_reports"
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

def parse_iq_log(log_path):
    """Parse IQ-TREE log for key metrics."""
    result = {}
    if not os.path.isfile(log_path): return result
    for line in open(log_path, errors="replace"):
        if m := re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line):
            result["best_ll"] = float(m.group(1))
        if m := re.search(r"Total wall-clock time used:\s+([\d.]+)", line):
            result["total_wall_s"] = float(m.group(1))
        if m := re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line):
            result["mf_wall_s"] = float(m.group(1))
        if m := re.search(r"CPU time for ModelFinder:\s+([\d.]+)", line):
            result["mf_cpu_s"] = float(m.group(1))
        if m := re.search(r"Best-fit model:\s+(\S+)\s+chosen", line):
            result["best_model"] = m.group(1)
        if "MF-MPI:" in line:
            result.setdefault("mf_mpi_lines", []).append(line.strip())
            if ma := re.search(r"rank (\d+)/(\d+) assigned (\d+)/(\d+)", line):
                result.setdefault("models_assigned", {})[int(ma.group(1))] = {
                    "assigned": int(ma.group(3)), "total": int(ma.group(4))
                }
        if m := re.search(r"MF-MPI: gather complete, (\d+) model scores", line):
            result["gather_n"] = int(m.group(1))
    return result

def parse_perf_stat(work_dir):
    """Parse perf stat files, return per-rank dict and aggregated metrics."""
    per_rank = {}
    agg = {}
    for fp in sorted(glob.glob(os.path.join(work_dir, "perf_stat.rank*.txt"))):
        rk = re.search(r"rank(\d+)", fp).group(1)
        pm = {}
        for line in open(fp, errors="replace"):
            m_c = re.match(r"\s*([\d,]+)\s+([\w.\-:/]+)", line)
            if m_c:
                try:
                    key = m_c.group(2).split(":", 1)[0]
                    pm[key] = int(m_c.group(1).replace(",", ""))
                except ValueError: pass
        per_rank[rk] = pm
        for k, v in pm.items(): agg[k] = agg.get(k, 0) + v
    def rate(n, d):
        return round(100.0 * n / d, 4) if n and d else None
    def g(*keys):
        for k in keys:
            if agg.get(k): return agg[k]
        return None
    cyc, ins = g("cycles"), g("instructions")
    metrics = {
        "IPC":                 round(ins/cyc, 4) if cyc and ins else None,
        "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
        "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
        "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
        "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
        "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
    }
    for k in ("cycles","instructions","LLC-loads","LLC-load-misses","cache-references",
              "cache-misses","L1-dcache-loads","L1-dcache-load-misses",
              "dTLB-loads","dTLB-load-misses","iTLB-load-misses"):
        if k in agg: metrics[k] = agg[k]
    per_rank_ipc = {}
    for rk, pm in per_rank.items():
        c, i = pm.get("cycles"), pm.get("instructions")
        if c and i: per_rank_ipc[rk] = round(i/c, 4)
    return metrics, per_rank_ipc

def parse_aps_summary(report_dir):
    """Extract key metrics from APS text reports."""
    result = {}
    summary_path = os.path.join(report_dir, "aps_summary.txt")
    if os.path.isfile(summary_path):
        text = open(summary_path, errors="replace").read()
        result["aps_summary_raw"] = text[:4096]  # first 4KB
        # Extract common APS fields
        for label, pattern in [
            ("mpi_time_pct",   r"MPI Time.*?:\s*([\d.]+)%"),
            ("eff_pct",        r"[Ee]fficiency.*?:\s*([\d.]+)%"),
            ("imbalance_pct",  r"[Ii]mbalance.*?:\s*([\d.]+)%"),
            ("cpu_util_pct",   r"CPU Util.*?:\s*([\d.]+)%"),
            ("vec_intensity",  r"[Vv]ectoriz.*?:\s*([\d.]+)"),
        ]:
            if m := re.search(pattern, text):
                try: result[label] = float(m.group(1))
                except ValueError: pass
    # MPI time per rank
    mpi_path = os.path.join(report_dir, "aps_mpi_time.txt")
    if os.path.isfile(mpi_path):
        result["aps_mpi_time_raw"] = open(mpi_path, errors="replace").read()[:2048]
    return result

def parse_proc_samples(jsonl_path):
    if not os.path.isfile(jsonl_path): return None
    snaps = []
    for raw in open(jsonl_path, errors="replace"):
        try: snaps.append(json.loads(raw))
        except json.JSONDecodeError: pass
    if not snaps: return None
    peak_rss = max((s.get("rss_kb") or 0 for s in snaps), default=0)
    peak_vms = max((s.get("vms_kb") or 0 for s in snaps), default=0)
    max_thr  = max((s.get("threads_now") or 0 for s in snaps), default=0)
    return {
        "sample_count": len(snaps),
        "duration_s":   snaps[-1].get("t_s"),
        "peak_rss_kb":  peak_rss,
        "peak_vms_kb":  peak_vms,
        "max_threads":  max_thr,
    }

# Parse all passes
aps_log = parse_iq_log(os.path.join(work, "iqtree_aps.log"))
perf_log = parse_iq_log(os.path.join(work, "iqtree_perf.log"))
clean_log = parse_iq_log(os.path.join(work, "iqtree_clean.log"))
perf_metrics, per_rank_ipc = parse_perf_stat(work)
aps_metrics = parse_aps_summary(aps_report_dir)
proc_aps = parse_proc_samples(os.path.join(work, "samples_aps.jsonl"))
proc_perf = parse_proc_samples(os.path.join(work, "samples_perf.jsonl"))

# Timing summary
passes = [
    {"pass": "aps",   "wall_s": wall_aps,   "rc": iqrc_aps,   "mf_wall_s": aps_log.get("mf_wall_s"), "best_model": aps_log.get("best_model")},
    {"pass": "perf",  "wall_s": wall_perf,  "rc": iqrc_perf,  "mf_wall_s": perf_log.get("mf_wall_s")},
    {"pass": "clean", "wall_s": wall_clean, "rc": iqrc_clean, "mf_wall_s": clean_log.get("mf_wall_s")},
]

# APS artefact list
artefacts = {}
for fname, key in [
    ("iqtree_aps.log",          "iqtree_aps_log"),
    ("iqtree_perf.log",         "iqtree_perf_log"),
    ("iqtree_clean.log",        "iqtree_clean_log"),
    ("iqtree_aps.bindings.log", "mpi_bindings_log"),
    ("samples_aps.jsonl",       "proc_timeseries_aps"),
    ("samples_perf.jsonl",      "proc_timeseries_perf"),
    ("hostfile.txt",            "hostfile"),
    ("env.json",                "env_json"),
]:
    p = os.path.join(work, fname)
    if os.path.exists(p): artefacts[key] = p
for fname, key in [
    ("aps_summary.txt",   "aps_summary"),
    ("aps_mpi_time.txt",  "aps_mpi_time"),
    ("aps_functions.txt", "aps_functions"),
    ("aps_messages.txt",  "aps_messages"),
    ("aps_counters.txt",  "aps_counters"),
]:
    p = os.path.join(aps_report_dir, fname)
    if os.path.exists(p): artefacts[key] = p
for f in sorted(glob.glob(os.path.join(work, "perf_stat.rank*.txt"))):
    artefacts.setdefault("perf_stat_files", []).append(f)
artefacts["aps_result_dir"] = aps_dir

record = {
  "run_id": rid, "pbs_id": "${PBS_ID_SHORT}",
  "platform": "gadi", "run_type": "profile", "label": label,
  "description": (
    f"MF2 dispatch — {nranks} ranks × {omp_per} OMP, 4 SPR nodes, "
    f"mega_dna.fa (500t×100K sites), full 968-model MF, seed={${SEED}}. "
    f"Intel APS profiling (APS_STAT_LEVEL=${APS_STAT_LEVEL})."
  ),
  "passes": passes,
  "dispatch": {
    "models_assigned_per_rank": aps_log.get("models_assigned", {}),
    "mf_mpi_lines":             aps_log.get("mf_mpi_lines", []),
    "gather_complete":          "gather_n" in aps_log,
    "gather_n":                 aps_log.get("gather_n"),
    "best_model":               aps_log.get("best_model"),
    "mf_wall_s":                aps_log.get("mf_wall_s"),
    "mf_cpu_s":                 aps_log.get("mf_cpu_s"),
    "best_ll":                  aps_log.get("best_ll"),
  },
  "perf_metrics": perf_metrics,
  "per_rank_ipc": per_rank_ipc,
  "aps_metrics":  aps_metrics,
  "proc_summary": {
    "aps_pass":  proc_aps,
    "perf_pass": proc_perf,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc","0") or 0),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "aps":      sh("aps --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "nodes":    4, "hosts": hosts,
    "pbs": {
      "job_id":   os.environ.get("PBS_JOBID"),
      "queue":    os.environ.get("PBS_QUEUE"),
      "project":  os.environ.get("PROJECT","${PROJECT}"),
      "ncpus":    os.environ.get("PBS_NCPUS"),
      "nnodes":   4,
    },
  },
  "summary": {
    "pass_aps":   iqrc_aps  == 0,
    "pass_perf":  iqrc_perf == 0,
    "pass_clean": iqrc_clean == 0,
    "all_pass":   iqrc_aps == 0 and iqrc_perf == 0,
    "exit_aps":   iqrc_aps,
    "exit_perf":  iqrc_perf,
    "exit_clean": iqrc_clean,
  },
  "profile": {
    "dataset":        "${DATASET_FILE}",
    "dataset_taxa":   500,
    "dataset_sites":  100000,
    "threads":        total_thr,
    "mpi_ranks":      nranks,
    "omp_per_rank":   omp_per,
    "nodes":          4,
    "mpirun_mapping": f"--map-by node:PE={omp_per}",
    "full_modelfinder": True,
    "model_count":    968,
    "aps_stat_level": int("${APS_STAT_LEVEL}"),
    "artefacts":      artefacts,
    "build_tag":      "mf2dispatch_lpt_aps_icx_mpi4x104_4node_fullnode_avx512_r2_v312",
    "mf2_commit":     "abd98764",
  },
  "mf2_commit":          "abd98764",
  "non_canonical":       True,
  "non_canonical_label": f"MF2 dispatch · APS · MPI {nranks}×{omp_per} 4-node · mega_dna",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path, "w"), indent=2, default=str)
print(f"[mega-aps] run record → {out_path}")
PYEOF

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " mega_dna.fa MF2 dispatch + Intel APS — 4-node SPR results"
echo "════════════════════════════════════════════════════════════════"
echo "  Dataset: ${DATA_PATH} (500 taxa × 100K sites)"
echo "  Layout:  ${NRANKS} MPI ranks × ${OMP_PER_RANK} OMP threads (${TOTAL_THREADS} total)"
echo "  Binary:  abd98764 (LPT + MF_WAITING fix)"
echo ""
echo "  Pass 1 (APS):   wall=${WALL_APS}s  rc=${IQRC_APS}"
echo "  Pass 2 (perf):  wall=${WALL_PERF}s  rc=${IQRC_PERF}"
echo "  Pass 3 (clean): wall=${WALL_CLEAN}s  rc=${IQRC_CLEAN}"
echo ""
echo "  MF-MPI dispatch (Pass 1):"
grep "MF-MPI:" "${WORK_DIR}/iqtree_aps.log" 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  Best-fit model:"
grep "Best-fit model:" "${WORK_DIR}/iqtree_aps.log" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"
echo ""
echo "  APS reports in: ${WORK_DIR}/aps_reports/"
ls "${WORK_DIR}/aps_reports/" 2>/dev/null | sed 's/^/    /' || echo "    (none generated)"
echo ""
echo "  perf_stat files:"
ls "${WORK_DIR}"/perf_stat.rank*.txt 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  work_dir: ${WORK_DIR}"
echo "════════════════════════════════════════════════════════════════"
echo "[mega-aps] done."
exit "${IQRC_APS}"
