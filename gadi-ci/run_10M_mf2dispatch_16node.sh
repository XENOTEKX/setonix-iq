#!/bin/bash
# run_10M_mf2dispatch_16node.sh
# IQ-TREE 3.1.2 + MF2 LPT dispatch — alignment_10000000.phy on 16 Gadi normalsr SPR nodes
# Full ModelFinder (968 models) with Intel APS profiling, then clean timing pass.
#
# Dataset: alignment_10000000.phy — 100 taxa × 10,000,000 sites, 0% pattern compression
# Layout:  16 MPI ranks (1 rank/node), 104 OMP threads/rank  →  1664 total threads
# Binary:  abd98764 (LPT + MF_WAITING fix, branch gadi-spr-r2-avx512)
# Profiling: Intel APS (APS_STAT_LEVEL=2) — pass 1 only; pass 2 is clean timing
#
# ┌─ Timing estimates (100 taxa × 10M sites, 104T/rank) ──────────────────────────────┐
# │  From PBS 167977883: 9 models in 2h @ 4 nodes 104T → ~729s/model at 100T          │
# │  16-rank LPT: ~61 models/rank (bottleneck) × 729s ≈ 12.4h for ModelFinder        │
# │  Starting tree (100 taxa) ≈ 8-15 min                                               │
# │  APS overhead: +10-15% → budget 15h for pass 1; pass 2 uses checkpoint → ~2h      │
# │  Total: 2 passes ≈ 17-18h; walltime budget: 20h                                   │
# └───────────────────────────────────────────────────────────────────────────────────┘
#
# Distribution analysis (from mega_dna 100K-site empirical sample, §13 of design doc):
#   At 10M sites: bimodal, CV≈36%, Max/Min≈2.7×
#     Fast cluster (+R2):     ~382 models × ~109 min each
#     Slow cluster (+R4-R6):  ~586 models × ~255 min each
#   LPT at 16 ranks: ~1.0% imbalance vs naive's ~6.9% → saves ~12h walltime
#
# RAM per node: IQ-TREE declares 324 GB for 10M/100taxa; 1 rank/node gives full node
# normalsr:     503 GB RAM/node, 104 cores/2-socket Sapphire Rapids
#
# Usage:
#   qsub run_10M_mf2dispatch_16node.sh
#   # With overrides:
#   qsub -v SEED=42            run_10M_mf2dispatch_16node.sh
#   qsub -v APS_STAT_LEVEL=1   run_10M_mf2dispatch_16node.sh   # lighter APS
#
#PBS -N iq-10M-mf2-16node
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=1664
#PBS -l mem=6400GB
#PBS -l walltime=20:00:00
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

DATASET_NAME="${DATASET_NAME:-xlarge_mf}"
DATASET_SUBDIR="${DATASET_SUBDIR:-100taxa_10M}"
DATASET_FILE="${DATASET_FILE:-alignment_10000000.phy}"
NRANKS="${NRANKS:-16}"
OMP_PER_RANK="${OMP_PER_RANK:-104}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
EXPECTED_NODES="${EXPECTED_NODES:-16}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_mf2dispatch_aps_mpi${NRANKS}x${OMP_PER_RANK}_${EXPECTED_NODES}node_fullnode}"

DATA_PATH="${BENCHMARKS}/${DATASET_SUBDIR}/${DATASET_FILE}"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }

# ── Module loading ─────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
    module load intel-vtune/2025.8.1         2>/dev/null || true
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
export OMP_NUM_THREADS="${OMP_PER_RANK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export GOMP_SPINCOUNT=10000
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

# ── APS tuning ─────────────────────────────────────────────────────────
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
if [[ "${#HOSTS[@]}" -ne "${EXPECTED_NODES}" ]]; then
    echo "ERROR: expected ${EXPECTED_NODES} nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2; exit 9
fi

# Build hostfile for reference
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
echo "║  IQ-TREE 3.1.2 + MF2 LPT dispatch + Intel APS — 16-node 10M"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads)"
echo "║  seed:          ${SEED}"
echo "║  full_mf:       968 DNA models (no --mrate restriction)"
echo "║  dispatch:      LPT stripe, 61 models/rank (bottleneck)"
echo "║  expected MF:   ~12.4h (61 models × 729s; LPT imbalance ≤1.0%)"
echo "║  binary:        ${IQTREE}"
echo "║  aps version:   $(aps --version 2>&1 | head -1)"
echo "║  APS_STAT_LEVEL: ${APS_STAT_LEVEL}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[10M-16node] hostfile (reference):"; cat "${HOSTFILE}" | sed 's/^/    /'

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
hosts_all = open("${PBS_NODEFILE}").read().splitlines() if os.path.isfile("${PBS_NODEFILE:-/dev/null}") else []
hosts_unique = sorted(set(hosts_all))
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
    "proc_sampler_interval_s": 30,
  },
  "mpirun_mapping": "--map-by node:PE=${OMP_PER_RANK}",
  "seed": ${SEED},
  "placement": "mf2_mpi_16node_fullnode",
  "nodes": ${EXPECTED_NODES},
  "hosts": hosts_unique,
  "hostname": sh("hostname"), "kernel": sh("uname -r"),
  "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
  "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets":          int(sh("lscpu | awk -F: '/Socket\\(s\\)/{print \\$2}' | xargs","0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\\(s\\) per socket/{print \\$2}' | xargs","0") or 0),
  "numa_nodes":           int(sh("lscpu | awk -F: '/NUMA node\\(s\\)/{print \\$2}' | xargs","0") or 0),
  "mem_total_kb":         int(sh("awk '/MemTotal/{print \\$2}' /proc/meminfo","0") or 0),
  "smt_active": sh("cat /sys/devices/system/cpu/smt/active 2>/dev/null") == "1",
  "mpi_version": sh("mpirun --version 2>&1 | head -1"),
  "icx":  sh("icx --version 2>/dev/null | head -1"),
  "iqtree_binary":  "${IQTREE}",
  "iqtree_version": sh("${IQTREE} --version 2>&1 | head -1"),
  "date": sh("date -Iseconds"),
  "dataset": {
    "path": ds, "file": "${DATASET_FILE}",
    "taxa": 100, "sites": 10000000, "seq_type": "DNA",
    "distinct_patterns": 10000000,
    "pattern_compression_pct": 0,
    "size_bytes": os.path.getsize(ds) if os.path.isfile(ds) else None,
    "sha256": sha256(ds),
    "note": "100 taxa × 10M sites, 0% pattern compression, worst-case dataset",
    "declared_ram_per_rank_gb": 324,
    "estimated_mf_wall_h": 12.4,
  },
  "dispatch_analysis": {
    "source": "empirical mega_dna 100K sample (119 models, PBS 168015597)",
    "distribution_shape": "bimodal",
    "fast_cluster_n": 382, "fast_cluster_mean_min": 109,
    "slow_cluster_n": 586, "slow_cluster_mean_min": 255,
    "cv_pct": 36.0, "max_min_ratio": 2.6,
    "lpt_imbalance_16ranks_pct": 1.0,
    "naive_imbalance_16ranks_pct": 6.9,
  },
  "pbs": {
    "job_id":     os.environ.get("PBS_JOBID"),
    "job_name":   os.environ.get("PBS_JOBNAME"),
    "queue":      os.environ.get("PBS_QUEUE"),
    "project":    os.environ.get("PROJECT","${PROJECT}"),
    "ncpus":      os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    "nnodes":     ${EXPECTED_NODES},
    "nodes":      hosts_unique,
    "submit_host": os.environ.get("PBS_O_HOST"),
    "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
    "scheduler":   "pbs_pro",
  },
}
print(json.dumps(env, indent=2))
PYENV
echo "  → ${ENV_JSON}"

# ── /proc sampler (30s interval — model evaluations take hours, not seconds) ──
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
import json, os, sys, time, pathlib
pid = int(sys.argv[1]); out = pathlib.Path(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
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
# PASS 1: Intel APS profiling
# ════════════════════════════════════════════════════════════════
# Per-rank APS wrapper: each rank runs aps -r <dir> -- numactl -- iqtree3-mpi
# APS_STAT_LEVEL=2: MPI tracing + hardware counters + function summary
echo ""
echo "[10M-16node] Pass 1: Intel APS (APS_STAT_LEVEL=${APS_STAT_LEVEL})"
echo "[10M-16node] Expected dispatch: rank 0/16 assigned ~61/968 models (cost-sorted LPT stripe)"
echo "[10M-16node] Expected MF wall:  ~12-15h (bottleneck rank × 729s/model)"
START_EPOCH=$(date +%s)
IQRC_APS=0

APS_WRAP="${WORK_DIR}/_aps_wrap.sh"
cat > "${APS_WRAP}" <<APSEOF
#!/bin/bash
exec aps -r '${APS_DIR}' \
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

# /proc sampler at 30s interval (model evals take hours)
sleep 15
INNER_PID="$(pgrep -P "${IQTREE_APS_PID}" 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_APS_PID}"
echo "  → mpirun pid=${IQTREE_APS_PID}, sampler tracking pid=${INNER_PID}"
python3 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples_aps.jsonl" 30 &
SAMPLER_APS_PID=$!

wait "${IQTREE_APS_PID}" || IQRC_APS=$?
IQRC_APS="${IQRC_APS:-0}"
END_EPOCH_APS=$(date +%s)
WALL_APS=$(( END_EPOCH_APS - START_EPOCH ))
kill "${SAMPLER_APS_PID}" 2>/dev/null || true
wait "${SAMPLER_APS_PID}" 2>/dev/null || true
echo "[10M-16node] Pass 1 done: rc=${IQRC_APS} wall=${WALL_APS}s ($(( WALL_APS/3600 ))h$(( (WALL_APS%3600)/60 ))m)"

# ── MF-MPI dispatch diagnostics ───────────────────────────────────────
echo ""
echo "[10M-16node] MF-MPI dispatch diagnostics (Pass 1):"
grep "MF-MPI:" "${WORK_DIR}/iqtree_aps.log" 2>/dev/null | sed 's/^/    /' || \
    echo "    (no MF-MPI: lines — check binary LPT fix)"
echo "[10M-16node] Best-fit model:"
grep "Best-fit model:" "${WORK_DIR}/iqtree_aps.log" 2>/dev/null | sed 's/^/    /' || \
    echo "    (not completed)"

# ── Per-model timing from checkpoint ─────────────────────────────────
# Extract model wall times from pass 1 checkpoint for empirical distribution
# at 10M sites (feeds back into §13 dispatch analysis).
if [[ -f "${WORK_DIR}/iqtree_aps.model.gz" ]]; then
    echo ""
    echo "[10M-16node] Extracting per-model wall times from checkpoint..."
    python3 - "${WORK_DIR}/iqtree_aps.model.gz" "${WORK_DIR}/model_times_10M.tsv" <<'PYEOF'
import gzip, sys, re

gz_path = sys.argv[1]
out_path = sys.argv[2]
records = []
model_name = None
df_val = None
wall_val = None

with gzip.open(gz_path, "rt", errors="replace") as f:
    for line in f:
        line = line.strip()
        # Lines look like: "model_name\t..." or key=value
        m = re.match(r"^(\S+)\s*=\s*(.+)$", line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            if key.endswith("_df"):
                try: df_val = int(val)
                except ValueError: pass
            elif key.endswith("_wall"):
                try: wall_val = float(val)
                except ValueError: pass
            elif key.endswith("_name") or re.match(r'^[A-Z]', key):
                model_name = val
            if model_name and df_val is not None and wall_val is not None:
                records.append((model_name, df_val, wall_val))
                model_name = df_val = wall_val = None

with open(out_path, "w") as f:
    f.write("model\tdf\twall_s\n")
    for rec in records:
        f.write(f"{rec[0]}\t{rec[1]}\t{rec[2]:.3f}\n")
print(f"  Extracted {len(records)} model timing records → {out_path}")
PYEOF
fi

# ════════════════════════════════════════════════════════════════
# PASS 2: clean timing (uses pass-1 checkpoint — MF skips instantly)
# ════════════════════════════════════════════════════════════════
# Pass 1 wrote a complete .model.gz checkpoint. Pass 2 re-runs IQ-TREE:
# ModelFinder reads the checkpoint, skips all 968 models (already computed),
# then proceeds to tree search + final report. This gives a clean wall-time
# measurement for the tree-search phase and validates checkpoint resume.
IQRC_CLEAN=0
WALL_CLEAN=0
if [[ "${IQRC_APS}" -eq 0 ]]; then
    echo ""
    echo "[10M-16node] Pass 2: clean timing (checkpoint resume — MF phase should skip)"
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
    echo "[10M-16node] Pass 2 done: rc=${IQRC_CLEAN} wall=${WALL_CLEAN}s ($(( WALL_CLEAN/3600 ))h$(( (WALL_CLEAN%3600)/60 ))m)"
else
    echo "[10M-16node] Skipping pass 2 — pass 1 failed (rc=${IQRC_APS})"
fi

# ════════════════════════════════════════════════════════════════
# APS REPORT GENERATION
# ════════════════════════════════════════════════════════════════
echo ""
echo "[10M-16node] Generating Intel APS reports from ${APS_DIR}..."
APS_REPORT_DIR="${WORK_DIR}/aps_reports"
mkdir -p "${APS_REPORT_DIR}"

if [[ "${IQRC_APS}" -eq 0 ]] || [[ -d "${APS_DIR}" ]]; then
    echo "[10M-16node] APS result dir: ${APS_DIR}"
    ls "${APS_DIR}" 2>/dev/null | sed 's/^/    /'

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
                echo "  ✓ ${RNAME}" || echo "  ✗ ${RNAME}"
        else
            aps --report "${APS_DIR}" ${report_flag} > "${RPATH}" 2>&1 && \
                echo "  ✓ ${RNAME}" || echo "  ✗ ${RNAME}"
        fi
    done

    echo ""
    echo "════ APS SUMMARY ════"
    cat "${APS_REPORT_DIR}/aps_summary.txt" 2>/dev/null | head -80 || echo "(no summary)"
    echo "════════════════════"
fi

# ════════════════════════════════════════════════════════════════
# RUN RECORD HARVESTING
# ════════════════════════════════════════════════════════════════
echo ""
echo "[10M-16node] Building run record..."
WALL="${WALL_APS}"

python3 - <<PYEOF
import json, os, re, glob
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
total_thr = ${TOTAL_THREADS}; nranks=${NRANKS}; omp_per=${OMP_PER_RANK}
wall_aps, wall_clean = int("${WALL_APS}"), int("${WALL_CLEAN}")
iqrc_aps, iqrc_clean = int("${IQRC_APS}"), int("${IQRC_CLEAN}")
dpath = "${DATA_PATH}"

def parse_iq_log(log_path):
    result = {}
    if not os.path.isfile(log_path): return result
    for line in open(log_path, errors="replace"):
        if m := re.search(r"Best-fit model:\s+(\S+)\s+chosen", line):
            result["best_model"] = m.group(1)
        if m := re.search(r"Total wall-clock time used:\s+([\d.]+)", line):
            result["total_wall_s"] = float(m.group(1))
        if m := re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line):
            result["mf_wall_s"] = float(m.group(1))
        if m := re.search(r"CPU time for ModelFinder:\s+([\d.]+)", line):
            result["mf_cpu_s"] = float(m.group(1))
        if "MF-MPI:" in line:
            result.setdefault("mf_mpi_lines", []).append(line.strip())
            if ma := re.search(r"rank (\d+)/(\d+) assigned (\d+)/(\d+)", line):
                result.setdefault("models_assigned", {})[int(ma.group(1))] = {
                    "assigned": int(ma.group(3)), "total": int(ma.group(4))
                }
    return result

aps_log = parse_iq_log(os.path.join(work, "iqtree_aps.log"))
clean_log = parse_iq_log(os.path.join(work, "iqtree_clean.log"))

# Per-model timing summary
model_times_path = os.path.join(work, "model_times_10M.tsv")
model_timing_summary = {}
if os.path.isfile(model_times_path):
    import statistics
    times = []
    for line in open(model_times_path):
        parts = line.strip().split("\t")
        if len(parts) == 3 and parts[0] != "model":
            try: times.append(float(parts[2]))
            except ValueError: pass
    if times:
        model_timing_summary = {
            "n": len(times),
            "mean_s": round(statistics.mean(times), 2),
            "min_s": round(min(times), 2),
            "max_s": round(max(times), 2),
            "stdev_s": round(statistics.stdev(times), 2) if len(times) > 1 else 0,
            "cv_pct": round(statistics.stdev(times) / statistics.mean(times) * 100, 1) if len(times) > 1 else 0,
        }

# Load env.json
env_data = {}
env_path = os.path.join(work, "env.json")
if os.path.isfile(env_path):
    env_data = json.load(open(env_path))

record = {
    "run_id": rid, "label": label,
    "dataset": "alignment_10000000.phy",
    "taxa": 100, "sites": 10000000, "distinct_patterns": 10000000,
    "threads": total_thr, "mpi_ranks": nranks, "omp_per_rank": omp_per,
    "nodes": ${EXPECTED_NODES},
    "mf2_dispatch": True, "dispatch_strategy": "lpt_cost_sorted_stripe",
    "mf2_commit": "abd98764",
    "passes": {
        "aps":   {"rc": iqrc_aps,   "wall_s": wall_aps,   **aps_log},
        "clean": {"rc": iqrc_clean, "wall_s": wall_clean, **clean_log},
    },
    "model_timing_10M": model_timing_summary,
    "env": env_data,
    "work_dir": work,
}
os.makedirs(runs, exist_ok=True)
out_path = os.path.join(runs, f"{rid}.json")
json.dump(record, open(out_path, "w"), indent=2)
print(f"  Run record: {out_path}")

# Quick summary to stdout
print(f"  best_model:    {aps_log.get('best_model','(not completed)')}")
print(f"  MF wall (APS): {wall_aps//3600}h{(wall_aps%3600)//60}m")
print(f"  MF wall (clean): {wall_clean//3600}h{(wall_clean%3600)//60}m")
if model_timing_summary:
    ts = model_timing_summary
    print(f"  per-model:     n={ts['n']}, mean={ts['mean_s']}s, min={ts['min_s']}s, max={ts['max_s']}s, CV={ts['cv_pct']}%")
mf_disp = aps_log.get("mf_mpi_lines", [])
for l in mf_disp[:${NRANKS}]: print(f"    {l}")
PYEOF

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  10M 16-node MF2 dispatch — run complete"
echo "  PBS job:       ${PBS_JOBID:-local}"
echo "  Pass 1 (APS):  wall=$(( WALL_APS/3600 ))h$(( (WALL_APS%3600)/60 ))m  rc=${IQRC_APS}"
echo "  Pass 2 (clean):wall=$(( WALL_CLEAN/3600 ))h$(( (WALL_CLEAN%3600)/60 ))m  rc=${IQRC_CLEAN}"
echo "  APS reports:   ${WORK_DIR}/aps_reports/"
echo "  Model times:   ${WORK_DIR}/model_times_10M.tsv"
echo "  Work dir:      ${WORK_DIR}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "[10M-16node] Check results with:"
echo "  grep 'MF-MPI:'      ${WORK_DIR}/iqtree_aps.log"
echo "  grep 'Best-fit'     ${WORK_DIR}/iqtree_aps.log"
echo "  cat ${WORK_DIR}/aps_reports/aps_summary.txt"
echo "  cat ${WORK_DIR}/model_times_10M.tsv | head"
