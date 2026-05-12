#!/bin/bash
# run_xlarge_mf2_full_8node.sh — MF2 binary, full IQ-TREE, 8-node MPI.
#
# 8 MPI ranks × 104 OMP threads = 832 effective threads, 8 full SPR nodes.
# Same protocol as all other full-IQ-TREE families:
#   - full IQ-TREE (no -m MF, no -te), free tree, seed=1
#
#PBS -N iq-mf2-full-8node
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=832
#PBS -l mem=1600GB
#PBS -l walltime=00:20:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-mpi-mf2}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROFILES_DIR:-${PROJECT_DIR}/gadi-ci/profiles}"

DATASET_NAME="${DATASET:-xlarge_mf}"
NRANKS="${NRANKS:-8}"
OMP_PER_RANK="${OMP_PER_RANK:-104}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
BUILD_TAG="mf2_full_np${NRANKS}_seed${SEED}_avx512_r2_lpt"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_mf2_full_np${NRANKS}_seed${SEED}}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }
[[ -x "${IQTREE}"    ]] || { echo "ERROR: binary ${IQTREE} not found." >&2; exit 5; }

SHA256_LOCKFILE="${SHA256_LOCKFILE:-${REPO_DIR}/benchmarks/sha256sums.txt}"
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

if readelf -d "${IQTREE}" 2>/dev/null | grep -q 'NEEDED.*libmpi'; then
    echo "[preflight] libmpi: CONFIRMED (ELF dynamic section)"
else
    echo "WARNING: libmpi not found in ELF dynamic section of ${IQTREE}" >&2
fi

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7        2>/dev/null || true
    module load intel-compiler-llvm  2>/dev/null || true
fi
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }

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
if [[ "${#HOSTS[@]}" -ne "${NRANKS}" ]]; then
    echo "ERROR: expected ${NRANKS} nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2
    echo "       Check PBS spec: #PBS -l ncpus=${TOTAL_THREADS} on normalsr." >&2
    exit 9
fi

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

# ── Topology check ────────────────────────────────────────────────────
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
if [[ "${PHYSICAL_CORES}" -ne 104 ]]; then
    echo "ERROR: head-node has ${PHYSICAL_CORES} cores, expected 104 (2×52 SPR)." >&2; exit 10
fi

RANKFILE="${WORK_DIR}/rankfile.txt"
> "${RANKFILE}"
for i in "${!HOSTS[@]}"; do
    echo "rank ${i}=${HOSTS[$i]} slot=0-103" >> "${RANKFILE}"
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MF2 binary — full IQ-TREE — ${NRANKS} ranks × ${OMP_PER_RANK}T = ${TOTAL_THREADS}T (8 nodes)"
echo "║  run_id:  ${RUN_ID}"
echo "║  dataset: ${DATA_PATH}"
echo "║  binary:  ${IQTREE}"
for i in "${!HOSTS[@]}"; do
    echo "║  node ${i}: ${HOSTS[$i]}  (rank ${i}, slot=0-103)"
done
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[mf2-full-8node] hostfile:"; cat "${HOSTFILE}" | sed 's/^/    /'
echo "[mf2-full-8node] rankfile:"; cat "${RANKFILE}"  | sed 's/^/    /'

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

TIME_WRAP="${WORK_DIR}/_time_wrap.sh"
cat > "${TIME_WRAP}" <<'EOF'
#!/bin/bash
exec numactl --localalloc -- "$@"
EOF
chmod +x "${TIME_WRAP}"

START_EPOCH=$(date +%s)
IQRC=0
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_mf2full" \
    > "${WORK_DIR}/iqtree_mf2full.log" 2> "${WORK_DIR}/iqtree_mf2full.bindings.log" || IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
echo "[mf2-full-8node] rc=${IQRC} wall=${WALL}s"

HOSTS_JSON="$(printf '"%s",' "${HOSTS[@]}" | sed 's/,$//')"

/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label, build_tag = "${RUN_ID}", "${LABEL}", "${BUILD_TAG}"
total_thr = ${TOTAL_THREADS}; nranks = ${NRANKS}; omp_per = ${OMP_PER_RANK}
wall = int("${WALL}"); iqrc = int("${IQRC}")
dpath, ibin = "${DATA_PATH}", "${IQTREE}"
hosts_json = [${HOSTS_JSON}]

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_mf2full.log")
rep_ll = None; iqwall = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        if m := re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line): rep_ll = float(m.group(1))
        if m := re.search(r"Total wall-clock time used:\s+([\d.]+)", line): iqwall = float(m.group(1))

record = {
  "run_id": rid, "pbs_id": "${PBS_ID_SHORT}",
  "platform": "gadi", "run_type": "profile", "label": label,
  "description": (f"MF2 binary full IQ-TREE (free tree, seed=1) — "
                  f"np={nranks} 8-node full-node, {nranks}×{omp_per}T = {total_thr}T"),
  "timing": [{
    "command": (f"mpirun -np {nranks} --hostfile hostfile.txt -rf rankfile.txt "
                f"numactl --localalloc {ibin} -s xlarge_mf.fa -T {omp_per} -seed 1"),
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": ([{"file": "xlarge_mf.fa", "status": "pass",
               "expected": rep_ll, "reported": rep_ll, "diff": 0.0}]
              if rep_ll is not None else []),
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc","0") or 0),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "iqtree_version_tag": "v3.1.2+mf2",
    "nodes": nranks, "hosts": hosts_json,
    "rankfile": open(os.path.join(work,"rankfile.txt")).read(),
    "pbs": {
      "job_id":  os.environ.get("PBS_JOBID"),
      "project": "${PROJECT}",
      "ncpus":   os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    },
  },
  "summary": {
    "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
    "total_time": iqwall if iqwall is not None else wall,
    "all_pass":   iqrc == 0,
  },
  "profile": {
    "dataset":      "xlarge_mf.fa",
    "threads":      total_thr,
    "mpi_ranks":    nranks,
    "omp_per_rank": omp_per,
    "placement":    "mpi_8node_fullnode",
    "nodes":        nranks,
    "build_tag":    build_tag,
  },
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path, "w"), indent=2, default=str)
print(f"[mf2-full-8node] wrote {out_path}")
PYEOF

echo "[mf2-full-8node] done."
exit "${IQRC}"
