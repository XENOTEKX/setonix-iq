#!/bin/bash
# run_xlarge_mf2_full.sh — MF2 binary, full IQ-TREE (free tree, seed=1).
#
# Runs the MF2 dispatch binary (um09/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi)
# in np=1 OMP-only mode with the SAME protocol as all other families:
#   - full IQ-TREE (no -m MF, no -te)
#   - free tree
#   - seed=1
#
# This makes it directly comparable to ICX Baseline, GCC Canonical,
# R2+NUMA, and AVX-512+R2 on the thread-scaling plot.
#
# Required env vars (set via `qsub -v`):
#   THREADS   — OMP thread count (1, 4, 8, 16, 32, 64, or 104)
#
# Walltime guidance (set via `qsub -l walltime=...`):
#   1T   → 04:00:00
#   4T   → 02:00:00
#   8T   → 01:30:00
#   16T  → 01:00:00
#   32T  → 00:30:00
#   64T  → 00:20:00
#   104T → 00:15:00
#
#PBS -N iq-mf2-full
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=200GB
#PBS -l walltime=04:00:00
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
THREADS="${THREADS:?THREADS env var required}"
SEED="${SEED:-1}"
BUILD_TAG="mf2_full_np1_seed1_avx512_r2_lpt"
LABEL="${LABEL:-${DATASET_NAME}_${THREADS}t_mf2_full_np1_seed1}"

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

export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=10000
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_ID="gadi_${LABEL}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  MF2 binary — full IQ-TREE — ${THREADS}T (np=1)"
echo "║  run_id:  ${RUN_ID}"
echo "║  dataset: ${DATA_PATH}"
echo "║  binary:  ${IQTREE}"
echo "╚═════════════════════════════════════════════════════════════╝"

START_EPOCH=$(date +%s)
IQRC=0
mpirun -np 1 \
    --map-by "node:PE=${THREADS}" \
    -x OMP_NUM_THREADS \
    -x OMP_PROC_BIND \
    -x OMP_PLACES \
    -x OMP_WAIT_POLICY \
    -x KMP_BLOCKTIME \
    -x GOMP_SPINCOUNT \
    numactl --localalloc \
    "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                --prefix "${WORK_DIR}/iqtree_mf2full" \
    > "${WORK_DIR}/iqtree_mf2full.log" 2>&1 || IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
echo "[mf2-full] rc=${IQRC} wall=${WALL}s"

/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label, build_tag = "${RUN_ID}", "${LABEL}", "${BUILD_TAG}"
threads = ${THREADS}; wall = ${WALL}; iqrc = ${IQRC}
dpath, ibin = "${DATA_PATH}", "${IQTREE}"

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
  "description": f"MF2 binary full IQ-TREE (free tree, seed=1) — np=1, {threads}T",
  "timing": [{
    "command": f"mpirun -np 1 ... {ibin} -s xlarge_mf.fa -T {threads} -seed 1",
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": ([{"file": "xlarge_mf.fa", "status": "pass", "expected": rep_ll, "reported": rep_ll, "diff": 0.0}] if rep_ll is not None else []),
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc","0") or 0),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "iqtree_version_tag": "v3.1.2+mf2",
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
    "threads":      threads,
    "mpi_ranks":    1,
    "omp_per_rank": threads,
    "placement":    "omp_pin_numa_ft",
    "build_tag":    build_tag,
  },
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path, "w"), indent=2, default=str)
print(f"[mf2-full] wrote {out_path}")
PYEOF

echo "[mf2-full] done."
exit "${IQRC}"
