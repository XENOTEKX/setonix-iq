#!/bin/bash
# run_xlarge_correctness_baseline.sh — Correctness reference run (no MF2 dispatch).
#
# Runs the MF2 binary with a SINGLE MPI rank (= no round-robin dispatch).
# One rank evaluates all 968 models sequentially, exactly as the original
# IQ-TREE 3 MPI baseline does.  Best-fit model is the ground-truth answer
# for the xlarge_mf.fa dataset.
#
# Pair with run_xlarge_correctness_mf2.sh (2 ranks, dispatch active).
# Both scripts use:
#   - identical binary  (iqtree3-mf2/build-mpi-mf2/iqtree3-mpi)
#   - identical dataset (xlarge_mf.fa, sha256 locked)
#   - identical seed    (1)
#   - identical node    (1 × normalsr SPR, 104 physical cores)
#   - identical total threads (104)
# Only difference: NRANKS=1 here vs NRANKS=2 in the MF2 script.
#
# Expected result: Best-fit model GTR+R4 (matches PBS 167969243, old OMP binary)
#
#PBS -N iq-xlarge-correctness-baseline
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=00:30:00
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
BENCHMARKS="${PROJECT_DIR}/benchmarks"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="xlarge_mf"
DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
NRANKS=1
OMP_PER_RANK=104
SEED=1
LABEL="${DATASET_NAME}_${OMP_PER_RANK}t_mf2binary_mpi${NRANKS}x${OMP_PER_RANK}_baseline_correctness"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7        2>/dev/null || true
    module load intel-compiler-llvm  2>/dev/null || true
fi

# ── Preflight ─────────────────────────────────────────────────────────
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }
[[ -x "${IQTREE}" ]] || { echo "ERROR: ${IQTREE} not found." >&2; exit 5; }
ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)' || { echo "ERROR: not an MPI binary." >&2; exit 6; }
ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp' && { echo "ERROR: links libgomp, expected libiomp5." >&2; exit 7; }

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CORRECTNESS BASELINE — MF2 binary, 1 rank, no dispatch"
echo "║  dataset:  ${DATA_PATH}"
echo "║  binary:   ${IQTREE}"
echo "║  ranks:    ${NRANKS} × ${OMP_PER_RANK} OMP  (total ${OMP_PER_RANK} threads)"
echo "║  seed:     ${SEED}"
echo "║  work_dir: ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"

export OMP_NUM_THREADS=${OMP_PER_RANK}
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"

T_START=$(date +%s)

mpirun \
    -np ${NRANKS} \
    --map-by node:PE=${OMP_PER_RANK} \
    -x OMP_NUM_THREADS \
    -x KMP_BLOCKTIME \
    "${IQTREE}" \
        -s "${DATA_PATH}" \
        -T ${OMP_PER_RANK} \
        -seed ${SEED} \
        --prefix "${WORK_DIR}/iqtree_run" \
    2>&1 | tee "${WORK_DIR}/iqtree_run.log"

T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  RESULT SUMMARY"
echo "══════════════════════════════════════════════════════════════"
grep "Best-fit model" "${WORK_DIR}/iqtree_run.log" || echo "(Best-fit model line not found)"
echo "  Wall time: ${ELAPSED}s"
echo "  Reference: GTR+R4 (PBS 167969243, old OMP binary)"
echo "══════════════════════════════════════════════════════════════"
