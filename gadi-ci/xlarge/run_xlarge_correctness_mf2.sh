#!/bin/bash
# run_xlarge_correctness_mf2.sh — Correctness test with MF2 dispatch active.
#
# Runs the MF2 binary with 2 MPI ranks × 52 OMP threads on 1 node.
# Round-robin dispatch assigns odd-indexed models to rank 0, even to rank 1
# (or vice versa).  After all models complete, MPI_Allreduce merges results
# and the best model is selected globally across all ranks.
#
# Pair with run_xlarge_correctness_baseline.sh (1 rank, no dispatch).
# Both scripts use:
#   - identical binary  (iqtree3-mf2/build-mpi-mf2/iqtree3-mpi)
#   - identical dataset (xlarge_mf.fa, sha256 locked)
#   - identical seed    (1)
#   - identical node    (1 × normalsr SPR, 104 physical cores)
#   - identical total threads (104 = 2 × 52)
# Only difference: NRANKS=2 here vs NRANKS=1 in the baseline script.
#
# PASS criterion: "Best-fit model: GTR+R4" — same as baseline.
# FAIL: any other model name, or missing Best-fit model line.
#
#PBS -N iq-xlarge-correctness-mf2
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
NRANKS=2
OMP_PER_RANK=52
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED=1
LABEL="${DATASET_NAME}_${TOTAL_THREADS}t_mf2binary_mpi${NRANKS}x${OMP_PER_RANK}_dispatch_correctness"

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
echo "║  CORRECTNESS TEST — MF2 binary, 2 ranks, dispatch ACTIVE"
echo "║  dataset:  ${DATA_PATH}"
echo "║  binary:   ${IQTREE}"
echo "║  ranks:    ${NRANKS} × ${OMP_PER_RANK} OMP  (total ${TOTAL_THREADS} threads)"
echo "║  seed:     ${SEED}"
echo "║  work_dir: ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Dispatch: rank 0 → odd-indexed models (1,3,5,...,967)"
echo "            rank 1 → even-indexed models (2,4,6,...,968)"
echo "  MPI_Allreduce merges all results → global best model selected"
echo ""

export OMP_NUM_THREADS=${OMP_PER_RANK}
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"

T_START=$(date +%s)

mpirun \
    -np ${NRANKS} \
    --map-by socket:PE=${OMP_PER_RANK} \
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
BEST=$(grep "Best-fit model" "${WORK_DIR}/iqtree_run.log" || echo "(not found)")
echo "  MF2 best model:  ${BEST}"
echo "  Reference:       Best-fit model: GTR+R4 chosen according to BIC"
echo "  Wall time:       ${ELAPSED}s"
echo ""
if echo "${BEST}" | grep -q "GTR+R4"; then
    echo "  ✔  CORRECTNESS PASS — MF2 dispatch selected GTR+R4 (matches baseline)"
else
    echo "  ✗  CORRECTNESS FAIL — model mismatch! Check ${WORK_DIR}/iqtree_run.log"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Verify dispatch was active (rank-0 log should show every other model No.):"
echo "  First 5 completed models in rank-0 log:"
grep -E "^\s*[0-9]+" "${WORK_DIR}/iqtree_run.log" | grep -v "%" | head -5 || true
