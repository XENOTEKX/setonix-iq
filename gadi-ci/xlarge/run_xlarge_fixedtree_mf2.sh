#!/bin/bash
# run_xlarge_fixedtree_mf2.sh — Correctness test, fixed tree, 2 ranks (MF2 dispatch).
#
# Uses the MF2 binary with 2 ranks and a fixed starting tree (-te), so the
# only work done is ModelFinder model selection with MF2 round-robin dispatch.
# This is the cleanest possible test of the dispatch patch: same tree, same data,
# same binary — only the rank count changes, activating Phase 1 dispatch and
# Phase 2 MPI_Allreduce merge.
#
# ModelFinder code path in NEW (MF2) binary:
#   np=2 → MF2 Phase 1 round-robin dispatch (MF_IGNORED stripe assignment)
#          → each rank evaluates 484/968 models via evaluateAll()
#          → Phase 2 MPI_Allreduce gathers all 968 model scores
#          → best model selected from full 968-model evaluation
#   evaluateAll() does NOT do BIC pruning, so finds SYM+G4 on xlarge_mf.fa.
#
# Fixed tree source: PBS 168004012 (1-rank free-search run, lnL -10956936.089)
# Copied to: /scratch/um09/as1708/iqtree3-mf2/gadi-ci/fixed_xlarge_mf2_tree.nwk
#
# Correctness comparison:
#   run_xlarge_fixedtree_baseline.sh   — OLD binary, test(), 1 rank  → expected GTR+R4
#   THIS SCRIPT (MF2 binary, 2 ranks)                                → expected SYM+G4
#   DISPATCH CORRECTNESS: MF2 binary 1-rank (PBS 168004710) = SYM+G4 ✔ matches 2-rank
#
# PASS: this 2-rank result matches 1-rank MF2 binary result (PBS 168004710: SYM+G4).
# FAIL: any difference vs 1-rank MF2 binary is a pure dispatch bug.
#
#PBS -N iq-xlarge-fixedtree-mf2
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=00:15:00
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
FIXED_TREE="${PROJECT_DIR}/gadi-ci/fixed_xlarge_mf2_tree.nwk"

DATASET_NAME="xlarge_mf"
DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
NRANKS=2
OMP_PER_RANK=52
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED=1
LABEL="${DATASET_NAME}_${TOTAL_THREADS}t_mf2binary_mpi${NRANKS}x${OMP_PER_RANK}_fixedtree_dispatch"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7        2>/dev/null || true
    module load intel-compiler-llvm  2>/dev/null || true
fi

[[ -f "${DATA_PATH}" ]]   || { echo "ERROR: dataset not found: ${DATA_PATH}" >&2; exit 2; }
[[ -f "${FIXED_TREE}" ]]  || { echo "ERROR: fixed tree not found: ${FIXED_TREE}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }
[[ -x "${IQTREE}" ]]      || { echo "ERROR: ${IQTREE} not found." >&2; exit 5; }

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FIXED-TREE MF2 DISPATCH — 2 ranks, -te (no NNI)"
echo "║  dataset:    ${DATA_PATH}"
echo "║  fixed tree: ${FIXED_TREE}"
echo "║  binary:     ${IQTREE}"
echo "║  ranks:      ${NRANKS} × ${OMP_PER_RANK} OMP  (total ${TOTAL_THREADS})"
echo "║  seed:       ${SEED}"
echo "║  dispatch:   rank 0 → odd models, rank 1 → even models"
echo "╚══════════════════════════════════════════════════════════════╝"

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
        -te "${FIXED_TREE}" \
        -T ${OMP_PER_RANK} \
        -seed ${SEED} \
        --prefix "${WORK_DIR}/iqtree_run" \
    2>&1 | tee "${WORK_DIR}/iqtree_run.log"

T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

BEST=$(grep "Best-fit model" "${WORK_DIR}/iqtree_run.log" | tail -1 || echo "(not found)")
DISPATCH=$(grep "MF-MPI: rank" "${WORK_DIR}/iqtree_run.log" | head -3 || echo "(MF-MPI lines not found)")
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  FIXED-TREE MF2 RESULT"
echo "  ${BEST}"
echo "  Dispatch confirmed: ${DISPATCH}"
echo "  Wall time: ${ELAPSED}s"
echo "══════════════════════════════════════════════════════════════"
