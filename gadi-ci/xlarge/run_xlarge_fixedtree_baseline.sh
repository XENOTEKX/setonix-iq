#!/bin/bash
# run_xlarge_fixedtree_baseline.sh — Correctness baseline: OLD binary, test() path.
#
# Uses the ORIGINAL iqtree3-3.1.2 MPI binary (R2+AVX512 patches, NO MF2 patches)
# with a fixed starting tree (-te) so no NNI tree search runs.
# This isolates ModelFinder model selection using the test() code path.
#
# ModelFinder code path in OLD binary:
#   np=1 → params.openmp_by_model = false → runModelFinder() calls model_set.test()
#   test() uses sequential BIC-pruning with early stopping — can skip better models.
#   Known to select GTR+R4 on xlarge_mf.fa (PBS 167969243, PBS 167997082).
#
# Compare against:
#   run_xlarge_fixedtree_mf2.sh  — NEW binary, evaluateAll(), 2 ranks, dispatch active
#   (NEW binary, 1-rank result already confirmed: SYM+G4, PBS 168004710/712)
#
# Fixed tree: /scratch/um09/as1708/iqtree3-mf2/gadi-ci/fixed_xlarge_mf2_tree.nwk
#   Source: PBS 168004012 (MF2 binary, 1-rank free search, lnL -10956936.089)
#
# Expected result: GTR+R4 — test() misses SYM+G4 due to BIC pruning.
# This is NOT a dispatch bug — it is the known test() vs evaluateAll() difference
# documented as Issue 6 (commit 1ac3c0a8).
#
#PBS -N iq-xlarge-fixedtree-baseline
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
# OLD binary: iqtree3-3.1.2, R2+AVX512, NO MF2 patches, uses test() for ModelFinder
OLD_PROJECT_DIR="/scratch/${PROJECT}/${USER_ID}/iqtree3-3.1.2"
BUILD_DIR="${BUILD_DIR:-${OLD_PROJECT_DIR}/build-profiling-mpi}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
# Dataset is in the MF2 project dir (same file, sha256-verified)
PROJECT_DIR="/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"
FIXED_TREE="${PROJECT_DIR}/gadi-ci/fixed_xlarge_mf2_tree.nwk"

DATASET_NAME="xlarge_mf"
DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
NRANKS=1
OMP_PER_RANK=104
SEED=1
LABEL="${DATASET_NAME}_${OMP_PER_RANK}t_oldbinary_mpi${NRANKS}x${OMP_PER_RANK}_fixedtree_testpath"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7        2>/dev/null || true
    module load intel-compiler-llvm  2>/dev/null || true
fi

[[ -f "${DATA_PATH}" ]]   || { echo "ERROR: dataset not found: ${DATA_PATH}" >&2; exit 2; }
[[ -f "${FIXED_TREE}" ]]  || { echo "ERROR: fixed tree not found: ${FIXED_TREE}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }
[[ -x "${IQTREE}" ]]      || { echo "ERROR: ${IQTREE} not found — old binary path: ${BUILD_DIR}" >&2; exit 5; }
# Old binary (iqtree3-3.1.2) may link libgomp (gcc OMP) — do NOT check for libiomp5
ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)' || { echo "ERROR: not an MPI binary." >&2; exit 6; }
echo "[preflight] old binary: ${IQTREE}"
echo "[preflight] MF code path: test() — np=1 MPI without openmp_by_model=true"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FIXED-TREE BASELINE — OLD binary (test() path), 1 rank"
echo "║  binary:     ${IQTREE}"
echo "║  dataset:    ${DATA_PATH}"
echo "║  fixed tree: ${FIXED_TREE}"
echo "║  MF path:    test() — BIC-pruning, may skip better models"
echo "║  ranks:      ${NRANKS} × ${OMP_PER_RANK} OMP"
echo "║  seed:       ${SEED}"
echo "║  expected:   GTR+R4 (test() misses SYM+G4 — Issue 6)"
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
        -te "${FIXED_TREE}" \
        -T ${OMP_PER_RANK} \
        -seed ${SEED} \
        --prefix "${WORK_DIR}/iqtree_run" \
    2>&1 | tee "${WORK_DIR}/iqtree_run.log"

T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

BEST=$(grep "Best-fit model" "${WORK_DIR}/iqtree_run.log" | tail -1 || echo "(not found)")
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  FIXED-TREE BASELINE RESULT  (OLD binary — test() code path)"
echo "  ${BEST}"
echo "  Expected: GTR+R4   (test() with BIC pruning)"
echo "  Compare:  SYM+G4   (evaluateAll() — MF2 binary 1-rank, PBS 168004710)"
echo "  Compare:  SYM+G4   (evaluateAll() + dispatch — MF2 binary 2-rank, PBS 168004711)"
echo "  Wall time: ${ELAPSED}s"
echo "  NOTE: GTR+R4 vs SYM+G4 is Issue 6 (test() vs evaluateAll()), NOT a dispatch bug"
echo "══════════════════════════════════════════════════════════════"
