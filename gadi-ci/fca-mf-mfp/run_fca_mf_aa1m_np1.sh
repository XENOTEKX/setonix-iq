#!/bin/bash
# run_fca_mf_aa1m_np1.sh — FCA -m MF baseline: np=1, AA 1M.
#
# PURPOSE: Establish np=1 reference for -m MF on the AA 1M alignment.
# Expected to be very long-running — used as parity reference for np≥2 jobs.
# Binary:  iqtree3-mpi-fca-ws-a2  md5 1547a906f1f75422514b0a0cdf2bc89e
# Ref (-m TEST np=16 job 168635616): MF=1,122s  lnL=-78,605,196.497

#PBS -N fca-mf-aa1m-np1
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l place=excl
#PBS -l walltime=24:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-a2}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
MODEL_FLAG="MF"
EXPECTED_MD5="1547a906f1f75422514b0a0cdf2bc89e"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="fca_mf_aa1m_np1_seed${SEED}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7              2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
    module load linaro-forge/24.0.2          2>/dev/null || true
fi

[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
MD5_ACTUAL="$(md5sum "${IQTREE}" | awk '{print $1}')"
[[ "${MD5_ACTUAL}" == "${EXPECTED_MD5}" ]] || echo "WARNING: md5 mismatch: ${MD5_ACTUAL}" >&2
echo "INFO: binary md5=${MD5_ACTUAL}"

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FCA -m MF baseline: np=1, AA 1M                             ║"
echo "║  binary:    $(basename "${IQTREE}")  md5: ${MD5_ACTUAL}"
echo "║  model:     -m ${MODEL_FLAG}  seed=${SEED}"
echo "║  NOTE: Long-running baseline — record lnL for np≥2 parity    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

PROFILE_REPORT="${WORK_DIR}/perf_report"
START=$(date +%s)
perf-report --no-mpi --output="${PROFILE_REPORT}" \
mpirun -np "${NRANKS}" --bind-to none "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m "${MODEL_FLAG}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1
RC=$?
WALL=$(( $(date +%s) - START ))

tail -30 "${WORK_DIR}/iqtree_run.log" 2>/dev/null || true

LNL=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | tail -1 || echo "")
[[ -z "${LNL}" ]] && LNL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | tail -1 || echo "")
BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "══ FCA -m MF AA 1M np=1 result ════════════════════════════════"
echo "  exit code:  ${RC}"
echo "  lnL:        ${LNL}"
echo "  best model: ${BEST}"
echo "  MF wall:    ${MF_WALL}s"
echo "  total wall: ${WALL}s"
echo "  work_dir:   ${WORK_DIR}"
echo "  (record lnL above as parity reference for np=2/4/16 runs)"

[[ "${RC}" -eq 0 ]] && echo "  ══ DONE ══" || { echo "  ✗ iqtree3 exited ${RC}"; exit "${RC}"; }
