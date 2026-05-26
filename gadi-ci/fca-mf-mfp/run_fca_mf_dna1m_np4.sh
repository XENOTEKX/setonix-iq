#!/bin/bash
# run_fca_mf_dna1m_np4.sh — FCA -m MF scaling: np=4, DNA 1M.
#
# PURPOSE: Measure FCA ModelFinder performance with -m MF at np=4 for DNA 1M.
# Compare lnL and best-model against np=1 baseline run (separate job).
# Binary:  iqtree3-mpi-fca-ws-a2  md5 1547a906f1f75422514b0a0cdf2bc89e

#PBS -N fca-mf-dna1m-np4
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l place=excl
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-a2}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
MODEL_FLAG="MF"
EXPECTED_MD5="1547a906f1f75422514b0a0cdf2bc89e"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="fca_mf_dna1m_np4_seed${SEED}"
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
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing" >&2; exit 8; }
MD5_ACTUAL="$(md5sum "${IQTREE}" | awk '{print $1}')"
[[ "${MD5_ACTUAL}" == "${EXPECTED_MD5}" ]] || echo "WARNING: md5 mismatch: ${MD5_ACTUAL}" >&2
echo "INFO: binary md5=${MD5_ACTUAL}"

mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -ge "${NRANKS}" ]] || { echo "ERROR: expected >=${NRANKS} nodes, got ${#HOSTS[@]}" >&2; exit 9; }

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
{
    for i in $(seq 0 $(( NRANKS - 1 ))); do
        echo "rank ${i}=${HOSTS[${i}]} slot=0-103"
    done
} > "${RANKFILE}"

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
echo "║  FCA -m MF scaling: DNA 1M np=4  seed=${SEED}"
echo "║  binary:    $(basename "${IQTREE}")  md5: ${MD5_ACTUAL}"
echo "║  alignment: $(basename "${ALIGNMENT}")"
echo "║  work_dir:  ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"

RANK_LOGS="${WORK_DIR}/rank_logs"; mkdir -p "${RANK_LOGS}"

PROFILE_REPORT="${WORK_DIR}/perf_report"
START=$(date +%s)
perf-report --no-mpi --output="${PROFILE_REPORT}" \
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
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
echo "══ FCA -m MF DNA 1M np=4 result ══════════════════════"
echo "  exit code:  ${RC}"
echo "  lnL:        ${LNL}"
echo "  best model: ${BEST}"
echo "  MF wall:    ${MF_WALL}s"
echo "  total wall: ${WALL}s"
echo "  work_dir:   ${WORK_DIR}"
echo "  (compare lnL/best against np=1 reference for parity)"

[[ "${RC}" -eq 0 ]] && echo "  ══ DONE ══" || { echo "  ✗ iqtree3 exited ${RC}"; exit "${RC}"; }
