#!/bin/bash
# run_fca_mfp_aa100k_np4.sh — FCA -m MFP parity: np=1 baseline + np=4 parallel, AA 100K.
#
# PURPOSE: Verify -m MFP (FreeRate + mixture models) FCA parity at np=4 vs np=1.
# Binary:  iqtree3-mpi-fca-ws-a2  md5 1547a906f1f75422514b0a0cdf2bc89e
# 4 × normalsr SPR nodes, 1 rank per node, 103 OMP threads per rank.

#PBS -N fca-mfp-aa100k-np4
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l place=excl
#PBS -l walltime=12:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-a2}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
MODEL_FLAG="MFP"
EXPECTED_MD5="1547a906f1f75422514b0a0cdf2bc89e"
LNL_TOL=1.0

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="fca_mfp_aa100k_np4_par_seed${SEED}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7              2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing" >&2; exit 8; }
MD5_ACTUAL="$(md5sum "${IQTREE}" | awk '{print $1}')"
[[ "${MD5_ACTUAL}" == "${EXPECTED_MD5}" ]] || echo "WARNING: md5 mismatch" >&2

mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -ge "${NRANKS}" ]] || { echo "ERROR: expected >=${NRANKS} nodes" >&2; exit 9; }

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
rank 1=${HOSTS[1]} slot=0-103
rank 2=${HOSTS[2]} slot=0-103
rank 3=${HOSTS[3]} slot=0-103
EOF

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
echo "║  FCA -m MFP parity: AA 100K  np=1 baseline + np=4 parallel   ║"
echo "║  binary:    $(basename "${IQTREE}")  md5: ${MD5_ACTUAL}"
echo "║  model:     -m ${MODEL_FLAG}  seed=${SEED}"
echo "╚══════════════════════════════════════════════════════════════╝"

BASE_DIR="${WORK_DIR}/base_np1"; mkdir -p "${BASE_DIR}"
echo "── Sub-run BASE (np=1, -m ${MODEL_FLAG}) ──────────────────────────────────"
START_BASE=$(date +%s)
mpirun -np 1 --host "${HOSTS[0]}" --bind-to none "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m "${MODEL_FLAG}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${BASE_DIR}/iqtree_run" \
    > "${BASE_DIR}/iqtree_run.log" 2>&1
BASE_RC=$?; WALL_BASE=$(( $(date +%s) - START_BASE ))
echo "  BASE exit=${BASE_RC} wall=${WALL_BASE}s"

FCA_DIR="${WORK_DIR}/fca_np4"; mkdir -p "${FCA_DIR}"
RANK_LOGS="${FCA_DIR}/rank_logs"; mkdir -p "${RANK_LOGS}"
echo "── Sub-run FCA (np=4, -m ${MODEL_FLAG}) ───────────────────────────────────"
START_FCA=$(date +%s)
mpirun -np "${NRANKS}" --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" -rf "${RANKFILE}" \
    --report-bindings --output-filename "${RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m "${MODEL_FLAG}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${FCA_DIR}/iqtree_run" \
    > "${FCA_DIR}/iqtree_run.log" 2>&1
FCA_RC=$?; WALL_FCA=$(( $(date +%s) - START_FCA ))
echo "  FCA exit=${FCA_RC} wall=${WALL_FCA}s"

_parse_lnl() { local f="$1"
    local v; v=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "$f" 2>/dev/null | tail -1 || echo "")
    [[ -z "$v" ]] && v=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "$f" 2>/dev/null | tail -1 || echo "")
    echo "$v"; }
_parse_best() { grep -oP 'Best-fit model.*?:\s*\K\S+' "$1" 2>/dev/null | head -1 || echo ""; }
_parse_mfwall() { grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "$1" 2>/dev/null | head -1 || echo ""; }

BASE_LNL=$(_parse_lnl  "${BASE_DIR}/iqtree_run.log")
BASE_BEST=$(_parse_best "${BASE_DIR}/iqtree_run.log")
BASE_MFW=$(_parse_mfwall "${BASE_DIR}/iqtree_run.log")
FCA_LNL=$(_parse_lnl  "${FCA_DIR}/iqtree_run.log")
FCA_BEST=$(_parse_best "${FCA_DIR}/iqtree_run.log")
FCA_MFW=$(_parse_mfwall "${FCA_DIR}/iqtree_run.log")

echo ""
echo "══ Parity: FCA -m ${MODEL_FLAG} AA 100K np=4 vs np=1 ══════════════════════"
echo "  BASE (np=1): rc=${BASE_RC}  lnL=${BASE_LNL}  best=${BASE_BEST}  MF=${BASE_MFW}s"
echo "  FCA  (np=4): rc=${FCA_RC}   lnL=${FCA_LNL}   best=${FCA_BEST}   MF=${FCA_MFW}s"
PASS=1
[[ "${BASE_RC}" -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}"; PASS=0; }
[[ "${FCA_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: FCA  rc=${FCA_RC}";  PASS=0; }
if [[ -n "${BASE_LNL}" && -n "${FCA_LNL}" ]]; then
    DLT=$(python3 -c "print(abs(${FCA_LNL} - (${BASE_LNL})))")
    OK=$(python3 -c "print('yes' if ${DLT} <= ${LNL_TOL} else 'no')")
    [[ "${OK}" == "yes" ]] && echo "  ✓ lnL |Δ|=${DLT}" || { echo "  ✗ lnL drift |Δ|=${DLT}"; PASS=0; }
else echo "  ✗ FAIL: parse error"; PASS=0; fi
[[ -n "${BASE_BEST}" && -n "${FCA_BEST}" ]] && \
    { [[ "${BASE_BEST}" == "${FCA_BEST}" ]] && echo "  ✓ best=${FCA_BEST}" || \
      echo "  ⚠ model mismatch BASE=${BASE_BEST} FCA=${FCA_BEST}"; }
[[ -n "${BASE_MFW}" && -n "${FCA_MFW}" ]] && \
    echo "  ✓ MF speedup: $(python3 -c "print(f'{float(${BASE_MFW})/float(${FCA_MFW}):.2f}x')")  (${BASE_MFW}s→${FCA_MFW}s)"
[[ "${PASS}" -eq 1 ]] && echo "  ══ PARITY PASS ══" || { echo "  ══ PARITY FAIL ══"; exit 10; }
