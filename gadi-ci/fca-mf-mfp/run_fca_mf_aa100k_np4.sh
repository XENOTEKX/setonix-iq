#!/bin/bash
# run_fca_mf_aa100k_np4.sh — FCA -m MF parity: np=1 baseline + np=4 parallel, AA 100K.
#
# PURPOSE: Verify -m MF (FreeRate models) FCA parity at np=4 vs np=1.
# Runs np=1 first on HOST[0] (baseline), then np=4 across all 4 nodes.
# Checks: |lnL(np=4) − lnL(np=1)| < 1.0 and best-model agreement.
#
# Binary:  iqtree3-mpi-fca-ws-a2  md5 1547a906f1f75422514b0a0cdf2bc89e
#          (Phase A.2 warm-start broadcast, FCA only — no ATMD/Mode-P)
# 4 × normalsr SPR nodes, 1 rank per node, 103 OMP threads per rank.

#PBS -N fca-mf-aa100k-np4
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l place=excl
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────
PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-a2}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
MODEL_FLAG="MF"
EXPECTED_MD5="1547a906f1f75422514b0a0cdf2bc89e"
LNL_TOL=1.0

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="fca_mf_aa100k_np4_par_seed${SEED}"
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
[[ "${MD5_ACTUAL}" == "${EXPECTED_MD5}" ]] || \
    echo "WARNING: md5 mismatch: ${MD5_ACTUAL} vs ${EXPECTED_MD5}" >&2

mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -ge "${NRANKS}" ]] || { echo "ERROR: expected >=${NRANKS} nodes, got ${#HOSTS[@]}" >&2; exit 9; }

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE_NP4="${WORK_DIR}/rankfile_np4.txt"
cat > "${RANKFILE_NP4}" <<EOF
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
echo "║  FCA -m MF parity: AA 100K  np=1 baseline + np=4 parallel    ║"
echo "║  binary:    $(basename "${IQTREE}")  md5: ${MD5_ACTUAL}"
echo "║  nodes:     ${HOSTS[0]} ${HOSTS[1]} ${HOSTS[2]} ${HOSTS[3]}"
echo "║  model:     -m ${MODEL_FLAG}  seed=${SEED}  lnL_tol=${LNL_TOL}"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Sub-run BASE: np=1 ───────────────────────────────────────────────────
BASE_DIR="${WORK_DIR}/base_np1"; mkdir -p "${BASE_DIR}"
echo ""
echo "── Sub-run BASE (np=1, -m ${MODEL_FLAG}) ──────────────────────────────────"
BASE_PROFILE="${BASE_DIR}/perf_report"
START_BASE=$(date +%s)
perf-report --no-mpi --output="${BASE_PROFILE}" \
mpirun -np 1 \
    --host "${HOSTS[0]}" \
    --bind-to none \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m "${MODEL_FLAG}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${BASE_DIR}/iqtree_run" \
    > "${BASE_DIR}/iqtree_run.log" 2>&1
BASE_RC=$?
END_BASE=$(date +%s)
WALL_BASE=$(( END_BASE - START_BASE ))
echo "  BASE exit=${BASE_RC} wall=${WALL_BASE}s"

# ── Sub-run FCA: np=4 ────────────────────────────────────────────────────
FCA_DIR="${WORK_DIR}/fca_np4"; mkdir -p "${FCA_DIR}"
RANK_LOGS="${FCA_DIR}/rank_logs"; mkdir -p "${RANK_LOGS}"
echo ""
echo "── Sub-run FCA (np=4, -m ${MODEL_FLAG}) ───────────────────────────────────"
FCA_PROFILE="${FCA_DIR}/perf_report"
START_FCA=$(date +%s)
perf-report --no-mpi --output="${FCA_PROFILE}" \
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE_NP4}" \
    --report-bindings \
    --output-filename "${RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m "${MODEL_FLAG}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${FCA_DIR}/iqtree_run" \
    > "${FCA_DIR}/iqtree_run.log" 2>&1
FCA_RC=$?
END_FCA=$(date +%s)
WALL_FCA=$(( END_FCA - START_FCA ))
echo "  FCA exit=${FCA_RC} wall=${WALL_FCA}s"

# ── Parse ────────────────────────────────────────────────────────────────
BASE_LNL=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "${BASE_DIR}/iqtree_run.log" 2>/dev/null | tail -1 || echo "")
[[ -z "${BASE_LNL}" ]] && BASE_LNL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${BASE_DIR}/iqtree_run.log" 2>/dev/null | tail -1 || echo "")
BASE_BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${BASE_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")
BASE_MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${BASE_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")

FCA_LNL=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "${FCA_DIR}/iqtree_run.log" 2>/dev/null | tail -1 || echo "")
[[ -z "${FCA_LNL}" ]] && FCA_LNL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${FCA_DIR}/iqtree_run.log" 2>/dev/null | tail -1 || echo "")
FCA_BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${FCA_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")
FCA_MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${FCA_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "══ Parity check: FCA -m ${MODEL_FLAG} AA 100K np=4 vs np=1 ════════════════"
echo "  BASE (np=1): exit=${BASE_RC}  lnL=${BASE_LNL}  best=${BASE_BEST}  MF=${BASE_MF_WALL}s  wall=${WALL_BASE}s"
echo "  FCA  (np=4): exit=${FCA_RC}   lnL=${FCA_LNL}   best=${FCA_BEST}   MF=${FCA_MF_WALL}s   wall=${WALL_FCA}s"

PASS=1
[[ "${BASE_RC}" -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}"; PASS=0; }
[[ "${FCA_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: FCA  rc=${FCA_RC}";  PASS=0; }

if [[ -n "${BASE_LNL}" && -n "${FCA_LNL}" ]]; then
    DLT=$(python3 -c "print(abs(${FCA_LNL} - (${BASE_LNL})))")
    OK=$(python3 -c "print('yes' if ${DLT} <= ${LNL_TOL} else 'no')")
    [[ "${OK}" == "yes" ]] \
        && echo "  ✓ lnL parity |Δ|=${DLT} ≤ ${LNL_TOL}" \
        || { echo "  ✗ FAIL: lnL drift |Δ|=${DLT} > ${LNL_TOL}"; PASS=0; }
else
    echo "  ✗ FAIL: could not parse lnL (BASE='${BASE_LNL}' FCA='${FCA_LNL}')"; PASS=0
fi

if [[ -n "${BASE_BEST}" && -n "${FCA_BEST}" ]]; then
    [[ "${BASE_BEST}" == "${FCA_BEST}" ]] \
        && echo "  ✓ best model: ${FCA_BEST}" \
        || echo "  ⚠ best model mismatch: BASE=${BASE_BEST} FCA=${FCA_BEST}"
fi

if [[ -n "${BASE_MF_WALL}" && -n "${FCA_MF_WALL}" ]]; then
    SPEEDUP=$(python3 -c "print(f'{float(${BASE_MF_WALL})/float(${FCA_MF_WALL}):.2f}x')" 2>/dev/null || echo "?")
    echo "  ✓ MF speedup np=4 vs np=1: ${SPEEDUP}  (${BASE_MF_WALL}s → ${FCA_MF_WALL}s)"
fi

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ PARITY PASS ══"
else
    echo "  ══ PARITY FAIL ══"; exit 10
fi
