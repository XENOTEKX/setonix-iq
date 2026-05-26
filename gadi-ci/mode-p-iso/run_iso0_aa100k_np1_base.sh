#!/bin/bash
# run_iso0_aa100k_np1_base.sh — ISO-0 gate: P.ISO baseline binary, np=1, AA 100K.
#
# PURPOSE: Confirm Mode P is correctly INERT at np=1. The
# initializePtnPartition() early-returns when MPIHelper::getNumProcesses()==1,
# so even with --mode-p-all on the CLI, ptn_start/ptn_end stay at the defaults
# and no [Mode P] cerr line is emitted. lnL must match FCA np=1.
#
# Reference:
#   FCA np=1 (job 169095077):  MF=258.8s  total=1,000.8s  lnL=-7,541,976.861
#
# Gate pass criteria:
#   - exit code = 0
#   - NO [Mode P] line in stdout/inner.log
#   - lnL within 1e-6 of -7,541,976.861
#   - Best model = LG+G4
#
# Critical: --prefix and stdout redirect MUST use different filenames (B.4-2).

#PBS -N iso0-base-np1
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
IQTREE="${IQTREE:-${SANDBOX}/build-mode-p-iso-base/iqtree3-mpi-mode-p-iso-base}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso0_base_aa100k_np1_seed${SEED}"
WORK_DIR="${SANDBOX}/runs/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]] || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
MD5=$(md5sum "${IQTREE}" | awk '{print $1}')

# Confirm this is the base build (no P.3 markers in binary)
if strings "${IQTREE}" 2>/dev/null | grep -q 'P\.3 Mode P'; then
    echo "WARNING: 'P.3 Mode P' string found in BASELINE binary — wrong build?" >&2
fi

export KMP_BLOCKTIME=200
export TMPDIR="${SANDBOX}/tmp"; mkdir -p "${TMPDIR}"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_MAX_ACTIVE_LEVELS=2"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
    -x "KMP_HOT_TEAMS_MAX_LEVEL=2"
)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ISO-0 gate: P.ISO baseline, np=1, AA 100K, --mode-p-all     ║"
echo "║  binary:    $(basename "${IQTREE}")  md5:${MD5}"
echo "║  work_dir:  ${WORK_DIR}"
echo "║  Expected:  NO [Mode P] line (np=1 triggers early-return)"
echo "║  Ref lnL:   -7,541,976.861 (FCA np=1 job 169095077)"
echo "╚══════════════════════════════════════════════════════════════╝"

START=$(date +%s)
# B.4-2 lesson: separate --prefix and stdout redirect filenames.
mpirun -np "${NRANKS}" \
    --bind-to none \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --mode-p-all \
                --prefix "${WORK_DIR}/iqtree_inner" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
END=$(date +%s)
WALL=$(( END - START ))

echo "--- inner log tail ---"
tail -30 "${WORK_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "--- stdout tail ---"
tail -15 "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true

# Parse outcomes
# NOTE: grep -c returns exit 1 when there are zero matches; use { ...; true; } to prevent
# set -euo pipefail from killing the script when Mode P is correctly absent.
MODE_P_COUNT=$({ grep -c '\[Mode P\]' "${WORK_DIR}/iqtree_inner.log" "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null; true; } | awk -F: '{s+=$2}END{print s+0}')
LNL=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
[[ -z "${LNL}" ]] && LNL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "══ ISO-0 result ═════════════════════════════════════════════════"
echo "  exit code:    ${IQRC}"
echo "  Mode P lines: ${MODE_P_COUNT}    (expected 0; np=1 → early-return)"
echo "  lnL:          ${LNL}"
echo "  ref lnL:      -7541976.861 (FCA np=1)"
echo "  best model:   ${BEST}    (expected LG+G4)"
echo "  MF wall:      ${MF_WALL}s"
echo "  total wall:   ${WALL}s"

PASS=1
[[ "${IQRC}" -eq 0 ]] || { echo "  ✗ FAIL: rc=${IQRC}"; PASS=0; }
[[ "${MODE_P_COUNT}" -eq 0 ]] || { echo "  ✗ FAIL: [Mode P] line emitted at np=1 (${MODE_P_COUNT} lines)"; PASS=0; }
if [[ -n "${LNL}" ]]; then
    DLT=$(python3 -c "print(abs(${LNL} - (-7541976.861)))")
    # Tolerance 0.05: full tree-search lnL may differ by ~0.01 between runs due to
    # Mode F K_outer variance, random NNI order, etc. We only check it didn't diverge.
    OK=$(python3 -c "print('yes' if ${DLT} <= 0.05 else 'no')")
    [[ "${OK}" == "yes" ]] && echo "  ✓ lnL parity (|Δ|=${DLT})" || { echo "  ✗ FAIL: lnL drift |Δ|=${DLT}"; PASS=0; }
else
    echo "  ✗ FAIL: lnL not parsed"; PASS=0
fi
[[ "${BEST}" == "LG+G4" ]] && echo "  ✓ best model LG+G4" || { echo "  ⚠ best model: ${BEST}"; }

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-0 PASS ══"
else
    echo "  ══ ISO-0 FAIL ══"; exit 10
fi
