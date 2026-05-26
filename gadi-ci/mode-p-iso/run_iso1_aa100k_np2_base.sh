#!/bin/bash
# run_iso1_aa100k_np2_base.sh — ISO-1 gate: P.ISO baseline binary, np=2, AA 100K.
#
# PURPOSE: Confirm P.2 partition wiring fires at np=2 (the [Mode P] cerr line
# is emitted per rank) BUT the kernel still computes the full likelihood per
# rank (because the baseline binary has NO P.3 kernel patches). lnL must
# therefore match FCA np=2 — Mode P partition is set but inert.
#
# Reference:
#   FCA np=2 (job 168584736):  lnL=-7,541,976.853
#
# Gate pass criteria:
#   - exit code = 0
#   - [Mode P] rank 0 ptn=[0, X) of N + [Mode P] rank 1 ptn=[X, N) of N emitted
#   - lnL within 1e-6 of -7,541,976.853 (kernel ignores partition; same as FCA)
#   - Best model = LG+G4

#PBS -N iso1-base-np2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1000GB
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

NRANKS=2
OMP_PER_RANK="${OMP_PER_RANK:-103}"  # one full 104-core SPR node per rank
SEED="${SEED:-1}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso1_base_aa100k_np2_seed${SEED}"
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

# Confirm this is the base build
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

RANK_LOGS_DIR="${WORK_DIR}/rank_logs"; mkdir -p "${RANK_LOGS_DIR}"

# B.4-3 lesson: 2 MPI ranks on 1 node + OMP_PROC_BIND=close crashes rank 1 (Intel OMP
# libiomp5 fights over core affinity). Use 2 dedicated nodes with a rankfile instead.
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing" >&2; exit 8; }
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -ge 2 ]] || { echo "ERROR: expected >=2 nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"
RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
rank 1=${HOSTS[1]} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ISO-1 gate: P.ISO baseline, np=2, AA 100K, --mode-p-all     ║"
echo "║  binary:    $(basename "${IQTREE}")  md5:${MD5}"
echo "║  work_dir:  ${WORK_DIR}"
echo "║  Expected:  [Mode P] lines emitted; lnL = FCA np=2 (unchanged)"
echo "║  Ref lnL:   -7,541,976.853 (FCA np=2 job 168584736)"
echo "╚══════════════════════════════════════════════════════════════╝"

START=$(date +%s)
# B.4-2 lesson: separate --prefix and stdout redirect filenames.
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${RANK_LOGS_DIR}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --mode-p-all \
                --atmd-k-outer 1 \
                --prefix "${WORK_DIR}/iqtree_inner" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
END=$(date +%s)
WALL=$(( END - START ))

echo "--- inner log tail ---"
tail -30 "${WORK_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "--- stdout tail ---"
tail -15 "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true
echo "--- [Mode P] lines ---"
grep '\[Mode P\]' "${WORK_DIR}/iqtree_inner.log" "${WORK_DIR}/iqtree_stdout.log" "${RANK_LOGS_DIR}"/*/stderr 2>/dev/null | head -5 || true

MODE_P_COUNT=$({ grep -h '\[Mode P\]' "${WORK_DIR}/iqtree_inner.log" "${WORK_DIR}/iqtree_stdout.log" "${RANK_LOGS_DIR}"/*/stderr 2>/dev/null; true; } | wc -l)
LNL=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
[[ -z "${LNL}" ]] && LNL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "══ ISO-1 result ═════════════════════════════════════════════════"
echo "  exit code:    ${IQRC}"
echo "  Mode P lines: ${MODE_P_COUNT}    (expected ≥1 per model × 2 ranks)"
echo "  lnL:          ${LNL}"
echo "  ref lnL:      -7541976.853 (FCA np=2)"
echo "  best model:   ${BEST}"
echo "  MF wall:      ${MF_WALL}s"
echo "  total wall:   ${WALL}s"

PASS=1
[[ "${IQRC}" -eq 0 ]] || { echo "  ✗ FAIL: rc=${IQRC}"; PASS=0; }
[[ "${MODE_P_COUNT}" -gt 0 ]] || { echo "  ✗ FAIL: no [Mode P] line emitted at np=2"; PASS=0; }
if [[ -n "${LNL}" ]]; then
    DLT=$(python3 -c "print(abs(${LNL} - (-7541976.853)))")
    OK=$(python3 -c "print('yes' if ${DLT} <= 1e-3 else 'no')")
    [[ "${OK}" == "yes" ]] && echo "  ✓ lnL parity (|Δ|=${DLT})" || { echo "  ✗ FAIL: lnL drift |Δ|=${DLT}"; PASS=0; }
else
    echo "  ✗ FAIL: lnL not parsed"; PASS=0
fi
[[ "${BEST}" == "LG+G4" ]] && echo "  ✓ best model LG+G4" || echo "  ⚠ best model: ${BEST}"

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-1 PASS — proceed to qsub build_mode_p_iso_p3.sh ══"
else
    echo "  ══ ISO-1 FAIL — investigate before applying P.3 ══"; exit 10
fi
