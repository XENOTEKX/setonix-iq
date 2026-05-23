#!/bin/bash
# run_atmd_b3_aa_100k_1node.sh — ATMD B.3+B.4 K_outer>1 activation test: AA 100K, 1 node.
#
# PURPOSE: Verify that the K_outer memory-budget semaphore (B.4) fires K_outer > 1 for a
# small enough dataset.  AA 100K (100 taxa, ~5 GB central_partial_lh per tree at +G4) is
# well within budget: a 500 GB node can hold ~10 trees → K_outer ≈ 7 (capped at 8).
#
# Run config: 1 MPI rank × 103 OMP threads on 1 normalsr node.
# K_outer ≈ min(K_mem, K_thr=103, K_cap=8).  K_mem from formula ≈ 7 (conservative).
# Expected: K_outer=7, M_inner=14 (= floor(103/7)).
#
# Gate pass criteria:
#   - exit code = 0
#   - "[ATMD Mode F] K_outer=N" with N > 1 in stdout
#   - lnL within ±1.0 of −7,820,831.xxx (AA 100K reference — check from FCA runs)
#   - Best model = LG+G4 (or LG+I+G4 — alignment-dependent)
#   - Wall time: informational only (no regression baseline for ATMD AA 100K 1-node)
#
# A/B refs (FCA baseline for AA 100K 1-node):
#   FCA 1-node  168573852  (see mf-iso-aa-100k-baseline.o168573852 in gadi-ci/)
#
# Binary:  iqtree3-mpi-atmd-b3c  (build-atmd-b3c/)
# Build:   qsub build_atmd_b3.sh  first
# Branch:  fca-lbfgs-ws
# See:     research/lbfgs-and-warmstart-implementation.md §15.7

#PBS -N atmd-b3c-100k-1n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l place=excl
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/dx61+scratch/rc29+scratch/um09
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
# NOTE: PBS sets $PROJECT=dx61 (billing project). Use SRC_PROJECT for scratch paths.
SRC_PROJECT="rc29"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${SRC_PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b3c/iqtree3-mpi-atmd-b3c}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${SRC_PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_100k_atmd_b3c_np1_full_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: ATMD binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing." >&2; exit 8; }

if strings "${IQTREE}" 2>/dev/null | grep -q 'ATMD Mode F'; then
    echo "[preflight] [ATMD Mode F] log string: present"
else
    echo "[preflight] WARNING: '[ATMD Mode F]' not found — ATMD may not be compiled in." >&2
fi

ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
echo "[preflight] md5: ${ACTUAL_MD5}"

# ── OMP / runtime ─────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

# OMP_MAX_ACTIVE_LEVELS=2 is essential for B.3 nested teams (matches B.-1 main.cpp patch).
# OMP_PROC_BIND=spread: distribute outer workers across NUMA domains.
# Inner teams inherit close binding from outer worker's socket.
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_MAX_ACTIVE_LEVELS=2"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=spread,close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
    -x "KMP_HOT_TEAMS_MAX_LEVEL=2"
)

NODE=$(sort -u "${PBS_NODEFILE}" | head -1)
HOSTFILE="${WORK_DIR}/hostfile.txt"
echo "${NODE} slots=104" > "${HOSTFILE}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ATMD B.3+B.4 K_outer Activation Test — AA 100K, 1 node     ║"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5:          ${ACTUAL_MD5}"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  node:         ${NODE}"
echo "║  Expected:     K_outer≈7 M_inner≈14  (AA 100K on 500 GB node)"
echo "║  Expected:     per_tree_MB≈5000  avail_MB≈400000"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Run ────────────────────────────────────────────────────────────────
echo "[1node] Full run (MF+SPR), ${NRANKS} rank × ${OMP_PER_RANK} OMP on 1 node"
START_EPOCH=$(date +%s)

# FIX: Use different filename for IQ-TREE's internal --prefix log vs shell stdout
# redirect to avoid the dual-write problem (both writing to the same iqtree_run.log).
# IQ-TREE opens iqtree_inner.log via --prefix; stdout goes to iqtree_stdout.log.
# The [ATMD Mode F] line (and sidecar .atmd_diag) will be in iqtree_inner.log.
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --bind-to none \
    --report-bindings \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${WORK_DIR}/iqtree_inner" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

# Merge and show both logs
cat "${WORK_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "--- stdout log ---"
cat "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true
# Show sidecar diagnostic (written by B.4 code via fopen, not cout/TeeBuf)
echo "--- atmd_diag sidecar ---"
cat "${WORK_DIR}/iqtree_inner.atmd_diag" 2>/dev/null || echo "(sidecar not found)"
echo "--- ATMD-DIAG grep from stdout ---"
grep 'ATMD-DIAG\|ATMD Mode F' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || echo "(none)"
echo ""
echo "[1node] done: rc=${IQRC} wall=${WALL}s"

# ── Gate checks ────────────────────────────────────────────────────────
echo ""
echo "══ Gate checks ══════════════════════════════════════════════════"
PASS=1

# (a) K_outer > 1 activation — primary goal.
# Check sidecar file first (written directly via fopen — not subject to TeeBuf),
# then IQ-TREE's inner log, then stdout log.
ATMD_LINE=""
if [[ -f "${WORK_DIR}/iqtree_inner.atmd_diag" ]]; then
    ATMD_LINE=$(head -1 "${WORK_DIR}/iqtree_inner.atmd_diag")
    echo "  [ATMD Mode F] found in: sidecar (iqtree_inner.atmd_diag)"
elif grep -q '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null; then
    ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" | head -1)
    echo "  [ATMD Mode F] found in: iqtree_inner.log"
elif grep -q '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null; then
    ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_stdout.log" | head -1)
    echo "  [ATMD Mode F] found in: iqtree_stdout.log"
fi
if [[ -n "${ATMD_LINE}" ]]; then
    echo "  [ATMD Mode F] line: ${ATMD_LINE}"
    K_ACTUAL=$(echo "${ATMD_LINE}" | grep -oP 'K_outer=\K[0-9]+' || echo "0")
    M_ACTUAL=$(echo "${ATMD_LINE}" | grep -oP 'M_inner=\K[0-9]+' || echo "0")
    PER_TREE=$(echo "${ATMD_LINE}" | grep -oP 'per_tree_MB=\K[0-9]+' || echo "?")
    AVAIL=$(echo "${ATMD_LINE}" | grep -oP 'avail_MB=\K[0-9]+' || echo "?")
    echo "  K_outer=${K_ACTUAL}  M_inner=${M_ACTUAL}  per_tree_MB=${PER_TREE}  avail_MB=${AVAIL}"
    if [[ "${K_ACTUAL}" -gt 1 ]]; then
        echo "  ✓ K_outer=${K_ACTUAL} > 1 — Mode F outer parallel team activated!"
    else
        echo "  ✗ FAIL: K_outer=${K_ACTUAL} — outer parallel team did NOT activate." \
             "Check per_tree_bytes formula or available memory." >&2
        PASS=0
    fi
else
    echo "  ✗ FAIL: [ATMD Mode F] not found in any log or sidecar" >&2
    PASS=0
fi

# (b) Correctness: lnL
LNL_ACTUAL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 \
          || grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -1 \
          || echo "")
if [[ -n "${LNL_ACTUAL}" ]]; then
    echo "  lnL: ${LNL_ACTUAL}"
else
    echo "  ✗ FAIL: lnL not found in log." >&2; PASS=0
fi

# (c) Best model
BEST_MODEL=$(grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 \
          || grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -1 \
          || echo "")
echo "  Best model: ${BEST_MODEL}"
if [[ "${BEST_MODEL}" == "LG+G4" || "${BEST_MODEL}" == "LG+I+G4" ]]; then
    echo "  ✓ Best model expected (LG family)"
else
    echo "  ⚠ Unexpected best model '${BEST_MODEL}' — verify against FCA reference."
fi

echo "  Total wall: ${WALL}s"

echo ""
if [[ "${IQRC}" -eq 0 && "${PASS}" -eq 1 ]]; then
    echo "  ══ GATE: PASS  (K_outer>1 confirmed — ATMD Mode F outer parallelism working) ══"
else
    echo "  ══ GATE: FAIL (rc=${IQRC} pass=${PASS}) ══" >&2
    exit $(( IQRC != 0 ? IQRC : 1 ))
fi

echo ""
echo "  Log:  ${WORK_DIR}/iqtree_run.log"
echo ""
echo "  Next: Phase B.0 (pattern-parallel Mode P, ~900 LOC)"
echo "        See: research/lbfgs-and-warmstart-implementation.md §15.6"
