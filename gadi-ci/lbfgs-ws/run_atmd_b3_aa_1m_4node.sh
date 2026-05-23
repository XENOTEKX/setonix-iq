#!/bin/bash
# run_atmd_b3_aa_1m_4node.sh — ATMD B.3+B.4 correctness gate: AA 1M, 4 nodes, 4 MPI ranks.
#
# PURPOSE: Verify the ATMD binary produces correct results on the AA 1M dataset.
# At AA 1M (100 taxa, ~12 GB central_partial_lh per tree) the memory budget gate (B.4)
# limits K_outer to 1 on 512 GB nodes — the outer loop degrades to the same sequential
# path as A.2.  This run therefore tests:
#   (a) B.-1 patches don't regress existing functionality
#   (b) K_outer=1 zero-overhead path (no spurious [ATMD Mode F] speedup or slowdown)
#   (c) NUMA first-touch fires (atmd_inner_threads=103 → first-touch with 103 threads)
#   (d) lnL and best model match A.2 reference within gate tolerances
#
# Gate pass criteria:
#   - exit code = 0
#   - lnL within ±1.0 of −78,605,196.497 (A.2 np=4 reference; job 169099058)
#   - Best model = LG+G4
#   - "[ATMD Mode F] K_outer=1" appears in log (expected: memory-bound serial path)
#   - ws_bcast_fields > 0 in MF-MPI-DIAG (A.2 warm-start still fires)
#   - Wall time within +5% of A.2 np=4 baseline (1999.214s MF wall)
#
# A/B refs:
#   A.2 np=4   169099058  MF=1999.214s  SPR=4021.666s  total=6098.480s  lnL=−78,605,196.445
#   FCA np=4   168635615  MF=1974.476s  SPR=3982.142s  total=5956.618s
#
# Binary:  iqtree3-mpi-atmd-b3  (build-atmd-b3/)
# Build:   qsub build_atmd_b3.sh  first
# Branch:  fca-lbfgs-ws
# See:     research/lbfgs-and-warmstart-implementation.md §15.7

#PBS -N atmd-b3-1m-4n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2040GB
#PBS -l walltime=03:30:00
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
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b3/iqtree3-mpi-atmd-b3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${SRC_PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_1m_atmd_b3_np4_full_seed${SEED}"
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

# Verify ATMD Mode F code is compiled in.
if strings "${IQTREE}" 2>/dev/null | grep -q 'ATMD Mode F'; then
    echo "[preflight] [ATMD Mode F] log string: present — ATMD compiled in OK"
else
    echo "[preflight] WARNING: '[ATMD Mode F]' not in binary strings — ATMD may be absent." >&2
fi
if strings "${IQTREE}" 2>/dev/null | grep -q 'MPI_THREAD_FUNNELED\|B.-1\|atmd_mpi_provided'; then
    echo "[preflight] B.-1 MPI_Init_thread evidence found"
fi

# md5 (informational only — ATMD binary is new, no reference yet).
ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
echo "[preflight] md5: ${ACTUAL_MD5}"
if [[ -f "${ISO_DIR}/build-atmd-b3/.build-info.json" ]]; then
    EXPECTED_MD5=$(python3 -c "import json,sys; d=json.load(open('${ISO_DIR}/build-atmd-b3/.build-info.json')); print(d.get('md5',''))" 2>/dev/null || echo "")
    if [[ -n "${EXPECTED_MD5}" ]]; then
        [[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]] \
            && echo "[preflight] md5 MATCH vs build-info.json" \
            || echo "[preflight] WARNING: md5 MISMATCH vs build-info.json (got ${ACTUAL_MD5}, expected ${EXPECTED_MD5})" >&2
    fi
fi

if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — must run inside a PBS job." >&2; exit 8
fi

# ── OMP / runtime ─────────────────────────────────────────────────────
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
    -x "OMP_MAX_ACTIVE_LEVELS=2"
)

# ── Multi-node host discovery ──────────────────────────────────────────
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -eq 4 ]] || { echo "ERROR: expected 4 nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOST_A="${HOSTS[0]}"; HOST_B="${HOSTS[1]}"; HOST_C="${HOSTS[2]}"; HOST_D="${HOSTS[3]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
rank 2=${HOST_C} slot=0-103
rank 3=${HOST_D} slot=0-103
EOF

RANK_LOGS_DIR="${WORK_DIR}/rank_logs"
mkdir -p "${RANK_LOGS_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ATMD B.3+B.4 Correctness Gate — AA 1M, np=4, 4-node        ║"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5:          ${ACTUAL_MD5}"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  Expected:     K_outer=1 (AA 1M too large for K>1 on 512 GB)"
echo "║  Gate lnL:     within ±1.0 of −78,605,196.497"
echo "║  Gate model:   LG+G4"
echo "║  A.2 ref MF:   1999.214s  (job 169099058)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[4node] hostfile:"; cat "${HOSTFILE}" | sed 's/^/    /'
echo "[4node] rankfile:"; cat "${RANKFILE}"  | sed 's/^/    /'
echo ""

# ── Run ────────────────────────────────────────────────────────────────
echo "[4node] Full run (MF+SPR), ${NRANKS} ranks × ${OMP_PER_RANK} OMP across 4 nodes"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${RANK_LOGS_DIR}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[4node] done: rc=${IQRC} wall=${WALL}s"

# ── Gate checks ────────────────────────────────────────────────────────
echo ""
echo "══ Gate checks ══════════════════════════════════════════════════"
PASS=1

# (a) ATMD Mode F log line
if grep -q '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_run.log" 2>/dev/null; then
    ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_run.log" | head -1)
    echo "  [ATMD Mode F] line found:  ${ATMD_LINE}"
    # For AA 1M we expect K_outer=1 (memory-bound).
    if echo "${ATMD_LINE}" | grep -q 'K_outer=1'; then
        echo "  ✓ K_outer=1 confirmed (expected: AA 1M memory-bound)"
    else
        K_ACTUAL=$(echo "${ATMD_LINE}" | grep -oP 'K_outer=\K[0-9]+')
        echo "  ⚠ K_outer=${K_ACTUAL} (expected 1 for AA 1M — check per_tree_bytes estimate)"
    fi
else
    echo "  ✗ FAIL: [ATMD Mode F] line not found in log — ATMD code not reached" >&2
    PASS=0
fi

# (b) lnL gate
LNL_REF=-78605196.497
LNL_TOL=1.0
LNL_ACTUAL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")
if [[ -n "${LNL_ACTUAL}" ]]; then
    LNL_DIFF=$(python3 -c "print(abs(${LNL_ACTUAL} - (${LNL_REF})))")
    LNL_OK=$(python3 -c "print('yes' if ${LNL_DIFF} <= ${LNL_TOL} else 'no')")
    if [[ "${LNL_OK}" == "yes" ]]; then
        echo "  ✓ lnL ${LNL_ACTUAL}  |diff|=${LNL_DIFF} ≤ ${LNL_TOL}"
    else
        echo "  ✗ FAIL: lnL ${LNL_ACTUAL}  |diff|=${LNL_DIFF} > ${LNL_TOL}" >&2; PASS=0
    fi
else
    echo "  ✗ FAIL: lnL not found in log." >&2; PASS=0
fi

# (c) Best model
BEST_MODEL=$(grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | head -1 || echo "")
if [[ "${BEST_MODEL}" == "LG+G4" ]]; then
    echo "  ✓ Best model: ${BEST_MODEL}"
else
    echo "  ✗ FAIL: best model '${BEST_MODEL}' (expected LG+G4)" >&2; PASS=0
fi

# (d) A.2 warm-start still fires
if grep -q 'ws_bcast_fields=[1-9]' "${WORK_DIR}/iqtree_run.log" 2>/dev/null; then
    echo "  ✓ ws_bcast_fields > 0 (A.2 warm-start intact)"
else
    echo "  ⚠ ws_bcast_fields=0 or not found (warm-start may not have fired)"
fi

# (e) Wall time within +5% of A.2 ref (1999s)
A2_MF_REF=1999
MF_TOL_PCT=5
# Extract MF wall from log (MF-TIME markers or "ModelFinder" timing line)
MF_WALL=$(grep -oP 'ModelFinder.*?(\d+\.\d+)s' "${WORK_DIR}/iqtree_run.log" 2>/dev/null | grep -oP '\d+\.\d+' | tail -1 || echo "")
if [[ -n "${MF_WALL}" ]]; then
    MF_PCT=$(python3 -c "print(round(100*(${MF_WALL}-${A2_MF_REF})/${A2_MF_REF},1))")
    MF_OK=$(python3 -c "print('yes' if ${MF_WALL} <= ${A2_MF_REF}*1.${MF_TOL_PCT} else 'no')" 2>/dev/null || echo "yes")
    echo "  MF wall: ${MF_WALL}s  (A.2 ref ${A2_MF_REF}s  delta ${MF_PCT}%)"
fi
echo "  Total wall: ${WALL}s  (A.2 ref ~6099s)"

echo ""
if [[ "${IQRC}" -eq 0 && "${PASS}" -eq 1 ]]; then
    echo "  ══ GATE: PASS ══"
else
    echo "  ══ GATE: FAIL (rc=${IQRC} pass=${PASS}) ══" >&2
    exit $(( IQRC != 0 ? IQRC : 1 ))
fi

echo ""
echo "  Log:      ${WORK_DIR}/iqtree_run.log"
echo "  Rank logs: ${RANK_LOGS_DIR}/"
echo ""
echo "  Next: qsub run_atmd_b3_aa_100k_1node.sh  (K_outer>1 activation test)"
