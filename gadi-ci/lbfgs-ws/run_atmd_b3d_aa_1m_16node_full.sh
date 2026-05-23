#!/bin/bash
# run_atmd_b3d_aa_1m_16node_full.sh — ATMD b3d full MF+SPR: AA 1M, 16 nodes (clean build).
#
# WHEN TO SUBMIT: Only after build_atmd_b3d.sh succeeds AND job 169112256 (b3c 16-node)
# showed ATMD outperforming the FCA np=16 baseline (MF < 1,122s or total < 2,410s).
#
# b3d vs b3c:
#   - No [ATMD-DIAG] fprintf overhead (removed by build_atmd_b3d.sh)
#   - No sidecar fopen/fclose per rank
#   - [ATMD Mode F] production log line retained (in iqtree_inner.log)
#   - All B.3+B.4 K_outer×M_inner nested-OMP dispatch unchanged
#   - All B.4-1 /proc/meminfo memory budget logic unchanged
#
# A/B references (AA 1M):
#   FCA np=16         168635616  MF=1,122s  SPR=1,288s  total=2,410s  lnL=−78,605,196.497
#   ATMD b3c np=16    169112256  pending
#
# See: research/lbfgs-and-warmstart-implementation.md §15.9.14

#PBS -N atmd-b3d-1m-16n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=1664
#PBS -l mem=8160GB
#PBS -l place=excl
#PBS -l walltime=03:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
SRC_PROJECT="rc29"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${SRC_PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b3d/iqtree3-mpi-atmd-b3d}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${SRC_PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=16
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_1m_atmd_b3d_np16_full_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: b3d binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing." >&2; exit 8; }

ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
echo "[preflight] b3d binary: ${IQTREE}"
echo "[preflight] md5: ${ACTUAL_MD5}"

# Verify no ATMD-DIAG overhead in binary
if grep -q 'ATMD-DIAG' "${IQTREE}" 2>/dev/null; then
    echo "WARNING: [ATMD-DIAG] present in binary — this may be b3c, not b3d" >&2
fi
if grep -q 'ATMD Mode F' "${IQTREE}" 2>/dev/null; then
    echo "[preflight] [ATMD Mode F] production log: present"
fi

# ── OMP / runtime ──────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

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

# ── Multi-node host discovery ──────────────────────────────────────────
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -eq 16 ]] || { echo "ERROR: expected 16 nodes, got ${#HOSTS[@]}" >&2; exit 9; }

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
: > "${RANKFILE}"
for i in $(seq 0 15); do
    echo "rank ${i}=${HOSTS[$i]} slot=0-103" >> "${RANKFILE}"
done

RANK_LOGS_DIR="${WORK_DIR}/rank_logs"
mkdir -p "${RANK_LOGS_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ATMD b3d Full MF+SPR — AA 1M, np=16, 16-node (clean)       ║"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}") (b3d: no diagnostic overhead)"
echo "║  md5:          ${ACTUAL_MD5}"
echo "║  FCA ref:      MF=1,122s  total=2,410s  (job 168635616)"
echo "║  b3c ref:      pending (job 169112256)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Run ────────────────────────────────────────────────────────────────
echo "[16node] Full run (MF+SPR), ${NRANKS} ranks × ${OMP_PER_RANK} OMP across 16 nodes"
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
                --prefix "${WORK_DIR}/iqtree_inner" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "--- stdout log (tail 20) ---"
tail -20 "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true
echo "[16node] done: rc=${IQRC} wall=${WALL}s"

# ── Gate checks ────────────────────────────────────────────────────────
echo ""
echo "══ Gate checks ═════════════════════════════════════════════════"
PASS=1

# (a) ATMD Mode F production log line
if grep -q '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null; then
    ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" | head -1)
    echo "  [ATMD Mode F] in inner log: ${ATMD_LINE}"
    K_ACTUAL=$(echo "${ATMD_LINE}" | grep -oP 'K_outer=\K[0-9]+' || echo "?")
    echo "  K_outer=${K_ACTUAL}"
else
    echo "  ⚠ [ATMD Mode F] not in iqtree_inner.log — check iqtree_stdout.log"
fi

# (b) Exit code
if [[ "${IQRC}" -ne 0 ]]; then
    echo "  ✗ FAIL: exit code ${IQRC}" >&2; PASS=0
else
    echo "  ✓ exit code 0"
fi

# (c) lnL
LNL_REF=-78605196.497
LNL_ACTUAL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
if [[ -n "${LNL_ACTUAL}" ]]; then
    LNL_DIFF=$(python3 -c "print(abs(${LNL_ACTUAL} - (${LNL_REF})))")
    if python3 -c "import sys; sys.exit(0 if ${LNL_DIFF} <= 1.0 else 1)"; then
        echo "  ✓ lnL ${LNL_ACTUAL}  |diff|=${LNL_DIFF} ≤ 1.0"
    else
        echo "  ✗ FAIL: lnL diff ${LNL_DIFF} > 1.0" >&2; PASS=0
    fi
else
    echo "  ✗ FAIL: lnL not found" >&2; PASS=0
fi

# (d) Best model
BEST_MODEL=$(grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
[[ "${BEST_MODEL}" == "LG+G4" ]] && echo "  ✓ Best model: ${BEST_MODEL}" || { echo "  ✗ FAIL: model '${BEST_MODEL}'" >&2; PASS=0; }

# (e) MF wall vs FCA np=16 and b3c np=16
FCA_MF_REF=1122
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
SPR_WALL=$(grep -oP 'Wall-clock time used for tree search: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
if [[ -n "${MF_WALL}" ]]; then
    MF_DELTA=$(python3 -c "print(round(${MF_WALL} - ${FCA_MF_REF}, 1))")
    MF_PCT=$(python3 -c "print(round(100*(${MF_WALL} - ${FCA_MF_REF})/${FCA_MF_REF}, 1))")
    if python3 -c "import sys; sys.exit(0 if ${MF_WALL} < ${FCA_MF_REF} else 1)"; then
        echo "  ✓ MF wall ${MF_WALL}s BEATS FCA np=16 ref ${FCA_MF_REF}s  (Δ=${MF_DELTA}s ${MF_PCT}%)"
    else
        echo "  ⚠ MF wall ${MF_WALL}s  FCA ref ${FCA_MF_REF}s  Δ=${MF_DELTA}s (${MF_PCT}%)"
    fi
fi
[[ -n "${SPR_WALL}" ]] && echo "  SPR wall: ${SPR_WALL}s  (FCA ref 1,288s)"
echo "  Total wall: ${WALL}s  (FCA ref 2,410s)"

echo ""
if [[ "${IQRC}" -eq 0 && "${PASS}" -eq 1 ]]; then
    echo "  ══ GATE: PASS ══"
else
    echo "  ══ GATE: FAIL (rc=${IQRC} pass=${PASS}) ══" >&2
fi

# ── JSON record ────────────────────────────────────────────────────────
python3 - <<PYEOF
import json, os, time
rec = {
    "run_id":       "${RUN_ID}",
    "label":        "${LABEL}",
    "job_id":       "${PBS_ID_SHORT}",
    "binary":       "iqtree3-mpi-atmd-b3d",
    "md5":          "${ACTUAL_MD5}",
    "dataset":      "AA_1M",
    "nranks":       ${NRANKS},
    "omp_per_rank": ${OMP_PER_RANK},
    "seed":         ${SEED},
    "wall_s":       ${WALL},
    "mf_wall_s":    "${MF_WALL:-null}",
    "spr_wall_s":   "${SPR_WALL:-null}",
    "lnl":          "${LNL_ACTUAL:-null}",
    "best_model":   "${BEST_MODEL:-null}",
    "pass":         ${PASS},
    "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
out = os.path.join("${RUNS_DIR}", "${RUN_ID}.json")
with open(out, "w") as f:
    json.dump(rec, f, indent=2)
print(f"  JSON: {out}")
PYEOF

echo ""
echo "  Work dir:  ${WORK_DIR}"
echo "  Inner log: ${WORK_DIR}/iqtree_inner.log"
echo "  Stdout:    ${WORK_DIR}/iqtree_stdout.log"
