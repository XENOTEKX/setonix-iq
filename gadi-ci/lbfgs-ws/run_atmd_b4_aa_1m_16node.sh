#!/bin/bash
# run_atmd_b4_aa_1m_16node.sh — ATMD b4 (B.5 per_tree fix) full MF+SPR, AA 1M, 16 nodes.
#
# PURPOSE: Empirical test of the bandwidth-saturation hypothesis at AA 1M scale.
# b4 fixes the per_tree_bytes formula in phylotesting.cpp B.5: real per-tree allocation
# at AA 1M is ~64 GB, not the ~458 GB the prior conservative-bad formula projected.
# This lifts K_mem at 1M from 1 to ~6, activating Mode F at the larger dataset.
#
# Expected outcomes (from §15.9.14 analysis):
#   • K_outer=6 confirmed in [ATMD Mode F] log line  (the headline mechanical change)
#   • MF wall: HYPOTHESIS — bandwidth-saturation regression similar to 100K K=8 case.
#     At K=6, 6 × ~64 GB = 384 GB working set vs ~500 GB/s aggregate DRAM bandwidth.
#     Predicted MF wall: 1.4-1.8× FCA np=16 ref (1,122s) → 1,570-2,020s.
#     If true: Mode F bandwidth-saturation is dataset-size-invariant, confirming the
#     diagnosis. If false (ATMD beats FCA): rerun A/B with K_outer overrides to map
#     the K_outer→wall curve.
#
# A/B references (AA 1M):
#   FCA np=16          168635616   MF=1,122s  SPR=1,288s  total=2,410s  lnL=−78,605,196.497
#   WS-A.2 np=16       169096801   MF=1,139s  SPR=1,199s  total=2,420s  lnL=−78,605,196.497
#   ATMD b3c np=16     169112256   MF≥2,000s (proj)  K_outer=1  (Mode F mechanically inert)
#   ATMD b4  np=16     THIS RUN    Expected K_outer=6
#
# Gate pass criteria:
#   - exit code = 0
#   - lnL within ±1.0 of −78,605,196.497 (FCA np=16 ref)
#   - Best model = LG+G4
#   - K_outer confirmed in iqtree_inner.log via [ATMD Mode F] cout line
#   - MF wall: informational (this is a hypothesis test, not a regression gate)
#
# Binary: iqtree3-mpi-atmd-b4  (build-atmd-b4/, clean diagnostics, B.5 formula)
# Branch: fca-lbfgs-ws
# See:    research/lbfgs-and-warmstart-implementation.md §15.9.15

#PBS -N atmd-b4-1m-16n
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
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b4/iqtree3-mpi-atmd-b4}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${SRC_PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=16
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

# Optional: cap K_outer via env (1 = force serial, 2-8 = bound below natural K_mem)
# Useful for mapping the K_outer→wall curve. Leave empty for natural K_outer.
ATMD_K_OUTER_OVERRIDE="${ATMD_K_OUTER_OVERRIDE:-}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL_SUFFIX=""
[[ -n "${ATMD_K_OUTER_OVERRIDE}" ]] && LABEL_SUFFIX="_k${ATMD_K_OUTER_OVERRIDE}"
LABEL="AA_1m_atmd_b4_np16_full_seed${SEED}${LABEL_SUFFIX}"
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
[[ -x "${IQTREE}" ]]    || { echo "ERROR: ATMD binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing." >&2; exit 8; }

if strings "${IQTREE}" 2>/dev/null | grep -q 'ATMD Mode F'; then
    echo "[preflight] [ATMD Mode F] log string: present"
else
    echo "[preflight] WARNING: '[ATMD Mode F]' not found in binary — ATMD may not be compiled in." >&2
fi
if strings "${IQTREE}" 2>/dev/null | grep -q 'ATMD-DIAG'; then
    echo "[preflight] WARNING: [ATMD-DIAG] still in binary — b4 should have removed these." >&2
fi

ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
echo "[preflight] md5: ${ACTUAL_MD5}"

# ── OMP / runtime ──────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

# OMP_MAX_ACTIVE_LEVELS=2 required for B.3 nested OMP teams.
# OMP_PROC_BIND=close: threads pack within each rank's socket (NUMA-local).
# At K_outer>1 (expected 6 for 1M b4), nested teams of M_inner=17 threads
# evaluate concurrent models per rank. Inner teams should be NUMA-local.
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

IQ_ARGS=(-s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}"
         --prefix "${WORK_DIR}/iqtree_inner")
[[ -n "${ATMD_K_OUTER_OVERRIDE}" ]] && IQ_ARGS+=(--atmd-K-outer "${ATMD_K_OUTER_OVERRIDE}")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ATMD b4 (B.5 fix) Full MF+SPR — AA 1M, np=16, 16-node      ║"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP:  ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5:          ${ACTUAL_MD5}"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  K_outer override: ${ATMD_K_OUTER_OVERRIDE:-<auto>}"
echo "║  Expected K_outer: 6  (B.5 formula; per_tree_MB ≈ 64,000)"
echo "║  Gate lnL:     within ±1.0 of −78,605,196.497 (FCA np=16)"
echo "║  FCA np=16 ref: MF=1,122s  SPR=1,288s  total=2,410s"
echo "║  Hypothesis:    K=6 → bandwidth saturation → MF ≥ 1.4× FCA"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[16node] hostfile:"; cat "${HOSTFILE}" | sed 's/^/    /'
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
    "${IQTREE}" "${IQ_ARGS[@]}" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

# Show logs
echo "--- inner log (tail 60) ---"
tail -60 "${WORK_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "--- stdout log (tail 20) ---"
tail -20 "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true
echo "--- [ATMD Mode F] line ---"
grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -3
echo ""
echo "[16node] done: rc=${IQRC} wall=${WALL}s"

# ── Gate checks ────────────────────────────────────────────────────────
echo ""
echo "══ Gate checks ═════════════════════════════════════════════════"
PASS=1

# (a) [ATMD Mode F] line (b4 only writes via cout/TeeBuf — no sidecar)
ATMD_LINE=""
if grep -q '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null; then
    ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" | head -1)
    echo "  [ATMD Mode F] found in: iqtree_inner.log"
elif grep -q '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null; then
    ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_stdout.log" | head -1)
    echo "  [ATMD Mode F] found in: iqtree_stdout.log"
fi

if [[ -n "${ATMD_LINE}" ]]; then
    echo "  ATMD line: ${ATMD_LINE}"
    K_ACTUAL=$(echo "${ATMD_LINE}" | grep -oP 'K_outer=\K[0-9]+' || echo "?")
    PER_TREE_MB=$(echo "${ATMD_LINE}" | grep -oP 'per_tree_MB=\K[0-9]+' || echo "?")
    AVAIL_MB=$(echo "${ATMD_LINE}" | grep -oP 'avail_MB=\K[0-9]+' || echo "?")
    K_MEM=$(echo "${ATMD_LINE}" | grep -oP 'K_mem=\K[0-9]+' || echo "?")
    if [[ "${K_ACTUAL}" -gt "1" ]] 2>/dev/null; then
        echo "  ✓ K_outer=${K_ACTUAL} (Mode F ACTIVE)  K_mem=${K_MEM}  per_tree_MB=${PER_TREE_MB}  avail_MB=${AVAIL_MB}"
    elif [[ "${K_ACTUAL}" == "1" ]]; then
        echo "  ⚠ K_outer=1 — Mode F inert. per_tree_MB=${PER_TREE_MB} (expected ~64,000 for B.5 1M)"
        echo "    Check: B.5 formula applied correctly? Expected per_tree_MB much less than avail_MB/2."
    else
        echo "  ? K_outer=${K_ACTUAL} — parse failure"
    fi
else
    echo "  ✗ FAIL: [ATMD Mode F] not found in any log — ATMD code not reached" >&2
    PASS=0
fi

# (b) Exit code
if [[ "${IQRC}" -ne 0 ]]; then
    echo "  ✗ FAIL: exit code ${IQRC}" >&2; PASS=0
else
    echo "  ✓ exit code 0"
fi

# (c) lnL gate
LNL_REF=-78605196.497
LNL_TOL=1.0
LNL_ACTUAL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
if [[ -z "${LNL_ACTUAL}" ]]; then
    LNL_ACTUAL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | tail -1 || echo "")
fi
if [[ -n "${LNL_ACTUAL}" ]]; then
    LNL_DIFF=$(python3 -c "print(abs(${LNL_ACTUAL} - (${LNL_REF})))")
    LNL_OK=$(python3 -c "print('yes' if ${LNL_DIFF} <= ${LNL_TOL} else 'no')")
    if [[ "${LNL_OK}" == "yes" ]]; then
        echo "  ✓ lnL ${LNL_ACTUAL}  |diff|=${LNL_DIFF} ≤ ${LNL_TOL}"
    else
        echo "  ✗ FAIL: lnL ${LNL_ACTUAL}  |diff|=${LNL_DIFF} > ${LNL_TOL}" >&2; PASS=0
    fi
else
    echo "  ✗ FAIL: lnL not found in log" >&2; PASS=0
fi

# (d) Best model
BEST_MODEL=$(grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
if [[ -z "${BEST_MODEL}" ]]; then
    BEST_MODEL=$(grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -1 || echo "")
fi
if [[ "${BEST_MODEL}" == "LG+G4" ]]; then
    echo "  ✓ Best model: ${BEST_MODEL}"
else
    echo "  ✗ FAIL: best model '${BEST_MODEL}' (expected LG+G4)" >&2; PASS=0
fi

# (e) MF wall vs FCA np=16 baseline
FCA_MF_REF=1122
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
if [[ -z "${MF_WALL}" ]]; then
    MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -1 || echo "")
fi
SPR_WALL=$(grep -oP 'Wall-clock time used for tree search: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
if [[ -n "${MF_WALL}" ]]; then
    MF_DELTA=$(python3 -c "print(round(${MF_WALL} - ${FCA_MF_REF}, 1))")
    MF_PCT=$(python3 -c "print(round(100*(${MF_WALL} - ${FCA_MF_REF})/${FCA_MF_REF}, 1))")
    if python3 -c "import sys; sys.exit(0 if ${MF_WALL} < ${FCA_MF_REF} else 1)"; then
        echo "  ★ MF wall ${MF_WALL}s BEATS FCA np=16 ref ${FCA_MF_REF}s  (Δ=${MF_DELTA}s ${MF_PCT}%)"
        echo "    → B.5 fix DELIVERS speedup. Bandwidth-saturation hypothesis falsified at 1M."
    else
        echo "  ⚠ MF wall ${MF_WALL}s vs FCA ref ${FCA_MF_REF}s  Δ=${MF_DELTA}s (${MF_PCT}%)"
        echo "    → Consistent with bandwidth-saturation hypothesis (K=6 contention)."
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
    "binary":       "iqtree3-mpi-atmd-b4",
    "md5":          "${ACTUAL_MD5}",
    "dataset":      "AA_1M",
    "nranks":       ${NRANKS},
    "omp_per_rank": ${OMP_PER_RANK},
    "seed":         ${SEED},
    "k_outer_override": "${ATMD_K_OUTER_OVERRIDE:-null}",
    "wall_s":       ${WALL},
    "mf_wall_s":    "${MF_WALL:-null}",
    "spr_wall_s":   "${SPR_WALL:-null}",
    "lnl":          "${LNL_ACTUAL:-null}",
    "best_model":   "${BEST_MODEL:-null}",
    "atmd_line":    "${ATMD_LINE:-null}",
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
echo "  Rank logs: ${RANK_LOGS_DIR}/"
echo ""
echo "  Next steps:"
echo "    - Update impl doc §15.9.15 with K_outer and MF wall result"
echo "    - If MF beats FCA: bandwidth-saturation hypothesis FALSIFIED → ATMD viable at 1M"
echo "    - If MF regression: hypothesis confirmed → Mode P is the only path forward"
echo "    - To map K_outer→wall curve: ATMD_K_OUTER_OVERRIDE=2 qsub run_atmd_b4_aa_1m_16node.sh"
