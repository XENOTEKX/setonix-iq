#!/bin/bash
# run_atmd_b4_aa_100k_1node.sh — ATMD b4 sanity at AA 100K, 1 node, np=1.
#
# PURPOSE: Regression-check that the B.5 formula does not break the 100K K_outer=8
# behaviour already validated by b3c (job 169111545). At 100K, real per-tree allocation
# is ~6.4 GB (B.5 formula), well below avail_MB/8 ≈ 62 GB → K_outer should hit the
# K_cap=8 ceiling (same as b3c's K_outer=8 result).
#
# Expected:
#   • K_outer=8 (cap; same as b3c at 100K)
#   • lnL = -7,541,976.853 (Δ < 1.0 vs FCA np=1 ref)
#   • Best model = LG+G4
#   • MF wall in 400-500s range (bandwidth-saturated K=8 path; same as b3c 423s)
#
# A/B references (AA 100K, 1 node):
#   FCA np=1            169095077   MF=258.8s  total=1,001s   lnL=-7,541,976.853
#   ATMD b3c K=8        169111545   MF=423.2s  total=1,720s   lnL=-7,541,976.853
#   ATMD b4  K=8        THIS RUN    Expected match to b3c (B.5 doesn't change 100K)
#
# Binary: iqtree3-mpi-atmd-b4  (build-atmd-b4/, B.5 formula, clean diagnostics)
# See:    research/lbfgs-and-warmstart-implementation.md §15.9.15

#PBS -N atmd-b4-100k-1n
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

SRC_PROJECT="rc29"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${SRC_PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-atmd-b4/iqtree3-mpi-atmd-b4}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${SRC_PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
ATMD_K_OUTER_OVERRIDE="${ATMD_K_OUTER_OVERRIDE:-}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL_SUFFIX=""
[[ -n "${ATMD_K_OUTER_OVERRIDE}" ]] && LABEL_SUFFIX="_k${ATMD_K_OUTER_OVERRIDE}"
LABEL="AA_100k_atmd_b4_np1_full_seed${SEED}${LABEL_SUFFIX}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]]    || { echo "ERROR: ATMD binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')

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

IQ_ARGS=(-s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}"
         --prefix "${WORK_DIR}/iqtree_inner")
[[ -n "${ATMD_K_OUTER_OVERRIDE}" ]] && IQ_ARGS+=(--atmd-K-outer "${ATMD_K_OUTER_OVERRIDE}")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ATMD b4 (B.5 fix) sanity — AA 100K, np=1, 1-node            ║"
echo "║  binary:       $(basename "${IQTREE}")  md5:${ACTUAL_MD5}"
echo "║  K_outer override: ${ATMD_K_OUTER_OVERRIDE:-<auto>}"
echo "║  Expected K_outer=8 (cap; per_tree_MB ≈ 6,400 < avail/8)"
echo "║  A/B: FCA np=1 MF=259s; b3c K=8 MF=423s; expect b4 ≈ b3c"
echo "╚══════════════════════════════════════════════════════════════╝"

START_EPOCH=$(date +%s)
mpirun -np "${NRANKS}" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" "${IQ_ARGS[@]}" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

echo "--- inner log tail ---"
tail -30 "${WORK_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "--- [ATMD Mode F] ---"
grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -3
echo "[1node] done: rc=${IQRC} wall=${WALL}s"

# Quick parse + gate
ATMD_LINE=$(grep '\[ATMD Mode F\]' "${WORK_DIR}/iqtree_inner.log" "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -1 || true)
K_ACTUAL=$(echo "${ATMD_LINE}" | grep -oP 'K_outer=\K[0-9]+' || echo "?")
LNL_ACTUAL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
BEST_MODEL=$(grep -oP 'Best-fit model: \K\S+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${WORK_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "══ Result ═════════════════════════════════════════════════════"
echo "  K_outer:     ${K_ACTUAL}    (expected 8 at AA 100K)"
echo "  lnL:         ${LNL_ACTUAL}  (FCA ref -7,541,976.853)"
echo "  best model:  ${BEST_MODEL}  (expected LG+G4)"
echo "  MF wall:     ${MF_WALL}s     (FCA ref 258.8s, b3c K=8 ref 423.2s)"
echo "  Total wall:  ${WALL}s"
echo "  Exit code:   ${IQRC}"

python3 - <<PYEOF
import json, os, time
rec = {
    "run_id":       "${RUN_ID}",
    "label":        "${LABEL}",
    "binary":       "iqtree3-mpi-atmd-b4",
    "md5":          "${ACTUAL_MD5}",
    "dataset":      "AA_100K",
    "nranks":       ${NRANKS},
    "omp_per_rank": ${OMP_PER_RANK},
    "k_outer_override": "${ATMD_K_OUTER_OVERRIDE:-null}",
    "wall_s":       ${WALL},
    "mf_wall_s":    "${MF_WALL:-null}",
    "k_outer":      "${K_ACTUAL}",
    "lnl":          "${LNL_ACTUAL:-null}",
    "best_model":   "${BEST_MODEL:-null}",
    "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(os.path.join("${RUNS_DIR}", "${RUN_ID}.json"), "w") as f:
    json.dump(rec, f, indent=2)
PYEOF
