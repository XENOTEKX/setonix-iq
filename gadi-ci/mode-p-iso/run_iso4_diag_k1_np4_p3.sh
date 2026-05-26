#!/bin/bash
# run_iso4_diag_k1_np4_p3.sh — B.4-10 diagnostic: iso4 with --atmd-k-outer 1
#
# PURPOSE: Determine whether the ISO-4 SIGFPE (B.4-10) is triggered by the
# ATMD batch dispatch grouping (+F models batched alongside others in an
# 8-model K_outer window) rather than by the +F kernel computation itself.
#
# K_outer=8 (default) batches 8 models per collective dispatch round.
# K_outer=1 forces each model through its own dispatch round (single-model
# batches) — eliminating any possibility of model-parameter state corruption
# between batch slots or incorrect buffer handoff to rank 3 for the +F model.
#
# If this run PASSES: root cause is in the batch dispatch / parameter handoff
#   → fix the K_outer>1 batch model-param MPI broadcast for +F models (B.4-10).
# If this run also crashes with SIGFPE: K_outer is not the trigger; root cause
#   is in the per-model +F kernel arithmetic itself (ptn_start=72016 specific).
#
# MODEP-only (no BASE sub-run) — BASE is known-good from ISO-4 job 169136585.
#
# References:
#   ISO-4 crash  job 169136585  SIGFPE rank 3 during LG+F  (K_outer=8)
#   FCA np=1     job 169095077  lnL=-7,541,976.861
#   ISO-2 tree   job 169135061  binary 50b4b172
#   Binary       9660575a       (B.4-9 patched, build 169136419)

#PBS -N iso4-diag-k1
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l walltime=00:20:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
IQTREE="${IQTREE:-${SANDBOX}/build-mode-p-iso-p3/iqtree3-mpi-mode-p-iso-p3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
ISO2_TREE="${ISO2_TREE:-${SANDBOX}/runs/iso2_p3_aa100k_np2_seed1_169135061/iqtree_inner.treefile}"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

EXPECTED_MD5="76cdfb199f7765b58d4e7f59cf22fdf0"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso4_diag_k1_p3_aa100k_np4_seed${SEED}"
WORK_DIR="${SANDBOX}/runs/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]] || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
[[ -f "${ISO2_TREE}" ]] || { echo "ERROR: ISO-2 starting tree not found: ${ISO2_TREE}" >&2; exit 4; }
MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
if [[ "${MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "WARNING: binary md5 ${MD5} does not match expected ${EXPECTED_MD5}" >&2
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

[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing" >&2; exit 8; }
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -ge 4 ]] || { echo "ERROR: expected >=4 nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
rank 1=${HOSTS[1]} slot=0-103
rank 2=${HOSTS[2]} slot=0-103
rank 3=${HOSTS[3]} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ISO-4 DIAG K1: B.4-10 trigger isolation, np=4, K_outer=1   ║"
echo "║  binary: $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  tree:   $(basename "${ISO2_TREE}") (ISO-2 PASS treefile)"
echo "║  Purpose: SIGFPE only if K_outer>1 batch dispatch is the bug ║"
echo "║  Ref lnL: -7,541,976.861 (FCA np=1 job 169095077)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────────────────
# Sub-run MODEP — np=4, --mode-p-all, --atmd-k-outer 1
# ─────────────────────────────────────────────────────────────
MODEP_DIR="${WORK_DIR}/modep_np4_k1"; mkdir -p "${MODEP_DIR}"
MODEP_RANK_LOGS="${MODEP_DIR}/rank_logs"; mkdir -p "${MODEP_RANK_LOGS}"

echo ""
echo "── Sub-run MODEP (np=4, --mode-p-all, --atmd-k-outer 1) ────────"
START_MODEP=$(date +%s)
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    --mca coll ^ucc \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${MODEP_RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                -te "${ISO2_TREE}" \
                --mode-p-all \
                --atmd-k-outer 1 \
                --prefix "${MODEP_DIR}/iqtree_inner" \
    > "${MODEP_DIR}/iqtree_stdout.log" 2>&1
MODEP_RC=$?
END_MODEP=$(date +%s)
WALL_MODEP=$(( END_MODEP - START_MODEP ))
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s"

# ─────────────────────────────────────────────────────────────
# Parse results
# ─────────────────────────────────────────────────────────────
MODEP_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{printf "%.6f\n", v}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
MODEP_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
MODEP_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "── [Mode P] partition lines (first 8) ───────────────────────────"
{ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | head -8

MODE_P_TOTAL=$({ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | wc -l)
MODE_P_RANKS=$({ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null || true; } \
    | grep -oP 'rank \K[0-9]+' | sort -un | wc -l)

# ─────────────────────────────────────────────────────────────
# Numeric pass/fail
# ─────────────────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - \
    "${MODEP_OPT}" "${MODEP_BEST}" "${MODE_P_TOTAL}" "${MODE_P_RANKS}" <<'PYEOF'
import sys

modep_opt   = sys.argv[1]
modep_best  = sys.argv[2]
mp_total    = int(sys.argv[3])
mp_ranks    = int(sys.argv[4])

REF_LNL        = -7541976.861
LNL_TOL        = 0.05
EXPECTED_NP    = 4
EXPECTED_MODEL = "LG+G4"

lines = []
ok = True

if modep_opt:
    delta = abs(float(modep_opt) - REF_LNL)
    sym = "✓" if delta <= LNL_TOL else "✗"
    lines.append(f"  {sym} MODEP lnL={modep_opt}  |Δ vs ref|={delta:.4f}  tol={LNL_TOL}")
    if delta > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ MODEP optimal lnL not parsed"); ok = False

if modep_best == EXPECTED_MODEL:
    lines.append(f"  ✓ MODEP best model: {modep_best}")
else:
    lines.append(f"  ✗ MODEP best model: {modep_best}  (expected {EXPECTED_MODEL})")
    ok = False

if mp_total > 0 and mp_ranks >= EXPECTED_NP:
    lines.append(f"  ✓ [Mode P] lines: {mp_total} total across {mp_ranks} ranks (all {EXPECTED_NP} present)")
elif mp_total > 0:
    lines.append(f"  ✗ [Mode P] lines: {mp_total} total but only {mp_ranks}/{EXPECTED_NP} ranks emitted")
    ok = False
else:
    lines.append("  ✗ No [Mode P] lines emitted"); ok = False

if ok:
    lines.append("  → B.4-10 is caused by K_outer>1 batch dispatch: fix parameter handoff for +F models")
else:
    lines.append("  → B.4-10 is NOT caused by K_outer batch dispatch: bug is in per-model +F kernel arithmetic")

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ ISO-4 DIAG K1 result (B.4-10 isolation) ═════════════════════"
echo "  binary md5: ${MD5}  (expected: ${EXPECTED_MD5})"
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s model=${MODEP_BEST} lnL=${MODEP_OPT} MF=${MODEP_MF_WALL}s"
echo "  [Mode P] total=${MODE_P_TOTAL} distinct_ranks=${MODE_P_RANKS}"
echo "  ref lnL: -7541976.861 (FCA np=1 job 169095077)  tolerance: 0.05"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

[[ "${MODEP_RC}" -eq 0 ]] || { echo "  ✗ FAIL: MODEP rc=${MODEP_RC}"; }
echo "${COMPARE_RESULT}" | grep -q '^PASS$' && PASS=1 || PASS=0
[[ "${MODEP_RC}" -eq 0 ]] || PASS=0

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ DIAG PASS: K_outer=1 survives LG+F — batch dispatch is B.4-10 trigger ══"
    exit 0
else
    echo "  ══ DIAG FAIL: K_outer=1 also crashes — B.4-10 is in per-model +F kernel arithmetic ══" >&2
    exit 10
fi
