#!/bin/bash
# run_iso4_aa100k_np4_aid.sh — ISO-4 correctness gate for ATMD-AID (Adaptive
# Island Dispatch), np=4, AA 100K, --atmd-aid.
#
# PURPOSE: Verify that ATMD-AID's wave dispatch produces correct results
# (lnL parity with FCA np=1 reference within 0.05) at np=4 before moving
# to the heavier perf gate (AA 1M np=16).
#
# Two sub-runs:
#   BASE  — np=1, no Mode P, -m TEST (reference)
#   AID   — np=4, --atmd-aid (Architecture C + wave dispatch)
#
# References:
#   FCA np=1   job 169095077  lnL=-7,541,976.861
#   ISO-2 tree job 169135061  binary 50b4b172
#   Design doc research/Modelfinder/novel-dispatch-architectures.md

#PBS -N iso4-aid-np4
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l walltime=00:30:00
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

# Expected binary: ATMD-AID build (Architecture C + wave dispatch). md5 set at
# build completion — update here after build job exits.
EXPECTED_MD5="${EXPECTED_MD5:-3e79db194ced77971a55c6a0ff476863}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso4_aid_aa100k_np4_seed${SEED}"
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
if [[ -n "${EXPECTED_MD5}" && "${MD5}" != "${EXPECTED_MD5}" ]]; then
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

RANKFILE_BASE="${WORK_DIR}/rankfile_base.txt"
cat > "${RANKFILE_BASE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ISO-4 AID gate: ATMD-AID, np=4, AA 100K, -m TEST           ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  Purpose: 4-rank Architecture C + wave dispatch correctness ║"
echo "║  Ref lnL: -7,541,976.861 (FCA np=1 job 169095077)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────────────────
# Sub-run 1: BASE — np=1, no Mode P, -m TEST, fixed tree
# ─────────────────────────────────────────────────────────────
BASE_DIR="${WORK_DIR}/base_np1"; mkdir -p "${BASE_DIR}"
BASE_RANK_LOGS="${BASE_DIR}/rank_logs"; mkdir -p "${BASE_RANK_LOGS}"

echo ""
echo "── Sub-run BASE (np=1, no Mode P) ──────────────────────────────"
START_BASE=$(date +%s)
mpirun -np 1 \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE_BASE}" \
    --output-filename "${BASE_RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                -te "${ISO2_TREE}" \
                --prefix "${BASE_DIR}/iqtree_inner" \
    > "${BASE_DIR}/iqtree_stdout.log" 2>&1
BASE_RC=$?
END_BASE=$(date +%s)
WALL_BASE=$(( END_BASE - START_BASE ))
echo "  BASE exit=${BASE_RC} wall=${WALL_BASE}s"

# ─────────────────────────────────────────────────────────────
# Sub-run 2: AID — np=4, --atmd-aid, -m TEST, fixed tree
# ─────────────────────────────────────────────────────────────
AID_DIR="${WORK_DIR}/aid_np4"; mkdir -p "${AID_DIR}"
AID_RANK_LOGS="${AID_DIR}/rank_logs"; mkdir -p "${AID_RANK_LOGS}"

echo ""
echo "── Sub-run AID (np=4, --atmd-aid) ──────────────────────────────"
START_AID=$(date +%s)
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    --mca coll ^ucc \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${AID_RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                -te "${ISO2_TREE}" \
                --atmd-aid \
                --atmd-k-outer 1 \
                --prefix "${AID_DIR}/iqtree_inner" \
    > "${AID_DIR}/iqtree_stdout.log" 2>&1
AID_RC=$?
END_AID=$(date +%s)
WALL_AID=$(( END_AID - START_AID ))
echo "  AID   exit=${AID_RC} wall=${WALL_AID}s"

# ─────────────────────────────────────────────────────────────
# Parse results
# ─────────────────────────────────────────────────────────────
BASE_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{if (v != "") print v}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
AID_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{if (v != "") print v}' \
    "${AID_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
BASE_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
AID_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${AID_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
BASE_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
AID_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${AID_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

# AID-DIAG markers
echo ""
echo "── AID-DIAG output (first 8 lines) ──────────────────────────────"
grep -h 'AID-DIAG\|AID-WAVE' \
    "${AID_DIR}/iqtree_inner.log" \
    "${AID_DIR}/iqtree_stdout.log" \
    "${AID_RANK_LOGS}"/*/stdout \
    "${AID_RANK_LOGS}"/*/stderr 2>/dev/null | head -8 || true

WAVE_COUNT=$({ grep -h 'AID-WAVE' \
    "${AID_DIR}/iqtree_inner.log" \
    "${AID_DIR}/iqtree_stdout.log" \
    "${AID_RANK_LOGS}"/*/stdout \
    "${AID_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | grep -c 'start' )

# ─────────────────────────────────────────────────────────────
# Numeric pass/fail
# ─────────────────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - \
    "${BASE_OPT}" "${AID_OPT}" \
    "${BASE_BEST}" "${AID_BEST}" \
    "${WAVE_COUNT}" \
    "${WALL_BASE}" "${WALL_AID}" <<'PYEOF'
import sys

base_opt    = sys.argv[1]
aid_opt     = sys.argv[2]
base_best   = sys.argv[3]
aid_best    = sys.argv[4]
wave_count  = int(sys.argv[5])
wall_base   = int(sys.argv[6])
wall_aid    = int(sys.argv[7])

REF_LNL        = -7541976.861
LNL_TOL        = 0.05
EXPECTED_MODEL = "LG+G4"

lines = []
ok = True

# 1. AID lnL vs reference
if aid_opt:
    delta = abs(float(aid_opt) - REF_LNL)
    sym = "✓" if delta <= LNL_TOL else "✗"
    lines.append(f"  {sym} AID  lnL={aid_opt}  |Δ vs ref|={delta:.4f}  tol={LNL_TOL}")
    if delta > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ AID optimal lnL not parsed"); ok = False

# 2. BASE lnL vs reference
if base_opt:
    delta_b = abs(float(base_opt) - REF_LNL)
    sym = "✓" if delta_b <= LNL_TOL else "✗"
    lines.append(f"  {sym} BASE lnL={base_opt}  |Δ vs ref|={delta_b:.4f}")
    if delta_b > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ BASE optimal lnL not parsed"); ok = False

# 3. Best model agreement
if aid_best == EXPECTED_MODEL:
    lines.append(f"  ✓ AID  best model: {aid_best}")
else:
    lines.append(f"  ✗ AID  best model: {aid_best}  (expected {EXPECTED_MODEL})")
    ok = False
if base_best:
    sym = "✓" if base_best == EXPECTED_MODEL else "⚠"
    lines.append(f"  {sym} BASE best model: {base_best}")

# 4. Wave dispatch fired (AID-WAVE start markers)
if wave_count > 0:
    lines.append(f"  ✓ AID-WAVE markers: {wave_count} (Phase 1 wave dispatch fired)")
else:
    lines.append(f"  ⚠ AID-WAVE markers: 0 (Phase 1 may not have fired — heavy queue empty?)")

# 5. Speedup (informational)
if wall_base > 0 and wall_aid > 0:
    speedup = wall_base / wall_aid
    lines.append(f"  {'✓' if speedup >= 1.0 else '⚠'} Wall: {wall_base}s → {wall_aid}s = {speedup:.2f}×")

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ ISO-4 AID result (Architecture C + wave dispatch, np=4) ═════"
echo "  binary md5: ${MD5}"
echo "  BASE  exit=${BASE_RC} wall=${WALL_BASE}s model=${BASE_BEST} lnL=${BASE_OPT} MF=${BASE_MF_WALL}s"
echo "  AID   exit=${AID_RC}  wall=${WALL_AID}s  model=${AID_BEST}  lnL=${AID_OPT}  MF=${AID_MF_WALL}s"
echo "  AID-WAVE invocations: ${WAVE_COUNT}"
echo "  ref lnL: -7541976.861 (FCA np=1 job 169095077)  tolerance: 0.05"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${BASE_RC}" -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}"; PASS=0; }
[[ "${AID_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: AID rc=${AID_RC}";  PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-4 AID PASS — ATMD-AID correctness confirmed at np=4 ══"
    exit 0
else
    echo "  ══ ISO-4 AID FAIL — investigate above ══" >&2
    exit 10
fi
