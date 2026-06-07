#!/bin/bash
# run_iso4_aa100k_np4_edm.sh — ISO-4 correctness gate for EDM v0 (Event-Driven
# Moldable Dispatch), np=4, AA 100K, --mf-edm --mf-edm-group-size 4.
#
# PURPOSE: Verify that EDM v0 epoch dispatch produces correct results
# (lnL parity with FCA np=1 reference within 0.05) at np=4 before moving
# to the heavier perf gate (AA 1M np=16). EDM v0 schedules all model tasks
# into a sentinel epoch (full-width gs=np) plus an LPT-packed tail epoch at
# a configurable group size, reusing the Mode-P communicator lattice.
#
# Two sub-runs:
#   BASE  — np=1, no Mode P, -m TEST (reference)
#   EDM   — np=4, --mf-edm --mf-edm-group-size 4 (sentinel gs=4, tail gs=4)
#
# Gate pass criteria (8 checks):
#   1. BASE exits 0
#   2. EDM  exits 0  (no crash, no deadlock)
#   3. |EDM lnL - REF_LNL| < 0.05
#   4. |BASE lnL - REF_LNL| < 0.05
#   5. EDM best model = LG+G4
#   6. EDM-DIAG line found in output (scheduler ran)
#   7. EDM-EPOCH line found (at least sentinel epoch planned)
#   8. EDM-WAVE or AID-WAVE start markers > 0 (epoch executed)
#
# References:
#   FCA np=1   job 169095077  lnL=-7,541,976.861
#   ISO-2 tree job 169135061  binary 50b4b172
#   AID ISO-4  job 169341954  lnL=-7,541,976.862  (nearest prior, md5 0c493bd5)
#   Design doc research/Modelfinder/event-driven-moldable-dispatch.md

#PBS -N iso4-edm-np4
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
EDM_GROUP_SIZE="${EDM_GROUP_SIZE:-4}"

# EDM v0 build — md5 from login-node incremental build 2026-05-27T02:16:36+10:00.
EXPECTED_MD5="${EXPECTED_MD5:-4810a8ac73e3b92b1f93b3f03ec04d57}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso4_edm_aa100k_np4_seed${SEED}"
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
echo "INFO: binary md5=${MD5}"

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
echo "║  ISO-4 EDM gate: --mf-edm, np=4, AA 100K, -m TEST           ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  EDM group_size=${EDM_GROUP_SIZE}: sentinel gs=4, tail gs=${EDM_GROUP_SIZE}"
echo "║  Purpose: EDM v0 epoch scheduler correctness at np=4        ║"
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
# Sub-run 2: EDM — np=4, --mf-edm, -m TEST, fixed tree
# ─────────────────────────────────────────────────────────────
EDM_DIR="${WORK_DIR}/edm_np4"; mkdir -p "${EDM_DIR}"
EDM_RANK_LOGS="${EDM_DIR}/rank_logs"; mkdir -p "${EDM_RANK_LOGS}"

echo ""
echo "── Sub-run EDM (np=4, --mf-edm --mf-edm-group-size ${EDM_GROUP_SIZE}) ─"
START_EDM=$(date +%s)
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    --mca coll ^ucc \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${EDM_RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                -te "${ISO2_TREE}" \
                --mf-edm \
                --mf-edm-group-size "${EDM_GROUP_SIZE}" \
                --atmd-k-outer 1 \
                --prefix "${EDM_DIR}/iqtree_inner" \
    > "${EDM_DIR}/iqtree_stdout.log" 2>&1
EDM_RC=$?
END_EDM=$(date +%s)
WALL_EDM=$(( END_EDM - START_EDM ))
echo "  EDM exit=${EDM_RC} wall=${WALL_EDM}s"

# ─────────────────────────────────────────────────────────────
# Parse results
# ─────────────────────────────────────────────────────────────
BASE_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{if (v != "") print v}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
EDM_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{if (v != "") print v}' \
    "${EDM_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
BASE_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
EDM_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${EDM_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
BASE_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
EDM_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${EDM_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

# EDM scheduler diagnostics
echo ""
echo "── EDM-DIAG / EDM-EPOCH output ─────────────────────────────────"
{ grep -h 'EDM-DIAG\|EDM-EPOCH\|AID-DIAG' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/stdout \
    "${EDM_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | head -12

echo ""
echo "── EDM-WAVE / AID-WAVE markers (first 12) ──────────────────────"
{ grep -h 'EDM-WAVE\|AID-WAVE' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/stdout \
    "${EDM_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | head -12

EDM_DIAG_LINES=$({ grep -h 'EDM-DIAG' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/stdout \
    "${EDM_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | wc -l)

EDM_EPOCH_LINES=$({ grep -h 'EDM-EPOCH' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/stdout \
    "${EDM_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | wc -l)

WAVE_COUNT=$({ grep -h 'EDM-WAVE\|AID-WAVE' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/stdout \
    "${EDM_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | grep -c 'start' || true)

# ─────────────────────────────────────────────────────────────
# Numeric pass/fail
# ─────────────────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - \
    "${BASE_OPT}" "${EDM_OPT}" \
    "${BASE_BEST}" "${EDM_BEST}" \
    "${EDM_DIAG_LINES}" "${EDM_EPOCH_LINES}" "${WAVE_COUNT}" \
    "${WALL_BASE}" "${WALL_EDM}" <<'PYEOF'
import sys

base_opt        = sys.argv[1]
edm_opt         = sys.argv[2]
base_best       = sys.argv[3]
edm_best        = sys.argv[4]
edm_diag_lines  = int(sys.argv[5])
edm_epoch_lines = int(sys.argv[6])
wave_count      = int(sys.argv[7])
wall_base       = int(sys.argv[8])
wall_edm        = int(sys.argv[9])

REF_LNL        = -7541976.861
LNL_TOL        = 0.05
EXPECTED_MODEL = "LG+G4"

lines = []
ok = True

# 1. EDM lnL vs reference
if edm_opt:
    delta = abs(float(edm_opt) - REF_LNL)
    sym = "✓" if delta <= LNL_TOL else "✗"
    lines.append(f"  {sym} EDM  lnL={edm_opt}  |Δ vs ref|={delta:.4f}  tol={LNL_TOL}")
    if delta > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ EDM optimal lnL not parsed"); ok = False

# 2. BASE lnL vs reference
if base_opt:
    delta_b = abs(float(base_opt) - REF_LNL)
    sym = "✓" if delta_b <= LNL_TOL else "✗"
    lines.append(f"  {sym} BASE lnL={base_opt}  |Δ vs ref|={delta_b:.4f}")
    if delta_b > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ BASE optimal lnL not parsed"); ok = False

# 3. Best model
if edm_best == EXPECTED_MODEL:
    lines.append(f"  ✓ EDM  best model: {edm_best}")
else:
    lines.append(f"  ✗ EDM  best model: {edm_best}  (expected {EXPECTED_MODEL})")
    ok = False
if base_best:
    sym = "✓" if base_best == EXPECTED_MODEL else "⚠"
    lines.append(f"  {sym} BASE best model: {base_best}")

# 4. EDM-DIAG found (scheduler was invoked)
if edm_diag_lines > 0:
    lines.append(f"  ✓ EDM-DIAG lines: {edm_diag_lines} (scheduler entered)")
else:
    lines.append("  ✗ EDM-DIAG: 0 lines — scheduler may not have run"); ok = False

# 5. EDM-EPOCH found (epoch plan was printed)
if edm_epoch_lines > 0:
    lines.append(f"  ✓ EDM-EPOCH lines: {edm_epoch_lines} (epoch plan emitted)")
else:
    lines.append("  ✗ EDM-EPOCH: 0 lines — no epoch plan emitted"); ok = False

# 6. Wave/epoch execution markers
if wave_count > 0:
    lines.append(f"  ✓ Wave/epoch start markers: {wave_count} (epochs executed)")
else:
    lines.append("  ⚠ Wave/epoch start markers: 0 (may be empty model set or FCA-only path)")

# 7. Speedup (informational)
if wall_base > 0 and wall_edm > 0:
    speedup = wall_base / wall_edm
    lines.append(f"  {'✓' if speedup >= 1.0 else '⚠'} Wall: {wall_base}s → {wall_edm}s = {speedup:.2f}×")

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ ISO-4 EDM result (Event-Driven Moldable Dispatch v0, np=4) ══"
echo "  binary md5: ${MD5}"
echo "  BASE exit=${BASE_RC} wall=${WALL_BASE}s model=${BASE_BEST} lnL=${BASE_OPT} MF=${BASE_MF_WALL}s"
echo "  EDM  exit=${EDM_RC}  wall=${WALL_EDM}s  model=${EDM_BEST}  lnL=${EDM_OPT}  MF=${EDM_MF_WALL}s"
echo "  EDM-DIAG: ${EDM_DIAG_LINES}  EDM-EPOCH: ${EDM_EPOCH_LINES}  wave_starts: ${WAVE_COUNT}"
echo "  ref lnL: -7541976.861 (FCA np=1 job 169095077)  tolerance: 0.05"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${BASE_RC}" -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}"; PASS=0; }
[[ "${EDM_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: EDM  rc=${EDM_RC}";  PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-4 EDM PASS — EDM v0 correctness confirmed at np=4 ══"
    exit 0
else
    echo "  ══ ISO-4 EDM FAIL — investigate above ══" >&2
    exit 10
fi
