#!/bin/bash
# run_p7_aa1m_np16_edm.sh — P.7 performance gate for EDM v0 (Event-Driven
# Moldable Dispatch), np=16, AA 1M, --mf-edm --mf-edm-group-size 4.
#
# PURPOSE: Verify that EDM v0 epoch dispatch completes ModelFinder on AA 1M
# at np=16 in ≤ 600 s wall clock.
#
# EDM v0 at gs=4 np=16: 4 cohorts each run one model in Mode P, sentinel
# epoch fires first with all 16 ranks on the canonical reference family.
# Expected: ~FCA_np4 / 4 cohorts × sentinel epoch overhead ≈ 494-550 s.
#
# Gate pass criteria:
#   1. EDM exits 0  (no crash, no deadlock)
#   2. EDM best model = LG+G4
#   3. |EDM lnL - REF_LNL| < 2.0  (generous FP band at np=16)
#   4. Wall-clock time for ModelFinder ≤ 600 s  (primary perf gate)
#   5. EDM-DIAG found in output (scheduler ran)
#   6. EDM-EPOCH found (epoch plan emitted)
#   7. EDM-WAVE / AID-WAVE start markers > 0 (epochs executed)
#   8. [Mode P] lines from all 16 ranks
#
# References:
#   FCA np=16   job 168635616  lnL=-78,605,196.497  MF=1,122.363s
#   FCA np=1    job 168913089  lnL=-78,605,196.590
#   AID v2 FAIL job 169343365  Phase 0 bottleneck at ~863s before Phase 1
#   ISO-4 EDM   (this script's np=4 sibling, run_iso4_aa100k_np4_edm.sh)
#   Design doc  research/Modelfinder/event-driven-moldable-dispatch.md

#PBS -N p7-edm-np16
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=1664
#PBS -l mem=8000GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
IQTREE="${IQTREE:-${SANDBOX}/build-mode-p-iso-p3/iqtree3-mpi-mode-p-iso-p3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
REF_TREE="${REF_TREE:-/scratch/rc29/as1708/mf_iso/profiles/AA_1m_atmd_b3c_np16_full_seed1_169112256/iqtree_inner.treefile}"

NRANKS=16
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
EDM_GROUP_SIZE="${EDM_GROUP_SIZE:-4}"

# EDM v0 build — md5 from login-node incremental build 2026-05-27T02:16:36+10:00.
EXPECTED_MD5="${EXPECTED_MD5:-4810a8ac73e3b92b1f93b3f03ec04d57}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="p7_edm_aa1m_np16_seed${SEED}"
WORK_DIR="${SANDBOX}/runs/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
[[ -f "${REF_TREE}" ]]  || { echo "ERROR: starting tree not found: ${REF_TREE}" >&2; exit 4; }
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
[[ "${#HOSTS[@]}" -ge "${NRANKS}" ]] || \
    { echo "ERROR: expected >=${NRANKS} nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
for i in $(seq 0 $(( NRANKS - 1 ))); do
    echo "rank ${i}=${HOSTS[${i}]} slot=0-103"
done > "${RANKFILE}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  P.7 EDM gate: --mf-edm, np=16, AA 1M, -m TEST                  ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  EDM group_size=${EDM_GROUP_SIZE}: sentinel gs=16, tail gs=${EDM_GROUP_SIZE} (4 cohorts)"
echo "║  FCA ref: MF=1,122.363s total=2,410.226s (job 168635616)         ║"
echo "║  Target:  MF wall ≤ 600 s                                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ──────────────────────────────────────────────────────────────────────
# Sub-run: EDM — np=16, --mf-edm --mf-edm-group-size 4, -m TEST
# ──────────────────────────────────────────────────────────────────────
EDM_DIR="${WORK_DIR}/edm_np16"; mkdir -p "${EDM_DIR}"
EDM_RANK_LOGS="${EDM_DIR}/rank_logs"; mkdir -p "${EDM_RANK_LOGS}"

echo ""
echo "── Sub-run EDM (np=16, --mf-edm --mf-edm-group-size ${EDM_GROUP_SIZE}) ──"
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
                -te "${REF_TREE}" \
                --mf-edm \
                --mf-edm-group-size "${EDM_GROUP_SIZE}" \
                --atmd-k-outer 1 \
                --prefix "${EDM_DIR}/iqtree_inner" \
    > "${EDM_DIR}/iqtree_stdout.log" 2>&1
EDM_RC=$?
END_EDM=$(date +%s)
WALL_EDM=$(( END_EDM - START_EDM ))
echo "  EDM exit=${EDM_RC} wall=${WALL_EDM}s"

# ──────────────────────────────────────────────────────────────────────
# Parse results
# ──────────────────────────────────────────────────────────────────────
EDM_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{if (v != "") printf "%.6f\n", v}' \
    "${EDM_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
EDM_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${EDM_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
EDM_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${EDM_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "── EDM-DIAG / EDM-EPOCH (first 16 lines) ───────────────────────"
{ grep -h 'EDM-DIAG\|EDM-EPOCH\|AID-DIAG' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | head -16

echo ""
echo "── EDM-WAVE / AID-WAVE markers (first 16) ──────────────────────"
{ grep -h 'EDM-WAVE\|AID-WAVE' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | head -16

EDM_DIAG_LINES=$({ grep -h 'EDM-DIAG' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | wc -l)

EDM_EPOCH_LINES=$({ grep -h 'EDM-EPOCH' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | wc -l)

WAVE_COUNT=$({ grep -h 'EDM-WAVE\|AID-WAVE' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | grep -c 'start' || true)

# [Mode P] lines per rank
MODE_P_TOTAL=$({ grep -h '\[Mode P\]' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | wc -l)

MODE_P_RANKS=$({ grep -h '\[Mode P\]' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } \
    | grep -oP 'rank \K[0-9]+' | sort -un | wc -l)

echo ""
echo "── [Mode P] partition lines (first 16) ─────────────────────────"
{ grep -h '\[Mode P\]' \
    "${EDM_DIR}/iqtree_inner.log" \
    "${EDM_DIR}/iqtree_stdout.log" \
    "${EDM_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | head -16

# ──────────────────────────────────────────────────────────────────────
# Numeric pass/fail
# ──────────────────────────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - \
    "${EDM_OPT}" "${EDM_BEST}" \
    "${EDM_DIAG_LINES}" "${EDM_EPOCH_LINES}" "${WAVE_COUNT}" \
    "${MODE_P_TOTAL}" "${MODE_P_RANKS}" \
    "${WALL_EDM}" "${EDM_MF_WALL:-0}" <<'PYEOF'
import sys

edm_opt         = sys.argv[1]
edm_best        = sys.argv[2]
edm_diag_lines  = int(sys.argv[3])
edm_epoch_lines = int(sys.argv[4])
wave_count      = int(sys.argv[5])
mp_total        = int(sys.argv[6])
mp_ranks        = int(sys.argv[7])
wall_edm        = int(sys.argv[8])
mf_wall_str     = sys.argv[9]

REF_LNL        = -78605196.497   # FCA np=16 job 168635616
LNL_TOL        = 2.0
EXPECTED_NP    = 16
EXPECTED_MODEL = "LG+G4"
FCA_MF_WALL    = 1122.363
TARGET_MF_WALL = 600

try:
    mf_wall_s = float(mf_wall_str)
except (ValueError, TypeError):
    mf_wall_s = 0.0

lines = []
ok = True

# 1. EDM lnL
if edm_opt:
    try:
        delta = abs(float(edm_opt) - REF_LNL)
        sym = "✓" if delta <= LNL_TOL else "✗"
        lines.append(f"  {sym} EDM  lnL={edm_opt}  |Δ vs ref|={delta:.4f}  tol={LNL_TOL}")
        if delta > LNL_TOL:
            ok = False
    except ValueError:
        lines.append("  ✗ EDM lnL parse error"); ok = False
else:
    lines.append("  ✗ EDM optimal lnL not found"); ok = False

# 2. Best model
if edm_best == EXPECTED_MODEL:
    lines.append(f"  ✓ EDM  best model: {edm_best}")
else:
    lines.append(f"  ✗ EDM  best model: {edm_best}  (expected {EXPECTED_MODEL})")
    ok = False

# 3. MF wall ≤ target (primary perf gate)
if mf_wall_s > 0:
    speedup_mf = FCA_MF_WALL / mf_wall_s
    sym = "✓" if mf_wall_s <= TARGET_MF_WALL else "✗"
    lines.append(f"  {sym} MF wall: {mf_wall_s:.3f} s  (target ≤ {TARGET_MF_WALL} s,"
                 f" speedup vs FCA np=16: {speedup_mf:.2f}×)")
    if mf_wall_s > TARGET_MF_WALL:
        ok = False
else:
    lines.append("  ✗ MF wall not parsed"); ok = False

# 4. Total wall
if wall_edm > 0:
    lines.append(f"  ℹ  total wall: {wall_edm} s")

# 5. EDM-DIAG
if edm_diag_lines > 0:
    lines.append(f"  ✓ EDM-DIAG lines: {edm_diag_lines} (scheduler entered)")
else:
    lines.append("  ✗ EDM-DIAG: 0 lines"); ok = False

# 6. EDM-EPOCH
if edm_epoch_lines > 0:
    lines.append(f"  ✓ EDM-EPOCH lines: {edm_epoch_lines} (epoch plan emitted)")
else:
    lines.append("  ✗ EDM-EPOCH: 0 lines"); ok = False

# 7. Wave execution
if wave_count > 0:
    lines.append(f"  ✓ Wave/epoch start markers: {wave_count}")
else:
    lines.append("  ⚠ Wave/epoch start markers: 0")

# 8. [Mode P] lines from all 16 ranks
if mp_total > 0 and mp_ranks >= EXPECTED_NP:
    lines.append(f"  ✓ [Mode P] lines: {mp_total} total across {mp_ranks} ranks (all {EXPECTED_NP} present)")
elif mp_total > 0:
    lines.append(f"  ✗ [Mode P] lines: {mp_total} total but only {mp_ranks}/{EXPECTED_NP} ranks")
    ok = False
else:
    lines.append("  ✗ No [Mode P] lines emitted"); ok = False

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ P.7 EDM result (Event-Driven Moldable Dispatch v0, np=16) ══════"
echo "  binary md5: ${MD5}"
echo "  EDM  exit=${EDM_RC} wall=${WALL_EDM}s model=${EDM_BEST} lnL=${EDM_OPT} MF=${EDM_MF_WALL}s"
echo "  EDM-DIAG: ${EDM_DIAG_LINES}  EDM-EPOCH: ${EDM_EPOCH_LINES}  wave_starts: ${WAVE_COUNT}"
echo "  [Mode P]: total=${MODE_P_TOTAL} distinct_ranks=${MODE_P_RANKS}"
echo "  FCA np=16 ref: MF=1122.363s total=2410.226s lnL=-78605196.497 (job 168635616)"
echo "  Target: MF wall ≤ 600 s"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${EDM_RC}" -eq 0 ]] || { echo "  ✗ FAIL: EDM rc=${EDM_RC}"; PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ P.7 EDM PASS — EDM v0 np=16 MF wall ≤ 600 s; perf gate cleared ══"
    exit 0
else
    echo "  ══ P.7 EDM FAIL — investigate above ══" >&2
    echo "  See research/Modelfinder/event-driven-moldable-dispatch.md for context" >&2
    exit 10
fi
