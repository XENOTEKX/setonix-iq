#!/bin/bash
# run_p7_aa1m_np16_p3.sh — P.7 performance gate: Mode P MPGC at scale.
#
# PURPOSE: Verify that Mode P with P.6 cost-threshold dispatch (--mode-p)
# completes ModelFinder on AA 1M at np=16 in ≤ 600 s wall clock.
#
# FCA np=16 baseline (job 168635616): MF wall = 1,122.363 s, total = 2,410.226 s.
# Target speedup: ≥ 1.87× improvement over FCA np=16 MF phase.
#
# Single sub-run: MODEP — np=16, -m TEST, --mode-p (P.6 auto-dispatcher), fixed tree.
# Starting tree from ATMD b3c np=16 run (169112256) so MF phase timing is clean.
#
# Gate pass criteria:
#   1. MODEP exits 0 (no crash)
#   2. MODEP best model = LG+G4
#   3. |MODEP lnL - REF_LNL| < 2.0  (generous FP band at np=16)
#   4. Wall-clock time for ModelFinder ≤ 600 s  (primary perf gate)
#   5. P6-DIAG threshold line found in output
#   6. At least one dispatch=MODEP (heavy model used Mode P)
#   7. At least one dispatch=FCA   (light model used FCA path)
#   8. [Mode P] lines from all 16 ranks
#
# References:
#   FCA np=16  job 168635616  lnL=-78,605,196.497  MF=1,122.363s  (primary perf ref)
#   FCA np=1   job 168913089  lnL=-78,605,196.590  (lnL reference for tolerance)
#   ISO-5 PASS job 169207131  binary cc3d403f  (correctness verified at np=4)

#PBS -N p7-perf-np16
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
# Starting tree from ATMD b3c np=16 run on AA 1M (job 169112256):
REF_TREE="${REF_TREE:-${SANDBOX}/../../mf_iso/profiles/AA_1m_atmd_b3c_np16_full_seed1_169112256/iqtree_inner.treefile}"
# Absolute path override in case relative fails:
REF_TREE_ABS="/scratch/rc29/as1708/mf_iso/profiles/AA_1m_atmd_b3c_np16_full_seed1_169112256/iqtree_inner.treefile"
[[ -f "${REF_TREE}" ]] || REF_TREE="${REF_TREE_ABS}"

NRANKS=16
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

# Expected binary: ATMD-AID canonical checkpoint/warm-start sync fix.
EXPECTED_MD5="3e79db194ced77971a55c6a0ff476863"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="p7_p3_aa1m_np16_seed${SEED}"
WORK_DIR="${SANDBOX}/runs/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7              2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
[[ -f "${REF_TREE}" ]]  || { echo "ERROR: starting tree not found: ${REF_TREE}" >&2; exit 4; }
MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
if [[ "${MD5}" != "${EXPECTED_MD5}" ]]; then
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
[[ "${#HOSTS[@]}" -ge "${NRANKS}" ]] || { echo "ERROR: expected >=${NRANKS} nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

# 1 rank per node, pinned to slots 0-103 (all 104 physical cores)
RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
rank 1=${HOSTS[1]} slot=0-103
rank 2=${HOSTS[2]} slot=0-103
rank 3=${HOSTS[3]} slot=0-103
rank 4=${HOSTS[4]} slot=0-103
rank 5=${HOSTS[5]} slot=0-103
rank 6=${HOSTS[6]} slot=0-103
rank 7=${HOSTS[7]} slot=0-103
rank 8=${HOSTS[8]} slot=0-103
rank 9=${HOSTS[9]} slot=0-103
rank 10=${HOSTS[10]} slot=0-103
rank 11=${HOSTS[11]} slot=0-103
rank 12=${HOSTS[12]} slot=0-103
rank 13=${HOSTS[13]} slot=0-103
rank 14=${HOSTS[14]} slot=0-103
rank 15=${HOSTS[15]} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  P.7 perf gate: Mode P MPGC, np=16, AA 1M, -m TEST               ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  tree:    $(basename "${REF_TREE}") (ATMD b3c np=16 treefile)"
echo "║  FCA ref: MF=1,122.363s total=2,410.226s (job 168635616)          ║"
echo "║  Target:  MF wall ≤ 600 s  (≥ 1.87× speedup over FCA np=16)      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ──────────────────────────────────────────────────
# Sub-run: MODEP — np=16, --mode-p (P.6 auto-dispatcher), -m TEST, fixed tree
# ──────────────────────────────────────────────────
MODEP_DIR="${WORK_DIR}/modep_np16"; mkdir -p "${MODEP_DIR}"
MODEP_RANK_LOGS="${MODEP_DIR}/rank_logs"; mkdir -p "${MODEP_RANK_LOGS}"

echo ""
echo "── Sub-run MODEP (np=16, --mode-p / P.6 auto-dispatcher) ────────"
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
                -te "${REF_TREE}" \
                --atmd-aid \
                --atmd-aid-heavy-mult 1.5 \
                --atmd-k-outer 1 \
                --prefix "${MODEP_DIR}/iqtree_inner" \
    > "${MODEP_DIR}/iqtree_stdout.log" 2>&1
MODEP_RC=$?
END_MODEP=$(date +%s)
WALL_MODEP=$(( END_MODEP - START_MODEP ))
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s"

# ──────────────────────────────────────────────────
# Parse results
# ──────────────────────────────────────────────────
MODEP_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{printf "%.6f\n", v}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
MODEP_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
MODEP_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

# Count [Mode P] lines per rank
echo ""
echo "── [Mode P] partition lines (MODEP, first 16) ───────────────────"
{ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | head -16

MODE_P_TOTAL=$({ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | wc -l)

MODE_P_RANKS=$({ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } \
    | grep -oP 'rank \K[0-9]+' | sort -un | wc -l)

# P.6 dispatcher diagnostics
echo ""
echo "── P.6 dispatcher diagnostics (MODEP) ──────────────────────────"
{ grep -h 'P6-DIAG:\|dispatch=' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | head -20

P6_DIAG_LINES=$({ grep -ch 'P6-DIAG:' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | awk '{s+=$1} END{print s+0}')

P6_MODEP_COUNT=$({ grep -h 'dispatch=MODEP' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | wc -l)

P6_FCA_COUNT=$({ grep -h 'dispatch=FCA' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | wc -l)

# ──────────────────────────────────────────────────
# Numeric pass/fail (Python)
# ──────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - \
    "${MODEP_OPT}" \
    "${MODEP_BEST}" \
    "${MODE_P_TOTAL}" "${MODE_P_RANKS}" \
    "${WALL_MODEP}" \
    "${MODEP_MF_WALL:-0}" \
    "${P6_DIAG_LINES}" "${P6_MODEP_COUNT}" "${P6_FCA_COUNT}" <<'PYEOF'
import sys

modep_opt      = sys.argv[1]
modep_best     = sys.argv[2]
mp_total       = int(sys.argv[3])
mp_ranks       = int(sys.argv[4])
wall_modep     = int(sys.argv[5])
mf_wall_str    = sys.argv[6]
p6_diag_lines  = int(sys.argv[7])
p6_modep_count = int(sys.argv[8])
p6_fca_count   = int(sys.argv[9])

REF_LNL        = -78605196.497   # FCA np=16 job 168635616 lnL (MF+SPR)
LNL_TOL        = 2.0             # generous — FP non-associativity at np=16
EXPECTED_NP    = 16
EXPECTED_MODEL = "LG+G4"
FCA_MF_WALL    = 1122.363        # FCA np=16 baseline MF wall (job 168635616)
TARGET_MF_WALL = 600             # perf gate: ≤ 600 s to PASS

try:
    mf_wall_s = float(mf_wall_str)
except (ValueError, TypeError):
    mf_wall_s = 0.0

lines = []
ok = True

# 1. MODEP lnL vs reference
if modep_opt:
    try:
        delta = abs(float(modep_opt) - REF_LNL)
        sym = "✓" if delta <= LNL_TOL else "✗"
        lines.append(f"  {sym} MODEP lnL={modep_opt}  |Δ vs ref|={delta:.4f}  tol={LNL_TOL}")
        if delta > LNL_TOL:
            ok = False
    except ValueError:
        lines.append("  ✗ MODEP lnL could not be parsed"); ok = False
else:
    lines.append("  ✗ MODEP optimal lnL not found in log"); ok = False

# 2. Best model
if modep_best == EXPECTED_MODEL:
    lines.append(f"  ✓ MODEP best model: {modep_best}")
else:
    lines.append(f"  ✗ MODEP best model: {modep_best}  (expected {EXPECTED_MODEL})")
    ok = False

# 3. MF wall ≤ TARGET (primary perf gate)
if mf_wall_s > 0:
    speedup_mf = FCA_MF_WALL / mf_wall_s
    sym = "✓" if mf_wall_s <= TARGET_MF_WALL else "✗"
    lines.append(f"  {sym} MF wall: {mf_wall_s:.3f} s  (target ≤ {TARGET_MF_WALL} s,"
                 f" speedup vs FCA np=16: {speedup_mf:.2f}×)")
    if mf_wall_s > TARGET_MF_WALL:
        ok = False
        if mf_wall_s < FCA_MF_WALL:
            lines.append(f"  ⚠  partial speedup: {speedup_mf:.2f}× (beats FCA np=16 but misses ≤{TARGET_MF_WALL}s target)")
else:
    lines.append("  ✗ MF wall not parsed from log (Wall-clock time for ModelFinder: line missing)")
    ok = False

# 4. Total wall (informational)
if wall_modep > 0:
    lines.append(f"  ℹ  total wall: {wall_modep} s  (FCA np=16 total was 2410 s)")

# 5. [Mode P] lines on all 16 ranks
if mp_total > 0 and mp_ranks >= EXPECTED_NP:
    lines.append(f"  ✓ [Mode P] lines: {mp_total} total across {mp_ranks} ranks (all {EXPECTED_NP} present)")
elif mp_total > 0:
    lines.append(f"  ✗ [Mode P] lines: {mp_total} total but only {mp_ranks}/{EXPECTED_NP} ranks emitted")
    ok = False
else:
    lines.append("  ✗ No [Mode P] lines emitted"); ok = False

# 6. P.6 diagnostic: threshold was computed
if p6_diag_lines > 0:
    lines.append(f"  ✓ P6-DIAG: threshold line found ({p6_diag_lines} occurrences)")
else:
    lines.append("  ✗ P6-DIAG: no threshold line found — P.6 code may not be compiled in")
    ok = False

# 7. P.6 dispatch: at least one heavy model used Mode P
if p6_modep_count > 0:
    lines.append(f"  ✓ P.6 dispatch=MODEP: {p6_modep_count} model evaluations used Mode P (heavy)")
else:
    lines.append("  ✗ P.6 dispatch=MODEP: 0 — no heavy models dispatched to Mode P")
    ok = False

# 8. P.6 dispatch: at least one light model used FCA serial path
if p6_fca_count > 0:
    lines.append(f"  ✓ P.6 dispatch=FCA:   {p6_fca_count} model evaluations used FCA serial (light)")
else:
    lines.append("  ✗ P.6 dispatch=FCA:   0 — no light models used FCA path (threshold may be too high)")
    ok = False

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ P.7 result (perf gate, np=16, AA 1M, Mode P MPGC) ══════════════"
echo "  binary md5: ${MD5}"
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s model=${MODEP_BEST} lnL=${MODEP_OPT} MF=${MODEP_MF_WALL}s"
echo "  [Mode P] total=${MODE_P_TOTAL} distinct_ranks=${MODE_P_RANKS}"
echo "  P6 diag_lines=${P6_DIAG_LINES} dispatch=MODEP:${P6_MODEP_COUNT} dispatch=FCA:${P6_FCA_COUNT}"
echo "  FCA np=16 ref: MF=1122.363s total=2410.226s lnL=-78605196.497 (job 168635616)"
echo "  Target: MF wall ≤ 600 s"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${MODEP_RC}" -eq 0 ]] || { echo "  ✗ FAIL: MODEP rc=${MODEP_RC}"; PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ P.7 PASS — Mode P np=16 MF wall ≤ 600 s; perf gate cleared ══"
    exit 0
else
    echo "  ══ P.7 FAIL — investigate above ══" >&2
    echo "  See research/Modelfinder/mode-p-implementation-status.md §P.7 for context" >&2
    exit 10
fi
