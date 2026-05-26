#!/bin/bash
# run_iso5_aa100k_np4_p3.sh — ISO-5 gate: P.6 cost-threshold dispatch correctness.
#
# PURPOSE: Verify that Mode P with cost-threshold dispatch (--mode-p, mode_p_enabled==1)
# correctly routes only heavy models (cost >= avg_cost × mode_p_min_cost_mult) through
# Mode P, while light models fall back to FCA serial path (full-pattern, no Allreduce).
#
# Two sub-runs inside one PBS job:
#   sub-run BASE  — np=1, -m TEST, no Mode P  — reference ModelFinder + NNI
#   sub-run MODEP — np=4, -m TEST, --mode-p   — auto-dispatcher (cost-threshold)
#
# Both start from the ISO-2 PASS treefile (fixed topology via -te) so the
# starting point is the same; ModelFinder re-evaluates all models on that tree.
#
# Gate pass criteria:
#   1. Both sub-runs exit 0 (no crash)
#   2. MODEP best model = LG+G4 (matches BASE and FCA reference)
#   3. |MODEP lnL - REF_LNL| < 0.05  (FP non-associativity band; Allreduce
#      reorders summation vs FCA np=1 single-rank sequential sum)
#   4. |BASE  lnL - REF_LNL| < 0.05
#   5. P6-DIAG: line appears in MODEP output (threshold was computed)
#   6. At least one MF-TIME line with dispatch=MODEP (≥1 heavy model used Mode P)
#   7. At least one MF-TIME line with dispatch=FCA   (≥1 light model used FCA path)
#   8. [Mode P] partition lines from all 4 ranks in MODEP run
#
# Key difference from ISO-4: uses --mode-p (not --mode-p-all) so the cost-threshold
# dispatcher (P.6) is active. Heavy models (e.g. LG+F+I+G4) use Mode P; light
# models (e.g. LG, WAG, LG+I) evaluate full-pattern locally without Allreduce.
#
# References:
#   FCA np=1  job 169095077  lnL=-7,541,976.861  (primary reference)
#   ISO-2 PASS job 169135061 binary 50b4b172
#   ISO-4 PASS job 169197750 lnL=-7,541,976.852 (MODEP np=4 --mode-p-all)

#PBS -N iso5-p3-np4
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
# Starting tree from ISO-2 PASS (job 169135061, binary 50b4b172):
ISO2_TREE="${ISO2_TREE:-${SANDBOX}/runs/iso2_p3_aa100k_np2_seed1_169135061/iqtree_inner.treefile}"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

# Expected binary: cc3d403f (B.4-12/13/14/15 + P.6 cost-threshold dispatch, default mult=1.5, build 169202992)
EXPECTED_MD5="cc3d403f9aac4eb44f3ef022efcce8d8"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso5_p3_aa100k_np4_seed${SEED}"
WORK_DIR="${SANDBOX}/runs/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7              2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]] || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
[[ -f "${ISO2_TREE}" ]] || { echo "ERROR: ISO-2 starting tree not found: ${ISO2_TREE}" >&2; exit 4; }
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
[[ "${#HOSTS[@]}" -ge 4 ]] || { echo "ERROR: expected >=4 nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

# 1 rank per node, pinned to slots 0-103 (all 104 physical cores)
RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
rank 1=${HOSTS[1]} slot=0-103
rank 2=${HOSTS[2]} slot=0-103
rank 3=${HOSTS[3]} slot=0-103
EOF

# BASE uses only node 0
RANKFILE_BASE="${WORK_DIR}/rankfile_base.txt"
cat > "${RANKFILE_BASE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ISO-5 gate: P.6 cost-threshold dispatch, np=4, AA 100K, -m TEST ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  tree:    $(basename "${ISO2_TREE}") (ISO-2 PASS treefile)"
echo "║  Purpose: --mode-p auto-dispatcher: heavy→Mode P, light→FCA      ║"
echo "║  Ref lnL: -7,541,976.861 (FCA np=1 job 169095077)                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ──────────────────────────────────────────────────
# Sub-run 1: BASE — np=1, no Mode P, -m TEST, fixed tree
# ──────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────
# Sub-run 2: MODEP — np=4, --mode-p (P.6 auto-dispatcher), -m TEST, fixed tree
# ──────────────────────────────────────────────────
MODEP_DIR="${WORK_DIR}/modep_np4"; mkdir -p "${MODEP_DIR}"
MODEP_RANK_LOGS="${MODEP_DIR}/rank_logs"; mkdir -p "${MODEP_RANK_LOGS}"

echo ""
echo "── Sub-run MODEP (np=4, --mode-p / P.6 auto-dispatcher) ────────"
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
                --mode-p \
                --mode-p-min-cost-mult 1.5 \
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
BASE_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{printf "%.6f\n", v}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
MODEP_OPT=$(awk '/Optimal log-likelihood:/{v=$NF} END{printf "%.6f\n", v}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null || echo "")
BASE_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
MODEP_BEST=$(awk -F': ' '/Best-fit model:/{print $2; exit}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | awk '{print $1}' || echo "")
BASE_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")
MODEP_MF_WALL=$(awk '/Wall-clock time for ModelFinder:/{print $NF}' \
    "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

# Count [Mode P] lines per rank
echo ""
echo "── [Mode P] partition lines (MODEP, first 8) ────────────────────"
{ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/*/stderr 2>/dev/null || true; } | head -8

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
    "${BASE_OPT}" "${MODEP_OPT}" \
    "${BASE_BEST}" "${MODEP_BEST}" \
    "${MODE_P_TOTAL}" "${MODE_P_RANKS}" \
    "${WALL_BASE}" "${WALL_MODEP}" \
    "${P6_DIAG_LINES}" "${P6_MODEP_COUNT}" "${P6_FCA_COUNT}" <<'PYEOF'
import sys

base_opt       = sys.argv[1]
modep_opt      = sys.argv[2]
base_best      = sys.argv[3]
modep_best     = sys.argv[4]
mp_total       = int(sys.argv[5])
mp_ranks       = int(sys.argv[6])
wall_base      = int(sys.argv[7])
wall_modep     = int(sys.argv[8])
p6_diag_lines  = int(sys.argv[9])
p6_modep_count = int(sys.argv[10])
p6_fca_count   = int(sys.argv[11])

REF_LNL        = -7541976.861
LNL_TOL        = 0.05        # FP non-associativity band from Allreduce reordering
EXPECTED_NP    = 4
EXPECTED_MODEL = "LG+G4"

lines = []
ok = True

# 1. MODEP lnL vs reference
if modep_opt:
    delta = abs(float(modep_opt) - REF_LNL)
    sym = "✓" if delta <= LNL_TOL else "✗"
    lines.append(f"  {sym} MODEP lnL={modep_opt}  |Δ vs ref|={delta:.4f}  tol={LNL_TOL}")
    if delta > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ MODEP optimal lnL not parsed"); ok = False

# 2. BASE lnL vs reference
if base_opt:
    delta_b = abs(float(base_opt) - REF_LNL)
    sym = "✓" if delta_b <= LNL_TOL else "✗"
    lines.append(f"  {sym} BASE  lnL={base_opt}   |Δ vs ref|={delta_b:.4f}")
    if delta_b > LNL_TOL:
        ok = False
else:
    lines.append("  ✗ BASE optimal lnL not parsed"); ok = False

# 3. Best model agreement
if modep_best == EXPECTED_MODEL:
    lines.append(f"  ✓ MODEP best model: {modep_best}")
else:
    lines.append(f"  ✗ MODEP best model: {modep_best}  (expected {EXPECTED_MODEL})")
    ok = False
if base_best:
    sym = "✓" if base_best == EXPECTED_MODEL else "⚠"
    lines.append(f"  {sym} BASE  best model: {base_best}")

# 4. [Mode P] lines on all 4 ranks
if mp_total > 0 and mp_ranks >= EXPECTED_NP:
    lines.append(f"  ✓ [Mode P] lines: {mp_total} total across {mp_ranks} ranks (all {EXPECTED_NP} present)")
elif mp_total > 0:
    lines.append(f"  ✗ [Mode P] lines: {mp_total} total but only {mp_ranks}/{EXPECTED_NP} ranks emitted")
    ok = False
else:
    lines.append("  ✗ No [Mode P] lines emitted"); ok = False

# 5. P.6 diagnostic: threshold was computed
if p6_diag_lines > 0:
    lines.append(f"  ✓ P6-DIAG: threshold line found ({p6_diag_lines} occurrences)")
else:
    lines.append("  ✗ P6-DIAG: no threshold line found — P.6 code may not be compiled in")
    ok = False

# 6. P.6 dispatch: at least one heavy model used Mode P
if p6_modep_count > 0:
    lines.append(f"  ✓ P.6 dispatch=MODEP: {p6_modep_count} model evaluations used Mode P (heavy)")
else:
    lines.append("  ✗ P.6 dispatch=MODEP: 0 — no heavy models dispatched to Mode P")
    ok = False

# 7. P.6 dispatch: at least one light model used FCA serial path
if p6_fca_count > 0:
    lines.append(f"  ✓ P.6 dispatch=FCA:   {p6_fca_count} model evaluations used FCA serial (light)")
else:
    lines.append("  ✗ P.6 dispatch=FCA:   0 — no light models used FCA path (threshold may be too high)")
    ok = False

# 8. Speedup (informational; not a hard gate)
if wall_base > 0 and wall_modep > 0:
    speedup = wall_base / wall_modep
    lines.append(f"  {'✓' if speedup >= 1.0 else '⚠'} Speedup: {wall_base}s → {wall_modep}s = {speedup:.2f}×"
                 f"  (ISO-4 was 1.79× with --mode-p-all)")

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ ISO-5 result (P.6 cost-threshold dispatch, np=4, correctness gate) ════"
echo "  binary md5: ${MD5}"
echo "  BASE  exit=${BASE_RC}  wall=${WALL_BASE}s  model=${BASE_BEST}  lnL=${BASE_OPT}  MF=${BASE_MF_WALL}s"
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s model=${MODEP_BEST} lnL=${MODEP_OPT} MF=${MODEP_MF_WALL}s"
echo "  [Mode P] total=${MODE_P_TOTAL} distinct_ranks=${MODE_P_RANKS}"
echo "  P6 diag_lines=${P6_DIAG_LINES} dispatch=MODEP:${P6_MODEP_COUNT} dispatch=FCA:${P6_FCA_COUNT}"
echo "  ref lnL: -7541976.861 (FCA np=1 job 169095077)  tolerance: 0.05"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${BASE_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}";  PASS=0; }
[[ "${MODEP_RC}" -eq 0 ]] || { echo "  ✗ FAIL: MODEP rc=${MODEP_RC}"; PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-5 PASS — P.6 cost-threshold dispatch correct; proceed to P.7 perf gate ══"
    exit 0
else
    echo "  ══ ISO-5 FAIL — investigate above ══" >&2
    echo "  See research/Modelfinder/mode-p-implementation-status.md §ISO-5 for context" >&2
    exit 10
fi
