#!/bin/bash
# run_iso4_aa100k_np4_p3.sh — ISO-4 gate: 4-rank Mode P correctness + partition check.
#
# PURPOSE: Verify that Mode P (P.3 BranchSIMD + P.4 DervSIMD + P.5a FromBuffer)
# produces correct results under np=4 (4 nodes, 4 MPI ranks, 103 OMP threads/rank).
#
# Two sub-runs inside one PBS job:
#   sub-run BASE  — np=1, -m TEST, no Mode P  — reference ModelFinder + NNI
#   sub-run MODEP — np=4, -m TEST, --mode-p-all — 4-way pattern partition
#
# Both start from the ISO-2 PASS treefile (fixed topology via -te) so the
# starting point is the same; ModelFinder re-evaluates all models on that tree.
#
# Gate pass criteria (updated from ISO-3 experience — NR-trace step comparison
# is NOT applicable under Mode P because Allreduce changes FP summation order):
#   1. Both sub-runs exit 0 (no crash)
#   2. [Mode P] partition lines from all 4 ranks in MODEP run
#      (each rank logs its [ptn_lo, ptn_hi) slice — 4 distinct ranges)
#   3. MODEP best model = LG+G4 (matches BASE and FCA reference)
#   4. |MODEP lnL - REF_LNL| < 0.05  (FP non-associativity band; Allreduce
#      reorders summation vs FCA np=1 single-rank sequential sum)
#   5. |BASE  lnL - REF_LNL| < 0.05
#
# ISO-3 lessons applied here:
#   - --atmd-k-outer 1 RESTORED to MODEP sub-run (B.4-12 fix, 2026-05-25):
#     K_outer=8 (default) dispatches 8 OMP worker threads simultaneously,
#     each calling Mode P MPI_Allreduce → violates MPI_THREAD_FUNNELED →
#     "Message truncated" SEGFAULT at LG+F+I (model 6). F-5 invariant
#     requires K_outer=1 for all Mode P runs. The step-count asymmetry
#     observed in ISO-3 (BASE=92 vs MODEP=61) is EXPECTED: Mode P pattern-
#     partitioned Allreduce changes the gradient landscape, so NNI step
#     counts legitimately differ from K_outer=8 single-rank; this is NOT
#     a correctness issue. Criterion 4 (lnL gate) is the correctness test.
#   - NR trace 1e-6 per-step compare removed (inherently fails for Mode P)
#   - lnL tolerance relaxed to 0.05 (Mode P Allreduce FP non-assoc ≈ 0.008)
#
# References:
#   FCA np=1  job 169095077  lnL=-7,541,976.861  (primary reference)
#   FCA np=2  job 168584736  lnL=-7,541,976.853  (informational)
#   ISO-2 PASS job 169135061 binary 50b4b172
#   ISO-3 PASS job 169136469 binary 9660575a  lnL=-7,541,976.861 (MODEP np=2)

#PBS -N iso4-p3-np4
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

# Expected binary: 9660575a (B.4-9 patched, job 169136419)
EXPECTED_MD5="76cdfb199f7765b58d4e7f59cf22fdf0"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso4_p3_aa100k_np4_seed${SEED}"
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

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ISO-4 gate: P.3+P.4+P.5a, np=4, AA 100K, -m TEST           ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  tree:    $(basename "${ISO2_TREE}") (ISO-2 PASS treefile)"
echo "║  Purpose: 4-rank Mode P correctness (no NR-trace; lnL+model) ║"
echo "║  Ref lnL: -7,541,976.861 (FCA np=1 job 169095077)            ║"
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
# Sub-run 2: MODEP — np=4, --mode-p-all, -m TEST, fixed tree
# ─────────────────────────────────────────────────────────────
MODEP_DIR="${WORK_DIR}/modep_np4"; mkdir -p "${MODEP_DIR}"
MODEP_RANK_LOGS="${MODEP_DIR}/rank_logs"; mkdir -p "${MODEP_RANK_LOGS}"

echo ""
echo "── Sub-run MODEP (np=4, --mode-p-all) ──────────────────────────"
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
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | head -8

MODE_P_TOTAL=$({ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null || true; } | wc -l)

# Count how many distinct rank numbers appear in [Mode P] lines
MODE_P_RANKS=$({ grep -h '\[Mode P\]' \
    "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null || true; } \
    | grep -oP 'rank \K[0-9]+' | sort -un | wc -l)

# ─────────────────────────────────────────────────────────────
# Numeric pass/fail (Python)
# ─────────────────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - \
    "${BASE_OPT}" "${MODEP_OPT}" \
    "${BASE_BEST}" "${MODEP_BEST}" \
    "${MODE_P_TOTAL}" "${MODE_P_RANKS}" \
    "${WALL_BASE}" "${WALL_MODEP}" <<'PYEOF'
import sys

base_opt      = sys.argv[1]
modep_opt     = sys.argv[2]
base_best     = sys.argv[3]
modep_best    = sys.argv[4]
mp_total      = int(sys.argv[5])
mp_ranks      = int(sys.argv[6])
wall_base     = int(sys.argv[7])
wall_modep    = int(sys.argv[8])

REF_LNL      = -7541976.861
LNL_TOL      = 0.05        # FP non-associativity band from Allreduce reordering
EXPECTED_NP  = 4
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

# 5. Speedup (informational; not a hard gate)
if wall_base > 0 and wall_modep > 0:
    speedup = wall_base / wall_modep
    lines.append(f"  {'✓' if speedup >= 1.5 else '⚠'} Speedup: {wall_base}s → {wall_modep}s = {speedup:.2f}×"
                 f"  (ISO-3 was 1.74× at np=2)")

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ ISO-4 result (P.3+P.4+P.5a, np=4, correctness gate) ═════════"
echo "  binary md5: ${MD5}  (expected: ${EXPECTED_MD5})"
echo "  BASE  exit=${BASE_RC}  wall=${WALL_BASE}s  model=${BASE_BEST}  lnL=${BASE_OPT}  MF=${BASE_MF_WALL}s"
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s model=${MODEP_BEST} lnL=${MODEP_OPT} MF=${MODEP_MF_WALL}s"
echo "  [Mode P] total=${MODE_P_TOTAL} distinct_ranks=${MODE_P_RANKS}"
echo "  ref lnL: -7541976.861 (FCA np=1 job 169095077)  tolerance: 0.05"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${BASE_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}";  PASS=0; }
[[ "${MODEP_RC}" -eq 0 ]] || { echo "  ✗ FAIL: MODEP rc=${MODEP_RC}"; PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-4 PASS — np=4 Mode P correct; proceed to ISO-5 (np=16, AA 1M) ══"
    exit 0
else
    echo "  ══ ISO-4 FAIL — investigate above ══" >&2
    echo "  See research/Modelfinder/mode-p-implementation-status.md §ISO-4 for context" >&2
    exit 10
fi
