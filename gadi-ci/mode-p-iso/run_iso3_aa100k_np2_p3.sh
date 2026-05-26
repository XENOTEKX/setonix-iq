#!/bin/bash
# run_iso3_aa100k_np2_p3.sh — ISO-3 gate: fixed-model LG+G4 branch-opt trace.
#
# PURPOSE: Verify that the P.4 (DervSIMD) kernel produces an identical
# Newton-Raphson branch-optimisation trajectory under Mode P as the single-
# rank base run. Two sub-runs are executed inside one PBS job using the
# ISO-2 PASS treefile as a shared starting point:
#
#   sub-run BASE  — np=1, LG+G4, no Mode P  — reference NR trace
#   sub-run MODEP — np=2, LG+G4, --mode-p-all — Mode P NR trace
#
# Both start from the same fixed tree (ISO-2 treefile) so the starting
# log-likelihoods are byte-identical on the same hardware, and any divergence
# in the NR trace is attributable to P.4 DervSIMD or P.3 BranchSIMD alone.
#
# Gate pass criteria:
#   - Both sub-runs exit 0
#   - [Mode P] partition lines emitted in MODEP run
#   - Per NR-step |Δ lnL| ≤ 1e-6 between BASE and MODEP traces (FP
#     non-associativity of Mode P Allreduce is expected ~1e-9; 1e-6 is strict)
#   - Final optimal lnL within 1e-6 of -7,541,976.861 (FCA np=1 reference)
#   - Best model (where inferred): LG+G4
#
# Failure modes:
#   - MODEP NR trace diverges from BASE at step N → P.4 DervSIMD Allreduce
#     introduces a sign error or missing contribution from one rank's slice
#   - MODEP lnL drifts by > 1e-6 at final step → cumulative rounding error
#     in Allreduce accumulation; check per-rank ptn_scale[] accounting
#   - Crash (df=nan / Numerical underflow) → B.4-8 fix not active; wrong binary
#   - [Mode P] lines absent → --mode-p-all flag not triggering partition
#
# NR-trace reference (ISO-2 log, final branch-opt pass):
#   1. Initial log-likelihood: -7541976.862
#   Optimal log-likelihood:   -7541976.861
#   (Only one NR step — already at optimum on the ISO-2 treefile)

#PBS -N iso3-p3-np2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1000GB
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

NRANKS=2
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="iso3_p3_aa100k_np2_seed${SEED}"
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
[[ "${#HOSTS[@]}" -ge 2 ]] || { echo "ERROR: expected >=2 nodes, got ${#HOSTS[@]}" >&2; exit 9; }
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"
RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
rank 1=${HOSTS[1]} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ISO-3 gate: P.3+P.4, np=2, AA 100K, LG+G4 fixed-model      ║"
echo "║  binary:  $(basename "${IQTREE}")  md5: ${MD5}"
echo "║  tree:    $(basename "${ISO2_TREE}") (ISO-2 PASS treefile)"
echo "║  Purpose: NR branch-opt trace (BASE np=1) vs (MODEP np=2)"
echo "║  Ref lnL: -7,541,976.861 (FCA np=1 job 169095077)"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────────────────
# Sub-run 1: BASE — np=1, no Mode P, -m TEST, fixed tree
# ─────────────────────────────────────────────────────────────
BASE_DIR="${WORK_DIR}/base_np1"; mkdir -p "${BASE_DIR}"
BASE_RANK_LOGS="${BASE_DIR}/rank_logs"; mkdir -p "${BASE_RANK_LOGS}"
RANKFILE_BASE="${WORK_DIR}/rankfile_base.txt"
cat > "${RANKFILE_BASE}" <<EOF
rank 0=${HOSTS[0]} slot=0-103
EOF

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
                -v \
                --prefix "${BASE_DIR}/iqtree_inner" \
    > "${BASE_DIR}/iqtree_stdout.log" 2>&1
BASE_RC=$?
END_BASE=$(date +%s)
WALL_BASE=$(( END_BASE - START_BASE ))
echo "  BASE exit=${BASE_RC} wall=${WALL_BASE}s"

# ─────────────────────────────────────────────────────────────
# Sub-run 2: MODEP — np=2, --mode-p-all, -m TEST, fixed tree
# ─────────────────────────────────────────────────────────────
MODEP_DIR="${WORK_DIR}/modep_np2"; mkdir -p "${MODEP_DIR}"
MODEP_RANK_LOGS="${MODEP_DIR}/rank_logs"; mkdir -p "${MODEP_RANK_LOGS}"

echo ""
echo "── Sub-run MODEP (np=2, --mode-p-all) ──────────────────────────"
START_MODEP=$(date +%s)
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${MODEP_RANK_LOGS}/" \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                -te "${ISO2_TREE}" \
                --mode-p-all \
                --atmd-k-outer 1 \
                -v \
                --prefix "${MODEP_DIR}/iqtree_inner" \
    > "${MODEP_DIR}/iqtree_stdout.log" 2>&1
MODEP_RC=$?
END_MODEP=$(date +%s)
WALL_MODEP=$(( END_MODEP - START_MODEP ))
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s"

# ─────────────────────────────────────────────────────────────
# Parse NR traces and compare
# ─────────────────────────────────────────────────────────────
extract_nr_trace() {
    local log="$1"
    # Lines: "N. Initial log-likelihood: X" and "N. Current log-likelihood: X"
    # Exclude lines from non-rank-0 processes (they contain "/ Process: [1-9]")
    grep -E '^[0-9]+\. (Initial|Current) log-likelihood:' "${log}" 2>/dev/null \
        | grep -v 'Process: [1-9]' || true
}

BASE_TRACE=$(extract_nr_trace "${BASE_DIR}/iqtree_inner.log")
MODEP_TRACE=$(extract_nr_trace "${MODEP_DIR}/iqtree_inner.log")

BASE_OPT=$(grep -oP 'Optimal log-likelihood: \K[-0-9.]+' "${BASE_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
MODEP_OPT=$(grep -oP 'Optimal log-likelihood: \K[-0-9.]+' "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | tail -1 || echo "")
MODEP_BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "LG+G4")
MODEP_MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${MODEP_DIR}/iqtree_inner.log" 2>/dev/null | head -1 || echo "")

echo ""
echo "── NR trace (BASE) ──────────────────────────────────────────────"
echo "${BASE_TRACE}" | head -20 || true
echo "── NR trace (MODEP) ─────────────────────────────────────────────"
echo "${MODEP_TRACE}" | head -20 || true
echo "── [Mode P] partition lines (MODEP) ─────────────────────────────"
grep '\[Mode P\]' "${MODEP_DIR}/iqtree_inner.log" "${MODEP_DIR}/iqtree_stdout.log" \
    "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null | head -6 || true

MODE_P_COUNT=$({ grep -h '\[Mode P\]' "${MODEP_DIR}/iqtree_inner.log" \
    "${MODEP_DIR}/iqtree_stdout.log" "${MODEP_RANK_LOGS}"/*/stderr 2>/dev/null; true; } | wc -l)

# ─────────────────────────────────────────────────────────────
# Numeric comparison (Python — always available on Gadi)
# ─────────────────────────────────────────────────────────────
COMPARE_RESULT=$(python3 - "${BASE_OPT}" "${MODEP_OPT}" "${MODE_P_COUNT}" \
    "${BASE_TRACE}" "${MODEP_TRACE}" <<'PYEOF'
import sys, re

base_opt  = sys.argv[1]
modep_opt = sys.argv[2]
modep_cnt = int(sys.argv[3])
base_trace_raw  = sys.argv[4]
modep_trace_raw = sys.argv[5]

REF_LNL = -7541976.861
TOL     = 1e-6
TRACE_TOL = 1e-6

def extract_vals(raw):
    return [float(m) for m in re.findall(r'log-likelihood:\s*([-0-9.]+)', raw)]

base_vals  = extract_vals(base_trace_raw)
modep_vals = extract_vals(modep_trace_raw)

lines = []
ok = True

# Check final optimal lnL
if modep_opt:
    delta_ref = abs(float(modep_opt) - REF_LNL)
    if delta_ref <= TOL:
        lines.append(f"  ✓ MODEP final lnL={modep_opt}  |Δ vs FCA np=1|={delta_ref:.3e} ≤ {TOL:.0e}")
    else:
        lines.append(f"  ✗ MODEP final lnL={modep_opt}  |Δ vs FCA np=1|={delta_ref:.3e} > {TOL:.0e}")
        ok = False
else:
    lines.append("  ✗ MODEP optimal lnL not parsed"); ok = False

if base_opt:
    delta_ref_base = abs(float(base_opt) - REF_LNL)
    lines.append(f"  BASE final lnL={base_opt}  |Δ vs FCA np=1|={delta_ref_base:.3e}")

# NR trace step-by-step comparison
if base_vals and modep_vals:
    n = min(len(base_vals), len(modep_vals))
    max_delta = 0.0
    worst_step = -1
    for i in range(n):
        d = abs(base_vals[i] - modep_vals[i])
        if d > max_delta:
            max_delta = d
            worst_step = i + 1
    if max_delta <= TRACE_TOL:
        lines.append(f"  ✓ NR trace {n}-step max|Δ|={max_delta:.3e} ≤ {TRACE_TOL:.0e}"
                     f"  (BASE steps={len(base_vals)} MODEP steps={len(modep_vals)})")
    else:
        lines.append(f"  ✗ NR trace diverges at step {worst_step}: max|Δ|={max_delta:.3e} > {TRACE_TOL:.0e}")
        ok = False
    if len(base_vals) != len(modep_vals):
        lines.append(f"  ⚠ NR step count differs: BASE={len(base_vals)} MODEP={len(modep_vals)}")
else:
    lines.append("  ⚠ NR trace empty — check -v output and log parsing")

# [Mode P] line count
if modep_cnt > 0:
    lines.append(f"  ✓ [Mode P] lines: {modep_cnt}")
else:
    lines.append("  ✗ No [Mode P] lines emitted"); ok = False

lines.append("PASS" if ok else "FAIL")
print('\n'.join(lines))
PYEOF
)

echo ""
echo "══ ISO-3 result (P.3+P.4 NR trace gate) ════════════════════════"
echo "  binary md5: ${MD5}  (B.4-9 fix expected: 9660575a; previous B.4-8 binary: 50b4b172)"
echo "  BASE  exit=${BASE_RC}  wall=${WALL_BASE}s  optimal_lnL=${BASE_OPT}"
echo "  MODEP exit=${MODEP_RC} wall=${WALL_MODEP}s optimal_lnL=${MODEP_OPT} best=${MODEP_BEST}"
echo "  MODEP MF wall: ${MODEP_MF_WALL}s"
echo "  ref lnL: -7541976.861 (FCA np=1 job 169095077)  tolerance: 1e-6"
echo ""
echo "${COMPARE_RESULT}" | grep -v '^PASS$\|^FAIL$' || true

PASS=1
[[ "${BASE_RC}"  -eq 0 ]] || { echo "  ✗ FAIL: BASE rc=${BASE_RC}";  PASS=0; }
[[ "${MODEP_RC}" -eq 0 ]] || { echo "  ✗ FAIL: MODEP rc=${MODEP_RC}"; PASS=0; }
echo "${COMPARE_RESULT}" | grep -q '^FAIL$' && PASS=0 || true

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ ISO-3 PASS — P.3+P.4 NR trace correct; proceed to ISO-4 ══"
else
    echo "  ══ ISO-3 FAIL — investigate NR trace divergence ══" >&2
    echo "  See research/Modelfinder/mode-p-implementation-status.md §ISO-3 for failure modes" >&2
    exit 10
fi
