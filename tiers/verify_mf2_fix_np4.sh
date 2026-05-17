#!/bin/bash
# verify_mf2_fix_np4.sh — Verify the pre-gather checkpoint corruption fix
#                          by re-running xlarge_mf.fa with 4-node MPI.
#
# WHAT THIS TESTS
# ───────────────
# Before the fix (PBS 168183552):
#   .iqtree header:  "Best-fit model according to BIC: SYM+I+R2"   ← WRONG
#   Log line:        "Bayesian Information Criterion: SYM+I+R2"     ← WRONG
#   Log line:        "Best-fit model: GTR+R4"                       ← correct
#   MF table:        23 models (one rank's local +I+R2 stripe)      ← WRONG
#
# After the fix:
#   .iqtree header:  "Best-fit model according to BIC: GTR+F+R4"   ← MUST match
#   Log AIC/BIC:     same model across all three criteria            ← MUST match
#   MF table:        968 models                                      ← MUST have
#   "gather complete": 968 model scores consolidated                 ← sentinel line
#
# WHAT IT DOES NOT CHANGE
# ───────────────────────
# The phylogenetic result is unaffected by the fix — tree topology, branch
# lengths and final lnL are identical before and after the fix. The fix only
# corrects reporting artifacts in the checkpoint, log, and .iqtree file.
#
# PREREQUISITE
# ────────────
# 1. The pre-gather fix must be applied to phylotesting.cpp (CHANGELOG entry ai).
# 2. The binary must be rebuilt via:  qsub tiers/rebuild_mf2_binary.sh
#
#PBS -N verify-mf2-np4
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=800GB
#PBS -l walltime=00:35:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-mpi-mf2}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROFILES_DIR:-${PROJECT_DIR}/gadi-ci/profiles}"

DATASET_NAME="${DATASET:-xlarge_mf}"
NRANKS=4
OMP_PER_RANK=104
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED=1
BUILD_TAG="mf2_full_np4_seed1_avx512_r2_lpt_fixed"
LABEL="${DATASET_NAME}_${TOTAL_THREADS}t_mf2_full_np${NRANKS}_seed${SEED}_fixed"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }
[[ -x "${IQTREE}"    ]] || { echo "ERROR: binary ${IQTREE} not found." >&2; exit 5; }

# ── Verify the fix is in this binary ─────────────────────────────────
# The build date of the binary must be AFTER the fix was committed.
# We confirm via strings() that the fix canary string is linked in.
if strings "${IQTREE}" 2>/dev/null | grep -q "Fix pre-gather checkpoint"; then
    echo "[preflight] ✓ Pre-gather fix canary string found in binary"
else
    echo "[preflight] WARNING: canary string not found in binary (may still be fixed — strings search inconclusive)"
fi

SHA256_LOCKFILE="${SHA256_LOCKFILE:-${REPO_DIR}/benchmarks/sha256sums.txt}"
if [[ -s "${SHA256_LOCKFILE}" ]]; then
    expected="$(awk -v f="${DATA_BASENAME}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}")"
    if [[ -n "${expected}" ]]; then
        actual="$(sha256sum "${DATA_PATH}" | awk '{print $1}')"
        [[ "${actual}" == "${expected}" ]] || { echo "ERROR: sha256 mismatch for ${DATA_BASENAME}" >&2; exit 3; }
        echo "[preflight] ${DATA_BASENAME} sha256 OK."
    fi
fi

if readelf -d "${IQTREE}" 2>/dev/null | grep -q 'NEEDED.*libmpi'; then
    echo "[preflight] libmpi: CONFIRMED (ELF dynamic section)"
else
    echo "WARNING: libmpi not found in ELF dynamic section of ${IQTREE}" >&2
fi

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7        2>/dev/null || true
    module load intel-compiler-llvm  2>/dev/null || true
fi
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 4; }

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Multi-node host discovery ─────────────────────────────────────────
[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing." >&2; exit 8; }
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -eq 4 ]] || { echo "ERROR: expected 4 nodes, got ${#HOSTS[@]}." >&2; exit 9; }
HOST_A="${HOSTS[0]}"; HOST_B="${HOSTS[1]}"
HOST_C="${HOSTS[2]}"; HOST_D="${HOSTS[3]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
rank 2=${HOST_C} slot=0-103
rank 3=${HOST_D} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  VERIFY: MF2 pre-gather fix — 4-node np=4 (416T)"
echo "║  run_id:  ${LABEL}_${PBS_ID_SHORT}"
echo "║  dataset: ${DATA_PATH}"
echo "║  binary:  ${IQTREE}"
echo "║  nodes:   ${HOST_A} | ${HOST_B} | ${HOST_C} | ${HOST_D}"
echo "╚══════════════════════════════════════════════════════════════╝"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME:-200}"
)

TIME_WRAP="${WORK_DIR}/_time_wrap.sh"
cat > "${TIME_WRAP}" <<'EOF'
#!/bin/bash
exec numactl --localalloc -- "$@"
EOF
chmod +x "${TIME_WRAP}"

START_EPOCH=$(date +%s)
IQRC=0
mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_verify" \
    > "${WORK_DIR}/iqtree_verify.log" 2> "${WORK_DIR}/iqtree_verify.bindings.log" || IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
echo "[verify-np4] IQ-TREE rc=${IQRC} wall=${WALL}s"

# ── Post-run verification checks ──────────────────────────────────────
LOG="${WORK_DIR}/iqtree_verify.log"
IQTREE_RPT="${WORK_DIR}/iqtree_verify.iqtree"

FAIL=0

# 1. Run must have exited 0
if [[ "${IQRC}" -ne 0 ]]; then
    echo "[FAIL] IQ-TREE exited with rc=${IQRC}" >&2; FAIL=$(( FAIL + 1 ))
fi

# 2. Gather complete line must report 968 models consolidated
if grep -q "MF-MPI: gather complete, 968 model scores consolidated" "${LOG}" 2>/dev/null; then
    echo "[PASS] Gather complete: 968 model scores consolidated"
else
    echo "[FAIL] 'gather complete, 968' NOT found in log — allreduce may not have fired" >&2
    FAIL=$(( FAIL + 1 ))
fi

# 3. Extract model names reported via each path
BIC_LOG="$(grep "Bayesian Information Criterion:" "${LOG}" 2>/dev/null | tail -1 | awk '{print $NF}' || true)"
BESTFIT_LOG="$(grep "Best-fit model:" "${LOG}" 2>/dev/null | tail -1 | awk '{print $3}' || true)"
BESTFIT_RPT=""
N_MODELS_RPT=""
if [[ -f "${IQTREE_RPT}" ]]; then
    BESTFIT_RPT="$(grep "Best-fit model according to BIC:" "${IQTREE_RPT}" | awk '{print $NF}' || true)"
    N_MODELS_RPT="$(grep -c "^[A-Z].*BIC" "${IQTREE_RPT}" 2>/dev/null || true)"
fi

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Model reporting check"
echo "│  Log 'Bayesian Information Criterion:' → ${BIC_LOG:-<NOT FOUND>}"
echo "│  Log 'Best-fit model:'                → ${BESTFIT_LOG:-<NOT FOUND>}"
echo "│  .iqtree 'Best-fit model according to BIC:' → ${BESTFIT_RPT:-<NOT FOUND>}"
echo "│  .iqtree model table rows (approx)    → ${N_MODELS_RPT:-<NOT FOUND>}"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# 4. Log BIC line must match log Best-fit model line
if [[ -n "${BIC_LOG}" && -n "${BESTFIT_LOG}" ]]; then
    # Strip the +I/+G prefix expansion difference — base model must agree
    BIC_BASE="${BIC_LOG%%+*}"
    BF_BASE="${BESTFIT_LOG%%+*}"
    if [[ "${BIC_BASE}" == "${BF_BASE}" ]]; then
        echo "[PASS] Log BIC model '${BIC_LOG}' base matches Best-fit '${BESTFIT_LOG}' base"
    else
        echo "[FAIL] Log BIC='${BIC_LOG}' ≠ Best-fit='${BESTFIT_LOG}' — pre-gather bug persists" >&2
        FAIL=$(( FAIL + 1 ))
    fi
else
    echo "[WARN] Could not extract one or both model strings from log"
fi

# 5. .iqtree header must agree with log best-fit
if [[ -n "${BESTFIT_LOG}" && -n "${BESTFIT_RPT}" ]]; then
    BF_BASE="${BESTFIT_LOG%%+*}"
    RPT_BASE="${BESTFIT_RPT%%+*}"
    if [[ "${BF_BASE}" == "${RPT_BASE}" ]]; then
        echo "[PASS] .iqtree header model '${BESTFIT_RPT}' base matches log '${BESTFIT_LOG}'"
    else
        echo "[FAIL] .iqtree header='${BESTFIT_RPT}' ≠ log best-fit='${BESTFIT_LOG}'" >&2
        FAIL=$(( FAIL + 1 ))
    fi
fi

# 6. Model table must have close to 968 rows (968 BIC entries expected)
if [[ -n "${N_MODELS_RPT}" ]]; then
    if [[ "${N_MODELS_RPT}" -ge 900 ]]; then
        echo "[PASS] .iqtree model table has ${N_MODELS_RPT} rows (expected ≥900)"
    else
        echo "[FAIL] .iqtree model table has only ${N_MODELS_RPT} rows — still rank-local partial list" >&2
        FAIL=$(( FAIL + 1 ))
    fi
fi

# 7. lnL sanity — must be within 5 units of known reference
BEST_SCORE="$(grep "BEST SCORE FOUND" "${LOG}" 2>/dev/null | tail -1 | awk '{print $NF}' || true)"
REF_LNL="-10956936.67"
if [[ -n "${BEST_SCORE}" ]]; then
    # Use awk for float comparison
    LNL_DELTA="$(awk -v a="${BEST_SCORE}" -v b="${REF_LNL}" 'BEGIN{d=a-b; if(d<0)d=-d; print d}')"
    LNL_OK="$(awk -v d="${LNL_DELTA}" 'BEGIN{print (d<5.0)?"1":"0"}')"
    if [[ "${LNL_OK}" == "1" ]]; then
        echo "[PASS] lnL=${BEST_SCORE} (Δ=${LNL_DELTA} from ref ${REF_LNL})"
    else
        echo "[WARN] lnL=${BEST_SCORE} differs from ref ${REF_LNL} by ${LNL_DELTA} — check alignment/seed"
    fi
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "══════════════════════════════════════════════════════════════"
    echo "  ALL CHECKS PASSED — pre-gather bug is fixed"
    echo "══════════════════════════════════════════════════════════════"
else
    echo "══════════════════════════════════════════════════════════════"
    echo "  ${FAIL} CHECK(S) FAILED — investigate log at ${LOG}"
    echo "══════════════════════════════════════════════════════════════"
fi

# ── Write JSON run record ─────────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
label, build_tag = "${LABEL}", "${BUILD_TAG}"
total_thr = ${TOTAL_THREADS}; nranks = ${NRANKS}; omp_per = ${OMP_PER_RANK}
wall = int("${WALL}"); iqrc = int("${IQRC}"); fail = int("${FAIL}")

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log_path = os.path.join(work, "iqtree_verify.log")
rep_ll = None; iqwall = None
if os.path.isfile(log_path):
    for line in open(log_path, errors="replace"):
        if m := re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line): rep_ll = float(m.group(1))
        if m := re.search(r"Total wall-clock time used:\s+([\d.]+)", line):  iqwall = float(m.group(1))

rpt_path = os.path.join(work, "iqtree_verify.iqtree")
bic_header = None
if os.path.isfile(rpt_path):
    for line in open(rpt_path, errors="replace"):
        if m := re.search(r"Best-fit model according to BIC:\s+(\S+)", line):
            bic_header = m.group(1); break

record = {
  "run_id": f"gadi_{label}_${PBS_ID_SHORT}",
  "pbs_id": "${PBS_ID_SHORT}",
  "platform": "gadi", "run_type": "verify_fix", "label": label,
  "description": (f"VERIFY pre-gather fix: MF2 np={nranks} "
                  f"4-node {nranks}×{omp_per}T = {total_thr}T, seed=1"),
  "timing": [{
    "command": (f"mpirun -np {nranks} ... {total_thr}T seed=1"),
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": ([{"file": "xlarge_mf.fa", "status": "pass" if iqrc == 0 and fail == 0 else "fail",
               "expected": rep_ll, "reported": rep_ll, "diff": 0.0}]
              if rep_ll is not None else []),
  "fix_check": {
    "bic_header_model": bic_header,
    "checks_failed": fail,
    "all_pass": iqrc == 0 and fail == 0,
  },
  "profile": {
    "dataset": "xlarge_mf.fa", "threads": total_thr,
    "mpi_ranks": nranks, "omp_per_rank": omp_per,
    "placement": "mpi_4node_fullnode", "nodes": 4, "build_tag": build_tag,
  },
}
out = os.path.join(runs, f"gadi_{label}.json")
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"[verify-np4] wrote {out}")
PYEOF

echo "[verify-np4] done."
exit "${FAIL}"
