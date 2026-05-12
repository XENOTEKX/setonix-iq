#!/bin/bash
# verify_mf2_fix_np2.sh — Verify the pre-gather checkpoint corruption fix
#                          with 2-node MPI (np=2, 208T).
#
# WHAT THIS TESTS
# ───────────────
# np2 shows the same pre-gather corruption as np4 but less dramatically
# because with 2 ranks the two BIC-local-best models are closer (both
# from GPT starting trees). Before the fix (PBS 168183551):
#   .iqtree header:  "GTR+I+R4"     (rank 1's stale local best)
#   Best-fit model:  "GTR+R4"       (correct post-gather)
#   MF table:        49 rows         (one rank's stripe, not 968)
#
# After the fix, both must agree and table must have ~968 rows.
#
# PREREQUISITE: rebuild_mf2_binary.sh must have completed.
#
#PBS -N verify-mf2-np2
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=400GB
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
NRANKS=2
OMP_PER_RANK=104
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED=1
BUILD_TAG="mf2_full_np2_seed1_avx512_r2_lpt_fixed"
LABEL="${DATASET_NAME}_${TOTAL_THREADS}t_mf2_full_np${NRANKS}_seed${SEED}_fixed"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }
[[ -x "${IQTREE}"    ]] || { echo "ERROR: binary ${IQTREE} not found." >&2; exit 5; }

SHA256_LOCKFILE="${SHA256_LOCKFILE:-${REPO_DIR}/benchmarks/sha256sums.txt}"
if [[ -s "${SHA256_LOCKFILE}" ]]; then
    expected="$(awk -v f="${DATA_BASENAME}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}")"
    if [[ -n "${expected}" ]]; then
        actual="$(sha256sum "${DATA_PATH}" | awk '{print $1}')"
        [[ "${actual}" == "${expected}" ]] || { echo "ERROR: sha256 mismatch." >&2; exit 3; }
        echo "[preflight] ${DATA_BASENAME} sha256 OK."
    fi
fi

readelf -d "${IQTREE}" 2>/dev/null | grep -q 'NEEDED.*libmpi' || \
    echo "WARNING: libmpi not found in ELF dynamic section" >&2

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

[[ -s "${PBS_NODEFILE:-/dev/null}" ]] || { echo "ERROR: PBS_NODEFILE missing." >&2; exit 8; }
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
[[ "${#HOSTS[@]}" -eq 2 ]] || { echo "ERROR: expected 2 nodes, got ${#HOSTS[@]}." >&2; exit 9; }
HOST_A="${HOSTS[0]}"; HOST_B="${HOSTS[1]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  VERIFY: MF2 pre-gather fix — 2-node np=2 (208T)"
echo "║  nodes: ${HOST_A} | ${HOST_B}"
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
echo "[verify-np2] IQ-TREE rc=${IQRC} wall=${WALL}s"

# ── Post-run verification checks ──────────────────────────────────────
LOG="${WORK_DIR}/iqtree_verify.log"
IQTREE_RPT="${WORK_DIR}/iqtree_verify.iqtree"
FAIL=0

[[ "${IQRC}" -eq 0 ]] || { echo "[FAIL] IQ-TREE rc=${IQRC}" >&2; FAIL=$(( FAIL + 1 )); }

if grep -q "MF-MPI: gather complete, 968 model scores consolidated" "${LOG}" 2>/dev/null; then
    echo "[PASS] gather complete: 968 model scores consolidated"
else
    echo "[FAIL] gather complete line missing or wrong count" >&2; FAIL=$(( FAIL + 1 ))
fi

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
echo "│  Model reporting check (np=2)"
echo "│  Log BIC:           ${BIC_LOG:-<NOT FOUND>}"
echo "│  Log Best-fit:      ${BESTFIT_LOG:-<NOT FOUND>}"
echo "│  .iqtree BIC header:${BESTFIT_RPT:-<NOT FOUND>}"
echo "│  .iqtree model rows:${N_MODELS_RPT:-<NOT FOUND>}"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

if [[ -n "${BIC_LOG}" && -n "${BESTFIT_LOG}" ]]; then
    BIC_BASE="${BIC_LOG%%+*}"; BF_BASE="${BESTFIT_LOG%%+*}"
    if [[ "${BIC_BASE}" == "${BF_BASE}" ]]; then
        echo "[PASS] Log BIC '${BIC_LOG}' base matches Best-fit '${BESTFIT_LOG}'"
    else
        echo "[FAIL] Log BIC='${BIC_LOG}' ≠ Best-fit='${BESTFIT_LOG}'" >&2; FAIL=$(( FAIL + 1 ))
    fi
fi

if [[ -n "${BESTFIT_LOG}" && -n "${BESTFIT_RPT}" ]]; then
    BF_BASE="${BESTFIT_LOG%%+*}"; RPT_BASE="${BESTFIT_RPT%%+*}"
    if [[ "${BF_BASE}" == "${RPT_BASE}" ]]; then
        echo "[PASS] .iqtree header '${BESTFIT_RPT}' base matches log '${BESTFIT_LOG}'"
    else
        echo "[FAIL] .iqtree header='${BESTFIT_RPT}' ≠ log='${BESTFIT_LOG}'" >&2; FAIL=$(( FAIL + 1 ))
    fi
fi

if [[ -n "${N_MODELS_RPT}" ]]; then
    if [[ "${N_MODELS_RPT}" -ge 900 ]]; then
        echo "[PASS] .iqtree model table has ${N_MODELS_RPT} rows (expected ≥900)"
    else
        echo "[FAIL] .iqtree model table has only ${N_MODELS_RPT} rows" >&2; FAIL=$(( FAIL + 1 ))
    fi
fi

BEST_SCORE="$(grep "BEST SCORE FOUND" "${LOG}" 2>/dev/null | tail -1 | awk '{print $NF}' || true)"
REF_LNL="-10956936.67"
if [[ -n "${BEST_SCORE}" ]]; then
    LNL_DELTA="$(awk -v a="${BEST_SCORE}" -v b="${REF_LNL}" 'BEGIN{d=a-b; if(d<0)d=-d; print d}')"
    LNL_OK="$(awk -v d="${LNL_DELTA}" 'BEGIN{print (d<5.0)?"1":"0"}')"
    [[ "${LNL_OK}" == "1" ]] && echo "[PASS] lnL=${BEST_SCORE} (Δ=${LNL_DELTA})" || \
        echo "[WARN] lnL=${BEST_SCORE} Δ=${LNL_DELTA} from ref"
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "══════════════════════════════════════════════════════════════"
    echo "  ALL CHECKS PASSED (np=2) — pre-gather bug is fixed"
    echo "══════════════════════════════════════════════════════════════"
else
    echo "══════════════════════════════════════════════════════════════"
    echo "  ${FAIL} CHECK(S) FAILED — see ${LOG}"
    echo "══════════════════════════════════════════════════════════════"
fi

# ── Write JSON ────────────────────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
label, build_tag = "${LABEL}", "${BUILD_TAG}"
total_thr = ${TOTAL_THREADS}; nranks = ${NRANKS}; omp_per = ${OMP_PER_RANK}
wall = int("${WALL}"); iqrc = int("${IQRC}"); fail = int("${FAIL}")

log_path = os.path.join(work, "iqtree_verify.log")
rpt_path = os.path.join(work, "iqtree_verify.iqtree")
rep_ll = None; iqwall = None; bic_header = None
if os.path.isfile(log_path):
    for line in open(log_path, errors="replace"):
        if m := re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line): rep_ll = float(m.group(1))
        if m := re.search(r"Total wall-clock time used:\s+([\d.]+)", line): iqwall = float(m.group(1))
if os.path.isfile(rpt_path):
    for line in open(rpt_path, errors="replace"):
        if m := re.search(r"Best-fit model according to BIC:\s+(\S+)", line):
            bic_header = m.group(1); break

record = {
  "run_id": f"gadi_{label}_${PBS_ID_SHORT}",
  "pbs_id": "${PBS_ID_SHORT}", "platform": "gadi",
  "run_type": "verify_fix", "label": label,
  "timing": [{"time_s": iqwall if iqwall is not None else wall}],
  "fix_check": {"bic_header_model": bic_header, "checks_failed": fail, "all_pass": iqrc == 0 and fail == 0},
  "profile": {"dataset": "xlarge_mf.fa", "threads": total_thr, "mpi_ranks": nranks,
               "omp_per_rank": omp_per, "nodes": 2, "build_tag": build_tag},
}
out = os.path.join(runs, f"gadi_{label}.json")
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"[verify-np2] wrote {out}")
PYEOF

echo "[verify-np2] done."
exit "${FAIL}"
