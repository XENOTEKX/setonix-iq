#!/bin/bash
# run_ws_a2_aa_100k_1node_w2.sh — W2 correctness gate: warm-start A.2 binary, AA 100K, 1-node MPI, 4 ranks.
#
# AA, 100K sites (100 taxa), 4 MPI ranks × 26 OpenMP threads = 104T on 1 node (SNC4-aligned).
#
# PURPOSE — W2 (gate 2 of the Phase A.2 validation ladder):
#   Confirm that the WarmStartPacket MPI_Bcast (Phase A.2) does NOT alter
#   correctness at np=4 on the AA 100K benchmark, AND that the broadcast
#   actually fires and carries non-zero fields (ws_bcast_fields > 0 in
#   MF-MPI-DIAG output), demonstrating that rank 0's early-converged params
#   reached the other ranks.
#
# Pass criteria (doc §5.7 W2):
#   - lnL within ±0.5 of baseline ref (168425673): −7,541,976.860
#   - Best model = LG+G4
#   - MF wall ≤ 100 s  (FCA np=4 AA 100K without WS: ~149 s)
#   - ws_bcast_fields > 0 in at least one MF-MPI-DIAG line
#   - exit code = 0
#
# Binary:  /scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a2
#          md5 1547a906f1f75422514b0a0cdf2bc89e (built 2026-05-23 15:58, +101 lines A.2)
#          Source commit: 5604606d (fca-lbfgs-ws, Phase A.2 WarmStartPacket MPI_Bcast)
# Branch:  fca-lbfgs-ws (harness) + fca-lbfgs-ws (source, from test_MF2 @ 9603247f)
# Parity:  4×26T (1 rank per SNC4 quadrant), numactl --localalloc, KMP_BLOCKTIME=200, seed=1
# A/B ref: W1 gate (ws-a1, np=1) — MF wall 254 s; FCA np=4 without WS: ~149 s (design est.)
#
# Diagnostic to check: MF-MPI-DIAG line includes ws_bcast_fields=N where N > 0
# Build tag: fca_ws_a2_icx_avx512
# Related:   CHANGELOG (cc), research/lbfgs-and-warmstart-implementation.md §6.3, §5.7 W2

#PBS -N ws-a2-aa-100k-w2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=200GB
#PBS -l place=excl
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61+scratch/um09
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-a2}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=4
OMP_PER_RANK="${OMP_PER_RANK:-26}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_100k"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_100k_ws_a2_np4_w2_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7             2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2  2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: A.2 binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found after module load." >&2; exit 4; }
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 5
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5." >&2; exit 6
fi

# Verify A.2 symbols are present (filterRatesMPI inherited from FCA).
if nm "${IQTREE}" 2>/dev/null | grep -q '_ZN17CandidateModelSet14filterRatesMPIEi'; then
    echo "[preflight] filterRatesMPI (Phase A.2): confirmed via nm"
elif strings "${IQTREE}" 2>/dev/null | grep -q 'filterRatesMPI'; then
    echo "[preflight] filterRatesMPI: found via strings (nm failed)"
else
    echo "[preflight] WARNING: filterRatesMPI symbol not found" >&2
fi

if nm "${IQTREE}" 2>/dev/null | grep -q '_ZN18RateWarmStartCache5clearEv'; then
    echo "[preflight] RateWarmStartCache (Phase A.1/A.2): confirmed via nm"
fi

MD5_ACTUAL="$(md5sum "${IQTREE}" | awk '{print $1}')"
echo "[preflight] md5: ${MD5_ACTUAL} (expected: 1547a906f1f75422514b0a0cdf2bc89e)"
if [[ "${MD5_ACTUAL}" != "1547a906f1f75422514b0a0cdf2bc89e" ]]; then
    echo "[preflight] WARNING: md5 mismatch — binary may have been rebuilt since W2 setup." >&2
fi

# ── OMP / runtime ─────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AA 100K WS-A.2 W2 Correctness Gate — 1-node np=4 TESTONLY"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5 expected: 1547a906f1f75422514b0a0cdf2bc89e"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  branch:       fca-lbfgs-ws (source commit 5604606d)"
echo "║  W1 ref:       169094526 (MF 254 s, lnL −7,541,976.862, LG+G4)"
echo "║  W2 MF target: ≤ 100 s (FCA np=4 without WS: ~149 s)"
echo "║  ws_bcast check: ws_bcast_fields > 0 in MF-MPI-DIAG line"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Probe (hardware/software/binary/source) ───────────────────────────
RANK_PROBE="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_probe.sh"
if [[ ! -x "${RANK_PROBE}" ]]; then
    echo "[probe] WARNING: rank_probe.sh not found at ${RANK_PROBE} — running without wrapper" >&2
    RANK_PROBE=""
fi

# ── ModelFinder-only run (W2 correctness gate) ────────────────────────
echo "[w2] ModelFinder-only run (TESTONLY), ${NRANKS} ranks × ${OMP_PER_RANK} OMP"
START_EPOCH=$(date +%s)

if [[ -n "${RANK_PROBE}" ]]; then
    mpirun -np "${NRANKS}" \
        --bind-to none \
        --report-bindings \
        "${OMP_ENV[@]}" \
        "${RANK_PROBE}" \
            numactl --localalloc -- \
                "${IQTREE}" -s "${ALIGNMENT}" -m TESTONLY -T "${OMP_PER_RANK}" -seed "${SEED}" \
                            --prefix "${WORK_DIR}/iqtree_run" \
        > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
else
    mpirun -np "${NRANKS}" \
        --bind-to none \
        --report-bindings \
        "${OMP_ENV[@]}" \
        numactl --localalloc -- \
            "${IQTREE}" -s "${ALIGNMENT}" -m TESTONLY -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_run" \
        > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
fi
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[w2] done: rc=${IQRC} wall=${WALL}s"

# ── Extract diagnostic lines ──────────────────────────────────────────
grep -E '^MF-TIME: '      "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_time.log"     || true
grep -E '^MF-MPI-DIAG: '  "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_diag.log"     || true
grep -E '^WS-HIT: '       "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/ws_hit.log"      || true
grep -E '^WS-MISS: '      "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/ws_miss.log"     || true

echo "[w2] MF-TIME lines:     $(wc -l < "${WORK_DIR}/mf_time.log"  2>/dev/null || echo 0)"
echo "[w2] MF-MPI-DIAG lines: $(wc -l < "${WORK_DIR}/mf_diag.log"  2>/dev/null || echo 0)"
echo "[w2] WS-HIT lines:      $(wc -l < "${WORK_DIR}/ws_hit.log"   2>/dev/null || echo 0)"
echo "[w2] WS-MISS lines:     $(wc -l < "${WORK_DIR}/ws_miss.log"  2>/dev/null || echo 0)"

# Check ws_bcast_fields in MF-MPI-DIAG output.
WS_BCAST_FIELDS_MAX=0
if [[ -s "${WORK_DIR}/mf_diag.log" ]]; then
    while IFS= read -r diag_line; do
        fields_val=$(echo "${diag_line}" | grep -oP 'ws_bcast_fields=\K[0-9]+' || echo "0")
        if (( fields_val > WS_BCAST_FIELDS_MAX )); then
            WS_BCAST_FIELDS_MAX="${fields_val}"
        fi
    done < "${WORK_DIR}/mf_diag.log"
fi
echo "[w2] ws_bcast_fields_max: ${WS_BCAST_FIELDS_MAX} (expected > 0 if broadcast fired)"

# ── Run record ────────────────────────────────────────────────────────
/usr/bin/python3 - <<'PYEOF' 2>/dev/null || python3 - <<'PYEOF'
import json, os, re, subprocess, sys
work  = os.environ.get("WORK_DIR",  "")
runs  = os.environ.get("RUNS_DIR",  "")
rid   = os.environ.get("RUN_ID",    "")
label = os.environ.get("LABEL",     "")
nranks      = int(os.environ.get("NRANKS",       "4"))
omp_per_rank= int(os.environ.get("OMP_PER_RANK", "26"))
threads     = nranks * omp_per_rank
wall        = int(os.environ.get("WALL",   "0"))
iqrc        = int(os.environ.get("IQRC",   "1"))
alignment   = os.environ.get("ALIGNMENT", "")
ibin        = os.environ.get("IQTREE",    "")

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None; mf_wall = None; best_model = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line)
        if m: mf_wall = float(m.group(1))
        m = re.search(r"Best-fit model:\s+(\S+)", line)
        if m: best_model = m.group(1)

# Phase A.2 broadcast fields from MF-MPI-DIAG.
ws_bcast_fields_max = 0
diag_log = os.path.join(work, "mf_diag.log")
if os.path.isfile(diag_log):
    for line in open(diag_log, errors="replace"):
        m = re.search(r'ws_bcast_fields=(\d+)', line)
        if m:
            v = int(m.group(1))
            if v > ws_bcast_fields_max:
                ws_bcast_fields_max = v

ws_hits   = sum(1 for _ in open(os.path.join(work, "ws_hit.log"),  errors="replace") if _.strip()) if os.path.isfile(os.path.join(work, "ws_hit.log")) else 0
ws_misses = sum(1 for _ in open(os.path.join(work, "ws_miss.log"), errors="replace") if _.strip()) if os.path.isfile(os.path.join(work, "ws_miss.log")) else 0

EXPECTED_LNL = -7541976.860
TOL = 0.5
verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({
        "check": "lnL",
        "status": "pass" if diff < TOL else "fail",
        "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6),
        "note": "W2: MF lnL vs baseline ref 168425673; tol=0.5",
    })
if best_model is not None:
    verify.append({
        "check": "best_model",
        "status": "pass" if best_model == "LG+G4" else "fail",
        "expected": "LG+G4", "reported": best_model,
    })
if mf_wall is not None:
    verify.append({
        "check": "mf_wall",
        "status": "pass" if mf_wall <= 100.0 else "warn",
        "criterion_s": 100.0, "reported_s": mf_wall,
        "note": "W2: MF ≤100s vs FCA np=4 without WS (~149s)",
    })
verify.append({
    "check": "ws_bcast_fired",
    "status": "pass" if ws_bcast_fields_max > 0 else "fail",
    "ws_bcast_fields_max": ws_bcast_fields_max,
    "note": "Phase A.2 broadcast must carry at least 1 non-sentinel field",
})
verify.append({
    "check": "exit_code",
    "status": "pass" if iqrc == 0 else "fail",
    "reported": iqrc,
})

overall = "PASS" if all(v["status"] == "pass" for v in verify) else "FAIL"
rec = {
    "run_id": rid, "label": label, "phase": "A.2-W2",
    "job_id": os.environ.get("PBS_JOBID", ""),
    "binary": os.path.basename(ibin),
    "binary_md5": sh(f"md5sum '{ibin}' | awk '{{print $1}}'"),
    "source_commit": "5604606d",
    "alignment": os.path.basename(alignment),
    "nranks": nranks, "omp_per_rank": omp_per_rank, "total_threads": threads,
    "seed": int(os.environ.get("SEED", "1")),
    "mode": "TESTONLY",
    "wall_s": wall,
    "iqtree_wall_s": iqwall, "mf_wall_s": mf_wall,
    "reported_lnL": rep_ll, "best_model": best_model,
    "ws_bcast_fields_max": ws_bcast_fields_max,
    "ws_hits": ws_hits, "ws_misses": ws_misses,
    "exit_code": iqrc,
    "verify": verify,
    "overall": overall,
    "work_dir": work,
}
out = os.path.join(runs, f"{rid}.json")
with open(out, "w") as f:
    json.dump(rec, f, indent=2)
print(f"[w2] run record: {out}")
print(f"[w2] OVERALL: {overall}")
for v in verify:
    status = v["status"].upper()
    check  = v["check"]
    extra  = {k: val for k, val in v.items() if k not in ("check", "status", "note")}
    print(f"  [{status}] {check}: {extra}")
PYEOF

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "[w2] ── Summary ──"
echo "[w2] exit_code: ${IQRC}"
echo "[w2] wall: ${WALL}s"
echo "[w2] ws_bcast_fields_max: ${WS_BCAST_FIELDS_MAX}"
echo "[w2] MF-MPI-DIAG lines:"
cat "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo "  (none)"
echo ""
if [[ ${IQRC} -ne 0 ]]; then
    echo "[w2] FAIL: iqtree3 returned non-zero exit code ${IQRC}" >&2
    exit "${IQRC}"
fi
echo "[w2] DONE"
