#!/bin/bash
# run_cpu_bench_aa_100k_spr_ws_b1_mtest_np1.sh
# Purpose: 1-node Phase B.1 parity check — np=1, -m TEST, AA 100K.
#
# At np=1 the progressiveWarmStartBcast() no-ops immediately (nranks==1 guard).
# This run verifies:
#   1. No regression vs row B-np1 (FCA+WS-A.2 np=1, 258.010 s MF, job 170138713)
#   2. Full parity: lnL within ±0.1 of -7,541,976.860, model LG+G4
#   3. Zero overhead from Phase B bitmask bookkeeping at single-rank
#
# Binary:  iqtree3-mpi-fca-ws-b1, md5 5aaf6da4
# Compare: 170138713 (B-np1, FCA+WS-A.2 np=1, MF=258.010 s)
#          170149201 (D,     FCA+WS-B.1 np=2, MF=146.792 s)
#
#PBS -N iq-aa-100k-ws-b1-np1
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-b1"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
NP=1
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="AA"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_100k_ws_b1_mtest_np${NP}_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

module load openmpi/4.1.7
module load intel-compiler-llvm/2025.3.2

# Preflight checks
[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5 (ICX build)." >&2; exit 7
fi

ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
EXPECTED_MD5="5aaf6da444252642b7b744d0696ec9fb"
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: binary md5 mismatch. Expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}" >&2
    exit 4
fi

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  AA 100K FCA+WS B.1 — normalsr SPR, np=${NP}, -T ${THREADS}"
echo "║  binary:    $(basename "${IQTREE}")"
echo "║  md5:       ${ACTUAL_MD5}"
echo "║  mf_mode:   ${MF_MODE}"
echo "║  pbs_id:    ${PBS_ID_SHORT}"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Run (np=1, --bind-to none so IQ-TREE sees all 103 OMP threads) ───────────
START_EPOCH=$(date +%s)

set +e
mpirun -np "${NP}" \
    --mca coll ^ucc \
    --bind-to none \
    numactl --localalloc \
    "${IQTREE}" \
    -s "${ALIGNMENT}" -T "${THREADS}" -seed "${SEED}" ${MF_MODE} \
    --prefix "${WORK_DIR}/iqtree_inner" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
set -e

END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
cat "${WORK_DIR}/iqtree_stdout.log"
echo ""
echo "rc=${IQRC}  wall=${WALL}s"

# ── Parse results + parity check ────────────────────────────────────────────
REF_LNL=-7541976.860
REF_TOL=0.1
REF_MODEL="LG+G4"

MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_LNL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_MODEL=$(grep -oP 'Best-fit model(?:: | according to BIC: )\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
TREE_WALL=$(grep -oP 'Wall-clock time used for tree search:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
TOTAL_WALL=$(grep -oP 'Total wall-clock time used:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
# Phase B should NOT fire at np=1; verify no progressive bcast lines
PROG_WS_EVENTS=$(grep -c 'progressiveWS bcast:' \
    "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true)

echo ""
echo "=== PARITY CHECK (ref B-np1 job 170138713, tol ${REF_TOL}) ==="
echo "  lnL:                 ${BEST_LNL:-MISSING}"
echo "  model:               ${BEST_MODEL:-MISSING}"
echo "  MF wall(s):          ${MF_WALL:-MISSING}"
echo "  tree wall(s):        ${TREE_WALL:-MISSING}"
echo "  total wall(s):       ${TOTAL_WALL:-MISSING}"
echo "  progressiveWS events:${PROG_WS_EVENTS:-0}  (expected 0 at np=1)"
echo "  ref lnL:             ${REF_LNL}"
echo "  row B-np1 (MF):      258.010 s  (FCA+WS-A.2 np=1, job 170138713)"

python3 - <<PYEOF
import json, os, subprocess, sys

ref_lnl      = ${REF_LNL}
tol          = ${REF_TOL}
mf_wall      = float("${MF_WALL}") if "${MF_WALL}" else None
tree_wall    = float("${TREE_WALL}") if "${TREE_WALL}" else None
total_wall   = float("${TOTAL_WALL}") if "${TOTAL_WALL}" else None
best_lnl     = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model   = "${BEST_MODEL}" or None
prog_ws_evts = int("${PROG_WS_EVENTS}") if "${PROG_WS_EVENTS}" else 0

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

checks = {}
checks["exit_0"]                 = ${IQRC} == 0
checks["lnL_parity"]             = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_LG+G4"]            = best_model == "${REF_MODEL}"
checks["no_progressiveWS_np1"]   = prog_ws_evts == 0   # guard must no-op at np=1
all_pass = all(checks.values())

print()
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

row_bnp1_mf = 258.010
if mf_wall:
    delta = mf_wall - row_bnp1_mf
    print(f"\n  MF wall:               {mf_wall:.3f} s")
    print(f"  vs row B-np1 (WS-A.2): {delta:+.3f} s  ({'regression' if delta > 2 else 'OK — within noise'})")
    if best_lnl:
        print(f"  lnL delta vs ref:      {abs(best_lnl - ref_lnl):.4f}")

record = {
  "run_id": "gadi_${RUN_LABEL}_${PBS_ID_SHORT}", "label": "${RUN_LABEL}",
  "platform": "gadi", "run_type": "fca_ws_b1_mtest_np1",
  "dataset": "${ALIGNMENT}", "dataset_short": "complex_aa_100k",
  "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
  "np": ${NP}, "threads": ${THREADS}, "seed": ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary": "${IQTREE}",
  "binary_md5": "${ACTUAL_MD5}",
  "binary_note": "FCA Phase B.1 progressive warm-start, ICX 2025.3.2 + OpenMPI 4.1.7",
  "summary": {
    "pass": 1 if ${IQRC} == 0 else 0,
    "all_pass": all_pass,
    "lnL": best_lnl,
    "model": best_model,
    "mf_wall_s": mf_wall,
    "tree_wall_s": tree_wall,
    "total_wall_s": total_wall,
    "progressiveWS_events": prog_ws_evts,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
    "parity_checks": checks,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu": sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
            "ncpus": os.environ.get("PBS_NCPUS"), "project": "dx61"},
  },
  "build_tag": "fca_ws_b1_icx_avx512_spr",
  "headline_table_row": "D-np1",
  "compares_to": {
    "row_B-np1": "170138713 (FCA+WS-A.2 np=1, 258.010 s MF)",
    "row_D":     "170149201 (FCA+WS-B.1 np=2, 146.792 s MF)",
  },
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[fca_ws_b1_np1] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
