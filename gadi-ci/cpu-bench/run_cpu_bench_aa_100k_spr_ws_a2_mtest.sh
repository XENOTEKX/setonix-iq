#!/bin/bash
# run_cpu_bench_aa_100k_spr_ws_a2_mtest.sh
# Purpose: Row C rerun — FCA + WS Phase A.2, 2-node, -m TEST (comparable to rows A and B).
#
# Row C original (169332772) used -m MF (all FreeRate models, ~1,386 candidates).
# This script reruns with -m TEST (~224 models, same as rows A and B) for direct comparison.
# Also isolates whether warm-start provides speedup over plain FCA at -m TEST np=2.
#
# Binary:  iqtree3-mpi-fca-ws-a2, md5 1547a906f1f75422514b0a0cdf2bc89e
#          Phase A.2 warm-start broadcast binary (cross-rank MPI_Bcast after ref family)
# Compare: 168584736 (row B, FCA no-WS -m TEST np=2, MF=149.029 s)
#          168425673 (row A, baseline R1+R2 -m TEST np=1, MF=399.456 s)
#
# Parity:  lnL within ±0.1 of -7,541,976.860, model LG+G4
#
#PBS -N iq-aa-100k-ws-a2-mtest
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1020GB
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a2"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
NP=2
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="AA"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_100k_ws_a2_mtest_np${NP}_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
LOG_DIR="${WORK_DIR}/rank_logs"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${RUNS_DIR}"
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
EXPECTED_MD5="1547a906f1f75422514b0a0cdf2bc89e"
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
echo "║  AA 100K FCA+WS A.2 — normalsr SPR, np=${NP}, -T ${THREADS}"
echo "║  binary:    $(basename "${IQTREE}")"
echo "║  md5:       ${ACTUAL_MD5}"
echo "║  mf_mode:   ${MF_MODE}"
echo "║  pbs_id:    ${PBS_ID_SHORT}"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Build hostfile — 1 rank per node, full-node OMP
HOSTFILE="${WORK_DIR}/hostfile"
RANKFILE="${WORK_DIR}/rankfile"
NODES=($(sort -u "$PBS_NODEFILE"))
echo "[topology] nodes: ${NODES[*]}"
> "${HOSTFILE}"
> "${RANKFILE}"
for i in "${!NODES[@]}"; do
    echo "${NODES[$i]} slots=104" >> "${HOSTFILE}"
    echo "rank ${i}=${NODES[$i]} slot=0-103"  >> "${RANKFILE}"
done

# ── Pass 1: full IQ-TREE run ────────────────────────────────────────────────
START_EPOCH=$(date +%s)

set +e
mpirun -np "${NP}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --mca coll ^ucc \
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
BEST_MODEL=$(grep -oP 'Best-fit model according to BIC:\s*\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
WS_BCAST=$(grep -oP 'ws_bcast_fields=\K\d+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)

echo ""
echo "=== PARITY CHECK (ref 168425673 / 168584736, tol ${REF_TOL}) ==="
echo "  lnL:            ${BEST_LNL:-MISSING}"
echo "  model:          ${BEST_MODEL:-MISSING}"
echo "  MF wall(s):     ${MF_WALL:-MISSING}"
echo "  ws_bcast_fields:${WS_BCAST:-0 (not fired)}"
echo "  ref lnL:        ${REF_LNL}"
echo "  row A (MF):     399.456 s  (ICX+R1+R2, non-MPI)"
echo "  row B (MF):     149.029 s  (FCA no-WS, np=2)"

python3 - <<PYEOF
import json, os, re, subprocess, sys
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

ref_lnl    = ${REF_LNL}
tol        = ${REF_TOL}
mf_wall    = float("${MF_WALL}") if "${MF_WALL}" else None
best_lnl   = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model = "${BEST_MODEL}" or None
ws_bcast   = int("${WS_BCAST}") if "${WS_BCAST}" else 0

checks = {}
checks["exit_0"]         = ${IQRC} == 0
checks["lnL_parity"]     = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_LG+G4"]    = best_model == "${REF_MODEL}"
checks["ws_bcast_fired"] = ws_bcast > 0
all_pass = all(checks.values())

print()
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

row_a_mf = 399.456
row_b_mf = 149.029
if mf_wall:
    print(f"\n  MF wall:       {mf_wall:.3f} s")
    print(f"  vs row A:      {row_a_mf/mf_wall:.2f}× speedup vs baseline")
    print(f"  vs row B:      {row_b_mf/mf_wall:.2f}× (WS vs no-WS FCA, same -m TEST np=2)")
    print(f"  lnL delta:     {abs(best_lnl - ref_lnl):.4f}" if best_lnl else "")

record = {
  "run_id": "gadi_${RUN_LABEL}_${PBS_ID_SHORT}", "label": "${RUN_LABEL}",
  "platform": "gadi", "run_type": "fca_ws_a2_mtest",
  "dataset": "${ALIGNMENT}", "dataset_short": "complex_aa_100k",
  "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
  "np": ${NP}, "threads": ${THREADS}, "seed": ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary": "${IQTREE}",
  "binary_md5": "${ACTUAL_MD5}",
  "binary_note": "FCA Phase A.2 warm-start broadcast, ICX 2025.3.2 + OpenMPI 4.1.7",
  "summary": {
    "pass": 1 if ${IQRC} == 0 else 0,
    "all_pass": all_pass,
    "lnL": best_lnl,
    "model": best_model,
    "mf_wall_s": mf_wall,
    "ws_bcast_fields": ws_bcast,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
    "speedup_vs_row_a": round(row_a_mf / mf_wall, 3) if mf_wall else None,
    "speedup_vs_row_b": round(row_b_mf / mf_wall, 3) if mf_wall else None,
    "parity_checks": checks,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu": sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
            "ncpus": os.environ.get("PBS_NCPUS"), "project": "dx61"},
  },
  "build_tag": "fca_ws_a2_icx_avx512_spr",
  "headline_table_row": "C-rerun",
  "compares_to": {
    "row_A": "168425673 (ICX+R1+R2 baseline, 399.456 s MF, -m TEST)",
    "row_B": "168584736 (FCA no-WS np=2, 149.029 s MF, -m TEST)",
    "row_C_orig": "169332772 (FCA+WS np=2, 481.295 s MF, -m MF — NOT COMPARABLE)",
  },
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[fca_ws] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
