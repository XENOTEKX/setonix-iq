#!/bin/bash
# run_cpu_bench_dna_1m_spr_ws_b1_mtest_np2.sh
# Purpose: FCA + WS Phase B.1 — 2-node MPI, np=2, -m TEST, DNA 1M.
#
# Phase B.1 (ws-b1): MPI_Allreduce(OR) in getNextModel() fires as soon as rank 0's
# first rate class (e.g. +G) has been evaluated, broadcasting converged parameters
# to all ranks *before* filterRatesMPI pruning.  This is the pre-pruning broadcast.
#
# Compares directly against:
#   WS-A.2 np=2 (job 170148472):  MF=2,439.701 s  (post-pruning broadcast, regresses +0.55%)
#   no-WS  np=2 (job 170148473):  MF=2,426.462 s  (no warm-start, baseline)
#
# Binary:  iqtree3-mpi-fca-ws-b1, md5 5aaf6da444252642b7b744d0696ec9fb
#
# Parity:  lnL within ±1.0 of -59,208,019.212, model F81+F+G4
#
#PBS -N iq-dna-1m-ws-b1-np2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1020GB
#PBS -l place=scatter:excl
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-b1"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy"
EXPECTED_MD5="5aaf6da444252642b7b744d0696ec9fb"
NP=2
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="DNA"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_1m_ws_b1_mtest_np${NP}_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
LOG_DIR="${WORK_DIR}/rank_logs"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Modules ───────────────────────────────────────────────────────────
module load openmpi/4.1.7
module load intel-compiler-llvm/2025.3.2

# ── Preflight ─────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }

ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: binary md5 mismatch. Expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}" >&2
    exit 4
fi
echo "[preflight] md5 MATCH: ${ACTUAL_MD5}"

if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5 (ICX build)." >&2; exit 7
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — expected MPI build." >&2; exit 5
fi

# Verify Phase B.1 symbol (progressiveWarmStartBcast)
if nm "${IQTREE}" 2>/dev/null | grep -q 'progressiveWarmStartBcast\|progressiveWS'; then
    echo "[preflight] Phase B.1 progressiveWarmStartBcast: confirmed via nm"
elif strings "${IQTREE}" 2>/dev/null | grep -q 'progressiveWS bcast'; then
    echo "[preflight] Phase B.1 progressiveWS bcast: found via strings"
else
    echo "[preflight] WARNING: Phase B.1 progressiveWarmStartBcast symbol not confirmed — binary may be wrong" >&2
fi

# ── OMP env ───────────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# ── Host topology ─────────────────────────────────────────────────────
HOSTFILE="${WORK_DIR}/hostfile"
RANKFILE="${WORK_DIR}/rankfile"
NODES=($(sort -u "$PBS_NODEFILE"))
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected 2 nodes, got ${#NODES[@]}" >&2; exit 9
fi
echo "[topology] nodes: ${NODES[*]}"
> "${HOSTFILE}"
> "${RANKFILE}"
for i in "${!NODES[@]}"; do
    echo "${NODES[$i]} slots=104" >> "${HOSTFILE}"
    echo "rank ${i}=${NODES[$i]} slot=0-103" >> "${RANKFILE}"
done

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE FCA+WS-B.1 — DNA 1M, np=2, 2-node, -m TEST"
echo "║  binary:   $(basename "${IQTREE}")"
echo "║  md5:      ${ACTUAL_MD5}"
echo "║  np:       ${NP}  threads: ${THREADS}  seed: ${SEED}"
echo "║  pbs_id:   ${PBS_ID_SHORT}"
echo "║  node 0:   ${NODES[0]}"
echo "║  node 1:   ${NODES[1]}"
echo "║  WS-A.2 np=2 ref: 170148472 (MF=2,439.701 s)"
echo "║  no-WS  np=2 ref: 170148473 (MF=2,426.462 s)"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Run ───────────────────────────────────────────────────────────────
START_EPOCH=$(date +%s)
set +e
mpirun -np "${NP}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --mca coll ^ucc \
    --output-filename "${LOG_DIR}/" \
    "${IQTREE}" \
    -s "${ALIGNMENT}" -T "${THREADS}" -seed "${SEED}" ${MF_MODE} \
    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
set -e
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_stdout.log"
echo ""
echo "rc=${IQRC}  wall=${WALL}s"

# ── Collect per-rank logs ──────────────────────────────────────────────
for f in "${LOG_DIR}"/*/rank.*/stdout; do
    [[ -f "$f" ]] || continue
    rank=$(echo "$f" | sed -E 's|.*/rank\.([0-9]+)/stdout|\1|')
    cp -f "$f" "${WORK_DIR}/rank_${rank}.stdout.log"
done

{ for r in "${WORK_DIR}"/rank_*.stdout.log; do
    [[ -f "$r" ]] && grep -E '^MF-TIME: ' "$r" || true
  done } > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
{ for r in "${WORK_DIR}"/rank_*.stdout.log; do
    [[ -f "$r" ]] && grep -E '^MF-MPI-DIAG: ' "$r" || true
  done } > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true

# Fall back to merged stdout
if [[ ! -s "${WORK_DIR}/mf_time.log" ]]; then
    grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_stdout.log" > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
    grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_stdout.log" > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true
fi

echo "[diag] MF-TIME lines:     $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[diag] MF-MPI-DIAG lines: $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"

# ── Phase B.1 specific diagnostics ────────────────────────────────────
PROG_WS_EVENTS=$(grep -c 'progressiveWS bcast:' \
    "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null || true)
PROG_WS_FIRST=$(grep 'progressiveWS bcast:' \
    "${WORK_DIR}/iqtree_stdout.log" 2>/dev/null | head -2 || true)

echo "[diag] Phase B.1 progressiveWS events: ${PROG_WS_EVENTS:-0}"
echo "[diag] Phase A.2 filterRatesMPI diag:"
grep -E "filterRatesMPI fired|ws_bcast_fields" "${WORK_DIR}/mf_diag.log" 2>/dev/null | sed 's/^/    /' || true

# ── Parse + parity ────────────────────────────────────────────────────
REF_LNL=-59208019.212
REF_TOL=1.0
REF_MODEL="F81+F+G4"

MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
SPR_WALL=$(grep -oP 'Wall-clock time used for tree search:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_LNL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_MODEL=$(grep -oP 'Best-fit model:\s*\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
TOTAL_WALL=$(grep -oP 'Total wall-clock time used:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
WS_BCAST=$(grep -ohP 'ws_bcast_fields=\K\d+' \
    "${WORK_DIR}/iqtree_stdout.log" "${WORK_DIR}/mf_diag.log" 2>/dev/null | tail -1 || true)

echo ""
echo "=== PARITY CHECK (ref ${REF_LNL}, tol ${REF_TOL}) ==="
echo "  MF wall(s):           ${MF_WALL:-MISSING}"
echo "  SPR wall(s):          ${SPR_WALL:-MISSING}"
echo "  total wall(s):        ${TOTAL_WALL:-MISSING}"
echo "  lnL:                  ${BEST_LNL:-MISSING}"
echo "  model:                ${BEST_MODEL:-MISSING}"
echo "  progressiveWS events: ${PROG_WS_EVENTS:-0}"
echo "  ws_bcast_fields (A2): ${WS_BCAST:-0 (not fired)}"
echo "  WS-A.2 ref (170148472): MF=2,439.701 s  (regresses vs no-WS)"
echo "  no-WS  ref (170148473): MF=2,426.462 s  (baseline)"

python3 - <<PYEOF
import json, os, re, subprocess, sys

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

ref_lnl        = ${REF_LNL}
tol            = ${REF_TOL}
mf_wall        = float("${MF_WALL}") if "${MF_WALL}" else None
spr_wall       = float("${SPR_WALL}") if "${SPR_WALL}" else None
total_wall     = float("${TOTAL_WALL}") if "${TOTAL_WALL}" else None
best_lnl       = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model     = "${BEST_MODEL}" or None
prog_ws_events = int("${PROG_WS_EVENTS}") if "${PROG_WS_EVENTS}" else 0
ws_bcast       = int("${WS_BCAST}") if "${WS_BCAST}" else 0

# Per-rank model counts from mf_time.log
per_rank = {}
mf_time_log = "${WORK_DIR}/mf_time.log"
if os.path.isfile(mf_time_log):
    for line in open(mf_time_log, errors="replace"):
        m = re.search(r'rank (\d+) .* dt=([\d.]+)', line)
        if m:
            r = int(m.group(1))
            per_rank.setdefault(r, []).append(float(m.group(2)))
mf_time_summary = {
    f"rank_{r}": {"n_models": len(v), "total_s": round(sum(v),3),
                  "mean_s": round(sum(v)/len(v),3) if v else None,
                  "max_s": round(max(v),3) if v else None}
    for r, v in sorted(per_rank.items())
}

checks = {}
checks["exit_0"]               = ${IQRC} == 0
checks["lnL_parity"]           = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_F81+G4"]         = best_model is not None and best_model.startswith("F81")
checks["progressiveWS_fired_B1"] = prog_ws_events > 0
all_pass = all(checks.values())

print()
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

ws_a2_mf  = 2439.701  # 170148472
nows_mf   = 2426.462  # 170148473
if mf_wall:
    delta_vs_nows  = mf_wall - nows_mf
    delta_vs_wsa2  = mf_wall - ws_a2_mf
    print(f"\n  MF wall:                {mf_wall:.3f} s")
    print(f"  vs no-WS  (170148473):  {delta_vs_nows:+.3f} s  ({'regression' if delta_vs_nows > 1 else 'improvement' if delta_vs_nows < -1 else 'neutral'})")
    print(f"  vs WS-A.2 (170148472):  {delta_vs_wsa2:+.3f} s  ({'regression' if delta_vs_wsa2 > 1 else 'improvement' if delta_vs_wsa2 < -1 else 'neutral'})")
    print(f"  progressiveWS events:   {prog_ws_events}")

record = {
  "run_id":   f"gadi_${RUN_LABEL}_${PBS_ID_SHORT}",
  "label":    "${RUN_LABEL}",
  "platform": "gadi",
  "run_type": "fca_ws_b1_mtest",
  "dataset":  "${ALIGNMENT}",
  "dataset_short": "complex_dna_1m",
  "data_type": "${DATA_TYPE}",
  "seq_len": 1000000,
  "n_taxa":  100,
  "np": ${NP},
  "threads": ${THREADS},
  "seed":    ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary":  "${IQTREE}",
  "binary_md5": "${ACTUAL_MD5}",
  "binary_note": "FCA Phase B.1 progressive pre-pruning warm-start broadcast, ICX 2025.3.2 + OpenMPI 4.1.7",
  "warm_start": {"phase": "B.1"},
  "summary": {
    "pass":          1 if ${IQRC} == 0 else 0,
    "all_pass":      all_pass,
    "lnL":           best_lnl,
    "model":         best_model,
    "mf_wall_s":     mf_wall,
    "spr_wall_s":    spr_wall,
    "total_wall_s":  total_wall,
    "progressiveWS_events": prog_ws_events,
    "ws_bcast_fields_A2": ws_bcast,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
    "delta_mf_vs_nows_s":  round(mf_wall - nows_mf, 3) if mf_wall else None,
    "delta_mf_vs_wsa2_s":  round(mf_wall - ws_a2_mf, 3) if mf_wall else None,
    "parity_checks": checks,
  },
  "mf_time_summary": mf_time_summary,
  "env": {
    "hostname": sh("hostname"),
    "date":     sh("date -Iseconds"),
    "cpu":      sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "pbs": {
      "job_id": os.environ.get("PBS_JOBID"),
      "queue":  os.environ.get("PBS_QUEUE"),
      "ncpus":  os.environ.get("PBS_NCPUS"),
      "project": "dx61",
    },
  },
  "build_tag": "fca_ws_b1_icx_avx512_spr",
  "headline_table_row": "DNA-1M WS-B.1 np=2",
  "compares_to": {
    "dna_1m_ws_a2_np2":  "170148472 (FCA+WS-A.2 np=2, MF=2,439.701 s — A.2 regresses)",
    "dna_1m_nows_np2":   "170148473 (FCA no-WS np=2, MF=2,426.462 s — baseline)",
    "aa_100k_ws_b1_np2": "170149201 (AA 100K WS-B.1 np=2, MF=146.792 s — Phase B reference)",
  },
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[ws_b1] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
