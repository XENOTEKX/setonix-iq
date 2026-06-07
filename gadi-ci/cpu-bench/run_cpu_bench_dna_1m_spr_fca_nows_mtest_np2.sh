#!/bin/bash
# run_cpu_bench_dna_1m_spr_fca_nows_mtest_np2.sh
# Purpose: FCA WITHOUT warm-start — 2-node MPI, np=2, -m TEST, DNA 1M.
#
# Baseline binary: iqtree3-mpi-fca-phase0506 (symlink → iqtree3-mpi-fca-lbfgs-ws)
# This is the FCA Phase 0.5+0.6 binary WITHOUT the WS A.2 broadcast patch.
# md5 a103bc6c97860145033206c47b184367  (same binary used in job 168913091 np=1)
#
# Parity:  lnL within ±1.0 of -59,208,019.212, model F81+F+G4
#          SPR ref: 168425675 (normalsr 103T, F81+F+G4, total=6114.5s)
#
# Purpose: Direct A/B comparison against run_cpu_bench_dna_1m_spr_ws_a2_mtest_np2.sh
#   to quantify how much warm-start (WS A.2) reduces MF wall time on DNA 1M at np=2.
#   The existing np=2 no-WS run (168580377) was MF-only (-m TESTO) and partial — NOT
#   comparable.  This script provides the apples-to-apples np=2 no-WS full run.
#
# Compares to:
#   FCA no-WS np=1:  168913091 (MF=5121.2s, same binary at np=1)
#   FCA no-WS np=8:  168592214 (MF=1274.7s)
#   FCA+WS-A.2 np=2: run_cpu_bench_dna_1m_spr_ws_a2_mtest_np2.sh (NEW)
#   AA 1M FCA no-WS np=2: 168635614 (MF=3076.9s, AA 1M — different dataset)
#
#PBS -N iq-dna-1m-fca-nows-np2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1020GB
#PBS -l place=scatter:excl
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="/scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-phase0506"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy"
EXPECTED_MD5="a103bc6c97860145033206c47b184367"
NP=2
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="DNA"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_1m_fca_nows_mtest_np${NP}_seed${SEED}"
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
    echo "WARNING: md5 mismatch. Expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}" >&2
    echo "         Continuing — post-run lnL parity check will validate." >&2
else
    echo "[preflight] md5 MATCH: ${ACTUAL_MD5}"
fi

if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5 (ICX build)." >&2; exit 7
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — expected MPI build." >&2; exit 5
fi

# Confirm NO warm-start symbols (sanity check we have the right binary)
if nm "${IQTREE}" 2>/dev/null | grep -q 'WarmStartPacket\|ws_bcast'; then
    echo "[preflight] WARNING: WarmStartPacket/ws_bcast found — this may be a WS binary, not no-WS!" >&2
elif strings "${IQTREE}" 2>/dev/null | grep -q 'ws_bcast_fields'; then
    echo "[preflight] WARNING: ws_bcast_fields found via strings — check binary identity!" >&2
else
    echo "[preflight] no-WS confirmed: WarmStartPacket/ws_bcast_fields NOT found"
fi

# Confirm filterRatesMPI is present (FCA Phase 0.5+0.6 required)
if nm "${IQTREE}" 2>/dev/null | grep -q '_ZN17CandidateModelSet14filterRatesMPIEi'; then
    echo "[preflight] filterRatesMPI: confirmed via nm"
elif strings "${IQTREE}" 2>/dev/null | grep -q 'filterRatesMPI'; then
    echo "[preflight] filterRatesMPI: found via strings"
else
    echo "[preflight] WARNING: filterRatesMPI not found — may be wrong binary" >&2
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
echo "║  IQ-TREE FCA no-WS (phase0506) — DNA 1M, np=2, 2-node, -m TEST"
echo "║  binary:   $(basename "${IQTREE}")"
echo "║  md5:      ${ACTUAL_MD5}  (expected: ${EXPECTED_MD5})"
echo "║  np:       ${NP}  threads: ${THREADS}  seed: ${SEED}"
echo "║  pbs_id:   ${PBS_ID_SHORT}"
echo "║  node 0:   ${NODES[0]}"
echo "║  node 1:   ${NODES[1]}"
echo "║  FCA no-WS np=1 ref:    168913091 (MF=5121.2s, same binary)"
echo "║  FCA+WS-A.2 np=2 pair:  run_cpu_bench_dna_1m_spr_ws_a2_mtest_np2.sh"
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

if [[ ! -s "${WORK_DIR}/mf_time.log" ]]; then
    grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_stdout.log" > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
    grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_stdout.log" > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true
fi

echo "[diag] MF-TIME lines:     $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[diag] MF-MPI-DIAG lines: $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"
echo "[diag] filterRatesMPI diagnostic:"
grep -E "filterRatesMPI fired" "${WORK_DIR}/mf_diag.log" 2>/dev/null | sed 's/^/    /' || true

# ── Parse + parity ────────────────────────────────────────────────────
REF_LNL=-59208019.212
REF_TOL=1.0
REF_MODEL="F81+F+G4"

MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_LNL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_MODEL=$(grep -oP 'Best-fit model:\s*\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
TOTAL_WALL=$(grep -oP 'Total wall-clock time used:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)

echo ""
echo "=== PARITY CHECK (ref ${REF_LNL}, tol ${REF_TOL}) ==="
echo "  MF wall(s):    ${MF_WALL:-MISSING}"
echo "  total wall(s): ${TOTAL_WALL:-MISSING}"
echo "  lnL:           ${BEST_LNL:-MISSING}"
echo "  model:         ${BEST_MODEL:-MISSING}"

python3 - <<PYEOF
import json, os, re, subprocess, sys

ref_lnl    = ${REF_LNL}
tol        = ${REF_TOL}
mf_wall    = float("${MF_WALL}") if "${MF_WALL}" else None
total_wall = float("${TOTAL_WALL}") if "${TOTAL_WALL}" else None
best_lnl   = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model = "${BEST_MODEL}" or None

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

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
checks["exit_0"]       = ${IQRC} == 0
checks["lnL_parity"]   = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_F81+G4"] = best_model is not None and best_model.startswith("F81")
all_pass = all(checks.values())

for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

fca_np1_mf = 5121.153  # 168913091
fca_np8_mf = 1274.686  # 168592214
if mf_wall:
    print(f"\n  MF wall:          {mf_wall:.3f} s")
    print(f"  vs FCA np=1 MF:   {fca_np1_mf/mf_wall:.3f}×  (ref 168913091, same binary)")
    print(f"  vs FCA np=8 MF:   {fca_np8_mf/mf_wall:.3f}×  (ref 168592214)")
    print(f"  NOTE: compare MF wall directly to WS-A.2 np=2 run to measure warm-start gain")

record = {
  "run_id":   "gadi_${RUN_LABEL}_${PBS_ID_SHORT}",
  "label":    "${RUN_LABEL}",
  "platform": "gadi",
  "run_type": "fca_nows_mtest",
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
  "binary_note": "FCA Phase 0.5+0.6 (iqtree3-mpi-fca-phase0506 → iqtree3-mpi-fca-lbfgs-ws) NO warm-start, ICX 2025.3.2 + OpenMPI 4.1.7",
  "warm_start": False,
  "summary": {
    "pass":         1 if ${IQRC} == 0 else 0,
    "all_pass":     all_pass,
    "lnL":          best_lnl,
    "model":        best_model,
    "mf_wall_s":    mf_wall,
    "total_wall_s": total_wall,
    "ws_bcast_fields": 0,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
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
  "build_tag": "mf_iso_phase0.5_0.6_icx_avx512",
  "headline_table_row": "FCA no-WS DNA-1M np=2",
  "compares_to": {
    "dna_1m_fca_np1_nows":  "168913091 (DNA 1M FCA no-WS np=1, MF=5121.153s, same binary)",
    "dna_1m_fca_np8_nows":  "168592214 (DNA 1M FCA no-WS np=8, MF=1274.686s)",
    "dna_1m_ws_a2_np2":     "run_cpu_bench_dna_1m_spr_ws_a2_mtest_np2.sh (A/B warm-start pair)",
    "aa_1m_fca_np2_nows":   "168635614 (AA 1M FCA no-WS np=2, MF=3076.9s — different dataset)",
  },
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[fca_nows] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
