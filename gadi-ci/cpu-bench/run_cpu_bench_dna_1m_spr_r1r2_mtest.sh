#!/bin/bash
# run_cpu_bench_dna_1m_spr_r1r2_mtest.sh
# Purpose: R1+R2+AVX-512 benchmark — 1-node non-MPI, -m TEST, DNA 1M.
#
# Same binary as AA 100K headline row "R1+R2+AVX512" (job 170139976, md5 9b60d9d2).
# Uses pre-built binary (no rebuild) to ensure bitwise comparison with AA 100K result.
#
# Patches applied vs vanilla v3.1.2:
#   P1/R1: schedule(static) + NUMA first-touch (phylokernelnew.h, phylotreesse.cpp)
#   P2:    CMakeLists.txt icpx/IntelLLVM AVX-512 detection fix (-mavx512f -mfma)
#   P3:    phylokernelavx512.cpp nonrev template-arity fixes
#   R2:    0004-thp-partial-lh-madvise.patch — madvise(MADV_HUGEPAGE) on central_partial_lh
#
# Binary:  /scratch/rc29/as1708/iqtree3-r1r2/build-icx-r1r2-avx512/iqtree3-r1r2
#          md5 9b60d9d24c27d44fa001acc90e300284  (same as AA 100K job 170139976)
# Parity:  lnL within ±1.0 of -59,208,019.212, model F81+F+G4
#          SPR ref: 168425675 (normalsr 103T, F81+F+G4, total=6114.5s)
#
# Purpose: Establish R1+R2+AVX512 baseline for DNA 1M.  Completes the
#   "headline progression" table for DNA 1M: vanilla → R1+R2+AVX512 → FCA np=2.
#
# Compares to:
#   FCA no-WS np=1:  168913091 (MF=5121.2s, total run)
#   FCA no-WS np=8:  168592214 (MF=1274.7s)
#   SPR baseline:    168425675 (total=6114.5s, non-MPI normalsr)
#
#PBS -N iq-dna-1m-r1r2-avx512
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/rc29+scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="/scratch/rc29/as1708/iqtree3-r1r2/build-icx-r1r2-avx512/iqtree3-r1r2"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy"
EXPECTED_MD5="9b60d9d24c27d44fa001acc90e300284"
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="DNA"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_1m_r1r2avx512_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"

# ── Modules ───────────────────────────────────────────────────────────
module load intel-compiler-llvm/2025.3.2 2>/dev/null || true

# ── Preflight ─────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }

ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "WARNING: md5 mismatch. Expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}" >&2
    echo "         Binary may differ from AA 100K job 170139976 — proceed with caution." >&2
else
    echo "[preflight] md5 MATCH: ${ACTUAL_MD5}"
fi

# Verify non-MPI (should not link libmpi)
if ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "WARNING: binary links libmpi — expected non-MPI build." >&2
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: binary links libgomp — expected libiomp5 (ICX build)." >&2; exit 7
fi
echo "[preflight] libs: $(ldd "${IQTREE}" | grep -oE 'libiomp5|libmpi[^.]*' | tr '\n' ' ')"

# ── OMP env ───────────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE R1+R2+AVX-512 — DNA 1M, 1-node non-MPI, -m TEST"
echo "║  binary:   $(basename "${IQTREE}")"
echo "║  md5:      ${ACTUAL_MD5}  (expected: ${EXPECTED_MD5})"
echo "║  threads:  ${THREADS}  seed: ${SEED}  mf_mode: ${MF_MODE}"
echo "║  pbs_id:   ${PBS_ID_SHORT}"
echo "║  work_dir: ${WORK_DIR}"
echo "║  SPR ref (168425675): lnL -59208019.212, F81+F+G4, total=6114.5s"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Run ───────────────────────────────────────────────────────────────
cd "${WORK_DIR}"

START_EPOCH=$(date +%s)
set +e
numactl --localalloc "${IQTREE}" \
    -s "${ALIGNMENT}" -nt "${THREADS}" -seed "${SEED}" ${MF_MODE} \
    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
set -e
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_stdout.log"
echo ""
echo "rc=${IQRC}  wall=${WALL}s"

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
echo "  MF wall(s):  ${MF_WALL:-MISSING}"
echo "  total(s):    ${TOTAL_WALL:-MISSING}"
echo "  lnL:         ${BEST_LNL:-MISSING}"
echo "  model:       ${BEST_MODEL:-MISSING}"

python3 - <<PYEOF
import json, os, subprocess, sys

ref_lnl    = ${REF_LNL}
tol        = ${REF_TOL}
mf_wall    = float("${MF_WALL}") if "${MF_WALL}" else None
total_wall = float("${TOTAL_WALL}") if "${TOTAL_WALL}" else None
best_lnl   = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model = "${BEST_MODEL}" or None

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

checks = {}
checks["exit_0"]       = ${IQRC} == 0
checks["lnL_parity"]   = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_F81+G4"] = best_model is not None and best_model.startswith("F81")
all_pass = all(checks.values())

for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

# Ref timings for context
fca_np1_mf = 5121.153  # 168913091
fca_np8_mf = 1274.686  # 168592214
spr_total  = 6114.5    # 168425675
if mf_wall:
    print(f"\n  MF wall:          {mf_wall:.3f} s")
    print(f"  vs FCA np=1 MF:   {fca_np1_mf/mf_wall:.3f}×  (ref 168913091)")
    print(f"  vs FCA np=8 MF:   {fca_np8_mf/mf_wall:.3f}×  (ref 168592214)")
if total_wall:
    print(f"  total wall:       {total_wall:.3f} s")
    print(f"  vs SPR total:     {spr_total/total_wall:.3f}×  (ref 168425675)")

record = {
  "run_id":   "gadi_${RUN_LABEL}_${PBS_ID_SHORT}",
  "label":    "${RUN_LABEL}",
  "platform": "gadi",
  "run_type": "r1r2_avx512_mtest_np1",
  "dataset":  "${ALIGNMENT}",
  "dataset_short": "complex_dna_1m",
  "data_type": "${DATA_TYPE}",
  "seq_len": 1000000,
  "n_taxa":  100,
  "np": 1,
  "threads": ${THREADS},
  "seed":    ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary":  "${IQTREE}",
  "binary_md5": "${ACTUAL_MD5}",
  "binary_note": "R1+R2+AVX-512 (gadi-spr-r2-avx512 @ 07966e4 + 0004-thp patch): schedule(static)+NUMA+cmake P2+P3+THP madvise, non-MPI, ICX 2025.3.2",
  "summary": {
    "pass":       1 if ${IQRC} == 0 else 0,
    "all_pass":   all_pass,
    "lnL":        best_lnl,
    "model":      best_model,
    "mf_wall_s":  mf_wall,
    "total_wall_s": total_wall,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
    "parity_checks": checks,
  },
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
  "build_tag": "r1_avx512_icx_spr_nonmpi",
  "headline_table_row": "R1+R2+AVX512 DNA-1M",
  "compares_to": {
    "aa_100k_r1r2_avx512": "170139976 (AA 100K, MF=221.594s)",
    "dna_1m_fca_np1_nows": "168913091 (DNA 1M FCA no-WS np=1, MF=5121.153s)",
    "dna_1m_fca_np8_nows": "168592214 (DNA 1M FCA no-WS np=8, MF=1274.686s)",
    "dna_1m_spr_baseline": "168425675 (DNA 1M SPR ref, total=6114.5s)",
  },
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[r1r2avx512] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
