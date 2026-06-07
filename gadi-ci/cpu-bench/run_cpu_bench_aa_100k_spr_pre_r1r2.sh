#!/bin/bash
# run_cpu_bench_aa_100k_spr_pre_r1r2.sh
# Purpose: Establish the pre-R1+R2-patch ICX baseline for AA 100K.
#
# Fills the "missing row" in the Headline Run Progression table:
#   Row A  = ICX + R1+R2 + AVX-512 non-MPI  (168425673, 399 s, -m TEST)
#   THIS   = ICX + AVX-512, no patches, MPI np=1 = FCA-inactive  (-m TEST)
#
# Binary:  /scratch/rc29/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi
#          Source: v3.1.2 tag (4e91dd61), zero patches, built 2026-05-08
#          Compiler: ICX/LLVM (libiomp5, -march=sapphirerapids)
#          md5: 869c010f23754e95f0805fa475eb9807
#          NOTE: MPI binary at np=1 — FCA dispatch inactive; behaves as
#                single-rank OMP-across-models (equivalent to non-MPI at np=1).
#                Minimal MPI startup overhead (~1-2 s) is negligible.
#
# Parity target: lnL within ±0.5 of -7,541,976.860 (ref 168425673), model LG+G4
# MF mode: -m TEST  (same as row A — 224 models, stops at +G4)
#
#PBS -N iq-aa-100k-pre-r1r2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="/scratch/rc29/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="AA"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_100k_pre_r1r2_seed${SEED}"
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
EXPECTED_MD5="869c010f23754e95f0805fa475eb9807"
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

NUMACTL=()
command -v numactl >/dev/null 2>&1 && NUMACTL=(numactl --localalloc)

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  AA 100K pre-R1+R2 baseline — normalsr SPR, np=1, -T ${THREADS}"
echo "║  binary:    $(basename "${IQTREE}")"
echo "║  md5:       ${ACTUAL_MD5}"
echo "║  mf_mode:   ${MF_MODE}"
echo "║  numactl:   ${NUMACTL[*]:-disabled}"
echo "║  pbs_id:    ${PBS_ID_SHORT}"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Pass 1: full IQ-TREE run ────────────────────────────────────────────────
START_EPOCH=$(date +%s)

set +e
mpirun -np 1 --mca coll ^ucc --bind-to none \
    "${NUMACTL[@]}" "${IQTREE}" \
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

# ── Pass 2: perf stat ───────────────────────────────────────────────────────
PERF_EVENTS="cycles:u,instructions:u,branch-instructions:u,branch-misses:u,\
cache-references:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,\
LLC-loads:u,LLC-load-misses:u,dTLB-loads:u,dTLB-load-misses:u,\
iTLB-loads:u,iTLB-load-misses:u"
if [[ ${IQRC} -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "Pass 2: perf stat..."
    perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
        mpirun -np 1 --mca coll ^ucc --bind-to none \
        "${NUMACTL[@]}" "${IQTREE}" \
        -s "${ALIGNMENT}" -T "${THREADS}" -seed "${SEED}" ${MF_MODE} \
        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Parse results + parity check ────────────────────────────────────────────
REF_LNL=-7541976.860
REF_TOL=0.5
REF_MODEL="LG+G4"

MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_LNL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_MODEL=$(grep -oP 'Best-fit model according to BIC:\s*\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)

echo ""
echo "=== PARITY CHECK (ref 168425673, tol ${REF_TOL}) ==="
echo "  lnL:        ${BEST_LNL:-MISSING}"
echo "  model:      ${BEST_MODEL:-MISSING}"
echo "  MF wall(s): ${MF_WALL:-MISSING}"
echo "  ref lnL:    ${REF_LNL}"

python3 - <<PYEOF
import json, os, re, subprocess, sys
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

ref_lnl = ${REF_LNL}
tol     = ${REF_TOL}
mf_wall = float("${MF_WALL}") if "${MF_WALL}" else None
best_lnl   = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model = "${BEST_MODEL}" or None

checks = {}
checks["exit_0"]     = ${IQRC} == 0
checks["lnL_parity"] = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_LG+G4"] = best_model == "${REF_MODEL}"
all_pass = all(checks.values())

print()
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

def _parse_perf(path):
    agg = {}
    if not os.path.isfile(path): return agg
    for line in open(path, errors="replace"):
        m = re.match(r"\s*([\d,]+)\s+([\w.\-:/]+)", line)
        if m:
            try: agg[m.group(2).split(":",1)[0]] = int(m.group(1).replace(",",""))
            except ValueError: pass
    return agg
def _rate(n, d):
    if not n or not d: return None
    return round(100.0 * n / d, 4)
_agg = _parse_perf("${WORK_DIR}/perf_stat.txt")
def _g(*keys):
    for k in keys:
        if _agg.get(k) is not None: return _agg[k]
    return None
_cyc, _ins = _g("cycles"), _g("instructions")
_metrics = {k: v for k, v in {
    "IPC":             round(_ins/_cyc, 4) if _cyc and _ins else None,
    "cache-miss-rate": _rate(_g("cache-misses"), _g("cache-references")),
}.items() if v is not None}

record = {
  "run_id": "gadi_${RUN_LABEL}_${PBS_ID_SHORT}", "label": "${RUN_LABEL}",
  "platform": "gadi", "run_type": "cpu_bench_pre_r1r2",
  "dataset": "${ALIGNMENT}", "dataset_short": "complex_aa_100k",
  "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
  "threads": ${THREADS}, "seed": ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary": "${IQTREE}",
  "binary_md5": "${ACTUAL_MD5}",
  "binary_note": "v3.1.2 tag (4e91dd61), no patches, ICX -march=sapphirerapids, MPI np=1",
  "summary": {
    "pass": 1 if ${IQRC} == 0 else 0,
    "all_pass": all_pass,
    "lnL": best_lnl,
    "model": best_model,
    "mf_wall_s": mf_wall,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
    "parity_checks": checks,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu": sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "cores": int(sh("nproc","0") or 0),
    "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
            "ncpus": os.environ.get("PBS_NCPUS"), "project": "dx61"},
  },
  "build_tag": "v3.1.2_icx_avx512_spr_no_patches",
  "headline_table_row": "pre-R1R2 (new)",
  "compares_to": "168425673 (row A, ICX+R1+R2, 399.456 s MF)",
}
if _metrics:
    record["profile"] = {"metrics": _metrics}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[cpu_bench] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
