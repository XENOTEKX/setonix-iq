#!/bin/bash
# run_cpu_bench_aa_100k_spr.sh — IQ-TREE3 cpu_opt_merge CPU benchmark
# AA, 100K sites (100 taxa), Sapphire Rapids node, -nt 103, seed=1
# Binary:   cpu_opt_merge build-intel-vanila (ICX + AVX-512 + -xSAPPHIRERAPIDS + R1+R2 patches)
# Energy:   Linaro Forge perf-report
# Runtime:  numactl --localalloc + Intel OMP (libiomp5, KMP_BLOCKTIME=200)
#
#PBS -N iq-cpu-aa-100k-spr
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=08:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

IQTREE="${IQTREE:-/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3}"
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
THREADS=103
SEED=1
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_100k"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_100k_spr_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILE_REPORT="${WORK_DIR}/perf_report"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

module load linaro-forge/24.0.2
module load intel-compiler-llvm/2024.2.1

[[ -x "${IQTREE}" ]]    || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5 (ICX build)." >&2; exit 7
fi

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=10000

NUMACTL=()
command -v numactl >/dev/null 2>&1 && NUMACTL=(numactl --localalloc)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  CPU Benchmark — AA 100K · normalsr · -nt ${THREADS}"
echo "║  binary:  $(basename "${IQTREE}")"
echo "║  numactl: ${NUMACTL[*]:-disabled}"
echo "║  pbs_id:  ${PBS_ID_SHORT}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

START_EPOCH=$(date +%s)

set +e
perf-report --no-mpi --output="${PROFILE_REPORT}" \
    "${NUMACTL[@]}" "${IQTREE}" -s "${ALIGNMENT}" -nt "${THREADS}" -seed "${SEED}" \
    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1
PERF_RC=$?
set -e

END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
cat "${WORK_DIR}/iqtree_run.log"
echo ""

# perf-report wraps iqtree3; infer IQ-TREE's actual exit from log
if grep -q "^Date and Time:" "${WORK_DIR}/iqtree_run.log" 2>/dev/null; then
    IQRC=0
    [[ ${PERF_RC} -ne 0 ]] && echo "NOTE: perf-report exited ${PERF_RC} (IQ-TREE completed OK — hw counters restricted on this node)" >&2
else
    IQRC=${PERF_RC}
fi
echo "rc=${IQRC}  perf_rc=${PERF_RC}  wall=${WALL}s"

# Pass 2 — perf stat (user-mode :u events, compatible with perf_event_paranoid=2)
PERF_EVENTS="cycles:u,instructions:u,branch-instructions:u,branch-misses:u,\
cache-references:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,\
LLC-loads:u,LLC-load-misses:u,dTLB-loads:u,dTLB-load-misses:u,\
iTLB-loads:u,iTLB-load-misses:u"
if [[ ${IQRC} -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "Pass 2: perf stat..."
    perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
        "${NUMACTL[@]}" "${IQTREE}" -s "${ALIGNMENT}" -nt "${THREADS}" -seed "${SEED}" \
        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

REP_LL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' "${WORK_DIR}/iqtree_run.log" | tail -1 || true)
IQ_WALL=$(grep -oP 'Total wall-clock time used:\s*\K[\d.]+' "${WORK_DIR}/iqtree_run.log" | tail -1 || true)

python3 - <<PYEOF
import json, os, re, subprocess
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d
rep_ll  = "${REP_LL}" or None
iq_wall = float("${IQ_WALL}") if "${IQ_WALL}" else ${WALL}
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
    "IPC":                 round(_ins/_cyc, 4) if _cyc and _ins else None,
    "cache-miss-rate":     _rate(_g("cache-misses"),          _g("cache-references")),
    "branch-miss-rate":    _rate(_g("branch-misses"),         _g("branch-instructions")),
    "L1-dcache-miss-rate": _rate(_g("L1-dcache-load-misses"), _g("L1-dcache-loads")),
    "LLC-miss-rate":       _rate(_g("LLC-load-misses"),       _g("LLC-loads")),
    "dTLB-miss-rate":      _rate(_g("dTLB-load-misses"),      _g("dTLB-loads")),
    "iTLB-miss-rate":      _rate(_g("iTLB-load-misses"),      _g("iTLB-loads")),
}.items() if v is not None}
for _k in ("cycles","instructions","cache-references","cache-misses",
           "branch-instructions","branch-misses","L1-dcache-loads","L1-dcache-load-misses",
           "LLC-loads","LLC-load-misses","dTLB-loads","dTLB-load-misses",
           "iTLB-loads","iTLB-load-misses"):
    if _k in _agg: _metrics[_k] = _agg[_k]
record = {
  "run_id": "gadi_${RUN_LABEL}_${PBS_ID_SHORT}", "label": "${RUN_LABEL}",
  "platform": "gadi", "run_type": "cpu_bench",
  "dataset": "${ALIGNMENT}", "dataset_short": "${DATASET_SHORT}",
  "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
  "threads": ${THREADS}, "seed": ${SEED},
  "timing": [{"command": "perf-report ... numactl --localalloc iqtree3 -s alignment_100000.phy -nt ${THREADS} -seed ${SEED}",
              "time_s": iq_wall}],
  "summary": {
    "pass": 1 if ${IQRC} == 0 else 0, "fail": 0 if ${IQRC} == 0 else 1,
    "total_time": iq_wall, "lnL": float(rep_ll) if rep_ll else None,
    "all_pass": ${IQRC} == 0,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu": sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "cores": int(sh("nproc","0") or 0), "kernel": sh("uname -r"),
    "omp": {"proc_bind": "close", "places": "cores", "kmp_blocktime": 200,
            "wait_policy": "PASSIVE", "numactl": "--localalloc"},
    "iqtree_binary": "${IQTREE}",
    "iqtree_version": sh("${IQTREE} --version 2>&1|head -1"),
    "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
            "ncpus": os.environ.get("PBS_NCPUS"), "project": "dx61"},
  },
  "build_tag": "cpu_opt_merge_icx_avx512_spr",
  "perf_report": "${PROFILE_REPORT}.html",
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
if _metrics:
    record["profile"] = {"metrics": _metrics}
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"[cpu_bench] wrote {out}")
PYEOF

exit "${IQRC}"
