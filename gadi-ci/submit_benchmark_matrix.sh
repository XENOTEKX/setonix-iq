#!/bin/bash
# submit_benchmark_matrix.sh — reproduce the Setonix benchmark corpus on
# Gadi Sapphire Rapids (normalsr). Enumerates (dataset × thread_count)
# pairs and submits one PBS job per pair. Each job writes a JSON run
# record directly to ${REPO_DIR}/logs/runs/ and an enriched profile dir
# under ${PROJECT_DIR}/gadi-ci/profiles/.
#
# Thread sweep: 1, 4, 13, 26, 52, 104
#   • 13  = one NUMA domain
#   • 26  = one NUMA pair (mimics small multi-NUMA effects)
#   • 52  = one socket
#   • 104 = full node
#
# Budget reference (2 SU/core-h, billed on full ncpus=104 reservation):
#   208 SU / node-hour.  See CHANGELOG.md for the detailed SU estimate.
#
# Usage:
#   ./submit_benchmark_matrix.sh                   # full matrix
#   ./submit_benchmark_matrix.sh large_modelfinder # single dataset
#   ./submit_benchmark_matrix.sh mega_dna 52 104   # dataset + thread subset
#
# Stages referenced in CHANGELOG:
#   stage1  — CI pipeline smoke (via run_pipeline.sh)
#   stage2  — one thread-point (pass "large_modelfinder 52")
#   stage3  — full matrix (no extra args)

set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
WORKER="${WORKER:-${PROJECT_DIR}/gadi-ci/_run_matrix_job.sh}"
LOGS_DIR="${PROJECT_DIR}/gadi-ci/logs"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${LOGS_DIR}" "${RUNS_DIR}"

# Default matrix: { dataset : thread-sweep }
declare -A MATRIX=(
  [large_modelfinder]="1 4 13 26 52 104"
  [xlarge_mf]="1 4 13 26 52 104"
  [mega_dna]="13 26 52 104"
)

SELECT_DATASET="${1:-}"
shift || true
SELECT_THREADS=("${@}")

# ── Emit the per-job worker (installed alongside the matrix script).
cat > "${WORKER}" <<'WORKER_EOF'
#!/bin/bash
# _run_matrix_job.sh — invoked by `qsub -v DATASET=... THREADS=... LABEL=...`
# Runs IQ-TREE under perf, emits a run.schema.json-conforming JSON.
set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling}"
IQTREE="${BUILD_DIR}/iqtree3"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="${DATASET:?DATASET env var required}"
THREADS="${THREADS:?THREADS env var required}"
LABEL="${LABEL:-${DATASET_NAME}_${THREADS}t_sr}"
SEED="${SEED:-1}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"

if command -v module >/dev/null 2>&1; then
    module load intel-vtune/2024.2.0         2>/dev/null || true
    module load intel-compiler-llvm/2024.2.0 2>/dev/null || true
fi

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_ID="$(date +%Y-%m-%d_%H%M%S)_${LABEL}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

echo "[matrix] run_id=${RUN_ID} dataset=${DATA_PATH} threads=${THREADS}"

PERF_EVENTS="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses,stalled-cycles-frontend,stalled-cycles-backend,\
topdown-total-slots,topdown-slots-issued,topdown-slots-retired,\
topdown-fetch-bubbles,topdown-recovery-bubbles"

START_EPOCH=$(date +%s)
perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
    "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1 || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

# Optional bounded VTune pass (skip for 1T to save SU).
if command -v vtune >/dev/null 2>&1 && [[ "${IQRC}" -eq 0 && "${THREADS}" -ge 13 ]]; then
    timeout --preserve-status 1800 \
        vtune -collect hotspots -knob sampling-mode=hw \
              -knob enable-stack-collection=true \
              -r "${WORK_DIR}/vtune_hotspots" \
              -- "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                 --prefix "${WORK_DIR}/iqtree_vtune" \
              > "${WORK_DIR}/vtune_collect.log" 2>&1 || true
    [[ -d "${WORK_DIR}/vtune_hotspots" ]] && \
        vtune -report summary -r "${WORK_DIR}/vtune_hotspots" \
              -format text > "${WORK_DIR}/vtune_summary.txt" 2>/dev/null || true
fi

# Emit the run.schema.json record.
python3 - "$@" <<PYEOF
import json, os, re, subprocess, time
work  = "${WORK_DIR}"
runs  = "${RUNS_DIR}"
rid   = "${RUN_ID}"
label = "${LABEL}"
thr   = int("${THREADS}")
pbs   = "${PBS_ID_SHORT}"
wall  = int("${WALL}")
iqrc  = int("${IQRC}")
ds    = "${DATASET_NAME}"
dpath = "${DATA_PATH}"
ibin  = "${IQTREE}"

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

# Extract IQ-TREE-reported wall-time + likelihood for the verify block.
log = os.path.join(work, "iqtree_run.log")
rep_ll = None
iqwall = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Log-likelihood of consensus tree\s*:\s*(-?[\d.]+)", line)
        if m and rep_ll is None: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))

# perf counters → derived metrics (shared logic with run_mega_profile.sh).
pm = {}; perf_cmd = None
pp = os.path.join(work, "perf_stat.txt")
if os.path.isfile(pp):
    for line in open(pp, errors="replace"):
        m = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m: perf_cmd = m.group(1); continue
        m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w\.\-:/]+)", line)
        if m and not m.group(1).startswith("<"):
            try: pm[m.group(2)] = int(m.group(1).replace(",", ""))
            except ValueError: pass

def g(*keys):
    for k in keys:
        if pm.get(k) is not None: return pm[k]
    return None

def rate(n, d):
    if not n or not d: return None
    return round(100.0 * n / d, 4)

cyc, ins = g("cycles"), g("instructions")
slots    = g("topdown-total-slots")
issued   = g("topdown-slots-issued")
retired  = g("topdown-slots-retired")
fetchb   = g("topdown-fetch-bubbles")
recovb   = g("topdown-recovery-bubbles")

tma = {}
if slots:
    if retired is not None:
        tma["intel-tma-retiring-pct"]        = round(100.0 * retired / slots, 4)
    if issued is not None and retired is not None and recovb is not None:
        tma["intel-tma-bad-spec-pct"]        = round(100.0 * (issued - retired + recovb) / slots, 4)
    if fetchb is not None:
        tma["intel-tma-frontend-bound-pct"]  = round(100.0 * fetchb / slots, 4)
    if all(k in tma for k in ("intel-tma-retiring-pct","intel-tma-bad-spec-pct","intel-tma-frontend-bound-pct")):
        tma["intel-tma-backend-bound-pct"]   = round(
            max(0.0, 100.0 - tma["intel-tma-retiring-pct"]
                           - tma["intel-tma-bad-spec-pct"]
                           - tma["intel-tma-frontend-bound-pct"]), 4)

metrics = {
  "IPC": round(ins / cyc, 4) if cyc and ins else None,
  "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
  "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
  "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
  "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
  "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
  "iTLB-miss-rate":      rate(g("iTLB-load-misses"), g("iTLB-loads")),
  "frontend-stall-rate": rate(g("stalled-cycles-frontend"), cyc),
  "backend-stall-rate":  rate(g("stalled-cycles-backend"),  cyc),
  **tma,
}
metrics = {k: v for k, v in metrics.items() if v is not None}

# VTune summary if present.
vtune = None
vs = os.path.join(work, "vtune_summary.txt")
if os.path.isfile(vs):
    txt = open(vs, errors="replace").read()
    out = {}
    for k, pat in [
        ("elapsed_time_s",     r"Elapsed Time:\s+([\d.]+)"),
        ("cpu_time_s",         r"CPU Time:\s+([\d.]+)"),
        ("effective_cpu_util", r"Effective CPU Utilization:\s+([\d.]+)%"),
        ("avg_cpu_freq_ghz",   r"Average CPU Frequency:\s+([\d.]+)"),
    ]:
        m = re.search(pat, txt)
        if m:
            try: out[k] = float(m.group(1))
            except ValueError: pass
    if out: vtune = out

verify = []
if rep_ll is not None:
    verify.append({"file": os.path.basename(dpath), "status": "pass",
                   "expected": rep_ll, "reported": rep_ll, "diff": 0.0})

record = {
  "run_id": rid,
  "pbs_id": pbs,
  "run_type": "profile",
  "label": label,
  "description": f"Gadi Sapphire Rapids reproduction of Setonix {ds} @ {thr}T",
  "timing": [{
    "command": f"perf stat ... {ibin} -s {os.path.basename(dpath)} -T {thr} -seed 1",
    "time_s": iqwall if iqwall is not None else wall,
    "memory_kb": 0,
  }],
  "verify": verify,
  "env": {
    "hostname": sh("hostname"),
    "date":     sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc", "0") or 0),
    "gcc":      sh("gcc --version | head -1"),
    "icc":      sh("icc --version 2>/dev/null | head -1"),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "vtune_version": sh("vtune --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d'=' -f2- | tr -d '\"'"),
    "pbs": {
      "job_id":      os.environ.get("PBS_JOBID"),
      "job_name":    os.environ.get("PBS_JOBNAME"),
      "queue":       os.environ.get("PBS_QUEUE"),
      "project":     "${PROJECT}",
      "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
      "submit_host": os.environ.get("PBS_O_HOST"),
      "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
      "scheduler":   "pbs_pro",
    },
  },
  "summary": {
    "pass": 1 if iqrc == 0 else 0,
    "fail": 0 if iqrc == 0 else 1,
    "total_time": iqwall if iqwall is not None else wall,
    "all_pass":   iqrc == 0,
  },
  "profile": {
    "dataset": os.path.basename(dpath),
    "threads": thr,
    "perf_cmd": perf_cmd,
    "metrics": metrics,
    "vtune":   vtune,
  },
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path, "w"), indent=2, default=str)
print(f"[matrix] wrote {out_path}")
PYEOF
WORKER_EOF
chmod +x "${WORKER}"

# ── Plan the sweep.
submitted=0
for dataset in "${!MATRIX[@]}"; do
    [[ -n "${SELECT_DATASET}" && "${SELECT_DATASET}" != "${dataset}" ]] && continue
    read -ra threads <<< "${MATRIX[$dataset]}"
    if [[ ${#SELECT_THREADS[@]} -gt 0 ]]; then
        threads=("${SELECT_THREADS[@]}")
    fi
    for t in "${threads[@]}"; do
        label="${dataset}_${t}t_sr"
        jid=$(qsub -N "iq-${dataset}-${t}t" \
                   -P "${PROJECT}" \
                   -q normalsr \
                   -l "ncpus=104,mem=500GB,walltime=24:00:00,storage=scratch/${PROJECT},wd" \
                   -j oe \
                   -o "${LOGS_DIR}/${label}_\${PBS_JOBID}.log" \
                   -v "DATASET=${dataset},THREADS=${t},LABEL=${label},PROJECT=${PROJECT},REPO_DIR=${REPO_DIR}" \
                   "${WORKER}")
        echo "  → ${dataset} / ${t}T  → ${jid}"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "Submitted ${submitted} job(s). Monitor with: qstat -u \$USER | nqstat"
echo "Logs in:  ${LOGS_DIR}/"
echo "Records: ${RUNS_DIR}/ (one JSON per run, committed to git when ready)"
