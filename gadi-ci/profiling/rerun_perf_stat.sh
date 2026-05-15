#!/usr/bin/env bash
# gadi-ci/rerun_perf_stat.sh
#
# Submit lightweight PBS jobs that run ONLY 'perf stat' for each completed
# Gadi run that is missing profile.metrics (IPC, cache rates, etc.).
#
# These jobs do NOT redo the full IQ-TREE profiling suite — they run a single
# 'perf stat -e events:u -- iqtree3 ...' command and then patch the existing
# gadi_<label>.json with the resulting metrics using a small Python script.
#
# Usage (from /home/272/as1708/setonix-iq):
#   cd /home/272/as1708/setonix-iq
#   bash gadi-ci/rerun_perf_stat.sh
#
# After all jobs complete, run 'make build' or 'python3.11 tools/build.py'
# to regenerate the dashboard.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${PROJECT:-rc29}"
SCRATCH="/scratch/${PROJECT}/${USER}/iqtree3"
PROFILES_DIR="${SCRATCH}/gadi-ci/profiles"
RUNS_DIR="${REPO_DIR}/logs/runs"
IQTREE="${SCRATCH}/build-profiling/iqtree3"
BENCHMARKS="${SCRATCH}/benchmarks"
LOGS_DIR="${SCRATCH}/gadi-ci/perfstat_reruns"
mkdir -p "${LOGS_DIR}"

# Events with ':u' suffix — user-mode only, compatible with paranoid=2.
PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

# Map label → dataset file and thread count.
declare -A LABEL_DATASET LABEL_THREADS
for jsonf in "${RUNS_DIR}"/gadi_*.json; do
    [[ -f "$jsonf" ]] || continue
    label=$(python3.11 -c "import json,sys; r=json.load(open('$jsonf')); print(r.get('label',''))" 2>/dev/null)
    ipc=$(python3.11 -c "import json,sys; r=json.load(open('$jsonf')); print(r.get('profile',{}).get('metrics',{}).get('IPC') or '')" 2>/dev/null)
    dataset=$(python3.11 -c "import json,sys; r=json.load(open('$jsonf')); print((r.get('profile') or {}).get('dataset',''))" 2>/dev/null)
    threads=$(python3.11 -c "import json,sys; r=json.load(open('$jsonf')); print((r.get('profile') or {}).get('threads',''))" 2>/dev/null)
    if [[ -z "$ipc" && -n "$label" && -n "$dataset" && -n "$threads" ]]; then
        LABEL_DATASET["$label"]="$dataset"
        LABEL_THREADS["$label"]="$threads"
    fi
done

echo "Labels needing perf stat rerun: ${!LABEL_DATASET[*]}"
echo ""

submitted=0
for label in "${!LABEL_DATASET[@]}"; do
    dataset="${LABEL_DATASET[$label]}"
    threads="${LABEL_THREADS[$label]}"
    data_path="${BENCHMARKS}/${dataset}"

    if [[ ! -f "$data_path" ]]; then
        echo "  SKIP ${label} — dataset not found: ${data_path}"
        continue
    fi

    # Estimate walltime: iqtree runs 1T for ~20m, 64T for ~3m; use generous cap.
    if [[ "$threads" -le 1 ]]; then
        walltime="02:00:00"
    elif [[ "$threads" -le 8 ]]; then
        walltime="01:30:00"
    else
        walltime="01:00:00"
    fi

    jid=$(qsub \
        -N "perfstat-${label}" \
        -P "${PROJECT}" \
        -q normalsr \
        -l "ncpus=104,mem=64GB,walltime=${walltime},storage=scratch/${PROJECT},wd" \
        -j oe \
        -o "${LOGS_DIR}/${label}_\${PBS_JOBID}.log" \
        -v "LABEL=${label},DATASET=${dataset},THREADS=${threads},\
DATA_PATH=${data_path},IQTREE=${IQTREE},PROFILES_DIR=${PROFILES_DIR},\
RUNS_DIR=${RUNS_DIR},PERF_EVENTS=${PERF_EVENTS},PROJECT=${PROJECT},\
REPO_DIR=${REPO_DIR}" \
        -- /bin/bash -s << 'WORKER_EOF'
#!/bin/bash
set -euo pipefail
# ── environment ──────────────────────────────────────────────────────────────
module load intel-compiler-llvm/2024.1.0 2>/dev/null || true
module load vtune/2024.1.0 2>/dev/null || true
export PATH="${PATH}:/opt/pbs/default/bin"

PBS_ID_SHORT="${PBS_JOBID%%.*}"

# Find the most recent profile directory for this label (for context).
WORK_DIR=""
if [[ -d "${PROFILES_DIR}" ]]; then
    WORK_DIR=$(ls -dt "${PROFILES_DIR}/${LABEL}_"* 2>/dev/null | head -1 || true)
fi
# Use a fresh subdirectory for the perf-stat-only outputs.
STAT_DIR="${PROFILES_DIR}/perfstat_${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${STAT_DIR}"

echo "[perfstat] label=${LABEL} dataset=${DATASET} threads=${THREADS}"
echo "[perfstat] output dir: ${STAT_DIR}"

# Run IQ-TREE under perf stat.
perf stat -e "${PERF_EVENTS}" \
    -o "${STAT_DIR}/perf_stat.txt" \
    "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed 1 \
                --prefix "${STAT_DIR}/iqtree_perf" \
    > "${STAT_DIR}/iqtree_perf.log" 2>"${STAT_DIR}/perf_stat.err" || RC=$?
RC="${RC:-0}"

if [[ ! -s "${STAT_DIR}/perf_stat.txt" ]]; then
    echo "[perfstat] ERROR: perf_stat.txt is empty or missing. See perf_stat.err:"
    cat "${STAT_DIR}/perf_stat.err" || true
    exit 1
fi
echo "[perfstat] perf_stat.txt produced ($(wc -l < "${STAT_DIR}/perf_stat.txt") lines)"

# Patch the run JSON with the new metrics.
python3 - << PYEOF
import json, re, os, subprocess

stat_dir = "${STAT_DIR}"
runs_dir = "${RUNS_DIR}"
label    = "${LABEL}"

def g_first(*keys):
    for k in keys:
        if pm.get(k) is not None: return pm[k]
    return None

def rate(n, d):
    if not n or not d: return None
    return round(100.0 * n / d, 4)

# Parse perf_stat.txt
pm = {}; perf_cmd = None
pp = os.path.join(stat_dir, "perf_stat.txt")
for line in open(pp, errors="replace"):
    m = re.match(r"Performance counter stats for '(.*)':", line.strip())
    if m: perf_cmd = m.group(1); continue
    m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w.\-:/]+)", line)
    if m and not m.group(1).startswith("<"):
        try:
            key = m.group(2).split(":", 1)[0]  # strip ':u'
            pm[key] = int(m.group(1).replace(",", ""))
        except ValueError: pass

cyc = g_first("cycles")
ins = g_first("instructions")
metrics = {
    "IPC": round(ins / cyc, 4) if cyc and ins else None,
    "cache-miss-rate":     rate(g_first("cache-misses"),         g_first("cache-references")),
    "branch-miss-rate":    rate(g_first("branch-misses"),        g_first("branch-instructions")),
    "L1-dcache-miss-rate": rate(g_first("L1-dcache-load-misses"),g_first("L1-dcache-loads")),
    "LLC-miss-rate":       rate(g_first("LLC-load-misses"),      g_first("LLC-loads")),
    "dTLB-miss-rate":      rate(g_first("dTLB-load-misses"),     g_first("dTLB-loads")),
    "iTLB-miss-rate":      rate(g_first("iTLB-load-misses"),     g_first("iTLB-loads")),
}
# Raw counters
for k in ("cycles","instructions","cache-references","cache-misses",
          "branch-instructions","branch-misses",
          "L1-dcache-loads","L1-dcache-load-misses",
          "LLC-loads","LLC-load-misses",
          "dTLB-loads","dTLB-load-misses",
          "iTLB-loads","iTLB-load-misses"):
    if k in pm:
        metrics[k] = pm[k]
metrics = {k: v for k, v in metrics.items() if v is not None}

print(f"[perfstat-patch] IPC={metrics.get('IPC')} events={len(pm)} keys={list(metrics.keys())[:6]}")

target = os.path.join(runs_dir, f"gadi_{label}.json")
if not os.path.exists(target):
    print(f"[perfstat-patch] WARNING: {target} not found, skipping patch")
else:
    run = json.load(open(target))
    profile = run.get("profile") or {}
    existing = profile.get("metrics") or {}
    existing.update(metrics)
    profile["metrics"] = existing
    if perf_cmd:
        profile["perf_cmd"] = perf_cmd
    # Record artefact path
    arts = profile.get("artefacts") or {}
    arts["perf_stat"] = os.path.join(stat_dir, "perf_stat.txt")
    profile["artefacts"] = arts
    run["profile"] = profile
    with open(target, "w") as f:
        json.dump(run, f, indent=2)
    print(f"[perfstat-patch] patched {target}")
PYEOF
WORKER_EOF
    )
    echo "  → ${label} (T=${threads}) → ${jid}"
    submitted=$((submitted + 1))
done

echo ""
echo "Submitted ${submitted} perf-stat-only job(s)."
echo "After all complete, run: cd ${REPO_DIR} && python3.11 tools/build.py"
