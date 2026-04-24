#!/bin/bash
# run_pipeline.sh — Gadi port of the Setonix IQ-TREE CI pipeline.
#
# Runs a small deterministic test pipeline (ModelFinder + tree search on
# turtle.fa and a few small alignments), captures timing + verification,
# and emits logs/runs/<YYYY-MM-DD_HHMMSS>.json matching run.schema.json.
#
# Designed to be runnable *directly on a Gadi login node* for tiny data,
# or submitted via PBS for larger datasets.
#
# Usage (login node):
#   ./run_pipeline.sh
#
# Usage (PBS, heavier datasets):
#   qsub -P rc29 -q normalsr -l ncpus=104,mem=500GB,walltime=1:00:00,wd \
#        -l storage=scratch/rc29 run_pipeline.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3}"
TEST_DATA="${TEST_DATA:-${PROJECT_DIR}/benchmarks}"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${RUNS_DIR}"

RUN_ID="$(date +%Y-%m-%d_%H%M%S)"
PBS_ID="${PBS_JOBID:-}"
PBS_ID="${PBS_ID%%.*}"
OUT_JSON="${RUNS_DIR}/${RUN_ID}.json"
WORK_TMP="$(mktemp -d)"
trap 'rm -rf "${WORK_TMP}"' EXIT

echo "Gadi IQ-TREE pipeline — run_id=${RUN_ID} pbs=${PBS_ID:-none}"

# Test matrix: each entry is "file expected_lnL command-args"
# Expected lnL values are calibrated against a reference run; update as needed.
declare -a TESTS=(
  "turtle.fa        -5681.1  -s turtle.fa -m GTR+G4 -seed 1"
  "example.phy      auto     -s example.phy -seed 1"
)

timing_json="["
verify_json="["
pass=0; fail=0; total_time=0

first=1
for t in "${TESTS[@]}"; do
    file="$(echo "${t}" | awk '{print $1}')"
    expected="$(echo "${t}" | awk '{print $2}')"
    args="$(echo "${t}" | cut -d' ' -f3-)"
    input="${TEST_DATA}/${file}"

    if [[ ! -f "${input}" ]]; then
        echo "  [SKIP] ${file} (not found at ${input})"
        continue
    fi

    prefix="${WORK_TMP}/${file}"
    cmd="${IQTREE} ${args//${file}/${input}} --prefix ${prefix}"
    echo "  [RUN] ${cmd}"

    start=$(date +%s.%N)
    eval "${cmd}" > "${prefix}.stdout" 2>&1 || true
    end=$(date +%s.%N)
    elapsed=$(python3 -c "print(f'{${end}-${start}:.3f}')")

    reported=""
    if [[ -f "${prefix}.iqtree" ]]; then
        reported=$(grep -E "^Log-likelihood of the tree:" "${prefix}.iqtree" \
                   | head -1 | awk '{print $NF}' || true)
    fi

    status="fail"
    diff="0"
    if [[ "${expected}" == "auto" ]]; then
        [[ -n "${reported}" ]] && status="pass"
        diff="0"
    elif [[ -n "${reported}" ]]; then
        diff=$(python3 -c "print(abs(${reported} - (${expected})))" 2>/dev/null || echo "999")
        # 1e-3 tolerance
        ok=$(python3 -c "print(int(${diff} < 1e-3))" 2>/dev/null || echo 0)
        [[ "${ok}" == "1" ]] && status="pass" || status="fail"
    fi

    if [[ "${status}" == "pass" ]]; then pass=$((pass+1)); else fail=$((fail+1)); fi
    total_time=$(python3 -c "print(${total_time} + ${elapsed})")

    sep=","; [[ ${first} -eq 1 ]] && sep=""; first=0
    timing_json="${timing_json}${sep}{\"command\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${cmd}"),\"time_s\":${elapsed}}"
    verify_json="${verify_json}${sep}{\"file\":\"${file}\",\"status\":\"${status}\",\"expected\":$([[ \"${expected}\" == \"auto\" ]] && echo 0 || echo ${expected}),\"reported\":${reported:-0},\"diff\":${diff}}"
done

timing_json="${timing_json}]"
verify_json="${verify_json}]"

all_pass="true"; [[ ${fail} -gt 0 ]] && all_pass="false"

cat > "${OUT_JSON}" <<JSON
{
  "run_id": "${RUN_ID}",
  "pbs_id": $([[ -n "${PBS_ID}" ]] && echo "\"${PBS_ID}\"" || echo "null"),
  "run_type": "pipeline",
  "label": "gadi_pipeline_${RUN_ID}",
  "timing": ${timing_json},
  "verify": ${verify_json},
  "env": {
    "date": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "cpu": "$(lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs)",
    "cores": $(nproc),
    "gcc": "$(gcc --version 2>/dev/null | head -1 | awk '{print $NF}')",
    "kernel": "$(uname -r)",
    "os": "$(grep PRETTY_NAME= /etc/os-release | cut -d'=' -f2- | tr -d '\"')",
    "iqtree_binary": "${IQTREE}",
    "iqtree_version": "$(${IQTREE} --version 2>&1 | head -1 || echo unknown)",
    "pbs": {
      "job_id": "${PBS_JOBID:-}",
      "job_name": "${PBS_JOBNAME:-}",
      "queue": "${PBS_QUEUE:-}",
      "project": "${PROJECT:-}",
      "ncpus": "${PBS_NCPUS:-}",
      "nodefile": "${PBS_NODEFILE:-}",
      "submit_host": "${PBS_O_HOST:-}",
      "submit_dir": "${PBS_O_WORKDIR:-}",
      "scheduler": "pbs_pro"
    }
  },
  "summary": {
    "pass": ${pass},
    "fail": ${fail},
    "total_time": ${total_time},
    "all_pass": ${all_pass}
  }
}
JSON

echo ""
echo "Wrote ${OUT_JSON}  (pass=${pass} fail=${fail} total=${total_time}s)"
