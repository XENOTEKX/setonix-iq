#!/bin/bash
# run_profiling.sh — lightweight perf-stat wrapper for Gadi.
#
# Profiles a single IQ-TREE invocation on a small/medium dataset with a
# subset of the mega-profile event list. Intended for quick iteration on
# a login node or an interactive PBS session (qsub -I -l ncpus=...).
#
# Usage:  ./run_profiling.sh <dataset> <threads> [model]

set -euo pipefail

DATASET="${1:-turtle.fa}"
THREADS="${2:-1}"
MODEL="${3:-GTR+G4}"

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling}"
IQTREE="${BUILD_DIR}/iqtree3"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/gadi-ci/profiles}"

[[ -f "${DATASET}" ]] || DATASET="${PROJECT_DIR}/test_scripts/test_data/${DATASET}"

RUN_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID%%.*}"
LABEL="$(basename "${DATASET}" .fa)_${THREADS}t"
WORK_DIR="${OUT_ROOT}/${LABEL}_${RUN_ID}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

PERF_EVENTS="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
stalled-cycles-frontend,stalled-cycles-backend,\
topdown-total-slots,topdown-slots-retired,topdown-fetch-bubbles"

echo "Gadi profiling: ${DATASET} × ${THREADS}T (${MODEL})"
echo "  work dir: ${WORK_DIR}"

perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
    "${IQTREE}" -s "${DATASET}" -T "${THREADS}" -m "${MODEL}" -seed 1 \
    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1 || true

echo "Done. See ${WORK_DIR}/perf_stat.txt"
