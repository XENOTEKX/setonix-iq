#!/bin/bash
# submit_mega_batch.sh — fan out run_mega_profile.sh across thread counts on Gadi.
#
# Usage:  ./submit_mega_batch.sh            # submits 4, 8, 16, 24, 48
#         ./submit_mega_batch.sh 16 48      # submits only those two
#
# Each thread count runs as a separate PBS job so partial failures don't
# lose all data. The normal queue on Gadi is Cascade Lake (Xeon 8268),
# 48 cores/node, 192 GB/node. We request a full node's worth of cpus+mem
# regardless of THREADS so the inner IQ-TREE OpenMP pool can spin up to
# the requested count without sharing with other users.

set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-as1708}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
SCRIPT="${PROJECT_DIR}/gadi-ci/run_mega_profile.sh"
LOGS_DIR="${PROJECT_DIR}/gadi-ci/logs"

mkdir -p "${LOGS_DIR}"

if [[ ! -x "${SCRIPT}" ]]; then
    echo "ERROR: ${SCRIPT} missing or not executable" >&2
    echo "Hint: rsync ./gadi-ci/ to ${PROJECT_DIR}/gadi-ci/ first" >&2
    exit 1
fi

THREAD_COUNTS=("${@:-4 8 16 24 48}")
if [[ ${#THREAD_COUNTS[@]} -eq 1 && "${THREAD_COUNTS[0]}" == *" "* ]]; then
    read -r -a THREAD_COUNTS <<< "${THREAD_COUNTS[0]}"
fi

echo "Submitting Gadi mega profiling jobs for threads: ${THREAD_COUNTS[*]}"
echo "  Project:    ${PROJECT}"
echo "  Script:     ${SCRIPT}"
echo "  Logs dir:   ${LOGS_DIR}"
echo ""

for t in "${THREAD_COUNTS[@]}"; do
    jid=$(qsub -N "iqtree-mega-${t}t" \
               -v "THREADS=${t},PROJECT=${PROJECT}" \
               -P "${PROJECT}" \
               -q normal \
               -l "ncpus=48,mem=190GB,walltime=24:00:00,storage=scratch/${PROJECT},wd" \
               -j oe \
               -o "${LOGS_DIR}/mega_${t}t_\${PBS_JOBID}.log" \
               "${SCRIPT}")
    echo "  → ${t}T  → job ${jid}"
done

echo ""
echo "Monitor with:  qstat -u \$USER"
echo "               nqstat \$USER"
echo "Logs in:       ${LOGS_DIR}/"
echo "Outputs in:    ${PROJECT_DIR}/gadi-ci/profiles/mega_<T>t_<jobid>/"
