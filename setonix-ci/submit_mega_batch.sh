#!/bin/bash
# submit_mega_batch.sh — fan out run_mega_profile.sh across thread counts.
#
# Usage:  ./submit_mega_batch.sh            # submits 16, 32, 64, 128
#         ./submit_mega_batch.sh 32 64      # submits only those two
#
# Each thread count runs as a separate SLURM job so partial failures don't
# lose all data. Jobs use 128 cpus-per-task regardless of --threads because
# IQ-TREE needs the allocation to contain the requested thread count, and we
# size up to 128 which covers 1 full socket (half node).

set -euo pipefail

SCRIPT="/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/run_mega_profile.sh"

if [[ ! -x "${SCRIPT}" ]]; then
    echo "ERROR: ${SCRIPT} missing or not executable" >&2
    exit 1
fi

# Default thread counts if none specified on the CLI.
THREAD_COUNTS=("${@:-16 32 64 128}")

# If the user passed nothing, bash expands "${@:-16 32 64 128}" as one string.
# Re-split that into an array.
if [[ ${#THREAD_COUNTS[@]} -eq 1 && "${THREAD_COUNTS[0]}" == *" "* ]]; then
    read -r -a THREAD_COUNTS <<< "${THREAD_COUNTS[0]}"
fi

echo "Submitting mega profiling jobs for threads: ${THREAD_COUNTS[*]}"
echo ""

for t in "${THREAD_COUNTS[@]}"; do
    jid=$(sbatch --parsable \
                 --job-name="iqtree-mega-${t}t" \
                 --export=ALL,THREADS="${t}" \
                 "${SCRIPT}")
    echo "  → ${t}T  → job ${jid}"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       /scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/"
echo "Outputs in:    /scratch/pawsey1351/asamuel/iqtree3/setonix-ci/profiles/mega_<T>t_<jobid>/"
