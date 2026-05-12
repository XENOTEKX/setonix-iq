#!/bin/bash
# submit_tier2.sh — Tier 2: complete GCC Canonical series (1 job).
#
# Adds the missing GCC 104T NUMA-penalty point.  Reuses the existing GCC
# benchmark matrix worker from gadi-ci/submit_benchmark_matrix.sh.
#
# Expected wall: ~20–30 min (interpolating from GCC 64T=1638s + ICX 104T=1112s).
# KSU: 104 cpu × 1h × 2.0 = 208 SU = 0.208 KSU.
#
# Usage:   ./tiers/submit_tier2.sh
#          DRY_RUN=1 ./tiers/submit_tier2.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${HERE}/.." && pwd)"

# The submit_benchmark_matrix.sh emits a worker script _run_matrix_job.sh under
# the project gadi-ci/ directory and then qsubs it.  For a single targeted run
# we go through that pipeline so we get exactly the same JSON schema and
# build_tag ("_sr_gcc_pin") as the existing 1T..64T points.
MATRIX_SCRIPT="${REPO_DIR}/gadi-ci/submit_benchmark_matrix.sh"
[[ -x "${MATRIX_SCRIPT}" ]] || { echo "ERROR: ${MATRIX_SCRIPT} not found/executable" >&2; exit 1; }

echo "════ Tier 2: GCC Canonical 104T (1 job) ════"
echo "→ GCC 104T (~20–30m wall, 104 cpu)"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    ${MATRIX_SCRIPT} xlarge_mf 104"
else
    "${MATRIX_SCRIPT}" xlarge_mf 104
fi

echo "════ Tier 2 submission done ════"
echo "Monitor with: qstat -u \$USER"
