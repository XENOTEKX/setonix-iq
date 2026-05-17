#!/bin/bash
# submit_all.sh — submit Tiers 1, 2, and 3 (16 jobs total).
#
# All jobs run independently in the queue — no PBS dependencies are used
# because each job writes a unique file under logs/runs/.  After all jobs
# complete, regenerate the chart:
#
#   python3.11 tools/scaling_10M_analysis.py
#
# Usage:   ./tiers/submit_all.sh
#          DRY_RUN=1 ./tiers/submit_all.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${HERE}/.." && pwd)"

mkdir -p "${REPO_DIR}/logs/jobs/tiers"

"${HERE}/submit_tier1.sh"
echo ""
"${HERE}/submit_tier2.sh"
echo ""
"${HERE}/submit_tier3.sh"

echo ""
echo "════ ALL TIERS SUBMITTED (16 jobs) ════"
echo "Track with: qstat -u \$USER"
echo "After completion: python3.11 tools/scaling_10M_analysis.py"
