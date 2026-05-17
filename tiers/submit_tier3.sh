#!/bin/bash
# submit_tier3.sh — Tier 3: MF-only MF2 scaling series + MF2 2-node mid-point.
#
# Adds an MF-only Amdahl baseline (7 jobs) so the MF2 dispatch ◆ point (PBS
# 168000131, 416T/59s) has a same-binary, same-protocol Amdahl curve to sit
# against.  The worker script uses the um09 build-mpi-mf2/iqtree3-mpi binary
# (AVX-512 + R2 patch + LPT fix) at np=1, seed=1, -te fixed_xlarge_tree.nwk.
# Also adds the MF2 2-rank 2-node mid-point to fill the dispatch scaling
# curve between 1-node and 4-node.
#
# 7 MF-only MF2 runs + 1 MF2 2-node dispatch run = 8 jobs.
#
# Usage:   ./tiers/submit_tier3.sh
#          DRY_RUN=1 ./tiers/submit_tier3.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${HERE}/.." && pwd)"
MF_SCRIPT="${HERE}/run_xlarge_mf_audit.sh"
MF2_SCRIPT="${REPO_DIR}/gadi-ci/run_xlarge_r2_mf2_dispatch.sh"

[[ -x "${MF_SCRIPT}"  ]] || chmod +x "${MF_SCRIPT}"
[[ -x "${MF2_SCRIPT}" ]] || { echo "ERROR: ${MF2_SCRIPT} not found/executable" >&2; exit 1; }

run_qsub () {
    local desc="$1"; shift
    local cmd=( qsub -o "${REPO_DIR}/logs/jobs/tiers" "$@" )
    echo "→ ${desc}"
    echo "    ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then return; fi
    "${cmd[@]}"
}

echo "════ Tier 3: MF-only MF2 scaling + MF2 2-node (8 jobs) ════"

# MF-only MF2 scaling — single-rank (np=1), seed=1, fixed tree.
# Binary: um09/build-mpi-mf2/iqtree3-mpi (AVX-512 + R2 + LPT fix).
# At 104T evaluateAll on this dataset measured ~62-68s (single-rank).
# Amdahl with f≈0.01 gives T_1 ~3200s; 4h walltime is the safety cap.
run_qsub "MF-only MF2 1T   (~1-3h wall, 1 cpu)" \
    -v THREADS=1,SEED=1   -l ncpus=1   -l mem=16GB -l walltime=04:00:00 -l place=excl \
    -N iq-mf-1t   "${MF_SCRIPT}"

run_qsub "MF-only MF2 4T   (~20-40m, 4 cpu)" \
    -v THREADS=4,SEED=1   -l ncpus=4   -l mem=32GB -l walltime=01:30:00 -l place=excl \
    -N iq-mf-4t   "${MF_SCRIPT}"

run_qsub "MF-only MF2 8T   (~10-20m, 8 cpu)" \
    -v THREADS=8,SEED=1   -l ncpus=8   -l mem=48GB -l walltime=01:00:00 -l place=excl \
    -N iq-mf-8t   "${MF_SCRIPT}"

run_qsub "MF-only MF2 16T  (~6-12m, 16 cpu)" \
    -v THREADS=16,SEED=1  -l ncpus=16  -l mem=64GB -l walltime=00:30:00 -l place=excl \
    -N iq-mf-16t  "${MF_SCRIPT}"

run_qsub "MF-only MF2 32T  (~3-6m, 32 cpu)" \
    -v THREADS=32,SEED=1  -l ncpus=32  -l mem=96GB -l walltime=00:30:00 \
    -N iq-mf-32t  "${MF_SCRIPT}"

run_qsub "MF-only MF2 64T  (~2-4m, 64 cpu)" \
    -v THREADS=64,SEED=1  -l ncpus=64  -l mem=128GB -l walltime=00:30:00 \
    -N iq-mf-64t  "${MF_SCRIPT}"

run_qsub "MF-only MF2 104T (~1-2m, 104 cpu)" \
    -v THREADS=104,SEED=1 -l ncpus=104 -l mem=200GB -l walltime=00:30:00 \
    -N iq-mf-104t "${MF_SCRIPT}"

# MF2 dispatch 2-node (np=2, 1 rank per node, 104T per rank), seed=1.
# Existing dispatch script uses NRANKS=4,SEED=42 by default — override both.
# Expected wall: between 1-node 62.5s and 4-node 59s; likely ~70-90s.
run_qsub "MF2 dispatch 2-node 208T (~2m wall, 208 cpu)" \
    -v NRANKS=2,OMP_PER_RANK=104,SEED=1 \
    -l ncpus=208 -l mem=400GB -l walltime=00:30:00 \
    -N iq-mf2-2nd "${MF2_SCRIPT}"

echo "════ Tier 3 submission done ════"
echo "Monitor with: qstat -u \$USER"
