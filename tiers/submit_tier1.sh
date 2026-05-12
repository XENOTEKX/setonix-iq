#!/bin/bash
# submit_tier1.sh — Tier 1: anchor T_1 for R2+NUMA and AVX-512+R2 families.
#
# Submits 7 jobs to remove the [⚠ T_1 extrap.] flags from
# tools/scaling_10M_analysis.py Panel 1.
#
# R2+NUMA Clang anchors: 1T, 4T, 8T, 16T (existing canonical script)
# AVX-512+R2 anchors:    1T, 4T, 8T      (new run_xlarge_avx_omp.sh)
#
# Total budget: ~0.3 KSU (1T runs use rc29/um09; 9h×1cpu×2 + 7h×1cpu×2 = ~32 SU;
# 4T/8T/16T runs add ~220 SU combined).
#
# Usage:   ./tiers/submit_tier1.sh
#          DRY_RUN=1 ./tiers/submit_tier1.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${HERE}/.." && pwd)"
R2_SCRIPT="${REPO_DIR}/gadi-ci/run_xlarge_r2_v312_canonical.sh"
AVX_SCRIPT="${HERE}/run_xlarge_avx_omp.sh"

[[ -x "${R2_SCRIPT}"  ]] || { echo "ERROR: ${R2_SCRIPT} not found/executable" >&2; exit 1; }
[[ -x "${AVX_SCRIPT}" ]] || chmod +x "${AVX_SCRIPT}"

run_qsub () {
    local desc="$1"; shift
    local cmd=( qsub -P um09 -o "${REPO_DIR}/logs/jobs/tiers" -l place=excl "$@" )
    echo "→ ${desc}"
    echo "    ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then return; fi
    "${cmd[@]}"
}

echo "════ Tier 1: R2+NUMA + AVX-512 anchor runs (7 jobs) ════"

# R2+NUMA Clang — existing canonical v3.1.2 script, parameterized by THREADS.
# Wall times from observed Pass-1 runs (168116041-168116047); script runs two
# passes (clean timing + perf stat), so walltime = 2×Pass1 + ~10% overhead.
#   1T observed Pass-1: killed at 14412s (>4h) → 09:00:00
#   4T observed Pass-1: 4951s → 03:30:00
#   8T observed Pass-1: 3113s → 02:30:00
#  16T observed Pass-1: 1900s → 01:30:00
run_qsub "R2+NUMA Clang 1T  (~14400s pass1, 1 cpu)" \
    -v THREADS=1  -l ncpus=1  -l mem=16GB  -l walltime=09:00:00 \
    -N iq-r2-1t   "${R2_SCRIPT}"

run_qsub "R2+NUMA Clang 4T  (~4951s pass1, 4 cpu)" \
    -v THREADS=4  -l ncpus=4  -l mem=32GB  -l walltime=03:30:00 \
    -N iq-r2-4t   "${R2_SCRIPT}"

run_qsub "R2+NUMA Clang 8T  (~3113s pass1, 8 cpu)" \
    -v THREADS=8  -l ncpus=8  -l mem=48GB  -l walltime=02:30:00 \
    -N iq-r2-8t   "${R2_SCRIPT}"

run_qsub "R2+NUMA Clang 16T (~1900s pass1, 16 cpu)" \
    -v THREADS=16 -l ncpus=16 -l mem=64GB  -l walltime=01:30:00 \
    -N iq-r2-16t  "${R2_SCRIPT}"

# AVX-512+R2 (um09 MPI binary, mpirun -np 1 OMP-only mode)
# Single-pass script (no perf stat); observed: 4T=4746s, 8T=2963s;
# 1T estimated ~19000s (4× 4T) → 07:00:00 (already-good 4T/8T unchanged).
run_qsub "AVX-512+R2 1T  (~19000s estimated, 1 cpu)" \
    -v THREADS=1 -l ncpus=1 -l mem=16GB -l walltime=07:00:00 \
    -N iq-avx-1t "${AVX_SCRIPT}"

run_qsub "AVX-512+R2 4T  (~4746s observed, 4 cpu)" \
    -v THREADS=4 -l ncpus=4 -l mem=32GB -l walltime=02:00:00 \
    -N iq-avx-4t "${AVX_SCRIPT}"

run_qsub "AVX-512+R2 8T  (~2963s observed, 8 cpu)" \
    -v THREADS=8 -l ncpus=8 -l mem=48GB -l walltime=01:30:00 \
    -N iq-avx-8t "${AVX_SCRIPT}"

echo "════ Tier 1 submission done ════"
echo "Monitor with: qstat -u \$USER"
