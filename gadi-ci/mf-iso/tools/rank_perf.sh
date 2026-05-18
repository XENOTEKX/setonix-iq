#!/bin/bash
# rank_perf.sh — per-MPI-rank perf stat wrapper.
#
# Invoked between rank_probe.sh and numactl in the mpirun command:
#   mpirun ... rank_probe.sh rank_perf.sh numactl --localalloc -- iqtree3-mpi ...
#
# Reads OMPI_COMM_WORLD_RANK to produce per-rank output files:
#   ${PERF_STAT_DIR}/perf_stat_rank_${RANK}.txt
#
# PERF_STAT_DIR must be exported by the job script before mpirun.
# Falls back to running without profiling if perf is unavailable.

set -e

RANK="${OMPI_COMM_WORLD_RANK:-0}"
OUT="${PERF_STAT_DIR:-/tmp}/perf_stat_rank_${RANK}.txt"

# IPC + L3/LLC cache behaviour — key metrics for FCA scaling analysis.
# :u suffix = user-space only (works at perf_event_paranoid=2).
EVENTS="cycles:u,instructions:u,\
cache-references:u,cache-misses:u,\
LLC-loads:u,LLC-load-misses:u,\
L1-dcache-loads:u,L1-dcache-load-misses:u"

if command -v perf >/dev/null 2>&1; then
    exec perf stat -e "${EVENTS}" -o "${OUT}" -- "$@"
else
    echo "RANK-PERF: rank=${RANK} WARNING: perf not available — running without profiling" >&2
    exec "$@"
fi
