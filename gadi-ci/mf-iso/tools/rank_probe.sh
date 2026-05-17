#!/bin/bash
# rank_probe.sh — pre-exec wrapper that captures per-rank binding info.
#
# Invoked as the first argument to mpirun:
#   mpirun ... rank_probe.sh /path/to/iqtree3-mpi -s ALIGN ...
#
# Each rank prints one block of RANK-PROBE: lines, then exec's iqtree.
# The output is captured by mpirun --output-filename so we have per-rank
# binding evidence — the missing piece in earlier debug runs where ranks
# 1+ stdout was lost.

set -e

RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-?}}"
NRANKS="${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-?}}"
LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-?}"
HOST="$(hostname)"

# Print everything as a single block so per-rank files are easy to parse.
{
    echo "RANK-PROBE: ===== rank ${RANK}/${NRANKS} (local_rank=${LOCAL_RANK}) on ${HOST} ====="
    echo "RANK-PROBE: rank=${RANK} pid=$$"
    echo "RANK-PROBE: rank=${RANK} cwd=$(pwd)"
    echo "RANK-PROBE: rank=${RANK} date=$(date -Iseconds)"
    echo "RANK-PROBE: rank=${RANK} hostname=${HOST}"

    # NUMA / CPU binding.
    if command -v numactl >/dev/null 2>&1; then
        # numactl --show prints policy + cpubind + membind + nodebind on stderr/stdout.
        numactl --show 2>&1 | sed "s|^|RANK-PROBE: rank=${RANK} numactl_show: |"
    fi

    if [[ -r /proc/self/status ]]; then
        cpus="$(grep 'Cpus_allowed_list:' /proc/self/status | awk '{print $2}')"
        mems="$(grep 'Mems_allowed_list:' /proc/self/status | awk '{print $2}')"
        echo "RANK-PROBE: rank=${RANK} cpus_allowed_list=${cpus}"
        echo "RANK-PROBE: rank=${RANK} mems_allowed_list=${mems}"
    fi

    # OpenMP settings as seen by this rank.
    echo "RANK-PROBE: rank=${RANK} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
    echo "RANK-PROBE: rank=${RANK} OMP_PROC_BIND=${OMP_PROC_BIND:-unset}"
    echo "RANK-PROBE: rank=${RANK} OMP_PLACES=${OMP_PLACES:-unset}"
    echo "RANK-PROBE: rank=${RANK} KMP_BLOCKTIME=${KMP_BLOCKTIME:-unset}"
    echo "RANK-PROBE: rank=${RANK} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-unset}"

    # Memory headroom at start.
    if [[ -r /proc/meminfo ]]; then
        memtot="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
        memavail="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
        echo "RANK-PROBE: rank=${RANK} mem_total_kb=${memtot} mem_available_kb=${memavail}"
    fi

    echo "RANK-PROBE: ===== end rank ${RANK} probe; exec'ing $* ====="
} >&2

# Exec the real binary — keeps stdout clean for IQ-TREE's own output.
exec "$@"
