#!/bin/bash
# submit_xlarge_r2_alternates.sh — submit the two MPI-placement variants
# of the R2 xlarge_mf benchmark on Gadi normalsr.
#
# These two runs sit alongside the canonical 1-process-×-104-OMP R2 result
# (gadi_xlarge_mf_104t_icx_omp_pin_numa_ft_r2.json, 523.7 s wall) so we can
# attribute any wall-time / IPC / cache-miss delta to the placement scheme
# alone:
#
#   ┌─────────────────────────────┬──────────────┬──────────────────────────────┐
#   │ scheme                      │ ranks × OMP  │ binding                      │
#   ├─────────────────────────────┼──────────────┼──────────────────────────────┤
#   │ canonical (already done)    │ 1 × 104      │ OMP close/cores, numactl     │
#   │ this script: mpi_socket     │ 2 × 52       │ --bind-to socket, numactl    │
#   │ this script: mpi_l3rank     │ 8 × 13       │ rankfile (1 per L3 quadrant) │
#   └─────────────────────────────┴──────────────┴──────────────────────────────┘
#
# Both placement runs share:
#   • same R2-patched source (build-profiling-mpi/iqtree3-mpi)
#   • same icpx + libiomp5 OpenMP runtime
#   • same KMP_BLOCKTIME=200, OMP_PROC_BIND=close, OMP_PLACES=cores
#   • same dataset (xlarge_mf.fa, sha256-gated)
#   • same -seed 1 (per-rank seed becomes 1+rank_id inside IQ-TREE MPI)
#
# Usage:
#   ./submit_xlarge_r2_alternates.sh                    # submit both
#   ./submit_xlarge_r2_alternates.sh socket             # only mpi_socket
#   ./submit_xlarge_r2_alternates.sh l3rank             # only mpi_l3rank
#   ./submit_xlarge_r2_alternates.sh --depend 16XXXXXX  # gate on bootstrap jobid
#   ./submit_xlarge_r2_alternates.sh --bootstrap        # also submit the MPI
#                                                        bootstrap and chain
#                                                        afterok
#   ./submit_xlarge_r2_alternates.sh --dry-run          # print, don't qsub
#
# A typical first-time invocation on a fresh checkout:
#   ./submit_xlarge_r2_alternates.sh --bootstrap

set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
GADI_CI_DIR="${GADI_CI_DIR:-${PROJECT_DIR}/gadi-ci}"
LOGS_DIR="${PROJECT_DIR}/gadi-ci/logs"

BOOTSTRAP_SCRIPT="${GADI_CI_DIR}/bootstrap_iqtree_mpi.sh"
SOCKET_SCRIPT="${GADI_CI_DIR}/run_xlarge_r2_mpi_socket.sh"
L3RANK_SCRIPT="${GADI_CI_DIR}/run_xlarge_r2_mpi_l3rank.sh"
MPI_BUILD_DIR="${PROJECT_DIR}/build-profiling-mpi"

mkdir -p "${LOGS_DIR}"

# ── Argument parsing ──────────────────────────────────────────────────
DRY_RUN=0
DEPEND_JID=""
SUBMIT_BOOTSTRAP=0
DO_SOCKET=1
DO_L3RANK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --depend)        DEPEND_JID="$2"; shift 2 ;;
        --bootstrap)     SUBMIT_BOOTSTRAP=1; shift ;;
        --dry-run|-n)    DRY_RUN=1; shift ;;
        socket)          DO_SOCKET=1; DO_L3RANK=0; shift ;;
        l3rank)          DO_SOCKET=0; DO_L3RANK=1; shift ;;
        both)            DO_SOCKET=1; DO_L3RANK=1; shift ;;
        -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
        *)               echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ── Pre-flight: ensure all referenced scripts exist on scratch ────────
# This script is invoked from a login node where the repo lives at
# ${REPO_DIR}; PBS jobs run from ${PROJECT_DIR}/gadi-ci/ on scratch. The
# operator is expected to rsync ./gadi-ci/ to ${PROJECT_DIR}/gadi-ci/
# before running this.
for s in "${BOOTSTRAP_SCRIPT}" "${SOCKET_SCRIPT}" "${L3RANK_SCRIPT}"; do
    if [[ ! -x "${s}" ]]; then
        echo "ERROR: ${s} missing or not executable." >&2
        echo "       Run the rsync that ships gadi-ci/ to ${PROJECT_DIR}/gadi-ci/ first," >&2
        echo "       or set GADI_CI_DIR to the directory containing the run scripts." >&2
        exit 3
    fi
done

# ── Optional: submit the bootstrap and chain everything afterok on it ──
if [[ "${SUBMIT_BOOTSTRAP}" -eq 1 ]]; then
    if [[ -n "${DEPEND_JID}" ]]; then
        echo "ERROR: --bootstrap and --depend are mutually exclusive." >&2
        exit 4
    fi
    echo "Submitting MPI bootstrap..."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [dry-run] qsub ${BOOTSTRAP_SCRIPT}"
        DEPEND_JID="<bootstrap-jid>"
    else
        DEPEND_JID="$(qsub \
            -P "${PROJECT}" \
            -o "${LOGS_DIR}/iqtree-mpi-bootstrap_%j.log" \
            "${BOOTSTRAP_SCRIPT}")"
        echo "  → bootstrap → ${DEPEND_JID}"
    fi
elif [[ -z "${DEPEND_JID}" ]]; then
    # Without --bootstrap and without --depend, the binary must already exist.
    if [[ ! -x "${MPI_BUILD_DIR}/iqtree3-mpi" ]]; then
        echo "ERROR: ${MPI_BUILD_DIR}/iqtree3-mpi not found." >&2
        echo "       Either:" >&2
        echo "         (a) run --bootstrap to build it now, or" >&2
        echo "         (b) submit gadi-ci/bootstrap_iqtree_mpi.sh manually and" >&2
        echo "             pass the resulting jobid via --depend <jobid>" >&2
        exit 5
    fi
fi

submit_one() {
    # $1=label  $2=script_path
    local label="$1" script="$2"
    local depend_args=()
    if [[ -n "${DEPEND_JID}" ]]; then
        depend_args=(-W "depend=afterok:${DEPEND_JID}")
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [dry-run] qsub -N ${label} ${depend_args[*]} \\"
        echo "             -P ${PROJECT} \\"
        echo "             -v PROJECT=${PROJECT},REPO_DIR=${REPO_DIR} \\"
        echo "             -o ${LOGS_DIR}/${label}_%j.log \\"
        echo "             ${script}"
        return 0
    fi
    local jid
    jid="$(qsub -N "${label}" \
                -P "${PROJECT}" \
                -v "PROJECT=${PROJECT},REPO_DIR=${REPO_DIR}" \
                -o "${LOGS_DIR}/${label}_%j.log" \
                "${depend_args[@]}" \
                "${script}")"
    echo "  → ${label} → ${jid}"
}

# ── Submit the placement variants ─────────────────────────────────────
echo ""
echo "Submitting xlarge_mf R2 alternate placement runs"
echo "  Project:    ${PROJECT}"
echo "  Repo:       ${REPO_DIR}"
echo "  Build dir:  ${MPI_BUILD_DIR}"
[[ -n "${DEPEND_JID}" ]] && echo "  Depend on:  afterok:${DEPEND_JID}"
echo "  Logs in:    ${LOGS_DIR}/"
echo ""

if [[ "${DO_SOCKET}" -eq 1 ]]; then
    submit_one "iq-xlarge-r2-mpi-socket"  "${SOCKET_SCRIPT}"
fi
if [[ "${DO_L3RANK}" -eq 1 ]]; then
    submit_one "iq-xlarge-r2-mpi-l3rank"  "${L3RANK_SCRIPT}"
fi

echo ""
echo "Monitor: qstat -u \$USER  /  nqstat \$USER"
echo "Records: ${REPO_DIR}/logs/runs/gadi_xlarge_mf_*_mpi*_numa_ft_r2.json"
echo "Profile dirs: ${PROJECT_DIR}/gadi-ci/profiles/<label>_<jobid>/"
