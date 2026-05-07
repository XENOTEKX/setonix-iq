#!/bin/bash
# submit_clang_bbblock.sh — targeted AOCC/libomp rerun of xlarge_mf at {8, 64}T
# with Deva Kumar Deeptimahanti's recommended `-m block:block:block` SLURM
# distribution (2026-05-07 feedback).
#
# Motivation:
#   Previous clang_omp_pin runs used the SLURM default task distribution
#   (cyclic). Deva's recommendation:
#     #SBATCH -m block:block:block
#     srun -c $OMP_NUM_THREADS -m block:block:block ...
#   ensures threads are packed into contiguous cores within a socket, giving
#   the best L3 cache utilisation on AMD Milan (Setonix EPYC 7763 Zen3).
#
# Thread counts:
#   8T  — single-socket, single-CCD baseline (multiples-of-8 per Deva's docs)
#   64T — cross-CCD, single-socket ceiling; the onset of the libgomp regression
#         sits here; canonical comparison point against GCC smtoff_pin series.
#
# Differences from clang_omp_pin series:
#   LABEL_SUFFIX  → clang_bbblock   (new label; files don't collide)
#   srun flag     → -m block:block:block   (via updated run_mega_profile.sh)
#   #SBATCH -m    → block:block:block      (via updated run_mega_profile.sh)
#
# Unchanged from clang_omp_pin:
#   BUILD_DIR, compiler (AOCC 5.1.0/libomp), arch flags (-O3 -march=znver3)
#   OMP_PROC_BIND=close, OMP_PLACES=cores, KMP_BLOCKTIME=200
#   Full ModelFinder (no -mset restriction)
#   sha256 gate against benchmarks/sha256sums.txt
#
# Usage:
#   ./submit_clang_bbblock.sh               # submit 8T + 64T
#   ./submit_clang_bbblock.sh --threads "8 16 32 64"   # custom subset
#   ./submit_clang_bbblock.sh --depend 12345            # gate on build job
#   ./submit_clang_bbblock.sh --dry-run                 # print, don't submit

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/pawsey1351/asamuel/iqtree3}"
SETONIX_CI_DIR="${SETONIX_CI_DIR:-${PROJECT_DIR}/setonix-ci}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
SHA256_LOCKFILE="${SHA256_LOCKFILE:-${PROJECT_DIR}/benchmarks/sha256sums.txt}"
WORKER="${WORKER:-${SETONIX_CI_DIR}/run_mega_profile.sh}"
AOCC_BUILD_DIR="${AOCC_BUILD_DIR:-${PROJECT_DIR}/build-profiling-aocc}"

DATASET="xlarge_mf.fa"
# Multiples of 8 as per Pawsey docs; 8T = single-CCD baseline, 64T = first
# full-socket, cross-CCD comparison point against the GCC regression.
DEFAULT_THREADS="8 64"

DRY_RUN=0
THREAD_FILTER=""
DEPEND_JID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)    THREAD_FILTER="$2"; shift 2 ;;
        --depend)     DEPEND_JID="$2"; shift 2 ;;
        --dry-run|-n) DRY_RUN=1; shift ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

[[ -x "${WORKER}" ]] || { echo "ERROR: ${WORKER} missing or not executable" >&2; exit 1; }

# ── Pre-flight: AOCC binary must exist OR be queued via --depend ──────────────
if [[ -z "${DEPEND_JID}" ]]; then
    if [[ ! -x "${AOCC_BUILD_DIR}/iqtree3" ]]; then
        echo "ERROR: ${AOCC_BUILD_DIR}/iqtree3 not found." >&2
        echo "       Either:" >&2
        echo "         (a) run bootstrap_iqtree_aocc.sh first, or" >&2
        echo "         (b) submit it and pass its jobid via --depend <jid>" >&2
        exit 3
    fi

    # The whole point of the Clang build is to use libomp, not libgomp.
    # Abort early if the binary inadvertently links libgomp.
    if ldd "${AOCC_BUILD_DIR}/iqtree3" 2>/dev/null | grep -q 'libgomp'; then
        echo "ERROR: ${AOCC_BUILD_DIR}/iqtree3 links libgomp, not libomp." >&2
        echo "       Re-run bootstrap_iqtree_aocc.sh — AOCC module may have" >&2
        echo "       failed to load and fallen back to gcc." >&2
        exit 4
    fi
    echo "[bbblock] AOCC binary OK (links libomp)."
else
    echo "[bbblock] queueing matrix on afterok:${DEPEND_JID}"
    echo "[bbblock] (ldd check will be performed at job start inside ${WORKER})"
fi

# ── Pre-flight: dataset sha256 gate ──────────────────────────────────────────
if [[ -s "${SHA256_LOCKFILE}" ]]; then
    expected="$(awk -v f="${DATASET}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}")"
    if [[ -n "${expected}" && -s "${BENCHMARKS}/${DATASET}" ]]; then
        actual="$(sha256sum "${BENCHMARKS}/${DATASET}" | awk '{print $1}')"
        if [[ "${actual}" != "${expected}" ]]; then
            echo "ERROR: ${DATASET} sha256 mismatch — refusing to submit." >&2
            echo "       expected: ${expected}" >&2
            echo "       actual:   ${actual}"   >&2
            exit 5
        fi
        echo "[bbblock] ${DATASET} sha256 OK (canonical)."
    fi
fi

# ── Submit the matrix ─────────────────────────────────────────────────────────
threads="${THREAD_FILTER:-${DEFAULT_THREADS}}"
DEPEND_ARG=()
if [[ -n "${DEPEND_JID}" ]]; then
    DEPEND_ARG+=( "--dependency=afterok:${DEPEND_JID}" )
fi

echo ""
echo "=== ${DATASET} (clang_bbblock)  threads: ${threads} ==="
total=0
for t in ${threads}; do
    cmd=( sbatch --parsable
          --job-name="iq-bbblock-${DATASET%.fa}-${t}t"
          --export=ALL,DATASET="${DATASET}",THREADS="${t}",BUILD_DIR="${AOCC_BUILD_DIR}",LABEL_SUFFIX="clang_bbblock",OMP_RUNTIME_TAG="libomp",KMP_BLOCKTIME="200"
          "${DEPEND_ARG[@]}"
          "${WORKER}" )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [dry-run] ${cmd[*]}"
    else
        jid="$( "${cmd[@]}" )"
        echo "  → ${DATASET}  ${t}T  → job ${jid}"
    fi
    total=$((total + 1))
done

echo ""
echo "Submitted ${total} job(s) (build=${AOCC_BUILD_DIR}, label=clang_bbblock)."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       ${PROJECT_DIR}/setonix-ci/logs/"
echo "Outputs in:    ${PROJECT_DIR}/setonix-ci/profiles/${DATASET%.fa}_<T>t_clang_bbblock_<jobid>/"
echo ""
echo "After harvest, compare clang_bbblock vs clang_omp_pin at 8T and 64T"
echo "to quantify the -m block:block:block scheduling improvement."
