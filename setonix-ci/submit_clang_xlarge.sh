#!/bin/bash
# submit_clang_xlarge.sh — fan out the AOCC/libomp variant across xlarge_mf
# threads {8, 16, 32, 64, 104, 128}.
#
# Reuses the canonical run_mega_profile.sh worker — every observable axis
# (perf events, OpenMP env, NUMA policy, sha256 gate, srun options) stays
# identical to the gcc/_smtoff_pin canonical sweep. The only deltas are:
#
#   BUILD_DIR     → ${PROJECT_DIR}/build-profiling-aocc   (clang+libomp binary)
#   LABEL_SUFFIX  → clang_omp_pin                          (so files don't
#                                                            collide with
#                                                            _smtoff_pin)
#   KMP_BLOCKTIME → 200                                    (libomp default,
#                                                            matches Gadi's
#                                                            libiomp5 default;
#                                                            we leave PASSIVE
#                                                            on as a no-op for
#                                                            libomp)
#
# Threads {8, 16, 32, 64, 104, 128}: chosen to cover the regression onset
# (16T = first cross-CCD step on EPYC 7763) up through the 128-logical
# Setonix node. 1T and 4T are skipped because the Clang-vs-libgomp story
# only materialises once threads cross the CCD boundary (>8T on Zen 3); a
# wall-time-only chart is acceptable for this comparison series.
#
# Tag: build_tag="clang_omp_pin", non_canonical=true.  These runs sit
# alongside the gcc canonical xlarge series as a comparison reference.
#
# Usage:
#   ./submit_clang_xlarge.sh                            # submit full sweep
#   ./submit_clang_xlarge.sh --threads "16 32"          # subset
#   ./submit_clang_xlarge.sh --depend 12345             # gate on existing build job
#   ./submit_clang_xlarge.sh --dry-run                  # print, don't submit

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/pawsey1351/asamuel/iqtree3}"
SETONIX_CI_DIR="${SETONIX_CI_DIR:-${PROJECT_DIR}/setonix-ci}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
SHA256_LOCKFILE="${SHA256_LOCKFILE:-${PROJECT_DIR}/benchmarks/sha256sums.txt}"
WORKER="${WORKER:-${SETONIX_CI_DIR}/run_mega_profile.sh}"
AOCC_BUILD_DIR="${AOCC_BUILD_DIR:-${PROJECT_DIR}/build-profiling-aocc}"

DATASET="xlarge_mf.fa"
DEFAULT_THREADS="8 16 32 64 104 128"

DRY_RUN=0
THREAD_FILTER=""
DEPEND_JID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)      THREAD_FILTER="$2"; shift 2 ;;
        --depend)       DEPEND_JID="$2"; shift 2 ;;
        --dry-run|-n)   DRY_RUN=1; shift ;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

[[ -x "${WORKER}" ]] || { echo "ERROR: ${WORKER} missing or not executable" >&2; exit 1; }

# ── Pre-flight: AOCC binary must exist OR be queued via --depend ───────────
if [[ -z "${DEPEND_JID}" ]]; then
    if [[ ! -x "${AOCC_BUILD_DIR}/iqtree3" ]]; then
        echo "ERROR: ${AOCC_BUILD_DIR}/iqtree3 not found." >&2
        echo "       Either:" >&2
        echo "         (a) run bootstrap_iqtree_aocc.sh first, or" >&2
        echo "         (b) submit it and pass its jobid via --depend <jid>" >&2
        exit 3
    fi

    # Verify the AOCC binary actually links libomp (not libgomp). The whole
    # point of this sweep is to swap the OpenMP runtime; if ldd shows libgomp
    # the experiment is invalid.
    if ldd "${AOCC_BUILD_DIR}/iqtree3" 2>/dev/null | grep -q 'libgomp'; then
        echo "ERROR: ${AOCC_BUILD_DIR}/iqtree3 links libgomp, not libomp." >&2
        echo "       Re-run bootstrap_iqtree_aocc.sh — the AOCC module load" >&2
        echo "       may have failed and fallen back to gcc." >&2
        exit 4
    fi
else
    echo "[clang-xlarge] queuing matrix on afterok:${DEPEND_JID}"
    echo "[clang-xlarge] (binary will be ldd-checked at sweep job start)"
fi

# ── Pre-flight: dataset sha256 gate ─────────────────────────────────────────
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
        echo "[clang-xlarge] ${DATASET} sha256 OK (canonical)."
    fi
fi

# ── Submit the matrix ───────────────────────────────────────────────────────
threads="${THREAD_FILTER:-${DEFAULT_THREADS}}"
DEPEND_ARG=()
if [[ -n "${DEPEND_JID}" ]]; then
    DEPEND_ARG+=( "--dependency=afterok:${DEPEND_JID}" )
fi
echo ""
echo "=== ${DATASET} (clang_omp_pin)  threads: ${threads} ==="
total=0
for t in ${threads}; do
    cmd=( sbatch --parsable
          --job-name="iq-clang-${DATASET%.fa}-${t}t"
          --export=ALL,DATASET="${DATASET}",THREADS="${t}",BUILD_DIR="${AOCC_BUILD_DIR}",LABEL_SUFFIX="clang_omp_pin",OMP_RUNTIME_TAG="libomp",KMP_BLOCKTIME="200"
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
echo "Submitted ${total} matrix job(s) (build=${AOCC_BUILD_DIR}, label_suffix=clang_omp_pin)."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       ${PROJECT_DIR}/setonix-ci/logs/"
echo "Outputs in:    ${PROJECT_DIR}/setonix-ci/profiles/${DATASET%.fa}_<T>t_clang_omp_pin_<jobid>/"
