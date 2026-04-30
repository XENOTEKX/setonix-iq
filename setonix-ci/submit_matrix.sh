#!/bin/bash
# submit_matrix.sh — fan out run_mega_profile.sh across (dataset × threads).
#
# Use this for the 2026-04-25 non-canonical-file rerun. It:
#
#   1. (optional) submits setonix-ci/generate_datasets.sh and grabs its jobid
#      so every benchmark job is gated on `--dependency=afterok:<gen_jid>`
#      — no run can ever start on a partial / mismatched alignment.
#   2. Verifies sha256 of any already-present alignments on the login node
#      against benchmarks/sha256sums.txt. Aborts (and refuses to submit)
#      if a mismatch is detected and --regen was not passed.
#   3. Submits the matrix:
#        large_modelfinder.fa   {1, 4, 8, 16, 32, 64}
#        xlarge_mf.fa           {1, 4, 8, 16, 32, 64, 128}
#      (mega_dna.fa intentionally excluded — already canonical.)
#
# Usage:
#   ./submit_matrix.sh                          # check + submit, no regen
#   ./submit_matrix.sh --regen                  # regen alignments first, hold matrix on it
#   ./submit_matrix.sh --depend 12345           # gate on an existing gen job id
#   ./submit_matrix.sh --dataset xlarge_mf.fa   # restrict to one dataset
#   ./submit_matrix.sh --threads "1 4 8"        # restrict to a thread subset
#   ./submit_matrix.sh --dry-run                # print sbatch invocations only
#
# All flags are optional and combinable.

set -euo pipefail

# NOTE: Do NOT use BASH_SOURCE[0] / SCRIPT_DIR for path resolution when this
# script is submitted via sbatch.  SLURM copies the script to a temp path
# (/var/spool/slurmd/job<id>/slurm_script) before execution, so a SCRIPT_DIR-
# relative path resolves into the SLURM daemon directory, not the project tree.
# All paths are anchored to PROJECT_DIR instead.
PROJECT_DIR="${PROJECT_DIR:-/scratch/pawsey1351/asamuel/iqtree3}"
SETONIX_CI_DIR="${SETONIX_CI_DIR:-${PROJECT_DIR}/setonix-ci}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
SHA256_LOCKFILE="${SHA256_LOCKFILE:-${PROJECT_DIR}/benchmarks/sha256sums.txt}"
WORKER="${WORKER:-${SETONIX_CI_DIR}/run_mega_profile.sh}"
GENERATOR="${GENERATOR:-${SETONIX_CI_DIR}/generate_datasets.sh}"

REGEN=0
DEPEND_JID=""
DRY_RUN=0
DATASET_FILTER=""
THREAD_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --regen)        REGEN=1; shift ;;
        --depend)       DEPEND_JID="$2"; shift 2 ;;
        --dataset)      DATASET_FILTER="$2"; shift 2 ;;
        --threads)      THREAD_FILTER="$2"; shift 2 ;;
        --dry-run|-n)   DRY_RUN=1; shift ;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

[[ -x "${WORKER}" ]] || { echo "ERROR: ${WORKER} missing or not executable" >&2; exit 1; }

# ── matrix definition ────────────────────────────────────────────────────────
# 2026-04-30 (round 2 audit): added 104T to every Setonix dataset to match
# the Gadi normalsr per-node maximum, so the cross-platform curves overlap
# at every thread point Gadi can run.  128T retained on Setonix only (Gadi
# nodes cap at 104 physical cores).  -mset removed from the worker, so this
# matrix now runs the *full* ModelFinder on every dataset \u2014 same as Gadi.
declare -A MATRIX
MATRIX[large_modelfinder.fa]="1 4 8 16 32 64 104"
MATRIX[xlarge_mf.fa]="1 4 8 16 32 64 104 128"
# mega_dna.fa intentionally excluded \u2014 already canonical (see CHANGELOG 2026-04-25).

# ── (1) optional: submit the regen job first ─────────────────────────────────
if [[ "${REGEN}" -eq 1 ]]; then
    [[ -x "${GENERATOR}" ]] || { echo "ERROR: ${GENERATOR} missing" >&2; exit 1; }
    echo "Submitting generator job (will hold matrix on its afterok)..."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [dry-run] sbatch --parsable ${GENERATOR}"
        DEPEND_JID="<GEN_JID>"
    else
        DEPEND_JID="$(sbatch --parsable "${GENERATOR}")"
        echo "  → generator job ${DEPEND_JID}"
    fi
fi

# ── (2) sha256 sanity check for any already-present alignments ──────────────
if [[ "${REGEN}" -ne 1 && -s "${SHA256_LOCKFILE}" ]]; then
    echo "Verifying sha256 of present alignments against ${SHA256_LOCKFILE}..."
    fail=0
    while read -r expected fname; do
        [[ -z "${expected}" || "${expected}" == \#* ]] && continue
        path="${BENCHMARKS}/${fname}"
        if [[ ! -s "${path}" ]]; then
            echo "  MISSING ${fname}  (re-run with --regen to generate)"
            continue
        fi
        actual="$(sha256sum "${path}" | awk '{print $1}')"
        if [[ "${actual}" == "${expected}" ]]; then
            echo "  OK      ${fname}"
        else
            echo "  FAIL    ${fname}"
            echo "          expected ${expected}"
            echo "          actual   ${actual}"
            fail=1
        fi
    done < "${SHA256_LOCKFILE}"
    if [[ "${fail}" -ne 0 ]]; then
        echo ""
        echo "ERROR: sha256 mismatch on at least one alignment." >&2
        echo "       Re-run with --regen to regenerate, or fix the file manually." >&2
        exit 4
    fi
fi

# ── (3) build the sbatch dependency arg, if any ──────────────────────────────
DEPEND_ARG=()
if [[ -n "${DEPEND_JID}" ]]; then
    DEPEND_ARG+=( "--dependency=afterok:${DEPEND_JID}" )
fi

# ── (4) submit the matrix ────────────────────────────────────────────────────
total=0
for ds in "${!MATRIX[@]}"; do
    if [[ -n "${DATASET_FILTER}" && "${ds}" != "${DATASET_FILTER}" ]]; then
        continue
    fi
    threads="${THREAD_FILTER:-${MATRIX[$ds]}}"
    echo ""
    echo "=== ${ds}  threads: ${threads} ==="
    for t in ${threads}; do
        cmd=( sbatch --parsable
              --job-name="iq-${ds%.fa}-${t}t"
              --export=ALL,DATASET="${ds}",THREADS="${t}"
              "${DEPEND_ARG[@]}"
              "${WORKER}" )
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "  [dry-run] ${cmd[*]}"
        else
            jid="$( "${cmd[@]}" )"
            echo "  → ${ds}  ${t}T  → job ${jid}"
        fi
        total=$((total + 1))
    done
done

echo ""
echo "Submitted ${total} matrix job(s)${DEPEND_JID:+ (held on afterok:${DEPEND_JID})}."
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       ${PROJECT_DIR}/setonix-ci/logs/"
echo "Outputs in:    ${PROJECT_DIR}/setonix-ci/profiles/<dataset>_<T>t_<jobid>/"
