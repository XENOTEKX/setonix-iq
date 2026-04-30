#!/bin/bash
# generate_datasets.sh — deterministic benchmark alignments for Setonix.
#
# Generates the CANONICAL benchmark files that must be used on every
# platform so cross-platform comparisons are scientifically valid.
# Files are bit-identical to the Gadi equivalents when built with the
# same IQ-TREE version (3.1.1) and the same fixed seeds.
#
# IMPORTANT: After generation the script verifies sha256 checksums
# against benchmarks/sha256sums.txt (committed to the repo).
# If checksums differ the script exits non-zero and prints a diff —
# do NOT use those files for cross-platform benchmarks.
#
# Output:
#   ${PROJECT_DIR}/benchmarks/
#     ├── large_modelfinder.fa    ( 100 taxa ×  50 000 bp, GTR+G4, seed 101)
#     ├── xlarge_mf.fa            ( 200 taxa × 100 000 bp, GTR+G4, seed 202)
#     └── mega_dna.fa             ( 500 taxa × 100 000 bp, GTR+G4, seed 303)
#
# Seeds / model parameters are identical to gadi-ci/generate_datasets.sh.
# Do NOT change them without also regenerating on Gadi and updating
# benchmarks/sha256sums.txt.
#
# Usage:
#   sbatch setonix-ci/generate_datasets.sh
#
#SBATCH --job-name=iqtree-datagen
#SBATCH --account=pawsey1351
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/datagen_%j.out
#SBATCH --error=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/datagen_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/pawsey1351/asamuel/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"

# Path to the canonical sha256 lockfile.
# NOTE: Do NOT use BASH_SOURCE[0] / SCRIPT_DIR here — SLURM copies the
# script to /var/spool/slurmd/job<id>/slurm_script before execution, so
# a SCRIPT_DIR-relative path resolves into the SLURM daemon directory,
# not the project tree.  Use PROJECT_DIR (hardcoded default above) instead.
SHA256_LOCKFILE="${SHA256_LOCKFILE:-${PROJECT_DIR}/benchmarks/sha256sums.txt}"

mkdir -p "${BENCHMARKS}"

if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found. Build IQ-TREE 3.1.1 first." >&2
    exit 2
fi

# Print IQ-TREE version so it ends up in the job log.
echo "[datagen] IQ-TREE version: $("${IQTREE}" --version 2>&1 | head -1)"

simulate() {
    # $1=label  $2=taxa  $3=length  $4=seed
    local label="$1" taxa="$2" length="$3" seed="$4"
    local out="${BENCHMARKS}/${label}"
    if [[ -s "${out}" ]]; then
        # File present: verify hash before deciding to skip.
        # A non-empty file with a wrong hash must be regenerated — do NOT skip.
        local actual
        actual="$(sha256sum "${out}" | awk '{print $1}')"
        local expected_for_file
        expected_for_file="$(awk -v f="${label}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}" 2>/dev/null || true)"
        if [[ -n "${expected_for_file}" && "${actual}" == "${expected_for_file}" ]]; then
            echo "[datagen] ${label} already present and sha256 matches canonical — skipping"
            return 0
        else
            echo "[datagen] ${label} present but sha256 mismatch — removing and regenerating"
            rm -f "${out}"
        fi
    fi
    echo "[datagen] simulating ${label}: ${taxa} taxa × ${length} bp, seed ${seed}"
    local work="${BENCHMARKS}/_sim_${label%.fa}"
    mkdir -p "${work}"
    ( cd "${work}"
      "${IQTREE}" \
          --alisim "${label%.fa}" \
          -t RANDOM{yh/${taxa}} \
          --seqtype DNA \
          --length "${length}" \
          -m "GTR{1.5,3.0,0.9,1.2,2.7}+F{0.25,0.25,0.25,0.25}+G4{0.8}" \
          --out-format fasta \
          --seed "${seed}" \
          -redo
    )
    if [[ -f "${work}/${label}" ]]; then
        mv -f "${work}/${label}" "${out}"
    else
        local produced
        produced="$(find "${work}" -maxdepth 1 \( -name '*.fa' -o -name '*.phy' \) | head -1)"
        [[ -n "${produced}" ]] && mv -f "${produced}" "${out}"
    fi
    rm -rf "${work}"
    ls -lh "${out}"
}

# ── Generate with the SAME seeds as gadi-ci/generate_datasets.sh ─────────────
simulate "large_modelfinder.fa"  100  50000   101
simulate "xlarge_mf.fa"          200 100000   202
simulate "mega_dna.fa"           500 100000   303

echo ""
echo "[datagen] benchmarks in ${BENCHMARKS}:"
ls -lh "${BENCHMARKS}/"

# ── Verify sha256 against canonical lockfile ──────────────────────────────────
echo ""
echo "[datagen] verifying sha256 checksums against ${SHA256_LOCKFILE} ..."
FAIL=0
while read -r expected_hash filename; do
    # Skip blank lines and comments.
    [[ -z "${expected_hash}" || "${expected_hash}" == \#* ]] && continue
    actual_hash="$(sha256sum "${BENCHMARKS}/${filename}" | awk '{print $1}')"
    if [[ "${actual_hash}" == "${expected_hash}" ]]; then
        echo "  OK  ${filename}"
    else
        echo "  FAIL ${filename}"
        echo "       expected: ${expected_hash}"
        echo "       got:      ${actual_hash}"
        echo "  This file was NOT generated by IQ-TREE 3.1.1 with the canonical"
        echo "  seeds. Cross-platform comparisons using this file are invalid."
        FAIL=1
    fi
done < "${SHA256_LOCKFILE}"

if [[ "${FAIL}" -ne 0 ]]; then
    echo ""
    echo "ERROR: sha256 mismatch(es) detected. The generated files differ from"
    echo "the canonical Gadi benchmarks. Likely cause: different IQ-TREE version."
    echo "Build IQ-TREE 3.1.1 exactly and re-run, or copy the canonical files"
    echo "from Gadi (see benchmarks/sha256sums.txt in the repo)." >&2
    exit 1
fi

echo ""
echo "[datagen] all checksums OK — files match canonical Gadi benchmarks."
