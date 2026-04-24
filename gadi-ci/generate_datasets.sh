#!/bin/bash
# generate_datasets.sh — deterministic benchmark alignments via AliSim.
#
# The original Setonix benchmarks (large_modelfinder.fa, xlarge_mf.fa,
# mega_dna.fa) live on Setonix /scratch and cannot be rsynced cross-site
# from a Gadi login node. We regenerate equivalent workloads here using
# IQ-TREE 3's built-in AliSim simulator with fixed seeds so the output
# is bit-identical across invocations.
#
# Output:
#   ${PROJECT_DIR}/benchmarks/
#     ├── turtle.fa               (copied from upstream iqtree3 examples)
#     ├── large_modelfinder.fa    ( 500 taxa ×   5 000 bp, GTR+G4)
#     ├── xlarge_mf.fa            (1000 taxa ×  10 000 bp, GTR+G4)
#     └── mega_dna.fa             ( 500 taxa × 100 000 bp, GTR+G4)
#
# Usage (submit):
#   qsub gadi-ci/generate_datasets.sh
#
#PBS -N iqtree-datagen
#PBS -P rc29
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=01:00:00
#PBS -l wd
#PBS -l storage=scratch/rc29
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3}"
BENCHMARKS="${BENCHMARKS:-${PROJECT_DIR}/benchmarks}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/src/iqtree3}"

mkdir -p "${BENCHMARKS}"

if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found. Run gadi-ci/bootstrap_iqtree.sh first." >&2
    exit 2
fi

# turtle.fa + example.phy come from the iqtree3 example_data directory.
for f in turtle.fa example.phy; do
    for candidate in \
        "${SRC_DIR}/example_data/${f}" \
        "${SRC_DIR}/example/${f}" \
        "${SRC_DIR}/test_scripts/test_data/${f}" \
        "${SRC_DIR}/${f}"; do
        if [[ -f "${candidate}" ]]; then
            cp -f "${candidate}" "${BENCHMARKS}/${f}"
            echo "[datagen] copied ${candidate}"
            break
        fi
    done
done

simulate() {
    # $1=label  $2=taxa  $3=length  $4=seed
    local label="$1" taxa="$2" length="$3" seed="$4"
    local out="${BENCHMARKS}/${label}"
    if [[ -s "${out}" ]]; then
        echo "[datagen] ${label} already present — skipping"
        return 0
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
    # AliSim writes <prefix>.fa in the working dir.
    if [[ -f "${work}/${label}" ]]; then
        mv -f "${work}/${label}" "${out}"
    else
        local produced
        produced="$(find "${work}" -maxdepth 1 -name '*.fa' -o -name '*.phy' | head -1)"
        [[ -n "${produced}" ]] && mv -f "${produced}" "${out}"
    fi
    rm -rf "${work}"
    ls -lh "${out}"
}

simulate "large_modelfinder.fa"  500   5000   101
simulate "xlarge_mf.fa"          1000 10000   202
simulate "mega_dna.fa"            500 100000  303

echo ""
echo "[datagen] benchmarks in ${BENCHMARKS}:"
ls -lh "${BENCHMARKS}/" || true
