#!/bin/bash
# bootstrap_iqtree.sh — clone + build IQ-TREE 3 on Gadi Sapphire Rapids.
#
# Submits as a small PBS job on normalsr (or runs directly if invoked in an
# already-allocated session). Produces two binaries:
#   ${PROJECT_DIR}/build/iqtree3              (release, -xSAPPHIRERAPIDS)
#   ${PROJECT_DIR}/build-profiling/iqtree3    (+ -fno-omit-frame-pointer -g)
#
# The -fno-omit-frame-pointer build is required for perf -g to unwind stacks
# cleanly (otherwise ~72 % of samples land on [unknown] frames).
#
# Source repo: https://github.com/iqtree/iqtree3.git
#
# Usage (submit):
#   qsub gadi-ci/bootstrap_iqtree.sh
#
# Usage (direct, e.g. inside `qsub -I -q normalsr -l ncpus=104,mem=500GB`):
#   bash gadi-ci/bootstrap_iqtree.sh
#
#PBS -N iqtree-bootstrap
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
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/src/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-main}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 bootstrap on Gadi (Sapphire Rapids)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Release build: ${BUILD_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6            2>/dev/null || true
    module load intel-compiler-llvm/2024.2.0 2>/dev/null || true
    module load gcc/14.2.0              2>/dev/null || true
fi

mkdir -p "$(dirname "${SRC_DIR}")"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "[bootstrap] cloning ${IQTREE_REPO} → ${SRC_DIR}"
    git clone --recursive "${IQTREE_REPO}" "${SRC_DIR}"
    cd "${SRC_DIR}"
    git checkout "${IQTREE_REF}"
else
    echo "[bootstrap] updating ${SRC_DIR}"
    cd "${SRC_DIR}"
    git fetch origin
    git checkout "${IQTREE_REF}"
    git pull --ff-only origin "${IQTREE_REF}" || true
    git submodule update --init --recursive
fi

# Compiler selection: prefer icx (Intel LLVM) for -xSAPPHIRERAPIDS. Fall back
# to gcc with -march=sapphirerapids if icx is not on PATH.
if command -v icx >/dev/null 2>&1; then
    CC="$(command -v icx)"
    CXX="$(command -v icpx)"
    ARCH_FLAGS="-O3 -xSAPPHIRERAPIDS"
    echo "[bootstrap] using Intel LLVM: $(${CC} --version | head -1)"
elif command -v gcc >/dev/null 2>&1; then
    CC="$(command -v gcc)"
    CXX="$(command -v g++)"
    ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids"
    echo "[bootstrap] falling back to GCC: $(${CC} --version | head -1)"
else
    echo "ERROR: no icx or gcc on PATH" >&2
    exit 2
fi

build_variant() {
    local build_dir="$1"
    local extra="$2"
    echo ""
    echo "[bootstrap] ── building ${build_dir} (extra='${extra}') ──"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"
    CC="${CC}" CXX="${CXX}" cmake "${SRC_DIR}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${extra}" \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${extra}"
    make -j"$(nproc)"
    if [[ ! -x "${build_dir}/iqtree3" ]]; then
        # Some IQ-TREE CMake configs put the binary in a subdir.
        mv -f "${build_dir}"/iqtree3*/iqtree3 "${build_dir}/iqtree3" 2>/dev/null || true
    fi
    "${build_dir}/iqtree3" --version | head -3 || true
}

build_variant "${BUILD_DIR}"       ""
build_variant "${BUILD_PROFILING}" "-fno-omit-frame-pointer -g"

echo ""
echo "[bootstrap] OK"
echo "  release:   ${BUILD_DIR}/iqtree3"
echo "  profiling: ${BUILD_PROFILING}/iqtree3"
