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
IQTREE_REF="${IQTREE_REF:-master}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 bootstrap on Gadi (Sapphire Rapids)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Release build: ${BUILD_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

# Compiler parity with Setonix (2026-04-30 audit): build with gcc on both
# clusters so libgomp / OpenMP barrier semantics, ABI, and codegen are the
# only-vary-by-arch component.  intel-compiler-llvm + intel-vtune are NOT
# loaded here on purpose — vtune is still loaded by the per-job worker for
# profiling (it can attach to a gcc-built binary).
if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6  2>/dev/null || true
    module load gcc/14.2.0    2>/dev/null || true
    # 2026-04-30 (round 2 audit, follow-up #5): Gadi module tree tops out at
    # eigen/3.3.7 and boost/1.84.0 — the 3.4.0 / 1.86.0 entries in the earlier
    # CHANGELOG comment were aspirational and never landed on NCI.  Parity-
    # matrix rows 25/26 carry a minor version delta (eigen 3.3.7 vs 3.4.0,
    # boost 1.84.0 vs 1.86.0) that has no effect on IQ-TREE build output or
    # benchmark results — both are header-only in the paths IQ-TREE uses.
    module load eigen/3.3.7   2>/dev/null || true
    module load boost/1.84.0  2>/dev/null || true
fi
# Eigen3 include dir — set from module env, or fall back to known Gadi path.
EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
# Boost root — set from module env, or fall back to known Gadi path.
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

mkdir -p "$(dirname "${SRC_DIR}")"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    # Compute nodes have no outbound internet — clone must be done on a login
    # node before submitting this job:
    #   git clone https://github.com/iqtree/iqtree3.git \
    #       /scratch/rc29/<user>/iqtree3/src/iqtree3
    #   cd /scratch/rc29/<user>/iqtree3/src/iqtree3
    #   git submodule update --init --recursive
    echo "ERROR: ${SRC_DIR} not found." >&2
    echo "       Clone on a login node first, then resubmit." >&2
    exit 1
else
    echo "[bootstrap] source already present at ${SRC_DIR} — skipping clone"
    cd "${SRC_DIR}"
    git submodule update --init --recursive 2>/dev/null || true
fi

# Submodules are REQUIRED — IQ-TREE3 CMake will fail without cmaple/ and lsd2/.
# Submodules cannot be fetched from compute nodes (no internet), so they must
# already be initialised on the login node before submitting this job.
for sub in cmaple lsd2; do
    if [[ ! -f "${SRC_DIR}/${sub}/CMakeLists.txt" ]]; then
        echo "ERROR: submodule ${SRC_DIR}/${sub} not initialised." >&2
        echo "       On a login node run:" >&2
        echo "         cd ${SRC_DIR} && git submodule update --init --recursive" >&2
        exit 1
    fi
done
echo "[bootstrap] submodules OK: cmaple, lsd2"

# IPO/LTO patch: cmaple's CMakeLists.txt unconditionally enables
# CMAKE_INTERPROCEDURAL_OPTIMIZATION on x86_64 Linux, which makes icx emit
# LLVM IR bitcode into static libraries. The system `ld` on Gadi cannot
# link LLVM IR archives ("File format not recognized"), and `lld` is not
# available. Disable IPO in-place (idempotent — only patches once).
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    echo "[bootstrap] patching ${CMAPLE_CML} to disable IPO (Gadi ld cannot link LLVM IR)"
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable IPO (LTO) by default|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: system ld cannot link LLVM IR bitcode|' "${CMAPLE_CML}"
fi
if ! grep -q 'Gadi: system ld cannot link LLVM IR bitcode' "${CMAPLE_CML}"; then
    echo "ERROR: failed to patch IPO in ${CMAPLE_CML}" >&2
    exit 3
fi
echo "[bootstrap] IPO disabled in cmaple"

# Unittest patch: cmaple/unittest needs GoogleTest via FetchContent (download
# at configure time). We don't run tests in this build, so just skip the
# whole subdirectory. Idempotent.
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    echo "[bootstrap] disabling cmaple/unittest subdirectory (not needed for build)"
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled — avoids GoogleTest FetchContent|' "${CMAPLE_CML}"
fi
echo "[bootstrap] cmaple unittest disabled"

# FetchContent_Declare(googletest URL ...) + FetchContent_MakeAvailable(googletest)
# run unconditionally in cmaple/CMakeLists.txt and require internet (which
# compute nodes do not have). Comment them out — we only need the googletest
# for the unittest subdir which is already disabled above.
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    echo "[bootstrap] disabling cmaple googletest FetchContent calls"
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
fi
echo "[bootstrap] cmaple FetchContent disabled"

# Compiler selection: gcc only (parity with Setonix).
# Was: prefer icx with -xSAPPHIRERAPIDS — switched to gcc 2026-04-30 so
# both clusters share the same OpenMP runtime (libgomp) and codegen.
if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not on PATH (did you 'module load gcc/14.2.0'?)" >&2
    exit 2
fi
CC="$(command -v gcc)"
CXX="$(command -v g++)"
ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids"
echo "[bootstrap] using GCC: $(${CC} --version | head -1)"
echo "[bootstrap] arch flags: ${ARCH_FLAGS}"

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
        -DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}" \
        -DBOOST_ROOT="${BOOST_ROOT}" \
        -DBoost_NO_SYSTEM_PATHS=ON \
        -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${extra}" \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${extra}"
    # IQTREE_BUILD_JOBS overrides nproc — needed on login nodes where cgroups
    # report nproc=1 but we still want parallel compile for verification runs.
    local jobs="${IQTREE_BUILD_JOBS:-$(nproc)}"
    make -j"${jobs}"
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
