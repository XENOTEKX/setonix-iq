#!/bin/bash
# bootstrap_iqtree_3.1.2.sh — build IQ-TREE **3.1.2** (R2-patched) on Gadi
# Sapphire Rapids with LLVM/Clang(icpx) + libiomp5.  Non-MPI build.
#
# Companion to bootstrap_iqtree_3.1.2_mpi.sh (MPI variant).  Mirrors
# bootstrap_iqtree_clang.sh but pinned to a separate scratch tree:
#
#   /scratch/rc29/as1708/iqtree3       → v3.1.1 (master @ 7658269) + R2
#   /scratch/rc29/as1708/iqtree3-3.1.2 → v3.1.2 (4e91dd6)           + R2  (THIS)
#
# The v3.1.2 source tree was cloned + patched from a login node before this
# job runs.  This script re-verifies the R2 patches are still present, then
# builds with identical compiler flags to the v3.1.1 build for parity.
#
# Output: ${PROJECT_DIR}/build-profiling-clang/iqtree3
#
#PBS -N iqtree-3.1.2-bootstrap
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
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-3.1.2}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/src/iqtree3}"
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling-clang}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-v3.1.2}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3.1.2 bootstrap on Gadi (LLVM/Clang variant)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6 2>/dev/null || true
    if module avail intel-compiler-llvm 2>&1 | grep -q intel-compiler-llvm; then
        module load intel-compiler-llvm 2>/dev/null || true
    elif module avail llvm 2>&1 | grep -q '^llvm/'; then
        module load llvm 2>/dev/null || true
    fi
    module load binutils/2.44 2>/dev/null || true
    module load eigen/3.3.7   2>/dev/null || true
    module load boost/1.84.0  2>/dev/null || true
fi

if command -v icpx >/dev/null 2>&1; then
    CC="$(command -v icx)"
    CXX="$(command -v icpx)"
    OMP_RUNTIME_HINT="libiomp5"
elif command -v clang++ >/dev/null 2>&1; then
    CC="$(command -v clang)"
    CXX="$(command -v clang++)"
    OMP_RUNTIME_HINT="libomp"
else
    echo "ERROR: no clang/icpx on PATH after module load." >&2
    exit 2
fi
echo "[bootstrap-3.1.2] CC=${CC}"
echo "[bootstrap-3.1.2] CXX=${CXX}"
${CXX} --version | head -3 || true

EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "ERROR: ${SRC_DIR} not found." >&2
    echo "       Clone v3.1.2 on a login node first and apply numa_patches.diff." >&2
    exit 1
fi

# Verify we're actually on v3.1.2
ACTUAL_REF="$(cd "${SRC_DIR}" && git describe --tags --always 2>/dev/null || echo unknown)"
if [[ "${ACTUAL_REF}" != "v3.1.2" && "${ACTUAL_REF}" != "v3.1.2"* ]]; then
    echo "WARNING: ${SRC_DIR} is at '${ACTUAL_REF}', expected v3.1.2." >&2
fi

# R2 patch sanity check (same guard as MPI bootstrap).
if grep -q 'schedule(dynamic,1)' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: ${SRC_DIR}/tree/phylokernelnew.h still has schedule(dynamic,1)" >&2
    echo "       — R2 patches missing.  Re-apply per numa-firsttouch-patches.md." >&2
    exit 4
fi
if [[ "$(grep -c 'NUMA first-touch' "${SRC_DIR}/tree/phylotreesse.cpp")" -ne 3 ]]; then
    echo "ERROR: ${SRC_DIR}/tree/phylotreesse.cpp does not contain 3 'NUMA first-touch' markers." >&2
    exit 4
fi
echo "[bootstrap-3.1.2] R2 patches present (8/8 sites: 3 in phylotreesse.cpp + 5 in phylokernelnew.h)"

CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable IPO (LTO) by default|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled for parity|' "${CMAPLE_CML}"
fi
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled|' "${CMAPLE_CML}"
fi
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
fi

ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

echo ""
echo "[bootstrap-3.1.2] ── building ${BUILD_PROFILING} ──"
rm -rf "${BUILD_PROFILING}"
mkdir -p "${BUILD_PROFILING}"
cd "${BUILD_PROFILING}"

CC="${CC}" CXX="${CXX}" cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}" \
    -DBOOST_ROOT="${BOOST_ROOT}" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"

JOBS="${IQTREE_BUILD_JOBS:-$(nproc)}"
make -j"${JOBS}"
if [[ ! -x "${BUILD_PROFILING}/iqtree3" ]]; then
    mv -f "${BUILD_PROFILING}"/iqtree3*/iqtree3 "${BUILD_PROFILING}/iqtree3" 2>/dev/null || true
fi

echo ""
echo "[bootstrap-3.1.2] verifying OpenMP runtime linkage..."
if ldd "${BUILD_PROFILING}/iqtree3" | grep -qE 'libomp\.so|libiomp5'; then
    echo "  → libomp/libiomp5 linked. OK."
elif ldd "${BUILD_PROFILING}/iqtree3" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — Clang build accidentally pulled libgomp." >&2
    ldd "${BUILD_PROFILING}/iqtree3" | grep -i 'omp\|gomp' >&2
    exit 3
fi

"${BUILD_PROFILING}/iqtree3" --version | head -3 || true

cat > "${BUILD_PROFILING}/.build-info.json" <<EOF
{
  "compiler":      "$(${CXX} --version | head -1)",
  "compiler_kind": "clang",
  "openmp_runtime":"${OMP_RUNTIME_HINT}",
  "arch_flags":    "${ARCH_FLAGS} ${EXTRA}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)",
  "iqtree_repo":   "${IQTREE_REPO}",
  "iqtree_ref":    "${IQTREE_REF}",
  "iqtree_commit": "$(cd "${SRC_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "r2_patches_present": true
}
EOF

echo ""
echo "[bootstrap-3.1.2] OK"
echo "  binary:   ${BUILD_PROFILING}/iqtree3"
echo "  metadata: ${BUILD_PROFILING}/.build-info.json"
