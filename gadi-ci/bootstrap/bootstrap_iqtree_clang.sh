#!/bin/bash
# bootstrap_iqtree_clang.sh — clone + build IQ-TREE 3 on Gadi Sapphire Rapids
# with LLVM/Clang + libomp (the LLVM OpenMP runtime, ABI-compatible with
# Intel libiomp5).
#
# Companion to gadi-ci/bootstrap_iqtree.sh (gcc + libgomp).  Pairs with
# setonix-ci/bootstrap_iqtree_aocc.sh on the Pawsey side.  Together these
# scripts form the cross-platform "Clang/libomp" reference series:
#
#                 gcc / libgomp        Clang / libomp
#   Setonix       smtoff_pin (canon)   clang_omp_pin   ← AOCC 5.1.0
#   Gadi          sr_gcc_pin (canon)   clang_omp_pin   ← LLVM (intel-llvm
#                                                          or system clang)
#
# Why mirror the Setonix variant on Gadi?
#   To keep one of the two confounders constant when interpreting the
#   Setonix gcc-vs-Clang delta.  If both clusters show the same shape under
#   Clang/libomp, the divergence above 8T on Setonix is genuinely an
#   AMD/CCD/libgomp issue.  If only Setonix flips behaviour under Clang,
#   the runtime alone explains the regression.
#
# Output: ${PROJECT_DIR}/build-profiling-clang/iqtree3
#
# Usage (submit):
#   qsub gadi-ci/bootstrap_iqtree_clang.sh
#
#PBS -N iqtree-clang-bootstrap
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
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling-clang}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-master}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 bootstrap on Gadi (LLVM/Clang variant)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Module load: LLVM/Clang + cmake + Eigen + Boost ────────────────────────
# We prefer intel-compiler-llvm (icx is built on top of LLVM/Clang and
# pulls in libiomp5, which is binary-compatible with libomp).  Fall back to
# `llvm` and finally a plain system clang.  Same Eigen/Boost as gcc build
# for parity.
if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6 2>/dev/null || true
    # Try in order: (1) intel-compiler-llvm (icx == clang+libiomp5),
    # (2) llvm, (3) plain clang on the system.
    if module avail intel-compiler-llvm 2>&1 | grep -q intel-compiler-llvm; then
        module load intel-compiler-llvm 2>/dev/null || true
    elif module avail llvm 2>&1 | grep -q '^llvm/'; then
        module load llvm 2>/dev/null || true
    fi
    module load binutils/2.44 2>/dev/null || true
    module load eigen/3.3.7   2>/dev/null || true
    module load boost/1.84.0  2>/dev/null || true
fi

# Pick the Clang front-end.  icx/icpx are LLVM-based compilers that link
# libiomp5; clang/clang++ link libomp.  Both share the LLVM OpenMP API, so
# either is acceptable for the libomp-vs-libgomp comparison.
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
    echo "       Try: module avail | grep -iE 'llvm|clang|intel-compiler-llvm'" >&2
    exit 2
fi
echo "[bootstrap-clang] CC=${CC}"
echo "[bootstrap-clang] CXX=${CXX}"
${CXX} --version | head -3 || true

EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "ERROR: ${SRC_DIR} not found." >&2
    echo "       Clone on a login node first:" >&2
    echo "         git clone ${IQTREE_REPO} ${SRC_DIR}" >&2
    echo "         cd ${SRC_DIR} && git submodule update --init --recursive" >&2
    exit 1
fi

# IPO disabled (parity with gcc build), unittest disabled (no internet on
# compute nodes for googletest FetchContent).
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

# Sapphire Rapids tuning, identical optimisation level to the gcc build.
ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

echo ""
echo "[bootstrap-clang] ── building ${BUILD_PROFILING} ──"
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
echo "[bootstrap-clang] verifying OpenMP runtime linkage..."
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
  "iqtree_commit": "$(cd "${SRC_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF

echo ""
echo "[bootstrap-clang] OK"
echo "  profiling: ${BUILD_PROFILING}/iqtree3"
echo "  metadata:  ${BUILD_PROFILING}/.build-info.json"
