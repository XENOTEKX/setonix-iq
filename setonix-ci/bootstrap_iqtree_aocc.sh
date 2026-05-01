#!/bin/bash
# bootstrap_iqtree_aocc.sh — clone + build IQ-TREE 3 on Setonix with AOCC (Clang).
#
# Companion to bootstrap_iqtree.sh (gcc/libgomp).  Builds the *same* source
# tree with AOCC 5.1.0 (AMD's tuned LLVM/Clang) so that the resulting binary
# links libomp (LLVM/Intel-OpenMP-API runtime) instead of libgomp.
#
#   gcc build  → libgomp   (Pawsey default — used by canonical _smtoff_pin runs)
#   AOCC build → libomp    (LLVM runtime — closer to Intel OpenMP semantics)
#
# Why a second build?
#   Minh (IQ-TREE author) noted that libgomp scaling regressions above the
#   8-thread CCD boundary on Setonix (EPYC 7763 Zen 3) are not seen on Gadi
#   (Sapphire Rapids, libiomp5 via Intel OneAPI). AOCC ships libomp by default,
#   so a Clang build on Setonix isolates the OpenMP-runtime variable while
#   holding source, optimisation flags, and Zen 3 tuning constant.
#
# Output:
#   ${PROJECT_DIR}/build-profiling-aocc/iqtree3
#
# Usage (sbatch):
#   sbatch setonix-ci/bootstrap_iqtree_aocc.sh
#
# Usage (interactive, e.g. inside `salloc -N1 -p work --exclusive ...`):
#   bash setonix-ci/bootstrap_iqtree_aocc.sh
#
#SBATCH --job-name=iqtree-aocc-bootstrap
#SBATCH --account=pawsey1351
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/bootstrap_aocc_%j.out
#SBATCH --error=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/bootstrap_aocc_%j.err

set -euo pipefail

PROJECT="${PROJECT:-pawsey1351}"
USER_ID="${USER:-$(whoami)}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/src/iqtree3}"
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling-aocc}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-master}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 bootstrap on Setonix (AOCC / Clang variant)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Module load: AOCC + cmake + Eigen + Boost ────────────────────────────────
# AOCC 5.1.0 ships clang 17 with Zen 3/4-tuned codegen and libomp/libomptarget
# (LLVM OpenMP runtime, same API surface as Intel libiomp5 on Gadi).
# Eigen / Boost — same versions as the gcc bootstrap so the only delta is
# the compiler + OpenMP runtime.
if command -v module >/dev/null 2>&1; then
    module load aocc/5.1.0                2>/dev/null || true
    module load cmake/3.30.5              2>/dev/null || true
    module load eigen/3.4.0               2>/dev/null || true
    module load boost/1.86.0-c++14-python 2>/dev/null || true
fi

if ! command -v clang >/dev/null 2>&1; then
    echo "ERROR: clang not on PATH after 'module load aocc/5.1.0'." >&2
    echo "       Available AOCC versions: 'module avail aocc'." >&2
    exit 2
fi
CC="$(command -v clang)"
CXX="$(command -v clang++)"
echo "[bootstrap-aocc] using AOCC: $(${CC} --version | head -1)"

# Eigen / Boost include paths (same defaults as the gcc bootstrap).
EIGEN_BASE="${EIGEN_ROOT:-${PAWSEY_EIGEN_HOME:-}}"
EIGEN3_INCLUDE_DIR="${EIGEN_BASE:+${EIGEN_BASE}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/software/setonix/2025.08/software/linux-sles15-zen3/gcc-14.2.0/eigen-3.4.0/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-${PAWSEY_BOOST_HOME:-/software/setonix/2025.08/software/linux-sles15-zen3/gcc-14.2.0/boost-1.86.0}}"

mkdir -p "$(dirname "${SRC_DIR}")"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    if command -v git >/dev/null 2>&1; then
        echo "[bootstrap-aocc] cloning ${IQTREE_REPO} (${IQTREE_REF})"
        git clone --branch "${IQTREE_REF}" "${IQTREE_REPO}" "${SRC_DIR}"
        ( cd "${SRC_DIR}" && git submodule update --init --recursive )
    else
        echo "ERROR: ${SRC_DIR} not found and git not available." >&2
        exit 1
    fi
else
    echo "[bootstrap-aocc] source already present at ${SRC_DIR} — skipping clone"
    ( cd "${SRC_DIR}" && git submodule update --init --recursive 2>/dev/null || true )
fi

for sub in cmaple lsd2; do
    if [[ ! -f "${SRC_DIR}/${sub}/CMakeLists.txt" ]]; then
        echo "ERROR: submodule ${SRC_DIR}/${sub} not initialised." >&2
        exit 1
    fi
done

# IPO disabled (matches gcc build for parity, same rationale as bootstrap_iqtree.sh).
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable IPO (LTO) by default|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Setonix: disabled for parity with Gadi|' "${CMAPLE_CML}"
fi
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Setonix: disabled|' "${CMAPLE_CML}"
fi
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# SETONIX-DISABLED: |' "${CMAPLE_CML}"
fi

# Compiler flags — Zen 3 tuning, identical to gcc build for fair comparison.
# -fopenmp on Clang/AOCC pulls in libomp at link time.
ARCH_FLAGS="-O3 -march=znver3 -mtune=znver3 -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

echo ""
echo "[bootstrap-aocc] ── building ${BUILD_PROFILING} ──"
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

# Confirm the binary links libomp, not libgomp — that is the whole point.
echo ""
echo "[bootstrap-aocc] verifying OpenMP runtime linkage..."
if ldd "${BUILD_PROFILING}/iqtree3" | grep -qE 'libomp\.so|libiomp5'; then
    echo "  → libomp linked (LLVM/Intel-OpenMP-API runtime). OK."
elif ldd "${BUILD_PROFILING}/iqtree3" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — AOCC build accidentally pulled libgomp." >&2
    ldd "${BUILD_PROFILING}/iqtree3" | grep -i 'omp\|gomp' >&2
    exit 3
else
    echo "  ! no OpenMP runtime found in ldd output:"
    ldd "${BUILD_PROFILING}/iqtree3" | grep -i 'omp\|gomp' || true
fi

"${BUILD_PROFILING}/iqtree3" --version | head -3 || true

cat > "${BUILD_PROFILING}/.build-info.json" <<EOF
{
  "compiler":      "$(${CC} --version | head -1)",
  "compiler_kind": "aocc-clang",
  "openmp_runtime":"libomp",
  "arch_flags":    "${ARCH_FLAGS} ${EXTRA}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)",
  "iqtree_repo":   "${IQTREE_REPO}",
  "iqtree_ref":    "${IQTREE_REF}",
  "iqtree_commit": "$(cd "${SRC_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF

echo ""
echo "[bootstrap-aocc] OK"
echo "  profiling: ${BUILD_PROFILING}/iqtree3"
echo "  metadata:  ${BUILD_PROFILING}/.build-info.json"
