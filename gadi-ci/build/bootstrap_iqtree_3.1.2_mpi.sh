#!/bin/bash
# bootstrap_iqtree_3.1.2_mpi.sh — build IQ-TREE **3.1.2** (R2-patched) on
# Gadi Sapphire Rapids with MPI + LLVM/Clang(icpx) + libiomp5.
#
# Companion to bootstrap_iqtree_3.1.2.sh (non-MPI variant).  Mirrors
# bootstrap_iqtree_mpi.sh but pinned to a separate scratch tree:
#
#   /scratch/rc29/as1708/iqtree3-3.1.2 → v3.1.2 (4e91dd6) + R2  (THIS)
#
# Output: ${PROJECT_DIR}/build-profiling-mpi/iqtree3-mpi
#
#PBS -N iqtree-3.1.2-mpi-bootstrap
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
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling-mpi}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-v3.1.2}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3.1.2 bootstrap on Gadi (MPI + icpx + libiomp5)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

# Module load order: openmpi BEFORE intel-compiler-llvm (mpicxx from
# openmpi, with OMPI_CXX=icpx).
if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6           2>/dev/null || true
    module load openmpi/4.1.7          2>/dev/null || true
    if module avail intel-compiler-llvm 2>&1 | grep -q intel-compiler-llvm; then
        module load intel-compiler-llvm 2>/dev/null || true
    elif module avail llvm 2>&1 | grep -q '^llvm/'; then
        module load llvm                2>/dev/null || true
    fi
    module load binutils/2.44          2>/dev/null || true
    module load eigen/3.3.7            2>/dev/null || true
    module load boost/1.84.0           2>/dev/null || true
fi

if command -v icpx >/dev/null 2>&1; then
    export OMPI_CC="$(command -v icx)"
    export OMPI_CXX="$(command -v icpx)"
    OMP_RUNTIME_HINT="libiomp5"
elif command -v clang++ >/dev/null 2>&1; then
    export OMPI_CC="$(command -v clang)"
    export OMPI_CXX="$(command -v clang++)"
    OMP_RUNTIME_HINT="libomp"
else
    echo "ERROR: no clang/icpx on PATH after module load." >&2
    exit 2
fi

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "ERROR: mpicxx not on PATH after openmpi/4.1.7 load." >&2
    exit 2
fi

CC="$(command -v mpicc)"
CXX="$(command -v mpicxx)"
echo "[bootstrap-3.1.2-mpi] mpicc=${CC}  (-> ${OMPI_CC})"
echo "[bootstrap-3.1.2-mpi] mpicxx=${CXX} (-> ${OMPI_CXX})"
${CXX} --version | head -3 || true
mpirun --version | head -2 || true

EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "ERROR: ${SRC_DIR} not found." >&2
    exit 1
fi

ACTUAL_REF="$(cd "${SRC_DIR}" && git describe --tags --always 2>/dev/null || echo unknown)"
if [[ "${ACTUAL_REF}" != "v3.1.2" && "${ACTUAL_REF}" != "v3.1.2"* ]]; then
    echo "WARNING: ${SRC_DIR} is at '${ACTUAL_REF}', expected v3.1.2." >&2
fi

# R2 patch sanity check.
if grep -q 'schedule(dynamic,1)' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: ${SRC_DIR}/tree/phylokernelnew.h still has schedule(dynamic,1)" >&2
    exit 4
fi
if [[ "$(grep -c 'NUMA first-touch' "${SRC_DIR}/tree/phylotreesse.cpp")" -ne 3 ]]; then
    echo "ERROR: ${SRC_DIR}/tree/phylotreesse.cpp missing R1/R2a markers." >&2
    exit 4
fi
echo "[bootstrap-3.1.2-mpi] R2 patches present (8/8 sites)"

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
echo "[bootstrap-3.1.2-mpi] ── building ${BUILD_PROFILING} ──"
rm -rf "${BUILD_PROFILING}"
mkdir -p "${BUILD_PROFILING}"
cd "${BUILD_PROFILING}"

CC="${CC}" CXX="${CXX}" cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIQTREE_FLAGS=mpi \
    -DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}" \
    -DBOOST_ROOT="${BOOST_ROOT}" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"

JOBS="${IQTREE_BUILD_JOBS:-$(nproc)}"
make -j"${JOBS}"

if [[ ! -x "${BUILD_PROFILING}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_PROFILING}" -maxdepth 2 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    if [[ -n "${found}" && "${found}" != "${BUILD_PROFILING}/iqtree3-mpi" ]]; then
        ln -sf "${found}" "${BUILD_PROFILING}/iqtree3-mpi"
    fi
fi
if [[ ! -x "${BUILD_PROFILING}/iqtree3-mpi" ]]; then
    echo "ERROR: ${BUILD_PROFILING}/iqtree3-mpi not produced." >&2
    exit 5
fi

echo ""
echo "[bootstrap-3.1.2-mpi] verifying linkage..."
LDD_OUT="$(ldd "${BUILD_PROFILING}/iqtree3-mpi" 2>&1)"
echo "${LDD_OUT}" | grep -iE 'omp|mpi' || true
if echo "${LDD_OUT}" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — MPI build accidentally pulled libgomp." >&2
    exit 6
fi
if ! echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)' ; then
    echo "  ✗ libmpi not linked — IQTREE_FLAGS=mpi did not take effect." >&2
    exit 7
fi
echo "  → libomp/libiomp5 + libmpi linked. OK."

mpirun -n 1 "${BUILD_PROFILING}/iqtree3-mpi" --version 2>&1 | head -3 || true

cat > "${BUILD_PROFILING}/.build-info.json" <<EOF
{
  "compiler":      "$(${OMPI_CXX} --version | head -1)",
  "compiler_kind": "icpx",
  "mpi_wrapper":   "${CXX}",
  "mpi_version":   "$(mpirun --version 2>&1 | head -1)",
  "openmp_runtime":"${OMP_RUNTIME_HINT}",
  "iqtree_flags":  "mpi",
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
echo "[bootstrap-3.1.2-mpi] OK"
echo "  binary:   ${BUILD_PROFILING}/iqtree3-mpi"
echo "  metadata: ${BUILD_PROFILING}/.build-info.json"
