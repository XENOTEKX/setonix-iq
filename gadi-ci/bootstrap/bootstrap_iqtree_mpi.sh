#!/bin/bash
# bootstrap_iqtree_mpi.sh — clone + build IQ-TREE 3 on Gadi Sapphire Rapids
# with **MPI + LLVM/Clang(icpx) + libiomp5**.
#
# Companion to:
#   gadi-ci/bootstrap_iqtree.sh        — gcc + libgomp        (no MPI)
#   gadi-ci/bootstrap_iqtree_clang.sh  — icpx + libiomp5      (no MPI)
#   gadi-ci/bootstrap_iqtree_mpi.sh    — icpx + libiomp5 + MPI  (THIS FILE)
#
# Why an MPI build?
#   The R2 NUMA-first-touch sweep (logs/runs/gadi_xlarge_mf_*_icx_omp_pin_numa_ft_r2.json)
#   was run as a single process × 104 OpenMP threads, with the OpenMP pool
#   spilling across both sockets and relying on first-touch + static scheduling
#   to keep DRAM accesses local. We now want to compare two finer-grained
#   placements that constrain the OpenMP pool to a smaller-than-node region:
#
#     • 2×MPI × 52 OMP  (one rank per socket)  — cross-socket traffic becomes
#                                                explicit MPI messages instead
#                                                of cache-coherence traffic.
#     • 8×MPI × 13 OMP  (one rank per L3/SNC4) — each OpenMP pool fits in a
#                                                single L3 quadrant; no shared-
#                                                cache interference between
#                                                ranks.
#
#   Both schemes require an MPI-enabled iqtree3-mpi binary built from the same
#   R2-patched source tree as build-profiling-clang, so wall-time / lnL deltas
#   are attributable to the placement scheme and not to source or compiler
#   differences.
#
# Build provenance
#   Source tree: ${SRC_DIR}                        (already R2-patched)
#   Compiler:    icpx via mpicxx (OMPI_CXX=icpx)   — same OpenMP runtime
#                                                    (libiomp5) as the
#                                                    non-MPI clang build.
#   MPI:         OpenMPI 4.1.7                      (matches the openmpi
#                                                    headers/libs the runtime
#                                                    will use).
#   Output:      ${PROJECT_DIR}/build-profiling-mpi/iqtree3-mpi
#
# Usage (submit on Gadi):
#   qsub gadi-ci/bootstrap_iqtree_mpi.sh
#
#PBS -N iqtree-mpi-bootstrap
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
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling-mpi}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-master}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 bootstrap on Gadi (MPI + icpx + libiomp5)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Module load: MPI + LLVM/Clang + cmake + Eigen + Boost ────────────────
# Order matters: openmpi must come BEFORE intel-compiler-llvm so mpicxx is
# resolved from openmpi but its underlying CXX (icpx) is taken from the
# intel-compiler-llvm module via OMPI_CXX/OMPI_CC env vars.
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

# Tell mpicxx/mpicc to call icpx/icx instead of g++/gcc. This is the
# canonical OpenMPI mechanism for swapping the underlying compiler at build
# time without rebuilding the MPI stack.
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
    echo "       Try: module avail openmpi" >&2
    exit 2
fi

CC="$(command -v mpicc)"
CXX="$(command -v mpicxx)"
echo "[bootstrap-mpi] mpicc=${CC}  (-> ${OMPI_CC})"
echo "[bootstrap-mpi] mpicxx=${CXX} (-> ${OMPI_CXX})"
${CXX} --version | head -3 || true
mpirun --version | head -2 || true

EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "ERROR: ${SRC_DIR} not found." >&2
    echo "       Clone on a login node first (and apply R1/R2 NUMA patches):" >&2
    echo "         git clone ${IQTREE_REPO} ${SRC_DIR}" >&2
    echo "         cd ${SRC_DIR} && git submodule update --init --recursive" >&2
    exit 1
fi

# Pre-flight: confirm the R2 patches are still present in the working tree.
# If they have been reverted, this build would silently produce a non-R2
# binary and the MPI placement experiment would be invalid.
if grep -q 'schedule(dynamic,1)' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: ${SRC_DIR}/tree/phylokernelnew.h still has schedule(dynamic,1)" >&2
    echo "       — R2 patches missing.  Re-apply per CHANGELOG entry 2026-05-08 (a)." >&2
    exit 4
fi
echo "[bootstrap-mpi] R2 schedule(static) sites present in phylokernelnew.h"

# Same source-side toggles as bootstrap_iqtree_clang.sh:
#   • cmaple IPO disabled for compiler/build-time parity
#   • cmaple unittest disabled (no internet for googletest FetchContent on
#     Gadi compute nodes)
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

# Sapphire Rapids tuning, identical optimisation level + debug to the non-MPI
# clang build so wall-time deltas are attributable to MPI/placement only.
ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

echo ""
echo "[bootstrap-mpi] ── building ${BUILD_PROFILING} ──"
rm -rf "${BUILD_PROFILING}"
mkdir -p "${BUILD_PROFILING}"
cd "${BUILD_PROFILING}"

# IQTREE_FLAGS=mpi triggers:
#   • -D_IQTREE_MPI                              (MPIHelper code paths)
#   • EXE_SUFFIX="-mpi"                          (output: iqtree3-mpi)
#   • find_package(MPI) + MPI_*_LIBRARIES link
# Because CMAKE_CXX_COMPILER is already mpicxx, find_package(MPI) succeeds
# trivially; the MPI libraries come from the wrapper's own link line.
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

# IQ-TREE's CMake names the MPI binary iqtree3-mpi (EXE_SUFFIX="-mpi").
# Some build layouts emit it inside iqtree3-mpi*/  — symlink to the canonical
# top-level path so downstream worker scripts can reference one location.
if [[ ! -x "${BUILD_PROFILING}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_PROFILING}" -maxdepth 2 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    if [[ -n "${found}" && "${found}" != "${BUILD_PROFILING}/iqtree3-mpi" ]]; then
        ln -sf "${found}" "${BUILD_PROFILING}/iqtree3-mpi"
    fi
fi

if [[ ! -x "${BUILD_PROFILING}/iqtree3-mpi" ]]; then
    echo "ERROR: ${BUILD_PROFILING}/iqtree3-mpi not produced." >&2
    echo "       Check build log; expected EXE_SUFFIX=-mpi from IQTREE_FLAGS=mpi." >&2
    exit 5
fi

echo ""
echo "[bootstrap-mpi] verifying linkage..."
LDD_OUT="$(ldd "${BUILD_PROFILING}/iqtree3-mpi" 2>&1)"
echo "${LDD_OUT}" | grep -iE 'omp|mpi' || true
if echo "${LDD_OUT}" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — MPI build accidentally pulled libgomp." >&2
    echo "     Rebuild with OMPI_CXX=$(command -v icpx) explicitly set." >&2
    exit 6
fi
if ! echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)' ; then
    echo "  ✗ libmpi not linked — IQTREE_FLAGS=mpi did not take effect." >&2
    exit 7
fi
echo "  → libomp/libiomp5 + libmpi linked. OK."

# IMPORTANT: even a serial --version call on an MPI binary needs to be
# wrapped in mpirun -n 1 on some MPI stacks (otherwise MPI_Init aborts
# without a node). Use mpirun -n 1 to be safe.
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
echo "[bootstrap-mpi] OK"
echo "  binary:   ${BUILD_PROFILING}/iqtree3-mpi"
echo "  metadata: ${BUILD_PROFILING}/.build-info.json"
