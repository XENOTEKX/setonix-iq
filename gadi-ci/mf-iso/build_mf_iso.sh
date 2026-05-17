#!/bin/bash
# build_mf_iso.sh — build the isolated ModelFinder MPI binary on Gadi.
#
# Source:  /scratch/dx61/as1708/iqtree3-mf-iso/src/iqtree3
#          (branch mf-iso-phase0.5-0.6, on top of FCA Phase 0 ffb79a14)
# Output:  /scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
#
# Why a separate tree on dx61 (instead of patching the production
# /scratch/um09/as1708/iqtree3-mf2 build): the Phase 0.7 + HH-NUMA
# experiments on the production tree resulted in a hung job (168486582,
# SIGTERM after 1h19m, no stdout). The isolated tree lets us test the
# proven-correct Phase 0.5 + 0.6 changes in a clean environment
# without disturbing the production binary used for ongoing AA 100K
# benchmark runs.
#
# Toolchain: icpx (intel-compiler-llvm/2025.1.1) + libiomp5 + openmpi/4.1.7
# Architecture: -march=sapphirerapids -mtune=sapphirerapids -O3 -g (-fopenmp via flag)

#PBS -N mf-iso-bootstrap
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=01:00:00
#PBS -l wd
#PBS -l storage=scratch/dx61+scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
SRC_DIR="${SRC_DIR:-${ISO_DIR}/src/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${ISO_DIR}/build-mpi-iso}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE3 MF-Isolation Build (Phase 0.5 + 0.6 + MF-TIME)"
echo "║  source:        ${SRC_DIR}"
echo "║  build:         ${BUILD_DIR}"
echo "║  branch:        $(cd "${SRC_DIR}" 2>/dev/null && git branch --show-current 2>/dev/null || echo unknown)"
echo "║  HEAD:          $(cd "${SRC_DIR}" 2>/dev/null && git log --oneline -1 2>/dev/null || echo unknown)"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6           2>/dev/null || true
    module load openmpi/4.1.7          2>/dev/null || true
    module load intel-compiler-llvm    2>/dev/null || true
    module load binutils/2.44          2>/dev/null || true
    module load eigen/3.3.7            2>/dev/null || true
    module load boost/1.84.0           2>/dev/null || true
fi

if ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icpx not on PATH after module load." >&2; exit 2
fi
if ! command -v mpicxx >/dev/null 2>&1; then
    echo "ERROR: mpicxx not on PATH after module load." >&2; exit 2
fi

export OMPI_CC="$(command -v icx)"
export OMPI_CXX="$(command -v icpx)"
CC="$(command -v mpicc)"
CXX="$(command -v mpicxx)"
echo "[build-mf-iso] mpicc=${CC}  (-> ${OMPI_CC})"
echo "[build-mf-iso] mpicxx=${CXX} (-> ${OMPI_CXX})"
${CXX} --version | head -3 || true
mpirun --version | head -2 || true

EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

# ── Source preflight ───────────────────────────────────────────────────
if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "ERROR: ${SRC_DIR}/.git missing — was the source mirrored to dx61?" >&2
    exit 1
fi
if ! grep -q 'CandidateModelSet::filterRatesMPI' "${SRC_DIR}/main/phylotesting.cpp"; then
    echo "ERROR: ${SRC_DIR} missing Phase 0.5 patch (filterRatesMPI)." >&2; exit 4
fi
if ! grep -q 'MF-TIME: rank ' "${SRC_DIR}/main/phylotesting.cpp"; then
    echo "ERROR: ${SRC_DIR} missing MF-TIME instrumentation." >&2; exit 4
fi
if ! grep -q '_IQTREE_MPI' "${SRC_DIR}/main/phylotesting.h"; then
    echo "ERROR: ${SRC_DIR}/main/phylotesting.h missing _IQTREE_MPI guards." >&2; exit 4
fi
echo "[build-mf-iso] source preflight OK (Phase 0.5 + 0.6 + MF-TIME markers present)"

# cmaple build flags (same as MF2 bootstrap)
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE).*|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled for parity|' "${CMAPLE_CML}"
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
echo "[build-mf-iso] ── configuring ${BUILD_DIR} ──"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

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
echo "[build-mf-iso] ── make -j${JOBS} ──"
make -j"${JOBS}"

# Locate the binary (cmake may place it in build/, build/main/, or top-level).
if [[ ! -x "${BUILD_DIR}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_DIR}" -maxdepth 3 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    if [[ -n "${found}" && "${found}" != "${BUILD_DIR}/iqtree3-mpi" ]]; then
        ln -sf "${found}" "${BUILD_DIR}/iqtree3-mpi"
    fi
fi
if [[ ! -x "${BUILD_DIR}/iqtree3-mpi" ]]; then
    echo "ERROR: ${BUILD_DIR}/iqtree3-mpi not produced." >&2; exit 5
fi

echo ""
echo "[build-mf-iso] ── verifying linkage ──"
LDD_OUT="$(ldd "${BUILD_DIR}/iqtree3-mpi" 2>&1)"
echo "${LDD_OUT}" | grep -iE 'omp|mpi' || true
if echo "${LDD_OUT}" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — expected libiomp5." >&2; exit 6
fi
if ! echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)' ; then
    echo "  ✗ libmpi not linked." >&2; exit 7
fi
echo "  → libiomp5 + libmpi linked. OK."

# ── Symbol checks for the Phase 0.5/0.6 functions ──────────────────────
# Warm ALL data pages (stat only flushes inode metadata; nm needs the symbol
# table section at ~140 MB offset which won't be paged in on a cold Lustre mount).
cat "${BUILD_DIR}/iqtree3-mpi" > /dev/null 2>&1 || true
echo ""
echo "[build-mf-iso] ── verifying Phase 0.5/0.6 symbols ──"
if ! nm "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -q '_ZN17CandidateModelSet14filterRatesMPIEi'; then
    if ! strings "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -q 'filterRatesMPI'; then
        echo "  ✗ filterRatesMPI symbol missing in binary (nm + strings both failed)." >&2; exit 8
    fi
    echo "  → filterRatesMPI: found via strings (nm symtab read failed on Lustre)"
else
    echo "  → filterRatesMPI: found (_ZN17CandidateModelSet14filterRatesMPIEi)"
fi

# Smoke test.
mpirun -n 1 "${BUILD_DIR}/iqtree3-mpi" --version 2>&1 | head -3 || true

# Build metadata.
cat > "${BUILD_DIR}/.build-info.json" <<EOF
{
  "build_tag":     "mf_iso_phase0.5_0.6_icx_avx512_mftime",
  "compiler":      "$(${OMPI_CXX} --version | head -1)",
  "compiler_kind": "icpx",
  "mpi_wrapper":   "${CXX}",
  "mpi_version":   "$(mpirun --version 2>&1 | head -1)",
  "openmp_runtime":"libiomp5",
  "iqtree_flags":  "mpi",
  "arch_flags":    "${ARCH_FLAGS} ${EXTRA}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)",
  "iqtree_branch": "$(cd "${SRC_DIR}" && git branch --show-current 2>/dev/null || echo unknown)",
  "iqtree_commit": "$(cd "${SRC_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "phases":        ["phase0_fca", "phase0.5_filterRatesMPI", "phase0.6_getNextModel_priority", "mf_time_instrumentation"]
}
EOF

echo ""
echo "[build-mf-iso] OK"
echo "  binary:   ${BUILD_DIR}/iqtree3-mpi"
echo "  metadata: ${BUILD_DIR}/.build-info.json"
