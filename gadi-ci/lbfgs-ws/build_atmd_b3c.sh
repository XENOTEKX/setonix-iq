#!/bin/bash
# build_atmd_b3.sh — build the ATMD B.3+B.4 (HH-NUMA Mode F) MPI binary on a Gadi SPR node.
#
# Source:  /scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3   (branch fca-lbfgs-ws)
# Output:  /scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3c/iqtree3-mpi
#          symlinked as:  iqtree3-mpi-atmd-b3
#
# Adds -DIQTREE_ATMD=ON to the same cmake invocation as build_mf_iso.sh.
# This defines -D_IQTREE_ATMD at compile time, enabling:
#   - B.-1: omp_set_max_active_levels(2), MPI_Init_thread(FUNNELED), named critical regions
#   - B.3+B.4: K_outer OMP outer team (memory-bounded), M_inner inner threads, NUMA first-touch
#
# MUST run on a SPR compute node (login nodes lack -march=sapphirerapids support in GCC).
# Submit with:  qsub build_atmd_b3.sh
#
# See: research/lbfgs-and-warmstart-implementation.md §15.3 – §15.4

#PBS -N build-atmd-b3c
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=00:45:00
#PBS -l wd
#PBS -l storage=scratch/dx61+scratch/rc29+scratch/um09
#PBS -j oe

set -euo pipefail

# NOTE: PBS sets $PROJECT=dx61 (billing project). Use SRC_PROJECT for the rc29 scratch
# path where source and builds live (separate from the dx61 billing project).
SRC_PROJECT="rc29"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${SRC_PROJECT}/${USER_ID}/iqtree3-mf-iso}"
SRC_DIR="${SRC_DIR:-${ISO_DIR}/src/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${ISO_DIR}/build-atmd-b3c}"
BINARY_NAME="iqtree3-mpi-atmd-b3c"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE3 ATMD B.3+B.4 Build  (HH-NUMA Mode F)              ║"
echo "║  source:   ${SRC_DIR}"
echo "║  build:    ${BUILD_DIR}"
echo "║  binary:   ${BINARY_NAME}"
echo "║  branch:   $(cd "${SRC_DIR}" 2>/dev/null && git branch --show-current 2>/dev/null || echo unknown)"
echo "║  HEAD:     $(cd "${SRC_DIR}" 2>/dev/null && git log --oneline -1 2>/dev/null || echo unknown)"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6                 2>/dev/null || true
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm          2>/dev/null || true
    module load binutils/2.44                2>/dev/null || true
    module load eigen/3.3.7                  2>/dev/null || true
    module load boost/1.84.0                 2>/dev/null || true
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
echo "[build] mpicc=${CC}  (-> ${OMPI_CC})"
echo "[build] mpicxx=${CXX} (-> ${OMPI_CXX})"
${CXX} --version | head -2 || true

EIGEN3_INCLUDE_DIR="${EIGEN_ROOT:+${EIGEN_ROOT}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

# ── Source preflight ───────────────────────────────────────────────────
[[ -d "${SRC_DIR}/.git" ]] || { echo "ERROR: ${SRC_DIR}/.git missing." >&2; exit 1; }

# Verify B.-1 and B.3+B.4 patches are present in source.
if ! grep -q '_IQTREE_ATMD' "${SRC_DIR}/main/main.cpp"; then
    echo "ERROR: B.-1 patch missing in main.cpp (_IQTREE_ATMD not found)." >&2; exit 4
fi
if ! grep -q 'MPI_Init_thread' "${SRC_DIR}/utils/MPIHelper.cpp"; then
    echo "ERROR: B.-1 MPI_Init_thread patch missing in MPIHelper.cpp." >&2; exit 4
fi
if ! grep -q 'atmd_K_outer' "${SRC_DIR}/main/phylotesting.cpp"; then
    echo "ERROR: B.3+B.4 atmd_K_outer patch missing in phylotesting.cpp." >&2; exit 4
fi
if ! grep -q 'IQTREE_ATMD' "${SRC_DIR}/CMakeLists.txt"; then
    echo "ERROR: IQTREE_ATMD option missing in CMakeLists.txt." >&2; exit 4
fi
echo "[build] source preflight OK (B.-1 + B.3+B.4 patches present)"

# ── cmaple tweaks (same as build_mf_iso.sh) ───────────────────────────
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE).*|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled|' "${CMAPLE_CML}"
fi
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled|' "${CMAPLE_CML}"
fi
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
fi

ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

# ── Configure ──────────────────────────────────────────────────────────
echo ""
echo "[build] ── configuring ${BUILD_DIR} ──"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CC="${CC}" CXX="${CXX}" cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIQTREE_FLAGS=mpi \
    -DIQTREE_ATMD=ON \
    -DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}" \
    -DBOOST_ROOT="${BOOST_ROOT}" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"

# ── Build ──────────────────────────────────────────────────────────────
JOBS="${IQTREE_BUILD_JOBS:-$(nproc)}"
echo "[build] ── make -j${JOBS} ──"
T_BUILD_START=$(date +%s)
make -j"${JOBS}"
T_BUILD_END=$(date +%s)
echo "[build] make done in $(( T_BUILD_END - T_BUILD_START ))s"

# Locate binary (cmake may put it in a subdirectory).
if [[ ! -x "${BUILD_DIR}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_DIR}" -maxdepth 3 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    [[ -n "${found}" ]] && ln -sf "${found}" "${BUILD_DIR}/iqtree3-mpi"
fi
[[ -x "${BUILD_DIR}/iqtree3-mpi" ]] || { echo "ERROR: iqtree3-mpi not produced." >&2; exit 5; }

# Create named symlink for script references.
ln -sf "${BUILD_DIR}/iqtree3-mpi" "${BUILD_DIR}/${BINARY_NAME}"
echo "[build] symlink: ${BUILD_DIR}/${BINARY_NAME} -> iqtree3-mpi"

# ── Linkage verification ───────────────────────────────────────────────
echo ""
echo "[build] ── verifying linkage ──"
LDD_OUT="$(ldd "${BUILD_DIR}/iqtree3-mpi" 2>&1)"
echo "${LDD_OUT}" | grep -iE 'omp|mpi' || true
if echo "${LDD_OUT}" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — expected libiomp5." >&2; exit 6
fi
if ! echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)'; then
    echo "  ✗ libmpi not linked." >&2; exit 7
fi
echo "  → libiomp5 + libmpi linked. OK."

# ── ATMD symbol checks ─────────────────────────────────────────────────
echo ""
echo "[build] ── verifying ATMD B.3+B.4 symbols ──"
cat "${BUILD_DIR}/iqtree3-mpi" > /dev/null 2>&1 || true  # warm Lustre pages

# icpx with -g stores cout<< string literals as split pieces; use substrings
# that appear contiguously in the binary rather than the full log line.
if strings "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -qE 'ATMD Mode F|K_outer=|atmd_K_outer'; then
    echo "  → [ATMD Mode F] / K_outer symbols: present — ATMD compiled in"
else
    echo "  ✗ WARNING: ATMD symbols not found — _IQTREE_ATMD may not be active." >&2
fi

# Check MPI_Init_thread is linked (B.-1 patch).
if nm "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -qE 'MPI_Init_thread|PMPI_Init_thread'; then
    echo "  → MPI_Init_thread: present via nm (B.-1 OK)"
elif strings "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -q 'MPI_Init_thread'; then
    echo "  → MPI_Init_thread: present via strings (B.-1 OK)"
else
    echo "  ✗ WARNING: MPI_Init_thread not found — B.-1 MPI patch may be missing." >&2
fi

# Also verify warm-start A.2 symbols are still present (regression check).
if strings "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -q 'ws_bcast_fields'; then
    echo "  → ws_bcast_fields: present (A.2 warm-start intact)"
else
    echo "  ✗ WARNING: ws_bcast_fields not found — A.2 warm-start may have regressed." >&2
fi

# ── Smoke test ─────────────────────────────────────────────────────────
echo ""
echo "[build] ── smoke test: --version ──"
mpirun -n 1 "${BUILD_DIR}/iqtree3-mpi" --version 2>&1 | head -3 || true

ACTUAL_MD5=$(md5sum "${BUILD_DIR}/iqtree3-mpi" | awk '{print $1}')
echo "[build] md5: ${ACTUAL_MD5}"

# ── Build metadata ─────────────────────────────────────────────────────
cat > "${BUILD_DIR}/.build-info.json" <<EOF
{
  "build_tag":     "atmd_b3c_icx_avx512_spr",
  "compiler":      "$(${OMPI_CXX} --version | head -1)",
  "compiler_kind": "icpx",
  "mpi_wrapper":   "${CXX}",
  "iqtree_flags":  "mpi",
  "atmd_flag":     "IQTREE_ATMD=ON  (-D_IQTREE_ATMD)",
  "arch_flags":    "${ARCH_FLAGS} ${EXTRA}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)",
  "iqtree_branch": "$(cd "${SRC_DIR}" && git branch --show-current 2>/dev/null || echo unknown)",
  "iqtree_commit": "$(cd "${SRC_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "phases":        ["phase0_fca", "phase0.5_filterRatesMPI", "phase0.6_getNextModel_priority",
                    "mf_time_instrumentation", "ws_a2_warm_start",
                    "atmd_b-1_infra", "atmd_b3_hhNuma_modeF", "atmd_b4_mem_semaphore"],
  "binary_name":   "${BINARY_NAME}",
  "md5":           "${ACTUAL_MD5}"
}
EOF

echo ""
echo "[build] ══ DONE ══════════════════════════════════════════════"
echo "  binary:   ${BUILD_DIR}/iqtree3-mpi"
echo "  symlink:  ${BUILD_DIR}/${BINARY_NAME}"
echo "  metadata: ${BUILD_DIR}/.build-info.json"
echo "  md5:      ${ACTUAL_MD5}"
echo ""
echo "  Next: qsub run_atmd_b3_aa_1m_4node.sh   (correctness gate, K_outer≈1)"
echo "        qsub run_atmd_b3_aa_100k_1node.sh  (K_outer>1 activation test)"
