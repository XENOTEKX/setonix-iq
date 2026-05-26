#!/bin/bash
# build_atmd_b4.sh — build ATMD b4: clean production binary with B.5 per_tree_bytes fix.
#
# What b4 changes vs b3c/b3d (in phylotesting.cpp source, committed in fca-lbfgs-ws):
#   B.5  Per_tree_bytes formula matched to initializeAllPartialLh() actual allocation.
#        Old: npat * nstates * nrates * 4 * nodeNum  (overestimated ~8x; K_outer=1 at AA 1M)
#        New: (leafN-2 + 2) * (npat * nstates * nrates) * 8 + (leafN-2) * npat * nrates
#        Effect: at AA 1M, K_outer rises 1 → ~6 (real per_tree ≈ 64 GB, not 458 GB).
#   B.4-diag removed: entry, pre-block, sidecar fprintf/fopen blocks deleted from source.
#                    [ATMD Mode F] cout log line retained (production diagnostic).
#
# DESIGN INTENT: Empirically test the bandwidth-saturation hypothesis at AA 1M scale.
# The §15.9.14 analysis (research/lbfgs-and-warmstart-implementation.md) predicts that
# at K_outer=6 the AA 1M kernel will exhibit the same memory-bandwidth contention
# observed at 100K K_outer=8 (1.64x regression vs K=1). b4 lets us verify this directly
# instead of inferring from the 100K-only data point.
#
# Source:  /scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3  (branch fca-lbfgs-ws)
# Output:  /scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b4/iqtree3-mpi-atmd-b4
# See:     research/lbfgs-and-warmstart-implementation.md §15.9.15

#PBS -N build-atmd-b4
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=00:45:00
#PBS -l wd
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -j oe

set -euo pipefail

SRC_PROJECT="rc29"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${SRC_PROJECT}/${USER_ID}/iqtree3-mf-iso}"
SRC_DIR="${SRC_DIR:-${ISO_DIR}/src/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${ISO_DIR}/build-atmd-b4}"
BINARY_NAME="iqtree3-mpi-atmd-b4"
PHYLO_CPP="${SRC_DIR}/main/phylotesting.cpp"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE3 ATMD b4 Build  (B.5 per_tree formula + clean)     ║"
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

export OMPI_CC="$(command -v icx)"
export OMPI_CXX="$(command -v icpx)"
CC="$(command -v mpicc)"
CXX="$(command -v mpicxx)"
${CXX} --version | head -1 || true

EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

# ── Source preflight ───────────────────────────────────────────────────
[[ -d "${SRC_DIR}/.git" ]] || { echo "ERROR: ${SRC_DIR}/.git missing." >&2; exit 1; }
[[ -f "${PHYLO_CPP}" ]]    || { echo "ERROR: ${PHYLO_CPP} missing." >&2; exit 1; }

# Verify B.5 formula present (the new max_lh_slots_est-based estimate)
if ! grep -q 'max_lh_slots_est' "${PHYLO_CPP}"; then
    echo "ERROR: B.5 per_tree formula (max_lh_slots_est) missing in phylotesting.cpp." >&2
    echo "       Re-apply the B.5 patch or restore from git." >&2
    exit 4
fi
# Verify diagnostics are NOT present (b3c remnants)
if grep -q 'ATMD-DIAG' "${PHYLO_CPP}"; then
    echo "ERROR: b3c diagnostic fprintf blocks still present. b4 expects clean source." >&2
    echo "       Run: git checkout -- ${PHYLO_CPP}  if you want to restore b3c diagnostics." >&2
    exit 4
fi
# Verify ATMD K_outer logic present
if ! grep -q 'atmd_K_outer' "${PHYLO_CPP}"; then
    echo "ERROR: B.3+B.4 atmd_K_outer logic missing in phylotesting.cpp." >&2; exit 4
fi
# Verify [ATMD Mode F] production log line present
if ! grep -q '\[ATMD Mode F\]' "${PHYLO_CPP}"; then
    echo "ERROR: [ATMD Mode F] production log line missing in phylotesting.cpp." >&2; exit 4
fi
echo "[build] source preflight OK (B.5 formula present, b3c diagnostics absent)"

# ── cmaple tweaks (same as all ATMD builds) ───────────────────────────
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

if [[ ! -x "${BUILD_DIR}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_DIR}" -maxdepth 3 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    [[ -n "${found}" ]] && ln -sf "${found}" "${BUILD_DIR}/iqtree3-mpi"
fi
[[ -x "${BUILD_DIR}/iqtree3-mpi" ]] || { echo "ERROR: iqtree3-mpi not produced." >&2; exit 5; }

ln -sf "${BUILD_DIR}/iqtree3-mpi" "${BUILD_DIR}/${BINARY_NAME}"
echo "[build] symlink: ${BUILD_DIR}/${BINARY_NAME} -> iqtree3-mpi"

# ── Linkage verification ───────────────────────────────────────────────
LDD_OUT="$(ldd "${BUILD_DIR}/iqtree3-mpi" 2>&1)"
if echo "${LDD_OUT}" | grep -q 'libgomp'; then
    echo "  ✗ libgomp linked — expected libiomp5." >&2; exit 6
fi
if ! echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)'; then
    echo "  ✗ libmpi not linked." >&2; exit 7
fi
echo "  → libiomp5 + libmpi: OK"

# ── Symbol checks ──────────────────────────────────────────────────────
echo ""
echo "[build] ── verifying b4 symbol state ──"
ACTUAL_MD5=$(md5sum "${BUILD_DIR}/iqtree3-mpi" | awk '{print $1}')
echo "  md5: ${ACTUAL_MD5}"

if grep -q 'ATMD-DIAG' "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null; then
    echo "  ✗ WARNING: [ATMD-DIAG] still present in binary — diagnostic not fully removed" >&2
else
    echo "  ✓ [ATMD-DIAG] NOT in binary (clean production)"
fi
if grep -q 'ATMD Mode F' "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null; then
    echo "  ✓ [ATMD Mode F] in binary (production log line retained)"
fi
if grep -q 'atmd_K_outer' "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null; then
    echo "  ✓ atmd_K_outer in binary (K_outer logic retained)"
fi

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  b4 build COMPLETE  (B.5 formula + clean diagnostics)        ║"
echo "║  binary:  ${BUILD_DIR}/${BINARY_NAME}"
echo "║  md5:     ${ACTUAL_MD5}"
echo "║  Expected K_outer at AA 1M:  6  (vs b3c K_outer=1)"
echo "║  Expected K_outer at AA 100K: 8 (cap; same as b3c)"
echo "║  Next:    qsub run_atmd_b4_aa_1m_16node.sh"
echo "║           qsub run_atmd_b4_aa_100k_1node.sh"
echo "╚══════════════════════════════════════════════════════════════╝"
