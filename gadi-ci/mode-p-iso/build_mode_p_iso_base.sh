#!/bin/bash
# build_mode_p_iso_base.sh — P.ISO baseline build: P.1 + P.2 scaffolding only.
#
# This binary serves the ISO-0 and ISO-1 gates (Mode P inert: kernel ignores
# ptn_start/ptn_end so likelihoods are identical to FCA at any np). It contains
# the CLI flags, Params fields, PhyloTree members, and helper methods from
# P.1/P.2 but NO kernel modifications.
#
# Source:  /scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-base
# Output:  /scratch/rc29/as1708/iqtree3-mode-p-iso/build-mode-p-iso-base/iqtree3-mpi-mode-p-iso-base
# Gates:   ISO-0 (np=1, --mode-p-all inert), ISO-1 (np=2, partition emitted but lnL unchanged)
# See:     research/Modelfinder/mode-p-implementation-status.md

#PBS -N build-mp-iso-base
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=00:45:00
#PBS -l wd
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -j oe

set -euo pipefail

SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
SRC_DIR="${SANDBOX}/src/iqtree3-mode-p-iso-base"
BUILD_DIR="${SANDBOX}/build-mode-p-iso-base"
BINARY_NAME="iqtree3-mpi-mode-p-iso-base"
PHYLO_CPP="${SRC_DIR}/main/phylotesting.cpp"
LOG_DIR="${SANDBOX}/logs/build"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  P.ISO baseline build (P.1 + P.2 only; kernel UNPATCHED)     ║"
echo "║  source:   ${SRC_DIR}"
echo "║  build:    ${BUILD_DIR}"
echo "║  binary:   ${BINARY_NAME}"
echo "║  log:      ${LOG_DIR}/build-base.log"
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

command -v icpx >/dev/null 2>&1 || { echo "ERROR: icpx not on PATH." >&2; exit 2; }

export OMPI_CC="$(command -v icx)"
export OMPI_CXX="$(command -v icpx)"
CC="$(command -v mpicc)"
CXX="$(command -v mpicxx)"
${CXX} --version | head -1 || true

EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

# ── Source preflight: P.1/P.2 must be present; P.3 must NOT be ───────
[[ -f "${PHYLO_CPP}" ]] || { echo "ERROR: ${PHYLO_CPP} missing." >&2; exit 1; }

if ! grep -q 'initializePtnPartition' "${SRC_DIR}/tree/phylotree.cpp"; then
    echo "ERROR: P.1/P.2 scaffolding (initializePtnPartition) missing in source." >&2
    echo "       The ISO rsync did not capture uncommitted P.1/P.2 edits." >&2
    exit 4
fi
if ! grep -q 'mode_p_enabled' "${SRC_DIR}/utils/tools.h"; then
    echo "ERROR: P.1 Params::mode_p_enabled missing in source." >&2; exit 4
fi
# Baseline must NOT have P.3 markers — that's the whole point.
if grep -q 'P\.3 Mode P' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: P.3 markers found in BASE source. Baseline must be P.3-free." >&2
    exit 4
fi
echo "[preflight] OK: P.1+P.2 present, P.3 absent."

# ── cmaple disables (same as all our builds) ──────────────────────────
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if [[ -f "${CMAPLE_CML}" ]]; then
    if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
        sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE).*|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled|' "${CMAPLE_CML}"
    fi
    if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
        sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled|' "${CMAPLE_CML}"
    fi
    if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
        sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
    fi
fi

ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

# ── Configure ──────────────────────────────────────────────────────────
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
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" 2>&1 | tee "${LOG_DIR}/cmake-base.log"

# ── Build ──────────────────────────────────────────────────────────────
JOBS="${IQTREE_BUILD_JOBS:-$(nproc)}"
echo "[build] ── make -j${JOBS} ──"
T_BUILD_START=$(date +%s)
make -j"${JOBS}" 2>&1 | tee "${LOG_DIR}/build-base.log"
T_BUILD_END=$(date +%s)
echo "[build] make done in $(( T_BUILD_END - T_BUILD_START ))s"

# Locate and symlink the binary
if [[ ! -x "${BUILD_DIR}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_DIR}" -maxdepth 3 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    [[ -n "${found}" ]] && ln -sf "${found}" "${BUILD_DIR}/iqtree3-mpi"
fi
[[ -x "${BUILD_DIR}/iqtree3-mpi" ]] || { echo "ERROR: iqtree3-mpi not produced." >&2; exit 5; }
ln -sf "${BUILD_DIR}/iqtree3-mpi" "${BUILD_DIR}/${BINARY_NAME}"

# ── Linkage + Mode P symbol verification ──────────────────────────────
LDD_OUT="$(ldd "${BUILD_DIR}/iqtree3-mpi" 2>&1)"
echo "${LDD_OUT}" | grep -q 'libgomp' && { echo "  ✗ libgomp linked — expected libiomp5." >&2; exit 6; }
echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)' || { echo "  ✗ libmpi not linked." >&2; exit 7; }
echo "[build] ✓ libiomp5 + libmpi linked"

# Mode P symbol presence (P.1 scaffolding should be in binary):
if strings "${BUILD_DIR}/iqtree3-mpi" | grep -q '\[Mode P\] rank'; then
    echo "[build] ✓ Mode P diagnostic string in binary (P.2 wiring present)"
fi
if nm "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null | grep -q 'initializePtnPartition'; then
    echo "[build] ✓ initializePtnPartition symbol present (P.1)"
fi
if strings "${BUILD_DIR}/iqtree3-mpi" | grep -q 'Use --atmd-K-outer'; then
    echo "[build] ✓ B.5 ATMD CLI flags also in binary (carried from working tree)"
fi

MD5=$(md5sum "${BUILD_DIR}/iqtree3-mpi" | awk '{print $1}')
echo "[build] md5: ${MD5}"

cat > "${BUILD_DIR}/.build-info.json" <<EOF
{
  "build_tag":     "mode_p_iso_base",
  "phases":        ["P.1 scaffolding", "P.2 partition wiring"],
  "kernel":        "UNPATCHED (Mode P inert)",
  "binary_name":   "${BINARY_NAME}",
  "md5":           "${MD5}",
  "source_dir":    "${SRC_DIR}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)"
}
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  P.ISO baseline build COMPLETE                                ║"
echo "║  binary:  ${BUILD_DIR}/${BINARY_NAME}"
echo "║  md5:     ${MD5}"
echo "║  Next:    qsub run_iso0_aa100k_np1_base.sh"
echo "║           qsub run_iso1_aa100k_np2_base.sh"
echo "╚══════════════════════════════════════════════════════════════╝"
