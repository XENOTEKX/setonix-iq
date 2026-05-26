#!/bin/bash
# build_mode_p_iso_p3.sh — P.ISO P.3 build: baseline + kernel Allreduce patches.
#
# This binary serves the ISO-2 gate. It contains P.1+P.2 (CLI, partition wiring)
# PLUS P.3 (limits-shift + Allreduce in computeLikelihoodBranch{,Generic}SIMD).
# With --mode-p-all on np>=2, each rank computes ONLY its slice of patterns and
# MPI_Allreduce sums the tree_lh at kernel exit. lnL must still match FCA.
#
# Source:  /scratch/rc29/as1708/iqtree3-mode-p-iso/src/iqtree3-mode-p-iso-p3
# Output:  /scratch/rc29/as1708/iqtree3-mode-p-iso/build-mode-p-iso-p3/iqtree3-mpi-mode-p-iso-p3
# Gate:    ISO-2 (np=2 --mode-p-all → lnL within 1e-6 of FCA baseline)
# See:     research/Modelfinder/mode-p-implementation-status.md §P.3

#PBS -N build-mp-iso-p3
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
SRC_DIR="${SANDBOX}/src/iqtree3-mode-p-iso-p3"
BUILD_DIR="${SANDBOX}/build-mode-p-iso-p3"
BINARY_NAME="iqtree3-mpi-mode-p-iso-p3"
PHYLO_CPP="${SRC_DIR}/main/phylotesting.cpp"
LOG_DIR="${SANDBOX}/logs/build"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  P.ISO P.3 build (P.1+P.2 scaffolding + kernel Allreduce)    ║"
echo "║  source:   ${SRC_DIR}"
echo "║  build:    ${BUILD_DIR}"
echo "║  binary:   ${BINARY_NAME}"
echo "║  log:      ${LOG_DIR}/build-p3.log"
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

EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/apps/eigen/3.3.7/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}"

# ── Source preflight: P.1+P.2 AND P.3 must be present ────────────────
[[ -f "${PHYLO_CPP}" ]] || { echo "ERROR: ${PHYLO_CPP} missing." >&2; exit 1; }

if ! grep -q 'initializePtnPartition' "${SRC_DIR}/tree/phylotree.cpp"; then
    echo "ERROR: P.1/P.2 scaffolding missing in p3 source." >&2; exit 4
fi
if ! grep -q 'P\.3 Mode P' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: P.3 markers missing in p3 source." >&2
    echo "       Expected: 'P.3 Mode P:' comment in phylokernelnew.h" >&2
    exit 4
fi
if ! grep -qE 'modePAllreduceLh\(tree_lh(_local)?\)' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: P.3 Allreduce call missing in kernel." >&2; exit 4
fi
# Count P.3 sites to confirm patch fully applied
P3_HITS=$(grep -c 'P\.3 Mode P' "${SRC_DIR}/tree/phylokernelnew.h")
if [[ "${P3_HITS}" -lt 2 ]]; then
    echo "ERROR: expected ≥2 P.3 markers (limits-shift + Allreduce), found ${P3_HITS}." >&2
    exit 4
fi
if ! grep -qE 'p6_lite_collective|P\.7-MPGC|mpgc_active' "${SRC_DIR}/main/phylotesting.cpp"; then
    echo "ERROR: collective Mode P dispatch missing (neither P.6-lite nor P.7-MPGC) in phylotesting.cpp." >&2
    echo "       Expected: 'p6_lite_collective' OR 'P.7-MPGC' OR 'mpgc_active' in evaluateAll()." >&2
    exit 4
fi
P4_HITS=$(grep -c 'P\.4 Mode P' "${SRC_DIR}/tree/phylokernelnew.h")
if [[ "${P4_HITS}" -lt 2 ]]; then
    echo "ERROR: expected ≥2 P.4 markers (Derv limits-shift + Allreduce), found ${P4_HITS}." >&2
    exit 4
fi
if ! grep -q 'modePAllreduceLhDfDdf' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: P.4 3-value Allreduce call missing in Derv kernel." >&2
    exit 4
fi
P5A_HITS=$(grep -c 'P\.5a Mode P' "${SRC_DIR}/tree/phylokernelnew.h")
if [[ "${P5A_HITS}" -lt 2 ]]; then
    echo "ERROR: expected ≥2 P.5a markers (FromBuffer loop bounds + Allreduce), found ${P5A_HITS}." >&2
    exit 4
fi
# B.4-9 (HALF-tree_lh leak fix): 2 kernel fallback sites + 1 EM accumulator site
# (matches "B.4-9 Mode P fix" or "B.4-9 / B.4-11 Mode P fix" — Branch fallback was
# updated to combine both bug references when B.4-11 fix removed inner Allreduce)
B49_KERNEL_HITS=$(grep -cE 'B\.4-9.* Mode P fix' "${SRC_DIR}/tree/phylokernelnew.h")
if [[ "${B49_KERNEL_HITS}" -lt 2 ]]; then
    echo "ERROR: expected ≥2 B.4-9 markers in kernel fallback paths (Branch + FromBuffer), found ${B49_KERNEL_HITS}." >&2
    exit 4
fi
if ! grep -q 'B\.4-9 Mode P fix' "${SRC_DIR}/model/rategammainvar.cpp"; then
    echo "ERROR: B.4-9 EM-accumulator fix missing in rategammainvar.cpp." >&2
    exit 4
fi
# B.4-14 (filterRatesMPI cross-group Bcast fix): fca_comm member + group-scoped Bcasts
B414_HEADER=$(grep -c 'B\.4-14' "${SRC_DIR}/main/phylotesting.h")
B414_CPP=$(grep -c 'B\.4-14\|fca_comm' "${SRC_DIR}/main/phylotesting.cpp")
if [[ "${B414_HEADER}" -lt 1 ]] || [[ "${B414_CPP}" -lt 4 ]]; then
    echo "ERROR: B.4-14 filterRatesMPI scope fix missing (header=${B414_HEADER}, cpp=${B414_CPP}; expect header≥1, cpp≥4)." >&2
    exit 4
fi
# B.4-15 (MPGC inheritance to per-model iqtree): in_tree param on evaluate() +
# setModePGroupComm call inside evaluate() body.
B415_HEADER=$(grep -c 'B\.4-15\|PhyloTree \*in_tree' "${SRC_DIR}/main/phylotesting.h")
B415_CPP=$(grep -c 'B\.4-15' "${SRC_DIR}/main/phylotesting.cpp")
# The setModePGroupComm call in evaluate() body — must appear at least once
# (the MPGC-setup-time call at line ~3997 and the cleanup call at line ~4646
# already exist; the new inheritance call adds a 3rd occurrence).
B415_SETCOMM=$(grep -c 'setModePGroupComm' "${SRC_DIR}/main/phylotesting.cpp")
if [[ "${B415_HEADER}" -lt 1 ]] || [[ "${B415_CPP}" -lt 1 ]] || [[ "${B415_SETCOMM}" -lt 3 ]]; then
    echo "ERROR: B.4-15 MPGC inheritance fix missing (header=${B415_HEADER}, cpp=${B415_CPP}, setComm=${B415_SETCOMM}; expect header≥1, cpp≥1, setComm≥3)." >&2
    exit 4
fi
# Architecture C (tree-traversal slice fix): partial_lh computation honours
# Mode P slice [ptn_start, ptn_end). Marker in phylokernelnew.h.
ARCH_C=$(grep -c 'Architecture C' "${SRC_DIR}/tree/phylokernelnew.h")
if [[ "${ARCH_C}" -lt 1 ]]; then
    echo "ERROR: Architecture C tree-traversal slice fix missing in phylokernelnew.h (expected 'Architecture C' marker, found ${ARCH_C})." >&2
    exit 4
fi
# ATMD-AID (Adaptive Island Dispatch): wave-based dispatch with sub-comm lattice.
AID_HEADER=$(grep -c 'aid_cost_pred\|aidComputeCostPred\|MF_AID_HEAVY\|aidExecuteWaves\|aid_lattice_comm' "${SRC_DIR}/main/phylotesting.h")
AID_CPP=$(grep -c 'aidComputeCostPred\|aidScheduleWaves\|aidBuildLattice\|aidExecuteWaves\|aid_active' "${SRC_DIR}/main/phylotesting.cpp")
AID_CLI=$(grep -c 'atmd_aid_enabled\|--atmd-aid' "${SRC_DIR}/utils/tools.h" "${SRC_DIR}/utils/tools.cpp" | awk -F: '{s+=$NF} END{print s+0}')
if [[ "${AID_HEADER}" -lt 5 ]] || [[ "${AID_CPP}" -lt 10 ]] || [[ "${AID_CLI}" -lt 3 ]]; then
    echo "ERROR: ATMD-AID dispatch fix missing (header=${AID_HEADER}, cpp=${AID_CPP}, cli=${AID_CLI}; expect header≥5, cpp≥10, cli≥3)." >&2
    exit 4
fi
echo "[preflight] OK: P.1+P.2 present, P.3 patches applied (${P3_HITS}), P.4 patches applied (${P4_HITS}), P.5a patches applied (${P5A_HITS}), P.6-lite present, B.4-9 fixes applied (kernel=${B49_KERNEL_HITS}, EM=1), B.4-14 fix applied (header=${B414_HEADER}, cpp=${B414_CPP}), B.4-15 fix applied (header=${B415_HEADER}, cpp=${B415_CPP}, setComm=${B415_SETCOMM}), Architecture C applied (${ARCH_C}), ATMD-AID applied (header=${AID_HEADER}, cpp=${AID_CPP}, cli=${AID_CLI})."

# ── cmaple disables ───────────────────────────────────────────────────
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

# ── Configure & build ──────────────────────────────────────────────────
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
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" 2>&1 | tee "${LOG_DIR}/cmake-p3.log"

JOBS="${IQTREE_BUILD_JOBS:-$(nproc)}"
echo "[build] ── make -j${JOBS} ──"
T_BUILD_START=$(date +%s)
make -j"${JOBS}" 2>&1 | tee "${LOG_DIR}/build-p3.log"
T_BUILD_END=$(date +%s)
echo "[build] make done in $(( T_BUILD_END - T_BUILD_START ))s"

if [[ ! -x "${BUILD_DIR}/iqtree3-mpi" ]]; then
    found="$(find "${BUILD_DIR}" -maxdepth 3 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    [[ -n "${found}" ]] && ln -sf "${found}" "${BUILD_DIR}/iqtree3-mpi"
fi
[[ -x "${BUILD_DIR}/iqtree3-mpi" ]] || { echo "ERROR: iqtree3-mpi not produced." >&2; exit 5; }
ln -sf "${BUILD_DIR}/iqtree3-mpi" "${BUILD_DIR}/${BINARY_NAME}"

# Linkage check
LDD_OUT="$(ldd "${BUILD_DIR}/iqtree3-mpi" 2>&1)"
echo "${LDD_OUT}" | grep -q 'libgomp' && { echo "  ✗ libgomp linked." >&2; exit 6; }
echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)' || { echo "  ✗ libmpi not linked." >&2; exit 7; }
echo "[build] ✓ libiomp5 + libmpi linked"

MD5=$(md5sum "${BUILD_DIR}/iqtree3-mpi" | awk '{print $1}')
echo "[build] md5: ${MD5}"

cat > "${BUILD_DIR}/.build-info.json" <<EOF
{
  "build_tag":     "mode_p_iso_p3",
  "phases":        ["P.1 scaffolding", "P.2 partition wiring", "P.3 kernel Allreduce"],
  "kernel":        "computeLikelihoodBranch{,Generic}SIMD: limits-shift + Allreduce",
  "binary_name":   "${BINARY_NAME}",
  "md5":           "${MD5}",
  "source_dir":    "${SRC_DIR}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)"
}
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  P.ISO P.3 build COMPLETE                                    ║"
echo "║  binary:  ${BUILD_DIR}/${BINARY_NAME}"
echo "║  md5:     ${MD5}"
echo "║  Next:    qsub run_iso2_aa100k_np2_p3.sh"
echo "║           (only after ISO-0 + ISO-1 pass on the base build!)"
echo "╚══════════════════════════════════════════════════════════════╝"
