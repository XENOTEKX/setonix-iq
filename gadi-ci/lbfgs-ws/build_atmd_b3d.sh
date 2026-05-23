#!/bin/bash
# build_atmd_b3d.sh — build ATMD b3d: clean production binary (no b3c diagnostics).
#
# WHEN TO SUBMIT: Only after job 169112256 (b3c 16-node 1M) shows ATMD beats FCA baseline.
# If K_outer=1 at 1M (expected) with no performance gain, b3d is NOT needed.
#
# What b3d removes vs b3c (phylotesting.cpp):
#   1. Entry diagnostic:  fprintf(stderr, "[ATMD-DIAG] evaluateAll() ENTRY: ...")
#   2. Pre-block diag:    fprintf(stderr, "[ATMD-DIAG] evaluateAll B.3+B.4 pre-block: ...")
#   3. Sidecar diag:      fopen(prefix + ".atmd_diag", ...) + fprintf sidecar lines
# What b3d KEEPS (production diagnostics):
#   - cout << "[ATMD Mode F] K_outer=..." (goes into IQ-TREE's normal log)
#   - All B.4-1 /proc/meminfo memory budget logic
#   - All B.3+B.4 K_outer×M_inner nested OMP dispatch
#
# Source:  /scratch/rc29/as1708/iqtree3-mf-iso/src/iqtree3  (branch fca-lbfgs-ws)
# Output:  /scratch/rc29/as1708/iqtree3-mf-iso/build-atmd-b3d/iqtree3-mpi-atmd-b3d
# See:     research/lbfgs-and-warmstart-implementation.md §15.9.14

#PBS -N build-atmd-b3d
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
BUILD_DIR="${BUILD_DIR:-${ISO_DIR}/build-atmd-b3d}"
BINARY_NAME="iqtree3-mpi-atmd-b3d"
PHYLO_CPP="${SRC_DIR}/main/phylotesting.cpp"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE3 ATMD b3d Build  (clean, no diagnostic overhead)   ║"
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
[[ -f "${PHYLO_CPP}" ]] || { echo "ERROR: ${PHYLO_CPP} missing." >&2; exit 1; }

# Verify ATMD patches present
if ! grep -q 'atmd_K_outer' "${PHYLO_CPP}"; then
    echo "ERROR: B.3+B.4 atmd_K_outer patch missing in phylotesting.cpp." >&2; exit 4
fi
echo "[build] source preflight OK"

# ── Remove b3c diagnostic blocks from phylotesting.cpp ────────────────
echo "[build] Stripping b3c diagnostic blocks from phylotesting.cpp..."
PHYLO_BACKUP="${PHYLO_CPP}.b3c.bak"
cp "${PHYLO_CPP}" "${PHYLO_BACKUP}"
echo "[build] Backup saved: ${PHYLO_BACKUP}"

python3 - "${PHYLO_CPP}" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path, 'r') as f:
    text = f.read()

orig_len = len(text)

# Block 1: entry diagnostic (lines ~3806-3809)
# Matches: "// B.4-diag: entry diagnostic..." comment + fprintf + arg line + fflush + blank
text = re.sub(
    r'[ \t]*// B\.4-diag: entry diagnostic.*?\n'
    r'[ \t]*fprintf\(stderr, "\[ATMD-DIAG\] evaluateAll\(\) ENTRY:.*?\n'
    r'[ \t]*params\.atmd_K_outer.*?\);\n'
    r'[ \t]*fflush\(stderr\);\n',
    '',
    text
)

# Block 2: pre-block diagnostic (lines ~4107-4128)
# Matches: "// B.4-diag: unconditional pre-block..." through fflush(stderr);
text = re.sub(
    r'[ \t]*// B\.4-diag: unconditional pre-block.*?\n'
    r'[ \t]*fprintf\(stderr, "\[ATMD-DIAG\] evaluateAll B\.3\+B\.4 pre-block:.*?fflush\(stderr\);\n',
    '',
    text,
    flags=re.DOTALL
)

# Block 3: sidecar diagnostic (lines ~4182-4196)
# Matches: "// B.4-diag: write to a sidecar..." block through the closing } and fflush
text = re.sub(
    r'[ \t]*// B\.4-diag: write to a sidecar.*?\n'
    r'[ \t]*\{.*?\}\n'
    r'[ \t]*fprintf\(stderr, "\[ATMD-DIAG\] sidecar written to:.*?\);\n'
    r'[ \t]*fflush\(stderr\);\n',
    '',
    text,
    flags=re.DOTALL
)

# Verify [ATMD-DIAG] is gone
if '[ATMD-DIAG]' in text:
    print(f"ERROR: [ATMD-DIAG] still present after patch!", file=sys.stderr)
    sys.exit(1)

# Verify [ATMD Mode F] production log line is still present
if '[ATMD Mode F]' not in text:
    print(f"ERROR: [ATMD Mode F] production log line was accidentally removed!", file=sys.stderr)
    sys.exit(1)

# Verify atmd_K_outer logic is still present
if 'atmd_K_outer' not in text:
    print(f"ERROR: atmd_K_outer logic was accidentally removed!", file=sys.stderr)
    sys.exit(1)

with open(path, 'w') as f:
    f.write(text)

removed = orig_len - len(text)
print(f"[patch] Removed {removed} bytes of diagnostic code from phylotesting.cpp")
print(f"[patch] [ATMD-DIAG] removed: OK")
print(f"[patch] [ATMD Mode F] production line: PRESENT")
print(f"[patch] atmd_K_outer logic: PRESENT")
PYEOF

echo "[build] Diagnostic strip complete. Verifying..."
if grep -q 'ATMD-DIAG' "${PHYLO_CPP}"; then
    echo "ERROR: [ATMD-DIAG] still in phylotesting.cpp after patch!" >&2
    cp "${PHYLO_BACKUP}" "${PHYLO_CPP}"
    exit 1
fi
if ! grep -q 'ATMD Mode F' "${PHYLO_CPP}"; then
    echo "ERROR: [ATMD Mode F] production log removed by patch!" >&2
    cp "${PHYLO_BACKUP}" "${PHYLO_CPP}"
    exit 1
fi
echo "[build] phylotesting.cpp patched cleanly: [ATMD-DIAG] gone, [ATMD Mode F] present"

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
echo "[build] ── verifying b3d symbol state ──"
ACTUAL_MD5=$(md5sum "${BUILD_DIR}/iqtree3-mpi" | awk '{print $1}')
echo "  md5: ${ACTUAL_MD5}"

if grep -q 'ATMD-DIAG' "${BUILD_DIR}/iqtree3-mpi" 2>/dev/null; then
    echo "  ✗ WARNING: [ATMD-DIAG] still present in binary — diagnostic not fully removed" >&2
else
    echo "  ✓ [ATMD-DIAG] NOT in binary (diagnostic overhead removed)"
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
echo "║  b3d build COMPLETE                                          ║"
echo "║  binary:  ${BUILD_DIR}/${BINARY_NAME}"
echo "║  md5:     ${ACTUAL_MD5}"
echo "║  Next:    qsub run_atmd_b3d_aa_1m_16node_full.sh"
echo "╚══════════════════════════════════════════════════════════════╝"
