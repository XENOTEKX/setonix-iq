#!/bin/bash
# rebuild_mf2_binary.sh — Rebuild the MF2 MPI binary on a Gadi SPR compute node.
#
# This script must run on a compute node (not a login node) because:
#   1. The MF2 build uses icpx (Intel compiler LLVM) + OpenMPI — both require
#      module load, which resolves only on compute nodes on this system.
#   2. The -march=sapphirerapids flag in the CMakeCache generates SIGILL on
#      non-SPR login nodes.
#
# CONTEXT — Why this rebuild is needed:
#   The pre-gather checkpoint corruption bug was fixed in phylotesting.cpp
#   (commit: post-Allreduce re-write of best_model_* checkpoint keys and
#   full model_list). The binary at:
#       /scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi
#   reflects the UNFIXED code until this script completes.
#
# After a successful build, run the verification tier:
#   cd ~/setonix-iq && qsub -v THREADS=104 tiers/verify_mf2_fix_np4.sh
#
#PBS -N mf2-rebuild
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l walltime=00:20:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/src/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-mpi-mf2}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MF2 binary rebuild — Gadi SPR (icpx + OpenMPI)"
echo "║  Source:  ${SRC_DIR}"
echo "║  Build:   ${BUILD_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Module load ───────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load cmake/3.31.6           2>/dev/null || true
    module load openmpi/4.1.7          2>/dev/null || true
    module load intel-compiler-llvm    2>/dev/null || true
    module load binutils/2.44          2>/dev/null || true
    module load eigen/3.3.7            2>/dev/null || true
    module load boost/1.84.0           2>/dev/null || true
fi

# Validate compiler is available
if command -v icpx >/dev/null 2>&1; then
    echo "[rebuild] Compiler: $(icpx --version 2>&1 | head -1)"
    export OMPI_CC="$(command -v icx)"
    export OMPI_CXX="$(command -v icpx)"
else
    echo "ERROR: icpx not found after module load." >&2; exit 1
fi
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found." >&2; exit 1; }
echo "[rebuild] MPI: $(mpirun --version 2>&1 | head -1)"
echo "[rebuild] cmake: $(cmake --version | head -1)"

# ── Confirm source directory exists and has our fix ───────────────────
[[ -d "${SRC_DIR}" ]] || { echo "ERROR: source dir ${SRC_DIR} not found." >&2; exit 2; }
[[ -d "${BUILD_DIR}" ]] || { echo "ERROR: build dir ${BUILD_DIR} not found." >&2; exit 2; }

# Verify our fix is in the source (canary string from the fix block)
if grep -q "Fix pre-gather checkpoint corruption" "${SRC_DIR}/main/phylotesting.cpp"; then
    echo "[rebuild] ✓ Pre-gather fix confirmed in phylotesting.cpp"
else
    echo "ERROR: Pre-gather fix NOT found in phylotesting.cpp — wrong source?" >&2
    exit 3
fi

# ── Stash the old binary ──────────────────────────────────────────────
OLD_BIN="${BUILD_DIR}/iqtree3-mpi"
if [[ -x "${OLD_BIN}" ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    BACKUP="${OLD_BIN}.bak_${STAMP}"
    cp "${OLD_BIN}" "${BACKUP}"
    echo "[rebuild] Old binary backed up to $(basename "${BACKUP}")"
fi

# ── Build ─────────────────────────────────────────────────────────────
echo "[rebuild] Building with gmake -j8 iqtree3 ..."
BUILD_LOG="${BUILD_DIR}/rebuild_${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}.log"
START_EPOCH=$(date +%s)

cd "${BUILD_DIR}"
/bin/gmake -j8 iqtree3 2>&1 | tee "${BUILD_LOG}"
BUILD_RC=${PIPESTATUS[0]}

END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

if [[ "${BUILD_RC}" -ne 0 ]]; then
    echo "ERROR: build failed (rc=${BUILD_RC}) after ${WALL}s — see ${BUILD_LOG}" >&2
    exit "${BUILD_RC}"
fi

# ── Verify the new binary ─────────────────────────────────────────────
NEW_BIN="${BUILD_DIR}/iqtree3-mpi"
if [[ ! -x "${NEW_BIN}" ]]; then
    echo "ERROR: build claimed success but binary ${NEW_BIN} not found." >&2; exit 4
fi

BIN_MTIME="$(stat -c '%Y' "${NEW_BIN}")"
if [[ "${BIN_MTIME}" -lt "${START_EPOCH}" ]]; then
    echo "ERROR: binary mtime (${BIN_MTIME}) < build start (${START_EPOCH}) — not rebuilt." >&2
    exit 5
fi

echo "[rebuild] ✓ Binary updated: ${NEW_BIN}"
echo "[rebuild]   Size:  $(du -h "${NEW_BIN}" | cut -f1)"
echo "[rebuild]   mtime: $(stat -c '%y' "${NEW_BIN}")"

# Confirm ELF has libmpi linked
if readelf -d "${NEW_BIN}" 2>/dev/null | grep -q 'NEEDED.*libmpi'; then
    echo "[rebuild] ✓ libmpi: CONFIRMED in ELF dynamic section"
else
    echo "WARNING: libmpi not found in ELF dynamic section of rebuilt binary" >&2
fi

echo "[rebuild] Build complete in ${WALL}s."
echo ""
echo "Next step — submit verification run:"
echo "  cd ~/setonix-iq && qsub tiers/verify_mf2_fix_np4.sh"
echo "  (Also qsub tiers/verify_mf2_fix_np2.sh for 2-node sanity check)"
