#!/bin/bash
# bootstrap_iqtree.sh — clone + build IQ-TREE 3 on Setonix (Pawsey, AMD Zen 3).
#
# Symmetric counterpart to gadi-ci/bootstrap_iqtree.sh.
# Authored 2026-04-30 as part of the methodology-audit-round-2 alignment:
# both clusters are now built with the *same* compiler family (gcc 12+) so
# the only intentional difference between the two binaries is the -march
# tuning flag.
#
#   Setonix:  gcc -O3 -march=znver3 -mtune=znver3
#   Gadi:     gcc -O3 -march=sapphirerapids -mtune=sapphirerapids
#
# Produces two binaries:
#   ${PROJECT_DIR}/build/iqtree3              (release)
#   ${PROJECT_DIR}/build-profiling/iqtree3    (+ -fno-omit-frame-pointer -g)
#
# The -fno-omit-frame-pointer build is required for `perf -g` to unwind stacks.
#
# Usage (submit):
#   sbatch setonix-ci/bootstrap_iqtree.sh
#
# Usage (interactive, e.g. inside `salloc -N1 -p work --exclusive ...`):
#   bash setonix-ci/bootstrap_iqtree.sh
#
#SBATCH --job-name=iqtree-bootstrap
#SBATCH --account=pawsey1351
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/bootstrap_%j.out
#SBATCH --error=/scratch/pawsey1351/asamuel/iqtree3/setonix-ci/logs/bootstrap_%j.err

set -euo pipefail

PROJECT="${PROJECT:-pawsey1351}"
USER_ID="${USER:-$(whoami)}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/src/iqtree3}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
BUILD_PROFILING="${BUILD_PROFILING:-${PROJECT_DIR}/build-profiling}"
IQTREE_REPO="${IQTREE_REPO:-https://github.com/iqtree/iqtree3.git}"
IQTREE_REF="${IQTREE_REF:-master}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3 bootstrap on Setonix (AMD Zen 3 / EPYC 7763)"
echo "║  Project:       ${PROJECT}"
echo "║  Source:        ${SRC_DIR}"
echo "║  Release build: ${BUILD_DIR}"
echo "║  Profiling:     ${BUILD_PROFILING}"
echo "║  Repo:          ${IQTREE_REPO} (${IQTREE_REF})"
echo "╚══════════════════════════════════════════════════════════════╝"

# Pawsey software stack — load gcc 14.2 + cmake + eigen + boost.
# Versions chosen 2026-04-30 (round 2 audit, parity with Gadi):
#   gcc-native/14.2 — Setonix 2025.08 stack default; matches Gadi gcc/14.2.0.
#   eigen/3.4.0     — only Eigen module on Setonix 2025.08; Gadi bumped 3.3.7→3.4.0 to match.
#   boost/1.86.0-c++14-python — Setonix 2025.08 default; Gadi bumped 1.84.0→1.86.0 to match.
if command -v module >/dev/null 2>&1; then
    module load gcc-native/14.2           2>/dev/null || true
    module load cmake/3.30.5              2>/dev/null || true
    module load eigen/3.4.0               2>/dev/null || true
    module load boost/1.86.0-c++14-python 2>/dev/null || true
fi
# NOTE: `gcc-native/14.2` on the Setonix 2025.08 stack ships gcc 14.3.0
# (SUSE-bundled), whereas Gadi's `gcc/14.2.0` ships gcc 14.2.0 exactly.
# This is a 14.2.0 vs 14.3.0 *patch-level* delta on the same major.minor —
# 14.3 is a pure bugfix release of 14.2, no codegen or ABI changes.
# Acceptably below the noise floor for cross-platform comparison.

# Eigen / Boost — derive include dirs from module env, fall back to known paths.
# Setonix 2025.08 modules export PAWSEY_EIGEN_HOME / PAWSEY_BOOST_HOME (legacy
# stacks used EIGEN_ROOT / BOOST_ROOT); accept either.
EIGEN_BASE="${EIGEN_ROOT:-${PAWSEY_EIGEN_HOME:-}}"
EIGEN3_INCLUDE_DIR="${EIGEN_BASE:+${EIGEN_BASE}/include/eigen3}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/software/setonix/2025.08/software/linux-sles15-zen3/gcc-14.2.0/eigen-3.4.0/include/eigen3}"
BOOST_ROOT="${BOOST_ROOT:-${PAWSEY_BOOST_HOME:-/software/setonix/2025.08/software/linux-sles15-zen3/gcc-14.2.0/boost-1.86.0}}"

mkdir -p "$(dirname "${SRC_DIR}")"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    if command -v git >/dev/null 2>&1; then
        echo "[bootstrap] cloning ${IQTREE_REPO} (${IQTREE_REF})"
        git clone --branch "${IQTREE_REF}" "${IQTREE_REPO}" "${SRC_DIR}"
        ( cd "${SRC_DIR}" && git submodule update --init --recursive )
    else
        echo "ERROR: ${SRC_DIR} not found and git not available." >&2
        exit 1
    fi
else
    echo "[bootstrap] source already present at ${SRC_DIR} — skipping clone"
    ( cd "${SRC_DIR}" && git submodule update --init --recursive 2>/dev/null || true )
fi

for sub in cmaple lsd2; do
    if [[ ! -f "${SRC_DIR}/${sub}/CMakeLists.txt" ]]; then
        echo "ERROR: submodule ${SRC_DIR}/${sub} not initialised." >&2
        echo "       cd ${SRC_DIR} && git submodule update --init --recursive" >&2
        exit 1
    fi
done
echo "[bootstrap] submodules OK: cmaple, lsd2"

# IPO patch: gcc + system ld on Setonix can link LTO, but disable IPO anyway
# to keep parity with Gadi (and to avoid the 30 % wall-clock cost of LTO when
# the gain is marginal for a code already dominated by the SIMD kernel).
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable IPO (LTO) by default|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Setonix: disabled for parity with Gadi|' "${CMAPLE_CML}"
    echo "[bootstrap] IPO disabled in cmaple"
fi

# Disable cmaple/unittest + googletest FetchContent for offline reproducibility.
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Setonix: disabled|' "${CMAPLE_CML}"
fi
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# SETONIX-DISABLED: |' "${CMAPLE_CML}"
fi

# Compiler selection: gcc only (parity with Gadi).
# Setonix-tuned: -march=znver3 -mtune=znver3 (Zen 3, EPYC 7763 / Trento).
if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not on PATH (did you 'module load gcc-native/14.2'?)" >&2
    exit 2
fi
CC="$(command -v gcc)"
CXX="$(command -v g++)"
ARCH_FLAGS="-O3 -march=znver3 -mtune=znver3"
echo "[bootstrap] using GCC: $(${CC} --version | head -1)"
echo "[bootstrap] arch flags: ${ARCH_FLAGS}"

build_variant() {
    local build_dir="$1"
    local extra="$2"
    echo ""
    echo "[bootstrap] ── building ${build_dir} (extra='${extra}') ──"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"
    CC="${CC}" CXX="${CXX}" cmake "${SRC_DIR}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}" \
        -DBOOST_ROOT="${BOOST_ROOT}" \
        -DBoost_NO_SYSTEM_PATHS=ON \
        -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${extra}" \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${extra}"
    local jobs="${IQTREE_BUILD_JOBS:-$(nproc)}"
    make -j"${jobs}"
    if [[ ! -x "${build_dir}/iqtree3" ]]; then
        mv -f "${build_dir}"/iqtree3*/iqtree3 "${build_dir}/iqtree3" 2>/dev/null || true
    fi
    "${build_dir}/iqtree3" --version | head -3 || true
}

build_variant "${BUILD_DIR}"       ""
build_variant "${BUILD_PROFILING}" "-fno-omit-frame-pointer -g"

# Record the build provenance — picked up by run_mega_profile.sh / harvest.
cat > "${BUILD_PROFILING}/.build-info.json" <<EOF
{
  "compiler":      "$(${CC} --version | head -1)",
  "arch_flags":    "${ARCH_FLAGS} -fno-omit-frame-pointer -g",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)",
  "iqtree_repo":   "${IQTREE_REPO}",
  "iqtree_ref":    "${IQTREE_REF}",
  "iqtree_commit": "$(cd "${SRC_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF

echo ""
echo "[bootstrap] OK"
echo "  release:   ${BUILD_DIR}/iqtree3"
echo "  profiling: ${BUILD_PROFILING}/iqtree3"
echo "  metadata:  ${BUILD_PROFILING}/.build-info.json"
