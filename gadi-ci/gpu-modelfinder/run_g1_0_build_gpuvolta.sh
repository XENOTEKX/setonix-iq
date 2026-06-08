#!/bin/bash
# run_g1_0_build_gpuvolta.sh — Phase G.1.0 build-scaffold validation (in-tree CUDA).
#
# Validates the G.1.0 deliverable (gpu-modelfinder-design.md PART II §II.8):
#   option(IQTREE_GPU) + gated enable_language(CUDA) + iqtree_gpu .cu static lib
#   + #cmakedefine IQTREE_GPU + --gpu flag + a hello-world .cu launched from a
#   diag hook. Pure plumbing, NO numerics.
#
# Five tests, all in one gpuvolta job:
#   T1  IQTREE_GPU=OFF  builds cleanly (no CUDA), binary runs CPU ModelFinder.
#   T2  IQTREE_GPU=ON   configures (finds nvcc) + builds iqtree_gpu .cu + links.
#   T3  ON  binary WITHOUT --gpu runs the normal CPU path (unchanged behaviour).
#   T4  ON  binary WITH --gpu launches the kernel, prints "diagnostic PASSED",
#           clean cudaGetLastError, THEN completes the CPU analysis.
#   T5  OFF binary WITH --gpu prints "built WITHOUT GPU support" + completes CPU.
#   + behavioural identity: T3 lnL == T1 lnL (same gcc build; IQTREE_GPU inert).
#
# Toolchain (design §II.9 all-GCC host path): gcc/12.2.0 + cuda/12.5.1 + cmake/3.24.2.
# gpuvolta = Cascade Lake V100 → NO -march=sapphirerapids (would SIGILL); generic
# build, IQ-TREE's runtime ISA dispatch picks AVX/AVX512 safely.
#
#PBS -N g1-0-gpu-scaffold
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail

SRC=/scratch/rc29/as1708/iqtree3-gpu
BUILD_ON="${SRC}/build-gpu-on"
BUILD_OFF="${SRC}/build-gpu-off"
ALN="${SRC}/example/example.phy"
WORK=/scratch/rc29/as1708/iqtree3-gpu/g1_0_runs
mkdir -p "${WORK}"

# ── Modules (all-GCC host; no intel, no MPI) ────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load cmake/3.24.2 2>/dev/null || true
    module load gcc/12.2.0   2>/dev/null || true
    module load cuda/12.5.1  2>/dev/null || true
    module load eigen/3.3.7  2>/dev/null || true
    module load boost/1.84.0 2>/dev/null || true
fi
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
GXX="$(command -v g++)"
NVCC="$(command -v nvcc)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
EIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3
BOOST_ROOT=/apps/boost/1.84.0

echo "════════════════════ G.1.0 build scaffold — $(hostname) ════════════════════"
echo "date     : $(date -Iseconds)"
echo "src      : ${SRC}  (branch $(cd "${SRC}" && git branch --show-current 2>/dev/null), HEAD $(cd "${SRC}" && git rev-parse --short HEAD 2>/dev/null))"
echo "gcc      : $("${CXX}" --version | head -1)"
echo "nvcc     : $(${NVCC} --version 2>/dev/null | tail -2 | tr '\n' ' ')"
echo "cmake    : $(cmake --version | head -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo

if [ -z "${NVCC}" ]; then echo "FATAL: nvcc not on PATH after module load"; exit 2; fi

# ── cmaple Gadi offline-build patches (same as build_mf_iso.sh) ─────────────
# cmaple's CMake does FetchContent(googletest)+IPO+unittest, which fail on an
# offline compute node. Disable them (idempotent, guarded).
CMAPLE_CML="${SRC}/cmaple/CMakeLists.txt"
if [ -f "${CMAPLE_CML}" ]; then
    grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}" && \
        sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE).*|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled|' "${CMAPLE_CML}"
    grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}" && \
        sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled|' "${CMAPLE_CML}"
    grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}" && \
        sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
    echo "[cmaple] offline-build patches applied (IPO/unittest/googletest)"
fi

COMMON_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}"
    -DBOOST_ROOT="${BOOST_ROOT}"
    -DBoost_NO_SYSTEM_PATHS=ON
)
JOBS=12

find_bin() { find "$1" -maxdepth 2 -name 'iqtree3' -type f -executable 2>/dev/null | head -1; }

RC_CFG_ON=1; RC_MAKE_ON=1; RC_CFG_OFF=1; RC_MAKE_OFF=1
BIN_ON=""; BIN_OFF=""

# ════════════════════ BUILD 1: IQTREE_GPU=ON (risky path first) ════════════
echo; echo "──────── configure+build  IQTREE_GPU=ON  (${BUILD_ON}) ────────"
rm -rf "${BUILD_ON}" && mkdir -p "${BUILD_ON}" && cd "${BUILD_ON}" || { echo "FATAL: cannot enter ${BUILD_ON}"; exit 1; }
cmake "${SRC}" "${COMMON_ARGS[@]}" \
    -DIQTREE_GPU=ON \
    -DCMAKE_CUDA_HOST_COMPILER="${GXX}" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    > cmake_on.log 2>&1
RC_CFG_ON=$?
echo "  configure ON exit=${RC_CFG_ON} (tail:)"; tail -n 15 cmake_on.log | sed 's/^/    /'
grep -iE "CUDA compiler|CUDA arch|Build CUDA GPU" cmake_on.log | sed 's/^/    » /' || true
if [ ${RC_CFG_ON} -eq 0 ]; then
    make -j${JOBS} > make_on.log 2>&1
    RC_MAKE_ON=${PIPESTATUS[0]}
    echo "  make ON exit=${RC_MAKE_ON} (last 12 lines:)"; tail -n 12 make_on.log | sed 's/^/    /'
    BIN_ON="$(find_bin "${BUILD_ON}")"
    echo "  iqtree_gpu lib: $(find "${BUILD_ON}" -name 'libiqtree_gpu.a' 2>/dev/null | head -1)"
    echo "  binary ON: ${BIN_ON}"
else
    echo "  !! ON configure failed — see cmake_on.log; continuing to OFF build."
fi

# ════════════════════ BUILD 2: IQTREE_GPU=OFF (baseline) ═══════════════════
echo; echo "──────── configure+build  IQTREE_GPU=OFF  (${BUILD_OFF}) ────────"
rm -rf "${BUILD_OFF}" && mkdir -p "${BUILD_OFF}" && cd "${BUILD_OFF}" || { echo "FATAL: cannot enter ${BUILD_OFF}"; exit 1; }
cmake "${SRC}" "${COMMON_ARGS[@]}" -DIQTREE_GPU=OFF > cmake_off.log 2>&1
RC_CFG_OFF=$?
echo "  configure OFF exit=${RC_CFG_OFF}"
if [ ${RC_CFG_OFF} -eq 0 ]; then
    make -j${JOBS} > make_off.log 2>&1
    RC_MAKE_OFF=${PIPESTATUS[0]}
    echo "  make OFF exit=${RC_MAKE_OFF} (last 8 lines:)"; tail -n 8 make_off.log | sed 's/^/    /'
    BIN_OFF="$(find_bin "${BUILD_OFF}")"
    echo "  binary OFF: ${BIN_OFF}"
fi

# ════════════════════ Linkage / symbol evidence ═══════════════════════════
echo; echo "──────── linkage evidence ────────"
if [ -n "${BIN_ON}" ]; then
    echo "  ON  ldd libcudart : $(ldd "${BIN_ON}" 2>/dev/null | grep -i cudart || echo 'NONE (unexpected!)')"
    echo "  ON  symbol diag   : $(nm "${BIN_ON}" 2>/dev/null | grep -c iqtree_gpu_diag) (strings: $(strings "${BIN_ON}" 2>/dev/null | grep -c 'diagnostic PASSED'))"
fi
if [ -n "${BIN_OFF}" ]; then
    echo "  OFF ldd libcudart : $(ldd "${BIN_OFF}" 2>/dev/null | grep -i cudart || echo 'NONE (correct — OFF has no CUDA)')"
fi

# ════════════════════ Runtime tests ═══════════════════════════════════════
lnl_of() { grep -hoE "BEST SCORE FOUND : -?[0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE -- "-?[0-9.]+$"; }
runit()  { # $1=tag $2=binary $3...=extra args
    local tag="$1"; local bin="$2"; shift 2
    local out="${WORK}/${tag}.stdout"
    echo; echo "──── TEST ${tag}: ${bin} $* ────"
    ( cd "${WORK}" && "${bin}" -s "${ALN}" -m TEST -nt 1 -redo -pre "${tag}" "$@" ) > "${out}" 2>&1
    local rc=$?
    echo "    exit=${rc}  lnL=$(lnl_of "${out}")"
    return ${rc}
}

T3_LNL=""; T1_LNL=""; T4_LNL=""
RC_T3=1; RC_T4=1; RC_T1=1; RC_T5=1
if [ -n "${BIN_ON}" ]; then
    runit "T3_on_nogpu" "${BIN_ON}";          RC_T3=$?; T3_LNL="$(lnl_of "${WORK}/T3_on_nogpu.stdout")"
    runit "T4_on_gpu"   "${BIN_ON}" --gpu;     RC_T4=$?; T4_LNL="$(lnl_of "${WORK}/T4_on_gpu.stdout")"
    echo "    GPU banner:"; grep -E "^GPU:" "${WORK}/T4_on_gpu.stdout" | sed 's/^/      /' || echo "      (no GPU: lines!)"
fi
if [ -n "${BIN_OFF}" ]; then
    runit "T1_off_run"  "${BIN_OFF}";          RC_T1=$?; T1_LNL="$(lnl_of "${WORK}/T1_off_run.stdout")"
    runit "T5_off_gpu"  "${BIN_OFF}" --gpu;     RC_T5=$?
    echo "    OFF --gpu message:"; grep -iE "without GPU support" "${WORK}/T5_off_gpu.stdout" | sed 's/^/      /' || echo "      (message missing!)"
fi

# ════════════════════ PASS/FAIL SUMMARY ═══════════════════════════════════
echo; echo "════════════════════════ G.1.0 SUMMARY ════════════════════════"
pf() { [ "$1" -eq 0 ] 2>/dev/null && echo "PASS" || echo "FAIL"; }
echo "  T2  ON  configure        : $(pf ${RC_CFG_ON})  (exit ${RC_CFG_ON})"
echo "  T2  ON  build+link        : $(pf ${RC_MAKE_ON})  (exit ${RC_MAKE_ON})"
echo "  T1  OFF configure         : $(pf ${RC_CFG_OFF})  (exit ${RC_CFG_OFF})"
echo "  T1  OFF build             : $(pf ${RC_MAKE_OFF})  (exit ${RC_MAKE_OFF})"
echo "  T3  ON  no-gpu CPU run    : $(pf ${RC_T3})  lnL=${T3_LNL}"
echo "  T4  ON  --gpu diag+run    : $(pf ${RC_T4})  banner=$(grep -qE 'diagnostic PASSED' "${WORK}/T4_on_gpu.stdout" 2>/dev/null && echo PASSED || echo MISSING)"
echo "  T1  OFF CPU run           : $(pf ${RC_T1})  lnL=${T1_LNL}"
echo "  T5  OFF --gpu guard msg   : $(pf ${RC_T5})  msg=$(grep -qiE 'without GPU support' "${WORK}/T5_off_gpu.stdout" 2>/dev/null && echo OK || echo MISSING)"
if [ -n "${T1_LNL}" ] && [ "${T1_LNL}" = "${T3_LNL}" ]; then
    echo "  BEHAVIOURAL IDENTITY      : PASS  (OFF lnL ${T1_LNL} == ON-no-gpu lnL ${T3_LNL})"
else
    echo "  BEHAVIOURAL IDENTITY      : CHECK (OFF=${T1_LNL} vs ON-no-gpu=${T3_LNL})"
fi
if [ -n "${T3_LNL}" ] && [ "${T3_LNL}" = "${T4_LNL}" ]; then
    echo "  GPU-HOOK NON-INTERFERENCE : PASS  (ON-no-gpu lnL ${T3_LNL} == ON --gpu lnL ${T4_LNL})"
else
    echo "  GPU-HOOK NON-INTERFERENCE : CHECK (T3=${T3_LNL} vs T4=${T4_LNL})"
fi
echo "════════════════════════════════════════════════════════════════"
echo "logs: ${BUILD_ON}/cmake_on.log ${BUILD_ON}/make_on.log ${WORK}/*.stdout"
echo "DONE $(date -Iseconds)"

# Reflect test results in the PBS exit code (so re-runs / CI can detect failure).
OVERALL=0
for rc in ${RC_CFG_ON} ${RC_MAKE_ON} ${RC_CFG_OFF} ${RC_MAKE_OFF} ${RC_T3} ${RC_T4} ${RC_T1} ${RC_T5}; do
    [ "${rc}" -ne 0 ] && OVERALL=1
done
exit ${OVERALL}
