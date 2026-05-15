#!/bin/bash
# build_cpu_bench_clx.sh — Build IQ-TREE3 cpu_opt_merge with ICX + AVX-512 + -march=cascadelake.
#
# Source:  /scratch/dx61/sa0557/iqtree2/cpu_opt_merge/iqtree3  (branch cpu_opt_merge)
# Output:  /scratch/dx61/as1708/cpu_bench/build-intel-clx/iqtree3
#
# Run BEFORE the four normal-queue cpu_bench scripts.  SPR scripts use the
# pre-built intel-vanila binary and do not require this build step.
#
#PBS -N iq-build-cpu-clx
#PBS -P dx61
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=50GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

SOURCE_DIR="/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/iqtree3"
BUILD_DIR="/scratch/dx61/as1708/cpu_bench/build-intel-clx"
EIGEN_DIR="/apps/eigen/3.3.7/include/eigen3"

echo "=== build_cpu_bench_clx: ICX + AVX-512 + march=cascadelake ==="
echo "Source:  ${SOURCE_DIR}"
echo "Output:  ${BUILD_DIR}/iqtree3"
echo ""

module load intel-compiler-llvm/2024.2.1
module load openmpi/4.1.5
module load boost/1.84.0

export CC=icx
export CXX=icpx

# Always start from a clean build dir to avoid stale cmake cache
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DIQTREE_FLAGS="avx512" \
    -DCMAKE_CXX_FLAGS="-O3 -march=cascadelake -fno-omit-frame-pointer" \
    -DCMAKE_C_FLAGS="-O3 -march=cascadelake -fno-omit-frame-pointer" \
    -DEIGEN3_INCLUDE_DIR="${EIGEN_DIR}" \
    -DUSE_CMAPLE=OFF \
    2>&1 | tee cmake.log

# Fix link ordering: pll/libpllavx.a defines symbols referenced by pll/libpll.a but the
# cmake link order causes a single-pass linker to see pllavx before pll pulls in its objects.
# Wrap all static archive arguments in --start-group/--end-group for multi-pass resolution.
LINK_TXT="${BUILD_DIR}/CMakeFiles/iqtree3.dir/link.txt"
if [[ -f "${LINK_TXT}" ]]; then
    python3 - "${LINK_TXT}" <<'PYEOF'
import sys, re
with open(sys.argv[1]) as f:
    cmd = f.read()
archives = list(re.finditer(r'\S+\.a\b', cmd))
if archives:
    start, end = archives[0].start(), archives[-1].end()
    cmd = cmd[:start] + '-Wl,--start-group ' + cmd[start:end] + ' -Wl,--end-group' + cmd[end:]
    with open(sys.argv[1], 'w') as f:
        f.write(cmd)
    print(f"link.txt: wrapped {len(archives)} archives in --start-group/--end-group")
PYEOF
fi

make -j8 iqtree3 2>&1 | tee build.log

if [[ ! -x "${BUILD_DIR}/iqtree3" ]]; then
    echo "ERROR: build failed — binary not found" >&2
    exit 1
fi

echo ""
echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/iqtree3"
"${BUILD_DIR}/iqtree3" --version 2>&1 | head -3 || true
ldd "${BUILD_DIR}/iqtree3" | grep -E "libiomp|libgomp|mkl" || true
