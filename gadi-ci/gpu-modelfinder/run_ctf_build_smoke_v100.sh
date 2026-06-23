#!/bin/bash
# run_ctf_build_smoke_v100.sh — Phase 6: rebuild both binaries with the native-CTF + production
# changes, then smoke-test --ctf / --jolt / --no-jolt and the CPU-parity build on example.phy.
#
#PBS -N ctf-build-smoke
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
ON=$SRC/build-gpu-on
OFF=$SRC/build-gpu-off
ALN=$SRC/example/example.phy
WORK=$SRC/ctf_smoke_${PBS_JOBID:-local}
mkdir -p "$WORK"

module load cmake/3.24.2 gcc/12.2.0 cuda/12.5.1 eigen/3.3.7 boost/1.84.0 2>/dev/null
export CC=$(command -v gcc) CXX=$(command -v g++)
export LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NVCC=$(command -v nvcc); GXX=$(command -v g++)
EIG=/apps/eigen/3.3.7/include/eigen3; BR=/apps/boost/1.84.0

echo "════ CTF build+smoke $(hostname) $(date -Iseconds) ════"
echo "src HEAD: $(cd $SRC && git rev-parse --short HEAD)  (+ uncommitted CTF changes)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

RC_ON_CFG=1 RC_ON_MAKE=1 RC_OFF_MAKE=1
# ── reconfigure + build GPU/production (IQTREE_GPU=ON, JOLT_DEBUG_BUILD=OFF) ──
echo; echo "──── build GPU (IQTREE_GPU=ON, JOLT_DEBUG_BUILD=OFF) ────"
( cd "$ON" && cmake "$SRC" -DCMAKE_BUILD_TYPE=Release -DIQTREE_GPU=ON -DJOLT_DEBUG_BUILD=OFF \
    -DCMAKE_CUDA_HOST_COMPILER="$GXX" -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DEIGEN3_INCLUDE_DIR="$EIG" -DBOOST_ROOT="$BR" -DBoost_NO_SYSTEM_PATHS=ON > "$WORK/cmake_on.log" 2>&1 )
RC_ON_CFG=$?; echo "  configure exit=$RC_ON_CFG"; tail -3 "$WORK/cmake_on.log" | sed 's/^/    /'
if [ $RC_ON_CFG -eq 0 ]; then
    ( cd "$ON" && make -j12 > "$WORK/make_on.log" 2>&1 ); RC_ON_MAKE=${PIPESTATUS[0]}
    echo "  make exit=$RC_ON_MAKE (last 6):"; tail -6 "$WORK/make_on.log" | sed 's/^/    /'
fi

# ── reconfigure + build CPU-parity (IQTREE_GPU=OFF) ──
echo; echo "──── build CPU parity (IQTREE_GPU=OFF) ────"
( cd "$OFF" && cmake "$SRC" -DCMAKE_BUILD_TYPE=Release -DIQTREE_GPU=OFF \
    -DEIGEN3_INCLUDE_DIR="$EIG" -DBOOST_ROOT="$BR" -DBoost_NO_SYSTEM_PATHS=ON > "$WORK/cmake_off.log" 2>&1
  make -j12 > "$WORK/make_off.log" 2>&1 ); RC_OFF_MAKE=$?
echo "  make exit=$RC_OFF_MAKE (last 4):"; tail -4 "$WORK/make_off.log" | sed 's/^/    /'

BON=$ON/iqtree3; BOFF=$OFF/iqtree3
[ -x "$BON" ] && echo "  GPU bin md5: $(md5sum "$BON"|cut -d' ' -f1)"
[ -x "$BOFF" ] && echo "  CPU bin md5: $(md5sum "$BOFF"|cut -d' ' -f1)"

# ════════════════════ smoke tests ════════════════════
run() { local tag="$1"; shift; echo; echo "──── $tag : $* ────"; ( cd "$WORK" && "$@" ) > "$WORK/$tag.out" 2>&1; echo "  exit=$?"; }

if [ -x "$BON" ]; then
  run T1_ctf   "$BON" --ctf -s "$ALN" --ctf-subsample 300 --ctf-topk 2 -nt 4 -pre "$WORK/t1" -redo
  run T2_jolt  "$BON" --jolt -s "$ALN" -m GTR+G4 -nt 4 -pre "$WORK/t2" -redo
  run T4_nojolt "$BON" --no-jolt -s "$ALN" -m GTR+G4 -nt 4 -pre "$WORK/t4" -redo
fi
[ -x "$BOFF" ] && run T3_cpu "$BOFF" -m MF -s "$ALN" -nt 4 -pre "$WORK/t3" -redo

echo; echo "════════ SMOKE SUMMARY ════════"
chk(){ grep -qE "$2" "$WORK/$1.out" 2>/dev/null && echo "  PASS $1: $3" || echo "  FAIL $1: $3"; }
chk T1_ctf   'Kernel:  JOLT \+ CTF'            "banner shows 'JOLT + CTF'"
chk T1_ctf   '^GPU:'                            "GPU info line present"
chk T1_ctf   'CTF \(coarse-to-fine\)'          "CTF orchestrator ran"
chk T1_ctf   'Best-fit model:'                  "CTF produced a best-fit model"
chk T2_jolt  'Kernel:  JOLT$'                   "banner shows 'JOLT' (not CTF)"
chk T2_jolt  '^GPU:'                            "GPU info line present under --jolt"
chk T4_nojolt 'Kernel:  (AVX|SSE|x86)'          "--no-jolt falls to CPU SIMD banner"
chk T3_cpu   'Best-fit model:'                  "CPU-parity build selects a model"
echo "  T1 best: $(grep -m1 'Best-fit model:' "$WORK/T1_ctf.out" 2>/dev/null)"
echo "  T3 best: $(grep -m1 'Best-fit model:' "$WORK/T3_cpu.out" 2>/dev/null)"
echo "  T1 [JOLT] engagements: $(grep -c '\[JOLT\] model' "$WORK/T1_ctf.out" 2>/dev/null)"
echo "  CTF coarse 'test up to N models': $(grep -m1 'will test up to' "$WORK/T1_ctf.out" 2>/dev/null)"
echo "logs: $WORK"; echo "DONE $(date -Iseconds)"
OVERALL=0; for rc in $RC_ON_CFG $RC_ON_MAKE $RC_OFF_MAKE; do [ "$rc" -ne 0 ] && OVERALL=1; done; exit $OVERALL
