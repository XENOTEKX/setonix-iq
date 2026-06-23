#!/bin/bash
# run_prod_cut_validate_v100.sh — validate the gpu-kernel-prod cut:
#  (1) prod GPU build compiles after the 929-line harness strip + runs --ctf with
#      CLEAN production output (no [JOLT]/[CTF detector] spam), banner correct;
#  (2) prod CPU build (IQTREE_GPU=OFF) disables --ctf -> standard MF, SIMD banner;
#  (3) dev branch still compiles with JOLT_DEBUG_BUILD=ON (the harness is intact there).
# Builds into FRESH dirs so the running parity job's build-gpu-on is untouched.
#
#PBS -N prodcut-val
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
SRC=/scratch/rc29/as1708/iqtree3-gpu
ALN=$SRC/example/example.phy
WORK=$SRC/prodcut_val_${PBS_JOBID:-local}
mkdir -p "$WORK"
PON=$SRC/build-prod-on; POFF=$SRC/build-prod-off; DDBG=$SRC/build-dev-dbg
module load cmake/3.24.2 gcc/12.2.0 cuda/12.5.1 eigen/3.3.7 boost/1.84.0 2>/dev/null
export CC=$(command -v gcc) CXX=$(command -v g++)
export LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NVCC=$(command -v nvcc); GXX=$(command -v g++); EIG=/apps/eigen/3.3.7/include/eigen3; BR=/apps/boost/1.84.0
cfg_gpu() { cmake "$SRC" -DCMAKE_BUILD_TYPE=Release -DIQTREE_GPU=ON "$@" \
    -DCMAKE_CUDA_HOST_COMPILER="$GXX" -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DEIGEN3_INCLUDE_DIR="$EIG" -DBOOST_ROOT="$BR" -DBoost_NO_SYSTEM_PATHS=ON; }

echo "════ prod-cut validate $(hostname) $(date -Iseconds) ════"
echo "branch: $(cd $SRC && git rev-parse --abbrev-ref HEAD) @ $(cd $SRC && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

# ── (1) PROD GPU build (the strip must compile) ──
echo; echo "──── build PROD GPU (IQTREE_GPU=ON, JOLT_DEBUG_BUILD=OFF) ────"
mkdir -p "$PON"; ( cd "$PON" && cfg_gpu -DJOLT_DEBUG_BUILD=OFF > "$WORK/cfg_prod_on.log" 2>&1 && make -j12 > "$WORK/make_prod_on.log" 2>&1 ); RC_PON=$?
echo "  exit=$RC_PON (last 3):"; tail -3 "$WORK/make_prod_on.log" | sed 's/^/    /'

# ── (2) PROD CPU build ──
echo; echo "──── build PROD CPU (IQTREE_GPU=OFF) ────"
mkdir -p "$POFF"; ( cd "$POFF" && cmake "$SRC" -DCMAKE_BUILD_TYPE=Release -DIQTREE_GPU=OFF -DEIGEN3_INCLUDE_DIR="$EIG" -DBOOST_ROOT="$BR" -DBoost_NO_SYSTEM_PATHS=ON > "$WORK/cfg_prod_off.log" 2>&1 && make -j12 > "$WORK/make_prod_off.log" 2>&1 ); RC_POFF=$?
echo "  exit=$RC_POFF (last 3):"; tail -3 "$WORK/make_prod_off.log" | sed 's/^/    /'

# ── (3) DEV harness compiles with JOLT_DEBUG_BUILD=ON ──
echo; echo "──── build DEV GPU JOLT_DEBUG_BUILD=ON (harness must still compile) ────"
( cd "$SRC" && git checkout -q gpu-kernel-dev )
mkdir -p "$DDBG"; ( cd "$DDBG" && cfg_gpu -DJOLT_DEBUG_BUILD=ON > "$WORK/cfg_dev_dbg.log" 2>&1 && make -j12 > "$WORK/make_dev_dbg.log" 2>&1 ); RC_DDBG=$?
echo "  exit=$RC_DDBG (last 3):"; tail -3 "$WORK/make_dev_dbg.log" | sed 's/^/    /'
( cd "$SRC" && git checkout -q gpu-kernel-prod )    # restore

BPON=$PON/iqtree3; BPOFF=$POFF/iqtree3
[ -x "$BPON" ] && echo "  prod GPU md5: $(md5sum "$BPON"|cut -d' ' -f1)"

run() { local tag="$1"; shift; ( cd "$WORK" && "$@" ) > "$WORK/$tag.out" 2>&1; echo "  $tag exit=$?"; }
echo; echo "──── smoke ────"
[ -x "$BPON" ]  && run pT_ctf  "$BPON"  --ctf -s "$ALN" --ctf-subsample 300 --ctf-topk 2 -nt 4 -pre "$WORK/pt" -redo
[ -x "$BPOFF" ] && run pCpu_ctf "$BPOFF" --ctf -s "$ALN" -nt 4 -pre "$WORK/pc" -redo

echo; echo "════════ PROD-CUT SUMMARY ════════"
echo "  builds: prod-GPU=$RC_PON  prod-CPU=$RC_POFF  dev-DEBUG=$RC_DDBG  (0=ok)"
chk(){ grep -qE "$2" "$WORK/$1.out" 2>/dev/null && echo "  PASS $1: $3" || echo "  FAIL $1: $3"; }
chk pT_ctf 'Kernel:  JOLT \+ CTF'  "prod banner JOLT + CTF"
chk pT_ctf 'Best-fit model:'       "prod --ctf produced best-fit"
grep -q 'SEGMENTATION FAULT' "$WORK/pT_ctf.out" 2>/dev/null && echo "  FAIL pT_ctf: SEGFAULT" || echo "  PASS pT_ctf: no segfault"
echo "  prod CLEAN output: [JOLT] lines=$(grep -c '\[JOLT\]' "$WORK/pT_ctf.out" 2>/dev/null) [CTF detector] lines=$(grep -c '\[CTF detector\]' "$WORK/pT_ctf.out" 2>/dev/null) (expect 0/0)"
chk pCpu_ctf 'Kernel:  (AVX|SSE|x86)'                 "prod CPU build: SIMD banner (not JOLT)"
chk pCpu_ctf 'WITHOUT GPU support|standard ModelFinder|Best-fit model:' "prod CPU --ctf -> standard MF"
echo "  prod best:  $(grep -m1 'Best-fit model:' "$WORK/pT_ctf.out" 2>/dev/null)"
echo "  cpu  best:  $(grep -m1 'Best-fit model:' "$WORK/pCpu_ctf.out" 2>/dev/null)"
echo "logs: $WORK"; echo "DONE $(date -Iseconds)"
OVERALL=0; for r in $RC_PON $RC_POFF $RC_DDBG; do [ "$r" -ne 0 ] && OVERALL=1; done; exit $OVERALL
