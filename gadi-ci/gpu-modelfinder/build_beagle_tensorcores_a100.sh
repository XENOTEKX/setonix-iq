#!/bin/bash
# build_beagle_tensorcores_a100.sh — PHASE 1 of the JOLT-vs-BEAGLE-4.0 apples-to-apples benchmark.
# Builds the tensor-core BEAGLE (github beagle-dev/beagle-lib branch tensor-cores, commit dd962d48 — the exact
# code behind Gangavarapu et al. 2026 Syst Biol) on an A100 (sm_80 = 3rd-gen FP64 tensor cores, the paper's card),
# then smoke-tests the synthetictest client which is our matched harness: --states 20 (AA), --doubleprecision (FP64,
# parity), --calcderivs (gradient), --rates 4 (gamma), --sites, --reps, --fulltiming. PHASE 2 (separate) runs the
# matched AA workload on tensor-core vs CUDA-core resources and compares per-eval time + lnL parity to JOLT.
# Submit: qsub gadi-ci/gpu-modelfinder/build_beagle_tensorcores_a100.sh
#PBS -N beagle-tc-build
#PBS -P dx61
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cuda/12.5.1  2>/dev/null || true
export CC="$(command -v gcc)" CXX="$(command -v g++)"
SRC=/scratch/rc29/as1708/beagle-tensorcores
BD="$SRC/build-tc"
echo "════════ BEAGLE tensor-core build — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$SRC"; git log --oneline -1
nvcc --version | tail -2

echo "──── configure (CUDA + TENSOR CORES, sm_80 A100 + sm_90 H200; OpenCL/JNI off to shrink failure surface) ────"
rm -rf "$BD"; mkdir -p "$BD"; cd "$BD"
cmake .. \
  -DBUILD_CUDA=ON -DBEAGLE_TENSOR_CORES=ON \
  -DBUILD_OPENCL=OFF -DBUILD_JNI=OFF -DBUILD_OPENMP=ON -DBUILD_SSE=ON \
  -DBEAGLE_BENCHMARK=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80;90" \
  -DCMAKE_INSTALL_PREFIX="$BD/install" > "$BD/cmake.log" 2>&1
RC=$?; echo "  cmake exit=$RC"; tail -8 "$BD/cmake.log" | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "CMAKE FAILED — see cmake.log"; grep -iE "error|not found|tensor" "$BD/cmake.log" | head; exit 1; }

echo "──── build ────"
make -j16 > "$BD/make.log" 2>&1; RC=$?
echo "  make exit=$RC"; tail -6 "$BD/make.log" | sed 's/^/    /'
make install >> "$BD/make.log" 2>&1 || true
[ $RC -ne 0 ] && { echo "MAKE FAILED — first errors:"; grep -iE "error:" "$BD/make.log" | head -12; exit 1; }

echo "──── artifacts ────"
find "$BD" -iname "libhmsbeagle*cuda*so*" -o -iname "synthetictest" 2>/dev/null | sed 's/^/    /'
export LD_LIBRARY_PATH="$BD/install/lib:$BD/libhmsbeagle:$LD_LIBRARY_PATH"
ST=$(find "$BD" -name synthetictest -type f | head -1)
[ -z "$ST" ] && { echo "synthetictest not built"; exit 1; }

echo "──── smoke: list GPU resources (expect CUDA device; tensor-core path compiled in) ────"
"$ST" --resourcelist --states 20 --sites 500 --reps 1 --doubleprecision 2>&1 | grep -iE "Resource|GPU|CUDA|NVIDIA|Tensor" | head

echo "──── smoke: AA-20 FP64 lnL+gradient, 2000 sites, GPU resource (rsrc 1), full timing ────"
"$ST" --states 20 --sites 2000 --rates 4 --reps 10 --doubleprecision --calcderivs \
      --postorder --rsrc 1 --fulltiming 2>&1 | tail -30

echo "════════ BUILD+SMOKE DONE $(date -Iseconds) ════════"
echo "  NEXT (phase 2): matched AA workload tensor-core vs cuda-core resource + lnL/grad parity + per-eval time vs JOLT k1_node/kj_pre"
