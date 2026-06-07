#!/bin/bash
# run_grad_gpu_a100.sh — Phase G.0 GPU branch-length gradient on an A100-80GB (dgxa100).
# The AA-100K LG+G4 gradient needs partialsBufferCount=402 (so the GPU setPartials root-seed index 397
# is < partialsBufferCount; see bug 15). On the GPU, AA is padded 20→32 states ⇒ 402 partials ≈ 41 GB,
# which does NOT fit a 32 GB V100 but fits the 80 GB A100. Tests: (a) the beagleSetPartials root-seed
# workaround for CUDA's stubbed setRootPrePartials, (b) BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO for 20-state
# pre-order, (c) FD self-check + gradient timing.
#PBS -N g0-grad-a100
#PBS -P dx61
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load beagle-lib/4.0.1 2>/dev/null || true
module load cuda/12.6.2 2>/dev/null || true
module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
export LD_LIBRARY_PATH=/apps/beagle-lib/4.0.1/lib:${LD_LIBRARY_PATH:-}
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_derisk.cpp
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_derisk_a100
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
echo "════════ build (in-job) ════════"
icpx -O2 -std=c++17 "$SRC" -o "$BIN" -I/apps/beagle-lib/4.0.1/include/libhmsbeagle-1 -L/apps/beagle-lib/4.0.1/lib -lhmsbeagle
RC=$?; echo "build exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "════════ G.0 GPU gradient on $(hostname) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
run(){ echo; echo "──────── $1 ────────"; shift; "$BIN" "$ALN" "$TREE" "$@"; echo "[exit=$?]"; }
# CPU first (re-confirms FD PASS holds after the matrixBufferCount change — A100 node has the GPU/OpenCL
# stack so the CPU plugin's createInstance enumeration won't crash; see bug 14), then GPU (manual transpose).
run "MODEL g4 — CPU gradient FD-check (regression guard)" cpu g4 3 0 1
run "MODEL g4 — GPU gradient (FD + timing) on A100-80GB"  gpu g4 30 0 1
echo
echo "════════ DONE ════════"
