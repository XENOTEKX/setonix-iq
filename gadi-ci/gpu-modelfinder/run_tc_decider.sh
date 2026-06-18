#!/bin/bash
# run_tc_decider.sh — Part 12 / T.0 KILL-SWITCH DECIDER.
# Builds tree/gpu/tc_decider.cu (FP64 wmma-double DMMA matvec vs JOLT's scalar matvec, 20->32 pad) and runs it at
# nptn = {100K, 1M}, ncat=4, on whatever GPU the queue gives. Prints per-dim speedup (scalar/dmma) + parity (rel vs
# scalar oracle, gate 1e-12) + a VERDICT line. Decides whether the FP64 tensor-core lever is worth pursuing in-tree.
# GATE (part12 sec XII.3): proceed ONLY if DMMA >= ~1.3x at nptn=1M with rel<=1e-12 on H200 (the 1M/10M deploy card).
#
# Submit BOTH (gate card = H200; A100 for comparison):
#   qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=32GB -l walltime=00:20:00 -v LBL=h200 \
#        -l storage=scratch/dx61+scratch/rc29 -l wd gadi-ci/gpu-modelfinder/run_tc_decider.sh
#   qsub -q dgxa100   -l ngpus=1 -l ncpus=16 -l mem=32GB -l walltime=00:20:00 -v LBL=a100 \
#        -l storage=scratch/dx61+scratch/rc29 -l wd gadi-ci/gpu-modelfinder/run_tc_decider.sh
#PBS -N tcdecide
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
LBL="${LBL:-gpu}"
SRC=/scratch/rc29/as1708/iqtree3-gpu
WB="$SRC/tc_decider_${LBL}"; mkdir -p "$WB"; cd "$WB"

echo "════════ T.0 tc_decider — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true

echo; echo "──── build (sm_80 + sm_90 fat binary) ────"
nvcc -O3 -std=c++14 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 \
     "$SRC/tree/gpu/tc_decider.cu" -o "$WB/tc_decider"
echo "  build exit=$?"; ls -la "$WB/tc_decider"

echo; echo "──── nptn=100000 ncat=4 (thesis-test arm) ────"
"$WB/tc_decider" 100000 4 50 5

echo; echo "──── nptn=1000000 ncat=4 (GATE arm) ────"
"$WB/tc_decider" 1000000 4 50 5

echo; echo "════════ DONE $(date -Iseconds) ════════"
