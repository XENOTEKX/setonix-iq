#!/bin/bash
# run_g1_3_k4_fused_v100.sh — G.1.3 perf pass: same-depth kernel FUSION of the postorder lnL sweep (K4).
#
# Compiles gpu_k4_fused.cu (standalone nvcc) and validates that batching the ~98 per-node k1_node launches
# into ~tree-height level launches (k4_level, 2D grid) is BIT-IDENTICAL to the per-node sweep + G.0 oracle,
# then measures per-node vs fused vs fused-graph wall-clock across pattern counts (where the launch/scheduling
# overhead the CUDA graph alone could NOT remove should now drop).
#
# Oracle (G.0): g4 -7541976.9391 | r8 -7556251.9185 | r10 -7554280.5776 | g1 -7974816.4323
#
#PBS -N g1-3-k4-fused
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0  2>/dev/null || true

SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k4_fused.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k4_fused
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.1.3 perf pass K4 (kernel fusion) — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"

echo "──── build (nvcc -O3 -arch=sm_70 -lineinfo, precise FP64 — NO fast-math) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

# args: <aln> <tree> <model> <reps> <ptncap=0>
echo; echo "════════ CORRECTNESS + FULL-nptn TIMING per model ════════"
for M in g4 r8 r10 g1; do echo; echo "──────── MODEL $M ────────"; "$BIN" "$ALN" "$TREE" "$M" 50 0; done

echo; echo "════════ LAUNCH-BOUND TIMING CURVE (g4; per-node vs fused vs fused-graph vs pattern count) ════════"
for N in 1000 10000 100000; do echo; echo "──── g4 nptn≈$N ────"; "$BIN" "$ALN" "$TREE" g4 200 "$N"; done

echo
echo "════════ DONE $(date -Iseconds) ════════"
