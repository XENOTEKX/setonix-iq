#!/bin/bash
# run_g1_2_k2_v100.sh — Phase G.1.2: build + validate the single-edge derivative kernel (K2).
#
# Builds gpu_k2_derv.cu (nvcc, standalone) and for g4/g1/r8/r10:
#   (1) cross-checks lnL(t0) == K1/G.0 oracle (proves the eigen-space freq-fold identity),
#   (2) FD-validates df/ddf at an off-optimum branch length (swept-eps; gate g4<3e-3, g1<1e-6),
#   (3) runs Newton from the off-optimum length -> converges to the tree's optimized edge length.
#
#PBS -N g1-2-k2-v100
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
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k2_derv.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k2_derv
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.1.2 K2 derivative build+validate — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"
echo "──── build (nvcc -O3 -arch=sm_70, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
for M in g4 g1 r8 r10; do echo; echo "════════ MODEL $M ════════"; "$BIN" "$ALN" "$TREE" "$M" 20; done
echo; echo "════════ DONE $(date -Iseconds) ════════"
