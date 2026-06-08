#!/bin/bash
# run_g1_1_k1_v100.sh — Phase G.1.1: build + validate the custom CUDA postorder lnL kernel (K1).
#
# Compiles gpu_k1_lnl.cu with nvcc (standalone, no IQ-TREE build) and runs it on the SAME AA-100K
# alignment + fixed reference tree as the G.0 harness, for g4/r8/r10/g1. The kernel's full-tree lnL must
# match the G.0 BEAGLE oracle (representation-independent), and r10 must run in ONE pass (NCAT=10, no
# BEAGLE kMatrixBlockSize<=8 cap) reproducing the r10split number bit-for-bit.
#
# Oracle (G.0, gpu-modelfinder-g0-log.md): g4 -7541976.9391 | r8 -7556251.9185 | r10 -7554280.5776 | g1 -7974816.4323
#
#PBS -N g1-1-k1-v100
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

SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k1_lnl.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k1_lnl
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.1.1 K1 build+validate — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING) $ALN"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING) $TREE"

echo "──── build (nvcc -O3 -arch=sm_70, precise FP64 — NO fast-math) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

run(){ echo; echo "──────── MODEL $1 ────────"; "$BIN" "$ALN" "$TREE" "$1" "${2:-20}"; }
for M in g4 r8 r10 g1; do run "$M" 20; done

echo
echo "════════ DONE $(date -Iseconds) ════════"
