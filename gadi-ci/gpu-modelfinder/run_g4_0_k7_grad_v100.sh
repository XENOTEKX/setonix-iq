#!/bin/bash
# run_g4_0_k7_grad_v100.sh — Phase G.4.0 (JOLT): build + validate the preorder ALL-BRANCH gradient kernel (K7).
#
# Builds gpu_k7_grad.cu (nvcc, standalone — reuses the validated K1/K2 scaffolding + one new kernel k7_pre)
# and for g4/g1/r8/r10 computes the gradient w.r.t. ALL 2N-3 branches from ONE postorder + ONE preorder sweep,
# then validates:
#   (1) lnL edge-invariance: for EVERY edge, lnL_e == K1/G.0 oracle (rel<=1e-9) — this VALIDATES pre_v via the
#       eigen-space freq-fold identity (a wrong preorder partial breaks lnL invariance immediately).
#   (2) all-branch df FD-validation: each branch's analytic df matches a central difference of lnL (gate g1<1e-6,
#       else <3e-3) — the non-negotiable gradient gate.
#   (3) the central edge (root->c0) reproduces the G.1.2/G.2.1a-validated K2 df (ties to the bit-validated path).
# r8/r10 here are a BONUS preview of G.4.0b: if the +R gradient computes without overflow on this UNSCALED FP64
# path, it is direct evidence for the JOLT make-or-break hypothesis (Mode-L's ~10^54 +R overflow was a CPU
# scale_log artifact this path does not have).
#
#PBS -N g4-0-k7-v100
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:45:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0  2>/dev/null || true
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k7_grad.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k7_grad
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.4.0 K7 all-branch gradient build+validate — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"
echo "──── build (nvcc -O3 -arch=sm_70, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
# g4 (the +G4 gate that matters for -m TEST) + g1 (tightest FD gate). r8/r10 are SKIPPED here: the naive
# one-preorder-buffer-per-node arena OOMs the 32GB V100 (35-45GB); +R needs Ji O(depth) recycling (G.4.0b).
for M in g4 g1; do echo; echo "════════ MODEL $M ════════"; "$BIN" "$ALN" "$TREE" "$M" 5; done
echo; echo "════════ DONE $(date -Iseconds) ════════"
