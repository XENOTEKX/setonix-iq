#!/bin/bash
# run_g4_1b_jolt_alpha_v100.sh — Phase G.4.1b (JOLT): joint (branches + alpha) optimisation, full +G MLE.
#
# Completes the standalone JOLT optimiser for +G: gpu_k8b_jolt_alpha.cu adds the gamma shape alpha to the
# joint parameter vector. alpha-gradient = sum_c (dr_c/dalpha)*gradR[c], where gradR[c] is the validated
# G.4.0b per-category rate gradient (k_ratenum) and dr_c/dalpha is a host FD of the mean-rate discrete-gamma
# (Yang 1994 — IQ-TREE's "MEAN of the portion"; validated in-harness to reproduce the .iqtree rates at
# alpha=0.9963). alpha is folded into the SAME joint LM diagonal-Newton step (NO Brent line search).
#
# GATES:
#   [gamma]    the discretisation reproduces IQ-TREE's {0.1362,0.4756,0.9994,2.3887} at alpha=0.9963 (maxdiff<5e-4).
#   PRE-CHECK  alpha-gradient FD-validated (analytic dlnL/dalpha vs central difference, |rel|<0.01).
#   (1) COLD (b=0.1, alpha=3.0 — BOTH far from optimal) reaches the same optimum as WARM, lnL rel<=1e-9.
#   (2) COLD reaches the FULL CPU MLE -7541976.8529 (.iqtree, alpha=0.9963) rel<=1e-9, alpha->0.9963.
#   (3) HEADLINE: joint-iteration count for the full (197 branches + alpha) cold-start optimisation, alpha
#       folded into the joint step (replacing IQ-TREE's ~10-20 sequential alpha-Brent full-tree traversals).
#
#PBS -N g4-1b-jolt-alpha-v100
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0  2>/dev/null || true
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k8b_jolt_alpha.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k8b_jolt_alpha
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.4.1b JOLT joint (branches+alpha) — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"; echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"
echo "──── build (nvcc -O3 -arch=sm_70, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo; echo "════════ MODEL g4 (LG+G4, joint branches+alpha) ════════"; "$BIN" "$ALN" "$TREE" g4 400
echo; echo "════════ DONE $(date -Iseconds) ════════"
