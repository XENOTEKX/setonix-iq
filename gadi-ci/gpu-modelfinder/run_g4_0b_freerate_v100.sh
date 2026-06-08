#!/bin/bash
# run_g4_0b_freerate_v100.sh — Phase G.4.0b (JOLT make-or-break): build + validate
#   (A) Ji O(depth) preorder-buffer RECYCLING (so r8/r10 fit the 32GB V100), and
#   (B) the FreeRate (+R) rate-parameter gradient OVERFLOW KILL-SWITCH.
#
# Builds gpu_k7b_freerate.cu (nvcc, standalone — reuses the validated K1/K2/k7_pre scaffolding + ONE new
# kernel k_ratenum) and runs:
#   g4  — REGRESSION: the O(depth)-recycled branch gradient must reproduce the G.4.0 result (gamma; no +R block)
#   r4  — the plan-named LG+R4 +R kill-switch (fits naively, but exercises the recycled pool + rate grad)
#   r8  — newly fits via recycling; +R kill-switch on a wider rate spread
#   r10 — newly fits via recycling; +R kill-switch on the widest rate spread (the OOM case in G.4.0)
#
# Gates:
#   (A) pre-pool peak == tree height (<< nnodes) AND branch-grad lnL edge-invariance rel<=1e-9 + df FD PASS
#       (recycling is numerically IDENTICAL to the naive G.4.0 buffer).
#   (B) for r4/r8/r10:  (B1) dlnL/dr_k FINITE & bounded (NO 1e54 overflow); (B2) the EXACT scaling identity
#       Sum_k r_k*gr_k == Sum_e b_e*gb_e (rel<=1e-6, ties +R grad to the validated branch grad); (B3) FD
#       |G-ratio|<0.01 (the Mode-L FDCHECK that read ~1e54). All three => the unscaled GPU path does NOT overflow.
#
#PBS -N g4-0b-freerate-v100
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
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k7b_freerate.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k7b_freerate
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.4.0b FreeRate recycling + overflow kill-switch — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"
echo "──── build (nvcc -O3 -arch=sm_70, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
# g4 first (recycling regression vs G.4.0), then the +R kill-switch on r4 (plan-named), r8, r10 (were OOM in G.4.0).
for M in g4 r4 r8 r10; do echo; echo "════════ MODEL $M ════════"; "$BIN" "$ALN" "$TREE" "$M" 1; done
echo; echo "════════ DONE $(date -Iseconds) ════════"
