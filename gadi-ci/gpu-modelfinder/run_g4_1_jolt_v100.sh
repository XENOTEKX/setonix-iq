#!/bin/bash
# run_g4_1_jolt_v100.sh — Phase G.4.1 (JOLT): standalone joint LM diagonal-Newton optimiser.
#
# Builds gpu_k8_jolt.cu (nvcc; reuses the validated K1/K7/k2_derv + O(depth) pool byte-for-byte, adds the
# joint optimiser driver + mmap/pinned data load) and tests the CORE JOLT thesis: that updating ALL 2N-3
# branches SIMULTANEOUSLY from a joint analytic gradient (ONE postorder + ONE preorder sweep per iteration)
# converges to the same MLE as IQ-TREE's 197 SEQUENTIAL per-edge Newton sweeps — in few parallel iterations.
#
# Optimiser: joint LM-damped diagonal-Newton. b_e += df_e/(|ddf_e|+mu) for ALL edges at once (ddf = the
# validated per-edge curvature as the diagonal preconditioner), accept-if-lnL-increases else grow mu (no
# line search to balloon). Alpha/rates FIXED at the validated MLE (g4 = LG+G4 MLE rates) so the target is
# clean and needs no gamma discretisation (joint-alpha = G.4.1b).
#
# GATES (advisor-hardened):
#   PRE-CHECK: gradient at the .treefile (MLE) branches reproduces the oracle lnL + calibrates ||g|| (catches
#              an assembly/sign bug before the convergence run).
#   (1) COLD start (b=0.1, clearly NON-optimal) reaches the SAME optimum as the WARM (.treefile) start, lnL
#       rel <= 1e-9 — the meaningful convergence test (a warm start alone would converge trivially).
#   (2) HEADLINE = the COLD-start joint-iteration + critical-path-sweep count (the JOLT thesis metric: few
#       parallel sweeps vs IQ-TREE's hundreds of sequential edge-evals). Absolute count is the headline; the
#       vs-IQ-TREE factor is approximate (IQ-TREE warm-starts each candidate from BIONJ, not cold).
#
# MMAP/pinned (user request): the alignment is mmap'd (RAM-resident page cache) and tip/echild stage through
# cudaHostAlloc pinned host buffers for faster H2D. HONEST: this is the one-time-load lever; the hot loop is
# dependent-kernel-bound (no disk, tiny per-iter H2D). The async double-buffered RAM->GPU win is the G.4.3 tiling regime.
#
#PBS -N g4-1-jolt-v100
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
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k8_jolt.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k8_jolt
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.4.1 JOLT joint optimiser — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"
echo "──── build (nvcc -O3 -arch=sm_70, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
for M in g4 g1; do echo; echo "════════ MODEL $M ════════"; "$BIN" "$ALN" "$TREE" "$M" 300; done
echo; echo "════════ DONE $(date -Iseconds) ════════"
