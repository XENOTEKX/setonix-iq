#!/bin/bash
# run_g1_3_k3_v100.sh — Phase G.1.3: build + validate the CUDA-graph postorder sweep + on-device echild (K3).
#
# Compiles gpu_k3_graph.cu with nvcc (standalone, no IQ-TREE build) and runs it on the SAME AA-100K alignment
# + fixed reference tree as G.0/G.1.1/G.1.2. For each model it runs the validation ladder:
#   V0 build_echild bit-reproduces host echild | V1 graph lnL == naive == G.0 oracle | V2 patlh bit-identical
#   graph-vs-naive (pre-reduction) | V3 deterministic replay | V4 perturbation Δ<1e-4 | V5 single-branch opt via
#   graph replay == naive | V6 (g4) multi-branch optimizeAllBranches-shaped sweep == naive | TIMING-CURVE.
# Then a launch-bound TIMING CURVE for g4 at nptn ∈ {1000, 10000, 100000} (graph wins as compute shrinks).
#
# Oracle (G.0): g4 -7541976.9391 | r8 -7556251.9185 | r10 -7554280.5776 | g1 -7974816.4323
#
#PBS -N g1-3-k3-v100
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

SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k3_graph.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k3_graph
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.1.3 K3 build+validate — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING) $ALN"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING) $TREE"

echo "──── build (nvcc -O3 -arch=sm_70 -lineinfo, precise FP64 — NO fast-math) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

# args: <aln> <tree> <model> <reps> <ptncap=0> <multiBranch=0>
echo; echo "════════ FULL VALIDATION LADDER (V0–V7) per model ════════"
echo; echo "──────── MODEL g4 (+ V6 multi-branch sweep) ────────"; "$BIN" "$ALN" "$TREE" g4 30 0 1
for M in r8 r10 g1; do echo; echo "──────── MODEL $M ────────"; "$BIN" "$ALN" "$TREE" "$M" 30 0 0; done

echo; echo "════════ LAUNCH-BOUND TIMING CURVE (g4; graph vs naive vs pattern count) ════════"
for N in 1000 10000 100000; do echo; echo "──── g4 nptn≈$N ────"; "$BIN" "$ALN" "$TREE" g4 200 "$N" 0; done

echo
echo "════════ DONE $(date -Iseconds) ════════"
