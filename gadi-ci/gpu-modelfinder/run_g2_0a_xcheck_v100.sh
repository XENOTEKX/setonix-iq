#!/bin/bash
# run_g2_0a_xcheck_v100.sh — Phase G.2.0a: build the in-tree GPU lnL cross-check + validate vs CPU.
#
# Incrementally rebuilds the GPU-ON iqtree3 (the build-gpu-on/ dir was configured at G.1.0) after adding:
#   tree/gpu/gpu_lnl_intree.cu      (extern-C K1 launcher, ptn_freq-weighted)
#   tree/phylotreegpu.cpp           (PhyloTree::gpuLnLCrossCheckOnce: clean-room extract+launch+compare)
#   + the gated one-shot hook in computeLikelihood (phylotree.cpp) + CMake wiring.
# Then runs `iqtree3 --gpu -te <tree> -m LG+G4` on AA-100K (NORM_LH, 100 taxa, 1 mixture, NCAT=4) so the
# first computeLikelihood fires the cross-check. GATE: [GPU-XCHECK] GPU lnL == CPU lnL rel <= 1e-6 (PASS).
# Also runs the CPU path (--gpu off) to confirm the CPU lnL is unchanged by the additive hook.
#
#PBS -N g2-0a-xcheck
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cmake/3.24.2 2>/dev/null || true
module load eigen/3.3.7  2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export CC=gcc CXX=g++

SRC=/scratch/rc29/as1708/iqtree3-gpu
BUILD=$SRC/build-gpu-on
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
RUNDIR=$SRC/g2_0a_runs; mkdir -p "$RUNDIR"

echo "════════ G.2.0a build+xcheck — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"

echo "──── incremental reconfigure + build (build-gpu-on already configured at G.1.0) ────"
cd "$BUILD" || { echo "no build dir"; exit 1; }
cmake . >/tmp/g2_0a_cmake.log 2>&1; RC=$?; tail -3 /tmp/g2_0a_cmake.log; [ $RC -ne 0 ] && { echo "CMAKE FAILED"; cat /tmp/g2_0a_cmake.log; exit 1; }
make -j12 iqtree3 2>/tmp/g2_0a_make.log; RC=$?
echo "make exit=$RC"
if [ $RC -ne 0 ]; then echo "==== BUILD FAILED (last 60 lines) ===="; tail -60 /tmp/g2_0a_make.log; exit 1; fi
echo "built: $(ls -la "$BUILD/iqtree3" | awk '{print $5, $6, $7, $8}')"

BIN="$BUILD/iqtree3"
echo; echo "════════ RUN --gpu (fires the one-shot cross-check on first computeLikelihood) ════════"
"$BIN" --gpu -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -pre "$RUNDIR/xcheck_gpu" -redo 2>&1 | \
  grep -E "GPU-XCHECK|BEST SCORE|Log-likelihood of the tree|Optimal log-likelihood|CPU lnL|GPU lnL" | head -30
echo "----- final lnL line from the .iqtree report -----"
grep -E "Log-likelihood|BEST SCORE|Optimal" "$RUNDIR/xcheck_gpu.iqtree" 2>/dev/null | head -5

echo; echo "════════ RUN CPU (no --gpu): confirm lnL unchanged by the additive hook ════════"
"$BIN" -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -pre "$RUNDIR/xcheck_cpu" -redo 2>&1 | \
  grep -E "Optimal log-likelihood|BEST SCORE|Log-likelihood of the tree" | head -5
grep -E "Log-likelihood|Optimal" "$RUNDIR/xcheck_cpu.iqtree" 2>/dev/null | head -5

echo
echo "════════ DONE $(date -Iseconds) ════════"
