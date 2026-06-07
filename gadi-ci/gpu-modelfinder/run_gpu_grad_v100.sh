#!/bin/bash
# run_gpu_grad_v100.sh — Phase G.0 gradient timing: BEAGLE pre-order branch-length gradient on a V100.
# Computes d lnL/d(branch length) for every edge via the Ji-et-al O(N) pre-order pass, FD-validates
# the 5 longest edges, and times the full lnL+gradient eval. CPU plugin = correctness reference.
# args: gpu_derisk <aln> <tree> <cpu|gpu> <model> <reps> <scale=0> <dograd=1>
#PBS -N g0-grad-v100
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
module load beagle-lib/4.0.1 2>/dev/null || true
module load cuda/12.6.2 2>/dev/null || true
module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
export LD_LIBRARY_PATH=/apps/beagle-lib/4.0.1/lib:${LD_LIBRARY_PATH:-}
SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_derisk.cpp
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_derisk
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
echo "════════ build (in-job) ════════"
icpx -O2 -std=c++17 "$SRC" -o "$BIN" -I/apps/beagle-lib/4.0.1/include/libhmsbeagle-1 -L/apps/beagle-lib/4.0.1/lib -lhmsbeagle
RC=$?; echo "build exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "════════ G.0 gradient: BEAGLE AA-100K pre-order branch gradient on $(hostname) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
run(){ echo; echo "──────── $1 ────────"; shift; "$BIN" "$ALN" "$TREE" "$@"; }
# V100 has 32GB; gradient mode doubles partial buffer count (pre-order + post-order).
# g4 NCAT=4: ~(2*199+4)*4cats*20states*100K sites*8B ≈ 25.7 GB — fits.
# fig4 NCAT=5: ~32 GB — marginal / OOM; run on dgxa100 (A100-80GB) instead.
for MODEL in g4; do
  # scale=0 (double precision, no rescaling), dograd=1
  run "MODEL $MODEL — CPU gradient (FD reference)" cpu "$MODEL" 5  0 1
  run "MODEL $MODEL — GPU gradient (FD + timing)"  gpu "$MODEL" 30 0 1
done
echo
echo "════════ DONE ════════"
