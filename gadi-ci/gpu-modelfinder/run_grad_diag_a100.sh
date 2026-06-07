#!/bin/bash
# run_grad_diag_a100.sh — Phase G.0 GPU-gradient DIAGNOSTIC: isolate whether the wrong CUDA gradient
# values (correct on CPU, wrong on A100 for LG+G4 even with the hmctest transpose recipe) are caused by
# the rate-category dimension or by the 20-state pre-order path itself. Runs single-rate LG (g1, NCAT=1):
#   cpu g1 — validates the g1 setup FD-passes on CPU.
#   gpu g1 — if PASS ⇒ bug is GPU category handling (category-split could rescue, à la r10split);
#            if FAIL ⇒ the 20-state CUDA pre-order/edge-derivative is broken in BEAGLE 4.0.1 regardless.
#   gpu g4 — reference (known wrong) for side-by-side.
#PBS -N g0-grad-diag
#PBS -P dx61
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=64GB
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
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_derisk_diag
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
echo "════════ build (in-job) ════════"
icpx -O2 -std=c++17 "$SRC" -o "$BIN" -I/apps/beagle-lib/4.0.1/include/libhmsbeagle-1 -L/apps/beagle-lib/4.0.1/lib -lhmsbeagle
RC=$?; echo "build exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "════════ G.0 GPU-gradient NCAT diagnostic on $(hostname) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
run(){ echo; echo "──────── $1 ────────"; shift; "$BIN" "$ALN" "$TREE" "$@"; echo "[exit=$?]"; }
run "g1 (NCAT=1) — CPU gradient FD-check"   cpu g1 3 0 1
run "g1 (NCAT=1) — GPU gradient FD + timing" gpu g1 30 0 1
run "g4 (NCAT=4) — GPU gradient (reference)"  gpu g4 5  0 1
echo
echo "════════ DONE ════════"
