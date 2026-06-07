#!/bin/bash
# run_grad_cpu_v100.sh — Phase G.0 gradient CORRECTNESS reference on the CPU plugin.
# Runs on a gpuvolta node (the CPU plugin still needs the GPU/OpenCL stack present at createInstance —
# see bug 14; a pure-CPU node crashes at GPUInterfaceOpenCL.cpp:105). Uses the CPU plugin only.
# Validates the BEAGLE pre-order branch-length gradient vs finite differences with a SWEPT eps (the
# objective is lnL≈-7.5e6, so eps=1e-5 is roundoff-limited; the sweep takes the best-converged FD).
#PBS -N g0-grad-cpu
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
echo "════════ G.0 gradient CPU FD-validation on $(hostname) ════════"
run(){ echo; echo "──────── $1 ────────"; shift; "$BIN" "$ALN" "$TREE" "$@"; echo "[exit=$?]"; }
# CPU plugin, double precision, no scaling (scale=0), gradient on (dograd=1), 3 reps (correctness > timing).
run "MODEL g4 — CPU gradient FD-check (eps sweep)" cpu g4 3 0 1
echo
echo "════════ DONE ════════"
