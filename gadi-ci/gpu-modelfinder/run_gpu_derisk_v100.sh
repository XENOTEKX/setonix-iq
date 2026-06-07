#!/bin/bash
# run_gpu_derisk_v100.sh — Phase G.0: run the BEAGLE de-risk harness on a V100 (gpuvolta).
# Runs the CPU plugin (parity vs IQ-TREE -7541976.853) then the CUDA/GPU plugin (parity + timing).
#PBS -N g0-derisk-v100
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
echo "════════ build (in-job, correct module env) ════════"
icpx -O2 -std=c++17 "$SRC" -o "$BIN" -I/apps/beagle-lib/4.0.1/include/libhmsbeagle-1 -L/apps/beagle-lib/4.0.1/lib -lhmsbeagle
RC=$?; echo "build exit=$RC"; if [ $RC -ne 0 ]; then echo "BUILD FAILED — aborting"; exit 1; fi
echo "════════ G.0 de-risk: BEAGLE AA-100K LG+G4 on $(hostname) ════════"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo "binary: $BIN"; ls -la "$BIN"
echo
echo "════════ available BEAGLE resources (GPU enumeration) ════════"
echo
# CPU plugin scaled = parity reference (CPU scaling always works); GPU unscaled = fast path (double
# precision doesn't underflow on this tree, and avoids BEAGLE-CUDA's >8-category rescale-kernel cap).
run(){ echo; echo "──────── $1 ────────"; shift; "$BIN" "$ALN" "$TREE" "$@"; }
for MODEL in g4 fig4 r8; do
  run "MODEL $MODEL — CPU scaled (reference)" cpu "$MODEL" 5 1
  run "MODEL $MODEL — GPU unscaled (timing)"  gpu "$MODEL" 30 0
done
run "MODEL r10 — GPU unscaled (expected BEAGLE-CUDA failure: >8 categories)" gpu r10 5 0
run "MODEL r10split — CPU scaled (reference)"            cpu r10split 5 1
run "MODEL r10split — GPU unscaled (5+5 cat-split workaround)" gpu r10split 30 0
echo
echo "════════ DONE ════════"
