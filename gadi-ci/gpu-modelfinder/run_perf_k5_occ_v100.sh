#!/bin/bash
# run_perf_k5_occ_v100.sh — G.1.3 intra-kernel perf lever: OCCUPANCY sweep (K5).
#
# k1_node is latency-bound at 25% occupancy (128 regs/thread -> 2 blocks/SM; ncu job 170195112). This A/B's
# __launch_bounds__ register caps to raise occupancy and find the sweet spot vs spill cost. Every config must
# give the bit-identical G.0 oracle lnL (the body is unchanged). Decision metric = sweep ms (lower is better).
#
#PBS -N perf-k5-occ
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:25:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0  2>/dev/null || true

D=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder
SRC=$D/gpu_k5_occ.cu
BIN=$D/gpu_k5_occ
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ G.1.3 perf K5 occupancy sweep — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

echo "──── build (nvcc -O3 -arch=sm_70 -lineinfo, FP64, NO fast-math) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

# report the actual register count per kernel variant at compile time (ptxas)
echo "──── ptxas register usage per kernel (recompile -Xptxas -v) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -Xptxas -v -c "$SRC" -o /dev/null 2>&1 | grep -E "k1_base|k1_lb|registers|spill" | head -40 || true

echo; echo "════════ TIMING SWEEP (all models, all occupancy configs) ════════"
"$BIN" "$ALN" "$TREE" all 30

echo; echo "════════ ncu CONFIRM register/occupancy mechanism (g4: k1_base vs k1_lb variants) ════════"
if command -v ncu >/dev/null 2>&1; then
  ncu --target-processes application-only --kernel-name "regex:k1_base|k1_lb" \
      --launch-skip 120 --launch-count 8 --section Occupancy --section SpeedOfLight \
      "$BIN" "$ALN" "$TREE" g4 1 2>&1 | \
    grep -E "k1_base|k1_lb|Registers Per Thread|Achieved Occupancy|Theoretical Occupancy|Block Limit Registers|Compute \(SM\)|Memory Throughput|Duration" || echo "[ncu] no rows (counters locked?)"
else
  echo "[ncu] MISSING"
fi

echo
echo "════════ DONE $(date -Iseconds) ════════"
