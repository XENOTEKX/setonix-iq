#!/bin/bash
# run_perf_profile_k1_v100.sh — G.1.3 intra-kernel perf lever: PROFILE the K1 postorder lnL kernel.
#
# Before optimizing, identify the ACTUAL bottleneck of k1_node (the dominant per-model cost) with Nsight
# Compute (ncu): SpeedOfLight (compute vs memory %), Occupancy (register/block limits), MemoryWorkloadAnalysis
# (HBM throughput, L1/L2 hit rates), LaunchStats. This decides whether the lever is shared-mem echild staging
# (kill redundant global reads), register-pressure reduction (prod[NS] limits occupancy), or wider/coalesced
# loads. ncu counter access can be restricted on shared clusters (ERR_NVGPUCTRPERM); if so we fall back to
# nsys (kernel timeline) + a hand roofline. g4 (NCAT=4) and r10 (NCAT=10) bracket the category dimension.
#
#PBS -N perf-prof-k1
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
SRC=$D/gpu_k1_lnl.cu
BIN=$D/gpu_k1_lnl
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile

echo "════════ K1 intra-kernel PROFILE — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[tools] ncu=$(command -v ncu || echo MISSING)  nsys=$(command -v nsys || echo MISSING)"

echo "──── build k1 (nvcc -O3 -arch=sm_70 -lineinfo) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

# representative interior k1_node launches (skip the first ~80 so we profile mid/upper-tree nodes with
# internal children, the common case); reps=1 keeps the run short under ncu instrumentation.
NCU_SECTIONS="--section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section LaunchStats"
for M in g4 r10; do
  echo; echo "════════ ncu PROFILE model=$M (k1_node launches 81-84) ════════"
  if command -v ncu >/dev/null 2>&1; then
    ncu --target-processes application-only --kernel-name "regex:k1_node" \
        --launch-skip 80 --launch-count 4 $NCU_SECTIONS \
        "$BIN" "$ALN" "$TREE" "$M" 1 2>&1 | \
      grep -E "k1_node|Duration|Compute \(SM\)|Memory Throughput|DRAM Throughput|Achieved Occupancy|Registers Per Thread|Block Limit|L1|L2 Hit|Theoretical Occupancy|Waves|Grid Size|Block Size|SM \[%\]|stall" \
      || echo "[ncu] produced no parseable rows (see raw above; likely ERR_NVGPUCTRPERM — counters locked)"
  else
    echo "[ncu] MISSING"
  fi
done

echo; echo "════════ nsys timeline (kernel durations, fallback if ncu counters locked) ════════"
if command -v nsys >/dev/null 2>&1; then
  nsys profile -o "$D/nsys_k1_g4" --force-overwrite true --stats=true \
      "$BIN" "$ALN" "$TREE" g4 1 2>&1 | grep -E "k1_node|Time|cuda_gpu_kern|Avg|Total|%|Instances" | head -40 \
    || echo "[nsys] no stats"
else
  echo "[nsys] MISSING"
fi

echo; echo "════════ reference timed run (no profiler) ════════"
"$BIN" "$ALN" "$TREE" g4 20 2>&1 | grep -E "lnL eval|model=|rel="

echo
echo "════════ DONE $(date -Iseconds) ════════"
