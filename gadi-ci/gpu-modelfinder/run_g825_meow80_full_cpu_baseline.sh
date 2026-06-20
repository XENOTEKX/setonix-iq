#!/bin/bash
# run_g825_meow80_full_cpu_baseline.sh — the CPU-vs-GPU WALLTIME baseline the paper does not provide.
# Same workload as the H200 headline (job 171733080: LG+MEOW80+G4 -te on the FULL CAT_100S93F.phy, 100x22,462),
# but the STANDARD CPU path (no --jolt) on a full Gadi normalsr node, IQ-TREE picking its own optimal thread count
# (-nt AUTO). Gives (a) the CPU wall for the speedup multiplier, (b) the standard Sum prop*tns=1 tree (branch-length
# ground truth for the rate-1 rescale validation), (c) the CPU lnL for a full-scale GPU>=CPU check.
# GPU reference: 1446 s (~24 min) on 1 H200, GPU lnL -1665670.9967.
#
# Submit: qsub gadi-ci/gpu-modelfinder/run_g825_meow80_full_cpu_baseline.sh
#PBS -N meow-cpu-base
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=48
#PBS -l mem=240GB
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true   # only for libstdc++ parity; CPU binary needs no GPU
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3      # same binary; no --jolt => pure CPU path
ALN=/scratch/rc29/as1708/eukaryote_williamson2025/CAT_100S93F.phy
NEX=/scratch/rc29/as1708/eukaryote_williamson2025/MEOW6020.nex
TREE=/scratch/rc29/as1708/iqtree3-gpu/euk_will2025_run/A_fasttree.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/g825_meow_full_cpu_$PBS_JOBID; mkdir -p "$WB"; cd "$WB"
[ -s "$ALN" ] || { echo "FATAL: $ALN missing"; exit 1; }

echo "════ CPU baseline LG+MEOW80+G4 -te full-data — $(hostname) $(date -Iseconds) ════"
echo "alignment: $(head -1 "$ALN")  | cores: $(nproc)  | BIN md5: $(md5sum "$BIN" | cut -c1-12)"
free -g | awk '/Mem:/{print "  host RAM total="$2" GB"}'

t0=$(date +%s)
# NOTE: -nt 48 FIXED (not AUTO). The -nt AUTO benchmark wastes ~1 h running a full ~500-1100 s lnL eval per thread
# count before optimizing, and MEOW80's 105 GB arena is memory-bandwidth-bound (68% efficiency at 2 threads), so
# AUTO would over-test. 48 Sapphire Rapids cores is a generous, defensible "well-resourced CPU node" baseline.
"$BIN" -te "$TREE" -s "$ALN" -mdef "$NEX" -m LG+ESmodel+G4 -mwopt -nt 48 -pre "$WB/meow_cpu" -redo \
   > "$WB/meow_cpu.console" 2>&1
RC=$?
WALL=$(( $(date +%s) - t0 ))
echo "  exit=$RC  CPU_WALL=${WALL}s ($(awk "BEGIN{printf \"%.1f\", $WALL/60}") min)"
echo "── threads actually used (-nt AUTO) ──"; grep -aE 'BEST NUMBER OF THREADS|Threads|-nt' "$WB/meow_cpu.console" | head -3 | sed 's/^/  /'
echo "── CPU lnL + tree length ──"
grep -aE 'Log-likelihood of the tree|Total tree length' "$WB/meow_cpu.iqtree" 2>/dev/null | sed 's/^/  /'
echo "── SPEEDUP vs H200 (1446 s) ──"; awk "BEGIN{printf \"  GPU 1446s vs CPU %ds = %.2fx\n\", $WALL, $WALL/1446.0}"
echo "════ DONE $(date -Iseconds) ════"
