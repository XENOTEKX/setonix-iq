#!/bin/bash
# run_g2_ifallback_v100.sh — fast confirmation that +I correctly routes to CPU fallback (small len_1000 AA aln).
# Control: WAG+G4 (GPU path MUST fire here) vs WAG+I+G4 (+I -> CPU fallback). Both --gpu vs CPU, -nt 1.
# GATES: WAG+G4 -> [GPU-*] active markers + lnL bit-identical to CPU.
#        WAG+I+G4 -> [GPU-KERNEL] install + [GPU-BRANCH] CPU-fallback marker, NO [GPU-DERV]/[GPU-FROMBUF] active,
#                    lnL == pure-CPU lnL (proves the +I gate fires and stays correct).
#PBS -N g2-ifb
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=00:20:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true; module load gcc/12.2.0 2>/dev/null || true
export CC=gcc CXX=g++
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN=$SRC/build-gpu-on/iqtree3
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000/tree_1/alignment_1000.phy
TREE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000/tree_1/tree_1.full.treefile
RUNDIR=$SRC/g2_ifb_runs; mkdir -p "$RUNDIR"
echo "════════ G.2 +I FALLBACK CHECK — $(hostname) $(date -Iseconds) ════════"
[ -x "$BIN" ] || { echo "no binary"; exit 1; }

for M in "WAG+G4:control_gpu" "WAG+I+G4:iplus_fallback"; do
  MODEL="${M%%:*}"; TAG="${M##*:}"
  echo; echo "════════ [$TAG] $MODEL ════════"
  echo "---- GPU run ----"
  "$BIN" --gpu -s "$ALN" -te "$TREE" -m "$MODEL" -nt 1 -pre "$RUNDIR/${TAG}_gpu" -redo 2>&1 | \
    grep -E "GPU-KERNEL|GPU-BRANCH|GPU-DERV|GPU-FROMBUF|GPU-XCHECK|Optimal log-likelihood" | head -20
  echo "---- CPU run ----"
  "$BIN" -s "$ALN" -te "$TREE" -m "$MODEL" -nt 1 -pre "$RUNDIR/${TAG}_cpu" -redo 2>&1 | \
    grep -E "Optimal log-likelihood" | head -3
  G=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/${TAG}_gpu.iqtree" 2>/dev/null | grep -oE "\-?[0-9.]+$")
  C=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/${TAG}_cpu.iqtree" 2>/dev/null | grep -oE "\-?[0-9.]+$")
  python3 -c "
g,c=float('${G:-nan}'),float('${C:-nan}')
rl=abs((g-c)/c) if c else abs(g-c)
print(f'[$TAG] GPU lnL={g} CPU lnL={c}  rel={rl:.3e} -> {\"PASS\" if rl<=1e-9 else \"CHECK\"}')"
done
echo; echo "════════ DONE $(date -Iseconds) ════════"
