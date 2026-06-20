#!/bin/bash
# run_g825_rescale_measure_v100.sh — MEASURE the rate-1 rescale question (no behavior change; JOLT_DBG diagnostic only).
# For each model: run --jolt (JOLTMix) AND standard CPU (no --jolt) on the SAME -te tree, then compare the
# "Total tree length (sum of branch lengths)" from each .iqtree. The CPU value is IQ-TREE's Sum prop*tns=1 convention
# ground truth. If GPU(jolt) treeLen != CPU treeLen by a clean global factor, that factor IS the rescale (and
# [JOLTMIX-RESCALE-DBG] rho tells us the live overall rate + direction). C20 has FIXED weights => GPU & CPU reach the
# SAME MLE (cleanest comparison); MEOW80 has ESTIMATED weights (the EM changes them => the suspected rescale case).
#
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=60GB -l walltime=00:30:00 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd gadi-ci/gpu-modelfinder/run_g825_rescale_measure_v100.sh
#PBS -N g825resc
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
TREE=/scratch/rc29/as1708/iqtree3-gpu/euk_will2025_run/A_fasttree.treefile
ALN=/scratch/rc29/as1708/iqtree3-gpu/g822_mix/euk400.phy
NEX=/scratch/rc29/as1708/eukaryote_williamson2025/MEOW6020.nex
WB=/scratch/rc29/as1708/iqtree3-gpu/g825_rescale; mkdir -p "$WB"; cd "$WB"
[ -s "$ALN" ] || { echo "FATAL: $ALN missing"; exit 1; }

echo "════ G.8.2.5 rate-1 rescale MEASUREMENT — $(hostname) $(date -Iseconds) ════"
echo "BIN md5: $(md5sum "$BIN" | cut -c1-12)"

tlen() { grep -aE 'Total tree length \(sum of branch lengths\):' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1; }

for SPEC in "C20:fixed:-m LG+C20+G4" "MEOW80:EM:-mdef $NEX -m LG+ESmodel+G4 -mwopt"; do
  NAME=${SPEC%%:*}; REST=${SPEC#*:}; WT=${REST%%:*}; ARGS=${REST#*:}
  echo "──────── $NAME (weights=$WT) ────────"
  echo "  -- GPU --jolt --"
  JOLT_MIX_HOSTDRIVEN=1 JOLT_DEBUG=1 $BIN --jolt -te "$TREE" -s "$ALN" $ARGS -nt 12 -pre "$WB/${NAME}_jolt" -redo > "$WB/${NAME}_jolt.console" 2>&1
  echo "    exit=$?"
  grep -aE '\[JOLTMIX-RESCALE-DBG\]|\[JOLTMIX\] model=' "$WB/${NAME}_jolt.console" | sed 's/^/    /'
  GLEN=$(tlen "$WB/${NAME}_jolt.iqtree")
  echo "  -- CPU standard (no --jolt) --"
  $BIN -te "$TREE" -s "$ALN" $ARGS -nt 12 -pre "$WB/${NAME}_cpu" -redo > "$WB/${NAME}_cpu.console" 2>&1
  echo "    exit=$?"
  CLEN=$(tlen "$WB/${NAME}_cpu.iqtree")
  echo "  GPU(jolt) treeLen=$GLEN   CPU treeLen=$CLEN"
  python3 - "$GLEN" "$CLEN" <<'PY'
import sys
g,c=sys.argv[1],sys.argv[2]
try:
    g=float(g); c=float(c)
    print("    ratio GPU/CPU = %.8f   CPU/GPU = %.8f   |GPU-CPU|/CPU = %.3e"%(g/c, c/g, abs(g-c)/c))
except Exception as e:
    print("    (parse fail GPU=%s CPU=%s)"%(g,c))
PY
done
echo "════ DONE $(date -Iseconds) ════"
