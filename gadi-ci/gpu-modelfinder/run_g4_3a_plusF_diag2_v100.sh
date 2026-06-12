#!/bin/bash
# run_g4_3a_plusF_diag2_v100.sh — G.4.3a +F diagnostic, ATTEMPT 2 (decisive, fast).
#
# Attempt 1 (job 170380392) timed out at -nt 1 before reaching the +F models (only 3/8 LG models ranked in 27 min).
# It DID confirm: model-freq variants (freqtype=1) reach the JOLT hook; +I/+I+G decline reason=pinvar (G.4.3b target).
# OPEN: do +F (freqtype=3) models reach optimizeParametersJOLT at all? -> evaluate EXPLICIT single models on a FIXED
# tree (one fast optimizeParameters call each, no model selection, no tree search). JOLT_DEBUG=1 logs the gate decision.
#   - LG+F+G4    : +F, NO +I, gamma -> if +F reaches the hook it should ENGAGE (getNDim()==0 for FREQ_EMPIRICAL)
#   - LG+F+I+G4  : +F + I -> should reach hook, decline reason=pinvar
#   - LG+G4      : control (known to engage)
# VERDICT: a [JOLT-GATE] line with freqtype=3 => +F REACHES the hook (mechanism b: see engage/decline);
#          NO freqtype=3 line for LG+F+G4 => +F never dispatched here (mechanism a).
# NO REBUILD: the build-gpu-on binary already carries the JOLT_DEBUG instrumentation (job 170380392's build).
#
#PBS -N g4-3a-diag2
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:20:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB="$SRC/g4_3a_diag2"; mkdir -p "$WB"; NT=4
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
echo "════════ G.4.3a +F DIAGNOSTIC 2 — $(hostname) $(date -Iseconds) ════════"
echo "binary $(ls -la "$BIN" | awk '{print $6,$7,$8}')  HEAD $(cd "$SRC" && git rev-parse --short HEAD)"
export JOLT_DEBUG=1
for M in "LG+F+G4" "LG+F+I+G4" "LG+G4"; do
  tag=$(echo "$M" | tr '+' '_')
  echo; echo "──── -m $M -te <fixed> -nt $NT ────"
  T0=$(date +%s)
  "$BIN" --jolt -m "$M" -s "$ALN" -te "$TREE" -nt $NT -pre "$WB/$tag" -redo > "$WB/$tag.stdout" 2> "$WB/$tag.stderr"
  echo "  exit $? wall $(( $(date +%s)-T0 ))s"
  echo "  [JOLT-GATE]:"; grep -E '^\[JOLT-GATE\]' "$WB/$tag.stderr" 2>/dev/null | sed 's/^/    /'
  echo "  [JOLT] engage:"; grep -E '^\[JOLT\]' "$WB/$tag.stdout" 2>/dev/null | sed 's/^/    /'
done
echo
echo "════════ VERDICT ════════"
NF=$(grep -hE '^\[JOLT-GATE\] reached hook' "$WB/LG_F_G4.stderr" 2>/dev/null | grep -c 'freqtype=3')
echo "  LG+F+G4 reached-hook lines with freqtype=3 (+F EMPIRICAL): $NF"
echo "  >0 => +F REACHES optimizeParametersJOLT (mechanism b — engage or see decline reason)"
echo "  ==0 => +F is NOT dispatched to the hook here (mechanism a — staged/different path)"
echo "════════ DONE $(date -Iseconds) ════════"
