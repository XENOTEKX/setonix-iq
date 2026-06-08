#!/bin/bash
# run_g2_multimodel_v100.sh — Phase G.2.1b MULTI-MODEL gate (parametrised by $TAG via qsub -v TAG=...).
#
# Validates the GPU full-branch-opt seam beyond the single LG+G4 case, on the SAME binary built for job
# 170259325 (NO rebuild — so multiple TAGs can run as parallel jobs without a make race). Three TAGs:
#   WAG_G4      AA  WAG+G4    GPU-handled (20-state)  -> expect all markers + GPU lnL/brlen bit-identical to CPU
#   DNA_GTR_G4  DNA GTR+G4    GPU-handled (4-state!)  -> first in-tree 4-state test; same bit-identical gate
#   WAG_I_G4    AA  WAG+I+G4  +I -> CPU fallback        -> expect [GPU-KERNEL] install + [GPU-BRANCH] CPU-fallback,
#                                                          NO [GPU-DERV]/[GPU-FROMBUF] active markers, lnL==CPU
# Each: GPU run (--gpu) vs CPU run (same binary, no --gpu); compare final lnL (.iqtree) + branch-length vector
# (.treefile). GATES: GPU-handled -> lnL rel<=1e-9 + brlen worst_rel<=1e-6; fallback -> markers + lnL rel<=1e-9.
#
#PBS -N g2-mm
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
export CC=gcc CXX=g++

SRC=/scratch/rc29/as1708/iqtree3-gpu
BIN=$SRC/build-gpu-on/iqtree3
AA_ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
AA_TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
DNA_ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
DNA_TREE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/tree_1.full.treefile

TAG="${TAG:-WAG_G4}"
case "$TAG" in
  WAG_G4)      ALN=$AA_ALN;  TREE=$AA_TREE;  MODEL="WAG+G4";    MODE="gpu" ;;
  DNA_GTR_G4)  ALN=$DNA_ALN; TREE=$DNA_TREE; MODEL="GTR+G4";    MODE="gpu" ;;
  WAG_I_G4)    ALN=$AA_ALN;  TREE=$AA_TREE;  MODEL="WAG+I+G4";  MODE="fallback" ;;
  *) echo "unknown TAG=$TAG"; exit 1 ;;
esac
RUNDIR=$SRC/g2_mm_runs/$TAG; mkdir -p "$RUNDIR"

echo "════════ G.2 MULTI-MODEL [$TAG : $MODEL, mode=$MODE] — $(hostname) $(date -Iseconds) ════════"
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
echo "binary: $(ls -la "$BIN" | awk '{print $5,$6,$7,$8}')   (no rebuild)"

echo; echo "════════ GPU run: --gpu -m $MODEL ════════"
T0=$(date +%s)
"$BIN" --gpu -s "$ALN" -te "$TREE" -m "$MODEL" -nt 1 -pre "$RUNDIR/gpu" -redo 2>&1 | \
  grep -E "GPU-KERNEL|GPU-BRANCH|GPU-DERV|GPU-FROMBUF|GPU-XCHECK|Optimal log-likelihood" | head -40
T1=$(date +%s); echo "[GPU wall] $((T1-T0)) s"

echo; echo "════════ CPU run: -m $MODEL (reference) ════════"
T2=$(date +%s)
"$BIN" -s "$ALN" -te "$TREE" -m "$MODEL" -nt 1 -pre "$RUNDIR/cpu" -redo 2>&1 | \
  grep -E "Optimal log-likelihood" | head -5
T3=$(date +%s); echo "[CPU wall] $((T3-T2)) s"

echo; echo "════════ COMPARE [$TAG] ════════"
G=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/gpu.iqtree" 2>/dev/null | head -1 | grep -oE "\-?[0-9.]+$")
C=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/cpu.iqtree" 2>/dev/null | head -1 | grep -oE "\-?[0-9.]+$")
echo "GPU lnL=${G:-NA}  CPU lnL=${C:-NA}"
python3 - "$RUNDIR/gpu.treefile" "$RUNDIR/cpu.treefile" "${G:-nan}" "${C:-nan}" "$MODE" <<'PY'
import sys, re
gtf, ctf, g, c, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
def brlens(f):
    s = open(f).read()
    return sorted(float(x) for x in re.findall(r':([0-9.]+(?:[eE][-+]?[0-9]+)?)', s))
try:
    rl = abs((g-c)/c) if c else abs(g-c)
    print(f"lnL: |d|={abs(g-c):.3e} rel={rl:.3e} -> {'PASS' if rl<=1e-9 else 'CHECK'}  (gate 1e-9)")
    gb, cb = brlens(gtf), brlens(ctf)
    if len(gb)!=len(cb):
        print(f"BRLEN: count mismatch GPU={len(gb)} CPU={len(cb)} -> CHECK"); sys.exit(0)
    worst=max((abs(a-b)/b if b>1e-12 else abs(a-b)) for a,b in zip(gb,cb)) if gb else 0.0
    print(f"BRLEN: n={len(gb)} worst_rel={worst:.3e} -> {'PASS' if worst<=1e-6 else 'CHECK'}  (gate 1e-6)")
except Exception as e:
    print("compare error:", e)
PY

echo; echo "════════ DONE [$TAG] $(date -Iseconds) ════════"
