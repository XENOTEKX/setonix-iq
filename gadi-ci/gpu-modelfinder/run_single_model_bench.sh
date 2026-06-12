#!/bin/bash
# run_single_model_bench.sh — single-model JOLT benchmark at 10K/100K/1M
# Matches the format of the external A100 table (single model, fixed tree, one alignment size).
# Runs LG+G4 and LG+I+G4 on AA alignments at three sizes.
# Submit:
#   H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=200GB -v ALABEL=h200 run_single_model_bench.sh
#   A100: qsub -q dgxa100   -l ngpus=1 -l ncpus=16 -l mem=200GB -v ALABEL=a100 run_single_model_bench.sh
#   V100: qsub -q gpuvolta  -l ngpus=1 -l ncpus=12 -l mem=60GB  -v ALABEL=v100 run_single_model_bench.sh
#
#PBS -N smb
#PBS -P dx61
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

ALABEL="${ALABEL:-gpu}"; NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu
BIN="$SRC/build-gpu-on/iqtree3"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100

WB="$SRC/smb_${ALABEL}"; mkdir -p "$WB"
[[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN"; exit 1; }

echo "════════ Single-model JOLT benchmark on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo
printf "%-10s %-12s %10s %10s\n" "Size" "Model" "wall(s)" "lnL"
echo "----------------------------------------------"

# ── benchmark function ───────────────────────────────────────────────
run_one() {
    local SIZE=$1 MODEL=$2
    local ALN="$BASE/len_${SIZE}/tree_1/alignment_${SIZE}.phy"
    local PRE="$WB/${MODEL//+/_}_${SIZE}"
    [[ -f "$ALN" ]] || { printf "%-10s %-12s %10s\n" "$SIZE" "$MODEL" "MISSING_ALN"; return; }

    # use NJ tree from a quick parsimony run as fixed tree (matches their "-te" parity)
    local TREE="${PRE}_tree.treefile"
    if [[ ! -f "$TREE" ]]; then
        "$BIN" -m LG+G4 -s "$ALN" -nt "$NT" -pre "${PRE}_tree" \
            --tree-fix -redo > "${PRE}_tree.stdout" 2>&1
        # fall back: just use the treefile produced
        TREE="${PRE}_tree.treefile"
    fi
    [[ -f "$TREE" ]] || TREE=""

    local T0; T0=$(date +%s)
    if [[ -n "$TREE" ]]; then
        "$BIN" --jolt --gpu -m "$MODEL" -s "$ALN" -te "$TREE" \
            -nt "$NT" -pre "$PRE" -redo > "${PRE}.stdout" 2>&1
    else
        "$BIN" --jolt --gpu -m "$MODEL" -s "$ALN" \
            -nt "$NT" -pre "$PRE" -redo > "${PRE}.stdout" 2>&1
    fi
    local WALL=$(( $(date +%s) - T0 ))
    local LNL; LNL=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND" "${PRE}.log" 2>/dev/null \
                     | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
    printf "%-10s %-12s %10s %10s\n" "$SIZE" "$MODEL" "${WALL}s" "${LNL:-NA}"
}

for SIZE in 10000 100000 1000000; do
    for MODEL in LG+G4 "LG+I+G4"; do
        # 1M needs H200/A100 — skip on V100 (OOM)
        if [[ "$SIZE" == "1000000" && "$ALABEL" == "v100" ]]; then
            printf "%-10s %-12s %10s\n" "$SIZE" "$MODEL" "SKIP(OOM)"
            continue
        fi
        run_one "$SIZE" "$MODEL"
    done
done

echo
echo "════════ Reference: External A100 table (single-model, no JOLT) ════════"
printf "%-10s %-12s %10s\n" "Size" "Model" "wall(s)"
printf "%-10s %-12s %10s\n" "10K"   "AA(LG+I+G4)" "37.9s"
printf "%-10s %-12s %10s\n" "100K"  "AA(LG+I+G4)" "271.4s"
printf "%-10s %-12s %10s\n" "1M"    "AA(LG+I+G4)" "3482.8s"
echo "  (Note: their numbers are single computeLikelihood/optimizeBranch — not ModelFinder)"
echo "════════ DONE $(date -Iseconds) ════════"
