#!/bin/bash
# run_gpu_treesearch_1m_energy.sh — GPU TREE-SEARCH phase + energy for the AA-1M parity table.
# CTF (run_ctf_1m_energy.sh) gives the ModelFinder phase; this gives the tree-search phase so the parity table has a
# GPU number directly comparable to the CPU -m TEST tree phase (np2 7868.9s). Runs a FULL ML tree search under the
# selected model with --jolt (NNI topology moves on CPU; every branch+param optimize call routed to JOLT on the GPU),
# same seed as the CPU baseline (-seed 1). Reports tree-search wall, final lnL, BIC, GPU energy (nvidia-smi integrator).
# Submit: H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=180GB -v ALABEL=h200,MODEL=LG+G4 run_gpu_treesearch_1m_energy.sh
#         A100: qsub -q dgxa100   -l ngpus=1 -l ncpus=16 -l mem=180GB -v ALABEL=a100,MODEL=LG+G4 run_gpu_treesearch_1m_energy.sh
#PBS -N gputree
#PBS -P dx61
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-gpu}"; NT="${PBS_NCPUS:-12}"; MODEL="${MODEL:-LG+G4}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
NFULL=946439
WB="$SRC/gputree_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ GPU tree-search AA-1M model=$MODEL on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT seed=1 ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "CPU -m TEST tree phase baseline (np2): 7868.9s ; total -m TEST np2 10945.8s ; oracle final lnL -78605196.44 LG+G4"

PWLOG="$WB/power.log"; ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 & PWPID=$!
export JOLT_DEBUG=1
T0=$(date +%s)
"$BIN" --jolt --gpu -m "$MODEL" -s "$ALN" -nt "$NT" -seed 1 -pre "$WB/tree" -redo > "$WB/tree.stdout" 2>&1
RC=$?; T_TREE=$(($(date +%s)-T0)); kill $PWPID 2>/dev/null; sleep 1
echo "iqtree exit=$RC  tree-search wall=${T_TREE}s"

echo; echo "════════ RESULT + ENERGY (${ALABEL}) ════════"
lnl=$(grep -iE "BEST SCORE FOUND|Log-likelihood of the tree" "$WB/tree.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
bic=$(grep -iE "Bayesian information criterion|BIC" "$WB/tree.iqtree" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
jn=$(grep -c '\[JOLT\] model' "$WB/tree.stdout" 2>/dev/null)
echo "  final lnL=${lnl:-NA}  BIC=${bic:-NA}  JOLT_calls=${jn:-0}  (oracle lnL -78605196.44)"
python3 - <<PY
v=[float(x) for x in open("$PWLOG") if x.strip() and x.strip()[0].isdigit()]; dt=2.0
J=sum(v)*dt
print(f"  GPU ENERGY: {J:.0f} J = {J/3600:.2f} Wh   (mean {sum(v)/max(len(v),1):.0f} W over {len(v)*dt:.0f}s, n={len(v)})")
print(f"  tree-search wall {$T_TREE}s vs CPU np2 7868.9s -> {7868.9/max($T_TREE,1):.2f}x")
PY
echo "════════ DONE $(date -Iseconds) ════════"
