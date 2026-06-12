#!/bin/bash
# run_jolt_refine_1m.sh — measure JOLT refining the JOLT-ELIGIBLE top models at FULL 1M on one GPU.
#
# WHY THIS RUN (the run_ctf_1m.sh post-mortem):
#   run_ctf_1m.sh refined top-k in BIC-rank order. The subsample scale-consistent BIC ranked LG+I+G4 #1
#   (a THIN-MARGIN flip: +I collapses, subLogL gap to LG+G4 was 0.097 nat on 5k sites; true full-data #1 is
#   LG+G4 by +14 BIC). So refine_1 = LG+I+G4, which is JOLT-INELIGIBLE (pinvar) -> PURE CPU. Measured on the
#   A100 node: ONE of 10 +I+G start-values took 8712 s on 16 cores, GPU 0% idle. 10 start-values ~= 24 h.
#   ONE +I start-value alone (8712 s) is already 2.8x the entire 2-SPR-node wall (3076.9 s). The +I CPU
#   heavy-tail on a 16-core GPU node is the killer; the GPU never engaged. This is the part5/part6 prediction
#   made concrete (N/S ceiling: 16 GPU-node cores vs 208 SPR-cluster cores; +I can't use the GPU).
#
# WHAT THIS MEASURES (the part that CAN work, never measured before):
#   JOLT refining the JOLT-ELIGIBLE top models {LG+G4, LG+F+G4} on the FULL 1M alignment, on the subsample
#   fixed tree (the CTF refine). Gives: (a) first-ever JOLT wall at 1M (100K was 47 s); (b) confirms the
#   ~57 GB +G4 partials fit (A100-80 / H200-141; V100-32 would OOM); (c) lnL + JOLT iters; (d) the
#   "eligible-only" CTF wall = coarse + refine(LG+G4) + refine(LG+F+G4), and the projection IF +I were
#   GPU-eligible (the deferred G.4.3b p_inv gradient): coarse + 3x JOLT-refine.
#
# Submit:
#   A100: qsub -q dgxa100  -l ngpus=1 -l ncpus=16 -l mem=120GB -v ALABEL=a100 run_jolt_refine_1m.sh
#   H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=120GB -v ALABEL=h200 run_jolt_refine_1m.sh
#
#PBS -N jolt1m
#PBS -P dx61
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-gpu}"
NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/ctf_1m_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
TREE="$WB/coarse.treefile"   # reuse the subsample tree built by run_ctf_1m.sh (already on scratch)
NFULL=940000
MODELS=(LG+G4 LG+F+G4)       # JOLT-eligible top models from the coarse rank (the +G4 variants)

[ -x "$BIN" ]  || { echo "no binary $BIN"; exit 1; }
[ -f "$ALN" ]  || { echo "no aln $ALN"; exit 1; }
[ -f "$TREE" ] || { echo "no coarse tree $TREE (run run_ctf_1m.sh coarse first)"; exit 1; }

echo "════════ JOLT 1M REFINE on ${ALABEL} — $(hostname) $(date -Iseconds) | nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "TARGETS (CPU MF-phase, measured): np2=3076.9s  np4=1974.5s ; oracle best=LG+G4 lnL~-78605196"
echo "fixed tree = $TREE (subsample topology); models = ${MODELS[*]}"

export JOLT_DEBUG=1
declare -A WALL LNL ITERS PEAKMEM
T_R_TOTAL=0
i=0
for M in "${MODELS[@]}"; do
  i=$((i+1))
  echo; echo "──── refine $i/${#MODELS[@]}: $M (--jolt --gpu, full 1M) ────"
  # background GPU-memory sampler (peak fit check)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/gpumem_${i}.log" 2>&1 &
  SMI=$!
  T0=$(date +%s)
  "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$TREE" -nt "$NT" -pre "$WB/jolt1m_${i}" -redo \
      > "$WB/jolt1m_${i}.stdout" 2>&1
  RC=$?; T=$(( $(date +%s)-T0 )); T_R_TOTAL=$((T_R_TOTAL+T))
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND|Optimal log-likelihood" "$WB/jolt1m_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  iters=$(grep -oE '\[JOLT\] [0-9]+ iters' "$WB/jolt1m_${i}.stdout" 2>/dev/null | grep -oE '[0-9]+' | head -1)
  peak=$(sort -n "$WB/gpumem_${i}.log" 2>/dev/null | tail -1)
  WALL[$M]=$T; LNL[$M]=${lnl:-NA}; ITERS[$M]=${iters:-NA}; PEAKMEM[$M]=${peak:-NA}
  echo "  exit=$RC wall=${T}s  lnL=${lnl:-NA}  jolt_iters=${iters:-NA}  peak_gpu_mem=${peak:-NA}MiB"
  # sanity: did JOLT actually engage on the GPU (not silently fall to CPU)?
  eng=$(grep -c '\[JOLT\]' "$WB/jolt1m_${i}.stdout" 2>/dev/null)
  dec=$(grep -c 'JOLT-GATE.*decline' "$WB/jolt1m_${i}.stdout" 2>/dev/null)
  echo "  JOLT engaged lines=${eng}  declines=${dec}"
done

echo; echo "════════ JOLT 1M REFINE RESULT (${ALABEL}) ════════"
printf "%-10s %10s %18s %8s %12s\n" model wall_s full_lnL iters peakMiB
for M in "${MODELS[@]}"; do
  printf "%-10s %10s %18s %8s %12s\n" "$M" "${WALL[$M]}" "${LNL[$M]}" "${ITERS[$M]}" "${PEAKMEM[$M]}"
done
echo
# coarse wall = mtime(coarse.iqtree) - mtime(sub.phy); already done, read from the dir
TSUB_C=$(python3 - "$WB" <<'PY'
import os,sys
wb=sys.argv[1]
try:
    t=os.path.getmtime(wb+"/coarse.iqtree")-os.path.getmtime(wb+"/sub.phy")
    print(int(max(t,0)))
except Exception:
    print(140)
PY
)
echo "  coarse+subsample wall (from prior run mtimes) ~= ${TSUB_C}s"
ELIG=$((TSUB_C + T_R_TOTAL))
echo "  ELIGIBLE-ONLY CTF wall (coarse + refine ${MODELS[*]}) = ${ELIG}s"
# projection: if +I were GPU-eligible (G.4.3b), 3rd refine ~= LG+G4 refine
RG4=${WALL[LG+G4]:-0}
PROJ=$((TSUB_C + T_R_TOTAL + RG4))
echo "  PROJECTED 3-model CTF IF +I were GPU-eligible (G.4.3b) = ${PROJ}s  (coarse + 2 refines + 1 more ~LG+G4-cost)"
echo
echo "  vs np2 3076.9s: eligible-only $(python3 -c "print(f'{3076.9/$ELIG:.2f}x' if $ELIG>0 else 'NA')")  | projected-3 $(python3 -c "print(f'{3076.9/$PROJ:.2f}x' if $PROJ>0 else 'NA')")"
echo "  vs np4 1974.5s: eligible-only $(python3 -c "print(f'{1974.5/$ELIG:.2f}x' if $ELIG>0 else 'NA')")  | projected-3 $(python3 -c "print(f'{1974.5/$PROJ:.2f}x' if $PROJ>0 else 'NA')")"
echo
echo "  NOTE: +I models (LG+I+G4, LG+F+I+G4) are JOLT-ineligible -> PURE CPU; measured >=8712s/start-value"
echo "        on this 16-core node (GPU idle). A valid CTF that selects among +I/+G4 CANNOT skip them on CPU"
echo "        and beat 2 nodes; the +I-on-GPU (G.4.3b p_inv gradient) is the missing piece."
echo "════════ DONE $(date -Iseconds) ════════"
