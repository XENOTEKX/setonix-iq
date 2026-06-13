#!/bin/bash
# run_freeqcheck.sh — G.6.0b: the free-Q OPTIMIZER convergence gate. The G.6.0a FD-check proved the GPU lnL is
# bit-exact under a moving eigensystem; this proves the JOLT FD-LM over the exchangeabilities CONVERGES to the same
# MLE as IQ-TREE's own (BFGS forward-FD) optimiser — the load-bearing risk (GTR's 5 COUPLED exchangeabilities under a
# diagonal LM). For each model: a CPU baseline and a --jolt run (JOLT_FREEQ=1 opens the env-gated free-Q path), on the
# SAME fixed topology (-te tree) so the ONLY difference is the optimiser. Compare final tree lnL + the in-tree [JOLT]
# write-back self-check. GATE: jolt_lnL >= cpu_lnL - 0.01 nat (JOLT not stuck at a worse optimum, cf. the +I 39.5-nat
# precedent) AND [JOLT] write-back rel <= 1e-6. nQ spans 1 (HKY), 2 (TN), 4 (TVM), 5 (GTR empirical / SYM equal).
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=01:00:00 -l storage=scratch/dx61+scratch/rc29 -l wd run_freeqcheck.sh
#PBS -N freeqchk
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
DNA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/freeqcheck_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ G.6.0b free-Q optimiser convergence gate — $(hostname) $(date -Iseconds) ════════"
ls -l --time-style=+%Y-%m-%dT%H:%M "$BIN"; nvidia-smi --query-gpu=name --format=csv,noheader

python3 - "$DNA" 5000 "$WB/dna.phy" <<'PY'
import sys,random
src,K,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
with open(src) as f:
    f.readline(); names=[];seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]);seqs.append(p[1].replace(" ",""))
L=len(seqs[0]);random.seed(1);cols=sorted(random.sample(range(L),K))
open(out,"w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print(f"wrote dna.phy {len(seqs)}x{K} (seed 1)")
PY

echo "──── fixed topology (CPU, GTR+G4 -fast) ────"
"$BIN" -s "$WB/dna.phy" -m GTR+G4 -fast -seed 1 -nt 4 -pre "$WB/tre" -redo > "$WB/tre.out" 2>&1
echo "  tree exit=$?"
TREE="$WB/tre.treefile"

lnl() { grep -E 'Log-likelihood of the tree:' "$1" 2>/dev/null | tail -1 | grep -oE '[-0-9.]+$'; }

printf "\n%-12s | %-18s | %-18s | %-14s | %s\n" "model(nQ)" "CPU lnL" "JOLT lnL" "jolt-cpu" "verdict"
echo "-----------------------------------------------------------------------------------------------------"
for spec in "HKY+F+G4:1" "TN+F+G4:2" "TVM+F+G4:4" "GTR+F+G4:5" "SYM+G4:5"; do
  M="${spec%%:*}"; NQ="${spec##*:}"; TAG="${M//+/_}"
  # CPU baseline (no --gpu): IQ-TREE's own BFGS Q-optimiser on the fixed topology
  "$BIN" -s "$WB/dna.phy" -m "$M" -te "$TREE" -seed 1 -nt 1 -pre "$WB/cpu_$TAG" -redo > "$WB/cpu_$TAG.out" 2>&1
  # JOLT free-Q (--jolt + JOLT_FREEQ opens the env-gated path)
  JOLT_FREEQ=1 JOLT_DEBUG=1 "$BIN" --jolt --gpu -s "$WB/dna.phy" -m "$M" -te "$TREE" -seed 1 -nt 1 -pre "$WB/jolt_$TAG" -redo > "$WB/jolt_$TAG.out" 2>&1
  CL=$(lnl "$WB/cpu_$TAG.iqtree"); JL=$(lnl "$WB/jolt_$TAG.iqtree")
  if [ -n "$CL" ] && [ -n "$JL" ]; then
    DIFF=$(python3 -c "print(f'{($JL)-($CL):+.6f}')" 2>/dev/null)
    OK=$(python3 -c "print('PASS' if ($JL) >= ($CL)-0.01 else 'STALL')" 2>/dev/null)
  else DIFF="?"; OK="NO-LNL"; fi
  printf "%-12s | %-18s | %-18s | %-14s | %s\n" "$M($NQ)" "${CL:-?}" "${JL:-?}" "$DIFF" "$OK"
  grep -hE '\[JOLT\] model=' "$WB/jolt_$TAG.out" 2>/dev/null | tail -1 | sed 's/^/    /'
  grep -hE '\[JOLT-GATE\] decline' "$WB/jolt_$TAG.out" 2>/dev/null | tail -1 | sed 's/^/    /'
done
echo "════════ DONE $(date -Iseconds) ════════"
