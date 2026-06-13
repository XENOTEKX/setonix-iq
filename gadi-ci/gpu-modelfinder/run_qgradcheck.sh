#!/bin/bash
# run_qgradcheck.sh — G.6.0a: FD-validate the free-Q (DNA exchangeability) gradient pipeline on the real GPU path,
# BEFORE building the G.6.0b free-Q JOLT optimiser. The gated gpuFreeQGradCheckOnce hook (env JOLT_QGRADCHECK) fires
# under --gpu (NOT --jolt): for each free exchangeability of a reversible DNA free-Q model it perturbs the param in
# rate-class space (param_spec), re-decomposes the 4x4 Q, re-uploads eval/U/Uinv, runs the GPU clean-room sweep, and
# compares GPU lnL vs CPU computeLikelihood at the perturbed Q (+ FD-grad). GATE: |GPU-CPU|/|CPU| <= 1e-9 at base AND
# every perturbed Q. A wrong eigendecompose->reupload->resweep pipeline cannot pass this — the G.4.0b discipline.
#
# Models span nQ = 1,2,4,5 across BOTH freq types (empirical +F and equal +FQ/SYM): bare HKY/GTR default to FREQ_ESTIMATE
# (+FO) which the hook SKIPS, so fixed-freq variants are forced. -te <tree> -blfix => no branch-opt (the slow stateless
# Derv path is never hit under --gpu); the model is fit + the hook fires on the initial GPU clean-room eval.
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=00:30:00 -l storage=scratch/dx61+scratch/rc29 -l wd run_qgradcheck.sh
#PBS -N qgradchk
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
DNA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/qgradcheck_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ G.6.0a free-Q gradient FD self-check — $(hostname) $(date -Iseconds) ════════"
ls -l --time-style=+%Y-%m-%dT%H:%M "$BIN"; nvidia-smi --query-gpu=name --format=csv,noheader

# ---- 5000-site DNA subsample (seed 1, same recipe as the AA audit) ----
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

# ---- fixed topology + brlens (one fast CPU fit; reused by every check via -te -blfix) ----
echo "──── building fixed tree (CPU, GTR+G4, -fast) ────"
"$BIN" -s "$WB/dna.phy" -m GTR+G4 -fast -seed 1 -nt 4 -pre "$WB/tre" -redo > "$WB/tre.out" 2>&1
echo "  tree exit=$?  $(ls -l "$WB/tre.treefile" 2>/dev/null | awk '{print $5" bytes"}')"

# ---- the gradient checks (nQ = 1,2,4,5; empirical + equal freq) ----
export JOLT_QGRADCHECK=1 JOLT_DEBUG=1
for M in HKY+F+G4 TNe+G4 TVM+F+G4 SYM+G4 GTR+F+G4; do
  TAG="${M//+/_}"
  echo "──── $M (free-Q; --gpu; JOLT_QGRADCHECK; -te -blfix) ────"
  "$BIN" --gpu -m "$M" -s "$WB/dna.phy" -te "$WB/tre.treefile" -blfix -seed 1 -nt 1 -pre "$WB/$TAG" -redo > "$WB/$TAG.out" 2>&1
  echo "  exit=$?"
  grep -E '\[QGRADCHECK\]' "$WB/$TAG.out" 2>/dev/null | sed 's/^/    /'
  if grep -qE 'QGRADCHECK PASS' "$WB/$TAG.out" 2>/dev/null; then echo "    => PASS";
  elif grep -qE 'QGRADCHECK FAIL' "$WB/$TAG.out" 2>/dev/null; then echo "    => FAIL";
  else echo "    => NO QGRADCHECK OUTPUT (hook skipped — check gate: freqtype/+I/getNDim)"; grep -E '\[QGRADCHECK\] skipped|\[JOLT-GATE\]' "$WB/$TAG.out" 2>/dev/null | head -3 | sed 's/^/      /'; fi
done
echo "════════ DONE $(date -Iseconds) ════════"
