#!/bin/bash
# run_rgradcheck.sh — G.5.1a: FD-validate the +R FreeRate WEIGHT gradient (gz_c = WN_c − w_c·N, PART IX §IX.8) on the
# real GPU path, BEFORE wiring the +R optimiser. Runs the gated JOLT_RGRADCHECK hook on PURE +R models (LG+R4, LG+R6)
# over the same 5000-site AA subsample (seed 1). The hook computes the weight gradient + central finite differences and
# prints [RGRADCHECK] lines; +R then declines to CPU as usual (the optimiser branch is G.5.1b). GATE: max FD rel < 1e-4
# AND |ΣWN−N|/N < 1e-9 (identity Σ_c WN_c = N). A wrong gradient cannot pass this — the G.4.0b discipline.
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=00:30:00 -l storage=scratch/dx61+scratch/rc29 -l wd run_rgradcheck.sh
#PBS -N rgradchk
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NT=1   # single-thread: the check is a one-shot gated hook, not a throughput run
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
AA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/rgradcheck_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ G.5.1a +R weight-gradient FD self-check — $(hostname) $(date -Iseconds) ════════"
ls -l --time-style=+%Y-%m-%dT%H:%M "$BIN"; nvidia-smi --query-gpu=name --format=csv,noheader
python3 - "$AA" 5000 "$WB/aa.phy" <<'PY'
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
print("wrote aa.phy 100x5000 (seed 1, == audit 170602983)")
PY
export JOLT_RGRADCHECK=1 JOLT_DEBUG=1
for M in LG+R4 LG+R6; do
  echo "──── $M (pure +R; JOLT_RGRADCHECK) ────"
  "$BIN" --jolt --gpu -m "$M" -s "$WB/aa.phy" -nt "$NT" -seed 1 -pre "$WB/${M//+/_}" -redo > "$WB/${M//+/_}.out" 2>&1
  echo "  exit=$?"
  grep -E '\[RGRADCHECK\]' "$WB/${M//+/_}.out" 2>/dev/null | sed 's/^/    /'
  grep -E 'RGRADCHECK PASS|RGRADCHECK FAIL' "$WB/${M//+/_}.out" >/dev/null 2>&1 && echo "    => $(grep -oE 'RGRADCHECK (PASS|FAIL)' "$WB/${M//+/_}.out" | tail -1)" || echo "    => NO RGRADCHECK OUTPUT (hook did not fire — check gate)"
done
echo "════════ DONE $(date -Iseconds) ════════"
