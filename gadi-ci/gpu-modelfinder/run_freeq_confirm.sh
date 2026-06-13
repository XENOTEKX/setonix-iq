#!/bin/bash
# run_freeq_confirm.sh — G.6.1 final confirm: free-Q is ON BY DEFAULT (no JOLT_FREEQ env needed) after the gate flip,
# and the JOLT_NO_FREEQ escape hatch declines. Also confirms a free-Q+I+G model engages and the safety gate doesn't
# spuriously trip. Quick -te runs on the 5000-site DNA subsample.
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=00:20:00 -l storage=scratch/dx61+scratch/rc29 -l wd run_freeq_confirm.sh
#PBS -N freeqcfm
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
DNA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/freeqcfm_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ G.6.1 confirm (free-Q default-on) — $(hostname) $(date -Iseconds) ════════"
ls -l --time-style=+%Y-%m-%dT%H:%M "$BIN"
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
PY
"$BIN" -s "$WB/dna.phy" -m GTR+G4 -fast -seed 1 -nt 4 -pre "$WB/tre" -redo > "$WB/tre.out" 2>&1
TREE="$WB/tre.treefile"

echo "──── (1) GTR+F+G4 -te, NO JOLT_FREEQ env (expect ENGAGE = default-on) ────"
JOLT_DEBUG=1 "$BIN" --jolt --gpu -s "$WB/dna.phy" -m GTR+F+G4 -te "$TREE" -seed 1 -nt 1 -pre "$WB/a" -redo > "$WB/a.out" 2>&1
grep -hE '\[JOLT\] model=|\[JOLT-GATE\] decline' "$WB/a.out" | tail -2 | sed 's/^/    /'

echo "──── (2) GTR+F+I+G4 -te, NO env (expect ENGAGE, free-Q + I jointly) ────"
JOLT_DEBUG=1 "$BIN" --jolt --gpu -s "$WB/dna.phy" -m GTR+F+I+G4 -te "$TREE" -seed 1 -nt 1 -pre "$WB/b" -redo > "$WB/b.out" 2>&1
grep -hE '\[JOLT\] model=|\[JOLT-GATE\] decline' "$WB/b.out" | tail -2 | sed 's/^/    /'

echo "──── (3) GTR+F+G4 -te, JOLT_NO_FREEQ=1 (expect DECLINE = escape hatch) ────"
JOLT_NO_FREEQ=1 JOLT_DEBUG=1 "$BIN" --jolt --gpu -s "$WB/dna.phy" -m GTR+F+G4 -te "$TREE" -seed 1 -nt 1 -pre "$WB/c" -redo > "$WB/c.out" 2>&1
grep -hE '\[JOLT\] model=|\[JOLT-GATE\] decline reason=free-subst' "$WB/c.out" | tail -2 | sed 's/^/    /'
echo "════════ DONE $(date -Iseconds) ════════"
