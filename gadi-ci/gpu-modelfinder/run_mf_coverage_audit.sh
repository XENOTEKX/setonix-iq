#!/bin/bash
# run_mf_coverage_audit.sh — EMPIRICAL GPU coverage map for the FULL `-m MF` candidate set (DNA + AA).
# Subsamples both 1M alignments to ~5000 sites and runs `--jolt --gpu -m MF -nt N` with JOLT_DEBUG=1, then tallies
# per-model [JOLT] engagements vs [JOLT-GATE] decline reasons. `-m MF` (not -m TEST) includes +R FreeRate models, so
# this shows exactly how much of the ultimate test currently reaches the GPU and what falls to CPU and WHY — the
# scoping measurement for phase G.5. Cheap (subsample, ~10 min on any GPU).
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=00:40:00 gadi-ci/gpu-modelfinder/run_mf_coverage_audit.sh
#PBS -N mfcov
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
AA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
DNA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/mfcov_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
echo "════════ -m MF GPU coverage audit — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name --format=csv,noheader

subsample(){ python3 - "$1" "$2" "$3" <<'PY'
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
print(f"wrote {out} {len(seqs)}x{K}")
PY
}
subsample "$AA" 5000 "$WB/aa.phy"
subsample "$DNA" 5000 "$WB/dna.phy"

export JOLT_DEBUG=1
for D in aa dna; do
  echo; echo "──────── $D : --jolt --gpu -m MF ────────"
  "$BIN" --jolt --gpu -m MF -s "$WB/$D.phy" -nt "$NT" -seed 1 -pre "$WB/${D}_mf" -redo > "$WB/${D}_mf.out" 2>&1
  echo "  exit=$? best=$(grep -iE 'Best-fit model according to BIC' "$WB/${D}_mf.iqtree" 2>/dev/null | head -1)"
  echo "  [JOLT] engagements (model -> iters):"
  grep -hoE '\[JOLT\] model=[^ ]+.*' "$WB/${D}_mf.out" 2>/dev/null | sed 's/^/      /' | head -40
  echo "  ENGAGE count: $(grep -c '\[JOLT\] model=' "$WB/${D}_mf.out" 2>/dev/null)"
  echo "  [JOLT-GATE] decline reasons (tally):"
  grep -hoE '\[JOLT-GATE\] decline reason=[^ ]+' "$WB/${D}_mf.out" 2>/dev/null | sort | uniq -c | sort -rn | sed 's/^/      /'
  echo "  total models in candidate set (from .iqtree ModelFinder table rows): $(grep -cE '^\S+\s+-?[0-9]+\.[0-9]+\s+[0-9]+' "$WB/${D}_mf.iqtree" 2>/dev/null)"
done
echo; echo "════════ DONE $(date -Iseconds) ════════"
