#!/bin/bash
# run_dna_mf_coverage.sh — G.6.1 validation: the DNA -m MF COVERAGE payoff. With free-Q JOLT (JOLT_FREEQ=1) the DNA
# ModelFinder candidates that used to decline at "free-subst-params" (HKY..GTR, the 62-of-90 dominant DNA gap) now
# ENGAGE the GPU. Compare GPU (--jolt) vs a CPU -m MF baseline on a 5000-site subsample of the GTR+I+G4 1M data:
#   (1) COVERAGE: count [JOLT] engagements vs [JOLT-GATE] declines + reasons (expect free-Q now engages; only
#       +R/+I+R (non-mean-gamma) + pure-+I still decline).
#   (2) CORRECTNESS: GPU best-fit model == CPU best-fit model (BIC), and per-candidate write-back rel ~1e-12.
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=01:30:00 -l storage=scratch/dx61+scratch/rc29 -l wd run_dna_mf_coverage.sh
#PBS -N dnamfcov
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
DNA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/dnamfcov_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ G.6.1 DNA -m MF coverage (free-Q JOLT) — $(hostname) $(date -Iseconds) ════════"
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

echo "──── CPU baseline (-m MF, no --jolt) ────"
/usr/bin/time -v "$BIN" -s "$WB/dna.phy" -m MF -seed 1 -nt 12 -pre "$WB/cpu" -redo > "$WB/cpu.out" 2>&1
echo "  cpu exit=$?"

echo "──── GPU (--jolt --gpu -m MF, JOLT_FREEQ=1) ────"
JOLT_FREEQ=1 JOLT_DEBUG=1 /usr/bin/time -v "$BIN" --jolt --gpu -s "$WB/dna.phy" -m MF -seed 1 -nt 12 -pre "$WB/jolt" -redo > "$WB/jolt.out" 2>&1
echo "  jolt exit=$?"

echo
echo "════════ COVERAGE (GPU run) ════════"
echo -n "  JOLT engagements: "; grep -cE '^\[JOLT\] model=' "$WB/jolt.out" 2>/dev/null
echo -n "  JOLT-GATE declines: "; grep -cE '\[JOLT-GATE\] decline' "$WB/jolt.out" 2>/dev/null
echo "  decline reasons (count):"
grep -oE '\[JOLT-GATE\] decline reason=[a-z-]+' "$WB/jolt.out" 2>/dev/null | sort | uniq -c | sed 's/^/    /'
echo "  free-Q engagements (HKY/TN/K81/TPM/TIM/TVM/SYM/GTR):"
grep -oE '^\[JOLT\] model=[A-Za-z0-9+.]+' "$WB/jolt.out" 2>/dev/null | grep -oE 'model=[A-Za-z0-9+.]+' | sort -u | grep -viE 'model=(JC|F81)' | sed 's/^/    /' | head -40
echo "  worst write-back rel (free-Q + all):"
grep -oE 'rel=[0-9.e+-]+ (PASS|OK|MISMATCH)' "$WB/jolt.out" 2>/dev/null | sort -t= -k2 -g | tail -3 | sed 's/^/    /'

echo
echo "════════ CORRECTNESS ════════"
echo -n "  CPU  best-fit (BIC): "; grep -E 'Best-fit model according to BIC' "$WB/cpu.iqtree" 2>/dev/null | tail -1
echo -n "  GPU  best-fit (BIC): "; grep -E 'Best-fit model according to BIC' "$WB/jolt.iqtree" 2>/dev/null | tail -1
CB=$(grep -E 'Best-fit model according to BIC' "$WB/cpu.iqtree" 2>/dev/null | tail -1 | grep -oE ': .*' | sed 's/: //')
JB=$(grep -E 'Best-fit model according to BIC' "$WB/jolt.iqtree" 2>/dev/null | tail -1 | grep -oE ': .*' | sed 's/: //')
[ -n "$CB" ] && [ "$CB" = "$JB" ] && echo "  => BEST-MODEL MATCH: $CB" || echo "  => BEST-MODEL DIFFER: cpu='$CB' gpu='$JB'"
echo -n "  GPU wall: "; grep -E 'wall clock' "$WB/jolt.out" 2>/dev/null | tail -1
echo "════════ DONE $(date -Iseconds) ════════"
