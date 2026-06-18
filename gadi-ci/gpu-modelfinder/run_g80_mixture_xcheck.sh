#!/bin/bash
# run_g80_mixture_xcheck.sh — G.8.0 VALIDATION: clean-room GPU profile-mixture lnL == CPU computeLikelihood, rel<=1e-9,
# for LG+C20+G4, LG+C60+G4, LG+MEOW80+G4. Under --gpu (not --jolt) the one-shot [GPU-XCHECK-MIX] hook fires on the
# first computeLikelihood of a getNMixtures()>1 model and prints GPU-vs-CPU rel. The mixture model itself runs on CPU
# (the production GPU gate still declines mixtures); this validates ONLY the new k1_node_mix kernel against the live
# model. Subsampled to 5000 sites — rel is size-independent, keeps the per-internal-slot partials small (C20 ~1.4GB,
# C60 ~4.2GB, MEOW80 ~5.6GB at 5000 ptn) and the CPU reference fast.
#
# Submit: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=01:00:00 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd gadi-ci/gpu-modelfinder/run_g80_mixture_xcheck.sh
#   (A100 also fine: -q dgxa100 -l ncpus=16. The 5000-site partials fit A100-80 for all three.)
#PBS -N g80mix
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
DATA=/scratch/rc29/as1708/eukaryote_williamson2025
ALN=$DATA/CAT_100S93F.phy
NEX=$DATA/MEOW6020.nex
TREE=/scratch/rc29/as1708/iqtree3-gpu/euk_will2025_run/A_fasttree.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/g80_mix_xcheck; mkdir -p "$WB"; cd "$WB"

# deterministic 5000-site subsample (same 100 taxa -> the fixed tree still matches).
# CAT_100S93F.phy is INTERLEAVED phylip (seq split across blocks, names only in block 1) -> parse accordingly + validate.
SUB="$WB/euk5k.phy"
python3 - "$ALN" 5000 "$SUB" <<'PY'
import sys, random
src, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = [ln.rstrip('\n') for ln in open(src)]
ntax, L = map(int, lines[0].split()[:2])
body = [ln for ln in lines[1:] if ln.strip()]
names = [None]*ntax; seqs = ['']*ntax
for i in range(ntax):                       # block 1: name + first chunk
    parts = body[i].split(None, 1)
    names[i] = parts[0]; seqs[i] = parts[1].replace(' ','') if len(parts)>1 else ''
for j, ln in enumerate(body[ntax:]):        # continuation blocks: cycle taxon order, no names
    seqs[j % ntax] += ln.replace(' ','')
bad = [i for i in range(ntax) if len(seqs[i]) != L]
if bad: sys.stderr.write("PARSE FAIL: %d/%d seqs != L=%d\n"%(len(bad),ntax,L)); sys.exit(3)
random.seed(1); cols = sorted(random.sample(range(L), K))
with open(out,'w') as o:
    o.write("%d %d\n"%(ntax,K))
    for n,s in zip(names,seqs): o.write("%s  %s\n"%(n,''.join(s[c] for c in cols)))
sys.stderr.write("OK: %d taxa x %d cols\n"%(ntax,K))
PY
if [ ! -s "$SUB" ]; then echo "FATAL: subsample failed/empty -> abort"; exit 1; fi
echo "[subsample] $(head -1 "$SUB")"

run(){ local lbl=$1; shift
  echo; echo "──── $lbl : $* ────"
  # IQ-TREE's logger writes <prefix>.log; the shell redirect MUST go to a DIFFERENT file ($lbl.console),
  # otherwise IQ-TREE's logger (its own FILE*) clobbers the raw printf/fprintf cross-check+probe output
  # (the G.4.3a double-logging artifact — what made 171596955 show "no XCHECK line" despite the hook firing).
  timeout 2400 "$BIN" --gpu -te "$TREE" -s "$SUB" "$@" -nt 12 -pre "$WB/$lbl" -redo > "$WB/$lbl.console" 2>&1
  echo "  exit=$?"
  grep -aE '\[GPU-XCHECK-MIX\]|\[GPU-XCHECK\]' "$WB/$lbl.console" | sed 's/^/  /' || echo "  (no XCHECK line — check $lbl.console)"
}

echo "════ G.8.0 mixture lnL cross-check — $(hostname) $(date -Iseconds) ════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

run c20    -m LG+C20+G4
run c60    -m LG+C60+G4
run meow80 -mdef "$NEX" -m LG+ESmodel+G4 -mwopt

echo
echo "════ VERDICT (gate rel<=1e-9) ════"
for l in c20 c60 meow80; do
  echo "  $l : $(grep -hoE 'rel=[0-9.e+-]+ +-> +[A-Z()0-9. ]+' "$WB/$l.console" 2>/dev/null | head -1 || echo 'no result')"
done
echo "════ DONE $(date -Iseconds) ════"
