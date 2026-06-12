#!/bin/bash
# run_ctf_phase0_recall_v100.sh — COARSE-TO-FINE (CTF) Phase 0: the FREE subsample-recall DECIDER.
#
# QUESTION (the only positive-arithmetic 100K path is gated on this): does ranking candidates on a small
# frequency-weighted pattern SUBSAMPLE recall the full-data BIC top-3 into its top-k? If yes, CTF (rank-cheap
# -> refine-few) is viable; if no, the only positive-arithmetic 100K path is dead and we pivot to 1M/10M.
#
# Full-data BIC top-3 (from the existing cov.iqtree, NO baseline re-run): LG+G4 (#1), LG+I+G4 (#2, dBIC 14.3),
# LG+F+G4 (#3, dBIC 263.7); then a 17,618-nat cliff to Q.PFAM+F+G4 (#4). So recall of the LG family is
# structurally near-certain; this MEASURES it.
#
# METHOD: random COLUMN subsample (= frequency-weighted pattern subsample, since frequent patterns appear
# proportionally) of the AA-100K alignment at K in {1000,2000,5000} sites, seeded. Stock -m TESTONLY on the
# FIXED full-data tree (-te), CPU path (NO --gpu/--jolt: this is an algorithm-agnostic recall test). This is a
# NEW small experiment on subsampled data, NOT a re-run of the full CPU baseline.
# GATE: top-5 subsample BIC recall of the full-data top-3 = 3/3.
#
#PBS -N ctf-p0-recall
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB="$SRC/ctf_p0"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
echo "════════ CTF Phase 0 recall — $(hostname) $(date -Iseconds) ════════"

echo "──── build subsample alignments (random columns, seed=1) ────"
python3 - "$ALN" <<'PY'
import sys, random
src = sys.argv[1]
# read relaxed PHYLIP: first token line "ntax nsites", then "name seq" rows (one line per taxon)
with open(src) as f:
    hdr = f.readline().split()
    ntax, nsites = int(hdr[0]), int(hdr[1])
    names, seqs = [], []
    for line in f:
        line = line.rstrip("\n")
        if not line.strip(): continue
        # split on first run of whitespace
        parts = line.split(None, 1)
        if len(parts) != 2: continue
        names.append(parts[0]); seqs.append(parts[1].replace(" ", ""))
print("read ntax=%d nsites=%d (got %d seqs, len0=%d)" % (ntax, nsites, len(seqs), len(seqs[0]) if seqs else -1))
L = len(seqs[0])
for K in (1000, 2000, 5000):
    random.seed(1)
    cols = sorted(random.sample(range(L), K))
    with open("sub_%d.phy" % K, "w") as out:
        out.write("%d %d\n" % (len(seqs), K))
        for nm, s in zip(names, seqs):
            sub = "".join(s[c] for c in cols)
            out.write("%s  %s\n" % (nm, sub))
    print("wrote sub_%d.phy" % K)
PY

for K in 1000 2000 5000; do
  echo; echo "════════ -m TESTONLY on sub_${K}.phy (fixed tree, CPU) ════════"
  T0=$(date +%s)
  "$BIN" -m TESTONLY -s "$WB/sub_${K}.phy" -te "$TREE" -nt 12 -pre "$WB/rank_${K}" -redo > "$WB/rank_${K}.stdout" 2>&1
  echo "  exit $? wall $(( $(date +%s)-T0 ))s"
  echo "  --- top-8 by BIC on the ${K}-site subsample ---"
  awk '/sorted by BIC/{f=1; next} f && /^[A-Za-z]/ {print "    "$1"  BIC="$NF} f && /^$/{exit}' "$WB/rank_${K}.iqtree" 2>/dev/null | head -8
  echo "  --- recall check: are full-data top-3 {LG+G4, LG+I+G4, LG+F+G4} in the subsample top-5? ---"
  top5=$(awk '/sorted by BIC/{f=1; next} f && /^[A-Za-z]/ {print $1} f && /^$/{exit}' "$WB/rank_${K}.iqtree" 2>/dev/null | head -5)
  hit=0
  for m in LG+G4 LG+I+G4 LG+F+G4; do
    if echo "$top5" | grep -qx "$m"; then echo "    [HIT] $m in top-5"; hit=$((hit+1)); else echo "    [MISS] $m NOT in top-5"; fi
  done
  echo "  RECALL @K=${K}: $hit/3  -> $([ $hit -eq 3 ] && echo PASS || echo CHECK)"
done
echo; echo "════════ DONE $(date -Iseconds) ════════"
