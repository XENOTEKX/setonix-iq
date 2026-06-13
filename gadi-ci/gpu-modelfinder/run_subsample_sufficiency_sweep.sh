#!/bin/bash
# run_subsample_sufficiency_sweep.sh — PART X empirical test of the subsample-sufficiency hypothesis.
# For each subsample length L and R independent random column subsamples (distinct seeds), run standard
# ModelFinder and record the full candidate BIC table. Then compute, as a function of L:
#   (1) RECALL  = fraction of resamples whose top-3 contains the full-data winner (LG+G4)   [H2, the headline]
#   (2) EXACT   = fraction whose rank-1 == the full winner                                   [H1]
#   (3) MARGIN  = winner-vs-runner-up ΔBIC at each L (theory predicts ~linear growth in L)
#   (4) STABILITY = how often the 3 resamples at a given L agree on the winner
# BIC at the MLE is optimiser-invariant, so this runs on the CPU reference binary (no GPU, no contention with the
# GPU -m MF benchmark). -m TEST gives a dense cheap sweep; -m MF at two lengths probes the +R overfitting twist (§X.3.1).
# Submit: qsub -q normalsr -l ncpus=104 -l mem=500GB -l walltime=05:00:00 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd run_subsample_sufficiency_sweep.sh
#PBS -N subsuff
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 eigen/3.3.7 boost/1.84.0 2>/dev/null || true

BIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi   # pure-CPU MPI fork == standard ModelFinder at np1
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
FULLWIN="LG+G4"            # the full-1M-data BIC winner (the recall target)
NT=104
LENS=(1000 2000 5000 10000 20000 50000 100000)
SEEDS=(1 2 3)
WB=/scratch/rc29/as1708/iqtree3-gpu/subsuff_sweep; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln: $BIN / $ALN"; exit 1; }
echo "════════ PART X subsample-sufficiency sweep — $(hostname) $(date -Iseconds) nt=$NT ════════"
echo "binary: $BIN"; echo "full winner (recall target): $FULLWIN"; echo "lengths: ${LENS[*]}  seeds: ${SEEDS[*]}"

# --- random column subsample (seed-controlled), same scheme as the CTF coarse stage ---
subsample () { # $1=L $2=seed -> writes sub_${L}_${seed}.phy
  python3 - "$ALN" "$1" "$2" <<'PY'
import sys, random
src,K,seed=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
with open(src) as f:
    f.readline(); names=[]; seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]); random.seed(seed); cols=sorted(random.sample(range(L),K))
out=f"sub_{K}_{seed}.phy"
open(out,"w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print(out)
PY
}

run_mf () { # $1=phy $2=mode(TEST|MF) $3=prefix
  mpirun -np 1 --bind-to none "$BIN" -s "$1" -m "$2" -T "$NT" -pre "$3" -redo > "${3}.stdout" 2>&1
}

# ---- the -m TEST dense sweep ----
for L in "${LENS[@]}"; do
  for S in "${SEEDS[@]}"; do
    PHY=$(subsample "$L" "$S")
    PRE="$WB/test_${L}_${S}"
    t0=$(date +%s); run_mf "$WB/$PHY" TEST "$PRE"; dt=$(($(date +%s)-t0))
    win=$(grep -m1 "Best-fit model according to BIC:" "${PRE}.iqtree" 2>/dev/null | sed 's/.*BIC: //')
    echo "  TEST L=$L seed=$S  ${dt}s  winner=${win:-NA}"
  done
done

# ---- the +R overfitting probe: -m MF at two lengths (§X.3.1) ----
for L in 5000 20000; do
  PHY=$(subsample "$L" 1)
  PRE="$WB/mf_${L}_1"
  t0=$(date +%s); run_mf "$WB/$PHY" MF "$PRE"; dt=$(($(date +%s)-t0))
  win=$(grep -m1 "Best-fit model according to BIC:" "${PRE}.iqtree" 2>/dev/null | sed 's/.*BIC: //')
  echo "  MF   L=$L seed=1  ${dt}s  winner=${win:-NA}"
done

echo; echo "════════ ANALYSIS ════════"
python3 - "$WB" "$FULLWIN" <<'PY'
import sys, re, glob, os, math
WB, FULLWIN = sys.argv[1], sys.argv[2]
# .iqtree "List of models sorted by BIC scores" table: Model  LogL  AIC  w-AIC  AICc  w-AICc  BIC  w-BIC
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+\S+\s+(\d+\.\d+)\s+\S+\s+(\d+\.\d+)\s+\S+')
def table(iq):
    rows=[]
    for line in open(iq):
        m=row.match(line)
        if m: rows.append((m.group(1), float(m.group(2)), float(m.group(5))))  # name, logL, BIC
    rows.sort(key=lambda r: r[2])  # ascending BIC
    return rows
def analyse(prefix_glob, mode):
    print(f"\n── {mode} ──")
    print(f"{'L':>7} {'seed':>4} {'winner':18} {'runnerup':18} {'ΔBIC(2-1)':>11} {'rank(FULLWIN)':>13} {'top3_has_win':>12}")
    bylen={}
    for iq in sorted(glob.glob(prefix_glob), key=lambda p:(int(re.search(r'_(\d+)_',os.path.basename(p)).group(1)), p)):
        b=os.path.basename(iq); mm=re.search(r'_(\d+)_(\d+)\.iqtree',b)
        if not mm: continue
        L,seed=int(mm.group(1)),int(mm.group(2)); t=table(iq)
        if not t: continue
        names=[r[0] for r in t]; winner=names[0]; runner=names[1] if len(t)>1 else "-"
        dbic=(t[1][2]-t[0][2]) if len(t)>1 else float('nan')
        rank=(names.index(FULLWIN)+1) if FULLWIN in names else -1
        top3=FULLWIN in names[:3]
        print(f"{L:>7} {seed:>4} {winner:18} {runner:18} {dbic:11.1f} {rank:>13} {str(top3):>12}")
        bylen.setdefault(L,[]).append((winner==FULLWIN, top3, dbic, winner))
    print(f"\n  {'L':>7} {'recall(top3)':>13} {'exact(rank1)':>13} {'winner_stable':>14} {'mean ΔBIC':>11}")
    for L in sorted(bylen):
        v=bylen[L]; n=len(v)
        recall=sum(1 for e in v if e[1])/n; exact=sum(1 for e in v if e[0])/n
        wins=set(e[3] for e in v); stable=(len(wins)==1)
        mdb=sum(e[2] for e in v if not math.isnan(e[2]))/max(1,sum(1 for e in v if not math.isnan(e[2])))
        print(f"  {L:>7} {recall:13.2f} {exact:13.2f} {str(stable):>14} {mdb:11.1f}")
analyse(f"{WB}/test_*_*.iqtree", "-m TEST  (dense sweep)")
analyse(f"{WB}/mf_*_*.iqtree",   "-m MF    (+R overfitting probe)")
print("\nH2 (CTF correctness) is LICENSED at the smallest L where recall(top3)=1.00 across all resamples.")
print("Theory (PART X §X.3) predicts ΔBIC margin grows ~linearly in L for the separated case.")
PY
echo "════════ DONE $(date -Iseconds) ════════"
