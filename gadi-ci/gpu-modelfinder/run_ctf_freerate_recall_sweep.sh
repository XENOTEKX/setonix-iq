#!/bin/bash
# run_ctf_freerate_recall_sweep.sh — THE decisive +R experiment for part5 §V.14.7/§V.14.8 (the LAST open recall gap).
#
# The panel's overfitting concern, and the §V.14.2b n=30 sweep, were both exercised ONLY on data whose full-data
# BIC winner is a +G model (AA→LG+G4, DNA→F81+F+G4). The doc itself flags the one untested regime:
#     "genuinely +R-favouring data (a true +Rk generative model where +R beats +G at full data) — the under-fit
#      direction; that remains the last open recall check."  (§V.14.7/§V.14.8)
#
# This sweep CLOSES it. We SIMULATE (AliSim) alignments under strongly *bimodal* FreeRate distributions that
# unimodal gamma structurally cannot fit, so the full-data BIC oracle is a genuine +R-family model (verified:
# at 10K AA, LG+R4 beats LG+G4 by 2847 nats / ΔBIC≈5648; the gain/penalty ratio only grows with N). Then for
# m∈{1000,2000,5000}×seeds{1..5} we rank the candidate set on the subsample under the shipped NATIVE gate
# (-2*lnL_sub + k*ln m) AND the old PROJECTED gate, and we measure, per cell:
#   - recall@3 of the +R oracle winner  (the screen-then-clean precondition, §V.14.1 Case A)
#   - the ACTUAL CTF OUTPUT = argmin_{M in coarse top-3} BIC_full(M)  (the exact full-data BIC refine — bit-for-bit
#       what stock ModelFinder computes; computed here from the oracle's own full-N BIC, non-circular) → CTF==oracle?
#   - UNDER-fit flag (CTF output has FEWER params than oracle — the predicted +R risk direction)
#   - OVER-fit flag  (CTF output has MORE params than oracle — the panel's stated fear)
#   - rate-het CLASS downgrade (oracle is +R-family but CTF output drops to +G/+I — a categorical underfit)
#
# BIC is optimiser-invariant ⇒ CPU reference binary (iqtree3-mpi), no GPU, no contention — the apples-to-apples
# gold standard for SELECTION (the panel's concern is the statistical procedure, not the optimizer; +R models
# decline to CPU in the GPU pipeline anyway). RESUME-safe per (regime,m,seed) via .iqtree presence.
# Submit: qsub gadi-ci/gpu-modelfinder/run_ctf_freerate_recall_sweep.sh
#PBS -N ctf-freerate-recall
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=480GB
#PBS -l walltime=08:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
CPUBIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi   # CPU reference (selection gold standard)
SIMBIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3           # AliSim simulator (any recent build)
AATREE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/tree_1.full.treefile
DNATREE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/tree_1.full.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/ctf_freerate_recall; mkdir -p "$WB"; cd "$WB"
SUM="$WB/FREERATE_RECALL_SUMMARY.tsv"; NT=104; NSIM=100000
RUN="mpirun -np 1 --bind-to none --mca rmaps_base_mapping_policy \"\" numactl --localalloc --"
SIZES="1000 2000 5000"; SEEDS="1 2 3 4 5"

# The three +R-favouring generative regimes (bimodal rates; mean rate normalised to 1):
#   AA  : LG  + bimodal R4  (two well-separated rate clusters; gamma cannot reproduce two modes)
#   DNA : GTR + bimodal R4
#   AAI : LG  + I{0.2} + bimodal R3  (genuine invariant fraction + bimodal variable rates → +I+R-family winner)
declare -A GEN SEQ TREE
GEN[AA]='LG+R4{0.45,0.1,0.05,0.4,0.05,1.6,0.45,1.9}';            SEQ[AA]=AA;  TREE[AA]=$AATREE
GEN[DNA]='GTR{2.0,5.0,1.5,1.2,4.5}+F{0.3,0.2,0.2,0.3}+R4{0.45,0.1,0.05,0.4,0.05,1.6,0.45,1.9}'; SEQ[DNA]=DNA; TREE[DNA]=$DNATREE
GEN[AAI]='LG+I{0.2}+R3{0.5,0.2,0.1,1.0,0.4,2.0}';                SEQ[AAI]=AA; TREE[AAI]=$AATREE

printf "regime\tgen_model\tN\tm\tseed\toracle\toracle_k\toracle_class\tnat_top1\tnat_top1_k\tnat_recall3\tctf_out\tctf_out_k\tctf_correct\tctf_underfit\tctf_overfit\tctf_class_down\tproj_top1\tproj_recall3\tproj_ctf_out\tproj_ctf_correct\n" > "$SUM"

echo "════════ CTF +R (FreeRate) adversarial recall sweep — $(hostname) $(date -Iseconds) ════════"
[ -x "$CPUBIN" ] || { echo "MISSING CPU binary $CPUBIN"; exit 1; }
[ -x "$SIMBIN" ] || { echo "MISSING sim binary $SIMBIN"; exit 1; }

# ---- python: parse an .iqtree model table -> {name: (lnL,k)} and ranked top-3 under native & projected BIC ----
rank_py () {  # $1=iqtree  $2=m  $3=N
python3 - "$1" "$2" "$3" <<'PY'
import sys,re,math,json
iqf,m,N=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
rows=[]; f=False
for line in open(iqf):
    if re.match(r'^Model\s+LogL',line): f=True; continue
    if f and line.startswith('AIC, w-AIC'): break
    if not f: continue
    p=line.split()
    if len(p)>=3 and re.match(r'^-?\d',p[1]) and p[1].startswith('-'):
        rows.append((p[0],float(p[1]),round(float(p[2])/2.0+float(p[1]))))   # name, lnL, k(from AIC)
nbic=lambda l,k:-2*l+k*math.log(m)
pbic=lambda l,k:-2*(N/m)*l+k*math.log(N)
nat=sorted(rows,key=lambda r:nbic(r[1],r[2])); prj=sorted(rows,key=lambda r:pbic(r[1],r[2]))
print(json.dumps({"nat_top3":[r[0] for r in nat[:3]],"prj_top3":[r[0] for r in prj[:3]],
                  "lnLmap":{r[0]:r[1] for r in rows},"kmap":{r[0]:r[2] for r in rows}}))
PY
}

# rate-het class of a model name: I+R / R / I+G / G / I / none (for categorical downgrade detection)
hclass () { local n="$1"
  if   [[ $n == *+I*+R* || $n == *+R*+I* ]]; then echo "I+R"
  elif [[ $n == *+R* ]]; then echo "R"
  elif [[ $n == *+I*+G* || $n == *+G*+I* ]]; then echo "I+G"
  elif [[ $n == *+G* ]]; then echo "G"
  elif [[ $n == *+I* ]]; then echo "I"
  else echo "none"; fi; }

run_regime () {  # $1=regime key
  local R="$1" G="${GEN[$1]}" ST="${SEQ[$1]}" TR="${TREE[$1]}"
  local OD="$WB/$R"; mkdir -p "$OD"
  echo; echo "════════ regime=$R  seqtype=$ST  gen='$G' ════════"
  [ -f "$TR" ] || { echo "  MISSING tree $TR — skip"; return; }
  # AliSim writes a screen-log next to the tree; the source trees live in a read-only dir → copy locally first.
  local LTR="$OD/tree.nwk"; cp -f "$TR" "$LTR"

  # ---- simulate the full-length alignment under the +R-favouring generative model (fixed sim seed 1) ----
  local ALN="$OD/sim_${R}.phy"
  if [ ! -f "$ALN" ]; then
    echo "  simulating N=$NSIM sites under '$G' ..."
    # Run the simulator DIRECTLY (no eval/mpirun): the model string contains {…} and `eval` would brace-expand it.
    # SIMBIN is the standalone binary (not the MPI build), so it needs neither the mpirun wrapper nor -T.
    "$SIMBIN" --alisim "$OD/sim_${R}" -t "$LTR" -m "$G" --length $NSIM --seed 1 -redo > "$OD/sim.log" 2>&1
    [ -f "$OD/sim_${R}.phy" ] || { echo "  SIM FAILED"; cat "$OD/sim.log" | tail -5; return; }
  fi
  local N=$(awk 'NR==1{print $2; exit}' "$ALN")

  # ---- oracle: full-data -m MF (this IS the +R-favouring precondition check) ----
  if [ ! -f "$OD/oracle.iqtree" ]; then
    echo "  oracle: full -m MF on N=$N sites ..."
    eval $RUN "$CPUBIN" -s "$ALN" -m MF -T $NT -pre "$OD/oracle" -redo > "$OD/oracle.log" 2>&1
  fi
  [ -f "$OD/oracle.iqtree" ] || { echo "  ORACLE FAILED"; return; }
  local OJ=$(rank_py "$OD/oracle.iqtree" "$N" "$N")
  read -r ORACLE ORACLE_K < <(echo "$OJ" | python3 -c "import sys,json;d=json.load(sys.stdin);w=d['nat_top3'][0];print(w,d['kmap'][w])")
  local OCLASS=$(hclass "$ORACLE")
  echo "  ORACLE winner (full-data BIC) = $ORACLE (k=$ORACLE_K, class=$OCLASS)"
  if [[ "$OCLASS" != "R" && "$OCLASS" != "I+R" ]]; then
    echo "  ⚠ PRECONDITION NOT MET: oracle is class=$OCLASS, not +R-family — this regime is vacuous for the +R question (recorded anyway)."
  fi

  # ---- subsample grid: coarse rank + EXACT-BIC CTF output ----
  for m in $SIZES; do for sd in $SEEDS; do
    local SP="$OD/sub_${m}_${sd}.phy"
    python3 - "$ALN" "$m" "$SP" "$sd" <<'PY'
import sys,random
src,K,out,seed=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
with open(src) as f:
    f.readline(); names=[];seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]); K=min(K,L); random.seed(seed); cols=sorted(random.sample(range(L),K))
open(out,"w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
PY
    [ -f "$OD/sub_${m}_${sd}.iqtree" ] || eval $RUN "$CPUBIN" -s "$SP" -m MF -T $NT -pre "$OD/sub_${m}_${sd}" -redo > "$OD/sub_${m}_${sd}.log" 2>&1
    [ -f "$OD/sub_${m}_${sd}.iqtree" ] || { echo "  $R m=$m seed=$sd FAILED"; continue; }
    local SJ=$(rank_py "$OD/sub_${m}_${sd}.iqtree" "$m" "$N")
    # CTF output = argmin over coarse top-k of the ORACLE's full-data BIC (exact refine, non-circular).
    read -r NTOP NK NREC CTF CTFK CTFOK CUF COF PTOP PREC PCTF PCTFOK < <(python3 -c "
import sys,json,math
sub=json.loads('''$SJ'''); orc=json.loads('''$OJ'''); N=float($N)
oracle='$ORACLE'; ok=$ORACLE_K
ob={m:-2*orc['lnLmap'][m]+orc['kmap'][m]*math.log(N) for m in orc['lnLmap']}   # exact full-data (native, n=N) BIC
def ctf(top3):   # refine = argmin full-data BIC over shortlist present in oracle table
    cand=[m for m in top3 if m in ob]; return min(cand,key=lambda m:ob[m]) if cand else top3[0]
nt=sub['nat_top3']; pt=sub['prj_top3']; km=orc['kmap']
ntop=nt[0]; nk=km.get(ntop,'NA'); nrec=1 if oracle in nt else 0
cmod=ctf(nt); ck=km.get(cmod,'NA'); cok=1 if cmod==oracle else 0
cuf=1 if (ck!='NA' and ck<ok) else 0; cof=1 if (ck!='NA' and ck>ok) else 0
ptop=pt[0]; prec=1 if oracle in pt else 0; pcmod=ctf(pt); pcok=1 if pcmod==oracle else 0
print(ntop,nk,nrec,cmod,ck,cok,cuf,cof,ptop,prec,pcmod,pcok)")
    local CCLASS=$(hclass "$CTF"); local CDOWN=0
    [[ ("$OCLASS" == "R" || "$OCLASS" == "I+R") && "$CCLASS" != "R" && "$CCLASS" != "I+R" ]] && CDOWN=1
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$R" "$G" "$N" "$m" "$sd" "$ORACLE" "$ORACLE_K" "$OCLASS" "$NTOP" "$NK" "$NREC" \
      "$CTF" "$CTFK" "$CTFOK" "$CUF" "$COF" "$CDOWN" "$PTOP" "$PREC" "$PCTF" "$PCTFOK" >> "$SUM"
    echo "  m=$m sd=$sd: coarse_top1=$NTOP rec3=$NREC | CTF_out=$CTF (correct=$CTFOK under=$CUF over=$COF classdown=$CDOWN) | proj_ctf=$PCTF(correct=$PCTFOK)"
  done; done
}

for R in AA DNA AAI; do run_regime "$R"; done

echo; echo "════════ FREERATE_RECALL_SUMMARY.tsv ════════"; column -t -s$'\t' "$SUM"
echo; echo "════════ aggregate: native gate (shipped) vs projected gate, on +R-favouring data ════════"
python3 - "$SUM" <<'PY'
import sys,csv
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
print(f"  {'regime':6s} {'oracle':14s} {'cells':>5s} {'nat_rec@3':>9s} {'CTF_correct':>11s} {'CTF_under':>9s} {'CTF_over':>8s} {'classdown':>9s} | {'proj_rec@3':>10s} {'proj_CTF_ok':>11s}")
for R in sorted(set(r['regime'] for r in rows)):
  g=[r for r in rows if r['regime']==R]; n=len(g); orc=g[0]['oracle']
  nr=sum(int(r['nat_recall3']) for r in g); cc=sum(int(r['ctf_correct']) for r in g)
  uf=sum(int(r['ctf_underfit']) for r in g); of=sum(int(r['ctf_overfit']) for r in g)
  cd=sum(int(r['ctf_class_down']) for r in g)
  pr=sum(int(r['proj_recall3']) for r in g); pc=sum(int(r['proj_ctf_correct']) for r in g)
  print(f"  {R:6s} {orc:14s} {n:5d} {nr:9d} {cc:11d} {uf:9d} {of:8d} {cd:9d} | {pr:10d} {pc:11d}")
print("  (CTF_correct = CTF output == full-data BIC oracle winner; the proof that screen-then-clean recovers the +R winner.)")
PY
echo "════════ DONE $(date -Iseconds) ════════"
