#!/bin/bash
# run_ctf_overfit_recall_sweep.sh — the confirming measurement for part5 §V.14 (the panel's overfitting concern).
# For real AA (LG+I+G4-gen) and DNA (GTR+I+G4-gen) 100K alignments: compute the full-data BIC oracle, then for a
# grid of subsample sizes m x seeds run stock -m MF on the subsample and rank candidates by BOTH the NATIVE
# subsample BIC (current gate: -2*lnL_sub + k*ln m) and the OLD PROJECTED gate (-2*(N/m)*lnL_sub + k*ln N).
# Reports per (dataset,m,seed): oracle winner, each gate's top-1 + recall@3 of the oracle winner, and whether
# each gate's top-1 OVER-selects complexity (more params than oracle). Turns §V.14.2's n=1 demonstration (projected
# demotes the simple winner, native is robust) into n = (2 datasets x sizes x seeds). BIC is optimiser-invariant
# => CPU reference binary, no GPU, no contention. Submit: qsub gadi-ci/gpu-modelfinder/run_ctf_overfit_recall_sweep.sh
#PBS -N ctf-overfit-recall
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=480GB
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
BIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
WB=/scratch/rc29/as1708/iqtree3-gpu/ctf_overfit_recall; mkdir -p "$WB"; cd "$WB"
SUM="$WB/RECALL_SUMMARY.tsv"; NT=104
# FIX (v2): iqtree3-mpi standalone mis-detects "1 CPU core" and aborts on -T 104. Launch via mpirun -np 1
# --bind-to none so the single rank sees all 104 cores (OpenMPI binds-to-core by default → 1-core detection).
RUN="mpirun -np 1 --bind-to none --mca rmaps_base_mapping_policy \"\" numactl --localalloc --"
SIZES="1000 2000 5000"; SEEDS="1 2 3 4 5"
printf "dataset\tN\tm\tseed\toracle_winner\toracle_k\tnative_top1\tnative_top1_k\tnative_recall3\tproj_top1\tproj_top1_k\tproj_recall3\tnative_overfit\tproj_overfit\n" > "$SUM"

echo "════════ CTF overfitting/recall sweep — $(hostname) $(date -Iseconds) ════════"
[ -x "$BIN" ] || { echo "MISSING CPU binary $BIN"; exit 1; }

# python helper: parse an .iqtree model table -> rank by native & projected BIC, emit JSON-ish lines
rank_py () {  # $1=iqtree file  $2=m(subsample sites or N for oracle)  $3=N(full)
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
        name=p[0]; lnL=float(p[1]); AIC=float(p[2]); k=round(AIC/2.0+lnL)
        rows.append((name,lnL,k))
def nbic(lnL,k): return -2*lnL + k*math.log(m)        # native (n=m; for oracle m==N)
def pbic(lnL,k): return -2*(N/m)*lnL + k*math.log(N)  # projected (only meaningful when m<N)
nat=sorted(rows,key=lambda r:nbic(r[1],r[2]))
prj=sorted(rows,key=lambda r:pbic(r[1],r[2]))
out={"n":len(rows),
     "nat_top3":[r[0] for r in nat[:3]], "nat_k":{r[0]:r[2] for r in nat[:3]},
     "prj_top3":[r[0] for r in prj[:3]], "prj_k":{r[0]:r[2] for r in prj[:3]},
     "kmap":{r[0]:r[2] for r in rows}}
print(json.dumps(out))
PY
}

run_dataset () {  # $1=tag  $2=alnpath
  local TAG="$1" ALN="$2"
  echo; echo "════════ $TAG  $ALN ════════"
  [ -f "$ALN" ] || { echo "  MISSING aln — skip"; return; }
  local N=$(awk 'NR==1{print $2; exit}' "$ALN")
  local OD="$WB/$TAG"; mkdir -p "$OD"

  # ---- oracle: full-data -m MF ----
  if [ ! -f "$OD/oracle.iqtree" ]; then
    echo "  oracle: full -m MF on N=$N sites ..."
    eval $RUN "$BIN" -s "$ALN" -m MF -T $NT -pre "$OD/oracle" -redo > "$OD/oracle.log" 2>&1
  fi
  local ORACLE_JSON=$(rank_py "$OD/oracle.iqtree" "$N" "$N")
  local ORACLE=$(echo "$ORACLE_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['nat_top3'][0])")
  local ORACLE_K=$(echo "$ORACLE_JSON" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['kmap'].get(d['nat_top3'][0],'NA'))")
  echo "  ORACLE winner (full-data BIC) = $ORACLE (k=$ORACLE_K)"

  # ---- subsample grid ----
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
    eval $RUN "$BIN" -s "$SP" -m MF -T $NT -pre "$OD/sub_${m}_${sd}" -redo > "$OD/sub_${m}_${sd}.log" 2>&1
    [ -f "$OD/sub_${m}_${sd}.iqtree" ] || { echo "  $TAG m=$m seed=$sd FAILED"; continue; }
    local J=$(rank_py "$OD/sub_${m}_${sd}.iqtree" "$m" "$N")
    read -r NTOP NK NREC PTOP PK PREC NOVF POVF < <(python3 -c "
import sys,json
d=json.loads('''$J'''); orc='$ORACLE'; ork=$ORACLE_K
nt=d['nat_top3'][0]; pt=d['prj_top3'][0]; km=d['kmap']
nk=km.get(nt,'NA'); pk=km.get(pt,'NA')
nrec=1 if orc in d['nat_top3'] else 0
prec=1 if orc in d['prj_top3'] else 0
novf=1 if (nk!='NA' and nk> ork) else 0   # native top1 more complex than oracle?
povf=1 if (pk!='NA' and pk> ork) else 0   # projected top1 more complex than oracle?
print(nt,nk,nrec,pt,pk,prec,novf,povf)")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$TAG" "$N" "$m" "$sd" "$ORACLE" "$ORACLE_K" "$NTOP" "$NK" "$NREC" "$PTOP" "$PK" "$PREC" "$NOVF" "$POVF" >> "$SUM"
    echo "  m=$m seed=$sd : native_top1=$NTOP (rec3=$NREC ovf=$NOVF) | proj_top1=$PTOP (rec3=$PREC ovf=$POVF)"
  done; done
}

run_dataset AA  "$BASE/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
run_dataset DNA "$BASE/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"

echo; echo "════════ RECALL_SUMMARY.tsv ════════"; column -t -s$'\t' "$SUM"
echo; echo "════════ aggregate: native vs projected recall@3 + overfit-top1 rate ════════"
python3 - "$SUM" <<'PY'
import sys,csv,collections
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
for ds in sorted(set(r['dataset'] for r in rows)):
  for m in sorted(set(r['m'] for r in rows if r['dataset']==ds),key=int):
    g=[r for r in rows if r['dataset']==ds and r['m']==m]; n=len(g)
    nr=sum(int(r['native_recall3']) for r in g); pr=sum(int(r['proj_recall3']) for r in g)
    no=sum(int(r['native_overfit']) for r in g); po=sum(int(r['proj_overfit']) for r in g)
    print(f"  {ds:3s} m={m:>5s}: native recall@3={nr}/{n} overfit-top1={no}/{n} | projected recall@3={pr}/{n} overfit-top1={po}/{n}")
PY
echo "════════ DONE $(date -Iseconds) ════════"
