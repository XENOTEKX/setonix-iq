#!/bin/bash
# run_ctf_1m_energy.sh — CTF AA-1M with the G.4.3c +I-single-start binary + ENERGY monitoring.
# GPU energy: a continuous nvidia-smi power.draw integrator across the WHOLE run (the definitive GPU-energy
# measure; perf-report cannot wrap a multi-command pipeline). CPU host energy on a GPU node is minor and reported
# by the PBS epilogue. Captures per-phase walls (subsample/coarse/refine), model, lnL, BIC, GPU energy (J + Wh).
# Compares vs the measured CPU -m TEST MF-phase walls (np2 3076.9s / np4 1974.5s).
# Submit:  H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=180GB -v ALABEL=h200e run_ctf_1m_energy.sh
#          A100: qsub -q dgxa100   -l ngpus=1 -l ncpus=16 -l mem=180GB -v ALABEL=a100e run_ctf_1m_energy.sh
#PBS -N ctf1me
#PBS -P dx61
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-gpue}"; NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
NFULL=946439; KSUB=5000; TOPK=3
WB="$SRC/ctf1me_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ CTF+energy AA-1M (+I 4-start) on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "CPU -m TEST MF baselines: np2=3076.9s np4=1974.5s ; full -m TEST np2 total=10945.8 (MF 3076.9 + tree 7868.9), lnL -78605196.4 LG+G4"

# whole-run GPU power sampler (2 s cadence) -> integrate to energy
PWLOG="$WB/power.log"; ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 & PWPID=$!
T_ALL0=$(date +%s)

T0=$(date +%s)
python3 - "$ALN" "$KSUB" <<'PY'
import sys, random
src,K=sys.argv[1],int(sys.argv[2])
with open(src) as f:
    f.readline(); names=[]; seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]); random.seed(1); cols=sorted(random.sample(range(L),K))
open("sub.phy","w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print("wrote sub.phy")
PY
T_SUB=$(($(date +%s)-T0))
T0=$(date +%s)
"$BIN" -m TESTONLY -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
T_C=$(($(date +%s)-T0)); echo "  subsample ${T_SUB}s ; coarse ${T_C}s"
[ -f "$WB/coarse.treefile" ] || { echo "COARSE FAILED"; kill $PWPID 2>/dev/null; exit 1; }
python3 - "$WB/coarse.iqtree" "$KSUB" "$NFULL" "$TOPK" > "$WB/topk.txt" <<'PY'
import sys,re,math
iq,m,N,K=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
rows=[]
for line in open(iq):
    mm=row.match(line)
    if not mm: continue
    name,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5)); p=(bic+2*logl)/math.log(m)
    rows.append((-2*(N/m)*logl+p*math.log(N),name,logl,round(p)))
rows.sort()
for b,n,l,p in rows[:K]: print(f"MODEL:{n}")
PY
mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
echo "  top-${TOPK}: ${TOPMODELS[*]}"

export JOLT_DEBUG=1
T_R_TOTAL=0; i=0; declare -A WALL LNL JN
for M in "${TOPMODELS[@]}"; do
  i=$((i+1)); T0=$(date +%s)
  "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine_${i}" -redo > "$WB/refine_${i}.stdout" 2>&1
  T_R=$(($(date +%s)-T0)); T_R_TOTAL=$((T_R_TOTAL+T_R))
  lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND" "$WB/refine_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  jn=$(grep -c '\[JOLT\] model' "$WB/refine_${i}.stdout" 2>/dev/null)
  WALL[$M]=$T_R; LNL[$M]=${lnl:-NA}; JN[$M]=${jn:-0}
  echo "  refine $i $M: wall=${T_R}s lnL=${lnl:-NA} JOLT_calls=${jn:-0} (+G4 -> ~1 call ; +I -> 4 spanning starts)"
done
T_ALL=$(($(date +%s)-T_ALL0)); kill $PWPID 2>/dev/null; sleep 1

echo; echo "════════ RESULT + ENERGY (${ALABEL}) ════════"
python3 - <<PY
import re,math,os
N=$NFULL; pmap={}
import subprocess
for line in open("$WB/topk.txt"): pass
# recover p from coarse.iqtree for the top models
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
for line in open("$WB/coarse.iqtree"):
    mm=row.match(line)
    if mm:
        nm,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5)); pmap[nm]=round((bic+2*logl)/math.log($KSUB))
models="""${TOPMODELS[*]}""".split(); best=None
print(f"{'model':16}{'full_lnL':>18}{'p':>5}{'full_BIC':>18}")
for i,M in enumerate(models,1):
    lnl=None
    for line in open(f"$WB/refine_{i}.log"):
        if "Log-likelihood of the tree" in line or "BEST SCORE FOUND" in line:
            mm=re.search(r'-?\d+\.\d+',line);
            if mm: lnl=float(mm.group()); break
    p=pmap.get(M)
    if lnl and p: bic=-2*lnl+p*math.log(N); print(f"{M:16}{lnl:18.3f}{p:5d}{bic:18.1f}");
    if lnl and p and (best is None or bic<best[1]): best=(M,bic,lnl)
if best: print(f"\nCTF WINNER: {best[0]} full lnL={best[2]:.3f} full BIC={best[1]:.1f} (oracle LG+G4 lnL -78605196.4)")
# GPU energy from power.log
v=[float(x) for x in open("$PWLOG") if x.strip() and x.strip()[0].isdigit()]; dt=2.0
J=sum(v)*dt
print(f"\nGPU ENERGY: {J:.0f} J = {J/3600:.2f} Wh   (mean {sum(v)/max(len(v),1):.0f} W over {len(v)*dt:.0f}s, n={len(v)} samples)")
PY
echo
echo "  WALL: subsample ${T_SUB}s + coarse ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${T_ALL}s"
echo "  vs CPU MF np2 3076.9s -> $(python3 -c "print(f'{3076.9/$T_ALL:.2f}x' if $T_ALL>0 else 'NA')") | np4 1974.5s -> $(python3 -c "print(f'{1974.5/$T_ALL:.2f}x' if $T_ALL>0 else 'NA')")"
echo "  per-model refine walls + JOLT call counts above (+I now 4 spanning starts vs 10, ~2.5x cheaper; correctness held at pinv=0.5 job 170580368)"
echo "════════ DONE $(date -Iseconds) ════════"
