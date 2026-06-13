#!/bin/bash
# run_ctf_10m_mf_aa_h200.sh — the AA-10M SCALE TEST on ONE H200, FULL PARITY with the AA-1M -m MF CTF runs
# (jobs 170756438 H200 / 170756440 A100). IDENTICAL pipeline, binary, seed, KSUB, TOPK, native-BIC gate, energy
# integrator — only the alignment (1M -> 10M), NFULL (distinct patterns), and the per-refine wall budget change.
# Why H200: the full-10M refine arena (~58GB for LG+G4, scaling the measured 14.9GB r10-native-20@1M by NCAT 4/10 and
# ~9.8x patterns) fits H200's 141GB with headroom; A100-80/V100 are too small. The CTF coarse is on a 5000-site
# subsample (scale-invariant ~467s); only the top-3 refine touches full 10M.
# Submit: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=180GB -l walltime=04:00:00 -v ALABEL=aa10m_h200 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd run_ctf_10m_mf_aa_h200.sh
#PBS -N ctf10mmf
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-aa10m_h200}"; NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3.frozen_ab"  # frozen parity binary, md5 b85d482f (== AA-1M -m MF runs)
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_10000000/tree_1/alignment_10000000.phy
NFULL=9251287; KSUB=5000; TOPK=3   # NFULL = 10M distinct site patterns (CPU log: "10000000 columns, 9251287 distinct patterns")
WB="$SRC/ctf10mmf_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ CTF -m MF (ALL MODELS, incl +R) AA-10M on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "ALN=$ALN (10000000 cols, ${NFULL} distinct patterns); BIN md5=$(md5sum "$BIN"|cut -d' ' -f1)"
echo "CPU 10M oracle: expected LG+G4 (CPU -m MTEST baseline by sa0557 in progress; AA data is LG+I+G4-generated, +I absorbed by +G as at 1M)"

# whole-run GPU power sampler -> energy  (IDENTICAL 2s integrator as the 1M runs)
PWLOG="$WB/power.log"; ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 & PWPID=$!
T_ALL0=$(date +%s)

# ---- subsample (5000 cols, seed 1 — IDENTICAL to 1M) ----
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

# ---- coarse: FULL -m MF candidate set on the subsample, JOLT on GPU for eligible ----
export JOLT_DEBUG=1
T0=$(date +%s)
"$BIN" --jolt --gpu -m MF -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
T_C=$(($(date +%s)-T0)); echo "  subsample ${T_SUB}s ; coarse -m MF ${T_C}s"
[ -f "$WB/coarse.treefile" ] || { echo "COARSE FAILED"; kill $PWPID 2>/dev/null; exit 1; }
echo "  COVERAGE on the full -m MF set:"
echo "    candidates in table: $(grep -cE '^\S+\s+-?[0-9]+\.[0-9]+\s+[0-9]+' "$WB/coarse.iqtree" 2>/dev/null)"
echo "    JOLT engagements (GPU): $(grep -c '\[JOLT\] model=' "$WB/coarse.stdout" 2>/dev/null)"
echo "    declines (CPU):"; grep -hoE '\[JOLT-GATE\] decline reason=[^ ]+' "$WB/coarse.stdout" 2>/dev/null | sort | uniq -c | sed 's/^/      /'

# ---- NATIVE subsample-BIC rerank (over ALL candidates) + rate-het detector -> top-k  (IDENTICAL gate as 1M) ----
python3 - "$WB/coarse.iqtree" "$KSUB" "$NFULL" "$TOPK" > "$WB/topk.txt" <<'PY'
import sys,re,math
iq,m,N,K=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
def ineligible(name):   # mirrors the JOLT eligibility gate: FreeRate (+R/+I+R) and pure-+I decline
    return ('+R' in name) or ('+I' in name and '+G' not in name)
rows=[]
for line in open(iq):
    mm=row.match(line)
    if not mm: continue
    name,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5))
    p=(bic+2*logl)/math.log(m); proj=-2*(N/m)*logl+p*math.log(N)
    rows.append((name,logl,bic,round(p),proj,ineligible(name)))
nat_all=sorted(rows, key=lambda r:r[2])                       # NATIVE BIC over ALL candidates = the gate
be=min((r for r in rows if not r[5]), key=lambda r:r[2], default=None)   # best eligible
bi=min((r for r in rows if r[5]),     key=lambda r:r[2], default=None)   # best ineligible (+R/+I)
sys.stderr.write("  [rerank] OLD projected top-5 (the bug): "+", ".join(r[0] for r in sorted(rows,key=lambda r:r[4])[:5])+"\n")
sys.stderr.write("  [rerank] NATIVE BIC top-5 (the gate):   "+", ".join(r[0] for r in nat_all[:5])+"\n")
if be and bi:
    margin=abs(bi[3]-be[3])/2.0     # ~Δp/2 nats AIC overfit cushion
    lead=be[2]-bi[2]                # >0 => an ineligible (+R/+I) model LEADS the eligible best on native BIC
    flag = lead > margin
    sys.stderr.write(f"  [detector] best_elig={be[0]}({be[2]:.1f}) best_inel={bi[0]}({bi[2]:.1f}) inel_lead={lead:.1f} margin={margin:.1f} RATE_HET_FLAG={flag}\n")
    if flag: sys.stderr.write("  [detector] *** WARNING: a +R/+I model genuinely leads on the subsample — eligible-refine may MISS the true winner; needs G.5.1 (+R JOLT) or CPU full-refine ***\n")
for r in nat_all[:K]:
    skip = r[5] and (be is not None) and (r[2] > be[2] + abs(r[3]-be[3])/2.0)
    print(f"MODEL:{r[0]}:{'skip' if skip else 'refine'}")
PY
mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
echo "  top-${TOPK} (NATIVE subsample BIC over all candidates — FIXED gate): ${TOPMODELS[*]}"

# ---- refine top-k on full 10M, with a per-model WALL BUDGET ----
# Eligible GPU refines at 10M legitimately run ~9.8x the 1M wall (LG+G4 ~510s, LG+I+G4 ~2400s on H200), so the budget
# rises to 5400s vs the 1M script's 900s — this still caps an INELIGIBLE (+R/pure-+I) CPU-fallback that would run for
# hours at 10M, but no longer truncates a legitimate eligible GPU refine. Over-budget => carried UNREFINED.
REFINE_BUDGET=5400   # s; eligible GPU +I+G4 refine at 10M ~2400s << 5400 << ineligible-CPU-at-10M blow-up
T_R_TOTAL=0; i=0
for ENTRY in "${TOPMODELS[@]}"; do
  i=$((i+1)); M="${ENTRY%%:*}"; ACT="${ENTRY##*:}"
  if [ "$ACT" = "skip" ]; then echo "  refine $i $M: SKIPPED (ineligible +R/+I; detector: native BIC behind best eligible -> cannot win full-data)"; continue; fi
  T0=$(date +%s)
  timeout ${REFINE_BUDGET}s "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine_${i}" -redo > "$WB/refine_${i}.stdout" 2>&1
  rc=$?; T_R=$(($(date +%s)-T0)); T_R_TOTAL=$((T_R_TOTAL+T_R))
  if [ $rc -eq 124 ]; then echo "  refine $i $M: OVER BUDGET (${REFINE_BUDGET}s) -> carried UNREFINED"; continue; fi
  lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND" "$WB/refine_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  jn=$(grep -c '\[JOLT\] model' "$WB/refine_${i}.stdout" 2>/dev/null)
  pk=$(grep -hoE 'GPU Memory Used: [0-9.]+GB|peak [0-9.]+ ?GB' "$WB/refine_${i}.stdout" 2>/dev/null | head -1)
  echo "  refine $i $M: wall=${T_R}s lnL=${lnl:-NA} JOLT_calls=${jn:-0} ${pk}"
done
T_ALL=$(($(date +%s)-T_ALL0)); kill $PWPID 2>/dev/null; sleep 1

echo; echo "════════ RESULT + ENERGY (${ALABEL}) ════════"
python3 - <<PY
import re,math
N=$NFULL; pmap={}
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
for line in open("$WB/coarse.iqtree"):
    mm=row.match(line)
    if mm:
        nm,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5)); pmap[nm]=round((bic+2*logl)/math.log($KSUB))
import os
models=[e.split(':')[0] for e in """${TOPMODELS[*]}""".split()]; best=None
print(f"{'model':16}{'full_lnL':>18}{'p':>5}{'full_BIC':>18}")
for i,M in enumerate(models,1):
    lnl=None; lg=f"$WB/refine_{i}.log"
    if os.path.exists(lg):
        for line in open(lg):
            if "Log-likelihood of the tree" in line or "BEST SCORE FOUND" in line:
                mm=re.search(r'-?\d+\.\d+',line)
                if mm: lnl=float(mm.group()); break
    p=pmap.get(M)
    if lnl and p: bic=-2*lnl+p*math.log(N); print(f"{M:16}{lnl:18.3f}{p:5d}{bic:18.1f}")
    if lnl and p and (best is None or bic<best[1]): best=(M,bic,lnl)
if best: print(f"\\nCTF -m MF WINNER (AA-10M): {best[0]} full lnL={best[2]:.3f} full BIC={best[1]:.1f} (expected oracle LG+G4)")
v=[float(x) for x in open("$PWLOG") if x.strip() and x.strip()[0].isdigit()]; dt=2.0; J=sum(v)*dt
print(f"\\nGPU ENERGY: {J:.0f} J = {J/3600:.2f} Wh (mean {sum(v)/max(len(v),1):.0f} W over {len(v)*dt:.0f}s)")
PY
echo
echo "  WALL: subsample ${T_SUB}s + coarse(-m MF) ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${T_ALL}s"
echo "  NOTE: CPU 10M -m MF baseline (sa0557 MTEST) in progress — speedup ratio computed when that completes; do NOT compare against 1M CPU walls (DNA-baseline lesson)."
echo "════════ DONE $(date -Iseconds) ════════"
