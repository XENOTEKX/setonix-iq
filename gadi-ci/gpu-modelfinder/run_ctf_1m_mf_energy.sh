#!/bin/bash
# run_ctf_1m_mf_energy.sh — the AA-1M ULTIMATE TEST: the FULL `-m MF` candidate set (includes +R FreeRate, unlike
# -m TEST) on one GPU, via CTF. Ranks ALL -m MF candidates on a 5000-site subsample with `--jolt --gpu -m MF`
# (eligible -> GPU/JOLT, +R/+I -> CPU), then JOLT-refines the top-3 on full 1M. This evaluates every model AND
# finishes (a direct full-data serial -m MF mutex-serializes ~116 models = the breadth case the CPU owns). Reports
# coverage (engage/decline), per-phase walls, winner vs the CPU oracle (LG+G4), and GPU energy (nvidia-smi integrator).
# Submit: H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=180GB -l walltime=02:00:00 -v ALABEL=h200mf run_ctf_1m_mf_energy.sh
#         A100: qsub -q dgxa100   -l ngpus=1 -l ncpus=16 -l mem=180GB -l walltime=02:00:00 -v ALABEL=a100mf run_ctf_1m_mf_energy.sh
#PBS -N ctf1mmf
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-gpumf}"; NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3.frozen_ab"  # frozen: G.5.0 PartB+fusion+base-skip+d_theta-reclaim, md5 b85d482f, jobs 170726673+170730082 PASS
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
NFULL=946439; KSUB=5000; TOPK=3
WB="$SRC/ctf1mmf_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ CTF -m MF (ALL MODELS, incl +R) AA-1M on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "CPU -m TEST MF baselines: np2=3076.9 np4=1974.5 np8=1443.9 np16=1122.4 s ; oracle best LG+G4"

# whole-run GPU power sampler -> energy
PWLOG="$WB/power.log"; ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 & PWPID=$!
T_ALL0=$(date +%s)

# ---- subsample ----
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

# ---- coarse: FULL -m MF candidate set on the subsample (the "all the models" pass), JOLT on GPU for eligible ----
export JOLT_DEBUG=1
T0=$(date +%s)
"$BIN" --jolt --gpu -m MF -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
T_C=$(($(date +%s)-T0)); echo "  subsample ${T_SUB}s ; coarse -m MF ${T_C}s"
[ -f "$WB/coarse.treefile" ] || { echo "COARSE FAILED"; kill $PWPID 2>/dev/null; exit 1; }
echo "  COVERAGE on the full -m MF set:"
echo "    candidates in table: $(grep -cE '^\S+\s+-?[0-9]+\.[0-9]+\s+[0-9]+' "$WB/coarse.iqtree" 2>/dev/null)"
echo "    JOLT engagements (GPU): $(grep -c '\[JOLT\] model=' "$WB/coarse.stdout" 2>/dev/null)"
echo "    declines (CPU):"; grep -hoE '\[JOLT-GATE\] decline reason=[^ ]+' "$WB/coarse.stdout" 2>/dev/null | sort | uniq -c | sed 's/^/      /'

# ---- NATIVE subsample-BIC rerank (over ALL candidates) + rate-het detector -> top-k  (FIX: jobs 170728179/182) ----
# The OLD scale-consistent PROJECTION (-2*(N/m)*logL + p*ln N) amplifies any subsample logL diff by 2*N/m (~378x)
# while leaving the p*ln(N) penalty fixed => it over-credits the +I/+R overfit and ranked [LG+I+G4,LG+R5,LG+I+R5]
# ABOVE the true winner LG+G4 (recall FAIL -> +R refined on CPU at 1M -> walltime). On 5000 sites every candidate is
# within <1 nat of FIT, so the rate-model choice is decided ENTIRELY by the penalty => NATIVE subsample BIC (penalty
# ln m) is the right gate and ranks LG+G4 #1 (verified at all 23 sweep runs, PART X). We rank ALL candidates (no
# pre-exclusion — red-team: don't hide the +R coverage gap), add a RATE-HET DETECTOR, and cap each refine with a
# wall BUDGET so an ineligible +R CPU-refine at 1M cannot blow the total (PART IX X.7 / PART X X.5.5).
python3 - "$WB/coarse.iqtree" "$KSUB" "$NFULL" "$TOPK" > "$WB/topk.txt" <<'PY'
import sys,re,math
iq,m,N,K=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
def ineligible(name):   # mirrors the JOLT eligibility gate (PART IX X.1): FreeRate (+R/+I+R) and pure-+I decline
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
    # refine eligible models always; refine an INELIGIBLE (+R/+I) model only if it could plausibly win (within the
    # overfit margin of the best eligible) — else SKIP it (detector-justified, honest: we checked it cannot win),
    # which keeps the wall competitive instead of burning the full CPU budget on a doomed +R refine.
    skip = r[5] and (be is not None) and (r[2] > be[2] + abs(r[3]-be[3])/2.0)
    print(f"MODEL:{r[0]}:{'skip' if skip else 'refine'}")
PY
mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
echo "  top-${TOPK} (NATIVE subsample BIC over all candidates — FIXED gate): ${TOPMODELS[*]}"

# ---- refine top-k on full 1M, with a per-model WALL BUDGET ----
# A JOLT-eligible refine is ~78-540s on GPU; an ineligible (+R/pure-+I) refine falls to the CPU EM optimiser at 1M
# and can run for hours (it timed out 170728179/182). `timeout` caps each model: over-budget => carried UNREFINED
# (excluded from the winner pick — its incomplete .log has no "Log-likelihood of the tree" line). The detector above
# tells us whether any unrefined ineligible model could actually be the winner (lead>margin); here it cannot.
REFINE_BUDGET=900   # s; > the ~540s +I 4-start GPU refine, << the ineligible-CPU-at-1M blow-up
T_R_TOTAL=0; i=0
for ENTRY in "${TOPMODELS[@]}"; do
  i=$((i+1)); M="${ENTRY%%:*}"; ACT="${ENTRY##*:}"
  if [ "$ACT" = "skip" ]; then echo "  refine $i $M: SKIPPED (ineligible +R/+I; detector: native BIC behind best eligible -> cannot win full-data)"; continue; fi
  T0=$(date +%s)
  timeout ${REFINE_BUDGET}s "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine_${i}" -redo > "$WB/refine_${i}.stdout" 2>&1
  rc=$?; T_R=$(($(date +%s)-T0)); T_R_TOTAL=$((T_R_TOTAL+T_R))
  if [ $rc -eq 124 ]; then echo "  refine $i $M: OVER BUDGET (${REFINE_BUDGET}s, CPU-fallback at 1M) -> carried UNREFINED"; continue; fi
  lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND" "$WB/refine_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  jn=$(grep -c '\[JOLT\] model' "$WB/refine_${i}.stdout" 2>/dev/null)
  echo "  refine $i $M: wall=${T_R}s lnL=${lnl:-NA} JOLT_calls=${jn:-0}"
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
models=[e.split(':')[0] for e in """${TOPMODELS[*]}""".split()]; best=None   # strip the :refine/:skip action tag
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
if best: print(f"\\nCTF -m MF WINNER: {best[0]} full lnL={best[2]:.3f} full BIC={best[1]:.1f} (oracle LG+G4 -78605196.4)")
v=[float(x) for x in open("$PWLOG") if x.strip() and x.strip()[0].isdigit()]; dt=2.0; J=sum(v)*dt
print(f"\\nGPU ENERGY: {J:.0f} J = {J/3600:.2f} Wh (mean {sum(v)/max(len(v),1):.0f} W over {len(v)*dt:.0f}s)")
PY
echo
echo "  WALL: subsample ${T_SUB}s + coarse(-m MF) ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${T_ALL}s"
echo "  vs CPU MF: np4 1974.5 -> $(python3 -c "print(f'{1974.5/$T_ALL:.2f}x')") | np8 1443.9 -> $(python3 -c "print(f'{1443.9/$T_ALL:.2f}x')") | np16 1122.4 -> $(python3 -c "print(f'{1122.4/$T_ALL:.2f}x')")"
echo "════════ DONE $(date -Iseconds) ════════"
