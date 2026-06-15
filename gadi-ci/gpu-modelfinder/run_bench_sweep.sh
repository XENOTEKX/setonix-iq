#!/bin/bash
# run_bench_sweep.sh — G.7.2 BENCHMARK SWEEP (website data). Generalises the validated CTF -m MF runner
# (run_ctf_1m_mf_energy.sh: energy poller + NATIVE-BIC gate + rate-het detector + wall-budgeted refine) across
# SCALES x TYPES on the TILING binary (build-gpu-on/iqtree3 — bit-identical to frozen_ab at nTile=1 per G.7.1 V.A,
# so it PARITY-MATCHES the existing 1M table; auto-nTile lets 10M fit). Per (TYPE,SCALE) it records the clean
# wall (per phase), GPU energy (nvidia-smi power integrator), peak VRAM, max GPU util, the [JOLT] GPU≡CPU parity
# rel (worst), coverage (engage/decline), and the winning model + full lnL/BIC. NO profiler attached (energy/wall
# must be clean; nsys/ncu are a SEPARATE pass — run_bench_profile.sh). Emits $WB/SUMMARY.tsv for the JSON builder.
#
# Driven by env: TYPES ("AA DNA"), SCALES ("10000 100000 1000000"), DEVLABEL (a100|h200), REFINE_BUDGET (s).
# Submit A100 small: qsub -q dgxa100  -lngpus=1 -lncpus=16 -lmem=180GB -lwalltime=03:00:00 -v DEVLABEL=a100,TYPES="AA DNA",SCALES="10000 100000 1000000" run_bench_sweep.sh
# Submit H200 small: qsub -q gpuhopper -lngpus=1 -lncpus=12 -lmem=180GB -lwalltime=03:00:00 -v DEVLABEL=h200,TYPES="AA DNA",SCALES="10000 100000 1000000" run_bench_sweep.sh
# Submit *  10M:     ... -lwalltime=06:00:00 -v DEVLABEL=<d>,TYPES="AA DNA",SCALES="10000000",REFINE_BUDGET=2400 ...
#PBS -N benchsweep
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cuda/12.5.1  2>/dev/null || true
module load eigen/3.3.7  2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LIBRARY_PATH:-}"

TYPES="${TYPES:-AA DNA}"; SCALES="${SCALES:-10000 100000 1000000}"; DEVLABEL="${DEVLABEL:-gpu}"
NT="${PBS_NCPUS:-12}"; KSUB=5000; TOPK=3; REFINE_BUDGET="${REFINE_BUDGET:-900}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
OUT="$SRC/bench_${DEVLABEL}_${PBS_JOBID:-local}"; mkdir -p "$OUT"; cd "$OUT"
SUMMARY="$OUT/SUMMARY.tsv"
printf "type\tscale\tdevice\thost\tnTile\twall_total_s\twall_sub_s\twall_coarse_s\twall_refine_s\tenergy_wh\tmean_w\tpeak_vram_mib\tmax_util\twinner\tfull_lnL\tfull_bic\tworst_parity_rel\tengage\tdecline\tbin_md5\n" > "$SUMMARY"

echo "════════ G.7.2 BENCH SWEEP ${DEVLABEL} — $(hostname) $(date -Iseconds) — TYPES='$TYPES' SCALES='$SCALES' ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "── rebuild on-node ──"; ( cd "$BUILD_ON" && make -j"$NT" iqtree3 > "$OUT/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; tail -3 "$BUILD_ON/make.log" | sed 's/^/    /'; [ $RC -ne 0 ] && { echo BUILD FAILED; exit 1; }
BINMD5=$(md5sum "$BIN" | cut -d' ' -f1); echo "  BIN md5=$BINMD5"
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

run_one () {
  local TYPE="$1" SCALE="$2"
  local SUB; [ "$TYPE" = AA ] && SUB=LG+I+G4 || SUB=GTR+I+G4
  local ALN="$BASE/$TYPE/$SUB/taxa_100/len_$SCALE/tree_1/alignment_$SCALE.phy"
  local WB="$OUT/${TYPE}_${SCALE}"; mkdir -p "$WB"
  [ -f "$ALN" ] || { echo "  [$TYPE $SCALE] MISSING aln $ALN"; return; }
  echo; echo "──── $TYPE $SCALE ($SUB) ────"

  # pollers: power(W), mem.used(MiB), util(%) — 1s cadence, whole run
  local PW="$WB/power.log" MU="$WB/memutil.log"
  ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > "$PW" 2>&1 & local PWPID=$!
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > "$MU" 2>&1 & local MUPID=$!
  local T_ALL0=$(date +%s)

  # ---- subsample 5000 sites (seed 1, == the validated CTF) ----
  local T0=$(date +%s)
  python3 - "$ALN" "$KSUB" "$WB/sub.phy" <<'PY'
import sys, random
src,K,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
with open(src) as f:
    f.readline(); names=[]; seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]); K=min(K,L); random.seed(1); cols=sorted(random.sample(range(L),K))
open(out,"w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
PY
  local T_SUB=$(($(date +%s)-T0))

  # ---- coarse: FULL -m MF on the subsample, --jolt --gpu (eligible->GPU, +R/+I->CPU) ----
  local T0=$(date +%s)
  JOLT_DEBUG=1 "$BIN" --jolt --gpu -m MF -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
  local T_C=$(($(date +%s)-T0))
  [ -f "$WB/coarse.treefile" ] || { echo "  COARSE FAILED ($TYPE $SCALE)"; kill $PWPID $MUPID 2>/dev/null; return; }
  local ENG=$(grep -c '\[JOLT\] model=' "$WB/coarse.stdout" 2>/dev/null)
  local DEC=$(grep -hcE '\[JOLT-GATE\] decline reason=' "$WB/coarse.stdout" 2>/dev/null)
  local NTILE=$(grep -hoE '\[JOLT-TILE\][^=]*nTile=[0-9]+' "$WB/coarse.stdout" 2>/dev/null | grep -oE 'nTile=[0-9]+' | head -1 | cut -d= -f2); NTILE="${NTILE:-1}"
  echo "  subsample ${T_SUB}s ; coarse -m MF ${T_C}s ; engage=$ENG decline=$DEC nTile=$NTILE"

  # ---- NATIVE subsample-BIC rerank + rate-het detector + ineligible-skip (the FIXED gate, PART X X.5.5) ----
  python3 - "$WB/coarse.iqtree" "$KSUB" "$SCALE" "$TOPK" > "$WB/topk.txt" 2> "$WB/rerank.log" <<'PY'
import sys,re,math
iq,m,N,K=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
def inelig(n): return ('+R' in n) or ('+I' in n and '+G' not in n)
rows=[]
for line in open(iq):
    mm=row.match(line)
    if not mm: continue
    nm,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5))
    p=(bic+2*logl)/math.log(m); rows.append((nm,logl,bic,round(p),inelig(nm)))
nat=sorted(rows,key=lambda r:r[2])
be=min((r for r in rows if not r[4]),key=lambda r:r[2],default=None)
bi=min((r for r in rows if r[4]),    key=lambda r:r[2],default=None)
sys.stderr.write("  NATIVE BIC top-5: "+", ".join(r[0] for r in nat[:5])+"\n")
if be and bi:
    margin=abs(bi[3]-be[3])/2.0; lead=be[2]-bi[2]; flag=lead>margin
    sys.stderr.write(f"  detector best_elig={be[0]}({be[2]:.1f}) best_inel={bi[0]}({bi[2]:.1f}) lead={lead:.1f} margin={margin:.1f} RATE_HET={flag}\n")
for r in nat[:K]:
    skip = r[4] and (be is not None) and (r[2] > be[2] + abs(r[3]-be[3])/2.0)
    print(f"MODEL:{r[0]}:{'skip' if skip else 'refine'}")
PY
  cat "$WB/rerank.log" | sed 's/^/  /'
  mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
  echo "  top-${TOPK}: ${TOPMODELS[*]}"

  # ---- refine top-k on FULL data, --jolt --gpu -te coarse, per-model wall budget ----
  local T_R_TOTAL=0 i=0
  for ENTRY in "${TOPMODELS[@]}"; do
    i=$((i+1)); local M="${ENTRY%%:*}" ACT="${ENTRY##*:}"
    if [ "$ACT" = skip ]; then echo "  refine $i $M: SKIPPED (ineligible, detector: cannot win)"; continue; fi
    local T0=$(date +%s)
    timeout ${REFINE_BUDGET}s env JOLT_DEBUG=1 "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine_${i}" -redo > "$WB/refine_${i}.stdout" 2>&1
    local rc=$?; local T_R=$(($(date +%s)-T0)); T_R_TOTAL=$((T_R_TOTAL+T_R))
    if [ $rc -eq 124 ]; then echo "  refine $i $M: OVER BUDGET ${REFINE_BUDGET}s -> carried UNREFINED"; continue; fi
    local lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND" "$WB/refine_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
    local jn=$(grep -c '\[JOLT\] model' "$WB/refine_${i}.stdout" 2>/dev/null)
    echo "  refine $i $M: wall=${T_R}s lnL=${lnl:-NA} JOLT_calls=${jn:-0}"
  done
  local T_ALL=$(($(date +%s)-T_ALL0)); kill $PWPID $MUPID 2>/dev/null; sleep 1

  # ---- reduce: winner (min full BIC from IQ-TREE's own .iqtree), worst parity rel, energy, peak VRAM, max util ----
  python3 - "$WB" "$TYPE" "$SCALE" "$DEVLABEL" "$(hostname)" "$NTILE" "$T_ALL" "$T_SUB" "$T_C" "$T_R_TOTAL" "$ENG" "$DEC" "$BINMD5" "$SUMMARY" <<'PY'
import sys,re,os,glob
WB,TYPE,SCALE,DEV,HOST,NTILE,T_ALL,T_SUB,T_C,T_R,ENG,DEC,MD5,SUMMARY=sys.argv[1:15]
# winner: among refine_*.iqtree, the one IQ-TREE scores best by BIC (its own correct N)
best=None
for iqf in sorted(glob.glob(f"{WB}/refine_*.iqtree")):
    txt=open(iqf).read()
    mb=re.search(r'Bayesian information criterion \(BIC\) score:\s*(-?\d+\.\d+)',txt)
    ml=re.search(r'Log-likelihood of the tree:\s*(-?\d+\.\d+)',txt)
    mm=re.search(r'Best-fit model.*?:\s*(\S+)',txt) or re.search(r'Model of substitution:\s*(\S+)',txt)
    if mb and ml:
        bic=float(mb.group(1)); lnl=float(ml.group(1)); name=mm.group(1) if mm else os.path.basename(iqf)
        if best is None or bic<best[2]: best=(name,lnl,bic)
# worst parity rel + max nTile across all [JOLT]/[JOLT-TILE] lines (refine on full data carries the real nTile)
worst=0.0; ntile=int(NTILE)
for sf in glob.glob(f"{WB}/*.stdout"):
    for line in open(sf):
        m=re.search(r'\[JOLT\].*rel=([0-9.eE+-]+)',line)
        if m:
            try: worst=max(worst,float(m.group(1)))
            except: pass
        t=re.search(r'nTile=([0-9]+)',line)
        if t: ntile=max(ntile,int(t.group(1)))
NTILE=str(ntile)
# energy from power.log (1s cadence)
pw=[float(x) for x in open(f"{WB}/power.log") if x.strip() and x.strip()[0].isdigit()]
J=sum(pw)*1.0; wh=J/3600.0; meanw=sum(pw)/max(len(pw),1)
# peak VRAM + max util from memutil.log ("mem, util")
pv=0; mu=0
for line in open(f"{WB}/memutil.log"):
    p=line.split(',')
    if len(p)==2:
        try: pv=max(pv,int(p[0])); mu=max(mu,int(p[1]))
        except: pass
wn = best[0] if best else "NA"; lnl = f"{best[1]:.3f}" if best else "NA"; bic=f"{best[2]:.1f}" if best else "NA"
row=[TYPE,SCALE,DEV,HOST,NTILE,T_ALL,T_SUB,T_C,T_R,f"{wh:.3f}",f"{meanw:.0f}",str(pv),str(mu),wn,lnl,bic,f"{worst:.3e}",ENG,DEC,MD5]
open(SUMMARY,"a").write("\t".join(map(str,row))+"\n")
print(f"  => winner={wn} full_lnL={lnl} full_BIC={bic} parity_worst={worst:.3e} energy={wh:.3f}Wh peakVRAM={pv}MiB maxutil={mu}% nTile={NTILE} TOTAL={T_ALL}s")
PY
}

for TYPE in $TYPES; do for SCALE in $SCALES; do run_one "$TYPE" "$SCALE"; done; done
echo; echo "════════ SWEEP DONE $(date -Iseconds) ════════"; echo "── SUMMARY.tsv ──"; column -t -s$'\t' "$SUMMARY"
