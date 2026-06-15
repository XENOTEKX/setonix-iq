#!/bin/bash
# run_bench_sweep_v100.sh — the FULL -m MF CTF sweep on a V100 (32 GB), to make the V100 device-comparison cells
# directly comparable to the H200/A100 -m MF cells (replacing the single-model -te capability run). Same pipeline as
# run_bench_sweep_clean_a100.sh: subsample 5000 -> coarse -m MF on GPU (JOLT for eligible, CPU fallback for +R/+I) ->
# native-BIC rerank + rate-het detector -> JOLT-refine top-k on full data (tiled). AA x {10K,100K,1M,10M} (the AA
# device table). Uses the SAME tiling binary proven on the V100 (job 171187221). Per-run [JOLT] GPU=CPU parity check.
# RESUME-safe (bench.done sentinel per point) — if it times out, resubmit to finish.
#PBS -N bench-mf-v100
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=96GB
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cuda/12.5.1  2>/dev/null || true
module load eigen/3.3.7  2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
DEV=v100; NT="${PBS_NCPUS:-12}"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
WB="$SRC/bench_${DEV}_mf"; mkdir -p "$WB"; cd "$WB"
SUM="$WB/SUMMARY.tsv"
KSUB=5000; TOPK=3; REFINE_BUDGET=3600   # V100 is slower than A100 — bump per-model GPU refine budget (still << CPU-at-10M blowup)

echo "════════ FULL -m MF CTF SWEEP on ${DEV} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
# Use the EXISTING binary proven on the V100 (job 171187221) — no on-node rebuild (avoids any arch drift; sm_70 OK).
[ -x "$BIN" ] || { echo "MISSING $BIN"; exit 1; }
BIN_MD5=$(md5sum "$BIN" | cut -d' ' -f1); echo "  bin md5=$BIN_MD5"

[ -f "$SUM" ] || printf "type\tscale\tdevice\thost\tbin_md5\twall_total_s\twall_sub_s\twall_coarse_s\twall_refine_s\tenergy_wh\tmean_w\tpeak_vram_mib\tmax_util\tworst_parity_rel\twinner\tfull_lnL\tfull_bic\tengage\tdecline\tnTile\n" > "$SUM"

bench_point () {   # $1=SEQTYPE  $2=scale  $3=ALN
  local TY="$1" SC="$2" ALN="$3"
  local PD="$WB/${TY}_${SC}"; mkdir -p "$PD"; cd "$PD"
  echo; echo "════════ ${TY} ${SC} sites ════════ $(date -Iseconds)"
  [ -f "$ALN" ] || { echo "  MISSING $ALN — skip"; cd "$WB"; return; }
  if [ -f "$PD/bench.done" ]; then echo "  RESUME: $TY $SC already complete — skipping"; cd "$WB"; return; fi

  local PWLOG="$PD/gpu_poll.csv"
  ( while true; do nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 &
  local POLL=$!
  local T_ALL0=$(date +%s)

  local T0=$(date +%s)
  python3 - "$ALN" "$KSUB" "$PD/sub.phy" <<'PY'
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

  export JOLT_DEBUG=1
  T0=$(date +%s)
  "$BIN" --jolt --gpu -m MF -s "$PD/sub.phy" -nt "$NT" -pre "$PD/coarse" -redo > "$PD/coarse.stdout" 2>&1
  local T_C=$(($(date +%s)-T0))
  if [ ! -f "$PD/coarse.treefile" ]; then echo "  COARSE FAILED"; kill $POLL 2>/dev/null; cd "$WB"; return; fi
  local ENGAGE=$(grep -c '\[JOLT\] model=' "$PD/coarse.stdout" 2>/dev/null)
  local DECLINE=$(grep -cE '\[JOLT-GATE\] decline|decline reason=' "$PD/coarse.stdout" 2>/dev/null)
  echo "  subsample ${T_SUB}s ; coarse -m MF ${T_C}s ; engage=$ENGAGE decline=$DECLINE"

  python3 - "$PD/coarse.iqtree" "$KSUB" "$SC" "$TOPK" > "$PD/topk.txt" 2>"$PD/rerank.log" <<'PY'
import sys,re,math
iq,m,N,K=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
def ineligible(name):
    return ('+R' in name) or ('+I' in name and '+G' not in name)
rows=[]
for line in open(iq):
    mm=row.match(line)
    if not mm: continue
    name,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5))
    rows.append((name,logl,bic,ineligible(name)))
nat_all=sorted(rows, key=lambda r:r[2])
be=min((r for r in rows if not r[3]), key=lambda r:r[2], default=None)
for r in nat_all[:K]:
    skip = r[3] and (be is not None) and (r[2] > be[2])
    print(f"MODEL:{r[0]}:{'skip' if skip else 'refine'}")
PY
  mapfile -t TOPMODELS < <(grep '^MODEL:' "$PD/topk.txt" | sed 's/^MODEL://')
  echo "  top-${TOPK}: ${TOPMODELS[*]}"

  local T_R_TOTAL=0 i=0
  for ENTRY in "${TOPMODELS[@]}"; do
    i=$((i+1)); local M="${ENTRY%%:*}" ACT="${ENTRY##*:}"
    if [ "$ACT" = "skip" ]; then echo "  refine $i $M: SKIPPED (ineligible, behind best eligible)"; continue; fi
    T0=$(date +%s)
    timeout ${REFINE_BUDGET}s "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$PD/coarse.treefile" -nt "$NT" -pre "$PD/refine_${i}" -redo > "$PD/refine_${i}.stdout" 2>&1
    local rc=$?; local T_R=$(($(date +%s)-T0)); T_R_TOTAL=$((T_R_TOTAL+T_R))
    if [ $rc -eq 124 ]; then echo "  refine $i $M: OVER BUDGET (${REFINE_BUDGET}s) -> carried UNREFINED"; continue; fi
    local lnl=$(grep -iE "Log-likelihood of the tree" "$PD/refine_${i}.iqtree" 2>/dev/null | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)
    echo "  refine $i $M: wall=${T_R}s lnL=${lnl:-NA}"
  done
  local T_ALL=$(($(date +%s)-T_ALL0)); kill $POLL 2>/dev/null; sleep 1

  read -r WINNER WLNL WBIC < <(python3 - "$PD" "${TOPMODELS[*]}" <<'PY'
import sys,re,os,glob
pd=sys.argv[1]; best=("NA","NA","NA"); bb=None
for iqf in sorted(glob.glob(os.path.join(pd,"refine_*.iqtree"))):
    txt=open(iqf).read()
    lm=re.search(r'Log-likelihood of the tree:\s*(-?\d+\.\d+)',txt)
    bm=re.search(r'Bayesian information criterion \(BIC\) score:\s*(-?\d+\.\d+)',txt)
    mm=re.search(r'Model of substitution:\s*(\S+)',txt) or re.search(r'Best-fit model.*?:\s*(\S+)',txt)
    if lm and bm:
        bic=float(bm.group(1))
        if bb is None or bic<bb:
            bb=bic; best=(mm.group(1) if mm else "?", lm.group(1), bm.group(1))
print(best[0],best[1],best[2])
PY
)

  read -r EWH MEANW < <(python3 -c "
v=[float(x.split(',')[0]) for x in open('$PWLOG') if x.strip() and x.strip()[0].isdigit()]
dt=2.0; J=sum(v)*dt; print(f'{J/3600:.3f} {sum(v)/max(len(v),1):.0f}')" 2>/dev/null || echo "0 0")
  local PEAKV=$(awk -F, 'NF>=2{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print m+0}' "$PWLOG" 2>/dev/null)
  local MAXU=$(awk -F, 'NF>=3{gsub(/ /,"",$3); if($3+0>u)u=$3+0} END{print u+0}' "$PWLOG" 2>/dev/null)
  local PAR=$(grep -hoE 'rel=[0-9.eE+-]+' "$PD"/coarse.stdout "$PD"/refine_*.stdout 2>/dev/null | sed 's/rel=//' | sort -g | tail -1)
  local NTILE=$(grep -hoE '\[JOLT-TILE\].*nTile=[0-9]+' "$PD"/refine_*.stdout "$PD"/coarse.stdout 2>/dev/null | grep -oE 'nTile=[0-9]+' | sed 's/nTile=//' | sort -n | tail -1)
  [ -z "$NTILE" ] && NTILE=1; [ -z "$PAR" ] && PAR=NA; [ -z "$PEAKV" ] && PEAKV=0; [ -z "$MAXU" ] && MAXU=0

  printf "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TY" "$SC" "$DEV" "$(hostname)" "$BIN_MD5" "$T_ALL" "$T_SUB" "$T_C" "$T_R_TOTAL" \
    "$EWH" "$MEANW" "$PEAKV" "$MAXU" "$PAR" "$WINNER" "$WLNL" "$WBIC" "$ENGAGE" "$DECLINE" "$NTILE" >> "$SUM"
  echo "  => winner=$WINNER lnL=$WLNL BIC=$WBIC | wall=${T_ALL}s E=${EWH}Wh VRAM=${PEAKV}MiB util=${MAXU}% nTile=$NTILE parity=$PAR"
  touch "$PD/bench.done"
  cd "$WB"
}

for SC in 10000 100000 1000000 10000000; do
  bench_point AA "$SC" "$BASE/AA/LG+I+G4/taxa_100/len_${SC}/tree_1/alignment_${SC}.phy"
done

echo; echo "════════ SUMMARY.tsv ════════"; column -t -s$'\t' "$SUM"
echo "════════ DONE $(date -Iseconds) ════════"
