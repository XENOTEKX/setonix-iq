#!/bin/bash
# run_bench_sweep_h200_dcgm.sh — H200 AA SWEEP with TRUE SM-active% (DCGM) + AA-10M full -m MF on the headline card.
# Twin of run_bench_sweep_a100_dcgm.sh, retargeted at the H200 (gpuhopper, 141 GB) so the true SM metrics are
# measured on the SAME device as the frozen headline sweep (job 171012178). It:
#   (1) runs full -m MF coarse-to-fine on AA at all four scales 10K/100K/1M/10M (10M is the genome-scale headline);
#   (2) attaches a DCGM sampler (dcgmi dmon, ~nvidia-smi overhead, NOT ncu) for the SM metrics nvidia-smi cannot
#       report: SM_ACTIVE (true SM utilisation), SM_OCCUPANCY, DRAM_ACTIVE (mem-BW util), PIPE_FP64_ACTIVE;
#   (3) emits an SM_SUMMARY.tsv sidecar directly comparable to the A100 DCGM run (bench_a100_dcgm/SM_SUMMARY.tsv).
# Honest caveat: the DCGM sampler adds a small overhead, so THIS run's wall/energy are H200-with-DCGM, not the
# pristine clean numbers (the frozen H200 sweep 171012178 remains the headline wall/energy). The deliverables
# here are: real per-cell SM-active%/occupancy/DRAM% on the H200, including the AA-10M -m MF genome-scale cell.
# Submit: qsub gadi-ci/gpu-modelfinder/run_bench_sweep_h200_dcgm.sh
#PBS -N bench-h200-dcgm
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=180GB
#PBS -l walltime=08:00:00
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

SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
DEV=h200; NT="${PBS_NCPUS:-12}"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
WB="$SRC/bench_${DEV}_dcgm"; mkdir -p "$WB"; cd "$WB"
SUM="$WB/SUMMARY.tsv"; SMSUM="$WB/SM_SUMMARY.tsv"
KSUB=5000; TOPK=3; REFINE_BUDGET=2400

echo "════════ H200 AA SWEEP + DCGM SM — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader

# ---- DCGM availability: true SM-active% (field 1002), occupancy (1003), DRAM-active (1005), FP64-pipe (1006) ----
DCGM_OK=0
if command -v dcgmi >/dev/null 2>&1; then
  command -v nv-hostengine >/dev/null 2>&1 && (nv-hostengine >/dev/null 2>&1 || true)
  sleep 2
  if dcgmi dmon -e 1002 -c 1 -d 1000 >/dev/null 2>&1; then DCGM_OK=1; fi
fi
echo "──── DCGM available: $DCGM_OK  (1 = true SM-active% sampled; 0 = fallback, SM columns = NA) ────"

echo "──── rebuild on-node (tiling binary; CUDA arch 70;80;90 incl. H200 sm_90) ────"
( cd "$BUILD_ON" && make -j16 iqtree3 > "$WB/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; tail -2 "$BUILD_ON/make.log" | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
BIN_MD5=$(md5sum "$BIN" | cut -d' ' -f1); echo "  bin md5=$BIN_MD5"

# main schema (identical to the H200 sweep → tools/bench_to_logs.py compatible)
[ -f "$SUM" ] || printf "type\tscale\tdevice\thost\tbin_md5\twall_total_s\twall_sub_s\twall_coarse_s\twall_refine_s\tenergy_wh\tmean_w\tpeak_vram_mib\tmax_util\tworst_parity_rel\twinner\tfull_lnL\tfull_bic\tengage\tdecline\tnTile\n" > "$SUM"
# SM sidecar (the DCGM deliverable)
[ -f "$SMSUM" ] || printf "type\tscale\twall_total_s\tnTile\tsm_active_mean_pct\tsm_active_max_pct\tsm_occupancy_mean_pct\tdram_active_mean_pct\tfp64_active_mean_pct\tgpu_busy_max_pct\n" > "$SMSUM"

bench_point () {   # $1=SEQTYPE  $2=scale  $3=ALN
  local TY="$1" SC="$2" ALN="$3"
  local PD="$WB/${TY}_${SC}"; mkdir -p "$PD"; cd "$PD"
  echo; echo "════════ ${TY} ${SC} sites ════════ $(date -Iseconds)"
  [ -f "$ALN" ] || { echo "  MISSING $ALN — skip"; cd "$WB"; return; }
  if [ -f "$PD/bench.done" ]; then echo "  RESUME: $TY $SC already complete — skipping"; cd "$WB"; return; fi

  # pollers: nvidia-smi (power→energy, mem→VRAM, util→busy%) AND DCGM (true SM%)
  local PWLOG="$PD/gpu_poll.csv" DCGMLOG="$PD/dcgm.csv" DCGM_PID=""
  ( while true; do nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 &
  local POLL=$!
  if [ "$DCGM_OK" = "1" ]; then
    ( dcgmi dmon -e 1002,1003,1005,1006 -d 2000 ) > "$DCGMLOG" 2>&1 &
    DCGM_PID=$!
  fi
  local T_ALL0=$(date +%s)

  # ---- subsample KSUB sites (seed 1) ----
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

  # ---- coarse: full -m MF on the subsample ----
  export JOLT_DEBUG=1
  T0=$(date +%s)
  "$BIN" --jolt --gpu -m MF -s "$PD/sub.phy" -nt "$NT" -pre "$PD/coarse" -redo > "$PD/coarse.stdout" 2>&1
  local T_C=$(($(date +%s)-T0))
  if [ ! -f "$PD/coarse.treefile" ]; then echo "  COARSE FAILED"; kill $POLL 2>/dev/null; [ -n "$DCGM_PID" ] && kill $DCGM_PID 2>/dev/null; cd "$WB"; return; fi
  local ENGAGE=$(grep -c '\[JOLT\] model=' "$PD/coarse.stdout" 2>/dev/null)
  local DECLINE=$(grep -cE '\[JOLT-GATE\] decline|decline reason=' "$PD/coarse.stdout" 2>/dev/null)
  echo "  subsample ${T_SUB}s ; coarse -m MF ${T_C}s ; engage=$ENGAGE decline=$DECLINE"

  # ---- native subsample-BIC rerank + rate-het detector -> top-k ----
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

  # ---- refine top-k on FULL data ----
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
  local T_ALL=$(($(date +%s)-T_ALL0)); kill $POLL 2>/dev/null; [ -n "$DCGM_PID" ] && kill $DCGM_PID 2>/dev/null; sleep 1

  # ---- winner = min full BIC over refined top-k ----
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

  # ---- energy / VRAM / busy% from nvidia-smi poll ----
  read -r EWH MEANW < <(python3 -c "
v=[float(x.split(',')[0]) for x in open('$PWLOG') if x.strip() and x.strip()[0].isdigit()]
dt=2.0; J=sum(v)*dt; print(f'{J/3600:.3f} {sum(v)/max(len(v),1):.0f}')" 2>/dev/null || echo "0 0")
  local PEAKV=$(awk -F, 'NF>=2{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print m+0}' "$PWLOG" 2>/dev/null)
  local MAXU=$(awk -F, 'NF>=3{gsub(/ /,"",$3); if($3+0>u)u=$3+0} END{print u+0}' "$PWLOG" 2>/dev/null)
  local PAR=$(grep -hoE 'rel=[0-9.eE+-]+' "$PD"/coarse.stdout "$PD"/refine_*.stdout 2>/dev/null | sed 's/rel=//' | sort -g | tail -1)
  local NTILE=$(grep -hoE '\[JOLT-TILE\].*nTile=[0-9]+' "$PD"/refine_*.stdout "$PD"/coarse.stdout 2>/dev/null | grep -oE 'nTile=[0-9]+' | sed 's/nTile=//' | sort -n | tail -1)
  [ -z "$NTILE" ] && NTILE=1; [ -z "$PAR" ] && PAR=NA; [ -z "$PEAKV" ] && PEAKV=0; [ -z "$MAXU" ] && MAXU=0

  # ---- DCGM: per-GPU means; pick the busiest GPU (our job's) so multi-GPU visibility can't dilute ----
  local SMACT_M="NA" SMACT_X="NA" SMOCC_M="NA" DRAMA_M="NA" FP64_M="NA"
  if [ "$DCGM_OK" = "1" ] && [ -s "$DCGMLOG" ]; then
    read -r SMACT_M SMACT_X SMOCC_M DRAMA_M FP64_M < <(awk '
      /^[[:space:]]*GPU[[:space:]]+[0-9]/ && $3 ~ /^[0-9.]+$/ {
        id=$2; c[id]++; a[id]+=$3; if($3>am[id])am[id]=$3; o[id]+=$4; d[id]+=$5; f[id]+=$6 }
      END{ best=""; bv=-1; for(id in c){ mm=a[id]/c[id]; if(mm>bv){bv=mm;best=id} }
           if(best!=""){id=best; printf "%.1f %.1f %.1f %.1f %.1f",100*a[id]/c[id],100*am[id],100*o[id]/c[id],100*d[id]/c[id],100*f[id]/c[id]}
           else print "NA NA NA NA NA" }' "$DCGMLOG" 2>/dev/null || echo "NA NA NA NA NA")
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TY" "$SC" "$DEV" "$(hostname)" "$BIN_MD5" "$T_ALL" "$T_SUB" "$T_C" "$T_R_TOTAL" \
    "$EWH" "$MEANW" "$PEAKV" "$MAXU" "$PAR" "$WINNER" "$WLNL" "$WBIC" "$ENGAGE" "$DECLINE" "$NTILE" >> "$SUM"
  printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n" \
    "$TY" "$SC" "$T_ALL" "$NTILE" "$SMACT_M" "$SMACT_X" "$SMOCC_M" "$DRAMA_M" "$FP64_M" "$MAXU" >> "$SMSUM"
  echo "  => winner=$WINNER | wall=${T_ALL}s VRAM=${PEAKV}MiB nTile=$NTILE busy%=${MAXU} parity=$PAR"
  echo "  => SM-active(mean/max)=${SMACT_M}/${SMACT_X}%  occupancy=${SMOCC_M}%  DRAM-active=${DRAMA_M}%  FP64-pipe=${FP64_M}%"
  touch "$PD/bench.done"
  cd "$WB"
}

# AA ONLY, all four scales (10M is the genome-scale headline + tiling/host-mem H200 verification)
for SC in 10000 100000 1000000 10000000; do
  bench_point AA "$SC" "$BASE/AA/LG+I+G4/taxa_100/len_${SC}/tree_1/alignment_${SC}.phy"
done

echo; echo "════════ SUMMARY.tsv ════════"; column -t -s$'\t' "$SUM"
echo; echo "════════ SM_SUMMARY.tsv (DCGM true SM-active%) ════════"; column -t -s$'\t' "$SMSUM"
command -v nv-hostengine >/dev/null 2>&1 && (nv-hostengine --term >/dev/null 2>&1 || true)
echo "════════ DONE $(date -Iseconds) ════════"
