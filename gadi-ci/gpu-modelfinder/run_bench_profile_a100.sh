#!/bin/bash
# run_bench_profile_a100.sh — G.7.2 Nsight PROFILING pass (SEPARATE from the clean energy/wall Job A — profilers
# perturb timing+power, so the headline wall/energy come from the unprofiled sweep). Per (TYPE,SCALE), self-contained:
#   build a coarse tree (single-model on a 5000-site subsample, UNPROFILED) then
#   (1) nsys timeline of the full-data JOLT refine -> per-kernel GPU-time breakdown (cuda_gpu_kern_sum) — ALL scales.
#   (2) ncu deep metrics (achieved occupancy, SM%, DRAM%, regs) on the hot kernels (kj_derv_fused/k1_node/kj_pre),
#       capped to a few launches — ONLY at SCALE<=100000 (ncu kernel-replay is infeasible at 1M/10M).
# Emits prof_a100_<jobid>/<TYPE>_<SCALE>/{kernsum.csv,ncu.csv} that tools/bench_to_logs.py ingests.
# Profiled model: AA=LG+G4 (gamma), DNA=GTR+G4 (free-Q — the heaviest/representative DNA path). A100, the tiling binary.
# Reviewed/owned by as1708 (adapted from an agent draft; fixed: ncu must NOT use --page raw — parse_ncu needs the
# long Kernel/Metric/Value CSV the default details page emits; added PBS directives).
#PBS -N benchprof-a100
#PBS -P dx61
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=180GB
#PBS -l walltime=04:00:00
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
NSYS="${CUDA_HOME:-/apps/cuda/12.5.1}/bin/nsys"; NCU="${CUDA_HOME:-/apps/cuda/12.5.1}/bin/ncu"

TYPES="${TYPES:-AA DNA}"; SCALES="${SCALES:-10000 100000 1000000 10000000}"; DEVLABEL=a100
NT="${PBS_NCPUS:-16}"; KSUB=5000
SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
OUT="$SRC/prof_${DEVLABEL}_${PBS_JOBID:-local}"; mkdir -p "$OUT"; cd "$OUT"

echo "════════ G.7.2 PROFILE ${DEVLABEL} — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "── rebuild on-node ──"; ( cd "$BUILD_ON" && make -j"$NT" iqtree3 > "$OUT/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; [ $RC -ne 0 ] && { tail -5 "$BUILD_ON/make.log"; echo BUILD FAILED; exit 1; }
echo "  bin md5=$(md5sum "$BIN"|cut -d' ' -f1)"
echo "  nsys=$($NSYS --version 2>/dev/null | head -1) ; ncu=$($NCU --version 2>/dev/null | head -1)"

prof_one () {
  local TYPE="$1" SCALE="$2"
  local SUB MODEL; if [ "$TYPE" = AA ]; then SUB=LG+I+G4; MODEL=LG+G4; else SUB=GTR+I+G4; MODEL=GTR+G4; fi
  local ALN="$BASE/$TYPE/$SUB/taxa_100/len_$SCALE/tree_1/alignment_$SCALE.phy"
  local WB="$OUT/${TYPE}_${SCALE}"; mkdir -p "$WB"
  [ -f "$ALN" ] || { echo "  [$TYPE $SCALE] MISSING aln"; return; }
  echo; echo "──── $TYPE $SCALE  model=$MODEL ────"

  # coarse tree (single model on the 5000-site subsample) — UNPROFILED, just a fixed topology for -te
  python3 - "$ALN" "$KSUB" "$WB/sub.phy" <<'PY'
import sys,random
src,K,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
with open(src) as f:
    f.readline(); names=[];seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]);K=min(K,L);random.seed(1);cols=sorted(random.sample(range(L),K))
open(out,"w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
PY
  "$BIN" --jolt --gpu -m "$MODEL" -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
  [ -f "$WB/coarse.treefile" ] || { echo "  coarse tree FAILED"; return; }

  # (1) nsys timeline of the full-data JOLT refine
  echo "  [nsys] refine $MODEL on full $SCALE ..."
  JOLT_DEBUG=1 "$NSYS" profile -t cuda --sample=none --cpuctxsw=none -f true -o "$WB/refine_nsys" \
      "$BIN" --jolt --gpu -m "$MODEL" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine" -redo \
      > "$WB/nsys.stdout" 2>&1
  echo "    nsys exit=$?"
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true "$WB/refine_nsys.nsys-rep" > "$WB/kernsum.csv" 2>"$WB/nsysstats.err" \
    && { echo "    --- top kernels by GPU time ---"; head -8 "$WB/kernsum.csv" | sed 's/^/      /'; } \
    || echo "    nsys stats failed: $(tail -1 "$WB/nsysstats.err")"
  local lnl=$(grep -iE "Log-likelihood of the tree" "$WB/refine.iqtree" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  local rel=$(grep -oE 'rel=[0-9.eE+-]+' "$WB/nsys.stdout" 2>/dev/null | head -1)
  echo "    refine lnL=${lnl:-NA} parity=${rel:-NA}"

  # (2) ncu deep metrics on the hot kernels — small scale only. NOTE: default details page (long CSV:
  # Kernel Name/Metric Name/Metric Value) — NOT --page raw (which is wide and breaks parse_ncu).
  if [ "$SCALE" -le 100000 ]; then
    echo "  [ncu] hot kernels (capped launches) on full $SCALE ..."
    JOLT_DEBUG=1 "$NCU" --target-processes all --csv \
        --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread \
        -k "regex:kj_derv_fused|k1_node|kj_pre" --launch-skip 30 --launch-count 6 \
        "$BIN" --jolt --gpu -m "$MODEL" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/ncu_refine" -redo \
        > "$WB/ncu.csv" 2>"$WB/ncu.err"
    local nrc=$?
    if [ $nrc -eq 0 ] && grep -qE "kj_derv_fused|k1_node|kj_pre" "$WB/ncu.csv" 2>/dev/null; then
      echo "    ncu OK ($(grep -cE 'kj_derv_fused|k1_node|kj_pre' "$WB/ncu.csv") metric rows)"
    else
      echo "    ncu FAILED/empty (exit=$nrc): $(tail -2 "$WB/ncu.err" | tr '\n' ' ')"
    fi
  else
    echo "  [ncu] SKIPPED at $SCALE (replay infeasible >100K — nsys timeline only)"
  fi
}

for TYPE in $TYPES; do for SCALE in $SCALES; do prof_one "$TYPE" "$SCALE"; done; done
echo; echo "════════ PROFILE DONE $(date -Iseconds) ════════"
ls -la "$OUT"/*/kernsum.csv "$OUT"/*/ncu.csv 2>/dev/null | sed 's/^/  /'
