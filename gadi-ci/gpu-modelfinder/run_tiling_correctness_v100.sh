#!/bin/bash
# run_tiling_correctness_v100.sh — G.7.1 PATTERN TILING, stage V.A (correctness gate).
# Validates that the new pattern-tiling outer loop in gpu_jolt_optimize reproduces the one-shot result:
#   chunked lnL (JOLT_NTILE=T) == one-shot lnL (JOLT_NTILE=1) to rel<=1e-12, and each tier's own GPU-vs-CPU
#   write-back self-check still PASSes (rel<=1e-9). Run LG+G4 -te on AA-100K (fits a V100 one-shot ~9 GB), at
#   T in {1,4,8,40}; AA-100K@T=40 is ~0.23 GB, exercising the deep-tile path. Reports per-tier peak VRAM (must
#   shrink ~T) + iters. THIS IS THE GATE BEFORE THE 10M ENGAGE TEST.
#
#PBS -N tiling-correct
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=01:00:00
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
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB="$SRC/tiling_correct_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"

echo "════════ G.7.1 V.A tiling correctness — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "──── rebuild on-node (gpu_lnl_intree.cu changed) ────"
( cd "$BUILD_ON" && make -j12 iqtree3 > "$WB/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; tail -4 "$BUILD_ON/make.log" | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "  binary md5=$(md5sum "$BIN" | cut -d' ' -f1)"

run_tier () {   # $1 = nTile
  local T="$1"; local pref="$WB/t${T}"; local gpulog="$WB/t${T}_gpu.log"
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$gpulog" 2>&1 &
  local poller=$!
  JOLT_NTILE=$T "$BIN" -s "$ALN" -m LG+G4 -te "$TREE" --jolt --gpu -nt 1 -pre "$pref" > "$WB/t${T}_run.log" 2>&1
  local rc=$?
  kill $poller 2>/dev/null
  local peak=$(awk -F, 'NR>0{gsub(/ /,"",$1); if($1+0>m)m=$1+0} END{print m}' "$gpulog")
  local jolt=$(grep -m1 "\[JOLT\]" "$WB/t${T}_run.log")
  echo "  T=$T exit=$rc peakVRAM=${peak}MiB"
  echo "    $jolt"
}

echo; echo "──── LG+G4 -te on AA-100K at T in {1,4,8,40} ────"
for T in 1 4 8 40; do run_tier "$T"; done

echo; echo "──── PARITY VERDICT (chunked vs one-shot) ────"
python3 - "$WB" <<'PY'
import re,sys,glob,os
wb=sys.argv[1]; vals={}
for f in sorted(glob.glob(os.path.join(wb,"t*_run.log"))):
    T=int(re.search(r"t(\d+)_run",f).group(1))
    txt=open(f).read()
    m=re.search(r"\[JOLT\].*?GPU lnL=(-?[\d.]+)\s+CPU lnL=(-?[\d.]+)\s+rel=([\d.eE+-]+)\s+(PASS|FAIL)",txt)
    it=re.search(r":\s*(\d+)\s+joint iters",txt)
    if m: vals[T]=(float(m.group(1)),float(m.group(2)),float(m.group(3)),m.group(4), it.group(1) if it else "?")
base=vals.get(1)
print(f"  {'T':>4} {'GPU_lnL':>20} {'GPUvsCPU_rel':>14} {'self':>5} {'iters':>6} {'chunked_vs_oneshot_rel':>24}")
ok=True
for T in sorted(vals):
    g,c,r,st,it=vals[T]
    cv = abs(g-base[0])/(abs(base[0])+1e-30) if base else float('nan')
    flag = "" if (T==1) else ("  <<<" if cv>1e-12 else "")
    if st!="PASS": ok=False
    if T!=1 and cv>1e-12: ok=False
    print(f"  {T:>4} {g:>20.6f} {r:>14.3e} {st:>5} {it:>6} {cv:>24.3e}{flag}")
print()
print("  GATE:", "PASS — chunked==one-shot rel<=1e-12 AND all self-checks PASS" if ok else "FAIL — see <<< rows")
PY
echo; echo "════════ DONE $(date -Iseconds) ════════"
