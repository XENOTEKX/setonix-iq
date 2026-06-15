#!/bin/bash
# run_beagle_vs_jolt_scale_h200.sh — phase-2b + LARGER scale: JOLT vs BEAGLE-4.0 (tensor-cores), lnL + GRADIENT, H200.
# Extends the A100 phase-2 (job 171265226, lnL only, 100K) with (a) the gradient head-to-head and (b) genome-scale lnL.
#
#   Point 1 — AA-100K, lnL + ALL-BRANCH GRADIENT (the phase-2b piece):
#     JOLT gpu_k1_lnl (k1_node lnL) + gpu_k7_grad ([GRAD-TIMING] 1 postorder + 1 preorder + all-edge theta+reduce)
#     BEAGLE beagle_jolt_bench grad=1 (lnL_only + grad_full = pre-order sweep + calculateEdgeDerivatives), tensor & cuda.
#   Point 2 — AA-1M, lnL ONLY (genome-scale; the standalone gradient harnesses OOM at 1M — documented, not hidden):
#     JOLT gpu_k1_lnl AA-1M ; BEAGLE grad=0 at nPat=1e6 (≈127GB, tight on 141GB — if it OOMs that is a real finding:
#     BEAGLE keeps ALL tip+internal partials resident; JOLT uses compact tip states + internal-only storage, ≈63GB).
# Both tools at matched dims (s=20, 100 taxa, +G4, FP64); JOLT uses nptn=nsite (no compression) → BEAGLE matched.
#PBS -N beagle-vs-jolt-scale
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=180GB
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load gcc/12.2.0  2>/dev/null || true
module load cuda/12.5.1 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/apps/cuda/12.5.1}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

GMF=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder
BEAGLE=/scratch/rc29/as1708/beagle-tensorcores
INC=$BEAGLE/build-tc/install/include/libhmsbeagle-1
LIB=$BEAGLE/build-tc/install/lib
BENCH=$GMF/beagle_jolt_bench.cpp      # repo source (version-controlled)
BENCHBIN=/scratch/rc29/as1708/iqtree3-gpu/beagle_vs_jolt/beagle_jolt_bench_h200
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100
ALN100K=$BASE/len_100000/tree_1/alignment_100000.phy
ALN1M=$BASE/len_1000000/tree_1/alignment_1000000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
TREE1M=$BASE/len_1000000/tree_1/tree_1.full.treefile   # real 1M simulation tree (so the 1M lnL is meaningful)
REPS=20
WB=/scratch/rc29/as1708/iqtree3-gpu/beagle_vs_jolt_scale; mkdir -p "$WB"; cd "$WB"

echo "════════ BEAGLE-4.0 vs JOLT — SCALE + GRADIENT — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | tail -2

echo; echo "──── build JOLT kernels (nvcc -O3 -arch=sm_90 for H200, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_90 "$GMF/gpu_k1_lnl.cu"  -o "$WB/gpu_k1_lnl"  && echo "k1 built" || { echo "k1 BUILD FAIL"; exit 1; }
nvcc -O3 -std=c++17 -arch=sm_90 "$GMF/gpu_k7_grad.cu" -o "$WB/gpu_k7_grad" && echo "k7 built" || { echo "k7 BUILD FAIL"; exit 1; }
echo "──── build BEAGLE client from repo source ────"
g++ -O2 -std=c++14 -I"$INC" "$BENCH" -o "$BENCHBIN" -L"$LIB" -lhmsbeagle -Wl,-rpath,"$LIB" && echo "bench built" || { echo "bench BUILD FAIL"; exit 1; }

run_jolt_lnl () { local A=$1 TAG=$2 TR=${3:-$TREE}; echo; echo "──── JOLT k1_node lnL — $TAG (tree=$(basename $TR)) ────"; "$WB/gpu_k1_lnl" "$A" "$TR" g4 $REPS 2>&1 | tee "$WB/jolt_k1_$TAG.log"; }
run_jolt_grad(){ echo; echo "──── JOLT k7 all-branch gradient — $2 ────"; "$WB/gpu_k7_grad" "$1" "$TREE" g4 $REPS 2>&1 | tee "$WB/jolt_k7_$2.log"; }
run_beagle  () { local NP=$1 MODE=$2 G=$3; echo; echo "──── BEAGLE $MODE (grad=$G) nPat=$NP ────"; "$BENCHBIN" 20 100 "$NP" 4 $REPS 3 "$MODE" "$G" 2>&1 | tee "$WB/beagle_${MODE}_${NP}_g${G}.log"; }

# ===== Point 1: AA-100K, lnL + GRADIENT =====
echo; echo "════════════════ POINT 1 — AA-100K (lnL + all-branch gradient) ════════════════"
run_jolt_lnl  "$ALN100K" 100k
run_jolt_grad "$ALN100K" 100k
run_beagle 100000 tensor 1
run_beagle 100000 cuda   1

# ===== Point 2: AA-1M, lnL only (genome scale) =====
echo; echo "════════════════ POINT 2 — AA-1M (lnL only; gradient harnesses OOM at 1M) ════════════════"
run_jolt_lnl "$ALN1M" 1m "$TREE1M"
run_beagle 1000000 tensor 0
run_beagle 1000000 cuda   0

# ===== comparison =====
echo; echo "════════════════ HEAD-TO-HEAD SUMMARY (per-eval ms, AA-20 +G4 FP64, H200) ════════════════"
python3 - "$WB" <<'PY'
import sys,re,os
wb=sys.argv[1]
def rd(f):
    p=os.path.join(wb,f); return open(p).read() if os.path.exists(p) else ""
def lastms(t): m=re.findall(r"([0-9]+\.[0-9]+)\s*ms",t); return m[-1] if m else "?"
def gradms(t): m=re.search(r"\[GRAD-TIMING\].*?:\s*([0-9.]+)\s*ms",t); return m.group(1) if m else "?"
def bget(t,pat): m=re.search(pat,t); return m.group(1) if m else "?"
print(f"{'nPat':>8} {'tool/mode':<20}{'lnL_eval_ms':>13}{'grad_eval_ms':>14}{'note':>8}")
# 100K
k1=rd("jolt_k1_100k.log"); k7=rd("jolt_k7_100k.log")
print(f"{'100000':>8} {'JOLT k1/k7':<20}{lastms(k1):>13}{gradms(k7):>14}{'real':>8}")
for m in ('tensor','cuda'):
    t=rd(f"beagle_{m}_100000_g1.log")
    lnl=bget(t,r'lnL_only_ms=([0-9.]+)'); gr=bget(t,r'grad_full_ms=([0-9.]+)'); fd='FD'+bget(t,r'gradFD=([0-9])')
    print(f"{'100000':>8} {'BEAGLE '+m:<20}{lnl:>13}{gr:>14}{fd:>8}")
# 1M
k1m=rd("jolt_k1_1m.log")
print(f"{'1000000':>8} {'JOLT k1':<20}{lastms(k1m):>13}{'(OOM)':>14}{'real':>8}")
for m in ('tensor','cuda'):
    t=rd(f"beagle_{m}_1000000_g0.log")
    ok="OK" if "RESULT" in t else "OOM/FAIL"
    print(f"{'1000000':>8} {'BEAGLE '+m:<20}{bget(t,r'lnL_only_ms=([0-9.]+)'):>13}{'-':>14}{ok:>8}")
print()
print("lnL_eval = 1 postorder sweep + root (JOLT k1_node ≡ BEAGLE lnL_only).")
print("grad_eval = JOLT [GRAD-TIMING] (1 postorder+1 preorder+all-edge theta+reduce) ≡ BEAGLE grad_full (pre-order+calculateEdgeDerivatives).")
print("BEAGLE gradFD = finite-difference self-check of its gradient (1=PASS). JOLT k7 FD-validated in-log.")
PY
echo "════════ DONE $(date -Iseconds) ════════"
