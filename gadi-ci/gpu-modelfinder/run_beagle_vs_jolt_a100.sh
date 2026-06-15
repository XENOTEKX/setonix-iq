#!/bin/bash
# run_beagle_vs_jolt_a100.sh — BEAGLE-4.0 (tensor-cores branch) vs JOLT, apples-to-apples, one A100, AA-20 +G4 FP64.
# Phase 2 of the user-requested JOLT-vs-BEAGLE study (phase 1 = build, job 171210531).
#
# The fair race is the per-eval cost of ONE post-order partial-likelihood sweep + root lnL (== JOLT k1_node) and ONE
# gradient sweep (== JOLT kj_pre). We run BOTH tools on the SAME A100 in one job:
#   (1) JOLT standalone kernels (gpu_k1_lnl = lnL, gpu_k7_grad = all-branch gradient) on the real AA-100K alignment +
#       fixed reference tree, model g4 (LG+G4). These print nptn + per-eval ms + lnL (oracle −7541976.9391).
#   (2) BEAGLE bench client at the SAME dims (s=20, nTaxa=100, nPat = JOLT's reported nptn, nCat=4), once on the FP64
#       tensor-core resource (VECTOR_TENSOR) and once on the standard FP64 CUDA-core resource.
# Caveat logged in the report: JOLT runs on the real ML tree (height ~42), BEAGLE bench on a caterpillar (depth ~99) —
# same node count (nTaxa-1 internals) so compute matches; the caterpillar is if anything DEEPER (latency-adverse), so
# this does not flatter BEAGLE. lnL absolute values differ (different Q/data) — parity is checked tensor-vs-cuda within
# BEAGLE and JOLT-vs-its-own-oracle; the comparison is per-eval WALL at matched dimensions.
#PBS -N beagle-vs-jolt-a100
#PBS -P dx61
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=120GB
#PBS -l walltime=01:00:00
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
BENCH=$BEAGLE/bench/beagle_jolt_bench
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
REPS=30
WB=/scratch/rc29/as1708/iqtree3-gpu/beagle_vs_jolt; mkdir -p "$WB"; cd "$WB"

echo "════════ BEAGLE-4.0 vs JOLT — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | tail -2

# ── (0) rebuild JOLT kernels for A100 (sm_80) ──
echo; echo "──── build JOLT kernels (nvcc -O3 -arch=sm_80, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo "$GMF/gpu_k1_lnl.cu"  -o "$WB/gpu_k1_lnl"  && echo "k1 built" || { echo "k1 BUILD FAIL"; exit 1; }
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo "$GMF/gpu_k7_grad.cu" -o "$WB/gpu_k7_grad" && echo "k7 built" || { echo "k7 BUILD FAIL"; exit 1; }

# ── (1) JOLT: lnL (k1) + gradient (k7) on real AA-100K, g4 ──
echo; echo "════════ JOLT k1_node (lnL) — AA-100K LG+G4 ════════"
"$WB/gpu_k1_lnl" "$ALN" "$TREE" g4 $REPS 2>&1 | tee "$WB/jolt_k1.log"
echo; echo "════════ JOLT kj_pre (all-branch gradient) — AA-100K LG+G4 ════════"
"$WB/gpu_k7_grad" "$ALN" "$TREE" g4 $REPS 2>&1 | tee "$WB/jolt_k7.log"

# JOLT uses nptn = nsite (all sites, weight 1, NO pattern compression) and prints "[aln] ntax=.. nsite=..".
# Match BEAGLE to that exact nsite so both tools evaluate the identical pattern count.
NPTN=$(grep -oiE "nsite=[0-9]+" "$WB/jolt_k1.log" | head -1 | grep -oE "[0-9]+")
[ -z "$NPTN" ] && { echo "WARN: could not parse nsite from JOLT log; defaulting 100000"; NPTN=100000; }
echo; echo ">>> matched nPatterns (= JOLT nsite, no compression) = $NPTN"

# ── (2) BEAGLE at matched dims: tensor-core resource, then CUDA-core resource ──
for MODE in tensor cuda; do
  echo; echo "════════ BEAGLE-4.0 $MODE — s=20 nTaxa=100 nPat=$NPTN +G4 FP64 ════════"
  "$BENCH" 20 100 "$NPTN" 4 $REPS 5 "$MODE" 2>&1 | tee "$WB/beagle_${MODE}.log"
done

# ── (3) one-screen comparison ──
echo; echo "════════ HEAD-TO-HEAD (per-eval ms, AA-20 +G4 FP64, A100, nPat=$NPTN) ════════"
python3 - "$WB" "$NPTN" <<'PY'
import sys,re,glob,os
wb,nptn=sys.argv[1],sys.argv[2]
def g(f,pat):
    try: m=re.search(pat,open(f).read())
    except FileNotFoundError: return None
    return m.group(1) if m else None
# JOLT k1: look for a per-eval ms (the harness prints something like 'g4 ... 37.8 ms' or 'k1_node ... ms')
k1=open(os.path.join(wb,"jolt_k1.log")).read() if os.path.exists(os.path.join(wb,"jolt_k1.log")) else ""
k7=open(os.path.join(wb,"jolt_k7.log")).read() if os.path.exists(os.path.join(wb,"jolt_k7.log")) else ""
def lastms(txt):
    ms=re.findall(r"([0-9]+\.[0-9]+)\s*ms",txt); return ms[-1] if ms else "?"
jolt_lnl_ms=lastms(k1); jolt_grad_ms=lastms(k7)
jolt_lnl=None
m=re.search(r"(-?7[0-9]{6}\.[0-9]+)",k1);  jolt_lnl=m.group(1) if m else "?"
rows=[]
for mode in ("tensor","cuda"):
    f=os.path.join(wb,f"beagle_{mode}.log")
    lnl=g(f,r"lnL=(-?[0-9.]+)\s+full_eval_ms");
    full=g(f,r"full_eval_ms=([0-9.]+)"); only=g(f,r"lnL_only_ms=([0-9.]+)")
    tens=g(f,r"tensor=(\d)"); res=g(f,r"resource=(\d+)")
    rows.append((mode,tens,res,lnl,full,only))
print(f"{'tool/mode':<18}{'tensor':>7}{'lnL':>20}{'lnL_eval_ms':>14}{'+matrices_ms':>14}")
print(f"{'JOLT k1_node':<18}{'-':>7}{jolt_lnl:>20}{jolt_lnl_ms:>14}{'-':>14}")
for mode,tens,res,lnl,full,only in rows:
    print(f"{'BEAGLE '+mode:<18}{(tens or '?'):>7}{(lnl or '?'):>20}{(only or '?'):>14}{(full or '?'):>14}")
print()
print(f"JOLT kj_pre (all-branch gradient) per-eval ms: {jolt_grad_ms}")
print("NOTE: lnL-eval_ms is the matched kernel (1 post-order sweep + root). JOLT k1 lnL oracle = -7541976.9391.")
print("      BEAGLE lnL differs (generalized-JC Q vs real LG) — parity is tensor-vs-cuda; race is per-eval WALL.")
PY
echo "════════ DONE $(date -Iseconds) ════════"
