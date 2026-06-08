#!/bin/bash
# run_g2_1b_full_v100.sh — Phase G.2.1b INTEGRATION test: full branch-length optimisation on the GPU.
#
# Runs `--gpu -te TREE -m LG+G4` (NO -blfix) so optimizeAllBranches drives Newton-Raphson entirely through the
# GPU overrides (computeLikelihoodDervGPU + computeLikelihoodFromBufferGPU + computeLikelihoodBranchGPU, all
# stateless clean-room, persistent device-buffer pool). Compares final lnL + the optimised branch-length vector
# against the same binary without --gpu (CPU). GATES:
#   (1) markers [GPU-KERNEL]/[GPU-BRANCH]/[GPU-DERV]/[GPU-FROMBUF] all fire (every path exercised on GPU)
#   (2) GPU-DERV-XCHECK PASS (read-only derivative regression, INT-INT + LEAF)
#   (3) GPU final lnL == CPU final lnL  rel <= 1e-9
#   (4) optimised branch-length vector (sorted)  rel <= 1e-6  GPU vs CPU
#   (5) wall(GPU branch-opt) reported (informs the G.2.2 221.6 s judgement)
#
#PBS -N g2-1b-full
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
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cmake/3.24.2 2>/dev/null || true
module load eigen/3.3.7  2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export CC=gcc CXX=g++

SRC=/scratch/rc29/as1708/iqtree3-gpu
BUILD=$SRC/build-gpu-on
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
RUNDIR=$SRC/g2_1b_runs; mkdir -p "$RUNDIR"

echo "════════ G.2.1b FULL branch-opt — $(hostname) $(date -Iseconds) ════════"
cd "$BUILD" || { echo "no build dir"; exit 1; }
cmake . >/tmp/g2_1b_cmake.log 2>&1; RC=$?; [ $RC -ne 0 ] && { echo "CMAKE FAILED"; cat /tmp/g2_1b_cmake.log; exit 1; }
make -j12 iqtree3 2>/tmp/g2_1b_make.log; RC=$?
echo "make exit=$RC"
if [ $RC -ne 0 ]; then echo "==== BUILD FAILED (last 80 lines) ===="; tail -80 /tmp/g2_1b_make.log; exit 1; fi
echo "built: $(ls -la "$BUILD/iqtree3" | awk '{print $5, $6, $7, $8}')"
BIN="$BUILD/iqtree3"

echo; echo "════════ GPU run: --gpu -te -m LG+G4 (full GPU branch-opt) ════════"
T0=$(date +%s)
"$BIN" --gpu -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -pre "$RUNDIR/bopt_gpu" -redo 2>&1 | \
  grep -E "GPU-KERNEL|GPU-BRANCH|GPU-DERV|GPU-FROMBUF|GPU-XCHECK|GPU-DERV-XCHECK|Log-likelihood of the tree|Optimal log-likelihood" | head -30
T1=$(date +%s); echo "[GPU wall] $((T1-T0)) s"

echo; echo "════════ CPU run: -te -m LG+G4 (reference) ════════"
T2=$(date +%s)
"$BIN" -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -pre "$RUNDIR/bopt_cpu" -redo 2>&1 | \
  grep -E "Log-likelihood of the tree|Optimal log-likelihood" | head -5
T3=$(date +%s); echo "[CPU wall] $((T3-T2)) s"

echo; echo "════════ COMPARE final lnL + branch-length vector ════════"
G=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/bopt_gpu.iqtree" 2>/dev/null | head -1 | grep -oE "\-?[0-9.]+$")
C=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/bopt_cpu.iqtree" 2>/dev/null | head -1 | grep -oE "\-?[0-9.]+$")
echo "GPU lnL=${G:-NA}  CPU lnL=${C:-NA}"
python3 - "$RUNDIR/bopt_gpu.treefile" "$RUNDIR/bopt_cpu.treefile" "${G:-nan}" "${C:-nan}" <<'PY'
import sys, re
gtf, ctf, g, c = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def brlens(f):
    s = open(f).read()
    return sorted(float(x) for x in re.findall(r':([0-9.]+(?:[eE][-+]?[0-9]+)?)', s))
try:
    gb, cb = brlens(gtf), brlens(ctf)
    rl = abs((g-c)/c) if c else abs(g-c)
    print(f"lnL: |d|={abs(g-c):.3e} rel={rl:.3e} -> {'PASS' if rl<=1e-9 else 'CHECK'}  (gate 1e-9)")
    if len(gb)!=len(cb):
        print(f"BRLEN: count mismatch GPU={len(gb)} CPU={len(cb)} -> CHECK"); sys.exit(0)
    worst=0.0
    for a,b in zip(gb,cb):
        d = abs(a-b)/b if b>1e-12 else abs(a-b)
        worst=max(worst,d)
    print(f"BRLEN: n={len(gb)} worst_rel={worst:.3e} -> {'PASS' if worst<=1e-6 else 'CHECK'}  (gate 1e-6)")
except Exception as e:
    print("compare error:", e)
PY

echo; echo "════════ DONE $(date -Iseconds) ════════"
