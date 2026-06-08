#!/bin/bash
# run_g2_0b_seam_v100.sh — Phase G.2.0b: wire computeLikelihoodBranchPointer -> GPU at the setLikelihoodKernel
# funnel, validate lnL-only under -blfix (the verified coherent regime: branch-length NR is unreachable, so
# Derv/FromBuffer never fire; only the Branch pointer is exercised; the GPU override mirrors _pattern_lh so the
# unconditional computeLogLVariance produces the correct s.e.).
#
# Incrementally rebuilds the GPU-ON iqtree3, then runs the SAME binary twice on AA-100K with a FIXED user tree
# and FIXED branch lengths (-te TREE -blfix), once with --gpu (GPU Branch pointer) and once without (CPU). The
# only difference is which kernel computes the branch lnL. GATES:
#   (1) [GPU-KERNEL] install marker prints (the funnel hook fired)
#   (2) [GPU-BRANCH] active marker prints (the GPU pointer actually computed at least one branch lnL)
#   (3) [GPU-XCHECK] in-process self-check PASS (GPU sweep == independent CPU recompute, rel<=1e-6)
#   (4) GPU final lnL + s.e. == CPU final lnL + s.e. (rel <= 1e-12) — same fixed tree, two kernels
#   (5) CPU run (no --gpu) unperturbed: the GPU override is gated on params->gpu, so the CPU path is unchanged.
#
#PBS -N g2-0b-seam
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:40:00
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
RUNDIR=$SRC/g2_0b_runs; mkdir -p "$RUNDIR"

echo "════════ G.2.0b build+seam — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"

echo "──── incremental reconfigure + build ────"
cd "$BUILD" || { echo "no build dir"; exit 1; }
cmake . >/tmp/g2_0b_cmake.log 2>&1; RC=$?; tail -3 /tmp/g2_0b_cmake.log; [ $RC -ne 0 ] && { echo "CMAKE FAILED"; cat /tmp/g2_0b_cmake.log; exit 1; }
make -j12 iqtree3 2>/tmp/g2_0b_make.log; RC=$?
echo "make exit=$RC"
if [ $RC -ne 0 ]; then echo "==== BUILD FAILED (last 80 lines) ===="; tail -80 /tmp/g2_0b_make.log; exit 1; fi
echo "built: $(ls -la "$BUILD/iqtree3" | awk '{print $5, $6, $7, $8}')"

BIN="$BUILD/iqtree3"
echo; echo "════════ RUN --gpu -te -m LG+G4 -blfix (GPU Branch pointer; fires install + active + self-check) ════════"
"$BIN" --gpu -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -blfix -pre "$RUNDIR/seam_gpu" -redo 2>&1 | \
  grep -E "GPU-KERNEL|GPU-BRANCH|GPU-XCHECK|BEST SCORE|Log-likelihood of the tree|Optimal log-likelihood" | head -40
echo "----- GPU final lnL line from the .iqtree report -----"
grep -E "Log-likelihood of the tree|BEST SCORE|Optimal log-likelihood" "$RUNDIR/seam_gpu.iqtree" 2>/dev/null | head -5

echo; echo "════════ RUN CPU (no --gpu) -te -m LG+G4 -blfix: confirm same lnL + s.e., CPU path unperturbed ════════"
"$BIN" -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -blfix -pre "$RUNDIR/seam_cpu" -redo 2>&1 | \
  grep -E "GPU-KERNEL|GPU-BRANCH|GPU-XCHECK|Optimal log-likelihood|BEST SCORE|Log-likelihood of the tree" | head -10
echo "----- CPU final lnL line from the .iqtree report -----"
grep -E "Log-likelihood of the tree|BEST SCORE|Optimal log-likelihood" "$RUNDIR/seam_cpu.iqtree" 2>/dev/null | head -5

echo; echo "════════ COMPARE (GPU vs CPU, same fixed tree) ════════"
G=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/seam_gpu.iqtree" 2>/dev/null | head -1 | grep -oE "\-?[0-9.]+$")
C=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$RUNDIR/seam_cpu.iqtree" 2>/dev/null | head -1 | grep -oE "\-?[0-9.]+$")
echo "GPU lnL = ${G:-NA}   CPU lnL = ${C:-NA}"
if [ -n "${G:-}" ] && [ -n "${C:-}" ]; then
  python3 - "$G" "$C" <<'PY'
import sys
g=float(sys.argv[1]); c=float(sys.argv[2])
rel=abs((g-c)/c) if c!=0 else abs(g-c)
print(f"|d|={abs(g-c):.4e}  rel={rel:.3e}  -> {'PASS (rel<=1e-12)' if rel<=1e-12 else 'CHECK'}")
PY
fi
echo "----- s.e. lines (must match GPU vs CPU) -----"
grep -hE "Log-likelihood of the tree" "$RUNDIR/seam_gpu.iqtree" "$RUNDIR/seam_cpu.iqtree" 2>/dev/null

echo
echo "════════ DONE $(date -Iseconds) ════════"
