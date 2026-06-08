#!/bin/bash
# run_g2_1a_derv_v100.sh — Phase G.2.1a: clean-room single-edge DERIVATIVE cross-check.
#
# De-risks the new G.2.1 math (arbitrary-edge directed partials via two sub-root sweeps + df/ddf) BEFORE
# wiring the Derv/FromBuffer pointers (G.2.1b). A read-only one-shot (gpuDervCrossCheckOnce) picks an
# internal-internal edge (R = internal node adjacent to the root leaf, C = an internal neighbour of R),
# computes GPU df/ddf clean-room (gpuComputeEdgeDervCleanRoom -> gpu_derv_crosscheck), and compares to
# IQ-TREE's OWN computeLikelihoodDerv (CPU pointer — not yet overridden). GATE: GPU df/ddf == CPU df/ddf
# rel <= 1e-9 (the sign/convention must match the RETURNED df/ddf, not just FD of lnL).
#
# Run under -blfix so it's fast AND also re-exercises the G.2.0b lnL seam + GPU-XCHECK (regression). Pure
# additive; the Derv pointer is still CPU at G.2.1a, so the actual run is unchanged.
#
#PBS -N g2-1a-derv
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:30:00
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
RUNDIR=$SRC/g2_1a_runs; mkdir -p "$RUNDIR"

echo "════════ G.2.1a build+derv-xcheck — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"

echo "──── incremental reconfigure + build ────"
cd "$BUILD" || { echo "no build dir"; exit 1; }
cmake . >/tmp/g2_1a_cmake.log 2>&1; RC=$?; tail -3 /tmp/g2_1a_cmake.log; [ $RC -ne 0 ] && { echo "CMAKE FAILED"; cat /tmp/g2_1a_cmake.log; exit 1; }
make -j12 iqtree3 2>/tmp/g2_1a_make.log; RC=$?
echo "make exit=$RC"
if [ $RC -ne 0 ]; then echo "==== BUILD FAILED (last 80 lines) ===="; tail -80 /tmp/g2_1a_make.log; exit 1; fi
echo "built: $(ls -la "$BUILD/iqtree3" | awk '{print $5, $6, $7, $8}')"

BIN="$BUILD/iqtree3"
echo; echo "════════ RUN --gpu -te -m LG+G4 -blfix (fires lnL + derivative cross-checks) ════════"
"$BIN" --gpu -s "$ALN" -te "$TREE" -m LG+G4 -nt 1 -blfix -pre "$RUNDIR/derv_gpu" -redo 2>&1 | \
  grep -E "GPU-KERNEL|GPU-BRANCH|GPU-XCHECK|GPU-DERV-XCHECK|Log-likelihood of the tree" | head -30

echo
echo "════════ DONE $(date -Iseconds) ════════"
