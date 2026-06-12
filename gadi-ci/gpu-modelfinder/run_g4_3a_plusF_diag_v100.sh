#!/bin/bash
# run_g4_3a_plusF_diag_v100.sh — Phase G.4.3a +F coverage DIAGNOSTIC.
#
# QUESTION: in G.4.2b, 0/29 +F models reached the JOLT GPU path, yet getNDim()==0 for +F (FREQ_EMPIRICAL;
# modelmarkov.cpp:964-976 — only FREQ_ESTIMATE/+FO adds num_states-1). By every documented eligibility gate,
# LG+F+G4 SHOULD be JOLT-eligible. Two candidate mechanisms:
#   (a) staged-search / different dispatch -> +F never reaches PhyloTree::optimizeParametersJOLT, OR
#   (b) +F reaches the hook but a specific gate declines it.
# DIAGNOSTIC: JOLT_DEBUG=1 makes optimizeParametersJOLT log a "[JOLT-GATE] reached hook ..." line (with
# ns/rev/nmix/ssm/ndim/pinv/ncat/alpha) for EVERY candidate that arrives, plus a "decline reason=..." line.
#   -> +F models print [JOLT-GATE] lines  => mechanism (b); the reason names the gate.
#   -> +F models print NO line            => mechanism (a); they bypass the hook (staged search).
# Tiny & fast: -mset LG restricts to the 8 LG-family candidates {LG, +I, +G4, +I+G4} x {model-freq, +F};
# -te <fixed tree> skips tree search. CPU-only edit (phylotreegpu.cpp) => g++ recompile + relink, no nvcc.
#
#PBS -N g4-3a-diag
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
module load cmake/3.24.2 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cuda/12.5.1  2>/dev/null || true
module load eigen/3.3.7  2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB="$SRC/g4_3a_diag"; mkdir -p "$WB"; NT=1   # single-thread: clean non-interleaved [JOLT-GATE] stderr; 8 LG models on a fixed tree is fast

echo "════════ G.4.3a +F DIAGNOSTIC — $(hostname) $(date -Iseconds) ════════"
echo "src $(cd "$SRC" && git branch --show-current) HEAD $(cd "$SRC" && git rev-parse --short HEAD)"

echo; echo "──── incremental rebuild (phylotreegpu.cpp gate logging, CPU file -> g++ + relink) ────"
[ -f "$BUILD_ON/Makefile" ] || { echo "FATAL: no configured build dir"; exit 1; }
cd "$BUILD_ON"; make -j4 > make_g43a.log 2>&1; RC=$?
echo "  make exit=$RC (last 6 lines:)"; tail -6 make_g43a.log | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "  binary: $(ls -la "$BIN" | awk '{print $6,$7,$8}')"

echo; echo "════════ DIAGNOSTIC run: --jolt -m TESTONLY -mset LG -te <fixed> -nt $NT (JOLT_DEBUG=1) ════════"
export JOLT_DEBUG=1
T0=$(date +%s)
"$BIN" --jolt -s "$ALN" -m TESTONLY -mset LG -te "$TREE" -nt $NT -pre "$WB/diag" -redo \
    > "$WB/run.stdout" 2> "$WB/run.stderr"
RC=$?; T1=$(date +%s)
echo "[run exit] $RC   [wall] $((T1-T0)) s"

echo; echo "──── ALL [JOLT-GATE] lines (the answer: which +F models reached the hook + decline reason) ────"
grep -E '^\[JOLT-GATE\]' "$WB/run.stderr" 2>/dev/null
echo; echo "──── [JOLT] engagements (models that RAN on GPU) ────"
grep -E '^\[JOLT\]' "$WB/run.stdout" 2>/dev/null
echo; echo "──── candidate models ModelFinder actually evaluated (from .iqtree) ────"
grep -iE "Best-fit|according to BIC" "$WB/diag.iqtree" 2>/dev/null | head -3
echo
echo "──── VERDICT HINT ────"
NF_HOOK=$(grep -E '^\[JOLT-GATE\] reached hook' "$WB/run.stderr" 2>/dev/null | grep -c 'freqtype=3')
echo "  +F (freqtype=3 EMPIRICAL) candidates that REACHED the hook: $NF_HOOK"
echo "  (>0 => mechanism (b) gate-decline, see decline reason; ==0 => mechanism (a) +F bypasses the hook entirely)"
echo "  freqtype legend: 1=USER_DEFINED(model freq) 3=EMPIRICAL(+F) 4=ESTIMATE(+FO)"
echo; echo "════════ DONE $(date -Iseconds) ════════"
