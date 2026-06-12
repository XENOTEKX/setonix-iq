#!/bin/bash
# run_jolt_full_mtest_100k_v100.sh — FIRST EVER full `-m TEST --jolt` run (ModelFinder + NNI tree search).
#
# PURPOSE (user: "see if JOLT is ready ... benchmark against a full run of IQ-TREE not just modelfinder"):
#   JOLT has only ever run on a FIXED topology (-te / -m TESTONLY). A full `-m TEST` adds the NNI tree-search
#   phase (~50% of whole-run wall, AA-100K) which is NOT hooked by JOLT — it runs on CPU. This run:
#     (1) CORRECTNESS/PARITY in a tree-search context (NEW, never tested): same tree + best model + final lnL
#         as the CPU baseline? The per-JOLT-call self-check (fresh CPU computeLikelihood, rel<=1e-9) is the net.
#     (2) PHASE-DECOMPOSED GPU baseline: MF-phase wall (JOLT, the only phase it touches) vs tree-search wall
#         (CPU) vs total. The binary emits "Wall-clock time for ModelFinder" and "...for tree search" separately.
#
# HONEST EXPECTATION (stated up front, NOT discovered): JOLT will LOSE on wall.
#   * MF phase: JOLT mutex-serializes on ONE GPU + CPU +I/+R tail => ~3493s @nt12 (G.4.2b TESTONLY), vs
#     FCA np1 -m TEST MF-phase 258.8s @103T (gadi_AA_100k_fca_np1_full_seed1_169095077). ~13x slower.
#   * Tree-search phase: pure CPU @ gpuvolta's 12 cores vs FCA's 103 => core-count-bound, not algorithmic.
#   The WIN this maps toward: (a) cross-model GPU batching (grid.z) to break MF serialization, (b) a 2nd hook
#   to put NNI branch-opt (optimizeAllBranches/optimizeOneBranch) on GPU. NEITHER loss = "can't parallelize".
#
# LIKE-FOR-LIKE: default -m TEST search is NNI (tree_spr=false, tools.cpp:7160) — matches FCA's -m TEST (NNI)
# and the CPU full-run baseline 168425673 (default => NNI, total 1169.556s @nt103, lnL -7541976.86, LG+G4).
# seed 1 to match. NOT SPR (don't cite the *_spr_* walls against this NNI run).
#
#PBS -N jolt-full-mtest-100k
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
WB="$SRC/jolt_full_mtest_100k"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
[ -f "$ALN" ] || { echo "no alignment $ALN"; exit 1; }
export JOLT_DEBUG=1   # emit [JOLT] / [JOLT-GATE] lines incl per-call self-check rel

echo "════════ JOLT full -m TEST (MF + NNI tree search) — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "BIN=$BIN  md5=$(md5sum "$BIN" | cut -d' ' -f1)"
echo "ALN=$ALN"
echo "CMD: $BIN --jolt --gpu -s ALN -m TEST -seed 1 -nt 12 -pre mf_jolt_full -redo"
echo

T0=$(date +%s)
"$BIN" --jolt --gpu -s "$ALN" -m TEST -seed 1 -nt 12 -pre "$WB/mf_jolt_full" -redo \
    > "$WB/mf_jolt_full.stdout" 2>&1
RC=$?
WALL=$(( $(date +%s) - T0 ))
echo "exit=$RC  driver_wall=${WALL}s"
echo

echo "════════ PHASE-DECOMPOSED WALL (from the binary's own timing lines) ════════"
grep -iE "Wall-clock time for ModelFinder|Wall-clock time used for tree search|Total wall-clock time used|CPU time for ModelFinder|Time for fast ML tree search" "$WB/mf_jolt_full.log" 2>/dev/null | sed 's/^/  /'
echo
echo "════════ PARITY: best model + final lnL + tree (vs CPU baseline LG+G4 / -7541976.86) ════════"
grep -iE "Best-fit model|BEST SCORE FOUND|Optimal log-likelihood|^BIC|Akaike" "$WB/mf_jolt_full.iqtree" 2>/dev/null | head -8 | sed 's/^/  /'
echo "  --- .iqtree best model line ---"
grep -iE "Best-fit model according to BIC" "$WB/mf_jolt_full.iqtree" 2>/dev/null | sed 's/^/  /'
echo "  --- final lnL (from .log) ---"
grep -iE "BEST SCORE FOUND|Log-likelihood of the tree|Optimal log-likelihood" "$WB/mf_jolt_full.log" 2>/dev/null | tail -4 | sed 's/^/  /'
echo
echo "════════ JOLT ENGAGEMENT + SELF-CHECK (correctness net; any rel>1e-9 or FAIL is a bug) ════════"
echo "  [JOLT] success lines: $(grep -c '\[JOLT\]' "$WB/mf_jolt_full.log" 2>/dev/null) ; gate declines: $(grep -c '\[JOLT-GATE\]\|JOLT_DECLINE\|decline' "$WB/mf_jolt_full.log" 2>/dev/null)"
grep -iE "\[JOLT\].*rel|self-check|GPU-BRANCH|rel=" "$WB/mf_jolt_full.log" 2>/dev/null | tail -15 | sed 's/^/    /'
echo "  worst self-check rel:"; grep -oE "rel[= ]*[0-9.eE+-]+" "$WB/mf_jolt_full.log" 2>/dev/null | grep -oE "[0-9.eE+-]+$" | sort -g | tail -1 | sed 's/^/    /'
echo
echo "════════ REFERENCE NUMBERS (do NOT re-run; from existing JSONs) ════════"
echo "  CPU full-run (NNI) total : 1169.556s @nt103  (job 168425673, lnL -7541976.86, LG+G4)"
echo "  FCA np1 -m TEST MF-phase : 258.773s  @103T   (job 169095077, LG+G4)"
echo "  JOLT MF-phase (TESTONLY) : 3493.5s   @nt12    (G.4.2b job 170367630) -- expect to re-confirm here"
echo "  ==> HEADLINE = MF-phase wall (JOLT vs 258.8s). Tree-search = CPU-vs-CPU footnote (12 vs 103 cores)."
echo "════════ DONE $(date -Iseconds) ════════"
