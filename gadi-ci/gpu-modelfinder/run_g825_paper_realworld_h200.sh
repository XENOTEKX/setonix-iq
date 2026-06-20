#!/bin/bash
# run_g825_paper_realworld_h200.sh — REPRODUCE Williamson et al. (Nature 2025) on their OWN published ML trees,
# and add the Anae- dataset as a second real-world check. Both use the .fasta alignments (full-name labels that
# match the published trees exactly — verified seqs==leaves==match, so NO relabeling needed).
#   (A) Anae+ : LG+MEOW80+G4 -mwopt --jolt -te MEOW6020_fulldataset.treefile  -s CAT_100S93F.fasta (100x22,462)
#       => the lnL of the paper's PUBLISHED primary topology under our GPU implementation.
#   (B) Anae- : LG+MEOW80+G4 -mwopt --jolt -te MEOW6020_noAnaeramoeba.treefile -s CAT_98S93F.fasta (98 taxa)
# Gates each: [JOLTMIX] weights=EM engages (not CPU fallback); [JOLTMIX-RATE1] rho==1 in-convention (no rescale);
# write-back coherence GPU lnL==CPU rel<=1e-6; auto-nTile fits; finite lnL. (Binary has the G.8.2.5 tiling + rate-1 guard.)
#
# Submit: qsub gadi-ci/gpu-modelfinder/run_g825_paper_realworld_h200.sh
#PBS -N g825paper-h200
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=180GB
#PBS -l walltime=02:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 2>/dev/null || true; module load gcc/12.2.0 2>/dev/null || true
module load cuda/12.5.1 2>/dev/null || true; module load eigen/3.3.7 2>/dev/null || true; module load boost/1.84.0 2>/dev/null || true
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
D=/scratch/rc29/as1708/eukaryote_williamson2025; DD="$D/anae_minus"; NEX="$D/MEOW6020.nex"
WB="$SRC/g825_paper_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"

echo "════ G.8.2.5 reproduce-the-paper (Anae+ published tree) + Anae- — $(hostname) $(date -Iseconds) ════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "──── rebuild on-node ────"; ( cd "$BUILD_ON" && make -j12 iqtree3 > "$WB/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; [ $RC -ne 0 ] && { tail -3 "$BUILD_ON/make.log"; echo BUILD-FAIL; exit 1; }
echo "BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/gpu.log" 2>&1 & MON=$!

run() {
  local TAG="$1" ALN="$2" TREE="$3"
  echo "──────── $TAG  aln=$(basename "$ALN")  tree=$(basename "$TREE") ────────"
  [ -s "$ALN" ] && [ -s "$TREE" ] || { echo "  MISSING input"; return; }
  local t0=$(date +%s)
  JOLT_MIX_HOSTDRIVEN=1 JOLT_DEBUG=1 "$BIN" --jolt -te "$TREE" -s "$ALN" -mdef "$NEX" -m LG+ESmodel+G4 -mwopt \
     -nt 12 -pre "$WB/${TAG}" -redo > "$WB/${TAG}.console" 2>&1
  echo "  exit=$?  wall=$(( $(date +%s) - t0 ))s"
  grep -aE '\[MIX-TILE\].*nTile|\[JOLTMIX-RATE1\]' "$WB/${TAG}.console" | head -3 | sed 's/^/  /'
  grep -aE '\[JOLTMIX\] model=' "$WB/${TAG}.console" | tail -1 | sed 's/^/  /' || echo "  (no JOLTMIX — CPU fallback?)"
  grep -aE 'Log-likelihood of the tree|Total tree length' "$WB/${TAG}.iqtree" 2>/dev/null | sed 's/^/  /'
}

run "AnaePlus_papertree"  "$DD/CAT_100S93F.fasta" "$DD/MEOW6020_fulldataset.treefile"
run "AnaeMinus_papertree" "$DD/CAT_98S93F.fasta"  "$DD/MEOW6020_noAnaeramoeba.treefile"
kill $MON 2>/dev/null
echo "──── GPU peak ────"; awk -F, 'NF>=2{if($1+0>mx)mx=$1+0;s+=$2;n++}END{if(n)printf "  peak=%d MiB mean util=%.0f%%\n",mx,s/n}' "$WB/gpu.log"
echo "════ DONE $(date -Iseconds) ════"
