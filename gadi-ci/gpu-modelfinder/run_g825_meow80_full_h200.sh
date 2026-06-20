#!/bin/bash
# run_g825_meow80_full_h200.sh — G.8.2.5 HEADLINE: the Williamson et al. (Nature 2025) primary model
# LG+MEOW80+G4 fit on the FULL eukaryote alignment (100 taxa x 22,462 sites) on ONE H200, via pattern tiling.
#
# WHY THIS NEEDS TILING: the all-branch gradient's resident partial arenas are nInternal*R*ns*nptn (postorder,
# ~102 GB) + nPool*R*ns*nptn (preorder pool, ~43 GB) = ~149 GB for MEOW80+G4 (R=320) at nptn~20k. That OOMs even
# the 141 GB H200 one-shot (the prior attempt D_gpu_meow80 fell back / never completed the GPU derivative). The
# G.8.2.5 launchers chunk nptn into nTile pieces (auto-picked from cudaMemGetInfo, 80% budget) and accumulate each
# edge's df/ddf with a CONTINUOUS per-edge Kahan sweep => BIT-IDENTICAL to one-shot. Expect auto nTile=2 on H200
# (~75 GB/chunk). Composes with the G.7.0 cgroup host-mem fix (host self-check runs under the lean LM_MEM_SAVE tier).
#
# GATES (capability headline):
#   (1) ENGAGES on GPU: a [JOLTMIX] line with weights=EM (NOT a CPU fallback), exit 0, high GPU util.
#   (2) auto-nTile fired and FIT: [MIX-TILE] nTile>=2, no DEVB/cudaMalloc NaN, no OOM.
#   (3) WRITE-BACK COHERENCE at full scale: in-process GPU lnL == fresh CPU computeLikelihood, rel <= 1e-6.
#   (4) lnL finite & sensible for 22,462 sites.
# NOTE (branch-length scale): per G.8.2.4 the rate-1 rescale write-back is a SEPARATE gated follow-up; lnL and model
# SELECTION are exact (rescale is lnL-invariant), but the written branch lengths are in the live-tns scale until then.
#
# Submit: qsub gadi-ci/gpu-modelfinder/run_g825_meow80_full_h200.sh
#PBS -N g825meow-h200
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=180GB
#PBS -l walltime=03:00:00
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
ALN=/scratch/rc29/as1708/eukaryote_williamson2025/CAT_100S93F.phy     # 100 taxa x 22,462 sites (full)
NEX=/scratch/rc29/as1708/eukaryote_williamson2025/MEOW6020.nex
TREE=$SRC/euk_will2025_run/A_fasttree.treefile                         # fixed -te starting tree
WB="$SRC/g825_meow_full_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
[ -s "$ALN" ] || { echo "FATAL: $ALN missing"; exit 1; }
[ -s "$NEX" ] || { echo "FATAL: $NEX missing"; exit 1; }
[ -s "$TREE" ] || { echo "FATAL: $TREE missing"; exit 1; }

echo "════════ G.8.2.5 HEADLINE — LG+MEOW80+G4 full-data --jolt on H200 — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "alignment: $(head -1 "$ALN")"
echo "──── rebuild on-node (binary must match source) ────"
( cd "$BUILD_ON" && make -j12 iqtree3 > "$WB/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; tail -3 "$BUILD_ON/make.log" | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
free -g | awk '/Mem:/{print "  host RAM total="$2" GB"}'

# GPU monitor (used-mem peak + utilisation)
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/gpu.log" 2>&1 &
MON=$!

echo "──── MEOW80 --jolt full-data (auto-nTile; JOLT_DEBUG prints the tiling decision) ────"
t0=$(date +%s)
JOLT_MIX_HOSTDRIVEN=1 JOLT_DEBUG=1 ALLDERV_DBG=1 \
  "$BIN" --jolt -te "$TREE" -s "$ALN" -mdef "$NEX" -m LG+ESmodel+G4 -mwopt -nt 12 -pre "$WB/meow_full" -redo \
  > "$WB/meow_full.console" 2>&1
RC=$?
kill $MON 2>/dev/null
echo "  exit=$RC  wall=$(( $(date +%s) - t0 ))s"

echo "──── tiling decision ────"
grep -aE '\[MIX-TILE\]' "$WB/meow_full.console" | grep -aE 'nTile|perPtn' | head -3 | sed 's/^/  /'
grep -aE '\[ALLDERV-DBG\] tiled proc done' "$WB/meow_full.console" | head -1 | sed 's/^/  /'
echo "──── JOLTMIX engage + write-back coherence ────"
grep -aE '\[JOLTMIX\]|\[JOLTMIX-GATE\]' "$WB/meow_full.console" | sed 's/^/  /' || echo "  (no JOLTMIX line — DID IT FALL TO CPU? check console)"
echo "──── final lnL ────"
grep -aE 'Log-likelihood of the tree' "$WB/meow_full.iqtree" 2>/dev/null | sed 's/^/  /' || echo "  (.iqtree missing)"
echo "──── GPU peak ────"
awk -F, 'NF>=2{if($1+0>mx)mx=$1+0; s+=$2; n++} END{if(n)printf "  peak mem=%d MiB  mean util=%.0f%%  samples=%d\n",mx,s/n,n}' "$WB/gpu.log" 2>/dev/null
echo "──── any CUDA/OOM errors ────"
grep -aiE 'out of memory|cuda error|DEVB|NaN|cudaMalloc|fall' "$WB/meow_full.console" | head -8 | sed 's/^/  /' || echo "  (none)"
echo "════════ DONE $(date -Iseconds) ════════"
