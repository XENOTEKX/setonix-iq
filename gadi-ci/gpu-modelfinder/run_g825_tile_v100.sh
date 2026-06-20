#!/bin/bash
# run_g825_tile_v100.sh — VALIDATE G.8.2.5 pattern tiling of the mix launchers: chunked (JOLT_NTILE=4) MUST be
# bit-identical to one-shot (JOLT_NTILE=1). The per-pattern values are chunk-independent and the Kahan reductions
# add patterns in order 0..nptn-1, so any nTile yields the SAME answer (only VRAM/launch trade-off).
#   (A1/A4) C20 --gpu kill-switch at NTILE=1 vs 4 — selfTest lnL rel (the lnL-launcher gate) + cold-vs-warm must MATCH.
#   (B4)    C20 --jolt JOLTMix at NTILE=4 — lnL + self-check must equal the untiled -30796.522464 / 5.08e-15.
# (This run covers whichever launchers are tiled so far; the [MIX-TILE] JOLT_DEBUG line confirms nTile fired.)
#
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=60GB -l walltime=00:25:00 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd gadi-ci/gpu-modelfinder/run_g825_tile_v100.sh
#PBS -N g825tile
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
TREE=/scratch/rc29/as1708/iqtree3-gpu/euk_will2025_run/A_fasttree.treefile
ALN=/scratch/rc29/as1708/iqtree3-gpu/g822_mix/euk400.phy
WB=/scratch/rc29/as1708/iqtree3-gpu/g825_tile; mkdir -p "$WB"; cd "$WB"
[ -s "$ALN" ] || { echo "FATAL: $ALN missing"; exit 1; }
M=LG+C20+G4

echo "════ G.8.2.5 tiling bit-parity — $(hostname) $(date -Iseconds) ════"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo "BIN md5: $(md5sum "$BIN" | cut -c1-12)"

for NT in 1 4; do
  echo "── (A) C20 --gpu kill-switch  JOLT_NTILE=$NT  (exercises BOTH lnL + all-branch-derivative tiling) ──"
  JOLT_NTILE=$NT JOLT_DEBUG=1 ALLDERV_DBG=1 "$BIN" --gpu -te "$TREE" -s "$ALN" -m "$M" -nt 12 -pre "$WB/A_nt$NT" -redo > "$WB/A_nt$NT.console" 2>&1
  echo "  exit=$?"
  grep -aE '\[MIX-TILE\]' "$WB/A_nt$NT.console" | head -2 | sed 's/^/  /' || echo "  (no MIX-TILE line)"
  grep -aE '\[ALLDERV-DBG\] tiled proc done' "$WB/A_nt$NT.console" | head -1 | sed 's/^/  /' || echo "  (no ALLDERV tiled line)"
  grep -aE '\[GPU-MIXJOINT-XCHECK\]' "$WB/A_nt$NT.console" | sed 's/^/  /' || echo "  (no kill-switch line)"
done

echo "── (B4) C20 --jolt JOLTMix  JOLT_NTILE=4 (expect lnL -30796.522464, self-check 5.08e-15) ──"
JOLT_MIX_HOSTDRIVEN=1 JOLT_NTILE=4 JOLT_DEBUG=1 "$BIN" --jolt -te "$TREE" -s "$ALN" -m "$M" -nt 12 -pre "$WB/B_nt4" -redo > "$WB/B_nt4.console" 2>&1
echo "  exit=$?"
grep -aE '\[MIX-TILE\]' "$WB/B_nt4.console" | head -2 | sed 's/^/  /' || echo "  (no MIX-TILE line)"
grep -aE '\[JOLTMIX\]' "$WB/B_nt4.console" | sed 's/^/  /' || echo "  (no JOLTMIX line)"
echo "════ DONE $(date -Iseconds) ════"
