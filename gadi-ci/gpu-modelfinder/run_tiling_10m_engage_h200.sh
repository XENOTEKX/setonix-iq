#!/bin/bash
# run_tiling_10m_engage_h200.sh — G.7.1 PATTERN TILING, stage V.C (the capability headline).
# THE TEST: run AA-10M --jolt -te on ONE H200 with JOLT actually ENGAGING on the GPU. Without tiling the 886 GB
# one-shot arena OOMs the H200 (141 GB) -> DEVB NaN -> CPU fallback (GPU util 0%, no [JOLT] line — measured in
# job 170934922). With auto-T tiling (~8 chunks => ~111 GB) JOLT engages: expect a [JOLT] PASS line, high GPU
# util, lnL parity vs the host self-check, exit 0. Composes with the G.7.0 host-mem fix (c04a9ce1): the host
# self-check runs under the lean LM_MEM_SAVE tier (~50 GB), so the box does not OOM either.
#
#PBS -N tile10m-h200
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
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_10000000/tree_1/alignment_10000000.phy
TREE=$SRC/ctf10mmf_aa10m_h200/coarse.treefile
WB="$SRC/tile10m_h200_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"

echo "════════ G.7.1 V.C  AA-10M --jolt -te ENGAGE on H200 — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "──── rebuild on-node ────"
( cd "$BUILD_ON" && make -j12 iqtree3 > "$WB/make.log" 2>&1 ); RC=$?
echo "  make exit=$RC"; tail -3 "$BUILD_ON/make.log" | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
echo "BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
free -g | awk '/Mem:/{print "host RAM total="$2" GB"}'
[ -f "$TREE" ] || { echo "MISSING coarse tree $TREE"; exit 1; }

( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/gpu.log" 2>&1 &
POLL=$!
# auto-T (no JOLT_NTILE) — the binary picks nTile from cudaMemGetInfo; JOLT_DEBUG prints the tiling decision.
JOLT_DEBUG=1 /usr/bin/time -v "$BIN" -s "$ALN" -m LG+G4 -te "$TREE" --jolt --gpu -nt 1 -pre "$WB/aa10m" \
    > "$WB/run.log" 2>&1
RC=$?
kill $POLL 2>/dev/null
PEAK=$(awk -F, 'NR>0{gsub(/ /,"",$1); if($1+0>m)m=$1+0} END{print m}' "$WB/gpu.log")
UTILMAX=$(awk -F, 'NR>0{gsub(/ /,"",$2); if($2+0>u)u=$2+0} END{print u}' "$WB/gpu.log")
RSS=$(awk '/Maximum resident set size/{printf "%.1f", $6/1048576}' "$WB/run.log")

echo; echo "──── RESULT ────"
echo "  exit=$RC  peakVRAM=${PEAK}MiB  maxGPUutil=${UTILMAX}%  host_maxRSS=${RSS}GB"
grep -E "NOTE: cgroup|NOTE: \[--jolt\] host LM_PER_NODE|nTile|TILING|\[JOLT\]" "$WB/run.log" | sed 's/^/  /'
echo
echo "  GATE: PASS requires -> [JOLT] line present (GPU engaged), maxGPUutil high (not 0), rel PASS, exit 0."
echo "════════ DONE $(date -Iseconds) ════════"
