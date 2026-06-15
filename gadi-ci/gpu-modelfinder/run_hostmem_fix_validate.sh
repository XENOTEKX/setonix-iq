#!/bin/bash
# run_hostmem_fix_validate.sh — validate the cgroup-aware host-memory fix (tools.cpp getCgroupMemoryLimit +
# phyloanalysis --jolt lean LM_MEM_SAVE tier). Reruns the EXACT AA-10M --jolt refine that OOM'd (job 170856902),
# reusing its coarse tree, on the SAME 180GB allocation. Plus a 1M regression (must stay LM_PER_NODE, rel ~1e-12).
# Gates: (A) 1M: NO "[--jolt] host LM_PER_NODE" NOTE (stays per-node), self-check rel <=1e-9, winner LG+G4.
#        (B) 10M: cgroup NOTE prints, "[--jolt] ... memory-saving" NOTE fires, JOLT ENGAGES (calls>0, not OOM),
#            self-check rel <=1e-9, exit 0 (no cgroup kill). This is the host-OOM fix proof.
# Submit: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=180GB -l walltime=02:00:00 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd run_hostmem_fix_validate.sh
#PBS -N hostmemfix
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN1M=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
ALN10M=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_10000000/tree_1/alignment_10000000.phy
TREE1M=$SRC/ctf1mmf_h200mf3/coarse.treefile
TREE10M=$SRC/ctf10mmf_aa10m_h200/coarse.treefile
WB="$SRC/hostmemfix_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ HOST-MEM FIX VALIDATE — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "BIN md5=$(md5sum "$BIN"|cut -d' ' -f1); physical=$(awk '/MemTotal/{printf "%.0f GB",$2/1048576}' /proc/meminfo); cgroup_alloc=180GB"

run() {  # $1=label $2=aln $3=tree
  local L="$1" A="$2" T="$3"
  echo; echo "──── $L: --jolt -m LG+G4 -te ($(basename $(dirname $A))) ────"
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/${L}_gpu.log" 2>&1 & SMI=$!
  /usr/bin/time -v env JOLT_DEBUG=1 "$BIN" --jolt --gpu -m LG+G4 -s "$A" -te "$T" -nt "$PBS_NCPUS" -pre "$WB/$L" -redo > "$WB/$L.out" 2>&1
  local rc=$?; kill $SMI 2>/dev/null
  echo "  exit=$rc"
  grep -hE "cgroup memory limit|\[--jolt\] host LM_PER_NODE|Switching to memory saving|NOTE: Switching" "$WB/$L.out" | sed 's/^/    /'
  grep -hE "\[JOLT\] model=" "$WB/$L.out" | tail -1 | sed 's/^/    /'
  grep -hiE "exceeded memory|Killed|bad_alloc|cannot work, switch" "$WB/$L.out" | head -2 | sed 's/^/    !! /'
  local pk=$(sort -n "$WB/${L}_gpu.log" 2>/dev/null | tail -1)
  local maxrss=$(grep -iE "Maximum resident set size" "$WB/$L.out" | grep -oE '[0-9]+' | head -1)
  echo "    peak GPU(MiB,util)=${pk:-NA}  host_maxRSS=$(python3 -c "print(f'{${maxrss:-0}/1048576:.1f} GB')" 2>/dev/null)"
}

run 1M  "$ALN1M"  "$TREE1M"      # regression: expect LM_PER_NODE (no --jolt mem-save NOTE), rel ~1e-12
run 10M "$ALN10M" "$TREE10M"     # the fix: expect cgroup+mem-save NOTEs, JOLT engages, no OOM, rel ~1e-12

echo; echo "════════ DONE $(date -Iseconds) ════════"
