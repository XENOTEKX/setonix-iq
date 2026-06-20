#!/bin/bash
# run_g825_scale_gpu.sh — GPU side of the scaling sweep: does the 1-H200-vs-CPU advantage GROW with alignment length?
# Fits LG+MEOW80+G4 -mwopt --jolt -te on an AliSim alignment (simulated under the same model on the paper's tree).
# Pass via qsub -v: ALN=<fasta> LBL=<tag>. auto-nTile tiling keeps VRAM bounded => scales where the CPU node OOMs.
# Submit: qsub -v ALN=/scratch/.../sim_meow80_100000.fa,LBL=100k gadi-ci/gpu-modelfinder/run_g825_scale_gpu.sh
#PBS -N g825scale-gpu
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=180GB
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 2>/dev/null||true; module load gcc/12.2.0 2>/dev/null||true; module load cuda/12.5.1 2>/dev/null||true
module load eigen/3.3.7 2>/dev/null||true; module load boost/1.84.0 2>/dev/null||true
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
TREE=/scratch/rc29/as1708/eukaryote_williamson2025/anae_minus/MEOW6020_fulldataset.treefile
NEX=/scratch/rc29/as1708/eukaryote_williamson2025/MEOW6020.nex
: "${ALN:?set -v ALN=}"; : "${LBL:?set -v LBL=}"
WB="$SRC/g825_scale_gpu_${LBL}_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════ GPU SCALE FIT  LBL=$LBL  $(hostname) $(date -Iseconds) ════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "aln: $ALN ($(grep -c '^>' "$ALN") seqs)"
( cd "$SRC/build-gpu-on" && make -j12 iqtree3 > "$WB/make.log" 2>&1 ); echo "make exit=$? md5=$(md5sum "$BIN"|cut -c1-12)"
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/gpu.log" 2>&1 & MON=$!
t0=$(date +%s)
JOLT_MIX_HOSTDRIVEN=1 JOLT_DEBUG=1 "$BIN" --jolt -te "$TREE" -s "$ALN" -mdef "$NEX" -m LG+ESmodel+G4 -mwopt \
   -nt 12 -pre "$WB/fit" -redo > "$WB/fit.console" 2>&1
echo "  exit=$?  GPU_WALL=$(( $(date +%s) - t0 ))s"
kill $MON 2>/dev/null
grep -aE '\[MIX-TILE\].*-> nTile|\[JOLTMIX-RATE1\]' "$WB/fit.console" | head -3 | sed 's/^/  /'
grep -aE '\[JOLTMIX\] model=' "$WB/fit.console" | tail -1 | grep -oE 'weights=EM: [0-9]+ iters \| GPU lnL=[-0-9.]+  CPU lnL=[-0-9.]+  rel=[0-9.e-]+ (OK|MISMATCH)' | sed 's/^/  /'
grep -aE 'Log-likelihood of the tree' "$WB/fit.iqtree" 2>/dev/null | sed 's/^/  /'
awk -F, 'NF>=2{if($1+0>mx)mx=$1+0}END{printf "  GPU peak=%d MiB\n",mx}' "$WB/gpu.log" 2>/dev/null
echo "════ DONE $(date -Iseconds) ════"
