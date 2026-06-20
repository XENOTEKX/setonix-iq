#!/bin/bash
# run_g825_scale_cpu.sh — CPU side of the scaling sweep (full node, -nt 104 + numactl interleave). Same fit as the
# GPU side. The MEOW80 likelihood arena scales ~linearly with patterns (~105 GB @ 22k -> ~467 GB @ 100k -> ~935 GB
# @ 200k), so a single 503 GB node OOMs past ~110k sites => the GPU (tiling, bounded VRAM) runs where the CPU node cannot.
# Pass via qsub -v: ALN=<fasta> LBL=<tag>.
# Submit: qsub -v ALN=/scratch/.../sim_meow80_100000.fa,LBL=100k gadi-ci/gpu-modelfinder/run_g825_scale_cpu.sh
#PBS -N g825scale-cpu
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=490GB
#PBS -l walltime=08:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null||true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
TREE=/scratch/rc29/as1708/eukaryote_williamson2025/anae_minus/MEOW6020_fulldataset.treefile
NEX=/scratch/rc29/as1708/eukaryote_williamson2025/MEOW6020.nex
: "${ALN:?set -v ALN=}"; : "${LBL:?set -v LBL=}"
WB="$SRC/g825_scale_cpu_${LBL}_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════ CPU SCALE FIT  LBL=$LBL  $(hostname) $(date -Iseconds) ════"
echo "aln: $ALN ($(grep -c '^>' "$ALN") seqs)  cores: $(nproc)"; free -g | awk '/Mem:/{print "  host RAM="$2" GB"}'
NUMA="numactl --interleave=all"; command -v numactl >/dev/null || NUMA=""
t0=$(date +%s)
$NUMA "$BIN" -te "$TREE" -s "$ALN" -mdef "$NEX" -m LG+ESmodel+G4 -mwopt -nt 104 -pre "$WB/fit" -redo \
   > "$WB/fit.console" 2>&1
RC=$?
echo "  exit=$RC  CPU_WALL=$(( $(date +%s) - t0 ))s"
grep -aiE 'out of memory|std::bad_alloc|cannot allocate|Killed|ERROR.*memory|required!' "$WB/fit.console" | head -3 | sed 's/^/  /'
grep -aE 'Log-likelihood of the tree' "$WB/fit.iqtree" 2>/dev/null | sed 's/^/  /' || echo "  (no lnL — OOM / did not finish?)"
echo "════ DONE $(date -Iseconds) ════"
