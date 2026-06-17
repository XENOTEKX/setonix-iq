#!/bin/bash
# run_g5_1b_jolt_freerate_v100.sh — Phase G.5.1b increment-2a: build + validate the STANDALONE +R (FreeRate) JOLT
# CONVERGENCE harness (gpu_k8c_jolt_freerate.cu). This is the make-or-break correctness gate that MUST pass before
# any in-tree +R eligibility flip (the +R gradient is already FD-validated; the OPEN risk is OPTIMISER convergence
# on the multimodal +R surface — IQ-TREE uses EM; a naive diagonal-LM once stalled on +I).
#
# Steps:
#   1. build gpu_k8c_jolt_freerate.cu (nvcc -O3 -arch=sm_70, FP64 only).
#   2. CPU-EM REFERENCE on the SAME FIXED TOPOLOGY (-te $TREE) for LG+R4 and LG+R6 (IQ-TREE's RateFree EM, the gold
#      standard); extract O (lnL) + per-category (weight,rate) into cpu_rK_ref.txt. LG model freqs (no +F) to match
#      the harness's fill_LG stationary freqs.
#   3. run the harness for r4 and r6: WARM (cpu-em seed) + COLD (spread rates/uniform weights) joint diagonal-LM,
#      gauge-fixed each accept; gate cold.lnL == warm.lnL == O (rel<=1e-9) + gz/gy FD + WN identity.
#
# Gate (BOTH r4 and r6): VERDICT PASS. A CHECK/FAIL means the diagonal-LM does not reach the CPU-EM MLE -> the
# in-tree +R increment must carry the multi-start / EM-warm-start fallback (IX.3 #1) — do NOT flip the gate.
#
#PBS -N g5-1b-freerate-v100
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load gcc/12.2.0  2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"   # for the CUDA-linked GPU-fork binary

SRC=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k8c_jolt_freerate.cu
BIN=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder/gpu_k8c_jolt_freerate
# CPU-EM reference: use the GPU FORK's own binary on the PURE CPU path (no --jolt/--gpu = upstream-identical RateFree
# EM). The MPI fork (iqtree3-mpi) is -march=sapphirerapids and SEGFAULTS on the gpuvolta node's non-SPR CPU (the
# ref_spr_binary_login_sigill gotcha); build-gpu-on/iqtree3 already runs on gpuvolta (it ran the whole CTF sweep there)
# and needs no mpirun (avoids the MPI_FINALIZE-multiple-times crash). Same binary as the harness's data, single device.
CPUBIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/g5_1b_freerate; mkdir -p "$WB"; cd "$WB"

echo "════════ G.5.1b +R JOLT convergence harness — $(hostname) $(date -Iseconds) ════════"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]    $([ -f "$ALN" ] && echo OK || echo MISSING)"
echo "[tree]   $([ -f "$TREE" ] && echo OK || echo MISSING)"
echo "[cpubin] $([ -x "$CPUBIN" ] && echo OK || echo MISSING)"

echo "──── build (nvcc -O3 -arch=sm_70, precise FP64) ────"
nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo "$SRC" -o "$BIN"
RC=$?; echo "nvcc exit=$RC"; [ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

extract_ref () {  # $1=iqtree  $2=NCAT  $3=out
  local O; O=$(grep "Log-likelihood of the tree:" "$1" | grep -oE '\-?[0-9]+\.[0-9]+' | head -1)
  echo "$O" > "$3"
  # FreeRate "Category Relative_rate Proportion" table: $2=rate, $3=proportion(weight); pure +R has rows 1..NCAT.
  awk -v K="$2" '/Relative_rate/{f=1;next} f&&/^[ ]*[0-9]+[ ]+[0-9.]/{print $3, $2; n++; if(n>=K)exit}' "$1" >> "$3"
}

for K in 4 6; do
  echo; echo "════════ LG+R${K} ════════ $(date -Iseconds)"
  REF="$WB/cpu_r${K}_ref.txt"
  if [ ! -f "$WB/cpu_r${K}.iqtree" ]; then
    echo "  CPU-EM reference: $CPUBIN -s ALN -te TREE -m LG+R${K} (PURE CPU path, RateFree EM) ..."
    timeout 2700 "$CPUBIN" -s "$ALN" -te "$TREE" -m "LG+R${K}" -pre "$WB/cpu_r${K}" -redo -nt 12 > "$WB/cpu_r${K}.log" 2>&1
    echo "    (cpu-ref exit=$? ; $(date -Iseconds))"
  fi
  [ -f "$WB/cpu_r${K}.iqtree" ] || { echo "  CPU-EM REFERENCE FAILED (see cpu_r${K}.log)"; tail -5 "$WB/cpu_r${K}.log"; continue; }
  extract_ref "$WB/cpu_r${K}.iqtree" "$K" "$REF"
  echo "  cpu_r${K}_ref.txt:"; sed 's/^/    /' "$REF"
  echo "──── harness LG+R${K} ────"
  "$BIN" "$ALN" "$TREE" "r${K}" "$REF" 1500   # bumped 400->1500: R6 (6 cats) needs more iters for cold==warm to converge tightly
  echo "  (harness exit=$?)"
done
echo; echo "════════ DONE $(date -Iseconds) ════════"
