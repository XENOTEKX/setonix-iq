#!/bin/bash
# baseline_cpu_modelfinder.sh — CPU ModelFinder baseline (parity lnL + timing) for the GPU project.
# Runs one model-scope (MODEL_FLAG ∈ TESTONLY|TEST|MF) on AA-100K, legacy optimiser (no --mode-l),
# captures best-fit model, lnL, ModelFinder wall, and the full model-fit table. These are the
# parity reference the BEAGLE GPU harness must match, and the per-scope CPU baselines.
#PBS -N gpumf-cpubase
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -euo pipefail
SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
IQTREE="${IQTREE:-${SANDBOX}/build-mode-p-iso-p3/iqtree3-mpi-mode-p-iso-p3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
MODEL_FLAG="${MODEL_FLAG:-TESTONLY}"
OMP_PER_RANK="${OMP_PER_RANK:-103}"; SEED="${SEED:-1}"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%s)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WORK_DIR="${SANDBOX}/runs/gpumf_cpubase_${MODEL_FLAG}_${PBS_ID_SHORT}"; mkdir -p "${WORK_DIR}"
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7 2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi
[[ -x "${IQTREE}" ]] || { echo "ERROR binary $IQTREE"; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR alignment"; exit 3; }
export KMP_BLOCKTIME=200 TMPDIR="${SANDBOX}/tmp"; mkdir -p "$TMPDIR"
OMP_ENV=( -x "OMP_NUM_THREADS=${OMP_PER_RANK}" -x "OMP_MAX_ACTIVE_LEVELS=2" -x "OMP_DYNAMIC=false"
          -x "OMP_PROC_BIND=close" -x "OMP_PLACES=cores" -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}" )
echo "══ CPU ModelFinder baseline: -m ${MODEL_FLAG}  AA-100K  np=1 T=${OMP_PER_RANK}  md5=$(md5sum "$IQTREE"|awk '{print $1}') ══"
echo "  alignment: ${ALIGNMENT}"
echo "  work_dir:  ${WORK_DIR}"
t0=$(date +%s)
mpirun -np 1 --bind-to none "${OMP_ENV[@]}" numactl --localalloc -- \
  "${IQTREE}" -s "${ALIGNMENT}" -m "${MODEL_FLAG}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
  --prefix "${WORK_DIR}/mf" > "${WORK_DIR}/stdout.log" 2>&1
rc=$?; t1=$(date +%s)
echo "  exit=${rc}  wall=$((t1-t0))s"
echo "── results ──"
grep -iE "Best-fit model|ModelFinder will|models will be|Wall-clock time for ModelFinder|BEST SCORE FOUND|Log-likelihood of the tree|Akaike|Total wall-clock" "${WORK_DIR}/mf.log" 2>/dev/null | head -15
echo "── best-fit + top models (.iqtree) ──"
grep -A6 -iE "Best-fit model according" "${WORK_DIR}/mf.iqtree" 2>/dev/null | head -8
