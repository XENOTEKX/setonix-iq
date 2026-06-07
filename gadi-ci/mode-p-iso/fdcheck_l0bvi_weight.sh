#!/bin/bash
# fdcheck_l0bvi_weight.sh — L.0b.vi FreeRate WEIGHT-gradient FDCHECK validation.
# Runs ModelFinder restricted to a +R model (LG+R4 via -mrate R) under --mode-l
# --mode-l-fd-check, which makes optimizeModeLAllParameters compute BOTH the analytic
# weight score (new L.0b.vi code) AND the finite-difference gradient, printing per LM
# iteration:  |G-ratio-prop0| = |G_analytic_prop0 - G_fd_prop0| / |G_fd_prop0|.
# PASS criterion: max |G-ratio-prop0| < 0.01 (the weight gradient matches FD).  This
# de-risks the gradient math BEFORE spending a 2h production +R traversal gate.
# Also reports |G-ratio-rate0| (L.0b.v, expected already <0.01) as a regression check.
#
#PBS -N fdcheck-l0bvi
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
IQTREE="${IQTREE:-${SANDBOX}/build-mode-p-iso-p3/iqtree3-mpi-mode-p-iso-p3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
# Validate on LG+R4 (k=4 -> ndim 2*(4-1)=6: 3 weight + 3 rate dims).  -mrate R overrides
# the -m TEST ratehet set to +R only; -cmin/-cmax 4 -> exactly +R4.  +F doubling kept
# (2 models LG+R4, LG+F+R4) — both exercise the weight gradient.
MODEL_FLAGS="${MODEL_FLAGS:- -m TEST -mset LG -mrate R -cmin 4 -cmax 4}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WORK_DIR="${SANDBOX}/runs/fdcheck_l0bvi_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]] || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
echo "binary md5: $(md5sum "${IQTREE}" | awk '{print $1}')  (expect 8469af7b2035cd110ae3b5be1d80474f)"
echo "model flags: ${MODEL_FLAGS}"

export KMP_BLOCKTIME=200
export TMPDIR="${SANDBOX}/tmp"; mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS="${OMP_PER_RANK}"

STDOUT="${WORK_DIR}/fdcheck_stdout.log"
set +e
mpirun -np 1 --bind-to none numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" ${MODEL_FLAGS} -T "${OMP_PER_RANK}" -seed "${SEED}" \
                --mode-l --mode-l-fd-check \
                --prefix "${WORK_DIR}/fdcheck_inner" \
    > "${STDOUT}" 2>&1
rc=$?
set -e
echo "iqtree exit=${rc}"

echo ""
echo "══ L.0b.vi weight-gradient FDCHECK result ══════════════════════════"
FDC=$(grep -c 'MODE-L-FDCHECK' "${STDOUT}" 2>/dev/null || echo 0)
echo "  MODE-L-FDCHECK lines: ${FDC}"
maxratio() { { grep -hoP "\|G-ratio-$1\|=\K[0-9.eE+-]+" "${STDOUT}" 2>/dev/null || true; } \
    | awk 'BEGIN{m=-1}{v=$1+0; if(v<0)v=-v; if(v>m)m=v} END{printf "%.6e", (m<0?0:m)}'; }
nlines()  { grep -hc "\|G-ratio-$1\|=" "${STDOUT}" 2>/dev/null || echo 0; }
PROP0_MAX=$(maxratio prop0); PROP0_N=$(nlines prop0)
RATE0_MAX=$(maxratio rate0); RATE0_N=$(nlines rate0)
echo "  |G-ratio-prop0|  lines=${PROP0_N}  max=${PROP0_MAX}   (L.0b.vi WEIGHT gradient — PASS if <0.01)"
echo "  |G-ratio-rate0|  lines=${RATE0_N}  max=${RATE0_MAX}   (L.0b.v rate gradient — regression check)"
echo "  sample FDCHECK lines:"
grep 'MODE-L-FDCHECK' "${STDOUT}" 2>/dev/null | grep -E 'prop0|rate0' | head -6 | sed 's/^/    /' || true

PASS=1
awk "BEGIN{exit !(${PROP0_MAX} < 0.01 && ${PROP0_N} > 0)}" || { echo "  ✗ FAIL: |G-ratio-prop0| max=${PROP0_MAX} (>=0.01 or no lines) — WEIGHT gradient WRONG"; PASS=0; }
[[ "${PASS}" -eq 1 ]] && echo "  ✓ PASS: weight gradient matches FD (|G-ratio-prop0|=${PROP0_MAX} < 0.01)"
echo "  work_dir: ${WORK_DIR}"
echo "════════════════════════════════════════════════════════════════════"
exit $(( PASS == 1 ? 0 : 10 ))
