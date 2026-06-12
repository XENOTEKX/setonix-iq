#!/bin/bash
# run_g4_2a_jolt_intree_v100.sh — Phase G.4.2a: the in-tree JOLT seam, single model (LG+G4), isolating WRITE-BACK.
#
# G.4.1/G.4.1b validated the JOLT joint optimiser STANDALONE. G.4.2a wires it into the REAL iqtree3 binary behind
# `--jolt`: ModelFactory::optimizeParameters routes JOLT-eligible candidates (fixed-Q reversible, ns in {4,20},
# no +I, gamma-or-uniform) through PhyloTree::optimizeParametersJOLT, which runs gpu_jolt_optimize on the GPU and
# writes the optimised (197 branches + alpha) back through the cache-invalidating setters (setGammaShape +
# clearAllPartialLH). The ONLY genuinely-new risk vs the standalone is the write-OUT; this job isolates it on a
# FIXED topology (-te the MLE .treefile) so convergence is trivial and any mismatch is a write-back/coherence bug.
#
# GATES (advisor's load-bearing one first):
#   (1) SELF-CHECK: after write-back, a FRESH CPU computeLikelihood() reproduces the JOLT lnL, rel <= 1e-9
#       (printed per-call as "[JOLT] ... GPU lnL=.. CPU lnL=.. rel=.. PASS"). This is THE write-back gate.
#   (2) MLE: the --jolt final lnL == the existing CPU MLE -7541976.8529 (reuse, NO CPU re-run), alpha->0.9963.
#   (3) NON-INTERFERENCE: the SAME binary WITHOUT --jolt (pure CPU path; the #ifdef hook inert) reaches the same
#       MLE -> the CPU path is byte-unchanged by the JOLT wiring.
#
#PBS -N g4-2a-jolt-intree
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=01:30:00
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
GXX="$(command -v g++)"; NVCC="$(command -v nvcc)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

SRC=/scratch/rc29/as1708/iqtree3-gpu
BUILD_ON="${SRC}/build-gpu-on"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WORK="${SRC}/g4_2a_runs"; mkdir -p "${WORK}"
MLE=-7541976.8529           # existing CPU MLE (LG+G4, .iqtree) — REUSED, no CPU re-run

echo "════════ G.4.2a in-tree JOLT (LG+G4, -te) — $(hostname) $(date -Iseconds) ════════"
echo "src $(cd "$SRC" && git branch --show-current) HEAD $(cd "$SRC" && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "[aln]  $([ -f "$ALN" ] && echo OK || echo MISSING)"; echo "[tree] $([ -f "$TREE" ] && echo OK || echo MISSING)"

# ── cmaple offline-build patches (idempotent) ──
CMAPLE_CML="${SRC}/cmaple/CMakeLists.txt"
if [ -f "${CMAPLE_CML}" ]; then
  grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}" && sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE).*|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi|' "${CMAPLE_CML}"
  grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}" && sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi|' "${CMAPLE_CML}"
  grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}" && sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
fi

echo; echo "──── build IQTREE_GPU=ON (incremental if configured) ────"
if [ ! -f "${BUILD_ON}/Makefile" ]; then
  echo "  (no existing build dir — full configure)"
  rm -rf "${BUILD_ON}"; mkdir -p "${BUILD_ON}"; cd "${BUILD_ON}"
  cmake "${SRC}" -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3 \
        -DBOOST_ROOT=/apps/boost/1.84.0 -DBoost_NO_SYSTEM_PATHS=ON \
        -DIQTREE_GPU=ON -DCMAKE_CUDA_HOST_COMPILER="${GXX}" -DCMAKE_CUDA_COMPILER="${NVCC}" > cmake_on.log 2>&1
  echo "  configure exit=$? (tail:)"; tail -6 cmake_on.log | sed 's/^/    /'
else
  cd "${BUILD_ON}"; echo "  (incremental make in $(pwd))"
fi
make -j12 > make_g42a.log 2>&1
RC_MAKE=$?
echo "  make exit=${RC_MAKE} (last 14 lines:)"; tail -14 make_g42a.log | sed 's/^/    /'
BIN="$(find "${BUILD_ON}" -maxdepth 2 -name iqtree3 -type f -executable 2>/dev/null | head -1)"
[ ${RC_MAKE} -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
echo "  binary: $BIN"

echo; echo "════════ (1)+(2) GPU --jolt  -m LG+G4 -te (fixed MLE topology) ════════"
T0=$(date +%s)
"$BIN" --jolt -s "$ALN" -m LG+G4 -te "$TREE" -seed 1 -nt 1 -pre "${WORK}/jolt" -redo 2>&1 | \
  grep -E "JOLT|GPU-KERNEL|Optimal log-likelihood|BEST SCORE|Log-likelihood of the tree|Gamma shape" | head -40
T1=$(date +%s); echo "[--jolt wall] $((T1-T0)) s"
JOLT_LNL=$(grep -oE "BEST SCORE FOUND : -?[0-9.]+" "${WORK}/jolt.iqtree" 2>/dev/null | grep -oE -- "-?[0-9.]+$" | tail -1)
[ -z "$JOLT_LNL" ] && JOLT_LNL=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "${WORK}/jolt.iqtree" 2>/dev/null | grep -oE -- "-?[0-9.]+" | tail -1)

echo; echo "════════ (3) NON-INTERFERENCE: SAME binary, NO --jolt (pure CPU path) ════════"
T2=$(date +%s)
"$BIN" -s "$ALN" -m LG+G4 -te "$TREE" -seed 1 -nt 1 -pre "${WORK}/cpu" -redo 2>&1 | \
  grep -E "Optimal log-likelihood|BEST SCORE|Log-likelihood of the tree|Gamma shape" | head -10
T3=$(date +%s); echo "[no-jolt wall] $((T3-T2)) s"
CPU_LNL=$(grep -oE "BEST SCORE FOUND : -?[0-9.]+" "${WORK}/cpu.iqtree" 2>/dev/null | grep -oE -- "-?[0-9.]+$" | tail -1)
[ -z "$CPU_LNL" ] && CPU_LNL=$(grep -oE "Log-likelihood of the tree: -?[0-9.]+" "${WORK}/cpu.iqtree" 2>/dev/null | grep -oE -- "-?[0-9.]+" | tail -1)

echo; echo "════════ G.4.2a VERDICT ════════"
echo "  reference CPU MLE (.iqtree, reused): ${MLE}"
echo "  (2) --jolt   final lnL = ${JOLT_LNL}"
echo "  (3) no-jolt  final lnL = ${CPU_LNL}"
python3 - "$JOLT_LNL" "$CPU_LNL" "$MLE" <<'PY' 2>/dev/null || echo "  (python compare unavailable)"
import sys
j,c,m=sys.argv[1:4]
def rel(a,b):
    try: a=float(a); b=float(b); return abs((a-b)/b) if b else abs(a-b)
    except: return float('nan')
print(f"  rel(--jolt, MLE)   = {rel(j,m):.3e}  -> {'PASS' if rel(j,m)<=1e-6 else 'CHECK'} (gate 1e-6; .treefile-precision floor ~2e-10)")
print(f"  rel(no-jolt, MLE)  = {rel(c,m):.3e}  -> {'PASS' if rel(c,m)<=1e-6 else 'CHECK'}")
print(f"  rel(--jolt,no-jolt)= {rel(j,c):.3e}  -> {'PASS' if rel(j,c)<=1e-6 else 'CHECK'} (GPU JOLT == CPU optimum)")
PY
echo "  NOTE: the load-bearing gate is the per-call [JOLT] self-check above (GPU lnL == fresh CPU computeLikelihood, rel<=1e-9)."
echo "════════ DONE $(date -Iseconds) ════════"
