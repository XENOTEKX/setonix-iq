#!/bin/bash
# run_cpu_bench_aa_100k_spr_r1r2_mtest.sh
# Purpose: R1+R2+AVX-512 baseline — 1-node, non-MPI, -m TEST, AA 100K.
#
# Source: XENOTEKX/setonix-iq branch gadi-spr-r2-avx512 (commit 07966e4) + R2 patch
# Patches applied vs vanilla v3.1.2:
#   P1/R1: schedule(static) + NUMA first-touch (phylokernelnew.h, phylotreesse.cpp)
#   P2:    CMakeLists.txt icpx/IntelLLVM AVX-512 detection fix (-mavx512f -mfma)
#   P3:    phylokernelavx512.cpp nonrev template-arity fixes
#   R2:    0004-thp-partial-lh-madvise.patch — madvise(MADV_HUGEPAGE) on central_partial_lh
#          (phylotree.cpp, applied via git apply on login node)
#
# Pre-cloned+patched at: /scratch/rc29/as1708/iqtree3-r1r2/src/iqtree3
# Build target:          /scratch/rc29/as1708/iqtree3-r1r2/build-icx-r1r2-avx512/iqtree3-r1r2
#
# Parity: lnL within ±0.5 of -7,541,976.860, model LG+G4
#
#PBS -N iq-aa-100k-r1r2-avx512
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/rc29+scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
THREADS=103
SEED=1
MF_MODE="-m TEST"
DATA_TYPE="AA"

PROJECT_DIR="/scratch/rc29/as1708/iqtree3-r1r2"
SRC_DIR="${PROJECT_DIR}/src/iqtree3"
BUILD_DIR="${PROJECT_DIR}/build-icx-r1r2-avx512"
IQTREE="${BUILD_DIR}/iqtree3-r1r2"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_100k_r1r2avx512_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE R1+R2+AVX-512 — clone gadi-spr-r2-avx512 + build + bench"
echo "║  target:    ${BUILD_DIR}"
echo "║  mf_mode:   ${MF_MODE}  threads: ${THREADS}  seed: ${SEED}"
echo "║  pbs_id:    ${PBS_ID_SHORT}"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Modules ──────────────────────────────────────────────────────────────────
module load cmake/3.31.6
module load intel-compiler-llvm/2025.3.2
module load binutils/2.44
module load eigen/3.3.7
module load boost/1.84.0

export CC="$(command -v icx)"
export CXX="$(command -v icpx)"
echo "[r1r2avx512] icpx: $(icpx --version 2>&1 | head -1)"

# ── Verify source (pre-cloned from login node) ────────────────────────────────
[[ -d "${SRC_DIR}/.git" ]] || { echo "ERROR: source not found at ${SRC_DIR} — clone from login node first." >&2; exit 5; }
cd "${SRC_DIR}"
COMMIT=$(git rev-parse HEAD)
echo "[r1r2avx512] commit: ${COMMIT}"
echo "[r1r2avx512] branch: $(git rev-parse --abbrev-ref HEAD)"

# Patch sanity checks
if ! grep -q 'schedule(static)' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: R1 patch (schedule(static)) not found in phylokernelnew.h" >&2; exit 4
fi
if ! grep -q 'NUMA first-touch' "${SRC_DIR}/tree/phylotreesse.cpp"; then
    echo "ERROR: R1 patch (NUMA first-touch) not found in phylotreesse.cpp" >&2; exit 4
fi
if ! grep -q 'MADV_HUGEPAGE' "${SRC_DIR}/tree/phylotree.cpp"; then
    echo "ERROR: R2 patch (MADV_HUGEPAGE) NOT found in phylotree.cpp — apply 0004-thp-partial-lh-madvise.patch first" >&2; exit 4
fi
echo "[r1r2avx512] patch check PASS — R1 present (schedule(static), NUMA first-touch), R2 present (MADV_HUGEPAGE)"

# ── cmaple tweaks (disable FetchContent googletest — no internet on compute nodes) ──
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable IPO (LTO) by default|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled|' "${CMAPLE_CML}"
fi
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled|' "${CMAPLE_CML}"
fi
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
fi

# ── Build (non-MPI, like the original sa0557 baseline-avx512-r1+r2) ───────────
ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

echo ""
echo "[r1r2avx512] building in ${BUILD_DIR}..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CC="${CC}" CXX="${CXX}" cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DEIGEN3_INCLUDE_DIR="${EIGEN_ROOT:-/apps/eigen/3.3.7}/include/eigen3" \
    -DBOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    2>&1 | tail -5

make -j 104 iqtree3 2>&1 | tail -10
mv "${BUILD_DIR}/iqtree3" "${BUILD_DIR}/iqtree3-r1r2"

[[ -x "${IQTREE}" ]] || { echo "ERROR: build failed — binary not found." >&2; exit 6; }
ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
echo "[r1r2avx512] binary: ${IQTREE}  md5: ${ACTUAL_MD5}"

# Verify non-MPI build (should link libiomp5, not libmpi)
if ldd "${IQTREE}" | grep -q 'libgomp'; then
    echo "ERROR: binary links libgomp — expected libiomp5 (ICX build)." >&2; exit 7
fi
echo "[r1r2avx512] library check: $(ldd "${IQTREE}" | grep -oE 'libiomp5|libmpi' | tr '\n' ' ')"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${WORK_DIR}"

START_EPOCH=$(date +%s)
set +e
numactl --localalloc "${IQTREE}" \
    -s "${ALIGNMENT}" -nt "${THREADS}" -seed "${SEED}" ${MF_MODE} \
    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
set -e
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
cat "${WORK_DIR}/iqtree_stdout.log"
echo ""
echo "rc=${IQRC}  wall=${WALL}s"

# ── Parse results + parity check ─────────────────────────────────────────────
REF_LNL=-7541976.860
REF_TOL=0.5
REF_MODEL="LG+G4"

MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_LNL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_MODEL=$(grep -oP 'Best-fit model:\s*\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)

echo ""
echo "=== PARITY CHECK (ref ${REF_LNL}, tol ${REF_TOL}) ==="
echo "  MF wall(s): ${MF_WALL:-MISSING}"
echo "  lnL:        ${BEST_LNL:-MISSING}"
echo "  model:      ${BEST_MODEL:-MISSING}"

python3 - <<PYEOF
import json, os, subprocess, sys

ref_lnl    = ${REF_LNL}
tol        = ${REF_TOL}
mf_wall    = float("${MF_WALL}") if "${MF_WALL}" else None
best_lnl   = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model = "${BEST_MODEL}" or None

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

checks = {}
checks["exit_0"]      = ${IQRC} == 0
checks["lnL_parity"]  = best_lnl is not None and abs(best_lnl - ref_lnl) < tol
checks["model_LG+G4"] = best_model == "${REF_MODEL}"
all_pass = all(checks.values())

for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

r1_only_mf  = 256.186   # 870135052, R1 + dirty build
vanilla_mf_est = 264.202  # 170138365 MF already done, pending tree
if mf_wall:
    print(f"\n  MF wall:          {mf_wall:.3f} s")
    print(f"  vs R1-only:       {r1_only_mf/mf_wall:.3f}× (committed R1+cmake vs dirty R1)")

record = {
  "run_id": "gadi_${RUN_LABEL}_${PBS_ID_SHORT}", "label": "${RUN_LABEL}",
  "platform": "gadi", "run_type": "r1_avx512_mtest_np1",
  "dataset": "${ALIGNMENT}", "dataset_short": "complex_aa_100k",
  "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
  "np": 1, "threads": ${THREADS}, "seed": ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary": "${IQTREE}",
  "binary_md5": "${ACTUAL_MD5}",
  "binary_note": "R1+R2+AVX-512 (gadi-spr-r2-avx512 @ 07966e4 + 0004-thp patch): schedule(static)+NUMA first-touch+cmake P2+P3+THP madvise(MADV_HUGEPAGE), non-MPI, ICX 2025.3.2",
  "source_commit": "${COMMIT}",
  "source_branch": "gadi-spr-r2-avx512",
  "summary": {
    "pass": 1 if ${IQRC} == 0 else 0,
    "all_pass": all_pass,
    "lnL": best_lnl,
    "model": best_model,
    "mf_wall_s": mf_wall,
    "lnL_delta_vs_ref": round(abs(best_lnl - ref_lnl), 4) if best_lnl else None,
    "parity_checks": checks,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu": sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
            "ncpus": os.environ.get("PBS_NCPUS"), "project": "dx61"},
  },
  "build_tag": "r1_avx512_icx_spr_nonmpi",
  "headline_table_row": "R1+R2+AVX512",
  "compares_to": {
    "vanilla":  "170138365 (no patches, -m TEST, np=1)",
    "R1-only":       "170135052 (R1 dirty build, -m TEST, np=1, MF=256.186s)",
    "R1+AVX512":     "170139827 (R1+cmake fixes, -m TEST, np=1, pending)",
    "R1+AVX512":    "170139827 (R1+cmake fixes, -m TEST, np=1, pending)",
  },
}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[r1r2avx512] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
