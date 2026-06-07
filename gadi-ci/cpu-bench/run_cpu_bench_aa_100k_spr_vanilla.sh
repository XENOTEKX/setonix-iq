#!/bin/bash
# run_cpu_bench_aa_100k_spr_vanilla.sh
# Purpose: True vanilla IQ-TREE 3.1.2 baseline — ICX, no patches, AA 100K -m TEST.
#
# This fills the genuine "pre-R1+R2" row in the Headline Run Progression table.
# Clones a fresh copy of v3.1.2 from GitHub, builds with ICX + -march=sapphirerapids,
# applies NO patches, then runs AA 100K -m TEST -T 103 seed=1.
#
# Parity target: lnL within ±0.5 of -7,541,976.860 (ref 168425673), model LG+G4
#
# Build target: /scratch/rc29/as1708/iqtree3-vanilla/build-icx-nopatch/iqtree3-mpi
# Source:       /scratch/rc29/as1708/iqtree3-vanilla/src/iqtree3
#
#PBS -N iq-aa-100k-vanilla
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

PROJECT_DIR="/scratch/rc29/as1708/iqtree3-vanilla"
SRC_DIR="${PROJECT_DIR}/src/iqtree3"
BUILD_DIR="${PROJECT_DIR}/build-icx-nopatch"
IQTREE="${BUILD_DIR}/iqtree3-mpi"

REPO_DIR="${HOME}/setonix-iq"
WORK_ROOT="/scratch/dx61/as1708/cpu_bench"
PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_LABEL="${DATA_TYPE}_100k_vanilla_icx_seed${SEED}"
WORK_DIR="${WORK_ROOT}/profiles/${RUN_LABEL}_${PBS_ID_SHORT}"
RUNS_DIR="${REPO_DIR}/logs/runs"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3.1.2 vanilla — clone + build + benchmark"
echo "║  target:    ${BUILD_DIR}"
echo "║  mf_mode:   ${MF_MODE}  threads: ${THREADS}  seed: ${SEED}"
echo "║  pbs_id:    ${PBS_ID_SHORT}"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Modules ──────────────────────────────────────────────────────────────────
module load cmake/3.31.6
module load openmpi/4.1.7
module load intel-compiler-llvm/2025.3.2
module load binutils/2.44
module load eigen/3.3.7
module load boost/1.84.0

export OMPI_CC="$(command -v icx)"
export OMPI_CXX="$(command -v icpx)"
echo "[vanilla] icpx: $(icpx --version 2>&1 | head -1)"
echo "[vanilla] mpirun: $(mpirun --version 2>&1 | head -1)"

# ── Clone (skip if already present and clean) ─────────────────────────────────
if [[ -d "${SRC_DIR}/.git" ]]; then
    echo "[vanilla] source dir exists — verifying cleanliness..."
    cd "${SRC_DIR}"
    DIRTY=$(git status --porcelain | { grep -v '^?' || true; } | wc -l)
    COMMIT=$(git rev-parse HEAD)
    if [[ "${DIRTY}" -ne 0 ]]; then
        echo "[vanilla] dirty working tree (${DIRTY} modified files), restoring to v3.1.2..."
        git checkout v3.1.2 --
        git submodule update --init --recursive
    fi
    echo "[vanilla] source at: $(git describe --tags --always)"
else
    echo "[vanilla] cloning IQ-TREE 3.1.2..."
    mkdir -p "${PROJECT_DIR}/src"
    git clone --depth 1 --branch v3.1.2 \
        https://github.com/iqtree/iqtree3.git "${SRC_DIR}"
    cd "${SRC_DIR}"
    git submodule update --init --recursive
    echo "[vanilla] cloned: $(git describe --tags --always)"
fi

cd "${SRC_DIR}"
COMMIT=$(git rev-parse HEAD)
echo "[vanilla] commit: ${COMMIT}"

# ── Patch sanity check — must have schedule(dynamic,1), must NOT have R1 ─────
if ! grep -q 'schedule(dynamic,1)' "${SRC_DIR}/tree/phylokernelnew.h"; then
    echo "ERROR: phylokernelnew.h missing schedule(dynamic,1) — R1 patch appears applied." >&2
    exit 4
fi
if grep -q 'NUMA first-touch' "${SRC_DIR}/tree/phylotreesse.cpp"; then
    echo "ERROR: phylotreesse.cpp has NUMA first-touch comments — R1 patch appears applied." >&2
    exit 4
fi
echo "[vanilla] patch check PASS — schedule(dynamic,1) present, no NUMA first-touch"

# ── cmaple tweaks (disable IPO and unittest to avoid LTO / GoogleTest issues) ─
# Method mirrors bootstrap_iqtree_3.1.2_mpi.sh — compute nodes have no internet.
CMAPLE_CML="${SRC_DIR}/cmaple/CMakeLists.txt"
if grep -q 'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)' "${CMAPLE_CML}"; then
    sed -i 's|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enable IPO (LTO) by default|set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) # Gadi: disabled|' "${CMAPLE_CML}"
fi
if grep -qE '^[[:space:]]*add_subdirectory\(unittest\)' "${CMAPLE_CML}"; then
    sed -i 's|^\([[:space:]]*\)add_subdirectory(unittest)|\1# add_subdirectory(unittest) # Gadi: disabled|' "${CMAPLE_CML}"
fi
# Disable the entire FetchContent block — fetches googletest from GitHub at configure time
if grep -qE 'FetchContent_MakeAvailable\(googletest\)' "${CMAPLE_CML}"; then
    sed -i '/^include(FetchContent)$/,/^FetchContent_MakeAvailable(googletest)$/ s|^|# GADI-DISABLED: |' "${CMAPLE_CML}"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
ARCH_FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids -fopenmp"
EXTRA="-fno-omit-frame-pointer -g"

echo ""
echo "[vanilla] building in ${BUILD_DIR}..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CC="$(command -v mpicc)" CXX="$(command -v mpicxx)" cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIQTREE_FLAGS=mpi \
    -DEIGEN3_INCLUDE_DIR="${EIGEN_ROOT:-/apps/eigen/3.3.7}/include/eigen3" \
    -DBOOST_ROOT="${BOOST_ROOT:-/apps/boost/1.84.0}" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${EXTRA}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
    2>&1 | tail -10

make -j"$(nproc)" 2>&1 | tail -20

# find binary (CMake sometimes places it in a subdirectory)
if [[ ! -x "${IQTREE}" ]]; then
    found="$(find "${BUILD_DIR}" -maxdepth 3 -name 'iqtree3-mpi' -type f -executable 2>/dev/null | head -1)"
    [[ -n "${found}" ]] && ln -sf "${found}" "${IQTREE}"
fi
[[ -x "${IQTREE}" ]] || { echo "ERROR: ${IQTREE} not produced." >&2; exit 5; }

BINARY_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
echo "[vanilla] binary:   ${IQTREE}"
echo "[vanilla] md5:      ${BINARY_MD5}"
echo "[vanilla] version:  $(mpirun -n 1 --bind-to none "${IQTREE}" --version 2>&1 | head -1)"

# linkage checks
LDD_OUT="$(ldd "${IQTREE}" 2>&1)"
if echo "${LDD_OUT}" | grep -q 'libgomp'; then
    echo "ERROR: libgomp linked — expected libiomp5." >&2; exit 6
fi
if ! echo "${LDD_OUT}" | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: libmpi not linked." >&2; exit 7
fi
echo "[vanilla] linkage OK (libiomp5 + libmpi)"

# write build metadata
cat > "${BUILD_DIR}/.build-info.json" <<EOF
{
  "source":        "${SRC_DIR}",
  "commit":        "${COMMIT}",
  "tag":           "v3.1.2",
  "patches":       "none",
  "compiler":      "$(icpx --version 2>&1 | head -1)",
  "arch_flags":    "${ARCH_FLAGS} ${EXTRA}",
  "binary_md5":    "${BINARY_MD5}",
  "host":          "$(hostname)",
  "date":          "$(date -Iseconds)"
}
EOF

# ── Benchmark ─────────────────────────────────────────────────────────────────
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found." >&2; exit 3; }

cd "${WORK_DIR}"
export KMP_BLOCKTIME=200
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

echo ""
echo "[vanilla] starting benchmark..."
START_EPOCH=$(date +%s)

set +e
mpirun -np 1 --mca coll ^ucc --bind-to none \
    numactl --localalloc "${IQTREE}" \
    -s "${ALIGNMENT}" -T "${THREADS}" -seed "${SEED}" ${MF_MODE} \
    --prefix "${WORK_DIR}/iqtree_inner" \
    > "${WORK_DIR}/iqtree_stdout.log" 2>&1
IQRC=$?
set -e

END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
cat "${WORK_DIR}/iqtree_stdout.log"
echo ""
echo "rc=${IQRC}  wall=${WALL}s"

# ── perf stat pass ────────────────────────────────────────────────────────────
PERF_EVENTS="cycles:u,instructions:u,cache-references:u,cache-misses:u"
if [[ ${IQRC} -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[vanilla] perf stat pass..."
    perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
        mpirun -np 1 --mca coll ^ucc --bind-to none \
        numactl --localalloc "${IQTREE}" \
        -s "${ALIGNMENT}" -T "${THREADS}" -seed "${SEED}" ${MF_MODE} \
        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Parity check + JSON record ────────────────────────────────────────────────
REF_LNL=-7541976.860
REF_TOL=0.5
REF_MODEL="LG+G4"

MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder:\s*\K[\d.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_LNL=$(grep -oP 'BEST SCORE FOUND\s*:\s*\K[-0-9.]+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)
BEST_MODEL=$(grep -oP 'Best-fit model according to BIC:\s*\K\S+' \
    "${WORK_DIR}/iqtree_stdout.log" | tail -1 || true)

echo ""
echo "=== PARITY CHECK (ref 168425673, tol ${REF_TOL}) ==="
echo "  lnL:        ${BEST_LNL:-MISSING}"
echo "  model:      ${BEST_MODEL:-MISSING}"
echo "  MF wall(s): ${MF_WALL:-MISSING}"
echo "  ref lnL:    ${REF_LNL}"

python3 - <<PYEOF
import json, os, re, subprocess, sys

def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return d

mf_wall    = float("${MF_WALL}") if "${MF_WALL}" else None
best_lnl   = float("${BEST_LNL}") if "${BEST_LNL}" else None
best_model = "${BEST_MODEL}" or None

checks = {
    "exit_0":      ${IQRC} == 0,
    "lnL_parity":  best_lnl is not None and abs(best_lnl - ${REF_LNL}) < ${REF_TOL},
    "model_LG+G4": best_model == "${REF_MODEL}",
}
all_pass = all(checks.values())
print()
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print()
print("OVERALL:", "ALL PASS ✓" if all_pass else "FAIL ✗")

# delta vs row A
row_a_mf = 399.456
if mf_wall:
    print(f"\n  MF wall:    {mf_wall:.3f} s")
    print(f"  vs row A (R1+R2, 399.456 s):  Δ = {mf_wall - row_a_mf:+.1f} s")
    print(f"  lnL delta: {abs(best_lnl - ${REF_LNL}):.4f}" if best_lnl else "")

def _parse_perf(path):
    agg = {}
    if not os.path.isfile(path): return agg
    for line in open(path, errors="replace"):
        m = re.match(r"\s*([\d,]+)\s+([\w.\-:/]+)", line)
        if m:
            try: agg[m.group(2).split(":",1)[0]] = int(m.group(1).replace(",",""))
            except ValueError: pass
    return agg
_agg = _parse_perf("${WORK_DIR}/perf_stat.txt")
_cyc, _ins = _agg.get("cycles"), _agg.get("instructions")
_metrics = {}
if _cyc and _ins:
    _metrics["IPC"] = round(_ins/_cyc, 4)
if _agg.get("cache-misses") and _agg.get("cache-references"):
    _metrics["cache-miss-rate"] = round(100.0 * _agg["cache-misses"] / _agg["cache-references"], 4)

record = {
  "run_id": "gadi_${RUN_LABEL}_${PBS_ID_SHORT}", "label": "${RUN_LABEL}",
  "platform": "gadi", "run_type": "vanilla_icx_nopatch",
  "dataset": "${ALIGNMENT}", "dataset_short": "complex_aa_100k",
  "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
  "threads": ${THREADS}, "seed": ${SEED},
  "mf_mode": "${MF_MODE}",
  "binary": "${IQTREE}",
  "binary_md5": "${BINARY_MD5}",
  "binary_note": "v3.1.2 fresh clone (4e91dd61), zero patches, ICX -march=sapphirerapids MPI np=1",
  "summary": {
    "pass": 1 if ${IQRC} == 0 else 0,
    "all_pass": all_pass,
    "lnL": best_lnl,
    "model": best_model,
    "mf_wall_s": mf_wall,
    "lnL_delta_vs_ref": round(abs(best_lnl - ${REF_LNL}), 4) if best_lnl else None,
    "mf_wall_delta_vs_row_a": round(mf_wall - row_a_mf, 1) if mf_wall else None,
    "parity_checks": checks,
  },
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu": sh("lscpu|grep 'Model name'|head -1|cut -d: -f2-|xargs"),
    "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
            "ncpus": os.environ.get("PBS_NCPUS"), "project": "rc29"},
  },
  "build_tag": "v3.1.2_icx_avx512_spr_no_patches",
  "headline_table_row": "vanilla (new)",
  "compares_to": "168425673 (row A, ICX+R1+R2, 399.456 s MF, -m TEST)",
}
if _metrics:
    record["profile"] = {"metrics": _metrics}
out = "${RUNS_DIR}/gadi_${RUN_LABEL}_${PBS_ID_SHORT}.json"
json.dump(record, open(out, "w"), indent=2, default=str)
print(f"\n[vanilla] wrote {out}")
sys.exit(0 if all_pass else 1)
PYEOF

exit "${IQRC}"
