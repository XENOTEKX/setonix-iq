#!/bin/bash
# run_mf_iso_dna_100k_1node.sh — IQ-TREE3 MF-iso ModelFinder benchmark, 1-node MPI.
#
# DNA, 100K sites (100 taxa), 1 MPI rank × 103 OpenMP threads.
#
# PURPOSE
#   Establish a Phase 0.5/0.6 ModelFinder baseline at np=1 (where FCA is
#   a no-op) to confirm correctness BEFORE scaling to 2-node.  At np=1,
#   both mf-iso and the standard binary should give the same lnL.
#
# ISOLATION KEY: --m TESTONLY stops after the ModelFinder phase.
#
# EXPECTED_LNL: TBD — run run_baseline_dna_100k_spr.sh first, then update
#   the EXPECTED_LNL constant in the Python block below.
#
# Binary:  /scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
#          (branch mf-iso-phase0.5-0.6)
# Parity:  OMP_PER_RANK=103, numactl --localalloc, KMP_BLOCKTIME=200, seed=1

#PBS -N mf-iso-dna-100k-1n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/dx61+scratch/um09
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="DNA"
DATASET_SHORT="complex_dna_100k"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="DNA_100k_mfiso_np1_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7          2>/dev/null || true
    module load intel-compiler-llvm    2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: MF-iso binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found after module load." >&2; exit 4; }
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 5
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5." >&2; exit 6
fi
if ! cat "${IQTREE}" > /dev/null; then
    echo "ERROR: ${IQTREE} not readable on this node (Lustre OST not yet synced?)." >&2
    echo "       On the login node: run 'sync' after copying the binary, then resubmit." >&2
    exit 2
fi
if ! nm "${IQTREE}" 2>/dev/null | grep -q '_ZN17CandidateModelSet14filterRatesMPIEi'; then
    if ! strings "${IQTREE}" 2>/dev/null | grep -q 'filterRatesMPI'; then
        echo "[preflight] WARNING: filterRatesMPI not verified (nm + strings both failed)" >&2
        echo "[preflight] WARNING: continuing — ldd confirmed MPI+libiomp5; post-run lnL will validate" >&2
    else
        echo "[preflight] filterRatesMPI: found via strings"
    fi
else
    echo "[preflight] filterRatesMPI: confirmed via nm"
fi

# ── OMP / runtime ─────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DNA 100K MF-iso Benchmark — 1-node, ModelFinder only (-m TESTONLY)"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  branch:       mf-iso-phase0.5-0.6"
echo "║  baseline ref: run_baseline_dna_100k_spr.sh (EXPECTED_LNL TBD)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Probe (hardware/software/binary/source) ───────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"
probe_hw_sw "${IQTREE}"
probe_env

RANK_PROBE="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_probe.sh"
[[ -x "${RANK_PROBE}" ]] || { echo "ERROR: rank_probe.sh not found at ${RANK_PROBE}" >&2; exit 9; }

# ── ModelFinder-only run ──────────────────────────────────────────────
echo "[1node] ModelFinder-only run, ${NRANKS} rank × ${OMP_PER_RANK} OMP"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --bind-to none \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${RANK_PROBE}" \
        numactl --localalloc -- \
            "${IQTREE}" -s "${ALIGNMENT}" -m TESTONLY -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

grep -E '^RANK-PROBE: |\[.*\]' "${WORK_DIR}/iqtree_run.bindings.log" > "${WORK_DIR}/rank_bindings.log" 2>/dev/null || true

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[1node] done: rc=${IQRC} wall=${WALL}s"

grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_time.log"  || true
grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_diag.log"  || true
grep -E '^PROBE: '        "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/probe.log"    2>/dev/null || true

{
    echo "# rank, model_idx, model_name, subst, rate, dt_seconds, ref_remaining"
    awk -F' ' '
    /^MF-TIME: rank / {
        for (i=1; i<=NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2]; }
        printf "%s, %s, %s, %s, %s, %s, %s\n",
            v["rank"], v["model"], v["name"], v["subst"], v["rate"], v["dt"], v["ref_remaining"];
    }' "${WORK_DIR}/mf_time.log"
} > "${WORK_DIR}/rank_models.csv" 2>/dev/null || true

echo "[1node] MF-TIME lines:     $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[1node] MF-MPI-DIAG lines: $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"
echo "[1node] PROBE lines:       $(wc -l < "${WORK_DIR}/probe.log" 2>/dev/null || echo 0)"
echo "[1node] RANK-PROBE lines:  $(grep -c '^RANK-PROBE: ' "${WORK_DIR}/rank_bindings.log" 2>/dev/null || echo 0)"

# ── Run record ────────────────────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
nranks, omp_per_rank, threads = ${NRANKS}, ${OMP_PER_RANK}, ${TOTAL_THREADS}
wall, iqrc = int("${WALL}"), int("${IQRC}")
alignment, ibin = "${ALIGNMENT}", "${IQTREE}"
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None; best_model = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"Best-fit model:\s+(\S+)", line)
        if m: best_model = m.group(1)

mf_time_log = os.path.join(work, "mf_time.log")
per_rank = {}
if os.path.isfile(mf_time_log):
    for line in open(mf_time_log, errors="replace"):
        m = re.search(r'rank (\d+) .* dt=([\d.]+)', line)
        if not m: continue
        r = int(m.group(1)); dt = float(m.group(2))
        per_rank.setdefault(r, []).append(dt)
mf_time_summary = {
    f"rank_{r}": {"n_models": len(v), "total_eval_s": round(sum(v),3),
                  "mean_s": round(sum(v)/len(v),3) if v else None,
                  "max_s": round(max(v),3) if v else None}
    for r, v in sorted(per_rank.items())
}

# Prior SPR full-run reference (job 168425674, standard binary, 103T SPR, seed=1):
#   lnL -5692984.539, F81+F+G4, MF wall 61.740 s
# TESTONLY lnL (NJ tree) vs full-run lnL (SPR tree) can differ by < 1.0 unit.
# Update to the run_baseline_dna_100k_spr.sh TESTONLY result once available.
EXPECTED_LNL = -5692984.539

verify = []
if rep_ll is not None and EXPECTED_LNL is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({"file": os.path.basename(alignment), "status": "pass" if diff < 1.0 else "fail",
                   "expected_spr_ref": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6),
                   "note": "NJ-tree TESTONLY lnL vs SPR-tree full-run ref; tol=1.0"})

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "mf_iso",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
    "threads": threads, "seed": ${SEED},
    "model_finder_only": True,
    "timing": [{"command": f"mpirun -np {nranks} numactl --localalloc iqtree3-mpi -s alignment_100000.phy -m TESTONLY -T {omp_per_rank} -seed ${SEED}",
                "time_s": iqwall if iqwall is not None else wall}],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "lnL": rep_ll,
        "best_model": best_model,
        "all_pass": iqrc == 0,
    },
    "mf_time_summary": mf_time_summary,
    "env": {
        "hostname": sh("hostname"), "date": sh("date -Iseconds"),
        "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
        "cores": int(sh("nproc","0") or 0),
        "iqtree_binary": ibin,
        "iqtree_version": sh(f"{ibin} --version 2>&1 | head -1"),
        "mpi_nranks": nranks,
        "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
                "ncpus": os.environ.get("PBS_NCPUS"), "project": "${PROJECT}"},
    },
    "profile": {"nranks": nranks, "omp_per_rank": omp_per_rank,
                "placement": "mpi_1node_excl_testonly"},
    "build_tag":     "mf_iso_phase0.5_0.6_icx_avx512_mftime",
    "branch":        "mf-iso-phase0.5-0.6",
    "non_canonical": True,
    "non_canonical_label": "MF-iso Phase 0.5+0.6 · ICX+MPI · AVX-512",
    "group":         "mf_iso_scaling",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[1node] wrote {out_path}")
print(f"[1node] lnL      = {rep_ll}")
print(f"[1node] best_mdl = {best_model}")
PYEOF

echo "[1node] done."
exit "${IQRC}"
