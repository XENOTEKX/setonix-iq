#!/bin/bash
# run_baseline_dna_1m_spr.sh — standard-binary ModelFinder baseline, DNA 1M.
#
# Binary: the STANDARD (non-MPI) IQ-TREE 3.1.2 SPR build by sa0557 at
#   /scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3
# Dataset: DNA GTR+I+G4, 100 taxa × 1,000,000 sites.
#
# PURPOSE
#   - Establish MF-only reference lnL and best-model for DNA 1M on SPR.
#   - Prior CLX reference (168422813): lnL −59,208,019.212, best F81+F+G4,
#     MF wall 10,230 s (CLX, 2h50m). SPR is ~3× faster → expect ~3,500 s.
#   - Uses -m TESTONLY (MF only, no tree search); full analysis ~18 h on CLX.
#
# Run this BEFORE run_mf_iso_dna_1m_1node.sh.

#PBS -N mf-iso-dna-1m-baseline
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/dx61+scratch/um09
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

THREADS="${THREADS:-103}"
SEED="${SEED:-1}"
DATA_TYPE="DNA"
DATASET_SHORT="complex_dna_1m"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="DNA_1m_baseline_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load intel-compiler-llvm    2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: baseline binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
if ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "WARNING: baseline binary appears to link libmpi — that's the MPI build, not the standard one." >&2
fi

# ── OMP / runtime ─────────────────────────────────────────────────────
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=10000
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

# ── Probe (hardware/software/binary) ──────────────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DNA 1M BASELINE — non-MPI, 1 node, ${THREADS} threads, -m TESTONLY"
echo "║  run_id:       ${RUN_ID}"
echo "║  binary:       ${IQTREE}"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  CLX ref (168422813): lnL -59208019.212, F81+F+G4, MF 10230 s"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

probe_hw_sw "${IQTREE}"
probe_env

# ── Run ───────────────────────────────────────────────────────────────
echo "[baseline] running standard binary, ${THREADS} OMP threads, -m TESTONLY"
START_EPOCH=$(date +%s)

numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -nt "${THREADS}" -seed "${SEED}" -m TESTONLY \
                --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[baseline] done: rc=${IQRC} wall=${WALL}s"

# ── Run record ────────────────────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
threads, seed = ${THREADS}, ${SEED}
wall, iqrc = int("${WALL}"), int("${IQRC}")
alignment, ibin = "${ALIGNMENT}", "${IQTREE}"
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None; best_model = None; mf_wall_direct = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line)
        if m: mf_wall_direct = float(m.group(1))
        m = re.search(r"Best-fit model:\s+(\S+)", line)
        if m: best_model = m.group(1)

# CLX reference (168422813): -59208019.212, F81+F+G4.
# This run establishes the SPR reference; tolerance is 0.1.
CLX_LNL = -59208019.212
EXPECTED_LNL = None  # updated after this run completes

verify = []
if rep_ll is not None and CLX_LNL is not None:
    diff = abs(rep_ll - CLX_LNL)
    verify.append({"file": os.path.basename(alignment),
                   "status": "pass" if diff < 0.5 else "warn",  # wider tolerance CLX vs SPR
                   "expected_clx": CLX_LNL, "reported": rep_ll, "diff_vs_clx": round(diff, 6)})

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "mf_iso_baseline_repro",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 1000000, "n_taxa": 100,
    "threads": threads, "seed": seed,
    "model_finder_only": True,
    "timing": [{"command": f"numactl --localalloc iqtree3 -s alignment_1000000.phy -nt {threads} -seed {seed} -m TESTONLY",
                "time_s": iqwall if iqwall is not None else wall,
                "mf_wall_direct_s": mf_wall_direct}],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "mf_wall": mf_wall_direct,
        "lnL": rep_ll,
        "best_model": best_model,
        "all_pass": iqrc == 0,
    },
    "reference_run_clx": "168422813",
    "reference_lnL_clx": -59208019.212,
    "reference_best_model_clx": "F81+F+G4",
    "reference_mf_wall_clx_s": 10230.229,
    "env": {
        "hostname": sh("hostname"), "date": sh("date -Iseconds"),
        "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
        "cores": int(sh("nproc","0") or 0),
        "iqtree_binary": ibin,
        "iqtree_version": sh(f"{ibin} --version 2>&1 | head -1"),
        "iqtree_md5": sh(f"md5sum {ibin} 2>&1 | awk '{{print \$1}}'"),
        "mpi_nranks": 1,
        "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
                "ncpus": os.environ.get("PBS_NCPUS"), "project": "${PROJECT}"},
    },
    "profile": {"placement": "non_mpi_1node_excl_baseline"},
    "build_tag":  "standard_cpu_opt_merge_intel_vanila",
    "branch":     "n/a (sa0557 standard build)",
    "non_canonical": False,
    "group":      "mf_iso_scaling",
    "note": "ESTABLISHES DNA_1M SPR REFERENCE — update EXPECTED_LNL in mf-iso scripts after this run.",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[baseline] wrote {out_path}")
print(f"[baseline] lnL      = {rep_ll}   (CLX ref: -59208019.212)")
print(f"[baseline] best_mdl = {best_model}  (CLX ref: F81+F+G4)")
print(f"[baseline] MF wall  = {mf_wall_direct} s  (CLX ref: 10230 s)")
print(f"")
print(f"  >>> UPDATE run_mf_iso_dna_1m_{{1,2}}node.sh: EXPECTED_LNL = {rep_ll}")
PYEOF

echo "[baseline] done."
exit "${IQRC}"
