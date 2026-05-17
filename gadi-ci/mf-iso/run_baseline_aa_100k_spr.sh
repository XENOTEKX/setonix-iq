#!/bin/bash
# run_baseline_aa_100k_spr.sh — reproduce the BASELINE OF RECORD (168425673).
#
# Binary: the STANDARD (non-MPI) IQ-TREE 3.1.2 SPR build by sa0557 at
#   /scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3
# Command: identical to 168425673 (1 node, 103 OMP threads, seed=1).
#
# PURPOSE
#   - Confirm the baseline reproduces: MF wall ~405 s, total ~1,170 s.
#   - Capture full hardware/software/binding/binary probe for the run
#     record so future comparisons have unambiguous evidence.
#   - Establish a same-day reference so MF-iso runs aren't compared
#     against a 2-day-old measurement that may have drifted (other
#     workloads on the node, BIOS/microcode updates, etc.).
#
# Run this BEFORE the MF-iso 1-node/2-node so we have a fresh baseline
# in the run record.

#PBS -N mf-iso-aa-100k-baseline
#PBS -P rc29
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/rc29+scratch/um09+scratch/dx61
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
IQTREE="${IQTREE:-/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

THREADS="${THREADS:-103}"
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_100k"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_100k_baseline_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load (mirror 168425673) ─────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load intel-compiler-llvm    2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: baseline binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
if ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "WARNING: baseline binary appears to link libmpi — that's the MPI build, not the standard one." >&2
fi

# ── OMP / runtime (mirror 168425673) ──────────────────────────────────
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=10000
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${ISO_DIR}/tmp"; mkdir -p "${TMPDIR}"

# ── Probe (hardware/software/binary) ───────────────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AA 100K BASELINE Reproduction — non-MPI, 1 node, 103 threads"
echo "║  run_id:       ${RUN_ID}"
echo "║  reference:    168425673 (MF ~405 s, total 1,169.556 s)"
echo "║  binary:       ${IQTREE}"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

probe_hw_sw "${IQTREE}"
probe_env

# ── Run ───────────────────────────────────────────────────────────────
echo "[baseline] running standard binary, ${THREADS} OMP threads"
START_EPOCH=$(date +%s)

numactl --localalloc -- \
    "${IQTREE}" -s "${ALIGNMENT}" -nt "${THREADS}" -seed "${SEED}" \
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
rep_ll = None; iqwall = None; iq_tree_wall = None; best_model = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"Wall-clock time used for tree search:\s+([\d.]+)", line)
        if m: iq_tree_wall = float(m.group(1))
        m = re.search(r"Best-fit model:\s+(\S+)", line)
        if m: best_model = m.group(1)

mf_wall = (iqwall - iq_tree_wall) if (iqwall is not None and iq_tree_wall is not None) else None

EXPECTED_LNL = -7541976.860
verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({"file": os.path.basename(alignment), "status": "pass" if diff < 0.1 else "fail",
                   "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6)})

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "mf_iso_baseline_repro",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
    "threads": threads, "seed": seed,
    "timing": [{
        "command": f"numactl --localalloc iqtree3 -s alignment_100000.phy -nt {threads} -seed {seed}",
        "time_s": iqwall if iqwall is not None else wall,
    }],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "tree_wall": iq_tree_wall,
        "mf_wall_derived": mf_wall,
        "lnL": rep_ll,
        "best_model": best_model,
        "all_pass": iqrc == 0,
    },
    "reference_run":  "168425673",
    "reference_total_wall_s": 1169.556,
    "reference_mf_wall_s_derived": 405.078,  # 1169.556 - 764.478
    "reference_lnL": -7541976.860,
    "env": {
        "hostname": sh("hostname"), "date": sh("date -Iseconds"),
        "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
        "cores": int(sh("nproc","0") or 0), "kernel": sh("uname -r"),
        "omp": {"proc_bind": "close", "places": "cores", "kmp_blocktime": 200,
                "wait_policy": "PASSIVE", "numactl": "--localalloc"},
        "iqtree_binary": ibin,
        "iqtree_version": sh(f"{ibin} --version 2>&1 | head -1"),
        "iqtree_md5":     sh(f"md5sum {ibin} 2>&1 | awk '{{print $1}}'"),
        "mpi_nranks":     1,
        "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
                "ncpus": os.environ.get("PBS_NCPUS"), "project": "${PROJECT}"},
    },
    "profile": {"placement": "non_mpi_1node_excl_baseline"},
    "build_tag":     "standard_cpu_opt_merge_intel_vanila",
    "branch":        "n/a (sa0557 standard build)",
    "non_canonical": False,
    "group":         "mf_iso_scaling",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[baseline] wrote {out_path}")
print(f"[baseline] MF wall (derived) = {mf_wall} s   (reference: 405.078 s)")
print(f"[baseline] Total wall        = {iqwall} s     (reference: 1169.556 s)")
print(f"[baseline] lnL               = {rep_ll}       (reference: -7,541,976.860)")
PYEOF

echo "[baseline] done."
exit "${IQRC}"
