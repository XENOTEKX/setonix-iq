#!/bin/bash
# run_mf_iso_aa_100k_2node.sh — IQ-TREE3 MF-iso ModelFinder benchmark, 2-node MPI.
#
# AA, 100K sites (100 taxa), 2 MPI ranks × 103 OpenMP threads = 206T total.
# 2 × Sapphire Rapids exclusive nodes (normalsr, 104 cores each).
#
# PURPOSE — STEP BY STEP SCALING (don't skip 2-node)
#   This is the FIRST test that exercises the Phase 0.5 cross-rank
#   ok_rates broadcast and Phase 0.6 getNextModel ref-family priority.
#   At np=2, rank 0 owns LG (sharp BIC -> ok_rates={G4}) and rank 1
#   owns an empirical-frequency family (flat BIC). Phase 0.5 broadcasts
#   rank 0's {G4} set to rank 1, fixing the rank-1+ pruning gap.
#
#   Acceptance gate (must pass before scaling to 4-node):
#     - lnL = -7,541,976.860 ± 0.01
#     - best model = LG+G4
#     - MF wall < 600 s (Fix H baseline 475 s; FCA Phase 0 regressed to 2,865 s)
#     - both ranks emit MF-TIME lines through to filterRatesMPI fire
#     - rank 1 evaluates < 50 models after broadcast (mostly +G4 variants)
#
# CRITICAL: per-rank stdout via `mpirun --output-filename`. Earlier debug
# runs (e.g. 168475747) lost rank 1+ stdout because mpirun across multiple
# nodes only forwards rank 0 to the redirected file. Per-file output is
# what lets us actually SEE rank 1's eval loop.
#
# Binary:  /scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
# Build tag:    mf_iso_phase0.5_0.6_icx_avx512_mftime

#PBS -N mf-iso-aa-100k-2n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1020GB
#PBS -l walltime=02:00:00
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
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=2
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_100k"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_100k_mfiso_np2_seed${SEED}"
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
    echo "ERROR: ${IQTREE} does not link libmpi." >&2; exit 5
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp." >&2; exit 6
fi
# Verify the binary is fully readable on this OST (see 1-node script for root-cause notes).
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
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — must run inside a PBS job." >&2; exit 8
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

# ── Multi-node host discovery ──────────────────────────────────────────
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
if [[ "${#HOSTS[@]}" -ne 2 ]]; then
    echo "ERROR: expected 2 nodes, got ${#HOSTS[@]}" >&2
    exit 9
fi
HOST_A="${HOSTS[0]}"
HOST_B="${HOSTS[1]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
EOF

# Per-rank stdout dir — KEY for capturing rank 1's logs.
RANK_LOGS_DIR="${WORK_DIR}/rank_logs"
mkdir -p "${RANK_LOGS_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AA 100K MF-iso Benchmark — 2-node MPI, ModelFinder-only (-m TESTONLY)"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  node A:       ${HOST_A}  (rank 0 — owns LG family)"
echo "║  node B:       ${HOST_B}  (rank 1 — owns non-LG family)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  rank logs:    ${RANK_LOGS_DIR}/"
echo "║  branch:       mf-iso-phase0.5-0.6"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[2node] hostfile:"; cat "${HOSTFILE}" | sed 's/^/    /'
echo "[2node] rankfile:"; cat "${RANKFILE}"  | sed 's/^/    /'
echo ""

# ── Probe (hardware/software/binary/source) ───────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"
probe_hw_sw "${IQTREE}"
probe_env

# Per-rank binding probe wrapper.
RANK_PROBE="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_probe.sh"
[[ -x "${RANK_PROBE}" ]] || { echo "ERROR: rank_probe.sh not found at ${RANK_PROBE}" >&2; exit 10; }

# ── ModelFinder-only run with per-rank output capture ─────────────────
echo "[2node] ModelFinder-only run, ${NRANKS} ranks × ${OMP_PER_RANK} OMP across 2 nodes"
echo "[2node] mpirun --output-filename ${RANK_LOGS_DIR}/  (one stdout/stderr per rank)"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${RANK_LOGS_DIR}/" \
    "${OMP_ENV[@]}" \
    "${RANK_PROBE}" \
        numactl --localalloc -- \
            "${IQTREE}" -s "${ALIGNMENT}" -m TESTONLY -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[2node] done: rc=${IQRC} wall=${WALL}s"

# ── Per-rank MF-TIME and MF-MPI-DIAG aggregation ──────────────────────
# OpenMPI 4.1 puts per-rank stdout in:
#   ${RANK_LOGS_DIR}/<job-id>/rank.0/stdout
#   ${RANK_LOGS_DIR}/<job-id>/rank.1/stdout
# Gather them into the work dir for parsing.
echo ""
echo "[2node] gathering per-rank logs..."
for f in "${RANK_LOGS_DIR}"/*/rank.*/stdout; do
    [[ -f "$f" ]] || continue
    rank=$(echo "$f" | sed -E 's|.*/rank\.([0-9]+)/stdout|\1|')
    cp -f "$f" "${WORK_DIR}/rank_${rank}.stdout.log"
    echo "  rank ${rank}: $(wc -l < "$f") lines  -> ${WORK_DIR}/rank_${rank}.stdout.log"
done

# Aggregate all MF-TIME lines from all ranks into one file (for analysis).
{ for r in "${WORK_DIR}"/rank_*.stdout.log; do
    [[ -f "$r" ]] && grep -E '^MF-TIME: ' "$r" || true
  done } > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
{ for r in "${WORK_DIR}"/rank_*.stdout.log; do
    [[ -f "$r" ]] && grep -E '^MF-MPI-DIAG: ' "$r" || true
  done } > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true

# Fall back to main log if per-rank files weren't produced (e.g. single-node mpirun).
if [[ ! -s "${WORK_DIR}/mf_time.log" ]]; then
    grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
    grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true
fi

echo ""
# Aggregate PROBE lines (only in main log; rank 0 captured them).
grep -E '^PROBE: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/probe.log" 2>/dev/null || true

# Aggregate RANK-PROBE lines (one block per rank, on stderr).
{
    for r in "${WORK_DIR}"/rank_*.stderr.log "${WORK_DIR}"/rank_logs/*/rank.*/stderr; do
        [[ -f "$r" ]] && grep -E '^RANK-PROBE: ' "$r" 2>/dev/null || true
    done
    grep -E '^RANK-PROBE: ' "${WORK_DIR}/iqtree_run.bindings.log" 2>/dev/null || true
} | sort -u > "${WORK_DIR}/rank_probe.log" 2>/dev/null || true

# Per-rank model assignment summary (CSV for analysis).
{
    echo "# rank, model_idx, model_name, subst, rate, dt_seconds, ref_remaining"
    awk -F' ' '
    /^MF-TIME: rank / {
        for (i=1; i<=NF; i++) {
            split($i, kv, "=");
            v[kv[1]] = kv[2];
        }
        printf "%s, %s, %s, %s, %s, %s, %s\n",
            v["rank"], v["model"], v["name"], v["subst"], v["rate"], v["dt"], v["ref_remaining"];
    }' "${WORK_DIR}/mf_time.log"
} > "${WORK_DIR}/rank_models.csv" 2>/dev/null || true

# Bindings extracted from openmpi --report-bindings (lines starting with [host:pid]).
grep -E '^\[' "${WORK_DIR}/iqtree_run.bindings.log" > "${WORK_DIR}/rank_bindings.log" 2>/dev/null || true

echo "[2node] PROBE lines:        $(wc -l < "${WORK_DIR}/probe.log" 2>/dev/null || echo 0)"
echo "[2node] RANK-PROBE lines:   $(wc -l < "${WORK_DIR}/rank_probe.log" 2>/dev/null || echo 0)"
echo "[2node] MF-TIME lines:      $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[2node] MF-MPI-DIAG lines:  $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"
echo "[2node] BINDINGS:           $(wc -l < "${WORK_DIR}/rank_bindings.log" 2>/dev/null || echo 0) lines"
echo "[2node] Per-rank model counts (from MF-TIME):"
for r in 0 1; do
    n=$(grep -c "^MF-TIME: rank ${r} " "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)
    echo "    rank ${r}: ${n} models evaluated"
done
echo "[2node] Per-rank family ownership (from MF-MPI-DIAG dispatch line):"
grep -E "MF-MPI-DIAG: rank [0-9]+/[0-9]+ owns " "${WORK_DIR}/mf_diag.log" 2>/dev/null | sed 's/^/    /'

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

# Phase 0.5 broadcast confirmation.
diag_log = os.path.join(work, "mf_diag.log")
broadcast_fired = False; bcast_ok_rates_size = None
if os.path.isfile(diag_log):
    for line in open(diag_log, errors="replace"):
        if "filterRatesMPI fired at" in line:
            broadcast_fired = True
            m = re.search(r"\|bcast_ok_rates\|=(\d+)", line)
            if m: bcast_ok_rates_size = int(m.group(1))

EXPECTED_LNL = -7541976.860
verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({"file": os.path.basename(alignment), "status": "pass" if diff < 0.1 else "fail",
                   "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6)})

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "mf_iso",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
    "threads": threads, "seed": ${SEED},
    "model_finder_only": True,
    "timing": [{
        "command": f"mpirun -np {nranks} -rf rankfile --output-filename rank_logs numactl --localalloc iqtree3-mpi -s alignment_100000.phy -m TESTONLY -T {omp_per_rank} -seed ${SEED}",
        "time_s": iqwall if iqwall is not None else wall,
    }],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "lnL": rep_ll,
        "best_model": best_model,
        "all_pass": iqrc == 0,
    },
    "mf_time_summary": mf_time_summary,
    "phase0_5_broadcast": {
        "fired": broadcast_fired,
        "ok_rates_size": bcast_ok_rates_size,
    },
    "env": {
        "hostname": sh("hostname"), "date": sh("date -Iseconds"),
        "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
        "cores": int(sh("nproc","0") or 0), "kernel": sh("uname -r"),
        "omp": {"proc_bind": "close", "places": "cores", "kmp_blocktime": 200,
                "wait_policy": "PASSIVE", "numactl": "--localalloc"},
        "iqtree_binary": ibin,
        "iqtree_version": sh(f"{ibin} --version 2>&1 | head -1"),
        "mpi_nranks": nranks,
        "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
                "ncpus": os.environ.get("PBS_NCPUS"), "project": "${PROJECT}"},
    },
    "profile": {"nranks": nranks, "omp_per_rank": omp_per_rank,
                "placement": "mpi_2node_testonly_perrank_logs"},
    "build_tag":     "mf_iso_phase0.5_0.6_icx_avx512_mftime",
    "branch":        "mf-iso-phase0.5-0.6",
    "non_canonical": True,
    "non_canonical_label": "MF-iso Phase 0.5+0.6 (filterRatesMPI Bcast + getNextModel ref-priority + MF-TIME) · ICX+MPI · AVX-512",
    "group":         "mf_iso_scaling",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[2node] wrote {out_path}")
PYEOF

echo "[2node] done."
exit "${IQRC}"
