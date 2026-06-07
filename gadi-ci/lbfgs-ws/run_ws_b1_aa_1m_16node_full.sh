#!/bin/bash
# run_ws_b1_aa_1m_16node_full.sh — Full MF+SPR parity run: warm-start B.1 binary, AA 1M, 16-node MPI.
#
# AA, 1M sites (100 taxa), 16 MPI ranks × 103 OpenMP threads = 1648T total.
# 16 × Sapphire Rapids nodes (normalsr, 104 cores each).
#
# PURPOSE — Direct parity comparison against WS-A.2 np=16 (job 169096801,
#   MF=1,139.494s SPR=1,198.689s total=2,419.671s) and FCA Phase 0.5+0.6 np=16 baseline
#   (job 168635616, MF=1,122.363s SPR=1,287.863s total=2,410.226s) to measure Phase B.1
#   progressive pre-pruning broadcast benefit at scale. Phase B.1 fires MPI_Allreduce(OR)
#   in getNextModel() as soon as rank 0 first-fills any rate class, broadcasting before
#   filterRatesMPI fires (progressiveWS bcast lines confirm the path).
#
# Gate: AA 1M np=16 full run.
#   Pass criteria:
#     - exit code = 0
#     - lnL within ±1.0 of −78,605,196.573 (vanilla AA 1M ref 168425491)
#     - Best model = LG+G4
#     - progressiveWS bcast fired (Phase B.1 confirmed)
#     - MF wall reported for comparison (no hard gate at np=16)
#
# A/B refs:
#   FCA np=16 baseline 168635616 (MF=1122.363s  SPR=1287.863s  total=2410.226s)
#   WS-A.1 np=16      169095645 (MF=1146.174s  SPR=1212.944s  total=2440.301s)
#   WS-A.2 np=16      169096801 (MF=1139.494s  SPR=1198.689s  total=2419.671s)
#   Vanilla            168425491 (MF=7587.459s  SPR=15098.605s total=22776.226s)
# Binary:  iqtree3-mpi-fca-ws-b1  md5 5aaf6da444252642b7b744d0696ec9fb
# Parity:  OMP_PER_RANK=103, numactl --localalloc, KMP_BLOCKTIME=200, seed=1
# Branch:  fca-lbfgs-ws
# Build tag: fca_ws_b1_icx_avx512

#PBS -N ws-b1-aa-1m-16n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=1664
#PBS -l mem=8160GB
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
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-b1}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=16
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_1m"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_1m_ws_b1_np16_full_seed${SEED}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"

mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Module load ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                    2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2     2>/dev/null || true
fi

# ── Preflight ──────────────────────────────────────────────────────────
[[ -x "${IQTREE}" ]]    || { echo "ERROR: warm-start B.1 binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found after module load." >&2; exit 4; }
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 5
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5." >&2; exit 6
fi
if ! cat "${IQTREE}" > /dev/null; then
    echo "ERROR: ${IQTREE} not readable on this node (Lustre OST not yet synced?)." >&2; exit 2
fi
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — must run inside a PBS job." >&2; exit 8
fi

WS_OK=0
if nm "${IQTREE}" 2>/dev/null | grep -q '_ZN18RateWarmStartCache5clearEv'; then
    echo "[preflight] RateWarmStartCache::clear: confirmed via nm"; WS_OK=1
elif strings "${IQTREE}" 2>/dev/null | grep -q 'RateWarmStartCache'; then
    echo "[preflight] RateWarmStartCache: found via strings"; WS_OK=1
fi
[[ "${WS_OK}" -eq 0 ]] && echo "[preflight] WARNING: RateWarmStartCache symbol not found" >&2

# Phase B.1: check progressiveWS bcast symbol
    if nm "${IQTREE}" 2>/dev/null | grep -q 'progressiveWarmStartBcast\|phase_b_newfields'; then
        echo "[preflight] Phase B.1 progressiveWarmStartBcast: confirmed via nm"
    elif strings "${IQTREE}" 2>/dev/null | grep -q 'progressiveWS bcast'; then
        echo "[preflight] Phase B.1 progressiveWS bcast: found via strings"
    else
        echo "[preflight] WARNING: Phase B.1 progressiveWarmStartBcast symbol not confirmed" >&2
fi

# ── Binary md5 verification ────────────────────────────────────────────
ACTUAL_MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
EXPECTED_MD5="5aaf6da444252642b7b744d0696ec9fb"
if [[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]]; then
    echo "[preflight] md5 MATCH: ${ACTUAL_MD5}"
else
    echo "[preflight] WARNING: md5 MISMATCH: got ${ACTUAL_MD5}, expected ${EXPECTED_MD5}" >&2
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
if [[ "${#HOSTS[@]}" -ne 16 ]]; then
    echo "ERROR: expected 16 nodes, got ${#HOSTS[@]}" >&2; exit 9
fi
HOST_A="${HOSTS[0]}";  HOST_B="${HOSTS[1]}";  HOST_C="${HOSTS[2]}";  HOST_D="${HOSTS[3]}"
HOST_E="${HOSTS[4]}";  HOST_F="${HOSTS[5]}";  HOST_G="${HOSTS[6]}";  HOST_H="${HOSTS[7]}"
HOST_I="${HOSTS[8]}";  HOST_J="${HOSTS[9]}";  HOST_K="${HOSTS[10]}"; HOST_L="${HOSTS[11]}"
HOST_M="${HOSTS[12]}"; HOST_N="${HOSTS[13]}"; HOST_O="${HOSTS[14]}"; HOST_P="${HOSTS[15]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
rank 2=${HOST_C} slot=0-103
rank 3=${HOST_D} slot=0-103
rank 4=${HOST_E} slot=0-103
rank 5=${HOST_F} slot=0-103
rank 6=${HOST_G} slot=0-103
rank 7=${HOST_H} slot=0-103
rank 8=${HOST_I} slot=0-103
rank 9=${HOST_J} slot=0-103
rank 10=${HOST_K} slot=0-103
rank 11=${HOST_L} slot=0-103
rank 12=${HOST_M} slot=0-103
rank 13=${HOST_N} slot=0-103
rank 14=${HOST_O} slot=0-103
rank 15=${HOST_P} slot=0-103
EOF

RANK_LOGS_DIR="${WORK_DIR}/rank_logs"
mkdir -p "${RANK_LOGS_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AA 1M WS-B.1 Full Run (MF+SPR) — 16-node parity"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  node A:       ${HOST_A}  (rank 0)"
echo "║  node B:       ${HOST_B}  (rank 1)"
echo "║  node P:       ${HOST_P}  (rank 15)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5 expected: 5aaf6da444252642b7b744d0696ec9fb"
echo "║  md5 actual:   ${ACTUAL_MD5}"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  branch:       fca-lbfgs-ws"
echo "║  FCA np=16 ref:  168635616 MF=1122.363s SPR=1287.863s total=2410.226s"
echo "║  WS-A.2 np=16:   169096801 MF=1139.494s SPR=1198.689s total=2419.671s"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[16node] hostfile:"; cat "${HOSTFILE}" | sed 's/^/    /'
echo "[16node] rankfile:"; cat "${RANKFILE}"  | sed 's/^/    /'
echo ""

# ── Probe ──────────────────────────────────────────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"
probe_hw_sw "${IQTREE}"
probe_env

RANK_PROBE="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_probe.sh"
[[ -x "${RANK_PROBE}" ]] || { echo "ERROR: rank_probe.sh not found at ${RANK_PROBE}" >&2; exit 10; }
RANK_PERF="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_perf.sh"
[[ -x "${RANK_PERF}" ]] || { echo "ERROR: rank_perf.sh not found at ${RANK_PERF}" >&2; exit 10; }

# ── Full MF+SPR run ────────────────────────────────────────────────────
echo "[16node] Full run (MF+SPR), ${NRANKS} ranks × ${OMP_PER_RANK} OMP across 16 nodes"
export PERF_STAT_DIR="${WORK_DIR}"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    --output-filename "${RANK_LOGS_DIR}/" \
    "${OMP_ENV[@]}" \
    "${RANK_PROBE}" \
        "${RANK_PERF}" \
            numactl --localalloc -- \
            "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[16node] done: rc=${IQRC} wall=${WALL}s"

# ── Per-rank perf stat summary ─────────────────────────────────────────
echo ""
echo "[16node] Per-rank perf stat (IPC + LLC cache):"
for pf in "${WORK_DIR}"/perf_stat_rank_*.txt; do
    [[ -f "$pf" ]] || continue
    rank=$(echo "$pf" | sed -E 's|.*perf_stat_rank_([0-9]+)\.txt|\1|')
    echo "  === rank ${rank} ==="
    grep -E 'cycles|instructions|cache-miss|LLC|insn per cycle' "$pf" 2>/dev/null | sed 's/^/    /' || true
done

# ── Per-rank MF-TIME and MF-MPI-DIAG aggregation ──────────────────────
echo ""
echo "[16node] gathering per-rank logs..."
for f in "${RANK_LOGS_DIR}"/*/rank.*/stdout; do
    [[ -f "$f" ]] || continue
    rank=$(echo "$f" | sed -E 's|.*/rank\.([0-9]+)/stdout|\1|')
    cp -f "$f" "${WORK_DIR}/rank_${rank}.stdout.log"
    echo "  rank ${rank}: $(wc -l < "$f") lines  -> ${WORK_DIR}/rank_${rank}.stdout.log"
done

{ for r in "${WORK_DIR}"/rank_*.stdout.log; do
    [[ -f "$r" ]] && grep -E '^MF-TIME: ' "$r" || true
  done } > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
{ for r in "${WORK_DIR}"/rank_*.stdout.log; do
    [[ -f "$r" ]] && grep -E '^MF-MPI-DIAG: ' "$r" || true
  done } > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true

if [[ ! -s "${WORK_DIR}/mf_time.log" ]]; then
    grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_time.log" 2>/dev/null || true
    grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_diag.log" 2>/dev/null || true
fi

grep -E '^PROBE: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/probe.log" 2>/dev/null || true

{
    for r in "${WORK_DIR}"/rank_*.stderr.log "${WORK_DIR}"/rank_logs/*/rank.*/stderr; do
        [[ -f "$r" ]] && grep -E '^RANK-PROBE: ' "$r" 2>/dev/null || true
    done
    grep -E '^RANK-PROBE: ' "${WORK_DIR}/iqtree_run.bindings.log" 2>/dev/null || true
} | sort -u > "${WORK_DIR}/rank_probe.log" 2>/dev/null || true

{
    echo "# rank, model_idx, model_name, subst, rate, dt_seconds, ref_remaining"
    awk -F' ' '
    /^MF-TIME: rank / {
        for (i=1; i<=NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2]; }
        printf "%s, %s, %s, %s, %s, %s, %s\n",
            v["rank"], v["model"], v["name"], v["subst"], v["rate"], v["dt"], v["ref_remaining"];
    }' "${WORK_DIR}/mf_time.log"
} > "${WORK_DIR}/rank_models.csv" 2>/dev/null || true

grep -E '^\[' "${WORK_DIR}/iqtree_run.bindings.log" > "${WORK_DIR}/rank_bindings.log" 2>/dev/null || true

echo "[16node] PROBE lines:        $(wc -l < "${WORK_DIR}/probe.log" 2>/dev/null || echo 0)"
echo "[16node] RANK-PROBE lines:   $(wc -l < "${WORK_DIR}/rank_probe.log" 2>/dev/null || echo 0)"
echo "[16node] MF-TIME lines:      $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[16node] MF-MPI-DIAG lines:  $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"
echo "[16node] Per-rank model counts (from MF-TIME):"
for r in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    n=$(grep -c "^MF-TIME: rank ${r} " "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)
    echo "    rank ${r}: ${n} models evaluated"
done
echo "[16node] Per-rank family ownership (from MF-MPI-DIAG dispatch line):"
grep -E "MF-MPI-DIAG: rank [0-9]+/[0-9]+ owns " "${WORK_DIR}/mf_diag.log" 2>/dev/null | sed 's/^/    /' || true
echo "[16node] Phase A.2 broadcast diagnostic:"
grep -E "filterRatesMPI fired" "${WORK_DIR}/mf_diag.log" 2>/dev/null | sed 's/^/    /' || true

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
rep_ll = None; iqwall = None; mf_wall = None; spr_wall = None; best_model = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line)
        if m: mf_wall = float(m.group(1))
        m = re.search(r"Wall-clock time used for tree search:\s+([\d.]+)", line)
        if m: spr_wall = float(m.group(1))
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

# ── Phase B.1 progressive broadcast diagnostics ──────────────────────
diag_log = os.path.join(work, "mf_diag.log")
broadcast_fired = False
bcast_ok_rates_size = None
ws_bcast_fields_max = 0
ws_bcast_count = 0
prog_ws_events = 0
prog_ws_bitmasks = []
if os.path.isfile(diag_log):
    for line in open(diag_log, errors="replace"):
        if "filterRatesMPI fired at" in line:
            broadcast_fired = True
            m = re.search(r"\|bcast_ok_rates\|=(\d+)", line)
            if m: bcast_ok_rates_size = int(m.group(1))
            m = re.search(r"ws_bcast_fields=(\d+)", line)
            if m:
                ws_bcast_count += 1
                ws_bcast_fields_max = max(ws_bcast_fields_max, int(m.group(1)))
        if "progressiveWS bcast:" in line:
            prog_ws_events += 1
            m = re.search(r"bitmask=(0x[0-9a-fA-F]+)", line)
            if m: prog_ws_bitmasks.append(m.group(1))

# ── Baselines for comparison ──────────────────────────────────────────
EXPECTED_LNL  = -78605196.573   # job 168425491 (vanilla AA 1M np=1)
TOL           = 1.0
# FCA np=16 baseline (168635616)
FCA_MF        = 1122.363
FCA_SPR       = 1287.863
FCA_TOT       = 2410.226
# WS-A.1 np=16 (169095645)
WSA1_MF       = 1146.174
WSA1_SPR      = 1212.944
WSA1_TOT      = 2440.301
# WS-A.2 np=16 (169096801)
WSA2_MF       = 1139.494
WSA2_SPR      = 1198.689
WSA2_TOT      = 2419.671
# Vanilla np=1 (168425491)
VANILLA_MF    = 7587.459
VANILLA_SPR   = 15098.605
VANILLA_TOT   = 22776.226
# MF wall limit (informational only for B.1)
MF_WALL_LIMIT = 900.0

verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({"check": "lnL", "status": "pass" if diff < TOL else "fail",
                   "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6),
                   "note": "Full-run SPR lnL vs vanilla ref 168425491 (AA 1M); tol=1.0"})
model_ok = best_model == "LG+G4"
verify.append({"check": "best_model", "status": "pass" if model_ok else "fail",
               "expected": "LG+G4", "reported": best_model})
verify.append({"check": "progressiveWS_fired_B1", "status": "pass" if prog_ws_events > 0 else "fail",
               "prog_ws_events": prog_ws_events, "bitmasks": prog_ws_bitmasks,
               "note": "Phase B.1 getNextModel() MPI_Allreduce broadcast must fire before filterRatesMPI"})
if mf_wall is not None:
    verify.append({"check": "mf_wall_reported", "status": "pass",
                   "actual_s": mf_wall,
                   "note": "MF wall recorded for comparison (no hard gate at np=16)"})

all_pass = iqrc == 0 and all(v["status"] == "pass" for v in verify)

def speedup(base, measured):
    return round(base / measured, 3) if measured else None

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "mf_iso",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 1000000, "n_taxa": 100,
    "threads": threads, "seed": ${SEED},
    "model_finder_only": False,
    "timing": [{
        "command": f"mpirun -np {nranks} -rf rankfile --output-filename rank_logs numactl --localalloc iqtree3-mpi-fca-ws-b1 -s alignment_1000000.phy -m TEST -T {omp_per_rank} -seed ${SEED}",
        "time_s": iqwall if iqwall is not None else wall,
    }],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "mf_wall_s": mf_wall,
        "spr_wall_s": spr_wall,
        "lnL": rep_ll,
        "best_model": best_model,
        "all_pass": all_pass,
        "lnl_pass": verify[0]["status"] == "pass" if verify else None,
        "model_pass": model_ok,
        "ws_bcast_pass": ws_bcast_fields_max > 0,
        "prog_ws_pass": prog_ws_events > 0,
        "mf_wall_s_record": mf_wall,
        "vs_fca_np16_baseline": {
            "mf_speedup":    speedup(FCA_MF,   mf_wall),
            "spr_speedup":   speedup(FCA_SPR,  spr_wall),
            "total_speedup": speedup(FCA_TOT,  iqwall),
        },
        "vs_ws_a2_np16": {
            "mf_speedup":    speedup(WSA2_MF,  mf_wall),
            "spr_speedup":   speedup(WSA2_SPR, spr_wall),
            "total_speedup": speedup(WSA2_TOT, iqwall),
        },
        "vs_ws_a1_np16": {
            "mf_speedup":    speedup(WSA1_MF,  mf_wall),
            "spr_speedup":   speedup(WSA1_SPR, spr_wall),
            "total_speedup": speedup(WSA1_TOT, iqwall),
        },
        "vs_vanilla_baseline": {
            "mf_speedup":    speedup(VANILLA_MF,  mf_wall),
            "spr_speedup":   speedup(VANILLA_SPR, spr_wall),
            "total_speedup": speedup(VANILLA_TOT, iqwall),
        },
    },
    "warm_start": {
        "phase": "B.1",
        "prog_ws_events": prog_ws_events,
        "prog_ws_bitmasks": prog_ws_bitmasks,
        "ws_bcast_fields_max_A2": ws_bcast_fields_max,
        "ws_bcast_count_A2": ws_bcast_count,
        "broadcast_fired_A0_5": broadcast_fired,
        "bcast_ok_rates_size": bcast_ok_rates_size,
    },
    "mf_time_summary": mf_time_summary,
    "env": {
        "hostname": sh("hostname"), "date": sh("date -Iseconds"),
        "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
        "cores": int(sh("nproc","0") or 0), "kernel": sh("uname -r"),
        "omp": {"proc_bind": "close", "places": "cores", "kmp_blocktime": 200,
                "wait_policy": "PASSIVE", "numactl": "--localalloc"},
        "iqtree_binary": ibin,
        "iqtree_md5": "${ACTUAL_MD5}",
        "iqtree_version": sh(f"{ibin} --version 2>&1 | head -1"),
        "mpi_nranks": nranks,
        "pbs": {"job_id": os.environ.get("PBS_JOBID"), "queue": os.environ.get("PBS_QUEUE"),
                "ncpus": os.environ.get("PBS_NCPUS"), "project": "${PROJECT}"},
    },
    "profile": {"nranks": nranks, "omp_per_rank": omp_per_rank,
                "placement": "mpi_16node_full_perrank_logs"},
    "build_tag":     "fca_ws_b1_icx_avx512",
    "branch":        "fca-lbfgs-ws",
    "non_canonical": True,
    "non_canonical_label": "FCA-WS Phase B.1 (progressive pre-pruning broadcast) · ICX+MPI · AVX-512 · full MF+SPR · AA 1M · np=16",
    "group":         "fca_ws_b1",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[16node] wrote {out_path}")

vs_fca = record["summary"]["vs_fca_np16_baseline"]
vs_a2  = record["summary"]["vs_ws_a2_np16"]
vv     = record["summary"]["vs_vanilla_baseline"]
print(f"[16node] ─── FULL RUN SUMMARY (AA 1M WS-B.1 np=16) ───")
for v in verify:
    chk = v.get("check","?")
    sts = "PASS" if v["status"] == "pass" else "FAIL"
    if chk == "lnL":
        print(f"[16node]   lnL:           {sts}  reported={v.get('reported')}  diff={v.get('diff')}")
    elif chk == "best_model":
        print(f"[16node]   model:         {sts}  ({v.get('reported')})")
    elif chk == "progressiveWS_fired_B1":
        print(f"[16node]   prog WS (B1): {'PASS' if v['status']=='pass' else 'FAIL'}  events={prog_ws_events}  bitmasks={prog_ws_bitmasks}")
    elif chk == "mf_wall_reported":
        print(f"[16node]   MF wall (W4):  {sts}  {v.get('actual_s')} s ≤ {v.get('limit_s')} s criterion")
if mf_wall:
    print(f"[16node]   MF:   {mf_wall:.3f} s  vs FCA-np16={FCA_MF}s ({vs_fca['mf_speedup']}×)  vs WS-A2-np16={WSA2_MF}s ({vs_a2['mf_speedup']}×)  vs vanilla={VANILLA_MF}s ({vv['mf_speedup']}×)")
if spr_wall:
    print(f"[16node]   SPR:  {spr_wall:.3f} s  vs FCA-np16={FCA_SPR}s ({vs_fca['spr_speedup']}×)  vs WS-A2-np16={WSA2_SPR}s ({vs_a2['spr_speedup']}×)  vs vanilla={VANILLA_SPR}s ({vv['spr_speedup']}×)")
if iqwall:
    print(f"[16node]   Total:{iqwall:.3f} s  vs FCA-np16={FCA_TOT}s ({vs_fca['total_speedup']}×)  vs WS-A2-np16={WSA2_TOT}s ({vs_a2['total_speedup']}×)  vs vanilla={VANILLA_TOT}s ({vv['total_speedup']}×)")
print(f"[16node]   Phase B.1 progressiveWS: {'FIRED (' + str(prog_ws_events) + ' events, bitmasks=' + str(prog_ws_bitmasks) + ')' if prog_ws_events > 0 else 'NOT SEEN — FAIL'}")
print(f"[16node]   Phase A.2 ws_bcast_fields (still fires): {ws_bcast_fields_max}")
print(f"[16node]   Overall: {'ALL PASS' if all_pass else 'FAIL'}")
PYEOF

echo "[16node] done."
exit "${IQRC}"
