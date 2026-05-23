#!/bin/bash
# run_ws_a1_aa_100k_1node_w1.sh — W1 correctness gate: warm-start A.1 binary, AA 100K, 1-node MPI.
#
# AA, 100K sites (100 taxa), 1 MPI rank × 103 OpenMP threads.
#
# PURPOSE — W1 (gate 1 of the Phase A.1 validation ladder):
#   Confirm that the cross-model warm-start cache (Phase A.1, local-only)
#   does NOT alter correctness at np=1 on the AA 100K benchmark.
#   At np=1 the warm-start cache populates from models evaluated on the
#   single rank and injects prior-fit α / p_invar / props / rates into
#   subsequent same-rate-class models. FCA dispatch is a no-op at np=1
#   (filterRatesMPI never fires). The only active change vs the baseline
#   binary is the warm-start injection path.
#
# Pass criteria (doc §5.7 W1):
#   - lnL within ±0.5 of baseline ref (168425673): −7,541,976.860
#   - Best model = LG+G4
#   - MF wall ≤ 380 s  (baseline 168425673: ~405 s; warm-start expected to reduce this)
#   - exit code = 0
#
# Binary:  /scratch/dx61/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi-fca-ws-a1
#          md5 fa9ee60103a1a922505cf4dfa26a2fca (built 2026-05-23 14:00, ~200 lines added)
#          Baseline copy: iqtree3-mpi-fca-lbfgs-ws (md5 a103bc6c..., unchanged)
# Branch:  fca-lbfgs-ws (harness) + fca-lbfgs-ws (source, from test_MF2 @ 9603247f)
# Parity:  OMP_PER_RANK=103, numactl --localalloc, KMP_BLOCKTIME=200, seed=1
#          — exact match to run_mf_iso_aa_100k_1node.sh baseline conditions
# A/B ref: run_mf_iso_aa_100k_1node.sh (baseline binary iqtree3-mpi, CHANGELOG (bs))
#
# Build tag: fca_ws_a1_icx_avx512
# Related:   CHANGELOG (bt), research/lbfgs-and-warmstart-implementation.md §5.7, §12.7

#PBS -N ws-a1-aa-100k-w1
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
IQTREE="${IQTREE:-${ISO_DIR}/build-mpi-iso/iqtree3-mpi-fca-ws-a1}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_100k"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_100k_ws_a1_np1_w1_seed${SEED}"
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
[[ -x "${IQTREE}" ]]    || { echo "ERROR: warm-start binary not found: ${IQTREE}" >&2; exit 2; }
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

# Verify warm-start A.1 symbols are present.
WS_OK=0
if nm "${IQTREE}" 2>/dev/null | grep -q '_ZN18RateWarmStartCache5clearEv'; then
    echo "[preflight] RateWarmStartCache::clear: confirmed via nm"
    WS_OK=1
elif strings "${IQTREE}" 2>/dev/null | grep -q 'RateWarmStartCache'; then
    echo "[preflight] RateWarmStartCache: found via strings (nm failed)"
    WS_OK=1
fi
if [[ "${WS_OK}" -eq 0 ]]; then
    echo "[preflight] WARNING: RateWarmStartCache symbol not found — may be baseline binary, not ws-a1" >&2
fi

# Also confirm FCA symbols are present (inherited from baseline).
if ! nm "${IQTREE}" 2>/dev/null | grep -q '_ZN17CandidateModelSet14filterRatesMPIEi'; then
    if ! strings "${IQTREE}" 2>/dev/null | grep -q 'filterRatesMPI'; then
        echo "[preflight] WARNING: filterRatesMPI not verified (nm + strings both failed)" >&2
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
echo "║  AA 100K WS-A.1 W1 Correctness Gate — 1-node TESTONLY"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5 expected: fa9ee60103a1a922505cf4dfa26a2fca"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  branch:       fca-lbfgs-ws"
echo "║  baseline ref: 168425673 (MF ~405 s, lnL −7,541,976.860, LG+G4)"
echo "║  W1 MF target: ≤ 380 s"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Probe (hardware/software/binary/source) ───────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"
probe_hw_sw "${IQTREE}"
probe_env

RANK_PROBE="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_probe.sh"
[[ -x "${RANK_PROBE}" ]] || { echo "ERROR: rank_probe.sh not found at ${RANK_PROBE}" >&2; exit 9; }

# ── ModelFinder-only run (W1 correctness gate) ────────────────────────
echo "[w1] ModelFinder-only run (TESTONLY), ${NRANKS} rank × ${OMP_PER_RANK} OMP"
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
echo "[w1] done: rc=${IQRC} wall=${WALL}s"

# Extract log lines for downstream analysis.
grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_time.log"     || true
grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_diag.log"     || true
grep -E '^WS-HIT: '      "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/ws_hit.log"      || true
grep -E '^WS-MISS: '     "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/ws_miss.log"     || true
grep -E '^PROBE: '       "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/probe.log"       2>/dev/null || true

echo "[w1] MF-TIME lines:     $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[w1] MF-MPI-DIAG lines: $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"
echo "[w1] WS-HIT lines:      $(wc -l < "${WORK_DIR}/ws_hit.log"  2>/dev/null || echo 0)"
echo "[w1] WS-MISS lines:     $(wc -l < "${WORK_DIR}/ws_miss.log" 2>/dev/null || echo 0)"
echo "[w1] PROBE lines:       $(wc -l < "${WORK_DIR}/probe.log"   2>/dev/null || echo 0)"

# Per-rank model list (rank 0 only at np=1).
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
rep_ll = None; iqwall = None; mf_wall = None; best_model = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line)
        if m: mf_wall = float(m.group(1))
        m = re.search(r"Best-fit model:\s+(\S+)", line)
        if m: best_model = m.group(1)

# Warm-start hit/miss summary.
ws_hits = ws_misses = 0
for fname, key in [("ws_hit.log", "ws_hits"), ("ws_miss.log", "ws_misses")]:
    p = os.path.join(work, fname)
    if os.path.isfile(p):
        n = sum(1 for _ in open(p, errors="replace") if _.strip())
        if key == "ws_hits": ws_hits = n
        else: ws_misses = n

# Per-rank MF-TIME aggregation.
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

EXPECTED_LNL = -7541976.860
TOL = 0.5
verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({
        "file": os.path.basename(alignment),
        "status": "pass" if diff < TOL else "fail",
        "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6),
        "note": "W1: MF lnL vs baseline ref 168425673; tol=0.5",
    })
model_ok = best_model == "LG+G4"
mf_ok = mf_wall is not None and mf_wall <= 380.0

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "mf_iso",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
    "threads": threads, "seed": ${SEED},
    "model_finder_only": True,
    "timing": [{
        "command": f"mpirun -np {nranks} numactl --localalloc iqtree3-mpi-fca-ws-a1 -s alignment_100000.phy -m TESTONLY -T {omp_per_rank} -seed ${SEED}",
        "time_s": iqwall if iqwall is not None else wall,
    }],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "mf_wall_s": mf_wall,
        "lnL": rep_ll,
        "best_model": best_model,
        "all_pass": iqrc == 0,
        "w1_lnl_pass": verify[0]["status"] == "pass" if verify else None,
        "w1_model_pass": model_ok,
        "w1_mf_wall_pass": mf_ok,
    },
    "warm_start": {
        "phase": "A.1",
        "ws_hits": ws_hits,
        "ws_misses": ws_misses,
        "hit_rate": round(ws_hits / (ws_hits + ws_misses), 4) if (ws_hits + ws_misses) > 0 else None,
    },
    "mf_time_summary": mf_time_summary,
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
                "placement": "mpi_1node_excl_testonly"},
    "build_tag":     "fca_ws_a1_icx_avx512",
    "branch":        "fca-lbfgs-ws",
    "non_canonical": True,
    "non_canonical_label": "FCA-WS Phase A.1 (cross-model warm-start, local-only) · ICX+MPI · AVX-512",
    "group":         "fca_ws_a1",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[w1] wrote {out_path}")

# Print W1 gate summary.
print(f"[w1] ─── W1 GATE SUMMARY ───")
for v in verify:
    status = v['status'].upper()
    print(f"[w1]   lnL: {status}  reported={v['reported']}  expected={v['expected']}  diff={v['diff']}")
print(f"[w1]   best_model: {'PASS' if model_ok else 'FAIL'}  ({best_model})")
mf_str = f"{mf_wall:.3f} s" if mf_wall is not None else "N/A"
print(f"[w1]   MF wall: {'PASS' if mf_ok else 'FAIL (>=380s)'}  ({mf_str})")
ws_total = ws_hits + ws_misses
print(f"[w1]   warm-start hits: {ws_hits}/{ws_total}  ({round(ws_hits/(ws_total)*100,1) if ws_total>0 else 'N/A'}%)")
PYEOF

echo "[w1] done."
exit "${IQRC}"
