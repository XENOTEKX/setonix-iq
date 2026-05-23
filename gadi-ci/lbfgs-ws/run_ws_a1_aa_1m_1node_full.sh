#!/bin/bash
# run_ws_a1_aa_1m_1node_full.sh — Full MF+SPR parity run: warm-start A.1 binary, AA 1M, 1-node MPI.
#
# AA, 1M sites (100 taxa), 1 MPI rank × 103 OpenMP threads.
#
# PURPOSE — Direct parity comparison against FCA Phase 0.5+0.6 np=1 baseline (job 168913089,
#   MF=5,119.929s SPR=15,060.551s total=20,180.480s) to measure warm-start A.1 benefit on AA 1M:
#     1. Does the warm-start cache reduce MF wall time beyond FCA baseline?
#     2. Does SPR remain unaffected?
#     3. End-to-end lnL correctness vs baseline 168425491.
#
# At np=1 the warm-start cache populates sequentially (intra-rank only — no Phase A.2 MPI broadcast
# yet). The gain here is from re-using convergence params within each rate class, e.g. LG+G4 seed →
# WAG+G4, and +G4 best α → +G4 next-family start.  The ~37× larger alignment (vs AA 100K) means each
# BFGS call costs ~37× more function evaluations, so even 1–2 fewer BFGS restarts per model saves
# significant absolute time.
#
# A/B ref: FCA np=1 baseline 168913089 (MF=5119.929s SPR=15060.551s total=20180.480s)
#          Vanilla baseline  168425491 (MF=~7587s    SPR=15098.605s total=22776.226s)
# Binary:  iqtree3-mpi-fca-ws-a1  md5 fa9ee60103a1a922505cf4dfa26a2fca
# Parity:  OMP_PER_RANK=103, numactl --localalloc, KMP_BLOCKTIME=200, seed=1
# Branch:  fca-lbfgs-ws
# Build tag: fca_ws_a1_icx_avx512
# Related:   CHANGELOG (bz) FCA baseline, (bx) WS-A.1 AA 100K, research/lbfgs-and-warmstart-implementation.md §12.7

#PBS -N ws-a1-aa-1m-full
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=08:00:00
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
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/mf_iso/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_1m"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_1m_ws_a1_np1_full_seed${SEED}"
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
    echo "ERROR: ${IQTREE} not readable on this node (Lustre OST not yet synced?)." >&2; exit 2
fi

WS_OK=0
if nm "${IQTREE}" 2>/dev/null | grep -q '_ZN18RateWarmStartCache5clearEv'; then
    echo "[preflight] RateWarmStartCache::clear: confirmed via nm"; WS_OK=1
elif strings "${IQTREE}" 2>/dev/null | grep -q 'RateWarmStartCache'; then
    echo "[preflight] RateWarmStartCache: found via strings"; WS_OK=1
fi
[[ "${WS_OK}" -eq 0 ]] && echo "[preflight] WARNING: RateWarmStartCache symbol not found" >&2

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
echo "║  AA 1M WS-A.1 Full Run (MF+SPR) — 1-node parity"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  md5 expected: fa9ee60103a1a922505cf4dfa26a2fca"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "║  branch:       fca-lbfgs-ws"
echo "║  FCA baseline: 168913089  MF=5119.929s  SPR=15060.551s  total=20180.480s"
echo "║  vanilla ref:  168425491  MF=~7587s     SPR=15098.605s  total=22776.226s"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Probe ─────────────────────────────────────────────────────────────
. "${REPO_DIR}/gadi-ci/mf-iso/tools/probe_header.sh"
probe_hw_sw "${IQTREE}"
probe_env

RANK_PROBE="${REPO_DIR}/gadi-ci/mf-iso/tools/rank_probe.sh"
[[ -x "${RANK_PROBE}" ]] || { echo "ERROR: rank_probe.sh not found at ${RANK_PROBE}" >&2; exit 9; }

# ── Full MF+SPR run ────────────────────────────────────────────────────
echo "[full] Full MF+SPR run (-m TEST), ${NRANKS} rank × ${OMP_PER_RANK} OMP"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --bind-to none \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${RANK_PROBE}" \
        numactl --localalloc -- \
            "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log"
IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))

grep -E '^RANK-PROBE: |\[.*\]' "${WORK_DIR}/iqtree_run.bindings.log" > "${WORK_DIR}/rank_bindings.log" 2>/dev/null || true

cat "${WORK_DIR}/iqtree_run.log" || true
echo ""
echo "[full] done: rc=${IQRC} wall=${WALL}s"

grep -E '^MF-TIME: '     "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_time.log"     || true
grep -E '^MF-MPI-DIAG: ' "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/mf_diag.log"     || true
grep -E '^PROBE: '       "${WORK_DIR}/iqtree_run.log" > "${WORK_DIR}/probe.log"       2>/dev/null || true

{
    echo "# rank, model_idx, model_name, subst, rate, dt_seconds, ref_remaining"
    awk -F' ' '
    /^MF-TIME: rank / {
        for (i=1; i<=NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2]; }
        printf "%s, %s, %s, %s, %s, %s, %s\n",
            v["rank"], v["model"], v["name"], v["subst"], v["rate"], v["dt"], v["ref_remaining"];
    }' "${WORK_DIR}/mf_time.log"
} > "${WORK_DIR}/rank_models.csv" 2>/dev/null || true

echo "[full] MF-TIME lines:     $(wc -l < "${WORK_DIR}/mf_time.log" 2>/dev/null || echo 0)"
echo "[full] MF-MPI-DIAG lines: $(wc -l < "${WORK_DIR}/mf_diag.log" 2>/dev/null || echo 0)"

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

EXPECTED_LNL   = -78605196.573  # job 168425491 (vanilla AA 1M)
TOL            = 0.5
# FCA np=1 baseline (168913089) timing for comparison
FCA_MF         = 5119.929
FCA_SPR        = 15060.551
FCA_TOT        = 20180.480
# Vanilla np=1 baseline (168425491) timing
VANILLA_MF     = 7587.459
VANILLA_SPR    = 15098.605
VANILLA_TOT    = 22776.226

verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({
        "file": os.path.basename(alignment),
        "status": "pass" if diff < TOL else "fail",
        "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6),
        "note": "Full SPR lnL vs vanilla ref 168425491 (AA 1M); tol=0.5",
    })
model_ok = best_model == "LG+G4"

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
        "command": f"mpirun -np {nranks} numactl --localalloc iqtree3-mpi-fca-ws-a1 -s alignment_1000000.phy -m TEST -T {omp_per_rank} -seed ${SEED}",
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
        "all_pass": iqrc == 0 and (verify[0]["status"] == "pass" if verify else False),
        "lnl_pass": verify[0]["status"] == "pass" if verify else None,
        "model_pass": model_ok,
        "vs_fca_baseline": {
            "mf_speedup":    speedup(FCA_MF,  mf_wall),
            "spr_speedup":   speedup(FCA_SPR, spr_wall),
            "total_speedup": speedup(FCA_TOT, iqwall),
        },
        "vs_vanilla_baseline": {
            "mf_speedup":    speedup(VANILLA_MF,  mf_wall),
            "spr_speedup":   speedup(VANILLA_SPR, spr_wall),
            "total_speedup": speedup(VANILLA_TOT, iqwall),
        },
    },
    "warm_start": {"phase": "A.1"},
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
                "placement": "mpi_1node_excl_full"},
    "build_tag":     "fca_ws_a1_icx_avx512",
    "branch":        "fca-lbfgs-ws",
    "non_canonical": True,
    "non_canonical_label": "FCA-WS Phase A.1 (cross-model warm-start, local-only) · ICX+MPI · AVX-512 · full MF+SPR · AA 1M",
    "group":         "fca_ws_a1",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[full] wrote {out_path}")

vs = record["summary"]["vs_fca_baseline"]
vv = record["summary"]["vs_vanilla_baseline"]
print(f"[full] ─── FULL RUN SUMMARY (AA 1M WS-A.1) ───")
for v in verify:
    print(f"[full]   lnL:      {'PASS' if v['status']=='pass' else 'FAIL'}  reported={v['reported']}  diff={v['diff']}")
print(f"[full]   model:    {'PASS' if model_ok else 'FAIL'}  ({best_model})")
if mf_wall:
    print(f"[full]   MF wall:  {mf_wall:.3f} s   vs FCA={FCA_MF}s ({vs['mf_speedup']}×)  vs vanilla={VANILLA_MF}s ({vv['mf_speedup']}×)")
if spr_wall:
    print(f"[full]   SPR wall: {spr_wall:.3f} s   vs FCA={FCA_SPR}s ({vs['spr_speedup']}×)  vs vanilla={VANILLA_SPR}s ({vv['spr_speedup']}×)")
if iqwall:
    print(f"[full]   Total:    {iqwall:.3f} s   vs FCA={FCA_TOT}s ({vs['total_speedup']}×)  vs vanilla={VANILLA_TOT}s ({vv['total_speedup']}×)")
PYEOF

echo "[full] done."
exit "${IQRC}"
