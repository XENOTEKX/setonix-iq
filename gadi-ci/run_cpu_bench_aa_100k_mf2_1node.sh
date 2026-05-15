#!/bin/bash
# run_cpu_bench_aa_100k_mf2_1node.sh — IQ-TREE3 MF2 CPU benchmark (MPI 1-node)
# AA, 100K sites (100 taxa), 1 MPI rank × 103 OpenMP threads = 103T total
# Sapphire Rapids exclusive node (normalsr-exec, 104 cores, 512 GB RAM)
#
# PURPOSE: Baseline MF2 single-node reference to isolate LPT-dispatch overhead
# vs standard single-node SPR cpu_bench (168425673, 1,169.556 s).
# Expected: ~identical wall to standard SPR run (MF2 adds only dispatch overhead).
#
# Binary:  iqtree3-mpi  (MF2 LPT dispatch, R2+AVX-512, ICX)
#   /scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi
# Build:   modelfinder2 branch (abd98764-derived), icpx + libiomp5, R2+LPT+AVX-512
# Parity:  OMP_PER_RANK=103, numactl --localalloc, KMP_BLOCKTIME=200, seed=1
#          — exactly matching run_cpu_bench_aa_100k_spr.sh (168425673)
#
# Build tag:    mf2_full_icx_avx512_r2_lpt
# Expected lnL: −7,541,976.860  (bit-identical to 168422809 / 168425673)
#
# Group:   aa_100k_mf2_scaling  (1-node, 2-node, 4-node series)
# Submit via: gadi-ci/run_cpu_bench_aa_100k_mf2_batch.sh
#
#PBS -N iq-aa-100k-mf2-1n
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=510GB
#PBS -l place=excl
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/um09
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT="${PROJECT:-dx61}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
MF2_DIR="${MF2_DIR:-/scratch/um09/${USER_ID}/iqtree3-mf2}"
IQTREE="${IQTREE:-${MF2_DIR}/build-mpi-mf2/iqtree3-mpi}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="/scratch/${PROJECT}/${USER_ID}/cpu_bench/profiles"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
DATA_TYPE="AA"
DATASET_SHORT="complex_aa_100k"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="AA_100k_mf2_np1_seed${SEED}"
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
[[ -x "${IQTREE}" ]]    || { echo "ERROR: MF2 binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
command -v mpirun >/dev/null 2>&1 || { echo "ERROR: mpirun not found after module load." >&2; exit 4; }
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 5
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5." >&2; exit 6
fi

# ── OMP / runtime ─────────────────────────────────────────────────────
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${MF2_DIR}/tmp"; mkdir -p "${TMPDIR}"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

PERF_WRAP="${WORK_DIR}/_perf_wrap.sh"
cat > "${PERF_WRAP}" <<EOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
exec perf stat -e '${PERF_EVENTS}' \\
    -o '${WORK_DIR}'/perf_stat.rank\${RANK}.txt \\
    numactl --localalloc -- "\$@"
EOF
chmod +x "${PERF_WRAP}"

cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/python3.11
import json, os, sys, time, pathlib
pid = int(sys.argv[1]); out = pathlib.Path(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
t0 = time.monotonic()
def read_status(p):
    d = {}
    try:
        with open(f"/proc/{p}/status") as f:
            for line in f:
                k, _, v = line.partition(":"); d[k.strip()] = v.strip()
    except FileNotFoundError:
        return None
    return d
def read_io(p):
    d = {}
    try:
        with open(f"/proc/{p}/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                try: d[k.strip()] = int(v.strip())
                except ValueError: pass
    except (FileNotFoundError, PermissionError):
        return None
    return d
fp = out.open("w")
while True:
    t = time.monotonic() - t0
    s = read_status(pid)
    if s is None: break
    snap = {
        "t_s": round(t, 2),
        "rss_kb":  int(s.get("VmRSS", "0 kB").split()[0]) if "VmRSS" in s else None,
        "peak_kb": int(s.get("VmHWM", "0 kB").split()[0]) if "VmHWM" in s else None,
        "voluntary_cs":   int(s.get("voluntary_ctxt_switches", "0")),
        "involuntary_cs": int(s.get("nonvoluntary_ctxt_switches", "0")),
        "io": read_io(pid) or {},
    }
    fp.write(json.dumps(snap) + "\n"); fp.flush()
    time.sleep(interval)
fp.close()
SAMPLER_EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AA 100K MF2 Benchmark — MPI 1-node (SPR exclusive)"
echo "║  run_id:       ${RUN_ID}"
echo "║  ranks × OMP: ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS}T)"
echo "║  binary:       $(basename "${IQTREE}")"
echo "║  alignment:    $(basename "${ALIGNMENT}")"
echo "║  work_dir:     ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Pass 1: clean timing run ──────────────────────────────────────────
echo "[1node] Pass 1: ${NRANKS} rank × ${OMP_PER_RANK} OMP"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --bind-to none \
    "${OMP_ENV[@]}" \
    numactl --localalloc -- \
        "${IQTREE}" -s "${ALIGNMENT}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2>&1 &
IQTREE_PID=$!

sleep 5
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler target=${INNER_PID}"
/usr/bin/python3.11 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

IQRC=0
wait "${IQTREE_PID}" || IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

cat "${WORK_DIR}/iqtree_run.log"
echo ""
echo "[1node] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── Pass 2: perf stat ─────────────────────────────────────────────────
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[1node] Pass 2: perf stat"
    mpirun -np "${NRANKS}" \
        --bind-to none \
        "${OMP_ENV[@]}" \
        "${PERF_WRAP}" \
            "${IQTREE}" -s "${ALIGNMENT}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Run record ────────────────────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
nranks, omp_per_rank, threads = ${NRANKS}, ${OMP_PER_RANK}, ${TOTAL_THREADS}
wall, iqrc = int("${WALL}"), int("${IQRC}")
alignment, ibin = "${ALIGNMENT}", "${IQTREE}"
pbs_id = "${PBS_ID_SHORT}"
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))

agg = {}; perf_cmd = None
pf = os.path.join(work, "perf_stat.rank0.txt")
if os.path.isfile(pf):
    for line in open(pf, errors="replace"):
        m = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m: perf_cmd = m.group(1); continue
        m = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w.\-:/]+)", line)
        if m and not m.group(1).startswith("<"):
            try:
                key = m.group(2).split(":",1)[0]
                agg[key] = int(m.group(1).replace(",",""))
            except ValueError: pass
def rate(n, d):
    if not n or not d: return None
    return round(100.0 * n / d, 4)
def g(*keys):
    for k in keys:
        if agg.get(k) is not None: return agg[k]
    return None
cyc, ins = g("cycles"), g("instructions")
metrics = {k: v for k, v in {
    "IPC":                 round(ins/cyc, 4) if cyc and ins else None,
    "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
    "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
    "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
    "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
    "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
}.items() if v is not None}
for _k in ("cycles","instructions","cache-references","cache-misses",
           "branch-instructions","branch-misses","L1-dcache-loads","L1-dcache-load-misses",
           "LLC-loads","LLC-load-misses","dTLB-loads","dTLB-load-misses"):
    if _k in agg: metrics[_k] = agg[_k]

EXPECTED_LNL = -7541976.860
verify = []
if rep_ll is not None:
    diff = abs(rep_ll - EXPECTED_LNL)
    verify.append({"file": os.path.basename(alignment), "status": "pass" if diff < 0.1 else "fail",
                   "expected": EXPECTED_LNL, "reported": rep_ll, "diff": round(diff, 6)})

record = {
    "run_id": rid, "label": label,
    "platform": "gadi", "run_type": "cpu_bench",
    "dataset": alignment, "dataset_short": "${DATASET_SHORT}",
    "data_type": "${DATA_TYPE}", "seq_len": 100000, "n_taxa": 100,
    "threads": threads, "seed": ${SEED},
    "timing": [{
        "command": f"mpirun -np {nranks} numactl --localalloc iqtree3-mpi -s alignment_100000.phy -T {omp_per_rank} -seed ${SEED}",
        "time_s": iqwall if iqwall is not None else wall,
    }],
    "verify": verify,
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "lnL": rep_ll,
        "all_pass": iqrc == 0,
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
                "placement": "mpi_1node_excl", "perf_cmd": perf_cmd, "metrics": metrics},
    "build_tag":           "mf2_full_icx_avx512_r2_lpt",
    "non_canonical":       True,
    "non_canonical_label": "MF2 Full \u00b7 ICX+MPI \u00b7 R2 \u00b7 AVX-512",
    "group":               "aa_100k_mf2_scaling",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[1node] wrote {out_path}")
PYEOF

echo "[1node] done."
exit "${IQRC}"
