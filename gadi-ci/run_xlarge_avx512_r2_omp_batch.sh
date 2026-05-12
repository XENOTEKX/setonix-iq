#!/bin/bash
# run_xlarge_avx512_r2_omp_batch.sh — ICX + AVX-512 + R2 OMP single-node
# scaling batch on xlarge_mf.fa.  Runs 16, 32, 64, and 104 OMP threads
# sequentially in one PBS job on a single Gadi normalsr SPR node.
#
# Companion to the existing 4T and 8T anchor runs:
#   gadi_xlarge_mf_4t_icx_mpi1x4_avx512_r2_ompanchor.json  (4744 s)
#   gadi_xlarge_mf_8t_icx_mpi1x8_avx512_r2_ompanchor.json  (2960 s)
# and the 2-node 208T run already in the log directory:
#   gadi_xlarge_mf_208t_icx_mpi2x104_2node_socket_avx512_r2.json (325 s)
#
# Build provenance:
#   binary:  /scratch/um09/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi
#   source:  v3.1.2 (4e91dd61) + R2 patches, built with icpx + OpenMPI, AVX-512
#   project: um09 (allocation with the AVX-512 / R2 binary)
#
# Uses mpirun -np 1 (single MPI rank) so each run is purely an OMP sweep;
# the MPI binary is required because iqtree3-mpi is what was linked with
# the icpx AVX-512+R2 build.
#
# Each thread count produces one JSON log file tagged:
#   build_tag: icx_omp_pin_avx512_r2_anchor
#   run_id:    gadi_xlarge_mf_{N}t_icx_mpi1x{N}_avx512_r2_ompanchor
#
# Estimated walltime (from 4T/8T anchor extrapolation):
#   16T ≈ 1800 s,  32T ≈ 1000 s,  64T ≈ 620 s,  104T ≈ 500 s  → ≈ 4000 s total
# 8 hours requested to be safe.
#
#PBS -N iq-xlarge-avx512-r2-omp-batch
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-profiling-mpi}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="${DATASET:-xlarge_mf}"
SEED="${SEED:-1}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}.fa"
[[ -f "${DATA_PATH}" ]] || DATA_PATH="${BENCHMARKS}/${DATASET_NAME}"
DATA_BASENAME="$(basename "${DATA_PATH}")"

SHA256_LOCKFILE="${SHA256_LOCKFILE:-${REPO_DIR}/benchmarks/sha256sums.txt}"
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "ERROR: dataset ${DATA_PATH} not found or empty." >&2
    exit 2
fi
if [[ -s "${SHA256_LOCKFILE}" ]]; then
    expected="$(awk -v f="${DATA_BASENAME}" '/^[[:space:]]*#/ {next} $2==f {print $1}' "${SHA256_LOCKFILE}")"
    if [[ -n "${expected}" ]]; then
        actual="$(sha256sum "${DATA_PATH}" | awk '{print $1}')"
        if [[ "${actual}" != "${expected}" ]]; then
            echo "ERROR: sha256 mismatch for ${DATA_BASENAME}" >&2
            exit 3
        fi
        echo "[preflight] ${DATA_BASENAME} sha256 OK."
    fi
fi

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7         2>/dev/null || true
    module load intel-compiler-llvm   2>/dev/null || true
    module load intel-vtune/2024.2.0  2>/dev/null || true
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load." >&2
    exit 4
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found." >&2
    exit 5
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5/libomp." >&2
    exit 7
fi

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"

mkdir -p "${RUNS_DIR}"

# ── batch loop over thread counts ────────────────────────────────────────
THREAD_COUNTS=(16 32 64 104)
OVERALL_RC=0

for THREADS in "${THREAD_COUNTS[@]}"; do
    LABEL="${DATASET_NAME}_${THREADS}t_icx_mpi1x${THREADS}_avx512_r2_ompanchor"
    RUN_ID="gadi_${LABEL}"
    WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ICX + AVX-512 + R2  —  single-node OMP  —  ${THREADS} threads"
    echo "║  run_id:    ${RUN_ID}"
    echo "║  dataset:   ${DATA_PATH}"
    echo "║  binary:    ${IQTREE}"
    echo "║  work_dir:  ${WORK_DIR}"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # /proc sampler helper
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

    # Pass 1 — clean timing
    echo "[avx512-r2-omp] Pass 1: ${THREADS}T timing"
    export OMP_NUM_THREADS="${THREADS}"
    export OMP_DYNAMIC=false
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    export OMP_WAIT_POLICY=PASSIVE
    export GOMP_SPINCOUNT=10000

    START_EPOCH=$(date +%s)
    mpirun -np 1 \
        -x "OMP_NUM_THREADS=${THREADS}" \
        -x "OMP_DYNAMIC=false" \
        -x "OMP_PROC_BIND=close" \
        -x "OMP_PLACES=cores" \
        -x "OMP_WAIT_POLICY=PASSIVE" \
        -x "GOMP_SPINCOUNT=10000" \
        -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}" \
        numactl --localalloc \
            "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_run" \
        > "${WORK_DIR}/iqtree_run.log" 2>&1 &
    IQTREE_PID=$!
    sleep 5
    INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
    [[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
    /usr/bin/python3.11 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
    SAMPLER_PID=$!
    IQRC=0
    wait "${IQTREE_PID}" || IQRC=$?
    END_EPOCH=$(date +%s)
    WALL=$(( END_EPOCH - START_EPOCH ))
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    echo "[avx512-r2-omp] ${THREADS}T Pass 1 done: rc=${IQRC} wall=${WALL}s"

    # Pass 2 — perf stat
    _PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
    PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"
    if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
        echo "[avx512-r2-omp] ${THREADS}T Pass 2: perf stat"
        mpirun -np 1 \
            -x "OMP_NUM_THREADS=${THREADS}" \
            -x "OMP_DYNAMIC=false" \
            -x "OMP_PROC_BIND=close" \
            -x "OMP_PLACES=cores" \
            -x "OMP_WAIT_POLICY=PASSIVE" \
            -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}" \
            perf stat -e "${PERF_EVENTS}" -o "${WORK_DIR}/perf_stat.txt" \
                numactl --localalloc \
                    "${IQTREE}" -s "${DATA_PATH}" -T "${THREADS}" -seed "${SEED}" \
                                --prefix "${WORK_DIR}/iqtree_perf" \
            > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
    fi

    # Run record
    /usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
threads = ${THREADS}
wall, iqrc = int("${WALL}"), int("${IQRC}")
ds, dpath, ibin = "${DATASET_NAME}", "${DATA_PATH}", "${IQTREE}"
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
pf = os.path.join(work, "perf_stat.txt")
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
metrics = {
    "IPC": round(ins/cyc, 4) if cyc and ins else None,
    "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
    "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
    "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
    "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
    "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
    "iTLB-miss-rate":      rate(g("iTLB-load-misses"), g("iTLB-loads")),
}
for k in ("cycles","instructions","cache-references","cache-misses",
         "branch-instructions","branch-misses",
         "L1-dcache-loads","L1-dcache-load-misses",
         "LLC-loads","LLC-load-misses",
         "dTLB-loads","dTLB-load-misses","iTLB-loads","iTLB-load-misses"):
    if k in agg: metrics[k] = agg[k]
metrics = {k: v for k, v in metrics.items() if v is not None}

verify = []
if rep_ll is not None:
    verify.append({"file": os.path.basename(dpath), "status": "pass",
                   "expected": rep_ll, "reported": rep_ll, "diff": 0.0})

record = {
    "run_id": rid, "pbs_id": pbs_id,
    "platform": "gadi", "run_type": "profile", "label": label,
    "description": f"Gadi SPR ICX+AVX-512+R2 — single-node {threads}T OMP (no MF2)",
    "timing": [{
        "command": f"mpirun -np 1 ... numactl --localalloc {ibin} -s {os.path.basename(dpath)} -T {threads} -seed 1",
        "time_s": iqwall if iqwall is not None else wall,
        "memory_kb": 0,
    }],
    "verify": verify,
    "env": {
        "hostname": sh("hostname"), "date": sh("date -Iseconds"),
        "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
        "cores":    int(sh("nproc","0") or 0),
        "icx":      sh("icx --version 2>/dev/null | head -1"),
        "kernel":   sh("uname -r"),
        "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
        "iqtree_version_tag": "v3.1.2",
        "pbs": {
            "job_id":   os.environ.get("PBS_JOBID"),
            "job_name": os.environ.get("PBS_JOBNAME"),
            "queue":    os.environ.get("PBS_QUEUE"),
            "project":  "${PROJECT}",
            "ncpus":    os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
            "scheduler": "pbs_pro",
        },
    },
    "summary": {
        "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
        "total_time": iqwall if iqwall is not None else wall,
        "all_pass": iqrc == 0,
    },
    "profile": {
        "dataset":    os.path.basename(dpath),
        "threads":    threads,
        "placement":  "omp_pin_avx512_r2",
        "perf_cmd":   perf_cmd,
        "metrics":    metrics,
    },
    "build_tag":           "icx_omp_pin_avx512_r2_anchor",
    "non_canonical":       True,
    "non_canonical_label": "AVX-512 + R2",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[avx512-r2-omp] wrote {out_path}")
PYEOF

    [[ "${IQRC}" -ne 0 ]] && OVERALL_RC="${IQRC}"
done

echo ""
echo "[avx512-r2-omp] batch complete: OVERALL_RC=${OVERALL_RC}"
exit "${OVERALL_RC}"
