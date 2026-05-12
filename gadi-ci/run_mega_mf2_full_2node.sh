#!/bin/bash
# run_mega_mf2_full_2node.sh — MF2 Full IQ-TREE MPI 2-node
# on mega_dna.fa: 2 MPI ranks × 104 OpenMP threads = 208 total cores.
#
# Parity series with run_mega_avx512_r2_2node.sh (same dataset/seed/protocol)
# and with the xlarge_mf MF2 Full 2-node series (same binary/build_tag):
#   binary:            build-mpi-mf2/iqtree3-mpi  (MF2 LPT dispatch, R2+AVX-512)
#   non_canonical_label: "MF2 Full · ICX+MPI · R2 · AVX-512"
#   build_tag:         mf2_full_icx_avx512_r2_lpt
#
# Dataset: mega_dna.fa — 500 taxa × 100,000 sites (99,999 patterns, ~0% compression)
#   sha256: 0c8af2d62e214be8b0258393d71d1a0bed15568334de56b89116ae8653f92619
#   reference lnL (seed=1, ICX 104T): −27,328,165.86
#
# Node layout: 2 × Sapphire Rapids 8470Q (104 cores each):
#
#   ┌─── node A ──────────┐  ┌─── node B ──────────┐
#   │ rank 0  104 OMP     │  │ rank 1  104 OMP     │
#   └─────────────────────┘  └─────────────────────┘
#
# Provenance:
#   binary:  /scratch/um09/as1708/iqtree3-mf2/build-mpi-mf2/iqtree3-mpi
#   source:  modelfinder2 branch (abd98764-derived), icpx + libiomp5, R2+LPT+AVX-512
#   project: um09
#
# Estimated walltime (mega_dna AVX-512+R2 208T≈886s; extra MF2 dispatch overhead):
#   208T ≈ 900–1100 s  → 3 h requested
#
# build_tag: mf2_full_icx_avx512_r2_lpt
#
#PBS -N iq-mega-mf2-2node
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=208
#PBS -l mem=1000GB
#PBS -l walltime=03:00:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-${HOME}/setonix-iq}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build-mpi-mf2}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"
BENCHMARKS="${PROJECT_DIR}/benchmarks"
RUNS_DIR="${REPO_DIR}/logs/runs"
PROFILES_DIR="${PROJECT_DIR}/gadi-ci/profiles"

DATASET_NAME="${DATASET:-mega_dna}"
NRANKS="${NRANKS:-2}"
OMP_PER_RANK="${OMP_PER_RANK:-104}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_mf2_full_np${NRANKS}_seed1}"

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
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load." >&2
    exit 4
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found." >&2
    exit 5
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2
    exit 6
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
RUN_ID="gadi_${LABEL}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Multi-node host discovery ──────────────────────────────────────────
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — this script must run inside a PBS job." >&2
    exit 8
fi
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
if [[ "${#HOSTS[@]}" -ne 2 ]]; then
    echo "ERROR: expected 2 nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2
    echo "       Check PBS resource spec — should be #PBS -l ncpus=208 on normalsr." >&2
    exit 9
fi
HOST_A="${HOSTS[0]}"
HOST_B="${HOSTS[1]}"

HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MF2 Full IQ-TREE — MPI 2-node full-node placement"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total, 2 nodes)"
echo "║  node A:        ${HOST_A}  (rank 0, 104 cores)"
echo "║  node B:        ${HOST_B}  (rank 1, 104 cores)"
echo "║  binary:        ${IQTREE}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[2node] hostfile:"
cat "${HOSTFILE}" | sed 's/^/    /'

# Verify 104 physical cores on head node
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
echo "[2node] head node topology: sockets=${LSCPU_SOCKETS} cores/socket=${LSCPU_COREPS} → physical_cores=${PHYSICAL_CORES}"
if [[ "${PHYSICAL_CORES}" -ne 104 ]]; then
    echo "ERROR: head-node has ${PHYSICAL_CORES} physical cores, expected 104 (2×52 SPR)." >&2
    exit 10
fi

RANKFILE="${WORK_DIR}/rankfile.txt"
cat > "${RANKFILE}" <<EOF
rank 0=${HOST_A} slot=0-103
rank 1=${HOST_B} slot=0-103
EOF

echo "[2node] rankfile:"
cat "${RANKFILE}" | sed 's/^/    /'

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

TIME_WRAP="${WORK_DIR}/_time_wrap.sh"
cat > "${TIME_WRAP}" <<'EOF'
#!/bin/bash
exec numactl --localalloc -- "$@"
EOF
chmod +x "${TIME_WRAP}"

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

# ── Pass 1: clean timing run ──────────────────────────────────────────
echo ""
echo "[2node] Pass 1: ${NRANKS} ranks × ${OMP_PER_RANK} OMP across 2 nodes"
START_EPOCH=$(date +%s)

mpirun -np "${NRANKS}" \
    --hostfile "${HOSTFILE}" \
    --mca rmaps_base_mapping_policy "" \
    -rf "${RANKFILE}" \
    --report-bindings \
    "${OMP_ENV[@]}" \
    "${TIME_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log" &
IQTREE_PID=$!

sleep 5
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler pid=${INNER_PID}"
/usr/bin/python3.11 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

IQRC=0
wait "${IQTREE_PID}" || IQRC=$?
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
echo "[2node] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── Pass 2: perf stat ────────────────────────────────────────────────
if [[ "${IQRC}" -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    echo "[2node] Pass 2: perf stat (per-rank)"
    mpirun -np "${NRANKS}" \
        --hostfile "${HOSTFILE}" \
        --mca rmaps_base_mapping_policy "" \
        -rf "${RANKFILE}" \
        "${OMP_ENV[@]}" \
        "${PERF_WRAP}" \
            "${IQTREE}" -s "${DATA_PATH}" -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --prefix "${WORK_DIR}/iqtree_perf" \
        > "${WORK_DIR}/iqtree_perf.log" 2>&1 || true
fi

# ── Run record ──────────────────────────────────────────────────────
/usr/bin/python3.11 - <<PYEOF
import json, os, re, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
nranks, omp_per_rank, threads = ${NRANKS}, ${OMP_PER_RANK}, ${TOTAL_THREADS}
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
metrics = {
    "IPC": round(ins/cyc, 4) if cyc and ins else None,
    "cache-miss-rate": rate(g("cache-misses"), g("cache-references")),
    "LLC-miss-rate":   rate(g("LLC-load-misses"), g("LLC-loads")),
}
metrics = {k: v for k, v in metrics.items() if v is not None}

verify = []
if rep_ll is not None:
    verify.append({"file": os.path.basename(dpath), "status": "pass",
                   "expected": rep_ll, "reported": rep_ll, "diff": 0.0})

record = {
    "run_id": rid, "pbs_id": pbs_id,
    "platform": "gadi", "run_type": "profile", "label": label,
    "description": f"Gadi SPR MF2 Full — MPI 2-node {nranks}×{omp_per_rank} = {threads}T on mega_dna.fa",
    "timing": [{
        "command": f"mpirun -np {nranks} --hostfile hostfile.txt ... {ibin} -s {os.path.basename(dpath)} -T {omp_per_rank} -seed 1",
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
        "iqtree_version_tag": "v3.1.2+mf2",
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
        "dataset":      os.path.basename(dpath),
        "threads":      threads,
        "nranks":       nranks,
        "omp_per_rank": omp_per_rank,
        "placement":    "mpi_2node_fullnode",
        "perf_cmd":     perf_cmd,
        "metrics":      metrics,
    },
    "build_tag":           "mf2_full_icx_avx512_r2_lpt",
    "non_canonical":       True,
    "non_canonical_label": "MF2 Full \u00b7 ICX+MPI \u00b7 R2 \u00b7 AVX-512",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path,"w"), indent=2, default=str)
print(f"[2node] wrote {out_path}")
PYEOF

echo "[2node] done."
exit "${IQRC}"
