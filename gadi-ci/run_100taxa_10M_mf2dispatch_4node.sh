#!/bin/bash
# run_100taxa_10M_mf2dispatch_4node.sh — IQ-TREE 3.1.2 + MF2 Model-Level MPI
# Dispatch on alignment_10000000.phy (100 taxa, 10 M DNA sites) across 4 Gadi
# normalsr SPR nodes: 4 MPI ranks × 104 OpenMP threads each.
#
# PURPOSE: 2-hour profiling run for direct comparison with PBS 167977883
# (same dataset, same 4-node SPR topology, seed=1, no fixed tree).
#
# ┌─ PBS 167977883 baseline (old binary — v3.1.1, no MF dispatch) ─────┐
# │  All 4 ranks evaluated the SAME 9 models redundantly in ~2 h wall. │
# │  9 unique models out of 968 (JC, JC+ASC, JC+G4, JC+ASC+G4, JC+R2–6).│
# │  Service Units: 1674.40  (416 cores × 2.013 h × 2 SU/core-h)      │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─ PBS 168000932 (MF2 dispatch, commit 1ac3c0a8 — BROKEN i%nranks) ──┐
# │  Rank 0 evaluated only 24/242 assigned models then sat IDLE.        │
# │  MF_WAITING cross-rank blocking: +Rk models blocked permanently     │
# │  because +R(k-1) belonged to a different rank and was never eval'd. │
# │  Job killed at 3h wall, exit -29. No Best-fit model produced.       │
# │  Service Units: 2514.03 SU wasted.  (Issue 7)                       │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─ This run (MF2 dispatch, commit abd98764 — LPT + MF_WAITING fix) ───┐
# │  LPT cost-sort: models sorted by rate-category cost desc,           │
# │  assigned p%nranks.  Rank gets mix of cheap+expensive models.       │
# │  MF_WAITING cleared: each rank evals full 242-model stripe without  │
# │  cross-rank promotion dependencies blocking +Rk models.             │
# │  --mrate G,I+G: 88 models (~11×) to complete in 3h on 4 nodes.     │
# │  For full 968-model sweep: use 16 nodes or 6h+ walltime.            │
# └─────────────────────────────────────────────────────────────────────┘
#
# MF2 patch inventory in binary (commit abd98764, branch gadi-spr-r2-avx512):
#   Phase 1: LPT cost-sorted stripe + MF_WAITING clear (Issue 7 fix)
#   Phase 2: MPI_Allreduce gather + checkpoint merge + model name restore
#   Phase 3: --mpi-ranks-per-node OMP thread budget (default 1 rank/node)
#   Issue 5: sequential model eval in MPI builds (eliminates OMP data race)
#   Issue 6: always use evaluateAll() in MPI builds (np=1 ≡ np=N code path)
#   Issue 7: LPT cost-sort + resetFlag(MF_WAITING) (this binary)
#
# mpirun mapping: --map-by node:PE=104
#   NOT hostfile+rankfile — conflicts with PBS BYCORE on normalsr.
#   Confirmed working in PBS 168000131 (xlarge_mf Phase 5 benchmark).
#
# Profiling strategy (1 pass, not 3):
#   perf stat is wrapped INTO Pass 1 so hardware counters are captured even
#   if the job is killed at walltime (perf flushes on SIGTERM). A clean
#   timing Pass 2 is retained but guarded by exit==0 (won't fire for a
#   2-hour interrupted run). VTune is skipped — deferred finalization on
#   a killed run produces unusable data.
#
#PBS -N iq-100taxa-10M-mf2-4node
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=416
#PBS -l mem=2000GB
#PBS -l walltime=06:00:00
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

DATASET_NAME="${DATASET:-100taxa_10M}"
DATASET_FILE="${DATASET_FILE:-alignment_10000000.phy}"
NRANKS="${NRANKS:-4}"
OMP_PER_RANK="${OMP_PER_RANK:-104}"
TOTAL_THREADS=$(( NRANKS * OMP_PER_RANK ))
SEED="${SEED:-1}"
LABEL="${LABEL:-${DATASET_NAME}_${TOTAL_THREADS}t_mf2dispatch_mpi${NRANKS}x${OMP_PER_RANK}_4node_fullnode}"

DATA_PATH="${BENCHMARKS}/${DATASET_NAME}/${DATASET_FILE}"
[[ -f "${DATA_PATH}" ]] || { echo "ERROR: dataset ${DATA_PATH} not found." >&2; exit 2; }

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7         2>/dev/null || true
    module load intel-compiler-llvm   2>/dev/null || true
fi

# ── Preflight checks ──────────────────────────────────────────────────
if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found after module load openmpi/4.1.7." >&2; exit 4
fi
if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found." >&2
    echo "       Build with: cd build-mpi-mf2 && gmake -j4 iqtree3" >&2
    exit 5
fi
if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong build?" >&2; exit 6
fi
if ldd "${IQTREE}" 2>/dev/null | grep -q 'libgomp'; then
    echo "ERROR: ${IQTREE} links libgomp — expected libiomp5 (icpx build)." >&2; exit 7
fi
# LPT fix verification (Issue 7): confirm binary contains the MF_WAITING clear string.
# The broken i%nranks binary (commit 1ac3c0a8) does NOT contain this string.
# PBS 168000932 was killed at 3h because it used the broken binary — rank 0 evaluated
# only 24/242 models, all others blocked by MF_WAITING cross-rank dependency.
if ! strings "${IQTREE}" 2>/dev/null | grep -q 'cost-sorted LPT stripe'; then
    echo "ERROR: ${IQTREE} is missing the LPT fix (Issue 7)." >&2
    echo "       This binary uses the broken i%nranks stripe — rank 0 will evaluate" >&2
    echo "       only ~24/242 models then sit idle (see PBS 168000932)." >&2
    echo "       Rebuild from commit abd98764 on gadi-spr-r2-avx512." >&2
    exit 11
fi
echo "[preflight] binary: ${IQTREE}"
echo "[preflight] LPT fix: CONFIRMED (cost-sorted LPT stripe + MF_WAITING clear)"
echo "[preflight] MPI:    $(mpirun --version 2>&1 | head -1)"
echo "[preflight] dataset: ${DATA_PATH} ($(du -sh "${DATA_PATH}" | cut -f1))"

export KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"
PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
RUN_ID="gadi_${LABEL}_${PBS_ID_SHORT}"
WORK_DIR="${PROFILES_DIR}/${LABEL}_${PBS_ID_SHORT}"
mkdir -p "${WORK_DIR}" "${RUNS_DIR}"
cd "${WORK_DIR}"

# ── Node discovery ────────────────────────────────────────────────────
if [[ ! -s "${PBS_NODEFILE:-/dev/null}" ]]; then
    echo "ERROR: PBS_NODEFILE missing — must run inside a PBS job." >&2; exit 8
fi
mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
if [[ "${#HOSTS[@]}" -ne 4 ]]; then
    echo "ERROR: expected 4 nodes, got ${#HOSTS[@]} (${HOSTS[*]:-empty})" >&2
    echo "       Check PBS spec: #PBS -l ncpus=416 on normalsr." >&2
    exit 9
fi
HOST_A="${HOSTS[0]}"; HOST_B="${HOSTS[1]}"
HOST_C="${HOSTS[2]}"; HOST_D="${HOSTS[3]}"

# Hostfile — for env.json snapshot only (NOT passed to mpirun; we use --map-by)
HOSTFILE="${WORK_DIR}/hostfile.txt"
awk '{c[$1]++} END {for (h in c) print h, "slots=" c[h]}' "${PBS_NODEFILE}" > "${HOSTFILE}"

# ── Topology check ────────────────────────────────────────────────────
LSCPU_SOCKETS="$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
LSCPU_COREPS="$(lscpu  | awk -F: '/Core\(s\) per socket/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
PHYSICAL_CORES="$(( ${LSCPU_SOCKETS:-2} * ${LSCPU_COREPS:-52} ))"
if [[ "${PHYSICAL_CORES}" -ne 104 ]]; then
    echo "ERROR: head-node has ${PHYSICAL_CORES} cores, expected 104 (2×52 SPR)." >&2; exit 10
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IQ-TREE 3.1.2 + MF2 dispatch — 4-node SPR 2-hour profile"
echo "║  Baseline: PBS 167977883 (no dispatch, 9 models in ~2h)"
echo "║  run_id:        ${RUN_ID}"
echo "║  dataset:       ${DATA_PATH}"
echo "║  ranks × OMP:   ${NRANKS} × ${OMP_PER_RANK}  (= ${TOTAL_THREADS} total threads)"
echo "║  seed:          ${SEED}"
echo "║  node A (rank 0): ${HOST_A}"
echo "║  node B (rank 1): ${HOST_B}"
echo "║  node C (rank 2): ${HOST_C}"
echo "║  node D (rank 3): ${HOST_D}"
echo "║  binary:        ${IQTREE}"
echo "║  mpirun:        --map-by node:PE=${OMP_PER_RANK}"
echo "║  work_dir:      ${WORK_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "[mf2-4node] hostfile (reference):"; cat "${HOSTFILE}" | sed 's/^/    /'

# ── env.json snapshot ────────────────────────────────────────────────
ENV_JSON="${WORK_DIR}/env.json"
python3 - <<PYENV > "${ENV_JSON}"
import json, os, subprocess, hashlib
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d
def sha256(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return h.hexdigest()
    except Exception: return None
ds = "${DATA_PATH}"
env = {
  "run_id": "${RUN_ID}", "label": "${LABEL}",
  "threads": ${TOTAL_THREADS}, "mpi_ranks": ${NRANKS}, "omp_per_rank": ${OMP_PER_RANK},
  "mf2_dispatch": True,
  "mf2_commit": "abd98764",
  "mf2_branch": "gadi-spr-r2-avx512",
  "mf2_patches": [
    "phase1_lpt_stripe", "phase2_allreduce", "phase3_thread_budget",
    "issue5_sequential_eval", "issue6_evaluateall_always",
    "issue7_lpt_mf_waiting_fix"
  ],
  "comparison_baseline": {
    "pbs_job": "167977883",
    "binary": "v3.1.1 no dispatch",
    "models_in_2h": 9,
    "su_used": 1674.40,
  },
  "broken_run_reference": {
    "pbs_job": "168000932",
    "binary_commit": "1ac3c0a8",
    "issue": "Issue 7 — MF_WAITING cross-rank blocking",
    "models_rank0": 24,
    "models_assigned_rank0": 242,
    "exit": -29,
    "su_wasted": 2514.03,
  },
  "mpirun_mapping": "--map-by node:PE=${OMP_PER_RANK}",
  "seed": ${SEED},
  "placement": "mf2_mpi_4node_fullnode",
  "nodes": 4,
  "hosts": ["${HOST_A}", "${HOST_B}", "${HOST_C}", "${HOST_D}"],
  "hostfile": open("${HOSTFILE}").read() if os.path.isfile("${HOSTFILE}") else None,
  "hostname": sh("hostname"), "kernel": sh("uname -r"),
  "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
  "cpu": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
  "cpu_sockets":          int(sh("lscpu | awk -F: '/Socket\(s\)/{print \$2}' | xargs","0") or 0),
  "cpu_cores_per_socket": int(sh("lscpu | awk -F: '/Core\(s\) per socket/{print \$2}' | xargs","0") or 0),
  "numa_nodes":           int(sh("lscpu | awk -F: '/NUMA node\(s\)/{print \$2}' | xargs","0") or 0),
  "mem_total_kb":         int(sh("awk '/MemTotal/{print \$2}' /proc/meminfo","0") or 0),
  "smt_active": sh("cat /sys/devices/system/cpu/smt/active 2>/dev/null") == "1",
  "mpi_version": sh("mpirun --version 2>&1 | head -1"),
  "icx":  sh("icx --version 2>/dev/null | head -1"),
  "iqtree_binary":  "${IQTREE}",
  "iqtree_version": sh("${IQTREE} --version 2>&1 | head -1"),
  "date": sh("date -Iseconds"),
  "dataset": {
    "path": ds, "file": os.path.basename(ds),
    "size_bytes": os.path.getsize(ds) if os.path.isfile(ds) else None,
    "sha256": sha256(ds),
  },
  "pbs": {
    "job_id":      os.environ.get("PBS_JOBID"),
    "job_name":    os.environ.get("PBS_JOBNAME"),
    "queue":       os.environ.get("PBS_QUEUE"),
    "project":     os.environ.get("PROJECT","${PROJECT}"),
    "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
    "nnodes":      4,
    "nodes":       ["${HOST_A}","${HOST_B}","${HOST_C}","${HOST_D}"],
    "submit_host": os.environ.get("PBS_O_HOST"),
    "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
    "scheduler":   "pbs_pro",
  },
}
print(json.dumps(env, indent=2))
PYENV
echo "  → ${ENV_JSON}"

# ── MPI/UCX transport ─────────────────────────────────────────────────
MPI_OPTS=(
    --mca pml ucx
    -x "UCX_TLS=rc_mlx5,ud_mlx5,sm,self"
    -x "UCX_NET_DEVICES=mlx5_0:1"
)

# ── OMP environment ───────────────────────────────────────────────────
OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
)

# ── perf events (':u' suffix for perf_event_paranoid=2) ───────────────
_PERF_EVENTS_BASE="cycles,instructions,branch-instructions,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$(echo "${_PERF_EVENTS_BASE}" | tr ',' '\n' | sed 's/$/:u/' | paste -sd,)"

# ── /proc sampler (rank 0 / node A only) ─────────────────────────────
cat > "${WORK_DIR}/_sampler.py" <<'SAMPLER_EOF'
#!/usr/bin/env python3
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
    except FileNotFoundError: return None
    return d
def read_io(p):
    d = {}
    try:
        with open(f"/proc/{p}/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                try: d[k.strip()] = int(v.strip())
                except ValueError: pass
    except (FileNotFoundError, PermissionError): return None
    return d
fp = out.open("w")
while True:
    t = time.monotonic() - t0
    s = read_status(pid)
    if s is None: break
    snap = {
      "t_s":         round(t, 2),
      "rss_kb":      int(s.get("VmRSS",  "0 kB").split()[0]) if "VmRSS"  in s else None,
      "peak_kb":     int(s.get("VmHWM",  "0 kB").split()[0]) if "VmHWM"  in s else None,
      "vms_kb":      int(s.get("VmSize", "0 kB").split()[0]) if "VmSize" in s else None,
      "threads_now": int(s.get("Threads", "0"))               if "Threads" in s else None,
      "io":          read_io(pid) or {},
    }
    fp.write(json.dumps(snap) + "\n"); fp.flush()
    time.sleep(interval)
fp.close()
SAMPLER_EOF

# ── Combined perf+numactl wrapper (used in Pass 1) ────────────────────
# Perf stat is wrapped INTO Pass 1 so hardware counters are recorded even if
# the job is killed at walltime (perf stat flushes counters on SIGTERM).
PERF_WRAP="${WORK_DIR}/_perf_wrap.sh"
cat > "${PERF_WRAP}" <<EOF
#!/bin/bash
RANK="\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}"
exec perf stat -e '${PERF_EVENTS}' \\
    -o '${WORK_DIR}'/perf_stat.rank\${RANK}.txt \\
    numactl --localalloc -- "\$@"
EOF
chmod +x "${PERF_WRAP}"

# Clean numactl wrapper (used in Pass 2 if Pass 1 completes successfully)
TIME_WRAP="${WORK_DIR}/_time_wrap.sh"
cat > "${TIME_WRAP}" <<'EOF'
#!/bin/bash
exec numactl --localalloc -- "$@"
EOF
chmod +x "${TIME_WRAP}"

# ── Pass 1: perf stat + sampler (combined) ────────────────────────────
# perf stat overhead: ~5%.  Captures CPU hardware counters for the full
# ~2h run, including partial data if killed at walltime.
echo ""
echo "[mf2-4node] Pass 1: ${NRANKS} ranks × ${OMP_PER_RANK} OMP + perf stat"
echo "[mf2-4node] Expected: MF-MPI: rank 0/4 assigned 242/968 models (cost-sorted LPT stripe, MF_WAITING cleared)"
START_EPOCH=$(date +%s)
IQRC=0

mpirun -np "${NRANKS}" \
    --map-by "node:PE=${OMP_PER_RANK}" \
    --report-bindings \
    "${MPI_OPTS[@]}" \
    "${OMP_ENV[@]}" \
    "${PERF_WRAP}" \
        "${IQTREE}" -s "${DATA_PATH}" -m MF -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    --mrate "${MRATE:-G,I+G}" \
                    --prefix "${WORK_DIR}/iqtree_run" \
    > "${WORK_DIR}/iqtree_run.log" 2> "${WORK_DIR}/iqtree_run.bindings.log" &
IQTREE_PID=$!

sleep 5
INNER_PID="$(pgrep -f 'iqtree3-mpi' 2>/dev/null | head -1 || true)"
[[ -z "${INNER_PID:-}" ]] && INNER_PID="${IQTREE_PID}"
echo "  → mpirun pid=${IQTREE_PID}, sampler on pid=${INNER_PID}"
python3 "${WORK_DIR}/_sampler.py" "${INNER_PID}" "${WORK_DIR}/samples.jsonl" 10 &
SAMPLER_PID=$!

wait "${IQTREE_PID}" || IQRC=$?
IQRC="${IQRC:-0}"
END_EPOCH=$(date +%s)
WALL=$(( END_EPOCH - START_EPOCH ))
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
echo "[mf2-4node] Pass 1 done: rc=${IQRC} wall=${WALL}s"

# ── MF-MPI diagnostic lines ───────────────────────────────────────────
echo ""
echo "[mf2-4node] MF-MPI dispatch diagnostics:"
grep "MF-MPI:" "${WORK_DIR}/iqtree_run.log" 2>/dev/null | sed 's/^/    /' || \
    echo "    (no MF-MPI: lines — dispatch may not have fired or log incomplete)"

echo "[mf2-4node] Models evaluated (rank 0 log lines):"
grep -cE "^  *[0-9]+ " "${WORK_DIR}/iqtree_run.log" 2>/dev/null | \
    xargs -I{} echo "    {} model rows in iqtree_run.log" || true

# ── Pass 2: clean timing (only if Pass 1 completed — won't fire for 2h kill) ──
if [[ "${IQRC}" -eq 0 ]]; then
    echo ""
    echo "[mf2-4node] Pass 2: ${NRANKS} ranks × ${OMP_PER_RANK} OMP (clean timing)"
    mpirun -np "${NRANKS}" \
        --map-by "node:PE=${OMP_PER_RANK}" \
        "${MPI_OPTS[@]}" \
        "${OMP_ENV[@]}" \
        "${TIME_WRAP}" \
            "${IQTREE}" -s "${DATA_PATH}" -m MF -T "${OMP_PER_RANK}" -seed "${SEED}" \
                        --mrate "${MRATE:-G,I+G}" \
                        --prefix "${WORK_DIR}/iqtree_clean" \
        > "${WORK_DIR}/iqtree_clean.log" 2>&1 || true
    echo "[mf2-4node] Pass 2 done."
fi

# ── Emit run record ───────────────────────────────────────────────────
python3 - <<PYEOF
import json, os, re, glob, subprocess
work, runs = "${WORK_DIR}", "${RUNS_DIR}"
rid, label = "${RUN_ID}", "${LABEL}"
total_thr = ${TOTAL_THREADS}; nranks=${NRANKS}; omp_per=${OMP_PER_RANK}
wall, iqrc = int("${WALL}"), int("${IQRC}")
dpath, ibin = "${DATA_PATH}", "${IQTREE}"
hosts = ["${HOST_A}","${HOST_B}","${HOST_C}","${HOST_D}"]
def sh(c, d=""):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return d

log = os.path.join(work, "iqtree_run.log")
rep_ll = None; iqwall = None; mf_cpu = None; mf_wall = None
mf_mpi_lines = []; models_assigned = {}; best_model = None
models_counted = 0
if os.path.isfile(log):
    for line in open(log, errors="replace"):
        m = re.search(r"BEST SCORE FOUND\s*:\s*(-?[\d.]+)", line)
        if m: rep_ll = float(m.group(1))
        m = re.search(r"Total wall-clock time used:\s+([\d.]+)", line)
        if m: iqwall = float(m.group(1))
        m = re.search(r"CPU time for ModelFinder:\s+([\d.]+)", line)
        if m: mf_cpu = float(m.group(1))
        m = re.search(r"Wall-clock time for ModelFinder:\s+([\d.]+)", line)
        if m: mf_wall = float(m.group(1))
        m = re.search(r"Best-fit model:\s+(\S+)\s+chosen", line)
        if m: best_model = m.group(1)
        if "MF-MPI:" in line:
            mf_mpi_lines.append(line.strip())
            ma = re.search(r"rank (\d+)/(\d+) assigned (\d+)/(\d+)", line)
            if ma:
                models_assigned[int(ma.group(1))] = {
                    "assigned": int(ma.group(3)), "total": int(ma.group(4))
                }
        m = re.match(r"\s{2,4}(\d+)\s+\S+\s+[\d.]+\s+\d+\s+[\d.]+", line)
        if m: models_counted += 1

# Count models with more relaxed pattern (rank-0 log only shows rank-0 models)
models_in_2h_rank0 = models_counted

# Perf stats per rank
perf_per_rank = {}; perf_cmd = None
for fp in sorted(glob.glob(os.path.join(work, "perf_stat.rank*.txt"))):
    rk = re.search(r"rank(\d+)", fp).group(1)
    pm = {}
    for line in open(fp, errors="replace"):
        m_cmd = re.match(r"Performance counter stats for '(.*)':", line.strip())
        if m_cmd and rk == "0": perf_cmd = m_cmd.group(1); continue
        m_c = re.match(r"\s*([\d,]+|<not supported>|<not counted>)\s+([\w.\-:/]+)", line)
        if m_c and not m_c.group(1).startswith("<"):
            try:
                key = m_c.group(2).split(":", 1)[0]
                pm[key] = int(m_c.group(1).replace(",", ""))
            except ValueError: pass
    perf_per_rank[rk] = pm

agg = {}
for pm in perf_per_rank.values():
    for k, v in pm.items(): agg[k] = agg.get(k, 0) + v
def rate(n, d):
    if not n or not d: return None
    return round(100.0 * n / d, 4)
def g(*keys):
    for k in keys:
        if agg.get(k) is not None: return agg[k]
    return None
cyc, ins = g("cycles"), g("instructions")
metrics = {
  "IPC":                 round(ins/cyc, 4) if cyc and ins else None,
  "cache-miss-rate":     rate(g("cache-misses"), g("cache-references")),
  "branch-miss-rate":    rate(g("branch-misses"), g("branch-instructions")),
  "L1-dcache-miss-rate": rate(g("L1-dcache-load-misses"), g("L1-dcache-loads")),
  "LLC-miss-rate":       rate(g("LLC-load-misses"), g("LLC-loads")),
  "dTLB-miss-rate":      rate(g("dTLB-load-misses"), g("dTLB-loads")),
}
for k in ("cycles","instructions","cache-references","cache-misses",
          "branch-instructions","branch-misses","L1-dcache-loads",
          "L1-dcache-load-misses","LLC-loads","LLC-load-misses",
          "dTLB-loads","dTLB-load-misses","iTLB-loads","iTLB-load-misses"):
    if k in agg: metrics[k] = agg[k]
metrics = {k: v for k, v in metrics.items() if v is not None}

per_rank_ipc = {}
for rk, pm in perf_per_rank.items():
    c, i = pm.get("cycles"), pm.get("instructions")
    if c and i: per_rank_ipc[rk] = round(i/c, 4)

proc_summary = None
sjf = os.path.join(work, "samples.jsonl")
if os.path.isfile(sjf):
    snaps = []
    for raw in open(sjf, errors="replace"):
        try: snaps.append(json.loads(raw))
        except json.JSONDecodeError: pass
    if snaps:
        peak_rss = max((s["rss_kb"] for s in snaps if s.get("rss_kb") is not None), default=None)
        peak_vms = max((s["vms_kb"] for s in snaps if s.get("vms_kb") is not None), default=None)
        max_thr  = max((s["threads_now"] for s in snaps if s.get("threads_now") is not None), default=None)
        final_io = snaps[-1].get("io") or {}
        proc_summary = {
            "sample_count": len(snaps), "duration_s": snaps[-1].get("t_s"),
            "peak_rss_kb": peak_rss, "peak_vms_kb": peak_vms, "max_threads": max_thr,
            "read_bytes":  final_io.get("read_bytes"),
            "write_bytes": final_io.get("write_bytes"),
        }

bindings_log = os.path.join(work, "iqtree_run.bindings.log")
bindings_excerpt = None
if os.path.isfile(bindings_log):
    bindings_excerpt = "\n".join(
        l for l in open(bindings_log, errors="replace").read().splitlines()
        if "MCW rank" in l or "binding" in l.lower())

artefacts = {}
for fname, key in [
    ("samples.jsonl",           "proc_timeseries"),
    ("iqtree_run.log",          "iqtree_log"),
    ("iqtree_run.bindings.log", "mpi_bindings_log"),
    ("hostfile.txt",            "hostfile"),
    ("env.json",                "env_json"),
]:
    p = os.path.join(work, fname)
    if os.path.exists(p): artefacts[key] = p
rank_perf_files = sorted(glob.glob(os.path.join(work, "perf_stat.rank*.txt")))
if rank_perf_files:
    artefacts["perf_stat_per_rank"] = rank_perf_files

# Baseline comparison (PBS 167977883)
baseline = {
    "pbs_job":          "167977883",
    "binary":           "v3.1.1 no MF dispatch",
    "wall_s":           7245,
    "models_unique":    9,
    "models_redundant": 9 * 4,
    "su_used":          1674.40,
    "exit":             271,
    "notes":            "9 JC variants evaluated redundantly by all 4 ranks; killed at 2h",
}
dispatch_summary = {
    "models_assigned_per_rank": models_assigned,
    "models_in_log_rank0":      models_in_2h_rank0,
    "models_unique_coverage":   models_in_2h_rank0 * nranks if models_in_2h_rank0 else None,
    "mf_mpi_lines":             mf_mpi_lines,
    "mf_cpu_s":                 mf_cpu,
    "mf_wall_s":                mf_wall,
    "best_model":               best_model,
}

record = {
  "run_id": rid, "pbs_id": "${PBS_ID_SHORT}",
  "platform": "gadi", "run_type": "profile", "label": label,
  "description": (f"MF2 dispatch — {nranks} ranks × {omp_per} OMP, 4 SPR nodes, "
                  f"seed={${SEED}}, -m MF, --map-by node:PE={omp_per}. "
                  f"Comparison with PBS 167977883 (same topology, no dispatch)."),
  "timing": [{
    "command": (f"mpirun -np {nranks} --map-by node:PE={omp_per} "
                f"numactl --localalloc {os.path.basename(ibin)} "
                f"-s {os.path.basename(dpath)} -m MF -T {omp_per} -seed ${SEED}"),
    "time_s":    iqwall if iqwall is not None else wall,
    "memory_kb": (proc_summary or {}).get("peak_rss_kb", 0) or 0,
  }],
  "verify": [{"file": os.path.basename(dpath), "status": "partial",
               "reported_ll": rep_ll, "note": "run killed at walltime — partial MF"}]
              if rep_ll is not None else [],
  "env": {
    "hostname": sh("hostname"), "date": sh("date -Iseconds"),
    "cpu":      sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs"),
    "cores":    int(sh("nproc","0") or 0),
    "icx":      sh("icx --version 2>/dev/null | head -1"),
    "mpi":      sh("mpirun --version 2>&1 | head -1"),
    "kernel":   sh("uname -r"),
    "os":       sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"'"),
    "nodes":    4, "hosts": hosts,
    "pbs": {
      "job_id":      os.environ.get("PBS_JOBID"),
      "job_name":    os.environ.get("PBS_JOBNAME"),
      "queue":       os.environ.get("PBS_QUEUE"),
      "project":     os.environ.get("PROJECT","${PROJECT}"),
      "ncpus":       os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS"),
      "nnodes":      4, "nodes": hosts,
      "submit_host": os.environ.get("PBS_O_HOST"),
      "submit_dir":  os.environ.get("PBS_O_WORKDIR"),
      "scheduler":   "pbs_pro",
    },
  },
  "summary": {
    "pass": 1 if iqrc == 0 else 0, "fail": 0 if iqrc == 0 else 1,
    "total_time": iqwall if iqwall is not None else wall,
    "all_pass": iqrc == 0,
    "exit_code": iqrc,
    "killed_at_walltime": iqrc != 0,
  },
  "profile": {
    "dataset":       os.path.basename(dpath),
    "threads":       total_thr,
    "mpi_ranks":     nranks,
    "omp_per_rank":  omp_per,
    "nodes":         4,
    "mpirun_mapping": f"--map-by node:PE={omp_per}",
    "mf2_dispatch":  dispatch_summary,
    "baseline":      baseline,
    "perf_cmd":      perf_cmd,
    "metrics":       metrics,
    "per_rank_ipc":  per_rank_ipc,
    "per_rank_host": {str(i): hosts[i] for i in range(nranks)},
    "bindings":      bindings_excerpt,
    "proc_summary":  proc_summary,
    "artefacts":     artefacts,
  },
  "build_tag":           "mf2dispatch_lpt_icx_mpi4x104_4node_fullnode_avx512_r2_v312",
  "mf2_commit":          "abd98764",
  "non_canonical":       True,
  "non_canonical_label": f"MF2 dispatch · MPI {nranks}×{omp_per} 4-node",
}
out_path = os.path.join(runs, rid + ".json")
json.dump(record, open(out_path, "w"), indent=2, default=str)
print(f"[mf2-4node] run record → {out_path}")
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " MF2 dispatch vs baseline comparison (PBS 167977883)"
echo "════════════════════════════════════════════════════════════════"
echo "  Baseline: 4 ranks, no dispatch, seed=1"
echo "    Wall: ~2h  |  Models evaluated: 9 unique (all 4 ranks redundant)"
echo "    Su: 1674.40"
echo ""
echo "  This run: 4 ranks + MF2 dispatch, seed=1"
if grep -q "MF-MPI:" "${WORK_DIR}/iqtree_run.log" 2>/dev/null; then
    grep "MF-MPI:" "${WORK_DIR}/iqtree_run.log" | sed 's/^/    /'
fi
MODEL_ROWS=$(grep -cE "^  {1,4}[0-9]+ " "${WORK_DIR}/iqtree_run.log" 2>/dev/null || echo 0)
echo "    Model rows in rank-0 log: ${MODEL_ROWS}"
echo "    Unique models covered (×${NRANKS} ranks): $(( MODEL_ROWS * NRANKS ))"
echo "    wall=${WALL}s  exit=${IQRC}"
echo "════════════════════════════════════════════════════════════════"
echo "[mf2-4node] done."
exit "${IQRC}"
