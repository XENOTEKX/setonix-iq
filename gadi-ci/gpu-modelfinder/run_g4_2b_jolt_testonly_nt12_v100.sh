#!/bin/bash
# run_g4_2b_jolt_testonly_nt12_v100.sh — Phase G.4.2b RE-RUN with the thread-safety fixes (part4 §IV.7.1).
#
# Two fixes since the killed -nt 1 attempt (job 170363332, which had ALREADY confirmed [GPU-BRANCH]=0 + genuine
# GPU-vs-CPU self-checks LG 3.86e-15 / LG+G4 2.77e-12):
#   (1) setLikelihoodKernelGPU no-ops under --jolt  -> ineligible +I/+R/+FO fall back to PURE CPU (done prior run).
#   (2) gpu_jolt_optimize takes a process-wide std::mutex; the G.2.x one-shot cross-checks are gated off under
#       --jolt -> SAFE under ModelFinder's across-model OpenMP parallelism (phylotesting.cpp:4097). JOLT candidates
#       serialize on the single GPU; CPU-fallback candidates run N-parallel.
# Run at -nt 12 (the -nt 1 full TESTONLY is impractical: ~200 single-threaded CPU-fallback models on 96K patterns).
#
# GATES: (1) best model == LG+G4; (2) per-model lnL parity vs the EXISTING CPU baseline (reuse, no re-run) —
#        JOLT +G/base rel<=1e-9 vs genuine CPU, CPU-fallback +I/+R rel~0, worst<=1e-6 << any AIC/BIC gap;
#        (3) AIC/AICc/BIC best-fit picks unchanged; (4) [GPU-BRANCH]=0 (stateless path off) + [JOLT] count;
#        (5) MF wall @ -nt 12 (informational — not directly comparable to the ~103-thread baseline; the aggregate
#        wall win is G.4.3, the +I/+R CPU tail dominates).
#
#PBS -N g4-2b-nt12
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 2>/dev/null || true
module load gcc/12.2.0   2>/dev/null || true
module load cuda/12.5.1  2>/dev/null || true
module load eigen/3.3.7  2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BUILD_ON="$SRC/build-gpu-on"; BIN="$BUILD_ON/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
BASELINE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/gpumf_cpubase_TESTONLY_169678141/mf.log
WB="$SRC/g4_2b_runs"; mkdir -p "$WB"; NT=12

echo "════════ G.4.2b RE-RUN (-nt $NT, thread-safe) — $(hostname) $(date -Iseconds) ════════"
echo "src $(cd "$SRC" && git branch --show-current) HEAD $(cd "$SRC" && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

echo; echo "──── incremental rebuild (gpu_lnl_intree.cu mutex + phylotree.cpp cross-check gate) ────"
[ -f "$BUILD_ON/Makefile" ] || { echo "FATAL: no configured build dir"; exit 1; }
cd "$BUILD_ON"; make -j12 > make_g42b_nt12.log 2>&1; RC=$?
echo "  make exit=$RC (last 8 lines:)"; tail -8 make_g42b_nt12.log | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
echo "  binary: $(ls -la "$BIN" | awk '{print $6,$7,$8}')"

echo; echo "════════ GPU run: --jolt -m TESTONLY -nt $NT ════════"
T0=$(date +%s)
"$BIN" --jolt -s "$ALN" -m TESTONLY -nt $NT -pre "$WB/mf_jolt" -redo > "$WB/run.stdout" 2>&1
RC=$?; T1=$(date +%s)
echo "[run exit] $RC   [GPU --jolt MF wall @ -nt $NT] $((T1-T0)) s"
echo "  [GPU-BRANCH] lines (MUST be 0 — stateless path off under --jolt): $(grep -cE '^\[GPU-BRANCH\]' "$WB/run.stdout" 2>/dev/null)"
echo "  [JOLT] self-check lines: $(grep -cE '^\[JOLT\]' "$WB/run.stdout" 2>/dev/null)"
echo "─── sample [JOLT] genuine GPU-vs-CPU self-checks ───"
grep -E "^\[JOLT\]" "$WB/run.stdout" 2>/dev/null | head -10
echo "─── worst self-check rel among JOLT models (parse the [JOLT] lines) ───"
grep -E "^\[JOLT\]" "$WB/run.stdout" 2>/dev/null | grep -oE "rel=[0-9.e+-]+" | sort -t= -k2 -g | tail -1
echo "─── best-fit picks ───"
grep -iE "Akaike Information|Bayesian Information|Best-fit model according" "$WB/mf_jolt.iqtree" "$WB/run.stdout" 2>/dev/null | head -6

echo; echo "════════ COMPARE tested models vs CPU baseline (reuse, no CPU re-run) ════════"
python3 - "$WB/mf_jolt.iqtree" "$WB/mf_jolt.log" "$BASELINE" <<'PY'
import sys, re, os
gpu_iq, gpu_log, base_log = sys.argv[1], sys.argv[2], sys.argv[3]
row = re.compile(r'^\s*\d+\s+([A-Za-z0-9._+]+)\s+(-?\d+\.\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
def table(f):
    d={}
    if not f or not os.path.exists(f): return d
    for ln in open(f):
        m=row.match(ln)
        if m: d[m.group(1)]=dict(lnL=float(m.group(2)), aic=float(m.group(3)), aicc=float(m.group(4)), bic=float(m.group(5)))
    return d
g = table(gpu_iq) or table(gpu_log); b = table(base_log)
print(f"GPU tested {len(g)} models; baseline table {len(b)} models")
worst=0.0; nchk=0; worst_mdl=''
for mdl,gv in sorted(g.items()):
    if mdl not in b: continue
    bl=b[mdl]['lnL']; rl=abs((gv['lnL']-bl)/bl) if bl else abs(gv['lnL']-bl)
    if rl>worst: worst=rl; worst_mdl=mdl
    nchk+=1
    if rl>1e-9: print(f"  {mdl:16s} GPU={gv['lnL']:.4f} base={bl:.4f} rel={rl:.2e}{'  <-- >1e-6' if rl>1e-6 else ''}")
print(f"LNL PARITY: n={nchk} worst_rel={worst:.3e} ({worst_mdl}) -> {'PASS' if worst<=1e-6 else 'CHECK'}  (gate 1e-6)")
for crit in ('bic','aic','aicc'):
    if g and b:
        gb=min(g,key=lambda m:g[m][crit]); bb=min(b,key=lambda m:b[m][crit])
        print(f"  BEST by {crit.upper():4s}: GPU -> {gb:16s} | baseline -> {bb:16s}  {'PASS' if gb==bb else 'CHECK'}")
PY
echo; echo "════════ DONE $(date -Iseconds) ════════"
