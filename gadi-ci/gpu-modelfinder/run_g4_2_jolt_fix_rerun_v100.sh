#!/bin/bash
# run_g4_2_jolt_fix_rerun_v100.sh — Phase G.4.2 re-run after the §IV.7.1 fix (setLikelihoodKernelGPU no-ops
# under --jolt, so the G.2.x stateless GPU overrides do NOT install: ineligible +I/+R/+FO candidates fall back
# to PURE CPU, not the slow stateless GPU sweep; and optimizeParametersJOLT's self-check becomes a genuine CPU
# recompute). Rebuilds (phylotreegpu.cpp changed) then runs BOTH G.4.2 gates in one job:
#   G.4.2a  --jolt -m LG+G4 -te  : the write-back self-check is now UNAMBIGUOUS GPU-JOLT-vs-CPU + non-interference.
#   G.4.2b  --jolt -m TESTONLY   : full ranking gate; +G/base on GPU JOLT, +I/+R/+FO on PURE CPU (fast).
#
#PBS -N g4-2-jolt-fix
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=02:00:00
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
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
BASELINE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/gpumf_cpubase_TESTONLY_169678141/mf.log
WA="$SRC/g4_2a_runs"; WB="$SRC/g4_2b_runs"; mkdir -p "$WA" "$WB"
MLE=-7541976.8529

echo "════════ G.4.2 fix re-run — $(hostname) $(date -Iseconds) ════════"
echo "src $(cd "$SRC" && git branch --show-current) HEAD $(cd "$SRC" && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

echo; echo "──── incremental rebuild (phylotreegpu.cpp changed: setLikelihoodKernelGPU no-op under --jolt) ────"
[ -f "$BUILD_ON/Makefile" ] || { echo "FATAL: no configured build dir"; exit 1; }
cd "$BUILD_ON"; make -j12 > make_g42fix.log 2>&1; RC=$?
echo "  make exit=$RC (last 8 lines:)"; tail -8 make_g42fix.log | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
echo "  binary: $(ls -la "$BIN" | awk '{print $6,$7,$8}')"

echo; echo "════════ G.4.2a  --jolt -m LG+G4 -te  (write-back self-check, now genuine CPU) ════════"
"$BIN" --jolt -s "$ALN" -m LG+G4 -te "$TREE" -seed 1 -nt 1 -pre "$WA/jolt_fix" -redo 2>&1 | \
  grep -E "JOLT|GPU-KERNEL|GPU-BRANCH|Optimal log-likelihood|BEST SCORE" | head -20
JLNL=$(grep -oE "BEST SCORE FOUND : -?[0-9.]+" "$WA/jolt_fix.iqtree" 2>/dev/null | grep -oE -- "-?[0-9.]+$" | tail -1)
echo "  [check] [GPU-BRANCH] should be ABSENT now (stateless override OFF under --jolt); self-check rel should be GPU-vs-CPU"
echo "  --jolt -te final lnL = ${JLNL} (MLE ${MLE})"

echo; echo "════════ G.4.2b  --jolt -m TESTONLY  (full ranking gate; +I/+R now PURE CPU) ════════"
T0=$(date +%s)
"$BIN" --jolt -s "$ALN" -m TESTONLY -seed 1 -nt 1 -pre "$WB/mf_jolt" -redo > "$WB/run.stdout" 2>&1
RC=$?; T1=$(date +%s)
echo "[run exit] $RC   [GPU --jolt MF wall] $((T1-T0)) s"
echo "─── [GPU-BRANCH] activations (should be 0 — stateless path off under --jolt) ───"
echo "  [GPU-BRANCH] lines: $(grep -cE '^\[GPU-BRANCH\]' "$WB/run.stdout" 2>/dev/null)"
echo "─── JOLT engagements + sample genuine-CPU self-checks ───"
grep -E "^\[JOLT\]" "$WB/run.stdout" | head -12
echo "  [JOLT] self-check lines: $(grep -cE '^\[JOLT\]' "$WB/run.stdout")"
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
