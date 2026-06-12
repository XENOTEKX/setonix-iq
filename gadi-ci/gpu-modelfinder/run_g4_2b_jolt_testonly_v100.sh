#!/bin/bash
# run_g4_2b_jolt_testonly_v100.sh — Phase G.4.2b: in-tree JOLT in the FULL ModelFinder loop (-m TESTONLY).
#
# G.4.2a validated the JOLT seam + write-back on ONE model (LG+G4). G.4.2b runs the WHOLE -m TESTONLY candidate
# set with --jolt: JOLT-eligible candidates (+G / base, fixed-Q) optimise on the GPU joint path; everything else
# (+I, +R, +I+R, +FO) falls back to the standard CPU path. Validates the SELECTION is unchanged and reports the
# real MF wall. Reuses the binary built in G.4.2a (NO rebuild) and the EXISTING CPU baseline (reuse, NO re-run).
#
# GATES:
#   (1) BEST MODEL unchanged: lowest-BIC model among tested == the baseline's choice (LG+G4).
#   (2) lnL PARITY: every tested model's -lnL matches the CPU baseline. JOLT-handled (+G/base) models converge to
#       the CPU optimum (rel ~1e-9, as G.4.2a); CPU-fallback (+I/+R) models are identical (rel ~0). worst_rel<=1e-6
#       is FAR tighter than any AIC/BIC gap (thousands), so the ranking cannot flip.
#   (3) RANKING: the AIC/AICc/BIC best-fit picks match the baseline.
#   (4) WALL: GPU --jolt MF wall reported. HONEST EXPECTATION: ~flat vs CPU — the +I/+R fallback tail dominates
#       full TESTONLY (the per-model +G speedup from G.4.2a does NOT move the aggregate); the wall win is G.4.3
#       (grid.z cross-model batching + tiling), which builds on this validated seam.
#   (5) JOLT ENGAGED: count the [JOLT] self-check lines (capped at 12) confirming GPU joint-opt fired on +G models.
#
#PBS -N g4-2b-jolt-testonly
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
module load cuda/12.5.1 2>/dev/null || true; module load gcc/12.2.0 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
BASELINE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/gpumf_cpubase_TESTONLY_169678141/mf.log
WORK="$SRC/g4_2b_runs"; mkdir -p "$WORK"

echo "════════ G.4.2b in-tree JOLT — FULL -m TESTONLY — $(hostname) $(date -Iseconds) ════════"
echo "src $(cd "$SRC" && git branch --show-current) HEAD $(cd "$SRC" && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
[ -x "$BIN" ] || { echo "no binary (run G.4.2a build first)"; exit 1; }
echo "[binary] $(ls -la "$BIN" | awk '{print $6,$7,$8}')  (no rebuild)"
echo "[aln] $([ -f "$ALN" ] && echo OK || echo MISSING) ; [baseline] $([ -f "$BASELINE" ] && echo OK || echo MISSING)"

echo; echo "════════ GPU run: --jolt -m TESTONLY -seed 1 -nt 1 ════════"
T0=$(date +%s)
"$BIN" --jolt -s "$ALN" -m TESTONLY -seed 1 -nt 1 -pre "$WORK/mf_jolt" -redo > "$WORK/run.stdout" 2>&1
RC=$?; T1=$(date +%s)
echo "[run exit] $RC   [GPU --jolt MF wall] $((T1-T0)) s"
echo "─── JOLT engagements (self-check lines, capped 12) ───"
grep -E "^\[JOLT\]" "$WORK/run.stdout" | head -12
NJOLT=$(grep -cE "^\[JOLT\]" "$WORK/run.stdout")
echo "  [JOLT] self-check lines printed: $NJOLT"
echo "─── best-fit picks (GPU run) ───"
grep -iE "Akaike Information|Bayesian Information|Best-fit model according" "$WORK/mf_jolt.iqtree" 2>/dev/null | head -6
grep -iE "Akaike Information|Bayesian Information|Best-fit model according" "$WORK/run.stdout" 2>/dev/null | head -6

echo; echo "════════ COMPARE tested models vs CPU baseline (reuse, no CPU re-run) ════════"
python3 - "$WORK/mf_jolt.iqtree" "$WORK/mf_jolt.log" "$BASELINE" <<'PY'
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
g = table(gpu_iq) or table(gpu_log)
b = table(base_log)
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
