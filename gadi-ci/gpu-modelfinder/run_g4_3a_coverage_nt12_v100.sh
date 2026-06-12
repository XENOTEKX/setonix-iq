#!/bin/bash
# run_g4_3a_coverage_nt12_v100.sh — G.4.3a: MEASURE TRUE JOLT COVERAGE with the fixed logging.
#
# The G.4.3a diagnostic proved +F already engages JOLT; the old "12/224 = 5%" was a double logging artifact
# (cap report_count<12 + the [JOLT] print dropped the +F suffix via model->name). Fixed: [JOLT] now uses
# model->getName() (incl. +F) and cap 1000; JOLT_DEBUG=1 logs every gate decision (engage vs decline reason).
# This run gives the DEFINITIVE coverage breakdown over the full TESTONLY candidate set AND re-validates the
# G.4.2b ranking with correct model labels.
#
# GATES: (1) best == LG+G4 (BIC/AIC/AICc); (2) JOLT engagements now correctly labelled (incl. +F+G4);
#        (3) coverage = #engage vs #decline-by-reason (pinvar = the +I/+I+G gap = G.4.3b target);
#        (4) lnL parity vs CPU baseline unchanged; (5) [GPU-BRANCH]=0.
#
#PBS -N g4-3a-cov
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
BASELINE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/gpumf_cpubase_TESTONLY_169678141/mf.log
WB="$SRC/g4_3a_cov"; mkdir -p "$WB"; NT=12

echo "════════ G.4.3a COVERAGE MEASURE (-nt $NT, fixed logging) — $(hostname) $(date -Iseconds) ════════"
echo "src $(cd "$SRC" && git rev-parse --short HEAD)"
echo; echo "──── rebuild (phylotreegpu.cpp [JOLT] print fix + gate logging) ────"
cd "$BUILD_ON"; make -j12 > make_g43a_cov.log 2>&1; RC=$?
echo "  make exit=$RC"; tail -4 make_g43a_cov.log | sed 's/^/    /'
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

echo; echo "════════ --jolt -m TESTONLY -nt $NT (JOLT_DEBUG=1) ════════"
export JOLT_DEBUG=1
T0=$(date +%s)
"$BIN" --jolt -s "$ALN" -m TESTONLY -nt $NT -pre "$WB/cov" -redo > "$WB/run.stdout" 2> "$WB/run.stderr"
RC=$?; T1=$(date +%s)
echo "[run exit] $RC   [MF wall @ -nt $NT] $((T1-T0)) s"

echo; echo "──── COVERAGE BREAKDOWN ────"
echo "  [JOLT] engagements (correctly labelled, incl. +F): $(grep -cE '^\[JOLT\] model=' "$WB/run.stdout" 2>/dev/null)"
echo "    distinct engaged model names:"
grep -oE '^\[JOLT\] model=[A-Za-z0-9._+]+' "$WB/run.stdout" 2>/dev/null | sort -u | sed 's/^/      /'
echo "  [JOLT] +F engagements (freqtype=3 reached + NOT declined): $(grep -E '^\[JOLT\] model=' "$WB/run.stdout" 2>/dev/null | grep -c '+F')"
echo "  [JOLT-GATE] reached hook total: $(grep -cE '^\[JOLT-GATE\] reached hook' "$WB/run.stderr" 2>/dev/null)"
echo "  [JOLT-GATE] declines by reason:"
grep -E '^\[JOLT-GATE\] decline' "$WB/run.stderr" 2>/dev/null | grep -oE 'reason=[a-z/]+' | sort | uniq -c | sed 's/^/      /'
echo "  [GPU-BRANCH] (must be 0): $(grep -cE '^\[GPU-BRANCH\]' "$WB/run.stdout" 2>/dev/null)"

echo; echo "──── best-fit + worst JOLT self-check rel ────"
grep -iE "Best-fit model according to BIC" "$WB/cov.iqtree" 2>/dev/null
grep -E '^\[JOLT\]' "$WB/run.stdout" 2>/dev/null | grep -oE 'rel=[0-9.e+-]+' | sort -t= -k2 -g | tail -1

echo; echo "════════ lnL parity vs CPU baseline (reuse) ════════"
python3 - "$WB/cov.iqtree" "$WB/cov.log" "$BASELINE" <<'PY'
import sys,re,os
gi,gl,bl=sys.argv[1:4]
row=re.compile(r'^\s*\d+\s+([A-Za-z0-9._+]+)\s+(-?\d+\.\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
def tab(f):
    d={}
    if f and os.path.exists(f):
        for ln in open(f):
            m=row.match(ln)
            if m: d[m.group(1)]={'lnL':float(m.group(2)),'bic':float(m.group(5))}
    return d
g=tab(gi) or tab(gl); b=tab(bl)
worst=0;wm=''
for k,v in g.items():
    if k in b and b[k]['lnL']:
        r=abs((v['lnL']-b[k]['lnL'])/b[k]['lnL'])
        if r>worst: worst=r;wm=k
print(f"  overlap n={sum(1 for k in g if k in b)} worst_rel={worst:.3e} ({wm}) -> {'PASS' if worst<=1e-6 else 'CHECK'}")
if g and b:
    print(f"  BEST by BIC: GPU={min(g,key=lambda m:g[m]['bic'])} | base={min(b,key=lambda m:b[m]['bic'])}")
PY
echo; echo "════════ DONE $(date -Iseconds) ════════"
