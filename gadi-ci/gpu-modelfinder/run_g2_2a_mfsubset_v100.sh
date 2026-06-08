#!/bin/bash
# run_g2_2a_mfsubset_v100.sh — Phase G.2.2a: GPU in the REAL ModelFinder model-evaluation loop (not -te).
# Runs `--gpu -m TESTONLY -mset <matrices> -mrate G -seed 1 -nt 1` so every candidate is +G4/+F+G4 (GPU-handled,
# NO +I -> no slow CPU fallback). Validates each tested model's lnL against the EXISTING 103-thread-SPR CPU
# baseline (gpumf_cpubase_TESTONLY_169678141/mf.log) — reuse, NO CPU re-run — and reports the GPU MF wall as a
# real MF-loop data point for the theta-reuse decision. Same seed + same alignment => same initial tree as the
# baseline (verified iff GPU LG+G4 == baseline LG+G4 7541976.853).
# GATES: (1) every tested model's -lnL matches the baseline rel<=1e-12; (2) best (lowest BIC) among tested = the
#        baseline's choice for that subset; (3) GPU MF wall reported (vs 221.6 s production target).
#PBS -N g2-2a
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=02:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true; module load gcc/12.2.0 2>/dev/null || true
export CC=gcc CXX=g++
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN=$SRC/build-gpu-on/iqtree3
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
BASELINE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/gpumf_cpubase_TESTONLY_169678141/mf.log
RUNDIR=$SRC/g2_2a_runs; mkdir -p "$RUNDIR"
MSET="LG,WAG,JTT,Dayhoff,cpREV,VT"   # 6 matrices x {+G4,+F+G4} = 12 GPU-handled candidates

echo "════════ G.2.2a GPU MF-loop subset — $(hostname) $(date -Iseconds) ════════"
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
echo "binary: $(ls -la "$BIN" | awk '{print $6,$7,$8}')  (no rebuild)"
echo "mset=$MSET  mrate=G  seed=1  -nt 1  --gpu"

echo; echo "════════ GPU run: --gpu -m TESTONLY -mset $MSET -mrate G ════════"
T0=$(date +%s)
"$BIN" --gpu -s "$ALN" -m TESTONLY -mset "$MSET" -mrate G -seed 1 -nt 1 -pre "$RUNDIR/mf_gpu" -redo 2>&1 | \
  grep -E "GPU-KERNEL|GPU-BRANCH active|GPU-DERV active|GPU-FROMBUF|GPU-XCHECK|will test|Akaike|Bayesian|best-fit|Best-fit" | head -25
T1=$(date +%s); echo "[GPU MF wall] $((T1-T0)) s"

echo; echo "════════ COMPARE tested models vs CPU baseline (reuse, no CPU re-run) ════════"
python3 - "$RUNDIR/mf_gpu.log" "$BASELINE" <<'PY'
import sys, re
gpu_log, base_log = sys.argv[1], sys.argv[2]
# model-table line: "<id> <model> <-lnL> <df> <AIC> <AICc> <BIC>"
row = re.compile(r'^\s*\d+\s+([A-Za-z0-9._+]+)\s+(-?\d+\.\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
def table(f):
    d={}
    for ln in open(f):
        m=row.match(ln)
        if m: d[m.group(1)]=dict(lnL=float(m.group(2)), bic=float(m.group(5)))
    return d
g, b = table(gpu_log), table(base_log)
print(f"GPU tested {len(g)} models; baseline table has {len(b)} models")
worst=0.0; nchk=0; missing=[]
for mdl,gv in sorted(g.items()):
    if mdl not in b: missing.append(mdl); continue
    bl=b[mdl]['lnL']; rl=abs((gv['lnL']-bl)/bl) if bl else abs(gv['lnL']-bl)
    worst=max(worst,rl); nchk+=1
    flag = '' if rl<=1e-12 else '  <-- CHECK'
    print(f"  {mdl:14s} GPU={gv['lnL']:.4f} base={bl:.4f} rel={rl:.2e}{flag}")
if missing: print("  (not in baseline table, skipped):", missing)
print(f"LNL MATCH: n={nchk} worst_rel={worst:.3e} -> {'PASS' if worst<=1e-12 else 'CHECK'}  (gate 1e-12)")
if g:
    gbest=min(g, key=lambda m:g[m]['bic'])
    print(f"BEST among tested (lowest BIC): GPU -> {gbest} (BIC {g[gbest]['bic']:.3f})")
PY
echo; echo "════════ DONE $(date -Iseconds) ════════"
