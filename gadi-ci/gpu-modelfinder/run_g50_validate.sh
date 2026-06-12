#!/bin/bash
# run_g50_validate.sh — validate G.5.0 (on-device reduction). Reruns the EXACT AA -m MF 5000-site subsample (seed 1,
# identical to coverage audit 170602983) with the rebuilt binary and checks: (1) every [JOLT] self-check
# (GPU lnL vs fresh CPU computeLikelihood) stays rel<=1e-9 (expect ~1e-12) — the on-device reduction did not perturb
# the likelihood; (2) per-model lnL + joint-iter counts match the pre-change reference (printed for manual diff vs
# o170602983 — same deterministic subsample => should match to ~1e-9 and identical iters). Also runs LG+I+G4 -te.
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=90GB -l walltime=00:30:00 gadi-ci/gpu-modelfinder/run_g50_validate.sh
#PBS -N g50val
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
AA=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$SRC/g50val_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
echo "════════ G.5.0 on-device-reduction validation — $(hostname) $(date -Iseconds) ════════"
ls -l --time-style=+%Y-%m-%dT%H:%M "$BIN"; nvidia-smi --query-gpu=name --format=csv,noheader
python3 - "$AA" 5000 "$WB/aa.phy" <<'PY'
import sys,random
src,K,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
with open(src) as f:
    f.readline(); names=[];seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]);seqs.append(p[1].replace(" ",""))
L=len(seqs[0]);random.seed(1);cols=sorted(random.sample(range(L),K))
open(out,"w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print("wrote aa.phy 100x5000 (seed 1, == audit 170602983)")
PY
export JOLT_DEBUG=1
echo "──── AA --jolt --gpu -m MF ────"
"$BIN" --jolt --gpu -m MF -s "$WB/aa.phy" -nt "$NT" -seed 1 -pre "$WB/aa_mf" -redo > "$WB/aa_mf.out" 2>&1
echo "  exit=$? best=$(grep -iE 'Best-fit model according to BIC' "$WB/aa_mf.iqtree" 2>/dev/null | head -1)"
echo "  ENGAGE count: $(grep -c '\[JOLT\] model=' "$WB/aa_mf.out" 2>/dev/null) (audit was 116)"
echo "  decline tally:"; grep -hoE '\[JOLT-GATE\] decline reason=[^ ]+' "$WB/aa_mf.out" 2>/dev/null | sort | uniq -c | sed 's/^/    /'
echo "  first 6 [JOLT] lines (compare lnL+iters to o170602983):"
grep -hoE '\[JOLT\] model=[^|]+\| [0-9]+ joint iters \| GPU lnL=[-0-9.]+ +CPU lnL=[-0-9.]+ rel=[0-9.e+-]+ (PASS|FAIL)' "$WB/aa_mf.out" 2>/dev/null | head -6 | sed 's/^/    /'
echo
echo "──── GATE: max self-check rel over ALL engaged models ────"
python3 - "$WB/aa_mf.out" <<'PY'
import re,sys
rels=[float(m) for m in re.findall(r'rel=([0-9.eE+-]+)\s+(?:PASS|FAIL)', open(sys.argv[1]).read())]
fails=sum(1 for r in rels if r>1e-9)
print(f"    engaged self-checks={len(rels)}  max_rel={max(rels) if rels else float('nan'):.3e}  rel>1e-9 count={fails}")
print(f"    GATE {'PASS' if rels and fails==0 else 'FAIL'} (all GPU==CPU rel<=1e-9 after on-device reduction)")
PY
echo "════════ DONE $(date -Iseconds) ════════"
