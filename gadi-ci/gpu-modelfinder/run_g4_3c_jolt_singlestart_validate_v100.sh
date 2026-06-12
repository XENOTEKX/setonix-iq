#!/bin/bash
# run_g4_3c_jolt_singlestart_validate_v100.sh — validate the +I single-start fix (G.4.3c) + test the energy harness.
# The fix: under --jolt, JOLT-eligible +I+G bypasses optimizeParametersGammaInvar's 10-start pinv sweep (one joint
# JOLT call instead of 10). GATES:
#   (1) single-start --jolt produces 1 [JOLT] call (not ~10) — the bypass engaged.
#   (2) single-start --jolt MLE (lnL,pinv,alpha) == stock 10-start CPU MLE, on BOTH the collapsed subsample
#       (pinv->0) AND the high-pinv synthetic (pinv~0.5, the multimodal stress the restarts exist for). Gate
#       |dlnL|<0.05 nat AND |dpinv|<0.005.
#   (3) wall: single-start should be ~1/N the per-call count of the 10-start.
# Also TESTS Linaro Forge perf-report --no-mpi energy mechanics on a GPU node (one wrapped --jolt run) + an
# nvidia-smi power.draw integrator (the direct GPU-energy measure) — de-risks the harness for the 1M reruns.
#PBS -N jolt3c
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load linaro-forge/24.0.2 2>/dev/null || true
export PATH=/apps/linaro-forge/24.0.2/bin:$PATH
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
SRC=/scratch/rc29/as1708/iqtree3-gpu
SUB=$SRC/ctf_1m_a100/sub.phy
TREE=$SRC/ctf_1m_a100/coarse.treefile
WB=$SRC/jolt_3c; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$SUB" ] && [ -f "$TREE" ] || { echo "missing inputs"; exit 1; }
echo "════════ G.4.3c +I single-start validation + energy-harness test — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name --format=csv,noheader
export JOLT_DEBUG=1

# high-pinv synthetic (5000 const + 5000 var) for the multimodal stress
python3 - "$SUB" "$WB/hi.phy" 5000 <<'PY'
import sys, random
sub, out, NC = sys.argv[1], sys.argv[2], int(sys.argv[3])
names=[]; seqs=[]
with open(sub) as f:
    f.readline()
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1])
AA="ARNDCQEGHILKMFPSTWYV"; random.seed(7)
cs=[random.choice(AA) for _ in range(NC)]
with open(out,"w") as o:
    o.write(f"{len(names)} {NC+len(seqs[0])}\n")
    for nm,var in zip(names,seqs): o.write(f"{nm}  {''.join(cs)}{var}\n")
print(f"wrote {out}: {len(names)}x{NC+len(seqs[0])} (~{NC/(NC+len(seqs[0])):.2f} const)")
PY

smpl(){ ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$1" 2>&1 & echo $!; }
energy_from(){ python3 -c "import sys; v=[float(x) for x in open('$1') if x.strip() and x.strip()[0].isdigit()]; dt=2.0; J=sum(v)*dt; print(f'{J:.0f} J ({J/3600:.3f} Wh), mean {sum(v)/max(len(v),1):.0f} W over {len(v)*dt:.0f}s, n={len(v)}')" 2>/dev/null || echo "n/a"; }

for TAG in sub hi; do
  ALN=$SUB; TE="$TREE"
  if [ "$TAG" = "hi" ]; then ALN="$WB/hi.phy";
    "$BIN" -m LG+G4 -s "$ALN" -nt 12 -pre "$WB/hi_tree" -redo > "$WB/hi_tree.out" 2>&1; TE="$WB/hi_tree.treefile"; fi
  echo; echo "════════ dataset=$TAG ════════"
  echo "──── (A) stock CPU 10-start (reference MLE) ────"
  t0=$(date +%s); "$BIN" -m LG+I+G4 -s "$ALN" -te "$TE" -nt 12 -pre "$WB/${TAG}_cpu" -redo > "$WB/${TAG}_cpu.out" 2>&1
  echo "   wall=$(($(date +%s)-t0))s"
  echo "──── (B) NEW --jolt single-start ────"
  P=$(smpl "$WB/${TAG}_pw.log"); t0=$(date +%s)
  "$BIN" --jolt --gpu -m LG+I+G4 -s "$ALN" -te "$TE" -nt 12 -pre "$WB/${TAG}_jolt" -redo > "$WB/${TAG}_jolt.out" 2>&1
  echo "   wall=$(($(date +%s)-t0))s ; JOLT calls=$(grep -c '\[JOLT\] model' "$WB/${TAG}_jolt.out") (expect 1)"
  kill $P 2>/dev/null; echo "   GPU energy: $(energy_from "$WB/${TAG}_pw.log")"
  python3 - "$WB/${TAG}_jolt.iqtree" "$WB/${TAG}_cpu.iqtree" <<'PY'
import sys,re,os
def grab(f):
    lnl=pinv=alpha=None
    if os.path.exists(f):
        for line in open(f):
            if "Log-likelihood of the tree" in line: m=re.search(r'-?\d+\.\d+',line); lnl=float(m.group()) if m else lnl
            if "Proportion of invariable sites" in line: m=re.findall(r'[\d.]+',line); pinv=float(m[-1]) if m else pinv
            if "Gamma shape alpha" in line: m=re.findall(r'[\d.]+',line); alpha=float(m[-1]) if m else alpha
    return lnl,pinv,alpha
jl,jp,ja=grab(sys.argv[1]); cl,cp,ca=grab(sys.argv[2])
print(f"   single-start JOLT: lnL={jl} pinv={jp} alpha={ja}")
print(f"   stock 10-start CPU: lnL={cl} pinv={cp} alpha={ca}")
if jl and cl: dl=abs(jl-cl); print(f"   GATE2 |dlnL|={dl:.4f} {'PASS' if dl<0.05 else 'FAIL'}  |dpinv|={abs((jp or 0)-(cp or 0)):.5f} {'PASS' if abs((jp or 0)-(cp or 0))<0.005 else 'FAIL'}")
PY
done

echo; echo "════════ (C) perf-report energy-harness test on the sub --jolt run ════════"
perf-report --no-mpi --output="$WB/perfrep_sub.txt" \
    "$BIN" --jolt --gpu -m LG+I+G4 -s "$SUB" -te "$TREE" -nt 12 -pre "$WB/perfrep_sub" -redo > "$WB/perfrep.out" 2>&1
echo "perf-report exit=$? ; report files:"; ls -la "$WB"/perfrep_sub.* 2>/dev/null
echo "--- energy/CPU section of the perf-report (txt) ---"
grep -iE "energy|wall|cpu|power|joule|watt" "$WB/perfrep_sub.txt" 2>/dev/null | head -20 || echo "(no txt report — check .html)"
echo "════════ DONE $(date -Iseconds) ════════"
