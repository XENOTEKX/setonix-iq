#!/bin/bash
# run_g4_3b_jolt_pinv_validate2_v100.sh — G.4.3b validation v2, after the +I rate-scaling fix
# (IQ-TREE's RateGammaInvar rates = mean-1 gamma / (1-pinv); JOLT now matches + FD pinv gradient).
# Two datasets: (1) the original 5000-site AA subsample (pinv~0.001 — Gate A should now tighten to ~1e-12),
# (2) a SYNTHETIC high-constant-site alignment (~50% constant cols => pinv~0.4) that STRESSES the 1/(1-pinv)
# rate scaling (where the old mean-1 bug would be a ~0.5x rate error). Gates per dataset:
#   A: [JOLT] self-check rel <= 1e-6 (ideally ~1e-12) — GPU lnL == fresh CPU computeLikelihood.
#   B: JOLT MLE == IQ-TREE's own CPU LG+I+G4 (lnL/pinv/alpha).
#PBS -N joltiv2
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
SRC=/scratch/rc29/as1708/iqtree3-gpu
SUB=$SRC/ctf_1m_a100/sub.phy
TREE=$SRC/ctf_1m_a100/coarse.treefile
WB=$SRC/jolt_iv2; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$SUB" ] && [ -f "$TREE" ] || { echo "missing inputs"; exit 1; }
echo "════════ G.4.3b +I validation v2 (rate-scaling fix) — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name --format=csv,noheader
export JOLT_DEBUG=1

# ---- build a high-constant-site alignment: NC constant columns (LG-ish freq) + the 5000 variable cols ----
echo; echo "──── synthesizing high-pinv alignment (5000 const + 5000 var) ────"
python3 - "$SUB" "$WB/hi.phy" 5000 <<'PY'
import sys, random
sub, out, NC = sys.argv[1], sys.argv[2], int(sys.argv[3])
names=[]; seqs=[]
with open(sub) as f:
    ntax, nsit = map(int, f.readline().split())
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1])
AA="ARNDCQEGHILKMFPSTWYV"; random.seed(7)
# NC constant columns: each column a single random AA repeated across all taxa
const_states=[random.choice(AA) for _ in range(NC)]
const_cols=["".join(s for _ in range(len(names))) for s in const_states]  # placeholder, rebuilt per-taxon below
with open(out,"w") as o:
    o.write(f"{len(names)} {NC+len(seqs[0])}\n")
    for i,(nm,var) in enumerate(zip(names,seqs)):
        const_part="".join(const_states)   # taxon i has the column's constant state at every const column
        o.write(f"{nm}  {const_part}{var}\n")
print(f"wrote {out}: {len(names)} taxa, {NC}+{len(seqs[0])} sites (~{NC/(NC+len(seqs[0])):.2f} constant)")
PY

for TAG in sub hi; do
  ALN=$SUB; [ "$TAG" = "hi" ] && ALN=$WB/hi.phy
  TE="-te $TREE"; [ "$TAG" = "hi" ] && TE=""   # hi.phy is a different alignment -> let IQ-TREE build its own tree (-m TESTONLY-ish: use -te only for sub)
  echo; echo "════════ dataset=$TAG ($ALN) ════════"
  echo "──── (1) LG+I+G4 via JOLT ────"
  if [ "$TAG" = "sub" ]; then
    "$BIN" --jolt --gpu -m LG+I+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$WB/${TAG}_jolt" -redo > "$WB/${TAG}_jolt.out" 2>&1
    "$BIN"             -m LG+I+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$WB/${TAG}_cpu"  -redo > "$WB/${TAG}_cpu.out" 2>&1
  else
    # build a fixed tree once (fast NJ/ML on hi.phy) so jolt & cpu compare on the SAME topology
    "$BIN" -m LG+G4 -s "$ALN" -nt 12 -pre "$WB/${TAG}_tree" -redo > "$WB/${TAG}_tree.out" 2>&1
    HT="$WB/${TAG}_tree.treefile"
    "$BIN" --jolt --gpu -m LG+I+G4 -s "$ALN" -te "$HT" -nt 12 -pre "$WB/${TAG}_jolt" -redo > "$WB/${TAG}_jolt.out" 2>&1
    "$BIN"             -m LG+I+G4 -s "$ALN" -te "$HT" -nt 12 -pre "$WB/${TAG}_cpu"  -redo > "$WB/${TAG}_cpu.out" 2>&1
  fi
  echo "  [JOLT] line:"; grep -E "\[JOLT\] model" "$WB/${TAG}_jolt.out" | tail -1
  python3 - "$WB/${TAG}_jolt.iqtree" "$WB/${TAG}_cpu.iqtree" <<'PY'
import sys,re,os
def grab(f):
    lnl=pinv=alpha=None
    if not os.path.exists(f): return lnl,pinv,alpha
    for line in open(f):
        if "Log-likelihood of the tree" in line: m=re.search(r'-?\d+\.\d+',line); lnl=float(m.group()) if m else lnl
        if "Proportion of invariable sites" in line: m=re.findall(r'[\d.]+',line); pinv=float(m[-1]) if m else pinv
        if "Gamma shape alpha" in line: m=re.findall(r'[\d.]+',line); alpha=float(m[-1]) if m else alpha
    return lnl,pinv,alpha
jl,jp,ja=grab(sys.argv[1]); cl,cp,ca=grab(sys.argv[2])
print(f"  JOLT lnL={jl} pinv={jp} alpha={ja}")
print(f"  CPU  lnL={cl} pinv={cp} alpha={ca}")
if jl and cl:
    dl=abs(jl-cl); print(f"  GATE B |dlnL|={dl:.4f} rel={dl/abs(cl):.3e} -> {'PASS' if dl<0.5 else 'CHECK'}  dpinv={abs((jp or 0)-(cp or 0)):.5f} dalpha={abs((ja or 0)-(ca or 0)):.5f}")
PY
  echo "  GATE A self-check:"; grep -oE "rel=[0-9.e+-]+ (PASS|OK\(gamma-resid\)|MISMATCH)" "$WB/${TAG}_jolt.out" | tail -1
done
echo "════════ DONE $(date -Iseconds) ════════"
