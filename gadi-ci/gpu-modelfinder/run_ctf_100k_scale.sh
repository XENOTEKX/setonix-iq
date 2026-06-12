#!/bin/bash
# run_ctf_100k_scale.sh — CTF AA-100K scaling run (H200 or V100)
# Purpose: measure per-model JOLT wall at 100K to confirm sublinear vs 1M scaling.
# Expected: LG+G4 ~47s on V100 (measured G.4.2 -te), ~25s on H200 (est. 2× faster).
# 1M reference: LG+G4 77s on H200, LG+I+G4 338s (+I 4-start). If 100K ≈ 10× less
# than 1M → linear scaling (Minh's claim). If 100K ≈ 3-4× less → sublinear (GPU better).
#
# Submit:
#   H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=60GB -v ALABEL=h200 run_ctf_100k_scale.sh
#   V100: qsub -q gpuvolta  -l ngpus=1 -l ncpus=12 -l mem=60GB -v ALABEL=v100 run_ctf_100k_scale.sh
#
#PBS -N ctf100k
#PBS -P dx61
#PBS -l walltime=00:45:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

ALABEL="${ALABEL:-gpu}"; NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu
BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy

# 96,017 distinct patterns in the 100K alignment
NFULL=96017
# Subsample: 1000 patterns (1.04% — same as P0 validated recall test, job 170396778)
KSUB=1000
TOPK=3

WB="$SRC/ctf100k_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN"; exit 1; }
[[ -f "$ALN" ]] || { echo "ERROR: alignment not found: $ALN"; exit 1; }

echo "════════ CTF AA-100K scaling run on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "1M reference walls (H200, job 170581208): LG+G4=77s, LG+I+G4=338s(4-start), LG+F+I+G4=335s"
echo "CPU 100K MF baselines: baseline-np1=399s, FCA-np2=149s"
echo "Oracle: lnL -7541976.860, best model LG+G4"

# ── GPU power sampler ────────────────────────────────────────────────
PWLOG="$WB/power.log"
( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) \
    > "$PWLOG" 2>&1 & PWPID=$!

T_ALL0=$(date +%s)

# ── Step 1: subsample ────────────────────────────────────────────────
T0=$(date +%s)
python3 - "$ALN" "$KSUB" <<'PY'
import sys, random
src, K = sys.argv[1], int(sys.argv[2])
with open(src) as f:
    f.readline(); names=[]; seqs=[]
    for line in f:
        line = line.rstrip("\n")
        if not line.strip(): continue
        p = line.split(None, 1)
        if len(p) == 2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L = len(seqs[0]); random.seed(1); cols = sorted(random.sample(range(L), K))
open("sub.phy","w").write(f"{len(seqs)} {K}\n" + "".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print(f"wrote sub.phy: {len(seqs)} seqs × {K} sites")
PY
T_SUB=$(($(date +%s)-T0))

# ── Step 2: coarse rank on subsample ────────────────────────────────
T0=$(date +%s)
"$BIN" -m TESTONLY -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo \
    > "$WB/coarse.stdout" 2>&1
T_C=$(($(date +%s)-T0))
echo "  subsample ${T_SUB}s ; coarse ${T_C}s"
[[ -f "$WB/coarse.treefile" ]] || { echo "COARSE FAILED — check $WB/coarse.stdout"; kill $PWPID 2>/dev/null; exit 1; }

# ── Step 3: pick top-k by scale-consistent BIC ──────────────────────
python3 - "$WB/coarse.iqtree" "$KSUB" "$NFULL" "$TOPK" > "$WB/topk.txt" <<'PY'
import sys, re, math
iq, m, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
row = re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
rows = []
for line in open(iq):
    mm = row.match(line)
    if not mm: continue
    name, logl, bic = mm.group(1), float(mm.group(2)), float(mm.group(5))
    p = (bic + 2*logl) / math.log(m)
    rows.append((-2*(N/m)*logl + p*math.log(N), name, logl, round(p)))
rows.sort()
for b, n, l, p in rows[:K]: print(f"MODEL:{n}")
PY
mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
echo "  top-${TOPK}: ${TOPMODELS[*]}"

# ── Step 4: full refine top-k on the full 100K alignment ────────────
export JOLT_DEBUG=1
T_R_TOTAL=0; i=0
declare -A WALL LNL JN
for M in "${TOPMODELS[@]}"; do
    i=$((i+1)); T0=$(date +%s)
    "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" \
        -nt "$NT" -pre "$WB/refine_${i}" -redo \
        > "$WB/refine_${i}.stdout" 2>&1
    T_R=$(($(date +%s)-T0)); T_R_TOTAL=$((T_R_TOTAL+T_R))
    lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND" "$WB/refine_${i}.log" 2>/dev/null \
          | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
    jn=$(grep -c '\[JOLT\] model' "$WB/refine_${i}.stdout" 2>/dev/null || echo 0)
    WALL[$M]=$T_R; LNL[$M]=${lnl:-NA}; JN[$M]=${jn:-0}
    echo "  refine $i $M: wall=${T_R}s lnL=${lnl:-NA} JOLT_calls=${jn:-0}"
done

T_ALL=$(($(date +%s)-T_ALL0))
kill $PWPID 2>/dev/null; sleep 1

# ── Result + energy summary ──────────────────────────────────────────
echo
echo "════════ RESULT + ENERGY (${ALABEL}) ════════"
python3 - <<PY
import re, math
N=$NFULL; m=$KSUB
pmap={}
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
for line in open("$WB/coarse.iqtree"):
    mm=row.match(line)
    if mm:
        nm,logl,bic=mm.group(1),float(mm.group(2)),float(mm.group(5)); pmap[nm]=round((bic+2*logl)/math.log(m))
models="""${TOPMODELS[*]}""".split(); best=None
print(f"{'model':16}{'full_lnL':>18}{'p':>5}{'full_BIC':>18}")
for i,M in enumerate(models,1):
    lnl=None
    for line in open(f"$WB/refine_{i}.log"):
        if "Log-likelihood of the tree" in line or "BEST SCORE FOUND" in line:
            mm=re.search(r'-?[0-9]+\.[0-9]+',line)
            if mm: lnl=float(mm.group()); break
    p=pmap.get(M)
    if lnl and p:
        bic=-2*lnl+p*math.log(N); print(f"{M:16}{lnl:18.4f}{p:5d}{bic:18.1f}")
        if best is None or bic<best[1]: best=(M,bic,lnl)
if best: print(f"\nCTF WINNER: {best[0]} full lnL={best[2]:.4f} full BIC={best[1]:.1f}")
print(f"Oracle:     LG+G4    lnL=-7541976.8600             (CPU baseline job 168425673)")
# Energy
v=[float(x) for x in open("$PWLOG") if x.strip() and x.strip()[0].isdigit()]
J=sum(v)*2.0
print(f"\nGPU ENERGY: {J:.0f} J = {J/3600:.2f} Wh  (mean {sum(v)/max(len(v),1):.0f} W, {len(v)*2:.0f}s sampled)")
PY

echo
echo "  WALL: subsample ${T_SUB}s + coarse ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${T_ALL}s"
echo "  vs CPU 100K MF baseline-np1 399s -> $(python3 -c "print(f'{399/$T_ALL:.2f}x')" )  FCA-np2 149s -> $(python3 -c "print(f'{149/$T_ALL:.2f}x')")"
echo
echo "  ── SCALING vs 1M (H200, job 170581208) ──"
echo "  1M LG+G4 refine = 77s ; 100K patterns = ${NFULL} vs 1M patterns = 946439"
echo "  Pattern ratio 1M/100K = $(python3 -c "print(f'{946439/96017:.2f}x')")"
echo "  If linear scaling: expected LG+G4 100K wall = $(python3 -c "print(f'{77*96017/946439:.0f}s')")"
echo "  Actual LG+G4 100K wall = ${WALL[${TOPMODELS[0]}]:-${WALL[LG+G4]:-NA}}s  (check refine outputs for exact model)"
echo "  (sublinear if actual < expected; linear if actual ≈ expected)"
echo
echo "════════ DONE $(date -Iseconds) ════════"
