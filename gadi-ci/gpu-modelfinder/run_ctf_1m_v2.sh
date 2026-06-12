#!/bin/bash
# run_ctf_1m_v2.sh — Coarse-to-Fine ModelFinder, AA-1M, with the G.4.3b +I-capable JOLT binary.
#
# The v1 post-mortem (jobs 170479448/572): refine top-k in BIC-rank order put LG+I+G4 (#1 on the subsample) first,
# and +I was JOLT-INELIGIBLE -> PURE CPU -> 8712s/start-value on 16 cores (GPU 0% idle) = 2.8x the 2-node wall.
# G.4.3b adds the +I/pinv gradient to JOLT, so +I+G models now refine on the GPU at JOLT-eligible cost. This run
# measures whether the full CTF (coarse rank + GPU refine of top-3, +I included) beats 2 / 4 SPR nodes at 1M.
#
# CPU MF-phase baselines (measured, -m TEST seed 1): np2=3076.9s  np4=1974.5s  (np8=1443.9  np16=1122.4).
# Oracle: FCA best = LG+G4, lnL=-78,605,196.4.
#
# MEMORY: one +I+G4/+G4 model's JOLT partials at 1M ~= 88 GB (full postorder nInternal*slotSz + O(depth) preorder
# pool). Fits H200-141GB; A100-80GB is likely OOM (-> JOLT returns NaN -> CPU fallback, flagged below);
# V100-32GB cannot. Each refine samples peak GPU mem to confirm the fit / catch a silent CPU fallback.
#
# Submit:
#   H200: qsub -q gpuhopper -l ngpus=1 -l ncpus=12 -l mem=180GB -v ALABEL=h200 run_ctf_1m_v2.sh
#   A100: qsub -q dgxa100   -l ngpus=1 -l ncpus=16 -l mem=180GB -v ALABEL=a100 run_ctf_1m_v2.sh
#
#PBS -N ctf1mv2
#PBS -P dx61
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-gpu}"; NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
NFULL=940000; KSUB=5000; TOPK=3
WB="$SRC/ctf1mv2_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ CTF v2 (+I JOLT) AA-1M on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "TARGETS: np2=3076.9s np4=1974.5s ; oracle best=LG+G4 lnL~-78605196"

# ---------- (A0) subsample (seed 1) ----------
T_SUB0=$(date +%s)
python3 - "$ALN" "$KSUB" <<'PY'
import sys, random
src, K = sys.argv[1], int(sys.argv[2])
with open(src) as f:
    ntax, nsit = map(int, f.readline().split()); names, seqs = [], []
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]); random.seed(1); cols=sorted(random.sample(range(L), K))
with open("sub.phy","w") as o:
    o.write(f"{len(seqs)} {K}\n")
    for nm,s in zip(names,seqs): o.write(f"{nm}  {''.join(s[c] for c in cols)}\n")
print(f"wrote sub.phy ntax={len(seqs)} K={K}")
PY
T_SUB=$(( $(date +%s)-T_SUB0 )); echo "  subsample ${T_SUB}s"

# ---------- (A) coarse rank: stock CPU -m TESTONLY on subsample ----------
T_C0=$(date +%s)
"$BIN" -m TESTONLY -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
T_C=$(( $(date +%s)-T_C0 )); echo "  coarse ${T_C}s ; tree=$WB/coarse.treefile"
[ -f "$WB/coarse.treefile" ] || { echo "COARSE FAILED"; tail -20 "$WB/coarse.stdout"; exit 1; }

# ---------- (B) scale-consistent BIC' -> top-k ----------
python3 - "$WB/coarse.iqtree" "$KSUB" "$NFULL" "$TOPK" > "$WB/topk.txt" <<'PY'
import sys, re, math
iq, m, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
rows=[]
for line in open(iq):
    mm=row.match(line)
    if not mm: continue
    name, logl, bic_sub = mm.group(1), float(mm.group(2)), float(mm.group(5))
    p=(bic_sub+2*logl)/math.log(m); bicp=-2*(N/m)*logl+p*math.log(N)
    rows.append((bicp,name,logl,round(p)))
rows.sort()
for bicp,name,logl,p in rows[:8]: print(f"# {name}\tBICp={bicp:.1f}\tsubLogL={logl:.3f}\tp={p}")
for bicp,name,logl,p in rows[:K]: print(f"MODEL:{name}")
PY
grep '^#' "$WB/topk.txt" | sed 's/^# /  /'
mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
echo "  top-${TOPK}: ${TOPMODELS[*]}"

# ---------- (C) refine each top-k on FULL 1M with --jolt --gpu (+I now GPU-eligible) ----------
export JOLT_DEBUG=1
declare -A WALL LNL PEAK JOLTN GATEDEC
T_R_TOTAL=0; i=0
for M in "${TOPMODELS[@]}"; do
  i=$((i+1)); echo; echo "──── refine $i/${TOPK}: $M ────"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 5; done ) > "$WB/gpumem_${i}.log" 2>&1 & SMI=$!
  T_R0=$(date +%s)
  "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine_${i}" -redo > "$WB/refine_${i}.stdout" 2>&1
  RC=$?; T_R=$(( $(date +%s)-T_R0 )); T_R_TOTAL=$((T_R_TOTAL+T_R)); kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND|Optimal log-likelihood" "$WB/refine_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  jn=$(grep -c '\[JOLT\] model' "$WB/refine_${i}.stdout" 2>/dev/null)
  dec=$(grep -oE 'JOLT-GATE\] decline reason=\S+' "$WB/refine_${i}.stdout" 2>/dev/null | head -1)
  peak=$(sort -n "$WB/gpumem_${i}.log" 2>/dev/null | tail -1)
  WALL[$M]=$T_R; LNL[$M]=${lnl:-NA}; PEAK[$M]=${peak:-NA}; JOLTN[$M]=${jn:-0}; GATEDEC[$M]=${dec:-none}
  echo "  exit=$RC wall=${T_R}s lnL=${lnl:-NA} jolt_engaged=${jn:-0} gate=${dec:-engaged} peak_gpu=${peak:-NA}MiB"
  [ "${jn:-0}" = "0" ] && echo "  ⚠ JOLT did NOT engage for $M (CPU fallback — OOM or ineligible)"
done

# ---------- report ----------
echo; echo "════════ CTF v2 RESULT (${ALABEL}) ════════"
python3 - <<PY
import re, math, os
N=$NFULL
pmap={}
for line in open("$WB/topk.txt"):
    if line.startswith('# '):
        f=line[2:].strip().split('\t'); nm=f[0]
        for t in f:
            if t.startswith('p='): pmap[nm]=int(t[2:])
models="""${TOPMODELS[*]}""".split(); best=None
print(f"{'model':16}{'full_lnL':>18}{'p':>5}{'full_BIC':>18}")
for i,M in enumerate(models,1):
    log=f"$WB/refine_{i}.log"; lnl=None
    if os.path.exists(log):
        for pat in ("Log-likelihood of the tree","BEST SCORE FOUND","Optimal log-likelihood"):
            for line in open(log):
                if pat in line:
                    mm=re.search(r'-?\d+\.\d+', line)
                    if mm: lnl=float(mm.group()); break
            if lnl is not None: break
    p=pmap.get(M)
    if lnl is None or p is None: print(f"{M:16}{'NA':>18}{str(p):>5}{'NA':>18}"); continue
    bic=-2*lnl+p*math.log(N); print(f"{M:16}{lnl:18.3f}{p:5d}{bic:18.1f}")
    if best is None or bic<best[1]: best=(M,bic,lnl)
print()
if best: print(f"CTF WINNER: {best[0]}  full lnL={best[2]:.3f}  full BIC={best[1]:.1f}  (oracle LG+G4 lnL~-78605196.4)")
PY
TOTAL=$((T_SUB+T_C+T_R_TOTAL))
echo
echo "  WALL: subsample ${T_SUB}s + coarse ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${TOTAL}s"
echo "  vs np2 3076.9s -> $(python3 -c "print(f'{3076.9/$TOTAL:.2f}x' if $TOTAL>0 else 'NA')")  $([ $TOTAL -lt 3077 ] && echo 'BEATS 2 NODES ✓' || echo 'loses to 2 nodes')"
echo "  vs np4 1974.5s -> $(python3 -c "print(f'{1974.5/$TOTAL:.2f}x' if $TOTAL>0 else 'NA')")  $([ $TOTAL -lt 1975 ] && echo 'BEATS 4 NODES ✓' || echo 'loses to 4 nodes')"
echo "════════ DONE $(date -Iseconds) ════════"
