#!/bin/bash
# run_ctf_1m.sh — Coarse-to-Fine (CTF) ModelFinder on 1 GPU vs FCA on N SPR nodes, AA-1M.
#
# GOAL (user, reduced-bar): does 1 GPU running CTF+JOLT beat FCA's ModelFinder phase on 2 / 4 SPR nodes?
#   CPU MF-phase baselines (measured, same alignment, -m TEST, seed 1):
#     np2 = 3076.9 s   np4 = 1974.5 s   np8 = 1443.9 s   np16 = 1122.4 s   (np1 = 5119.9 s)
#   FCA best model = LG+G4, lnL = -78,605,196.4 (the correctness oracle for "did CTF pick the right model").
#
# CTF ARCHITECTURE (plays each device to its strength — the honest design):
#   (A) COARSE: subsample 1M -> K sites; rank ALL ~224 candidates on the subsample with stock CPU -m TESTONLY
#       (-nt N = breadth across models, the CPU's strength; ~tens of s). This ALSO builds the fixed subsample
#       tree (no expensive fast-ML-on-1M tree, which is a real chunk of FCA's 3077s -> a legitimate CTF saving).
#   (B) RANK by SCALE-CONSISTENT BIC: BIC' = -2*(N/m)*LogL_sub + p*ln(N)  (full-data N in the penalty; p
#       recovered from the printed subsample BIC: p = (BIC_sub + 2*LogL_sub)/ln(m)). Take top-k=3.
#   (C) FINE: refine each top-k model on the FULL 1M alignment with --jolt (GPU depth; +I auto-falls to CPU).
#       Winner = min full-data BIC. Total CTF wall = coarse + sum(refine).
#
# HONEST FRAMING (do not overclaim): CTF is an ALGORITHM (CPU-portable; a cluster could run it too). This
# measures "1 GPU box running the new method" vs "N SPR nodes running stock IQ-TREE" — a real accessibility
# comparison, NOT pure-hardware GPU-beats-CPU. Pass = beat np2 (3077s); stretch = beat np4 (1974s).
#
# ARCH: binary embeds sm_70/80/90 cubins+PTX (portable V100/A100/H100). 1M +G4 partials ~60GB => A100/H100 only
# for the GPU refine; V100 (32GB) runs coarse + falls the refine to CPU (reported, not a GPU result).
#
# Submit (override resources per queue), e.g.:
#   A100: qsub -q dgxa100  -l ngpus=1 -l ncpus=16 -l mem=120GB -v ALABEL=a100 run_ctf_1m.sh
#   H100: qsub -q gpuhopper -l ngpus=1 -l ncpus=16 -l mem=120GB -v ALABEL=h100 run_ctf_1m.sh
#   V100: qsub -q gpuvolta  -l ngpus=1 -l ncpus=12 -l mem=90GB  -v ALABEL=v100 run_ctf_1m.sh
#
#PBS -N ctf-1m
#PBS -P dx61
#PBS -l walltime=04:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-gpu}"
NT="${PBS_NCPUS:-12}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
TRUE_TREE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/tree_1.full.treefile
NFULL=940000          # ~unique-pattern scale for BIC' penalty (AA-1M; refined from real run below)
KSUB=5000             # coarse subsample sites (P0: 1-5k recalls top-3; 5k = safe)
TOPK=3
WB="$SRC/ctf_1m_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
[ -f "$ALN" ] || { echo "no aln $ALN"; exit 1; }
echo "════════ CTF AA-1M on ${ALABEL} — $(hostname) $(date -Iseconds) | nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "TARGETS (CPU MF-phase, measured): np2=3076.9s  np4=1974.5s ; oracle best=LG+G4 lnL~-78605196"

# ---------- (A0) subsample ----------
echo; echo "──── subsample 1M -> ${KSUB} sites (seed 1) ────"
T_SUB0=$(date +%s)
python3 - "$ALN" "$KSUB" <<'PY'
import sys, random
src, K = sys.argv[1], int(sys.argv[2])
with open(src) as f:
    ntax, nsit = map(int, f.readline().split())
    names, seqs = [], []
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
T_SUB=$(( $(date +%s)-T_SUB0 )); echo "  subsample wall ${T_SUB}s"

# ---------- (A) coarse rank: stock CPU -m TESTONLY on the subsample (breadth) ----------
echo; echo "──── COARSE: -m TESTONLY on sub.phy (CPU breadth, builds subsample tree) ────"
T_C0=$(date +%s)
"$BIN" -m TESTONLY -s "$WB/sub.phy" -nt "$NT" -pre "$WB/coarse" -redo > "$WB/coarse.stdout" 2>&1
RC_C=$?; T_C=$(( $(date +%s)-T_C0 ))
echo "  coarse exit=$RC_C wall=${T_C}s ; tree=$WB/coarse.treefile"
[ -f "$WB/coarse.treefile" ] || { echo "  COARSE FAILED (no tree)"; tail -20 "$WB/coarse.stdout"; exit 1; }

# ---------- (B) scale-consistent BIC -> top-k ----------
echo; echo "──── RANK by scale-consistent BIC' (N=${NFULL}) -> top-${TOPK} ────"
python3 - "$WB/coarse.iqtree" "$KSUB" "$NFULL" "$TOPK" > "$WB/topk.txt" <<'PY'
import sys, re, math
iq, m, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# row: NAME LogL  AIC +/- wAIC  AICc +/- wAICc  BIC +/- wBIC  -> capture name, LogL, BIC (group 5)
row=re.compile(r'^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]\s+\S+\s+(\d+\.\d+)\s+[+-]')
rows=[]
for line in open(iq):
    mm=row.match(line)
    if not mm: continue
    name, logl, bic_sub = mm.group(1), float(mm.group(2)), float(mm.group(5))
    p = (bic_sub + 2*logl)/math.log(m)          # recover #params from printed subsample BIC
    bicp = -2*(N/m)*logl + p*math.log(N)        # scale-consistent BIC
    rows.append((bicp, name, logl, round(p)))
rows.sort()
for bicp,name,logl,p in rows[:8]:
    print(f"# {name}\tBICp={bicp:.1f}\tsubLogL={logl:.3f}\tp={p}")
# emit just the top-k model names (first token, machine-readable) on lines starting MODEL:
for bicp,name,logl,p in rows[:K]:
    print(f"MODEL:{name}")
PY
cat "$WB/topk.txt" | grep '^#' | sed 's/^# /  /'
mapfile -t TOPMODELS < <(grep '^MODEL:' "$WB/topk.txt" | sed 's/^MODEL://')
echo "  top-${TOPK} to refine: ${TOPMODELS[*]}"

# ---------- (C) fine refine: each top-k model on FULL 1M with --jolt ----------
echo; echo "──── FINE: refine top-${TOPK} on full 1M (--jolt; +I auto CPU-fallback) ────"
export JOLT_DEBUG=1
T_R_TOTAL=0
declare -A FULL_LNL
i=0
for M in "${TOPMODELS[@]}"; do
  i=$((i+1))
  echo "  [refine $i/${TOPK}] $M"
  T_R0=$(date +%s)
  "$BIN" --jolt --gpu -m "$M" -s "$ALN" -te "$WB/coarse.treefile" -nt "$NT" -pre "$WB/refine_${i}" -redo \
      > "$WB/refine_${i}.stdout" 2>&1
  RC=$?; T_R=$(( $(date +%s)-T_R0 )); T_R_TOTAL=$((T_R_TOTAL+T_R))
  lnl=$(grep -iE "Log-likelihood of the tree|BEST SCORE FOUND|Optimal log-likelihood" "$WB/refine_${i}.log" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  jolt=$(grep -c '\[JOLT\]' "$WB/refine_${i}.stdout" 2>/dev/null)
  echo "     exit=$RC wall=${T_R}s lnL=${lnl:-NA} jolt_engaged=${jolt}"
  FULL_LNL[$M]=${lnl:-NA}
done

# ---------- report ----------
echo; echo "════════ CTF RESULT (${ALABEL}) ════════"
python3 - "$NFULL" "$T_SUB" "$T_C" "$T_R_TOTAL" "$WB/topk.txt" <<PY
import sys, math
N=int(sys.argv[1]); tsub=int(sys.argv[2]); tc=int(sys.argv[3]); tr=int(sys.argv[4])
# recover p per top model from topk.txt
pmap={}; logl_sub={}
for line in open(sys.argv[5]):
    if line.startswith('# '):
        parts=line[2:].split('\t'); nm=parts[0]
        for tok in parts:
            if tok.startswith('p='): pmap[nm]=int(tok[2:])
fulls={}
PY
# compute full BIC from refine lnL + recovered p
python3 - <<PY
import re, math, glob, os
N=$NFULL
# parse recovered p from topk.txt
pmap={}
for line in open("$WB/topk.txt"):
    if line.startswith('# '):
        f=line[2:].strip().split('\t'); nm=f[0]
        for t in f:
            if t.startswith('p='): pmap[nm]=int(t[2:])
models="""${TOPMODELS[*]}""".split()
best=None
print(f"{'model':14} {'full_lnL':>16} {'p':>4} {'full_BIC':>18}")
for i,M in enumerate(models,1):
    log=f"$WB/refine_{i}.log"
    lnl=None
    if os.path.exists(log):
        for pat in ("Log-likelihood of the tree","BEST SCORE FOUND","Optimal log-likelihood"):
            for line in open(log):
                if pat in line:
                    mm=re.search(r'-?\d+\.\d+', line)
                    if mm: lnl=float(mm.group(0)); break
            if lnl is not None: break
    p=pmap.get(M, None)
    if lnl is None or p is None:
        print(f"{M:14} {'NA':>16} {str(p):>4} {'NA':>18}"); continue
    bic=-2*lnl + p*math.log(N)
    print(f"{M:14} {lnl:16.3f} {p:4d} {bic:18.1f}")
    if best is None or bic<best[1]: best=(M,bic,lnl)
print()
if best: print(f"CTF WINNER: {best[0]}  full lnL={best[2]:.3f}  full BIC={best[1]:.1f}")
print(f"  (oracle: FCA picked LG+G4, lnL ~ -78605196.4)")
PY
TOTAL=$((T_SUB+T_C+T_R_TOTAL))
echo
echo "  WALL BREAKDOWN: subsample ${T_SUB}s + coarse ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${TOTAL}s"
echo "  vs FCA np2 3076.9s -> $(python3 -c "print(f'{3076.9/$TOTAL:.2f}x faster' if $TOTAL>0 else 'NA')")  | $([ $TOTAL -lt 3077 ] && echo 'BEATS 2 NODES ✓' || echo 'loses to 2 nodes')"
echo "  vs FCA np4 1974.5s -> $(python3 -c "print(f'{1974.5/$TOTAL:.2f}x faster' if $TOTAL>0 else 'NA')")  | $([ $TOTAL -lt 1975 ] && echo 'BEATS 4 NODES ✓' || echo 'loses to 4 nodes')"
echo "════════ DONE $(date -Iseconds) ════════"
