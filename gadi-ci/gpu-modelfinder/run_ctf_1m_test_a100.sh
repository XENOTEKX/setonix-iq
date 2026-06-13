#!/bin/bash
# run_ctf_1m_test_a100.sh — RE-RUN of the AA-1M -m TEST CTF on A100, FULL PARITY with the prior -m TEST runs
# (the 893s H200 / 1355s A100 parity rows), but on the OPTIMIZATION-FROZEN binary iqtree3.frozen_ab (md5 b85d482f =
# G.5.0 PartB + kernel-fusion + base-sweep-skip + d_theta-reclaim, == the -m MF parity runs 170756438/440). The 1355s
# A100 row (job 170636493) used a PartA-ONLY binary; this measures whether PartB+fusion now puts A100 under np16 (1122s).
# IDENTICAL to run_ctf_1m_v2.sh in every other respect: -m TESTONLY CPU coarse, projected-BIC top-k gate (harmless on
# -m TEST — no +R to amplify, §X.5.4), +I 4-start JOLT refine, seed-1 5000-site subsample, NFULL=940000, TOPK=3.
# Submit: qsub -q dgxa100 -l ngpus=1 -l ncpus=16 -l mem=180GB -l walltime=03:00:00 -v ALABEL=a100test \
#              -l storage=scratch/dx61+scratch/rc29 -l wd run_ctf_1m_test_a100.sh
#PBS -N ctf1mtst
#PBS -P dx61
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
ALABEL="${ALABEL:-a100test}"; NT="${PBS_NCPUS:-16}"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3.frozen_ab"  # frozen parity binary md5 b85d482f
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
NFULL=940000; KSUB=5000; TOPK=3
WB="$SRC/ctf1mtst_${ALABEL}"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ CTF -m TEST (re-run, frozen_ab) AA-1M on ${ALABEL} — $(hostname) $(date -Iseconds) nt=$NT ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo "BIN md5=$(md5sum "$BIN"|cut -d' ' -f1) (parity == -m MF runs 170756438/440)"
echo "CPU -m TEST MF baselines: np2=3076.9 np4=1974.5 np8=1443.9 np16=1122.4 s ; oracle best=LG+G4 lnL~-78605196.4"
echo "PRIOR -m TEST CTF: H200 893s (1.26x np16) | A100 1355s PartA (loses np16, beats np8) — does PartB+fusion beat np16?"

# whole-run GPU power sampler -> energy (IDENTICAL 2s integrator as the -m MF / 10M parity runs)
PWLOG="$WB/power.log"; ( while true; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PWLOG" 2>&1 & PWPID=$!

# ---------- (A0) subsample (seed 1 — IDENTICAL) ----------
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
[ -f "$WB/coarse.treefile" ] || { echo "COARSE FAILED"; tail -20 "$WB/coarse.stdout"; kill $PWPID 2>/dev/null; exit 1; }

# ---------- (B) scale-consistent BIC' -> top-k (projected gate — IDENTICAL to v2; harmless on -m TEST) ----------
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

# ---------- (C) refine each top-k on FULL 1M with --jolt --gpu (+I 4-start GPU-eligible) ----------
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
kill $PWPID 2>/dev/null; sleep 1

# ---------- report ----------
echo; echo "════════ CTF -m TEST RESULT + ENERGY (${ALABEL}) ════════"
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
if best: print(f"CTF -m TEST WINNER: {best[0]}  full lnL={best[2]:.3f}  full BIC={best[1]:.1f}  (oracle LG+G4 lnL~-78605196.4)")
v=[float(x) for x in open("$PWLOG") if x.strip() and x.strip()[0].isdigit()]; dt=2.0; J=sum(v)*dt
print(f"\\nGPU ENERGY: {J:.0f} J = {J/3600:.2f} Wh (mean {sum(v)/max(len(v),1):.0f} W over {len(v)*dt:.0f}s)")
PY
TOTAL=$((T_SUB+T_C+T_R_TOTAL))
echo
echo "  WALL: subsample ${T_SUB}s + coarse ${T_C}s + refine ${T_R_TOTAL}s = TOTAL ${TOTAL}s"
echo "  vs CPU MF: np4 1974.5 -> $(python3 -c "print(f'{1974.5/$TOTAL:.2f}x')") | np8 1443.9 -> $(python3 -c "print(f'{1443.9/$TOTAL:.2f}x')") | np16 1122.4 -> $(python3 -c "print(f'{1122.4/$TOTAL:.2f}x')")  $([ $TOTAL -lt 1122 ] && echo '*** BEATS 16 NODES ✓ ***' || echo '(does not beat 16 nodes)')"
echo "════════ DONE $(date -Iseconds) ════════"
