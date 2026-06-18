#!/bin/bash
# run_g80_dbg_v100.sh — DIAGNOSTIC: why did the [GPU-XCHECK-MIX] cross-check NOT fire for a --gpu (non-jolt) mixture -te
# run (job 171596955: C20/C60/MEOW80 all exit 0 but ZERO [GPU- lines)? Two one-shot stderr probes were added:
#   [CL-DBG]  at computeLikelihood's cross-check decision point (prints gpu/jolt/nmix/ssm)  -> is the site reached?
#   [SLK-DBG] at setLikelihoodKernelGPU entry (prints gpu/jolt/ns/nmix)                      -> is GPU setup reached?
# Tiny 400-site C20 -te on V100 (fast). If [CL-DBG] never prints, computeLikelihood() isn't on the mixture path.
#
# Submit: qsub -q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=60GB -l walltime=00:20:00 \
#              -l storage=scratch/dx61+scratch/rc29 -l wd gadi-ci/gpu-modelfinder/run_g80_dbg_v100.sh
#PBS -N g80dbg
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
DATA=/scratch/rc29/as1708/eukaryote_williamson2025
TREE=/scratch/rc29/as1708/iqtree3-gpu/euk_will2025_run/A_fasttree.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/g80_dbg; mkdir -p "$WB"; cd "$WB"

python3 - "$DATA/CAT_100S93F.phy" 400 "$WB/euk400.phy" <<'PY'
import sys, random
src, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = [ln.rstrip('\n') for ln in open(src)]
ntax, L = map(int, lines[0].split()[:2])
body = [ln for ln in lines[1:] if ln.strip()]
names = [None]*ntax; seqs = ['']*ntax
for i in range(ntax):
    parts = body[i].split(None, 1)
    names[i] = parts[0]; seqs[i] = parts[1].replace(' ','') if len(parts)>1 else ''
for j, ln in enumerate(body[ntax:]):
    seqs[j % ntax] += ln.replace(' ','')
bad = [i for i in range(ntax) if len(seqs[i]) != L]
if bad: sys.stderr.write("PARSE FAIL\n"); sys.exit(3)
random.seed(1); cols = sorted(random.sample(range(L), K))
with open(out,'w') as o:
    o.write("%d %d\n"%(ntax,K))
    for n,s in zip(names,seqs): o.write("%s  %s\n"%(n,''.join(s[c] for c in cols)))
PY
[ -s "$WB/euk400.phy" ] || { echo "FATAL: subsample failed"; exit 1; }

echo "════ G.8 engagement diag — $(hostname) $(date -Iseconds) ════"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
# IMPORTANT: IQ-TREE's own logger writes <prefix>.log; the shell redirect MUST target a DIFFERENT file,
# else IQ-TREE's logger (separate FILE*) clobbers the raw printf/fprintf cross-check+probe output (the
# G.4.3a double-logging artifact). Send console (raw stdout+stderr) to c20dbg.console, IQ-TREE log to c20dbg.log.
"$BIN" --gpu -te "$TREE" -s "$WB/euk400.phy" -m LG+C20+G4 -nt 12 -pre "$WB/c20dbg" -redo > "$WB/c20dbg.console" 2>&1
echo "  exit=$?"
echo "── probes + cross-check (from c20dbg.console, NOT the -pre log) ──"
grep -aE '\[CL-DBG\]|\[SLK-DBG\]|\[GPU-XCHECK-MIX\]|\[GPU-XCHECK\]|\[GPU-KERNEL\]' "$WB/c20dbg.console" | sed 's/^/  /' || echo "  (none)"
echo "════ DONE $(date -Iseconds) ════"
