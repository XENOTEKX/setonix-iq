#!/bin/bash
# probe_perfreport_cpu.sh — QUICK probe: does Linaro Forge perf-report expose CPU/node energy on a Gadi normalsr node?
# The GPU node (gpuvolta) reported "Mean node power: not supported / Cray power not supported", so before committing
# a multi-hour CPU -m TEST energy run we must confirm RAPL/node-power is actually readable here. Subsamples the 1M AA
# to 2000 sites and runs iqtree3-mpi (single process, --no-mpi) under perf-report -m TESTONLY (~2-3 min).
# Submit: qsub -q normalsr -l ncpus=104 -l mem=200GB -l walltime=00:20:00 gadi-ci/gpu-modelfinder/probe_perfreport_cpu.sh
#PBS -N pfprobe
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 2>/dev/null || true
module load linaro-forge/24.0.2
BIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB=/scratch/rc29/as1708/iqtree3-mf-iso/pfprobe_$PBS_JOBID; mkdir -p "$WB"; cd "$WB"
echo "════════ perf-report CPU energy probe — $(hostname) $(date -Iseconds) ════════"
which perf-report; perf-report --version 2>&1 | head -2
KSUB=2000
python3 - "$ALN" "$KSUB" <<'PY'
import sys, random
src,K=sys.argv[1],int(sys.argv[2])
with open(src) as f:
    f.readline(); names=[]; seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]); seqs.append(p[1].replace(" ",""))
L=len(seqs[0]); random.seed(1); cols=sorted(random.sample(range(L),K))
open("sub.phy","w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print("wrote sub.phy", len(seqs),"x",K)
PY
echo "──── running iqtree3-mpi -m TESTONLY under perf-report --no-mpi (2000 sites) ────"
T0=$(date +%s)
perf-report --no-mpi --output="$WB/pfprobe.txt" -- \
    "$BIN" -m TESTONLY -s "$WB/sub.phy" -nt 104 -pre "$WB/sub" -redo
RC=$?
echo "perf-report exit=$RC  wall=$(($(date +%s)-T0))s"
echo "──── full Energy section of the perf-report ────"
ls -l "$WB"/pfprobe.* 2>/dev/null
[ -f "$WB/pfprobe.txt" ] && awk '/^Energy:/{f=1} f{print} /^CPU:/&&f>1{exit}' "$WB/pfprobe.txt" | head -40
echo "──── grep for power/energy keywords ────"
grep -iE "energy|power|watt|joule|rapl|cpu:|accelerator" "$WB/pfprobe.txt" 2>/dev/null | head -30
echo "════════ DONE $(date -Iseconds) ════════"
