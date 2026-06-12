#!/bin/bash
# probe_rapl_cpu.sh — confirm (1) CPU package RAPL energy_uj is user-readable on a normalsr compute node, and
# (2) IQ-TREE uses ALL cores when run NORMALLY (proving the "1 core detected" in the perf-report probe was the
# Linaro sampler, not the binary). Reads RAPL before/after a quick full-core -m TESTONLY (2000 sites), no profiler.
# Submit: qsub -q normalsr -l ncpus=104 -l mem=200GB -l walltime=00:15:00 gadi-ci/gpu-modelfinder/probe_rapl_cpu.sh
#PBS -N raplprobe
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
BIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB=/scratch/rc29/as1708/iqtree3-mf-iso/raplprobe_$PBS_JOBID; mkdir -p "$WB"; cd "$WB"
echo "════════ RAPL probe — $(hostname) $(date -Iseconds) ════════"
echo "──── powercap domains ────"
ls -l /sys/class/powercap/ 2>/dev/null
for d in /sys/class/powercap/intel-rapl:*; do
  [ -e "$d/energy_uj" ] || continue
  nm=$(cat "$d/name" 2>/dev/null); v=$(cat "$d/energy_uj" 2>/dev/null && echo "READABLE" || echo "DENIED")
  echo "  $d name=$nm energy_uj=$v"
done
# sum all package domains (intel-rapl:N, not the :N:M subzones)
rapl_sum(){ local s=0; for d in /sys/class/powercap/intel-rapl:[0-9]*; do [ -e "$d/energy_uj" ] || continue; local x=$(cat "$d/energy_uj" 2>/dev/null||echo 0); s=$((s+x)); done; echo "$s"; }
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
print("wrote sub.phy",len(seqs),"x",K)
PY
echo "──── run iqtree3-mpi -m TESTONLY -nt 104 NORMALLY (no profiler) ────"
E0=$(rapl_sum); T0=$(date +%s)
"$BIN" -m TESTONLY -s "$WB/sub.phy" -nt 104 -pre "$WB/sub" -redo > "$WB/sub.stdout" 2>&1
RC=$?; T1=$(date +%s); E1=$(rapl_sum)
WALL=$((T1-T0)); DE_UJ=$((E1-E0));
echo "iqtree exit=$RC wall=${WALL}s"
grep -iE "threads|CPU cores detected|Kernel:" "$WB/sub.stdout" | head -3
python3 -c "print(f'  RAPL package energy delta = {$DE_UJ/1e6:.1f} J = {$DE_UJ/3.6e9:.4f} Wh over {$WALL}s -> mean {$DE_UJ/1e6/max($WALL,1):.0f} W (full-node)')" 2>/dev/null || echo "  (rapl counter may have wrapped)"
echo "  best model: $(grep -iE 'Best-fit model' "$WB/sub.iqtree" 2>/dev/null | head -1)"
echo "════════ DONE $(date -Iseconds) ════════"
