#!/bin/bash
# probe_rapl_mpirun.sh — the "1 core detected" was the BARE MPI binary (OpenMPI singleton binds to 1 core), NOT
# perf-report. Confirm that launching via mpirun -np 1 --bind-to none lets iqtree3-mpi see all 104 cores and run
# multithreaded, and measure full-load CPU package power via RAPL. 5000-site TESTONLY (~1-2 min).
# Submit: qsub -q normalsr -l ncpus=104 -l mem=200GB -l walltime=00:15:00 gadi-ci/gpu-modelfinder/probe_rapl_mpirun.sh
#PBS -N raplmpi
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
BIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB=/scratch/rc29/as1708/iqtree3-mf-iso/raplmpi_$PBS_JOBID; mkdir -p "$WB"; cd "$WB"
rapl_sum(){ local s=0; for d in /sys/class/powercap/intel-rapl:[0-9]*; do [ -e "$d/energy_uj" ] || continue; local x=$(cat "$d/energy_uj" 2>/dev/null||echo 0); s=$((s+x)); done; echo "$s"; }
echo "════════ RAPL+mpirun probe — $(hostname) $(date -Iseconds) ════════"
KSUB=5000
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
echo "──── mpirun -np 1 --bind-to none iqtree3-mpi -m TESTONLY -nt 104 ────"
E0=$(rapl_sum); T0=$(date +%s)
mpirun -np 1 --bind-to none -- "$BIN" -m TESTONLY -s "$WB/sub.phy" -nt 104 -pre "$WB/sub" -redo > "$WB/sub.stdout" 2>&1
RC=$?; T1=$(date +%s); E1=$(rapl_sum)
WALL=$((T1-T0)); DE_UJ=$((E1-E0))
echo "iqtree exit=$RC wall=${WALL}s"
grep -iE "threads|CPU cores detected|Kernel:" "$WB/sub.stdout" | head -3
python3 -c "print(f'  RAPL package energy delta = {$DE_UJ/1e6:.1f} J = {$DE_UJ/3.6e9:.4f} Wh over {$WALL}s -> mean {$DE_UJ/1e6/max($WALL,1):.0f} W (full-node, both packages)')"
echo "  best model: $(grep -iE 'Best-fit model|Akaike|Bayesian' "$WB/sub.iqtree" 2>/dev/null | head -1)"
echo "  CPU-time vs wall (>> 1x ratio means multithreaded): see epilogue CPU Time Used below"
echo "════════ DONE $(date -Iseconds) ════════"
