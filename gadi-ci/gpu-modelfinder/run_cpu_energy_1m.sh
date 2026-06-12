#!/bin/bash
# run_cpu_energy_1m.sh — MEASURED single-node CPU ModelFinder energy at AA-1M for the parity table.
# The clean "1 CPU node vs 1 GPU" per-device anchor: vanilla iqtree3-mpi -m TESTONLY (the MF phase, directly
# comparable to the GPU CTF MF phase), 1 normalsr node, mpirun -np 1 --bind-to none -nt 104 (the recipe proven to
# use all 104 cores; bare launch hits OpenMPI singleton 1-core binding). Energy from a RAPL energy_uj sampler that
# sums WRAP-CORRECTED deltas across all package domains (energy_uj wraps ~262 kJ/pkg, so before/after delta is wrong
# over a multi-hour run). Same RAPL source Linaro Forge reports internally; read directly so the full-104-thread
# mpirun launch works (Forge's --no-mpi wrapper precludes it). Seed 1 to match the CPU baselines.
# CRITICAL: allocate the WHOLE node — ncpus=104 (all cores of a normalsr node) + mem=500GB (full-node, baseline
# parity). Gadi supports neither `-l select` nor `-l place=excl`; requesting all 104 cores fills the node so
# nothing else can land. This is REQUIRED for a clean RAPL reading: the counter sums the whole socket regardless
# of tenant, so a shared/partially-allocated node would contaminate the energy. Submit: qsub gadi-ci/gpu-modelfinder/run_cpu_energy_1m.sh
#PBS -N cpuE1m
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
BIN=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB=/scratch/rc29/as1708/iqtree3-mf-iso/cpuE1m_$PBS_JOBID; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 1; }
echo "════════ CPU MF energy AA-1M (1 normalsr node) — $(hostname) $(date -Iseconds) seed=1 ════════"
echo "CPU -m TEST baselines (FCA MPI, multi-node): np2 MF 3076.9s / np4 1974.5s / np16 1122.4s ; lnL -78605196.44 LG+G4"

# RAPL sampler: append 'epoch_s  total_energy_uj_all_pkgs' every 5s; post-process handles wraps.
PWLOG="$WB/rapl.log"
( while true; do
    now=$(date +%s); s=0
    for d in /sys/class/powercap/intel-rapl:[0-9]*; do
      [ -e "$d/energy_uj" ] || continue; x=$(cat "$d/energy_uj" 2>/dev/null||echo 0); s=$((s+x))
    done
    echo "$now $s"; sleep 5
  done ) > "$PWLOG" 2>&1 & PWPID=$!
# record per-package max_energy_range_uj for wrap correction
for d in /sys/class/powercap/intel-rapl:[0-9]*; do
  [ -e "$d/max_energy_range_uj" ] && echo "RANGE $d $(cat $d/max_energy_range_uj)" >> "$WB/rapl_ranges.txt"
done
cat "$WB/rapl_ranges.txt" 2>/dev/null

T0=$(date +%s)
mpirun -np 1 --bind-to none -- "$BIN" -m TESTONLY -s "$ALN" -nt 104 -seed 1 -pre "$WB/mf" -redo > "$WB/mf.stdout" 2>&1
RC=$?; T_MF=$(($(date +%s)-T0)); kill $PWPID 2>/dev/null; sleep 1
echo "iqtree exit=$RC  MF wall=${T_MF}s"

echo; echo "════════ RESULT + ENERGY ════════"
echo "  best model: $(grep -iE 'Best-fit model according to BIC' "$WB/mf.iqtree" 2>/dev/null | head -1)"
grep -iE "^LG\+G4 |Akaike Information|Bayesian Information" "$WB/mf.iqtree" 2>/dev/null | head -4
# integrate RAPL with PER-DOMAIN wrap correction. The log holds the SUM of all domains per timestamp, but each
# domain wraps INDEPENDENTLY (pkg ~262 kJ, dram ~65.7 kJ). A wrap shows as a negative delta ~= the wrapped domain's
# range; adding the *combined* range over-counts ~3x (the original bug -> ~1900 W non-physical). Find the smallest
# non-negative a*PKG+b*DRAM correction giving a plausible per-interval delta.
python3 - "$PWLOG" "$WB/rapl_ranges.txt" <<'PY'
import sys
log=open(sys.argv[1]).read().split("\n")
ranges=sorted(set(int(l.split()[2]) for l in open(sys.argv[2]) if l.startswith("RANGE"))) if len(sys.argv)>2 else []
PKG=ranges[-1] if ranges else 262143328850; DRAM=ranges[0] if ranges else 65712999613
CAP=40e9   # max plausible 4-domain energy per sample interval (uj) ~ 8 kW * 5 s
pts=[]
for l in log:
    p=l.split()
    if len(p)==2 and p[0].isdigit():
        pts.append((int(p[0]),int(p[1])))
J=0.0
for i in range(1,len(pts)):
    d=pts[i][1]-pts[i-1][1]
    if d<0:
        best=None
        for a in range(0,4):
            for b in range(0,5):
                c=d+a*PKG+b*DRAM
                if 0<=c<=CAP and (best is None or c<best): best=c
        d=best if best is not None else 0
    elif d>CAP:
        d=0
    J+=d/1e6
dur=pts[-1][0]-pts[0][0] if len(pts)>1 else 0
print(f"  CPU RAPL energy (MF phase) = {J:.0f} J = {J/3600:.2f} Wh  over {dur}s  -> mean {J/max(dur,1):.0f} W (full node, pkg+dram)")
print(f"  (samples={len(pts)}, per-domain ranges PKG={PKG/1e6:.0f}J DRAM={DRAM/1e6:.0f}J)")
PY
echo "  MF wall ${T_MF}s vs FCA np2 3076.9s / np16 1122.4s (this is vanilla 1-node single-rank; FCA multi-node is faster)"
echo "════════ DONE $(date -Iseconds) ════════"
