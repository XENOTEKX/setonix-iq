#!/bin/bash
# run_fca_energy_aa1m.sh — multi-node FCA -m TEST AA-1M with WHOLE-CLUSTER RAPL energy (all N nodes).
# Reproduces the np4/np8 parity-table baselines (mpirun -np N -rf rankfile numactl --localalloc iqtree3 -m TEST
# -T 103 -seed 1) and measures CPU package energy on EVERY node via a per-node RAPL sampler launched with
# `pbsdsh -u` (one per node). The existing FCA scripts wrap `perf-report --no-mpi mpirun`, which only profiles the
# HEAD node — remote ranks are invisible — so the multi-node energy was never captured. This fixes that.
# energy_uj wraps (~262 kJ/pkg) so each node's log is integrated with wrap correction, then summed across nodes.
# Submit (WHOLE nodes via ncpus+mem; Gadi supports neither -l select nor -l place=excl):
#   np4: qsub -l ncpus=416 -l mem=2000GB -l walltime=04:00:00 -v NRANKS=4 run_fca_energy_aa1m.sh
#   np8: qsub -l ncpus=832 -l mem=4000GB -l walltime=03:00:00 -v NRANKS=8 run_fca_energy_aa1m.sh
#PBS -N fcaErg
#PBS -P dx61
#PBS -q normalsr
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
# NOTE: allocate WHOLE nodes via total ncpus + mem (Gadi supports neither -l select nor -l place=excl). ncpus=
# NRANKS*104 forces NRANKS full normalsr nodes (104 cores each); mem=NRANKS*500GB matches the FCA baseline. Full
# nodes are REQUIRED: RAPL sums the whole socket regardless of tenant, so a shared node would contaminate energy.
#   np4: qsub -l ncpus=416 -l mem=2000GB -l walltime=04:00:00 -v NRANKS=4 run_fca_energy_aa1m.sh
#   np8: qsub -l ncpus=832 -l mem=4000GB -l walltime=03:00:00 -v NRANKS=8 run_fca_energy_aa1m.sh
set -uo pipefail

NRANKS="${NRANKS:-4}"
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
ISO_DIR=/scratch/rc29/as1708/iqtree3-mf-iso
IQTREE="${IQTREE:-$ISO_DIR/build-mpi-iso/iqtree3-mpi-fca-ws-a2}"   # 1547a906 (no-WS baseline binary a103bc6c overwritten)
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
PBS_ID_SHORT="${PBS_JOBID:-local}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
WB="$ISO_DIR/fcaErg_np${NRANKS}_${PBS_ID_SHORT}"; mkdir -p "$WB"; cd "$WB"

module load openmpi/4.1.7 2>/dev/null || true
module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
[ -x "$IQTREE" ] && [ -f "$ALN" ] || { echo "missing binary/aln"; exit 2; }
[ -s "${PBS_NODEFILE:-/dev/null}" ] || { echo "PBS_NODEFILE missing"; exit 8; }
md5sum "$IQTREE"
mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
[ "${#HOSTS[@]}" -ge "$NRANKS" ] || { echo "need >=$NRANKS nodes, got ${#HOSTS[@]}"; exit 9; }
echo "════════ FCA -m TEST AA-1M np=${NRANKS} +RAPL — $(date -Iseconds) nodes=${#HOSTS[@]} seed=$SEED ════════"
echo "baseline (no-WS): np4 MF 1974.5s/total 5956.6s ; np8 MF 1443.9s/total 3671.6s ; lnL ~-78605196.45 LG+G4"

HOSTFILE="$WB/hostfile.txt"; awk '{c[$1]++} END{for(h in c)print h," slots="c[h]}' "$PBS_NODEFILE" > "$HOSTFILE"
RANKFILE="$WB/rankfile.txt"; : > "$RANKFILE"
for i in $(seq 0 $((NRANKS-1))); do echo "rank ${i}=${HOSTS[$i]} slot=0-103" >> "$RANKFILE"; done

# ---- per-node RAPL sampler (WB baked in; STOP sentinel on shared Lustre terminates all nodes) ----
SAMP="$WB/rapl_sampler.sh"
cat > "$SAMP" <<EOF
#!/bin/bash
WB="$WB"; h=\$(hostname -s); OUT="\$WB/rapl_\${h}.log"; RG="\$WB/ranges_\${h}.txt"
: > "\$RG"; for d in /sys/class/powercap/intel-rapl:[0-9]*; do [ -e "\$d/max_energy_range_uj" ] && echo "RANGE \$d \$(cat \$d/max_energy_range_uj)" >> "\$RG"; done
: > "\$OUT"
while [ ! -f "\$WB/STOP" ]; do
  now=\$(date +%s); s=0
  for d in /sys/class/powercap/intel-rapl:[0-9]*; do [ -e "\$d/energy_uj" ] || continue; x=\$(cat "\$d/energy_uj" 2>/dev/null||echo 0); s=\$((s+x)); done
  echo "\$now \$s" >> "\$OUT"; sleep 5
done
EOF
chmod +x "$SAMP"
echo "launching RAPL samplers on all $NRANKS nodes via mpirun -rf rankfile (1 per node; pbsdsh has no -u on Gadi) …"
mpirun -np "$NRANKS" --hostfile "$HOSTFILE" --mca rmaps_base_mapping_policy "" -rf "$RANKFILE" -- /bin/bash "$SAMP" &
SAMP_PID=$!
sleep 8   # let every node write at least one sample before compute starts

export KMP_BLOCKTIME=200; export TMPDIR="$ISO_DIR/tmp"; mkdir -p "$TMPDIR"
OMP_ENV=(-x OMP_NUM_THREADS=$OMP_PER_RANK -x OMP_DYNAMIC=false -x OMP_PROC_BIND=close -x OMP_PLACES=cores -x OMP_WAIT_POLICY=PASSIVE -x GOMP_SPINCOUNT=10000 -x KMP_BLOCKTIME=200)

T0=$(date +%s)
mpirun -np "$NRANKS" --hostfile "$HOSTFILE" --mca rmaps_base_mapping_policy "" -rf "$RANKFILE" \
    --output-filename "$WB/rank_logs/" "${OMP_ENV[@]}" \
    numactl --localalloc -- \
    "$IQTREE" -s "$ALN" -m TEST -T "$OMP_PER_RANK" -seed "$SEED" --prefix "$WB/iqtree_run" \
    > "$WB/iqtree_run.log" 2>&1
RC=$?; T_TOTAL=$(($(date +%s)-T0))
touch "$WB/STOP"; sleep 8; kill $SAMP_PID 2>/dev/null; wait $SAMP_PID 2>/dev/null || true

echo; echo "════════ RESULT + WHOLE-CLUSTER ENERGY (np=${NRANKS}) ════════"
echo "  exit=$RC  total wall=${T_TOTAL}s"
LNL=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "$WB/iqtree_run.log" 2>/dev/null | tail -1)
[ -z "$LNL" ] && LNL=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "$WB/iqtree_run.log" 2>/dev/null | tail -1)
BEST=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "$WB/iqtree_run.log" 2>/dev/null | head -1)
MF_WALL=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "$WB/iqtree_run.log" 2>/dev/null | head -1)
echo "  best model: ${BEST:-NA}  lnL: ${LNL:-NA}  MF wall: ${MF_WALL:-NA}s  (tree = total - MF)"
echo "  RAPL logs captured: $(ls "$WB"/rapl_*.log 2>/dev/null | wc -l) of $NRANKS nodes"
python3 - "$WB" "$NRANKS" <<'PY'
import sys,glob,os
WB,N=sys.argv[1],int(sys.argv[2])
def integ(logf,rangef):
    # Per-domain wrap correction: energy_uj wraps INDEPENDENTLY per domain (pkg ~262 kJ, dram ~65.7 kJ). The log
    # holds the SUM of all domains, so a wrap shows as a negative delta ~= the wrapped domain's range. Adding the
    # *combined* range over-counts ~3x (the original bug). Find the smallest non-negative a*PKG+b*DRAM correction
    # giving a physically plausible per-interval delta.
    rs=sorted(set(int(l.split()[2]) for l in open(rangef))) if os.path.exists(rangef) else []
    PKG=rs[-1] if rs else 262143328850; DRAM=rs[0] if rs else 65712999613
    CAP=40e9   # max plausible 4-domain energy per sample interval (uj) ~ 8 kW * 5 s
    pts=[(int(a),int(b)) for a,b in (l.split() for l in open(logf) if len(l.split())==2 and l.split()[0].isdigit())]
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
    return J,dur,len(pts)
tot=0.0; per=[]
for logf in sorted(glob.glob(os.path.join(WB,"rapl_*.log"))):
    h=os.path.basename(logf)[5:-4]
    J,dur,n=integ(logf, os.path.join(WB,f"ranges_{h}.txt"))
    tot+=J; per.append((h,J,dur,n))
for h,J,dur,n in per: print(f"    node {h}: {J:.0f} J ({J/3600:.2f} Wh) over {dur}s, {n} samples, mean {J/max(dur,1):.0f} W")
print(f"  CLUSTER TOTAL CPU energy = {tot:.0f} J = {tot/3600:.2f} Wh = {tot/3.6e6:.3f} kWh  (sum of {len(per)} nodes)")
PY
echo "════════ DONE $(date -Iseconds) ════════"
