#!/bin/bash
# probe_pbsdsh_rapl.sh — de-risk the multi-node RAPL sampler before the expensive np4/np8 runs.
# Confirms `pbsdsh -u` launches the sampler on BOTH nodes (2 rapl_<host>.log appear), the STOP sentinel terminates
# them, and energy integrates to sane per-node values under a real 2-node mpirun load. 5000-site TESTONLY (~2 min).
# Submit: qsub -q normalsr -l ncpus=208 -l mem=400GB -l place=excl -l walltime=00:20:00 gadi-ci/gpu-modelfinder/probe_pbsdsh_rapl.sh
#PBS -N pbsraplprobe
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load openmpi/4.1.7 2>/dev/null || true
module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
NRANKS=2; ISO_DIR=/scratch/rc29/as1708/iqtree3-mf-iso
IQTREE="$ISO_DIR/build-mpi-iso/iqtree3-mpi-fca-ws-a2"
ALN=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
WB="$ISO_DIR/pbsrapl_$PBS_JOBID"; mkdir -p "$WB"; cd "$WB"
mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
echo "════════ pbsdsh+RAPL probe — $(date -Iseconds) nodes=${#HOSTS[@]}: ${HOSTS[*]} ════════"
[ "${#HOSTS[@]}" -ge 2 ] || { echo "need 2 nodes, got ${#HOSTS[@]}"; exit 9; }

# subsample 5000 sites
python3 - "$ALN" 5000 <<'PY'
import sys,random
src,K=sys.argv[1],int(sys.argv[2])
with open(src) as f:
    f.readline(); names=[];seqs=[]
    for line in f:
        line=line.rstrip("\n")
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)==2: names.append(p[0]);seqs.append(p[1].replace(" ",""))
L=len(seqs[0]);random.seed(1);cols=sorted(random.sample(range(L),K))
open("sub.phy","w").write(f"{len(seqs)} {K}\n"+"".join(f"{nm}  {''.join(s[c] for c in cols)}\n" for nm,s in zip(names,seqs)))
print("wrote sub.phy")
PY

HOSTFILE="$WB/hostfile.txt"; awk '{c[$1]++} END{for(h in c)print h," slots="c[h]}' "$PBS_NODEFILE" > "$HOSTFILE"
RANKFILE="$WB/rankfile.txt"; : > "$RANKFILE"
for i in 0 1; do echo "rank ${i}=${HOSTS[$i]} slot=0-103" >> "$RANKFILE"; done

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
echo "launching samplers via mpirun -rf rankfile (1 per node) …"
mpirun -np "$NRANKS" --hostfile "$HOSTFILE" --mca rmaps_base_mapping_policy "" -rf "$RANKFILE" -- /bin/bash "$SAMP" & SAMP_PID=$!
sleep 8
OMP_ENV=(-x OMP_NUM_THREADS=103 -x OMP_PROC_BIND=close -x OMP_PLACES=cores)
mpirun -np 2 --hostfile "$HOSTFILE" --mca rmaps_base_mapping_policy "" -rf "$RANKFILE" "${OMP_ENV[@]}" \
    numactl --localalloc -- "$IQTREE" -s "$WB/sub.phy" -m TESTONLY -T 103 -seed 1 -pre "$WB/sub" -redo > "$WB/sub.stdout" 2>&1
RC=$?
touch "$WB/STOP"; sleep 8; kill $SAMP_PID 2>/dev/null; wait $SAMP_PID 2>/dev/null || true

echo "  mpirun exit=$RC  best=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "$WB/sub.iqtree" 2>/dev/null|head -1)"
echo "  RAPL logs: $(ls "$WB"/rapl_*.log 2>/dev/null | wc -l) (expect 2)"
for f in "$WB"/rapl_*.log; do [ -e "$f" ] && echo "    $(basename $f): $(wc -l <$f) samples, first/last: $(head -1 $f|awk '{print $2}')/$(tail -1 $f|awk '{print $2}')"; done
python3 - "$WB" <<'PY'
import glob,os,sys
WB=sys.argv[1]; tot=0
for logf in sorted(glob.glob(WB+"/rapl_*.log")):
    h=os.path.basename(logf)[5:-4]; rf=WB+f"/ranges_{h}.txt"
    rs=[int(l.split()[2]) for l in open(rf)] if os.path.exists(rf) else []; MAX=sum(rs)
    pts=[(int(a),int(b)) for a,b in (l.split() for l in open(logf) if len(l.split())==2 and l.split()[0].isdigit())]
    J=0
    for i in range(1,len(pts)):
        d=pts[i][1]-pts[i-1][1];
        if d<0 and MAX: d+=MAX
        if d<0: d=0
        J+=d/1e6
    dur=pts[-1][0]-pts[0][0] if len(pts)>1 else 0; tot+=J
    print(f"  node {h}: {J:.0f} J over {dur}s -> mean {J/max(dur,1):.0f} W")
print(f"  2-NODE TOTAL = {tot:.0f} J ({tot/3600:.3f} Wh)  [PASS if 2 nodes both ~hundreds W under load]")
PY
echo "════════ DONE $(date -Iseconds) ════════"
