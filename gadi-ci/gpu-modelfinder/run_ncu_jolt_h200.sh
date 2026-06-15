#!/bin/bash
# run_ncu_jolt_h200.sh — TRUE SM% on the HEADLINE H200, the right way: Nsight Compute (ncu) on the production
# JOLT kernels. The DCGM sweep failed (hostengine connection, not permission); probe 171184799 proved ncu works
# non-root on gpuhopper (RmProfilingAdminOnly=0). This gives the H200 analogue of the V100 §9.1 k1_node profile:
# SM throughput %, DRAM throughput %, achieved-occupancy %, registers/thread — per production kernel, at scale.
# k1_node = postorder lnL (the dominant kernel, the §9.1 comparison); kj_pre = preorder gradient; kj_derv/kj_ratenum.
# Bounded by --launch-count so ncu's kernel-replay overhead stays at minutes, not the full sweep.
# Submit: qsub gadi-ci/gpu-modelfinder/run_ncu_jolt_h200.sh
#PBS -N ncu-jolt-h200
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=80GB
#PBS -l walltime=00:45:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
WB="$SRC/prof_h200_ncu"; mkdir -p "$WB"; cd "$WB"
PROF="$WB/PROF_SUMMARY.tsv"
KREGEX='regex:k1_node|kj_pre|kj_derv|kj_ratenum'
# robust metrics (sm__throughput + sm__warps_active proven in probe 171184799; dram/duration/registers are standard)
MET='sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum,launch__registers_per_thread'

echo "════════ ncu JOLT profile — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v ncu && ncu --version 2>&1 | head -2
printf "scale\tkernel\tn\tsm_throughput_pct\tdram_throughput_pct\tachieved_occ_pct\tdur_us_mean\tregs_per_thread\n" > "$PROF"

ncu_point () {  # $1=scale  $2=ALN  $3=TREE
  local SC="$1" ALN="$2" TREE="$3"; local PD="$WB/AA_${SC}"; mkdir -p "$PD"; cd "$PD"
  echo; echo "──── AA ${SC}: lean metrics pass (per-kernel SM/DRAM/occ/regs) ────"
  [ -f "$ALN" ] && [ -f "$TREE" ] || { echo "  MISSING aln/tree — skip"; cd "$WB"; return; }
  ncu --target-processes all --kernel-name "$KREGEX" --launch-skip 40 --launch-count 48 --csv \
      --metrics "$MET" \
      "$BIN" --jolt --gpu -m LG+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$PD/prof" -redo \
      > "$PD/metrics.csv" 2> "$PD/metrics.log"
  echo "  ncu exit=$?  rows=$(wc -l < "$PD/metrics.csv")"

  # parse long-format CSV -> per-(kernel,metric) mean -> PROF_SUMMARY row
  python3 - "$PD/metrics.csv" "$SC" >> "$PROF" 2>"$PD/parse.log" <<'PY'
import sys,csv,collections
csvf,sc=sys.argv[1],sys.argv[2]
agg=collections.defaultdict(lambda:collections.defaultdict(list))
with open(csvf, newline='') as f:
    # ncu --csv: header row then data; columns include "Kernel Name","Metric Name","Metric Value"
    rd=csv.reader(f); rows=[r for r in rd if r]
    if not rows: sys.exit(0)
    hdr=None
    for i,r in enumerate(rows):
        if 'Kernel Name' in r and 'Metric Name' in r: hdr=r; data=rows[i+1:]; break
    if hdr is None: sys.exit(0)
    ki=hdr.index('Kernel Name'); mi=hdr.index('Metric Name'); vi=hdr.index('Metric Value')
    def short(k):
        k=k.split('(')[0].strip()
        for n in ('k1_node','kj_pre','kj_ratenum','kj_derv_fused','kj_derv'):
            if n in k: return n
        return k.split()[-1] if k else k
    for r in data:
        if len(r)<=max(ki,mi,vi): continue
        try: v=float(r[vi].replace(',',''))
        except: continue
        agg[short(r[ki])][r[mi].strip()].append(v)
def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
order=['k1_node','kj_pre','kj_derv','kj_derv_fused','kj_ratenum']
for k in sorted(agg, key=lambda x:(order.index(x) if x in order else 99, x)):
    m=agg[k]
    n=max((len(v) for v in m.values()), default=0)
    def g(name):
        for key in m:
            if name in key: return mean(m[key])
        return float('nan')
    sm=g('sm__throughput'); dram=g('dram__throughput'); occ=g('sm__warps_active')
    dur=g('gpu__time_duration'); regs=g('launch__registers_per_thread')
    print(f"{sc}\t{k}\t{n}\t{sm:.1f}\t{dram:.1f}\t{occ:.1f}\t{dur/1000.0:.1f}\t{regs:.0f}")
PY
  cd "$WB"
}

# full-section single-kernel pass on the dominant k1_node at 1M -> the latency-bound stall breakdown on H200
ncu_full_k1 () {
  local SC="$1" ALN="$2" TREE="$3"; local PD="$WB/AA_${SC}"; mkdir -p "$PD"; cd "$PD"
  echo; echo "──── AA ${SC}: --set full on ONE k1_node launch (stall/scoreboard breakdown, H200) ────"
  ncu --target-processes all --kernel-name 'regex:k1_node' --launch-skip 60 --launch-count 1 \
      --set full --export "$PD/k1_full" --force-overwrite \
      "$BIN" --jolt --gpu -m LG+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$PD/proffull" -redo \
      > "$PD/k1_full.log" 2>&1
  echo "  ncu exit=$?  rep=$([ -f "$PD/k1_full.ncu-rep" ] && echo OK || echo NONE)"
  [ -f "$PD/k1_full.ncu-rep" ] && ncu --import "$PD/k1_full.ncu-rep" --page details 2>/dev/null \
      | grep -iE 'Compute \(SM\)|Memory \[%\]|Achieved Occupancy|Duration|Registers Per Thread|Stall|Long Scoreboard|Throughput' \
      | sed 's/^/    /' | tee "$PD/k1_full_summary.txt"
  cd "$WB"
}

ncu_point  100000  "$BASE/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"   "$SRC/bench_h200_dcgm/AA_100000/coarse.treefile"
ncu_point  1000000 "$BASE/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy" "$SRC/bench_h200_dcgm/AA_1000000/coarse.treefile"
ncu_full_k1 1000000 "$BASE/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy" "$SRC/bench_h200_dcgm/AA_1000000/coarse.treefile"

echo; echo "════════ PROF_SUMMARY.tsv (H200 true SM% per JOLT kernel) ════════"
column -t -s$'\t' "$PROF"
echo "════════ DONE $(date -Iseconds) ════════"
