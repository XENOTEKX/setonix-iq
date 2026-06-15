#!/bin/bash
# run_v100_10m_profile.sh — does AA-10M -m MF FIT + run correctly on a 32 GB V100 via pattern tiling, and a
# FULL ncu profile of the production JOLT kernels on the V100. The V100 is the smallest card (32 GB SXM2) — if
# tiling fits 9.25M patterns here, it fits anywhere. Binary has sm_70 (verified). ncu works non-root on gpuvolta
# (the §9.1 V100 profiles were collected there). Both phases seed the ALREADY-OPTIMISED 10M tree so JOLT runs a
# couple of warm iters → bounded wall on the slow card (capability = nTile/VRAM/parity are tree-independent;
# the wall is labelled warm-start, NOT the cold ~14-iter refine).
# PART 1 capability: clean run -> nTile, peak VRAM (must be < 32 GB), GPU lnL, parity vs CPU self-check.
# PART 2 profile: ncu lean --metrics (per-kernel SM/DRAM/occ/regs at the 10M chunk size) + --set full k1_node
#                 (the complete stall/SOL/occupancy breakdown on V100).
# Submit: qsub gadi-ci/gpu-modelfinder/run_v100_10m_profile.sh
#PBS -N v100-10m-prof
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=96GB
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
ALN="$BASE/AA/LG+I+G4/taxa_100/len_10000000/tree_1/alignment_10000000.phy"
TREE="$SRC/bench_h200_dcgm/AA_10000000/refine_1.treefile"   # optimised 10M tree -> bounded warm-start
WB="$SRC/prof_v100_10m"; mkdir -p "$WB"; cd "$WB"
PROF="$WB/PROF_SUMMARY.tsv"; CAP="$WB/CAPABILITY.tsv"
KREGEX='regex:k1_node|kj_pre|kj_derv|kj_ratenum'
MET='sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum,launch__registers_per_thread'

echo "════════ V100 AA-10M tiling + ncu profile — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
command -v ncu && ncu --version 2>&1 | head -2
[ -f "$ALN" ]  || { echo "MISSING aln $ALN";  exit 1; }
[ -f "$TREE" ] || { echo "MISSING tree $TREE"; exit 1; }

# ════════ PART 1 — CAPABILITY: does 10M fit + run correctly on 32 GB V100? ════════
echo; echo "════════ PART 1: capability (clean run, warm-start) ════════"
PD1="$WB/cap"; mkdir -p "$PD1"
( while true; do nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 2; done ) > "$PD1/poll.csv" 2>&1 &
POLL=$!
export JOLT_DEBUG=1
T0=$(date +%s)
"$BIN" --jolt --gpu -m LG+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$PD1/cap" -redo > "$PD1/cap.stdout" 2>&1
RC=$?; WALL=$(($(date +%s)-T0)); kill $POLL 2>/dev/null; sleep 1
NTILE=$(grep -hoE 'nTile=[0-9]+' "$PD1/cap.stdout" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
PEAKV=$(awk -F, 'NF>=2{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print m+0}' "$PD1/poll.csv" 2>/dev/null)
GPULNL=$(grep -hoE '\[JOLT\][^|]*lnL [-0-9.]+' "$PD1/cap.stdout" 2>/dev/null | grep -oE '[-0-9.]+' | tail -1)
PARITY=$(grep -hoE 'rel=[0-9.eE+-]+' "$PD1/cap.stdout" 2>/dev/null | sed 's/rel=//' | sort -g | tail -1)
ITERS=$(grep -hoE '\[JOLT\] [0-9]+ iters' "$PD1/cap.stdout" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
printf "device\twall_s\tnTile\tpeak_vram_mib\tfits_32gb\titers\tgpu_lnL\tparity_rel\texit\n" > "$CAP"
printf "v100\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$WALL" "${NTILE:-NA}" "${PEAKV:-NA}" \
  "$([ "${PEAKV:-999999}" -lt 32000 ] 2>/dev/null && echo YES || echo CHECK)" "${ITERS:-NA}" "${GPULNL:-NA}" "${PARITY:-NA}" "$RC" >> "$CAP"
echo "  exit=$RC wall=${WALL}s nTile=${NTILE:-NA} peakVRAM=${PEAKV:-NA}MiB iters=${ITERS:-NA} lnL=${GPULNL:-NA} parity=${PARITY:-NA}"
echo "  --- [JOLT-TILE] / [JOLT] lines ---"; grep -hE '\[JOLT-TILE\]|\[JOLT\]' "$PD1/cap.stdout" 2>/dev/null | head -8 | sed 's/^/    /'

# ════════ PART 2 — FULL ncu PROFILE of the production kernels on V100 ════════
echo; echo "════════ PART 2: ncu profile ════════"
PD2="$WB/prof"; mkdir -p "$PD2"
printf "kernel\tn\tsm_throughput_pct\tdram_throughput_pct\tachieved_occ_pct\tdur_us_mean\tregs_per_thread\n" > "$PROF"
echo "──── lean metrics pass (per-kernel SM/DRAM/occ/regs at the 10M chunk) ────"
ncu --target-processes all --kernel-name "$KREGEX" --launch-skip 40 --launch-count 60 --csv --metrics "$MET" \
    "$BIN" --jolt --gpu -m LG+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$PD2/prof" -redo \
    > "$PD2/metrics.csv" 2> "$PD2/metrics.log"
echo "  ncu(metrics) exit=$?  rows=$(wc -l < "$PD2/metrics.csv")"
python3 - "$PD2/metrics.csv" >> "$PROF" 2>"$PD2/parse.log" <<'PY'
import sys,csv,collections
csvf=sys.argv[1]
agg=collections.defaultdict(lambda:collections.defaultdict(list))
with open(csvf, newline='') as f:
    rows=[r for r in csv.reader(f) if r]
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
    m=agg[k]; n=max((len(v) for v in m.values()), default=0)
    def g(name):
        for key in m:
            if name in key: return mean(m[key])
        return float('nan')
    print(f"{k}\t{n}\t{g('sm__throughput'):.1f}\t{g('dram__throughput'):.1f}\t{g('sm__warps_active'):.1f}\t{g('gpu__time_duration')/1000.0:.1f}\t{g('launch__registers_per_thread'):.0f}")
PY

echo; echo "──── --set full on ONE k1_node launch (complete stall/SOL/occupancy breakdown, V100) ────"
ncu --target-processes all --kernel-name 'regex:k1_node' --launch-skip 60 --launch-count 1 \
    --set full --export "$PD2/k1_full" --force-overwrite \
    "$BIN" --jolt --gpu -m LG+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$PD2/proffull" -redo \
    > "$PD2/k1_full.log" 2>&1
echo "  ncu(full) exit=$?  rep=$([ -f "$PD2/k1_full.ncu-rep" ] && echo OK || echo NONE)"
[ -f "$PD2/k1_full.ncu-rep" ] && ncu --import "$PD2/k1_full.ncu-rep" --page details 2>/dev/null \
    | grep -iE 'Compute \(SM\)|Memory \[%\]|Achieved Occupancy|Duration|Registers Per Thread|Stall|Long Scoreboard|DRAM Throughput|Throughput' \
    | sed 's/^/    /' | tee "$PD2/k1_full_summary.txt"

echo; echo "════════ CAPABILITY.tsv ════════"; column -t -s$'\t' "$CAP"
echo; echo "════════ PROF_SUMMARY.tsv (V100 true SM% per JOLT kernel @10M chunk) ════════"; column -t -s$'\t' "$PROF"
echo "════════ DONE $(date -Iseconds) ════════"
