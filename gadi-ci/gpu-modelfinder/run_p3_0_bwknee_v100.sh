#!/bin/bash
# run_p3_0_bwknee_v100.sh — P3.0 BANDWIDTH-KNEE KILL-SWITCH (gates the WHOLE GPU scaling thesis).
#
# QUESTION (advisor-sharpened): does the postorder lnL kernel (k1_node) become BANDWIDTH-BOUND as nptn grows,
# or is it latency-bound at EVERY scale (occupancy caps latency-hiding at 16/64 warps regardless of nptn)?
#   climbing DRAM%/GB/s toward saturation -> the 1M/10M bandwidth win is real -> build tiling (P3.1).
#   FLAT ~40% DRAM% -> the "unconditional 1M/10M win" is FALSIFIED -> GPU win is occupancy-gated -> pivot to P2 occupancy moonshot.
#
# METHOD (advisor): profile a SINGLE k1_node launch IN STEADY STATE (skip cold launches), NOT the whole sweep.
# Report ABSOLUTE GB/s + DRAM% + achieved warps/SM + Compute% per nptn. V100 trend 100K->300K (g4 native-20
# OOMs ~400K on 32GB). The 1M point is a SEPARATE A100 job (lnL fits 80GB; --jolt preorder would OOM).
# Profile under -blfix (fixed brlen -> lnL-only, no slow branch-opt; the one-shot GPU lnL cross-check + lnL evals
# launch k1_node). NOT --jolt. Every k1_node launches ceil(nptn/256) blocks regardless of depth, so any
# mid-sweep launch is steady-state.
#
#PBS -N p3-0-bwknee
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NCU="${CUDA_HOME:-/apps/cuda/12.5.1}/bin/ncu"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
ALN1M=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1/alignment_1000000.phy
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB="$SRC/p3_0_bwknee"; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
[ -x "$NCU" ] || { echo "no ncu $NCU"; exit 1; }
echo "════════ P3.0 bandwidth-knee — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "──── subsample 1M -> {100000,200000,300000} sites (random cols, seed=1) ────"
python3 - "$ALN1M" <<'PY'
import sys, random
src = sys.argv[1]
with open(src) as f:
    hdr = f.readline().split(); ntax, nsites = int(hdr[0]), int(hdr[1])
    names, seqs = [], []
    for line in f:
        line = line.rstrip("\n")
        if not line.strip(): continue
        p = line.split(None, 1)
        if len(p) == 2: names.append(p[0]); seqs.append(p[1].replace(" ", ""))
print("read ntax=%d nsites=%d got=%d len0=%d" % (ntax, nsites, len(seqs), len(seqs[0]) if seqs else -1))
L = len(seqs[0])
for K in (100000, 200000, 300000):
    random.seed(1); cols = sorted(random.sample(range(L), K))
    with open("aa_%d.phy" % K, "w") as out:
        out.write("%d %d\n" % (len(seqs), K))
        for nm, s in zip(names, seqs): out.write("%s  %s\n" % (nm, "".join(s[c] for c in cols)))
    print("wrote aa_%d.phy" % K)
PY

METRICS="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__occupancy_limit_registers"
for K in 100000 200000 300000; do
  echo; echo "════════ nptn≈${K}: ncu profile of k1_node (steady-state launches) ════════"
  # skip 3 cold launches, profile 8 steady ones; -blfix => lnL-only (no branch-opt); --gpu (NOT --jolt)
  timeout 1800 "$NCU" --target-processes all --launch-skip 3 --launch-count 8 \
      --kernel-name "regex:k1_node" --metrics "$METRICS" --csv \
      "$BIN" --gpu -m LG+G4 -s "$WB/aa_${K}.phy" -te "$TREE" -blfix -nt 4 -pre "$WB/prof_${K}" -redo \
      > "$WB/ncu_${K}.csv" 2> "$WB/ncu_${K}.log"
  echo "  ncu exit $? (last 3 log lines:)"; tail -3 "$WB/ncu_${K}.log" | sed 's/^/    /'
  echo "  --- per-launch k1_node metrics (nptn=${K}) ---"
  # print the CSV header + the k1_node rows compactly
  grep -iE "k1_node|Kernel Name|dram__throughput|warps_active" "$WB/ncu_${K}.csv" 2>/dev/null | head -12
done

echo; echo "════════ P3.0 SUMMARY: DRAM% / GB/s / warps%/SM trend vs nptn ════════"
python3 - <<'PY'
import csv, glob, os, statistics
def med(xs):
    xs=[x for x in xs if x is not None]
    return statistics.median(xs) if xs else float('nan')
print(f"{'nptn':>8} {'DRAM%':>8} {'GB/s':>10} {'warps%':>8} {'SM%':>8} {'regs':>6} {'dur_us':>9}")
for K in (100000,200000,300000):
    f=f"ncu_{K}.csv"
    if not os.path.exists(f): print(f"{K:>8}  (no csv)"); continue
    dram=[]; bw=[]; wp=[]; sm=[]; rg=[]; dur=[]
    try:
        rows=list(csv.DictReader(open(f)))
    except Exception as e:
        print(f"{K:>8}  parse-fail {e}"); continue
    for r in rows:
        nm=r.get('Kernel Name','') or r.get('"Kernel Name"','')
        if 'k1_node' not in nm: continue
        def g(key):
            for k in r:
                if key in k:
                    v=r[k].replace(',','')
                    try: return float(v)
                    except: return None
            return None
        dram.append(g('dram__throughput.avg.pct')); bw.append(g('dram__bytes.sum.per_second'))
        wp.append(g('sm__warps_active.avg.pct')); sm.append(g('sm__throughput.avg.pct'))
        rg.append(g('registers_per_thread')); dur.append(g('gpu__time_duration'))
    gbps=med(bw)/1e9 if bw and med(bw)==med(bw) else float('nan')
    print(f"{K:>8} {med(dram):>8.1f} {gbps:>10.1f} {med(wp):>8.1f} {med(sm):>8.1f} {med(rg):>6.0f} {med(dur)/1000 if med(dur)==med(dur) else float('nan'):>9.1f}")
print("\nVERDICT: DRAM% climbing toward ~70-90% (GB/s toward ~700-900 on V100) => bandwidth thesis HOLDS -> build tiling.")
print("         DRAM% FLAT ~40% => latency-bound at all scales => 1M/10M win FALSIFIED -> pivot to occupancy (P2-parallel).")
PY
echo; echo "════════ DONE $(date -Iseconds) ════════"
