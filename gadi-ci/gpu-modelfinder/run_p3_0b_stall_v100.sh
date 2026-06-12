#!/bin/bash
# run_p3_0b_stall_v100.sh — P3.0b STALL-REASON DISCRIMINATOR (gates promotion of P2∥ occupancy attack).
#
# P3.0 (job 170398260) falsified the bandwidth thesis: k1_node sits at ~33% DRAM, ~49% warps (occupancy-capped
# at 4 blocks/SM by 56 regs), SM% rising 38->56 with nptn. Advisor flagged a contradiction in calling that
# "compute-bound" while also saying "occupancy is the lever" — those fight. The resolution is a THIRD category:
# MEMORY-LATENCY-bound (rising SM% = better latency-hiding as the grid fills, plateauing below 100% BECAUSE
# occupancy is half-capped). Occupancy is the right lever ONLY for that category — so MEASURE it, don't assume.
#
# THE GATE (advisor): re-profile k1_node for the LIMITER, not just throughput %s.
#   * smsp__average_warps_eligible_per_active_cycle.ratio  — if <1, schedulers starve -> more occupancy helps.
#   * warp stall breakdown: stall_long_scoreboard (memory latency) vs math_pipe_throttle / mio_throttle (compute).
#   VERDICT: eligible<1 AND long_scoreboard dominant -> P2∥ CONFIRMED worth building.
#            math/mio-pipe dominant            -> P2∥ DEAD ON ARRIVAL, save the build.
#
# REUSES the subsample alignments already written by P3.0 in $WB (no regeneration). 100K + 300K = scale check.
# -blfix => lnL-only; --gpu (NOT --jolt); skip 3 cold launches, profile 6 steady internal-node launches.
#
#PBS -N p3-0b-stall
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
NCU="${CUDA_HOME:-/apps/cuda/12.5.1}/bin/ncu"
SRC=/scratch/rc29/as1708/iqtree3-gpu; BIN="$SRC/build-gpu-on/iqtree3"
TREE=/scratch/rc29/as1708/iqtree3-mode-p-iso/runs/lfd_modeL_aa100k_np1_seed1_169643959/base/iqtree_inner.treefile
WB="$SRC/p3_0_bwknee"; cd "$WB" || { echo "no $WB (run P3.0 first)"; exit 1; }
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
[ -x "$NCU" ] || { echo "no ncu $NCU"; exit 1; }
echo "════════ P3.0b stall-reason — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# eligible-warps + the full warp-issue stall taxonomy (per-warp-active %, sums ~100%)
ELIG="smsp__average_warps_eligible_per_active_cycle.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active"
STALLS="smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_wait_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_barrier_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_not_selected_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"
STALLS+=",smsp__warp_issue_stalled_drain_per_warp_active.pct"
METRICS="${ELIG},${STALLS}"

for K in 100000 300000; do
  [ -f "aa_${K}.phy" ] || { echo "MISSING aa_${K}.phy in $WB"; continue; }
  echo; echo "════════ nptn≈${K}: k1_node stall breakdown (6 steady launches) ════════"
  timeout 1500 "$NCU" --target-processes all --launch-skip 3 --launch-count 6 \
      --kernel-name "regex:k1_node" --metrics "$METRICS" --csv \
      "$BIN" --gpu -m LG+G4 -s "$WB/aa_${K}.phy" -te "$TREE" -blfix -nt 4 -pre "$WB/stall_${K}" -redo \
      > "$WB/stall_${K}.csv" 2> "$WB/stall_${K}.log"
  echo "  ncu exit $? (last 2 log lines:)"; tail -2 "$WB/stall_${K}.log" | sed 's/^/    /'
done

echo; echo "════════ P3.0b VERDICT: limiter discrimination ════════"
python3 - <<'PY'
import csv, statistics as st
def med(x):
    x=[v for v in x if v==v]
    return st.median(x) if x else float('nan')
SHORT={
 'smsp__average_warps_eligible_per_active_cycle.ratio':'eligible_warps',
 'smsp__issue_active.avg.pct_of_peak_sustained_active':'issue_active%',
 'long_scoreboard':'STALL long_scoreboard (mem-lat)',
 'short_scoreboard':'STALL short_scoreboard',
 'math_pipe_throttle':'STALL math_pipe (compute)',
 'mio_throttle':'STALL mio_throttle',
 'lg_throttle':'STALL lg_throttle (mem-pipe)',
 'stalled_wait':'STALL wait (fixed-lat dep)',
 'stalled_barrier':'STALL barrier',
 'not_selected':'STALL not_selected (have work)',
 'no_instruction':'STALL no_instruction',
 'stalled_drain':'STALL drain',
}
for K in (100000,300000):
    f=f"stall_{K}.csv"
    try: rows=list(csv.reader(open(f)))
    except Exception as e:
        print(f"nptn={K}: no csv ({e})"); continue
    # find header
    hdr=None
    for i,r in enumerate(rows):
        if 'Kernel Name' in r and 'Metric Name' in r: hdr=r; body=rows[i+1:]; break
    if not hdr: print(f"nptn={K}: header not found"); continue
    iK=hdr.index('Kernel Name'); iN=hdr.index('Metric Name'); iV=hdr.index('Metric Value'); iID=hdr.index('ID')
    agg={}
    for r in body:
        if len(r)<=max(iK,iN,iV): continue
        if 'k1_node' not in r[iK]: continue
        try: v=float(r[iV].replace(',',''))
        except: continue
        agg.setdefault(r[iN],[]).append(v)
    print(f"\n── nptn={K} (median over {len(set(r[iID] for r in body if len(r)>iK and 'k1_node' in r[iK]))} launches) ──")
    def show(metric_substr,label):
        for m in agg:
            if metric_substr in m:
                print(f"   {label:<34} {med(agg[m]):8.2f}")
                return med(agg[m])
        return float('nan')
    elig=show('warps_eligible_per_active_cycle','eligible_warps/cyc')
    show('issue_active','issue_active %')
    print("   -- warp stall reasons (% of warp-active cycles, higher = more time stalled here) --")
    pairs=[('long_scoreboard','long_scoreboard (MEM-LATENCY)'),
           ('short_scoreboard','short_scoreboard'),
           ('math_pipe_throttle','math_pipe (COMPUTE)'),
           ('mio_throttle','mio_throttle (MIO/shared)'),
           ('lg_throttle','lg_throttle (mem-pipe)'),
           ('_wait_','wait (fixed-lat dep)'),
           ('_barrier_','barrier'),
           ('not_selected','not_selected (ready, no slot)'),
           ('no_instruction','no_instruction (I-cache)'),
           ('_drain_','drain')]
    vals={}
    for sub,lab in pairs:
        for m in agg:
            if sub in m: vals[lab]=med(agg[m]); break
    for lab,_ in [(l,s) for s,l in pairs]:
        if lab in vals: print(f"      {lab:<32} {vals[lab]:7.2f}")
    # verdict
    ls=vals.get('long_scoreboard (MEM-LATENCY)',float('nan'))
    mp=vals.get('math_pipe (COMPUTE)',float('nan'))
    mio=vals.get('mio_throttle (MIO/shared)',float('nan'))
    print(f"   ==> eligible_warps={elig:.2f} ({'<1 STARVED' if elig<1 else '>=1 not starved'}); "
          f"long_scoreboard={ls:.1f}  math_pipe={mp:.1f}  mio={mio:.1f}")
print("\nGATE: eligible<1 AND long_scoreboard the dominant stall => MEMORY-LATENCY-bound => P2∥ occupancy attack CONFIRMED.")
print("      math_pipe/mio dominant (or eligible>=1) => compute/pipe-bound => P2∥ DEAD ON ARRIVAL, save the build.")
PY
echo; echo "════════ DONE $(date -Iseconds) ════════"
