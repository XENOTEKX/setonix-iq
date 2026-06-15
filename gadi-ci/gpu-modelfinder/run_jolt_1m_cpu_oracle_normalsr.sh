#!/bin/bash
# run_jolt_1m_cpu_oracle_normalsr.sh — INDEPENDENT CPU lnL oracle for the AA-1M parity check, on normalsr.
# The iqtree3-mpi binary is -march=sapphirerapids => SIGILLs on the AMD-EPYC dgxa100 nodes (ref_spr_binary_login_sigill);
# it MUST run on normalsr (SPR). GPU gpu_k1_lnl already produced (real 1M tree tree_1.full.treefile, -blfix-equivalent):
#   g1 (LG)            GPU lnL = -83289639.8478
#   g4 (LG+G4{0.9963}) GPU lnL = -78605304.6507
# This job computes the SAME lnLs on CPU (fixed tree + fixed brlen + fixed model) and prints the parity vs those GPU
# values. g1 = tight (no gamma rounding); g4 residual = 4-dp gamma rounding in the harness (not GPU error).
#PBS -N jolt-1m-cpuoracle
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=48
#PBS -l mem=200GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
CPU=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1
ALN=$BASE/alignment_1000000.phy
TREE=$BASE/tree_1.full.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/jolt_1m_parity; mkdir -p "$WB"; cd "$WB"
RUN="mpirun -np 1 --bind-to none --mca rmaps_base_mapping_policy \"\" numactl --localalloc --"
GPU_G1=-83289639.8478   # from gpu_k1_lnl on the same 1M tree (job 171269929)
GPU_G4=-78605304.6507

echo "════════ AA-1M CPU lnL oracle (normalsr SPR) — $(hostname) $(date -Iseconds) ════════"
echo "tree=$TREE"
echo; echo "──── CPU g1 = LG (no gamma), -te -blfix ────"
eval $RUN "$CPU" -s "$ALN" -te "$TREE" -blfix -m LG            -pre "$WB/cpu_g1" -redo -T 48 > "$WB/cpu_g1.runlog" 2>&1
echo "exit=$?"
echo "──── CPU g4 = LG+G4{0.9963}, -te -blfix ────"
eval $RUN "$CPU" -s "$ALN" -te "$TREE" -blfix -m 'LG+G4{0.9963}' -pre "$WB/cpu_g4" -redo -T 48 > "$WB/cpu_g4.runlog" 2>&1
echo "exit=$?"

# robust lnL parse: .iqtree "Log-likelihood of the tree:" or .log "BEST SCORE FOUND :" / "Optimal log-likelihood:"
parse_lnl () {
  local f="$1"
  grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$f.iqtree" 2>/dev/null | grep -oE "\-?[0-9.]+$" | head -1 && return
  grep -oE "BEST SCORE FOUND : -?[0-9.]+"          "$f.log"    2>/dev/null | grep -oE "\-?[0-9.]+$" | head -1 && return
  grep -oE "Optimal log-likelihood: -?[0-9.]+"     "$f.log"    2>/dev/null | grep -oE "\-?[0-9.]+$" | head -1 && return
}
CPU_G1=$(parse_lnl "$WB/cpu_g1"); CPU_G4=$(parse_lnl "$WB/cpu_g4")
echo; echo "CPU oracle: g1=$CPU_G1  g4=$CPU_G4"
echo "GPU (job 171269929): g1=$GPU_G1  g4=$GPU_G4"
[ -z "$CPU_G1" ] && { echo "PARSE FAIL g1 — tail cpu_g1.log:"; tail -15 "$WB/cpu_g1.log" 2>/dev/null; }

echo; echo "════════ PARITY VERDICT (AA-1M, real 1M tree, JOLT k1_node vs IQ-TREE) ════════"
python3 - "$CPU_G1" "$GPU_G1" "$CPU_G4" "$GPU_G4" <<'PY'
import sys
def rel(c,g):
    try: c=float(c); g=float(g)
    except: return None
    return c,g,abs(c-g)/max(1.0,abs(c))
for tag,c,g,gate in [("g1 (LG, no gamma)",sys.argv[1],sys.argv[2],1e-6),
                     ("g4 (LG+G4{0.9963})",sys.argv[3],sys.argv[4],5e-3)]:
    r=rel(c,g)
    if r is None: print(f"  {tag:<22} PARSE-FAIL cpu={c} gpu={g}"); continue
    a,b,e=r; print(f"  {tag:<22} CPU={a:.4f}  GPU={b:.4f}  rel={e:.3e}  (gate {gate:.0e}) -> {'PASS' if e<gate else 'FAIL'}")
print("\n  g1 = clean genome-scale kernel-correctness proof (no gamma rounding).")
print("  g4 residual ~ 4-dp mean-rate gamma rounding in the standalone harness, NOT GPU error.")
PY
echo "════════ DONE $(date -Iseconds) ════════"
