#!/bin/bash
# run_jolt_1m_parity_a100.sh — clean AA-1M lnL PARITY for JOLT k1_node, on the REAL 1M simulation tree.
# Fixes the cosmetic "FAIL" of the scale benchmark's Point 2 (which reused the 100K-fit tree + hard-coded 100K oracle).
# Here: GPU gpu_k1_lnl and an INDEPENDENT CPU IQ-TREE oracle both evaluate the lnL of the SAME 1M alignment on the
# SAME 1M tree (tree_1.full.treefile, branch lengths fixed) with the SAME model:
#   g1  = LG, no rate heterogeneity  -> NO gamma-rate discretisation -> TIGHT parity (gate rel < 1e-6), the clean
#         genome-scale kernel-correctness proof (the postorder accumulation is NCAT-independent).
#   g4  = LG+G4 (alpha 0.9963)       -> the standalone harness hard-codes 4-dp mean-rate gamma {0.1362,0.4756,0.9994,
#         2.3887}; IQ-TREE uses full-precision rates, so the residual here is RATE ROUNDING, not GPU error (proven by
#         the 100K g4 bit-match rel 5.8e-12 vs the same-rounded-rates G.0 oracle). gate rel < 5e-3, report actual.
# lnL is FP64-deterministic and device-independent, so this A100 value == the H200 benchmark value.
#PBS -N jolt-1m-parity
#PBS -P dx61
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=120GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load gcc/12.2.0  2>/dev/null || true
module load cuda/12.5.1 2>/dev/null || true
module load intel-compiler-llvm/2024.2.0 openmpi/4.1.7 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/apps/cuda/12.5.1}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

GMF=/home/272/as1708/setonix-iq/gadi-ci/gpu-modelfinder
CPU=/scratch/rc29/as1708/iqtree3-mf-iso/build-mpi-iso/iqtree3-mpi
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1
ALN=$BASE/alignment_1000000.phy
TREE=$BASE/tree_1.full.treefile
WB=/scratch/rc29/as1708/iqtree3-gpu/jolt_1m_parity; mkdir -p "$WB"; cd "$WB"
RUN="mpirun -np 1 --bind-to none --mca rmaps_base_mapping_policy \"\" numactl --localalloc --"

echo "════════ AA-1M lnL PARITY (JOLT k1_node vs IQ-TREE CPU oracle) — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "tree=$TREE  aln=$ALN"

echo; echo "──── build GPU gpu_k1_lnl (sm_80) ────"
nvcc -O3 -std=c++17 -arch=sm_80 "$GMF/gpu_k1_lnl.cu" -o "$WB/gpu_k1_lnl" && echo "built" || { echo "BUILD FAIL"; exit 1; }

# ---- CPU IQ-TREE oracle: fixed tree, fixed branch lengths, fixed model ----
echo; echo "──── CPU oracle g1 = LG (no gamma), -blfix ────"
eval $RUN "$CPU" -s "$ALN" -te "$TREE" -blfix -m LG -pre "$WB/cpu_g1" -redo -T 16 > "$WB/cpu_g1.runlog" 2>&1
echo "──── CPU oracle g4 = LG+G4{0.9963}, -blfix ────"
eval $RUN "$CPU" -s "$ALN" -te "$TREE" -blfix -m 'LG+G4{0.9963}' -pre "$WB/cpu_g4" -redo -T 16 > "$WB/cpu_g4.runlog" 2>&1

cpu_lnl () { grep -oE "Log-likelihood of the tree: -?[0-9.]+" "$1" 2>/dev/null | grep -oE "\-?[0-9.]+$" | head -1; }
CPU_G1=$(cpu_lnl "$WB/cpu_g1.iqtree"); CPU_G4=$(cpu_lnl "$WB/cpu_g4.iqtree")
echo "CPU lnL: g1=$CPU_G1  g4=$CPU_G4"

# ---- GPU on the SAME tree ----
echo; echo "──── GPU gpu_k1_lnl g1 ────"; "$WB/gpu_k1_lnl" "$ALN" "$TREE" g1 10 2>&1 | tee "$WB/gpu_g1.log"
echo; echo "──── GPU gpu_k1_lnl g4 ────"; "$WB/gpu_k1_lnl" "$ALN" "$TREE" g4 10 2>&1 | tee "$WB/gpu_g4.log"
gpu_lnl () { grep -oE "lnL\(Kahan\)= -?[0-9.]+" "$1" | grep -oE "\-?[0-9.]+$" | head -1; }
GPU_G1=$(gpu_lnl "$WB/gpu_g1.log"); GPU_G4=$(gpu_lnl "$WB/gpu_g4.log")

# ---- compare ----
echo; echo "════════ PARITY VERDICT (AA-1M, real 1M tree) ════════"
python3 - "$CPU_G1" "$GPU_G1" "$CPU_G4" "$GPU_G4" <<'PY'
import sys
def rel(a,b):
    try: a=float(a); b=float(b)
    except: return None,None,None
    return a,b,abs(a-b)/max(1.0,abs(a))
for tag,c,g,gate in [("g1 (LG, no gamma)",sys.argv[1],sys.argv[2],1e-6),
                     ("g4 (LG+G4{0.9963})",sys.argv[3],sys.argv[4],5e-3)]:
    a,b,r=rel(c,g)
    if r is None: print(f"  {tag:<22} PARSE-FAIL  cpu={c} gpu={g}"); continue
    verd="PASS" if r<gate else "FAIL"
    print(f"  {tag:<22} CPU={a:.4f}  GPU={b:.4f}  rel={r:.3e}  (gate {gate:.0e}) -> {verd}")
print()
print("  g1 = clean kernel-correctness proof at 1M (no gamma-rate rounding).")
print("  g4 residual (if ~1e-4) = 4-dp mean-rate gamma rounding in the standalone harness, NOT GPU error")
print("       (100K g4 matches the same-rounded-rates G.0 oracle to rel 5.8e-12).")
PY
echo "════════ DONE $(date -Iseconds) ════════"
