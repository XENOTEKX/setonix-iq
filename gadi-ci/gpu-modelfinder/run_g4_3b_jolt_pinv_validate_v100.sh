#!/bin/bash
# run_g4_3b_jolt_pinv_validate_v100.sh — G.4.3b validation: JOLT now jointly optimises +I (pinv) for +I+G models.
# Small AA subsample (5000 sites, 100 taxa) on V100. Gates:
#   (A) [JOLT] in-tree self-check rel <= 1e-6  (GPU lnL == fresh CPU computeLikelihood at the converged pinv/alpha
#       => the +I likelihood term L_p = pinv*base_invar + lh is CORRECT, parity vs IQ-TREE's own ptn_invar).
#   (B) JOLT MLE (lnL, pinv, alpha) == IQ-TREE's OWN CPU +I+G4 optimisation (no --jolt) => the joint pinv gradient
#       climbs to the SAME optimum (gradient validated by convergence).
#   (C) regression: +G4 via JOLT (no +I) still PASSes (the pinv=0 path is byte-unchanged).
#PBS -N jolt-iv
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=00:25:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"
BIN=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
SRC=/scratch/rc29/as1708/iqtree3-gpu
ALN=$SRC/ctf_1m_a100/sub.phy
TREE=$SRC/ctf_1m_a100/coarse.treefile
WB=$SRC/jolt_iv; mkdir -p "$WB"; cd "$WB"
[ -x "$BIN" ] && [ -f "$ALN" ] && [ -f "$TREE" ] || { echo "missing inputs"; exit 1; }
echo "════════ G.4.3b +I JOLT validation — $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=name --format=csv,noheader
export JOLT_DEBUG=1

echo; echo "──── (1) LG+I+G4 via JOLT (--jolt --gpu) ────"
"$BIN" --jolt --gpu -m LG+I+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$WB/jolt_ig" -redo > "$WB/jolt_ig.out" 2>&1
grep -E "\[JOLT\]|\[JOLT-GATE\]" "$WB/jolt_ig.out" | head

echo; echo "──── (2) LG+I+G4 pure CPU reference (no --jolt) ────"
"$BIN" -m LG+I+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$WB/cpu_ig" -redo > "$WB/cpu_ig.out" 2>&1
echo "  (done)"

echo; echo "──── (3) LG+G4 via JOLT (regression: pinv=0 path) ────"
"$BIN" --jolt --gpu -m LG+G4 -s "$ALN" -te "$TREE" -nt 12 -pre "$WB/jolt_g" -redo > "$WB/jolt_g.out" 2>&1
grep -E "\[JOLT\]" "$WB/jolt_g.out" | head -2

echo; echo "════════ COMPARISON ════════"
python3 - "$WB" <<'PY'
import sys, re, os
wb=sys.argv[1]
def grab(iqfile):
    lnl=pinv=alpha=None
    if not os.path.exists(iqfile): return lnl,pinv,alpha
    for line in open(iqfile):
        if "Log-likelihood of the tree" in line or "BEST SCORE FOUND" in line:
            m=re.search(r'-?\d+\.\d+', line); lnl=float(m.group()) if m else lnl
        if "Proportion of invariable sites" in line:
            m=re.search(r'[-\d.]+\s*$', line.strip());
            m2=re.findall(r'[\d.]+', line); pinv=float(m2[-1]) if m2 else pinv
        if "Gamma shape alpha" in line:
            m2=re.findall(r'[\d.]+', line); alpha=float(m2[-1]) if m2 else alpha
    return lnl,pinv,alpha
jl,jp,ja=grab(wb+"/jolt_ig.iqtree")
cl,cp,ca=grab(wb+"/cpu_ig.iqtree")
print(f"  LG+I+G4  JOLT : lnL={jl}  pinv={jp}  alpha={ja}")
print(f"  LG+I+G4  CPU  : lnL={cl}  pinv={cp}  alpha={ca}")
if jl and cl:
    dl=abs(jl-cl); rel=dl/abs(cl)
    print(f"  GATE B  |dlnL|={dl:.4f}  rel={rel:.3e}  -> {'PASS (same MLE)' if dl<0.5 else 'CHECK (optimiser landed differently)'}")
    if jp is not None and cp is not None: print(f"          dpinv={abs(jp-cp):.5f}  dalpha={abs((ja or 0)-(ca or 0)):.5f}")
PY
echo
echo "  GATE A (correctness) — JOLT self-check rel from (1):"
grep -oE "rel=[0-9.e+-]+ (PASS|OK\(gamma-resid\)|MISMATCH)" "$WB/jolt_ig.out" | head -1
echo "  GATE C (regression)  — +G4 JOLT self-check from (3):"
grep -oE "rel=[0-9.e+-]+ (PASS|OK\(gamma-resid\)|MISMATCH)" "$WB/jolt_g.out" | head -1
echo "════════ DONE $(date -Iseconds) ════════"
