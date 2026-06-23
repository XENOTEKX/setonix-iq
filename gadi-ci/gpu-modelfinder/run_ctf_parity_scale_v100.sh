#!/bin/bash
# run_ctf_parity_scale_v100.sh — production validation of native --ctf:
#  (1) winner PARITY vs standard -m MF on real DNA + AA (5000 sites, both finish fast);
#  (2) NO-CRASH AT SCALE on the colleague's exact 100K DNA alignment.
# Reuses the already-built binaries (no rebuild).
#
#PBS -N ctf-parity
#PBS -P dx61
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -uo pipefail
SRC=/scratch/rc29/as1708/iqtree3-gpu
GPU=$SRC/build-gpu-on/iqtree3      # JOLT+GPU (md5 46b8d079, validated job 172091310)
CPU=$SRC/build-gpu-off/iqtree3     # CPU-parity reference
DNA=$SRC/mfcov_170602983.gadi-pbs/dna.phy        # 100 x 5000 DNA
AA=$SRC/mfcov_170602983.gadi-pbs/aa.phy          # 100 x 5000 AA
K100=/scratch/rc29/as1708/repro_hashara_171999649.gadi-pbs/alignment_100000.phy  # 100 x 100000 DNA (colleague)
WORK=$SRC/ctf_parity_${PBS_JOBID:-local}
mkdir -p "$WORK"

module load cuda/12.5.1 2>/dev/null
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

echo "════ CTF parity+scale $(hostname) $(date -Iseconds) ════"
echo "GPU bin md5: $(md5sum "$GPU"|cut -d' ' -f1)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

run() { local tag="$1"; shift; echo; echo "──── $tag : $* ────"; ( cd "$WORK" && "$@" ) > "$WORK/$tag.out" 2>&1; echo "  exit=$?"; }
best() { grep -m1 'Best-fit model:' "$WORK/$1.out" 2>/dev/null | sed 's/.*Best-fit model: //; s/ chosen.*//'; }

# ── (1) winner parity: CTF (GPU) vs standard -m MF (CPU reference), DNA + AA ──
run dna_ctf  "$GPU" --ctf -s "$DNA" -nt 12 -pre "$WORK/dna_ctf" -redo
run dna_mf   "$CPU" -m MF -s "$DNA" -nt 12 -pre "$WORK/dna_mf"  -redo
run aa_ctf   "$GPU" --ctf -s "$AA"  -nt 12 -pre "$WORK/aa_ctf"  -redo
run aa_mf    "$CPU" -m MF -s "$AA"   -nt 12 -pre "$WORK/aa_mf"   -redo
# ── (2) scale / no-crash: colleague's 100K DNA, --ctf only (CPU -m MF baseline too slow) ──
run k100_ctf "$GPU" --ctf -s "$K100" -nt 12 -pre "$WORK/k100_ctf" -redo

echo; echo "════════ PARITY + SCALE SUMMARY ════════"
DNA_CTF=$(best dna_ctf); DNA_MF=$(best dna_mf); AA_CTF=$(best aa_ctf); AA_MF=$(best aa_mf); K100=$(best k100_ctf)
echo "  DNA-5000  CTF=[$DNA_CTF]  MF=[$DNA_MF]  $( [ -n "$DNA_CTF" ] && [ "$DNA_CTF" = "$DNA_MF" ] && echo 'PARITY ✓' || echo 'DIFFER ✗' )"
echo "  AA-5000   CTF=[$AA_CTF]  MF=[$AA_MF]  $( [ -n "$AA_CTF" ] && [ "$AA_CTF" = "$AA_MF" ] && echo 'PARITY ✓' || echo 'DIFFER ✗' )"
echo "  DNA-100K  CTF=[$K100]  $( [ -n "$K100" ] && echo 'COMPLETED (no crash) ✓' || echo 'NO BEST-FIT / CRASH ✗' )"
for t in dna_ctf dna_mf aa_ctf aa_mf k100_ctf; do
  grep -q 'SEGMENTATION FAULT' "$WORK/$t.out" 2>/dev/null && echo "  ⚠ $t SEGFAULTED"
done
echo; echo "  100K CTF coarse top-k + timing:"
grep -E '^    (refine|SKIP)|Wall-clock time for CTF' "$WORK/k100_ctf.out" 2>/dev/null | sed 's/^/    /'
echo "logs: $WORK"; echo "DONE $(date -Iseconds)"
