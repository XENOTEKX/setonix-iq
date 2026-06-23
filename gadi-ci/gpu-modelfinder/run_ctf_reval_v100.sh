#!/bin/bash
# run_ctf_reval_v100.sh — incremental re-validation after the CTF SIGSEGV fix
# (PLL-init before computeInitialTree on the coarse_tree) + self-test prod-noise fix.
# build-gpu-on persists, so this only recompiles phylotesting.cpp + relinks (~2 min),
# then runs the CTF crash repro end-to-end + a real-data CTF + JOLT/CPU banner checks.
#
#PBS -N ctf-reval
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
SRC=/scratch/rc29/as1708/iqtree3-gpu
ON=$SRC/build-gpu-on
ALN=$SRC/example/example.phy
WORK=$SRC/ctf_reval_${PBS_JOBID:-local}
mkdir -p "$WORK"

module load cmake/3.24.2 gcc/12.2.0 cuda/12.5.1 eigen/3.3.7 boost/1.84.0 2>/dev/null
export LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

echo "════ CTF re-validation $(hostname) $(date -Iseconds) ════"
echo "src HEAD: $(cd $SRC && git rev-parse --short HEAD) (+ CTF crash fix, uncommitted)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

echo; echo "──── incremental rebuild (phylotesting.cpp + relink) ────"
( cd "$ON" && make -j12 > "$WORK/make.log" 2>&1 ); RC=${PIPESTATUS[0]}
echo "  make exit=$RC (last 4):"; tail -4 "$WORK/make.log" | sed 's/^/    /'
BON=$ON/iqtree3
[ -x "$BON" ] && echo "  GPU bin md5: $(md5sum "$BON"|cut -d' ' -f1)"
[ $RC -ne 0 ] && { echo "BUILD FAILED — aborting smoke"; exit 1; }

run() { local tag="$1"; shift; echo; echo "──── $tag ────"; ( cd "$WORK" && "$@" ) > "$WORK/$tag.out" 2>&1; echo "  exit=$?"; }

# T1: the exact command that SIGSEGV'd before the fix (small subsample, top-2)
run T1_ctf    "$BON" --ctf -s "$ALN" --ctf-subsample 300 --ctf-topk 2 -nt 4 -pre "$WORK/t1" -redo
# T1b: a realistic CTF (full default 5000-cap clamps to 1998; top-3) to exercise the refine loop
run T1b_ctf   "$BON" --ctf -s "$ALN" -nt 4 -pre "$WORK/t1b" -redo
# T2/T4: banner-only sanity (already known-good; cheap re-confirm)
run T2_jolt   "$BON" --jolt -s "$ALN" -m GTR+G4 -nt 4 -pre "$WORK/t2" -redo

echo; echo "════════ RE-VAL SUMMARY ════════"
chk(){ grep -qE "$2" "$WORK/$1.out" 2>/dev/null && echo "  PASS $1: $3" || echo "  FAIL $1: $3"; }
chk T1_ctf  'Kernel:  JOLT \+ CTF'                 "banner 'JOLT + CTF'"
chk T1_ctf  '^GPU:'                                 "GPU info line"
chk T1_ctf  'Best-fit model:'                       "CTF produced a best-fit model (NO CRASH)"
grep -q 'SEGMENTATION FAULT' "$WORK/T1_ctf.out" && echo "  FAIL T1_ctf: still SEGFAULTS" || echo "  PASS T1_ctf: no segfault"
chk T1_ctf  'CTF: coarse native subsample-BIC'      "coarse ranking printed"
chk T1b_ctf 'Best-fit model:'                       "realistic CTF (top-3) produced best-fit"
grep -q 'SEGMENTATION FAULT' "$WORK/T1b_ctf.out" && echo "  FAIL T1b_ctf: SEGFAULTS" || echo "  PASS T1b_ctf: no segfault"
# T2 banner: match 'JOLT' as a word but NOT 'JOLT + CTF' (fixed regex — thread suffix follows)
grep -qE 'Kernel:  JOLT - ' "$WORK/T2_jolt.out" && echo "  PASS T2_jolt: banner 'JOLT' (not CTF)" || echo "  FAIL T2_jolt: banner"
# production cleanliness: the self-test fixtures must NOT leak [CTF detector] lines
N_DET=$(grep -c '\[CTF detector\]' "$WORK/T1_ctf.out" 2>/dev/null)
echo "  [CTF detector] lines in T1 (expect 0 without JOLT_DEBUG): $N_DET"

echo; echo "  T1 best:  $(grep -m1 'Best-fit model:' "$WORK/T1_ctf.out" 2>/dev/null)"
echo "  T1b best: $(grep -m1 'Best-fit model:' "$WORK/T1b_ctf.out" 2>/dev/null)"
echo "  T1 coarse top-k:"; grep -E '^    (refine|SKIP)' "$WORK/T1_ctf.out" 2>/dev/null | sed 's/^/    /'
echo "logs: $WORK"; echo "DONE $(date -Iseconds)"
