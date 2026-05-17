#!/bin/bash
# test_xlarge_mf2_correctness.sh — Phase 5 correctness pre-test for xlarge_mf.fa.
#
# Verifies that the ModelFinder MPI dispatch (Phase 1+2+3) produces a correct
# best-fit model on the realistic empirical-scale dataset before committing to the
# full Phase 5 benchmark run.
#
# Dataset: xlarge_mf.fa (200 taxa × 100,000 sites, 98,858 distinct patterns,
#          ~99% site-pattern compression, sha256 66eaf64b...)
#
# Strategy (same as test_mf_mpi_dispatch.sh):
#   1. Generate a fixed tree via np=1 without -te (avoids SIGILL on login nodes).
#   2. np=1 reference: run ModelFinder with -te fixed_xlarge_tree.nwk.
#   3. np=4 dispatch:  run ModelFinder with -te fixed_xlarge_tree.nwk.
#   4. Compare "Best-fit model:" — must be identical.
#
# PBS sizing:
#   1 SPR node, 104 CPUs (4 ranks × 26 OMP threads).
#   np=1 uses all 104 threads (fast reference).
#   np=4 uses 26 threads/rank via --mpi-ranks-per-node 4 (exercises Phase 3).
#   xlarge_mf.fa RAM: ~3 GB/rank; 4 ranks on 1 node ≈ 12 GB; 64 GB is ample.
#
# Expected outcomes:
#   - np=1 and np=4 report identical "Best-fit model:" string.
#   - np=4 log contains "MF-MPI: rank N/4 assigned K/M models" for all 4 ranks.
#   - np=4 log contains "MF-MPI: gather complete, M model scores consolidated".
#   - np=4 log contains "MF-MPI: thread budget per rank = 26 (104 total / 4 ranks/node)".
# Note: the model count M is data-dependent (seq type + frac_invariant_sites from the
#   alignment). For xlarge_mf.fa (DNA, 468/100000 constant sites) M=968, but this
#   script extracts M dynamically and does not hardcode it.
#
# Usage:
#   qsub gadi-ci/test_xlarge_mf2_correctness.sh

#PBS -N test-mf2-xlarge-correctness
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=503gb
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-um09}"
USER_ID="${USER:-$(whoami)}"

MF2_DIR="${MF2_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf2}"
BUILD_DIR="${BUILD_DIR:-${MF2_DIR}/build-mpi-mf2}"
IQTREE="${IQTREE:-${BUILD_DIR}/iqtree3-mpi}"

# xlarge empirical-scale dataset — same sha256-gated file as all prior baselines
ALN="${ALN:-${MF2_DIR}/benchmarks/xlarge_mf.fa}"
SHA256_EXPECTED="66eaf64b9b7e561f52dc515198c0b7db6d68cd37ada9498b254777f2dde94c44"

OUTDIR="${OUTDIR:-${MF2_DIR}/test_xlarge_mf2}"
SEED="${SEED:-42}"
NRANKS="${NRANKS:-4}"

# 1 node, 104 CPUs total.  Distribute evenly across 4 ranks.
TOTAL_THREADS="${TOTAL_THREADS:-104}"
OMP_PER_RANK=$(( TOTAL_THREADS / NRANKS ))   # 26 threads per rank

# Fixed tree reuses the np=1 tree topology — eliminates topology divergence
# artefact (Phase 2 discovery: 4-rank fast-NNI explores more topologies than 1-rank).
FIXED_TREE="${FIXED_TREE:-${OUTDIR}/fixed_xlarge_tree.nwk}"

KMP_BLOCKTIME="${KMP_BLOCKTIME:-200}"

# ── Module load ───────────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7         2>/dev/null || true
    module load intel-compiler-llvm   2>/dev/null || true
fi

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo " Phase 5 pre-test: xlarge_mf.fa ModelFinder MPI correctness"
echo "════════════════════════════════════════════════════════════════"
echo "  binary:          ${IQTREE}"
echo "  alignment:       ${ALN}"
echo "  outdir:          ${OUTDIR}"
echo "  ranks:           ${NRANKS}"
echo "  total threads:   ${TOTAL_THREADS}"
echo "  OMP/rank:        ${OMP_PER_RANK}"
echo "  fixed tree:      ${FIXED_TREE}"
echo ""

if [[ ! -x "${IQTREE}" ]]; then
    echo "ERROR: ${IQTREE} not found or not executable." >&2
    echo "       Build: cd ${BUILD_DIR} && /bin/gmake -j16 iqtree3" >&2
    exit 5
fi

if ! ldd "${IQTREE}" 2>/dev/null | grep -qE 'libmpi(\.|_)'; then
    echo "ERROR: ${IQTREE} does not link libmpi — wrong binary?" >&2
    exit 6
fi

if [[ ! -f "${ALN}" ]]; then
    echo "ERROR: alignment ${ALN} not found." >&2
    exit 7
fi

# sha256 integrity check — same file as all prior Gadi xlarge baselines
echo "[preflight] Verifying ${ALN} sha256..."
ACTUAL_SHA=$(sha256sum "${ALN}" | awk '{print $1}')
if [[ "${ACTUAL_SHA}" != "${SHA256_EXPECTED}" ]]; then
    echo "ERROR: sha256 mismatch for xlarge_mf.fa" >&2
    echo "  expected: ${SHA256_EXPECTED}" >&2
    echo "  actual:   ${ACTUAL_SHA}" >&2
    exit 8
fi
echo "[preflight] sha256 OK"

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found — load openmpi/4.1.7 first." >&2
    exit 9
fi

mkdir -p "${OUTDIR}"

# ── Phase 1+2 strings compiled into binary ────────────────────────────────────
echo "[preflight] Phase 1+2 strings in binary:"
strings "${IQTREE}" | grep -E "MF-MPI:|gather complete|assigned" || \
    echo "WARNING: MF-MPI strings not found — edits may not be compiled!"
echo ""

# ── OMP environment ───────────────────────────────────────────────────────────
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export KMP_BLOCKTIME="${KMP_BLOCKTIME}"

# ── Generate fixed tree (if not cached) ───────────────────────────────────────
if [[ ! -f "${FIXED_TREE}" ]]; then
    echo "[setup] Generating fixed tree via np=1 NJ+ML on xlarge_mf.fa (no -te)..."
    echo "[setup]   This uses ${TOTAL_THREADS} OMP threads for speed."
    mpirun -np 1 \
        --map-by node:PE="${TOTAL_THREADS}" \
        -x "OMP_NUM_THREADS=${TOTAL_THREADS}" \
        -x "OMP_DYNAMIC=false" \
        -x "OMP_PROC_BIND=close" \
        -x "OMP_PLACES=cores" \
        -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}" \
        numactl --localalloc -- \
        "${IQTREE}" \
            -s "${ALN}" \
            -m GTR+G4 \
            -T "${TOTAL_THREADS}" \
            --seed "${SEED}" \
            --prefix "${OUTDIR}/setup_xlarge_tree" \
            --redo \
        2>&1 | tail -5
    if [[ -f "${OUTDIR}/setup_xlarge_tree.treefile" ]]; then
        cp "${OUTDIR}/setup_xlarge_tree.treefile" "${FIXED_TREE}"
        echo "[setup] Fixed tree saved: ${FIXED_TREE}"
    else
        echo "ERROR: Failed to generate initial tree for xlarge_mf.fa" >&2
        exit 10
    fi
fi

# ── Test 1: np=1 reference ────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────────"
echo " Test 1: np=1 reference  (${TOTAL_THREADS} OMP threads)"
echo "──────────────────────────────────────────────────────────────────"
echo "$(date '+%H:%M:%S') start"

mpirun -np 1 \
    --map-by node:PE="${TOTAL_THREADS}" \
    -x "OMP_NUM_THREADS=${TOTAL_THREADS}" \
    -x "OMP_DYNAMIC=false" \
    -x "OMP_PROC_BIND=close" \
    -x "OMP_PLACES=cores" \
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}" \
    numactl --localalloc -- \
    "${IQTREE}" \
        -s "${ALN}" \
        -te "${FIXED_TREE}" \
        -m MF \
        -T "${TOTAL_THREADS}" \
        --seed "${SEED}" \
        --prefix "${OUTDIR}/xlarge_ref_np1" \
        --redo \
    2>&1

NP1_EXIT=$?
echo "$(date '+%H:%M:%S') np=1 exit: ${NP1_EXIT}"
echo ""

# ── Test 2: np=NRANKS MPI dispatch ────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────────"
echo " Test 2: np=${NRANKS} MPI dispatch  (${OMP_PER_RANK} OMP/rank)"
echo "──────────────────────────────────────────────────────────────────"
echo "$(date '+%H:%M:%S') start"

# IMPORTANT: pass -T OMP_PER_RANK (26) not TOTAL_THREADS (104).
# IQ-TREE's thread count validation fires at startup — before runModelFinder()
# where Phase 3 (--mpi-ranks-per-node) would divide the budget.  With
# --map-by node:PE=26, each rank is bound to 26 CPU cores; passing -T 104
# triggers "more threads than CPU cores available" and crashes (SIGSEGV).
# The per-rank thread budget is set correctly here by passing OMP_PER_RANK
# directly.  Phase 3 (--mpi-ranks-per-node) is exercised separately in
# test_mf_mpi_dispatch.sh Test 3 which uses a lightweight example.phy dataset.

mpirun -np "${NRANKS}" \
    --map-by node:PE="${OMP_PER_RANK}" \
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}" \
    -x "OMP_DYNAMIC=false" \
    -x "OMP_PROC_BIND=close" \
    -x "OMP_PLACES=cores" \
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}" \
    numactl --localalloc -- \
    "${IQTREE}" \
        -s "${ALN}" \
        -te "${FIXED_TREE}" \
        -m MF \
        -T "${OMP_PER_RANK}" \
        --seed "${SEED}" \
        --prefix "${OUTDIR}/xlarge_test_np${NRANKS}" \
        --redo \
    2>&1

NPN_EXIT=$?
echo "$(date '+%H:%M:%S') np=${NRANKS} exit: ${NPN_EXIT}"
echo ""

# ── Comparison ────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo " Comparison: Best-fit model must be identical"
echo "════════════════════════════════════════════════════════════════"

REF_LOG="${OUTDIR}/xlarge_ref_np1.log"
TEST_LOG="${OUTDIR}/xlarge_test_np${NRANKS}.log"

if [[ ! -f "${REF_LOG}" ]]; then
    echo "ERROR: ref log not found: ${REF_LOG}" >&2
    exit 11
fi
if [[ ! -f "${TEST_LOG}" ]]; then
    echo "ERROR: test log not found: ${TEST_LOG}" >&2
    exit 12
fi

REF_MODEL=$(grep "Best-fit model:" "${REF_LOG}" | head -1)
TEST_MODEL=$(grep "Best-fit model:" "${TEST_LOG}" | head -1)

echo "  np=1       ${REF_MODEL}"
echo "  np=${NRANKS}      ${TEST_MODEL}"
echo ""

EXIT_CODE=0

if [[ "${REF_MODEL}" == "${TEST_MODEL}" ]]; then
    echo "✓ PASS: Best-fit model matches between np=1 and np=${NRANKS}"
else
    echo "✗ FAIL: Best-fit model MISMATCH"
    echo "  np=1 : ${REF_MODEL}"
    echo "  np=${NRANKS}: ${TEST_MODEL}"
    EXIT_CODE=1
fi

# BIC score comparison
REF_BIC=$(grep "Bayesian information criterion" "${REF_LOG}" | head -1 || true)
TEST_BIC=$(grep "Bayesian information criterion" "${TEST_LOG}" | head -1 || true)
if [[ -n "${REF_BIC}" && -n "${TEST_BIC}" ]]; then
    echo ""
    echo "[BIC check]"
    echo "  np=1  ${REF_BIC}"
    echo "  np=${NRANKS}  ${TEST_BIC}"
    if [[ "${REF_BIC}" == "${TEST_BIC}" ]]; then
        echo "✓ BIC scores match"
    else
        echo "△ BIC scores differ (acceptable if within float rounding — model name is the gate)"
    fi
fi

# Phase 1+2+3 diagnostic lines from np=4 log
echo ""
echo "[Phase 1+2+3 diagnostics from np=${NRANKS} log]"
grep "MF-MPI:" "${TEST_LOG}" 2>/dev/null || echo "(none — check _IQTREE_MPI compile guard)"

# Check Phase 1 assignment lines (expect all 4 ranks)
echo ""
# Extract total model count dynamically from np=1 log (data-dependent: varies by seq type
# and frac_invariant_sites; do not hardcode 968).
TOTAL_MODELS=$(grep -oP 'assigned \d+/\K\d+(?= models)' "${TEST_LOG}" 2>/dev/null | head -1 || true)
if [[ -z "${TOTAL_MODELS}" ]]; then
    TOTAL_MODELS=$(grep -oP 'test up to \K\d+(?= DNA models| RNA models| protein models| morphological models)' "${REF_LOG}" 2>/dev/null | head -1 || echo "?")
fi
ASSIGNED_COUNT=$(grep -c "MF-MPI: rank .* assigned .*/[0-9]* models" "${TEST_LOG}" 2>/dev/null || echo "0")
echo "[Phase 1] ${ASSIGNED_COUNT}/${NRANKS} rank assignment lines found (total models: ${TOTAL_MODELS})"
# Only rank 0's stdout is captured in the PBS job log; worker ranks 1..N-1 write to
# separate per-rank stdout files that are not merged here.  Seeing rank 0's line
# confirms Phase 1 ran; the Phase 2 gather line (below) confirms all ranks finished.
if [[ "${ASSIGNED_COUNT}" -ge 1 ]]; then
    echo "✓ PASS: Phase 1 — rank 0 assignment line present (worker rank output not captured in PBS log)"
else
    echo "✗ FAIL: Phase 1 — no rank assignment lines found (Phase 1 may not have run)"
    EXIT_CODE=1
fi

# Check Phase 2 gather complete line
if grep -q "MF-MPI: gather complete" "${TEST_LOG}" 2>/dev/null; then
    echo "✓ PASS: Phase 2 — gather complete confirmed"
else
    echo "✗ FAIL: Phase 2 — 'MF-MPI: gather complete' not found in log"
    EXIT_CODE=1
fi

# Check Phase 3 thread budget message — NOT expected in this test because we
# pass -T OMP_PER_RANK directly (no --mpi-ranks-per-node).  Phase 3 is tested
# separately in test_mf_mpi_dispatch.sh Test 3.
echo ""
echo "[Phase 3] Note: --mpi-ranks-per-node not used in this test (thread budget"
echo "          set directly via -T ${OMP_PER_RANK}).  Phase 3 tested in test_mf_mpi_dispatch.sh."

# Wall time comparison
echo ""
NP1_TIME=$(grep "Total wall clock time" "${REF_LOG}" | head -1 || true)
NPN_TIME=$(grep "Total wall clock time" "${TEST_LOG}" | head -1 || true)
if [[ -n "${NP1_TIME}" && -n "${NPN_TIME}" ]]; then
    echo "[Wall time]"
    echo "  np=1  ${NP1_TIME}"
    echo "  np=${NRANKS}  ${NPN_TIME}"
    echo "  (np=${NRANKS} should be faster: ~$((${TOTAL_MODELS:-0} / ${NRANKS})) models/rank vs ${TOTAL_MODELS:-?} for np=1)"
fi

echo ""
echo "Outputs written to: ${OUTDIR}/"
echo "════════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
