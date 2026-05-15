#!/bin/bash
# submit_cpu_bench_all.sh — Submit all 8 CPU benchmark jobs.
#
# Usage:
#   cd ~/setonix-iq
#   bash gadi-ci/submit_cpu_bench_all.sh
#
# Prerequisites (complete before submitting normal-queue jobs):
#   1. CLX binary built:
#        qsub gadi-ci/build/build_cpu_bench_clx.sh
#        # binary → /scratch/dx61/as1708/cpu_bench/build-intel-clx/iqtree3
#
#   2. AA 1M directory permissions fixed by sa0557:
#        chmod o+x /scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_1000000/tree_1
#
# SPR jobs (cases 2, 4, 6, 8) can be submitted immediately.
# Normal-queue jobs (cases 1, 3, 5, 7) require the CLX build first.

set -euo pipefail

REPO_DIR="${HOME}/setonix-iq"
CI="${REPO_DIR}/gadi-ci/cpu-bench"
BUILD_CI="${REPO_DIR}/gadi-ci/build"

echo "=== CPU Benchmark submission (project dx61) ==="
echo ""

# ── SPR jobs (immediate) ──────────────────────────────────────────────────────
echo "--- Submitting SPR (normalsr) jobs ---"

JOB_AA_100K_SPR=$(qsub "${CI}/run_cpu_bench_aa_100k_spr.sh")
echo "  AA  100K  SPR  → ${JOB_AA_100K_SPR}"

JOB_DNA_100K_SPR=$(qsub "${CI}/run_cpu_bench_dna_100k_spr.sh")
echo "  DNA 100K  SPR  → ${JOB_DNA_100K_SPR}"

JOB_DNA_1M_SPR=$(qsub "${CI}/run_cpu_bench_dna_1m_spr.sh")
echo "  DNA 1M    SPR  → ${JOB_DNA_1M_SPR}"

JOB_AA_1M_SPR=$(qsub "${CI}/run_cpu_bench_aa_1m_spr.sh")
echo "  AA  1M    SPR  → ${JOB_AA_1M_SPR}  ⚠ blocked on AA 1M permission fix"

echo ""

# ── Normal-queue jobs (require CLX build first) ───────────────────────────────
CLX_BINARY="/scratch/dx61/as1708/cpu_bench/build-intel-clx/iqtree3"

if [[ ! -x "${CLX_BINARY}" ]]; then
    echo "--- CLX binary not found: ${CLX_BINARY} ---"
    echo "  Submitting build job first, then normal-queue bench jobs as dependencies."
    BUILD_JOB=$(qsub "${BUILD_CI}/build_cpu_bench_clx.sh")
    echo "  CLX build job → ${BUILD_JOB}"
    DEPEND="-W depend=afterok:${BUILD_JOB}"
else
    echo "--- CLX binary found — submitting normal-queue jobs directly ---"
    DEPEND=""
fi

JOB_AA_100K_CLX=$(qsub ${DEPEND} "${CI}/run_cpu_bench_aa_100k_normal.sh")
echo "  AA  100K  CLX  → ${JOB_AA_100K_CLX}"

JOB_DNA_100K_CLX=$(qsub ${DEPEND} "${CI}/run_cpu_bench_dna_100k_normal.sh")
echo "  DNA 100K  CLX  → ${JOB_DNA_100K_CLX}"

JOB_DNA_1M_CLX=$(qsub ${DEPEND} "${CI}/run_cpu_bench_dna_1m_normal.sh")
echo "  DNA 1M    CLX  → ${JOB_DNA_1M_CLX}"

JOB_AA_1M_CLX=$(qsub ${DEPEND} "${CI}/run_cpu_bench_aa_1m_normal.sh")
echo "  AA  1M    CLX  → ${JOB_AA_1M_CLX}  ⚠ blocked on AA 1M permission fix"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: qstat -u $(whoami)"
echo "Results: ${REPO_DIR}/logs/runs/gadi_*_cpu_bench*.json"
echo "Perf:    /scratch/dx61/as1708/cpu_bench/profiles/"
