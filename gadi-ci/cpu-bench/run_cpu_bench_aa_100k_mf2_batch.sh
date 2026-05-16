#!/bin/bash
# run_cpu_bench_aa_100k_mf2_batch.sh — batch submitter for AA 100K MF2 scaling series
#
# Submits all three AA 100K MF2 jobs (1-node, 2-node, 4-node) simultaneously.
# Jobs are independent (no afterok dependencies) — they will be scheduled and run
# as soon as normalsr resources become available.
#
# Usage (from repo root, on a Gadi login node):
#   bash gadi-ci/run_cpu_bench_aa_100k_mf2_batch.sh
#
# Override project or binary location via environment:
#   MF2_DIR=/scratch/um09/as1708/iqtree3-mf2 \
#   bash gadi-ci/run_cpu_bench_aa_100k_mf2_batch.sh
#
# Expected results (post-Fix A+B+C, SPR+AVX-512 binary, tree on rank 0 only):
#   Baseline: SPR binary 168425673 = 1,169.556 s (std MF + SPR tree, 1 node 103T)
#   Tree wall (rank 0, T=103, SPR+AVX-512): ~717 s (fixed regardless of np)
#   MF wall improves with np via Fix A (LPT stripe) + Fix B (OMP-across-models)
#              + Fix C (per-rank filterRates ref + rate_block recompute)
#
#   Run           | Nodes | MF wall  | Tree wall | Total wall | Speedup vs baseline
#   ______________|_______|__________|___________|____________|____________________
#   aa_100k_mf2   |     1 | ~399 s   | ~717 s    | ~1,116 s   | ~1.05× (SPR+AVX-512 tree)
#   aa_100k_mf2   |     2 | ~145 s   | ~717 s    | ~862 s     | ~1.36×
#   aa_100k_mf2   |     4 | ~100 s   | ~717 s    | ~817 s     | ~1.43×
#
# Amdahl ceiling: tree search (~64% of np1 total) runs on rank 0 only.
# Max achievable speedup ≈ 1/(0.64 + 0.36/N):
#   N=1: 1.00×  N=2: 1.20×  N=4: 1.33×  N=8: 1.40×
# Fix C narrows MF fraction further → actual speedup tracks Amdahl more closely.
# Fix D (proc_bind=spread): neutral for T=103 on 104-core SPR — no MF change expected.
#
# Group: aa_100k_mf2_scaling (build_tag: mf2_full_icx_avx512_r2_lpt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "═══════════════════════════════════════════════════════════"
echo "  AA 100K MF2 Scaling Batch Submission"
echo "  Repo:    ${REPO_DIR}"
echo "  Queue:   normalsr (SPR exclusive nodes)"
echo "  Project: ${PROJECT:-dx61}"
echo "  Group:   aa_100k_mf2_scaling"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Verify scripts exist
for script in \
    "${SCRIPT_DIR}/run_cpu_bench_aa_100k_mf2_1node.sh" \
    "${SCRIPT_DIR}/run_cpu_bench_aa_100k_mf2_2node.sh" \
    "${SCRIPT_DIR}/run_cpu_bench_aa_100k_mf2_4node.sh"; do
    [[ -f "${script}" ]] || { echo "ERROR: script not found: ${script}" >&2; exit 1; }
done

# Verify MF2 binary exists
# Binary lives on um09 scratch regardless of project used for job charging
MF2_DIR="${MF2_DIR:-/scratch/um09/${USER:-$(whoami)}/iqtree3-mf2}"
IQTREE_MPI="${IQTREE:-${MF2_DIR}/build-mpi-mf2/iqtree3-mpi}"
if [[ ! -x "${IQTREE_MPI}" ]]; then
    echo "WARNING: MF2 binary not found at: ${IQTREE_MPI}" >&2
    echo "         Jobs will still be submitted but will exit with rc=2 at runtime." >&2
    echo "         Build the MF2 binary first: see ${MF2_DIR}/README.md" >&2
    echo ""
fi

# Verify alignment exists
ALIGNMENT="/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
if [[ ! -f "${ALIGNMENT}" ]]; then
    echo "WARNING: alignment not found at: ${ALIGNMENT}" >&2
    echo "         Jobs will still be submitted but will exit with rc=3 at runtime." >&2
    echo ""
fi

echo "Submitting jobs..."
echo ""

JOB_1N=$(qsub "${SCRIPT_DIR}/run_cpu_bench_aa_100k_mf2_1node.sh")
echo "  1-node submitted: ${JOB_1N}"

JOB_2N=$(qsub "${SCRIPT_DIR}/run_cpu_bench_aa_100k_mf2_2node.sh")
echo "  2-node submitted: ${JOB_2N}"

JOB_4N=$(qsub "${SCRIPT_DIR}/run_cpu_bench_aa_100k_mf2_4node.sh")
echo "  4-node submitted: ${JOB_4N}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Submitted PBS IDs:"
echo "    1-node (iq-aa-100k-mf2-1n): ${JOB_1N}"
echo "    2-node (iq-aa-100k-mf2-2n): ${JOB_2N}"
echo "    4-node (iq-aa-100k-mf2-4n): ${JOB_4N}"
echo ""
echo "  Monitor:  qstat -u \$USER"
echo "  Output:   ${SCRIPT_DIR}/../iq-aa-100k-mf2-*.o<JOBID>"
echo "  Data:     ${REPO_DIR}/logs/runs/gadi_AA_100k_mf2_np{1,2,4}_seed1_<JOBID>.json"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "After all three complete, run:"
echo "  cd ${REPO_DIR}"
echo "  python3.11 tools/normalize.py && python3.11 tools/build.py"
echo "  git add -A && git commit -m 'dashboard: add AA 100K MF2 1/2/4-node results'"
