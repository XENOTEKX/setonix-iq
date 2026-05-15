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
# Expected results (based on AA 100K SPR baseline 168425673, 1,169.556 s total):
#
#   Run           | Nodes | MF wall  | Tree wall | Total wall | Speedup vs baseline
#   ______________|_______|__________|___________|____________|____________________
#   aa_100k_mf2   |     1 | ~399 s   | ~764 s    | ~1,170 s   | ~1.00× (overhead only)
#   aa_100k_mf2   |     2 | ~200 s   | ~764 s    | ~965 s     | ~1.21×
#   aa_100k_mf2   |     4 | ~100 s   | ~764 s    | ~866 s     | ~1.35×
#
# Amdahl ceiling: tree search (65% of total) runs on rank 0 only.
# Max achievable speedup ≈ 1/(0.65 + 0.35/N):
#   N=1: 1.00×  N=2: 1.21×  N=4: 1.35×  N=8: 1.42×
#
# Group: aa_100k_mf2_scaling (build_tag: mf2_full_icx_avx512_r2_lpt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
