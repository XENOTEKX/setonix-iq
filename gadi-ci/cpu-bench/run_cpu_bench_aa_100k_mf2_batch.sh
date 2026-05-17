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
# Expected results (Phase 0.5 FCA ok_rates broadcast, tree search MPI-distributed):
#   Baseline: SPR binary 168425673 = 1,169.556 s (std MF + SPR tree, 1 node 103T)
#   Tree wall scales near-linearly with np (observed 168446151-153):
#     np=1: ~717 s  np=2: ~383 s  np=4: ~198 s  (3.63× speedup for 4× ranks)
#   MF wall with Phase 0.5 FCA broadcast (ok_rates from rank 0 → all ranks):
#
#   Run           | Nodes | MF wall (proj) | Tree wall (obs) | Total (proj) | Speedup
#   ______________|_______|________________|_________________|______________|________
#   aa_100k_mf2   |     1 | ~1,277 s       | ~717 s          | ~1,994 s     | ~0.59× (MPI overhead at np=1)
#   aa_100k_mf2   |     2 | ~180-240 s     | ~383 s          | ~565-625 s   | ~1.9-2.1×
#   aa_100k_mf2   |     4 | ~95-150 s      | ~198 s          | ~295-350 s   | ~3.3-4.0×
#
# Tree search is MPI-distributed across ALL ranks (not rank 0 only).
# No Amdahl ceiling from tree search — both MF and tree phases scale with np.
# Observed pre-Phase-0.5 baseline (168446151-153): MF unscaled due to filterRates bug.
#   np=1: 1,309+717=2,026s  np=2: 969+383=1,355s  np=4: 573+198=776s (1.51×)

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
