#!/bin/bash
# submit_mf_iso.sh — submit MF-isolation runs in the recommended order.
#
# Sequential gating (don't skip 2-node!):
#   stage 1: build the binary if missing
#   stage 2: baseline reproduction (standard binary, ~20 min) ← establishes
#            same-day reference for 168425673 (MF ~405 s, total ~1,170 s)
#   stage 3: 1-node MF-iso correctness check (~22 min)
#   stage 4: 2-node MF-iso Phase 0.5/0.6 validation (~45 min) ← the real test
#   stage 5: 4-node (ONLY after stage 4 passes — submit manually after review)
#
# Usage:
#   ./submit_mf_iso.sh build      # qsub build only
#   ./submit_mf_iso.sh baseline   # qsub baseline reproduction (standard binary)
#   ./submit_mf_iso.sh 1node      # qsub MF-iso 1-node run
#   ./submit_mf_iso.sh 2node      # qsub MF-iso 2-node run
#   ./submit_mf_iso.sh all        # qsub all four chained with afterok deps
#
# Each qsub returns a job id printed to stdout.

set -euo pipefail

cd "$(dirname "$0")"

PROJECT="${PROJECT:-rc29}"
USER_ID="${USER:-$(whoami)}"
ISO_DIR="${ISO_DIR:-/scratch/${PROJECT}/${USER_ID}/iqtree3-mf-iso}"
BIN="${ISO_DIR}/build-mpi-iso/iqtree3-mpi"
STD_BIN="/scratch/dx61/sa0557/iqtree2/cpu_opt_merge/builds/build-intel-vanila/iqtree3"

qsub_stage() {
    local script="$1"; shift
    local depend="${1:-}"
    local args=()
    [[ -n "${depend}" ]] && args+=(-W "depend=afterok:${depend}")
    qsub "${args[@]}" "${script}"
}

case "${1:-}" in
    build)
        [[ -x "${BIN}" ]] && echo "WARNING: ${BIN} already exists; rebuilding will overwrite." >&2
        jid="$(qsub_stage build_mf_iso.sh)"
        echo "  build job:    ${jid}"
        ;;
    baseline)
        [[ -x "${STD_BIN}" ]] || { echo "ERROR: standard binary not found: ${STD_BIN}" >&2; exit 2; }
        jid="$(qsub_stage run_baseline_aa_100k_spr.sh)"
        echo "  baseline job: ${jid}"
        ;;
    1node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_100k_1node.sh)"
        echo "  1-node job:   ${jid}"
        ;;
    2node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_100k_2node.sh)"
        echo "  2-node job:   ${jid}"
        ;;
    all)
        echo "[submit-mf-iso] qsub build → baseline → 1node → 2node (afterok chain)"
        if [[ -x "${BIN}" ]]; then
            echo "  (binary already exists at ${BIN}; skipping build stage)"
            b_jid=""
        else
            b_jid="$(qsub_stage build_mf_iso.sh)"
            echo "  build job:    ${b_jid}"
        fi
        # Baseline can run in parallel with build (uses a different binary)
        # but we chain it after build for SU accounting cleanliness.
        base_jid="$(qsub_stage run_baseline_aa_100k_spr.sh "${b_jid}")"
        echo "  baseline job: ${base_jid}  (depends on ${b_jid:-none})"
        one_jid="$(qsub_stage run_mf_iso_aa_100k_1node.sh "${base_jid}")"
        echo "  1-node job:   ${one_jid}  (depends on ${base_jid})"
        two_jid="$(qsub_stage run_mf_iso_aa_100k_2node.sh "${one_jid}")"
        echo "  2-node job:   ${two_jid}  (depends on ${one_jid})"
        echo ""
        echo "  Monitor with: qstat -fx ${b_jid:-${base_jid}} ${base_jid} ${one_jid} ${two_jid}"
        echo ""
        echo "  After 2-node completes:"
        echo "    ./tools/parse_mf_time.py /scratch/rc29/${USER_ID}/mf_iso/profiles/AA_100k_mfiso_np2_seed1_<jobid>"
        echo "  Acceptance: MF wall < 600 s AND broadcast-arrival spread < 60 s."
        echo "  ONLY THEN submit run_cpu_bench_aa_100k_mf2_4node.sh with IQTREE=${BIN}."
        ;;
    *)
        echo "usage: $0 {build|baseline|1node|2node|all}" >&2
        exit 2
        ;;
esac
