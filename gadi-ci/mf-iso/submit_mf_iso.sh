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
# Usage (AA 100K — original):
#   ./submit_mf_iso.sh build              # qsub build only
#   ./submit_mf_iso.sh baseline           # qsub AA 100K baseline (standard binary)
#   ./submit_mf_iso.sh 1node              # qsub AA 100K MF-iso 1-node run
#   ./submit_mf_iso.sh 2node              # qsub AA 100K MF-iso 2-node run
#   ./submit_mf_iso.sh all                # qsub all four chained with afterok deps
#
# Usage (DNA 100K):
#   ./submit_mf_iso.sh dna_100k_baseline  # qsub DNA 100K baseline (standard binary)
#   ./submit_mf_iso.sh dna_100k_1node     # qsub DNA 100K MF-iso 1-node
#   ./submit_mf_iso.sh dna_100k_2node     # qsub DNA 100K MF-iso 2-node
#   ./submit_mf_iso.sh dna_100k_all       # qsub baseline → 1node → 2node chained
#
# Usage (DNA 1M):
#   ./submit_mf_iso.sh dna_1m_baseline    # qsub DNA 1M baseline (standard binary)
#   ./submit_mf_iso.sh dna_1m_1node       # qsub DNA 1M MF-iso 1-node
#   ./submit_mf_iso.sh dna_1m_2node       # qsub DNA 1M MF-iso 2-node
#   ./submit_mf_iso.sh dna_1m_all         # qsub baseline → 1node → 2node chained
#   ./submit_mf_iso.sh dna_1m_8node_full   # qsub DNA 1M FCA full run (8-node MF+SPR)
#
# Usage (xlarge_mf full run — MF+SPR, FCA mf-iso binary, 8 nodes):
#   ./submit_mf_iso.sh xlarge_8node_full  # qsub xlarge_mf FCA full run (8-node MF+SPR)
#
# Usage (AA 1M MF-only / TESTONLY — no baseline needed; ref = 168425491):
#   ./submit_mf_iso.sh aa_1m_2node        # qsub AA 1M MF-iso 2-node (MF-only)
#   ./submit_mf_iso.sh aa_1m_4node        # qsub AA 1M MF-iso 4-node (MF-only)
#   ./submit_mf_iso.sh aa_1m_all          # qsub 2node → 4node chained (afterok)
#   ./submit_mf_iso.sh aa_1m_8node_full   # qsub AA 1M FCA full run (8-node MF+SPR)
#
# Usage (AA 1M full runs — MF+SPR, FCA mf-iso binary, scaling study):
#   ./submit_mf_iso.sh aa_1m_1node_full   # qsub AA 1M FCA full run (1-node,  103T)
#   ./submit_mf_iso.sh aa_1m_2node_full   # qsub AA 1M FCA full run (2-node,  206T)
#   ./submit_mf_iso.sh aa_1m_4node_full   # qsub AA 1M FCA full run (4-node,  412T)
#   ./submit_mf_iso.sh aa_1m_16node_full      # qsub AA 1M FCA full run (16-node, 1648T)
#   ./submit_mf_iso.sh aa_1m_16node_full_thp  # qsub AA 1M THP binary re-run (16-node, 1648T)
#   ./submit_mf_iso.sh aa_1m_full_all     # qsub 1→2→4→16-node chained (afterok)
#   # Note: 8-node full run already completed as job 168586094.
#
# Usage (Full runs — MF+SPR end-to-end parity, same FCA mf-iso binary):
#   ./submit_mf_iso.sh aa_100k_full       # qsub AA 100K FCA full run (2-node)
#   ./submit_mf_iso.sh dna_100k_full      # qsub DNA 100K FCA full run (2-node)
#   ./submit_mf_iso.sh full_100k_all      # qsub both in parallel
#
# Each qsub returns a job id printed to stdout.

set -euo pipefail

cd "$(dirname "$0")"

PROJECT="${PROJECT:-dx61}"
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
        echo "    ./tools/parse_mf_time.py /scratch/dx61/${USER_ID}/mf_iso/profiles/AA_100k_mfiso_np2_seed1_<jobid>"
        echo "  Acceptance: MF wall < 600 s AND broadcast-arrival spread < 60 s."
        echo "  ONLY THEN submit run_cpu_bench_aa_100k_mf2_4node.sh with IQTREE=${BIN}."
        ;;
    # ── DNA 100K individual stages ──────────────────────────────────
    dna_100k_baseline)
        [[ -x "${STD_BIN}" ]] || { echo "ERROR: standard binary not found: ${STD_BIN}" >&2; exit 2; }
        jid="$(qsub_stage run_baseline_dna_100k_spr.sh)"
        echo "  DNA 100K baseline job: ${jid}"
        ;;
    dna_100k_1node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_100k_1node.sh)"
        echo "  DNA 100K 1-node job:   ${jid}"
        ;;
    dna_100k_2node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_100k_2node.sh)"
        echo "  DNA 100K 2-node job:   ${jid}"
        ;;

    # ── DNA 100K full chain ──────────────────────────────────────────
    dna_100k_all)
        echo "[submit-mf-iso] qsub DNA 100K: baseline → 1node → 2node (afterok chain)"
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        base_jid="$(qsub_stage run_baseline_dna_100k_spr.sh)"
        echo "  DNA 100K baseline job: ${base_jid}"
        one_jid="$(qsub_stage run_mf_iso_dna_100k_1node.sh "${base_jid}")"
        echo "  DNA 100K 1-node job:   ${one_jid}  (depends on ${base_jid})"
        two_jid="$(qsub_stage run_mf_iso_dna_100k_2node.sh "${one_jid}")"
        echo "  DNA 100K 2-node job:   ${two_jid}  (depends on ${one_jid})"
        echo ""
        echo "  Monitor with: qstat -fx ${base_jid} ${one_jid} ${two_jid}"
        echo ""
        echo "  After 2-node completes:"
        echo "    grep EXPECTED_LNL run_mf_iso_dna_100k_1node.sh  # update from baseline result"
        echo "    grep EXPECTED_LNL run_mf_iso_dna_100k_2node.sh  # update from baseline result"
        ;;

    # ── DNA 1M individual stages ─────────────────────────────────────
    dna_1m_baseline)
        [[ -x "${STD_BIN}" ]] || { echo "ERROR: standard binary not found: ${STD_BIN}" >&2; exit 2; }
        jid="$(qsub_stage run_baseline_dna_1m_spr.sh)"
        echo "  DNA 1M baseline job: ${jid}"
        ;;
    dna_1m_1node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_1m_1node.sh)"
        echo "  DNA 1M 1-node job:   ${jid}"
        ;;
    dna_1m_2node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_1m_2node.sh)"
        echo "  DNA 1M 2-node job:   ${jid}"
        ;;

    # ── DNA 1M full chain ────────────────────────────────────────────
    dna_1m_all)
        echo "[submit-mf-iso] qsub DNA 1M: baseline → 1node → 2node (afterok chain)"
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        base_jid="$(qsub_stage run_baseline_dna_1m_spr.sh)"
        echo "  DNA 1M baseline job: ${base_jid}"
        one_jid="$(qsub_stage run_mf_iso_dna_1m_1node.sh "${base_jid}")"
        echo "  DNA 1M 1-node job:   ${one_jid}  (depends on ${base_jid})"
        two_jid="$(qsub_stage run_mf_iso_dna_1m_2node.sh "${one_jid}")"
        echo "  DNA 1M 2-node job:   ${two_jid}  (depends on ${one_jid})"
        echo ""
        echo "  Monitor with: qstat -fx ${base_jid} ${one_jid} ${two_jid}"
        echo ""
        echo "  After 2-node completes:"
        echo "    CLX ref lnL: -59208019.212 (168422813)  tol=0.5"
        echo "    SPR baseline ref will be written by run_baseline_dna_1m_spr.sh."
        ;;

    # ── DNA 1M full run: MF+SPR end-to-end (FCA, 8 nodes) ───────────
    dna_1m_8node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_1m_8node_full.sh)"
        echo "  DNA 1M 8-node full run job: ${jid}"
        echo "  Ref: lnL=-59,208,019.212  F81+F+G4  (168425675)  tol=0.5"
        ;;
    # ── xlarge_mf full run: MF+SPR end-to-end (FCA, 8 nodes) ────────
    xlarge_8node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_xlarge_8node_full.sh)"
        echo "  xlarge_mf 8-node full run job: ${jid}"
        echo "  Ref: lnL=-10,956,936.089  SYM+G4  (MF2 Full np=8 ref 168195261)  tol=1.0"
        echo "  Gate: total wall < 139.483s (MF2 Full 8-node best)"
        ;;
    # ── AA 1M individual stages ──────────────────────────────────
    aa_1m_2node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_2node.sh)"
        echo "  AA 1M 2-node job:   ${jid}"
        ;;
    aa_1m_4node)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_4node.sh)"
        echo "  AA 1M 4-node job:   ${jid}"
        ;;

    # ── AA 1M full run: MF+SPR end-to-end (FCA, 8 nodes) ───────────
    aa_1m_8node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_8node_full.sh)"
        echo "  AA 1M 8-node full run job: ${jid}"
        echo "  Ref: lnL=-78,605,196.573  LG+G4  (168425491)  tol=1.0"
        ;;

    # ── AA 1M full chain (no baseline needed — ref = 168425491) ─────
    aa_1m_all)
        echo "[submit-mf-iso] qsub AA 1M: 2node → 4node (afterok chain)"
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        two_jid="$(qsub_stage run_mf_iso_aa_1m_2node.sh)"
        echo "  AA 1M 2-node job:   ${two_jid}"
        four_jid="$(qsub_stage run_mf_iso_aa_1m_4node.sh "${two_jid}")"
        echo "  AA 1M 4-node job:   ${four_jid}  (depends on ${two_jid})"
        echo ""
        echo "  Monitor with: qstat -fx ${two_jid} ${four_jid}"
        echo ""
        echo "  Baseline ref (168425491): lnL=-78605196.573  LG+G4  MF=7587.459s  tol=1.0"
        ;;

    # ── AA 1M full runs: MF+SPR scaling study (FCA mf-iso binary) ────────────
    aa_1m_1node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_1node_full.sh)"
        echo "  AA 1M 1-node full run job: ${jid}"
        echo "  Ref: lnL=-78,605,196.573  LG+G4  (168425491)  tol=1.0"
        ;;
    aa_1m_2node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_2node_full.sh)"
        echo "  AA 1M 2-node full run job: ${jid}"
        echo "  Ref: lnL=-78,605,196.573  LG+G4  (168425491)  tol=1.0"
        ;;
    aa_1m_4node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_4node_full.sh)"
        echo "  AA 1M 4-node full run job: ${jid}"
        echo "  Ref: lnL=-78,605,196.573  LG+G4  (168425491)  tol=1.0"
        ;;
    aa_1m_16node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_16node_full.sh)"
        echo "  AA 1M 16-node full run job: ${jid}"
        echo "  Ref: lnL=-78,605,196.573  LG+G4  (168425491)  tol=1.0"
        echo "  np=8 full ref (168586094): lnL=-78,605,196.506, MF 1443.892 s"
        ;;
    aa_1m_16node_full_thp)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_1m_16node_full_thp.sh)"
        echo "  AA 1M 16-node full THP run job: ${jid}"
        echo "  Ref: lnL=-78,605,196.573  LG+G4  (168425491)  tol=1.0"
        echo "  pre-THP np=16 ref (168635616): MF 1122.363 s  (beat this to confirm THP gain)"
        ;;
    aa_1m_full_all)
        echo "[submit-mf-iso] qsub AA 1M full scaling: 1node → 2node → 4node → 16node (afterok chain)"
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        one_jid="$(qsub_stage run_mf_iso_aa_1m_1node_full.sh)"
        echo "  AA 1M 1-node full run job:  ${one_jid}"
        two_jid="$(qsub_stage run_mf_iso_aa_1m_2node_full.sh "${one_jid}")"
        echo "  AA 1M 2-node full run job:  ${two_jid}  (depends on ${one_jid})"
        four_jid="$(qsub_stage run_mf_iso_aa_1m_4node_full.sh "${two_jid}")"
        echo "  AA 1M 4-node full run job:  ${four_jid}  (depends on ${two_jid})"
        sixteen_jid="$(qsub_stage run_mf_iso_aa_1m_16node_full.sh "${four_jid}")"
        echo "  AA 1M 16-node full run job: ${sixteen_jid}  (depends on ${four_jid})"
        echo ""
        echo "  Monitor with: qstat -fx ${one_jid} ${two_jid} ${four_jid} ${sixteen_jid}"
        echo ""
        echo "  Acceptance gate: lnL=-78,605,196.573 ±1.0  LG+G4  (SPR ref 168425491)"
        echo "  Note: np=8 full run already completed as job 168586094."
        ;;

    # ── Full runs: MF+SPR end-to-end parity (FCA mf-iso binary) ────────────
    aa_100k_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_100k_2node_full.sh)"
        echo "  AA 100K full run job: ${jid}"
        echo "  Ref: lnL=-7,541,976.860  LG+G4  (168425673)"
        ;;
    dna_100k_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_100k_2node_full.sh)"
        echo "  DNA 100K full run job: ${jid}"
        echo "  Ref: lnL=-5,692,984.539  F81+F+G4  (168425674)"
        ;;
    full_100k_all)
        echo "[submit-mf-iso] qsub AA 100K full + DNA 100K full (independent, run in parallel)"
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        aa_jid="$(qsub_stage run_mf_iso_aa_100k_2node_full.sh)"
        echo "  AA 100K full run job:  ${aa_jid}"
        dna_jid="$(qsub_stage run_mf_iso_dna_100k_2node_full.sh)"
        echo "  DNA 100K full run job: ${dna_jid}"
        echo ""
        echo "  Monitor with: qstat -fx ${aa_jid} ${dna_jid}"
        echo ""
        echo "  Acceptance:"
        echo "    AA  100K: lnL=-7,541,976.860 ±0.1  LG+G4    SPR ref 168425673"
        echo "    DNA 100K: lnL=-5,692,984.539 ±0.1  F81+F+G4 SPR ref 168425674"
        ;;

    # ── T3 / T4: AA 100K and DNA 100K 4-node full runs (§5.1 test matrix) ────
    aa_100k_4node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_aa_100k_4node_full.sh)"
        echo "  AA 100K 4-node full run job: ${jid}"
        echo "  Ref: lnL=-7,541,976.860 ±0.1  LG+G4  (168425673)  T3 §5.1"
        ;;
    dna_100k_4node_full)
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        jid="$(qsub_stage run_mf_iso_dna_100k_4node_full.sh)"
        echo "  DNA 100K 4-node full run job: ${jid}"
        echo "  Ref: lnL=-5,692,984.539 ±0.1  F81+F+G4  (168425674)  T4 §5.1"
        ;;
    full_100k_4node_all)
        echo "[submit-mf-iso] qsub T3+T4: AA 100K 4-node full + DNA 100K 4-node full (independent, run in parallel)"
        [[ -x "${BIN}" ]] || { echo "ERROR: ${BIN} missing — './submit_mf_iso.sh build' first." >&2; exit 2; }
        aa_jid="$(qsub_stage run_mf_iso_aa_100k_4node_full.sh)"
        echo "  T3 AA  100K 4-node full run job: ${aa_jid}"
        dna_jid="$(qsub_stage run_mf_iso_dna_100k_4node_full.sh)"
        echo "  T4 DNA 100K 4-node full run job: ${dna_jid}"
        echo ""
        echo "  Monitor with: qstat -fx ${aa_jid} ${dna_jid}"
        echo ""
        echo "  Acceptance:"
        echo "    T3 AA  100K np=4: lnL=-7,541,976.860 ±0.1  LG+G4    SPR ref 168425673"
        echo "    T4 DNA 100K np=4: lnL=-5,692,984.539 ±0.1  F81+F+G4 SPR ref 168425674"
        ;;

    *)
        echo "usage: $0 {build|baseline|1node|2node|all" >&2
        echo "           |dna_100k_baseline|dna_100k_1node|dna_100k_2node|dna_100k_all" >&2
        echo "           |dna_1m_baseline|dna_1m_1node|dna_1m_2node|dna_1m_all|dna_1m_8node_full" >&2
        echo "           |xlarge_8node_full" >&2
        echo "           |aa_1m_2node|aa_1m_4node|aa_1m_all|aa_1m_8node_full" >&2
        echo "           |aa_1m_1node_full|aa_1m_2node_full|aa_1m_4node_full|aa_1m_16node_full|aa_1m_16node_full_thp|aa_1m_full_all" >&2
        echo "           |aa_100k_full|dna_100k_full|full_100k_all" >&2
        echo "           |aa_100k_4node_full|dna_100k_4node_full|full_100k_4node_all}" >&2
        exit 2
        ;;
esac
