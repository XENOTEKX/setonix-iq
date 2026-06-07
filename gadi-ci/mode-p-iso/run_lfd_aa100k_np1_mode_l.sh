#!/bin/bash
# run_lfd_aa100k_np1_mode_l.sh — Mode L L.1 gate: full-tree traversal count (BASE vs LM).
# Built on the L.0b.viii binary; adds a per-PhyloTree [L1-TRAV] counter (postorder/preorder/derv)
# emitted per model at CandidateModel::evaluate EXIT.  Answers the L.1 question: does the joint
# LM CUT total full-tree traversals vs the legacy alpha-Brent + p_inv-EM + branch loop?
# PRODUCTION mode (no --mode-l-fd-check).
# full_analytic=true for +G/+I+G models; partial analytic (rate dims) for +Rk.
#
#   BASE arm:  legacy alternating BFGS/Brent/EM optimizer.
#   MODE-L arm: --mode-l (production; OMP-parallel preorder + analytic gradient)
#
# Gate pass criteria (L.1 — traversal-count gate; -m TEST):
#   (1) both arms exit 0; best model = LG+G4 in both.
#   (2) lnL parity: |lnL_base - lnL_modeL| <= 0.05 (post-SPR with -m TEST; SPR is
#       byte-identical across arms since Mode-L fires ONLY during ModelFinder).
#   (3) MODE-L accepted_iters > 0 — parsed from the [MODE-L-DBG] ... EXIT accepted_iters=N
#       lines (always written), NOT the verbose-only `MODE-L:` cout line (production runs
#       at verbose_mode < VB_MED, so that line is never emitted).
#   (4 → L.1, HARD): MODE-L MF-phase full-tree traversals (postorder+preorder, summed over
#       all evaluated models) STRICTLY < BASE — AND the dominant model LG+G4 individually.
#       THE L.1 QUESTION: does the joint LM CUT total full-tree traversals vs the legacy
#       alternating (alpha-Brent + p_inv-EM + branch) loop?  Parsed from the per-arm
#       [L1-TRAV] lines (emitted at CandidateModel::evaluate EXIT — MF phase ONLY; the post-MF
#       SPR/NNI search contributes 0 because the counters are guarded by mode_l_context_active).
#       WALL TIME IS NOT GATED: the Mode-L preorder kernel is scalar/single-threaded, so wall
#       confounds fewer-traversals with slower-per-traversal; wall is reported informationally.
#       The derv bucket (legacy Newton-derv + Mode-L analytic-gradient sweeps; both cached/partial)
#       is reported but kept OUT of the strict criterion so BASE is not unfairly inflated.
#       A FAIL here is a legitimate verdict ("LM does NOT cut traversals → reconsider Layer 2"),
#       not a build bug — all diagnostics still print.
#
# ModelFinder scope: -m TEST (224 models, MF + SPR) — the parity benchmark going forward
# (FCA np=1 -m TEST reference MF = 259s, job 169095077).
# Reference lnL (FCA np=1 job 169095077): -7,541,976.861.

#PBS -N lfd-mode-l-np1
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l place=excl
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe

set -euo pipefail

SANDBOX="/scratch/rc29/as1708/iqtree3-mode-p-iso"
IQTREE="${IQTREE:-${SANDBOX}/build-mode-p-iso-p3/iqtree3-mpi-mode-p-iso-p3}"
ALIGNMENT="${ALIGNMENT:-/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy}"

NRANKS=1
OMP_PER_RANK="${OMP_PER_RANK:-103}"
SEED="${SEED:-1}"
EXPECTED_MD5="${EXPECTED_MD5:-8469af7b2035cd110ae3b5be1d80474f}"  # L.0b.vi build 169647747 (weight gradient; -m TEST behaviour unchanged — no +R)

PBS_ID_SHORT="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)}"; PBS_ID_SHORT="${PBS_ID_SHORT%%.*}"
LABEL="lfd_modeL_aa100k_np1_seed${SEED}"
WORK_DIR="${SANDBOX}/runs/${LABEL}_${PBS_ID_SHORT}"
BASE_DIR="${WORK_DIR}/base"
MODL_DIR="${WORK_DIR}/mode_l"
mkdir -p "${BASE_DIR}" "${MODL_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.7                2>/dev/null || true
    module load intel-compiler-llvm/2025.3.2 2>/dev/null || true
fi

[[ -x "${IQTREE}" ]] || { echo "ERROR: binary not found: ${IQTREE}" >&2; exit 2; }
[[ -f "${ALIGNMENT}" ]] || { echo "ERROR: alignment not found: ${ALIGNMENT}" >&2; exit 3; }
MD5=$(md5sum "${IQTREE}" | awk '{print $1}')
if [[ -n "${EXPECTED_MD5}" && "${MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: binary md5 ${MD5} != expected ${EXPECTED_MD5} — wrong build?" >&2
    exit 4
fi
# (Login-node verification already confirmed the binary contains
#  ModelFactory::optimizeModeLAllParameters and the --mode-l* strings;
#  do NOT re-grep here because set -o pipefail + grep -q + strings = SIGPIPE
#  and a false-positive exit. The md5 above is the canonical check.)

export KMP_BLOCKTIME=200
export TMPDIR="${SANDBOX}/tmp"; mkdir -p "${TMPDIR}"

OMP_ENV=(
    -x "OMP_NUM_THREADS=${OMP_PER_RANK}"
    -x "OMP_MAX_ACTIVE_LEVELS=2"
    -x "OMP_DYNAMIC=false"
    -x "OMP_PROC_BIND=close"
    -x "OMP_PLACES=cores"
    -x "OMP_WAIT_POLICY=PASSIVE"
    -x "GOMP_SPINCOUNT=10000"
    -x "KMP_BLOCKTIME=${KMP_BLOCKTIME}"
    -x "KMP_HOT_TEAMS_MAX_LEVEL=2"
)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Mode L L.1 gate: full-tree traversal count BASE vs LM  AA 100K -m TEST  ║"
echo "║  binary:    $(basename "${IQTREE}")  md5:${MD5}"
echo "║  work_dir:  ${WORK_DIR}"
echo "║  Ref lnL:   -7,541,976.861 (FCA np=1 job 169095077)"
echo "╚══════════════════════════════════════════════════════════════╝"

run_arm() {
    local arm="$1"; local outdir="$2"; shift 2
    local arm_args=("$@")
    echo ""
    echo "── arm: ${arm} ──  flags: ${arm_args[*]:-<legacy>}"
    local t0=$(date +%s)
    set +e
    mpirun -np "${NRANKS}" \
        --bind-to none \
        "${OMP_ENV[@]}" \
        numactl --localalloc -- \
        "${IQTREE}" -s "${ALIGNMENT}" -m TEST -T "${OMP_PER_RANK}" -seed "${SEED}" \
                    "${arm_args[@]}" \
                    --prefix "${outdir}/iqtree_inner" \
        > "${outdir}/iqtree_stdout.log" 2>&1
    local rc=$?
    set -e
    local t1=$(date +%s)
    local wall=$(( t1 - t0 ))
    echo "${arm}: exit=${rc} wall=${wall}s"
    echo "${rc}" > "${outdir}/exit.rc"
    echo "${wall}" > "${outdir}/wall.sec"
}

# ── BASE arm: legacy optimizer ─────────────────────────────────────────
run_arm "BASE" "${BASE_DIR}"

# ── MODE-L arm: production mode (analytic gradient, no FD check overhead) ─
run_arm "MODE-L" "${MODL_DIR}" --mode-l

echo ""
echo "── inner-log tails ──"
echo "BASE  inner.log:";  tail -10 "${BASE_DIR}/iqtree_inner.log" 2>/dev/null || true
echo "MODEL inner.log:"; tail -10 "${MODL_DIR}/iqtree_inner.log" 2>/dev/null || true

# ── parse outcomes ────────────────────────────────────────────────────
parse_arm() {
    local d="$1"
    local lnl best mf
    lnl=$(grep -oP 'BEST SCORE FOUND :\s*\K[-0-9.]+' "${d}/iqtree_inner.log" 2>/dev/null | tail -1 || true)
    [[ -z "${lnl}" ]] && lnl=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "${d}/iqtree_inner.log" 2>/dev/null | tail -1 || true)
    [[ -z "${lnl}" ]] && lnl=$(grep -oP 'Optimal log-likelihood: \K[-0-9.]+' "${d}/iqtree_inner.log" 2>/dev/null | tail -1 || true)
    best=$(grep -oP 'Best-fit model.*?:\s*\K\S+' "${d}/iqtree_inner.log" 2>/dev/null | head -1 || true)
    mf=$(grep -oP 'Wall-clock time for ModelFinder: \K[0-9.]+' "${d}/iqtree_inner.log" 2>/dev/null | head -1 || true)
    printf '%s|%s|%s\n' "${lnl:-}" "${best:-}" "${mf:-}"
}
B_PARSE=$(parse_arm "${BASE_DIR}")
M_PARSE=$(parse_arm "${MODL_DIR}")
BASE_LNL=$(echo "${B_PARSE}" | awk -F'|' '{print $1}')
BASE_BEST=$(echo "${B_PARSE}" | awk -F'|' '{print $2}')
BASE_MF=$(echo "${B_PARSE}" | awk -F'|' '{print $3}')
MODL_LNL=$(echo "${M_PARSE}" | awk -F'|' '{print $1}')
MODL_BEST=$(echo "${M_PARSE}" | awk -F'|' '{print $2}')
MODL_MF=$(echo "${M_PARSE}" | awk -F'|' '{print $3}')

# Mode L diagnostics — parse from stdout (where MODE-L-FDCHECK and MODE-L: emit via cerr/cout)
MODEL_DIAG_SRC=( "${MODL_DIR}/iqtree_inner.log" "${MODL_DIR}/iqtree_stdout.log" )
# DBG log (always written, regardless of verbose_mode) — carries the per-call
# [MODE-L-DBG] optimizeModeLAllParams EXIT ... accepted_iters=N lines.  The verbose
# `MODE-L: ... accepted_iters=` cout line only prints at verbose_mode>=VB_MED, so in
# production mode criterion (3) MUST read accepted_iters from this DBG file (L.0b.viii).
MODEL_DBG_LOG="${MODL_DIR}/iqtree_inner.mode_l_debug.log"
FDCHECK_LINES=$(grep -h 'MODE-L-FDCHECK' "${MODEL_DIAG_SRC[@]}" 2>/dev/null | wc -l || true)
SUMMARY_LINES=$(grep -h '^MODE-L: '       "${MODEL_DIAG_SRC[@]}" 2>/dev/null | wc -l || true)
MAX_RECON_ABS=$( { grep -hoP 'MODE-L-FDCHECK[^|]*\|lnl-recon\|=\K[0-9.eE+-]+' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } \
    | awk 'BEGIN{m=0}{v=$1+0; if (v<0) v=-v; if (v>m) m=v} END{printf "%.6e\n", m+0}' )
TOTAL_ACCEPTED_ITERS=$( { \
        grep -hoP 'optimizeModeLAllParams EXIT.*accepted_iters=\K[0-9]+' "${MODEL_DBG_LOG}" 2>/dev/null; \
        grep -hoP '^MODE-L: ndim=\S+\s+accepted_iters=\K[0-9]+'          "${MODEL_DIAG_SRC[@]}" 2>/dev/null; \
        true; } \
    | awk '{s+=$1} END{print s+0}')
# Also report how many models took at least one accepted LM step (DBG EXIT lines).
MODELS_WITH_ACCEPT=$( { grep -hoP 'optimizeModeLAllParams EXIT.*accepted_iters=\K[0-9]+' "${MODEL_DBG_LOG}" 2>/dev/null || true; } \
    | awk '$1>0{n++} END{print n+0}')

# ── L.1 traversal-count gate parsing ──────────────────────────────────────
# [L1-TRAV] model=<name> arm=<base|model> traversals=<post+pre> postorder=<a> preorder=<b> derv=<c>
# is emitted once per model at CandidateModel::evaluate EXIT (MF phase ONLY — guarded by
# mode_l_context_active, so the post-MF SPR/NNI search contributes 0).  Each arm writes to
# its OWN <prefix>.mode_l_debug.log (BASE_DIR vs MODL_DIR), so parse each arm's own file.
BASE_DBG="${BASE_DIR}/iqtree_inner.mode_l_debug.log"
sum_field()   { { grep -hoP "\[L1-TRAV\] .*\b$2=\K[0-9]+" "$1" 2>/dev/null || true; } | awk '{s+=$1} END{print s+0}'; }
heavy_field() { local hm="$4"; { grep -hoP "\[L1-TRAV\] model=\Q${hm}\E arm=$2 .*\b$3=\K[0-9]+" "$1" 2>/dev/null || true; } | awk '{s+=$1} END{print s+0}'; }
count_l1()    { { grep -hc '\[L1-TRAV\]' "$1" 2>/dev/null || true; } | awk '{s+=$1} END{print s+0}'; }
BASE_TRAV=$(sum_field "${BASE_DBG}" traversals)
MODL_TRAV=$(sum_field "${MODEL_DBG_LOG}" traversals)
BASE_POST=$(sum_field "${BASE_DBG}" postorder); BASE_PRE=$(sum_field "${BASE_DBG}" preorder); BASE_DERV=$(sum_field "${BASE_DBG}" derv)
MODL_POST=$(sum_field "${MODEL_DBG_LOG}" postorder); MODL_PRE=$(sum_field "${MODEL_DBG_LOG}" preorder); MODL_DERV=$(sum_field "${MODEL_DBG_LOG}" derv)
# L.1 hardening (2026-05-31): anchor the dominant-model spot check on the ACTUAL
# winning model (BASE_BEST), not a hardcoded LG+G4 — so the +R / -m MF follow-up
# (winner e.g. LG+R4) is checked, not silently skipped. \Q\E quotes '+' as literal.
HEAVY_MODEL="${BASE_BEST:-LG+G4}"
BASE_HEAVY=$(heavy_field "${BASE_DBG}" base traversals "${HEAVY_MODEL}")
MODL_HEAVY=$(heavy_field "${MODEL_DBG_LOG}" model traversals "${HEAVY_MODEL}")
L1_MODELS_BASE=$(count_l1 "${BASE_DBG}")
L1_MODELS_MODL=$(count_l1 "${MODEL_DBG_LOG}")
# L.0b.ii: analytic alpha gradient cross-check (|G-ratio|)
GRATIO_LINES=$( { grep -h '|G-ratio|=' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } | wc -l)
MAX_G_RATIO=$( { grep -hoP '\|G-ratio\|=\K[0-9.eE+-]+' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } \
    | awk 'BEGIN{m=0}{v=$1+0; if (v>m) m=v} END{printf "%.6e\n", m+0}')
# L.0b.iv: analytic p_inv gradient cross-check (|G-ratio-pinv|) — diagnostic only in prod mode
GRATIO_PINV_LINES=$( { grep -h '|G-ratio-pinv|=' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } | wc -l)
MAX_G_RATIO_PINV=$( { grep -hoP '\|G-ratio-pinv\|=\K[0-9.eE+-]+' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } \
    | awk 'BEGIN{m=0}{v=$1+0; if (v>m) m=v} END{printf "%.6e\n", m+0}')
# L.0b.v: FreeRate rate dim gradient check (|G-ratio-rate0|) — diagnostic only in prod mode
GRATIO_RATE0_LINES=$( { grep -h '|G-ratio-rate0|=' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } | wc -l)
MAX_G_RATIO_RATE0=$( { grep -hoP '\|G-ratio-rate0\|=\K[0-9.eE+-]+' "${MODEL_DIAG_SRC[@]}" 2>/dev/null || true; } \
    | awk 'BEGIN{m=0}{v=$1+0; if (v>m) m=v} END{printf "%.6e\n", m+0}')

echo ""
echo "══ L.1 (full-tree traversal count) gate result ══════════════════════"
echo "  binary md5:           ${MD5}"
echo "  expected md5:         ${EXPECTED_MD5}"
echo "  BASE   exit | wall:   $(cat ${BASE_DIR}/exit.rc) | $(cat ${BASE_DIR}/wall.sec)s   MF=${BASE_MF}s"
echo "  MODE-L exit | wall:   $(cat ${MODL_DIR}/exit.rc) | $(cat ${MODL_DIR}/wall.sec)s   MF=${MODL_MF}s"
echo "  BASE   lnL | model:   ${BASE_LNL} | ${BASE_BEST}"
echo "  MODE-L lnL | model:   ${MODL_LNL} | ${MODL_BEST}"
echo "  ref lnL:              -7541976.861 (FCA np=1)"
echo "  total accepted_iters: ${TOTAL_ACCEPTED_ITERS}    (gate: >0, from [MODE-L-DBG] EXIT lines)"
echo "  models w/ >=1 accepted LM step: ${MODELS_WITH_ACCEPT:-0}"
echo "  MODE-L: summary lines:${SUMMARY_LINES}"
echo "  [diag] FDCHECK lines: ${FDCHECK_LINES}    (0 in production mode; non-zero if --mode-l-fd-check used)"
echo "  [diag] max |lnl-recon|:   ${MAX_RECON_ABS}"
echo "  [diag] |G-ratio| lines:   ${GRATIO_LINES}  max=${MAX_G_RATIO}"
echo "  [diag] |G-ratio-pinv| lines: ${GRATIO_PINV_LINES:-0}  max=${MAX_G_RATIO_PINV:-0.000000e+00}"
echo "  [diag] |G-ratio-rate0| lines:${GRATIO_RATE0_LINES:-0}  max=${MAX_G_RATIO_RATE0:-0.000000e+00}"
echo "  ──────────────────────────────────────────────────────────────────"
echo "  L.1 MF-phase full-tree traversals (headline = postorder + preorder):"
echo "     BASE  total=${BASE_TRAV}   (post=${BASE_POST} pre=${BASE_PRE} | derv=${BASE_DERV})  models=${L1_MODELS_BASE}"
echo "     MODEL total=${MODL_TRAV}   (post=${MODL_POST} pre=${MODL_PRE} | derv=${MODL_DERV})  models=${L1_MODELS_MODL}"
echo "     ${HEAVY_MODEL} (dominant model):  base=${BASE_HEAVY:-NA}   model=${MODL_HEAVY:-NA}"

PASS=1
BASE_WALL=$(cat ${BASE_DIR}/wall.sec 2>/dev/null || echo 9999)
MODL_WALL=$(cat ${MODL_DIR}/wall.sec 2>/dev/null || echo 9999)

# (1) both arms exit 0
[[ "$(cat ${BASE_DIR}/exit.rc)" -eq 0 ]] || { echo "  ✗ FAIL: BASE exit != 0"; PASS=0; }
[[ "$(cat ${MODL_DIR}/exit.rc)" -eq 0 ]] || { echo "  ✗ FAIL: MODE-L exit != 0"; PASS=0; }

# (2) lnL parity
[[ -n "${BASE_LNL}" && -n "${MODL_LNL}" ]] || { echo "  ✗ FAIL: lnL not parsed (B=${BASE_LNL}, M=${MODL_LNL})"; PASS=0; }
if [[ -n "${BASE_LNL}" && -n "${MODL_LNL}" ]]; then
    DLT=$(python3 -c "print(abs(${BASE_LNL} - (${MODL_LNL})))")
    OK=$(python3 -c "print('yes' if ${DLT} <= 0.05 else 'no')")
    [[ "${OK}" == "yes" ]] && echo "  ✓ lnL parity (|Δ|=${DLT})" || { echo "  ✗ FAIL: paired lnL drift |Δ|=${DLT}"; PASS=0; }
fi

# (1) best model = LG+G4 in both arms
[[ "${BASE_BEST}" == "LG+G4" ]]  && echo "  ✓ BASE  best LG+G4"  || { echo "  ✗ FAIL: BASE best=${BASE_BEST}"; PASS=0; }
[[ "${MODL_BEST}" == "LG+G4" ]]  && echo "  ✓ MODEL best LG+G4"  || { echo "  ✗ FAIL: MODEL best=${MODL_BEST}"; PASS=0; }

# (3) accepted_iters > 0 (LM optimizer actually ran)
if [[ "${TOTAL_ACCEPTED_ITERS}" -gt 0 ]]; then
    echo "  ✓ accepted_iters=${TOTAL_ACCEPTED_ITERS} > 0 (LM optimizer active)"
else
    echo "  ✗ FAIL: accepted_iters=0 — LM optimizer did not take any steps"
    PASS=0
fi

# (4 → L.1) MODE-L MF-phase full-tree traversals STRICTLY fewer than BASE (aggregate AND LG+G4).
# Wall time is NOT gated (scalar preorder kernel confounds it); reported informationally only.
echo "  [info] wall: MODE-L=${MODL_WALL}s  BASE=${BASE_WALL}s  (informational; not gated — see header)"
# Hardening (2026-05-31): require MODL_TRAV>0 and a non-empty MF phase so a zeroed /
# short-circuited MODE-L run cannot spoof a "cut" via 0 < N. (We do NOT require equal
# model counts across arms — filterRates legitimately prunes differently: 84 vs 80.)
if [[ -n "${BASE_TRAV}" && "${BASE_TRAV}" -gt 0 && -n "${MODL_TRAV}" && "${MODL_TRAV}" -gt 0 && "${L1_MODELS_MODL:-0}" -gt 0 ]]; then
    if [[ "${MODL_TRAV}" -lt "${BASE_TRAV}" ]]; then
        RATIO=$(python3 -c "print('%.2f' % (${BASE_TRAV}/${MODL_TRAV}))" 2>/dev/null || echo "?")
        echo "  ✓ L.1 aggregate cut: MODE-L=${MODL_TRAV} < BASE=${BASE_TRAV} full-tree traversals (${RATIO}× fewer)"
    else
        echo "  ✗ FAIL (L.1): MODE-L traversals ${MODL_TRAV} >= BASE ${BASE_TRAV} — joint LM did NOT cut full-tree traversals"
        PASS=0
    fi
    # LG+G4 dominant-model spot check (catches a model where trust-step rejections lose even if aggregate wins)
    if [[ -n "${BASE_HEAVY}" && "${BASE_HEAVY}" -gt 0 ]]; then
        if [[ "${MODL_HEAVY}" -lt "${BASE_HEAVY}" ]]; then
            echo "  ✓ L.1 ${HEAVY_MODEL} cut: MODE-L=${MODL_HEAVY} < BASE=${BASE_HEAVY}"
        else
            echo "  ✗ FAIL (L.1): ${HEAVY_MODEL} MODE-L=${MODL_HEAVY} >= BASE=${BASE_HEAVY}"
            PASS=0
        fi
    else
        # Fail-closed (was a silent ⚠ skip): a missing dominant-model line means the
        # anchor didn't match the winning model — do NOT pass on absent evidence.
        echo "  ✗ FAIL (L.1): ${HEAVY_MODEL} dominant-model spot check found no BASE [L1-TRAV] line (heavy=${BASE_HEAVY:-NA}) — check model name in [L1-TRAV] lines"
        PASS=0
    fi
else
    echo "  ✗ FAIL (L.1): traversal counts not parsed (B=${BASE_TRAV}, M=${MODL_TRAV}) — check [L1-TRAV] emission / DBG logs"
    PASS=0
fi

# Diagnostic: if FDCHECK lines present (e.g. manual run), check gradient ratios
if [[ "${GRATIO_PINV_LINES:-0}" -gt 0 ]]; then
    GRATIO_PINV_OK=$(python3 -c "print('yes' if float('${MAX_G_RATIO_PINV:-1}') < 0.01 else 'no')" 2>/dev/null || echo "no")
    [[ "${GRATIO_PINV_OK}" == "yes" ]] && echo "  [diag] ✓ L.0b.iv p_inv gradient: max |G-ratio-pinv|=${MAX_G_RATIO_PINV}" \
        || echo "  [diag] ✗ L.0b.iv p_inv gradient: max |G-ratio-pinv|=${MAX_G_RATIO_PINV} >= 0.01"
fi
if [[ "${GRATIO_RATE0_LINES:-0}" -gt 0 ]]; then
    GRATIO_RATE0_OK=$(python3 -c "print('yes' if float('${MAX_G_RATIO_RATE0:-1}') < 0.01 else 'no')" 2>/dev/null || echo "no")
    [[ "${GRATIO_RATE0_OK}" == "yes" ]] && echo "  [diag] ✓ L.0b.v rate0 gradient: max |G-ratio-rate0|=${MAX_G_RATIO_RATE0}" \
        || echo "  [diag] ✗ L.0b.v rate0 gradient: max |G-ratio-rate0|=${MAX_G_RATIO_RATE0} >= 0.01"
fi

if [[ "${PASS}" -eq 1 ]]; then
    echo "  ══ L-FD PASS ══"
else
    echo "  ══ L-FD FAIL ══"; exit 10
fi
