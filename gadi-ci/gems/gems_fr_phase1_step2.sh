#!/bin/bash
#PBS -N frp1s2
#PBS -P dx61
#PBS -q normalsr
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l jobfs=200GB
#PBS -l walltime=03:00:00
#PBS -l storage=scratch/rc29+scratch/dx61+gdata/um09
#PBS -l wd
#PBS -j oe
#
# PHASE-1 STEP 2 — what does re-profiling weights at every rate trial actually COST?
#
# Plan section 7.1 step 3 requires a weight re-profile at every rate trial. This gate measures the
# resulting cost and gain on the CHEAP cells before any of the outer machinery (section 7.6 acceptance,
# chi_r, support events, start portfolio) is built on top of it.
#
# THE DECISION THIS GATE MAKES:
#   * the pass REALISES an endpoint (realised=1), profile_capped == 0, and no stall sits above the
#     forcing gap  => the increment continues to step 3.
#   * profile_capped > 0, or a stall ABOVE the forcing gap, or the rate pass dominated by the convex
#     solver  => the increment PAUSES and the convex solver's degenerate tail is addressed first.
#     Building four more layers on a solver that cannot finish its inner problem would be building on sand.
#
# WHAT THE PREVIOUS RUNS ESTABLISHED, AND WHAT WAS WITHDRAWN (keep this current -- a stale header here
# is read as a result):
#   * 174461894 REACHED A POINT worth +4.192102582 nats over the weight-committed incumbent on DNA-100K
#     R8, likelihood evaluated directly and backed by a certified weight solve. That is an EXISTENCE
#     result: at least 4.19 nats of recoverable slack sit in the rate block beyond the weight block.
#   * THE VALUE REPRODUCES; THE TRIAL COUNT DOES NOT. Across a 1e6x range of inner forcing gap and two
#     different binaries, three independent runs converged to the same endpoint:
#         174461894 dna_gap8 (1e-8): 4.192102582   over 176 weight solves
#         174464179 dna_gap5 (1e-5): 4.192094513   over 169
#         174464179 dna_gap2 (1e-2): 4.192115585   over 166
#     Spread 2.107e-05 nats, 5.03e-06 relative. So the converged value is stable to five significant
#     figures while the path length varies by ~6%.
#   * A CAUTION AGAINST A HASTY RETRACTION. 174464179's dna_gap8 aborted, and that was briefly read as
#     "the result is chaotically sensitive to the rejection penalty". It was not: the pass hit the
#     200-trial BUDGET, which binds at the tightest gap, and the reproducing rungs were in the same
#     table two rows below. The budget is now 800. Read the whole table before concluding from one row.
#   * NOTE the ladder's own COST-ONLY warning is about GAIN COMPARABILITY IN PRINCIPLE; the three rungs
#     agreeing to 5 s.f. is empirical evidence the inner gap does not move THIS endpoint, not a licence
#     to compare gains across rungs in general.
#   * WITHDRAWN: "moment_out = 1.000000000000 proves the section 4.1 contract holds". That field was
#     computed AFTER the state restore, where RateFree renormalises sum_j w_j r_j to 1, so it printed
#     exactly 1 for any input -- including on four ABORTED cells, which is what exposed it. It is now
#     captured from the live fitted state and ASSERTED (G6), not printed.
#   * WITHDRAWN: "these NUMERICAL_STALL solves ARE the degenerate tail", and its over-correction. The
#     stall test must use the TIGHTEST VALID BOUND (Newton when global, else Frank-Wolfe when valid),
#     never the raw signed residual, which overstates by >=1e7x -- the two gave opposite verdicts on the
#     same solves (4.5e-04 vs 3.9e-13 on dna_gap8). Solves with no valid bound are counted, not skipped.
#     The avian tail is a separate, still-unmeasured thing.
#
# THE FORCING LADDER IS COST-ONLY -- see G6. It cannot compare gains.
#
# Nothing here certifies anything: the probe emits LEGACY_UNCERTIFIED, commits nothing, and restores its
# state element-wise. The gate asserts that restoration, because the probe runs inside a live fit.

module load gcc/12.2.0 cuda/12.5.1 2>/dev/null || true

SRC=/scratch/rc29/as1708/iqtree3-freerate-profile/build-baseline-head/iqtree3
DNA=/scratch/rc29/as1708/datasets/complex_data_shared/DNA/GTR+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy
AA=/scratch/rc29/as1708/datasets/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy

WB=/scratch/rc29/as1708/gems-verify/fr_p1s2_${PBS_JOBID%%.*}
mkdir -p "$WB"; cd "$WB"
export TMPDIR="$PBS_JOBFS/tmp"; mkdir -p "$TMPDIR"

FAIL=0
note(){ echo "$*" | tee -a "$WB/VERDICT.txt"; }
fail(){ FAIL=1; note "🔴 FAIL: $*"; }

BIN="$WB/iqtree3.frozen"
cp "$SRC" "$BIN" 2>/dev/null && chmod +x "$BIN"
[ -x "$BIN" ] || { note "🔴 FAIL: could not snapshot the binary"; exit 1; }

note "============ PHASE-1 STEP 2: cost of the re-profiled rate objective ============"
note "frozen binary = $BIN"
note "sha256(frozen)= $(sha256sum "$BIN" | cut -c1-16)"
note "sha256(source)= $(sha256sum "$SRC" | cut -c1-16)"
note "date          = $(date -Is)"
note ""

# ---------------------------------------------------------------- G1 proof of build
note "---- G1 proof-of-build (stock IQ-TREE cannot produce these) ----"
NP=$(strings "$BIN" | grep -c 'PHASE1PROBE')
NC=$(strings "$BIN" | grep -c 'PHASE1COST')
note "  PHASE1PROBE format strings : $NP"
note "  PHASE1COST  format strings : $NC"
[ "$NP" -ge 1 ] || fail "G1 phase-1 probe absent from the binary"
[ "$NC" -ge 1 ] || fail "G1 phase-1 cost instrument absent from the binary"

note ""
note "---- G0 preflight ----"
MISS=0
for f in "$DNA" "$AA"; do [ -s "$f" ] || { note "  MISSING: $f"; MISS=1; }; done
UNRES=$(ldd "$BIN" 2>/dev/null | grep -c "not found")
note "  unresolved shared libs = $UNRES"
[ "$UNRES" -eq 0 ] || fail "G0 unresolved shared libraries"
[ "$MISS" -eq 0 ] || fail "G0 an input is missing"
[ "$FAIL" -eq 0 ] || { note "======= ABORTED: preflight failed ======="; exit 1; }
note "✅ PASS: G0"

# $1 = cell, $2 = forcing gap, $3 = trial budget, rest = iqtree args
run(){
    local CELL="$1"; local GAP="$2"; local TRIALS="$3"; shift 3
    local T0; T0=$(date +%s)
    IQ_FR_ATTRIB=1 IQ_FR_SOLVE=1 IQ_FR_SOLVE_GAP="$GAP" IQ_FR_SOLVE_TRIALS="$TRIALS" \
        "$BIN" "$@" -pre "$CELL" --no-jolt -nt 104 \
        > "$WB/$CELL.stdout" 2> "$WB/$CELL.console"
    local RC=$?; echo "$RC" > "$WB/$CELL.rc"
    local T1; T1=$(date +%s)
    local KL JL
    KL=$(grep -c "Kernel:.*AVX" "$WB/$CELL.log" 2>/dev/null)
    JL=$(grep -c "JOLT" "$WB/$CELL.log" 2>/dev/null)
    note "  [$CELL] gap=$GAP trials=$TRIALS rc=$RC secs=$((T1-T0)) cpu_kernel=$KL jolt=$JL"
    [ "$RC" -eq 0 ] || fail "$CELL exited $RC"
    [ "$KL" -ge 1 ] || fail "$CELL did not report a CPU AVX kernel -- the control is not a control"
    [ "$JL" -eq 0 ] || fail "$CELL emitted JOLT markers despite --no-jolt"
}

# BUDGET RAISED 200 -> 800. At 200 the ceiling was BINDING, not slack: gate 174464179's dna_gap8 used
# every one of them and aborted, and 174461894's converged at 176 -- so the previous cost table was
# partly a measurement of the budget rather than of the objective. A ceiling that binds must be reported
# (it is, via abort_reason=trial-budget), but it must not be the default operating point.
note ""
note "---- G2 the forcing ladder on DNA-100K R8 (same cell, same state, gap varied) ----"
run dna_gap8 1e-8 800 -s "$DNA" -m GTR+F+R8 -n 0 -seed 1 -starttree PARS
run dna_gap5 1e-5 800 -s "$DNA" -m GTR+F+R8 -n 0 -seed 1 -starttree PARS
run dna_gap2 1e-2 800 -s "$DNA" -m GTR+F+R8 -n 0 -seed 1 -starttree PARS
note ""
note "---- G3 a second data type at the default gap ----"
run aa_gap8 1e-8 800 -s "$AA" -m LG+R8 -n 0 -seed 1 -starttree PARS

# ---------------------------------------------------------------- G2b REPRODUCIBILITY
# THE OPEN QUESTION, and it is now the most important one this gate can answer.
#
# Gates 174461894 and 174464179 ran the same cell from the same state and diverged completely: one
# converged at 176 trials with +4.192 nats, the other was still wandering at 200. The only
# value-affecting difference was the rejection penalty, which moved by 4.196 nats out of 6.70e6 -- a
# relative 4.19e-6 in a ~1e10 poisoned gradient. If that is the mechanism, then a BIT-IDENTICAL binary
# on a bit-identical cell must reproduce EXACTLY, because nothing perturbs it; and section 8.3 requires
# exactly that ("same hardware, binary, point ... must reproduce the canonical trace digest exactly").
# A repeat that does NOT reproduce would mean the divergence has a second source -- thread scheduling in
# the extraction, say -- and that would be far worse than a sensitive constant.
note ""
note "---- G2b reproducibility: the same binary on the same cell must give the same answer ----"
run dna_rep1 1e-8 800 -s "$DNA" -m GTR+F+R8 -n 0 -seed 1 -starttree PARS
for FLD in rate_gain cycle_gain w1_gain; do
    A=$(grep "PHASE1PROBE k=" "$WB/dna_gap8.console" 2>/dev/null | head -1 | grep -o "$FLD=[-0-9.]*" | cut -d= -f2)
    B=$(grep "PHASE1PROBE k=" "$WB/dna_rep1.console" 2>/dev/null | head -1 | grep -o "$FLD=[-0-9.]*" | cut -d= -f2)
    if [ -n "$A" ] && [ "$A" = "$B" ]; then note "  ✅ $FLD reproduces EXACTLY: $A"
    else fail "G2b $FLD did NOT reproduce: dna_gap8=$A vs dna_rep1=$B"; fi
done
for FLD in weight_solves domain_rejects profile_stall; do
    A=$(grep "PHASE1COST" "$WB/dna_gap8.console" 2>/dev/null | head -1 | grep -o "$FLD=[0-9]*" | cut -d= -f2)
    B=$(grep "PHASE1COST" "$WB/dna_rep1.console" 2>/dev/null | head -1 | grep -o "$FLD=[0-9]*" | cut -d= -f2)
    if [ -n "$A" ] && [ "$A" = "$B" ]; then note "  ✅ $FLD reproduces: $A"
    else fail "G2b $FLD did NOT reproduce: dna_gap8=$A vs dna_rep1=$B"; fi
done

CELLS="dna_gap8 dna_rep1 dna_gap5 dna_gap2 aa_gap8"

# ---------------------------------------------------------------- G3c FAULT INJECTION
# The out-of-domain rejection path only fires when a trial rate vector makes the weight problem
# insoluble. On a cheap cell that never happens, so a cheap-cell run exercises NONE of that code and
# passes blind -- a passing test that never reaches the branch it exists to test. IQ_FR_SOLVE_FAULT_AT
# forces a rejection at a chosen evaluation index so the recovery behaviour is MEASURED.
#
# WHAT THIS ALREADY FOUND, by sweeping the injection index 2..13 on example.phy (GTR+F+R4, ndim=3):
# recovery is NOT general. rate_gain by fault index --
#     eval 2      -0.268   monotone=0   ends BELOW its own start
#     evals 3,4,5  0.000   monotone=1   pass stalls, gains nothing at all
#     evals 6-11  +0.0377  monotone=1   full recovery, shortfall ~1e-8
#     eval 12     -0.425   monotone=0   ends BELOW its own start
#     eval 13     +0.0014  monotone=1   partial: 4% of the available gain
# So a large finite penalty is a WORKAROUND, not a fix: it keeps the pass alive and measurable instead
# of destroying it, but where the rejection lands decides whether the answer is right.
#
# MECHANISM, VERIFIED against utils/optimization.cpp rather than inferred from the pattern:
# Optimization::derivativeFunk costs 1+ndim evaluations -- one fx at p, then ndim forward-difference
# probes, combined as dfx[dim] = (dfx[dim] - fx)/h[dim] -- and dfpmin calls it once before its loop and
# once per iteration. With the probe's own seed call as eval 1 and ndim=3, evals 2-5 are the first
# gradient, 6-11 the first line search, 12-15 the second gradient. The two CATASTROPHIC indices (2, 12)
# are exactly the fx BASES: fx enters every gradient component, so poisoning it corrupts all ndim at
# once, at magnitude ~1e6/(1e-4*|x|) ~ 1e10 since ERROR_X = 1e-4. A single probe (3,4,5,13) corrupts one
# component and merely stalls. Line-search evaluations backtrack correctly.
# The real fix is section 7.2's bounded trust region (keep trials IN the domain) plus section 7.6
# acceptance (never publish a point worse than the incumbent) -- i.e. step 3. This arm exists so that
# is a measured requirement rather than an assumption.
note ""
note "---- G3c fault injection: does the line search RECOVER from an out-of-domain trial? ----"
FAULT_CELLS=""
for FA in 6 12; do
    CELL="dna_fault$FA"
    FAULT_CELLS="$FAULT_CELLS $CELL"
    IQ_FR_SOLVE_FAULT_AT="$FA" run "$CELL" 1e-8 800 -s "$DNA" -m GTR+F+R8 -n 0 -seed 1 -starttree PARS
    C=$(grep "PHASE1COST" "$WB/$CELL.console" 2>/dev/null | head -1)
    if [ -z "$C" ]; then fail "G3c $CELL emitted no cost line"; continue; fi
    IF=$(echo "$C" | grep -o "injected_faults=[0-9]*" | cut -d= -f2)
    DR=$(echo "$C" | grep -o "domain_rejects=[0-9]*" | cut -d= -f2)
    MON=$(echo "$C" | grep -o "rate_monotone=[-01]*" | cut -d= -f2)
    GS=$(echo "$C" | grep -o "gain_suspect=[01]" | cut -d= -f2)
    SF=$(echo "$C" | grep -o "endpoint_shortfall=[0-9.enai+-]*" | cut -d= -f2)
    RG=$(grep "PHASE1PROBE k=" "$WB/$CELL.console" 2>/dev/null | head -1 \
         | grep -o "rate_gain=[-0-9.]*" | cut -d= -f2)
    # THE INJECTION MUST HAVE FIRED, or this arm proved nothing and must not read as a pass.
    # Asserted on injected_faults, NOT on domain_rejects: the previous revision required
    # domain_rejects == 1, which silently assumed a cell that never rejects on its own. The real DNA
    # cell rejects 10 points NATURALLY, so the total was 11 and the gate declared "tested NOTHING" on an
    # arm whose injection had in fact fired. A dedicated counter is the only honest attribution.
    if [ "$IF" = "1" ]; then
        note "  ✅ $CELL: the injection FIRED (injected_faults=1; natural domain_rejects also present: $DR)"
    else
        fail "G3c $CELL: injection did NOT fire (injected_faults=$IF) -- this arm tested NOTHING"
    fi
    # The un-injected control on the same cell/gap/budget, for attribution.
    CRG=$(grep "PHASE1PROBE k=" "$WB/dna_gap8.console" 2>/dev/null | head -1 \
          | grep -o "rate_gain=[-0-9.]*" | cut -d= -f2)
    note "     rate_gain=$RG   (un-injected control dna_gap8: $CRG)"
    if [ "$GS" = "1" ]; then
        note "  ⚠️  $CELL: GAIN_SUSPECT -- monotone=$MON shortfall=$SF"
        note "     -> the pass rejected a point and ended with no material gain. Correctly refused as a"
        note "        measurement. This is the mode endpoint_shortfall alone is BLIND to (a frozen search"
        note "        walks past nothing), and it is why section 7.2's trust region is step-3 work."
    elif [ "$MON" != "1" ]; then
        note "  ⚠️  $CELL: did NOT recover -- rate_monotone=$MON, endpoint_shortfall=$SF nats"
    else
        note "     recovered: rate_monotone=1, endpoint_shortfall=$SF"
    fi
done
# ---------------------------------------------------------------- G3b INERT-WHEN-OFF
# The claim "an ordinary run is byte-for-byte unchanged" has been asserted in every header comment of
# this track and never once TESTED. G4 only proves the probe restores the state it perturbed, which is
# a weaker statement: a probe could restore its parameters perfectly and still have moved the run
# through extra likelihood evaluations, RNG draws, or cache invalidations. This arm runs the SAME cell
# with IQ_FR_SOLVE and IQ_FR_ATTRIB UNSET and requires the published result to be identical.
note ""
note "---- G3b inert-when-off control (the probe must not move the run it measures) ----"
T0=$(date +%s)
env -u IQ_FR_ATTRIB -u IQ_FR_SOLVE \
    "$BIN" -s "$DNA" -m GTR+F+R8 -n 0 -seed 1 -starttree PARS -pre off_ctl --no-jolt -nt 104 \
    > "$WB/off_ctl.stdout" 2> "$WB/off_ctl.console"
OFFRC=$?; T1=$(date +%s)
note "  [off_ctl] rc=$OFFRC secs=$((T1-T0))"
[ "$OFFRC" -eq 0 ] || fail "G3b control exited $OFFRC"
if grep -q "FRSOLVE\|FRATTRIB" "$WB/off_ctl.console"; then
    fail "G3b the probe EMITTED with its switch unset -- it is not inert"
else
    note "  ✅ no probe markers with the switch unset"
fi
ON_LNL=$(grep -m1 "^Log-likelihood of the tree:" "$WB/dna_gap8.iqtree" 2>/dev/null)
OFF_LNL=$(grep -m1 "^Log-likelihood of the tree:" "$WB/off_ctl.iqtree" 2>/dev/null)
if [ -z "$ON_LNL" ] || [ -z "$OFF_LNL" ]; then
    fail "G3b could not read a published lnL from one of the arms"
elif [ "$ON_LNL" = "$OFF_LNL" ]; then
    note "  ✅ published lnL identical with the probe ON and OFF"
    note "     $ON_LNL"
else
    fail "G3b the probe CHANGED the run it measures: ON [$ON_LNL] vs OFF [$OFF_LNL]"
fi
if cmp -s "$WB/dna_gap8.treefile" "$WB/off_ctl.treefile"; then
    note "  ✅ treefile byte-identical with the probe ON and OFF"
else
    fail "G3b treefile differs between the ON and OFF arms"
fi

# ---------------------------------------------------------------- G4 integrity
note ""
note "---- G4 integrity: the probe runs inside a live fit and MUST restore exactly ----"
# The FAULT cells are included deliberately: restoration after an aborted/rejected pass is exactly the
# path most likely to leak state, and it is the one a clean run never exercises.
for c in $CELLS $FAULT_CELLS; do
    L=$(grep "PHASE1PROBE STATUS=" "$WB/$c.console" 2>/dev/null | head -1)
    if [ -z "$L" ]; then fail "G4 $c produced no PHASE1PROBE status line"; continue; fi
    PE=$(echo "$L" | grep -o "max_param_err=[0-9.e+-]*" | cut -d= -f2)
    RE=$(echo "$L" | grep -o "restore_err=[0-9.e+-]*" | cut -d= -f2)
    note "  [$c] max_param_err=$PE restore_err=$RE"
    awk -v v="$PE" 'BEGIN{exit !(v+0==0)}' || fail "G4 $c did not restore exactly (max_param_err=$PE)"
    awk -v v="$RE" 'BEGIN{exit !(v+0<1e-6 && v+0>-1e-6)}' || fail "G4 $c restore_err out of tolerance ($RE)"
    grep -q "STATE-NOT-RESTORED" "$WB/$c.console" && fail "G4 $c reported STATE-NOT-RESTORED"
done

# ---------------------------------------------------------------- G5 THE MEASUREMENT
note ""
note "---- G5 THE COST MEASUREMENT (this is the deliverable) ----"
printf "%-10s %-8s %-9s %-9s %-7s %-4s %-5s %-9s %-9s %-12s %s\n" \
    cell gap w_solves prof_iter capped real susp secs_rate per_trial cycle_gain rate_gain \
    | tee -a "$WB/VERDICT.txt"
for c in $CELLS; do
    P=$(grep "PHASE1PROBE k=" "$WB/$c.console" 2>/dev/null | head -1)
    C=$(grep "PHASE1COST"     "$WB/$c.console" 2>/dev/null | head -1)
    [ -n "$P" ] && [ -n "$C" ] || { fail "G5 $c missing probe/cost line"; continue; }
    # `g` returns empty for a field that does not exist. A silently blank column is how the previous run
    # printed no per-solve cost at all (the emitter had been renamed to solve_per_trial and this grep was
    # never updated), so every field is now asserted non-empty rather than trusted to appear.
    # Match to the next SPACE, not a numeric character class. The class version could not see `nan`, so
    # when the not-realised path started printing cycle_gain=nan (deliberately -- an unrealised endpoint
    # has no cycle gain) the gate reported the field as ABSENT from the emitter. The value was there; the
    # reader could not represent it. A parser that only accepts the healthy case cannot report the sick one.
    g(){ echo "$2" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2; }
    gq(){ local v; v=$(g "$1" "$2"); [ -n "$v" ] || fail "G5 $c: field '$1' ABSENT from the emitter"; echo "$v"; }
    printf "%-10s %-8s %-9s %-9s %-7s %-4s %-5s %-9s %-9s %-12s %s\n" \
        "$c" "$(gq forcing_gap "$C")" "$(gq weight_solves "$C")" "$(gq profile_iters "$C")" \
        "$(gq profile_capped "$C")" "$(gq realised "$C")" "$(gq gain_suspect "$C")" \
        "$(gq secs_rate "$C")" "$(gq solve_per_trial "$C")" \
        "$(gq cycle_gain "$P")" "$(gq rate_gain "$P")" | tee -a "$WB/VERDICT.txt"
done
note "  A gain is a MEASUREMENT only when real=1 AND susp=0. Otherwise the number in the column is what"
note "  the code computed, not what the rate block is worth, and must not be quoted."

note ""
note "---- G5a cost ATTRIBUTION (an undecomposed ratio is partly a thread-count artefact) ----"
printf "%-10s %-10s %-10s %-10s %s\n" cell secs_extract secs_solve secs_realise dominant \
    | tee -a "$WB/VERDICT.txt"
for c in $CELLS; do
    C=$(grep "PHASE1COST" "$WB/$c.console" 2>/dev/null | head -1)
    [ -n "$C" ] || continue
    EX=$(echo "$C" | grep -o "secs_extract=[0-9.e+-]*" | cut -d= -f2)
    SO=$(echo "$C" | grep -o "secs_solve=[0-9.e+-]*"   | cut -d= -f2)
    RE=$(echo "$C" | grep -o "secs_realise=[0-9.e+-]*" | cut -d= -f2)
    DOM=$(awk -v e="$EX" -v s="$SO" -v r="$RE" 'BEGIN{
        if (e+0>=s+0 && e+0>=r+0) print "extract"; else if (s+0>=r+0) print "convex-solve"; else print "realise"}')
    printf "%-10s %-10s %-10s %-10s %s\n" "$c" "$EX" "$SO" "$RE" "$DOM" | tee -a "$WB/VERDICT.txt"
done

note ""
note "---- G5b same-cell control: the three DNA rungs must share a base_lnl ----"
B8=$(grep "PHASE1PROBE k=" "$WB/dna_gap8.console" 2>/dev/null | grep -o "base_lnl=[0-9.e+-]*" | cut -d= -f2)
for c in dna_gap5 dna_gap2; do
    B=$(grep "PHASE1PROBE k=" "$WB/$c.console" 2>/dev/null | grep -o "base_lnl=[0-9.e+-]*" | cut -d= -f2)
    if [ "$B" = "$B8" ]; then note "  ✅ $c base_lnl identical to dna_gap8"
    else fail "G5b $c base_lnl differs ($B vs $B8) -- the rungs are NOT the same cell/state"; fi
done

# ---------------------------------------------------------------- G2c ENDPOINT AGREEMENT
# THE TEST THE LADDER ALWAYS NEEDED AND NEVER HAD. A gain difference across rungs was previously read as
# one rung "finding a better optimum" -- a claim about ENDPOINTS made without ever observing an endpoint,
# because the probe printed gains and never the point producing them. With the measurement gap now
# separated from the search gap, rungs that reach the same point must report the same gain AND the same
# ratios. Disagreement here is a real basin/trajectory difference (section 8.3 MULTIBASIN_UNRESOLVED);
# agreement means the forcing gap is a cost knob only, as section 7.6 assumes.
note ""
note "---- G2c endpoint agreement across the forcing ladder (gain comparison is NOT enough) ----"
E8=$(grep "PHASE1ENDPOINT" "$WB/dna_gap8.console" 2>/dev/null | head -1 | grep -o "ratios=.*" | cut -d= -f2)
for c in dna_gap5 dna_gap2 dna_rep1; do
    E=$(grep "PHASE1ENDPOINT" "$WB/$c.console" 2>/dev/null | head -1 | grep -o "ratios=.*" | cut -d= -f2)
    if [ -z "$E8" ] || [ -z "$E" ]; then fail "G2c $c: no PHASE1ENDPOINT line to compare"; continue; fi
    # Compare component-wise to 1e-6 relative; exact string equality is too strict across search paths.
    MAXREL=$(awk -v a="$E8" -v b="$E" 'BEGIN{
        na=split(a,A,","); nb=split(b,B,","); if(na!=nb){print "NA"; exit}
        m=0; for(i=1;i<=na;i++){d=A[i]-B[i]; if(d<0)d=-d; s=(A[i]<0?-A[i]:A[i]); r=(s>0?d/s:d); if(r>m)m=r}
        printf "%.3e", m}')
    # CLASSIFY ON LIKELIHOOD, NOT ON LABELS. An earlier revision declared MULTIBASIN_UNRESOLVED from the
    # PARAMETER distance alone, which is precisely what section 8.2 forbids ("compare the canonical fitted
    # mixture, total likelihood, and per-pattern likelihood digest -- not arbitrary labels") and is not
    # section 8.3's criterion either: that one is tau_restart = 0.1 nat on the LIKELIHOOD. Two endpoints
    # 7% apart in a rate ratio whose likelihoods differ by 0.006 nats are the SAME basin on a flat ridge,
    # not two basins -- and a gate that calls them multibasin manufactures an identifiability crisis.
    G8=$(grep "PHASE1PROBE k=" "$WB/dna_gap8.console" 2>/dev/null | head -1 | grep -o "rate_gain=[-0-9.]*" | cut -d= -f2)
    GC=$(grep "PHASE1PROBE k=" "$WB/$c.console"       2>/dev/null | head -1 | grep -o "rate_gain=[-0-9.]*" | cut -d= -f2)
    DL=$(awk -v a="$G8" -v b="$GC" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.6e", d}')
    TAU_RESTART=0.1
    if [ "$MAXREL" = "NA" ]; then fail "G2c $c: endpoint dimension differs from dna_gap8"
    elif awk -v v="$MAXREL" 'BEGIN{exit !(v+0 <= 1e-6)}'; then
        note "  ✅ $c endpoint agrees with dna_gap8 (max rel diff $MAXREL, dLnL $DL)"
    elif awk -v d="$DL" -v t="$TAU_RESTART" 'BEGIN{exit !(d+0 <= t+0)}'; then
        note "  ⚠️  $c: FLAT RIDGE -- endpoint differs by $MAXREL but dLnL is only $DL nats"
        note "     -> SAME basin by section 8.3 (tau_restart=$TAU_RESTART); the parameters are weakly"
        note "        identified along this direction. Do NOT read the rate table as a point estimate,"
        note "        and do NOT compare runs on parameters -- section 8.2 says compare lnL/pattern probs."
    else
        note "  🔴 $c: endpoint differs by $MAXREL AND dLnL $DL EXCEEDS tau_restart=$TAU_RESTART"
        note "     -> genuinely different basins; section 8.3's MULTIBASIN_UNRESOLVED applies"
    fi
done
for c in dna_gap8 dna_gap5 dna_gap2 aa_gap8; do
    B=$(grep "PHASE1ENDPOINT" "$WB/$c.console" 2>/dev/null | head -1 | grep -o "box_hi_active=[0-9]*" | cut -d= -f2)
    L=$(grep "PHASE1ENDPOINT" "$WB/$c.console" 2>/dev/null | head -1 | grep -o "box_lo_active=[0-9]*" | cut -d= -f2)
    { [ -n "$B" ] && [ "$B" -gt 0 ] 2>/dev/null; } || { [ -n "$L" ] && [ "$L" -gt 0 ] 2>/dev/null; } && \
        note "  ⚠️  $c: endpoint on the ratio box (lo=$L hi=$B) -- the search could not reorder past the"
    [ -n "$B" ] && [ "$B" -gt 0 ] 2>/dev/null && \
        note "         anchor; section 7.2 requires a literal fallback, so this gain is not a free optimum"
done

note ""
note "---- G6 the decision ----"
note "  🔴 THE LADDER IS COST-ONLY. A looser inner gap changes the OBJECTIVE, not merely its precision,"
note "  so each rung follows a different path to a different point and the gains are gains of DIFFERENT"
note "  problems. Worse, dfpmin uses a FORWARD difference with h = 1e-4*|x|, so an inner gap g injects a"
note "  gradient error of order g/h: at g=1e-2 that is ~1e3-1e5 nats per unit ratio against a 1e-3"
note "  target, i.e. the loose rungs are differentiating noise. Compare seconds and iterations across"
note "  rungs; NEVER write 'the gain did not fall, therefore the default over-solves'."
note "  If profile_capped > 0 on a CHEAP cell, the convex solver's degenerate tail must be"
note "  addressed BEFORE steps 3-6 are built on top of it."
for c in $CELLS; do
    C=$(grep "PHASE1COST" "$WB/$c.console" 2>/dev/null | head -1)
    [ -n "$C" ] || continue
    f(){ echo "$C" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2; }   # [^ ]* so `nan`/`inf` are readable

    CAP=$(f profile_capped)
    [ -n "$CAP" ] && [ "$CAP" -gt 0 ] 2>/dev/null && \
        note "  ⚠️  $c: $CAP CAPPED weight solves on a CHEAP cell -- degenerate tail is real"

    # A stall is only a TAIL if it gave up ABOVE the gap it was asked for. At or below the forcing gap it
    # is the arithmetic floor tripping a negative-gap guard on an already-converged solve, and calling
    # that a degenerate tail is how the previous run's verdict line got it backwards.
    # Stall classification uses the TIGHTEST VALID BOUND (Newton decrement when global, else Frank-Wolfe
    # when valid), never the raw signed directional residual. Both errors have been made here: filtering
    # on isfinite(bestGapBound()) silently skipped the negative-gap-guard path, while substituting the
    # signed residual over-corrected -- it is the Frank-Wolfe quantity that overstates the true shortfall
    # by >=1e7x on near-degenerate high-k problems, and on dna_gap8 the two disagreed by ~1.2e9x, i.e.
    # opposite verdicts from identical solves. Solves with NO valid bound are counted, not swallowed.
    ST=$(f profile_stall); SGAP=$(f max_gap_stall); SSIG=$(f max_signed_stall)
    SBF=$(f stall_below_forcing); SNB=$(f stall_no_valid_bound)
    NF=$(f max_noise_floor); FG=$(f forcing_gap)
    if [ -n "$ST" ] && [ "$ST" -gt 0 ] 2>/dev/null; then
        BAR=$(awk -v a="$NF" -v b="$FG" 'BEGIN{print (a+0>b+0)?a:b}')
        if awk -v g="$SGAP" -v t="$BAR" 'BEGIN{exit !(g+0 > t+0)}'; then
            note "  🔴 $c: $ST stalls, worst BOUND $SGAP ABOVE max(noise floor $NF, forcing $FG)"
            note "       -> a REAL degenerate tail: address the convex solver BEFORE steps 3-6"
        else
            note "  ✅ $c: $ST stalls, worst BOUND $SGAP <= max(noise floor $NF, forcing $FG); $SBF at/below"
            note "       -> CONVERGED solves mislabelled by the negative-gap guard, NOT a tail"
        fi
        note "       (raw signed residual $SSIG is reported for sign only and is NOT a bound)"
        [ -n "$SNB" ] && [ "$SNB" -gt 0 ] 2>/dev/null && \
            note "  ⚠️  $c: $SNB stalls had NO valid bound at all -- unresolved, not certified"
    fi

    UNC=$(f uncertified)
    if [ -n "$UNC" ] && [ "$UNC" -gt 0 ] 2>/dev/null; then
        note "  ⚠️  $c: $UNC weight solves REFUSED as uncertified (gap not proven <= forcing gap)"
        note "     -> these were previously written back and returned as phi(r) with the gap unconsulted;"
        note "        a truncated iterate is start-dependent, so accepting one admits an unbounded value."
    fi

    DR=$(f domain_rejects); FRE=$(f first_reject_eval)
    FRR=$(echo "$C" | grep -o "first_reject_reason=[A-Za-z_-]*" | cut -d= -f2)
    if [ -n "$DR" ] && [ "$DR" -gt 0 ] 2>/dev/null; then
        note "  ⚠️  $c: $DR trial points OUT OF DOMAIN (first at eval $FRE, reason $FRR)"
        note "       -> phi is undefined there; the line search must back off. If eval $FRE lies inside"
        note "          the first line search, that is the missing section 7.2 trust region, not a bug."
    fi

    SR=$(f max_start_resid)
    if awk -v v="$SR" 'BEGIN{exit !(v+0 > 1e-10)}'; then
        note "  🔴 $c: max_start_resid=$SR -- the writeRates gauge pin DRIFTED; the start is infeasible"
    fi

    AB=$(f aborted); RL=$(f realised)
    if [ "$AB" = "1" ]; then
        AR=$(echo "$C" | grep -o "abort_reason=[a-z-]*" | cut -d= -f2)
        note "  🔴 $c: the pass ABORTED ($AR) -- its gain is NOT a measurement"
    elif [ "$RL" != "1" ]; then
        note "  🔴 $c: endpoint NOT REALISED -- rate_gain is NOT a measurement"
    fi

    # moment_out is now captured from the LIVE FITTED state, before restoreAll(). It used to be read
    # after the restore, where RateFree has already renormalised sum_j prop_j*rate_j to 1, so it printed
    # 1.000000000000 for any input and certified nothing -- including on four ABORTED passes in gate
    # 174459875, which is what exposed it. Now that it can differ from 1 (measured: 0.999999999448 on a
    # deliberately budget-aborted pass), it is worth asserting -- but ONLY on a realised endpoint, since
    # an aborted pass legitimately leaves a mismatched (r, w) pair.
    MO=$(f moment_out); RL=$(f realised)
    if [ "$RL" = "1" ]; then
        if awk -v v="$MO" 'BEGIN{exit !(v+0 > 1-1e-9 && v+0 < 1+1e-9)}'; then
            note "  ✅ [$c] fitted mean rate on the section 4.1 contract: $MO"
        else
            fail "G6 $c: fitted mean rate $MO is OFF the section 4.1 contract -- the rate block has"
            note "       drifted into the global-scale direction section 7.2 assigns to the branch block"
        fi
    else
        note "  [$c] fitted mean rate $MO (endpoint not realised -- a mismatched pair is expected here)"
    fi
done

note ""
if [ "$FAIL" -eq 0 ]; then
    note "============ RUN COMPLETE — all structural checks passed ============"
else
    note "============ RUN COMPLETE — STRUCTURAL FAILURES ABOVE ============"
fi
note "No pass/fail is asserted on the SCIENCE: the cost table is the deliverable and it decides"
note "whether the increment continues to step 3 or pauses for the convex solver."
note "workdir = $WB"
exit $FAIL
