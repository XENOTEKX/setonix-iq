/*
 * Phase-1 CPU pure-+R fixed-tree solver — MODELFINDER-FULL-GPU-PLAN.md section 12, Phase 1.
 *
 * THIS FILE IS STEP 2 OF THE INCREMENT AND IS DELIBERATELY NOT A CERTIFYING SOLVER.
 *
 * Section 7.1 step 3 requires the weights to be re-profiled at EVERY rate trial. That single sentence is
 * the whole cost question for Phase 1: it turns each rate-block function evaluation from one tree
 * likelihood into a full convex weight solve plus its component extraction. Before building the outer
 * machinery on top of it (section 7.6 acceptance, chi_r, support events, the start portfolio), the honest
 * move is to build the re-profiled objective alone, run ONE major cycle, and MEASURE what it costs.
 *
 * So this step emits a cost and gain measurement, and a status of LEGACY_UNCERTIFIED. It certifies
 * nothing. If a single re-profiled rate pass on a cheap cell is already expensive, the increment stops
 * here and the convex solver's degenerate tail is addressed first, rather than discovering the problem
 * after four more layers are built on top.
 *
 * SCOPE (unchanged from the planning correction):
 *  * LITERAL mass-and-mean pocket (section 5.4, k-2 weight DOF), NOT the quotient pocket. Schema v1's
 *    certifiedForSelection admits only LITERAL_MASS_MEAN with BRLEN_FIX, so a quotient solver could not
 *    pass its own gate; and section 5.2 agrees on the mathematics, because at genuinely fixed branches
 *    the transform b' = m*b is illegal, so the mean constraint is a real constraint and not a gauge.
 *  * Branches and Q are FIXED and never written.
 *  * p_invar > 0 is refused (section 9 is Phase 3).
 */

#ifndef IQTREE_MODEL_FREERATESOLVER_H
#define IQTREE_MODEL_FREERATESOLVER_H

#include <cstddef>

class PhyloTree;

namespace freerate {

/** Step-2 probe knobs. Every one of these exists to be varied by the gate, not tuned by the solver. */
struct Phase1ProbeOptions {
    /**
     * Inner forcing tolerance for the per-trial weight solve (section 7.6).
     *
     * The profiler's default is 1e-8, which on a hard cell can run to its 10,000-iteration cap. Section
     * 7.6 explicitly permits a LOOSE early gap -- `G_w <= c * max(predicted gain, tau_L)` with c < 1 --
     * tightening geometrically only as the solve converges. Running every trial at 1e-8 is therefore
     * over-solving by orders of magnitude, and this knob exists to measure exactly that.
     */
    double forcing_gap = 1e-8;

    /**
     * Gap for the MEASUREMENT solves -- the w1 commit, the seed identity check, the endpoint realise and
     * the w2 re-profile -- as distinct from the SEARCH solves inside the rate pass.
     *
     * THESE MUST NOT BE ONE KNOB, and conflating them invalidated a published comparison.
     * `lnl_after_rate` is the exact tree likelihood at weights certified only to the gap they were solved
     * at, so with a single knob a loose rung reports a value up to `forcing_gap` BELOW the true profiled
     * value AT THE IDENTICAL ENDPOINT. Measured on DNA-100K R8: the three ladder rungs spread 6.073e-03
     * nats, which is 60.7% of the loosest rung's own 1e-2 gap -- entirely inside what the instrument
     * alone can manufacture -- and the loose 1e-2 rung actually BEAT the tighter 1e-5 rung, which no
     * "tighter gap finds a better optimum" account can produce. That spread was briefly reported as a
     * better optimum; it was measurement tolerance, not the point being measured.
     *
     * Pinning this tight and independent makes gains comparable ACROSS rungs, which is the only way the
     * forcing ladder can speak to accuracy rather than cost alone.
     */
    double measure_gap = 1e-11;

    /** Hard ceiling on convex iterations per weight solve; binding is reported, never silently absorbed. */
    std::size_t profile_max_iterations = 10000;

    /** Absolute rate-gradient target, converted to dfpmin's RELATIVE gtol by dividing by |lnL|. */
    double rate_grad_target = 1e-3;

    /**
     * Ceiling on rate trials, i.e. on WEIGHT SOLVES. The one-block arm's budget of 4000 was sized for a
     * cheap fixed-weight objective; against a re-profiling objective 4000 means 4000 convex solves, so
     * the default here is deliberately small and the binding case is reported.
     */
    std::size_t rate_trial_budget = 250;

    /** Step 2 runs a single major cycle by construction. Multi-cycle is step 3. */
    int major_cycles = 1;

    /** Below this proportion a category is frozen out of the rate block (section 7.2: no identifiable location). */
    double active_weight_floor = 1e-8;
};

/**
 * Run the step-2 probe: one major cycle of {weight profile, re-profiled rate pass, weight profile} at
 * fixed branches and Q, measuring cost and gain, then restore the state exactly.
 *
 * Restores unconditionally and proves restoration element-wise. This never commits: step 2 exists to
 * measure, and a probe that perturbed the run it measured would be worthless. Inert unless IQ_FR_SOLVE
 * is set, so an ordinary run is byte-for-byte unchanged.
 */
void reportPhase1SolveProbe(PhyloTree *tree, const char *context);

} // namespace freerate

#endif
