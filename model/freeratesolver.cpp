#include "freeratesolver.h"

#include "freerateinternal.h"
#include "freerateprofile.h"
#include "ratefree.h"
#include "tree/phylotree.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace freerate {

namespace {

/**
 * Build the literal mass-and-mean weight problem at the CURRENT rates.
 *
 * `target_moment` is the CANONICAL 1.0, not the live state's moment. That distinction is the difference
 * between measuring a function and measuring a random walk. The one-block diagnostics profile once, at
 * the published state, and must not move it -- for them `actual_moment` is right. Inside a rate pass the
 * live proportions are overwritten by the previous trial, so `actual_moment` makes the CONSTRAINT LEVEL
 * itself history-dependent: the feasible set at trial n becomes {w : r_n'w = r_n . p_{n-1}}. The
 * objective then stops being a function of its argument, and the realised mean rate drifts off the
 * section 4.1 contract sum_j w_j r_j = 1 -- re-admitting the global-scale direction that section 7.2
 * assigns to the branch block and explicitly excludes from the rate block.
 *
 * Pinning to 1.0 is legal at every trial precisely because writeRates pinned the rates so that
 * sum_j w_j r_j == 1 for the pass's weight vector, which is therefore a FEASIBLE start by construction.
 */
freerate_profile::ProfileProblem literalProblemFrom(const ComponentExtraction &e,
                                                   double target_moment) {
    freerate_profile::ProfileProblem problem;
    problem.pattern_count = e.pattern_count;
    problem.category_count = e.category_count;
    problem.component_likelihood = e.component_likelihood;
    problem.component_log_scale = e.component_log_scale;
    problem.multiplicity = e.multiplicity;
    problem.rate = e.rate;
    problem.geometry = freerate_profile::FeasibleGeometry::LITERAL_MASS_MEAN;
    problem.target_moment = target_moment;
    return problem;
}

/** A profiled point is only worth writing if it is a LEGAL +R state. */
bool profiledPointIsFeasible(const freerate_profile::ProfileResult &r) {
    // The monotonicity of a likelihood cannot substitute for this. A mass residual is a mechanical
    // N_sites * eps shift with no model content, and a NEGATIVE proportion is not caught at all because
    // the kernel takes tree/phylokernelnew.h:`lh_ptn = abs(lh_ptn) + ptn_invar[ptn]`, turning an
    // infeasible mixture into a plausible finite likelihood that can exceed the incumbent.
    return (r.primal.mass <= 1e-9) && (r.primal.literal_moment <= 1e-9) && (r.primal.negativity <= 0.0);
}

/**
 * THE STEP-2 OBJECT: a rate oracle whose objective RE-PROFILES THE WEIGHTS (section 7.1 step 3).
 *
 * The inherited oracle holds weights fixed and costs one tree likelihood per evaluation. This one solves
 * the convex weight block at every trial rate vector, which is what the plan's block cycle actually
 * specifies and what makes the rate step a step on the PROFILED objective phi(r) = max_w l(w; r) rather
 * than on a slice through stale weights.
 *
 * ON THE GAUGE PIN, AND WHY IT IS KEPT.
 * The inherited writeRates normalises rates so that sum_j w_j r_j == 1 using the oracle's FIXED weight
 * vector w. With re-profiling that pin is no longer what enforces the moment constraint -- the literal
 * profiler enforces sum_j w_j r_j = target_moment itself. The pin is kept anyway, for a reason that is
 * easy to miss: because w is a probability vector, sum_j w_j r_j == 1 forces min_j r_j <= 1 <= max_j r_j,
 * which is exactly the condition for the literal profiler's feasible set to be NON-EMPTY. So the pin
 * guarantees, by construction, that every trial rate vector admits a feasible weight solve.
 *
 * The cost is that the rate pass explores only the slice {r : sum_j w_j r_j = 1} for the w this oracle
 * was constructed with, rather than the full k-dimensional rate space. That is coherent with section
 * 7.1, whose major cycle profiles weights FIRST and then moves rates: the slice is refreshed by step 1
 * of every cycle. It is recorded here as a scope limit rather than presented as the full rate block.
 */
class ProfiledRateOracle : public RateBlockOracle {
public:
    double forcing_gap = 1e-8;
    std::size_t profile_max_iterations = 10000;
    std::size_t trial_budget = 250;

    // Cost, decomposed. A single seconds-per-solve ratio is not a measurement of the convex solver:
    // every trial also pays three full tree traversals and two O(nptn*k) buffer builds, and the
    // traversals are threaded while the convex solve is serial, so an undecomposed ratio is partly a
    // function of the thread count. These three are what let the cost be attributed honestly.
    double secs_extract = 0.0;
    double secs_solve = 0.0;
    double secs_realise = 0.0;

    std::size_t weight_solves = 0;
    std::size_t profile_iterations_total = 0;
    std::size_t profile_capped_count = 0;      // MAX_ITERATIONS
    std::size_t profile_stall_count = 0;       // NUMERICAL_STALL -- the degenerate tail's ACTUAL exit
    std::size_t profile_other_count = 0;       // anything not CONVERGED_GAP
    std::size_t refused_start_count = 0;
    std::size_t infeasible_count = 0;
    std::size_t extract_failed_count = 0;
    double max_gap_seen = 0.0;
    std::size_t inf_gap_count = 0;

    // Separating these two settles a question a single max cannot: whether NUMERICAL_STALL is the
    // degenerate tail (a solve giving up far ABOVE the forcing gap) or the arithmetic floor (a solve that
    // reached machine precision and tripped the negative-gap guard). They call for opposite responses --
    // fix the solver, or relabel the exit -- so the counter that cannot tell them apart is useless.
    double max_gap_stall = 0.0;
    std::size_t stall_below_forcing = 0;

    // The domain rejections: trial points where phi is UNDEFINED, not merely expensive.
    std::size_t domain_rejects = 0;
    std::size_t first_reject_eval = 0;
    const char *first_reject_reason = "-";
    double max_start_resid = 0.0;

    // A typed abort. The objective must never silently switch to a DIFFERENT function mid-solve: since
    // phi(r) >= l(w_stale; r), such a switch is one-sided, the Armijo test fails, the line search
    // collapses to the incumbent and dfpmin exits via TOLX -- reporting a small gain that reads as
    // "the rate block is converged". Instead, latch a reason, return the last valid value so the pass
    // terminates quickly and predictably, and refuse to report a gain at all.
    bool aborted = false;
    const char *abort_reason = "-";
    double last_valid_value = 0.0;
    bool have_last_valid = false;

    // THE BEST POINT ACTUALLY VISITED, and why it must be tracked separately from the endpoint.
    // dfpmin returns wherever its loop happened to stop, which is NOT necessarily the best point it
    // evaluated: a rejected trial perturbs the inverse-Hessian update and the forward-difference
    // gradient, and the search can then walk DOWNHILL and exit there. Measured, not assumed --
    // injecting a single rejection at evaluation 12 on the example alignment turned a +0.0377-nat rate
    // gain into -0.4246, reported as a clean realised endpoint. Recording the best visited value is what
    // makes that visible instead of silently publishing a loss as "the gain".
    double best_value = std::numeric_limits<double>::infinity();
    std::vector<double> best_var;
    bool have_best = false;

    double abortWith(const char *reason) {
        if (!aborted) { aborted = true; abort_reason = reason; }
        ++evaluations;
        return have_last_valid ? last_valid_value : 0.0;
    }

    /**
     * REJECT a trial point rather than abort the pass.
     *
     * This is the correction to step 2's first cut, and the distinction is not cosmetic. An out-of-domain
     * trial is not a solver failure: phi(r) = max_w l(w; r) is genuinely UNDEFINED where no feasible w
     * gives a finite likelihood, so on the extended reals the minimised objective is +INF there and the
     * line search is supposed to back off. Aborting instead threw away the whole pass on the first
     * overshoot -- which is exactly what happened, on 4 of 4 cells, and it reported the loss as a
     * "refused start" because the start check sat in front of the exit-reason check.
     *
     * A large FINITE value is returned, not HUGE_VAL: dfpmin does arithmetic on f (slopes, quadratic and
     * cubic interpolation inside lnsrch), and an infinity there propagates NaN into the step length
     * instead of shortening it. The offset is scaled off the incumbent so it dominates any real gain at
     * any alignment size without overflowing.
     */
    double rejectDomain(const char *why) {
        ++domain_rejects;
        if (first_reject_eval == 0) { first_reject_eval = evaluations + 1; first_reject_reason = why; }
        ++evaluations;
        const double base = have_last_valid ? last_valid_value : 0.0;
        return base + 1.0e-3 * std::fabs(base) + 1.0e6;
    }

    std::size_t ndimOfFree() const { return free_cat.size(); }

    // FAULT INJECTION, and why it is not optional. The out-of-domain path only fires when a trial point
    // makes the weight problem insoluble, which on a cheap cell never happens -- so a cheap-cell run
    // exercises none of it and passes blind. This forces a rejection at a chosen evaluation index, which
    // is what lets a gate prove that the line search RECOVERS from a rejection instead of assuming it.
    std::size_t fault_at_eval = 0;

    double targetFunk(double x[]) override {
        if (aborted) return abortWith(abort_reason);
        writeRates(x);
        if (write_failed) return abortWith("arithmetic-failure");
        tree->clearAllPartialLH();

        if (fault_at_eval != 0 && evaluations + 1 == fault_at_eval)
            return rejectDomain("injected-fault");

        if (weight_solves >= trial_budget) return abortWith("trial-budget");

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        ComponentExtraction e;
        std::string why;
        const bool ok = extractComponentsAtUniformProp(tree, &e, &why);
        secs_extract += std::chrono::duration<double>(Clock::now() - t0).count();
        if (!ok) { ++extract_failed_count; return rejectDomain("extract-failed"); }

        // THE DETERMINISTIC START. `w` is the pass's fixed weight vector, and writeRates pinned the rates
        // so that sum_j w_j r_j == 1, so `w` is feasible for the target_moment = 1 problem BY
        // CONSTRUCTION at every trial. Using the LIVE proportions instead would make the start -- and
        // hence the returned value -- depend on which trial ran last.
        freerate_profile::ProfileProblem problem = literalProblemFrom(e, 1.0);
        freerate_profile::ProfileOptions options;
        options.gap_tolerance = forcing_gap;
        options.max_iterations = profile_max_iterations;

        // Measure how feasible the supplied start actually is. writeRates pins sum_j w_j r_j == 1, so this
        // SHOULD be at rounding; recording it is what distinguishes "the pin drifted" from "the objective
        // went non-finite", which the profiler's single supplied_start_used flag conflates.
        double mass_resid = -1.0, moment_resid = 0.0;
        for (int c = 0; c < k; ++c) {
            mass_resid += w[(std::size_t)c];
            moment_resid += w[(std::size_t)c] * e.rate[(std::size_t)c];
        }
        max_start_resid = std::max(max_start_resid,
                                   std::max(std::fabs(mass_resid), std::fabs(moment_resid - 1.0)));

        const auto t1 = Clock::now();
        const freerate_profile::ProfileResult r = freerate_profile::solve(problem, options, w);
        secs_solve += std::chrono::duration<double>(Clock::now() - t1).count();

        ++weight_solves;
        profile_iterations_total += r.iterations;

        const double gap = r.bestGapBound();
        if (std::isfinite(gap)) max_gap_seen = std::max(max_gap_seen, gap);
        else ++inf_gap_count;          // an infinite gap must never read as a clean zero

        if (r.reason == freerate_profile::ExitReason::MAX_ITERATIONS) ++profile_capped_count;
        else if (r.reason == freerate_profile::ExitReason::NUMERICAL_STALL) {
            ++profile_stall_count;
            if (std::isfinite(gap)) {
                max_gap_stall = std::max(max_gap_stall, gap);
                if (gap <= forcing_gap) ++stall_below_forcing;
            }
        }
        else if (r.reason != freerate_profile::ExitReason::CONVERGED_GAP) ++profile_other_count;

        // ORDER MATTERS HERE. INVALID_INPUT / NONFINITE_OBJECTIVE / INFEASIBLE_POLYTOPE mean the weight
        // PROBLEM had no finite solution, and the profiler returns from those paths early with
        // supplied_start_used still false. Testing the start first therefore blamed the start for a failure
        // of the problem, and named the wrong cause in the abort reason on every cell.
        if (r.reason == freerate_profile::ExitReason::INVALID_INPUT ||
            r.reason == freerate_profile::ExitReason::NONFINITE_OBJECTIVE ||
            r.reason == freerate_profile::ExitReason::INFEASIBLE_POLYTOPE)
            return rejectDomain(freerate_profile::exitReasonName(r.reason));

        // A refused start is NOT a failure. The literal weight problem is concave over a polytope, so a
        // solve certified to gap G returns the same maximiser to within G whatever point it started from;
        // the start buys iterations, not correctness. Count it as the cost signal it is.
        if (!r.supplied_start_used) ++refused_start_count;

        if (!profiledPointIsFeasible(r)) { ++infeasible_count; return rejectDomain("infeasible-profiled-point"); }
        if (r.weight.size() != (std::size_t)k) return abortWith("size-mismatch");

        for (int c = 0; c < k; ++c) site_rate->setProp(c, r.weight[(std::size_t)c]);
        const auto t2 = Clock::now();
        tree->clearAllPartialLH();
        const double value = -tree->computeLikelihood();
        secs_realise += std::chrono::duration<double>(Clock::now() - t2).count();

        last_valid_value = value;
        have_last_valid = true;
        if (value < best_value) {
            best_value = value;
            best_var.assign(x + 1, x + 1 + ndimOfFree());   // dfpmin indexes from 1
            have_best = true;
        }
        ++evaluations;
        return value;
    }
};

} // namespace

void reportPhase1SolveProbe(PhyloTree *tree, const char *context) {
    if (getenv("IQ_FR_SOLVE") == nullptr)
        return;                       // inert by default: an unset run is byte-for-byte unchanged

    const char *tag = (context != nullptr) ? context : "?";
    if (tree == nullptr) return;

    // ---- scope screen: mirror the diagnostics' refusals so the probe covers the same cell set ----
    if (tree->isSuperTree()) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=UNSUPPORTED reason=partitioned\n", tag);
        return;
    }
    RateHeterogeneity *site_rate = tree->getRate();
    RateFree *free_rate = dynamic_cast<RateFree *>(site_rate);
    if (free_rate == nullptr) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=UNSUPPORTED reason=not-a-freerate-model\n", tag);
        return;
    }
    const int k = site_rate->getNRate();
    if (k < 3) {
        // The literal pocket has k-2 weight DOF, so k=2 has none (section 5.4 says this is expected).
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=UNSUPPORTED reason=k-too-small k=%d\n", tag, k);
        return;
    }
    if (site_rate->getPInvar() > 0.0) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=UNSUPPORTED reason=additive-invariant-background\n",
                tag);
        return;
    }
    if (free_rate->getNDim() == 0) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=UNSUPPORTED reason=rate-params-fixed-by-user\n", tag);
        return;
    }

    Phase1ProbeOptions opt;
    opt.forcing_gap = envPositiveDouble("IQ_FR_SOLVE_GAP", opt.forcing_gap);
    opt.rate_trial_budget =
        (std::size_t)envPositiveDouble("IQ_FR_SOLVE_TRIALS", (double)opt.rate_trial_budget);
    opt.profile_max_iterations =
        (std::size_t)envPositiveDouble("IQ_FR_SOLVE_PROFITER", (double)opt.profile_max_iterations);

    // ---- full snapshot. Every exit below is post-mutation. ----
    std::vector<double> saved_rate((std::size_t)k), saved_prop((std::size_t)k);
    for (int c = 0; c < k; ++c) {
        saved_rate[(std::size_t)c] = site_rate->getRate(c);
        saved_prop[(std::size_t)c] = site_rate->getProp(c);
    }
    DoubleVector saved_len;
    tree->saveBranchLengths(saved_len);

    tree->clearAllPartialLH();
    const double base_lnl = tree->computeLikelihood();

    auto restoreAll = [&]() {
        for (int c = 0; c < k; ++c) {
            site_rate->setRate(c, saved_rate[(std::size_t)c]);
            site_rate->setProp(c, saved_prop[(std::size_t)c]);
        }
        tree->restoreBranchLengths(saved_len);
        tree->clearAllPartialLH();
    };

    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    auto secondsSince = [](const Clock::time_point &t) {
        return std::chrono::duration<double>(Clock::now() - t).count();
    };

    // ---- cycle step 1: profile weights at the incumbent rates ----
    const auto t_w1 = Clock::now();
    ComponentExtraction e1;
    std::string why;
    if (!extractComponentsAtUniformProp(tree, &e1, &why)) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=EXTRACT_FAILED reason=%s\n", tag, why.c_str());
        restoreAll();
        return;
    }
    freerate_profile::ProfileOptions w_opt;
    w_opt.gap_tolerance = opt.forcing_gap;
    w_opt.max_iterations = opt.profile_max_iterations;
    const freerate_profile::ProfileResult w1 =
        freerate_profile::solve(literalProblemFrom(e1, e1.actual_moment), w_opt, e1.weight);
    const double w1_secs = secondsSince(t_w1);

    double lnl_after_w1 = base_lnl;
    bool w1_committed = false;
    if (w1.supplied_start_used && profiledPointIsFeasible(w1) && w1.weight.size() == (std::size_t)k) {
        for (int c = 0; c < k; ++c) site_rate->setProp(c, w1.weight[(std::size_t)c]);
        tree->clearAllPartialLH();
        const double moved = tree->computeLikelihood();
        if (moved > base_lnl) { lnl_after_w1 = moved; w1_committed = true; }
        else {
            for (int c = 0; c < k; ++c) site_rate->setProp(c, saved_prop[(std::size_t)c]);
            tree->clearAllPartialLH();
        }
    }

    // ---- cycle step 2/3: the RE-PROFILED rate pass ----
    std::vector<double> w_now((std::size_t)k);
    for (int c = 0; c < k; ++c) w_now[(std::size_t)c] = site_rate->getProp(c);

    // Freeze zero-weight atoms: their rate is an exact null coordinate (section 7.2).
    std::vector<int> active;
    for (int c = 0; c < k; ++c)
        if (w_now[(std::size_t)c] >= opt.active_weight_floor) active.push_back(c);
    int anchor = active.empty() ? -1 : active[0];
    for (std::size_t i = 1; i < active.size(); ++i)
        if (site_rate->getRate(active[i]) > site_rate->getRate(anchor)) anchor = active[i];

    if (active.size() < 2 || anchor < 0 || !(site_rate->getRate(anchor) > 0.0)) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=TOO_FEW_ACTIVE active=%zu\n", tag, active.size());
        restoreAll();
        return;
    }

    ProfiledRateOracle oracle;
    oracle.tree = tree;
    oracle.site_rate = site_rate;
    oracle.k = k;
    oracle.w = w_now;
    oracle.anchor = anchor;
    oracle.forcing_gap = opt.forcing_gap;
    oracle.profile_max_iterations = opt.profile_max_iterations;
    oracle.trial_budget = opt.rate_trial_budget;
    oracle.fault_at_eval =
        (std::size_t)envPositiveDouble("IQ_FR_SOLVE_FAULT_AT", 0.0);   // 0 = off; gate-only
    oracle.frozen_ratio.assign((std::size_t)k, 0.0);
    oracle.free_cat.clear();
    for (std::size_t i = 0; i < active.size(); ++i)
        if (active[i] != anchor) oracle.free_cat.push_back(active[i]);
    for (int c = 0; c < k; ++c)
        oracle.frozen_ratio[(std::size_t)c] =
            site_rate->getRate(c) / site_rate->getRate(anchor);

    const int ndim = (int)oracle.free_cat.size();
    std::vector<double> var((std::size_t)ndim + 1, 0.0), lower((std::size_t)ndim + 1, 0.0),
        upper((std::size_t)ndim + 1, 0.0);
    bool box_excludes = false;
    for (int i = 0; i < ndim; ++i) {
        const double vi =
            site_rate->getRate(oracle.free_cat[(std::size_t)i]) / site_rate->getRate(anchor);
        var[(std::size_t)i + 1] = vi;
        lower[(std::size_t)i + 1] = FR_RATIO_LOWER;
        upper[(std::size_t)i + 1] = FR_RATIO_UPPER;
        if (vi < FR_RATIO_LOWER) box_excludes = true;
        if (vi > FR_RATIO_UPPER) var[(std::size_t)i + 1] = FR_RATIO_UPPER;
    }
    if (box_excludes) {
        fprintf(stderr, "[FRSOLVE] ctx=%s PROBE STATUS=BOX_EXCLUDES_INCUMBENT\n", tag);
        restoreAll();
        return;
    }
    std::unique_ptr<bool[]> bound_check(new bool[(std::size_t)ndim + 1]);
    for (int i = 0; i <= ndim; ++i) bound_check[(std::size_t)i] = false;

    // The identity point must reproduce the incumbent, or the parameterisation is wrong and every number
    // below is meaningless. Measured, not assumed. Note this costs one weight solve.
    const double seed_lnl = -oracle.targetFunk(var.data());
    const double seed_err = seed_lnl - lnl_after_w1;

    const double gtol = opt.rate_grad_target / std::max(std::fabs(base_lnl), 1.0);
    const auto t_rate = Clock::now();
    oracle.minimizeMultiDimen(var.data(), ndim, lower.data(), upper.data(), bound_check.get(), gtol);
    const double rate_secs = secondsSince(t_rate);

    // Realise at a COHERENT point. Writing the rates alone is not enough: three of dfpmin's four exits
    // leave the model holding the PROPORTIONS from a finite-difference probe or a rejected line-search
    // trial, not the ones profiled at `var`. Reporting that pair would bias rate_gain DOWN and the
    // following weight gain UP by the same amount -- a spurious transfer of gain from the rate block to
    // the weight block, which is exactly the signature this track already had to retract once. Re-running
    // targetFunk at `var` re-profiles the weights there, so the reported value is phi(var).
    const std::size_t rejects_before_realise = oracle.domain_rejects;
    double lnl_after_rate = lnl_after_w1;
    bool realised = false;
    if (!oracle.aborted) {
        const double v = -oracle.targetFunk(var.data());
        // If the re-profile at `var` itself rejects, `v` is the +1e6 penalty, not a likelihood. Reporting
        // it would print a catastrophic loss for a point the pass never actually stood on.
        if (oracle.domain_rejects == rejects_before_realise && !oracle.aborted) {
            lnl_after_rate = v;
            realised = true;
        }
    }

    // How much better was the BEST point the pass visited than the point it stopped at? This is a
    // property of the OPTIMISER, not of the model, and step 3's section 7.6 acceptance is what will
    // eventually make it zero by construction. Step 2 only has to refuse to hide it.
    const double best_lnl = oracle.have_best ? -oracle.best_value : lnl_after_rate;
    const double endpoint_shortfall = realised ? std::max(0.0, best_lnl - lnl_after_rate) : 0.0;
    const bool rate_monotone = realised && (lnl_after_rate >= lnl_after_w1);

    // ---- cycle step 5: re-profile weights at the new rates ----
    const auto t_w2 = Clock::now();
    ComponentExtraction e2;
    double lnl_final = lnl_after_rate;
    double w2_gap = std::numeric_limits<double>::infinity();
    if (extractComponentsAtUniformProp(tree, &e2, &why)) {
        const freerate_profile::ProfileResult w2 =
            freerate_profile::solve(literalProblemFrom(e2, e2.actual_moment), w_opt, e2.weight);
        w2_gap = w2.bestGapBound();
        if (w2.supplied_start_used && profiledPointIsFeasible(w2) &&
            w2.weight.size() == (std::size_t)k) {
            for (int c = 0; c < k; ++c) site_rate->setProp(c, w2.weight[(std::size_t)c]);
            tree->clearAllPartialLH();
            const double moved = tree->computeLikelihood();
            if (moved > lnl_after_rate) lnl_final = moved;
            else {
                for (int c = 0; c < k; ++c) site_rate->setProp(c, w_now[(std::size_t)c]);
                tree->clearAllPartialLH();
            }
        }
    }
    const double w2_secs = secondsSince(t_w2);
    const double total_secs = secondsSince(t_start);

    // ---- restore and PROVE it element-wise ----
    restoreAll();
    const double restored_lnl = tree->computeLikelihood();
    double max_param_err = 0.0;
    for (int c = 0; c < k; ++c) {
        max_param_err = std::max(max_param_err,
                                 std::fabs(site_rate->getRate(c) - saved_rate[(std::size_t)c]));
        max_param_err = std::max(max_param_err,
                                 std::fabs(site_rate->getProp(c) - saved_prop[(std::size_t)c]));
    }
    DoubleVector now_len;
    tree->saveBranchLengths(now_len);
    if (now_len.size() == saved_len.size()) {
        for (std::size_t i = 0; i < now_len.size(); ++i)
            max_param_err = std::max(max_param_err, std::fabs(now_len[i] - saved_len[i]));
    } else {
        max_param_err = std::numeric_limits<double>::infinity();
    }

    // ---- report: cost first, because cost is what step 2 exists to decide ----
    // The realised mean rate of the FITTED mixture. The section 4.1 contract is sum_j w_j r_j = 1, and
    // nothing above enforces it on the live proportions -- only measuring it can show whether the pass
    // stayed on the contract or drifted off it into the global-scale direction section 7.2 excludes.
    double moment_out = 0.0;
    for (int c = 0; c < k; ++c) moment_out += site_rate->getProp(c) * site_rate->getRate(c);

    const double weight_solves = (double)oracle.weight_solves;
    fprintf(stderr,
            "[FRSOLVE] ctx=%s PHASE1PROBE k=%d ndim=%d anchor=%d frozen=%d base_lnl=%.9f "
            "lnl_after_w1=%.9f lnl_after_rate=%.9f lnl_final=%.9f cycle_gain=%.9f "
            "w1_gain=%.9f rate_gain=%.9f w2_gain=%.9f seed_err=%.3e\n",
            tag, k, ndim, anchor, k - (int)active.size(), base_lnl, lnl_after_w1, lnl_after_rate,
            lnl_final, lnl_final - base_lnl, lnl_after_w1 - base_lnl,
            lnl_after_rate - lnl_after_w1, lnl_final - lnl_after_rate, seed_err);
    fprintf(stderr,
            "[FRSOLVE] ctx=%s PHASE1COST weight_solves=%zu rate_evals=%zu profile_iters=%zu "
            "profile_capped=%zu profile_stall=%zu profile_other=%zu refused_start=%zu infeasible=%zu "
            "extract_failed=%zu aborted=%d abort_reason=%s realised=%d "
            "rate_monotone=%d best_lnl=%.9f endpoint_shortfall=%.9f "
            "domain_rejects=%zu first_reject_eval=%zu first_reject_reason=%s max_start_resid=%.3e "
            "forcing_gap=%.1e max_gap_seen=%.3e max_gap_stall=%.3e stall_below_forcing=%zu "
            "inf_gap=%zu w2_gap=%.3e moment_out=%.12f "
            "secs_total=%.2f secs_w1=%.2f secs_rate=%.2f secs_w2=%.2f "
            "secs_extract=%.3f secs_solve=%.3f secs_realise=%.3f solve_per_trial=%.4f\n",
            tag, oracle.weight_solves, oracle.evaluations, oracle.profile_iterations_total,
            oracle.profile_capped_count, oracle.profile_stall_count, oracle.profile_other_count,
            oracle.refused_start_count, oracle.infeasible_count,
            oracle.extract_failed_count, oracle.aborted ? 1 : 0, oracle.abort_reason, realised ? 1 : 0,
            rate_monotone ? 1 : 0, best_lnl, endpoint_shortfall,
            oracle.domain_rejects, oracle.first_reject_eval, oracle.first_reject_reason,
            oracle.max_start_resid, opt.forcing_gap,
            oracle.max_gap_seen, oracle.max_gap_stall, oracle.stall_below_forcing,
            oracle.inf_gap_count, w2_gap, moment_out,
            total_secs, w1_secs, rate_secs, w2_secs,
            oracle.secs_extract, oracle.secs_solve, oracle.secs_realise,
            weight_solves > 0.0 ? oracle.secs_solve / weight_solves : 0.0);
    fprintf(stderr,
            "[FRSOLVE] ctx=%s PHASE1PROBE STATUS=%s restore_err=%.3e max_param_err=%.3e "
            "-- step 2 MEASURES the re-profiled objective's cost; it certifies nothing, applies no "
            "section 7.6 acceptance, and commits nothing%s\n",
            tag, oracle.aborted ? "ABORTED" : (realised ? "LEGACY_UNCERTIFIED" : "NOT_REALISED"),
            restored_lnl - base_lnl, max_param_err,
            oracle.aborted
                ? " -- the pass ABORTED, so its gain is NOT a measurement"
                : (realised ? ""
                            : " -- the endpoint could not be realised, so its gain is NOT a measurement"));

    if (max_param_err > 0.0) {
        fprintf(stderr,
                "[FRSOLVE] ctx=%s PHASE1PROBE STATUS=STATE-NOT-RESTORED max_param_err=%.6e "
                "-- this run has been perturbed and its result must not be used\n",
                tag, max_param_err);
    }
}

} // namespace freerate
