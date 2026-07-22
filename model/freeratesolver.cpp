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

/** Build the literal mass-and-mean weight problem at the CURRENT rates from an extraction. */
freerate_profile::ProfileProblem literalProblemFrom(const ComponentExtraction &e) {
    freerate_profile::ProfileProblem problem;
    problem.pattern_count = e.pattern_count;
    problem.category_count = e.category_count;
    problem.component_likelihood = e.component_likelihood;
    problem.component_log_scale = e.component_log_scale;
    problem.multiplicity = e.multiplicity;
    problem.rate = e.rate;
    problem.geometry = freerate_profile::FeasibleGeometry::LITERAL_MASS_MEAN;
    // Profile on the surface the live state actually occupies. Hard-coding 1.0 would silently MOVE the
    // point whenever the state sits off the unit-mean contract, and the reported gain would then include
    // that displacement rather than measuring the block.
    problem.target_moment = e.actual_moment;
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

    // Telemetry -- the entire point of step 2.
    std::size_t weight_solves = 0;
    std::size_t profile_iterations_total = 0;
    std::size_t profile_capped_count = 0;
    std::size_t refused_start_count = 0;
    std::size_t infeasible_count = 0;
    std::size_t extract_failed_count = 0;
    double last_gap = std::numeric_limits<double>::infinity();
    double max_gap_seen = 0.0;
    bool budget_hit = false;

    double targetFunk(double x[]) override {
        writeRates(x);
        tree->clearAllPartialLH();

        if (weight_solves >= trial_budget) {
            // Budget bound. Return the fixed-weight value rather than spending another convex solve; the
            // caller reports budget_hit so this is never mistaken for a converged pass.
            budget_hit = true;
            ++evaluations;
            return -tree->computeLikelihood();
        }

        ComponentExtraction e;
        std::string why;
        if (!extractComponentsAtUniformProp(tree, &e, &why)) {
            ++extract_failed_count;
            ++evaluations;
            return -tree->computeLikelihood();
        }

        freerate_profile::ProfileProblem problem = literalProblemFrom(e);
        freerate_profile::ProfileOptions options;
        options.gap_tolerance = forcing_gap;
        options.max_iterations = profile_max_iterations;

        const freerate_profile::ProfileResult r =
            freerate_profile::solve(problem, options, e.weight);
        ++weight_solves;
        profile_iterations_total += r.iterations;
        if (r.reason == freerate_profile::ExitReason::MAX_ITERATIONS) ++profile_capped_count;
        last_gap = r.bestGapBound();
        if (std::isfinite(last_gap)) max_gap_seen = std::max(max_gap_seen, last_gap);

        // A profiler that did not accept the incumbent as its start began from a hull vertex, which is a
        // 1-or-2-support point; writing it would TELEPORT the weights and the resulting objective value
        // would describe a different point than the one requested. Keep the incumbent weights instead.
        if (!r.supplied_start_used) {
            ++refused_start_count;
        } else if (!profiledPointIsFeasible(r)) {
            ++infeasible_count;
        } else if (r.weight.size() == (std::size_t)k) {
            for (int c = 0; c < k; ++c) site_rate->setProp(c, r.weight[(std::size_t)c]);
        }

        tree->clearAllPartialLH();
        ++evaluations;
        return -tree->computeLikelihood();
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
        freerate_profile::solve(literalProblemFrom(e1), w_opt, e1.weight);
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

    // Realise the value the way production does: write the point, clear, ask IQ-TREE. Never report the
    // optimizer's own returned scalar -- it is the value at whatever point the line search last
    // evaluated, which need not be the point now written into the model.
    oracle.writeRates(var.data());
    tree->clearAllPartialLH();
    const double lnl_after_rate = tree->computeLikelihood();

    // ---- cycle step 5: re-profile weights at the new rates ----
    const auto t_w2 = Clock::now();
    ComponentExtraction e2;
    double lnl_final = lnl_after_rate;
    double w2_gap = std::numeric_limits<double>::infinity();
    if (extractComponentsAtUniformProp(tree, &e2, &why)) {
        const freerate_profile::ProfileResult w2 =
            freerate_profile::solve(literalProblemFrom(e2), w_opt, e2.weight);
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
            "profile_capped=%zu refused_start=%zu infeasible=%zu extract_failed=%zu budget_hit=%d "
            "forcing_gap=%.1e max_gap_seen=%.3e w2_gap=%.3e "
            "secs_total=%.2f secs_w1=%.2f secs_rate=%.2f secs_w2=%.2f secs_per_weight_solve=%.4f\n",
            tag, oracle.weight_solves, oracle.evaluations, oracle.profile_iterations_total,
            oracle.profile_capped_count, oracle.refused_start_count, oracle.infeasible_count,
            oracle.extract_failed_count, oracle.budget_hit ? 1 : 0, opt.forcing_gap,
            oracle.max_gap_seen, w2_gap, total_secs, w1_secs, rate_secs, w2_secs,
            weight_solves > 0.0 ? rate_secs / weight_solves : 0.0);
    fprintf(stderr,
            "[FRSOLVE] ctx=%s PHASE1PROBE STATUS=LEGACY_UNCERTIFIED restore_err=%.3e max_param_err=%.3e "
            "-- step 2 MEASURES the re-profiled objective's cost; it certifies nothing, applies no "
            "section 7.6 acceptance, and commits nothing\n",
            tag, restored_lnl - base_lnl, max_param_err);

    if (max_param_err > 0.0) {
        fprintf(stderr,
                "[FRSOLVE] ctx=%s PHASE1PROBE STATUS=STATE-NOT-RESTORED max_param_err=%.6e "
                "-- this run has been perturbed and its result must not be used\n",
                tag, max_param_err);
    }
}

} // namespace freerate
