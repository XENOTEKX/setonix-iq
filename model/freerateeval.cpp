/*
 * freerateeval.cpp — see freerateeval.h.
 *
 * The extraction itself is three lines of arithmetic. Everything else in this file exists to prove the
 * result, because the failure modes here are silent: a stale buffer, a still-interleaved buffer, or a
 * weight divided out twice all yield plausible-looking numbers that are wrong.
 */

#include "freerateeval.h"

#include "model/freerateprofile.h"
#include "model/ratefree.h"
#include "tree/phylotree.h"
#include "utils/tools.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace freerate {

// Below this proportion the division F = lh_cat / prop is not trustworthy. The bound is not arbitrary:
// the EM path regularises proportions at MIN_PROP = 1e-4 (model/ratefree.cpp), while the 2-BFGS
// parameterisation admits ratios down to roughly 1e-7 for k=10. Exactly zero is also reachable through a
// user-specified +R{0,...} or a direct setProp call, so this is a real guard rather than a formality.
const double FREERATE_MIN_SAFE_PROPORTION = 1e-9;

static ComponentExtraction fail(const std::string &why) {
    ComponentExtraction e;
    e.ok = false;
    e.failure_reason = why;
    return e;
}

ComponentExtraction extractUnweightedComponents(PhyloTree *tree) {
    if (tree == nullptr)
        return fail("null tree");
    if (tree->getModelFactory() == nullptr || tree->getModel() == nullptr)
        return fail("tree has no model");

    RateHeterogeneity *site_rate = tree->getRate();
    if (site_rate == nullptr)
        return fail("tree has no rate model");

    // Name partitions BEFORE blaming the rate model. A super-tree carries a dummy RateHeterogeneity, so
    // the RateFree check below would otherwise reject it with "rate model is not FreeRate" -- true, but
    // misleading: the real reason is that this seam has no joint partitioned weight block at all.
    if (tree->isSuperTree())
        return fail("partitioned super-tree; no joint weight block is defined for this seam");

    RateFree *free_rate = dynamic_cast<RateFree *>(site_rate);
    if (free_rate == nullptr)
        return fail("rate model is not FreeRate; this seam is +R only");

    ComponentExtraction out;
    const int ncat = site_rate->getNDiscreteRate();
    if (ncat <= 0)
        return fail("non-positive category count");

    const size_t nptn = tree->getAlnNPattern();
    if (nptn == 0)
        return fail("alignment has no patterns");

    out.category_count = (size_t)ncat;
    out.pattern_count = nptn;

    // ---------------------------------------------------------------- physical parameters
    out.rate.resize((size_t)ncat);
    out.weight.resize((size_t)ncat);
    double moment = 0.0;
    out.min_weight = std::numeric_limits<double>::infinity();
    for (int c = 0; c < ncat; ++c) {
        out.rate[(size_t)c] = site_rate->getRate(c);
        out.weight[(size_t)c] = site_rate->getProp(c);
        moment += out.weight[(size_t)c] * out.rate[(size_t)c];
        if (out.weight[(size_t)c] < out.min_weight)
            out.min_weight = out.weight[(size_t)c];
        if (out.weight[(size_t)c] < FREERATE_MIN_SAFE_PROPORTION)
            out.degenerate_weight_count++;
    }
    out.actual_moment = moment;
    out.moment_deviation = std::fabs(moment - 1.0);

    const double pinv = site_rate->getPInvar();
    out.additive_background_present = (pinv > 0.0);

    if (out.degenerate_weight_count > 0)
        return fail("a category proportion is below the safe division threshold");

    // ---------------------------------------------------------------- production evaluation
    // computePatternLhCat both recomputes the likelihood and de-interleaves the category buffer. Reading
    // _pattern_lh_cat after a plain computeLikelihood() would return SIMD-interleaved values, because the
    // de-interleaving step lives only here:
    //   tree/phylotree.h:`/** transform _pattern_lh_cat from "interleaved" to "sequential", due to vector_size > 1 */`
    const double score_cat = tree->computePatternLhCat(WSL_RATECAT);

    const double *lh_cat = tree->getPatternLhCatPointer();
    if (lh_cat == nullptr)
        return fail("category likelihood buffer is not allocated");

    // Snapshot immediately: the next production call may recompute and re-interleave this buffer.
    std::vector<double> cat_snapshot((size_t)ncat * nptn, 0.0);
    for (size_t p = 0; p < nptn; ++p)
        for (int c = 0; c < ncat; ++c)
            cat_snapshot[p * (size_t)ncat + (size_t)c] = lh_cat[p * (size_t)ncat + (size_t)c];

    // Independent second entry point: per-pattern log-likelihoods and the total. Using a different
    // routine for the offsets than for the columns is deliberate, so the cross-check has real content.
    //
    // WSL_RATECAT (not WSL_NONE) with a NULL category buffer, for two reasons:
    //  1. computePatternLikelihood calls getNumLhCat(wsl) UNCONDITIONALLY at entry, before it tests
    //     ptn_lh_cat, even though ncat is only consumed when that pointer is non-null. WSL_NONE
    //     therefore always aborts on `tree/phylotree.cpp:`case WSL_NONE: ASSERT(0 && "is not WSL_NONE"); return 0;``
    //     regardless of the null buffer. That is a latent defect in the callee, worked around here.
    //  2. Passing NULL keeps it on the `if (ptn_lh_cat)` false path, so it does NOT re-enter
    //     computePatternLhCat and cannot re-interleave the category buffer we just snapshotted.
    //
    // 🔴 Its `cur_logl` out-parameter is NOT usable on this path: the function returns at
    //    `tree/phylotree.cpp:`if (!ptn_lh_cat)`` BEFORE assigning cur_logl, so it stays at whatever the
    //    caller initialised. A second latent defect in the same callee. The per-pattern array IS filled
    //    by then, so the total is re-derived from it below and cross-checked against computePatternLhCat.
    std::vector<double> pattern_lh(nptn, 0.0);
    tree->computePatternLikelihood(pattern_lh.data(), nullptr, nullptr, WSL_RATECAT);

    // Route 1: the value production itself returned.
    out.production_log_likelihood = score_cat;

    // ---------------------------------------------------------------- extraction + reconstruction
    out.component_likelihood.assign(nptn * (size_t)ncat, 0.0);
    out.component_log_scale.assign(nptn, 0.0);
    out.multiplicity.assign(nptn, 0.0);

    double worst_rel = 0.0;
    double recomposed_total = 0.0;
    double max_invar = 0.0;
    // Route 2: the same production total re-derived from the independently-filled per-pattern array.
    double pattern_total = 0.0;

    for (size_t p = 0; p < nptn; ++p) {
        const double freq = tree->ptn_freq[p];
        out.multiplicity[p] = freq;
        pattern_total += freq * pattern_lh[p];

        const double invar = (tree->ptn_invar != nullptr) ? tree->ptn_invar[p] : 0.0;
        if (invar > max_invar)
            max_invar = invar;

        // Production mixture sum for this pattern, in the pattern's own scaled frame.
        double weighted_sum = 0.0;
        for (int c = 0; c < ncat; ++c)
            weighted_sum += cat_snapshot[p * (size_t)ncat + (size_t)c];

        // Divide the folded proportion back out to recover the unweighted column.
        double recomposed = 0.0;
        for (int c = 0; c < ncat; ++c) {
            const double w = out.weight[(size_t)c];
            const double f = cat_snapshot[p * (size_t)ncat + (size_t)c] / w;
            out.component_likelihood[p * (size_t)ncat + (size_t)c] = f;
            recomposed += w * f;
        }

        // Round-trip check of the division only. Weak by construction, but it localises a fault to the
        // division rather than to the offsets if the total later disagrees.
        const double denom = std::fabs(weighted_sum) > 0.0 ? std::fabs(weighted_sum) : 1.0;
        const double rel = std::fabs(recomposed - weighted_sum) / denom;
        if (rel > worst_rel)
            worst_rel = rel;

        // One common offset per pattern, recovered from the production per-pattern log value. The kernel
        // renormalises every category into a single frame before summing, which is what makes a single
        // offset correct here:
        //   tree/phylokernelnew.h:`min_scale = min(min_scale, sum_scale[c]);`
        const double s = weighted_sum + invar;
        if (!(s > 0.0)) {
            out.component_log_scale[p] = 0.0;
            continue;
        }
        const double offset = pattern_lh[p] - std::log(s);
        out.component_log_scale[p] = offset;
        recomposed_total += freq * (std::log(recomposed + invar) + offset);
    }

    out.max_pattern_recompose_rel_error = worst_rel;
    out.max_invariant_contribution = max_invar;
    out.reconstructed_log_likelihood = recomposed_total;
    // Route 3 (our columns) against route 1 (production's own return value).
    out.total_likelihood_abs_error =
        std::fabs(recomposed_total - out.production_log_likelihood);
    // Route 1 against route 2: two production paths that must agree before any of this means anything.
    out.score_cross_check_abs_error = std::fabs(out.production_log_likelihood - pattern_total);

    out.ok = true;
    return out;
}

static void reportWritebackValidation(PhyloTree *tree, const char *tag,
                                      const ComponentExtraction &e,
                                      const freerate_profile::ProfileResult &r,
                                      double claimed_gain);

void reportPartitionedDecline(PhyloTree *tree, const char *context,
                              const char *model_kind, int partition_count) {
    if (getenv("IQ_FR_ATTRIB") == nullptr)
        return;
    fprintf(stderr,
            "[FRATTRIB] ctx=%s STATUS=DECLINED reason=partitioned-no-joint-weight-block "
            "model=%s npart=%d supertree=%d\n",
            (context != nullptr) ? context : "?",
            (model_kind != nullptr) ? model_kind : "?", partition_count,
            (tree != nullptr && tree->isSuperTree()) ? 1 : 0);
}

void reportWeightBlockAttribution(PhyloTree *tree, const char *context) {
    if (getenv("IQ_FR_ATTRIB") == nullptr)
        return;                       // inert by default: an unset run is byte-for-byte unchanged

    const char *tag = (context != nullptr) ? context : "?";

    ComponentExtraction e = extractUnweightedComponents(tree);
    if (!e.ok) {
        fprintf(stderr, "[FRATTRIB] ctx=%s STATUS=DECLINED reason=%s\n", tag,
                e.failure_reason.c_str());
        return;
    }

    // Reconstruction evidence first. If these do not hold, nothing downstream is meaningful and the
    // numbers must not be read as a measurement of the model.
    fprintf(stderr,
            "[FRATTRIB] ctx=%s k=%zu nptn=%zu prod_lnl=%.9f recon_lnl=%.9f "
            "abs_err=%.3e recompose_rel=%.3e xcheck=%.3e minw=%.3e moment=%.12f mdev=%.3e "
            "invar=%d maxinvar=%.3e\n",
            tag, e.category_count, e.pattern_count, e.production_log_likelihood,
            e.reconstructed_log_likelihood, e.total_likelihood_abs_error,
            e.max_pattern_recompose_rel_error, e.score_cross_check_abs_error, e.min_weight,
            e.actual_moment, e.moment_deviation, e.additive_background_present ? 1 : 0,
            e.max_invariant_contribution);

    // The +R oracle has no additive background. With p_invar > 0 the extracted columns do not describe
    // the model, so refuse rather than emit a number that looks like a weight-block residual.
    if (e.additive_background_present) {
        fprintf(stderr, "[FRATTRIB] ctx=%s STATUS=UNSUPPORTED reason=additive-invariant-background\n",
                tag);
        return;
    }

    freerate_profile::ProfileProblem problem;
    problem.pattern_count = e.pattern_count;
    problem.category_count = e.category_count;
    problem.component_likelihood = e.component_likelihood;
    problem.component_log_scale = e.component_log_scale;
    problem.multiplicity = e.multiplicity;
    problem.rate = e.rate;
    problem.geometry = freerate_profile::FeasibleGeometry::LITERAL_MASS_MEAN;
    // Profile on the surface the live state actually occupies. Hard-coding 1.0 would silently MOVE the
    // point whenever the state sits off the unit-mean contract, and the reported "gain" would then
    // include that displacement rather than measuring the weight block.
    problem.target_moment = e.actual_moment;

    freerate_profile::ProfileOptions options;
    freerate_profile::ProfileResult r =
        freerate_profile::solve(problem, options, e.weight);

    // The measured one-block gain: what the weight block alone can still recover from HERE.
    //
    // NOTE ON gain VERSUS fw_gap. They answer different questions and must not be compared as if they
    // were the same quantity. `gain` is what the solve RECOVERED moving from the incumbent weights to
    // the profiled ones; `fw_gap` is evaluated AT the profiled point and bounds what REMAINS there. A
    // converged gap of 2e-9 alongside a 1.5-nat gain is therefore consistent, not contradictory: the
    // solve moved 1.5 nats and then had essentially nothing left.
    const double gain = r.log_likelihood - e.reconstructed_log_likelihood;

    // Emit the fields a caller is supposed to GATE on, not just the headline gap. gap_is_valid_bound is
    // the documented gating field, yet it was the one field not printed -- a reader could only infer it
    // from the sign of an already-clamped number. Also report the tightest valid bound and the second-
    // order certificate, since the Frank-Wolfe value is the loose one on near-degenerate high-k cells.
    fprintf(stderr,
            "[FRATTRIB] ctx=%s WEIGHTBLOCK exit=%s gain=%.9f profiled_lnl=%.9f fw_gap=%.6e "
            "signed_gap=%.6e gap_valid=%d gap_noise=%.3e newton_lambda=%.6e newton_bound=%.6e "
            "newton_global=%d maxdual=%.3e best_bound=%.6e active=%zu/%zu iters=%zu verts=%zu "
            "res_mass=%.3e res_mom=%.3e res_neg=%.3e\n",
            tag, freerate_profile::exitReasonName(r.reason), gain, r.log_likelihood,
            r.frank_wolfe_gap, r.signed_directional_gap, r.gap_is_valid_bound ? 1 : 0,
            r.gap_noise_floor, r.newton_decrement, r.newton_gap_bound,
            r.newton_bound_is_global ? 1 : 0, r.max_inactive_reduced_cost, r.bestGapBound(),
            r.active_weight_count, e.category_count, r.iterations, r.feasible_vertex_count,
            r.primal.mass, r.primal.literal_moment, r.primal.negativity);

    // The certified two-sided bracket on the INCUMBENT's weight-block shortfall. `gain` is what the
    // solve recovered; `bestGapBound()` bounds what remains at the solved point. Their sum bounds the
    // incumbent's distance to the weight-block optimum from above, and `gain` bounds it from below.
    // bestGapBound() admits the second-order bound only when newton_global is set, so the upper end
    // stays a bound over the whole feasible set and not merely over the active face; when it is not set
    // the width falls back to the Frank-Wolfe value, which is global but loose.
    fprintf(stderr, "[FRATTRIB] ctx=%s SHORTFALL bracket=[%.9f, %.9f] width=%.6e\n",
            tag, gain, gain + r.bestGapBound(), r.bestGapBound());

    reportWritebackValidation(tree, tag, e, r, gain);
}

/**
 * THE DECISIVE CHECK ON THE GAIN.
 *
 * Everything above is computed by this module from its own reconstructed columns F[p][j]. A
 * reconstruction that is exact AT THE INCUMBENT still proves nothing about the objective's SHAPE in w --
 * and the gain is precisely a statement about that shape, since it claims the likelihood rises by
 * `gain` when the weights move. If F were subtly wrong as a function of w, the profiled likelihood and
 * the gain would be wrong together and every self-consistency check in this file would still pass.
 *
 * So: write the profiled weights into the live RateFree, let IQ-TREE recompute the likelihood with its
 * own machinery, and compare the REALISED improvement against the CLAIMED gain. Only production can
 * settle this, and it either confirms the gain or exposes the reconstruction.
 *
 * The original weights are restored unconditionally, including on every early return, so the run this
 * measurement is embedded in continues from exactly the state it would otherwise have had.
 */
static void reportWritebackValidation(PhyloTree *tree, const char *tag,
                                      const ComponentExtraction &e,
                                      const freerate_profile::ProfileResult &r,
                                      double claimed_gain) {
    RateHeterogeneity *site_rate = tree->getRate();
    if (site_rate == nullptr || r.weight.size() != e.category_count)
        return;

    // The profiled point must be a legal +R state before it is worth writing anywhere.
    if (!(r.primal.mass <= 1e-9) || !(r.primal.literal_moment <= 1e-9) ||
        !(r.primal.negativity <= 0.0)) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s WRITEBACK STATUS=SKIPPED reason=profiled-point-not-feasible "
                "mass=%.3e mom=%.3e neg=%.3e\n",
                tag, r.primal.mass, r.primal.literal_moment, r.primal.negativity);
        return;
    }

    const std::vector<double> saved = e.weight;   // physical proportions as production left them

    // Baseline from IQ-TREE itself, recomputed the same way the comparison point will be, so the two
    // differ only by the weights and not by which routine produced them.
    tree->clearAllPartialLH();
    const double base_lnl = tree->computeLikelihood();

    for (std::size_t c = 0; c < e.category_count; ++c)
        site_rate->setProp((int)c, r.weight[c]);
    tree->clearAllPartialLH();
    const double moved_lnl = tree->computeLikelihood();

    // Restore before doing anything else with the numbers.
    for (std::size_t c = 0; c < e.category_count; ++c)
        site_rate->setProp((int)c, saved[c]);
    tree->clearAllPartialLH();
    const double restored_lnl = tree->computeLikelihood();

    const double realised_gain = moved_lnl - base_lnl;
    const double discrepancy = realised_gain - claimed_gain;
    const double restore_error = restored_lnl - base_lnl;

    fprintf(stderr,
            "[FRATTRIB] ctx=%s WRITEBACK base_lnl=%.9f moved_lnl=%.9f realised_gain=%.9f "
            "claimed_gain=%.9f discrepancy=%.6e restore_err=%.6e\n",
            tag, base_lnl, moved_lnl, realised_gain, claimed_gain, discrepancy, restore_error);

    // A failed restore is far more serious than a failed validation: it means this diagnostic perturbed
    // the run it was measuring. Say so loudly rather than letting it pass as a rounding note.
    if (std::fabs(restore_error) > 1e-6) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s WRITEBACK STATUS=STATE-NOT-RESTORED restore_err=%.6e "
                "-- this run has been perturbed and its result must not be used\n",
                tag, restore_error);
    }
}

} // namespace freerate
