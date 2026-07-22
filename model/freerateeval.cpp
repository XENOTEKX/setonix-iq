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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

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
// ---------------------------------------------------------------------------------------------------
// Rate block
// ---------------------------------------------------------------------------------------------------

namespace {

/**
 * One-block rate oracle: optimises ONLY the k category rates, at fixed weights, branches and Q.
 *
 * WHY THIS IS NOT THE WEIGHT BLOCK'S CHEAP TRICK. The weight block is solvable from a single likelihood
 * evaluation because the component columns F[p][j] do not depend on w. Rates do appear in them -- a rate
 * multiplies a branch length inside the transition-matrix exponential
 * (tree/phylokernelnew.h:`cat_length[c] = site_rate->getRate(c) * dad_branch->length;`) -- so every trial
 * rate vector needs a genuine tree traversal. This class therefore pays a full likelihood per evaluation,
 * exactly as IQ-TREE's own rate block does (model/ratefree.cpp clears the partials whenever
 * optimizing_params != 2).
 *
 * GAUGE. The parameterisation is the ratio form r_j = v_j / s with s chosen so that sum_j w_j r_j == 1
 * identically, which keeps every trial point on the same unit-mean cross-section the caller measured the
 * weight block on.
 *
 * BE PRECISE ABOUT WHY. It is tempting to justify the pin by invariance -- likelihood is unchanged along
 * (r, b) -> (s*r, b/s), since rates enter only as rate*branch_length and MTree::scaleLength is a pure
 * multiply. That argument does NOT apply here, because this arm holds branches FIXED. At fixed b the
 * direction r -> s*r is not a null direction at all: its derivative is sum_e (dlnL/db_e) * b_e, which is
 * nonzero whenever the published branches are off-optimum. So pinning the gauge EXCLUDES a genuinely
 * likelihood-increasing direction.
 *
 * That exclusion is deliberate and is the definition, not an oversight. Plan section 7.2 defines rate
 * directions as having zero common-scale component, assigning the global scale to the branch block. The
 * reported number is therefore a lower bound on the slack of the GAUGE-FIXED rate block -- the object the
 * plan defines -- and NOT of an unconstrained k-rate move. Do not describe it as the latter.
 */
// Box on the RATIO coordinates v_i = r_i / r_{k-1}.
//
// This is deliberately NOT a copy of model/ratefree.cpp's MIN/MAX_FREE_RATE. Those bound the RAW rate
// (`variables[i+1] = rates[i]` for optimizing_params == 1, with rates[k-1] pinned and never written),
// whereas this arm optimises ratios because it must hold the gauge. Reusing the raw-rate numbers on a
// ratio coordinate silently changes what they mean: with rates sorted ascending the anchor is the
// LARGEST rate, so every ratio is <= 1, the upper bound becomes unreachable, and the lower bound turns
// into a constraint on r_min/r_max rather than on r_min. On the published avian R8 state that is a
// ~6x tighter downward span for the slowest category than production allows -- and the slow category is
// exactly where the +R ridge lives, so the artificial squeeze would bias the measured gain toward zero.
//
// Use the canonical rate-ratio envelope instead: the same [1e-7, 1] declared in FreeRateFitScope and
// audited over 607 archived endpoints in Phase 0B. A wider box can only INCREASE the measured rate gain,
// which is the conservative direction for any "the weight block dominates" reading.
const double FR_RATIO_LOWER = 1e-7;
const double FR_RATIO_UPPER = 1.0;

// Absolute gradient target in nats per unit ratio, converted to dfpmin's relative gtol by dividing by
// |lnL| at the call site. See the solve-strength note in reportRateBlockAttribution.
const double FR_RATE_GRAD_TARGET = 1e-3;
// Re-enter the solver while a pass still pays more than this many nats.
const double FR_RATE_STANDSTILL = 1e-4;
const int FR_RATE_MAX_PASSES = 12;

/**
 * Read a positive double from the environment, or return the default.
 *
 * The solve-strength constants are overridable ONLY so the arm's own convergence can be tested against
 * itself: a reported gain that moves when the budget moves is not a block optimum, and there is no other
 * way to establish that without rebuilding. They are not tuning knobs -- the gate pins the defaults, and
 * a run that overrides them says so in its own telemetry.
 */
double envPositiveDouble(const char *name, double fallback) {
    const char *raw = getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    const double value = atof(raw);
    return (value > 0.0 && std::isfinite(value)) ? value : fallback;
}
// Hard ceiling on full tree likelihoods spent by this arm, so its cost stays bounded on a 692k-pattern
// alignment. Reported via passcap when it binds.
const std::size_t FR_RATE_EVAL_BUDGET = 4000;

class RateBlockOracle : public Optimization {
public:
    PhyloTree *tree = nullptr;
    RateHeterogeneity *site_rate = nullptr;
    int k = 0;
    std::vector<double> w;              // FIXED weights
    std::size_t evaluations = 0;
    bool write_failed = false;

    int getNDim() override { return k - 1; }

    /** Write rates from optimizer variables, restoring sum_j w_j r_j == 1 by construction. */
    void writeRates(const double *v) {
        double s = w[k - 1];
        for (int i = 0; i < k - 1; ++i) s += w[i] * v[i + 1];
        // s = w_{k-1} + sum_{i<k-1} w_i v_i is strictly positive for any feasible point: the weights are
        // non-negative and sum to one, and every v_i is bounded below by FR_RATIO_LOWER > 0. This guard is
        // therefore unreachable in exact arithmetic and exists only for a non-finite v produced by an
        // upstream NaN. Returning without writing leaves the model at the PREVIOUS rates, so targetFunk
        // would stop being a function of its argument; refuse loudly rather than silently score a
        // different point than the one requested.
        if (!(s > 0.0) || !std::isfinite(s)) {
            fprintf(stderr, "[FRATTRIB] RATEBLOCK STATUS=ARITHMETIC-FAILURE s=%.6e\n", s);
            write_failed = true;
            return;
        }
        for (int i = 0; i < k - 1; ++i) site_rate->setRate(i, v[i + 1] / s);
        site_rate->setRate(k - 1, 1.0 / s);
    }

    double targetFunk(double x[]) override {
        writeRates(x);
        tree->clearAllPartialLH();
        ++evaluations;
        return -tree->computeLikelihood();
    }
};

} // namespace

void reportRateBlockAttribution(PhyloTree *tree, const char *context) {
    if (getenv("IQ_FR_ATTRIB") == nullptr)
        return;

    const char *tag = (context != nullptr) ? context : "?";
    if (tree == nullptr) return;

    if (tree->isSuperTree()) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED reason=partitioned-no-joint-rate-block\n",
                tag);
        return;
    }
    RateHeterogeneity *site_rate = tree->getRate();
    RateFree *free_rate = dynamic_cast<RateFree *>(site_rate);
    if (free_rate == nullptr) {
        fprintf(stderr, "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED reason=not-a-freerate-model\n", tag);
        return;
    }
    const int k = site_rate->getNRate();
    if (k < 2) {
        fprintf(stderr, "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED reason=k-too-small k=%d\n", tag, k);
        return;
    }
    // Match the weight arm's scope exactly, so the two gains describe the same cell. The rate block has
    // no structural objection to p_invar, but a cell the weight arm refused must not silently appear
    // here with only one of the two numbers reported.
    if (site_rate->getPInvar() > 0.0) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s RATEBLOCK STATUS=UNSUPPORTED reason=additive-invariant-background\n",
                tag);
        return;
    }
    // Production may have been told to hold the rates. RateFree::getNDim() returns 0 in that case and the
    // shipped optimiser never touches them, so measuring "available rate-block gain" would report slack
    // for a block the user pinned.
    if (free_rate->getNDim() == 0) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED reason=rate-params-fixed-by-user\n", tag);
        return;
    }

    std::vector<double> saved_rate((std::size_t)k), weight((std::size_t)k);
    for (int c = 0; c < k; ++c) {
        saved_rate[(std::size_t)c] = site_rate->getRate(c);
        weight[(std::size_t)c] = site_rate->getProp(c);
    }
    if (!(saved_rate[(std::size_t)k - 1] > 0.0)) {
        fprintf(stderr, "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED reason=nonpositive-anchor-rate\n",
                tag);
        return;
    }
    // The weight arm refuses a cell whose smallest proportion is below its safe-division threshold. Mirror
    // that refusal so the two arms cover the SAME set of cells: a RATEBLOCK line with no WEIGHTBLOCK
    // control beside it is not a controlled comparison, it is one number pretending to be two.
    for (int c = 0; c < k; ++c) {
        if (weight[(std::size_t)c] < FREERATE_MIN_SAFE_PROPORTION) {
            fprintf(stderr,
                    "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED "
                    "reason=degenerate-proportion-weight-arm-would-refuse minw=%.3e\n",
                    tag, weight[(std::size_t)c]);
            return;
        }
    }
    double moment_in = 0.0;
    for (int c = 0; c < k; ++c) moment_in += weight[(std::size_t)c] * saved_rate[(std::size_t)c];

    // Branch lengths are never written by this arm, but snapshot them anyway so a future edit that adds
    // a regauge does not go unnoticed.
    //
    // Note what restore_err below can and cannot show. setRate is a plain store and the saved values are
    // the original doubles, so the PARAMETERS are bit-restored by construction; restore_err compares
    // LIKELIHOODS at bit-identical parameters and is therefore a determinism probe, not a proof of
    // restoration. It is still worth printing -- a nonzero value means computeLikelihood is not
    // reproducible, which would invalidate every gain here -- but it must not be read as evidence that
    // nothing else was perturbed.
    DoubleVector saved_len;
    tree->saveBranchLengths(saved_len);

    tree->clearAllPartialLH();
    const double base_lnl = tree->computeLikelihood();

    RateBlockOracle oracle;
    oracle.tree = tree;
    oracle.site_rate = site_rate;
    oracle.k = k;
    oracle.w = weight;

    const int ndim = k - 1;
    std::vector<double> var((std::size_t)ndim + 1, 0.0);
    std::vector<double> lower((std::size_t)ndim + 1, 0.0);
    std::vector<double> upper((std::size_t)ndim + 1, 0.0);
    for (int i = 0; i < ndim; ++i) {
        var[(std::size_t)i + 1] = saved_rate[(std::size_t)i] / saved_rate[(std::size_t)k - 1];
        lower[(std::size_t)i + 1] = FR_RATIO_LOWER;
        upper[(std::size_t)i + 1] = FR_RATIO_UPPER;
    }
    // A box that excludes the incumbent is not a measurement. With rates sorted ascending the anchor
    // r_{k-1} is the LARGEST, so every seed ratio is <= 1 and only the lower bound can bite. If the seed
    // is outside, the first gradient is taken off-box while every line-search trial is clamped back, so
    // the search can never return to the incumbent and the arm prints gain=0 -- indistinguishable from
    // "the rate block is converged". Refuse instead.
    for (int i = 0; i < ndim; ++i) {
        const double vi = var[(std::size_t)i + 1];
        if (vi < FR_RATIO_LOWER || vi > FR_RATIO_UPPER) {
            fprintf(stderr,
                    "[FRATTRIB] ctx=%s RATEBLOCK STATUS=DECLINED reason=box-excludes-incumbent "
                    "cat=%d ratio=%.6e box=[%.1e,%.1e]\n",
                    tag, i, vi, FR_RATIO_LOWER, FR_RATIO_UPPER);
            return;
        }
    }
    // bound_check false everywhere, matching model/ratefree.cpp. It disables Optimization's random
    // boundary restart, so this measures one deterministic descent and cannot silently wander into a
    // different basin and report the difference as a rate-block gain.
    std::unique_ptr<bool[]> bound_check(new bool[(std::size_t)ndim + 1]);
    for (int i = 0; i <= ndim; ++i) bound_check[(std::size_t)i] = false;

    // The identity point must reproduce the baseline; if it does not, the parameterisation is wrong and
    // every number below is meaningless. Measured, not assumed.
    //
    // It round-trips only because moment_in == 1: v_i = r_i/r_{k-1} gives s = moment_in/r_{k-1}, so
    // writeRates maps r_j -> r_j/moment_in. That holds here because rescaleRates() runs a few lines
    // before this hook in modelfactory.cpp. Move the hook and this silently starts measuring across two
    // gauge slices -- which is exactly what seed_err is printed to catch.
    const double seed_lnl = -oracle.targetFunk(var.data());
    const double seed_err = seed_lnl - base_lnl;

    // SOLVE STRENGTH. dfpmin's gradient exit test is RELATIVE: it stops when
    // max_i |g_i|*max(|p_i|,1) / max(|lnL|,1) < gtol. At the shipped gtol of 1e-4 and an avian |lnL| of
    // 1.1e7 that permits an absolute coordinate gradient of ~1100 nats per unit ratio -- three orders
    // above the gain being measured. Measured consequence on the first gate run: 24-41 evaluations, i.e.
    // at most 2-3 BFGS iterations on 5-7 dimensional problems, FEWER ITERATIONS THAN DIMENSIONS, so the
    // inverse-Hessian never leaves its identity seed and this was 2-3 steepest-descent steps rather than
    // a block solve. A weak rate solve biases the gain toward zero, which manufactures exactly the
    // "weights dominate the rate block" signature this arm exists to test. Scale gtol by |lnL| so the
    // test becomes an ABSOLUTE gradient target, then re-enter until the pass gain stops paying.
    const double grad_target = envPositiveDouble("IQ_FR_RATE_GRAD", FR_RATE_GRAD_TARGET);
    const double standstill = envPositiveDouble("IQ_FR_RATE_STANDSTILL", FR_RATE_STANDSTILL);
    const int max_passes =
        (int)envPositiveDouble("IQ_FR_RATE_PASSES", (double)FR_RATE_MAX_PASSES);
    const std::size_t eval_budget =
        (std::size_t)envPositiveDouble("IQ_FR_RATE_BUDGET", (double)FR_RATE_EVAL_BUDGET);
    const double gtol = grad_target / std::max(std::fabs(base_lnl), 1.0);
    // Bounded explicitly. dfpmin's own ITMAX is 200 and each of its iterations costs 1 line search plus
    // (1+ndim) finite-difference evaluations, so an unbudgeted re-entry loop could reach ~20,000 full
    // tree likelihoods on a 692k-pattern alignment -- hours per cell. A diagnostic must not be able to
    // dominate the run it is diagnosing. Whichever limit binds is reported, so a budget-truncated result
    // is never mistaken for a converged one.
    double running = seed_lnl;
    int passes = 0;
    bool budget_hit = false;
    for (passes = 1; passes <= max_passes; ++passes) {
        oracle.minimizeMultiDimen(var.data(), ndim, lower.data(), upper.data(),
                                  bound_check.get(), gtol);
        oracle.writeRates(var.data());
        tree->clearAllPartialLH();
        const double pass_lnl = tree->computeLikelihood();
        const double pass_gain = pass_lnl - running;
        running = pass_lnl;
        if (!(pass_gain > standstill)) break;
        if (oracle.evaluations >= eval_budget) { budget_hit = true; break; }
    }
    const bool pass_cap_hit = (passes > max_passes) || budget_hit;

    // Realise the value the way production does: write the point, clear, and ask IQ-TREE. Never report
    // the optimizer's own returned scalar -- it is the value at whatever point the line search last
    // evaluated, which is not necessarily the point now written into the model.
    oracle.writeRates(var.data());
    tree->clearAllPartialLH();
    const double opt_lnl = tree->computeLikelihood();

    // Achieved ABSOLUTE gradient at the reported point, by forward differences in nats per unit ratio.
    // Without this the reader cannot tell a converged rate block from a truncated descent, and the two
    // support opposite conclusions. This is the same telemetry gap that cost the +R instrument track a
    // whole build: a stop rule that reports no stationarity evidence proves nothing.
    const double probe_h = envPositiveDouble("IQ_FR_RATE_PROBE_H", 1e-6);
    double max_abs_grad = 0.0;
    double predicted_remaining = 0.0;
    {
        std::vector<double> probe(var);
        for (int i = 0; i < ndim; ++i) {
            const double vi = var[(std::size_t)i + 1];
            const double h = std::max(1e-12, probe_h * std::fabs(vi));
            if (vi + h > FR_RATIO_UPPER || vi - h < FR_RATIO_LOWER) continue;
            probe[(std::size_t)i + 1] = vi + h;
            const double up = -oracle.targetFunk(probe.data());
            probe[(std::size_t)i + 1] = vi - h;
            const double dn = -oracle.targetFunk(probe.data());
            probe[(std::size_t)i + 1] = vi;

            const double g = (up - dn) / (2.0 * h);          // central: no curvature bias
            const double curv = (up - 2.0 * opt_lnl + dn) / (h * h);
            if (std::fabs(g) > max_abs_grad) max_abs_grad = std::fabs(g);
            // Newton step value along this coordinate, g^2/(2|H|), summed over coordinates.
            if (curv < 0.0) predicted_remaining += (g * g) / (2.0 * (-curv));
        }
        oracle.writeRates(var.data());          // undo the last probe write
        tree->clearAllPartialLH();
    }

    // Read the gauge back off the MODEL rather than recomputing it from the same arithmetic that wrote
    // it. sum_j w_j r_j == 1 holds algebraically for every v by construction of writeRates, so a check
    // built from `weight[]` and `var[]` is a tautology that prints 1.000000000000 unconditionally and
    // can never fail. Reading getRate()/getProp() back tests that the model actually holds the state.
    double moment_out = 0.0, minr = 0.0, maxr = 0.0;
    for (int c = 0; c < k; ++c) {
        const double rc = site_rate->getRate(c);
        moment_out += site_rate->getProp(c) * rc;
        if (c == 0 || rc < minr) minr = rc;
        if (c == 0 || rc > maxr) maxr = rc;
    }
    // Bound activity makes the result a CONSTRAINED optimum, not a stationary point. Counted per side:
    // with sorted rates the upper bound is structurally unreachable, so a single count would have been a
    // one-sided indicator wearing a two-sided name.
    int bound_lo = 0, bound_hi = 0;
    for (int i = 0; i < ndim; ++i) {
        const double vi = var[(std::size_t)i + 1];
        if (vi <= FR_RATIO_LOWER * (1.0 + 1e-6)) ++bound_lo;
        if (vi >= FR_RATIO_UPPER * (1.0 - 1e-6)) ++bound_hi;
    }

    for (int c = 0; c < k; ++c) site_rate->setRate(c, saved_rate[(std::size_t)c]);
    tree->restoreBranchLengths(saved_len);
    tree->clearAllPartialLH();
    const double restored_lnl = tree->computeLikelihood();

    fprintf(stderr,
            "[FRATTRIB] ctx=%s RATEBLOCK k=%d dim=%d base_lnl=%.9f opt_lnl=%.9f gain=%.9f "
            "seed_err=%.3e moment_in=%.12f moment_out=%.12f evals=%zu passes=%d passcap=%d "
            "maxgrad=%.6e predrem=%.3e probeh=%.1e gtol=%.3e bound_lo=%d bound_hi=%d "
            "minrate=%.6e maxrate=%.6e restore_err=%.6e\n",
            tag, k, ndim, base_lnl, opt_lnl, opt_lnl - base_lnl, seed_err, moment_in, moment_out,
            oracle.evaluations, passes, pass_cap_hit ? 1 : 0, max_abs_grad, predicted_remaining, probe_h, gtol,
            bound_lo, bound_hi, minr, maxr, restored_lnl - base_lnl);
    // maxgrad is the field that decides whether `gain` is a converged rate-block optimum or a truncated
    // descent. The two support opposite conclusions and nothing else printed here separates them.
    // Gate on the CURVATURE-AWARE estimate, not on |g|.
    //
    // A raw gradient threshold flagged every real cell UNCONVERGED with maxgrad 11-24, and that was a
    // FALSE ALARM: on the DNA-10K cell the measured slope is ~0.93 while the measured curvature is
    // ~1.7e4, so the achievable remaining gain g^2/(2|H|) is ~2.6e-5 nats -- 4,700x below the gain
    // already captured and 380x below IQ-TREE's own 0.010-nat stopping epsilon. Confirmed independently:
    // the reported gain is byte-identical under a 200x larger pass budget and a 60,000-evaluation
    // ceiling, so the solve really has nowhere left to go. This is the same error the weight block
    // taught -- the Frank-Wolfe gap extrapolates a gradient linearly and overstates the shortfall by
    // orders on a curved ridge, which is exactly why the Newton bound exists there.
    //
    // 🔴 predrem is an ESTIMATE, NOT A BOUND. It is built from DIAGONAL secants, so it is blind to
    // off-diagonal coupling and under-predicts on a ridge -- the precise mechanism that killed the `dec`
    // stop rule on the sister instrument track. The weight block's Newton bound is rigorous; this is not,
    // and must never be described as though it were.
    if (predicted_remaining > standstill || pass_cap_hit) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s RATEBLOCK STATUS=UNCONVERGED predrem=%.3e maxgrad=%.6e passcap=%d "
                "-- gain is a LOWER BOUND on rate-block slack, not a block optimum\n",
                tag, predicted_remaining, max_abs_grad, pass_cap_hit ? 1 : 0);
    }

    if (std::fabs(restored_lnl - base_lnl) > 1e-6) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s RATEBLOCK STATUS=STATE-NOT-RESTORED restore_err=%.6e "
                "-- this run has been perturbed and its result must not be used\n",
                tag, restored_lnl - base_lnl);
    }
    if (oracle.write_failed) {
        fprintf(stderr,
                "[FRATTRIB] ctx=%s RATEBLOCK STATUS=VOID reason=rate-write-failed-during-search "
                "-- at least one trial scored a different point than requested\n",
                tag);
    }
}

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
