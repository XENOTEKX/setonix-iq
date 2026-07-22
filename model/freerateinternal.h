/*
 * Internal shared surface for the FreeRate CPU work.
 *
 * NOT a public API. This header exists only so the Phase-1 solver
 * (model/freeratesolver.cpp) can reuse machinery that was written and validated inside
 * model/freerateeval.cpp for the Stage-0 diagnostics. Everything here is already exercised on real data
 * by those diagnostics; promoting it out of an anonymous namespace is a linkage change, not a redesign,
 * and it must remain byte-for-byte behaviour-preserving.
 *
 * Do not widen this surface casually. Each symbol below is here because the solver genuinely needs it:
 * the rate oracle because the solver subclasses it to re-profile weights at every rate trial
 * (MODELFINDER-FULL-GPU-PLAN.md section 7.1 step 3), and the uniform-proportion extraction because an
 * exact convex weight solve legitimately drives weights to zero, which the ordinary extraction path
 * refuses.
 */

#ifndef IQTREE_MODEL_FREERATEINTERNAL_H
#define IQTREE_MODEL_FREERATEINTERNAL_H

#include "freerateeval.h"
#include "utils/optimization.h"

#include <cstddef>
#include <string>
#include <vector>

class PhyloTree;
class RateHeterogeneity;

namespace freerate {

/**
 * Box on the RATIO coordinates q_i = r_i / r_anchor.
 *
 * This is deliberately NOT model/ratefree.cpp's MIN/MAX_FREE_RATE, which bound the RAW rate. Reusing
 * raw-rate numbers on a ratio coordinate silently changes what they mean: with rates sorted ascending
 * the anchor is the LARGEST rate, so every ratio is <= 1, the upper bound becomes unreachable, and the
 * lower bound turns into a constraint on r_min/r_max rather than on r_min. These are the canonical
 * scale-free envelope from plan section 4.1, audited over 607 archived endpoints in Phase 0B.
 */
extern const double FR_RATIO_LOWER;   // 1e-7
extern const double FR_RATIO_UPPER;   // 1.0

/**
 * Read a positive double from the environment, or return the default.
 *
 * Solve-strength constants are overridable ONLY so a solver's own convergence can be tested against
 * itself: a reported gain that moves when the budget moves is not a block optimum. They are not tuning
 * knobs -- gates pin the defaults, and a run that overrides them says so in its own telemetry.
 */
double envPositiveDouble(const char *name, double fallback);

/**
 * One-block rate oracle: optimises the k category rates at FIXED weights, branches and Q.
 *
 * GAUGE. The parameterisation is the ratio form r_j = v_j / s with s chosen so that sum_j w_j r_j == 1
 * identically. Be precise about why: it is tempting to justify the pin by invariance, since likelihood
 * is unchanged along (r, b) -> (s*r, b/s). That argument does NOT apply here, because this arm holds
 * branches FIXED. At fixed b the direction r -> s*r is not a null direction at all -- its derivative is
 * sum_e (dlnL/db_e) * b_e, nonzero whenever the published branches are off-optimum. So pinning the gauge
 * EXCLUDES a genuinely likelihood-increasing direction. That exclusion is deliberate and is the
 * definition, not an oversight: plan section 7.2 defines rate directions as having zero common-scale
 * component, assigning the global scale to the branch block.
 *
 * SUPPORT EVENTS. `configureDefault()` reproduces the original "anchor at k-1, optimise the other k-1
 * ratios" parameterisation exactly, including the order in which the pin denominator is accumulated, so
 * the shipped one-block arm is bit-unchanged. `free_cat` / `frozen_ratio` / `anchor` generalise that to
 * freeze zero-weight atoms: such an atom's rate is an EXACT null coordinate (its column is multiplied by
 * zero in the likelihood AND it drops out of the pin), so optimising it hands the optimiser a zero
 * finite-difference gradient and degenerates the BFGS curvature denominator. Plan section 7.2: a
 * zero-weight atom has no identifiable location. Freeze it -- do not delete it, which would drop to
 * R(k-1) and measure a different model (section 4.1 keeps the nominal Rk count).
 */
class RateBlockOracle : public Optimization {
public:
    PhyloTree *tree = nullptr;
    RateHeterogeneity *site_rate = nullptr;
    int k = 0;
    std::vector<double> w;              // FIXED weights
    std::size_t evaluations = 0;
    bool write_failed = false;

    std::vector<int> free_cat;          // categories this solve may move
    std::vector<double> frozen_ratio;   // r_c / r_anchor for categories neither free nor anchor
    int anchor = -1;                    // category carrying the gauge

    virtual ~RateBlockOracle() {}

    void configureDefault();

    int getNDim() override { return (int)free_cat.size(); }

    /** ratio_c = r_c / r_anchor, from the optimizer variables for free categories. */
    double ratioOf(int c, const double *v) const;

    /** Write rates from optimizer variables, restoring sum_j w_j r_j == 1 by construction. */
    void writeRates(const double *v);

    double targetFunk(double x[]) override;
};

/**
 * Extract the unweighted components at a well-conditioned UNIFORM proportion vector.
 *
 * `extractUnweightedComponents` recovers F by dividing the folded proportion back out, so it refuses
 * outright when any proportion is below the safe-division threshold. An exact convex weight solve
 * legitimately drives weights to zero and plan section 4.1 forbids a positive weight floor, so any
 * committing loop meets that refusal.
 *
 * F does not depend on the weights: `prop` multiplies the whole category block as a scalar
 * (tree/phylokernelnew.h:`this_val[i] = exp(eval_ptr[i]*len) * prop;`) and the partial-likelihood pass
 * never reads a proportion at all, so the partials AND the integer scaling exponents behind the common
 * per-pattern offset are prop-independent. Evaluating at prop = 1/k therefore returns the same F with no
 * ill-conditioned division. Equality is to a few ulp, NOT bit-exact -- prop is folded in before the dot
 * product, so summation rounding differs. Every check against this F must be relative.
 *
 * On return the proportion-derived fields describe the TRUE weights, and the reconstruction is redone at
 * those weights, which doubles as a live test of the trick.
 */
bool extractComponentsAtUniformProp(PhyloTree *tree, ComponentExtraction *out, std::string *why);

} // namespace freerate

#endif
