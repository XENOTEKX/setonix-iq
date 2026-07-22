/*
 * freerateeval.h
 *
 * Phase 0B deliverable: "unweighted common-scale component reconstruction"
 * (MODELFINDER-FULL-GPU-PLAN.md §12 Phase 0B, §4.2, §10.2).
 *
 * Extracts genuinely UNWEIGHTED per-category, per-pattern component likelihoods F[p][j] from a live
 * PhyloTree, together with ONE common per-pattern log-scale offset, so that
 *
 *     s_p = sum_j w_j F[p][j]                 lnL = sum_p n_p ( log s_p + offset_p )
 *
 * reproduces IQ-TREE's production total log-likelihood. §4.2 forbids using the raw category buffers
 * directly, because IQ-TREE folds the category proportion into them before the final contraction:
 *
 *     tree/phylokernelnew.h:`this_val[i] = exp(eval_ptr[i]*len) * prop;`
 *
 * so the stored value is prop[c] * F[p][c] (times a scaling power), NOT F[p][c].
 *
 * Nothing here modifies tree or model state. The extraction is read-only apart from the likelihood
 * buffers the tree recomputes for itself.
 */

#ifndef IQTREE_MODEL_FREERATEEVAL_H
#define IQTREE_MODEL_FREERATEEVAL_H

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

class PhyloTree;

namespace freerate {

/**
 * Unweighted component columns plus the evidence that they reconstruct production.
 *
 * `ok` is false unless every structural precondition held AND the reconstruction was measured. It is
 * deliberately NOT a quality judgement: the caller compares the reported errors against its own
 * thresholds. A caller must never treat `ok` alone as proof that the columns are usable.
 */
struct ComponentExtraction {
    bool ok = false;
    std::string failure_reason;

    std::size_t pattern_count = 0;
    std::size_t category_count = 0;

    /** Pattern-major, element p*category_count+j. Unweighted. */
    std::vector<double> component_likelihood;
    /** One common offset per pattern, in log units. */
    std::vector<double> component_log_scale;
    /** Pattern multiplicity (ptn_freq). */
    std::vector<double> multiplicity;
    /** Physical category rates and IQ-TREE's physical proportions. */
    std::vector<double> rate;
    std::vector<double> weight;

    // ---------------------------------------------------------------- the proof
    /** Total from IQ-TREE's own likelihood routine. */
    double production_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    /** Total recomposed from the extracted columns and weights. */
    double reconstructed_log_likelihood = std::numeric_limits<double>::quiet_NaN();
    /** |production - reconstructed|. */
    double total_likelihood_abs_error = std::numeric_limits<double>::infinity();
    /** Worst per-pattern relative error of sum_j w_j F[p][j] against the production mixture sum. */
    double max_pattern_recompose_rel_error =
        std::numeric_limits<double>::infinity();
    /** Agreement between the two independent production entry points used. */
    double score_cross_check_abs_error = std::numeric_limits<double>::infinity();

    /** Smallest category proportion. The extraction divides by this, so it bounds the conditioning. */
    double min_weight = std::numeric_limits<double>::infinity();
    /** sum_j w_j r_j as actually found in the live state. */
    double actual_moment = std::numeric_limits<double>::quiet_NaN();
    /** |actual_moment - 1|. Nonzero means the state is off the unit-mean contract. */
    double moment_deviation = std::numeric_limits<double>::infinity();

    /**
     * True when p_invar > 0. The +R oracle has no additive per-pattern background term, so the
     * extracted columns alone do NOT describe the model in that case
     * (tree/phylokernelnew.h:`lh_ptn = abs(lh_ptn) + VectorClass().load_a(&ptn_invar[ptn]);`).
     */
    bool additive_background_present = false;
    /** Largest ptn_invar entry, so a caller can see how much mass sits outside the simplex. */
    double max_invariant_contribution = 0.0;

    /** Number of categories whose proportion was too small to divide by safely. */
    std::size_t degenerate_weight_count = 0;
};

/**
 * Extract unweighted component columns from a live tree carrying a FreeRate model.
 *
 * Preconditions checked and reported rather than asserted: the rate model must be FreeRate, the
 * category count must be positive, and the buffers must be allocated. p_invar > 0 is reported via
 * additive_background_present, not rejected, so a caller can still inspect the positive-category
 * block knowingly.
 *
 * The reconstruction errors are the real guard. If a GPU kernel override left the category buffer
 * stale, or the buffer was read while still SIMD-interleaved, the per-pattern recomposition error
 * becomes large and the caller must refuse the columns. That check is strictly stronger than
 * comparing kernel function pointers, because it tests the data rather than the dispatch.
 */
ComponentExtraction extractUnweightedComponents(PhyloTree *tree);

/** Smallest proportion the extraction will divide by; below this a category is called degenerate. */
extern const double FREERATE_MIN_SAFE_PROPORTION;

/**
 * Stage-0 residual attribution for the weight block (MODELFINDER-FULL-GPU-PLAN.md §3.4 items 1-2).
 *
 * Extracts the unweighted columns at the CURRENT state, re-solves the convex weight block exactly, and
 * reports how much likelihood the weight block alone still has available. That number is the whole point
 * of §3.4: it decides whether weights are the block responsible for the observed plateau, and §3.5 makes
 * that a kill gate for the rest of the plan.
 *
 * The reported one-block GAIN is the measured quantity. The Frank-Wolfe gap is reported alongside it as
 * the certificate, but the two are not interchangeable: the gap is an upper bound that has been measured
 * to be many orders of magnitude loose on near-degenerate over-specified mixtures, so a large gap alone
 * is not evidence of an available gain.
 *
 * Entirely inert unless the environment variable IQ_FR_ATTRIB is set, so an unset run is unchanged.
 */
void reportWeightBlockAttribution(PhyloTree *tree, const char *context);

/**
 * Emit an explicit typed decline for an edge-linked partitioned model.
 *
 * `-p`/`-q` set BRLEN_SCALE/BRLEN_FIX, which route to PartitionModelPlen::optimizeParameters. That
 * override never calls ModelFactory::optimizeParameters, so the attribution hook is on a code path a
 * partitioned run never enters and such a run emits NOTHING AT ALL. Silence is the worst outcome
 * available here: this project has repeatedly had single-alignment gates report green while partitioned
 * behaviour was broken, so an unmeasured cell must announce itself rather than look like a pass.
 *
 * This deliberately does NOT measure per-partition weight blocks. Each partition does have its own
 * ModelFactory and RateFree, and measuring them is only a few lines, but k independent per-partition
 * weight blocks are NOT the joint object: partitions are coupled through shared branch lengths and gene
 * rate multipliers. Publishing that sum unlabelled would be a certified-looking number for a quantity
 * MODELFINDER-FULL-GPU-PLAN.md §10.5 explicitly declares UNSUPPORTED in certified mode.
 *
 * Inert unless IQ_FR_ATTRIB is set.
 */
void reportPartitionedDecline(PhyloTree *tree, const char *context,
                              const char *model_kind, int partition_count);

} // namespace freerate

#endif // IQTREE_MODEL_FREERATEEVAL_H
