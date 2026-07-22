/*
 * Typed result, immutable-state, and provenance primitives for the
 * certified FreeRate fitting path.
 *
 * This file deliberately has no dependency on IQ-TREE model objects.  A
 * solver owns these values while it evaluates trial points and commits to
 * an IQ-TREE object only after FreeRateFitResult::certifiedForSelection()
 * succeeds.
 */
#ifndef IQTREE_MODEL_FREERATEFIT_H
#define IQTREE_MODEL_FREERATEFIT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace freerate {

// v2 adds the weight-certificate provenance fields (validity flag, second-order bound and its
// globality flag, tightest bound) to FreeRateBlockMetrics. The serialized field set is part of the
// schema, so adding them is a version change even though no existing field moved or changed meaning.
static const std::uint32_t FREERATE_FIT_SCHEMA_VERSION = 2;

/**
 * Terminal states required by MODELFINDER-FULL-GPU-PLAN.md.
 *
 * GRID_CERTIFIED is an inner, fixed-grid diagnostic result.  It is not a
 * production-candidate success.  In particular, no success state claims a
 * global maximum-likelihood estimate.
 */
enum class FreeRateFitStatus {
    LOCAL_STATIONARY_CERTIFIED,
    BOUNDARY_LOCAL_STATIONARY_CERTIFIED,
    BOUNDARY_DIRECTIONALLY_TESTED,
    FALLBACK_CERTIFIED,
    GRID_CERTIFIED,
    MAXITER,
    REJECT_STALL,
    MULTIBASIN_UNRESOLVED,
    INFEASIBLE_START,
    GPU_CPU_MISMATCH,
    NUMERICAL_FAILURE,
    UNSUPPORTED,
    LEGACY_UNCERTIFIED
};

const char *freeRateFitStatusName(FreeRateFitStatus status);
bool parseFreeRateFitStatus(const std::string &text,
                            FreeRateFitStatus *status);
std::vector<FreeRateFitStatus> allFreeRateFitStatuses();

/** True only for statuses eligible to publish a ModelFinder score. */
bool isProductionCertifiedStatus(FreeRateFitStatus status);

/** Optional data that must remain paired with a rate/weight category. */
struct NamedCategoryVector {
    std::string name;
    std::vector<double> values;
};

/**
 * A self-contained physical parameter snapshot.
 *
 * weights are IQ-TREE's physical positive-category proportions.  Therefore
 * they sum to one for +R and to (1-pinv) for +I+R, while
 * sum(weights[j] * rates[j]) is one in both cases.  Zero-weight categories
 * are retained because fixed-k boundary semantics must not be silently
 * changed by serialization.
 */
struct FreeRatePoint {
    std::vector<double> rates;
    std::vector<double> weights;
    // Real physical edges only.  Root/virtual zero-length sentinels are not
    // part of this vector; every stored length is strictly positive.
    std::vector<double> branch_lengths;
    double pinv = 0.0;
    bool has_invariant = false;
    std::vector<double> substitution_rates;
    std::vector<double> frequencies;

    // Optional category-aligned metadata used by traces and diagnostics.
    std::vector<std::string> category_labels;
    std::vector<NamedCategoryVector> category_data;
};

struct FreeRatePointValidation {
    bool valid = false;
    std::string code;
    std::string message;
    double mass_residual = std::numeric_limits<double>::infinity();
    double mean_residual = std::numeric_limits<double>::infinity();
    double negativity_residual = std::numeric_limits<double>::infinity();
};

FreeRatePointValidation validateFreeRatePoint(
    const FreeRatePoint &point,
    double constraint_tolerance = 1e-10);

/**
 * Sort categories by rate and then weight, permuting all category-aligned
 * metadata.  Named category vectors are themselves sorted by name.  Exact
 * ties preserve input order after all physical/category keys compare equal.
 */
bool canonicalizeFreeRatePoint(FreeRatePoint *point, std::string *error);

/**
 * As above, additionally permuting caller-owned category vectors.  This is
 * useful for per-component likelihood or derivative diagnostics that do not
 * belong in the physical point.  Every vector must have rates.size() items.
 */
bool canonicalizeFreeRatePoint(
    FreeRatePoint *point,
    const std::vector<std::vector<double> *> &paired_vectors,
    std::string *error);

/** A certificate-bearing first-order block diagnostic. */
struct FreeRateBlockMetric {
    bool applicable = false;
    bool evaluated = false;
    double improvement_upper_bound =
        std::numeric_limits<double>::infinity();
    std::uint64_t evaluations = 0;
};

struct FreeRateBlockMetrics {
    bool weight_profile_evaluated = false;
    std::uint64_t weight_profile_evaluations = 0;
    double weight_gap = std::numeric_limits<double>::infinity();

    /**
     * Certificate provenance for weight_gap. Copy these from the oracle's ProfileResult; do not
     * reconstruct them, and do not fill weight_gap without them.
     *
     * weight_gap alone is not a certificate. Its value is clamped at zero, so a broken vertex enumeration
     * -- a point lying OUTSIDE the enumerated hull, whose true directional maximum is negative -- reads as
     * a small non-negative gap and looks like a perfect certificate. weight_gap_is_valid_bound is the flag
     * that separates those two cases and it must be checked.
     *
     * weight_best_bound is the tightest bound the oracle can justify (ProfileResult::bestGapBound()). It
     * exists because the Frank-Wolfe gap ignores curvature and overstates the shortfall by many orders on
     * near-degenerate high-k fits -- measured >=1e7x -- which would fail correct fits. Certification tests
     * this, not weight_gap, so that a curvature-aware bound can rescue them; weight_gap remains recorded
     * because the two disagreeing is itself diagnostic.
     */
    bool weight_gap_is_valid_bound = false;
    double weight_newton_bound = std::numeric_limits<double>::infinity();
    /** False when the second-order bound covers only the active face; see ProfileResult. */
    bool weight_newton_bound_is_global = false;
    double weight_best_bound = std::numeric_limits<double>::infinity();
    double mass_residual = std::numeric_limits<double>::infinity();
    double mean_residual = std::numeric_limits<double>::infinity();
    double negativity_residual = std::numeric_limits<double>::infinity();

    // Weight and rate blocks are intrinsic to every certified +R fit.
    FreeRateBlockMetric rate = {true, false,
        std::numeric_limits<double>::infinity(), 0};
    FreeRateBlockMetric branch;
    FreeRateBlockMetric substitution;
    FreeRateBlockMetric pinv;

    struct BoundaryActivity {
        bool zero_weight = false;
        bool rate_collision = false;
        bool rate_ratio_lower = false;
        bool branch_lower = false;
        bool branch_upper = false;
        bool substitution_parameter = false;
        bool frequency_simplex = false;
        bool pinv_lower = false;
        bool pinv_upper = false;
        bool gauge_interval_lower = false;
        bool gauge_interval_upper = false;

        bool supportBoundary() const;
        bool any() const;
    } boundary;

    bool support_events_evaluated = false;
    double best_tested_support_gain =
        std::numeric_limits<double>::infinity();
    bool continuous_insertion_evaluated = false;
    double continuous_insertion_gain_upper_bound =
        std::numeric_limits<double>::infinity();

    double profiled_likelihood_change =
        std::numeric_limits<double>::infinity();
    double max_scaled_step = std::numeric_limits<double>::infinity();
    std::uint32_t consecutive_small_cycles = 0;

    bool restart_portfolio_evaluated = false;
    double best_restart_gain = std::numeric_limits<double>::infinity();
    bool cpu_gpu_parity_evaluated = false;
    double cpu_gpu_likelihood_delta =
        std::numeric_limits<double>::infinity();

    /**
     * Witness that FreeRateFitResult::final_likelihood was re-evaluated AT final_point.
     *
     * Without this, the certificate constrains the point, every block residual, the provenance and the
     * scope -- but never the one number ModelFinder actually ranks on. A producer that filled the score
     * from a stale, pre-gauge, or wrong-backend evaluation would pass every other gate. The delta is the
     * signed difference between the published score and that fresh re-evaluation.
     */
    bool final_likelihood_verified = false;
    double final_likelihood_recheck_delta =
        std::numeric_limits<double>::infinity();

    bool iteration_cap_reached = false;
    bool line_search_failed = false;
    bool unresolved_support_event = false;
    bool arithmetic_error = false;
};

struct FreeRateEvaluationCounts {
    std::uint64_t value = 0;
    std::uint64_t gradient = 0;
    std::uint64_t hessian_vector = 0;
    std::uint64_t cpu_parity = 0;
    std::uint64_t accepted_steps = 0;
    std::uint64_t rejected_steps = 0;
};

/** Inputs that determine whether two runs really solve the same cell. */
struct FreeRateProvenance {
    std::string schema_identifier = "iqtree-freerate-fit";
    std::uint32_t schema_version = FREERATE_FIT_SCHEMA_VERSION;
    std::string source_commit;
    std::string source_diff_digest;
    std::string solver_version;
    std::string domain_version;
    std::string binary_digest_algorithm;
    std::string binary_digest;
    std::string input_digest_algorithm;
    std::string alignment_digest;
    std::string tree_digest;
    std::string parameter_state_digest;
    std::string candidate_manifest_digest;
    std::string model_name;
    std::string branch_mode;
    std::string command_line;
    std::string run_identifier;
    std::string host;
    std::string evaluator_backend;
    std::string accelerator;
    std::string accelerator_uuid;
    std::string compiler_identifier;
    std::string compiler_version;
    std::string build_type;
    std::string build_flags;
    std::string cuda_toolkit_version;
    std::string cuda_driver_version;
    bool source_tree_dirty = false;
    double runtime_min_branch_length =
        std::numeric_limits<double>::quiet_NaN();
    double runtime_max_branch_length =
        std::numeric_limits<double>::quiet_NaN();
    std::uint64_t seed = 0;
    std::uint32_t thread_count = 0;
    std::uint32_t jolt_ntile = 0;

    bool complete(std::string *reason) const;
};

/** Exact post-gauge state recorded at a trace boundary. */
struct FreeRateStateSnapshot {
    std::uint64_t iteration = 0;
    FreeRatePoint point;
    double pre_gauge_likelihood =
        std::numeric_limits<double>::quiet_NaN();
    double post_gauge_likelihood =
        std::numeric_limits<double>::quiet_NaN();
    double gauge_likelihood_delta =
        std::numeric_limits<double>::quiet_NaN();
    bool accepted = false;
    bool would_clamp = false;
    FreeRateEvaluationCounts counts;
    FreeRateFitStatus status = FreeRateFitStatus::LEGACY_UNCERTIFIED;
};

struct FreeRateCertificationThresholds {
    double likelihood_gain = 1e-4;
    double weight_gap = 1e-5;
    double constraint_residual = 1e-12;
    double scaled_step = 1e-6;
    /**
     * Largest restart improvement a certified fit may leave on the table, in nats.
     *
     * MUST NOT exceed likelihood_gain. It previously defaulted to 0.1 -- a thousand times tau_L -- and
     * valid() only refused values above 0.1, so a portfolio that found a start 0.0999 nats BETTER than
     * the published point still certified as locally stationary at the defaults. That is a larger
     * discrepancy than the entire measured weight-block shortfall on three of the four Stage-0 cells,
     * so the certificate would have been declaring stationarity across a gap it was built to detect.
     */
    double restart_gain = 1e-4;
    double cpu_gpu_likelihood_delta = 1e-4;
    std::uint32_t required_small_cycles = 2;

    bool valid(std::string *reason) const;
};

/** Which parameter blocks belong to the declared optimization problem. */
struct FreeRateFitScope {
    // Schema v1 can certify only the literal mass-and-mean, pure-+R,
    // fixed-branch/fixed-Q slice.  Later scopes require a schema bump once
    // their runtime bounds can be derived rather than asserted by callers.
    std::string weight_formulation = "LITERAL_MASS_MEAN";
    bool optimize_weights = true;
    bool optimize_rates = true;
    bool optimize_branches = false;
    bool optimize_substitution = false;
    bool optimize_pinv = false;

    // Versioned certified-solver domains used to recompute bound activity.
    double rate_ratio_lower = 1e-7;
    double rate_ratio_upper = 1.0;
    double pinv_lower = 0.0;
    double pinv_upper = 1.0;
};

struct FreeRateFitResult {
    // Failure-by-default is intentional: a zero-initialized or partially
    // populated result must never resemble a certificate.
    FreeRateFitStatus status = FreeRateFitStatus::NUMERICAL_FAILURE;
    double final_likelihood = std::numeric_limits<double>::quiet_NaN();
    FreeRatePoint final_point;
    FreeRateBlockMetrics metrics;
    FreeRateEvaluationCounts counts;
    FreeRateProvenance provenance;
    FreeRateCertificationThresholds thresholds;
    FreeRateFitScope scope;
    std::uint32_t starts_attempted = 0;
    std::uint32_t starts_completed = 0;
    std::string trace_digest;
    std::string detail;

    /** Validate both the typed status and every terminal requirement. */
    bool certifiedForSelection(std::string *reason = nullptr) const;

    /** Construct a fail-closed result; certified statuses are rejected. */
    static FreeRateFitResult failure(FreeRateFitStatus failure_status,
                                     const std::string &detail);
};

// Stable compact JSON.  Field order and finite-double rendering are part of
// schema version 1; non-finite doubles are emitted as quoted symbolic values.
std::string stableJson(const FreeRatePoint &point);
std::string stableJson(const FreeRateBlockMetrics &metrics);
std::string stableJson(const FreeRateEvaluationCounts &counts);
std::string stableJson(const FreeRateProvenance &provenance);
std::string stableJson(const FreeRateStateSnapshot &snapshot);
std::string stableJson(const FreeRateFitResult &result);

/**
 * Stable, non-cryptographic FNV-1a-64 trace checksum.
 *
 * Each record is framed by its 64-bit little-endian byte length before its
 * stable JSON bytes are added.  The printable prefix names this algorithm;
 * it must never be described or used as a cryptographic/SHA digest.
 */
class StableTraceDigest {
public:
    StableTraceDigest();
    void reset();
    void addRecord(const std::string &stable_record);
    void addSnapshot(const FreeRateStateSnapshot &snapshot);
    std::uint64_t value() const;
    std::string str() const;

private:
    void addByte(std::uint8_t byte);
    std::uint64_t state_;
};

} // namespace freerate

#endif // IQTREE_MODEL_FREERATEFIT_H
