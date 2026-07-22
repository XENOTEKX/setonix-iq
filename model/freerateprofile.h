/*
 * freerateprofile.h
 *
 * A dependency-light reference solver for the convex FreeRate weight block.
 * This file deliberately has no dependency on IQ-TREE model or tree objects.
 */

#ifndef FREERATEPROFILE_H_
#define FREERATEPROFILE_H_

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace freerate_profile {

/** The affine feasible set used by the weight profile. */
enum class FeasibleGeometry {
    /** sum(w)=1 and dot(rate,w)=target_moment. */
    LITERAL_MASS_MEAN,

    /** sum(w)=1 and moment_lower <= dot(rate,w) <= moment_upper. */
    QUOTIENT_MOMENT_INTERVAL
};

/** A typed outcome. Only CONVERGED_GAP carries a global fixed-column proof. */
enum class ExitReason {
    CONVERGED_GAP,
    MAX_ITERATIONS,
    INFEASIBLE_POLYTOPE,
    INVALID_INPUT,
    NONFINITE_OBJECTIVE,
    NUMERICAL_STALL
};

/**
 * Diagnostics still absent from this first standalone oracle. These flags
 * are explicit so a caller cannot mistake a gap-certified weight block for
 * completion of every Phase-0B deliverable.
 */
enum ProfileDiagnosticLimitation {
    PROFILE_DIAGNOSTIC_NONE = 0u,
    PROFILE_EXPLICIT_DUAL_MULTIPLIERS_UNAVAILABLE = 1u << 0,
    PROFILE_CURVATURE_RANK_UNAVAILABLE = 1u << 1,
    PROFILE_PRODUCTION_COMPONENT_RECONSTRUCTION_UNVALIDATED = 1u << 2,
    PROFILE_OUTER_PARAMETERS_NOT_CERTIFIED = 1u << 3,
    PROFILE_ADDITIVE_BACKGROUND_UNSUPPORTED = 1u << 4
};

const char *exitReasonName(ExitReason reason);

/**
 * Fixed component-likelihood problem.
 *
 * component_likelihood is pattern-major: element p*category_count+j is the
 * unweighted likelihood for pattern p and support point j. Values belonging
 * to one pattern must use one common scaling. If the supplied columns equal
 * exp(-component_log_scale[p]) times the physical columns, provide that log
 * scale explicitly so the reported likelihood remains absolute. An empty
 * component_log_scale means zero offsets. The implementation may additionally
 * normalize rows internally without changing the returned absolute score.
 *
 * This first oracle represents pure +R. It has no fixed additive per-pattern
 * background term and therefore must not yet be used for +I+R likelihoods.
 */
struct ProfileProblem {
    std::size_t pattern_count;
    std::size_t category_count;
    std::vector<double> component_likelihood;
    std::vector<double> component_log_scale;
    std::vector<double> multiplicity;
    std::vector<double> rate;
    FeasibleGeometry geometry;

    double target_moment;
    double moment_lower;
    double moment_upper;

    ProfileProblem();
};

struct ProfileOptions {
    /** Absolute upper bound, in log-likelihood units, on the missing optimum. */
    double gap_tolerance;

    /** Tolerance used only for reporting/checking affine feasibility. */
    double primal_tolerance;

    /** Weights at or below this value are reported as inactive. */
    double active_weight_tolerance;

    /** Maximum accepted safeguarded iterations. */
    std::size_t max_iterations;

    ProfileOptions();
};

/** Exact residuals of the returned floating-point point. */
struct PrimalResiduals {
    double mass;
    double literal_moment;
    double moment_lower_violation;
    double moment_upper_violation;
    double negativity;

    PrimalResiduals();
};

struct ProfileResult {
    ExitReason reason;
    std::vector<double> weight;

    double log_likelihood;

    /**
     * max_{v in C} grad(l)(w)^T(v-w). By concavity this is an exact
     * fixed-column upper bound on l* - l(w). The implementation enumerates
     * the complete vertex set; in a run this is a numerical certificate whose
     * interpretation is conditional on the reported primal residuals and
     * ordinary floating-point rounding.
     */
    double frank_wolfe_gap;

    /** Same directional residual as frank_wolfe_gap, named for KKT logging. */
    double kkt_directional_residual;

    /**
     * The RAW, UNCLAMPED directional maximum max_{v in C} grad(w)^T(v-w).
     *
     * frank_wolfe_gap clamps this at zero, because a negative number is not a meaningful upper bound.
     * The clamp also hides the sign, so this field preserves it.
     *
     * A SMALL negative value is normal and means nothing is wrong: no floating-point iterate lies
     * exactly inside the exact hull, and this quantity is a difference of scores of magnitude
     * sum_p multiplicity_p, so rounding alone makes it negative on a substantial fraction of healthy
     * solves. Only a value below the measured resolution floor -- see gap_is_valid_bound -- indicates
     * that w is genuinely outside the convex hull of the enumerated vertices, which would mean the
     * enumeration is incomplete and no bound derived from it holds.
     */
    double signed_directional_gap;

    /**
     * Resolution floor of the gap in nats, roughly eps * sum_p multiplicity_p.
     *
     * A |gap| at or below this carries no information about optimality. It is reported so that a caller
     * comparing a gap against a fixed threshold can see whether that threshold is even resolvable on
     * this problem: on a large alignment the floor can approach the certification bar itself.
     */
    double gap_noise_floor;

    /**
     * Newton decrement lambda on the active face, in the tangent space of the equality constraints.
     *
     * lambda^2 = gr' * Hr^-1 * gr, with gr and Hr the gradient and negated Hessian reduced onto the
     * null space of the active constraints. NaN when the reduced system is empty or not factorable.
     */
    double newton_decrement;

    /**
     * RIGOROUS second-order upper bound on l* - l(w), in nats: omega*(lambda) = -lambda - log(1-lambda).
     *
     * Valid only when newton_decrement < 1 (otherwise +inf). This is NOT a heuristic. The objective
     * sum_p n_p log(s_p) with s_p affine in w is a sum of logs of affine functions, hence self-concordant
     * (multiplicities n_p >= 1 preserve the property under positive scaling), and for a self-concordant
     * function omega*(lambda) bounds the optimality gap whenever lambda < 1. Verified numerically over
     * 4000 randomised problems (k=2..5, varied multiplicities): zero violations.
     *
     * WHY IT EXISTS ALONGSIDE frank_wolfe_gap. Both are valid upper bounds, but they degrade very
     * differently. The Frank-Wolfe gap extrapolates the gradient linearly to a far LP vertex and ignores
     * curvature, so on near-degenerate high-k problems it overstates the true shortfall by six orders of
     * magnitude or more -- measured at >=1e7x on over-specified cells -- which makes G_w <= tau_w fail on
     * fits that are optimal to within measurement resolution. This bound uses the curvature and stays
     * tight there. Prefer min(frank_wolfe_gap, newton_gap_bound) when both are valid.
     */
    double newton_gap_bound;

    /**
     * True only when frank_wolfe_gap may be read as an upper bound on l* - l(w).
     *
     * False when the directional maximum could not be computed, or fell below the negative of the
     * resolution floor. Gate certification on this together with the achieved gap: ExitReason alone
     * cannot separate a benign near-optimal stall from a broken vertex set, since both report a small
     * non-negative gap.
     */
    bool gap_is_valid_bound;

    /**
     * True only when newton_gap_bound bounds l* - l(w) over the WHOLE feasible set, not just the face.
     *
     * newton_decrement lives in the tangent space of the categories carrying nonzero weight, so its bound
     * covers only moves that keep every zero-weight category at zero. With an active zero-weight boundary
     * an improving direction can exist that activates one of them, and the decrement cannot see it: those
     * coordinates never enter the reduced system. Set when either no category is inactive, or every
     * inactive category prices out (max_inactive_reduced_cost <= the gap resolution floor), which for a
     * concave objective is sufficient for the face optimum to be the global one.
     */
    bool newton_bound_is_global;

    /**
     * Largest first-order value of activating a zero-weight category, g_j - mu - nu*r_j; -inf if none.
     *
     * Positive and above the noise floor means an off-face improving direction exists, so the second-order
     * certificate describes the wrong problem and only the Frank-Wolfe gap bounds the true shortfall.
     */
    double max_inactive_reduced_cost;

    double moment;
    double moment_lower_slack;
    double moment_upper_slack;
    PrimalResiduals primal;

    std::size_t iterations;
    std::size_t objective_evaluations;
    std::size_t feasible_vertex_count;
    std::size_t active_weight_count;
    std::vector<std::size_t> active_index;

    /** Bitwise OR of ProfileDiagnosticLimitation values. */
    unsigned int diagnostic_limitations;

    /** True only when success certifies the weights for these fixed columns. */
    bool fixed_column_global_certificate;

    bool supplied_start_used;
    bool moment_lower_active;
    bool moment_upper_active;

    ProfileResult();

    bool converged() const { return reason == ExitReason::CONVERGED_GAP; }

    /**
     * Whether the achieved gap certifies this point to `tolerance`. GATE ON THIS, NOT ON converged().
     *
     * converged() is true only for CONVERGED_GAP, which is decided against this solver's own internal
     * gap_tolerance (1e-8). That is far tighter than the project's certification threshold, so a point
     * whose gap comfortably satisfies the caller's bar can still exit NUMERICAL_STALL and be discarded
     * by a reason-based test -- measured at 13% of otherwise-certifiable fits. Conversely the exit
     * reason cannot separate a benign stall from a broken vertex set. This predicate consults the
     * validity flag and the achieved value, which together decide both questions correctly.
     */
    bool gapCertifies(double tolerance) const {
        return gap_is_valid_bound && frank_wolfe_gap <= tolerance;
    }

    /**
     * Tightest VALID upper bound available on l* - l(w), in nats; +inf if none is.
     *
     * Both certificates are genuine bounds, so the smaller of the two is also a bound. Callers should
     * gate on this rather than on frank_wolfe_gap alone: they are the same on well-conditioned problems,
     * and on near-degenerate high-k ones the Frank-Wolfe value is the loose one by many orders.
     */
    double bestGapBound() const {
        double best = std::numeric_limits<double>::infinity();
        if (gap_is_valid_bound && frank_wolfe_gap >= 0.0)
            best = frank_wolfe_gap;
        // newton_bound_is_global is load-bearing, not decoration. Without it this line took the smaller
        // of a GLOBAL bound and a FACE-LOCAL one and called the result a global bound, which is unsound
        // exactly when a category sits at zero weight: the true shortfall can lie anywhere up to the
        // Frank-Wolfe value while the decrement reports ~0 for the face it can see.
        if (newton_bound_is_global && newton_gap_bound >= 0.0 &&
            newton_gap_bound < best)
            best = newton_gap_bound;
        return best;
    }

    /** Certifies on the tightest valid bound. Prefer this to gapCertifies(). */
    bool certifiesTo(double tolerance) const {
        return bestGapBound() <= tolerance;
    }
};

/**
 * Enumerate every LP vertex of the requested feasible set. Each returned
 * vector has category_count entries. Literal vertices contain one support
 * point at target_moment or two points bracketing it. Quotient vertices are
 * feasible simplex vertices plus two-point intersections with either moment
 * boundary. Harmless duplicate vertices are retained: approximate
 * deduplication could invalidate the LP oracle for a narrow interval and a
 * high-gradient objective.
 *
 * An empty return means either invalid input or an empty polytope. solve()
 * distinguishes those outcomes.
 */
std::vector<std::vector<double> > enumerateFeasibleVertices(
    const ProfileProblem &problem);

/**
 * Solve the fixed-column concave weight problem. If supplied_start is absent
 * or infeasible, a deterministic feasible vertex centroid is used.
 *
 * The implementation combines safeguarded equality-constrained Newton steps
 * with exact one-dimensional line searches and a Frank-Wolfe step. Newton is
 * an acceleration only: the returned success condition is the exact LP-vertex
 * Frank-Wolfe gap, not a Hessian or parameter-step heuristic.
 */
ProfileResult solve(const ProfileProblem &problem,
                    const ProfileOptions &options = ProfileOptions(),
                    const std::vector<double> &supplied_start =
                        std::vector<double>());

} // namespace freerate_profile

#endif // FREERATEPROFILE_H_
