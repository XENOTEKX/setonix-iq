/* Standalone tests for model/freerateprofile.{h,cpp}. */

#include "freerateprofile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using freerate_profile::ExitReason;
using freerate_profile::FeasibleGeometry;
using freerate_profile::ProfileOptions;
using freerate_profile::ProfileProblem;
using freerate_profile::ProfileResult;

void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) {
        fail(message);
    }
}

bool closeEnough(double left, double right, double tolerance) {
    return std::fabs(left - right) <=
           tolerance * std::max(1.0, std::max(std::fabs(left),
                                             std::fabs(right)));
}

ProfileOptions strictOptions() {
    ProfileOptions options;
    options.gap_tolerance = 1.0e-9;
    options.primal_tolerance = 2.0e-12;
    options.active_weight_tolerance = 1.0e-10;
    options.max_iterations = 20000;
    return options;
}

ProfileProblem makeProblem(std::size_t patterns,
                           std::size_t categories,
                           const std::vector<double> &components,
                           const std::vector<double> &counts,
                           const std::vector<double> &rates) {
    ProfileProblem problem;
    problem.pattern_count = patterns;
    problem.category_count = categories;
    problem.component_likelihood = components;
    problem.multiplicity = counts;
    problem.rate = rates;
    return problem;
}

double directLogLikelihood(const ProfileProblem &problem,
                           const std::vector<double> &weight) {
    long double value = 0.0L;
    for (std::size_t p = 0; p < problem.pattern_count; ++p) {
        long double mixture = 0.0L;
        for (std::size_t j = 0; j < problem.category_count; ++j) {
            mixture += static_cast<long double>(weight[j]) *
                       problem.component_likelihood[p *
                                                        problem.category_count +
                                                    j];
        }
        if (!(mixture > 0.0L)) {
            return -std::numeric_limits<double>::infinity();
        }
        value += static_cast<long double>(problem.multiplicity[p]) *
                 std::log(mixture);
    }
    return static_cast<double>(value);
}

void checkCertified(const ProfileResult &result, const std::string &label,
                    double gap_tolerance = 1.1e-9) {
    if (result.reason != ExitReason::CONVERGED_GAP) {
        std::cerr << label << " diagnostics: gap="
                  << result.frank_wolfe_gap << " iterations="
                  << result.iterations << " mass=" << result.primal.mass
                  << " moment=" << result.moment << '\n';
    }
    require(result.reason == ExitReason::CONVERGED_GAP,
            label + " returned " +
                freerate_profile::exitReasonName(result.reason));
    require(result.frank_wolfe_gap >= 0.0 &&
                result.frank_wolfe_gap <= gap_tolerance,
            label + " did not pass its exact gap");
    require(result.fixed_column_global_certificate,
            label + " did not label its certificate scope");
    require(result.kkt_directional_residual == result.frank_wolfe_gap,
            label + " reported inconsistent FW/KKT directional gaps");
    require(result.primal.mass <= 2.0e-12,
            label + " has a mass residual");
    require(result.primal.literal_moment <= 2.0e-12,
            label + " has a literal moment residual");
    require(result.primal.moment_lower_violation <= 2.0e-12 &&
                result.primal.moment_upper_violation <= 2.0e-12,
            label + " violates its moment interval");
    require(result.primal.negativity <= 2.0e-12,
            label + " has a negative weight");
}

void testK2ThroughK10() {
    ProfileOptions options = strictOptions();
    /* The broad k sweep uses a realistic absolute likelihood certificate;
     * focused low-dimensional cells below exercise 1e-9. */
    options.gap_tolerance = 1.0e-6;
    for (std::size_t k = 2; k <= 10; ++k) {
        std::vector<double> diagonal(k * k, 0.0);
        std::vector<double> counts(k, 0.0);
        std::vector<double> rates(k, 0.0);
        double count_sum = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            diagonal[j * k + j] = 1.0;
            counts[j] = static_cast<double>(j + 1);
            count_sum += counts[j];
            rates[j] = 0.5 + 0.2 * static_cast<double>(j);
        }

        ProfileProblem quotient =
            makeProblem(k, k, diagonal, counts, rates);
        quotient.geometry =
            FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
        quotient.moment_lower = 0.25;
        quotient.moment_upper = 3.0;
        const ProfileResult quotient_result =
            freerate_profile::solve(quotient, options);
        checkCertified(quotient_result,
                       "quotient synthetic k=" + std::to_string(k),
                       1.1e-6);
        for (std::size_t j = 0; j < k; ++j) {
            require(closeEnough(quotient_result.weight[j],
                                counts[j] / count_sum, 1.0e-7),
                    "simplex analytic optimum mismatch for k=" +
                        std::to_string(k));
        }

        for (std::size_t j = 0; j < k; ++j) {
            rates[j] = (j % 2 == 0) ? 0.5 : 2.0;
        }
        ProfileProblem literal =
            makeProblem(k, k, diagonal, counts, rates);
        literal.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
        literal.target_moment = 1.0;
        const ProfileResult literal_result =
            freerate_profile::solve(literal, options);
        checkCertified(literal_result,
                       "literal synthetic k=" + std::to_string(k),
                       1.1e-6);
    }
}

void testVertexEnumeration() {
    ProfileProblem literal;
    literal.category_count = 3;
    literal.rate = std::vector<double>{0.5, 1.0, 2.0};
    literal.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    literal.target_moment = 1.0;
    const std::vector<std::vector<double> > literal_vertices =
        freerate_profile::enumerateFeasibleVertices(literal);
    require(literal_vertices.size() >= 2,
            "literal vertex enumeration missed a feasible vertex");

    ProfileProblem quotient = literal;
    quotient.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    quotient.moment_lower = 0.75;
    quotient.moment_upper = 1.25;
    const std::vector<std::vector<double> > quotient_vertices =
        freerate_profile::enumerateFeasibleVertices(quotient);
    require(quotient_vertices.size() >= 5,
            "quotient vertex enumeration missed a one/two-point vertex");
    for (std::size_t v = 0; v < quotient_vertices.size(); ++v) {
        double mass = 0.0;
        double moment = 0.0;
        std::size_t nonzero = 0;
        for (std::size_t j = 0; j < quotient.category_count; ++j) {
            mass += quotient_vertices[v][j];
            moment += quotient_vertices[v][j] * quotient.rate[j];
            if (quotient_vertices[v][j] > 1.0e-14) {
                ++nonzero;
            }
        }
        require(closeEnough(mass, 1.0, 1.0e-13),
                "enumerated quotient vertex has wrong mass");
        require(moment >= quotient.moment_lower - 1.0e-13 &&
                    moment <= quotient.moment_upper + 1.0e-13,
                "enumerated quotient vertex has infeasible moment");
        require(nonzero <= 2,
                "moment-polytope vertex has more than two atoms");
    }
}

void testNarrowIntervalExactOracle() {
    ProfileProblem problem = makeProblem(
        1, 2, std::vector<double>{1.0e-12, 1.0},
        std::vector<double>{1.0e12}, std::vector<double>{0.5, 2.0});
    problem.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    problem.moment_lower = 1.0;
    problem.moment_upper = 1.0 + 1.0e-14;

    const std::vector<std::vector<double> > vertices =
        freerate_profile::enumerateFeasibleVertices(problem);
    require(vertices.size() == 2,
            "a narrow nonzero interval must retain both boundary vertices");

    ProfileOptions options = strictOptions();
    options.primal_tolerance = 2.0e-13;
    const ProfileResult result = freerate_profile::solve(problem, options);
    if (!result.converged()) {
        std::cerr << "narrow debug reason="
                  << freerate_profile::exitReasonName(result.reason)
                  << " gap=" << result.frank_wolfe_gap
                  << " moment=" << result.moment
                  << " weights=" << result.weight[0] << ','
                  << result.weight[1] << '\n';
    }
    checkCertified(result, "narrow-interval high-gradient cell");
    require(result.moment_upper_active,
            "the exact oracle omitted the improving narrow upper boundary");
    require(result.weight[1] > vertices[0][1],
            "the high-gradient objective did not reach the retained upper vertex");
}

void testForcedLiteralR2() {
    ProfileProblem problem = makeProblem(
        2, 2, std::vector<double>{0.8, 0.3, 0.2, 0.7},
        std::vector<double>{7.0, 11.0}, std::vector<double>{0.5, 2.0});
    problem.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    problem.target_moment = 1.0;

    const ProfileResult result =
        freerate_profile::solve(problem, strictOptions());
    checkCertified(result, "forced literal R2");
    require(closeEnough(result.weight[0], 2.0 / 3.0, 2.0e-13) &&
                closeEnough(result.weight[1], 1.0 / 3.0, 2.0e-13),
            "literal R2 did not return the unique feasible weights");
    require(result.iterations == 0,
            "a zero-dimensional literal profile should not iterate");
}

void testUnconstrainedSimplexOptimum() {
    ProfileProblem problem = makeProblem(
        3, 3,
        std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0},
        std::vector<double>{2.0, 3.0, 5.0},
        std::vector<double>{0.5, 1.0, 2.0});
    problem.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    problem.moment_lower = 0.4;
    problem.moment_upper = 2.1;

    const ProfileResult result =
        freerate_profile::solve(problem, strictOptions());
    checkCertified(result, "unconstrained simplex");
    require(closeEnough(result.weight[0], 0.2, 2.0e-9) &&
                closeEnough(result.weight[1], 0.3, 2.0e-9) &&
                closeEnough(result.weight[2], 0.5, 2.0e-9),
            "simplex solution differs from the multinomial optimum");
    require(!result.moment_lower_active && !result.moment_upper_active,
            "an inactive quotient bound was reported active");
}

void testActiveQuotientBound() {
    ProfileProblem problem = makeProblem(
        3, 3,
        std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0},
        std::vector<double>{2.0, 3.0, 5.0},
        std::vector<double>{0.5, 1.0, 2.0});
    problem.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    problem.moment_lower = 0.6;
    problem.moment_upper = 1.0;

    const ProfileResult result =
        freerate_profile::solve(problem, strictOptions());
    checkCertified(result, "active quotient bound");
    require(result.moment_upper_active,
            "the binding upper moment constraint was not reported");
    require(closeEnough(result.moment, 1.0, 2.0e-12),
            "the quotient optimum is not on its required upper face");
    require(result.weight[0] > 0.0 && result.weight[1] > 0.0 &&
                result.weight[2] > 0.0,
            "the test's bound optimum unexpectedly lost an atom");
}

void testLiteralAgainstDenseReference() {
    ProfileProblem problem = makeProblem(
        2, 3, std::vector<double>{0.9, 0.6, 0.2, 0.1, 0.4, 0.8},
        std::vector<double>{7.0, 11.0},
        std::vector<double>{0.5, 1.0, 2.0});
    problem.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    problem.target_moment = 1.0;

    const ProfileResult result =
        freerate_profile::solve(problem, strictOptions());
    checkCertified(result, "literal dense-reference cell");

    double best_grid = -std::numeric_limits<double>::infinity();
    for (int step = 0; step <= 200000; ++step) {
        const double t = static_cast<double>(step) / 200000.0;
        const std::vector<double> weight{
            (1.0 - t) * (2.0 / 3.0), t,
            (1.0 - t) * (1.0 / 3.0)};
        best_grid =
            std::max(best_grid, directLogLikelihood(problem, weight));
    }
    require(result.log_likelihood + 2.0e-9 >= best_grid,
            "certified literal result is below a dense feasible search");
}

void testDuplicateColumns() {
    ProfileProblem problem = makeProblem(
        2, 4,
        std::vector<double>{0.8, 0.8, 0.3, 0.3,
                            0.2, 0.2, 0.7, 0.7},
        std::vector<double>{13.0, 17.0},
        std::vector<double>{0.5, 0.5, 2.0, 2.0});
    problem.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    problem.target_moment = 1.0;

    const ProfileResult result =
        freerate_profile::solve(problem, strictOptions());
    checkCertified(result, "duplicate-column literal cell");
    require(closeEnough(result.weight[0] + result.weight[1], 2.0 / 3.0,
                        2.0e-12) &&
                closeEnough(result.weight[2] + result.weight[3], 1.0 / 3.0,
                            2.0e-12),
            "duplicate columns broke the literal affine constraint");
}

void testZeroOptimalWeight() {
    ProfileProblem problem = makeProblem(
        2, 3, std::vector<double>{0.9, 0.1, 0.2, 0.1, 0.9, 0.05},
        std::vector<double>{10.0, 10.0},
        std::vector<double>{0.5, 1.0, 2.0});
    problem.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    problem.moment_lower = 0.4;
    problem.moment_upper = 2.1;

    const ProfileResult result =
        freerate_profile::solve(problem, strictOptions());
    checkCertified(result, "zero-weight boundary cell");
    require(result.weight[2] <= 1.0e-10,
            "a component dominated pointwise did not receive zero weight");
    require(result.active_weight_count == 2,
            "zero-weight component was reported active");
}

void testSuppliedStartAndScaling() {
    ProfileProblem base = makeProblem(
        2, 3, std::vector<double>{0.9, 0.5, 0.2, 0.1, 0.5, 0.8},
        std::vector<double>{4.0, 9.0},
        std::vector<double>{0.5, 1.0, 2.0});
    base.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    base.moment_lower = 0.4;
    base.moment_upper = 2.1;
    const std::vector<double> start{0.2, 0.3, 0.5};
    const ProfileResult reference =
        freerate_profile::solve(base, strictOptions(), start);
    checkCertified(reference, "supplied-start reference");
    require(reference.supplied_start_used,
            "a feasible finite supplied start was ignored");

    ProfileProblem scaled = base;
    for (std::size_t j = 0; j < 3; ++j) {
        scaled.component_likelihood[j] *= 1.0e180;
        scaled.component_likelihood[3 + j] *= 1.0e-180;
    }
    const ProfileResult scaled_result =
        freerate_profile::solve(scaled, strictOptions(), start);
    checkCertified(scaled_result, "row-scaled reference");
    for (std::size_t j = 0; j < 3; ++j) {
        require(closeEnough(reference.weight[j], scaled_result.weight[j],
                            2.0e-9),
                "common per-pattern scaling changed the fitted weights");
    }
    const double expected_shift = 4.0 * std::log(1.0e180) +
                                  9.0 * std::log(1.0e-180);
    require(closeEnough(scaled_result.log_likelihood -
                            reference.log_likelihood,
                        expected_shift, 2.0e-12),
            "common per-pattern scaling produced the wrong likelihood offset");

    ProfileProblem offset_scaled = base;
    offset_scaled.component_log_scale = std::vector<double>{1000.0, -1000.0};
    const ProfileResult offset_result =
        freerate_profile::solve(offset_scaled, strictOptions(), start);
    checkCertified(offset_result, "explicit-log-scale reference");
    for (std::size_t j = 0; j < 3; ++j) {
        require(closeEnough(reference.weight[j], offset_result.weight[j],
                            2.0e-9),
                "explicit common log scaling changed fitted weights");
    }
    const double offset_shift = 4.0 * 1000.0 + 9.0 * -1000.0;
    require(closeEnough(offset_result.log_likelihood -
                            reference.log_likelihood,
                        offset_shift, 2.0e-12),
            "explicit per-pattern log scale was not restored in the score");
}

void testTypedFailures() {
    ProfileProblem infeasible = makeProblem(
        1, 2, std::vector<double>{0.5, 0.5}, std::vector<double>{1.0},
        std::vector<double>{0.2, 0.8});
    infeasible.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    infeasible.target_moment = 1.0;
    require(freerate_profile::solve(infeasible).reason ==
                ExitReason::INFEASIBLE_POLYTOPE,
            "empty literal polytope did not return a typed failure");

    ProfileProblem invalid = infeasible;
    invalid.rate[0] = -0.2;
    require(freerate_profile::solve(invalid).reason ==
                ExitReason::INVALID_INPUT,
            "invalid rate did not return INVALID_INPUT");

    ProfileProblem impossible = infeasible;
    impossible.rate = std::vector<double>{0.5, 2.0};
    impossible.component_likelihood = std::vector<double>{0.0, 0.0};
    require(freerate_profile::solve(impossible).reason ==
                ExitReason::NONFINITE_OBJECTIVE,
            "an impossible positive-count pattern was not typed");

    ProfileProblem unrepresentable = makeProblem(
        1, 1, std::vector<double>{1.0}, std::vector<double>{1.0e308},
        std::vector<double>{1.0});
    unrepresentable.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    unrepresentable.target_moment = 1.0;
    unrepresentable.component_log_scale = std::vector<double>{10.0};
    require(freerate_profile::solve(unrepresentable).reason ==
                ExitReason::NONFINITE_OBJECTIVE,
            "a total score outside the result type was certified");
}

/*
 * F1/F2/F3 regression (adversarial review, 2026-07-22).
 *
 * F1: two-point vertices were built as w_i = (r_j - t)/(r_j - r_i), w_j = 1 - w_i. Subtracting from one
 * cancels catastrophically for a lopsided bracket, and the acceptance tolerance scaled only with the
 * target, never with the rates. Genuine vertices were therefore dropped once max(rate) exceeded the
 * target by enough orders; the Frank-Wolfe maximum was then taken over a strict subset of the true
 * vertex set, the reported gap stopped bounding l* - l(w), and a false global certificate was observed.
 *
 * F2: a materially negative directional maximum -- the direct proof that the point lies outside the
 * enumerated hull -- was clamped to 0.0, which is the strongest certificate value there is.
 *
 * F3: converged() is decided against the internal gap_tolerance, so gating on it discards fits whose
 * achieved gap already satisfies a looser caller threshold.
 */
void testWideRateSpanVertexCompleteness() {
    const double spans[] = {1.0e3, 1.0e5, 1.0e6, 1.0e7, 1.0e9};
    for (std::size_t s = 0; s < sizeof(spans) / sizeof(spans[0]); ++s) {
        ProfileProblem problem;
        problem.pattern_count = 2;
        problem.category_count = 3;
        problem.rate.clear();
        problem.rate.push_back(0.5);
        problem.rate.push_back(2.0);
        problem.rate.push_back(spans[s]);
        problem.multiplicity.assign(2, 1.0);
        problem.component_likelihood.assign(6, 0.5);
        problem.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
        problem.target_moment = 1.0;

        // The pairs bracketing 1.0 are (0.5, 2.0) and (0.5, span): exactly two vertices, at every span.
        const std::vector<std::vector<double> > vertices =
            freerate_profile::enumerateFeasibleVertices(problem);
        require(vertices.size() == 2,
                "a feasible two-point vertex was dropped at a wide rate span");

        for (std::size_t v = 0; v < vertices.size(); ++v) {
            long double mass = 0.0L;
            long double moment = 0.0L;
            for (std::size_t j = 0; j < problem.category_count; ++j) {
                require(vertices[v][j] >= 0.0, "negative weight in an enumerated vertex");
                mass += (long double)vertices[v][j];
                moment += (long double)vertices[v][j] * (long double)problem.rate[j];
            }
            require(fabsl(mass - 1.0L) < 1e-12, "vertex violates the mass constraint");
            require(fabsl(moment - 1.0L) < 1e-9, "vertex violates the moment constraint");
        }
    }
}

void testSignedGapIsPublishedAndGatesCorrectly() {
    ProfileProblem problem;
    problem.pattern_count = 3;
    problem.category_count = 3;
    problem.rate.clear();
    problem.rate.push_back(0.25);
    problem.rate.push_back(1.0);
    problem.rate.push_back(4.0);
    problem.multiplicity.assign(3, 7.0);
    problem.component_likelihood.clear();
    for (std::size_t p = 0; p < 3; ++p) {
        problem.component_likelihood.push_back(0.2 + 0.1 * (double)p);
        problem.component_likelihood.push_back(0.5);
        problem.component_likelihood.push_back(0.9 - 0.2 * (double)p);
    }
    problem.geometry = FeasibleGeometry::LITERAL_MASS_MEAN;
    problem.target_moment = 1.0;

    const ProfileResult result = freerate_profile::solve(problem);

    // The extra field must not drift from the quantity it explains.
    require(result.gap_is_valid_bound, "a well-posed solve reported an invalid gap bound");
    require(result.signed_directional_gap >= -1e-12,
            "signed gap is materially negative on a well-posed problem");
    require(std::fabs(std::max(0.0, result.signed_directional_gap) -
                      result.frank_wolfe_gap) <= 1e-15,
            "signed gap and clamped gap disagree");

    require(result.gapCertifies(1e-5),
            "gapCertifies rejected a point whose achieved gap satisfies the threshold");

    ProfileResult broken = result;
    broken.gap_is_valid_bound = false;
    require(!broken.gapCertifies(1e-5),
            "gapCertifies accepted a point whose gap is not a valid bound");
}

/**
 * bestGapBound() must not pass off a face-local second-order bound as a global one.
 *
 * The Newton decrement is computed in the tangent space of the categories carrying nonzero weight, so it
 * bounds l*_face - l(w). When a category sits at zero weight, an improving feasible direction can exist
 * that activates it, and the decrement is structurally blind to that move. Taking min(FW, Newton)
 * unconditionally therefore reports a bound orders of magnitude below the true shortfall -- the same
 * class of error as F1, a number that does not bound the quantity it names.
 */
void testFaceLocalBoundIsNotGlobal() {
    ProfileProblem problem = makeProblem(
        2, 3, std::vector<double>{0.9, 0.1, 0.2, 0.1, 0.9, 0.05},
        std::vector<double>{10.0, 10.0},
        std::vector<double>{0.5, 1.0, 2.0});
    problem.geometry = FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    problem.moment_lower = 0.4;
    problem.moment_upper = 2.1;
    const ProfileResult solved = freerate_profile::solve(problem, strictOptions());

    // Guard against the fix degenerating into "always refuse", which would silently disable the
    // second-order certificate everywhere instead of only where it is unsound. At a genuine optimum the
    // dominated category must price out, so the face bound IS global here even though a weight is zero.
    require(solved.active_weight_count == 2,
            "test precondition lost: expected one zero-weight category");
    require(solved.newton_bound_is_global,
            "a zero-weight category at the true optimum failed to price out, so the second-order "
            "certificate is being refused even when it is sound");
    require(solved.max_inactive_reduced_cost <= solved.gap_noise_floor,
            "inactive reduced cost is positive at a certified optimum");

    // Now the unsound configuration: a tight face bound beside a loose but valid global one.
    ProfileResult facelocal = solved;
    facelocal.gap_is_valid_bound = true;
    facelocal.frank_wolfe_gap = 1e-3;
    facelocal.newton_gap_bound = 1e-18;
    facelocal.newton_bound_is_global = false;
    require(facelocal.bestGapBound() == 1e-3,
            "bestGapBound published a face-local second-order bound as a global bound");
    require(!facelocal.certifiesTo(1e-5),
            "certifiesTo certified a point using a bound that does not cover off-face directions");

    // The same numbers with a globalised bound must certify, or the certificate is worthless.
    ProfileResult globalised = facelocal;
    globalised.newton_bound_is_global = true;
    require(globalised.bestGapBound() == 1e-18,
            "bestGapBound ignored a valid global second-order bound");
    require(globalised.certifiesTo(1e-5),
            "certifiesTo rejected a point carrying a valid global second-order bound");
}

} // namespace

int main() {
    testWideRateSpanVertexCompleteness();
    testSignedGapIsPublishedAndGatesCorrectly();
    testFaceLocalBoundIsNotGlobal();
    testVertexEnumeration();
    testK2ThroughK10();
    testNarrowIntervalExactOracle();
    testForcedLiteralR2();
    testUnconstrainedSimplexOptimum();
    testActiveQuotientBound();
    testLiteralAgainstDenseReference();
    testDuplicateColumns();
    testZeroOptimalWeight();
    testSuppliedStartAndScaling();
    testTypedFailures();
    std::cout << "freerateprofile_unit: all tests passed\n";
    return 0;
}
