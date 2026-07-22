/*
 * freerateprofile.cpp
 *
 * Convex fixed-column FreeRate weight profiling reference implementation.
 */

#include "freerateprofile.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace freerate_profile {

namespace {

const double kVertexTolerance = 2.0e-13;
const double kSolverActiveTolerance = 2.0e-14;

double quietNaN() {
    return std::numeric_limits<double>::quiet_NaN();
}

double scaleFor(double a, double b) {
    return std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

bool approximatelyEqual(double a, double b, double tolerance) {
    return std::fabs(a - b) <= tolerance * scaleFor(a, b);
}

bool geometryIsValid(FeasibleGeometry geometry) {
    return geometry == FeasibleGeometry::LITERAL_MASS_MEAN ||
           geometry == FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
}

bool affineInputIsValid(const ProfileProblem &problem) {
    if (problem.category_count == 0 ||
        problem.rate.size() != problem.category_count ||
        !geometryIsValid(problem.geometry)) {
        return false;
    }
    for (std::size_t j = 0; j < problem.category_count; ++j) {
        if (!std::isfinite(problem.rate[j]) || problem.rate[j] <= 0.0) {
            return false;
        }
    }
    if (problem.geometry == FeasibleGeometry::LITERAL_MASS_MEAN) {
        return std::isfinite(problem.target_moment) &&
               problem.target_moment > 0.0;
    }
    return std::isfinite(problem.moment_lower) &&
           std::isfinite(problem.moment_upper) &&
           problem.moment_lower > 0.0 &&
           problem.moment_lower <= problem.moment_upper;
}

double vectorMoment(const std::vector<double> &weight,
                    const std::vector<double> &rate) {
    long double value = 0.0L;
    for (std::size_t j = 0; j < weight.size(); ++j) {
        value += static_cast<long double>(weight[j]) * rate[j];
    }
    return static_cast<double>(value);
}

double vectorMass(const std::vector<double> &weight) {
    long double value = 0.0L;
    for (std::size_t j = 0; j < weight.size(); ++j) {
        value += weight[j];
    }
    return static_cast<double>(value);
}

PrimalResiduals residualsFor(const ProfileProblem &problem,
                             const std::vector<double> &weight) {
    PrimalResiduals residual;
    if (weight.size() != problem.category_count) {
        residual.mass = std::numeric_limits<double>::infinity();
        residual.literal_moment = std::numeric_limits<double>::infinity();
        residual.moment_lower_violation =
            std::numeric_limits<double>::infinity();
        residual.moment_upper_violation =
            std::numeric_limits<double>::infinity();
        residual.negativity = std::numeric_limits<double>::infinity();
        return residual;
    }

    const double mass = vectorMass(weight);
    const double moment = vectorMoment(weight, problem.rate);
    residual.mass = std::fabs(mass - 1.0);
    residual.negativity = 0.0;
    for (std::size_t j = 0; j < weight.size(); ++j) {
        residual.negativity = std::max(residual.negativity, -weight[j]);
    }
    residual.negativity = std::max(0.0, residual.negativity);

    if (problem.geometry == FeasibleGeometry::LITERAL_MASS_MEAN) {
        residual.literal_moment = std::fabs(moment - problem.target_moment);
        residual.moment_lower_violation = 0.0;
        residual.moment_upper_violation = 0.0;
    } else {
        residual.literal_moment = 0.0;
        residual.moment_lower_violation =
            std::max(0.0, problem.moment_lower - moment);
        residual.moment_upper_violation =
            std::max(0.0, moment - problem.moment_upper);
    }
    return residual;
}

double maximumResidual(const PrimalResiduals &residual) {
    return std::max(
        std::max(residual.mass, residual.literal_moment),
        std::max(std::max(residual.moment_lower_violation,
                          residual.moment_upper_violation),
                 residual.negativity));
}

void addVertex(const ProfileProblem &problem,
               const std::vector<double> &candidate,
               std::vector<std::vector<double> > *vertices) {
    const PrimalResiduals residual = residualsFor(problem, candidate);
    /* Bound this at the scale of the AFFINE TARGETS, never at the scale of the rates.
     *
     * It is tempting to widen this by max_j r_j on the theory that the moment residual
     * |sum_j w_j r_j - target| inherits the rates' magnitude. It does not. For any candidate with
     * w >= 0 and r > 0 whose moment is the target, every term w_j*r_j lies in [0, target], so the sum
     * accumulates no cancellation and its rounding error is bounded by roughly k*eps*target. Measured
     * on the vertices this file actually builds, the worst residual is 5.55e-17 at EVERY rate span from
     * 1e2 to 1e15 -- flat, and tracking the target rather than the rates, refuting the rate theory by up
     * to twelve orders of magnitude.
     *
     * Widening by the rate scale would therefore buy nothing and cost a great deal: maximumResidual()
     * maxes over mass and NEGATIVITY as well as the moment, so a rate-driven bound loosens two checks
     * that have no connection to rate conditioning. At max_j r_j = 1e15 the gate would admit six nats of
     * constraint violation. The genuine defect here was the catastrophic cancellation in the old
     * `w_j = 1 - w_i` vertex construction, which is fixed at its source in
     * addMomentBoundaryVertices(). */
    const double affine_scale =
        problem.geometry == FeasibleGeometry::LITERAL_MASS_MEAN
            ? scaleFor(problem.target_moment, 1.0)
            : scaleFor(problem.moment_upper, problem.moment_lower);
    if (maximumResidual(residual) > 32.0 * kVertexTolerance * affine_scale) {
        return;
    }
    vertices->push_back(candidate);
}

void addMomentBoundaryVertices(
    const ProfileProblem &problem,
    double boundary,
    std::vector<std::vector<double> > *vertices) {
    const std::size_t k = problem.category_count;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            const double ri = problem.rate[i];
            const double rj = problem.rate[j];
            const double difference = rj - ri;
            if (difference == 0.0) {
                continue;
            }
            const double lo = std::min(ri, rj);
            const double hi = std::max(ri, rj);
            /* A nearly feasible endpoint is not an LP vertex. Keep this
             * comparison literal; the computed two-point vertex itself is
             * checked with a floating-point residual below. */
            if (boundary < lo || boundary > hi) {
                continue;
            }

            /* Solve for BOTH weights directly as ratios of differences.
             *
             * The previous form computed one weight and derived the other as 1.0 - w. That subtraction
             * is a catastrophic cancellation exactly when the pair straddles the boundary asymmetrically:
             * with hi >> boundary the first weight sits within ~boundary/hi of 1, so 1.0 - w keeps only
             * the few bits that differ and the reformed moment drifts by roughly eps*hi. For rates around
             * 1e6 against the FreeRate unit-mean gauge that drift exceeded the acceptance bound above,
             * and the vertex -- exactly representable and feasible to ~1e-17 -- was discarded.
             *
             * Neither expression below subtracts nearly equal quantities of similar magnitude from 1, so
             * both weights keep full relative precision however lopsided the bracket is. */
            const double span = hi - lo;
            const double weight_on_hi = (boundary - lo) / span;
            const double weight_on_lo = (hi - boundary) / span;
            double wi = (ri <= rj) ? weight_on_lo : weight_on_hi;
            double wj = (ri <= rj) ? weight_on_hi : weight_on_lo;
            if (wi < 0.0 && wi > -32.0 * kVertexTolerance) {
                wi = 0.0;
            }
            if (wj < 0.0 && wj > -32.0 * kVertexTolerance) {
                wj = 0.0;
            }
            if (wi < 0.0 || wj < 0.0) {
                continue;
            }
            std::vector<double> vertex(k, 0.0);
            vertex[i] = wi;
            vertex[j] = wj;
            addVertex(problem, vertex, vertices);
        }
    }
}

struct PreparedProblem {
    const ProfileProblem *source;
    std::vector<double> component;
    std::vector<double> log_scale;
};

enum class PreparationStatus {
    OK,
    INVALID,
    IMPOSSIBLE_PATTERN
};

PreparationStatus prepareProblem(const ProfileProblem &problem,
                                 PreparedProblem *prepared) {
    if (!affineInputIsValid(problem) || problem.pattern_count == 0 ||
        problem.multiplicity.size() != problem.pattern_count ||
        problem.pattern_count >
            std::numeric_limits<std::size_t>::max() / problem.category_count ||
        problem.component_likelihood.size() !=
            problem.pattern_count * problem.category_count ||
        (!problem.component_log_scale.empty() &&
         problem.component_log_scale.size() != problem.pattern_count)) {
        return PreparationStatus::INVALID;
    }

    prepared->source = &problem;
    prepared->component.assign(problem.component_likelihood.size(), 0.0);
    prepared->log_scale.assign(problem.pattern_count, 0.0);

    for (std::size_t p = 0; p < problem.pattern_count; ++p) {
        const double count = problem.multiplicity[p];
        if (!std::isfinite(count) || count < 0.0) {
            return PreparationStatus::INVALID;
        }
        const double supplied_log_scale = problem.component_log_scale.empty()
                                              ? 0.0
                                              : problem.component_log_scale[p];
        if (!std::isfinite(supplied_log_scale)) {
            return PreparationStatus::INVALID;
        }
        double row_max = 0.0;
        for (std::size_t j = 0; j < problem.category_count; ++j) {
            const double value =
                problem.component_likelihood[p * problem.category_count + j];
            if (!std::isfinite(value) || value < 0.0) {
                return PreparationStatus::INVALID;
            }
            row_max = std::max(row_max, value);
        }
        if (row_max == 0.0) {
            if (count > 0.0) {
                return PreparationStatus::IMPOSSIBLE_PATTERN;
            }
            continue;
        }
        prepared->log_scale[p] =
            supplied_log_scale + std::log(row_max);
        for (std::size_t j = 0; j < problem.category_count; ++j) {
            prepared->component[p * problem.category_count + j] =
                problem.component_likelihood[p * problem.category_count + j] /
                row_max;
        }
    }
    return PreparationStatus::OK;
}

struct Evaluation {
    bool finite;
    long double variable_log_likelihood;
    long double total_log_likelihood;
    std::vector<double> gradient;
    std::vector<double> curvature;
    std::vector<double> mixture;

    Evaluation()
        : finite(false),
          variable_log_likelihood(0.0L),
          total_log_likelihood(0.0L) {}
};

Evaluation evaluate(const PreparedProblem &prepared,
                    const std::vector<double> &weight,
                    bool need_curvature,
                    std::size_t *evaluation_count) {
    const ProfileProblem &problem = *prepared.source;
    const std::size_t k = problem.category_count;
    Evaluation result;
    result.gradient.assign(k, 0.0);
    result.mixture.assign(problem.pattern_count, 0.0);
    if (need_curvature) {
        result.curvature.assign(k * k, 0.0);
    }
    if (evaluation_count != NULL) {
        ++(*evaluation_count);
    }

    std::vector<long double> gradient(k, 0.0L);
    std::vector<long double> curvature;
    if (need_curvature) {
        curvature.assign(k * k, 0.0L);
    }

    long double variable = 0.0L;
    long double offset = 0.0L;
    for (std::size_t p = 0; p < problem.pattern_count; ++p) {
        const double count = problem.multiplicity[p];
        if (count == 0.0) {
            continue;
        }
        long double mixture = 0.0L;
        for (std::size_t j = 0; j < k; ++j) {
            mixture += static_cast<long double>(weight[j]) *
                       prepared.component[p * k + j];
        }
        if (!(mixture > 0.0L) || !std::isfinite(mixture)) {
            return result;
        }
        result.mixture[p] = static_cast<double>(mixture);
        variable += static_cast<long double>(count) * std::log(mixture);
        offset += static_cast<long double>(count) * prepared.log_scale[p];

        const long double factor = static_cast<long double>(count) / mixture;
        for (std::size_t j = 0; j < k; ++j) {
            gradient[j] +=
                factor * prepared.component[p * k + j];
        }
        if (need_curvature) {
            const long double second =
                static_cast<long double>(count) / (mixture * mixture);
            for (std::size_t i = 0; i < k; ++i) {
                const long double fi = prepared.component[p * k + i];
                for (std::size_t j = 0; j < k; ++j) {
                    curvature[i * k + j] +=
                        second * fi * prepared.component[p * k + j];
                }
            }
        }
    }

    if (!std::isfinite(variable) || !std::isfinite(offset)) {
        return result;
    }
    for (std::size_t j = 0; j < k; ++j) {
        if (!std::isfinite(gradient[j]) ||
            std::fabs(gradient[j]) >
                static_cast<long double>(std::numeric_limits<double>::max())) {
            return result;
        }
        result.gradient[j] = static_cast<double>(gradient[j]);
    }
    if (need_curvature) {
        for (std::size_t i = 0; i < k * k; ++i) {
            if (!std::isfinite(curvature[i]) ||
                std::fabs(curvature[i]) > static_cast<long double>(
                                               std::numeric_limits<double>::max())) {
                return result;
            }
            result.curvature[i] = static_cast<double>(curvature[i]);
        }
    }

    result.finite = true;
    result.variable_log_likelihood = variable;
    result.total_log_likelihood = variable + offset;
    return result;
}

struct GapResult {
    bool valid;
    double gap;
    std::size_t best_vertex;

    /**
     * Resolution floor of `gap`, in the same units (nats).
     *
     * The gap is a difference of two directional scores whose own magnitude is sum_p multiplicity_p:
     * with grad_j = sum_p n_p F_pj / s_p, the identity sum_j w_j grad_j = sum_p n_p holds exactly. So
     * ordinary rounding perturbs the gap by about eps * sum_p n_p, and on a large alignment that is
     * nowhere near zero -- at 1e8 sites it is ~1e-8 nats. A gap below this floor carries no
     * information about optimality in either direction.
     */
    double noise;

    GapResult()
        : valid(false), gap(quietNaN()), best_vertex(0), noise(quietNaN()) {}
};

bool likelihoodFitsResultType(const Evaluation &evaluation) {
    return evaluation.finite &&
           std::isfinite(evaluation.total_log_likelihood) &&
           std::fabs(evaluation.total_log_likelihood) <=
               static_cast<long double>(std::numeric_limits<double>::max());
}

GapResult computeGap(const std::vector<double> &weight,
                     const std::vector<double> &gradient,
                     const std::vector<std::vector<double> > &vertices) {
    GapResult result;
    if (vertices.empty() || weight.size() != gradient.size()) {
        return result;
    }
    long double current_score = 0.0L;
    for (std::size_t j = 0; j < weight.size(); ++j) {
        current_score += static_cast<long double>(gradient[j]) * weight[j];
    }

    long double best_score = -std::numeric_limits<long double>::infinity();
    for (std::size_t v = 0; v < vertices.size(); ++v) {
        long double score = 0.0L;
        for (std::size_t j = 0; j < weight.size(); ++j) {
            score += static_cast<long double>(gradient[j]) * vertices[v][j];
        }
        if (score > best_score) {
            best_score = score;
            result.best_vertex = v;
        }
    }
    const long double gap = best_score - current_score;
    if (!std::isfinite(gap) ||
        std::fabs(gap) >
            static_cast<long double>(std::numeric_limits<double>::max())) {
        return result;
    }
    result.valid = true;
    result.gap = static_cast<double>(gap);
    /* Scale the floor by the magnitude of the scores actually differenced, not by any fixed constant.
     * Each of the k products contributes about one ulp of the running score, and the gradient itself
     * arrives already rounded to double, so a small multiple of eps*|score| is the honest resolution.
     * The 1.0 floor keeps the bound meaningful on trivially small problems. */
    long double magnitude = std::fabs(current_score);
    if (std::isfinite(static_cast<double>(best_score))) {
        magnitude = std::max(magnitude, fabsl(best_score));
    }
    magnitude = std::max(magnitude, 1.0L);
    result.noise = static_cast<double>(
        32.0L * static_cast<long double>(
                    std::numeric_limits<double>::epsilon()) * magnitude);
    return result;
}

bool solveDenseLinearSystem(std::vector<double> matrix,
                            std::vector<double> rhs,
                            std::vector<double> *solution) {
    const std::size_t n = rhs.size();
    if (matrix.size() != n * n || n == 0) {
        return false;
    }
    double largest = 0.0;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        largest = std::max(largest, std::fabs(matrix[i]));
    }
    const double pivot_tolerance =
        128.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, largest);

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        double pivot_size = std::fabs(matrix[column * n + column]);
        for (std::size_t row = column + 1; row < n; ++row) {
            const double candidate = std::fabs(matrix[row * n + column]);
            if (candidate > pivot_size) {
                pivot = row;
                pivot_size = candidate;
            }
        }
        if (!(pivot_size > pivot_tolerance)) {
            return false;
        }
        if (pivot != column) {
            for (std::size_t j = column; j < n; ++j) {
                std::swap(matrix[column * n + j], matrix[pivot * n + j]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        const double diagonal = matrix[column * n + column];
        for (std::size_t row = column + 1; row < n; ++row) {
            const double multiplier = matrix[row * n + column] / diagonal;
            matrix[row * n + column] = 0.0;
            for (std::size_t j = column + 1; j < n; ++j) {
                matrix[row * n + j] -=
                    multiplier * matrix[column * n + j];
            }
            rhs[row] -= multiplier * rhs[column];
        }
    }

    solution->assign(n, 0.0);
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const std::size_t row = n - reverse - 1;
        long double value = rhs[row];
        for (std::size_t j = row + 1; j < n; ++j) {
            value -= static_cast<long double>(matrix[row * n + j]) *
                     (*solution)[j];
        }
        const double diagonal = matrix[row * n + row];
        if (!(std::fabs(diagonal) > pivot_tolerance)) {
            return false;
        }
        (*solution)[row] = static_cast<double>(value / diagonal);
        if (!std::isfinite((*solution)[row])) {
            return false;
        }
    }
    return true;
}

void projectDirectionToEqualities(const ProfileProblem &problem,
                                  const std::vector<std::size_t> &active,
                                  bool include_moment,
                                  std::vector<double> *direction) {
    if (active.empty()) {
        return;
    }
    long double sum_d = 0.0L;
    long double sum_rd = 0.0L;
    long double sum_r = 0.0L;
    long double sum_r2 = 0.0L;
    for (std::size_t a = 0; a < active.size(); ++a) {
        const std::size_t j = active[a];
        sum_d += (*direction)[j];
        sum_rd += static_cast<long double>(problem.rate[j]) * (*direction)[j];
        sum_r += problem.rate[j];
        sum_r2 += static_cast<long double>(problem.rate[j]) * problem.rate[j];
    }

    const long double count = static_cast<long double>(active.size());
    if (!include_moment) {
        const double correction = static_cast<double>(sum_d / count);
        for (std::size_t a = 0; a < active.size(); ++a) {
            (*direction)[active[a]] -= correction;
        }
        return;
    }

    const long double determinant = count * sum_r2 - sum_r * sum_r;
    const long double determinant_scale =
        std::max(1.0L, std::fabs(count * sum_r2));
    if (std::fabs(determinant) <=
        256.0L * std::numeric_limits<double>::epsilon() * determinant_scale) {
        const double correction = static_cast<double>(sum_d / count);
        for (std::size_t a = 0; a < active.size(); ++a) {
            (*direction)[active[a]] -= correction;
        }
        return;
    }
    const long double lambda_mass =
        (sum_r2 * sum_d - sum_r * sum_rd) / determinant;
    const long double lambda_moment =
        (-sum_r * sum_d + count * sum_rd) / determinant;
    for (std::size_t a = 0; a < active.size(); ++a) {
        const std::size_t j = active[a];
        (*direction)[j] -= static_cast<double>(
            lambda_mass + lambda_moment * problem.rate[j]);
    }
}

bool buildNewtonDirection(const ProfileProblem &problem,
                          const ProfileOptions &options,
                          const std::vector<double> &weight,
                          const Evaluation &evaluation,
                          bool force_moment_tangent,
                          std::vector<double> *direction) {
    const std::size_t k = problem.category_count;
    std::vector<std::size_t> active;
    for (std::size_t j = 0; j < k; ++j) {
        if (weight[j] > kSolverActiveTolerance) {
            active.push_back(j);
        }
    }
    if (active.size() < 2) {
        return false;
    }

    bool include_moment =
        problem.geometry == FeasibleGeometry::LITERAL_MASS_MEAN ||
        force_moment_tangent;
    if (include_moment) {
        double minimum_rate = problem.rate[active[0]];
        double maximum_rate = minimum_rate;
        for (std::size_t a = 1; a < active.size(); ++a) {
            minimum_rate = std::min(minimum_rate, problem.rate[active[a]]);
            maximum_rate = std::max(maximum_rate, problem.rate[active[a]]);
        }
        if (approximatelyEqual(minimum_rate, maximum_rate,
                               kVertexTolerance)) {
            include_moment = false;
        }
    }

    const std::size_t equality_count = include_moment ? 2 : 1;
    if (active.size() <= equality_count) {
        return false;
    }
    const std::size_t dimension = active.size() + equality_count;

    double curvature_scale = 0.0;
    for (std::size_t a = 0; a < active.size(); ++a) {
        curvature_scale = std::max(
            curvature_scale,
            std::fabs(evaluation.curvature[active[a] * k + active[a]]));
    }
    curvature_scale = std::max(1.0, curvature_scale);

    const double ridge_factors[] = {0.0, 1.0e-14, 1.0e-12, 1.0e-10,
                                    1.0e-8, 1.0e-6};
    for (std::size_t attempt = 0;
         attempt < sizeof(ridge_factors) / sizeof(ridge_factors[0]);
         ++attempt) {
        std::vector<double> matrix(dimension * dimension, 0.0);
        std::vector<double> rhs(dimension, 0.0);
        for (std::size_t a = 0; a < active.size(); ++a) {
            rhs[a] = evaluation.gradient[active[a]];
            for (std::size_t b = 0; b < active.size(); ++b) {
                matrix[a * dimension + b] =
                    evaluation.curvature[active[a] * k + active[b]];
            }
            matrix[a * dimension + a] +=
                ridge_factors[attempt] * curvature_scale;
            matrix[a * dimension + active.size()] = 1.0;
            matrix[active.size() * dimension + a] = 1.0;
            if (include_moment) {
                matrix[a * dimension + active.size() + 1] =
                    problem.rate[active[a]];
                matrix[(active.size() + 1) * dimension + a] =
                    problem.rate[active[a]];
            }
        }

        std::vector<double> solution;
        if (!solveDenseLinearSystem(matrix, rhs, &solution)) {
            continue;
        }
        direction->assign(k, 0.0);
        for (std::size_t a = 0; a < active.size(); ++a) {
            (*direction)[active[a]] = solution[a];
        }
        projectDirectionToEqualities(problem, active, include_moment,
                                     direction);

        long double directional_derivative = 0.0L;
        long double norm = 0.0L;
        for (std::size_t j = 0; j < k; ++j) {
            directional_derivative +=
                static_cast<long double>(evaluation.gradient[j]) *
                (*direction)[j];
            norm += static_cast<long double>((*direction)[j]) *
                    (*direction)[j];
        }
        const long double derivative_floor =
            64.0L * std::numeric_limits<double>::epsilon() *
            std::max(1.0L, std::sqrt(norm));
        if (std::isfinite(directional_derivative) &&
            directional_derivative > derivative_floor) {
            return true;
        }
    }
    (void)options;
    return false;
}

double feasibleStepMaximum(const ProfileProblem &problem,
                           const std::vector<double> &weight,
                           const std::vector<double> &direction,
                           double requested_maximum) {
    double maximum = requested_maximum;
    for (std::size_t j = 0; j < weight.size(); ++j) {
        if (direction[j] < 0.0) {
            maximum = std::min(maximum, -weight[j] / direction[j]);
        }
    }
    if (problem.geometry == FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL) {
        const double moment = vectorMoment(weight, problem.rate);
        const double moment_direction =
            vectorMoment(direction, problem.rate);
        if (moment_direction < 0.0) {
            maximum = std::min(
                maximum,
                (moment - problem.moment_lower) / (-moment_direction));
        } else if (moment_direction > 0.0) {
            maximum = std::min(
                maximum,
                (problem.moment_upper - moment) / moment_direction);
        }
    }
    if (!std::isfinite(maximum)) {
        return 0.0;
    }
    return std::max(0.0, maximum);
}

struct LineCandidate {
    bool valid;
    std::vector<double> weight;
    Evaluation evaluation;

    LineCandidate() : valid(false) {}
};

long double lineDerivative(const PreparedProblem &prepared,
                           const std::vector<double> &base_mixture,
                           const std::vector<double> &mixture_direction,
                           double alpha,
                           bool *finite,
                           std::size_t *evaluation_count) {
    if (evaluation_count != NULL) {
        ++(*evaluation_count);
    }
    const ProfileProblem &problem = *prepared.source;
    long double derivative = 0.0L;
    *finite = true;
    for (std::size_t p = 0; p < problem.pattern_count; ++p) {
        if (problem.multiplicity[p] == 0.0) {
            continue;
        }
        const long double mixture =
            static_cast<long double>(base_mixture[p]) +
            static_cast<long double>(alpha) * mixture_direction[p];
        if (!(mixture > 0.0L) || !std::isfinite(mixture)) {
            *finite = false;
            return -std::numeric_limits<long double>::infinity();
        }
        derivative += static_cast<long double>(problem.multiplicity[p]) *
                      mixture_direction[p] / mixture;
    }
    if (!std::isfinite(derivative)) {
        *finite = false;
    }
    return derivative;
}

LineCandidate maximizeOnLine(const PreparedProblem &prepared,
                             const ProfileOptions &options,
                             const std::vector<double> &weight,
                             const Evaluation &current,
                             const std::vector<double> &direction,
                             double maximum_alpha,
                             bool endpoint_is_known_feasible,
                             std::size_t *evaluation_count) {
    LineCandidate candidate;
    const ProfileProblem &problem = *prepared.source;
    const std::size_t k = problem.category_count;
    if (!endpoint_is_known_feasible) {
        maximum_alpha =
            feasibleStepMaximum(problem, weight, direction, maximum_alpha);
    }
    if (!(maximum_alpha > 0.0)) {
        return candidate;
    }

    std::vector<double> mixture_direction(problem.pattern_count, 0.0);
    for (std::size_t p = 0; p < problem.pattern_count; ++p) {
        long double value = 0.0L;
        for (std::size_t j = 0; j < k; ++j) {
            value += static_cast<long double>(direction[j]) *
                     prepared.component[p * k + j];
        }
        mixture_direction[p] = static_cast<double>(value);
    }

    bool finite_at_zero = false;
    const long double derivative_at_zero =
        lineDerivative(prepared, current.mixture, mixture_direction, 0.0,
                       &finite_at_zero, evaluation_count);
    const long double derivative_floor =
        64.0L * std::numeric_limits<double>::epsilon() *
        std::max(1.0L, std::fabs(derivative_at_zero));
    if (!finite_at_zero || !(derivative_at_zero > derivative_floor)) {
        return candidate;
    }

    bool finite_at_maximum = false;
    const long double derivative_at_maximum =
        lineDerivative(prepared, current.mixture, mixture_direction,
                       maximum_alpha, &finite_at_maximum, evaluation_count);
    double alpha = maximum_alpha;
    if (!finite_at_maximum || derivative_at_maximum < 0.0L) {
        double lower = 0.0;
        double upper = maximum_alpha;
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double midpoint = lower + 0.5 * (upper - lower);
            bool midpoint_finite = false;
            const long double derivative =
                lineDerivative(prepared, current.mixture, mixture_direction,
                               midpoint, &midpoint_finite, evaluation_count);
            if (midpoint_finite && derivative > 0.0L) {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
            if (upper - lower <=
                8.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, maximum_alpha)) {
                break;
            }
        }
        alpha = lower + 0.5 * (upper - lower);
    }
    if (!(alpha > 0.0)) {
        return candidate;
    }

    candidate.weight.resize(k);
    for (std::size_t j = 0; j < k; ++j) {
        candidate.weight[j] = weight[j] + alpha * direction[j];
        if (candidate.weight[j] < 0.0 &&
            candidate.weight[j] > -128.0 *
                                      std::numeric_limits<double>::epsilon()) {
            candidate.weight[j] = 0.0;
        }
    }
    const PrimalResiduals residual = residualsFor(problem, candidate.weight);
    if (maximumResidual(residual) > 16.0 * options.primal_tolerance) {
        return LineCandidate();
    }
    candidate.evaluation =
        evaluate(prepared, candidate.weight, false, evaluation_count);
    if (!candidate.evaluation.finite ||
        candidate.evaluation.variable_log_likelihood +
                32.0L * std::numeric_limits<long double>::epsilon() *
                    std::max(1.0L,
                             std::fabs(current.variable_log_likelihood)) <
            current.variable_log_likelihood) {
        return LineCandidate();
    }
    candidate.valid = true;
    return candidate;
}

std::vector<double> vertexCentroid(
    const std::vector<std::vector<double> > &vertices) {
    std::vector<double> centroid(vertices[0].size(), 0.0);
    const double scale = 1.0 / static_cast<double>(vertices.size());
    for (std::size_t v = 0; v < vertices.size(); ++v) {
        for (std::size_t j = 0; j < centroid.size(); ++j) {
            centroid[j] += scale * vertices[v][j];
        }
    }
    return centroid;
}

/**
 * Newton decrement on the active face, reduced onto the tangent space of the active equality
 * constraints, and the self-concordant bound it yields.
 *
 * Method. Take the active atoms A (weight above the active tolerance). Build the constraint rows that
 * genuinely bind on A: the mass row, plus the moment row when the geometry pins it (literal) or a moment
 * bound is active (quotient). Form an orthonormal basis Z of the null space of those rows restricted to
 * A by Gram-Schmidt on the coordinate directions -- k <= 10 here, so an explicit basis is cheap and
 * avoids pulling in a factorisation dependency. Then
 *     lambda^2 = gr' * Hr^-1 * gr,   gr = Z'g|A,   Hr = Z'(-H)|A Z
 * solved by Cholesky, since -H is positive semi-definite by concavity.
 *
 * Zero-weight atoms are EXCLUDED. Including them would ask the quadratic model about directions the
 * feasible set forbids and inflate the decrement; their first-order pricing is the Frank-Wolfe gap's job.
 * So this bound covers the face, and the Frank-Wolfe gap covers leaving it -- which is exactly why the
 * caller should take the min of the two rather than either alone.
 */
// Relative floor on det(normal matrix)/(s0*s2) below which the two constraint rows are too close to
// parallel for the multiplier solve to be trusted. det is the SQUARE of the row conditioning, so this
// corresponds to a rate spread of roughly 1e-3 -- the point at which the measured multiplier error first
// exceeded the acceptance tolerance.
const double FREERATE_DUAL_CONDITION_FLOOR = 1e-6;

/** Smallest pattern multiplicity in the problem; +inf when there are none. */
double minMultiplicity(const ProfileProblem &problem) {
    double m = std::numeric_limits<double>::infinity();
    for (std::size_t p = 0; p < problem.multiplicity.size(); ++p) {
        if (problem.multiplicity[p] < m) m = problem.multiplicity[p];
    }
    return m;
}

struct NewtonCertificate {
    bool valid;
    double decrement;
    double bound;

    /**
     * True only when `bound` may be read as a bound on l* - l(w) over the WHOLE feasible set.
     *
     * The decrement is computed in the tangent space of the ACTIVE face, i.e. over the categories
     * carrying nonzero weight. That makes `bound` a bound on l*_face - l(w), where l*_face maximises
     * only over points keeping every zero-weight category at zero. If some category sits at w_j = 0, a
     * feasible improving direction can leave the face by activating it, and the decrement is structurally
     * blind to that move -- it never enters the reduced system. Publishing the face bound as a global one
     * would be the F1 error again: a number that does not bound the quantity it names.
     *
     * The missing test is dual feasibility on the inactive coordinates. At a face-stationary point the
     * active gradient is spanned by the binding constraint rows, g_i = mu + nu*r_i; the first-order value
     * of activating an inactive j is its reduced cost d_j = g_j - mu - nu*r_j. If every d_j <= 0 then no
     * feasible direction improves, and for a CONCAVE objective that is sufficient for global optimality,
     * so the face bound is then a global bound as well.
     */
    bool global;
    /** max over inactive categories of the reduced cost d_j; -inf when every category is active. */
    double max_inactive_reduced_cost;

    NewtonCertificate()
        : valid(false), decrement(quietNaN()), bound(quietNaN()),
          global(false),
          max_inactive_reduced_cost(-std::numeric_limits<double>::infinity()) {}
};

NewtonCertificate computeNewtonCertificate(const ProfileProblem &problem,
                                           const ProfileOptions &options,
                                           const std::vector<double> &weight,
                                           const Evaluation &current,
                                           bool moment_lower_active,
                                           bool moment_upper_active,
                                           double min_multiplicity) {
    NewtonCertificate result;
    const std::size_t k = problem.category_count;
    if (current.curvature.size() != k * k || current.gradient.size() != k) {
        return result;
    }

    std::vector<std::size_t> active;
    for (std::size_t j = 0; j < k; ++j) {
        if (weight[j] > options.active_weight_tolerance) {
            active.push_back(j);
        }
    }
    const std::size_t n = active.size();
    if (n == 0) {
        return result;
    }

    // Self-concordance of sum_p n_p log(s_p) survives positive scaling only for n_p >= 1; below that the
    // parameter degrades as 2/sqrt(min n_p) and omega*(lambda) STOPS being a bound. Measured: at
    // n_p = 0.01 the published bound was exceeded on 86,841 of 120,000 probes, worst case by 6.3x. The
    // multiplicity is a double fed straight from ptn_freq, and prepareProblem only rejects negatives, so
    // withhold the second-order bound rather than publish an invalid one. The Frank-Wolfe gap is
    // unaffected and still certifies.
    if (min_multiplicity < 1.0) {
        return result;
    }

    // Constraint rows that bind on the active face.
    std::vector<std::vector<double> > rows;
    rows.push_back(std::vector<double>(n, 1.0));                 // mass always binds
    const bool moment_inequality_active =
        problem.geometry != FeasibleGeometry::LITERAL_MASS_MEAN &&
        (moment_lower_active || moment_upper_active);
    const bool moment_binds =
        problem.geometry == FeasibleGeometry::LITERAL_MASS_MEAN ||
        moment_lower_active || moment_upper_active;
    if (moment_binds) {
        std::vector<double> row(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) row[i] = problem.rate[active[i]];
        rows.push_back(row);
    }

    // ---- Dual feasibility on the inactive coordinates: does the face bound also bound globally? ----
    //
    // Computed BEFORE any early return, because the pinned-face path (m == 0) reports bound = 0 and is
    // exactly the case where an off-face move is the only thing left that could improve.
    //
    // Least-squares multipliers over the ACTIVE coordinates: g_i ~= mu * 1 + nu * r_i. At a face-
    // stationary point the residual of this fit is the reduced gradient, which the decrement already
    // measures, so solving it here does not assume stationarity -- it just prices the constraints.
    {
        std::vector<std::size_t> inactive;
        for (std::size_t j = 0; j < k; ++j) {
            if (!(weight[j] > options.active_weight_tolerance)) inactive.push_back(j);
        }
        long double s0 = static_cast<long double>(n), s1 = 0.0L, s2 = 0.0L;
        long double gs = 0.0L, gr_ = 0.0L, gmax = 0.0L, rmax = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            const long double r = problem.rate[active[i]];
            const long double g = current.gradient[active[i]];
            s1 += r; s2 += r * r; gs += g; gr_ += g * r;
            if (std::fabs(g) > gmax) gmax = std::fabs(g);
            if (std::fabs(r) > rmax) rmax = std::fabs(r);
        }

        // Multipliers for g_i ~= mu + nu*r_i over the active coordinates.
        //
        // CONDITIONING. det = n^2 * Var(active rates), so it is the SQUARE of the constraint
        // conditioning and collapses long before any underflow. The former guard admitted the solve
        // unless |det| underflowed entirely, and the resulting multipliers are cancellation noise:
        // at an active-rate separation of 1e-9 they come back as exact powers of two, with the reduced
        // cost wrong in SIGN -- a real escape direction reported as pricing out. Near-duplicate rates are
        // precisely the over-specified-k regime this workstream exists to study, so this is not a corner.
        //
        // Refuse the ill-conditioned solve and fall back to nu = 0, which is legitimate: for an EQUALITY
        // the multiplier is sign-unrestricted, so ANY pair whose reduced costs are non-positive is a valid
        // dual certificate; a worse-but-honest pair can only fail to certify, never falsely certify.
        double mu = 0.0, nu = 0.0;
        bool nu_free = false;
        const long double det = s0 * s2 - s1 * s1;
        const long double det_scale = s0 * s2;
        const bool well_conditioned =
            moment_binds && det_scale > 0.0L &&
            det > FREERATE_DUAL_CONDITION_FLOOR * det_scale;
        if (well_conditioned) {
            mu = static_cast<double>((gs * s2 - gr_ * s1) / det);
            nu = static_cast<double>((s0 * gr_ - s1 * gs) / det);
            nu_free = true;
        } else if (n > 0) {
            mu = static_cast<double>(gs / static_cast<long double>(n));
            nu = 0.0;
        }

        // Error budget for d_j = g_j - mu - nu*r_j. gap.noise models the rounding of ONE directional
        // score difference and has no relation to this cancellation: it was measured 57x too small at a
        // rate separation of 1e-3 and, in a near-degenerate case, 0.53 nats too LARGE -- which would
        // dismiss a half-nat escape direction as rounding against a 1e-5 bar. Build the budget from the
        // magnitudes that actually cancel here.
        const double rc_noise =
            64.0 * std::numeric_limits<double>::epsilon() *
            (static_cast<double>(gmax) + std::fabs(mu) +
             std::fabs(nu) * static_cast<double>(rmax) + 1.0);

        // KKT sign condition on an ACTIVE MOMENT INEQUALITY.
        //
        // The moment row is promoted into the face-defining set whenever a quotient bound is active, so
        // every direction that LEAVES that bound is excluded from the decrement. Nothing then priced the
        // multiplier of that inequality, and the reduced-cost loop only ever looked at weight
        // coordinates -- so a point pinned against a moment bound reported global with bound 0 while a
        // feasible competitor sat over 1000 nats higher. For maximisation with r.w <= m_U the multiplier
        // must satisfy nu >= 0, and for r.w >= m_L it must satisfy nu <= 0; a violated sign IS the
        // improving direction. This is only meaningful when nu was actually identified.
        bool moment_sign_ok = true;
        if (moment_inequality_active) {
            if (!nu_free) {
                moment_sign_ok = false;      // could not price the binding inequality at all
            } else if (moment_upper_active && nu < -rc_noise) {
                moment_sign_ok = false;
            } else if (moment_lower_active && nu > rc_noise) {
                moment_sign_ok = false;
            }
        }

        double worst = -std::numeric_limits<double>::infinity();
        for (std::size_t t = 0; t < inactive.size(); ++t) {
            const std::size_t j = inactive[t];
            const double d = current.gradient[j] - mu - nu * problem.rate[j];
            if (d > worst) worst = d;
        }
        // Categories held at a POSITIVE weight below the activity tolerance are dropped from the reduced
        // Hessian, so the decrement cannot see them, and the loop above prices only the direction of
        // INCREASE. Their improving move is to shed their remaining mass, worth about |d_j| * w_j, which
        // no other test bounds. The activity tolerance is documented as a reporting knob, yet it silently
        // scales this leak, so bound it explicitly instead of trusting the default.
        double shed_bound = 0.0;
        for (std::size_t t = 0; t < inactive.size(); ++t) {
            const std::size_t j = inactive[t];
            if (weight[j] > 0.0) {
                const double d = current.gradient[j] - mu - nu * problem.rate[j];
                if (d < 0.0) shed_bound += (-d) * weight[j];
            }
        }
        result.max_inactive_reduced_cost =
            inactive.empty() ? -std::numeric_limits<double>::infinity() : worst;
        result.global = moment_sign_ok &&
                        (inactive.empty() || worst <= rc_noise) &&
                        shed_bound <= rc_noise;
    }

    // Orthonormal basis of the null space, by Gram-Schmidt against the constraint rows and each other.
    std::vector<std::vector<double> > basis;
    // ORTHONORMALISE THE CONSTRAINT ROWS AGAINST EACH OTHER FIRST.
    //
    // Projecting a seed vector against the mass row and then the moment row in sequence does NOT leave it
    // orthogonal to both, because those two rows are not mutually orthogonal. Skipping this step puts the
    // "null space" basis outside the null space, so the reduced gradient never vanishes at the optimum
    // and the decrement explodes -- measured lambda = 341 on a cell whose true value is ~3e-7.
    std::vector<std::vector<double> > qrows;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        std::vector<double> u = rows[r];
        for (std::size_t q = 0; q < qrows.size(); ++q) {
            long double num = 0.0L;
            for (std::size_t i = 0; i < n; ++i)
                num += static_cast<long double>(qrows[q][i]) * u[i];
            const double f = static_cast<double>(num);
            for (std::size_t i = 0; i < n; ++i) u[i] -= f * qrows[q][i];
        }
        long double rn2 = 0.0L;
        for (std::size_t i = 0; i < n; ++i)
            rn2 += static_cast<long double>(u[i]) * u[i];
        const double rn = std::sqrt(static_cast<double>(rn2));
        if (!(rn > 1e-10)) continue;              // dependent constraint row on this face
        for (std::size_t i = 0; i < n; ++i) u[i] /= rn;
        qrows.push_back(u);
    }

    for (std::size_t seed = 0; seed < n && basis.size() + qrows.size() < n; ++seed) {
        std::vector<double> v(n, 0.0);
        v[seed] = 1.0;
        for (std::size_t q = 0; q < qrows.size(); ++q) {
            long double num = 0.0L;
            for (std::size_t i = 0; i < n; ++i)
                num += static_cast<long double>(qrows[q][i]) * v[i];
            const double f = static_cast<double>(num);
            for (std::size_t i = 0; i < n; ++i) v[i] -= f * qrows[q][i];
        }
        for (std::size_t b = 0; b < basis.size(); ++b) {
            long double num = 0.0L;
            for (std::size_t i = 0; i < n; ++i)
                num += static_cast<long double>(basis[b][i]) * v[i];
            const double f = static_cast<double>(num);
            for (std::size_t i = 0; i < n; ++i) v[i] -= f * basis[b][i];
        }
        long double norm2 = 0.0L;
        for (std::size_t i = 0; i < n; ++i)
            norm2 += static_cast<long double>(v[i]) * v[i];
        const double norm = std::sqrt(static_cast<double>(norm2));
        if (!(norm > 1e-8)) continue;                 // dependent direction
        for (std::size_t i = 0; i < n; ++i) v[i] /= norm;
        basis.push_back(v);
    }

    const std::size_t m = basis.size();
    if (m == 0) {
        // No free direction on this face: the constraints pin it, so nothing is left to gain here.
        result.valid = true;
        result.decrement = 0.0;
        result.bound = 0.0;
        return result;
    }

    // Reduced gradient and reduced negated Hessian.
    std::vector<double> gr(m, 0.0);
    for (std::size_t b = 0; b < m; ++b) {
        long double acc = 0.0L;
        for (std::size_t i = 0; i < n; ++i)
            acc += static_cast<long double>(basis[b][i]) * current.gradient[active[i]];
        gr[b] = static_cast<double>(acc);
    }
    std::vector<double> hr(m * m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b < m; ++b) {
            long double acc = 0.0L;
            for (std::size_t i = 0; i < n; ++i) {
                long double inner = 0.0L;
                for (std::size_t j = 0; j < n; ++j) {
                    // curvature holds -H (positive semi-definite by concavity)
                    inner += static_cast<long double>(
                                 current.curvature[active[i] * k + active[j]]) *
                             basis[b][j];
                }
                acc += static_cast<long double>(basis[a][i]) * inner;
            }
            hr[a * m + b] = static_cast<double>(acc);
        }
    }

    // Cholesky; a non-positive pivot means the reduced curvature is singular (duplicate or
    // observationally equivalent columns), in which case no second-order bound is available.
    std::vector<double> chol(hr);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            long double acc = static_cast<long double>(chol[i * m + j]);
            for (std::size_t p = 0; p < j; ++p)
                acc -= static_cast<long double>(chol[i * m + p]) * chol[j * m + p];
            if (i == j) {
                if (!(acc > 0.0L) || !std::isfinite(static_cast<double>(acc)))
                    return result;
                chol[i * m + i] = std::sqrt(static_cast<double>(acc));
            } else {
                chol[i * m + j] = static_cast<double>(acc) / chol[j * m + j];
            }
        }
    }
    std::vector<double> y(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        long double acc = static_cast<long double>(gr[i]);
        for (std::size_t p = 0; p < i; ++p)
            acc -= static_cast<long double>(chol[i * m + p]) * y[p];
        y[i] = static_cast<double>(acc) / chol[i * m + i];
    }
    long double lambda2 = 0.0L;
    for (std::size_t i = 0; i < m; ++i)
        lambda2 += static_cast<long double>(y[i]) * y[i];
    if (!(lambda2 >= 0.0L) || !std::isfinite(static_cast<double>(lambda2)))
        return result;

    const double lambda = std::sqrt(static_cast<double>(lambda2));
    result.valid = true;
    result.decrement = lambda;

    // omega*(lambda) = -lambda - log(1-lambda), valid for lambda < 1 on a self-concordant objective.
    // Outside that radius the second-order model says nothing and the bound is refused.
    //
    // Evaluate it by its series sum_{n>=2} lambda^n / n whenever lambda is small. The closed form
    // subtracts two nearly equal quantities: at lambda = 2.2e-12 the true value is ~2.5e-24, far below
    // the rounding of either term, and the expression returned -1.8e-17 -- a NEGATIVE "upper bound" on a
    // quantity that is non-negative by construction. It failed safe only because bestGapBound() rejects
    // negatives, which silently threw away a valid and much tighter certificate.
    if (lambda >= 1.0) {
        result.bound = std::numeric_limits<double>::infinity();
    } else if (lambda < 0.5) {
        long double term = static_cast<long double>(lambda) * lambda;   // n = 2
        long double acc = term / 2.0L;
        for (int n = 3; n <= 24; ++n) {
            term *= static_cast<long double>(lambda);
            const long double add = term / static_cast<long double>(n);
            if (add <= 0.0L) break;
            acc += add;
            if (add < acc * 1e-22L) break;
        }
        result.bound = static_cast<double>(acc);
    } else {
        result.bound = -lambda - std::log(1.0 - lambda);
    }
    if (!(result.bound >= 0.0)) {
        result.bound = 0.0;      // the series is non-negative by construction; clamp residual rounding
    }
    return result;
}

void populateResultDiagnostics(const ProfileProblem &problem,
                               const ProfileOptions &options,
                               const std::vector<std::vector<double> > &vertices,
                               const std::vector<double> &weight,
                               const Evaluation &evaluation,
                               const GapResult &gap,
                               ProfileResult *result) {
    result->weight = weight;
    result->log_likelihood = likelihoodFitsResultType(evaluation)
                                 ? static_cast<double>(
                                       evaluation.total_log_likelihood)
                                 : quietNaN();
    /* Publish the raw value BEFORE clamping. The clamp keeps frank_wolfe_gap meaningful as a bound, but
     * on its own it erases the one signal that distinguishes a genuine near-optimum from an incomplete
     * vertex set: both then read 0.0, which is the strongest certificate value there is. */
    result->signed_directional_gap = gap.valid ? gap.gap : quietNaN();
    result->gap_noise_floor = gap.valid ? gap.noise : quietNaN();
    result->frank_wolfe_gap = gap.valid ? std::max(0.0, gap.gap) : quietNaN();
    result->kkt_directional_residual = result->frank_wolfe_gap;
    /* A materially negative directional maximum proves w lies outside conv(enumerated vertices), so the
     * enumeration is incomplete and nothing derived from it bounds l* - l(w). Same threshold the solver
     * loop uses to type this condition as NUMERICAL_STALL, kept in one place deliberately. */
    result->gap_is_valid_bound =
        gap.valid && !(gap.gap < -gap.noise);

    result->moment = vectorMoment(weight, problem.rate);
    result->primal = residualsFor(problem, weight);
    result->feasible_vertex_count = vertices.size();
    result->active_weight_count = 0;
    result->active_index.clear();
    for (std::size_t j = 0; j < weight.size(); ++j) {
        if (weight[j] > options.active_weight_tolerance) {
            ++result->active_weight_count;
            result->active_index.push_back(j);
        }
    }
    if (problem.geometry == FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL) {
        result->moment_lower_slack = result->moment - problem.moment_lower;
        result->moment_upper_slack = problem.moment_upper - result->moment;
        result->moment_lower_active =
            std::fabs(result->moment_lower_slack) <= options.primal_tolerance;
        result->moment_upper_active =
            std::fabs(result->moment_upper_slack) <= options.primal_tolerance;
    } else {
        result->moment_lower_slack = quietNaN();
        result->moment_upper_slack = quietNaN();
        result->moment_lower_active = false;
        result->moment_upper_active = false;
    }
    /* Second-order certificate on the active face. MUST come after the moment-activity flags above:
     * it needs them to know which constraint rows actually bind. Independent of the vertex enumeration,
     * so it also cross-checks it -- two different derivations should not disagree about optimality. */
    const NewtonCertificate newton = computeNewtonCertificate(
        problem, options, weight, evaluation,
        result->moment_lower_active, result->moment_upper_active,
        minMultiplicity(problem));
    result->newton_decrement = newton.valid ? newton.decrement : quietNaN();
    result->newton_gap_bound =
        newton.valid ? newton.bound : std::numeric_limits<double>::infinity();
    result->newton_bound_is_global = newton.valid && newton.global;
    result->max_inactive_reduced_cost = newton.max_inactive_reduced_cost;
}

} // namespace

ProfileProblem::ProfileProblem()
    : pattern_count(0),
      category_count(0),
      geometry(FeasibleGeometry::LITERAL_MASS_MEAN),
      target_moment(1.0),
      moment_lower(1.0),
      moment_upper(1.0) {}

ProfileOptions::ProfileOptions()
    : gap_tolerance(1.0e-8),
      primal_tolerance(1.0e-12),
      active_weight_tolerance(1.0e-12),
      max_iterations(10000) {}

PrimalResiduals::PrimalResiduals()
    : mass(quietNaN()),
      literal_moment(quietNaN()),
      moment_lower_violation(quietNaN()),
      moment_upper_violation(quietNaN()),
      negativity(quietNaN()) {}

ProfileResult::ProfileResult()
    : reason(ExitReason::INVALID_INPUT),
      log_likelihood(quietNaN()),
      frank_wolfe_gap(quietNaN()),
      kkt_directional_residual(quietNaN()),
      signed_directional_gap(quietNaN()),
      gap_noise_floor(quietNaN()),
      newton_decrement(quietNaN()),
      newton_gap_bound(std::numeric_limits<double>::infinity()),
      gap_is_valid_bound(false),
      newton_bound_is_global(false),
      max_inactive_reduced_cost(-std::numeric_limits<double>::infinity()),
      moment(quietNaN()),
      moment_lower_slack(quietNaN()),
      moment_upper_slack(quietNaN()),
      iterations(0),
      objective_evaluations(0),
      feasible_vertex_count(0),
      active_weight_count(0),
      diagnostic_limitations(
          PROFILE_EXPLICIT_DUAL_MULTIPLIERS_UNAVAILABLE |
          PROFILE_CURVATURE_RANK_UNAVAILABLE |
          PROFILE_PRODUCTION_COMPONENT_RECONSTRUCTION_UNVALIDATED |
          PROFILE_OUTER_PARAMETERS_NOT_CERTIFIED |
          PROFILE_ADDITIVE_BACKGROUND_UNSUPPORTED),
      fixed_column_global_certificate(false),
      supplied_start_used(false),
      moment_lower_active(false),
      moment_upper_active(false) {}

const char *exitReasonName(ExitReason reason) {
    switch (reason) {
    case ExitReason::CONVERGED_GAP:
        return "CONVERGED_GAP";
    case ExitReason::MAX_ITERATIONS:
        return "MAX_ITERATIONS";
    case ExitReason::INFEASIBLE_POLYTOPE:
        return "INFEASIBLE_POLYTOPE";
    case ExitReason::INVALID_INPUT:
        return "INVALID_INPUT";
    case ExitReason::NONFINITE_OBJECTIVE:
        return "NONFINITE_OBJECTIVE";
    case ExitReason::NUMERICAL_STALL:
        return "NUMERICAL_STALL";
    }
    return "UNKNOWN";
}

std::vector<std::vector<double> > enumerateFeasibleVertices(
    const ProfileProblem &problem) {
    std::vector<std::vector<double> > vertices;
    if (!affineInputIsValid(problem)) {
        return vertices;
    }
    const std::size_t k = problem.category_count;

    if (problem.geometry == FeasibleGeometry::LITERAL_MASS_MEAN) {
        for (std::size_t j = 0; j < k; ++j) {
            if (problem.rate[j] == problem.target_moment) {
                std::vector<double> vertex(k, 0.0);
                vertex[j] = 1.0;
                addVertex(problem, vertex, &vertices);
            }
        }
        addMomentBoundaryVertices(problem, problem.target_moment, &vertices);
        return vertices;
    }

    for (std::size_t j = 0; j < k; ++j) {
        if (problem.rate[j] >= problem.moment_lower &&
            problem.rate[j] <= problem.moment_upper) {
            std::vector<double> vertex(k, 0.0);
            vertex[j] = 1.0;
            addVertex(problem, vertex, &vertices);
        }
    }
    addMomentBoundaryVertices(problem, problem.moment_lower, &vertices);
    if (problem.moment_lower != problem.moment_upper) {
        addMomentBoundaryVertices(problem, problem.moment_upper, &vertices);
    }
    return vertices;
}

ProfileResult solve(const ProfileProblem &problem,
                    const ProfileOptions &options,
                    const std::vector<double> &supplied_start) {
    ProfileResult result;
    if (!std::isfinite(options.gap_tolerance) ||
        options.gap_tolerance < 0.0 ||
        !std::isfinite(options.primal_tolerance) ||
        options.primal_tolerance <= 0.0 ||
        !std::isfinite(options.active_weight_tolerance) ||
        options.active_weight_tolerance < 0.0 ||
        options.max_iterations == 0) {
        return result;
    }

    PreparedProblem prepared;
    const PreparationStatus preparation = prepareProblem(problem, &prepared);
    if (preparation == PreparationStatus::INVALID) {
        result.reason = ExitReason::INVALID_INPUT;
        return result;
    }
    if (preparation == PreparationStatus::IMPOSSIBLE_PATTERN) {
        result.reason = ExitReason::NONFINITE_OBJECTIVE;
        return result;
    }

    const std::vector<std::vector<double> > vertices =
        enumerateFeasibleVertices(problem);
    result.feasible_vertex_count = vertices.size();
    if (vertices.empty()) {
        result.reason = ExitReason::INFEASIBLE_POLYTOPE;
        return result;
    }

    std::vector<double> weight;
    if (supplied_start.size() == problem.category_count &&
        maximumResidual(residualsFor(problem, supplied_start)) <=
            options.primal_tolerance) {
        Evaluation supplied_evaluation =
            evaluate(prepared, supplied_start, false,
                     &result.objective_evaluations);
        if (supplied_evaluation.finite) {
            weight = supplied_start;
            result.supplied_start_used = true;
        }
    }
    if (weight.empty()) {
        weight = vertexCentroid(vertices);
    }

    Evaluation current =
        evaluate(prepared, weight, true, &result.objective_evaluations);
    if (!current.finite) {
        result.reason = ExitReason::NONFINITE_OBJECTIVE;
        return result;
    }

    GapResult gap;
    for (std::size_t iteration = 0; iteration <= options.max_iterations;
         ++iteration) {
        gap = computeGap(weight, current.gradient, vertices);
        const PrimalResiduals primal = residualsFor(problem, weight);
        if (!likelihoodFitsResultType(current)) {
            result.reason = ExitReason::NONFINITE_OBJECTIVE;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }
        /* Compare the gap against ITS OWN resolution floor, not against primal_tolerance.
         * primal_tolerance measures feasibility in weight/moment units; the gap is a directional
         * derivative in nats whose noise grows with the total pattern multiplicity. Testing one against
         * the other is dimensionally incoherent and wrong at both ends: on a small problem a genuinely
         * significant negative gap slips through the fixed slack and gets certified, while on a large
         * alignment pure rounding trips it and a correct fit is flagged. */
        if (!gap.valid || gap.gap < -gap.noise) {
            result.reason = ExitReason::NUMERICAL_STALL;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }
        if (std::max(0.0, gap.gap) <= options.gap_tolerance &&
            maximumResidual(primal) <= options.primal_tolerance) {
            result.reason = ExitReason::CONVERGED_GAP;
            result.fixed_column_global_certificate = true;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }
        if (iteration == options.max_iterations) {
            result.reason = ExitReason::MAX_ITERATIONS;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }

        std::vector<double> fw_direction(problem.category_count, 0.0);
        for (std::size_t j = 0; j < problem.category_count; ++j) {
            fw_direction[j] = vertices[gap.best_vertex][j] - weight[j];
        }
        LineCandidate best = maximizeOnLine(
            prepared, options, weight, current, fw_direction, 1.0, true,
            &result.objective_evaluations);

        std::vector<double> newton_direction;
        bool have_newton = buildNewtonDirection(
            problem, options, weight, current, false, &newton_direction);
        if (have_newton &&
            problem.geometry ==
                FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL) {
            const double moment = vectorMoment(weight, problem.rate);
            const double moment_direction =
                vectorMoment(newton_direction, problem.rate);
            const double boundary_tolerance =
                std::max(options.primal_tolerance,
                         128.0 * std::numeric_limits<double>::epsilon() *
                             scaleFor(problem.moment_lower,
                                      problem.moment_upper));
            const bool points_outward =
                (moment - problem.moment_lower <= boundary_tolerance &&
                 moment_direction < 0.0) ||
                (problem.moment_upper - moment <= boundary_tolerance &&
                 moment_direction > 0.0);
            if (points_outward) {
                have_newton = buildNewtonDirection(
                    problem, options, weight, current, true,
                    &newton_direction);
            }
        }
        if (have_newton) {
            const LineCandidate newton = maximizeOnLine(
                prepared, options, weight, current, newton_direction, 1.0,
                false, &result.objective_evaluations);
            if (newton.valid &&
                (!best.valid ||
                 newton.evaluation.variable_log_likelihood >
                     best.evaluation.variable_log_likelihood)) {
                best = newton;
            }
        }

        if (!best.valid) {
            result.reason = ExitReason::NUMERICAL_STALL;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }
        double maximum_weight_step = 0.0;
        for (std::size_t j = 0; j < problem.category_count; ++j) {
            maximum_weight_step =
                std::max(maximum_weight_step,
                         std::fabs(best.weight[j] - weight[j]));
        }
        if (maximum_weight_step <=
                2.0 * std::numeric_limits<double>::epsilon() &&
            std::max(0.0, gap.gap) > options.gap_tolerance) {
            result.reason = ExitReason::NUMERICAL_STALL;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }

        weight.swap(best.weight);
        current = evaluate(prepared, weight, true,
                           &result.objective_evaluations);
        if (!current.finite) {
            result.reason = ExitReason::NUMERICAL_STALL;
            populateResultDiagnostics(problem, options, vertices, weight,
                                      current, gap, &result);
            return result;
        }
        ++result.iterations;
    }

    result.reason = ExitReason::MAX_ITERATIONS;
    populateResultDiagnostics(problem, options, vertices, weight, current, gap,
                              &result);
    return result;
}

} // namespace freerate_profile
