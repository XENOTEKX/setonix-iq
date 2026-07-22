#include "model/freeratefit.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace freerate;

namespace {

int failures = 0;

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        std::cerr << __FILE__ << ':' << __LINE__                               \
                  << ": CHECK failed: " #condition << '\n';                   \
        ++failures;                                                             \
    }                                                                           \
} while (false)

FreeRatePoint validPoint() {
    FreeRatePoint point;
    point.rates = {0.5, 1.25};
    point.weights = {1.0 / 3.0, 2.0 / 3.0};
    point.branch_lengths = {0.05, 0.1, 0.2};
    point.substitution_rates = {1.0, 2.0};
    point.frequencies = {0.25, 0.75};
    return point;
}

FreeRateProvenance completeProvenance() {
    FreeRateProvenance provenance;
    provenance.source_commit =
        "ccabc96e111b08460e1e5e3acf55ac281624987e";
    provenance.solver_version = "freerate-profile-v1";
    provenance.domain_version = "q-ratio-v1";
    provenance.binary_digest_algorithm = "sha256";
    provenance.input_digest_algorithm = "sha256";
    provenance.binary_digest = std::string(64, 'a');
    provenance.alignment_digest = std::string(64, 'b');
    provenance.tree_digest = std::string(64, 'c');
    provenance.parameter_state_digest = std::string(64, 'd');
    provenance.candidate_manifest_digest = std::string(64, 'e');
    provenance.model_name = "GTR+F+R2";
    provenance.branch_mode = "BRLEN_FIX";
    provenance.command_line = "iqtree3 --fixed-test";
    provenance.run_identifier = "unit-test";
    provenance.host = "test-host";
    provenance.evaluator_backend = "CPU";
    provenance.compiler_identifier = "GNU";
    provenance.compiler_version = "test";
    provenance.build_type = "unit";
    provenance.build_flags = "-std=c++17";
    provenance.runtime_min_branch_length = 1e-6;
    provenance.runtime_max_branch_length = 20.0;
    provenance.seed = 17;
    provenance.thread_count = 1;
    provenance.jolt_ntile = 0;
    return provenance;
}

FreeRateFitResult passingFit() {
    FreeRateFitResult fit;
    fit.status = FreeRateFitStatus::LOCAL_STATIONARY_CERTIFIED;
    fit.final_likelihood = -123.5;
    fit.final_point = validPoint();
    fit.provenance = completeProvenance();
    fit.metrics.weight_profile_evaluated = true;
    fit.metrics.weight_profile_evaluations = 2;
    fit.metrics.weight_gap = 1e-7;
    // A conforming producer copies the certificate provenance alongside the value. Without these the fit
    // is not certifiable by design: a bare gap cannot distinguish a genuine bound from a clamped one
    // produced by a broken vertex enumeration.
    fit.metrics.weight_gap_is_valid_bound = true;
    fit.metrics.weight_newton_bound = 1e-12;
    fit.metrics.weight_newton_bound_is_global = true;
    fit.metrics.weight_best_bound = 1e-12;
    fit.metrics.mass_residual = 0.0;
    fit.metrics.mean_residual = 0.0;
    fit.metrics.negativity_residual = 0.0;
    fit.metrics.rate.evaluated = true;
    fit.metrics.rate.improvement_upper_bound = 1e-7;
    fit.metrics.rate.evaluations = 1;
    fit.metrics.support_events_evaluated = true;
    fit.metrics.best_tested_support_gain = 1e-7;
    fit.metrics.profiled_likelihood_change = 1e-7;
    fit.metrics.max_scaled_step = 1e-8;
    fit.metrics.consecutive_small_cycles = 2;
    fit.metrics.restart_portfolio_evaluated = true;
    // Must be at or below thresholds.restart_gain, which is now tied to tau_L. The old fixture used 0.01
    // and passed only because the restart bar defaulted to 0.1, a hundred times looser.
    fit.metrics.best_restart_gain = 1e-6;
    fit.metrics.cpu_gpu_parity_evaluated = true;
    fit.metrics.cpu_gpu_likelihood_delta = 1e-6;
    // A conforming producer corroborates the score it publishes by re-evaluating it at the final
    // point; without this witness the result is not certifiable, by design.
    fit.metrics.final_likelihood_verified = true;
    fit.metrics.final_likelihood_recheck_delta = 1e-9;
    // The producer must witness that it tracked the terminal failure conditions; unset flags alone are
    // not evidence that nothing went wrong.
    fit.metrics.termination_flags_tracked = true;
    fit.starts_attempted = 2;
    fit.starts_completed = 2;
    fit.counts.value = 12;
    fit.counts.gradient = 4;
    fit.counts.cpu_parity = 1;
    fit.counts.accepted_steps = 2;
    StableTraceDigest digest;
    digest.addRecord("passing-fit");
    fit.trace_digest = digest.str();
    return fit;
}

void testStatusRoundTrip() {
    const std::vector<FreeRateFitStatus> statuses = allFreeRateFitStatuses();
    CHECK(statuses.size() == 13);
    for (std::size_t i = 0; i < statuses.size(); ++i) {
        const std::string name = freeRateFitStatusName(statuses[i]);
        CHECK(name.find("MLE") == std::string::npos);
        FreeRateFitStatus parsed = FreeRateFitStatus::NUMERICAL_FAILURE;
        CHECK(parseFreeRateFitStatus(name, &parsed));
        CHECK(parsed == statuses[i]);
    }
    FreeRateFitStatus unchanged = FreeRateFitStatus::MAXITER;
    CHECK(!parseFreeRateFitStatus("CONVERGED", &unchanged));
    CHECK(unchanged == FreeRateFitStatus::MAXITER);
    CHECK(!parseFreeRateFitStatus("MAXITER", nullptr));
}

void testCanonicalization() {
    FreeRatePoint point;
    point.rates = {2.0, 0.5, 1.0};
    point.weights = {0.2, 0.4, 0.4};
    point.branch_lengths = {0.05, 0.2};
    point.pinv = -0.0;
    point.category_labels = {"slow?", "slow", "middle"};
    point.category_data = {
        {"score", {20.0, 5.0, 10.0}},
        {"gradient", {200.0, 50.0, 100.0}}
    };
    std::vector<double> external = {2000.0, 500.0, 1000.0};
    std::vector<std::vector<double> *> paired = {&external};
    std::string error;
    CHECK(canonicalizeFreeRatePoint(&point, paired, &error));
    CHECK(error.empty());
    CHECK(point.rates == std::vector<double>({0.5, 1.0, 2.0}));
    CHECK(point.weights == std::vector<double>({0.4, 0.4, 0.2}));
    CHECK(point.category_labels ==
          std::vector<std::string>({"slow", "middle", "slow?"}));
    CHECK(point.category_data[0].name == "gradient");
    CHECK(point.category_data[0].values ==
          std::vector<double>({50.0, 100.0, 200.0}));
    CHECK(point.category_data[1].name == "score");
    CHECK(point.category_data[1].values ==
          std::vector<double>({5.0, 10.0, 20.0}));
    CHECK(external == std::vector<double>({500.0, 1000.0, 2000.0}));
    CHECK(!std::signbit(point.pinv));

    const std::string once = stableJson(point);
    CHECK(canonicalizeFreeRatePoint(&point, &error));
    CHECK(stableJson(point) == once);

    std::vector<double> wrong(2, 0.0);
    paired = {&wrong};
    const std::string before_failed_canonicalize = stableJson(point);
    const std::vector<double> wrong_before = wrong;
    CHECK(!canonicalizeFreeRatePoint(&point, paired, &error));
    CHECK(!error.empty());
    CHECK(stableJson(point) == before_failed_canonicalize);
    CHECK(wrong == wrong_before);

    // A failure discovered only after sorting must also be atomic.
    FreeRatePoint invalid_after_sort = point;
    invalid_after_sort.rates = {2.0, 0.5, 1.0};
    invalid_after_sort.weights = {0.1, 0.4, 0.4}; // Invalid mass.
    invalid_after_sort.category_labels = {"last", "first", "middle"};
    invalid_after_sort.category_data = {{"x", {2.0, 0.5, 1.0}}};
    std::vector<double> valid_size_external = {20.0, 5.0, 10.0};
    const std::string invalid_before = stableJson(invalid_after_sort);
    const std::vector<double> external_before = valid_size_external;
    paired = {&valid_size_external};
    CHECK(!canonicalizeFreeRatePoint(&invalid_after_sort, paired, &error));
    CHECK(stableJson(invalid_after_sort) == invalid_before);
    CHECK(valid_size_external == external_before);
}

void testInvalidPoints() {
    FreeRatePoint point = validPoint();
    CHECK(validateFreeRatePoint(point).valid);

    FreeRatePoint invariant;
    invariant.rates = {0.5, 2.0};
    invariant.weights = {0.4, 0.4};
    invariant.has_invariant = true;
    invariant.pinv = 0.2;
    invariant.branch_lengths = {0.1};
    CHECK(validateFreeRatePoint(invariant).valid);

    point = validPoint();
    point.weights.pop_back();
    CHECK(validateFreeRatePoint(point).code == "CATEGORY_SIZE_MISMATCH");

    point = validPoint();
    point.rates[0] = 0.0;
    CHECK(validateFreeRatePoint(point).code == "INVALID_RATE");

    point = validPoint();
    point.weights = {-0.1, 1.1};
    CHECK(validateFreeRatePoint(point).code == "NEGATIVE_WEIGHT");

    point = validPoint();
    point.weights = {0.5, 0.5};
    CHECK(validateFreeRatePoint(point).code == "MEAN_CONSTRAINT");

    point = validPoint();
    point.has_invariant = true;
    point.pinv = 1.0;
    CHECK(validateFreeRatePoint(point).code == "INVALID_PINV");

    point = validPoint();
    point.pinv = 0.1;
    CHECK(validateFreeRatePoint(point).code == "UNDECLARED_PINV");

    point = validPoint();
    point.branch_lengths[1] = 0.0;
    CHECK(validateFreeRatePoint(point).code == "INVALID_BRANCH");

    point = validPoint();
    point.frequencies = {0.3, 0.6};
    CHECK(validateFreeRatePoint(point).code == "FREQUENCY_CONSTRAINT");

    point = validPoint();
    point.category_data = {{"x", {1.0}}};
    CHECK(validateFreeRatePoint(point).code == "CATEGORY_DATA_SIZE_MISMATCH");
}

void testSerializationAndDigest() {
    FreeRatePoint schema_fixture;
    schema_fixture.rates = {1.0};
    schema_fixture.weights = {1.0};
    schema_fixture.branch_lengths = {0.5};
    const std::string schema_fixture_json =
        "{\"rates\":[1.00000000000000000e+00],"
        "\"weights\":[1.00000000000000000e+00],"
        "\"branch_lengths\":[5.00000000000000000e-01],"
        "\"pinv\":0,\"has_invariant\":false,"
        "\"substitution_rates\":[],\"frequencies\":[],"
        "\"category_labels\":[],\"category_data\":[]}";
    CHECK(stableJson(schema_fixture) == schema_fixture_json);

    FreeRatePoint point = validPoint();
    point.category_labels = {"a\n", "b\""};
    const std::string first = stableJson(point);
    const std::string second = stableJson(point);
    CHECK(first == second);
    CHECK(first.find("a\\n") != std::string::npos);
    CHECK(first.find("b\\\"") != std::string::npos);
    CHECK(first.find("-0.000") == std::string::npos);

    FreeRateStateSnapshot snapshot;
    snapshot.iteration = 400;
    snapshot.point = point;
    snapshot.pre_gauge_likelihood = -100.0;
    snapshot.post_gauge_likelihood = -99.5;
    snapshot.gauge_likelihood_delta = 0.5;
    snapshot.accepted = true;
    snapshot.counts.value = 7;
    snapshot.status = FreeRateFitStatus::MAXITER;

    StableTraceDigest a;
    StableTraceDigest b;
    a.addSnapshot(snapshot);
    b.addRecord(stableJson(snapshot));
    CHECK(a.value() == b.value());
    CHECK(a.str() == b.str());
    CHECK(a.str().find("fnv1a64-v1:") == 0);

    snapshot.iteration = 401;
    StableTraceDigest changed;
    changed.addSnapshot(snapshot);
    CHECK(changed.str() != a.str());

    StableTraceDigest framed_a;
    StableTraceDigest framed_b;
    framed_a.addRecord("ab");
    framed_a.addRecord("c");
    framed_b.addRecord("a");
    framed_b.addRecord("bc");
    CHECK(framed_a.str() != framed_b.str());

    StableTraceDigest algorithm_fixture;
    algorithm_fixture.addRecord("abc");
    CHECK(algorithm_fixture.str() == "fnv1a64-v1:c11ab6d2519bc2b2");

    a.reset();
    StableTraceDigest empty;
    CHECK(a.str() == empty.str());
}

void testFailClosedCertification() {
    FreeRateFitResult fit;
    std::string reason;
    CHECK(!fit.certifiedForSelection(&reason));
    CHECK(!reason.empty());
    CHECK(stableJson(fit).find("\"certified_for_selection\":false") !=
          std::string::npos);

    fit = passingFit();
    CHECK(fit.certifiedForSelection(&reason));
    CHECK(reason.empty());
    CHECK(stableJson(fit).find("\"certified_for_selection\":true") !=
          std::string::npos);

    fit.status = FreeRateFitStatus::MAXITER;
    CHECK(!fit.certifiedForSelection(&reason));
    CHECK(stableJson(fit).find("\"certified_for_selection\":false") !=
          std::string::npos);

    // Budget exhaustion alone must not disqualify: the residual gates decide.
    fit = passingFit();
    fit.metrics.iteration_cap_reached = true;
    CHECK(fit.certifiedForSelection(&reason));
    CHECK(reason.empty());

    // A capped fit with a material residual is still refused, by the gap gate. The residual that counts
    // is the TIGHTEST justified bound, so both must be material -- a loose Frank-Wolfe value beside a
    // valid tighter second-order bound is no longer a rejection, which is the entire reason the
    // curvature-aware certificate exists: FW overstates the shortfall by >=1e7x on over-specified fits.
    fit = passingFit();
    fit.metrics.iteration_cap_reached = true;
    fit.metrics.weight_gap = 10.0 * fit.thresholds.weight_gap;
    fit.metrics.weight_best_bound = fit.metrics.weight_gap;
    CHECK(!fit.certifiedForSelection(&reason));

    // ... and that relaxation is real: the same capped fit WITH a valid tight global bound certifies.
    fit = passingFit();
    fit.metrics.iteration_cap_reached = true;
    fit.metrics.weight_gap = 10.0 * fit.thresholds.weight_gap;
    fit.metrics.weight_newton_bound = 1e-12;
    fit.metrics.weight_newton_bound_is_global = true;
    fit.metrics.weight_best_bound = 1e-12;
    CHECK(fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.counts.value = 0;
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.metrics.weight_gap = -1e-7;
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.metrics.rate.improvement_upper_bound = -1e-7;
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.thresholds.constraint_residual = 1e-10;
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.thresholds.restart_gain = 1.0;
    CHECK(!fit.certifiedForSelection(&reason));

    // F4. At the shipped defaults a restart portfolio that found a start 0.0999 nats BETTER than the
    // published point used to certify, because restart_gain defaulted to 0.1 independently of tau_L.
    // 0.0999 nats is a larger discrepancy than the entire measured weight-block shortfall on three of
    // the four Stage-0 cells, so the certificate was asserting stationarity across a gap of the very
    // size it exists to detect.
    fit = passingFit();
    fit.metrics.best_restart_gain = 0.0999;
    CHECK(!fit.certifiedForSelection(&reason));
    // and the bar itself may not be declared looser than tau_L
    fit = passingFit();
    fit.thresholds.restart_gain = 10.0 * fit.thresholds.likelihood_gain;
    CHECK(!fit.certifiedForSelection(&reason));

    // A gap value with no validity witness is not a certificate: a point outside a broken vertex hull
    // reports a clamped, small, non-negative gap that is indistinguishable from a converged one.
    fit = passingFit();
    fit.metrics.weight_gap_is_valid_bound = false;
    CHECK(!fit.certifiedForSelection(&reason));

    // The tightest bound must actually be a bound: never looser than the gap it refines.
    fit = passingFit();
    fit.metrics.weight_best_bound = 10.0 * fit.metrics.weight_gap;
    CHECK(!fit.certifiedForSelection(&reason));

    // A face-local second-order bound must not certify a point whose global gap is unbounded. A producer
    // that leaves the bound un-globalised has to fall back to the Frank-Wolfe value, which here fails.
    fit = passingFit();
    fit.metrics.weight_newton_bound_is_global = false;
    fit.metrics.weight_gap = 10.0 * fit.thresholds.weight_gap;
    fit.metrics.weight_best_bound = fit.metrics.weight_gap;
    CHECK(!fit.certifiedForSelection(&reason));

    // F9. Unset failure flags certify only when something was watching. A producer with no line search
    // and no support-event handler must not pass the "nothing went wrong" gate by silence.
    fit = passingFit();
    fit.metrics.termination_flags_tracked = false;
    CHECK(!fit.certifiedForSelection(&reason));

    // FALLBACK_CERTIFIED must certify at a BOUNDARY as well as in the interior.
    //
    // This pins the corrected reading of the status. It is a provenance label -- plan §11.2, the
    // higher-budget CPU re-run after GPU fitting failed -- not a geometry claim, and §1.3/§5.2 make the
    // literal fallback the production path precisely WHEN a quotient moment bound is active. An earlier
    // revision rejected boundary fallbacks; by §11.2 step 5 that escalates the whole analysis to
    // INCOMPLETE, so the false rejection was expensive rather than conservative.
    //
    // rate_ratio_lower is used because the recomputation actually derives it (schema-v1 scope has
    // optimize_branches/pinv/substitution false, so those boundaries are never derived and reporting one
    // trips the disagreement check instead) and because it is NOT a support boundary, so no insertion
    // pricing is owed and the geometry is the only thing under test.
    {
        FreeRateFitResult fb = passingFit();
        const double ratio = fb.scope.rate_ratio_lower;
        const double w0 = 0.5, w1 = 0.5;
        const double r1 = 1.0 / (w0 * ratio + w1);   // pins sum(w*r) == 1
        const double r0 = ratio * r1;                // pins r0/r1 == rate_ratio_lower
        fb.final_point.rates = {r0, r1};
        fb.final_point.weights = {w0, w1};
        fb.metrics.boundary.rate_ratio_lower = true;

        fb.status = FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED;
        CHECK(fb.certifiedForSelection(&reason));
        fb.status = FreeRateFitStatus::FALLBACK_CERTIFIED;
        CHECK(fb.certifiedForSelection(&reason));

        // A support boundary is different: its obligation is insertion pricing, and that binds EVERY
        // status including the fallback. This is what actually stops a fallback from dodging anything.
        FreeRateFitResult sb = passingFit();
        sb.status = FreeRateFitStatus::FALLBACK_CERTIFIED;
        sb.final_point.rates = {1.0, 1.0 + 2e-11};
        sb.final_point.weights = {0.5, 0.5};
        const double m2 = 0.5 * sb.final_point.rates[0] + 0.5 * sb.final_point.rates[1];
        sb.final_point.rates[0] /= m2;
        sb.final_point.rates[1] /= m2;
        sb.metrics.boundary.rate_collision = true;
        sb.metrics.continuous_insertion_evaluated = false;
        CHECK(!sb.certifiedForSelection(&reason));

        // The collision threshold must not be switchable by the producer it constrains. Declaring
        // scaled_step = 0 (which valid() permits) previously disabled collision detection entirely:
        // the colliding point below would report no collision, raise no support boundary, and certify
        // with no insertion pricing.
        FreeRateFitResult ev = passingFit();
        ev.thresholds.scaled_step = 0.0;
        ev.metrics.max_scaled_step = 0.0;
        ev.final_point.rates = {1.0, 1.0 + 2e-11};
        ev.final_point.weights = {0.5, 0.5};
        const double m3 = 0.5 * ev.final_point.rates[0] + 0.5 * ev.final_point.rates[1];
        ev.final_point.rates[0] /= m3;
        ev.final_point.rates[1] /= m3;
        ev.metrics.boundary.rate_collision = false;   // the evasion: claim there is no collision
        CHECK(!ev.certifiedForSelection(&reason));

        fit = passingFit();
        fit.status = FreeRateFitStatus::FALLBACK_CERTIFIED;
        CHECK(fit.certifiedForSelection(&reason));
    }

    // F7. Two rates 2e-11 apart are the same mixture component to twelve significant figures, and an
    // unidentifiable direction. Priced on the feasibility tolerance (1e-12) they read as DISTINCT, so the
    // point certified as interior with no continuous insertion pricing. Priced on the solver's own
    // log-rate resolution they are a collision, which is a support boundary and demands pricing.
    {
        FreeRateFitResult collide = passingFit();
        collide.final_point.rates = {1.0, 1.0 + 2e-11};
        collide.final_point.weights = {0.5, 0.5};
        // Re-gauge so the point stays legal: sum(w)=1 and sum(w*r)=1 must still hold.
        const double m = 0.5 * collide.final_point.rates[0] +
                         0.5 * collide.final_point.rates[1];
        collide.final_point.rates[0] /= m;
        collide.final_point.rates[1] /= m;
        collide.status = FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED;
        // Without insertion pricing a collided point must NOT certify.
        collide.metrics.continuous_insertion_evaluated = false;
        CHECK(!collide.certifiedForSelection(&reason));
        // With pricing it may.
        collide.metrics.boundary.rate_collision = true;
        collide.metrics.continuous_insertion_evaluated = true;
        collide.metrics.continuous_insertion_gain_upper_bound = 1e-9;
        CHECK(collide.certifiedForSelection(&reason));
    }

    fit = passingFit();
    fit.final_point.rates = {1.25, 0.5};
    fit.final_point.weights = {2.0 / 3.0, 1.0 / 3.0};
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.trace_digest = "fnv1a64-v1:000000000000000g";
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.provenance.binary_digest[0] = 'G';
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.status = FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED;
    fit.final_point.rates = {0.5, 1.0, 1.5};
    fit.final_point.weights = {0.5, 0.0, 0.5};
    fit.metrics.boundary.zero_weight = true;
    CHECK(!fit.certifiedForSelection(&reason));
    fit.metrics.continuous_insertion_evaluated = true;
    fit.metrics.continuous_insertion_gain_upper_bound = 1e-7;
    CHECK(fit.certifiedForSelection(&reason));

    fit = FreeRateFitResult::failure(FreeRateFitStatus::MAXITER, "cap");
    CHECK(fit.status == FreeRateFitStatus::MAXITER);
    CHECK(!fit.certifiedForSelection(&reason));

    fit = FreeRateFitResult::failure(
        FreeRateFitStatus::LOCAL_STATIONARY_CERTIFIED, "bad caller");
    CHECK(fit.status == FreeRateFitStatus::NUMERICAL_FAILURE);
    CHECK(!fit.certifiedForSelection(&reason));

    // F5 regression: the published score must be corroborated by a fresh re-evaluation at the final
    // point. An unverified score never certifies, however extreme, and a verified-but-divergent one is
    // rejected too.
    fit = passingFit();
    fit.metrics.final_likelihood_verified = false;
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.final_likelihood = 1.0e12;   // absurd score, but the witness was never set
    fit.metrics.final_likelihood_verified = false;
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.metrics.final_likelihood_verified = true;
    fit.metrics.final_likelihood_recheck_delta = 1.0;   // score disagrees with re-evaluation
    CHECK(!fit.certifiedForSelection(&reason));

    fit = passingFit();
    fit.metrics.final_likelihood_verified = true;
    fit.metrics.final_likelihood_recheck_delta =
        std::numeric_limits<double>::quiet_NaN();
    CHECK(!fit.certifiedForSelection(&reason));
}

} // namespace

int main() {
    testStatusRoundTrip();
    testCanonicalization();
    testInvalidPoints();
    testSerializationAndDigest();
    testFailClosedCertification();
    if (failures != 0) {
        std::cerr << failures << " freeratefit unit test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "freeratefit_unit: all tests passed\n";
    return EXIT_SUCCESS;
}
