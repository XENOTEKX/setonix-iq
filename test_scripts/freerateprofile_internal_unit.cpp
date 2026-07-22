/*
 * White-box tests for the internals of model/freerateprofile.cpp.
 *
 * WHY THIS FILE EXISTS SEPARATELY. The dual-feasibility logic that decides whether the second-order
 * bound may be published as a GLOBAL bound lives in an anonymous namespace, and the states that break it
 * are ones solve() does not reach: it converges to the optimum, where a zero bound is correct. Three
 * successive attempts to test this through the public entry point all passed while the logic was broken
 * -- one declared moment bounds the simplex could never reach, one was solved to optimality in a single
 * iteration, and a randomised sweep never produced the pathological active sets. Mutation testing caught
 * all three, and the fix is to test the function directly rather than to keep guessing at inputs that
 * might steer the solver into the corner.
 *
 * It #includes the .cpp so those internals are callable, so it must be compiled as its OWN translation
 * unit and must not be linked alongside freerateprofile.cpp.
 */

#include "model/freerateprofile.cpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

/** Build the Evaluation the certificate consumes, at a given weight vector. */
freerate_profile::Evaluation evalAt(const freerate_profile::ProfileProblem &problem,
                                    const std::vector<double> &w) {
    freerate_profile::PreparedProblem prepared;
    const freerate_profile::PreparationStatus st =
        freerate_profile::prepareProblem(problem, &prepared);
    require(st == freerate_profile::PreparationStatus::OK,
            "test problem failed preparation");
    std::size_t count = 0;
    return freerate_profile::evaluate(prepared, w, true, &count);
}

double directLnL(const freerate_profile::ProfileProblem &p, const std::vector<double> &w) {
    long double v = 0.0L;
    for (std::size_t i = 0; i < p.pattern_count; ++i) {
        long double m = 0.0L;
        for (std::size_t j = 0; j < p.category_count; ++j)
            m += (long double)w[j] * p.component_likelihood[i * p.category_count + j];
        if (!(m > 0.0L)) return -std::numeric_limits<double>::infinity();
        v += (long double)p.multiplicity[i] * std::log(m);
    }
    return (double)v;
}

/**
 * D1. An ACTIVE MOMENT INEQUALITY must be priced, not silently promoted to a pinning equality.
 *
 * The moment row is added to the face-defining set whenever a quotient bound is active, so every
 * direction leaving that bound is excluded from the decrement. If the multiplier of that inequality is
 * never sign-checked, a point pinned against a bound reports a bound of zero while a feasible competitor
 * sits far higher. For maximisation, r.w <= m_U active requires nu >= 0 and r.w >= m_L active requires
 * nu <= 0; a violated sign IS the improving direction.
 */
void testActiveMomentInequalityIsPriced() {
    freerate_profile::ProfileProblem p;
    p.pattern_count = 2;
    p.category_count = 2;
    p.component_likelihood = {0.01, 0.99, 0.02, 0.98};
    p.multiplicity = {1000.0, 1000.0};
    p.rate = {0.5, 1.5};
    p.geometry = freerate_profile::FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    p.moment_lower = 1.0;
    p.moment_upper = 1.4;

    const std::vector<double> pinned{0.5, 0.5};     // moment == 1.0, the LOWER bound
    const std::vector<double> better{0.1, 0.9};     // moment == 1.4, feasible
    const double here = directLnL(p, pinned);
    const double there = directLnL(p, better);
    require(there > here + 1.0,
            "test precondition lost: the competitor no longer beats the pinned point");

    freerate_profile::ProfileOptions o;
    const freerate_profile::Evaluation e = evalAt(p, pinned);
    const freerate_profile::NewtonCertificate c =
        freerate_profile::computeNewtonCertificate(p, o, pinned, e,
                                                   /*moment_lower_active=*/true,
                                                   /*moment_upper_active=*/false,
                                                   freerate_profile::minMultiplicity(p));

    // The face is fully pinned here (2 unknowns, mass + moment), so the decrement is 0 by construction.
    // That is fine; what must NOT happen is publishing it as a bound over the whole feasible set.
    require(!c.global,
            "a point pinned against an active moment LOWER bound was certified as global while a "
            "feasible competitor scored materially higher");

    // Symmetric case at the upper bound, where the true optimum lies: there the sign condition holds and
    // refusing would gut the certificate.
    const freerate_profile::Evaluation e2 = evalAt(p, better);
    const freerate_profile::NewtonCertificate c2 =
        freerate_profile::computeNewtonCertificate(p, o, better, e2, false, true,
                                                   freerate_profile::minMultiplicity(p));
    require(c2.global,
            "the constrained OPTIMUM at an active upper bound was refused; the sign test is "
            "over-refusing rather than discriminating");
}

/**
 * D3. The multiplier solve must not be trusted when the constraint rows are near-parallel.
 *
 * det of the normal matrix is n^2 * Var(active rates), i.e. the SQUARE of the row conditioning, so it
 * collapses long before underflow. With near-duplicate rates the multipliers become cancellation noise
 * and the reduced cost can come back with the WRONG SIGN -- a real escape direction reported as pricing
 * out. Near-duplicate rates are precisely the over-specified-k regime this workstream studies.
 */
void testIllConditionedMultipliersDoNotCertify() {
    freerate_profile::ProfileProblem p;
    p.pattern_count = 3;
    p.category_count = 3;
    p.component_likelihood = {
        0.1578947368421052,  0.15789473699999995, 0.9,
        0.28947368421052622, 0.28947368478947355, 0.05,
        0.06315789473684208, 0.063157894926315766, 0.62};
    p.multiplicity = {400000.0, 350000.0, 250000.0};
    p.rate = {1.0, 1.0000000010000001, 3.0};
    p.geometry = freerate_profile::FeasibleGeometry::LITERAL_MASS_MEAN;
    p.target_moment = 1.0000000005;

    const std::vector<double> w{0.5, 0.5, 0.0};
    freerate_profile::ProfileOptions o;
    const freerate_profile::Evaluation e = evalAt(p, w);
    const freerate_profile::NewtonCertificate c =
        freerate_profile::computeNewtonCertificate(p, o, w, e, false, false,
                                                   freerate_profile::minMultiplicity(p));

    // Category 2 carries a genuine escape direction here. Whether the arithmetic can see it or not, the
    // certificate must not claim globality off a solve whose conditioning cannot support it.
    require(!c.global,
            "an ill-conditioned multiplier solve (near-duplicate active rates) certified globality; "
            "the reduced cost there is cancellation noise and can carry the wrong sign");
}

/**
 * D4. A category held at a POSITIVE weight below the activity tolerance is invisible to both tests.
 *
 * It is dropped from the reduced Hessian, so the decrement cannot see it, and the reduced-cost loop
 * prices only the direction of INCREASE -- while its improving move is to shed its remaining mass, worth
 * about |d_j| * w_j. The activity tolerance is documented as a reporting knob but silently scales this
 * leak, so it must be bounded explicitly.
 */
void testTinyPositiveWeightSheddingIsBounded() {
    freerate_profile::ProfileProblem p;
    p.pattern_count = 2;
    p.category_count = 3;
    p.component_likelihood = {
        0.60, 0.30, 1e-9,
        0.30, 0.60, 1e-9};
    p.multiplicity = {500000.0, 500000.0};
    p.rate = {0.5, 2.0, 1.0};
    p.geometry = freerate_profile::FeasibleGeometry::LITERAL_MASS_MEAN;
    p.target_moment = 1.25;

    freerate_profile::ProfileOptions o;
    o.active_weight_tolerance = 1e-3;          // the knob the leak scales with
    const std::vector<double> w{0.4994, 0.4997, 0.0009};

    const freerate_profile::Evaluation e = evalAt(p, w);
    const freerate_profile::NewtonCertificate c =
        freerate_profile::computeNewtonCertificate(p, o, w, e, false, false,
                                                   freerate_profile::minMultiplicity(p));

    // Shedding category 2's mass is worth hundreds of nats on this construction.
    std::vector<double> shed{w[0], w[1], 0.0};
    const double s = shed[0] + shed[1];
    shed[0] /= s; shed[1] /= s;
    const double improvement = directLnL(p, shed) - directLnL(p, w);
    if (improvement > 1.0) {
        require(!c.global,
                "a category at a tiny POSITIVE weight was certified as pricing out, but shedding its "
                "mass improves the objective by more than a nat");
    }
}

/** D2. omega*(lambda) is not a bound when any multiplicity is below one. */
void testSubUnitMultiplicityWithholdsTheBound() {
    freerate_profile::ProfileProblem p;
    p.pattern_count = 2;
    p.category_count = 2;
    p.component_likelihood = {8.6253795432219509e-08, 6.7581553809433783e-04,
                              5.5248403613362105e-07, 1.0214045374718468e-02};
    p.multiplicity = {0.01, 0.01};
    p.rate = {0.5, 1.5};
    p.geometry = freerate_profile::FeasibleGeometry::QUOTIENT_MOMENT_INTERVAL;
    p.moment_lower = 0.5;
    p.moment_upper = 1.5;

    const std::vector<double> w{0.96974819185387728, 0.030251808146122716};
    freerate_profile::ProfileOptions o;
    const freerate_profile::Evaluation e = evalAt(p, w);
    const freerate_profile::NewtonCertificate c =
        freerate_profile::computeNewtonCertificate(p, o, w, e, false, false,
                                                   freerate_profile::minMultiplicity(p));
    require(!c.valid,
            "a second-order bound was published for a problem with multiplicity below 1, where "
            "self-concordance and therefore omega*(lambda) do not hold");
}

} // namespace

int main() {
    testActiveMomentInequalityIsPriced();
    testIllConditionedMultipliersDoNotCertify();
    testTinyPositiveWeightSheddingIsBounded();
    testSubUnitMultiplicityWithholdsTheBound();
    std::cout << "freerateprofile_internal_unit: all tests passed\n";
    return 0;
}
