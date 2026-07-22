#include "freeratefit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <utility>

namespace freerate {
namespace {

const std::uint64_t FNV1A64_OFFSET_BASIS = UINT64_C(14695981039346656037);
const std::uint64_t FNV1A64_PRIME = UINT64_C(1099511628211);

bool finite(double value) {
    return std::isfinite(value) != 0;
}

void setError(std::string *error, const std::string &message) {
    if (error != nullptr) {
        *error = message;
    }
}

std::string quoteJson(const std::string &text) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(text.size() + 2);
    result.push_back('"');
    for (std::string::const_iterator it = text.begin(); it != text.end(); ++it) {
        const unsigned char ch = static_cast<unsigned char>(*it);
        switch (ch) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (ch < 0x20) {
                result += "\\u00";
                result.push_back(hex[(ch >> 4) & 0x0f]);
                result.push_back(hex[ch & 0x0f]);
            } else {
                result.push_back(static_cast<char>(ch));
            }
        }
    }
    result.push_back('"');
    return result;
}

std::string stableDouble(double value) {
    if (std::isnan(value)) {
        return quoteJson("NaN");
    }
    if (std::isinf(value)) {
        return quoteJson(value > 0.0 ? "+Inf" : "-Inf");
    }
    if (value == 0.0) {
        return "0"; // Canonicalize negative zero.
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value;
    return out.str();
}

std::string stableBool(bool value) {
    return value ? "true" : "false";
}

template <class T>
std::string stableUnsigned(T value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << value;
    return out.str();
}

std::string doubleVectorJson(const std::vector<double> &values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) result += ',';
        result += stableDouble(values[i]);
    }
    result += ']';
    return result;
}

std::string stringVectorJson(const std::vector<std::string> &values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) result += ',';
        result += quoteJson(values[i]);
    }
    result += ']';
    return result;
}

template <class T>
std::vector<T> permuted(const std::vector<T> &input,
                        const std::vector<std::size_t> &order) {
    std::vector<T> output;
    output.reserve(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        output.push_back(input[order[i]]);
    }
    return output;
}

bool blockPasses(const FreeRateBlockMetric &block,
                 double tolerance,
                 const char *name,
                 std::string *reason) {
    if (!block.applicable) {
        return true;
    }
    if (!block.evaluated) {
        setError(reason, std::string(name) + " block was not evaluated");
        return false;
    }
    if (block.evaluations == 0) {
        setError(reason, std::string(name) + " block has no evaluations");
        return false;
    }
    if (!finite(block.improvement_upper_bound) ||
        block.improvement_upper_bound < 0.0 ||
        block.improvement_upper_bound > tolerance) {
        setError(reason, std::string(name) +
            " improvement bound exceeds the likelihood tolerance");
        return false;
    }
    return true;
}

bool pointHasZeroWeight(const FreeRatePoint &point,
                        double tolerance) {
    for (std::size_t i = 0; i < point.weights.size(); ++i) {
        if (point.weights[i] <= tolerance) {
            return true;
        }
    }
    return false;
}

bool pointHasRateCollision(const FreeRatePoint &point,
                           double tolerance) {
    for (std::size_t i = 1; i < point.rates.size(); ++i) {
        const double scale = std::max(1.0,
            std::max(std::fabs(point.rates[i - 1]),
                     std::fabs(point.rates[i])));
        if (std::fabs(point.rates[i] - point.rates[i - 1]) <=
            tolerance * scale) {
            return true;
        }
    }
    return false;
}

bool cryptographicDigestAlgorithm(const std::string &algorithm) {
    // Schema v1 intentionally chooses one representation, avoiding aliases
    // such as SHA-256/sha_256 and algorithm-dependent payload lengths.
    return algorithm == "sha256";
}

bool sha256Hex(const std::string &digest) {
    if (digest.size() != 64) return false;
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const char ch = digest[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
            return false;
    }
    return true;
}

bool gitObjectHex(const std::string &object_id) {
    if (object_id.size() != 40 && object_id.size() != 64) return false;
    for (std::size_t i = 0; i < object_id.size(); ++i) {
        const char ch = object_id[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
            return false;
    }
    return true;
}

bool taggedTraceChecksum(const std::string &digest) {
    const std::string prefix = "fnv1a64-v1:";
    if (digest.size() != prefix.size() + 16 ||
        digest.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    for (std::size_t i = prefix.size(); i < digest.size(); ++i) {
        const char ch = digest[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
            return false;
    }
    return true;
}

bool nearBound(double value, double bound, double tolerance) {
    const double scale = std::max(1.0,
        std::max(std::fabs(value), std::fabs(bound)));
    return std::fabs(value - bound) <= tolerance * scale;
}

std::string blockJson(const FreeRateBlockMetric &block) {
    std::string result = "{";
    result += "\"applicable\":" + stableBool(block.applicable);
    result += ",\"evaluated\":" + stableBool(block.evaluated);
    result += ",\"improvement_upper_bound\":" +
        stableDouble(block.improvement_upper_bound);
    result += ",\"evaluations\":" + stableUnsigned(block.evaluations);
    result += '}';
    return result;
}

std::string boundaryJson(
    const FreeRateBlockMetrics::BoundaryActivity &boundary) {
    std::string result = "{";
    result += "\"zero_weight\":" + stableBool(boundary.zero_weight);
    result += ",\"rate_collision\":" + stableBool(boundary.rate_collision);
    result += ",\"rate_ratio_lower\":" +
        stableBool(boundary.rate_ratio_lower);
    result += ",\"branch_lower\":" + stableBool(boundary.branch_lower);
    result += ",\"branch_upper\":" + stableBool(boundary.branch_upper);
    result += ",\"substitution_parameter\":" +
        stableBool(boundary.substitution_parameter);
    result += ",\"frequency_simplex\":" +
        stableBool(boundary.frequency_simplex);
    result += ",\"pinv_lower\":" + stableBool(boundary.pinv_lower);
    result += ",\"pinv_upper\":" + stableBool(boundary.pinv_upper);
    result += ",\"gauge_interval_lower\":" +
        stableBool(boundary.gauge_interval_lower);
    result += ",\"gauge_interval_upper\":" +
        stableBool(boundary.gauge_interval_upper);
    result += '}';
    return result;
}

} // namespace

const char *freeRateFitStatusName(FreeRateFitStatus status) {
    switch (status) {
    case FreeRateFitStatus::LOCAL_STATIONARY_CERTIFIED:
        return "LOCAL_STATIONARY_CERTIFIED";
    case FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED:
        return "BOUNDARY_LOCAL_STATIONARY_CERTIFIED";
    case FreeRateFitStatus::BOUNDARY_DIRECTIONALLY_TESTED:
        return "BOUNDARY_DIRECTIONALLY_TESTED";
    case FreeRateFitStatus::FALLBACK_CERTIFIED:
        return "FALLBACK_CERTIFIED";
    case FreeRateFitStatus::GRID_CERTIFIED:
        return "GRID_CERTIFIED";
    case FreeRateFitStatus::MAXITER:
        return "MAXITER";
    case FreeRateFitStatus::REJECT_STALL:
        return "REJECT_STALL";
    case FreeRateFitStatus::MULTIBASIN_UNRESOLVED:
        return "MULTIBASIN_UNRESOLVED";
    case FreeRateFitStatus::INFEASIBLE_START:
        return "INFEASIBLE_START";
    case FreeRateFitStatus::GPU_CPU_MISMATCH:
        return "GPU_CPU_MISMATCH";
    case FreeRateFitStatus::NUMERICAL_FAILURE:
        return "NUMERICAL_FAILURE";
    case FreeRateFitStatus::UNSUPPORTED:
        return "UNSUPPORTED";
    case FreeRateFitStatus::LEGACY_UNCERTIFIED:
        return "LEGACY_UNCERTIFIED";
    }
    return "NUMERICAL_FAILURE";
}

std::vector<FreeRateFitStatus> allFreeRateFitStatuses() {
    return {
        FreeRateFitStatus::LOCAL_STATIONARY_CERTIFIED,
        FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED,
        FreeRateFitStatus::BOUNDARY_DIRECTIONALLY_TESTED,
        FreeRateFitStatus::FALLBACK_CERTIFIED,
        FreeRateFitStatus::GRID_CERTIFIED,
        FreeRateFitStatus::MAXITER,
        FreeRateFitStatus::REJECT_STALL,
        FreeRateFitStatus::MULTIBASIN_UNRESOLVED,
        FreeRateFitStatus::INFEASIBLE_START,
        FreeRateFitStatus::GPU_CPU_MISMATCH,
        FreeRateFitStatus::NUMERICAL_FAILURE,
        FreeRateFitStatus::UNSUPPORTED,
        FreeRateFitStatus::LEGACY_UNCERTIFIED
    };
}

bool parseFreeRateFitStatus(const std::string &text,
                            FreeRateFitStatus *status) {
    if (status == nullptr) return false;
    const std::vector<FreeRateFitStatus> statuses = allFreeRateFitStatuses();
    for (std::size_t i = 0; i < statuses.size(); ++i) {
        if (text == freeRateFitStatusName(statuses[i])) {
            *status = statuses[i];
            return true;
        }
    }
    return false;
}

bool isProductionCertifiedStatus(FreeRateFitStatus status) {
    return status == FreeRateFitStatus::LOCAL_STATIONARY_CERTIFIED ||
        status == FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED ||
        status == FreeRateFitStatus::FALLBACK_CERTIFIED;
}

bool FreeRateBlockMetrics::BoundaryActivity::supportBoundary() const {
    return zero_weight || rate_collision;
}

bool FreeRateBlockMetrics::BoundaryActivity::any() const {
    return supportBoundary() || rate_ratio_lower || branch_lower ||
        branch_upper || substitution_parameter || frequency_simplex ||
        pinv_lower || pinv_upper || gauge_interval_lower ||
        gauge_interval_upper;
}

FreeRatePointValidation validateFreeRatePoint(const FreeRatePoint &point,
                                               double tolerance) {
    FreeRatePointValidation result;
    const auto fail = [&result](const std::string &code,
                                const std::string &message) {
        result.code = code;
        result.message = message;
        return result;
    };

    if (!finite(tolerance) || tolerance < 0.0) {
        return fail("INVALID_TOLERANCE", "constraint tolerance is invalid");
    }
    if (point.rates.empty()) {
        return fail("EMPTY_CATEGORIES", "at least one rate category is required");
    }
    if (point.rates.size() != point.weights.size()) {
        return fail("CATEGORY_SIZE_MISMATCH", "rates and weights have different sizes");
    }
    if (!point.category_labels.empty() &&
        point.category_labels.size() != point.rates.size()) {
        return fail("CATEGORY_LABEL_SIZE_MISMATCH",
                    "category labels are not category-aligned");
    }

    std::set<std::string> field_names;
    for (std::size_t field = 0; field < point.category_data.size(); ++field) {
        const NamedCategoryVector &data = point.category_data[field];
        if (data.name.empty() || !field_names.insert(data.name).second) {
            return fail("INVALID_CATEGORY_DATA_NAME",
                        "category data names must be nonempty and unique");
        }
        if (data.values.size() != point.rates.size()) {
            return fail("CATEGORY_DATA_SIZE_MISMATCH",
                        "category data are not category-aligned");
        }
        for (std::size_t i = 0; i < data.values.size(); ++i) {
            if (!finite(data.values[i])) {
                return fail("NONFINITE_CATEGORY_DATA",
                            "category data contain a non-finite value");
            }
        }
    }

    double mass = 0.0;
    double mean = 0.0;
    double negativity = 0.0;
    for (std::size_t i = 0; i < point.rates.size(); ++i) {
        if (!finite(point.rates[i]) || point.rates[i] <= 0.0) {
            return fail("INVALID_RATE", "rates must be finite and positive");
        }
        if (!finite(point.weights[i])) {
            return fail("NONFINITE_WEIGHT", "weights must be finite");
        }
        if (point.weights[i] < 0.0) {
            negativity = std::max(negativity, -point.weights[i]);
        }
        mass += point.weights[i];
        mean += point.weights[i] * point.rates[i];
    }
    result.negativity_residual = negativity;
    if (negativity > tolerance) {
        return fail("NEGATIVE_WEIGHT", "a weight violates nonnegativity");
    }

    if (!finite(point.pinv)) {
        return fail("NONFINITE_PINV", "pinv is non-finite");
    }
    if (point.has_invariant) {
        if (point.pinv < 0.0 || point.pinv >= 1.0) {
            return fail("INVALID_PINV", "pinv must satisfy 0 <= pinv < 1");
        }
    } else if (std::fabs(point.pinv) > tolerance) {
        return fail("UNDECLARED_PINV", "pinv is nonzero without +I semantics");
    }

    const double target_mass = point.has_invariant ? 1.0 - point.pinv : 1.0;
    result.mass_residual = std::fabs(mass - target_mass);
    result.mean_residual = std::fabs(mean - 1.0);
    if (!finite(mass) || result.mass_residual > tolerance) {
        return fail("MASS_CONSTRAINT", "positive-category mass constraint failed");
    }
    if (!finite(mean) || result.mean_residual > tolerance) {
        return fail("MEAN_CONSTRAINT", "unit-mean rate constraint failed");
    }

    if (point.branch_lengths.empty()) {
        return fail("EMPTY_BRANCHES", "physical branch snapshot is empty");
    }
    for (std::size_t i = 0; i < point.branch_lengths.size(); ++i) {
        if (!finite(point.branch_lengths[i]) || point.branch_lengths[i] <= 0.0) {
            return fail("INVALID_BRANCH",
                        "real physical branch lengths must be finite and positive");
        }
    }
    for (std::size_t i = 0; i < point.substitution_rates.size(); ++i) {
        if (!finite(point.substitution_rates[i]) ||
            point.substitution_rates[i] <= 0.0) {
            return fail("INVALID_SUBSTITUTION_RATE",
                        "substitution rates must be finite and positive");
        }
    }
    if (!point.frequencies.empty()) {
        double frequency_sum = 0.0;
        for (std::size_t i = 0; i < point.frequencies.size(); ++i) {
            if (!finite(point.frequencies[i]) || point.frequencies[i] < 0.0) {
                return fail("INVALID_FREQUENCY",
                            "frequencies must be finite and nonnegative");
            }
            frequency_sum += point.frequencies[i];
        }
        if (!finite(frequency_sum) ||
            std::fabs(frequency_sum - 1.0) > tolerance) {
            return fail("FREQUENCY_CONSTRAINT", "frequencies do not sum to one");
        }
    }

    result.valid = true;
    result.code = "OK";
    result.message.clear();
    return result;
}

bool canonicalizeFreeRatePoint(FreeRatePoint *point, std::string *error) {
    const std::vector<std::vector<double> *> none;
    return canonicalizeFreeRatePoint(point, none, error);
}

bool canonicalizeFreeRatePoint(
    FreeRatePoint *point,
    const std::vector<std::vector<double> *> &paired_vectors,
    std::string *error) {
    if (point == nullptr) {
        setError(error, "point is null");
        return false;
    }
    const std::size_t count = point->rates.size();
    if (count == 0 || point->weights.size() != count) {
        setError(error, "rates and weights must have the same nonzero size");
        return false;
    }
    if (!point->category_labels.empty() &&
        point->category_labels.size() != count) {
        setError(error, "category labels are not category-aligned");
        return false;
    }
    for (std::size_t i = 0; i < point->category_data.size(); ++i) {
        if (point->category_data[i].values.size() != count) {
            setError(error, "named category data are not category-aligned");
            return false;
        }
    }
    for (std::size_t i = 0; i < paired_vectors.size(); ++i) {
        if (paired_vectors[i] == nullptr || paired_vectors[i]->size() != count) {
            setError(error, "external paired vector is null or has the wrong size");
            return false;
        }
        if (paired_vectors[i] == &point->rates ||
            paired_vectors[i] == &point->weights ||
            paired_vectors[i] == &point->branch_lengths ||
            paired_vectors[i] == &point->substitution_rates ||
            paired_vectors[i] == &point->frequencies) {
            setError(error, "external paired vectors must not alias point storage");
            return false;
        }
        for (std::size_t field = 0; field < point->category_data.size(); ++field) {
            if (paired_vectors[i] == &point->category_data[field].values) {
                setError(error,
                         "external paired vectors must not alias point storage");
                return false;
            }
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!finite(point->rates[i]) || !finite(point->weights[i])) {
            setError(error, "non-finite rate or weight cannot be canonicalized");
            return false;
        }
    }

    // Work on copies so every validation failure leaves both the point and
    // caller-owned paired diagnostics byte-for-byte unchanged.
    FreeRatePoint candidate = *point;
    std::vector<std::vector<double> > paired_candidates;
    paired_candidates.reserve(paired_vectors.size());
    for (std::size_t i = 0; i < paired_vectors.size(); ++i)
        paired_candidates.push_back(*paired_vectors[i]);

    // Stable category-field order makes serialized maps independent of the
    // order in which diagnostics were registered.
    std::stable_sort(candidate.category_data.begin(), candidate.category_data.end(),
        [](const NamedCategoryVector &left, const NamedCategoryVector &right) {
            return left.name < right.name;
        });

    std::vector<std::size_t> order(count);
    for (std::size_t i = 0; i < count; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&candidate](std::size_t left, std::size_t right) {
            if (candidate.rates[left] != candidate.rates[right])
                return candidate.rates[left] < candidate.rates[right];
            if (candidate.weights[left] != candidate.weights[right])
                return candidate.weights[left] < candidate.weights[right];
            if (!candidate.category_labels.empty() &&
                candidate.category_labels[left] != candidate.category_labels[right])
                return candidate.category_labels[left] < candidate.category_labels[right];
            for (std::size_t field = 0;
                 field < candidate.category_data.size(); ++field) {
                const std::vector<double> &values =
                    candidate.category_data[field].values;
                if (values[left] != values[right])
                    return values[left] < values[right];
            }
            return left < right;
        });

    candidate.rates = permuted(candidate.rates, order);
    candidate.weights = permuted(candidate.weights, order);
    if (!candidate.category_labels.empty())
        candidate.category_labels = permuted(candidate.category_labels, order);
    for (std::size_t field = 0; field < candidate.category_data.size(); ++field)
        candidate.category_data[field].values =
            permuted(candidate.category_data[field].values, order);
    for (std::size_t i = 0; i < paired_candidates.size(); ++i)
        paired_candidates[i] = permuted(paired_candidates[i], order);

    // Remove representation-only negative zeros without changing values.
    for (std::size_t i = 0; i < candidate.weights.size(); ++i)
        if (candidate.weights[i] == 0.0) candidate.weights[i] = 0.0;
    if (candidate.pinv == 0.0) candidate.pinv = 0.0;

    const FreeRatePointValidation validation = validateFreeRatePoint(candidate);
    if (!validation.valid) {
        setError(error, validation.code + ": " + validation.message);
        return false;
    }

    *point = std::move(candidate);
    for (std::size_t i = 0; i < paired_vectors.size(); ++i)
        *paired_vectors[i] = std::move(paired_candidates[i]);
    if (error != nullptr) error->clear();
    return true;
}

bool FreeRateProvenance::complete(std::string *reason) const {
    if (schema_identifier != "iqtree-freerate-fit" ||
        schema_version != FREERATE_FIT_SCHEMA_VERSION) {
        setError(reason, "unsupported provenance schema identifier or version");
        return false;
    }
    if (!gitObjectHex(source_commit) || solver_version.empty() ||
        domain_version.empty() || model_name.empty() ||
        branch_mode.empty() || command_line.empty() ||
        run_identifier.empty() || host.empty() || evaluator_backend.empty() ||
        compiler_identifier.empty() || compiler_version.empty() ||
        build_type.empty() || build_flags.empty()) {
        setError(reason, "required provenance field is empty");
        return false;
    }
    if (!cryptographicDigestAlgorithm(binary_digest_algorithm) ||
        !cryptographicDigestAlgorithm(input_digest_algorithm) ||
        !sha256Hex(binary_digest) || !sha256Hex(alignment_digest) ||
        !sha256Hex(tree_digest) || !sha256Hex(parameter_state_digest) ||
        !sha256Hex(candidate_manifest_digest) ||
        (source_tree_dirty && !sha256Hex(source_diff_digest))) {
        setError(reason, "binary/input provenance requires lowercase SHA-256 hex");
        return false;
    }
    if (thread_count == 0) {
        setError(reason, "thread_count must be nonzero");
        return false;
    }
    if (!finite(runtime_min_branch_length) ||
        !finite(runtime_max_branch_length) ||
        runtime_min_branch_length <= 0.0 ||
        runtime_max_branch_length < runtime_min_branch_length) {
        setError(reason, "runtime branch bounds are invalid");
        return false;
    }
    if (branch_mode != "BRLEN_FIX" && branch_mode != "BRLEN_SCALE" &&
        branch_mode != "BRLEN_OPTIMIZE") {
        setError(reason, "branch mode is not represented by schema v1");
        return false;
    }
    if (evaluator_backend == "CUDA_GPU") {
        if (jolt_ntile == 0 || accelerator.empty() ||
            accelerator_uuid.empty() || cuda_toolkit_version.empty() ||
            cuda_driver_version.empty()) {
            setError(reason, "CUDA backend provenance is incomplete");
            return false;
        }
    } else if (evaluator_backend == "CPU") {
        if (jolt_ntile != 0) {
            setError(reason, "CPU backend must record jolt_ntile as zero");
            return false;
        }
    } else {
        setError(reason, "unknown evaluator backend");
        return false;
    }
    if (reason != nullptr) reason->clear();
    return true;
}

bool FreeRateCertificationThresholds::valid(std::string *reason) const {
    if (!finite(likelihood_gain) || likelihood_gain < 0.0 ||
        likelihood_gain > 0.01 ||
        !finite(weight_gap) || weight_gap < 0.0 ||
        weight_gap > 0.1 * likelihood_gain ||
        !finite(constraint_residual) || constraint_residual < 0.0 ||
        constraint_residual > 1e-12 ||
        !finite(scaled_step) || scaled_step < 0.0 || scaled_step > 1e-6 ||
        // Tied to likelihood_gain, not an independent constant: a restart bar looser than tau_L lets a
        // fit certify while a known-better start exists, which is the one thing the portfolio is for.
        !finite(restart_gain) || restart_gain < 0.0 ||
        restart_gain > likelihood_gain ||
        !finite(cpu_gpu_likelihood_delta) ||
        cpu_gpu_likelihood_delta < 0.0 ||
        cpu_gpu_likelihood_delta > 1e-4 ||
        required_small_cycles < 2) {
        setError(reason, "certification thresholds violate the schema-v1 policy");
        return false;
    }
    if (reason != nullptr) reason->clear();
    return true;
}

bool FreeRateFitResult::certifiedForSelection(std::string *reason) const {
    if (!isProductionCertifiedStatus(status)) {
        setError(reason, "status is not production-certified");
        return false;
    }
    if (!thresholds.valid(reason)) return false;
    if (!provenance.complete(reason)) return false;
    if (!finite(final_likelihood)) {
        setError(reason, "final likelihood is non-finite");
        return false;
    }
    // The published score must itself be corroborated.  Every other gate in this function constrains
    // the point, the block residuals, the provenance or the scope; none of them touches the one number
    // ModelFinder actually ranks on.  Without this check a score copied from a stale, pre-gauge, or
    // wrong-backend evaluation passes everything else and silently corrupts model selection.
    if (!metrics.final_likelihood_verified ||
        !finite(metrics.final_likelihood_recheck_delta) ||
        std::fabs(metrics.final_likelihood_recheck_delta) >
            thresholds.likelihood_gain) {
        setError(reason,
                 "final likelihood is not corroborated by a fresh evaluation "
                 "at the final point");
        return false;
    }
    const FreeRatePointValidation point_validation =
        validateFreeRatePoint(final_point, thresholds.constraint_residual);
    if (!point_validation.valid) {
        setError(reason, "invalid final point: " + point_validation.code);
        return false;
    }
    for (std::size_t i = 1; i < final_point.rates.size(); ++i) {
        if (final_point.rates[i] < final_point.rates[i - 1]) {
            setError(reason, "final point is not in canonical rate order");
            return false;
        }
    }
    if (!scope.optimize_weights || !scope.optimize_rates ||
        !finite(scope.rate_ratio_lower) ||
        !finite(scope.rate_ratio_upper) || scope.rate_ratio_lower <= 0.0 ||
        scope.rate_ratio_lower > scope.rate_ratio_upper ||
        !nearBound(scope.rate_ratio_upper, 1.0,
                   thresholds.constraint_residual) ||
        !finite(scope.pinv_lower) || !finite(scope.pinv_upper) ||
        scope.pinv_lower < 0.0 || scope.pinv_upper > 1.0 ||
        scope.pinv_lower > scope.pinv_upper) {
        setError(reason, "declared fit scope or parameter domain is invalid");
        return false;
    }
    if (scope.weight_formulation != "LITERAL_MASS_MEAN" ||
        final_point.has_invariant || scope.optimize_branches ||
        scope.optimize_substitution || scope.optimize_pinv ||
        provenance.branch_mode != "BRLEN_FIX") {
        setError(reason,
            "schema v1 certifies only pure +R with fixed branches/Q and "
            "the literal mass-and-mean profile");
        return false;
    }
    const bool branch_mode_optimizes =
        provenance.branch_mode == "BRLEN_OPTIMIZE" ||
        provenance.branch_mode == "BRLEN_SCALE";
    if (scope.optimize_branches != branch_mode_optimizes ||
        metrics.rate.applicable != scope.optimize_rates ||
        metrics.branch.applicable != scope.optimize_branches ||
        metrics.substitution.applicable != scope.optimize_substitution ||
        metrics.pinv.applicable != scope.optimize_pinv ||
        (scope.optimize_pinv && !final_point.has_invariant)) {
        setError(reason, "block applicability disagrees with the typed fit scope");
        return false;
    }
    if (scope.optimize_pinv && scope.pinv_upper >= 1.0) {
        setError(reason, "optimized pinv domain includes the singular endpoint");
        return false;
    }
    if (metrics.boundary.substitution_parameter ||
        metrics.boundary.frequency_simplex || metrics.boundary.pinv_lower ||
        metrics.boundary.pinv_upper || metrics.boundary.gauge_interval_lower ||
        metrics.boundary.gauge_interval_upper || metrics.boundary.branch_lower ||
        metrics.boundary.branch_upper) {
        setError(reason,
            "schema v1 cannot certify structural or gauge-bound activity");
        return false;
    }

    if (!metrics.weight_profile_evaluated ||
        metrics.weight_profile_evaluations == 0 || !finite(metrics.weight_gap) ||
        metrics.weight_gap < 0.0) {
        setError(reason, "weight profile lacks a passing gap certificate");
        return false;
    }
    // The bound must be one the oracle can justify, and it is the tightest justified bound that has to
    // clear the bar. Testing weight_gap alone did both halves wrong: it accepted a clamped gap from a
    // broken enumeration, and it rejected correct near-degenerate fits whose curvature-aware bound is
    // orders tighter. A second-order bound that covers only the active face is refused here, because a
    // zero-weight category can carry an improving direction the decrement never sees.
    if (!metrics.weight_gap_is_valid_bound) {
        setError(reason,
                 "weight gap is not a valid upper bound; the vertex enumeration "
                 "does not certify this point");
        return false;
    }
    if (metrics.weight_newton_bound_is_global &&
        !(metrics.weight_newton_bound >= 0.0)) {
        setError(reason, "second-order weight bound is negative or non-finite");
        return false;
    }
    if (!finite(metrics.weight_best_bound) || metrics.weight_best_bound < 0.0 ||
        metrics.weight_best_bound > metrics.weight_gap ||
        metrics.weight_best_bound > thresholds.weight_gap) {
        setError(reason, "weight profile lacks a passing gap certificate");
        return false;
    }
    if (!finite(metrics.mass_residual) ||
        !finite(metrics.mean_residual) ||
        !finite(metrics.negativity_residual) ||
        metrics.mass_residual < 0.0 || metrics.mean_residual < 0.0 ||
        metrics.negativity_residual < 0.0 ||
        metrics.mass_residual > thresholds.constraint_residual ||
        metrics.mean_residual > thresholds.constraint_residual ||
        metrics.negativity_residual > thresholds.constraint_residual) {
        setError(reason, "reported feasibility residuals do not pass");
        return false;
    }
    const double report_tolerance = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        0.01 * thresholds.constraint_residual);
    if (std::fabs(metrics.mass_residual - point_validation.mass_residual) >
            report_tolerance ||
        std::fabs(metrics.mean_residual - point_validation.mean_residual) >
            report_tolerance ||
        std::fabs(metrics.negativity_residual -
                  point_validation.negativity_residual) > report_tolerance) {
        setError(reason, "reported residuals disagree with the canonical point");
        return false;
    }
    if (!blockPasses(metrics.rate, thresholds.likelihood_gain, "rate", reason) ||
        !blockPasses(metrics.branch, thresholds.likelihood_gain, "branch", reason) ||
        !blockPasses(metrics.substitution, thresholds.likelihood_gain,
                     "substitution", reason) ||
        !blockPasses(metrics.pinv, thresholds.likelihood_gain,
                     "pinv", reason)) {
        return false;
    }
    if (!metrics.support_events_evaluated ||
        !finite(metrics.best_tested_support_gain) ||
        metrics.best_tested_support_gain < 0.0 ||
        metrics.best_tested_support_gain > thresholds.likelihood_gain) {
        setError(reason, "support-event gate does not pass");
        return false;
    }

    const double bound_tolerance = thresholds.constraint_residual;
    const bool zero_weight = pointHasZeroWeight(final_point, bound_tolerance);
    const bool rate_collision =
        pointHasRateCollision(final_point, bound_tolerance);
    bool rate_ratio_lower = false;
    const double reference_rate = final_point.rates.back();
    for (std::size_t i = 0; i + 1 < final_point.rates.size(); ++i) {
        const double ratio = final_point.rates[i] / reference_rate;
        if (ratio < scope.rate_ratio_lower - bound_tolerance ||
            ratio > scope.rate_ratio_upper + bound_tolerance) {
            setError(reason, "canonical rate ratio lies outside the declared domain");
            return false;
        }
        rate_ratio_lower = rate_ratio_lower ||
            nearBound(ratio, scope.rate_ratio_lower, bound_tolerance);
    }
    bool branch_lower = false;
    bool branch_upper = false;
    for (std::size_t i = 0; i < final_point.branch_lengths.size(); ++i) {
        const double branch = final_point.branch_lengths[i];
        if (branch < provenance.runtime_min_branch_length - bound_tolerance ||
            branch > provenance.runtime_max_branch_length + bound_tolerance) {
            setError(reason, "physical branch lies outside the recorded runtime bounds");
            return false;
        }
        if (scope.optimize_branches) {
            branch_lower = branch_lower || nearBound(branch,
                provenance.runtime_min_branch_length, bound_tolerance);
            branch_upper = branch_upper || nearBound(branch,
                provenance.runtime_max_branch_length, bound_tolerance);
        }
    }
    bool pinv_lower = false;
    bool pinv_upper = false;
    if (scope.optimize_pinv) {
        if (final_point.pinv < scope.pinv_lower - bound_tolerance ||
            final_point.pinv > scope.pinv_upper + bound_tolerance) {
            setError(reason, "pinv lies outside the declared runtime domain");
            return false;
        }
        pinv_lower = nearBound(final_point.pinv, scope.pinv_lower,
                               bound_tolerance);
        pinv_upper = nearBound(final_point.pinv, scope.pinv_upper,
                               bound_tolerance);
    }
    bool frequency_simplex = false;
    if (scope.optimize_substitution) {
        for (std::size_t i = 0; i < final_point.frequencies.size(); ++i)
            frequency_simplex = frequency_simplex ||
                final_point.frequencies[i] <= bound_tolerance;
    }
    if (metrics.boundary.zero_weight != zero_weight ||
        metrics.boundary.rate_collision != rate_collision ||
        metrics.boundary.rate_ratio_lower != rate_ratio_lower ||
        metrics.boundary.branch_lower != branch_lower ||
        metrics.boundary.branch_upper != branch_upper ||
        metrics.boundary.frequency_simplex != frequency_simplex ||
        metrics.boundary.pinv_lower != pinv_lower ||
        metrics.boundary.pinv_upper != pinv_upper) {
        setError(reason, "reported boundary activity disagrees with the point/domain");
        return false;
    }
    const bool boundary = metrics.boundary.any();
    if (status == FreeRateFitStatus::LOCAL_STATIONARY_CERTIFIED && boundary) {
        setError(reason, "interior status was used for a boundary point");
        return false;
    }
    if (status == FreeRateFitStatus::BOUNDARY_LOCAL_STATIONARY_CERTIFIED &&
        !boundary) {
        setError(reason, "boundary status was used for an interior point");
        return false;
    }
    if (metrics.boundary.supportBoundary() &&
        (!metrics.continuous_insertion_evaluated ||
        !finite(metrics.continuous_insertion_gain_upper_bound) ||
        metrics.continuous_insertion_gain_upper_bound < 0.0 ||
        metrics.continuous_insertion_gain_upper_bound >
            thresholds.likelihood_gain)) {
        setError(reason, "boundary point lacks continuous insertion pricing");
        return false;
    }

    if (!finite(metrics.profiled_likelihood_change) ||
        std::fabs(metrics.profiled_likelihood_change) >
            thresholds.likelihood_gain ||
        !finite(metrics.max_scaled_step) ||
        metrics.max_scaled_step < 0.0 ||
        metrics.max_scaled_step > thresholds.scaled_step ||
        metrics.consecutive_small_cycles < thresholds.required_small_cycles) {
        setError(reason, "major-cycle termination tests do not pass");
        return false;
    }
    if (!metrics.restart_portfolio_evaluated ||
        !finite(metrics.best_restart_gain) ||
        metrics.best_restart_gain < 0.0 ||
        metrics.best_restart_gain > thresholds.restart_gain) {
        setError(reason, "restart portfolio does not pass");
        return false;
    }
    if (!metrics.cpu_gpu_parity_evaluated ||
        !finite(metrics.cpu_gpu_likelihood_delta) ||
        std::fabs(metrics.cpu_gpu_likelihood_delta) >
            thresholds.cpu_gpu_likelihood_delta) {
        setError(reason, "fresh CPU/GPU parity does not pass");
        return false;
    }
    // Budget exhaustion is deliberately NOT a disqualifier.  Every residual
    // and gap gate above has already passed by this point, so a capped fit
    // that meets them is converged and the cap is incidental; four simulated
    // cells exhaust the 400-iteration budget while gaining 0.001-0.016 nat
    // over their last ten iterations.  A capped fit whose residuals do not
    // pass is rejected by those gates and reported as MAXITER.
    // metrics.iteration_cap_reached remains recorded and serialized.
    if (metrics.line_search_failed ||
        metrics.unresolved_support_event || metrics.arithmetic_error) {
        setError(reason, "a terminal failure flag is set");
        return false;
    }
    if (starts_attempted == 0 || starts_completed == 0 ||
        starts_completed > starts_attempted) {
        setError(reason, "restart counters are inconsistent");
        return false;
    }
    if (counts.value == 0 || counts.gradient == 0 ||
        counts.cpu_parity == 0 || counts.accepted_steps == 0) {
        setError(reason, "evaluation counters are inconsistent with a certificate");
        return false;
    }
    if (!taggedTraceChecksum(trace_digest)) {
        setError(reason, "trace digest is absent or has the wrong algorithm tag");
        return false;
    }
    if (reason != nullptr) reason->clear();
    return true;
}

FreeRateFitResult FreeRateFitResult::failure(
    FreeRateFitStatus failure_status,
    const std::string &failure_detail) {
    FreeRateFitResult result;
    if (isProductionCertifiedStatus(failure_status) ||
        failure_status == FreeRateFitStatus::GRID_CERTIFIED) {
        result.status = FreeRateFitStatus::NUMERICAL_FAILURE;
        result.detail = "invalid certified status passed to failure(): " +
            std::string(freeRateFitStatusName(failure_status));
        if (!failure_detail.empty()) result.detail += "; " + failure_detail;
    } else {
        result.status = failure_status;
        result.detail = failure_detail;
    }
    return result;
}

std::string stableJson(const FreeRatePoint &point) {
    std::string result = "{";
    result += "\"rates\":" + doubleVectorJson(point.rates);
    result += ",\"weights\":" + doubleVectorJson(point.weights);
    result += ",\"branch_lengths\":" + doubleVectorJson(point.branch_lengths);
    result += ",\"pinv\":" + stableDouble(point.pinv);
    result += ",\"has_invariant\":" + stableBool(point.has_invariant);
    result += ",\"substitution_rates\":" +
        doubleVectorJson(point.substitution_rates);
    result += ",\"frequencies\":" + doubleVectorJson(point.frequencies);
    result += ",\"category_labels\":" + stringVectorJson(point.category_labels);
    result += ",\"category_data\":[";
    for (std::size_t i = 0; i < point.category_data.size(); ++i) {
        if (i != 0) result += ',';
        result += "{\"name\":" + quoteJson(point.category_data[i].name);
        result += ",\"values\":" + doubleVectorJson(point.category_data[i].values);
        result += '}';
    }
    result += "]}";
    return result;
}

std::string stableJson(const FreeRateBlockMetrics &metrics) {
    std::string result = "{";
    result += "\"weight_profile_evaluated\":" +
        stableBool(metrics.weight_profile_evaluated);
    result += ",\"weight_profile_evaluations\":" +
        stableUnsigned(metrics.weight_profile_evaluations);
    result += ",\"weight_gap\":" + stableDouble(metrics.weight_gap);
    result += ",\"weight_gap_is_valid_bound\":" +
        stableBool(metrics.weight_gap_is_valid_bound);
    result += ",\"weight_newton_bound\":" +
        stableDouble(metrics.weight_newton_bound);
    result += ",\"weight_newton_bound_is_global\":" +
        stableBool(metrics.weight_newton_bound_is_global);
    result += ",\"weight_best_bound\":" +
        stableDouble(metrics.weight_best_bound);
    result += ",\"mass_residual\":" + stableDouble(metrics.mass_residual);
    result += ",\"mean_residual\":" + stableDouble(metrics.mean_residual);
    result += ",\"negativity_residual\":" +
        stableDouble(metrics.negativity_residual);
    result += ",\"rate\":" + blockJson(metrics.rate);
    result += ",\"branch\":" + blockJson(metrics.branch);
    result += ",\"substitution\":" + blockJson(metrics.substitution);
    result += ",\"pinv\":" + blockJson(metrics.pinv);
    result += ",\"boundary\":" + boundaryJson(metrics.boundary);
    result += ",\"support_events_evaluated\":" +
        stableBool(metrics.support_events_evaluated);
    result += ",\"best_tested_support_gain\":" +
        stableDouble(metrics.best_tested_support_gain);
    result += ",\"continuous_insertion_evaluated\":" +
        stableBool(metrics.continuous_insertion_evaluated);
    result += ",\"continuous_insertion_gain_upper_bound\":" +
        stableDouble(metrics.continuous_insertion_gain_upper_bound);
    result += ",\"profiled_likelihood_change\":" +
        stableDouble(metrics.profiled_likelihood_change);
    result += ",\"max_scaled_step\":" + stableDouble(metrics.max_scaled_step);
    result += ",\"consecutive_small_cycles\":" +
        stableUnsigned(metrics.consecutive_small_cycles);
    result += ",\"restart_portfolio_evaluated\":" +
        stableBool(metrics.restart_portfolio_evaluated);
    result += ",\"best_restart_gain\":" +
        stableDouble(metrics.best_restart_gain);
    result += ",\"cpu_gpu_parity_evaluated\":" +
        stableBool(metrics.cpu_gpu_parity_evaluated);
    result += ",\"cpu_gpu_likelihood_delta\":" +
        stableDouble(metrics.cpu_gpu_likelihood_delta);
    result += ",\"final_likelihood_verified\":" +
        stableBool(metrics.final_likelihood_verified);
    result += ",\"final_likelihood_recheck_delta\":" +
        stableDouble(metrics.final_likelihood_recheck_delta);
    result += ",\"iteration_cap_reached\":" +
        stableBool(metrics.iteration_cap_reached);
    result += ",\"line_search_failed\":" +
        stableBool(metrics.line_search_failed);
    result += ",\"unresolved_support_event\":" +
        stableBool(metrics.unresolved_support_event);
    result += ",\"arithmetic_error\":" + stableBool(metrics.arithmetic_error);
    result += '}';
    return result;
}

std::string stableJson(const FreeRateEvaluationCounts &counts) {
    std::string result = "{";
    result += "\"value\":" + stableUnsigned(counts.value);
    result += ",\"gradient\":" + stableUnsigned(counts.gradient);
    result += ",\"hessian_vector\":" + stableUnsigned(counts.hessian_vector);
    result += ",\"cpu_parity\":" + stableUnsigned(counts.cpu_parity);
    result += ",\"accepted_steps\":" + stableUnsigned(counts.accepted_steps);
    result += ",\"rejected_steps\":" + stableUnsigned(counts.rejected_steps);
    result += '}';
    return result;
}

std::string stableJson(const FreeRateProvenance &p) {
    std::string result = "{";
    result += "\"schema_identifier\":" + quoteJson(p.schema_identifier);
    result += ",\"schema_version\":" + stableUnsigned(p.schema_version);
    result += ",\"source_commit\":" + quoteJson(p.source_commit);
    result += ",\"source_diff_digest\":" + quoteJson(p.source_diff_digest);
    result += ",\"solver_version\":" + quoteJson(p.solver_version);
    result += ",\"domain_version\":" + quoteJson(p.domain_version);
    result += ",\"binary_digest_algorithm\":" +
        quoteJson(p.binary_digest_algorithm);
    result += ",\"binary_digest\":" + quoteJson(p.binary_digest);
    result += ",\"input_digest_algorithm\":" +
        quoteJson(p.input_digest_algorithm);
    result += ",\"alignment_digest\":" + quoteJson(p.alignment_digest);
    result += ",\"tree_digest\":" + quoteJson(p.tree_digest);
    result += ",\"parameter_state_digest\":" +
        quoteJson(p.parameter_state_digest);
    result += ",\"candidate_manifest_digest\":" +
        quoteJson(p.candidate_manifest_digest);
    result += ",\"model_name\":" + quoteJson(p.model_name);
    result += ",\"branch_mode\":" + quoteJson(p.branch_mode);
    result += ",\"command_line\":" + quoteJson(p.command_line);
    result += ",\"run_identifier\":" + quoteJson(p.run_identifier);
    result += ",\"host\":" + quoteJson(p.host);
    result += ",\"evaluator_backend\":" + quoteJson(p.evaluator_backend);
    result += ",\"accelerator\":" + quoteJson(p.accelerator);
    result += ",\"accelerator_uuid\":" + quoteJson(p.accelerator_uuid);
    result += ",\"compiler_identifier\":" +
        quoteJson(p.compiler_identifier);
    result += ",\"compiler_version\":" + quoteJson(p.compiler_version);
    result += ",\"build_type\":" + quoteJson(p.build_type);
    result += ",\"build_flags\":" + quoteJson(p.build_flags);
    result += ",\"cuda_toolkit_version\":" +
        quoteJson(p.cuda_toolkit_version);
    result += ",\"cuda_driver_version\":" +
        quoteJson(p.cuda_driver_version);
    result += ",\"source_tree_dirty\":" + stableBool(p.source_tree_dirty);
    result += ",\"runtime_min_branch_length\":" +
        stableDouble(p.runtime_min_branch_length);
    result += ",\"runtime_max_branch_length\":" +
        stableDouble(p.runtime_max_branch_length);
    result += ",\"seed\":" + stableUnsigned(p.seed);
    result += ",\"thread_count\":" + stableUnsigned(p.thread_count);
    result += ",\"jolt_ntile\":" + stableUnsigned(p.jolt_ntile);
    result += '}';
    return result;
}

std::string stableJson(const FreeRateStateSnapshot &snapshot) {
    std::string result = "{";
    result += "\"iteration\":" + stableUnsigned(snapshot.iteration);
    result += ",\"point\":" + stableJson(snapshot.point);
    result += ",\"pre_gauge_likelihood\":" +
        stableDouble(snapshot.pre_gauge_likelihood);
    result += ",\"post_gauge_likelihood\":" +
        stableDouble(snapshot.post_gauge_likelihood);
    result += ",\"gauge_likelihood_delta\":" +
        stableDouble(snapshot.gauge_likelihood_delta);
    result += ",\"accepted\":" + stableBool(snapshot.accepted);
    result += ",\"would_clamp\":" + stableBool(snapshot.would_clamp);
    result += ",\"counts\":" + stableJson(snapshot.counts);
    result += ",\"status\":" +
        quoteJson(freeRateFitStatusName(snapshot.status));
    result += '}';
    return result;
}

std::string stableJson(const FreeRateFitResult &fit) {
    std::string certificate_reason;
    const bool certified = fit.certifiedForSelection(&certificate_reason);
    std::string result = "{";
    result += "\"schema_identifier\":\"iqtree-freerate-fit-result\"";
    result += ",\"schema_version\":" +
        stableUnsigned(FREERATE_FIT_SCHEMA_VERSION);
    result += ",\"status\":" + quoteJson(freeRateFitStatusName(fit.status));
    result += ",\"certified_for_selection\":" + stableBool(certified);
    result += ",\"certification_reason\":" + quoteJson(certificate_reason);
    result += ",\"final_likelihood\":" + stableDouble(fit.final_likelihood);
    result += ",\"final_point\":" + stableJson(fit.final_point);
    result += ",\"metrics\":" + stableJson(fit.metrics);
    result += ",\"counts\":" + stableJson(fit.counts);
    result += ",\"provenance\":" + stableJson(fit.provenance);
    result += ",\"thresholds\":{";
    result += "\"likelihood_gain\":" + stableDouble(fit.thresholds.likelihood_gain);
    result += ",\"weight_gap\":" + stableDouble(fit.thresholds.weight_gap);
    result += ",\"constraint_residual\":" +
        stableDouble(fit.thresholds.constraint_residual);
    result += ",\"scaled_step\":" + stableDouble(fit.thresholds.scaled_step);
    result += ",\"restart_gain\":" + stableDouble(fit.thresholds.restart_gain);
    result += ",\"cpu_gpu_likelihood_delta\":" +
        stableDouble(fit.thresholds.cpu_gpu_likelihood_delta);
    result += ",\"required_small_cycles\":" +
        stableUnsigned(fit.thresholds.required_small_cycles);
    result += '}';
    result += ",\"scope\":{";
    result += "\"weight_formulation\":" +
        quoteJson(fit.scope.weight_formulation);
    result += ",\"optimize_weights\":" + stableBool(fit.scope.optimize_weights);
    result += ",\"optimize_rates\":" + stableBool(fit.scope.optimize_rates);
    result += ",\"optimize_branches\":" +
        stableBool(fit.scope.optimize_branches);
    result += ",\"optimize_substitution\":" +
        stableBool(fit.scope.optimize_substitution);
    result += ",\"optimize_pinv\":" + stableBool(fit.scope.optimize_pinv);
    result += ",\"rate_ratio_lower\":" +
        stableDouble(fit.scope.rate_ratio_lower);
    result += ",\"rate_ratio_upper\":" +
        stableDouble(fit.scope.rate_ratio_upper);
    result += ",\"pinv_lower\":" + stableDouble(fit.scope.pinv_lower);
    result += ",\"pinv_upper\":" + stableDouble(fit.scope.pinv_upper);
    result += '}';
    result += ",\"starts_attempted\":" + stableUnsigned(fit.starts_attempted);
    result += ",\"starts_completed\":" + stableUnsigned(fit.starts_completed);
    result += ",\"trace_digest\":" + quoteJson(fit.trace_digest);
    result += ",\"detail\":" + quoteJson(fit.detail);
    result += '}';
    return result;
}

StableTraceDigest::StableTraceDigest() : state_(FNV1A64_OFFSET_BASIS) {}

void StableTraceDigest::reset() {
    state_ = FNV1A64_OFFSET_BASIS;
}

void StableTraceDigest::addByte(std::uint8_t byte) {
    state_ ^= static_cast<std::uint64_t>(byte);
    state_ *= FNV1A64_PRIME;
}

void StableTraceDigest::addRecord(const std::string &record) {
    std::uint64_t length = static_cast<std::uint64_t>(record.size());
    for (unsigned int shift = 0; shift < 64; shift += 8)
        addByte(static_cast<std::uint8_t>((length >> shift) & UINT64_C(0xff)));
    for (std::string::const_iterator it = record.begin(); it != record.end(); ++it)
        addByte(static_cast<std::uint8_t>(static_cast<unsigned char>(*it)));
}

void StableTraceDigest::addSnapshot(const FreeRateStateSnapshot &snapshot) {
    addRecord(stableJson(snapshot));
}

std::uint64_t StableTraceDigest::value() const {
    return state_;
}

std::string StableTraceDigest::str() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "fnv1a64-v1:" << std::hex << std::nouppercase
        << std::setw(16) << std::setfill('0') << state_;
    return out.str();
}

} // namespace freerate
