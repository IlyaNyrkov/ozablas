#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace matrix_utils {
namespace compare {

    template <typename T>
    struct ErrorMetrics {
        T max_abs_error;
        T frobenius_error;
        T relative_frobenius_error;
    };

    // -----------------------------------------------------------------
    // Compute error metrics between a baseline and a custom implementation
    // -----------------------------------------------------------------
    template <typename T>
    ErrorMetrics<T> compute_errors(const std::vector<T>& expected, const std::vector<T>& actual) {
        if (expected.size() != actual.size()) {
            throw std::invalid_argument("Matrix sizes do not match for metric computation.");
        }

        T max_abs = 0.0;
        T sum_sq_diff = 0.0;
        T sum_sq_expected = 0.0;

        for (size_t i = 0; i < expected.size(); ++i) {
            T diff = actual[i] - expected[i];
            T abs_diff = std::abs(diff);

            if (abs_diff > max_abs) {
                max_abs = abs_diff;
            }

            sum_sq_diff += diff * diff;
            sum_sq_expected += expected[i] * expected[i];
        }

        T frob_error = std::sqrt(sum_sq_diff);
        T expected_frob = std::sqrt(sum_sq_expected);

        // Protect against division by zero if the expected matrix is literally all zeros
        T rel_frob_error = (expected_frob > 0) ? (frob_error / expected_frob) : frob_error;

        return {max_abs, frob_error, rel_frob_error};
    }

    // -----------------------------------------------------------------
    // Helper to cleanly print the results to standard output
    // -----------------------------------------------------------------
    template <typename T>
    void print_metrics(const ErrorMetrics<T>& metrics) {
        std::cout << "--- Accuracy Metrics ---\n";
        std::cout << "Max Absolute Error:       " << metrics.max_abs_error << "\n";
        std::cout << "Frobenius Norm Error:     " << metrics.frobenius_error << "\n";
        std::cout << "Relative Frobenius Error: " << metrics.relative_frobenius_error << "\n";
        std::cout << "------------------------\n";
    }

} // namespace metrics
} // namespace matrix_utils