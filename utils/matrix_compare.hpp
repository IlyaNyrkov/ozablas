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
        T max_relative_error; // Added Max Relative Error
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

        T max_abs = static_cast<T>(0.0);
        T max_rel = static_cast<T>(0.0);
        T sum_sq_diff = static_cast<T>(0.0);
        T sum_sq_expected = static_cast<T>(0.0);

        for (size_t i = 0; i < expected.size(); ++i) {
            T exp_val = expected[i];
            T act_val = actual[i];

            T diff = act_val - exp_val;
            T abs_diff = std::abs(diff);

            // 1. Max Absolute Error
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
            }

            // 2. Max Relative Error
            T rel_error = (std::abs(exp_val) > static_cast<T>(0.0)) ? (abs_diff / std::abs(exp_val)) : abs_diff;
            if (rel_error > max_rel) {
                max_rel = rel_error;
            }

            // Frobenius tracking
            sum_sq_diff += diff * diff;
            sum_sq_expected += exp_val * exp_val;
        }

        // 3. Frobenius Norm Error
        T frob_error = std::sqrt(sum_sq_diff);
        T expected_frob = std::sqrt(sum_sq_expected);

        // 4. Relative Frobenius Error
        T rel_frob_error = (expected_frob > static_cast<T>(0.0)) ? (frob_error / expected_frob) : frob_error;

        return {max_abs, max_rel, frob_error, rel_frob_error};
    }

    // -----------------------------------------------------------------
    // Helper to cleanly print the results to standard output
    // -----------------------------------------------------------------
    template <typename T>
    void print_metrics(const ErrorMetrics<T>& metrics) {
        std::cout << "--- Accuracy Metrics ---\n";
        std::cout << "Max Absolute Error:       " << metrics.max_abs_error << "\n";
        std::cout << "Max Relative Error:       " << metrics.max_relative_error << "\n";
        std::cout << "Frobenius Norm Error:     " << metrics.frobenius_error << "\n";
        std::cout << "Relative Frobenius Error: " << metrics.relative_frobenius_error << "\n";
        std::cout << "------------------------\n";
    }

} // namespace compare
} // namespace matrix_utils