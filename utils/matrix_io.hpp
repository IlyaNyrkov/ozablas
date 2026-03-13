#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>
#include "matrix_compare.hpp" // Bring in the ErrorMetrics struct

namespace matrix_utils {
namespace io {

    // -----------------------------------------------------------------
    // Helper to print a single value safely (handles int8_t chars and __half)
    // -----------------------------------------------------------------
    template <typename T>
    inline void print_val(const T& val, int width = 12) {
        std::cout << std::setw(width) << std::setprecision(4) << std::scientific
                  << static_cast<double>(val) << " ";
    }

    // -----------------------------------------------------------------
    // 3A. Print 1D Dense Matrix (Assuming Row-Major)
    // -----------------------------------------------------------------
    template <typename T>
    void print_dense(const std::vector<T>& mat, int rows, int cols) {
        std::cout << "--- Dense Matrix (" << rows << "x" << cols << ") ---\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                print_val(mat[i * cols + j]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // -----------------------------------------------------------------
    // 3B. Print Principal Submatrix of a 1D Dense Matrix
    // -----------------------------------------------------------------
    template <typename T>
    void print_dense_submatrix(const std::vector<T>& mat, int total_cols,
                               int start_row, int end_row, int start_col, int end_col) {
        std::cout << "--- Submatrix Rows[" << start_row << ":" << end_row - 1
                  << "] Cols[" << start_col << ":" << end_col - 1 << "] ---\n";

        for (int i = start_row; i < end_row; ++i) {
            for (int j = start_col; j < end_col; ++j) {
                print_val(mat[i * total_cols + j]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // -----------------------------------------------------------------
    // 3C. Print Validation Metrics
    // -----------------------------------------------------------------
    template <typename T>
    void print_error_metrics(const matrix_utils::compare::ErrorMetrics<T>& metrics) {
        std::cout << "--- Accuracy Validation ---\n";
        std::cout << "Max Absolute Error:       " << std::scientific << std::setprecision(8)
                  << static_cast<double>(metrics.max_abs_error) << "\n";
        std::cout << "Frobenius Norm Error:     " << static_cast<double>(metrics.frobenius_error) << "\n";
        std::cout << "Relative Frobenius Error: " << static_cast<double>(metrics.relative_frobenius_error) << "\n";
        std::cout << "---------------------------\n\n";
    }

} // namespace io
} // namespace matrix_utils
