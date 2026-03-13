#pragma once

#include <vector>
#include <iostream>

// Standard compiler extension for 128-bit floats in GCC and Clang
#if defined(__GNUC__) || defined(__clang__)
    typedef __float128 fp128_t;
#else
    #pragma message("FP128 not natively supported, falling back to long double")
    typedef long double fp128_t;
#endif

namespace matrix_utils {
namespace reference {

/**
 * @brief Computes exact matrix multiplication on the CPU using 128-bit precision.
 * * Accumulates the dot product into an FP128 register to completely avoid
 * catastrophic cancellation, then truncates the final perfect result to FP64.
 */
template <typename T>
std::vector<T> compute_exact_gemm_fp128(const std::vector<T>& A, const std::vector<T>& B, size_t M, size_t N, size_t K) {
    std::vector<T> C(M * N, static_cast<T>(0.0));

    std::cout << "[Reference] Transposing B to optimize CPU cache..." << std::endl;
    std::vector<T> B_T(K * N);

    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            B_T[j * K + k] = B[k * N + j];
        }
    }

    std::cout << "[Reference] Computing FP128 exact GEMM (this will take a while for large N)..." << std::endl;

    // Use OpenMP to parallelize the row computations across all CPU cores
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < M; ++i) {

        // Print progress every 512 rows (to know its not stuck)
        if (i % 512 == 0) {
            #pragma omp critical
            {
                std::cout << "  -> Computed " << i << " / " << M << " rows" << std::endl;
            }
        }

        for (size_t j = 0; j < N; ++j) {
            fp128_t sum = 0.0;

            // Because B is transposed, both arrays are read contiguously in memory.
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<fp128_t>(A[i * K + k]) * static_cast<fp128_t>(B_T[j * K + k]);
            }

            // Downcast back to FP64 only at the very end
            C[i * N + j] = static_cast<T>(sum);
        }
    }

    std::cout << "[Reference] FP128 computation complete!" << std::endl;
    return C;
}

} // namespace reference
} // namespace matrix_utils