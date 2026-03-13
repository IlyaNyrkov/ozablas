#pragma once

#include <cstdint>
#include <cmath>

#include "common/crt_math.hpp"
#include "common/crt_tables.hpp"
#include "pipeline/constants.hpp"

namespace ozablas {
namespace pipeline {

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Core bit-extraction mechanic for Scheme I.
 */
__device__ inline double extract_fp64_rn(double val, double sigma) {
    // aggresive compiler optimizations turn (val + sigma) - sigma into just val
    // for bit slicing to work this needs to be actual computation
    volatile double tmp = val + sigma;
    return tmp - sigma;
}
/**
 * @brief Highly optimized, purely integer symmetric modulo for Scheme II.
 */
__device__ inline int8_t symmetric_mod_8(int64_t a, uint8_t m) {
    int64_t im = static_cast<int64_t>(m);
    int64_t r = a % im;
    int64_t half_m = im >> 1;

    if (r > half_m) r -= im;
    else if (r < -half_m) r += im;

    return static_cast<int8_t>(r);
}

// =============================================================================
// SCHEME I SLICING KERNELS (Sum and Scale)
// =============================================================================

__global__ void slice_scheme1_A(
    const double* __restrict__ A,
    const int32_t* __restrict__ neg_exponents,
    int rows, int cols, int slices, int beta,
    int8_t* __restrict__ A_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double aij = A[idx];
        int e_i = -neg_exponents[row];

        double scale_factor = ldexp(1.0, e_i + 1 - beta);
        double inv_scale_factor = ldexp(1.0, -(e_i + 1 - beta));

        for (int s = 0; s < slices; ++s) {
            double sigma = 0.75 * ldexp(scale_factor, 53 - beta * s);
            double A_s = extract_fp64_rn(aij, sigma);

            double normalized = A_s * inv_scale_factor;
            // FIXED: Removed the minus sign to match Algorithm 8 Line 6 perfectly
            double scaled = ldexp(normalized, beta * s);
            A_slices[s * rows * cols + idx] = static_cast<int8_t>(scaled);
            aij -= A_s;
        }
    }
}

__global__ void slice_scheme1_B(
    const double* __restrict__ B,
    const int32_t* __restrict__ neg_exponents,
    int rows, int cols, int slices, int beta,
    int8_t* __restrict__ B_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double bij = B[idx];
        int e_j = -neg_exponents[col];

        double scale_factor = ldexp(1.0, e_j + 1 - beta);
        double inv_scale_factor = ldexp(1.0, -(e_j + 1 - beta));

        for (int s = 0; s < slices; ++s) {
            double sigma = 0.75 * ldexp(scale_factor, 53 - beta * s);
            double B_s = extract_fp64_rn(bij, sigma);

            double normalized = B_s * inv_scale_factor;
            // FIXED: Removed the minus sign to match Algorithm 8 Line 6 perfectly
            double scaled = ldexp(normalized, beta * s);
            B_slices[s * rows * cols + idx] = static_cast<int8_t>(scaled);
            bij -= B_s;
        }
    }
}

// =============================================================================
// SCHEME II SLICING KERNELS (CRT)
// =============================================================================

/**
 * @brief Fused scaling, truncation, and modulo slicing for Matrix A.
 * Includes adaptive overflow protection for large slice counts (> 18).
 */
__global__ void slice_scheme2_A(
    const double* __restrict__ A,
    const int32_t* __restrict__ shifts,
    int rows, int cols, int slices,
    int8_t* __restrict__ A_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double scaled = ldexp(A[idx], shifts[row]);

        // Protection against INT64 overflow for 19+ slices
        if (fabs(scaled) < 9.0e18) {
            // Fast path: fits perfectly in 64-bit int
            int64_t truncated_val = static_cast<int64_t>(llrint(scaled));
            for (int s = 0; s < slices; ++s) {
                #if defined(__CUDACC__) || defined(__HIPCC__)
                    uint8_t m = OZA_c_moduli_all[s];
                #else
                    uint8_t m = crt::moduli_all[s];
                #endif
                A_slices[s * rows * cols + idx] = symmetric_mod_8(truncated_val, m);
            }
        } else {
            // Heavy path: uses exact floating-point remainder to avoid 64-bit overflow
            double rounded = rint(scaled); // Round to nearest integer exactly
            for (int s = 0; s < slices; ++s) {
                #if defined(__CUDACC__) || defined(__HIPCC__)
                    int32_t m = static_cast<int32_t>(OZA_c_moduli_all[s]);
                #else
                    int32_t m = static_cast<int32_t>(crt::moduli_all[s]);
                #endif

                double rem_f = fmod(rounded, static_cast<double>(m));
                int32_t r = static_cast<int32_t>(rem_f);

                // Euclidean correction & Symmetric map
                if (r < 0) r += m;
                int32_t half_m = m >> 1;
                if (r > half_m) r -= m;

                A_slices[s * rows * cols + idx] = static_cast<int8_t>(r);
            }
        }
    }
}

/**
 * @brief Fused scaling, truncation, and modulo slicing for Matrix B.
 * Includes adaptive overflow protection for large slice counts (> 18).
 */
__global__ void slice_scheme2_B(
    const double* __restrict__ B,
    const int32_t* __restrict__ shifts,
    int rows, int cols, int slices,
    int8_t* __restrict__ B_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double scaled = ldexp(B[idx], shifts[col]);
        // if slice count > 18, scaling factor is too large and ldexp causes overflow
        if (fabs(scaled) < 9.0e18) {
            int64_t truncated_val = static_cast<int64_t>(llrint(scaled));
            for (int s = 0; s < slices; ++s) {
                #if defined(__CUDACC__) || defined(__HIPCC__)
                    uint8_t m = OZA_c_moduli_all[s];
                #else
                    uint8_t m = crt::moduli_all[s];
                #endif
                B_slices[s * rows * cols + idx] = symmetric_mod_8(truncated_val, m);
            }
        } else {
            double rounded = rint(scaled);
            for (int s = 0; s < slices; ++s) {
                #if defined(__CUDACC__) || defined(__HIPCC__)
                    int32_t m = static_cast<int32_t>(OZA_c_moduli_all[s]);
                #else
                    int32_t m = static_cast<int32_t>(crt::moduli_all[s]);
                #endif

                double rem_f = fmod(rounded, static_cast<double>(m));
                int32_t r = static_cast<int32_t>(rem_f);

                if (r < 0) r += m;
                int32_t half_m = m >> 1;
                if (r > half_m) r -= m;

                B_slices[s * rows * cols + idx] = static_cast<int8_t>(r);
            }
        }
    }
}

} // namespace pipeline
} // namespace ozablas