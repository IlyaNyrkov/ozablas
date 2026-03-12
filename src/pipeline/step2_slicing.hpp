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
 * @brief Core bit-extraction mechanic for Scheme I in Round-to-Nearest-Even mode.
 * Extracted into a separate function as requested.
 */
__device__ inline double extract_fp64_rn(double val, double sigma) {
    return (val + sigma) - sigma;
}

/**
 * @brief Highly optimized, purely integer symmetric modulo for Scheme II.
 * Replaces the slow floating-point std::floor implementation.
 */
__device__ inline int8_t symmetric_mod_8(int64_t a, uint8_t m) {
    int64_t im = static_cast<int64_t>(m);
    int64_t r = a % im;
    int64_t half_m = im >> 1;

    // Shift remainder into the symmetric range [-m/2, m/2]
    if (r > half_m) r -= im;
    else if (r < -half_m) r += im;

    return static_cast<int8_t>(r);
}

// =============================================================================
// SCHEME I SLICING KERNELS (Sum and Scale)
// =============================================================================

/**
 * @brief Slices Matrix A using Minamihata's technique for FP64 -> INT8.
 */
__global__ void slice_scheme1_A(
    const double* __restrict__ A,
    const int32_t* __restrict__ neg_exponents, // From Step 1 (offset = 0)
    int rows, int cols, int slices, int beta,
    int8_t* __restrict__ A_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double aij = A[idx];

        // Recover original exponent from Step 1
        int e_i = -neg_exponents[row];

        // mu''_i = 2^{e_i + 1 - beta}
        double scale_factor = ldexp(1.0, e_i + 1 - beta);
        double inv_scale_factor = ldexp(1.0, -(e_i + 1 - beta));

        // Loop sequentially due to residual dependency
        for (int s = 0; s < slices; ++s) {
            // sigma = 0.75 * 2^{53 - beta(s)} * mu''_i
            double sigma = 0.75 * ldexp(scale_factor, 53 - beta * s);

            // Extract in RN mode
            double A_s = extract_fp64_rn(aij, sigma);

            // Convert to INT8
            double normalized = A_s * inv_scale_factor;
            double scaled = ldexp(normalized, -beta * s);
            A_slices[s * rows * cols + idx] = static_cast<int8_t>(scaled);

            // Update residual for the next slice
            aij -= A_s;
        }
    }
}

/**
 * @brief Slices Matrix B using Minamihata's technique for FP64 -> INT8.
 */
__global__ void slice_scheme1_B(
    const double* __restrict__ B,
    const int32_t* __restrict__ neg_exponents, // From Step 1 (offset = 0)
    int rows, int cols, int slices, int beta,
    int8_t* __restrict__ B_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double bij = B[idx];

        // For B, the scaling is per column
        int e_j = -neg_exponents[col];

        double scale_factor = ldexp(1.0, e_j + 1 - beta);
        double inv_scale_factor = ldexp(1.0, -(e_j + 1 - beta));

        for (int s = 0; s < slices; ++s) {
            double sigma = 0.75 * ldexp(scale_factor, 53 - beta * s);
            double B_s = extract_fp64_rn(bij, sigma);

            double normalized = B_s * inv_scale_factor;
            double scaled = ldexp(normalized, -beta * s);
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
 */
__global__ void slice_scheme2_A(
    const double* __restrict__ A,
    const int32_t* __restrict__ shifts, // From Step 1 (offset = K)
    int rows, int cols, int slices,
    int8_t* __restrict__ A_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        // Scale and truncate
        int64_t truncated_val = static_cast<int64_t>(llrint(ldexp(A[idx], shifts[row])));

        // Unlike Scheme I, Scheme II slices are independent and can be unrolled
        for (int s = 0; s < slices; ++s) {
            #if defined(__CUDACC__) || defined(__HIPCC__)
                uint8_t m = c_moduli_all[s]; // Read directly from __constant__ cache
            #else
                uint8_t m = crt::moduli_all[s]; // Fallback for CPU compilation
            #endif

            A_slices[s * rows * cols + idx] = symmetric_mod_8(truncated_val, m);
        }
    }
}

/**
 * @brief Fused scaling, truncation, and modulo slicing for Matrix B.
 */
__global__ void slice_scheme2_B(
    const double* __restrict__ B,
    const int32_t* __restrict__ shifts, // From Step 1 (offset = K)
    int rows, int cols, int slices,
    int8_t* __restrict__ B_slices)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        int64_t truncated_val = static_cast<int64_t>(llrint(ldexp(B[idx], shifts[col])));

        for (int s = 0; s < slices; ++s) {
            #if defined(__CUDACC__) || defined(__HIPCC__)
                uint8_t m = c_moduli_all[s];
            #else
                uint8_t m = crt::moduli_all[s];
            #endif

            B_slices[s * rows * cols + idx] = symmetric_mod_8(truncated_val, m);
        }
    }
}

} // namespace pipeline
} // namespace ozablas
