#pragma once

#include <cstdint>
#include <cmath>

#include "common/crt_math.hpp"
#include "common/crt_tables.hpp"
#include "pipeline/constants.hpp"

namespace ozablas {
namespace pipeline {

// =============================================================================
// GLOBAL CONSTANT MEMORY DECLARATIONS
// =============================================================================

// =============================================================================
// SCHEME I RECONSTRUCTION
// =============================================================================

/**
 * @brief Reconstructs the final FP64 matrix for Scheme I using Group-wise accumulation.
 * Implements Algorithm 7: C = C + diag(mu'') 2^{-beta*g + 2} FP64(C''') diag(nu'')
 */
__global__ void reconstruct_scheme1(
    const int32_t* __restrict__ C_tc,
    const int32_t* __restrict__ neg_exp_A, // From Step 1
    const int32_t* __restrict__ neg_exp_B, // From Step 1
    int rows, int cols, int k_slices, int beta,
    double* __restrict__ C_out)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        double accum = 0.0;

        // Recover original exponents
        int e_i = -neg_exp_A[row];
        int e_j = -neg_exp_B[col];

        // mu''_i = 2^{e_i + 1 - beta} and nu''_j = 2^{e_j + 1 - beta}
        double u_i = ldexp(1.0, e_i + 1 - beta);
        double v_j = ldexp(1.0, e_j + 1 - beta);

        int gemm_idx = 0;

        // Accumulate upper-triangular combinations
        for (int s = 0; s < k_slices; ++s) {
            for (int t = 0; t < k_slices - s; ++t) {
                int32_t val = C_tc[gemm_idx * rows * cols + idx];

                // Calculate 'g' (1-indexed algorithm mapping)
                // If s and t are 0-indexed, g = (s+1) + (t+1) = s + t + 2
                int g = s + t + 2;

                // Scale factor: 2^{-beta * g + 2} * mu''_i * nu''_j
                double scale = ldexp(u_i * v_j, -beta * g + 2);

                accum += static_cast<double>(val) * scale;
                gemm_idx++;
            }
        }

        C_out[idx] = accum;
    }
}

// =============================================================================
// SCHEME II RECONSTRUCTION (CRT)
// =============================================================================

/**
 * @brief Fast-path reconstruction for <= 7 slices using native 64-bit math.
 * Avoids uint256 overhead by leveraging standard hardware registers.
 */
    __global__ void reconstruct_scheme2_leq7(
        const int32_t* __restrict__ C_tc,
        const int32_t* __restrict__ shift_A,
        const int32_t* __restrict__ shift_B,
        int rows, int cols, int slices,
        double* __restrict__ C_out)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

#if defined(__CUDACC__) || defined(__HIPCC__)
        // Read the lowest 64-bits directly from the 256-bit tables
        uint64_t M = c_M_prod_20[slices - 1][0];
        uint64_t M_half = c_M_half_20[slices - 1][0];
#else
        uint64_t M = 1; uint64_t M_half = 0;
#endif

        uint64_t acc = 0;

        for (int s = 0; s < slices; ++s) {
#if defined(__CUDACC__) || defined(__HIPCC__)
            uint32_t m_i = c_moduli_all[s];
            // Read lowest 64-bits of the partial moduli
            uint64_t partial_mod = c_partial_moduli_20[slices - 1][s][0];
            // The modular inverse table is already natively 64-bit
            uint64_t inv = c_mod_inv_20[slices - 1][s];
#else
            uint32_t m_i = 1; uint64_t partial_mod = 1; uint64_t inv = 1;
#endif

            int32_t val = C_tc[s * rows * cols + idx];

            int32_t rem = val % m_i;
            if (rem < 0) rem += m_i;

            uint64_t c_i = (static_cast<uint64_t>(rem) * inv) % m_i;
            uint64_t term = c_i * partial_mod;
            acc = (acc + term) % M;
        }

        double final_f64;
        if (acc >= M_half) {
            final_f64 = -static_cast<double>(M - acc);
        } else {
            final_f64 = static_cast<double>(acc);
        }

        int32_t e = shift_A[row] + shift_B[col];
        C_out[idx] = ldexp(final_f64, -e);
    }
}

/**
 * @brief Heavy-path reconstruction for > 7 slices using 256-bit compound math.
 * Based on the fuse_output_processing kernel.
 */
__global__ void reconstruct_scheme2_gt7(
    const int32_t* __restrict__ C_tc,
    const int32_t* __restrict__ shift_A,
    const int32_t* __restrict__ shift_B,
    int rows, int cols, int slices,
    double* __restrict__ C_out)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        #if defined(__CUDACC__) || defined(__HIPCC__)
        crt::uint256_t M(c_M_prod_20[slices - 1]);
        crt::uint256_t M_half(c_M_half_20[slices - 1]);
        #else
        crt::uint256_t M, M_half;
        #endif

        crt::uint256_t acc; // Initializes to 0

        for (int s = 0; s < slices; ++s) {
            #if defined(__CUDACC__) || defined(__HIPCC__)
            uint32_t m_i = c_moduli_all[s];
            uint64_t inv = c_mod_inv_20[slices - 1][s];
            const uint64_t* partial_mod_ptr = c_partial_moduli_20[slices - 1][s];
            #else
            uint32_t m_i = 1; uint64_t inv = 1; const uint64_t partial_mod_ptr[4] = {0};
            #endif

            int32_t val = C_tc[s * rows * cols + idx];

            int32_t rem = val % m_i;
            if (rem < 0) rem += m_i;

            uint64_t c_i = (static_cast<uint64_t>(rem) * inv) % m_i;

            // Multiply partial_mod by c_i into a 256-bit term
            crt::uint256_t term;
            uint64_t carry = 0;
            for (int i = 0; i < 4; ++i) {
                unsigned __int128 p = (unsigned __int128)partial_mod_ptr[i] * c_i + carry;
                term.data[i] = (uint64_t)p;
                carry = (uint64_t)(p >> 64);
            }

            crt::add_256(acc, term);

            // Modulo reduction: if acc >= M, acc -= M
            if (crt::cmp_ge_256(acc, M)) {
                crt::sub_256(acc, M);
            }
        }

        // Symmetric signed mapping
        double final_f64;
        if (crt::cmp_ge_256(acc, M_half)) {
            crt::uint256_t diff = M;
            crt::sub_256(diff, acc);
            final_f64 = -crt::to_double_256(diff);
        } else {
            final_f64 = crt::to_double_256(acc);
        }

        int32_t e = shift_A[row] + shift_B[col];
        C_out[idx] = ldexp(final_f64, -e);
    }
}

} // namespace pipeline
} // namespace ozablas