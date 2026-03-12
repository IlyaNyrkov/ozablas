#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

// OZA_HOST_DEVICE is defined in crt_math.hpp
#include "common/crt_math.hpp"

namespace ozablas {
namespace pipeline {

/**
 * @brief Host-side helper to calculate the bit-budget for Scheme I.
 * Based on: beta = min(7, floor((31 - log2(n))/2))
 */
inline int calculate_scheme1_beta(int n) {
    int beta = static_cast<int>(std::floor((31.0 - std::log2(static_cast<double>(n))) / 2.0));
    return std::clamp(beta, 1, 7);
}

/**
 * @brief Host-side helper to calculate the K parameter for Scheme II.
 * Based on: K = floor(0.5 * log2((M_prod/2 - 1)/Q))
 */
inline int32_t calculate_scheme2_k_param(double M_prod, int Q) {
    double frac = (M_prod / 2.0 - 1.0) / static_cast<double>(Q);
    return static_cast<int32_t>(std::floor(0.5 * std::log2(frac)));
}

/**
 * @brief Step 1: Compute Shifts for Rows of A.
 * Fuses the exponent extraction and shift calculation into a single kernel.
 * Requires dynamic shared memory size during launch: (blockDim.x / WarpSize) * sizeof(int)
 */
template <int WarpSize>
__global__ void compute_row_shifts_A(
    const double* __restrict__ A,
    int rows,
    int cols,
    int32_t offset,
    int32_t* __restrict__ shifts)
{
    // Dynamically sized shared memory to accommodate ANY block size
    extern __shared__ int shared_max[];

    int row = blockIdx.x; // One block per row for high precision statistics
    if (row >= rows) return;

    int local_max = -1024; // Min possible double exponent

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        double v = fabs(A[row * cols + col]);
        if (v > 0.0) {
            int e;
            frexp(v, &e);
            local_max = max(local_max, e);
        }
    }

    // Warp-level reduction
    for (int step = WarpSize / 2; step > 0; step /= 2) {
        #if defined(__CUDACC__)
            local_max = max(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, step));
        #elif defined(__HIPCC__)
            local_max = max(local_max, __shfl_down(local_max, step));
        #endif
    }

    // Block-level reduction via dynamic shared memory
    int lane = threadIdx.x % WarpSize;
    int wid = threadIdx.x / WarpSize;

    if (lane == 0) {
        shared_max[wid] = local_max;
    }
    __syncthreads();

    if (wid == 0) {
        local_max = (threadIdx.x < (blockDim.x / WarpSize)) ? shared_max[lane] : -1024;
        for (int step = WarpSize / 2; step > 0; step /= 2) {
            #if defined(__CUDACC__)
                local_max = max(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, step));
            #elif defined(__HIPCC__)
                local_max = max(local_max, __shfl_down(local_max, step));
            #endif
        }

        // Fused Shift Calculation: Write the final shift directly to memory
        if (threadIdx.x == 0) {
            shifts[row] = offset - local_max;
        }
    }
}

/**
 * @brief Step 1: Compute Shifts for Columns of B.
 * Fuses the exponent extraction and shift calculation into a single kernel.
 * Note: Since this is 1 thread per column, no cross-thread reduction is needed.
 */
__global__ void compute_col_shifts_B(
    const double* __restrict__ B,
    int rows,
    int cols,
    int32_t offset,
    int32_t* __restrict__ shifts)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    int local_max = -1024;
    for (int row = 0; row < rows; ++row) {
        double v = fabs(B[row * cols + col]);
        if (v > 0.0) {
            int e;
            frexp(v, &e);
            local_max = max(local_max, e);
        }
    }

    // Fused Shift Calculation: Write the final shift directly to memory
    shifts[col] = offset - local_max;
}

} // namespace pipeline
} // namespace ozablas