#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

// OZA_HOST_DEVICE is defined in crt_math.hpp
#include "common/crt_math.hpp"

namespace ozablas {
namespace pipeline {

inline int calculate_scheme1_beta(int n) {
    int beta = static_cast<int>(std::floor((31.0 - std::log2(static_cast<double>(n))) / 2.0));
    return std::clamp(beta, 1, 7);
}

inline int32_t calculate_scheme2_k_param(double M_prod, int Q) {
    double frac = (M_prod / 2.0 - 1.0) / static_cast<double>(Q);
    return static_cast<int32_t>(std::floor(0.5 * std::log2(frac)));
}

template <int WarpSize>
__global__ void compute_row_shifts_A(
    const double* __restrict__ A,
    int rows,
    int cols,
    int32_t offset,
    int32_t* __restrict__ shifts)
{
    extern __shared__ int shared_max[];

    int row = blockIdx.x;
    if (row >= rows) return;

    int local_max = -1024;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        double v = fabs(A[row * cols + col]);
        if (v > 0.0) {
            // Reverted to your exact benchmark math
            double lg = log2(v);
            int e = static_cast<int>(ceil(lg));
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

        if (threadIdx.x == 0) {
            shifts[row] = offset - local_max;
        }
    }
}

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
            // Reverted to your exact benchmark math
            double lg = log2(v);
            int e = static_cast<int>(ceil(lg));
            local_max = max(local_max, e);
        }
    }

    shifts[col] = offset - local_max;
}

} // namespace pipeline
} // namespace ozablas