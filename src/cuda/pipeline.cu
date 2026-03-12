#include "cuda/ozablas.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <memory>

// Include our shared __device__ math and logic
#include "pipeline/step1_statistics.hpp"
#include "pipeline/step2_slicing.hpp"
#include "pipeline/step4_reconstruction.hpp"

namespace ozablas {
namespace cuda {

// =============================================================================
// LAZY INITIALIZATION TRIGGER
// =============================================================================
void initialize_constant_memory() {
    cudaMemcpyToSymbol(pipeline::c_moduli_all, crt::moduli_all, sizeof(crt::moduli_all));

    cudaMemcpyToSymbol(pipeline::c_M_prod_20, crt::M_prod_20, sizeof(crt::M_prod_20));
    cudaMemcpyToSymbol(pipeline::c_M_half_20, crt::M_half_20, sizeof(crt::M_half_20));
    cudaMemcpyToSymbol(pipeline::c_partial_moduli_20, crt::partial_moduli_20, sizeof(crt::partial_moduli_20));
    cudaMemcpyToSymbol(pipeline::c_mod_inv_20, crt::mod_inv_20, sizeof(crt::mod_inv_20));
}

// =============================================================================
// HELPER: GRID CONFIGURATION
// =============================================================================

inline dim3 get_block_dim() { return dim3(16, 16); }
inline dim3 get_grid_dim(size_t rows, size_t cols) {
    return dim3((cols + 15) / 16, (rows + 15) / 16);
}

// =============================================================================
// SCHEME I: SUM AND SCALE PIPELINE
// =============================================================================

void ozaki_scheme1_gemm(WorkspaceScheme1& ws, const double* A, const double* B, double* C) {
    int M = static_cast<int>(ws.get_M());
    int N = static_cast<int>(ws.get_N());
    int K = static_cast<int>(ws.get_K());
    int slices = ws.get_slices();

    dim3 block2D = get_block_dim();
    dim3 gridA = get_grid_dim(M, K);
    dim3 gridB = get_grid_dim(K, N);
    dim3 gridC = get_grid_dim(M, N);

    int threads1D = 256;

    // Step 1: Compute Shifts
    int beta = pipeline::calculate_scheme1_beta(K);

    // NVIDIA WarpSize is 32. Dynamic shared memory: (256 / 32) * sizeof(int)
    size_t shared_mem_bytes = (threads1D / 32) * sizeof(int);
    pipeline::compute_row_shifts_A<32><<<M, threads1D, shared_mem_bytes>>>(
        A, M, K, 0, ws.get_shift_A()
    );

    int blocks_B = (N + threads1D - 1) / threads1D;
    pipeline::compute_col_shifts_B<<<blocks_B, threads1D>>>(
        B, K, N, 0, ws.get_shift_B()
    );

    // Step 2: Extract Slices
    pipeline::slice_scheme1_A<<<gridA, block2D>>>(A, ws.get_shift_A(), M, K, slices, beta, ws.get_A_slices());
    pipeline::slice_scheme1_B<<<gridB, block2D>>>(B, ws.get_shift_B(), K, N, slices, beta, ws.get_B_slices());

    // Step 3: Tensor Core Multiply (Cross-Terms)
    auto exec = std::dynamic_pointer_cast<const CudaExecutor>(ws.get_executor());
    cublasHandle_t handle = static_cast<cublasHandle_t>(exec->get_blas_handle());

    const int32_t alpha = 1, beta_gemm = 0;
    int gemm_idx = 0;

    for (int s = 0; s < slices; ++s) {
        for (int t = 0; t < slices - s; ++t) {
            int8_t* A_t = ws.get_A_slices() + s * M * K;
            int8_t* B_t = ws.get_B_slices() + t * K * N;
            int32_t* C_t = ws.get_C_tc() + gemm_idx * M * N;

            // Row-major trick: C^T = B^T * A^T
            cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B_t, CUDA_R_8I, N,
                A_t, CUDA_R_8I, K, &beta_gemm,
                C_t, CUDA_R_32I, N,
                CUDA_R_32I, CUBLAS_GEMM_DEFAULT
            );
            gemm_idx++;
        }
    }

    // Step 4: Group-wise Accumulation
    pipeline::reconstruct_scheme1<<<gridC, block2D>>>(
        ws.get_C_tc(), ws.get_shift_A(), ws.get_shift_B(), M, N, slices, beta, C
    );
}

// =============================================================================
// SCHEME II: CRT PIPELINE
// =============================================================================

void ozaki_scheme2_gemm(WorkspaceScheme2& ws, const double* A, const double* B, double* C) {
    int M = static_cast<int>(ws.get_M());
    int N = static_cast<int>(ws.get_N());
    int K = static_cast<int>(ws.get_K());
    int slices = ws.get_slices();

    dim3 block2D = get_block_dim();
    dim3 gridA = get_grid_dim(M, K);
    dim3 gridB = get_grid_dim(K, N);
    dim3 gridC = get_grid_dim(M, N);
    int threads1D = 256;

    // Step 1: Compute Shifts
    crt::uint256_t M_prod(crt::M_prod_20[slices - 1]);
    double M_double = crt::to_double_256(M_prod);
    int32_t K_param = pipeline::calculate_scheme2_k_param(M_double, K);

    size_t shared_mem_bytes = (threads1D / 32) * sizeof(int);
    pipeline::compute_row_shifts_A<32><<<M, threads1D, shared_mem_bytes>>>(
        A, M, K, K_param, ws.get_shift_A()
    );

    int blocks_B = (N + threads1D - 1) / threads1D;
    pipeline::compute_col_shifts_B<<<blocks_B, threads1D>>>(
        B, K, N, K_param, ws.get_shift_B()
    );

    // Step 2: Symmetric Modulo Slicing
    pipeline::slice_scheme2_A<<<gridA, block2D>>>(A, ws.get_shift_A(), M, K, slices, ws.get_A_slices());
    pipeline::slice_scheme2_B<<<gridB, block2D>>>(B, ws.get_shift_B(), K, N, slices, ws.get_B_slices());

    // Step 3: Strided Batched Tensor Core Multiply
    auto exec = std::dynamic_pointer_cast<const CudaExecutor>(ws.get_executor());
    cublasHandle_t handle = static_cast<cublasHandle_t>(exec->get_blas_handle());

    const int32_t alpha = 1, beta_gemm = 0;

    // Row-major trick: C^T = B^T * A^T
    cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        ws.get_B_slices(), CUDA_R_8I, N, K * N,
        ws.get_A_slices(), CUDA_R_8I, K, M * K, &beta_gemm,
        ws.get_C_tc(), CUDA_R_32I, N, M * N,
        slices, CUDA_R_32I, CUBLAS_GEMM_DEFAULT
    );

    // Step 4: CRT Reconstruction
    if (slices <= 7) {
        pipeline::reconstruct_scheme2_leq7<<<gridC, block2D>>>(
            ws.get_C_tc(), ws.get_shift_A(), ws.get_shift_B(), M, N, slices, C
        );
    } else {
        pipeline::reconstruct_scheme2_gt7<<<gridC, block2D>>>(
            ws.get_C_tc(), ws.get_shift_A(), ws.get_shift_B(), M, N, slices, C
        );
    }
}

} // namespace cuda
} // namespace ozablas
