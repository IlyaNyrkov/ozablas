#include "cuda/ozablas.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <memory>

#include "pipeline/step1_statistics.hpp"
#include "pipeline/step2_slicing.hpp"
#include "pipeline/step4_reconstruction.hpp"

namespace ozablas {
namespace cuda {

// =============================================================================
// LAZY INITIALIZATION TRIGGER
// =============================================================================
void initialize_constant_memory() {
    cudaMemcpyToSymbol(OZA_c_moduli_all, crt::moduli_all, sizeof(crt::moduli_all));

    cudaMemcpyToSymbol(OZA_c_M_prod_20, crt::M_prod_20, sizeof(crt::M_prod_20));
    cudaMemcpyToSymbol(OZA_c_M_half_20, crt::M_half_20, sizeof(crt::M_half_20));
    cudaMemcpyToSymbol(OZA_c_partial_moduli_20, crt::partial_moduli_20, sizeof(crt::partial_moduli_20));
    cudaMemcpyToSymbol(OZA_c_mod_inv_20, crt::mod_inv_20, sizeof(crt::mod_inv_20));
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

void ozaki_scheme1_gemm(WorkspaceScheme1& ws, const double* A, const double* B, double* C, OzaTimings* timings) {

    const int MAX_EVENTS = 64; // Safe upper bound for group tracking
    cudaEvent_t start, e1, e2;
    cudaEvent_t ev_gemm_start[MAX_EVENTS], ev_gemm_end[MAX_EVENTS], ev_recon_end[MAX_EVENTS];

    if (timings) {
        cudaEventCreate(&start); cudaEventCreate(&e1); cudaEventCreate(&e2);
        for(int i = 0; i < MAX_EVENTS; ++i) {
            cudaEventCreate(&ev_gemm_start[i]);
            cudaEventCreate(&ev_gemm_end[i]);
            cudaEventCreate(&ev_recon_end[i]);
        }
        cudaEventRecord(start);
    }

    int M = static_cast<int>(ws.get_M());
    int N = static_cast<int>(ws.get_N());
    int K = static_cast<int>(ws.get_K());
    int slices = ws.get_slices();

    dim3 block2D = get_block_dim();
    dim3 gridA = get_grid_dim(M, K);
    dim3 gridB = get_grid_dim(K, N);
    dim3 gridC = get_grid_dim(M, N);

    int threads1D = 256;
    int beta = pipeline::calculate_scheme1_beta(K);

    // ==========================================
    // Step 1: Compute Shifts & Zero Output
    // ==========================================
    cudaMemset(C, 0, M * N * sizeof(double));

    pipeline::compute_row_shifts_A<<<M, threads1D>>>(
        A, M, K, 0, ws.get_shift_A()
    );

    int blocks_B = (N + threads1D - 1) / threads1D;
    pipeline::compute_col_shifts_B<<<blocks_B, threads1D>>>(
        B, K, N, 0, ws.get_shift_B()
    );

    if (timings) cudaEventRecord(e1);

    // ==========================================
    // Step 2: Extract Slices
    // ==========================================
    pipeline::slice_scheme1_A<<<gridA, block2D>>>(A, ws.get_shift_A(), M, K, slices, beta, ws.get_A_slices());
    pipeline::slice_scheme1_B<<<gridB, block2D>>>(B, ws.get_shift_B(), K, N, slices, beta, ws.get_B_slices());

    if (timings) cudaEventRecord(e2);

    // ==========================================
    // Step 3 & 4: Interleaved GEMM and Reconstruction
    // ==========================================
    auto exec = std::dynamic_pointer_cast<const CudaExecutor>(ws.get_executor());
    cublasHandle_t handle = static_cast<cublasHandle_t>(exec->get_blas_handle());

    const int32_t alpha = 1;
    int32_t* C_t = ws.get_C_tc();

    for (int g = 2; g <= slices + 1; ++g) {
        if (timings && g < MAX_EVENTS) cudaEventRecord(ev_gemm_start[g]);

        bool first_in_group = true;

        for (int s = 1; s <= g - 1; ++s) {
            int t = g - s;

            if (s > slices || t > slices) continue;

            int8_t* A_t = ws.get_A_slices() + (s - 1) * M * K;
            int8_t* B_t = ws.get_B_slices() + (t - 1) * K * N;

            int32_t beta_gemm = first_in_group ? 0 : 1;

            cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B_t, CUDA_R_8I, N,
                A_t, CUDA_R_8I, K, &beta_gemm,
                C_t, CUDA_R_32I, N,
                CUDA_R_32I, CUBLAS_GEMM_DEFAULT
            );
            first_in_group = false;
        }

        if (timings && g < MAX_EVENTS) cudaEventRecord(ev_gemm_end[g]);

        pipeline::reconstruct_scheme1_group<<<gridC, block2D>>>(
            C_t, ws.get_shift_A(), ws.get_shift_B(), M, N, g, beta, C
        );

        if (timings && g < MAX_EVENTS) cudaEventRecord(ev_recon_end[g]);
    }

    // ==========================================
    // Timing Collection
    // ==========================================
    if (timings) {
        cudaEventSynchronize(ev_recon_end[slices + 1]);

        cudaEventElapsedTime(&timings->step1_ms, start, e1);
        cudaEventElapsedTime(&timings->step2_ms, e1, e2);

        float total_gemm = 0.0f, total_recon = 0.0f;
        for (int g = 2; g <= slices + 1; ++g) {
            if (g >= MAX_EVENTS) break;
            float ms_gemm = 0.0f, ms_recon = 0.0f;
            cudaEventElapsedTime(&ms_gemm, ev_gemm_start[g], ev_gemm_end[g]);
            cudaEventElapsedTime(&ms_recon, ev_gemm_end[g], ev_recon_end[g]);
            total_gemm += ms_gemm;
            total_recon += ms_recon;
        }

        timings->step3_ms = total_gemm;
        timings->step4_ms = total_recon;
        cudaEventElapsedTime(&timings->total_ms, start, ev_recon_end[slices + 1]);

        cudaEventDestroy(start); cudaEventDestroy(e1); cudaEventDestroy(e2);
        for(int i = 0; i < MAX_EVENTS; ++i) {
            cudaEventDestroy(ev_gemm_start[i]);
            cudaEventDestroy(ev_gemm_end[i]);
            cudaEventDestroy(ev_recon_end[i]);
        }
    }
}

// =============================================================================
// SCHEME II: CRT PIPELINE
// =============================================================================

void ozaki_scheme2_gemm(WorkspaceScheme2& ws, const double* A, const double* B, double* C, OzaTimings* timings) {

    cudaEvent_t start, e1, e2, e3, e4;
    if (timings) {
        cudaEventCreate(&start); cudaEventCreate(&e1); cudaEventCreate(&e2);
        cudaEventCreate(&e3); cudaEventCreate(&e4);
        cudaEventRecord(start);
    }

    int M = static_cast<int>(ws.get_M());
    int N = static_cast<int>(ws.get_N());
    int K = static_cast<int>(ws.get_K());
    int slices = ws.get_slices();

    dim3 block2D = get_block_dim();
    dim3 gridA = get_grid_dim(M, K);
    dim3 gridB = get_grid_dim(K, N);
    dim3 gridC = get_grid_dim(M, N);
    int threads1D = 256;

    // ==========================================
    // Step 1: Compute Shifts
    // ==========================================
    crt::uint256_t M_prod(crt::M_prod_20[slices - 1]);
    double M_double = crt::to_double_256(M_prod);
    int32_t K_param = pipeline::calculate_scheme2_k_param(M_double, K);

    pipeline::compute_row_shifts_A<<<M, threads1D>>>(
        A, M, K, K_param, ws.get_shift_A()
    );

    int blocks_B = (N + threads1D - 1) / threads1D;
    pipeline::compute_col_shifts_B<<<blocks_B, threads1D>>>(
        B, K, N, K_param, ws.get_shift_B()
    );

    if (timings) cudaEventRecord(e1);

    // ==========================================
    // Step 2: Symmetric Modulo Slicing
    // ==========================================
    pipeline::slice_scheme2_A<<<gridA, block2D>>>(A, ws.get_shift_A(), M, K, slices, ws.get_A_slices());
    pipeline::slice_scheme2_B<<<gridB, block2D>>>(B, ws.get_shift_B(), K, N, slices, ws.get_B_slices());

    if (timings) cudaEventRecord(e2);

    // ==========================================
    // Step 3: Strided Batched Tensor Core Multiply
    // ==========================================
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

    if (timings) cudaEventRecord(e3);

    // ==========================================
    // Step 4: CRT Reconstruction
    // ==========================================
    if (slices <= 7) {
        pipeline::reconstruct_scheme2_leq7<<<gridC, block2D>>>(
            ws.get_C_tc(), ws.get_shift_A(), ws.get_shift_B(), M, N, slices, C
        );
    } else {
        pipeline::reconstruct_scheme2_gt7<<<gridC, block2D>>>(
            ws.get_C_tc(), ws.get_shift_A(), ws.get_shift_B(), M, N, slices, C
        );
    }

    if (timings) {
        cudaEventRecord(e4);
        cudaEventSynchronize(e4);

        cudaEventElapsedTime(&timings->step1_ms, start, e1);
        cudaEventElapsedTime(&timings->step2_ms, e1, e2);
        cudaEventElapsedTime(&timings->step3_ms, e2, e3);
        cudaEventElapsedTime(&timings->step4_ms, e3, e4);
        cudaEventElapsedTime(&timings->total_ms, start, e4);

        cudaEventDestroy(start); cudaEventDestroy(e1); cudaEventDestroy(e2);
        cudaEventDestroy(e3); cudaEventDestroy(e4);
    }
}

} // namespace cuda
} // namespace ozablas