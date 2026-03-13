#include "hip/ozablas.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/step1_statistics.hpp"
#include "pipeline/step2_slicing.hpp"
#include "pipeline/step4_reconstruction.hpp"

namespace ozablas {
namespace hip {

// Helper macro to guarantee constant memory doesn't silently fail
#define CHECK_HIP_SYMBOL(cmd) \
    do { \
        hipError_t err = cmd; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("OzaBLAS Constant Memory Error: ") + hipGetErrorString(err)); \
        } \
    } while(0)


// =============================================================================
// LAZY INITIALIZATION TRIGGER
// =============================================================================
void initialize_constant_memory() {
    CHECK_HIP_SYMBOL(hipMemcpyToSymbol(HIP_SYMBOL(OZA_c_moduli_all), crt::moduli_all, sizeof(crt::moduli_all)));

    CHECK_HIP_SYMBOL(hipMemcpyToSymbol(HIP_SYMBOL(OZA_c_M_prod_20), crt::M_prod_20, sizeof(crt::M_prod_20)));
    CHECK_HIP_SYMBOL(hipMemcpyToSymbol(HIP_SYMBOL(OZA_c_M_half_20), crt::M_half_20, sizeof(crt::M_half_20)));
    CHECK_HIP_SYMBOL(hipMemcpyToSymbol(HIP_SYMBOL(OZA_c_partial_moduli_20), crt::partial_moduli_20, sizeof(crt::partial_moduli_20)));
    CHECK_HIP_SYMBOL(hipMemcpyToSymbol(HIP_SYMBOL(OZA_c_mod_inv_20), crt::mod_inv_20, sizeof(crt::mod_inv_20)));
}

inline dim3 get_block_dim() { return dim3(16, 16); }
inline dim3 get_grid_dim(size_t rows, size_t cols) {
    return dim3((cols + 15) / 16, (rows + 15) / 16);
}

// =============================================================================
// SCHEME I: SUM AND SCALE PIPELINE
// =============================================================================

void ozaki_scheme1_gemm(WorkspaceScheme1& ws, const double* A, const double* B, double* C, OzaTimings* timings) {

    const int MAX_EVENTS = 64; // Safe upper bound for group tracking
    hipEvent_t start, e1, e2;
    hipEvent_t ev_gemm_start[MAX_EVENTS], ev_gemm_end[MAX_EVENTS], ev_recon_end[MAX_EVENTS];

    if (timings) {
        hipEventCreate(&start); hipEventCreate(&e1); hipEventCreate(&e2);
        for(int i = 0; i < MAX_EVENTS; ++i) {
            hipEventCreate(&ev_gemm_start[i]);
            hipEventCreate(&ev_gemm_end[i]);
            hipEventCreate(&ev_recon_end[i]);
        }
        hipEventRecord(start);
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
    hipMemset(C, 0, M * N * sizeof(double));

    pipeline::compute_row_shifts_A<<<M, threads1D>>>(
        A, M, K, 0, ws.get_shift_A()
    );

    int blocks_B = (N + threads1D - 1) / threads1D;
    pipeline::compute_col_shifts_B<<<blocks_B, threads1D>>>(
        B, K, N, 0, ws.get_shift_B()
    );

    if (timings) hipEventRecord(e1);

    // ==========================================
    // Step 2: Extract Slices
    // ==========================================
    pipeline::slice_scheme1_A<<<gridA, block2D>>>(A, ws.get_shift_A(), M, K, slices, beta, ws.get_A_slices());
    pipeline::slice_scheme1_B<<<gridB, block2D>>>(B, ws.get_shift_B(), K, N, slices, beta, ws.get_B_slices());

    if (timings) hipEventRecord(e2);

    // ==========================================
    // Step 3 & 4: Interleaved GEMM and Reconstruction
    // ==========================================
    auto exec = std::dynamic_pointer_cast<const HipExecutor>(ws.get_executor());
    rocblas_handle handle = static_cast<rocblas_handle>(exec->get_blas_handle());

    const int32_t alpha = 1;
    int32_t* C_t = ws.get_C_tc();

    for (int g = 2; g <= slices + 1; ++g) {
        if (timings && g < MAX_EVENTS) hipEventRecord(ev_gemm_start[g]);

        bool first_in_group = true;

        for (int s = 1; s <= g - 1; ++s) {
            int t = g - s;

            if (s > slices || t > slices) continue;

            int8_t* A_t = ws.get_A_slices() + (s - 1) * M * K;
            int8_t* B_t = ws.get_B_slices() + (t - 1) * K * N;

            int32_t beta_gemm = first_in_group ? 0 : 1;

            rocblas_gemm_ex(
                handle, rocblas_operation_none, rocblas_operation_none,
                N, M, K, &alpha,
                B_t, rocblas_datatype_i8_r, N,
                A_t, rocblas_datatype_i8_r, K, &beta_gemm,
                C_t, rocblas_datatype_i32_r, N,
                C_t, rocblas_datatype_i32_r, N,
                rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0
            );
            first_in_group = false;
        }

        if (timings && g < MAX_EVENTS) hipEventRecord(ev_gemm_end[g]);

        pipeline::reconstruct_scheme1_group<<<gridC, block2D>>>(
            C_t, ws.get_shift_A(), ws.get_shift_B(), M, N, g, beta, C
        );

        if (timings && g < MAX_EVENTS) hipEventRecord(ev_recon_end[g]);
    }

    // ==========================================
    // Timing Collection
    // ==========================================
    if (timings) {
        hipEventSynchronize(ev_recon_end[slices + 1]);

        hipEventElapsedTime(&timings->step1_ms, start, e1);
        hipEventElapsedTime(&timings->step2_ms, e1, e2);

        float total_gemm = 0.0f, total_recon = 0.0f;
        for (int g = 2; g <= slices + 1; ++g) {
            if (g >= MAX_EVENTS) break;
            float ms_gemm = 0.0f, ms_recon = 0.0f;
            hipEventElapsedTime(&ms_gemm, ev_gemm_start[g], ev_gemm_end[g]);
            hipEventElapsedTime(&ms_recon, ev_gemm_end[g], ev_recon_end[g]);
            total_gemm += ms_gemm;
            total_recon += ms_recon;
        }

        timings->step3_ms = total_gemm;
        timings->step4_ms = total_recon;
        hipEventElapsedTime(&timings->total_ms, start, ev_recon_end[slices + 1]);

        hipEventDestroy(start); hipEventDestroy(e1); hipEventDestroy(e2);
        for(int i = 0; i < MAX_EVENTS; ++i) {
            hipEventDestroy(ev_gemm_start[i]);
            hipEventDestroy(ev_gemm_end[i]);
            hipEventDestroy(ev_recon_end[i]);
        }
    }
}

// =============================================================================
// SCHEME II: CRT PIPELINE
// =============================================================================

void ozaki_scheme2_gemm(WorkspaceScheme2& ws, const double* A, const double* B, double* C, OzaTimings* timings) {

    hipEvent_t start, e1, e2, e3, e4;
    if (timings) {
        hipEventCreate(&start); hipEventCreate(&e1); hipEventCreate(&e2);
        hipEventCreate(&e3); hipEventCreate(&e4);
        hipEventRecord(start);
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

    if (timings) hipEventRecord(e1);

    // ==========================================
    // Step 2: Symmetric Modulo Slicing
    // ==========================================
    pipeline::slice_scheme2_A<<<gridA, block2D>>>(A, ws.get_shift_A(), M, K, slices, ws.get_A_slices());
    pipeline::slice_scheme2_B<<<gridB, block2D>>>(B, ws.get_shift_B(), K, N, slices, ws.get_B_slices());

    if (timings) hipEventRecord(e2);

    // ==========================================
    // Step 3: Strided Batched Tensor Core Multiply
    // ==========================================
    auto exec = std::dynamic_pointer_cast<const HipExecutor>(ws.get_executor());
    rocblas_handle handle = static_cast<rocblas_handle>(exec->get_blas_handle());

    const int32_t alpha = 1, beta_gemm = 0;

    // Row-major trick: C^T = B^T * A^T
    rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none,
        N, M, K, &alpha,
        ws.get_B_slices(), rocblas_datatype_i8_r, N, K * N,
        ws.get_A_slices(), rocblas_datatype_i8_r, K, M * K, &beta_gemm,
        ws.get_C_tc(), rocblas_datatype_i32_r, N, M * N,
        ws.get_C_tc(), rocblas_datatype_i32_r, N, M * N,
        slices, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0
    );

    if (timings) hipEventRecord(e3);

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
        hipEventRecord(e4);
        hipEventSynchronize(e4);

        hipEventElapsedTime(&timings->step1_ms, start, e1);
        hipEventElapsedTime(&timings->step2_ms, e1, e2);
        hipEventElapsedTime(&timings->step3_ms, e2, e3);
        hipEventElapsedTime(&timings->step4_ms, e3, e4);
        hipEventElapsedTime(&timings->total_ms, start, e4);

        hipEventDestroy(start); hipEventDestroy(e1); hipEventDestroy(e2);
        hipEventDestroy(e3); hipEventDestroy(e4);
    }
}

} // namespace hip
} // namespace ozablas
