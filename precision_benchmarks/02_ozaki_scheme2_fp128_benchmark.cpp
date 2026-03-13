#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <chrono>

#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"

// User Utilities
#include "matrix_gen.hpp"
#include "matrix_io.hpp"
#include "matrix_compare.hpp"
#include "matrix_reference.hpp" // The new FP128 ground truth generator

#ifdef OZA_BUILD_CUDA
#include <cublas_v2.h>
#endif

#ifdef OZA_BUILD_HIP
#include <rocblas/rocblas.h>
#endif

int main() {
    // Note: FP128 CPU math is extremely computationally heavy.
    // M=1024 takes a few seconds. M=4096 or 8192 will take several minutes to an hour on CPU.
    // Adjust these dimensions based on how much patience you have!
    const size_t M = 1024;
    const size_t N = 1024;
    const size_t K = 1024;
    const double phi = 4.0; // Extreme dynamic range to force FP64 errors

    std::cout << "========================================================\n";
    std::cout << " OzaBLAS Scheme II vs FP128 Ground Truth (" << M << "x" << N << ")\n";
    std::cout << "========================================================\n\n";

    // 1. Generate random matrices
    std::cout << "[1/4] Generating random matrices A and B (phi=" << phi << ")...\n";
    auto h_A = matrix_utils::generation::generate_ozaki_matrix<double>(M, K, phi, 42);
    auto h_B = matrix_utils::generation::generate_ozaki_matrix<double>(K, N, phi, 43);

    std::vector<double> h_C_ref(M * N, 0.0);
    std::vector<double> h_C_ozaki(M * N, 0.0);

    // 2. Compute the exact mathematically perfect answer on the CPU using 128-bit precision
    std::cout << "\n[2/4] Computing true mathematical answer (FP128) on CPU...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();

    std::vector<double> h_C_exact = matrix_utils::reference::compute_exact_gemm_fp128(h_A, h_B, M, N, K);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    std::cout << "      Done in " << diff_cpu.count() << " seconds.\n\n";

    // 3. Initialize Hardware Executor
    std::shared_ptr<ozablas::Executor> exec;
#ifdef OZA_BUILD_CUDA
    std::cout << "[3/4] Initializing NVIDIA CUDA Executor...\n";
    exec = std::make_shared<ozablas::CudaExecutor>(0);
#elif defined(OZA_BUILD_HIP)
    std::cout << "[3/4] Initializing AMD HIP Executor...\n";
    exec = std::make_shared<ozablas::HipExecutor>(0);
#else
    std::cerr << "Error: OzaBLAS must be built with either CUDA or HIP enabled.\n";
    return 1;
#endif

    double *d_A, *d_B, *d_C_ref, *d_C_ozaki;
    exec->allocate((void**)&d_A, M * K * sizeof(double));
    exec->allocate((void**)&d_B, K * N * sizeof(double));
    exec->allocate((void**)&d_C_ref, M * N * sizeof(double));
    exec->allocate((void**)&d_C_ozaki, M * N * sizeof(double));

    exec->copy_from_host(d_A, h_A.data(), M * K * sizeof(double));
    exec->copy_from_host(d_B, h_B.data(), K * N * sizeof(double));

    // 4. Compute Reference GEMM using Native Vendor BLAS
    std::cout << "\n[4/4] Computing native GPU reference (FP64)...\n";
    const double alpha = 1.0;
    const double beta = 0.0;

#ifdef OZA_BUILD_CUDA
    cublasHandle_t handle = static_cast<cublasHandle_t>(exec->get_blas_handle());
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_B, N, d_A, K, &beta, d_C_ref, N);
#elif defined(OZA_BUILD_HIP)
    rocblas_handle handle = static_cast<rocblas_handle>(exec->get_blas_handle());
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N);
#endif

    exec->synchronize();
    exec->copy_to_host(h_C_ref.data(), d_C_ref, M * N * sizeof(double));

    // Evaluate Vendor BLAS limitations
    std::cout << "\n========================================================\n";
    std::cout << " NATIVE VENDOR BLAS (FP64) vs FP128 EXACT\n";
    std::cout << "========================================================\n";
    auto ref_metrics = matrix_utils::compare::compute_errors(h_C_exact, h_C_ref);
    matrix_utils::compare::print_metrics(ref_metrics);

    // 5. Test Ozaki Scheme II at various slice counts against the FP128 EXACT
    std::vector<int> test_slices = {4, 6, 8, 10, 12, 14, 16, 18, 20};

    std::cout << "\n========================================================\n";
    std::cout << " OZAKI SCHEME II (INT8 TC) vs FP128 EXACT\n";
    std::cout << "========================================================\n";
    for (int slices : test_slices) {
        std::cout << "\n--- Running Ozaki with " << slices << " slices ---\n";

        exec->copy_from_host(d_C_ozaki, h_C_ozaki.data(), M * N * sizeof(double)); // Zero out
        ozablas::WorkspaceScheme2 ws(exec, M, N, K, slices);

        // Execute
        ozablas::ozaki_scheme2_gemm(ws, d_A, d_B, d_C_ozaki);
        exec->synchronize();

        exec->copy_to_host(h_C_ozaki.data(), d_C_ozaki, M * N * sizeof(double));

        // Compare Ozaki to the EXACT FP128 matrix, not the Vendor BLAS matrix
        auto ozaki_metrics = matrix_utils::compare::compute_errors(h_C_exact, h_C_ozaki);

        std::cout << "Max Relative Error:       " << std::scientific << ozaki_metrics.max_relative_error << "\n";
        std::cout << "Relative Frobenius Error: " << std::scientific << ozaki_metrics.relative_frobenius_error << "\n";
    }

    // Cleanup
    exec->free(d_A);
    exec->free(d_B);
    exec->free(d_C_ref);
    exec->free(d_C_ozaki);

    std::cout << "\nDone!\n";
    return 0;
}
