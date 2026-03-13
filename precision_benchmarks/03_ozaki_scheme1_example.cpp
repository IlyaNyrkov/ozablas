#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"

// User Utilities
#include "matrix_gen.hpp"
#include "matrix_io.hpp"
#include "matrix_compare.hpp"

// Vendor headers for the reference GEMM
#ifdef OZA_BUILD_CUDA
#include <cublas_v2.h>
#endif

#ifdef OZA_BUILD_HIP
#include <rocblas/rocblas.h>
#endif

int main() {
    // Problem dimensions
    const size_t M = 8192;
    const size_t N = 8192;
    const size_t K = 8192;

    std::cout << "==========================================\n";
    std::cout << " OzaBLAS Scheme I Example (" << M << "x" << N << ")\n";
    std::cout << "==========================================\n\n";

    // 1. Generate random matrices
    std::cout << "Generating random matrices A and B...\n";
    auto h_A = matrix_utils::generation::generate_ozaki_matrix<double>(M, K, 4.0, 42);
    auto h_B = matrix_utils::generation::generate_ozaki_matrix<double>(K, N, 4.0, 43);

    std::vector<double> h_C_ref(M * N, 0.0);
    std::vector<double> h_C_ozaki(M * N, 0.0);

    // Print a tiny 3x3 chunk of matrix A to verify generation
    std::cout << "\nMatrix A (Top-Left 3x3):\n";
    matrix_utils::io::print_dense_submatrix(h_A, K, 0, 3, 0, 3);
    std::cout << "\n";

    // 2. Initialize Hardware Executor
    std::shared_ptr<ozablas::Executor> exec;

#ifdef OZA_BUILD_CUDA
    std::cout << "[Backend] Initializing NVIDIA CUDA Executor...\n";
    exec = std::make_shared<ozablas::CudaExecutor>(0);
#elif defined(OZA_BUILD_HIP)
    std::cout << "[Backend] Initializing AMD HIP Executor...\n";
    exec = std::make_shared<ozablas::HipExecutor>(0);
#else
    std::cerr << "Error: OzaBLAS must be built with either CUDA or HIP enabled.\n";
    return 1;
#endif

    // 3. Allocate device memory and copy host data
    double *d_A, *d_B, *d_C_ref, *d_C_ozaki;
    exec->allocate((void**)&d_A, M * K * sizeof(double));
    exec->allocate((void**)&d_B, K * N * sizeof(double));
    exec->allocate((void**)&d_C_ref, M * N * sizeof(double));
    exec->allocate((void**)&d_C_ozaki, M * N * sizeof(double));

    exec->copy_from_host(d_A, h_A.data(), M * K * sizeof(double));
    exec->copy_from_host(d_B, h_B.data(), K * N * sizeof(double));

    // 4. Compute Reference GEMM using Native Vendor BLAS
    std::cout << "Computing native reference GEMM...\n";
    const double alpha = 1.0;
    const double beta = 0.0;

    // Note: C and C++ store matrices in Row-Major format.
    // cuBLAS/rocBLAS expect Column-Major format.
    // The standard math trick to solve this without transposing memory is: C = B * A
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

    // 5. Test Ozaki Scheme I at various slice counts
    // For Scheme I, slices represent the number of mantissa chunks extracted.
    // Max useful slices is roughly ceil(53 / beta).
    std::vector<int> test_slices = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    for (int slices : test_slices) {
        std::cout << "\n==========================================\n";
        std::cout << " Running Scheme I with " << slices << " slices...\n";

        // Zero out the target matrix to prevent state leaking
        exec->copy_from_host(d_C_ozaki, h_C_ozaki.data(), M * N * sizeof(double));

        // Create the workspace specifically for Scheme 1
        ozablas::WorkspaceScheme1 ws(exec, M, N, K, slices);

        // Execute the Scheme 1 pipeline
        ozablas::ozaki_scheme1_gemm(ws, d_A, d_B, d_C_ozaki);
        exec->synchronize();

        // Copy back and compare
        exec->copy_to_host(h_C_ozaki.data(), d_C_ozaki, M * N * sizeof(double));

        auto metrics = matrix_utils::compare::compute_errors(h_C_ref, h_C_ozaki);
        matrix_utils::compare::print_metrics(metrics);

        // Print a tiny chunk of the result to verify it isn't zero
        std::cout << "Output C (Top-Left 2x2):\n";
        matrix_utils::io::print_dense_submatrix(h_C_ozaki, N, 0, 2, 0, 2);
    }

    // 6. Cleanup Raw Pointers
    exec->free(d_A);
    exec->free(d_B);
    exec->free(d_C_ref);
    exec->free(d_C_ozaki);

    std::cout << "\nDone!\n";
    return 0;
}
