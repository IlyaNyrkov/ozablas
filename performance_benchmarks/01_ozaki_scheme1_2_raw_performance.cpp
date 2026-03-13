#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <chrono>

#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"
#include "matrix_gen.hpp"

#ifdef OZA_BUILD_CUDA
#include <cublas_v2.h>
#endif

#ifdef OZA_BUILD_HIP
#include <rocblas/rocblas.h>
#endif

void run_benchmark() {
    std::vector<size_t> sizes = {1024, 2048, 3072, 4096,
                                 5120, 6144, 7168, 8192,
                                 9216, 10240, 11264, 12288,
                                 13312, 14336, 15360, 16396};
    std::vector<int> slices_to_test = {3, 4, 5, 6, 7, 8};
    const int WARMUP = 2;
    const int ITERS = 5;

    std::shared_ptr<ozablas::Executor> exec;
#ifdef OZA_BUILD_CUDA
    exec = std::make_shared<ozablas::CudaExecutor>(0);
#elif defined(OZA_BUILD_HIP)
    exec = std::make_shared<ozablas::HipExecutor>(0);
#endif

    // Added the Speedup column to the CSV header
    std::cout << "Algorithm,Size,Slices,Time(ms),GFLOPS,Speedup\n";

    for (size_t N : sizes) {
        size_t M = N, K = N;
        double ops = 2.0 * static_cast<double>(M) * N * K;

        auto h_A = matrix_utils::generation::generate_ozaki_matrix<double>(M, K, 1.0, 42);
        auto h_B = matrix_utils::generation::generate_ozaki_matrix<double>(K, N, 1.0, 43);

        double *d_A, *d_B, *d_C;
        exec->allocate((void**)&d_A, M * K * sizeof(double));
        exec->allocate((void**)&d_B, K * N * sizeof(double));
        exec->allocate((void**)&d_C, M * N * sizeof(double));

        exec->copy_from_host(d_A, h_A.data(), M * K * sizeof(double));
        exec->copy_from_host(d_B, h_B.data(), K * N * sizeof(double));

        // ---------------------------------------------------------
        // 1. Native Vendor DGEMM (Baseline)
        // ---------------------------------------------------------
        const double alpha = 1.0, beta = 0.0;

        auto run_native = [&]() {
#ifdef OZA_BUILD_CUDA
            cublasHandle_t handle = static_cast<cublasHandle_t>(exec->get_blas_handle());
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
#elif defined(OZA_BUILD_HIP)
            rocblas_handle handle = static_cast<rocblas_handle>(exec->get_blas_handle());
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
#endif
        };

        for (int i = 0; i < WARMUP; i++) run_native();
        exec->synchronize();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) run_native();
        exec->synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        // Save the baseline execution time to calculate speedups later
        double native_ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;
        double native_gflops = (ops * 1e-9) / (native_ms * 1e-3);

        std::cout << "Native DGEMM," << N << ",-,"
                  << std::fixed << std::setprecision(4) << native_ms << ","
                  << std::fixed << std::setprecision(2) << native_gflops << ","
                  << "1.00x\n";

        // ---------------------------------------------------------
        // 2. Ozaki Scheme I & II
        // ---------------------------------------------------------
        for (int s : slices_to_test) {

            // --- Scheme I ---
            ozablas::WorkspaceScheme1 ws1(exec, M, N, K, s);
            for (int i = 0; i < WARMUP; i++) ozablas::ozaki_scheme1_gemm(ws1, d_A, d_B, d_C);
            exec->synchronize();

            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++) ozablas::ozaki_scheme1_gemm(ws1, d_A, d_B, d_C);
            exec->synchronize();
            end = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;
            double gflops = (ops * 1e-9) / (ms * 1e-3);
            double speedup = native_ms / ms;

            std::cout << "Scheme I," << N << "," << s << ","
                      << std::fixed << std::setprecision(4) << ms << ","
                      << std::fixed << std::setprecision(2) << gflops << ","
                      << std::fixed << std::setprecision(2) << speedup << "x\n";

            // --- Scheme II ---
            ozablas::WorkspaceScheme2 ws2(exec, M, N, K, s);
            for (int i = 0; i < WARMUP; i++) ozablas::ozaki_scheme2_gemm(ws2, d_A, d_B, d_C);
            exec->synchronize();

            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++) ozablas::ozaki_scheme2_gemm(ws2, d_A, d_B, d_C);
            exec->synchronize();
            end = std::chrono::high_resolution_clock::now();

            ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;
            gflops = (ops * 1e-9) / (ms * 1e-3);
            speedup = native_ms / ms;

            std::cout << "Scheme II," << N << "," << s << ","
                      << std::fixed << std::setprecision(4) << ms << ","
                      << std::fixed << std::setprecision(2) << gflops << ","
                      << std::fixed << std::setprecision(2) << speedup << "x\n";
        }

        exec->free(d_A);
        exec->free(d_B);
        exec->free(d_C);
    }
}

int main() {
    run_benchmark();
    return 0;
}