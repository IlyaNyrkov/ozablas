#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <chrono>
#include <map>
#include <algorithm>

#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"

// User Utilities
#include "matrix_gen.hpp"
#include "matrix_io.hpp"
#include "matrix_compare.hpp"
#include "matrix_reference.hpp" // The FP128 ground truth generator

#ifdef OZA_BUILD_CUDA
#include <cublas_v2.h>
#endif

#ifdef OZA_BUILD_HIP
#include <rocblas/rocblas.h>
#endif

// Structure to hold data for the final summary report
struct ResultRecord {
    size_t size;
    double phi;
    double vendor_err;
    std::vector<std::pair<int, double>> ozaki_errs;
};

int main() {
    // Configurations
    std::vector<size_t> sizes = {1024};
    std::vector<double> phis = {0.0, 1.0, 2.0, 4.0};
    std::vector<int> test_slices = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    const int ITERS = 5;

    std::shared_ptr<ozablas::Executor> exec;
#ifdef OZA_BUILD_CUDA
    exec = std::make_shared<ozablas::CudaExecutor>(0);
#elif defined(OZA_BUILD_HIP)
    exec = std::make_shared<ozablas::HipExecutor>(0);
#else
    return 1;
#endif

    std::vector<ResultRecord> all_results;

    // Print CSV Header to stdout
    std::cout << "Algorithm,Size,Phi,Slices,MaxRelativeError\n";

    for (size_t N : sizes) {
        size_t M = N, K = N;

        // Allocate device memory once per matrix size to save overhead
        double *d_A, *d_B, *d_C_ref, *d_C_ozaki;
        exec->allocate((void**)&d_A, M * K * sizeof(double));
        exec->allocate((void**)&d_B, K * N * sizeof(double));
        exec->allocate((void**)&d_C_ref, M * N * sizeof(double));
        exec->allocate((void**)&d_C_ozaki, M * N * sizeof(double));

        std::vector<double> h_C_ref(M * N, 0.0);
        std::vector<double> h_C_ozaki(M * N, 0.0);

        for (double phi : phis) {
            double worst_vendor_err = 0.0;
            std::map<int, double> worst_ozaki_err;
            for (int s : test_slices) worst_ozaki_err[s] = 0.0;

            for (int iter = 0; iter < ITERS; ++iter) {
                // 1. Generate NEW random matrices by using iter as an offset for the RNG seed
                auto h_A = matrix_utils::generation::generate_ozaki_matrix<double>(M, K, phi, 42 + iter);
                auto h_B = matrix_utils::generation::generate_ozaki_matrix<double>(K, N, phi, 43 + iter);

                // 2. Compute exact FP128 ground truth on CPU
                auto h_C_exact = matrix_utils::reference::compute_exact_gemm_fp128(h_A, h_B, M, N, K);

                // Copy the newly generated matrices to the GPU device
                exec->copy_from_host(d_A, h_A.data(), M * K * sizeof(double));
                exec->copy_from_host(d_B, h_B.data(), K * N * sizeof(double));

                // 3. Compute Native Vendor BLAS (FP64)
                const double alpha = 1.0, beta_val = 0.0;
#ifdef OZA_BUILD_CUDA
                cublasHandle_t handle = static_cast<cublasHandle_t>(exec->get_blas_handle());
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, d_B, N, d_A, K, &beta_val, d_C_ref, N);
#elif defined(OZA_BUILD_HIP)
                rocblas_handle handle = static_cast<rocblas_handle>(exec->get_blas_handle());
                rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                              N, M, K, &alpha, d_B, N, d_A, K, &beta_val, d_C_ref, N);
#endif
                exec->synchronize();
                exec->copy_to_host(h_C_ref.data(), d_C_ref, M * N * sizeof(double));

                // Track the worst-case error for Native FP64
                auto ref_metrics = matrix_utils::compare::compute_errors(h_C_exact, h_C_ref);
                worst_vendor_err = std::max(worst_vendor_err, ref_metrics.max_relative_error);

                // 4. Ozaki Scheme I Slices Sweep
                for (int slices : test_slices) {
                    exec->copy_from_host(d_C_ozaki, h_C_ozaki.data(), M * N * sizeof(double)); // Zero out the target

                    ozablas::WorkspaceScheme1 ws(exec, M, N, K, slices);
                    ozablas::ozaki_scheme1_gemm(ws, d_A, d_B, d_C_ozaki);
                    exec->synchronize();

                    exec->copy_to_host(h_C_ozaki.data(), d_C_ozaki, M * N * sizeof(double));

                    // Track the worst-case error for this specific slice count
                    auto ozaki_metrics = matrix_utils::compare::compute_errors(h_C_exact, h_C_ozaki);
                    worst_ozaki_err[slices] = std::max(worst_ozaki_err[slices], ozaki_metrics.max_relative_error);
                }
            }

            // Print the worst-case CSV records to stdout for this Phi value
            std::cout << "Native DGEMM," << N << "," << std::fixed << std::setprecision(1) << phi << ",-,"
                      << std::scientific << std::setprecision(6) << worst_vendor_err << "\n";

            ResultRecord record;
            record.size = N;
            record.phi = phi;
            record.vendor_err = worst_vendor_err;

            for (int slices : test_slices) {
                std::cout << "Scheme I," << N << "," << std::fixed << std::setprecision(1) << phi << "," << slices << ","
                          << std::scientific << std::setprecision(6) << worst_ozaki_err[slices] << "\n";
                std::cout << std::flush; // Ensure output streams periodically

                record.ozaki_errs.push_back({slices, worst_ozaki_err[slices]});
            }
            all_results.push_back(record);
        }

        exec->free(d_A);
        exec->free(d_B);
        exec->free(d_C_ref);
        exec->free(d_C_ozaki);
    }

    // =========================================================================
    // Print Summary Report
    // =========================================================================
    std::cout << "\n\n========================================================================\n";
    std::cout << " SUMMARY REPORT: SCHEME I PRECISION CROSSOVER POINTS\n";
    std::cout << "========================================================================\n";
    std::cout << "Size\tPhi\tVendor Error (Max)\tCrossover Slices\tOzaki Error (Max)\n";
    std::cout << "------------------------------------------------------------------------\n";

    for (const auto& res : all_results) {
        int crossover_slice = -1;
        double crossover_err = 0.0;

        // Find the lowest slice count where Ozaki mathematically beats/matches native FP64
        for (const auto& pair : res.ozaki_errs) {
            if (pair.second <= res.vendor_err) {
                crossover_slice = pair.first;
                crossover_err = pair.second;
                break;
            }
        }

        std::cout << res.size << "\t"
                  << std::fixed << std::setprecision(1) << res.phi << "\t"
                  << std::scientific << std::setprecision(4) << res.vendor_err << "\t\t";

        if (crossover_slice != -1) {
            std::cout << crossover_slice << "\t\t\t" << crossover_err << "\n";
        } else {
            std::cout << "None\t\t\tN/A\n";
        }
    }

    return 0;
}