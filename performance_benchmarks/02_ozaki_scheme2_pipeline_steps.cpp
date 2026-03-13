#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"
#include "matrix_gen.hpp"

int main() {
    std::vector<size_t> sizes = {1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384};
    std::vector<int> test_slices = {3, 4, 5, 6, 7, 8}; // Test multiple slice configurations
    const int WARMUP = 2;
    const int ITERS = 5;

    std::shared_ptr<ozablas::Executor> exec;
#ifdef OZA_BUILD_CUDA
    exec = std::make_shared<ozablas::CudaExecutor>(0);
#elif defined(OZA_BUILD_HIP)
    exec = std::make_shared<ozablas::HipExecutor>(0);
#endif

    std::cout << "Size,Slices,Step1(%),Step2(%),Step3_GEMM(%),Step4(%),Total(ms)\n";

    for (size_t N : sizes) {
        // Generate and copy matrices once per size to save overhead
        auto h_A = matrix_utils::generation::generate_ozaki_matrix<double>(N, N, 1.0, 42);
        auto h_B = matrix_utils::generation::generate_ozaki_matrix<double>(N, N, 1.0, 43);

        double *d_A, *d_B, *d_C;
        exec->allocate((void**)&d_A, N * N * sizeof(double));
        exec->allocate((void**)&d_B, N * N * sizeof(double));
        exec->allocate((void**)&d_C, N * N * sizeof(double));

        exec->copy_from_host(d_A, h_A.data(), N * N * sizeof(double));
        exec->copy_from_host(d_B, h_B.data(), N * N * sizeof(double));

        // Iterate over the requested slice counts
        for (int slices : test_slices) {
            ozablas::WorkspaceScheme2 ws(exec, N, N, N, slices);

            // Warmup (No timing struct passed)
            for (int i = 0; i < WARMUP; i++) {
                ozablas::ozaki_scheme2_gemm(ws, d_A, d_B, d_C);
            }
            exec->synchronize();

            // Benchmark (Passing timing struct)
            ozablas::OzaTimings accum_timings;
            for (int i = 0; i < ITERS; i++) {
                ozablas::OzaTimings iter_timings;
                ozablas::ozaki_scheme2_gemm(ws, d_A, d_B, d_C, &iter_timings);

                accum_timings.step1_ms += iter_timings.step1_ms;
                accum_timings.step2_ms += iter_timings.step2_ms;
                accum_timings.step3_ms += iter_timings.step3_ms;
                accum_timings.step4_ms += iter_timings.step4_ms;
                accum_timings.total_ms += iter_timings.total_ms;
            }

            // Average and calculate percentages
            float avg_total = accum_timings.total_ms / ITERS;
            float p1 = (accum_timings.step1_ms / accum_timings.total_ms) * 100.0f;
            float p2 = (accum_timings.step2_ms / accum_timings.total_ms) * 100.0f;
            float p3 = (accum_timings.step3_ms / accum_timings.total_ms) * 100.0f;
            float p4 = (accum_timings.step4_ms / accum_timings.total_ms) * 100.0f;

            std::cout << N << "," << slices << ","
                      << std::fixed << std::setprecision(2) << p1 << "," << p2 << "," << p3 << "," << p4 << ","
                      << std::fixed << std::setprecision(4) << avg_total << "\n";
        }

        // Free matrices after testing all slices for this size
        exec->free(d_A);
        exec->free(d_B);
        exec->free(d_C);
    }

    return 0;
}
