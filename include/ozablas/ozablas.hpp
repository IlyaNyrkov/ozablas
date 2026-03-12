// include/ozablas/ozablas.hpp
#pragma once

namespace ozablas {

    // Forward declarations to keep host compilation lightning fast.
    // Users will include <ozablas/core/workspace.hpp> separately to construct these.
    class WorkspaceScheme1;
    class WorkspaceScheme2;

    /**
     * @brief Computes C = A * B using Ozaki Scheme I (Sum and Scale).
     * * @param ws The pre-allocated workspace containing execution context and dimensions.
     * @param A  Pointer to device memory for matrix A (size M x K).
     * @param B  Pointer to device memory for matrix B (size K x N).
     * @param C  Pointer to device memory for output matrix C (size M x N).
     */
    void ozaki_scheme1_gemm(
        WorkspaceScheme1& ws,
        const double* A,
        const double* B,
        double* C
    );

    /**
     * @brief Computes C = A * B using Ozaki Scheme II (Chinese Remainder Theorem).
     * * Note: The internal dispatcher will automatically route to highly optimized
     * 64-bit kernels for slices <= 7, and 256-bit kernels for slices > 7.
     * * @param ws The pre-allocated workspace containing execution context and dimensions.
     * @param A  Pointer to device memory for matrix A (size M x K).
     * @param B  Pointer to device memory for matrix B (size K x N).
     * @param C  Pointer to device memory for output matrix C (size M x N).
     */
    void ozaki_scheme2_gemm(
        WorkspaceScheme2& ws,
        const double* A,
        const double* B,
        double* C
    );

} // namespace ozablas