#pragma once

// We include the full workspace definition here because the backend
// implementations will need to call ws.get_M(), ws.get_A_slices(), etc.
#include "ozablas/core/workspace.hpp"

namespace ozablas {
    namespace cuda {

        /**
         * @brief Copies the pure C++ constexpr CRT lookup tables into
         * the ultra-fast hardware __constant__ memory cache.
         * This is called exactly once by the WorkspaceScheme2 constructor.
         */
        void initialize_constant_memory();

        /**
         * @brief CUDA backend implementation for Ozaki Scheme I.
         */
        void ozaki_scheme1_gemm(WorkspaceScheme1& ws, const double* A, const double* B, double* C, OzaTimings* timings = nullptr);

        /**
         * @brief CUDA backend implementation for Ozaki Scheme II.
         * Automatically dispatches to 64-bit or 256-bit kernels internally.
         */
        void ozaki_scheme2_gemm(WorkspaceScheme2& ws, const double* A, const double* B, double* C, OzaTimings* timings = nullptr);

    } // namespace cuda
} // namespace ozablas
