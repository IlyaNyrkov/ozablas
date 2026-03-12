#include "ozablas/ozablas.hpp"
#include "ozablas/core/executor.hpp"
#include "ozablas/core/workspace.hpp"

#include <stdexcept>

// Include Bridge Headers for hardware backends
#ifdef OZA_BUILD_CUDA
#include "cuda/ozablas.hpp"
#endif

#ifdef OZA_BUILD_HIP
#include "hip/ozablas.hpp"
#endif

namespace ozablas {

// =============================================================================
// OZAKI SCHEME I DISPATCHER
// =============================================================================

void ozaki_scheme1_gemm(WorkspaceScheme1& ws, const double* A, const double* B, double* C) {
    // Extract the executor from the workspace
    auto exec = ws.get_executor();

    // Route to NVIDIA CUDA Backend
    if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec.get())) {
#ifdef OZA_BUILD_CUDA
        cuda::ozaki_scheme1_gemm(ws, A, B, C);
#else
        throw std::runtime_error("OzaBLAS: CUDA backend requested but library was not built with CUDA support.");
#endif
    }
    // Route to AMD HIP Backend
    else if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec.get())) {
#ifdef OZA_BUILD_HIP
        hip::ozaki_scheme1_gemm(ws, A, B, C);
#else
        throw std::runtime_error("OzaBLAS: HIP backend requested but library was not built with HIP support.");
#endif
    }
    // Future CPU fallback could be added here
    else {
        throw std::runtime_error("OzaBLAS: No suitable hardware backend found for the provided Executor.");
    }
}

// =============================================================================
// OZAKI SCHEME II (CRT) DISPATCHER
// =============================================================================

void ozaki_scheme2_gemm(WorkspaceScheme2& ws, const double* A, const double* B, double* C) {
    auto exec = ws.get_executor();

    if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec.get())) {
#ifdef OZA_BUILD_CUDA
        // The CUDA bridge internally handles the S <= 7 vs S > 7 kernel selection
        cuda::ozaki_scheme2_gemm(ws, A, B, C);
#else
        throw std::runtime_error("OzaBLAS: CUDA backend requested but library was not built with CUDA support.");
#endif
    }
    else if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec.get())) {
#ifdef OZA_BUILD_HIP
        // The HIP bridge internally handles the S <= 7 vs S > 7 kernel selection
        hip::ozaki_scheme2_gemm(ws, A, B, C);
#else
        throw std::runtime_error("OzaBLAS: HIP backend requested but library was not built with HIP support.");
#endif
    }
    else {
        throw std::runtime_error("OzaBLAS: No suitable hardware backend found for the provided Executor.");
    }
}

} // namespace ozablas