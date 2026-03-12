#include "ozablas/core/workspace.hpp"
#include <stdexcept>
#include <mutex>

// We conditionally include the bridge headers.
// These contain the pure C++ declarations for our lazy-loading functions.
#ifdef OZA_BUILD_CUDA
#include "cuda/ozablas.hpp"
#endif

#ifdef OZA_BUILD_HIP
#include "hip/ozablas.hpp"
#endif

namespace ozablas {

// =================================================================================================
// WORKSPACE: OZAKI SCHEME I
// =================================================================================================

WorkspaceScheme1::WorkspaceScheme1(std::shared_ptr<const Executor> exec, size_t M, size_t N, size_t K, int slices)
    : exec_(exec), M_(M), N_(N), K_(K), slices_(slices)
{
    if (!exec_) throw std::invalid_argument("OzaBLAS: Executor cannot be null.");
    if (slices_ < 1) throw std::invalid_argument("OzaBLAS: Slices must be >= 1.");

    try {
        // Step 1: Matrix Statistics (Scale factors and shifts)
        exec_->allocate((void**)&d_shift_A_,   M_ * sizeof(int32_t));
        exec_->allocate((void**)&d_shift_B_,   N_ * sizeof(int32_t));

        // Step 2: Slice Matrices (INT8 representation)
        exec_->allocate((void**)&d_A_slices_, slices_ * M_ * K_ * sizeof(int8_t));
        exec_->allocate((void**)&d_B_slices_, slices_ * K_ * N_ * sizeof(int8_t));

        // Step 3: Multiply Slices
        // Scheme 1 accumulates S(S+1)/2 intermediate matrices
        size_t total_gemms = slices_ * (slices_ + 1) / 2;
        exec_->allocate((void**)&d_C_tc_, total_gemms * M_ * N_ * sizeof(int32_t));

    } catch (const std::exception& e) {
        // If an allocation fails midway, the destructor won't run. We must clean up manually.
        this->~WorkspaceScheme1();
        throw;
    }
}

WorkspaceScheme1::~WorkspaceScheme1() {
    if (!exec_) return;
    if (d_shift_A_)   exec_->free(d_shift_A_);
    if (d_shift_B_)   exec_->free(d_shift_B_);
    if (d_A_slices_)  exec_->free(d_A_slices_);
    if (d_B_slices_)  exec_->free(d_B_slices_);
    if (d_C_tc_)      exec_->free(d_C_tc_);
}

// =================================================================================================
// WORKSPACE: OZAKI SCHEME II (CRT)
// =================================================================================================

// Global thread-safe flag to ensure __constant__ memory is loaded strictly once per application lifetime.
static std::once_flag init_crt_tables_flag;

WorkspaceScheme2::WorkspaceScheme2(std::shared_ptr<const Executor> exec, size_t M, size_t N, size_t K, int slices)
    : exec_(exec), M_(M), N_(N), K_(K), slices_(slices)
{
    if (!exec_) throw std::invalid_argument("OzaBLAS: Executor cannot be null.");
    if (slices_ < 1 || slices_ > 20) {
        throw std::invalid_argument("OzaBLAS: Scheme II supports between 1 and 20 slices.");
    }

    try {
        // Step 1: Matrix Statistics (Scale factors and shifts)
        exec_->allocate((void**)&d_shift_A_,   M_ * sizeof(int32_t));
        exec_->allocate((void**)&d_shift_B_,   N_ * sizeof(int32_t));

        // Step 2: Slice Matrices (INT8 representation)
        exec_->allocate((void**)&d_A_slices_, slices_ * M_ * K_ * sizeof(int8_t));
        exec_->allocate((void**)&d_B_slices_, slices_ * K_ * N_ * sizeof(int8_t));

        // Step 3: Multiply Slices
        // Scheme II only requires S intermediate matrices
        exec_->allocate((void**)&d_C_tc_, slices_ * M_ * N_ * sizeof(int32_t));

    } catch (const std::exception& e) {
        this->~WorkspaceScheme2();
        throw;
    }

    // ---------------------------------------------------------------------------------------------
    // LAZY INITIALIZATION: Load CRT lookup tables into GPU __constant__ memory
    // ---------------------------------------------------------------------------------------------
    std::call_once(init_crt_tables_flag, [this]() {

        if (auto cuda_exec = dynamic_cast<const CudaExecutor*>(exec_.get())) {
#ifdef OZA_BUILD_CUDA
            cuda::initialize_constant_memory();
#else
            throw std::runtime_error("OzaBLAS was not compiled with CUDA support.");
#endif
        }
        else if (auto hip_exec = dynamic_cast<const HipExecutor*>(exec_.get())) {
#ifdef OZA_BUILD_HIP
            hip::initialize_constant_memory();
#else
            throw std::runtime_error("OzaBLAS was not compiled with HIP support.");
#endif
        }
        else {
            // CPU fallback. The CPU doesn't have __constant__ memory; it just reads
            // the constexpr arrays from crt_tables.hpp directly. So this is a no-op.
        }
    });
}

WorkspaceScheme2::~WorkspaceScheme2() {
    if (!exec_) return;
    if (d_shift_A_)   exec_->free(d_shift_A_);
    if (d_shift_B_)   exec_->free(d_shift_B_);
    if (d_A_slices_)  exec_->free(d_A_slices_);
    if (d_B_slices_)  exec_->free(d_B_slices_);
    if (d_C_tc_)      exec_->free(d_C_tc_);
}

} // namespace ozablas
