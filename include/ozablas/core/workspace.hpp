#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include "ozablas/core/executor.hpp"

namespace ozablas {

// =================================================================================================
// WORKSPACE: OZAKI SCHEME I
// =================================================================================================

class WorkspaceScheme1 {
public:
    WorkspaceScheme1(std::shared_ptr<const Executor> exec, size_t M, size_t N, size_t K, int slices);
    ~WorkspaceScheme1();

    WorkspaceScheme1(const WorkspaceScheme1&) = delete;
    WorkspaceScheme1& operator=(const WorkspaceScheme1&) = delete;

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    size_t get_M() const { return M_; }
    size_t get_N() const { return N_; }
    size_t get_K() const { return K_; }
    int get_slices() const { return slices_; }

    int32_t* get_shift_A()   const { return d_shift_A_; }
    int32_t* get_shift_B()   const { return d_shift_B_; }
    int8_t* get_A_slices()  const { return d_A_slices_; }
    int8_t* get_B_slices()  const { return d_B_slices_; }
    int32_t* get_C_tc()      const { return d_C_tc_; }

private:
    std::shared_ptr<const Executor> exec_;

    size_t M_, N_, K_;
    int slices_;

    // Step 1: Matrix Statistics (FUSED)
    int32_t* d_shift_A_   = nullptr;
    int32_t* d_shift_B_   = nullptr;

    // Step 2: Slice Matrices
    int8_t* d_A_slices_  = nullptr;
    int8_t* d_B_slices_  = nullptr;

    // Step 3: Multiply Slices
    int32_t* d_C_tc_      = nullptr; // Holds S(S+1)/2 matrices
};

// =================================================================================================
// WORKSPACE: OZAKI SCHEME II (CRT)
// =================================================================================================

class WorkspaceScheme2 {
public:
    WorkspaceScheme2(std::shared_ptr<const Executor> exec, size_t M, size_t N, size_t K, int slices);
    ~WorkspaceScheme2();

    WorkspaceScheme2(const WorkspaceScheme2&) = delete;
    WorkspaceScheme2& operator=(const WorkspaceScheme2&) = delete;

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    size_t get_M() const { return M_; }
    size_t get_N() const { return N_; }
    size_t get_K() const { return K_; }
    int get_slices() const { return slices_; }

    int32_t* get_shift_A()   const { return d_shift_A_; }
    int32_t* get_shift_B()   const { return d_shift_B_; }
    int8_t* get_A_slices()  const { return d_A_slices_; }
    int8_t* get_B_slices()  const { return d_B_slices_; }
    int32_t* get_C_tc()      const { return d_C_tc_; }

private:
    std::shared_ptr<const Executor> exec_;

    size_t M_, N_, K_;
    int slices_;

    // Step 1: Matrix Statistics (FUSED)
    int32_t* d_shift_A_   = nullptr;
    int32_t* d_shift_B_   = nullptr;

    // Step 2: Slice Matrices
    int8_t* d_A_slices_  = nullptr;
    int8_t* d_B_slices_  = nullptr;

    // Step 3: Multiply Slices
    int32_t* d_C_tc_      = nullptr; // Holds exactly S matrices
};

// A lightweight struct to hold pipeline execution times
struct OzaTimings {
    float step1_ms = 0.0f; // Matrix Statistics
    float step2_ms = 0.0f; // Slicing
    float step3_ms = 0.0f; // Tensor Core GEMM
    float step4_ms = 0.0f; // Sum & Scale / Reconstruction
    float total_ms = 0.0f;
};

} // namespace ozablas
