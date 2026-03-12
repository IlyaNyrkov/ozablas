// src/core/executor.cpp
#include "ozablas/core/executor.hpp"
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#ifdef OZA_BUILD_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef OZA_BUILD_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace ozablas {

// =================================================================================================
// CPU EXECUTOR IMPLEMENTATION
// =================================================================================================

constexpr size_t ALIGNMENT = 64;

void CpuExecutor::allocate(void** ptr, size_t bytes) const {
    size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    *ptr = std::aligned_alloc(ALIGNMENT, aligned_bytes);
    if (*ptr == nullptr) {
        throw std::bad_alloc();
    }
}

void CpuExecutor::free(void* ptr) const {
    std::free(ptr);
}

void CpuExecutor::copy_from_host(void* device_dst, const void* host_src, size_t bytes) const {
    std::memcpy(device_dst, host_src, bytes);
}

void CpuExecutor::copy_to_host(void* host_dst, const void* device_src, size_t bytes) const {
    std::memcpy(host_dst, device_src, bytes);
}

void CpuExecutor::copy_device_to_device(void* dst, const void* src, size_t bytes) const {
    std::memcpy(dst, src, bytes);
}

void CpuExecutor::synchronize() const {
    // No-op for CPU
}

// =================================================================================================
// CUDA EXECUTOR IMPLEMENTATION
// =================================================================================================

#ifdef OZA_BUILD_CUDA

CudaExecutor::CudaExecutor(int device_id) : device_id_(device_id), cublas_handle_(nullptr) {
    if (cudaSetDevice(device_id_) != cudaSuccess) {
        throw std::runtime_error("CUDA: Failed to set device.");
    }
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("CUDA: Failed to create cuBLAS handle.");
    }
    cublas_handle_ = static_cast<void*>(handle);
}

CudaExecutor::~CudaExecutor() {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
    }
}

void CudaExecutor::allocate(void** ptr, size_t bytes) const {
    if (cudaMalloc(ptr, bytes) != cudaSuccess) {
        throw std::bad_alloc();
    }
}

void CudaExecutor::free(void* ptr) const {
    cudaFree(ptr);
}

void CudaExecutor::copy_from_host(void* device_dst, const void* host_src, size_t bytes) const {
    if (cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CUDA: Memcpy HostToDevice failed.");
    }
}

void CudaExecutor::copy_to_host(void* host_dst, const void* device_src, size_t bytes) const {
    if (cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("CUDA: Memcpy DeviceToHost failed.");
    }
}

void CudaExecutor::copy_device_to_device(void* dst, const void* src, size_t bytes) const {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        throw std::runtime_error("CUDA: Memcpy DeviceToDevice failed.");
    }
}

void CudaExecutor::synchronize() const {
    if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("CUDA: Device synchronize failed.");
    }
}

void* CudaExecutor::get_blas_handle() const {
    return cublas_handle_;
}

#else // Stubs if CUDA is not enabled

CudaExecutor::CudaExecutor(int) { throw std::runtime_error("OzaBLAS was not compiled with CUDA support."); }
CudaExecutor::~CudaExecutor() = default;
void CudaExecutor::allocate(void**, size_t) const {}
void CudaExecutor::free(void*) const {}
void CudaExecutor::copy_from_host(void*, const void*, size_t) const {}
void CudaExecutor::copy_to_host(void*, const void*, size_t) const {}
void CudaExecutor::copy_device_to_device(void*, const void*, size_t) const {}
void CudaExecutor::synchronize() const {}
void* CudaExecutor::get_blas_handle() const { return nullptr; }

#endif // OZA_BUILD_CUDA


// =================================================================================================
// HIP EXECUTOR IMPLEMENTATION
// =================================================================================================

#ifdef OZA_BUILD_HIP

HipExecutor::HipExecutor(int device_id) : device_id_(device_id), rocblas_handle_(nullptr) {
    if (hipSetDevice(device_id_) != hipSuccess) {
        throw std::runtime_error("HIP: Failed to set device.");
    }
    rocblas_handle handle;
    if (rocblas_create_handle(&handle) != rocblas_status_success) { //
        throw std::runtime_error("HIP: Failed to create rocBLAS handle.");
    }
    rocblas_handle_ = static_cast<void*>(handle);
}

HipExecutor::~HipExecutor() {
    if (rocblas_handle_) {
        rocblas_destroy_handle(static_cast<rocblas_handle>(rocblas_handle_)); //
    }
}

void HipExecutor::allocate(void** ptr, size_t bytes) const {
    if (hipMalloc(ptr, bytes) != hipSuccess) { //
        throw std::bad_alloc();
    }
}

void HipExecutor::free(void* ptr) const {
    (void)hipFree(ptr);
}

void HipExecutor::copy_from_host(void* device_dst, const void* host_src, size_t bytes) const {
    if (hipMemcpy(device_dst, host_src, bytes, hipMemcpyHostToDevice) != hipSuccess) { //
        throw std::runtime_error("HIP: Memcpy HostToDevice failed.");
    }
}

void HipExecutor::copy_to_host(void* host_dst, const void* device_src, size_t bytes) const {
    if (hipMemcpy(host_dst, device_src, bytes, hipMemcpyDeviceToHost) != hipSuccess) { //
        throw std::runtime_error("HIP: Memcpy DeviceToHost failed.");
    }
}

void HipExecutor::copy_device_to_device(void* dst, const void* src, size_t bytes) const {
    if (hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice) != hipSuccess) {
        throw std::runtime_error("HIP: Memcpy DeviceToDevice failed.");
    }
}

void HipExecutor::synchronize() const {
    if (hipDeviceSynchronize() != hipSuccess) { //
        throw std::runtime_error("HIP: Device synchronize failed.");
    }
}

void* HipExecutor::get_blas_handle() const {
    return rocblas_handle_;
}

#else // Stubs if HIP is not enabled

HipExecutor::HipExecutor(int) { throw std::runtime_error("OzaBLAS was not compiled with HIP support."); }
HipExecutor::~HipExecutor() = default;
void HipExecutor::allocate(void**, size_t) const {}
void HipExecutor::free(void*) const {}
void HipExecutor::copy_from_host(void*, const void*, size_t) const {}
void HipExecutor::copy_to_host(void*, const void*, size_t) const {}
void HipExecutor::copy_device_to_device(void*, const void*, size_t) const {}
void HipExecutor::synchronize() const {}
void* HipExecutor::get_blas_handle() const { return nullptr; }

#endif // OZA_BUILD_HIP

} // namespace ozablas