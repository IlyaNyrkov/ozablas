#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace ozablas {

/**
 * @brief Abstract base class for hardware-specific execution contexts.
 * * The Executor manages memory allocations, data transfers, and hardware
 * synchronization. It completely hides vendor-specific API calls
 * (like cudaMalloc or hipMemcpy) from the user and the Workspace.
 */
class Executor {
public:
    virtual ~Executor() = default;
    /**
     * @brief Allocates raw bytes on the target device.
     */
    virtual void allocate(void** ptr, size_t bytes) const = 0;

    /**
     * @brief Frees raw bytes on the target device.
     */
    virtual void free(void* ptr) const = 0;

    virtual void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const = 0;
    virtual void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const = 0;
    virtual void copy_device_to_device(void* dst, const void* src, size_t bytes) const = 0;

    /**
     * @brief Blocks the host thread until all previously issued tasks on this device complete.
     */
    virtual void synchronize() const = 0;

    /**
     * @brief Returns the underlying vendor-specific BLAS handle (e.g., cublasHandle_t).
     * Returns a void* to prevent leaking vendor headers into the public C++ API.
     */
    virtual void* get_blas_handle() const { return nullptr; }
};

/**
 * @brief CPU execution context. Uses standard C++ malloc/free/memcpy.
 */
class CpuExecutor : public Executor {
public:
    CpuExecutor() = default;
    ~CpuExecutor() override = default;

    void allocate(void** ptr, size_t bytes) const override;
    void free(void* ptr) const override;

    void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const override;
    void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const override;
    void copy_device_to_device(void* dst, const void* src, size_t bytes) const override;

    void synchronize() const override;
};

/**
 * @brief NVIDIA GPU execution context. Uses CUDA and cuBLAS.
 */
class CudaExecutor : public Executor {
public:
    /**
     * @param device_id The target GPU index (default 0).
     */
    explicit CudaExecutor(int device_id = 0);
    ~CudaExecutor() override;

    void allocate(void** ptr, size_t bytes) const override;
    void free(void* ptr) const override;

    void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const override;
    void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const override;
    void copy_device_to_device(void* dst, const void* src, size_t bytes) const override;

    void synchronize() const override;
    void* get_blas_handle() const override;

private:
    int device_id_;
    void* cublas_handle_; // Opaque pointer to cublasHandle_t
};

/**
 * @brief AMD GPU execution context. Uses HIP and rocBLAS.
 */
class HipExecutor : public Executor {
public:
    /**
     * @param device_id The target GPU index (default 0).
     */
    explicit HipExecutor(int device_id = 0);
    ~HipExecutor() override;

    void allocate(void** ptr, size_t bytes) const override;
    void free(void* ptr) const override;

    void copy_from_host(void* device_dst, const void* host_src, size_t bytes) const override;
    void copy_to_host(void* host_dst, const void* device_src, size_t bytes) const override;
    void copy_device_to_device(void* dst, const void* src, size_t bytes) const override;

    void synchronize() const override;
    void* get_blas_handle() const override;

private:
    int device_id_;
    void* rocblas_handle_; // Opaque pointer to rocblas_handle
};

} // namespace ozablas