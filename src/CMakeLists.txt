# ==========================================
# DEFINE HOST SOURCES
# ==========================================
# These files contain standard C++ and are strictly compiled by host compiler (GCC/Clang/MSVC).
set(OZABLAS_HOST_SOURCES
        ozablas.cpp
        core/executor.cpp
        core/workspace.cpp
)

# Create the main library target.
# BUILD_SHARED_LIBS from the root CMakeLists.txt automatically determines if this is .a or .so
add_library(ozablas ${OZABLAS_HOST_SOURCES})

# ==========================================
# INCLUDE DIRECTORIES
# ==========================================
target_include_directories(ozablas
        PUBLIC
        # Users linking against ozablas will automatically get the public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        # Internal files (like pipeline.cu) need to see the private common/ and pipeline/ headers
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ==========================================
# CUDA BACKEND (NVIDIA)
# ==========================================
if(OZABLAS_ENABLE_CUDA)
    # Appending the .cu file tells CMake to hand this specific file over to nvcc
    target_sources(ozablas PRIVATE cuda/pipeline.cu)

    # Step 3 of the Ozaki scheme requires batched dense GEMMs
    # We link against cuBLAS (and the CUDA runtime) which were found by FindCUDAToolkit
    target_link_libraries(ozablas PUBLIC CUDA::cublas CUDA::cudart)

    # If building shared libraries, position independent code is required
    set_property(TARGET ozablas PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# ==========================================
# HIP BACKEND (AMD)
# ==========================================
if(OZABLAS_ENABLE_HIP)
    # Appending the .hip file tells CMake to hand this specific file over to hipcc
    target_sources(ozablas PRIVATE hip/pipeline.hip)

    # Step 3 of the Ozaki scheme uses rocblas_gemm_strided_batched_ex
    # We link against rocBLAS which was found by find_package in the root CMake
    target_link_libraries(ozablas PUBLIC roc::rocblas)

    # Ensure position independent code for the HIP backend as well
    set_property(TARGET ozablas PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()