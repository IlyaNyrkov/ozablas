# CMake 3.24+ is highly recommended for stable, native HIP language support
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(OzaBLAS
        VERSION 1.0.0
        DESCRIPTION "A multi-platform HPC library implementing Ozaki Scheme I and II"
        LANGUAGES CXX
)

# ==========================================
# BUILD TYPE & C++20 SETTINGS
# ==========================================
# Force Release mode if not specified to guarantee -O3 optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================================
# AGGRESSIVE HOST COMPILER FLAGS
# ==========================================
# -O3: Maximum optimization
# -march=native: Vectorize (AVX2/AVX-512) for the specific CPU compiling the code
# -funroll-loops: Unroll tight loops to hide instruction latency
set(HOST_OPT_FLAGS "-O3 -march=native -funroll-loops")

# WARNING: -ffast-math breaks IEEE-754 strictness. Enable carefully!
# set(HOST_OPT_FLAGS "${HOST_OPT_FLAGS} -ffast-math")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${HOST_OPT_FLAGS}")

# ==========================================
# USER OPTIONS
# ==========================================
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)

option(OZABLAS_ENABLE_CUDA "Enable NVIDIA CUDA backend" OFF)
option(OZABLAS_ENABLE_HIP  "Enable AMD HIP backend" OFF)

option(OZABLAS_BUILD_TESTS      "Build the GoogleTest suite" ON)
option(OZABLAS_BUILD_BENCHMARKS "Build the Google Benchmark suite" ON)
option(OZABLAS_BUILD_EXAMPLES   "Build user tutorials" ON)

# ==========================================
# HARDWARE TOOLCHAIN SETUP
# ==========================================

if(OZABLAS_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(OZA_BUILD_CUDA)

    # Aggressive CUDA optimization flags
    # -Xptxas -O3: Maximize PTX assembly optimization and register allocation
    # --use_fast_math: Combines -ftz=true, -prec-div=false, -prec-sqrt=false (Enable with caution for Ozaki)
    set(CUDA_OPT_FLAGS "-O3 -Xcompiler=-O3,-march=native,-funroll-loops -Xptxas=-O3")
    # set(CUDA_OPT_FLAGS "${CUDA_OPT_FLAGS} --use_fast_math")

    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${CUDA_OPT_FLAGS}")

    message(STATUS "OzaBLAS: CUDA backend enabled with aggressive optimizations.")
endif()

if(OZABLAS_ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()

    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)

    add_compile_definitions(OZA_BUILD_HIP)

    # Aggressive HIP optimization flags
    # HIP uses clang under the hood, so standard LLVM flags apply
    set(HIP_OPT_FLAGS "-O3 -funroll-loops")
    # set(HIP_OPT_FLAGS "${HIP_OPT_FLAGS} -ffast-math -munsafe-fp-atomics")

    set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} ${HIP_OPT_FLAGS}")

    message(STATUS "OzaBLAS: HIP backend enabled with aggressive optimizations.")
endif()

if(NOT OZABLAS_ENABLE_CUDA AND NOT OZABLAS_ENABLE_HIP)
    message(FATAL_ERROR "OzaBLAS requires at least one hardware backend. Please set OZABLAS_ENABLE_CUDA=ON or OZABLAS_ENABLE_HIP=ON.")
endif()

# ==========================================
# DEPENDENCIES & SUBDIRECTORIES
# ==========================================

include_directories(include)

add_subdirectory(src)

if(OZABLAS_BUILD_TESTS OR OZABLAS_BUILD_BENCHMARKS OR OZABLAS_BUILD_EXAMPLES)
    add_subdirectory(utils)
endif()

if(OZABLAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(OZABLAS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(OZABLAS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()