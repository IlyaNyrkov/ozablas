# ozablas
Repository implementing multiplatform ozaki scheme I and II library.

## Project structure

![ozaki scheme anatomy](docs/media/ozaki_scheme_anatomy.png)

```shell
ozablas/
├── CMakeLists.txt                 # Root config: Detects hardware, sets up subdirectories.
├── include/ozablas/               # PUBLIC API (User includes these).
│   ├── core/
│   │   ├── executor.hpp           # Hardware memory managers (CPU, CUDA, HIP).
│   │   └── workspace.hpp          # WorkspaceScheme1 and WorkspaceScheme2 to pre-allocate memory.
│   └── ozablas.hpp                # Main API: ozaki_scheme1_gemm and ozaki_scheme2_gemm.
│
├── src/                           # PRIVATE IMPLEMENTATION (User never sees this).
│   ├── CMakeLists.txt             # Builds the libozablas.so shared library.
│   ├── ozablas.cpp                # The Dispatcher: Routes math ops to the correct GPU.
│   ├── core/
│   │   ├── executor.cpp           # Implements malloc/free routing (The only place with #ifdefs).
│   │   └── workspace.cpp          # Size calculations and allocations for the workspaces.
│   ├── common/                    
│   │   ├── hardware_traits.hpp    # Compile-time hardware tuning (Warp size, load size).
│   │   ├── crt_tables.hpp         # Pure C++ constexpr arrays for Scheme II moduli and lookup tables.
│   │   └── crt_math.hpp           # Custom 256-bit struct (`uint256_t`) and math (`add_256`, `sub_256`) for large slices.
│   ├── pipeline/                  # The 4 anatomical steps of the algorithm.
│   │   ├── step1_statistics.hpp   # Extract max exponents / scale factors.
│   │   ├── step2_slicing.hpp      # Bit extraction & INT8 downcasting.
│   │   └── step4_reconstruction.hpp  # Reconstruct FP64 via Scaling or CRT.
│   ├── cuda/
│   │   ├── ozablas.hpp            # Bridge header for CUDA backend.
│   │   └── pipeline.cu            # NVIDIA __global__ launches, __constant__ memory, & cuBLAS.
│   └── hip/
│       ├── ozablas.hpp            # Bridge header for HIP backend.
│       └── pipeline.hip           # AMD __global__ launches, __constant__ memory, & rocBLAS.
│
├── utils/                         # NON-LIBRARY CODE: Utilities strictly for testing and benchmarking.
│   ├── matrix_gen.hpp             # Declarations for random matrix generation.
│   ├── matrix_gen.cpp             # Implementation of matrix generation using the paper's exponential distribution.
│   ├── matrix_compare.hpp         # Mathematical error comparison logic (max absolute error, relative error).
│   └── matrix_io.hpp              # Console printing helpers for debugging slices and matrices.
│
├── tests/                         # GOOGLE TEST SUITE
│   ├── CMakeLists.txt             
│   ├── test_step1_stats.cpp       # Unit tests verifying exponent extraction accuracy.
│   ├── test_step2_slicing.cpp     # Unit tests verifying INT8 bit extraction and symmetric modulo logic.
│   ├── test_step4_crt.cpp         # Unit tests comparing 64-bit vs 256-bit CRT reconstruction.
│   └── test_integration.cpp       # End-to-end Scheme 1 and 2 tests against standard double-precision BLAS.
│
├── benchmarks/                    # GOOGLE BENCHMARK SUITE
│   ├── CMakeLists.txt             
│   ├── bench_memory.cpp           # Measures the overhead of allocating workspaces.
│   ├── bench_crt_large_slices.cpp # Benchmark focusing on the performance of slices > 7 and 256-bit math.
│   └── bench_end_to_end.cpp       # Comprehensive throughput comparison (GFLOPS) of OzaBLAS vs standard BLAS.
│
└── examples/                      # USER TUTORIALS
    ├── CMakeLists.txt             
    ├── 01_simple_scheme.cpp       # A basic, out-of-the-box example calling `ozaki_scheme1_gemm`.
    └── 02_persistent_workspace.cpp# Advanced usage showing how to instantiate and reuse an `OzakiWorkspace` in a loop.
```

## Getting Started

### Prerequisites
* **CMake 3.24+** (Required for native CUDA/HIP language support)
* **NVIDIA Toolkit** (for CUDA builds) or **AMD ROCm** (for HIP builds)
* A C++20 compatible host compiler (GCC/Clang/MSVC)

### Building the Project
We provide a convenience `Makefile` that wraps the CMake commands. By default, this compiles the library as a shared object (`libozablas.so`), along with the test suites and examples.

**To build for NVIDIA GPUs:**
```bash
make release-cuda
```

**To build for AMD GPUs:**
```bash
make release-hip
```

### Running the Examples
The build process automatically compiles the user tutorials in the examples/ directory. These applications demonstrate how to instantiate an Executor, allocate an OzakiWorkspace, and execute standard double-precision GEMM operations using Scheme I and Scheme II.

**If built for CUDA:**

```bash
./build_cuda_release/examples/01_simple_scheme
```

**If built for HIP:**
```bash
./build_hip_release/examples/01_simple_scheme
Running the Test Suite
```
To verify the accuracy of the bit-extraction and CRT reconstruction mathematics on your specific hardware architecture:

### Running the Test Suite
To verify the accuracy of the bit-extraction and CRT reconstruction mathematics on your specific hardware architecture:

```bash
# For NVIDIA:
make test-cuda

# For AMD:
make test-hip
```