# precision_benchmarks/CMakeLists.txt

# OpenMP to speed up the FP128 CPU reference matrix calculation
find_package(OpenMP REQUIRED)

# ----------------------------------------------------
# 01 Example
# ----------------------------------------------------
add_executable(01_ozaki_scheme2_example 01_ozaki_scheme2_example.cpp)
target_include_directories(01_ozaki_scheme2_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(01_ozaki_scheme2_example PRIVATE ozablas ozablas_utils)

# ----------------------------------------------------
# 02 Scheme 2 FP128 Benchmark
# ----------------------------------------------------
add_executable(02_ozaki_scheme2_fp128_benchmark 02_ozaki_scheme2_fp128_benchmark.cpp)
target_include_directories(02_ozaki_scheme2_fp128_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)

# Link OpenMP alongside the library to enable parallel CPU execution
target_link_libraries(02_ozaki_scheme2_fp128_benchmark PRIVATE
        ozablas
        ozablas_utils
        OpenMP::OpenMP_CXX
)

# ----------------------------------------------------
# 03 Scheme I Basic Example
# ----------------------------------------------------
add_executable(03_ozaki_scheme1_example 03_ozaki_scheme1_example.cpp)
target_include_directories(03_ozaki_scheme1_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(03_ozaki_scheme1_example PRIVATE ozablas ozablas_utils)

# ----------------------------------------------------
# 04 Scheme I FP128 Benchmark
# ----------------------------------------------------
add_executable(04_ozaki_scheme1_fp128_benchmark 04_ozaki_scheme1_fp128_benchmark.cpp)
target_include_directories(04_ozaki_scheme1_fp128_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)

# Link OpenMP alongside the library to enable parallel CPU execution
target_link_libraries(04_ozaki_scheme1_fp128_benchmark PRIVATE
        ozablas
        ozablas_utils
        OpenMP::OpenMP_CXX
)
