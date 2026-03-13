# ----------------------------------------------------
# 01 Scheme I & II Raw Performance benchmarks
# ----------------------------------------------------
add_executable(01_performance_benchmark 01_ozaki_scheme1_2_raw_performance.cpp)
target_include_directories(01_performance_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(01_performance_benchmark PRIVATE ozablas ozablas_utils)

# ----------------------------------------------------
# 02 Scheme II Step-wise Performance benchmarks
# ----------------------------------------------------
add_executable(02_ozaki_scheme2_pipeline_steps 02_ozaki_scheme2_pipeline_steps.cpp)
target_include_directories(02_ozaki_scheme2_pipeline_steps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(02_ozaki_scheme2_pipeline_steps PRIVATE ozablas ozablas_utils)

# ----------------------------------------------------
# 03 Scheme II Step-wise Performance benchmarks
# ----------------------------------------------------
add_executable(03_ozaki_scheme1_pipeline_steps 03_ozaki_scheme1_pipeline_steps.cpp)
target_include_directories(03_ozaki_scheme1_pipeline_steps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(03_ozaki_scheme1_pipeline_steps PRIVATE ozablas ozablas_utils)