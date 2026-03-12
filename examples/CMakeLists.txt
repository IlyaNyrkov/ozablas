# examples/CMakeLists.txt

# Create the executable for the example
add_executable(01_ozaki_scheme2_example 01_ozaki_scheme2_example.cpp)

# Link against the main library and the utility library
# We also include the utils directory so the #include paths resolve cleanly
target_include_directories(01_ozaki_scheme2_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_link_libraries(01_ozaki_scheme2_example PRIVATE ozablas ozablas_utils)

# If built with CUDA/HIP, CMake will automatically link the necessary BLAS
# libraries because they are PUBLIC dependencies of 'ozablas'.