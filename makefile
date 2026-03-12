.PHONY: clean release-cuda release-hip debug-cuda debug-hip test-cuda test-hip

# Build directories
BUILD_CUDA_REL = build_cuda_release
BUILD_HIP_REL  = build_hip_release
BUILD_CUDA_DBG = build_cuda_debug
BUILD_HIP_DBG  = build_hip_debug

# ==========================================
# RELEASE BUILDS (Maximum Performance)
# ==========================================

release-cuda:
	@echo "Building OzaBLAS for NVIDIA GPUs (Release)..."
	cmake -S . -B $(BUILD_CUDA_REL) -DCMAKE_BUILD_TYPE=Release -DOZABLAS_ENABLE_CUDA=ON -DOZABLAS_BUILD_EXAMPLES=ON
	cmake --build $(BUILD_CUDA_REL) -j

release-hip:
	@echo "Building OzaBLAS for AMD GPUs (Release)..."
	cmake -S . -B $(BUILD_HIP_REL) -DCMAKE_BUILD_TYPE=Release -DOZABLAS_ENABLE_HIP=ON -DOZABLAS_BUILD_EXAMPLES=ON
	cmake --build $(BUILD_HIP_REL) -j

# ==========================================
# DEBUG BUILDS (For Development)
# ==========================================

debug-cuda:
	@echo "Building OzaBLAS for NVIDIA GPUs (Debug)..."
	cmake -S . -B $(BUILD_CUDA_DBG) -DCMAKE_BUILD_TYPE=Debug -DOZABLAS_ENABLE_CUDA=ON -DOZABLAS_BUILD_EXAMPLES=ON
	cmake --build $(BUILD_CUDA_DBG) -j

debug-hip:
	@echo "Building OzaBLAS for AMD GPUs (Debug)..."
	cmake -S . -B $(BUILD_HIP_DBG) -DCMAKE_BUILD_TYPE=Debug -DOZABLAS_ENABLE_HIP=ON -DOZABLAS_BUILD_EXAMPLES=ON
	cmake --build $(BUILD_HIP_DBG) -j

# ==========================================
# TESTING
# ==========================================

test-cuda: release-cuda
	@echo "Running CUDA tests..."
	cd $(BUILD_CUDA_REL) && ctest --output-on-failure

test-hip: release-hip
	@echo "Running HIP tests..."
	cd $(BUILD_HIP_REL) && ctest --output-on-failure

# ==========================================
# UTILITIES
# ==========================================

clean:
	@echo "Cleaning all build directories..."
	rm -rf $(BUILD_CUDA_REL) $(BUILD_HIP_REL) $(BUILD_CUDA_DBG) $(BUILD_HIP_DBG)
