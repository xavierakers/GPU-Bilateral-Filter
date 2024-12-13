# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O3 -use_fast_math -arch=sm_61 -Xcompiler -fPIC

# Source files
CUDA_SRC_FILES := cudaBilatColorVid.cu

# Output shared objects
CUDA_SO_FILES := $(CUDA_SRC_FILES:.cu=.so)

# Default target
all: $(CUDA_SO_FILES) # $(C_SO_FILES)

# Rule to compile CUDA .cu files into shared libraries (.so)
%.so: %.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $<

# Clean rule
clean:
	rm -f $(CUDA_SO_FILES)
	