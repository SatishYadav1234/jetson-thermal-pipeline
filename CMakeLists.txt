cmake_minimum_required(VERSION 3.18)
project(CudaThermalDenoisePipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure CUDA Architecture
# sm_87 = Jetson Orion Nano / Orin Series
# sm_89 = RTX 40-series (Ada Lovelace)
# sm_86 = RTX 30-series (Ampere)
# sm_75 = RTX 20-series / GTX 16-series (Turing)
if (NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "87;89;86;75")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 1. Build Shared Library for Python ctypes / PyCUDA / C++ Bridge
add_library(denoise SHARED
    src/kernels/denoise_kernels.cu
)
set_target_properties(denoise PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    PREFIX "lib"
)

# 2. Build Standalone C++ Benchmark Binary
add_executable(cuda_thermal_benchmark
    src/cpp/main.cpp
)
target_link_libraries(cuda_thermal_benchmark PRIVATE denoise)

# Output installation
install(TARGETS denoise cuda_thermal_benchmark
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

message(STATUS "Configured for CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
