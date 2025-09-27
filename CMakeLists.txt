cmake_minimum_required(VERSION 3.18)
project(unet_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Conservative optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)

# Make OpenCV optional
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_compile_definitions(HAS_OPENCV)
else()
    message(STATUS "OpenCV not found. Building without image I/O support.")
endif()

# Include directories
include_directories(src)

# Core library sources
set(LIB_SOURCES
    src/unet/UNetModel.cu
    src/unet/WeightLoader.cu
    src/gpu_ops/Conv2d.cu
    src/gpu_ops/ReLU.cu
    src/gpu_ops/MaxPool2d.cu
    src/gpu_ops/ConvTranspose2d.cu
    src/gpu_ops/Concat.cu
    src/gpu_ops/Sigmoid.cu
    src/gpu_ops/Interpolate.cu
    src/CudaBuffer.cu
    src/HostBuffer.cpp
)

add_library(unet_gpu_lib ${LIB_SOURCES})
target_link_libraries(unet_gpu_lib CUDA::cudart)

add_executable(unet_demo src/main.cu)
target_link_libraries(unet_demo unet_gpu_lib)

if(OpenCV_FOUND)
    target_link_libraries(unet_demo ${OpenCV_LIBS})
    target_include_directories(unet_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

add_executable(conv2d_test src/tests/Conv2dTest.cu)
target_link_libraries(conv2d_test unet_gpu_lib)

add_executable(relu_test src/tests/ReLUTest.cu)
target_link_libraries(relu_test unet_gpu_lib)

add_executable(maxpool_test src/tests/MaxPool2dTest.cu)
target_link_libraries(maxpool_test unet_gpu_lib)

add_executable(unet_benchmark src/benchmark.cu)
target_link_libraries(unet_benchmark unet_gpu_lib)

enable_testing()
add_test(NAME conv2d_test COMMAND conv2d_test)
add_test(NAME relu_test COMMAND relu_test)
add_test(NAME maxpool_test COMMAND maxpool_test) 