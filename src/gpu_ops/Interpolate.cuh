#pragma once
#include "../CudaBuffer.cuh"
#include <memory>
#include <cuda_runtime.h>

class Interpolate {
public:
    static void forward(const std::shared_ptr<CudaBuffer>& input,
                       const std::shared_ptr<CudaBuffer>& output,
                       int batch_size, int channels,
                       int input_height, int input_width,
                       int output_height, int output_width,
                       cudaStream_t stream = 0);
}; 