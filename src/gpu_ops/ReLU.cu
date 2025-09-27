#include "ReLU.cuh"
#include "../ErrorCheck.h"
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size, bool inplace)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    if (inplace) {
        // In-place operation: input and output are the same pointer
        float* data = const_cast<float*>(input);
        for (int i = idx; i < size; i += stride) {
            data[i] = fmaxf(0.0f, data[i]);
        }
    } else {
        // Out-of-place operation: separate input and output
        for (int i = idx; i < size; i += stride) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

void ReLU::forward_inplace(const std::shared_ptr<CudaBuffer>& data,
                           int size, cudaStream_t stream)
{
    const int blockSize = 512; 
    const int gridSize = min((size + blockSize - 1) / blockSize, 2048);
    
    relu_kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(data->data), 
        static_cast<float*>(data->data), 
        size, true);
    
    CHECK_CUDA(cudaGetLastError());
}

void ReLU::forward(const std::shared_ptr<CudaBuffer>& input,
                   const std::shared_ptr<CudaBuffer>& output,
                   int size, cudaStream_t stream)
{
    const int blockSize = 512;
    const int gridSize = min((size + blockSize - 1) / blockSize, 2048);
    
    relu_kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(input->data), 
        static_cast<float*>(output->data), 
        size, false);
    
    CHECK_CUDA(cudaGetLastError());
} 