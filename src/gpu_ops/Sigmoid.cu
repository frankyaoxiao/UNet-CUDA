#include "Sigmoid.cuh"
#include "../ErrorCheck.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void sigmoid_kernel(const float* input, float* output, int size, bool inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    if (inplace) {
        float* data = const_cast<float*>(input);
        for (int i = idx; i < size; i += stride) {
            float x = data[i];
            x = fmaxf(-88.0f, fminf(88.0f, x));
            data[i] = __fdividef(1.0f, 1.0f + __expf(-x));
        }
    } else {
        for (int i = idx; i < size; i += stride) {
            float x = input[i];
            x = fmaxf(-88.0f, fminf(88.0f, x));
            output[i] = __fdividef(1.0f, 1.0f + __expf(-x));
        }
    }
}

void Sigmoid::forward_inplace(const std::shared_ptr<CudaBuffer>& data,
                              int size, cudaStream_t stream) {
    const int blockSize = 512;
    const int gridSize = min((size + blockSize - 1) / blockSize, 2048);
    
    sigmoid_kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(data->data), 
        static_cast<float*>(data->data), 
        size, true);
    
    CHECK_CUDA(cudaGetLastError());
}

void Sigmoid::forward(const std::shared_ptr<CudaBuffer>& input,
                      const std::shared_ptr<CudaBuffer>& output,
                      int size, cudaStream_t stream) {
    const int blockSize = 512;
    const int gridSize = min((size + blockSize - 1) / blockSize, 2048);
    
    sigmoid_kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(input->data),
        static_cast<float*>(output->data), 
        size, false);
    
    CHECK_CUDA(cudaGetLastError());
} 