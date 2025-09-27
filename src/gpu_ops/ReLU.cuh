#pragma once

#include "../CudaBuffer.cuh"
#include <memory>

class ReLU {
public:
    /**
     * Apply ReLU activation in-place
     * @param data GPU float32 tensor to apply ReLU to
     * @param size Number of elements in tensor
     * @param stream CUDA stream for asynchronous operation
     */
    static void forward_inplace(const std::shared_ptr<CudaBuffer>& data,
                                int size, cudaStream_t stream);
    
    /**
     * Apply ReLU activation
     * @param input GPU float32 input tensor
     * @param output GPU float32 output tensor
     * @param size Number of elements in tensor
     * @param stream CUDA stream for asynchronous operation
     */
    static void forward(const std::shared_ptr<CudaBuffer>& input,
                        const std::shared_ptr<CudaBuffer>& output,
                        int size, cudaStream_t stream);
}; 