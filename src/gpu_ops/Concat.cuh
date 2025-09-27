#pragma once

#include "../CudaBuffer.cuh"
#include <memory>

/**
 * Channel concatenation operation for UNet skip connections
 * Concatenates two tensors along the channel dimension
 */
class Concat {
public:
    /**
     * Concatenate two tensors along channel dimension
     * @param input1 GPU float32 tensor of shape (N, C1, H, W)
     * @param input2 GPU float32 tensor of shape (N, C2, H, W)
     * @param output GPU float32 tensor of shape (N, C1+C2, H, W)
     * @param batch_size Number of samples in batch
     * @param channels1 Number of channels in first tensor
     * @param channels2 Number of channels in second tensor
     * @param height Height of tensors
     * @param width Width of tensors
     * @param stream CUDA stream for asynchronous operation
     */
    static void forward(const std::shared_ptr<CudaBuffer>& input1,
                        const std::shared_ptr<CudaBuffer>& input2,
                        const std::shared_ptr<CudaBuffer>& output,
                        int batch_size, int channels1, int channels2,
                        int height, int width, cudaStream_t stream);
}; 