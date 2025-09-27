#pragma once

#include "../CudaBuffer.cuh"
#include <memory>

class MaxPool2d {
public:
    int kernel_size;
    int stride;
    int padding;

    MaxPool2d(int kernel_size, int stride = -1, int padding = 0);

    /**
     * Apply 2D max pooling
     * @param input GPU float32 input of shape (N, C, H_in, W_in)
     * @param output GPU float32 output of shape (N, C, H_out, W_out)
     * @param batch_size Number of samples in batch
     * @param channels Number of channels
     * @param input_h Input height
     * @param input_w Input width
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& input,
                 const std::shared_ptr<CudaBuffer>& output,
                 int batch_size, int channels, int input_h, int input_w,
                 cudaStream_t stream);
    
    void get_output_size(int input_h, int input_w, int& output_h, int& output_w);
}; 