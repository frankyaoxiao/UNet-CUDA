#pragma once

#include "../CudaBuffer.cuh"
#include <memory>

class ConvTranspose2d {
public:
    std::shared_ptr<CudaBuffer> weights;
    std::shared_ptr<CudaBuffer> bias;
    
    int in_channels;
    int out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int padding_h, padding_w;
    bool has_bias;

    ConvTranspose2d(int in_channels, int out_channels, int kernel_size, 
                    int stride = 1, int padding = 0, bool use_bias = true);

    /**
     * Apply 2D transpose convolution
     * @param input GPU float32 input of shape (N, C_in, H_in, W_in)
     * @param output GPU float32 output of shape (N, C_out, H_out, W_out)
     * @param batch_size Number of samples in batch
     * @param input_h Input height
     * @param input_w Input width
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& input,
                 const std::shared_ptr<CudaBuffer>& output,
                 int batch_size, int input_h, int input_w,
                 cudaStream_t stream);
    
    void get_output_size(int input_h, int input_w, int& output_h, int& output_w);
}; 