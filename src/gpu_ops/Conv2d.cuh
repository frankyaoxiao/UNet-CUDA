#pragma once

#include "../CudaBuffer.cuh"
#include <memory>


class Conv2d {
public:
    std::shared_ptr<CudaBuffer> weights;
    std::shared_ptr<CudaBuffer> bias;
    
    int in_channels;
    int out_channels;
    int kernel_h, kernel_w;
    int padding_h, padding_w;
    int stride_h, stride_w;
    bool has_bias;

    Conv2d(int in_channels, int out_channels, int kernel_size, 
           int padding = 0, int stride = 1, bool use_bias = true);
    
    Conv2d(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int padding_h, int padding_w, int stride_h, int stride_w, bool use_bias = true);

    /**
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
                 cudaStream_t stream = 0);
    
    /**
     * Apply 2D convolution with fused ReLU activation
     */
    void forward_relu(const std::shared_ptr<CudaBuffer>& input,
                      const std::shared_ptr<CudaBuffer>& output,
                      int batch_size, int input_h, int input_w,
                      cudaStream_t stream = 0);
    

    void get_output_size(int input_h, int input_w, int& output_h, int& output_w);
}; 