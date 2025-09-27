#include "Conv2d.cuh"
#include "../ErrorCheck.h"
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weights,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int batch_size, int in_channels, int out_channels,
                              int input_h, int input_w, int output_h, int output_w,
                              int kernel_h, int kernel_w,
                              int padding_h, int padding_w,
                              int stride_h, int stride_w,
                              bool has_bias, bool fused_relu)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_channels * output_h * output_w) return;
    
    const int w_out = idx % output_w;
    const int temp1 = idx / output_w;
    const int h_out = temp1 % output_h;
    const int temp2 = temp1 / output_h;
    const int c_out = temp2 % out_channels;
    const int n = temp2 / out_channels;
    
    float sum = 0.0f;
    
    // We have a specialized 3x3 kernel because its the most common
    if (kernel_h == 3 && kernel_w == 3) {
        const int input_base = n * in_channels * input_h * input_w;
        const int weight_base = c_out * in_channels * 9; // 3x3 = 9
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            const int input_channel_base = input_base + c_in * input_h * input_w;
            const int weight_channel_base = weight_base + c_in * 9;
            
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                const int h_in = h_out * stride_h - padding_h + kh;
                if (h_in >= 0 && h_in < input_h) {
                    const int input_row_base = input_channel_base + h_in * input_w;
                    const int weight_row_base = weight_channel_base + kh * 3;
                    
                    #pragma unroll
                    for (int kw = 0; kw < 3; kw++) {
                        const int w_in = w_out * stride_w - padding_w + kw;
                        if (w_in >= 0 && w_in < input_w) {
                            sum = __fmaf_rn(input[input_row_base + w_in], 
                                           weights[weight_row_base + kw], sum);
                        }
                    }
                }
            }
        }
    } else {
        // For all other kernel sizes other than 3x3
        const int in_offset_base = n * in_channels * input_h * input_w;
        const int weight_offset_base = c_out * in_channels * kernel_h * kernel_w;
        
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            const int in_channel_offset = in_offset_base + c_in * input_h * input_w;
            const int weight_channel_offset = weight_offset_base + c_in * kernel_h * kernel_w;
            
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int h_in = h_out * stride_h - padding_h + kh;
                if (h_in < 0 || h_in >= input_h) continue;
                
                const int in_row_offset = in_channel_offset + h_in * input_w;
                const int weight_row_offset = weight_channel_offset + kh * kernel_w;
                
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int w_in = w_out * stride_w - padding_w + kw;
                    if (w_in >= 0 && w_in < input_w) {
                        sum += input[in_row_offset + w_in] * weights[weight_row_offset + kw];
                    }
                }
            }
        }
    }
    
    if (has_bias) sum += bias[c_out];
    if (fused_relu) sum = fmaxf(sum, 0.0f);
    
    output[((n * out_channels + c_out) * output_h + h_out) * output_w + w_out] = sum;
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, 
               int padding, int stride, bool use_bias)
    : Conv2d(in_channels, out_channels, kernel_size, kernel_size,
             padding, padding, stride, stride, use_bias) {}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_h, int kernel_w,
               int padding_h, int padding_w, int stride_h, int stride_w, bool use_bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_h(kernel_h), kernel_w(kernel_w),
      padding_h(padding_h), padding_w(padding_w),
      stride_h(stride_h), stride_w(stride_w), has_bias(use_bias)
{
    size_t weight_size = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    weights = std::make_shared<CudaBuffer>(weight_size);
    
    if (has_bias) {
        size_t bias_size = out_channels * sizeof(float);
        bias = std::make_shared<CudaBuffer>(bias_size);
    }
}

void Conv2d::forward(const std::shared_ptr<CudaBuffer>& input,
                     const std::shared_ptr<CudaBuffer>& output,
                     int batch_size, int input_h, int input_w,
                     cudaStream_t stream)
{
    int output_h, output_w;
    get_output_size(input_h, input_w, output_h, output_w);
    
    const float* d_input = reinterpret_cast<const float*>(input->data);
    const float* d_weights = reinterpret_cast<const float*>(weights->data);
    const float* d_bias = has_bias ? reinterpret_cast<const float*>(bias->data) : nullptr;
    float* d_output = reinterpret_cast<float*>(output->data);
    
    int total_output = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 512;
    int blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    conv2d_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w, padding_h, padding_w,
        stride_h, stride_w, has_bias, false
    );
    
    checkCuda(cudaPeekAtLastError());
}

void Conv2d::forward_relu(const std::shared_ptr<CudaBuffer>& input,
                          const std::shared_ptr<CudaBuffer>& output,
                          int batch_size, int input_h, int input_w,
                          cudaStream_t stream)
{
    int output_h, output_w;
    get_output_size(input_h, input_w, output_h, output_w);
    
    const float* d_input = reinterpret_cast<const float*>(input->data);
    const float* d_weights = reinterpret_cast<const float*>(weights->data);
    const float* d_bias = has_bias ? reinterpret_cast<const float*>(bias->data) : nullptr;
    float* d_output = reinterpret_cast<float*>(output->data);
    
    int total_output = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 512;
    int blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    conv2d_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w, padding_h, padding_w,
        stride_h, stride_w, has_bias, true
    );
    
    checkCuda(cudaPeekAtLastError());
}

void Conv2d::get_output_size(int input_h, int input_w, int& output_h, int& output_w)
{
    output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
    output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
} 