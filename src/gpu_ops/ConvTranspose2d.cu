#include "ConvTranspose2d.cuh"
#include "../ErrorCheck.h"
#include <cuda_runtime.h>

__global__
void conv_transpose2d_kernel(const float* __restrict__ input,
                             const float* __restrict__ weights,
                             const float* __restrict__ bias,
                             float* __restrict__ output,
                             int batch_size, int in_channels, int out_channels,
                             int input_h, int input_w, int output_h, int output_w,
                             int kernel_h, int kernel_w,
                             int stride_h, int stride_w,
                             int padding_h, int padding_w,
                             bool has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_output = batch_size * out_channels * output_h * output_w;
    if (idx >= total_output) return;
    
    int w_out = idx % output_w;
    idx /= output_w;
    int h_out = idx % output_h;
    idx /= output_h;
    int c_out = idx % out_channels;
    int n = idx / out_channels;
    
    int batch_input_offset = n * in_channels * input_h * input_w;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        int input_channel_offset = batch_input_offset + c_in * input_h * input_w;
        int weight_channel_offset = (c_in * out_channels + c_out) * kernel_h * kernel_w;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            int h_in = (h_out + padding_h - kh);
            
            if (h_in >= 0 && h_in % stride_h == 0) {
                h_in /= stride_h;
                if (h_in < input_h) {
                    int input_row_offset = input_channel_offset + h_in * input_w;
                    int weight_row_offset = weight_channel_offset + kh * kernel_w;
                    
                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int w_in = (w_out + padding_w - kw);
                        if (w_in >= 0 && w_in % stride_w == 0) {
                            w_in /= stride_w;
                            if (w_in < input_w) {
                                sum += input[input_row_offset + w_in] * weights[weight_row_offset + kw];
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (has_bias) {
        sum += bias[c_out];
    }
    
    int output_idx = ((n * out_channels + c_out) * output_h + h_out) * output_w + w_out;
    output[output_idx] = sum;
}

ConvTranspose2d::ConvTranspose2d(int in_channels, int out_channels, int kernel_size, 
                                 int stride, int padding, bool use_bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_h(kernel_size), kernel_w(kernel_size),
      stride_h(stride), stride_w(stride),
      padding_h(padding), padding_w(padding), has_bias(use_bias)
{
    size_t weight_size = in_channels * out_channels * kernel_h * kernel_w * sizeof(float);
    weights = std::make_shared<CudaBuffer>(weight_size);
    
    if (has_bias) {
        size_t bias_size = out_channels * sizeof(float);
        bias = std::make_shared<CudaBuffer>(bias_size);
    }
}

void ConvTranspose2d::forward(const std::shared_ptr<CudaBuffer>& input,
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
    
    CHECK_CUDA(cudaMemsetAsync(d_output, 0, batch_size * out_channels * output_h * output_w * sizeof(float), stream));
    
    int total_output = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 512;
    int blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    conv_transpose2d_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, has_bias
    );
    
    checkCuda(cudaPeekAtLastError());
}

void ConvTranspose2d::get_output_size(int input_h, int input_w, int& output_h, int& output_w)
{
    output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h;
    output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w;
} 