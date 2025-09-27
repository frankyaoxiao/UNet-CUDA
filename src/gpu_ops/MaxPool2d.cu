#include "MaxPool2d.cuh"
#include "../ErrorCheck.h"
#include <cuda_runtime.h>
#include <cfloat>

__global__ void maxpool2d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int batch_size, int channels,
                                 int input_h, int input_w, int output_h, int output_w,
                                 int kernel_size, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_output = batch_size * channels * output_h * output_w;
    if (idx >= total_output) return;
    
    int w_out = idx % output_w;
    idx /= output_w;
    int h_out = idx % output_h;
    idx /= output_h;
    int c = idx % channels;
    int n = idx / channels;
    
    int input_channel_offset = (n * channels + c) * input_h * input_w;
    
    float max_val = -FLT_MAX;
    
    // Optimized for 2x2 cuz most common
    if (kernel_size == 2 && stride == 2 && padding == 0) {
        int h_in = h_out * 2;
        int w_in = w_out * 2;
        
        if (h_in < input_h && w_in < input_w) {
            float val1 = __ldg(&input[input_channel_offset + h_in * input_w + w_in]);
            max_val = fmaxf(max_val, val1);
        }
        if (h_in < input_h && w_in + 1 < input_w) {
            float val2 = __ldg(&input[input_channel_offset + h_in * input_w + w_in + 1]);
            max_val = fmaxf(max_val, val2);
        }
        if (h_in + 1 < input_h && w_in < input_w) {
            float val3 = __ldg(&input[input_channel_offset + (h_in + 1) * input_w + w_in]);
            max_val = fmaxf(max_val, val3);
        }
        if (h_in + 1 < input_h && w_in + 1 < input_w) {
            float val4 = __ldg(&input[input_channel_offset + (h_in + 1) * input_w + w_in + 1]);
            max_val = fmaxf(max_val, val4);
        }
    } else {
        // General case
        for (int kh = 0; kh < kernel_size; kh++) {
            int h_in = h_out * stride - padding + kh;
            if (h_in < 0 || h_in >= input_h) continue;
            
            int input_row_offset = input_channel_offset + h_in * input_w;
            
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                int w_in = w_out * stride - padding + kw;
                if (w_in >= 0 && w_in < input_w) {
                    max_val = fmaxf(max_val, __ldg(&input[input_row_offset + w_in]));
                }
            }
        }
    }
    
    int output_idx = ((n * channels + c) * output_h + h_out) * output_w + w_out;
    output[output_idx] = max_val;
}

MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride), padding(padding) {}

void MaxPool2d::forward(const std::shared_ptr<CudaBuffer>& input,
                        const std::shared_ptr<CudaBuffer>& output,
                        int batch_size, int channels, int input_h, int input_w,
                        cudaStream_t stream)
{
    int output_h, output_w;
    get_output_size(input_h, input_w, output_h, output_w);
    
    int total_output = batch_size * channels * output_h * output_w;
    int threads_per_block = 512;
    int blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    maxpool2d_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const float*>(input->data),
        static_cast<float*>(output->data),
        batch_size, channels,
        input_h, input_w, output_h, output_w,
        kernel_size, stride, padding
    );
    
    checkCuda(cudaPeekAtLastError());
}

void MaxPool2d::get_output_size(int input_h, int input_w, int& output_h, int& output_w)
{
    output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
} 