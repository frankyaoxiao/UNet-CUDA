#include "Interpolate.cuh"
#include "../ErrorCheck.h"

__global__ void interpolate_kernel(const float* input, float* output,
                                  int batch_size, int channels,
                                  int input_height, int input_width,
                                  int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * channels * output_height * output_width;
    if (idx >= total_elements) return;
    
    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % channels;
    int b = idx / (output_width * output_height * channels);
    
    float scale_h = __fdividef((float)input_height, (float)output_height);
    float scale_w = __fdividef((float)input_width, (float)output_width);
    
    float ih_f = __fmaf_rn(oh + 0.5f, scale_h, -0.5f);
    float iw_f = __fmaf_rn(ow + 0.5f, scale_w, -0.5f);
    
    ih_f = fmaxf(0.0f, fminf(ih_f, input_height - 1.0f));
    iw_f = fmaxf(0.0f, fminf(iw_f, input_width - 1.0f));
    
    int ih0 = __float2int_rd(ih_f);  // Fast float to int conversion
    int iw0 = __float2int_rd(iw_f);
    int ih1 = fminf(ih0 + 1, input_height - 1);
    int iw1 = fminf(iw0 + 1, input_width - 1);
    
    float h_weight = ih_f - ih0;
    float w_weight = iw_f - iw0;
    
    int input_spatial_size = input_height * input_width;
    int input_channel_stride = input_spatial_size;
    int input_batch_stride = channels * input_channel_stride;
    
    int input_base = b * input_batch_stride + c * input_channel_stride;
    
    float v00 = input[input_base + ih0 * input_width + iw0];
    float v01 = input[input_base + ih0 * input_width + iw1]; 
    float v10 = input[input_base + ih1 * input_width + iw0];
    float v11 = input[input_base + ih1 * input_width + iw1];
    
    float v0 = __fmaf_rn(v01 - v00, w_weight, v00);  // v00 * (1-w) + v01 * w
    float v1 = __fmaf_rn(v11 - v10, w_weight, v10);  // v10 * (1-w) + v11 * w
    float result = __fmaf_rn(v1 - v0, h_weight, v0); // v0 * (1-h) + v1 * h
    
    output[idx] = result;
}

void Interpolate::forward(const std::shared_ptr<CudaBuffer>& input,
                         const std::shared_ptr<CudaBuffer>& output,
                         int batch_size, int channels,
                         int input_height, int input_width,
                         int output_height, int output_width,
                         cudaStream_t stream) {
    
    int total_elements = batch_size * channels * output_height * output_width;
    int threads_per_block = 512;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    interpolate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        (const float*)input->data, (float*)output->data,
        batch_size, channels,
        input_height, input_width,
        output_height, output_width
    );
    
    CHECK_CUDA(cudaGetLastError());
} 