#include "Concat.cuh"
#include "../ErrorCheck.h"
#include <cuda_runtime.h>

__global__
void concat_kernel_optimized(const float* __restrict__ input1,
                             const float* __restrict__ input2,
                             float* __restrict__ output,
                             int batch_size, int channels1, int channels2,
                             int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int spatial_size = height * width;
    int output_channels = channels1 + channels2;
    int total_elements = batch_size * output_channels * spatial_size;
    
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    idx /= spatial_size;
    int c_out = idx % output_channels;
    int n = idx / output_channels;
    
    int batch_spatial_offset = n * spatial_size;
    int output_offset = (n * output_channels + c_out) * spatial_size + spatial_idx;
    
    if (c_out < channels1) {
        int input1_offset = (batch_spatial_offset + c_out * spatial_size) + spatial_idx;
        output[output_offset] = __ldg(&input1[input1_offset]);
    } else {
        int c_in2 = c_out - channels1;
        int input2_offset = (batch_spatial_offset + c_in2 * spatial_size) + spatial_idx;
        output[output_offset] = __ldg(&input2[input2_offset]);
    }
}

__global__
void concat_kernel(const float* __restrict__ input1,
                   const float* __restrict__ input2,
                   float* __restrict__ output,
                   int batch_size, int channels1, int channels2,
                   int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int spatial_size = height * width;
    int output_channels = channels1 + channels2;
    int total_output = batch_size * output_channels * spatial_size;
    
    if (idx >= total_output) return;
    
    int spatial_idx = idx % spatial_size;
    idx /= spatial_size;
    int c_out = idx % output_channels;
    int n = idx / output_channels;
    
    int output_idx = ((n * output_channels + c_out) * height * width) + spatial_idx;
    
    if (c_out < channels1) {
        int input1_idx = ((n * channels1 + c_out) * height * width) + spatial_idx;
        output[output_idx] = input1[input1_idx];
    } else {
        int c_in2 = c_out - channels1;
        int input2_idx = ((n * channels2 + c_in2) * height * width) + spatial_idx;
        output[output_idx] = input2[input2_idx];
    }
}

void Concat::forward(const std::shared_ptr<CudaBuffer>& input1,
                     const std::shared_ptr<CudaBuffer>& input2,
                     const std::shared_ptr<CudaBuffer>& output,
                     int batch_size, int channels1, int channels2,
                     int height, int width, cudaStream_t stream)
{
    const float* d_input1 = reinterpret_cast<const float*>(input1->data);
    const float* d_input2 = reinterpret_cast<const float*>(input2->data);
    float* d_output = reinterpret_cast<float*>(output->data);
    
    int spatial_size = height * width;
    int output_channels = channels1 + channels2;
    int total_output = batch_size * output_channels * spatial_size;
    
    int threads_per_block = 512; // this was found optimal through testing
    int blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    concat_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_input1, d_input2, d_output,
        batch_size, channels1, channels2,
        height, width
    );
    
    checkCuda(cudaPeekAtLastError());
} 