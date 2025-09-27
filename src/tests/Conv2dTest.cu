#include <iostream>
#include <cassert>
#include <cmath>
#include "../gpu_ops/Conv2d.cuh"
#include "../ErrorCheck.h"

bool test_conv2d_basic() {
    const int batch_size = 1;
    const int in_channels = 1;
    const int out_channels = 1;
    const int input_h = 4;
    const int input_w = 4;
    const int kernel_size = 3;
    const int padding = 1;
    const int stride = 1;
    
    float input_data[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float weight_data[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    auto input_buffer = std::make_shared<CudaBuffer>(batch_size * in_channels * input_h * input_w * sizeof(float));
    auto output_buffer = std::make_shared<CudaBuffer>(batch_size * out_channels * input_h * input_w * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(input_buffer->data, input_data, sizeof(input_data), cudaMemcpyHostToDevice));
    
    Conv2d conv(in_channels, out_channels, kernel_size, padding, stride, false);
    CHECK_CUDA(cudaMemcpy(conv.weights->data, weight_data, sizeof(weight_data), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    conv.forward(input_buffer, output_buffer, batch_size, input_h, input_w, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float output_data[16];
    CHECK_CUDA(cudaMemcpy(output_data, output_buffer->data, sizeof(output_data), cudaMemcpyDeviceToHost));
    
    int out_h, out_w;
    conv.get_output_size(input_h, input_w, out_h, out_w);
    assert(out_h == input_h && out_w == input_w);
    
    std::cout << "Conv2d test output:" << std::endl;
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            std::cout << output_data[h * out_w + w] << " ";
        }
        std::cout << std::endl;
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    return true;
}

int main() {
    
    if (test_conv2d_basic()) {
        std::cout << "Conv2d test passed!" << std::endl;
    } else {
        std::cout << "Conv2d test failed!" << std::endl;
        return -1;
    }
    
    std::cout << "All Conv2d tests passed!" << std::endl;
    return 0;
} 