#include <iostream>
#include <cassert>
#include <cmath>
#include "../gpu_ops/MaxPool2d.cuh"
#include "../ErrorCheck.h"

bool test_maxpool2d_basic() {
    const int batch_size = 1;
    const int channels = 1;
    const int input_h = 4;
    const int input_w = 4;
    const int kernel_size = 2;
    const int stride = 2;
    const int padding = 0;
    
    float input_data[16] = {
        1.0f, 2.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 13.0f, 14.0f,
        11.0f, 12.0f, 15.0f, 16.0f
    };
    
    float expected[4] = {4.0f, 8.0f, 12.0f, 16.0f};
    
    auto input_buffer = std::make_shared<CudaBuffer>(batch_size * channels * input_h * input_w * sizeof(float));
    auto output_buffer = std::make_shared<CudaBuffer>(batch_size * channels * 2 * 2 * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(input_buffer->data, input_data, sizeof(input_data), cudaMemcpyHostToDevice));
    
    MaxPool2d maxpool(kernel_size, stride, padding);
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    maxpool.forward(input_buffer, output_buffer, batch_size, channels, input_h, input_w, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float output_data[4];
    CHECK_CUDA(cudaMemcpy(output_data, output_buffer->data, sizeof(output_data), cudaMemcpyDeviceToHost));
    
    int out_h, out_w;
    maxpool.get_output_size(input_h, input_w, out_h, out_w);
    assert(out_h == 2 && out_w == 2);
    
    std::cout << "MaxPool2d test output:" << std::endl;
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            int idx = h * out_w + w;
            std::cout << "Output[" << h << "][" << w << "]: " << output_data[idx] 
                      << " (Expected: " << expected[idx] << ")" << std::endl;
            assert(std::abs(output_data[idx] - expected[idx]) < 1e-6f);
        }
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    return true;
}

bool test_maxpool2d_with_padding() {
    const int batch_size = 1;
    const int channels = 1;
    const int input_h = 3;
    const int input_w = 3;
    const int kernel_size = 2;
    const int stride = 1;
    const int padding = 1;
    
    float input_data[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    
    auto input_buffer = std::make_shared<CudaBuffer>(batch_size * channels * input_h * input_w * sizeof(float));
    
    MaxPool2d maxpool(kernel_size, stride, padding);
    int out_h, out_w;
    maxpool.get_output_size(input_h, input_w, out_h, out_w);
    
    auto output_buffer = std::make_shared<CudaBuffer>(batch_size * channels * out_h * out_w * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(input_buffer->data, input_data, sizeof(input_data), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    maxpool.forward(input_buffer, output_buffer, batch_size, channels, input_h, input_w, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float* output_data = new float[out_h * out_w];
    CHECK_CUDA(cudaMemcpy(output_data, output_buffer->data, out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "MaxPool2d with padding test output (" << out_h << "x" << out_w << "):" << std::endl;
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            std::cout << output_data[h * out_w + w] << " ";
        }
        std::cout << std::endl;
    }
    
    delete[] output_data;
    CHECK_CUDA(cudaStreamDestroy(stream));
    return true;
}

int main() {
    
    if (test_maxpool2d_basic()) {
        std::cout << "MaxPool2d test passed!" << std::endl;
    } else {
        std::cout << "MaxPool2d test failed!" << std::endl;
        return -1;
    }
    
    if (test_maxpool2d_with_padding()) {
        std::cout << "MaxPool2d with padding test passed!" << std::endl;
    } else {
        std::cout << "MaxPool2d with padding test failed!" << std::endl;
        return -1;
    }
    
    std::cout << "All MaxPool2d tests passed!" << std::endl;
    return 0;
} 