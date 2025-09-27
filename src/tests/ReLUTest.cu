#include <iostream>
#include <cassert>
#include <cmath>
#include "../gpu_ops/ReLU.cuh"
#include "../ErrorCheck.h"

bool test_relu_basic() {
    const int size = 8;
    
    float input_data[8] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 3.0f, -3.0f};
    float expected[8] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 3.0f, 0.0f};
    
    auto input_buffer = std::make_shared<CudaBuffer>(size * sizeof(float));
    auto output_buffer = std::make_shared<CudaBuffer>(size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(input_buffer->data, input_data, sizeof(input_data), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ReLU::forward(input_buffer, output_buffer, size, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float output_data[8];
    CHECK_CUDA(cudaMemcpy(output_data, output_buffer->data, sizeof(output_data), cudaMemcpyDeviceToHost));
    
    std::cout << "ReLU test results:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << "Input: " << input_data[i] << " -> Output: " << output_data[i] 
                  << " (Expected: " << expected[i] << ")" << std::endl;
        assert(std::abs(output_data[i] - expected[i]) < 1e-6f);
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    return true;
}

bool test_relu_inplace() {
    const int size = 6;
    
    float data[6] = {-1.5f, 0.0f, 2.5f, -0.1f, 1.0f, -2.0f};
    float expected[6] = {0.0f, 0.0f, 2.5f, 0.0f, 1.0f, 0.0f};
    
    auto buffer = std::make_shared<CudaBuffer>(size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(buffer->data, data, sizeof(data), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ReLU::forward_inplace(buffer, size, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float output_data[6];
    CHECK_CUDA(cudaMemcpy(output_data, buffer->data, sizeof(output_data), cudaMemcpyDeviceToHost));
    
    std::cout << "ReLU in-place test results:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << "Input: " << data[i] << " -> Output: " << output_data[i] 
                  << " (Expected: " << expected[i] << ")" << std::endl;
        assert(std::abs(output_data[i] - expected[i]) < 1e-6f);
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    return true;
}

int main() {
    
    if (test_relu_basic()) {
        std::cout << "ReLU test passed!" << std::endl;
    } else {
        std::cout << "ReLU test failed!" << std::endl;
        return -1;
    }
    
    if (test_relu_inplace()) {
        std::cout << "ReLU in-place test passed!" << std::endl;
    } else {
        std::cout << "ReLU in-place test failed!" << std::endl;
        return -1;
    }
    
    std::cout << "All ReLU tests passed!" << std::endl;
    return 0;
} 