#pragma once

#include "../CudaBuffer.cuh"
#include "../gpu_ops/Conv2d.cuh"
#include "../gpu_ops/MaxPool2d.cuh"
#include "../gpu_ops/ConvTranspose2d.cuh"
#include "../gpu_ops/Concat.cuh"
#include "../gpu_ops/Sigmoid.cuh"
#include <memory>
#include <vector>

class UNetModel {
public:
    static constexpr int NUM_FEATURES = 4;
    static constexpr int FEATURES[NUM_FEATURES] = {64, 128, 256, 512};
    
    std::vector<std::unique_ptr<Conv2d>> encoder_conv1;
    std::vector<std::unique_ptr<Conv2d>> encoder_conv2;
    
    std::unique_ptr<Conv2d> bottleneck_conv1;
    std::unique_ptr<Conv2d> bottleneck_conv2;
    
    std::vector<std::unique_ptr<ConvTranspose2d>> upconvs;
    std::vector<std::unique_ptr<Conv2d>> decoder_conv1;
    std::vector<std::unique_ptr<Conv2d>> decoder_conv2;
    
    std::unique_ptr<Conv2d> final_conv;
    
    std::unique_ptr<MaxPool2d> pool;
    
    std::vector<std::shared_ptr<CudaBuffer>> encoder_outputs;
    std::vector<std::shared_ptr<CudaBuffer>> temp_buffers;
    
    cudaStream_t stream1, stream2;
    
    int input_channels;
    int output_channels;
    int input_height;
    int input_width;

    UNetModel(int input_channels = 3, int output_channels = 1, 
              int input_height = 128, int input_width = 128);

    ~UNetModel();

    /**
     * Forward pass through the network
     * @param input GPU float32 input of shape (1, input_channels, input_height, input_width)
     * @param output GPU float32 output of shape (1, output_channels, input_height, input_width)
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& input,
                 const std::shared_ptr<CudaBuffer>& output,
                 cudaStream_t stream);

private:
    void allocate_temp_buffers();
}; 