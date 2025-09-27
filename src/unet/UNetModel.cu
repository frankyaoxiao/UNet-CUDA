#include "UNetModel.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/Interpolate.cuh"
#include <cuda_runtime.h>
#include <iostream>

constexpr int UNetModel::FEATURES[UNetModel::NUM_FEATURES];

UNetModel::UNetModel(int input_channels, int output_channels, 
                     int input_height, int input_width)
    : input_channels(input_channels), output_channels(output_channels),
      input_height(input_height), input_width(input_width)
{
    encoder_conv1.resize(NUM_FEATURES);
    encoder_conv2.resize(NUM_FEATURES);
    
    int in_ch = input_channels;
    for (int i = 0; i < NUM_FEATURES; i++) {
        encoder_conv1[i] = std::make_unique<Conv2d>(in_ch, FEATURES[i], 3, 1, 1, true);
        encoder_conv2[i] = std::make_unique<Conv2d>(FEATURES[i], FEATURES[i], 3, 1, 1, true);
        in_ch = FEATURES[i];
    }
    
    bottleneck_conv1 = std::make_unique<Conv2d>(FEATURES[NUM_FEATURES-1], FEATURES[NUM_FEATURES-1] * 2, 3, 1, 1, true);
    bottleneck_conv2 = std::make_unique<Conv2d>(FEATURES[NUM_FEATURES-1] * 2, FEATURES[NUM_FEATURES-1] * 2, 3, 1, 1, true);
    
    upconvs.resize(NUM_FEATURES);
    decoder_conv1.resize(NUM_FEATURES);
    decoder_conv2.resize(NUM_FEATURES);
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        int idx = NUM_FEATURES - 1 - i;
        int upconv_in = (i == 0) ? FEATURES[NUM_FEATURES-1] * 2 : FEATURES[idx + 1];
        
        upconvs[i] = std::make_unique<ConvTranspose2d>(upconv_in, FEATURES[idx], 2, 2, 0, true);
        decoder_conv1[i] = std::make_unique<Conv2d>(FEATURES[idx] * 2, FEATURES[idx], 3, 1, 1, true);
        decoder_conv2[i] = std::make_unique<Conv2d>(FEATURES[idx], FEATURES[idx], 3, 1, 1, true);
    }
    
    final_conv = std::make_unique<Conv2d>(FEATURES[0], output_channels, 1, 0, 1, true);
    
    pool = std::make_unique<MaxPool2d>(2, 2, 0);
    
    allocate_temp_buffers();
    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
}

UNetModel::~UNetModel()
{
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

void UNetModel::allocate_temp_buffers()
{
    encoder_outputs.resize(NUM_FEATURES);
    temp_buffers.resize(10);
    
    int h = input_height, w = input_width;
    for (int i = 0; i < NUM_FEATURES; i++) {
        size_t size = 1 * FEATURES[i] * h * w * sizeof(float);
        encoder_outputs[i] = std::make_shared<CudaBuffer>(size);
        h /= 2;
        w /= 2;
    }
    
    size_t max_elements = 0;
    int temp_h = input_height, temp_w = input_width;
    
    for (int i = 0; i <= NUM_FEATURES; i++) {
        if (i < NUM_FEATURES) {
            size_t encoder_elements = 1 * FEATURES[i] * temp_h * temp_w;
            max_elements = std::max(max_elements, encoder_elements);
        }
        if (i == NUM_FEATURES) {
            // Bottleneck stage  
            size_t bottleneck_elements = 1 * (FEATURES[NUM_FEATURES-1] * 2) * temp_h * temp_w;
            max_elements = std::max(max_elements, bottleneck_elements);
        }
        if (i > 0) {
            // Decoder stage 
            temp_h *= 2;
            temp_w *= 2;
            if (i <= NUM_FEATURES) {
                int idx = NUM_FEATURES - i;
                if (idx >= 0) {
                    size_t decoder_elements = 1 * (FEATURES[idx] * 2) * temp_h * temp_w;
                    max_elements = std::max(max_elements, decoder_elements);
                }
            }
        } else {
            temp_h /= 2;
            temp_w /= 2;
        }
    }
    
    for (int i = 0; i < temp_buffers.size(); i++) {
        size_t size = max_elements * sizeof(float);
        temp_buffers[i] = std::make_shared<CudaBuffer>(size);
    }
}

void UNetModel::forward(const std::shared_ptr<CudaBuffer>& input,
                        const std::shared_ptr<CudaBuffer>& output,
                        cudaStream_t stream)
{
    auto current_input = input;
    int current_h = input_height, current_w = input_width;
    
    // Encoder path
    for (int i = 0; i < NUM_FEATURES; i++) {
        // First convolution with fused ReLU
        encoder_conv1[i]->forward_relu(current_input, temp_buffers[0], 1, current_h, current_w, stream);
        
        // Second convolution 
        encoder_conv2[i]->forward_relu(temp_buffers[0], encoder_outputs[i], 1, current_h, current_w, stream);
        
        // Max pooling everything but last layer
        if (i < NUM_FEATURES - 1) {
            current_h /= 2;
            current_w /= 2;
            pool->forward(encoder_outputs[i], temp_buffers[1], 1, FEATURES[i], current_h * 2, current_w * 2, stream);
            current_input = temp_buffers[1];
        } else {
            current_input = encoder_outputs[i];
        }
    }
    
    // Bottleneck
    bottleneck_conv1->forward_relu(current_input, temp_buffers[2], 1, current_h, current_w, stream);
    bottleneck_conv2->forward_relu(temp_buffers[2], temp_buffers[3], 1, current_h, current_w, stream);
    current_input = temp_buffers[3];
    
    // Decoder path
    for (int decoder_stage = 0; decoder_stage < NUM_FEATURES; decoder_stage++) {
        int encoder_stage = NUM_FEATURES - 1 - decoder_stage;
        
        // Transpose convolution
        upconvs[decoder_stage]->forward(current_input, temp_buffers[4], 1, current_h, current_w, stream);
        
        // Calculate upconv output size
        int upconv_h = current_h * 2;
        int upconv_w = current_w * 2;
        
        // Get skip connection size
        int skip_h, skip_w;
        if (encoder_stage == 3) { skip_h = 16; skip_w = 16; }      
        else if (encoder_stage == 2) { skip_h = 32; skip_w = 32; } 
        else if (encoder_stage == 1) { skip_h = 64; skip_w = 64; } 
        else { skip_h = 128; skip_w = 128; }                       
        
        std::shared_ptr<CudaBuffer> upconv_result;
        if (upconv_h != skip_h || upconv_w != skip_w) {
            Interpolate::forward(temp_buffers[4], temp_buffers[8], 
                               1, FEATURES[encoder_stage], 
                               upconv_h, upconv_w, 
                               skip_h, skip_w, stream);
            upconv_result = temp_buffers[8];
            current_h = skip_h;
            current_w = skip_w;
        } else {
            upconv_result = temp_buffers[4];
            current_h = upconv_h;
            current_w = upconv_w;
        }
        
        Concat::forward(encoder_outputs[encoder_stage], upconv_result, temp_buffers[5],
                       1, FEATURES[encoder_stage], FEATURES[encoder_stage], current_h, current_w, stream);
        
        decoder_conv1[decoder_stage]->forward_relu(temp_buffers[5], temp_buffers[6], 1, current_h, current_w, stream);
        decoder_conv2[decoder_stage]->forward_relu(temp_buffers[6], temp_buffers[7], 1, current_h, current_w, stream);
        
        current_input = temp_buffers[7];
    }
    
    final_conv->forward(current_input, temp_buffers[8], 1, input_height, input_width, stream);
    
    // sigmoid for segmentation 
    Sigmoid::forward(temp_buffers[8], output, input_height * input_width * output_channels, stream);
} 