#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include "unet/UNetModel.cuh"
#include "unet/WeightLoader.h"
#include "ErrorCheck.h"

void generate_random_input(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> [num_iterations]" << std::endl;
        return -1;
    }
    
    std::string weights_path = argv[1];
    int num_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    
    const int HEIGHT = 128;
    const int WIDTH = 128;
    const int BATCH_SIZE = 1;
    
    try {
        cudaSetDevice(0);
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        
        auto input_buffer = std::make_shared<CudaBuffer>(BATCH_SIZE * 3 * HEIGHT * WIDTH * sizeof(float));
        auto output_buffer = std::make_shared<CudaBuffer>(BATCH_SIZE * 1 * HEIGHT * WIDTH * sizeof(float));
        
        float* host_input = new float[BATCH_SIZE * 3 * HEIGHT * WIDTH];
        generate_random_input(host_input, BATCH_SIZE * 3 * HEIGHT * WIDTH);
        CHECK_CUDA(cudaMemcpy(input_buffer->data, host_input, BATCH_SIZE * 3 * HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
        
        std::cout << "Initializing UNet model..." << std::endl;
        UNetModel model(3, 1, HEIGHT, WIDTH);
        
        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        WeightLoader::load_weights(model, weights_path);
        
        std::cout << "Running warmup iterations..." << std::endl;
        for (int i = 0; i < 10; i++) {
            model.forward(input_buffer, output_buffer, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        
        std::cout << "Running " << num_iterations << " benchmark iterations..." << std::endl;
        std::vector<double> times;
        times.reserve(num_iterations);
        
        for (int i = 0; i < num_iterations; i++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            model.forward(input_buffer, output_buffer, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            times.push_back(duration.count() / 1000.0); 
            
            if ((i + 1) % 10 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_iterations << " iterations" << std::endl;
            }
        }
        
        double total_time = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double avg_time = total_time / num_iterations;
        double fps = 1000.0 / avg_time;
        
        double variance = 0.0;
        for (double time : times) {
            variance += (time - avg_time) * (time - avg_time);
        }
        double std_dev = sqrt(variance / num_iterations);
        
        std::cout << "\nUNet CUDA Benchmark Results" << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
        std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;
        std::cout << "Throughput: " << fps << " FPS" << std::endl;
        
        size_t model_memory = 0;
        for (const auto& layer : model.encoder_conv1) {
            model_memory += layer->weights->size;
        }
        for (const auto& layer : model.encoder_conv2) {
            model_memory += layer->weights->size;
        }
        model_memory += model.bottleneck_conv1->weights->size;
        model_memory += model.bottleneck_conv2->weights->size;
        for (const auto& layer : model.upconvs) {
            model_memory += layer->weights->size;
        }
        for (const auto& layer : model.decoder_conv1) {
            model_memory += layer->weights->size;
        }
        for (const auto& layer : model.decoder_conv2) {
            model_memory += layer->weights->size;
        }
        model_memory += model.final_conv->weights->size;
        
        std::cout << "Estimated model memory: " << model_memory / (1024 * 1024) << " MB" << std::endl;
        
        delete[] host_input;
        CHECK_CUDA(cudaStreamDestroy(stream));
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 