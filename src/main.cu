#include <iostream>
#include <string>
#include <cstring>
#include <random>
#include <cmath>
#include <fstream>
#include <memory>
#include <chrono>

#include "unet/UNetModel.cuh"
#include "unet/WeightLoader.h"
#include "CudaBuffer.cuh"
#include "HostBuffer.h"
#include "ErrorCheck.h"

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
};

bool load_image(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open image file " << filename << std::endl;
        return false;
    }
    
    std::string magic;
    file >> magic;
    
    if (magic == "P6") {
        img.channels = 3;
    } else if (magic == "P5") {
        img.channels = 1;
    } else {
        std::cerr << "Error: Unsupported image format " << magic << std::endl;
        return false;
    }
    
    file >> img.width >> img.height;
    int maxval;
    file >> maxval;
    file.ignore(); 
    
    img.data.resize(img.width * img.height * img.channels);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    
    std::cout << "Loaded image: " << img.width << "x" << img.height << " channels=" << img.channels << std::endl;
    return true;
}

#ifdef HAS_OPENCV
bool save_png(const std::string& filename, const float* data, int width, int height, bool is_grayscale = true) {
    try {
        if (is_grayscale) {
            cv::Mat img(height, width, CV_8UC1);
            for (int i = 0; i < width * height; i++) {
                img.data[i] = static_cast<unsigned char>(data[i] * 255.0f);
            }
            cv::imwrite(filename, img);
        } else {
            cv::Mat img(height, width, CV_8UC3);
            for (int i = 0; i < width * height * 3; i++) {
                float clamped = std::min(255.0f, std::max(0.0f, data[i] * 255.0f));
                img.data[i] = static_cast<unsigned char>(clamped);
            }
            cv::imwrite(filename, img);
        }
        std::cout << "Saved " << (is_grayscale ? "segmentation" : "input image") << " to: " << filename << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error saving " << filename << ": " << e.what() << std::endl;
        return false;
    }
}
#endif

bool save_segmentation(const std::string& filename, const float* data, int width, int height) {
#ifdef HAS_OPENCV
    std::string png_filename = filename;
    if (png_filename.find('.') == std::string::npos) {
        png_filename += ".png";
    }
    
    return save_png(png_filename, data, width, height, true);
#else
    std::cerr << "Error: OpenCV not available for PNG output" << std::endl;
    return false;
#endif
}

bool save_input_image(const std::string& filename, const float* data, int width, int height) {
#ifdef HAS_OPENCV
    std::string png_filename = filename;
    if (png_filename.find('.') == std::string::npos) {
        png_filename += ".png";
    }
    
    return save_png(png_filename, data, width, height, false);
#else
    std::cerr << "Error: OpenCV not available for PNG output" << std::endl;
    return false;
#endif
}

void generate_random_shape(float* data, float* ground_truth, int width, int height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(10, std::min(width, height) / 2);
    std::uniform_real_distribution<> color_dist(0.0, 1.0);
    
    std::fill(data, data + width * height * 3, 0.0f);
    std::fill(ground_truth, ground_truth + width * height, 0.0f);
    
    int w = size_dist(gen);
    int h = size_dist(gen);
    int x1 = std::uniform_int_distribution<>(0, width - w)(gen);
    int y1 = std::uniform_int_distribution<>(0, height - h)(gen);
    int x2 = x1 + w;
    int y2 = y1 + h;
    
    float r = color_dist(gen);
    float g = color_dist(gen);
    float b = color_dist(gen);
    
    int spatial_size = width * height;
    for (int y = y1; y < y2; y++) {
        for (int x = x1; x < x2; x++) {
            int hw_idx = y * width + x;
            
            data[0 * spatial_size + hw_idx] = r;
            data[1 * spatial_size + hw_idx] = g;
            data[2 * spatial_size + hw_idx] = b;
            
            ground_truth[hw_idx] = 1.0f;
        }
    }
    
    std::cout << "Generated training-style shape: " << w << "x" << h << " at (" << x1 << "," << y1 << ")" << std::endl;
    std::cout << "Colors: R=" << r << ", G=" << g << ", B=" << b << " (range [0,1])" << std::endl;
}

void preprocess_image(const Image& input, float* output, int target_width, int target_height) {
    float x_scale = static_cast<float>(input.width) / target_width;
    float y_scale = static_cast<float>(input.height) / target_height;
    
    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            int src_x = static_cast<int>(x * x_scale);
            int src_y = static_cast<int>(y * y_scale);
            
            src_x = std::min(input.width - 1, std::max(0, src_x));
            src_y = std::min(input.height - 1, std::max(0, src_y));
            
            int src_idx = (src_y * input.width + src_x) * input.channels;
            int dst_idx = (y * target_width + x) * 3;
            
            if (input.channels == 3) {
                output[dst_idx + 0] = input.data[src_idx + 0] / 255.0f;
                output[dst_idx + 1] = input.data[src_idx + 1] / 255.0f;
                output[dst_idx + 2] = input.data[src_idx + 2] / 255.0f;
            } else {
                float gray = input.data[src_idx] / 255.0f;
                output[dst_idx + 0] = gray;
                output[dst_idx + 1] = gray;
                output[dst_idx + 2] = gray;
            }
        }
    }
}

void rgb_to_chw(const float* rgb_data, float* chw_data, int width, int height) {
    int spatial_size = width * height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int hwc_idx = (y * width + x) * 3;
            int hw_idx = y * width + x;
            
            chw_data[0 * spatial_size + hw_idx] = rgb_data[hwc_idx + 0];
            chw_data[1 * spatial_size + hw_idx] = rgb_data[hwc_idx + 1];
            chw_data[2 * spatial_size + hw_idx] = rgb_data[hwc_idx + 2];
        }
    }
}

void chw_to_hwc(const float* chw_data, float* hwc_data, int width, int height) {
    int spatial_size = width * height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int hw_idx = y * width + x;
            int hwc_idx = hw_idx * 3;
            
            hwc_data[hwc_idx + 0] = chw_data[0 * spatial_size + hw_idx];
            hwc_data[hwc_idx + 1] = chw_data[1 * spatial_size + hw_idx];
            hwc_data[hwc_idx + 2] = chw_data[2 * spatial_size + hw_idx];
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        return 1;
    }
    
    std::string weights_file = argv[1];
    std::string image_file;
    std::string output_prefix = "gpu_demo";
    bool use_random = false;
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_file = argv[++i];
        } else if (strcmp(argv[i], "--random") == 0) {
            use_random = true;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } 
        else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            return 1;
        }
    }
    
    if (!use_random && image_file.empty()) {
        std::cerr << "Error: Must specify either --image or --random\n";
        return 1;
    }
    
    std::cout << "=== CUDA UNet Image Segmentation Demo ===" << std::endl;
    
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    const int INPUT_WIDTH = 128;
    const int INPUT_HEIGHT = 128;
    const int INPUT_CHANNELS = 3;
    const int OUTPUT_CHANNELS = 1;
    
    std::cout << "Creating UNet model..." << std::endl;
    UNetModel model(INPUT_CHANNELS, OUTPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
    
    std::cout << "Loading weights from " << weights_file << "..." << std::endl;
    WeightLoader::load_weights(model, weights_file);
    
    std::vector<float> input_chw(INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS);
    std::vector<float> ground_truth(INPUT_WIDTH * INPUT_HEIGHT, 0.0f);
    bool has_ground_truth = false;
    
    if (use_random) {
        std::cout << "Generating training-style random shape..." << std::endl;
        generate_random_shape(input_chw.data(), ground_truth.data(), INPUT_WIDTH, INPUT_HEIGHT);
        has_ground_truth = true;
    } else {
        std::cout << "Loading image from " << image_file << "..." << std::endl;
        Image img(0, 0, 0);
        if (!load_image(image_file, img)) {
            return 1;
        }
        
        std::cout << "Preprocessing image..." << std::endl;
        std::vector<float> input_rgb(INPUT_WIDTH * INPUT_HEIGHT * 3);
        preprocess_image(img, input_rgb.data(), INPUT_WIDTH, INPUT_HEIGHT);
        
        rgb_to_chw(input_rgb.data(), input_chw.data(), INPUT_WIDTH, INPUT_HEIGHT);
        
        has_ground_truth = false;
    }
    
    size_t input_size = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float);
    size_t output_size = INPUT_WIDTH * INPUT_HEIGHT * OUTPUT_CHANNELS * sizeof(float);
    
    auto input_buffer = std::make_shared<CudaBuffer>(input_size);
    auto output_buffer = std::make_shared<CudaBuffer>(output_size);
    
    CHECK_CUDA(cudaMemcpy(input_buffer->data, input_chw.data(), input_size, cudaMemcpyHostToDevice));
    
    std::cout << "Running UNet inference..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    model.forward(input_buffer, output_buffer, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    
    std::vector<float> output_data(INPUT_WIDTH * INPUT_HEIGHT * OUTPUT_CHANNELS);
    CHECK_CUDA(cudaMemcpy(output_data.data(), output_buffer->data, output_size, cudaMemcpyDeviceToHost));
    
    std::string input_filename = output_prefix + "_input.png";
    std::string output_filename = output_prefix + "_segmentation.png";
    std::string gt_filename = output_prefix + "_ground_truth.png";
    
    std::cout << "Saving results..." << std::endl;
    
    std::vector<float> input_hwc(INPUT_WIDTH * INPUT_HEIGHT * 3);
    chw_to_hwc(input_chw.data(), input_hwc.data(), INPUT_WIDTH, INPUT_HEIGHT);
    
    save_input_image(input_filename, input_hwc.data(), INPUT_WIDTH, INPUT_HEIGHT);
    save_segmentation(output_filename, output_data.data(), INPUT_WIDTH, INPUT_HEIGHT);
    
    if (has_ground_truth) {
        save_segmentation(gt_filename, ground_truth.data(), INPUT_WIDTH, INPUT_HEIGHT);
        std::cout << "Ground truth: " << gt_filename << std::endl;
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    std::cout << "=== Demo completed successfully! ===" << std::endl;
    std::cout << "Input image: " << input_filename << std::endl;
    std::cout << "Segmentation: " << output_filename << std::endl;
    if (has_ground_truth) {
        std::cout << "Ground truth: " << gt_filename << std::endl;
    }
    
    return 0;
} 