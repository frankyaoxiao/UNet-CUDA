#pragma once

#include "UNetModel.cuh"
#include <string>

class WeightLoader {
public:
    /**
     * Load weights from PyTorch state dict file (.pth)
     * @param model UNet model to load weights into
     * @param weight_path Path to PyTorch weights file
     */
    static void load_weights(UNetModel& model, const std::string& weight_path);

private:
    static void load_conv_weights(const std::string& name, Conv2d* conv_layer, void* weights_dict);
    static void load_conv_transpose_weights(const std::string& name, ConvTranspose2d* conv_layer, void* weights_dict);
}; 