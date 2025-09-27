import torch
import struct
import os
import sys
import numpy as np

def fuse_conv_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Fuse Conv2d and BatchNorm2d layers
    """
    bn_scale = bn_weight / torch.sqrt(bn_var + eps)
    
    fused_weight = conv_weight * bn_scale.view(-1, 1, 1, 1)
    
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.shape[0])
    fused_bias = (conv_bias - bn_mean) * bn_scale + bn_bias
    
    return fused_weight, fused_bias

def export_fused_unet_weights(model_path, output_path):
    """
    Export UNet weights with BatchNorm fused into Conv layers
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    fused_weights = {}
    
    layer_groups = [
        ('encoder.0.double_conv.0', 'encoder.0.double_conv.1'),
        ('encoder.0.double_conv.3', 'encoder.0.double_conv.4'),
        ('encoder.1.double_conv.0', 'encoder.1.double_conv.1'),
        ('encoder.1.double_conv.3', 'encoder.1.double_conv.4'),
        ('encoder.2.double_conv.0', 'encoder.2.double_conv.1'),
        ('encoder.2.double_conv.3', 'encoder.2.double_conv.4'),
        ('encoder.3.double_conv.0', 'encoder.3.double_conv.1'),
        ('encoder.3.double_conv.3', 'encoder.3.double_conv.4'),
        
        ('bottleneck.double_conv.0', 'bottleneck.double_conv.1'),
        ('bottleneck.double_conv.3', 'bottleneck.double_conv.4'),
        
        ('decoder.0.double_conv.0', 'decoder.0.double_conv.1'),
        ('decoder.0.double_conv.3', 'decoder.0.double_conv.4'),
        ('decoder.1.double_conv.0', 'decoder.1.double_conv.1'),
        ('decoder.1.double_conv.3', 'decoder.1.double_conv.4'),
        ('decoder.2.double_conv.0', 'decoder.2.double_conv.1'),
        ('decoder.2.double_conv.3', 'decoder.2.double_conv.4'),
        ('decoder.3.double_conv.0', 'decoder.3.double_conv.1'),
        ('decoder.3.double_conv.3', 'decoder.3.double_conv.4'),
    ]
    
    for conv_name, bn_name in layer_groups:
        conv_weight_key = f"{conv_name}.weight"
        bn_weight_key = f"{bn_name}.weight"
        bn_bias_key = f"{bn_name}.bias"
        bn_mean_key = f"{bn_name}.running_mean"
        bn_var_key = f"{bn_name}.running_var"
        
        if all(key in state_dict for key in [conv_weight_key, bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]):
            conv_weight = state_dict[conv_weight_key]
            conv_bias = None
            bn_weight = state_dict[bn_weight_key]
            bn_bias = state_dict[bn_bias_key]
            bn_mean = state_dict[bn_mean_key]
            bn_var = state_dict[bn_var_key]
            
            fused_weight, fused_bias = fuse_conv_bn(
                conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var
            )
            
            fused_weights[f"{conv_name}.weight"] = fused_weight
            fused_weights[f"{conv_name}.bias"] = fused_bias
            
            print(f"Fused {conv_name} + {bn_name}")
    
    upconv_layers = ['upconvs.0', 'upconvs.1', 'upconvs.2', 'upconvs.3']
    for layer in upconv_layers:
        weight_key = f"{layer}.weight"
        bias_key = f"{layer}.bias"
        if weight_key in state_dict:
            fused_weights[weight_key] = state_dict[weight_key]
        if bias_key in state_dict:
            fused_weights[bias_key] = state_dict[bias_key]
    
    if 'final_conv.weight' in state_dict:
        fused_weights['final_conv.weight'] = state_dict['final_conv.weight']
    if 'final_conv.bias' in state_dict:
        fused_weights['final_conv.bias'] = state_dict['final_conv.bias']
    
    print(f"\nFused weights summary:")
    for key in sorted(fused_weights.keys()):
        tensor = fused_weights[key]
        print(f"  {key}: {tensor.shape}")
    
    with open(output_path, 'wb') as f:
        f.write(b'FUSED')
        f.write(struct.pack('I', 1))
        f.write(struct.pack('I', len(fused_weights)))
        
        for name, tensor in fused_weights.items():
            name_bytes = name.encode('utf-8')
            tensor_data = tensor.float().numpy()
            
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('I', dim))
            
            tensor_bytes = tensor_data.tobytes()
            f.write(struct.pack('I', len(tensor_bytes)))
            f.write(tensor_bytes)
    
    print(f"Exported fused weights to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    export_fused_unet_weights(model_path, output_path) 