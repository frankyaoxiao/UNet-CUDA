import torch
import numpy as np
import sys
import os
import subprocess
import time
import matplotlib.pyplot as plt
from PIL import Image

def create_demo_input(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    image = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
    
    w = np.random.randint(20, 60)
    h = np.random.randint(20, 60) 
    x1 = np.random.randint(10, 128 - w - 10)
    y1 = np.random.randint(10, 128 - h - 10)
    x2 = x1 + w
    y2 = y1 + h
    
    colors = torch.rand(3)
    for c in range(3):
        image[0, c, y1:y2, x1:x2] = colors[c]
    
    return image

def save_input_for_cuda(input_tensor, filename="demo_input.ppm"):
    input_np = input_tensor.squeeze(0).numpy() 
    
    hwc_data = np.transpose(input_np, (1, 2, 0))
    
    hwc_uint8 = (hwc_data * 255).astype(np.uint8)
    
    with open(filename, "wb") as f:
        f.write(f"P6\n{128} {128}\n255\n".encode())
        f.write(hwc_uint8.tobytes())
    
    return filename

def generate_ground_truth(input_image):
    input_np = input_image.squeeze().numpy()
    mask = np.any(input_np > 0.01, axis=0).astype(np.float32)
    
    foreground_pixels = mask.sum()
    total_pixels = mask.size
    
    return mask

def test_python_unet(input_image):
    from model import Unet
    model = Unet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('./unet.pth', map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        _ = model(input_image)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_image)
        sigmoid_output = torch.sigmoid(output)
    end_time = time.time()
    
    python_time_ms = (end_time - start_time) * 1000
    python_result = sigmoid_output.squeeze().numpy()
    
    return python_result, python_time_ms

def test_cuda_unet(input_image):
    input_file = save_input_for_cuda(input_image, "demo_input.ppm")
    
    start_time = time.time()
    result = subprocess.run(['./build/unet_demo', 'unet_weights_proper.bin', '--image', input_file, '--output', 'demo_cuda'], 
                          capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return None, 0
    
    cuda_time_ms = None
    for line in result.stdout.split('\n'):
        if 'Inference time:' in line:
            try:
                cuda_time_ms = float(line.split(':')[1].strip().split()[0])
                break
            except:
                pass
    
    if cuda_time_ms is None:
        total_time_ms = (end_time - start_time) * 1000
        cuda_time_ms = total_time_ms  # Fallback to total time
    
    if os.path.exists('demo_cuda_segmentation.png'):
        img = Image.open('demo_cuda_segmentation.png').convert('L')
        cuda_result = np.array(img, dtype=np.float32) / 255.0
        
        print(f"CUDA inference: {cuda_time_ms:.1f} ms")
        print(f"Output range: [{cuda_result.min():.3f}, {cuda_result.max():.3f}], mean: {cuda_result.mean():.3f}")
        
        return cuda_result, cuda_time_ms
    else:
        return None, cuda_time_ms

def compute_metrics(pred, target):
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(np.maximum(pred_binary, target_binary))
    iou = intersection / (union + 1e-8)
    
    correlation = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
    
    mse = np.mean((pred - target) ** 2)
    
    return {
        'iou': iou,
        'correlation': correlation,
        'mse': mse
    }

def create_comparison_visualization(input_img, ground_truth, python_pred, cuda_pred, 
                                 python_time, cuda_time, filename='demo_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    input_rgb = input_img.squeeze().numpy().transpose(1, 2, 0)
    axes[0, 0].imshow(input_rgb)
    axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(python_pred, cmap='gray', vmin=0, vmax=1)
    py_metrics = compute_metrics(python_pred, ground_truth)
    axes[1, 0].set_title(f'Python UNet\n{python_time:.1f} ms\nIoU: {py_metrics["iou"]:.3f}', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    if cuda_pred is not None:
        axes[1, 1].imshow(cuda_pred, cmap='gray', vmin=0, vmax=1)
        cuda_metrics = compute_metrics(cuda_pred, ground_truth)
        axes[1, 1].set_title(f'CUDA UNet\n{cuda_time:.1f} ms\nIoU: {cuda_metrics["iou"]:.3f}', 
                            fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'CUDA Failed', ha='center', va='center', fontsize=16)
        axes[1, 1].set_title('CUDA UNet\nFailed', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('UNet Comparison', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')

def main():
    demo_input = create_demo_input(seed=42)
    ground_truth = generate_ground_truth(demo_input)
    
    python_pred, python_time = test_python_unet(demo_input)
    
    cuda_pred, cuda_time = test_cuda_unet(demo_input)
    
    if python_pred is not None and cuda_pred is not None:
        py_metrics = compute_metrics(python_pred, ground_truth)
        cuda_metrics = compute_metrics(cuda_pred, ground_truth)
        comparison_metrics = compute_metrics(cuda_pred, python_pred)
        
        create_comparison_visualization(demo_input, ground_truth, python_pred, cuda_pred,
                                      python_time, cuda_time)
        
        print(f"Python UNet: {python_time:.1f} ms")
        print(f"CUDA UNet: {cuda_time:.1f} ms")
        
        if cuda_time > 0:
            speedup = python_time / cuda_time
            print(f"Speedup: {speedup:.1f}x")
        
        print(f"Correlation: {comparison_metrics['correlation']:.3f}")
        
    else:
        print("\nDemo failed - could not compare results")

if __name__ == "__main__":
    main() 