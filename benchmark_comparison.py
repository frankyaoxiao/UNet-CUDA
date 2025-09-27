import torch
import numpy as np
import sys
import os
import subprocess
import time
from PIL import Image

def create_random_input(seed):
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

def save_input_for_cuda(input_tensor, filename="bench_input.ppm"):
    input_np = input_tensor.squeeze(0).numpy()
    
    hwc_data = np.transpose(input_np, (1, 2, 0))
    
    hwc_uint8 = (hwc_data * 255).astype(np.uint8)
    
    with open(filename, "wb") as f:
        f.write(f"P6\n{128} {128}\n255\n".encode())
        f.write(hwc_uint8.tobytes())
    
    return filename

def run_python_unet(input_image, model):
    start_time = time.time()
    with torch.no_grad():
        output = model(input_image)
        sigmoid_output = torch.sigmoid(output)
    end_time = time.time()
    
    return (end_time - start_time) * 1000

def run_cuda_unet(input_image):
    input_file = save_input_for_cuda(input_image, "bench_input.ppm")
    
    start_time = time.time()
    result = subprocess.run(['./build/unet_demo', 'unet_weights_proper.bin', '--image', input_file, '--output', 'bench_cuda'], 
                          capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        return None
    
    cuda_time_ms = None
    for line in result.stdout.split('\n'):
        if 'Inference time:' in line:
            try:
                cuda_time_ms = float(line.split(':')[1].strip().split()[0])
                break
            except:
                pass
    
    if cuda_time_ms is None:
        cuda_time_ms = (end_time - start_time) * 1000
    
    return cuda_time_ms

def load_python_model():
    from model import Unet
    model = Unet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('./unet.pth', map_location='cpu'))
    model.eval()
    return model

def main():
    python_model = load_python_model()
    
    python_times = []
    cuda_times = []
    cuda_failures = 0
    
    for i in range(100):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/100")
        
        input_image = create_random_input(i + 1000)
        
        python_time = run_python_unet(input_image, python_model)
        python_times.append(python_time)
        
        cuda_time = run_cuda_unet(input_image)
        if cuda_time is not None:
            cuda_times.append(cuda_time)
        else:
            cuda_failures += 1
    
    python_avg = np.mean(python_times)
    python_std = np.std(python_times)
    python_min = np.min(python_times)
    python_max = np.max(python_times)
    
    if cuda_times:
        cuda_avg = np.mean(cuda_times)
        cuda_std = np.std(cuda_times)
        cuda_min = np.min(cuda_times)
        cuda_max = np.max(cuda_times)
        speedup = python_avg / cuda_avg
    else:
        cuda_avg = cuda_std = cuda_min = cuda_max = speedup = 0
    
    # Cleanup temporary files
    for filename in ['bench_input.ppm', 'bench_cuda_segmentation.png', 'bench_cuda_input.png', 'bench_cuda_ground_truth.png']:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nPython UNet:")
    print(f"  Average time: {python_avg:.2f} ± {python_std:.2f} ms")
    
    print(f"\nCUDA UNet:")
    print(f"  Average time: {cuda_avg:.2f} ± {cuda_std:.2f} ms")
    
    if cuda_times:
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"CUDA failures: {cuda_failures}/100")
    else:
        print("\nAll CUDA runs failed!")

if __name__ == "__main__":
    main() 