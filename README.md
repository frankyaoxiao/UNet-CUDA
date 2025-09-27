# CUDA UNet Implementation 

## Overview

This is a GPU implementation of the popular UNet architecture for segmentation. This implementation achieves around
a 2x speedup compared to an optimized torch CPU version.

## Requirements
torch (for python comparisons)

## Usage

First, build the project using 

```bash
./build_clean.sh
```


In order to use the demo we provide, run

```
python3 demo_comparison.py
```

This will randomly generate a shape and compare the python and CUDA segmentation results and the corresponding runtimes. Note there is
a Python overhead introduced in these times. 

In order to get a more accurate pass@100 results, please run

```
python3 benchmark_comparison.py
```

If you just want the CUDA inference, use 

```
./build/unet_demo unet_weights_proper.bin --image input_file --output output_file
```

To use a UNet trained on some other data, please take the .pth file generated via torch and use 
``export_weights_fused.py`` in order to export it to a format that this implementation can use


## Project Features
Our Primary features are
1. CUDA implementation of UNet (2x speed of torch on CPU)
2. Benchmarking for GPU implementation vs CPU
3. Pipeline to generate images using CUDA implementation given input
4. Comprehensive Unit Tests


## Expected results
Sample demo runs are included in this folder as ``demo_comparison.png``. We expect the
torch inference and CUDA inference to produce essentially the same image no matter the input.
Note you might have to change the random seed to try a different rectangle with the implementations. 

```
Python UNet:
  Average time: 98.04 ± 12.81 ms

CUDA UNet:
  Average time: 51.23 ± 2.29 ms
```


