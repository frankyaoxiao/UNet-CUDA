#!/bin/bash
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "To run benchmark: ./build/unet_benchmark unet_weights_proper.bin 15"
echo "To run demo: python3 demo_comparison.py" 