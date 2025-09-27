#include "CudaBuffer.cuh"
#include "HostBuffer.h"
#include "ErrorCheck.h"

CudaBuffer::CudaBuffer(void *data, size_t size) : data(data), size(size) {}

CudaBuffer::CudaBuffer(const HostBuffer &host_buffer) : size(host_buffer.size) {
    checkCuda(cudaMallocManaged(&data, size));
    checkCuda(cudaMemcpy(data, host_buffer.data, size, cudaMemcpyHostToDevice));
}

CudaBuffer::CudaBuffer(size_t size) : size(size) {
    checkCuda(cudaMallocManaged(&data, size));
    checkCuda(cudaMemset(data, 0, size));
}

CudaBuffer::~CudaBuffer() {
    if (data) {
        cudaFree(data);
    }
} 