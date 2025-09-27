#pragma once

class HostBuffer;

class CudaBuffer {
public:
    CudaBuffer(void *data, size_t size);
    explicit CudaBuffer(const HostBuffer &host_buffer);
    explicit CudaBuffer(size_t size);
    ~CudaBuffer();

    void* data{};
    size_t size{};
}; 