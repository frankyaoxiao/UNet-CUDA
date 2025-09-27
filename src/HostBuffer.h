#pragma once

#include <cstddef>

class HostBuffer {
public:
    explicit HostBuffer(size_t size);
    ~HostBuffer();

    void* data{};
    size_t size{};
}; 