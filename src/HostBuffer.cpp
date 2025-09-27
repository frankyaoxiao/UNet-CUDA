#include "HostBuffer.h"
#include <cstdlib>

HostBuffer::HostBuffer(size_t size) : size(size) {
    data = std::malloc(size);
}

HostBuffer::~HostBuffer() {
    if (data) {
        std::free(data);
    }
} 