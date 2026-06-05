#include "yolos/core/MemoryArena.hpp"
#include <iostream>
#include <cstring>
#include <cassert>

namespace yolos {

MemoryArena::MemoryArena(size_t totalSizeBytes) {
    arena_.resize(totalSizeBytes, 0);
    std::cout << "[INFO] MemoryArena initialized: " 
              << (totalSizeBytes >> 20) << " MB" << std::endl;
}

void* MemoryArena::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

    // Align the current offset
    size_t alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);

    if (alignedOffset + size > arena_.size()) {
        std::cerr << "[ERROR] MemoryArena out of memory! Requested: " << size 
                  << " bytes | Used: " << offset_ << "/" << arena_.size() << std::endl;
        return nullptr;
    }

    void* ptr = arena_.data() + alignedOffset;
    offset_ = alignedOffset + size;

    return ptr;
}

void MemoryArena::reset() {
    offset_ = 0;
}

void MemoryArena::clear() {
    std::memset(arena_.data(), 0, arena_.size());
    reset();
}

} // namespace yolos
