#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace yolos {

/**
 * Simple Linear Arena Allocator (bump allocator)
 * Fast, cache-friendly, no fragmentation for inference buffers
 */
class MemoryArena {
public:
    explicit MemoryArena(size_t totalSizeBytes = 256ULL * 1024 * 1024);
    ~MemoryArena() = default;

    void* allocate(size_t size, size_t alignment = 64);  // default 64-byte alignment
    void reset();                                         // reset for next inference
    void clear();                                         // zero memory (optional)

    size_t used() const { return offset_; }
    size_t capacity() const { return arena_.size(); }

private:
    std::vector<uint8_t> arena_;
    size_t offset_ = 0;
};

} // namespace yolos
