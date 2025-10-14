/**
 * @file step_arena.h  
 * @brief Step-level memory arena (temporary tensor reuse within training step)
 * 
 * Purpose:
 * - Temporary tensors in forward/backward (score blocks, hidden blocks, transposes, etc.) allocated from arena
 * - Reset at end of step, reuse same memory block in next step
 * - Reduce fragmentation, stabilize RSS
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace ops {

class StepArena {
private:
    std::vector<char> buffer_;
    size_t offset_;
    size_t capacity_;
    
public:
    explicit StepArena(size_t capacity_mb = 64) 
        : offset_(0), capacity_(capacity_mb * 1024 * 1024) {
        buffer_.resize(capacity_);
    }
    
    /**
     * @brief Allocate aligned memory
     */
    void* allocate(size_t size, size_t alignment = 64) {
        // Align to alignment
        size_t aligned_offset = (offset_ + alignment - 1) / alignment * alignment;
        
        if (aligned_offset + size > capacity_) {
            throw std::runtime_error("StepArena exhausted: need " + std::to_string(size) + 
                                   " bytes but only " + std::to_string(capacity_ - aligned_offset) + " available");
        }
        
        void* ptr = &buffer_[aligned_offset];
        offset_ = aligned_offset + size;
        
        return ptr;
    }
    
    /**
     * @brief Allocate float array
     */
    float* allocate_floats(size_t count) {
        return static_cast<float*>(allocate(count * sizeof(float), 64));
    }
    
    /**
     * @brief Reset arena (called at end of step)
     */
    void reset() {
        offset_ = 0;
    }
    
    /**
     * @brief Get current usage
     */
    size_t current_usage() const {
        return offset_;
    }
    
    size_t get_capacity() const {
        return capacity_;
    }
};

// Global step-level arena (singleton)
StepArena& get_step_arena();

} // namespace ops

