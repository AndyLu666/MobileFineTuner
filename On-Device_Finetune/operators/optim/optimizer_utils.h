/**
 * @file optimizer_utils.h
 * @brief Auxiliary utility functions for Optimizer state manager
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>

// Type definition (DType) imported from core module to avoid indirect inclusion by dependents
#include "core/dtype.h"

namespace ops {
namespace optim {

/**
 * @brief FP32 ↔ FP16 conversion tools
 * Based on IEEE 754 standard implementation
 */

// FP32 -> FP16 conversion
inline uint16_t float_to_half(float value) {
    // FP32 format: 1 sign bit + 8 exp bits + 23 mantissa bits
    // FP16 format: 1 sign bit + 5 exp bits + 10 mantissa bits
    
    // Use memcpy to avoid undefined behavior caused by aliasing rules
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(uint32_t));
    
    uint16_t sign = (bits >> 16) & 0x8000;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    // Handle special cases
    if (exponent == 0xFF) {
        // Inf or NaN
        return sign | 0x7C00 | (mantissa ? 0x0200 : 0);
    }
    
    if (exponent == 0) {
        // Zero or denormal
        return sign;
    }
    
    // Adjust exponent bias: FP32 bias 127 -> FP16 bias 15
    int32_t new_exp = static_cast<int32_t>(exponent) - 127 + 15;
    
    if (new_exp >= 31) {
        // Overflow -> Inf
        return sign | 0x7C00;
    }
    
    if (new_exp <= 0) {
        // Underflow -> zero
        return sign;
    }
    
    // Truncate mantissa: 23 bits -> 10 bits
    uint16_t new_mantissa = mantissa >> 13;
    
    return sign | (new_exp << 10) | new_mantissa;
}

// FP16 -> FP32 conversion
inline float half_to_float(uint16_t half) {
    // Extract fields
    uint16_t sign = half & 0x8000;
    uint16_t exponent = (half >> 10) & 0x1F;
    uint16_t mantissa = half & 0x3FF;
    
    // Handle special cases
    if (exponent == 0x1F) {
        // Inf or NaN
        if (mantissa == 0) {
            // Inf
            uint32_t bits = (sign << 16) | 0x7F800000;
            float out;
            std::memcpy(&out, &bits, sizeof(float));
            return out;
        } else {
            // NaN
            uint32_t bits = (sign << 16) | 0x7FC00000;
            float out;
            std::memcpy(&out, &bits, sizeof(float));
            return out;
        }
    }
    
    if (exponent == 0 && mantissa == 0) {
        // Zero
        uint32_t bits = sign << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(float));
        return out;
    }
    
    // Adjust exponent bias: FP16 bias 15 -> FP32 bias 127
    uint32_t new_exp = exponent + 127 - 15;
    
    // Expand mantissa: 10 bits -> 23 bits
    uint32_t new_mantissa = static_cast<uint32_t>(mantissa) << 13;
    
    uint32_t bits = (static_cast<uint32_t>(sign) << 16) | (new_exp << 23) | new_mantissa;
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

/**
 * @brief INT8 quantization tools
 */
struct QuantizationParams {
    float scale;
    float zero_point;
    float min_val;
    float max_val;
};

// Calculate quantization parameters
inline QuantizationParams calculate_quantization_params(const float* data, size_t count) {
    QuantizationParams params;
    
    // Find min and max values
    params.min_val = *std::min_element(data, data + count);
    params.max_val = *std::max_element(data, data + count);
    
    // Calculate scale and zero_point
    params.scale = (params.max_val - params.min_val) / 255.0f;
    params.zero_point = -params.min_val / params.scale;
    
    return params;
}

// FP32 -> INT8 quantization
inline void quantize_fp32_to_int8(const float* src, int8_t* dst, size_t count, 
                                  const QuantizationParams& params) {
    for (size_t i = 0; i < count; ++i) {
        float quantized = src[i] / params.scale + params.zero_point;
        int32_t clamped = std::max(0, std::min(255, static_cast<int32_t>(std::round(quantized))));
        dst[i] = static_cast<int8_t>(clamped - 128);
    }
}

// INT8 -> FP32 dequantization
inline void dequantize_int8_to_fp32(const int8_t* src, float* dst, size_t count,
                                    const QuantizationParams& params) {
    for (size_t i = 0; i < count; ++i) {
        float dequantized = (static_cast<float>(src[i]) + 128.0f - params.zero_point) * params.scale;
        dst[i] = dequantized;
    }
}

/**
 * @brief DType utility class
 */
class DTypeUtils {
public:
    static size_t size_of(DType dtype) {
        switch (dtype) {
            case kFloat32: return 4;
            case kFloat16: return 2;
            case kInt32: return 4;
            case kInt8: return 1;
            default: return 4;
        }
    }
};

/**
 * @brief Memory alignment tools
 */
inline size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

inline bool is_aligned(void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Memory statistics tools
 */
struct MemorySnapshot {
    size_t active_memory;
    size_t standby_memory;
    size_t compressed_memory;
    size_t disk_storage;
    
    size_t total() const {
        return active_memory + standby_memory + compressed_memory + disk_storage;
    }
    
    float fragmentation_ratio;
    float compression_ratio;
    
    void print() const {
        std::cout << "Memory Snapshot:" << std::endl;
        std::cout << "  Active: " << active_memory / 1024 / 1024 << "MB" << std::endl;
        std::cout << "  Standby: " << standby_memory / 1024 / 1024 << "MB" << std::endl;
        std::cout << "  Compressed: " << compressed_memory / 1024 / 1024 << "MB" << std::endl;
        std::cout << "  Disk: " << disk_storage / 1024 / 1024 << "MB" << std::endl;
        std::cout << "  Total: " << total() / 1024 / 1024 << "MB" << std::endl;
        std::cout << "  Fragmentation: " << fragmentation_ratio * 100 << "%" << std::endl;
        std::cout << "  Compression: " << compression_ratio << "x" << std::endl;
    }
};

/**
 * @brief Performance timing tools
 */
class ScopedTimer {
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    bool print_on_destruct_;

public:
    explicit ScopedTimer(const std::string& name, bool print = true)
        : name_(name), print_on_destruct_(print) {
        start_ = std::chrono::steady_clock::now();
    }
    
    ~ScopedTimer() {
        if (print_on_destruct_) {
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
            std::cout << "[Timer] " << name_ << ": " << duration << "ms" << std::endl;
        }
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
    }
};

/**
 * @brief Automatic memory management RAII wrapper
 */
template<typename T>
class AlignedMemoryGuard {
private:
    T* ptr_;
    size_t size_;
    size_t alignment_;

public:
    AlignedMemoryGuard(size_t count, size_t alignment = 64)
        : size_(count * sizeof(T)), alignment_(alignment) {
        ptr_ = static_cast<T*>(std::aligned_alloc(alignment, size_));
        if (!ptr_) {
            throw std::bad_alloc();
        }
    }
    
    ~AlignedMemoryGuard() {
        if (ptr_) {
            std::free(ptr_);
        }
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // Prohibit copying
    AlignedMemoryGuard(const AlignedMemoryGuard&) = delete;
    AlignedMemoryGuard& operator=(const AlignedMemoryGuard&) = delete;
    
    // Allow moving
    AlignedMemoryGuard(AlignedMemoryGuard&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), alignment_(other.alignment_) {
        other.ptr_ = nullptr;
    }
};

} // namespace memory
} // namespace ops

