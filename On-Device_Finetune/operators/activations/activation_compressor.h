/**
 * @file activation_compressor.h
 * @brief Mobile-optimized activation compression system
 * 
 * This component implements advanced compression techniques specifically designed
 * for mobile activation memory optimization. It supports multiple compression modes
 * with mobile-specific considerations like power efficiency and decompression speed.
 * 
 * Key Features:
 * - Multi-mode compression (quantization, sparsification, lossy compression)
 * - Mobile-aware compression selection
 * - Hardware-accelerated compression when available
 * - Adaptive compression based on system state
 */

#pragma once

#include "../core/tensor.h"
#include <memory>
#include <vector>

namespace ops {
namespace memory {
    enum class ActivationCompressionMode;  // Forward declaration
}
}

#include <unordered_map>
#include <atomic>
#include <mutex>

namespace ops {
namespace memory {

#ifndef ACTIVATION_COMPRESSION_MODE_DEFINED
#define ACTIVATION_COMPRESSION_MODE_DEFINED
enum class ActivationCompressionMode {
    NONE = 0,              // No compression
    QUANTIZE_INT8 = 1,     // 8-bit quantization (3-4x compression)
    QUANTIZE_INT4 = 2,     // 4-bit quantization (6-8x compression)
    SPARSE_50 = 3,         // 50% sparsification
    SPARSE_75 = 4,         // 75% sparsification  
    LOSSY_COMPRESS = 5,    // Lossy compression for non-critical activations
    ADAPTIVE = 6           // Adaptive compression based on system state
};
#endif

using ::ops::TensorPtr;
using ::ops::Tensor;
using ::ops::DType;

/**
 * @brief Compression metadata for activation tensors
 */
struct CompressionMetadata {
    ActivationCompressionMode mode;
    float compression_ratio;
    size_t original_size;
    size_t compressed_size;
    
    // Quantization parameters
    float scale = 1.0f;
    int zero_point = 0;
    DType original_dtype;
    
    // Sparsification parameters
    float sparsity_ratio = 0.0f;
    size_t sparse_elements = 0;
    
    // Compression quality metrics
    float mse_error = 0.0f;          // Mean squared error after compression
    float max_error = 0.0f;          // Maximum absolute error
    bool is_lossy = false;           // Whether compression is lossy
    
    // Mobile optimization metadata
    std::chrono::steady_clock::time_point compression_time;
    std::chrono::steady_clock::time_point last_access_time;
    int access_count = 0;
    bool is_frequently_accessed = false;
    
    CompressionMetadata(ActivationCompressionMode m) 
        : mode(m), compression_ratio(1.0f), original_size(0), compressed_size(0),
          original_dtype(::ops::kFloat32), compression_time(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Compressed activation container
 */
struct CompressedActivation {
    std::vector<uint8_t> compressed_data;    // Compressed data buffer
    std::unique_ptr<CompressionMetadata> metadata;  // Compression metadata
    std::vector<int64_t> original_shape;     // Original tensor shape
    
    CompressedActivation(size_t size, ActivationCompressionMode mode)
        : compressed_data(size), metadata(std::make_unique<CompressionMetadata>(mode)) {}
};

/**
 * @brief Configuration for activation compression
 */
struct CompressionConfig {
    // Quantization settings
    bool enable_int8_quantization = true;    // Enable INT8 quantization
    bool enable_int4_quantization = true;    // Enable INT4 quantization (experimental)
    bool enable_dynamic_quantization = true; // Enable dynamic quantization per-tensor
    float quantization_error_threshold = 0.01f; // Max acceptable quantization error
    
    // Sparsification settings
    bool enable_sparsification = true;       // Enable sparsification
    float default_sparsity_threshold = 0.01f; // Default sparsity threshold
    float aggressive_sparsity_threshold = 0.05f; // Aggressive sparsity for memory pressure
    bool enable_structured_sparsity = false; // Enable structured sparsity (N:M patterns)
    
    // Lossy compression settings
    bool allow_lossy_compression = false;    // Allow lossy compression for non-critical activations
    float lossy_compression_ratio = 0.1f;   // Target compression ratio for lossy mode
    float max_lossy_error = 0.05f;          // Maximum error for lossy compression
    
    // Mobile optimization settings
    bool optimize_for_decompression_speed = true; // Optimize for fast decompression
    bool optimize_for_power_efficiency = true;    // Optimize for power efficiency
    bool enable_hardware_acceleration = true;     // Use hardware acceleration if available
    
    // Adaptive compression settings
    bool enable_adaptive_compression = true; // Enable adaptive compression based on system state
    float memory_pressure_compression_factor = 1.5f; // Increase compression under memory pressure
    float battery_low_compression_factor = 2.0f;     // Increase compression when battery low
    
    // Quality control settings
    float quality_vs_compression_balance = 0.7f; // 0.0=max compression, 1.0=max quality
    bool enable_compression_verification = false; // Verify compression quality
    int max_compression_threads = 2;             // Maximum threads for compression
};

/**
 * @brief Mobile-optimized activation compressor
 */
class ActivationCompressor {
private:
    CompressionConfig config_;
    std::unordered_map<size_t, std::unique_ptr<CompressedActivation>> compressed_cache_;
    std::mutex compression_mutex_;
    
    // Statistics
    std::atomic<size_t> total_compressions_;
    std::atomic<size_t> total_decompressions_;
    std::atomic<size_t> total_bytes_saved_;
    std::atomic<double> total_compression_time_;
    std::atomic<double> total_decompression_time_;
    
    // Mobile state awareness
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<bool> is_thermal_throttling_;

public:
    explicit ActivationCompressor(const CompressionConfig& config);
    ~ActivationCompressor();
    
    /**
     * @brief Compress an activation tensor
     * @param activation The activation tensor to compress
     * @param compression_mode Compression mode to use
     * @param activation_id Unique ID for this activation
     * @return Compressed activation container
     */
    std::unique_ptr<CompressedActivation> compress_activation(
        const TensorPtr& activation,
        ActivationCompressionMode compression_mode,
        size_t activation_id
    );
    
    /**
     * @brief Decompress an activation tensor
     * @param compressed_activation Compressed activation container
     * @return Decompressed activation tensor
     */
    TensorPtr decompress_activation(const CompressedActivation& compressed_activation);
    
    /**
     * @brief Select optimal compression mode based on current system state
     * @param activation The activation tensor to analyze
     * @param layer_name Name of the layer (for layer-specific optimization)
     * @param is_critical Whether this activation is critical for accuracy
     * @return Recommended compression mode
     */
    ActivationCompressionMode select_optimal_compression_mode(
        const TensorPtr& activation,
        const std::string& layer_name = "",
        bool is_critical = true
    );
    
    /**
     * @brief Estimate compression ratio for different modes
     * @param activation The activation tensor to analyze
     * @param mode Compression mode to estimate
     * @return Estimated compression ratio
     */
    float estimate_compression_ratio(const TensorPtr& activation, ActivationCompressionMode mode);
    
    /**
     * @brief Update mobile system state for adaptive compression
     * @param memory_pressure Current memory pressure (0.0-1.0)
     * @param battery_level Current battery level (0-100)
     * @param is_thermal_throttling Whether device is thermally throttling
     */
    void update_system_state(float memory_pressure, int battery_level, bool is_thermal_throttling);
    
    /**
     * @brief Configure compression parameters
     * @param config New compression configuration
     */
    void configure_compression(const CompressionConfig& config);
    
    /**
     * @brief Get compression statistics
     */
    struct CompressionStats {
        size_t total_compressions;
        size_t total_decompressions;
        size_t total_bytes_saved;
        double average_compression_time_ms;
        double average_decompression_time_ms;
        float average_compression_ratio;
        float average_quality_score;
    };
    CompressionStats get_compression_stats() const;
    
    /**
     * @brief Clear compressed cache
     */
    void clear_compressed_cache();

private:
    // Quantization methods
    std::unique_ptr<CompressedActivation> quantize_int8(const TensorPtr& activation, size_t activation_id);
    std::unique_ptr<CompressedActivation> quantize_int4(const TensorPtr& activation, size_t activation_id);
    TensorPtr dequantize_int8(const CompressedActivation& compressed);
    TensorPtr dequantize_int4(const CompressedActivation& compressed);
    
    // Sparsification methods
    std::unique_ptr<CompressedActivation> sparsify_activation(const TensorPtr& activation, 
                                                             float threshold, size_t activation_id);
    TensorPtr desparsify_activation(const CompressedActivation& compressed);
    
    // Lossy compression methods
    std::unique_ptr<CompressedActivation> lossy_compress(const TensorPtr& activation, 
                                                        float target_ratio, size_t activation_id);
    TensorPtr lossy_decompress(const CompressedActivation& compressed);
    
    // Mobile optimization methods
    ActivationCompressionMode adapt_compression_for_mobile_state(ActivationCompressionMode base_mode);
    bool should_use_hardware_acceleration(const TensorPtr& activation);
    void optimize_compression_for_power_efficiency(CompressionConfig& config);
    
    // Utility methods
    float calculate_mse_error(const TensorPtr& original, const TensorPtr& reconstructed);
    float calculate_compression_quality(const TensorPtr& original, const CompressedActivation& compressed);
    void update_compression_statistics(const CompressionMetadata& metadata, double compression_time);
    
    // Hardware-specific optimizations
    void initialize_hardware_accelerators();
    bool compress_with_neon_simd(const TensorPtr& activation, CompressedActivation& compressed);
    bool compress_with_gpu_acceleration(const TensorPtr& activation, CompressedActivation& compressed);
};

/**
 * @brief Quantization utilities specifically optimized for mobile
 */
namespace mobile_quantization {
    
    /**
     * @brief Calculate optimal quantization parameters for mobile
     * @param tensor Input tensor to analyze
     * @param target_bits Target bit width (4 or 8)
     * @return Scale and zero point for quantization
     */
    std::pair<float, int> calculate_quantization_params(const TensorPtr& tensor, int target_bits = 8);
    
    /**
     * @brief Quantize tensor with mobile-optimized implementation
     * @param tensor Input tensor
     * @param scale Quantization scale
     * @param zero_point Quantization zero point
     * @param target_bits Target bit width
     * @return Quantized data buffer
     */
    std::vector<uint8_t> quantize_tensor_mobile(const TensorPtr& tensor, float scale, int zero_point, int target_bits);
    
    /**
     * @brief Dequantize tensor with mobile-optimized implementation
     * @param quantized_data Quantized data buffer
     * @param scale Quantization scale
     * @param zero_point Quantization zero point
     * @param target_bits Source bit width
     * @param shape Target tensor shape
     * @return Dequantized tensor
     */
    TensorPtr dequantize_tensor_mobile(const std::vector<uint8_t>& quantized_data, 
                                     float scale, int zero_point, int target_bits,
                                     const std::vector<int64_t>& shape);
    
    /**
     * @brief Check if tensor is suitable for quantization
     * @param tensor Input tensor to check
     * @return True if tensor is suitable for quantization
     */
    bool is_quantization_suitable(const TensorPtr& tensor);
}

/**
 * @brief Sparsification utilities optimized for mobile
 */
namespace mobile_sparsification {
    
    /**
     * @brief Calculate optimal sparsity threshold for mobile
     * @param tensor Input tensor to analyze
     * @param target_sparsity Target sparsity ratio (0.0-1.0)
     * @return Optimal threshold for sparsification
     */
    float calculate_sparsity_threshold(const TensorPtr& tensor, float target_sparsity);
    
    /**
     * @brief Apply sparsification with mobile-optimized storage
     * @param tensor Input tensor
     * @param threshold Sparsity threshold
     * @return Sparse representation (indices, values, shape)
     */
    struct SparseRepresentation {
        std::vector<uint32_t> indices;
        std::vector<float> values;
        std::vector<int64_t> shape;
        float sparsity_ratio;
    };
    SparseRepresentation sparsify_tensor_mobile(const TensorPtr& tensor, float threshold);
    
    /**
     * @brief Reconstruct tensor from sparse representation
     * @param sparse Sparse representation
     * @return Reconstructed dense tensor
     */
    TensorPtr desparsify_tensor_mobile(const SparseRepresentation& sparse);
    
    /**
     * @brief Check if tensor benefits from sparsification
     * @param tensor Input tensor to check
     * @param min_sparsity_ratio Minimum sparsity ratio to be beneficial
     * @return True if sparsification would be beneficial
     */
    bool is_sparsification_beneficial(const TensorPtr& tensor, float min_sparsity_ratio = 0.3f);
}

} // namespace memory
} // namespace ops
