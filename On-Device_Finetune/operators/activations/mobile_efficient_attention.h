/**
 * @file mobile_efficient_attention.h
 * @brief Mobile memory-efficient attention mechanism system
 * 
 * This component implements FlashAttention-like memory-efficient attention computation,
 * specifically optimized for mobile:
 * 1. Blocked attention computation, avoiding storing full attention matrix
 * 2. Online softmax algorithm, reducing intermediate result storage
 * 3. Mobile hardware optimization (ARM NEON, GPU, etc.)
 * 4. Dynamic precision adjustment and compression
 * 5. Battery and thermal-aware computation scheduling
 * 
 * Core innovation: Compared to standard attention's O(N²) memory complexity,
 * achieves O(N) memory complexity while maintaining computational precision and performance.
 */

#pragma once

#include "../core/tensor.h"
#include "../core/ops.h"
#include <memory>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <cmath>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::Tensor;

/**
 * @brief Attention computation strategy enum
 */
enum class AttentionStrategy {
    STANDARD = 0,           // Standard attention (high memory)
    FLASH_ATTENTION = 1,    // FlashAttention style
    MOBILE_OPTIMIZED = 2,   // Mobile-optimized version
    MEMORY_FIRST = 3,       // Memory-first (lowest memory usage)
    SPEED_FIRST = 4         // Speed-first (may use more memory)
};

/**
 * @brief Block computation strategy
 */
enum class BlockStrategy {
    UNIFORM_BLOCKS = 0,     // Uniform blocking
    ADAPTIVE_BLOCKS = 1,    // Adaptive blocking
    IMPORTANCE_BLOCKS = 2,  // Importance-based blocking
    MOBILE_AWARE = 3        // Mobile-aware blocking
};

/**
 * @brief Precision mode enum
 */
enum class AttentionPrecision {
    FP32 = 0,              // 32-bit floating point
    FP16 = 1,              // 16-bit floating point
    BF16 = 2,              // BFloat16
    MIXED = 3,             // Mixed precision
    DYNAMIC = 4            // Dynamic precision adjustment
};

/**
 * @brief Attention head grouping strategy
 */
enum class HeadGroupStrategy {
    NO_GROUPING = 0,       // No grouping, each head computed independently
    UNIFORM_GROUPING = 1,  // Uniform grouping
    ADAPTIVE_GROUPING = 2, // Adaptive grouping (based on similarity)
    MOBILE_GROUPING = 3    // Mobile-optimized grouping
};

/**
 * @brief Mobile attention configuration
 */
struct MobileAttentionConfig {
    // Basic parameters
    AttentionStrategy strategy = AttentionStrategy::MOBILE_OPTIMIZED;
    BlockStrategy block_strategy = BlockStrategy::MOBILE_AWARE;
    AttentionPrecision precision = AttentionPrecision::MIXED;
    HeadGroupStrategy head_grouping = HeadGroupStrategy::MOBILE_GROUPING;
    
    // Blocking parameters
    size_t block_size = 64;                    // Default block size
    size_t min_block_size = 16;                // Minimum block size
    size_t max_block_size = 256;               // Maximum block size
    bool enable_adaptive_block_sizing = true;  // Adaptive block sizing
    
    // Memory management parameters
    size_t max_attention_memory_mb = 128;      // Max attention memory usage (MB)
    float memory_pressure_threshold = 0.8f;    // Memory pressure threshold
    bool enable_attention_caching = true;      // Enable attention caching
    bool enable_kv_caching = true;             // Enable KV caching
    
    // Mobile optimization parameters
    bool enable_mobile_optimizations = true;   // Enable mobile optimizations
    bool enable_neon_acceleration = true;      // Enable ARM NEON acceleration
    bool enable_gpu_acceleration = true;       // Enable GPU acceleration
    bool optimize_for_power_efficiency = true; // Optimize power efficiency
    
    // Precision and quality parameters
    float attention_dropout = 0.0f;           // Attention dropout rate
    float precision_threshold = 1e-4f;        // Precision threshold
    bool enable_numerical_stability = true;   // Enable numerical stability optimization
    float temperature_scaling = 1.0f;         // Temperature scaling parameter
    
    // Dynamic optimization parameters
    bool enable_dynamic_optimization = true;  // Enable dynamic optimization
    float battery_aware_scaling = 0.8f;       // Computation scaling when battery low
    float thermal_aware_scaling = 0.7f;       // Computation scaling when overheating
    
    // Performance analysis parameters
    bool enable_attention_profiling = false;  // Enable attention profiling
    bool log_attention_events = false;        // Log attention events
    std::string profiling_output_path = "./attention_profile.json";
};

/**
 * @brief Attention computation statistics
 */
struct AttentionStats {
    // Basic statistics
    size_t total_attention_calls;
    size_t total_blocks_processed;
    size_t cache_hits;
    size_t cache_misses;
    
    // Memory statistics
    size_t peak_memory_usage_bytes;
    size_t average_memory_usage_bytes;
    size_t memory_saved_by_blocking;
    float average_memory_efficiency;
    
    // Performance statistics
    double total_attention_time_ms;
    double average_attention_time_ms;
    double total_softmax_time_ms;
    double total_matmul_time_ms;
    
    // Mobile statistics
    size_t neon_accelerated_operations;
    size_t gpu_accelerated_operations;
    size_t battery_optimized_calls;
    size_t thermal_optimized_calls;
    
    // Precision statistics
    double average_attention_entropy;
    size_t precision_downgrades;
    size_t precision_upgrades;
};

/**
 * @brief Attention block context
 */
struct AttentionBlock {
    size_t block_id;
    size_t start_row;
    size_t end_row;
    size_t start_col;
    size_t end_col;
    
    // Block data
    TensorPtr q_block;    // Query block
    TensorPtr k_block;    // Key block
    TensorPtr v_block;    // Value block
    TensorPtr scores_block; // Attention scores block
    
    // Online softmax state
    TensorPtr row_max;    // Row max values (for numerical stability)
    TensorPtr row_sum;    // Row sum (for normalization)
    TensorPtr output_block; // Output block
    
    // Computation state
    bool is_computed;
    bool is_cached;
    std::chrono::steady_clock::time_point last_access_time;
    
    AttentionBlock(size_t id, size_t sr, size_t er, size_t sc, size_t ec)
        : block_id(id), start_row(sr), end_row(er), start_col(sc), end_col(ec),
          is_computed(false), is_cached(false) {
        last_access_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief KV cache entry
 */
struct KVCacheEntry {
    TensorPtr cached_k;
    TensorPtr cached_v;
    size_t sequence_length;
    std::vector<int64_t> original_shape;
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_access_time;
    size_t access_count;
    bool is_compressed;
    
    KVCacheEntry(const TensorPtr& k, const TensorPtr& v, size_t seq_len)
        : cached_k(k), cached_v(v), sequence_length(seq_len), access_count(0), is_compressed(false) {
        creation_time = std::chrono::steady_clock::now();
        last_access_time = creation_time;
        if (k) original_shape = k->shape();
    }
};

/**
 * @brief Mobile efficient attention calculator
 */
class MobileEfficientAttention {
private:
    MobileAttentionConfig config_;
    
    // Block management
    std::vector<std::unique_ptr<AttentionBlock>> current_blocks_;
    std::unordered_map<size_t, std::unique_ptr<AttentionBlock>> block_cache_;
    
    // KV cache management
    std::unordered_map<std::string, std::unique_ptr<KVCacheEntry>> kv_cache_;
    size_t max_kv_cache_size_;
    size_t current_kv_cache_size_;
    
    // Mobile state monitoring
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_app_foreground_;
    
    // Thread safety
    mutable std::mutex attention_mutex_;
    mutable std::mutex cache_mutex_;
    mutable std::mutex stats_mutex_;
    
    // Statistics
    AttentionStats stats_;
    
    // Dynamic optimization state
    std::atomic<AttentionPrecision> current_precision_;
    std::atomic<size_t> current_block_size_;
    std::deque<double> recent_computation_times_;

public:
    explicit MobileEfficientAttention(const MobileAttentionConfig& config);
    ~MobileEfficientAttention();
    
    /**
     * @brief Compute memory-efficient multi-head attention
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, seq_len, num_heads, head_dim]
     * @param value Value tensor [batch, seq_len, num_heads, head_dim]
     * @param mask Optional attention mask
     * @param cache_key KV cache key (for inference optimization)
     * @return Attention output tensor
     */
    TensorPtr compute_attention(
        const TensorPtr& query,
        const TensorPtr& key,
        const TensorPtr& value,
        const TensorPtr& mask = nullptr,
        const std::string& cache_key = ""
    );
    
    /**
     * @brief Compute causal attention (for autoregressive models)
     * @param query Query tensor
     * @param key Key tensor  
     * @param value Value tensor
     * @param cache_key KV cache key
     * @return Attention output tensor
     */
    TensorPtr compute_causal_attention(
        const TensorPtr& query,
        const TensorPtr& key,
        const TensorPtr& value,
        const std::string& cache_key = ""
    );
    
    /**
     * @brief Compute cross attention (for encoder-decoder models)
     * @param query Query tensor (from decoder)
     * @param key Key tensor (from encoder)
     * @param value Value tensor (from encoder)
     * @param mask Cross attention mask
     * @return Attention output tensor
     */
    TensorPtr compute_cross_attention(
        const TensorPtr& query,
        const TensorPtr& key,
        const TensorPtr& value,
        const TensorPtr& mask = nullptr
    );
    
    /**
     * @brief Update KV cache (for incremental inference)
     * @param cache_key Cache key
     * @param new_key New Key tensor
     * @param new_value New Value tensor
     */
    void update_kv_cache(
        const std::string& cache_key,
        const TensorPtr& new_key,
        const TensorPtr& new_value
    );
    
    /**
     * @brief Clear KV cache
     * @param cache_key Specific cache key, if empty clears all
     */
    void clear_kv_cache(const std::string& cache_key = "");
    
    /**
     * @brief Update mobile system state
     * @param memory_pressure Memory pressure (0.0-1.0)
     * @param battery_level Battery level (0-100)  
     * @param temperature Device temperature
     * @param is_foreground Whether running in foreground
     */
    void update_mobile_state(float memory_pressure, int battery_level,
                            float temperature, bool is_foreground);
    
    /**
     * @brief Configure attention parameters
     * @param config New attention configuration
     */
    void configure_attention(const MobileAttentionConfig& config);
    
    /**
     * @brief Get attention statistics
     * @return Current statistics
     */
    AttentionStats get_attention_stats() const;
    
    /**
     * @brief Optimize attention computation configuration
     * Dynamically adjust configuration based on current system state and historical performance
     */
    void optimize_attention_configuration();
    
    /**
     * @brief PRODUCTION: Precisely calculate attention computation memory requirement
     * @param query_shape Query tensor shape
     * @param key_shape Key tensor shape
     * @param strategy Computation strategy
     * @return Precisely analyzed memory requirement (bytes)
     */
    size_t estimate_memory_requirement(
        const std::vector<int64_t>& query_shape,
        const std::vector<int64_t>& key_shape,
        AttentionStrategy strategy = AttentionStrategy::MOBILE_OPTIMIZED
    );
    
    /**
     * @brief Export attention profiling report
     * @param report_path Report save path
     */
    void export_profiling_report(const std::string& report_path) const;

private:
    // Core attention algorithms
    TensorPtr compute_flash_attention(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
        const TensorPtr& mask);
    TensorPtr compute_mobile_optimized_attention(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
        const TensorPtr& mask);
    TensorPtr compute_memory_first_attention(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
        const TensorPtr& mask);
    
    // Blocking algorithms
    std::vector<std::unique_ptr<AttentionBlock>> create_attention_blocks(
        const std::vector<int64_t>& query_shape,
        const std::vector<int64_t>& key_shape);
    void compute_block_attention(AttentionBlock& block, const TensorPtr& mask);
    TensorPtr merge_attention_blocks(const std::vector<std::unique_ptr<AttentionBlock>>& blocks);
    
    // Online softmax algorithm (FlashAttention core)
    void online_softmax_forward(
        const TensorPtr& scores, TensorPtr& row_max, TensorPtr& row_sum, TensorPtr& output);
    void online_softmax_update(
        const TensorPtr& new_scores, TensorPtr& row_max, TensorPtr& row_sum, TensorPtr& output);
    
    // Mobile hardware optimization
    void apply_neon_acceleration(TensorPtr& tensor);
    void apply_gpu_acceleration(TensorPtr& tensor);
    bool should_use_neon_for_operation(size_t tensor_size);
    bool should_use_gpu_for_operation(size_t tensor_size);
    
    // Dynamic precision management
    AttentionPrecision determine_optimal_precision(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value);
    TensorPtr convert_precision(const TensorPtr& tensor, AttentionPrecision target_precision);
    void adapt_precision_for_mobile_state();
    
    // KV cache management
    bool should_cache_kv(const std::string& cache_key, size_t sequence_length);
    void evict_old_kv_cache();
    void compress_kv_cache_if_needed();
    std::pair<TensorPtr, TensorPtr> get_cached_kv(const std::string& cache_key);
    
    // Block optimization algorithms
    size_t calculate_optimal_block_size(
        const std::vector<int64_t>& query_shape, 
        const std::vector<int64_t>& key_shape);
    void adapt_block_size_for_mobile_state();
    bool should_use_adaptive_blocking();
    
    // Mobile optimization strategies
    void apply_battery_aware_optimizations();
    void apply_thermal_aware_optimizations();  
    void apply_memory_pressure_optimizations();
    void apply_ui_responsiveness_optimizations();
    
    // Numerical stability and quality control
    TensorPtr apply_temperature_scaling(const TensorPtr& scores);
    TensorPtr apply_attention_dropout(const TensorPtr& attention_weights);
    void ensure_numerical_stability(TensorPtr& tensor);
    
    // Performance monitoring and statistics
    void update_attention_stats(double computation_time, size_t memory_used);
    void record_mobile_optimization(const std::string& optimization_type);
    void analyze_attention_patterns();
    
    // Utility methods
    std::vector<int64_t> calculate_output_shape(
        const std::vector<int64_t>& query_shape,
        const std::vector<int64_t>& value_shape);
    bool is_causal_mask_needed(const std::vector<int64_t>& query_shape);
    TensorPtr create_causal_mask(size_t sequence_length);
    
    // Memory management utilities
    void cleanup_expired_blocks();
    void handle_attention_memory_pressure();
    size_t get_current_attention_memory_usage();
    
    // Event logging and analytics
    void log_attention_event(const std::string& event, const std::string& details = "");
    void validate_attention_inputs(
        const TensorPtr& query, const TensorPtr& key, const TensorPtr& value);
};

/**
 * @brief Attention utility functions namespace
 */
namespace attention_utils {
    
    /**
     * @brief Calculate theoretical memory complexity of attention computation
     * @param sequence_length Sequence length
     * @param hidden_size Hidden layer size
     * @param num_heads Number of attention heads
     * @return Memory requirement (standard attention vs FlashAttention)
     */
    std::pair<size_t, size_t> calculate_memory_complexity(
        size_t sequence_length, size_t hidden_size, size_t num_heads);
    
    /**
     * @brief Analyze sparsity of attention pattern
     * @param attention_weights Attention weight matrix
     * @param sparsity_threshold Sparsity threshold
     * @return Sparsity analysis result
     */
    struct SparsityAnalysis {
        float sparsity_ratio;        // Sparsity ratio
        std::vector<int> sparse_heads; // Sparse attention heads
        bool can_optimize;           // Whether can optimize
    };
    SparsityAnalysis analyze_attention_sparsity(
        const TensorPtr& attention_weights, float sparsity_threshold = 0.1f);
    
    /**
     * @brief Optimize attention head grouping for mobile
     * @param num_heads Total number of attention heads
     * @param available_memory Available memory
     * @param target_groups Target number of groups
     * @return Optimized grouping strategy
     */
    std::vector<std::vector<int>> optimize_head_grouping(
        int num_heads, size_t available_memory, int target_groups = -1);
    
    /**
     * @brief Create mobile-optimized attention mask
     * @param sequence_length Sequence length
     * @param mask_type Mask type ("causal", "padding", "custom")
     * @param custom_mask Custom mask (optional)
     * @return Optimized attention mask
     */
    TensorPtr create_mobile_optimized_mask(
        size_t sequence_length, 
        const std::string& mask_type,
        const TensorPtr& custom_mask = nullptr);
}

} // namespace memory
} // namespace ops
