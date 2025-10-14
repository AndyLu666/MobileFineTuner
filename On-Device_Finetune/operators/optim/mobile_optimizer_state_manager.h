/**
 * @file mobile_optimizer_state_manager.h
 * @brief Mobile optimizer state memory management system - Based on DeepSpeed ZeRO technology adaptation
 * 
 * This file implements a memory optimization system for optimizer states on mobile devices.
 * Key innovations:
 * 1. Temporal-sliced ZeRO: Adapt DeepSpeed's spatial partitioning to single-device temporal partitioning
 * 2. State compression: Use FP16/INT8 to compress momentum and variance states
 * 3. CPU optimization: SIMD optimization and cache-friendly design for CPU
 * 4. Tiered offloading: Multi-level offloading strategy from memory to storage
 * 5. Deep integration with parameter/activation management
 * 
 * DeepSpeed ZeRO core technology migration:
 * - ZeRO Stage 1: Optimizer State Partitioning (temporal dimension adaptation)
 * - CPU Offload: Optimizer state CPU offload (already CPU, optimize to storage offload)
 * - Pin Memory: Changed to CPU cache alignment optimization
 * - Contiguous Buffers: Contiguous memory buffers to avoid fragmentation
 * 
 * @author Your Name
 * @date 2025
 */

#pragma once

#include "../core/tensor.h"
#include "../core/device.h"
#include "param_manager_lite.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <chrono>
#include <string>

namespace ops {
namespace optim {

// Forward declarations (actual definition in param_manager_lite.h)
// class MobileParameterManager;  // Already defined in param_manager_lite.h

/**
 * @brief Optimizer state type (corresponding to Adam optimizer)
 */
enum class OptimizerStateType {
    MOMENTUM = 0,          // First moment (Adam's exp_avg, SGD's momentum)
    VARIANCE = 1,          // Second moment (Adam's exp_avg_sq)
    MASTER_WEIGHTS = 2,    // FP32 master weights (mixed precision training)
    STEP_COUNT = 3         // Step count
};

/**
 * @brief Optimizer state storage location
 */
enum class OptimizerStateTier {
    ACTIVE_MEMORY = 0,     // Active memory (currently in use)
    STANDBY_MEMORY = 1,    // Standby memory (loaded but not active)
    COMPRESSED = 2,        // Compressed storage (in memory but compressed)
    DISK_STORAGE = 3       // Disk storage (completely offloaded)
};

/**
 * @brief Optimizer state compression mode
 */
enum class OptimizerStateCompression {
    NONE = 0,              // No compression (FP32)
    FP16 = 1,              // Half precision (2x compression)
    BFLOAT16 = 2,          // BFloat16 (2x compression, better numerical stability)
    INT8_QUANTIZED = 3,    // 8-bit quantization (4x compression)
    INT8_SPARSE = 4,       // 8-bit sparse quantization (4-8x compression)
    ADAPTIVE = 5           // Adaptive compression (select based on importance)
};

/**
 * @brief Optimizer state metadata for a single parameter
 */
struct OptimizerStateMetadata {
    size_t param_id;                              // Parameter ID
    std::string param_name;                       // Parameter name
    size_t param_size;                            // Parameter size (element count)
    
    // State type and location
    OptimizerStateTier momentum_tier;             // Momentum state location
    OptimizerStateTier variance_tier;             // Variance state location
    OptimizerStateCompression compression_mode;    // Compression mode
    
    // Memory usage
    size_t momentum_size_bytes;                   // Momentum memory usage
    size_t variance_size_bytes;                   // Variance memory usage
    size_t original_size_bytes;                   // Original size (uncompressed)
    float compression_ratio;                      // Actual compression ratio
    
    // Access pattern
    std::chrono::steady_clock::time_point last_access_time;
    std::chrono::steady_clock::time_point creation_time;
    size_t access_count;                          // Access count
    int priority;                                 // Priority (0-10)
    
    // State flags
    bool is_loaded;                               // Whether loaded
    bool is_dirty;                                // Whether modified
    bool requires_grad;                           // Whether requires gradient
    bool is_trainable;                            // Whether trainable
    
    // Mobile-specific
    bool is_cpu_optimized;                        // Whether CPU optimized
    bool use_simd_acceleration;                   // Whether using SIMD acceleration
    bool is_cache_aligned;                        // Whether cache aligned
    
    // Storage paths (for disk offload)
    std::string momentum_storage_path;
    std::string variance_storage_path;
    
    OptimizerStateMetadata(size_t id, const std::string& name, size_t size)
        : param_id(id), param_name(name), param_size(size),
          momentum_tier(OptimizerStateTier::ACTIVE_MEMORY),
          variance_tier(OptimizerStateTier::ACTIVE_MEMORY),
          compression_mode(OptimizerStateCompression::NONE),
          momentum_size_bytes(size * sizeof(float)),
          variance_size_bytes(size * sizeof(float)),
          original_size_bytes(size * sizeof(float)),
          compression_ratio(1.0f),
          access_count(0), priority(5),
          is_loaded(false), is_dirty(false),
          requires_grad(true), is_trainable(true),
          is_cpu_optimized(false), use_simd_acceleration(false),
          is_cache_aligned(false) {
        creation_time = std::chrono::steady_clock::now();
        last_access_time = creation_time;
    }
};

/**
 * @brief Optimizer state group - Corresponding to DeepSpeed's Parameter Group concept
 * Organize related parameters' optimizer states together for batch management
 */
struct OptimizerStateGroup {
    std::string group_name;                       // Group name (e.g., "transformer.layer.0")
    std::vector<size_t> param_ids;                // Parameter IDs included in this group
    
    // Group-level state
    bool is_active;                               // Whether active (currently training)
    size_t total_memory_usage;                    // Total memory usage
    OptimizerStateCompression group_compression;   // Group-level compression strategy
    
    // Group-level optimization strategy
    bool enable_group_prefetch;                   // Enable group prefetch
    bool enable_group_offload;                    // Enable group offload
    int access_priority;                          // Access priority
    
    // Statistics
    std::chrono::steady_clock::time_point last_group_access;
    size_t group_access_count;
    
    OptimizerStateGroup(const std::string& name)
        : group_name(name), is_active(false), total_memory_usage(0),
          group_compression(OptimizerStateCompression::NONE),
          enable_group_prefetch(true), enable_group_offload(true),
          access_priority(5), group_access_count(0) {
        last_group_access = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Optimizer state buffer - Corresponding to DeepSpeed's Contiguous Buffer
 * Use contiguous memory to avoid fragmentation and improve CPU cache utilization
 */
class OptimizerStateBuffer {
private:
    void* buffer_ptr_;                            // Buffer pointer
    size_t buffer_size_;                          // Buffer size (bytes)
    size_t used_size_;                            // Used size
    [[maybe_unused]] bool is_cache_aligned_;      // Whether cache aligned
    mutable std::mutex buffer_mutex_;             // Buffer mutex (mutable for const methods)
    
    // Fragmentation management
    std::vector<std::pair<size_t, size_t>> free_chunks_; // (offset, size)
    
public:
    OptimizerStateBuffer(size_t size_bytes, bool cache_align = true);
    ~OptimizerStateBuffer();
    
    // Allocation and deallocation
    void* allocate(size_t size_bytes);
    void deallocate(void* ptr, size_t size_bytes);
    
    // Defragmentation
    void defragment();
    float get_fragmentation_ratio() const;
    
    // Statistics
    size_t get_used_size() const { return used_size_; }
    size_t get_free_size() const { return buffer_size_ - used_size_; }
    size_t get_total_size() const { return buffer_size_; }
};

/**
 * @brief Optimizer state compressor
 * Implements various compression algorithms, optimized for CPU
 */
class OptimizerStateCompressor {
private:
    [[maybe_unused]] OptimizerStateCompression default_compression_;
    
public:
    explicit OptimizerStateCompressor(OptimizerStateCompression mode = OptimizerStateCompression::FP16);
    
    /**
     * @brief Compress optimizer state
     * @param input Input tensor (FP32)
     * @param mode Compression mode
     * @return Compressed tensor and compression ratio
     */
    std::pair<TensorPtr, float> compress(const TensorPtr& input, OptimizerStateCompression mode);
    
    /**
     * @brief Decompress optimizer state
     * @param compressed Compressed tensor
     * @param mode Compression mode
     * @return Decompressed tensor (FP32)
     */
    TensorPtr decompress(const TensorPtr& compressed, OptimizerStateCompression mode);
    
    /**
     * @brief Adaptive compression - select compression mode based on state importance
     * @param input Input tensor
     * @param importance Importance score (0.0-1.0)
     * @return Compressed tensor, used compression mode, and compression ratio
     */
    std::tuple<TensorPtr, OptimizerStateCompression, float> 
    adaptive_compress(const TensorPtr& input, float importance);
    
private:
    // Various compression algorithm implementations
    TensorPtr compress_fp16(const TensorPtr& input);
    TensorPtr decompress_fp16(const TensorPtr& compressed);
    
    TensorPtr compress_int8_quantized(const TensorPtr& input);
    TensorPtr decompress_int8_quantized(const TensorPtr& compressed);
    
    // CPU-optimized SIMD implementation
    void compress_fp32_to_fp16_simd(const float* src, uint16_t* dst, size_t count);
    void decompress_fp16_to_fp32_simd(const uint16_t* src, float* dst, size_t count);
};

/**
 * @brief Optimizer state I/O manager
 * Responsible for disk offloading and loading of optimizer states
 */
class OptimizerStateIOManager {
private:
    std::string storage_path_;
    [[maybe_unused]] bool enable_compression_;
    std::atomic<size_t> total_io_operations_;
    std::atomic<size_t> total_bytes_written_;
    std::atomic<size_t> total_bytes_read_;
    
public:
    explicit OptimizerStateIOManager(const std::string& path, bool compress = true);
    
    /**
     * @brief Save optimizer state to disk
     * @param state_id State ID
     * @param state_type State type (MOMENTUM/VARIANCE)
     * @param data State tensor
     * @return Save path
     */
    std::string save_state_to_disk(size_t state_id, OptimizerStateType state_type, const TensorPtr& data);
    
    /**
     * @brief Load optimizer state from disk
     * @param path Storage path
     * @return State tensor
     */
    TensorPtr load_state_from_disk(const std::string& path);
    
    /**
     * @brief Delete state file on disk
     * @param path Storage path
     */
    void delete_state_file(const std::string& path);
    
    /**
     * @brief Get I/O statistics
     */
    struct IOStats {
        size_t total_operations;
        size_t total_bytes_written;
        size_t total_bytes_read;
        double average_write_speed_mbps;
        double average_read_speed_mbps;
    };
    IOStats get_io_stats() const;
};

/**
 * @brief Configuration structure
 */
struct MobileOptimizerStateConfig {
    // Basic configuration
    size_t max_active_memory_mb = 256;            // Max active memory (256MB)
    size_t max_standby_memory_mb = 512;           // Max standby memory (512MB)
    std::string storage_path = "./optimizer_states"; // Storage path
    
    // Compression configuration
    bool enable_compression = true;               // Enable compression
    OptimizerStateCompression default_compression = OptimizerStateCompression::FP16;
    bool enable_adaptive_compression = true;      // Adaptive compression
    float compression_threshold = 0.7f;           // Compression threshold (70% memory usage)
    
    // CPU optimization configuration
    bool enable_cpu_simd = true;                  // Enable SIMD optimization
    bool enable_cache_alignment = true;           // Enable cache alignment
    size_t cache_line_size = 64;                  // Cache line size
    bool enable_prefetch = true;                  // Enable prefetch
    
    // Offload configuration
    bool enable_disk_offload = true;              // Enable disk offload
    float offload_threshold = 0.8f;               // Offload threshold (80% memory)
    bool enable_async_io = true;                  // Async I/O
    
    // Memory management configuration
    bool use_contiguous_buffers = true;           // Use contiguous buffers
    size_t buffer_size_mb = 128;                  // Buffer size
    bool enable_defragmentation = true;           // Enable defragmentation
    float defrag_threshold = 0.3f;                // Defragmentation threshold (30% fragmentation)
    
    // Mobile-specific configuration
    bool optimize_for_mobile_cpu = true;          // Mobile CPU optimization
    bool respect_thermal_limits = true;           // Respect thermal limits
    bool respect_battery_limits = true;           // Respect battery limits
    float cpu_utilization_target = 0.7f;          // CPU utilization target (70%)
    
    // Group management configuration
    bool enable_group_management = true;          // Enable group management
    bool enable_group_prefetch = true;            // Group prefetch
    bool enable_group_offload = true;             // Group offload
    
    // Advanced optimization configuration
    bool enable_gradient_accumulation = true;     // Gradient accumulation
    int gradient_accumulation_steps = 1;          // Accumulation steps
    bool enable_mixed_precision = true;           // Mixed precision
    bool enable_loss_scaling = false;             // Loss scaling
};

/**
 * @brief Statistics information
 */
struct OptimizerStateStats {
    // Memory statistics
    size_t total_states;                          // Total state count
    size_t active_states;                         // Active state count
    size_t compressed_states;                     // Compressed state count
    size_t offloaded_states;                      // Offloaded state count
    
    size_t active_memory_used;                    // Active memory usage
    size_t standby_memory_used;                   // Standby memory usage
    size_t compressed_memory_used;                // Compressed memory usage
    size_t disk_storage_used;                     // Disk usage
    
    // Compression statistics
    size_t total_compressions;                    // Total compression count
    size_t total_decompressions;                  // Total decompression count
    float average_compression_ratio;              // Average compression ratio
    size_t memory_saved_by_compression;           // Memory saved by compression
    
    // I/O statistics
    size_t total_loads;                           // Total load count
    size_t total_offloads;                        // Total offload count
    double average_load_time_ms;                  // Average load time
    double average_offload_time_ms;               // Average offload time
    
    // Performance statistics
    size_t cache_hits;                            // Cache hits
    size_t cache_misses;                          // Cache misses
    float cache_hit_ratio;                        // Cache hit ratio
    size_t defragmentation_count;                 // Defragmentation count
    
    // Mobile statistics
    size_t thermal_throttle_events;               // Thermal throttle events
    size_t battery_optimization_events;           // Battery optimization events
    float cpu_utilization;                        // CPU utilization
};

/**
 * @brief Main class: Mobile Optimizer state manager
 * 
 * Core responsibilities:
 * 1. Manage optimizer states for all parameters (momentum, variance, etc.)
 * 2. Implement temporally-sliced ZeRO optimization (layer-by-layer load/offload states)
 * 3. Compress and decompress optimizer states
 * 4. CPU-optimized memory management
 * 5. Integration with MobileParameterManager
 */
class MobileOptimizerStateManager {
private:
    MobileOptimizerStateConfig config_;
    
    // State storage
    std::unordered_map<size_t, std::unique_ptr<OptimizerStateMetadata>> state_metadata_;
    std::unordered_map<size_t, TensorPtr> momentum_states_;     // param_id -> momentum tensor
    std::unordered_map<size_t, TensorPtr> variance_states_;     // param_id -> variance tensor
    std::unordered_map<size_t, TensorPtr> master_weights_;      // param_id -> FP32 master weight
    
    // Group management
    std::vector<std::unique_ptr<OptimizerStateGroup>> state_groups_;
    std::unordered_map<size_t, size_t> param_to_group_map_;     // param_id -> group_id
    std::unordered_map<std::string, size_t> group_name_to_id_;  // group_name -> group_id
    
    // Memory management
    std::unique_ptr<OptimizerStateBuffer> active_buffer_;
    std::unique_ptr<OptimizerStateBuffer> standby_buffer_;
    std::unique_ptr<OptimizerStateCompressor> compressor_;
    std::unique_ptr<OptimizerStateIOManager> io_manager_;
    
    // Parameter manager reference (deep integration)
    MobileParameterManager* param_manager_;
    
    // Memory usage tracking
    std::atomic<size_t> active_memory_used_;
    std::atomic<size_t> standby_memory_used_;
    std::atomic<size_t> compressed_memory_used_;
    
    // Statistics
    OptimizerStateStats stats_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex manager_mutex_;
    
    // Mobile monitoring
    std::atomic<float> current_cpu_utilization_;
    std::atomic<bool> is_thermal_throttling_;
    std::atomic<bool> is_low_battery_;

public:
    /**
     * @brief Constructor
     * @param config Configuration
     * @param param_manager Parameter manager (optional, for deep integration)
     */
    explicit MobileOptimizerStateManager(
        const MobileOptimizerStateConfig& config,
        MobileParameterManager* param_manager = nullptr
    );
    
    ~MobileOptimizerStateManager();
    
    // Core API: Optimizer state registration and access
    
    /**
     * @brief Register optimizer state for parameter
     * @param param_id Parameter ID
     * @param param_name Parameter name
     * @param param_size Parameter size (element count)
     * @param group_name Group name (optional)
     * @param requires_grad Whether requires gradient
     */
    void register_parameter_state(
        size_t param_id,
        const std::string& param_name,
        size_t param_size,
        const std::string& group_name = "default",
        bool requires_grad = true
    );
    
    /**
     * @brief Get parameter's momentum state
     * @param param_id Parameter ID
     * @return Momentum tensor (auto-loaded)
     */
    TensorPtr get_momentum_state(size_t param_id);
    
    /**
     * @brief Get parameter's variance state
     * @param param_id Parameter ID
     * @return Variance tensor (auto-loaded)
     */
    TensorPtr get_variance_state(size_t param_id);
    
    /**
     * @brief Update momentum state
     * @param param_id Parameter ID
     * @param new_momentum New momentum value
     */
    void update_momentum_state(size_t param_id, const TensorPtr& new_momentum);
    
    /**
     * @brief Update variance state
     * @param param_id Parameter ID
     * @param new_variance New variance value
     */
    void update_variance_state(size_t param_id, const TensorPtr& new_variance);
    
    /**
     * @brief Release parameter's optimizer state (mark as offloadable)
     * @param param_id Parameter ID
     */
    void release_parameter_state(size_t param_id);
    
    // Group management API: Implement batch state management
    
    /**
     * @brief Load entire group's optimizer states
     * @param group_name Group name
     */
    void load_group_states(const std::string& group_name);
    
    /**
     * @brief Offload entire group's optimizer states
     * @param group_name Group name
     * @param force Whether to force offload (even if marked dirty)
     */
    void offload_group_states(const std::string& group_name, bool force = false);
    
    /**
     * @brief Set group's compression mode
     * @param group_name Group name
     * @param compression Compression mode
     */
    void set_group_compression(const std::string& group_name, OptimizerStateCompression compression);
    
    // Memory optimization API
    
    /**
     * @brief Compress optimizer state to save memory
     * @param param_id Parameter ID
     * @param compression Compression mode
     * @return Memory saved (bytes)
     */
    size_t compress_parameter_state(size_t param_id, OptimizerStateCompression compression);
    
    /**
     * @brief Offload optimizer state to disk
     * @param param_id Parameter ID
     * @return Memory freed (bytes)
     */
    size_t offload_parameter_state(size_t param_id);
    
    /**
     * @brief Automatically optimize memory usage
     * Automatically compress or offload states based on current memory pressure
     */
    void optimize_memory_usage();
    
    /**
     * @brief Execute defragmentation
     * @return Memory reclaimed (bytes)
     */
    size_t defragment_memory();
    
    /**
     * @brief Emergency memory cleanup
     * Free as much memory as possible in critical memory situations
     * @return Memory freed (bytes)
     */
    size_t emergency_memory_cleanup();
    
    // Mobile-specific API
    
    /**
     * @brief Update mobile system state
     * @param cpu_util CPU utilization (0.0-1.0)
     * @param is_thermal_throttle Whether thermal throttling
     * @param is_low_battery Whether low battery
     */
    void update_mobile_state(float cpu_util, bool is_thermal_throttle, bool is_low_battery);
    
    /**
     * @brief Enable/disable CPU SIMD optimization
     * @param enable Whether to enable
     */
    void enable_cpu_simd_optimization(bool enable);
    
    /**
     * @brief Set CPU utilization target
     * @param target Target utilization (0.0-1.0)
     */
    void set_cpu_utilization_target(float target);
    
    // Statistics and monitoring API
    
    /**
     * @brief Get statistics
     */
    OptimizerStateStats get_statistics() const;
    
    /**
     * @brief Get state metadata for specified parameter
     */
    const OptimizerStateMetadata* get_state_metadata(size_t param_id) const;
    
    /**
     * @brief Export detailed report
     * @param report_path Report save path
     */
    void export_detailed_report(const std::string& report_path) const;
    
    // Checkpoint API
    
    /**
     * @brief Save all optimizer states to checkpoint
     * @param checkpoint_path Checkpoint path
     */
    void save_checkpoint(const std::string& checkpoint_path);
    
    /**
     * @brief Load optimizer states from checkpoint
     * @param checkpoint_path Checkpoint path
     */
    void load_checkpoint(const std::string& checkpoint_path);
    
    // Configuration API
    
    /**
     * @brief Get current configuration
     */
    const MobileOptimizerStateConfig& get_config() const { return config_; }
    
    /**
     * @brief Set parameter manager (for deep integration)
     */
    void set_parameter_manager(MobileParameterManager* param_manager);

private:
    // Internal implementation methods
    
    // Initialization methods
    void initialize_components();
    void cleanup_components();
    
    // State loading and offloading
    void load_state_internal(size_t param_id, OptimizerStateType state_type);
    void offload_state_internal(size_t param_id, OptimizerStateType state_type);
    
    // Memory management
    void* allocate_from_buffer(size_t size_bytes, bool is_active);
    void deallocate_from_buffer(void* ptr, size_t size_bytes, bool is_active);
    size_t calculate_memory_pressure() const;
    
    // Selection algorithms
    std::vector<size_t> select_states_to_compress(size_t target_memory_reduction);
    std::vector<size_t> select_states_to_offload(size_t target_memory_reduction);
    OptimizerStateCompression select_optimal_compression(size_t param_id);
    
    // Statistics update
    void update_statistics();
    void update_access_pattern(size_t param_id);
    
    // Mobile optimization
    void apply_cpu_optimization();
    void apply_thermal_optimization();
    void apply_battery_optimization();
};

/**
 * @brief Factory function: Create mobile optimizer state manager
 */
std::unique_ptr<MobileOptimizerStateManager> create_mobile_optimizer_state_manager(
    size_t available_memory_mb = 256,
    const std::string& storage_path = "./optimizer_states",
    MobileParameterManager* param_manager = nullptr
);

} // namespace memory
} // namespace ops

