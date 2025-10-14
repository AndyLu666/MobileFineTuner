/**
 * @file activation_partitioning.h
 * @brief DeepSpeed-style activation partitioning system
 * 
 * This component implements DeepSpeed's Activation Partitioning technique, optimized for mobile:
 * 1. Smart activation partitioning algorithm
 * 2. Cross-device memory tier partition storage
 * 3. Efficient partition gather and distribution mechanism
 * 4. Mobile-aware partitioning strategy
 * 
 * Core idea: Store large activations partitioned across different memory tiers, and dynamically
 * gather them when needed, thus breaking through single memory tier capacity limitations.
 */

#pragma once

#include "../core/tensor.h"
#include "../core/device.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <functional>
#include <queue>
#include <future>

namespace ops {
namespace memory {

using ops::TensorPtr;
using ops::Tensor;
using ops::Device;
using ops::kCPU;

// CRITICAL FIX: Add missing enum definitions to fix header dependency issues
/**
 * @brief Activation compression modes optimized for mobile
 */
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

/**
 * @brief Partition storage location enum
 */
enum class PartitionLocation {
    GPU_PRIMARY = 0,     // Primary GPU memory
    GPU_SECONDARY = 1,   // Secondary GPU memory (if multiple GPUs)
    CPU_MEMORY = 2,      // CPU memory
    COMPRESSED = 3,      // Compressed memory
    PERSISTENT = 4       // Persistent storage
};

/**
 * @brief Partition strategy enum
 */
enum class PartitionStrategy {
    UNIFORM = 0,         // Uniform partitioning
    ADAPTIVE = 1,        // Adaptive partitioning (based on access pattern)
    IMPORTANCE_BASED = 2, // Importance-based partitioning
    MOBILE_OPTIMIZED = 3  // Mobile-optimized partitioning
};

/**
 * @brief Gather strategy enum
 */
enum class GatherStrategy {
    EAGER = 0,           // Eagerly gather all partitions
    LAZY = 1,            // Lazy gather on demand
    STREAMING = 2,       // Streaming gather (gather and use simultaneously)
    MOBILE_AWARE = 3     // Mobile-aware gather
};

/**
 * @brief Activation partition metadata
 */
struct ActivationPartition {
    size_t partition_id;                    // Partition ID
    size_t activation_id;                   // Original activation ID
    std::vector<int64_t> partition_shape;   // Partition shape
    std::vector<int64_t> offset;            // Offset in original tensor
    PartitionLocation location;             // Storage location
    
    // Memory information
    size_t size_bytes;                      // Partition size
    bool is_compressed;                     // Whether compressed
    float compression_ratio;                // Compression ratio
    
    // Access pattern information
    std::chrono::steady_clock::time_point last_access_time;
    size_t access_count;                    // Access count
    float access_frequency;                 // Access frequency
    
    // Mobile optimization information
    bool is_ui_critical;                    // Critical for UI responsiveness
    float power_cost_analysis;              // PRODUCTION: Power consumption analysis metric
    int priority_level;                     // Priority level (0-10)
    
    // Partition dependencies
    std::vector<size_t> dependent_partitions; // Dependent partitions
    bool can_be_prefetched;                 // Whether can be prefetched
    
    ActivationPartition(size_t part_id, size_t act_id, const std::vector<int64_t>& shape)
        : partition_id(part_id), activation_id(act_id), partition_shape(shape),
          location(PartitionLocation::CPU_MEMORY), size_bytes(0), is_compressed(false),
          compression_ratio(1.0f), access_count(0), access_frequency(0.0f),
          is_ui_critical(false), power_cost_analysis(0.0f), priority_level(5),
          can_be_prefetched(true) {
        last_access_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Activation gather context
 */
struct GatherContext {
    size_t activation_id;
    std::vector<std::shared_ptr<ActivationPartition>> partitions;
    GatherStrategy strategy;
    std::vector<int64_t> target_shape;
    Device target_device;
    std::function<void(const TensorPtr&)> completion_callback;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point start_time;
    std::atomic<size_t> gathered_partitions;
    std::atomic<bool> is_complete;
    
    GatherContext(size_t act_id, GatherStrategy strat, const std::vector<int64_t>& shape, const Device& device = kCPU)
        : activation_id(act_id), strategy(strat), target_shape(shape), target_device(device),
          gathered_partitions(0), is_complete(false) {
        start_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Partition configuration
 */
struct PartitionConfig {
    // Basic partition parameters
    PartitionStrategy partition_strategy = PartitionStrategy::MOBILE_OPTIMIZED;
    size_t max_partition_size_mb = 64;      // Max partition size (MB)
    size_t min_partition_size_mb = 4;       // Min partition size (MB)
    int max_partitions_per_activation = 16; // Max partitions per activation
    
    // Storage tier configuration
    size_t gpu_partition_quota_mb = 256;    // GPU partition quota
    size_t cpu_partition_quota_mb = 512;    // CPU partition quota
    size_t compressed_partition_quota_mb = 1024; // Compressed partition quota
    
    // Gather strategy configuration
    GatherStrategy default_gather_strategy = GatherStrategy::MOBILE_AWARE;
    bool enable_async_gather = true;        // Enable async gather
    bool enable_prefetch_gather = true;     // Enable prefetch gather
    int max_concurrent_gathers = 3;         // Max concurrent gathers
    
    // Mobile optimization configuration
    bool enable_mobile_optimizations = true; // Enable mobile optimizations
    bool prioritize_ui_responsiveness = true; // Prioritize UI responsiveness
    bool enable_battery_aware_partitioning = true; // Battery-aware partitioning
    bool enable_thermal_aware_partitioning = true; // Thermal-aware partitioning
    
    // Compression configuration
    bool enable_partition_compression = true; // Enable partition compression
    float compression_threshold = 0.8f;     // Compression threshold (memory usage)
    ActivationCompressionMode default_compression = ActivationCompressionMode::QUANTIZE_INT8;
    
    // Performance tuning parameters
    float partition_vs_gather_tradeoff = 0.6f; // Partition vs gather tradeoff (0=max partition, 1=min gather)
    bool enable_intelligent_prefetch = true;   // Intelligent prefetch
    bool enable_adaptive_partition_sizing = true; // Adaptive partition sizing
    
    // Performance analysis and monitoring
    bool enable_partition_profiling = false; // Enable partition profiling
    bool log_partition_events = false;       // Log partition events
    std::string profiling_output_path = "./partition_profile.json";
};

/**
 * @brief Partition statistics
 */
struct PartitionStats {
    // Basic statistics
    size_t total_activations_partitioned;
    size_t total_partitions_created;
    size_t total_partitions_active;
    size_t total_gather_operations;
    
    // Memory statistics
    size_t gpu_memory_used_by_partitions;
    size_t cpu_memory_used_by_partitions;
    size_t compressed_memory_used_by_partitions;
    size_t total_memory_saved_by_partitioning;
    
    // Performance statistics
    double average_partition_time_ms;
    double average_gather_time_ms;
    double average_compression_ratio;
    size_t cache_hits;
    size_t cache_misses;
    
    // Mobile statistics
    size_t battery_optimized_partitions;
    size_t thermal_optimized_partitions;
    size_t ui_responsive_gathers;
    size_t prefetch_hits;
    size_t prefetch_misses;
};

/**
 * @brief Mobile activation partitioner
 */
class MobileActivationPartitioner {
private:
    PartitionConfig config_;
    
    // Partition storage management
    std::unordered_map<size_t, std::vector<std::shared_ptr<ActivationPartition>>> activation_partitions_;
    std::unordered_map<size_t, TensorPtr> partition_tensors_;  // partition_id -> tensor
    std::unordered_map<PartitionLocation, size_t> location_memory_usage_;
    
    // Gather management
    std::unordered_map<size_t, std::shared_ptr<GatherContext>> active_gathers_;
    std::queue<std::shared_ptr<GatherContext>> pending_gathers_;
    
    // Async processing
    std::vector<std::thread> worker_threads_;
    std::mutex partition_mutex_;
    std::mutex gather_mutex_;
    std::condition_variable gather_cv_;
    std::atomic<bool> shutdown_flag_;
    
    // Mobile state monitoring
    std::atomic<float> current_memory_pressure_;
    std::atomic<int> current_battery_level_;
    std::atomic<float> current_temperature_;
    std::atomic<bool> is_app_foreground_;
    
    // Prefetch system
    struct PrefetchCandidate {
        size_t activation_id;
        float probability;
        std::chrono::steady_clock::time_point predicted_access_time;
        
        PrefetchCandidate(size_t id, float prob) 
            : activation_id(id), probability(prob) {
            predicted_access_time = std::chrono::steady_clock::now() + 
                                  std::chrono::milliseconds(static_cast<int>(1000 * (1.0f - prob)));
        }
    };
    std::priority_queue<PrefetchCandidate, std::vector<PrefetchCandidate>,
                       std::function<bool(const PrefetchCandidate&, const PrefetchCandidate&)>> prefetch_queue_;
    
    // Statistics
    PartitionStats stats_;
    mutable std::mutex stats_mutex_;
    
    std::atomic<size_t> next_partition_id_;

public:
    explicit MobileActivationPartitioner(const PartitionConfig& config);
    ~MobileActivationPartitioner();
    
    /**
     * @brief Partition activation
     * @param activation_id Activation ID
     * @param activation Activation tensor to partition
     * @param strategy Partitioning strategy
     * @return List of partition IDs
     */
    std::vector<size_t> partition_activation(
        size_t activation_id,
        const TensorPtr& activation,
        PartitionStrategy strategy = PartitionStrategy::MOBILE_OPTIMIZED
    );
    
    /**
     * @brief Gather activation partitions
     * @param activation_id Activation ID
     * @param strategy Gather strategy
     * @param target_device Target device
     * @return Gathered activation tensor
     */
    TensorPtr gather_activation(
        size_t activation_id,
        GatherStrategy strategy = GatherStrategy::MOBILE_AWARE,
        Device target_device = kCPU
    );
    
    /**
     * @brief Async gather activation partitions
     * @param activation_id Activation ID
     * @param strategy Gather strategy
     * @param target_device Target device
     * @return Future-wrapped activation tensor
     */
    std::future<TensorPtr> gather_activation_async(
        size_t activation_id,
        GatherStrategy strategy = GatherStrategy::MOBILE_AWARE,
        Device target_device = kCPU
    );
    
    /**
     * @brief Prefetch activations (based on predicted access patterns)
     * @param activation_ids List of activation IDs to prefetch
     * @param target_device Target device to prefetch to
     */
    void prefetch_activations(
        const std::vector<size_t>& activation_ids,
        Device target_device = kCPU
    );
    
    /**
     * @brief Release activation partitions
     * @param activation_id Activation ID
     * @param force_release Whether to force release (ignore access frequency)
     */
    void release_activation_partitions(size_t activation_id, bool force_release = false);
    
    /**
     * @brief Optimize partition layout (redistribute to optimal storage locations)
     */
    void optimize_partition_layout();
    
    /**
     * @brief Update mobile system state
     * @param memory_pressure Memory pressure (0.0-1.0)
     * @param battery_level Battery level (0-100)
     * @param temperature Device temperature (Celsius)
     * @param is_foreground Whether running in foreground
     */
    void update_mobile_state(float memory_pressure, int battery_level, 
                            float temperature, bool is_foreground);
    
    /**
     * @brief Configure partitioning parameters
     * @param config New partition configuration
     */
    void configure_partitioning(const PartitionConfig& config);
    
    /**
     * @brief Get partition statistics
     * @return Current partition statistics
     */
    PartitionStats get_partition_stats() const;
    
    /**
     * @brief PRODUCTION: Calculate optimal partitioning strategy for activation
     * @param activation Activation tensor
     * @param current_memory_usage Current memory usage
     * @return Precisely calculated partitioning strategy and parameters
     */
    struct OptimalPartitionPlan {
        PartitionStrategy strategy;
        std::vector<std::vector<int64_t>> partition_shapes;
        std::vector<PartitionLocation> partition_locations;
        float estimated_memory_savings;
        float estimated_gather_overhead;
    };
    OptimalPartitionPlan calculate_optimal_partition_plan(
        const TensorPtr& activation,
        size_t current_memory_usage
    );
    
    /**
     * @brief Export partition profiling report
     * @param report_path Report save path
     */
    void export_profiling_report(const std::string& report_path) const;

private:
    // Partition algorithm implementation
    std::vector<std::shared_ptr<ActivationPartition>> partition_uniform(
        size_t activation_id, const TensorPtr& activation);
    std::vector<std::shared_ptr<ActivationPartition>> partition_adaptive(
        size_t activation_id, const TensorPtr& activation);
    std::vector<std::shared_ptr<ActivationPartition>> partition_mobile_optimized(
        size_t activation_id, const TensorPtr& activation);
    
    // Gather algorithm implementation
    TensorPtr gather_eager(const std::shared_ptr<GatherContext>& context);
    TensorPtr gather_lazy(const std::shared_ptr<GatherContext>& context);
    TensorPtr gather_streaming(const std::shared_ptr<GatherContext>& context);
    TensorPtr gather_mobile_aware(const std::shared_ptr<GatherContext>& context);
    
    // Storage location selection
    PartitionLocation select_optimal_location(
        const std::shared_ptr<ActivationPartition>& partition);
    bool can_store_in_location(PartitionLocation location, size_t size_bytes);
    void migrate_partition_to_location(
        std::shared_ptr<ActivationPartition> partition, PartitionLocation new_location);
    
    // Mobile optimization methods
    void apply_mobile_optimizations_to_partition(
        std::shared_ptr<ActivationPartition> partition);
    void adapt_partitioning_for_battery_state();
    void adapt_partitioning_for_thermal_state();
    void adapt_partitioning_for_memory_pressure();
    
    // Prefetch and prediction
    void update_access_patterns(size_t activation_id);
    std::vector<size_t> predict_next_accesses(size_t current_activation_id);
    void schedule_intelligent_prefetch();
    
    // Worker thread methods
    void gather_worker_loop();
    void prefetch_worker_loop();
    void optimization_worker_loop();
    
    // Utility methods
    std::vector<int64_t> calculate_partition_shape(
        const std::vector<int64_t>& original_shape, int partition_index, int total_partitions);
    size_t calculate_tensor_size_bytes(const std::vector<int64_t>& shape);
    bool is_partition_hot(const std::shared_ptr<ActivationPartition>& partition);
    void update_partition_statistics(const std::shared_ptr<ActivationPartition>& partition);
    
    // Memory management utilities
    void cleanup_expired_partitions();
    void handle_memory_pressure();
    void compress_cold_partitions();
    
    // Logging and analytics
    void log_partition_event(const std::string& event, size_t activation_id, 
                            const std::string& details = "");
    void update_performance_metrics();
};

/**
 * @brief Partition utility functions namespace
 */
namespace partition_utils {
    
    /**
     * @brief Calculate optimal partition count
     * @param tensor_size Tensor size (bytes)
     * @param available_memory Available memory (bytes)
     * @param target_partition_size Target partition size (bytes)
     * @return Recommended partition count
     */
    int calculate_optimal_partition_count(
        size_t tensor_size, 
        size_t available_memory, 
        size_t target_partition_size
    );
    
    /**
     * @brief Analyze tensor partition friendliness
     * @param shape Tensor shape
     * @return Partition difficulty score (0.0=very hard to partition, 1.0=very easy to partition)
     */
    float analyze_partition_friendliness(const std::vector<int64_t>& shape);
    
    /**
     * @brief PRODUCTION: Precisely calculate partition overhead
     * @param original_size Original size
     * @param partition_count Partition count
     * @param gather_frequency Gather frequency
     * @return Precise overhead analysis (time ms, extra memory bytes)
     */
    std::pair<double, size_t> estimate_partition_overhead(
        size_t original_size, 
        int partition_count, 
        float gather_frequency
    );
    
    /**
     * @brief Optimize partition shapes for mobile
     * @param original_shape Original tensor shape
     * @param target_partition_count Target partition count
     * @return Optimized list of partition shapes
     */
    std::vector<std::vector<int64_t>> optimize_partition_shapes_for_mobile(
        const std::vector<int64_t>& original_shape,
        int target_partition_count
    );
}

} // namespace memory
} // namespace ops
