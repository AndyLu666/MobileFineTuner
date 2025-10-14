/**
 * @file mobile_optimizer_advanced.h
 * @brief Mobile optimizer advanced features - Supplement missing DeepSpeed advanced techniques
 * 
 * Key missing features supplemented:
 * 1. Parameter Groups - Different optimization strategies for different parameter groups
 * 2. Warm Restart - Learning rate restart and training recovery
 * 3. Advanced checkpoint system - Complete state save/load
 * 4. More learning rate schedulers - Polynomial, MultiStep, etc.
 * 5. Numerical stability enhancement - Overflow detection and recovery
 * 6. Mobile power optimization - Power-aware scheduling
 * 7. Async optimizer updates - ZenFlow-style importance awareness
 * 8. More fine-grained memory budget management
 */

#pragma once

#include "mobile_optimizer_extensions.h"
#include <fstream>
#include <thread>
#include <queue>
#include <condition_variable>

namespace ops {
namespace optim {

/**
 * @brief Parameter group configuration - Support different optimization strategies for different parameters
 */
struct ParameterGroupConfig {
    std::string group_name;                    // Group name
    std::vector<size_t> param_ids;             // Parameter ID list
    
    // Group-specific hyperparameters
    float group_lr_multiplier = 1.0f;         // Learning rate multiplier
    float group_weight_decay = -1.0f;         // Weight decay (-1 means use global value)
    float group_beta1 = -1.0f;                // Beta1 (-1 means use global value)
    float group_beta2 = -1.0f;                // Beta2 (-1 means use global value)
    
    // Group-specific optimization strategy
    OptimizerStateCompression group_compression = OptimizerStateCompression::ADAPTIVE;
    bool freeze_group = false;                 // Freeze this group's parameters
    bool use_sparse_updates = false;          // Use sparse updates
    
    // Mobile-specific
    float mobile_priority = 1.0f;             // Mobile priority (0-10)
    bool thermal_sensitive = true;            // Sensitive to thermal management
    bool battery_sensitive = true;            // Sensitive to battery state
    
    ParameterGroupConfig(const std::string& name) 
        : group_name(name) {}
};

/**
 * @brief Training recovery configuration
 */
struct WarmRestartConfig {
    bool enabled = false;                     // Enable warm restart
    int restart_period = 1000;               // Restart period
    float restart_mult = 2.0f;               // Period multiplier
    float min_lr_ratio = 0.1f;               // Minimum learning rate ratio
    
    // Mobile-specific
    bool adaptive_restart = true;             // Adaptive restart
    float performance_threshold = 0.001f;     // Performance threshold
    int patience_steps = 100;                // Patience steps
};

/**
 * @brief Numerical stability configuration
 */
struct NumericalStabilityConfig {
    bool enable_overflow_detection = true;   // Overflow detection
    bool enable_gradient_scaling = true;     // Gradient scaling
    float initial_scale = 65536.0f;          // Initial scale factor
    float scale_growth_factor = 2.0f;        // Scale growth factor
    float scale_backoff_factor = 0.5f;       // Scale backoff factor
    int scale_growth_interval = 2000;       // Scale growth interval
    
    // Numerical range checking
    float max_grad_norm = 10.0f;            // Maximum gradient norm
    float min_loss_scale = 1.0f;            // Minimum loss scale
    float max_loss_scale = 65536.0f;        // Maximum loss scale
};

/**
 * @brief Power optimization configuration
 */
struct PowerOptimizationConfig {
    bool enable_power_aware = true;          // Enable power awareness
    float target_power_consumption = 3.0f;   // Target power consumption (watts)
    float max_power_consumption = 5.0f;      // Maximum power consumption
    
    // Power scheduling strategy
    bool enable_dynamic_voltage_scaling = true;    // Dynamic voltage scaling
    bool enable_frequency_scaling = true;          // Frequency scaling
    bool enable_core_migration = true;             // Core migration
    
    // Battery optimization
    float battery_critical_threshold = 0.15f;      // Battery critical threshold (15%)
    float battery_low_threshold = 0.30f;           // Battery low threshold (30%)
    float power_reduction_factor = 0.7f;           // Power reduction factor
};

/**
 * @brief Async optimizer configuration (ZenFlow style)
 */
struct AsyncOptimizerConfig {
    bool enable_async_updates = false;       // Enable async updates
    float importance_threshold = 0.1f;       // Importance threshold
    int accumulation_window = 4;             // Accumulation window
    int max_async_ops = 2;                   // Maximum async operations
    
    // Importance calculation
    bool use_gradient_magnitude = true;      // Use gradient magnitude
    bool use_parameter_magnitude = true;     // Use parameter magnitude
    bool use_historical_importance = true;   // Use historical importance
};

/**
 * @brief Advanced checkpoint manager
 */
class AdvancedCheckpointManager {
private:
    std::string base_checkpoint_dir_;
    int max_checkpoints_to_keep_;
    std::vector<std::string> saved_checkpoints_;
    
public:
    explicit AdvancedCheckpointManager(const std::string& dir, int max_keep = 5);
    
    /**
     * @brief Save complete training state
     */
    struct TrainingState {
        int global_step;
        float best_loss;
        float current_lr;
        std::unordered_map<std::string, float> group_lrs;
        OptimizerExtensionStats optimizer_stats;
        NumericalStabilityConfig stability_config;
        std::vector<float> loss_history;
        
        // Mobile state
        std::vector<float> power_history;
        std::vector<float> thermal_history;
        int total_thermal_events;
        int total_battery_events;
    };
    
    bool save_training_state(const TrainingState& state, const std::string& checkpoint_name);
    bool load_training_state(TrainingState& state, const std::string& checkpoint_name);
    
    /**
     * @brief Automatic checkpoint management
     */
    void auto_checkpoint(const TrainingState& state, float current_loss);
    void cleanup_old_checkpoints();
    std::string get_best_checkpoint() const;
};

/**
 * @brief Advanced learning rate scheduler (supplement more types)
 */
class AdvancedLRScheduler : public MobileLRScheduler {
private:
    WarmRestartConfig restart_config_;
    std::vector<float> performance_history_;
    int restart_count_ = 0;
    int last_restart_step_ = 0;
    
public:
    AdvancedLRScheduler(const LRSchedulerConfig& config, 
                       const WarmRestartConfig& restart_config,
                       OptimizerExtensionStats* stats = nullptr);
    
    /**
     * @brief Polynomial decay
     */
    float compute_polynomial_decay_lr(int step, float power = 1.0f);
    
    /**
     * @brief Multi-step decay  
     */
    float compute_multistep_decay_lr(int step, const std::vector<int>& milestones, float gamma = 0.1f);
    
    /**
     * @brief Warm restart
     */
    float compute_warm_restart_lr(int step);
    
    /**
     * @brief Adaptive restart check
     */
    bool should_restart(float current_performance);
    
    /**
     * @brief Perform restart
     */
    void perform_restart();
    
private:
    void update_performance_history(float performance);
};

/**
 * @brief Numerical stability manager
 */
class NumericalStabilityManager {
private:
    NumericalStabilityConfig config_;
    float current_loss_scale_ = 1.0f;
    int overflow_count_ = 0;
    int successful_steps_ = 0;
    std::queue<bool> recent_overflows_;
    
public:
    explicit NumericalStabilityManager(const NumericalStabilityConfig& config);
    
    /**
     * @brief Check gradient overflow
     */
    bool check_gradient_overflow(const std::vector<TensorPtr>& gradients);
    
    /**
     * @brief Handle overflow
     */
    void handle_overflow();
    
    /**
     * @brief Scale gradients
     */
    void scale_gradients(std::vector<TensorPtr>& gradients);
    
    /**
     * @brief Unscale gradients
     */
    void unscale_gradients(std::vector<TensorPtr>& gradients);
    
    /**
     * @brief Update loss scale
     */
    void update_loss_scale(bool overflow_detected);
    
    float get_current_loss_scale() const { return current_loss_scale_; }
    int get_overflow_count() const { return overflow_count_; }
};

/**
 * @brief Power optimization manager
 */
class PowerOptimizationManager {
private:
    PowerOptimizationConfig config_;
    std::vector<float> power_history_;
    float current_power_consumption_ = 0.0f;
    std::chrono::steady_clock::time_point last_power_measurement_;
    
    // Mobile system interface
    bool is_plugged_in_ = false;
    float battery_level_ = 1.0f;
    float device_temperature_ = 30.0f;
    
public:
    explicit PowerOptimizationManager(const PowerOptimizationConfig& config);
    
    /**
     * @brief Update power state
     */
    void update_power_state(float battery_level, bool plugged_in, float temperature);
    
    /**
     * @brief Calculate power adjustment factor
     */
    float compute_power_adjustment_factor();
    
    /**
     * @brief Apply power optimizations
     */
    void apply_power_optimizations(float& learning_rate, int& batch_size);
    
    /**
     * @brief Power consumption prediction
     */
    float predict_power_consumption(float lr, int batch_size);
    
private:
    void log_power_event(const std::string& event);
    float get_battery_drain_rate();
};

/**
 * @brief Async optimizer updater (ZenFlow style)
 */
class AsyncOptimizerUpdater {
private:
    AsyncOptimizerConfig config_;
    std::thread worker_thread_;
    std::queue<std::pair<size_t, TensorPtr>> async_gradient_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> should_stop_{false};
    
    // Importance evaluation
    std::unordered_map<size_t, float> parameter_importance_;
    std::unordered_map<size_t, std::vector<float>> importance_history_;
    
public:
    explicit AsyncOptimizerUpdater(const AsyncOptimizerConfig& config);
    ~AsyncOptimizerUpdater();
    
    /**
     * @brief Submit gradient update
     */
    void submit_gradient_update(size_t param_id, const TensorPtr& gradient);
    
    /**
     * @brief Calculate parameter importance
     */
    float compute_parameter_importance(size_t param_id, const TensorPtr& gradient);
    
    /**
     * @brief Whether should sync update
     */
    bool should_sync_update(size_t param_id, float importance);
    
private:
    void worker_loop();
    void process_async_updates();
    void update_importance_history(size_t param_id, float importance);
};

/**
 * @brief Complete advanced mobile optimizer
 */
class CompleteMobileOptimizerAdvanced {
private:
    std::unique_ptr<CompleteMobileTrainingOptimizer> base_optimizer_;
    
    // Advanced feature managers
    std::vector<std::unique_ptr<ParameterGroupConfig>> parameter_groups_;
    std::unique_ptr<AdvancedCheckpointManager> checkpoint_manager_;
    std::unique_ptr<AdvancedLRScheduler> advanced_lr_scheduler_;
    std::unique_ptr<NumericalStabilityManager> stability_manager_;
    std::unique_ptr<PowerOptimizationManager> power_manager_;
    std::unique_ptr<AsyncOptimizerUpdater> async_updater_;
    
    // Advanced configuration
    WarmRestartConfig restart_config_;
    NumericalStabilityConfig stability_config_;
    PowerOptimizationConfig power_config_;
    AsyncOptimizerConfig async_config_;
    
    // Advanced statistics
    std::vector<float> training_loss_history_;
    std::vector<float> validation_loss_history_;
    float best_validation_loss_ = std::numeric_limits<float>::max();
    int steps_without_improvement_ = 0;
    
public:
    CompleteMobileOptimizerAdvanced(
        size_t available_memory_mb,
        const std::string& storage_path,
        const OptimizerHyperParams& opt_params,
        const GradientClippingConfig& clip_config,
        const LRSchedulerConfig& lr_config,
        const WarmRestartConfig& restart_config,
        const NumericalStabilityConfig& stability_config,
        const PowerOptimizationConfig& power_config,
        const AsyncOptimizerConfig& async_config
    );
    
    /**
     * @brief Create parameter group
     */
    void create_parameter_group(const ParameterGroupConfig& group_config);
    
    /**
     * @brief Advanced training step
     */
    bool advanced_training_step(
        const std::unordered_map<size_t, TensorPtr>& param_gradients,
        float training_loss,
        float validation_loss = -1.0f
    );
    
    /**
     * @brief Auto-adjust optimization strategy
     */
    void auto_adjust_optimization_strategy();
    
    /**
     * @brief Get complete advanced statistics
     */
    struct AdvancedTrainingStats {
        CompleteMobileTrainingOptimizer::CompleteTrainingStats base_stats;
        int restart_count;
        int overflow_count;
        float current_loss_scale;
        float average_power_consumption;
        float total_energy_consumed;
        std::vector<float> loss_history;
        float convergence_rate;
        float training_efficiency;
    };
    
    AdvancedTrainingStats get_advanced_stats() const;
    
    /**
     * @brief Advanced checkpoint
     */
    void save_advanced_checkpoint(const std::string& name, float current_loss);
    bool load_advanced_checkpoint(const std::string& name);
    
    /**
     * @brief Training completion analysis
     */
    void generate_training_analysis_report(const std::string& report_path);
    
private:
    void initialize_advanced_components();
    void update_training_metrics(float training_loss, float validation_loss);
    bool should_early_stop();
    void apply_parameter_group_optimizations();
};

/**
 * @brief Factory function - Create complete advanced mobile optimizer
 */
std::unique_ptr<CompleteMobileOptimizerAdvanced> create_advanced_mobile_optimizer(
    size_t available_memory_mb,
    const std::string& storage_path,
    const std::string& checkpoint_dir,
    bool enable_all_advanced_features = true
);

} // namespace memory  
} // namespace ops
