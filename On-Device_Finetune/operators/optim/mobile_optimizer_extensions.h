/**
 * @file mobile_optimizer_extensions.h
 * @brief Mobile optimizer extension features - Supplement missing DeepSpeed key technologies
 * 
 * This file implements key features missing from the original optimizer manager:
 * 1. Gradient Clipping - Prevent gradient explosion
 * 2. Learning Rate Scheduling - Training convergence
 * 3. AdamW optimizer - Correct weight decay implementation
 * 4. Support for other optimizers like SGD
 * 5. Numerical stability optimization - bias correction
 * 6. Sparse gradient optimization - Critical for mobile memory
 */

#pragma once

#include "mobile_optimizer_state_manager.h"
#include <cmath>
#include <algorithm>

namespace ops {
namespace optim {

/**
 * @file mobile_optimizer_extensions.h
 * @brief Mobile optimizer extension features - Supplement missing DeepSpeed key technologies
 * 
 * This file implements key features missing from the original optimizer manager:
 * 1. Gradient Clipping - Prevent gradient explosion
 * 2. Learning Rate Scheduling - Training convergence
 * 3. AdamW optimizer - Correct weight decay implementation
 * 4. Support for other optimizers like SGD
 * 5. Numerical stability optimization - bias correction
 * 6. Sparse gradient optimization - Critical for mobile memory
 */

#pragma once

#include "mobile_optimizer_state_manager.h"
#include <cmath>
#include <algorithm>

namespace ops {
namespace optim {

/**
 * @brief Supported optimizer types
 */
enum class MobileOptimizerType {
    ADAM = 0,
    ADAMW = 1,
    SGD = 2,
    SGD_MOMENTUM = 3,
    ADAGRAD = 4,
    RMSPROP = 5
};

/**
 * @brief Learning rate scheduler types  
 */
enum class LRSchedulerType {
    CONSTANT = 0,           // Constant learning rate
    LINEAR_DECAY = 1,       // Linear decay
    COSINE_DECAY = 2,       // Cosine decay
    EXPONENTIAL_DECAY = 3,  // Exponential decay
    STEP_DECAY = 4,         // Step decay
    WARM_UP_COSINE = 5      // Warm-up + cosine decay (commonly used)
};

/**
 * @brief Gradient clipping configuration
 */
struct GradientClippingConfig {
    bool enabled = true;
    float max_grad_norm = 1.0f;        // Maximum gradient norm
    float clip_value = 0.0f;           // Clip by value (0 means not used)
    bool use_global_norm = true;       // Use global norm vs per-parameter
    bool adaptive_clipping = false;    // Adaptive clipping (mobile-specific)
    float adaptive_factor = 0.01f;     // Adaptive factor
};

/**
 * @brief Learning rate scheduler configuration
 */
struct LRSchedulerConfig {
    LRSchedulerType type = LRSchedulerType::WARM_UP_COSINE;
    float base_lr = 1e-4f;             // Base learning rate
    float min_lr = 1e-6f;              // Minimum learning rate
    
    // Warm-up configuration
    int warmup_steps = 1000;           // Warm-up steps
    float warmup_start_lr = 1e-6f;     // Warm-up starting learning rate
    
    // Decay configuration
    int decay_steps = 10000;           // Decay steps
    float decay_rate = 0.95f;          // Decay rate (exponential decay)
    int step_size = 1000;              // Step size (step decay)
    
    // Mobile-specific
    bool thermal_scaling = true;       // Thermal management related scaling
    bool battery_aware = true;         // Battery-aware adjustment
    float mobile_lr_factor = 0.8f;     // Mobile lr scaling factor
};

/**
 * @brief Optimizer hyperparameters
 */
struct OptimizerHyperParams {
    MobileOptimizerType type = MobileOptimizerType::ADAMW;
    
    // Adam/AdamW parameters
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;        // L2 regularization
    bool amsgrad = false;
    bool bias_correction = true;       // Important: bias correction
    bool adamw_mode = true;            // AdamW vs Adam
    
    // SGD parameters
    float momentum_sgd = 0.9f;
    bool nesterov = false;
    
    // Adagrad parameters
    float adagrad_eps = 1e-10f;
    
    // RMSprop parameters
    float rmsprop_alpha = 0.99f;
    
    // Mobile-specific
    bool fp32_optimizer_states = false; // Mobile usually uses FP16
    bool sparse_gradients = false;      // Sparse gradient optimization
    float gradient_sparsity_threshold = 0.01f; // Sparsity threshold
};

/**
 * @brief Optimizer statistics extension
 */
struct OptimizerExtensionStats {
    // Gradient statistics
    size_t gradient_clips_applied = 0;
    float average_grad_norm = 0.0f;
    float max_grad_norm = 0.0f;
    size_t gradient_overflow_count = 0;
    
    // Learning rate statistics
    float current_lr = 0.0f;
    size_t lr_updates = 0;
    
    // Sparse gradient statistics
    float gradient_sparsity_ratio = 0.0f;
    size_t sparse_updates = 0;
    size_t memory_saved_by_sparsity = 0;
    
    // Numerical stability
    size_t nan_gradients = 0;
    size_t inf_gradients = 0;
    
    // Mobile-related
    size_t thermal_lr_reductions = 0;
    size_t battery_optimizations = 0;
    
    // Training statistics
    size_t total_parameters = 0;
    size_t trainable_parameters = 0;
    size_t successful_steps = 0;
    size_t failed_steps = 0;
};

/**
 * @brief Gradient clipper - Mobile optimization
 */
class MobileGradientClipper {
private:
    GradientClippingConfig config_;
    OptimizerExtensionStats* stats_;
    
    // Mobile adaptive clipping
    float adaptive_norm_history_[10];  // Historical gradient norms
    int history_index_ = 0;
    bool history_filled_ = false;

public:
    explicit MobileGradientClipper(const GradientClippingConfig& config,
                                  OptimizerExtensionStats* stats = nullptr);
    
    /**
     * @brief Clip gradients
     * @param gradients Gradient list
     * @return Gradient norm before clipping
     */
    float clip_gradients(std::vector<TensorPtr>& gradients);
    
    /**
     * @brief Compute global gradient norm
     */
    float compute_global_grad_norm(const std::vector<TensorPtr>& gradients);
    
    /**
     * @brief Adaptive clipping (mobile-specific)
     */
    float compute_adaptive_clip_value(float current_norm);

private:
    void update_gradient_history(float grad_norm);
};

/**
 * @brief Learning rate scheduler
 */
class MobileLRScheduler {
protected:
    LRSchedulerConfig config_;  // Protected for derived classes
    OptimizerExtensionStats* stats_;
    int current_step_ = 0;
    float current_lr_;

public:
    explicit MobileLRScheduler(const LRSchedulerConfig& config,
                              OptimizerExtensionStats* stats = nullptr);
    
    /**
     * @brief Get current learning rate
     */
    float get_learning_rate(int step = -1);
    
    /**
     * @brief Update learning rate (called every step)
     */
    float step();
    
    /**
     * @brief Mobile state-aware adjustment
     */
    void adjust_for_mobile_state(bool is_thermal_throttle, bool is_low_battery);

protected:
    float compute_warmup_lr(int step);
    float compute_cosine_decay_lr(int step);
    float compute_linear_decay_lr(int step);
    float compute_exponential_decay_lr(int step);
    float compute_step_decay_lr(int step);
};

/**
 * @brief Mobile optimizer implementation - Support multiple algorithms
 */
class MobileOptimizer {
private:
    OptimizerHyperParams hyperparams_;
    std::unique_ptr<MobileGradientClipper> gradient_clipper_;
    std::unique_ptr<MobileLRScheduler> lr_scheduler_;
    MobileOptimizerStateManager* state_manager_;
    
    OptimizerExtensionStats extension_stats_;
    int global_step_ = 0;
    
    // Sparse gradient support
    bool enable_sparse_gradients_;
    std::vector<bool> sparse_mask_;

public:
    MobileOptimizer(const OptimizerHyperParams& hyperparams,
                   const GradientClippingConfig& clip_config,
                   const LRSchedulerConfig& lr_config,
                   MobileOptimizerStateManager* state_manager);
    
    /**
     * @brief Execute optimization step
     * @param param_gradients Mapping of parameter IDs to gradients
     * @return Whether update was successful
     */
    bool step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    
    /**
     * @brief Zero gradients
     */
    void zero_grad();
    
    /**
     * @brief Get current learning rate
     */
    float get_current_lr() const { return lr_scheduler_->get_learning_rate(); }
    
    /**
     * @brief Get extension statistics
     */
    const OptimizerExtensionStats& get_extension_stats() const { return extension_stats_; }

private:
    // Various optimizer implementations
    bool adam_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool adamw_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool sgd_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool sgd_momentum_step(const std::unordered_map<size_t, TensorPtr>& param_gradients);
    
    // Adam/AdamW core implementation
    void adam_update_single_param(size_t param_id, const TensorPtr& param, 
                                 const TensorPtr& gradient, float lr, bool adamw_mode);
    
    // Numerical stability check
    bool check_gradient_validity(const TensorPtr& gradient);
    void handle_gradient_overflow();
    
    // Sparse gradient optimization
    void apply_gradient_sparsification(std::unordered_map<size_t, TensorPtr>& param_gradients);
    bool is_gradient_sparse(const TensorPtr& gradient);
    TensorPtr sparsify_gradient(const TensorPtr& gradient);
};

/**
 * @brief Mobile optimizer factory
 */
class MobileOptimizerFactory {
public:
    /**
     * @brief Create default optimizer configuration suitable for mobile
     */
    static OptimizerHyperParams create_mobile_adamw_config(float lr = 1e-4f) {
        OptimizerHyperParams config;
        config.type = MobileOptimizerType::ADAMW;
        config.lr = lr;
        config.weight_decay = 0.01f;
        config.adamw_mode = true;
        config.bias_correction = true;
        config.fp32_optimizer_states = false; // Mobile uses FP16
        return config;
    }
    
    /**
     * @brief Create gradient clipping configuration suitable for mobile
     */
    static GradientClippingConfig create_mobile_clipping_config() {
        GradientClippingConfig config;
        config.enabled = true;
        config.max_grad_norm = 1.0f;
        config.adaptive_clipping = true; // Mobile-specific
        config.adaptive_factor = 0.01f;
        return config;
    }
    
    /**
     * @brief Create learning rate scheduler configuration suitable for mobile
     */
    static LRSchedulerConfig create_mobile_lr_config(int total_steps) {
        LRSchedulerConfig config;
        config.type = LRSchedulerType::WARM_UP_COSINE;
        config.base_lr = 1e-4f;
        config.min_lr = 1e-6f;
        config.warmup_steps = total_steps / 20; // 5% warm-up
        config.decay_steps = total_steps;
        config.thermal_scaling = true;
        config.battery_aware = true;
        config.mobile_lr_factor = 0.8f;
        return config;
    }
};

/**
 * @brief Complete mobile training optimizer - Integrate all features
 */
class CompleteMobileTrainingOptimizer {
private:
    std::unique_ptr<MobileOptimizerStateManager> state_manager_;
    std::unique_ptr<MobileOptimizer> optimizer_;
    
    // Additional mobile optimizations
    [[maybe_unused]] bool enable_mixed_precision_;
    bool enable_gradient_accumulation_;
    int accumulation_steps_;
    int current_accumulation_step_;
    
    // Statistics and monitoring
    OptimizerExtensionStats total_stats_;

public:
    CompleteMobileTrainingOptimizer(
        size_t available_memory_mb,
        const std::string& storage_path,
        const OptimizerHyperParams& opt_params,
        const GradientClippingConfig& clip_config,
        const LRSchedulerConfig& lr_config
    );
    
    /**
     * @brief Complete training step
     * @param param_gradients Parameter gradients
     * @param accumulate Whether to accumulate gradients
     * @return Whether parameter update was performed
     */
    bool training_step(const std::unordered_map<size_t, TensorPtr>& param_gradients,
                      bool accumulate = false);
    
    /**
     * @brief Register training parameter
     */
    void register_training_parameter(size_t param_id, const std::string& param_name, 
                                   size_t param_size, const std::string& group_name);
    
    /**
     * @brief Update mobile system state
     */
    void update_mobile_system_state(float cpu_util, bool thermal, bool low_battery);
    
    /**
     * @brief Get complete statistics
     */
    struct CompleteTrainingStats {
        OptimizerStateStats state_stats;
        OptimizerExtensionStats extension_stats;
        size_t total_parameters;
        size_t trainable_parameters;
        float average_update_time_ms;
        size_t successful_steps;
        size_t failed_steps;
    };
    
    CompleteTrainingStats get_complete_stats() const;
    
    /**
     * @brief Save/load checkpoint
     */
    void save_training_checkpoint(const std::string& path);
    void load_training_checkpoint(const std::string& path);
};

} // namespace memory
} // namespace ops
