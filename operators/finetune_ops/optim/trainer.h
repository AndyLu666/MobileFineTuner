/**
 * @file trainer.h
 * @brief LoRA微调Trainer（最小闭环）
 */

#pragma once

#include "../core/tensor.h"
#include "../graph/gpt2_model.h"
#include "../graph/lora_injector.h"
#include "adam.h"
#include <string>
#include <vector>
#include <memory>

namespace ops {
    // 前向声明
    class WikiText2Dataset;
    struct Batch;
}

namespace ops {

struct TrainerConfig {
    // 优化器
    float learning_rate = 2e-4f;
    float weight_decay = 0.01f;      // AdamW decoupled
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;
    
    // 训练
    int num_epochs = 3;
    int gradient_accumulation_steps = 1;
    float max_grad_norm = 1.0f;      // clip_grad_norm
    
    // 学习率调度
    std::string lr_scheduler = "linear";  // "linear" or "cosine"
    float warmup_ratio = 0.03f;
    
    // 日志
    int logging_steps = 10;
    int eval_steps = 100;
    std::string output_dir = "./lora_checkpoints";
};

class LoRATrainer {
public:
    LoRATrainer(GPT2Model& model, 
                LoraInjector& lora,
                WikiText2Dataset& train_data,
                WikiText2Dataset& eval_data,
                const TrainerConfig& config);
    
    /**
     * @brief 运行完整训练
     */
    void train();
    
    /**
     * @brief 单步训练（用于调试）
     */
    float train_step(const Batch& batch);
    
    /**
     * @brief 评估
     */
    float evaluate();
    
    /**
     * @brief 保存LoRA权重
     */
    void save_lora(const std::string& path);
    
private:
    GPT2Model& model_;
    LoraInjector& lora_;
    WikiText2Dataset& train_data_;
    WikiText2Dataset& eval_data_;
    TrainerConfig config_;
    
    std::unique_ptr<Adam> optimizer_;
    int global_step_;
    
    float get_lr(int step);  // 学习率调度
    void clip_gradients();
};

}  // namespace ops

