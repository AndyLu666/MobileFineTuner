/**
 * @file trainer.cpp
 * @brief LoRA微调Trainer实现
 */

#include "trainer.h"
#include "../data/wikitext2_dataset.h"
#include "../core/lm_loss.h"
#include "../core/logger.h"
#include "../core/performance_monitor.h"
#include "../core/memory_manager.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ops {

LoRATrainer::LoRATrainer(GPT2Model& model,
                         LoraInjector& lora,
                         WikiText2Dataset& train_data,
                         WikiText2Dataset& eval_data,
                         const TrainerConfig& config)
    : model_(model), lora_(lora), train_data_(train_data), eval_data_(eval_data),
      config_(config), global_step_(0) {
    
    // 初始化优化器（只优化LoRA参数）
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.beta1 = config_.adam_beta1;
    adam_cfg.beta2 = config_.adam_beta2;
    adam_cfg.epsilon = config_.adam_eps;
    adam_cfg.weight_decay = 0.0f;  // LoRA参数通常不加weight decay
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    
    optimizer_ = std::make_unique<Adam>(adam_cfg);
    
    std::cout << "[Trainer] Initialized with:" << std::endl;
    std::cout << "  LR: " << config_.learning_rate << std::endl;
    std::cout << "  Epochs: " << config_.num_epochs << std::endl;
    std::cout << "  Grad accum steps: " << config_.gradient_accumulation_steps << std::endl;
    std::cout << "  Max grad norm: " << config_.max_grad_norm << std::endl;
}

float LoRATrainer::get_lr(int step) {
    // 计算总步数
    int64_t total_steps = train_data_.num_sequences() / config_.gradient_accumulation_steps * config_.num_epochs;
    int warmup_steps = static_cast<int>(total_steps * config_.warmup_ratio);
    
    if (step < warmup_steps) {
        // Linear warmup
        return config_.learning_rate * (static_cast<float>(step) / warmup_steps);
    } else {
        // Linear decay或cosine
        if (config_.lr_scheduler == "linear") {
            float progress = static_cast<float>(step - warmup_steps) / (total_steps - warmup_steps);
            return config_.learning_rate * (1.0f - progress);
        } else if (config_.lr_scheduler == "cosine") {
            float progress = static_cast<float>(step - warmup_steps) / (total_steps - warmup_steps);
            return config_.learning_rate * 0.5f * (1.0f + std::cos(3.14159265f * progress));
        } else {
            return config_.learning_rate;
        }
    }
}

void LoRATrainer::clip_gradients() {
    // 收集所有LoRA参数的梯度
    auto lora_params = lora_.get_trainable_params();
    
    // 计算全局梯度范数
    float total_norm = 0.0f;
    for (const auto& param : lora_params) {
        if (!param->grad()) continue;
        const float* grad_data = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            total_norm += grad_data[i] * grad_data[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // 如果超过阈值，缩放所有梯度
    if (total_norm > config_.max_grad_norm) {
        float scale = config_.max_grad_norm / (total_norm + 1e-6f);
        for (const auto& param : lora_params) {
            if (!param->grad()) continue;
            float* grad_data = param->grad()->data<float>();
            for (int64_t i = 0; i < param->grad()->numel(); ++i) {
                grad_data[i] *= scale;
            }
        }
    }
}

float LoRATrainer::train_step(const Batch& batch) {
    // 1. Forward
    auto logits = model_.forward(batch.input_ids, batch.attention_mask);
    
    // 2. Loss
    auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
    float loss_val = loss->data<float>()[0];
    
    // 3. Backward
    loss->backward();
    
    // 4. Clip gradients
    clip_gradients();
    
    // 5. Optimizer step
    auto lora_params = lora_.get_trainable_params();
    std::vector<TensorPtr> grads;
    for (const auto& param : lora_params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(lora_params, grads);
    
    // 6. Zero grad
    for (const auto& param : lora_params) {
        param->zero_grad();
    }
    
    global_step_++;
    
    // 7. 每步强制内存清理：避免 RSS 长时间累积
    MemoryManager::instance().force_cleanup();
    
    return loss_val;
}

float LoRATrainer::evaluate() {
    std::cout << "\n[Eval] Running evaluation..." << std::endl;
    
    eval_data_.reset_cursor();
    float total_loss = 0.0f;
    int num_batches = 0;
    
    // 评估时禁用dropout（TODO: 添加模式切换）
    while (true) {
        auto batch = eval_data_.next_batch(config_.gradient_accumulation_steps, false);
        if (!batch.input_ids) break;  // 无更多数据
        
        // Forward only
        auto logits = model_.forward(batch.input_ids, batch.attention_mask);
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        
        total_loss += loss->data<float>()[0];
        num_batches++;
        
        // 🔧 评测时每个batch后清理内存（避免累积）
        if (num_batches % 5 == 0) {
            MemoryManager::instance().cleanup_dead_references();
            MemoryManager::instance().clear_unused_memory();
        }
    }
    
    // 评测结束后强制清理
    MemoryManager::instance().force_cleanup();
    
    float mean_loss = (num_batches > 0) ? (total_loss / num_batches) : 0.0f;
    float ppl = perplexity_from_loss(mean_loss);
    
    std::cout << "  Eval Loss: " << mean_loss << std::endl;
    std::cout << "  Perplexity: " << ppl << std::endl;
    
    return mean_loss;
}

void LoRATrainer::train() {
    std::cout << "\n========== Training Start ==========" << std::endl;
    
    // 打印初始内存状态
    auto initial_memory = get_current_memory_snapshot();
    std::cout << "\n📊 Initial Memory State:" << std::endl;
    initial_memory.print();
    
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << config_.num_epochs << " ---" << std::endl;
        
        train_data_.reset_cursor();
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        while (true) {
            auto batch = train_data_.next_batch(config_.gradient_accumulation_steps, false);
            if (!batch.input_ids) break;  // epoch结束
            
            float loss = train_step(batch);
            epoch_loss += loss;
            num_batches++;
            
            // 日志
            if (global_step_ % config_.logging_steps == 0) {
                float current_lr = get_lr(global_step_);
                float ppl = perplexity_from_loss(loss);
                std::cout << "[Step " << global_step_ << "] "
                          << "Loss: " << loss << ", "
                          << "PPL: " << ppl << ", "
                          << "LR: " << current_lr << std::endl;
            }
            
            // 评估
            if (global_step_ % config_.eval_steps == 0) {
                evaluate();  // 执行评估
                train_data_.reset_cursor();  // 恢复训练数据迭代器
            }
            
            // 更新学习率
            float new_lr = get_lr(global_step_);
            optimizer_->set_learning_rate(new_lr);
        }
        
        float mean_loss = (num_batches > 0) ? (epoch_loss / num_batches) : 0.0f;
        std::cout << "Epoch " << (epoch + 1) << " finished, Mean Loss: " << mean_loss << std::endl;
        
        // Epoch 结束时强制清理内存
        MemoryManager::instance().force_cleanup();
        
        // 打印当前内存状态
        auto current_memory = get_current_memory_snapshot();
        std::cout << "\n📊 Memory after Epoch " << (epoch + 1) << ":" << std::endl;
        current_memory.print();
    }
    
    std::cout << "\n========== Training Finished ==========" << std::endl;
    
    // 打印最终内存统计与优化建议
    auto final_memory = get_current_memory_snapshot();
    final_memory.print();
    print_memory_optimization_tips(final_memory);
}

void LoRATrainer::save_lora(const std::string& path) {
    std::cout << "[Trainer] Saving LoRA weights to: " << path << std::endl;
    // TODO: 实现safetensors保存
    // LoraSaver::save(lora_, path);
}

}  // namespace ops

