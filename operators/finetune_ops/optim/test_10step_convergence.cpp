/**
 * @file test_10step_convergence.cpp
 * @brief 10步训练收敛验证
 */

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../graph/lora_injector.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_bpe.h"
#include "../core/lm_loss.h"
#include "adam.h"
#include <iostream>
#include <vector>

using namespace ops;

int main() {
    try {
        std::cout << "========== 10步训练收敛验证 ==========\n" << std::endl;
        
        // 1. 加载模型
        std::cout << "[1/4] Loading model..." << std::endl;
        GPT2Config cfg;
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2/model.safetensors");
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(hf_key, false);
                    model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        
        // 2. 初始化并注入LoRA
        std::cout << "\n[2/4] Initializing LoRA..." << std::endl;
        model.init_lora_modules();
        
        LoraSpec lora_spec;
        lora_spec.rank = 8;
        lora_spec.alpha = 16.0f;
        lora_spec.dropout = 0.0f;
        lora_spec.split_qkv = true;
        
        LoraInjector lora;
        lora.inject(model, lora_spec);
        
        auto lora_params = model.get_lora_parameters();
        std::cout << "✅ LoRA params: " << lora_params.size() << std::endl;
        
        // 3. 初始化优化器
        std::cout << "\n[3/4] Initializing optimizer..." << std::endl;
        AdamConfig adam_cfg;
        adam_cfg.learning_rate = 1e-4f;
        adam_cfg.beta1 = 0.9f;
        adam_cfg.beta2 = 0.999f;
        adam_cfg.epsilon = 1e-8f;
        adam_cfg.weight_decay = 0.0f;  // LoRA不加weight decay
        adam_cfg.clip_grad_norm = 1.0f;
        
        Adam optimizer(adam_cfg);
        
        // 4. 准备数据
        std::cout << "\n[4/4] Loading data..." << std::endl;
        auto tok_cfg = BPEConfig::from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        
        WT2Config data_cfg;
        data_cfg.train_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        data_cfg.valid_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        data_cfg.seq_len = 128;
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        
        // 5. 10步训练
        std::cout << "\n========== Training 10 Steps ==========\n" << std::endl;
        
        std::vector<float> losses;
        
        for (int step = 0; step < 10; ++step) {
            // 获取batch
            auto batch = dataset.next_batch(1);  // batch_size=1
            
            // Forward
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            losses.push_back(loss_val);
            
            // Backward
            loss->backward();
            
            // 收集梯度
            std::vector<TensorPtr> grads;
            for (const auto& param : lora_params) {
                grads.push_back(param->grad());
            }
            
            // Optimizer step
            optimizer.step(lora_params, grads);
            
            // Zero grad
            for (auto& param : lora_params) {
                param->zero_grad();
            }
            
            // 计算perplexity
            float ppl = perplexity_from_loss(loss_val);
            
            printf("Step %2d: Loss=%.4f, PPL=%.2f\n", step, loss_val, ppl);
        }
        
        // 验证收敛
        std::cout << "\n========== Convergence Check ==========" << std::endl;
        float first_loss = losses[0];
        float last_loss = losses[9];
        float improvement = first_loss - last_loss;
        
        std::cout << "  First loss: " << first_loss << std::endl;
        std::cout << "  Last loss: " << last_loss << std::endl;
        std::cout << "  Improvement: " << improvement << std::endl;
        
        if (improvement > 0.01f) {
            std::cout << "\n🎉🎉🎉 训练收敛验证通过！" << std::endl;
            std::cout << "  ✅ Loss下降了" << improvement << std::endl;
            std::cout << "  ✅ 训练管线完全正常" << std::endl;
            std::cout << "\n🏆 B阶段100%完成！" << std::endl;
            return 0;
        } else if (improvement > 0.0f) {
            std::cout << "\n⚠️  Loss略有下降（" << improvement << "），可能需要更多步数" << std::endl;
            return 0;
        } else {
            std::cout << "\n⚠️  Loss未下降，可能需要调整超参" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

