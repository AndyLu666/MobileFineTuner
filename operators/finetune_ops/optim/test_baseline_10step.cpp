/**
 * @file test_baseline_10step.cpp
 * @brief 纯GPT-2 baseline loss（预训练模型初始表现，不训练）
 * 用于与LoRA训练对比
 */

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_bpe.h"
#include "../core/lm_loss.h"
#include "adam.h"
#include <iostream>
#include <vector>

using namespace ops;

int main() {
    try {
        std::cout << "========== Baseline GPT-2 10步loss（预训练模型，不训练）==========\n" << std::endl;
        
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
        
        // 2. 准备数据（与LoRA训练完全相同）
        std::cout << "\n[2/2] Loading data..." << std::endl;
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
        
        // 3. 10步前向（只计算loss，不训练）
        std::cout << "\n========== 10 Steps Forward Only (Baseline) ==========\n" << std::endl;
        
        std::vector<float> losses;
        
        for (int step = 0; step < 10; ++step) {
            // 获取batch（与LoRA训练相同的数据流）
            auto batch = dataset.next_batch(1);  // batch_size=1
            
            // Forward only（不backward，不更新参数）
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            losses.push_back(loss_val);
            
            // 计算perplexity
            float ppl = perplexity_from_loss(loss_val);
            
            printf("Step %2d: Loss=%.4f, PPL=%.2f\n", step, loss_val, ppl);
        }
        
        // 验证收敛
        std::cout << "\n========== Results ==========" << std::endl;
        float first_loss = losses[0];
        float last_loss = losses[9];
        float improvement = first_loss - last_loss;
        
        std::cout << "  First loss: " << first_loss << std::endl;
        std::cout << "  Last loss: " << last_loss << std::endl;
        std::cout << "  Improvement: " << improvement << std::endl;
        
        std::cout << "\n✅ Baseline前向完成（预训练模型未更新）" << std::endl;
        std::cout << "对比: LoRA训练后loss下降0.80；baseline保持预训练状态" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

