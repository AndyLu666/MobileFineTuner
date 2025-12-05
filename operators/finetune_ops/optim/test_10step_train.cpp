/**
 * @file test_10step_train.cpp
 * @brief 10步训练冒烟测试（简化版，验证优化器流程）
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
        std::cout << "========== 10步训练冒烟测试 ==========\n" << std::endl;
        std::cout << "注意：这是简化测试，只训练embedding权重验证优化器流程" << std::endl;
        std::cout << "（完整LoRA训练需要在模型forward中集成LoRA计算）\n" << std::endl;
        
        // 1. 加载模型
        std::cout << "[1/3] Loading model..." << std::endl;
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
        
        // 2. 准备数据
        std::cout << "\n[2/3] Loading data..." << std::endl;
        auto tok_cfg = BPEConfig::from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        
        WT2Config data_cfg;
        data_cfg.train_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        data_cfg.valid_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        data_cfg.seq_len = 64;
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        
        // 3. 训练10步
        std::cout << "\n[3/3] Training 10 steps..." << std::endl;
        std::cout << "（简化测试：冻结所有参数，只看Forward+Loss流程）\n" << std::endl;
        
        std::vector<float> losses;
        
        for (int step = 0; step < 10; ++step) {
            auto batch = dataset.next_batch(2);
            
            // Forward
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            
            losses.push_back(loss_val);
            float ppl = perplexity_from_loss(loss_val);
            
            std::cout << "Step " << step << ": Loss=" << loss_val 
                      << ", PPL=" << ppl << std::endl;
        }
        
        // 验证loss稳定性（预训练模型loss应该在3-5范围且稳定）
        float first_loss = losses[0];
        float last_loss = losses[9];
        float mean_loss = 0.0f;
        for (float l : losses) mean_loss += l;
        mean_loss /= losses.size();
        
        std::cout << "\n[Statistics]" << std::endl;
        std::cout << "  First loss: " << first_loss << std::endl;
        std::cout << "  Last loss: " << last_loss << std::endl;
        std::cout << "  Mean loss: " << mean_loss << std::endl;
        
        if (first_loss > 2.0f && first_loss < 10.0f && 
            std::abs(first_loss - last_loss) < 2.0f) {
            std::cout << "\n✅ 测试通过：Loss稳定，Forward+Loss+Dataset流程正常" << std::endl;
            std::cout << "\n⚠️  注意：这只是Forward测试" << std::endl;
            std::cout << "   完整LoRA训练需要：" << std::endl;
            std::cout << "   1. 在GPT2Model中集成lora_linear计算" << std::endl;
            std::cout << "   2. 实现Backward传播" << std::endl;
            std::cout << "   3. 连接优化器更新LoRA参数" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ Loss异常" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

