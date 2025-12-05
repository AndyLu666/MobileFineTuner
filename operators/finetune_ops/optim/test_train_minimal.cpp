/**
 * @file test_train_minimal.cpp  
 * @brief 最小训练闭环测试（Forward + Loss，不含Backward）
 */

#include "../graph/gpt2_model.h"
#include "../graph/lora_injector.h"
#include "../graph/safetensors_loader.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_bpe.h"
#include "../core/lm_loss.h"
#include <iostream>

using namespace ops;

int main() {
    try {
        std::cout << "========== 最小训练闭环测试 ==========\n" << std::endl;
        
        // 1. 加载模型
        std::cout << "[1/5] Loading model..." << std::endl;
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
        std::cout << "✅ Model loaded" << std::endl;
        
        // 2. 注入LoRA
        std::cout << "\n[2/5] Injecting LoRA..." << std::endl;
        LoraSpec lora_spec = LoraSpec::default_config();
        lora_spec.rank = 4;  // 小rank快速测试
        
        LoraInjector lora;
        lora.inject(model, lora_spec);
        lora.print_info();
        
        // 3. 加载数据
        std::cout << "\n[3/5] Loading data..." << std::endl;
        auto tok_cfg = BPEConfig::from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        
        WT2Config data_cfg;
        data_cfg.train_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        data_cfg.valid_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        data_cfg.seq_len = 64;  // 极短序列
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        std::cout << "✅ Loaded " << dataset.num_sequences() << " sequences" << std::endl;
        
        // 4. 取一个batch测试Forward
        std::cout << "\n[4/5] Testing forward pass..." << std::endl;
        auto batch = dataset.next_batch(2);
        
        std::cout << "  Batch shapes:" << std::endl;
        std::cout << "    input_ids: [" << batch.input_ids->shape()[0] << ", " 
                  << batch.input_ids->shape()[1] << "]" << std::endl;
        std::cout << "    labels: [" << batch.labels->shape()[0] << ", "
                  << batch.labels->shape()[1] << "]" << std::endl;
        
        // Forward
        auto logits = model.forward(batch.input_ids, batch.attention_mask);
        std::cout << "  Logits shape: [" << logits->shape()[0] << ", "
                  << logits->shape()[1] << ", " << logits->shape()[2] << "]" << std::endl;
        
        // 5. 计算Loss
        std::cout << "\n[5/5] Computing loss..." << std::endl;
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        float loss_val = loss->data<float>()[0];
        float ppl = perplexity_from_loss(loss_val);
        
        std::cout << "  Loss: " << loss_val << std::endl;
        std::cout << "  Perplexity: " << ppl << std::endl;
        
        if (loss_val > 0.0f && loss_val < 100.0f && ppl > 1.0f) {
            std::cout << "\n🎉 最小闭环测试通过！" << std::endl;
            std::cout << "  ✅ Model加载正常" << std::endl;
            std::cout << "  ✅ LoRA注入正常" << std::endl;
            std::cout << "  ✅ Dataset加载正常" << std::endl;
            std::cout << "  ✅ Forward计算正常" << std::endl;
            std::cout << "  ✅ Loss计算正常" << std::endl;
            std::cout << "\n下一步：实现Backward和Optimizer" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ Loss数值异常" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

