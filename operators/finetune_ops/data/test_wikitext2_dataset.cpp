/**
 * @file test_wikitext2_dataset.cpp
 * @brief WikiText-2 Dataset 冒烟测试
 */

#include "wikitext2_dataset.h"
#include <iostream>

using namespace ops;

int main() {
    try {
        std::cout << "========== WikiText-2 Dataset Test ==========\n" << std::endl;
        
        // 1. 初始化tokenizer
        std::cout << "[1/4] Loading tokenizer..." << std::endl;
        auto tok_cfg = BPEConfig::from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        std::cout << "✅ Tokenizer loaded" << std::endl;
        
        // 2. 配置Dataset
        std::cout << "\n[2/4] Configuring dataset..." << std::endl;
        WT2Config cfg;
        cfg.train_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        cfg.valid_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        cfg.test_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.test.raw";
        cfg.seq_len = 128;     // 测试用短序列
        cfg.stride = -1;       // 非重叠
        cfg.drop_last = true;
        cfg.eos_id = 50256;
        
        // 3. 加载训练集
        std::cout << "\n[3/4] Loading train split..." << std::endl;
        WikiText2Dataset train_ds(cfg, &tokenizer);
        train_ds.load(Split::Train);
        std::cout << "✅ Train sequences: " << train_ds.num_sequences() << std::endl;
        
        // 4. 取一个batch验证
        std::cout << "\n[4/4] Fetching first batch..." << std::endl;
        auto batch = train_ds.next_batch(2);
        
        std::cout << "\nBatch info:" << std::endl;
        std::cout << "  input_ids shape: [" << batch.input_ids->shape()[0] << ", " 
                  << batch.input_ids->shape()[1] << "]" << std::endl;
        std::cout << "  attention_mask shape: [" << batch.attention_mask->shape()[0] << ", " 
                  << batch.attention_mask->shape()[1] << "]" << std::endl;
        std::cout << "  labels shape: [" << batch.labels->shape()[0] << ", " 
                  << batch.labels->shape()[1] << "]" << std::endl;
        
        // 打印第一个样本的前10个tokens
        const int32_t* input_data = batch.input_ids->data<int32_t>();
        const float* mask_data = batch.attention_mask->data<float>();
        const int32_t* label_data = batch.labels->data<int32_t>();
        
        std::cout << "\nSample 0 first 10 tokens:" << std::endl;
        std::cout << "  input_ids:  [";
        for (int i = 0; i < 10; ++i) {
            std::cout << input_data[i];
            if (i < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  attn_mask:  [";
        for (int i = 0; i < 10; ++i) {
            std::cout << static_cast<int>(mask_data[i]);
            if (i < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  labels:     [";
        for (int i = 0; i < 10; ++i) {
            std::cout << label_data[i];
            if (i < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 验证1：右移正确（labels[i] == input_ids[i+1]）
        std::cout << "\n[Verify 1] Label shift:" << std::endl;
        bool shift_ok = true;
        for (int i = 0; i < 10; ++i) {
            if (mask_data[i] > 0.5f) {  // 有效位置
                if (i < cfg.seq_len - 1) {
                    if (label_data[i] != input_data[i + 1]) {
                        shift_ok = false;
                        std::cout << "  ❌ labels[" << i << "]=" << label_data[i] 
                                  << " != input_ids[" << (i+1) << "]=" << input_data[i+1] << std::endl;
                    }
                }
            }
        }
        if (shift_ok) {
            std::cout << "  ✅ Labels正确右移" << std::endl;
        }
        
        // 验证2：PAD位置的label=-100
        std::cout << "\n[Verify 2] PAD labels:" << std::endl;
        bool pad_ok = true;
        int pad_count = 0;
        for (int i = 0; i < cfg.seq_len; ++i) {
            if (mask_data[i] < 0.5f) {  // PAD位置
                pad_count++;
                if (label_data[i] != -100) {
                    pad_ok = false;
                }
            }
        }
        std::cout << "  PAD count: " << pad_count << std::endl;
        if (pad_ok) {
            std::cout << "  ✅ PAD位置的labels=-100" << std::endl;
        } else {
            std::cout << "  ❌ PAD位置的labels不是-100" << std::endl;
        }
        
        // 验证3：decode回环
        std::cout << "\n[Verify 3] Encode-decode round-trip:" << std::endl;
        std::vector<int> first_10(input_data, input_data + 10);
        std::string decoded = tokenizer.decode(first_10, false);
        std::cout << "  Decoded text: " << decoded << std::endl;
        
        std::cout << "\n🎉 WikiText-2 Dataset 冒烟测试通过！" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

