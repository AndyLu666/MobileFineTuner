/**
 * @file test_lora_roundtrip.cpp
 * @brief LoRA save/load round-trip测试
 */

#include "gpt2_model.h"
#include "safetensors_loader.h"
#include "lora_injector.h"
#include "lora_saver.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>

using namespace ops;

int main() {
    try {
        std::cout << "========== LoRA Round-Trip 测试 ==========" << std::endl;
        
        // 1. 加载base模型
        std::cout << "\n[1/5] Loading base model..." << std::endl;
        GPT2Config cfg;
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2/model.safetensors");
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& kv : key_map) {
            try {
                auto info = reader.get_tensor_info(kv.second);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(kv.second, false);
                    model.assign_weight(kv.first, tensor);
                }
            } catch (...) {}
        }
        
        // 2. 注入LoRA
        std::cout << "\n[2/5] Injecting LoRA..." << std::endl;
        model.init_lora_modules();
        
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.0f;
        spec.split_qkv = true;
        
        LoraInjector injector;
        injector.inject(model, spec);
        
        // 3. 准备测试输入
        std::cout << "\n[3/5] Preparing test input..." << std::endl;
        std::vector<int32_t> ids = {1, 2, 3, 4, 5};
        std::vector<float> attn(5, 1.0f);
        
        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 5}, ids.data(), kInt32, kCPU);
        auto attention = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 5}, attn.data(), kFloat32, kCPU);
        
        // 前向1（动态）
        auto logits1 = model.forward(input_ids, attention);
        
        // 4. Save
        std::cout << "\n[4/5] Saving LoRA..." << std::endl;
        std::string lora_path = "/tmp/test_lora_roundtrip.safetensors";
        LoraSaver::save_safetensors(lora_path, model, spec);
        
        // 5. 清空LoRA（重新init）
        std::cout << "\n[5/5] Reloading and testing..." << std::endl;
        GPT2Model model2(cfg);
        model2.tie_weights();
        for (const auto& kv : key_map) {
            try {
                auto info = reader.get_tensor_info(kv.second);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(kv.second, false);
                    model2.assign_weight(kv.first, tensor);
                }
            } catch (...) {}
        }
        
        // 只init，不inject（avoid重复创建LoRA参数）
        model2.init_lora_modules();
        
        // Load & Attach
        std::cout << "  Loading LoRA state..." << std::endl;
        auto loaded_state = LoraSaver::load_safetensors(lora_path);
        std::cout << "  Loaded tensors: " << loaded_state.tensors.size() << std::endl;
        std::cout << "  Attaching to model..." << std::endl;
        LoraSaver::attach_from_state(model2, loaded_state);
        std::cout << "  Attach completed" << std::endl;
        
        LoraInjector injector2;
        
        // 前向2（加载后动态）
        auto logits2 = model2.forward(input_ids, attention);
        
        // 前向3（merge后）
        injector2.merge_all(model2);
        auto logits3 = model2.forward(input_ids, attention);
        
        // 比较
        const float* d1 = logits1->data<float>();
        const float* d2 = logits2->data<float>();
        const float* d3 = logits3->data<float>();
        int64_t n = logits1->numel();
        
        float max_diff_12 = 0.0f;
        float max_diff_13 = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            max_diff_12 = std::max(max_diff_12, std::abs(d1[i] - d2[i]));
            max_diff_13 = std::max(max_diff_13, std::abs(d1[i] - d3[i]));
        }
        
        std::cout << "\n========== Results ==========" << std::endl;
        printf("  Save→Load动态前向差异: %.2e\n", max_diff_12);
        printf("  Save→Load merge前向差异: %.2e\n", max_diff_13);
        
        bool pass = true;
        if (max_diff_12 > 1e-5f) {
            std::cout << "  ❌ FAIL: 动态前向差异过大" << std::endl;
            pass = false;
        }
        if (max_diff_13 > 1e-5f) {
            std::cout << "  ❌ FAIL: merge前向差异过大" << std::endl;
            pass = false;
        }
        
        if (pass) {
            std::cout << "\n🎉 LoRA Round-Trip测试通过！" << std::endl;
            std::cout << "  ✅ Save→Load等价性验证" << std::endl;
            std::cout << "  ✅ 动态前向 ≈ merge前向" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ 测试失败" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

