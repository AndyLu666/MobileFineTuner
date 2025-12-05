/**
 * @file test_lora_correctness.cpp
 * @brief LoRA 正确性测试（零影响 + merge/unmerge 幂等性）
 * 
 * 验收标准：
 * 1. 零影响：注入后 A~N(0,1/r), B=0 → 输出与未注入逐元素相同
 * 2. merge 幂等：merge() 两次 → 权重不变；unmerge() 回到 W0
 * 3. unmerge 幂等：unmerge() 两次 → 权重不变
 */

#include "gpt2_model.h"
#include "lora_injector.h"
#include "safetensors_loader.h"
#include "../core/tokenizer_bpe.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace ops;

// 计算两个 Tensor 的最大绝对误差
float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("Shape mismatch in max_abs_diff");
    }
    
    int64_t n = a->numel();
    const float* a_data = a->data<float>();
    const float* b_data = b->data<float>();
    
    float max_err = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float err = std::abs(a_data[i] - b_data[i]);
        if (err > max_err) max_err = err;
    }
    
    return max_err;
}

// 克隆权重（深拷贝）
TensorPtr clone_tensor(const TensorPtr& src) {
    auto dst = std::make_shared<Tensor>(src->shape(), src->dtype(), src->device());
    std::memcpy(dst->data_ptr(), src->data_ptr(), src->numel() * sizeof(float));
    return dst;
}

int main() {
    try {
        std::cout << "========== LoRA Correctness Tests ==========" << std::endl;
        
        // 准备模型（加载权重）
        std::cout << "\n[Setup] Loading GPT-2 model..." << std::endl;
        GPT2Config cfg;
        cfg.n_layer = 2;  // 简化测试，只用 2 层
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2/model.safetensors");
        reader.parse_header();
        
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            auto info = reader.get_tensor_info(hf_key);
            if (!info.dtype.empty()) {
                // HuggingFace GPT-2 使用 Conv1D，权重已經是 [in, out] 格式，不需要轉置！
                bool transpose = false;
                auto tensor = reader.load_tensor(hf_key, transpose);
                model.assign_weight(internal_key, tensor);
            }
        }
        
        // 准备固定输入
        auto tokenizer_cfg = BPEConfig::from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tokenizer_cfg);
        tokenizer.load();
        
        auto input_ids_vec = tokenizer.encode("Test LoRA.", false, 0, false);
        
        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, static_cast<int64_t>(input_ids_vec.size())},
            input_ids_vec.data(), kInt32, kCPU);
        
        // 创建 attention_mask（全 1）
        auto attn_mask = std::make_shared<Tensor>(
            std::vector<int64_t>{1, static_cast<int64_t>(input_ids_vec.size())},
            kFloat32, kCPU);
        float* mask_data = attn_mask->data<float>();
        for (size_t i = 0; i < input_ids_vec.size(); ++i) {
            mask_data[i] = 1.0f;
        }
        
        // ============================================================
        // Test 1: LoRA 零影响（B=0 初始化 → 输出不变）
        // ============================================================
        std::cout << "\n[Test 1/3] LoRA Zero Impact (B=0 init)" << std::endl;
        
        // 基座前向
        auto logits_base = model.forward(input_ids, attn_mask);
        
        // 注入 LoRA（split_qkv=true, rank=8）
        LoraSpec spec = LoraSpec::default_config();
        spec.rank = 8;
        spec.split_qkv = true;
        
        LoraInjector injector;
        injector.inject(model, spec);
        
        // LoRA 注入后前向（B=0，应输出相同）
        auto logits_lora = model.forward(input_ids, attn_mask);
        
        float err_zero_impact = max_abs_diff(logits_base, logits_lora);
        std::cout << "  Max abs diff (base vs LoRA B=0): " << err_zero_impact << std::endl;
        
        bool pass_zero_impact = (err_zero_impact < 1e-6f);
        if (pass_zero_impact) {
            std::cout << "  ✅ PASS: LoRA zero impact" << std::endl;
        } else {
            std::cout << "  ❌ FAIL: LoRA zero impact (expected < 1e-6)" << std::endl;
        }
        
        // ============================================================
        // Test 2: merge 幂等性
        // ============================================================
        std::cout << "\n[Test 2/3] Merge Idempotency" << std::endl;
        
        // 手动设置 A/B 为非零（便于验证 merge）
        auto lora_params = injector.collect_lora_parameters();
        for (auto& p : lora_params) {
            float* data = p->data<float>();
            for (int64_t i = 0; i < p->numel(); ++i) {
                data[i] = 0.01f * (i % 10 - 5);  // 小随机值
            }
        }
        
        // 保存原始权重 W0（第 0 层 attn.qkv.weight）
        auto W0 = clone_tensor(*model.attn_qkv_params(0).first);
        
        // 第一次 merge
        injector.merge();
        auto W1 = clone_tensor(*model.attn_qkv_params(0).first);
        
        // 第二次 merge（应该无变化）
        injector.merge();
        auto W2 = clone_tensor(*model.attn_qkv_params(0).first);
        
        float err_merge_idempotent = max_abs_diff(W1, W2);
        std::cout << "  Max abs diff (merge 1 vs merge 2): " << err_merge_idempotent << std::endl;
        
        bool pass_merge_idempotent = (err_merge_idempotent < 1e-8f);
        if (pass_merge_idempotent) {
            std::cout << "  ✅ PASS: merge idempotent" << std::endl;
        } else {
            std::cout << "  ❌ FAIL: merge not idempotent" << std::endl;
        }
        
        // unmerge 回到 W0
        injector.unmerge();
        auto W_back = clone_tensor(*model.attn_qkv_params(0).first);
        
        float err_unmerge_restore = max_abs_diff(W0, W_back);
        std::cout << "  Max abs diff (W0 vs unmerge): " << err_unmerge_restore << std::endl;
        
        bool pass_unmerge_restore = (err_unmerge_restore < 1e-6f);
        if (pass_unmerge_restore) {
            std::cout << "  ✅ PASS: unmerge restores W0" << std::endl;
        } else {
            std::cout << "  ❌ FAIL: unmerge does not restore W0" << std::endl;
        }
        
        // ============================================================
        // Test 3: unmerge 幂等性
        // ============================================================
        std::cout << "\n[Test 3/3] Unmerge Idempotency" << std::endl;
        
        // 再次 unmerge（应该无变化）
        injector.unmerge();
        auto W_back2 = clone_tensor(*model.attn_qkv_params(0).first);
        
        float err_unmerge_idempotent = max_abs_diff(W_back, W_back2);
        std::cout << "  Max abs diff (unmerge 1 vs unmerge 2): " << err_unmerge_idempotent << std::endl;
        
        bool pass_unmerge_idempotent = (err_unmerge_idempotent < 1e-8f);
        if (pass_unmerge_idempotent) {
            std::cout << "  ✅ PASS: unmerge idempotent" << std::endl;
        } else {
            std::cout << "  ❌ FAIL: unmerge not idempotent" << std::endl;
        }
        
        // ============================================================
        // 总结
        // ============================================================
        std::cout << "\n========== Summary ==========" << std::endl;
        
        bool all_pass = pass_zero_impact && pass_merge_idempotent 
                       && pass_unmerge_restore && pass_unmerge_idempotent;
        
        if (all_pass) {
            std::cout << "🎉 All LoRA correctness tests passed!" << std::endl;
            return 0;
        } else {
            std::cout << "⚠️  Some LoRA correctness tests failed." << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

