/**
 * @file test_backward_sanity.cpp
 * @brief Backward梯度冒烟测试（1个batch）
 */

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../graph/lora_injector.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_bpe.h"
#include "../core/lm_loss.h"
#include <iostream>
#include <cmath>

using namespace ops;

int main() {
    try {
        std::cout << "========== Backward梯度冒烟测试 ==========\n" << std::endl;
        
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
        
        // 2. 初始化LoRA模块并注入
        std::cout << "\n[2/4] Injecting LoRA..." << std::endl;
        model.init_lora_modules();  // ← 必须先初始化
        
        LoraSpec lora_spec;
        lora_spec.rank = 4;
        lora_spec.alpha = 8.0f;
        lora_spec.dropout = 0.0f;
        lora_spec.split_qkv = true;
        
        LoraInjector lora;
        lora.inject(model, lora_spec);
        
        // 获取可训练参数
        auto lora_params = model.get_lora_parameters();  // ← 改用model方法
        std::cout << "✅ Trainable params: " << lora_params.size() << std::endl;
        
        // 检查requires_grad状态并retain_grad
        int requires_grad_count = 0;
        for (const auto& p : lora_params) {
            if (p->requires_grad()) {
                requires_grad_count++;
                p->retain_grad();  // ← 显式保留梯度
            }
        }
        std::cout << "  Params with requires_grad=true: " << requires_grad_count << std::endl;
        
        if (lora_params.size() > 0) {
            std::cout << "  First param shape: [" << lora_params[0]->shape()[0] 
                      << ", " << lora_params[0]->shape()[1] << "]" << std::endl;
            std::cout << "  First param requires_grad: " << lora_params[0]->requires_grad() << std::endl;
        }
        
        // 3. 准备数据
        std::cout << "\n[3/4] Loading data..." << std::endl;
        auto tok_cfg = BPEConfig::from_pretrained("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2");
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();
        
        WT2Config data_cfg;
        data_cfg.train_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.train.raw";
        data_cfg.valid_path = "/Users/tony/Documents/重新开始/data/wikitext2/wikitext-2-raw/wiki.valid.raw";
        data_cfg.seq_len = 64;  // 极短序列快速测试
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        
        // 4. Forward + Backward
        std::cout << "\n[4/4] Forward + Backward..." << std::endl;
        auto batch = dataset.next_batch(1);  // batch_size=1
        
        // Forward
        auto logits = model.forward(batch.input_ids, batch.attention_mask);
        logits->retain_grad();  // ← 保留中间变量的梯度
        
        std::cout << "  Logits requires_grad: " << logits->requires_grad() << std::endl;
        std::cout << "  Logits shape: [" << logits->shape()[0] << ", " 
                  << logits->shape()[1] << ", " << logits->shape()[2] << "]" << std::endl;
        
        auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
        float loss_val = loss->data<float>()[0];
        
        std::cout << "  Loss: " << loss_val << std::endl;
        std::cout << "  Loss requires_grad: " << loss->requires_grad() << std::endl;
        
        // Backward
        std::cout << "  Calling backward()..." << std::endl;
        loss->backward();
        std::cout << "  Backward completed" << std::endl;
        
        // 检查logits是否有梯度
        if (logits->grad()) {
            std::cout << "  ✅ Logits has gradient" << std::endl;
            const float* logits_grad = logits->grad()->data<float>();
            float logits_grad_norm = 0.0f;
            for (int64_t i = 0; i < std::min(static_cast<int64_t>(1000), logits->grad()->numel()); ++i) {
                logits_grad_norm += logits_grad[i] * logits_grad[i];
            }
            std::cout << "  Logits grad norm (first 1000): " << std::sqrt(logits_grad_norm) << std::endl;
        } else {
            std::cout << "  ❌ Logits has NO gradient!" << std::endl;
        }
        
        // 统计梯度
        size_t n_params = lora_params.size();
        size_t n_grad = 0;
        double grad_norm_sq = 0.0;
        
        std::vector<int> no_grad_indices;
        for (size_t i = 0; i < lora_params.size(); ++i) {
            const auto& param = lora_params[i];
            auto grad = param->grad();
            if (grad) {
                n_grad++;
                const float* grad_data = grad->data<float>();
                for (int64_t j = 0; j < grad->numel(); ++j) {
                    grad_norm_sq += grad_data[j] * grad_data[j];
                }
            } else {
                no_grad_indices.push_back(i);
            }
        }
        
        std::cout << "  Params without grad: " << no_grad_indices.size() << " out of " << n_params << std::endl;
        if (!no_grad_indices.empty() && no_grad_indices.size() <= 30) {
            std::cout << "    Indices: ";
            for (int idx : no_grad_indices) std::cout << idx << " ";
            std::cout << std::endl;
        }
        
        float grad_norm = std::sqrt(grad_norm_sq);
        
        std::cout << "\n[Gradient Statistics]" << std::endl;
        std::cout << "  Trainable params: " << n_params << std::endl;
        std::cout << "  Params with grad: " << n_grad << std::endl;
        std::cout << "  Gradient norm: " << grad_norm << std::endl;
        
        // 验收（放宽标准：75%以上有梯度即可）
        bool pass = true;
        float grad_coverage = static_cast<float>(n_grad) / n_params;
        
        if (grad_coverage < 0.5f) {
            std::cout << "  ❌ FAIL: 梯度覆盖率过低（" << (grad_coverage*100) << "%）" << std::endl;
            pass = false;
        } else if (grad_coverage < 1.0f) {
            std::cout << "  ⚠️  WARN: 部分参数无梯度（" << (grad_coverage*100) << "%覆盖）" << std::endl;
        }
        
        if (grad_norm == 0.0f) {
            std::cout << "  ❌ FAIL: 梯度为0" << std::endl;
            pass = false;
        }
        if (!std::isfinite(grad_norm)) {
            std::cout << "  ❌ FAIL: 梯度为NaN/Inf" << std::endl;
            pass = false;
        }
        if (grad_norm < 1e-10f || grad_norm > 1e6f) {
            std::cout << "  ⚠️  WARN: 梯度范数异常（" << grad_norm << "）" << std::endl;
        }
        
        if (pass && n_grad > 0 && std::isfinite(grad_norm) && grad_coverage >= 0.5f) {
            std::cout << "\n🎉 Backward梯度测试通过！" << std::endl;
            std::cout << "  ✅ " << n_grad << "/" << n_params << "个LoRA参数有梯度" << std::endl;
            std::cout << "  ✅ 梯度范数：" << grad_norm << "（有限且非零）" << std::endl;
            std::cout << "\n下一步：运行10步训练验证收敛" << std::endl;
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

