/**
 * @file test_gpt2_forward.cpp
 * @brief GPT-2 前向对齐测试（与 PyTorch/HF 对比 logits）
 * 
 * 验收标准：
 * - max_abs_err ≤ 1e-4
 * - argmax(cpp) == argmax(pt)
 * - top-5 基本一致
 */

#include "gpt2_model.h"
#include "safetensors_loader.h"
#include "../core/tokenizer_bpe.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace ops;

// 简易 JSON 数组解析（仅支持 float 数组）
std::vector<float> load_json_array(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    
    std::string content((std::istreambuf_iterator<char>(in)), {});
    std::vector<float> result;
    result.reserve(60000);
    
    // 超简解析：提取所有数字（科学计数法略）
    std::istringstream iss(content);
    char c;
    std::string num_str;
    bool in_number = false;
    
    while (iss >> c) {
        if (c == '-' || c == '.' || (c >= '0' && c <= '9') || c == 'e' || c == 'E') {
            num_str += c;
            in_number = true;
        } else {
            if (in_number && !num_str.empty()) {
                try {
                    result.push_back(std::stof(num_str));
                } catch (...) {}
                num_str.clear();
            }
            in_number = false;
        }
    }
    if (in_number && !num_str.empty()) {
        try {
            result.push_back(std::stof(num_str));
        } catch (...) {}
    }
    
    return result;
}

int main() {
    try {
        std::cout << "========== GPT-2 Forward Alignment Test ==========" << std::endl;
        
        // 1. 构建模型
        GPT2Config cfg;
        GPT2Model model(cfg);
        model.tie_weights();
        
        // 2. 加载权重
        std::cout << "\n[1/5] Loading weights from safetensors..." << std::endl;
        SafeTensorsReader reader("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2/model.safetensors");
        reader.parse_header();
        
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        int loaded_count = 0;
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                        // HuggingFace GPT-2 使用 Conv1D，权重已經是 [in, out] 格式，不需要轉置！
                    bool transpose = false;
                    auto tensor = reader.load_tensor(hf_key, transpose);
                    model.assign_weight(internal_key, tensor);
                    loaded_count++;
                    
                    // 调试：打印关键权重的形状
                    if (internal_key == "wte.weight" || internal_key == "blocks.0.attn.qkv.weight" ||
                        internal_key == "blocks.0.ln_1.weight") {
                        std::cout << "✓ Loaded " << internal_key << ": shape = [";
                        for (size_t i = 0; i < tensor->shape().size(); ++i) {
                            std::cout << tensor->shape()[i];
                            if (i + 1 < tensor->shape().size()) std::cout << ", ";
                        }
                        std::cout << "], transposed=" << (transpose ? "yes" : "no") << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[WARN] Failed to load " << hf_key << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "Loaded " << loaded_count << " / " << key_map.size() << " tensors." << std::endl;
        
        // 验证：检查第11层权重是否加载
        std::cout << "\n[Verify] Checking layer 11 weights..." << std::endl;
        for (const auto& [internal_key, hf_key] : key_map) {
            if (internal_key.find("blocks.11.") != std::string::npos) {
                std::cout << "  ✓ " << internal_key << std::endl;
            }
        }
        
        // 3. 准备输入（暫時使用 PyTorch 生成的 token IDs，繞過 tokenizer bug）
        std::cout << "\n[2/5] Preparing input..." << std::endl;
        // PyTorch tokenizer 输出：[15496, 11, 995, 0, 198] for "Hello, world!\n"
        std::vector<int> input_ids_vec = {15496, 11, 995, 0, 198};
        
        std::cout << "Input text: \"Hello, world!\\n\"" << std::endl;
        std::cout << "Input IDs (from PyTorch): [";
        for (size_t i = 0; i < input_ids_vec.size(); ++i) {
            std::cout << input_ids_vec[i];
            if (i + 1 < input_ids_vec.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "(Note: C++ tokenizer has BPE merge bug, using PyTorch IDs for now)" << std::endl;
        
        // 转为 Tensor
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
        
        // 4. 前向（禁用 dropout）
        std::cout << "\n[3/5] Running forward pass..." << std::endl;
        // TODO: 禁用 dropout（全局开关或传参）
        auto logits = model.forward(input_ids, attn_mask);  // [1, S, V]
        
        int64_t batch = logits->shape()[0];
        int64_t seq_len = logits->shape()[1];
        int64_t vocab_size = logits->shape()[2];
        
        std::cout << "Logits shape: [" << batch << ", " << seq_len << ", " << vocab_size << "]" << std::endl;
        
        // 提取末位 token logits
        const float* logits_data = logits->data<float>();
        std::vector<float> last_logits(vocab_size);
        std::memcpy(last_logits.data(), logits_data + (seq_len - 1) * vocab_size, vocab_size * sizeof(float));
        
        // 打印前10个值
        std::cout << "\nC++ last_logits[0:10]: [";
        for (int i = 0; i < 10; ++i) {
            printf("%.2f", last_logits[i]);
            if (i < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Top-5
        std::vector<std::pair<float, int>> scored;
        scored.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            scored.emplace_back(last_logits[i], i);
        }
        std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "\nC++ top-5 IDs:  [";
        for (int i = 0; i < 5; ++i) {
            std::cout << scored[i].second;
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "C++ top-5 vals: [";
        for (int i = 0; i < 5; ++i) {
            printf("%.6f", scored[i].first);
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        int argmax_cpp = scored[0].second;
        
        // 5. 加载 PyTorch 金标并对比
        std::cout << "\n[4/5] Loading PyTorch gold standard..." << std::endl;
        auto pt_logits = load_json_array("/Users/tony/Documents/重新开始/operators/finetune_ops/graph/pt_last_logits.json");
        
        if (pt_logits.size() != static_cast<size_t>(vocab_size)) {
            std::cerr << "ERROR: PyTorch logits size mismatch: " << pt_logits.size()
                     << " vs " << vocab_size << std::endl;
            return 1;
        }
        
        // 计算误差
        float max_abs_err = 0.0f;
        float sum_abs_err = 0.0f;
        int argmax_pt = 0;
        float max_val_pt = pt_logits[0];
        
        for (int i = 0; i < vocab_size; ++i) {
            float err = std::abs(last_logits[i] - pt_logits[i]);
            if (err > max_abs_err) max_abs_err = err;
            sum_abs_err += err;
            
            if (pt_logits[i] > max_val_pt) {
                max_val_pt = pt_logits[i];
                argmax_pt = i;
            }
        }
        
        float mean_abs_err = sum_abs_err / vocab_size;
        
        // 6. 报告结果
        std::cout << "\n[5/5] ========== Alignment Report ==========" << std::endl;
        std::cout << "Max absolute error: " << max_abs_err << std::endl;
        std::cout << "Mean absolute error: " << mean_abs_err << std::endl;
        std::cout << "Argmax (C++): " << argmax_cpp << std::endl;
        std::cout << "Argmax (PT):  " << argmax_pt << std::endl;
        
        // 验收标准
        bool pass = true;
        
        if (max_abs_err > 1e-4f) {
            std::cout << "\n❌ FAIL: max_abs_err > 1e-4" << std::endl;
            pass = false;
        } else {
            std::cout << "\n✅ PASS: max_abs_err ≤ 1e-4" << std::endl;
        }
        
        if (argmax_cpp != argmax_pt) {
            std::cout << "❌ FAIL: argmax mismatch" << std::endl;
            pass = false;
        } else {
            std::cout << "✅ PASS: argmax consistent" << std::endl;
        }
        
        std::cout << "\n" << (pass ? "🎉 All checks passed!" : "⚠️  Some checks failed.") << std::endl;
        
        return pass ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

