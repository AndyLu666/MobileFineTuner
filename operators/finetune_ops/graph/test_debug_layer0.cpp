/**
 * @file test_debug_layer0.cpp
 * @brief 逐层调试第一层，对比 PyTorch 激活值
 */

#include "gpt2_model.h"
#include "safetensors_loader.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace ops;

// 计算张量统计
struct TensorStats {
    float mean;
    float std;
    float max_abs;
    
    TensorStats(const TensorPtr& t) {
        const float* data = t->data<float>();
        int64_t n = t->numel();
        
        // Mean
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            sum += data[i];
        }
        mean = sum / n;
        
        // Std
        double var = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double diff = data[i] - mean;
            var += diff * diff;
        }
        std = std::sqrt(var / n);
        
        // Max abs
        max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            max_abs = std::max(max_abs, std::abs(data[i]));
        }
    }
};

void print_stats(const std::string& name, const TensorPtr& t) {
    TensorStats stats(t);
    printf("%-20s | shape=[", name.c_str());
    for (size_t i = 0; i < t->shape().size(); ++i) {
        printf("%ld", t->shape()[i]);
        if (i + 1 < t->shape().size()) printf(", ");
    }
    printf("] | mean=%10.6f | std=%10.6f | max=%10.6f\n", stats.mean, stats.std, stats.max_abs);
}

int main() {
    try {
        std::cout << "========== GPT-2 Layer 0 Debug ==========\n" << std::endl;
        
        // 1. 加载模型
        GPT2Config cfg;
        cfg.n_layer = 1;  // 只加载第一层
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader("/Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2/model.safetensors");
        reader.parse_header();
        
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        // 加载权重
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    bool transpose = false;
                    auto tensor = reader.load_tensor(hf_key, transpose);
                    model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        
        std::cout << "✅ Loaded weights\n" << std::endl;
        
        // 2. 准备输入
        std::vector<int> input_ids_vec = {15496, 11, 995, 0, 198};
        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, 5}, input_ids_vec.data(), kInt32, kCPU);
        
        // 3. 手动运行 embedding
        std::cout << "========== Manual Forward (Layer 0) ==========\n" << std::endl;
        
        // Embedding (使用 model 的内部函数不可行，直接调用 forward 看统计)
        auto logits = model.forward(input_ids, nullptr);
        
        print_stats("logits", logits);
        
        std::cout << "\n✅ Compare these stats with Python output" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}

