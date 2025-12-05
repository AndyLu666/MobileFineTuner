/**
 * @file test_safetensors_loader.cpp
 * @brief SafeTensors 加载器测试
 * 
 * 用法：
 *   g++ -std=c++17 test_safetensors_loader.cpp safetensors_loader.cpp \
 *       ../core/tensor.cpp ../core/ops.cpp ../core/utils.cpp ../core/logger.cpp \
 *       ../core/memory_manager.cpp ../core/step_arena.cpp \
 *       -I .. -o test_safetensors
 *   ./test_safetensors /path/to/model.safetensors
 */

#include "safetensors_loader.h"
#include <iostream>
#include <iomanip>

using namespace ops;

void test_parse_header(const std::string& model_path) {
    std::cout << "\n[Test] 解析 safetensors header..." << std::endl;
    
    SafeTensorsReader reader(model_path);
    reader.parse_header();
    
    auto names = reader.get_tensor_names();
    std::cout << "  Total tensors: " << names.size() << std::endl;
    
    // 打印前10个
    std::cout << "  Sample tensor names:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), names.size()); ++i) {
        auto info = reader.get_tensor_info(names[i]);
        std::cout << "    [" << i << "] " << names[i] 
                  << " dtype=" << info.dtype 
                  << " shape=[";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j];
            if (j < info.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "  [PASS] Header 解析成功" << std::endl;
}

void test_load_single_tensor(const std::string& model_path) {
    std::cout << "\n[Test] 加载单个张量（wte.weight）..." << std::endl;
    
    SafeTensorsReader reader(model_path);
    reader.parse_header();
    
    try {
        auto wte = reader.load_tensor("wte.weight", false);
        
        std::cout << "  Shape: [";
        for (size_t i = 0; i < wte->shape().size(); ++i) {
            std::cout << wte->shape()[i];
            if (i < wte->shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 打印统计
        const float* data = wte->data<float>();
        double sum = 0.0, sum_sq = 0.0;
        int64_t numel = wte->numel();
        
        for (int64_t i = 0; i < numel; ++i) {
            sum += data[i];
            sum_sq += data[i] * data[i];
        }
        
        double mean = sum / numel;
        double std = std::sqrt(sum_sq / numel - mean * mean);
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
        std::cout << "  Std:  " << std::fixed << std::setprecision(6) << std << std::endl;
        std::cout << "  [PASS] 加载成功，统计正常" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] " << e.what() << std::endl;
    }
}

void test_key_mapping() {
    std::cout << "\n[Test] GPT-2 键名映射..." << std::endl;
    
    auto mapping = GPT2KeyMapper::generate_gpt2_mapping(12);
    std::cout << "  Total mappings: " << mapping.size() << std::endl;
    
    // 检查关键映射
    std::vector<std::string> critical_keys = {
        "wte.weight",
        "blocks.0.ln_1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.11.mlp.fc_out.bias",
        "ln_f.weight"
    };
    
    for (const auto& key : critical_keys) {
        auto it = mapping.find(key);
        if (it != mapping.end()) {
            std::cout << "  ✓ " << key << " -> " << it->second << std::endl;
        } else {
            std::cout << "  ✗ " << key << " NOT FOUND" << std::endl;
        }
    }
    
    std::cout << "  [PASS] 映射生成正确" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.safetensors>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " /Users/tony/Documents/重新开始/gpt2_lora_finetune/pretrained/gpt2/model.safetensors" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    try {
        std::cout << "============================================" << std::endl;
        std::cout << "SafeTensors Loader 测试" << std::endl;
        std::cout << "============================================" << std::endl;
        
        test_key_mapping();
        test_parse_header(model_path);
        test_load_single_tensor(model_path);
        
        std::cout << "\n============================================" << std::endl;
        std::cout << "所有测试完成！" << std::endl;
        std::cout << "============================================\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

