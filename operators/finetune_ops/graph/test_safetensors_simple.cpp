/**
 * @file test_safetensors_simple.cpp
 * @brief SafeTensors 加载器最小测试（不依赖完整 ops.cpp）
 */

#include "safetensors_loader.h"
#include <iostream>
#include <iomanip>

using namespace ops;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.safetensors>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    try {
        std::cout << "============================================" << std::endl;
        std::cout << "SafeTensors Header 解析测试" << std::endl;
        std::cout << "============================================" << std::endl;
        
        SafeTensorsReader reader(model_path);
        reader.parse_header();
        
        auto names = reader.get_tensor_names();
        std::cout << "\nTotal tensors: " << names.size() << std::endl;
        
        // 打印所有张量名与形状
        std::cout << "\nAll tensors:\n" << std::endl;
        for (const auto& name : names) {
            auto info = reader.get_tensor_info(name);
            std::cout << "  " << std::setw(40) << std::left << name 
                      << " dtype=" << info.dtype 
                      << " shape=[";
            for (size_t j = 0; j < info.shape.size(); ++j) {
                std::cout << info.shape[j];
                if (j < info.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "\n[PASS] Header 解析完成！\n" << std::endl;
        
        // 测试键名映射
        std::cout << "\n[Test] GPT-2 键名映射生成..." << std::endl;
        auto mapping = GPT2KeyMapper::generate_gpt2_mapping(12);
        std::cout << "Total mappings: " << mapping.size() << std::endl;
        std::cout << "[PASS] 映射生成完成\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

