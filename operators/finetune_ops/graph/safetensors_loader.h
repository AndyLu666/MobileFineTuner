/**
 * @file safetensors_loader.h
 * @brief SafeTensors 格式权重加载器（纯 C++ 实现）
 * 
 * 支持：
 * - 解析 safetensors 格式（8B header_len + JSON header + raw data）
 * - 加载 FP32/FP16 张量（FP16 自动升格为 FP32）
 * - 键名映射（HF → 内部命名）
 * - 自动转置 Linear 权重（[out,in] → [in,out]）
 */

#pragma once

#include "../core/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>

namespace ops {

/**
 * @brief SafeTensors 文件中的张量元信息
 */
struct SafeTensorInfo {
    std::string dtype;              // "F32", "F16", "I32", "I64" 等
    std::vector<int64_t> shape;     // 张量形状
    std::vector<size_t> data_offsets;  // [start, end) in file
};

/**
 * @brief SafeTensors 加载选项
 */
struct SafeTensorsLoadOptions {
    bool transpose_linear = true;   // 是否自动转置线性层权重 [out,in]→[in,out]
    bool auto_promote_fp16 = true;  // FP16 自动升格为 FP32
    bool verbose = true;            // 是否打印加载日志
    bool strict_shape_check = true; // 严格校验形状
};

/**
 * @brief SafeTensors 文件读取器
 */
class SafeTensorsReader {
public:
    explicit SafeTensorsReader(const std::string& filepath);
    ~SafeTensorsReader();
    
    /**
     * @brief 解析文件头（8B header_len + JSON header）
     */
    void parse_header();
    
    /**
     * @brief 获取所有张量的键名列表
     */
    std::vector<std::string> get_tensor_names() const;
    
    /**
     * @brief 获取指定张量的元信息
     */
    SafeTensorInfo get_tensor_info(const std::string& name) const;
    
    /**
     * @brief 加载指定张量到内存
     * @param name 张量名
     * @param transpose 是否转置（针对 2D 张量）
     * @return 张量指针
     */
    TensorPtr load_tensor(const std::string& name, bool transpose = false);
    
    /**
     * @brief 批量加载张量（带键名映射）
     * @param key_mapping {"internal_key": "hf_key"}
     * @param options 加载选项
     * @return {"internal_key": tensor}
     */
    std::unordered_map<std::string, TensorPtr> 
    load_tensors_mapped(const std::unordered_map<std::string, std::string>& key_mapping,
                        const SafeTensorsLoadOptions& options = SafeTensorsLoadOptions());

private:
    std::string filepath_;
    std::ifstream file_;
    size_t header_len_;
    size_t data_offset_;  // header 之后的数据起始位置
    std::unordered_map<std::string, SafeTensorInfo> tensor_map_;
    
    void parse_tensor_metadata(const std::string& json_str);
    TensorPtr read_tensor_data(const SafeTensorInfo& info, bool transpose);
};

/**
 * @brief GPT-2 HuggingFace → 内部键名映射生成器
 */
class GPT2KeyMapper {
public:
    /**
     * @brief 生成完整映射表（GPT-2 12层，n_embd=768）
     * @param num_layers 层数（默认12）
     * @return {"internal_key": "hf_key"}
     */
    static std::unordered_map<std::string, std::string> 
    generate_gpt2_mapping(int num_layers = 12);
    
    /**
     * @brief 打印映射表（调试用）
     */
    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

/**
 * @brief Gemma3 HuggingFace → 内部键名映射生成器
 */
class GemmaKeyMapper {
public:
    static std::unordered_map<std::string, std::string>
    generate_gemma_mapping(int num_layers = 18);

    static void print_mapping(const std::unordered_map<std::string, std::string>& mapping);
};

}  // namespace ops
